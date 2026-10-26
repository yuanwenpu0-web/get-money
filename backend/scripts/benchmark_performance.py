"""Offline performance checks using synthetic data and temporary in-memory SQLite.

Run from the project root:
    .venv/Scripts/python.exe backend/scripts/benchmark_performance.py

No exchange requests, trading database access, orders, or model writes are needed.
Timings depend on the host; equality assertions, rather than speed, determine
whether the benchmark succeeds.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sqlalchemy import case, create_engine, func, insert, select
from sqlalchemy.orm import Session

from app.db import Base
from app.market.data_service import MarketDataService
from app.models import Kline, Order
from app.strategy.engine import StrategyEngine

START_MS = 1_767_225_600_000
MINUTE_MS = 60_000


def _timings_ms(operation, repeats: int) -> dict:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - started) * 1000)
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def benchmark_market_sync(
    symbol_count: int = 24, latency_ms: float = 30.0, workers: int = 8,
) -> dict:
    """Compare one worker with a bounded pool, including real page upsert costs."""
    symbols = [f"SYNTHETIC{index}" for index in range(symbol_count)]

    class DelayedClient:
        def klines(self, symbol, interval, start_ms, limit):
            time.sleep(latency_ms / 1000)
            return [[START_MS, "100", "101", "99", "100.5", "1",
                     START_MS + MINUTE_MS - 1, "100", 1]]

    results, snapshots = {}, []
    for count in (1, workers):
        engine = create_engine("sqlite:///:memory:")
        try:
            Base.metadata.create_all(engine)
            with Session(engine) as db, patch("app.market.data_service.now_ms", return_value=START_MS + MINUTE_MS):
                started = time.perf_counter()
                result = MarketDataService(db, DelayedClient()).sync_many(symbols, days=1, workers=count)
                elapsed = (time.perf_counter() - started) * 1000
                assert result == {"requested": symbol_count, "inserted": symbol_count, "errors": []}, result
                columns = [column for column in Kline.__table__.columns if column.name != "id"]
                snapshots.append(db.execute(select(*columns).order_by(Kline.symbol, Kline.open_time)).all())
                results[f"workers_{count}"] = {"elapsed_ms": elapsed, "inserted": result["inserted"]}
        finally:
            engine.dispose()
    assert snapshots[0] == snapshots[1], "Concurrent sync changed stored candle values"
    return {
        "symbols": symbol_count,
        "simulated_request_latency_ms": latency_ms,
        "equivalent": True,
        **results,
        "speedup": results["workers_1"]["elapsed_ms"] / results[f"workers_{workers}"]["elapsed_ms"],
    }


def benchmark_candle_tail(minutes: int = 216_000, limit: int = 400, repeats: int = 5) -> dict:
    """Compare full-history aggregation with the bounded tail on identical data."""
    engine = create_engine("sqlite:///:memory:")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as db, patch(
            "app.market.data_service.now_ms", return_value=START_MS + (minutes + 1) * MINUTE_MS,
        ):
            for base in range(0, minutes, 2000):
                db.execute(insert(Kline), [
                    dict(symbol="SYNTHETIC", interval="1m", open_time=START_MS + index * MINUTE_MS,
                         close_time=START_MS + (index + 1) * MINUTE_MS - 1,
                         open=100.0, high=101.0, low=99.0, close=100.5,
                         volume=1.0, quote_volume=100.0, trades=1)
                    for index in range(base, min(base + 2000, minutes))
                ])
            db.commit()
            service = MarketDataService(db)

            def full_history():
                query = service._candle_query("SYNTHETIC", "1h").subquery()
                return db.execute(select(query).order_by(query.c.open_time.desc()).limit(limit)).mappings().all()

            def bounded_tail():
                return service._tail_candle_rows("SYNTHETIC", "1h", limit)

            expected = full_history()
            assert bounded_tail() == expected, "Bounded aggregation changed candle values"
            baseline = _timings_ms(full_history, repeats)
            optimized = _timings_ms(bounded_tail, repeats)
            return {
                "input_minutes": minutes,
                "interval": "1h",
                "requested_candles": limit,
                "returned_candles": len(expected),
                "equivalent": True,
                "full_history": baseline,
                "bounded_tail": optimized,
                "speedup": baseline["median_ms"] / optimized["median_ms"],
            }
    finally:
        engine.dispose()


def benchmark_signal_reuse(repeats: int = 5) -> dict:
    """Five strategies, real candle reads/features, without saving or placing orders."""
    engine = create_engine("sqlite:///:memory:")
    minutes = 24_000
    codes = ("ml_ensemble", "trend_follow", "mean_reversion", "funding_extreme", "mtf_resonance")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as db, patch(
            "app.strategy.engine.now_ms", return_value=START_MS + minutes * MINUTE_MS,
        ), patch("app.market.data_service.now_ms", return_value=START_MS + minutes * MINUTE_MS):
            for base in range(0, minutes, 2000):
                rows = []
                for index in range(base, min(base + 2000, minutes)):
                    price = 100 + index * .001 + float(np.sin(index / 480))
                    rows.append(dict(
                        symbol="SYNTHETIC", interval="1m", open_time=START_MS + index * MINUTE_MS,
                        close_time=START_MS + (index + 1) * MINUTE_MS - 1,
                        open=price, high=price + .1, low=price - .1, close=price + .01,
                        volume=100.0, quote_volume=10_000.0, trades=10,
                    ))
                db.execute(insert(Kline), rows)
            db.commit()

            def evaluate(cached):
                evaluator = StrategyEngine(db, cycle_snapshot=cached)
                return [evaluator.evaluate("SYNTHETIC", "1h", code).to_dict() for code in codes]

            assert evaluate(False) == evaluate(True), "Feature reuse changed strategy signals"
            baseline = _timings_ms(lambda: evaluate(False), repeats)
            optimized = _timings_ms(lambda: evaluate(True), repeats)
            return {"strategies": len(codes), "input_minutes": minutes,
                    "signal_writes": False, "equivalent": True,
                    "per_strategy_features": baseline, "cycle_snapshot": optimized,
                    "speedup": baseline["median_ms"] / optimized["median_ms"]}
    finally:
        engine.dispose()


def benchmark_closed_statistics(repeats: int = 5) -> dict:
    """Compare historical order statistics without loading any live account data."""
    engine = create_engine("sqlite:///:memory:")
    count = 20_000
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as db:
            for base in range(0, count, 2000):
                db.execute(insert(Order), [
                    dict(symbol="SYNTHETIC", interval="1h", side="LONG", qty=1.0,
                         entry_price=100.0, status="CLOSED", realized_pnl=float(index % 7 - 3))
                    for index in range(base, min(base + 2000, count))
                ])
            db.commit()

            def orm_statistics():
                orders = db.query(Order).filter(Order.status == "CLOSED").all()
                return len(orders), sum(o.realized_pnl for o in orders), sum(o.realized_pnl > 0 for o in orders)

            def sql_statistics():
                return tuple(db.query(
                    func.count(Order.id), func.coalesce(func.sum(Order.realized_pnl), 0.0),
                    func.coalesce(func.sum(case((Order.realized_pnl > 0, 1), else_=0)), 0),
                ).filter(Order.status == "CLOSED").one())

            assert orm_statistics() == sql_statistics(), "SQL aggregation changed historical statistics"
            baseline = _timings_ms(orm_statistics, repeats)
            optimized = _timings_ms(sql_statistics, repeats)
            return {"closed_orders": count, "equivalent": True,
                    "orm_loading": baseline, "sql_aggregation": optimized,
                    "speedup": baseline["median_ms"] / optimized["median_ms"]}
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=int, default=216_000)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--symbols", type=int, default=24)
    parser.add_argument("--latency-ms", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional JSON report; otherwise print only")
    args = parser.parse_args()
    if min(args.minutes, args.limit, args.symbols, args.repeats) < 1:
        parser.error("minutes, limit, symbols, and repeats must be positive")
    if not 1 <= args.workers <= 32 or args.latency_ms < 0:
        parser.error("workers must be 1..32 and latency-ms must be nonnegative")
    report = {
        "database": "temporary in-memory SQLite",
        "network": "simulated",
        "market_sync": benchmark_market_sync(args.symbols, args.latency_ms, args.workers),
        "candle_tail": benchmark_candle_tail(args.minutes, args.limit, args.repeats),
        "signal_reuse": benchmark_signal_reuse(args.repeats),
        "closed_statistics": benchmark_closed_statistics(args.repeats),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
