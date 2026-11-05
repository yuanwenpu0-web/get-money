"""Compare Supertrend exits: R-multiple targets vs fixed-bar time take-profit.

Signals are generated once with production parameters, so every variant trades the
same entries. Only exit rules change: ATR stop and opposite-signal exit stay, while
the 1R/2R/3R scaled targets are replaced by a time exit after N closed bars.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.backtest.execution import simulate  # noqa: E402
from app.backtest.significance import bootstrap_expectancy_ci  # noqa: E402
from app.backtest.trades import aggregate_positions  # noqa: E402
from app.config import DEFAULT_STATE  # noqa: E402
from app.strategy.rules import resolve_params  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "realized", ROOT / "backend/scripts/supertrend_reversal_ml_realized.py"
)
realized = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(realized)

CODE, INTERVAL, CASH = "supertrend_current", "15m", 10_000.0
FAR = 1_000.0


def variants(base: dict) -> dict:
    def pure(bars: int) -> dict:
        # Unreachable R targets; max_hold_bars closes the position after N bars.
        return {**base, "tp1_r": FAR, "tp2_r": FAR + 1, "target_r": FAR + 2, "max_hold_bars": bars}
    return {
        "baseline_1r2r3r": dict(base),
        "hold_3_bars": pure(3),
        "hold_5_bars": pure(5),
        "hold_8_bars": pure(8),
        "hold_10_bars": pure(10),
        "hold_16_bars": pure(16),
        "hold_5_plus_scaled_tp": {**base, "max_hold_bars": 5},
    }


def run_symbol(task: dict) -> dict:
    parts = realized.load_symbol_data(task["symbol"], task["start_ms"], task["end_ms"], task["base"])
    out = {"symbol": task["symbol"], "windows": {}}
    for window_name, start, end in (
        ("validation", task["validation_start"], task["validation_end"]),
        ("test", task["test_start"], task["test_end"]),
    ):
        per_variant = {}
        for name, params in task["variants"].items():
            factors, equity_parts, trades = [], [], []
            for frame, signals in parts:
                mask = (frame.index >= start) & (frame.index < end)
                pf, ps = frame.loc[mask], signals.loc[mask]
                if len(pf) < 2:
                    continue
                equity, raw = simulate(pf, ps, params, CASH, task["fee"], task["slippage"], INTERVAL)
                factor = float(equity.iloc[-1]) / CASH
                factors.append(factor)
                equity_parts.append(equity * float(np.prod(factors[:-1])) if len(factors) > 1 else equity)
                trades.extend(aggregate_positions(raw))
            if not factors:
                per_variant[name] = None
                continue
            combined = pd.concat(equity_parts)
            per_variant[name] = {
                "return_pct": float(np.prod(factors) - 1.0) * 100,
                "max_drawdown_pct": float((combined / combined.cummax().clip(lower=CASH) - 1).min() * 100),
                "trades": len(trades),
                "returns": [float(t["return_pct"]) for t in trades],
                "sides": ["long" if str(t["side"]).lower() in ("long", "1") else "short" for t in trades],
                "status": [str(t["status"]) for t in trades],
            }
        out["windows"][window_name] = per_variant
    return out


def summarize(results: list[dict], name: str, window: str) -> dict:
    accounts, returns, longs, shorts, status = [], [], [], [], {}
    for item in results:
        value = item["windows"][window].get(name)
        if not value:
            continue
        accounts.append(value["return_pct"])
        returns.extend(value["returns"])
        for ret, side in zip(value["returns"], value["sides"]):
            (longs if side == "long" else shorts).append(ret)
        for s in value["status"]:
            status[s] = status.get(s, 0) + 1
    ci = bootstrap_expectancy_ci([{"return_pct": x} for x in returns], min_trades=30)
    return {
        "symbols": len(accounts),
        "account_mean_pct": round(float(np.mean(accounts)), 4) if accounts else None,
        "account_median_pct": round(float(np.median(accounts)), 4) if accounts else None,
        "positive_accounts": int(sum(x > 0 for x in accounts)),
        "trades": len(returns),
        "pooled": ci,
        "long_mean_pct": round(float(np.mean(longs)), 4) if longs else None,
        "long_ci": bootstrap_expectancy_ci([{"return_pct": x} for x in longs], min_trades=30),
        "short_mean_pct": round(float(np.mean(shorts)), 4) if shorts else None,
        "short_ci": bootstrap_expectancy_ci([{"return_pct": x} for x in shorts], min_trades=30),
        "status_counts": dict(sorted(status.items(), key=lambda x: -x[1])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-bars", type=int, default=15_000)
    ap.add_argument("--processes", type=int, default=max(1, min(10, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--out", default="output/supertrend_hold_exit_20261104.json")
    ap.add_argument("--validation-start", default="2026-08-01T00:00:00Z")
    ap.add_argument("--validation-end", default="2026-09-15T00:00:00Z")
    ap.add_argument("--test-start", default="2026-09-15T00:00:00Z")
    ap.add_argument("--test-end", default="2026-11-03T00:00:00Z")
    args = ap.parse_args()

    fee = float(DEFAULT_STATE["fee_rate"])
    slippage = float(DEFAULT_STATE["slippage"])
    base = resolve_params(CODE, {"fee_rate": fee, "slippage": slippage}, interval=INTERVAL)
    if args.min_bars:
        from sqlalchemy import text
        from app.db import make_engine
        engine = make_engine(read_only=True)
        with engine.connect() as conn:
            symbols = [r[0] for r in conn.execute(text(
                "select symbol from klines where interval='15m' group by symbol having count(*) >= :n order by symbol"
            ), {"n": args.min_bars})]
        engine.dispose()
    else:
        symbols = []
    end_ms = int(pd.Timestamp.now(tz="UTC").floor("15min").timestamp() * 1000)
    start_ms = int(pd.Timestamp("2025-12-01T00:00:00Z").timestamp() * 1000)
    tasks = [{
        "symbol": s, "start_ms": start_ms, "end_ms": end_ms, "base": base,
        "variants": variants(base), "fee": fee, "slippage": slippage,
        "validation_start": pd.Timestamp(args.validation_start),
        "validation_end": pd.Timestamp(args.validation_end),
        "test_start": pd.Timestamp(args.test_start),
        "test_end": pd.Timestamp(args.test_end),
    } for s in symbols]
    print(f"出场规则对照：{len(symbols)} 币 × {len(tasks[0]['variants'])} 变体", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        futures = [pool.submit(run_symbol, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if done % 10 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}", flush=True)
    names = list(tasks[0]["variants"])
    report = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "costs": {"fee": fee, "slippage": slippage},
              "note": "所有变体共用同一批信号；时间止盈在持有N根后的下一根开盘平仓",
              "windows": {w: {n: summarize(results, n, w) for n in names} for w in ("validation", "test")}}
    Path(ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for window in ("validation", "test"):
        print(f"\n== {window} ==")
        for name in names:
            value = report["windows"][window][name]
            ci = value["pooled"]
            print(f"{name:22s} account {value['account_mean_pct']:+7.2f}%  trades {value['trades']:5d}  "
                  f"per-trade {ci.get('mean_return_pct', 0):+.3f}% [{ci.get('ci_low_pct', 0):+.3f}, {ci.get('ci_high_pct', 0):+.3f}] "
                  f"{ci.get('verdict')}  pos {value['positive_accounts']}/{value['symbols']}")
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
