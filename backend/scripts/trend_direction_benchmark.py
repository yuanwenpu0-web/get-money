"""补算 always-long / always-short 基准：用来判断「最优指标」是否只是吃了市场 beta。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

load_dotenv(ROOT / ".env")

from app.db import make_engine  # noqa: E402
from app.models import Kline  # noqa: E402
from app.research.trend_direction import evaluate_symbol  # noqa: E402
from app.research.trend_direction_report import cluster_bootstrap  # noqa: E402


def load(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame | None:
    with Session(make_engine(read_only=True)) as db:
        rows = db.execute(
            select(Kline.open_time, Kline.open, Kline.high, Kline.low, Kline.close,
                   Kline.volume, Kline.quote_volume, Kline.trades)
            .where(Kline.symbol == symbol, Kline.interval == "1h",
                   Kline.open_time >= start_ms, Kline.open_time <= end_ms)
            .order_by(Kline.open_time.asc())
        ).mappings().all()
    if len(rows) < 320:
        return None
    frame = pd.DataFrame([dict(row) for row in rows])
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("open_time"), unit="ms", utc=True), name="time")
    return frame


def main() -> int:
    payload = json.loads((ROOT / "docs/trend-direction-backtest.json").read_text(encoding="utf-8"))
    last_complete = int(pd.Timestamp(payload["data"]["last_complete"]).timestamp() * 1000)
    start_ms = int(payload["config"]["evaluation_start_ms"])
    pools = {pool["code"]: pool for pool in payload["pools"]}

    from types import SimpleNamespace
    from sqlalchemy import text
    from app.strategy.pools import load_pool_spec, resolve_members
    with Session(make_engine(read_only=True)) as db:
        universe = [str(s) for s in db.execute(
            select(Kline.symbol).where(Kline.interval == "1h", Kline.open_time >= start_ms).distinct()
        ).scalars().all()]
        pool_rows = db.execute(text(
            "select code, kind, symbols, rule from symbol_pools where enabled = 1 order by code"
        )).mappings().all()
    members: dict[str, list[str]] = {}
    for row in pool_rows:
        spec = load_pool_spec(SimpleNamespace(**dict(row)))
        members[row["code"]] = resolve_members(
            spec["kind"], spec["symbols"], spec["rule"],
            universe=lambda: universe, volumes=lambda: {}, metrics=None,
        )
    targets = sorted(set().union(*members.values()) & set(universe))
    print(f"benchmark universe={len(targets)}", flush=True)

    per_symbol: dict[str, dict] = {}
    for i, symbol in enumerate(targets, 1):
        frame = load(symbol, start_ms, last_complete)
        if frame is None:
            continue
        direction_long = pd.Series(1.0, index=frame.index)
        direction_short = pd.Series(-1.0, index=frame.index)
        result = evaluate_symbol(frame, {"long": direction_long, "short": direction_short},
                                 horizon=48, cost=0.0012, eval_days=int(payload["config"]["days"]))
        if result is None:
            continue
        per_symbol[symbol] = {
            "long": float(np.mean(result["indicators"]["long"]["net"])),
            "short": float(np.mean(result["indicators"]["short"]["net"])),
            "buy_hold_2d": result["two_day_return"],
        }
        if i % 100 == 0:
            print(f"  {i}/{len(targets)}", flush=True)

    benchmarks: dict[str, dict] = {}
    for code, pool_members in members.items():
        values = [per_symbol[s] for s in pool_members if s in per_symbol]
        if not values:
            continue
        longs = [v["long"] for v in values]
        shorts = [v["short"] for v in values]
        benchmarks[code] = {
            "n_symbols": len(values),
            "always_long_net": float(np.mean(longs)),
            "always_short_net": float(np.mean(shorts)),
            "always_long_ci": cluster_bootstrap(longs),
            "always_short_ci": cluster_bootstrap(shorts),
        }
        print(code, f"long={np.mean(longs) * 100:+.4f}%  short={np.mean(shorts) * 100:+.4f}%", flush=True)

    payload["data"]["benchmarks"] = benchmarks
    (ROOT / "docs/trend-direction-backtest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
