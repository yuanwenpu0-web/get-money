r"""时点池（PIT）对照：研究池的正收益是选币还是策略。

问题
----
`research_high/low_turnover_high_vol` 的两个池由「最近 168 小时成交额 + 波动率」筛出，
库里存的是**当前**成员（`rule.window_end_ms` ≈ 现在）。拿这批成员回放过去 180 天，
等于用今天知道会涨的币去赌昨天——典型事后选币，读数必然虚高。
`docs/research-interval-exit-study.md` 已把这一点写进边界，但没有量化它值多少钱。

做法
----
按 30 天一期往回推 5 期：对每一期起点 T，用 **T 之前的 168 小时** 重建池成员
（规则与 `strategy/pool_maintenance.py::research_pool_candidates` 一致），再只用
[T, T+30d) 的行情回放。两臂参数、成本、撮合完全相同，唯一变量是「成员怎么来的」：

- `pit`：T 时刻按规则重建（线上每轮维护的真实形态，可交易）
- `posthoc`：当前库里的成员（事后选币，不可交易但读数虚高）

口径偏差：原规则还要求每个币在窗口内有完整 1m 数据，但 1m 只覆盖约 90 个币，
照抄会让 PIT 重建几乎无币可用。这里放宽成「1h 至少 160 根且末根不早于 T−2h」，
两臂共用同一份重建逻辑，差异只体现在成员构成上。

    .\.venv\Scripts\python.exe backend/scripts/pit_pool_check.py --folds 5 --span 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.db import SessionLocal, make_engine  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.models import SymbolPool  # noqa: E402
from app.research import combo_engine as ce  # noqa: E402
from app.strategy.bindings import active_bindings  # noqa: E402
from app.strategy.pool_maintenance import EXCLUDED, WINDOW_HOURS  # noqa: E402
from app.utils import now_ms  # noqa: E402

HOUR = 3_600_000
POOLS = {"high": "research_high_liquidity_high_volatility",
         "low": "research_low_liquidity_high_volatility"}
ARMS = {
    "A": {"pool": "high", "params": [("current", 0, 120), ("best", 0, 168)]},
    "B": {"pool": "low", "params": [("current", 1, 48), ("best", 0, 168)]},
}
WORKER_WARMUP = 420

_DB = None


def db():
    global _DB
    if _DB is None:
        _DB = sessionmaker(bind=make_engine(read_only=True, null_pool=True))()
    return _DB


def rebuild(end_ms: int) -> tuple[list[str], list[str], dict]:
    """按 168 小时成交额/波动率重建两池成员（规则与线上维护一致）。"""
    start = end_ms - WINDOW_HOURS * HOUR
    rows = db().execute(text(
        "select symbol, open_time, close, quote_volume from klines "
        "where interval='1h' and open_time>=:s and open_time<:e"
    ), {"s": start, "e": end_ms}).all()
    frame = pd.DataFrame(rows, columns=["symbol", "open_time", "close", "quote_volume"])
    stats = []
    for symbol, group in frame.groupby("symbol"):
        if (not symbol.endswith("USDT") or symbol in EXCLUDED
                or symbol.endswith(("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"))):
            continue
        group = group.sort_values("open_time")
        if len(group) < 160 or group["open_time"].iloc[-1] + HOUR < end_ms - 2 * HOUR:
            continue
        volume = float(group["quote_volume"].sum())
        closes = group["close"].to_numpy(float)
        if not np.isfinite(closes).all() or (closes <= 0).any() or volume <= 0:
            continue
        stats.append((symbol, volume, float(np.std(np.diff(np.log(closes)), ddof=1))))
    top = sorted(stats, key=lambda item: (-item[1], item[0]))[:100]
    volatile = {item[0] for item in sorted(top, key=lambda item: (-item[2], item[0]))[:50]}
    high = [item[0] for item in top[:50] if item[0] in volatile]
    low = [item[0] for item in top[50:] if item[0] in volatile]
    return high, low, {"eligible": len(stats)}


def replay(symbols: list[str], start: int, end: int, channel: int, hold_hours: int) -> list[float]:
    hold = int(hold_hours * HOUR // HOUR)
    out = []
    service = MarketDataService(db())
    for symbol in symbols:
        frame = service.load_frame(symbol, "1h", start_ms=start - WORKER_WARMUP * HOUR, end_ms=end)
        if len(frame) < hold + 300:
            continue
        for trade in ce.research_trades(frame, channel, hold):
            if trade["t0"] >= pd.Timestamp(start, unit="ms", tz="UTC"):
                out.append(trade["ret"])
    return out


def beta(symbols: list[str], start: int, end: int) -> float | None:
    service = MarketDataService(db())
    marks = []
    for symbol in symbols:
        frame = service.load_frame(symbol, "1h", start_ms=start, end_ms=end)
        if len(frame) > 10:
            marks.append(float(frame["close"].iloc[-1] / frame["close"].iloc[0] - 1) * 100)
    return round(float(np.mean(marks)), 2) if marks else None


def main() -> int:
    ap = argparse.ArgumentParser(description="研究池时点（PIT）对照")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--span", type=int, default=30, help="每期长度（天）")
    ap.add_argument("--out", default=str(ROOT / "output" / "pit-pool-check"))
    args = ap.parse_args()

    session = SessionLocal()
    current = {}
    for key, code in POOLS.items():
        pool = session.query(SymbolPool).filter(SymbolPool.code == code).first()
        current[key] = json.loads(pool.symbols or "[]") if pool else []
    session.close()

    end = now_ms()
    folds = []
    store: dict[str, list[float]] = {}
    for index in range(args.folds):
        stop = end - index * args.span * 86_400_000
        start = stop - args.span * 86_400_000
        high, low, meta = rebuild(start)
        members = {"high": high, "low": low}
        print(f"fold {index + 1}: {str(pd.Timestamp(start, unit='ms', tz='UTC'))[:10]} → "
              f"{str(pd.Timestamp(stop, unit='ms', tz='UTC'))[:10]} · "
              f"PIT high={len(high)} low={len(low)}", flush=True)
        record = {"start": str(pd.Timestamp(start, unit="ms", tz="UTC"))[:10],
                  "end": str(pd.Timestamp(stop, unit="ms", tz="UTC"))[:10],
                  "eligible": meta["eligible"],
                  "overlap_high": len(set(high) & set(current["high"])),
                  "overlap_low": len(set(low) & set(current["low"])),
                  "arms": {}}
        for arm, spec in ARMS.items():
            pool = spec["pool"]
            for tag, channel, hold in spec["params"]:
                for label, symbols in (("pit", members[pool]), ("posthoc", current[pool])):
                    values = replay(symbols, start, stop, channel, hold)
                    key = f"{arm}|{tag}|{label}"
                    store.setdefault(key, []).extend(values)
                    record["arms"][key] = {"pooled": ce.pooled(values), "symbols": len(symbols)}
            record["beta_" + pool] = beta(members[pool], start, stop)
        folds.append(record)
        print("  " + "  ".join(f"{k}=n{v['pooled']['trades']}/{(v['pooled']['mean'] or 0):.2f}"
                              for k, v in record["arms"].items()), flush=True)

    summary = {}
    for key in folds[0]["arms"]:
        values = [f["arms"][key]["pooled"] for f in folds if key in f["arms"]]
        merged = [item for item in values if item["trades"]]
        total = sum(item["trades"] for item in merged)
        weighted = (sum((item["mean"] or 0) * item["trades"] for item in merged) / total) if total else None
        summary[key] = {"folds": len(merged),
                        "trades": total,
                        "pooled": ce.pooled(store.get(key, [])),
                        "weighted_mean": round(weighted, 4) if weighted is not None else None,
                        "fold_means": [item["mean"] for item in merged],
                        "positive_folds": sum(1 for item in merged if (item["mean"] or 0) > 0)}
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "folds": args.folds, "span_days": args.span,
               "costs": {"round_trip": 0.0012, "basis": "notional"},
               "current_members": {k: len(v) for k, v in current.items()},
               "records": folds, "summary": summary}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str),
                                        encoding="utf-8")
    print(f"\n已写出 {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
