"""把 klines 里的上游周期派生成下游周期并落库（默认 1h -> 4h）。

用法：
    python scripts/derive_candles.py                    # 1h -> 4h，全币种
    python scripts/derive_candles.py --symbols BTCUSDT,ETHUSDT
    python scripts/derive_candles.py --source 1h --target 4h --min-fill 0.75
    python scripts/derive_candles.py --dry-run          # 只统计不写库

幂等：按 (symbol, interval, open_time) upsert，可反复执行。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select, text
from sqlalchemy.orm import sessionmaker

from app.db import make_engine
from app.market.derive import derive_interval
from app.models import Kline


def coverage(db, interval: str) -> dict:
    row = db.execute(
        select(
            func.count(func.distinct(Kline.symbol)),
            func.count(),
            func.min(Kline.open_time),
            func.max(Kline.open_time),
        ).where(Kline.interval == interval)
    ).one()
    span = (row[3] - row[2]) / 86_400_000 if row[2] else 0
    return {"symbols": row[0], "bars": row[1], "span_days": round(span)}


def main() -> int:
    ap = argparse.ArgumentParser(description="派生下游周期K线")
    ap.add_argument("--source", default="1h")
    ap.add_argument("--target", default="4h")
    ap.add_argument("--symbols", default="", help="逗号分隔，留空表示全币种")
    ap.add_argument("--min-fill", type=float, default=0.75)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    eng = make_engine(read_only=args.dry_run)
    db = sessionmaker(bind=eng)()

    before = coverage(db, args.target)
    print(f"[before] {args.target}: {before}")

    if args.dry_run:
        n_src = db.execute(
            select(func.count(func.distinct(Kline.symbol))).where(Kline.interval == args.source)
        ).scalar()
        print(f"[dry-run] 源 {args.source} 有 {n_src} 个币；未写入任何数据")
        return 0

    started = time.time()
    result = derive_interval(
        db, source=args.source, target=args.target,
        symbols=symbols or None, min_fill=args.min_fill, batch=args.batch,
    )
    print(f"[derive] {result}")
    after = coverage(db, args.target)
    print(f"[after ] {args.target}: {after}")
    print(f"[done  ] {time.time() - started:.1f}s")

    # 抽样校验：派生出的 4h 与 1h 逐根比对首尾与极值
    sample = db.execute(
        text(
            """
            select symbol from klines where interval = :t
             group by symbol order by count(*) desc limit 3
            """
        ),
        {"t": args.target},
    ).scalars().all()
    for sym in sample:
        ok = db.execute(
            text(
                """
                select count(*) from (
                  select k.open_time,
                         k.high as k_high, k.low as k_low,
                         (select max(h.high) from klines h
                           where h.symbol=k.symbol and h.interval=:s
                             and h.open_time >= k.open_time
                             and h.open_time < k.close_time) as src_high,
                         (select min(h.low) from klines h
                           where h.symbol=k.symbol and h.interval=:s
                             and h.open_time >= k.open_time
                             and h.open_time < k.close_time) as src_low
                    from klines k
                   where k.symbol=:sym and k.interval=:t
                ) x
                where abs(k_high - src_high) > 1e-9 or abs(k_low - src_low) > 1e-9
                """
            ),
            {"sym": sym, "s": args.source, "t": args.target},
        ).scalar()
        print(f"[check ] {sym}: high/low 不一致的桶 = {ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
