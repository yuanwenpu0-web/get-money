"""候选冗余度检查：被反复选中的信号是不是同一笔交易重复计数。

滚动检验每折都选出 rsi_reversal / mfi_reversal 这一族，如果它们的入场时点
高度重合，那么「3 条腿组合」只是同一笔交易被复制三次：成交数虚高、置信区间
被人为收窄，组合带来的分散是假象。这里直接量化两两重合度。

用法：
    .venv\\Scripts\\python.exe backend/scripts/correlation_check.py --interval 1h --days 400 --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sqlalchemy.orm import sessionmaker

from app.db import make_engine
from app.research import lab_experiment, lab_grid
from app.research.lab_experiment import strategy_trades
from app.research.lab_grid import Candidate
from app.research.lab_sim import SimConfig


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--side", choices=["both", "long", "short"], default="long")
    ap.add_argument("--top", type=int, default=10, help="只看训练盈亏前 N 个候选")
    ap.add_argument("--out", default="docs/_correlation_check.json")
    return ap.parse_args()


def entry_set(trades) -> set[tuple[str, int, int]]:
    return {(t.symbol, t.entry_ts, t.side) for t in trades}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    args = parse_args()
    side_code = {"both": 0, "long": 1, "short": -1}[args.side]
    cfg = SimConfig(side=side_code)
    db = sessionmaker(bind=make_engine(read_only=True))()
    ds = lab_experiment.load_dataset(db, args.interval, args.days, args.limit)

    cands = lab_grid.candidates()
    rows = []
    for cand in cands:
        trades = strategy_trades(ds, cand, cfg)
        train = [t for t in trades if t.exit_ts < ds.split_ts]
        rows.append((cand, trades, train))
    rows.sort(key=lambda item: sum(t.pnl for t in item[2]), reverse=True)
    top = rows[: args.top]

    sets = {cand.label: entry_set(trades) for cand, trades, _ in top}
    labels = list(sets)
    matrix = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            matrix.append({"a": a, "b": b, "jaccard": round(jaccard(sets[a], sets[b]), 4)})
    matrix.sort(key=lambda row: row["jaccard"], reverse=True)

    # 组合后的唯一入场数 vs 各腿入场数之和：衡量分散是真还是假
    union = set()
    total = 0
    for cand, trades, _ in top[:3]:
        union |= sets[cand.label]
        total += len(sets[cand.label])
    payload = {
        "config": {"interval": args.interval, "days": args.days, "limit": args.limit,
                   "side": args.side, "top": args.top},
        "dataset": ds.meta(),
        "top_candidates": [
            {"label": c.label, "train_trades": len(tr), "train_pnl": round(sum(t.pnl for t in tr), 4)}
            for c, _, tr in top
        ],
        "pairwise_jaccard_top": matrix[:20],
        "top3_union_entries": len(union),
        "top3_sum_entries": total,
        "redundancy_ratio": round(1.0 - len(union) / total, 4) if total else None,
    }
    out = Path(__file__).resolve().parents[2] / args.out
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"已写 {out}")
    print(f"Top3 入场数之和 {total}，去重后唯一入场 {len(union)}，冗余比例 {payload['redundancy_ratio']}")
    for row in matrix[:10]:
        print(f"  {row['jaccard']:.3f}  {row['a']}  ||  {row['b']}")


if __name__ == "__main__":
    main()
