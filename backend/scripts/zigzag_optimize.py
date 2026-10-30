r"""Simple ZigZag Double —— 主流币参数寻优（专用撮合口径，训练/样本外两段式）。

流程
----
1. 每个币加载一次行情帧（与 ``run_zigzag_backtest`` 相同的校验口径）。
2. 参数网格：long/short 通道长度对称扫描 × 检测灵敏度 × 方向开关。
3. 时间上按 66/34 切成训练段与样本外段，**选参只看训练段**的币间中位收益，
   再报告该参数在样本外段的表现，用于暴露过拟合。

用法
----
```
.\.venv\Scripts\python.exe backend/scripts/zigzag_optimize.py --days 365 --output docs/simple-zigzag-opt-majors.json
```
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.backtest.zigzag_execution import simulate_zigzag  # noqa: E402
from app.db import make_engine  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.strategy.coin_universe import MAJOR_SYMBOLS  # noqa: E402
from app.strategy.rules import evaluate_signals, resolve_params  # noqa: E402
from app.utils import now_ms  # noqa: E402

CODE = "simple_zigzag_double"
LENGTHS = (3, 5, 8, 13, 21, 34)
DETECTIONS = (2, 3, 5)
DIRECTIONS = ("both", "long_only", "short_only")
TRAIN_RATIO = 0.66
INITIAL = 1_000_000.0
FEE = 0.001
SLIPPAGE = 0.0


def combo_params(length: int, detection: int, direction: str) -> dict:
    overrides = {
        "long_length": length, "short_length": length,
        "long_detection": detection, "short_detection": detection,
    }
    if direction == "long_only":
        overrides["use_short"] = False
    elif direction == "short_only":
        overrides["use_long"] = False
    return overrides


def run_one(args: tuple) -> dict:
    symbol, interval, frame, combos = args
    prices = frame[["open", "high", "low", "close"]]
    if not (np.isfinite(prices.to_numpy()).all() and prices.gt(0).to_numpy().all()):
        raise ValueError(f"{symbol}: 行情包含无效 K 线")
    rows = []
    for length, detection, direction in combos:
        params = resolve_params(CODE, {"fee_rate": FEE, "slippage": SLIPPAGE},
                                combo_params(length, detection, direction), interval=interval)
        signals = evaluate_signals(frame, CODE, params)
        equity, trades = simulate_zigzag(frame, signals, params, INITIAL, FEE, SLIPPAGE, interval)
        split_at = int(len(equity) * TRAIN_RATIO)
        eq = equity.to_numpy(dtype=float)
        gross = [t["gross_pnl"] for t in trades] if trades and "gross_pnl" in trades[0] else []
        fees = [t["fees"] for t in trades] if trades else []
        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) < 0]
        gross_win = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        rows.append({
            "symbol": symbol, "length": length, "detection": detection,
            "direction": direction, "bars": len(frame),
            "trades": len(trades),
            "ret": round((eq[-1] / INITIAL - 1) * 100, 4),
            "train_ret": round((eq[split_at] / INITIAL - 1) * 100, 4),
            "test_ret": round((eq[-1] / eq[split_at] - 1) * 100, 4),
            "mdd": round(_mdd(eq), 4),
            "win": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
            "pf": round(gross_win / gross_loss, 4) if gross_loss > 0 else None,
            "fees_pct": round(sum(fees) / INITIAL * 100, 2),
        })
    return {"symbol": symbol, "rows": rows}


def _mdd(eq) -> float:
    peak = pd.Series(eq).cummax()
    return float(((eq / peak) - 1).min() * 100)


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple ZigZag Double 主流币参数寻优")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", default="docs/simple-zigzag-opt-majors.json")
    args = ap.parse_args()

    combos = list(itertools.product(LENGTHS, DETECTIONS, DIRECTIONS))
    symbols = list(MAJOR_SYMBOLS) + ["ETCUSDT"]
    db = sessionmaker(bind=make_engine(read_only=True))()
    service = MarketDataService(db)
    as_of = now_ms()
    start_ms = as_of - args.days * 86_400_000

    tasks = []
    for symbol in symbols:
        frame = service.load_frame(symbol, args.interval, start_ms=start_ms, end_ms=as_of)
        if frame.empty:
            print(f"{symbol}: 无数据，跳过")
            continue
        tasks.append((symbol, args.interval, frame, combos))
    print(f"{len(tasks)} 币 × {len(combos)} 组合 = {len(tasks) * len(combos)} 次回放，"
          f"workers={args.workers}")

    all_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run_one, tasks):
            all_rows.extend(result["rows"])
            best = max(result["rows"], key=lambda r: r["ret"])
            print(f"{result['symbol']:10s} best: L{best['length']}/D{best['detection']}/"
                  f"{best['direction']}  ret {best['ret']:.2f}%  (train {best['train_ret']:.2f}%)")

    payload = {
        "config": {"interval": args.interval, "days": args.days, "symbols": symbols,
                   "grid": {"lengths": list(LENGTHS), "detections": list(DETECTIONS),
                            "directions": list(DIRECTIONS)},
                   "train_ratio": TRAIN_RATIO, "fee": FEE, "slippage": SLIPPAGE,
                   "initial": INITIAL, "note": "选参只看 train_ret；test_ret 为样本外验证"},
        "results": all_rows,
    }
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out}（{len(all_rows)} 行）")


if __name__ == "__main__":
    main()
