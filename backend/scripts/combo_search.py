"""组合与参数搜索：在滚动样本外口径下比较单信号、合成规则与等权组合。

与 `indicator_lab.py` 的关系：那边是「单次切分 + 单指标横评 + 组合」，
这边是「共享资金池 + 多折滚动 + 组合级指标」。产出给「能不能上线」用，
不用于挑选历史最佳参数。

用法：
    .venv\\Scripts\\python.exe backend/scripts/combo_search.py --interval 1h --days 400 --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sqlalchemy.orm import sessionmaker

from app.db import make_engine
from app.research import lab_experiment, lab_grid, lab_regime, rolling_portfolio
from app.research.lab_grid import Candidate
from app.research.lab_signals import Signals
from app.research.lab_sim import Trade, to_ms
from app.research.rolling_portfolio import PortfolioConfig


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--slippage", type=float, default=0.0002)
    ap.add_argument("--max-hold", type=int, default=240)
    ap.add_argument("--max-concurrent", type=int, default=0, help="同时最多几条腿（0=按币种数）")
    ap.add_argument("--risk-per-trade", type=float, default=0.0, help="单笔风险预算占总资金比例")
    ap.add_argument("--atr-mult", type=float, default=0.0, help="止损 ATR 倍数（配合风险预算）")
    ap.add_argument("--side", choices=["both", "long", "short"], default="both")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--train-ratio", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--families", default="")
    ap.add_argument("--regimes", default="", help="逗号分隔，如 market_ema:200")
    ap.add_argument("--out-prefix", default="docs/combo-search")
    return ap.parse_args()


def signal_frames(ds, cand: Candidate) -> dict[str, Signals]:
    return {s: lab_experiment.sign(cand, f) for s, f in ds.frames.items()}


def gate(signals: dict[str, Signals], masks: dict[str, np.ndarray], side: int) -> dict[str, Signals]:
    """只拦入场；只做多时忽略空头信号。"""
    out: dict[str, Signals] = {}
    for symbol, sig in signals.items():
        mask = masks.get(symbol)
        longs = sig.long if mask is None else (sig.long & mask)
        shorts = sig.short if mask is None else (sig.short & mask)
        if side == 1:
            shorts = np.zeros_like(shorts)
        elif side == -1:
            longs = np.zeros_like(longs)
        out[symbol] = Signals(longs, shorts)
    return out


def combine(a: dict[str, Signals], b: dict[str, Signals], rule: str) -> dict[str, Signals]:
    out: dict[str, Signals] = {}
    for symbol in a:
        x, y = a[symbol], b[symbol]
        if rule == "and":
            out[symbol] = Signals(x.long & y.long, x.short & y.short)
        else:
            out[symbol] = Signals(x.long | y.long, x.short | y.short)
    return out


def buy_hold_legs(ds, cfg: PortfolioConfig, start: int, end: int, warmup: int = 200) -> list[Trade]:
    """买入持有对照：在给定折区间内每币一直持有多单，用来分离 beta 与 alpha。

    必须按折切段，否则「一笔跨全样本」的持仓不会落在任何一折的窗口里，
    对照会退化成 0 收益而失去意义。
    """
    out: list[Trade] = []
    slots = cfg.max_concurrent if cfg.max_concurrent > 0 else max(len(ds.frames), 1)
    stake = cfg.capital / float(slots)
    for symbol, frame in ds.frames.items():
        stamps = to_ms(frame.index)
        open_ = frame["open"].to_numpy(dtype=float)
        close = frame["close"].to_numpy(dtype=float)
        lo = int(np.searchsorted(stamps, start))
        hi = int(np.searchsorted(stamps, end, side="right")) - 1
        if hi - lo <= warmup + 1:
            continue
        entry_price = float(open_[lo]) * (1.0 + cfg.slippage)
        exit_price = float(close[hi]) * (1.0 - cfg.slippage)
        qty = stake / entry_price
        pnl = qty * (exit_price - entry_price) - cfg.fee * stake - cfg.fee * qty * exit_price
        out.append(
            Trade(symbol, 1, int(stamps[lo]), int(stamps[hi]), entry_price, exit_price,
                  hi - lo, float(pnl), float(pnl / stake), 0.0, "buy_hold")
        )
    return out


def _fold_bounds(ds, folds: int, train_ratio: float) -> list[int]:
    lo = min(int(to_ms(f.index)[0]) for f in ds.frames.values())
    hi = max(int(to_ms(f.index)[-1]) for f in ds.frames.values())
    first_test = lo + int((hi - lo) * train_ratio)
    step = (hi - first_test) / max(folds, 1)
    return [first_test] + [int(first_test + step * (i + 1)) for i in range(folds)]


def main() -> None:
    args = parse_args()
    families = [f.strip() for f in args.families.split(",") if f.strip()] or None
    side_code = {"both": 0, "long": 1, "short": -1}[args.side]
    cfg = PortfolioConfig(
        capital=args.capital, fee=args.fee, slippage=args.slippage,
        max_hold_bars=args.max_hold, side=side_code, min_trades=args.min_trades,
        max_concurrent=args.max_concurrent,
        risk_per_trade_pct=args.risk_per_trade, atr_mult=args.atr_mult,
    )
    started = time.time()
    db = sessionmaker(bind=make_engine(read_only=True))()
    ds = lab_experiment.load_dataset(db, args.interval, args.days, args.limit, train_ratio=args.train_ratio)
    print(f"{len(ds.frames)} 币 / {ds.meta()['bars_total']} 根 bar / 切点 {ds.meta()['split'][:16]}", flush=True)

    cands = lab_grid.candidates(families)
    base_signals = {c.key: signal_frames(ds, c) for c in cands}
    legs: dict[str, dict[str, Signals]] = {c.label: base_signals[c.key] for c in cands}
    print(f"单信号腿 {len(legs)} 条，开始滚动检验 ...", flush=True)

    regime_specs = [p.split(":") for p in args.regimes.split(",") if p.strip()]
    for kind, n_text in regime_specs:
        masks = lab_regime.build(ds.frames, kind, int(n_text))
        label = lab_regime.describe(kind, int(n_text))
        for c in cands:
            legs[f"{c.label} + {label}"] = gate(base_signals[c.key], masks, side_code)

    singles = rolling_portfolio.rolling_evaluate(
        ds, legs, cfg, folds=args.folds, train_ratio=args.train_ratio, top_k=1
    )
    portfolio = rolling_portfolio.rolling_evaluate(
        ds, legs, cfg, folds=args.folds, train_ratio=args.train_ratio, top_k=args.top_k
    )

    # 对照：每一折区间内「一直做多」的收益，用来判断组合是不是只在赚市场 beta
    bounds = _fold_bounds(ds, args.folds, args.train_ratio)
    hold_folds = []
    for k in range(len(bounds) - 1):
        start, end = bounds[k], bounds[k + 1]
        hold_folds.append(
            {
                "fold": k + 1,
                "start": start,
                "end": end,
                "test": rolling_portfolio.portfolio_metrics(
                    buy_hold_legs(ds, cfg, start, end), cfg,
                    rolling_portfolio._bars_between(ds, start, end), len(ds.frames)
                ),
            }
        )

    # 统计判定：把每折选中的腿合并，给每笔期望的 bootstrap CI
    leg_trades = portfolio.get("leg_trades", {})
    pooled_trades: list[Trade] = []
    for row in portfolio["folds"]:
        start, end = row["start"], row["end"]
        for label in row["selected"]:
            pooled_trades.extend(
                [t for t in leg_trades.get(label, []) if start <= t.exit_ts < end]
            )
    oos_stats = rolling_portfolio.expectancy(pooled_trades, cfg.capital)

    payload = {
        "config": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "interval": args.interval, "days": args.days, "limit": args.limit,
            "capital": args.capital, "fee": args.fee, "slippage": args.slippage,
            "side": args.side, "folds": args.folds, "train_ratio": args.train_ratio,
            "top_k": args.top_k, "min_trades": args.min_trades,
            "max_concurrent": args.max_concurrent,
            "risk_per_trade_pct": args.risk_per_trade, "atr_mult": args.atr_mult,
            "regimes": args.regimes, "families": families,
        },
        "dataset": ds.meta(),
        "single_leg_folds": singles["folds"],
        "portfolio_folds": portfolio["folds"],
        "buy_hold_folds": hold_folds,
        "oos_expectancy": oos_stats,
        "legs_count": len(legs),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    root = Path(__file__).resolve().parents[2]
    out = root / args.out_prefix
    out.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"已写 {out.relative_to(root)}.json（{len(legs)} 条腿，{payload['elapsed_seconds']}s）")
    for row in portfolio["folds"]:
        test = row.get("test") or {}
        hold = next((h["test"] for h in hold_folds if h["fold"] == row["fold"]), {})
        print(f"折 {row['fold']}: 选中 {row['selected']} 样本外收益 {test.get('return_pct')}% 回撤 {test.get('max_drawdown_pct')}% 买入持有 {hold.get('return_pct')}%")


if __name__ == "__main__":
    main()
