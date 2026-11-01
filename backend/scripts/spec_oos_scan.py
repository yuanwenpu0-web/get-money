"""对两个通用策略（supertrend_current / absorption_reversal）做样本内/外扫描。

在 1h × 400 天 × top-N 流动性币上，用 run_backtest(optimize=True) 给出每币样本外指标，
并汇总「是否跑赢买入持有 / 随机基线 / 期望 bootstrap CI 是否跨 0」。

专用 broker 策略（OCB/RGD/zigzag）走各自脚本、不在本扫描内，结论引用 docs/*-strategy.md。

用法（项目根目录）：
    .venv\\Scripts\\python.exe backend/scripts/spec_oos_scan.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "backend")

import numpy as np
from sqlalchemy.orm import sessionmaker

from app.backtest.runner import run_backtest
from app.db import make_engine
from app.research.lab_experiment import list_symbols

INTERVAL = "1h"
DAYS = 400
TOP = 12
TRAIN = 0.75
FEE = 0.0004
SLIP = 0.0002
STRATS = ["supertrend_current", "absorption_reversal"]


def main() -> None:
    eng = make_engine(read_only=True)
    db = sessionmaker(bind=eng)()
    syms = list_symbols(db, INTERVAL, DAYS, TOP, int(time.time() * 1000))
    if syms and isinstance(syms[0], tuple):
        syms = [s for s, _ in syms]
    print(f"symbols({len(syms)}): {syms}", flush=True)

    out: dict = {}
    for strat in STRATS:
        rows = []
        for i, sym in enumerate(syms):
            try:
                r = run_backtest(
                    db, sym, INTERVAL, strat, days=DAYS, fee=FEE,
                    slippage=SLIP, train_ratio=TRAIN, optimize=True,
                )
                m = r["metrics"]
                if i == 0:
                    print("METRICS_KEYS:", sorted(m.keys()), flush=True)
                rows.append(
                    {
                        "symbol": sym,
                        "oos_return_pct": m.get("total_return_pct"),
                        "oos_win_rate": m.get("win_rate"),
                        "oos_sharpe": _trade_sharpe(r.get("trades"), m.get("initial_cash") or 10000.0),
                        "oos_trades": m.get("total_trades"),
                        "oos_max_dd_pct": m.get("max_drawdown_pct"),
                        "buy_hold_pct": m.get("buy_hold_return_pct"),
                        "baseline_return_pct": (m.get("baseline") or {}).get("total_return_pct"),
                        "random_return_pct": (m.get("random_baseline") or {}).get("total_return_pct"),
                        "expectancy_verdict": (m.get("significance") or {})
                        .get("expectancy_ci", {})
                        .get("verdict"),
                        "validation": (m.get("validation") or {}).get("selected"),
                    }
                )
            except Exception as e:  # noqa: BLE001
                rows.append({"symbol": sym, "error": repr(e)[:200]})
            print(f"  [{strat}] {i + 1}/{len(syms)} {sym} done", flush=True)
        out[strat] = {"symbols": rows, **_agg(rows)}

    Path("docs/spec-oos-scan.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("WRITTEN docs/spec-oos-scan.json", flush=True)


def _trade_sharpe(trades: list, capital: float = 10000.0):
    """单笔收益率(pnl/资金)口径的 Sharpe，与 combo_search 的 _sharpe 同分母，避免绝对值放大。"""
    pnls = [
        float(t["pnl"]) / float(capital)
        for t in (trades or [])
        if isinstance(t, dict) and t.get("pnl") is not None
    ]
    if len(pnls) < 2:
        return None
    arr = np.array(pnls, dtype=float)
    std = arr.std()
    if std == 0 or not np.isfinite(std):
        return None
    return round(float(arr.mean() / std), 3)


def _agg(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]

    def mean(key: str):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    return {
        "n": len(ok),
        "mean_oos_return_pct": mean("oos_return_pct"),
        "mean_oos_win_rate": mean("oos_win_rate"),
        "mean_oos_sharpe": mean("oos_sharpe"),
        "beats_buyhold": sum(1 for r in ok if r["oos_return_pct"] > r["buy_hold_pct"]),
        "beats_random": sum(1 for r in ok if r["oos_return_pct"] > r["random_return_pct"]),
        "positive_edge": sum(1 for r in ok if r.get("expectancy_verdict") == "positive"),
        "inconclusive_edge": sum(
            1 for r in ok if r.get("expectancy_verdict") in ("inconclusive", None)
        ),
        "negative_edge": sum(1 for r in ok if r.get("expectancy_verdict") == "negative"),
    }


if __name__ == "__main__":
    main()
