"""P5 探针：降换手（提高 min_score 开仓阈值）能否改善净收益率。

路线图判定：现货价格序列在 1m~1h 全频率上无方向性 alpha（P1/P2/P3/P4 已证），
信号 ≈随机。此时「少交易 = 少交手续费 = 少亏」是唯一确定有效的框架内净收益改善。

本脚本扫描 ml_ensemble 的 min_score（ML 概率开仓阈值，越高越保守、换手越低），
对多币在 1h/400 天回测，量化：
  - 成交笔数（换手代理）
  - 净收益% / 费用%（成本吸收率 = 费用% / |毛收益%|）
  - 跑赢随机基线比例
找到「净收益拐点」：再降阈值（更激进）不再显著改善、反而多交费的保守档。

复用：run_backtest(threshold=min_score) 直接覆盖 min_score（见 runner.py:141）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/
from app.db import make_engine
from app.backtest.runner import run_backtest
from app.utils import now_ms

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT"
DEFAULT_MIN_SCORES = "0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9"


def scan_one(db, symbol, interval, strategy, days, initial, fee, slippage, min_score,
             triple_barrier):
    try:
        result = run_backtest(
            db, symbol, interval, strategy, algo="lightgbm", days=days,
            initial_cash=initial, fee=fee, slippage=slippage, optimize=False,
            threshold=min_score, triple_barrier=triple_barrier,
        )
    except Exception as exc:
        return {"symbol": symbol, "min_score": min_score, "error": f"{type(exc).__name__}: {exc}"}
    m = result["metrics"]
    trades = result.get("trades", [])
    fees = sum(float(t.get("fees", 0.0)) for t in trades)
    net = float(m["total_return_pct"])
    gross = net + fees / initial * 100  # 毛收益 ≈ 净 + 费用
    cost_abs = abs(fees / initial * 100)
    absorption = round(cost_abs / abs(gross), 3) if gross != 0 else None
    return {
        "symbol": symbol,
        "min_score": min_score,
        "trades": len(trades),
        "net_return_pct": round(net, 3),
        "fees_pct": round(fees / initial * 100, 4),
        "cost_absorption": absorption,
        "beats_random": bool(net > float(m["random_baseline"]["total_return_pct"])),
        "random_ret": round(float(m["random_baseline"]["total_return_pct"]), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--min-scores", default=DEFAULT_MIN_SCORES)
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--strategy", default="ml_ensemble")
    ap.add_argument("--initial", type=float, default=10000.0)
    ap.add_argument("--fee", type=float, default=0.00075)
    ap.add_argument("--slippage", type=float, default=0.0005)
    ap.add_argument("--triple-barrier", action="store_true")
    ap.add_argument("--output", default="docs/probe-turnover.json")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    scores = [float(x) for x in args.min_scores.split(",") if x.strip()]
    engine = make_engine(read_only=True)
    end = now_ms()

    per_score = {s: [] for s in scores}
    with Session(engine) as db:
        for ms in scores:
            print(f"[probe] min_score={ms:.2f} ...", flush=True)
            for sym in symbols:
                r = scan_one(db, sym, args.interval, args.strategy, args.days,
                             args.initial, args.fee, args.slippage, ms, args.triple_barrier)
                if "error" not in r:
                    per_score[ms].append(r)

    summary = []
    for ms in scores:
        rows = per_score[ms]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        summary.append({
            "min_score": ms,
            "trades_mean": round(float(df["trades"].mean()), 1),
            "net_return_pct_mean": round(float(df["net_return_pct"].mean()), 3),
            "fees_pct_mean": round(float(df["fees_pct"].mean()), 4),
            "cost_absorption_mean": (round(float(df["cost_absorption"].dropna().mean()), 3)
                                     if df["cost_absorption"].notna().any() else None),
            "beats_random_ratio": round(float(df["beats_random"].mean()), 3),
            "n_symbols": len(rows),
        })

    # 推荐档：净收益最高（最不亏）且成交>=10（仍有效交易）
    valid = [s for s in summary if s["trades_mean"] >= 10]
    recommended = max(valid, key=lambda s: s["net_return_pct_mean"]) if valid else None
    base = next((s for s in summary if abs(s["min_score"] - 0.5) < 1e-9), None)

    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "params": vars(args),
        "symbols": symbols,
        "summary": summary,
        "baseline_min_score_0.5": base,
        "recommended": recommended,
        "verdict": (
            f"提高 min_score 到 {recommended['min_score']} 可将净收益从 "
            f"{base['net_return_pct_mean']}% 改善到 {recommended['net_return_pct_mean']}%，"
            f"成交 {base['trades_mean']}→{recommended['trades_mean']} 笔、费用降 "
            f"{base['fees_pct_mean']}→{recommended['fees_pct_mean']}%；"
            f"降换手是框架内确定有效的净收益改善（信号无 edge 时少交易=少亏）"
            if recommended and base else "无有效扫描结果"
        ),
    }
    with open(args.output, "w", encoding="utf-8") as h:
        json.dump(report, h, ensure_ascii=False, indent=2)

    print("\n=== 降换手扫描（min_score → 净收益/换手）===")
    print(f"{'min_score':>9s} {'成交':>7s} {'净收益%':>9s} {'费用%':>8s} {'成本吸收':>9s} {'跑赢随机':>9s}")
    for s in summary:
        print(f"{s['min_score']:>9.2f} {s['trades_mean']:>7.1f} {s['net_return_pct_mean']:>9.3f} "
              f"{s['fees_pct_mean']:>8.4f} {str(s['cost_absorption_mean']):>9} {s['beats_random_ratio']:>9.3f}")
    print(f"\n推荐档：{recommended['min_score'] if recommended else '无'}")
    print(f"判定：{report['verdict']}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
