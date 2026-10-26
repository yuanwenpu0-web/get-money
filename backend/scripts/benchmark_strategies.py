"""Read-only, reproducible comparison against stored market history.

Run from the repository root; no model, order or state rows are persisted.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.backtest.runner import run_backtest  # noqa: E402
from app.db import make_engine  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default=datetime.now(timezone.utc).isoformat())
    parser.add_argument("--days", type=int, nargs="+", default=[30, 40])
    parser.add_argument("--interval", nargs="+", default=["1m"], help="K线周期，可多选")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["ml_ensemble", "trend_follow", "mean_reversion", "funding_extreme"],
        help="策略列表，可多选",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs" / "strategy-benchmark"
    )
    parser.add_argument(
        "--fee", type=float, default=None,
        help="单边手续费率，默认 0.0004；--maker 等价于 0.0002",
    )
    parser.add_argument(
        "--maker", action="store_true",
        help="maker 限价单费率假设：手续费 0.0002，滑点保持 0.0002 不变",
    )
    args = parser.parse_args()
    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
    if as_of.tzinfo is None:
        parser.error("--as-of must include a timezone")
    fee = args.fee if args.fee is not None else (0.0002 if args.maker else 0.0004)
    slippage = 0.0002
    if fee < 0:
        parser.error("--fee 不能为负")
    end_ms = int(as_of.timestamp() * 1000)
    engine = make_engine(read_only=True)
    results = []
    with Session(engine) as db:
        for interval in args.interval:
            for days in args.days:
                for symbol in ("BTCUSDT", "ETHUSDT"):
                    for strategy in args.strategies:
                        result = run_backtest(
                            db,
                            symbol,
                            interval,
                            strategy,
                            days=days,
                            initial_cash=10000.0,
                            fee=fee,
                            slippage=slippage,
                            optimize=True,
                            end_ms=end_ms,
                        )
                        result.update(
                            symbol=symbol, interval=interval, strategy=strategy, days=days
                        )
                        results.append(result)
                        metrics = result["metrics"]
                        print(
                            json.dumps(
                                {
                                    "interval": interval,
                                    "days": days,
                                    "symbol": symbol,
                                    "strategy": strategy,
                                    "baseline_return": metrics["baseline"][
                                        "total_return_pct"
                                    ],
                                    "return": metrics["total_return_pct"],
                                    "drawdown": metrics["max_drawdown_pct"],
                                    "trades": metrics["total_trades"],
                                    "selected": metrics["validation"]["selected"],
                                }
                            ),
                            flush=True,
                        )
    engine.dispose()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"as_of": as_of.isoformat(), "days": args.days, "intervals": args.interval,
               "fee": fee, "slippage": slippage, "maker": bool(args.maker), "results": results}
    args.output.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 策略优化验证报告",
        "",
        f"评估截止：{as_of.isoformat()}；本地 BTC/ETH 数据（周期：{', '.join(args.interval)}），只读运行。",
        "",
        "前 60% 训练、15% 验证选择固定候选参数、后 25% 样本外测试；边界清除预测跨度。"
        f"每次初始资金 10,000 USDT，单边手续费 {fee * 100:.4g}%、滑点 {slippage * 100:.4g}%"
        + ("（maker 限价单费率假设）。" if args.maker else "。"),
        "",
        "基线使用原始方向信号并关闭行情/成本过滤，两组共享风险预算、杠杆、冷却、移动止损、"
        "下一根开盘撮合与双边成本。它用于隔离过滤及选参效果，不能与旧 vectorbt 的收益直接比较。",
        "",
        "| 周期 | 天数 | 币种 | 策略 | 基线收益 % | 优化收益 % | 基线回撤 % | 优化回撤 % | 成交 基线→优化 | 费用 基线→优化 |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        m, b = r["metrics"], r["metrics"]["baseline"]
        lines.append(
            f"| {r['interval']} | {r['days']} | {r['symbol']} | {r['strategy']} | {b['total_return_pct']:.4f} | {m['total_return_pct']:.4f} | {b['max_drawdown_pct']:.4f} | {m['max_drawdown_pct']:.4f} | {b['total_trades']} → {m['total_trades']} | {b['total_fees']:.2f} → {m['total_fees']:.2f} |"
        )
    lines += ["", "## 各次选参与数据质量", ""]
    for r in results:
        m = r["metrics"]
        lines += [
            f"- {r['interval']} {r['days']} 天 {r['symbol']} {r['strategy']}：{m['validation']['selected']}（{m['validation']['status']}）；测试 {m['start']} → {m['end']}；{m['total_trades']} 笔。"
        ]
        lines += [f"  {warning}" for warning in m["warnings"]]
    lines += [
        "",
        "## 结论范围",
        "",
        "- 所有组合均列出，未根据测试收益删除不利结果；没有把测试最优参数自动写回模拟盘。",
        "- 负收益表示亏损，即使优于基线也不代表已盈利；少量成交和零成交不能证明稳定优势。",
        "- 两个历史窗口的测试段重叠，不是两份独立证据；仍需要更长、连续行情及向前模拟。",
        "- 缺口处无法重建真实路径；报告采用下一可见开盘平仓，可能低估缺口内风险。",
        "- 不含资金费率、强平、盘口冲击；止损风险预算在跳空时可能被突破。",
        "",
        "## 复现",
        "",
        "```powershell",
        f'.\\.venv\\Scripts\\python.exe backend/scripts/benchmark_strategies.py --as-of "{as_of.isoformat()}" --days '
        + " ".join(map(str, args.days))
        + (" --maker" if args.maker else "")
        + (f" --fee {args.fee}" if args.fee is not None else ""),
        "```",
        "",
        "需保留相同的本地 K 线和策略/全局参数；完整参数与曲线记录于同名 JSON。",
        "",
        "方法参考：[scikit-learn 时间序列隔离](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)、"
        "[Freqtrade 前视偏差检查](https://www.freqtrade.io/en/stable/lookahead-analysis/)。",
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
