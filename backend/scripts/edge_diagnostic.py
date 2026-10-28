"""交易经济学诊断：把「毛 edge」和「成本」拆开，回答"该往哪个方向优化"。

为什么要有这个脚本
--------------------
净收益 = 毛 edge − 成本。二者混在一起时，看到"收益 0.3%"无法判断该去
（a）找更强的信号，还是（b）降成本 / 降换手。本脚本把两件事隔离：

1. **成本吸收率** = 总费用 / 毛收益。接近 1 说明策略几乎在给交易所打工，
   此时继续调参数的天花板极低，必须先降成本或降换手。
2. **零成本对照**：把 fee/slippage 设为 0 重跑同一段行情。若零成本收益也很低，
   说明问题在信号本身（方向没信息量），而不是成本。
3. **随机方向基线**：复用 execution.randomize_directions 的口径，
   若实盘方向收益不显著优于随机翻转，说明方向判断无信息量。
4. **阻断漏斗**：blocked_signals 说明候选信号被哪条规则杀掉，
   用于判断是"信号太少"还是"过滤太狠"。

输出：docs/edge-diagnostic.md + 同名 .json。只读运行，不写订单、不改系统状态。

用法：
    .\\.venv\\Scripts\\python.exe backend/scripts/edge_diagnostic.py --days 60
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.backtest.runner import run_backtest  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.utils import now_ms  # noqa: E402
from app.db import make_engine  # noqa: E402

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "AVAXUSDT", "PENDLEUSDT"]


def data_width(db: Session, symbol: str, interval: str) -> dict:
    """对比「引擎实际能用的宽度」与「库里躺着的 native 宽度」。

    merged 口径 = 1m 聚合 + native 回落（回测/训练真实使用的路径），直接实测；
    minute 口径 = 1m 原始跨度（回落前的上界）；
    native 口径 = 直查该 interval 的已存行。
    minute 与 native 的差距就是曾经被白白浪费的历史。
    """
    out = {}
    try:
        frame = MarketDataService(db).load_frame(symbol, interval, start_ms=0, end_ms=now_ms())
        if not frame.empty:
            first_ms = int(frame.index[0].timestamp() * 1000)
            last_ms = int(frame.index[-1].timestamp() * 1000)
            out["merged"] = {
                "rows": len(frame),
                "span_days": round((last_ms - first_ms) / 86_400_000, 1),
                "first": str(frame.index[0].date()),
                "last": str(frame.index[-1].date()),
                "sources": frame.attrs.get("candle_sources"),
            }
    except Exception as exc:
        out["merged"] = {"error": f"{type(exc).__name__}: {exc}"}
    for label, iv in (("native", interval), ("minute", "1m")):
        row = db.execute(
            text(
                "select count(*), min(open_time), max(open_time) from klines "
                "where symbol=:s and interval=:i"
            ),
            {"s": symbol, "i": iv},
        ).one()
        count, first, last = int(row[0]), row[1], row[2]
        span = (last - first) / 86_400_000 if count and first is not None else 0.0
        out[label] = {
            "rows": count,
            "span_days": round(span, 1),
            "first": str(datetime.fromtimestamp(first / 1000, timezone.utc).date()) if first else None,
            "last": str(datetime.fromtimestamp(last / 1000, timezone.utc).date()) if last else None,
        }
    # 回落前引擎只能用 1m 覆盖的跨度；回落后 merged 才是真实可用宽度
    out["derived_span_days"] = out["minute"]["span_days"]
    out["wasted_ratio"] = (
        round(1 - out["derived_span_days"] / out["native"]["span_days"], 3)
        if out["native"]["span_days"]
        else None
    )
    merged_span = out.get("merged", {}).get("span_days", 0)
    out["gain_ratio"] = (
        round(merged_span / out["derived_span_days"], 2) if out["derived_span_days"] else None
    )
    return out


def trade_stats(trades: list[dict], initial: float) -> dict:
    """从成交明细拆出毛/净/成本，以及胜率与持仓时长分布。"""
    if not trades:
        return {"trades": 0}
    n = len(trades)
    gross = sum(t["pnl"] + t["fees"] for t in trades)          # 未扣费用的盈亏
    fees = sum(t["fees"] for t in trades)
    net = sum(t["pnl"] for t in trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    gross_wins = sum(1 for t in trades if t["pnl"] + t["fees"] > 0)
    margins = [t["margin"] for t in trades]
    return {
        "trades": n,
        "gross_pnl": round(gross, 2),
        "fees": round(fees, 2),
        "net_pnl": round(net, 2),
        "gross_return_pct": round(gross / initial * 100, 4),
        "cost_return_pct": round(fees / initial * 100, 4),
        "net_return_pct": round(net / initial * 100, 4),
        # 成本吸收率：费用占毛收益的比例。>0.7 视为"成本主导"
        "cost_absorption": round(fees / gross, 3) if gross > 0 else None,
        "win_rate": round(wins / n * 100, 2),
        "gross_win_rate": round(gross_wins / n * 100, 2),
        "avg_margin": round(sum(margins) / n, 2),
        "avg_margin_pct": round(sum(margins) / n / initial * 100, 3),
        "avg_leverage": round(sum(t["leverage"] for t in trades) / n, 2),
        "avg_fee_per_trade": round(fees / n, 4),
        "exit_reasons": _counts(t["status"] for t in trades),
    }


def _counts(values) -> dict:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def diagnose_one(db: Session, symbol: str, interval: str, strategy: str,
                 days: int, initial: float, fee: float, slippage: float,
                 triple_barrier: bool = False) -> dict:
    row: dict = {"symbol": symbol, "label_mode": "triple_barrier" if triple_barrier else "binary"}
    row["data_width"] = data_width(db, symbol, interval)

    # 主跑：真实成本
    try:
        result = run_backtest(
            db, symbol, interval, strategy, algo="lightgbm", days=days,
            initial_cash=initial, fee=fee, slippage=slippage, optimize=False,
            triple_barrier=triple_barrier,
        )
    except Exception as exc:  # 样本不足/无数据都要如实记录，不能静默跳过
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    metrics = result["metrics"]
    row["stats"] = trade_stats(result["trades"], initial)
    row["metrics"] = {
        "total_return_pct": metrics["total_return_pct"],
        "max_drawdown_pct": metrics["max_drawdown_pct"],
        "random_baseline_return_pct": metrics["random_baseline"]["total_return_pct"],
        "buy_hold_return_pct": metrics["buy_hold_return_pct"],
        "exposure_pct": metrics.get("exposure_pct"),
        "avg_bars_held": metrics.get("avg_bars_held"),
        "signal_bars": metrics["signal_bars"],
        "blocked_signals": metrics["blocked_signals"],
        "test_rows": metrics["split"]["test_rows"],
        "significance": metrics.get("significance"),
    }
    row["beats_random"] = (
        metrics["total_return_pct"] > metrics["random_baseline"]["total_return_pct"]
    )

    # 零成本对照：同一段行情、同一信号，只把撮合成本清零
    try:
        zero = run_backtest(
            db, symbol, interval, strategy, algo="lightgbm", days=days,
            initial_cash=initial, fee=0.0, slippage=0.0, optimize=False,
            triple_barrier=triple_barrier,
        )
        row["zero_cost"] = trade_stats(zero["trades"], initial)
    except Exception as exc:
        row["zero_cost"] = {"error": f"{type(exc).__name__}: {exc}"}
    return row


def to_markdown(report: dict) -> str:
    lines: list[str] = []
    p = report["params"]
    lines.append("# 交易经济学诊断（优化方向定位）\n")
    lines.append(
        f"- as-of：{report['as_of']}；策略：{p['strategy']}；周期：{p['interval']}；"
        f"窗口：{p['days']} 天\n"
    )
    lines.append(
        f"- 初始资金 {p['initial']:.0f} USDT；fee={p['fee']}、slippage={p['slippage']}"
        f"（往返 {2 * (p['fee'] + p['slippage']) * 100:.3f}%）\n"
    )
    lines.append("")
    lines.append(
        "## 读法\n\n"
        "| 指标 | 高/低意味着什么 |\n|---|---|\n"
        "| 成本吸收率 = 费用/毛收益 | 接近或超过 0.7 → 成本主导，调参数天花板极低 |\n"
        "| 零成本收益 | 若也很低 → 问题在信号本身，不是成本 |\n"
        "| 毛胜率 vs 净胜率 | 差距大 → 大量交易卡在成本线上 |\n"
        "| beats_random | False → 方向判断无信息量，应淘汰而非调参 |\n"
    )
    lines.append("")
    lines.append("## 1. 数据宽度：库里有多少 vs 引擎能用多少\n")
    lines.append(
        "| 币种 | native 行数 | native 跨度(天) | 回落前可用(天) | **回落后可用(天)** | "
        "回落后行数 | 放大倍数 |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    for r in report["symbols"]:
        w = r.get("data_width", {})
        merged = w.get("merged", {})
        lines.append(
            f"| {r['symbol']} | {w.get('native', {}).get('rows', 0)} | "
            f"{w.get('native', {}).get('span_days', 0)} | {w.get('derived_span_days', 0)} | "
            f"**{merged.get('span_days', '-')}** | {merged.get('rows', '-')} | {w.get('gain_ratio')} |\n"
        )
    lines.append("")
    lines.append("## 2. 毛/净/成本拆解（真实成本）\n")
    lines.append(
        "| 币种 | 成交 | 毛收益% | 费用% | 净收益% | 成本吸收率 | 毛胜率 | 净胜率 | "
        "零成本净收益% | 平均仓位% | 平均杠杆 | 跑赢随机 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for r in report["symbols"]:
        s = r.get("stats", {})
        if not s.get("trades"):
            lines.append(f"| {r['symbol']} | 0 | - | - | - | - | - | - | - | - | - | {r.get('error','')[:30]} |\n")
            continue
        z = r.get("zero_cost", {})
        lines.append(
            f"| {r['symbol']} | {s['trades']} | {s['gross_return_pct']} | {s['cost_return_pct']} | "
            f"{s['net_return_pct']} | {s['cost_absorption']} | {s['gross_win_rate']} | {s['win_rate']} | "
            f"{z.get('net_return_pct', '-')} | {s['avg_margin_pct']} | {s['avg_leverage']} | "
            f"{'是' if r.get('beats_random') else '否'} |\n"
        )
    lines.append("")
    lines.append("## 3. 统计显著性：这个收益是真 edge 还是运气\n")
    lines.append(
        "| 币种 | 成交 | 每笔期望% | 95% CI 下界 | 95% CI 上界 | 判定 | IC | 随机基线均值% | 随机基线 5%~95% |\n"
        "|---|---:|---:|---:|---:|---|---:|---:|---:|\n"
    )
    for r in report["symbols"]:
        sig = (r.get("metrics") or {}).get("significance") or {}
        ci = sig.get("expectancy_ci") or {}
        ic = sig.get("information_coefficient") or {}
        rd = sig.get("random_distribution") or {}
        if not ci:
            lines.append(f"| {r['symbol']} | - | - | - | - | {r.get('error','无结果')[:24]} | - | - | - |\n")
            continue
        verdict = {"positive": "正 edge", "negative": "显著为负",
                   "inconclusive": "**无法判定**"}.get(ci.get("verdict"), ci.get("verdict"))
        lines.append(
            f"| {r['symbol']} | {ci.get('trades')} | {ci.get('mean_return_pct')} | "
            f"{ci.get('ci_low_pct')} | {ci.get('ci_high_pct')} | {verdict} | "
            f"{ic.get('ic')} | {rd.get('mean_pct')} | {rd.get('p05_pct')}~{rd.get('p95_pct')} |\n"
        )
    lines.append("")
    lines.append("## 4. 信号阻断漏斗（候选信号被谁杀掉）\n")
    for r in report["symbols"]:
        m = r.get("metrics")
        if not m:
            continue
        blocked = m["blocked_signals"]
        if not blocked:
            continue
        top = "、".join(f"{k}×{v}" for k, v in list(blocked.items())[:5])
        lines.append(
            f"- **{r['symbol']}**：开仓 bar {m['signal_bars']}，测试 {m['test_rows']} 根，"
            f"持仓占比 {m.get('exposure_pct')}%，平均持有 {m.get('avg_bars_held')} 根；主要拦截：{top}\n"
        )
    lines.append("")
    lines.append("## 5. 结论口径\n")
    lines.append(
        "- 本报告只拆解已存历史，不写订单、不改系统状态，不构成盈利证明。\n"
        "- 零成本对照会同时改变 cost_filter 与仓位计算，仅用于量级判断，不可直接当作可实现的收益。\n"
        "- 技术信号仅为概率性描述，历史形态统计参考，不代表未来走势，不构成投资建议。\n"
    )
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="交易经济学诊断")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--strategy", default="ml_ensemble")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--initial", type=float, default=10000.0)
    parser.add_argument("--fee", type=float, default=0.0004)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--output", default="docs/edge-diagnostic.json")
    parser.add_argument("--markdown-output", default="docs/edge-diagnostic.md")
    parser.add_argument("--report-only", action="store_true", help="用已有 JSON 重渲染 Markdown，不重跑回测")
    parser.add_argument("--triple-barrier", action="store_true",
                        help="用 triple-barrier 三分类标签（与实盘 sl/tp/max_hold 对齐）替换 future_return>0 二分类")
    args = parser.parse_args()

    if args.report_only:
        with open(args.output, encoding="utf-8") as handle:
            report = json.load(handle)
        with open(args.markdown_output, "w", encoding="utf-8") as handle:
            handle.write(to_markdown(report))
        print(f"re-rendered: {args.markdown_output}")
        return

    engine = make_engine(read_only=True)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "params": vars(args) | {"interval": args.interval},
        "symbols": [],
    }
    with Session(engine) as db:
        for symbol in args.symbols:
            print(f"[diagnose] {symbol} ...", flush=True)
            report["symbols"].append(
                diagnose_one(
                    db, symbol, args.interval, args.strategy, args.days,
                    args.initial, args.fee, args.slippage,
                    triple_barrier=args.triple_barrier,
                )
            )
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with open(args.markdown_output, "w", encoding="utf-8") as handle:
        handle.write(to_markdown(report))
    print(f"written: {args.output} / {args.markdown_output}")


if __name__ == "__main__":
    main()
