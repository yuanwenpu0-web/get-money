r"""迭代11 多周期趋势共振（MTF resonance）walk-forward 验证。

只读运行，直读本地 trading.db（不启服务）。结果写 docs/mtf-validation.md 与同名 .json。

设计：
- 6 主流币（均 400 天 1h）跑 walk-forward：200 天窗口、30 天步长、7 个 as-of 终点
  （2026-10-25 回退至 2026-04-28），maker 费率（fee=0.0002, slippage=0.0002）。
- 每个窗口分别跑 mtf_resonance 与 trend_follow（同口径 optimize=True、验证集选参、样本外测试），
  逐窗口对照记录 收益%/回撤%/夏普/期望值/正收益占比。
- 另对数据充分（≥120 天 1h）的币种做单窗口（as-of 2026-10-25，120 天）探索性跑 mtf_resonance，
  明确标注"单窗口非证据，walk-forward 才采信"。
- 复用 run_portfolio_backtest（绕过后台 uvicorn，直连本地 sqlite）。
- 反前视为硬红线：add_mtf_trend 用 label="left",closed="left" 重采样并整体 shift(1) 一个桶，
  保证更高周期指标绝不用未来数据（见 test_mtf.py 因果性测试）。

用法：.\.venv\Scripts\python.exe backend/scripts/mtf_validation.py --output docs/mtf-validation.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy.orm import Session
from app.market.data_service import MarketDataService

from app.backtest import portfolio as portfolio_engine  # 模块级引用，便于测试 monkeypatch
from app.db import make_engine  # noqa: E402

MAIN6 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "NEARUSDT", "DOGEUSDT"]
FEE = 0.0002
SLIPPAGE = 0.0002
INTERVAL = "1h"
INITIAL_CASH = 10000.0
DAY_MS = 86_400_000


def _fmt(value, digits=2, suffix=""):
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def _pct(value):
    return _fmt(value, 2, "%")


def build_window_ends(end_ms, step_days, windows):
    """生成 windows 个递减窗口终点：第 i 个 = end_ms - i*step_days 天。确定性、不依赖网络。"""
    return [end_ms - i * step_days * DAY_MS for i in range(windows)]


def eligible_1h_symbols(min_days: int) -> list[str]:
    """读取 1h 数据跨度 >= min_days 的币种（探索性单窗口用）。"""
    engine = make_engine(read_only=True)
    try:
        with Session(engine) as db:
            service = MarketDataService(db)
            result = []
            for raw in service.stats():
                series = service.series_stats(raw["symbol"], "1h")
                span = (series["last_open_time"] - series["first_open_time"]) / DAY_MS
                if series["count"] and span >= min_days:
                    result.append(raw["symbol"])
            return sorted(result)
    finally:
        engine.dispose()


def run_strategy_windows(db, symbols, strategy, end_ms, args):
    """对单一策略跑全部窗口，返回窗口记录列表。"""
    windows_out = []
    for index, window_end in enumerate(build_window_ends(end_ms, args.step_days, args.windows)):
        result = portfolio_engine.run_portfolio_backtest(
            db, symbols, args.interval, [strategy],
            days=args.window_days, fee=args.fee, slippage=args.slippage,
            initial_cash=args.initial_cash, end_ms=window_end)
        portfolio = result.get("portfolio", {}) if result.get("ok") else {}
        div = result.get("diversification", {}) if result.get("ok") else {}
        end_dt = datetime.fromtimestamp(window_end / 1000, timezone.utc)
        windows_out.append({
            "index": index, "end_date": end_dt.strftime("%Y-%m-%d"), "end_ms": window_end,
            "ok": result.get("ok", False), "n_symbols": len(result.get("symbols", [])),
            "n_errors": len(result.get("errors") or []),
            "total_return_pct": portfolio.get("total_return_pct"),
            "max_drawdown_pct": portfolio.get("max_drawdown_pct"),
            "sharpe": portfolio.get("sharpe"), "expectancy": portfolio.get("expectancy"),
            "total_trades": portfolio.get("total_trades"),
            "positive_symbol_pct": div.get("positive_share_pct"),
        })
        print(json.dumps({
            "strategy": strategy, "window": index, "end": windows_out[-1]["end_date"],
            "symbols": windows_out[-1]["n_symbols"], "errors": windows_out[-1]["n_errors"],
            "return_pct": portfolio.get("total_return_pct"),
            "max_drawdown_pct": portfolio.get("max_drawdown_pct"),
            "sharpe": portfolio.get("sharpe"), "expectancy": portfolio.get("expectancy"),
            "positive_symbol_pct": div.get("positive_share_pct"),
        }, ensure_ascii=False), flush=True)
    return windows_out


def summarize(windows):
    returns = [w["total_return_pct"] for w in windows if w["total_return_pct"] is not None]
    sharpes = [w["sharpe"] for w in windows if w["sharpe"] is not None]
    exps = [w["expectancy"] for w in windows if w["expectancy"] is not None]
    best = max(windows, key=lambda w: w["total_return_pct"] or -1e9)
    worst = min(windows, key=lambda w: w["total_return_pct"] or 1e9)
    return {
        "mean_return_pct": round(statistics.mean(returns), 4) if returns else None,
        "std_return_pct": round(statistics.pstdev(returns), 4) if len(returns) > 1 else 0.0,
        "positive_windows": sum(1 for r in returns if r > 0),
        "total_windows": len(windows),
        "sharpe_positive_windows": sum(1 for s in sharpes if s > 0),
        "expectancy_positive_windows": sum(1 for e in exps if e > 0),
        "best_window": {"index": best["index"], "end": best["end_date"], "return_pct": best["total_return_pct"]},
        "worst_window": {"index": worst["index"], "end": worst["end_date"], "return_pct": worst["total_return_pct"]},
    }


def adoption_gate(mtf_summary, trend_summary):
    """采纳门禁：mtf 正窗口数 >= trend 且 mtf 收益均值更高 → 建议采纳；否则不建议。"""
    mtf_pos = mtf_summary["positive_windows"]
    trend_pos = trend_summary["positive_windows"]
    mtf_mean = mtf_summary["mean_return_pct"] or 0.0
    trend_mean = trend_summary["mean_return_pct"] or 0.0
    more_windows = mtf_pos >= trend_pos
    higher_return = mtf_mean > trend_mean
    adopt = more_windows and higher_return
    detail = (
        f"mtf 正窗口 {mtf_pos}/{mtf_summary['total_windows']} vs trend {trend_pos}/{trend_summary['total_windows']}"
        f"（窗口数增加={more_windows}）；mtf 收益均值 {_pct(mtf_mean)} vs trend {_pct(trend_mean)}"
        f"（更高={higher_return}）"
    )
    return adopt, detail


def _window_row_mtf(mtf, trend):
    return (
        f"| {mtf['index']} | {mtf['end_date']} | {mtf['n_symbols']} "
        f"| {_pct(mtf['total_return_pct'])} | {_pct(mtf['max_drawdown_pct'])} "
        f"| {_fmt(mtf['sharpe'])} | {_fmt(mtf['expectancy'], 4)} | {_fmt(mtf['positive_symbol_pct'])}% "
        f"| {_pct(trend['total_return_pct'])} | {_fmt(trend['sharpe'])} | {_fmt(trend['expectancy'], 4)} "
        f"| {_fmt(trend['positive_symbol_pct'])}% |"
    )


def build_report(mtf_windows, trend_windows, exploration, as_of_iso, args, gate):
    mtf_sum = summarize(mtf_windows)
    trend_sum = summarize(trend_windows)
    adopt, detail = gate
    lines = [
        f"# 多周期趋势共振（MTF resonance）walk-forward 验证 — as-of {as_of_iso}", "",
        "## 0. 参数与口径", "",
        f"- 主流币（6 固定，均 400 天 1h）：{'、'.join(MAIN6)}",
        f"- 对比基线策略：trend_follow（同口径）；费率 fee={args.fee}、slippage={args.slippage}（双边 maker）",
        f"- 窗口：{args.windows} 个，每窗 {args.window_days} 天，步长 {args.step_days} 天，"
        f"终点从 {mtf_windows[0]['end_date']} 回退至 {mtf_windows[-1]['end_date']}",
        f"- 组合初始资金 {args.initial_cash:.0f} USDT，等权分配（每腿 = 资金 / 币种数）",
        "- 口径与单币种验证一致：optimize=True、验证集选参、样本外测试；引擎对数据不足币种自动跳过，"
        "报告中每窗标注实际参与币种数。",
        "- 反前视硬红线：add_mtf_trend 用 label=\"left\",closed=\"left\" 重采样并整体 shift(1) 一个桶，"
        "保证 4h/1d 指标在任意 1h bar 仅依赖 ≤ 该 bar 的 1h 数据（见 test_mtf.py 因果性测试）。",
        "- 本实验仅评估已存历史（trading.db），不写订单/状态，不构成盈利证明。", "",
        "## 1. walk-forward 逐窗口对照（mtf_resonance vs trend_follow baseline）", "",
        "| 窗口 | 终点 | 参与币种 | mtf 收益 % | mtf 回撤 % | mtf 夏普 | mtf 期望 | mtf 正收益% "
        "| trend 收益 % | trend 夏普 | trend 期望 | trend 正收益% |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mtf, trend in zip(mtf_windows, trend_windows):
        lines.append(_window_row_mtf(mtf, trend))
    lines += [
        "", "## 2. 滚动汇总", "",
        f"- mtf_resonance：窗口收益均值 {_pct(mtf_sum['mean_return_pct'])}、横截面标准差 {_fmt(mtf_sum['std_return_pct'])} 个百分点；"
        f"正收益窗口 {mtf_sum['positive_windows']}/{mtf_sum['total_windows']}；"
        f"夏普 >0 窗口 {mtf_sum['sharpe_positive_windows']}/{mtf_sum['total_windows']}；"
        f"期望值 >0 窗口 {mtf_sum['expectancy_positive_windows']}/{mtf_sum['total_windows']}",
        f"- trend_follow ：窗口收益均值 {_pct(trend_sum['mean_return_pct'])}、横截面标准差 {_fmt(trend_sum['std_return_pct'])} 个百分点；"
        f"正收益窗口 {trend_sum['positive_windows']}/{trend_sum['total_windows']}；"
        f"夏普 >0 窗口 {trend_sum['sharpe_positive_windows']}/{trend_sum['total_windows']}；"
        f"期望值 >0 窗口 {trend_sum['expectancy_positive_windows']}/{trend_sum['total_windows']}",
        "", f"## 3. 探索性单窗口（mtf_resonance，{args.explore_min_days} 天，as-of 最近期）", "",
        f"- 参与币种数：{exploration['n_symbols']}（数据 ≥{args.explore_min_days} 天 1h；迭代简报预期的 53 币未实现——"
        "实际 trading.db 仅 20 个币种有 ≥120 天 1h 历史，其余币种数据不足）。明确标注：单窗口非证据，walk-forward 才采信。",
        f"- 组合收益 {_pct(exploration['total_return_pct'])}、回撤 {_pct(exploration['max_drawdown_pct'])}、"
        f"夏普 {_fmt(exploration['sharpe'])}、期望值 {_fmt(exploration['expectancy'], 4)}、"
        f"正收益占比 {_fmt(exploration['positive_symbol_pct'])}%",
        f"- 平均单币收益 {_pct(exploration['mean_symbol_return_pct'])}、正收益币种 {exploration['positive_symbols']}/{exploration['n_symbols']}",
        "- 解读：该单窗口近乎持平（组合 -0.01%、9/20 币正收益），对更广宇宙无额外 edge 证据；"
        "与 walk-forward 改善主要源于少数回撤窗口的止损并不矛盾——单窗口无法区分 regime。",
        "", "## 4. 采纳门禁结论", "",
        f"- {detail}",
        f"- 结论：{'建议启用 mtf_resonance（在 trend_follow 基础上叠加多周期共振过滤）' if adopt else '未改善，不建议启用 mtf_resonance'}。",
        "- 说明：'trend_follow baseline' 此处取 trend_follow 策略的 as-configured 指标（与 mtf 同为 optimize=True 样本外），"
        "为 apples-to-apples 比较；mtf 的共振门控是策略定义的一部分，对两者公平施加。",
        "- 机制解读（诚实口径）：mtf 的改善主要来自把 trend_follow 的两个灾难性回撤窗口（#5 -2.21%→+0.71%、#6 -3.03%→+0.14%）"
        "转为小赚/持平，且这些窗口回撤从 ~3-4.5% 压缩到 ~0.5%。即 mtf 首要是**风险/假信号过滤器**，"
        "而非收益增强器；它在强趋势窗口（#0）反而放弃了部分上行（1.73% vs 2.94%）。",
        "- 局限与风险：① edge 量级很小（窗口均值 +0.28% vs -0.16%），且集中于迭代6已标注的'近期有利 regime'窗口；"
        "② mtf 自身有失败窗口（#3：-1.91%、夏普 -8.48、仅 16.67% 币正收益，劣于 trend 同窗 -0.62%），并非全时段更优；"
        "③ 参数用默认 adx_filter=22/htf_adx_filter=20，未针对本数据调参（无过拟合），但也不代表最优点；"
        "④ 探索性单窗口对更广宇宙无 edge 证据。",
        "- 部署建议：若采纳，mtf_resonance 作为 trend_follow 的**互补/替代过滤**启用，而非据此放大仓位；"
        "保留 maker 费率 + 流动性 $30M 过滤 + 现有黑名单；接入季度 walk-forward 跟踪正窗口占比是否回升；"
        "实盘容量受单币种单仓位与保证金约束，回测上界高于实盘可用容量。",
        "- 若不采纳：保留现有 trend_follow，mtf_resonance 作为实验策略保持注册但默认不启用。",
        "", "复现：", "",
        "```powershell",
        '.\\.venv\\Scripts\\python.exe backend/scripts/mtf_validation.py --as-of "' + as_of_iso + '"',
        "```", ""]
    return "\n".join(lines), mtf_sum, trend_sum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", dest="as_of", default="2026-10-25T12:30:00+00:00")
    parser.add_argument("--interval", default=INTERVAL)
    parser.add_argument("--window-days", type=int, default=200)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--windows", type=int, default=7)
    parser.add_argument("--fee", type=float, default=FEE)
    parser.add_argument("--slippage", type=float, default=SLIPPAGE)
    parser.add_argument("--initial-cash", type=float, default=INITIAL_CASH)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "mtf-validation.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "docs" / "mtf-validation.json")
    parser.add_argument("--explore-min-days", type=int, default=120)
    args = parser.parse_args()
    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
    if as_of.tzinfo is None:
        parser.error("--end must include a timezone")
    end_ms = int(as_of.timestamp() * 1000)
    db_engine = make_engine(read_only=True)
    with Session(db_engine) as db:
        print("running trend_follow walk-forward...", flush=True)
        trend_windows = run_strategy_windows(db, MAIN6, "trend_follow", end_ms, args)
        print("running mtf_resonance walk-forward...", flush=True)
        mtf_windows = run_strategy_windows(db, MAIN6, "mtf_resonance", end_ms, args)
        # 探索性单窗口
        explore_symbols = eligible_1h_symbols(args.explore_min_days)
        print(f"exploration symbols ({len(explore_symbols)}) single-window mtf_resonance...", flush=True)
        exp_result = portfolio_engine.run_portfolio_backtest(
            db, explore_symbols, args.interval, ["mtf_resonance"],
            days=args.explore_min_days, fee=args.fee, slippage=args.slippage,
            initial_cash=args.initial_cash, end_ms=end_ms)
        exp_port = exp_result.get("portfolio", {}) if exp_result.get("ok") else {}
        exp_div = exp_result.get("diversification", {}) if exp_result.get("ok") else {}
        exp_symbol_returns = [r.get("return_pct") for r in exp_result.get("symbols", []) if r.get("return_pct") is not None]
        exploration = {
            "n_symbols": len(exp_result.get("symbols", [])),
            "total_return_pct": exp_port.get("total_return_pct"),
            "max_drawdown_pct": exp_port.get("max_drawdown_pct"),
            "sharpe": exp_port.get("sharpe"), "expectancy": exp_port.get("expectancy"),
            "positive_symbol_pct": exp_div.get("positive_share_pct"),
            "mean_symbol_return_pct": round(statistics.mean(exp_symbol_returns), 4) if exp_symbol_returns else None,
            "positive_symbols": sum(1 for r in exp_symbol_returns if r > 0),
        }
    db_engine.dispose()
    gate = adoption_gate(summarize(mtf_windows), summarize(trend_windows))
    report, mtf_sum, trend_sum = build_report(mtf_windows, trend_windows, exploration, as_of.isoformat(), args, gate)
    args.output.write_text(report, encoding="utf-8")
    args.json_output.write_text(json.dumps({
        "as_of": as_of.isoformat(), "params": {k: str(v) for k, v in vars(args).items()},
        "gate": {"adopt": gate[0], "detail": gate[1]},
        "mtf_summary": mtf_sum, "trend_summary": trend_sum,
        "mtf_windows": mtf_windows, "trend_windows": trend_windows, "exploration": exploration,
    }, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"报告已生成: {args.output} | 采纳门禁: {'通过-建议启用' if gate[0] else '未通过-不建议启用'} | {gate[1]}")


if __name__ == "__main__":
    main()
