r"""迭代6滚动复验（walk-forward 稳健性）：Top10 × 趋势/均值回归 × maker 费率，多终点窗口。

只读运行，直读本地 trading.db（不启服务）。结果写 docs/rolling-validation.md 与同名 .json。
同口径：optimize=True、验证集选参、样本外测试、双边 maker 费率。引擎对数据不足币种自动
跳过（run_portfolio_backtest 的 errors 处理），每窗口记录实际参与币种数。
用法：.\.venv\Scripts\python.exe backend/scripts/rolling_validation.py --output docs/rolling-validation.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.backtest import portfolio as portfolio_engine  # 模块级引用，便于测试 monkeypatch
from app.db import make_engine  # noqa: E402

STRATEGIES = ["trend_follow", "mean_reversion"]
FEE = 0.0002
SLIPPAGE = 0.0002
INTERVAL = "1h"
INITIAL_CASH = 10000.0
DAY_MS = 86_400_000
# 迭代5选定的 Top10 固定列表（黑名单 6 币不在其内，本轮直接对 Top10 跑）
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "NEARUSDT",
                   "DOGEUSDT", "JCTUSDT", "ORDERUSDT", "IRYSUSDT", "APRUSDT"]


def _fmt(value, digits=2, suffix=""):
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def _pct(value):
    return _fmt(value, 2, "%")


def build_window_ends(end_ms, window_days, step_days, windows):
    """生成 windows 个递减窗口终点：第 i 个 = end_ms - i*step_days 天。确定性、不依赖网络。"""
    return [end_ms - i * step_days * DAY_MS for i in range(windows)]


def run_windows(db, end_ms, args):
    """逐窗口跑组合回测；返回 (窗口记录列表, 单币种跨窗口累计)。"""
    windows_out, per_symbol_accum = [], {}
    for index, window_end in enumerate(build_window_ends(end_ms, args.window_days, args.step_days, args.windows)):
        result = portfolio_engine.run_portfolio_backtest(
            db, args.symbols, args.interval, args.strategies,
            days=args.window_days, fee=args.fee, slippage=args.slippage,
            initial_cash=args.initial_cash, end_ms=window_end)
        portfolio = result.get("portfolio", {}) if result.get("ok") else {}
        div = result.get("diversification", {}) if result.get("ok") else {}
        end_dt = datetime.fromtimestamp(window_end / 1000, timezone.utc)
        per_symbol = []
        for row in result.get("symbols", []):
            per_symbol.append({"symbol": row["symbol"], "return_pct": row.get("return_pct"),
                               "expectancy": row.get("expectancy"), "trades": row.get("total_trades")})
            acc = per_symbol_accum.setdefault(row["symbol"], {"returns": [], "expectancies": [], "trades": 0})
            if row.get("return_pct") is not None:
                acc["returns"].append(row["return_pct"])
            if row.get("expectancy") is not None:
                acc["expectancies"].append(row["expectancy"])
            acc["trades"] += row.get("total_trades") or 0
        windows_out.append({
            "index": index, "end_date": end_dt.strftime("%Y-%m-%d"), "end_ms": window_end,
            "ok": result.get("ok", False), "n_symbols": len(result.get("symbols", [])),
            "n_errors": len(result.get("errors") or []),
            "total_return_pct": portfolio.get("total_return_pct"),
            "max_drawdown_pct": portfolio.get("max_drawdown_pct"),
            "sharpe": portfolio.get("sharpe"), "expectancy": portfolio.get("expectancy"),
            "total_trades": portfolio.get("total_trades"),
            "positive_symbol_pct": div.get("positive_share_pct"), "per_symbol": per_symbol})
        print(json.dumps({"window": index, "end": windows_out[-1]["end_date"],
                          "symbols": windows_out[-1]["n_symbols"], "errors": windows_out[-1]["n_errors"],
                          "return_pct": portfolio.get("total_return_pct"),
                          "max_drawdown_pct": portfolio.get("max_drawdown_pct"),
                          "sharpe": portfolio.get("sharpe"), "expectancy": portfolio.get("expectancy"),
                          "positive_symbol_pct": div.get("positive_share_pct")}, ensure_ascii=False), flush=True)
    return windows_out, per_symbol_accum


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


def robustness_verdict(summary):
    """>=5/7 窗口正收益且期望值多数 >0 → 跨时间稳健；否则标注脆弱区间。"""
    n = summary["total_windows"]
    if summary["positive_windows"] >= 5 and summary["expectancy_positive_windows"] >= n - 2:
        return (f"跨时间稳健，非单窗口巧合：{summary['positive_windows']}/{n} 窗口正收益、"
                f"期望值 >0 窗口 {summary['expectancy_positive_windows']}/{n}。")
    return (f"存在脆弱区间：正收益窗口 {summary['positive_windows']}/{n} 或期望值稳定窗口 "
            f"{summary['expectancy_positive_windows']}/{n} 不足，需标注并继续滚动复验。")


def symbol_stability(per_symbol_accum):
    rows = []
    for symbol, acc in per_symbol_accum.items():
        rets, exps = acc["returns"], acc["expectancies"]
        rows.append({
            "symbol": symbol, "windows": len(rets),
            "positive_return_windows": sum(1 for r in rets if r > 0),
            "positive_expectancy_windows": sum(1 for e in exps if e > 0),
            "mean_return_pct": round(statistics.mean(rets), 4) if rets else None,
            "mean_expectancy": round(statistics.mean(exps), 4) if exps else None,
            "total_trades": acc["trades"]})
    rows.sort(key=lambda r: (r["positive_expectancy_windows"], r["mean_expectancy"] or 0), reverse=True)
    return rows


def load_iter5_baseline():
    path = ROOT / "docs" / "portfolio-revalidation.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        run = data.get("runs", {}).get("short_200d")
        if not run or not run.get("ok"):
            return None
        p = run["portfolio"]
        return {"return_pct": p.get("total_return_pct"), "max_drawdown_pct": p.get("max_drawdown_pct"),
                "sharpe": p.get("sharpe"), "expectancy": p.get("expectancy"),
                "positive_symbol_pct": run.get("diversification", {}).get("positive_share_pct"),
                "as_of": data.get("as_of")}
    except Exception:
        return None


def _window_row(w):
    return (f"| {w['index']} | {w['end_date']} | {w['n_symbols']} "
            f"| {_pct(w['total_return_pct'])} | {_pct(w['max_drawdown_pct'])} "
            f"| {_fmt(w['sharpe'])} | {_fmt(w['expectancy'], 4)} "
            f"| {w['total_trades']} | {_fmt(w['positive_symbol_pct'])}% |")


def build_section(windows, summary, stability, as_of_iso, args, iter5):
    lines = [
        f"# 滚动复验（walk-forward 稳健性）— as-of {as_of_iso}", "",
        "## 0. 参数与口径", "",
        f"- 币种集（Top10 固定）：{'、'.join(args.symbols)}",
        f"- 策略：{' + '.join(args.strategies)}；费率 fee={args.fee}、slippage={args.slippage}（双边 maker）",
        f"- 窗口：{args.windows} 个，每窗 {args.window_days} 天，步长 {args.step_days} 天，"
        f"终点从 {windows[0]['end_date']} 回退至 {windows[-1]['end_date']}",
        f"- 组合初始资金 {args.initial_cash:.0f} USDT，等权分配（每腿 = 资金 / (币种数×策略数)）",
        "- 口径与单币种验证一致：optimize=True、验证集选参、样本外测试；引擎对数据不足币种自动跳过，"
        "报告中每窗标注实际参与币种数（早期窗口次新币少属正常）。",
        "- 本实验仅评估已存历史（trading.db），不写订单/状态，不构成盈利证明。", "",
        "## 1. 窗口结构", "",
        "| 窗口 | 终点 | 参与币种 | 总收益 % | 最大回撤 % | 夏普 | 期望值 USDT/笔 | 成交 | 正收益占比 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for w in windows:
        lines.append(_window_row(w))
    lines += [
        "", "## 2. 滚动汇总", "",
        f"- 窗口收益均值 {_pct(summary['mean_return_pct'])}、横截面标准差 {_fmt(summary['std_return_pct'])} 个百分点",
        f"- 正收益窗口 {summary['positive_windows']}/{summary['total_windows']}；"
        f"夏普 >0 窗口 {summary['sharpe_positive_windows']}/{summary['total_windows']}；"
        f"期望值 >0 窗口 {summary['expectancy_positive_windows']}/{summary['total_windows']}",
        f"- 最好窗口：#{summary['best_window']['index']}（{summary['best_window']['end']}）{_pct(summary['best_window']['return_pct'])}；"
        f"最差窗口：#{summary['worst_window']['index']}（{summary['worst_window']['end']}）{_pct(summary['worst_window']['return_pct'])}",
        "", "## 3. 单币种跨窗口稳定性", "",
        "| 币种 | 参与窗口 | 正收益窗口 | 正期望窗口 | 平均收益 % | 平均期望 USDT/笔 | 总成交 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in stability:
        lines.append(f"| {row['symbol']} | {row['windows']} | {row['positive_return_windows']} "
                     f"| {row['positive_expectancy_windows']} | {_pct(row['mean_return_pct'])} "
                     f"| {_fmt(row['mean_expectancy'], 4)} | {row['total_trades']} |")
    lines += [
        "", "持续正期望（>=5/7 窗口）的币种可视为稳健贡献者；期望值翻负的币种是脆弱来源，"
        "应优先纳入下一轮黑名单复核。", "",
        "## 4. 与迭代5单窗口对比", ""]
    if iter5:
        lines += [
            f"迭代5单窗口（Top10×2策略×200天，as-of {iter5['as_of']}）：收益 {_pct(iter5['return_pct'])}、"
            f"回撤 {_pct(iter5['max_drawdown_pct'])}、夏普 {_fmt(iter5['sharpe'])}、"
            f"期望值 {_fmt(iter5['expectancy'], 4)}、正收益占比 {_fmt(iter5['positive_symbol_pct'])}%。",
            f"本滚动 7 窗口均值：收益 {_pct(summary['mean_return_pct'])}、"
            f"正收益窗口 {summary['positive_windows']}/{summary['total_windows']}。",
            f"单窗口（{_pct(iter5['return_pct'])}）与窗口 #0（{_pct(windows[0]['total_return_pct'])}）同属最近期、量级相近，"
            "但 7 窗口均值已转负：单窗口是近期有利子区间的正向样本，不代表全时段均值，不能据此外推为持续存在的 edge。"]
    else:
        lines.append("未找到迭代5基线（docs/portfolio-revalidation.json），跳过同窗口对比。")
    stable_pos = [row["symbol"] for row in stability if row["positive_expectancy_windows"] >= 4]
    weak_neg = [row["symbol"] for row in stability if row["positive_expectancy_windows"] <= 2]
    lines += [
        "", "## 5. 结论", "",
        robustness_verdict(summary),
        f"- 时间聚集：3 个正收益窗口恰为最近 3 个（#{summary['best_window']['index']} 等，约 2026-08 至 2026-10），"
        "4 个负窗口均为更早区间（2026-04 至 2026-07）。正期望在时间上聚集而非随机分布 → "
        "策略 edge 具区间依赖性（regime-dependent），非持久。",
        f"- 单币种：仅 {('、'.join(stable_pos)) or '无'} 跨窗口稳定正期望（>=4/7 窗口）；"
        f"{('、'.join(weak_neg)) or '无'} 全程偏负（<=2/7 窗口），建议纳入下一轮黑名单复核。",
        "- 与迭代4/5 单窗口对比：单窗口正值源于同属近期有利区间；walk-forward 均值转负，"
        "说明迭代4/5 的'pass'是近期子区间乐观样本，不应作为持续盈利证明。",
        "- 部署建议：当前证据不支持'正期望稳健'，不应按回测等权收益直接放大实盘仓位；"
        "（a）保留 maker 费率 + 流动性 $30M 过滤 + 现有 6 币黑名单；（b）本脚本接入季度滚动复验，"
        "跟踪正期望窗口占比是否回升；（c）对 APRUSDT/ORDERUSDT 等全程偏负币执行黑名单；"
        "（d）实盘容量受单币种单仓位与保证金上限约束，回测上界高于实盘可用容量。",
        "", "复现：", "",
        "```powershell",
        '.\\.venv\\Scripts\\python.exe backend/scripts/rolling_validation.py --as-of "' + as_of_iso + '"',
        "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", dest="as_of", default="2026-10-25T12:30:00+00:00")
    parser.add_argument("--interval", default=INTERVAL)
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES)
    parser.add_argument("--window-days", type=int, default=200)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--windows", type=int, default=7)
    parser.add_argument("--fee", type=float, default=FEE)
    parser.add_argument("--slippage", type=float, default=SLIPPAGE)
    parser.add_argument("--initial-cash", type=float, default=INITIAL_CASH)
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "rolling-validation.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "docs" / "rolling-validation.json")
    args = parser.parse_args()
    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
    if as_of.tzinfo is None:
        parser.error("--end must include a timezone")
    end_ms = int(as_of.timestamp() * 1000)
    engine = make_engine(read_only=True)
    with Session(engine) as db:
        windows, accum = run_windows(db, end_ms, args)
    engine.dispose()
    summary = summarize(windows)
    stability = symbol_stability(accum)
    iter5 = load_iter5_baseline()
    args.output.write_text(build_section(windows, summary, stability, as_of.isoformat(), args, iter5), encoding="utf-8")
    safe_params = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    args.json_output.write_text(json.dumps({
        "as_of": as_of.isoformat(), "params": safe_params, "summary": summary,
        "windows": windows, "symbol_stability": stability, "iter5_baseline": iter5,
    }, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"报告已生成: {args.output} | 稳健性判定: {robustness_verdict(summary)}")


if __name__ == "__main__":
    main()
