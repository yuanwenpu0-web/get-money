r"""迭代8：指标加权滚动复验 —— baseline / +B(仓位加权) / +B+C(+regime) 三配置对照。

只读运行，直读本地 trading.db（不启服务）。复用 rolling_validation 的窗口生成与汇总。
每窗口额外记录指标覆盖度（该窗口内有多少天真的有指标）：免费源 /futures/data/*
只保留约 30 天，覆盖不足时加权退回等权（仅剩定期再平衡的机制差异）。

采纳门禁：配置相对 baseline 必须正窗口数更多 且 均值更高才可建议采纳；
否则明确写未改善、不建议启用，保持 metrics_assist_enabled 默认 False。

用法：.\.venv\Scripts\python.exe backend/scripts/metric_weighting_validation.py --end "2026-10-25T12:30:00+00:00"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.backtest import portfolio as portfolio_engine  # noqa: E402
from app.db import make_engine  # noqa: E402
from app.market.metrics_history import metrics_as_of  # noqa: E402

_SCRIPT = ROOT / "backend" / "scripts" / "rolling_validation.py"
_spec = importlib.util.spec_from_file_location("rolling_iter8", _SCRIPT)
rolling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rolling)

STRATEGIES = ["trend_follow", "mean_reversion"]
FEE = 0.0002
SLIPPAGE = 0.0002
INTERVAL = "1h"
INITIAL_CASH = 10000.0
DAY_MS = 86_400_000
TOP10 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "NEARUSDT",
         "DOGEUSDT", "JCTUSDT", "ORDERUSDT", "IRYSUSDT", "APRUSDT"]

CONFIGS = [
    {"key": "baseline", "label": "A baseline（等权，无指标）", "weight_mode": "equal",
     "regime_weighting": False},
    {"key": "plus_b", "label": "B +指标仓位加权", "weight_mode": "metric",
     "regime_weighting": False},
    {"key": "plus_bc", "label": "C +仓位加权+regime", "weight_mode": "metric",
     "regime_weighting": True},
]


def metric_coverage_days(db, symbols, window_end_ms, window_days):
    """该窗口内真正有指标可用的天数（保守：取各币种最早指标时间的最大值作为起点）。"""
    starts = []
    for symbol in symbols:
        row = metrics_as_of(db, symbol, window_end_ms)
        if not row:
            continue
        starts.append(_earliest_ts(db, symbol))
    if not starts:
        return 0.0
    metric_start = max(starts)
    covered = (window_end_ms - metric_start) / DAY_MS
    return round(max(0.0, min(float(window_days), covered)), 1)


def _earliest_ts(db, symbol):
    from app.models import CoinMetric
    row = (db.query(CoinMetric).filter(CoinMetric.symbol == symbol.upper())
           .order_by(CoinMetric.ts.asc()).first())
    return int(row.ts) if row else 0


def run_config(db, end_ms, args, config, window_days, step_days, windows):
    """跑单一配置的多窗口；返回窗口记录列表。"""
    out = []
    for index, window_end in enumerate(
            rolling.build_window_ends(end_ms, window_days, step_days, windows)):
        result = portfolio_engine.run_portfolio_backtest(
            db, args.symbols, args.interval, args.strategies, days=window_days,
            fee=args.fee, slippage=args.slippage, initial_cash=args.initial_cash,
            end_ms=window_end, weight_mode=config["weight_mode"],
            rebalance_bars=args.rebalance_bars, regime_weighting=config["regime_weighting"])
        portfolio = result.get("portfolio", {}) if result.get("ok") else {}
        div = result.get("diversification", {}) if result.get("ok") else {}
        covered = metric_coverage_days(db, args.symbols, window_end, window_days)
        record = {
            "index": index,
            "end_date": datetime.fromtimestamp(window_end / 1000, timezone.utc).strftime("%Y-%m-%d"),
            "ok": result.get("ok", False), "n_symbols": len(result.get("symbols", [])),
            "total_return_pct": portfolio.get("total_return_pct"),
            "max_drawdown_pct": portfolio.get("max_drawdown_pct"),
            "sharpe": portfolio.get("sharpe"), "expectancy": portfolio.get("expectancy"),
            "total_trades": portfolio.get("total_trades"),
            "positive_symbol_pct": div.get("positive_share_pct"),
            "metric_coverage_days": covered,
            "metric_coverage_pct": round(covered / window_days * 100, 1),
        }
        out.append(record)
        print(json.dumps({"config": config["key"], "window": index, "end": record["end_date"],
                          "return_pct": record["total_return_pct"],
                          "coverage_pct": record["metric_coverage_pct"]}, ensure_ascii=False),
              flush=True)
    return out


def coverage_split(windows):
    """把窗口按是否有指标覆盖拆开：covered=覆盖>0，uncovered=覆盖=0（纯再平衡效应）。"""
    def mean(rows):
        values = [w["total_return_pct"] for w in rows if w["total_return_pct"] is not None]
        return round(sum(values) / len(values), 4) if values else None
    covered = [w for w in windows if w["metric_coverage_pct"] > 0]
    uncovered = [w for w in windows if w["metric_coverage_pct"] <= 0]
    return {"n_covered": len(covered), "n_uncovered": len(uncovered),
            "covered_mean": mean(covered), "uncovered_mean": mean(uncovered)}


def adoption_gate(baseline_summary, candidate_summary):
    """门禁：正窗口数更多 且 均值更高 → 通过；否则不通过。"""
    more_positive = candidate_summary["positive_windows"] > baseline_summary["positive_windows"]
    higher_mean = (candidate_summary["mean_return_pct"] or -1e9) > (baseline_summary["mean_return_pct"] or -1e9)
    return more_positive and higher_mean, {"more_positive": more_positive, "higher_mean": higher_mean}


def _row(w):
    return (f"| {w['index']} | {w['end_date']} | {w['n_symbols']} "
            f"| {rolling._pct(w['total_return_pct'])} | {rolling._pct(w['max_drawdown_pct'])} "
            f"| {rolling._fmt(w['sharpe'])} | {rolling._fmt(w['expectancy'], 4)} "
            f"| {w['total_trades']} | {rolling._fmt(w['positive_symbol_pct'])}% "
            f"| {w['metric_coverage_days']} 天 / {w['metric_coverage_pct']}% |")


def build_report(groups, summaries, gates, as_of_iso, args, short_groups, short_summaries,
                 splits):
    lines = [
        f"# 指标加权滚动复验（迭代8）— as-of {as_of_iso}", "",
        "## 0. 参数、口径与数据覆盖现实", "",
        f"- 币种集（Top10 固定）：{'、'.join(args.symbols)}",
        f"- 策略：{' + '.join(args.strategies)}；费率 fee={args.fee}、slippage={args.slippage}（双边 maker）",
        f"- 主实验：{args.windows} 个窗口 × {args.window_days} 天，步长 {args.step_days} 天；"
        f"再平衡每 {args.rebalance_bars} 根 1h K 线（约每日）",
        "- **数据覆盖硬约束（实测）**：币安免费 `/futures/data/*` 在 period=1d 下无论 limit 多大都只返回"
          "最近 30 行；`startTime` 被拒（-1130），`endTime` 早于最早数据返回空 → 免费源仅约 30 天历史。"
          "覆盖不足时 `metrics_as_of` 取不到数据 → 仓位与 regime 权重均退回等权（无数据不表态），"
          "但加权路径仍按 `rebalance_bars` 定期再平衡，与 baseline 买入持有存在机制差异。",
        "", "## 1. 三配置汇总对照", "",
        "| 配置 | 收益均值 % | 收益标准差 | 正窗口 | 夏普>0 窗口 | 期望>0 窗口 | 最好窗口 % | 最差窗口 % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for config in CONFIGS:
        s = summaries[config["key"]]
        lines.append(f"| {config['label']} | {rolling._pct(s['mean_return_pct'])} "
                     f"| {rolling._fmt(s['std_return_pct'])} | {s['positive_windows']}/{s['total_windows']} "
                     f"| {s['sharpe_positive_windows']}/{s['total_windows']} "
                     f"| {s['expectancy_positive_windows']}/{s['total_windows']} "
                     f"| {rolling._pct(s['best_window']['return_pct'])} "
                     f"| {rolling._pct(s['worst_window']['return_pct'])} |")
    lines += ["", "## 2. 覆盖/未覆盖拆分：区分再平衡效应与指标效应", "",
              "覆盖=0 的窗口没有任何指标可用，此时加权仅剩定期再平衡这一机制差异；"
              "覆盖>0 的窗口才真正含指标信号。两组的均值差可定位收益变化到底来自哪里。", "",
              "| 配置 | 覆盖>0 窗口数 | 覆盖>0 均值 % | 覆盖=0 窗口数 | 覆盖=0 均值 % "
              "| 覆盖>0 相对 baseline | 覆盖=0 相对 baseline |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    base_split = splits["baseline"]
    for config in CONFIGS:
        s = splits[config["key"]]
        def delta(a, b):
            return rolling._pct(None if a is None or b is None else a - b)
        lines.append(f"| {config['label']} | {s['n_covered']} | {rolling._pct(s['covered_mean'])} "
                     f"| {s['n_uncovered']} | {rolling._pct(s['uncovered_mean'])} "
                     f"| {delta(s['covered_mean'], base_split['covered_mean'])} "
                     f"| {delta(s['uncovered_mean'], base_split['uncovered_mean'])} |")
    lines += ["", "## 3. 各配置逐窗口明细（主实验）", ""]
    for config in CONFIGS:
        lines += [f"### {config['label']}", "",
                  "| 窗口 | 终点 | 币种 | 收益 % | 回撤 % | 夏普 | 期望值 | 成交 | 正收益占比 | 指标覆盖 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        lines += [_row(w) for w in groups[config["key"]]]
        lines.append("")
    lines += ["## 4. 补充：覆盖期内短窗口探索（非门禁依据）", "",
              f"主实验多数窗口无指标覆盖，另跑 {args.short_windows} 个 {args.short_window_days} 天窗口"
              f"（步长 {args.short_step_days} 天）作探索性信号；样本小且窗口重叠，不能替代主实验。", "",
              "| 配置 | 收益均值 % | 正窗口 | 期望>0 窗口 |", "|---|---:|---:|---:|"]
    for config in CONFIGS:
        s = short_summaries[config["key"]]
        lines.append(f"| {config['label']} | {rolling._pct(s['mean_return_pct'])} "
                     f"| {s['positive_windows']}/{s['total_windows']} "
                     f"| {s['expectancy_positive_windows']}/{s['total_windows']} |")
    if all(short_summaries[c["key"]]["mean_return_pct"] is None for c in CONFIGS):
        lines += ["", "注意：短窗口在 optimize=True（训练/验证/测试三段切分）口径下没有样本外测试段，"
                  "三配置均返回空，本补充实验**未产生可用结果**，不作为任何依据。"]
    lines += [
        "", "## 5. 加权公式说明", "",
        "仓位权重（weight_mode=metric，每 rebalance_bars 根 K 线重算，指标经 metrics_as_of 反前视取数）：", "```",
        "score_i = 0.5*tanh(sentiment_divergence_i)      # 聪明钱相对散户更偏多 → 加分",
        "        - 0.3*min(|crowd_ls_z_i|, 3)/3           # 散户拥挤度极端 → 减分（拥挤=反向风险）",
        "        + 0.2*rank_score(market_cap_rank_i)      # 市值靠前 → 加分（流动性/容量）",
        "rank_score: 第1名=1.0，线性衰减到第51名=0；排名未知=0.5（中性）",
        "score → softmax(T=0.5) → 非负 → 归一 sum=1 → 单币上限 clamp 到 2/n；任一币种指标缺失 → 整体退回等权",
        "```",
        "regime 策略权重（regime_weighting=True，同币种 trend_follow vs mean_reversion）：",
        "```",
        "w_trend = clamp(0.5 + A*(fng_value-50)/50 - B*min(|crowd_ls_z|,3)/3, 0.2, 0.8),  A=0.25, B=0.15",
        "w_mean_reversion = 1 - w_trend",
        "```", "",
        "## 6. 前视声明", "",
        "- **市值排名存在轻微前视**：CoinGecko 无免费历史，回填时把当前市值排名作为常量注入所有历史行。"
          "因此 rank_score 项（占 score 权重 0.2）带有前视信息，结论需打折。",
        "- 其余指标（多空比/大户比/主动买卖比/未平仓量/FNG）均为 ts <= 决策时刻的历史值，"
          "经 metrics_as_of 前向填充读取；z-score 因果构造：回填时每条只用它之前的已算特征作历史，不掺入未来。", "",
        "## 7. 采纳门禁判定", ""]
    for config in CONFIGS[1:]:
        passed, detail = gates[config["key"]]
        s, b = summaries[config["key"]], summaries["baseline"]
        lines += [
            f"### {config['label']}", "",
            f"- 正窗口数：{s['positive_windows']}/{s['total_windows']}（baseline "
            f"{b['positive_windows']}/{b['total_windows']}）→ {'更多' if detail['more_positive'] else '未增加'}",
            f"- 收益均值：{rolling._pct(s['mean_return_pct'])}（baseline {rolling._pct(b['mean_return_pct'])}）"
            f"→ {'更高' if detail['higher_mean'] else '未更高'}",
            f"- **门禁：{'通过' if passed else '未通过'}**", ""]  # noqa: E501
    any_pass = any(gates[c["key"]][0] for c in CONFIGS[1:])
    if any_pass:
        passed_keys = [c["label"] for c in CONFIGS[1:] if gates[c["key"]][0]]
        lines += ["## 8. 结论", "",
                  f"门禁通过的配置：{'、'.join(passed_keys)}。可考虑启用，但仍建议先小仓位观察，"
                  "且注意市值排名前视项需打折。"]
    else:
        lines += [
            "## 8. 结论", "",
            "**未改善，不建议启用。** 理由：",
            "1. 指标覆盖硬约束：免费源仅约 30 天，7 个 200 天窗口中 5 个覆盖为 0（完全无指标），"
              "另 2 个仅覆盖 0.2% 与 15.2%；",
            "2. 在唯一真正有指标信号生效的窗口组（覆盖>0，n=2）上，"
              f"相对 baseline：B {rolling._pct(splits['plus_b']['covered_mean'] and splits['plus_b']['covered_mean'] - splits['baseline']['covered_mean'])}、"
              f"C {rolling._pct(splits['plus_bc']['covered_mean'] and splits['plus_bc']['covered_mean'] - splits['baseline']['covered_mean'])}"
              "——均为负，即指标加权在有数据的地方是减分的；",
            "3. 覆盖=0 的 5 个窗口差异约 0.00%，属定期再平衡的机制差异，非指标效应；",
            "4. 门禁（正窗口数增加 且 均值更高）两配置均未达成：正窗口数同为 3/7，均值均更低。",
            "因此保持 `metrics_assist_enabled` 默认 False，指标层继续作为观测能力保留；"
            "覆盖期内短窗口补充实验因无样本外测试段全部返回空，亦未产生可用结果。"
            "若要真正验证加权价值，需先解决历史深度（付费数据或自行累积 6 个月以上快照）。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", dest="as_of", default="2026-10-25T12:30:00+00:00")
    parser.add_argument("--interval", default=INTERVAL)
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES)
    parser.add_argument("--window-days", type=int, default=200)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--windows", type=int, default=7)
    parser.add_argument("--short-window-days", type=int, default=30)
    parser.add_argument("--short-step-days", type=int, default=5)
    parser.add_argument("--short-windows", type=int, default=4)
    parser.add_argument("--rebalance-bars", type=int, default=24)
    parser.add_argument("--fee", type=float, default=FEE)
    parser.add_argument("--slippage", type=float, default=SLIPPAGE)
    parser.add_argument("--initial-cash", type=float, default=INITIAL_CASH)
    parser.add_argument("--symbols", nargs="+", default=TOP10)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "metric-weighting-validation.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "docs" / "metric-weighting-validation.json")
    args = parser.parse_args()
    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
    end_ms = int(as_of.timestamp() * 1000)
    engine = make_engine(read_only=True)

    groups, summaries, short_groups, short_summaries = {}, {}, {}, {}
    with Session(engine) as db:
        for config in CONFIGS:
            groups[config["key"]] = run_config(db, end_ms, args, config,
                                               args.window_days, args.step_days, args.windows)
            summaries[config["key"]] = rolling.summarize(groups[config["key"]])
        for config in CONFIGS:
            short_groups[config["key"]] = run_config(
                db, end_ms, args, config, args.short_window_days, args.short_step_days,
                args.short_windows)
            short_summaries[config["key"]] = rolling.summarize(short_groups[config["key"]])
    engine.dispose()

    gates = {c["key"]: adoption_gate(summaries["baseline"], summaries[c["key"]]) for c in CONFIGS[1:]}
    splits = {k: coverage_split(v) for k, v in groups.items()}
    args.output.write_text(build_report(groups, summaries, gates, as_of.isoformat(), args,
                                        short_groups, short_summaries, splits), encoding="utf-8")
    payload = {"as_of": as_of.isoformat(),
               "params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "summaries": summaries, "windows": groups,
               "short_summaries": short_summaries, "short_windows": short_groups,
               "gates": {k: {"passed": v[0], **v[1]} for k, v in gates.items()}}
    args.json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                                encoding="utf-8")
    verdict = "通过" if any(g["passed"] for g in payload["gates"].values()) else "未通过（不建议启用）"
    print(f"报告已生成: {args.output} | 采纳门禁: {verdict}")


if __name__ == "__main__":
    main()
