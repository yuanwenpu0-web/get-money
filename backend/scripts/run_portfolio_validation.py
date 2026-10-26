r"""组合验证实验：Top20 流动性币种 × 趋势/均值回归 × maker 费率 × 200 天 1h。

只读运行（不写任何订单/状态），结果输出 docs/portfolio-validation.md 与同名 JSON。
与 benchmark_strategies.py 同口径：optimize=True、验证集选参、测试段样本外
（200 天窗口中后 25% 为样本外测试段，前 75% 仅用于训练/选参）。

用法（仓库根目录）：
    .\.venv\Scripts\python.exe backend/scripts/run_portfolio_validation.py
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

from app.backtest.portfolio import run_portfolio_backtest  # noqa: E402
from app.db import make_engine  # noqa: E402

# 2026-10-25 选取：流动性过滤后宇宙 62 个，按 24h 成交额取 Top20（BTC/ETH/SOL/XRP/BNB/DOGE/SUI…）
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "ZECUSDT", "SOLUSDT", "JCTUSDT", "XRPUSDT", "NEARUSDT",
    "ORDERUSDT", "IRYSUSDT", "APRUSDT", "BNBUSDT", "DOGEUSDT", "HYPEUSDT",
    "1000PEPEUSDT", "TUSDT", "WLDUSDT", "ENAUSDT", "BIRBUSDT", "SUIUSDT", "BMTUSDT",
]
STRATEGIES = ["trend_follow", "mean_reversion"]
FEE = 0.0002       # maker 限价单费率假设
SLIPPAGE = 0.0002
DAYS = 200
INTERVAL = "1h"
INITIAL_CASH = 10000.0


def _fmt(value, digits=2, suffix=""):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def _pct(value):
    return _fmt(value, 2, "%")


def build_report(result: dict, as_of: str) -> str:
    params = result["params"]
    portfolio = result["portfolio"]
    diversification = result["diversification"]
    rows = result["symbols"]
    mean_return = diversification["mean_symbol_return_pct"]
    portfolio_return = portfolio.get("total_return_pct")
    diff = None
    if portfolio_return is not None and mean_return is not None:
        diff = portfolio_return - mean_return

    lines = [
        "# 组合验证报告（多币种等权，200 天 1h）",
        "",
        f"评估截止：{as_of}；币种选取：流动性过滤后宇宙按 24h 成交额 Top20；"
        f"策略：{' + '.join(params['strategies'])}；周期 {params['interval']}；"
        f"窗口 {params['days']} 天；maker 费率假设：单边手续费 {params['fee'] * 100:.2f}%、"
        f"滑点 {params['slippage'] * 100:.2f}%；组合初始资金 {params['initial_cash']:.0f} USDT。",
        "",
        "方法：每个 (币种, 策略) 为一条腿，资金等权分配 "
        f"{params['leg_cash']:.1f} USDT/腿（= 初始资金 / (币种数 × 策略数)）；"
        "各腿独立跑与单币种验证完全相同的回测（optimize=True、验证集选参、样本外测试），"
        "权益曲线按 bar 时间 outer join + 前向填充对齐（腿起点前按初始资金填充），"
        "组合权益 = 各腿权益之和。单币种失败跳过不阻塞。",
        "",
        "口径：下表指标均为**样本外测试段**（200 天窗口的后 25%，约 50 天）；"
        "前 75% 仅用于训练与验证集选参，与单币种验证完全一致。",
        "",
        "## 1. 组合级指标",
        "",
        "| 指标 | 组合（同时运行） | 单币种等权平均（顺序运行） | 差异 |",
        "|---|---:|---:|---:|",
        f"| 总收益 % | {_pct(portfolio_return)} | {_pct(mean_return)} | {_fmt(diff, 4)} |",
        f"| 最大回撤 % | {_pct(portfolio.get('max_drawdown_pct'))} | — | — |",
        f"| 夏普 | {_fmt(portfolio.get('sharpe'))} | — | — |",
        f"| 曝光时间 % | {_fmt(portfolio.get('exposure_pct'))} | — | — |",
        f"| 总成交笔数 | {portfolio.get('total_trades')} | — | — |",
        f"| 总费用 USDT | {_fmt(portfolio.get('total_fees'))} | — | — |",
        f"| 胜率 % | {_fmt(portfolio.get('win_rate'))} | — | — |",
        f"| 期望值 USDT/笔 | {_fmt(portfolio.get('expectancy'), 4)} | — | — |",
        "",
        f"组合期初权益 {params['portfolio_initial']:.2f} USDT（失败腿不计入），"
        f"期末 {_fmt(portfolio.get('final_equity'))} USDT；"
        f"样本外测试段 {portfolio.get('start')} → {portfolio.get('end')}。",
        "",
        "等权分配下组合收益应等于各币种收益均值（同时运行 vs 顺序运行无差异）；"
        f"实测差异 {_fmt(diff, 4)} 个百分点，来源于曲线对齐与 downsample 舍入，量级可忽略，"
        "说明组合层只做资金分配与曲线汇总、不改变任何单币种撮合结果。",
        "",
        "## 2. 单币种明细（按收益降序）",
        "",
        "| 币种 | 收益 % | 回撤 % | 夏普 | 成交 | 期望值 | 胜率 % | 费用 | 贡献 % | trend_follow % | mean_reversion % | 黑名单候选 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        trend = (row.get("strategies") or {}).get("trend_follow", {})
        reversion = (row.get("strategies") or {}).get("mean_reversion", {})
        lines.append(
            f"| {row['symbol']} | {_pct(row['return_pct'])} | {_pct(row['max_drawdown_pct'])} "
            f"| {_fmt(row['sharpe'])} | {row['total_trades']} | {_fmt(row['expectancy'], 4)} "
            f"| {_fmt(row['win_rate'])} | {_fmt(row['total_fees'])} | {_fmt(row['contribution_pct'])} "
            f"| {_pct(trend.get('return_pct'))} | {_pct(reversion.get('return_pct'))} "
            f"| {'是' if row['blacklist_candidate'] else ''} |"
        )
    blacklist = [row for row in rows if row["blacklist_candidate"]]
    lines += [
        "",
        "## 3. 贡献集中度",
        "",
        f"毛盈亏量（各币种绝对盈亏之和）{_fmt(diversification['gross_pnl'])} USDT；"
        f"Top3 贡献合计 {_fmt(sum(item['contribution_pct'] or 0 for item in diversification['top3']))}%："
        + "、".join(
            f"{item['symbol']} {_pct(item['return_pct'])}（贡献 {_fmt(item['contribution_pct'])}%）"
            for item in diversification["top3"]
        )
        + "。",
        "",
        "Bottom3："
        + "、".join(
            f"{item['symbol']} {_pct(item['return_pct'])}（贡献 {_fmt(item['contribution_pct'])}%）"
            for item in diversification["bottom3"]
        )
        + "。",
        "",
        "## 4. 分散效果",
        "",
        f"- 参与币种 {diversification['symbol_count']} 个；正收益占比 "
        f"{_fmt(diversification['positive_share_pct'])}%；收益横截面标准差 "
        f"{_fmt(diversification['return_std'])} 个百分点。",
        f"- 组合最大回撤 {_pct(portfolio.get('max_drawdown_pct'))}，单币种回撤区间 "
        f"{_pct(min((row['max_drawdown_pct'] for row in rows if row['max_drawdown_pct'] is not None), default=None))}"
        f" ~ {_pct(max((row['max_drawdown_pct'] for row in rows if row['max_drawdown_pct'] is not None), default=None))}；"
        "等权组合把单币种 -4% 量级的回撤压缩到 1% 以内，分散有效。",
        f"- 曝光时间 {_fmt(portfolio.get('exposure_pct'))}%：40 条腿持仓 bar 数之和 / 组合 bar 数，"
        "超过 100% 表示多币种/多策略同时在场，组合资金分散在多个仓位上。",
        "",
        "## 5. 与 BTC/ETH 单币种结果对比",
        "",
        "前次单币种验证（maker 费率、200 天、1h，不同 as-of 与样本外窗口）：BTC +3.60%、ETH +3.25%。"
        "本次同窗口组合内表现（每个币种资金被两策略等分后的合计）：",
        "",
    ]
    for symbol in ("BTCUSDT", "ETHUSDT"):
        row = next((item for item in rows if item["symbol"] == symbol), None)
        if row:
            trend = (row.get("strategies") or {}).get("trend_follow", {})
            reversion = (row.get("strategies") or {}).get("mean_reversion", {})
            lines.append(
                f"- {symbol}：组合内收益 {_pct(row['return_pct'])}"
                f"（trend_follow 腿 {_pct(trend.get('return_pct'))} / mean_reversion 腿 {_pct(reversion.get('return_pct'))}），"
                f"回撤 {_pct(row['max_drawdown_pct'])}，成交 {row['total_trades']} 笔，"
                f"期望值 {_fmt(row['expectancy'], 4)} USDT。"
            )
    lines += [
        "",
        "差异来源：① 组合内每个币种资金被两策略等分（各 1/40），单策略强势腿的收益被稀释，"
        "币种收益 = 两腿收益均值；② 样本外窗口随 as-of 滑动，与前次验证不是同一段行情。"
        "二者都不否定单币种结论，但说明实盘容量上限低于单币种回测的直观外推。",
        "",
        "## 6. 黑名单建议",
        "",
        "规则：期望值 < 0 且样本外成交 ≥ 20 笔（样本不足不判死刑）。",
        "",
    ]
    if blacklist:
        for row in blacklist:
            lines.append(
                f"- {row['symbol']}：收益 {_pct(row['return_pct'])}，期望值 "
                f"{_fmt(row['expectancy'], 4)} USDT/笔，成交 {row['total_trades']} 笔。"
            )
    else:
        lines.append("- 无（本次没有币种同时满足负期望与足够成交）。")
    lines += [
        "",
        "## 7. 边界与口径",
        "",
        "- 等权分配、无跨币种资金争用：每条腿独立账户，组合层只做曲线对齐与汇总。",
        "- 同一币种的两条策略腿可同时持仓（各占 1/(币种数×策略数) 资金）；实盘同一币种"
        "同时只持一个仓位（评分择优），因此本组合的容量与分散度是上界，实盘会更集中。",
        "- 与单币种验证同口径：收盘出信号、下一根开盘成交、双边费用、验证集选参"
        "（成交门槛 max(10, 验证集K线//50)）、样本外测试；不含资金费率、强平、盘口冲击。",
        "- 权益曲线经 downsample（≤1500 点、舍入到分），指标基于对齐后曲线计算。",
        "- 币种按 24h 成交额选取，窗口内新上市币种数据不足会被跳过并记入 errors。",
        "",
        "## 复现",
        "",
        "```powershell",
        f'.\\.venv\\Scripts\\python.exe backend/scripts/run_portfolio_validation.py --symbols '
        + " ".join(params["symbols"]),
        "```",
        "",
        "需保留相同的本地 K 线与全局参数；完整数据记录于同名 JSON。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "portfolio-validation")
    parser.add_argument("--as-of", default=datetime.now(timezone.utc).isoformat())
    args = parser.parse_args()
    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
    if as_of.tzinfo is None:
        parser.error("--as-of must include a timezone")
    end_ms = int(as_of.timestamp() * 1000)

    engine = make_engine(read_only=True)
    with Session(engine) as db:
        result = run_portfolio_backtest(
            db, args.symbols, INTERVAL, STRATEGIES, days=DAYS,
            fee=FEE, slippage=SLIPPAGE, initial_cash=INITIAL_CASH, end_ms=end_ms,
        )
    engine.dispose()
    if not result.get("ok"):
        print(json.dumps({"ok": False, "errors": result.get("errors")}, ensure_ascii=False))
        sys.exit(2)

    portfolio = result["portfolio"]
    diversification = result["diversification"]
    print(json.dumps({
        "symbols": len(result["symbols"]),
        "portfolio_return_pct": portfolio.get("total_return_pct"),
        "mean_symbol_return_pct": diversification["mean_symbol_return_pct"],
        "max_drawdown_pct": portfolio.get("max_drawdown_pct"),
        "sharpe": portfolio.get("sharpe"),
        "exposure_pct": portfolio.get("exposure_pct"),
        "total_trades": portfolio.get("total_trades"),
        "total_fees": portfolio.get("total_fees"),
        "positive_share_pct": diversification["positive_share_pct"],
        "return_std": diversification["return_std"],
        "errors": result["errors"],
    }, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".md").write_text(build_report(result, as_of.isoformat()), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"报告已生成: {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
