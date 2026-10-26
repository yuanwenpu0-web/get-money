r"""迭代5复验实验：Top10 拉长样本 + 黑名单剔除前后同窗口对比。

只读运行（不写订单/状态），结果追加到 docs/portfolio-validation.md，
完整数据写 docs/portfolio-revalidation.json。与单币种验证同口径：
optimize=True、验证集选参、样本外测试、双边 maker 费率。

引擎的 days 是统一上界：build_dataset 从 as_of-days 起加载本地数据，
新上市币种自然获得较短跨度，因此 days=400 即"最多 400 天、实际以各币种
最大可得跨度为准"，无需取最小跨度统一跑。

用法（仓库根目录）：
    .\.venv\Scripts\python.exe backend/scripts/run_iteration5_revalidation.py
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
from app.market.data_service import MarketDataService
from app.db import SessionLocal

STRATEGIES = ["trend_follow", "mean_reversion"]
FEE = 0.0002
SLIPPAGE = 0.0002
INTERVAL = "1h"
INITIAL_CASH = 10000.0

# 2026-10-25 选取：黑名单生效后宇宙（57）按 24h 成交额 Top10
TOP10 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "JCTUSDT", "XRPUSDT",
         "NEARUSDT", "ORDERUSDT", "IRYSUSDT", "APRUSDT", "DOGEUSDT"]
# 迭代4的 Top20（黑名单生效前的同一批流动性头部币种）
PRE_BLACKLIST_TOP20 = [
    "BTCUSDT", "ETHUSDT", "ZECUSDT", "SOLUSDT", "JCTUSDT", "XRPUSDT", "NEARUSDT",
    "ORDERUSDT", "IRYSUSDT", "APRUSDT", "BNBUSDT", "DOGEUSDT", "HYPEUSDT",
    "1000PEPEUSDT", "TUSDT", "WLDUSDT", "ENAUSDT", "BIRBUSDT", "SUIUSDT", "BMTUSDT",
]
BLACKLIST = ["ZECUSDT", "1000PEPEUSDT", "HYPEUSDT", "BNBUSDT", "BIRBUSDT", "BMTUSDT"]
POST_BLACKLIST_TOP14 = [s for s in PRE_BLACKLIST_TOP20 if s not in BLACKLIST]


def _fmt(value, digits=2, suffix=""):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def _pct(value):
    return _fmt(value, 2, "%")


def data_spans(symbols):
    db = SessionLocal()
    spans = {}
    for symbol in symbols:
        stats = MarketDataService(db).series_stats(symbol, INTERVAL)
        count, first, last = stats["count"], stats["first_open_time"], stats["last_open_time"]
        spans[symbol] = {
            "bars": count,
            "first": datetime.fromtimestamp(first / 1000, timezone.utc).strftime("%Y-%m-%d") if count else None,
            "last": datetime.fromtimestamp(last / 1000, timezone.utc).strftime("%Y-%m-%d") if count else None,
            "span_days": round((last - first) / 86_400_000, 1) if count else 0.0,
        }
    db.close()
    return spans


def run_all(db, end_ms):
    runs = {}
    plans = [
        ("long_400d", TOP10, 400),
        ("short_200d", TOP10, 200),
        ("pre_blacklist_200d", PRE_BLACKLIST_TOP20, 200),
        ("post_blacklist_200d", POST_BLACKLIST_TOP14, 200),
        ("blacklist_200d", BLACKLIST, 200),
    ]
    for name, symbols, days in plans:
        result = run_portfolio_backtest(
            db, symbols, INTERVAL, STRATEGIES, days=days, fee=FEE, slippage=SLIPPAGE,
            initial_cash=INITIAL_CASH, end_ms=end_ms,
        )
        runs[name] = result
        portfolio = result.get("portfolio", {}) if result.get("ok") else {}
        diversification = result.get("diversification", {}) if result.get("ok") else {}
        print(json.dumps({
            "run": name, "symbols": len(symbols), "days": days,
            "ok": result.get("ok"),
            "return_pct": portfolio.get("total_return_pct"),
            "max_drawdown_pct": portfolio.get("max_drawdown_pct"),
            "sharpe": portfolio.get("sharpe"),
            "trades": portfolio.get("total_trades"),
            "expectancy": portfolio.get("expectancy"),
            "positive_share_pct": diversification.get("positive_share_pct"),
            "errors": len(result.get("errors") or []),
        }, ensure_ascii=False), flush=True)
    return runs


def _metrics_row(label, result):
    portfolio = result["portfolio"]
    diversification = result["diversification"]
    return (
        f"| {label} | {_pct(portfolio.get('total_return_pct'))} "
        f"| {_pct(portfolio.get('max_drawdown_pct'))} | {_fmt(portfolio.get('sharpe'))} "
        f"| {_fmt(portfolio.get('expectancy'), 4)} | {portfolio.get('total_trades')} "
        f"| {_fmt(portfolio.get('total_fees'))} "
        f"| {_fmt(diversification.get('positive_share_pct'))}% |"
    )


def build_section(runs, spans, as_of_iso):
    long_run, short_run = runs["long_400d"], runs["short_200d"]
    pre_run, post_run = runs["pre_blacklist_200d"], runs["post_blacklist_200d"]
    black_run = runs["blacklist_200d"]
    lines = [
        "",
        "---",
        "",
        f"# 附：黑名单剔除效果与拉长样本复验（as-of {as_of_iso}）",
        "",
        "## A. 黑名单落地",
        "",
        "- 新设置项 `symbol_blacklist`（list[str]，默认 `[]`）：校验字符串数组、元素非空、"
        "统一大写、去重；与流动性过滤在 `market/universe.py` 叠加；宇宙缓存键含过滤配置，"
        "设置变更立即生效（不必等 1h TTL）。",
        f"- 已写入 6 个负期望候选：{'、'.join(BLACKLIST)}。",
        "- 实测：PUT /api/settings 后宇宙 63 → 57，6 个币种全部排除，缓存命中确认已持久化。",
        "",
        "## B. Top10 数据跨度（1h 400 天同步，92244 行插入、0 错误）",
        "",
        "| 币种 | K线根数 | 跨度(天) | 起始 | 结束 |",
        "|---|---:|---:|---|---|",
    ]
    for symbol in TOP10:
        span = spans[symbol]
        lines.append(f"| {symbol} | {span['bars']} | {span['span_days']} | {span['first']} | {span['last']} |")
    lines += [
        "",
        "6 个币种满 400 天；JCT/IRYS/APR/ORDER 上市较晚（333-394 天），跨度不足属正常，"
        "按实际可得数据参与回测（引擎 days 为上界，实际跨度由数据决定）。",
        "",
        "## C. Top10：400 天 vs 200 天（同 as-of、同参数、同 maker 费率）",
        "",
        "| 场景 | 总收益 % | 最大回撤 % | 夏普 | 期望值 USDT/笔 | 成交 | 费用 | 正收益占比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _metrics_row("Top10 × 2 策略，最多 400 天", long_run),
        _metrics_row("Top10 × 2 策略，200 天", short_run),
        "",
        "注意：拉长窗口同时拉长了样本外测试段（200 天窗口测试约 50 天，400 天窗口测试约 "
        "83-100 天），因此这是“更长数据 + 更大样本”的比较，不是纯窗口受控实验。",
        "",
        "单币种收益对比（组合内，按 400 天 / 200 天）：",
        "",
        "| 币种 | 400天收益 % | 200天收益 % | 400天成交 | 200天成交 | 400天期望值 | 200天期望值 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    long_rows = {row["symbol"]: row for row in long_run["symbols"]}
    short_rows = {row["symbol"]: row for row in short_run["symbols"]}
    for symbol in TOP10:
        long_row, short_row = long_rows.get(symbol), short_rows.get(symbol)
        if not long_row or not short_row:
            continue
        lines.append(
            f"| {symbol} | {_pct(long_row['return_pct'])} | {_pct(short_row['return_pct'])} "
            f"| {long_row['total_trades']} | {short_row['total_trades']} "
            f"| {_fmt(long_row['expectancy'], 4)} | {_fmt(short_row['expectancy'], 4)} |"
        )
    lines += [
        "",
        "## D. 黑名单剔除前后（同窗口 200 天、同 as-of、同参数）",
        "",
        "| 场景 | 总收益 % | 最大回撤 % | 夏普 | 期望值 USDT/笔 | 成交 | 费用 | 正收益占比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _metrics_row("剔除前 Top20（含 6 个黑名单）", pre_run),
        _metrics_row("剔除后 Top14（黑名单生效）", post_run),
        "",
        "等权组合下移除低于均值的腿会机械性抬高组合收益，因此收益变化不构成证据；"
        "关键看回撤、夏普与期望值是否同向改善，以及被剔除者是否确实持续负期望（见 E）。",
        "",
        "## E. 6 个黑名单币种在 200 天窗口的表现（同 as-of）",
        "",
        "| 币种 | 收益 % | 最大回撤 % | 成交 | 期望值 USDT/笔 | 胜率 % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(black_run["symbols"], key=lambda item: item["return_pct"] or 0.0):
        lines.append(
            f"| {row['symbol']} | {_pct(row['return_pct'])} | {_pct(row['max_drawdown_pct'])} "
            f"| {row['total_trades']} | {_fmt(row['expectancy'], 4)} | {_fmt(row['win_rate'])} |"
        )
    borderline = [row for row in black_run["symbols"] if (row["expectancy"] or 0) >= 0]
    if borderline:
        lines += [
            "",
            "诚实备注：本窗口中 "
            + "、".join(row["symbol"] for row in borderline)
            + " 期望值非负（上一窗口为负），黑名单理由弱于其余币种；保留在名单中但标注为待复验，"
            "下个窗口若继续转正应移出。",
        ]
    lines += [
        "",
        "## F. 结论",
        "",
        f"1. 拉长样本（Top10 最多 400 天）：组合收益 {_pct(long_run['portfolio'].get('total_return_pct'))}、"
        f"回撤 {_pct(long_run['portfolio'].get('max_drawdown_pct'))}、"
        f"夏普 {_fmt(long_run['portfolio'].get('sharpe'))}；对比 200 天窗口的 "
        f"{_pct(short_run['portfolio'].get('total_return_pct'))} / "
        f"{_pct(short_run['portfolio'].get('max_drawdown_pct'))} / "
        f"{_fmt(short_run['portfolio'].get('sharpe'))}。",
        f"2. 黑名单剔除（同窗口 200 天）：收益 {_pct(pre_run['portfolio'].get('total_return_pct'))} → "
        f"{_pct(post_run['portfolio'].get('total_return_pct'))}，回撤 "
        f"{_pct(pre_run['portfolio'].get('max_drawdown_pct'))} → "
        f"{_pct(post_run['portfolio'].get('max_drawdown_pct'))}，夏普 "
        f"{_fmt(pre_run['portfolio'].get('sharpe'))} → {_fmt(post_run['portfolio'].get('sharpe'))}。",
        "3. 以上均为单次样本外窗口结果，不构成盈利证明；建议按季度用新数据滚动复验，"
        "黑名单与流动性门槛随复验结果调整。",
        "",
        "复现：",
        "",
        "```powershell",
        '.\\.venv\\Scripts\\python.exe backend/scripts/run_iteration5_revalidation.py --as-of "' + as_of_iso + '"',
        "```",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default=datetime.now(timezone.utc).isoformat())
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "portfolio-validation.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "docs" / "portfolio-revalidation.json")
    args = parser.parse_args()
    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
    if as_of.tzinfo is None:
        parser.error("--as-of must include a timezone")
    end_ms = int(as_of.timestamp() * 1000)

    engine = make_engine(read_only=True)
    with Session(engine) as db:
        runs = run_all(db, end_ms)
    engine.dispose()
    for name, result in runs.items():
        if not result.get("ok"):
            print(f"运行 {name} 失败: {result.get('errors') or result.get('message')}", file=sys.stderr)

    spans = data_spans(TOP10)
    section = build_section(runs, spans, as_of.isoformat())
    with args.report.open("a", encoding="utf-8") as handle:
        handle.write(section)
    args.json_output.write_text(json.dumps(
        {"as_of": as_of.isoformat(), "fee": FEE, "slippage": SLIPPAGE,
         "strategies": STRATEGIES, "runs": runs, "spans": spans},
        ensure_ascii=False, indent=2, allow_nan=False,
    ), encoding="utf-8")
    print(f"报告已追加: {args.report}")


if __name__ == "__main__":
    main()
