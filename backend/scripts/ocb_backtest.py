r"""开盘首根突破 [OCB] —— 主流币历史检验。

目的
----
把 TradingView 脚本 `Opening Candle Breakout [OCB] Strategy`（© Swissy 2026，MPL-2.0）
逐条移植到本地后，在一批主流币上做一次**诚实回放**：

- 入场：会话首窗那一根的收盘判定方向 → 次根开盘成交（按氢氧化物/滑点计价）；
- 出场：脚本的三条 —— 结构止损、固定 R 止盈、**强平窗平掉所有持仓**，谁先到算谁；
- 仓位：不是名义百分比，而是「每笔风险预算 ÷ 每单位风险距离」，每天最多一笔。

对照组（决定「有没有 edge」而不是「收益多少」）
----------------------------------------------
- **随机方向基线**：相同入场时点、相同风险距离、相同成本，只把 setup 方向 50/50 翻转
  （止损/止盈同步镜像），跑 `--seeds` 个种子看账户收益分布。
- **每笔期望的自举置信区间**：跨币种池化后的 95% CI，跨 0 即不可判定。

会话与周期的坑
--------------
脚本的会话窗是按**K线开盘时间**落进区间来判定的，必须随图表周期选窗：
- `--interval 5m` + 首窗 `570-575`(09:30-09:35) + 强平窗 `955-960`(15:55-16:00) = 脚本原文；
- `--interval 15m` 上首窗仍然只命中 09:30 那一根，但 `1555-1600` **一根都命中不了**
  （15m 落的开盘时间是 15:45 / 16:00），所以 15m 必须把强平窗放宽到 `945-960`。

用法
----
```
.\.venv\Scripts\python.exe backend/scripts/ocb_backtest.py                       # 主流币 15m，180 天
.\.venv\Scripts\python.exe backend/scripts/ocb_backtest.py --interval 5m --days 60 \
    --flat-start 955 --flat-end 960                                              # 脚本原文会话
.\.venv\Scripts\python.exe backend/scripts/ocb_backtest.py --params '{"target_r": 4}'
```

进程池用于逐 bar 撮合（CPU 密集、被 GIL 锁死），线程用于取数（IO 密集）。
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.backtest.metrics import metrics_from_equity  # noqa: E402
from app.backtest.ocb_execution import simulate_ocb  # noqa: E402
from app.backtest.significance import bootstrap_expectancy_ci  # noqa: E402
from app.config import DEFAULT_STATE  # noqa: E402
from app.db import make_engine  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.strategy.coin_universe import MAJOR_SYMBOLS  # noqa: E402
from app.strategy.rules import evaluate_signals, resolve_params  # noqa: E402
from app.utils import interval_ms, now_ms  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

CODE = "opening_candle_breakout"
DAY_MS = 86_400_000

_LOCAL = threading.local()


def _service():
    """每个线程一个会话：SQLAlchemy 会话不是线程安全的。"""
    service = getattr(_LOCAL, "service", None)
    if service is None:
        engine = make_engine(read_only=True)
        session = sessionmaker(bind=engine, expire_on_commit=False)()
        _LOCAL.service = MarketDataService(session)
        service = _LOCAL.service
    return service


# ---------------------------------------------------------------- 数据


def load_symbol(symbol: str, interval: str, days: int, end_ms: int) -> dict:
    try:
        frame = _service().load_frame(
            symbol, interval, start_ms=end_ms - days * DAY_MS, end_ms=end_ms
        )
    except Exception as exc:  # 数据缺口/连接抖动：跳过而不是中断整轮
        return {"symbol": symbol, "error": str(exc)}
    if frame.empty:
        return {"symbol": symbol, "error": "本地无K线数据"}
    step = interval_ms(interval)
    frame = frame.loc[(frame.index.as_unit("ms").asi8 + step) <= end_ms]
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    if len(frame) < 300:
        return {"symbol": symbol, "error": f"样本不足 {len(frame)} 根"}
    prices = frame[["open", "high", "low", "close"]]
    valid = prices.gt(0).all(axis=1)
    valid &= frame["high"].ge(frame[["open", "close", "low"]].max(axis=1))
    valid &= frame["low"].le(frame[["open", "close", "high"]].min(axis=1))
    return {
        "symbol": symbol,
        "frame": frame.loc[valid, ["open", "high", "low", "close", "volume"]],
        "rows": int(len(frame)),
        "dropped": int((~valid).sum()),
    }


# ---------------------------------------------------------------- 单币回测


def flip_directions(signals: pd.DataFrame, close: pd.Series, seed: int, target_r: float) -> pd.DataFrame:
    """随机方向基线：只翻 setup 的方向，止损/止盈按信号收盘镜像，风险距离与成本不变。"""
    flipped = signals.copy()
    rng = random.Random(20261030 + seed)
    for stamp in flipped.index[flipped["ocb_dir"].ne(0)]:
        if rng.random() >= 0.5:
            continue
        risk = float(flipped.at[stamp, "ocb_risk"])
        price = float(close.at[stamp])
        direction = -float(flipped.at[stamp, "ocb_dir"])
        flipped.at[stamp, "ocb_dir"] = direction
        flipped.at[stamp, "ocb_stop"] = price - direction * risk
        flipped.at[stamp, "ocb_target"] = price + direction * risk * target_r
    return flipped


def evaluate_symbol(task: dict) -> dict:
    """单币 worker（模块级函数，进程池需要 pickle）：信号 → 专用撮合 → 指标 → 随机基线。"""
    frame = task["frame"]
    interval = task["interval"]
    cash, fee, slippage = task["cash"], task["fee"], task["slippage"]
    params = task["params"]
    signals = evaluate_signals(frame, CODE, params)
    setups = int(signals["ocb_dir"].ne(0).sum())
    blocked = {
        str(key): int(value)
        for key, value in signals["block_reason"][signals["block_reason"].ne("")].value_counts().items()
    }
    payload = {
        "symbol": task["symbol"],
        "rows": int(len(frame)),
        "setups": setups,
        "trades": 0,
        "blocked": blocked,
        "status_counts": {},
        "trade_returns": [],
        "random_returns": [],
    }
    if setups == 0:
        return payload
    equity, trades = simulate_ocb(frame, signals, params, cash, fee, slippage, interval)
    for seed in range(task["seeds"]):
        randomized = flip_directions(signals, frame["close"], seed, float(params["target_r"]))
        random_equity, random_trades = simulate_ocb(frame, randomized, params, cash, fee, slippage, interval)
        random_metrics = metrics_from_equity(random_equity, random_trades, cash, interval)
        payload["random_returns"].append(round(float(random_metrics.get("total_return_pct", 0.0)), 4))
    payload.update({
        "trades": len(trades),
        "metrics": {
            key: (round(float(value), 4) if isinstance(value, (int, float)) else str(value))
            for key, value in metrics_from_equity(equity, trades, cash, interval).items()
            if key in ("total_return_pct", "win_rate", "max_drawdown_pct", "sharpe", "profit_factor",
                       "avg_trade_return_pct", "avg_bars_held", "exposure_pct")
        },
        "status_counts": dict(Counter(item["status"] for item in trades)),
        "trade_returns": [round(float(item["return_pct"]), 6) for item in trades],
        "expectancy_ci": bootstrap_expectancy_ci(trades),
        "r_by_status": {
            status: round(
                statistics.fmean(
                    [item["return_pct"] for item in trades if item["status"] == status]
                ),
                4,
            )
            for status in sorted({item["status"] for item in trades})
        },
    })
    return payload


# ---------------------------------------------------------------- 汇总


def merge_counts(sequence) -> dict:
    totals: dict[str, int] = {}
    for mapping in sequence:
        for key, value in (mapping or {}).items():
            totals[str(key)] = totals.get(str(key), 0) + int(value)
    return dict(sorted(totals.items(), key=lambda item: -item[1]))


def summarise(results: list[dict]) -> dict:
    usable = [row for row in results if row["trades"] > 0]
    returns = [row["metrics"]["total_return_pct"] for row in usable]
    pooled = [value for row in usable for value in row["trade_returns"]]
    pooled_ci = bootstrap_expectancy_ci(
        [{"return_pct": value, "pnl": value} for value in pooled]
    ) if pooled else {"verdict": "no_trades"}
    random_pool = [value for row in usable for value in row["random_returns"]]
    return {
        "symbols_total": len(results),
        "symbols_with_trades": len(usable),
        "trades": sum(row["trades"] for row in usable),
        "return_median_pct": round(statistics.median(returns), 4) if returns else 0.0,
        "return_mean_pct": round(statistics.fmean(returns), 4) if returns else 0.0,
        "positive_symbols": int(sum(1 for value in returns if value > 0)),
        "win_rate_mean_pct": round(statistics.fmean([row["metrics"]["win_rate"] for row in usable]), 2)
        if usable else 0.0,
        "max_drawdown_mean_pct": round(
            statistics.fmean([row["metrics"]["max_drawdown_pct"] for row in usable]), 4)
        if usable else 0.0,
        "avg_trade_return_pct": round(statistics.fmean(pooled), 4) if pooled else 0.0,
        "return_ci": {
            "ci_low_pct": round(statistics.fmean(pooled) - 1.96 * statistics.stdev(pooled) / len(pooled) ** 0.5, 4)
            if len(pooled) > 1 else None,
            "ci_high_pct": round(statistics.fmean(pooled) + 1.96 * statistics.stdev(pooled) / len(pooled) ** 0.5, 4)
            if len(pooled) > 1 else None,
        },
        "expectancy_verdict": pooled_ci.get("verdict", "no_trades"),
        "random_baseline_return_mean_pct": round(statistics.fmean(random_pool), 4) if random_pool else 0.0,
        "beats_random_mean": int(
            sum(1 for row in usable if row["metrics"]["total_return_pct"] > statistics.fmean(row["random_returns"]))
        ) if usable and all(row["random_returns"] for row in usable) else 0,
        "status_counts": merge_counts(row.get("status_counts", {}) for row in results),
        "blocked": merge_counts(row.get("blocked", {}) for row in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="开盘首根突破（OCB）历史回测")
    parser.add_argument("--symbols", default="", help="逗号分隔；默认用主流币清单")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--cash", type=float, default=DEFAULT_STATE["initial_capital"])
    parser.add_argument("--fee", type=float, default=DEFAULT_STATE["fee_rate"])
    parser.add_argument("--slippage", type=float, default=DEFAULT_STATE["slippage"])
    parser.add_argument("--flat-start", type=int, default=945, help="强平窗起点（会话时区分钟数）")
    parser.add_argument("--flat-end", type=int, default=960, help="强平窗终点（会话时区分钟数）")
    parser.add_argument("--params", default="{}", help="策略参数覆盖，JSON 字符串")
    parser.add_argument("--seeds", type=int, default=8, help="随机方向基线种子数")
    parser.add_argument("--workers", type=int, default=6, help="取数线程数")
    parser.add_argument("--processes", type=int, default=6, help="撮合进程数")
    parser.add_argument("--out", default="docs/opening-candle-breakout-strategy")
    args = parser.parse_args()

    overrides = json.loads(args.params) if args.params.strip() else {}
    overrides.setdefault("flat_session_start_min", args.flat_start)
    overrides.setdefault("flat_session_end_min", args.flat_end)
    params = resolve_params(CODE, {"fee_rate": args.fee, "slippage": args.slippage}, overrides)
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()] or list(MAJOR_SYMBOLS)
    end_ms = now_ms()
    started = time.monotonic()

    print(f"[取数] {len(symbols)} 币 × {args.interval} × {args.days} 天")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        frames = dict(zip(symbols, pool.map(
            lambda item: load_symbol(item, args.interval, args.days, end_ms), symbols)))
    skipped = [f"{name}: {payload.get('error')}" for name, payload in frames.items()
               if payload.get("frame") is None]
    if skipped:
        print("  跳过 " + "; ".join(skipped[:4]) + ("…" if len(skipped) > 4 else ""))
    tasks = [
        {
            "symbol": name, "frame": payload["frame"], "params": params,
            "interval": args.interval, "cash": args.cash, "fee": args.fee,
            "slippage": args.slippage, "seeds": args.seeds,
        }
        for name, payload in frames.items()
        if payload.get("frame") is not None
    ]
    print(f"[撮合] {len(tasks)} 币 / {time.monotonic() - started:.1f}s 数据就绪")
    if not tasks:
        print("没有任何可用样本，退出")
        return 1
    if args.processes > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(args.processes, len(tasks))) as pool:
            results = list(pool.map(evaluate_symbol, tasks))
    else:
        results = [evaluate_symbol(task) for task in tasks]
    summary = summarise(results)
    print(f"[汇总] 成交 {summary['trades']} 笔 / {summary['symbols_with_trades']} 币；"
          f"收益中位数 {summary['return_median_pct']:+.2f}%；"
          f"每笔均值 {summary['avg_trade_return_pct']:+.4f}%（随机基线 "
          f"{summary['random_baseline_return_mean_pct']:+.2f}%）")

    report = {
        "strategy": CODE,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "interval": args.interval,
        "days": args.days,
        "symbols": symbols,
        "parameters": params,
        "costs": {"fee_rate": args.fee, "slippage": args.slippage, "cash_per_symbol": args.cash},
        "random_baseline_seeds": args.seeds,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "skipped": skipped,
        "aggregate": summary,
        "per_symbol": results,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _write_markdown(out.with_suffix(".md"), report)
    print(f"[输出] {out.with_suffix('.json').relative_to(ROOT)} / {out.with_suffix('.md').relative_to(ROOT)}")
    return 0


def _verdict(summary: dict) -> tuple[str, str]:
    """按全项目统一判据给结论：CI 下界 > 0 才算有正 edge，跨 0 不可判定，上界 < 0 为负贡献。"""
    low, high = summary["return_ci"]["ci_low_pct"], summary["return_ci"]["ci_high_pct"]
    if low is None or high is None:
        return "样本不足", "无法估计置信区间，按默认处置：**不启用**。"
    if high < 0:
        return "负 edge", (
            f"池化每笔收益 95% CI 上界 {high:+.4f}% < 0 —— 在当前样本与成本下这套规则是**净负贡献**。\n"
            "处置：**不启用**，也不再往参数/过滤上投入（每次螺旋 tuning 只会拟合噪声）。"
        )
    if low > 0:
        return "正 edge", "池化 CI 下界 > 0：可以进入小额实测阶段，但仍需滚动窗口复核方向是否稳定。"
    return "不可判定", "池化 CI 跨 0：既不能说有 edge，也不能说没有；默认值是「不启用」，而不是「先跑跑看」。"


def _write_markdown(path: Path, report: dict) -> None:
    summary = report["aggregate"]
    params = report["parameters"]
    verdict, disposition = _verdict(summary)
    lines = [
        f"# 开盘首根突破 [OCB] —— 主流币回测报告（{report['interval']}）",
        "",
        f"生成时间：{report['generated_at']}　周期：{report['interval']}　窗口：{report['days']} 天",
        "",
        "## 结论速览",
        "",
        f"- 有交易的币种 {summary['symbols_with_trades']}/{summary['symbols_total']}，合计成交 "
        f"**{summary['trades']} 笔**",
        f"- 单币收益中位数 **{summary['return_median_pct']:+.2f}%**，均值 {summary['return_mean_pct']:+.2f}%",
        f"- 每笔收益均值 **{summary['avg_trade_return_pct']:+.4f}%**，"
        f"95% CI [{summary['return_ci']['ci_low_pct']}, {summary['return_ci']['ci_high_pct']}] → "
        f"**{verdict}**",
        f"- 随机方向基线（同成本同时点，{report['random_baseline_seeds']} 种子，均值）："
        f"{summary['random_baseline_return_mean_pct']:+.2f}%",
        f"- 跑赢随机基线均值的币种：{summary['beats_random_mean']}/{summary['symbols_with_trades']}",
        "",
        "## 成本、会话与判读规则",
        "",
        f"- 成本：双边费率 {report['costs']['fee_rate']}、双边滑点 {report['costs']['slippage']}；"
        f"每币初始资金 {report['costs']['cash_per_symbol']}",
        f"- 会话：首窗 {params['first_session_start_min']}-{params['first_session_end_min']} 分、"
        f"强平窗 {params['flat_session_start_min']}-{params['flat_session_end_min']} 分"
        f"（America/New_York 墙上时间，按K线**开盘时间**判定）",
        "- 判读：CI 下界 > 0 才算有正 edge；跨 0 不可判定；上界 < 0 说明在当前样本下是负贡献。",
        "",
        "## 结论与处置",
        "",
        disposition,
        "",
        "出场结构：",
        "",
        "| 出场原因 | 笔数 |",
        "| --- | ---: |",
    ]
    for status, count in summary["status_counts"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend([
        "",
        "## 逐币明细",
        "",
        "| 币种 | setup | 成交 | 收益% | 胜率% | 回撤% |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in report["per_symbol"]:
        if not row["trades"]:
            lines.append(f"| {row['symbol']} | {row['setups']} | 0 | — | — | — |")
            continue
        metrics = row["metrics"]
        lines.append(
            f"| {row['symbol']} | {row['setups']} | {row['trades']} | "
            f"{metrics['total_return_pct']:+.2f} | {metrics['win_rate']} | {metrics['max_drawdown_pct']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
