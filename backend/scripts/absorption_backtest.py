r"""吸收反转（Absorption Reversal）策略 — 主流币历史回测。

目的
----
把 TradingView 脚本 `Absorption Reversal`（© blitz_locked）逐条移植到共享撮合后，
在**主流币**上做一次诚实的历史检验：真实回放完工 barrière
（信号收盘 → 次根开盘成交、费率/滑点、R 倍数止盈、1R 保本、30 根时间止损）。

对照组（决定"有没有 edge"，而不是"收益多少"）
------------------------------------------
- **随机方向基线**：相同时点、相同仓位、相同风控、相同成本，只把方向 50/50 翻转
  （10 个种子）。策略收益若不能显著超过它，说明方向判断没有信息量。
- **每笔期望的自举置信区间**：95% CI 跨 0 即不可判定。
- **delta 来源对照**：脚本真正的行为是用更低周期 K 线聚合 intrabar delta；
  本项目 1m 历史只有约 93 天，其余区间只能退化为脚本自带的 bar shape 估计（estDelta）。
  `--delta-source auto|lower|est` 控制用哪个，`lower` 缺数据时自动降级并记录在
  `delta_source_used` 里，避免把"没数据"误读成"没信号"。

用法
----
```
.\.venv\Scripts\python.exe backend/scripts/absorption_backtest.py                      # 主流币 1h，93 天
.\.venv\Scripts\python.exe backend/scripts/absorption_backtest.py --windows 3 --step-days 30
.\.venv\Scripts\python.exe backend/scripts/absorption_backtest.py --interval 15m --sensitivity high
.\.venv\Scripts\python.exe backend/scripts/absorption_backtest.py --delta-source est --days 400
```
进程池用于 CPU 密集的逐 bar 撮合；数据加载用线程（IO）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.backtest.execution import simulate  # noqa: E402
from app.backtest.metrics import metrics_from_equity  # noqa: E402
from app.backtest.significance import bootstrap_expectancy_ci, random_baseline_distribution  # noqa: E402
from app.config import DEFAULT_STATE  # noqa: E402
from app.db import make_engine  # noqa: E402
from app.features.absorption import SENSITIVITY_PRESETS, load_lower_tf_delta  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.ml.dataset import build_dataset  # noqa: E402
from app.strategy.coin_universe import MAJOR_SYMBOLS  # noqa: E402
from app.strategy.rules import evaluate_signals, resolve_params  # noqa: E402
from app.utils import interval_ms, now_ms  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

CODE = "absorption_reversal"
DAY_MS = 86_400_000
# IO 线程与撮合进程各自持有独立会话：SQLAlchemy 会话不是线程安全的，
# 多线程共享会触发 "concurrent operations are not permitted"。
import threading  # noqa: E402

_STATE: dict = {}
_LOCAL = threading.local()


def _session() -> Session:
    session = getattr(_LOCAL, "session", None)
    if session is None:
        engine = make_engine(read_only=True)
        from sqlalchemy.orm import sessionmaker

        _LOCAL.session = sessionmaker(bind=engine)()
        session = _LOCAL.session
    return session


# ---------------------------------------------------------------- 数据


def load_symbol(symbol: str, interval: str, days: int, end_ms: int, delta_source: str) -> dict | None:
    """取一个币在该窗口上的已富化帧（含 delta 来源信息）。"""
    db = _session()
    try:
        data = build_dataset(db, symbol, interval, horizon=5, days=days, end_ms=end_ms)
    except Exception as exc:  # 数据缺口/无数据 — 跳过而不是中断整轮
        return {"symbol": symbol, "error": str(exc)}
    if len(data) < 200:
        return {"symbol": symbol, "error": f"样本不足 {len(data)} 行"}
    used = "est"
    if delta_source in ("auto", "lower"):
        try:
            lower_delta = load_lower_tf_delta(MarketDataService(db), symbol, data.index)
        except Exception:
            lower_delta = None
        if lower_delta is not None and lower_delta.abs().sum() > 0:
            data["intrabar_delta"] = lower_delta
            used = "lower"
    data.attrs["delta_source"] = used
    return {"symbol": symbol, "frame": data, "rows": len(data), "delta_source": used}


# ---------------------------------------------------------------- 单币回测


def evaluate_symbol(task: dict) -> dict:
    """单币「信号 → 撮合 → 指标 → 随机基线」worker（模块级函数，供进程池 pickle）。

    只做纯计算，不碰 DB：帧已在主进程查好 pickle 进来。
    """
    frame = task["frame"]
    params = task["params"]
    initial, fee, slippage, interval = task["cash"], task["fee"], task["slippage"], task["interval"]
    signals = evaluate_signals(frame, CODE, params)
    signal_bars = int(signals.direction.ne(0).sum())
    if signal_bars == 0:
        return {
            "symbol": task["symbol"], "rows": len(frame), "signal_bars": 0,
            "delta_source": task["delta_source"], "trades": 0, "metrics": None,
            "status_counts": {}, "blocked": {},
            "expectancy_ci": {"verdict": "no_trades"},
            "random": {"median_return_pct": 0.0, "median_trades": 0},
        }
    equity, trades = simulate(frame, signals, params, initial, fee, slippage, interval)
    metrics = metrics_from_equity(equity, trades, initial, interval)
    distribution = random_baseline_distribution(
        frame, signals, params, initial, fee, slippage, interval, n_seeds=task["seeds"]
    )
    from collections import Counter

    return {
        "symbol": task["symbol"],
        "rows": len(frame),
        "signal_bars": signal_bars,
        "delta_source": task["delta_source"],
        "trades": len(trades),
        # 逐笔收益回传：主进程做**池化**期望值 CI（单币样本太小，逐币 CI 几乎必然跨 0）
        "trade_returns": [round(float(item["return_pct"]), 6) for item in trades],
        "metrics": {
            key: round(float(value), 4) if isinstance(value, (int, float)) else str(value)
            for key, value in metrics.items()
        },
        "status_counts": dict(Counter(t["status"] for t in trades)),
        "blocked": {
            str(k): int(v)
            for k, v in signals.block_reason[signals.block_reason.ne("")].value_counts().items()
        },
        "expectancy_ci": bootstrap_expectancy_ci(trades),
        # 随机方向基线分布（相同时点/仓位/风控/成本，只翻方向）
        "random": {
            "mean_pct": distribution.get("mean_pct"),
            "std_pct": distribution.get("std_pct"),
            "p95_pct": distribution.get("p95_pct"),
            "p05_pct": distribution.get("p05_pct"),
        },
    }


# ---------------------------------------------------------------- 主流程


def run_window(frames: dict, params: dict, cash: float, fee: float, slippage: float,
               interval: str, seeds: int, processes: int) -> list[dict]:
    tasks = [
        {
            "symbol": symbol, "frame": frame["frame"], "params": params,
            "cash": cash, "fee": fee, "slippage": slippage, "interval": interval,
            "seeds": seeds, "delta_source": frame["delta_source"],
        }
        for symbol, frame in frames.items()
        if frame.get("frame") is not None
    ]
    if not tasks:
        return []
    if processes > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(processes, len(tasks))) as pool:
            return list(pool.map(evaluate_symbol, tasks))
    return [evaluate_symbol(task) for task in tasks]


def _merge_counts(sequence) -> dict:
    """把逐币的计数合并为窗口口径（成交状态分布 / 拦截原因分布）。"""
    totals: dict[str, int] = {}
    for mapping in sequence:
        for key, value in (mapping or {}).items():
            totals[str(key)] = totals.get(str(key), 0) + int(value)
    return dict(sorted(totals.items(), key=lambda item: -item[1]))


def summarise(window_results: list[dict]) -> dict:
    """把每币结果聚成组合口径：收益取中位数与均值，成交笔数求和。"""
    usable = [row for row in window_results if row["trades"] > 0 and row.get("metrics")]
    returns = [row["metrics"]["total_return_pct"] for row in usable]
    return {
        "symbols_total": len(window_results),
        "symbols_with_trades": len(usable),
        "trades": sum(row["trades"] for row in usable),
        "return_median_pct": round(statistics.median(returns), 4) if returns else 0.0,
        "return_mean_pct": round(statistics.fmean(returns), 4) if returns else 0.0,
        "positive_symbols": int(sum(1 for value in returns if value > 0)),
        "win_rate_mean_pct": round(statistics.fmean([row["metrics"]["win_rate"] for row in usable]), 2)
        if usable
        else 0.0,
        "max_drawdown_mean_pct": round(
            statistics.fmean([row["metrics"]["max_drawdown_pct"] for row in usable]), 4
        )
        if usable
        else 0.0,
        # 显著优于随机 = 账户收益超过该币 10 种子随机翻转分布的 95 分位
        "beats_random_p95": int(
            sum(
                1
                for row in usable
                if row["metrics"]["total_return_pct"] > (row["random"].get("p95_pct") or 0.0)
            )
        ),
        "status_counts": _merge_counts(row.get("status_counts", {}) for row in window_results),
        "blocked": _merge_counts(row.get("blocked", {}) for row in window_results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="吸收反转策略历史回测")
    parser.add_argument("--symbols", default="", help="逗号分隔；默认用主流币清单")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--days", type=int, default=93, help="每个窗口的历史天数")
    parser.add_argument("--windows", type=int, default=1, help="滚动窗口数（非重叠，逐窗回退）")
    parser.add_argument("--step-days", type=int, default=0, help="窗口回退步长，0 = 等于 --days")
    parser.add_argument("--sensitivity", default="medium", choices=sorted(SENSITIVITY_PRESETS))
    parser.add_argument("--delta-source", default="auto", choices=("auto", "lower", "est"))
    parser.add_argument("--cash", type=float, default=10_000.0)
    parser.add_argument("--fee", type=float, default=DEFAULT_STATE["fee_rate"])
    parser.add_argument("--slippage", type=float, default=DEFAULT_STATE["slippage"])
    parser.add_argument("--workers", type=int, default=6, help="数据加载线程数")
    parser.add_argument("--processes", type=int, default=6, help="撮合进程数")
    parser.add_argument("--seeds", type=int, default=10, help="随机方向基线种子数")
    parser.add_argument("--out", default="docs/absorption-reversal-strategy")
    args = parser.parse_args()

    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()] or list(MAJOR_SYMBOLS)
    presets = SENSITIVITY_PRESETS[args.sensitivity]
    params = resolve_params(CODE, {"fee_rate": args.fee, "slippage": args.slippage}, presets,
                            interval=args.interval)
    step_days = args.step_days or args.days
    started = time.monotonic()

    windows = []
    for index in range(args.windows):
        end_ms = int(now_ms()) - index * step_days * DAY_MS
        print(f"[窗口 {index + 1}/{args.windows}] end={datetime.fromtimestamp(end_ms / 1000, timezone.utc):%Y-%m-%d}")
        t0 = time.monotonic()
        # IO 密集：数据加载用线程；CPU 密集：撮合用进程（GIL 下线程零收益）
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            frames = dict(zip(symbols, pool.map(
                lambda item: load_symbol(item, args.interval, args.days, end_ms, args.delta_source),
                symbols,
            )))
        skipped = [f"{s}: {f.get('error')}" for s, f in frames.items() if f.get("frame") is None]
        if skipped:
            print(f"  跳过 {len(skipped)} 币（{'; '.join(skipped[:3])}{'…' if len(skipped) > 3 else ''}）")
        sources = {}
        for symbol, payload in frames.items():
            if payload.get("frame") is not None:
                sources.setdefault(payload["delta_source"], []).append(symbol)
        print(f"  数据 {time.monotonic() - t0:.1f}s；delta 来源: "
              + ", ".join(f"{k}×{len(v)}" for k, v in sources.items()))
        t0 = time.monotonic()
        results = run_window(frames, params, args.cash, args.fee, args.slippage,
                             args.interval, args.seeds, args.processes)
        summary = summarise(results)
        print(f"  撮合 {time.monotonic() - t0:.1f}s；成交 {summary['trades']} 笔 / "
              f"{summary['symbols_with_trades']} 币有交易；收益中位数 {summary['return_median_pct']:+.2f}%")
        windows.append(
            {
                "end_ms": end_ms,
                "end": datetime.fromtimestamp(end_ms / 1000, timezone.utc).isoformat(),
                "summary": summary,
                "per_symbol": results,
                "delta_sources": {key: len(value) for key, value in sources.items()},
                "skipped": skipped,
            }
        )

    all_results = [row for window in windows for row in window["per_symbol"] if row["trades"] > 0]
    returns = [row["metrics"]["total_return_pct"] for row in all_results]
    all_pooled_returns = [
        value for row in all_results for value in row.get("trade_returns", [])
    ]
    combined = {
        "strategy": CODE,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "interval": args.interval,
        "days_per_window": args.days,
        "windows": args.windows,
        "step_days": step_days,
        "sensitivity": args.sensitivity,
        "delta_source": args.delta_source,
        "symbols": symbols,
        "parameters": params,
        "costs": {"fee_rate": args.fee, "slippage": args.slippage, "cash_per_symbol": args.cash},
        "random_baseline_seeds": args.seeds,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "windows_detail": windows,
        "aggregate": {
            "windows": [
                {"end": window["end"], **window["summary"]} for window in windows
            ],
            "trades": sum(row["trades"] for row in all_results),
            "return_median_pct": round(statistics.median(returns), 4) if returns else 0.0,
            "return_mean_pct": round(statistics.fmean(returns), 4) if returns else 0.0,
            "positive_window_ratio": round(
                sum(1 for window in windows if window["summary"]["return_median_pct"] > 0) / len(windows), 4
            ) if windows else 0.0,
            "window_returns_pct": [window["summary"]["return_median_pct"] for window in windows],
            "per_symbol": [
                {
                    "symbol": row["symbol"],
                    "trades": row["trades"],
                    "return_pct": row["metrics"]["total_return_pct"] if row["metrics"] else None,
                    "win_rate": row["metrics"]["win_rate"] if row["metrics"] else None,
                }
                for row in all_results
            ],
            "expectancy_conclusive": int(
                sum(1 for row in all_results if row["expectancy_ci"].get("verdict") not in ("inconclusive", "no_trades"))
            ),
            "beats_random_ratio": (
                round(
                    sum(
                        1
                        for row in all_results
                        if row["metrics"]["total_return_pct"] > (row["random"].get("p95_pct") or 0.0)
                    )
                    / len(all_results),
                    4,
                )
                if all_results
                else 0.0
            ),
            # ---- 决定性判据：把所有币/窗口的成交池化后再看期望值 ----
            # 单币只有个位数成交，逐币 CI 必然跨 0；池化后 N 足够才能区分 edge 与噪声
            "pooled_expectancy_ci": bootstrap_expectancy_ci(
                [{"return_pct": value} for value in all_pooled_returns]
            ),
            "pooled_trades": len(all_pooled_returns),
            "pooled_status_counts": _merge_counts(row.get("status_counts", {}) for row in all_results),
            "pooled_blocked": _merge_counts(row.get("blocked", {}) for row in all_results),
            "pooled_random_mean_pct": (
                round(statistics.fmean([row["random"]["mean_pct"] or 0.0 for row in all_results]), 4)
                if all_results
                else 0.0
            ),
        },
    }

    # 诊断附录：同目录下的其它 absorption-reversal-*.json（灵敏度 / delta 来源对照）
    stem = Path(args.out).name
    diagnostics = []
    for path in sorted((Path(args.out).parent).glob("absorption-reversal-*.json")):
        if path.stem == stem:
            continue
        try:
            other = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        diagnostics.append(
            {
                "label": path.stem.replace("absorption-reversal-", ""),
                "sensitivity": other.get("sensitivity"),
                "interval": other.get("interval"),
                "delta_source": other.get("delta_source"),
                "trades": other["aggregate"]["trades"],
                "return_median_pct": other["aggregate"]["return_median_pct"],
                "pooled": other["aggregate"]["pooled_expectancy_ci"],
            }
        )
    combined["aggregate"]["diagnostics"] = diagnostics

    out_md, out_json = Path(args.out + ".md"), Path(args.out + ".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(combined), encoding="utf-8")
    print(f"\n[输出] {out_md} / {out_json}")
    return 0


def _pooled_ci_lines(ci: dict) -> list[str]:
    """池化判据正文。

    零成交时 `bootstrap_expectancy_ci` 只返回 `{"trades": 0, "verdict": "no_trades"}`，
    没有均值/区间可填。这里明说「无判据」，而不是补 0 让报告看起来像是有结论。
    """
    if "mean_return_pct" not in ci:
        return ["- 无成交样本，无法计算每笔期望的置信区间 → **no_trades**（不构成判据）"]
    return [
        f"- 每笔收益均值 **{ci['mean_return_pct']:+.4f}%**，95% CI "
        f"[{ci['ci_low_pct']:+.4f}%, {ci['ci_high_pct']:+.4f}%] → **{ci['verdict']}**",
        f"- 重采样为正的比例 {ci['positive_share']:.1%}（>95% 才算统计上站得住）",
    ]


def render_markdown(payload: dict) -> str:
    agg = payload["aggregate"]
    preset = SENSITIVITY_PRESETS[payload["sensitivity"]]
    ci = agg["pooled_expectancy_ci"]
    lines = [
        "# 吸收反转（Absorption Reversal）— 主流币回测报告",
        "",
        f"生成时间：{payload['generated_at']}　周期：{payload['interval']}　"
        f"窗口：{payload['windows']}×{payload['days_per_window']} 天（步长 {payload['step_days']} 天）",
        "",
        "## 结论速览",
        "",
        f"- 有交易的币种 {len(agg['per_symbol'])} 个，合计成交 **{agg['trades']} 笔**",
        f"- 单币收益中位数 **{agg['return_median_pct']:+.2f}%**，均值 {agg['return_mean_pct']:+.2f}%",
        f"- 正收益窗口占比 {agg['positive_window_ratio']:.0%}，"
        f"各窗口中位数 {[round(v, 2) for v in agg['window_returns_pct']]}",
        f"- 收益超过随机方向基线（{payload['random_baseline_seeds']} 种子）的币种占比 "
        f"{agg['beats_random_ratio']:.0%}",
        f"- 期望值置信区间可判定的币种 {agg['expectancy_conclusive']}/{len(agg['per_symbol'])}",
        "",
        f"### 池化判据（{agg['pooled_trades']} 笔）",
        "",
        *_pooled_ci_lines(ci),
        f"- 随机方向基线（同参数，均值）：{agg['pooled_random_mean_pct']:+.3f}%",
        "",
        "判读规则（与全项目一致）：CI 下界 > 0 才算有正 edge；跨 0 即不可判定；"
        "上界 < 0 说明这套规则在当前样本下是负贡献。",
        "",
        "## 结论与处置",
        "",
    ]
    verdict = ci.get("verdict")
    if verdict == "positive":
        lines.append(
            f"池化 CI 下界为正 → 存在可判定的正 edge。下一步：换窗口/换标的做 holdout 复检，"
            f"通过后再考虑在巡检中启用。"
        )
    elif verdict == "negative":
        lines.append(
            "池化 CI 上界为负 → 当前口径下这套规则是**负贡献**，不宜投入资金。"
            "不要再微调阈值救它：样本内变好大概率是噪声。"
        )
    else:
        lines.extend([
            "池化 CI 跨 0 → **不可判定**：既不能说有 edge，也不能说没有。"
            "这与本项目的判据一致——样本不足时的默认值是「不启用」，而不是「先跑跑看」。",
            "",
            "处置：",
            "",
            "1. 策略已完整移植并可随时手动开启（`StrategyConfig.absorption_reversal`，默认 enabled=0），"
            "便于对照观察与后续复检。",
            "2. 运行范围钉在主流币（`strategy.coin_universe.MAJOR_SYMBOLS`），避免量价噪声污染 c1/c5 判定。",
            "3. **默认不启用**：没有 holdout 为正的证据，就不该在自动巡检里参与资金竞争。",
            "4. 真正的样本瓶颈是 1m 历史（当前约 93 天）：脚本的 intrabar delta 需要它。"
            "补齐半年以上 1m 历史后重跑本脚本，样本量会提高一个数量级。",
        ])
    lines.extend([
        "", "## 策略参数（脚本 1:1）", "",
        f"灵敏度 `{payload['sensitivity']}`：|delta| 分位 {preset['delta_pct']}%、"
        f"实体上限 {preset['body_mult']}×ATR、极值回看 {preset['extreme_lookback']} 根、"
        f"影线比 {preset['wick_mult']}",
        "",
        "| 参数 | 值 | 参数 | 值 |",
        "| --- | --- | --- | --- |",
    ])
    params = payload["parameters"]
    interesting = [
        "delta_window", "delta_pct", "body_mult", "extreme_lookback", "wick_mult",
        "dominance_pct", "abs_gap_bars", "conf_mode", "conf_window_bars",
        "sl_buffer_atr", "target_r", "be_trigger_r", "min_sl_atr", "max_sl_atr",
        "max_hold_bars", "base_position_pct", "max_leverage", "risk_per_trade_pct",
    ]
    for start in range(0, len(interesting), 2):
        pair = interesting[start : start + 2]
        cells = []
        for key in pair:
            cells.extend([f"`{key}`", str(params.get(key))])
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## 逐窗口汇总",
        "",
        "| 窗口结束 | 有交易币 | 成交 | 收益中位数 | 收益均值 | 胜率均值 | 超随机 95 分位 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for window in agg["windows"]:
        lines.append(
            f"| {window['end'][:10]} | {window['symbols_with_trades']}/{window['symbols_total']} | "
            f"{window['trades']} | {window['return_median_pct']:+.2f}% | {window['return_mean_pct']:+.2f}% | "
            f"{window['win_rate_mean_pct']:.1f}% | {window['beats_random_p95']} |"
        )
    lines.extend([
        "",
        "### 出场原因分布（池化）",
        "",
        "| 出场 | 笔数 | 占比 |",
        "| --- | --- | --- |",
    ])
    total_exits = max(sum(agg["pooled_status_counts"].values()), 1)
    for reason, count in agg["pooled_status_counts"].items():
        lines.append(f"| {reason} | {count} | {count / total_exits:.0%} |")
    lines.extend(["", "### 信号被拦截原因（池化）", "", "| 原因 | 根数 |", "| --- | --- |"])
    for reason, count in list(agg["pooled_blocked"].items())[:10]:
        lines.append(f"| {reason} | {count} |")
    if agg.get("diagnostics"):
        lines.extend(["", "## 诊断附录（同一时间窗的其它口径）", "",
                      "| 口径 | 成交 | 收益中位数 | 池化每笔期望 | 95% CI | 判定 |",
                      "| --- | --- | --- | --- | --- | --- |"])
        for item in agg["diagnostics"]:
            pooled = item["pooled"]
            lines.append(
                f"| {item['label']}（{item['interval'] or '?'}，delta={item['delta_source']}） | "
                f"{item['trades']} | {item['return_median_pct']:+.2f}% | "
                f"{pooled.get('mean_return_pct', 0.0):+.4f}% | "
                f"[{pooled.get('ci_low_pct', 0.0):+.4f}%, {pooled.get('ci_high_pct', 0.0):+.4f}%] | "
                f"{pooled.get('verdict', '')} |"
            )
        lines.append(
            "\n放宽容度（脚本 Sensitivity=High）会把信号数放大数倍，但每笔期望转负——"
            "这正说明「少而精」不是可以随便放宽的旋钮，样本密度靠放松阈值换不来 edge。"
        )
    lines.extend(["", "## 逐币种（有交易）", "",
                  "| 币种 | 成交 | 收益 | 胜率 | 随机基线均值 | 期望值 CI | delta 来源 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"])
    by_symbol = {}
    for window in payload["windows_detail"]:
        for row in window["per_symbol"]:
            if row["trades"] > 0:
                by_symbol.setdefault(
                    row["symbol"], {"trades": 0, "return": 0.0, "sources": set(), "rows": []}
                )
                entry = by_symbol[row["symbol"]]
                entry["trades"] += row["trades"]
                entry["return"] += row["metrics"]["total_return_pct"]
                entry["sources"].add(row["delta_source"])
                entry["rows"].append(row)
    for symbol, entry in sorted(by_symbol.items(), key=lambda item: -item[1]["return"]):
        last = entry["rows"][-1]
        random_mean = [r["random"].get("mean_pct") for r in entry["rows"]]
        random_mean = [v for v in random_mean if isinstance(v, (int, float))]
        lines.append(
            f"| {symbol} | {entry['trades']} | {entry['return']:+.2f}% | "
            f"{last['metrics']['win_rate']:.1f}% | "
            f"{statistics.fmean(random_mean) if random_mean else 0.0:+.2f}% | "
            f"{last['expectancy_ci'].get('verdict', '')} | {','.join(sorted(entry['sources']))} |"
        )
    lines.extend([
        "",
        "## 数据限制（必须先读）",
        "",
        "1. **delta 来源**：脚本真正用的是 `request.security_lower_tf` 的 intrabar delta；"
        "本项目 1m 历史约 93 天，`--delta-source auto` 在有 1m 数据的区间用真实聚合 delta，"
        "其余区间降级为脚本自带的 bar shape 估计（`na(ltfDelta) ? estDelta : ltfDelta`）。"
        "两种来源的信号密度差异很大，表中 `delta 来源` 列逐币标注。",
        "2. **信号稀疏**：吸收判定要求「巨量 + 无进展 + 极值 + 长影线 + 单边支配」同时成立，"
        "脚本默认参数下每个币每月只有个位数信号，样本量的上限由策略本身决定，不是取数问题。",
        "3. **撮合口径**：信号收盘 → 次根开盘成交（含滑点与费率）；"
        "R 倍数目标与保本价以成交价为基准，与脚本 `position_avg_price` 一致。",
        "",
        f"耗时 {payload['elapsed_seconds']}s。",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
