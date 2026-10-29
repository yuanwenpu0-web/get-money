r"""Supertrend 趋势反转 — 主流币种周期/参数寻优（样本内选择 + 样本外验证）。

为什么不是一个「网格搜索取最大值」的脚本
------------------------------------------
单窗口上跑 90 组参数、挑收益最高的那组，等于在噪声里挑最大值：95% 置信水平下
90 次独立检验里期望有 ~4.5 次「显著为正」纯属运气。所以本脚本把**时间**当作
第二个正交维度：

    每个连续K线段按 70% / 30% 切分（前者训练、后者测试），收益与置信区间
    分别统计。选择只看**训练段**，判定只看**测试段**。训练段最好的组合在测试段
    全部塌回去，就说明寻优本身在拟合噪声。

阶段划分
--------
- 阶段 1（结构网格）：`supertrend_factor` × `atr_period` × `sl_atr_mult` × `target_r`
  = 7 × 5 × 5 × 4 = 700 组。这一层决定「信号灵敏度 + 止损止盈几何」。
- 阶段 2（叠加层）：在阶段 1 每个周期训练段前 8 名上，叠加 9 种出场/过滤变体
  （EMA200 方向过滤、ADX 强度过滤、ATR 追踪止损、时间止损及其组合）。
  原始 spec 的 62% 止损出局说明「持有到翻转」的出场结构本身有问题，这一层就是
  针对它设计的最小改动集合。
- 阶段 3（随机基线）：每个周期最终前 3 名跑 20 个随机方向种子。
  账户收益若跑不进自身随机分布的 95 分位，说明方向判断没有信息量。

周期为什么只有 15m / 1h / 4h
----------------------------
`SUPPORTED_INTERVALS` 有 1m/5m/15m/1h/4h/1d，但一年窗口下实测覆盖：
5m 全市场 0 币、1d 全市场 0 币、15m 14/16 主流、1h 与 4h 16/16 主流。
1m/5m/1d 不是「没跑」，是**没有一年历史**，跑了也只能拿 90 天样本和一年样本
比收益，那种对比没有意义。

口径（与 `supertrend_backtest.py` 完全一致，不要在这里另立一套）
---------------------------------------------------------------
- 信号收盘确认 → 次根开盘成交；费率 + 滑点双边计；ATR 止损、R 倍数止盈、
  反向翻转出场；`simulate` 的「翻转根先出场、不再进场」门控照旧。
- 跨缺口逐段独立撮合，账户权益按段复利结转。
- `prepare` 结果按「真正影响指标输出的参数」缓存：`sl_atr_mult` / `target_r`
  不进缓存键（它们只影响止盈止损价，不影响信号帧）——但要求重入类开关全关，
  否则 `_setup_signals` 的武装状态会依赖它们，缓存就错了。脚本里断言了这点。

用法
----
```
.\.venv\Scripts\python.exe backend/scripts/supertrend_optimize.py --limit 3   # 冒烟
.\.venv\Scripts\python.exe backend/scripts/supertrend_optimize.py             # 全量
.\.venv\Scripts\python.exe backend/scripts/supertrend_optimize.py --intervals 1h,4h
```
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# 必须在 numpy/BLAS 于子进程里初始化之前设置（原因见 supertrend_backtest.py）：
# 20+ 个 worker 各自默认开 cpu_count 个 OpenBLAS 线程会直接 OOM 杀掉进程池。
for _threads_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_threads_var, "1")

import numpy as np
import pandas as pd
from sqlalchemy.orm import sessionmaker

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(SCRIPT_DIR))

import supertrend_backtest as SB  # noqa: E402  取数与分片逻辑复用，避免两套实现漂移
from app.backtest.execution import exposure_metrics, randomize_directions, simulate  # noqa: E402
from app.backtest.metrics import metrics_from_equity  # noqa: E402
from app.backtest.trades import aggregate_positions  # noqa: E402
from app.backtest.optimizer import symbol_universe  # noqa: E402
from app.config import DEFAULT_STATE  # noqa: E402
from app.db import make_engine  # noqa: E402
from app.strategy.coin_universe import MAJOR_SYMBOLS  # noqa: E402
from app.strategy.rules import evaluate_signals, prepare_frame, resolve_params  # noqa: E402
from app.utils import interval_ms, now_ms  # noqa: E402

CODE = SB.CODE
TRAIN_SPLIT = 0.70          # 每段前 70% 训练、后 30% 测试
MIN_TRAIN_TRADES = 200      # 训练段池化成交门槛（低于此值不参与选优）
MIN_TEST_TRADES = 200       # 测试段池化成交门槛（低于此值不给判定）
MIN_SEGMENT_TAIL = 120      # 测试段至少这么多根，否则该段不切（整段算训练）
BOOT_CAP = 20_000           # 池化 bootstrap 的成交上限：超了确定性降采样

# 只影响**指标列**（st_atr / st_line / st_direction / st_adx / st_ema200 / 信号列）的参数。
# `sl_atr_mult` / `target_r` 不在其中：它们只进 sl_pct / tp_pct 两列，走向量化的
# `signals()` 重算，不重跑 Python 循环版的 `prepare()`。
# 注意：这要求 `prepare` 的输出与它们无关 —— 即重入类开关全关（脚本里 assert 了）。
INDICATOR_KEY_KEYS = (
    "supertrend_factor", "atr_period", "adx_length", "adx_smoothing",
    "use_adx_filter", "adx_threshold", "use_ema200_filter", "enable_breakout",
    "pivot_length",
)

# 阶段 1：结构网格（7 × 5 × 5 × 4 = 700 组）
# 上限刻意开得宽：目的是让「最优组合」真的有机会出现，而不是在 9 组里挑一个。
STAGE1_GRID = {
    "supertrend_factor": (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0),
    "atr_period": (7, 10, 14, 21, 30),
    "sl_atr_mult": (0.75, 1.0, 1.5, 2.0, 3.0),
    "target_r": (2.5, 3.0, 4.0, 6.0),
}

# 边界扩展网格：1050 组的结果显示四个维度都是**单调**的（factor 越大、止损越宽、
# 目标越远、ATR 越慢 → 收益越好），最优解顶在默认网格边界上。这一组就是往
# 「更少交易、更宽风险、更远目标」的方向再推两档，验证是真趋势还是要发散。
BOUNDARY_GRID = {
    "supertrend_factor": (4.0, 5.0, 6.0, 8.0),
    "atr_period": (21, 30, 44),
    "sl_atr_mult": (3.0, 4.0, 5.0),
    "target_r": (6.0, 8.0, 10.0),
}

# 阶段 2：叠加层（阶段 1 前 8 名上各跑一遍）
# 时间止损按**小时**表达再换算成 bar，避免「120 根」在 15m 和 4h 上差 16 倍。
STAGE2_OVERLAYS = {
    "ema200": {"use_ema200_filter": True},
    "adx30": {"use_adx_filter": True, "adx_threshold": 30.0},
    "adx20": {"use_adx_filter": True, "adx_threshold": 20.0},
    "trail12": {"trailing_enabled": True, "trailing_atr_mult": 1.2, "trail_activation_atr": 1.5},
    "trail10": {"trailing_enabled": True, "trailing_atr_mult": 1.0, "trail_activation_atr": 1.0},
    "trail20": {"trailing_enabled": True, "trailing_atr_mult": 2.0, "trail_activation_atr": 2.0},
    "hold48h": {"__hold_hours__": 48},
    "hold240h": {"__hold_hours__": 240},
    "trail12_hold48h": {"trailing_enabled": True, "trailing_atr_mult": 1.2,
                        "trail_activation_atr": 1.5, "__hold_hours__": 48},
    "ema200_trail12": {"use_ema200_filter": True, "trailing_enabled": True,
                       "trailing_atr_mult": 1.2, "trail_activation_atr": 1.5},
    "ema200_hold48h": {"use_ema200_filter": True, "__hold_hours__": 48},
    "breakout": {"enable_breakout": True},
    "breakout_trail12": {"enable_breakout": True, "trailing_enabled": True,
                         "trailing_atr_mult": 1.2, "trail_activation_atr": 1.5},
    "cooldown0": {"cooldown_bars": 0},
}


# ---------------------------------------------------------------- 组合定义


def _label(params: dict) -> str:
    return "_".join(f"{key}{value}" for key, value in params.items() if not key.startswith("__"))


def build_stage1(reference: dict, grid: dict | None = None) -> list[dict]:
    """结构网格：基准组合尽量包含（边界扩展网格里没有基准，故允许缺失）。"""
    grid = grid or STAGE1_GRID
    combos = []
    keys = list(grid)
    for factor in grid["supertrend_factor"]:
        for atr in grid["atr_period"]:
            for sl in grid["sl_atr_mult"]:
                for target in grid["target_r"]:
                    params = {"supertrend_factor": factor, "atr_period": atr,
                              "sl_atr_mult": sl, "target_r": target}
                    is_base = all(params[key] == reference[key] for key in keys)
                    combos.append({
                        "label": "基准" if is_base else _label(params),
                        "params": params,
                        "stage": 1,
                        "base_label": None,
                    })
    return combos


def build_stage2(finalists: list[dict], interval: str) -> list[dict]:
    """在阶段 1 入选组合上叠加出场/过滤变体。"""
    combos = []
    for finalist in finalists:
        base_params = dict(finalist["params"])
        for name, overlay in STAGE2_OVERLAYS.items():
            params = dict(base_params)
            hold_hours = overlay.get("__hold_hours__")
            for key, value in overlay.items():
                if key.startswith("__"):
                    continue
                params[key] = value
            if hold_hours:
                params["max_hold_bars"] = max(1, round(hold_hours * 3600_000 / interval_ms(interval)))
            combos.append({
                "label": f"{finalist['label']}__{name}",
                "params": params,
                "stage": 2,
                "base_label": finalist["label"],
                "overlay": name,
                "hold_bars": params.get("max_hold_bars", 0) if hold_hours else 0,
            })
    return combos


# ---------------------------------------------------------------- worker


def _segment_windows(frame_len: int) -> tuple[int, int]:
    """返回 (切分点, 测试段长度)。测试段太短就不切，整段当训练。"""
    cut = int(frame_len * TRAIN_SPLIT)
    tail = frame_len - cut
    if tail < MIN_SEGMENT_TAIL or cut < SB.MIN_SEGMENT_BARS // 2:
        return frame_len, 0
    return cut, tail


def _equity_return_pct(equity: pd.Series, cash: float) -> float:
    if equity.empty:
        return 0.0
    return float(equity.iloc[-1] / cash - 1.0) * 100.0


def evaluate_symbol(task: dict) -> dict:
    """一个 (币, 周期) = 一个任务：取数一次，跑完全部组合的训练/测试两段。"""
    symbol = task["symbol"]
    try:
        segments = SB.load_segments(
            symbol, task["interval"], task["days"], task["end_ms"], task["min_bars"]
        )
    except Exception as exc:
        return {"symbol": symbol, "status": "error",
                "error": f"{type(exc).__name__}: {exc}"[:200]}
    if not segments:
        return {"symbol": symbol, "status": "skipped", "reason": "无满足长度的连续K线段"}

    cash, fee, slippage = task["cash"], task["fee"], task["slippage"]
    interval = task["interval"]
    cached_key = None
    frame_list: list[pd.DataFrame] = []
    signal_list: list[pd.DataFrame] = []
    results = []

    for combo in task["combos"]:
        # combo = {"label": ..., **参数}：run_batch 已经把 params 平铺进来了
        params = {**task["base"], **{key: value for key, value in combo.items()
                                     if key != "label"}}
        indicator_key = tuple(params[key] for key in INDICATOR_KEY_KEYS)
        if indicator_key != cached_key:
            # `simulate` 逐 bar 读 atr_pct，必须喂 prepare 后的富化帧，不能喂原始 OHLCV
            frame_list = [prepare_frame(segment, CODE, params) for segment in segments]
            cached_key = indicator_key
        # sl_pct / tp_pct / 仓位比例由 `signals()` 从 atr 与 params 现算（全向量化），
        # 因此 sl_atr_mult / target_r 变化只走这一步，指标循环不重跑。
        signal_list = [evaluate_signals(frame, CODE, params) for frame in frame_list]

        ret_train: list[np.ndarray] = []
        side_train: list[np.ndarray] = []
        ret_test: list[np.ndarray] = []
        side_test: list[np.ndarray] = []
        train_factor = test_factor = 1.0
        status: Counter = Counter()
        blocked: Counter = Counter()
        bars_held: list[float] = []
        exposure: list[float] = []
        held_total = bars_total = trade_total = 0.0

        for frame, signals in zip(frame_list, signal_list):
            cut, tail = _segment_windows(len(frame))
            parts = [(frame.iloc[:cut], signals.iloc[:cut], "train")]
            if tail:
                parts.append((frame.iloc[cut:], signals.iloc[cut:], "test"))
            for sub_frame, sub_signals, window in parts:
                equity, trades = simulate(sub_frame, sub_signals, params, cash,
                                          fee, slippage, interval)
                trades = aggregate_positions(trades)
                if window == "train":
                    train_factor *= float(equity.iloc[-1]) / cash
                else:
                    test_factor *= float(equity.iloc[-1]) / cash
                if trades:
                    returns = np.asarray([t["return_pct"] for t in trades], dtype=np.float32)
                    sides = np.asarray([1 if t["side"] == "long" else -1 for t in trades],
                                       dtype=np.int8)
                    if window == "train":
                        ret_train.append(returns)
                        side_train.append(sides)
                    else:
                        ret_test.append(returns)
                        side_test.append(sides)
                    status.update(t["status"] for t in trades)
                    bars_held.extend(int(t.get("bars", 0) or 0) for t in trades)
                # 暴露/持仓根数在 execution.exposure_metrics 里，metrics_from_equity 不返回
                stats = exposure_metrics(equity, trades, interval)
                exposure.append(float(stats["exposure_pct"]))
                held_total += float(stats["avg_bars_held"]) * len(trades)
                bars_total += len(equity)
                trade_total += len(trades)
                blocked.update({
                    str(key): int(value)
                    for key, value in sub_signals.block_reason[
                        sub_signals.block_reason.ne("")
                    ].value_counts().items()
                })

        # 随机方向基线只在阶段 3 跑：相同时点、相同仓位与成本，只把方向 50/50 翻转。
        random_stats = None
        if task["seeds"]:
            samples = []
            for offset in range(task["seeds"]):
                seed = task["base_seed"] + offset * 7919
                factor = 1.0
                for frame, signals in zip(frame_list, signal_list):
                    randomized = randomize_directions(signals, seed=seed)
                    equity, _ = simulate(frame, randomized, params, cash, fee, slippage, interval)
                    factor *= float(equity.iloc[-1]) / cash
                samples.append((factor - 1) * 100)
            ordered = sorted(samples)
            random_stats = {
                "mean_pct": round(statistics.fmean(samples), 4),
                "p95_pct": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 4),
            }

        results.append({
            "label": combo["label"],
            "ret_train": np.concatenate(ret_train) if ret_train else np.zeros(0, np.float32),
            "side_train": np.concatenate(side_train) if side_train else np.zeros(0, np.int8),
            "ret_test": np.concatenate(ret_test) if ret_test else np.zeros(0, np.float32),
            "side_test": np.concatenate(side_test) if side_test else np.zeros(0, np.int8),
            "account_return_train_pct": round((train_factor - 1) * 100, 4),
            "account_return_test_pct": round((test_factor - 1) * 100, 4),
            "account_return_full_pct": round(train_factor * test_factor * 100 - 100, 4),
            "avg_bars_held": round(held_total / trade_total, 1) if trade_total else 0.0,
            "exposure_pct": round(held_total / bars_total * 100, 2) if bars_total else 0.0,
            "status_counts": dict(status),
            "blocked": dict(blocked),
            "random": random_stats if task["seeds"] else None,
        })

    # 按 label 索引：同一批任务里的组合可能来自不同的结构基线（阶段 2），
    # 用列表下标聚合会在分组之间错位。
    return {"symbol": symbol, "status": "ok",
            "bars": int(sum(len(segment) for segment in segments)),
            "segments": len(segments),
            "combos": {item["label"]: item for item in results}}


# ---------------------------------------------------------------- 池化与判定


def pooled_ci(values: np.ndarray, min_trades: int) -> dict:
    """池化每笔期望的 bootstrap CI；样本过大时确定性降采样（报告里标注）。"""
    values = values[np.isfinite(values)].astype(np.float64)
    n = len(values)
    if n == 0:
        return {"trades": 0, "verdict": "no_trades"}
    mean_full = float(values.mean())
    if n > BOOT_CAP:
        rng = np.random.default_rng(20261027)
        sample = values[rng.choice(n, size=BOOT_CAP, replace=False)]
        sampled = True
    else:
        sample = values
        sampled = False
    m = len(sample)
    rng = np.random.default_rng(20261027)
    idx = rng.integers(0, m, size=(2000, m))
    means = sample[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    verdict = "insufficient" if n < min_trades else (
        "positive" if lo > 0 else "negative" if hi < 0 else "inconclusive"
    )
    return {
        "trades": n,
        "mean_return_pct": round(mean_full, 4),
        "ci_low_pct": round(float(lo), 4),
        "ci_high_pct": round(float(hi), 4),
        "verdict": verdict,
        "bootstrap_sample": BOOT_CAP if sampled else n,
        "sampled": sampled,
    }


def _random_summary(pairs, rows) -> dict | None:
    """账户收益是否稳定超过自身随机方向分布的 95 分位（逐币比较，不能用全市场均值）。"""
    usable = [(symbol, entry) for (symbol, _), entry in zip(pairs, rows) if entry["random"]]
    if not usable:
        return None
    beats = sum(1 for _, entry in usable
                if entry["account_return_full_pct"] > entry["random"]["p95_pct"])
    return {
        "symbols": len(usable),
        "random_mean_pct": round(statistics.fmean(e["random"]["mean_pct"] for _, e in usable), 4),
        "beats_p95": beats,
        "beats_ratio": round(beats / len(usable), 4),
    }


def summarise(results: list[dict], combos: list[dict], interval: str) -> list[dict]:
    """按组合聚合各币，产出训练/测试两套口径。"""
    ok = [row for row in results if row["status"] == "ok"]
    by_label: dict[str, list[tuple[str, dict]]] = {}
    for row in ok:
        for label, entry in row["combos"].items():
            by_label.setdefault(label, []).append((row["symbol"], entry))
    summaries = []
    for combo in combos:
        pairs = sorted(by_label.get(combo["label"], []), key=lambda item: item[0])
        rows = [entry for _, entry in pairs]
        has_trades = [row for row in rows if row["ret_test"].size or row["ret_train"].size]
        train_values = np.concatenate([row["ret_train"] for row in rows]) if rows else np.zeros(0)
        test_values = np.concatenate([row["ret_test"] for row in rows]) if rows else np.zeros(0)
        train_side = np.concatenate([row["side_train"] for row in rows]) if rows else np.zeros(0)
        test_side = np.concatenate([row["side_test"] for row in rows]) if rows else np.zeros(0)
        train_account = [row["account_return_train_pct"] for row in rows]
        test_account = [row["account_return_test_pct"] for row in rows]

        def side_ci(values, sides, sign):
            picked = values[sides == sign]
            return pooled_ci(picked, MIN_TEST_TRADES) if picked.size else {"trades": 0, "verdict": "no_trades"}

        summaries.append({
            "label": combo["label"],
            "stage": combo.get("stage", 1),
            "base_label": combo.get("base_label"),
            "overlay": combo.get("overlay"),
            "params": {key: value for key, value in combo["params"].items()
                       if not key.startswith("__")},
            "interval": interval,
            "symbols_total": len(ok),
            "symbols_with_trades": len(has_trades),
            "train": {
                "trades": int(train_values.size),
                "expectancy": pooled_ci(train_values, MIN_TRAIN_TRADES),
                "long": side_ci(train_values, train_side, 1),
                "short": side_ci(train_values, train_side, -1),
                "account_median_pct": round(statistics.median(train_account), 4) if train_account else 0.0,
                "account_mean_pct": round(statistics.fmean(train_account), 4) if train_account else 0.0,
                "positive_symbols": int(sum(1 for value in train_account if value > 0)),
            },
            "test": {
                "trades": int(test_values.size),
                "expectancy": pooled_ci(test_values, MIN_TEST_TRADES),
                "long": side_ci(test_values, test_side, 1),
                "short": side_ci(test_values, test_side, -1),
                "account_median_pct": round(statistics.median(test_account), 4) if test_account else 0.0,
                "account_mean_pct": round(statistics.fmean(test_account), 4) if test_account else 0.0,
                "positive_symbols": int(sum(1 for value in test_account if value > 0)),
            },
            "account_return_full_median_pct": round(
                statistics.median([row["account_return_full_pct"] for row in rows]), 4
            ) if rows else 0.0,
            "avg_bars_held": round(statistics.fmean(
                [row["avg_bars_held"] for row in has_trades]), 1) if has_trades else 0.0,
            "exposure_pct": round(statistics.fmean(
                [row["exposure_pct"] for row in has_trades]), 2) if has_trades else 0.0,
            "status_counts": SB._merge_counts(row["status_counts"] for row in rows),
            "blocked": SB._merge_counts(row["blocked"] for row in rows),
            "per_symbol": [
                {"symbol": symbol,
                 "trades_train": int(entry["ret_train"].size),
                 "trades_test": int(entry["ret_test"].size),
                 "ret_train_pct": entry["account_return_train_pct"],
                 "ret_test_pct": entry["account_return_test_pct"],
                 "ret_full_pct": entry["account_return_full_pct"],
                 "random_mean_pct": (entry["random"] or {}).get("mean_pct"),
                 "random_p95_pct": (entry["random"] or {}).get("p95_pct")}
                for (symbol, _), entry in zip(pairs, rows)
            ],
            "random": _random_summary(pairs, rows),
        })
    return summaries


def rank_by_train(summaries: list[dict], top: int, min_trades: int = MIN_TRAIN_TRADES) -> list[dict]:
    """按训练段池化每笔期望选优（这就是「样本内」那一步，绝不能拿测试段选）。"""
    usable = [item for item in summaries
              if item["train"]["expectancy"].get("trades", 0) >= min_trades]
    usable.sort(key=lambda item: item["train"]["expectancy"].get("mean_return_pct", -99.0),
                reverse=True)
    return usable[:top]


# ---------------------------------------------------------------- 主流程


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Supertrend 主流币参数寻优")
    parser.add_argument("--intervals", default="15m,1h,4h")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 个主流币（冒烟）")
    parser.add_argument("--cash", type=float, default=10_000.0)
    parser.add_argument("--fee", type=float, default=DEFAULT_STATE["fee_rate"])
    parser.add_argument("--slippage", type=float, default=DEFAULT_STATE["slippage"])
    parser.add_argument("--processes", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=20, help="阶段 3 随机基线种子数")
    parser.add_argument("--finalists", type=int, default=8, help="进入阶段 2 的组合数")
    parser.add_argument("--stage", default="123", choices=("1", "12", "123", "4"))
    parser.add_argument("--from-report", default="", help="阶段 4：从既有报告取候选组合")
    parser.add_argument("--grid-choice", default="default", choices=("default", "boundary"),
                        help="boundary = 往「更少交易/更宽止损/更远目标」方向再推两档")
    parser.add_argument("--out", default="docs/supertrend-optimization")
    return parser.parse_args(argv)


def run_batch(symbols: list[str], interval: str, combos: list[dict], args,
              base_groups: dict[str, dict], seeds: int, processes: int,
              label: str) -> tuple[list[dict], float]:
    """按 (币, 参数基线) 分组投递任务。

    `combos` 里每组参数可能来自不同基线（阶段 2 的 finalists 各有各的结构参数），
    但同一批任务必须共用一份 `base`——所以按基线分组，各投一次进程池。
    进程池只建一次，避免 Windows spawn 的重复 import 开销。
    """
    started = time.monotonic()
    end_ms = int(now_ms())
    tasks = []
    for group_key, base in base_groups.items():
        group_combos = [combo for combo in combos if combo["group"] == group_key]
        if not group_combos:
            continue
        for symbol in symbols:
            tasks.append({
                "symbol": symbol,
                "interval": interval,
                "days": args.days,
                "end_ms": end_ms,
                "min_bars": SB.MIN_SEGMENT_BARS,
                "cash": args.cash,
                "fee": args.fee,
                "slippage": args.slippage,
                "base": base,
                "combos": [{"label": item["label"], **item["params"]} for item in group_combos],
                "seeds": seeds,
                "base_seed": 20261025,
                "group": group_key,
            })
    if not tasks:
        return [], 0.0

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=processes) as pool:
        futures = {pool.submit(evaluate_symbol, task): task for task in tasks}
        done = 0
        for future in as_completed(futures):
            task = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"symbol": task["symbol"], "status": "error",
                                "error": f"{type(exc).__name__}: {exc}"[:200],
                                "group": task.get("group")})
            done += 1
            if done % 8 == 0 or done == len(tasks):
                elapsed = time.monotonic() - started
                print(f"  [{label}] {done}/{len(tasks)} 任务 · {elapsed:.0f}s", flush=True)

    # 固定输入顺序，池化 bootstrap 才可复现
    results.sort(key=lambda row: (row.get("group") or "", row["symbol"]))
    elapsed = round(time.monotonic() - started, 1)
    return results, elapsed


def run_robustness(args, processes: int) -> int:
    """阶段 4：把既有报告里的候选组合放到**更长窗口**上重跑（跨周期稳健性检查）。

    主流程固定 365 天是为了三个周期同口径可比；但 1h / 4h 在主流币上有两年覆盖，
    用更长的窗口复核「赢家」是最便宜的抗过拟合检验：如果它只在某一年有效，
    第二年就会塌回去。
    """
    source = json.loads(Path(args.from_report).read_text(encoding="utf-8"))
    wanted = {item.strip() for item in args.intervals.split(",") if item.strip()}
    intervals = [item for item in source["intervals"] if item in wanted]
    payload = {
        "strategy": CODE,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_report": args.from_report,
        "days": args.days,
        "train_split": TRAIN_SPLIT,
        "intervals": {},
    }
    engine = make_engine(read_only=True)
    db = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)()
    for interval in intervals:
        available = set(symbol_universe(db, interval, args.days))
        symbols = [item for item in MAJOR_SYMBOLS if item in available]
        if args.limit:
            symbols = symbols[: args.limit]
        # 主报告有阶段 3（已含随机基线），边界扩展报告只有阶段 1 —— 两者都取训练段前 3
        stage3 = (source["stages"].get(interval, {}).get("stage3", {}).get("combos")
                  or source["stages"].get(interval, {}).get("stage1", {}).get("combos", []))
        if not stage3 or not symbols:
            continue
        # 取训练段前 3：与阶段 3 相同的选择口径，避免用测试段挑稳健性样本
        picked = sorted(stage3,
                        key=lambda item: item["train"]["expectancy"].get("mean_return_pct", -99.0),
                        reverse=True)[:3]
        combos = [{"label": item["label"], "params": item["params"],
                   "group": "r", "stage": 4} for item in picked]
        base = resolve_params(CODE, {"fee_rate": args.fee, "slippage": args.slippage},
                              interval=interval)
        print(f"=== {interval} · 阶段 4：{len(symbols)} 币 × {len(combos)} 组 × {args.days} 天 ===")
        results, elapsed = run_batch(symbols, interval, combos, args, {"r": base},
                                     args.seeds, processes, f"{interval} R")
        payload["intervals"][interval] = {
            "symbols": symbols,
            "elapsed_seconds": elapsed,
            "combos": summarise(results, combos, interval),
        }

    out_md, out_json = Path(args.out + ".md"), Path(args.out + ".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_robustness(payload), encoding="utf-8")
    print(f"\n[输出] {out_md} / {out_json}")
    return 0


def render_robustness(payload: dict) -> str:
    lines = [
        "# Supertrend 寻优 — 长窗口稳健性复核",
        "",
        f"生成时间：{payload['generated_at']}　窗口：**{payload['days']} 天**　"
        f"来源：`{payload['source_report']}`",
        "",
        "候选组合按**训练段**前 3 名取（与主报告同一选择口径），"
        "换上更长窗口后重新做 70/30 切分。",
        "",
    ]
    for interval, block in payload["intervals"].items():
        lines.extend([f"## {interval}（{len(block['symbols'])} 币）", "",
                      "| 组合 | 训练段每笔期望 | 测试段每笔期望 | 测试段 CI | 判定 | 测试段账户中位 |",
                      "| --- | --- | --- | --- | --- | --- |"])
        for item in block["combos"]:
            test = item["test"]["expectancy"]
            if "mean_return_pct" not in test:
                continue
            lines.append(
                f"| `{item['label']}` | "
                f"{item['train']['expectancy'].get('mean_return_pct', 0):+.4f}% | "
                f"{test['mean_return_pct']:+.4f}% | "
                f"[{test['ci_low_pct']:+.4f}, {test['ci_high_pct']:+.4f}] | "
                f"{test['verdict']} | {item['test']['account_median_pct']:+.2f}% |")
        lines.append("")
    lines.extend([
        "## 读法",
        "",
        "长窗口上仍然 positive 才算稳健；窗口变长后期望变差、CI 变宽跨 0，"
        "说明原窗口的「最优」是拟合。注意长窗口包含更多不同市场状态，"
        "样本内（训练段）也会随之变化，因此这里只看**测试段**。",
        "",
    ])
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    processes = args.processes or max(1, min(24, (os.cpu_count() or 4) - 2))
    if args.stage == "4":
        if not args.from_report:
            raise SystemExit("阶段 4 需要 --from-report 指向主报告 json")
        return run_robustness(args, processes)
    intervals = [item.strip() for item in args.intervals.split(",") if item.strip()]

    engine = make_engine(read_only=True)
    db = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)()
    symbols_all: dict[str, list[str]] = {}
    for interval in intervals:
        available = set(symbol_universe(db, interval, args.days))
        picked = [item for item in MAJOR_SYMBOLS if item in available]
        missing = [item for item in MAJOR_SYMBOLS if item not in available]
        if missing:
            print(f"  [{interval}] 主流池覆盖不足、已跳过: {', '.join(missing)}")
        symbols_all[interval] = picked[: args.limit] if args.limit else picked

    print(f"[Supertrend 参数寻优] 周期={intervals} · {processes} 进程 · "
          f"训练/测试切分 {TRAIN_SPLIT:.0%}/{1 - TRAIN_SPLIT:.0%}")

    report = {
        "strategy": CODE,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "days": args.days,
        "intervals": intervals,
        "universe": {
            "kind": "major",
            "members": [item for item in MAJOR_SYMBOLS],
            "per_interval": {key: value for key, value in symbols_all.items()},
        },
        "costs": {"fee_rate": args.fee, "slippage": args.slippage,
                  "cash_per_symbol": args.cash},
        "protocol": {
            "train_split": TRAIN_SPLIT,
            "min_train_trades": MIN_TRAIN_TRADES,
            "min_test_trades": MIN_TEST_TRADES,
            "stage1_grid": {key: list(value) for key, value in
                            (BOUNDARY_GRID if args.grid_choice == "boundary" else STAGE1_GRID).items()},
            "grid_choice": args.grid_choice,
            "stage2_overlays": list(STAGE2_OVERLAYS),
            "finalists_per_interval": args.finalists,
            "random_seeds": args.seeds,
            "bootstrap_cap": BOOT_CAP,
        },
        "stages": {},
        "elapsed_seconds": {},
    }

    for interval in intervals:
        symbols = symbols_all[interval]
        if not symbols:
            continue
        base = resolve_params(CODE, {"fee_rate": args.fee, "slippage": args.slippage},
                              interval=interval)
        # 重入开关必须全关：否则 `_setup_signals` 的武装状态会依赖 sl_atr_mult/target_r，
        # 而缓存键不含这两个参数（只缓指标帧），那样缓存会串味。
        assert not any(base[key] for key in
                       ("enable_tp3_reentry", "enable_ema_reentry", "enable_aggressive_ema_reentry"))

        grid = BOUNDARY_GRID if args.grid_choice == "boundary" else STAGE1_GRID
        stage1 = build_stage1(base, grid)
        for combo in stage1:
            combo["group"] = "structure"
        print(f"\n=== {interval} · 阶段 1：{len(symbols)} 币 × {len(stage1)} 组结构参数 ===")
        results, elapsed = run_batch(symbols, interval, stage1,
                                     args, {"structure": base}, 0, processes,
                                     f"{interval} S1")
        summaries = summarise(results, stage1, interval)
        report["stages"].setdefault(interval, {})["stage1"] = {
            "elapsed_seconds": elapsed,
            "combos": summaries,
            "skipped": [f"{row['symbol']}: {row.get('reason') or row.get('error')}"
                        for row in results if row["status"] != "ok"],
        }
        report["elapsed_seconds"][f"{interval}_s1"] = elapsed
        top = rank_by_train(summaries, args.finalists)
        best_train = top[0] if top else None
        if best_train:
            print(f"  [{interval}] 训练段最佳 {best_train['label']} "
                  f"期望 {best_train['train']['expectancy'].get('mean_return_pct', 0):+.4f}%/笔 "
                  f"→ 测试段 {best_train['test']['expectancy'].get('mean_return_pct', 0):+.4f}%/笔")
        if args.stage == "1":
            continue

        # ---------- 阶段 2：叠加层 ----------
        groups: dict[str, dict] = {}
        stage2: list[dict] = []
        for index, finalist in enumerate(top):
            group_key = f"g{index}"
            params = {key: value for key, value in finalist["params"].items()}
            groups[group_key] = resolve_params(
                CODE, {"fee_rate": args.fee, "slippage": args.slippage},
                {**params}, interval=interval)
            for combo in build_stage2([{"label": finalist["label"], "params": params}], interval):
                combo["group"] = group_key
                stage2.append(combo)
        print(f"=== {interval} · 阶段 2：{len(symbols)} 币 × {len(stage2)} 组叠加变体 ===")
        results2, elapsed2 = run_batch(symbols, interval, stage2,
                                       args, groups, 0, processes, f"{interval} S2")
        summaries2 = summarise(results2, stage2, interval)
        report["stages"][interval]["stage2"] = {
            "elapsed_seconds": elapsed2,
            "combos": summaries2,
        }
        report["elapsed_seconds"][f"{interval}_s2"] = elapsed2
        if args.stage == "12":
            continue

        # ---------- 阶段 3：随机基线 ----------
        top2 = rank_by_train(summaries2, 3)
        if not top2:
            continue
        groups3: dict[str, dict] = {}
        stage3: list[dict] = []
        for index, finalist in enumerate(top2):
            group_key = f"f{index}"
            source = next(item for item in stage2 if item["label"] == finalist["label"])
            groups3[group_key] = resolve_params(
                CODE, {"fee_rate": args.fee, "slippage": args.slippage},
                {**source["params"], "max_hold_bars": source["params"].get("max_hold_bars", 0)},
                interval=interval)
            stage3.append({"label": finalist["label"], "params": source["params"],
                           "group": group_key, "stage": 3,
                           "base_label": source.get("base_label"),
                           "overlay": source.get("overlay")})
        print(f"=== {interval} · 阶段 3：{len(symbols)} 币 × {len(stage3)} 组 × {args.seeds} 随机种子 ===")
        results3, elapsed3 = run_batch(symbols, interval, stage3,
                                       args, groups3, args.seeds, processes, f"{interval} S3")
        # 随机基线在 worker 里没跑（阶段 1/2 不需要），这里用已保存的成交重建不了，
        # 所以阶段 3 直接把 evaluate_symbol 的 seeds 参数用上——见 _random_stats。
        summaries3 = summarise(results3, stage3, interval)
        report["stages"][interval]["stage3"] = {
            "elapsed_seconds": elapsed3,
            "combos": summaries3,
        }
        report["elapsed_seconds"][f"{interval}_s3"] = elapsed3

    out_md, out_json = Path(args.out + ".md"), Path(args.out + ".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"\n[输出] {out_md} / {out_json}")
    return 0


# ---------------------------------------------------------------- 报告


def _fmt_ci(entry: dict) -> str:
    if "mean_return_pct" not in entry:
        return "无成交"
    return (f"{entry['mean_return_pct']:+.4f}% "
            f"[{entry['ci_low_pct']:+.4f}, {entry['ci_high_pct']:+.4f}] {entry['verdict']}")


def render_markdown(report: dict) -> str:
    lines = [
        "# Supertrend 趋势反转 — 主流币周期与参数寻优",
        "",
        f"生成时间：{report['generated_at']}　窗口：{report['days']} 天　"
        f"周期：{', '.join(report['intervals'])}",
        "",
        "## 方法：时间切分，不是单窗口取最大值",
        "",
        f"每个连续K线段按 **{report['protocol']['train_split']:.0%} / "
        f"{1 - report['protocol']['train_split']:.0%}** 切分：前段训练、后段测试。",
        "参数选择**只看训练段**（池化每笔期望），判定**只看测试段**。",
        "",
        f"- 阶段 1 结构网格 {len(_grid_combos(report['protocol']['stage1_grid']))} 组："
        f"`supertrend_factor` × `atr_period` × `sl_atr_mult` × `target_r`",
        f"- 阶段 2 出场/过滤叠加 {len(report['protocol']['stage2_overlays'])} 种："
        f"{', '.join(report['protocol']['stage2_overlays'])}",
        f"- 阶段 3 随机方向基线 {report['protocol']['random_seeds']} 种子",
        "",
    ]

    lines.extend(_interval_verdict_table(report))
    benchmark = report.get("benchmark")
    if benchmark:
        lines.extend(["## 同一窗口的买入持有", "",
                      "| 周期 | 币数 | 训练段买入持有(中位) | 测试段买入持有(中位) | 测试段 positive 组合 |",
                      "| --- | --- | --- | --- | --- |"])
        for interval, block in benchmark["intervals"].items():
            lines.append(
                f"| {interval} | {block['symbols']} | {block['train_median_pct']:+.2f}% | "
                f"{block['test_median_pct']:+.2f}% | {block['positive_combos']} 组 |")
        lines.extend([
            "",
            "这张表解释了两个周期的符号为什么会反：**训练段是熊市、测试段是牛市**。",
            "趋势策略在熊市里做多必亏、在牛市里做多必赚 —— 于是「哪组参数更好」实际测的是"
            "「这组参数有多偏多」，而不是它有没有信息量。这也是训练段与测试段相关系数为负的机制："
            "偏多在熊市排最后、在牛市排最前，排名整个翻转。",
            "",
        ])

    for interval in report["intervals"]:
        stage_data = report["stages"].get(interval)
        if not stage_data:
            continue
        stage1 = stage_data["stage1"]["combos"]
        best_train = rank_by_train(stage1, 1)
        lines.extend([f"## {interval}", ""])
        if best_train:
            best = best_train[0]
            lines.extend([
                "训练段最优结构参数：",
                "",
                f"- 参数 `{best['label']}`，训练段 {_fmt_ci(best['train']['expectancy'])}"
                f"（{best['train']['trades']} 笔）",
                f"- 同一组在测试段：{_fmt_ci(best['test']['expectancy'])}"
                f"（{best['test']['trades']} 笔）",
                f"- 账户收益中位：训练 {best['train']['account_median_pct']:+.2f}% → "
                f"测试 {best['test']['account_median_pct']:+.2f}%",
                "",
            ])
        diag = _transfer_diagnostics(stage1)
        if diag:
            lines.extend([
                f"样本内→样本外传导：训练段与测试段每笔期望相关 **{diag['pearson']:+.3f}**"
                f"（Spearman {diag['spearman']:+.3f}，n={diag['combos']}）；"
                f"测试段 positive {diag['positive']} 组，纯噪声下应有约 "
                f"{diag['false_positive_expected']:.0f} 组。",
                "",
            ])
        lines.extend(_top_return_table(stage1))
        lines.extend(_sensitivity_table(stage1))
        stage2 = stage_data.get("stage2", {}).get("combos")
        if stage2:
            lines.extend(["### 出场/过滤叠加效果（训练段前 8 名上）", ""])
            lines.extend(_overlay_table(stage2))
        stage3 = stage_data.get("stage3", {}).get("combos")
        if stage3:
            lines.extend(["### 最终候选（含随机基线）", ""])
            lines.extend(_final_table(stage3))

    coverage = report.get("data_coverage")
    if coverage:
        lines.extend(["## 数据覆盖上限（为什么没有「两年复核」）", "",
                      "| 周期 | 币数 | 最早K线 | 可用天数(中位) |", "| --- | --- | --- | --- |"])
        for interval, block in coverage["intervals"].items():
            lines.append(f"| {interval} | {block['symbols']} | {block['earliest']} | {block['median_days']} 天 |")
        lines.extend(["", coverage["note"], ""])

    boundary = report.get("boundary")
    if boundary:
        lines.extend(["## 边界扩展：最优解在默认网格之外", "",
                      f"默认网格的四个维度边际是**单调**的（factor 越大、止损越宽、目标越远、ATR 越慢"
                      f" → 每笔期望越好），最优点全部顶在网格边界上。于是往同一方向再推一档"
                      f"（{boundary['grid_label']}），验证这是真趋势还是随机游走的端点：", "",
                      "| 周期 | 扩展组数 | 扩展后最好全窗口收益 | 该组训练段 | 该组测试段 | 测试段判定 | 相对默认网格最好值 |",
                      "| --- | --- | --- | --- | --- | --- | --- |"])
        for interval, block in boundary["intervals"].items():
            best = block["best"]
            if not best:
                continue
            lines.append(
                f"| {interval} | {block['combos']} | {best['account_return_full_median_pct']:+.2f}% | "
                f"{best['train']['expectancy'].get('mean_return_pct', 0):+.4f}% | "
                f"{best['test']['expectancy'].get('mean_return_pct', 0):+.4f}% | "
                f"{best['test']['expectancy'].get('verdict', 'no_trades')} | "
                f"{best['account_return_full_median_pct'] - block['baseline_best']:+.2f}pp |")
        lines.extend([
            "",
            boundary["note"],
            "",
        ])

    profile = report.get("best_profile")
    if profile:
        lines.extend(["## 最高收益组合的交易画像", "",
                      f"参数：`{profile['label']}`（{profile['interval']}，全窗口 {profile['days']} 天）", "",
                      "| 指标 | 值 |", "| --- | --- |"])
        for key, value in profile["summary"].items():
            lines.append(f"| {key} | {value} |")
        lines.extend(["", "逐币：", "",
                      "| 币种 | 收益 | 成交 | 胜率 | 最大回撤 | 暴露 | 均持仓(根) |",
                      "| --- | --- | --- | --- | --- | --- | --- |"])
        for row in profile["per_symbol"]:
            lines.append(
                f"| {row['symbol']} | {row['ret_pct']:+.2f}% | {row['trades']} | "
                f"{row['win_rate']:.1f}% | {row['max_drawdown_pct']:.2f}% | "
                f"{row['exposure_pct']:.1f}% | {row['avg_bars_held']:.1f} |")
        lines.extend(["", profile["note"], ""])

    robustness = report.get("robustness")
    if robustness:
        lines.extend(["## 长窗口复核（{} 天）".format(robustness["days"]), "",
                      "主流程固定 {} 天是为了三个周期同口径可比。这里把每个周期训练段前 3 名"
                      "换到更长窗口上重跑，验证「最优」是否只在某一年成立：".format(robustness["days"]), ""])
        for interval, block in robustness["intervals"].items():
            lines.extend([f"### {interval}（{len(block['symbols'])} 币）", "",
                          "| 组合 | 训练段每笔期望 | 测试段每笔期望 | 测试段 CI | 判定 |",
                          "| --- | --- | --- | --- | --- |"])
            for item in block["combos"]:
                test = item["test"]["expectancy"]
                if "mean_return_pct" not in test:
                    continue
                lines.append(
                    f"| `{item['label']}` | "
                    f"{item['train']['expectancy'].get('mean_return_pct', 0):+.4f}% | "
                    f"{test['mean_return_pct']:+.4f}% | "
                    f"[{test['ci_low_pct']:+.4f}, {test['ci_high_pct']:+.4f}] | {test['verdict']} |")
            lines.append("")

    lines.extend([
        "## 判读规则",
        "",
        "- **positive** = 池化 CI 下界 > 0；**negative** = 上界 < 0；跨 0 = **inconclusive**。",
        f"- 池化成交少于 {report['protocol']['min_test_trades']} 笔记 **insufficient**，"
        "不构成任何结论（少笔数的 bootstrap 会给假精确区间）。",
        "- 训练段与测试段结论不一致时，一律以测试段为准；"
        "训练段好看但测试段翻负 = 拟合噪声。",
        f"- 本报告累计检验 {len(report['intervals'])} 个周期 × "
        f"{len(_grid_combos(report['protocol']['stage1_grid']))} 组结构参数 + 叠加层。"
        "95% 双侧区间下，若真 edge 为零，单侧「CI 下界 > 0」的假阳性期望是 2.5%；"
        "实测 positive 组数若低于这个期望，说明连运气都没站在策略这边。"
        "单个 positive 不足以支撑启用，必须同时满足：测试段 positive + 超随机 + 逐币一致 + 长窗口复现。",
        "",
        "## 口径",
        "",
        "- 信号收盘确认 → 次根开盘成交；双边费率与滑点；ATR 止损 / R 倍数止盈 / 反向翻转出场。",
        "- 跨缺口逐段独立撮合，账户权益按段复利结转；段长 < 800 根丢弃。",
        "- 训练段与测试段在切点处各自独立撮合，切点上的持仓被截断（影响可忽略但存在）。",
        f"- 池化成交超过 {report['protocol']['bootstrap_cap']} 笔时 bootstrap 确定性降采样，"
        "报告里标 `sampled`。",
        "",
    ])
    return "\n".join(lines)


def _grid_combos(grid: dict) -> list[tuple]:
    from itertools import product
    return list(product(*[grid[key] for key in grid]))


def _transfer_diagnostics(stage1: list[dict]) -> dict:
    """样本内→样本外是否传导：相关性 + 假阳性数量对比。

    如果参数真有优劣之分，训练段排名与测试段排名应当正相关。相关系数为负，
    说明训练段选出的「好参数」在测试段系统性更差 —— 那不是在选参数，是在选噪声。
    """
    usable = [item for item in stage1
              if "mean_return_pct" in item["test"]["expectancy"]
              and "mean_return_pct" in item["train"]["expectancy"]]
    if len(usable) < 30:
        return {}
    train = np.array([item["train"]["expectancy"]["mean_return_pct"] for item in usable])
    test = np.array([item["test"]["expectancy"]["mean_return_pct"] for item in usable])

    def ranks(values: np.ndarray) -> np.ndarray:
        order = values.argsort()
        output = np.empty(len(values))
        output[order] = np.arange(len(values))
        return output

    # insufficient 组的每笔期望是几十笔成交算出来的，噪声极大（4h 上能把均值从
    # -0.08% 抬到 +0.50%）。统计口径只取「可判定」组，不足的单独计数。
    judgeable = [item for item in usable
                 if item["test"]["expectancy"]["verdict"] != "insufficient"]
    judge_test = np.array([item["test"]["expectancy"]["mean_return_pct"] for item in judgeable])
    positives = [item for item in judgeable
                 if item["test"]["expectancy"]["verdict"] == "positive"]
    return {
        "combos": len(usable),
        "judgeable": len(judgeable),
        "pearson": round(float(np.corrcoef(train, test)[0, 1]), 3),
        "spearman": round(float(np.corrcoef(ranks(train), ranks(test))[0, 1]), 3),
        "train_mean_pct": round(float(train.mean()), 4),
        "test_mean_pct": round(float(judge_test.mean()), 4) if judgeable else 0.0,
        "test_median_pct": round(float(np.median(judge_test)), 4) if judgeable else 0.0,
        "positive": len(positives),
        "positive_train_mean_pct": round(float(np.mean([
            item["train"]["expectancy"]["mean_return_pct"] for item in positives])), 4)
        if positives else None,
        "negative": sum(1 for item in judgeable
                        if item["test"]["expectancy"]["verdict"] == "negative"),
        "false_positive_expected": round(len(judgeable) * 0.025, 1),
    }


def _interval_verdict_table(report: dict) -> list[str]:
    lines = ["## 周期判定", "",
             "| 周期 | 币数 | 可判定/总组数 | 测试段每笔期望(均值/中位) | 训练-测试相关 | negative | positive | 纯噪声期望假阳性 | 全窗口最好账户收益 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for interval in report["intervals"]:
        block = report["stages"].get(interval)
        if not block:
            continue
        stage1 = block["stage1"]["combos"]
        diag = _transfer_diagnostics(stage1)
        if not diag:
            continue
        best_full = max((item["account_return_full_median_pct"] for item in stage1), default=0.0)
        symbols = stage1[0]["symbols_total"] if stage1 else 0
        lines.append(
            f"| **{interval}** | {symbols} | {diag['judgeable']}/{diag['combos']} | "
            f"{diag['test_mean_pct']:+.4f}% / {diag['test_median_pct']:+.4f}% | "
            f"{diag['pearson']:+.3f} | {diag['negative']} | {diag['positive']} | "
            f"{diag['false_positive_expected']:.0f} | {best_full:+.2f}% |")
    lines.extend([
        "",
        "三列读法：**训练-测试相关**为负 = 训练段选出的好参数在测试段更差（选参无意义）；"
        "**测试段 positive** 少于**纯噪声期望** = 连运气都不到纯噪声水平；"
        "**全窗口最好账户收益**是网格里事后最好的那组，属于样本内上限。",
        "",
    ])
    return lines


def _top_return_table(stage1: list[dict], top: int = 5) -> list[str]:
    """收益最高的组合：直接回答「哪组赚最多」，同时把它在样本外的样子摆在旁边。"""
    usable = [item for item in stage1 if item["symbols_with_trades"] >= 8]
    if not usable:
        return []
    ranked = sorted(usable, key=lambda item: item["account_return_full_median_pct"], reverse=True)[:top]
    lines = [f"### 收益最高的 {len(ranked)} 组（全窗口账户收益中位，样本内）", "",
             "这一列是「如果一年前就选中它」的结果，**含训练段**，所以天然偏高。"
             "右边两列是同一组在各段的真实表现：", "",
             "| # | 组合 | 全窗口账户中位 | 训练段每笔期望 | 测试段每笔期望 | 测试段判定 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for index, item in enumerate(ranked, start=1):
        train = item["train"]["expectancy"]
        test = item["test"]["expectancy"]
        lines.append(
            f"| {index} | `{item['label']}` | {item['account_return_full_median_pct']:+.2f}% | "
            f"{train.get('mean_return_pct', 0):+.4f}% | {test.get('mean_return_pct', 0):+.4f}% | "
            f"{test.get('verdict', 'no_trades')} |")
    lines.append("")
    return lines


def _sensitivity_table(stage1: list[dict]) -> list[str]:
    """各维度取值对测试段每笔期望的边际均值——比看单点更能说明参数方向。"""
    usable = [item for item in stage1
              if "mean_return_pct" in item["test"]["expectancy"]
              and item["test"]["expectancy"]["verdict"] != "insufficient"]
    if not usable:
        return ["_测试段无足够成交，参数敏感性不可测。_", ""]
    keys = ["supertrend_factor", "atr_period", "sl_atr_mult", "target_r"]
    lines = ["### 参数敏感性（测试段池化每笔期望的边际均值）", "",
             "| 维度 | 取值 | 组合数 | 边际每笔期望 | 训练段边际 |", "| --- | --- | --- | --- | --- |"]
    for key in keys:
        values = sorted({item["params"][key] for item in usable})
        for value in values:
            subset = [item for item in usable if item["params"][key] == value]
            test_mean = statistics.fmean(
                item["test"]["expectancy"]["mean_return_pct"] for item in subset)
            train_mean = statistics.fmean(
                item["train"]["expectancy"].get("mean_return_pct", 0.0) for item in subset)
            lines.append(f"| `{key}` | {value} | {len(subset)} | {test_mean:+.4f}% | {train_mean:+.4f}% |")
    lines.append("")
    return lines


def _overlay_table(stage2: list[dict]) -> list[str]:
    usable = [item for item in stage2
              if "mean_return_pct" in item["test"]["expectancy"]
              and item["test"]["expectancy"]["verdict"] != "insufficient"]
    lines = ["| 基准结构 | 叠加 | 测试段每笔期望 | 测试段 CI | 判定 | 测试段账户中位 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for item in sorted(usable, key=lambda row: row["test"]["expectancy"].get("mean_return_pct", -99),
                       reverse=True):
        expectation = item["test"]["expectancy"]
        lines.append(
            f"| `{item['base_label']}` | {item['overlay']} | "
            f"{expectation.get('mean_return_pct', 0):+.4f}% | "
            f"[{expectation.get('ci_low_pct', 0):+.4f}, {expectation.get('ci_high_pct', 0):+.4f}] | "
            f"{expectation.get('verdict')} | {item['test']['account_median_pct']:+.2f}% |")
    lines.append("")
    return lines


def _final_table(stage3: list[dict]) -> list[str]:
    lines = ["| 组合 | 测试段成交 | 每笔期望 | CI | 判定 | 做多 | 做空 | 账户中位 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for item in stage3:
        expectation = item["test"]["expectancy"]
        if "mean_return_pct" not in expectation:
            continue
        lines.append(
            f"| `{item['label']}` | {expectation['trades']} | {expectation['mean_return_pct']:+.4f}% | "
            f"[{expectation['ci_low_pct']:+.4f}, {expectation['ci_high_pct']:+.4f}] | "
            f"{expectation['verdict']} | "
            f"{item['test']['long'].get('mean_return_pct', 0):+.4f}% | "
            f"{item['test']['short'].get('mean_return_pct', 0):+.4f}% | "
            f"{item['test']['account_median_pct']:+.2f}% |")
    lines.append("")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
