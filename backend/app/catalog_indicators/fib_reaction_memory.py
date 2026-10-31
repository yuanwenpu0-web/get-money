"""Fib Reaction Memory [WavesUnchained] — 用 A→B 参考腿内部的微观反应打分斐波那契位。

**参考腿**：结构性枢轴 `swing_pivot_len` 交替收集（同类型只保留更极端者），
低点→高点形成看涨腿（A 低、B 高），高点→低点形成看跌腿。腿存续期内，只有当
确认枢轴**越过 B** 时才被替换（看涨腿用更新的高点、看跌腿用更新的低点），
并用「旧 B 之后、新枢轴之前最近的反向枢轴」作为新起点 A。

**失效**：当前腿的归一化回撤（含当根极值）> `invalidation_ratio` 即腿失效并重启
枢轴收集。

**微观反应**：`ta.pivothigh/pivotlow(reaction_pivot_len)` 确认时，
反应幅度 = 枢轴价 − `lowest(low, reaction_pivot_len)`（低点镜像用 `highest`），
换算成 ATR 倍数；只保留 ≥ `min_reaction_atr` 且枢轴落在 (A, B) 内的反应。

**打分（每个斐波那契位）**：反应强度 `min(ATR倍数/strong_reaction_atr, 1)`
× 邻近度 `1 − |位置−fib|/fib_tolerance`（距离须 ≤ 半径）
× 时间权重 `1 + recency_weight×腿内进度`
× 消耗衰减 `1/(1+consumption_decay×此前测试次数)`
× 破位惩罚（当前回撤曾越过 `fib + test_tolerance` 时乘 `1 − broken_level_penalty`）。
原始分经饱和转换 `100×(1 − exp(−raw/saturation))` 得到 0..100 分。
**Hold Rate** = 未被破位的测试占比（0% / 100%，因为破位状态对本腿是全局的）。

**每档状态**（由当前回撤与「本腿最深回撤」两个单调量导出）：
`APPROACHING` 从未进入容差带 / `TESTING` 正在带内 / `REJECTED` 进过带但未破位
/ `ACCEPTED` 破位且仍在位外 / `RECLAIMED` 破位后回到带内浅侧。

**NEXT 位**：在所有「仍在当前回撤之前」的启用位里，优先取分数 ≥ `next_min_score`
的最高分者，否则回退到最近的几何位。

实现说明：源码在最后一根 bar 上整段重扫整条 A→B 腿窗口。这里改为**逐 bar 对当前
腿窗口做同样的扫描**（偏移量等价于 `ba=` 当前 bar − 确认 bar，破位乘子对当前腿是
常量），因此每个 bar 的输出都等价于 Pine 在「该 bar 作为最后一根」时的结果，
反应标记也落在各自确认 bar 上。

丢弃：全部 line/label 绘图、颜色/宽度/标签样式参数。

Source: pine-collection/indicators/market_structure/fib_reaction_memory/fib_reaction_memory.pine
License: WavesUnchained 公开脚本（随清单收录）。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import atr, boolean, flags, integer, lowest, number, ohlcv, pivothigh, pivotlow

__all__ = ["fib_reaction_memory"]

_RATIOS = (0.236, 0.382, 0.500, 0.618, 0.786, 1.000)
_NAMES = ("23.6", "38.2", "50.0", "61.8", "78.6", "100.0")


def _level_state(current: float, deepest: float, fib: float, tolerance: float) -> str:
    if current >= fib - tolerance and current <= fib + tolerance:
        return "TESTING"
    if deepest < fib - tolerance:
        return "APPROACHING"
    if deepest >= fib + tolerance:
        return "RECLAIMED" if current < fib - tolerance else "ACCEPTED"
    return "REJECTED"


def fib_reaction_memory(
    frame: pd.DataFrame,
    *,
    swing_pivot_len: int = 8,
    invalidation_ratio: float = 1.03,
    reaction_pivot_len: int = 2,
    atr_len: int = 14,
    min_reaction_atr: float = 0.20,
    strong_reaction_atr: float = 1.00,
    fib_tolerance: float = 0.035,
    recency_weight: float = 0.25,
    score_saturation: float = 1.75,
    consumption_decay: float = 0.35,
    broken_level_penalty: float = 0.5,
    max_scan_bars: int = 1500,
    show_236: bool = True,
    show_382: bool = True,
    show_500: bool = True,
    show_618: bool = True,
    show_786: bool = True,
    show_100: bool = True,
    test_tolerance: float = 0.015,
    next_min_score: float = 15.0,
    highlight_next: bool = True,
    strong_score: float = 65.0,
    medium_score: float = 35.0,
) -> pd.DataFrame:
    """返回参考腿、六档斐波那契分数/持有率/状态与 NEXT 位。"""
    swing_pivot_len = integer("swing_pivot_len", swing_pivot_len, 2, 50)
    invalidation_ratio = number("invalidation_ratio", invalidation_ratio, 1.0, 1.50)
    reaction_pivot_len = integer("reaction_pivot_len", reaction_pivot_len, 1, 10)
    atr_len = integer("atr_len", atr_len, 2, 100)
    min_reaction_atr = number("min_reaction_atr", min_reaction_atr, 0.0, 5.0)
    strong_reaction_atr = number("strong_reaction_atr", strong_reaction_atr, 0.1, 10.0)
    fib_tolerance = number("fib_tolerance", fib_tolerance, 0.005, 0.15)
    recency_weight = number("recency_weight", recency_weight, 0.0, 1.0)
    score_saturation = number("score_saturation", score_saturation, 0.25, 10.0)
    consumption_decay = number("consumption_decay", consumption_decay, 0.0, 2.0)
    broken_level_penalty = number("broken_level_penalty", broken_level_penalty, 0.0, 1.0)
    max_scan_bars = integer("max_scan_bars", max_scan_bars, 100, 4500)
    for name, value in (("show_236", show_236), ("show_382", show_382),
                        ("show_500", show_500), ("show_618", show_618),
                        ("show_786", show_786), ("show_100", show_100)):
        boolean(name, value)
    enabled = (show_236, show_382, show_500, show_618, show_786, show_100)
    test_tolerance = number("test_tolerance", test_tolerance, 0.002, 0.10)
    next_min_score = number("next_min_score", next_min_score, 0.0, 100.0)
    highlight_next = boolean("highlight_next", highlight_next)
    strong_score = number("strong_score", strong_score, 0.0, 100.0)
    medium_score = number("medium_score", medium_score, 0.0, 100.0)

    high, low, close = ohlcv(frame, "high", "low", "close")
    highs, lows = high.to_numpy(), low.to_numpy()
    size = len(frame)
    atr_values = atr(frame, atr_len).to_numpy()

    structural_ph = pivothigh(high, swing_pivot_len, swing_pivot_len).to_numpy()
    structural_pl = pivotlow(low, swing_pivot_len, swing_pivot_len).to_numpy()
    micro_ph = pivothigh(high, reaction_pivot_len, reaction_pivot_len).to_numpy()
    micro_pl = pivotlow(low, reaction_pivot_len, reaction_pivot_len).to_numpy()
    recent_low = lowest(low, reaction_pivot_len).to_numpy()
    recent_high = pd.Series(high).rolling(reaction_pivot_len, min_periods=reaction_pivot_len).max().to_numpy()

    # Reaction ATR sized at the confirmation bar (matches Pine's `...ATR[offset]`).
    high_reaction_atr = np.full(size, np.nan)
    low_reaction_atr = np.full(size, np.nan)
    for bar in range(size):
        ph = micro_ph[bar]
        if np.isfinite(ph):
            dist = ph - recent_low[bar]
            atr_now = atr_values[bar]
            if np.isfinite(dist) and np.isfinite(atr_now) and atr_now > 0:
                high_reaction_atr[bar] = dist / atr_now
        pl = micro_pl[bar]
        if np.isfinite(pl):
            dist = recent_high[bar] - pl
            atr_now = atr_values[bar]
            if np.isfinite(dist) and np.isfinite(atr_now) and atr_now > 0:
                low_reaction_atr[bar] = dist / atr_now

    out = {name: np.zeros(size, dtype=bool) for name in
           ("leg_active", "leg_new", "leg_invalidated", "reaction_high", "reaction_low")}
    for name in ("leg_a", "leg_a_bar", "leg_b", "leg_b_bar", "leg_direction", "leg_range",
                 "current_retracement", "pullback_deepest", "next_fib", "next_score"):
        out[name] = np.full(size, np.nan)
    out["next_index"] = np.full(size, np.nan)
    for index in range(6):
        out[f"score_{_NAMES[index]}"] = np.full(size, np.nan)
        out[f"hold_rate_{_NAMES[index]}"] = np.full(size, np.nan)
        out[f"price_{_NAMES[index]}"] = np.full(size, np.nan)
        out[f"state_{_NAMES[index]}"] = np.full(size, "", dtype=object)
        out[f"is_next_{_NAMES[index]}"] = np.zeros(size, dtype=bool)

    # Per-bar leg state captured by the forward pass for the reaction re-scan.
    la_arr = np.full(size, np.nan)
    lb_arr = np.full(size, np.nan)
    dir_arr = np.full(size, 0, dtype=int)
    legb_arr = np.full(size, np.nan)
    range_arr = np.full(size, np.nan)
    current_arr = np.full(size, np.nan)
    pullback_arr = np.full(size, np.nan)
    active_arr = np.zeros(size, dtype=bool)

    leg_active = False
    leg_a = leg_a_bar = leg_b = leg_b_bar = math.nan
    leg_direction = 0
    leg_generation = 0
    candidate_price = math.nan
    candidate_bar = math.nan
    candidate_type = 0
    last_high_price = last_high_bar = math.nan
    last_low_price = last_low_bar = math.nan
    pullback_deepest = 0.0

    for bar in range(size):
        pivot_bar = bar - swing_pivot_len
        previous_generation = leg_generation

        if not np.isnan(structural_ph[bar]):
            last_high_price, last_high_bar = structural_ph[bar], pivot_bar
            if not leg_active:
                if candidate_type == 0:
                    candidate_price, candidate_bar, candidate_type = structural_ph[bar], pivot_bar, 1
                elif candidate_type == 1:
                    if structural_ph[bar] > candidate_price:
                        candidate_price, candidate_bar = structural_ph[bar], pivot_bar
                else:
                    leg_a, leg_a_bar = candidate_price, candidate_bar
                    leg_b, leg_b_bar = structural_ph[bar], pivot_bar
                    leg_direction = 1
                    leg_active = True
                    leg_generation += 1
                    candidate_price, candidate_bar, candidate_type = structural_ph[bar], pivot_bar, 1
            elif leg_direction == 1 and structural_ph[bar] > leg_b:
                fresh = (np.isfinite(last_low_price) and np.isfinite(last_low_bar)
                         and last_low_bar > leg_b_bar and last_low_bar < pivot_bar)
                if fresh:
                    leg_a, leg_a_bar = last_low_price, last_low_bar
                leg_b, leg_b_bar = structural_ph[bar], pivot_bar
                leg_generation += 1

        if not np.isnan(structural_pl[bar]):
            last_low_price, last_low_bar = structural_pl[bar], pivot_bar
            if not leg_active:
                if candidate_type == 0:
                    candidate_price, candidate_bar, candidate_type = structural_pl[bar], pivot_bar, -1
                elif candidate_type == -1:
                    if structural_pl[bar] < candidate_price:
                        candidate_price, candidate_bar = structural_pl[bar], pivot_bar
                else:
                    leg_a, leg_a_bar = candidate_price, candidate_bar
                    leg_b, leg_b_bar = structural_pl[bar], pivot_bar
                    leg_direction = -1
                    leg_active = True
                    leg_generation += 1
                    candidate_price, candidate_bar, candidate_type = structural_pl[bar], pivot_bar, -1
            elif leg_direction == -1 and structural_pl[bar] < leg_b:
                fresh = (np.isfinite(last_high_price) and np.isfinite(last_high_bar)
                         and last_high_bar > leg_b_bar and last_high_bar < pivot_bar)
                if fresh:
                    leg_a, leg_a_bar = last_high_price, last_high_bar
                leg_b, leg_b_bar = structural_pl[bar], pivot_bar
                leg_generation += 1

        # --- reference leg geometry -----------------------------------------
        leg_range = abs(leg_b - leg_a) if leg_active else math.nan
        usable = leg_active and np.isfinite(leg_range) and leg_range > 0
        price = float(close.iloc[bar])
        current_retracement = ((leg_b - price) * leg_direction) / leg_range if usable else math.nan
        deep_price = lows[bar] if leg_direction == 1 else highs[bar]
        bar_deep = ((leg_b - deep_price) * leg_direction) / leg_range if usable else math.nan

        invalidated = bool(usable and np.isfinite(bar_deep) and bar_deep > invalidation_ratio)
        if invalidated:
            leg_active = False
            leg_generation += 1
            out["leg_invalidated"][bar] = True
            high_is_newest = (np.isfinite(last_high_bar)
                              and (not np.isfinite(last_low_bar) or last_high_bar > last_low_bar))
            if high_is_newest:
                candidate_price, candidate_bar, candidate_type = last_high_price, last_high_bar, 1
            elif np.isfinite(last_low_bar):
                candidate_price, candidate_bar, candidate_type = last_low_price, last_low_bar, -1
            else:
                candidate_price, candidate_bar, candidate_type = math.nan, math.nan, 0
            leg_range = math.nan
            usable = False

        new_leg = leg_generation != previous_generation
        if new_leg:
            pullback_deepest = 0.0
            out["leg_new"][bar] = True
        if leg_active and np.isfinite(leg_range) and leg_range > 0 and np.isfinite(leg_b_bar) \
                and bar > leg_b_bar and np.isfinite(bar_deep):
            pullback_deepest = max(pullback_deepest, bar_deep)

        # capture state for the per-bar reaction re-scan
        active_arr[bar] = leg_active
        la_arr[bar] = leg_a_bar
        lb_arr[bar] = leg_b_bar
        dir_arr[bar] = leg_direction
        legb_arr[bar] = leg_b
        range_arr[bar] = leg_range if leg_active else math.nan
        current_arr[bar] = current_retracement
        pullback_arr[bar] = pullback_deepest if leg_active else math.nan
        out["leg_active"][bar] = leg_active
        out["leg_a"][bar] = leg_a if leg_active else math.nan
        out["leg_a_bar"][bar] = leg_a_bar if leg_active else math.nan
        out["leg_b"][bar] = leg_b if leg_active else math.nan
        out["leg_b_bar"][bar] = leg_b_bar if leg_active else math.nan
        out["leg_direction"][bar] = leg_direction if leg_active else math.nan
        out["leg_range"][bar] = leg_range if leg_active else math.nan
        out["current_retracement"][bar] = current_retracement
        out["pullback_deepest"][bar] = pullback_deepest if leg_active else math.nan

    # ----------------------------------------------------------------------
    # Per-bar reaction memory + score: mirror Pine's last-bar scan evaluated
    # for every bar over the current A->B leg window.
    # ----------------------------------------------------------------------
    for bar in range(size):
        if not active_arr[bar]:
            continue
        rng = range_arr[bar]
        if not np.isfinite(rng) or rng <= 0:
            continue
        la = int(la_arr[bar])
        lb = int(lb_arr[bar])
        direction = int(dir_arr[bar])
        leg_b_val = legb_arr[bar]
        pullback_max = pullback_arr[bar]
        current_ret = current_arr[bar]

        raw_scores = [0.0] * 6
        test_counts = [0] * 6
        broken_counts = [0] * 6

        available = min(bar, max_scan_bars)
        for offset in range(available + 1):
            j = bar - offset
            pivot_bar = j - reaction_pivot_len
            if not (la < pivot_bar < lb):
                continue
            # historical micro HIGH reaction
            ph = micro_ph[j]
            if np.isfinite(ph):
                r_atr = high_reaction_atr[j]
                if np.isfinite(r_atr) and r_atr >= min_reaction_atr:
                    position = ((leg_b_val - ph) * direction) / rng
                    if 0.0 <= position <= 1.0:
                        strength = min(max(r_atr / strong_reaction_atr, 0.0), 1.0)
                        progress = (pivot_bar - la) / max(lb - la, 1.0)
                        temporal = 1.0 + recency_weight * min(max(progress, 0.0), 1.0)
                        for index in range(6):
                            ratio = _RATIOS[index]
                            distance = abs(position - ratio)
                            if distance > fib_tolerance:
                                continue
                            proximity = 1.0 - distance / fib_tolerance
                            decay = 1.0 / (1.0 + consumption_decay * test_counts[index])
                            was_broken = pullback_max >= ratio + test_tolerance
                            multiplier = 1.0 - broken_level_penalty if was_broken else 1.0
                            contribution = strength * proximity * temporal * decay * multiplier
                            raw_scores[index] += contribution
                            test_counts[index] += 1
                            if was_broken:
                                broken_counts[index] += 1
                            out["reaction_high"][j] = True
            # historical micro LOW reaction
            pl = micro_pl[j]
            if np.isfinite(pl):
                r_atr = low_reaction_atr[j]
                if np.isfinite(r_atr) and r_atr >= min_reaction_atr:
                    position = ((leg_b_val - pl) * direction) / rng
                    if 0.0 <= position <= 1.0:
                        strength = min(max(r_atr / strong_reaction_atr, 0.0), 1.0)
                        progress = (pivot_bar - la) / max(lb - la, 1.0)
                        temporal = 1.0 + recency_weight * min(max(progress, 0.0), 1.0)
                        for index in range(6):
                            ratio = _RATIOS[index]
                            distance = abs(position - ratio)
                            if distance > fib_tolerance:
                                continue
                            proximity = 1.0 - distance / fib_tolerance
                            decay = 1.0 / (1.0 + consumption_decay * test_counts[index])
                            was_broken = pullback_max >= ratio + test_tolerance
                            multiplier = 1.0 - broken_level_penalty if was_broken else 1.0
                            contribution = strength * proximity * temporal * decay * multiplier
                            raw_scores[index] += contribution
                            test_counts[index] += 1
                            if was_broken:
                                broken_counts[index] += 1
                            out["reaction_low"][j] = True

        next_index = None
        next_score = -1.0
        nearest_index = None
        nearest_distance = 10.0
        for index in range(6):
            ratio = _RATIOS[index]
            broken_mult = 1.0 - broken_level_penalty if pullback_max >= ratio + test_tolerance else 1.0
            score = 100.0 * (1.0 - math.exp(-raw_scores[index] / score_saturation)) * broken_mult
            out[f"score_{_NAMES[index]}"][bar] = score
            out[f"price_{_NAMES[index]}"][bar] = leg_b_val - direction * rng * ratio
            out[f"state_{_NAMES[index]}"][bar] = _level_state(
                current_ret, pullback_max, ratio, test_tolerance)
            if test_counts[index] > 0:
                out[f"hold_rate_{_NAMES[index]}"][bar] = 100.0 * (
                    1.0 - broken_counts[index] / test_counts[index])
            # NEXT selection
            if highlight_next and np.isfinite(current_ret) and enabled[index]:
                if ratio <= current_ret + test_tolerance:
                    continue
                distance = ratio - current_ret
                if distance < nearest_distance:
                    nearest_distance, nearest_index = distance, index
                if score >= next_min_score and score > next_score:
                    next_score, next_index = score, index
        if next_index is None:
            next_index = nearest_index
            next_score = raw_scores[nearest_index] if nearest_index is not None else math.nan
            if next_index is not None:
                next_score = out[f"score_{_NAMES[next_index]}"][bar]
        if next_index is not None:
            out["next_index"][bar] = next_index
            out["next_fib"][bar] = _RATIOS[next_index]
            out["next_score"][bar] = next_score
            out[f"is_next_{_NAMES[next_index]}"][bar] = True

    index = frame.index
    data: dict[str, pd.Series] = {
        "leg_active": pd.Series(out["leg_active"], index=index, dtype=bool),
        "leg_new": pd.Series(out["leg_new"], index=index, dtype=bool),
        "leg_invalidated": pd.Series(out["leg_invalidated"], index=index, dtype=bool),
        "leg_a": pd.Series(out["leg_a"], index=index),
        "leg_a_bar": pd.Series(out["leg_a_bar"], index=index).astype("Int64"),
        "leg_b": pd.Series(out["leg_b"], index=index),
        "leg_b_bar": pd.Series(out["leg_b_bar"], index=index).astype("Int64"),
        "leg_direction": pd.Series(out["leg_direction"], index=index).astype("Int64"),
        "leg_range": pd.Series(out["leg_range"], index=index),
        "current_retracement": pd.Series(out["current_retracement"], index=index),
        "pullback_deepest": pd.Series(out["pullback_deepest"], index=index),
        "reaction_high": pd.Series(out["reaction_high"], index=index, dtype=bool),
        "reaction_low": pd.Series(out["reaction_low"], index=index, dtype=bool),
        "next_index": pd.Series(out["next_index"], index=index).astype("Int64"),
        "next_fib": pd.Series(out["next_fib"], index=index),
        "next_score": pd.Series(out["next_score"], index=index),
        "strong_score": pd.Series(strong_score, index=index),
        "medium_score": pd.Series(medium_score, index=index),
    }
    for index_name in _NAMES:
        data[f"score_{index_name}"] = pd.Series(out[f"score_{index_name}"], index=index)
        data[f"hold_rate_{index_name}"] = pd.Series(out[f"hold_rate_{index_name}"], index=index)
        data[f"price_{index_name}"] = pd.Series(out[f"price_{index_name}"], index=index)
        data[f"state_{index_name}"] = pd.Series(out[f"state_{index_name}"], index=index, dtype=object)
        data[f"is_next_{index_name}"] = flags(pd.Series(out[f"is_next_{index_name}"], index=index))
    return pd.DataFrame(data, index=index)
