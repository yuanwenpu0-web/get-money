"""Relative Leg Efficiency [WavesUnchained] — 段落（leg）质量评分。

以 pivot(L,R) 检测段落端点：同向新 pivot 更极值则替换端点；反向 pivot 且
间距 ≥ `min_leg_bars`、位移 ≥ `min_leg_move_atr`×段首 ATR → 段落完成。

每段三个 0..1 传感器：

- **PE（路径效率）**：|端点位移| / Σ|hlc3 逐 bar 变化|，直线段 = 1。
- **TF（时间效率）**：Speed Norm（速度/近 `speed_lookback` 段均速，钳到
  `speed_clamp_max`）/ Time Penalty（1/(1+bars/中位时长)）/ Combined。
- **CP（反向压力）**：反向 K 的 body（或计数）占比 → 1−占比。

RLE = 100×(wPath·PE + wTime·TF + wCounter·CP)/Σw。可选 Spike Guard 把
段内最大实体 > mult×ATR 的段落钳到 `spike_cap`。活跃段实时评分（端点取
high/low），完成段落更新历史。

丢弃：leg 线/标签/直方图与全部绘图。

Source: pine-collection/indicators/relative_strength/relative_leg_efficiency/relative_leg_efficiency.pine
License: WavesUnchained 公开脚本（随清单收录）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    boolean,
    choice,
    flags,
    integer,
    number,
    ohlcv,
    pivothigh,
    pivotlow,
)

__all__ = ["relative_leg_efficiency"]


def relative_leg_efficiency(
    frame: pd.DataFrame,
    *,
    pivot_left: int = 3,
    pivot_right: int = 3,
    min_leg_bars: int = 6,
    atr_length: int = 14,
    min_leg_move_atr: float = 1.0,
    time_mode: str = "Speed Norm",
    speed_lookback: int = 20,
    speed_clamp_max: float = 2.0,
    time_ref_lookback: int = 20,
    counter_mode: str = "Body Weighted",
    use_meaningful_counter: bool = False,
    counter_body_atr: float = 0.2,
    weight_path: float = 0.5,
    weight_time: float = 0.2,
    weight_counter: float = 0.3,
    high_threshold: int = 75,
    mid_threshold: int = 55,
    enable_spike_guard: bool = False,
    spike_atr_mult: float = 2.5,
    spike_cap: float = 85.0,
) -> pd.DataFrame:
    """计算段落检测与 RLE 评分。"""
    pivot_left = integer("pivot_left", pivot_left, 1, 20)
    pivot_right = integer("pivot_right", pivot_right, 1, 20)
    min_leg_bars = integer("min_leg_bars", min_leg_bars, 1, 100)
    atr_length = integer("atr_length", atr_length, 5, 100)
    min_leg_move_atr = number("min_leg_move_atr", min_leg_move_atr, 0.1, 10.0)
    time_mode = choice("time_mode", time_mode,
                       ("Speed Norm", "Time Penalty", "Combined"))
    speed_lookback = integer("speed_lookback", speed_lookback, 5, 100)
    speed_clamp_max = number("speed_clamp_max", speed_clamp_max, 1.0, 5.0)
    time_ref_lookback = integer("time_ref_lookback", time_ref_lookback, 5, 100)
    counter_mode = choice("counter_mode", counter_mode,
                          ("Count", "Body Weighted", "Both"))
    use_meaningful_counter = boolean("use_meaningful_counter",
                                     use_meaningful_counter)
    counter_body_atr = number("counter_body_atr", counter_body_atr, 0.0, 2.0)
    weight_path = number("weight_path", weight_path, 0.0, 1.0)
    weight_time = number("weight_time", weight_time, 0.0, 1.0)
    weight_counter = number("weight_counter", weight_counter, 0.0, 1.0)
    integer("high_threshold", high_threshold, 1, 100)
    integer("mid_threshold", mid_threshold, 1, 100)
    enable_spike_guard = boolean("enable_spike_guard", enable_spike_guard)
    spike_atr_mult = number("spike_atr_mult", spike_atr_mult, 0.5, 10.0)
    number("spike_cap", spike_cap, 1.0, 100.0)

    high, low, open_, close = ohlcv(frame, "high", "low", "open", "close")
    src = (high + low + close) / 3.0
    closes = close.to_numpy(dtype=float)
    opens = open_.to_numpy(dtype=float)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    src_arr = src.to_numpy(dtype=float)
    n = len(frame)

    atr_arr = atr(frame, atr_length).to_numpy(dtype=float)
    ph_arr = pivothigh(high, pivot_left, pivot_right).to_numpy(dtype=float)
    pl_arr = pivotlow(low, pivot_left, pivot_right).to_numpy(dtype=float)

    body_arr = np.abs(closes - opens)
    bull_mask = closes > opens
    bear_mask = closes < opens

    wsum = weight_path + weight_time + weight_counter

    def time_eff_speed(speed: float, speed_avg: float | None) -> float:
        if speed_avg is None or speed_avg <= 0.0:
            return 0.5
        clamped = min(max(speed / speed_avg, 0.0), speed_clamp_max)
        return clamped / speed_clamp_max

    def time_eff_penalty(bars: int, time_ref: float | None) -> float:
        if time_ref is None or time_ref <= 0.0:
            return 0.5
        return 1.0 / (1.0 + bars / time_ref)

    def counter_ratio(bars: int, counter_count: float, counter_body: float,
                      total_body: float) -> float:
        ratio_count = counter_count / bars if bars > 0 else 0.0
        ratio_body = counter_body / total_body if total_body > 0 else 0.0
        if counter_mode == "Count":
            ratio = ratio_count
        elif counter_mode == "Body Weighted":
            ratio = ratio_body
        else:
            ratio = 0.3 * ratio_count + 0.7 * ratio_body
        return min(max(ratio, 0.0), 1.0)

    def rle_score(pe: float, tf: float, cp: float) -> float:
        if wsum <= 0.0:
            return np.nan
        return 100.0 * (weight_path * pe + weight_time * tf
                        + weight_counter * cp) / wsum

    out = {
        "leg_completed": np.zeros(n, dtype=bool),
        "leg_direction": np.zeros(n, dtype=int),
        "leg_bars": np.full(n, np.nan),
        "leg_pe": np.full(n, np.nan),
        "leg_tf": np.full(n, np.nan),
        "leg_cp": np.full(n, np.nan),
        "rle_completed": np.full(n, np.nan),
        "rle_live": np.full(n, np.nan),
        "pe_live": np.full(n, np.nan),
        "tf_live": np.full(n, np.nan),
        "cp_live": np.full(n, np.nan),
        "active_leg_direction": np.zeros(n, dtype=int),
        "active_leg_bars": np.zeros(n, dtype=int),
        "rle_band": np.full(n, "-", dtype=object),
    }

    last_pivot_index = -1
    last_pivot_price = np.nan
    last_pivot_type = 0
    last_pivot_atr = np.nan
    path_cum = 0.0
    body_cum = 0.0
    bull_body_cum = bear_body_cum = 0.0
    bull_count_cum = bear_count_cum = 0.0
    cum_path: list[float] = []      # 每根结束后的累计值，支持前缀差分
    cum_body: list[float] = []
    cum_bull_body: list[float] = []
    cum_bear_body: list[float] = []
    cum_bull_count: list[float] = []
    cum_bear_count: list[float] = []
    speed_history: list[float] = []
    length_history: list[int] = []
    last_completed = (np.nan, np.nan, np.nan, np.nan, np.nan)  # rle,pe,tf,cp,bars

    for bar in range(n):
        prev_src = src_arr[bar - 1] if bar > 0 else src_arr[bar]
        path_cum += abs(src_arr[bar] - prev_src)
        body = body_arr[bar]
        is_meaningful = body >= atr_arr[bar] * counter_body_atr
        body_add = body if (is_meaningful or not use_meaningful_counter) else 0.0
        if bull_mask[bar]:
            bull_body_cum += body_add
            bull_count_cum += 1.0 if (not use_meaningful_counter or is_meaningful) else 0.0
        elif bear_mask[bar]:
            bear_body_cum += body_add
            bear_count_cum += 1.0 if (not use_meaningful_counter or is_meaningful) else 0.0
        cum_path.append(path_cum)
        cum_body.append((cum_body[-1] + body_add) if cum_body else body_add)
        cum_bull_body.append(bull_body_cum)
        cum_bear_body.append(bear_body_cum)
        cum_bull_count.append(bull_count_cum)
        cum_bear_count.append(bear_count_cum)

        def prefix(series: list[float], offset_from: int) -> float:
            base = series[bar - offset_from - 1] if bar - offset_from - 1 >= 0 else 0.0
            return series[bar] - base

        # pivot 更新与 leg 完成。
        has_ph = not np.isnan(ph_arr[bar])
        has_pl = not np.isnan(pl_arr[bar])
        new_type = 0
        new_index = -1
        new_price = np.nan
        if has_ph and has_pl:
            use_high = (np.isnan(last_pivot_price)
                        or abs(ph_arr[bar] - last_pivot_price)
                        >= abs(pl_arr[bar] - last_pivot_price))
            new_type = 1 if use_high else -1
            new_index = bar - pivot_right
            new_price = ph_arr[bar] if use_high else pl_arr[bar]
        elif has_ph:
            new_type, new_index, new_price = 1, bar - pivot_right, ph_arr[bar]
        elif has_pl:
            new_type, new_index, new_price = -1, bar - pivot_right, pl_arr[bar]

        leg_completed = False
        leg_start = leg_end = -1
        leg_start_price = leg_end_price = np.nan
        leg_direction = 0
        if new_type != 0:
            if last_pivot_type == 0:
                last_pivot_index, last_pivot_price = new_index, new_price
                last_pivot_type = new_type
                last_pivot_atr = atr_arr[bar - pivot_right] if bar >= pivot_right \
                    else atr_arr[bar]
            elif new_type == last_pivot_type:
                replace = ((new_type == 1 and new_price >= last_pivot_price)
                           or (new_type == -1 and new_price <= last_pivot_price))
                if replace:
                    last_pivot_index, last_pivot_price = new_index, new_price
                    last_pivot_atr = atr_arr[bar - pivot_right] if bar >= pivot_right \
                        else atr_arr[bar]
            else:
                bars_between = new_index - last_pivot_index
                min_move = min_leg_move_atr * (last_pivot_atr if not np.isnan(last_pivot_atr)
                                               else atr_arr[bar])
                move = abs(new_price - last_pivot_price)
                if bars_between >= min_leg_bars and move >= min_move:
                    leg_completed = True
                    leg_start = last_pivot_index
                    leg_start_price = last_pivot_price
                    leg_end = new_index
                    leg_end_price = new_price
                    leg_direction = 1 if last_pivot_type == -1 else -1
                    last_pivot_index, last_pivot_price = new_index, new_price
                    last_pivot_type = new_type
                    last_pivot_atr = atr_arr[bar - pivot_right] if bar >= pivot_right \
                        else atr_arr[bar]
        out["leg_completed"][bar] = leg_completed

        speed_avg = (sum(speed_history) / len(speed_history)
                     if speed_history else None)
        time_ref = (float(np.median(length_history))
                    if length_history else None)

        if leg_completed:
            bars_in_leg = leg_end - leg_start
            leg_air = abs(leg_end_price - leg_start_price)
            start_off = bar - leg_start  # leg_start 距当根
            end_off = bar - leg_end
            leg_path = (cum_path[bar - end_off]
                        - (cum_path[bar - start_off - 1]
                           if bar - start_off - 1 >= 0 else 0.0))
            leg_path_air = abs(src_arr[bar - end_off] - src_arr[bar - start_off])
            leg_total_body = (cum_body[bar - end_off]
                              - (cum_body[bar - start_off - 1]
                                 if bar - start_off - 1 >= 0 else 0.0))
            if leg_direction == 1:
                leg_counter_count = (cum_bear_count[bar - end_off]
                                     - (cum_bear_count[bar - start_off - 1]
                                        if bar - start_off - 1 >= 0 else 0.0))
                leg_counter_body = (cum_bear_body[bar - end_off]
                                    - (cum_bear_body[bar - start_off - 1]
                                       if bar - start_off - 1 >= 0 else 0.0))
            else:
                leg_counter_count = (cum_bull_count[bar - end_off]
                                     - (cum_bull_count[bar - start_off - 1]
                                        if bar - start_off - 1 >= 0 else 0.0))
                leg_counter_body = (cum_bull_body[bar - end_off]
                                    - (cum_bull_body[bar - start_off - 1]
                                       if bar - start_off - 1 >= 0 else 0.0))
            pe = min(max(leg_path_air / leg_path, 0.0), 1.0) if leg_path > 0 else 0.0
            speed = leg_air / bars_in_leg if bars_in_leg > 0 else 0.0
            tf_speed = time_eff_speed(speed, speed_avg)
            tf_penalty = time_eff_penalty(bars_in_leg, time_ref)
            tf = (tf_speed if time_mode == "Speed Norm"
                  else tf_penalty if time_mode == "Time Penalty"
                  else (tf_speed + tf_penalty) / 2.0)
            ratio = counter_ratio(bars_in_leg, leg_counter_count,
                                  leg_counter_body, leg_total_body)
            cp = 1.0 - ratio
            rle = rle_score(pe, tf, cp)
            if enable_spike_guard and bars_in_leg > 0:
                leg_max_body = max(body_arr[bar - start_off + 1: bar - end_off + 1]
                                   ) if start_off > end_off else 0.0
                if leg_max_body > spike_atr_mult * atr_arr[bar] and not np.isnan(rle):
                    rle = min(rle, spike_cap)
            last_completed = (rle, pe, tf, cp, float(bars_in_leg))
            if bars_in_leg > 0:
                speed_history.append(leg_air / bars_in_leg)
                length_history.append(bars_in_leg)
                if len(speed_history) > speed_lookback:
                    speed_history.pop(0)
                if len(length_history) > time_ref_lookback:
                    length_history.pop(0)
            out["leg_direction"][bar] = leg_direction
            out["leg_bars"][bar] = bars_in_leg
            out["leg_pe"][bar] = pe
            out["leg_tf"][bar] = tf
            out["leg_cp"][bar] = cp
            out["rle_completed"][bar] = rle

        # 活跃段实时评分。
        has_active = last_pivot_type != 0
        if has_active:
            active_start = last_pivot_index
            active_dir = 1 if last_pivot_type == -1 else -1
            active_bars = bar - active_start
            out["active_leg_direction"][bar] = active_dir
            out["active_leg_bars"][bar] = active_bars
            active_end_price = (highs[bar] if active_dir == 1
                                else lows[bar] if active_dir == -1 else src_arr[bar])
            active_air = abs(active_end_price - last_pivot_price)
            start_off = bar - active_start
            eligible = active_bars >= min_leg_bars and active_air >= min_leg_move_atr * atr_arr[bar]
            if eligible and start_off - 1 < len(cum_path):
                base = start_off - 1
                base_path = cum_path[base] if base >= 0 else 0.0
                active_path = cum_path[bar] - base_path
                active_path_air = abs(src_arr[bar] - src_arr[bar - start_off])
                total_body = cum_body[bar] - (cum_body[base] if base >= 0 else 0.0)
                if active_dir == 1:
                    counter_count = cum_bear_count[bar] - (cum_bear_count[base] if base >= 0 else 0.0)
                    counter_body = cum_bear_body[bar] - (cum_bear_body[base] if base >= 0 else 0.0)
                else:
                    counter_count = cum_bull_count[bar] - (cum_bull_count[base] if base >= 0 else 0.0)
                    counter_body = cum_bull_body[bar] - (cum_bull_body[base] if base >= 0 else 0.0)
                pe = min(max(active_path_air / active_path, 0.0), 1.0) if active_path > 0 else np.nan
                speed = active_air / active_bars if active_bars > 0 else np.nan
                tf_speed = time_eff_speed(speed, speed_avg)
                tf_penalty = time_eff_penalty(active_bars, time_ref)
                tf = (tf_speed if time_mode == "Speed Norm"
                      else tf_penalty if time_mode == "Time Penalty"
                      else (tf_speed + tf_penalty) / 2.0)
                ratio = counter_ratio(active_bars, counter_count, counter_body, total_body)
                cp = 1.0 - ratio
                rle = rle_score(pe, tf, cp)
                if enable_spike_guard and active_bars > 0 and not np.isnan(rle):
                    max_body = max(body_arr[bar - active_bars + 1: bar + 1])
                    if max_body > spike_atr_mult * atr_arr[bar]:
                        rle = min(rle, spike_cap)
                out["rle_live"][bar] = rle
                out["pe_live"][bar] = pe
                out["tf_live"][bar] = tf
                out["cp_live"][bar] = cp

        if not np.isnan(last_completed[0]):
            out["rle_band"][bar] = (
                "High" if last_completed[0] >= high_threshold
                else "Mid" if last_completed[0] >= mid_threshold else "Low")

    index = frame.index
    data: dict[str, pd.Series] = {
        "leg_completed": flags(pd.Series(out["leg_completed"], index=index)),
        "leg_direction": pd.Series(out["leg_direction"], index=index, dtype="int64"),
        "leg_bars": pd.Series(out["leg_bars"], index=index, dtype="Int64"),
        "leg_pe": pd.Series(out["leg_pe"], index=index),
        "leg_tf": pd.Series(out["leg_tf"], index=index),
        "leg_cp": pd.Series(out["leg_cp"], index=index),
        "rle_completed": pd.Series(out["rle_completed"], index=index),
        "rle_live": pd.Series(out["rle_live"], index=index),
        "pe_live": pd.Series(out["pe_live"], index=index),
        "tf_live": pd.Series(out["tf_live"], index=index),
        "cp_live": pd.Series(out["cp_live"], index=index),
        "active_leg_direction": pd.Series(out["active_leg_direction"], index=index,
                                          dtype="int64"),
        "active_leg_bars": pd.Series(out["active_leg_bars"], index=index,
                                     dtype="int64"),
        "rle_band": pd.Series(out["rle_band"], index=index, dtype=object),
    }
    return pd.DataFrame(data, index=index)
