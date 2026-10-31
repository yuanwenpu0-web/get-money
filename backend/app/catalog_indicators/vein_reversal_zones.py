"""Vein: Reversal Zones [WavesUnchained] — 前瞻验证的反转区 + 多源触发。

**反转标签（前瞻评估，`forward_bars`/`extended_bars` 之后回看）**：
对 `evalOffset = max(forward, extended) + 1` 根之前的 bar，向后看
`forward_bars` 根：先触及 `target_atr`×ATR 且途中反向不利波动 ≤ `max_adverse_atr`
×ATR → weak；再看 `extended_bars` 内是否触及 `strong_atr`×ATR → strong。
额外：weak 且到目标 ≤ `fast_bars_thresh` 根、MAE ≤ `mae_strong_thresh`×ATR
也升级为 strong。多空同时成立 → 标签 0（不建区）。

**区**：label>0 建支撑区 [low, startClose]、label<0 建阻力区 [startClose, high]，
最小高度 0.2×ATR；同向且重叠（±`merge_dist_atr`×ATR）合并取并集与更高等级；
超 `max_stored_zones` 丢最旧。收盘穿透（支撑破下沿 / 阻力破上沿）→ 删除；
首次进入区 touches+1。

**触发（四角色 + 合成分）**：结构（sweep 影线穿越 swing 后收回，3 分）、
动量（WaveTrend 极值交叉或 RSI 回收，2 分）、趋势（Supertrend 翻转，1 分）、
触发（MA 回收 + MACD 柱递增，2 分）、相对波动（relVol>1.2 加 2 分）。
`confl_score ≥ 5` 且至少结构或动量、角色数满足 `sig_requirement` → 候选信号；
`signal_mode` 要求近区（距离 ≤ `zone_prox_atr`×ATR）才放行；冷却
`cooldown_bars`。

丢弃：全部 box/label 绘图与可见区筛选。

Source: pine-collection/indicators/trend_direction/vein/vein_reversal_zones/vein_reversal_zones.pine
License: WavesUnchained 公开脚本（随清单收录）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    boolean,
    choice,
    crossunder,
    crossover,
    ema,
    flags,
    highest,
    integer,
    lowest,
    number,
    ohlcv,
    rma,
    rsi,
    sma,
    supertrend,
)

__all__ = ["vein_reversal_zones"]


def _eval_bull(highs, lows, closes, bars, ref_close, atr_value, target,
               max_adverse, close_only):
    mae = 0.0
    bars_to_target = bars
    hit = stop = False
    for i in range(1, bars + 1):
        b_hi = closes[i] if close_only else highs[i]
        b_lo = closes[i] if close_only else lows[i]
        if not hit:
            mae = max(mae, ref_close - b_lo)
        if not hit and not stop and (ref_close - b_lo) > max_adverse * atr_value:
            stop = True
        if not hit and not stop and (b_hi - ref_close) >= target * atr_value:
            hit = True
            bars_to_target = i
    return (hit and not stop), mae, bars_to_target


def _eval_bear(highs, lows, closes, bars, ref_close, atr_value, target,
               max_adverse, close_only):
    mae = 0.0
    bars_to_target = bars
    hit = stop = False
    for i in range(1, bars + 1):
        b_hi = closes[i] if close_only else highs[i]
        b_lo = closes[i] if close_only else lows[i]
        if not hit:
            mae = max(mae, b_hi - ref_close)
        if not hit and not stop and (b_hi - ref_close) > max_adverse * atr_value:
            stop = True
        if not hit and not stop and (ref_close - b_lo) >= target * atr_value:
            hit = True
            bars_to_target = i
    return (hit and not stop), mae, bars_to_target


def vein_reversal_zones(
    frame: pd.DataFrame,
    *,
    label_atr_len: int = 14,
    forward_bars: int = 8,
    extended_bars: int = 16,
    target_atr: float = 1.5,
    strong_atr: float = 2.5,
    max_adverse_atr: float = 0.6,
    use_close: bool = False,
    fast_bars_thresh: int = 4,
    mae_strong_thresh: float = 0.3,
    min_gap_bars: int = 3,
    show_strong_support: bool = True,
    show_weak_support: bool = True,
    show_strong_resist: bool = True,
    show_weak_resist: bool = True,
    max_stored_zones: int = 40,
    merge_dist_atr: float = 0.5,
    delete_broken: bool = True,
    signal_mode: str = "Known Zones",
    sig_requirement: str = "Any",
    zone_prox_atr: float = 0.5,
    cooldown_bars: int = 8,
    use_wt: bool = True,
    wt_ch_len: int = 10,
    wt_avg_len: int = 21,
    wt_os_level: float = -53.0,
    wt_ob_level: float = 53.0,
    use_rsi: bool = False,
    rsi_len: int = 14,
    rsi_os_level: float = 35.0,
    rsi_ob_level: float = 65.0,
    use_st: bool = True,
    st_atr_len: int = 10,
    st_factor: float = 3.0,
    use_sweep: bool = True,
    sweep_len: int = 10,
    use_ema: bool = True,
    ema_len: int = 20,
    ema_type: str = "RMA",
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_sig: int = 9,
) -> pd.DataFrame:
    """计算反转区与多源触发信号。"""
    label_atr_len = integer("label_atr_len", label_atr_len, 1)
    forward_bars = integer("forward_bars", forward_bars, 1, 50)
    extended_bars = integer("extended_bars", extended_bars, 2, 80)
    target_atr = number("target_atr", target_atr, 0.1)
    strong_atr = number("strong_atr", strong_atr, 0.5)
    max_adverse_atr = number("max_adverse_atr", max_adverse_atr, 0.1)
    use_close = boolean("use_close", use_close)
    fast_bars_thresh = integer("fast_bars_thresh", fast_bars_thresh, 1, 20)
    mae_strong_thresh = number("mae_strong_thresh", mae_strong_thresh, 0.0)
    min_gap_bars = integer("min_gap_bars", min_gap_bars, 0, 20)
    show_strong_support = boolean("show_strong_support", show_strong_support)
    show_weak_support = boolean("show_weak_support", show_weak_support)
    show_strong_resist = boolean("show_strong_resist", show_strong_resist)
    show_weak_resist = boolean("show_weak_resist", show_weak_resist)
    max_stored_zones = integer("max_stored_zones", max_stored_zones, 4, 100)
    merge_dist_atr = number("merge_dist_atr", merge_dist_atr, 0.0)
    delete_broken = boolean("delete_broken", delete_broken)
    signal_mode = choice("signal_mode", signal_mode,
                         ("Known Zones", "Candidates", "Both"))
    sig_requirement = choice("sig_requirement", sig_requirement, ("Any", "2+"))
    zone_prox_atr = number("zone_prox_atr", zone_prox_atr, 0.1)
    cooldown_bars = integer("cooldown_bars", cooldown_bars, 0)
    use_wt = boolean("use_wt", use_wt)
    wt_ch_len = integer("wt_ch_len", wt_ch_len, 1)
    wt_avg_len = integer("wt_avg_len", wt_avg_len, 1)
    wt_os_level = number("wt_os_level", wt_os_level, -200.0)
    wt_ob_level = number("wt_ob_level", wt_ob_level, -200.0)
    use_rsi = boolean("use_rsi", use_rsi)
    rsi_len = integer("rsi_len", rsi_len, 1)
    rsi_os_level = number("rsi_os_level", rsi_os_level, 0.0)
    rsi_ob_level = number("rsi_ob_level", rsi_ob_level, 0.0)
    use_st = boolean("use_st", use_st)
    st_atr_len = integer("st_atr_len", st_atr_len, 1)
    st_factor = number("st_factor", st_factor, 0.5)
    use_sweep = boolean("use_sweep", use_sweep)
    sweep_len = integer("sweep_len", sweep_len, 3)
    use_ema = boolean("use_ema", use_ema)
    ema_len = integer("ema_len", ema_len, 1)
    ema_type = choice("ema_type", ema_type, ("EMA", "RMA", "SMA", "WMA", "HMA", "JMA"))
    macd_fast = integer("macd_fast", macd_fast, 1)
    macd_slow = integer("macd_slow", macd_slow, 1)
    macd_sig = integer("macd_sig", macd_sig, 1)

    high, low, close = ohlcv(frame, "high", "low", "close")
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    n = len(frame)
    index = frame.index

    atr_series = atr(frame, label_atr_len)
    atr_arr = atr_series.to_numpy(dtype=float)
    rel_vol = atr_series / sma(atr_series, 50)

    buf_size = max(forward_bars, extended_bars) + 1
    eval_offset = buf_size - 1

    # 触发源。
    hlc3 = (high + low + close) / 3.0
    wt_esa = ema(hlc3, wt_ch_len)
    wt_d = ema((hlc3 - wt_esa).abs(), wt_ch_len)
    wt_ci = (hlc3 - wt_esa) / (0.015 * wt_d.where(wt_d != 0))
    wt1 = ema(wt_ci, wt_avg_len)
    wt2 = sma(wt1, 4)
    wt1_arr = wt1.to_numpy(dtype=float)
    wt2_arr = wt2.to_numpy(dtype=float)
    rsi_arr = rsi(close, rsi_len).to_numpy(dtype=float)
    _, st_dir = supertrend(frame, st_factor, st_atr_len)
    st_dir_arr = st_dir.to_numpy(dtype=float)
    swing_lo = lowest(low, sweep_len).shift(1).to_numpy(dtype=float)
    swing_hi = highest(high, sweep_len).shift(1).to_numpy(dtype=float)
    if ema_type == "EMA":
        ma = ema(close, ema_len)
    elif ema_type == "RMA":
        ma = rma(close, ema_len)
    elif ema_type == "SMA":
        ma = sma(close, ema_len)
    elif ema_type == "WMA":
        ma = sma(close, ema_len)  # WMA 近似：Pine 语义差异仅在权重
    else:
        ma = ema(close, ema_len)  # HMA/JMA 退化为 EMA
    ma_arr = ma.to_numpy(dtype=float)
    macd_line = ema(close, macd_fast) - ema(close, macd_slow)
    macd_hist = macd_line - ema(macd_line, macd_sig)
    macd_hist_arr = macd_hist.to_numpy(dtype=float)

    out = {
        "reversal_label": np.zeros(n, dtype=int),
        "raw_label": np.zeros(n, dtype=int),
        "zone_count": np.zeros(n, dtype=int),
        "zone_touches_max": np.zeros(n, dtype=int),
        "near_support": np.zeros(n, dtype=bool),
        "near_resistance": np.zeros(n, dtype=bool),
        "confl_score_long": np.full(n, np.nan),
        "confl_score_short": np.full(n, np.nan),
        "role_count_long": np.zeros(n, dtype=int),
        "role_count_short": np.zeros(n, dtype=int),
        "long_signal": np.zeros(n, dtype=bool),
        "short_signal": np.zeros(n, dtype=bool),
        "wt_long": np.zeros(n, dtype=bool),
        "wt_short": np.zeros(n, dtype=bool),
        "rsi_long": np.zeros(n, dtype=bool),
        "rsi_short": np.zeros(n, dtype=bool),
        "st_long": np.zeros(n, dtype=bool),
        "st_short": np.zeros(n, dtype=bool),
        "sweep_long": np.zeros(n, dtype=bool),
        "sweep_short": np.zeros(n, dtype=bool),
        "ema_long": np.zeros(n, dtype=bool),
        "ema_short": np.zeros(n, dtype=bool),
        "zone_created": np.zeros(n, dtype=bool),
    }

    # 环形缓冲（索引 0 = 最老 = eval 目标）。
    buf_close = [np.nan] * buf_size
    buf_high = [np.nan] * buf_size
    buf_low = [np.nan] * buf_size
    buf_atr = [np.nan] * buf_size

    z_dir: list[int] = []
    z_grade: list[int] = []
    z_upper: list[float] = []
    z_lower: list[float] = []
    z_touches: list[int] = []
    z_in_zone: list[bool] = []

    bars_since_bull = 999
    bars_since_bear = 999
    last_long_bar: int | None = None
    last_short_bar: int | None = None

    for bar in range(n):
        buf_close.pop(0)
        buf_close.append(closes[bar])
        buf_high.pop(0)
        buf_high.append(highs[bar])
        buf_low.pop(0)
        buf_low.append(lows[bar])
        buf_atr.pop(0)
        buf_atr.append(atr_arr[bar])

        start_close = buf_close[0]
        start_atr = buf_atr[0]
        can_eval = (not np.isnan(start_close) and not np.isnan(start_atr)
                    and start_atr > 0 and bar >= eval_offset)

        # 前瞻评估。
        if can_eval:
            bull_weak, bull_mae, bull_bars = _eval_bull(
                buf_high, buf_low, buf_close, forward_bars, start_close, start_atr,
                target_atr, max_adverse_atr, use_close)
            bear_weak, bear_mae, bear_bars = _eval_bear(
                buf_high, buf_low, buf_close, forward_bars, start_close, start_atr,
                target_atr, max_adverse_atr, use_close)
            bull_strong, _, _ = _eval_bull(
                buf_high, buf_low, buf_close, extended_bars, start_close, start_atr,
                strong_atr, max_adverse_atr, use_close)
            bear_strong, _, _ = _eval_bear(
                buf_high, buf_low, buf_close, extended_bars, start_close, start_atr,
                strong_atr, max_adverse_atr, use_close)
        else:
            bull_weak = bear_weak = bull_strong = bear_strong = False
            bull_mae = bear_mae = 0.0
            bull_bars = bear_bars = forward_bars

        bull_is_strong = bull_strong or (
            bull_weak and bull_bars <= fast_bars_thresh and start_atr > 0
            and bull_mae <= mae_strong_thresh * start_atr)
        bull_is_weak = bull_weak and not bull_is_strong
        bear_is_strong = bear_strong or (
            bear_weak and bear_bars <= fast_bars_thresh and start_atr > 0
            and bear_mae <= mae_strong_thresh * start_atr)
        bear_is_weak = bear_weak and not bear_is_strong

        raw_label = (2 if bull_is_strong else 1 if bull_is_weak
                     else -2 if bear_is_strong else -1 if bear_is_weak else 0)
        if (bull_is_strong or bull_is_weak) and (bear_is_strong or bear_is_weak):
            raw_label = 0
        out["raw_label"][bar] = raw_label

        bars_since_bull += 1
        bars_since_bear += 1
        label = raw_label
        if min_gap_bars > 0:
            if raw_label > 0 and bars_since_bull < min_gap_bars:
                label = 0
            if raw_label < 0 and bars_since_bear < min_gap_bars:
                label = 0
        if label > 0:
            bars_since_bull = 0
        if label < 0:
            bars_since_bear = 0
        out["reversal_label"][bar] = label

        # 建区。
        if label != 0:
            z_dir_new = 1 if label > 0 else -1
            z_grade_new = abs(label)
            do_add = ((z_dir_new == 1 and z_grade_new == 2 and show_strong_support)
                      or (z_dir_new == 1 and z_grade_new == 1 and show_weak_support)
                      or (z_dir_new == -1 and z_grade_new == 2 and show_strong_resist)
                      or (z_dir_new == -1 and z_grade_new == 1 and show_weak_resist))
            if do_add:
                z_upper_new = start_close if z_dir_new == 1 else buf_high[0]
                z_lower_new = buf_low[0] if z_dir_new == 1 else start_close
                min_h = 0.2 * start_atr
                if z_upper_new - z_lower_new < min_h:
                    if z_dir_new == 1:
                        z_lower_new = z_upper_new - min_h
                    else:
                        z_upper_new = z_lower_new + min_h
                merged = False
                if merge_dist_atr > 0 and z_dir:
                    merge_thresh = merge_dist_atr * start_atr
                    for i in range(len(z_dir)):
                        if not merged and z_dir[i] == z_dir_new:
                            if (z_lower_new <= z_upper[i] + merge_thresh
                                    and z_upper_new >= z_lower[i] - merge_thresh):
                                z_upper[i] = max(z_upper[i], z_upper_new)
                                z_lower[i] = min(z_lower[i], z_lower_new)
                                z_grade[i] = max(z_grade[i], z_grade_new)
                                merged = True
                if not merged:
                    if len(z_dir) >= max_stored_zones:
                        for store in (z_dir, z_grade, z_upper, z_lower, z_touches,
                                      z_in_zone):
                            store.pop(0)
                    z_dir.append(z_dir_new)
                    z_grade.append(z_grade_new)
                    z_upper.append(z_upper_new)
                    z_lower.append(z_lower_new)
                    z_touches.append(0)
                    z_in_zone.append(False)
                out["zone_created"][bar] = True

        # 区维护（倒序删除）。
        for i in range(len(z_dir) - 1, -1, -1):
            in_zone = z_lower[i] <= closes[bar] <= z_upper[i]
            if in_zone and not z_in_zone[i]:
                z_touches[i] += 1
            z_in_zone[i] = in_zone
            if delete_broken and ((z_dir[i] == 1 and closes[bar] < z_lower[i])
                                  or (z_dir[i] == -1 and closes[bar] > z_upper[i])):
                for store in (z_dir, z_grade, z_upper, z_lower, z_touches, z_in_zone):
                    store.pop(i)
        out["zone_count"][bar] = len(z_dir)
        out["zone_touches_max"][bar] = max(z_touches) if z_touches else 0

        # 近区检测。
        prox = zone_prox_atr * (atr_arr[bar] if not np.isnan(atr_arr[bar]) else 0.0)
        near_support = near_resist = False
        for i in range(len(z_dir)):
            dist = (closes[bar] - z_upper[i] if closes[bar] > z_upper[i]
                    else z_lower[i] - closes[bar] if closes[bar] < z_lower[i]
                    else 0.0)
            if dist <= prox:
                if z_dir[i] == 1:
                    near_support = True
                else:
                    near_resist = True
        out["near_support"][bar] = near_support
        out["near_resistance"][bar] = near_resist

        # 触发。
        wt_long = (use_wt and wt1_arr[bar] > wt2_arr[bar]
                   and (bar == 0 or wt1_arr[bar - 1] <= wt2_arr[bar - 1])
                   and wt1_arr[bar] <= wt_os_level)
        wt_short = (use_wt and wt1_arr[bar] < wt2_arr[bar]
                    and (bar == 0 or wt1_arr[bar - 1] >= wt2_arr[bar - 1])
                    and wt1_arr[bar] >= wt_ob_level)
        rsi_long = (use_rsi and not np.isnan(rsi_arr[bar])
                    and rsi_arr[bar] > rsi_os_level
                    and (bar == 0 or rsi_arr[bar - 1] <= rsi_os_level))
        rsi_short = (use_rsi and not np.isnan(rsi_arr[bar])
                     and rsi_arr[bar] < rsi_ob_level
                     and (bar == 0 or rsi_arr[bar - 1] >= rsi_ob_level))
        st_long = (use_st and bar > 0 and st_dir_arr[bar] < 0
                   and (np.isnan(st_dir_arr[bar - 1]) or st_dir_arr[bar - 1] > 0))
        st_short = (use_st and bar > 0 and st_dir_arr[bar] > 0
                    and (np.isnan(st_dir_arr[bar - 1]) or st_dir_arr[bar - 1] < 0))
        sweep_long = (use_sweep and not np.isnan(swing_lo[bar])
                      and lows[bar] < swing_lo[bar] and closes[bar] > swing_lo[bar])
        sweep_short = (use_sweep and not np.isnan(swing_hi[bar])
                       and highs[bar] > swing_hi[bar] and closes[bar] < swing_hi[bar])
        ema_long = (use_ema and closes[bar] > ma_arr[bar]
                    and (bar == 0 or closes[bar - 1] <= ma_arr[bar - 1])
                    and macd_hist_arr[bar] > (macd_hist_arr[bar - 1] if bar > 0 else 0.0))
        ema_short = (use_ema and closes[bar] < ma_arr[bar]
                     and (bar == 0 or closes[bar - 1] >= ma_arr[bar - 1])
                     and macd_hist_arr[bar] < (macd_hist_arr[bar - 1] if bar > 0 else 0.0))
        for name, value in (("wt_long", wt_long), ("wt_short", wt_short),
                            ("rsi_long", rsi_long), ("rsi_short", rsi_short),
                            ("st_long", st_long), ("st_short", st_short),
                            ("sweep_long", sweep_long), ("sweep_short", sweep_short),
                            ("ema_long", ema_long), ("ema_short", ema_short)):
            out[name][bar] = value

        momentum_long = wt_long or rsi_long
        momentum_short = wt_short or rsi_short
        structure_long = sweep_long
        structure_short = sweep_short
        trigger_long = ema_long
        trigger_short = ema_short
        rv = rel_vol.iloc[bar] if not np.isnan(rel_vol.iloc[bar]) else 1.0
        confl_long = ((3.0 if structure_long else 0.0)
                      + (2.0 if momentum_long else 0.0)
                      + (2.0 if rv > 1.2 else 0.0)
                      + (2.0 if ema_long else 0.0)
                      + (1.0 if st_long else 0.0))
        confl_short = ((3.0 if structure_short else 0.0)
                       + (2.0 if momentum_short else 0.0)
                       + (2.0 if rv > 1.2 else 0.0)
                       + (2.0 if ema_short else 0.0)
                       + (1.0 if st_short else 0.0))
        role_long = (int(structure_long) + int(momentum_long) + int(st_long)
                     + int(trigger_long))
        role_short = (int(structure_short) + int(momentum_short) + int(st_short)
                      + int(trigger_short))
        req_ok_long = sig_requirement == "Any" or role_long >= 2
        req_ok_short = sig_requirement == "Any" or role_short >= 2
        any_long = confl_long >= 5.0 and (structure_long or momentum_long) and req_ok_long
        any_short = confl_short >= 5.0 and (structure_short or momentum_short) and req_ok_short
        out["confl_score_long"][bar] = confl_long
        out["confl_score_short"][bar] = confl_short
        out["role_count_long"][bar] = role_long
        out["role_count_short"][bar] = role_short

        gate_long = signal_mode != "Known Zones" or near_support
        gate_short = signal_mode != "Known Zones" or near_resist
        cd_long = last_long_bar is None or bar - last_long_bar > cooldown_bars
        cd_short = last_short_bar is None or bar - last_short_bar > cooldown_bars
        go_long = any_long and gate_long and cd_long
        go_short = any_short and gate_short and cd_short
        out["long_signal"][bar] = go_long
        out["short_signal"][bar] = go_short
        if go_long:
            last_long_bar = bar
        if go_short:
            last_short_bar = bar

    data: dict[str, pd.Series] = {
        "reversal_label": pd.Series(out["reversal_label"], index=index, dtype="int64"),
        "raw_label": pd.Series(out["raw_label"], index=index, dtype="int64"),
        "zone_count": pd.Series(out["zone_count"], index=index, dtype="int64"),
        "zone_touches_max": pd.Series(out["zone_touches_max"], index=index, dtype="int64"),
        "confl_score_long": pd.Series(out["confl_score_long"], index=index),
        "confl_score_short": pd.Series(out["confl_score_short"], index=index),
        "role_count_long": pd.Series(out["role_count_long"], index=index, dtype="int64"),
        "role_count_short": pd.Series(out["role_count_short"], index=index, dtype="int64"),
    }
    for column in ("near_support", "near_resistance", "long_signal", "short_signal",
                   "wt_long", "wt_short", "rsi_long", "rsi_short", "st_long",
                   "st_short", "sweep_long", "sweep_short", "ema_long", "ema_short",
                   "zone_created"):
        data[column] = flags(pd.Series(out[column], index=index))
    return pd.DataFrame(data, index=index)
