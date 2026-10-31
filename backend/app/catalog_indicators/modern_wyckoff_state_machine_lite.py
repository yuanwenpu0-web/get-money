"""Modern Wyckoff State Machine Lite [WavesUnchained] — 区间 context 下的 A-E 相位状态机。

移植要点（对照 modern_wyckoff_state_machine_visual.pine v1.4.3）：

- **核心测量**：ATR、ADX/±DI（`_pine.dmi`，Wilder 平滑与 Pine 一致）、成交量 Z/相对量、
  K 线 spread/实体/影线比率、区间高低（`highest/lowest(rangeLen)`）。
- **激活区间**：进入 Accumulation/Distribution/Markup/Markdown 时锁定区间高低（与价格结构同步，
  直到 Spring/UTAD 推进到 C 相再冻结）。`activeRange` = 锁定时取锁定值否则取实时值。
- **事件**：Spring / UTAD（含深度 ATR、影线比率、K线 ATR 的质量分级）、SOS / SOW、
  Spring/UTAD Test、LPS / LPSY（枢轴确认）、Selling/Buying Climax、上下吸收。
  事件有 cooldown 与 eventLookback 记忆窗。
- **打分**：accumulation/distribution/cause/rangeQuality 各分量（与源码逐项一致），
  confidence 取当前状态的偏向分。
- **状态机**：Neutral→A→B→C→D→E（E 翻 Markup/Markdown），含 hardBull/Bear 失效、
  Markup/Markdown 回归 Neutral、卡死重置、Phase A/B 区间刷新。逐 bar 用 `var` 状态复刻。
- **丢弃**：全部 box/label/table/fill/barcolor/alert（绘图与告警），仅输出数值与状态列。

实现说明：区间量价测量统一采用「每 bar 入口处锁定的 activeRange」（与事件/打分同源，
对应 Pine 行 176-183 的定义）；Pine 行 555-562 的二次重算只用于绘图，数据输出不采用，
二者差异仅在状态机当根改写锁定区间的那一根，属 1-bar 边角，不影响手算校验。

Source: pine-collection/indicators/market_structure/modern_wyckoff_state_machine_visual/modern_wyckoff_state_machine_visual.pine
License: WavesUnchained 公开脚本（随清单收录）。
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd

from ._pine import atr, boolean, dmi, ema, flags, highest, integer, lowest, number, ohlcv, pivothigh, pivotlow, sma, stdev

__all__ = ["modern_wyckoff_state_machine_lite"]

# WState codes
_NEUTRAL, _ACC, _DIST, _MARKUP, _MARKDOWN = 0, 1, 2, 3, 4
# WPhase codes
_PH_NONE, _PH_A, _PH_B, _PH_C, _PH_D, _PH_E = 0, 1, 2, 3, 4, 5

_STATE_NAME = {_NEUTRAL: "NEUTRAL", _ACC: "ACCUMULATION", _DIST: "DISTRIBUTION",
               _MARKUP: "MARKUP", _MARKDOWN: "MARKDOWN"}
_PHASE_NAME = {_PH_NONE: "None", _PH_A: "A", _PH_B: "B", _PH_C: "C", _PH_D: "D", _PH_E: "E"}


def _phase_text(state: int, phase: int) -> str:
    if phase == _PH_A:
        return "A: Stopping Action"
    if phase == _PH_B:
        return "B: Building Cause"
    if phase == _PH_C:
        return "C: UTAD / Test" if state == _DIST else "C: Spring / Test"
    if phase == _PH_D:
        return "D: Markdown Prep" if state == _DIST else "D: Markup Prep"
    if phase == _PH_E:
        return "E: Trend Release"
    return "None"


def _event_quality(triggered: bool, wick_ratio: float, candle_atr_ratio: float,
                   volume_ratio: float, sweep_atr: float,
                   spring_atr_min: float, strong_wick: float, strong_candle: float,
                   strong_vol: float) -> int:
    if not triggered:
        return 0
    factors = 0
    if wick_ratio >= strong_wick:
        factors += 1
    if candle_atr_ratio >= strong_candle:
        factors += 1
    if volume_ratio >= strong_vol:
        factors += 1
    if sweep_atr >= spring_atr_min * 2.0:
        factors += 1
    return 3 if factors >= 3 else 2 if factors >= 1 else 1


def modern_wyckoff_state_machine_lite(
    frame: pd.DataFrame,
    *,
    range_len: int = 80,
    pivot_len: int = 5,
    adx_len: int = 14,
    atr_len: int = 14,
    vol_len: int = 50,
    range_atr_max: float = 8.0,
    adx_max: float = 24.0,
    refresh_cause_range: bool = True,
    event_lookback: int = 30,
    min_phase_bars: int = 5,
    event_cooldown: int = 10,
    breakout_confirm_bars: int = 2,
    reset_stuck_range: bool = True,
    max_cause_bars: int = 120,
    stale_range_bars: int = 40,
    spring_atr_min: float = 0.25,
    spring_pct_min: float = 0.75,
    min_wick_ratio: float = 0.30,
    min_candle_atr: float = 0.50,
    strong_wick_ratio: float = 0.50,
    strong_candle_atr: float = 1.00,
    strong_vol_ratio: float = 1.40,
    test_atr_window: float = 1.00,
    test_vol_factor: float = 0.85,
    lps_vol_factor: float = 0.85,
) -> pd.DataFrame:
    """返回 Wyckoff 状态/相位/事件/打分等数值列。"""
    range_len = integer("range_len", range_len, 30, 500)
    pivot_len = integer("pivot_len", pivot_len, 2, 50)
    adx_len = integer("adx_len", adx_len, 2, 200)
    atr_len = integer("atr_len", atr_len, 1, 200)
    vol_len = integer("vol_len", vol_len, 5, 500)
    range_atr_max = number("range_atr_max", range_atr_max, 0.25, 50.0)
    adx_max = number("adx_max", adx_max, 1.0, 100.0)
    refresh_cause_range = boolean("refresh_cause_range", refresh_cause_range)
    event_lookback = integer("event_lookback", event_lookback, 5, 500)
    min_phase_bars = integer("min_phase_bars", min_phase_bars, 1, 200)
    event_cooldown = integer("event_cooldown", event_cooldown, 1, 200)
    breakout_confirm_bars = integer("breakout_confirm_bars", breakout_confirm_bars, 1, 5)
    reset_stuck_range = boolean("reset_stuck_range", reset_stuck_range)
    max_cause_bars = integer("max_cause_bars", max_cause_bars, 20, 1000)
    stale_range_bars = integer("stale_range_bars", stale_range_bars, 5, 1000)
    spring_atr_min = number("spring_atr_min", spring_atr_min, 0.05, 10.0)
    spring_pct_min = number("spring_pct_min", spring_pct_min, 0.1, 50.0)
    min_wick_ratio = number("min_wick_ratio", min_wick_ratio, 0.1, 0.8)
    min_candle_atr = number("min_candle_atr", min_candle_atr, 0.1, 20.0)
    strong_wick_ratio = number("strong_wick_ratio", strong_wick_ratio, 0.2, 0.9)
    strong_candle_atr = number("strong_candle_atr", strong_candle_atr, 0.3, 20.0)
    strong_vol_ratio = number("strong_vol_ratio", strong_vol_ratio, 0.5, 10.0)
    test_atr_window = number("test_atr_window", test_atr_window, 0.2, 10.0)
    test_vol_factor = number("test_vol_factor", test_vol_factor, 0.2, 1.2)
    lps_vol_factor = number("lps_vol_factor", lps_vol_factor, 0.2, 1.2)

    high, low, close = ohlcv(frame, "high", "low", "close")
    open_ = ohlcv(frame, "open")[0]
    hs = high.to_numpy(dtype=float)
    ls = low.to_numpy(dtype=float)
    cs = close.to_numpy(dtype=float)
    os = open_.to_numpy(dtype=float)
    n = len(frame)

    volume = frame.get("volume")
    if volume is None or volume.isna().all() or float(volume.fillna(0).sum()) == 0.0:
        volume_is_real = False
        vs = np.zeros(n)
    else:
        volume_is_real = True
        vs = volume.to_numpy(dtype=float)

    mintick = 1e-9
    atr_series = atr(frame, atr_len).to_numpy()
    plus_di, minus_di, adx_series = dmi(frame, adx_len)
    plus_di = plus_di.to_numpy()
    minus_di = minus_di.to_numpy()
    adx_series = adx_series.to_numpy()

    vol_mean = sma(pd.Series(vs), vol_len).to_numpy()
    vol_dev = stdev(pd.Series(vs), vol_len).to_numpy()
    vol_z = np.where(vol_dev > 0, (vs - vol_mean) / np.where(vol_dev > 0, vol_dev, mintick), 0.0)
    rel_vol = np.where(vol_mean > 0, vs / np.where(vol_mean > 0, vol_mean, mintick), 1.0)

    spread = hs - ls
    body = np.abs(cs - os)
    body_pct = body / np.maximum(spread, mintick)
    lower_wick = np.minimum(os, cs) - ls
    upper_wick = hs - np.maximum(os, cs)
    lower_wick_ratio = lower_wick / np.maximum(spread, mintick)
    upper_wick_ratio = upper_wick / np.maximum(spread, mintick)
    candle_atr = spread / np.maximum(atr_series, mintick)
    spread_mean = sma(pd.Series(spread), vol_len).to_numpy()
    spread_dev = stdev(pd.Series(spread), vol_len).to_numpy()
    spread_z = np.where(spread_dev > 0, (spread - spread_mean) / np.where(spread_dev > 0, spread_dev, mintick), 0.0)

    range_high = highest(pd.Series(hs), range_len).to_numpy()
    range_low = lowest(pd.Series(ls), range_len).to_numpy()
    range_high_prev = np.concatenate([[np.nan], range_high[:-1]])
    range_low_prev = np.concatenate([[np.nan], range_low[:-1]])

    ema_fast = ema(pd.Series(cs), 21).to_numpy()
    ema_slow = ema(pd.Series(cs), 55).to_numpy()
    ema_trend = ema(pd.Series(cs), 34).to_numpy()
    prior_ref = int(max(1, round(range_len / 2.0)))
    ema_trend_shift = np.concatenate([[np.nan] * prior_ref, ema_trend[:-prior_ref]])

    ph_series = pivothigh(pd.Series(hs), pivot_len, pivot_len).to_numpy()
    pl_series = pivotlow(pd.Series(ls), pivot_len, pivot_len).to_numpy()

    prev_high = np.concatenate([[np.nan], hs[:-1]])
    prev_low = np.concatenate([[np.nan], ls[:-1]])
    low_shift = np.concatenate([[np.nan] * pivot_len, ls[:-pivot_len]])
    high_shift = np.concatenate([[np.nan] * pivot_len, hs[:-pivot_len]])
    close_shift = np.concatenate([[np.nan] * pivot_len, cs[:-pivot_len]])
    open_shift = np.concatenate([[np.nan] * pivot_len, os[:-pivot_len]])
    vol_shift = np.concatenate([[np.nan] * pivot_len, vs[:-pivot_len]])
    vol_mean_shift = np.concatenate([[np.nan] * pivot_len, vol_mean[:-pivot_len]])

    # ---- output buffers ----------------------------------------------------
    out_state = np.full(n, _NEUTRAL, dtype=int)
    out_phase = np.full(n, _PH_NONE, dtype=int)
    acc_score = np.full(n, np.nan)
    dist_score = np.full(n, np.nan)
    cause_score = np.full(n, np.nan)
    range_quality = np.full(n, np.nan)
    confidence = np.full(n, np.nan)
    out_range_high = np.full(n, np.nan)
    out_range_low = np.full(n, np.nan)
    out_range_mid = np.full(n, np.nan)
    out_range_w = np.full(n, np.nan)
    out_range_atr = np.full(n, np.nan)
    pos = np.full(n, np.nan)
    near_low = np.zeros(n, dtype=bool)
    near_high = np.zeros(n, dtype=bool)
    range_ok = np.zeros(n, dtype=bool)
    out_locked = np.zeros(n, dtype=bool)
    bars_in_phase = np.full(n, 0, dtype=int)
    spring = np.zeros(n, dtype=bool)
    utad = np.zeros(n, dtype=bool)
    sos = np.zeros(n, dtype=bool)
    sow = np.zeros(n, dtype=bool)
    test_spring = np.zeros(n, dtype=bool)
    test_utad = np.zeros(n, dtype=bool)
    lps = np.zeros(n, dtype=bool)
    lpsy = np.zeros(n, dtype=bool)
    selling_climax = np.zeros(n, dtype=bool)
    buying_climax = np.zeros(n, dtype=bool)
    absorb_low = np.zeros(n, dtype=bool)
    absorb_high = np.zeros(n, dtype=bool)
    spring_quality = np.zeros(n, dtype=int)
    utad_quality = np.zeros(n, dtype=int)
    markup_started = np.zeros(n, dtype=bool)
    markdown_started = np.zeros(n, dtype=bool)
    last_spring_bar = np.full(n, -1, dtype="int64")
    last_utad_bar = np.full(n, -1, dtype="int64")
    last_sos_bar = np.full(n, -1, dtype="int64")
    last_sow_bar = np.full(n, -1, dtype="int64")
    last_lps_bar = np.full(n, -1, dtype="int64")
    last_lpsy_bar = np.full(n, -1, dtype="int64")

    # ---- state -------------------------------------------------------------
    wy_state = _NEUTRAL
    wy_phase = _PH_NONE
    phase_start_bar = -1
    locked_high = math.nan
    locked_low = math.nan
    locked_start_bar = -1
    last_swing_high = math.nan
    last_swing_low = math.nan
    last_spring_b = -1
    last_utad_b = -1
    last_sos_b = -1
    last_sow_b = -1
    last_test_spring_b = -1
    last_test_utad_b = -1
    last_lps_b = -1
    last_lpsy_b = -1
    last_sc_b = -1
    last_bc_b = -1
    last_spring_price = math.nan
    last_spring_vol = math.nan
    last_spring_q = 0
    last_utad_price = math.nan
    last_utad_vol = math.nan
    last_utad_q = 0
    spring_left_band = False
    utad_left_band = False
    bars_since_bull_ref = 10 ** 9
    bars_since_bear_ref = 10 ** 9
    bars_since_range_ok = 10 ** 9
    prev_cause = 0.0
    prev_state = _NEUTRAL
    prev_phase = _PH_NONE

    spring_buf: deque[int] = deque(maxlen=event_lookback)
    utad_buf: deque[int] = deque(maxlen=event_lookback)
    sos_buf: deque[int] = deque(maxlen=event_lookback)
    sow_buf: deque[int] = deque(maxlen=event_lookback)

    for i in range(n):
        atr_i = atr_series[i]
        adx_i = adx_series[i]
        # active range (locked if locked, else live) — step-4 definition
        range_locked = not math.isnan(locked_high) and not math.isnan(locked_low)
        ar_high = locked_high if range_locked else range_high[i]
        ar_low = locked_low if range_locked else range_low[i]
        ar_high_ref = locked_high if range_locked else (range_high_prev[i] if not math.isnan(range_high_prev[i]) else range_high[i])
        ar_low_ref = locked_low if range_locked else (range_low_prev[i] if not math.isnan(range_low_prev[i]) else range_low[i])
        ar_w = ar_high - ar_low
        ar_mid = (ar_high + ar_low) / 2.0
        ar_atr = ar_w / max(atr_i, mintick)
        pos_i = (cs[i] - ar_low) / max(ar_w, mintick)
        pos_clamped = min(1.0, max(0.0, pos_i))
        near_low_i = pos_clamped < 0.28
        near_high_i = pos_clamped > 0.72
        range_ok_i = (adx_i < adx_max and ar_atr < range_atr_max and i > range_len)

        lock_active = wy_state in (_ACC, _DIST, _MARKUP, _MARKDOWN)
        if lock_active and math.isnan(locked_high):
            locked_high = range_high[i]
            locked_low = range_low[i]
            locked_start_bar = i
        if wy_state == _NEUTRAL and not math.isnan(locked_high):
            locked_high = math.nan
            locked_low = math.nan
            locked_start_bar = -1

        # structure
        if not math.isnan(ph_series[i]):
            last_swing_high = ph_series[i]
        if not math.isnan(pl_series[i]):
            last_swing_low = pl_series[i]
        bull_bos = (not math.isnan(last_swing_high)) and (cs[i] > last_swing_high)
        bear_bos = (not math.isnan(last_swing_low)) and (cs[i] < last_swing_low)
        trend_up = ema_fast[i] > ema_slow[i]
        trend_down = ema_fast[i] < ema_slow[i]
        prior_trend_down = (not math.isnan(ema_trend_shift[i])) and (cs[i] < ema_trend_shift[i])
        prior_trend_up = (not math.isnan(ema_trend_shift[i])) and (cs[i] > ema_trend_shift[i])

        lwr = lower_wick_ratio[i]
        uwr = upper_wick_ratio[i]
        catr = candle_atr[i]
        rvol = rel_vol[i]
        vz = vol_z[i]

        # Spring / UTAD
        spring_depth = ar_low_ref - ls[i]
        spring_depth_pct = (spring_depth / ar_low_ref * 100.0) if (not math.isnan(ar_low_ref) and abs(ar_low_ref) > mintick) else 0.0
        spring_sweep_atr = spring_depth / max(atr_i, mintick)
        spring_depth_ok = spring_depth > atr_i * spring_atr_min or spring_depth_pct >= spring_pct_min
        spring_raw = range_ok_i and spring_depth > 0 and cs[i] > ar_low_ref
        spring_i = (spring_raw and spring_depth_ok and near_low_i and cs[i] > os[i]
                    and lwr >= min_wick_ratio and catr >= min_candle_atr)
        spring_q = _event_quality(spring_i, lwr, catr, rvol, spring_sweep_atr,
                                  spring_atr_min, strong_wick_ratio, strong_candle_atr, strong_vol_ratio)

        utad_depth = hs[i] - ar_high_ref
        utad_depth_pct = (utad_depth / ar_high_ref * 100.0) if (not math.isnan(ar_high_ref) and abs(ar_high_ref) > mintick) else 0.0
        utad_sweep_atr = utad_depth / max(atr_i, mintick)
        utad_depth_ok = utad_depth > atr_i * spring_atr_min or utad_depth_pct >= spring_pct_min
        utad_raw = range_ok_i and utad_depth > 0 and cs[i] < ar_high_ref
        utad_i = (utad_raw and utad_depth_ok and near_high_i and cs[i] < os[i]
                  and uwr >= min_wick_ratio and catr >= min_candle_atr)
        utad_q = _event_quality(utad_i, uwr, catr, rvol, utad_sweep_atr,
                                 spring_atr_min, strong_wick_ratio, strong_candle_atr, strong_vol_ratio)

        spring_fire = bool(spring_i and (last_spring_b < 0 or i - last_spring_b > event_cooldown))
        if spring_fire:
            last_spring_b = i
            last_spring_price = ls[i]
            last_spring_vol = rvol
            last_spring_q = spring_q
        utad_fire = bool(utad_i and (last_utad_b < 0 or i - last_utad_b > event_cooldown))
        if utad_fire:
            last_utad_b = i
            last_utad_price = hs[i]
            last_utad_vol = rvol
            last_utad_q = utad_q

        spring_buf.append(1 if spring_fire else 0)
        utad_buf.append(1 if utad_fire else 0)
        spring_memory = sum(spring_buf) > 0
        utad_memory = sum(utad_buf) > 0

        sos_i = (range_ok_i and spring_memory and cs[i] > ar_mid and cs[i] > prev_high[i]
                 and (not volume_is_real or vz > -0.5))
        sow_i = (range_ok_i and utad_memory and cs[i] < ar_mid and cs[i] < prev_low[i]
                 and (not volume_is_real or vz > -0.5))
        sos_fire = bool(sos_i and (last_sos_b < 0 or i - last_sos_b > event_cooldown))
        if sos_fire:
            last_sos_b = i
        sow_fire = bool(sow_i and (last_sow_b < 0 or i - last_sow_b > event_cooldown))
        if sow_fire:
            last_sow_b = i
        sos_buf.append(1 if sos_fire else 0)
        sow_buf.append(1 if sow_fire else 0)
        sos_memory = sum(sos_buf) > 0
        sow_memory = sum(sow_buf) > 0

        # Test of Spring/UTAD
        spring_test_window = (not math.isnan(last_spring_price) and last_spring_b >= 0
                              and i > last_spring_b and i - last_spring_b <= event_lookback)
        utad_test_window = (not math.isnan(last_utad_price) and last_utad_b >= 0
                            and i > last_utad_b and i - last_utad_b <= event_lookback)
        if spring_fire:
            spring_left_band = False
        elif (not math.isnan(last_spring_price)
              and (ls[i] > last_spring_price + atr_i * test_atr_window
                   or hs[i] < last_spring_price - atr_i * test_atr_window)):
            spring_left_band = True
        if utad_fire:
            utad_left_band = False
        elif (not math.isnan(last_utad_price)
              and (hs[i] < last_utad_price - atr_i * test_atr_window
                   or ls[i] > last_utad_price + atr_i * test_atr_window)):
            utad_left_band = True
        spring_test_price = (spring_test_window and spring_left_band
                             and ls[i] <= last_spring_price + atr_i * test_atr_window
                             and ls[i] >= last_spring_price - atr_i * test_atr_window
                             and cs[i] > last_spring_price)
        utad_test_price = (utad_test_window and utad_left_band
                           and hs[i] >= last_utad_price - atr_i * test_atr_window
                           and hs[i] <= last_utad_price + atr_i * test_atr_window
                           and cs[i] < last_utad_price)
        test_spring_i = (range_ok_i and spring_memory and not spring_i and spring_test_price
                         and (not volume_is_real or rvol <= (last_spring_vol if not math.isnan(last_spring_vol) else rvol) * test_vol_factor))
        test_utad_i = (range_ok_i and utad_memory and not utad_i and utad_test_price
                       and (not volume_is_real or rvol <= (last_utad_vol if not math.isnan(last_utad_vol) else rvol) * test_vol_factor))
        test_spring_fire = bool(test_spring_i and (last_test_spring_b < 0 or i - last_test_spring_b > event_cooldown))
        if test_spring_fire:
            last_test_spring_b = i
        test_utad_fire = bool(test_utad_i and (last_test_utad_b < 0 or i - last_test_utad_b > event_cooldown))
        if test_utad_fire:
            last_test_utad_b = i

        # LPS / LPSY (pivot-confirmed)
        pivot_low_conf = not math.isnan(pl_series[i])
        pivot_high_conf = not math.isnan(ph_series[i])
        pivot_low_vol_ratio = (vs[i] / vol_mean[i]) if (i >= pivot_len and not math.isnan(vol_mean_shift[i]) and vol_mean_shift[i] > 0) else 1.0
        pivot_high_vol_ratio = pivot_low_vol_ratio
        pivot_low_in_lps = (not math.isnan(low_shift[i]) and low_shift[i] > ar_low
                            and low_shift[i] < ar_low + ar_w * 0.45)
        pivot_high_in_lpsy = (not math.isnan(high_shift[i]) and high_shift[i] < ar_high
                              and high_shift[i] > ar_high - ar_w * 0.45)
        lps_i = (range_ok_i and sos_memory and not sos_i and pivot_low_conf and pivot_low_in_lps
                 and (not volume_is_real or pivot_low_vol_ratio <= lps_vol_factor)
                 and (not math.isnan(close_shift[i]) and not math.isnan(open_shift[i]) and close_shift[i] > open_shift[i]))
        lpsy_i = (range_ok_i and sow_memory and not sow_i and pivot_high_conf and pivot_high_in_lpsy
                  and (not volume_is_real or pivot_high_vol_ratio <= lps_vol_factor)
                  and (not math.isnan(close_shift[i]) and not math.isnan(open_shift[i]) and close_shift[i] < open_shift[i]))
        lps_fire = bool(lps_i and (last_lps_b < 0 or i - last_lps_b > event_cooldown))
        if lps_fire:
            last_lps_b = i
        lpsy_fire = bool(lpsy_i and (last_lpsy_b < 0 or i - last_lpsy_b > event_cooldown))
        if lpsy_fire:
            last_lpsy_b = i

        # Climax / absorption
        sc_i = (volume_is_real and prior_trend_down and vz > 1.5 and spread_z[i] > 1.0
                and body_pct[i] > 0.45 and ls[i] <= ar_low and cs[i] > ls[i] + spread[i] * 0.35)
        bc_i = (volume_is_real and prior_trend_up and vz > 1.5 and spread_z[i] > 1.0
                and body_pct[i] > 0.45 and hs[i] >= ar_high and cs[i] < hs[i] - spread[i] * 0.35)
        sc_fire = bool(sc_i and (last_sc_b < 0 or i - last_sc_b > event_cooldown))
        if sc_fire:
            last_sc_b = i
        bc_fire = bool(bc_i and (last_bc_b < 0 or i - last_bc_b > event_cooldown))
        if bc_fire:
            last_bc_b = i
        absorb_low_i = (volume_is_real and range_ok_i and near_low_i and vz > 0.75
                        and cs[i] > ls[i] + spread[i] * 0.5)
        absorb_high_i = (volume_is_real and range_ok_i and near_high_i and vz > 0.75
                         and cs[i] < hs[i] - spread[i] * 0.5)

        # ---- scores --------------------------------------------------------
        range_score = 20.0 if range_ok_i else 0.0
        acc_location = 18.0 if near_low_i else (10.0 if pos_clamped < 0.5 else 0.0)
        dist_location = 18.0 if near_high_i else (10.0 if pos_clamped > 0.5 else 0.0)
        acc_event = (15.0 + spring_q * 3.0 if spring_fire else
                     16.0 if test_spring_fire else
                     12.0 if sc_fire else 0.0)
        dist_event = (15.0 + utad_q * 3.0 if utad_fire else
                      16.0 if test_utad_fire else
                      12.0 if bc_fire else 0.0)
        acc_absorb = 18.0 if absorb_low_i else 0.0
        dist_absorb = 18.0 if absorb_high_i else 0.0
        acc_reclaim = 15.0 if cs[i] > ar_mid else 0.0
        dist_loss = 15.0 if cs[i] < ar_mid else 0.0
        acc_struct = 15.0 if (bull_bos or trend_up) else 0.0
        dist_struct = 15.0 if (bear_bos or trend_down) else 0.0
        acc_s = min(100.0, range_score + acc_location + acc_event + acc_absorb + acc_reclaim + acc_struct)
        dist_s = min(100.0, range_score + dist_location + dist_event + dist_absorb + dist_loss + dist_struct)

        range_quality_i = min(100.0,
                              (50.0 if adx_i < adx_max else 0.0)
                              + (40.0 if ar_atr < range_atr_max else 0.0)
                              + (10.0 if abs(pos_clamped - 0.5) < 0.35 else 0.0))

        vol_mean_slow = sma(pd.Series(vs), vol_len * 2).to_numpy()
        compression_score = 100.0 - min(100.0, ar_atr * 10.0)
        range_dwell_bars = (i - locked_start_bar) if (range_locked and locked_start_bar >= 0) else 0.0
        dwell_score = min(30.0, range_dwell_bars)
        swing_score = 20.0 if (not math.isnan(last_swing_high) and not math.isnan(last_swing_low)) else 0.0
        volume_dry_score = 15.0 if (volume_is_real and vol_mean[i] < vol_mean_slow[i]) else 0.0
        respect_score = 15.0 if (hs[i] <= ar_high_ref and ls[i] >= ar_low_ref) else 0.0
        cause_s = min(100.0, compression_score * 0.35 + dwell_score + swing_score
                      + volume_dry_score + respect_score)
        cause_building = cause_s >= 45.0 and (cause_s >= prev_cause or cause_s >= 65.0)

        # ---- state machine --------------------------------------------------
        bars_in_phase_i = 0 if phase_start_bar < 0 else i - phase_start_bar
        phase_can_advance = bars_in_phase_i >= min_phase_bars
        acc_bias = acc_s > dist_s and acc_s >= 45
        dist_bias = dist_s > acc_s and dist_s >= 45
        strong_acc = acc_s >= 65 and acc_s > dist_s + 15
        strong_dist = dist_s >= 65 and dist_s > acc_s + 15
        bull_breakout_raw = (cs[i] > ar_high_ref and acc_s > dist_s
                             and (bull_bos or spread_z[i] > 0.5 or cs[i] > ar_high_ref + atr_i * 0.1))
        bear_breakout_raw = (cs[i] < ar_low_ref and dist_s > acc_s
                             and (bear_bos or spread_z[i] > 0.5 or cs[i] < ar_low_ref - atr_i * 0.1))
        if cs[i] <= ar_high_ref:
            bars_since_bull_ref = 0
        else:
            bars_since_bull_ref += 1
        if cs[i] >= ar_low_ref:
            bars_since_bear_ref = 0
        else:
            bars_since_bear_ref += 1
        bull_breakout_confirmed = bars_since_bull_ref >= breakout_confirm_bars and bull_breakout_raw
        bear_breakout_confirmed = bars_since_bear_ref >= breakout_confirm_bars and bear_breakout_raw

        acc_phase_a = sc_fire or (volume_is_real and prior_trend_down and range_ok_i and near_low_i and vz > 1.0)
        acc_phase_b = range_ok_i and acc_bias and cause_building and not spring_fire and not sos_fire
        acc_phase_c = spring_fire or test_spring_fire
        acc_phase_d = sos_fire or lps_fire or (cs[i] > ar_mid and cs[i] <= ar_high)
        acc_phase_e = bull_breakout_confirmed
        dist_phase_a = bc_fire or (volume_is_real and prior_trend_up and range_ok_i and near_high_i and vz > 1.0)
        dist_phase_b = range_ok_i and dist_bias and cause_building and not utad_fire and not sow_fire
        dist_phase_c = utad_fire or test_utad_fire
        dist_phase_d = sow_fire or lpsy_fire or (cs[i] < ar_mid and cs[i] >= ar_low)
        dist_phase_e = bear_breakout_confirmed

        new_state = wy_state
        new_phase = wy_phase
        new_phase_start = phase_start_bar
        new_locked_high = locked_high
        new_locked_low = locked_low
        new_locked_start = locked_start_bar

        def _enter(state, phase, start):
            nonlocal new_state, new_phase, new_phase_start, new_locked_high, new_locked_low, new_locked_start
            new_state = state
            new_phase = phase
            new_phase_start = i
            new_locked_high = range_high[i]
            new_locked_low = range_low[i]
            new_locked_start = i

        if wy_state == _NEUTRAL:
            if acc_phase_a:
                _enter(_ACC, _PH_A, i)
            elif dist_phase_a:
                _enter(_DIST, _PH_A, i)
            elif strong_acc and range_ok_i and cause_building:
                _enter(_ACC, _PH_B, i)
            elif strong_dist and range_ok_i and cause_building:
                _enter(_DIST, _PH_B, i)
        elif wy_state == _ACC and phase_can_advance:
            if wy_phase == _PH_A and acc_phase_b:
                new_phase, new_phase_start = _PH_B, i
            elif wy_phase == _PH_B and acc_phase_c:
                new_phase, new_phase_start = _PH_C, i
            elif wy_phase == _PH_C and acc_phase_d:
                new_phase, new_phase_start = _PH_D, i
            elif wy_phase == _PH_D and acc_phase_e:
                new_state, new_phase, new_phase_start = _MARKUP, _PH_E, i
        elif wy_state == _DIST and phase_can_advance:
            if wy_phase == _PH_A and dist_phase_b:
                new_phase, new_phase_start = _PH_B, i
            elif wy_phase == _PH_B and dist_phase_c:
                new_phase, new_phase_start = _PH_C, i
            elif wy_phase == _PH_C and dist_phase_d:
                new_phase, new_phase_start = _PH_D, i
            elif wy_phase == _PH_D and dist_phase_e:
                new_state, new_phase, new_phase_start = _MARKDOWN, _PH_E, i

        hard_bull = (wy_state == _DIST and cs[i] > ar_high_ref and acc_s > dist_s + 25)
        hard_bear = (wy_state == _ACC and cs[i] < ar_low_ref and dist_s > acc_s + 25)
        if hard_bull:
            _enter(_ACC, _PH_B, i)
        if hard_bear:
            _enter(_DIST, _PH_B, i)

        if wy_state == _MARKUP and range_ok_i and bars_in_phase_i > 20 and cs[i] < ar_high_ref:
            new_state, new_phase, new_phase_start = _NEUTRAL, _PH_NONE, -1
        if wy_state == _MARKDOWN and range_ok_i and bars_in_phase_i > 20 and cs[i] > ar_low_ref:
            new_state, new_phase, new_phase_start = _NEUTRAL, _PH_NONE, -1

        in_cause = (wy_state == _ACC or wy_state == _DIST)
        bars_since_range_ok = 0 if range_ok_i else bars_since_range_ok + 1
        cause_stale = (reset_stuck_range and in_cause
                       and (bars_in_phase_i >= max_cause_bars or bars_since_range_ok >= stale_range_bars))
        if cause_stale:
            new_state, new_phase, new_phase_start = _NEUTRAL, _PH_NONE, -1

        cause_building_phase = (wy_state == _ACC or wy_state == _DIST) and (wy_phase == _PH_A or wy_phase == _PH_B)
        if refresh_cause_range and cause_building_phase and range_ok_i and not math.isnan(locked_high):
            new_locked_high = range_high[i]
            new_locked_low = range_low[i]

        wy_state, wy_phase, phase_start_bar = new_state, new_phase, new_phase_start
        locked_high, locked_low, locked_start_bar = new_locked_high, new_locked_low, new_locked_start

        markup_started_i = (wy_state == _MARKUP and wy_phase == _PH_E
                            and (prev_state != _MARKUP or prev_phase != _PH_E))
        markdown_started_i = (wy_state == _MARKDOWN and wy_phase == _PH_E
                              and (prev_state != _MARKDOWN or prev_phase != _PH_E))

        # ---- record ---------------------------------------------------------
        out_state[i] = wy_state
        out_phase[i] = wy_phase
        acc_score[i] = acc_s
        dist_score[i] = dist_s
        cause_score[i] = cause_s
        range_quality[i] = range_quality_i
        confidence[i] = (acc_s if wy_state in (_ACC, _MARKUP) else
                        dist_s if wy_state in (_DIST, _MARKDOWN) else max(acc_s, dist_s))
        out_range_high[i] = ar_high
        out_range_low[i] = ar_low
        out_range_mid[i] = ar_mid
        out_range_w[i] = ar_w
        out_range_atr[i] = ar_atr
        pos[i] = pos_i
        near_low[i] = near_low_i
        near_high[i] = near_high_i
        range_ok[i] = range_ok_i
        out_locked[i] = not math.isnan(locked_high)
        bars_in_phase[i] = 0 if phase_start_bar < 0 else i - phase_start_bar
        spring[i] = spring_i
        utad[i] = utad_i
        sos[i] = sos_i
        sow[i] = sow_i
        test_spring[i] = test_spring_i
        test_utad[i] = test_utad_i
        lps[i] = lps_i
        lpsy[i] = lpsy_i
        selling_climax[i] = sc_i
        buying_climax[i] = bc_i
        absorb_low[i] = absorb_low_i
        absorb_high[i] = absorb_high_i
        spring_quality[i] = spring_q
        utad_quality[i] = utad_q
        markup_started[i] = markup_started_i
        markdown_started[i] = markdown_started_i
        last_spring_bar[i] = last_spring_b
        last_utad_bar[i] = last_utad_b
        last_sos_bar[i] = last_sos_b
        last_sow_bar[i] = last_sow_b
        last_lps_bar[i] = last_lps_b
        last_lpsy_bar[i] = last_lpsy_b

        prev_cause = cause_s
        prev_state = wy_state
        prev_phase = wy_phase

    index = frame.index
    state_str = pd.Series([_STATE_NAME[s] for s in out_state], index=index)
    phase_str = pd.Series([_PHASE_NAME[p] for p in out_phase], index=index)
    phase_text_str = pd.Series([_phase_text(out_state[k], out_phase[k]) for k in range(n)], index=index)

    data: dict[str, pd.Series] = {
        "state": state_str,
        "phase": phase_str,
        "phase_text": phase_text_str,
        "acc_score": pd.Series(acc_score, index=index),
        "dist_score": pd.Series(dist_score, index=index),
        "cause_score": pd.Series(cause_score, index=index),
        "range_quality": pd.Series(range_quality, index=index),
        "confidence": pd.Series(confidence, index=index),
        "range_high": pd.Series(out_range_high, index=index),
        "range_low": pd.Series(out_range_low, index=index),
        "range_mid": pd.Series(out_range_mid, index=index),
        "range_w": pd.Series(out_range_w, index=index),
        "range_atr": pd.Series(out_range_atr, index=index),
        "pos": pd.Series(pos, index=index),
        "near_low": pd.Series(near_low, index=index, dtype=bool),
        "near_high": pd.Series(near_high, index=index, dtype=bool),
        "range_ok": pd.Series(range_ok, index=index, dtype=bool),
        "range_locked": pd.Series(out_locked, index=index, dtype=bool),
        "bars_in_phase": pd.Series(bars_in_phase, index=index),
        "spring": pd.Series(spring, index=index, dtype=bool),
        "utad": pd.Series(utad, index=index, dtype=bool),
        "sos": pd.Series(sos, index=index, dtype=bool),
        "sow": pd.Series(sow, index=index, dtype=bool),
        "test_spring": pd.Series(test_spring, index=index, dtype=bool),
        "test_utad": pd.Series(test_utad, index=index, dtype=bool),
        "lps": pd.Series(lps, index=index, dtype=bool),
        "lpsy": pd.Series(lpsy, index=index, dtype=bool),
        "selling_climax": pd.Series(selling_climax, index=index, dtype=bool),
        "buying_climax": pd.Series(buying_climax, index=index, dtype=bool),
        "absorption_low": pd.Series(absorb_low, index=index, dtype=bool),
        "absorption_high": pd.Series(absorb_high, index=index, dtype=bool),
        "spring_quality": pd.Series(spring_quality, index=index),
        "utad_quality": pd.Series(utad_quality, index=index),
        "markup_started": pd.Series(markup_started, index=index, dtype=bool),
        "markdown_started": pd.Series(markdown_started, index=index, dtype=bool),
        "last_spring_bar": pd.Series(last_spring_bar, index=index),
        "last_utad_bar": pd.Series(last_utad_bar, index=index),
        "last_sos_bar": pd.Series(last_sos_bar, index=index),
        "last_sow_bar": pd.Series(last_sow_bar, index=index),
        "last_lps_bar": pd.Series(last_lps_bar, index=index),
        "last_lpsy_bar": pd.Series(last_lpsy_bar, index=index),
    }
    return pd.DataFrame(data, index=index)
