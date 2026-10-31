"""Market Tradability Engine v2 [WavesUnchained] v2.5.4 的 Python 移植。

把市场质量压缩成三个基本属性，再由它们的组合推导八个状态：

* **STRUCTURE (0-100)** —— 现在有没有一个连贯的框架？被尊重的震荡区间，或干净的
  趋势。两者取**较大**而不是平均（平均会低估「明显只属于一种」的市场）。
* **ENERGY (0-100)** —— 波动强度在积累还是衰竭：波动扩张、动量加速度、K 线实体
  承诺度，以及移动转化为净进度的效率。
* **ACCEPTANCE (-100..100)** —— 新价格是被验证（收在高位并守住）还是被拒绝
  （影线打回），以及朝哪个方向。

状态（八个）
------------
0 NO TRADE / 1 BALANCE / 2 COMPRESSION / 3 READY / 4 BREAKOUT /
5 TREND / 6 EXHAUSTION / 7 AFTERMATH

关键机制
--------
* **武装区间**：区间一旦成立就冻结上下边界，突破判定对着冻结边界做 —— 否则滚动的
  区间窗口会被突破那几根「追着跑」，变成要求不断创新高/新低。
* **压缩武装**：READY 原本要求在**同一根**上既中性（压缩）又有方向压力，这在任何
  reactivity 档位下都自相矛盾。所以压缩只负责「武装」一个记忆状态，READY 只要在
  武装仍有效期间出现压力即可。
* **边界疲劳**：连续贴着边界算**一次**测试序列（带冷却），不重复计数；测试质量取
  冷却窗口内到达过的最深回旋，最终化后才计入均值（EMA 0.7/0.3）。
* **状态机**：分歧连续计数 + 最小停留时长滞回；BREAKOUT 有单独的 hold 窗口。

移植说明
--------
``syminfo.mintick`` 不可得，用 ``mintick`` 参数代替（默认 1e-8，仅用于避免除零）。
``syminfo.volumetype`` 不可得，用 ``volume_is_real`` 参数代替。绘图/表格/日志不移植。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    boolean,
    choice,
    dmi,
    ema,
    highest,
    integer,
    lowest,
    number,
    ohlcv,
    percentrank,
    rma,
    sma,
    wma,
)

__all__ = ["market_tradability_engine_v2", "NO_TRADE_REASONS", "STATE_NAMES"]

STATE_NO_TRADE, STATE_BALANCE, STATE_COMPRESSION, STATE_READY = 0, 1, 2, 3
STATE_BREAKOUT, STATE_TREND, STATE_EXHAUSTION, STATE_AFTERMATH = 4, 5, 6, 7

STATE_NAMES = (
    "NO_TRADE", "BALANCE", "COMPRESSION", "READY",
    "BREAKOUT", "TREND", "EXHAUSTION", "AFTERMATH",
)

NO_TRADE_REASONS = (
    "NONE",
    "WARMUP",
    "NO_FRAMEWORK",
    "FAILED_BREAKOUT_UP",
    "FAILED_BREAKOUT_DOWN",
    "BREAKOUT_HOLD_PENDING",
    "BREAKOUT_ENERGY_LOW",
    "BREAKOUT_ACCEPTANCE_WEAK",
    "BREAKOUT_VOLUME_WEAK",
    "POST_BREAKOUT_LOCK",
    "RANGE_TOO_WIDE",
    "READY_ENERGY_FLAT",
    "READY_BOUNDARY_STRESS_MISSING",
    "READY_NO_BOUNDARY_PRESSURE",
    "UPPER_BOUNDARY_FATIGUED",
    "LOWER_BOUNDARY_FATIGUED",
    "RANGE_EXPANSION_UNCONFIRMED",
    "RANGE_MID_ENERGY",
    "RANGE_DIRECTIONAL_UNCONFIRMED",
    "RANGE_UNCLASSIFIED",
    "TREND_ENERGY_LOW",
    "TREND_ACCEPTANCE_WEAK",
    "TREND_DIRECTION_MISMATCH",
)
_REASON_CODE = {name: k for k, name in enumerate(NO_TRADE_REASONS)}

_PRESET_LENGTHS = {
    "Very Fast": (10, 16, 18, 10, 8, 80),
    "Fast": (14, 22, 25, 12, 10, 110),
    "Balanced": (20, 30, 34, 14, 14, 150),
    "Calm": (30, 45, 50, 18, 18, 220),
    "Position": (45, 70, 80, 21, 21, 350),
}

# reactivity → (structure, energy_low, energy_very_low, energy_high,
#               acceptance_neutral, acceptance_pressure, acceptance_strong,
#               breakout_acceptance, breakout_acceptance_bars,
#               confirmation_bars, minimum_state_bars)
_REACTIVITY = {
    "Early": (42.0, 36.0, 23.0, 43.0, 12.0, 16.0, 22.0, 8.0, 1, 1, 1),
    "Balanced": (55.0, 30.0, 18.0, 53.0, 18.0, 24.0, 30.0, 15.0, 2, 2, 3),
    "Conservative": (65.0, 24.0, 13.0, 63.0, 24.0, 32.0, 38.0, 22.0, 3, 3, 5),
}

_BASELINE_TYPES = ("EMA", "SMA", "RMA", "WMA", "SuperSmoother", "T3", "KAMA", "JMA Approx")


def _clamp(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.maximum(low, np.minimum(high, values))


def _safe_div(numerator, denominator, fallback: float) -> np.ndarray:
    """Pine 的 ``f_safeDiv``：分母绝对值不超过 1e-7（或 na）时取 fallback。"""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    usable = np.abs(den) > 0.0000001
    return np.where(usable, num / np.where(usable, den, 1.0), fallback)


def _scale(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    span = upper - lower
    if span == 0.0:
        return np.zeros_like(values, dtype=float)
    return _clamp((values - lower) / span, 0.0, 1.0)


def _inverse_scale(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return 1.0 - _scale(values, lower, upper)


def _shift(values: np.ndarray, steps: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if steps < len(values):
        out[steps:] = values[:-steps]
    return out


def _ema_na(series: pd.Series, length: int) -> pd.Series:
    """``ta.ema`` 的 na 容忍版本（跳过开头的暖机期 na）。"""
    values = series.astype(float)
    out = pd.Series(np.nan, index=values.index, dtype=float)
    positions = np.flatnonzero(values.notna().to_numpy())
    if positions.size == 0:
        return out
    start = int(positions[0])
    tail = values.iloc[start:]
    if tail.isna().any():
        tail = tail.ffill()
    out.iloc[start:] = ema(tail.reset_index(drop=True), length).to_numpy()
    return out


def _supersmoother(values: np.ndarray, length: int) -> np.ndarray:
    a1 = math.exp(-1.414 * math.pi / length)
    b1 = 2.0 * a1 * math.cos(1.414 * math.pi / length)
    c2 = b1
    c3 = -(a1 * a1)
    c1 = 1.0 - c2 - c3
    out = np.full(len(values), np.nan)
    prev1 = 0.0
    prev2 = 0.0
    prev_x = 0.0
    for i, x in enumerate(values):
        if not np.isfinite(x):
            out[i] = np.nan
            continue
        current = c1 * (x + prev_x) / 2.0 + c2 * prev1 + c3 * prev2
        out[i] = current
        prev2, prev1, prev_x = prev1, current, x
    return out


def _kama(values: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    previous = math.nan
    for i in range(len(values)):
        x = values[i]
        if i < length or not np.isfinite(x):
            if np.isfinite(x):
                previous = x
            out[i] = previous
            continue
        window = values[i - length:i + 1]
        change = abs(x - values[i - length])
        volatility = np.nansum(np.abs(np.diff(window)))
        er = change / volatility if volatility != 0 else 0.0
        sc = (er * (2.0 / 3.0 - 2.0 / 31.0) + 2.0 / 31.0) ** 2
        base = previous if np.isfinite(previous) else x
        previous = base + sc * (x - base)
        out[i] = previous
    return out


def _jma(values: np.ndarray, length: int, phase: int, power: float) -> np.ndarray:
    out = np.full(len(values), np.nan)
    phase_ratio = 0.5 if phase < -100 else (2.5 if phase > 100 else phase / 100.0 + 1.5)
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2.0)
    alpha = beta ** power
    e0 = e1 = e2 = jma_val = math.nan
    for i, x in enumerate(values):
        if not np.isfinite(x):
            out[i] = jma_val
            continue
        prev_e0 = e0 if np.isfinite(e0) else x
        prev_e1 = e1 if np.isfinite(e1) else 0.0
        prev_jma = jma_val if np.isfinite(jma_val) else x
        prev_e2 = e2 if np.isfinite(e2) else 0.0
        e0 = (1.0 - alpha) * x + alpha * prev_e0
        e1 = (x - e0) * (1.0 - beta) + beta * prev_e1
        e2 = (e0 + phase_ratio * e1 - prev_jma) * (1.0 - alpha) ** 2 + alpha ** 2 * prev_e2
        jma_val = prev_jma + e2
        out[i] = jma_val
    return out


def _smooth(values: np.ndarray, length: int, mode: str, phase: int, power: float) -> np.ndarray:
    series = pd.Series(values)
    if mode == "SMA":
        return sma(series, length).to_numpy(dtype=float)
    if mode == "RMA":
        return rma(series, length).to_numpy(dtype=float)
    if mode == "WMA":
        return wma(series, length).to_numpy(dtype=float)
    if mode == "SuperSmoother":
        return _supersmoother(values, length)
    if mode == "T3":
        b = 0.7
        e1 = _ema_na(series, length)
        e2 = _ema_na(e1, length)
        e3 = _ema_na(e2, length)
        e4 = _ema_na(e3, length)
        e5 = _ema_na(e4, length)
        e6 = _ema_na(e5, length)
        c1 = -(b * b * b)
        c2 = 3.0 * b * b + 3.0 * b * b * b
        c3 = -6.0 * b * b - 3.0 * b - 3.0 * b * b * b
        c4 = 1.0 + 3.0 * b + b * b * b + 3.0 * b * b
        return (c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3).to_numpy(dtype=float)
    if mode == "KAMA":
        return _kama(values, length)
    if mode == "JMA Approx":
        return _jma(values, length, phase, power)
    return ema(series, length).to_numpy(dtype=float)


def market_tradability_engine_v2(
    frame: pd.DataFrame,
    *,
    preset: str = "Balanced",
    reactivity: str = "Balanced",
    custom_quality_length: int = 20,
    custom_range_length: int = 30,
    custom_baseline_length: int = 34,
    custom_atr_length: int = 14,
    custom_adx_length: int = 14,
    custom_rank_length: int = 150,
    custom_structure_threshold: float = 55.0,
    maximum_range_atr: float = 7.0,
    custom_energy_low_threshold: float = 35.0,
    custom_energy_very_low_threshold: float = 22.0,
    custom_energy_high_threshold: float = 60.0,
    custom_acceptance_neutral_band: float = 18.0,
    custom_acceptance_pressure_threshold: float = 24.0,
    custom_acceptance_strong_threshold: float = 30.0,
    custom_breakout_acceptance_threshold: float = 15.0,
    breakout_buffer_atr: float = 0.12,
    custom_breakout_acceptance_bars: int = 2,
    require_volume_confirmation: bool = False,
    volume_confirmation_ratio: float = 1.05,
    boundary_touch_distance_atr: float = 0.20,
    boundary_test_cooldown_bars: int = 0,
    boundary_fresh_test_maximum: int = 2,
    boundary_stressed_test_minimum: int = 4,
    weak_rejection_threshold: float = 0.25,
    require_boundary_stress_for_ready: bool = False,
    detect_failed_breakouts: bool = True,
    failed_breakout_reentry_atr: float = 0.10,
    custom_confirmation_bars: int = 2,
    custom_minimum_state_bars: int = 3,
    breakout_hold_bars: int = 3,
    post_breakout_lock_bars: int = 10,
    impulse_lookback: int = 10,
    recent_impulse_energy_threshold: float = 55.0,
    recent_impulse_acceptance_threshold: float = 45.0,
    aftermath_energy_drop_pct: float = 50.0,
    baseline_type: str = "EMA",
    jma_phase: int = 50,
    jma_power: float = 2.0,
    mintick: float = 1e-8,
    volume_is_real: bool | None = None,
) -> pd.DataFrame:
    """Market Tradability Engine v2：逐 bar 的三分数 + 八态市场机。"""
    preset = choice("preset", preset, tuple(_PRESET_LENGTHS) + ("Custom",))
    reactivity = choice("reactivity", reactivity, ("Early", "Balanced", "Conservative", "Custom"))
    custom_quality_length = integer("custom_quality_length", custom_quality_length, 8, 100)
    custom_range_length = integer("custom_range_length", custom_range_length, 10, 200)
    custom_baseline_length = integer("custom_baseline_length", custom_baseline_length, 5, 200)
    custom_atr_length = integer("custom_atr_length", custom_atr_length, 5, 100)
    custom_adx_length = integer("custom_adx_length", custom_adx_length, 5, 50)
    custom_rank_length = integer("custom_rank_length", custom_rank_length, 50, 500)
    custom_structure_threshold = number("custom_structure_threshold", custom_structure_threshold, 20.0, 90.0)
    maximum_range_atr = number("maximum_range_atr", maximum_range_atr, 2.0, 20.0)
    custom_energy_low_threshold = number("custom_energy_low_threshold", custom_energy_low_threshold, 5.0, 95.0)
    custom_energy_very_low_threshold = number("custom_energy_very_low_threshold", custom_energy_very_low_threshold, 1.0, 90.0)
    custom_energy_high_threshold = number("custom_energy_high_threshold", custom_energy_high_threshold, 20.0, 95.0)
    custom_acceptance_neutral_band = number("custom_acceptance_neutral_band", custom_acceptance_neutral_band, 0.0, 60.0)
    custom_acceptance_pressure_threshold = number("custom_acceptance_pressure_threshold", custom_acceptance_pressure_threshold, 5.0, 80.0)
    custom_acceptance_strong_threshold = number("custom_acceptance_strong_threshold", custom_acceptance_strong_threshold, 10.0, 90.0)
    custom_breakout_acceptance_threshold = number("custom_breakout_acceptance_threshold", custom_breakout_acceptance_threshold, 0.0, 60.0)
    breakout_buffer_atr = number("breakout_buffer_atr", breakout_buffer_atr, 0.0, 1.0)
    custom_breakout_acceptance_bars = integer("custom_breakout_acceptance_bars", custom_breakout_acceptance_bars, 1, 10)
    require_volume_confirmation = boolean("require_volume_confirmation", require_volume_confirmation)
    volume_confirmation_ratio = number("volume_confirmation_ratio", volume_confirmation_ratio, 0.5, 3.0)
    boundary_touch_distance_atr = number("boundary_touch_distance_atr", boundary_touch_distance_atr, 0.0, 1.0)
    boundary_test_cooldown_bars = integer("boundary_test_cooldown_bars", boundary_test_cooldown_bars, 0, 100)
    boundary_fresh_test_maximum = integer("boundary_fresh_test_maximum", boundary_fresh_test_maximum, 0, 20)
    boundary_stressed_test_minimum = integer("boundary_stressed_test_minimum", boundary_stressed_test_minimum, 1, 20)
    weak_rejection_threshold = number("weak_rejection_threshold", weak_rejection_threshold, 0.0, 1.0)
    require_boundary_stress_for_ready = boolean("require_boundary_stress_for_ready", require_boundary_stress_for_ready)
    detect_failed_breakouts = boolean("detect_failed_breakouts", detect_failed_breakouts)
    failed_breakout_reentry_atr = number("failed_breakout_reentry_atr", failed_breakout_reentry_atr, 0.0, 1.0)
    custom_confirmation_bars = integer("custom_confirmation_bars", custom_confirmation_bars, 1, 10)
    custom_minimum_state_bars = integer("custom_minimum_state_bars", custom_minimum_state_bars, 0, 20)
    breakout_hold_bars = integer("breakout_hold_bars", breakout_hold_bars, 0, 50)
    post_breakout_lock_bars = integer("post_breakout_lock_bars", post_breakout_lock_bars, 0, 100)
    impulse_lookback = integer("impulse_lookback", impulse_lookback, 1, 100)
    recent_impulse_energy_threshold = number("recent_impulse_energy_threshold", recent_impulse_energy_threshold, 0.0, 100.0)
    recent_impulse_acceptance_threshold = number("recent_impulse_acceptance_threshold", recent_impulse_acceptance_threshold, 0.0, 100.0)
    aftermath_energy_drop_pct = number("aftermath_energy_drop_pct", aftermath_energy_drop_pct, 0.0, 100.0)
    baseline_type = choice("baseline_type", baseline_type, _BASELINE_TYPES)
    jma_phase = integer("jma_phase", jma_phase, -100, 100)
    jma_power = number("jma_power", jma_power, 0.1, 5.0)

    # --- 预设解析 -----------------------------------------------------------
    if preset == "Custom":
        quality_length, range_length, baseline_length = custom_quality_length, custom_range_length, custom_baseline_length
        atr_length, adx_length, rank_length = custom_atr_length, custom_adx_length, custom_rank_length
    else:
        quality_length, range_length, baseline_length, atr_length, adx_length, rank_length = _PRESET_LENGTHS[preset]

    if reactivity == "Custom":
        (structure_threshold, energy_low, energy_very_low, energy_high, acceptance_neutral,
         acceptance_pressure, acceptance_strong, breakout_acceptance, breakout_acceptance_bars,
         confirmation_bars, minimum_state_bars) = (
            custom_structure_threshold, custom_energy_low_threshold, custom_energy_very_low_threshold,
            custom_energy_high_threshold, custom_acceptance_neutral_band, custom_acceptance_pressure_threshold,
            custom_acceptance_strong_threshold, custom_breakout_acceptance_threshold,
            custom_breakout_acceptance_bars, custom_confirmation_bars, custom_minimum_state_bars,
        )
    else:
        (structure_threshold, energy_low, energy_very_low, energy_high, acceptance_neutral,
         acceptance_pressure, acceptance_strong, breakout_acceptance, breakout_acceptance_bars,
         confirmation_bars, minimum_state_bars) = _REACTIVITY[reactivity]

    open_s, high_s, low_s, close_s, volume_s = ohlcv(frame, "open", "high", "low", "close", "volume")
    open_a = open_s.to_numpy(dtype=float)
    high = high_s.to_numpy(dtype=float)
    low = low_s.to_numpy(dtype=float)
    close = close_s.to_numpy(dtype=float)
    volume = volume_s.to_numpy(dtype=float)
    idx = frame.index
    n = len(frame)

    # --- 基础计算 -----------------------------------------------------------
    short_window = max(3, quality_length // 4)
    atr_a = atr(frame, atr_length).to_numpy(dtype=float)
    atr_reference = ema(pd.Series(atr_a, index=idx), quality_length * 2).to_numpy(dtype=float)
    atr_ratio = _safe_div(atr_a, atr_reference, 1.0)

    baseline = _smooth(close, baseline_length, baseline_type, jma_phase, jma_power)
    baseline_slope = baseline - _shift(baseline, quality_length)
    normalized_slope = _safe_div(baseline_slope, atr_a * math.sqrt(quality_length), 0.0)

    plus_di, minus_di, adx = dmi(frame, adx_length)
    plus_di = plus_di.to_numpy(dtype=float)
    minus_di = minus_di.to_numpy(dtype=float)
    adx = adx.to_numpy(dtype=float)

    floor_range = mintick if mintick and mintick > 0 else 1e-8
    bar_range = np.maximum(high - low, floor_range)
    body_ratio = _safe_div(np.abs(close - open_a), bar_range, 0.0)
    upper_wick = high - np.maximum(open_a, close)
    lower_wick = np.minimum(open_a, close) - low
    wick_ratio = _safe_div(upper_wick + lower_wick, bar_range, 0.0)
    average_body_ratio = ema(pd.Series(body_ratio, index=idx), quality_length).to_numpy(dtype=float)
    average_wick_ratio = ema(pd.Series(wick_ratio, index=idx), quality_length).to_numpy(dtype=float)

    price_travel = pd.Series(np.abs(np.diff(close, prepend=close[0])), index=idx).rolling(
        quality_length, min_periods=quality_length).sum().to_numpy(dtype=float)
    net_movement = np.abs(close - _shift(close, quality_length))
    efficiency_ratio = _safe_div(net_movement, price_travel, 0.0)

    prev_close = _shift(close, 1)
    direction_step = np.where(close > prev_close, 1.0, np.where(close < prev_close, -1.0, 0.0))
    direction_step = np.where(np.isfinite(prev_close), direction_step, 0.0)
    signed_consistency = _safe_div(
        pd.Series(direction_step, index=idx).rolling(quality_length, min_periods=quality_length).sum().to_numpy(),
        quality_length, 0.0)

    cross_flag = (close > baseline) & (prev_close <= _shift(baseline, 1))
    cross_flag = cross_flag | ((close < baseline) & (prev_close >= _shift(baseline, 1)))
    cross_flag = cross_flag & np.isfinite(prev_close)
    baseline_cross_rate = _safe_div(
        pd.Series(cross_flag.astype(float), index=idx).rolling(quality_length, min_periods=quality_length).sum().to_numpy(),
        quality_length, 0.0)

    close_displacement = _safe_div(close - baseline, atr_a, 0.0)

    up_close_strength = _safe_div(close - low, bar_range, 0.5) * 2.0 - 1.0
    down_close_strength = _safe_div(high - close, bar_range, 0.5) * 2.0 - 1.0
    bar_acceptance = np.where(direction_step > 0.0, up_close_strength,
                              np.where(direction_step < 0.0, -down_close_strength, 0.0))
    average_bar_acceptance = ema(pd.Series(bar_acceptance, index=idx), quality_length).to_numpy(dtype=float)

    # --- 区间结构 -----------------------------------------------------------
    rolling_range_high = highest(pd.Series(_shift(high, 1), index=idx), range_length).to_numpy(dtype=float)
    rolling_range_low = lowest(pd.Series(_shift(low, 1), index=idx), range_length).to_numpy(dtype=float)
    rolling_range_width = rolling_range_high - rolling_range_low
    range_width_atr = _safe_div(rolling_range_width, atr_a, maximum_range_atr * 2.0)
    range_position = _clamp(_safe_div(close - rolling_range_low, rolling_range_width, 0.5), 0.0, 1.0)
    upper_boundary_change = _safe_div(np.abs(rolling_range_high - _shift(rolling_range_high, 1)), atr_a, 0.0)
    lower_boundary_change = _safe_div(np.abs(rolling_range_low - _shift(rolling_range_low, 1)), atr_a, 0.0)
    boundary_stability = 1.0 - _clamp((upper_boundary_change + lower_boundary_change) * 0.5, 0.0, 1.0)

    # --- 波动率 -------------------------------------------------------------
    basis = sma(close_s, quality_length).to_numpy(dtype=float)
    deviation = close_s.rolling(quality_length, min_periods=quality_length).std(ddof=0).to_numpy(dtype=float)
    band_width = _safe_div(deviation * 4.0, basis, 0.0)
    band_width_reference = ema(pd.Series(band_width, index=idx), quality_length * 2).to_numpy(dtype=float)
    band_width_ratio = _safe_div(band_width, band_width_reference, 1.0)
    volatility_expansion = percentrank(pd.Series(atr_ratio, index=idx), rank_length).to_numpy(dtype=float)

    # --- 量能 ---------------------------------------------------------------
    if volume_is_real is None:
        volume_is_real = bool(np.isfinite(volume).any() and np.nansum(volume) > 0)
    else:
        volume_is_real = boolean("volume_is_real", volume_is_real)
    volume_reference = ema(pd.Series(np.nan_to_num(volume, nan=0.0), index=idx), quality_length).to_numpy(dtype=float)
    volume_ratio = np.ones(n) if not volume_is_real else _safe_div(volume, volume_reference, 1.0)
    volume_confirmed = (not require_volume_confirmation) or (not volume_is_real) or (volume_ratio >= volume_confirmation_ratio)

    # --- 结构分 -------------------------------------------------------------
    sustained_stability = ema(pd.Series(boundary_stability, index=idx), quality_length).to_numpy(dtype=float)
    range_compactness = _inverse_scale(range_width_atr, maximum_range_atr * 0.5, maximum_range_atr)
    range_structure_quality = np.sqrt(np.maximum(sustained_stability, 0.0) * np.maximum(range_compactness, 0.0)) * 100.0

    trend_cross_quality = percentrank(pd.Series(-baseline_cross_rate, index=idx), rank_length).to_numpy(dtype=float)
    trend_efficiency_quality = percentrank(pd.Series(efficiency_ratio, index=idx), rank_length).to_numpy(dtype=float)
    trend_adx_quality = percentrank(pd.Series(adx, index=idx), rank_length).to_numpy(dtype=float)
    trend_slope_quality = percentrank(pd.Series(np.abs(normalized_slope), index=idx), rank_length).to_numpy(dtype=float)
    trend_consistency_quality = percentrank(pd.Series(np.abs(signed_consistency), index=idx), rank_length).to_numpy(dtype=float)
    trend_structure_quality = (trend_cross_quality * 0.25 + trend_efficiency_quality * 0.25
                              + trend_adx_quality * 0.20 + trend_slope_quality * 0.20
                              + trend_consistency_quality * 0.10)
    structure_score = np.fmax(range_structure_quality, trend_structure_quality)
    range_framework = range_structure_quality >= trend_structure_quality

    # --- 接受度分 -----------------------------------------------------------
    slope_direction = _clamp(normalized_slope / 0.75, -1.0, 1.0)
    baseline_direction = _clamp(close_displacement / 2.0, -1.0, 1.0)
    di_direction = _safe_div(plus_di - minus_di, plus_di + minus_di, 0.0)
    range_pressure = _clamp((range_position - 0.5) * 2.0, -1.0, 1.0)
    acceptance_score = _clamp((
        slope_direction * 0.22 + baseline_direction * 0.16 + di_direction * 0.16
        + signed_consistency * 0.14 + range_pressure * 0.12 + average_bar_acceptance * 0.20
    ) * 100.0, -100.0, 100.0)

    # --- 能量分 -------------------------------------------------------------
    energy_efficiency = percentrank(pd.Series(efficiency_ratio, index=idx), rank_length).to_numpy(dtype=float)
    acceptance_magnitude_change = np.abs(acceptance_score) - np.abs(_shift(acceptance_score, short_window))
    energy_momentum_accel = _clamp(acceptance_magnitude_change / 20.0, 0.0, 1.0) * 100.0
    candle_commitment = percentrank(pd.Series(average_body_ratio, index=idx), rank_length).to_numpy(dtype=float)
    energy_score = _clamp(volatility_expansion * 0.30 + energy_efficiency * 0.30
                          + energy_momentum_accel * 0.20 + candle_commitment * 0.20, 0.0, 100.0)

    # --- 预计算（逐 bar 复用） ----------------------------------------------
    data_ready_threshold = max(max(range_length * 2 + 5, quality_length * 3 + 5), rank_length)
    has_framework = structure_score >= structure_threshold
    atr_ratio_trough = lowest(pd.Series(atr_ratio, index=idx), short_window).to_numpy(dtype=float)
    band_trough = lowest(pd.Series(band_width_ratio, index=idx), short_window).to_numpy(dtype=float)
    energy_building = (atr_ratio > atr_ratio_trough * 1.03) | (band_width_ratio > band_trough * 1.03)
    near_upper = range_position >= 0.68
    near_lower = range_position <= 0.32
    bullish_pressure = (acceptance_score >= acceptance_pressure) & near_upper
    bearish_pressure = (acceptance_score <= -acceptance_pressure) & near_lower

    bullish_trend = has_framework & (~range_framework) & (energy_score >= energy_high) & \
        (acceptance_score >= acceptance_strong) & (close > baseline)
    bearish_trend = has_framework & (~range_framework) & (energy_score >= energy_high) & \
        (acceptance_score <= -acceptance_strong) & (close < baseline)
    trend_condition = bullish_trend | bearish_trend
    extension_from_baseline = np.abs(_safe_div(close - baseline, atr_a, 0.0))
    energy_decay = energy_score < _shift(energy_score, short_window)
    wick_rank = percentrank(pd.Series(average_wick_ratio, index=idx), rank_length).to_numpy(dtype=float)
    atr_rank = percentrank(pd.Series(atr_ratio, index=idx), rank_length).to_numpy(dtype=float)
    exhaustion_condition = trend_condition & (extension_from_baseline >= 2.6) & (
        energy_decay | (wick_rank >= 80.0) | (atr_rank >= 90.0))

    impulse_bar = (energy_score >= recent_impulse_energy_threshold) & \
        (np.abs(acceptance_score) >= recent_impulse_acceptance_threshold)
    impulse_direction = np.zeros(n, dtype=np.int64)
    impulse_energy = np.full(n, np.nan)
    last_direction = 0
    last_energy = math.nan
    last_impulse_bar = -1
    bars_since_impulse = np.full(n, np.inf)
    for i in range(n):
        if impulse_bar[i]:
            last_direction = 1 if acceptance_score[i] > 0.0 else -1
            last_energy = energy_score[i]
            last_impulse_bar = i
        impulse_direction[i] = last_direction
        impulse_energy[i] = last_energy
        bars_since_impulse[i] = (i - last_impulse_bar) if last_impulse_bar >= 0 else np.inf
    recent_impulse = bars_since_impulse <= impulse_lookback
    impulse_aftermath = recent_impulse & (energy_score <= impulse_energy * (1.0 - aftermath_energy_drop_pct / 100.0)) & \
        (atr_ratio <= _shift(atr_ratio, 3))

    # --- 逐 bar 顺序状态机 ---------------------------------------------------
    def zeros(**names):
        return {name: np.zeros(n, dtype=np.int64) for name in names}

    def nans(**names):
        return {name: np.full(n, np.nan) for name in names}

    ints = zeros(
        state=0, candidate_state=0, pending_state=0, pending_state_bars=0, state_bars=0,
        invalid_state_bars=0, active_direction=0, breakout_bars_remaining=0,
        breakout_armed=0, compression_armed=0, bars_since_balance=0,
        upper_tests=0, lower_tests=0, upper_fresh=0, lower_fresh=0,
        bullish_stress=0, bearish_stress=0, state_changed=0, left_tradable=0,
        no_trade_reason=0, data_ready=0, has_framework=0, balance_allowed=0,
        established_balance=0, established_compression=0, energy_building=0,
        bullish_pressure=0, bearish_pressure=0, ready_condition=0, trend_condition=0,
        exhaustion_condition=0, impulse_aftermath=0, impulse_direction=0,
        raw_bullish_breakout=0, raw_bearish_breakout=0, bullish_breakout=0, bearish_breakout=0,
        failed_breakout_bull=0, failed_breakout_bear=0, volume_confirmed=0,
        range_framework=0, upper_contact=0, lower_contact=0, entered_ready=0,
        entered_breakout=0, entered_trend=0, entered_no_trade=0,
    )
    floats = nans(
        breakout_reference_high=0.0, breakout_reference_low=0.0,
        armed_range_high=0.0, armed_range_low=0.0,
        upper_rejection_quality=0.0, lower_rejection_quality=0.0,
        avg_upper_rejection_quality=0.0, avg_lower_rejection_quality=0.0,
        upper_fatigue=0.0, lower_fatigue=0.0, dominant_fatigue=0.0,
    )

    cooldown = boundary_test_cooldown_bars if boundary_test_cooldown_bars > 0 else max(2, quality_length // 5)
    frozen_span = max(boundary_stressed_test_minimum - boundary_fresh_test_maximum, 1)

    bars_since_breakout = 100000
    armed_high = math.nan
    armed_low = math.nan
    breakout_armed = False
    compression_armed = False
    bars_since_balance = 100000
    bull_streak = 0
    bear_streak = 0
    upper_tests = 0
    lower_tests = 0
    last_upper_contact = -1
    last_lower_contact = -1
    upper_rejection = math.nan
    lower_rejection = math.nan
    avg_upper_rejection = math.nan
    avg_lower_rejection = math.nan
    pending_upper_bar = -1
    pending_lower_bar = -1
    pending_upper_boundary = math.nan
    pending_lower_boundary = math.nan
    pending_upper_mid = math.nan
    pending_lower_mid = math.nan
    pending_upper_best = math.nan
    pending_lower_best = math.nan
    prev_breakout_armed = False

    market_state = STATE_NO_TRADE
    pending_state = STATE_NO_TRADE
    pending_state_bars = 0
    invalid_state_bars = 0
    state_bars = 0
    breakout_bars_remaining = 0
    active_direction = 0

    for i in range(n):
        ready_flag = i > data_ready_threshold
        framework = bool(has_framework[i])
        in_range_framework = bool(range_framework[i])
        energy_i = energy_score[i]
        acceptance_i = acceptance_score[i]
        balance_allowed = bars_since_breakout > post_breakout_lock_bars
        established_balance = (ready_flag and balance_allowed and framework and in_range_framework
                               and np.isfinite(energy_i) and energy_i < energy_low
                               and np.isfinite(acceptance_i) and abs(acceptance_i) < acceptance_neutral
                               and range_width_atr[i] <= maximum_range_atr)
        established_compression = established_balance and energy_i < energy_very_low

        # 武装区间复制（在突破条出现之前冻结边界）
        bars_since_balance = 0 if established_balance else bars_since_balance + 1
        if established_balance and bull_streak == 0 and bear_streak == 0:
            armed_high = rolling_range_high[i]
            armed_low = rolling_range_low[i]
            breakout_armed = True
        if established_compression:
            compression_armed = True
        if bars_since_balance > range_length:
            breakout_armed = False
            compression_armed = False
        if not in_range_framework:
            compression_armed = False

        reference_high = armed_high if (breakout_armed and np.isfinite(armed_high)) else rolling_range_high[i]
        reference_low = armed_low if (breakout_armed and np.isfinite(armed_low)) else rolling_range_low[i]
        buffer = atr_a[i] * breakout_buffer_atr if np.isfinite(atr_a[i]) else math.nan
        touch_buffer = atr_a[i] * boundary_touch_distance_atr if np.isfinite(atr_a[i]) else math.nan

        has_ref = breakout_armed and np.isfinite(reference_high) and np.isfinite(reference_low)
        upper_contact = bool(has_ref and np.isfinite(touch_buffer)
                             and high[i] >= reference_high - touch_buffer
                             and close[i] <= reference_high + buffer)
        lower_contact = bool(has_ref and np.isfinite(touch_buffer)
                             and low[i] <= reference_low + touch_buffer
                             and close[i] >= reference_low - buffer)

        new_armed_balance = established_balance and not prev_breakout_armed
        if new_armed_balance:
            upper_tests = 0
            lower_tests = 0
            last_upper_contact = -1
            last_lower_contact = -1
            upper_rejection = math.nan
            lower_rejection = math.nan
            avg_upper_rejection = math.nan
            avg_lower_rejection = math.nan
            pending_upper_bar = -1
            pending_lower_bar = -1

        upper_cooldown_done = last_upper_contact < 0 or (i - last_upper_contact) >= cooldown
        lower_cooldown_done = last_lower_contact < 0 or (i - last_lower_contact) >= cooldown
        new_upper_test = upper_contact and upper_cooldown_done
        new_lower_test = lower_contact and lower_cooldown_done

        frozen_mid = (reference_high + reference_low) * 0.5

        if upper_contact:
            last_upper_contact = i
        if lower_contact:
            last_lower_contact = i

        if pending_upper_bar >= 0:
            pending_upper_best = max(pending_upper_best, float(_clamp(
                _safe_div(pending_upper_boundary - close[i], pending_upper_boundary - pending_upper_mid, 0.0),
                0.0, 1.0)))
            if i - pending_upper_bar >= cooldown:
                upper_rejection = pending_upper_best
                avg_upper_rejection = (pending_upper_best if not np.isfinite(avg_upper_rejection)
                                       else avg_upper_rejection * 0.70 + pending_upper_best * 0.30)
                pending_upper_bar = -1

        if pending_lower_bar >= 0:
            pending_lower_best = max(pending_lower_best, float(_clamp(
                _safe_div(close[i] - pending_lower_boundary, pending_lower_mid - pending_lower_boundary, 0.0),
                0.0, 1.0)))
            if i - pending_lower_bar >= cooldown:
                lower_rejection = pending_lower_best
                avg_lower_rejection = (pending_lower_best if not np.isfinite(avg_lower_rejection)
                                       else avg_lower_rejection * 0.70 + pending_lower_best * 0.30)
                pending_lower_bar = -1

        if new_upper_test:
            upper_tests += 1
            pending_upper_bar = i
            pending_upper_boundary = reference_high
            pending_upper_mid = frozen_mid
            pending_upper_best = float(_clamp(
                _safe_div(reference_high - close[i], reference_high - frozen_mid, 0.0), 0.0, 1.0))

        if new_lower_test:
            lower_tests += 1
            pending_lower_bar = i
            pending_lower_boundary = reference_low
            pending_lower_mid = frozen_mid
            pending_lower_best = float(_clamp(
                _safe_div(close[i] - reference_low, frozen_mid - reference_low, 0.0), 0.0, 1.0))

        upper_repeated = upper_tests >= boundary_stressed_test_minimum
        lower_repeated = lower_tests >= boundary_stressed_test_minimum
        upper_weak = upper_repeated and np.isfinite(avg_upper_rejection) and avg_upper_rejection < weak_rejection_threshold
        lower_weak = lower_repeated and np.isfinite(avg_lower_rejection) and avg_lower_rejection < weak_rejection_threshold

        bullish_stress = upper_repeated and upper_tests > lower_tests and (
            upper_weak or (np.isfinite(acceptance_i) and acceptance_i >= acceptance_pressure))
        bearish_stress = lower_repeated and lower_tests > upper_tests and (
            lower_weak or (np.isfinite(acceptance_i) and acceptance_i <= -acceptance_pressure))

        upper_fatigue = float(_clamp(_safe_div(upper_tests - boundary_fresh_test_maximum, frozen_span, 0.0), 0.0, 1.0)) * \
            (0.5 if not np.isfinite(avg_upper_rejection) else 1.0 - avg_upper_rejection)
        lower_fatigue = float(_clamp(_safe_div(lower_tests - boundary_fresh_test_maximum, frozen_span, 0.0), 0.0, 1.0)) * \
            (0.5 if not np.isfinite(avg_lower_rejection) else 1.0 - avg_lower_rejection)
        dominant_fatigue = max(upper_fatigue, lower_fatigue)

        raw_bull_break = bool(ready_flag and breakout_armed and np.isfinite(buffer)
                              and close[i] > reference_high + buffer)
        raw_bear_break = bool(ready_flag and breakout_armed and np.isfinite(buffer)
                              and close[i] < reference_low - buffer)

        reentry = atr_a[i] * failed_breakout_reentry_atr if np.isfinite(atr_a[i]) else math.nan
        bull_sweep = bool(detect_failed_breakouts and ready_flag and breakout_armed
                          and np.isfinite(buffer) and np.isfinite(reentry)
                          and high[i] > reference_high + buffer
                          and close[i] <= reference_high - reentry)
        bear_sweep = bool(detect_failed_breakouts and ready_flag and breakout_armed
                          and np.isfinite(buffer) and np.isfinite(reentry)
                          and low[i] < reference_low - buffer
                          and close[i] >= reference_low + reentry)
        bull_failed = bull_sweep and bar_acceptance[i] < 0.0
        bear_failed = bear_sweep and bar_acceptance[i] > 0.0

        bull_streak = bull_streak + 1 if raw_bull_break else 0
        bear_streak = bear_streak + 1 if raw_bear_break else 0
        bull_held = bull_streak >= breakout_acceptance_bars
        bear_held = bear_streak >= breakout_acceptance_bars
        volume_ok = bool(volume_confirmed[i]) if np.ndim(volume_confirmed) else bool(volume_confirmed)

        bull_breakout = (raw_bull_break and bull_held and acceptance_i >= breakout_acceptance
                         and energy_i >= energy_high and volume_ok)
        bear_breakout = (raw_bear_break and bear_held and acceptance_i <= -breakout_acceptance
                         and energy_i >= energy_high and volume_ok)
        breakout_condition = bull_breakout or bear_breakout

        if breakout_condition:
            bars_since_breakout = 0
            breakout_armed = False
            compression_armed = False
        else:
            bars_since_breakout += 1

        bull_ready_pressure = bool(bullish_pressure[i]) and (
            not require_boundary_stress_for_ready or bullish_stress)
        bear_ready_pressure = bool(bearish_pressure[i]) and (
            not require_boundary_stress_for_ready or bearish_stress)
        ready_condition = bool(compression_armed and energy_building[i]
                               and (bull_ready_pressure or bear_ready_pressure))

        trend_now = bool(trend_condition[i])
        exhaustion_now = bool(exhaustion_condition[i])
        aftermath_now = bool(impulse_aftermath[i])

        if not ready_flag:
            candidate = STATE_NO_TRADE
        elif breakout_condition:
            candidate = STATE_BREAKOUT
        elif exhaustion_now:
            candidate = STATE_EXHAUSTION
        elif trend_now:
            candidate = STATE_TREND
        elif ready_condition:
            candidate = STATE_READY
        elif aftermath_now:
            candidate = STATE_AFTERMATH
        elif established_compression:
            candidate = STATE_COMPRESSION
        elif established_balance:
            candidate = STATE_BALANCE
        else:
            candidate = STATE_NO_TRADE

        any_raw = raw_bull_break or raw_bear_break
        hold_pending = (raw_bull_break and not bull_held) or (raw_bear_break and not bear_held)
        accept_weak = ((raw_bull_break and bull_held and acceptance_i < breakout_acceptance)
                       or (raw_bear_break and bear_held and acceptance_i > -breakout_acceptance))
        if candidate != STATE_NO_TRADE:
            reason = "NONE"
        elif not ready_flag:
            reason = "WARMUP"
        elif not framework:
            reason = "NO_FRAMEWORK"
        elif bull_failed:
            reason = "FAILED_BREAKOUT_UP"
        elif bear_failed:
            reason = "FAILED_BREAKOUT_DOWN"
        elif any_raw and hold_pending:
            reason = "BREAKOUT_HOLD_PENDING"
        elif any_raw and energy_i < energy_high:
            reason = "BREAKOUT_ENERGY_LOW"
        elif any_raw and accept_weak:
            reason = "BREAKOUT_ACCEPTANCE_WEAK"
        elif any_raw and not volume_ok:
            reason = "BREAKOUT_VOLUME_WEAK"
        elif in_range_framework and not balance_allowed:
            reason = "POST_BREAKOUT_LOCK"
        elif in_range_framework and range_width_atr[i] > maximum_range_atr:
            reason = "RANGE_TOO_WIDE"
        elif in_range_framework and compression_armed and not energy_building[i]:
            reason = "READY_ENERGY_FLAT"
        elif (in_range_framework and compression_armed and energy_building[i]
              and require_boundary_stress_for_ready and not bullish_stress and not bearish_stress):
            reason = "READY_BOUNDARY_STRESS_MISSING"
        elif in_range_framework and compression_armed and energy_building[i]:
            reason = "READY_NO_BOUNDARY_PRESSURE"
        elif in_range_framework and upper_weak and acceptance_i > 0.0:
            reason = "UPPER_BOUNDARY_FATIGUED"
        elif in_range_framework and lower_weak and acceptance_i < 0.0:
            reason = "LOWER_BOUNDARY_FATIGUED"
        elif in_range_framework and energy_i >= energy_high:
            reason = "RANGE_EXPANSION_UNCONFIRMED"
        elif in_range_framework and energy_i >= energy_low:
            reason = "RANGE_MID_ENERGY"
        elif in_range_framework and abs(acceptance_i) >= acceptance_neutral:
            reason = "RANGE_DIRECTIONAL_UNCONFIRMED"
        elif in_range_framework:
            reason = "RANGE_UNCLASSIFIED"
        elif energy_i < energy_high:
            reason = "TREND_ENERGY_LOW"
        elif abs(acceptance_i) < acceptance_strong:
            reason = "TREND_ACCEPTANCE_WEAK"
        else:
            reason = "TREND_DIRECTION_MISMATCH"

        # --- 状态机 ---------------------------------------------------------
        if candidate == market_state:
            pending_state = market_state
            pending_state_bars = 0
        elif candidate == pending_state:
            pending_state_bars += 1
        else:
            pending_state = candidate
            pending_state_bars = 1

        if bull_breakout:
            active_direction = 1
        elif bear_breakout:
            active_direction = -1
        elif market_state == STATE_TREND:
            if acceptance_i >= 20.0:
                active_direction = 1
            elif acceptance_i <= -20.0:
                active_direction = -1

        if breakout_condition:
            breakout_bars_remaining = breakout_hold_bars
        if breakout_bars_remaining > 0:
            breakout_bars_remaining -= 1

        force_breakout = pending_state == STATE_BREAKOUT
        hold_breakout = market_state == STATE_BREAKOUT and breakout_bars_remaining > 0 and not exhaustion_now

        if market_state == STATE_NO_TRADE:
            still_valid = candidate == STATE_NO_TRADE
        elif market_state == STATE_BALANCE:
            still_valid = established_balance
        elif market_state == STATE_COMPRESSION:
            still_valid = established_compression or ready_condition
        elif market_state == STATE_READY:
            still_valid = compression_armed and (ready_condition or energy_building[i])
        elif market_state == STATE_BREAKOUT:
            still_valid = breakout_bars_remaining > 0 or trend_now
        elif market_state == STATE_TREND:
            still_valid = trend_now or exhaustion_now
        elif market_state == STATE_AFTERMATH:
            still_valid = aftermath_now
        elif market_state == STATE_EXHAUSTION:
            still_valid = exhaustion_now
        else:
            still_valid = False

        invalid_state_bars = 0 if still_valid else invalid_state_bars + 1
        current_invalidated = invalid_state_bars >= confirmation_bars
        may_leave = (state_bars >= minimum_state_bars or force_breakout
                     or market_state == STATE_BREAKOUT or current_invalidated)
        required = 1 if pending_state == STATE_BREAKOUT else (
            confirmation_bars + 1 if pending_state == STATE_NO_TRADE else confirmation_bars)
        pending_confirmed = pending_state != market_state and pending_state_bars >= required

        previous_state = market_state
        if hold_breakout:
            market_state = STATE_BREAKOUT
            state_bars += 1
        elif pending_confirmed and may_leave:
            market_state = pending_state
            state_bars = 0
            pending_state_bars = 0
            invalid_state_bars = 0
        else:
            state_bars += 1

        state_changed = market_state != previous_state
        left_tradable = (state_changed and market_state == STATE_NO_TRADE
                         and previous_state in (STATE_READY, STATE_BREAKOUT, STATE_TREND))

        # --- 落盘 -----------------------------------------------------------
        ints["state"][i] = market_state
        ints["candidate_state"][i] = candidate
        ints["pending_state"][i] = pending_state
        ints["pending_state_bars"][i] = pending_state_bars
        ints["state_bars"][i] = state_bars
        ints["invalid_state_bars"][i] = invalid_state_bars
        ints["active_direction"][i] = active_direction
        ints["breakout_bars_remaining"][i] = breakout_bars_remaining
        ints["breakout_armed"][i] = 1 if breakout_armed else 0
        ints["compression_armed"][i] = 1 if compression_armed else 0
        ints["bars_since_balance"][i] = bars_since_balance
        ints["upper_tests"][i] = upper_tests
        ints["lower_tests"][i] = lower_tests
        ints["upper_fresh"][i] = 1 if upper_tests <= boundary_fresh_test_maximum else 0
        ints["lower_fresh"][i] = 1 if lower_tests <= boundary_fresh_test_maximum else 0
        ints["bullish_stress"][i] = 1 if bullish_stress else 0
        ints["bearish_stress"][i] = 1 if bearish_stress else 0
        ints["state_changed"][i] = 1 if state_changed else 0
        ints["left_tradable"][i] = 1 if left_tradable else 0
        ints["no_trade_reason"][i] = _REASON_CODE[reason]
        ints["data_ready"][i] = 1 if ready_flag else 0
        ints["has_framework"][i] = 1 if framework else 0
        ints["balance_allowed"][i] = 1 if balance_allowed else 0
        ints["established_balance"][i] = 1 if established_balance else 0
        ints["established_compression"][i] = 1 if established_compression else 0
        ints["energy_building"][i] = 1 if energy_building[i] else 0
        ints["bullish_pressure"][i] = 1 if bullish_pressure[i] else 0
        ints["bearish_pressure"][i] = 1 if bearish_pressure[i] else 0
        ints["ready_condition"][i] = 1 if ready_condition else 0
        ints["trend_condition"][i] = 1 if trend_now else 0
        ints["exhaustion_condition"][i] = 1 if exhaustion_now else 0
        ints["impulse_aftermath"][i] = 1 if aftermath_now else 0
        ints["impulse_direction"][i] = impulse_direction[i]
        ints["raw_bullish_breakout"][i] = 1 if raw_bull_break else 0
        ints["raw_bearish_breakout"][i] = 1 if raw_bear_break else 0
        ints["bullish_breakout"][i] = 1 if bull_breakout else 0
        ints["bearish_breakout"][i] = 1 if bear_breakout else 0
        ints["failed_breakout_bull"][i] = 1 if bull_failed else 0
        ints["failed_breakout_bear"][i] = 1 if bear_failed else 0
        ints["volume_confirmed"][i] = 1 if volume_ok else 0
        ints["range_framework"][i] = 1 if in_range_framework else 0
        ints["upper_contact"][i] = 1 if upper_contact else 0
        ints["lower_contact"][i] = 1 if lower_contact else 0
        ints["entered_ready"][i] = 1 if (state_changed and market_state == STATE_READY) else 0
        ints["entered_breakout"][i] = 1 if (state_changed and market_state == STATE_BREAKOUT) else 0
        ints["entered_trend"][i] = 1 if (state_changed and market_state == STATE_TREND) else 0
        ints["entered_no_trade"][i] = 1 if (state_changed and market_state == STATE_NO_TRADE) else 0

        floats["breakout_reference_high"][i] = reference_high
        floats["breakout_reference_low"][i] = reference_low
        floats["armed_range_high"][i] = armed_high
        floats["armed_range_low"][i] = armed_low
        floats["upper_rejection_quality"][i] = upper_rejection
        floats["lower_rejection_quality"][i] = lower_rejection
        floats["avg_upper_rejection_quality"][i] = avg_upper_rejection
        floats["avg_lower_rejection_quality"][i] = avg_lower_rejection
        floats["upper_fatigue"][i] = upper_fatigue
        floats["lower_fatigue"][i] = lower_fatigue
        floats["dominant_fatigue"][i] = dominant_fatigue

        prev_breakout_armed = breakout_armed

    data = {
        "mte_structure_score": structure_score,
        "mte_energy_score": energy_score,
        "mte_acceptance_score": acceptance_score,
        "mte_range_structure_quality": range_structure_quality,
        "mte_trend_structure_quality": trend_structure_quality,
        "mte_range_compactness": range_compactness,
        "mte_sustained_boundary_stability": sustained_stability,
        "mte_boundary_stability": boundary_stability,
        "mte_trend_cross_quality": trend_cross_quality,
        "mte_trend_efficiency_quality": trend_efficiency_quality,
        "mte_trend_adx_quality": trend_adx_quality,
        "mte_trend_slope_quality": trend_slope_quality,
        "mte_trend_consistency_quality": trend_consistency_quality,
        "mte_volatility_expansion": volatility_expansion,
        "mte_energy_efficiency": energy_efficiency,
        "mte_energy_momentum_accel": energy_momentum_accel,
        "mte_candle_commitment": candle_commitment,
        "mte_atr": atr_a,
        "mte_atr_ratio": atr_ratio,
        "mte_baseline": baseline,
        "mte_normalized_slope": normalized_slope,
        "mte_close_displacement": close_displacement,
        "mte_range_width_atr": range_width_atr,
        "mte_range_position": range_position,
        "mte_efficiency_ratio": efficiency_ratio,
        "mte_signed_consistency": signed_consistency,
        "mte_baseline_cross_rate": baseline_cross_rate,
        "mte_body_ratio": body_ratio,
        "mte_wick_ratio": wick_ratio,
        "mte_bar_acceptance": bar_acceptance,
        "mte_average_bar_acceptance": average_bar_acceptance,
        "mte_adx": adx,
        "mte_plus_di": plus_di,
        "mte_minus_di": minus_di,
        "mte_volume_ratio": volume_ratio,
        "mte_rolling_range_high": rolling_range_high,
        "mte_rolling_range_low": rolling_range_low,
        "mte_threshold_structure": np.full(n, structure_threshold),
        "mte_threshold_energy_low": np.full(n, energy_low),
        "mte_threshold_energy_high": np.full(n, energy_high),
        "mte_threshold_acceptance_neutral": np.full(n, acceptance_neutral),
        "mte_threshold_acceptance_strong": np.full(n, acceptance_strong),
        "mte_impulse_energy": impulse_energy,
    }
    for name, values in floats.items():
        data["mte_" + name] = values
    for name, values in ints.items():
        data["mte_" + name] = pd.Series(values, index=idx).astype("Int64")
    return pd.DataFrame(data, index=idx)
