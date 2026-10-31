"""Market Pressure Scale — Setup/Impulse pressure pair with coil and reversal logic.

Source: pine-collection/indicators/momentum/market_pressure_scale/market_pressure_scale.pine
(WavesUnchained, MIT)

Setup Pressure blends ATR/BB-width compression (55%), proximity to the range
edge (30%) and quiet volume (15%); Impulse Pressure blends range expansion
(45%), body ratio (40%) and relative volume (15%). Without real turnover the
volume legs are dropped and the remaining weights rescale to sum to one
(0.55/0.85, 0.30/0.85, 0.45/0.85, 0.40/0.85). Both are smoothed and clamped to
0..100. Move strength = compression 45% + range-edge proximity 35% + ADX 20%,
suppressed to 35% inside the chop filter (ADX < 16, |DI-| small, mid-range).
Coil state machine arms in the setup zone when setup pressure crosses above 65
with a quiet impulse, ignites once impulse reaches 45, and fails if setup
crosses back under 52 before ignition. Reversal scores weight impulse
exhaustion (45%), rejection wick (30%) and setup pressure (25%) at range
extremes.

Dropped: dashboard table, columns colouring, signal labels/tooltips, Pine Logs
debug output and the alert plumbing. `syminfo.mintick` is approximated by a
1e-10 epsilon; `syminfo.volumetype != "tick"` cannot be observed from data and
is treated as real whenever any positive volume exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (atr, barssince, boolean, choice, crossover, crossunder, dmi, ema,
                    flags, highest, hma, integer, lowest, number, ohlcv, rma, sma, stdev,
                    wma)

_MINTICK = 1e-10
_SMOOTH_MODES = ("EMA", "RMA", "SMA", "WMA", "HMA", "JMA")


def _jma(src: pd.Series, length: int, phase: int, power: float) -> pd.Series:
    values = src.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    phase_ratio = 0.5 if phase < -100 else 2.5 if phase > 100 else phase / 100.0 + 1.5
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2.0)
    alpha = beta ** power
    e0 = e1 = e2 = jma = np.nan
    for position, value in enumerate(values):
        if np.isnan(value):
            continue
        e0 = (1.0 - alpha) * value + alpha * (value if np.isnan(e0) else e0)
        e1 = (value - e0) * (1.0 - beta) + beta * (0.0 if np.isnan(e1) else e1)
        e2 = ((e0 + phase_ratio * e1 - (value if np.isnan(jma) else jma))
              * (1.0 - alpha) ** 2 + alpha ** 2 * (0.0 if np.isnan(e2) else e2))
        jma = (value if np.isnan(jma) else jma) + e2
        out[position] = jma
    return pd.Series(out, index=src.index)


def _smooth(src: pd.Series, length: int, mode: str, phase: int, power: float) -> pd.Series:
    if mode == "SMA":
        return sma(src, length)
    if mode == "RMA":
        return rma(src, length)
    if mode == "WMA":
        return wma(src, length)
    if mode == "HMA":
        return hma(src, length)
    if mode == "JMA":
        return _jma(src, length, phase, power)
    return ema(src, length)


def _norm100(src: pd.Series, length: int) -> pd.Series:
    window_high = highest(src, length)
    window_low = lowest(src, length)
    spread = window_high - window_low
    return pd.Series(np.where(spread == 0, 50.0, 100.0 * (src - window_low) / spread),
                     index=src.index)


def _clamp(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=100.0)


def market_pressure_scale(
        frame: pd.DataFrame, *, norm_len: int = 100, atr_len: int = 14,
        bb_len: int = 20, bb_mult: float = 2.0, vol_len: int = 20, sr_len: int = 50,
        bias_type: str = "HMA", bias_fast_len: int = 9, bias_slow_len: int = 21,
        smooth_len: int = 3, smooth_type: str = "EMA", jma_phase: int = 50,
        jma_power: float = 2.0, dmi_len: int = 14, adx_smooth: int = 14,
        chop_adx: float = 16.0, chop_di_spread: float = 5.0,
        coil_arm_level: float = 65.0, coil_quiet_impulse: float = 40.0,
        coil_ignite_impulse: float = 45.0, coil_fail_level: float = 52.0,
        naked_impulse_level: float = 70.0, naked_setup_level: float = 40.0,
        naked_adx_max: float = 25.0, exhaustion_window: int = 10,
        reversal_score_level: float = 60.0) -> pd.DataFrame:
    integer("norm_len", norm_len, 20)
    integer("atr_len", atr_len, 1)
    integer("bb_len", bb_len, 2)
    number("bb_mult", bb_mult, 0.1)
    integer("vol_len", vol_len, 1)
    integer("sr_len", sr_len, 2)
    choice("bias_type", bias_type, _SMOOTH_MODES)
    integer("bias_fast_len", bias_fast_len, 1)
    integer("bias_slow_len", bias_slow_len, 1)
    integer("smooth_len", smooth_len, 1, 50)
    choice("smooth_type", smooth_type, _SMOOTH_MODES)
    integer("jma_phase", jma_phase, -100, 100)
    number("jma_power", jma_power, 1.0, 4.0)
    integer("dmi_len", dmi_len, 1)
    integer("adx_smooth", adx_smooth, 1)
    number("chop_adx", chop_adx, 0.0, 100.0)
    number("coil_arm_level", coil_arm_level, 0.0, 100.0)
    number("coil_fail_level", coil_fail_level, 0.0, 100.0)
    integer("exhaustion_window", exhaustion_window, 1)

    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    open_ = frame["open"].astype(float)

    average_true_range = atr(frame, atr_len)
    atr_percent = average_true_range / close

    basis = sma(close, bb_len)
    deviation = bb_mult * stdev(close, bb_len)
    bb_width = (2.0 * deviation) / basis

    volume_is_real = bool(volume.fillna(0.0).gt(0).any())
    volume_sma = sma(volume, vol_len)
    vol_rel = pd.Series(np.where(~volume_is_real | volume_sma.isna() | volume_sma.le(0),
                                 1.0, volume / volume_sma), index=frame.index)
    bar_range = high - low
    body_size = (close - open_).abs()
    body_ratio = body_size / bar_range.clip(lower=_MINTICK)
    lower_wick = np.minimum(close, open_) - low
    upper_wick = high - np.maximum(close, open_)

    bias_fast = _smooth(close, bias_fast_len, bias_type, jma_phase, jma_power)
    bias_slow = _smooth(close, bias_slow_len, bias_type, jma_phase, jma_power)
    bull_bias = flags(bias_fast > bias_slow)
    bear_bias = flags(bias_fast < bias_slow)

    range_high = highest(high, sr_len)
    range_low = lowest(low, sr_len)
    range_size = range_high - range_low
    near_high = (range_high - close).abs() / average_true_range.clip(lower=_MINTICK)
    near_low = (close - range_low).abs() / average_true_range.clip(lower=_MINTICK)
    sr_proximity = _clamp(100.0 - near_low.combine(near_high, np.minimum) * 50.0)
    range_position = pd.Series(np.where(range_size == 0, 50.0,
                                        100.0 * (close - range_low) / range_size),
                               index=frame.index)

    di_plus, di_minus, adx = dmi(frame, dmi_len)
    if adx_smooth != dmi_len:
        adx = rma(100 * (di_plus - di_minus).abs()
                  / (di_plus + di_minus).replace(0, np.nan), adx_smooth)
    chop_market = flags((adx < chop_adx) & ((di_plus - di_minus).abs() < chop_di_spread)
                        & (range_position > 20) & (range_position < 80))
    valid_setup_zone = flags((range_position < 20) | (range_position > 80))

    atr_compression = 100.0 - _norm100(atr_percent, norm_len)
    bb_compression = 100.0 - _norm100(bb_width, norm_len)
    compression = (atr_compression + bb_compression) / 2.0
    vol_quiet = 100.0 - _norm100(vol_rel, norm_len)

    if volume_is_real:
        setup = compression * 0.55 + sr_proximity * 0.30 + vol_quiet * 0.15
    else:
        setup = compression * (0.55 / 0.85) + sr_proximity * (0.30 / 0.85)
    setup_pressure = _clamp(setup)

    range_expansion = _norm100(bar_range / average_true_range.clip(lower=_MINTICK), norm_len)
    volume_pressure = _norm100(vol_rel, norm_len)
    body_pressure = _norm100(body_ratio, norm_len)
    if volume_is_real:
        impulse = range_expansion * 0.45 + body_pressure * 0.40 + volume_pressure * 0.15
    else:
        impulse = range_expansion * (0.45 / 0.85) + body_pressure * (0.40 / 0.85)
    impulse_pressure = _clamp(impulse)

    setup_pressure = _smooth(setup_pressure, smooth_len, smooth_type, jma_phase, jma_power)
    impulse_pressure = _smooth(impulse_pressure, smooth_len, smooth_type, jma_phase,
                               jma_power)

    adx_norm = _clamp(adx / 50.0 * 100.0)
    strength = compression * 0.45 + sr_proximity * 0.35 + adx_norm * 0.20
    move_strength = _clamp(pd.Series(np.where(chop_market, strength * 0.35, strength),
                                     index=frame.index))

    # Coil state machine (Pine `var` persistence).
    coil_entry = flags(valid_setup_zone & ~chop_market
                       & crossover(setup_pressure, coil_arm_level)
                       & (impulse_pressure < coil_quiet_impulse))
    in_coil = np.zeros(len(frame), dtype=bool)
    coil_ignited = np.zeros(len(frame), dtype=bool)
    armed = False
    ignited = False
    setup_values = setup_pressure.to_numpy(dtype=float)
    impulse_values = impulse_pressure.to_numpy(dtype=float)
    coil_entry_values = coil_entry.to_numpy(dtype=bool)
    setup_under_fail = crossunder(setup_pressure, coil_fail_level).to_numpy(dtype=bool)
    range_back = ((range_position > 25) & (range_position < 75)).to_numpy(dtype=bool)
    sig_failed_values = np.zeros(len(frame), dtype=bool)
    for position in range(len(frame)):
        if coil_entry_values[position]:
            armed, ignited = True, False
        if armed and not np.isnan(impulse_values[position]) \
                and impulse_values[position] >= coil_ignite_impulse:
            ignited = True
        if armed and not ignited and setup_under_fail[position] and range_back[position]:
            sig_failed_values[position] = True
        if sig_failed_values[position] \
                or (not np.isnan(setup_values[position]) and setup_values[position] < 40.0):
            armed = False
        in_coil[position] = armed
        coil_ignited[position] = ignited
    sig_failed = flags(pd.Series(sig_failed_values, index=frame.index))

    naked_signal = flags((impulse_pressure > naked_impulse_level)
                         & crossunder(setup_pressure, naked_setup_level)
                         & (adx < naked_adx_max))

    impulse_high = highest(impulse_pressure, exhaustion_window)
    exhaustion_score = _clamp(pd.Series(
        np.where(impulse_high > 50.0, impulse_high - impulse_pressure, 0.0),
        index=frame.index))
    lower_wick_ratio = lower_wick / bar_range.clip(lower=_MINTICK) * 100.0
    upper_wick_ratio = upper_wick / bar_range.clip(lower=_MINTICK) * 100.0
    reversal_active = flags(valid_setup_zone & ~chop_market)
    reversal_long_score = (exhaustion_score * 0.45 + lower_wick_ratio * 0.30
                           + setup_pressure * 0.25)
    reversal_short_score = (exhaustion_score * 0.45 + upper_wick_ratio * 0.30
                            + setup_pressure * 0.25)
    bull_reject = flags((close > open_) & (close > low + bar_range * 0.6))
    bear_reject = flags((close < open_) & (close < high - bar_range * 0.6))
    reversal_long = flags(reversal_active & (range_position < 20) & bull_reject
                          & crossover(reversal_long_score, reversal_score_level))
    reversal_short = flags(reversal_active & (range_position > 80) & bear_reject
                           & crossover(reversal_short_score, reversal_score_level))
    recent_reversal_long = flags(barssince(reversal_long) <= 2)
    recent_reversal_short = flags(barssince(reversal_short) <= 2)

    phase = pd.Series(
        np.where(chop_market, "Sideways / Chop",
                 np.where(impulse_pressure > 70.0, "Impulse Running",
                          np.where((setup_pressure < 40.0) & (impulse_pressure < 40.0),
                                   "Sideways / Quiet", "Neutral"))),
        index=frame.index, dtype=object)
    bias_text = pd.Series(np.where(bull_bias, "Bullish",
                                   np.where(bear_bias, "Bearish", "Flat")),
                          index=frame.index, dtype=object)

    return pd.DataFrame({
        "setup_pressure": setup_pressure,
        "impulse_pressure": impulse_pressure,
        "move_strength": move_strength,
        "compression": compression,
        "sr_proximity": sr_proximity,
        "range_position": range_position,
        "chop_market": chop_market,
        "valid_setup_zone": valid_setup_zone,
        "bull_bias": bull_bias,
        "bear_bias": bear_bias,
        "bias_text": bias_text,
        "in_coil": flags(pd.Series(in_coil, index=frame.index)),
        "coil_ignited": flags(pd.Series(coil_ignited, index=frame.index)),
        "coil_entry": coil_entry,
        "failed_coil": sig_failed,
        "naked_impulse": naked_signal,
        "exhaustion_score": exhaustion_score,
        "reversal_long_score": reversal_long_score,
        "reversal_short_score": reversal_short_score,
        "reversal_long": reversal_long,
        "reversal_short": reversal_short,
        "recent_reversal_long": recent_reversal_long,
        "recent_reversal_short": recent_reversal_short,
        "phase": phase,
    }, index=frame.index)
