"""Adaptive Cycle Detector — dominant cycle length from WaveTrend zero crossings.

Source: pine-collection/indicators/trend_strength/adaptive_cycle_detector/adaptive_cycle_detector.pine
(WavesUnchained, MIT)

Counting how often a WaveTrend oscillator crosses its zero line over a lookback gives a
half-cycle count, so `2 * lookback / crossings` estimates the dominant period. The estimate
is clamped, smoothed by an EMA and then converted into the adaptive lengths the script
suggests for WaveTrend, Stochastics and MFI. ADX above the trend threshold stretches the
suggested WaveTrend length by 1.25, contracting ATR shrinks it by 0.8.

`mintick` mirrors `syminfo.mintick` in the WaveTrend denominator; leave it at 0 for a pure
close-based reading (Pine then divides by `0.015 * |deviation|`).

Dropped: the cycle/mode dashboard, the min/max band and the colour ramp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (atr, crossover, crossunder, dmi, ema, flags, integer, nullable_int,
                    number, ohlcv, sma, sum_)


def adaptive_cycle_detector(frame: pd.DataFrame, *, cycle_lookback: int = 100,
                            period_smoothing: int = 5, min_period: int = 8, max_period: int = 60,
                            channel_length: int = 10, average_length: int = 21,
                            adx_length: int = 14, trend_threshold: float = 25.0,
                            mintick: float = 0.0) -> pd.DataFrame:
    integer("cycle_lookback", cycle_lookback, 20, 500)
    integer("period_smoothing", period_smoothing, 1, 20)
    integer("min_period", min_period, 4, 30)
    integer("max_period", max_period, 20, 200)
    integer("channel_length", channel_length, 1)
    integer("average_length", average_length, 1)
    integer("adx_length", adx_length, 5)
    number("trend_threshold", trend_threshold, 10.0, 50.0)
    number("mintick", mintick, 0.0, 1e6)
    if min_period >= max_period:
        raise ValueError("参数 min_period 必须小于 max_period")

    high, low, close, _ = ohlcv(frame, "high", "low", "close", "volume")
    typical = (high + low + close) / 3
    channel = ema(typical, channel_length)
    deviation = ema((typical - channel).abs(), channel_length)
    denominator = (0.015 * deviation.clip(lower=mintick if mintick > 0 else 0.0))
    wave = ema((typical - channel) / denominator.replace(0, np.nan), average_length)

    average_range = atr(frame, 14)
    average_of_range = sma(average_range, 20)
    expanding = flags(average_range > average_of_range * 1.05)
    contracting = flags(average_range < average_of_range * 0.95)
    _, _, adx_values = dmi(frame, adx_length)
    trending = flags(adx_values > trend_threshold)

    crossings = crossover(wave, pd.Series(0.0, index=frame.index)) | crossunder(
        wave, pd.Series(0.0, index=frame.index))
    crossing_count = sum_(crossings.astype(float), cycle_lookback)
    raw_period = pd.Series(float(max_period), index=frame.index, dtype=float)
    raw_period = raw_period.where(crossing_count <= 0,
                                  2.0 * cycle_lookback / crossing_count.replace(0, np.nan))
    clamped = raw_period.clip(lower=min_period, upper=max_period)
    cycle_period = ema(clamped, period_smoothing)

    base_length = (cycle_period / 4.0).fillna(0).astype(float)
    trend_factor = pd.Series(1.0, index=frame.index)
    trend_factor = trend_factor.mask(trending, 1.25).mask(~trending & contracting, 0.80)
    adapted = nullable_int((base_length * trend_factor).clip(lower=5, upper=50))

    mode = np.select([trending, expanding, contracting], ["Trend", "Expansion", "Compression"],
                     default="Normal")
    return pd.DataFrame({
        "wave_trend": wave,
        "atr": average_range,
        "atr_average": average_of_range,
        "atr_expanding": expanding,
        "atr_contracting": contracting,
        "adx": adx_values,
        "trending": trending,
        "crossing_up": crossover(wave, pd.Series(0.0, index=frame.index)),
        "crossing_down": crossunder(wave, pd.Series(0.0, index=frame.index)),
        "crossing_count": crossing_count,
        "raw_period": raw_period,
        "clamped_period": clamped,
        "cycle_period": cycle_period,
        "adapted_wavetrend_length": adapted,
        "suggested_stochastic_length": nullable_int((cycle_period / 3.0).clip(lower=8, upper=30)),
        "suggested_mfi_length": nullable_int((cycle_period / 4.0).clip(lower=7, upper=30)),
        "market_mode": pd.Series(mode, index=frame.index, dtype=object).where(cycle_period.notna(), "Normal"),
    }, index=frame.index)
