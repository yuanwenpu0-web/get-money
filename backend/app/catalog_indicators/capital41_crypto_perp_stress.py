"""Capital41 加密永续压力：把基差、持仓量变化与插针回收加权成 0..100 的压力分。

Source: pine-crypto-collection/capital41/Capital41_Crypto_Perp_Stress/Capital41_Crypto_Perp_Stress.pine

MIT License — Copyright (c) 2026 Capital41

The score sums four weighted legs: the perpetual/spot basis against the
`basis_positive_pct` threshold (35 points), the smoothed open-interest change against
`oi_impulse_pct` (30 points), a volume-backed wick rejection (25 points) and a crowded
trend continuation (10 points). Squeeze flags combine a crowded basis, an OI build and a
rejection candle; a cascade additionally needs a break of the previous bar's extreme.

The script reads three instruments with `request.security(..., lookahead_off)` on the
chart's own timeframe, which is just an aligned column here, so the caller supplies
`spot_close` and `perp_close` as **required** columns and, optionally, `open_interest`.
Upstream keeps the OI symbol as an input and substitutes volume bar by bar whenever the
OI reading is missing; an absent `open_interest` column does the same, and
`use_volume_fallback` decides whether that substitute still counts as valid open interest.
No feed is fetched. `volume` is required for the spike test and for the substitution.

`barstate.isconfirmed` guards are dropped (every historical bar is closed). Weekend
scaling follows `dayofweek` in the timestamps' own timezone, as Pine does for the
exchange session.

Dropped: the signal markers, the stress background, the status badge, the optional trend
EMA plot (`showEMA`) and the five alert conditions.
"""
from __future__ import annotations

import pandas as pd

from ._pine import atr, boolean, clamp, flags, integer, number, ohlcv, sma
from ._pine import ema as pine_ema


def _scaled(value: pd.Series, span: pd.Series, weight: float) -> pd.Series:
    """`f_clamp(value / span * weight, 0, weight)` for one stress component."""
    return clamp(value / span * weight, 0.0, weight)


def _factor(condition: pd.Series, value: float) -> pd.Series:
    return pd.Series(1.0, index=condition.index).mask(condition, value)


def capital41_crypto_perp_stress(frame: pd.DataFrame, *, basis_positive_pct: float = 0.25,
                                 basis_negative_pct: float = -0.25,
                                 basis_neutral_pct: float = 0.10,
                                 oi_impulse_pct: float = 1.00, oi_flush_pct: float = 1.20,
                                 oi_smoothing: int = 3, atr_length: int = 14,
                                 atr_average_length: int = 20,
                                 wick_atr_multiplier: float = 0.8,
                                 wick_recovery_ratio: float = 0.55,
                                 volume_length: int = 50, volume_multiplier: float = 1.2,
                                 compression_atr_multiplier: float = 0.9,
                                 use_weekend_sensitivity: bool = True,
                                 weekend_basis_factor: float = 0.85,
                                 weekend_oi_factor: float = 0.85,
                                 weekend_wick_factor: float = 0.90,
                                 use_volume_fallback: bool = True,
                                 trend_ema_length: int = 50) -> pd.DataFrame:
    number("basis_positive_pct", basis_positive_pct, 0.01, 5.0)
    number("basis_negative_pct", basis_negative_pct, -5.0, -0.01)
    number("basis_neutral_pct", basis_neutral_pct, 0.01, 3.0)
    number("oi_impulse_pct", oi_impulse_pct, 0.10, 20.0)
    number("oi_flush_pct", oi_flush_pct, 0.10, 20.0)
    integer("oi_smoothing", oi_smoothing, 1, 30)
    integer("atr_length", atr_length, 1)
    integer("atr_average_length", atr_average_length, 5)
    number("wick_atr_multiplier", wick_atr_multiplier, 0.1, 5.0)
    number("wick_recovery_ratio", wick_recovery_ratio, 0.10, 0.95)
    integer("volume_length", volume_length, 5)
    number("volume_multiplier", volume_multiplier, 0.1, 5.0)
    number("compression_atr_multiplier", compression_atr_multiplier, 0.1, 2.0)
    boolean("use_weekend_sensitivity", use_weekend_sensitivity)
    number("weekend_basis_factor", weekend_basis_factor, 0.5, 1.5)
    number("weekend_oi_factor", weekend_oi_factor, 0.5, 1.5)
    number("weekend_wick_factor", weekend_wick_factor, 0.5, 1.5)
    boolean("use_volume_fallback", use_volume_fallback)
    integer("trend_ema_length", trend_ema_length, 5)

    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    opening, spot, perp = ohlcv(frame, "open", "spot_close", "perp_close")
    if "open_interest" in frame.columns:
        (oi_column,) = ohlcv(frame, "open_interest")
        oi_known = oi_column.notna()
        oi_series = oi_column.where(oi_known, volume)
    else:
        oi_known = pd.Series(False, index=frame.index)
        oi_series = volume

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("索引必须为 K 线开盘时间")
    weekend = pd.Series(frame.index.day_of_week >= 5, index=frame.index)

    basis_pct = ((perp - spot) / spot.where(spot > 0) * 100.0).where(spot > 0, 0.0)
    oi_previous = oi_series.shift(1)
    oi_change_raw = ((oi_series - oi_previous) / oi_previous.where(oi_previous > 0)
                     * 100.0).where(oi_previous > 0, 0.0)
    oi_change = pine_ema(oi_change_raw, oi_smoothing)
    oi_valid = flags(oi_known) | use_volume_fallback

    adjusted = use_weekend_sensitivity & weekend
    basis_threshold_up = basis_positive_pct * _factor(adjusted, weekend_basis_factor)
    basis_threshold_down = basis_negative_pct * _factor(adjusted, weekend_basis_factor)
    basis_threshold_neutral = basis_neutral_pct * _factor(adjusted, weekend_basis_factor)
    oi_threshold = oi_impulse_pct * _factor(adjusted, weekend_oi_factor)
    oi_flush_threshold = oi_flush_pct * _factor(adjusted, weekend_oi_factor)
    average_true_range = atr(frame, atr_length)
    wick_threshold = average_true_range * (wick_atr_multiplier * _factor(
        adjusted, weekend_wick_factor))

    average_atr = sma(average_true_range, atr_average_length)
    trend_ema = pine_ema(close, trend_ema_length)
    trend_up = flags(close > trend_ema)
    trend_down = flags(close < trend_ema)

    volume_average = sma(volume, volume_length)
    volume_spike = flags(volume_average > 0) & flags(volume > volume_average * volume_multiplier)

    bar_range = high - low
    upper = pd.concat([opening, close], axis=1).max(axis=1)
    lower = pd.concat([opening, close], axis=1).min(axis=1)
    up_wick = high - upper
    down_wick = lower - low
    recover_up = flags(bar_range > 0) & flags(close <= high - bar_range * wick_recovery_ratio)
    recover_down = flags(bar_range > 0) & flags(close >= low + bar_range * wick_recovery_ratio)

    liquidation_up = flags(up_wick > wick_threshold) & volume_spike & recover_up
    liquidation_down = flags(down_wick > wick_threshold) & volume_spike & recover_down

    crowded_long = flags(basis_pct >= basis_threshold_up)
    crowded_short = flags(basis_pct <= basis_threshold_down)
    oi_build = flags(oi_change >= oi_threshold) & oi_valid
    oi_flush = flags(oi_change <= -oi_flush_threshold) & oi_valid

    long_squeeze_risk = crowded_long & oi_build & liquidation_down
    short_squeeze_risk = crowded_short & oi_build & liquidation_up
    cascade_down = long_squeeze_risk & flags(close < low.shift(1))
    cascade_up = short_squeeze_risk & flags(close > high.shift(1))

    basis_neutral = flags(basis_pct.abs() <= basis_threshold_neutral)
    volatility_compression = flags(average_atr > 0) & flags(
        average_true_range < average_atr * compression_atr_multiplier)
    reset_complete = basis_neutral & oi_flush & volatility_compression

    crowded_trend_up = trend_up & crowded_long & oi_build & ~liquidation_down
    crowded_trend_down = trend_down & crowded_short & oi_build & ~liquidation_up

    basis_stress = _scaled(basis_pct.abs(), basis_threshold_up.abs(), 35.0)
    oi_stress = _scaled(oi_change.clip(lower=0.0), oi_threshold, 30.0)
    wick_stress = pd.Series(0.0, index=frame.index).mask(
        liquidation_up | liquidation_down, 25.0)
    trend_stress = pd.Series(0.0, index=frame.index).mask(
        crowded_trend_up | crowded_trend_down, 10.0)
    stress = clamp(basis_stress + oi_stress + wick_stress + trend_stress, 0.0, 100.0)

    return pd.DataFrame({
        "basis_pct": basis_pct,
        "oi_change_pct": oi_change,
        "oi_valid": oi_valid,
        "is_weekend": weekend,
        "trend_ema": trend_ema,
        "trend_up": trend_up,
        "trend_down": trend_down,
        "atr": average_true_range,
        "atr_average": average_atr,
        "volume_spike": volume_spike,
        "liquidation_up": liquidation_up,
        "liquidation_down": liquidation_down,
        "crowded_long": crowded_long,
        "crowded_short": crowded_short,
        "oi_build": oi_build,
        "oi_flush": oi_flush,
        "long_squeeze_risk": long_squeeze_risk,
        "short_squeeze_risk": short_squeeze_risk,
        "cascade_down": cascade_down,
        "cascade_up": cascade_up,
        "basis_neutral": basis_neutral,
        "volatility_compression": volatility_compression,
        "reset_complete": reset_complete,
        "crowded_trend_up": crowded_trend_up,
        "crowded_trend_down": crowded_trend_down,
        "basis_stress": basis_stress,
        "oi_stress": oi_stress,
        "wick_stress": wick_stress,
        "trend_stress": trend_stress,
        "stress_score": stress,
    }, index=frame.index)
