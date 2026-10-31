"""Smooth Trend Radar — WMA/EMA-smoothed Supertrend baseline with rejections.

Source: pine-collection/indicators/trend_direction/smooth_trend_radar/smooth_trend_radar.pine
(WavesUnchained, MIT)

The baseline is the EMA-of-WMA of the Supertrend midline (hl2 ± factor×ATR
with ratcheting bands). Trend direction follows the baseline slope over
`trend_slope_lookback` bars, but only commits when the move clears
`trend_slope_threshold` × Supertrend ATR (hysteresis). Flip signals fire when
the direction changes while ADX confirms a trend. Rejection signals need the
low/high to pierce the baseline and close back on the trend side, at least
`cont_factor` baseline touches inside the pivot window, a baseline sloping in
the signal direction over `slope_lookback` bars, optional polarity
(Body/Pressure) and volume-above-average filters. Strength ranks the slope
magnitude against its own history; overextension ranks the close-to-baseline
distance.

Dropped: TP/SL level tracking (entry/stop/target persistence belongs to the
broker), daily pivot targets, plots, labels and colours.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (atr, boolean, choice, flags, highest, integer, lowest, number, ohlcv,
                    percentrank, dmi, ema, wma, sma)


def smooth_trend_radar(
        frame: pd.DataFrame, *, st_factor: float = 3.0, st_atr_period: int = 90,
        wma_length: int = 40, ema_length: int = 14, adx_threshold: float = 25.0,
        trend_slope_lookback: int = 3, trend_slope_threshold: float = 0.05,
        vol_adaptive: bool = False, strength_lookback: int = 100,
        strength_threshold: float = 70.0, cont_factor: int = 2,
        slope_lookback: int = 5, polarity_method: str = "Off",
        volume_filter: str = "Off", volume_ma_len: int = 20,
        overext_lookback: int = 100, overext_threshold: float = 85.0,
        atr_period: int = 14, sl_multiplier: float = 2.0) -> pd.DataFrame:
    number("st_factor", st_factor, 0.1)
    integer("st_atr_period", st_atr_period, 1)
    integer("wma_length", wma_length, 1)
    integer("ema_length", ema_length, 1)
    number("adx_threshold", adx_threshold, 5.0, 50.0)
    integer("trend_slope_lookback", trend_slope_lookback, 1)
    number("trend_slope_threshold", trend_slope_threshold, 0.0)
    boolean("vol_adaptive", vol_adaptive)
    integer("strength_lookback", strength_lookback, 20)
    number("strength_threshold", strength_threshold, 50.0, 99.0)
    integer("cont_factor", cont_factor, 1, 5)
    integer("slope_lookback", slope_lookback, 1)
    choice("polarity_method", polarity_method, ("Off", "Body", "Pressure"))
    choice("volume_filter", volume_filter, ("Off", "Above Average"))
    integer("volume_ma_len", volume_ma_len, 1)
    integer("overext_lookback", overext_lookback, 20)
    number("overext_threshold", overext_threshold, 50.0, 99.0)
    integer("atr_period", atr_period, 1)
    number("sl_multiplier", sl_multiplier, 0.1)
    high, low, close = ohlcv(frame, "high", "low", "close")
    volume = frame["volume"].astype(float) if "volume" in frame.columns \
        else pd.Series(np.nan, index=frame.index)
    open_ = frame["open"].astype(float)
    hl2 = (high + low) / 2

    supertrend_atr = atr(frame, st_atr_period)
    vol_pct_rank = percentrank(supertrend_atr, 200)
    factor = st_factor * (0.85 + 0.3 * vol_pct_rank / 100.0) if vol_adaptive \
        else pd.Series(st_factor, index=frame.index)

    upper_raw = hl2 + factor * supertrend_atr
    lower_raw = hl2 - factor * supertrend_atr
    upper = lower = np.nan
    rows = len(frame)
    upper_arr = np.full(rows, np.nan)
    lower_arr = np.full(rows, np.nan)
    closes = close.to_numpy(dtype=float)
    for position in range(rows):
        u = upper_raw.iloc[position]
        d = lower_raw.iloc[position]
        if np.isnan(u):
            continue
        prev_u = upper if not np.isnan(upper) else u
        prev_l = lower if not np.isnan(lower) else d
        lower = d if (d > prev_l or closes[position - 1] < prev_l) else prev_l \
            if position > 0 else d
        upper = u if (u < prev_u or closes[position - 1] > prev_u) else prev_u \
            if position > 0 else u
        upper_arr[position] = upper
        lower_arr[position] = lower
    midline = (pd.Series(lower_arr, index=frame.index)
               + pd.Series(upper_arr, index=frame.index)) / 2
    baseline = ema(wma(midline, wma_length), ema_length)

    st_atr_values = supertrend_atr.to_numpy(dtype=float)
    baseline_values = baseline.to_numpy(dtype=float)
    trend_slope = baseline - baseline.shift(trend_slope_lookback)

    trend = np.zeros(rows, dtype=int)
    current = 0
    slope_values = trend_slope.to_numpy(dtype=float)
    for position in range(rows):
        slope = slope_values[position]
        if np.isnan(slope) or np.isnan(st_atr_values[position]):
            trend[position] = current
            continue
        if slope > st_atr_values[position] * trend_slope_threshold:
            current = 1
        elif slope < -st_atr_values[position] * trend_slope_threshold:
            current = -1
        trend[position] = current
    trend_series = pd.Series(trend, index=frame.index)

    _, _, adx = dmi(frame, 14)
    is_trending = flags(adx > adx_threshold)
    long_signal = flags((trend_series == 1) & (trend_series.shift(1) != 1) & is_trending)
    short_signal = flags((trend_series == -1) & (trend_series.shift(1) != -1) & is_trending)

    slope_magnitude = (trend_slope.abs() / supertrend_atr.replace(0, np.nan)).fillna(0.0)
    strength_rank = percentrank(slope_magnitude, strength_lookback)
    strong_trend = flags(strength_rank > strength_threshold)

    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    opens = open_.to_numpy(dtype=float)
    bull_touches = np.zeros(rows)
    bear_touches = np.zeros(rows)
    for position in range(rows):
        count_bull = 0
        count_bear = 0
        for back in range(cont_factor):
            bar = position - back
            if bar < 0:
                break
            if trend[bar] == 1 and lows[bar] <= baseline_values[bar]:
                count_bull += 1
            if trend[bar] == -1 and highs[bar] >= baseline_values[bar]:
                count_bear += 1
        bull_touches[position] = count_bull
        bear_touches[position] = count_bear
    bull_touches_series = pd.Series(bull_touches, index=frame.index)
    bear_touches_series = pd.Series(bear_touches, index=frame.index)

    pol_bull = pd.Series(True, index=frame.index)
    pol_bear = pd.Series(True, index=frame.index)
    if polarity_method == "Body":
        pol_bull = flags(pd.Series(closes > opens, index=frame.index))
        pol_bear = flags(pd.Series(closes < opens, index=frame.index))
    elif polarity_method == "Pressure":
        pol_bull = flags(pd.Series((closes - lows) > (highs - closes), index=frame.index))
        pol_bear = flags(pd.Series((closes - lows) < (highs - closes), index=frame.index))

    volume_is_real = bool(volume.fillna(0.0).gt(0).any())
    if volume_filter == "Above Average" and volume_is_real:
        vol_ok = flags(volume > sma(volume, volume_ma_len)) if isinstance(volume, pd.Series) \
            else flags(pd.Series(True, index=frame.index))
    else:
        vol_ok = flags(pd.Series(True, index=frame.index))

    baseline_series = baseline
    baseline_rising = flags(baseline_series > baseline_series.shift(slope_lookback))
    baseline_falling = flags(baseline_series < baseline_series.shift(slope_lookback))
    rejection_bull = flags((low <= baseline_series) & (close > baseline_series)
                           & (trend_series == 1) & baseline_rising
                           & (bull_touches_series >= cont_factor) & pol_bull & vol_ok)
    rejection_bear = flags((high >= baseline_series) & (close < baseline_series)
                           & (trend_series == -1) & baseline_falling
                           & (bear_touches_series >= cont_factor) & pol_bear & vol_ok)

    distance = (close - baseline_series).abs()
    overext_rank = percentrank(distance, overext_lookback)
    overextended = flags(overext_rank > overext_threshold)
    overext_bull = flags(overextended & (close < baseline_series))
    overext_bear = flags(overextended & (close > baseline_series))

    trade_atr = atr(frame, atr_period)
    rejection_dir = pd.Series(0, index=frame.index)
    rejection_dir = rejection_dir.mask(rejection_bull, 1)
    rejection_dir = rejection_dir.mask(rejection_bear, -1)
    stop_long = low - trade_atr * sl_multiplier
    stop_short = high + trade_atr * sl_multiplier

    return pd.DataFrame({
        "baseline": baseline,
        "trend": trend_series.astype("Int64"),
        "trend_slope": trend_slope,
        "long_signal": long_signal,
        "short_signal": short_signal,
        "strong_trend": strong_trend,
        "strength_rank": strength_rank,
        "is_trending": is_trending,
        "bull_touches": bull_touches_series.astype("Int64"),
        "bear_touches": bear_touches_series.astype("Int64"),
        "rejection_bull": rejection_bull,
        "rejection_bear": rejection_bear,
        "rejection_regime": pd.Series(
            np.where(is_trending, "Trend Pullback", "Range Rejection"),
            index=frame.index, dtype=object),
        "overextended": overextended,
        "overext_bull": overext_bull,
        "overext_bear": overext_bear,
        "supertrend_atr": supertrend_atr,
        "atr": trade_atr,
        "stop_long": stop_long,
        "stop_short": stop_short,
        "rejection_dir": rejection_dir.astype("Int64"),
    }, index=frame.index)
