"""Elder Ray Pressure Engine — bull/bear pressure vs a consensus MA, normalized.

Source: pine-collection/indicators/momentum/elder_ray_pressure_engine/elder_ray_pressure_engine.pine
(WavesUnchained, MIT)

Four calculation models express how far price stretches away from the consensus
average: classic Elder (high/low minus basis), directional Elder (clamped to one
side), true-range Elder (including the previous close in the extreme), and
close-weighted Elder (clamped and weighted by the bar's close location). The raw
pressure is normalized (Raw / ATR / Average Range / Percent / RMS), smoothed,
then condensed into net pressure, dominance, impulse and acceleration. Exhaustion
flags combine a rolling percentile rank of the pressure with a falling/recovering
pressure and price still pushing to new extremes; the divergence block compares
pressure at consecutive price pivots.

Dropped: the dashboard table, all colouring, background fills, plot style
switches and the `barstate.isconfirmed` real-time gate (historical bars are
always confirmed, so the flag is a no-op here).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (atr, barssince, boolean, choice, crossunder, crossover, ema, flags,
                    highest, hma, integer, lowest, number, ohlcv, pivothigh, pivotlow,
                    percentrank, rma, sma, valuewhen, vwma, wma)

_ENGINES = ("Classic Elder", "Directional Elder", "True Range Elder",
            "Close-Weighted Elder")
_MA_TYPES = ("EMA", "RMA", "SMA", "WMA", "HMA", "VWMA")
_SMOOTH_TYPES = ("EMA", "RMA", "SMA")
_NORMALIZATIONS = ("Raw", "ATR", "Average Range", "Percent", "RMS")


def _ma(series: pd.Series, length: int, kind: str, volume: pd.Series | None) -> pd.Series:
    if kind == "EMA":
        return ema(series, length)
    if kind == "RMA":
        return rma(series, length)
    if kind == "SMA":
        return sma(series, length)
    if kind == "WMA":
        return wma(series, length)
    if kind == "HMA":
        return hma(series, length)
    if kind == "VWMA":
        if volume is None:
            raise ValueError("VWMA 均线需要 volume 列")
        return vwma(series, volume, length)
    raise ValueError(f"未知均线类型: {kind}")


def _smooth(series: pd.Series, length: int, kind: str) -> pd.Series:
    if length <= 1:
        return series
    if kind == "EMA":
        return ema(series, length)
    if kind == "RMA":
        return rma(series, length)
    if kind == "SMA":
        return sma(series, length)
    raise ValueError(f"未知平滑类型: {kind}")


def elder_ray_pressure_engine(
        frame: pd.DataFrame, *, engine: str = "Classic Elder", length: int = 13,
        ma_type: str = "EMA", normalization: str = "ATR", norm_length: int = 14,
        smooth_length: int = 3, smooth_type: str = "EMA", impulse_length: int = 3,
        extreme_lookback: int = 200, extreme_threshold: float = 90.0,
        extreme_recency_bars: int = 3, price_push_lookback: int = 3,
        show_divergence: bool = True, pivot_left: int = 5, pivot_right: int = 5,
        basis_source: str = "close") -> pd.DataFrame:
    choice("engine", engine, _ENGINES)
    integer("length", length, 2)
    choice("ma_type", ma_type, _MA_TYPES)
    choice("normalization", normalization, _NORMALIZATIONS)
    integer("norm_length", norm_length, 2)
    integer("smooth_length", smooth_length, 1)
    choice("smooth_type", smooth_type, _SMOOTH_TYPES)
    integer("impulse_length", impulse_length, 1)
    integer("extreme_lookback", extreme_lookback, 20)
    number("extreme_threshold", extreme_threshold, 50.0, 99.0)
    integer("extreme_recency_bars", extreme_recency_bars, 1, 10)
    integer("price_push_lookback", price_push_lookback, 1, 10)
    boolean("show_divergence", show_divergence)
    integer("pivot_left", pivot_left, 1)
    integer("pivot_right", pivot_right, 1)
    choice("basis_source", basis_source, ("close", "hlc3", "hl2", "ohlc4"))

    columns = ["high", "low", "close"]
    if basis_source in ("hlc3", "hl2", "ohlc4"):
        columns.append("open")
    if ma_type == "VWMA":
        columns.append("volume")
    ohlcv(frame, *columns)
    high, low, close = frame["high"].astype(float), frame["low"].astype(float), frame["close"].astype(float)
    if basis_source == "close":
        source = close
    elif basis_source == "hlc3":
        source = (high + low + close) / 3
    elif basis_source == "hl2":
        source = (high + low) / 2
    else:
        source = (high + low + close + frame["open"].astype(float)) / 4
    volume = frame["volume"].astype(float) if "volume" in frame.columns else None

    basis = _ma(source, length, ma_type, volume)

    classic_bull = high - basis
    classic_bear = low - basis
    previous_close = close.shift(1).fillna(high)
    previous_close_low = close.shift(1).fillna(low)
    true_high = np.maximum(high, previous_close)
    true_low = np.minimum(low, previous_close_low)
    tr_bull = true_high - basis
    tr_bear = true_low - basis
    bar_range = high - low
    clv = np.where(bar_range > 0, (2.0 * close - high - low) / bar_range.replace(0, np.nan), 0.0)
    clv = pd.Series(clv, index=frame.index).fillna(0.0)
    bull_close_weight = (clv + 1.0) * 0.5
    bear_close_weight = (1.0 - clv) * 0.5

    if engine == "Classic Elder":
        bull_raw, bear_raw = classic_bull, classic_bear
    elif engine == "Directional Elder":
        bull_raw, bear_raw = classic_bull.clip(lower=0.0), classic_bear.clip(upper=0.0)
    elif engine == "True Range Elder":
        bull_raw, bear_raw = tr_bull, tr_bear
    else:  # Close-Weighted Elder
        bull_raw = classic_bull.clip(lower=0.0) * bull_close_weight
        bear_raw = classic_bear.clip(upper=0.0) * bear_close_weight

    average_true_range = atr(frame, norm_length)
    avg_range = ema(bar_range, norm_length)
    raw_magnitude = np.sqrt((bull_raw * bull_raw + bear_raw * bear_raw) / 2.0)
    rms_scale = np.sqrt(sma(raw_magnitude * raw_magnitude, norm_length))

    def normalize(value: pd.Series) -> pd.Series:
        if normalization == "Raw":
            return value
        if normalization == "ATR":
            return np.where(average_true_range.isna(), np.nan,
                            np.where(average_true_range != 0, value / average_true_range, 0.0))
        if normalization == "Average Range":
            return np.where(avg_range.isna(), np.nan,
                            np.where(avg_range != 0, value / avg_range, 0.0))
        if normalization == "Percent":
            return np.where(basis.isna(), np.nan,
                            np.where(basis != 0, 100.0 * value / basis.abs(), 0.0))
        return np.where(rms_scale.isna(), np.nan,
                        np.where(rms_scale != 0, value / rms_scale, 0.0))

    bull = _smooth(pd.Series(normalize(bull_raw), index=frame.index), smooth_length, smooth_type)
    bear = _smooth(pd.Series(normalize(bear_raw), index=frame.index), smooth_length, smooth_type)

    net = bull + bear
    gross_pressure = bull.abs() + bear.abs()
    dominance = pd.Series(np.where(gross_pressure.isna(), np.nan,
                                   np.where(gross_pressure > 0, net / gross_pressure, 0.0)),
                          index=frame.index)
    impulse = net - net.shift(impulse_length).fillna(net)
    acceleration = impulse - impulse.shift(1).fillna(impulse)

    basis_up = flags(basis > basis.shift(1))
    basis_down = flags(basis < basis.shift(1))
    bar_position = pd.Series(np.where(low > basis, "Above",
                                      np.where(high < basis, "Below", "Straddle")),
                             index=frame.index, dtype=object)

    bull_rank = percentrank(bull, extreme_lookback)
    bear_severity = -bear
    bear_rank = percentrank(bear_severity, extreme_lookback)
    bull_extreme = flags(bull_rank >= extreme_threshold)
    bear_extreme = flags(bear_rank >= extreme_threshold)
    recent_bull_extreme = flags(barssince(bull_extreme) < extreme_recency_bars)
    recent_bear_extreme = flags(barssince(bear_extreme) < extreme_recency_bars)
    bull_pressure_falling = flags(bull < bull.shift(1))
    bear_pressure_recovering = flags(bear > bear.shift(1))
    price_push_high = flags(high >= highest(high, price_push_lookback).shift(1))
    price_push_low = flags(low <= lowest(low, price_push_lookback).shift(1))
    bull_exhaustion_candidate = (recent_bull_extreme & bull_pressure_falling
                                 & price_push_high)
    bear_exhaustion_candidate = (recent_bear_extreme & bear_pressure_recovering
                                 & price_push_low)
    bull_exhaustion = flags(bull_exhaustion_candidate & ~bear_exhaustion_candidate)
    bear_exhaustion = flags(bear_exhaustion_candidate & ~bull_exhaustion_candidate)
    dual_exhaustion = flags(bull_exhaustion_candidate & bear_exhaustion_candidate)

    impulse_positive = flags(impulse > 0)
    impulse_negative = flags(impulse < 0)
    net_positive = flags(net > 0)
    net_negative = flags(net < 0)
    pressure_state = pd.Series(
        np.where(dual_exhaustion, "Dual Exhaustion",
                 np.where(bull_exhaustion, "Bull Exhaustion",
                          np.where(bear_exhaustion, "Bear Exhaustion",
                                   np.where(net_positive & impulse_positive, "Bull Expansion",
                                            np.where(net_positive & impulse_negative, "Bull Deceleration",
                                                     np.where(net_negative & impulse_negative, "Bear Expansion",
                                                              np.where(net_negative & impulse_positive, "Bear Deceleration",
                                                                       "Balanced"))))))),
        index=frame.index, dtype=object)

    regime = pd.Series(
        np.where(basis_up & net_positive, "Bull Confirmed",
                 np.where(basis_up & net_negative, "Bull Pullback",
                          np.where(basis_down & net_negative, "Bear Confirmed",
                                   np.where(basis_down & net_positive, "Bear Rally",
                                            "Neutral")))),
        index=frame.index, dtype=object)

    price_pivot_high = pivothigh(high, pivot_left, pivot_right)
    price_pivot_low = pivotlow(low, pivot_left, pivot_right)
    bull_at_high_pivot = bull.shift(pivot_right).where(price_pivot_high.notna())
    bear_at_low_pivot = bear.shift(pivot_right).where(price_pivot_low.notna())
    pivot_high_true = flags(price_pivot_high.notna())
    pivot_low_true = flags(price_pivot_low.notna())
    previous_price_high = valuewhen(pivot_high_true, price_pivot_high.fillna(np.nan), 1)
    previous_bull_high = valuewhen(pivot_high_true, bull_at_high_pivot, 1)
    previous_price_low = valuewhen(pivot_low_true, price_pivot_low.fillna(np.nan), 1)
    previous_bear_low = valuewhen(pivot_low_true, bear_at_low_pivot, 1)
    if show_divergence:
        bearish_divergence = flags(price_pivot_high.notna() & previous_price_high.notna()
                                   & (price_pivot_high > previous_price_high)
                                   & (bull_at_high_pivot < previous_bull_high))
        bullish_divergence = flags(price_pivot_low.notna() & previous_price_low.notna()
                                   & (price_pivot_low < previous_price_low)
                                   & (bear_at_low_pivot > previous_bear_low))
    else:
        false = pd.Series(False, index=frame.index)
        bearish_divergence = bullish_divergence = false

    return pd.DataFrame({
        "basis": basis,
        "bull": bull,
        "bear": bear,
        "net": net,
        "gross_pressure": gross_pressure,
        "dominance": dominance,
        "impulse": impulse,
        "acceleration": acceleration,
        "bar_position": bar_position,
        "bull_rank": bull_rank,
        "bear_rank": bear_rank,
        "bull_extreme": bull_extreme,
        "bear_extreme": bear_extreme,
        "bull_exhaustion": bull_exhaustion,
        "bear_exhaustion": bear_exhaustion,
        "dual_exhaustion": dual_exhaustion,
        "pressure_state": pressure_state,
        "regime": regime,
        "bearish_divergence": bearish_divergence,
        "bullish_divergence": bullish_divergence,
    }, index=frame.index)
