"""Stoch RSI Advanced — double-smoothed StochRSI with context/stall/quality.

Source: pine-collection/indicators/momentum/stoch_rsi_advanced/stoch_rsi_advanced.pine
(WavesUnchained, MIT)

Core is the classic StochRSI (RSI of the source, then stochastics of the RSI
over `stoch_len`, both bounded 0..100), smoothed twice into the K line and the
D signal line. A slow trend-context line (longer RSI/Stoch lengths) sits
behind it: fast crosses against the context side of the midline are flagged
counter-trend. Stall detection marks bars where the oscillator moves sharply
against its recent self while price barely moves (ATR-normalised). Extreme
signals require the K line to have been inside the OB/OS zone within the
lookback; quality scores weight zone depth (50%), context agreement (25%) and
the absence of an opposing stall (25%). The sentiment score maps K's position
between the midline and the OB/OS levels to a signed ±100 reading.

Dropped: all plots, fills, hlines, markers, the sentiment bar label and the
gradient colours.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (atr, boolean, choice, crossover, crossunder, flags, highest, integer,
                    lowest, number, ohlcv, rsi, stoch)

_SMOOTHINGS = ("EMA", "SMA", "RMA", "WMA", "SuperSmoother", "T3", "KAMA", "JMA Approx")


def stoch_rsi_advanced(
        frame: pd.DataFrame, *, source: str = "close", rsi_len: int = 14,
        stoch_len: int = 14, k_len: int = 3, d_len: int = 3, smoothing: str = "SMA",
        jma_phase: int = 50, jma_power: float = 2.0, mid_line: float = 50.0,
        ob_level: float = 80.0, os_level: float = 20.0, use_stall: bool = True,
        stall_len: int = 5, stall_flat_atr: float = 0.5, stall_osc_min: float = 15.0,
        use_context: bool = True, ctx_rsi_len: int = 50, ctx_stoch_len: int = 50,
        use_extreme_only: bool = True, lookback_extreme: int = 5,
        divergence_len: int = 4, divergence_min_slope: float = 10.0) -> pd.DataFrame:
    choice("source", source, ("close", "hlc3", "hl2", "ohlc4"))
    integer("rsi_len", rsi_len, 1)
    integer("stoch_len", stoch_len, 1)
    integer("k_len", k_len, 1)
    integer("d_len", d_len, 1)
    choice("smoothing", smoothing, _SMOOTHINGS)
    integer("jma_phase", jma_phase, -100, 100)
    number("jma_power", jma_power, 1.0, 4.0)
    number("mid_line", mid_line, 0.0, 100.0)
    number("ob_level", ob_level, 50.0, 100.0)
    number("os_level", os_level, 0.0, 50.0)
    integer("stall_len", stall_len, 1)
    number("stall_flat_atr", stall_flat_atr, 0.0)
    integer("ctx_rsi_len", ctx_rsi_len, 2)
    integer("ctx_stoch_len", ctx_stoch_len, 2)
    integer("lookback_extreme", lookback_extreme, 1)
    integer("divergence_len", divergence_len, 1)
    number("divergence_min_slope", divergence_min_slope, 0.0)
    high, low, close = ohlcv(frame, "high", "low", "close")
    if source == "close":
        src = close
    elif source == "hlc3":
        src = (high + low + close) / 3
    elif source == "hl2":
        src = (high + low) / 2
    else:
        src = (high + low + close + frame["open"].astype(float)) / 4

    from .commodity_flowtrend import _smooth as flow_smooth

    def smooth(series: pd.Series, length: int) -> pd.Series:
        return flow_smooth(series, length, smoothing, jma_phase, jma_power)

    def stoch_rsi(series: pd.Series, r_length: int, s_length: int) -> pd.Series:
        r_value = rsi(series, r_length)
        return stoch(r_value, r_value, r_value, s_length)

    stoch_raw = stoch_rsi(src, rsi_len, stoch_len)
    k_line = smooth(stoch_raw, k_len)
    signal = smooth(k_line, d_len)
    hist = k_line - signal

    ctx_raw = stoch_rsi(src, ctx_rsi_len, ctx_stoch_len)
    ctx_line = smooth(ctx_raw, k_len)

    was_oversold = flags(lowest(k_line, lookback_extreme) <= os_level)
    was_overbought = flags(highest(k_line, lookback_extreme) >= ob_level)
    bull_cross = crossover(k_line, signal)
    bear_cross = crossunder(k_line, signal)
    bull_extreme = flags(bull_cross & ((not use_extreme_only) | was_oversold))
    bear_extreme = flags(bear_cross & ((not use_extreme_only) | was_overbought))
    bull_mid_cross = crossover(k_line, mid_line)
    bear_mid_cross = crossunder(k_line, mid_line)

    atr_reference = atr(frame, 14)
    oscillator_change = k_line - k_line.shift(stall_len)
    price_change = pd.Series(np.where(atr_reference != 0.0,
                                      (close - close.shift(stall_len)) / atr_reference, 0.0),
                             index=frame.index)
    flat_price = flags(price_change.abs() < stall_flat_atr)
    if use_stall:
        stall_bull = flags((oscillator_change < -stall_osc_min) & flat_price)
        stall_bear = flags((oscillator_change > stall_osc_min) & flat_price)
    else:
        stall_bull = flags(pd.Series(False, index=frame.index))
        stall_bear = flags(pd.Series(False, index=frame.index))

    ctx_bull = flags(ctx_line > mid_line)
    ctx_bear = flags(ctx_line < mid_line)
    bear_counter = flags(use_context & bear_extreme & ctx_bull) if use_context \
        else flags(pd.Series(False, index=frame.index))
    bull_counter = flags(use_context & bull_extreme & ctx_bear) if use_context \
        else flags(pd.Series(False, index=frame.index))

    bear_strong = flags(bear_extreme & ~(stall_bull | bear_counter))
    bear_weak = flags(bear_extreme & (stall_bull | bear_counter))
    bull_strong = flags(bull_extreme & ~(stall_bear | bull_counter))
    bull_weak = flags(bull_extreme & (stall_bear | bull_counter))

    bull_pct = np.where(ob_level != mid_line,
                        (k_line.clip(lower=mid_line) - mid_line) / (ob_level - mid_line) * 100.0,
                        0.0)
    bear_pct = np.where(mid_line != os_level,
                        (mid_line - k_line.clip(upper=mid_line)) / (mid_line - os_level) * 100.0,
                        0.0)
    sentiment = pd.Series(np.clip(bull_pct - bear_pct, -100.0, 100.0), index=frame.index)

    k_lowest = lowest(k_line, lookback_extreme)
    k_highest = highest(k_line, lookback_extreme)
    bull_depth = np.where(os_level > 0.0,
                          np.clip((os_level - k_lowest) / os_level, 0.0, 1.0), 0.0)
    bear_depth = np.where(ob_level < 100.0,
                          np.clip((k_highest - ob_level) / (100.0 - ob_level), 0.0, 1.0), 0.0)
    bull_quality = pd.Series(np.minimum(100.0, bull_depth * 50.0 + (ctx_line > mid_line) * 25.0
                                        + (~stall_bear) * 25.0), index=frame.index)
    bear_quality = pd.Series(np.minimum(100.0, bear_depth * 50.0 + (ctx_line < mid_line) * 25.0
                                        + (~stall_bull) * 25.0), index=frame.index)

    fast_direction = k_line - k_line.shift(divergence_len)
    slow_direction = ctx_line - ctx_line.shift(divergence_len)
    divergence_bull = flags(use_context & (fast_direction < -divergence_min_slope)
                            & (slow_direction > 0.0))
    divergence_bear = flags(use_context & (fast_direction > divergence_min_slope)
                            & (slow_direction < 0.0))

    return pd.DataFrame({
        "k": k_line,
        "d": signal,
        "histogram": hist,
        "context_line": ctx_line,
        "was_oversold": was_oversold,
        "was_overbought": was_overbought,
        "bull_cross": bull_cross,
        "bear_cross": bear_cross,
        "bull_extreme": bull_extreme,
        "bear_extreme": bear_extreme,
        "bull_strong": bull_strong,
        "bull_weak": bull_weak,
        "bear_strong": bear_strong,
        "bear_weak": bear_weak,
        "bull_mid_cross": bull_mid_cross,
        "bear_mid_cross": bear_mid_cross,
        "stall_bull": stall_bull,
        "stall_bear": stall_bear,
        "counter_trend_bull": bull_counter,
        "counter_trend_bear": bear_counter,
        "sentiment_score": sentiment,
        "bull_quality": bull_quality,
        "bear_quality": bear_quality,
        "divergence_bull": divergence_bull,
        "divergence_bear": divergence_bear,
    }, index=frame.index)
