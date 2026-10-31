"""CCI Advanced — smoothed CCI with context/stall/scale options.

Source: pine-collection/indicators/momentum/cci_advanced/cci_advanced.pine
(WavesUnchained, MIT)

The raw CCI of the chosen source is smoothed into the fast line, then again
into the signal line; a slow trend-context CCI (longer length) sits behind.
Three scale modes: raw classic values, a hard ±200 clamp, or an adaptive
rescale dividing by the rolling max absolute value of the fast/signal pair.
Crosses of the fast line against the signal, the zero level and the OB/OS
zone history feed the extreme signals; stall detection flags sharp CCI moves
with flat ATR-normalised price; quality blends zone depth (50%), context
agreement (25%) and the absence of an opposing stall (25%). Sentiment maps
the CCI into a signed ±100 reading against the OB/OS levels.

Dropped: all plots, fills, hlines, markers and the sentiment bar label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (atr, choice, crossover, crossunder, dev, edge, flags, highest, integer,
                    lowest, number, ohlcv, sma)

_SMOOTHINGS = ("EMA", "SMA", "RMA", "WMA", "SuperSmoother", "T3", "KAMA", "JMA Approx")
_SCALES = ("Classic", "Clamp ±200", "Adaptive ±100")


def cci_advanced(
        frame: pd.DataFrame, *, source: str = "hlc3", cci_len: int = 20,
        avg_len: int = 3, sig_len: int = 9, smoothing: str = "EMA",
        jma_phase: int = 50, jma_power: float = 2.0, scale_mode: str = "Classic",
        scale_lookback: int = 200, overbought1: float = 100.0,
        overbought2: float = 200.0, oversold1: float = -100.0,
        oversold2: float = -200.0, use_extreme_only: bool = True,
        lookback_extreme: int = 5, use_stall: bool = True, stall_len: int = 5,
        stall_flat_atr: float = 0.5, stall_cci_min: float = 40.0,
        use_context: bool = True, ctx_len: int = 100, divergence_len: int = 4,
        divergence_min_slope: float = 25.0) -> pd.DataFrame:
    choice("source", source, ("close", "hlc3", "hl2", "ohlc4"))
    integer("cci_len", cci_len, 1)
    integer("avg_len", avg_len, 1)
    integer("sig_len", sig_len, 1)
    choice("smoothing", smoothing, _SMOOTHINGS)
    integer("jma_phase", jma_phase, -100, 100)
    number("jma_power", jma_power, 1.0, 4.0)
    choice("scale_mode", scale_mode, _SCALES)
    integer("scale_lookback", scale_lookback, 20)
    integer("lookback_extreme", lookback_extreme, 1)
    integer("stall_len", stall_len, 1)
    number("stall_flat_atr", stall_flat_atr, 0.0)
    number("stall_cci_min", stall_cci_min, 0.0)
    integer("ctx_len", ctx_len, 10)
    integer("divergence_len", divergence_len, 1)
    number("divergence_min_slope", divergence_min_slope, 0.0)
    high, low, close = ohlcv(frame, "high", "low", "close")
    src = {"close": close, "hlc3": (high + low + close) / 3,
           "hl2": (high + low) / 2,
           "ohlc4": (high + low + close + frame["open"].astype(float)) / 4}[source]

    from .commodity_flowtrend import _smooth as flow_smooth

    def smooth(series: pd.Series, length: int) -> pd.Series:
        return flow_smooth(series, length, smoothing, jma_phase, jma_power)

    def cci_of(series: pd.Series, length: int) -> pd.Series:
        average = sma(series, length)
        deviation = dev(series, length)
        return (series - average) / (0.015 * deviation.replace(0, np.nan))

    cci_line = smooth(cci_of(src, cci_len), avg_len)
    signal = smooth(cci_line, sig_len)
    hist = cci_line - signal
    ctx_line = smooth(cci_of(src, ctx_len), avg_len)

    scale_abs = highest(np.maximum(cci_line.abs(), signal.abs()), scale_lookback)

    def apply_scale(values: pd.Series) -> pd.Series:
        if scale_mode == "Clamp ±200":
            return values.clip(lower=-200.0, upper=200.0)
        if scale_mode == "Adaptive ±100":
            return pd.Series(np.where(scale_abs != 0.0, 100.0 * values / scale_abs, 0.0),
                             index=frame.index)
        return values

    cci_disp = apply_scale(cci_line)
    sig_disp = apply_scale(signal)
    ctx_disp = apply_scale(ctx_line)

    was_oversold = flags(lowest(cci_line, lookback_extreme) <= oversold1)
    was_overbought = flags(highest(cci_line, lookback_extreme) >= overbought1)
    bull_cross = crossover(cci_line, signal)
    bear_cross = crossunder(cci_line, signal)
    bull_extreme = flags(bull_cross & ((not use_extreme_only) | was_oversold))
    bear_extreme = flags(bear_cross & ((not use_extreme_only) | was_overbought))
    bull_zero_cross = crossover(cci_line, 0.0)
    bear_zero_cross = crossunder(cci_line, 0.0)

    atr_reference = atr(frame, 14)
    cci_change = cci_line - cci_line.shift(stall_len)
    price_change = pd.Series(np.where(atr_reference != 0.0,
                                      (close - close.shift(stall_len)) / atr_reference, 0.0),
                             index=frame.index)
    flat_price = flags(price_change.abs() < stall_flat_atr)
    if use_stall:
        stall_bull = flags((cci_change < -stall_cci_min) & flat_price)
        stall_bear = flags((cci_change > stall_cci_min) & flat_price)
    else:
        stall_bull = flags(pd.Series(False, index=frame.index))
        stall_bear = flags(pd.Series(False, index=frame.index))

    ctx_bull = flags(ctx_line > 0.0)
    ctx_bear = flags(ctx_line < 0.0)
    bear_counter = flags(use_context & bear_extreme & ctx_bull)
    bull_counter = flags(use_context & bull_extreme & ctx_bear)

    fast_direction = cci_line - cci_line.shift(divergence_len)
    slow_direction = ctx_line - ctx_line.shift(divergence_len)
    divergence_bull = flags(use_context & (fast_direction < -divergence_min_slope)
                            & (slow_direction > 0.0))
    divergence_bear = flags(use_context & (fast_direction > divergence_min_slope)
                            & (slow_direction < 0.0))

    bear_strong = flags(bear_extreme & ~(stall_bull | bear_counter))
    bear_weak = flags(bear_extreme & (stall_bull | bear_counter))
    bull_strong = flags(bull_extreme & ~(stall_bear | bull_counter))
    bull_weak = flags(bull_extreme & (stall_bear | bull_counter))

    sent_bull = np.where(overbought1 != 0.0,
                         cci_line.clip(lower=0.0) / overbought1 * 100.0, 0.0)
    sent_bear = np.where(oversold1 != 0.0,
                         (-cci_line).clip(lower=0.0) / abs(oversold1) * 100.0, 0.0)
    sentiment = pd.Series(np.clip(sent_bull - sent_bear, -100.0, 100.0),
                          index=frame.index)

    cci_lowest = lowest(cci_line, lookback_extreme)
    cci_highest = highest(cci_line, lookback_extreme)
    bull_depth = np.where(oversold1 != 0.0,
                          np.clip((oversold1 - cci_lowest) / abs(oversold1), 0.0, 1.0), 0.0)
    bear_depth = np.where(overbought1 != 0.0,
                          np.clip((cci_highest - overbought1) / abs(overbought1), 0.0, 1.0),
                          0.0)
    bull_quality = np.minimum(100.0, bull_depth * 50.0 + (ctx_line > 0.0) * 25.0
                              + (~stall_bear) * 25.0)
    bear_quality = np.minimum(100.0, bear_depth * 50.0 + (ctx_line < 0.0) * 25.0
                              + (~stall_bull) * 25.0)

    return pd.DataFrame({
        "cci": cci_disp,
        "cci_raw": cci_line,
        "signal": sig_disp,
        "signal_raw": signal,
        "histogram": hist,
        "context_line": ctx_disp,
        "context_raw": ctx_line,
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
        "bull_zero_cross": bull_zero_cross,
        "bear_zero_cross": bear_zero_cross,
        "stall_bull": stall_bull,
        "stall_bear": stall_bear,
        "counter_trend_bull": bull_counter,
        "counter_trend_bear": bear_counter,
        "divergence_bull": divergence_bull,
        "divergence_bear": divergence_bear,
        "sentiment_score": sentiment,
        "bull_quality": pd.Series(bull_quality, index=frame.index),
        "bear_quality": pd.Series(bear_quality, index=frame.index),
        "overbought2_level": pd.Series(float(overbought2), index=frame.index),
        "oversold2_level": pd.Series(float(oversold2), index=frame.index),
    }, index=frame.index)
