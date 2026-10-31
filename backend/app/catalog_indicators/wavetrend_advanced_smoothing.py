"""WaveTrend Advanced Smoothing — WaveTrend oscillator with a smoothing menu.

Source: pine-collection/indicators/momentum/wavetrend_advanced_smoothing/wavetrend_advanced_smoothing.pine
(WavesUnchained, MIT)

Classic LazyBear WaveTrend (ESA -> CI -> double smoothing) where every moving
average runs through a selectable smoothing family (EMA/SMA/RMA/WMA/
SuperSmoother/T3/KAMA/JMA). Three scale modes: raw classic values, a hard
±100 clamp, or an adaptive rescale dividing by the rolling max absolute value
of the two lines. Cross flags fire on the raw pair; extreme markers require
the cross to happen outside the OB/OS band; the sentiment score maps WT into
a signed ±100 reading against the OB/OS levels.

Dropped: all plots, fills, hlines, dots, labels and the sentiment bar text.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (choice, crossover, crossunder, flags, highest, integer, number, ohlcv,
                    choice as _choice)

_SMOOTHINGS = ("EMA", "SMA", "RMA", "WMA", "SuperSmoother", "T3", "KAMA", "JMA Approx")
_SCALES = ("Classic", "Clamp ±100", "Adaptive ±100")


def wavetrend_advanced_smoothing(
        frame: pd.DataFrame, *, source: str = "hlc3", channel_length: int = 10,
        average_length: int = 21, signal_length: int = 4, smoothing: str = "EMA",
        jma_phase: int = 50, jma_power: float = 2.0, scale_mode: str = "Classic",
        scale_lookback: int = 200, overbought1: float = 60.0, overbought2: float = 100.0,
        oversold1: float = -60.0, oversold2: float = -100.0) -> pd.DataFrame:
    _choice("source", source, ("close", "hlc3", "hl2", "ohlc4"))
    integer("channel_length", channel_length, 1)
    integer("average_length", average_length, 1)
    integer("signal_length", signal_length, 1)
    choice("smoothing", smoothing, _SMOOTHINGS)
    integer("jma_phase", jma_phase, -100, 100)
    number("jma_power", jma_power, 1.0, 4.0)
    choice("scale_mode", scale_mode, _SCALES)
    integer("scale_lookback", scale_lookback, 20)
    number("overbought1", overbought1, -1000.0, 1000.0)
    number("oversold1", oversold1, -1000.0, 1000.0)
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

    esa = smooth(src, channel_length)
    deviation = smooth((src - esa).abs(), channel_length)
    ci = (src - esa) / (0.015 * deviation.abs().clip(lower=1e-10))
    wt1_raw = smooth(ci, average_length)
    wt2_raw = smooth(wt1_raw, signal_length)

    scale_abs = highest(np.maximum(wt1_raw.abs(), wt2_raw.abs()), scale_lookback)

    def apply_scale(values: pd.Series) -> pd.Series:
        if scale_mode == "Clamp ±100":
            return values.clip(lower=-100.0, upper=100.0)
        if scale_mode == "Adaptive ±100":
            return pd.Series(np.where(scale_abs != 0.0, 100.0 * values / scale_abs, 0.0),
                             index=frame.index)
        return values

    wt1 = apply_scale(wt1_raw)
    wt2 = apply_scale(wt2_raw)
    hist = wt1 - wt2

    bull_cross = crossover(wt1_raw, wt2_raw)
    bear_cross = crossunder(wt1_raw, wt2_raw)
    bull_extreme = flags(bull_cross & (wt1_raw < oversold1))
    bear_extreme = flags(bear_cross & (wt1_raw > overbought1))

    sent_bull = np.where(overbought1 != 0.0,
                         wt1_raw.clip(lower=0.0) / overbought1 * 100.0, 0.0)
    sent_bear = np.where(oversold1 != 0.0,
                         (-wt1_raw).clip(lower=0.0) / abs(oversold1) * 100.0, 0.0)
    sentiment = pd.Series(np.clip(sent_bull - sent_bear, -100.0, 100.0),
                          index=frame.index)

    return pd.DataFrame({
        "wt1": wt1,
        "wt2": wt2,
        "wt1_raw": wt1_raw,
        "wt2_raw": wt2_raw,
        "histogram": hist,
        "bull_cross": bull_cross,
        "bear_cross": bear_cross,
        "bull_extreme": bull_extreme,
        "bear_extreme": bear_extreme,
        "sentiment_score": sentiment,
        "overbought2_level": pd.Series(float(overbought2), index=frame.index),
        "oversold2_level": pd.Series(float(oversold2), index=frame.index),
    }, index=frame.index)
