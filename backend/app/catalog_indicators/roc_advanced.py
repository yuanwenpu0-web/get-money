"""ROC Advanced — smoothed rate of change with stall, context and conviction.

Source: pine-collection/indicators/momentum/roc_advanced/roc_advanced.pine
(WavesUnchained, MIT)

The raw rate of change `100 * (src - src[len]) / src[len]` (0 when the earlier
value is exactly 0, warm-up stays NaN) is smoothed twice with the same family:
once for the ROC line and once more for the signal line, and a longer-length
raw ROC smoothed the same way provides the trend-context line. Extreme signals
need a signal-line cross plus, optionally, a recent stretch beyond the
positive/negative level; stalls flag ROC moving against an essentially flat
price (ATR-normalised); the sentiment bar rescales the ROC position inside the
stretch zone to ±100 and the quality scores blend stretch depth, context
agreement and stall-freedom.

Dropped: the gradient colouring, histogram/fill styling, hlines, the sentiment
label and quality labels and the alert plumbing. All computed series are
returned as columns regardless of the display toggles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (atr, boolean, choice, crossover, crossunder, flags, integer, number,
                    ohlcv, wma, sma, rma, ema)

_SMOOTHINGS = ("EMA", "SMA", "RMA", "WMA", "SuperSmoother", "T3", "KAMA", "JMA Approx")
_SOURCES = ("close", "hlc3", "hl2", "ohlc4")


def _supersmoother(series: pd.Series, length: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    a1 = np.exp(-1.414 * np.pi / length)
    c2 = 2.0 * a1 * np.cos(1.414 * np.pi / length)
    c3 = -(a1 * a1)
    c1 = 1.0 - c2 - c3
    previous, older = np.nan, np.nan
    for position, value in enumerate(values):
        if np.isnan(value):
            continue
        out[position] = (c1 * (value + (0.0 if np.isnan(previous) else previous)) / 2.0
                         + c2 * (0.0 if np.isnan(previous) else previous)
                         + c3 * (0.0 if np.isnan(older) else older))
        older, previous = previous, out[position]
    return pd.Series(out, index=series.index)


def _t3(series: pd.Series, length: int) -> pd.Series:
    e1 = ema(series, length)
    e2 = ema(e1, length)
    e3 = ema(e2, length)
    e4 = ema(e3, length)
    e5 = ema(e4, length)
    e6 = ema(e5, length)
    b = 0.7
    c1 = -(b * b * b)
    c2 = 3.0 * b * b + 3.0 * b * b * b
    c3 = -6.0 * b * b - 3.0 * b - 3.0 * b * b * b
    c4 = 1.0 + 3.0 * b + b * b * b + 3.0 * b * b
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def _kama(series: pd.Series, length: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    current = np.nan
    for position in range(length, len(values)):
        value = values[position]
        if np.isnan(value) or np.isnan(values[position - 1]):
            continue
        change = abs(value - values[position - length])
        noise = sum(abs(values[i] - values[i - 1]) for i in range(position - length + 1, position + 1))
        efficiency = change / noise if noise != 0.0 else 0.0
        smoothing = (efficiency * (2.0 / 3.0 - 2.0 / 31.0) + 2.0 / 31.0) ** 2
        seed = value if np.isnan(current) else current
        current = seed + smoothing * (value - seed)
        out[position] = current
    return pd.Series(out, index=series.index)


def _jma(series: pd.Series, length: int, phase: int, power: float) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    phase_ratio = phase / 100.0 + 1.5
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2.0)
    alpha = beta ** power
    e0 = e1 = e2 = jma = np.nan
    for position, value in enumerate(values):
        if np.isnan(value):
            continue
        e0 = (1.0 - alpha) * value + alpha * (value if np.isnan(e0) else e0)
        e1 = (value - e0) * (1.0 - beta) + beta * (0.0 if np.isnan(e1) else e1)
        e2 = ((e0 + phase_ratio * e1 - (value if np.isnan(jma) else jma)) * (1.0 - alpha) ** 2
              + alpha ** 2 * (0.0 if np.isnan(e2) else e2))
        jma = (value if np.isnan(jma) else jma) + e2
        out[position] = jma
    return pd.Series(out, index=series.index)


def _smooth(series: pd.Series, length: int, kind: str, jma_phase: int, jma_power: float) -> pd.Series:
    if kind == "SMA":
        return sma(series, length)
    if kind == "RMA":
        return rma(series, length)
    if kind == "WMA":
        return wma(series, length)
    if kind == "SuperSmoother":
        return _supersmoother(series, length)
    if kind == "T3":
        return _t3(series, length)
    if kind == "KAMA":
        return _kama(series, length)
    if kind == "JMA Approx":
        return _jma(series, length, jma_phase, jma_power)
    return ema(series, length)


def _roc(series: pd.Series, length: int) -> pd.Series:
    """Pine's f_roc: 0 when the earlier value is exactly 0, NaN in warm-up."""
    earlier = series.shift(length)
    return (100.0 * (series - earlier) / earlier.where(earlier != 0)).fillna(
        0.0).where(earlier.notna())


def roc_advanced(frame: pd.DataFrame, *, source: str = "close", roc_length: int = 12,
                 avg_length: int = 3, signal_length: int = 9, smoothing: str = "EMA",
                 jma_phase: int = 50, jma_power: float = 2.0,
                 midline: float = 0.0, ob_level: float = 5.0, os_level: float = -5.0,
                 use_stall: bool = True, stall_len: int = 5, stall_flat_atr: float = 0.5,
                 stall_roc_min: float = 3.0,
                 use_ctx: bool = True, ctx_len: int = 50,
                 use_extreme_only: bool = True, lookback_extreme: int = 5,
                 show_div: bool = True, div_len: int = 4, div_min: float = 2.0) -> pd.DataFrame:
    choice("source", source, _SOURCES)
    integer("roc_length", roc_length, 1)
    integer("avg_length", avg_length, 1)
    integer("signal_length", signal_length, 1)
    choice("smoothing", smoothing, _SMOOTHINGS)
    integer("jma_phase", jma_phase, -100, 100)
    number("jma_power", jma_power, 1.0, 4.0)
    number("midline", midline, -1e12, 1e12)
    number("ob_level", ob_level, 0.0)
    number("os_level", os_level, -1e12, 0.0)
    boolean("use_stall", use_stall)
    integer("stall_len", stall_len, 1)
    number("stall_flat_atr", stall_flat_atr, 0.0)
    number("stall_roc_min", stall_roc_min, 0.0)
    boolean("use_ctx", use_ctx)
    integer("ctx_len", ctx_len, 2)
    boolean("use_extreme_only", use_extreme_only)
    integer("lookback_extreme", lookback_extreme, 1)
    boolean("show_div", show_div)
    integer("div_len", div_len, 1)
    number("div_min", div_min, 0.0)
    high, low, close = ohlcv(frame, "high", "low", "close")
    if source == "close":
        src = close
    elif source == "hlc3":
        src = (high + low + close) / 3
    elif source == "hl2":
        src = (high + low) / 2
    else:
        src = (high + low + close + frame["open"].astype(float)) / 4

    def smooth(series: pd.Series, length: int) -> pd.Series:
        return _smooth(series, length, smoothing, jma_phase, jma_power)

    roc_raw = _roc(src, roc_length)
    roc_line = smooth(roc_raw, avg_length)
    signal = smooth(roc_line, signal_length)
    hist = roc_line - signal
    ctx_line = smooth(_roc(src, ctx_len), avg_length)

    was_oversold = flags(roc_line.rolling(lookback_extreme, min_periods=lookback_extreme).min()
                         <= os_level)
    was_overbought = flags(roc_line.rolling(lookback_extreme, min_periods=lookback_extreme).max()
                           >= ob_level)
    bull_cross = crossover(roc_line, signal)
    bear_cross = crossunder(roc_line, signal)
    bull_extreme = bull_cross & ((not use_extreme_only) | was_oversold)
    bear_extreme = bear_cross & ((not use_extreme_only) | was_overbought)
    bull_mid_cross = crossover(roc_line, pd.Series(midline, index=frame.index))
    bear_mid_cross = crossunder(roc_line, pd.Series(midline, index=frame.index))

    atr_ref = atr(frame, 14)
    roc_chg = roc_line - roc_line.shift(stall_len)
    price_chg = (close - close.shift(stall_len)) / atr_ref.where(atr_ref != 0)
    flat_px = flags(price_chg.abs() < stall_flat_atr)
    stall_bull = flags(use_stall & (roc_chg < -stall_roc_min) & flat_px)
    stall_bear = flags(use_stall & (roc_chg > stall_roc_min) & flat_px)

    ctx_bull = flags(ctx_line > midline)
    ctx_bear = flags(ctx_line < midline)
    bear_counter = flags(use_ctx & bear_extreme & ctx_bull)
    bull_counter = flags(use_ctx & bull_extreme & ctx_bear)
    bear_strong = bear_extreme & ~(stall_bull | bear_counter)
    bear_weak = bear_extreme & (stall_bull | bear_counter)
    bull_strong = bull_extreme & ~(stall_bear | bull_counter)
    bull_weak = bull_extreme & (stall_bear | bull_counter)

    span_bull = ob_level - midline
    span_bear = midline - os_level
    sent_bull = (roc_line - midline).clip(lower=0.0) / span_bull * 100.0 if span_bull != 0 \
        else pd.Series(0.0, index=frame.index)
    sent_bear = (midline - roc_line).clip(lower=0.0) / span_bear * 100.0 if span_bear != 0 \
        else pd.Series(0.0, index=frame.index)
    sentiment = (sent_bull - sent_bear).clip(-100.0, 100.0)

    recent_low = roc_line.rolling(lookback_extreme, min_periods=lookback_extreme).min()
    recent_high = roc_line.rolling(lookback_extreme, min_periods=lookback_extreme).max()
    bull_depth = ((os_level - recent_low) / abs(span_bear)).clip(0.0, 1.0) if span_bear != 0 \
        else pd.Series(0.0, index=frame.index)
    bear_depth = ((recent_high - ob_level) / abs(span_bull)).clip(0.0, 1.0) if span_bull != 0 \
        else pd.Series(0.0, index=frame.index)
    bull_quality = (bull_depth * 50.0 + ctx_bull * 25.0 + (~stall_bear) * 25.0).clip(upper=100.0)
    bear_quality = (bear_depth * 50.0 + ctx_bear * 25.0 + (~stall_bull) * 25.0).clip(upper=100.0)

    fast_dir = roc_line - roc_line.shift(div_len)
    slow_dir = ctx_line - ctx_line.shift(div_len)
    div_bull = flags(use_ctx & show_div & (fast_dir < -div_min) & (slow_dir > 0.0))
    div_bear = flags(use_ctx & show_div & (fast_dir > div_min) & (slow_dir < 0.0))

    return pd.DataFrame({
        "roc_raw": roc_raw,
        "roc": roc_line,
        "signal": signal,
        "histogram": hist,
        "ctx": ctx_line,
        "atr_ref": atr_ref,
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
        "ctx_bull": ctx_bull,
        "ctx_bear": ctx_bear,
        "sentiment_score": sentiment,
        "bull_quality": bull_quality,
        "bear_quality": bear_quality,
        "div_bull": div_bull,
        "div_bear": div_bear,
    }, index=frame.index)
