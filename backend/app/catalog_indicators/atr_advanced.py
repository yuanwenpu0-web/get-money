"""ATR Advanced — ATR in four display modes with eight smoothing families.

Source: pine-collection/indicators/volatility/atr_advanced/atr_advanced.pine
(WavesUnchained, v1.0.3)

The raw ATR (Wilder) is optionally re-expressed as a percentage of close, a
z-score against its own rolling mean/stdev, or a percentile rank against the
previous `normalization_lookback` bars, and the selected display series is
smoothed into a signal line whose histogram (display minus signal) drives the
expansion/contraction flags.

The percentile rank counts the previous `normalization_lookback` values that are
less than or equal to the current one (Pine's f_percentile); unlike Pine, where
the incomplete warm-up windows still produce a (misleading) count, warm-up rows
stay NaN here.

Dropped: the gradient colouring, histogram/plot styling, level hlines and the
alert plumbing. All four display variants are reported as columns regardless of
the selected mode.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import atr, choice, crossover, crossunder, ema, flags, integer, number, ohlcv, rma, sma, stdev, wma

_SMOOTHINGS = ("EMA", "SMA", "RMA", "WMA", "SuperSmoother", "T3", "KAMA", "JMA Approx")


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
        volume = sum(abs(values[i] - values[i - 1]) for i in range(position - length + 1, position + 1))
        efficiency = change / volume if volume != 0.0 else 0.0
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


def atr_advanced(frame: pd.DataFrame, *, atr_length: int = 14, signal_length: int = 20,
                 normalization_lookback: int = 100, display_mode: str = "ATR%",
                 smoothing: str = "RMA", jma_phase: int = 50, jma_power: float = 2.0) -> pd.DataFrame:
    integer("atr_length", atr_length, 1)
    integer("signal_length", signal_length, 1)
    integer("normalization_lookback", normalization_lookback, 10)
    choice("display_mode", display_mode, ("Raw", "ATR%", "Normalized", "Percentile Rank"))
    choice("smoothing", smoothing, _SMOOTHINGS)
    integer("jma_phase", jma_phase, -100, 100)
    number("jma_power", jma_power, 1.0, 4.0)
    _, _, close = ohlcv(frame, "high", "low", "close")

    raw = atr(frame, atr_length)
    atr_percent = (100.0 * raw / close).where(close > 0)
    deviation = stdev(raw, normalization_lookback)
    normalized = ((raw - sma(raw, normalization_lookback)) / deviation.where(deviation > 0)).where(
        deviation > 0, 0.0)

    values = raw.to_numpy(dtype=float)
    rank = np.full(len(values), np.nan)
    span = normalization_lookback
    for position in range(span, len(values)):
        if not np.isnan(values[position]) and not np.isnan(values[position - span:position]).any():
            rank[position] = 100.0 * float((values[position - span:position] <= values[position]).sum()) / span
    percentile = pd.Series(rank, index=frame.index)

    display = {"Raw": raw, "ATR%": atr_percent,
               "Normalized": normalized, "Percentile Rank": percentile}[display_mode]
    signal = _smooth(display, signal_length, smoothing, jma_phase, jma_power)
    histogram = display - signal

    return pd.DataFrame({
        "atr_raw": raw,
        "atr_percent": atr_percent,
        "atr_zscore": normalized,
        "atr_percentile": percentile,
        "atr_display": display,
        "atr_signal": signal,
        "atr_histogram": histogram,
        "bull_expansion": crossover(display, signal),
        "bear_contraction": crossunder(display, signal),
        "histogram_rising": flags(histogram >= histogram.shift(1)),
        "histogram_above": flags(display >= signal),
    }, index=frame.index)
