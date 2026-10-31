"""Market Structure Advanced — pivot-structure score with divergence weighting.

Source: pine-collection/indicators/market_structure/market_structure_advanced/market_structure_advanced.pine
(WavesUnchained, MIT)

Confirmed price pivots produce a structure score per event: a higher high adds
+2 and a lower high −1, a higher low +1 and a lower low −2 (so bearish
structure counts double). The last `score_window` events are summed, scaled to
±100 and smoothed; the histogram is score minus its smoothed signal line, and
cross flags fire on score/signal and score/zero crossings. The structure
labels (HH/HL/LH/LL classification) are exposed as boolean columns.

Dropped: all plots, fills, gradient colours, chart labels and the alert
plumbing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (choice, crossover, crossunder, flags, integer, ohlcv, pivothigh,
                    pivotlow, sma)


def market_structure_advanced(
        frame: pd.DataFrame, *, pivot_left: int = 5, pivot_right: int = 5,
        score_window: int = 2, signal_length: int = 3,
        smoothing: str = "EMA") -> pd.DataFrame:
    integer("pivot_left", pivot_left, 1)
    integer("pivot_right", pivot_right, 1)
    integer("score_window", score_window, 2)
    integer("signal_length", signal_length, 1)
    choice("smoothing", smoothing,
           ("EMA", "SMA", "RMA", "WMA", "SuperSmoother", "T3", "KAMA", "JMA Approx"))
    high, low, _ = ohlcv(frame, "high", "low", "close")

    piv_high = pivothigh(high, pivot_left, pivot_right)
    piv_low = pivotlow(low, pivot_left, pivot_right)

    last_high = prev_high = np.nan
    last_low = prev_low = np.nan
    is_higher_high = np.zeros(len(frame), dtype=object)  # None / True / False
    is_higher_low = np.zeros(len(frame), dtype=object)
    score = np.full(len(frame), np.nan)
    ring: list[float] = []

    for position in range(len(frame)):
        current_score = np.nan
        pivot_high_value = piv_high.iloc[position] if position < len(piv_high) else np.nan
        pivot_low_value = piv_low.iloc[position] if position < len(piv_low) else np.nan
        if not np.isnan(pivot_high_value):
            prev_high = last_high
            last_high = float(pivot_high_value)
            if not np.isnan(prev_high):
                is_higher_high[position] = pivot_high_value > prev_high
                current_score = 2.0 if pivot_high_value > prev_high else -1.0
        if not np.isnan(pivot_low_value):
            prev_low = last_low
            last_low = float(pivot_low_value)
            if not np.isnan(prev_low):
                is_higher_low[position] = pivot_low_value > prev_low
                low_delta = 1.0 if pivot_low_value > prev_low else -2.0
                current_score = (low_delta if np.isnan(current_score)
                                 else current_score + low_delta)
        if not np.isnan(current_score):
            ring.append(current_score)
            if len(ring) > score_window:
                ring.pop(0)
            raw = float(sum(ring))
            score[position] = max(-100.0, min(100.0, 100.0 * raw / (score_window * 2.0)))
        elif position > 0 and not np.isnan(score[position - 1]):
            score[position] = score[position - 1]  # Pine: score persists between events

    score_series = pd.Series(score, index=frame.index)
    signal = _smooth(score_series, signal_length, smoothing)
    hist = score_series - signal

    def label_series(values):
        return pd.Series([v if v is not None else False for v in values], index=frame.index,
                         dtype=object).eq(True)

    higher_high = label_series(is_higher_high)
    lower_high = label_series([v if v is not None else False for v in is_higher_high]) \
        .eq(False) & flags(piv_high.notna())
    higher_low = label_series(is_higher_low)
    lower_low = label_series([v if v is not None else False for v in is_higher_low]) \
        .eq(False) & flags(piv_low.notna())

    return pd.DataFrame({
        "score": score_series,
        "signal": signal,
        "histogram": hist,
        "bull_cross": crossover(score_series, signal),
        "bear_cross": crossunder(score_series, signal),
        "bull_zero_cross": crossover(score_series, 0.0),
        "bear_zero_cross": crossunder(score_series, 0.0),
        "higher_high": higher_high,
        "lower_high": lower_high,
        "higher_low": higher_low,
        "lower_low": lower_low,
    }, index=frame.index)


def _smooth(series: pd.Series, length: int, mode: str) -> pd.Series:
    from ._pine import ema, rma, wma
    if mode == "SMA":
        return sma(series, length)
    if mode == "RMA":
        return rma(series, length)
    if mode == "WMA":
        return wma(series, length)
    from .commodity_flowtrend import _smooth as flow_smooth
    return flow_smooth(series, length, mode, 50, 2.0)
