"""Support & Resistance Finder — pivots merged into persistent level pools.

Source: pine-crypto-collection/tamheed_pro/indicators/support-resistance.pine
(tamheed_pro, unknown license — no license header in the source file)

Every confirmed pivot (left = right = `strength`) is merged into a rolling pool:
if the pivot sits within 1% of the confirming bar's high-low range of an existing
level, that level's hit counter is incremented, otherwise the level is appended.
Once the pool exceeds `max_levels` the weakest entry (first minimum) is evicted.
The pools persist across bars, so each bar's flags and nearest levels reflect
every pivot confirmed up to and including that bar — strictly causal.

Breakouts fire when the close crosses a level from the other side; retests need
the close within 2% of the confirming bar's range of a level while staying on
the "rejected" side of it.

`lookback` only positions the drawn lines on the chart and is not a parameter.

Dropped: the line/label drawings, the bottom-right dashboard and the
`showBreakouts`/`showRetests` display toggles (the flags themselves are always
computed, exactly like the Pine variables feeding the status text). Added as a
convenience: `nearest_resistance` (lowest level above the close) and
`nearest_support` (highest level below the close), NaN when no such level exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import flags, integer, ohlcv, pivothigh, pivotlow


def _absorb(levels: list[float], counts: list[int], pivot: float, tolerance: float,
            max_levels: int) -> None:
    for index, level in enumerate(levels):
        if abs(pivot - level) < tolerance:
            counts[index] += 1
            return
    levels.append(pivot)
    counts.append(1)
    if len(levels) > max_levels:
        weakest = counts.index(min(counts))
        levels.pop(weakest)
        counts.pop(weakest)


def support_resistance(frame: pd.DataFrame, *, strength: int = 3,
                       max_levels: int = 5) -> pd.DataFrame:
    integer("strength", strength, 1, 10)
    integer("max_levels", max_levels, 1, 10)
    high, low, close = ohlcv(frame, "high", "low", "close")
    previous_close = close.shift(1)

    pivot_high = pivothigh(high, strength, strength)
    pivot_low = pivotlow(low, strength, strength)

    n = len(frame)
    resistance_levels: list[float] = []
    resistance_counts: list[int] = []
    support_levels: list[float] = []
    support_counts: list[int] = []
    nearest_resistance = np.full(n, np.nan)
    nearest_support = np.full(n, np.nan)
    resistance_count = np.zeros(n, dtype=np.int64)
    support_count = np.zeros(n, dtype=np.int64)
    resistance_breakout = np.zeros(n, dtype=bool)
    support_breakout = np.zeros(n, dtype=bool)
    resistance_retest = np.zeros(n, dtype=bool)
    support_retest = np.zeros(n, dtype=bool)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    previous = previous_close.to_numpy(dtype=float)

    for bar in range(n):
        if not np.isnan(pivot_high.iat[bar]):
            _absorb(resistance_levels, resistance_counts, pivot_high.iat[bar],
                    (highs[bar] - lows[bar]) * 0.01, max_levels)
        if not np.isnan(pivot_low.iat[bar]):
            _absorb(support_levels, support_counts, pivot_low.iat[bar],
                    (highs[bar] - lows[bar]) * 0.01, max_levels)

        retest_tolerance = (highs[bar] - lows[bar]) * 0.02
        for level in resistance_levels:
            if not np.isnan(previous[bar]) and closes[bar] > level and previous[bar] <= level:
                resistance_breakout[bar] = True
            if abs(closes[bar] - level) < retest_tolerance and closes[bar] < level:
                resistance_retest[bar] = True
        for level in support_levels:
            if not np.isnan(previous[bar]) and closes[bar] < level and previous[bar] >= level:
                support_breakout[bar] = True
            if abs(closes[bar] - level) < retest_tolerance and closes[bar] > level:
                support_retest[bar] = True

        above = [level for level in resistance_levels if level > closes[bar]]
        below = [level for level in support_levels if level < closes[bar]]
        if above:
            nearest_resistance[bar] = min(above)
        if below:
            nearest_support[bar] = max(below)
        resistance_count[bar] = len(resistance_levels)
        support_count[bar] = len(support_levels)

    return pd.DataFrame({
        "pivot_high": pivot_high,
        "pivot_low": pivot_low,
        "resistance_count": pd.Series(resistance_count, index=frame.index, dtype="Int64"),
        "support_count": pd.Series(support_count, index=frame.index, dtype="Int64"),
        "nearest_resistance": pd.Series(nearest_resistance, index=frame.index),
        "nearest_support": pd.Series(nearest_support, index=frame.index),
        "resistance_breakout": flags(pd.Series(resistance_breakout, index=frame.index)),
        "support_breakout": flags(pd.Series(support_breakout, index=frame.index)),
        "resistance_retest": flags(pd.Series(resistance_retest, index=frame.index)),
        "support_retest": flags(pd.Series(support_retest, index=frame.index)),
    }, index=frame.index)
