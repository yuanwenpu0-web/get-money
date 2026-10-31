"""MTF Structure Bias — higher-low / higher-high structure across four timeframes.

Source: pine-collection/indicators/market_structure/mtf_structure_bias/mtf_structure_bias.pine
(WavesUnchained, MIT)

Structure per timeframe: compare the `length`-bar highest high and lowest low with the
same reading `length` bars earlier. Both rising → 1 (bull), both falling → -1 (bear),
otherwise 0 (mixed). The chart timeframe counts half, the three higher ones count
1.0 / 1.5 / 2.0, and the weighted total is scaled to ±100.

`request.security(..., lookahead_off)` is mapped to the previous *completed* higher
timeframe period (see `_tf.lift_series`), so the score never repaints.

Dropped: the dashboard table, its colour helpers and alert dialogs.
"""
from __future__ import annotations

import pandas as pd

from ._pine import flags, highest, integer, lowest, ohlcv, crossover, crossunder
from ._tf import base_step_minutes, htf_frame, lift_series

_WEIGHTS = (0.5, 1.0, 1.5, 2.0)


def _structure(high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    window_high, window_low = highest(high, length), lowest(low, length)
    high_up = window_high > window_high.shift(length)
    low_up = window_low > window_low.shift(length)
    values = pd.Series(0, index=high.index, dtype=float)
    values = values.mask(high_up & low_up, 1.0).mask(~high_up & ~low_up, -1.0)
    defined = window_high.notna() & window_high.shift(length).notna()
    return values.where(defined)


def mtf_structure_bias(frame: pd.DataFrame, *, lookback: int = 20, htf1_minutes: int = 60,
                       htf2_minutes: int = 240, htf3_minutes: int = 1440) -> pd.DataFrame:
    integer("lookback", lookback, 5, 100)
    for name, value in (("htf1_minutes", htf1_minutes), ("htf2_minutes", htf2_minutes),
                        ("htf3_minutes", htf3_minutes)):
        integer(name, value, base_step_minutes(frame), 10080)
    high, low, _ = ohlcv(frame, "high", "low", "close")

    base = _structure(high, low, lookback)
    higher = []
    for minutes in (htf1_minutes, htf2_minutes, htf3_minutes):
        aggregated = htf_frame(frame, minutes)
        structure = _structure(aggregated["high"], aggregated["low"], lookback)
        higher.append(lift_series(frame, structure, minutes))

    score = (base * _WEIGHTS[0] + higher[0] * _WEIGHTS[1]
             + higher[1] * _WEIGHTS[2] + higher[2] * _WEIGHTS[3]) / 5.0 * 100.0
    return pd.DataFrame({
        "structure_base": base,
        "structure_htf1": higher[0],
        "structure_htf2": higher[1],
        "structure_htf3": higher[2],
        "score": score,
        "bull_zone_entry": crossover(score, pd.Series(60.0, index=frame.index)),
        "bear_zone_entry": crossunder(score, pd.Series(-60.0, index=frame.index)),
    }, index=frame.index)
