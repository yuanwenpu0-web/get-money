"""Bullish Lines — four-state classification of four SMMA (RMA) lines.

Source: pine-crypto-collection/vitopod_four_state/bullish-lines-four-state.pine (Bullish Lines — Four-State Classification, MIT)

Four Wilder averages of the same source (HL2 by default) are compared pairwise. When the
fast line is on the same side of both mid lines and also above the slow line the state is
bullish; mirror-image readings give bearish. Any disagreement between the two pair tests
is "mixed", and the mixed state is split in two by whether the fast line still leads the
slow one (yellow) or not (blue).

All four lengths must be configured (>= 2) for a state to exist — upstream defaults them
to 0 and plots nothing, so the columns stay NaN / "Unconfigured" until lengths are given.

Dropped: the four plots, the fill between fastest and slowest, the background tint and the
"set the lengths" notice label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import choice, edge, flags, integer, ohlcv, rma

_STATES = ("Bullish", "Bearish", "Neutral Yellow", "Neutral Blue", "Unconfigured")


def bullish_lines_four_state(frame: pd.DataFrame, *, fast_length: int = 0, mid_warmer_length: int = 0,
                             mid_cooler_length: int = 0, slow_length: int = 0,
                             source: str = "HL2") -> pd.DataFrame:
    for name, value in (("fast_length", fast_length), ("mid_warmer_length", mid_warmer_length),
                        ("mid_cooler_length", mid_cooler_length), ("slow_length", slow_length)):
        integer(name, value, 0)
    choice("source", source, ("HL2", "CLOSE"))
    high, low, close = ohlcv(frame, "high", "low", "close")
    values = (high + low) / 2 if source == "HL2" else close

    configured = all(length >= 2 for length in
                     (fast_length, mid_warmer_length, mid_cooler_length, slow_length))
    if configured:
        fast = rma(values, fast_length)
        warmer = rma(values, mid_warmer_length)
        cooler = rma(values, mid_cooler_length)
        slow = rma(values, slow_length)
    else:
        empty = pd.Series(np.nan, index=frame.index)
        fast = warmer = cooler = slow = empty

    fast_below_warmer = fast < warmer
    fast_below_slow = fast < slow
    cooler_below_slow = cooler < slow
    mixed = ((fast_below_warmer != fast_below_slow) | (cooler_below_slow != fast_below_slow))
    lead = fast > slow

    bullish = ~mixed & lead
    bearish = ~mixed & ~lead
    neutral_yellow = mixed & lead
    neutral_blue = mixed & ~lead
    defined = fast.notna() & slow.notna() & warmer.notna() & cooler.notna()

    state = np.select([bullish, bearish, neutral_yellow, neutral_blue], list(_STATES[:4]),
                      default="Unconfigured")
    state = pd.Series(state, index=frame.index, dtype=object).where(defined, "Unconfigured")

    def _entered(flag: pd.Series) -> pd.Series:
        return (defined & edge(flag)).fillna(False)

    return pd.DataFrame({
        "fast_smma": fast,
        "mid_warmer_smma": warmer,
        "mid_cooler_smma": cooler,
        "slow_smma": slow,
        "mixed": flags(mixed),
        "is_bullish": bullish.fillna(False),
        "is_bearish": bearish.fillna(False),
        "is_neutral_yellow": neutral_yellow.fillna(False),
        "is_neutral_blue": neutral_blue.fillna(False),
        "state": state,
        "new_bullish": _entered(bullish),
        "new_bearish": _entered(bearish),
        "new_neutral_yellow": _entered(neutral_yellow),
        "new_neutral_blue": _entered(neutral_blue),
    }, index=frame.index)
