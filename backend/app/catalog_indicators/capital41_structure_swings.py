"""Capital41 Structure Swings — pivots, carried levels and break of structure.

Source: https://github.com/Capital41/TradingView-Indicators
        (Capital41_Structure_Swings_v3.pine)

MIT License — Copyright (c) 2026 Capital41

The swing values surface on the bar that confirms them (a pivot needs
`pivot_len` bars on each side, so it appears `pivot_len` bars later). The most
recent swing high/low is carried forward as a standing level, and a BOS fires on
the first close beyond that level. Labels, lines and alert payloads are
presentation only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import boolean, integer, ohlcv, pivotlow, pivothigh


def capital41_structure_swings(frame: pd.DataFrame, *, pivot_len: int = 5,
                               show_swings: bool = True, show_bos: bool = True,
                               show_levels: bool = True) -> pd.DataFrame:
    integer("pivot_len", pivot_len, 2, 20)
    boolean("show_swings", show_swings)
    boolean("show_bos", show_bos)
    boolean("show_levels", show_levels)
    high, low, close = ohlcv(frame, "high", "low", "close")

    swing_high = pivothigh(high, pivot_len, pivot_len)
    swing_low = pivotlow(low, pivot_len, pivot_len)
    highs, lows, closes = swing_high.to_numpy(), swing_low.to_numpy(), close.to_numpy()

    count = len(frame)
    last_high, last_low = np.full(count, np.nan), np.full(count, np.nan)
    bos_up, bos_down = np.zeros(count, dtype=bool), np.zeros(count, dtype=bool)
    standing_high = standing_low = np.nan
    for position in range(count):
        # Upstream assigns the new level before it evaluates the break on the same bar.
        if not np.isnan(highs[position]):
            standing_high = highs[position]
        if not np.isnan(lows[position]):
            standing_low = lows[position]
        last_high[position], last_low[position] = standing_high, standing_low
        if position == 0:
            continue
        previous_close = closes[position - 1]
        if not (np.isnan(standing_high) or np.isnan(previous_close)):
            bos_up[position] = closes[position] > standing_high and previous_close <= standing_high
        if not (np.isnan(standing_low) or np.isnan(previous_close)):
            bos_down[position] = closes[position] < standing_low and previous_close >= standing_low

    return pd.DataFrame({
        "swing_high": swing_high,
        "swing_low": swing_low,
        "last_swing_high": last_high,
        "last_swing_low": last_low,
        "bos_up": bos_up,
        "bos_down": bos_down,
    }, index=frame.index)
