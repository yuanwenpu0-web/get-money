"""Capital41 Liquidity Sweep Simple — stop-hunt wick that closes back inside range.

Source: https://github.com/Capital41/TradingView-Indicators
        (Capital41_Liquidity_Sweep_Simple.pine)

MIT License — Copyright (c) 2026 Capital41

A sweep needs three things on the same closed bar: a wick beyond the prior
`lookback`-bar extreme, a close back inside that extreme, and above-average
volume. Consecutive sweeps at a similar level (within 0.5% of price, at most
`max_bars_between` bars apart) are counted so callers can prefer the repeats.
Markers and alert payloads are presentation only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import boolean, flags, highest, integer, lowest, number, ohlcv, sma


def capital41_liquidity_sweep(frame: pd.DataFrame, *, lookback: int = 20,
                              volume_sma_len: int = 50, min_volume_ratio: float = 1.2,
                              track_consecutive: bool = True,
                              max_bars_between: int = 10) -> pd.DataFrame:
    integer("lookback", lookback, 2)
    integer("volume_sma_len", volume_sma_len, 5)
    number("min_volume_ratio", min_volume_ratio, 1, 3)
    boolean("track_consecutive", track_consecutive)
    integer("max_bars_between", max_bars_between, 1, 100)
    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")

    prior_high = highest(high.shift(1), lookback)
    prior_low = lowest(low.shift(1), lookback)
    volume_sma = sma(volume, volume_sma_len)
    volume_ratio = volume / volume_sma.where(volume_sma > 0, np.nan)
    volume_ratio = volume_ratio.mask(volume_sma <= 0, 0.0)
    high_volume = flags(volume_ratio >= min_volume_ratio)

    bull_sweep = flags((low < prior_low) & (close >= prior_low) & high_volume)
    bear_sweep = flags((high > prior_high) & (close <= prior_high) & high_volume)

    count = len(frame)
    consecutive_bull = np.zeros(count, dtype=int)
    consecutive_bear = np.zeros(count, dtype=int)
    bull_flags, bear_flags = bull_sweep.to_numpy(), bear_sweep.to_numpy()
    highs, lows, closes = prior_high.to_numpy(), prior_low.to_numpy(), close.to_numpy()
    running_bull = running_bear = 0
    last_bull_level = last_bear_level = np.nan
    last_bull_bar = last_bear_bar = -1
    for position in range(count):
        tolerance = closes[position] * 0.005 if not np.isnan(closes[position]) else np.nan
        if bull_flags[position]:
            same_level = (not np.isnan(last_bull_level)
                          and abs(lows[position] - last_bull_level) <= tolerance)
            within_bars = last_bull_bar >= 0 and position - last_bull_bar <= max_bars_between
            running_bull = running_bull + 1 if (same_level and within_bars) else 1
            last_bull_level, last_bull_bar = lows[position], position
        if bear_flags[position]:
            same_level = (not np.isnan(last_bear_level)
                          and abs(highs[position] - last_bear_level) <= tolerance)
            within_bars = last_bear_bar >= 0 and position - last_bear_bar <= max_bars_between
            running_bear = running_bear + 1 if (same_level and within_bars) else 1
            last_bear_level, last_bear_bar = highs[position], position
        consecutive_bull[position] = running_bull
        consecutive_bear[position] = running_bear

    high_prob_bull = bull_sweep & pd.Series(consecutive_bull >= 2, index=frame.index)
    high_prob_bear = bear_sweep & pd.Series(consecutive_bear >= 2, index=frame.index)
    return pd.DataFrame({
        "prior_high": prior_high,
        "prior_low": prior_low,
        "volume_ratio": volume_ratio,
        "high_volume": high_volume,
        "bull_sweep": bull_sweep,
        "bear_sweep": bear_sweep,
        "consecutive_bull": consecutive_bull,
        "consecutive_bear": consecutive_bear,
        "high_prob_bull": flags(high_prob_bull),
        "high_prob_bear": flags(high_prob_bear),
    }, index=frame.index)
