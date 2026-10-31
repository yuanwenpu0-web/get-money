"""Candle Pressure Index — where the close sits in the range, weighted by body and volume.

Source: pine-collection/indicators/money_flow/candle_pressure_index/candle_pressure_index.pine
(WavesUnchained, MIT)

Three ingredients per bar: the close location inside the high-low range mapped to -1..+1,
the body's share of that range, and the bar's volume ranked against the highest volume of
the lookback. Their product is the raw pressure, smoothed by an EMA into the CPI, and the
bar-over-bar change is the acceleration term.

Upstream suppresses the whole reading when the feed's volume is tick-based or never
positive (`volumeIsReal`). Here a frame whose volume column is entirely zero is treated the
same way: the volume rank becomes NaN and everything downstream stays NaN.

Dropped: the CPI columns plot, spike markers, the dashboard table and the "volume not real"
warning table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import ema, flags, highest, integer, number, ohlcv


def candle_pressure_index(frame: pd.DataFrame, *, smoothing_length: int = 20,
                          volume_lookback: int = 20, spike_threshold: float = 0.6) -> pd.DataFrame:
    integer("smoothing_length", smoothing_length, 2, 100)
    integer("volume_lookback", volume_lookback, 5, 100)
    number("spike_threshold", spike_threshold, 0.2, 1.0)

    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    (opening,) = ohlcv(frame, "open")
    has_volume = bool(volume.gt(0).any())
    vol = volume.where(volume.gt(0)) if has_volume else pd.Series(np.nan, index=frame.index)

    spread = high - low
    positive_spread = spread.where(spread > 0)
    close_location = ((close - low) / positive_spread).where(spread > 0, 0.5)
    body_ratio = ((close - opening).abs() / positive_spread).where(spread > 0, 0.0)

    volume_max = highest(vol, volume_lookback)
    volume_rank = (vol / volume_max.where(volume_max > 0)).where(volume_max > 0, 0.5)
    if not has_volume:
        volume_rank = pd.Series(np.nan, index=frame.index)  # Pine gates the whole reading
    raw = (2.0 * close_location - 1.0) * body_ratio * volume_rank
    cpi = ema(raw, smoothing_length)
    acceleration = cpi - cpi.shift(1)

    return pd.DataFrame({
        "close_location": close_location,
        "body_ratio": body_ratio,
        "volume_rank": volume_rank,
        "raw_pressure": raw,
        "cpi": cpi,
        "cpi_acceleration": acceleration,
        "spike_bull": flags(raw >= spike_threshold),
        "spike_bear": flags(raw <= -spike_threshold),
    }, index=frame.index)
