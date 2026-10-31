"""CVD Bias — cumulative signed-volume proxy with normalisation and pivots divergences.

Source: pine-collection/indicators/money_flow/cvd_bias/cvd_bias.pine (WavesUnchained, MIT)

True cumulative delta needs bid/ask trades, which an OHLCV bar does not carry, so the script
builds a proxy: the close's position inside the range maps to -1..+1 and multiplies the bar
volume. The proxy is summed over the accumulation window, divided by the largest absolute
reading of the normalisation lookback to land on -1..+1, and differentiated over `roc_length`
for the acceleration term.

Divergences compare each confirmed swing with the one before it: price making a higher high
while the bias makes a lower one is bearish, and the mirror case is bullish.

Upstream gates everything on a real (non-tick) volume feed; a frame whose volume is entirely
zero is treated the same way and stays NaN.

Dropped: the CVD columns plot, the dashboard, the "volume not real" warning and alerts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import boolean, flags, highest, integer, ohlcv, pivotlow, pivothigh, sum_, valuewhen


def cvd_bias(frame: pd.DataFrame, *, accumulation_window: int = 50, normalization_lookback: int = 200,
             roc_length: int = 5, detect_divergence: bool = True, pivot_left: int = 5,
             pivot_right: int = 5) -> pd.DataFrame:
    integer("accumulation_window", accumulation_window, 10, 500)
    integer("normalization_lookback", normalization_lookback, 20, 1000)
    integer("roc_length", roc_length, 1, 30)
    boolean("detect_divergence", detect_divergence)
    integer("pivot_left", pivot_left, 1)
    integer("pivot_right", pivot_right, 1)

    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    has_volume = bool(volume.gt(0).any())
    vol = volume.where(volume.gt(0)) if has_volume else pd.Series(np.nan, index=frame.index)

    spread = high - low
    close_location = ((close - low) / spread.where(spread > 0)).where(spread > 0, 0.5)
    signed_volume = (2.0 * close_location - 1.0) * vol

    cvd_raw = sum_(signed_volume, accumulation_window)
    amplitude = highest(cvd_raw.abs(), normalization_lookback)
    cvd = (cvd_raw / amplitude.where(amplitude > 0)).where(amplitude > 0, 0.0)
    if not has_volume:
        cvd = pd.Series(np.nan, index=frame.index)
    cvd_roc = cvd - cvd.shift(roc_length).fillna(0.0)

    swing_high = pivothigh(high, pivot_left, pivot_right)
    swing_low = pivotlow(low, pivot_left, pivot_right)
    previous_high = valuewhen(swing_high.notna(), high.shift(pivot_right), 1)
    previous_low = valuewhen(swing_low.notna(), low.shift(pivot_right), 1)
    previous_cvd_high = valuewhen(swing_high.notna(), cvd.shift(pivot_right), 1)
    previous_cvd_low = valuewhen(swing_low.notna(), cvd.shift(pivot_right), 1)

    bearish_raw = (swing_high.notna() & previous_high.notna()
                   & (high.shift(pivot_right) > previous_high)
                   & (cvd.shift(pivot_right) < previous_cvd_high))
    bullish_raw = (swing_low.notna() & previous_low.notna()
                   & (low.shift(pivot_right) < previous_low)
                   & (cvd.shift(pivot_right) > previous_cvd_low))

    return pd.DataFrame({
        "close_location": close_location,
        "signed_volume": signed_volume,
        "cvd_raw": cvd_raw,
        "cvd": cvd,
        "cvd_amplitude": amplitude,
        "cvd_roc": cvd_roc,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "bearish_divergence": flags(bearish_raw) & detect_divergence,
        "bullish_divergence": flags(bullish_raw) & detect_divergence,
    }, index=frame.index)
