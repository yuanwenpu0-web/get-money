"""Capital41 Confluence Core — moving average, RSI, Williams %R and a sweep must agree.

Source: pine-crypto-collection/capital41_btc_general/Capital41_Confluence_Core/Capital41_Confluence_Core.pine

MIT License — Copyright (c) 2026 Capital41

A bar only signals when four independent reads line up: price is on the trend side of
the MA, the oscillator is turning out of an extreme (or simply sits in the zone, for the
stricter modes), Williams %R agrees, and the bar swept the prior `lookback` extreme
before closing back inside on above-average volume.

`signal_mode` widens or narrows the oscillator test exactly as upstream: Strict needs a
turn out of the extreme, Standard accepts "already oversold" as well, Loose only needs
the buffered zone. `zone_buffer` is applied in oscillator points for RSI and in the same
points for W%R, matching the script.

Dropped: the MA plot, the L/S markers and the five-cell status badge. `showSignals`
only hid the markers, so it has no port.
"""
from __future__ import annotations

import pandas as pd

from ._pine import (choice, highest, integer, lowest, ma, number, ohlcv, rsi, sma)


def _williams_r_manual(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    window_high, window_low = highest(high, length), lowest(low, length)
    spread = window_high - window_low
    return ((window_high - close) / spread * -100).mask(spread.eq(0), -50.0)


def capital41_confluence_core(frame: pd.DataFrame, *, ma_type: str = "EMA", ma_length: int = 50,
                              rsi_length: int = 14, rsi_overbought: float = 65.0,
                              rsi_oversold: float = 35.0, wr_length: int = 14,
                              wr_overbought: float = -20.0, wr_oversold: float = -80.0,
                              sweep_lookback: int = 20, volume_lookback: int = 50,
                              min_volume_ratio: float = 1.2, signal_mode: str = "Standard",
                              zone_buffer: int = 5) -> pd.DataFrame:
    choice("ma_type", ma_type, ("EMA", "SMA"))
    integer("ma_length", ma_length, 5)
    integer("rsi_length", rsi_length, 2)
    number("rsi_overbought", rsi_overbought, 50.0, 80.0)
    number("rsi_oversold", rsi_oversold, 20.0, 50.0)
    integer("wr_length", wr_length, 2)
    number("wr_overbought", wr_overbought, -100.0, 0.0)
    number("wr_oversold", wr_oversold, -100.0, 0.0)
    integer("sweep_lookback", sweep_lookback, 5)
    integer("volume_lookback", volume_lookback, 5)
    number("min_volume_ratio", min_volume_ratio, 1.0, 3.0)
    choice("signal_mode", signal_mode, ("Strict", "Standard", "Loose"))
    integer("zone_buffer", zone_buffer, 0, 20)

    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    average = ma(close, ma_length, ma_type)

    rsi_values = rsi(close, rsi_length)
    rsi_loose_oversold, rsi_loose_overbought = rsi_oversold + zone_buffer, rsi_overbought - zone_buffer
    previous_rsi = rsi_values.shift(1)
    rsi_long = {"Strict": (rsi_values < rsi_oversold) & (rsi_values > previous_rsi),
                "Standard": (rsi_values < rsi_oversold)
                | ((rsi_values < rsi_loose_oversold) & (rsi_values > previous_rsi)),
                "Loose": rsi_values < rsi_loose_oversold}[signal_mode]
    rsi_short = {"Strict": (rsi_values > rsi_overbought) & (rsi_values < previous_rsi),
                 "Standard": (rsi_values > rsi_overbought)
                 | ((rsi_values > rsi_loose_overbought) & (rsi_values < previous_rsi)),
                 "Loose": rsi_values > rsi_loose_overbought}[signal_mode]

    wr = _williams_r_manual(high, low, close, wr_length)
    wr_loose_oversold, wr_loose_overbought = wr_oversold + zone_buffer, wr_overbought - zone_buffer
    previous_wr = wr.shift(1)
    wr_long = {"Strict": (wr < wr_oversold) & (wr > previous_wr),
               "Standard": (wr < wr_oversold) | ((wr < wr_loose_oversold) & (wr > previous_wr)),
               "Loose": wr < wr_loose_oversold}[signal_mode]
    wr_short = {"Strict": (wr > wr_overbought) & (wr < previous_wr),
                "Standard": (wr > wr_overbought) | ((wr > wr_loose_overbought) & (wr < previous_wr)),
                "Loose": wr > wr_loose_overbought}[signal_mode]

    prior_high = highest(high.shift(1), sweep_lookback)
    prior_low = lowest(low.shift(1), sweep_lookback)
    volume_average = sma(volume, volume_lookback)
    volume_ratio = (volume / volume_average.where(volume_average > 0)).where(volume_average > 0, 0.0)
    threshold = min_volume_ratio * {"Strict": 1.2, "Standard": 1.0, "Loose": 0.9}[signal_mode]
    high_volume = volume_ratio.ge(threshold)

    bull_sweep = (low < prior_low) & (close >= prior_low) & high_volume
    bear_sweep = (high > prior_high) & (close <= prior_high) & high_volume
    trend_long, trend_short = close > average, close < average

    return pd.DataFrame({
        "ma": average,
        "rsi": rsi_values,
        "williams_r": wr,
        "prior_high": prior_high,
        "prior_low": prior_low,
        "volume_ratio": volume_ratio,
        "bull_sweep": bull_sweep.fillna(False),
        "bear_sweep": bear_sweep.fillna(False),
        "rsi_long": rsi_long.fillna(False),
        "rsi_short": rsi_short.fillna(False),
        "wr_long": wr_long.fillna(False),
        "wr_short": wr_short.fillna(False),
        "trend_long": trend_long.fillna(False),
        "trend_short": trend_short.fillna(False),
        "long_signal": (trend_long & bull_sweep & rsi_long & wr_long).fillna(False),
        "short_signal": (trend_short & bear_sweep & rsi_short & wr_short).fillna(False),
    }, index=frame.index)
