"""Capital41 Divergence Detector v2 — RSI + Williams %R pivot divergences.

Source: pine-crypto-collection/capital41_btc_general/Capital41_Divergence_Detector/Capital41_Divergence_Detector_v2.pine

MIT License — Copyright (c) 2026 Capital41

Confirmed pivot highs/lows (left = right = `pivot_length`) update a last/previous
pair of swing records. On the bar a pivot is confirmed, the two most recent highs
(and lows) are compared: price making a higher high while RSI or Williams %R made
a lower one is a regular bearish divergence, the mirror case at the lows is
regular bullish, and the inverted comparisons are the hidden variants. A
"double" fires when both oscillators agree on the same bar. The distance between
the two pivot bars must be at most `max_pivot_bars`.

Dropped: the divergence lines, labels, the status badge table and the alert
payloads. The last-divergence state is kept as three string columns instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import boolean, integer, ohlcv, pivothigh, pivotlow, rsi


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    span = high.rolling(length, min_periods=length).max() - low.rolling(length, min_periods=length).min()
    wr = (high.rolling(length, min_periods=length).max() - close) / span.where(span != 0) * -100.0
    return wr.mask(span.eq(0), -50.0)


def capital41_divergence_detector(frame: pd.DataFrame, *, pivot_length: int = 5,
                                  rsi_length: int = 14, wr_length: int = 14,
                                  max_pivot_bars: int = 100, show_regular: bool = True,
                                  show_hidden: bool = True, use_rsi: bool = True,
                                  use_wr: bool = True) -> pd.DataFrame:
    integer("pivot_length", pivot_length, 2, 20)
    integer("rsi_length", rsi_length, 2)
    integer("wr_length", wr_length, 2)
    integer("max_pivot_bars", max_pivot_bars, 10, 500)
    boolean("show_regular", show_regular)
    boolean("show_hidden", show_hidden)
    boolean("use_rsi", use_rsi)
    boolean("use_wr", use_wr)
    high, low, close = ohlcv(frame, "high", "low", "close")

    rsi_series = rsi(close, rsi_length)
    wr_series = _williams_r(high, low, close, wr_length)
    rsi_values = rsi_series.to_numpy(dtype=float)
    wr_values = wr_series.to_numpy(dtype=float)
    high_values = high.to_numpy(dtype=float)
    low_values = low.to_numpy(dtype=float)
    pivot_high = pivothigh(high, pivot_length, pivot_length).to_numpy(dtype=float)
    pivot_low = pivotlow(low, pivot_length, pivot_length).to_numpy(dtype=float)

    rows = len(frame)
    columns = {name: np.zeros(rows, dtype=bool) for name in (
        "bear_reg_rsi", "bear_reg_wr", "bull_reg_rsi", "bull_reg_wr",
        "bear_hidden_rsi", "bear_hidden_wr", "bull_hidden_rsi", "bull_hidden_wr",
        "bullish_regular", "bearish_regular", "bullish_hidden", "bearish_hidden",
        "bullish_double", "bearish_double")}
    last_dir = np.full(rows, "---", dtype=object)
    last_type = np.full(rows, "---", dtype=object)
    last_osc = np.full(rows, "---", dtype=object)

    # Swing records: (price, rsi, wr, pivot bar) for the last two highs and lows.
    last_high = prev_high = (np.nan, np.nan, np.nan, -1)
    last_low = prev_low = (np.nan, np.nan, np.nan, -1)
    for bar in range(rows):
        if not np.isnan(pivot_high[bar]):
            prev_high = last_high
            pivot_bar = bar - pivot_length
            last_high = (pivot_high[bar], rsi_values[pivot_bar], wr_values[pivot_bar], pivot_bar)
        if not np.isnan(pivot_low[bar]):
            prev_low = last_low
            pivot_bar = bar - pivot_length
            last_low = (pivot_low[bar], rsi_values[pivot_bar], wr_values[pivot_bar], pivot_bar)

        high_gap = last_high[3] - prev_high[3]
        low_gap = last_low[3] - prev_low[3]
        high_ok = 0 < high_gap <= max_pivot_bars
        low_ok = 0 < low_gap <= max_pivot_bars
        confirmed_high = not np.isnan(pivot_high[bar]) and high_ok
        confirmed_low = not np.isnan(pivot_low[bar]) and low_ok

        bear_reg_rsi = (show_regular and use_rsi and confirmed_high
                        and last_high[0] > prev_high[0] and last_high[1] < prev_high[1])
        bear_reg_wr = (show_regular and use_wr and confirmed_high
                       and last_high[0] > prev_high[0] and last_high[2] < prev_high[2])
        bull_reg_rsi = (show_regular and use_rsi and confirmed_low
                        and last_low[0] < prev_low[0] and last_low[1] > prev_low[1])
        bull_reg_wr = (show_regular and use_wr and confirmed_low
                       and last_low[0] < prev_low[0] and last_low[2] > prev_low[2])
        bear_hidden_rsi = (show_hidden and use_rsi and confirmed_high
                           and last_high[0] < prev_high[0] and last_high[1] > prev_high[1])
        bear_hidden_wr = (show_hidden and use_wr and confirmed_high
                          and last_high[0] < prev_high[0] and last_high[2] > prev_high[2])
        bull_hidden_rsi = (show_hidden and use_rsi and confirmed_low
                           and last_low[0] > prev_low[0] and last_low[1] < prev_low[1])
        bull_hidden_wr = (show_hidden and use_wr and confirmed_low
                          and last_low[0] > prev_low[0] and last_low[2] < prev_low[2])

        # Comparisons against a warm-up NaN pivot are false in numpy, which
        # matches Pine where `na` comparisons never fire.
        for name, value in (("bear_reg_rsi", bear_reg_rsi), ("bear_reg_wr", bear_reg_wr),
                            ("bull_reg_rsi", bull_reg_rsi), ("bull_reg_wr", bull_reg_wr),
                            ("bear_hidden_rsi", bear_hidden_rsi),
                            ("bear_hidden_wr", bear_hidden_wr),
                            ("bull_hidden_rsi", bull_hidden_rsi),
                            ("bull_hidden_wr", bull_hidden_wr)):
            columns[name][bar] = bool(value)
        columns["bullish_regular"][bar] = columns["bull_reg_rsi"][bar] or columns["bull_reg_wr"][bar]
        columns["bearish_regular"][bar] = columns["bear_reg_rsi"][bar] or columns["bear_reg_wr"][bar]
        columns["bullish_hidden"][bar] = columns["bull_hidden_rsi"][bar] or columns["bull_hidden_wr"][bar]
        columns["bearish_hidden"][bar] = columns["bear_hidden_rsi"][bar] or columns["bear_hidden_wr"][bar]
        columns["bullish_double"][bar] = columns["bull_reg_rsi"][bar] and columns["bull_reg_wr"][bar]
        columns["bearish_double"][bar] = columns["bear_reg_rsi"][bar] and columns["bear_reg_wr"][bar]

        if columns["bullish_double"][bar]:
            last_dir[bar], last_type[bar], last_osc[bar] = "BULL", "REG", "RSI+WR"
        elif columns["bearish_double"][bar]:
            last_dir[bar], last_type[bar], last_osc[bar] = "BEAR", "REG", "RSI+WR"
        elif columns["bullish_regular"][bar]:
            last_dir[bar], last_type[bar] = "BULL", "REG"
            last_osc[bar] = "RSI" if columns["bull_reg_rsi"][bar] else "WR"
        elif columns["bearish_regular"][bar]:
            last_dir[bar], last_type[bar] = "BEAR", "REG"
            last_osc[bar] = "RSI" if columns["bear_reg_rsi"][bar] else "WR"
        elif columns["bullish_hidden"][bar]:
            last_dir[bar], last_type[bar] = "BULL", "HID"
            last_osc[bar] = "RSI" if columns["bull_hidden_rsi"][bar] else "WR"
        elif columns["bearish_hidden"][bar]:
            last_dir[bar], last_type[bar] = "BEAR", "HID"
            last_osc[bar] = "RSI" if columns["bear_hidden_rsi"][bar] else "WR"
        else:
            last_dir[bar], last_type[bar], last_osc[bar] = (last_dir[bar - 1] if bar else "---",
                                                            last_type[bar - 1] if bar else "---",
                                                            last_osc[bar - 1] if bar else "---")

    return pd.DataFrame({
        "rsi": rsi_series,
        "williams_r": wr_series,
        "pivot_high": pivothigh(high, pivot_length, pivot_length),
        "pivot_low": pivotlow(low, pivot_length, pivot_length),
        **columns,
        "last_divergence_direction": pd.Series(last_dir, index=frame.index),
        "last_divergence_type": pd.Series(last_type, index=frame.index),
        "last_divergence_oscillator": pd.Series(last_osc, index=frame.index),
    }, index=frame.index)
