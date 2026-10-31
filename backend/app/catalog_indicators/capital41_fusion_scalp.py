"""Capital41 Fusion Scalp v3 — EMA trend plus oscillator turns, gated by expansion.

Source: pine-crypto-collection/capital41_btc_general/Capital41_Fusion Strategy/Capital41_Fusion_Scalp_v3.pine

MIT License — Copyright (c) 2026 Capital41

Same voting core as the Fusion Oscillator, but the scalp variant only emits discrete
signals: EMA trend, RSI turn (or level), Williams %R turn (or level) and an expansion gate
(ATR above its average times a multiplier, volume above its SMA times a multiplier) must
all agree on the same bar. There is no sweep requirement and no oscillator output.

Dropped: the EMA plot, L/S markers, the five-figure badge, the three background tints and
the bar colouring — every one of them is a pure display toggle.
"""
from __future__ import annotations

import pandas as pd

from ._pine import (atr, boolean, flags, highest, integer, lowest, number, ohlcv, rsi, sma)
from ._pine import ema as pine_ema


def _williams_r_manual(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    window_high, window_low = highest(high, length), lowest(low, length)
    spread = window_high - window_low
    return ((window_high - close) / spread * -100).mask(spread.eq(0), -50.0)


def capital41_fusion_scalp(frame: pd.DataFrame, *, ema_length: int = 50, rsi_length: int = 14,
                           rsi_overbought: float = 65.0, rsi_oversold: float = 35.0,
                           wr_length: int = 14, wr_overbought: float = -20.0,
                           wr_oversold: float = -80.0, require_turn: bool = True,
                           atr_length: int = 14, atr_average_length: int = 20,
                           atr_expansion_multiplier: float = 1.1, volume_length: int = 50,
                           volume_multiplier: float = 1.0) -> pd.DataFrame:
    integer("ema_length", ema_length, 5)
    integer("rsi_length", rsi_length, 2)
    number("rsi_overbought", rsi_overbought, 50.0, 90.0)
    number("rsi_oversold", rsi_oversold, 10.0, 50.0)
    integer("wr_length", wr_length, 2)
    number("wr_overbought", wr_overbought, -50.0, 0.0)
    number("wr_oversold", wr_oversold, -100.0, -50.0)
    boolean("require_turn", require_turn)
    integer("atr_length", atr_length, 1)
    integer("atr_average_length", atr_average_length, 5)
    number("atr_expansion_multiplier", atr_expansion_multiplier, 1.0, 3.0)
    integer("volume_length", volume_length, 5)
    number("volume_multiplier", volume_multiplier, 0.5, 3.0)

    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    trend_ema = pine_ema(close, ema_length)
    trend_long, trend_short = flags(close > trend_ema), flags(close < trend_ema)

    rsi_values = rsi(close, rsi_length)
    rsi_long = flags((rsi_values.shift(1) < rsi_oversold) & (rsi_values > rsi_values.shift(1))) \
        if require_turn else flags(rsi_values < rsi_oversold)
    rsi_short = flags((rsi_values.shift(1) > rsi_overbought) & (rsi_values < rsi_values.shift(1))) \
        if require_turn else flags(rsi_values > rsi_overbought)

    wr = _williams_r_manual(high, low, close, wr_length)
    wr_long = flags((wr.shift(1) < wr_oversold) & (wr > wr.shift(1))) \
        if require_turn else flags(wr < wr_oversold)
    wr_short = flags((wr.shift(1) > wr_overbought) & (wr < wr.shift(1))) \
        if require_turn else flags(wr > wr_overbought)

    average_range = atr(frame, atr_length)
    range_average = sma(average_range, atr_average_length)
    atr_expansion = flags(range_average.gt(0) & (average_range > range_average * atr_expansion_multiplier))
    volume_average = sma(volume, volume_length)
    volume_ok = flags(volume_average.gt(0) & (volume > volume_average * volume_multiplier))
    expansion_ok = atr_expansion & volume_ok

    return pd.DataFrame({
        "ema": trend_ema,
        "rsi": rsi_values,
        "williams_r": wr,
        "atr": average_range,
        "atr_average": range_average,
        "atr_expansion": atr_expansion,
        "volume_ok": volume_ok,
        "expansion_ok": expansion_ok,
        "trend_long": trend_long,
        "trend_short": trend_short,
        "rsi_long": rsi_long,
        "rsi_short": rsi_short,
        "wr_long": wr_long,
        "wr_short": wr_short,
        "long_signal": trend_long & rsi_long & wr_long & expansion_ok,
        "short_signal": trend_short & rsi_short & wr_short & expansion_ok,
    }, index=frame.index)
