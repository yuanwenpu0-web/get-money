"""Capital41 Fusion Oscillator v3 — four-vote long/short oscillator on a 0-100 scale.

Source: pine-crypto-collection/capital41_btc_general/Capital41_Fusion Strategy/Capital41_Fusion_Oscillator_v3.pine

MIT License — Copyright (c) 2026 Capital41

Every bar casts up to four votes per side: EMA trend, RSI turn (or level), Williams %R
turn (or level), and an expansion vote that needs ATR above its own average times a
multiplier *and* volume above its SMA times a multiplier. `scoreLong - scoreShort`
runs from -4 to +4 and is rescaled to 0-100, then smoothed.

Dropped: 25/50/75 zone lines, background highlighting, the colour ramp and alerts.
"""
from __future__ import annotations

import pandas as pd

from ._pine import (atr, boolean, crossover, crossunder, ema, flags, highest, integer,
                    lowest, number, ohlcv, rsi, sma)


def _williams_r_manual(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    window_high, window_low = highest(high, length), lowest(low, length)
    spread = window_high - window_low
    return ((window_high - close) / spread * -100).mask(spread.eq(0), -50.0)


def capital41_fusion_oscillator(frame: pd.DataFrame, *, ema_length: int = 50, rsi_length: int = 14,
                                rsi_overbought: float = 65.0, rsi_oversold: float = 35.0,
                                wr_length: int = 14, wr_overbought: float = -20.0,
                                wr_oversold: float = -80.0, require_turn: bool = True,
                                atr_length: int = 14, atr_average_length: int = 20,
                                atr_expansion_multiplier: float = 1.1, volume_length: int = 50,
                                volume_multiplier: float = 1.0, oscillator_smoothing: int = 3) -> pd.DataFrame:
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
    integer("oscillator_smoothing", oscillator_smoothing, 1, 20)

    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    fast_ema = ema(close, ema_length)
    trend_long = flags(close > fast_ema)
    trend_short = flags(close < fast_ema)

    rsi_values = rsi(close, rsi_length)
    previous_rsi = rsi_values.shift(1)
    rsi_turn_up = flags((previous_rsi < rsi_oversold) & (rsi_values > previous_rsi))
    rsi_turn_down = flags((previous_rsi > rsi_overbought) & (rsi_values < previous_rsi))
    rsi_long = rsi_turn_up if require_turn else flags(rsi_values < rsi_oversold)
    rsi_short = rsi_turn_down if require_turn else flags(rsi_values > rsi_overbought)

    wr = _williams_r_manual(high, low, close, wr_length)
    previous_wr = wr.shift(1)
    wr_turn_up = flags((previous_wr < wr_oversold) & (wr > previous_wr))
    wr_turn_down = flags((previous_wr > wr_overbought) & (wr < previous_wr))
    wr_long = wr_turn_up if require_turn else flags(wr < wr_oversold)
    wr_short = wr_turn_down if require_turn else flags(wr > wr_overbought)

    average_range = atr(frame, atr_length)
    average_of_range = sma(average_range, atr_average_length)
    atr_expansion = flags(average_of_range.gt(0) & (average_range > average_of_range * atr_expansion_multiplier))
    volume_average = sma(volume, volume_length)
    volume_ok = flags(volume_average.gt(0) & (volume > volume_average * volume_multiplier))
    expansion_ok = atr_expansion & volume_ok

    long_score = (trend_long.astype(int) + rsi_long.astype(int)
                  + wr_long.astype(int) + (expansion_ok & trend_long).astype(int))
    short_score = (trend_short.astype(int) + rsi_short.astype(int)
                   + wr_short.astype(int) + (expansion_ok & trend_short).astype(int))
    raw = long_score - short_score
    fusion = (raw + 4) * 12.5
    smoothed = sma(fusion, oscillator_smoothing)

    return pd.DataFrame({
        "ema": fast_ema,
        "rsi": rsi_values,
        "williams_r": wr,
        "atr": average_range,
        "atr_average": average_of_range,
        "atr_expansion": atr_expansion,
        "volume_ok": volume_ok,
        "expansion_ok": expansion_ok,
        "trend_long": trend_long,
        "trend_short": trend_short,
        "rsi_long": rsi_long,
        "rsi_short": rsi_short,
        "wr_long": wr_long,
        "wr_short": wr_short,
        "score_long": long_score,
        "score_short": short_score,
        "fusion_raw": raw,
        "fusion": fusion,
        "fusion_smoothed": smoothed,
        "upper_zone_cross": crossover(smoothed, pd.Series(75.0, index=frame.index)),
        "lower_zone_cross": crossunder(smoothed, pd.Series(25.0, index=frame.index)),
        "midline_cross_up": crossover(smoothed, pd.Series(50.0, index=frame.index)),
        "midline_cross_down": crossunder(smoothed, pd.Series(50.0, index=frame.index)),
    }, index=frame.index)
