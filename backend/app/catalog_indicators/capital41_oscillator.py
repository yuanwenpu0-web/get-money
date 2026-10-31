"""C41 Oscillator — a 0-100 companion to Capital41's core strategy.

Source: https://github.com/Capital41/TradingView-Indicators
        (Capital41_Core_Strategy_Oscillator.pine)

MIT License — Copyright (c) 2026 Capital41

Three votes (price above/below the MA, RSI zone, Williams %R zone) are summed
into a -3..3 score and rescaled to 0..100, then smoothed. With turn confirmation
on (the upstream default) a vote only counts once the oscillator has left its
extremity. Zone lines, colours and alert payloads are presentation only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (boolean, choice, crossover, crossunder, ema, flags, highest, integer,
                    lowest, number, ohlcv, rma, sma)


def capital41_oscillator(frame: pd.DataFrame, *, ma_type: str = "EMA", ma_len: int = 50,
                         rsi_len: int = 14, rsi_overbought: float = 65.0,
                         rsi_oversold: float = 35.0, wr_len: int = 14,
                         wr_overbought: float = -20.0, wr_oversold: float = -80.0,
                         require_turn: bool = True, osc_smooth: int = 3,
                         upper_zone: float = 66.7, mid_zone: float = 50.0,
                         lower_zone: float = 33.3, show_zones: bool = True,
                         show_background: bool = True) -> pd.DataFrame:
    choice("ma_type", ma_type, ("EMA", "SMA"))
    integer("ma_len", ma_len, 5)
    integer("rsi_len", rsi_len, 2)
    number("rsi_overbought", rsi_overbought, 50, 90)
    number("rsi_oversold", rsi_oversold, 10, 50)
    integer("wr_len", wr_len, 2)
    number("wr_overbought", wr_overbought, -100, 0)
    number("wr_oversold", wr_oversold, -100, 0)
    boolean("require_turn", require_turn)
    integer("osc_smooth", osc_smooth, 1, 20)
    number("upper_zone", upper_zone, 50, 100)
    number("mid_zone", mid_zone, 0, 100)
    number("lower_zone", lower_zone, 0, 50)
    boolean("show_zones", show_zones)
    boolean("show_background", show_background)
    high, low, close = ohlcv(frame, "high", "low", "close")

    ma = ema(close, ma_len) if ma_type == "EMA" else sma(close, ma_len)
    trend_long, trend_short = flags(close > ma), flags(close < ma)

    change = close.diff()
    gain, loss = rma(change.clip(lower=0), rsi_len), rma(-change.clip(upper=0), rsi_len)
    rsi = (100 - 100 / (1 + gain / loss.replace(0, np.nan)))
    rsi = rsi.mask(loss.eq(0) & gain.gt(0), 100.0)

    span = highest(high, wr_len) - lowest(low, wr_len)
    wr = (highest(high, wr_len) - close) / span.replace(0, np.nan) * -100
    wr = wr.mask(span.eq(0), -50.0)

    rsi_long = ((rsi.shift(1) < rsi_oversold) & (rsi > rsi.shift(1)) if require_turn
                else (rsi < rsi_oversold))
    rsi_short = ((rsi.shift(1) > rsi_overbought) & (rsi < rsi.shift(1)) if require_turn
                 else (rsi > rsi_overbought))
    wr_long = ((wr.shift(1) < wr_oversold) & (wr > wr.shift(1)) if require_turn
               else (wr < wr_oversold))
    wr_short = ((wr.shift(1) > wr_overbought) & (wr < wr.shift(1)) if require_turn
                else (wr > wr_overbought))
    rsi_long, rsi_short = flags(rsi_long), flags(rsi_short)
    wr_long, wr_short = flags(wr_long), flags(wr_short)

    score_long = trend_long.astype(int) + rsi_long.astype(int) + wr_long.astype(int)
    score_short = trend_short.astype(int) + rsi_short.astype(int) + wr_short.astype(int)
    raw = score_long - score_short
    oscillator = (raw + 3.0) * (100.0 / 6.0)
    smoothed = sma(oscillator, osc_smooth)

    return pd.DataFrame({
        "ma": ma,
        "rsi": rsi,
        "williams_r": wr,
        "trend_long": trend_long,
        "trend_short": trend_short,
        "score_long": score_long,
        "score_short": score_short,
        "oscillator_raw": raw,
        "oscillator": oscillator,
        "oscillator_smoothed": smoothed,
        "long_setup": flags(trend_long & rsi_long & wr_long),
        "short_setup": flags(trend_short & rsi_short & wr_short),
        "bull_cross": crossover(smoothed, pd.Series(upper_zone, index=frame.index)),
        "bear_cross": crossunder(smoothed, pd.Series(lower_zone, index=frame.index)),
    }, index=frame.index)
