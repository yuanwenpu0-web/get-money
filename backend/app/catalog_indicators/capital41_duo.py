"""Capital41 Duo — RSI and Williams %R on one panel.

Source: https://github.com/Capital41/TradingView-Indicators (Capital41_Duo.pine)

MIT License — Copyright (c) 2026 Capital41

Both oscillators are evaluated against their own thresholds; with turn
confirmation on (the upstream default) a zone only counts after the oscillator
was beyond the threshold on the previous bar and is now ticking back. The banner,
background colour and alert payloads are presentation only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import boolean, flags, highest, integer, lowest, number, ohlcv, rma


def _wilder_rsi(close: pd.Series, length: int) -> pd.Series:
    change = close.diff()
    gain, loss = rma(change.clip(lower=0), length), rma(-change.clip(upper=0), length)
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    return rsi.mask(loss.eq(0) & gain.gt(0), 100.0)


def capital41_duo(frame: pd.DataFrame, *, rsi_len: int = 14, wr_len: int = 14,
                  rsi_oversold: float = 25.0, rsi_overbought: float = 78.0,
                  wr_oversold: float = -92.0, wr_overbought: float = -8.0,
                  require_turn: bool = True, show_background: bool = True,
                  show_badge: bool = True) -> pd.DataFrame:
    integer("rsi_len", rsi_len, 2)
    integer("wr_len", wr_len, 2)
    number("rsi_oversold", rsi_oversold, 1, 50)
    number("rsi_overbought", rsi_overbought, 50, 99)
    number("wr_oversold", wr_oversold, -100, -50)
    number("wr_overbought", wr_overbought, -50, 0)
    boolean("require_turn", require_turn)
    boolean("show_background", show_background)
    boolean("show_badge", show_badge)
    high, low, close = ohlcv(frame, "high", "low", "close")

    rsi = _wilder_rsi(close, rsi_len)
    span = highest(high, wr_len) - lowest(low, wr_len)
    wr = (highest(high, wr_len) - close) / span.replace(0, np.nan) * -100
    wr = wr.mask(span.eq(0), -50.0)

    rsi_up = (rsi.shift(1) < rsi_oversold) & (rsi > rsi.shift(1))
    wr_up = (wr.shift(1) <= wr_oversold) & (wr > wr.shift(1))
    rsi_down = (rsi.shift(1) > rsi_overbought) & (rsi < rsi.shift(1))
    wr_down = (wr.shift(1) > wr_overbought) & (wr < wr.shift(1))

    if require_turn:
        both_oversold = flags(rsi_up & wr_up)
        both_overbought = flags(rsi_down & wr_down)
    else:
        both_oversold = flags((rsi < rsi_oversold) & (wr <= wr_oversold))
        both_overbought = flags((rsi > rsi_overbought) & (wr > wr_overbought))

    return pd.DataFrame({
        "rsi": rsi,
        "williams_r": wr,
        "rsi_turn_up": flags(rsi_up),
        "williams_r_turn_up": flags(wr_up),
        "rsi_turn_down": flags(rsi_down),
        "williams_r_turn_down": flags(wr_down),
        "both_oversold": both_oversold,
        "both_overbought": both_overbought,
    }, index=frame.index)
