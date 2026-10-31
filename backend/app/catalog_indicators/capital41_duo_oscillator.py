"""Capital41 Duo Oscillator — RSI and normalized Williams %R fused into 0..100.

Source: pine-crypto-collection/capital41_btc_general/Capital41_Duo/Capital41_Duo_Oscillator.pine

MIT License — Copyright (c) 2026 Capital41

Williams %R runs from -100 to 0; shifting it by +100 puts it on the same 0..100
scale as RSI, so the two can be averaged. The fused level is then smoothed with a
plain SMA and compared against zone lines that are themselves the average of the
two component thresholds — `(25 + (-92 + 100)) / 2 = 16.5` and
`(78 + (-8 + 100)) / 2 = 85` at the defaults.

Zone callouts use the same turn logic as the companion Duo indicator: with
`require_turn` on, an extreme only counts once the oscillator has left it on the
previous bar. Bulk crosses of the smoothed line over those zones are reported as
`bull_cross` / `bear_cross`, exactly as upstream's alert conditions.

Discarded presentation: the five individual threshold `plot`s, the midline, the
`bgcolor` of the oscillator line and the four `alertcondition` payloads.

Input must carry `high`, `low` and `close`; no other series is read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import boolean, crossover, crossunder, flags, highest, integer, lowest, number, ohlcv, rsi, sma


def capital41_duo_oscillator(frame: pd.DataFrame, *, rsi_len: int = 14, wr_len: int = 14,
                             rsi_oversold: float = 25.0, rsi_overbought: float = 78.0,
                             wr_oversold: float = -92.0, wr_overbought: float = -8.0,
                             require_turn: bool = True, osc_smooth: int = 3) -> pd.DataFrame:
    integer("rsi_len", rsi_len, 2)
    integer("wr_len", wr_len, 2)
    number("rsi_oversold", rsi_oversold, 1, 50)
    number("rsi_overbought", rsi_overbought, 50, 99)
    number("wr_oversold", wr_oversold, -100, -50)
    number("wr_overbought", wr_overbought, -50, 0)
    boolean("require_turn", require_turn)
    integer("osc_smooth", osc_smooth, 1, 20)
    high, low, close = ohlcv(frame, "high", "low", "close")

    rsi_values = rsi(close, rsi_len)
    window_high, window_low = highest(high, wr_len), lowest(low, wr_len)
    span = window_high - window_low
    williams = (window_high - close) / span.replace(0, np.nan) * -100
    williams = williams.mask(span.eq(0), -50.0)
    normalized = williams + 100.0

    duo_raw = (rsi_values + normalized) / 2.0
    duo_smoothed = sma(duo_raw, osc_smooth)
    oversold_zone = (rsi_oversold + (wr_oversold + 100.0)) / 2.0
    overbought_zone = (rsi_overbought + (wr_overbought + 100.0)) / 2.0

    rsi_turn_up = (rsi_values.shift(1) < rsi_oversold) & (rsi_values > rsi_values.shift(1))
    wr_turn_up = (williams.shift(1) <= wr_oversold) & (williams > williams.shift(1))
    rsi_turn_down = (rsi_values.shift(1) > rsi_overbought) & (rsi_values < rsi_values.shift(1))
    wr_turn_down = (williams.shift(1) > wr_overbought) & (williams < williams.shift(1))

    if require_turn:
        both_oversold = flags(rsi_turn_up & wr_turn_up)
        both_overbought = flags(rsi_turn_down & wr_turn_down)
    else:
        both_oversold = flags((rsi_values < rsi_oversold) & (williams <= wr_oversold))
        both_overbought = flags((rsi_values > rsi_overbought) & (williams > wr_overbought))

    overbought_line = pd.Series(overbought_zone, index=frame.index)
    oversold_line = pd.Series(oversold_zone, index=frame.index)
    return pd.DataFrame({
        "rsi": rsi_values,
        "williams_r": williams,
        "williams_r_normalized": normalized,
        "duo_raw": duo_raw,
        "duo_smoothed": duo_smoothed,
        "oversold_zone": oversold_line,
        "overbought_zone": overbought_line,
        "rsi_turn_up": flags(rsi_turn_up),
        "williams_r_turn_up": flags(wr_turn_up),
        "rsi_turn_down": flags(rsi_turn_down),
        "williams_r_turn_down": flags(wr_turn_down),
        "both_oversold": both_oversold,
        "both_overbought": both_overbought,
        "bull_cross": crossover(duo_smoothed, overbought_line),
        "bear_cross": crossunder(duo_smoothed, oversold_line),
    }, index=frame.index)
