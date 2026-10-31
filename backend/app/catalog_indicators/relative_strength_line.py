"""Relative Strength Line — price versus a benchmark, Mansfield form or raw ratio.

Source: pine-collection/indicators/relative_strength/relative_strength_line/relative_strength_line.pine
(WavesUnchained, MIT)

`request.security(benchmark, close)` becomes a required `bench_close` column, so the caller
decides which instrument (SPY, QQQ, BTC…) the series is measured against. Raw RS is
`close / bench_close`; Mansfield RS subtracts its own SMA and scales to percent, so zero
means "in line with the benchmark".

A long regime needs both an absolute uptrend (price above its EMA) and positive relative
strength; short is the mirror. Regime *entries* are then throttled by `min_gap` bars, which
reproduces the script's `lastUpBar` / `lastDownBar` bookkeeping.

Dropped: the RS plot, the confluence markers, the last-bar tag and the dashboard.
"""
from __future__ import annotations

import pandas as pd

from ._pine import choice, crossover, crossunder, integer, flags, number, ohlcv, sma
from ._pine import ema as pine_ema


def relative_strength_line(frame: pd.DataFrame, *, display_mode: str = "Mansfield RS",
                           mansfield_length: int = 50, trend_length: int = 50,
                           min_gap: int = 5) -> pd.DataFrame:
    choice("display_mode", display_mode, ("Mansfield RS", "Raw RS Line"))
    integer("mansfield_length", mansfield_length, 5)
    integer("trend_length", trend_length, 5)
    integer("min_gap", min_gap, 0, 1000)

    close, bench = ohlcv(frame, "close", "bench_close")
    rs = (close / bench.where(bench.ne(0))).where(bench.notna() & bench.ne(0))
    rs_average = sma(rs, mansfield_length)
    mansfield = ((rs - rs_average) / rs_average.where(rs_average.ne(0)) * 100.0).where(
        rs_average.notna() & rs_average.ne(0))

    price_ema = pine_ema(close, trend_length)
    price_bull, price_bear = flags(close > price_ema), flags(close < price_ema)
    rs_bull = flags(mansfield.notna() & (mansfield > 0))
    rs_bear = flags(mansfield.notna() & (mansfield < 0))

    long_regime = price_bull & rs_bull
    short_regime = price_bear & rs_bear
    long_entry = long_regime & ~long_regime.shift(1, fill_value=False)
    short_entry = short_regime & ~short_regime.shift(1, fill_value=False)

    fired_up = _throttle(long_entry, min_gap)
    fired_down = _throttle(short_entry, min_gap)

    return pd.DataFrame({
        "rs": rs,
        "rs_ma": rs_average,
        "mansfield_rs": mansfield,
        "price_ema": price_ema,
        "price_bull": price_bull,
        "price_bear": price_bear,
        "rs_bull": rs_bull,
        "rs_bear": rs_bear,
        "long_regime": long_regime,
        "short_regime": short_regime,
        "long_entry": long_entry,
        "short_entry": short_entry,
        "fire_up": fired_up,
        "fire_down": fired_down,
        "rs_cross_up": crossover(mansfield, pd.Series(0.0, index=frame.index)),
        "rs_cross_down": crossunder(mansfield, pd.Series(0.0, index=frame.index)),
    }, index=frame.index)


def _throttle(entries: pd.Series, min_gap: int) -> pd.Series:
    """Allow an entry only when the previous fired entry is at least `min_gap` bars back."""
    out = []
    last = None
    for position, entry in enumerate(entries.to_numpy(dtype=bool)):
        if entry and (last is None or position - last >= min_gap):
            out.append(True)
            last = position
        else:
            out.append(False)
    return pd.Series(out, index=entries.index, dtype=bool)
