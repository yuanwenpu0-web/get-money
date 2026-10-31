"""Stock Stats — ADR%, ATR, 52-week high, volume buzz and volume run rate.

Source: pine-crypto-collection/peterdmv_stockstats/indicators/StockStats.pine

Mozilla Public License 2.0 — © peterdmv

Every stat in the original is computed on the daily (or weekly) timeframe and
pulled onto the chart with `request.security(..., lookahead_off)`; here the same
aggregations run on `calendar_frame` and are published as the previous
*completed* period via `lift_previous_calendar`, so a replay never sees the
developing day's final numbers.

The volume run rate projects the current day's full-session volume from what has
already traded. The crypto branch (the one this port implements) assumes a flat
intraday distribution and compares against the average of the previous eight
daily bars at a fixed 7-bar offset; the real-time clock `timenow` is replaced by
each bar's closing time, which keeps the projection strictly causal.

Dropped: market cap and shares float (they need `syminfo.shares_outstanding_*`,
which no OHLCV feed carries), the equity-session run-rate model (needs the ET
session calendar and the real-time clock), the chart-timeframe ATR switch (both
daily and weekly ATR are reported instead) and the stat table with all its
colour and formatting helpers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import atr, highest, integer, ohlcv, sma
from ._tf import aligned_index, base_step_minutes, calendar_frame, lift_previous_calendar

_SAME_DOW_WEEKS = 8


def _same_dow_average(starts: pd.DatetimeIndex, daily_volume: np.ndarray,
                      day_start: pd.Timestamp) -> float:
    """Mean of the previous 8 daily bars at a fixed 7-bar offset (Pine f_same_dow_avg)."""
    position = int(starts.searchsorted(day_start, side="right"))
    current = position - 1 if position and starts[position - 1] == day_start else position
    if current < 7 * _SAME_DOW_WEEKS:
        return float("nan")
    window = [daily_volume[current - 7 * week] for week in range(1, _SAME_DOW_WEEKS + 1)]
    if any(np.isnan(value) for value in window):
        return float("nan")
    return float(sum(window) / _SAME_DOW_WEEKS)


def stock_stats(frame: pd.DataFrame, *, adr_length: int = 20, atr_length: int = 14) -> pd.DataFrame:
    integer("adr_length", adr_length, 1)
    integer("atr_length", atr_length, 1)
    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    utc = aligned_index(frame, 1)
    step_minutes = base_step_minutes(frame)

    daily = calendar_frame(frame, "D")
    adr_daily = 100.0 * (sma(daily["high"] / daily["low"], adr_length) - 1.0)
    atr_daily = atr(daily, atr_length)
    low_of_day = 100.0 * (daily["close"] - daily["low"]) / atr_daily.where(atr_daily != 0)
    volume_buzz_daily = (daily["volume"] / sma(daily["volume"], 100) - 1.0) * 100.0

    weekly = calendar_frame(frame, "W")
    high_52_daily = highest(weekly["high"], 52)

    day = utc.floor("D")
    volume_today = volume.groupby(day).cumsum()
    closing_clock = utc + pd.Timedelta(minutes=step_minutes)
    fraction_elapsed = (closing_clock.hour * 60 + closing_clock.minute) / 1440.0
    starts = daily.index
    daily_volume = daily["volume"].to_numpy(dtype=float)
    same_dow = pd.Series(
        [_same_dow_average(starts, daily_volume, moment) for moment in day],
        index=frame.index, dtype=float)
    run_rate = ((volume_today / fraction_elapsed.where(fraction_elapsed > 0))
                / same_dow.replace(0, np.nan) - 1.0) * 100.0

    return pd.DataFrame({
        "adr_percent": lift_previous_calendar(frame, adr_daily, "D"),
        "atr_daily": lift_previous_calendar(frame, atr_daily, "D"),
        "atr_weekly": lift_previous_calendar(frame, atr(weekly, atr_length), "W"),
        "low_of_day_distance": lift_previous_calendar(frame, low_of_day, "D"),
        "high_52_week": lift_previous_calendar(frame, high_52_daily, "W"),
        "volume_buzz": lift_previous_calendar(frame, volume_buzz_daily, "D"),
        "volume_today": volume_today,
        "run_rate_percent": run_rate,
    }, index=frame.index)
