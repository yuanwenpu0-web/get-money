"""Capital41 Session VWAP — intraday VWAP that restarts with every session.

Source: pine-crypto-collection/capital41_btc_general/Capital41_Session_VWAP/Capital41_Session_VWAP_v2.pine

MIT License — Copyright (c) 2026 Capital41

VWAP is accumulated by hand from `hlc3 * volume` inside the session window and
reset on the first in-session bar of a new session. The band is the running
standard deviation of the same session, taken from the volume-weighted second
moment (`cumP2V / cumV - vwap^2`) rather than from a window, and the four
crosses of price over VWAP and the bands are exposed as flags.

The session window is a time-of-day range in a timezone: `session_tz` mirrors the
five zones upstream offers and `session_times` the "HHMM-HHMM" session input. With
`use_exchange_tz` the bar timestamps' own timezone is used, which is what Pine
does when `time()` is called without a timezone argument. A bar belongs to the
session when its opening time falls inside the window, so a bar at the closing
edge (16:00 for "0930-1600") is outside.

Accumulators are only reset on a session start, so `vwap` and its bands keep the
last session's closing values until the next session opens, exactly as the raw
series upstream keeps them behind its `show_only_session` plot toggle.

Discarded presentation: the VWAP/band `plot` toggles, the 3x1 status badge and its
distance readout, and the four `alertcondition` messages.

Input must carry `high`, `low`, `close` and `volume` on a timezone-aware bar-open
index; VWAP is undefined without traded volume, so `volume` is a required column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import boolean, choice, crossover, crossunder, number, ohlcv

_SESSION_ZONES = ("America/New_York", "America/Chicago", "Etc/UTC", "Europe/London", "Asia/Tokyo")


def _session_bounds(text: str) -> tuple[int, int]:
    """Parse Pine's "HHMM-HHMM" session input into minutes of the day."""
    span = text.split(":")[0]
    parts = span.split("-")
    if len(parts) != 2:
        raise ValueError("参数 session_times 必须形如 HHMM-HHMM")

    def parse(token: str) -> int:
        if len(token) != 4 or not token.isdigit():
            raise ValueError("参数 session_times 必须形如 HHMM-HHMM")
        hours, minutes = int(token[:2]), int(token[2:])
        if hours > 23 or minutes > 59:
            raise ValueError("参数 session_times 必须形如 HHMM-HHMM")
        return hours * 60 + minutes

    return parse(parts[0]), parse(parts[1])


def _utc_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    index = frame.index
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is None or index.hasnans
            or not index.is_monotonic_increasing or not index.is_unique):
        raise ValueError("索引必须为有时区、递增且唯一的 K 线开盘时间")
    return index.tz_convert("UTC")


def _in_session(utc: pd.DatetimeIndex, zone, start: int, end: int) -> np.ndarray:
    """Pine's `not na(time(...))`: does the bar open inside the session window?"""
    local = utc.tz_convert(zone)
    minutes = local.hour * 60 + local.minute
    if start <= end:
        return (minutes >= start) & (minutes < end)
    return (minutes >= start) | (minutes < end)


def capital41_session_vwap(frame: pd.DataFrame, *, use_exchange_tz: bool = False,
                           session_tz: str = "America/New_York",
                           session_times: str = "0930-1600",
                           band_mult: float = 1.0) -> pd.DataFrame:
    boolean("use_exchange_tz", use_exchange_tz)
    choice("session_tz", session_tz, _SESSION_ZONES)
    number("band_mult", band_mult, 0.1, 3.0)
    start, end = _session_bounds(session_times)
    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")

    utc = _utc_index(frame)
    zone = frame.index.tz if use_exchange_tz else session_tz
    inside = _in_session(utc, zone, start, end)
    session_start = inside & ~np.concatenate(([False], inside[:-1]))

    typical = (high + low + close) / 3.0
    typical_values = typical.to_numpy(dtype=float)
    volume_values = volume.to_numpy(dtype=float)
    count = len(frame)
    vwap = np.full(count, np.nan)
    deviation = np.full(count, np.nan)
    cum_pv = cum_v = cum_p2v = 0.0
    for position in range(count):
        if session_start[position]:
            cum_pv = cum_v = cum_p2v = 0.0
        if inside[position]:
            price, size = typical_values[position], volume_values[position]
            cum_pv += price * size
            cum_v += size
            cum_p2v += price * price * size
        if cum_v > 0:
            vwap[position] = cum_pv / cum_v
            variance = cum_p2v / cum_v - vwap[position] * vwap[position]
            deviation[position] = np.sqrt(variance) if variance > 0 else 0.0

    vwap_series = pd.Series(vwap, index=frame.index)
    band = deviation * band_mult
    upper = vwap_series + band
    lower = vwap_series - band
    return pd.DataFrame({
        "in_session": pd.Series(inside, index=frame.index),
        "session_start": pd.Series(session_start, index=frame.index),
        "vwap": vwap_series,
        "vwap_upper": upper,
        "vwap_lower": lower,
        "vwap_cross_up": crossover(close, vwap_series),
        "vwap_cross_down": crossunder(close, vwap_series),
        "upper_band_cross_up": crossover(close, upper),
        "lower_band_cross_down": crossunder(close, lower),
    }, index=frame.index)
