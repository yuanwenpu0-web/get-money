"""Current Day POC / VAH / VAL — running intraday volume profile of the trading day.

Source: pine-crypto-collection/amadeo987/SK_Current_Day_POC_VAH_VAL.pine

The script stores every intraday bar of the current profile day, spreads each bar's volume
across `rows` price buckets, and reports the point of control plus the value-area edges that
together hold `value_area_percent` of the day's volume. Upstream only draws the profile on
the last bar; here every bar gets the profile **as it stood at that bar's close**, which is
the causal reading a backtest needs.

Session handling: the day rolls at UTC midnight (upstream default `0000-2359:1234567`). Pass
`session_start` / `session_end` as `HHMM` to restrict the accumulation window; a bar counts
when its open time falls inside the window, and the profile still resets at midnight.

Dropped: the POC/VAH/VAL lines and labels, the "intraday chart required" warning label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import integer, number, ohlcv
from ._tf import aligned_index


def _parse_clock(text: str, name: str) -> int:
    if not isinstance(text, str) or len(text) != 4 or not text.isdigit():
        raise ValueError(f"参数 {name} 必须是 HHMM 格式")
    hours, minutes = int(text[:2]), int(text[2:])
    if not (0 <= hours <= 23 and 0 <= minutes <= 59):
        raise ValueError(f"参数 {name} 超出允许范围")
    return hours * 60 + minutes


def _profile(highs: np.ndarray, lows: np.ndarray, volumes: np.ndarray, rows: int,
             value_area_percent: float) -> tuple[float, float, float, float, float]:
    """Return (poc, vah, val, profile_low, profile_high) for one day's stored bars."""
    profile_low, profile_high = float(lows.min()), float(highs.max())
    price_range = profile_high - profile_low
    if price_range <= 0:
        return (np.nan,) * 5
    row_height = price_range / rows
    row_volumes = np.zeros(rows)
    spans = highs - lows
    for index in range(len(volumes)):
        bar_volume = float(volumes[index])
        if spans[index] > 0:
            first_row = int(np.clip(np.floor((lows[index] - profile_low) / row_height), 0, rows - 1))
            last_row = int(np.clip(np.floor((highs[index] - profile_low) / row_height), 0, rows - 1))
            for row in range(first_row, last_row + 1):
                bottom = profile_low + row * row_height
                overlap = max(0.0, min(highs[index], bottom + row_height) - max(lows[index], bottom))
                if overlap > 0:
                    row_volumes[row] += bar_volume * overlap / spans[index]
        else:
            row = int(np.clip(np.floor((lows[index] - profile_low) / row_height), 0, rows - 1))
            row_volumes[row] += bar_volume

    total_volume = float(volumes.sum())
    if total_volume <= 0:
        return (np.nan,) * 5
    poc_index = int(np.argmax(row_volumes))
    low_index = high_index = poc_index
    included = row_volumes[poc_index]
    target = total_volume * value_area_percent / 100.0
    while included < target and (low_index > 0 or high_index < rows - 1):
        above = row_volumes[high_index + 1] if high_index < rows - 1 else -1.0
        below = row_volumes[low_index - 1] if low_index > 0 else -1.0
        if above >= below:
            high_index += 1
            included += max(above, 0.0)
        else:
            low_index -= 1
            included += max(below, 0.0)
    return (profile_low + (poc_index + 0.5) * row_height,
            profile_low + (high_index + 1.0) * row_height,
            profile_low + low_index * row_height, profile_low, profile_high)


def ad_current_day_poc_vah_val(frame: pd.DataFrame, *, rows: int = 48,
                               value_area_percent: float = 70.0, max_stored_bars: int = 2000,
                               session_start: str = "0000", session_end: str = "2359") -> pd.DataFrame:
    integer("rows", rows, 24, 100)
    number("value_area_percent", value_area_percent, 50.0, 99.0)
    integer("max_stored_bars", max_stored_bars, 250, 5000)
    start, end = _parse_clock(session_start, "session_start"), _parse_clock(session_end, "session_end")
    if end <= start:
        raise ValueError("参数 session_end 必须晚于 session_start")

    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    utc = aligned_index(frame, 1)
    days = utc.normalize()
    minutes = utc.hour * 60 + utc.minute

    poc = np.full(len(frame), np.nan)
    vah = np.full(len(frame), np.nan)
    val = np.full(len(frame), np.nan)
    kept = np.zeros(len(frame), dtype=int)
    lows: list[float] = []
    highs: list[float] = []
    volumes: list[float] = []

    for position in range(len(frame)):
        if position == 0 or days[position] != days[position - 1]:
            lows, highs, volumes = [], [], []
        if start <= minutes[position] < end:
            lows.append(float(low.iloc[position]))
            highs.append(float(high.iloc[position]))
            volumes.append(float(volume.iloc[position]))
            if len(lows) > max_stored_bars:
                lows.pop(0)
                highs.pop(0)
                volumes.pop(0)
        kept[position] = len(lows)
        if lows:
            poc[position], vah[position], val[position], _, _ = _profile(
                np.asarray(highs), np.asarray(lows), np.asarray(volumes), rows, value_area_percent)

    return pd.DataFrame({"poc": poc, "vah": vah, "val": val, "bars_in_profile": kept},
                        index=frame.index)
