"""Capital41 PDH/PDL Levels v3 — previous day, week and month highs and lows.

Source: pine-crypto-collection/capital41_btc_general/Capital41_PDH_PDL/Capital41_PDH_PDL_v3.pine

MIT License — Copyright (c) 2026 Capital41

`request.security(syminfo.tickerid, "D", high[1])` is the previous *completed* calendar
period's high, which `_tf.calendar_frame` + `_tf.lift_previous_calendar` reproduce exactly
for D / W / M. Midpoints are the average of the two levels.

Every bar also reports whether price is touching a level (within `touch_tolerance` percent of
close) and whether it closed through it (a close beyond the level after a close on the other
side). Week and month levels stay NaN until the first complete period is available.

Dropped: the level plots and fills, the right-edge labels and the position badge.
"""
from __future__ import annotations

import pandas as pd

from ._pine import flags, number, ohlcv
from ._tf import calendar_frame, lift_previous_calendar

_RULES = (("day", "D"), ("week", "W"), ("month", "M"))


def capital41_pdh_pdl(frame: pd.DataFrame, *, touch_tolerance: float = 0.1) -> pd.DataFrame:
    number("touch_tolerance", touch_tolerance, 0.01, 1.0)
    _, _, close = ohlcv(frame, "high", "low", "close")

    columns: dict[str, pd.Series] = {}
    levels: dict[str, pd.Series] = {}
    for prefix, rule in _RULES:
        aggregated = calendar_frame(frame, rule)
        previous_high = lift_previous_calendar(frame, aggregated["high"], rule)
        previous_low = lift_previous_calendar(frame, aggregated["low"], rule)
        columns[f"previous_{prefix}_high"] = previous_high
        columns[f"previous_{prefix}_low"] = previous_low
        columns[f"previous_{prefix}_mid"] = (previous_high + previous_low) / 2.0
        levels[prefix] = previous_high, previous_low

    tolerance = close * touch_tolerance / 100.0
    for prefix, (previous_high, previous_low) in levels.items():
        columns[f"touch_{prefix}_high"] = flags((close - previous_high).abs() <= tolerance)
        columns[f"touch_{prefix}_low"] = flags((close - previous_low).abs() <= tolerance)
        columns[f"break_{prefix}_high"] = flags((close > previous_high) & (close.shift(1) <= previous_high))
        columns[f"break_{prefix}_low"] = flags((close < previous_low) & (close.shift(1) >= previous_low))

    previous_high, previous_low = levels["day"]
    columns["distance_to_day_high"] = ((close - previous_high) / previous_high.where(previous_high.ne(0)) * 100.0)
    columns["distance_to_day_low"] = ((close - previous_low) / previous_low.where(previous_low.ne(0)) * 100.0)
    columns["position"] = pd.Series("IN RANGE", index=frame.index, dtype=object).mask(
        close > previous_high, "ABOVE").mask(close < previous_low, "BELOW")
    return pd.DataFrame(columns, index=frame.index)
