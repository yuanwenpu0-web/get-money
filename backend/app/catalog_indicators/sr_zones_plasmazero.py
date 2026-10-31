"""PlasmaZero S/R & Liquidity Zones — pivot zones with dedup, touches and breakout flags.

Source: pine-crypto-collection/plasmazero_sr/main.pine

Mozilla Public License 2.0 — © PlasmaZero

Confirmed pivot highs (left = `pivot_left`, right = `pivot_right`) open a
resistance zone of `zone_width_percent` above the pivot, pivot lows open a
mirrored support zone. A new pivot landing within twice the zone width of an
existing level increments that level's touch counter instead of opening a zone.
Zones created on a volume spike (volume above `volume_threshold_percent` of the
20-bar average, and liquidity detection on) are flagged as liquidity zones. The
level pool keeps at most `max_levels` entries: when the classified box pool
overflows, the oldest level is evicted (Pine shifts the level array together
with the box array). Breakout flags cross the close over the 20-bar high /
under the 20-bar low.

Dropped: the boxes, labels, strength dots and the pivot marker plots; the zone
geometry survives as the newest zone's upper/lower bounds and the nearest level
above/below close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import boolean, crossunder, crossover, flags, highest, integer, lowest, nullable_int, number, ohlcv, pivothigh, pivotlow, sma


def sr_zones_plasmazero(frame: pd.DataFrame, *, pivot_left: int = 5, pivot_right: int = 5,
                        max_levels: int = 5, lookback: int = 100,
                        zone_width_percent: float = 0.5, volume_threshold_percent: int = 150,
                        show_resistance: bool = True, show_support: bool = True,
                        show_liquidity: bool = True) -> pd.DataFrame:
    integer("pivot_left", pivot_left, 1, 20)
    integer("pivot_right", pivot_right, 1, 20)
    integer("max_levels", max_levels, 1, 20)
    integer("lookback", lookback, 20, 500)
    number("zone_width_percent", zone_width_percent, 0.1, 5.0)
    integer("volume_threshold_percent", volume_threshold_percent, 100, 300)
    boolean("show_resistance", show_resistance)
    boolean("show_support", show_support)
    boolean("show_liquidity", show_liquidity)
    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")

    pivot_high = pivothigh(high, pivot_left, pivot_right).to_numpy(dtype=float)
    pivot_low = pivotlow(low, pivot_left, pivot_right).to_numpy(dtype=float)
    average_volume = sma(volume, 20).to_numpy(dtype=float)
    high_volume = (volume.to_numpy(dtype=float)
                   > average_volume * (volume_threshold_percent / 100.0))
    break_resistance = crossover(close, highest(high, 20)).to_numpy(dtype=bool)
    break_support = crossunder(close, lowest(low, 20)).to_numpy(dtype=bool)

    rows = len(frame)
    levels = {"resistance": [], "support": []}   # entries: [price, touches, liquidity]
    box_counts = {"resistance": 0, "liquidity": 0, "support": 0, "support_liq": 0}
    columns = {
        "new_resistance": np.zeros(rows, dtype=bool),
        "new_support": np.zeros(rows, dtype=bool),
        "resistance_touched": np.zeros(rows, dtype=bool),
        "support_touched": np.zeros(rows, dtype=bool),
        "resistance_liquidity": np.zeros(rows, dtype=bool),
        "support_liquidity": np.zeros(rows, dtype=bool),
        "newest_resistance_level": np.full(rows, np.nan),
        "newest_resistance_upper": np.full(rows, np.nan),
        "newest_support_level": np.full(rows, np.nan),
        "newest_support_lower": np.full(rows, np.nan),
        "newest_resistance_touches": np.full(rows, np.nan),
        "newest_support_touches": np.full(rows, np.nan),
        "nearest_resistance_above": np.full(rows, np.nan),
        "nearest_support_below": np.full(rows, np.nan),
        "active_resistance_levels": np.zeros(rows, dtype=float),
        "active_support_levels": np.zeros(rows, dtype=float),
        "break_resistance": break_resistance,
        "break_support": break_support,
    }

    def find_existing(pool, price):
        for entry in pool:
            if abs(price - entry[0]) / entry[0] < zone_width_percent * 2 / 100.0:
                return entry
        return None

    for bar in range(rows):
        confirmed = bar - pivot_right
        within_lookback = pivot_right <= lookback
        if show_resistance and not np.isnan(pivot_high[bar]) and within_lookback:
            existing = find_existing(levels["resistance"], pivot_high[bar])
            if existing is not None:
                existing[1] += 1
                columns["resistance_touched"][bar] = True
            else:
                is_liquidity = bool(high_volume[bar]) and show_liquidity
                levels["resistance"].append([pivot_high[bar], 1, is_liquidity])
                columns["new_resistance"][bar] = True
                columns["resistance_liquidity"][bar] = is_liquidity
                box_counts["liquidity" if is_liquidity else "resistance"] += 1
                if box_counts["liquidity" if is_liquidity else "resistance"] > max_levels:
                    box_counts["liquidity" if is_liquidity else "resistance"] -= 1
                    levels["resistance"].pop(0)
        if show_support and not np.isnan(pivot_low[bar]) and within_lookback:
            existing = find_existing(levels["support"], pivot_low[bar])
            if existing is not None:
                existing[1] += 1
                columns["support_touched"][bar] = True
            else:
                is_liquidity = bool(high_volume[bar]) and show_liquidity
                levels["support"].append([pivot_low[bar], 1, is_liquidity])
                columns["new_support"][bar] = True
                columns["support_liquidity"][bar] = is_liquidity
                box_counts["support_liq" if is_liquidity else "support"] += 1
                if box_counts["support_liq" if is_liquidity else "support"] > max_levels:
                    box_counts["support_liq" if is_liquidity else "support"] -= 1
                    levels["support"].pop(0)

        columns["active_resistance_levels"][bar] = len(levels["resistance"])
        columns["active_support_levels"][bar] = len(levels["support"])
        if levels["resistance"]:
            newest = levels["resistance"][-1]
            columns["newest_resistance_level"][bar] = newest[0]
            columns["newest_resistance_upper"][bar] = newest[0] * (1 + zone_width_percent / 100.0)
            columns["newest_resistance_touches"][bar] = newest[1]
        if levels["support"]:
            newest = levels["support"][-1]
            columns["newest_support_level"][bar] = newest[0]
            columns["newest_support_lower"][bar] = newest[0] * (1 - zone_width_percent / 100.0)
            columns["newest_support_touches"][bar] = newest[1]
        above = [entry[0] for entry in levels["resistance"] if entry[0] > close.iloc[bar]]
        below = [entry[0] for entry in levels["support"] if entry[0] < close.iloc[bar]]
        if above:
            columns["nearest_resistance_above"][bar] = min(above)
        if below:
            columns["nearest_support_below"][bar] = max(below)

    columns = {name: (flags(pd.Series(values, index=frame.index))
                      if values.dtype == bool else pd.Series(values, index=frame.index))
               for name, values in columns.items()}
    for name in ("active_resistance_levels", "active_support_levels",
                 "newest_resistance_touches", "newest_support_touches"):
        columns[name] = nullable_int(columns[name])
    return pd.DataFrame(columns, index=frame.index)
