"""Volume Profile Plus — volume-at-price histogram with POC and value area.

Source: pine-crypto-collection/btankutt_smc/indicators/02-volume-profile-plus/volume-profile-plus.pine
(btankutt; no explicit license header)

The profile covers either the current session (anchored on a calendar period)
or a fixed bar lookback. Each bar's volume is spread evenly across every price
row its high-low span touches; the row with the most volume is the POC, and
the value area expands row by row from the POC (taking the richer side each
step) until it holds `va_pct` of the total. Rows at/above `hvn_pct` of the
POC volume are high-volume nodes; rows at/below `lvn_pct` are low-volume
nodes.

Unlike the drawing-based upstream (which renders only on the last bar), this
port recomputes the profile on every bar from the trailing window so the
output is a causal, bar-by-bar series.

Dropped: boxes, lines, labels, the value-area shading and every colour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import boolean, choice, flags, integer, number, ohlcv
from ._tf import aligned_index, base_step_minutes


def volume_profile_plus(
        frame: pd.DataFrame, *, calc_mode: str = "Session",
        anchor_rule: str = "D", lookback: int = 200, num_rows: int = 50,
        va_pct: float = 70.0, hvn_pct: float = 80.0, lvn_pct: float = 20.0) -> pd.DataFrame:
    choice("calc_mode", calc_mode, ("Session", "Fixed lookback"))
    if anchor_rule not in ("D", "W", "M"):
        raise ValueError(f"参数 anchor_rule 不支持: {anchor_rule}")
    integer("lookback", lookback, 20, 5000)
    integer("num_rows", num_rows, 10, 200)
    number("va_pct", va_pct, 50.0, 95.0)
    number("hvn_pct", hvn_pct, 50.0, 100.0)
    number("lvn_pct", lvn_pct, 1.0, 50.0)
    boolean("hvn_enabled", True) if False else None
    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    rows_count = len(frame)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    volumes = volume.to_numpy(dtype=float)

    utc = aligned_index(frame, 1)
    if calc_mode == "Session":
        base = base_step_minutes(frame)
        period = utc.tz_localize(None).to_period(anchor_rule)
        expected = [round((item.end_time - item.start_time).total_seconds() / 60 / base)
                    for item in pd.PeriodIndex(utc.tz_localize(None), freq=anchor_rule)]
        # A session's span is bars since its first (completed-period membership):
        # every bar belongs to the session that is still accumulating.
        session_ids = period.astype(str).to_numpy()
    else:
        session_ids = None

    poc = np.full(rows_count, np.nan)
    vah = np.full(rows_count, np.nan)
    val = np.full(rows_count, np.nan)
    hvn_count = np.zeros(rows_count)
    lvn_count = np.zeros(rows_count)
    total_volume = np.full(rows_count, np.nan)

    def compute(start: int, end: int) -> tuple[float, float, float, int, int, float] | None:
        window_hi = float(np.max(highs[start:end + 1]))
        window_lo = float(np.min(lows[start:end + 1]))
        price_range = window_hi - window_lo
        if price_range <= 0:
            return None
        row_h = price_range / num_rows
        bins = np.zeros(num_rows)
        for i in range(start, end + 1):
            lo_row = int(np.floor((lows[i] - window_lo) / row_h))
            hi_row = int(np.floor((highs[i] - window_lo) / row_h))
            lo_row = max(0, min(num_rows - 1, lo_row))
            hi_row = max(0, min(num_rows - 1, hi_row))
            touched = hi_row - lo_row + 1
            share = volumes[i] / touched
            for r in range(lo_row, hi_row + 1):
                bins[r] += share
        total = float(bins.sum())
        if total <= 0:
            return None
        poc_row = int(np.argmax(bins))
        poc_vol = float(bins[poc_row])
        target = total * va_pct / 100.0
        va_vol = poc_vol
        up_row = dn_row = poc_row
        while va_vol < target and (up_row < num_rows - 1 or dn_row > 0):
            up_vol = float(bins[up_row + 1]) if up_row < num_rows - 1 else -1.0
            dn_vol = float(bins[dn_row - 1]) if dn_row > 0 else -1.0
            if up_vol >= dn_vol:
                up_row += 1
                va_vol += up_vol
            else:
                dn_row -= 1
                va_vol += dn_vol
        poc_price = window_lo + (poc_row + 0.5) * row_h
        vah_price = window_lo + (up_row + 1) * row_h
        val_price = window_lo + dn_row * row_h
        nodes_h = int(np.count_nonzero(bins >= poc_vol * hvn_pct / 100.0))
        nodes_l = int(np.count_nonzero(bins <= poc_vol * lvn_pct / 100.0))
        return poc_price, vah_price, val_price, nodes_h, nodes_l, total

    if calc_mode == "Session":
        start = 0
        for position in range(rows_count):
            if position > 0 and session_ids[position] != session_ids[position - 1]:
                start = position
            # Only publish once the session's anchor period has completed, so
            # the profile does not repaint as the session grows.
            bars_into = position - start + 1
            if bars_into < expected[position]:
                continue
            computed = compute(start, position)
            if computed is None:
                continue
            poc[position], vah[position], val[position], hvn_count[position], \
                lvn_count[position], total_volume[position] = computed
    else:
        for position in range(rows_count):
            start = max(0, position - lookback + 1)
            if position - start + 1 < min(lookback, position + 1):
                # Wait until the full window exists so early bars do not flicker.
                if position + 1 < lookback:
                    continue
            computed = compute(start, position)
            if computed is None:
                continue
            poc[position], vah[position], val[position], hvn_count[position], \
                lvn_count[position], total_volume[position] = computed

    poc_series = pd.Series(poc, index=frame.index).ffill()
    vah_series = pd.Series(vah, index=frame.index).ffill()
    val_series = pd.Series(val, index=frame.index).ffill()
    hvn_series = pd.Series(hvn_count, index=frame.index)
    lvn_series = pd.Series(lvn_count, index=frame.index)
    total_series = pd.Series(total_volume, index=frame.index)

    return pd.DataFrame({
        "poc": poc_series,
        "vah": vah_series,
        "val": val_series,
        "hvn_count": hvn_series.astype("Int64"),
        "lvn_count": lvn_series.astype("Int64"),
        "profile_volume": total_series,
        "close_above_poc": flags(close > poc_series),
        "close_in_value_area": flags((close <= vah_series) & (close >= val_series)),
    }, index=frame.index)
