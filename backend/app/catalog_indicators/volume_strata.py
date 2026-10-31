"""Volume Strata — two-sided volume-at-price profile with POC/VA analytics.

Source: pine-collection/indicators/money_flow/volume_strata/volume_strata.pine
(no explicit license header in the source file)

Over the trailing `bbars` window the price range splits into `cnum` rows.
Each bar's volume distributes across the rows its shape touches, weighted by
geometry: the body gets `body_share`, the upper and lower wicks get half of
`2×wick_share` each (the up-column takes the body volume only on green bars,
the down-column only on red bars; wick volume splits evenly). "Money Flow"
mode multiplies each row by the row midpoint. From the row totals the engine
derives the POC (heaviest row), the value area (expand from POC taking the
richer neighbour until `va_pct` of volume), HVN/LVN thresholds at ±1 row
standard deviation, the up/down volume split, POC strength (POC row share of
all volume) and the virgin/touched state of the POC (any of the last 5 bars
spanning it touches it). A POC that moved further than `npoc_min_atr`×ATR
without being retested becomes a naked POC, tracked until price spans it.

Dropped: profile bars, labels, table, naked-POC lines and all colours.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import atr, choice, flags, integer, number, ohlcv


def volume_strata(
        frame: pd.DataFrame, *, bbars: int = 150, cnum: int = 24, va_pct: float = 70.0,
        npoc_min_atr: float = 0.1, src_kind: str = "Volume") -> pd.DataFrame:
    integer("bbars", bbars, 1, 500)
    integer("cnum", cnum, 5, 100)
    number("va_pct", va_pct, 0.0, 100.0)
    number("npoc_min_atr", npoc_min_atr, 0.0)
    choice("src_kind", src_kind, ("Volume", "Money Flow"))
    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    open_ = frame["open"].astype(float)
    rows = len(frame)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    opens = open_.to_numpy(dtype=float)
    volumes = volume.fillna(0.0).to_numpy(dtype=float)
    atr_value = atr(frame, 14)
    atr_values = atr_value.to_numpy(dtype=float)

    volume_is_real = bool(volumes.max(initial=0.0) > 0)
    naked_pocs: list[float] = []
    previous_poc = np.nan

    poc_level = np.full(rows, np.nan)
    vah_level = np.full(rows, np.nan)
    val_level = np.full(rows, np.nan)
    up_pct_out = np.full(rows, np.nan)
    poc_strength_out = np.full(rows, np.nan)
    hvn_count = np.zeros(rows)
    lvn_count = np.zeros(rows)
    virgin_out = np.zeros(rows, dtype=bool)
    naked_alive = np.zeros(rows)

    def overlap(y11: float, y12: float, y21: float, y22: float) -> float:
        return max(min(max(y11, y12), max(y21, y22))
                   - max(min(y11, y12), min(y21, y22)), 0.0)

    for position in range(rows):
        if not volume_is_real:
            continue
        start = max(0, position - bbars + 1)
        window = range(start, position + 1)
        window_high = float(np.max(highs[start:position + 1]))
        window_low = float(np.min(lows[start:position + 1]))
        rng = max(window_high - window_low, 1e-10)
        step = rng / cnum
        levels = [window_low + step * x for x in range(cnum + 1)]

        up_volume = np.zeros(cnum)
        down_volume = np.zeros(cnum)
        for bar in window:
            if np.isnan(volumes[bar]):
                continue
            body_top = max(closes[bar], opens[bar])
            body_bot = min(closes[bar], opens[bar])
            is_green = closes[bar] >= opens[bar]
            top_wick = highs[bar] - body_top
            bottom_wick = body_bot - lows[bar]
            body = body_top - body_bot
            denom = max(2 * top_wick + 2 * bottom_wick + body, 1e-10)
            body_vol = body * volumes[bar] / denom
            top_vol = 2 * top_wick * volumes[bar] / denom
            bottom_vol = 2 * bottom_wick * volumes[bar] / denom
            for x in range(cnum):
                lx, lx1 = levels[x], levels[x + 1]
                row_mid = (lx + lx1) / 2
                weight = row_mid if src_kind == "Money Flow" else 1.0
                body_part = (body_vol if is_green else 0.0) if False else \
                    (body_vol * overlap(lx, lx1, body_bot, body_top) / body
                     if body > 0 else 0.0)
                body_part_down = 0.0 if is_green else \
                    (body_vol * overlap(lx, lx1, body_bot, body_top) / body
                     if body > 0 else 0.0)
                wick_top = top_vol * overlap(lx, lx1, body_top, highs[bar]) / top_wick \
                    if top_wick > 0 else 0.0
                wick_bottom = bottom_vol * overlap(lx, lx1, body_bot, lows[bar]) / bottom_wick \
                    if bottom_wick > 0 else 0.0
                up_volume[x] += (body_part + wick_top / 2 + wick_bottom / 2) * weight
                down_volume[x] += (body_part_down + wick_top / 2 + wick_bottom / 2) * weight

        total_rows = up_volume + down_volume
        total_all = float(total_rows.sum())
        if total_all <= 0:
            continue
        total_up = float(up_volume.sum())
        up_pct_out[position] = total_up / total_all * 100.0
        poc_row = int(np.argmax(total_rows))

        row_mean = float(total_rows.mean())
        row_std = float(total_rows.std())
        hvn_threshold = row_mean + row_std
        lvn_threshold = max(0.0, row_mean - row_std)
        hvn_count[position] = float(np.count_nonzero(total_rows >= hvn_threshold))
        lvn_count[position] = float(np.count_nonzero(total_rows <= lvn_threshold))

        target = total_all * va_pct / 100.0
        va_total = float(total_rows[poc_row])
        va_up = va_down = poc_row
        for _ in range(cnum):
            if va_total >= target:
                break
            upper_vol = float(total_rows[va_up + 1]) if va_up < cnum - 1 else 0.0
            lower_vol = float(total_rows[va_down - 1]) if va_down > 0 else 0.0
            if upper_vol == 0.0 and lower_vol == 0.0:
                break
            if upper_vol >= lower_vol:
                va_total += upper_vol
                va_up += 1
            else:
                va_total += lower_vol
                va_down -= 1

        poc = (levels[poc_row] + levels[poc_row + 1]) / 2
        vah = levels[va_up + 1]
        val = levels[va_down]
        poc_level[position] = poc
        vah_level[position] = vah
        val_level[position] = val
        poc_strength_out[position] = total_rows[poc_row] / total_all * 100.0

        virgin = True
        for back in range(5):
            bar = position - back
            if bar < 0:
                break
            if highs[bar] >= poc and lows[bar] <= poc:
                virgin = False
                break
        virgin_out[position] = virgin

        # Naked POC bookkeeping.
        span_now = highs[position] >= poc and lows[position] <= poc
        if not np.isnan(previous_poc):
            moved = abs(poc - previous_poc) > max(1e-10, npoc_min_atr * atr_values[position]
                                                  if not np.isnan(atr_values[position]) else 1e-10)
            touched_before = highs[position] >= previous_poc and lows[position] <= previous_poc
            if moved and not touched_before:
                if len(naked_pocs) >= 8:
                    naked_pocs.pop(0)
                naked_pocs.append(previous_poc)
        still_naked = []
        for level in naked_pocs:
            if not (highs[position] >= level and lows[position] <= level):
                still_naked.append(level)
        naked_pocs = still_naked
        naked_alive[position] = float(len(naked_pocs))
        previous_poc = poc

    poc_series = pd.Series(poc_level, index=frame.index).ffill()
    vah_series = pd.Series(vah_level, index=frame.index).ffill()
    val_series = pd.Series(val_level, index=frame.index).ffill()
    closes_series = close

    return pd.DataFrame({
        "poc": poc_series,
        "vah": vah_series,
        "val": val_series,
        "up_volume_pct": pd.Series(up_pct_out, index=frame.index),
        "poc_strength": pd.Series(poc_strength_out, index=frame.index),
        "hvn_count": pd.Series(hvn_count, index=frame.index).astype("Int64"),
        "lvn_count": pd.Series(lvn_count, index=frame.index).astype("Int64"),
        "poc_virgin": flags(pd.Series(virgin_out, index=frame.index)),
        "naked_poc_count": pd.Series(naked_alive, index=frame.index).astype("Int64"),
        "va_state": pd.Series(
            np.where(closes_series > vah_series, "Above VA",
                     np.where(closes_series < val_series, "Below VA", "Inside VA")),
            index=frame.index, dtype=object),
        "close_above_poc": flags(closes_series > poc_series),
        "atr": atr_value,
    }, index=frame.index)
