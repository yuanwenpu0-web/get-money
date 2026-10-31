"""ZigZag Core [WavesUnchained]  v2.2.2  —  catalog port.

Source: data/pine-collection/indicators/market_structure/zigzag_core/
        zigzag_core.pine

Configurable ZigZag engine:
  * Pivot depth (bars each side) + minimum pivot spacing (backstep) +
    alternation, with Percent / ATR / Hybrid reversal thresholds measured
    against the ATR at the reference pivot bar.
  * Confirmed pivots stay stable; same-side confirmed replacement updates the
    endpoint; the active endpoint tracks live extension.
  * Optional recursive Level-2 ZigZag built from confirmed Level-1 pivots and
    an optional independent Secondary ZigZag (own depth/spacing/deviation).

Ported core; multi-timeframe projection, drawing/labels/diagnostics and the
developing-leg rendering are dropped. All three engines run on the chart frame.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import (
    atr, boolean, choice, integer, number, ohlcv, pivothigh_strict, pivotlow_strict,
)


def _required_move(ref_price, ref_atr, mode, dev_pct, atr_mult, mult=1.0):
    pct = abs(ref_price) * dev_pct / 100.0 * mult
    atr_move = ref_atr * atr_mult * mult
    if mode == "Percent":
        return pct
    if mode == "ATR":
        return atr_move
    return max(pct, atr_move)


def _run_zigzag(events, n, backstep, mode, dev_pct, atr_mult, mult=1.0):
    last_type = 0
    last_price = float("nan"); last_bar = None; last_atr = float("nan")
    active_price = float("nan"); active_bar = None
    seed_type = 0; seed_price = float("nan"); seed_bar = None; seed_atr = float("nan")
    out_type = np.zeros(n, dtype=np.int64)
    out_last = np.full(n, np.nan)
    out_active = np.full(n, np.nan)
    new_pivot = np.zeros(n, dtype=np.int64)
    structure = np.zeros(n, dtype=np.int64)
    confirmed = []  # (type, bar, price)
    prev_high = float("nan"); prev_low = float("nan")
    structure_now = 0

    for i in range(n):
        evs = events.get(i)
        if evs:
            types = [e[0] for e in evs]
            if 1 in types and -1 in types:
                expected = -1 if last_type == 1 else 1
                evs = [e for e in evs if e[0] == expected]
            for (t, price, atr_val) in evs:
                if last_type == 0:
                    if seed_type == 0:
                        seed_type, seed_price, seed_bar, seed_atr = t, price, i, atr_val
                    elif seed_type == t:
                        if (t == 1 and price > seed_price) or (t == -1 and price < seed_price):
                            seed_price, seed_bar, seed_atr = price, i, atr_val
                    else:
                        req = _required_move(seed_price, seed_atr, mode, dev_pct, atr_mult, mult)
                        spacing = (i - seed_bar) >= backstep
                        dev = (price - seed_price >= req) if t == 1 else (seed_price - price >= req)
                        if spacing and dev:
                            confirmed.append((seed_type, seed_bar, seed_price))
                            if seed_type == 1:
                                structure_now = 1 if (not math.isnan(prev_high) and seed_price > prev_high) else (3 if not math.isnan(prev_high) else 1)
                                prev_high = seed_price
                            else:
                                structure_now = 2 if (not math.isnan(prev_low) and seed_price > prev_low) else (4 if not math.isnan(prev_low) else 2)
                                prev_low = seed_price
                            last_type, last_price, last_bar, last_atr = t, price, i, atr_val
                            active_price, active_bar = price, i
                            new_pivot[i] = t
                            seed_type = 0; seed_price = float("nan"); seed_bar = None; seed_atr = float("nan")
                elif t == last_type:
                    if (t == 1 and price > last_price) or (t == -1 and price < last_price):
                        last_price, last_bar, last_atr = price, i, atr_val
                        if math.isnan(active_price) or (t == 1 and price > active_price) or (t == -1 and price < active_price):
                            active_price, active_bar = price, i
                else:
                    req = _required_move(last_price, last_atr, mode, dev_pct, atr_mult, mult)
                    spacing = (i - last_bar) >= backstep
                    dev = (price - last_price >= req) if t == 1 else (last_price - price >= req)
                    if spacing and dev:
                        confirmed.append((last_type, last_bar, last_price))
                        if last_type == 1:
                            structure_now = 1 if (not math.isnan(prev_high) and last_price > prev_high) else (3 if not math.isnan(prev_high) else 1)
                            prev_high = last_price
                        else:
                            structure_now = 2 if (not math.isnan(prev_low) and last_price > prev_low) else (4 if not math.isnan(prev_low) else 2)
                            prev_low = last_price
                        last_type, last_price, last_bar, last_atr = t, price, i, atr_val
                        active_price, active_bar = price, i
                        new_pivot[i] = t
        out_type[i] = last_type
        out_last[i] = last_price
        out_active[i] = active_price
        structure[i] = structure_now
    return out_type, out_last, out_active, new_pivot, structure, confirmed


def zigzag_core(
    frame: pd.DataFrame,
    *,
    pivot_depth: int = 5,
    backstep_bars: int = 3,
    use_high_low: bool = True,
    filter_mode: str = "Hybrid",
    deviation_percent: float = 0.50,
    atr_length: int = 14,
    atr_multiplier: float = 1.50,
    show_l2: bool = False,
    l2_min_spacing: int = 2,
    l2_deviation_multiplier: float = 2.0,
    show_secondary: bool = False,
    secondary_pivot_depth: int = 3,
    secondary_backstep_bars: int = 2,
    secondary_deviation_multiplier: float = 0.5,
) -> pd.DataFrame:
    pivot_depth = integer("pivot_depth", pivot_depth, 1, 100)
    backstep_bars = integer("backstep_bars", backstep_bars, 0, 100)
    filter_mode = choice("filter_mode", filter_mode, ("Percent", "ATR", "Hybrid"))
    atr_length = integer("atr_length", atr_length, 1, 500)
    deviation_percent = number("deviation_percent", deviation_percent, 0.0, 100.0)
    atr_multiplier = number("atr_multiplier", atr_multiplier, 0.0, 20.0)
    l2_min_spacing = integer("l2_min_spacing", l2_min_spacing, 1)
    l2_deviation_multiplier = number("l2_deviation_multiplier", l2_deviation_multiplier, 1.0)
    secondary_pivot_depth = integer("secondary_pivot_depth", secondary_pivot_depth, 1, 100)
    secondary_backstep_bars = integer("secondary_backstep_bars", secondary_backstep_bars, 0, 100)
    secondary_deviation_multiplier = number("secondary_deviation_multiplier", secondary_deviation_multiplier, 0.0)

    o, h, l, c, vol = ohlcv(frame, "open", "high", "low", "close", "volume")
    h = h.to_numpy(dtype=float); l = l.to_numpy(dtype=float); c = c.to_numpy(dtype=float)
    idx = frame.index
    n = len(frame)
    atr_a = np.maximum(atr(frame, atr_length).to_numpy(dtype=float), 1e-8)

    def build_events(depth):
        src_high = frame["high"] if use_high_low else frame["close"]
        src_low = frame["low"] if use_high_low else frame["close"]
        ph = pivothigh_strict(src_high, depth, depth).to_numpy(dtype=float)
        pl = pivotlow_strict(src_low, depth, depth).to_numpy(dtype=float)
        ev = {}
        for i in range(n):
            if not math.isnan(ph[i]):
                b = i - depth
                ev.setdefault(b, []).append((1, ph[i], atr_a[max(0, b)]))
            if not math.isnan(pl[i]):
                b = i - depth
                ev.setdefault(b, []).append((-1, pl[i], atr_a[max(0, b)]))
        return ev

    ev = build_events(pivot_depth)
    zz_type, zz_last, zz_active, zz_new, zz_struct, confirmed = _run_zigzag(
        ev, n, backstep_bars, filter_mode, deviation_percent, atr_multiplier)

    # nearest confirmed pivot at or before each bar
    prev_pivot_price = np.full(n, np.nan)
    ci = 0
    for i in range(n):
        # nearest confirmed pivot at or before i
        while ci + 1 < len(confirmed) and confirmed[ci + 1][1] <= i:
            ci += 1
        if confirmed and confirmed[ci][1] <= i:
            prev_pivot_price[i] = confirmed[ci][2]

    # Level 2 from confirmed L1 pivots
    l2_type = np.zeros(n, dtype=np.int64); l2_last = np.full(n, np.nan); l2_conf_n = np.zeros(n, dtype=np.int64)
    if show_l2 and len(confirmed) >= 2:
        ev2 = {}
        for (t, b, p) in confirmed:
            ev2.setdefault(b, []).append((t, p, atr_a[max(0, b)]))
        t2, l2p, a2, n2, s2, conf2 = _run_zigzag(ev2, n, l2_min_spacing, filter_mode, deviation_percent,
                                                  atr_multiplier, l2_deviation_multiplier)
        l2_type, l2_last = t2, l2p
        # count confirmed L2 pivots up to each bar
        conf2_bars = np.array([b for (_, b, _) in conf2]) if conf2 else np.array([], dtype=int)
        for i in range(n):
            l2_conf_n[i] = int((conf2_bars <= i).sum())

    # Secondary
    sec_type = np.zeros(n, dtype=np.int64); sec_last = np.full(n, np.nan); sec_count = np.zeros(n, dtype=np.int64)
    if show_secondary:
        ev_s = build_events(secondary_pivot_depth)
        ts, sp, as_, ns, ss, confs = _run_zigzag(ev_s, n, secondary_backstep_bars, filter_mode,
                                                 deviation_percent, atr_multiplier, secondary_deviation_multiplier)
        sec_type, sec_last = ts, sp
        confs_bars = np.array([b for (_, b, _) in confs]) if confs else np.array([], dtype=int)
        for i in range(n):
            sec_count[i] = int((confs_bars <= i).sum())

    confirmed_count = np.zeros(n, dtype=np.int64)
    conf_bars = np.array([b for (_, b, _) in confirmed]) if confirmed else np.array([], dtype=int)
    for i in range(n):
        confirmed_count[i] = int((conf_bars <= i).sum())

    df = pd.DataFrame(index=idx)
    df["zz_dir"] = pd.Series(zz_type, index=idx).astype("Int64")
    df["zz_last_price"] = zz_last
    df["zz_active_price"] = zz_active
    df["zz_new_pivot"] = pd.Series(zz_new, index=idx).astype("Int64")
    df["zz_structure"] = pd.Series(zz_struct, index=idx).astype("Int64")
    df["zz_confirmed_pivot"] = prev_pivot_price
    df["zz_confirmed_count"] = pd.Series(confirmed_count, index=idx).astype("Int64")
    df["zz_l2_dir"] = pd.Series(l2_type, index=idx).astype("Int64")
    df["zz_l2_last_price"] = l2_last
    df["zz_l2_confirmed_count"] = pd.Series(l2_conf_n, index=idx).astype("Int64")
    df["zz_secondary_dir"] = pd.Series(sec_type, index=idx).astype("Int64")
    df["zz_secondary_last_price"] = sec_last
    df["zz_secondary_count"] = pd.Series(sec_count, index=idx).astype("Int64")
    return df
