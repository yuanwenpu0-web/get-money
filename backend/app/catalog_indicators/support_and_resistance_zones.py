"""Support & Resistance Zones [WavesUnchained]  v0.3.1  —  Python port.

Source: data/pine-collection/indicators/market_structure/
        support_resistance_zones/support_resistance_zones.pine

Detector-driven S/R zone engine. Independent detectors (pivot clusters,
multiple tests, base/supply-demand, order blocks, fair-value gaps, price gaps,
equal-high/low liquidity) feed candidates into a central registry that merges,
scores, applies confluence and tracks each zone through a
touch/reaction/break/flip lifecycle.

Ported computational core:
  * Pivot history retention + ``f_findCluster`` clustering
  * All seven detectors and the candidate registry
    (``f_canMerge`` / ``f_ingestCandidate`` with ATR merging + impulse de-dupe)
  * Confluence scoring (EMA / VWAP / Fibonacci / psychological / relative volume)
  * Lifecycle: width re-check, break, flip, touch, reaction validation, decay
  * Per-bar nearest support / resistance selection (renderer eligibility)

Dropped (display): box/label drawing, tooltip text, dashboard/validation table,
debug candidate markers, Pine logs, colour/visual inputs.

Pine exposes no numeric series, so the port returns the engine's per-bar state:
nearest support/resistance levels + scores, zone counts by state, per-bar
lifecycle events, and the cumulative detector/registry counters.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import atr, boolean, ema, highest, integer, lowest, number, ohlcv, pivothigh_strict, pivotlow_strict, sma

_LIVE_STATES = ("NEW", "TESTED", "WEAK", "CONFIRMED", "FLIPPED")


def _has_source(sources: str, code: str) -> bool:
    return f"|{code}|" in f"|{sources}|"


def _add_source(sources: str, code: str) -> str:
    if _has_source(sources, code):
        return sources
    return code if sources == "" else f"{sources}|{code}"


def _zone_mid(top: float, bottom: float) -> float:
    return (top + bottom) * 0.5


def _clamp_score(v: float) -> float:
    return max(0.0, min(100.0, v))


def support_and_resistance_zones(
    frame: pd.DataFrame,
    *,
    lookback_bars: int = 1200,
    max_stored_zones: int = 180,
    max_zones_per_side: int = 5,
    min_zone_score: float = 20.0,
    atr_length: int = 14,
    max_zone_width_atr: float = 2.0,
    max_merged_width_atr: float = 3.0,
    merge_distance_atr: float = 0.20,
    enable_pivot_clusters: bool = True,
    pivot_depth: int = 5,
    pivot_cluster_min_count: int = 3,
    pivot_cluster_tolerance_atr: float = 0.35,
    pivot_padding_atr: float = 0.08,
    pivot_history_bars: int = 400,
    enable_multiple_tests: bool = True,
    minimum_tests: int = 3,
    test_tolerance_atr: float = 0.22,
    touch_cooldown_bars: int = 3,
    enable_base_zones: bool = True,
    enable_supply_demand: bool = True,
    base_bars: int = 5,
    base_max_width_atr: float = 1.25,
    base_impulse_atr: float = 0.80,
    enable_order_blocks: bool = True,
    ob_impulse_atr: float = 1.0,
    ob_structure_lookback: int = 8,
    ob_search_bars: int = 6,
    ob_use_body: bool = False,
    enable_fvg: bool = True,
    fvg_min_atr: float = 0.10,
    enable_price_gaps: bool = True,
    gap_min_atr: float = 0.15,
    enable_liquidity: bool = True,
    liquidity_tolerance_atr: float = 0.18,
    liquidity_padding_atr: float = 0.05,
    enable_ema: bool = True,
    ema_length: int = 50,
    enable_vwap: bool = True,
    enable_fib: bool = True,
    fib_lookback: int = 120,
    enable_psychological: bool = True,
    psychological_step: float = 10.0,
    enable_volume_confluence: bool = True,
    volume_length: int = 30,
    relative_volume_threshold: float = 1.5,
    weight_pivot: float = 18.0,
    weight_tests: float = 14.0,
    weight_base: float = 12.0,
    weight_sd: float = 18.0,
    weight_ob: float = 18.0,
    weight_fvg: float = 10.0,
    weight_gap: float = 8.0,
    weight_liquidity: float = 12.0,
    weight_breakout: float = 14.0,
    weight_ema: float = 5.0,
    weight_vwap: float = 6.0,
    weight_fib: float = 5.0,
    weight_psych: float = 3.0,
    weight_volume: float = 6.0,
    break_threshold_atr: float = 0.20,
    enable_zone_flips: bool = True,
    reaction_threshold_atr: float = 0.75,
    reaction_horizon: int = 20,
    score_decay_pct: float = 0.25,
    failed_reaction_penalty: float = 6.0,
    broken_keep_bars: int = 100,
    volume_is_real: bool | None = None,
) -> pd.DataFrame:
    # ---- validation ---------------------------------------------------------- #
    lookback_bars = integer("lookback_bars", lookback_bars, 100, 5000)
    max_stored_zones = integer("max_stored_zones", max_stored_zones, 20, 400)
    max_zones_per_side = integer("max_zones_per_side", max_zones_per_side, 1, 20)
    atr_length = integer("atr_length", atr_length, 2, 200)
    pivot_depth = integer("pivot_depth", pivot_depth, 1, 50)
    pivot_cluster_min_count = integer("pivot_cluster_min_count", pivot_cluster_min_count, 2, 10)
    pivot_history_bars = integer("pivot_history_bars", pivot_history_bars, 30, 3000)
    minimum_tests = integer("minimum_tests", minimum_tests, 2, 10)
    touch_cooldown_bars = integer("touch_cooldown_bars", touch_cooldown_bars, 1, 50)
    base_bars = integer("base_bars", base_bars, 2, 30)
    ob_structure_lookback = integer("ob_structure_lookback", ob_structure_lookback, 2, 100)
    ob_search_bars = integer("ob_search_bars", ob_search_bars, 1, 20)
    ema_length = integer("ema_length", ema_length, 2, 500)
    fib_lookback = integer("fib_lookback", fib_lookback, 20, 1000)
    volume_length = integer("volume_length", volume_length, 2, 300)
    reaction_horizon = integer("reaction_horizon", reaction_horizon, 1, 200)
    broken_keep_bars = integer("broken_keep_bars", broken_keep_bars, 0, 1000)
    min_zone_score = number("min_zone_score", min_zone_score, 0.0, 100.0)
    max_zone_width_atr = number("max_zone_width_atr", max_zone_width_atr, 0.1, 10.0)
    max_merged_width_atr = number("max_merged_width_atr", max_merged_width_atr, 0.25, 15.0)
    merge_distance_atr = number("merge_distance_atr", merge_distance_atr, 0.0, 3.0)
    pivot_cluster_tolerance_atr = number("pivot_cluster_tolerance_atr", pivot_cluster_tolerance_atr, 0.05, 3.0)
    pivot_padding_atr = number("pivot_padding_atr", pivot_padding_atr, 0.0, 1.0)
    test_tolerance_atr = number("test_tolerance_atr", test_tolerance_atr, 0.02, 2.0)
    base_max_width_atr = number("base_max_width_atr", base_max_width_atr, 0.1, 5.0)
    base_impulse_atr = number("base_impulse_atr", base_impulse_atr, 0.1, 5.0)
    ob_impulse_atr = number("ob_impulse_atr", ob_impulse_atr, 0.2, 5.0)
    fvg_min_atr = number("fvg_min_atr", fvg_min_atr, 0.0, 3.0)
    gap_min_atr = number("gap_min_atr", gap_min_atr, 0.0, 5.0)
    liquidity_tolerance_atr = number("liquidity_tolerance_atr", liquidity_tolerance_atr, 0.01, 2.0)
    liquidity_padding_atr = number("liquidity_padding_atr", liquidity_padding_atr, 0.0, 1.0)
    psychological_step = number("psychological_step", psychological_step, 1e-7)
    relative_volume_threshold = number("relative_volume_threshold", relative_volume_threshold, 1.0, 10.0)
    break_threshold_atr = number("break_threshold_atr", break_threshold_atr, 0.0, 3.0)
    reaction_threshold_atr = number("reaction_threshold_atr", reaction_threshold_atr, 0.1, 10.0)
    score_decay_pct = number("score_decay_pct", score_decay_pct, 0.0, 5.0)
    failed_reaction_penalty = number("failed_reaction_penalty", failed_reaction_penalty, 0.0, 50.0)
    weight_pivot = number("weight_pivot", weight_pivot, 0.0)
    weight_tests = number("weight_tests", weight_tests, 0.0)
    weight_base = number("weight_base", weight_base, 0.0)
    weight_sd = number("weight_sd", weight_sd, 0.0)
    weight_ob = number("weight_ob", weight_ob, 0.0)
    weight_fvg = number("weight_fvg", weight_fvg, 0.0)
    weight_gap = number("weight_gap", weight_gap, 0.0)
    weight_liquidity = number("weight_liquidity", weight_liquidity, 0.0)
    weight_breakout = number("weight_breakout", weight_breakout, 0.0)
    weight_ema = number("weight_ema", weight_ema, 0.0)
    weight_vwap = number("weight_vwap", weight_vwap, 0.0)
    weight_fib = number("weight_fib", weight_fib, 0.0)
    weight_psych = number("weight_psych", weight_psych, 0.0)
    weight_volume = number("weight_volume", weight_volume, 0.0)

    open_, high, low, close, volume = ohlcv(frame, "open", "high", "low", "close", "volume")
    idx = frame.index
    n = len(frame)
    o = open_.to_numpy(dtype=float)
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    v = volume.to_numpy(dtype=float)

    atr_a = atr(frame, atr_length).to_numpy(dtype=float)
    ema_a = ema(close, ema_length).to_numpy(dtype=float)
    fib_high = highest(high, fib_lookback).to_numpy(dtype=float)
    fib_low = lowest(low, fib_lookback).to_numpy(dtype=float)
    ph = pivothigh_strict(high, pivot_depth, pivot_depth).to_numpy(dtype=float)
    pl = pivotlow_strict(low, pivot_depth, pivot_depth).to_numpy(dtype=float)

    vol_avg = sma(volume, volume_length).to_numpy(dtype=float)
    real = volume_is_real
    if real is None:
        real = bool(np.any(v > 0))
    real = bool(real)

    # session-anchored VWAP (reset on calendar-date change)
    typical = (h + l + c) / 3.0
    vwap_a = np.full(n, np.nan)
    if isinstance(idx, pd.DatetimeIndex):
        dates = idx.normalize()
    else:
        dates = None
    cum_pv = cum_v = 0.0
    last_d: object = None
    for i in range(n):
        d = dates[i] if dates is not None else None
        if d != last_d:
            cum_pv = cum_v = 0.0
            last_d = d
        vv = v[i] if not math.isnan(v[i]) else 0.0
        cum_pv += typical[i] * vv
        cum_v += vv
        vwap_a[i] = cum_pv / cum_v if cum_v > 0 else np.nan
    rel_vol = np.where((vol_avg > 0) & real, v / vol_avg, np.nan)

    prev_base_high = pd.Series(h).rolling(base_bars).max().shift(1).to_numpy(dtype=float)
    prev_base_low = pd.Series(l).rolling(base_bars).min().shift(1).to_numpy(dtype=float)
    prior_struct_high = pd.Series(h).rolling(ob_structure_lookback).max().shift(1).to_numpy(dtype=float)
    prior_struct_low = pd.Series(l).rolling(ob_structure_lookback).min().shift(1).to_numpy(dtype=float)

    confluence_ceiling = weight_ema + weight_vwap + weight_fib + weight_psych + weight_volume
    active_confluence = weight_ema + weight_fib + weight_psych
    confluence_scale = 1.0 if (real or active_confluence <= 0) else confluence_ceiling / active_confluence

    # ---- state --------------------------------------------------------------- #
    zones: list[dict] = []
    pivot_hi_px: list[float] = []
    pivot_hi_bar: list[int] = []
    pivot_lo_px: list[float] = []
    pivot_lo_bar: list[int] = []
    next_id = 1
    stat = dict(candidates=0, created=0, merged=0, rejected=0,
                touches=0, success=0, weak=0, breaks=0, flips=0)

    def find_cluster(prices, bars, ref_price, ref_bar, tol, history):
        clo = chi = ref_price
        count = 0
        earliest = ref_bar
        for p, b in zip(prices, bars):
            if (ref_bar - b) <= history and abs(p - ref_price) <= tol:
                count += 1
                clo = min(clo, p)
                chi = max(chi, p)
                earliest = min(earliest, b)
        return clo, chi, count, earliest

    def ingest(cand):
        top, bottom, w = cand["top"], cand["bottom"], cand["top"] - cand["bottom"]
        if not (top > bottom and w <= max_zone_width_atr * atr_a_cur):
            stat["rejected"] += 1
            return 0
        merge_idx = -1
        best_dist = 1e30
        for zi, z in enumerate(zones):
            if z["dir"] != cand["dir"] or z["state"] not in _LIVE_STATES:
                continue
            overlap = min(z["top"], cand["top"]) >= max(z["bottom"], cand["bottom"])
            distance = 0.0
            if not overlap:
                if cand["bottom"] > z["top"]:
                    distance = cand["bottom"] - z["top"]
                elif z["bottom"] > cand["top"]:
                    distance = z["bottom"] - cand["top"]
            if not (overlap or distance <= merge_distance_atr * atr_a_cur):
                continue
            merged_top = max(z["top"], cand["top"])
            merged_bottom = min(z["bottom"], cand["bottom"])
            if merged_top - merged_bottom > max_merged_width_atr * atr_a_cur:
                continue
            cd = abs(_zone_mid(z["top"], z["bottom"]) - _zone_mid(cand["top"], cand["bottom"]))
            if cd < best_dist:
                best_dist = cd
                merge_idx = zi
        if merge_idx >= 0:
            z = zones[merge_idx]
            new_evidence = not _has_source(z["sources"], cand["sources"])
            z["top"] = max(z["top"], cand["top"])
            z["bottom"] = min(z["bottom"], cand["bottom"])
            z["originBar"] = min(z["originBar"], cand["originBar"])
            z["lastEvidenceBar"] = i
            if new_evidence:
                z["sources"] = cand["sources"] if z["sources"] == "" else f"{z['sources']}|{cand['sources']}"
                credit = cand["initialScore"]
                if cand["impulseBar"] is not None and z["impulseBar"] == cand["impulseBar"]:
                    credit = max(0.0, cand["initialScore"] - z["impulseWeight"])
                    z["impulseWeight"] = max(z["impulseWeight"], cand["initialScore"])
                elif cand["impulseBar"] is not None:
                    z["impulseBar"] = cand["impulseBar"]
                    z["impulseWeight"] = cand["initialScore"]
                z["score"] = _clamp_score(z["score"] + credit)
            stat["merged"] += 1
            return 2
        zones.append({
            "id": next_id_cur, "dir": cand["dir"], "top": top, "bottom": bottom,
            "originBar": cand["originBar"], "createdBar": i, "lastEvidenceBar": i,
            "confirmedBar": cand["confirmedBar"], "sources": cand["sources"], "state": "NEW",
            "score": _clamp_score(cand["initialScore"]), "touches": 0, "reactions": 0, "breaks": 0,
            "lastTouchBar": None, "lastBreakBar": None, "pendingReaction": False,
            "reactionStartBar": None, "reactionReference": None, "reactionAtr": None,
            "maxFavorableExcursion": 0.0, "multipleTestsAwarded": False,
            "impulseBar": cand["impulseBar"],
            "impulseWeight": 0.0 if cand["impulseBar"] is None else cand["initialScore"],
        })
        stat["created"] += 1
        return 1

    def candidate(direction, top, bottom, origin, sources, score, impulse_bar=None):
        return {"dir": direction, "top": top, "bottom": bottom, "originBar": origin,
                "confirmedBar": i, "sources": sources, "initialScore": score,
                "impulseBar": impulse_bar}

    def apply_confluence(z, cur_atr):
        tol = 0.05 * cur_atr
        if (enable_ema and (ema_a[i] >= z["bottom"] - tol) and (ema_a[i] <= z["top"] + tol)
                and not _has_source(z["sources"], "MA")):
            z["sources"] = _add_source(z["sources"], "MA")
            z["score"] = _clamp_score(z["score"] + weight_ema * confluence_scale)
        if (enable_vwap and real and not math.isnan(vwap_a[i])
                and (vwap_a[i] >= z["bottom"] - tol) and (vwap_a[i] <= z["top"] + tol)
                and not _has_source(z["sources"], "VWAP")):
            z["sources"] = _add_source(z["sources"], "VWAP")
            z["score"] = _clamp_score(z["score"] + weight_vwap)
        f38, f50, f62 = fib382[i], fib500[i], fib618[i]
        fib_inside = ((z["bottom"] <= f38 <= z["top"]) or (z["bottom"] <= f50 <= z["top"])
                      or (z["bottom"] <= f62 <= z["top"]))
        if enable_fib and fib_inside and not _has_source(z["sources"], "FIB"):
            z["sources"] = _add_source(z["sources"], "FIB")
            z["score"] = _clamp_score(z["score"] + weight_fib * confluence_scale)
        mid = _zone_mid(z["top"], z["bottom"])
        nearest_psych = round(mid / psychological_step) * psychological_step
        if (enable_psychological and z["bottom"] <= nearest_psych <= z["top"]
                and not _has_source(z["sources"], "PSY")):
            z["sources"] = _add_source(z["sources"], "PSY")
            z["score"] = _clamp_score(z["score"] + weight_psych * confluence_scale)
        if (enable_volume_confluence and real and not math.isnan(rel_vol[i])
                and rel_vol[i] >= relative_volume_threshold and not _has_source(z["sources"], "VOL")):
            z["sources"] = _add_source(z["sources"], "VOL")
            z["score"] = _clamp_score(z["score"] + weight_volume)

    # Fibonacci levels per bar
    fib_range = fib_high - fib_low
    fib382 = fib_high - fib_range * 0.382
    fib500 = fib_high - fib_range * 0.500
    fib618 = fib_high - fib_range * 0.618

    # ---- output buffers ------------------------------------------------------ #
    sup_top = np.full(n, np.nan)
    sup_bot = np.full(n, np.nan)
    sup_score = np.full(n, np.nan)
    res_top = np.full(n, np.nan)
    res_bot = np.full(n, np.nan)
    res_score = np.full(n, np.nan)
    sup_dist = np.full(n, np.nan)
    res_dist = np.full(n, np.nan)
    active_cnt = np.zeros(n, dtype=np.int64)
    confirmed_cnt = np.zeros(n, dtype=np.int64)
    weak_cnt = np.zeros(n, dtype=np.int64)
    flipped_cnt = np.zeros(n, dtype=np.int64)
    broken_cnt = np.zeros(n, dtype=np.int64)
    touch_ev = np.zeros(n, dtype=np.int64)
    break_ev = np.zeros(n, dtype=np.int64)
    flip_ev = np.zeros(n, dtype=np.int64)
    success_ev = np.zeros(n, dtype=np.int64)
    weak_ev = np.zeros(n, dtype=np.int64)
    stat_cand = np.zeros(n, dtype=np.int64)
    stat_created = np.zeros(n, dtype=np.int64)
    stat_merged = np.zeros(n, dtype=np.int64)
    stat_rejected = np.zeros(n, dtype=np.int64)
    stat_touches = np.zeros(n, dtype=np.int64)
    stat_breaks = np.zeros(n, dtype=np.int64)
    stat_flips = np.zeros(n, dtype=np.int64)

    next_id_cur = 1
    for i in range(n):
        atr_a_cur = atr_a[i]
        if math.isnan(atr_a_cur):
            atr_a_cur = 0.0
        # ---- record pivots ----
        if not math.isnan(ph[i]):
            pbar = i - pivot_depth
            pivot_hi_px.append(ph[i])
            pivot_hi_bar.append(pbar)
            while pivot_hi_px and pbar - pivot_hi_bar[0] > pivot_history_bars:
                pivot_hi_px.pop(0)
                pivot_hi_bar.pop(0)
        if not math.isnan(pl[i]):
            pbar = i - pivot_depth
            pivot_lo_px.append(pl[i])
            pivot_lo_bar.append(pbar)
            while pivot_lo_px and pbar - pivot_lo_bar[0] > pivot_history_bars:
                pivot_lo_px.pop(0)
                pivot_lo_bar.pop(0)

        pc_handled_high = pc_handled_low = False

        # ---- PIVOT CLUSTERS ----
        if enable_pivot_clusters and not math.isnan(ph[i]):
            pbar = i - pivot_depth
            clo, chi, cnt, earliest = find_cluster(
                pivot_hi_px, pivot_hi_bar, ph[i], pbar, pivot_cluster_tolerance_atr * atr_a_cur, pivot_history_bars)
            if cnt >= pivot_cluster_min_count:
                cand = candidate(-1, chi + pivot_padding_atr * atr_a_cur,
                                 clo - pivot_padding_atr * atr_a_cur, earliest, "PC", weight_pivot)
                stat["candidates"] += 1
                act = ingest(cand)
                pc_handled_high = act > 0
                if act == 1:
                    next_id_cur += 1
        if enable_pivot_clusters and not math.isnan(pl[i]):
            pbar = i - pivot_depth
            clo, chi, cnt, earliest = find_cluster(
                pivot_lo_px, pivot_lo_bar, pl[i], pbar, pivot_cluster_tolerance_atr * atr_a_cur, pivot_history_bars)
            if cnt >= pivot_cluster_min_count:
                cand = candidate(1, chi + pivot_padding_atr * atr_a_cur,
                                 clo - pivot_padding_atr * atr_a_cur, earliest, "PC", weight_pivot)
                stat["candidates"] += 1
                act = ingest(cand)
                pc_handled_low = act > 0
                if act == 1:
                    next_id_cur += 1

        # ---- MULTIPLE TESTS ----
        if enable_multiple_tests and not pc_handled_high and not math.isnan(ph[i]):
            pbar = i - pivot_depth
            clo, chi, cnt, earliest = find_cluster(
                pivot_hi_px, pivot_hi_bar, ph[i], pbar, test_tolerance_atr * atr_a_cur, pivot_history_bars)
            if cnt >= minimum_tests:
                cand = candidate(-1, chi + pivot_padding_atr * atr_a_cur,
                                 clo - pivot_padding_atr * atr_a_cur, earliest, "MT", weight_tests)
                stat["candidates"] += 1
                act = ingest(cand)
                if act == 1:
                    next_id_cur += 1
        if enable_multiple_tests and not pc_handled_low and not math.isnan(pl[i]):
            pbar = i - pivot_depth
            clo, chi, cnt, earliest = find_cluster(
                pivot_lo_px, pivot_lo_bar, pl[i], pbar, test_tolerance_atr * atr_a_cur, pivot_history_bars)
            if cnt >= minimum_tests:
                cand = candidate(1, chi + pivot_padding_atr * atr_a_cur,
                                 clo - pivot_padding_atr * atr_a_cur, earliest, "MT", weight_tests)
                stat["candidates"] += 1
                act = ingest(cand)
                if act == 1:
                    next_id_cur += 1

        # ---- LIQUIDITY / EQUAL HIGHS & LOWS ----
        if enable_liquidity and not pc_handled_high and not math.isnan(ph[i]):
            pbar = i - pivot_depth
            clo, chi, cnt, earliest = find_cluster(
                pivot_hi_px, pivot_hi_bar, ph[i], pbar, liquidity_tolerance_atr * atr_a_cur, pivot_history_bars)
            if cnt >= 2:
                cand = candidate(-1, chi + liquidity_padding_atr * atr_a_cur,
                                 clo - liquidity_padding_atr * atr_a_cur, earliest, "LIQ", weight_liquidity)
                stat["candidates"] += 1
                act = ingest(cand)
                if act == 1:
                    next_id_cur += 1
        if enable_liquidity and not pc_handled_low and not math.isnan(pl[i]):
            pbar = i - pivot_depth
            clo, chi, cnt, earliest = find_cluster(
                pivot_lo_px, pivot_lo_bar, pl[i], pbar, liquidity_tolerance_atr * atr_a_cur, pivot_history_bars)
            if cnt >= 2:
                cand = candidate(1, chi + liquidity_padding_atr * atr_a_cur,
                                 clo - liquidity_padding_atr * atr_a_cur, earliest, "LIQ", weight_liquidity)
                stat["candidates"] += 1
                act = ingest(cand)
                if act == 1:
                    next_id_cur += 1

        # ---- BASE / SUPPLY & DEMAND ----
        if enable_base_zones or enable_supply_demand:
            pb_hi, pb_lo = prev_base_high[i], prev_base_low[i]
            if not (math.isnan(pb_hi) or math.isnan(pb_lo)):
                compact = (pb_hi - pb_lo) <= base_max_width_atr * atr_a_cur
                body = abs(c[i] - o[i])
                strong_exit = body >= base_impulse_atr * atr_a_cur
                if compact and strong_exit:
                    source = "SD" if enable_supply_demand else "BASE"
                    score = weight_sd if enable_supply_demand else weight_base
                    if c[i] > pb_hi:
                        cand = candidate(1, pb_hi, pb_lo, i - base_bars, source, score, impulse_bar=i)
                        stat["candidates"] += 1
                        act = ingest(cand)
                        if act == 1:
                            next_id_cur += 1
                    if c[i] < pb_lo:
                        cand = candidate(-1, pb_hi, pb_lo, i - base_bars, source, score, impulse_bar=i)
                        stat["candidates"] += 1
                        act = ingest(cand)
                        if act == 1:
                            next_id_cur += 1

        # ---- ORDER BLOCKS ----
        if enable_order_blocks:
            body = abs(c[i] - o[i])
            ps_hi, ps_lo = prior_struct_high[i], prior_struct_low[i]
            bull_disp = (c[i] > o[i]) and body >= ob_impulse_atr * atr_a_cur and not math.isnan(ps_hi) and c[i] > ps_hi
            bear_disp = (c[i] < o[i]) and body >= ob_impulse_atr * atr_a_cur and not math.isnan(ps_lo) and c[i] < ps_lo
            if bull_disp:
                found = None
                for off in range(1, ob_search_bars + 1):
                    if i - off >= 0 and c[i - off] < o[i - off]:
                        found = off
                        break
                if found is not None:
                    if ob_use_body:
                        ob_top = max(o[i - found], c[i - found])
                        ob_bot = min(o[i - found], c[i - found])
                    else:
                        ob_top = h[i - found]
                        ob_bot = l[i - found]
                    cand = candidate(1, ob_top, ob_bot, i - found, "OB", weight_ob, impulse_bar=i)
                    stat["candidates"] += 1
                    act = ingest(cand)
                    if act == 1:
                        next_id_cur += 1
            if bear_disp:
                found = None
                for off in range(1, ob_search_bars + 1):
                    if i - off >= 0 and c[i - off] > o[i - off]:
                        found = off
                        break
                if found is not None:
                    if ob_use_body:
                        ob_top = max(o[i - found], c[i - found])
                        ob_bot = min(o[i - found], c[i - found])
                    else:
                        ob_top = h[i - found]
                        ob_bot = l[i - found]
                    cand = candidate(-1, ob_top, ob_bot, i - found, "OB", weight_ob, impulse_bar=i)
                    stat["candidates"] += 1
                    act = ingest(cand)
                    if act == 1:
                        next_id_cur += 1

        # ---- FAIR VALUE GAPS ----
        if enable_fvg and i >= 2:
            if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= fvg_min_atr * atr_a_cur:
                cand = candidate(1, l[i], h[i - 2], i - 2, "FVG", weight_fvg, impulse_bar=i - 1)
                stat["candidates"] += 1
                act = ingest(cand)
                if act == 1:
                    next_id_cur += 1
            if h[i] < l[i - 2] and (l[i - 2] - h[i]) >= fvg_min_atr * atr_a_cur:
                cand = candidate(-1, l[i - 2], h[i], i - 2, "FVG", weight_fvg, impulse_bar=i - 1)
                stat["candidates"] += 1
                act = ingest(cand)
                if act == 1:
                    next_id_cur += 1

        # ---- PRICE GAPS ----
        if enable_price_gaps and i >= 1:
            if l[i] > h[i - 1] and (l[i] - h[i - 1]) >= gap_min_atr * atr_a_cur:
                cand = candidate(1, l[i], h[i - 1], i - 1, "GAP", weight_gap)
                stat["candidates"] += 1
                act = ingest(cand)
                if act == 1:
                    next_id_cur += 1
            if h[i] < l[i - 1] and (l[i - 1] - h[i]) >= gap_min_atr * atr_a_cur:
                cand = candidate(-1, l[i - 1], h[i], i - 1, "GAP", weight_gap)
                stat["candidates"] += 1
                act = ingest(cand)
                if act == 1:
                    next_id_cur += 1

        # ---- LIFECYCLE ----
        lifecycle_n = len(zones)
        for zi in range(lifecycle_n):
            z = zones[zi]
            if z["state"] == "INVALID":
                continue
            if i - z["lastEvidenceBar"] > lookback_bars:
                z["state"] = "INVALID"
                continue
            # width re-check
            max_w = max_merged_width_atr * atr_a_cur
            if z["top"] - z["bottom"] > max_w:
                mid = _zone_mid(z["top"], z["bottom"])
                z["top"] = mid + max_w * 0.5
                z["bottom"] = mid - max_w * 0.5
            # break detection
            support_broken = z["dir"] == 1 and c[i] < z["bottom"] - break_threshold_atr * atr_a_cur
            resistance_broken = z["dir"] == -1 and c[i] > z["top"] + break_threshold_atr * atr_a_cur
            broken = support_broken or resistance_broken
            if broken and z["state"] != "BROKEN":
                z["breaks"] += 1
                z["lastBreakBar"] = i
                stat["breaks"] += 1
                break_ev[i] = 1
                z["pendingReaction"] = False
                if enable_zone_flips and z["breaks"] == 1:
                    z["dir"] = -z["dir"]
                    z["state"] = "FLIPPED"
                    flip_ev[i] = 1
                    stat["flips"] += 1
                    if not _has_source(z["sources"], "BO"):
                        z["sources"] = _add_source(z["sources"], "BO")
                        z["score"] = _clamp_score(z["score"] + weight_breakout)
                else:
                    z["state"] = "BROKEN"
            # remove old broken zones
            if z["state"] == "BROKEN" and z["lastBreakBar"] is not None and i - z["lastBreakBar"] > broken_keep_bars:
                z["state"] = "INVALID"
            # touch detection
            zone_intersection = h[i] >= z["bottom"] and l[i] <= z["top"]
            cooldown_passed = z["lastTouchBar"] is None or i - z["lastTouchBar"] > touch_cooldown_bars
            after_conf = i > z["confirmedBar"]
            not_break_bar = z["lastBreakBar"] is None or i != z["lastBreakBar"]
            if (zone_intersection and cooldown_passed and not z["pendingReaction"]
                    and z["state"] != "BROKEN" and after_conf and not_break_bar):
                z["touches"] += 1
                z["lastTouchBar"] = i
                stat["touches"] += 1
                touch_ev[i] = 1
                if z["state"] == "NEW":
                    z["state"] = "TESTED"
                z["pendingReaction"] = True
                z["reactionStartBar"] = i
                z["reactionReference"] = z["top"] if z["dir"] == 1 else z["bottom"]
                z["reactionAtr"] = atr_a_cur
                z["maxFavorableExcursion"] = 0.0
                if enable_multiple_tests and z["touches"] >= minimum_tests and not z["multipleTestsAwarded"]:
                    z["multipleTestsAwarded"] = True
                    if not _has_source(z["sources"], "MT"):
                        z["sources"] = _add_source(z["sources"], "MT")
                        z["score"] = _clamp_score(z["score"] + weight_tests)
                apply_confluence(z, atr_a_cur)
            # reaction validation
            if z["pendingReaction"] and i > z["reactionStartBar"]:
                favorable = (h[i] - z["reactionReference"]) if z["dir"] == 1 else (z["reactionReference"] - l[i])
                z["maxFavorableExcursion"] = max(z["maxFavorableExcursion"], favorable)
                required = reaction_threshold_atr * z["reactionAtr"]
                success = z["maxFavorableExcursion"] >= required
                expired = i - z["reactionStartBar"] > reaction_horizon
                if success:
                    z["pendingReaction"] = False
                    z["reactions"] += 1
                    z["state"] = "CONFIRMED"
                    stat["success"] += 1
                    success_ev[i] = 1
                elif expired:
                    z["pendingReaction"] = False
                    if z["state"] != "FLIPPED":
                        z["state"] = "WEAK"
                    z["score"] = _clamp_score(z["score"] - failed_reaction_penalty)
                    stat["weak"] += 1
                    weak_ev[i] = 1
            # score decay
            if z["state"] in _LIVE_STATES and z["state"] != "CONFIRMED":
                z["score"] = z["score"] * (1.0 - score_decay_pct / 100.0)

        # ---- prune storage ----
        while len(zones) > max_stored_zones:
            zones.pop(0)

        # ---- record per-bar output ----
        best_sup = best_res = None
        best_sup_d = best_res_d = 1e30
        a_cnt = c_cnt = w_cnt = f_cnt = b_cnt = 0
        for z in zones:
            st = z["state"]
            if st in _LIVE_STATES:
                a_cnt += 1
            if st == "CONFIRMED":
                c_cnt += 1
            elif st == "WEAK":
                w_cnt += 1
            elif st == "FLIPPED":
                f_cnt += 1
            elif st == "BROKEN":
                b_cnt += 1
            if st not in _LIVE_STATES or z["score"] < min_zone_score:
                continue
            mid = _zone_mid(z["top"], z["bottom"])
            dist = abs(mid - c[i])
            if z["dir"] == 1 and z["bottom"] <= c[i] and dist < best_sup_d:
                best_sup_d = dist
                best_sup = z
            elif z["dir"] == -1 and z["top"] >= c[i] and dist < best_res_d:
                best_res_d = dist
                best_res = z
        if best_sup is not None:
            sup_top[i] = best_sup["top"]
            sup_bot[i] = best_sup["bottom"]
            sup_score[i] = best_sup["score"]
            sup_dist[i] = best_sup_d
        if best_res is not None:
            res_top[i] = best_res["top"]
            res_bot[i] = best_res["bottom"]
            res_score[i] = best_res["score"]
            res_dist[i] = best_res_d
        active_cnt[i] = a_cnt
        confirmed_cnt[i] = c_cnt
        weak_cnt[i] = w_cnt
        flipped_cnt[i] = f_cnt
        broken_cnt[i] = b_cnt
        stat_cand[i] = stat["candidates"]
        stat_created[i] = stat["created"]
        stat_merged[i] = stat["merged"]
        stat_rejected[i] = stat["rejected"]
        stat_touches[i] = stat["touches"]
        stat_breaks[i] = stat["breaks"]
        stat_flips[i] = stat["flips"]

    out = pd.DataFrame(index=idx)
    out["srz_support_top"] = sup_top
    out["srz_support_bottom"] = sup_bot
    out["srz_support_score"] = sup_score
    out["srz_nearest_support_dist"] = sup_dist
    out["srz_resistance_top"] = res_top
    out["srz_resistance_bottom"] = res_bot
    out["srz_resistance_score"] = res_score
    out["srz_nearest_resistance_dist"] = res_dist
    out["srz_active_count"] = pd.Series(active_cnt, index=idx).astype("Int64")
    out["srz_confirmed_count"] = pd.Series(confirmed_cnt, index=idx).astype("Int64")
    out["srz_weak_count"] = pd.Series(weak_cnt, index=idx).astype("Int64")
    out["srz_flipped_count"] = pd.Series(flipped_cnt, index=idx).astype("Int64")
    out["srz_broken_count"] = pd.Series(broken_cnt, index=idx).astype("Int64")
    out["srz_touch_event"] = pd.Series(touch_ev, index=idx).astype("Int64")
    out["srz_break_event"] = pd.Series(break_ev, index=idx).astype("Int64")
    out["srz_flip_event"] = pd.Series(flip_ev, index=idx).astype("Int64")
    out["srz_reaction_success"] = pd.Series(success_ev, index=idx).astype("Int64")
    out["srz_reaction_weak"] = pd.Series(weak_ev, index=idx).astype("Int64")
    out["srz_stat_candidates"] = pd.Series(stat_cand, index=idx).astype("Int64")
    out["srz_stat_created"] = pd.Series(stat_created, index=idx).astype("Int64")
    out["srz_stat_merged"] = pd.Series(stat_merged, index=idx).astype("Int64")
    out["srz_stat_rejected"] = pd.Series(stat_rejected, index=idx).astype("Int64")
    out["srz_stat_touches"] = pd.Series(stat_touches, index=idx).astype("Int64")
    out["srz_stat_breaks"] = pd.Series(stat_breaks, index=idx).astype("Int64")
    out["srz_stat_flips"] = pd.Series(stat_flips, index=idx).astype("Int64")
    return out
