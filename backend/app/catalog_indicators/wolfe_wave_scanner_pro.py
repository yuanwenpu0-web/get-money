"""Wolfe Wave Scanner Pro [WavesUnchained]  v4.15  —  Python port.

Source: data/pine-collection/indicators/market_structure/
        wolfe_wave_scanner_pro/wolfe_wave_scanner_pro.pine

Searches a depth window of confirmed swings for every non-overlapping valid
5-point Wolfe Wave wedge, projects each one's dynamic EPA target, and tracks it
until the target is hit, the setup invalidates, or its ETA lifecycle expires.

Ported computational core:
  * Best-of-window combinatorial swing search (1/3/5 same direction, 2/4
    opposite) for every point-5 anchor in the depth window
  * ``f_eval``: contraction, channel traversal, apex/ETA, directional P5 zone,
    risk/reward, timing, symmetry, leg purity, swing- and bar-boundary holds
  * Per-point-5 best tracking + multi-pattern overlap/evict/dedupe
  * ACTIVE → PARTIAL / TARGET_HIT / INVALIDATED / EXPIRED lifecycle

Dropped (display): line/label drawing, dashboard/debug tables, candidate
marker, alertcondition, log.info, colour inputs.

Pine exposes no numeric series, so the port returns the scanner's per-bar state:
best-in-window candidate, best active pattern, and the per-bar lifecycle events.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import atr, integer, number, pivothigh_strict, pivotlow_strict

_NAN = float("nan")


def _price(x1, y1, x2, y2, x):
    return y1 if x2 == x1 else y1 + (y2 - y1) * float(x - x1) / float(x2 - x1)


def _slope(x1, y1, x2, y2):
    return _NAN if x2 == x1 else (y2 - y1) / float(x2 - x1)


def _clamp01(v):
    return min(max(v, 0.0), 1.0)


def _sgn(v):
    return _NAN if math.isnan(v) else (1 if v > 0 else -1 if v < 0 else 0)


def _intersect_x(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    m1 = _slope(ax1, ay1, ax2, ay2)
    m2 = _slope(bx1, by1, bx2, by2)
    if math.isnan(m1) or math.isnan(m2) or abs(m1 - m2) < 1e-7:
        return None
    b1 = ay1 - m1 * ax1
    b2 = by1 - m2 * bx1
    return int(round((b2 - b1) / (m1 - m2)))


def _empty_pattern():
    nan = _NAN
    return {
        "valid": False, "bull": False, "score": 0,
        "x1": nan, "x2": nan, "x3": nan, "x4": nan, "x5": nan,
        "y1": nan, "y2": nan, "y3": nan, "y4": nan, "y5": nan,
        "targetNow": nan, "targetETA": nan, "invalidate": nan, "patternAtr": nan,
        "riskReward": nan, "legPurityRatio": 0.0, "contractionRatio": nan,
        "widthStartAtr": nan, "widthMidAtr": nan, "widthEndAtr": nan,
        "sameDirectionOk": False, "slopesComparableOk": False, "slopeRatio": nan,
        "p4WidthAtr": nan, "channelWidthAt5Atr": nan, "traversal45": nan,
        "p5OffsetAtr": nan, "purityOffenderX": None, "purityOffenderLeg": 0,
        "swingBoundaryViolationAtr": nan, "swingBoundaryViolationX": None,
        "barBoundaryViolationAtr": 0.0, "barBoundaryViolationX": None, "etaBar": nan,
        "structureOk": False, "contractionOk": False, "channelOk": False, "etaOk": False,
        "p5ZoneOk": False, "rewardOk": False, "timingOk": False, "symmetryOk": False,
        "legPurityOk": False, "boundaryHoldOk": False, "barsInsideOk": True,
        "structureScore": 0, "contractionScore": 0, "channelScore": 0, "etaScore": 0,
        "p5Score": 0, "rewardScore": 0, "timingScore": 0, "symmetryScore": 0, "purityScore": 0,
    }


def wolfe_wave_scanner_pro(
    frame: pd.DataFrame,
    *,
    pivot_len: int = 6,
    max_swings: int = 80,
    search_depth: int = 16,
    max_concurrent_patterns: int = 1,
    min_score: int = 60,
    atr_len: int = 14,
    p5_inside_tol_atr: float = 0.75,
    p5_overshoot_tol_atr: float = 1.0,
    swing_boundary_tol_atr: float = 0.35,
    bar_boundary_tol_atr: float = 0.25,
    max_target_distance: float = 8.0,
    min_rr: float = 1.5,
    require_reward: bool = False,
    invalidation_atr: float = 0.5,
    dynamic_invalidation_bars: int = 10,
    expiry_buffer_bars: int = 10,
    max_leg_purity_ratio: float = 0.50,
    invalidate_on_close: bool = True,
    show_old_patterns: bool = False,
    show_candidate: bool = True,
    candidate_threshold: int = 25,
) -> pd.DataFrame:
    pivot_len = integer("pivot_len", pivot_len, 2, 30)
    max_swings = integer("max_swings", max_swings, 20, 200)
    search_depth = integer("search_depth", search_depth, 5, 24)
    max_concurrent_patterns = integer("max_concurrent_patterns", max_concurrent_patterns, 1, 8)
    min_score = integer("min_score", min_score, 0, 100)
    atr_len = integer("atr_len", atr_len, 2, 200)
    dynamic_invalidation_bars = integer("dynamic_invalidation_bars", dynamic_invalidation_bars, 0, 100)
    expiry_buffer_bars = integer("expiry_buffer_bars", expiry_buffer_bars, 0, 200)
    candidate_threshold = integer("candidate_threshold", candidate_threshold, 0, 100)
    p5_inside_tol_atr = number("p5_inside_tol_atr", p5_inside_tol_atr, 0.1, 2.0)
    p5_overshoot_tol_atr = number("p5_overshoot_tol_atr", p5_overshoot_tol_atr, 0.1, 3.0)
    swing_boundary_tol_atr = number("swing_boundary_tol_atr", swing_boundary_tol_atr, 0.05, 2.0)
    bar_boundary_tol_atr = number("bar_boundary_tol_atr", bar_boundary_tol_atr, 0.05, 2.0)
    max_target_distance = number("max_target_distance", max_target_distance, 0.5)
    min_rr = number("min_rr", min_rr, 0.25, 10.0)
    invalidation_atr = number("invalidation_atr", invalidation_atr, 0.1, 3.0)
    max_leg_purity_ratio = number("max_leg_purity_ratio", max_leg_purity_ratio, 0.10, 1.0)

    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    idx = frame.index
    n = len(frame)

    atr_a = atr(frame, atr_len).to_numpy(dtype=float)
    ph = pivothigh_strict(frame["high"], pivot_len, pivot_len).to_numpy(dtype=float)
    pl = pivotlow_strict(frame["low"], pivot_len, pivot_len).to_numpy(dtype=float)

    swings: list[list] = []  # [x, y, dir]
    last_swing_x = -1
    best = _empty_pattern()
    tracked: list[dict] = []

    best_score = np.zeros(n, dtype=np.int64)
    best_bull = np.zeros(n, dtype=np.int64)
    best_structure = np.zeros(n, dtype=np.int64)
    best_valid = np.zeros(n, dtype=np.int64)
    best_target_now = np.full(n, np.nan)
    best_target_eta = np.full(n, np.nan)
    best_invalidate = np.full(n, np.nan)
    best_rr = np.full(n, np.nan)
    best_eta_bar = np.full(n, np.nan)
    best_watch = np.zeros(n, dtype=np.int64)
    tracked_score = np.zeros(n, dtype=np.int64)
    tracked_bull = np.zeros(n, dtype=np.int64)
    active_count = np.zeros(n, dtype=np.int64)
    ev_new_bull = np.zeros(n, dtype=np.int64)
    ev_new_bear = np.zeros(n, dtype=np.int64)
    ev_partial = np.zeros(n, dtype=np.int64)
    ev_target = np.zeros(n, dtype=np.int64)
    ev_invalidated = np.zeros(n, dtype=np.int64)
    ev_expired = np.zeros(n, dtype=np.int64)

    def add_swing(x, y, d):
        if not swings:
            swings.append([x, y, d])
            return
        last = swings[-1]
        if last[2] == d:
            stronger = (y > last[1]) if d == 1 else (y < last[1])
            if stronger:
                swings[-1] = [x, y, d]
        else:
            swings.append([x, y, d])
        while len(swings) > max_swings:
            swings.pop(0)

    def overlaps(a, b):
        margin_a = (a["x5"] - a["x1"]) / 3.0
        margin_b = (b["x5"] - b["x1"]) / 3.0
        margin = max(margin_a, margin_b)
        return a["bull"] == b["bull"] and max(a["x4"] - margin, b["x4"] - margin) <= min(a["x5"] + margin, b["x5"] + margin)

    def find_overlap_index(p):
        for i2, tp in enumerate(tracked):
            if overlaps(p, tp["p"]):
                return i2
        return -1

    def find_weakest_index():
        w, ms = -1, 1_000_000
        for i2, tp in enumerate(tracked):
            if tp["p"]["score"] < ms:
                ms = tp["p"]["score"]
                w = i2
        return w

    def find_best_active_index():
        b, ms = -1, -1
        for i2, tp in enumerate(tracked):
            if tp["p"]["score"] > ms:
                ms = tp["p"]["score"]
                b = i2
        return b

    def dedupe_overlaps():
        m = len(tracked)
        if m <= 1:
            return
        keep = [True] * m
        for a in range(m - 1):
            for b in range(a + 1, m):
                if keep[a] and keep[b] and overlaps(tracked[a]["p"], tracked[b]["p"]):
                    if tracked[a]["p"]["score"] >= tracked[b]["p"]["score"]:
                        keep[b] = False
                    else:
                        keep[a] = False
        for a in range(m - 1, -1, -1):
            if not keep[a]:
                tracked.pop(a)

    def bars_inside_metrics(p):
        ok = True
        max_violation = 0.0
        offender_x = None
        pattern_atr = p["patternAtr"]
        for bar in range(int(p["x1"]), int(p["x5"]) + 1):
            lvl13 = _price(p["x1"], p["y1"], p["x3"], p["y3"], bar)
            lvl24 = _price(p["x2"], p["y2"], p["x4"], p["y4"], bar)
            bar_high = high[bar]
            bar_low = low[bar]
            tol = pattern_atr * bar_boundary_tol_atr
            if bar == p["x5"]:
                viol13 = 0.0
            elif p["bull"]:
                viol13 = max(lvl13 - tol - bar_low, 0.0)
            else:
                viol13 = max(bar_high - lvl13 - tol, 0.0)
            if p["bull"]:
                viol24 = max(bar_high - lvl24 - tol, 0.0)
            else:
                viol24 = max(lvl24 - tol - bar_low, 0.0)
            viol = max(viol13, viol24)
            if viol > 0:
                ok = False
                if viol > max_violation:
                    max_violation = viol
                    offender_x = bar
        max_atr = (max_violation / pattern_atr) if pattern_atr > 0 else 1_000_000.0
        return ok, max_atr, offender_x

    def evaluate(p1, p2, p3, p4, p5, i1, i2, i3, i4, i5, bar_index):
        x1, x2, x3, x4, x5 = p1[0], p2[0], p3[0], p4[0], p5[0]
        y1, y2, y3, y4, y5 = p1[1], p2[1], p3[1], p4[1], p5[1]
        d1, d2, d3, d4, d5 = p1[2], p2[2], p3[2], p4[2], p5[2]

        atr_offset = bar_index - x5
        raw_atr = atr_a[atr_offset] if 0 <= atr_offset < n else _NAN
        pattern_atr = atr_a[bar_index] if math.isnan(raw_atr) else raw_atr

        bull_seq = d1 == -1 and d2 == 1 and d3 == -1 and d4 == 1 and d5 == -1
        bear_seq = d1 == 1 and d2 == -1 and d3 == 1 and d4 == -1 and d5 == 1
        bull_structure = bull_seq and y3 < y1 and y5 < y3
        bear_structure = bear_seq and y3 > y1 and y5 > y3
        is_bull, is_bear = bull_structure, bear_structure

        bull_channel = bull_seq and y4 > y1 and y4 < y2
        bear_channel = bear_seq and y4 < y1 and y4 > y2
        channel_ok = bull_channel or bear_channel

        line13_at5 = _price(x1, y1, x3, y3, x5)
        p5_offset = (line13_at5 - y5) if bull_seq else ((y5 - line13_at5) if bear_seq else 0.0)
        p5_offset_atr = (p5_offset / pattern_atr) if pattern_atr > 0 else 1_000_000.0
        p5_zone_ok = (bull_seq or bear_seq) and (-p5_inside_tol_atr <= p5_offset_atr <= p5_overshoot_tol_atr)
        if p5_offset_atr >= 0:
            p5_quality = 1.0 if p5_offset_atr <= 0.50 else 1.0 - _clamp01((p5_offset_atr - 0.50) / max(p5_overshoot_tol_atr - 0.50, 1e-6))
        else:
            if p5_offset_atr >= -0.25:
                p5_quality = 0.80 + 0.20 * _clamp01((p5_offset_atr + 0.25) / 0.25)
            else:
                p5_quality = 0.80 * (1.0 - _clamp01((-p5_offset_atr - 0.25) / max(p5_inside_tol_atr - 0.25, 1e-6)))

        line24_at1 = _price(x2, y2, x4, y4, x1)
        line24_at3 = _price(x2, y2, x4, y4, x3)
        line24_at5 = _price(x2, y2, x4, y4, x5)
        width_start = (line24_at1 - y1) if bull_seq else ((y1 - line24_at1) if bear_seq else -1.0)
        width_mid = (line24_at3 - y3) if bull_seq else ((y3 - line24_at3) if bear_seq else -1.0)
        width_end = (line24_at5 - y5) if bull_seq else ((y5 - line24_at5) if bear_seq else -1.0)

        contraction_ratio = (width_end / width_start) if width_start > 0 else 1_000_000.0
        width_start_atr = (width_start / pattern_atr) if pattern_atr > 0 else 1_000_000.0
        width_mid_atr = (width_mid / pattern_atr) if pattern_atr > 0 else 1_000_000.0
        width_end_atr = (width_end / pattern_atr) if pattern_atr > 0 else 1_000_000.0
        contraction = (width_start > pattern_atr * 0.5 and width_mid > 0 and width_end >= 0
                       and width_mid <= width_start and width_end <= width_mid and contraction_ratio <= 0.70)
        contraction_quality = _clamp01((1.0 - contraction_ratio) / 0.30) if width_start > 0 else 0.0

        m13 = _slope(x1, y1, x3, y3)
        m24 = _slope(x2, y2, x4, y4)
        same_direction = (not math.isnan(m13)) and (not math.isnan(m24)) and _sgn(m13) == _sgn(m24)
        slope_ratio = abs(m13) / abs(m24) if ((not math.isnan(m13)) and (not math.isnan(m24)) and abs(m24) > 1e-7) else _NAN
        slopes_comparable = (not math.isnan(slope_ratio)) and 0.3 <= slope_ratio <= 3.0

        line13_at4 = _price(x1, y1, x3, y3, x4)
        p4_width = (y4 - line13_at4) if bull_seq else ((line13_at4 - y4) if bear_seq else -1.0)
        channel_width_at5 = (line24_at5 - line13_at5) if bull_seq else ((line13_at5 - line24_at5) if bear_seq else -1.0)
        p4_width_atr = (p4_width / pattern_atr) if pattern_atr > 0 else 1_000_000.0
        channel_width_at5_atr = (channel_width_at5 / pattern_atr) if pattern_atr > 0 else 1_000_000.0
        if channel_width_at5 > 0:
            p5_pos_from13 = (y5 - line13_at5) / channel_width_at5 if bull_seq else (line13_at5 - y5) / channel_width_at5
        else:
            p5_pos_from13 = 0.5
        traversal45 = _clamp01(abs(1.0 - p5_pos_from13))
        channel_ok = channel_ok and p4_width >= pattern_atr * 0.50 and channel_width_at5 > 0 and traversal45 >= 0.60
        channel_quality = (0.40 + 0.30 * _clamp01(p4_width / max(pattern_atr, 1e-6)) + 0.30 * traversal45
                           if (bull_channel or bear_channel) else 0.0)

        xi = _intersect_x(x1, y1, x3, y3, x2, y2, x4, y4)
        width_bars = x5 - x1
        max_forward = width_bars * 2
        eta_ok = (xi is not None) and (xi > x5) and (xi <= x5 + max_forward)
        eta_distance_ratio = ((xi - x5) / width_bars) if ((xi is not None) and width_bars > 0) else 1_000_000.0
        eta_quality = _clamp01(1.0 - abs(eta_distance_ratio - 0.75) / 1.25) if eta_ok else 0.0
        xi_bounded = xi if eta_ok else x5 + max_forward

        target_now = _price(x1, y1, x4, y4, x5)
        target_eta = _price(x1, y1, x4, y4, xi_bounded)

        if is_bull:
            invalidate = min(y5, line13_at5) - pattern_atr * invalidation_atr
        elif is_bear:
            invalidate = max(y5, line13_at5) + pattern_atr * invalidation_atr
        else:
            invalidate = _NAN
        risk = abs(y5 - invalidate) if not math.isnan(invalidate) else 0.0
        reward = abs(target_now - y5)
        reward_dir_ok = (bull_seq and target_now > y5) or (bear_seq and target_now < y5)
        target_distance_ok = reward <= pattern_atr * max_target_distance
        risk_reward = (reward / risk) if risk > 0 else 0.0
        reward_ok = reward_dir_ok and target_distance_ok and risk_reward >= min_rr
        reward_quality = _clamp01(risk_reward / 2.0) if (reward_dir_ok and target_distance_ok) else 0.0

        t12, t23, t34, t45 = x2 - x1, x3 - x2, x4 - x3, x5 - x4
        total_time = max(1, t12 + t23 + t34 + t45)
        max_leg = max(max(t12, t23), max(t34, t45))
        no_single_dominates = float(max_leg) / float(total_time) <= 0.4
        timing = (t12 > 0 and t23 > 0 and t34 > 0 and t45 > 0
                  and t45 <= (t12 + t23 + t34) * 1.5 and no_single_dominates)
        symmetry_raw = abs(t12 - t34) + abs(t23 - t45)
        symmetry_base = max(1, t12 + t23 + t34 + t45)
        symmetry_ratio = float(symmetry_raw) / float(symmetry_base)
        symmetry_ok = symmetry_ratio < 0.45

        max_skipped = 0.0
        purity_offender_x = None
        purity_offender_leg = 0
        if i5 > i1 + 1:
            for k in range(i1 + 1, i5):
                if k == i2 or k == i3 or k == i4:
                    continue
                s = swings[k]
                if k < i2:
                    lx1, ly1, lx2, ly2 = x1, y1, x2, y2
                elif k < i3:
                    lx1, ly1, lx2, ly2 = x2, y2, x3, y3
                elif k < i4:
                    lx1, ly1, lx2, ly2 = x3, y3, x4, y4
                else:
                    lx1, ly1, lx2, ly2 = x4, y4, x5, y5
                leg_amp = abs(ly2 - ly1)
                skipped = abs(s[1] - _price(lx1, ly1, lx2, ly2, s[0])) / leg_amp if leg_amp > 0 else 1_000_000.0
                if skipped > max_skipped:
                    max_skipped = skipped
                    purity_offender_x = s[0]
                    purity_offender_leg = 1 if k < i2 else 2 if k < i3 else 3 if k < i4 else 4
        leg_purity_ok = max_skipped <= max_leg_purity_ratio
        purity_quality = 1.0 - _clamp01(max_skipped / max_leg_purity_ratio)

        structure_ok = is_bull or is_bear
        contraction_ok = contraction and same_direction and slopes_comparable

        boundary_hold_ok = True
        swing_tol = pattern_atr * swing_boundary_tol_atr
        max_swing_violation = 0.0
        swing_violation_x = None
        if i5 > i1 + 1:
            for k in range(i1 + 1, i5):
                s = swings[k]
                lvl13 = _price(x1, y1, x3, y3, s[0])
                lvl24 = _price(x2, y2, x4, y4, s[0])
                if bull_seq:
                    v13 = max(lvl13 - swing_tol - s[1], 0.0)
                    v24 = max(s[1] - lvl24 - swing_tol, 0.0)
                elif bear_seq:
                    v13 = max(s[1] - lvl13 - swing_tol, 0.0)
                    v24 = max(lvl24 - swing_tol - s[1], 0.0)
                else:
                    v13 = v24 = 0.0
                viol = max(v13, v24)
                if viol > 0:
                    boundary_hold_ok = False
                    if viol > max_swing_violation:
                        max_swing_violation = viol
                        swing_violation_x = s[0]
        swing_violation_atr = (max_swing_violation / pattern_atr) if pattern_atr > 0 else 1_000_000.0

        structure_score = 20 if structure_ok else 0
        contraction_score = int(round(15.0 * (contraction_quality if contraction_ok else contraction_quality * 0.50)))
        channel_score = int(round(10.0 * _clamp01(channel_quality)))
        eta_score = int(round(10.0 * eta_quality))
        p5_score = int(round(15.0 * (_clamp01(p5_quality) if p5_zone_ok else 0.0)))
        reward_score = int(round(15.0 * reward_quality))
        max_leg_share = float(max_leg) / float(total_time)
        timing_quality = _clamp01((0.50 - max_leg_share) / 0.25) if timing else 0.0
        timing_score = int(round(5.0 * timing_quality))
        symmetry_score = int(round(5.0 * (1.0 - _clamp01(symmetry_ratio / 0.45))))
        purity_score = int(round(5.0 * purity_quality))

        raw_score = (contraction_score + channel_score + eta_score + p5_score + reward_score
                     + timing_score + symmetry_score + purity_score)
        score = int(round(float(raw_score) * 100.0 / 80.0))

        valid = (score >= min_score and structure_ok and contraction_ok and channel_ok
                 and boundary_hold_ok and leg_purity_ok and p5_zone_ok and eta_ok
                 and (not require_reward or reward_ok))

        pat = _empty_pattern()
        pat.update({
            "valid": valid, "bull": is_bull, "score": score,
            "x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5,
            "y1": y1, "y2": y2, "y3": y3, "y4": y4, "y5": y5,
            "targetNow": target_now, "targetETA": target_eta, "invalidate": invalidate,
            "patternAtr": pattern_atr, "riskReward": risk_reward, "legPurityRatio": max_skipped,
            "contractionRatio": contraction_ratio, "widthStartAtr": width_start_atr,
            "widthMidAtr": width_mid_atr, "widthEndAtr": width_end_atr,
            "sameDirectionOk": same_direction, "slopesComparableOk": slopes_comparable,
            "slopeRatio": slope_ratio, "p4WidthAtr": p4_width_atr,
            "channelWidthAt5Atr": channel_width_at5_atr, "traversal45": traversal45,
            "p5OffsetAtr": p5_offset_atr, "purityOffenderX": purity_offender_x,
            "purityOffenderLeg": purity_offender_leg,
            "swingBoundaryViolationAtr": swing_violation_atr, "swingBoundaryViolationX": swing_violation_x,
            "barBoundaryViolationAtr": 0.0, "etaBar": xi_bounded,
            "structureOk": structure_ok, "contractionOk": contraction_ok, "channelOk": channel_ok,
            "etaOk": eta_ok, "p5ZoneOk": p5_zone_ok, "rewardOk": reward_ok, "timingOk": timing,
            "symmetryOk": symmetry_ok, "legPurityOk": leg_purity_ok, "boundaryHoldOk": boundary_hold_ok,
            "barsInsideOk": True, "structureScore": structure_score, "contractionScore": contraction_score,
            "channelScore": channel_score, "etaScore": eta_score, "p5Score": p5_score,
            "rewardScore": reward_score, "timingScore": timing_score, "symmetryScore": symmetry_score,
            "purityScore": purity_score,
        })
        return pat

    for i in range(n):
        if not math.isnan(ph[i]):
            add_swing(i - pivot_len, ph[i], 1)
        if not math.isnan(pl[i]):
            add_swing(i - pivot_len, pl[i], -1)

        sz = len(swings)
        cur_x = swings[-1][0] if sz > 0 else -1
        new_swing = cur_x != last_swing_x
        last_swing_x = cur_x

        new_bull = new_bear = False

        if new_swing and sz >= 5:
            start = max(0, sz - search_depth)
            per_i5 = []
            local_best = _empty_pattern()
            for i5 in range(start + 4, sz):
                dir5 = swings[i5][2]
                same_idx = [k for k in range(start, i5) if swings[k][2] == dir5]
                opp_idx = [k for k in range(start, i5) if swings[k][2] != dir5]
                local_best_i5 = _empty_pattern()
                local_best_raw = _empty_pattern()
                if len(same_idx) >= 2 and len(opp_idx) >= 2:
                    for a1 in range(len(same_idx) - 1):
                        i1 = same_idx[a1]
                        for a3 in range(a1 + 1, len(same_idx)):
                            i3 = same_idx[a3]
                            for a2 in range(len(opp_idx)):
                                i2 = opp_idx[a2]
                                if not (i1 < i2 < i3):
                                    continue
                                for a4 in range(len(opp_idx)):
                                    i4 = opp_idx[a4]
                                    if not (i3 < i4 < i5):
                                        continue
                                    cand = evaluate(swings[i1], swings[i2], swings[i3], swings[i4], swings[i5],
                                                    i1, i2, i3, i4, i5, i)
                                    if cand["score"] > local_best_raw["score"]:
                                        local_best_raw = cand
                                    if cand["valid"] and cand["score"] > local_best_i5["score"]:
                                        local_best_i5 = cand
                if local_best_i5["score"] == 0:
                    local_best_i5 = local_best_raw
                if local_best_i5["structureOk"]:
                    ok, viol_atr, off_x = bars_inside_metrics(local_best_i5)
                    local_best_i5["barsInsideOk"] = ok
                    local_best_i5["barBoundaryViolationAtr"] = viol_atr
                    local_best_i5["barBoundaryViolationX"] = off_x
                    if not ok:
                        local_best_i5["valid"] = False
                if local_best_i5["score"] > local_best["score"]:
                    local_best = local_best_i5
                per_i5.append(local_best_i5)
            best = local_best

            for c in per_i5:
                if not c["valid"]:
                    continue
                already_aged = i > c["etaBar"] + expiry_buffer_bars
                arrival_level = c["invalidate"]
                if i <= c["x5"] + dynamic_invalidation_bars:
                    arrival_line13 = _price(c["x1"], c["y1"], c["x3"], c["y3"], i)
                    arrival_level = (min(c["y5"], arrival_line13) - c["patternAtr"] * invalidation_atr
                                     if c["bull"] else max(c["y5"], arrival_line13) + c["patternAtr"] * invalidation_atr)
                if invalidate_on_close:
                    arrival_invalid = close[i] < arrival_level if c["bull"] else close[i] > arrival_level
                else:
                    arrival_invalid = low[i] < arrival_level if c["bull"] else high[i] > arrival_level
                arrival_epa = _price(c["x1"], c["y1"], c["x4"], c["y4"], i)
                arrival_target = high[i] >= arrival_epa if c["bull"] else low[i] <= arrival_epa
                still_open = (not arrival_invalid) and (not arrival_target)
                if not (still_open and not already_aged):
                    continue
                ov = find_overlap_index(c)
                if ov < 0:
                    if len(tracked) < max_concurrent_patterns:
                        tracked.append({"p": c, "invalidateLevel": c["invalidate"], "state": "ACTIVE",
                                        "partialHit": False, "createdBar": i})
                        new_bull = new_bull or c["bull"]
                        new_bear = new_bear or (not c["bull"])
                    else:
                        wi = find_weakest_index()
                        if c["score"] > tracked[wi]["p"]["score"]:
                            tracked[wi] = {"p": c, "invalidateLevel": c["invalidate"], "state": "ACTIVE",
                                           "partialHit": False, "createdBar": i}
                            new_bull = new_bull or c["bull"]
                            new_bear = new_bear or (not c["bull"])
                else:
                    existing = tracked[ov]
                    if c["x5"] != existing["p"]["x5"] and c["score"] > existing["p"]["score"]:
                        tracked[ov] = {"p": c, "invalidateLevel": c["invalidate"], "state": "ACTIVE",
                                       "partialHit": False, "createdBar": i}
                        new_bull = new_bull or c["bull"]
                        new_bear = new_bear or (not c["bull"])
            dedupe_overlaps()

        # ---- lifecycle ----
        any_invalidated = any_target = any_partial = any_expired = False
        for t in range(len(tracked) - 1, -1, -1):
            tp = tracked[t]
            p = tp["p"]
            if i <= p["x5"] + dynamic_invalidation_bars:
                line13_now = _price(p["x1"], p["y1"], p["x3"], p["y3"], i)
                tp["invalidateLevel"] = (min(p["y5"], line13_now) - p["patternAtr"] * invalidation_atr
                                         if p["bull"] else max(p["y5"], line13_now) + p["patternAtr"] * invalidation_atr)
            level = tp["invalidateLevel"]
            if invalidate_on_close:
                invalidated = close[i] < level if p["bull"] else close[i] > level
            else:
                invalidated = low[i] < level if p["bull"] else high[i] > level
            epa_now = _price(p["x1"], p["y1"], p["x4"], p["y4"], i)
            target_hit = high[i] >= epa_now if p["bull"] else low[i] <= epa_now
            partial_level = p["y5"] + (epa_now - p["y5"]) * 0.50
            if p["bull"]:
                partial_now = (not tp["partialHit"]) and high[i] >= partial_level
            else:
                partial_now = (not tp["partialHit"]) and low[i] <= partial_level
            expired = i > p["etaBar"] + expiry_buffer_bars
            just_created = i <= tp["createdBar"]
            if just_created:
                invalidated = target_hit = partial_now = expired = False
            if invalidated:
                tp["state"] = "INVALIDATED"
                any_invalidated = True
                tracked.pop(t)
            elif target_hit:
                tp["state"] = "TARGET_HIT"
                any_target = True
                tracked.pop(t)
            elif expired:
                tp["state"] = "EXPIRED"
                any_expired = True
                tracked.pop(t)
            else:
                if partial_now:
                    tp["state"] = "PARTIAL_SUCCESS"
                    tp["partialHit"] = True
                    any_partial = True

        # ---- output ----
        best_score[i] = best["score"]
        best_bull[i] = 1 if best["bull"] else 0
        best_structure[i] = 1 if best["structureOk"] else 0
        best_valid[i] = 1 if best["valid"] else 0
        best_target_now[i] = best["targetNow"]
        best_target_eta[i] = best["targetETA"]
        best_invalidate[i] = best["invalidate"]
        best_rr[i] = best["riskReward"]
        best_eta_bar[i] = best["etaBar"]
        best_watch[i] = 1 if (show_candidate and best["structureOk"] and best["score"] >= candidate_threshold
                              and not best["valid"]) else 0
        bi = find_best_active_index()
        if bi >= 0:
            tracked_score[i] = tracked[bi]["p"]["score"]
            tracked_bull[i] = 1 if tracked[bi]["p"]["bull"] else 0
        active_count[i] = len(tracked)
        ev_new_bull[i] = 1 if new_bull else 0
        ev_new_bear[i] = 1 if new_bear else 0
        ev_partial[i] = 1 if any_partial else 0
        ev_target[i] = 1 if any_target else 0
        ev_invalidated[i] = 1 if any_invalidated else 0
        ev_expired[i] = 1 if any_expired else 0

    out = pd.DataFrame(index=idx)
    out["ww_best_score"] = pd.Series(best_score, index=idx).astype("Int64")
    out["ww_best_bull"] = pd.Series(best_bull, index=idx).astype("Int64")
    out["ww_best_structure"] = pd.Series(best_structure, index=idx).astype("Int64")
    out["ww_best_valid"] = pd.Series(best_valid, index=idx).astype("Int64")
    out["ww_best_target_now"] = best_target_now
    out["ww_best_target_eta"] = best_target_eta
    out["ww_best_invalidate"] = best_invalidate
    out["ww_best_rr"] = best_rr
    out["ww_best_eta_bar"] = best_eta_bar
    out["ww_best_watch"] = pd.Series(best_watch, index=idx).astype("Int64")
    out["ww_tracked_score"] = pd.Series(tracked_score, index=idx).astype("Int64")
    out["ww_tracked_bull"] = pd.Series(tracked_bull, index=idx).astype("Int64")
    out["ww_active_count"] = pd.Series(active_count, index=idx).astype("Int64")
    out["ww_new_bull"] = pd.Series(ev_new_bull, index=idx).astype("Int64")
    out["ww_new_bear"] = pd.Series(ev_new_bear, index=idx).astype("Int64")
    out["ww_partial"] = pd.Series(ev_partial, index=idx).astype("Int64")
    out["ww_target_hit"] = pd.Series(ev_target, index=idx).astype("Int64")
    out["ww_invalidated"] = pd.Series(ev_invalidated, index=idx).astype("Int64")
    out["ww_expired"] = pd.Series(ev_expired, index=idx).astype("Int64")
    return out
