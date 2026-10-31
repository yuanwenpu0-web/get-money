"""Broadening Wedge Scanner Pro [WavesUnchained]  v1.15  —  Python port.

Slides a fixed-size window of confirmed swings across the search depth and fits
two boundary lines (upper from the window's outer highs, lower from its outer
lows) to find EXPANDING wedges:

  * Descending broadening  — both lines fall, lower faster (bullish bias)
  * Ascending broadening   — both lines rise, upper faster (bearish bias)
  * Symmetric broadening   — upper rising, lower falling ("megaphone")

Each valid wedge is scored 0-100 from eight weighted components (traversal,
realized expansion, extra touches, line quality, anchor span, duration,
relevance, plus a non-scored path gate) and tracked until either boundary
breaks (live breakout) or it ages out.

Computational core ported (per-bar state machine):
  * Confirmed-swing engine (pivothigh/low + same-direction extremum replacement)
  * Sliding-window wedge search across the swing search depth
  * ``f_best_boundary`` two-real-touch boundary fit (fewest violations, then most
    touches, then widest span) -- the chartist's trendline, not a LSQ average
  * ``f_eval_wedge`` full geometric classification + 8-component score + gates
  * Multi-pattern tracking (insert / evict-weakest / replace / dedupe overlaps)
  * Per-bar breakout & age resolution

Dropped (display / logging / non-computational):
  * box / line / label drawing, dashboard & debug tables, candidate marker
    (``plotshape``), ``alertcondition``, ``log.info``
  * ``lineAheadBars`` / ``showOldPatterns`` / colour / size inputs (visual only)

Pine exposes *no* numeric series for this indicator (it only draws), so the
port returns the scanner's per-bar state as columns: the best-in-depth
candidate (score / type / validity / key ratios) and the best currently
tracked (active) pattern, plus the live breakout signal.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import atr, integer, number, pivothigh_strict, pivotlow_strict

_TYPE_CODE = {
    "None": 0,
    "Descending Broadening": 1,
    "Ascending Broadening": 2,
    "Symmetric Broadening": 3,
}
_TYPE_NAME = {v: k for k, v in _TYPE_CODE.items()}


# --------------------------------------------------------------------------- #
# Small helpers (mirror the Pine f_* helpers)                                  #
# --------------------------------------------------------------------------- #
def _f_price(x1, y1, x2, y2, x):
    return y1 if x2 == x1 else y1 + (y2 - y1) * float(x - x1) / float(x2 - x1)


def _f_slope(x1, y1, x2, y2):
    return float("nan") if x2 == x1 else (y2 - y1) / float(x2 - x1)


def _f_clamp01(v):
    return min(max(v, 0.0), 1.0)


def _empty_pattern():
    return {
        "valid": False, "ttype": "None", "score": 0,
        "xU1": None, "xU2": None, "yU1": None, "yU2": None,
        "xL1": None, "xL2": None, "yL1": None, "yL2": None,
        "windowStartX": 0, "windowEndX": 0,
        "structureOk": False, "pathOk": False, "traversalOk": False,
        "realizedExpansionOk": False, "boundaryOk": False, "lineQualityOk": False,
        "durationOk": False, "barsInsideOk": False, "boundariesOrderedOk": False,
        "widthOk": False, "anchorSpanOk": False, "driftOk": False,
        "upperAnchorSpanRatio": float("nan"), "lowerAnchorSpanRatio": float("nan"),
        "upperDriftAtr": float("nan"), "lowerDriftAtr": float("nan"),
        "containmentRatio": float("nan"), "progressionMatches": 0, "progressionTests": 0,
        "realizedExpansionRatio": float("nan"), "legGrowthMatches": 0, "legGrowthTests": 0,
        "legGrowthRatio": float("nan"), "projectedExpansionRatio": float("nan"),
        "pathScore": 0, "traversalScore": 0, "rangeScore": 0, "touchScore": 0,
        "lineScore": 0, "anchorScore": 0, "durationScore": 0, "relevanceScore": 0,
    }


def _overlaps(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) <= min(a_end, b_end)


def _best_boundary(xs, ys, tol_price, is_upper):
    """Two-real-touch boundary fit (fewest violations -> most touches -> widest)."""
    n = len(xs)
    best_i, best_j = 0, n - 1
    best_viol, best_touch, best_span = 1_000_000, -1, -1
    for i in range(n - 1):
        xi, yi = xs[i], ys[i]
        for j in range(i + 1, n):
            xj, yj = xs[j], ys[j]
            touches = 0
            violations = 0
            for k in range(n):
                xk, yk = xs[k], ys[k]
                dist = yk - _f_price(xi, yi, xj, yj, xk)
                if abs(dist) <= tol_price:
                    touches += 1
                elif is_upper and dist > tol_price:
                    violations += 1
                elif (not is_upper) and dist < -tol_price:
                    violations += 1
            span = xj - xi
            better = (violations < best_viol) or (
                violations == best_viol and (
                    touches > best_touch or (touches == best_touch and span > best_span)
                )
            )
            if better:
                best_viol, best_touch, best_span = violations, touches, span
                best_i, best_j = i, j
    return xs[best_i], ys[best_i], xs[best_j], ys[best_j]


def _eval_wedge(high, low, close, atr_i, pathX, pathY, pathDir, highX, highY,
                lowX, lowY, window_start_x, window_end_x, search_bar, P):
    nH, nL, nPath = len(highX), len(lowX), len(pathX)
    touch_tol_price = P["touch_tol_atr"] * (atr_i if not math.isnan(atr_i) else 0.0)

    xU1, yU1, xU2, yU2 = _best_boundary(highX, highY, touch_tol_price, True)
    xL1, yL1, xL2, yL2 = _best_boundary(lowX, lowY, touch_tol_price, False)

    slopeU = _f_slope(xU1, yU1, xU2, yU2)
    slopeL = _f_slope(xL1, yL1, xL2, yL2)

    both_falling = (not math.isnan(slopeU)) and (not math.isnan(slopeL)) and slopeU < 0 and slopeL < 0
    both_rising = (not math.isnan(slopeU)) and (not math.isnan(slopeL)) and slopeU > 0 and slopeL > 0
    diverging = (not math.isnan(slopeU)) and (not math.isnan(slopeL)) and slopeU > 0 and slopeL < 0

    is_desc = both_falling and abs(slopeL) > abs(slopeU)
    is_asc = both_rising and abs(slopeU) > abs(slopeL)
    is_sym = diverging

    ttype = ("Descending Broadening" if is_desc else
             "Ascending Broadening" if is_asc else
             "Symmetric Broadening" if is_sym else "None")
    structure_ok = is_desc or is_asc or is_sym

    mid_x = int(round((window_start_x + window_end_x) / 2.0))
    upper_start = _f_price(xU1, yU1, xU2, yU2, window_start_x)
    lower_start = _f_price(xL1, yL1, xL2, yL2, window_start_x)
    upper_mid = _f_price(xU1, yU1, xU2, yU2, mid_x)
    lower_mid = _f_price(xL1, yL1, xL2, yL2, mid_x)
    upper_end = _f_price(xU1, yU1, xU2, yU2, window_end_x)
    lower_end = _f_price(xL1, yL1, xL2, yL2, window_end_x)

    width_start = abs(upper_start - lower_start)
    width_mid = abs(upper_mid - lower_mid)
    width_end = abs(upper_end - lower_end)

    boundaries_ordered = (upper_start > lower_start and upper_mid > lower_mid and upper_end > lower_end)
    projected_expansion_ratio = width_end / width_start if width_start > 0 else 0.0

    progression_tests = progression_matches = 0
    alternating_path = True
    if nPath > 1:
        for i in range(1, nPath):
            if pathDir[i] == pathDir[i - 1]:
                alternating_path = False
    if nH > 1:
        for i in range(1, nH):
            prev, cur = highY[i - 1], highY[i]
            expected_up = is_asc or is_sym
            progression_tests += 1
            if (expected_up and cur > prev) or ((not expected_up) and cur < prev):
                progression_matches += 1
    if nL > 1:
        for i in range(1, nL):
            prev, cur = lowY[i - 1], lowY[i]
            expected_down = is_desc or is_sym
            progression_tests += 1
            if (expected_down and cur < prev) or ((not expected_down) and cur > prev):
                progression_matches += 1
    progression_ratio = progression_matches / progression_tests if progression_tests > 0 else 0.0
    path_ok = (structure_ok and alternating_path and nPath >= 6 and nH >= 3 and nL >= 3
               and progression_matches == progression_tests)
    path_score = int(round(20.0 * progression_ratio)) if (structure_ok and alternating_path) else 0

    traversal_sum = traversal_count = qualifying_traversals = 0
    longest_run = current_run = 0
    leg_amplitudes = []
    if nPath > 1:
        for i in range(1, nPath):
            x1, y1 = pathX[i - 1], pathY[i - 1]
            x2, y2 = pathX[i], pathY[i]
            u1 = _f_price(xU1, yU1, xU2, yU2, x1)
            l1 = _f_price(xL1, yL1, xL2, yL2, x1)
            u2 = _f_price(xU1, yU1, xU2, yU2, x2)
            l2 = _f_price(xL1, yL1, xL2, yL2, x2)
            w1 = u1 - l1
            w2 = u2 - l2
            pos1 = (y1 - l1) / w1 if w1 > 0 else 0.5
            pos2 = (y2 - l2) / w2 if w2 > 0 else 0.5
            traversal = _f_clamp01(abs(pos2 - pos1))
            traversal_sum += traversal
            traversal_count += 1
            leg_amplitudes.append(abs(y2 - y1))
            if traversal >= P["min_traversal_ratio"]:
                qualifying_traversals += 1
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0
    avg_traversal = traversal_sum / traversal_count if traversal_count > 0 else 0.0
    qualifying_ratio = qualifying_traversals / traversal_count if traversal_count > 0 else 0.0
    traversal_ok = longest_run >= 3
    traversal_score = int(round(20.0 * (0.65 * _f_clamp01(avg_traversal / P["min_traversal_ratio"])
                                       + 0.35 * qualifying_ratio)))

    leg_count = len(leg_amplitudes)
    first_avg = (leg_amplitudes[0] + leg_amplitudes[1]) / 2.0 if leg_count >= 2 else 0.0
    last_avg = ((leg_amplitudes[leg_count - 2] + leg_amplitudes[leg_count - 1]) / 2.0) if leg_count >= 2 else 0.0
    realized_expansion_ratio = last_avg / first_avg if first_avg > 0 else 0.0
    leg_growth_tests = leg_growth_matches = 0
    prev_up_leg = float("nan")
    prev_down_leg = float("nan")
    if leg_count > 0:
        for i in range(leg_count):
            amp = leg_amplitudes[i]
            ends_high = pathDir[i + 1] == 1
            prev_same = prev_up_leg if ends_high else prev_down_leg
            if not math.isnan(prev_same):
                leg_growth_tests += 1
                if amp >= prev_same * 0.95:
                    leg_growth_matches += 1
            if ends_high:
                prev_up_leg = amp
            else:
                prev_down_leg = amp
    leg_growth_ratio = leg_growth_matches / leg_growth_tests if leg_growth_tests > 0 else 0.0
    realized_expansion_ok = (leg_count >= 5 and realized_expansion_ratio >= P["min_realized_expansion_ratio"]
                             and leg_growth_ratio >= 0.60)
    realized_quality = (_f_clamp01((realized_expansion_ratio - 1.0) / (P["min_realized_expansion_ratio"] - 1.0))
                        if P["min_realized_expansion_ratio"] > 1.0 else _f_clamp01(realized_expansion_ratio))
    projected_quality = _f_clamp01((projected_expansion_ratio - 1.0) / (P["projected_expansion_target"] - 1.0))
    range_score = int(round(15.0 * (0.70 * realized_quality + 0.20 * leg_growth_ratio + 0.10 * projected_quality)))

    window_bars = window_end_x - window_start_x
    upper_anchor_span = float(xU2 - xU1) / float(window_bars) if window_bars > 0 else 0.0
    lower_anchor_span = float(xL2 - xL1) / float(window_bars) if window_bars > 0 else 0.0
    atr_pos = atr_i if (not math.isnan(atr_i) and atr_i > 0) else 1e-9
    upper_drift = (abs(slopeU) * float(window_bars) / atr_pos) if (not math.isnan(slopeU) and atr_i > 0) else 1_000_000.0
    lower_drift = (abs(slopeL) * float(window_bars) / atr_pos) if (not math.isnan(slopeL) and atr_i > 0) else 1_000_000.0
    width_ok = width_end > P["min_width_atr"] * atr_i if (not math.isnan(atr_i)) else False
    anchor_span_ok = upper_anchor_span >= P["min_anchor_span_ratio"] and lower_anchor_span >= P["min_anchor_span_ratio"]
    drift_ok = upper_drift <= P["max_boundary_drift_atr"] and lower_drift <= P["max_boundary_drift_atr"]
    boundary_ok = boundaries_ordered and width_ok and anchor_span_ok and drift_ok
    anchor_score = int(round(10.0 * (_f_clamp01(upper_anchor_span) + _f_clamp01(lower_anchor_span)) / 2.0))

    extra_high_touches = high_residual_sum = high_violations = 0
    if nH > 0:
        for i in range(nH):
            hx, hy = highX[i], highY[i]
            line_y = _f_price(xU1, yU1, xU2, yU2, hx)
            dist = hy - line_y
            high_residual_sum += abs(dist)
            if abs(hy - line_y) <= touch_tol_price:
                if hx != xU1 and hx != xU2:
                    extra_high_touches += 1
            elif dist > touch_tol_price:
                high_violations += 1
    extra_low_touches = low_residual_sum = low_violations = 0
    if nL > 0:
        for i in range(nL):
            lx, ly = lowX[i], lowY[i]
            line_y = _f_price(xL1, yL1, xL2, yL2, lx)
            dist = ly - line_y
            low_residual_sum += abs(dist)
            if abs(ly - line_y) <= touch_tol_price:
                if lx != xL1 and lx != xL2:
                    extra_low_touches += 1
            elif dist < -touch_tol_price:
                low_violations += 1
    extra_high_quality = _f_clamp01(float(extra_high_touches) / 2.0)
    extra_low_quality = _f_clamp01(float(extra_low_touches) / 2.0)
    touch_score = int(round(15.0 * (extra_high_quality + extra_low_quality) / 2.0))

    faster_ratio = (abs(slopeL) / abs(slopeU) if (is_desc and abs(slopeU) > 1e-7) else
                    abs(slopeU) / abs(slopeL) if (is_asc and abs(slopeL) > 1e-7) else float("nan"))
    sym_ratio = (abs(slopeU) / abs(slopeL) if (is_sym and abs(slopeL) > 1e-7) else float("nan"))
    slope_quality_ok = (
        (not math.isnan(faster_ratio) and faster_ratio >= P["faster_ratio_min"])
        if (is_desc or is_asc) else
        (not math.isnan(sym_ratio) and 0.3 <= sym_ratio <= 3.0) if is_sym else False
    )
    duration_ok = window_bars >= P["pivot_len"] * 3
    duration_score = int(round(5.0 * _f_clamp01(float(window_bars) / float(P["pivot_len"] * 6))))

    bars_inside = mid_zone = evaluated = 0
    for k in range(window_start_x, window_end_x + 1):
        upper_at = _f_price(xU1, yU1, xU2, yU2, k)
        lower_at = _f_price(xL1, yL1, xL2, yL2, k)
        bar_high = high[k]
        bar_low = low[k]
        bar_close = close[k]
        evaluated += 1
        if bar_high <= upper_at + touch_tol_price and bar_low >= lower_at - touch_tol_price:
            bars_inside += 1
        local_width = upper_at - lower_at
        if local_width > 0:
            close_pos = (bar_close - lower_at) / local_width
            if 0.30 <= close_pos <= 0.70:
                mid_zone += 1
    containment_ratio = bars_inside / evaluated if evaluated > 0 else 0.0
    mid_zone_ratio = mid_zone / evaluated if evaluated > 0 else 1.0
    bars_inside_ok = containment_ratio >= 0.90

    point_count = nH + nL
    mean_residual = ((high_residual_sum + low_residual_sum) / point_count) if point_count > 0 else touch_tol_price * 2.0
    residual_quality = (1.0 - _f_clamp01(mean_residual / (touch_tol_price * 2.0))) if touch_tol_price > 0 else 0.0
    violation_ratio = ((high_violations + low_violations) / point_count) if point_count > 0 else 1.0
    line_score = int(round(10.0 * (0.40 * residual_quality + 0.35 * containment_ratio
                                   + 0.15 * (1.0 - violation_ratio) + 0.10 * (1.0 - mid_zone_ratio))))
    line_quality_ok = slope_quality_ok

    max_relevant_age = P["max_pattern_age_mult"] * float(max(window_bars, 1))
    candidate_age = max(search_bar - window_end_x, 0)
    relevance_score = int(round(5.0 * (1.0 - _f_clamp01(candidate_age / max_relevant_age))
                                if max_relevant_age > 0 else 0.0))

    raw_score = (traversal_score + range_score + touch_score + line_score
                 + anchor_score + duration_score + relevance_score)
    score = int(round(float(raw_score) * 100.0 / 80.0))
    hard_gates_ok = (structure_ok and path_ok and traversal_ok and realized_expansion_ok
                     and boundary_ok and line_quality_ok and duration_ok)

    pat = _empty_pattern()
    pat.update({
        "valid": score >= P["min_score"] and hard_gates_ok,
        "ttype": ttype, "score": score,
        "xU1": xU1, "xU2": xU2, "yU1": yU1, "yU2": yU2,
        "xL1": xL1, "xL2": xL2, "yL1": yL1, "yL2": yL2,
        "windowStartX": window_start_x, "windowEndX": window_end_x,
        "structureOk": structure_ok, "pathOk": path_ok, "traversalOk": traversal_ok,
        "realizedExpansionOk": realized_expansion_ok, "boundaryOk": boundary_ok,
        "lineQualityOk": line_quality_ok, "durationOk": duration_ok, "barsInsideOk": bars_inside_ok,
        "boundariesOrderedOk": boundaries_ordered, "widthOk": width_ok,
        "anchorSpanOk": anchor_span_ok, "driftOk": drift_ok,
        "upperAnchorSpanRatio": upper_anchor_span, "lowerAnchorSpanRatio": lower_anchor_span,
        "upperDriftAtr": upper_drift, "lowerDriftAtr": lower_drift,
        "containmentRatio": containment_ratio, "progressionMatches": progression_matches,
        "progressionTests": progression_tests, "realizedExpansionRatio": realized_expansion_ratio,
        "legGrowthMatches": leg_growth_matches, "legGrowthTests": leg_growth_tests,
        "legGrowthRatio": leg_growth_ratio, "projectedExpansionRatio": projected_expansion_ratio,
        "pathScore": path_score, "traversalScore": traversal_score, "rangeScore": range_score,
        "touchScore": touch_score, "lineScore": line_score, "anchorScore": anchor_score,
        "durationScore": duration_score, "relevanceScore": relevance_score,
    })
    return pat


def broadening_wedge_scanner_pro(
    frame: pd.DataFrame,
    *,
    pivot_len: int = 6,
    max_swings: int = 80,
    search_depth: int = 24,
    wedge_window: int = 8,
    max_concurrent_patterns: int = 1,
    max_pattern_age_mult: float = 1.5,
    min_score: int = 60,
    atr_len: int = 14,
    touch_tol_atr: float = 0.5,
    projected_expansion_target: float = 1.3,
    min_width_atr: float = 1.0,
    faster_ratio_min: float = 1.2,
    min_traversal_ratio: float = 0.60,
    min_realized_expansion_ratio: float = 1.15,
    min_anchor_span_ratio: float = 0.55,
    max_boundary_drift_atr: float = 8.0,
    candidate_threshold: int = 25,
    show_candidate: bool = True,
):
    pivot_len = integer("pivot_len", pivot_len, 2, 30)
    max_swings = integer("max_swings", max_swings, 20, 200)
    search_depth = integer("search_depth", search_depth, 8, 60)
    wedge_window = integer("wedge_window", wedge_window, 6, 20)
    max_concurrent_patterns = integer("max_concurrent_patterns", max_concurrent_patterns, 1, 15)
    min_score = integer("min_score", min_score, 0, 100)
    atr_len = integer("atr_len", atr_len, 2, 200)
    max_pattern_age_mult = number("max_pattern_age_mult", max_pattern_age_mult, 1.0, 20.0)
    touch_tol_atr = number("touch_tol_atr", touch_tol_atr, 0.05, 20.0)
    projected_expansion_target = number("projected_expansion_target", projected_expansion_target, 1.05, 3.0)
    min_width_atr = number("min_width_atr", min_width_atr, 0.2, 50.0)
    faster_ratio_min = number("faster_ratio_min", faster_ratio_min, 1.0, 3.0)
    min_traversal_ratio = number("min_traversal_ratio", min_traversal_ratio, 0.40, 0.90)
    min_realized_expansion_ratio = number("min_realized_expansion_ratio", min_realized_expansion_ratio, 1.0, 3.0)
    min_anchor_span_ratio = number("min_anchor_span_ratio", min_anchor_span_ratio, 0.30, 0.90)
    max_boundary_drift_atr = number("max_boundary_drift_atr", max_boundary_drift_atr, 1.0, 20.0)
    candidate_threshold = integer("candidate_threshold", candidate_threshold, 0, 100)

    idx = frame.index
    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    n = len(close)

    atr_series = atr(frame, atr_len).to_numpy(dtype=float)
    ph = pivothigh_strict(frame["high"], pivot_len, pivot_len).to_numpy(dtype=float)
    pl = pivotlow_strict(frame["low"], pivot_len, pivot_len).to_numpy(dtype=float)

    P = dict(
        pivot_len=pivot_len, touch_tol_atr=touch_tol_atr,
        projected_expansion_target=projected_expansion_target, min_width_atr=min_width_atr,
        faster_ratio_min=faster_ratio_min, min_traversal_ratio=min_traversal_ratio,
        min_realized_expansion_ratio=min_realized_expansion_ratio,
        min_anchor_span_ratio=min_anchor_span_ratio, max_boundary_drift_atr=max_boundary_drift_atr,
        min_score=min_score, max_pattern_age_mult=max_pattern_age_mult,
    )

    swings = []  # [x, y, dir]
    last_swing_x = -1
    best = None
    tracked = []  # {"p": Pattern dict, "created_bar": int}

    best_score = np.zeros(n, dtype=np.int64)
    best_type = np.zeros(n, dtype=np.int64)
    best_valid = np.zeros(n, dtype=np.int64)
    best_proj = np.zeros(n, dtype=float)
    best_real = np.zeros(n, dtype=float)
    best_watch = np.zeros(n, dtype=np.int64)
    tracked_score = np.zeros(n, dtype=np.int64)
    tracked_type = np.zeros(n, dtype=np.int64)
    active_count = np.zeros(n, dtype=np.int64)
    breakout = np.zeros(n, dtype=np.int64)

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

    def line_cap_x(p):
        return p["windowEndX"] + (p["windowEndX"] - p["windowStartX"])

    def find_overlap_index(p):
        for i, tp in enumerate(tracked):
            if _overlaps(p["windowStartX"], p["windowEndX"], tp["p"]["windowStartX"], tp["p"]["windowEndX"]):
                return i
        return -1

    def find_weakest_index():
        idx_w, min_s = -1, 1_000_000
        for i, tp in enumerate(tracked):
            if tp["p"]["score"] < min_s:
                min_s = tp["p"]["score"]
                idx_w = i
        return idx_w

    def find_best_active_index():
        idx_b, max_s = -1, -1
        for i, tp in enumerate(tracked):
            if tp["p"]["score"] > max_s:
                max_s = tp["p"]["score"]
                idx_b = i
        return idx_b

    def dedupe_overlaps():
        m = len(tracked)
        if m <= 1:
            return
        keep = [True] * m
        for i in range(m - 1):
            for j in range(i + 1, m):
                if keep[i] and keep[j] and _overlaps(
                        tracked[i]["p"]["windowStartX"], tracked[i]["p"]["windowEndX"],
                        tracked[j]["p"]["windowStartX"], tracked[j]["p"]["windowEndX"]):
                    if tracked[i]["p"]["score"] >= tracked[j]["p"]["score"]:
                        keep[j] = False
                    else:
                        keep[i] = False
        for i in range(m - 1, -1, -1):
            if not keep[i]:
                tracked.pop(i)

    for i in range(n):
        if not math.isnan(ph[i]):
            add_swing(i - pivot_len, ph[i], 1)
        if not math.isnan(pl[i]):
            add_swing(i - pivot_len, pl[i], -1)

        sz = len(swings)
        cur_x = swings[-1][0] if sz > 0 else -1
        new_swing = cur_x != last_swing_x
        last_swing_x = cur_x

        local_best = None
        if new_swing and sz >= wedge_window:
            start = max(0, sz - search_depth)
            candidates = []
            local_best = _empty_pattern()
            first_end = max(start + wedge_window - 1, wedge_window - 1)
            if first_end <= sz - 1:
                for end in range(first_end, sz):
                    win_start = end - wedge_window + 1
                    hX, hY, lX, lY, pX, pY, pD = [], [], [], [], [], [], []
                    for k in range(win_start, end + 1):
                        s = swings[k]
                        pX.append(s[0]); pY.append(s[1]); pD.append(s[2])
                        if s[2] == 1:
                            hX.append(s[0]); hY.append(s[1])
                        else:
                            lX.append(s[0]); lY.append(s[1])
                    cand = _empty_pattern()
                    if len(hX) >= 2 and len(lX) >= 2:
                        cand = _eval_wedge(
                            high, low, close, atr_series[i], pX, pY, pD, hX, hY, lX, lY,
                            swings[win_start][0], swings[end][0], i, P)
                    if cand["score"] > local_best["score"]:
                        local_best = cand
                    candidates.append(cand)

            best = local_best
            # Track valid candidates (insert / evict / replace / dedupe).
            for c in candidates:
                if not c["valid"]:
                    continue
                cap_x = line_cap_x(c)
                still_open_eval_x = min(i, cap_x)
                upper_now = _f_price(c["xU1"], c["yU1"], c["xU2"], c["yU2"], still_open_eval_x)
                lower_now = _f_price(c["xL1"], c["yL1"], c["xL2"], c["yL2"], still_open_eval_x)
                still_open = close[i] < upper_now and close[i] > lower_now
                already_aged = (i - c["windowEndX"]) > max_pattern_age_mult * max(c["windowEndX"] - c["windowStartX"], 1)
                if still_open and not already_aged:
                    ov = find_overlap_index(c)
                    if ov < 0:
                        if len(tracked) < max_concurrent_patterns:
                            tracked.append({"p": c, "created_bar": i})
                        else:
                            wi = find_weakest_index()
                            if c["score"] > tracked[wi]["p"]["score"]:
                                tracked[wi] = {"p": c, "created_bar": i}
                    else:
                        existing = tracked[ov]
                        if c["windowEndX"] != existing["p"]["windowEndX"] and c["score"] > existing["p"]["score"]:
                            tracked[ov] = {"p": c, "created_bar": i}
            dedupe_overlaps()

        # Per-bar breakout & age resolution.
        bo_bull = bo_bear = False
        for t in range(len(tracked) - 1, -1, -1):
            tp = tracked[t]
            p = tp["p"]
            cap_x = line_cap_x(p)
            eval_x = min(i, cap_x)
            upper_now = _f_price(p["xU1"], p["yU1"], p["xU2"], p["yU2"], eval_x)
            lower_now = _f_price(p["xL1"], p["yL1"], p["xL2"], p["yL2"], eval_x)
            just_created = i <= tp["created_bar"]
            bull_hit = (not just_created) and close[i] > upper_now
            bear_hit = (not just_created) and close[i] < lower_now
            win_bars = p["windowEndX"] - p["windowStartX"]
            aged = (i - p["windowEndX"]) > max_pattern_age_mult * max(win_bars, 1)
            if bull_hit or bear_hit:
                bo_bull = bo_bull or bull_hit
                bo_bear = bo_bear or bear_hit
                tracked.pop(t)
            elif aged:
                tracked.pop(t)

        # Record per-bar output.
        if best is not None:
            best_score[i] = best["score"]
            best_type[i] = _TYPE_CODE[best["ttype"]]
            best_valid[i] = 1 if best["valid"] else 0
            best_proj[i] = best["projectedExpansionRatio"] if not math.isnan(best["projectedExpansionRatio"]) else 0.0
            best_real[i] = best["realizedExpansionRatio"] if not math.isnan(best["realizedExpansionRatio"]) else 0.0
            best_watch[i] = (1 if (best["structureOk"] and best["score"] >= candidate_threshold
                                    and not best["valid"] and show_candidate) else 0)
        bi = find_best_active_index()
        if bi >= 0:
            tracked_score[i] = tracked[bi]["p"]["score"]
            tracked_type[i] = _TYPE_CODE[tracked[bi]["p"]["ttype"]]
        active_count[i] = len(tracked)
        breakout[i] = 1 if bo_bull else (-1 if bo_bear else 0)

    out = pd.DataFrame(index=idx)
    out["bws_best_score"] = best_score
    out["bws_best_type"] = pd.Series(best_type, index=idx).astype("Int64")
    out["bws_best_valid"] = pd.Series(best_valid, index=idx).astype("Int64")
    out["bws_best_projected_expansion"] = best_proj
    out["bws_best_realized_expansion"] = best_real
    out["bws_best_watch"] = pd.Series(best_watch, index=idx).astype("Int64")
    out["bws_tracked_score"] = tracked_score
    out["bws_tracked_type"] = pd.Series(tracked_type, index=idx).astype("Int64")
    out["bws_active_count"] = pd.Series(active_count, index=idx).astype("Int64")
    out["bws_breakout"] = pd.Series(breakout, index=idx).astype("Int64")
    return out
