"""Triangle Compression Scanner Pro [WavesUnchained]  v1.15  —  Python port.

Source: data/pine-collection/indicators/market_structure/
        triangle_compression_scanner_pro/triangle_compression_scanner_pro.pine

Slides a fixed-size window of confirmed swings across the search depth and fits
two boundary lines (upper from the window's outer highs, lower from its outer
lows) to find COMPRESSING triangles — symmetric, ascending, or descending.
Every distinct, non-overlapping valid triangle is tracked independently until a
confirmed breakout resolves it.

Computational core ported (per-bar state machine):
  * Confirmed-swing engine (pivothigh/low + same-direction extremum replacement)
  * Sliding-window triangle search across the swing search depth
  * ``f_best_boundary`` two-real-touch boundary fit (fewest violations, then
    widest span, then smallest mean residual) — the chartist's trendline
  * ``f_eval_triangle``: ATR-relative flat/rising/falling classification,
    signed-width geometry with projected + realized compression and leg-shrink
    tests, traversal, contact distribution, flat-side, apex + line-quality,
    containment, duration and relevance — weighted 0-100 score + hard gates
  * Multi-pattern tracking (insert / evict-weakest / replace / dedupe overlaps)
    plus the ``keptPatterns`` layer that stops a lower-scoring candidate from
    redrawing over a resolved triangle at the same spot
  * Per-bar breakout & age resolution

Dropped (display / logging / non-computational): line/label drawing, dashboard
& debug tables, candidate marker, alertcondition, log.info, colour/size inputs.

Pine exposes *no* numeric series, so the port returns the scanner's per-bar
state as columns: the best-in-depth candidate and the best currently tracked
(active) triangle, plus the live breakout signal.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import atr, integer, number, pivothigh_strict, pivotlow_strict

_TYPE_CODE = {
    "None": 0,
    "Symmetric Compression": 1,
    "Ascending Compression": 2,
    "Descending Compression": 3,
}


# --------------------------------------------------------------------------- #
# Small helpers (mirror the Pine f_* helpers)                                  #
# --------------------------------------------------------------------------- #
def _f_price(x1, y1, x2, y2, x):
    return y1 if x2 == x1 else y1 + (y2 - y1) * float(x - x1) / float(x2 - x1)


def _f_slope(x1, y1, x2, y2):
    return float("nan") if x2 == x1 else (y2 - y1) / float(x2 - x1)


def _f_clamp01(v):
    return min(max(v, 0.0), 1.0)


def _proximity_quality(distance_atr):
    if distance_atr <= 0.25:
        return 1.0
    if distance_atr >= 0.75:
        return 0.0
    return 1.0 - (distance_atr - 0.25) / 0.50


def _empty_pattern():
    return {
        "valid": False, "ttype": "None", "score": 0,
        "xU1": None, "xU2": None, "yU1": None, "yU2": None,
        "xL1": None, "xL2": None, "yL1": None, "yL2": None,
        "windowStartX": 0, "windowEndX": 0,
        "structureOk": False, "pathOk": False, "traversalOk": False,
        "realizedCompressionOk": False, "boundaryOk": False, "apexOk": False,
        "lineQualityOk": False, "durationOk": False, "barsInsideOk": False,
        "swingInsideOk": False, "distributionOk": False, "flatSideOk": False,
        "upperAnchorSpanRatio": float("nan"), "lowerAnchorSpanRatio": float("nan"),
        "meanResidualAtr": float("nan"), "containmentRatio": float("nan"),
        "projectedCompressionRatio": float("nan"), "realizedCompressionRatio": float("nan"),
        "slopeRatio": float("nan"), "apexDistanceRatio": float("nan"),
        "apexProgress": float("nan"), "progressionMatches": 0, "progressionTests": 0,
        "longestTraversalRun": 0, "averageTraversal": 0.0,
        "upperZoneCount": 0, "lowerZoneCount": 0,
        "pathScore": 0, "traversalScore": 0, "compressionScore": 0, "touchScore": 0,
        "lineScore": 0, "boundaryScore": 0, "durationScore": 0, "relevanceScore": 0,
    }


def _overlaps(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) <= min(a_end, b_end)


def _best_boundary(xs, ys, tol_price, is_upper):
    """Two-real-touch boundary fit (fewest violations -> widest span -> residual)."""
    n = len(xs)
    best_i, best_j = 0, n - 1
    best_viol, best_touch, best_span = 1_000_000, -1, -1
    best_mean_residual = 1_000_000_000.0
    for i in range(n - 1):
        xi, yi = xs[i], ys[i]
        for j in range(i + 1, n):
            xj, yj = xs[j], ys[j]
            touches = 0
            violations = 0
            residual_sum = 0.0
            for k in range(n):
                xk, yk = xs[k], ys[k]
                dist = yk - _f_price(xi, yi, xj, yj, xk)
                residual_sum += abs(dist)
                if abs(dist) <= tol_price:
                    touches += 1
                elif is_upper and dist > tol_price:
                    violations += 1
                elif (not is_upper) and dist < -tol_price:
                    violations += 1
            span = xj - xi
            mean_residual = residual_sum / float(n)
            better = (violations < best_viol) or (
                violations == best_viol and (
                    span > best_span or (
                        span == best_span and (
                            mean_residual < best_mean_residual or (
                                mean_residual == best_mean_residual and touches > best_touch
                            )
                        )
                    )
                )
            )
            if better:
                best_viol, best_touch, best_span = violations, touches, span
                best_mean_residual = mean_residual
                best_i, best_j = i, j
    return xs[best_i], ys[best_i], xs[best_j], ys[best_j]


def _eval_triangle(high, low, close, atr_i, pathX, pathY, pathDir, highX, highY,
                   lowX, lowY, window_start_x, window_end_x, search_bar, P):
    nH, nL, nPath = len(highX), len(lowX), len(pathX)

    window_bars = window_end_x - window_start_x
    touch_tol_price = P["touch_tol_atr"] * atr_i

    xU1, yU1, xU2, yU2 = _best_boundary(highX, highY, touch_tol_price, True)
    xL1, yL1, xL2, yL2 = _best_boundary(lowX, lowY, touch_tol_price, False)

    slopeU = _f_slope(xU1, yU1, xU2, yU2)
    slopeL = _f_slope(xL1, yL1, xL2, yL2)

    flat_tol = (P["flat_slope_atr_tol"] * atr_i / window_bars) if window_bars > 0 else P["flat_slope_atr_tol"] * atr_i
    upper_flat = (not math.isnan(slopeU)) and abs(slopeU) < flat_tol
    lower_flat = (not math.isnan(slopeL)) and abs(slopeL) < flat_tol
    upper_falling = (not math.isnan(slopeU)) and slopeU < 0 and not upper_flat
    lower_rising = (not math.isnan(slopeL)) and slopeL > 0 and not lower_flat

    is_symmetric = upper_falling and lower_rising
    is_ascending = upper_flat and lower_rising
    is_descending = upper_falling and lower_flat
    ttype = ("Symmetric Compression" if is_symmetric else
             "Ascending Compression" if is_ascending else
             "Descending Compression" if is_descending else "None")
    structure_ok = is_symmetric or is_ascending or is_descending

    mid_x = int(round((window_start_x + window_end_x) / 2.0))
    upper_start = _f_price(xU1, yU1, xU2, yU2, window_start_x)
    lower_start = _f_price(xL1, yL1, xL2, yL2, window_start_x)
    upper_mid = _f_price(xU1, yU1, xU2, yU2, mid_x)
    lower_mid = _f_price(xL1, yL1, xL2, yL2, mid_x)
    upper_end = _f_price(xU1, yU1, xU2, yU2, window_end_x)
    lower_end = _f_price(xL1, yL1, xL2, yL2, window_end_x)

    width_start = upper_start - lower_start
    width_mid = upper_mid - lower_mid
    width_end = upper_end - lower_end

    projected_ratio = (width_end / width_start) if width_start > 0 else 1_000_000.0
    compression_geometry_ok = (width_start > P["min_width_atr"] * atr_i and width_mid > 0
                               and width_end > 0 and width_mid <= width_start
                               and width_end <= width_mid
                               and projected_ratio <= P["projected_compression_target"])

    alternating_path = True
    if nPath > 1:
        for i in range(1, nPath):
            if pathDir[i] == pathDir[i - 1]:
                alternating_path = False
    progression_tests = progression_matches = 0
    if (is_symmetric or is_descending) and nH > 1:
        for i in range(1, nH):
            progression_tests += 1
            if highY[i] < highY[i - 1]:
                progression_matches += 1
    if (is_symmetric or is_ascending) and nL > 1:
        for i in range(1, nL):
            progression_tests += 1
            if lowY[i] > lowY[i - 1]:
                progression_matches += 1
    progression_ratio = progression_matches / progression_tests if progression_tests > 0 else 0.0
    path_ok = (structure_ok and alternating_path and nPath >= 6 and nH >= 3 and nL >= 3
               and progression_tests >= 2 and progression_matches == progression_tests)
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
    realized_ratio = last_avg / first_avg if first_avg > 0 else 1_000_000.0
    leg_shrink_tests = leg_shrink_matches = 0
    prev_up_leg = float("nan")
    prev_down_leg = float("nan")
    if leg_count > 0:
        for i in range(leg_count):
            amp = leg_amplitudes[i]
            ends_high = pathDir[i + 1] == 1
            prev_same = prev_up_leg if ends_high else prev_down_leg
            if not math.isnan(prev_same):
                leg_shrink_tests += 1
                if amp <= prev_same * 1.05:
                    leg_shrink_matches += 1
            if ends_high:
                prev_up_leg = amp
            else:
                prev_down_leg = amp
    leg_shrink_ratio = leg_shrink_matches / leg_shrink_tests if leg_shrink_tests > 0 else 0.0
    realized_ok = (compression_geometry_ok and leg_count >= 5
                   and realized_ratio <= P["max_realized_compression_ratio"]
                   and leg_shrink_ratio >= 0.60)
    realized_quality = (_f_clamp01((1.0 - realized_ratio) / (1.0 - P["max_realized_compression_ratio"]))
                        if P["max_realized_compression_ratio"] < 1.0
                        else (1.0 if realized_ratio <= 1.0 else 0.0))
    projected_quality = _f_clamp01((1.0 - projected_ratio) / (1.0 - P["projected_compression_target"]))
    compression_score = int(round(15.0 * (0.70 * realized_quality + 0.20 * leg_shrink_ratio
                                          + 0.10 * projected_quality)))

    upper_anchor_span = float(xU2 - xU1) / float(window_bars) if window_bars > 0 else 0.0
    lower_anchor_span = float(xL2 - xL1) / float(window_bars) if window_bars > 0 else 0.0

    extra_high_touches = high_residual_sum = high_violations = 0
    upper_early = upper_middle = upper_late = False
    proximity_sum = proximity_count = 0
    if nH > 0:
        for i in range(nH):
            hx, hy = highX[i], highY[i]
            line_y = _f_price(xU1, yU1, xU2, yU2, hx)
            dist = hy - line_y
            abs_dist = abs(dist)
            high_residual_sum += abs_dist
            is_anchor = hx == xU1 or hx == xU2
            if not is_anchor:
                proximity_sum += _proximity_quality(abs_dist / atr_i if atr_i > 0 else 1_000_000.0)
                proximity_count += 1
            if abs_dist <= touch_tol_price:
                if not is_anchor:
                    extra_high_touches += 1
                position_ratio = float(hx - window_start_x) / float(window_bars) if window_bars > 0 else 0.0
                if position_ratio < 0.333333:
                    upper_early = True
                elif position_ratio < 0.666667:
                    upper_middle = True
                else:
                    upper_late = True
            elif dist > touch_tol_price:
                high_violations += 1
    extra_low_touches = low_residual_sum = low_violations = 0
    lower_early = lower_middle = lower_late = False
    if nL > 0:
        for i in range(nL):
            lx, ly = lowX[i], lowY[i]
            line_y = _f_price(xL1, yL1, xL2, yL2, lx)
            dist = ly - line_y
            abs_dist = abs(dist)
            low_residual_sum += abs_dist
            is_anchor = lx == xL1 or lx == xL2
            if not is_anchor:
                proximity_sum += _proximity_quality(abs_dist / atr_i if atr_i > 0 else 1_000_000.0)
                proximity_count += 1
            if abs_dist <= touch_tol_price:
                if not is_anchor:
                    extra_low_touches += 1
                position_ratio = float(lx - window_start_x) / float(window_bars) if window_bars > 0 else 0.0
                if position_ratio < 0.333333:
                    lower_early = True
                elif position_ratio < 0.666667:
                    lower_middle = True
                else:
                    lower_late = True
            elif dist < -touch_tol_price:
                low_violations += 1

    touch_score = int(round(15.0 * (proximity_sum / proximity_count if proximity_count > 0 else 0.0)))
    upper_zone_count = (1 if upper_early else 0) + (1 if upper_middle else 0) + (1 if upper_late else 0)
    lower_zone_count = (1 if lower_early else 0) + (1 if lower_middle else 0) + (1 if lower_late else 0)
    distribution_ok = upper_zone_count >= 2 and lower_zone_count >= 2 and (upper_late or lower_late)
    flat_side_ok = (extra_high_touches >= 1 if is_ascending
                    else extra_low_touches >= 1 if is_descending else True)

    point_count = nH + nL
    mean_residual = ((high_residual_sum + low_residual_sum) / point_count) if point_count > 0 else atr_i * 1_000_000.0
    mean_residual_atr = mean_residual / atr_i if atr_i > 0 else 1_000_000.0
    swing_inside_ok = high_violations == 0 and low_violations == 0
    boundary_ok = (upper_anchor_span >= P["min_anchor_span_ratio"]
                   and lower_anchor_span >= P["min_anchor_span_ratio"]
                   and mean_residual_atr <= P["max_mean_residual_atr"]
                   and swing_inside_ok and distribution_ok and flat_side_ok)

    slope_ratio = (abs(slopeU) / abs(slopeL)
                   if (is_symmetric and not math.isnan(slopeU) and not math.isnan(slopeL)
                       and abs(slopeL) > 0.0000001) else float("nan"))
    line_quality_ok = ((not math.isnan(slope_ratio) and 0.3 <= slope_ratio <= 3.0)
                       if is_symmetric else structure_ok)

    duration_ok = window_bars >= P["pivot_len"] * 3
    duration_score = int(round(5.0 * _f_clamp01(float(window_bars) / float(P["pivot_len"] * 6))))

    convergence_slope = slopeU - slopeL
    apex_x = (float(window_start_x) - width_start / convergence_slope
              if (not math.isnan(convergence_slope)) and convergence_slope < 0 else float("nan"))
    apex_distance = (apex_x - float(window_end_x)) if not math.isnan(apex_x) else -1.0
    apex_distance_ratio = (apex_distance / float(window_bars)) if window_bars > 0 else -1.0
    apex_progress = (float(window_bars) / (apex_x - float(window_start_x))
                     if (not math.isnan(apex_x)) and apex_x > float(window_start_x) else 0.0)
    apex_ok = (not math.isnan(apex_x)) and apex_distance > 0 and \
        apex_distance <= P["max_apex_distance_mult"] * float(max(window_bars, 1))
    apex_quality = _f_clamp01(1.0 - abs(apex_progress - 0.65) / 0.35)
    anchor_quality = (_f_clamp01(upper_anchor_span) + _f_clamp01(lower_anchor_span)) / 2.0
    boundary_score = int(round(10.0 * (0.60 * anchor_quality + 0.40 * apex_quality)))

    bars_inside = evaluated = 0
    for k in range(window_start_x, window_end_x + 1):
        upper_at = _f_price(xU1, yU1, xU2, yU2, k)
        lower_at = _f_price(xL1, yL1, xL2, yL2, k)
        bar_high = high[k]
        bar_low = low[k]
        evaluated += 1
        if bar_high <= upper_at + touch_tol_price and bar_low >= lower_at - touch_tol_price:
            bars_inside += 1
    containment_ratio = bars_inside / evaluated if evaluated > 0 else 0.0
    bars_inside_ok = containment_ratio >= 0.90
    residual_quality = 1.0 - _f_clamp01(mean_residual_atr / P["max_mean_residual_atr"])
    violation_ratio = ((high_violations + low_violations) / point_count) if point_count > 0 else 1.0
    distribution_quality = (float(upper_zone_count) + float(lower_zone_count)) / 6.0
    line_score = int(round(10.0 * (0.45 * residual_quality + 0.35 * containment_ratio
                                   + 0.10 * (1.0 - violation_ratio) + 0.10 * distribution_quality)))

    max_relevant_age = P["max_pattern_age_mult"] * float(max(window_bars, 1))
    candidate_age = float(max(search_bar - window_end_x, 0))
    relevance_score = int(round(5.0 * (1.0 - _f_clamp01(candidate_age / max_relevant_age))
                                if max_relevant_age > 0 else 0.0))

    raw_score = (traversal_score + compression_score + touch_score + line_score
                 + boundary_score + duration_score + relevance_score)
    score = int(round(float(raw_score) * 100.0 / 80.0))
    hard_gates_ok = (structure_ok and path_ok and traversal_ok and realized_ok
                     and boundary_ok and apex_ok and line_quality_ok and duration_ok)

    pat = _empty_pattern()
    pat.update({
        "valid": score >= P["min_score"] and hard_gates_ok,
        "ttype": ttype, "score": score,
        "xU1": xU1, "xU2": xU2, "yU1": yU1, "yU2": yU2,
        "xL1": xL1, "xL2": xL2, "yL1": yL1, "yL2": yL2,
        "windowStartX": window_start_x, "windowEndX": window_end_x,
        "structureOk": structure_ok, "pathOk": path_ok, "traversalOk": traversal_ok,
        "realizedCompressionOk": realized_ok, "boundaryOk": boundary_ok, "apexOk": apex_ok,
        "lineQualityOk": line_quality_ok, "durationOk": duration_ok, "barsInsideOk": bars_inside_ok,
        "swingInsideOk": swing_inside_ok, "distributionOk": distribution_ok, "flatSideOk": flat_side_ok,
        "upperAnchorSpanRatio": upper_anchor_span, "lowerAnchorSpanRatio": lower_anchor_span,
        "meanResidualAtr": mean_residual_atr, "containmentRatio": containment_ratio,
        "projectedCompressionRatio": projected_ratio, "realizedCompressionRatio": realized_ratio,
        "slopeRatio": slope_ratio, "apexDistanceRatio": apex_distance_ratio,
        "apexProgress": apex_progress, "progressionMatches": progression_matches,
        "progressionTests": progression_tests, "longestTraversalRun": longest_run,
        "averageTraversal": avg_traversal, "upperZoneCount": upper_zone_count,
        "lowerZoneCount": lower_zone_count, "pathScore": path_score,
        "traversalScore": traversal_score, "compressionScore": compression_score,
        "touchScore": touch_score, "lineScore": line_score, "boundaryScore": boundary_score,
        "durationScore": duration_score, "relevanceScore": relevance_score,
    })
    return pat


def triangle_compression_scanner_pro(
    frame: pd.DataFrame,
    *,
    pivot_len: int = 6,
    max_swings: int = 80,
    search_depth: int = 24,
    triangle_window: int = 8,
    max_concurrent_patterns: int = 1,
    max_pattern_age_mult: float = 1.5,
    min_score: int = 60,
    atr_len: int = 14,
    flat_slope_atr_tol: float = 0.5,
    touch_tol_atr: float = 0.5,
    projected_compression_target: float = 0.75,
    min_width_atr: float = 1.0,
    min_traversal_ratio: float = 0.55,
    max_realized_compression_ratio: float = 0.85,
    min_anchor_span_ratio: float = 0.45,
    max_mean_residual_atr: float = 1.0,
    max_apex_distance_mult: float = 1.25,
    breakout_buffer_atr: float = 0.15,
    show_old_patterns: bool = True,
    show_candidate: bool = True,
    candidate_threshold: int = 25,
) -> pd.DataFrame:
    pivot_len = integer("pivot_len", pivot_len, 2, 30)
    max_swings = integer("max_swings", max_swings, 20, 200)
    search_depth = integer("search_depth", search_depth, 8, 60)
    triangle_window = integer("triangle_window", triangle_window, 6, 20)
    max_concurrent_patterns = integer("max_concurrent_patterns", max_concurrent_patterns, 1, 15)
    min_score = integer("min_score", min_score, 0, 100)
    atr_len = integer("atr_len", atr_len, 2, 200)
    candidate_threshold = integer("candidate_threshold", candidate_threshold, 0, 100)
    max_pattern_age_mult = number("max_pattern_age_mult", max_pattern_age_mult, 1.0, 20.0)
    flat_slope_atr_tol = number("flat_slope_atr_tol", flat_slope_atr_tol, 0.05, 20.0)
    touch_tol_atr = number("touch_tol_atr", touch_tol_atr, 0.05, 20.0)
    projected_compression_target = number("projected_compression_target", projected_compression_target, 0.3, 0.95)
    min_width_atr = number("min_width_atr", min_width_atr, 0.2, 50.0)
    min_traversal_ratio = number("min_traversal_ratio", min_traversal_ratio, 0.40, 0.90)
    max_realized_compression_ratio = number("max_realized_compression_ratio", max_realized_compression_ratio, 0.40, 1.0)
    min_anchor_span_ratio = number("min_anchor_span_ratio", min_anchor_span_ratio, 0.30, 0.90)
    max_mean_residual_atr = number("max_mean_residual_atr", max_mean_residual_atr, 0.25, 3.0)
    max_apex_distance_mult = number("max_apex_distance_mult", max_apex_distance_mult, 0.25, 3.0)
    breakout_buffer_atr = number("breakout_buffer_atr", breakout_buffer_atr, 0.0, 1.0)

    idx = frame.index
    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    n = len(close)

    atr_series = atr(frame, atr_len).to_numpy(dtype=float)
    ph = pivothigh_strict(frame["high"], pivot_len, pivot_len).to_numpy(dtype=float)
    pl = pivotlow_strict(frame["low"], pivot_len, pivot_len).to_numpy(dtype=float)

    P = dict(
        pivot_len=pivot_len, flat_slope_atr_tol=flat_slope_atr_tol, touch_tol_atr=touch_tol_atr,
        projected_compression_target=projected_compression_target, min_width_atr=min_width_atr,
        min_traversal_ratio=min_traversal_ratio,
        max_realized_compression_ratio=max_realized_compression_ratio,
        min_anchor_span_ratio=min_anchor_span_ratio, max_mean_residual_atr=max_mean_residual_atr,
        max_apex_distance_mult=max_apex_distance_mult, min_score=min_score,
        max_pattern_age_mult=max_pattern_age_mult,
    )

    swings = []  # [x, y, dir]
    last_swing_x = -1
    best = None
    tracked = []  # {"p": pattern dict, "created_bar": int}
    kept = []  # {"windowStartX", "windowEndX", "score"}

    best_score = np.zeros(n, dtype=np.int64)
    best_type = np.zeros(n, dtype=np.int64)
    best_valid = np.zeros(n, dtype=np.int64)
    best_proj = np.zeros(n, dtype=float)
    best_real = np.zeros(n, dtype=float)
    best_apex = np.zeros(n, dtype=float)
    best_resid = np.zeros(n, dtype=float)
    best_contain = np.zeros(n, dtype=float)
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

    def find_overlap_index(p, arr):
        for i, tp in enumerate(arr):
            q = tp["p"] if "p" in tp else tp
            if _overlaps(p["windowStartX"], p["windowEndX"], q["windowStartX"], q["windowEndX"]):
                return i
        return -1

    def find_weakest_index():
        w, min_s = -1, 1_000_000
        for i, tp in enumerate(tracked):
            if tp["p"]["score"] < min_s:
                min_s = tp["p"]["score"]
                w = i
        return w

    def find_best_active_index():
        b, max_s = -1, -1
        for i, tp in enumerate(tracked):
            if tp["p"]["score"] > max_s:
                max_s = tp["p"]["score"]
                b = i
        return b

    def finalize_and_keep(p):
        kept.append({"windowStartX": p["windowStartX"], "windowEndX": p["windowEndX"],
                     "score": p["score"]})

    def dedupe_overlaps():
        m = len(tracked)
        if m <= 1:
            return
        keep_flags = [True] * m
        for i in range(m - 1):
            for j in range(i + 1, m):
                if keep_flags[i] and keep_flags[j] and _overlaps(
                        tracked[i]["p"]["windowStartX"], tracked[i]["p"]["windowEndX"],
                        tracked[j]["p"]["windowStartX"], tracked[j]["p"]["windowEndX"]):
                    if tracked[i]["p"]["score"] >= tracked[j]["p"]["score"]:
                        keep_flags[j] = False
                    else:
                        keep_flags[i] = False
        for i in range(m - 1, -1, -1):
            if not keep_flags[i]:
                if show_old_patterns:
                    finalize_and_keep(tracked[i]["p"])
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

        if new_swing and sz >= triangle_window:
            start = max(0, sz - search_depth)
            candidates = []
            local_best = _empty_pattern()
            first_end = max(start + triangle_window - 1, triangle_window - 1)
            if first_end <= sz - 1:
                for end in range(first_end, sz):
                    win_start = end - triangle_window + 1
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
                        cand = _eval_triangle(
                            high, low, close, atr_series[i], pX, pY, pD, hX, hY, lX, lY,
                            swings[win_start][0], swings[end][0], i, P)
                    if cand["score"] > local_best["score"]:
                        local_best = cand
                    candidates.append(cand)

            best = local_best
            atr_i = atr_series[i]
            breakout_buffer = breakout_buffer_atr * atr_i

            for c in candidates:
                if not c["valid"]:
                    continue
                cap_x = line_cap_x(c)
                still_open_eval_x = min(i, cap_x)
                upper_now = _f_price(c["xU1"], c["yU1"], c["xU2"], c["yU2"], still_open_eval_x)
                lower_now = _f_price(c["xL1"], c["yL1"], c["xL2"], c["yL2"], still_open_eval_x)
                still_open = (close[i] < upper_now + breakout_buffer
                              and close[i] > lower_now - breakout_buffer)
                already_aged = (i - c["windowEndX"]) > max_pattern_age_mult * max(c["windowEndX"] - c["windowStartX"], 1)
                if not (still_open and not already_aged):
                    continue
                kept_overlap_idx = find_overlap_index(c, kept)
                kept_blocks = kept_overlap_idx >= 0 and kept[kept_overlap_idx]["score"] >= c["score"]
                if kept_blocks:
                    continue
                if kept_overlap_idx >= 0:
                    kept.pop(kept_overlap_idx)
                overlap_idx = find_overlap_index(c, tracked)
                if overlap_idx < 0:
                    if len(tracked) < max_concurrent_patterns:
                        tracked.append({"p": c, "created_bar": i})
                    else:
                        wi = find_weakest_index()
                        if c["score"] > tracked[wi]["p"]["score"]:
                            if show_old_patterns:
                                finalize_and_keep(tracked[wi]["p"])
                            tracked[wi] = {"p": c, "created_bar": i}
                else:
                    existing = tracked[overlap_idx]
                    if c["windowEndX"] != existing["p"]["windowEndX"] and c["score"] > existing["p"]["score"]:
                        if show_old_patterns:
                            finalize_and_keep(existing["p"])
                        tracked[overlap_idx] = {"p": c, "created_bar": i}
            dedupe_overlaps()

            # Trim kept entries too old to ever be compared against again.
            for j in range(len(kept) - 1, -1, -1):
                kp = kept[j]
                kp_bars = kp["windowEndX"] - kp["windowStartX"]
                if i - kp["windowEndX"] > max_pattern_age_mult * max(kp_bars, 1):
                    kept.pop(j)

        # Per-bar breakout & age resolution.
        bo_bull = bo_bear = False
        atr_i = atr_series[i]
        breakout_buffer = breakout_buffer_atr * atr_i
        for t in range(len(tracked) - 1, -1, -1):
            tp = tracked[t]
            p = tp["p"]
            cap_x = line_cap_x(p)
            eval_x = min(i, cap_x)
            upper_now = _f_price(p["xU1"], p["yU1"], p["xU2"], p["yU2"], eval_x)
            lower_now = _f_price(p["xL1"], p["yL1"], p["xL2"], p["yL2"], eval_x)
            just_created = i <= tp["created_bar"]
            bull_hit = (not just_created) and close[i] > upper_now + breakout_buffer
            bear_hit = (not just_created) and close[i] < lower_now - breakout_buffer
            win_bars = p["windowEndX"] - p["windowStartX"]
            aged = (i - p["windowEndX"]) > max_pattern_age_mult * max(win_bars, 1)
            if bull_hit or bear_hit:
                bo_bull = bo_bull or bull_hit
                bo_bear = bo_bear or bear_hit
                if show_old_patterns:
                    finalize_and_keep(p)
                tracked.pop(t)
            elif aged:
                if show_old_patterns:
                    finalize_and_keep(p)
                tracked.pop(t)

        if best is not None:
            best_score[i] = best["score"]
            best_type[i] = _TYPE_CODE[best["ttype"]]
            best_valid[i] = 1 if best["valid"] else 0
            best_proj[i] = best["projectedCompressionRatio"] if not math.isnan(best["projectedCompressionRatio"]) else 0.0
            best_real[i] = best["realizedCompressionRatio"] if not math.isnan(best["realizedCompressionRatio"]) else 0.0
            best_apex[i] = best["apexProgress"] if not math.isnan(best["apexProgress"]) else 0.0
            best_resid[i] = best["meanResidualAtr"] if not math.isnan(best["meanResidualAtr"]) else 0.0
            best_contain[i] = best["containmentRatio"] if not math.isnan(best["containmentRatio"]) else 0.0
            best_watch[i] = (1 if (best["structureOk"] and best["score"] >= candidate_threshold
                                   and not best["valid"] and show_candidate) else 0)
        bi = find_best_active_index()
        if bi >= 0:
            tracked_score[i] = tracked[bi]["p"]["score"]
            tracked_type[i] = _TYPE_CODE[tracked[bi]["p"]["ttype"]]
        active_count[i] = len(tracked)
        breakout[i] = 1 if bo_bull else (-1 if bo_bear else 0)

    out = pd.DataFrame(index=idx)
    out["tcs_best_score"] = best_score
    out["tcs_best_type"] = pd.Series(best_type, index=idx).astype("Int64")
    out["tcs_best_valid"] = pd.Series(best_valid, index=idx).astype("Int64")
    out["tcs_best_projected_compression"] = best_proj
    out["tcs_best_realized_compression"] = best_real
    out["tcs_best_apex_progress"] = best_apex
    out["tcs_best_mean_residual_atr"] = best_resid
    out["tcs_best_containment"] = best_contain
    out["tcs_best_watch"] = pd.Series(best_watch, index=idx).astype("Int64")
    out["tcs_tracked_score"] = tracked_score
    out["tcs_tracked_type"] = pd.Series(tracked_type, index=idx).astype("Int64")
    out["tcs_active_count"] = pd.Series(active_count, index=idx).astype("Int64")
    out["tcs_breakout"] = pd.Series(breakout, index=idx).astype("Int64")
    return out
