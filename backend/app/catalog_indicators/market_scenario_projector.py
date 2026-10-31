"""Market Scenario Projector [WavesUnchained]  v0.6.9  —  catalog port.

Source: data/pine-collection/indicators/market_structure/
        market_scenario_projector/market_scenario_projector.pine

Scores the current impulse, swing structure, nearest decision zone and
rejection evidence into three forward scenarios — direct continuation,
pullback-then-continuation and structural failure — and normalizes them to
100%. Also derives the target / pullback-zone / failure levels and a global
setup-quality gate.

Ported computational core: pivot storage + structure bias, impulse/efficiency/
acceleration, level clustering + recency-weighted quality, rejection evidence,
target/pullback/failure projections, the three weighted scenario scores and
the setup-quality gate. Outcome recorder (log-only) and all drawing are dropped.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import atr, integer, number, ohlcv, pivothigh_strict, pivotlow_strict


def _clamp(v, lo, hi):
    if math.isnan(v):
        return v
    return max(lo, min(hi, v))


def _nearest_above(prices, price, max_dist):
    result = None
    for lv in prices:
        d = lv - price
        if 0 < d <= max_dist and (result is None or lv < result):
            result = lv
    return result


def _nearest_below(prices, price, max_dist):
    result = None
    for lv in prices:
        d = price - lv
        if 0 < d <= max_dist and (result is None or lv > result):
            result = lv
    return result


def _next_high(prices, level, min_sep, max_dist):
    result = None
    for cand in prices:
        d = cand - level
        if min_sep < d <= max_dist and (result is None or cand < result):
            result = cand
    return result


def _next_low(prices, level, min_sep, max_dist):
    result = None
    for cand in prices:
        d = level - cand
        if min_sep < d <= max_dist and (result is None or cand > result):
            result = cand
    return result


def _level_quality(prices, bars, level, zone_width, influence, half_life, i):
    if level is None:
        return 0.0, None, None
    weighted = 0.0
    zlow = zhigh = None
    for cand, tb in zip(prices, bars):
        if abs(cand - level) <= zone_width:
            age = i - tb
            w = 1.0 - influence + influence * math.pow(0.5, age / half_life)
            weighted += w
            zlow = cand if zlow is None else min(zlow, cand)
            zhigh = cand if zhigh is None else max(zhigh, cand)
    return _clamp(weighted / 3.0, 0.0, 1.0), zlow, zhigh


def _rejection_score(dir_, level, tolerance, lookback, h, l, o, c, i):
    if level is None or lookback <= 0:
        return 0.0
    wick_sum = body_sum = 0.0
    touches = fails = 0
    for k in range(lookback):
        j = i - k
        if j < 0:
            break
        bar_high, bar_low, bar_open, bar_close = h[j], l[j], o[j], c[j]
        rng = bar_high - bar_low
        if rng <= 0:
            continue
        touched = (bar_high >= level - tolerance) if dir_ > 0 else (bar_low <= level + tolerance)
        if touched:
            touches += 1
            body_sum += abs(bar_close - bar_open) / rng
            wick = (bar_high - max(bar_open, bar_close)) if dir_ > 0 else (min(bar_open, bar_close) - bar_low)
            wick_sum += wick / rng
            failed = (bar_high >= level and bar_close < level) if dir_ > 0 else (bar_low <= level and bar_close > level)
            if failed:
                fails += 1
    if touches <= 0:
        return 0.0
    avg_wick = wick_sum / touches
    fail_ratio = fails / touches
    avg_body = body_sum / touches
    compression = _clamp(1.0 - avg_body / 0.6, 0.0, 1.0)
    return _clamp(avg_wick * 0.4 + fail_ratio * 0.4 + compression * 0.2, 0.0, 1.0)


def market_scenario_projector(
    frame: pd.DataFrame,
    *,
    impulse_length: int = 12,
    atr_length: int = 14,
    analysis_lookback: int = 250,
    minimum_setup_score: float = 45.0,
    pivot_left: int = 4,
    pivot_right: int = 4,
    max_stored_pivots: int = 60,
    max_level_distance_atr: float = 6.0,
    level_zone_atr: float = 0.20,
    level_recency_influence: float = 0.3,
    require_barrier: bool = True,
    rejection_lookback: int = 3,
    rejection_zone_atr: float = 0.35,
    fallback_target_atr: float = 2.0,
    fallback_pullback_atr: float = 1.0,
    impulse_retracement_ratio: float = 0.382,
) -> pd.DataFrame:
    impulse_length = integer("impulse_length", impulse_length, 4, 100)
    atr_length = integer("atr_length", atr_length, 5, 100)
    analysis_lookback = integer("analysis_lookback", analysis_lookback, 50, 1000)
    pivot_left = integer("pivot_left", pivot_left, 1, 50)
    pivot_right = integer("pivot_right", pivot_right, 1, 50)
    max_stored_pivots = integer("max_stored_pivots", max_stored_pivots, 10, 200)
    rejection_lookback = integer("rejection_lookback", rejection_lookback, 1, 10)
    minimum_setup_score = number("minimum_setup_score", minimum_setup_score, 0.0, 100.0)
    max_level_distance_atr = number("max_level_distance_atr", max_level_distance_atr, 0.5, 20.0)
    level_zone_atr = number("level_zone_atr", level_zone_atr, 0.05, 1.0)
    level_recency_influence = number("level_recency_influence", level_recency_influence, 0.0, 1.0)
    rejection_zone_atr = number("rejection_zone_atr", rejection_zone_atr, 0.05, 2.0)
    fallback_target_atr = number("fallback_target_atr", fallback_target_atr, 0.25, 10.0)
    fallback_pullback_atr = number("fallback_pullback_atr", fallback_pullback_atr, 0.25, 5.0)
    impulse_retracement_ratio = number("impulse_retracement_ratio", impulse_retracement_ratio, 0.1, 0.9)

    o, h, l, c = ohlcv(frame, "open", "high", "low", "close")
    o = o.to_numpy(dtype=float); h = h.to_numpy(dtype=float)
    l = l.to_numpy(dtype=float); c = c.to_numpy(dtype=float)
    idx = frame.index
    n = len(frame)

    atr_a = np.maximum(atr(frame, atr_length).to_numpy(dtype=float), 1e-8)
    ph = pivothigh_strict(frame["high"], pivot_left, pivot_right).to_numpy(dtype=float)
    pl = pivotlow_strict(frame["low"], pivot_left, pivot_right).to_numpy(dtype=float)
    chg = np.abs(np.diff(c, prepend=c[0]))
    abs_sum = pd.Series(chg).rolling(impulse_length, min_periods=impulse_length).sum().to_numpy()

    ph_prices: list = []; ph_bars: list = []
    pl_prices: list = []; pl_bars: list = []

    cols = {k: np.full(n, np.nan) for k in [
        "direction", "impulse", "efficiency", "acceleration", "structure_bias", "structure_score",
        "barrier", "barrier_quality", "proximity", "rejection", "direct_score", "pullback_score",
        "failure_score", "direct_weight", "pullback_weight", "failure_weight", "dominant_weight",
        "dominance_gap", "target", "failure_target", "pullback", "pullback_low", "pullback_high",
        "setup_quality"]}
    valid = np.zeros(n, dtype=np.int64)

    for i in range(n):
        if not math.isnan(ph[i]):
            ph_prices.append(ph[i]); ph_bars.append(i - pivot_right)
        if not math.isnan(pl[i]):
            pl_prices.append(pl[i]); pl_bars.append(i - pivot_right)
        while len(ph_prices) > max_stored_pivots:
            ph_prices.pop(0); ph_bars.pop(0)
        while len(pl_prices) > max_stored_pivots:
            pl_prices.pop(0); pl_bars.pop(0)
        while ph_bars and ph_bars[0] < i - analysis_lookback:
            ph_prices.pop(0); ph_bars.pop(0)
        while pl_bars and pl_bars[0] < i - analysis_lookback:
            pl_prices.pop(0); pl_bars.pop(0)

        # structure bias
        bias = 0.0
        if len(ph_prices) >= 2 and len(pl_prices) >= 2:
            latest_h, prev_h = ph_prices[-1], ph_prices[-2]
            latest_l, prev_l = pl_prices[-1], pl_prices[-2]
            hh = latest_h > prev_h; hl = latest_l > prev_l
            lh = latest_h < prev_h; ll = latest_l < prev_l
            if hh and hl:
                bias = 1.0
            elif lh and ll:
                bias = -1.0
            elif hl:
                bias = 0.5
            elif lh:
                bias = -0.5

        atr_i = atr_a[i]
        enough = i > impulse_length * 2 + atr_length
        raw_move = (c[i] - c[i - impulse_length]) if (enough and i >= impulse_length) else 0.0
        direction = 1 if raw_move >= 0 else -1
        move_atr = abs(raw_move) / atr_i if atr_i > 0 else 0.0
        impulse = _clamp((move_atr - 0.35) / 2.65, 0.0, 1.0)

        if enough and i >= impulse_length and abs_sum[i] and abs_sum[i] > 0:
            efficiency = abs(c[i] - c[i - impulse_length]) / abs_sum[i]
        else:
            efficiency = 0.0

        half = max(2, int(math.floor(impulse_length / 2)))
        recent_v = prev_v = 0.0
        if enough and atr_i > 0 and i >= half * 2:
            recent_v = direction * (c[i] - c[i - half]) / (half * atr_i)
            prev_v = direction * (c[i - half] - c[i - half * 2]) / (half * atr_i)
        accel_delta = recent_v - prev_v
        acceleration = _clamp(0.5 + accel_delta * 1.5, 0.0, 1.0)

        alignment = bias * direction
        if alignment >= 1.0:
            structure_score = 1.0
        elif alignment >= 0.5:
            structure_score = 0.75
        elif alignment <= -1.0:
            structure_score = 0.0
        elif alignment <= -0.5:
            structure_score = 0.25
        else:
            structure_score = 0.5

        max_dist = atr_i * max_level_distance_atr
        cluster_width = atr_i * level_zone_atr
        resistance = _nearest_above(ph_prices, c[i], max_dist)
        support = _nearest_below(pl_prices, c[i], max_dist)
        barrier = resistance if direction > 0 else support
        pullback_level = support if direction > 0 else resistance
        half_life = analysis_lookback * 0.5
        if direction > 0:
            barrier_q, _, _ = _level_quality(ph_prices, ph_bars, barrier, cluster_width, level_recency_influence, half_life, i)
        else:
            barrier_q, _, _ = _level_quality(pl_prices, pl_bars, barrier, cluster_width, level_recency_influence, half_life, i)
        barrier_dist_atr = abs(barrier - c[i]) / atr_i if (barrier is not None and atr_i > 0) else max_level_distance_atr
        proximity = 1.0 - _clamp(barrier_dist_atr / max_level_distance_atr, 0.0, 1.0) if barrier is not None else 0.0
        rejection = _rejection_score(direction, barrier, atr_i * rejection_zone_atr, rejection_lookback, h, l, o, c, i)

        # target
        if direction > 0:
            if barrier is not None:
                next_target = _next_high(ph_prices, barrier, cluster_width, max_dist)
                if next_target is None:
                    next_target = barrier + fallback_target_atr * atr_i
            else:
                next_target = c[i] + fallback_target_atr * atr_i
        else:
            if barrier is not None:
                next_target = _next_low(pl_prices, barrier, cluster_width, max_dist)
                if next_target is None:
                    next_target = barrier - fallback_target_atr * atr_i
            else:
                next_target = c[i] - fallback_target_atr * atr_i

        structural_pullback = pullback_level
        atr_pullback = c[i] - direction * fallback_pullback_atr * atr_i
        impulse_pullback = c[i] - direction * impulse_retracement_ratio * abs(raw_move)
        pb_low = min(atr_pullback, impulse_pullback)
        pb_high = max(atr_pullback, impulse_pullback)
        if structural_pullback is not None:
            pb_low = min(pb_low, structural_pullback)
            pb_high = max(pb_high, structural_pullback)
        projected_pullback = structural_pullback if structural_pullback is not None else (atr_pullback + impulse_pullback) / 2.0

        if direction > 0:
            ft = _next_low(pl_prices, projected_pullback, cluster_width, max_dist)
            failure_target = ft if ft is not None else projected_pullback - fallback_target_atr * atr_i
        else:
            ft = _next_high(ph_prices, projected_pullback, cluster_width, max_dist)
            failure_target = ft if ft is not None else projected_pullback + fallback_target_atr * atr_i

        direct_score = (impulse * 0.24 + efficiency * 0.19 + acceleration * 0.17 + structure_score * 0.15
                        + (1.0 - barrier_q) * 0.06 + proximity * 0.04 + (1.0 - rejection) * 0.15)
        pullback_score = (structure_score * 0.19 + impulse * 0.15 + efficiency * 0.10
                          + (1.0 - acceleration) * 0.15 + barrier_q * 0.15 + proximity * 0.11
                          + rejection * 0.15)
        failure_score = ((1.0 - structure_score) * 0.26 + (1.0 - impulse) * 0.17
                         + (1.0 - efficiency) * 0.13 + (1.0 - acceleration) * 0.17
                         + barrier_q * 0.12 + rejection * 0.15)
        total = direct_score + pullback_score + failure_score
        if total > 0:
            dw = direct_score / total * 100
            pw = pullback_score / total * 100
            fw = failure_score / total * 100
        else:
            dw = pw = fw = 0.0
        dominant = max(dw, pw, fw)
        second = max(min(dw, pw), max(min(dw, fw), min(pw, fw)))
        gap = dominant - second

        setup_quality = 100 * (impulse * 0.30 + efficiency * 0.20 + structure_score * 0.20
                               + proximity * 0.15 + barrier_q * 0.15)
        valid_level = (not require_barrier) or (barrier is not None)
        valid[i] = 1 if (enough and valid_level and setup_quality >= minimum_setup_score) else 0

        cols["direction"][i] = direction
        cols["impulse"][i] = impulse
        cols["efficiency"][i] = efficiency
        cols["acceleration"][i] = acceleration
        cols["structure_bias"][i] = bias
        cols["structure_score"][i] = structure_score
        cols["barrier"][i] = barrier if barrier is not None else np.nan
        cols["barrier_quality"][i] = barrier_q
        cols["proximity"][i] = proximity
        cols["rejection"][i] = rejection
        cols["direct_score"][i] = direct_score
        cols["pullback_score"][i] = pullback_score
        cols["failure_score"][i] = failure_score
        cols["direct_weight"][i] = dw
        cols["pullback_weight"][i] = pw
        cols["failure_weight"][i] = fw
        cols["dominant_weight"][i] = dominant
        cols["dominance_gap"][i] = gap
        cols["target"][i] = next_target
        cols["failure_target"][i] = failure_target
        cols["pullback"][i] = projected_pullback
        cols["pullback_low"][i] = pb_low
        cols["pullback_high"][i] = pb_high
        cols["setup_quality"][i] = setup_quality

    df = pd.DataFrame(index=idx)
    df["msp_direction"] = pd.Series(cols["direction"], index=idx).astype("Int64")
    df["msp_impulse"] = cols["impulse"]
    df["msp_efficiency"] = cols["efficiency"]
    df["msp_acceleration"] = cols["acceleration"]
    df["msp_structure_bias"] = cols["structure_bias"]
    df["msp_structure_score"] = cols["structure_score"]
    df["msp_barrier"] = cols["barrier"]
    df["msp_barrier_quality"] = cols["barrier_quality"]
    df["msp_proximity"] = cols["proximity"]
    df["msp_rejection"] = cols["rejection"]
    df["msp_direct_score"] = cols["direct_score"]
    df["msp_pullback_score"] = cols["pullback_score"]
    df["msp_failure_score"] = cols["failure_score"]
    df["msp_direct_weight"] = cols["direct_weight"]
    df["msp_pullback_weight"] = cols["pullback_weight"]
    df["msp_failure_weight"] = cols["failure_weight"]
    df["msp_dominant_weight"] = cols["dominant_weight"]
    df["msp_dominance_gap"] = cols["dominance_gap"]
    df["msp_target"] = cols["target"]
    df["msp_failure_target"] = cols["failure_target"]
    df["msp_pullback"] = cols["pullback"]
    df["msp_pullback_low"] = cols["pullback_low"]
    df["msp_pullback_high"] = cols["pullback_high"]
    df["msp_setup_quality"] = cols["setup_quality"]
    df["msp_valid"] = pd.Series(valid, index=idx).astype("Int64")
    return df
