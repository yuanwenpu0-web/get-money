"""Harmonic Pattern Recognition (v0.0.1) — catalog port.

Source: data/pine-crypto-collection/lucky_harmonic/src/harmonic_patterns.pine

Detects four harmonic formations from the last four confirmed pivots (X, A, B,
C) and projects the D completion price:

  * Bat        — AB/XA in [0.382, 0.5]  · BC/AB in [0.382, 0.886] · D = 0.886 XA
  * Butterfly  — AB/XA ≈ 0.786          · BC/AB in [0.382, 0.886] · D = 1.272 XA
  * Crab       — AB/XA in [0.382, 0.618]· BC/AB in [0.382, 0.886] · D = 1.618 XA
  * Shark      — AB/XA ≈ 0.5            · BC/AB in [1.13, 1.618]  · D = 0.886 XA

Pivots follow the source's ``find_pivots_safe`` rule: a bar is a pivot high/low
when its **close** is a strict N-bar extreme versus the previous N bars (left
side only, N = max(5, floor(lookback/10))).

The best-quality valid pattern (crab 0.85 > bat 0.80 > butterfly 0.75 > shark
0.70, gated by ``min_quality_for_alert``) becomes the bar's trade signal.

Display (labels/lines/alerts/error stats) is dropped. Numeric output is the
per-bar signal: pattern code/direction/quality, entry/stop/targets and the four
pivot prices.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import boolean, choice, integer, number, ohlcv

FIB_382 = 0.382
FIB_5 = 0.5
FIB_618 = 0.618
FIB_786 = 0.786
FIB_886 = 0.886
FIB_1272 = 1.272
FIB_1618 = 1.618

PATTERN_CODE = {"None": 0, "Bat": 1, "Butterfly": 2, "Crab": 3, "Shark": 4}


def _calculate_ratio(move1, move2):
    if math.isnan(move1) or math.isnan(move2) or move1 == 0:
        return float("nan")
    return move2 / move1


def _validate_ratio(actual, expected, tolerance):
    if math.isnan(actual) or math.isnan(expected):
        return False
    return (expected - tolerance) <= actual <= (expected + tolerance)


def _validate_bat(x, a, b, c, tolerance):
    xa, ab, bc = a - x, b - a, c - b
    ab_xa = _calculate_ratio(xa, ab)
    bc_ab = _calculate_ratio(ab, bc)
    ab_xa_valid = (_validate_ratio(ab_xa, FIB_382, tolerance)
                   or _validate_ratio(ab_xa, FIB_5, tolerance)
                   or (FIB_382 - tolerance < ab_xa < FIB_5 + tolerance))
    bc_ab_valid = (_validate_ratio(bc_ab, FIB_382, tolerance)
                   or _validate_ratio(bc_ab, FIB_786, tolerance)
                   or (FIB_382 - tolerance < bc_ab < FIB_786 + tolerance))
    return ab_xa_valid and bc_ab_valid, a - xa * FIB_886


def _validate_butterfly(x, a, b, c, tolerance):
    xa, ab, bc = a - x, b - a, c - b
    ab_xa = _calculate_ratio(xa, ab)
    bc_ab = _calculate_ratio(ab, bc)
    ab_xa_valid = _validate_ratio(ab_xa, FIB_786, tolerance)
    bc_ab_valid = (_validate_ratio(bc_ab, FIB_382, tolerance)
                   or _validate_ratio(bc_ab, FIB_886, tolerance)
                   or (FIB_382 - tolerance < bc_ab < FIB_886 + tolerance))
    return ab_xa_valid and bc_ab_valid, a - xa * FIB_1272


def _validate_crab(x, a, b, c, tolerance):
    xa, ab, bc = a - x, b - a, c - b
    ab_xa = _calculate_ratio(xa, ab)
    bc_ab = _calculate_ratio(ab, bc)
    ab_xa_valid = (_validate_ratio(ab_xa, FIB_382, tolerance)
                   or _validate_ratio(ab_xa, FIB_618, tolerance)
                   or (FIB_382 - tolerance < ab_xa < FIB_618 + tolerance))
    bc_ab_valid = (_validate_ratio(bc_ab, FIB_382, tolerance)
                   or _validate_ratio(bc_ab, FIB_886, tolerance)
                   or (FIB_382 - tolerance < bc_ab < FIB_886 + tolerance))
    return ab_xa_valid and bc_ab_valid, a - xa * FIB_1618


def _validate_shark(x, a, b, c, tolerance):
    xa, ab, bc = a - x, b - a, c - b
    ab_xa = _calculate_ratio(xa, ab)
    bc_ab = _calculate_ratio(ab, bc)
    ab_xa_valid = _validate_ratio(ab_xa, FIB_5, tolerance)
    bc_ab_valid = (_validate_ratio(bc_ab, 1.13, tolerance)
                   or _validate_ratio(bc_ab, FIB_1618, tolerance)
                   or (1.13 - tolerance < bc_ab < FIB_1618 + tolerance))
    return ab_xa_valid and bc_ab_valid, a - xa * FIB_886


def _find_pivots(close, i, length):
    if i < length:
        return 0.0, 0.0
    pivot_high = pivot_low = 0.0
    is_high = all(close[i] > close[i - k] for k in range(1, length + 1))
    is_low = all(close[i] < close[i - k] for k in range(1, length + 1))
    if is_high:
        pivot_high = close[i]
    if is_low:
        pivot_low = close[i]
    return pivot_high, pivot_low


def harmonic_patterns(
    frame: pd.DataFrame,
    *,
    mode: str = "Auto Detection",
    lookback: int = 100,
    tolerance: float = 0.05,
    enable_bat: bool = True,
    enable_butterfly: bool = True,
    enable_crab: bool = True,
    enable_shark: bool = True,
    min_quality_for_alert: float = 0.6,
    risk_percent: float = 0.01,
) -> pd.DataFrame:
    mode = choice("mode", mode, ("Auto Detection", "Manual Drawing"))
    lookback = integer("lookback", lookback, 20, 500)
    tolerance = number("tolerance", tolerance, 0.01)
    min_quality_for_alert = number("min_quality_for_alert", min_quality_for_alert, 0.1, 1.0)
    risk_percent = number("risk_percent", risk_percent, 0.0, 1.0)
    enable_bat = boolean("enable_bat", enable_bat)
    enable_butterfly = boolean("enable_butterfly", enable_butterfly)
    enable_crab = boolean("enable_crab", enable_crab)
    enable_shark = boolean("enable_shark", enable_shark)

    close = ohlcv(frame, "close")[0].to_numpy(dtype=float)
    idx = frame.index
    n = len(close)

    pivot_length = max(5, int(math.floor(lookback / 10)))

    piv_x: list[int] = []
    piv_y: list[float] = []
    piv_t: list[int] = []

    valid_arr = np.zeros(n, dtype=np.int64)
    code_arr = np.zeros(n, dtype=np.int64)
    dir_arr = np.zeros(n, dtype=np.int64)
    quality_arr = np.zeros(n, dtype=float)
    entry_arr = np.full(n, np.nan)
    stop_arr = np.full(n, np.nan)
    t1_arr = np.full(n, np.nan)
    t2_arr = np.full(n, np.nan)
    rr1_arr = np.full(n, np.nan)
    rr2_arr = np.full(n, np.nan)
    x_arr = np.full(n, np.nan)
    a_arr = np.full(n, np.nan)
    b_arr = np.full(n, np.nan)
    c_arr = np.full(n, np.nan)
    count_arr = np.zeros(n, dtype=np.int64)

    def make_signal(name, quality, direction, d_price, a_price):
        stop = d_price * (1 - risk_percent) if direction == 1 else d_price * (1 + risk_percent)
        d_to_a = abs(a_price - d_price)
        if direction == 1:
            t1 = d_price + d_to_a * 0.382
            t2 = d_price + d_to_a * 0.618
        else:
            t1 = d_price - d_to_a * 0.382
            t2 = d_price - d_to_a * 0.618
        risk = abs(d_price - stop)
        r1 = abs(d_price - t1) / risk if risk > 0 else 0.0
        r2 = abs(d_price - t2) / risk if risk > 0 else 0.0
        return name, quality, direction, d_price, stop, t1, t2, r1, r2

    for i in range(n):
        ph, pl = _find_pivots(close, i, pivot_length)
        if ph > 0:
            piv_x.append(i)
            piv_y.append(ph)
            piv_t.append(1)
        if pl > 0:
            piv_x.append(i)
            piv_y.append(pl)
            piv_t.append(-1)
        if piv_x and i - piv_x[0] > lookback:
            piv_x.pop(0)
            piv_y.pop(0)
            piv_t.pop(0)

        count_arr[i] = len(piv_x)

        if len(piv_x) >= 4:
            size = len(piv_x)
            x_price = piv_y[size - 4]
            a_price = piv_y[size - 3]
            b_price = piv_y[size - 2]
            c_price = piv_y[size - 1]
            x_arr[i] = x_price
            a_arr[i] = a_price
            b_arr[i] = b_price
            c_arr[i] = c_price
            direction = -1 if a_price > x_price else 1

            best = None
            best_q = 0.0
            if enable_bat:
                ok, d = _validate_bat(x_price, a_price, b_price, c_price, tolerance)
                if ok and 0.8 >= min_quality_for_alert and 0.8 > best_q:
                    best, best_q = make_signal("Bat", 0.8, direction, d, a_price), 0.8
            if enable_butterfly:
                ok, d = _validate_butterfly(x_price, a_price, b_price, c_price, tolerance)
                if ok and 0.75 >= min_quality_for_alert and 0.75 > best_q:
                    best, best_q = make_signal("Butterfly", 0.75, direction, d, a_price), 0.75
            if enable_crab:
                ok, d = _validate_crab(x_price, a_price, b_price, c_price, tolerance)
                if ok and 0.85 >= min_quality_for_alert and 0.85 > best_q:
                    best, best_q = make_signal("Crab", 0.85, direction, d, a_price), 0.85
            if enable_shark:
                ok, d = _validate_shark(x_price, a_price, b_price, c_price, tolerance)
                if ok and 0.7 >= min_quality_for_alert and 0.7 > best_q:
                    best, best_q = make_signal("Shark", 0.7, direction, d, a_price), 0.7

            if best is not None:
                name, quality, direction, d_price, stop, t1, t2, r1, r2 = best
                valid_arr[i] = 1
                code_arr[i] = PATTERN_CODE[name]
                dir_arr[i] = direction
                quality_arr[i] = quality
                entry_arr[i] = d_price
                stop_arr[i] = stop
                t1_arr[i] = t1
                t2_arr[i] = t2
                rr1_arr[i] = r1
                rr2_arr[i] = r2

    out = pd.DataFrame(index=idx)
    out["harmonic_valid"] = pd.Series(valid_arr, index=idx).astype("Int64")
    out["harmonic_code"] = pd.Series(code_arr, index=idx).astype("Int64")
    out["harmonic_direction"] = pd.Series(dir_arr, index=idx).astype("Int64")
    out["harmonic_quality"] = quality_arr
    out["harmonic_entry"] = entry_arr
    out["harmonic_stop_loss"] = stop_arr
    out["harmonic_target1"] = t1_arr
    out["harmonic_target2"] = t2_arr
    out["harmonic_rr1"] = rr1_arr
    out["harmonic_rr2"] = rr2_arr
    out["harmonic_x_price"] = x_arr
    out["harmonic_a_price"] = a_arr
    out["harmonic_b_price"] = b_arr
    out["harmonic_c_price"] = c_arr
    out["harmonic_pivot_count"] = pd.Series(count_arr, index=idx).astype("Int64")
    return out
