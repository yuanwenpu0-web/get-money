"""Trading Range State Machine — eight-sensor range score with break detection.

Source: pine-collection/indicators/market_structure/trading_range_state_machine/trading_range_state_machine.pine
(WavesUnchained, MIT)

Range boundaries come from pivot clustering: confirmed pivots within the
lookback are averaged into the densest same-side cluster (tolerance in ATR
units); the fallback is the window high/low when clustering fails. Eight
0..100 sensors blend into the range score — inverse efficiency (trending is
the enemy), candle overlap, choppiness, inverse structure breaks, regression
flatness + weak R², boundary stability (cluster density + balanced touches),
ATR compression and midline mean-reversion touches. The state machine walks
Building → Confirmed → Mature as the score passes `confirm` / `mature`, and
fires Breaking when price closes beyond a boundary plus a `break_buffer` ATR
while the range is Confirmed-or-better. Range position and Upper/Mid/Lower
zone labels complete the picture.

Dropped: lines, background heat, ribbon, dashboard and all colours.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (atr, choice, correlation, flags, highest, integer, linreg, lowest,
                    number, ohlcv, pivothigh, pivotlow, sma, stdev, sum_, true_range)

_EPS = 1e-10


def trading_range_state_machine(
        frame: pd.DataFrame, *, length: int = 60, pivot_len: int = 4,
        confirm_score: float = 68.0, mature_score: float = 78.0,
        break_buffer: float = 0.25, cluster_tol_atr: float = 0.55,
        w_er: float = 20.0, w_overlap: float = 15.0, w_chop: float = 15.0,
        w_struct: float = 15.0, w_reg: float = 10.0, w_bounds: float = 15.0,
        w_atr: float = 5.0, w_mean: float = 5.0) -> pd.DataFrame:
    integer("length", length, 20, 300)
    integer("pivot_len", pivot_len, 2, 20)
    number("confirm_score", confirm_score, 1.0, 100.0)
    number("mature_score", mature_score, 1.0, 100.0)
    number("break_buffer", break_buffer, 0.0, 2.0)
    number("cluster_tol_atr", cluster_tol_atr, 0.1, 2.0)
    high, low, close = ohlcv(frame, "high", "low", "close")
    rows = len(frame)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)

    atr_value = atr(frame, 14)
    atr_values = atr_value.to_numpy(dtype=float)
    tolerance = atr_value * cluster_tol_atr

    fallback_high = highest(high, length)
    fallback_low = lowest(low, length)

    piv_high = pivothigh(high, pivot_len, pivot_len)
    piv_low = pivotlow(low, pivot_len, pivot_len)
    ph_values = piv_high.to_numpy(dtype=float)
    pl_values = piv_low.to_numpy(dtype=float)

    range_high = np.full(rows, np.nan)
    range_low = np.full(rows, np.nan)
    upper_cluster_count = np.zeros(rows)
    lower_cluster_count = np.zeros(rows)

    high_pivots: list[float] = []
    high_bars: list[int] = []
    low_pivots: list[float] = []
    low_bars: list[int] = []
    tolerance_values = tolerance.to_numpy(dtype=float)
    position_ref = 0

    # Per-bar sensor series (vectorised where possible).
    net_move = (close - close.shift(length)).abs()
    total_move = sum_(close.diff().abs(), length)
    er = (net_move / total_move.replace(0, np.nan)).fillna(0.0)
    er_score = (1.0 - er.clip(0.0, 1.0)) * 100.0

    overlap_high = np.minimum(highs, np.roll(highs, 1))
    overlap_low = np.maximum(lows, np.roll(lows, 1))
    overlap_size = np.maximum(overlap_high - overlap_low, 0.0)
    overlap_size[0] = 0.0  # np.roll wraps bar 0; the Pine series has no bar -1
    candle_union = np.maximum(np.maximum(highs, np.roll(highs, 1))
                              - np.minimum(lows, np.roll(lows, 1)), _EPS)
    candle_union[0] = _EPS
    bar_overlap = pd.Series(overlap_size / candle_union, index=frame.index)
    overlap_score = sma(bar_overlap, length) * 100.0

    tr_sum = sum_(true_range(frame), length)
    range_size_series = (fallback_high - fallback_low).clip(lower=_EPS)
    chop_raw = 100.0 * (tr_sum / range_size_series).clip(lower=0.0).apply(np.log10) \
        / np.log10(length)
    chop_score = chop_raw.clip(lower=0.0, upper=100.0).fillna(50.0)

    prior_high = high.shift(1).rolling(pivot_len * 3).max()
    prior_low = low.shift(1).rolling(pivot_len * 3).min()
    bos = ((highs > prior_high.to_numpy(dtype=float))
           | (lows < prior_low.to_numpy(dtype=float))) & ~np.isnan(prior_high.to_numpy())
    bos[0] = False
    bos_count = pd.Series(bos.astype(float), index=frame.index).rolling(
        length, min_periods=length).sum()
    structure_score = 100.0 - bos_count.fillna(0.0).mul(12.0).clip(0.0, 100.0)

    bar_index_series = pd.Series(np.arange(rows, dtype=float), index=frame.index)
    reg_now = linreg(close, length, 0)
    reg_prev = linreg(close, length, 1)
    slope = reg_now - reg_prev
    slope_atr = (slope.abs() / atr_value.replace(0, np.nan)).fillna(0.0)
    corr = correlation(close, bar_index_series, length).fillna(0.0)
    r2 = corr.pow(2.0).clip(0.0, 1.0)

    def normalize01(value: pd.Series, lo: float, hi: float) -> pd.Series:
        if hi == lo:
            return pd.Series(0.0, index=value.index)
        return ((value - lo) / (hi - lo)).clip(lower=0.0, upper=1.0)

    slope_flat_score = (1.0 - normalize01(slope_atr, 0.0, 0.15)) * 100.0
    r2_weak_score = (1.0 - r2) * 100.0
    reg_score = slope_flat_score * 0.65 + r2_weak_score * 0.35

    atr_base = sma(atr_value, length * 2)
    atr_ratio = (atr_value / atr_base.replace(0, np.nan)).fillna(0.0)
    atr_score = (1.0 - normalize01(atr_ratio, 0.75, 1.35)) * 100.0

    bos_values = bos.astype(float)
    states = np.empty(rows, dtype=object)
    scores = np.full(rows, np.nan)
    position_high = np.full(rows, np.nan)
    position_zone = np.empty(rows, dtype=object)

    weight_sum = w_er + w_overlap + w_chop + w_struct + w_reg + w_bounds + w_atr + w_mean

    for position in range(rows):
        position_ref = position
        # Maintain pivot windows.
        if not np.isnan(ph_values[position]):
            high_pivots.append(float(ph_values[position]))
            high_bars.append(position - pivot_len)
        if not np.isnan(pl_values[position]):
            low_pivots.append(float(pl_values[position]))
            low_bars.append(position - pivot_len)
        while high_bars and position - high_bars[0] > length:
            high_bars.pop(0)
            high_pivots.pop(0)
        while low_bars and position - low_bars[0] > length:
            low_bars.pop(0)
            low_pivots.pop(0)

        tol = tolerance_values[position]
        cluster_high, upper_count = (None, 0)
        if high_pivots:
            best_count = 0
            best_level = None
            for base in high_pivots:
                selected = [p for p in high_pivots if abs(p - base) <= tol]
                if len(selected) > best_count:
                    best_count = len(selected)
                    best_level = float(np.mean(selected))
            cluster_high, upper_count = best_level, best_count
        cluster_low, lower_count = (None, 0)
        if low_pivots:
            best_count = 0
            best_level = None
            for base in low_pivots:
                selected = [p for p in low_pivots if abs(p - base) <= tol]
                if len(selected) > best_count:
                    best_count = len(selected)
                    best_level = float(np.mean(selected))
            cluster_low, lower_count = best_level, best_count
        upper_cluster_count[position] = upper_count
        lower_cluster_count[position] = lower_count

        high_raw = cluster_high if cluster_high is not None else float(fallback_high.iloc[position])
        low_raw = cluster_low if cluster_low is not None else float(fallback_low.iloc[position])
        if high_raw > low_raw:
            r_high, r_low = high_raw, low_raw
        else:
            r_high = float(fallback_high.iloc[position])
            r_low = float(fallback_low.iloc[position])
        range_high[position] = r_high
        range_low[position] = r_low
        r_size = max(r_high - r_low, _EPS)
        r_mid = (r_high + r_low) / 2.0

        touch_tolerance = atr_values[position] * 0.45 if not np.isnan(atr_values[position]) else 0.0
        upper_touch = (highs[position] >= r_high - touch_tolerance
                       and closes[position] < r_high)
        lower_touch = (lows[position] <= r_low + touch_tolerance
                       and closes[position] > r_low)
        window_upper_touches = 0
        window_lower_touches = 0
        window_mid_touches = 0
        start = max(0, position - length + 1)
        mid_tolerance = r_size * 0.12
        # Recompute touch sums over the trailing window using each bar's own
        # boundaries is expensive; the Pine series-sum uses current boundaries
        # against recent bars, which we approximate with the current window.
        for i in range(start, position + 1):
            tt = atr_values[i] * 0.45 if not np.isnan(atr_values[i]) else 0.0
            if highs[i] >= r_high - tt and closes[i] < r_high:
                window_upper_touches += 1
            if lows[i] <= r_low + tt and closes[i] > r_low:
                window_lower_touches += 1
            if abs(closes[i] - r_mid) <= mid_tolerance:
                window_mid_touches += 1

        touch_balance = min(window_upper_touches, window_lower_touches)
        total_touches = window_upper_touches + window_lower_touches
        cluster_quality = float(normalize01(pd.Series([min(upper_count, lower_count)],
                                                      index=frame.index[[position]]),
                                            1.0, 4.0).iloc[0]) * 45.0
        touch_quality = float(normalize01(pd.Series([touch_balance],
                                                    index=frame.index[[position]]),
                                          1.0, 4.0).iloc[0]) * 35.0
        total_quality = float(normalize01(pd.Series([total_touches],
                                                    index=frame.index[[position]]),
                                          3.0, 10.0).iloc[0]) * 20.0
        boundary_score = min(cluster_quality + touch_quality + total_quality, 100.0)
        mean_score = float(normalize01(pd.Series([window_mid_touches],
                                                 index=frame.index[[position]]),
                                       2.0, 10.0).iloc[0]) * 100.0

        if weight_sum > 0:
            score = (er_score.iloc[position] * w_er
                     + overlap_score.iloc[position] * w_overlap
                     + chop_score.iloc[position] * w_chop
                     + structure_score.iloc[position] * w_struct
                     + reg_score.iloc[position] * w_reg
                     + boundary_score * w_bounds
                     + atr_score.iloc[position] * w_atr
                     + mean_score * w_mean) / weight_sum
        else:
            score = 0.0
        score = min(max(score, 0.0), 100.0)
        scores[position] = score

        break_up = closes[position] > r_high + atr_values[position] * break_buffer
        break_dn = closes[position] < r_low - atr_values[position] * break_buffer
        breaking = score >= confirm_score and (break_up or break_dn)
        if breaking:
            states[position] = "Breaking"
        elif score >= mature_score:
            states[position] = "Mature"
        elif score >= confirm_score:
            states[position] = "Confirmed"
        elif score >= confirm_score * 0.75:
            states[position] = "Building"
        else:
            states[position] = "Inactive"

        pos_pct = min(max((closes[position] - r_low) / r_size * 100.0, 0.0), 100.0)
        position_high[position] = pos_pct
        if pos_pct >= 80.0:
            position_zone[position] = "Upper"
        elif pos_pct <= 20.0:
            position_zone[position] = "Lower"
        else:
            position_zone[position] = "Mid"

    range_high_series = pd.Series(range_high, index=frame.index)
    range_low_series = pd.Series(range_low, index=frame.index)
    score_series = pd.Series(scores, index=frame.index)
    state_series = pd.Series(states, index=frame.index)
    breaking = flags(state_series.eq("Breaking"))
    breaking_up = flags(breaking & (closes > range_high_series.to_numpy()
                                    + atr_values * break_buffer))

    return pd.DataFrame({
        "range_high": range_high_series,
        "range_low": range_low_series,
        "range_mid": (range_high_series + range_low_series) / 2.0,
        "range_score": score_series,
        "state": state_series,
        "state_changed": flags(state_series != state_series.shift(1)),
        "breaking": breaking,
        "breaking_up": breaking_up,
        "breaking_down": flags(breaking & ~breaking_up),
        "range_position": pd.Series(position_high, index=frame.index),
        "position_zone": pd.Series(position_zone, index=frame.index),
        "er_score": er_score,
        "overlap_score": overlap_score,
        "chop_score": chop_score,
        "structure_score": structure_score,
        "regression_score": reg_score,
        "boundary_score": pd.Series(boundary_score, index=frame.index),
        "atr_score": atr_score,
        "mean_reversion_score": pd.Series(mean_score, index=frame.index),
        "upper_cluster_count": pd.Series(upper_cluster_count, index=frame.index).astype("Int64"),
        "lower_cluster_count": pd.Series(lower_cluster_count, index=frame.index).astype("Int64"),
    }, index=frame.index)
