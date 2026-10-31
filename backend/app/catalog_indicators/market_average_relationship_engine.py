"""Market–Average Relationship Engine [WavesUnchained]  v1.3.2  —  catalog port.

Source: data/pine-collection/indicators/trend_direction/
        market_average_relationship_engine/market_average_relationship_engine.pine

Evaluates the relationship between price and one consistently selected moving
average across six diagnostics — Trend, Respect, Extension, Compression,
Acceleration, Exhaustion — combined into a -100..+100 relationship oscillator
with a state engine and event family (pullback, momentum release, do-not-chase,
exhaustion warning, relationship break, divergence, turn-from-extreme).

Ported computational core; drawing/labels/tables/alerts and the generated
strategy block are dropped.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import (
    alma, atr, choice, crossover, crossunder, ema, falling, hma, integer, number,
    ohlcv, pivothigh_strict, pivotlow_strict, rma, rising, sma, stdev, vwma, wma,
)

_MINTICK = 1e-8


def _clamp(v, lo, hi):
    if math.isnan(v):
        return v
    return max(lo, min(hi, v))


def _normalize(v, lo, hi):
    return _clamp((v - lo) / (hi - lo), 0.0, 1.0) if hi > lo else 0.0


def _safe_div(a, b, fb=0.0):
    return a / b if abs(b) > 1e-10 else fb


def market_average_relationship_engine(
    frame: pd.DataFrame,
    *,
    ma_type: str = "EMA",
    ma_length: int = 50,
    alma_offset: float = 0.85,
    alma_sigma: float = 6.0,
    atr_length: int = 14,
    slope_length: int = 10,
    full_slope_atr: float = 0.50,
    persistence_length: int = 20,
    efficiency_length: int = 14,
    statistics_length: int = 100,
    extension_normal_atr: float = 0.75,
    extension_extreme_atr: float = 2.50,
    proximity_atr: float = 0.35,
    respect_lookback: int = 80,
    respect_response_bars: int = 4,
    compression_length: int = 20,
    compression_band_atr: float = 0.45,
    acceleration_length: int = 5,
    exhaustion_length: int = 8,
    signal_cooldown: int = 8,
    minimum_trend: float = 55.0,
    minimum_respect: float = 55.0,
    pullback_prior_extension: float = 0.70,
    maximum_pullback_exhaustion: float = 65.0,
    acceleration_threshold: float = 60.0,
    compression_threshold: float = 65.0,
    chase_extension_threshold: float = 75.0,
    exhaustion_threshold: float = 68.0,
    break_confirmation: int = 3,
    divergence_pivot_left: int = 5,
    divergence_pivot_right: int = 5,
    turn_lookback: int = 30,
    turn_extreme_level: float = 55.0,
    turn_confirm_bars: int = 3,
) -> pd.DataFrame:
    ma_type = choice("ma_type", ma_type, ("EMA", "SMA", "WMA", "HMA", "VWMA", "RMA", "DEMA", "TEMA", "ALMA"))
    ma_length = integer("ma_length", ma_length, 2, 1000)
    atr_length = integer("atr_length", atr_length, 2, 200)
    slope_length = integer("slope_length", slope_length, 1, 100)
    persistence_length = integer("persistence_length", persistence_length, 3, 200)
    efficiency_length = integer("efficiency_length", efficiency_length, 3, 100)
    statistics_length = integer("statistics_length", statistics_length, 20, 500)
    respect_lookback = integer("respect_lookback", respect_lookback, 20, 250)
    respect_response_bars = integer("respect_response_bars", respect_response_bars, 1, 20)
    compression_length = integer("compression_length", compression_length, 5, 100)
    acceleration_length = integer("acceleration_length", acceleration_length, 1, 50)
    exhaustion_length = integer("exhaustion_length", exhaustion_length, 2, 50)
    signal_cooldown = integer("signal_cooldown", signal_cooldown, 0, 100)
    break_confirmation = integer("break_confirmation", break_confirmation, 1, 20)
    divergence_pivot_left = integer("divergence_pivot_left", divergence_pivot_left, 2, 30)
    divergence_pivot_right = integer("divergence_pivot_right", divergence_pivot_right, 2, 30)
    turn_lookback = integer("turn_lookback", turn_lookback, 5, 150)
    turn_confirm_bars = integer("turn_confirm_bars", turn_confirm_bars, 2, 20)
    full_slope_atr = number("full_slope_atr", full_slope_atr, 0.05, 5.0)
    extension_normal_atr = number("extension_normal_atr", extension_normal_atr, 0.05, 10.0)
    extension_extreme_atr = number("extension_extreme_atr", extension_extreme_atr, 0.10, 20.0)
    proximity_atr = number("proximity_atr", proximity_atr, 0.05, 3.0)
    compression_band_atr = number("compression_band_atr", compression_band_atr, 0.05, 3.0)
    minimum_trend = number("minimum_trend", minimum_trend, 0.0, 100.0)
    minimum_respect = number("minimum_respect", minimum_respect, 0.0, 100.0)

    o, h, l, c, vol = ohlcv(frame, "open", "high", "low", "close", "volume")
    o = o.to_numpy(dtype=float); h = h.to_numpy(dtype=float)
    l = l.to_numpy(dtype=float); c = c.to_numpy(dtype=float); vol = vol.to_numpy(dtype=float)
    idx = frame.index
    n = len(frame)
    src = frame["close"]

    def moving_average():
        ema1 = ema(src, ma_length).to_numpy()
        if ma_type == "EMA":
            return ema1
        if ma_type == "SMA":
            return sma(src, ma_length).to_numpy()
        if ma_type == "WMA":
            return wma(src, ma_length).to_numpy()
        if ma_type == "HMA":
            return hma(src, ma_length).to_numpy()
        if ma_type == "VWMA":
            return vwma(src, pd.Series(vol, index=src.index), ma_length).to_numpy()
        if ma_type == "RMA":
            return rma(src, ma_length).to_numpy()
        if ma_type == "ALMA":
            return alma(src, ma_length, alma_offset, alma_sigma).to_numpy()
        ema2 = ema(pd.Series(ema1), ma_length).to_numpy()
        if ma_type == "DEMA":
            return 2.0 * ema1 - ema2
        ema3 = ema(pd.Series(ema2), ma_length).to_numpy()
        return 3.0 * ema1 - 3.0 * ema2 + ema3

    ma = moving_average()
    atr_a = np.maximum(atr(frame, atr_length).to_numpy(dtype=float), _MINTICK)
    abs_dist = np.abs(c - ma) / atr_a
    signed_dist = (c - ma) / atr_a
    ma_slope_atr = np.array([_safe_div(ma[i] - ma[i - slope_length], atr_a[i], 0.0) if i >= slope_length else 0.0 for i in range(n)])
    ma_slope_strength = 100.0 * np.array([_normalize(abs(ma_slope_atr[i]), 0.0, full_slope_atr) for i in range(n)])
    ma_dir = np.where(ma_slope_atr > full_slope_atr * 0.05, 1.0, np.where(ma_slope_atr < -full_slope_atr * 0.05, -1.0, 0.0))
    price_side = np.where(c - ma > atr_a * 0.03, 1.0, np.where(c - ma < -atr_a * 0.03, -1.0, 0.0))

    bull_side = sma(pd.Series((c > ma).astype(float)), persistence_length).to_numpy()
    bear_side = sma(pd.Series((c < ma).astype(float)), persistence_length).to_numpy()
    side_persist = np.where(price_side > 0, bull_side, np.where(price_side < 0, bear_side, 0.5))
    bull_slope = sma(pd.Series((np.concatenate([[np.nan], ma[:-1]]) < ma).astype(float)), persistence_length).to_numpy()
    bear_slope = sma(pd.Series((np.concatenate([[np.nan], ma[:-1]]) > ma).astype(float)), persistence_length).to_numpy()
    slope_persist = np.where(ma_dir > 0, bull_slope, np.where(ma_dir < 0, bear_slope, 0.5))
    # bullish slope ratio: ma > ma[1]
    bull_slope = sma(pd.Series((ma > np.concatenate([[np.nan], ma[:-1]])).astype(float)), persistence_length).to_numpy()
    slope_persist = np.where(ma_dir > 0, bull_slope, np.where(ma_dir < 0, 1.0 - bull_slope, 0.5))
    path_travel = sma(pd.Series(np.abs(np.diff(c, prepend=c[0]))), efficiency_length).to_numpy() * efficiency_length
    net_travel = np.array([abs(c[i] - c[i - efficiency_length]) if i >= efficiency_length else np.nan for i in range(n)])
    price_eff = np.clip(np.array([_safe_div(net_travel[i], path_travel[i], 0.0) for i in range(n)]), 0.0, 1.0)

    direction_agree = np.where((price_side != 0) & (ma_dir != 0) & (price_side == ma_dir), 1.0,
                               np.where((price_side != 0) & (ma_dir == 0), 0.45,
                                        np.where(price_side == 0, 0.25, 0.0)))
    trend_quality = 100.0 * np.clip(0.30 * (ma_slope_strength / 100.0) + 0.25 * side_persist
                                    + 0.20 * slope_persist + 0.15 * price_eff + 0.10 * direction_agree, 0.0, 1.0)
    trend_dir = np.where(direction_agree >= 1.0, price_side, np.where(ma_dir != 0, ma_dir, price_side))
    signed_trend = trend_quality * trend_dir

    def respect_stats(cur):
        succ = fail = weighted = 0.0
        for k in range(respect_response_bars, respect_lookback + 1):
            j = cur - k
            jb = cur - (k + 1)
            jo = cur - (k - respect_response_bars)
            if j < 0 or jb < 0 or jo < 0:
                continue
            atr_touch = atr_a[j]; ma_touch = ma[j]
            if math.isnan(atr_touch) or math.isnan(ma_touch):
                continue
            dist = abs(c[j] - ma_touch)
            touched = dist <= atr_touch * proximity_atr
            atr_before = atr_a[jb]; ma_before = ma[jb]
            touched_before = (not math.isnan(atr_before)) and (not math.isnan(ma_before)) and abs(c[jb] - ma_before) <= atr_before * proximity_atr
            if touched and not touched_before:
                local_slope = ma_touch - ma[jb]
                trend_side = 1.0 if local_slope >= 0 else -1.0
                outcome_close = c[jo]; outcome_ma = ma[jo]; outcome_atr = atr_a[jo]
                progress = _safe_div((outcome_close - c[j]) * trend_side, outcome_atr, 0.0)
                held = (outcome_close - outcome_ma) * trend_side > 0
                succeeded = held and progress > 0
                succ += 1.0 if succeeded else 0.0
                fail += 0.0 if succeeded else 1.0
                weighted += _clamp(progress, -1.0, 1.0)
        samples = succ + fail
        hit = succ / samples if samples > 0 else 0.5
        outcome_q = _normalize(weighted / samples, -0.35, 0.50) if samples > 0 else 0.5
        confidence = _normalize(samples, 2.0, 10.0)
        raw = 100.0 * (0.70 * hit + 0.30 * outcome_q)
        return _clamp(50.0 + (raw - 50.0) * confidence, 0.0, 100.0)

    respect_score = np.array([respect_stats(i) for i in range(n)])
    # side flip rate: cross of close and ma
    cross_sign = ((c > ma) & (np.concatenate([[np.nan], c[:-1]]) <= np.concatenate([[np.nan], ma[:-1]]))) | \
                 ((c < ma) & (np.concatenate([[np.nan], c[:-1]]) >= np.concatenate([[np.nan], ma[:-1]])))
    side_flip = sma(pd.Series(cross_sign.astype(float)), persistence_length).to_numpy()
    current_stability = 100.0 * (1.0 - np.clip(side_flip * 4.0, 0.0, 1.0))
    respect_score = np.clip(0.80 * respect_score + 0.20 * current_stability, 0.0, 100.0)

    ext_atr_comp = 100.0 * np.array([_normalize(abs_dist[i], extension_normal_atr, extension_extreme_atr) for i in range(n)])
    dist_mean = sma(pd.Series(abs_dist), statistics_length).to_numpy()
    dist_sd = stdev(pd.Series(abs_dist), statistics_length).to_numpy()
    ext_z = np.array([_safe_div(abs_dist[i] - dist_mean[i], dist_sd[i], 0.0) for i in range(n)])
    ext_rarity = 100.0 * np.array([_normalize(ext_z[i], 0.0, 2.5) for i in range(n)])
    extension_score = np.clip(0.70 * ext_atr_comp + 0.30 * ext_rarity, 0.0, 100.0)

    near_ratio = sma(pd.Series((abs_dist <= compression_band_atr).astype(float)), compression_length).to_numpy()
    avg_dist = sma(pd.Series(abs_dist), compression_length).to_numpy()
    tightness = 1.0 - np.array([_normalize(avg_dist[i], compression_band_atr * 0.40, compression_band_atr * 1.80) for i in range(n)])
    atr_base = sma(pd.Series(atr_a), statistics_length).to_numpy()
    vol_contr = 1.0 - np.array([_normalize(_safe_div(atr_a[i], atr_base[i], 1.0), 0.70, 1.20) for i in range(n)])
    compression_score = 100.0 * np.clip(0.50 * near_ratio + 0.30 * tightness + 0.20 * vol_contr, 0.0, 1.0)

    directed = signed_dist * trend_dir
    sep_change = np.array([directed[i] - directed[i - acceleration_length] if i >= acceleration_length else 0.0 for i in range(n)])
    sep_vel = sep_change / acceleration_length
    current_slope_vel = (ma_slope_atr / slope_length) * trend_dir
    prior_slope = np.array([_safe_div(ma[i - acceleration_length] - ma[i - acceleration_length - slope_length],
                                      atr_a[i - acceleration_length], 0.0) if i >= acceleration_length + slope_length else 0.0 for i in range(n)])
    prior_slope_vel = (prior_slope / slope_length) * trend_dir
    slope_accel = current_slope_vel - prior_slope_vel
    recent_ret = np.array([_safe_div(c[i] - c[i - acceleration_length], atr_a[i], 0.0) * trend_dir[i] if i >= acceleration_length else 0.0 for i in range(n)])
    prior_ret = np.array([_safe_div(c[i - acceleration_length] - c[i - 2 * acceleration_length], atr_a[i - acceleration_length], 0.0) * trend_dir[i] if i >= 2 * acceleration_length else 0.0 for i in range(n)])
    ret_accel = recent_ret - prior_ret
    sep_acc_score = 100.0 * np.array([_normalize(sep_vel[i], 0.0, 0.25) for i in range(n)])
    slope_acc_score = 100.0 * np.array([_normalize(slope_accel[i], 0.0, 0.04) for i in range(n)])
    ret_acc_score = 100.0 * np.array([_normalize(ret_accel[i], 0.0, 1.50) for i in range(n)])
    acceleration_score = np.clip(0.50 * sep_acc_score + 0.20 * slope_acc_score + 0.30 * ret_acc_score, 0.0, 100.0)
    acceleration_score = np.nan_to_num(acceleration_score, nan=0.0)
    price_eff = np.nan_to_num(price_eff, nan=0.0)

    candle_range = np.maximum(h - l, _MINTICK)
    upper_wick = h - np.maximum(o, c); lower_wick = np.minimum(o, c) - l
    opposing = np.where(trend_dir > 0, upper_wick / candle_range, np.where(trend_dir < 0, lower_wick / candle_range, 0.0))
    wick_pressure = sma(pd.Series(opposing), exhaustion_length).to_numpy()
    eff_prior = np.array([price_eff[i - exhaustion_length] if i >= exhaustion_length else np.nan for i in range(n)])
    eff_decay = np.array([_normalize(eff_prior[i] - price_eff[i], 0.0, 0.35) for i in range(n)])
    dir_progress = np.array([_safe_div((c[i] - c[i - exhaustion_length]) * trend_dir[i], atr_a[i], 0.0) if i >= exhaustion_length else 0.0 for i in range(n)])
    progress_fail = 1.0 - np.array([_normalize(dir_progress[i], 0.0, 1.50) for i in range(n)])
    accel_prior = np.array([acceleration_score[i - exhaustion_length] if i >= exhaustion_length else np.nan for i in range(n)])
    accel_loss = np.array([_normalize(accel_prior[i] - acceleration_score[i], 0.0, 50.0) for i in range(n)])
    ext_pre = extension_score / 100.0
    exhaustion_score = 100.0 * np.clip(ext_pre * (0.35 * eff_decay + 0.25 * np.nan_to_num(wick_pressure, nan=0.0)
                                                 + 0.25 * progress_fail + 0.15 * np.nan_to_num(accel_loss, nan=0.0)), 0.0, 1.0)

    ext_pen = np.array([_normalize(extension_score[i], chase_extension_threshold, 100.0) for i in range(n)])
    exh_pen = np.array([_normalize(exhaustion_score[i], exhaustion_threshold, 100.0) for i in range(n)])
    quality_pen = np.clip(0.18 * ext_pen + 0.32 * exh_pen, 0.0, 0.50)
    relationship_raw = signed_trend * (1.0 - quality_pen)
    relationship = ema(pd.Series(relationship_raw), 3).to_numpy()
    relationship_signal = ema(pd.Series(relationship), 5).to_numpy()

    # state engine
    state_code = np.zeros(n, dtype=np.int64)
    for i in range(n):
        td = trend_dir[i]; tq = trend_quality[i]
        compression_state = compression_score[i] >= compression_threshold and tq < minimum_trend
        bull_accel = td > 0 and tq >= minimum_trend and acceleration_score[i] >= acceleration_threshold and exhaustion_score[i] < exhaustion_threshold
        bear_accel = td < 0 and tq >= minimum_trend and acceleration_score[i] >= acceleration_threshold and exhaustion_score[i] < exhaustion_threshold
        overext = tq >= minimum_trend and extension_score[i] >= chase_extension_threshold and exhaustion_score[i] < exhaustion_threshold
        exh_state = tq >= minimum_trend and exhaustion_score[i] >= exhaustion_threshold
        healthy = (tq >= minimum_trend and respect_score[i] >= minimum_respect and extension_score[i] < chase_extension_threshold
                   and exhaustion_score[i] < exhaustion_threshold and not bull_accel and not bear_accel)
        if compression_state:
            state_code[i] = 1
        elif exh_state and td > 0:
            state_code[i] = 2
        elif exh_state and td < 0:
            state_code[i] = 3
        elif overext and td > 0:
            state_code[i] = 4
        elif overext and td < 0:
            state_code[i] = 5
        elif bull_accel:
            state_code[i] = 6
        elif bear_accel:
            state_code[i] = 7
        elif healthy and td > 0:
            state_code[i] = 8
        elif healthy and td < 0:
            state_code[i] = 9
        elif td > 0 and tq >= 35.0:
            state_code[i] = 10
        elif td < 0 and tq >= 35.0:
            state_code[i] = 11

    # events (phase 2 loop)
    prev_td = np.concatenate([[np.nan], trend_dir[:-1]])
    prev_tq = np.concatenate([[np.nan], trend_quality[:-1]])
    prior_bull = (prev_td > 0) & (prev_tq >= minimum_trend)
    prior_bear = (prev_td < 0) & (prev_tq >= minimum_trend)
    sd_prev = np.concatenate([[np.nan], signed_dist[:-1]])
    prior_ext_len = max(2, acceleration_length + 1)
    recent_bull_ext = pd.Series(sd_prev).rolling(prior_ext_len, min_periods=prior_ext_len).max().to_numpy()
    recent_bear_ext = -pd.Series(sd_prev).rolling(prior_ext_len, min_periods=prior_ext_len).min().to_numpy()
    bull_rej = (l <= ma + atr_a * proximity_atr) & (c > ma) & (c > o) & (c >= (h + l) * 0.5)
    bear_rej = (h >= ma - atr_a * proximity_atr) & (c < ma) & (c < o) & (c <= (h + l) * 0.5)
    bull_pull = prior_bull & (trend_dir > 0) & (trend_quality >= minimum_trend) & (respect_score >= minimum_respect) & (recent_bull_ext >= pullback_prior_extension) & bull_rej & (exhaustion_score <= maximum_pullback_exhaustion)
    bear_pull = prior_bear & (trend_dir < 0) & (trend_quality >= minimum_trend) & (respect_score >= minimum_respect) & (recent_bear_ext >= pullback_prior_extension) & bear_rej & (exhaustion_score <= maximum_pullback_exhaustion)
    comp_was = pd.Series(np.concatenate([[np.nan], compression_score[:-1]])).rolling(compression_length, min_periods=compression_length).max().to_numpy() >= compression_threshold
    bull_mom = comp_was & crossover(pd.Series(signed_dist), pd.Series(np.full(n, compression_band_atr))).to_numpy() & (ma_slope_atr > 0) & (acceleration_score >= acceleration_threshold) & (price_eff >= 0.35) & (exhaustion_score < exhaustion_threshold)
    bear_mom = comp_was & crossunder(pd.Series(signed_dist), pd.Series(np.full(n, -compression_band_atr))).to_numpy() & (ma_slope_atr < 0) & (acceleration_score >= acceleration_threshold) & (price_eff >= 0.35) & (exhaustion_score < exhaustion_threshold)
    bull_chase = (trend_dir > 0) & (trend_quality >= minimum_trend) & crossover(pd.Series(extension_score), pd.Series(np.full(n, chase_extension_threshold))).to_numpy()
    bear_chase = (trend_dir < 0) & (trend_quality >= minimum_trend) & crossover(pd.Series(extension_score), pd.Series(np.full(n, chase_extension_threshold))).to_numpy()
    bull_exh = (trend_dir > 0) & crossover(pd.Series(exhaustion_score), pd.Series(np.full(n, exhaustion_threshold))).to_numpy()
    bear_exh = (trend_dir < 0) & crossover(pd.Series(exhaustion_score), pd.Series(np.full(n, exhaustion_threshold))).to_numpy()
    recent_bull_acc = sma(pd.Series((c > ma).astype(float)), break_confirmation).to_numpy()
    recent_bear_acc = sma(pd.Series((c < ma).astype(float)), break_confirmation).to_numpy()
    td_lag = np.array([trend_dir[i - break_confirmation] if i >= break_confirmation else np.nan for i in range(n)])
    tq_lag = np.array([trend_quality[i - break_confirmation] if i >= break_confirmation else np.nan for i in range(n)])
    bull_break_raw = (td_lag > 0) & (tq_lag >= minimum_trend) & (recent_bear_acc >= 0.80) & (relationship < 10.0) & (ma_slope_atr <= full_slope_atr * 0.10)
    bear_break_raw = (td_lag < 0) & (tq_lag >= minimum_trend) & (recent_bull_acc >= 0.80) & (relationship > -10.0) & (ma_slope_atr >= -full_slope_atr * 0.10)
    bull_break = bull_break_raw & ~np.concatenate([[False], bull_break_raw[:-1]])
    bear_break = bear_break_raw & ~np.concatenate([[False], bear_break_raw[:-1]])

    bull_div = np.zeros(n, dtype=bool); bear_div = np.zeros(n, dtype=bool)
    piv_lo = pivotlow_strict(frame["low"], divergence_pivot_left, divergence_pivot_right).to_numpy()
    piv_hi = pivothigh_strict(frame["high"], divergence_pivot_left, divergence_pivot_right).to_numpy()
    last_pl_price = last_pl_rel = float("nan")
    last_ph_price = last_ph_rel = float("nan")
    rel_lag = np.array([relationship[i - divergence_pivot_right] if i >= divergence_pivot_right else np.nan for i in range(n)])
    for i in range(n):
        if not math.isnan(piv_lo[i]):
            if not math.isnan(last_pl_price) and piv_lo[i] < last_pl_price and rel_lag[i] > last_pl_rel:
                bull_div[i] = True
            last_pl_price = piv_lo[i]; last_pl_rel = rel_lag[i]
        if not math.isnan(piv_hi[i]):
            if not math.isnan(last_ph_price) and piv_hi[i] > last_ph_price and rel_lag[i] < last_ph_rel:
                bear_div[i] = True
            last_ph_price = piv_hi[i]; last_ph_rel = rel_lag[i]

    rel_low = pd.Series(relationship).rolling(turn_lookback, min_periods=turn_lookback).min().to_numpy()
    rel_high = pd.Series(relationship).rolling(turn_lookback, min_periods=turn_lookback).max().to_numpy()
    bull_turn_raw = (rel_low <= -turn_extreme_level) & rising(pd.Series(relationship), turn_confirm_bars).to_numpy()
    bear_turn_raw = (rel_high >= turn_extreme_level) & falling(pd.Series(relationship), turn_confirm_bars).to_numpy()
    bull_turn = bull_turn_raw & ~np.concatenate([[False], bull_turn_raw[:-1]])
    bear_turn = bear_turn_raw & ~np.concatenate([[False], bear_turn_raw[:-1]])

    # cooldown application
    families = {
        "pullback": [(bull_pull, 1), (bear_pull, -1)],
        "momentum": [(bull_mom, 1), (bear_mom, -1)],
        "chase": [(bull_chase, 1), (bear_chase, -1)],
        "exhaustion": [(bull_exh, 1), (bear_exh, -1)],
        "break": [(bull_break, 1), (bear_break, -1)],
        "divergence": [(bull_div, 1), (bear_div, -1)],
        "turn": [(bull_turn, 1), (bear_turn, -1)],
    }
    ev = {f"{fam}_{d}": np.zeros(n, dtype=np.int64) for fam in families for d in ("bull", "bear")}
    for fam, items in families.items():
        last_bar = None
        for i in range(n):
            for arr, sign in items:
                if arr[i] and (last_bar is None or i - last_bar - 1 > signal_cooldown):
                    ev[f"{fam}_{'bull' if sign > 0 else 'bear'}"][i] = 1
                    last_bar = i

    df = pd.DataFrame(index=idx)
    df["mare_ma"] = ma
    df["mare_relationship"] = relationship
    df["mare_relationship_signal"] = relationship_signal
    df["mare_trend_quality"] = trend_quality
    df["mare_trend_direction"] = pd.Series(trend_dir, index=idx).astype("Int64")
    df["mare_respect"] = respect_score
    df["mare_extension"] = extension_score
    df["mare_compression"] = compression_score
    df["mare_acceleration"] = acceleration_score
    df["mare_exhaustion"] = exhaustion_score
    df["mare_state"] = pd.Series(state_code, index=idx).astype("Int64")
    for key, arr in ev.items():
        df[f"mare_{key}"] = pd.Series(arr, index=idx).astype("Int64")
    return df
