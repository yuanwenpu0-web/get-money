"""Regime Transition Engine — seven-state market regime state machine.

Source: pine-collection/indicators/trend_strength/regime_transition_engine/regime_transition_engine.pine
(WavesUnchained, MIT)

Seven internal scores — noise, compression, expansion, trend-up, trend-down,
exhaustion and reversion — are built each bar from volatility percentile,
choppiness, Kaufman-style efficiency, a linreg slope in ATR units, return
z-scores, a 5-bar variance ratio and 1-lag autocorrelation. The engine holds
its current state for at least `dwell_bars` and only switches when another
state's score beats the active state's own score by `hysteresis` (the initial
state is noise). Each state carries a dedicated transition score measuring how
close it is to the next regime; a directional pressure sign flows from the
state and trend direction. The transition score is EMA-smoothed over 5 bars.

State codes: 0 noise, 1 compression, 2 expansion, 3 trend-up, 4 trend-down,
5 exhaustion, 6 reversion.

Dropped: state-track visuals, dashboard, internal-score plots and alerts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (atr, correlation, ema, flags, highest, integer, lowest, number,
                    ohlcv, percentile_nearest_rank, percentile_nearest_rank as _pnr, rma,
                    sma, stdev, true_range, variance, linreg, stdev as _stdev)

_STATE_NAMES = {0: "Noise", 1: "Compression", 2: "Expansion", 3: "Trend Up",
                4: "Trend Down", 5: "Exhaustion", 6: "Reversion"}


def regime_transition_engine(
        frame: pd.DataFrame, *, source: str = "close", len_fast: int = 20,
        len_mid: int = 50, len_slow: int = 100, atr_len: int = 14,
        pct_len: int = 120, dwell_bars: int = 4, hysteresis: float = 0.08) -> pd.DataFrame:
    integer("len_fast", len_fast, 8)
    integer("len_mid", len_mid, 20)
    integer("len_slow", len_slow, 40)
    integer("atr_len", atr_len, 5)
    integer("pct_len", pct_len, 40)
    integer("dwell_bars", dwell_bars, 1)
    number("hysteresis", hysteresis, 0.0, 1.0)
    high, low, close = ohlcv(frame, "high", "low", "close")
    src = close if source == "close" else (high + low + close) / 3

    def safe_div(numerator, denominator, fallback):
        return np.where((denominator != 0.0) & (~np.isnan(denominator)),
                        numerator / denominator, fallback)

    def score01(series):
        return series.clip(lower=0.0, upper=1.0)

    log_ret = np.log(src / src.shift(1))
    average_true_range = atr(frame, atr_len)
    atr_low = percentile_nearest_rank(average_true_range, pct_len, 20)
    atr_high = percentile_nearest_rank(average_true_range, pct_len, 80)
    vol_pct = pd.Series(
        safe_div(average_true_range - atr_low, atr_high - atr_low, 0.5),
        index=frame.index).clip(lower=0.0, upper=1.0)
    vol_slope = ema(vol_pct - vol_pct.shift(1), 5)

    direct = (src - src.shift(len_mid)).abs()
    path = sma(src.diff().abs(), len_mid) * len_mid
    efficiency = pd.Series(safe_div(direct, path, 0.0), index=frame.index)

    chop_range = highest(high, len_mid) - lowest(low, len_mid)
    tr_sum = sma(true_range(frame), len_mid) * len_mid
    chop = pd.Series(np.where(chop_range != 0.0,
                              100.0 * np.log10(tr_sum / chop_range) / np.log10(len_mid),
                              50.0), index=frame.index)
    chop_score = ((chop - 45.0) / 25.0).clip(lower=0.0, upper=1.0)

    basis = linreg(src, len_mid, 0)
    basis_prev = linreg(src, len_mid, 1)
    slope = pd.Series(safe_div(basis - basis_prev, average_true_range, 0.0),
                      index=frame.index)
    trend_dir = pd.Series(np.where(slope > 0, 1.0, np.where(slope < 0, -1.0, 0.0)),
                          index=frame.index)
    trend_power = (slope.abs() * 1.8 + efficiency * 1.2).clip(lower=0.0, upper=1.0)

    ret_mean = sma(log_ret, len_mid)
    ret_sd = stdev(log_ret, len_mid)
    ret_z = pd.Series(safe_div(log_ret - ret_mean, ret_sd, 0.0), index=frame.index)
    close_z = pd.Series(
        safe_div(src - sma(src, len_mid), stdev(src, len_mid), 0.0), index=frame.index)
    extension = (close_z.abs() / 2.7).clip(lower=0.0, upper=1.0)

    vr_k = 5
    rk = np.log(src / src.shift(vr_k))
    var1 = variance(log_ret, len_mid)
    var_k = variance(pd.Series(rk, index=frame.index), len_mid)
    vr = pd.Series(safe_div(var_k, vr_k * var1, 1.0), index=frame.index)
    vr_momentum = ((vr - 1.0) / 0.45).clip(lower=0.0, upper=1.0)
    vr_reversion = ((1.0 - vr) / 0.35).clip(lower=0.0, upper=1.0)

    ac = correlation(log_ret, log_ret.shift(1), len_mid).fillna(0.0)
    ac_momentum = (ac * 3.0).clip(lower=0.0, upper=1.0)
    ac_reversion = (-ac * 3.0).clip(lower=0.0, upper=1.0)

    compression = (score01(1.0 - vol_pct) * 0.45 + chop_score * 0.35
                   + score01(1.0 - efficiency * 3.0) * 0.20)
    expansion = (vol_pct * 0.40 + score01(vol_slope * 8.0) * 0.30
                 + (ret_z.abs() / 2.5).clip(lower=0.0, upper=1.0) * 0.30)
    trend_common = (trend_power * 0.45 + vr_momentum * 0.25 + ac_momentum * 0.15
                    + score01(1.0 - chop_score) * 0.15)
    trend_up = trend_common.where(trend_dir > 0, 0.0)
    trend_down = trend_common.where(trend_dir < 0, 0.0)
    exhaustion = (extension * 0.35 + score01(1.0 - efficiency * 2.5) * 0.25
                  + score01(-vol_slope * 6.0) * 0.20 + vr_reversion * 0.20)
    reversion = (vr_reversion * 0.35 + ac_reversion * 0.25 + chop_score * 0.20
                 + extension * 0.20)
    noise = (chop_score * 0.35
             + score01(1.0 - (vr - 1.0).abs() * 2.0) * 0.30
             + score01(1.0 - efficiency * 3.0) * 0.25
             + score01(1.0 - vol_pct) * 0.10)

    score_frame = pd.DataFrame({
        0: noise, 1: compression, 2: expansion, 3: trend_up,
        4: trend_down, 5: exhaustion, 6: reversion,
    })
    rows = len(frame)
    states = np.zeros(rows, dtype=int)
    state_ages = np.zeros(rows, dtype=int)
    state = 0
    state_age = 0
    noise_values = noise.to_numpy(dtype=float)
    score_values = {key: score_frame[key].to_numpy(dtype=float)
                    for key in score_frame.columns}
    for position in range(rows):
        state_age += 1
        active = score_values[state][position]
        best_state, best_score = 0, noise_values[position]
        for candidate in range(1, 7):
            if score_values[candidate][position] > best_score:
                best_score = score_values[candidate][position]
                best_state = candidate
        can_switch = state_age >= dwell_bars and best_score > active + hysteresis
        if can_switch:
            state = best_state
            state_age = 0
        states[position] = state
        state_ages[position] = state_age

    state_series = pd.Series(states, index=frame.index)
    active_score = pd.Series([score_values[states[i]][i] for i in range(rows)],
                             index=frame.index)

    transition_raw = best_score_frame = pd.Series(
        [max(score_values[c][i] for c in range(7)) for i in range(rows)],
        index=frame.index)
    transition_raw = transition_raw.copy()
    in_state = state_series
    to_expansion = (expansion * 0.45 + score01(vol_slope * 10.0) * 0.30
                    + (ret_z.abs() / 2.0).clip(lower=0.0, upper=1.0) * 0.25)
    to_trend = (pd.concat([trend_up, trend_down], axis=1).max(axis=1) * 0.55
                + vr_momentum * 0.25 + efficiency * 0.20)
    to_exhaustion = (exhaustion * 0.60 + extension * 0.25
                     + score01(-vol_slope * 6.0) * 0.15)
    to_reversion = reversion * 0.65 + vr_reversion * 0.20 + ac_reversion * 0.15
    transition_raw = transition_raw.mask(in_state == 1, to_expansion)
    transition_raw = transition_raw.mask(in_state == 2, to_trend)
    transition_raw = transition_raw.mask(in_state.isin([3, 4]), to_exhaustion)
    transition_raw = transition_raw.mask(in_state == 5, to_reversion)
    transition_score = ema(transition_raw * 100.0, 5)

    directional_pressure = pd.Series(0.0, index=frame.index)
    directional_pressure = directional_pressure.mask(in_state == 3, transition_score)
    directional_pressure = directional_pressure.mask(in_state == 4, -transition_score)
    directional_pressure = directional_pressure.mask(
        in_state.isin([5, 6]), -trend_dir * transition_score)
    directional_pressure = directional_pressure.mask(in_state == 2,
                                                     trend_dir * transition_score)

    labels = pd.Series([_STATE_NAMES[states[i]] for i in range(rows)],
                       index=frame.index, dtype=object)
    return pd.DataFrame({
        "state": state_series.astype("Int64"),
        "state_name": labels,
        "state_age": pd.Series(state_ages, index=frame.index).astype("Int64"),
        "state_changed": flags(state_series != state_series.shift(1)),
        "active_score": active_score,
        "trend_score": trend_score_column(trend_up, trend_down),
        "vol_percentile": vol_pct,
        "efficiency": efficiency,
        "choppiness": chop,
        "variance_ratio": vr,
        "compression_score": compression,
        "expansion_score": expansion,
        "trend_up_score": trend_up,
        "trend_down_score": trend_down,
        "exhaustion_score": exhaustion,
        "reversion_score": reversion,
        "noise_score": noise,
        "transition_score": transition_score,
        "directional_pressure": directional_pressure,
    }, index=frame.index)


def trend_score_column(trend_up: pd.Series, trend_down: pd.Series) -> pd.Series:
    return pd.concat([trend_up, trend_down], axis=1).max(axis=1)
