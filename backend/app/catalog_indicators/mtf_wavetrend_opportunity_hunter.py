"""MTF WaveTrend Opportunity Hunter [WavesUnchained]  v2.7.0  —  catalog port.

Source: data/pine-collection/indicators/momentum/
        mtf_wavetrend_opportunity_hunter/mtf_wavetrend_opportunity_hunter.pine

A WaveTrend confluence engine: Regime / Opportunity / Timing / Quality scores
are blended into a long/short final score, and stop-and-reverse entries fire on
zone-filtered WT crosses (with optional turn/exhaustion hook exits and a
non-blocking tradeable flag).

Ported core: WaveTrend (EMA / JMA blend / Ultimate Smoother), efficiency ratio,
ADX, Shannon sign-entropy + percentile rank, the four score families, net
confluence, the entry/hook state machine and tradeable flags.

Not ported: multi-timeframe request.security (Regime/Setup/Trigger/Context TFs
are approximated by the chart-timeframe WaveTrend), confluence ribbons, rotation
map, dashboard/stats, drawing and alerts.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import (
    atr, barssince, boolean, choice, crossover, crossunder, dmi, ema, highest,
    integer, linreg, lowest, number, ohlcv, percentrank, sma, stdev, wma,
)


def _jma_series(src, length, phase, power):
    n = len(src)
    pr = 0.5 if phase < -100 else 2.5 if phase > 100 else phase / 100.0 + 1.5
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = beta ** power
    out = np.empty(n)
    e0 = e1 = e2 = jma = np.nan
    for i in range(n):
        s = src[i]
        e0_prev = e0 if not math.isnan(e0) else s
        e0 = (1 - alpha) * s + alpha * e0_prev
        e1_prev = e1 if not math.isnan(e1) else 0.0
        e1 = (s - e0) * (1 - beta) + beta * e1_prev
        jma_prev = jma if not math.isnan(jma) else s
        e2_prev = e2 if not math.isnan(e2) else 0.0
        e2 = (e0 + pr * e1 - jma_prev) * (1 - alpha) ** 2 + alpha ** 2 * e2_prev
        jma = jma_prev + e2
        out[i] = jma
    return out


def _ultimate_series(src, period):
    n = len(src)
    f = 1.414 * math.pi / period
    a1 = math.exp(-f)
    c2 = 2 * a1 * math.cos(f)
    c3 = -a1 * a1
    c1 = (1 - c2 - c3) / 4
    out = np.empty(n)
    us1 = us2 = np.nan
    for i in range(n):
        s0 = src[i]
        s1 = src[i - 1] if i >= 1 else s0
        s2 = src[i - 2] if i >= 2 else s0
        us1_prev = us1 if not math.isnan(us1) else s0
        us2_prev = us2 if not math.isnan(us2) else s0
        us = (1 - c1) * s0 + (2 * c1 - c2) * s1 - (c1 + c3) * s2 + c2 * us1_prev + c3 * us2_prev
        out[i] = us
        us2 = us1
        us1 = us
    return out


def mtf_wavetrend_opportunity_hunter(
    frame: pd.DataFrame,
    *,
    wt_ch_len: int = 10,
    wt_avg_len: int = 21,
    wt_sig_len: int = 4,
    wt_extreme: float = 55.0,
    use_jma_core: bool = True,
    jma_core_blend: float = 0.35,
    use_ultimate: bool = False,
    ema_fast_len: int = 50,
    ema_slow_len: int = 200,
    adx_len: int = 14,
    adx_min: float = 18.0,
    jma_len: int = 34,
    jma_phase: int = 50,
    jma_power: float = 2.0,
    er_len: int = 20,
    atr_len: int = 14,
    entropy_len: int = 40,
    entropy_rank_len: int = 200,
    min_regime: float = 65.0,
    min_opportunity: float = 35.0,
    min_opp_signal: float = 40.0,
    gate_buffer: float = 10.0,
    cooldown_bars: int = 8,
    wt_cross_zone: float = 0.0,
    use_hook_exit: bool = False,
    trigger_turn_level: float = 35.0,
    turn_fall_bars: int = 2,
    use_exhaustion: bool = True,
    exhaust_level: float = 50.0,
    allow_long: bool = True,
    allow_short: bool = True,
) -> pd.DataFrame:
    wt_ch_len = integer("wt_ch_len", wt_ch_len, 1)
    wt_avg_len = integer("wt_avg_len", wt_avg_len, 1)
    wt_sig_len = integer("wt_sig_len", wt_sig_len, 1)
    ema_fast_len = integer("ema_fast_len", ema_fast_len, 1)
    ema_slow_len = integer("ema_slow_len", ema_slow_len, 1)
    adx_len = integer("adx_len", adx_len, 1)
    jma_len = integer("jma_len", jma_len, 1)
    er_len = integer("er_len", er_len, 2)
    atr_len = integer("atr_len", atr_len, 1)
    entropy_len = integer("entropy_len", entropy_len, 10)
    entropy_rank_len = integer("entropy_rank_len", entropy_rank_len, 50)
    cooldown_bars = integer("cooldown_bars", cooldown_bars, 1, 30)
    turn_fall_bars = integer("turn_fall_bars", turn_fall_bars, 1, 10)
    wt_extreme = number("wt_extreme", wt_extreme, 10.0)
    jma_core_blend = number("jma_core_blend", jma_core_blend, 0.0, 1.0)
    adx_min = number("adx_min", adx_min, 1.0)
    min_regime = number("min_regime", min_regime, 0.0, 100.0)
    min_opportunity = number("min_opportunity", min_opportunity, 0.0, 100.0)
    min_opp_signal = number("min_opp_signal", min_opp_signal, 0.0, 100.0)
    gate_buffer = number("gate_buffer", gate_buffer, 0.0, 30.0)
    wt_cross_zone = number("wt_cross_zone", wt_cross_zone, -40.0, 40.0)
    trigger_turn_level = number("trigger_turn_level", trigger_turn_level, 5.0, 100.0)
    exhaust_level = number("exhaust_level", exhaust_level, 20.0, 80.0)

    o, h, l, c = ohlcv(frame, "open", "high", "low", "close")
    o = o.to_numpy(dtype=float); h = h.to_numpy(dtype=float)
    l = l.to_numpy(dtype=float); c = c.to_numpy(dtype=float)
    idx = frame.index
    n = len(frame)
    hlc3 = (h + l + c) / 3.0

    def smooth_core(src, length):
        e = ema(pd.Series(src), length).to_numpy()
        if use_ultimate:
            return _ultimate_series(src, length)
        if use_jma_core:
            j = _jma_series(e, length, jma_phase, jma_power)
            return e * (1.0 - jma_core_blend) + j * jma_core_blend
        return e

    esa = smooth_core(hlc3, wt_ch_len)
    de_raw = smooth_core(np.abs(hlc3 - esa), wt_ch_len)
    de = np.maximum(de_raw, 0.0)
    ci = np.where(de > 0, (hlc3 - esa) / np.where(de == 0, np.nan, 0.015 * de), 0.0)
    wt1 = smooth_core(ci, wt_avg_len)
    wt2 = sma(pd.Series(wt1), wt_sig_len).to_numpy()
    hist = wt1 - wt2

    # regime
    ema_fast = ema(frame["close"], ema_fast_len).to_numpy()
    ema_slow = ema(frame["close"], ema_slow_len).to_numpy()
    _, _, adx = dmi(frame, adx_len)
    adx = adx.to_numpy(dtype=float)
    trend_bull = (c > ema_fast) & (ema_fast > ema_slow)
    trend_bear = (c < ema_fast) & (ema_fast < ema_slow)
    adx_ok = adx >= adx_min
    wt1_prev = np.roll(wt1, 1); wt2_prev = np.roll(wt2, 1); hist_prev = np.roll(hist, 1)
    wt1_prev[0] = np.nan; wt2_prev[0] = np.nan; hist_prev[0] = np.nan

    bull_regime = (np.where(wt1 > wt2, 25, 0) + np.where(wt1 > wt1_prev, 20, 0)
                   + np.where(trend_bull, 30, 0) + np.where(adx_ok, 15, 0)
                   + np.where(hist > hist_prev, 10, 0)).astype(float)
    bear_regime = (np.where(wt1 < wt2, 25, 0) + np.where(wt1 < wt1_prev, 20, 0)
                   + np.where(trend_bear, 30, 0) + np.where(adx_ok, 15, 0)
                   + np.where(hist < hist_prev, 10, 0)).astype(float)

    # opportunity (chart JMA distance)
    atr_a = np.maximum(atr(frame, atr_len).to_numpy(dtype=float), 1e-8)
    jma_close = _jma_series(c, jma_len, jma_phase, jma_power)
    dist_atr = (c - jma_close) / atr_a
    long_pullback = np.clip((-dist_atr + 0.5) / 2.0 * 100.0, 0, 100)
    short_pullback = np.clip((dist_atr + 0.5) / 2.0 * 100.0, 0, 100)
    long_wt_ext = np.clip((-wt1 / wt_extreme) * 100.0, 0, 100)
    short_wt_ext = np.clip((wt1 / wt_extreme) * 100.0, 0, 100)
    long_opp = long_pullback * 0.45 + long_wt_ext * 0.35 + np.where(wt1 > wt1_prev, 20, 0)
    short_opp = short_pullback * 0.45 + short_wt_ext * 0.35 + np.where(wt1 < wt1_prev, 20, 0)

    # timing
    plot_wt1 = np.clip(wt1, -80, 80); plot_wt2 = np.clip(wt2, -80, 80)
    bull_cross = crossover(pd.Series(plot_wt1), pd.Series(plot_wt2)).to_numpy()
    bear_cross = crossunder(pd.Series(plot_wt1), pd.Series(plot_wt2)).to_numpy()
    def shift(a, k):
        out = np.full(n, np.nan); out[k:] = a[:-k]; return out
    bull_turn_raw = ((wt1 < -trigger_turn_level) & (wt1 > shift(wt1, 1))
                     & (shift(wt1, 1) < shift(wt1, 1 + turn_fall_bars)))
    bear_turn_raw = ((wt1 > trigger_turn_level) & (wt1 < shift(wt1, 1))
                     & (shift(wt1, 1) > shift(wt1, 1 + turn_fall_bars)))
    bull_turn = bull_turn_raw & ~shift(bull_turn_raw.astype(float), 1).astype(bool)
    bear_turn = bear_turn_raw & ~shift(bear_turn_raw.astype(float), 1).astype(bool)
    bear_exh_raw = use_exhaustion & (shift(wt1, 1) >= exhaust_level) & (wt1 < shift(wt1, 1))
    bull_exh_raw = use_exhaustion & (shift(wt1, 1) <= -exhaust_level) & (wt1 > shift(wt1, 1))
    bear_exh = bear_exh_raw & ~shift(bear_exh_raw.astype(float), 1).astype(bool)
    bull_exh = bull_exh_raw & ~shift(bull_exh_raw.astype(float), 1).astype(bool)
    raw_bull_cross = bull_cross & (wt2_prev < wt_cross_zone)
    raw_bear_cross = bear_cross & (wt2_prev > -wt_cross_zone)
    bull_entry_trig = raw_bull_cross
    bear_entry_trig = raw_bear_cross
    bull_hook = (bull_turn | bull_exh) if use_hook_exit else np.zeros(n, dtype=bool)
    bear_hook = (bear_turn | bear_exh) if use_hook_exit else np.zeros(n, dtype=bool)
    bull_trigger = raw_bull_cross | bull_turn | bull_exh
    bear_trigger = raw_bear_cross | bear_turn | bear_exh

    jma_slope = jma_close - np.concatenate([[np.nan], jma_close[:-1]])
    jma_prev_arr = np.concatenate([[np.nan], jma_close[:-1]])
    bull_momentum = (wt1 > wt1_prev) & (hist > hist_prev)
    bear_momentum = (wt1 < wt1_prev) & (hist < hist_prev)
    bull_structure = (c > jma_close) & (jma_close > jma_prev_arr)
    bear_structure = (c < jma_close) & (jma_close < jma_prev_arr)
    long_timing = 39 * bull_trigger.astype(float) + 33 * bull_momentum.astype(float) + 28 * bull_structure.astype(float)
    short_timing = 39 * bear_trigger.astype(float) + 33 * bear_momentum.astype(float) + 28 * bear_structure.astype(float)

    # quality
    jma_s5 = (linreg(pd.Series(jma_close), 5, 0) - linreg(pd.Series(jma_close), 5, 1)).to_numpy()
    jma_s10 = (linreg(pd.Series(jma_close), 10, 0) - linreg(pd.Series(jma_close), 10, 1)).to_numpy()
    accel_mag = jma_s5 - jma_s10
    accel_std = stdev(pd.Series(accel_mag), 50).to_numpy()
    accel_norm = np.where(accel_std > 0, accel_mag / np.where(accel_std == 0, np.nan, accel_std), 0.0)
    long_accel = np.clip(accel_norm, 0, 2) / 2.0 * 15.0
    short_accel = np.clip(-accel_norm, 0, 2) / 2.0 * 15.0
    vol_amt = np.abs(np.diff(c, prepend=c[0]))
    er_vol = sma(pd.Series(vol_amt), er_len).to_numpy() * er_len
    er_change = np.abs(c - np.concatenate([np.full(er_len, np.nan), c[:-er_len]])) if n > er_len else np.full(n, np.nan)
    er = np.where(er_vol != 0, er_change / np.where(er_vol == 0, np.nan, er_vol), 0.0)
    er_score_raw = np.clip(er * 100.0, 0, 100)
    long_er = np.where(jma_s5 > 0, er_score_raw, er_score_raw * 0.35)
    short_er = np.where(jma_s5 < 0, er_score_raw, er_score_raw * 0.35)
    atr_sma20 = sma(pd.Series(atr_a), 20).to_numpy()
    vol_ratio = np.where(atr_sma20 > 0, atr_a / np.where(atr_sma20 == 0, np.nan, atr_sma20), 1.0)
    vol_score = np.where(vol_ratio < 0.6, 25.0,
                         np.where(vol_ratio < 1.0, 25.0 + (vol_ratio - 0.6) / 0.4 * 55.0,
                                  np.where(vol_ratio <= 1.4, 80.0,
                                           np.where(vol_ratio <= 2.2, 80.0 - (vol_ratio - 1.4) / 0.8 * 45.0, 30.0))))
    up = (np.diff(c, prepend=c[0]) > 0).astype(float)
    ups = pd.Series(up).rolling(entropy_len, min_periods=entropy_len).sum().to_numpy()
    p = ups / entropy_len
    with np.errstate(invalid="ignore", divide="ignore"):
        entropy = np.where((p <= 0) | (p >= 1), 0.0,
                           -(p * np.log(np.where(p <= 0, np.nan, p)) + (1 - p) * np.log(np.where(p >= 1, np.nan, 1 - p))) / math.log(2))
    entropy = np.nan_to_num(entropy, nan=0.0)
    entropy_rank = percentrank(pd.Series(entropy), entropy_rank_len).to_numpy()
    entropy_score = 100.0 - entropy_rank
    long_quality = long_er * 0.30 + vol_score * 0.20 + long_accel + entropy_score * 0.35
    short_quality = short_er * 0.30 + vol_score * 0.20 + short_accel + entropy_score * 0.35

    long_final = bull_regime * 0.35 + long_opp * 0.30 + long_timing * 0.25 + long_quality * 0.10
    short_final = bear_regime * 0.35 + short_opp * 0.30 + short_timing * 0.25 + short_quality * 0.10
    net_final = long_final - short_final

    min_regime_eff = max(0.0, min(min_regime - gate_buffer, 100.0))
    min_opp_eff = max(0.0, min(min_opportunity - gate_buffer, 100.0))
    regime_long_dom = (bull_regime >= min_regime_eff) & (bull_regime > bear_regime)
    regime_short_dom = (bear_regime >= min_regime_eff) & (bear_regime > bull_regime)
    long_tradeable = long_opp >= min_opp_signal
    short_tradeable = short_opp >= min_opp_signal

    # position state machine
    in_long = in_short = False
    pos_bar = -999
    entry_long = np.zeros(n, dtype=np.int64)
    entry_short = np.zeros(n, dtype=np.int64)
    hook_exit_long = np.zeros(n, dtype=np.int64)
    hook_exit_short = np.zeros(n, dtype=np.int64)
    state_long = np.zeros(n, dtype=np.int64)
    state_short = np.zeros(n, dtype=np.int64)

    for i in range(n):
        held = (i - pos_bar) >= cooldown_bars
        flat = not in_long and not in_short
        le = allow_long and bool(bull_entry_trig[i]) and not in_long and (flat or held)
        se = allow_short and bool(bear_entry_trig[i]) and not in_short and (flat or held)
        if le and se:
            le = se = False
        lh = in_long and held and bool(bear_hook[i]) and not se
        sh = in_short and held and bool(bull_hook[i]) and not le
        if le:
            in_long = True; in_short = False; pos_bar = i; entry_long[i] = 1
        if se:
            in_long = False; in_short = True; pos_bar = i; entry_short[i] = 1
        if lh:
            in_long = False; hook_exit_long[i] = 1
        if sh:
            in_short = False; hook_exit_short[i] = 1
        state_long[i] = 1 if in_long else 0
        state_short[i] = 1 if in_short else 0

    df = pd.DataFrame(index=idx)
    df["mwoh_wt1"] = wt1
    df["mwoh_wt2"] = wt2
    df["mwoh_hist"] = hist
    df["mwoh_regime_long"] = bull_regime
    df["mwoh_regime_short"] = bear_regime
    df["mwoh_opp_long"] = long_opp
    df["mwoh_opp_short"] = short_opp
    df["mwoh_timing_long"] = long_timing
    df["mwoh_timing_short"] = short_timing
    df["mwoh_quality_long"] = long_quality
    df["mwoh_quality_short"] = short_quality
    df["mwoh_final_long"] = long_final
    df["mwoh_final_short"] = short_final
    df["mwoh_net_final"] = net_final
    df["mwoh_entry_long"] = pd.Series(entry_long, index=idx).astype("Int64")
    df["mwoh_entry_short"] = pd.Series(entry_short, index=idx).astype("Int64")
    df["mwoh_hook_exit_long"] = pd.Series(hook_exit_long, index=idx).astype("Int64")
    df["mwoh_hook_exit_short"] = pd.Series(hook_exit_short, index=idx).astype("Int64")
    df["mwoh_tradeable_long"] = pd.Series(long_tradeable.astype(int), index=idx).astype("Int64")
    df["mwoh_tradeable_short"] = pd.Series(short_tradeable.astype(int), index=idx).astype("Int64")
    df["mwoh_in_long"] = pd.Series(state_long, index=idx).astype("Int64")
    df["mwoh_in_short"] = pd.Series(state_short, index=idx).astype("Int64")
    return df
