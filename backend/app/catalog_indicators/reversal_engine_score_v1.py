"""Reversal Engine Score v1 — gated liquidity-sweep reversal signals.

Source: pine-collection/indicators/momentum/reversal_engine_score/reversal_engine_score_v1.pine
(German-language script; no explicit license header)

A long setup sweeps the prior `swing_len`-bar low and closes back above it
(liquidity sweep), then passes optional gates: close in the upper ⅓ of the
bar, body ≥ `min_body_atr`×ATR, ATR above `atr_low_factor`× its SMA base, a
not-too-deep sweep (`spike_atr_mult`), no HTF bear trend (EMA fast/slow on
the `htf_minutes` timeframe — previous completed period, per repo
convention), WaveTrend confluence on 1h/4h and a session window. Quality
scores 0-2 (RSI > 50, no lower-low pivot structure) tier into A/B/C and gate
against `min_quality_long/short`. A shared cooldown suppresses back-to-back
signals; stops sit beyond the sweep extreme plus an ATR buffer, targets at
`tp_r_mult` R. HH/HL/LH/LL pivot structure and outcome-window stats round out
the context.

Dropped: labels, trend background, level drawings and the 15m-only gate
(pass `base_minutes` explicitly if you need it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (atr, boolean, choice, crossover, edge, ema, flags, highest, integer,
                    lowest, number, ohlcv, pivothigh, pivotlow, rsi, sma)
from ._tf import base_step_minutes, htf_frame, lift_series


def reversal_engine_score_v1(
        frame: pd.DataFrame, *, min_quality_long: int = 2, min_quality_short: int = 1,
        rsi_len: int = 14, use_gate_a: bool = True, swing_len: int = 10,
        use_gate_b: bool = False, use_gate_c: bool = False, min_body_atr: float = 0.25,
        use_gate_d: bool = True, atr_len: int = 14, atr_base_len: int = 50,
        atr_low_factor: float = 0.60, use_gate_e: bool = False,
        spike_atr_mult: float = 0.80, use_gate_f: bool = True, htf_minutes: int = 60,
        htf_fast_len: int = 50, htf_slow_len: int = 200, use_gate_g: bool = False,
        wt_channel_len: int = 10, wt_average_len: int = 21, gate_g_os: float = -30.0,
        gate_g_ob: float = 30.0, use_gate_h: bool = False, gate_h_os: float = -30.0,
        gate_h_ob: float = 30.0, use_gate_i: bool = False, session: str = "0700-1200",
        ema_len: int = 21, smoothing: str = "EMA", jma_phase: int = 50,
        jma_power: float = 2.0, sl_atr_buffer: float = 0.30, tp_r_mult: float = 1.50,
        signal_cooldown: int = 5, pivot_len: int = 5, trade_dir: str = "Both",
        outcome_window: int = 20) -> pd.DataFrame:
    integer("min_quality_long", min_quality_long, 0, 2)
    integer("min_quality_short", min_quality_short, 0, 2)
    integer("rsi_len", rsi_len, 1)
    integer("swing_len", swing_len, 2)
    number("min_body_atr", min_body_atr, 0.0)
    integer("atr_len", atr_len, 1)
    integer("atr_base_len", atr_base_len, 5)
    number("atr_low_factor", atr_low_factor, 0.0)
    number("spike_atr_mult", spike_atr_mult, 0.0)
    integer("htf_minutes", htf_minutes, 1)
    integer("htf_fast_len", htf_fast_len, 1)
    integer("htf_slow_len", htf_slow_len, 1)
    number("gate_g_os", gate_g_os, -100.0, 100.0)
    number("gate_g_ob", gate_g_ob, -100.0, 100.0)
    number("sl_atr_buffer", sl_atr_buffer, 0.0)
    number("tp_r_mult", tp_r_mult, 0.0)
    integer("signal_cooldown", signal_cooldown, 0)
    integer("pivot_len", pivot_len, 2, 20)
    choice("trade_dir", trade_dir, ("Both", "Long Only", "Short Only"))
    integer("outcome_window", outcome_window, 5, 500)
    choice("smoothing", smoothing, ("EMA", "RMA", "SMA", "WMA", "HMA", "JMA"))
    if use_gate_i:
        parts = session.split("-")
        if len(parts) != 2 or len(parts[0]) != 4 or len(parts[1]) != 4:
            raise ValueError("参数 session 需要形如 0700-1200")
    high, low, close = ohlcv(frame, "high", "low", "close")
    open_ = frame["open"].astype(float)
    rows = len(frame)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)

    from .commodity_flowtrend import _smooth as flow_smooth

    def smooth(series: pd.Series, length: int) -> pd.Series:
        mapping = {"EMA": "EMA", "RMA": "RMA", "SMA": "SMA", "WMA": "WMA", "HMA": "HMA",
                   "JMA": "JMA Approx"}
        return flow_smooth(series, length, mapping[smoothing], jma_phase, jma_power)

    ema_reclaim = smooth(close, ema_len)
    rsi_value = rsi(close, rsi_len)
    atr_value = atr(frame, atr_len)
    atr_base = sma(atr_value, atr_base_len)

    base = base_step_minutes(frame)
    if htf_minutes % base or htf_minutes < base:
        raise ValueError(f"高周期 {htf_minutes} 分钟必须是基础周期 {base} 分钟的整数倍")
    htf = htf_frame(frame, htf_minutes)
    htf_ema_fast = lift_series(frame, ema(htf["close"], htf_fast_len), htf_minutes)
    htf_ema_slow = lift_series(frame, ema(htf["close"], htf_slow_len), htf_minutes)
    bull_trend = flags(htf_ema_fast > htf_ema_slow)
    bear_trend = flags(htf_ema_fast < htf_ema_slow)

    def wt1_of(series: pd.Series) -> pd.Series:
        esa = ema(series, wt_channel_len)
        de = ema((series - esa).abs(), wt_channel_len)
        ci = (series - esa) / (0.015 * de.replace(0, np.nan))
        return ema(ci, wt_average_len)

    ap = (high + low + close) / 3
    wt1_1h = lift_series(frame, wt1_of(htf_frame(frame, 60)["close"]), 60) \
        if 60 % base == 0 and 60 >= base else pd.Series(np.nan, index=frame.index)
    wt1_4h = lift_series(frame, wt1_of(htf_frame(frame, 240)["close"]), 240) \
        if 240 % base == 0 and 240 >= base else pd.Series(np.nan, index=frame.index)

    prev_swing_low = lowest(low.shift(1), swing_len)
    prev_swing_high = highest(high.shift(1), swing_len)
    range_value = high - low
    body_value = (close - open_).abs()
    close_in_upper_third = flags((range_value > 0)
                                 & (close > low + range_value * 0.65))
    close_in_lower_third = flags((range_value > 0)
                                 & (close < high - range_value * 0.65))
    liquidity_sweep_long = flags((low < prev_swing_low) & (close > prev_swing_low))
    liquidity_sweep_short = flags((high > prev_swing_high) & (close < prev_swing_high))

    atr_ok = flags(atr_value > atr_base * atr_low_factor)
    body_ok = flags(body_value > atr_value * min_body_atr)
    sweep_depth_long = prev_swing_low - low
    sweep_depth_short = high - prev_swing_high
    sweep_not_too_deep_long = flags(~use_gate_e | (sweep_depth_long <= atr_value * spike_atr_mult))
    sweep_not_too_deep_short = flags(~use_gate_e | (sweep_depth_short <= atr_value * spike_atr_mult))

    if use_gate_i:
        utc = frame.index.tz_convert("UTC")
        start_minutes = int(session[:2]) * 60 + int(session[2:4])
        end_minutes = int(session[5:7]) * 60 + int(session[7:9])
        minutes_of_day = utc.hour * 60 + utc.minute
        if start_minutes <= end_minutes:
            mask = (minutes_of_day >= start_minutes) & (minutes_of_day < end_minutes)
        else:
            mask = (minutes_of_day >= start_minutes) | (minutes_of_day < end_minutes)
        session_ok = flags(pd.Series(mask, index=frame.index))
    else:
        session_ok = flags(pd.Series(True, index=frame.index))

    # HH/HL/LH/LL pivot structure.
    ph_values = pivothigh(high, pivot_len, pivot_len).to_numpy(dtype=float)
    pl_values = pivotlow(low, pivot_len, pivot_len).to_numpy(dtype=float)
    last_sh1 = last_sh2 = last_sl1 = last_sl2 = np.nan
    hh_detected = np.zeros(rows, dtype=bool)
    lh_detected = np.zeros(rows, dtype=bool)
    hl_detected = np.zeros(rows, dtype=bool)
    ll_detected = np.zeros(rows, dtype=bool)
    for position in range(rows):
        if not np.isnan(ph_values[position]):
            last_sh2 = last_sh1
            last_sh1 = highs[position - pivot_len]
        if not np.isnan(pl_values[position]):
            last_sl2 = last_sl1
            last_sl1 = lows[position - pivot_len]
        if not np.isnan(last_sh1) and not np.isnan(last_sh2):
            hh_detected[position] = last_sh1 > last_sh2
            lh_detected[position] = last_sh1 < last_sh2
        if not np.isnan(last_sl1) and not np.isnan(last_sl2):
            hl_detected[position] = last_sl1 > last_sl2
            ll_detected[position] = last_sl1 < last_sl2
    hh_flags = flags(pd.Series(hh_detected, index=frame.index))
    ll_flags = flags(pd.Series(ll_detected, index=frame.index))

    if use_gate_g:
        gate_g_long = flags(wt1_1h <= gate_g_os)
        gate_g_short = flags(wt1_1h >= gate_g_ob)
    else:
        gate_g_long = gate_g_short = flags(pd.Series(True, index=frame.index))
    if use_gate_h:
        gate_h_long = flags(wt1_4h <= gate_h_os)
        gate_h_short = flags(wt1_4h >= gate_h_ob)
    else:
        gate_h_long = gate_h_short = flags(pd.Series(True, index=frame.index))

    long_gate = flags((~use_gate_a | liquidity_sweep_long)
                      & (~use_gate_b | close_in_upper_third)
                      & (~use_gate_c | body_ok)
                      & (~use_gate_d | atr_ok)
                      & sweep_not_too_deep_long
                      & (~use_gate_f | ~bear_trend)
                      & gate_g_long & gate_h_long & session_ok)
    short_gate = flags((~use_gate_a | liquidity_sweep_short)
                       & (~use_gate_b | close_in_lower_third)
                       & (~use_gate_c | body_ok)
                       & (~use_gate_d | atr_ok)
                       & sweep_not_too_deep_short
                       & (~use_gate_f | ~bull_trend)
                       & gate_g_short & gate_h_short & session_ok)

    long_quality = (rsi_value > 50).astype(float) + (~ll_flags).astype(float)
    short_quality = (rsi_value > 50).astype(float) + (~hh_flags).astype(float)

    rows_index = frame.index
    min_long = float(min_quality_long)
    min_short = float(min_quality_short)
    long_signal_values = np.zeros(rows, dtype=bool)
    short_signal_values = np.zeros(rows, dtype=bool)
    long_gate_values = long_gate.to_numpy(dtype=bool)
    short_gate_values = short_gate.to_numpy(dtype=bool)
    long_quality_values = long_quality.to_numpy(dtype=float)
    short_quality_values = short_quality.to_numpy(dtype=float)
    last_signal = None
    for position in range(rows):
        cooldown_ok = last_signal is None or position - last_signal > signal_cooldown
        raw_long = bool(long_gate_values[position]
                        and long_quality_values[position] >= min_long
                        and trade_dir != "Short Only")
        raw_short = bool(short_gate_values[position]
                         and short_quality_values[position] >= min_short
                         and trade_dir != "Long Only")
        fired = bool((raw_long or raw_short) and cooldown_ok)
        if fired:
            last_signal = position
        long_signal_values[position] = bool(raw_long and cooldown_ok)
        short_signal_values[position] = bool(raw_short and cooldown_ok)

    long_signal = flags(pd.Series(long_signal_values, index=rows_index))
    short_signal = flags(pd.Series(short_signal_values, index=rows_index))
    any_signal = flags(pd.Series(long_signal_values | short_signal_values, index=rows_index))

    long_stop = pd.Series(np.where(long_signal,
                                   np.minimum(lows, prev_swing_low.to_numpy(dtype=float))
                                   - atr_value * sl_atr_buffer, np.nan), index=rows_index)
    long_target = pd.Series(np.where(long_signal,
                                     closes + (closes - (np.minimum(lows, prev_swing_low.to_numpy(dtype=float))
                                                       - atr_value * sl_atr_buffer)) * tp_r_mult,
                                     np.nan), index=rows_index)
    short_stop = pd.Series(np.where(short_signal,
                                    np.maximum(highs, prev_swing_high.to_numpy(dtype=float))
                                    + atr_value * sl_atr_buffer, np.nan), index=rows_index)
    short_target = pd.Series(np.where(short_signal,
                                      closes - ((np.maximum(highs, prev_swing_high.to_numpy(dtype=float))
                                                 + atr_value * sl_atr_buffer) - closes) * tp_r_mult,
                                      np.nan), index=rows_index)

    def tier(quality: pd.Series) -> pd.Series:
        return pd.Series(np.where(quality >= 2, "A", np.where(quality >= 1, "B", "C")),
                         index=rows_index)

    return pd.DataFrame({
        "long_signal": long_signal,
        "short_signal": short_signal,
        "any_signal": any_signal,
        "long_quality": long_quality.astype("Int64"),
        "short_quality": short_quality.astype("Int64"),
        "long_tier": tier(long_quality),
        "short_tier": tier(short_quality),
        "liquidity_sweep_long": liquidity_sweep_long,
        "liquidity_sweep_short": liquidity_sweep_short,
        "long_gate_ok": long_gate,
        "short_gate_ok": short_gate,
        "atr_ok": atr_ok,
        "bull_trend": bull_trend,
        "bear_trend": bear_trend,
        "ema_reclaim": ema_reclaim,
        "reclaim_above": flags(close > ema_reclaim),
        "hh_detected": hh_flags,
        "lh_detected": flags(pd.Series(lh_detected, index=rows_index)),
        "hl_detected": flags(pd.Series(hl_detected, index=rows_index)),
        "ll_detected": ll_flags,
        "long_stop": long_stop,
        "long_target": long_target,
        "short_stop": short_stop,
        "short_target": short_target,
    }, index=rows_index)
