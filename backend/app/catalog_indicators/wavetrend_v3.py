"""WaveTrend v3 [WavesUnchained]  v3.2.6  —  catalog port (chart-timeframe core).

Source: data/pine-collection/indicators/momentum/wavetrend/wavetrend_v3.pine

Composite momentum system:
  1. WaveTrend oscillator (v2 formula) + signal SMA, OB/OS zones and fresh-zone
     reversal triggers.
  2. Market structure (HH/HL/LH/LL) from confirmed swing pivots, with the
     reversal-friendly long-ok / short-ok definitions.
  3. Volume context: POC (local + macro), directional volume spike, POC
     reclaim/reject events.
  4. Setup gates: range-edge or swing-proximity location, payoff room,
     impulse, pullback continuation, per-family volume gate.
  5. Composite trigger + context score → long/short signals + ATR stops/targets.

Ported core only. Multi-timeframe HTF/anchor WT require request.security and
are approximated by the chart WaveTrend; profile presets resolve the main
numeric gates. Exit virtual-position tracking, drawing and alerts are dropped.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import (
    atr, barssince, boolean, choice, crossover, crossunder, ema, highest, integer,
    lowest, number, ohlcv, pivothigh_strict, pivotlow_strict, sma,
)


def _compute_poc(i, lookback, bins, h, l, v):
    if i + 1 < lookback:
        return np.nan
    lo = l[i - lookback + 1:i + 1].min()
    hi = h[i - lookback + 1:i + 1].max()
    bin_size = (hi - lo) / bins
    if bin_size <= 0:
        return np.nan
    hist = np.zeros(bins)
    for j in range(lookback):
        idx = i - j
        mid = (h[idx] + l[idx]) / 2.0
        b = int(math.floor((mid - lo) / bin_size))
        b = max(0, min(bins - 1, b))
        hist[b] += v[idx] if not math.isnan(v[idx]) else 0.0
    max_idx = int(np.argmax(hist))
    return lo + (max_idx + 0.5) * bin_size


def wavetrend_v3(
    frame: pd.DataFrame,
    *,
    ch_length: int = 10,
    avg_length: int = 21,
    sig_length: int = 4,
    ci_const: float = 0.015,
    upper_band: int = 56,
    lower_band: int = -56,
    zone_exit_window: int = 4,
    use_auto_tf: bool = True,
    struct_piv_len: int = 5,
    struct_lookback: int = 50,
    vol_ma_len: int = 50,
    vol_spike_mult: float = 1.5,
    use_vol_poc: bool = True,
    poc_lookback: int = 100,
    poc_macro_lookback: int = 300,
    poc_bins: int = 20,
    min_context_score: float = 3.0,
    poc_recent_bars: int = 5,
    atr_len: int = 14,
    setup_range_len: int = 120,
    setup_edge_pct: float = 0.35,
    swing_proximity_atr: float = 1.5,
    use_setup_gate: bool = True,
    use_pullback_setup: bool = True,
    pullback_ema_len: int = 34,
    pullback_proximity_atr: float = 1.75,
    min_impulse_atr: float = 2.0,
    min_target_room_atr: float = 2.0,
    stop_buffer_atr: float = 0.5,
    fallback_stop_atr: float = 2.5,
    pullback_trigger_zone: float = 35.0,
    profile_mode: str = "Balanced",
    htf_weight: float = 1.0,
    anchor_weight: float = 1.0,
    anchor_score_mode: str = "Contrarian",
    mtf_gate_mode: str = "None",
    mtf_gate_apply_to: str = "Pullback Only",
    anchor_score_apply_to: str = "All",
    volume_gate_mode: str = "Lower TF Only",
    volatility_regime_mode: str = "Score Only",
    volatility_regime_min: float = 0.7,
    volatility_regime_max: float = 1.5,
    short_rr_require_confirm: bool = True,
    long_pb_require_confirm: bool = True,
) -> pd.DataFrame:
    ch_length = integer("ch_length", ch_length, 2)
    avg_length = integer("avg_length", avg_length, 2)
    sig_length = integer("sig_length", sig_length, 1)
    zone_exit_window = integer("zone_exit_window", zone_exit_window, 0, 20)
    struct_piv_len = integer("struct_piv_len", struct_piv_len, 2, 20)
    struct_lookback = integer("struct_lookback", struct_lookback, 20, 200)
    vol_ma_len = integer("vol_ma_len", vol_ma_len, 10, 200)
    poc_lookback = integer("poc_lookback", poc_lookback, 30, 500)
    poc_macro_lookback = integer("poc_macro_lookback", poc_macro_lookback, 50, 1000)
    poc_bins = integer("poc_bins", poc_bins, 10, 50)
    poc_recent_bars = integer("poc_recent_bars", poc_recent_bars, 1, 20)
    atr_len = integer("atr_len", atr_len, 1)
    setup_range_len = integer("setup_range_len", setup_range_len, 30, 500)
    pullback_ema_len = integer("pullback_ema_len", pullback_ema_len, 5, 200)
    min_context_score = number("min_context_score", min_context_score, 0.0, 7.0)
    ci_const = number("ci_const", ci_const, 0.001, 0.5)
    vol_spike_mult = number("vol_spike_mult", vol_spike_mult, 1.0, 5.0)
    setup_edge_pct = number("setup_edge_pct", setup_edge_pct, 0.10, 0.50)
    swing_proximity_atr = number("swing_proximity_atr", swing_proximity_atr, 0.25, 10.0)
    pullback_proximity_atr = number("pullback_proximity_atr", pullback_proximity_atr, 0.25, 5.0)
    min_impulse_atr = number("min_impulse_atr", min_impulse_atr, 1.0, 15.0)
    min_target_room_atr = number("min_target_room_atr", min_target_room_atr, 0.5, 10.0)
    stop_buffer_atr = number("stop_buffer_atr", stop_buffer_atr, 0.0, 5.0)
    fallback_stop_atr = number("fallback_stop_atr", fallback_stop_atr, 0.5, 10.0)
    pullback_trigger_zone = number("pullback_trigger_zone", pullback_trigger_zone, 10.0, 50.0)
    profile_mode = choice("profile_mode", profile_mode,
                          ("Strict", "Balanced", "Aggressive", "Reversal", "Pullback", "Diagnostic", "Custom"))
    anchor_score_mode = choice("anchor_score_mode", anchor_score_mode, ("Ignore", "Alignment", "Contrarian"))
    mtf_gate_mode = choice("mtf_gate_mode", mtf_gate_mode, ("None", "Setup Only", "Setup + Anchor"))
    mtf_gate_apply_to = choice("mtf_gate_apply_to", mtf_gate_apply_to, ("All", "Reversal Only", "Pullback Only", "None"))
    anchor_score_apply_to = choice("anchor_score_apply_to", anchor_score_apply_to, ("All", "Reversal Only", "Pullback Only", "None"))
    volume_gate_mode = choice("volume_gate_mode", volume_gate_mode, ("Off", "Lower TF Only", "All TF", "Reversal Only", "Pullback Only"))
    volatility_regime_mode = choice("volatility_regime_mode", volatility_regime_mode, ("Off", "Score Only"))

    h, l, c, vol = ohlcv(frame, "high", "low", "close", "volume")
    h = h.to_numpy(dtype=float); l = l.to_numpy(dtype=float); c = c.to_numpy(dtype=float)
    vol = vol.to_numpy(dtype=float)
    idx = frame.index
    n = len(frame)
    hlc3 = (h + l + c) / 3.0

    # chart seconds estimate
    if isinstance(idx, pd.DatetimeIndex) and n > 1:
        deltas = np.diff(idx.values).astype("timedelta64[s]").astype(float)
        chart_sec = float(np.median(deltas))
    else:
        chart_sec = 14400.0
    tf_scale = 14400.0 / max(chart_sec, 60.0) if use_auto_tf else 1.0

    def scale_bars(base, fl, cap):
        return int(min(max(round(base * tf_scale), fl), cap))

    # --- profile resolution ---
    eff_min = min_context_score
    eff_edge = setup_edge_pct
    eff_prox = swing_proximity_atr
    eff_zone = pullback_trigger_zone
    eff_upper, eff_lower = upper_band, lower_band
    eff_vol_mode = volume_gate_mode
    eff_mtf_gate = mtf_gate_mode
    eff_mtf_apply = mtf_gate_apply_to
    eff_anchor_apply = anchor_score_apply_to
    eff_short_confirm = short_rr_require_confirm
    eff_long_pb = long_pb_require_confirm
    eff_use_pb = use_pullback_setup
    eff_regime_mode = volatility_regime_mode
    if profile_mode == "Strict":
        eff_min, eff_edge, eff_prox, eff_vol_mode = 4.0, 0.30, 1.25, "All TF"
        eff_mtf_gate, eff_mtf_apply, eff_anchor_apply = "Setup Only", "All", "All"
        eff_zone, eff_upper, eff_lower = 25.0, 60, -60
    elif profile_mode == "Balanced":
        eff_min, eff_edge, eff_prox, eff_vol_mode = 3.0, 0.45, 2.5, "Lower TF Only"
        eff_mtf_gate, eff_mtf_apply, eff_anchor_apply = "None", "None", "All"
        eff_zone, eff_upper, eff_lower = 50.0, 45, -45
    elif profile_mode == "Aggressive":
        eff_min, eff_edge, eff_prox, eff_vol_mode = 2.0, 0.45, 2.5, "Off"
        eff_mtf_gate, eff_mtf_apply = "None", "None"
        eff_zone, eff_upper, eff_lower = 50.0, 45, -45
    elif profile_mode == "Reversal":
        eff_min, eff_edge, eff_prox, eff_vol_mode = 3.0, 0.40, 2.0, "Reversal Only"
        eff_mtf_apply, eff_anchor_apply, eff_use_pb = "None", "Reversal Only", False
        eff_zone, eff_upper, eff_lower = 35.0, 45, -45
    elif profile_mode == "Pullback":
        eff_min, eff_edge, eff_prox, eff_vol_mode = 2.5, 0.35, 1.5, "Pullback Only"
        eff_mtf_gate, eff_mtf_apply, eff_anchor_apply = "Setup Only", "Pullback Only", "Pullback Only"
        eff_zone, eff_upper, eff_lower = 35.0, 56, -56
    elif profile_mode == "Diagnostic":
        eff_min, eff_edge, eff_prox, eff_vol_mode = 0.0, 0.50, 5.0, "Off"
        eff_mtf_apply, eff_anchor_apply = "None", "None"
        eff_short_confirm = eff_long_pb = False
        eff_zone, eff_upper, eff_lower = 50.0, 20, -20
        eff_regime_mode = "Off"

    # --- WaveTrend ---
    esa = ema(pd.Series(hlc3), ch_length).to_numpy()
    d = ema(pd.Series(np.abs(hlc3 - esa)), ch_length).to_numpy()
    den = ci_const * d
    ci = np.where(den != 0, (hlc3 - esa) / np.where(den == 0, np.nan, den), 0.0)
    wt_osc = ema(pd.Series(ci), avg_length).to_numpy()
    wt_sig = sma(pd.Series(wt_osc), sig_length).to_numpy()

    os_zone = wt_osc < eff_lower
    ob_zone = wt_osc > eff_upper
    zone_exit_a = scale_bars(zone_exit_window, 1, 40)
    bars_since_os = barssince(pd.Series(os_zone)).to_numpy()
    bars_since_ob = barssince(pd.Series(ob_zone)).to_numpy()
    os_recent = np.nan_to_num(bars_since_os, nan=999) <= zone_exit_a
    ob_recent = np.nan_to_num(bars_since_ob, nan=999) <= zone_exit_a
    cross_up = crossover(pd.Series(wt_osc), pd.Series(wt_sig)).to_numpy() & os_recent & (wt_osc < 0)
    cross_down = crossunder(pd.Series(wt_osc), pd.Series(wt_sig)).to_numpy() & ob_recent & (wt_osc > 0)

    # --- structure ---
    ph = pivothigh_strict(frame["high"], struct_piv_len, struct_piv_len).to_numpy(dtype=float)
    pl = pivotlow_strict(frame["low"], struct_piv_len, struct_piv_len).to_numpy(dtype=float)
    struct_lookback_a = scale_bars(struct_lookback, 20, 500)
    setup_range_a = scale_bars(setup_range_len, 30, 500)
    vol_ma_a = scale_bars(vol_ma_len, 10, 300)
    pullback_ema_a = scale_bars(pullback_ema_len, 5, 300)
    poc_recent_a = scale_bars(poc_recent_bars, 1, 30)

    # --- volume / POC ---
    vol_ma = sma(pd.Series(vol), vol_ma_a).to_numpy()
    real = bool(np.any(vol > 0))
    span = h - l
    close_pos = np.where(span > 0, (c - l) / np.where(span == 0, np.nan, span), 0.5)
    vol_spike = real & (vol > vol_ma * vol_spike_mult) if real else np.zeros(n, dtype=bool)
    bull_spike = vol_spike & (close_pos > 0.65)
    bear_spike = vol_spike & (close_pos < 0.35)

    poc_lb_a = scale_bars(poc_lookback, 30, 500)
    pocm_lb_a = scale_bars(poc_macro_lookback, 50, 1000)
    poc_a = np.full(n, np.nan)
    pocm_a = np.full(n, np.nan)
    if use_vol_poc:
        for i in range(n):
            poc_a[i] = _compute_poc(i, poc_lb_a, poc_bins, h, l, vol)
            pocm_a[i] = _compute_poc(i, pocm_lb_a, poc_bins, h, l, vol)

    # --- ATR / setup ranges ---
    atr_a = atr(frame, atr_len).to_numpy(dtype=float)
    atr_sma_a = sma(pd.Series(atr_a), 100).to_numpy(dtype=float)
    setup_high = highest(frame["high"], setup_range_a).to_numpy(dtype=float)
    setup_low = lowest(frame["low"], setup_range_a).to_numpy(dtype=float)
    pullback_ema = ema(frame["close"], pullback_ema_a).to_numpy(dtype=float)

    out = {k: np.full(n, np.nan) for k in [
        "osc", "sig", "structure_bias", "setup_range_pos", "poc", "poc_macro",
        "long_ctx_rr", "short_ctx_rr", "long_ctx_pb", "short_ctx_pb",
        "long_ctx", "short_ctx", "long_family", "short_family",
        "long_stop", "short_stop", "long_tp", "short_tp"]}
    flags = {k: np.zeros(n, dtype=np.int64) for k in [
        "cross_up", "cross_down", "in_os", "in_ob", "long_ok", "short_ok",
        "vol_spike", "bull_spike", "bear_spike", "poc_reclaim", "poc_reject",
        "long_signal", "short_signal"]}

    lsh1 = lsh2 = lsl1 = lsl2 = float("nan")
    last_struct_bar = None

    for i in range(n):
        # structure update
        if not math.isnan(ph[i]):
            lsh2 = lsh1; lsh1 = ph[i]; last_struct_bar = i
        if not math.isnan(pl[i]):
            lsl2 = lsl1; lsl1 = pl[i]; last_struct_bar = i
        hh = not (math.isnan(lsh1) or math.isnan(lsh2)) and lsh1 > lsh2
        hl = not (math.isnan(lsl1) or math.isnan(lsl2)) and lsl1 > lsl2
        lh = not (math.isnan(lsh1) or math.isnan(lsh2)) and lsh1 < lsh2
        ll = not (math.isnan(lsl1) or math.isnan(lsl2)) and lsl1 < lsl2
        struct_fresh = last_struct_bar is not None and (i - last_struct_bar) <= struct_lookback_a
        has_low = not (math.isnan(lsl1) or math.isnan(lsl2))
        has_high = not (math.isnan(lsh1) or math.isnan(lsh2))
        long_conflict = hl and lh and not hh
        short_conflict = ll and lh and not hh
        long_ok = struct_fresh and has_low and not long_conflict and (hl or not ll)
        short_ok = struct_fresh and has_high and not short_conflict and (lh or not hh)
        bias = 1 if (hh and hl) else -1 if (ll and lh) else 0

        atr_i = atr_a[i]
        # HTF approximated by chart WT
        htf_bull = not math.isnan(wt_osc[i]) and wt_osc[i] > 0
        htf_bear = not math.isnan(wt_osc[i]) and wt_osc[i] < 0
        anchor_bull, anchor_bear = htf_bull, htf_bear

        # POC reclaim / reject
        poc_reclaim = (use_vol_poc and not math.isnan(poc_a[i]) and i >= 1 and not math.isnan(poc_a[i - 1])
                       and c[i] > poc_a[i] and c[i - 1] <= poc_a[i - 1])
        poc_reject = (use_vol_poc and not math.isnan(poc_a[i]) and i >= 1 and not math.isnan(poc_a[i - 1])
                      and c[i] < poc_a[i] and c[i - 1] >= poc_a[i - 1])
        bsr = 999
        for j in range(i, -1, -1):
            if (use_vol_poc and not math.isnan(poc_a[j]) and j >= 1 and not math.isnan(poc_a[j - 1])
                    and c[j] > poc_a[j] and c[j - 1] <= poc_a[j - 1]):
                bsr = i - j
                break
        bsj = 999
        for j in range(i, -1, -1):
            if (use_vol_poc and not math.isnan(poc_a[j]) and j >= 1 and not math.isnan(poc_a[j - 1])
                    and c[j] < poc_a[j] and c[j - 1] >= poc_a[j - 1]):
                bsj = i - j
                break
        reclaim_recent = bsr <= poc_recent_a
        reject_recent = bsj <= poc_recent_a

        # setup location / room
        s_high, s_low = setup_high[i], setup_low[i]
        s_range = s_high - s_low if not (math.isnan(s_high) or math.isnan(s_low)) else 0.0
        s_pos = (c[i] - s_low) / s_range if s_range > 0 else 0.5
        near_sup = atr_i > 0 and not math.isnan(lsl1) and lsl1 < c[i] and (c[i] - lsl1) / atr_i <= eff_prox
        near_res = atr_i > 0 and not math.isnan(lsh1) and lsh1 > c[i] and (lsh1 - c[i]) / atr_i <= eff_prox
        long_loc = (s_pos <= eff_edge) or (near_sup and long_ok)
        short_loc = (s_pos >= 1.0 - eff_edge) or (near_res and short_ok)
        long_target = lsh1 if (not math.isnan(lsh1) and lsh1 > c[i]) else s_high
        short_target = lsl1 if (not math.isnan(lsl1) and lsl1 < c[i]) else s_low
        long_room = ((long_target - c[i]) / atr_i) if atr_i > 0 else 0.0
        short_room = ((c[i] - short_target) / atr_i) if atr_i > 0 else 0.0
        long_room_ok = long_room >= min_target_room_atr
        short_room_ok = short_room >= min_target_room_atr

        # per-family volume gate
        chart_sec_le_1h = chart_sec <= 3600
        vol_req_rr = eff_vol_mode in ("All TF", "Reversal Only") or (eff_vol_mode == "Lower TF Only" and chart_sec_le_1h)
        vol_req_pb = eff_vol_mode in ("All TF", "Pullback Only") or (eff_vol_mode == "Lower TF Only" and chart_sec_le_1h)
        long_vol_ok_rr = (not vol_req_rr) or (not real) or bull_spike[i]
        long_vol_ok_pb = (not vol_req_pb) or (not real) or bull_spike[i]
        short_vol_ok_rr = (not vol_req_rr) or (not real) or bear_spike[i]
        short_vol_ok_pb = (not vol_req_pb) or (not real) or bear_spike[i]

        near_pb_ema = atr_i > 0 and abs(c[i] - pullback_ema[i]) / atr_i <= pullback_proximity_atr
        impulse_atr = abs(lsh1 - lsl1) / atr_i if (atr_i > 0 and has_low and has_high) else 0.0
        long_pb_trend = long_ok
        short_pb_trend = short_ok
        long_pb_loc = near_pb_ema or near_sup
        short_pb_loc = near_pb_ema or near_res
        long_pb_confirm = (not eff_long_pb) or (s_pos <= eff_edge) or (bull_spike[i] and long_ok)
        long_pb_setup = (eff_use_pb and long_pb_trend and long_pb_loc and impulse_atr >= min_impulse_atr
                         and long_room_ok and long_vol_ok_pb and long_pb_confirm)
        short_pb_setup = (eff_use_pb and short_pb_trend and short_pb_loc and impulse_atr >= min_impulse_atr
                          and short_room_ok and short_vol_ok_pb)
        short_rr_confirm = (not eff_short_confirm) or short_pb_loc or short_pb_trend
        long_rr_setup = long_loc and long_room_ok and long_vol_ok_rr
        short_rr_setup = short_loc and short_room_ok and short_vol_ok_rr and short_rr_confirm

        # MTF gates (chart WT proxy)
        mtf_hard_rr = eff_mtf_gate != "None" and eff_mtf_apply in ("All", "Reversal Only")
        mtf_hard_pb = eff_mtf_gate != "None" and eff_mtf_apply in ("All", "Pullback Only")
        mtf_long_ok_rr = (not mtf_hard_rr) or htf_bull
        mtf_short_ok_rr = (not mtf_hard_rr) or htf_bear
        mtf_long_ok_pb = (not mtf_hard_pb) or htf_bull
        mtf_short_ok_pb = (not mtf_hard_pb) or htf_bear

        long_htf_rr = 0.0 if mtf_hard_rr else (htf_weight if htf_bull else 0.0)
        short_htf_rr = 0.0 if mtf_hard_rr else (htf_weight if htf_bear else 0.0)
        long_htf_pb = 0.0 if mtf_hard_pb else (htf_weight if htf_bull else 0.0)
        short_htf_pb = 0.0 if mtf_hard_pb else (htf_weight if htf_bear else 0.0)
        anchor_req_rr = eff_anchor_apply in ("All", "Reversal Only")
        anchor_req_pb = eff_anchor_apply in ("All", "Pullback Only")
        a_long_raw = a_short_raw = 0.0
        if anchor_score_mode == "Alignment":
            a_long_raw = anchor_weight if anchor_bull else 0.0
            a_short_raw = anchor_weight if anchor_bear else 0.0
        elif anchor_score_mode == "Contrarian":
            a_long_raw = anchor_weight if not anchor_bull else 0.0
            a_short_raw = anchor_weight if not anchor_bear else 0.0
        long_anchor_rr = a_long_raw if anchor_req_rr else 0.0
        short_anchor_rr = a_short_raw if anchor_req_rr else 0.0
        long_anchor_pb = a_long_raw if anchor_req_pb else 0.0
        short_anchor_pb = a_short_raw if anchor_req_pb else 0.0

        # volatility regime bonus
        atr_sma = atr_sma_a[i]
        atr_norm = atr_a[i] / atr_sma if (not math.isnan(atr_sma) and atr_sma > 0) else np.nan
        in_band = (not math.isnan(atr_norm)) and volatility_regime_min <= atr_norm <= volatility_regime_max
        regime_bonus = 0.5 if (volatility_regime_mode != "Off" and in_band) else 0.0

        struct_hard_pb = use_setup_gate
        vol_hard_rr = use_setup_gate and vol_req_rr and real
        vol_hard_pb = use_setup_gate and vol_req_pb and real
        long_ctx_rr = (1 if long_ok else 0) + (1 if reclaim_recent else 0) + (1 if (not vol_hard_rr and bull_spike[i]) else 0) + long_htf_rr + long_anchor_rr + regime_bonus
        long_ctx_pb = (0 if struct_hard_pb else (1 if long_ok else 0)) + (1 if reclaim_recent else 0) + (1 if (not vol_hard_pb and bull_spike[i]) else 0) + long_htf_pb + long_anchor_pb + regime_bonus
        short_ctx_rr = (1 if short_ok else 0) + (1 if reject_recent else 0) + (1 if (not vol_hard_rr and bear_spike[i]) else 0) + short_htf_rr + short_anchor_rr + regime_bonus
        short_ctx_pb = (0 if struct_hard_pb else (1 if short_ok else 0)) + (1 if reject_recent else 0) + (1 if (not vol_hard_pb and bear_spike[i]) else 0) + short_htf_pb + short_anchor_pb + regime_bonus

        rr_trigger_long = cross_up[i]
        rr_trigger_short = cross_down[i]
        pb_trigger_long = cross_up[i] and (-eff_zone < wt_osc[i] < eff_zone)
        pb_trigger_short = cross_down[i] and (-eff_zone < wt_osc[i] < eff_zone)

        long_rr_pass = (not use_setup_gate) or long_rr_setup
        long_pb_pass = (not use_setup_gate) or long_pb_setup
        short_rr_pass = (not use_setup_gate) or short_rr_setup
        short_pb_pass = (not use_setup_gate) or short_pb_setup
        long_rr_sig = rr_trigger_long and long_rr_pass and mtf_long_ok_rr and long_ctx_rr >= eff_min
        long_pb_sig = pb_trigger_long and long_pb_pass and mtf_long_ok_pb and long_ctx_pb >= eff_min
        short_rr_sig = rr_trigger_short and short_rr_pass and mtf_short_ok_rr and short_ctx_rr >= eff_min
        short_pb_sig = pb_trigger_short and short_pb_pass and mtf_short_ok_pb and short_ctx_pb >= eff_min
        long_signal = long_rr_sig or long_pb_sig
        short_signal = short_rr_sig or short_pb_sig

        # stops / targets
        long_stop_struct = (lsl1 - stop_buffer_atr * atr_i) if (not math.isnan(lsl1) and lsl1 < c[i]) else (c[i] - fallback_stop_atr * atr_i)
        short_stop_struct = (lsh1 + stop_buffer_atr * atr_i) if (not math.isnan(lsh1) and lsh1 > c[i]) else (c[i] + fallback_stop_atr * atr_i)
        long_stop = min(long_stop_struct, c[i] - 1e-8)
        short_stop = max(short_stop_struct, c[i] + 1e-8)
        long_risk = max(c[i] - long_stop, 1e-8)
        short_risk = max(short_stop - c[i], 1e-8)
        long_tp = long_target if long_target > c[i] else c[i] + long_risk * min_target_room_atr
        short_tp = short_target if short_target < c[i] else c[i] - short_risk * min_target_room_atr

        out["osc"][i] = wt_osc[i]
        out["sig"][i] = wt_sig[i]
        out["structure_bias"][i] = bias
        out["setup_range_pos"][i] = s_pos
        out["poc"][i] = poc_a[i]
        out["poc_macro"][i] = pocm_a[i]
        out["long_ctx_rr"][i] = long_ctx_rr
        out["short_ctx_rr"][i] = short_ctx_rr
        out["long_ctx_pb"][i] = long_ctx_pb
        out["short_ctx_pb"][i] = short_ctx_pb
        out["long_ctx"][i] = max(long_ctx_rr, long_ctx_pb)
        out["short_ctx"][i] = max(short_ctx_rr, short_ctx_pb)
        out["long_family"][i] = 1 if long_rr_sig else 2 if long_pb_sig else 0
        out["short_family"][i] = 1 if short_rr_sig else 2 if short_pb_sig else 0
        out["long_stop"][i] = long_stop
        out["short_stop"][i] = short_stop
        out["long_tp"][i] = long_tp
        out["short_tp"][i] = short_tp
        flags["cross_up"][i] = 1 if cross_up[i] else 0
        flags["cross_down"][i] = 1 if cross_down[i] else 0
        flags["in_os"][i] = 1 if os_zone[i] else 0
        flags["in_ob"][i] = 1 if ob_zone[i] else 0
        flags["long_ok"][i] = 1 if long_ok else 0
        flags["short_ok"][i] = 1 if short_ok else 0
        flags["vol_spike"][i] = 1 if vol_spike[i] else 0
        flags["bull_spike"][i] = 1 if bull_spike[i] else 0
        flags["bear_spike"][i] = 1 if bear_spike[i] else 0
        flags["poc_reclaim"][i] = 1 if poc_reclaim else 0
        flags["poc_reject"][i] = 1 if poc_reject else 0
        flags["long_signal"][i] = 1 if long_signal else 0
        flags["short_signal"][i] = 1 if short_signal else 0

    df = pd.DataFrame(index=idx)
    df["wt3_osc"] = out["osc"]
    df["wt3_sig"] = out["sig"]
    df["wt3_structure_bias"] = pd.Series(out["structure_bias"], index=idx).astype("Int64")
    df["wt3_setup_range_pos"] = out["setup_range_pos"]
    df["wt3_poc"] = out["poc"]
    df["wt3_poc_macro"] = out["poc_macro"]
    df["wt3_long_ctx_rr"] = out["long_ctx_rr"]
    df["wt3_short_ctx_rr"] = out["short_ctx_rr"]
    df["wt3_long_ctx_pb"] = out["long_ctx_pb"]
    df["wt3_short_ctx_pb"] = out["short_ctx_pb"]
    df["wt3_long_ctx"] = out["long_ctx"]
    df["wt3_short_ctx"] = out["short_ctx"]
    df["wt3_long_family"] = pd.Series(out["long_family"], index=idx).astype("Int64")
    df["wt3_short_family"] = pd.Series(out["short_family"], index=idx).astype("Int64")
    df["wt3_long_stop"] = out["long_stop"]
    df["wt3_short_stop"] = out["short_stop"]
    df["wt3_long_tp"] = out["long_tp"]
    df["wt3_short_tp"] = out["short_tp"]
    for k, v in flags.items():
        df[f"wt3_{k}"] = pd.Series(v, index=idx).astype("Int64")
    return df
