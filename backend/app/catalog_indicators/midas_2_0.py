"""MIDAS 2.0 [WavesUnchained]  v2.7.3  —  Python port.

Anchored MIDAS support/resistance curve (launch-anchored volume-weighted price)
with a true topfinder / bottomfinder accelerated exhaustion curve.

Computational core ported as a per-bar sequential state machine (the MIDAS
cumulative episode cannot be vectorised because the anchor re-seeds the running
statistics and the TBF fit is stateful):

  * Anchor engine: Manual Time / Daily Open / Auto Last Swing / Auto Swing Low /
    Auto Swing High, with Backpainted (re-seed from the real pivot bar) or
    Confirmed (start at the live-known confirmation bar) display.
  * Episode cumulative sums A=ΣpV, C=ΣV, P2=Σp²V, B=ΣpVd, E=ΣVd (d = running ΣV).
  * MIDAS curve = A/C plus VW-StDev / ATR / Hybrid σ bands at two multiples.
  * Topfinder / bottomfinder closed-form auto-fit with stable-fit gating and the
    accelerated TBF curve value + phase (developing/mature/late/passed).
  * Location (normalized distance, zone) and reclaim/stretch context scores.

Dropped (display / logging / non-computational):
  * polyline / label / fill / plotshape / table / dashboard / alertcondition /
    log.info / live badge / debug log / colour inputs.
  * ``showBands`` / ``bandDisplay`` / ``visualMode`` / ``showMarkers`` /
    ``liveBadgeMode`` / ``tablePos`` are visual-only and do not change numbers.

Volume contract: the port assumes the provided ``frame`` carries meaningful
``volume``.  When ``volume`` is absent or ``volume_weight_mode == "Equal weight"``
every bar is weighted ``1`` (Pine's deliberate equal-weight path).  The original
"volume not available → indicator is NaN" guard is dropped because a backtest
frame is expected to supply real volume; pass ``volume_is_real=False`` to force
the equal-weight fallback regardless.

``barstate.isconfirmed`` is taken as always true for historical bars, so
marker-gated conditions are evaluated every bar.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import atr, choice, ema, integer, number, pivotlow_strict, pivothigh_strict

_TBF_STATE_CODE = {
    "off": 0, "no fit": 1, "developing": 2, "mature": 3, "late": 4, "passed": 5,
}
_ZONE_CODE = {  # from close relative to the four bands
    "below L2": 0, "L1-L2": 1, "inside": 2, "U1-U2": 3, "above U2": 4,
}


def _src_series(frame: pd.DataFrame, src: str) -> np.ndarray:
    o = frame["open"].to_numpy(dtype=float)
    h = frame["high"].to_numpy(dtype=float)
    l = frame["low"].to_numpy(dtype=float)
    c = frame["close"].to_numpy(dtype=float)
    if src == "open":
        return o
    if src == "high":
        return h
    if src == "low":
        return l
    if src == "close":
        return c
    if src == "hl2":
        return (h + l) / 2.0
    if src == "hlc3":
        return (h + l + c) / 3.0
    if src == "ohlc4":
        return (o + h + l + c) / 4.0
    raise ValueError(f"未知 src 选择器: {src}")


def midas_2_0(
    frame: pd.DataFrame,
    *,
    anchor_mode: str = "Auto Last Swing",
    anchor_display_mode: str = "Backpainted",
    anchor_time=None,
    anchor_left: int = 20,
    anchor_right: int = 20,
    src: str = "hlc3",
    volume_weight_mode: str = "Per-bar fallback",
    show_bands: bool = True,
    band_display: str = "Zones",
    band_mode: str = "VW StDev",
    band_mult1: float = 1.0,
    band_mult2: float = 2.0,
    atr_len: int = 14,
    enable_tbf: bool = True,
    warn_thresh: float = 0.85,
    fit_min_progress: float = 0.20,
    fit_stable_tolerance: float = 0.25,
    tbf_history_mode: str = "From Fit",
    tbf_stale_progress: float = 1.50,
    fit_left: int = 5,
    fit_right: int = 5,
    curve_bias: str = "Auto EMA",
    ema_len: int = 55,
    bias_hold_bars: int = 3,
    visual_mode: str = "Active",
    show_markers: bool = True,
    reclaim_look: int = 10,
    reclaim_vol_len: int = 20,
    reclaim_struct_len: int = 5,
    reclaim_close_location_min: float = 0.60,
    markers_on_bar_close: bool = True,
    volume_is_real=None,
):
    anchor_mode = choice("anchor_mode", anchor_mode,
                         ["Manual Time", "Daily Open", "Auto Last Swing",
                          "Auto Swing Low", "Auto Swing High"])
    anchor_display_mode = choice("anchor_display_mode", anchor_display_mode,
                                 ["Backpainted", "Confirmed"])
    anchor_left = integer("anchor_left", anchor_left, 1, 200)
    anchor_right = integer("anchor_right", anchor_right, 1, 200)
    src = choice("src", src, ["open", "high", "low", "close", "hl2", "hlc3", "ohlc4"])
    volume_weight_mode = choice("volume_weight_mode", volume_weight_mode,
                                ["Per-bar fallback", "Equal weight"])
    band_display = choice("band_display", band_display, ["Zones", "Lines", "Zones + Lines"])
    band_mode = choice("band_mode", band_mode, ["VW StDev", "ATR", "Hybrid"])
    band_mult1 = number("band_mult1", band_mult1, 0.0, 20.0)
    band_mult2 = number("band_mult2", band_mult2, 0.0, 20.0)
    atr_len = integer("atr_len", atr_len, 1, 500)
    warn_thresh = number("warn_thresh", warn_thresh, 0.5, 0.99)
    fit_min_progress = number("fit_min_progress", fit_min_progress, 0.05, 0.50)
    fit_stable_tolerance = number("fit_stable_tolerance", fit_stable_tolerance, 0.05, 1.0)
    tbf_history_mode = choice("tbf_history_mode", tbf_history_mode, ["From Fit", "Retrospective"])
    tbf_stale_progress = number("tbf_stale_progress", tbf_stale_progress, 1.10, 3.0)
    fit_left = integer("fit_left", fit_left, 1, 100)
    fit_right = integer("fit_right", fit_right, 1, 100)
    curve_bias = choice("curve_bias", curve_bias, ["Auto EMA", "Up", "Down"])
    ema_len = integer("ema_len", ema_len, 5, 500)
    bias_hold_bars = integer("bias_hold_bars", bias_hold_bars, 1, 50)
    reclaim_look = integer("reclaim_look", reclaim_look, 1, 200)
    reclaim_vol_len = integer("reclaim_vol_len", reclaim_vol_len, 1, 500)
    reclaim_struct_len = integer("reclaim_struct_len", reclaim_struct_len, 1, 200)
    reclaim_close_location_min = number("reclaim_close_location_min", reclaim_close_location_min, 0.5, 1.0)

    idx = frame.index
    n = len(frame)
    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    src_arr = _src_series(frame, src)
    atr_arr = atr(frame, atr_len).to_numpy(dtype=float)
    ema_arr = ema(pd.Series(close, index=idx), ema_len).to_numpy(dtype=float)

    has_volume = "volume" in frame.columns and bool(np.nansum(frame["volume"].to_numpy(dtype=float)) > 0)
    if has_volume:
        vol = frame["volume"].to_numpy(dtype=float)
        real = True if volume_is_real is None else bool(volume_is_real)
    else:
        vol = None
        real = False if volume_is_real is None else bool(volume_is_real)

    # Pivot confirmation series (value reported at the confirmation bar).
    anch_low = pivotlow_strict(pd.Series(low, index=idx), anchor_left, anchor_right).to_numpy(dtype=float)
    anch_high = pivothigh_strict(pd.Series(high, index=idx), anchor_left, anchor_right).to_numpy(dtype=float)
    fit_low = pivotlow_strict(pd.Series(low, index=idx), fit_left, fit_right).to_numpy(dtype=float)
    fit_high = pivothigh_strict(pd.Series(high, index=idx), fit_left, fit_right).to_numpy(dtype=float)

    if anchor_mode == "Manual Time":
        if anchor_time is None:
            anchor_time = idx[0]
        anchor_time = pd.Timestamp(anchor_time)

    # Reclaim support series (vectorised where possible).
    if has_volume and real and volume_weight_mode != "Equal weight":
        avg_vol = pd.Series(vol, index=idx).rolling(reclaim_vol_len).mean().to_numpy(dtype=float)
    else:
        avg_vol = None
    hi_prev = pd.Series(high, index=idx).shift(1).rolling(reclaim_struct_len).max().to_numpy(dtype=float)
    lo_prev = pd.Series(low, index=idx).shift(1).rolling(reclaim_struct_len).min().to_numpy(dtype=float)

    def bar_weight(i: int) -> float:
        if not real:
            return 1.0
        if volume_weight_mode == "Equal weight":
            return 1.0
        vi = vol[i] if vol is not None else 1.0
        if math.isnan(vi) or vi <= 0:
            return 1.0
        return float(vi)

    is_pivot_mode = anchor_mode in ("Auto Swing Low", "Auto Swing High", "Auto Last Swing")
    backpainted = anchor_display_mode == "Backpainted"

    # Episode cumulative history (full-length arrays indexed by bar).
    cumPV = np.full(n, np.nan)
    cumV = np.full(n, np.nan)
    cumP2V = np.full(n, np.nan)
    cumPVd = np.full(n, np.nan)
    cumVd = np.full(n, np.nan)
    age = np.full(n, np.nan)

    # Output arrays.
    midas = np.full(n, np.nan)
    vw_dev = np.full(n, np.nan)
    atr_dev = np.full(n, np.nan)
    dev = np.full(n, np.nan)
    up1 = np.full(n, np.nan)
    lo1 = np.full(n, np.nan)
    up2 = np.full(n, np.nan)
    lo2 = np.full(n, np.nan)
    tbf = np.full(n, np.nan)
    tbf_progress = np.full(n, np.nan)
    norm_dist = np.full(n, np.nan)
    stretch_above = np.zeros(n, dtype=np.int64)
    stretch_below = np.zeros(n, dtype=np.int64)
    reclaim_up_score = np.zeros(n, dtype=np.int64)
    reclaim_dn_score = np.zeros(n, dtype=np.int64)
    tbf_reaction_score = np.zeros(n, dtype=np.int64)
    bias_arr = np.ones(n, dtype=np.int64)
    tbf_top_arr = np.zeros(n, dtype=np.int64)

    open_arr = frame["open"].to_numpy(dtype=float)

    # Running loop state.
    anchor_conf_bar = -1
    anchor_is_low = True
    cur_bias = 1
    bias_streak = 0
    tbfD = np.nan
    tbf_fit_count = 0
    tbf_fit_quality = "none"
    tbf_fit_accepted_bar = -1
    tbf_pending_D = np.nan
    tbf_pending_confirm = 0
    tbf_fit_reaction_seen = False
    tbf_fit_best_reaction = 0
    tbf_episode_reaction_seen = False
    tbf_episode_best_reaction = 0

    prev_tbf_top = None
    prev_close = np.nan
    prev_midas = np.nan
    prev_up1 = np.nan
    prev_lo1 = np.nan
    since_low = 10 ** 9
    since_high = 10 ** 9

    for b in range(n):
        # ----- anchor pulses -----
        if anchor_mode == "Manual Time":
            anchor_active = idx[b] >= anchor_time
            manual_pulse = anchor_active and (b == 0 or idx[b - 1] < anchor_time)
            anchor_pulse = manual_pulse
        else:
            anchor_active = True
            if anchor_mode == "Daily Open":
                anchor_pulse = (b == 0) or (idx[b].date() != idx[b - 1].date())
            elif anchor_mode == "Auto Swing Low":
                anchor_pulse = not math.isnan(anch_low[b])
            elif anchor_mode == "Auto Swing High":
                anchor_pulse = not math.isnan(anch_high[b])
            else:  # Auto Last Swing
                anchor_pulse = (not math.isnan(anch_low[b])) or (not math.isnan(anch_high[b]))

        if anchor_pulse:
            anchor_conf_bar = b
        if anchor_pulse and is_pivot_mode:
            anchor_is_low = not math.isnan(anch_low[b])  # prefer the low if both

        # ----- MIDAS cumulative sums -----
        if not anchor_active:
            cumPV[b] = cumV[b] = cumP2V[b] = cumPVd[b] = cumVd[b] = np.nan
            age[b] = np.nan
        elif anchor_pulse or math.isnan(cumPV[b - 1]) if b > 0 else True:
            # (re)seed the episode
            if is_pivot_mode and anchor_pulse and backpainted:
                start = max(0, b - anchor_right)
                a = 0.0
                c = 0.0
                p2 = 0.0
                bd = 0.0
                ed = 0.0
                for j in range(start, b + 1):
                    p = src_arr[j]
                    w = bar_weight(j)
                    c += w
                    a += p * w
                    p2 += p * p * w
                    bd += p * w * c
                    ed += w * c
                    cumPV[j] = a
                    cumV[j] = c
                    cumP2V[j] = p2
                    cumPVd[j] = bd
                    cumVd[j] = ed
                age[b] = (b - start) + 1
            else:
                w = bar_weight(b)
                cumPV[b] = src_arr[b] * w
                cumV[b] = w
                cumP2V[b] = src_arr[b] * src_arr[b] * w
                cumPVd[b] = src_arr[b] * w * w
                cumVd[b] = w * w
                age[b] = 1
        else:
            w = bar_weight(b)
            cv = cumV[b - 1]
            dj = cv + w
            cumPVd[b] = cumPVd[b - 1] + src_arr[b] * w * dj
            cumVd[b] = cumVd[b - 1] + w * dj
            cumPV[b] = cumPV[b - 1] + src_arr[b] * w
            cumV[b] = dj
            cumP2V[b] = cumP2V[b - 1] + src_arr[b] * src_arr[b] * w
            age[b] = age[b - 1] + 1

        cV = cumV[b]
        if not math.isnan(cV) and cV > 0:
            m = cumPV[b] / cV
            midas[b] = m
            var = max(cumP2V[b] / cV - m * m, 0.0)
            vw_dev[b] = math.sqrt(var)
        atr_dev[b] = atr_arr[b]
        if band_mode == "VW StDev":
            dev[b] = vw_dev[b]
        elif band_mode == "ATR":
            dev[b] = atr_dev[b]
        else:
            dev[b] = (vw_dev[b] + atr_dev[b]) / 2.0 if not math.isnan(vw_dev[b]) else atr_dev[b]

        d = dev[b]
        if not math.isnan(midas[b]) and not math.isnan(d):
            up1[b] = midas[b] + d * band_mult1
            lo1[b] = midas[b] - d * band_mult1
            up2[b] = midas[b] + d * band_mult2
            lo2[b] = midas[b] - d * band_mult2

        # ----- bias -----
        if curve_bias == "Up":
            cur_bias = 1
        elif curve_bias == "Down":
            cur_bias = -1
        else:
            opposite = (cur_bias == 1 and close[b] < ema_arr[b]) or (cur_bias == -1 and close[b] >= ema_arr[b])
            if opposite:
                bias_streak += 1
                if bias_streak >= bias_hold_bars:
                    cur_bias *= -1
                    bias_streak = 0
            else:
                bias_streak = 0
        bias_arr[b] = cur_bias

        # ----- TBF direction -----
        if anchor_mode == "Auto Swing Low":
            tbf_top = True
        elif anchor_mode == "Auto Swing High":
            tbf_top = False
        elif anchor_mode == "Auto Last Swing":
            tbf_top = anchor_is_low
        else:
            tbf_top = cur_bias == 1
        tbf_top_arr[b] = 1 if tbf_top else 0

        # Reset fit state on re-anchor or direction flip (non-pivot modes).
        dir_changed = (b > 0) and (not is_pivot_mode) and (prev_tbf_top is not None) and (tbf_top != prev_tbf_top)
        if anchor_pulse or dir_changed:
            tbfD = np.nan
            tbf_fit_count = 0
            tbf_fit_quality = "none"
            tbf_fit_accepted_bar = -1
            tbf_pending_D = np.nan
            tbf_pending_confirm = 0
            tbf_fit_reaction_seen = False
            tbf_fit_best_reaction = 0
            tbf_episode_reaction_seen = False
            tbf_episode_best_reaction = 0

        # ----- TBF fit -----
        fitT = np.nan
        do_fit = False
        fit_pivot_bar = b - fit_right
        if enable_tbf and not math.isnan(cumPV[b]) and age[b] >= fit_right + 1:
            if tbf_top and not math.isnan(fit_low[b]) and fit_pivot_bar >= 0:
                fitT = low[fit_pivot_bar]
                do_fit = True
            elif (not tbf_top) and not math.isnan(fit_high[b]) and fit_pivot_bar >= 0:
                fitT = high[fit_pivot_bar]
                do_fit = True

        if do_fit:
            Af = cumPV[fit_pivot_bar]
            Bf = cumPVd[fit_pivot_bar]
            Cf = cumV[fit_pivot_bar]
            Ef = cumVd[fit_pivot_bar]
            den = Af - fitT * Cf
            Dcand = (Bf - fitT * Ef) / den if den != 0 else np.nan
            fit_prog = (Cf / Dcand) if (not math.isnan(Dcand) and Dcand > 0) else np.nan
            forward = (not math.isnan(Dcand)) and Dcand > cumV[b] and Dcand < cumV[b] * 50
            mature = (not math.isnan(fit_prog)) and fit_prog >= fit_min_progress
            if forward and mature:
                prevD = tbfD
                shift = (abs(Dcand - prevD) / prevD) if (not math.isnan(prevD) and prevD > 0) else np.nan
                first = math.isnan(prevD)
                stable = (not first) and (not math.isnan(shift)) and shift <= fit_stable_tolerance
                if first or stable:
                    tbfD = Dcand
                    tbf_fit_count += 1
                    tbf_fit_quality = "provisional" if first else "stable"
                    tbf_fit_accepted_bar = b
                    tbf_pending_D = np.nan
                    tbf_pending_confirm = 0
                    if first:
                        tbf_fit_reaction_seen = False
                        tbf_fit_best_reaction = 0
                else:
                    agree = (not math.isnan(tbf_pending_D) and tbf_pending_D > 0
                             and abs(Dcand - tbf_pending_D) / tbf_pending_D <= fit_stable_tolerance)
                    if tbf_pending_confirm > 0 and agree:
                        tbfD = Dcand
                        tbf_fit_count += 1
                        tbf_fit_quality = "stable"
                        tbf_fit_accepted_bar = b
                        tbf_pending_D = np.nan
                        tbf_pending_confirm = 0
                        tbf_fit_reaction_seen = False
                        tbf_fit_best_reaction = 0
                    else:
                        tbf_pending_D = Dcand
                        tbf_pending_confirm = 1

        # ----- TBF curve value + phase -----
        progress = (cumV[b] / tbfD) if (enable_tbf and not math.isnan(tbfD)) else np.nan
        if enable_tbf and not math.isnan(tbfD) and tbfD > cumV[b] and not anchor_pulse:
            denom = cumV[b] * tbfD - cumVd[b]
            tbf[b] = (cumPV[b] * tbfD - cumPVd[b]) / denom if denom != 0 else np.nan
        if not math.isnan(progress):
            tbf_progress[b] = progress

        if not enable_tbf:
            tbf_state = "off"
        elif math.isnan(progress):
            tbf_state = "no fit"
        elif progress >= 1.0:
            tbf_state = "passed"
        elif progress >= warn_thresh:
            tbf_state = "late"
        elif progress >= 0.50:
            tbf_state = "mature"
        else:
            tbf_state = "developing"

        # ----- location + markers -----
        if not math.isnan(midas[b]) and not math.isnan(dev[b]) and dev[b] > 0:
            norm_dist[b] = (close[b] - midas[b]) / dev[b]

        if not math.isnan(lo1[b]) and close[b] < lo1[b]:
            since_low = 0
        else:
            since_low += 1
        if not math.isnan(up1[b]) and close[b] > up1[b]:
            since_high = 0
        else:
            since_high += 1
        stretch_low_recent = since_low <= reclaim_look
        stretch_high_recent = since_high <= reclaim_look

        raw_stretch_above = (not math.isnan(up2[b])) and high[b] > up2[b] and close[b] < up2[b]
        raw_stretch_below = (not math.isnan(lo2[b])) and low[b] < lo2[b] and close[b] > lo2[b]
        stretch_above[b] = 1 if raw_stretch_above else 0
        stretch_below[b] = 1 if raw_stretch_below else 0
        reclaim_up = stretch_low_recent and (prev_close <= prev_midas if not math.isnan(prev_midas) else False) and close[b] > midas[b]
        reclaim_dn = stretch_high_recent and (prev_close >= prev_midas if not math.isnan(prev_midas) else False) and close[b] < midas[b]

        # Reclaim Quality Score (0-4)
        bar_range = high[b] - low[b]
        close_loc = (close[b] - low[b]) / bar_range if bar_range > 0 else 0.5
        strong_up = close_loc >= reclaim_close_location_min
        strong_dn = (1 - close_loc) >= reclaim_close_location_min
        vol_above = (avg_vol is not None and not math.isnan(avg_vol[b])
                     and vol is not None and not math.isnan(vol[b]) and vol[b] > 0 and vol[b] > avg_vol[b])
        struct_up = close[b] > hi_prev[b] if not math.isnan(hi_prev[b]) else False
        struct_dn = close[b] < lo_prev[b] if not math.isnan(lo_prev[b]) else False
        tbf_exhausted = (tbf_state in ("late", "passed")) and not (not math.isnan(progress)
                         and progress >= tbf_stale_progress and not tbf_fit_reaction_seen)
        sup_bull = (not tbf_top) and tbf_exhausted
        sup_bear = tbf_top and tbf_exhausted
        if reclaim_up:
            reclaim_up_score[b] = int(strong_up) + int(vol_above) + int(struct_up) + int(sup_bull)
        if reclaim_dn:
            reclaim_dn_score[b] = int(strong_dn) + int(vol_above) + int(struct_dn) + int(sup_bear)

        # TBF price-reaction score (2/3+)
        if not math.isnan(tbf[b]):
            close_pos = (close[b] - low[b]) / (high[b] - low[b]) if high[b] > low[b] else 0.5
            react_candle = (tbf_top and close[b] < open_arr[b] and close_pos <= 0.35) or \
                           ((not tbf_top) and close[b] > open_arr[b] and close_pos >= 0.65)
            react_body = abs(close[b] - open_arr[b])
            react_struct = (tbf_top and close[b] < low[b - 1] and react_body >= atr_dev[b] * 0.5) or \
                           ((not tbf_top) and close[b] > high[b - 1] and react_body >= atr_dev[b] * 0.5)
            cross_under = (not math.isnan(prev_up1) and prev_close >= prev_up1 and close[b] < up1[b])
            cross_over = (not math.isnan(prev_lo1) and prev_close <= prev_lo1 and close[b] > lo1[b])
            react_band = cross_under if tbf_top else cross_over
            sc = int(react_candle) + int(react_struct) + int(react_band)
            tbf_reaction_score[b] = sc
            advanced = tbf_state in ("late", "passed")
            if enable_tbf and advanced and sc >= 2:
                tbf_fit_reaction_seen = True
                tbf_fit_best_reaction = max(tbf_fit_best_reaction, sc)
                tbf_episode_reaction_seen = True
                tbf_episode_best_reaction = max(tbf_episode_best_reaction, sc)

        prev_tbf_top = tbf_top
        prev_close = close[b]
        prev_midas = midas[b]
        prev_up1 = up1[b]
        prev_lo1 = lo1[b]

    # ----- assemble output -----
    def col(values, is_int=False):
        if is_int:
            return pd.Series(values, index=idx).astype("Int64")
        return pd.Series(values, index=idx)

    # zone (close vs bands) — recomputed vectorised
    zone = np.full(n, 2, dtype=np.int64)
    for b in range(n):
        if math.isnan(midas[b]):
            zone[b] = 2
        elif not math.isnan(up2[b]) and close[b] > up2[b]:
            zone[b] = 4
        elif not math.isnan(up1[b]) and close[b] > up1[b]:
            zone[b] = 3
        elif not math.isnan(lo1[b]) and close[b] > lo1[b]:
            zone[b] = 2
        elif not math.isnan(lo2[b]) and close[b] > lo2[b]:
            zone[b] = 1
        else:
            zone[b] = 0

    tbf_state_code = np.array([_TBF_STATE_CODE["off"]] * n, dtype=np.int64)
    # state was computed per-bar but not stored; recompute cheaply from progress
    for b in range(n):
        if not enable_tbf:
            tbf_state_code[b] = 0
        elif math.isnan(tbf_progress[b]):
            tbf_state_code[b] = 1
        elif tbf_progress[b] >= 1.0:
            tbf_state_code[b] = 5
        elif tbf_progress[b] >= warn_thresh:
            tbf_state_code[b] = 4
        elif tbf_progress[b] >= 0.50:
            tbf_state_code[b] = 3
        else:
            tbf_state_code[b] = 2

    out = pd.DataFrame(index=idx)
    out["midas"] = col(midas)
    out["midas_upper1"] = col(up1)
    out["midas_lower1"] = col(lo1)
    out["midas_upper2"] = col(up2)
    out["midas_lower2"] = col(lo2)
    out["midas_dev"] = col(dev)
    out["midas_vw_dev"] = col(vw_dev)
    out["midas_atr_dev"] = col(atr_dev)
    out["tbf"] = col(tbf)
    out["tbf_progress"] = col(tbf_progress)
    out["tbf_state"] = pd.Series(tbf_state_code, index=idx).astype("Int64")
    out["normalized_distance"] = col(norm_dist)
    out["zone"] = pd.Series(zone, index=idx).astype("Int64")
    out["reclaim_up_score"] = pd.Series(reclaim_up_score, index=idx).astype("Int64")
    out["reclaim_dn_score"] = pd.Series(reclaim_dn_score, index=idx).astype("Int64")
    out["tbf_reaction_score"] = pd.Series(tbf_reaction_score, index=idx).astype("Int64")
    out["bias"] = pd.Series(bias_arr, index=idx).astype("Int64")
    out["tbf_top"] = pd.Series(tbf_top_arr, index=idx).astype("Int64")
    out["stretch_above"] = pd.Series(stretch_above, index=idx).astype("Int64")
    out["stretch_below"] = pd.Series(stretch_below, index=idx).astype("Int64")
    return out
