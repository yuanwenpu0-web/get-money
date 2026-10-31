"""Klinger Volume Force Map [WavesUnchained] (v1.0.3) — catalog port.

Source: data/pine-collection/indicators/money_flow/klinger_volume_force_map/
        klinger_volume_force_map_v1_0_0.pine  (//@version=6, 1135 lines)

Stephen J. Klinger's 1997 volume-force concept, ported cell-for-cell:

  * Volume Force (VF)  = V · f(ratio) · trend · 100  with three formula variants
    (Original 1997 / TradingView Docs / Signed-Volume baseline).
  * KVO    = EMA(VF, fast) − EMA(VF, slow);  Signal = EMA(KVO, sig).
  * Normalization (zero-preserving) is the canonical cross-market view;
    Relative Force = KVO / EMA(|KVO|, N) is also the regime metric.
  * Hysteretic Bull / Neutral / Bear regime engine with optional confirmation.
  * Confirmed regular + hidden divergence on confirmed PRICE pivots (quality
    filtered by ATR / KVO displacement / separation).
  * Flow Rejection continuation events and flow-confirmed structure breaks.

The function computes the numeric/flag surface only (no plots / dashboard).
Visual-only toggles from the original are intentionally omitted; every input
that changes a computed value is mirrored as a keyword-only parameter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (
    atr, choice, boolean, correlation, crossover, crossunder, ema,
    highest, integer, lowest, nz, number, ohlcv, pivotlow_strict,
    pivothigh_strict, stdev, sum_,
)


def klinger_volume_force_map(
    frame: pd.DataFrame,
    *,
    formula_mode: str = "Original 1997",
    fast_len: int = 34,
    slow_len: int = 55,
    sig_len: int = 13,
    norm_mode: str = "Relative Force",
    norm_len: int = 100,
    norm_floor: float = 1e-10,
    slope_smooth_len: int = 3,
    hist_smooth_len: int = 1,
    regime_enter: float = 0.35,
    regime_exit: float = 0.20,
    regime_confirm_bars: int = 1,
    strong_needs_histogram: bool = True,
    strong_needs_slope: bool = True,
    div_regular: bool = True,
    div_hidden: bool = True,
    pivot_left: int = 5,
    pivot_right: int = 5,
    div_min_bars: int = 5,
    div_max_bars: int = 80,
    div_quality_filter: str = "Moderate",
    div_price_atr_min: float = 0.35,
    div_kvo_norm_min: float = 0.15,
    div_strong_sep: int = 12,
    div_require_regime_context: bool = False,
    show_signal_crosses: bool = False,
    show_zero_crosses: bool = False,
    show_rejections: bool = True,
    rejection_context_bars: int = 35,
    rejection_window: int = 20,
    rejection_touch: float = 0.10,
    rejection_require_hist_turn: bool = True,
    show_flow_confirm: bool = True,
    flow_confirm_strong_only: bool = False,
    correlation_len: int = 200,
    volume_is_real: bool | None = None,
) -> pd.DataFrame:
    # ---- parameter validation (in-function, mirrors Pine minval/maxval) ----
    formula_mode = choice(
        "formula_mode", formula_mode,
        ("Original 1997", "TradingView Documentation", "Signed Volume Simplified"))
    norm_mode = choice("norm_mode", norm_mode, ("Relative Force", "StdDev Scale", "Raw"))
    div_quality_filter = choice("div_quality_filter", div_quality_filter, ("All", "Moderate", "Strong"))
    fast_len = integer("fast_len", fast_len, 1)
    slow_len = integer("slow_len", slow_len, 2)
    sig_len = integer("sig_len", sig_len, 1)
    norm_len = integer("norm_len", norm_len, 20, 1000)
    slope_smooth_len = integer("slope_smooth_len", slope_smooth_len, 1, 50)
    hist_smooth_len = integer("hist_smooth_len", hist_smooth_len, 1, 20)
    regime_confirm_bars = integer("regime_confirm_bars", regime_confirm_bars, 1, 20)
    pivot_left = integer("pivot_left", pivot_left, 1, 50)
    pivot_right = integer("pivot_right", pivot_right, 1, 50)
    div_min_bars = integer("div_min_bars", div_min_bars, 1, 200)
    div_max_bars = integer("div_max_bars", div_max_bars, 2, 500)
    div_strong_sep = integer("div_strong_sep", div_strong_sep, 2, 200)
    rejection_context_bars = integer("rejection_context_bars", rejection_context_bars, 5, 300)
    rejection_window = integer("rejection_window", rejection_window, 2, 100)
    correlation_len = integer("correlation_len", correlation_len, 30, 2000)
    norm_floor = number("norm_floor", norm_floor, 1e-12)
    regime_enter = number("regime_enter", regime_enter, 0.01)
    regime_exit = number("regime_exit", regime_exit, 0.0)
    div_price_atr_min = number("div_price_atr_min", div_price_atr_min, 0.0)
    div_kvo_norm_min = number("div_kvo_norm_min", div_kvo_norm_min, 0.0)
    rejection_touch = number("rejection_touch", rejection_touch, 0.0)
    strong_needs_histogram = boolean("strong_needs_histogram", strong_needs_histogram)
    strong_needs_slope = boolean("strong_needs_slope", strong_needs_slope)
    div_regular = boolean("div_regular", div_regular)
    div_hidden = boolean("div_hidden", div_hidden)
    div_require_regime_context = boolean("div_require_regime_context", div_require_regime_context)
    show_signal_crosses = boolean("show_signal_crosses", show_signal_crosses)
    show_zero_crosses = boolean("show_zero_crosses", show_zero_crosses)
    show_rejections = boolean("show_rejections", show_rejections)
    rejection_require_hist_turn = boolean("rejection_require_hist_turn", rejection_require_hist_turn)
    show_flow_confirm = boolean("show_flow_confirm", show_flow_confirm)
    flow_confirm_strong_only = boolean("flow_confirm_strong_only", flow_confirm_strong_only)
    if volume_is_real is not None:
        volume_is_real = boolean("volume_is_real", volume_is_real)

    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    idx = frame.index
    n = len(frame)

    high_a = high.to_numpy(dtype=float)
    low_a = low.to_numpy(dtype=float)
    close_a = close.to_numpy(dtype=float)
    vol_a = volume.to_numpy(dtype=float)

    # ---- volume validity guard ----
    if volume_is_real is False:
        vol = np.full(n, np.nan)
    else:
        feed = np.maximum.accumulate((vol_a > 0).astype(float))
        vol = np.where(feed > 0, vol_a, np.nan)

    # ---- Klinger trend (stateful: flat bars keep prior trend) ----
    sum_hlc = high_a + low_a + close_a
    diff = np.zeros(n)
    diff[1:] = sum_hlc[1:] - sum_hlc[:-1]
    sign = np.sign(diff)
    sign[sign == 0] = np.nan
    ktr = pd.Series(sign, index=idx).ffill().fillna(1.0).to_numpy().astype(int)

    # ---- CM: cumulative dm within a trend run, reset base at flip ----
    dm = np.maximum(high_a - low_a, 0.0)
    cum = np.cumsum(dm)
    starts = np.ones(n, dtype=bool)
    starts[1:] = ktr[1:] != ktr[:-1]
    run_start = np.empty(n, dtype=int)
    ptr = 0
    for i in range(n):
        if starts[i]:
            ptr = i
        run_start[i] = ptr
    base = np.zeros(n)
    valid = run_start >= 2
    base[valid] = cum[run_start[valid] - 2]
    cm = cum - base

    # ---- ratio + Volume Force variants ----
    ratio = np.where(np.abs(cm) > norm_floor, dm / cm, 0.0)
    kt = ktr.astype(float)
    vf_original = vol * np.abs(2.0 * ratio - 1.0) * kt * 100.0
    vf_tv = vol * (2.0 * (ratio - 1.0)) * kt * 100.0
    vf_signed = vol * kt * 100.0

    if formula_mode == "Original 1997":
        vf = vf_original
    elif formula_mode == "TradingView Documentation":
        vf = vf_tv
    else:
        vf = vf_signed

    # ---- KVO / Signal for every formula variant ----
    kvo_original = ema(pd.Series(vf_original, index=idx), fast_len) - ema(pd.Series(vf_original, index=idx), slow_len)
    sig_original = ema(kvo_original, sig_len)
    kvo_tv = ema(pd.Series(vf_tv, index=idx), fast_len) - ema(pd.Series(vf_tv, index=idx), slow_len)
    sig_tv = ema(kvo_tv, sig_len)
    kvo_signed = ema(pd.Series(vf_signed, index=idx), fast_len) - ema(pd.Series(vf_signed, index=idx), slow_len)
    sig_signed = ema(kvo_signed, sig_len)

    if formula_mode == "Original 1997":
        kvo_raw, signal_raw = kvo_original, sig_original
    elif formula_mode == "TradingView Documentation":
        kvo_raw, signal_raw = kvo_tv, sig_tv
    else:
        kvo_raw, signal_raw = kvo_signed, sig_signed

    kvo_raw_a = kvo_raw.to_numpy(dtype=float)
    signal_raw_a = signal_raw.to_numpy(dtype=float)

    hist_unsmoothed = kvo_raw - signal_raw
    hist_raw = hist_unsmoothed if hist_smooth_len <= 1 else ema(hist_unsmoothed, hist_smooth_len)

    # ---- normalization ----
    rel_den = np.maximum(ema(kvo_raw.abs(), norm_len).to_numpy(dtype=float), norm_floor)
    relative_kvo = kvo_raw_a / rel_den
    relative_signal = signal_raw_a / rel_den
    hist_raw_a = hist_raw.to_numpy(dtype=float)
    relative_hist = hist_raw_a / rel_den

    std_den = np.maximum(nz(stdev(kvo_raw, norm_len)).to_numpy(dtype=float), norm_floor)
    std_kvo = kvo_raw_a / std_den
    std_signal = signal_raw_a / std_den
    std_hist = hist_raw_a / std_den

    if norm_mode == "Relative Force":
        kvo_disp, signal_disp, hist_disp = relative_kvo, relative_signal, relative_hist
    elif norm_mode == "StdDev Scale":
        kvo_disp, signal_disp, hist_disp = std_kvo, std_signal, std_hist
    else:
        kvo_disp, signal_disp, hist_disp = kvo_raw_a, signal_raw_a, hist_raw_a

    # ---- dynamics ----
    kvo_slope_raw = kvo_raw - kvo_raw.shift(1)
    kvo_slope = kvo_slope_raw if slope_smooth_len <= 1 else ema(kvo_slope_raw, slope_smooth_len)
    kvo_slope_a = kvo_slope.to_numpy(dtype=float)
    hist_delta_raw = hist_raw - hist_raw.shift(1)
    hist_delta_raw_a = hist_delta_raw.to_numpy(dtype=float)
    hist_delta_rel_a = (pd.Series(relative_hist, index=idx) - pd.Series(relative_hist, index=idx).shift(1)).to_numpy(dtype=float)

    if strong_needs_slope:
        slope_ok_bull = kvo_slope_a > 0
        slope_ok_bear = kvo_slope_a < 0
    else:
        slope_ok_bull = np.ones(n, dtype=bool)
        slope_ok_bear = np.ones(n, dtype=bool)
    if strong_needs_histogram:
        hist_ok_bull = (hist_raw_a > 0) & (hist_delta_raw_a >= 0)
        hist_ok_bear = (hist_raw_a < 0) & (hist_delta_raw_a <= 0)
    else:
        hist_ok_bull = np.ones(n, dtype=bool)
        hist_ok_bear = np.ones(n, dtype=bool)

    bull_exp = ((relative_kvo > 0) & (kvo_raw_a > signal_raw_a) & slope_ok_bull & hist_ok_bull).astype(bool)
    bear_exp = ((relative_kvo < 0) & (kvo_raw_a < signal_raw_a) & slope_ok_bear & hist_ok_bear).astype(bool)

    # ---- regime engine (hysteresis + confirmation) ----
    enter_threshold = max(regime_enter, regime_exit + 1e-6)
    exit_threshold = min(regime_exit, enter_threshold - 1e-6)
    bull_enter_raw = relative_kvo >= enter_threshold
    bear_enter_raw = relative_kvo <= -enter_threshold
    bull_conf = (sum_(pd.Series(bull_enter_raw.astype(float), index=idx), regime_confirm_bars) >= regime_confirm_bars).fillna(False).to_numpy()
    bear_conf = (sum_(pd.Series(bear_enter_raw.astype(float), index=idx), regime_confirm_bars) >= regime_confirm_bars).fillna(False).to_numpy()

    base_regime = 0
    base_arr = np.empty(n, dtype=int)
    for b in range(n):
        if base_regime == 1:
            if relative_kvo[b] < exit_threshold:
                base_regime = -1 if bear_conf[b] else 0
        elif base_regime == -1:
            if relative_kvo[b] > -exit_threshold:
                base_regime = 1 if bull_conf[b] else 0
        else:
            if bull_conf[b]:
                base_regime = 1
            elif bear_conf[b]:
                base_regime = -1
        base_arr[b] = base_regime

    regime_state = np.where(
        base_arr == 1, np.where(bull_exp, 2, 1),
        np.where(base_arr == -1, np.where(bear_exp, -2, -1), 0))
    compression = np.abs(relative_kvo) < enter_threshold

    # ---- price structure pivots ----
    price_ph = pivothigh_strict(high, pivot_left, pivot_right)
    price_pl = pivotlow_strict(low, pivot_left, pivot_right)
    last_high = price_ph.ffill()
    last_low = price_pl.ffill()
    ph_arr = price_ph.to_numpy(dtype=float)
    pl_arr = price_pl.to_numpy(dtype=float)
    high_conf = np.where(~np.isnan(ph_arr))[0]
    low_conf = np.where(~np.isnan(pl_arr))[0]

    atr_series = atr(frame, 14)
    atr_a = atr_series.to_numpy(dtype=float)
    atr_fallback = np.nanmedian(atr_a)
    if not np.isfinite(atr_fallback):
        atr_fallback = 1e-8

    def quality_allowed(q: int) -> bool:
        if div_quality_filter == "All":
            return q >= 1
        if div_quality_filter == "Moderate":
            return q >= 2
        return q >= 3

    regular_bear = np.zeros(n, dtype=bool)
    hidden_bear = np.zeros(n, dtype=bool)
    regular_bull = np.zeros(n, dtype=bool)
    hidden_bull = np.zeros(n, dtype=bool)
    qmap_high: dict[int, int] = {}
    qmap_low: dict[int, int] = {}

    # --- high pivots: regular / hidden bear divergence ---
    prev_idx = None
    for c in high_conf:
        pivot_bar = c - pivot_right
        curr_price = ph_arr[c]
        curr_kvo = kvo_raw_a[pivot_bar]
        curr_rel = relative_kvo[pivot_bar]
        atr_piv = atr_a[pivot_bar]
        if not np.isfinite(atr_piv):
            atr_piv = atr_fallback
        if prev_idx is not None:
            sep = (c - pivot_right) - prev_idx
            valid_sep = (sep >= div_min_bars) and (sep <= div_max_bars)
            reg = valid_sep and (curr_price > prev_price) and (curr_kvo < prev_kvo)
            hid = valid_sep and (curr_price < prev_price) and (curr_kvo > prev_kvo)
            price_disp = abs(curr_price - prev_price) / max(atr_piv, 1e-12)
            kvo_disp_q = abs(curr_rel - prev_rel)
            q = (1 if price_disp >= div_price_atr_min else 0) + \
                (1 if kvo_disp_q >= div_kvo_norm_min else 0) + \
                (1 if sep >= div_strong_sep else 0)
            regime_ctx = max(curr_rel, prev_rel) > 0
            if div_require_regime_context and not regime_ctx:
                q = max(q - 1, 0)
            qmap_high[c] = q
            if div_regular and reg and quality_allowed(q):
                regular_bear[c] = True
            if div_hidden and hid and quality_allowed(q):
                hidden_bear[c] = True
        prev_idx = c - pivot_right
        prev_price = curr_price
        prev_kvo = curr_kvo
        prev_rel = curr_rel

    # --- low pivots: regular / hidden bull divergence ---
    prev_idx = None
    for c in low_conf:
        pivot_bar = c - pivot_right
        curr_price = pl_arr[c]
        curr_kvo = kvo_raw_a[pivot_bar]
        curr_rel = relative_kvo[pivot_bar]
        atr_piv = atr_a[pivot_bar]
        if not np.isfinite(atr_piv):
            atr_piv = atr_fallback
        if prev_idx is not None:
            sep = (c - pivot_right) - prev_idx
            valid_sep = (sep >= div_min_bars) and (sep <= div_max_bars)
            reg = valid_sep and (curr_price < prev_price) and (curr_kvo > prev_kvo)
            hid = valid_sep and (curr_price > prev_price) and (curr_kvo < prev_kvo)
            price_disp = abs(curr_price - prev_price) / max(atr_piv, 1e-12)
            kvo_disp_q = abs(curr_rel - prev_rel)
            q = (1 if price_disp >= div_price_atr_min else 0) + \
                (1 if kvo_disp_q >= div_kvo_norm_min else 0) + \
                (1 if sep >= div_strong_sep else 0)
            regime_ctx = min(curr_rel, prev_rel) < 0
            if div_require_regime_context and not regime_ctx:
                q = max(q - 1, 0)
            qmap_low[c] = q
            if div_regular and reg and quality_allowed(q):
                regular_bull[c] = True
            if div_hidden and hid and quality_allowed(q):
                hidden_bull[c] = True
        prev_idx = c - pivot_right
        prev_price = curr_price
        prev_kvo = curr_kvo
        prev_rel = curr_rel

    def _carry(qmap: dict[int, int], conf, length: int) -> np.ndarray:
        arr = np.zeros(length, dtype=int)
        last = 0
        ci = 0
        for b in range(length):
            if ci < len(conf) and conf[ci] == b:
                last = qmap.get(b, 0)
                ci += 1
            arr[b] = last
        return arr

    quality_bear_filled = _carry(qmap_high, high_conf, n)
    quality_bull_filled = _carry(qmap_low, low_conf, n)

    # ---- basic crosses ----
    bull_signal_cross = crossover(kvo_raw, signal_raw).to_numpy()
    bear_signal_cross = crossunder(kvo_raw, signal_raw).to_numpy()
    bull_zero_cross = crossover(kvo_raw, 0.0).to_numpy()
    bear_zero_cross = crossunder(kvo_raw, 0.0).to_numpy()

    # ---- flow rejection (stateful arm windows) ----
    recent_bull_ctx = (highest(pd.Series(relative_kvo, index=idx), rejection_context_bars) >= enter_threshold).to_numpy()
    recent_bear_ctx = (lowest(pd.Series(relative_kvo, index=idx), rejection_context_bars) <= -enter_threshold).to_numpy()
    bull_touch = recent_bull_ctx & (relative_kvo <= rejection_touch)
    bear_touch = recent_bear_ctx & (relative_kvo >= -rejection_touch)

    if rejection_require_hist_turn:
        bull_hist_ok = hist_delta_raw_a > 0
        bear_hist_ok = hist_delta_raw_a < 0
    else:
        bull_hist_ok = np.ones(n, dtype=bool)
        bear_hist_ok = np.ones(n, dtype=bool)

    bull_rej = np.zeros(n, dtype=bool)
    bear_rej = np.zeros(n, dtype=bool)
    bull_arm = None
    bear_arm = None
    for b in range(n):
        if bull_touch[b] and not recent_bear_ctx[b]:
            bull_arm = b
        if bear_touch[b] and not recent_bull_ctx[b]:
            bear_arm = b
        if bull_arm is not None and b - bull_arm > rejection_window:
            bull_arm = None
        if bear_arm is not None and b - bear_arm > rejection_window:
            bear_arm = None
        bf = (show_rejections and bull_arm is not None and bull_signal_cross[b]
              and kvo_slope_a[b] > 0 and bull_hist_ok[b] and relative_kvo[b] > -enter_threshold)
        br = (show_rejections and bear_arm is not None and bear_signal_cross[b]
              and kvo_slope_a[b] < 0 and bear_hist_ok[b] and relative_kvo[b] < enter_threshold)
        if bf:
            bull_arm = None
            bear_arm = None
        if br:
            bull_arm = None
            bear_arm = None
        bull_rej[b] = bf
        bear_rej[b] = br

    # ---- flow-confirmed structure breaks ----
    last_high_prev = last_high.shift(1)
    last_low_prev = last_low.shift(1)
    bull_sb = last_high_prev.notna().to_numpy() & crossover(close, last_high_prev).to_numpy()
    bear_sb = last_low_prev.notna().to_numpy() & crossunder(close, last_low_prev).to_numpy()
    bull_regime_ok = (regime_state == 2) if flow_confirm_strong_only else (regime_state > 0)
    bear_regime_ok = (regime_state == -2) if flow_confirm_strong_only else (regime_state < 0)
    bull_fc = show_flow_confirm & bull_sb & bull_regime_ok & (kvo_raw_a > signal_raw_a) & (kvo_slope_a > 0) & (hist_raw_a > 0)
    bear_fc = show_flow_confirm & bear_sb & bear_regime_ok & (kvo_raw_a < signal_raw_a) & (kvo_slope_a < 0) & (hist_raw_a < 0)

    # ---- last event priority chain ----
    code = np.zeros(n, dtype=int)
    for mask, c in [
        (regular_bull, 1), (regular_bear, 2), (hidden_bull, 3), (hidden_bear, 4),
        (bull_rej, 5), (bear_rej, 6), (bull_fc, 7), (bear_fc, 8),
        (show_zero_crosses & bull_zero_cross, 9), (show_zero_crosses & bear_zero_cross, 10),
        (show_signal_crosses & bull_signal_cross, 11), (show_signal_crosses & bear_signal_cross, 12),
    ]:
        code = np.where(mask & (code == 0), c, code)

    # ---- research diagnostics ----
    corr_orig_tv = correlation(kvo_original, kvo_tv, correlation_len).to_numpy()
    corr_orig_signed = correlation(kvo_original, kvo_signed, correlation_len).to_numpy()
    disagreement_den = np.maximum(
        ema(kvo_original.abs(), norm_len).to_numpy(dtype=float) + ema(kvo_tv.abs(), norm_len).to_numpy(dtype=float),
        norm_floor)
    formula_disagreement = np.abs(kvo_original.to_numpy(dtype=float) - kvo_tv.to_numpy(dtype=float)) / disagreement_den

    # ---- assemble output ----
    out = pd.DataFrame(index=idx)
    out["kvo_raw"] = kvo_raw_a
    out["signal_raw"] = signal_raw_a
    out["kvo"] = kvo_disp
    out["signal"] = signal_disp
    out["hist"] = hist_disp
    out["relative_kvo"] = relative_kvo
    out["kvo_slope"] = kvo_slope_a
    out["hist_delta_relative"] = hist_delta_rel_a
    out["vf"] = vf
    out["formula_disagreement"] = formula_disagreement
    out["corr_orig_tv"] = corr_orig_tv
    out["corr_orig_signed"] = corr_orig_signed
    out["regime_state"] = pd.Series(regime_state, index=idx).astype("Int64")
    out["bull_expansion"] = pd.Series(bull_exp.astype(int), index=idx).astype("Int64")
    out["bear_expansion"] = pd.Series(bear_exp.astype(int), index=idx).astype("Int64")
    out["compression"] = pd.Series(compression.astype(int), index=idx).astype("Int64")
    out["regular_bull_div"] = pd.Series(regular_bull.astype(int), index=idx).astype("Int64")
    out["regular_bear_div"] = pd.Series(regular_bear.astype(int), index=idx).astype("Int64")
    out["hidden_bull_div"] = pd.Series(hidden_bull.astype(int), index=idx).astype("Int64")
    out["hidden_bear_div"] = pd.Series(hidden_bear.astype(int), index=idx).astype("Int64")
    out["bull_signal_cross"] = pd.Series(bull_signal_cross.astype(int), index=idx).astype("Int64")
    out["bear_signal_cross"] = pd.Series(bear_signal_cross.astype(int), index=idx).astype("Int64")
    out["bull_zero_cross"] = pd.Series(bull_zero_cross.astype(int), index=idx).astype("Int64")
    out["bear_zero_cross"] = pd.Series(bear_zero_cross.astype(int), index=idx).astype("Int64")
    out["bull_flow_rejection"] = pd.Series(bull_rej.astype(int), index=idx).astype("Int64")
    out["bear_flow_rejection"] = pd.Series(bear_rej.astype(int), index=idx).astype("Int64")
    out["bull_flow_confirm"] = pd.Series(bull_fc.astype(int), index=idx).astype("Int64")
    out["bear_flow_confirm"] = pd.Series(bear_fc.astype(int), index=idx).astype("Int64")
    out["bear_div_quality"] = pd.Series(quality_bear_filled, index=idx).astype("Int64")
    out["bull_div_quality"] = pd.Series(quality_bull_filled, index=idx).astype("Int64")
    out["last_event_code"] = pd.Series(code, index=idx).astype("Int64")
    return out
