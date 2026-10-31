"""Structural Wave Field [WavesUnchained]  v1.0.2  —  catalog port.

Source: data/pine-collection/indicators/market_structure/
        structural_wave_field/structural_wave_field.pine

A chart-native market-structure engine: multi-horizon structural flow with
hysteresis, an internal wave engine (momentum/velocity/acceleration),
non-overlapping wave segmentation with metric memory, maturity/compression,
counter-pressure + structural integrity, a re-synchronization engine, an
Elliott-context sequence read, an exhaustion engine and a persistent phase
engine.

Ported computational core (per-bar state machine); display renderer, gradient
field, labels, dashboard and alerts are dropped.

Output is the numeric surface: structural direction/drive/strength/confidence,
internal/aligned wave, counter pressure, integrity, maturity, exhaustion,
compression, resync stage/quality, phase, segment polarity, wave-strength decay,
and the confirmed event flags.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import atr, boolean, choice, ema, integer, number, ohlcv, sma, stdev

_MINTICK = 1e-8


def _clamp(x, lo, hi):
    if math.isnan(x):
        return x
    return max(lo, min(hi, x))


def _safe_div(num, den, fallback=0.0):
    return num / den if abs(den) > 1e-10 else fallback


def _squash(x):
    return x / (1.0 + abs(x)) if not math.isnan(x) else x


def _decay(cur, ref, floor):
    return _clamp(1.0 - _safe_div(cur, max(ref, floor), 1.0), 0.0, 1.0)


def _median_or(arr, fallback):
    return float(np.median(arr)) if arr else fallback


def _recent_mean(arr, count, fallback):
    if count <= 0 or len(arr) < count:
        return fallback
    return float(sum(arr[-1 - o] for o in range(count))) / count


def _metric_for_pol(pols, metric, target, nth, fallback):
    found = 0
    for j in range(len(pols) - 1, -1, -1):
        if pols[j] == target:
            if found == nth:
                return metric[j]
            found += 1
    return fallback


def _max_metric_for_pol(pols, metric, target, fallback):
    result = fallback
    for j in range(len(pols)):
        if pols[j] == target:
            v = metric[j]
            result = v if (isinstance(result, float) and math.isnan(result)) else max(result, v)
    return result


def _last_pattern5(pols):
    if len(pols) < 5:
        return False
    return (pols[-5] == 1 and pols[-4] == -1 and pols[-3] == 1 and pols[-2] == -1 and pols[-1] == 1)


def _last_pattern3_correction(pols):
    if len(pols) < 3:
        return False
    return pols[-3] == -1 and pols[-2] == 1 and pols[-1] == -1


class _EMA:
    """Pine ta.ema seeding (SMA of the first `length` values), update on scalar."""

    def __init__(self, length):
        self.length = length
        self.count = 0
        self.total = 0.0
        self.value = float("nan")

    def update(self, x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return self.value
        alpha = 2.0 / (self.length + 1.0)
        if self.count < self.length:
            self.count += 1
            self.total += x
            if self.count == self.length:
                self.value = self.total / self.length
        else:
            self.value = x * alpha + self.value * (1.0 - alpha)
        return self.value


def structural_wave_field(
    frame: pd.DataFrame,
    *,
    advanced_calibration: bool = False,
    sensitivity_mode: str = "Balanced",
    atr_len: int = 14,
    struct_base: int = 24,
    medium_ratio: float = 2.0,
    slow_ratio: float = 4.0,
    struct_entry: float = 0.34,
    struct_hold: float = 0.14,
    struct_flip: float = 0.42,
    disp_target: float = 3.0,
    wave_len: int = 10,
    wave_smooth: int = 5,
    velocity_lag: int = 2,
    accel_lag: int = 2,
    quality_len: int = 12,
    w_momentum: float = 0.55,
    w_velocity: float = 0.28,
    w_accel: float = 0.17,
    wave_start_threshold: float = 0.18,
    wave_end_threshold: float = 0.08,
    min_wave_bars: int = 2,
    wave_memory_size: int = 30,
    context_min_score: float = 0.44,
    leg_archive_size: int = 30,
    mat_age_w: float = 0.20,
    mat_ext_w: float = 0.25,
    mat_cycle_w: float = 0.15,
    mat_decay_w: float = 0.20,
    mat_eff_w: float = 0.10,
    mat_mismatch_w: float = 0.10,
    recovery_len: int = 8,
    resync_confirm: float = 0.16,
    resync_min_correction_bars: int = 3,
    resync_min_counter_peak: float = 0.24,
    resync_min_quality: float = 0.38,
    resync_cooldown_bars: int = 15,
    rs_counter_w: float = 0.30,
    rs_accel_w: float = 0.30,
    rs_recovery_w: float = 0.20,
    rs_integrity_w: float = 0.20,
    exhaustion_maturity_min: float = 0.58,
    failed_peak_ratio: float = 0.62,
    failed_disp_ratio: float = 0.70,
    participation_mode: str = "Relative Volume",
    volume_len: int = 20,
    participation_weight: float = 0.10,
    transition_memory_bars: int = 12,
    volume_is_real: bool | None = None,
) -> pd.DataFrame:
    sensitivity_mode = choice("sensitivity_mode", sensitivity_mode, ("Fast", "Balanced", "Structural"))
    participation_mode = choice("participation_mode", participation_mode, ("Off", "Volume", "Relative Volume", "Money Flow"))
    atr_len = integer("atr_len", atr_len, 5, 500)
    struct_base = integer("struct_base", struct_base, 8, 400)
    wave_len = integer("wave_len", wave_len, 3, 500)
    wave_smooth = integer("wave_smooth", wave_smooth, 1, 100)
    velocity_lag = integer("velocity_lag", velocity_lag, 1, 50)
    accel_lag = integer("accel_lag", accel_lag, 1, 50)
    quality_len = integer("quality_len", quality_len, 4, 500)
    min_wave_bars = integer("min_wave_bars", min_wave_bars, 1, 100)
    wave_memory_size = integer("wave_memory_size", wave_memory_size, 8, 100)
    leg_archive_size = integer("leg_archive_size", leg_archive_size, 8, 100)
    recovery_len = integer("recovery_len", recovery_len, 2, 500)
    resync_min_correction_bars = integer("resync_min_correction_bars", resync_min_correction_bars, 1, 50)
    resync_cooldown_bars = integer("resync_cooldown_bars", resync_cooldown_bars, 1, 100)
    volume_len = integer("volume_len", volume_len, 5, 500)
    transition_memory_bars = integer("transition_memory_bars", transition_memory_bars, 2, 100)
    medium_ratio = number("medium_ratio", medium_ratio, 1.2, 4.0)
    slow_ratio = number("slow_ratio", slow_ratio, 2.0, 5.0)
    struct_entry = number("struct_entry", struct_entry, 0.05, 0.90)
    struct_hold = number("struct_hold", struct_hold, 0.01, 0.80)
    struct_flip = number("struct_flip", struct_flip, 0.10, 0.95)
    disp_target = number("disp_target", disp_target, 0.5, 20.0)
    w_momentum = number("w_momentum", w_momentum, 0.0, 1.0)
    w_velocity = number("w_velocity", w_velocity, 0.0, 1.0)
    w_accel = number("w_accel", w_accel, 0.0, 1.0)
    wave_start_threshold = number("wave_start_threshold", wave_start_threshold, 0.03, 0.80)
    wave_end_threshold = number("wave_end_threshold", wave_end_threshold, 0.01, 0.60)
    context_min_score = number("context_min_score", context_min_score, 0.20, 0.90)
    resync_confirm = number("resync_confirm", resync_confirm, 0.03, 0.70)
    resync_min_counter_peak = number("resync_min_counter_peak", resync_min_counter_peak, 0.05, 0.80)
    resync_min_quality = number("resync_min_quality", resync_min_quality, 0.10, 0.90)
    exhaustion_maturity_min = number("exhaustion_maturity_min", exhaustion_maturity_min, 0.2, 0.95)
    failed_peak_ratio = number("failed_peak_ratio", failed_peak_ratio, 0.2, 0.95)
    failed_disp_ratio = number("failed_disp_ratio", failed_disp_ratio, 0.2, 1.2)
    participation_weight = number("participation_weight", participation_weight, 0.0, 0.30)

    # ---- effective preset parameters ----
    e_struct_base = struct_base if advanced_calibration else (16 if sensitivity_mode == "Fast" else 36 if sensitivity_mode == "Structural" else 24)
    e_medium = medium_ratio if advanced_calibration else (1.75 if sensitivity_mode == "Fast" else 2.25 if sensitivity_mode == "Structural" else 2.0)
    e_slow = slow_ratio if advanced_calibration else (3.0 if sensitivity_mode == "Fast" else 4.5 if sensitivity_mode == "Structural" else 4.0)
    e_entry = struct_entry if advanced_calibration else (0.30 if sensitivity_mode == "Fast" else 0.38 if sensitivity_mode == "Structural" else 0.34)
    e_hold = struct_hold if advanced_calibration else (0.11 if sensitivity_mode == "Fast" else 0.17 if sensitivity_mode == "Structural" else 0.14)
    e_flip = struct_flip if advanced_calibration else (0.38 if sensitivity_mode == "Fast" else 0.48 if sensitivity_mode == "Structural" else 0.42)
    e_disp = disp_target if advanced_calibration else (2.5 if sensitivity_mode == "Fast" else 3.5 if sensitivity_mode == "Structural" else 3.0)
    e_wave_len = wave_len if advanced_calibration else (7 if sensitivity_mode == "Fast" else 14 if sensitivity_mode == "Structural" else 10)
    e_wave_smooth = wave_smooth if advanced_calibration else (3 if sensitivity_mode == "Fast" else 7 if sensitivity_mode == "Structural" else 5)
    e_vel_lag = velocity_lag if advanced_calibration else (1 if sensitivity_mode == "Fast" else 3 if sensitivity_mode == "Structural" else 2)
    e_acc_lag = accel_lag if advanced_calibration else (1 if sensitivity_mode == "Fast" else 3 if sensitivity_mode == "Structural" else 2)
    e_quality_len = quality_len if advanced_calibration else (9 if sensitivity_mode == "Fast" else 18 if sensitivity_mode == "Structural" else 12)
    e_wave_start = wave_start_threshold if advanced_calibration else (0.14 if sensitivity_mode == "Fast" else 0.22 if sensitivity_mode == "Structural" else 0.18)
    e_wave_end = wave_end_threshold if advanced_calibration else (0.06 if sensitivity_mode == "Fast" else 0.10 if sensitivity_mode == "Structural" else 0.08)
    e_min_wave_bars = min_wave_bars if advanced_calibration else (1 if sensitivity_mode == "Fast" else 3 if sensitivity_mode == "Structural" else 2)
    e_recovery_len = recovery_len if advanced_calibration else (6 if sensitivity_mode == "Fast" else 12 if sensitivity_mode == "Structural" else 8)
    e_context_min = context_min_score if advanced_calibration else (0.38 if sensitivity_mode == "Fast" else 0.50 if sensitivity_mode == "Structural" else 0.44)
    e_resync_min_bars = resync_min_correction_bars if advanced_calibration else (2 if sensitivity_mode == "Fast" else 4 if sensitivity_mode == "Structural" else 3)
    e_resync_min_peak = resync_min_counter_peak if advanced_calibration else (0.19 if sensitivity_mode == "Fast" else 0.29 if sensitivity_mode == "Structural" else 0.24)
    e_resync_min_retr = (0.07 if sensitivity_mode == "Fast" else 0.14 if sensitivity_mode == "Structural" else 0.10) if not advanced_calibration else 0.10
    e_resync_min_quality = resync_min_quality if advanced_calibration else (0.32 if sensitivity_mode == "Fast" else 0.45 if sensitivity_mode == "Structural" else 0.38)
    e_resync_cooldown = resync_cooldown_bars if advanced_calibration else int(round(e_wave_len * (1.0 if sensitivity_mode == "Fast" else 1.8 if sensitivity_mode == "Structural" else 1.5)))

    if not (e_medium < e_slow):
        raise ValueError("medium_ratio 必须小于 slow_ratio")
    if not (e_hold < e_entry):
        raise ValueError("struct_hold 必须小于 struct_entry")
    if not (e_flip > e_entry):
        raise ValueError("struct_flip 必须大于 struct_entry")
    if not (e_wave_end < e_wave_start):
        raise ValueError("wave_end_threshold 必须小于 wave_start_threshold")

    o = frame["open"].to_numpy(dtype=float)
    h = frame["high"].to_numpy(dtype=float)
    l = frame["low"].to_numpy(dtype=float)
    c = frame["close"].to_numpy(dtype=float)
    vol = frame["volume"].to_numpy(dtype=float)
    idx = frame.index
    n = len(frame)

    atr_a = np.maximum(atr(frame, atr_len).to_numpy(dtype=float), _MINTICK)

    cs = pd.Series(c, index=idx)
    os_ = pd.Series(o, index=idx)
    chg = cs.diff().fillna(0.0).to_numpy(dtype=float)
    abs_chg = np.abs(chg)
    pos_chg = np.maximum(chg, 0.0)
    neg_chg = np.maximum(-chg, 0.0)

    def _sums(length):
        a = pd.Series(abs_chg).rolling(length, min_periods=length).sum().to_numpy()
        p = pd.Series(pos_chg).rolling(length, min_periods=length).sum().to_numpy()
        ng = pd.Series(neg_chg).rolling(length, min_periods=length).sum().to_numpy()
        return a, p, ng

    len_fast = e_struct_base
    len_medium = int(round(e_struct_base * e_medium))
    len_slow = int(round(e_struct_base * e_slow))

    hlens = {}
    for L in {len_fast, len_medium, len_slow}:
        hlens[L] = (ema(cs, L).to_numpy(), *_sums(L))
    q_abs, q_pos, q_neg = _sums(e_quality_len)
    noise_a = pd.Series(chg).rolling(e_quality_len, min_periods=e_quality_len).std(ddof=0).to_numpy()

    def horizon(i, L):
        if i < L:
            return float("nan")
        ema_arr, abs_sum, pos_sum, neg_sum = hlens[L]
        d = c[i] - c[i - L]
        path = abs_sum[i]
        if math.isnan(path):
            return float("nan")
        eff = _safe_div(abs(d), path, 0.0)
        signed_eff = math.copysign(eff, d) if d != 0 else 0.0
        disp = _squash(_safe_div(d, atr_a[i] * e_disp, 0.0))
        direction = 0 if d == 0 else (1 if d > 0 else -1)
        directional = (pos_sum[i] if direction > 0 else neg_sum[i] if direction < 0 else 0.0)
        if math.isnan(directional):
            directional = 0.0
        persistence = _safe_div(directional, path, 0.0) * direction
        location = _squash(_safe_div(c[i] - ema_arr[i], atr_a[i], 0.0))
        return 0.35 * disp + 0.30 * signed_eff + 0.20 * persistence + 0.15 * location

    # ---- internal wave (precomputed) ----
    price_momentum = np.array([_squash(_safe_div(c[i] - c[i - e_wave_len], atr_a[i] * 1.75, 0.0))
                               if i >= e_wave_len else float("nan") for i in range(n)])
    ema_wave = ema(cs, e_wave_len).to_numpy()
    ema_slope = np.full(n, np.nan)
    for i in range(1, n):
        if not math.isnan(ema_wave[i]) and not math.isnan(ema_wave[i - 1]):
            ema_slope[i] = _squash(_safe_div((ema_wave[i] - ema_wave[i - 1]) * e_wave_len, atr_a[i] * 1.25, 0.0))
    k_imp = max(2, int(round(e_wave_len * 0.5)))
    candle_imp_base = ema(cs - os_, k_imp).to_numpy()
    candle_impulse = np.array([_squash(_safe_div(candle_imp_base[i], atr_a[i] * 0.65, 0.0)) for i in range(n)])
    raw_momentum = np.array([0.52 * np.nan_to_num(price_momentum[i]) + 0.30 * np.nan_to_num(ema_slope[i])
                             + 0.18 * np.nan_to_num(candle_impulse[i]) for i in range(n)])
    momentum_core = ema(pd.Series(raw_momentum), e_wave_smooth).to_numpy()
    velocity = np.full(n, np.nan)
    for i in range(e_vel_lag, n):
        velocity[i] = _squash((momentum_core[i] - momentum_core[i - e_vel_lag]) * 3.0)
    acceleration = np.full(n, np.nan)
    for i in range(e_acc_lag, n):
        acceleration[i] = _squash((velocity[i] - velocity[i - e_acc_lag]) * 3.0)
    wsum = max(w_momentum + w_velocity + w_accel, 0.001)
    internal_wave = np.array([_clamp((w_momentum * momentum_core[i] + w_velocity * velocity[i]
                                      + w_accel * acceleration[i]) / wsum, -1.0, 1.0)
                              if not (math.isnan(momentum_core[i]) or math.isnan(velocity[i]) or math.isnan(acceleration[i]))
                              else float("nan") for i in range(n)])
    wave_energy_a = np.abs(internal_wave)

    vol_base = sma(pd.Series(vol), volume_len).to_numpy()
    vol_highest = pd.Series(vol).rolling(volume_len, min_periods=volume_len).max().to_numpy()
    rel_vol_a = np.array([_safe_div(vol[i], vol_base[i], 1.0) for i in range(n)])
    real = volume_is_real
    if real is None:
        real = bool(np.any(vol > 0))
    real = bool(real)
    participation_active = participation_mode != "Off" and real

    # ---- state ----
    struct_dir = 0
    leg_start_bar = None
    leg_start_price = float("nan")
    leg_extreme = float("nan")
    leg_atr_sum = 0.0
    leg_atr_bars = 0
    leg_aligned = 0
    leg_correction = 0
    leg_dur: list = []
    leg_disp: list = []
    leg_cycles: list = []

    seg_polarity = 0
    seg_start_bar = None
    seg_start_price = float("nan")
    seg_peak_energy = 0.0
    seg_energy_integral = 0.0
    seg_quality_sum = 0.0
    seg_eff_sum = 0.0
    seg_part_sum = 0.0
    seg_max_retr = 0.0
    seg_max_dir_move = 0.0
    seg_last_atr = float("nan")
    leg_segment_serial = 0

    wave_pol: list = []
    wave_dur: list = []
    wave_disp: list = []
    wave_peak: list = []
    wave_integral: list = []
    wave_quality_a: list = []
    wave_eff_a: list = []
    wave_part_a: list = []
    wave_retr_a: list = []

    # resync state
    correction_seen = False
    resync_stage = 0
    resync_hold = 99
    corr_ctx_bars = 0
    corr_ctx_start = None
    corr_peak_pressure = 0.0
    corr_peak_retr = 0.0
    last_conf_resync_q = 0.0
    last_resync_bar = None
    pending_serial = None
    consumed_serial = None
    pending_end_bar = None
    pending_bars = 0
    pending_counter_peak = 0.0
    pending_retr = 0.0

    # transition memory
    last_leg_dir = 0
    last_leg_exit_bar = None
    last_leg_maturity = 0.0
    last_leg_exhaustion = 0.0
    last_leg_strength = 0.0
    last_leg_conf = 0.0
    last_leg_aligned = 0.0
    last_leg_counter = 0.0
    last_leg_wq = 0.0
    last_leg_deterioration = 0.0
    last_leg_context = "Unclassified"

    # phase
    phase = 0
    pending_phase = 0
    pending_phase_bars = 0

    # ema states
    part_ema = _EMA(5)
    qeff_ema = _EMA(5)
    ppe_ema = _EMA(5)
    mf_ema = _EMA(volume_len)
    ema_volume = ema(pd.Series(vol), volume_len).to_numpy()
    fast_recovery = ema(cs, e_recovery_len).to_numpy()

    # previous values
    prev = {"struct_strength": float("nan"), "wave_energy": float("nan"), "weakening": False,
            "exhaustion": float("nan"), "phase": 0}
    cp_hist: list = []

    out = {k: np.full(n, np.nan) for k in [
        "struct_dir", "structural_drive", "structural_strength", "structural_confidence",
        "internal_wave", "aligned_wave", "wave_energy", "wave_quality", "counter_pressure",
        "structural_integrity", "retracement", "maturity", "exhaustion", "compression",
        "resync_stage", "resync_quality", "phase", "phase_confidence", "seg_polarity",
        "wave_strength_decay", "leg_bars", "participation"]}
    flags = {k: np.zeros(n, dtype=np.int64) for k in [
        "structural_flip", "structural_initiated", "structural_exit", "resync_confirmed",
        "failed_expansion", "weakening_sequence", "exhaustion_event",
        "impulse_context", "correction_context"]}

    for i in range(n):
        atr_i = atr_a[i]
        h_fast, h_med, h_slow = horizon(i, len_fast), horizon(i, len_medium), horizon(i, len_slow)
        if math.isnan(h_fast):
            h_fast = 0.0
        if math.isnan(h_med):
            h_med = 0.0
        if math.isnan(h_slow):
            h_slow = 0.0
        raw_drive = 0.20 * h_fast + 0.35 * h_med + 0.45 * h_slow
        s_fast = 1 if h_fast > 0.05 else -1 if h_fast < -0.05 else 0
        s_med = 1 if h_med > 0.05 else -1 if h_med < -0.05 else 0
        s_slow = 1 if h_slow > 0.05 else -1 if h_slow < -0.05 else 0
        sign_agree = 1.0 if (s_fast == s_med and s_med == s_slow) else (0.68 if (s_fast == s_med or s_fast == s_slow or s_med == s_slow) else 0.34)
        spread = (abs(h_fast - h_med) + abs(h_med - h_slow) + abs(h_fast - h_slow)) / 3.0
        structural_conf = _clamp(sign_agree * (1.0 - 0.65 * spread), 0.0, 1.0)
        structural_drive = raw_drive * (0.72 + 0.28 * sign_agree)
        structural_strength = _clamp(abs(structural_drive), 0.0, 1.0)

        prev_dir = struct_dir
        if struct_dir == 0:
            if structural_drive > e_entry:
                struct_dir = 1
            elif structural_drive < -e_entry:
                struct_dir = -1
        elif struct_dir == 1:
            if structural_drive < -e_flip:
                struct_dir = -1
            elif structural_drive < e_hold:
                struct_dir = 0
        else:
            if structural_drive > e_flip:
                struct_dir = 1
            elif structural_drive > -e_hold:
                struct_dir = 0

        structural_flip = prev_dir != 0 and struct_dir != 0 and struct_dir != prev_dir
        structural_exit = prev_dir != 0 and struct_dir == 0
        structural_initiated = prev_dir == 0 and struct_dir != 0
        structural_reset = structural_flip or structural_exit

        if structural_initiated or (struct_dir != 0 and leg_start_bar is None):
            leg_start_bar = i
            leg_start_price = c[i]
            leg_extreme = c[i]
            leg_atr_sum = 0.0
            leg_atr_bars = 0
            leg_aligned = 0
            leg_correction = 0

        if structural_reset and leg_start_bar is not None and not math.isnan(leg_start_price) and not math.isnan(leg_extreme):
            old_dur = max(1.0, i - leg_start_bar)
            old_atr = _safe_div(leg_atr_sum, max(float(leg_atr_bars), 1.0), atr_i)
            old_disp = abs(leg_extreme - leg_start_price) / max(old_atr, _MINTICK)
            leg_dur.append(old_dur); leg_disp.append(old_disp)
            leg_cycles.append(max(1.0, leg_aligned + 0.5 * leg_correction))
            if len(leg_dur) > leg_archive_size:
                leg_dur.pop(0); leg_disp.pop(0); leg_cycles.pop(0)
            leg_start_bar = i if structural_flip else None
            leg_start_price = c[i] if structural_flip else float("nan")
            leg_extreme = c[i] if structural_flip else float("nan")
            leg_atr_sum = 0.0
            leg_atr_bars = 0
            leg_aligned = 0
            leg_correction = 0

        if struct_dir == 1:
            leg_extreme = h[i] if math.isnan(leg_extreme) else max(leg_extreme, h[i])
        elif struct_dir == -1:
            leg_extreme = l[i] if math.isnan(leg_extreme) else min(leg_extreme, l[i])

        if struct_dir != 0 and leg_start_bar is not None:
            leg_atr_sum += atr_i
            leg_atr_bars += 1

        leg_range = abs(leg_extreme - leg_start_price) if not (math.isnan(leg_extreme) or math.isnan(leg_start_price)) else float("nan")
        if struct_dir == 1:
            retracement = _safe_div(leg_extreme - c[i], max(leg_range, atr_i * 0.25), 0.0)
        elif struct_dir == -1:
            retracement = _safe_div(c[i] - leg_extreme, max(leg_range, atr_i * 0.25), 0.0)
        else:
            retracement = 0.0
        retracement = _clamp(retracement, 0.0, 1.5)

        iw = internal_wave[i]
        aligned_wave = 0.0 if struct_dir == 0 else (iw * struct_dir if not math.isnan(iw) else 0.0)
        wave_energy = wave_energy_a[i] if not math.isnan(wave_energy_a[i]) else 0.0
        acc_i = acceleration[i] if not math.isnan(acceleration[i]) else 0.0
        aligned_accel = acc_i * struct_dir

        # ---- wave quality ----
        if i >= e_quality_len and not math.isnan(q_abs[i]):
            q_net = abs(c[i] - c[i - e_quality_len])
            q_eff = _clamp(_safe_div(q_net, q_abs[i], 0.0), 0.0, 1.0)
            sm = chg[i] * struct_dir
            a_path = (q_pos[i] if struct_dir > 0 else q_neg[i] if struct_dir < 0 else 0.0)
            c_path = (q_neg[i] if struct_dir > 0 else q_pos[i] if struct_dir < 0 else q_abs[i])
            if math.isnan(a_path):
                a_path = 0.0
            if math.isnan(c_path):
                c_path = 0.0
            path_total = max(a_path + c_path, _MINTICK)
            q_persist = _clamp(max(a_path, c_path) / path_total, 0.0, 1.0)
            noise = noise_a[i]
            q_smooth = _clamp(1.0 - _safe_div(noise, atr_i * 0.90, 1.0), 0.0, 1.0) if not math.isnan(noise) else 0.0
            wave_quality = _clamp(0.45 * q_eff + 0.30 * q_persist + 0.25 * q_smooth, 0.0, 1.0)
        else:
            q_eff = 0.0
            a_path = c_path = 0.0
            wave_quality = 0.0
        counter_energy = max(-aligned_wave, 0.0)
        counter_persist = _clamp(c_path / max(a_path + c_path, _MINTICK), 0.0, 1.0)
        counter_pressure = _clamp(counter_energy * (0.42 + 0.22 * wave_quality + 0.18 * _clamp(retracement, 0.0, 1.0) + 0.18 * counter_persist), 0.0, 1.0)
        structural_stress = _clamp(max(-structural_drive * struct_dir, 0.0) * 2.0, 0.0, 1.0)
        structural_integrity = _clamp(1.0 - (0.52 * counter_pressure + 0.30 * _clamp(retracement, 0.0, 1.0) + 0.18 * structural_stress), 0.0, 1.0)

        # ---- participation ----
        if participation_mode == "Volume":
            participation = _clamp(_safe_div(vol[i], vol_highest[i], 0.0), 0.0, 1.0) if (vol_base[i] > 0 and not math.isnan(vol_highest[i])) else 0.50
        elif participation_mode == "Relative Volume":
            participation = _clamp((rel_vol_a[i] - 0.50) / 1.50, 0.0, 1.0) if not math.isnan(rel_vol_a[i]) else 0.333
        elif participation_mode == "Money Flow":
            candle_range = max(h[i] - l[i], _MINTICK)
            smf = vol[i] * _safe_div(c[i] - o[i], candle_range, 0.0) * struct_dir
            mf = mf_ema.update(smf)
            money_flow_base = max(ema_volume[i], 1.0) if not math.isnan(ema_volume[i]) else 1.0
            participation = _clamp(0.5 + 0.5 * _squash(_safe_div(mf, money_flow_base, 0.0) * 3.0), 0.0, 1.0) if not math.isnan(mf) else 0.5
        else:
            participation = 0.50
        prev_part = part_ema.value
        cur_part = part_ema.update(participation)
        if participation_active and not math.isnan(prev_part) and not math.isnan(cur_part):
            participation_loss = _clamp(prev_part - cur_part, 0.0, 1.0)
        else:
            participation_loss = 0.0

        # ---- segmentation ----
        desired = 1 if aligned_wave >= e_wave_start else (-1 if aligned_wave <= -e_wave_start else 0)
        if structural_reset:
            seg_polarity = 0
            seg_start_bar = None
            seg_start_price = float("nan")
            seg_peak_energy = 0.0
            seg_energy_integral = 0.0
            seg_quality_sum = 0.0
            seg_eff_sum = 0.0
            seg_part_sum = 0.0
            seg_max_retr = 0.0
            seg_max_dir_move = 0.0
            seg_last_atr = float("nan")
            wave_pol.clear(); wave_dur.clear(); wave_disp.clear(); wave_peak.clear()
            wave_integral.clear(); wave_quality_a.clear(); wave_eff_a.clear()
            wave_part_a.clear(); wave_retr_a.clear()

        segment_ended = False
        ended_pol = 0
        ended_peak = ended_disp = ended_integral = ended_quality = ended_eff = ended_part = ended_retr = float("nan")

        if seg_polarity == 0:
            if desired != 0:
                seg_polarity = desired
                seg_start_bar = i
                seg_start_price = c[i]
                seg_peak_energy = wave_energy
                seg_energy_integral = wave_energy
                seg_quality_sum = wave_quality
                seg_eff_sum = q_eff
                seg_part_sum = participation
                seg_max_retr = retracement
                seg_max_dir_move = 0.0
                seg_last_atr = atr_i
        else:
            opposite_confirmed = desired == -seg_polarity
            energy_died = wave_energy <= e_wave_end
            if opposite_confirmed or energy_died:
                dur = max(1, i - seg_start_bar)
                if dur >= e_min_wave_bars:
                    segment_ended = True
                    ended_pol = seg_polarity
                    ended_peak = seg_peak_energy
                    ended_disp = seg_max_dir_move / max(seg_last_atr if not math.isnan(seg_last_atr) else atr_i, _MINTICK)
                    ended_integral = seg_energy_integral
                    ended_quality = _safe_div(seg_quality_sum, dur, wave_quality)
                    ended_eff = _safe_div(seg_eff_sum, dur, q_eff)
                    ended_part = _safe_div(seg_part_sum, dur, participation)
                    ended_retr = seg_max_retr
                    for arr, val in [(wave_pol, ended_pol), (wave_dur, dur), (wave_disp, ended_disp),
                                     (wave_peak, ended_peak), (wave_integral, ended_integral),
                                     (wave_quality_a, ended_quality), (wave_eff_a, ended_eff),
                                     (wave_part_a, ended_part), (wave_retr_a, ended_retr)]:
                        arr.append(val)
                        if len(arr) > wave_memory_size:
                            arr.pop(0)
                if opposite_confirmed:
                    seg_polarity = desired
                    seg_start_bar = i
                    seg_start_price = c[i]
                    seg_peak_energy = wave_energy
                    seg_energy_integral = wave_energy
                    seg_quality_sum = wave_quality
                    seg_eff_sum = q_eff
                    seg_part_sum = participation
                    seg_max_retr = retracement
                    seg_max_dir_move = 0.0
                    seg_last_atr = atr_i
                else:
                    seg_polarity = 0
                    seg_start_bar = None
                    seg_start_price = float("nan")
                    seg_peak_energy = seg_energy_integral = seg_quality_sum = seg_eff_sum = seg_part_sum = 0.0
                    seg_max_retr = seg_max_dir_move = 0.0
                    seg_last_atr = float("nan")
            else:
                seg_dir = struct_dir * seg_polarity
                seg_peak_energy = max(seg_peak_energy, wave_energy)
                seg_energy_integral += wave_energy
                seg_quality_sum += wave_quality
                seg_eff_sum += q_eff
                seg_part_sum += participation
                seg_max_retr = max(seg_max_retr, retracement)
                seg_max_dir_move = max(seg_max_dir_move, max((c[i] - seg_start_price) * seg_dir, 0.0))
                seg_last_atr = atr_i

        if segment_ended:
            if ended_pol == 1:
                leg_aligned += 1
            elif ended_pol == -1:
                leg_correction += 1

        if structural_reset:
            leg_segment_serial = 0
        elif segment_ended:
            leg_segment_serial += 1

        # ---- wave memory derived metrics ----
        last_aligned_peak = _metric_for_pol(wave_pol, wave_peak, 1, 0, float("nan"))
        prev_aligned_peak = _metric_for_pol(wave_pol, wave_peak, 1, 1, float("nan"))
        last_aligned_disp = _metric_for_pol(wave_pol, wave_disp, 1, 0, float("nan"))
        prev_aligned_disp = _metric_for_pol(wave_pol, wave_disp, 1, 1, float("nan"))
        last_aligned_integral = _metric_for_pol(wave_pol, wave_integral, 1, 0, float("nan"))
        prev_aligned_integral = _metric_for_pol(wave_pol, wave_integral, 1, 1, float("nan"))
        last_aligned_dur = _metric_for_pol(wave_pol, wave_dur, 1, 0, float("nan"))
        prev_aligned_dur = _metric_for_pol(wave_pol, wave_dur, 1, 1, float("nan"))
        last_aligned_quality = _metric_for_pol(wave_pol, wave_quality_a, 1, 0, float("nan"))
        prev_aligned_quality = _metric_for_pol(wave_pol, wave_quality_a, 1, 1, float("nan"))
        last_aligned_eff = _metric_for_pol(wave_pol, wave_eff_a, 1, 0, float("nan"))
        prev_aligned_eff = _metric_for_pol(wave_pol, wave_eff_a, 1, 1, float("nan"))
        last_aligned_part = _metric_for_pol(wave_pol, wave_part_a, 1, 0, float("nan"))
        prev_aligned_part = _metric_for_pol(wave_pol, wave_part_a, 1, 1, float("nan"))

        aligned_active = seg_polarity == 1
        live_bars = max(1.0, i - seg_start_bar + 1) if aligned_active else (last_aligned_dur if not math.isnan(last_aligned_dur) else 1.0)
        live_disp = (seg_max_dir_move / atr_i) if aligned_active else (last_aligned_disp if not math.isnan(last_aligned_disp) else 0.0)
        live_integral = seg_energy_integral if aligned_active else (last_aligned_integral if not math.isnan(last_aligned_integral) else 0.0)
        live_quality = _safe_div(seg_quality_sum, live_bars, wave_quality) if aligned_active else (last_aligned_quality if not math.isnan(last_aligned_quality) else wave_quality)
        live_eff = _safe_div(seg_eff_sum, live_bars, q_eff) if aligned_active else (last_aligned_eff if not math.isnan(last_aligned_eff) else q_eff)
        live_part = _safe_div(seg_part_sum, live_bars, participation) if aligned_active else (last_aligned_part if not math.isnan(last_aligned_part) else participation)

        current_aligned_peak = seg_peak_energy if aligned_active else (last_aligned_peak if not math.isnan(last_aligned_peak) else max(aligned_wave, 0.0))
        ref_peak = (last_aligned_peak if not math.isnan(last_aligned_peak) else max(current_aligned_peak, 0.01)) if aligned_active else (prev_aligned_peak if not math.isnan(prev_aligned_peak) else (last_aligned_peak if not math.isnan(last_aligned_peak) else max(current_aligned_peak, 0.01)))
        ref_disp = (last_aligned_disp if not math.isnan(last_aligned_disp) else max(live_disp, 0.01)) if aligned_active else (prev_aligned_disp if not math.isnan(prev_aligned_disp) else (last_aligned_disp if not math.isnan(last_aligned_disp) else max(live_disp, 0.01)))
        ref_integral = (last_aligned_integral if not math.isnan(last_aligned_integral) else max(live_integral, 0.01)) if aligned_active else (prev_aligned_integral if not math.isnan(prev_aligned_integral) else (last_aligned_integral if not math.isnan(last_aligned_integral) else max(live_integral, 0.01)))
        ref_dur = (last_aligned_dur if not math.isnan(last_aligned_dur) else max(live_bars, 1.0)) if aligned_active else (prev_aligned_dur if not math.isnan(prev_aligned_dur) else (last_aligned_dur if not math.isnan(last_aligned_dur) else max(live_bars, 1.0)))
        ref_quality = (last_aligned_quality if not math.isnan(last_aligned_quality) else max(live_quality, 0.05)) if aligned_active else (prev_aligned_quality if not math.isnan(prev_aligned_quality) else (last_aligned_quality if not math.isnan(last_aligned_quality) else max(live_quality, 0.05)))
        ref_eff = (last_aligned_eff if not math.isnan(last_aligned_eff) else max(live_eff, 0.05)) if aligned_active else (prev_aligned_eff if not math.isnan(prev_aligned_eff) else (last_aligned_eff if not math.isnan(last_aligned_eff) else max(live_eff, 0.05)))
        ref_part = (last_aligned_part if not math.isnan(last_aligned_part) else max(live_part, 0.05)) if aligned_active else (prev_aligned_part if not math.isnan(prev_aligned_part) else (last_aligned_part if not math.isnan(last_aligned_part) else max(live_part, 0.05)))

        peak_decay = _decay(current_aligned_peak, ref_peak, 0.05)
        disp_decay = _decay(live_disp, ref_disp, 0.05)
        integral_decay = _decay(live_integral, ref_integral, 0.05)
        dur_decay = _decay(live_bars, ref_dur, 1.0)
        qual_decay = _decay(live_quality, ref_quality, 0.05)
        eff_decay = _decay(live_eff, ref_eff, 0.05)
        part_decay = _decay(live_part, ref_part, 0.05) if participation_active else 0.0
        mem_part_w = 0.05 if participation_active else 0.0
        decay_wsum = 0.95 + mem_part_w
        wave_strength_decay = _clamp((0.35 * peak_decay + 0.18 * disp_decay + 0.17 * integral_decay
                                      + 0.08 * dur_decay + 0.10 * qual_decay + 0.07 * eff_decay
                                      + mem_part_w * part_decay) / decay_wsum, 0.0, 1.0)

        prior_aligned_peak_before = _metric_for_pol(wave_pol, wave_peak, 1, 0, float("nan"))
        prior_aligned_disp_before = _metric_for_pol(wave_pol, wave_disp, 1, 0, float("nan"))
        failed_expansion = (segment_ended and ended_pol == 1 and not math.isnan(prior_aligned_peak_before)
                            and ended_peak < prior_aligned_peak_before * failed_peak_ratio
                            and (math.isnan(prior_aligned_disp_before) or ended_disp < prior_aligned_disp_before * failed_disp_ratio))

        # ---- maturity ----
        hist_leg_bars = _median_or(leg_dur, e_struct_base * 3.0)
        hist_leg_disp = _median_or(leg_disp, 4.0)
        hist_cycles = _median_or(leg_cycles, 3.0)
        cur_leg_bars = 0.0 if leg_start_bar is None else max(1.0, i - leg_start_bar + 1)
        cur_ext = 0.0 if struct_dir == 0 or math.isnan(leg_extreme) or math.isnan(leg_start_price) else _safe_div(max((leg_extreme - leg_start_price) * struct_dir, 0.0), atr_i, 0.0)
        cur_prog = 0.0 if struct_dir == 0 else _safe_div(max((c[i] - leg_start_price) * struct_dir, 0.0), atr_i, 0.0)
        cur_cycles = leg_aligned + 0.5 * leg_correction + (0.5 if seg_polarity == 1 else 0.25 if seg_polarity == -1 else 0.0)
        age_score = _clamp(_safe_div(cur_leg_bars, max(hist_leg_bars, 1.0), 0.0), 0.0, 1.0)
        ext_score = _clamp(_safe_div(cur_ext, max(hist_leg_disp, 0.5), 0.0), 0.0, 1.0)
        cycle_score = _clamp(_safe_div(cur_cycles, max(hist_cycles, 1.0), 0.0), 0.0, 1.0)
        prev_qeff = qeff_ema.value
        cur_qeff = qeff_ema.update(q_eff)
        eff_loss = _clamp((prev_qeff - cur_qeff) * 4.0, 0.0, 1.0) if (not math.isnan(prev_qeff) and not math.isnan(cur_qeff)) else 0.0
        price_prog = _clamp(_safe_div(cur_prog, max(hist_leg_disp, 0.5), 0.0), 0.0, 1.5)
        max_aligned_peak = _max_metric_for_pol(wave_pol, wave_peak, 1, current_aligned_peak)
        pressure_prog = _clamp(_safe_div(current_aligned_peak, max(max_aligned_peak, 0.05), 1.0), 0.0, 1.5)
        price_pressure_mismatch = _clamp(price_prog - pressure_prog, 0.0, 1.0)
        mat_w = max(mat_age_w + mat_ext_w + mat_cycle_w + mat_decay_w + mat_eff_w + mat_mismatch_w, 0.001)
        maturity = _clamp((mat_age_w * age_score + mat_ext_w * ext_score + mat_cycle_w * cycle_score
                           + mat_decay_w * wave_strength_decay + mat_eff_w * eff_loss
                           + mat_mismatch_w * price_pressure_mismatch) / mat_w, 0.0, 1.0)

        cur_seg_bars = max(1.0, i - seg_start_bar + 1) if (seg_polarity == -1 and seg_start_bar is not None) else 0.0
        corr_dur_norm = _clamp(_safe_div(cur_seg_bars, max(hist_leg_bars * 0.25, 2.0), 0.0), 0.0, 1.0)
        compression = _clamp(corr_dur_norm * structural_strength * (1.0 - _clamp(retracement, 0.0, 1.0)) * (1.0 - counter_pressure), 0.0, 1.0) if seg_polarity == -1 else 0.0

        sm = chg[i] * struct_dir
        ppe = _clamp(_safe_div(max(sm, 0.0) / atr_i, max(aligned_wave, 0.05), 0.0) / 2.0, 0.0, 1.0)
        prev_ppe = ppe_ema.value
        cur_ppe = ppe_ema.update(ppe)
        ppe_loss = _clamp(prev_ppe - cur_ppe, 0.0, 1.0) if (not math.isnan(prev_ppe) and not math.isnan(cur_ppe)) else 0.0

        # ---- re-synchronization engine ----
        if structural_reset:
            correction_seen = False
            resync_stage = 0
            resync_hold = 99
            corr_ctx_bars = 0
            corr_ctx_start = None
            corr_peak_pressure = corr_peak_retr = 0.0
            last_conf_resync_q = 0.0
            last_resync_bar = None
            pending_serial = consumed_serial = pending_end_bar = None
            pending_bars = 0
            pending_counter_peak = pending_retr = 0.0
        if not correction_seen:
            resync_stage = 0
        counter_ctx = aligned_wave <= -e_wave_end or counter_pressure > 0.18
        new_counter_after_pending = counter_ctx and pending_serial is not None and leg_segment_serial > pending_serial
        if new_counter_after_pending:
            correction_seen = False
            corr_ctx_bars = 0
            corr_ctx_start = None
            corr_peak_pressure = corr_peak_retr = 0.0
            pending_serial = None
            pending_end_bar = None
        if counter_ctx:
            if not correction_seen:
                resync_hold = 99
                corr_ctx_bars = 0
                corr_ctx_start = i
                corr_peak_pressure = corr_peak_retr = 0.0
            correction_seen = True
            corr_ctx_bars += 1
            corr_peak_pressure = max(corr_peak_pressure, counter_pressure)
            corr_peak_retr = max(corr_peak_retr, retracement)

        resync_max_ctx = e_wave_len * 3
        corr_ctx_fresh = correction_seen and corr_ctx_start is not None and (counter_ctx or i - corr_ctx_start <= resync_max_ctx)
        if correction_seen and not corr_ctx_fresh:
            correction_seen = False
            corr_ctx_bars = 0
            corr_ctx_start = None
            corr_peak_pressure = corr_peak_retr = 0.0

        if segment_ended and ended_pol == -1:
            qualified = corr_ctx_bars >= e_resync_min_bars and (corr_peak_pressure >= e_resync_min_peak or corr_peak_retr >= e_resync_min_retr)
            if qualified:
                pending_serial = leg_segment_serial
                pending_end_bar = i
                pending_bars = corr_ctx_bars
                pending_counter_peak = corr_peak_pressure
                pending_retr = corr_peak_retr
            else:
                correction_seen = False
                corr_ctx_bars = 0
                corr_ctx_start = None
                corr_peak_pressure = corr_peak_retr = 0.0

        resync_recovery_window = e_wave_len * 2
        pending_fresh = pending_serial is not None and pending_end_bar is not None and i - pending_end_bar <= resync_recovery_window
        if pending_serial is not None and not pending_fresh:
            pending_serial = None
            pending_end_bar = None
            correction_seen = False
            corr_ctx_bars = 0
            corr_ctx_start = None
            corr_peak_pressure = corr_peak_retr = 0.0

        cp_lag = cp_hist[-1 - e_vel_lag] if len(cp_hist) > e_vel_lag else float("nan")
        counter_decay = _clamp((cp_lag - counter_pressure) * 3.5, 0.0, 1.0) if not math.isnan(cp_lag) else 0.0
        accel_trend = _clamp(aligned_accel * 1.8, 0.0, 1.0)
        aligned_velocity = (velocity[i] if not math.isnan(velocity[i]) else 0.0) * struct_dir
        velocity_recovery = _clamp(aligned_velocity * 2.0, 0.0, 1.0)
        fast_recovery_ema = fast_recovery[i]
        if struct_dir == 1:
            recovery_location = 1.0 if c[i] > fast_recovery_ema else 0.0
        elif struct_dir == -1:
            recovery_location = 1.0 if c[i] < fast_recovery_ema else 0.0
        else:
            recovery_location = 0.0
        recovery_disp = 0.0 if struct_dir == 0 else _clamp(_squash(_safe_div((c[i] - c[i - e_recovery_len]) * struct_dir, atr_i, 0.0)) * 1.5, 0.0, 1.0) if i >= e_recovery_len else 0.0
        recovery_score = 0.45 * recovery_location + 0.55 * recovery_disp
        recovery_evidence = 0.55 * recovery_score + 0.45 * velocity_recovery
        rs_wsum = max(rs_counter_w + rs_accel_w + rs_recovery_w + rs_integrity_w, 0.001)
        resync_raw = ((rs_counter_w * counter_decay + rs_accel_w * accel_trend + rs_recovery_w * recovery_evidence + rs_integrity_w * structural_integrity) / rs_wsum) if correction_seen else 0.0
        resync_quality = _clamp(resync_raw * (0.72 + 0.28 * structural_strength) * (1.0 + 0.20 * compression), 0.0, 1.0)

        resync_owned = pending_fresh and (consumed_serial is None or pending_serial != consumed_serial)
        cooldown_ready = last_resync_bar is None or i - last_resync_bar >= e_resync_cooldown
        resync_min_structure = 0.22 if advanced_calibration else (0.18 if sensitivity_mode == "Fast" else 0.28 if sensitivity_mode == "Structural" else 0.22)
        resync_min_conf = 0.35 if advanced_calibration else (0.28 if sensitivity_mode == "Fast" else 0.42 if sensitivity_mode == "Structural" else 0.35)
        resync_confirmed = (not structural_reset and resync_owned and cooldown_ready and aligned_wave > resync_confirm
                            and velocity_recovery > 0.08 and structural_integrity > 0.50
                            and structural_strength > resync_min_structure and structural_conf > resync_min_conf
                            and resync_quality >= e_resync_min_quality)
        confirmed_bars = pending_bars
        confirmed_counter_peak = pending_counter_peak
        confirmed_retr = pending_retr
        if not correction_seen:
            resync_stage = 0
        else:
            detected = 1 if counter_decay > 0.20 else 0
            ctx_developed = corr_ctx_bars >= max(1, e_resync_min_bars - 1) and (corr_peak_pressure >= e_resync_min_peak * 0.75 or corr_peak_retr >= e_resync_min_retr * 0.75)
            if not ctx_developed:
                detected = 0
            if ctx_developed and abs(aligned_wave) < e_wave_end and counter_decay > 0.15:
                detected = 2
            if ctx_developed and aligned_accel > 0.08 and (velocity_recovery > 0.05 or recovery_score > 0.25):
                detected = 3
            if resync_confirmed:
                detected = 4
            resync_stage = max(resync_stage, detected)

        if resync_confirmed:
            resync_hold = 0
            last_conf_resync_q = resync_quality
            last_resync_bar = i
            consumed_serial = pending_serial
            pending_serial = None
            pending_end_bar = None
            correction_seen = False
            corr_ctx_bars = 0
            corr_ctx_start = None
            corr_peak_pressure = corr_peak_retr = 0.0
        elif not structural_reset:
            resync_hold = min(resync_hold + 1, 99)
        resync_glow = 1.0 - resync_hold / 6.0 if resync_hold <= 5 else 0.0

        # ---- elliott context ----
        impulse_pattern = _last_pattern5(wave_pol)
        corr_pattern = _last_pattern3_correction(wave_pol)
        p1 = _metric_for_pol(wave_pol, wave_peak, 1, 2, float("nan"))
        p2 = _metric_for_pol(wave_pol, wave_peak, 1, 1, float("nan"))
        p3 = _metric_for_pol(wave_pol, wave_peak, 1, 0, float("nan"))
        d1 = _metric_for_pol(wave_pol, wave_disp, 1, 2, float("nan"))
        d2 = _metric_for_pol(wave_pol, wave_disp, 1, 1, float("nan"))
        d3 = _metric_for_pol(wave_pol, wave_disp, 1, 0, float("nan"))
        i1 = _metric_for_pol(wave_pol, wave_integral, 1, 2, float("nan"))
        i2 = _metric_for_pol(wave_pol, wave_integral, 1, 1, float("nan"))
        i3 = _metric_for_pol(wave_pol, wave_integral, 1, 0, float("nan"))
        q1 = _metric_for_pol(wave_pol, wave_quality_a, 1, 2, float("nan"))
        q2 = _metric_for_pol(wave_pol, wave_quality_a, 1, 1, float("nan"))
        q3 = _metric_for_pol(wave_pol, wave_quality_a, 1, 0, float("nan"))
        e1 = _metric_for_pol(wave_pol, wave_eff_a, 1, 2, float("nan"))
        e2 = _metric_for_pol(wave_pol, wave_eff_a, 1, 1, float("nan"))
        e3 = _metric_for_pol(wave_pol, wave_eff_a, 1, 0, float("nan"))
        v1 = _metric_for_pol(wave_pol, wave_part_a, 1, 2, float("nan"))
        v2 = _metric_for_pol(wave_pol, wave_part_a, 1, 1, float("nan"))
        v3 = _metric_for_pol(wave_pol, wave_part_a, 1, 0, float("nan"))
        last_corr_retr = _metric_for_pol(wave_pol, wave_retr_a, -1, 0, float("nan"))
        has3 = not (math.isnan(p1) or math.isnan(p2) or math.isnan(p3))
        expansion_hierarchy = _clamp((p2 / max(p1, 0.05) - 0.8) * 0.8 + (1.0 - p3 / max(p2, 0.05)) * 0.6, 0.0, 1.0) if has3 else 0.0
        has_d3 = not (math.isnan(d1) or math.isnan(d2) or math.isnan(d3))
        displacement_hierarchy = _clamp((d2 / max(d1, 0.05) - 0.8) * 0.8 + (1.0 - d3 / max(d2, 0.05)) * 0.6, 0.0, 1.0) if has_d3 else expansion_hierarchy
        has_i3 = not (math.isnan(i1) or math.isnan(i2) or math.isnan(i3))
        integral_hierarchy = _clamp((i2 / max(i1, 0.05) - 0.8) * 0.8 + (1.0 - i3 / max(i2, 0.05)) * 0.6, 0.0, 1.0) if has_i3 else expansion_hierarchy
        sequence_hierarchy = _clamp(0.50 * expansion_hierarchy + 0.25 * displacement_hierarchy + 0.25 * integral_hierarchy, 0.0, 1.0)
        quality_seq_mean = (q1 + q2 + q3) / 3.0 if not (math.isnan(q1) or math.isnan(q2) or math.isnan(q3)) else wave_quality
        eff_seq_mean = (e1 + e2 + e3) / 3.0 if not (math.isnan(e1) or math.isnan(e2) or math.isnan(e3)) else q_eff
        part_seq_mean = (v1 + v2 + v3) / 3.0 if not (math.isnan(v1) or math.isnan(v2) or math.isnan(v3)) else participation
        seq_part_w = 0.10 if participation_active else 0.0
        sequence_quality = _clamp((0.72 * quality_seq_mean + 0.28 * eff_seq_mean + seq_part_w * part_seq_mean) / (1.0 + seq_part_w), 0.0, 1.0)
        correction_control = structural_integrity if math.isnan(last_corr_retr) else _clamp(1.0 - last_corr_retr, 0.0, 1.0)
        impulse_seq_score = _clamp(0.30 * sequence_hierarchy + 0.30 * sequence_quality + 0.20 * correction_control + 0.20 * structural_integrity, 0.0, 1.0) if impulse_pattern else 0.0
        recent_imp_q = _recent_mean(wave_quality_a, 5, wave_quality)
        recent_imp_eff = _recent_mean(wave_eff_a, 5, q_eff)
        impulse_ctx_score = _clamp(0.55 * impulse_seq_score + 0.25 * recent_imp_q + 0.20 * recent_imp_eff, 0.0, 1.0) if impulse_pattern else 0.0
        impulse_qualified = impulse_pattern and impulse_ctx_score >= e_context_min
        recent_corr_q = _recent_mean(wave_quality_a, 3, wave_quality)
        recent_corr_eff = _recent_mean(wave_eff_a, 3, q_eff)
        corr_depth = counter_pressure if math.isnan(last_corr_retr) else _clamp(last_corr_retr / 0.35, 0.0, 1.0)
        corr_seq_score = _clamp(0.40 * recent_corr_q + 0.25 * recent_corr_eff + 0.20 * structural_integrity + 0.15 * corr_depth, 0.0, 1.0) if corr_pattern else 0.0
        corr_qualified = corr_pattern and corr_seq_score >= e_context_min

        late_impulse = _clamp(maturity * (0.45 * wave_strength_decay + 0.30 * sequence_hierarchy + 0.25 * impulse_ctx_score), 0.0, 1.0) if impulse_qualified else 0.0
        if impulse_qualified:
            elliott_context = "Late impulse context" if late_impulse > 0.55 else ("Dominant middle expansion" if sequence_hierarchy > 0.45 else "Five-segment impulse context")
        elif corr_qualified:
            elliott_context = "Three-segment correction context"
        elif leg_aligned >= 2:
            elliott_context = "Developing wave sequence"
        else:
            elliott_context = "Unclassified"

        # ---- exhaustion ----
        decay_evidence_base = 0.42 * wave_strength_decay + 0.28 * eff_loss + 0.20 * ppe_loss + participation_weight * participation_loss
        decay_evidence = _clamp(decay_evidence_base / max(0.90 + participation_weight, 0.01), 0.0, 1.0)
        continuation_failure = _clamp(0.53 * price_pressure_mismatch + 0.47 * (1.0 if failed_expansion else 0.0), 0.0, 1.0)
        exhaustion = _clamp(maturity * (0.55 * decay_evidence + 0.45 * continuation_failure) * 1.35, 0.0, 1.0) if maturity >= exhaustion_maturity_min else 0.0
        deterioration = _clamp(0.45 * wave_strength_decay + 0.30 * price_pressure_mismatch + 0.25 * continuation_failure, 0.0, 1.0)
        weak_state = (not (math.isnan(p2) or math.isnan(p3))) and p3 < p2 * 0.80 and maturity > 0.45
        weakening = weak_state and not prev["weakening"]

        # ---- transition memory ----
        if structural_exit:
            last_leg_dir = prev_dir
            last_leg_exit_bar = i
            last_leg_maturity = maturity
            last_leg_exhaustion = exhaustion
            last_leg_strength = structural_strength
            last_leg_conf = structural_conf
            last_leg_aligned = max(aligned_wave, 0.0)
            last_leg_counter = counter_pressure
            last_leg_wq = wave_quality
            last_leg_deterioration = deterioration
            last_leg_context = elliott_context
        trans_age = (e_wave_len * 2) if last_leg_exit_bar is None else i - last_leg_exit_bar
        e_transition_memory = transition_memory_bars if advanced_calibration else int(round(e_wave_len * (1.0 if sensitivity_mode == "Fast" else 1.5 if sensitivity_mode == "Structural" else 1.2)))
        transition_active = struct_dir == 0 and last_leg_dir != 0 and trans_age <= e_transition_memory
        transition_fade = _clamp(1.0 - trans_age / max(float(e_transition_memory), 1.0), 0.0, 1.0) if transition_active else 0.0

        # ---- phase engine ----
        structural_flow_flip = structural_flip
        transition = struct_dir == 0 or structural_strength < 0.16 or structural_conf < 0.18
        initiation = (not transition and cur_leg_bars <= max(4.0, hist_leg_bars * 0.18)
                      and (math.isnan(prev["struct_strength"]) or structural_strength > prev["struct_strength"]))
        acceleration_phase = (not transition and aligned_wave > 0.08 and aligned_accel > 0.10
                              and (math.isnan(prev["wave_energy"]) or wave_energy > prev["wave_energy"]))
        correction_phase = not transition and (seg_polarity == -1 or counter_pressure > 0.22) and structural_integrity > 0.28
        compression_phase = correction_phase and compression > 0.34
        resync_phase = resync_stage >= 2 or resync_glow > 0.0
        late_expansion = not transition and aligned_wave > 0.06 and maturity > 0.56 and (wave_strength_decay > 0.18 or late_impulse > 0.35)
        exhaustion_phase = not transition and exhaustion > 0.42
        expansion_phase = not transition and aligned_wave > 0.12 and wave_quality > 0.25 and counter_pressure < 0.28

        if structural_flow_flip:
            phase_candidate = 9
        elif transition:
            phase_candidate = 0
        elif exhaustion_phase:
            phase_candidate = 8
        elif resync_phase:
            phase_candidate = 6
        elif compression_phase:
            phase_candidate = 5
        elif correction_phase:
            phase_candidate = 4
        elif late_expansion:
            phase_candidate = 7
        elif expansion_phase:
            phase_candidate = 3
        elif acceleration_phase:
            phase_candidate = 2
        elif initiation:
            phase_candidate = 1
        else:
            phase_candidate = 0

        phase_confirm_bars = 1 if (sensitivity_mode == "Fast" and not advanced_calibration) else 3 if (sensitivity_mode == "Structural" and not advanced_calibration) else 2
        if structural_flow_flip:
            phase = 9
            pending_phase, pending_phase_bars = 9, 0
        elif structural_exit or struct_dir == 0:
            phase = 0
            pending_phase, pending_phase_bars = 0, 0
        elif phase == 9:
            phase = phase_candidate
            pending_phase, pending_phase_bars = phase_candidate, 0
        elif phase_candidate == phase:
            pending_phase, pending_phase_bars = phase, 0
        else:
            if phase_candidate != pending_phase:
                pending_phase, pending_phase_bars = phase_candidate, 1
            else:
                pending_phase_bars += 1
            if pending_phase_bars >= phase_confirm_bars:
                phase = pending_phase
                pending_phase_bars = 0

        if phase == 1:
            phase_conf = _clamp(0.5 * structural_strength + 0.5 * (1.0 - age_score), 0.0, 1.0)
        elif phase == 2:
            phase_conf = _clamp(0.45 * max(aligned_wave, 0.0) + 0.35 * accel_trend + 0.20 * wave_quality, 0.0, 1.0)
        elif phase == 3:
            phase_conf = _clamp(0.35 * structural_strength + 0.35 * max(aligned_wave, 0.0) + 0.30 * wave_quality, 0.0, 1.0)
        elif phase == 4:
            phase_conf = _clamp(0.55 * counter_pressure + 0.45 * structural_integrity, 0.0, 1.0)
        elif phase == 5:
            phase_conf = compression
        elif phase == 6:
            phase_conf = max(resync_quality, resync_glow)
        elif phase == 7:
            phase_conf = _clamp(0.55 * maturity + 0.45 * max(aligned_wave, 0.0), 0.0, 1.0)
        elif phase == 8:
            phase_conf = exhaustion
        elif phase == 9:
            phase_conf = 1.0
        else:
            phase_conf = _clamp(1.0 - structural_conf, 0.0, 1.0)

        exhaustion_event = exhaustion > 0.60 and (math.isnan(prev["exhaustion"]) or prev["exhaustion"] <= 0.60)

        # ---- record ----
        out["struct_dir"][i] = struct_dir
        out["structural_drive"][i] = structural_drive
        out["structural_strength"][i] = structural_strength
        out["structural_confidence"][i] = structural_conf
        out["internal_wave"][i] = iw
        out["aligned_wave"][i] = aligned_wave
        out["wave_energy"][i] = wave_energy
        out["wave_quality"][i] = wave_quality
        out["counter_pressure"][i] = counter_pressure
        out["structural_integrity"][i] = structural_integrity
        out["retracement"][i] = retracement
        out["maturity"][i] = maturity
        out["exhaustion"][i] = exhaustion
        out["compression"][i] = compression
        out["resync_stage"][i] = resync_stage
        out["resync_quality"][i] = resync_quality
        out["phase"][i] = phase
        out["phase_confidence"][i] = phase_conf
        out["seg_polarity"][i] = seg_polarity
        out["wave_strength_decay"][i] = wave_strength_decay
        out["leg_bars"][i] = cur_leg_bars
        out["participation"][i] = participation
        flags["structural_flip"][i] = 1 if structural_flip else 0
        flags["structural_initiated"][i] = 1 if structural_initiated else 0
        flags["structural_exit"][i] = 1 if structural_exit else 0
        flags["resync_confirmed"][i] = 1 if resync_confirmed else 0
        flags["failed_expansion"][i] = 1 if failed_expansion else 0
        flags["weakening_sequence"][i] = 1 if weakening else 0
        flags["exhaustion_event"][i] = 1 if exhaustion_event else 0
        flags["impulse_context"][i] = 1 if (segment_ended and impulse_qualified) else 0
        flags["correction_context"][i] = 1 if (segment_ended and corr_qualified) else 0

        cp_hist.append(counter_pressure)
        if len(cp_hist) > 200:
            cp_hist.pop(0)
        prev["struct_strength"] = structural_strength
        prev["wave_energy"] = wave_energy
        prev["weakening"] = weak_state
        prev["exhaustion"] = exhaustion
        prev["phase"] = phase

    df = pd.DataFrame(index=idx)
    df["swf_struct_dir"] = pd.Series(out["struct_dir"], index=idx).astype("Int64")
    df["swf_structural_drive"] = out["structural_drive"]
    df["swf_structural_strength"] = out["structural_strength"]
    df["swf_structural_confidence"] = out["structural_confidence"]
    df["swf_internal_wave"] = out["internal_wave"]
    df["swf_aligned_wave"] = out["aligned_wave"]
    df["swf_wave_energy"] = out["wave_energy"]
    df["swf_wave_quality"] = out["wave_quality"]
    df["swf_counter_pressure"] = out["counter_pressure"]
    df["swf_structural_integrity"] = out["structural_integrity"]
    df["swf_retracement"] = out["retracement"]
    df["swf_maturity"] = out["maturity"]
    df["swf_exhaustion"] = out["exhaustion"]
    df["swf_compression"] = out["compression"]
    df["swf_resync_stage"] = pd.Series(out["resync_stage"], index=idx).astype("Int64")
    df["swf_resync_quality"] = out["resync_quality"]
    df["swf_phase"] = pd.Series(out["phase"], index=idx).astype("Int64")
    df["swf_phase_confidence"] = out["phase_confidence"]
    df["swf_seg_polarity"] = pd.Series(out["seg_polarity"], index=idx).astype("Int64")
    df["swf_wave_strength_decay"] = out["wave_strength_decay"]
    df["swf_leg_bars"] = out["leg_bars"]
    df["swf_participation"] = out["participation"]
    for k, v in flags.items():
        df[f"swf_{k}"] = pd.Series(v, index=idx).astype("Int64")
    return df
