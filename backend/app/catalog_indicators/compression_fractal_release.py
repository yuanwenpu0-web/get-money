"""Compression Fractal Release — coil detection with trend-aligned break signals.

Source: pine-collection/indicators/volatility/compression_fractal_release/compression_fractal_release.pine
(WavesUnchained, MIT)

Coil strength blends symbolised-return entropy (70/30 with the transition
rate), a fractal-dimension score and inverse efficiency. The release dynamics
(drop in complexity, gain in efficiency versus their 3-bar averages) scale the
coil into a setup score between `rel_floor` and 1. Breaks of the previous
`break_len` high/low band fire as continuation releases when aligned with the
HTF EMA slope direction, or as counter-trend base breaks when the window was a
base for at least `base_min_share` of its bars; both require a minimum ATR
impulse beyond the band and are vetoed by excessive choppiness, with a
per-direction cooldown.

The HTF trend EMA and ATR follow the repo-wide convention of reading the
*previous completed* HTF period (the lookahead-free equivalent of the upstream
`request.security(..., lookahead_off)`), computed on HTF bars aggregated from
the input frame and lifted back with `_tf.lift_series`.

Dropped: dashboard table, compression box, background/labels/markers, Pine
Logs debug output and the alert plumbing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import atr, choice, ema, flags, highest, integer, lowest, number, ohlcv, sma, sum_
from ._tf import base_step_minutes, htf_frame, lift_series

_LOG5 = float(np.log(5.0))
_LOG2 = float(np.log(2.0))
_LOG10 = float(np.log(10.0))


def _clamp01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


def _symbols(close: pd.Series, atr_s: pd.Series, unit_mult: float) -> pd.Series:
    diff = close.diff()
    unit = atr_s * unit_mult
    return pd.Series(np.where(diff > unit, 2.0,
                              np.where(diff > 0, 1.0,
                                       np.where(diff < -unit, -2.0,
                                                np.where(diff < 0, -1.0, 0.0)))),
                     index=close.index)


def _complexity(symbols: pd.Series, length: int) -> pd.Series:
    values = symbols.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for position in range(length - 1, len(values)):
        window = values[position - length + 2:position + 1]
        total = max(length - 1, 1)
        entropy = 0.0
        for symbol in (-2.0, -1.0, 0.0, 1.0, 2.0):
            count = float(np.count_nonzero(window == symbol))
            if count > 0:
                share = count / total
                entropy -= share * np.log(share)
        entropy /= _LOG5
        transitions = float(np.count_nonzero(window[1:] != window[:-1]))
        transition_rate = transitions / max(total - 1, 1)
        out[position] = min(100.0, max(0.0, (entropy * 0.70 + transition_rate * 0.30) * 100.0))
    return pd.Series(out, index=symbols.index)


def _fdi(high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    half = max(2, length // 2)
    range_now = (highest(high, half) - lowest(low, half)) / half
    range_past = ((highest(high, half) - lowest(low, half))
                  .shift(half)) / half
    range_full = (highest(high, length) - lowest(low, length)) / length
    usable = ((range_now + range_past) > 0.0) & (range_full > 0.0)
    dimension = ((np.log(range_now + range_past) - np.log(range_full)) / _LOG2).where(usable)
    return dimension.fillna(1.5).clip(lower=1.0, upper=2.0)


def _efficiency(close: pd.Series, length: int) -> pd.Series:
    path = sum_(close.diff().abs(), length - 1)
    direct = (close - close.shift(length - 1)).abs()
    return pd.Series(np.where(path > 0, (direct / path * 100.0).clip(0.0, 100.0), 0.0),
                     index=close.index)


def _choppiness(frame: pd.DataFrame, length: int) -> pd.Series:
    high, low, close = ohlcv(frame, "high", "low", "close")
    from ._pine import true_range
    atr_sum = sum_(true_range(frame), length)
    rng = highest(high, length) - lowest(low, length)
    return pd.Series(np.where(rng > 0,
                              (100.0 * np.log10(atr_sum / rng) / np.log10(length)).clip(0.0, 100.0),
                              0.0), index=frame.index)


def compression_fractal_release(
        frame: pd.DataFrame, *, length: int = 34, break_len: int = 12,
        symbol_atr_mult: float = 0.15, htf_minutes: int = 240, ema_len: int = 55,
        slope_len: int = 5, htf_slope_min: float = 0.10, complex_high: float = 62.0,
        complex_low: float = 48.0, eff_high: float = 42.0, eff_low: float = 25.0,
        rel_floor: float = 0.35, coil_now_min: float = 0.45, thr_cont: float = 0.45,
        thr_base: float = 0.55, base_min_share: float = 0.35,
        break_atr_min: float = 0.10, cooldown_bars: int = 8, chop_len: int = 14,
        chop_veto_max: float = 61.8) -> pd.DataFrame:
    integer("length", length, 10, 150)
    integer("break_len", break_len, 3, 50)
    number("symbol_atr_mult", symbol_atr_mult, 0.01, 1.0)
    integer("htf_minutes", htf_minutes, 1)
    integer("ema_len", ema_len, 10, 300)
    integer("slope_len", slope_len, 1, 30)
    number("htf_slope_min", htf_slope_min, 0.0, 2.0)
    number("complex_high", complex_high, 40.0, 90.0)
    number("complex_low", complex_low, 20.0, 70.0)
    number("eff_high", eff_high, 20.0, 80.0)
    number("eff_low", eff_low, 5.0, 60.0)
    number("rel_floor", rel_floor, 0.05, 0.9)
    number("coil_now_min", coil_now_min, 0.0, 1.0)
    number("thr_cont", thr_cont, 0.0, 1.0)
    number("thr_base", thr_base, 0.0, 1.0)
    number("base_min_share", base_min_share, 0.05, 1.0)
    number("break_atr_min", break_atr_min, 0.0, 2.0)
    integer("cooldown_bars", cooldown_bars, 0, 100)
    integer("chop_len", chop_len, 2, 50)
    number("chop_veto_max", chop_veto_max, 30.0, 80.0)

    high, low, close = ohlcv(frame, "high", "low", "close")
    atr14 = atr(frame, 14)

    # --- Higher timeframe trend context (previous completed HTF period) ---
    base = base_step_minutes(frame)
    if htf_minutes % base or htf_minutes < base:
        raise ValueError(f"高周期 {htf_minutes} 分钟必须是基础周期 {base} 分钟的整数倍")
    htf = htf_frame(frame, htf_minutes)
    htf_ema = ema(htf["close"], ema_len)
    htf_ema_past = htf_ema.shift(slope_len)
    htf_atr = atr(htf, 14)
    htf_slope = pd.Series(np.where(htf_atr > 0, (htf_ema - htf_ema_past) / htf_atr, 0.0),
                          index=htf.index)
    ema_lifted = lift_series(frame, htf_ema, htf_minutes)
    past_lifted = lift_series(frame, htf_ema_past, htf_minutes)
    slope_lifted = lift_series(frame, htf_slope, htf_minutes)
    htf_slope_atr = slope_lifted.fillna(0.0)
    htf_dir = pd.Series(np.where(htf_slope_atr > htf_slope_min, 1,
                                 np.where(htf_slope_atr < -htf_slope_min, -1, 0)),
                        index=frame.index)

    # --- Coil metrics ---
    symbols = _symbols(close, atr14, symbol_atr_mult)
    complexity = _complexity(symbols, length)
    fdi = _fdi(high, low, length)
    fd_score = (fdi - 1.0) * 100.0
    efficiency = _efficiency(close, length)
    chop = _choppiness(frame, chop_len)

    coil_strength = _clamp01((complexity / 100.0) * 0.5 + (fd_score / 100.0) * 0.3
                             + (1.0 - efficiency / 100.0) * 0.2)
    coil_recent = highest(coil_strength, break_len).shift(1)
    coil_blend = _clamp01(coil_recent * 0.70 + coil_strength * 0.30)
    coil_current_ok = coil_strength >= coil_now_min
    coil_score = pd.Series(
        np.where(coil_current_ok, coil_blend,
                 coil_blend * _clamp01(coil_strength / max(coil_now_min, 0.01))),
        index=frame.index)

    cmp_drop = _clamp01((sma(complexity, 3).shift(1) - complexity) / 8.0)
    eff_gain = _clamp01((efficiency - sma(efficiency, 3).shift(1)) / 8.0)
    release_dyn = (cmp_drop + eff_gain) / 2.0
    release_mult = rel_floor + (1.0 - rel_floor) * release_dyn
    setup_score = _clamp01(coil_score * release_mult)

    clean_trend = flags((complexity < complex_low) & (fd_score < complex_low)
                        & (efficiency > eff_high))
    base_regime = flags((complexity > complex_high) & (efficiency < eff_low))
    correction = flags((htf_dir != 0) & ~base_regime & (coil_strength > 0.55))

    base_bars = sum_(flags(base_regime).astype(float), break_len).shift(1)
    was_base = flags(base_bars >= round(break_len * base_min_share))

    # --- Break band and signals ---
    corr_high = highest(high, break_len).shift(1)
    corr_low = lowest(low, break_len).shift(1)
    band_mid = (corr_high + corr_low) / 2.0
    break_up = flags(close > corr_high)
    break_dn = flags(close < corr_low)
    first_up = flags(break_up & ~break_up.shift(1, fill_value=False))
    first_dn = flags(break_dn & ~break_dn.shift(1, fill_value=False))
    break_atr_up = pd.Series(np.where(atr14 > 0, (close - corr_high).abs() / atr14, 0.0),
                             index=frame.index)
    break_atr_dn = pd.Series(np.where(atr14 > 0, (close - corr_low).abs() / atr14, 0.0),
                             index=frame.index)
    break_impulse = pd.Series(np.where(first_up, break_atr_up,
                                       np.where(first_dn, break_atr_dn, 0.0)),
                              index=frame.index)
    chop_ok = flags(chop < chop_veto_max)
    quality = flags(chop_ok & (break_impulse >= break_atr_min))

    release_long_raw = flags(first_up & (htf_dir == 1) & (setup_score >= thr_cont) & quality)
    release_short_raw = flags(first_dn & (htf_dir == -1) & (setup_score >= thr_cont) & quality)
    base_break_long_raw = flags(first_up & (htf_dir == -1) & was_base
                                & (setup_score >= thr_base) & quality)
    base_break_short_raw = flags(first_dn & (htf_dir == 1) & was_base
                                 & (setup_score >= thr_base) & quality)

    # Cooldown persistence (Pine `var int lastUp/lastDn`).
    rows = len(frame)
    last_up = last_dn = None
    release_long_values = np.zeros(rows, dtype=bool)
    release_short_values = np.zeros(rows, dtype=bool)
    base_break_long_values = np.zeros(rows, dtype=bool)
    base_break_short_values = np.zeros(rows, dtype=bool)
    up_raw = release_long_raw.to_numpy(dtype=bool) | base_break_long_raw.to_numpy(dtype=bool)
    dn_raw = release_short_raw.to_numpy(dtype=bool) | base_break_short_raw.to_numpy(dtype=bool)
    release_long_raw_values = release_long_raw.to_numpy(dtype=bool)
    release_short_raw_values = release_short_raw.to_numpy(dtype=bool)
    base_break_long_raw_values = base_break_long_raw.to_numpy(dtype=bool)
    base_break_short_raw_values = base_break_short_raw.to_numpy(dtype=bool)
    for position in range(rows):
        up_ok = last_up is None or (position - last_up) >= cooldown_bars
        dn_ok = last_dn is None or (position - last_dn) >= cooldown_bars
        release_long_values[position] = release_long_raw_values[position] and up_ok
        base_break_long_values[position] = base_break_long_raw_values[position] and up_ok
        release_short_values[position] = release_short_raw_values[position] and dn_ok
        base_break_short_values[position] = base_break_short_raw_values[position] and dn_ok
        if up_raw[position]:
            last_up = position
        if dn_raw[position]:
            last_dn = position

    coiled = flags(coil_strength > 0.60)
    near_up = flags(high >= corr_high * 0.995)
    near_dn = flags(low <= corr_low * 1.005)

    return pd.DataFrame({
        "complexity": complexity,
        "fractal_dimension": fdi,
        "fd_score": fd_score,
        "efficiency": efficiency,
        "choppiness": chop,
        "coil_strength": coil_strength,
        "coil_score": coil_score,
        "release_dyn": release_dyn,
        "setup_score": setup_score,
        "htf_dir": pd.Series(htf_dir, index=frame.index).astype("Int64"),
        "htf_slope_atr": htf_slope_atr,
        "htf_ema": ema_lifted,
        "clean_trend": clean_trend,
        "base_regime": base_regime,
        "correction": correction,
        "was_base": was_base,
        "base_bars": base_bars,
        "corr_high": corr_high,
        "corr_low": corr_low,
        "band_mid": band_mid,
        "break_up": break_up,
        "break_dn": break_dn,
        "first_up": first_up,
        "first_dn": first_dn,
        "break_impulse": break_impulse,
        "chop_ok": chop_ok,
        "quality": quality,
        "release_long": flags(pd.Series(release_long_values, index=frame.index)),
        "release_short": flags(pd.Series(release_short_values, index=frame.index)),
        "base_break_long": flags(pd.Series(base_break_long_values, index=frame.index)),
        "base_break_short": flags(pd.Series(base_break_short_values, index=frame.index)),
        "coiled": coiled,
        "near_break_up": near_up,
        "near_break_dn": near_dn,
    }, index=frame.index)
