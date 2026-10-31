"""Market Memory & Decay Oscillator — carry vs. decay composite oscillator.

Source: pine-collection/indicators/trend_strength/market_memory_decay_oscillator/market_memory_decay_oscillator.pine
(WavesUnchained, v1.2.1, MIT)

The oscillator fuses four memory ingredients — the autocorrelation half-life of the
smoothed impulse line, a single-window rescaled-range (Hurst) approximation, the
path efficiency of the price move and a fast-vs-slow true-range expansion ratio —
into a signed carry score. `impulse_dir * impulse_strength` signs and sizes both the
carry leg (weighted by the exponentially decaying memory) and the decay leg, and a
final EMA smooths the net reading.

Port notes:
- `ta.tr(true)` keeps the first bar's true range as `high - low` (matches
  `_pine.true_range`).
- Pine's `f_safeDiv(a, b, fallback)` returns `fallback` when `b` is 0 *or* NaN, so
  warm-up rows of the z-score, efficiency and range ratio fall back instead of
  staying NaN.
- `retZ` is 0.0 during the mean/stdev warm-up (fallback), so the impulse EMA seeds
  on zeros exactly like Pine.
- `age` uses 999.0 before the first impulse (Pine's `na(barsFromImpulse)` guard);
  the half-life, Hurst and efficiency loops treat warm-up returns as 0 via `nz`.

Dropped: zones/fills, impulse marker plot, background shading, dashboard table,
state text banner and alertcondition payloads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (barssince, correlation, ema, flags, integer, number, ohlcv,
                    sma, stdev, true_range)


def _clamp_series(values: pd.Series, low: float, high: float) -> pd.Series:
    return values.clip(lower=low, upper=high)


def _safe_div(numerator: pd.Series, denominator: pd.Series, fallback: float) -> pd.Series:
    """Pine f_safeDiv: fallback when the denominator is 0 or NaN (numerator may be NaN)."""
    ok = denominator.notna() & denominator.ne(0.0)
    return numerator.where(ok, fallback) / denominator.where(ok, 1.0)


def market_memory_and_decay_oscillator(frame: pd.DataFrame, *, memory_len: int = 64,
                                       fast_len: int = 16, slow_len: int = 96,
                                       smooth_len: int = 8, impulse_z: float = 1.2,
                                       max_half_life: float = 48.0) -> pd.DataFrame:
    integer("memory_len", memory_len, 20)
    integer("fast_len", fast_len, 5)
    integer("slow_len", slow_len, 30)
    integer("smooth_len", smooth_len, 1)
    number("impulse_z", impulse_z, 0.0)
    number("max_half_life", max_half_life, 0.1)
    close = ohlcv(frame, "close")[0]
    src = close

    ret = np.log(src / src.shift(1))
    ret_mean = sma(ret, memory_len)
    ret_sd = stdev(ret, memory_len)
    ret_z = _safe_div(ret - ret_mean, ret_sd, 0.0)

    impulse_raw = ema(ret_z, fast_len)
    impulse_dir = pd.Series(np.sign(impulse_raw.fillna(0.0)).where(impulse_raw.notna()),
                            index=frame.index)
    impulse_strength = _clamp_series(impulse_raw.abs() / 2.5, 0.0, 1.0)

    new_impulse = flags(ret_z.abs() > impulse_z)
    bars_from_impulse = barssince(new_impulse)
    age = bars_from_impulse.fillna(999.0)

    phi = correlation(impulse_raw, impulse_raw.shift(1), memory_len)
    persistent_phi = _clamp_series(phi.fillna(0.0), 0.0, 0.98)
    half_life = pd.Series(0.0, index=frame.index)
    positive = persistent_phi > 0.01
    half_life[positive] = -np.log(2.0) / np.log(persistent_phi[positive])
    half_life = _clamp_series(half_life, 0.0, max_half_life)

    decay_curve = np.exp(-age / half_life.where(half_life > 0)).fillna(0.0)
    half_life_score = _clamp_series(half_life / max_half_life, 0.0, 1.0)

    # Rescaled range over the memory window: cumulative deviations of nz(ret) from
    # nz(retMean). Pine always sums exactly memory_len terms, treating out-of-range
    # (future-free, i.e. pre-history) returns as 0 via nz().
    def _hurst_row(position: int) -> float:
        mean = ret_mean.iat[position]
        sd = ret_sd.iat[position]
        available = np.nan_to_num(ret.iloc[:position + 1].to_numpy(dtype=float), nan=0.0)
        window = np.zeros(memory_len)
        window[memory_len - min(memory_len, len(available)):] = available[-memory_len:]
        deviations = window - (0.0 if np.isnan(mean) else mean)
        cumulative = np.cumsum(deviations)
        spread = max(cumulative.max(), 0.0) - min(cumulative.min(), 0.0)
        rs = spread / sd if not np.isnan(sd) and sd != 0.0 else 1.0
        return float(np.clip(np.log(max(rs, 1.0)) / np.log(memory_len), 0.0, 1.0))

    hurst = pd.Series([_hurst_row(i) for i in range(len(frame))], index=frame.index)
    hurst_memory = _clamp_series((hurst - 0.50) * 2.5, -1.0, 1.0)

    direct = (src - src.shift(memory_len)).abs()
    # path = ta.sum(|src - src[1]|, memoryLen); |src - src[1]| is NaN only on bar 0.
    path = src.diff().abs().rolling(memory_len, min_periods=memory_len).sum()
    efficiency = _safe_div(direct, path, 0.0)
    eff_score = _clamp_series((efficiency - 0.20) * 2.0, 0.0, 1.0)

    tr = true_range(frame)
    range_fast = ema(tr, fast_len)
    range_slow = ema(tr, slow_len)
    range_ratio = _safe_div(range_fast, range_slow, 1.0)
    range_expansion = _clamp_series((range_ratio - 1.0) / 0.8, 0.0, 1.0)
    range_decay = _clamp_series((1.0 - range_ratio) / 0.45, 0.0, 1.0)

    memory_strength = (half_life_score * 0.35 + hurst_memory.clip(lower=0.0) * 0.25
                       + eff_score * 0.25 + range_expansion * 0.15)
    decay_pressure = (range_decay * 0.35 + _clamp_series(age / max_half_life, 0.0, 1.0) * 0.25
                      + (-hurst_memory).clip(lower=0.0) * 0.25
                      + _clamp_series(1.0 - efficiency * 3.0, 0.0, 1.0) * 0.15)

    carry_raw = impulse_dir * impulse_strength * memory_strength * decay_curve * 100.0
    decay_raw = impulse_dir * decay_pressure * impulse_strength * 100.0

    mmdo = ema(carry_raw - decay_raw * 0.55, smooth_len)
    decay_line = ema(decay_raw, smooth_len)
    memory_line = ema(impulse_dir * memory_strength * 100.0, smooth_len)

    carry_bull = flags(mmdo > 25)
    carry_bear = flags(mmdo < -25)
    decaying = flags(decay_line.abs() > memory_line.abs() * 0.65)
    decaying &= mmdo.abs() < memory_line.abs()

    return pd.DataFrame({
        "mmdo": mmdo,
        "memory_line": memory_line,
        "decay_line": decay_line,
        "ret_z": ret_z,
        "impulse_raw": impulse_raw,
        "half_life": half_life,
        "impulse_age": age,
        "hurst": hurst,
        "efficiency": efficiency,
        "range_ratio": range_ratio,
        "new_impulse": new_impulse,
        "carry_bull": carry_bull,
        "carry_bear": carry_bear,
        "decaying": decaying,
    }, index=frame.index)
