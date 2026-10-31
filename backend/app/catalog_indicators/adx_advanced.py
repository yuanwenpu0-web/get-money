"""ADX 进阶：双长度 Wilder DMI 之后，再挂一层可选平滑核得到 ADX 信号线与柱差。

Source: pine-collection/indicators/trend_strength/adx_advanced/adx_advanced.pine
(WavesUnchained, MIT)

`ta.dmi(diLength, adxSmoothing)` is reproduced with its two independent lengths: the
Wilder-smoothed +DI/-DI use `di_length`, the DX average uses `adx_smoothing`. The ADX
line is then smoothed once more by `signal_length` bars with `smoothing_type`; the
histogram is the raw ADX minus that signal line, and the four crosses (DI+/DI- and ADX
across `level_weak`) are exposed as flags.

The script's own recursive smoothers are reimplemented here because `ta.*` cannot express
them: SuperSmoother (Ehlers two-pole), T3 (six chained EMAs), KAMA (Kaufman adaptive) and
the JMA approximation. Each keeps Pine's `nz()` handling, so an undefined predecessor is
read as zero and the warm-up gap never poisons the recursion.

`levelStrong` only feeds the colour gradient and a horizontal line, and the display
switches (`showDI`, `showDIFill`, `showShadow`, `showHist`, `showSignals`) only gate
plots, so they are dropped along with the colour inputs, the markers and the alerts.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import (choice, crossover, crossunder, ema, integer, ma, number, ohlcv, rma,
                    true_range)

_SMOOTHERS = ("EMA", "SMA", "RMA", "WMA", "SuperSmoother", "T3", "KAMA", "JMA Approx")


def _dmi(frame: pd.DataFrame, di_length: int, adx_length: int) -> tuple[pd.Series, ...]:
    """`ta.dmi(diLength, adxSmoothing)`: Wilder +DI/-DI on one length, ADX on the other."""
    high, low, _ = ohlcv(frame, "high", "low", "close")
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    span = rma(true_range(frame), di_length).replace(0, np.nan)
    plus = 100.0 * rma(plus_dm, di_length) / span
    minus = 100.0 * rma(minus_dm, di_length) / span
    dx = 100.0 * (plus - minus).abs() / (plus + minus).replace(0, np.nan)
    return plus, minus, rma(dx, adx_length)


def _super_smoother(series: pd.Series, length: int) -> pd.Series:
    """Ehlers two-pole smoother; `nz` on the two lags keeps the recursion finite."""
    values = series.to_numpy(dtype=float)
    out = np.empty(len(values))
    angle = 1.414 * math.pi / length
    a1 = math.exp(-angle)
    c2 = 2.0 * a1 * math.cos(angle)
    c3 = -(a1 * a1)
    c1 = 1.0 - c2 - c3
    lag1 = lag2 = 0.0
    for position, value in enumerate(values):
        previous = values[position - 1] if position else np.nan
        previous = 0.0 if np.isnan(previous) else previous
        current = (c1 * (value + previous) / 2.0
                   + c2 * (0.0 if np.isnan(lag1) else lag1)
                   + c3 * (0.0 if np.isnan(lag2) else lag2))
        out[position] = current
        lag2, lag1 = lag1, current
    return pd.Series(out, index=series.index)


def _t3(series: pd.Series, length: int) -> pd.Series:
    """Tim Tillson's T3: six chained EMAs combined with the 0.7 volume factor."""
    volume_factor = 0.7
    e1 = ema(series, length)
    e2 = ema(e1, length)
    e3 = ema(e2, length)
    e4 = ema(e3, length)
    e5 = ema(e4, length)
    e6 = ema(e5, length)
    b = volume_factor
    c1 = -(b * b * b)
    c2 = 3.0 * b * b + 3.0 * b * b * b
    c3 = -6.0 * b * b - 3.0 * b - 3.0 * b * b * b
    c4 = 1.0 + 3.0 * b + b * b * b + 3.0 * b * b
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def _kama(series: pd.Series, length: int) -> pd.Series:
    """Kaufman adaptive average: efficiency ratio scales the smoothing constant."""
    values = series.to_numpy(dtype=float)
    change = (series - series.shift(length)).abs()
    volatility = (series.diff().abs().rolling(length, min_periods=length).sum())
    efficiency = (change / volatility.where(volatility.ne(0))).where(volatility.ne(0), 0.0)
    smoothing = (efficiency * (2.0 / 3.0 - 2.0 / 31.0) + 2.0 / 31.0) ** 2
    out = np.empty(len(values))
    previous = np.nan
    for position, value in enumerate(values):
        base = value if np.isnan(previous) else previous
        current = base + smoothing.iloc[position] * (value - base)
        out[position] = current
        previous = current
    return pd.Series(out, index=series.index)


def _jma(series: pd.Series, length: int, phase: int, power: float) -> pd.Series:
    """The script's JMA approximation (three-pole recursive filter on the source)."""
    values = series.to_numpy(dtype=float)
    out = np.empty(len(values))
    phase_ratio = 0.5 if phase < -100 else 2.5 if phase > 100 else phase / 100.0 + 1.5
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2.0)
    alpha = beta ** power
    e0_previous = e1_previous = e2_previous = jma_previous = np.nan
    for position, value in enumerate(values):
        e0_base = value if np.isnan(e0_previous) else e0_previous
        e0 = (1.0 - alpha) * value + alpha * e0_base
        e1 = (value - e0) * (1.0 - beta) + beta * (0.0 if np.isnan(e1_previous) else e1_previous)
        jma_base = value if np.isnan(jma_previous) else jma_previous
        e2 = ((e0 + phase_ratio * e1 - jma_base) * (1.0 - alpha) ** 2
              + alpha ** 2 * (0.0 if np.isnan(e2_previous) else e2_previous))
        current = jma_base + e2
        out[position] = current
        e0_previous, e1_previous, e2_previous, jma_previous = e0, e1, e2, current
    return pd.Series(out, index=series.index)


def _smooth(series: pd.Series, length: int, mode: str, phase: int, power: float) -> pd.Series:
    if mode == "SuperSmoother":
        return _super_smoother(series, length)
    if mode == "T3":
        return _t3(series, length)
    if mode == "KAMA":
        return _kama(series, length)
    if mode == "JMA Approx":
        return _jma(series, length, phase, power)
    return ma(series, length, mode)


def adx_advanced(frame: pd.DataFrame, *, di_length: int = 14, adx_smoothing: int = 14,
                 signal_length: int = 3, smoothing_type: str = "EMA", jma_phase: int = 50,
                 jma_power: float = 2.0, level_weak: float = 20.0) -> pd.DataFrame:
    integer("di_length", di_length, 1)
    integer("adx_smoothing", adx_smoothing, 1)
    integer("signal_length", signal_length, 1)
    choice("smoothing_type", smoothing_type, _SMOOTHERS)
    integer("jma_phase", jma_phase, -100, 100)
    number("jma_power", jma_power, 1.0, 4.0)
    number("level_weak", level_weak, 0.0, 100.0)

    di_plus, di_minus, adx_raw = _dmi(frame, di_length, adx_smoothing)
    adx_signal = _smooth(adx_raw, signal_length, smoothing_type, jma_phase, jma_power)
    threshold = pd.Series(level_weak, index=frame.index)

    return pd.DataFrame({
        "di_plus": di_plus,
        "di_minus": di_minus,
        "adx": adx_raw,
        "adx_signal": adx_signal,
        "adx_histogram": adx_raw - adx_signal,
        "di_bull_cross": crossover(di_plus, di_minus),
        "di_bear_cross": crossunder(di_plus, di_minus),
        "adx_activated": crossover(adx_raw, threshold),
        "adx_deactivated": crossunder(adx_raw, threshold),
    }, index=frame.index)
