"""Oscillator Topology — WaveTrend pivot shapes (V-Spike / Flat / Asymm / Round).

Source: pine-collection/indicators/momentum/oscillator_topology/oscillator_topology.pine
(WavesUnchained, MIT)

A classic WaveTrend oscillator (EMA of hlc3 minus its EMA, normalised by the EMA of
the absolute distance) is scanned for pivots. Each confirmed pivot is classified by
the geometry around it: a normalised curvature above `spike_threshold` is a V-Spike,
an unbroken run of `flat_bars` bars hugging the extreme (within `flat_tolerance` of
the pivot height above the 30-bar opposite extreme) is a Flat zone, a rise/fall
speed asymmetry above 0.5 is Asymm, everything else Round.

Port notes:
- `syminfo.mintick` is not available offline; the WaveTrend denominator uses a
  1e-10 floor instead of `0.015 * mintick`. Only matters when the price is exactly
  flat (then the numerator is 0 as well).
- The shape is evaluated on the bar the pivot is *confirmed* (pivot bar + right
  bars), reading only bars ≤ that bar — strictly causal.

Dropped: the shape-colour helpers, labels, dashboard table and the alert
payloads. `last_high_shape`/`last_low_shape` keep the most recent classification
(along with its curvature) instead of the dashboard cells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import ema, flags, highest, integer, lowest, number, ohlcv, pivothigh, pivotlow

_MINTICK_FLOOR = 1e-10


def _classify(values: np.ndarray, bar: int, pivot_bar: int, is_high: bool,
              pivot_left: int, spike_threshold: float, flat_bars: int,
              flat_tolerance: float) -> tuple[str, float, int]:
    """Pine f_shape evaluated at confirmation `bar` for the pivot at `pivot_bar`."""
    pivot_value = values[pivot_bar]
    # curvAtPiv = wtCurv[pivLbR]: the normalised curvature *at the pivot bar*.
    if pivot_bar >= 2 and np.isfinite(values[pivot_bar - 2: pivot_bar + 1]).all():
        acceleration = values[pivot_bar] - 2.0 * values[pivot_bar - 1] + values[pivot_bar - 2]
        spread = np.nanmax(values[max(0, pivot_bar - 19): pivot_bar + 1]) \
            - np.nanmin(values[max(0, pivot_bar - 19): pivot_bar + 1])
        curvature_at_pivot = acceleration / spread * 100.0 if spread > 0 else 0.0
    else:
        curvature_at_pivot = np.nan
    abs_curvature = abs(curvature_at_pivot) if np.isfinite(curvature_at_pivot) else np.nan

    baseline_window = values[max(0, bar - 29): bar + 1]
    baseline = np.nanmin(baseline_window) if is_high else np.nanmax(baseline_window)
    tolerance = abs(pivot_value - baseline) * flat_tolerance

    flat_count = 1
    for offset in range(1, bar - pivot_bar + 1):  # bars after the pivot, up to `bar`
        if abs(values[pivot_bar + offset] - pivot_value) < tolerance:
            flat_count += 1
        else:
            break
    for offset in range(1, min(pivot_left + (bar - pivot_bar) - 1, 20) + 1):
        index = pivot_bar - offset
        if index < 0 or not abs(values[index] - pivot_value) < tolerance:
            break
        flat_count += 1

    rise_speed = (pivot_value - values[pivot_bar - pivot_left]) / pivot_left \
        if pivot_bar - pivot_left >= 0 else np.nan
    fall_speed = (pivot_value - values[bar]) / (bar - pivot_bar)
    total = rise_speed + fall_speed
    asymmetry = (abs(rise_speed - fall_speed) / max(abs(total), 0.01)
                 if np.isfinite(total) and total != 0 else 0.0)

    if not np.isnan(abs_curvature) and abs_curvature >= spike_threshold:
        shape = "V-Spike"
    elif flat_count >= flat_bars:
        shape = "Flat"
    elif asymmetry > 0.5:
        shape = "Asymm"
    else:
        shape = "Round"
    return shape, abs_curvature, flat_count


def oscillator_topology(frame: pd.DataFrame, *, wt_channel_length: int = 10,
                        wt_average_length: int = 21, pivot_left: int = 5,
                        pivot_right: int = 5, spike_threshold: float = 3.0,
                        flat_bars: int = 4, flat_tolerance: float = 0.15) -> pd.DataFrame:
    integer("wt_channel_length", wt_channel_length, 1)
    integer("wt_average_length", wt_average_length, 1)
    integer("pivot_left", pivot_left, 2, 20)
    integer("pivot_right", pivot_right, 2, 20)
    number("spike_threshold", spike_threshold, 0.5, 10.0)
    integer("flat_bars", flat_bars, 2, 15)
    number("flat_tolerance", flat_tolerance, 0.05, 0.5)

    high, low, close = ohlcv(frame, "high", "low", "close")
    typical = (high + low + close) / 3.0
    esa = ema(typical, wt_channel_length)
    distance = ema((typical - esa).abs(), wt_channel_length)
    ci = (typical - esa) / (0.015 * distance.clip(lower=_MINTICK_FLOOR))
    wt_osc = ema(ci, wt_average_length)

    velocity = wt_osc.diff()
    acceleration = velocity.diff()
    osc_range = highest(wt_osc, 20) - lowest(wt_osc, 20)
    curvature = (acceleration / osc_range * 100.0).where(osc_range > 0, 0.0)

    pivot_high = pivothigh(wt_osc, pivot_left, pivot_right)
    pivot_low = pivotlow(wt_osc, pivot_left, pivot_right)

    values = wt_osc.to_numpy(dtype=float)
    n = len(frame)
    shape_high = np.full(n, np.nan, dtype=object)
    shape_low = np.full(n, np.nan, dtype=object)
    curvature_high = np.full(n, np.nan)
    curvature_low = np.full(n, np.nan)
    flat_high = np.full(n, np.nan)
    flat_low = np.full(n, np.nan)

    for bar in range(n):
        if not np.isnan(pivot_high.iat[bar]):
            shape, curve, flat = _classify(values, bar, bar - pivot_right, True,
                                           pivot_left, spike_threshold, flat_bars,
                                           flat_tolerance)
            shape_high[bar], curvature_high[bar], flat_high[bar] = shape, curve, flat
        if not np.isnan(pivot_low.iat[bar]):
            shape, curve, flat = _classify(values, bar, bar - pivot_right, False,
                                           pivot_left, spike_threshold, flat_bars,
                                           flat_tolerance)
            shape_low[bar], curvature_low[bar], flat_low[bar] = shape, curve, flat

    shape_high_series = pd.Series(shape_high, index=frame.index)
    shape_low_series = pd.Series(shape_low, index=frame.index)
    return pd.DataFrame({
        "wt_osc": wt_osc,
        "wt_velocity": velocity,
        "wt_acceleration": acceleration,
        "wt_curvature": curvature,
        "pivot_high": pivot_high,
        "pivot_low": pivot_low,
        "shape_high": shape_high_series,
        "shape_low": shape_low_series,
        "curvature_high": pd.Series(curvature_high, index=frame.index),
        "curvature_low": pd.Series(curvature_low, index=frame.index),
        "flat_count_high": pd.Series(flat_high, index=frame.index, dtype="Int64"),
        "flat_count_low": pd.Series(flat_low, index=frame.index, dtype="Int64"),
        "last_high_shape": shape_high_series.ffill(),
        "last_low_shape": shape_low_series.ffill(),
        "last_high_curvature": pd.Series(curvature_high, index=frame.index).ffill(),
        "last_low_curvature": pd.Series(curvature_low, index=frame.index).ffill(),
        "spike_forming": flags(curvature.abs() > spike_threshold),
    }, index=frame.index)
