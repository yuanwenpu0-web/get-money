"""Nearest confirmed pivot support / resistance from up to three timeframes.

Source: pine-crypto-collection/amadeo987/SK_MTF_Nearest_Support_Resistance.pine

Author: amadeo987 (script title "[ad] MTF Nearest Support / Resistance v1.0").
Upstream ships no licence header; only the formulas it publishes are reproduced
here.

Per timeframe the script snapshots the last `source_memory` confirmed pivot highs
and lows, keeps the closest level above the current close as resistance and the
closest level below it as support, and paints both with `line` / `label` objects
on the last bar. That drawing, the level colours and the `line_history` /
`show_labels` inputs are presentation only and are not ported.

Input must carry `open`, `high`, `low` and `close` on a timezone-aware, strictly
increasing bar-open index; nothing else is read and no data is fetched.

Timeframes are given in minutes (`tf1_minutes` = 60 for Pine's "60", 240 for
"240", 1440 for "1D") because a bare frame carries no exchange calendar. The base
period is inferred from the finest gap between bars; a timeframe must be a whole
multiple of it, and a higher-timeframe bar is dropped unless all of its base bars
are present.

`request.security(..., lookahead_on)` is replaced by the previous completed
period, so a base bar inside period `p` reads the snapshot of `p - 1`. That is
exactly what upstream sees: its snapshot is `ta.valuewhen(...)[1]`, and the
lookahead call lets the last base bar of `p` refill the whole period with the
value of the already closed `p - 1`. `source_memory` caps how many pivots per side
are inspected, exactly like the upstream `for i = 0 to count - 1` loop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import boolean, integer, ohlcv, pivothigh, pivotlow, valuewhen
from ._tf import aligned_index, lift_previous_period, period_frame


def _utc_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    index = frame.index
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is None or index.hasnans
            or not index.is_monotonic_increasing or not index.is_unique):
        raise ValueError("索引必须为有时区、递增且唯一的 K 线开盘时间")
    return index.tz_convert("UTC")


def _base_minutes(utc: pd.DatetimeIndex) -> int:
    """Infer the base bar period from the finest gap between consecutive bars."""
    if len(utc) < 2:
        raise ValueError("至少需要两根 K 线才能推断基础周期")
    gaps = (utc.to_series().diff().dropna() / pd.Timedelta(minutes=1)).to_numpy()
    if gaps.min() <= 0:
        raise ValueError("K 线时间必须递增")
    return int(round(float(gaps.min())))


def _period(base: int, minutes: int) -> tuple[str, pd.Timedelta, int]:
    return f"{minutes}min", pd.Timedelta(minutes=minutes), minutes // base


def _levels(frame: pd.DataFrame, utc: pd.DatetimeIndex, base: int, minutes: int, *,
            pivot_left: int, pivot_right: int, source_memory: int) -> tuple[pd.Series, pd.Series]:
    """Lifted resistance / support candidates of one timeframe."""
    rule, step, expected = _period(base, minutes)
    higher = period_frame(frame, rule=rule, step=step, expected_bars=expected)
    highs = pivothigh(higher["high"], pivot_left, pivot_right)
    lows = pivotlow(higher["low"], pivot_left, pivot_right)
    upper = [lift_previous_period(valuewhen(highs.notna(), highs, order), utc, rule=rule, step=step)
             for order in range(source_memory)]
    lower = [lift_previous_period(valuewhen(lows.notna(), lows, order), utc, rule=rule, step=step)
             for order in range(source_memory)]
    resistance, support = pd.concat(upper, axis=1), pd.concat(lower, axis=1)
    resistance.index = support.index = frame.index
    return resistance, support


def _nearest(candidates: pd.DataFrame, reference: pd.Series, *, above: bool) -> pd.Series:
    values = candidates.to_numpy(dtype=float)
    reference_values = reference.to_numpy(dtype=float)[:, None]
    mask = values > reference_values if above else values < reference_values
    filled = np.where(mask, values, np.inf if above else -np.inf)
    best = filled.min(axis=1) if above else filled.max(axis=1)
    return pd.Series(np.where(mask.any(axis=1), best, np.nan), index=candidates.index)


def mtf_nearest_support_resistance(frame: pd.DataFrame, *, use_tf1: bool = True,
                                   tf1_minutes: int = 60, use_tf2: bool = True,
                                   tf2_minutes: int = 240, use_tf3: bool = True,
                                   tf3_minutes: int = 1440, pivot_left: int = 5,
                                   pivot_right: int = 5,
                                   source_memory: int = 6) -> pd.DataFrame:
    boolean("use_tf1", use_tf1)
    integer("tf1_minutes", tf1_minutes, 1)
    boolean("use_tf2", use_tf2)
    integer("tf2_minutes", tf2_minutes, 1)
    boolean("use_tf3", use_tf3)
    integer("tf3_minutes", tf3_minutes, 1)
    integer("pivot_left", pivot_left, 2, 20)
    integer("pivot_right", pivot_right, 2, 20)
    integer("source_memory", source_memory, 2, 6)
    close = ohlcv(frame, "open", "high", "low", "close")[-1]

    utc = _utc_index(frame)
    base = _base_minutes(utc)
    aligned_index(frame, base)
    for name, minutes in (("tf1_minutes", tf1_minutes), ("tf2_minutes", tf2_minutes),
                          ("tf3_minutes", tf3_minutes)):
        if minutes < base or minutes % base:
            raise ValueError(f"参数 {name} 必须是基础周期 {base} 分钟的整数倍")

    # A timeframe is skipped when it repeats one that is already in use upstream.
    active1 = use_tf1
    active2 = use_tf2 and (not active1 or tf2_minutes != tf1_minutes)
    active3 = (use_tf3 and (not active1 or tf3_minutes != tf1_minutes)
               and (not active2 or tf3_minutes != tf2_minutes))

    slots = ((active1, tf1_minutes), (active2, tf2_minutes), (active3, tf3_minutes))
    empty = pd.Series(np.nan, index=frame.index)
    columns: dict[str, pd.Series] = {}
    for slot_index, (active, minutes) in enumerate(slots, start=1):
        if not active:
            columns[f"resistance_{slot_index}"] = empty
            columns[f"support_{slot_index}"] = empty
            continue
        highs, lows = _levels(frame, utc, base, minutes, pivot_left=pivot_left,
                              pivot_right=pivot_right, source_memory=source_memory)
        columns[f"resistance_{slot_index}"] = _nearest(highs, close, above=True)
        columns[f"support_{slot_index}"] = _nearest(lows, close, above=False)
    return pd.DataFrame(columns, index=frame.index)
