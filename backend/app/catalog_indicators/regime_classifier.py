"""Regime Classifier — ER/FDI/Choppiness blend into a Trend/Range/Chaos regime.

Source: pine-collection/indicators/trend_strength/regime_classifier/regime_classifier.pine
(WavesUnchained, MIT)

Three sensors — Kaufman efficiency ratio, a box-counting fractal dimension and
the choppiness index — are each mapped linearly from "full trend" to "full
range" onto 0..1, then weight-averaged into a trendiness score. The regime is
hysteretic: Trend enters at `trend_hi`, Range enters at `range_lo`, and in
between the previous state holds (the first bar decides at 0.5). A chaos
override fires only when all three sensors agree there is no exploitable
structure (very high chop + very low ER + very high FDI); while chaotic both
permission scores are damped by `chaos_damp`. The 4× higher-timeframe score is
recomputed the same way on aggregated HTF bars (previous completed period —
the lookahead-free reading of the upstream `request.security`).

Dropped: dashboard, per-sensor plots, regime-change markers, colours and the
debug log.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import boolean, flags, highest, integer, lowest, number, ohlcv, sum_
from ._pine import true_range
from ._tf import base_step_minutes, htf_frame, lift_series

_LOG2 = float(np.log(2.0))
_LOG10 = float(np.log(10.0))


def _regime_inputs(frame: pd.DataFrame, er_len: int, fdi_len: int, chop_len: int):
    high, low, close = ohlcv(frame, "high", "low", "close")
    closes = close.to_numpy(dtype=float)

    change = (close - close.shift(er_len)).abs()
    volume_ = sum_(close.diff().abs(), er_len)
    er = pd.Series(np.where(volume_ > 0.0, change / volume_, 0.0), index=frame.index)

    fdi_half = max(2, fdi_len // 2)
    range_now = (highest(high, fdi_half) - lowest(low, fdi_half)) / fdi_half
    range_past = (highest(high, fdi_half) - lowest(low, fdi_half)).shift(fdi_half) / fdi_half
    range_full = (highest(high, fdi_len) - lowest(low, fdi_len)) / fdi_len
    usable = ((range_now + range_past) > 0.0) & (range_full > 0.0)
    fdi = ((np.log(range_now + range_past) - np.log(range_full)) / _LOG2).where(usable)
    fdi = fdi.fillna(1.5).clip(lower=1.0, upper=2.0)

    tr_sum = sum_(true_range(frame), chop_len)
    chop_range = highest(high, chop_len) - lowest(low, chop_len)
    chop = pd.Series(np.where(chop_range > 0.0,
                              (100.0 * np.log10(tr_sum / chop_range) / np.log10(chop_len)).clip(0.0, 100.0),
                              50.0), index=frame.index).fillna(50.0)
    return er, fdi, chop


def _map(value: pd.Series, trend_at: float, range_at: float) -> pd.Series:
    span = trend_at - range_at
    if span == 0.0:
        return pd.Series(0.5, index=value.index)
    return ((value - range_at) / span).clip(lower=0.0, upper=1.0)


def regime_classifier(
        frame: pd.DataFrame, *, er_len: int = 20, fdi_len: int = 30,
        chop_len: int = 14, er_trend_at: float = 0.55, er_range_at: float = 0.20,
        fdi_trend_at: float = 1.35, fdi_range_at: float = 1.55,
        chop_trend_at: float = 38.2, chop_range_at: float = 61.8,
        w_er: float = 1.0, w_fdi: float = 1.0, w_chop: float = 1.0,
        trend_hi: float = 0.62, range_lo: float = 0.38, chop_chaos: float = 65.0,
        er_chaos: float = 0.18, fdi_chaos: float = 1.55, chaos_damp: float = 0.30,
        include_htf: bool = True) -> pd.DataFrame:
    integer("er_len", er_len, 2)
    integer("fdi_len", fdi_len, 6)
    integer("chop_len", chop_len, 2)
    number("er_trend_at", er_trend_at, 0.0, 1.0)
    number("er_range_at", er_range_at, 0.0, 1.0)
    number("fdi_trend_at", fdi_trend_at, 1.0, 2.0)
    number("fdi_range_at", fdi_range_at, 1.0, 2.0)
    number("chop_trend_at", chop_trend_at, 0.0, 100.0)
    number("chop_range_at", chop_range_at, 0.0, 100.0)
    number("trend_hi", trend_hi, 0.5, 1.0)
    number("range_lo", range_lo, 0.0, 0.5)
    number("chaos_damp", chaos_damp, 0.0, 1.0)
    boolean("include_htf", include_htf)

    er, fdi, chop = _regime_inputs(frame, er_len, fdi_len, chop_len)
    trend_score = (_map(er, er_trend_at, er_range_at) * w_er
                   + _map(fdi, fdi_trend_at, fdi_range_at) * w_fdi
                   + _map(chop, chop_trend_at, chop_range_at) * w_chop)
    weight_sum = w_er + w_fdi + w_chop
    if weight_sum > 0:
        trend_score = trend_score / weight_sum
    else:
        trend_score = pd.Series(0.5, index=frame.index)

    rows = len(frame)
    states = np.zeros(rows, dtype=int)
    current = 0
    scores = trend_score.to_numpy(dtype=float)
    for position in range(rows):
        value = scores[position]
        if value >= trend_hi:
            current = 1
        elif value <= range_lo:
            current = -1
        elif current == 0:
            current = 1 if value >= 0.5 else -1
        states[position] = current
    chaos = flags((chop >= chop_chaos) & (er <= er_chaos) & (fdi >= fdi_chaos))
    chaos_values = chaos.to_numpy(dtype=bool)
    labels = pd.Series(["Chaos" if chaos_values[i] else
                        ("Trend" if states[i] == 1 else "Range")
                        for i in range(rows)], index=frame.index, dtype=object)
    chaos_multiplier = pd.Series(np.where(chaos_values, chaos_damp, 1.0), index=frame.index)
    reversal_permission = (1.0 - trend_score) * chaos_multiplier
    trend_permission = trend_score * chaos_multiplier

    output = pd.DataFrame({
        "trend_score": trend_score,
        "efficiency_ratio": er,
        "fractal_dimension": fdi,
        "choppiness": chop,
        "regime_state": pd.Series(states, index=frame.index).astype("Int64"),
        "regime": labels,
        "chaos": chaos,
        "to_trend": flags(labels.eq("Trend") & ~labels.shift(1, fill_value="").eq("Trend")),
        "to_range": flags(labels.eq("Range") & ~labels.shift(1, fill_value="").eq("Range")),
        "to_chaos": flags(labels.eq("Chaos") & ~labels.shift(1, fill_value="").eq("Chaos")),
        "reversal_permission": reversal_permission,
        "trend_permission": trend_permission,
    }, index=frame.index)

    if include_htf:
        base = base_step_minutes(frame)
        minutes = base * 4
        htf = htf_frame(frame, minutes)
        er_h, fdi_h, chop_h = _regime_inputs(htf, er_len, fdi_len, chop_len)
        htf_score = (_map(er_h, er_trend_at, er_range_at) * w_er
                     + _map(fdi_h, fdi_trend_at, fdi_range_at) * w_fdi
                     + _map(chop_h, chop_trend_at, chop_range_at) * w_chop) / weight_sum
        lifted = lift_series(frame, htf_score, minutes)
        output["htf_score"] = lifted
        output["htf_regime"] = pd.Series(
            np.where(lifted >= trend_hi, "Trend",
                     np.where(lifted <= range_lo, "Range", "Neutral")),
            index=frame.index, dtype=object)
    return output
