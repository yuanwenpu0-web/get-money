"""Trend Persistence Score — four-sensor blend scoring how persistent a trend is.

Source: pine-collection/indicators/trend_strength/trend_persistence_score/trend_persistence_score.pine
(WavesUnchained, MIT)

Four sensors, each mapped to 0..100 and weight-averaged: the R² of price
against bar index (linear fit quality), the Kaufman efficiency ratio, an ADX
composite (70% level normalised between `adx_lo`/`adx_hi` + 30% normalised ADX
slope, damped when both R² and ER fall below `dead_thresh`) and a path/range
fractal dimension score between `fdi_trend`/`fdi_range` anchors. The blend is
EMA-smoothed into the TPS and bucketed into Strong/Healthy/Transition/Weak/
Dead states with an asymmetric hysteresis: upgrades are immediate, downgrades
need the TPS to fall 3 points below the state's edge. Transition risk blends
the shortfall from a full score (70%) with low efficiency (30%); event flags
fire on crossings of the Healthy/Transition levels.

Dropped: colours, background zones, heat columns, info label and plots.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (atr, choice, correlation, crossover, crossunder, dmi, ema, flags,
                    highest, integer, lowest, number, ohlcv, percentrank, sma, stdev, sum_)


def _normalize(value: pd.Series, lo: float, hi: float) -> pd.Series:
    if hi == lo:
        return pd.Series(0.0, index=value.index)
    return ((value - lo) / (hi - lo)).clip(lower=0.0, upper=1.0)


def trend_persistence_score(
        frame: pd.DataFrame, *, source: str = "close", length: int = 34,
        adx_len: int = 14, smooth_len: int = 5, w_r2: float = 40.0,
        w_er: float = 25.0, w_adx: float = 20.0, w_fdi: float = 15.0,
        adx_lo: float = 12.0, adx_hi: float = 35.0, fdi_trend: float = 1.20,
        fdi_range: float = 1.65, dead_thresh: float = 20.0, adx_damp: float = 0.35,
        lvl_strong: int = 75, lvl_healthy: int = 60, lvl_trans: int = 45,
        lvl_weak: int = 30, transition_lead: float = 3.0,
        compression_lookback: int = 100) -> pd.DataFrame:
    choice("source", source, ("close", "hlc3", "hl2", "ohlc4"))
    integer("length", length, 10)
    integer("adx_len", adx_len, 5)
    integer("smooth_len", smooth_len, 1)
    integer("lvl_strong", lvl_strong, 0, 100)
    integer("lvl_healthy", lvl_healthy, 0, 100)
    integer("lvl_trans", lvl_trans, 0, 100)
    integer("lvl_weak", lvl_weak, 0, 100)
    number("transition_lead", transition_lead, 0.0)
    high, low, close = ohlcv(frame, "high", "low", "close")
    src = {"close": close, "hlc3": (high + low + close) / 3,
           "hl2": (high + low) / 2,
           "ohlc4": (high + low + close + frame["open"].astype(float)) / 4}[source]
    bar_index = pd.Series(np.arange(len(frame), dtype=float), index=frame.index)

    # --- Sensor 1: R² of price vs time ---
    corr = correlation(src, bar_index, length).fillna(0.0)
    r2_score = corr.pow(2.0) * 100.0

    # --- Sensor 2: efficiency ratio ---
    change = (src - src.shift(length)).abs()
    volatility = sum_(src.diff().abs(), length)
    er_raw = pd.Series(np.where(volatility == 0.0, 0.0, change / volatility),
                       index=frame.index)
    er_score = er_raw * 100.0

    # --- Sensor 3: ADX composite ---
    _, _, adx = dmi(frame, adx_len)
    adx_strength = _normalize(adx, adx_lo, adx_hi)
    adx_slope = ema(adx - adx.shift(1), 3)
    adx_slope_norm = _normalize(adx_slope, -1.0, 1.5)
    adx_score = (adx_strength * 0.7 + adx_slope_norm * 0.3) * 100.0
    structure_dead = flags((r2_score < dead_thresh) & (er_score < dead_thresh))
    adx_final = pd.Series(np.where(structure_dead, adx_score * adx_damp, adx_score),
                          index=frame.index)

    # --- Sensor 4: path/range fractal dimension ---
    highest_high = highest(high, length)
    lowest_low = lowest(low, length)
    range_hl = highest_high - lowest_low
    path_length = sum_(src.diff().abs(), length)
    fdi_raw = pd.Series(
        np.where((range_hl > 0.0) & (path_length > 0.0),
                 np.log(path_length / range_hl) / np.log(length) + 1.0, 1.5),
        index=frame.index)
    fdi_trend_score = (1.0 - _normalize(fdi_raw, fdi_trend, fdi_range)) * 100.0

    weight_sum = w_r2 + w_er + w_adx + w_fdi
    if weight_sum > 0:
        tps_raw = (r2_score * w_r2 + er_score * w_er + adx_final * w_adx
                   + fdi_trend_score * w_fdi) / weight_sum
    else:
        tps_raw = pd.Series(0.0, index=frame.index)
    tps = ema(tps_raw, smooth_len)

    transition_risk = (100.0 - tps) * 0.7 + (100.0 - er_score) * 0.3

    # State machine with asymmetric hysteresis.
    def state_index(values: pd.Series) -> pd.Series:
        return pd.Series(np.where(values >= lvl_strong, 4,
                                  np.where(values >= lvl_healthy, 3,
                                           np.where(values >= lvl_trans, 2,
                                                    np.where(values >= lvl_weak, 1, 0)))),
                         index=values.index)

    raw_idx = state_index(tps)
    rows = len(frame)
    raw_values = raw_idx.to_numpy(dtype=float)
    tps_values = tps.to_numpy(dtype=float)
    edges = {4: float(lvl_strong), 3: float(lvl_healthy), 2: float(lvl_trans),
             1: float(lvl_weak)}
    states = np.zeros(rows, dtype=int)
    state = 0
    for position in range(rows):
        raw = int(raw_values[position])
        if np.isnan(tps_values[position]):
            states[position] = state
            continue
        if raw > state:
            state = raw
        elif raw < state:
            edge_level = edges.get(state, float(lvl_weak))
            if tps_values[position] < edge_level - 3.0:
                state = raw
        states[position] = state
    state_series = pd.Series(states, index=frame.index)
    state_names = {4: "Trend Strong", 3: "Trend Healthy", 2: "Transition",
                   1: "Weak / Range", 0: "Trend Dead"}

    dir_up = flags(corr > 0.15)
    dir_down = flags(corr < -0.15)
    basis = sma(src, length)
    deviation = stdev(src, length)
    bb_width = pd.Series(np.where(basis == 0.0, 0.0, 4.0 * deviation / basis),
                         index=frame.index)
    compress = flags(percentrank(bb_width, compression_lookback) < 30)

    weakening = crossunder(tps, float(lvl_healthy))
    transition_hard = crossunder(tps, float(lvl_trans))
    transition_lead_event = (crossunder(tps, float(lvl_trans) + transition_lead)
                             & (tps > lvl_trans) & (tps < tps.shift(1))
                             & (transition_risk > transition_risk.shift(1)))
    transition_event = flags(transition_hard | transition_lead_event)
    recovery = crossover(tps, float(lvl_healthy))

    driver_values = pd.concat([r2_score, er_score, adx_final, fdi_trend_score], axis=1)
    driver_names = driver_values.idxmax(axis=1).map(
        {0: "Fit", 1: "Efficiency", 2: "ADX building", 3: "Clean path"})
    driver_names = driver_names.mask(structure_dead & (driver_names == "ADX building"),
                                     "ADX damped")

    return pd.DataFrame({
        "tps": tps,
        "state": state_series.astype("Int64"),
        "state_name": state_series.map(state_names).astype(object),
        "direction_up": dir_up,
        "direction_down": dir_down,
        "compressed": compress,
        "r2_score": r2_score,
        "er_score": er_score,
        "adx_score": adx_final,
        "fdi_score": fdi_trend_score,
        "fractal_dimension": fdi_raw,
        "structure_dead": structure_dead,
        "transition_risk": transition_risk,
        "weakening_event": flags(weakening),
        "transition_event": transition_event,
        "recovery_event": flags(recovery),
        "driver": driver_names,
    }, index=frame.index)
