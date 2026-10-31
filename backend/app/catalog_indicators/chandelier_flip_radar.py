"""Chandelier Flip Radar — ratcheting ATR stops with a body-filtered flip.

Source: pine-collection/indicators/trend_direction/chandelier_flip_radar/chandelier_flip_radar.pine
(WavesUnchained, MIT)

Chandelier stops ratchet in the direction of the trend: the long stop is the
highest extremum minus `multiplier × ATR` and only moves up; the short stop
mirrors down. Extrema use close or high/low per `use_close`. The direction
flips when the close crosses the opposing stop **and** the candle body is at
least `body_filter` × ATR — dojis and inside bars cannot flip the state.
Three multiplier modes: fixed, "Conviction" (tightens in trends via a
fast/slow EMA trend-force and ATR percentile, widens in chop) and "K-means"
(a factor grid with performance memory; each instance ratchets its own stop,
performance is an EMA of signed bar moves against the stop direction, and the
selected cluster's factor feeds the multiplier). Traps mark wicks through the
trigger that held on close; weak flips mark close-throughs rejected by the
body filter. Warning/danger zones measure the ATR distance to the trigger.

Dropped: plots, fills, labels, colour bars, MTF confluence background.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (atr, boolean, choice, crossover, flags, highest, integer, lowest,
                    number, ohlcv, percentrank, sma, ema)


def _run_chandelier(frame: pd.DataFrame, atr_len: int, factor: float, use_close: bool,
                    perf_alpha: float) -> dict[str, np.ndarray]:
    """One Chandelier instance: ratcheting stops, direction and performance."""
    high, low, close = ohlcv(frame, "high", "low", "close")
    open_ = frame["open"].astype(float)
    rows = len(frame)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    opens = open_.to_numpy(dtype=float)
    atr_raw = atr(frame, atr_len).to_numpy(dtype=float)
    upper_raw = (highest(close, atr_len) if use_close else highest(high, atr_len)).to_numpy(dtype=float)
    lower_raw = (lowest(close, atr_len) if use_close else lowest(low, atr_len)).to_numpy(dtype=float)

    upper = lower = np.nan
    direction = 1
    output = np.nan
    performance = 0.0
    trend = 1
    out_dir = np.ones(rows, dtype=int)
    out_stop = np.full(rows, np.nan)
    out_performance = np.zeros(rows)
    out_factor = np.full(rows, factor)
    out_trend = np.ones(rows, dtype=int)
    alpha = 2.0 / (perf_alpha + 1.0)
    for position in range(rows):
        a = atr_raw[position]
        if np.isnan(a):
            continue
        up = upper_raw[position] - factor * a
        dn = lower_raw[position] + factor * a
        upper = up if np.isnan(upper) else (min(up, upper) if closes[position - 1] < upper else up) \
            if position > 0 else up
        lower = dn if np.isnan(lower) else (max(dn, lower) if closes[position - 1] > lower else dn) \
            if position > 0 else dn
        trend = 1 if closes[position] > upper else 0 if closes[position] < lower else trend
        diff = np.sign(closes[position - 1] - output) if position > 0 and not np.isnan(output) else 0.0
        performance += alpha * ((closes[position] - closes[position - 1]) * diff - performance)
        output = lower if trend == 1 else upper
        out_dir[position] = trend
        out_stop[position] = output
        out_performance[position] = performance
        out_factor[position] = factor
        out_trend[position] = trend
    del direction, out_dir
    return {"dir": out_trend, "stop": out_stop, "performance": out_performance,
            "factor": out_factor}


def chandelier_flip_radar(
        frame: pd.DataFrame, *, atr_len: int = 30, atr_mult: float = 4.5,
        use_close: bool = True, adapt_mode: str = "Off",
        ai_range: float = 0.5, ai_step: float = 0.5, perf_alpha: float = 10.0,
        ai_cluster: str = "Best", conv_adapt: float = 0.7, conv_fast: int = 5,
        conv_slow: int = 50, conv_norm_cap: float = 2.0, conv_chop_cut: float = 0.5,
        body_filter: float = 0.80, danger_dist: float = 0.35,
        warn_dist: float = 0.75, htf_minutes: int = 240) -> pd.DataFrame:
    integer("atr_len", atr_len, 1)
    number("atr_mult", atr_mult, 0.1)
    choice("adapt_mode", adapt_mode, ("Off", "Conviction", "K-means"))
    number("ai_range", ai_range, 0.5)
    number("ai_step", ai_step, 0.1)
    number("perf_alpha", perf_alpha, 2.0)
    choice("ai_cluster", ai_cluster, ("Best", "Average", "Worst"))
    number("conv_adapt", conv_adapt, 0.0, 2.0)
    integer("conv_fast", conv_fast, 2)
    integer("conv_slow", conv_slow, 5)
    number("conv_norm_cap", conv_norm_cap, 0.5)
    number("conv_chop_cut", conv_chop_cut, 0.05)
    number("body_filter", body_filter, 0.0)
    number("danger_dist", danger_dist, 0.05)
    number("warn_dist", warn_dist, 0.10)
    if warn_dist <= danger_dist:
        raise ValueError("参数 warn_dist 必须大于 danger_dist")
    high, low, close = ohlcv(frame, "high", "low", "close")
    open_ = frame["open"].astype(float)
    rows = len(frame)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    opens = open_.to_numpy(dtype=float)

    atr_raw = atr(frame, atr_len)
    ch_hi = (highest(close, atr_len) if use_close else highest(high, atr_len))
    ch_lo = (lowest(close, atr_len) if use_close else lowest(low, atr_len))

    # --- Multiplier selection ---
    if adapt_mode == "Conviction":
        conv_fast_ema = ema(close, conv_fast)
        conv_slow_ema = ema(close, conv_slow)
        conv_force = pd.Series(np.where(atr_raw > 0,
                                        (conv_fast_ema - conv_slow_ema).abs() / atr_raw, 0.0),
                               index=frame.index)
        conv_norm = (conv_force / conv_norm_cap).clip(upper=1.0)
        conv_chop = conv_force < conv_chop_cut
        atr_pct = percentrank(atr_raw, 100)
        atr_part = pd.Series(np.where(conv_force > conv_chop_cut,
                                      np.minimum(atr_pct / 100.0, 0.85), 0.50),
                             index=frame.index)
        conviction_base = (atr_part + conv_norm) / 2.0
        conviction = conviction_base.mask(conv_chop, conviction_base * 0.50)
        raw_mult = atr_mult * (1.0 + conv_adapt * (0.5 - conviction))
        dyn_mult = raw_mult.clip(lower=atr_mult * 0.60, upper=atr_mult * 1.80)
    elif adapt_mode == "K-means":
        factors = []
        f = atr_mult - ai_range
        while f <= atr_mult + ai_range + ai_step * 0.01:
            factors.append(round(f, 6))
            f += ai_step
        instances = [_run_chandelier(frame, atr_len, factor, use_close, perf_alpha)
                     for factor in factors]
        performance_matrix = pd.DataFrame(
            {factor: inst["performance"] for factor, inst in zip(factors, instances)},
            index=frame.index)
        if ai_cluster == "Best":
            chosen = performance_matrix.idxmax(axis=1)
        elif ai_cluster == "Worst":
            chosen = performance_matrix.idxmin(axis=1)
        else:
            chosen = performance_matrix.median(axis=1)
        dyn_mult = pd.Series(chosen.to_numpy(dtype=float), index=frame.index).fillna(atr_mult)
    else:
        dyn_mult = pd.Series(atr_mult, index=frame.index)

    atr_series = dyn_mult * atr_raw
    atr_values = atr_series.to_numpy(dtype=float)
    atr_raw_values = atr_raw.to_numpy(dtype=float)
    ch_hi_values = ch_hi.to_numpy(dtype=float)
    ch_lo_values = ch_lo.to_numpy(dtype=float)

    # Main state machine (fixed/multiplier path; K-means instances already ran).
    long_stop_raw = ch_hi_values - atr_values
    short_stop_raw = ch_lo_values + atr_values
    long_stop = np.full(rows, np.nan)
    short_stop = np.full(rows, np.nan)
    direction = np.ones(rows, dtype=int)
    current_dir = 1
    current_long = np.nan
    current_short = np.nan
    body_values = np.abs(closes - opens)
    body_ok = (body_filter <= 0.0) | (body_values >= atr_raw_values * body_filter)

    traps_bull = np.zeros(rows, dtype=bool)
    traps_bear = np.zeros(rows, dtype=bool)
    weak_long = np.zeros(rows, dtype=bool)
    weak_short = np.zeros(rows, dtype=bool)

    for position in range(rows):
        if np.isnan(atr_values[position]):
            continue
        long_raw = long_stop_raw[position]
        short_raw = short_stop_raw[position]
        long_prev = current_long if not np.isnan(current_long) else long_raw
        short_prev = current_short if not np.isnan(current_short) else short_raw
        current_long = max(long_raw, long_prev) if current_dir == 1 else long_raw
        current_short = min(short_raw, short_prev) if current_dir == -1 else short_raw
        if position > 0:
            flip_long = closes[position] > short_prev and body_ok[position]
            flip_short = closes[position] < long_prev and body_ok[position]
            new_dir = 1 if flip_long else -1 if flip_short else current_dir
            if current_dir == -1 and highs[position] > short_prev \
                    and closes[position] <= short_prev:
                traps_bull[position] = True
            if current_dir == 1 and lows[position] < long_prev \
                    and closes[position] >= long_prev:
                traps_bear[position] = True
            if current_dir == -1 and closes[position] > short_prev and not body_ok[position]:
                weak_long[position] = True
            if current_dir == 1 and closes[position] < long_prev and not body_ok[position]:
                weak_short[position] = True
            current_dir = new_dir
        long_stop[position] = current_long
        short_stop[position] = current_short
        direction[position] = current_dir

    direction_series = pd.Series(direction, index=frame.index)
    long_stop_series = pd.Series(long_stop, index=frame.index)
    short_stop_series = pd.Series(short_stop, index=frame.index)
    buy_signal = flags((direction_series == 1) & (direction_series.shift(1) == -1))
    sell_signal = flags((direction_series == -1) & (direction_series.shift(1) == 1))

    trigger_stop = pd.Series(np.where(direction == 1,
                                      pd.Series(long_stop, index=frame.index).shift(1),
                                      pd.Series(short_stop, index=frame.index).shift(1)),
                             index=frame.index)
    dist_atr = ((close - trigger_stop).abs() / atr_raw.replace(0, np.nan)).fillna(1e9)
    warn_level = max(warn_dist, danger_dist + 0.05)
    danger = flags(dist_atr < danger_dist)
    warning = flags((~danger) & (dist_atr < warn_level))

    ema5 = ema(close, 5)
    pullback_long = flags((direction_series == 1) & (close < ema5))
    pullback_short = flags((direction_series == -1) & (close > ema5))

    active_stop = pd.Series(np.where(direction == 1, long_stop, short_stop), index=frame.index)
    state = pd.Series(np.where(direction == 1,
                               np.where(pullback_long, -1, np.where(danger, 3,
                                                                    np.where(warning, 2, 4))),
                               np.where(pullback_short, 1, np.where(danger, -3,
                                                                    np.where(warning, -2, -4)))),
                      index=frame.index).astype("Int64")

    if adapt_mode == "K-means":
        kmeans_factor = pd.Series(
            _run_chandelier(frame, atr_len, float(dyn_mult.iloc[-1]), use_close,
                            perf_alpha)["factor"], index=frame.index)
    else:
        kmeans_factor = pd.Series(np.nan, index=frame.index)

    return pd.DataFrame({
        "direction": direction_series.astype("Int64"),
        "long_stop": long_stop_series,
        "short_stop": short_stop_series,
        "active_stop": active_stop,
        "buy_signal": buy_signal,
        "sell_signal": sell_signal,
        "bull_trap": flags(pd.Series(traps_bull, index=frame.index)),
        "bear_trap": flags(pd.Series(traps_bear, index=frame.index)),
        "weak_flip_long": flags(pd.Series(weak_long, index=frame.index)),
        "weak_flip_short": flags(pd.Series(weak_short, index=frame.index)),
        "state": state,
        "danger_zone": danger,
        "warning_zone": warning,
        "pullback_long": pullback_long,
        "pullback_short": pullback_short,
        "distance_atr": dist_atr.replace(1e9, np.nan),
        "atr": atr_raw,
        "dynamic_multiplier": dyn_mult,
        "kmeans_factor": kmeans_factor,
    }, index=frame.index)
