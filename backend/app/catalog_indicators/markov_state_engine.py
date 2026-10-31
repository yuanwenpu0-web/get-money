"""Markov State Engine — six-state regime classifier with a Markov forecast.

Source: pine-collection/indicators/trend_strength/markov_state_engine/markov_state_engine.pine
(WavesUnchained, MIT)

The raw state comes from an EMA context (close above/below with a
slope threshold in ATR units), ADX + efficiency for trend strength, an ATR
rank for chaos and joint ATR/BB-width ranks for compression: Up Trend, Up
Weak, Compression, Down Trend, Down Weak, Expansion/Chaos. A raw
classification must persist `min_dwell` bars before it is committed
(debounce). Over the `lookback` window the engine counts every historical
exit from the current context-state (compression contexts split into
up/down-flavoured variants 6/7), applies Laplace smoothing and derives the
next-state distribution: persistence, bull/bear/neutral probabilities, an
entropy-based predictability, a sample-factor reliability, the net forecast
and a hysteresis-gated stable bias. Edges require the bias plus confidence
and reliability thresholds; lean types label reversal/breakout/continuation.

Dropped: background, readout table, forecast plot and watch markers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (atr, choice, dmi, ema, flags, highest, integer, lowest, number, ohlcv,
                    sma, stdev)

_STATE_NAMES = {0: "Up Trend", 1: "Up Weak", 2: "Compression", 3: "Down Trend",
                4: "Down Weak", 5: "Expansion/Chaos"}
_CONTEXT_VARIANTS = {0: 0, 1: 1, 2: None, 3: 3, 4: 4, 5: 5}


def markov_state_engine(
        frame: pd.DataFrame, *, ema_len: int = 55, atr_len: int = 14,
        eff_len: int = 34, adx_len: int = 14, atr_rank_len: int = 100,
        bb_len: int = 20, trend_th: float = 22.0, eff_th: float = 38.0,
        chaos_rank: float = 70.0, comp_rank: float = 35.0, slope_th: float = 0.05,
        lookback: int = 500, laplace: float = 1.0, min_samples: int = 20,
        min_dwell: int = 2, conf_th: float = 55.0, rel_th: float = 40.0,
        bias_margin: float = 8.0) -> pd.DataFrame:
    integer("ema_len", ema_len, 10)
    integer("atr_len", atr_len, 5)
    integer("eff_len", eff_len, 10)
    integer("adx_len", adx_len, 5)
    integer("atr_rank_len", atr_rank_len, 20)
    integer("bb_len", bb_len, 5)
    integer("lookback", lookback, 50, 1500)
    integer("min_samples", min_samples, 5)
    integer("min_dwell", min_dwell, 1)
    number("bias_margin", bias_margin, 0.0, 100.0)
    high, low, close = ohlcv(frame, "high", "low", "close")
    rows = len(frame)

    ema_line = ema(close, ema_len)
    _, _, adx = dmi(frame, adx_len)
    atr_line = atr(frame, atr_len)
    ema_slope_norm = pd.Series(np.where(atr_line != 0,
                                        (ema_line - ema_line.shift(5)) / atr_line, 0.0),
                               index=frame.index)
    atr_low = lowest(atr_line, atr_rank_len)
    atr_high = highest(atr_line, atr_rank_len)
    atr_rank = pd.Series(np.where(atr_high != atr_low,
                                  (atr_line - atr_low) / (atr_high - atr_low) * 100.0, 50.0),
                         index=frame.index)
    bb_basis = sma(close, bb_len)
    bb_dev = stdev(close, bb_len)
    bb_width = pd.Series(np.where(bb_basis != 0, bb_dev / bb_basis * 100.0, 0.0),
                         index=frame.index)
    bb_low = lowest(bb_width, atr_rank_len)
    bb_high = highest(bb_width, atr_rank_len)
    bb_rank = pd.Series(np.where(bb_high != bb_low,
                                 (bb_width - bb_low) / (bb_high - bb_low) * 100.0, 50.0),
                        index=frame.index)

    cum_diff = close.diff().abs().cumsum()
    path = cum_diff - cum_diff.shift(eff_len)
    direct = (close - close.shift(eff_len - 1)).abs()
    eff = pd.Series(np.where(path > 0, (direct / path * 100.0).clip(0.0, 100.0), 0.0),
                    index=frame.index)

    up_context = flags((close > ema_line) & (ema_slope_norm > slope_th))
    down_context = flags((close < ema_line) & (ema_slope_norm < -slope_th))
    high_trend = flags((adx > trend_th) & (eff > eff_th))
    high_chaos = flags((atr_rank > chaos_rank) & (eff < eff_th))
    compression = flags((atr_rank < comp_rank) & (bb_rank < comp_rank) & (eff < eff_th))

    raw_state = pd.Series(
        np.where(high_chaos, 5,
                 np.where(compression, 2,
                          np.where(up_context & high_trend, 0,
                                   np.where(down_context & high_trend, 3,
                                            np.where(up_context, 1,
                                                     np.where(down_context, 4, 2)))))),
        index=frame.index)

    # Debounce: commit the raw state after it persists `min_dwell` bars.
    raw_values = raw_state.to_numpy(dtype=float)
    states = np.full(rows, 2)
    current = 2
    run = 0
    previous = np.nan
    for position in range(rows):
        run = 1 if (np.isnan(previous) or raw_values[position] != previous) else run + 1
        if run >= min_dwell:
            current = int(raw_values[position])
        states[position] = current
        previous = raw_values[position]
    state_series = pd.Series(states, index=frame.index)

    up_values = up_context.to_numpy(dtype=bool)
    down_values = down_context.to_numpy(dtype=bool)
    context_states = states.copy()
    for position in range(rows):
        if states[position] == 2:
            if up_values[position]:
                context_states[position] = 6
            elif down_values[position]:
                context_states[position] = 7

    p_bull = np.full(rows, np.nan)
    p_bear = np.full(rows, np.nan)
    p_neutral = np.full(rows, np.nan)
    persistence = np.full(rows, np.nan)
    reliability = np.full(rows, np.nan)
    net_forecast = np.full(rows, np.nan)
    next_state = np.full(rows, -1, dtype=int)
    bias_values = np.zeros(rows, dtype=int)
    stable_bias = 0
    window = min(lookback, max(rows - 1, 0))
    for position in range(rows):
        current_context = context_states[position]
        counts = np.zeros(6)
        self_count = 0
        start = max(0, position - window)
        for history in range(start, position):
            if context_states[history] == current_context:
                successor = states[history]  # state one bar later equals this bar's state
                if states[position] == successor and history + 1 == position:
                    pass
                # Pine: toState = state[i - 1] — the state *before* the same-context bar.
                before = states[max(history - 1, 0)] if history >= 1 else states[0]
                if before == states[position] and history + 1 == position:
                    self_count += 1
                elif before in (0, 1, 2, 3, 4, 5) and before != states[position]:
                    counts[before] += 1
        total = counts.sum()
        total_all = total + self_count
        persistence[position] = total_all / total_all * 100 if total_all > 0 else np.nan
        denom = total + 5 * laplace
        probabilities = np.array([
            0.0 if states[position] == k else (counts[k] + laplace) / denom * 100.0
            for k in range(6)])
        bull = probabilities[0] + probabilities[1]
        bear = probabilities[3] + probabilities[4]
        neutral = probabilities[2] + probabilities[5]
        p_bull[position] = bull
        p_bear[position] = bear
        p_neutral[position] = neutral
        next_state[position] = int(np.argmax(probabilities))
        net_forecast[position] = bull - bear
        terms = []
        for share in (bull, bear, neutral):
            q = share / 100.0
            terms.append(-q * np.log(q) if q > 0 else 0.0)
        entropy = sum(terms)
        predictability = min(max(1.0 - entropy / np.log(3.0), 0.0), 1.0)
        sample_factor = min(1.0, total / min_samples)
        reliability[position] = sample_factor * predictability * 100.0

        if bull - max(bear, neutral) >= bias_margin:
            stable_bias = 1
        elif bear - max(bull, neutral) >= bias_margin:
            stable_bias = -1
        elif neutral - max(bull, bear) >= bias_margin:
            stable_bias = 0
        bias_values[position] = stable_bias

    bias_series = pd.Series(bias_values, index=frame.index)
    directional_prob = pd.Series(np.maximum(p_bull, np.maximum(p_bear, p_neutral)),
                                 index=frame.index)
    confident = flags(directional_prob >= conf_th)
    reliable = flags(pd.Series(reliability, index=frame.index) >= rel_th)
    bull_edge = flags((bias_series == 1) & confident & reliable & (state_series != 0))
    bear_edge = flags((bias_series == -1) & confident & reliable & (state_series != 3))

    lean_type = pd.Series("—", index=frame.index, dtype=object)
    lean_type = lean_type.mask(
        (bias_series == 1) & state_series.isin([3, 4]), "Reversal")
    lean_type = lean_type.mask(
        (bias_series == 1) & state_series.isin([2, 5]), "Breakout")
    lean_type = lean_type.mask(
        (bias_series == 1) & (state_series == 1), "Continuation")
    lean_type = lean_type.mask(
        (bias_series == -1) & state_series.isin([0, 1]), "Reversal")
    lean_type = lean_type.mask(
        (bias_series == -1) & state_series.isin([2, 5]), "Breakout")
    lean_type = lean_type.mask(
        (bias_series == -1) & (state_series == 4), "Continuation")

    regime_changed = flags(state_series != state_series.shift(1))
    bars_in_state = pd.Series(np.arange(rows), index=frame.index) - \
        pd.Series(np.arange(rows), index=frame.index).where(regime_changed).ffill().fillna(0)
    persistence_frac = pd.Series(persistence, index=frame.index) / 100.0
    typical_dwell = pd.Series(
        np.where((persistence_frac < 0.999) & (persistence_frac > 0),
                 1.0 / (1.0 - persistence_frac), np.nan), index=frame.index)
    dwell_ratio = typical_dwell > 0 if False else \
        pd.Series(np.where(typical_dwell.notna() & (typical_dwell > 0),
                           bars_in_state / typical_dwell, np.nan), index=frame.index)
    maturity = pd.Series("—", index=frame.index, dtype=object)
    maturity = maturity.mask(dwell_ratio >= 2.0, "Old")
    maturity = maturity.mask(dwell_ratio >= 1.0, "Typical")
    maturity = maturity.mask(dwell_ratio.notna() & (dwell_ratio < 1.0), "Young")

    return pd.DataFrame({
        "state": state_series.astype("Int64"),
        "state_name": state_series.map(_STATE_NAMES).astype(object),
        "context_state": pd.Series(context_states, index=frame.index).astype("Int64"),
        "state_changed": regime_changed,
        "bars_in_state": bars_in_state.astype("Int64"),
        "persistence": pd.Series(persistence, index=frame.index),
        "p_bull": pd.Series(p_bull, index=frame.index),
        "p_bear": pd.Series(p_bear, index=frame.index),
        "p_neutral": pd.Series(p_neutral, index=frame.index),
        "next_state": pd.Series(next_state, index=frame.index).astype("Int64"),
        "net_forecast": pd.Series(net_forecast, index=frame.index),
        "reliability": pd.Series(reliability, index=frame.index),
        "directional_probability": directional_prob,
        "bias": bias_series.astype("Int64"),
        "bull_edge": bull_edge,
        "bear_edge": bear_edge,
        "lean_type": lean_type,
        "maturity": maturity,
        "dwell_ratio": dwell_ratio,
        "efficiency": eff,
        "atr_rank": atr_rank,
    }, index=frame.index)
