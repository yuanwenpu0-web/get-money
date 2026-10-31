"""Candle Story Engine [WavesUnchained]  v1.10.2  —  Python port.

Interprets candle behaviour as a weighted multi-candle sequence (pressure,
quality, dominance, persistence, deceleration, exhaustion) instead of relying
on isolated candlestick patterns.

Computational core ported (per-bar, stateful where required by the source):
  * Candle pressure / quality (-100..+100) from close position, body & wick balance
  * Recency-weighted multi-candle sequence engine (pressure, dominance, persistence)
  * Deceleration engine (older vs recent progress-per-range)
  * Range compression (short vs long SMA) with hysteresis armed state
  * Compression-zone breakout story (attempt / accepted / trap / unresolved)
  * Classic candle patterns as supporting evidence (engulfing / pin / star / piercing / harami)
  * Structural candle roles (marubozu / doji variants / spinning top)
  * Liquidity sweep detection (wick pierces N-bar extreme, close recovers inside)
  * Directional evidence, exhaustion, fresh BUY/SELL signal hysteresis
  * Follow-through & signal-failure on the bar after a fresh signal

Dropped (display / logging / non-computational):
  * box / line / label drawing, table, barcolor, alertcondition, plot
  * HTF request.security overlays (none present in this source)

All bars are treated as *confirmed* (we only process historical/closed bars);
the `confirm_on_close` input therefore has no effect on the returned series.

Note: ``breakout_story_event`` is intentionally *sticky* (the source declares it
as a persistent ``var int`` and only rewrites it on a breakout event), so once a
breakout is accepted/trapped/unresolved the code stays until the next attempt.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from ._pine import atr, boolean, ema, highest, integer, lowest, number, ohlcv, sma

_COLUMNS = [
    # ---- scalar evidence / state (float) ----
    "candle_pressure", "sequence_pressure", "bull_quality", "bear_quality",
    "bull_dominance", "bear_dominance", "close_persistence_bull", "close_persistence_bear",
    "bull_deceleration", "bear_deceleration", "bull_exhaustion", "bear_exhaustion",
    "compression_score", "candle_role_code", "trend_score", "range_atr_ratio",
    "volume_ratio", "bull_evidence", "bear_evidence", "directional_balance",
    # ---- signed / boolean event codes (Int64) ----
    "range_expansion", "compression_active", "breakout_attempt", "breakout_story_event",
    "liquidity_sweep", "bull_signal", "bear_signal", "bull_signal_state", "bear_signal_state",
    "bull_exhausted", "bear_exhausted", "bull_follow_through", "bear_follow_through",
    "bull_failure", "bear_failure", "strong_bull", "strong_bear", "bull_control", "bear_control",
    "indecision", "bull_pattern", "bear_pattern", "bull_liquidity_sweep", "bear_liquidity_sweep",
    "doji", "bull_engulfing", "bear_engulfing", "bull_pin_bar", "bear_pin_bar",
    "morning_star", "evening_star", "bull_marubozu", "bear_marubozu",
    "dragonfly_doji", "gravestone_doji", "spinning_top",
]


def candle_story_engine(
    frame: pd.DataFrame,
    *,
    sequence_length: int = 5,
    signal_threshold: float = 65.0,
    strong_threshold: float = 78.0,
    minimum_signal_change: float = 5.0,
    signal_exit_buffer: float = 15.0,
    confirm_on_close: bool = True,
    atr_length: int = 14,
    expansion_threshold: float = 1.25,
    compression_short: int = 3,
    compression_long: int = 10,
    minimum_compression_bars: int = 3,
    breakout_resolution_bars: int = 3,
    use_trend_context: bool = True,
    trend_ma_length: int = 34,
    trend_slope_length: int = 5,
    trend_weight: float = 20.0,
    use_volume: bool = True,
    volume_length: int = 20,
    volume_influence: float = 0.15,
    exhaustion_threshold: float = 65.0,
    exhaustion_trend_min: float = 25.0,
    wick_exhaustion_weight: float = 30.0,
    body_contraction_weight: float = 25.0,
    use_patterns: bool = True,
    pattern_weight: float = 10.0,
    use_extended_pattern_evidence: bool = False,
    pin_bar_wick_min: float = 0.60,
    pin_bar_close_min: float = 0.66,
    pin_bar_body_max: float = 0.35,
    strong_pattern_body_min: float = 0.55,
    harami_body_max: float = 0.60,
    pattern_validation_bars: int = 3,
    doji_body_max: float = 0.10,
    marubozu_body_min: float = 0.85,
    marubozu_wick_max: float = 0.075,
    spinning_top_body_max: float = 0.30,
    spinning_top_wick_min: float = 0.25,
    use_liquidity_sweep: bool = True,
    sweep_lookback: int = 10,
    sweep_wick_min: float = 0.50,
    sweep_close_back_min: float = 0.55,
    mintick: float = 1e-8,
) -> pd.DataFrame:
    open_, high, low, close = ohlcv(frame, "open", "high", "low", "close")
    # Every parameter is validated here (not in the default) so non-default
    # calls are also checked, mirroring Pine's `input.*` boundaries.
    sequence_length = integer("sequence_length", sequence_length, 2, 20)
    signal_threshold = number("signal_threshold", signal_threshold, 40.0, 95.0)
    strong_threshold = number("strong_threshold", strong_threshold, 50.0, 100.0)
    minimum_signal_change = number("minimum_signal_change", minimum_signal_change, 0.0, 30.0)
    signal_exit_buffer = number("signal_exit_buffer", signal_exit_buffer, 0.0, 40.0)
    confirm_on_close = boolean("confirm_on_close", confirm_on_close)
    atr_length = integer("atr_length", atr_length, 2, 100)
    expansion_threshold = number("expansion_threshold", expansion_threshold, 0.5, 5.0)
    compression_short = integer("compression_short", compression_short, 2, 10)
    compression_long = integer("compression_long", compression_long, 5, 50)
    minimum_compression_bars = integer("minimum_compression_bars", minimum_compression_bars, 2, 20)
    breakout_resolution_bars = integer("breakout_resolution_bars", breakout_resolution_bars, 1, 10)
    use_trend_context = boolean("use_trend_context", use_trend_context)
    trend_ma_length = integer("trend_ma_length", trend_ma_length, 5, 300)
    trend_slope_length = integer("trend_slope_length", trend_slope_length, 1, 30)
    trend_weight = number("trend_weight", trend_weight, 0.0, 40.0)
    use_volume = boolean("use_volume", use_volume)
    volume_length = integer("volume_length", volume_length, 2, 200)
    volume_influence = number("volume_influence", volume_influence, 0.0, 0.50)
    exhaustion_threshold = number("exhaustion_threshold", exhaustion_threshold, 40.0, 95.0)
    exhaustion_trend_min = number("exhaustion_trend_min", exhaustion_trend_min, 0.0, 100.0)
    wick_exhaustion_weight = number("wick_exhaustion_weight", wick_exhaustion_weight, 0.0, 50.0)
    body_contraction_weight = number("body_contraction_weight", body_contraction_weight, 0.0, 50.0)
    use_patterns = boolean("use_patterns", use_patterns)
    pattern_weight = number("pattern_weight", pattern_weight, 0.0, 30.0)
    use_extended_pattern_evidence = boolean("use_extended_pattern_evidence", use_extended_pattern_evidence)
    pin_bar_wick_min = number("pin_bar_wick_min", pin_bar_wick_min, 0.40, 0.90)
    pin_bar_close_min = number("pin_bar_close_min", pin_bar_close_min, 0.50, 0.90)
    pin_bar_body_max = number("pin_bar_body_max", pin_bar_body_max, 0.10, 0.50)
    strong_pattern_body_min = number("strong_pattern_body_min", strong_pattern_body_min, 0.30, 0.90)
    harami_body_max = number("harami_body_max", harami_body_max, 0.20, 0.90)
    pattern_validation_bars = integer("pattern_validation_bars", pattern_validation_bars, 1, 10)
    doji_body_max = number("doji_body_max", doji_body_max, 0.02, 0.25)
    marubozu_body_min = number("marubozu_body_min", marubozu_body_min, 0.60, 0.98)
    marubozu_wick_max = number("marubozu_wick_max", marubozu_wick_max, 0.01, 0.20)
    spinning_top_body_max = number("spinning_top_body_max", spinning_top_body_max, 0.11, 0.50)
    spinning_top_wick_min = number("spinning_top_wick_min", spinning_top_wick_min, 0.05, 0.45)
    use_liquidity_sweep = boolean("use_liquidity_sweep", use_liquidity_sweep)
    sweep_lookback = integer("sweep_lookback", sweep_lookback, 3, 50)
    sweep_wick_min = number("sweep_wick_min", sweep_wick_min, 0.30, 0.90)
    sweep_close_back_min = number("sweep_close_back_min", sweep_close_back_min, 0.50, 0.90)

    idx = frame.index
    n = len(frame)
    _ = confirm_on_close  # all processed bars are historical/confirmed

    def clamp(v, lo, hi):
        return np.clip(v, lo, hi)

    # ---- base per-bar scalars (vectorised) ----
    o = open_.to_numpy(dtype=float)
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    rng = np.maximum(h - l, mintick)
    body = np.abs(c - o)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    body_pct = body / rng
    upper_wick_pct = upper_wick / rng
    lower_wick_pct = lower_wick / rng
    close_pos = (c - l) / rng
    bull_candle = c > o
    bear_candle = c < o
    direction = np.where(bull_candle, 1.0, np.where(bear_candle, -1.0, 0.0))
    doji = body_pct <= doji_body_max

    atr_s = atr(frame, atr_length).to_numpy(dtype=float)
    range_atr_ratio = np.where(atr_s > 0, rng / np.where(atr_s > 0, atr_s, 1.0), 1.0)
    range_expansion = range_atr_ratio >= expansion_threshold

    # ---- volume context ----
    if "volume" in frame.columns:
        vol = pd.to_numeric(frame["volume"], errors="raise").astype(float)
        vol_avg = sma(vol, volume_length)
        cum = vol.cumsum()
        has_vol = (cum > 0) & vol.notna() & vol_avg.notna() & (vol_avg > 0)
        vol_ratio = np.where(has_vol.to_numpy(),
                             (vol / vol_avg).to_numpy(),
                             1.0)
        va = np.where(use_volume,
                      clamp(1.0 + (vol_ratio - 1.0) * volume_influence,
                            1.0 - volume_influence, 1.0 + volume_influence),
                      1.0)
    else:
        vol_ratio = np.ones(n)
        va = np.ones(n)

    # ---- candle pressure / quality ----
    close_pressure = (close_pos - 0.5) * 2.0 * 45.0
    body_pressure = direction * body_pct * 40.0
    wick_pressure = (lower_wick_pct - upper_wick_pct) * 15.0
    raw_candle_pressure = close_pressure + body_pressure + wick_pressure
    candle_pressure = clamp(raw_candle_pressure * va, -100.0, 100.0)

    bull_quality = clamp(body_pct * 35.0 + close_pos * 40.0 + lower_wick_pct * 15.0
                         + clamp(range_atr_ratio, 0.0, 1.5) / 1.5 * 10.0, 0.0, 100.0)
    bear_quality = clamp(body_pct * 35.0 + (1.0 - close_pos) * 40.0 + upper_wick_pct * 15.0
                         + clamp(range_atr_ratio, 0.0, 1.5) / 1.5 * 10.0, 0.0, 100.0)

    # ---- trend context ----
    trend_ma = ema(close, trend_ma_length).to_numpy(dtype=float)
    ts = pd.Series(trend_ma, index=idx).shift(trend_slope_length).to_numpy(dtype=float)
    atr_pos = atr_s > 0
    trend_slope_raw = np.where(atr_pos & ~np.isnan(ts), (trend_ma - ts) / np.where(atr_pos, atr_s, 1.0), 0.0)
    trend_slope_score = clamp(trend_slope_raw * 35.0, -100.0, 100.0)
    price_trend_dist = np.where(atr_pos, (c - trend_ma) / np.where(atr_pos, atr_s, 1.0), 0.0)
    price_trend_score = clamp(price_trend_dist * 35.0, -100.0, 100.0)
    if use_trend_context:
        trend_score = clamp(trend_slope_score * 0.55 + price_trend_score * 0.45, -100.0, 100.0)
    else:
        trend_score = np.zeros(n)
    bull_trend = trend_score > exhaustion_trend_min
    bear_trend = trend_score < -exhaustion_trend_min

    # ---- range compression (pre-hysteresis) ----
    candle_range_s = pd.Series(rng, index=idx)
    short_avg = sma(candle_range_s, compression_short).to_numpy(dtype=float)
    long_avg = sma(candle_range_s, compression_long).to_numpy(dtype=float)
    comp_ratio = np.where(long_avg > 0, short_avg / np.where(long_avg > 0, long_avg, 1.0), 1.0)
    comp_base = clamp((1.0 - comp_ratio) * 150.0, 0.0, 70.0)
    inside_bar = (h <= np.roll(h, 1)) & (l >= np.roll(l, 1))
    inside_bar[0] = False
    narrower = rng < np.roll(rng, 1)
    narrower[0] = False
    compression_score = clamp(comp_base
                              + np.where(inside_bar, 20.0, 0.0)
                              + np.where(narrower, 10.0, 0.0), 0.0, 100.0)

    # ---- classic patterns (vectorised via shifts) ----
    o1, h1, l1, c1 = np.roll(o, 1), np.roll(h, 1), np.roll(l, 1), np.roll(c, 1)
    o2, c2, op2 = np.roll(o, 2), np.roll(c, 2), np.roll(open_, 2)
    bp1 = np.roll(body_pct, 1)
    # indices 0,1 have no history -> disable
    for arr in (o1, h1, l1, c1, o2, c2, op2, bp1):
        arr[:2] = np.nan
    bull_engulf = bull_candle & (c1 < o1) & (o <= c1) & (c >= o1)
    bear_engulf = bear_candle & (c1 > o1) & (o >= c1) & (c <= o1)
    bull_pin = (~doji) & (body_pct <= pin_bar_body_max) & (lower_wick_pct >= pin_bar_wick_min) \
        & (upper_wick_pct <= 1.0 - pin_bar_close_min) & (close_pos >= pin_bar_close_min)
    bear_pin = (~doji) & (body_pct <= pin_bar_body_max) & (upper_wick_pct >= pin_bar_wick_min) \
        & (lower_wick_pct <= 1.0 - pin_bar_close_min) & (close_pos <= 1.0 - pin_bar_close_min)
    morning_star = (c2 < op2) & (np.abs(c1 - o1) < np.abs(c2 - op2) * 0.55) & bull_candle \
        & (c > (op2 + c2) / 2.0)
    evening_star = (c2 > op2) & (np.abs(c1 - o1) < np.abs(c2 - op2) * 0.55) & bear_candle \
        & (c < (op2 + c2) / 2.0)
    prior_bull_strong = bull_candle & (bp1 >= strong_pattern_body_min)
    prior_bear_strong = bear_candle & (bp1 >= strong_pattern_body_min)
    bull_piercing = prior_bear_strong & bull_candle & (body_pct >= strong_pattern_body_min) \
        & (o <= c1) & (c > (o1 + c1) / 2.0) & (c < o1)
    bear_dark_cloud = prior_bull_strong & bear_candle & (body_pct >= strong_pattern_body_min) \
        & (o >= c1) & (c < (o1 + c1) / 2.0) & (c > o1)
    bull_harami = prior_bear_strong & bull_candle & (body <= np.roll(body, 1) * harami_body_max) \
        & (np.minimum(o, c) >= np.minimum(o1, c1)) & (np.maximum(o, c) <= np.maximum(o1, c1))
    bear_harami = prior_bull_strong & bear_candle & (body <= np.roll(body, 1) * harami_body_max) \
        & (np.minimum(o, c) >= np.minimum(o1, c1)) & (np.maximum(o, c) <= np.maximum(o1, c1))
    bull_core = bull_engulf | bull_pin | morning_star
    bear_core = bear_engulf | bear_pin | evening_star
    bull_ext = bull_piercing | bull_harami
    bear_ext = bear_dark_cloud | bear_harami
    bull_pattern = bull_core | (use_extended_pattern_evidence & bull_ext)
    bear_pattern = bear_core | (use_extended_pattern_evidence & bear_ext)
    pattern_evidence_bull = np.where(use_patterns & bull_pattern & (not use_trend_context or bear_trend),
                                     pattern_weight, 0.0)
    pattern_evidence_bear = np.where(use_patterns & bear_pattern & (not use_trend_context or bull_trend),
                                     pattern_weight, 0.0)

    # ---- structural candle roles ----
    bull_marubozu = bull_candle & (body_pct >= marubozu_body_min) \
        & (upper_wick_pct <= marubozu_wick_max) & (lower_wick_pct <= marubozu_wick_max)
    bear_marubozu = bear_candle & (body_pct >= marubozu_body_min) \
        & (upper_wick_pct <= marubozu_wick_max) & (lower_wick_pct <= marubozu_wick_max)
    dragonfly_doji = doji & (lower_wick_pct >= 0.60) & (upper_wick_pct <= 0.15)
    gravestone_doji = doji & (upper_wick_pct >= 0.60) & (lower_wick_pct <= 0.15)
    long_legged_doji = doji & (upper_wick_pct >= 0.30) & (lower_wick_pct >= 0.30)
    standard_doji = doji & ~dragonfly_doji & ~gravestone_doji & ~long_legged_doji
    spinning_top = (~doji) & ~bull_marubozu & ~bear_marubozu & (body_pct <= spinning_top_body_max) \
        & (upper_wick_pct >= spinning_top_wick_min) & (lower_wick_pct >= spinning_top_wick_min)
    candle_role_code = np.select(
        [bull_marubozu, bear_marubozu, dragonfly_doji, gravestone_doji,
         long_legged_doji, standard_doji, spinning_top],
        [1.0, -1.0, 2.0, -2.0, 3.0, 4.0, 5.0], default=0.0)

    # ---- liquidity sweep ----
    prior_low = lowest(low, sweep_lookback).shift(1).to_numpy(dtype=float)
    prior_high = highest(high, sweep_lookback).shift(1).to_numpy(dtype=float)
    bull_sweep = use_liquidity_sweep & ~np.isnan(prior_low) & (l < prior_low) & (c > prior_low) \
        & (lower_wick_pct >= sweep_wick_min) & (close_pos >= sweep_close_back_min)
    bear_sweep = use_liquidity_sweep & ~np.isnan(prior_high) & (h > prior_high) & (c < prior_high) \
        & (upper_wick_pct >= sweep_wick_min) & (close_pos <= 1.0 - sweep_close_back_min)

    # =========================================================================
    # STATEFUL PASS A — sequence aggregates + compression hysteresis + breakout
    # =========================================================================
    seq_pressure = np.zeros(n)
    bull_dom = np.zeros(n)
    bear_dom = np.zeros(n)
    close_persist_bull = np.zeros(n)
    close_persist_bear = np.zeros(n)
    bull_decel = np.zeros(n)
    bear_decel = np.zeros(n)
    compression_active = np.zeros(n, dtype=bool)
    breakout_attempt = np.zeros(n, dtype=np.int8)
    breakout_story = np.zeros(n, dtype=np.int8)

    dq = deque(maxlen=sequence_length)
    recent_split = int(np.ceil((sequence_length - 1) / 2.0))
    weight_sum = sequence_length * (sequence_length + 1) / 2.0

    compression_armed = False
    zH = zL = None
    zBars = 0
    zConsumed = False
    comp_active_prev = False
    # previous bar's finished zone (for breakout reference)
    pzH = pzL = None
    pzBars = 0
    pzConsumed = False

    bdir = 0
    bstart = -1
    bAttemptHigh = bAttemptLow = bBoundary = np.nan
    bstory = 0

    for t in range(n):
        # ---- sequence aggregates from deque ----
        dq.append((close_pos[t], body_pct[t], direction[t], c[t], rng[t],
                   (close_pos[t] - 0.5) * 2.0 * 45.0 + direction[t] * body_pct[t] * 40.0
                   + (lower_wick_pct[t] - upper_wick_pct[t]) * 15.0))
        if len(dq) == sequence_length:
            wsum = 0.0
            bull_body_total = 0.0
            bear_body_total = 0.0
            bull_candles = 0
            bear_candles = 0
            rising = 0
            falling = 0
            bull_recent = 0.0
            bear_recent = 0.0
            bull_older = 0.0
            bear_older = 0.0
            recent_cnt = 0
            older_cnt = 0
            L = sequence_length
            for k in range(L):
                cp, bp, d, cl, rg, hp = dq[k]
                wsum += hp * (k + 1)
                if d > 0:
                    bull_body_total += bp
                    bull_candles += 1
                elif d < 0:
                    bear_body_total += bp
                    bear_candles += 1
            seq_pressure[t] = clamp(wsum / weight_sum, -100.0, 100.0) if weight_sum > 0 else 0.0
            bull_dom[t] = clamp(bull_candles / L * 40.0 + bull_body_total / L * 60.0, 0.0, 100.0)
            bear_dom[t] = clamp(bear_candles / L * 40.0 + bear_body_total / L * 60.0, 0.0, 100.0)
            # adjacent close-progress pairs, newest -> oldest
            for j in range(L - 1):
                p = L - 1 - j
                cl_new = dq[p][3]
                cl_old = dq[p - 1][3]
                rg_i = dq[p][4]
                prog = clamp((cl_new - cl_old) / rg_i, -1.0, 1.0) if rg_i > 0 else 0.0
                if cl_new > cl_old:
                    rising += 1
                elif cl_new < cl_old:
                    falling += 1
                bpr = max(prog, 0.0)
                epr = max(-prog, 0.0)
                if j < recent_split:
                    bull_recent += bpr
                    bear_recent += epr
                    recent_cnt += 1
                else:
                    bull_older += bpr
                    bear_older += epr
                    older_cnt += 1
            denom = (L - 1.0) if L > 1 else 1.0
            close_persist_bull[t] = rising / denom * 100.0
            close_persist_bear[t] = falling / denom * 100.0
            if recent_cnt > 0 and older_cnt > 0:
                bull_decel[t] = clamp((bull_older / older_cnt - bull_recent / recent_cnt) * 150.0, 0.0, 100.0)
                bear_decel[t] = clamp((bear_older / older_cnt - bear_recent / recent_cnt) * 150.0, 0.0, 100.0)

        # ---- compression hysteresis (confirmed bar) ----
        cs = compression_score[t]
        if cs >= 60.0:
            compression_armed = True
        elif cs <= 45.0:
            compression_armed = False
        cactive = compression_armed
        compression_active[t] = cactive

        # ---- breakout story ----
        zone_avail = (comp_active_prev and not pzConsumed
                      and pzBars >= minimum_compression_bars
                      and pzH is not None and pzL is not None)
        bull_att = zone_avail and range_expansion[t] and (c[t] > pzH) and (close_pos[t] >= 0.60)
        bear_att = zone_avail and range_expansion[t] and (c[t] < pzL) and (close_pos[t] <= 0.40)
        comp_started = cactive and not comp_active_prev
        comp_ended = (not cactive) and comp_active_prev
        if comp_started:
            zH, zL, zBars, zConsumed = h[t], l[t], 1, False
        elif cactive and not (bull_att or bear_att):
            zH = max(zH, h[t])
            zL = min(zL, l[t])
            zBars += 1
        # comp_ended purely affects drawing -> ignored
        if bull_att or bear_att:
            zConsumed = True

        if bdir == 0 and (bull_att or bear_att):
            bdir = 1 if bull_att else -1
            bstart = t
            bAttemptHigh = h[t]
            bAttemptLow = l[t]
            bBoundary = pzH if bull_att else pzL
            bstory = bdir
        elif bdir != 0 and t > bstart:
            age = t - bstart
            accepted = (bdir == 1 and c[t] > bAttemptHigh and close_pos[t] >= 0.55) \
                or (bdir == -1 and c[t] < bAttemptLow and close_pos[t] <= 0.45)
            failed = (bdir == 1 and c[t] < bBoundary) or (bdir == -1 and c[t] > bBoundary)
            unresolved = age >= breakout_resolution_bars
            if accepted or failed or unresolved:
                bstory = bdir * 2 if accepted else (bdir * 3 if failed else bdir * 4)
                bdir = 0
        breakout_story[t] = bstory
        breakout_attempt[t] = 1 if bull_att else (-1 if bear_att else 0)

        # snapshot zone for next bar's breakout reference
        pzH, pzL, pzBars, pzConsumed = zH, zL, zBars, zConsumed
        comp_active_prev = cactive

    # =========================================================================
    # DIRECTIONAL EVIDENCE (vectorised)
    # =========================================================================
    bull_pressure0 = np.maximum(candle_pressure, 0.0)
    bear_pressure0 = np.maximum(-candle_pressure, 0.0)
    bull_candle_ev = bull_pressure0 * (0.55 + bull_quality / 100.0 * 0.45)
    bear_candle_ev = bear_pressure0 * (0.55 + bear_quality / 100.0 * 0.45)
    bull_seq_ev = np.maximum(seq_pressure, 0.0) * 0.50 + bull_dom * 0.30 + close_persist_bull * 0.20
    bear_seq_ev = np.maximum(-seq_pressure, 0.0) * 0.50 + bear_dom * 0.30 + close_persist_bear * 0.20
    base_bull = bull_candle_ev * 0.45 + bull_seq_ev * 0.55
    base_bear = bear_candle_ev * 0.45 + bear_seq_ev * 0.55
    if use_trend_context:
        bull_trend_ev = np.maximum(trend_score, 0.0) / 100.0 * trend_weight
        bear_trend_ev = np.maximum(-trend_score, 0.0) / 100.0 * trend_weight
    else:
        bull_trend_ev = np.zeros(n)
        bear_trend_ev = np.zeros(n)
    bull_ev = clamp(base_bull + bull_trend_ev + pattern_evidence_bull, 0.0, 100.0)
    bear_ev = clamp(base_bear + bear_trend_ev + pattern_evidence_bear, 0.0, 100.0)
    directional_balance = bull_ev - bear_ev

    # ---- exhaustion ----
    bull_opp_wick = clamp(upper_wick_pct * 100.0, 0.0, 100.0)
    bear_opp_wick = clamp(lower_wick_pct * 100.0, 0.0, 100.0)
    body_shrinking = (body < np.roll(body, 1)) & (np.roll(body, 1) < np.roll(body, 2))
    body_shrinking[:2] = False
    bull_contraction = np.where(body_shrinking, 100.0, np.where(body < np.roll(body, 1), 50.0, 0.0))
    bear_contraction = bull_contraction.copy()
    bull_trend_ctx = np.where(bull_trend, clamp(trend_score, 0.0, 100.0), 0.0)
    bear_trend_ctx = np.where(bear_trend, clamp(-trend_score, 0.0, 100.0), 0.0)
    bull_exh = np.where(bull_trend,
                        clamp(bull_opp_wick * (wick_exhaustion_weight / 100.0)
                              + bull_contraction * (body_contraction_weight / 100.0)
                              + bull_decel * 0.25 + bull_trend_ctx * 0.20, 0.0, 100.0),
                        0.0)
    bear_exh = np.where(bear_trend,
                        clamp(bear_opp_wick * (wick_exhaustion_weight / 100.0)
                              + bear_contraction * (body_contraction_weight / 100.0)
                              + bear_decel * 0.25 + bear_trend_ctx * 0.20, 0.0, 100.0),
                        0.0)
    bull_exhausted = bull_exh >= exhaustion_threshold
    bear_exhausted = bear_exh >= exhaustion_threshold

    # =========================================================================
    # STATEFUL PASS B — signal hysteresis + follow-through / failure
    # =========================================================================
    bull_signal = np.zeros(n, dtype=bool)
    bear_signal = np.zeros(n, dtype=bool)
    bull_state = np.zeros(n, dtype=bool)
    bear_state = np.zeros(n, dtype=bool)
    bull_ft = np.zeros(n, dtype=bool)
    bear_ft = np.zeros(n, dtype=bool)
    bull_fail = np.zeros(n, dtype=bool)
    bear_fail = np.zeros(n, dtype=bool)

    bull_armed = False
    bear_armed = False
    bull_armed_prev = False
    bear_armed_prev = False
    new_bull_prev = False
    new_bear_prev = False
    bull_ev_prev = bull_ev[0]
    bear_ev_prev = bear_ev[0]

    for t in range(n):
        be = bull_ev[t]
        re = bear_ev[t]
        bex = bool(bull_exhausted[t])
        rex = bool(bear_exhausted[t])
        bull_improving = be - bull_ev_prev >= minimum_signal_change
        bear_improving = re - bear_ev_prev >= minimum_signal_change
        bull_entry = (be >= signal_threshold and be > re and not bex and bull_improving)
        bear_entry = (re >= signal_threshold and re > be and not rex and bear_improving)
        if bull_entry and not bull_armed:
            bull_armed = True
        elif be < signal_threshold - signal_exit_buffer or be <= re or bex:
            bull_armed = False
        if bear_entry and not bear_armed:
            bear_armed = True
        elif re < signal_threshold - signal_exit_buffer or re <= be or rex:
            bear_armed = False
        new_bull = bull_armed and not bull_armed_prev
        new_bear = bear_armed and not bear_armed_prev
        bull_signal[t] = new_bull
        bear_signal[t] = new_bear
        bull_state[t] = bull_armed
        bear_state[t] = bear_armed
        if t >= 1:
            bull_ft[t] = new_bull_prev and (h[t] > h[t - 1]) and (c[t] > c[t - 1]) and (close_pos[t] >= 0.60)
            bear_ft[t] = new_bear_prev and (l[t] < l[t - 1]) and (c[t] < c[t - 1]) and (close_pos[t] <= 0.40)
            bull_fail[t] = new_bull_prev and (c[t] < l[t - 1])
            bear_fail[t] = new_bear_prev and (c[t] > h[t - 1])
        bull_armed_prev = bull_armed
        bear_armed_prev = bear_armed
        new_bull_prev = new_bull
        new_bear_prev = new_bear
        bull_ev_prev = be
        bear_ev_prev = re

    # ---- market story states ----
    strong_bull = (bull_ev >= strong_threshold) & (bull_ev > bear_ev)
    strong_bear = (bear_ev >= strong_threshold) & (bear_ev > bull_ev)
    bull_control = (bull_ev >= signal_threshold) & (bull_ev > bear_ev)
    bear_control = (bear_ev >= signal_threshold) & (bear_ev > bear_ev)
    indecision = (np.abs(directional_balance) < 12.0) & (body_pct < 0.35)

    # ---- assemble ----
    out = pd.DataFrame(index=idx)
    out["candle_pressure"] = candle_pressure
    out["sequence_pressure"] = seq_pressure
    out["bull_quality"] = bull_quality
    out["bear_quality"] = bear_quality
    out["bull_dominance"] = bull_dom
    out["bear_dominance"] = bear_dom
    out["close_persistence_bull"] = close_persist_bull
    out["close_persistence_bear"] = close_persist_bear
    out["bull_deceleration"] = bull_decel
    out["bear_deceleration"] = bear_decel
    out["bull_exhaustion"] = bull_exh
    out["bear_exhaustion"] = bear_exh
    out["compression_score"] = compression_score
    out["candle_role_code"] = candle_role_code
    out["trend_score"] = trend_score
    out["range_atr_ratio"] = range_atr_ratio
    out["volume_ratio"] = vol_ratio
    out["bull_evidence"] = bull_ev
    out["bear_evidence"] = bear_ev
    out["directional_balance"] = directional_balance

    def I8(a):
        return pd.Series(np.asarray(a), index=idx).astype("Int64")

    out["range_expansion"] = I8(range_expansion)
    out["compression_active"] = I8(compression_active)
    out["breakout_attempt"] = I8(breakout_attempt)
    out["breakout_story_event"] = I8(breakout_story)
    out["liquidity_sweep"] = I8(np.where(bull_sweep, 1, np.where(bear_sweep, -1, 0)))
    out["bull_signal"] = I8(bull_signal)
    out["bear_signal"] = I8(bear_signal)
    out["bull_signal_state"] = I8(bull_state)
    out["bear_signal_state"] = I8(bear_state)
    out["bull_exhausted"] = I8(bull_exhausted)
    out["bear_exhausted"] = I8(bear_exhausted)
    out["bull_follow_through"] = I8(bull_ft)
    out["bear_follow_through"] = I8(bear_ft)
    out["bull_failure"] = I8(bull_fail)
    out["bear_failure"] = I8(bear_fail)
    out["strong_bull"] = I8(strong_bull)
    out["strong_bear"] = I8(strong_bear)
    out["bull_control"] = I8(bull_control)
    out["bear_control"] = I8(bear_control)
    out["indecision"] = I8(indecision)
    out["bull_pattern"] = I8(bull_pattern)
    out["bear_pattern"] = I8(bear_pattern)
    out["bull_liquidity_sweep"] = I8(bull_sweep)
    out["bear_liquidity_sweep"] = I8(bear_sweep)
    out["doji"] = I8(doji)
    out["bull_engulfing"] = I8(bull_engulf)
    out["bear_engulfing"] = I8(bear_engulf)
    out["bull_pin_bar"] = I8(bull_pin)
    out["bear_pin_bar"] = I8(bear_pin)
    out["morning_star"] = I8(morning_star)
    out["evening_star"] = I8(evening_star)
    out["bull_marubozu"] = I8(bull_marubozu)
    out["bear_marubozu"] = I8(bear_marubozu)
    out["dragonfly_doji"] = I8(dragonfly_doji)
    out["gravestone_doji"] = I8(gravestone_doji)
    out["spinning_top"] = I8(spinning_top)

    return out[_COLUMNS]
