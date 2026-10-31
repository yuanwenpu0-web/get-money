"""Commodity Heat Reversal — exhaustion-scored reversal signals with HTF bias.

Source: pine-collection/indicators/composite/commodity_heat_reversal/commodity_heat_reversal.pine
(WavesUnchained, MIT)

Every bar scores stretching evidence: distance of the close beyond MA50 in ATR
units (weight 2), ATR expansion against its SMA (weight 2), RSI beyond 72/28,
an upper/lower wick beyond 35% of the range and a close outside the 20-period
2σ Bollinger band. A reversal additionally needs the mandatory exhaustion
(RSI/BB extreme), the wick-rejection gate and a mode-dependent confirmation;
the selected MA family (default HMA) smooths the fast/mean/trend averages and
the 9-bar confirmation average. The optional higher-timeframe bias reads the
previous *completed* HTF period (see `_tf.lift_series`), so `request.security`
never leaks the forming HTF bar.

Dropped: the score table, plots/labels/fills, the SL/TP overlay styling and the
alert plumbing. The stop levels (`short_sl` / `long_sl`) are kept as columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (atr, boolean, choice, ema, flags, hma, integer, number, ohlcv,
                    rma, rsi, sma, stdev, wma)
from ._tf import htf_frame, lift_series

_MA_TYPES = ("EMA", "RMA", "SMA", "WMA", "HMA", "JMA")


def _jma(series: pd.Series, length: int, phase: int, power: float) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    phase_ratio = phase / 100.0 + 1.5
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2.0)
    alpha = beta ** power
    e0 = e1 = e2 = jma = np.nan
    for position, value in enumerate(values):
        if np.isnan(value):
            continue
        e0 = (1.0 - alpha) * value + alpha * (value if np.isnan(e0) else e0)
        e1 = (value - e0) * (1.0 - beta) + beta * (0.0 if np.isnan(e1) else e1)
        e2 = ((e0 + phase_ratio * e1 - (value if np.isnan(jma) else jma)) * (1.0 - alpha) ** 2
              + alpha ** 2 * (0.0 if np.isnan(e2) else e2))
        jma = (value if np.isnan(jma) else jma) + e2
        out[position] = jma
    return pd.Series(out, index=series.index)


def _smooth(series: pd.Series, length: int, kind: str, jma_phase: int, jma_power: float) -> pd.Series:
    if kind == "SMA":
        return sma(series, length)
    if kind == "RMA":
        return rma(series, length)
    if kind == "WMA":
        return wma(series, length)
    if kind == "HMA":
        return hma(series, length)
    if kind == "JMA":
        return _jma(series, length, jma_phase, jma_power)
    return ema(series, length)


def _barssince_shifted(condition: pd.Series) -> pd.Series:
    """Pine's `ta.barssince(cond[1])`: bars since `cond` was last true strictly
    before the current bar."""
    truthy = condition.fillna(False).shift(1, fill_value=False).to_numpy(dtype=bool)
    out = np.full(len(condition), np.nan)
    last = None
    for position, value in enumerate(truthy):
        if value:
            last = position
        if last is not None:
            out[position] = position - last
    return pd.Series(out, index=condition.index)


def _barssince_flag(condition: pd.Series, lookback: int) -> pd.Series:
    """`not na(ta.barssince(cond)) and ta.barssince(cond) <= lookback`."""
    truthy = condition.fillna(False).to_numpy(dtype=bool)
    out = np.full(len(condition), False)
    last = None
    for position, value in enumerate(truthy):
        if value:
            last = position
        out[position] = last is not None and (position - last) <= lookback
    return pd.Series(out, index=condition.index)


def commodity_heat_reversal(
        frame: pd.DataFrame, *,
        ma_type: str = "HMA", jma_phase: int = 50, jma_power: float = 2.0,
        use_trend_filter: bool = False, allow_counter_trend_extreme: bool = True,
        use_htf_bias: bool = True, htf_minutes: int = 1440,
        ma_fast_len: int = 20, ma_mean_len: int = 50, ma_trend_len: int = 200,
        htf_trend_len: int = 50, atr_fast_len: int = 14, atr_slow_len: int = 50,
        rsi_len: int = 14,
        reversal_heat_threshold: int = 1, extreme_heat_threshold: int = 2,
        confirmation_mode: str = "Balanced", signal_cooldown_bars: int = 3,
        counter_trend_display: str = "Hide",
        show_continuation_signals: bool = False, continuation_score: int = 2,
        pullback_lookback: int = 12, continuation_cooldown_bars: int = 8,
        atr_distance_mult: float = 2.5, atr_expansion_mult: float = 1.5,
        sl_atr_mult: float = 1.2) -> pd.DataFrame:
    choice("ma_type", ma_type, _MA_TYPES)
    integer("jma_phase", jma_phase, -100, 100)
    number("jma_power", jma_power, 1.0, 4.0)
    boolean("use_trend_filter", use_trend_filter)
    boolean("allow_counter_trend_extreme", allow_counter_trend_extreme)
    boolean("use_htf_bias", use_htf_bias)
    integer("htf_minutes", htf_minutes, 1)
    integer("ma_fast_len", ma_fast_len, 1)
    integer("ma_mean_len", ma_mean_len, 1)
    integer("ma_trend_len", ma_trend_len, 1)
    integer("htf_trend_len", htf_trend_len, 1)
    integer("atr_fast_len", atr_fast_len, 1)
    integer("atr_slow_len", atr_slow_len, 1)
    integer("rsi_len", rsi_len, 1)
    integer("reversal_heat_threshold", reversal_heat_threshold, 1, 2)
    integer("extreme_heat_threshold", extreme_heat_threshold, 1, 2)
    choice("confirmation_mode", confirmation_mode, ("Early", "Balanced", "Strict"))
    integer("signal_cooldown_bars", signal_cooldown_bars, 0, 50)
    choice("counter_trend_display", counter_trend_display, ("Show", "Mark", "Hide"))
    boolean("show_continuation_signals", show_continuation_signals)
    integer("continuation_score", continuation_score, 1, 7)
    integer("pullback_lookback", pullback_lookback, 1, 50)
    integer("continuation_cooldown_bars", continuation_cooldown_bars, 0, 50)
    number("atr_distance_mult", atr_distance_mult, 0.0)
    number("atr_expansion_mult", atr_expansion_mult, 0.0)
    number("sl_atr_mult", sl_atr_mult, 0.0)
    open_, high, low, close = ohlcv(frame, "open", "high", "low", "close")

    def smooth(series: pd.Series, length: int) -> pd.Series:
        return _smooth(series, length, ma_type, jma_phase, jma_power)

    ma20 = smooth(close, ma_fast_len)
    ma50 = smooth(close, ma_mean_len)
    ma200 = smooth(close, ma_trend_len)
    ma9 = smooth(close, 9)

    atr14 = atr(frame, atr_fast_len)
    atr50 = sma(atr14, atr_slow_len)
    rsi_line = rsi(close, rsi_len)
    bb_basis = sma(close, 20)
    bb_dev = stdev(close, 20)
    bb_upper = bb_basis + bb_dev * 2
    bb_lower = bb_basis - bb_dev * 2

    # Higher-timeframe bias: previous completed HTF period (no lookahead).
    if use_htf_bias:
        aggregated = htf_frame(frame, htf_minutes)
        htf_ema_htf = ema(aggregated["close"], htf_trend_len)
        htf_close = lift_series(frame, aggregated["close"], htf_minutes)
        htf_ema = lift_series(frame, htf_ema_htf, htf_minutes)
        htf_slope = lift_series(frame, htf_ema_htf - htf_ema_htf.shift(1), htf_minutes)
    else:
        empty = pd.Series(np.nan, index=frame.index)
        htf_close = htf_ema = htf_slope = empty
    htf_up_trend = (not use_htf_bias) | flags(htf_close > htf_ema) | flags(htf_slope > 0)
    htf_down_trend = (not use_htf_bias) | flags(htf_close < htf_ema) | flags(htf_slope < 0)

    candle_range = high - low
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    upper_wick_pct = (upper_wick / candle_range.where(candle_range > 0)).fillna(0.0)
    lower_wick_pct = (lower_wick / candle_range.where(candle_range > 0)).fillna(0.0)

    short_heat = (flags(close > ma50 + atr_distance_mult * atr14).astype(float)
                  + flags(atr14 > atr50 * atr_expansion_mult).astype(float))
    long_heat = (flags(close < ma50 - atr_distance_mult * atr14).astype(float)
                 + flags(atr14 > atr50 * atr_expansion_mult).astype(float))
    short_score = (short_heat * 2
                   + flags(rsi_line > 72).astype(float)
                   + flags(upper_wick_pct > 0.35).astype(float)
                   + flags(close > bb_upper).astype(float))
    long_score = (long_heat * 2
                  + flags(rsi_line < 28).astype(float)
                  + flags(lower_wick_pct > 0.35).astype(float)
                  + flags(close < bb_lower).astype(float))

    up_trend = flags(close > ma200)
    down_trend = flags(close < ma200)
    ma50_slope = ma50 - ma50.shift(3)
    local_up = flags(close > ma50) & flags(ma50_slope >= 0)
    local_down = flags(close < ma50) & flags(ma50_slope <= 0)
    long_counter = ~local_up | down_trend
    short_counter = ~local_down | up_trend

    short_base = flags(close < low.shift(1)) | flags(close < ma9)
    long_base = flags(close > high.shift(1)) | flags(close > ma9)
    short_strict = (flags(close < open_) & flags(close < ma9) & flags(close < low.shift(1))
                    & flags(upper_wick_pct > 0.20))
    long_strict = (flags(close > open_) & flags(close > ma9) & flags(close > high.shift(1))
                   & flags(lower_wick_pct > 0.20))
    short_balanced = flags(close < ma9) & (flags(close < low.shift(1))
                                           | flags(upper_wick_pct > 0.30))
    long_balanced = flags(close > ma9) & (flags(close > high.shift(1))
                                          | flags(lower_wick_pct > 0.30))
    counter_map = {"Strict": (short_strict, long_strict),
                   "Balanced": (short_balanced, long_balanced),
                   "Early": (short_base, long_base)}
    short_counter_confirm, long_counter_confirm = counter_map[confirmation_mode]
    short_confirm = short_counter_confirm.where(short_counter, short_base)
    long_confirm = long_counter_confirm.where(long_counter, long_base)

    short_allowed = ((not use_trend_filter) | (down_trend & htf_down_trend)
                     | (allow_counter_trend_extreme & flags(short_heat >= extreme_heat_threshold)))
    long_allowed = ((not use_trend_filter) | (up_trend & htf_up_trend)
                    | (allow_counter_trend_extreme & flags(long_heat >= extreme_heat_threshold)))

    short_raw = (flags(short_heat >= reversal_heat_threshold) & short_confirm & short_allowed
                 & (flags(rsi_line > 72) | flags(close > bb_upper))
                 & flags(upper_wick_pct > 0.35))
    long_raw = (flags(long_heat >= reversal_heat_threshold) & long_confirm & long_allowed
                & (flags(rsi_line < 28) | flags(close < bb_lower))
                & flags(lower_wick_pct > 0.35))

    def recent(raw: pd.Series, cooldown: int) -> pd.Series:
        """Pine's `cooldown > 0 and not na(ta.barssince(raw[1]))
        and ta.barssince(raw[1]) <= cooldown`."""
        if cooldown <= 0:
            return pd.Series(False, index=frame.index)
        since = _barssince_shifted(raw)
        return flags(since.notna() & since.le(cooldown))

    short_signal = short_raw & ~recent(short_raw, signal_cooldown_bars)
    long_signal = long_raw & ~recent(long_raw, signal_cooldown_bars)

    short_ct_signal = (short_signal & flags(close > ma50) & flags(ma50_slope >= 0)
                       & ~flags(short_heat >= extreme_heat_threshold))
    long_ct_signal = (long_signal & flags(close < ma50) & flags(ma50_slope <= 0)
                      & ~flags(long_heat >= extreme_heat_threshold))
    show_ct = counter_trend_display == "Show"
    mark_ct = counter_trend_display == "Mark"
    short_reversal = short_signal & (show_ct | ~short_ct_signal)
    long_reversal = long_signal & (show_ct | ~long_ct_signal)

    long_pullback_touched = flags(low <= ma20) | flags(low <= ma50) | flags(close <= ma20)
    short_pullback_touched = flags(high >= ma20) | flags(high >= ma50) | flags(close >= ma20)
    long_pullback_heat = flags(long_score >= continuation_score) | flags(rsi_line < 45)
    short_pullback_heat = flags(short_score >= continuation_score) | flags(rsi_line > 55)
    local_cont_up = flags(close > ma50) & flags(ma20 > ma50) & flags(ma50_slope > 0)
    local_cont_down = flags(close < ma50) & flags(ma20 < ma50) & flags(ma50_slope < 0)
    long_pullback_setup = local_cont_up & htf_up_trend & long_pullback_touched & long_pullback_heat
    short_pullback_setup = local_cont_down & htf_down_trend & short_pullback_touched & short_pullback_heat

    long_cont_confirm = (_barssince_flag(long_pullback_setup, pullback_lookback)
                         & flags(close > open_) & flags(close > ma9)
                         & flags(close > high.shift(1)))
    short_cont_confirm = (_barssince_flag(short_pullback_setup, pullback_lookback)
                          & flags(close < open_) & flags(close < ma9)
                          & flags(close < low.shift(1)))
    long_cont_raw = flags(show_continuation_signals & long_cont_confirm)
    short_cont_raw = flags(show_continuation_signals & short_cont_confirm)
    long_continuation = long_cont_raw & ~recent(long_cont_raw, continuation_cooldown_bars)
    short_continuation = short_cont_raw & ~recent(short_cont_raw, continuation_cooldown_bars)

    swing_high = high.rolling(10, min_periods=10).max()
    swing_low = low.rolling(10, min_periods=10).min()

    return pd.DataFrame({
        "ma_fast": ma20,
        "ma_mean": ma50,
        "ma_trend": ma200,
        "ma9": ma9,
        "atr_fast": atr14,
        "atr_slow": atr50,
        "rsi": rsi_line,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "htf_close": htf_close,
        "htf_ema": htf_ema,
        "htf_ema_slope": htf_slope,
        "htf_up_trend": htf_up_trend,
        "htf_down_trend": htf_down_trend,
        "upper_wick_pct": upper_wick_pct,
        "lower_wick_pct": lower_wick_pct,
        "short_heat": short_heat,
        "long_heat": long_heat,
        "short_score": short_score.round().astype("Int64"),
        "long_score": long_score.round().astype("Int64"),
        "short_reversal_signal": short_reversal,
        "long_reversal_signal": long_reversal,
        "short_counter_trend_signal": flags(mark_ct & short_ct_signal),
        "long_counter_trend_signal": flags(mark_ct & long_ct_signal),
        "long_continuation_signal": long_continuation,
        "short_continuation_signal": short_continuation,
        "short_sl": swing_high + sl_atr_mult * atr14,
        "long_sl": swing_low - sl_atr_mult * atr14,
    }, index=frame.index)
