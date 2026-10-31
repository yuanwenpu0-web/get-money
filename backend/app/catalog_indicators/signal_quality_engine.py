"""Signal Quality Engine — range-fade setup scorer with confirmation window.

Source: pine-collection/indicators/composite/signal_quality_engine/signal_quality_engine.pine
(WavesUnchained, MIT)

Scores fade setups at the edges of an 80-bar range. The distance term
(40%) combines ATR stretch above/below the EMA-50 and a clamped Bollinger
z-score; the structure term (30%) requires the close to sit in the outer 20%
of the range and within `near_atr_mult` ATR of the edge; the momentum term
(30%) blends RSI, StochRSI-K and WaveTrend against their overextended
readings. A regime gate (ADX and EMA slope below profile limits) disables
everything in trends. An extreme (score ≥ `score_min`) arms a
`confirm_window`-bar window; a trigger needs the armed extreme, range
position on the right side, and a candle rejection (sweep back inside or a
40% wick). Cooldowns and per-direction persistence complete the state
machine. Asset profiles map to band/ATR/ADX/slope constants; strictness maps
to score floors and cooldowns.

Dropped: plots, bands, zones, risk-level drawings, pivot markers, tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (atr, boolean, choice, crossover, dmi, ema, flags, highest, integer,
                    lowest, number, ohlcv, pivotlow, pivothigh, rsi, sma, stdev, stoch)

_PROFILES = {
    "Crypto": (2.1, 0.80, 24.0, 0.45),
    "Natural Gas": (2.5, 0.95, 28.0, 0.55),
    "Oil": (2.3, 0.85, 26.0, 0.50),
    "Gold / Silver": (2.0, 0.75, 23.0, 0.42),
    "Broad Commodities": (2.2, 0.82, 25.0, 0.48),
    "Stocks / Indices": (1.9, 0.68, 20.0, 0.35),
}
_STRICTNESS = {
    "Normal": (38.0, 8),
    "Strict": (48.0, 12),
    "Very Strict": (56.0, 18),
}


def signal_quality_engine(
        frame: pd.DataFrame, *, profile: str = "Crypto",
        strictness: str = "Normal", confirm_window: int = 3) -> pd.DataFrame:
    if profile not in _PROFILES:
        raise ValueError(f"参数 profile 不支持: {profile}")
    if strictness not in _STRICTNESS:
        raise ValueError(f"参数 strictness 不支持: {strictness}")
    integer("confirm_window", confirm_window, 1, 8)
    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    open_ = frame["open"].astype(float)
    bb_mult, near_atr_mult, adx_max, slope_max = _PROFILES[profile]
    score_min, cooldown_bars = _STRICTNESS[strictness]
    setup_min = score_min - 12.0

    def safe_div(a, b):
        return np.where(b != 0.0, a / b, 0.0)

    def score01(series: pd.Series) -> pd.Series:
        return series.clip(lower=0.0, upper=1.0) * 100.0

    average_true_range = atr(frame, 14)
    ema50 = ema(close, 50)
    basis = sma(close, 20)
    deviation = stdev(close, 20)
    upper_band = basis + deviation * bb_mult
    lower_band = basis - deviation * bb_mult
    dist_atr = pd.Series(safe_div(close - ema50, average_true_range), index=frame.index)
    bb_z = ((close - basis) / deviation.replace(0, np.nan)).clip(-4.0, 4.0)

    _, _, adx = dmi(frame, 14)
    ema_slope = pd.Series(safe_div(ema50 - ema50.shift(10), average_true_range),
                          index=frame.index)
    range_ok = flags((adx < adx_max) & (ema_slope.abs() < slope_max))

    rsi14 = rsi(close, 14)
    stoch_raw = stoch(rsi14, rsi14, rsi14, 14)
    k_line = sma(stoch_raw, 3)

    ap = (high + low + close) / 3
    esa = ema(ap, 10)
    de = ema((ap - esa).abs(), 10)
    ci = (ap - esa) / (0.015 * de.replace(0, np.nan))
    wt1 = ema(ci, 21)

    recent_high = highest(high, 80)
    recent_low = lowest(low, 80)
    range_size = recent_high - recent_low
    range_pos = pd.Series(safe_div(close - recent_low, range_size), index=frame.index)
    near_high = flags((range_pos > 0.80)
                      & (recent_high - close <= average_true_range * near_atr_mult))
    near_low = flags((range_pos < 0.20)
                     & (close - recent_low <= average_true_range * near_atr_mult))

    bar_range = (high - low).replace(0, np.nan)
    upper_wick = (high - np.maximum(open_, close)) / bar_range
    lower_wick = (np.minimum(open_, close) - low) / bar_range
    swept_high = flags((high > recent_high.shift(1)) & (close < recent_high.shift(1)))
    swept_low = flags((low < recent_low.shift(1)) & (close > recent_low.shift(1)))
    top_reject = flags(swept_high
                       | ((close < open_) & (upper_wick >= 0.40) & (close < ap)))
    bottom_reject = flags(swept_low
                          | ((close > open_) & (lower_wick >= 0.40) & (close > ap)))

    dist_top = score01(pd.Series(
        np.maximum(safe_div(dist_atr, 3.0), safe_div(bb_z, 2.5)), index=frame.index))
    dist_bottom = score01(pd.Series(
        np.maximum(safe_div(-dist_atr, 3.0), safe_div(-bb_z, 2.5)), index=frame.index))
    struct_top = score01(pd.Series(near_high.astype(float), index=frame.index))
    struct_bottom = score01(pd.Series(near_low.astype(float), index=frame.index))
    mom_top = score01(0.40 * (rsi14 - 60.0).clip(lower=0.0) / 20.0
                      + 0.30 * (k_line - 75.0).clip(lower=0.0) / 25.0
                      + 0.30 * (wt1 - 45.0).clip(lower=0.0) / 25.0)
    mom_bottom = score01(0.40 * (40.0 - rsi14).clip(lower=0.0) / 20.0
                         + 0.30 * (25.0 - k_line).clip(lower=0.0) / 25.0
                         + 0.30 * (-45.0 - wt1).clip(lower=0.0) / 25.0)

    top_score = dist_top * 0.40 + struct_top * 0.30 + mom_top * 0.30
    bottom_score = dist_bottom * 0.40 + struct_bottom * 0.30 + mom_bottom * 0.30

    edge_dir = pd.Series(0, index=frame.index)
    edge_dir = edge_dir.mask((bottom_score >= setup_min) & (bottom_score >= top_score), 1)
    edge_dir = edge_dir.mask((top_score >= setup_min) & (top_score > bottom_score), -1)

    bottom_extreme = flags((bottom_score >= score_min) & (bottom_score >= top_score))
    top_extreme = flags((top_score >= score_min) & (top_score > bottom_score))

    from ._pine import barssince
    rows = len(frame)
    bars_since_bottom = barssince(bottom_extreme)
    bars_since_top = barssince(top_extreme)
    bottom_armed = flags(bottom_extreme | (bars_since_bottom <= confirm_window))
    top_armed = flags(top_extreme | (bars_since_top <= confirm_window))

    long_setup = flags((edge_dir == 1) & (bottom_score >= setup_min) & ~bottom_extreme)
    short_setup = flags((edge_dir == -1) & (top_score >= setup_min) & ~top_extreme)

    # Cooldown persistence.
    last_long = last_short = None
    long_trigger_values = np.zeros(rows, dtype=bool)
    short_trigger_values = np.zeros(rows, dtype=bool)
    long_watch_values = np.zeros(rows, dtype=bool)
    short_watch_values = np.zeros(rows, dtype=bool)
    range_ok_values = range_ok.to_numpy(dtype=bool)
    bottom_armed_values = bottom_armed.to_numpy(dtype=bool)
    top_armed_values = top_armed.to_numpy(dtype=bool)
    bottom_reject_values = bottom_reject.to_numpy(dtype=bool)
    top_reject_values = top_reject.to_numpy(dtype=bool)
    range_pos_values = range_pos.to_numpy(dtype=float)
    for position in range(rows):
        long_cd = last_long is None or position - last_long > cooldown_bars
        short_cd = last_short is None or position - last_short > cooldown_bars
        final_long = bool(range_ok_values[position] and bottom_armed_values[position]
                          and range_pos_values[position] < 0.35
                          and bottom_reject_values[position] and long_cd)
        final_short = bool(range_ok_values[position] and top_armed_values[position]
                           and range_pos_values[position] > 0.65
                           and top_reject_values[position] and short_cd)
        long_trigger_values[position] = final_long
        short_trigger_values[position] = final_short
        long_watch_values[position] = bool(range_ok_values[position]
                                           and bottom_armed_values[position]
                                           and not final_long)
        short_watch_values[position] = bool(range_ok_values[position]
                                            and top_armed_values[position]
                                            and not final_short)
        if final_long:
            last_long = position
        if final_short:
            last_short = position

    long_trigger = flags(pd.Series(long_trigger_values, index=frame.index))
    short_trigger = flags(pd.Series(short_trigger_values, index=frame.index))

    def grade(score: pd.Series, base: float) -> pd.Series:
        result = pd.Series("—", index=frame.index, dtype=object)
        result = result.mask(score >= base + 24, "A+")
        result = result.mask((score < base + 24) & (score >= base + 12), "A")
        result = result.mask((score < base + 12) & (score >= base), "B")
        result = result.mask((score < base) & (score >= base - 10), "C")
        return result

    return pd.DataFrame({
        "bottom_score": bottom_score,
        "top_score": top_score,
        "edge_dir": edge_dir.astype("Int64"),
        "bottom_extreme": bottom_extreme,
        "top_extreme": top_extreme,
        "bottom_armed": bottom_armed,
        "top_armed": top_armed,
        "long_setup": long_setup,
        "short_setup": short_setup,
        "long_trigger": long_trigger,
        "short_trigger": short_trigger,
        "long_watch": flags(pd.Series(long_watch_values, index=frame.index)),
        "short_watch": flags(pd.Series(short_watch_values, index=frame.index)),
        "range_ok": range_ok,
        "range_position": range_pos,
        "rsi": rsi14,
        "stochrsi_k": k_line,
        "wavetrend": wt1,
        "dist_atr": dist_atr,
        "bb_z": bb_z,
        "long_stop": low - average_true_range * 1.2,
        "long_tp1": basis,
        "long_tp2": upper_band,
        "short_stop": high + average_true_range * 1.2,
        "short_tp1": basis,
        "short_tp2": lower_band,
        "bottom_grade": grade(bottom_score, score_min),
        "top_grade": grade(top_score, score_min),
    }, index=frame.index)
