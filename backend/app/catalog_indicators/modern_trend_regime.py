"""Modern Trend Regime [WavesUnchained]  v1.8.0  —  catalog port.

Source: data/pine-collection/indicators/trend_direction/modern_trend_regime/
        modern_trend_regime.pine

Score-based trend / range / transition regime detector:
  * Fast / base / slow structure lines (price-range midpoint, EMA, WMA or SMA).
  * Trend-strength inputs: ADX + DI, directional efficiency ratio, base slope
    (ATR-normalized) and structure width (ATR).
  * A 4-factor range score and a 5-factor trend score, resolved by a
    confirmation-bar hysteresis state machine (1 bull / -1 bear / 2 range / 0
    transition).
  * Frozen range bounds on range entry, edge reactions (debounced) and
    time-bounded breakouts, plus regime-coupled trend flips and a
    preview-offset structure breakout.

Ported computational core only; all drawing / band fills / diagnostics table
are dropped.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import choice, dmi, ema, highest, integer, lowest, number, ohlcv, sma, wma

_PRESET = {
    "Standard": dict(fast=9, base=26, slow=52, regime=20, adx=14, r_adx=18.0, t_adx=23.0,
                     r_eff=0.28, t_eff=0.42, slope=0.025, comp=0.45, confirm=3, buf=0.10, atr=14),
    "Schnell": dict(fast=7, base=18, slow=36, regime=14, adx=10, r_adx=17.0, t_adx=21.0,
                    r_eff=0.25, t_eff=0.38, slope=0.035, comp=0.50, confirm=2, buf=0.05, atr=10),
    "Ruhig": dict(fast=12, base=30, slow=60, regime=30, adx=18, r_adx=19.0, t_adx=25.0,
                  r_eff=0.30, t_eff=0.46, slope=0.020, comp=0.40, confirm=5, buf=0.15, atr=20),
    "Trendfolge": dict(fast=10, base=28, slow=65, regime=25, adx=14, r_adx=16.0, t_adx=27.0,
                       r_eff=0.22, t_eff=0.50, slope=0.018, comp=0.35, confirm=4, buf=0.15, atr=14),
    "Range Trading": dict(fast=8, base=22, slow=44, regime=18, adx=12, r_adx=21.0, t_adx=24.0,
                          r_eff=0.34, t_eff=0.44, slope=0.040, comp=0.60, confirm=2, buf=0.10, atr=14),
    "Ausbruch": dict(fast=10, base=26, slow=52, regime=20, adx=14, r_adx=20.0, t_adx=25.0,
                     r_eff=0.30, t_eff=0.48, slope=0.025, comp=0.38, confirm=3, buf=0.20, atr=14),
}
_PRESETS = tuple(list(_PRESET.keys()) + ["Benutzerdefiniert"])


def modern_trend_regime(
    frame: pd.DataFrame,
    *,
    preset: str = "Standard",
    calculation_mode: str = "Preisspanne",
    custom_fast_length: int = 9,
    custom_base_length: int = 26,
    custom_slow_length: int = 52,
    custom_atr_length: int = 14,
    custom_regime_length: int = 20,
    custom_adx_length: int = 14,
    custom_range_adx_threshold: float = 18.0,
    custom_trend_adx_threshold: float = 23.0,
    custom_range_efficiency_threshold: float = 0.28,
    custom_trend_efficiency_threshold: float = 0.42,
    custom_range_slope_threshold: float = 0.025,
    custom_compression_threshold: float = 0.45,
    custom_regime_confirmation_bars: int = 3,
    custom_breakout_buffer_atr: float = 0.10,
    custom_breakout_validity_bars: int = 10,
    custom_range_structure_length: int = 20,
    custom_range_reaction_zone_atr: float = 0.10,
    range_reaction_needs_candle: bool = True,
    strong_range_reaction_enabled: bool = False,
    range_reaction_close_position: float = 0.65,
    structure_preview_enabled: bool = True,
    structure_preview_offset: int = 26,
) -> pd.DataFrame:
    preset = choice("preset", preset, _PRESETS)
    calculation_mode = choice("calculation_mode", calculation_mode, ("Preisspanne", "EMA", "WMA", "SMA"))
    custom_fast_length = integer("custom_fast_length", custom_fast_length, 2)
    custom_base_length = integer("custom_base_length", custom_base_length, 2)
    custom_slow_length = integer("custom_slow_length", custom_slow_length, 3)
    custom_atr_length = integer("custom_atr_length", custom_atr_length, 2)
    custom_regime_length = integer("custom_regime_length", custom_regime_length, 5)
    custom_adx_length = integer("custom_adx_length", custom_adx_length, 3)
    custom_regime_confirmation_bars = integer("custom_regime_confirmation_bars", custom_regime_confirmation_bars, 1, 10)
    custom_breakout_validity_bars = integer("custom_breakout_validity_bars", custom_breakout_validity_bars, 1, 100)
    custom_range_structure_length = integer("custom_range_structure_length", custom_range_structure_length, 3, 200)
    custom_range_adx_threshold = number("custom_range_adx_threshold", custom_range_adx_threshold, 5.0, 40.0)
    custom_trend_adx_threshold = number("custom_trend_adx_threshold", custom_trend_adx_threshold, 5.0, 50.0)
    custom_range_efficiency_threshold = number("custom_range_efficiency_threshold", custom_range_efficiency_threshold, 0.05, 0.8)
    custom_trend_efficiency_threshold = number("custom_trend_efficiency_threshold", custom_trend_efficiency_threshold, 0.05, 1.0)
    custom_range_slope_threshold = number("custom_range_slope_threshold", custom_range_slope_threshold, 0.0)
    custom_compression_threshold = number("custom_compression_threshold", custom_compression_threshold, 0.05)
    custom_breakout_buffer_atr = number("custom_breakout_buffer_atr", custom_breakout_buffer_atr, 0.0)
    custom_range_reaction_zone_atr = number("custom_range_reaction_zone_atr", custom_range_reaction_zone_atr, 0.0, 1.0)
    range_reaction_close_position = number("range_reaction_close_position", range_reaction_close_position, 0.50, 0.95)

    if preset == "Benutzerdefiniert":
        p = dict(fast=custom_fast_length, base=custom_base_length, slow=custom_slow_length,
                 regime=custom_regime_length, adx=custom_adx_length,
                 r_adx=custom_range_adx_threshold, t_adx=custom_trend_adx_threshold,
                 r_eff=custom_range_efficiency_threshold, t_eff=custom_trend_efficiency_threshold,
                 slope=custom_range_slope_threshold, comp=custom_compression_threshold,
                 confirm=custom_regime_confirmation_bars, buf=custom_breakout_buffer_atr,
                 atr=custom_atr_length)
        range_structure_len = custom_range_structure_length
        validity = custom_breakout_validity_bars
        reaction_zone = custom_range_reaction_zone_atr
    else:
        p = _PRESET[preset]
        range_structure_len = min(p["regime"], p["base"])
        validity = 10
        reaction_zone = 0.10

    o, h, l, c = ohlcv(frame, "open", "high", "low", "close")
    o = o.to_numpy(dtype=float); h = h.to_numpy(dtype=float)
    l = l.to_numpy(dtype=float); c = c.to_numpy(dtype=float)
    idx = frame.index
    n = len(frame)

    def line(source, length, mode):
        if mode == "EMA":
            return ema(source, length).to_numpy(dtype=float)
        if mode == "WMA":
            return wma(source, length).to_numpy(dtype=float)
        if mode == "SMA":
            return sma(source, length).to_numpy(dtype=float)
        return ((highest(frame["high"], length) + lowest(frame["low"], length)) / 2.0).to_numpy(dtype=float)

    src = frame["close"]
    fast = line(src, p["fast"], calculation_mode)
    base = line(src, p["base"], calculation_mode)
    slow = line(src, p["slow"], calculation_mode)

    atr_a = np.maximum(atr_series(frame, p["atr"]), 1e-8)
    plus_di, minus_di, adx = dmi(frame, p["adx"])
    plus_di = plus_di.to_numpy(dtype=float); minus_di = minus_di.to_numpy(dtype=float); adx = adx.to_numpy(dtype=float)

    chg = np.abs(np.diff(c, prepend=c[0]))
    total_move = pd.Series(chg).rolling(p["regime"], min_periods=p["regime"]).sum().to_numpy()

    structure_fast = (fast + base) / 2.0
    structure_upper = np.fmax(structure_fast, slow)
    structure_lower = np.fmin(structure_fast, slow)
    structure_width = structure_upper - structure_lower
    swing_high_a = highest(frame["high"], range_structure_len).to_numpy(dtype=float)
    swing_low_a = lowest(frame["low"], range_structure_len).to_numpy(dtype=float)

    required_range = 3 if preset == "Range Trading" else 4
    required_trend = 5 if preset == "Range Trading" else 4

    # state
    candidate_state = 0
    candidate_count = 0
    market_regime = 0
    active_range_high = float("nan")
    active_range_low = float("nan")
    range_initialized = False
    breakout_armed = False
    bars_since_range_end = None
    prev_is_range = False
    prev_is_bull = False
    prev_is_bear = False
    prev_raw_long = False
    prev_raw_short = False

    out = {k: np.full(n, np.nan) for k in [
        "regime", "fast", "base", "slow", "structure_upper", "structure_lower", "adx",
        "plus_di", "minus_di", "efficiency", "base_slope_atr", "structure_width_atr",
        "range_score", "bull_score", "bear_score", "range_high", "range_low", "range_mid",
        "breakout"]}
    flags = {k: np.zeros(n, dtype=np.int64) for k in [
        "range_long_reaction", "range_short_reaction", "bull_flip", "bear_flip",
        "structure_breakout_bull", "structure_breakout_bear"]}

    for i in range(n):
        atr_i = atr_a[i]
        dir_move = abs(c[i] - c[i - p["regime"]]) if i >= p["regime"] else 0.0
        tm = total_move[i]
        eff = (dir_move / tm) if (tm is not None and not math.isnan(tm) and tm > 0) else 0.0
        base_slope = (abs(base[i] - base[i - p["regime"]]) / (atr_i * p["regime"])
                      if (i >= p["regime"] and atr_i > 0 and p["regime"] > 0) else 0.0)
        width_atr = structure_width[i] / atr_i if atr_i > 0 else 0.0

        range_score = ((1 if adx[i] < p["r_adx"] else 0) + (1 if eff < p["r_eff"] else 0)
                       + (1 if base_slope < p["slope"] else 0) + (1 if width_atr < p["comp"] else 0))
        range_entry = range_score >= required_range

        shared = (1 if adx[i] > p["t_adx"] else 0) + (1 if eff > p["t_eff"] else 0)
        bull_dir = ((1 if plus_di[i] > minus_di[i] else 0)
                    + (1 if (fast[i] > base[i] and base[i] > slow[i]) else 0)
                    + (1 if c[i] > base[i] else 0))
        bear_dir = ((1 if minus_di[i] > plus_di[i] else 0)
                    + (1 if (fast[i] < base[i] and base[i] < slow[i]) else 0)
                    + (1 if c[i] < base[i] else 0))
        bull_score = shared + bull_dir
        bear_score = shared + bear_dir
        bull_entry = bull_score >= required_trend
        bear_entry = bear_score >= required_trend

        bull_hold = market_regime == 1 and bull_score >= max(required_trend - 1, 1) and not bear_entry and not range_entry
        bear_hold = market_regime == -1 and bear_score >= max(required_trend - 1, 1) and not bull_entry and not range_entry
        range_hold = market_regime == 2 and range_score >= max(required_range - 1, 1) and not bull_entry and not bear_entry

        if market_regime == 1 and bear_entry:
            cand = -1
        elif market_regime == -1 and bull_entry:
            cand = 1
        elif bull_entry:
            cand = 1
        elif bear_entry:
            cand = -1
        elif range_entry:
            cand = 2
        elif bull_hold:
            cand = 1
        elif bear_hold:
            cand = -1
        elif range_hold:
            cand = 2
        else:
            cand = 0

        if cand == 0:
            market_regime = 0
            candidate_state = 0
            candidate_count = 0
        else:
            if market_regime != 0 and cand != market_regime:
                market_regime = 0
            if cand == candidate_state:
                candidate_count += 1
            else:
                candidate_state = cand
                candidate_count = 1
            if candidate_count >= p["confirm"]:
                market_regime = candidate_state

        is_bull = market_regime == 1
        is_bear = market_regime == -1
        is_range = market_regime == 2
        range_started = is_range and not prev_is_range
        range_ended = (not is_range) and prev_is_range

        # range structure freeze
        swing_high = swing_high_a[i] if i + 1 >= range_structure_len else np.nan
        swing_low = swing_low_a[i] if i + 1 >= range_structure_len else np.nan
        if range_started:
            active_range_high = swing_high
            active_range_low = swing_low
            range_initialized = True
        if range_ended:
            range_initialized = False

        range_mid = (active_range_high + active_range_low) / 2.0 if range_initialized else np.nan
        candle_range = max(h[i] - l[i], 1e-8)
        close_pos = (c[i] - l[i]) / candle_range
        strong_bull = close_pos >= range_reaction_close_position
        strong_bear = close_pos <= 1.0 - range_reaction_close_position
        lower_level = min(active_range_low + atr_i * reaction_zone, range_mid) if range_initialized else np.nan
        upper_level = max(active_range_high - atr_i * reaction_zone, range_mid) if range_initialized else np.nan
        lower_touched = range_initialized and l[i] <= lower_level and c[i] > active_range_low
        upper_touched = range_initialized and h[i] >= upper_level and c[i] < active_range_high
        raw_long = is_range and lower_touched and ((not range_reaction_needs_candle) or c[i] > o[i]) and ((not strong_range_reaction_enabled) or strong_bull)
        raw_short = is_range and upper_touched and ((not range_reaction_needs_candle) or c[i] < o[i]) and ((not strong_range_reaction_enabled) or strong_bear)
        range_long_reaction = raw_long and not prev_raw_long and not range_started
        range_short_reaction = raw_short and not prev_raw_short and not range_started

        # breakouts
        if range_started:
            breakout_armed = True
            bars_since_range_end = None
        if range_ended:
            bars_since_range_end = 0
        elif breakout_armed and bars_since_range_end is not None:
            bars_since_range_end += 1
        if breakout_armed and bars_since_range_end is not None and bars_since_range_end > validity:
            breakout_armed = False

        prev_atr = atr_a[i - 1] if i >= 1 else atr_i
        bull_breakout = (breakout_armed and not is_range and bars_since_range_end is not None
                         and bars_since_range_end <= validity and not math.isnan(active_range_high)
                         and c[i] > active_range_high + atr_i * p["buf"]
                         and c[i - 1] <= active_range_high + prev_atr * p["buf"] if i >= 1 else False)
        bear_breakout = (breakout_armed and not is_range and bars_since_range_end is not None
                         and bars_since_range_end <= validity and not math.isnan(active_range_low)
                         and c[i] < active_range_low - atr_i * p["buf"]
                         and c[i - 1] >= active_range_low - prev_atr * p["buf"] if i >= 1 else False)
        if bull_breakout or bear_breakout:
            breakout_armed = False

        bull_flip = is_bull and not prev_is_bull
        bear_flip = is_bear and not prev_is_bear

        sb_bull = sb_bear = False
        if structure_preview_enabled and i > structure_preview_offset:
            vis_upper = structure_upper[i - structure_preview_offset]
            vis_lower = structure_lower[i - structure_preview_offset]
            prev_vis_upper = structure_upper[i - 1 - structure_preview_offset]
            prev_vis_lower = structure_lower[i - 1 - structure_preview_offset]
            sb_bull = (not math.isnan(vis_upper) and c[i] > vis_upper and c[i - 1] <= prev_vis_upper
                       and not is_range and structure_fast[i] > slow[i])
            sb_bear = (not math.isnan(vis_lower) and c[i] < vis_lower and c[i - 1] >= prev_vis_lower
                       and not is_range and structure_fast[i] < slow[i])

        out["regime"][i] = market_regime
        out["fast"][i] = fast[i]
        out["base"][i] = base[i]
        out["slow"][i] = slow[i]
        out["structure_upper"][i] = structure_upper[i]
        out["structure_lower"][i] = structure_lower[i]
        out["adx"][i] = adx[i]
        out["plus_di"][i] = plus_di[i]
        out["minus_di"][i] = minus_di[i]
        out["efficiency"][i] = eff
        out["base_slope_atr"][i] = base_slope
        out["structure_width_atr"][i] = width_atr
        out["range_score"][i] = range_score
        out["bull_score"][i] = bull_score
        out["bear_score"][i] = bear_score
        out["range_high"][i] = active_range_high if range_initialized else np.nan
        out["range_low"][i] = active_range_low if range_initialized else np.nan
        out["range_mid"][i] = range_mid
        out["breakout"][i] = 1 if bull_breakout else -1 if bear_breakout else 0
        flags["range_long_reaction"][i] = 1 if range_long_reaction else 0
        flags["range_short_reaction"][i] = 1 if range_short_reaction else 0
        flags["bull_flip"][i] = 1 if bull_flip else 0
        flags["bear_flip"][i] = 1 if bear_flip else 0
        flags["structure_breakout_bull"][i] = 1 if sb_bull else 0
        flags["structure_breakout_bear"][i] = 1 if sb_bear else 0

        prev_is_range = is_range
        prev_is_bull = is_bull
        prev_is_bear = is_bear
        prev_raw_long = raw_long
        prev_raw_short = raw_short

    df = pd.DataFrame(index=idx)
    df["mtr_regime"] = pd.Series(out["regime"], index=idx).astype("Int64")
    df["mtr_fast"] = out["fast"]
    df["mtr_base"] = out["base"]
    df["mtr_slow"] = out["slow"]
    df["mtr_structure_upper"] = out["structure_upper"]
    df["mtr_structure_lower"] = out["structure_lower"]
    df["mtr_adx"] = out["adx"]
    df["mtr_plus_di"] = out["plus_di"]
    df["mtr_minus_di"] = out["minus_di"]
    df["mtr_efficiency"] = out["efficiency"]
    df["mtr_base_slope_atr"] = out["base_slope_atr"]
    df["mtr_structure_width_atr"] = out["structure_width_atr"]
    df["mtr_range_score"] = pd.Series(out["range_score"], index=idx).astype("Int64")
    df["mtr_bull_score"] = pd.Series(out["bull_score"], index=idx).astype("Int64")
    df["mtr_bear_score"] = pd.Series(out["bear_score"], index=idx).astype("Int64")
    df["mtr_range_high"] = out["range_high"]
    df["mtr_range_low"] = out["range_low"]
    df["mtr_range_mid"] = out["range_mid"]
    df["mtr_breakout"] = pd.Series(out["breakout"], index=idx).astype("Int64")
    for k, v in flags.items():
        df[f"mtr_{k}"] = pd.Series(v, index=idx).astype("Int64")
    return df


def atr_series(frame, length):
    from ._pine import atr
    return atr(frame, length).to_numpy(dtype=float)
