"""Swing Conviction Radar — per-swing conviction score with divergences.

Source: pine-collection/indicators/market_structure/swing_conviction_radar/swing_conviction_radar.pine
(WavesUnchained, MIT)

Every confirmed price pivot closes a swing leg. The leg is scored 0..100 from
three ingredients: speed (leg distance per bar in ATR units against
`speed_ref`), cleanliness (share of leg bars whose close stepped in the leg
direction) and — only when turnover is real — a volume gradient (first-half vs
second-half average volume of the leg, mapped around parity). High-pivot legs
run from the previous low pivot to the new high pivot and vice versa. A bearish
divergence prints when a higher high pivot carries lower conviction than the
previous high pivot (and the mirror for bullish divergences).

The conviction columns are step lines: they hold the latest leg score until the
next same-side pivot confirms. Divergence flags land on the confirmation bar
(pivot bar + `pivot_right`), matching `ta.pivothigh` semantics.

Dropped: dashboard table, divergence labels and all styling. The
`syminfo.volumetype != "tick"` check is approximated by "any positive volume
exists", as elsewhere in this catalog.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import atr, flags, integer, number, ohlcv, pivothigh, pivotlow


def swing_conviction_radar(
        frame: pd.DataFrame, *, pivot_left: int = 5, pivot_right: int = 5,
        max_leg: int = 100, w_speed: float = 0.4, w_clean: float = 0.4,
        w_vol: float = 0.2, atr_len: int = 14, speed_ref: float = 1.0) -> pd.DataFrame:
    integer("pivot_left", pivot_left, 2, 20)
    integer("pivot_right", pivot_right, 2, 20)
    integer("max_leg", max_leg, 10, 300)
    number("w_speed", w_speed, 0.0, 1.0)
    number("w_clean", w_clean, 0.0, 1.0)
    number("w_vol", w_vol, 0.0, 1.0)
    integer("atr_len", atr_len, 5, 50)
    number("speed_ref", speed_ref, 0.5, 10.0)
    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")

    piv_high = pivothigh(high, pivot_left, pivot_right)
    piv_low = pivotlow(low, pivot_left, pivot_right)
    atr_value = atr(frame, atr_len)
    volume_is_real = bool(volume.fillna(0.0).gt(0).any())

    rows = len(frame)
    closes = close.to_numpy(dtype=float)
    volumes = volume.fillna(0.0).to_numpy(dtype=float)
    atr_values = atr_value.to_numpy(dtype=float)
    high_values = high.to_numpy(dtype=float)
    low_values = low.to_numpy(dtype=float)
    piv_high_values = piv_high.to_numpy(dtype=float)
    piv_low_values = piv_low.to_numpy(dtype=float)

    def leg_score(is_bull: bool, leg_bars: int, leg_dist: float,
                  current_atr: float, end_position: int) -> float:
        speed_raw = (leg_dist / leg_bars / current_atr
                     if leg_bars > 0 and current_atr > 0 and not np.isnan(current_atr)
                     else 0.0)
        speed_n = min(speed_raw / speed_ref, 1.0)
        max_i = min(leg_bars - 1, max_leg - 1)
        on_count = 0
        vol1 = vol2 = 0.0
        vol1_count = vol2_count = 0
        for step in range(max_i + 1):
            offset = end_position - (pivot_right + step)
            if offset < 1:
                break
            if is_bull:
                is_on = closes[offset] > closes[offset - 1]
            else:
                is_on = closes[offset] < closes[offset - 1]
            if is_on:
                on_count += 1
            if step * 2 <= max_i:
                vol1 += volumes[offset]
                vol1_count += 1
            else:
                vol2 += volumes[offset]
                vol2_count += 1
        loop_len = max_i + 1
        clean_n = on_count / loop_len if loop_len > 0 else 0.0
        vol1_avg = vol1 / vol1_count if vol1_count > 0 else 0.0
        vol2_avg = vol2 / vol2_count if vol2_count > 0 else 1.0
        vol_grad = (vol1_avg / vol2_avg - 1.0) if vol2_avg > 0 else 0.0
        vol_n = max(min(vol_grad + 0.5, 1.0), 0.0)
        effective_w_vol = w_vol if volume_is_real else 0.0
        total_w = w_speed + w_clean + effective_w_vol
        if total_w <= 0:
            return 0.0
        return (speed_n * w_speed + clean_n * w_clean + vol_n * effective_w_vol) \
            / total_w * 100.0

    # Persistence mirrors of the Pine `var` block.
    last_low_bar, last_low_val, has_low = 0, 0.0, False
    last_high_bar, last_high_val, has_high = 0, 0.0, False
    last_high_conv = last_high_price = np.nan
    last_low_conv = last_low_price = np.nan

    high_step = np.full(rows, np.nan)
    low_step = np.full(rows, np.nan)
    bear_div = np.zeros(rows, dtype=bool)
    bull_div = np.zeros(rows, dtype=bool)

    for position in range(rows):
        if not np.isnan(piv_high_values[position]):
            pivot_bar = position - pivot_right
            pivot_value = high_values[position - pivot_right]
            leg_bars = pivot_bar - last_low_bar if has_low else 0
            leg_dist = abs(pivot_value - last_low_val) if has_low else 0.0
            conviction = (leg_score(True, leg_bars, leg_dist, atr_values[position], position)
                          if has_low and leg_bars > 0 else np.nan)
            if not np.isnan(conviction):
                if not np.isnan(last_high_conv) and not np.isnan(last_high_price):
                    if pivot_value > last_high_price and conviction < last_high_conv:
                        bear_div[position] = True
                last_high_conv = conviction
                last_high_price = pivot_value
            has_high = True
            last_high_bar = pivot_bar
            last_high_val = pivot_value
            if not np.isnan(conviction):
                high_step[position] = conviction
        if not np.isnan(piv_low_values[position]):
            pivot_bar = position - pivot_right
            pivot_value = low_values[position - pivot_right]
            leg_bars = pivot_bar - last_high_bar if has_high else 0
            leg_dist = abs(pivot_value - last_high_val) if has_high else 0.0
            conviction = (leg_score(False, leg_bars, leg_dist, atr_values[position], position)
                          if has_high and leg_bars > 0 else np.nan)
            if not np.isnan(conviction):
                if not np.isnan(last_low_conv) and not np.isnan(last_low_price):
                    if pivot_value < last_low_price and conviction < last_low_conv:
                        bull_div[position] = True
                last_low_conv = conviction
                last_low_price = pivot_value
            has_low = True
            last_low_bar = pivot_bar
            last_low_val = pivot_value
            if not np.isnan(conviction):
                low_step[position] = conviction

    return pd.DataFrame({
        "high_conviction": pd.Series(high_step, index=frame.index).ffill(),
        "low_conviction": pd.Series(low_step, index=frame.index).ffill(),
        "bearish_divergence": flags(pd.Series(bear_div, index=frame.index)),
        "bullish_divergence": flags(pd.Series(bull_div, index=frame.index)),
        "atr": atr_value,
    }, index=frame.index)
