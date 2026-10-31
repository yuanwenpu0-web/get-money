"""Capital41 Liquidity Void + Imbalance Map — fair-value-gap zone tracker.

Source: pine-crypto-collection/capital41_btc_general/Capital41_LiquidityVoid Imbalance Map/Capital41_Liquidity_Void_Imbalance_Map.pine
(Capital41; no explicit license header in the source file)

A three-bar fair-value gap (bar's low above the high two bars back = bullish
void; bar's high below the low two bars back = bearish void) opens a price
zone. Zones from the higher timeframe (computed on aggregated HTF bars and
published one completed period later — the lookahead-free reading of the
upstream `request.security`) carry a 1.2× urgency weight. Overlapping
same-direction zones merge; the oldest zone is evicted past `max_zones`. Each
bar the fill percentage is recomputed from how far price has traded back into
the zone; urgency = unfilled share × (0.55 + 0.45 × freshness) × source
weight. A fully-filled zone flips its breaker flag when price closes beyond
the far edge. Gaps smaller than `min_ticks` ticks are ignored.

Output columns report the top-3 urgent zones per bar plus aggregate flags.

Dropped: boxes, labels, urgency table, all colours/transparency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import boolean, flags, integer, ohlcv
from ._tf import base_step_minutes, htf_frame, lift_series

_MINTICK = 1e-10


def capital41_liquidity_void_imbalance_map(
        frame: pd.DataFrame, *, use_htf: bool = True, htf_minutes: int = 240,
        merge_overlaps: bool = True, max_zones: int = 35, min_ticks: int = 2,
        fresh_bars: int = 80, remove_filled: bool = False) -> pd.DataFrame:
    boolean("use_htf", use_htf)
    integer("htf_minutes", htf_minutes, 1)
    boolean("merge_overlaps", merge_overlaps)
    integer("max_zones", max_zones, 5, 150)
    integer("min_ticks", min_ticks, 1, 100)
    integer("fresh_bars", fresh_bars, 10, 500)
    boolean("remove_filled", remove_filled)
    high, low, close = ohlcv(frame, "high", "low", "close")
    rows = len(frame)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)

    # zone state: [top, bottom, direction, born, fill, active, source_weight]
    zones: list[list] = []

    def fill_pct(direction: int, top: float, bottom: float) -> float:
        height = top - bottom
        if height <= 0:
            return 0.0
        if direction == 1:
            raw = (top - max(lows[position], bottom)) / height * 100.0
        else:
            raw = (min(highs[position], top) - bottom) / height * 100.0
        return min(max(raw, 0.0), 100.0)

    def add_zone(top: float, bottom: float, direction: int, source_weight: float,
                 position: int) -> None:
        if top <= bottom or (top - bottom) < _MINTICK * min_ticks:
            return
        if merge_overlaps:
            for zone in zones:
                if zone[5] and zone[2] == direction:
                    zone_top, zone_bottom = zone[0], zone[1]
                    if bottom <= zone_top and top >= zone_bottom:
                        zone[0] = max(zone_top, top)
                        zone[1] = min(zone_bottom, bottom)
                        zone[3] = min(zone[3], position)
                        zone[4] = min(zone[4], fill_pct(direction, zone[0], zone[1]))
                        return
        if len(zones) >= max_zones:
            zones.pop(0)
        zones.append([top, bottom, direction, position, 0.0, True, source_weight])

    ltf_bull = np.zeros(rows, dtype=bool)
    ltf_bear = np.zeros(rows, dtype=bool)
    htf_bull_signal = np.zeros(rows, dtype=bool)
    htf_bear_signal = np.zeros(rows, dtype=bool)

    # HTF voids, published one completed period later.
    htf_bull_top = htf_bull_bottom = htf_bear_top = htf_bear_bottom = None
    if use_htf:
        base = base_step_minutes(frame)
        if htf_minutes % base or htf_minutes < base:
            raise ValueError(f"高周期 {htf_minutes} 分钟必须是基础周期 {base} 分钟的整数倍")
        htf = htf_frame(frame, htf_minutes)
        htf_high = htf["high"].to_numpy(dtype=float)
        htf_low = htf["low"].to_numpy(dtype=float)
        htf_bull_raw = np.zeros(len(htf), dtype=bool)
        htf_bear_raw = np.zeros(len(htf), dtype=bool)
        htf_bull_top = np.full(len(htf), np.nan)
        htf_bull_bottom = np.full(len(htf), np.nan)
        htf_bear_top = np.full(len(htf), np.nan)
        htf_bear_bottom = np.full(len(htf), np.nan)
        for i in range(2, len(htf)):
            htf_bull_raw[i] = htf_low[i] > htf_high[i - 2]
            htf_bear_raw[i] = htf_high[i] < htf_low[i - 2]
            htf_bull_top[i] = htf_low[i]
            htf_bull_bottom[i] = htf_high[i - 2]
            htf_bear_top[i] = htf_low[i - 2]
            htf_bear_bottom[i] = htf_high[i]
        htf_bull_lifted = lift_series(frame, pd.Series(htf_bull_raw, index=htf.index),
                                      htf_minutes)
        htf_bear_lifted = lift_series(frame, pd.Series(htf_bear_raw, index=htf.index),
                                      htf_minutes)
        htf_bull_top_l = lift_series(frame, pd.Series(htf_bull_top, index=htf.index),
                                     htf_minutes)
        htf_bull_bottom_l = lift_series(frame, pd.Series(htf_bull_bottom, index=htf.index),
                                        htf_minutes)
        htf_bear_top_l = lift_series(frame, pd.Series(htf_bear_top, index=htf.index),
                                     htf_minutes)
        htf_bear_bottom_l = lift_series(frame, pd.Series(htf_bear_bottom, index=htf.index),
                                        htf_minutes)

    top1 = np.full((rows, 4), np.nan)  # score/top/bottom/direction of best zone
    top2 = np.full((rows, 4), np.nan)
    top3 = np.full((rows, 4), np.nan)
    breaker_up = np.zeros(rows, dtype=bool)
    breaker_down = np.zeros(rows, dtype=bool)
    entered_top = np.zeros(rows, dtype=bool)
    entered_bottom = np.zeros(rows, dtype=bool)

    for position in range(rows):
        if position >= 2:
            if lows[position] > highs[position - 2]:
                ltf_bull[position] = True
                add_zone(lows[position], highs[position - 2], 1, 1.0, position)
            if highs[position] < lows[position - 2]:
                ltf_bear[position] = True
                add_zone(lows[position - 2], highs[position], -1, 1.0, position)
        if use_htf:
            bull_now = bool(htf_bull_lifted.iloc[position]) \
                if not np.isnan(htf_bull_lifted.iloc[position]) else False
            bear_now = bool(htf_bear_lifted.iloc[position]) \
                if not np.isnan(htf_bear_lifted.iloc[position]) else False
            if position >= 1:
                bull_prev = bool(htf_bull_lifted.iloc[position - 1]) \
                    if not np.isnan(htf_bull_lifted.iloc[position - 1]) else False
                bear_prev = bool(htf_bear_lifted.iloc[position - 1]) \
                    if not np.isnan(htf_bear_lifted.iloc[position - 1]) else False
                if bull_now and not bull_prev:
                    top = htf_bull_top_l.iloc[position]
                    bottom = htf_bull_bottom_l.iloc[position]
                    if not np.isnan(top) and not np.isnan(bottom):
                        htf_bull_signal[position] = True
                        add_zone(float(top), float(bottom), 1, 1.2, position)
                if bear_now and not bear_prev:
                    top = htf_bear_top_l.iloc[position]
                    bottom = htf_bear_bottom_l.iloc[position]
                    if not np.isnan(top) and not np.isnan(bottom):
                        htf_bear_signal[position] = True
                        add_zone(float(top), float(bottom), -1, 1.2, position)

        scored: list[tuple[float, list]] = []
        for zone in zones:
            if not zone[5]:
                continue
            direction, top, bottom = zone[2], zone[0], zone[1]
            age = position - zone[3]
            fill_now = fill_pct(direction, top, bottom)
            prev_fill = zone[4]
            zone[4] = fill_now
            if fill_now >= 100.0:
                if prev_fill < 100.0 and position >= 1:
                    # Pine: a bullish void fully filled with a close below its
                    # bottom becomes a bearish breaker; the mirror for bears.
                    if direction == 1 and closes[position] < bottom:
                        breaker_down[position] = True
                    if direction == -1 and closes[position] > top:
                        breaker_up[position] = True
                if remove_filled:
                    zone[5] = False
                continue
            if direction == 1:
                touch_top = lows[position] <= top and lows[position - 1] > top if position else False
                touch_bottom = lows[position] <= bottom and lows[position - 1] > bottom if position else False
            else:
                touch_top = highs[position] >= top and highs[position - 1] < top if position else False
                touch_bottom = highs[position] >= bottom and highs[position - 1] < bottom if position else False
            entered_top[position] = entered_top[position] or touch_top
            entered_bottom[position] = entered_bottom[position] or touch_bottom
            freshness = min(max(1.0 - age / fresh_bars, 0.0), 1.0)
            score = (100.0 - fill_now) * (0.55 + 0.45 * freshness) * zone[6]
            scored.append((score, zone))

        scored.sort(key=lambda item: item[0], reverse=True)
        for rank, (score, zone) in enumerate(scored[:3]):
            target = (top1, top2, top3)[rank]
            target[position] = (score, zone[0], zone[1], zone[2])

    def block(target: np.ndarray, column: int) -> pd.Series:
        return pd.Series(target[:, column], index=frame.index)

    return pd.DataFrame({
        "new_void": flags(pd.Series(ltf_bull | ltf_bear | htf_bull_signal | htf_bear_signal,
                                    index=frame.index)),
        "ltf_bull_void": flags(pd.Series(ltf_bull, index=frame.index)),
        "ltf_bear_void": flags(pd.Series(ltf_bear, index=frame.index)),
        "htf_bull_void": flags(pd.Series(htf_bull_signal, index=frame.index)),
        "htf_bear_void": flags(pd.Series(htf_bear_signal, index=frame.index)),
        "breaker_up": flags(pd.Series(breaker_up, index=frame.index)),
        "breaker_down": flags(pd.Series(breaker_down, index=frame.index)),
        "entered_zone_top": flags(pd.Series(entered_top, index=frame.index)),
        "entered_zone_bottom": flags(pd.Series(entered_bottom, index=frame.index)),
        "top1_score": block(top1, 0),
        "top1_upper": block(top1, 1),
        "top1_lower": block(top1, 2),
        "top1_direction": block(top1, 3).astype("Int64"),
        "top2_score": block(top2, 0),
        "top2_upper": block(top2, 1),
        "top2_lower": block(top2, 2),
        "top2_direction": block(top2, 3).astype("Int64"),
        "top3_score": block(top3, 0),
        "top3_upper": block(top3, 1),
        "top3_lower": block(top3, 2),
        "top3_direction": block(top3, 3).astype("Int64"),
    }, index=frame.index)
