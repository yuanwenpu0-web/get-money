"""Williams VIX Fix Advanced — fear/complacency spikes with quality scoring.

Source: pine-collection/indicators/volatility/williams_vix_fix_advanced/williams_vix_fix_advanced.pine
(WavesUnchained, MIT)

The classic Williams VIX Fix measures how far the close sits below its
`wvf_len`-day high: `wvf = (highest(close) - low) / highest(close) * 100`; the
bearish mirror (`bwvf`) gauges complacency above the low. A spike fires when
WVF crosses above its SMA+`mult`·stdev band (and/or a percentile level of its
`range_len` window) and clears `min_spike_level`. Stall detection flags bars
where WVF surges but price barely moves in ATR terms — thin wicks, not real
fear — and downgrades those spikes to "weak". Quality blends spike excess over
the band (50%), excess over the context EMA (25%) and the absence of a stall
(25%). Divergences mark fast-vs-context opposite moves; sentiment scales the
WVF/BWVF ratios against their bands into a signed ±100 reading.

Dropped: view modes, all plots/fills/labels and the reference level markers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (atr, boolean, choice, edge, ema, flags, highest, integer, lowest,
                    number, ohlcv, sma, stdev)

_SMOOTHINGS = ("EMA", "SMA", "RMA", "WMA", "SuperSmoother", "T3", "KAMA", "JMA Approx")


def williams_vix_fix_advanced(
        frame: pd.DataFrame, *, wvf_len: int = 22, smooth_output: bool = False,
        smooth_len: int = 3, smoothing: str = "EMA", jma_phase: int = 50,
        jma_power: float = 2.0, show_bear_wvf: bool = True, band_len: int = 20,
        band_mult: float = 2.0, range_len: int = 100, range_high_pct: float = 90.0,
        use_band: bool = True, use_percentile: bool = False,
        min_spike_level: float = 0.0, use_stall: bool = True, stall_len: int = 5,
        stall_flat_atr: float = 0.5, stall_wvf_min: float = 2.0, ctx_len: int = 44,
        divergence_len: int = 4, divergence_min_slope: float = 3.0) -> pd.DataFrame:
    integer("wvf_len", wvf_len, 2)
    integer("smooth_len", smooth_len, 1)
    choice("smoothing", smoothing, _SMOOTHINGS)
    integer("jma_phase", jma_phase, -100, 100)
    number("jma_power", jma_power, 1.0, 4.0)
    integer("band_len", band_len, 2)
    number("band_mult", band_mult, 0.1)
    integer("range_len", range_len, 10)
    number("range_high_pct", range_high_pct, 50.0, 100.0)
    number("min_spike_level", min_spike_level, 0.0)
    integer("stall_len", stall_len, 1)
    number("stall_flat_atr", stall_flat_atr, 0.0)
    number("stall_wvf_min", stall_wvf_min, 0.0)
    integer("ctx_len", ctx_len, 4)
    integer("divergence_len", divergence_len, 1)
    number("divergence_min_slope", divergence_min_slope, 0.0)
    high, low, close = ohlcv(frame, "high", "low", "close")

    from .commodity_flowtrend import _smooth as flow_smooth

    def smooth(series: pd.Series, length: int) -> pd.Series:
        return flow_smooth(series, length, smoothing, jma_phase, jma_power)

    hc = highest(close, wvf_len)
    lc = lowest(close, wvf_len)
    wvf_raw = pd.Series(np.where(hc != 0.0, (hc - low) / hc * 100.0, 0.0),
                        index=frame.index)
    bwvf_raw = pd.Series(np.where(lc != 0.0, (high - lc) / lc * 100.0, 0.0),
                         index=frame.index)
    wvf = smooth(wvf_raw, smooth_len) if smooth_output else wvf_raw
    bwvf = smooth(bwvf_raw, smooth_len) if smooth_output else bwvf_raw

    ctx_wvf = ema(wvf, ctx_len)
    ctx_bwvf = ema(bwvf, ctx_len)

    upper_band = sma(wvf, band_len) + band_mult * stdev(wvf, band_len)
    wvf_high = highest(wvf, range_len)
    wvf_low = lowest(wvf, range_len)
    percentile_level = wvf_low + range_high_pct / 100.0 * (wvf_high - wvf_low)

    spike_condition = pd.Series(False, index=frame.index)
    if use_band:
        spike_condition = spike_condition | (wvf >= upper_band)
    if use_percentile:
        spike_condition = spike_condition | (wvf >= percentile_level)
    spike_raw = flags(spike_condition & (wvf >= min_spike_level))

    b_upper_band = sma(bwvf, band_len) + band_mult * stdev(bwvf, band_len)
    b_percentile_level = (lowest(bwvf, range_len)
                          + range_high_pct / 100.0
                          * (highest(bwvf, range_len) - lowest(bwvf, range_len)))
    b_condition = pd.Series(False, index=frame.index)
    if use_band:
        b_condition = b_condition | (bwvf >= b_upper_band)
    if use_percentile:
        b_condition = b_condition | (bwvf >= b_percentile_level)
    b_spike_raw = (flags(b_condition & (bwvf >= min_spike_level)) if show_bear_wvf
                   else flags(pd.Series(False, index=frame.index)))

    atr_reference = atr(frame, 14)
    price_change = pd.Series(np.where(atr_reference != 0.0,
                                      (close - close.shift(stall_len)).abs() / atr_reference,
                                      0.0), index=frame.index)
    flat_price = flags(price_change < stall_flat_atr)
    wvf_change = wvf - wvf.shift(stall_len)
    bwvf_change = bwvf - bwvf.shift(stall_len)
    if use_stall:
        stall_now = flags((wvf_change > stall_wvf_min) & flat_price)
        b_stall_now = flags((bwvf_change > stall_wvf_min) & flat_price)
    else:
        stall_now = flags(pd.Series(False, index=frame.index))
        b_stall_now = flags(pd.Series(False, index=frame.index))

    spike_cross = edge(spike_raw)
    b_spike_cross = edge(b_spike_raw)
    spike_strong = flags(spike_cross & ~stall_now)
    spike_weak = flags(spike_cross & stall_now)
    b_spike_strong = flags(b_spike_cross & ~b_stall_now)
    b_spike_weak = flags(b_spike_cross & b_stall_now)

    spike_excess = pd.Series(np.where(upper_band != 0.0,
                                      (wvf / upper_band - 1.0).clip(0.0, 1.0), 0.0),
                             index=frame.index)
    ctx_excess = pd.Series(np.where(ctx_wvf != 0.0,
                                    (wvf / ctx_wvf - 1.0).clip(0.0, 1.0), 0.0),
                           index=frame.index)
    bull_quality = (spike_excess * 50.0 + ctx_excess * 25.0 + (~stall_now) * 25.0) \
        .clip(upper=100.0)
    b_spike_excess = pd.Series(np.where(b_upper_band != 0.0,
                                        (bwvf / b_upper_band - 1.0).clip(0.0, 1.0), 0.0),
                               index=frame.index)
    b_ctx_excess = pd.Series(np.where(ctx_bwvf != 0.0,
                                      (bwvf / ctx_bwvf - 1.0).clip(0.0, 1.0), 0.0),
                             index=frame.index)
    bear_quality = (b_spike_excess * 50.0 + b_ctx_excess * 25.0 + (~b_stall_now) * 25.0) \
        .clip(upper=100.0)

    fast_direction = wvf - wvf.shift(divergence_len)
    slow_direction = ctx_wvf - ctx_wvf.shift(divergence_len)
    divergence_bull = flags((fast_direction < -divergence_min_slope)
                            & (slow_direction > 0.0))
    divergence_bear = flags((fast_direction > divergence_min_slope)
                            & (slow_direction < 0.0))

    bull_pct = pd.Series(np.where(upper_band != 0.0, wvf / upper_band * 100.0, 0.0),
                         index=frame.index)
    bear_pct = pd.Series(np.where(b_upper_band != 0.0, bwvf / b_upper_band * 100.0, 0.0),
                         index=frame.index)
    sentiment = (bull_pct - bear_pct).clip(lower=-100.0, upper=100.0)

    return pd.DataFrame({
        "wvf": wvf,
        "bear_wvf": bwvf,
        "context_line": ctx_wvf,
        "bear_context_line": ctx_bwvf,
        "upper_band": upper_band,
        "percentile_level": percentile_level,
        "spike": spike_raw,
        "spike_cross": spike_cross,
        "spike_strong": spike_strong,
        "spike_weak": spike_weak,
        "bear_spike": b_spike_raw,
        "bear_spike_cross": b_spike_cross,
        "bear_spike_strong": b_spike_strong,
        "bear_spike_weak": b_spike_weak,
        "stall": stall_now,
        "bear_stall": b_stall_now,
        "bull_quality": bull_quality,
        "bear_quality": bear_quality,
        "divergence_bull": divergence_bull,
        "divergence_bear": divergence_bear,
        "sentiment_score": sentiment,
    }, index=frame.index)
