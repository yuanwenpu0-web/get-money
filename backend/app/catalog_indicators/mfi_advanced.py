"""MFI Advanced — smoothed Money Flow Index with context/stall/quality.

Source: pine-collection/indicators/money_flow/mfi_advanced/mfi_advanced.pine
(WavesUnchained, MIT)

The Money Flow Index of the chosen source: flow = source × volume signed by
the source's own direction, summed over `mfi_len` and squashed to 0..100
(ties at zero flow read 50, all-positive 100, all-negative 0). The raw MFI is
smoothed into the fast line, then again into the signal line; a slow context
MFI (longer length) sits behind. Crosses against the signal, the midline and
the OB/OS zone history feed the extreme signals — all gated on real turnover
(the whole indicator goes inert when the feed carries no volume). Stall
detection flags sharp MFI moves with flat ATR-normalised price; quality
blends zone depth (50%), context agreement (25%) and the absence of an
opposing stall (25%). Sentiment maps the MFI into a signed ±100 reading
between the midline and the OB/OS levels.

Dropped: all plots, fills, hlines, markers and the sentiment bar label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (atr, choice, crossover, crossunder, flags, highest, integer, lowest,
                    number, ohlcv, sma, sum_)

_SMOOTHINGS = ("EMA", "SMA", "RMA", "WMA", "SuperSmoother", "T3", "KAMA", "JMA Approx")


def mfi_advanced(
        frame: pd.DataFrame, *, source: str = "hlc3", mfi_len: int = 14,
        avg_len: int = 3, sig_len: int = 9, smoothing: str = "EMA",
        jma_phase: int = 50, jma_power: float = 2.0, mid_line: float = 50.0,
        ob_level: float = 80.0, os_level: float = 20.0, use_stall: bool = True,
        stall_len: int = 5, stall_flat_atr: float = 0.5, stall_mfi_min: float = 15.0,
        use_context: bool = True, ctx_len: int = 100, use_extreme_only: bool = True,
        lookback_extreme: int = 5, divergence_len: int = 4,
        divergence_min_slope: float = 10.0) -> pd.DataFrame:
    choice("source", source, ("close", "hlc3", "hl2", "ohlc4"))
    integer("mfi_len", mfi_len, 1)
    integer("avg_len", avg_len, 1)
    integer("sig_len", sig_len, 1)
    choice("smoothing", smoothing, _SMOOTHINGS)
    integer("jma_phase", jma_phase, -100, 100)
    number("jma_power", jma_power, 1.0, 4.0)
    number("mid_line", mid_line, 0.0, 100.0)
    number("ob_level", ob_level, 50.0, 100.0)
    number("os_level", os_level, 0.0, 50.0)
    integer("stall_len", stall_len, 1)
    number("stall_flat_atr", stall_flat_atr, 0.0)
    number("stall_mfi_min", stall_mfi_min, 0.0)
    integer("ctx_len", ctx_len, 2)
    integer("lookback_extreme", lookback_extreme, 1)
    integer("divergence_len", divergence_len, 1)
    number("divergence_min_slope", divergence_min_slope, 0.0)
    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    src = {"close": close, "hlc3": (high + low + close) / 3,
           "hl2": (high + low) / 2,
           "ohlc4": (high + low + close + frame["open"].astype(float)) / 4}[source]

    volume_is_real = bool(volume.fillna(0.0).gt(0).any())
    vol = volume if volume_is_real else pd.Series(np.nan, index=frame.index)

    from .commodity_flowtrend import _smooth as flow_smooth

    def smooth(series: pd.Series, length: int) -> pd.Series:
        return flow_smooth(series, length, smoothing, jma_phase, jma_power)

    def mfi_of(series: pd.Series, length: int) -> pd.Series:
        raw_flow = series * vol
        positive = raw_flow.where(series > series.shift(1), 0.0)
        negative = raw_flow.where(series < series.shift(1), 0.0)
        positive_sum = sum_(positive, length)
        negative_sum = sum_(negative, length)
        total = positive_sum + negative_sum
        values = pd.Series(
            np.where(total == 0.0, 50.0,
                     np.where(negative_sum == 0.0, 100.0,
                              np.where(positive_sum == 0.0, 0.0,
                                       100.0 - 100.0 / (1.0 + positive_sum / negative_sum)))),
            index=frame.index)
        if not volume_is_real:
            return pd.Series(np.nan, index=frame.index)
        return values

    mfi_raw = mfi_of(src, mfi_len)
    mfi_line = smooth(mfi_raw, avg_len)
    signal = smooth(mfi_line, sig_len)
    hist = mfi_line - signal
    ctx_line = smooth(mfi_of(src, ctx_len), avg_len)

    has_volume = flags(sum_(volume.fillna(0.0), mfi_len) > 0.0)
    was_oversold = flags(lowest(mfi_line, lookback_extreme) <= os_level)
    was_overbought = flags(highest(mfi_line, lookback_extreme) >= ob_level)
    bull_cross = crossover(mfi_line, signal)
    bear_cross = crossunder(mfi_line, signal)
    bull_extreme = flags(has_volume & bull_cross
                         & ((not use_extreme_only) | was_oversold))
    bear_extreme = flags(has_volume & bear_cross
                         & ((not use_extreme_only) | was_overbought))
    bull_mid_cross = flags(has_volume & crossover(mfi_line, mid_line))
    bear_mid_cross = flags(has_volume & crossunder(mfi_line, mid_line))

    atr_reference = atr(frame, 14)
    mfi_change = mfi_line - mfi_line.shift(stall_len)
    price_change = pd.Series(np.where(atr_reference != 0.0,
                                      (close - close.shift(stall_len)) / atr_reference, 0.0),
                             index=frame.index)
    flat_price = flags(price_change.abs() < stall_flat_atr)
    if use_stall:
        stall_bull = flags(has_volume & (mfi_change < -stall_mfi_min) & flat_price)
        stall_bear = flags(has_volume & (mfi_change > stall_mfi_min) & flat_price)
    else:
        stall_bull = flags(pd.Series(False, index=frame.index))
        stall_bear = flags(pd.Series(False, index=frame.index))

    ctx_bull = flags(ctx_line > mid_line)
    ctx_bear = flags(ctx_line < mid_line)
    bear_counter = flags(use_context & bear_extreme & ctx_bull)
    bull_counter = flags(use_context & bull_extreme & ctx_bear)

    bear_strong = flags(bear_extreme & ~(stall_bull | bear_counter))
    bear_weak = flags(bear_extreme & (stall_bull | bear_counter))
    bull_strong = flags(bull_extreme & ~(stall_bear | bull_counter))
    bull_weak = flags(bull_extreme & (stall_bear | bull_counter))

    bull_pct = np.where(ob_level != mid_line,
                        (mfi_line.clip(lower=mid_line) - mid_line) / (ob_level - mid_line) * 100.0,
                        0.0)
    bear_pct = np.where(mid_line != os_level,
                        (mid_line - mfi_line.clip(upper=mid_line)) / (mid_line - os_level) * 100.0,
                        0.0)
    sentiment = pd.Series(np.clip(bull_pct - bear_pct, -100.0, 100.0), index=frame.index)

    mfi_lowest = lowest(mfi_line, lookback_extreme)
    mfi_highest = highest(mfi_line, lookback_extreme)
    bull_depth = np.where(os_level > 0.0,
                          np.clip((os_level - mfi_lowest) / os_level, 0.0, 1.0), 0.0)
    bear_depth = np.where(ob_level < 100.0,
                          np.clip((mfi_highest - ob_level) / (100.0 - ob_level), 0.0, 1.0),
                          0.0)
    bull_quality = np.minimum(100.0, bull_depth * 50.0 + (ctx_line > mid_line) * 25.0
                              + (~stall_bear) * 25.0)
    bear_quality = np.minimum(100.0, bear_depth * 50.0 + (ctx_line < mid_line) * 25.0
                              + (~stall_bull) * 25.0)

    fast_direction = mfi_line - mfi_line.shift(divergence_len)
    slow_direction = ctx_line - ctx_line.shift(divergence_len)
    divergence_bull = flags(has_volume & use_context
                            & (fast_direction < -divergence_min_slope)
                            & (slow_direction > 0.0))
    divergence_bear = flags(has_volume & use_context
                            & (fast_direction > divergence_min_slope)
                            & (slow_direction < 0.0))

    return pd.DataFrame({
        "mfi": mfi_line,
        "mfi_raw": mfi_raw,
        "signal": signal,
        "histogram": hist,
        "context_line": ctx_line,
        "has_volume": has_volume,
        "was_oversold": was_oversold,
        "was_overbought": was_overbought,
        "bull_cross": bull_cross,
        "bear_cross": bear_cross,
        "bull_extreme": bull_extreme,
        "bear_extreme": bear_extreme,
        "bull_strong": bull_strong,
        "bull_weak": bull_weak,
        "bear_strong": bear_strong,
        "bear_weak": bear_weak,
        "bull_mid_cross": bull_mid_cross,
        "bear_mid_cross": bear_mid_cross,
        "stall_bull": stall_bull,
        "stall_bear": stall_bear,
        "counter_trend_bull": bull_counter,
        "counter_trend_bear": bear_counter,
        "divergence_bull": divergence_bull,
        "divergence_bear": divergence_bear,
        "sentiment_score": sentiment,
        "bull_quality": pd.Series(bull_quality, index=frame.index),
        "bear_quality": pd.Series(bear_quality, index=frame.index),
    }, index=frame.index)
