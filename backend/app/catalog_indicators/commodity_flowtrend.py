"""Commodity FlowTrend — smoothed custom MFI plus CCI confirmation signals.

Source: pine-collection/indicators/money_flow/commodity_flow_trend/commodity_flow_trend.pine
(WavesUnchained, MIT)

Money flow uses the selected source times volume; the flow is signed by the
source's own direction, summed over `mfi_len` and squashed to 0..100. Both the
MFI and the CCI (of the same source) are smoothed twice with the same family
into signal lines, and the CCI is normalised to 0..100 via `50 + cci / 4`
clamped. Bull signals need an MFI/signal cross with CCI confirmation, an
OB/OS-zone origin (optional), and — optionally — a close above the price-trend
EMA; "CCI Lead" swaps the trigger for a zero-line CCI cross that still requires
MFI momentum. Expansion/accumulation/distribution states and the extreme CCI
turns are reported alongside.

A feed whose volume is entirely zero is treated as a non-real (tick) volume
feed: the flow columns stay NaN and the signals stay false, mirroring the
upstream `volumeIsReal` gate.

Dropped: the histogram/gradient colouring, background tint, plot markers, the
volume warning table and the alert plumbing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (boolean, choice, crossover, crossunder, dev, ema, flags, highest,
                    integer, lowest, number, ohlcv, sma, wma, rma)

_SMOOTHINGS = ("EMA", "SMA", "RMA", "WMA", "SuperSmoother", "T3", "KAMA", "JMA Approx")
_SOURCES = ("close", "hlc3", "hl2", "ohlc4")


def _supersmoother(series: pd.Series, length: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    a1 = np.exp(-1.414 * np.pi / length)
    c2 = 2.0 * a1 * np.cos(1.414 * np.pi / length)
    c3 = -(a1 * a1)
    c1 = 1.0 - c2 - c3
    previous, older = np.nan, np.nan
    for position, value in enumerate(values):
        if np.isnan(value):
            continue
        out[position] = (c1 * (value + (0.0 if np.isnan(previous) else previous)) / 2.0
                         + c2 * (0.0 if np.isnan(previous) else previous)
                         + c3 * (0.0 if np.isnan(older) else older))
        older, previous = previous, out[position]
    return pd.Series(out, index=series.index)


def _t3(series: pd.Series, length: int) -> pd.Series:
    e1 = ema(series, length)
    e2 = ema(e1, length)
    e3 = ema(e2, length)
    e4 = ema(e3, length)
    e5 = ema(e4, length)
    e6 = ema(e5, length)
    b = 0.7
    c1 = -(b * b * b)
    c2 = 3.0 * b * b + 3.0 * b * b * b
    c3 = -6.0 * b * b - 3.0 * b - 3.0 * b * b * b
    c4 = 1.0 + 3.0 * b + b * b * b + 3.0 * b * b
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def _kama(series: pd.Series, length: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    current = np.nan
    for position in range(length, len(values)):
        value = values[position]
        if np.isnan(value) or np.isnan(values[position - 1]):
            continue
        change = abs(value - values[position - length])
        noise = sum(abs(values[i] - values[i - 1]) for i in range(position - length + 1, position + 1))
        efficiency = change / noise if noise != 0.0 else 0.0
        smoothing = (efficiency * (2.0 / 3.0 - 2.0 / 31.0) + 2.0 / 31.0) ** 2
        seed = value if np.isnan(current) else current
        current = seed + smoothing * (value - seed)
        out[position] = current
    return pd.Series(out, index=series.index)


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
    if kind == "SuperSmoother":
        return _supersmoother(series, length)
    if kind == "T3":
        return _t3(series, length)
    if kind == "KAMA":
        return _kama(series, length)
    if kind == "JMA Approx":
        return _jma(series, length, jma_phase, jma_power)
    return ema(series, length)


def _cci_of(series: pd.Series, length: int) -> pd.Series:
    """Pine's ta.cci against an arbitrary source series."""
    average = sma(series, length)
    deviation = dev(series, length)
    return (series - average) / (0.015 * deviation.replace(0, np.nan))


def commodity_flowtrend(frame: pd.DataFrame, *, source: str = "hlc3", mfi_len: int = 14,
                        cci_len: int = 20, sig_len: int = 5, smoothing: str = "EMA",
                        jma_phase: int = 50, jma_power: float = 2.0,
                        mfi_ob: float = 80.0, mfi_os: float = 20.0, mfi_mid: float = 50.0,
                        trigger_mode: str = "MFI Lead", use_cci_confirm: bool = True,
                        use_extreme_only: bool = True, lookback_extreme: int = 8,
                        use_price_trend_filter: bool = True,
                        price_trend_len: int = 200) -> pd.DataFrame:
    choice("source", source, _SOURCES)
    integer("mfi_len", mfi_len, 1)
    integer("cci_len", cci_len, 1)
    integer("sig_len", sig_len, 1)
    choice("smoothing", smoothing, _SMOOTHINGS)
    integer("jma_phase", jma_phase, -100, 100)
    number("jma_power", jma_power, 1.0, 4.0)
    number("mfi_ob", mfi_ob, 0.0, 200.0)
    number("mfi_os", mfi_os, 0.0, 200.0)
    number("mfi_mid", mfi_mid, 0.0, 200.0)
    choice("trigger_mode", trigger_mode, ("MFI Lead", "CCI Lead"))
    boolean("use_cci_confirm", use_cci_confirm)
    boolean("use_extreme_only", use_extreme_only)
    integer("lookback_extreme", lookback_extreme, 1)
    boolean("use_price_trend_filter", use_price_trend_filter)
    integer("price_trend_len", price_trend_len, 10)
    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    if source == "close":
        src = close
    elif source == "hlc3":
        src = (high + low + close) / 3
    elif source == "hl2":
        src = (high + low) / 2
    else:
        src = (high + low + close + frame["open"].astype(float)) / 4

    volume_is_real = bool(volume.gt(0).any())
    vol = volume if volume_is_real else pd.Series(np.nan, index=frame.index)

    def smooth(series: pd.Series, length: int) -> pd.Series:
        return _smooth(series, length, smoothing, jma_phase, jma_power)

    raw_money_flow = src * vol
    positive_flow = raw_money_flow.where(src > src.shift(1), 0.0)
    negative_flow = raw_money_flow.where(src < src.shift(1), 0.0)
    positive_sum = positive_flow.rolling(mfi_len, min_periods=mfi_len).sum()
    negative_sum = negative_flow.rolling(mfi_len, min_periods=mfi_len).sum()
    mfi_raw = pd.Series(np.nan, index=frame.index)
    if volume_is_real:
        both_positive = negative_sum.gt(0) & positive_sum.gt(0)
        mfi_raw = mfi_raw.where(~(negative_sum.eq(0) & positive_sum.notna()), 100.0)
        mfi_raw = mfi_raw.mask(positive_sum.eq(0) & negative_sum.gt(0), 0.0)
        mfi_raw = mfi_raw.where(~both_positive,
                                100.0 - 100.0 / (1.0 + positive_sum / negative_sum))
    mfi = smooth(mfi_raw, sig_len)
    mfi_signal = smooth(mfi, sig_len)
    hist = mfi - mfi_signal

    cci_raw = _cci_of(src, cci_len)
    cci = smooth(cci_raw, sig_len)
    cci_norm = (50.0 + cci / 4.0).clip(0.0, 100.0)

    mfi_bull = flags(mfi > mfi_mid)
    mfi_bear = flags(mfi < mfi_mid)
    mfi_rising = flags(mfi > mfi.shift(1))
    mfi_falling = flags(mfi < mfi.shift(1))
    cci_bullish = flags(cci > 0.0)
    cci_bearish = flags(cci < 0.0)
    bull_expansion = mfi_bull & cci_bullish & mfi_rising
    bear_expansion = mfi_bear & cci_bearish & mfi_falling
    bull_accumulation = ~mfi_bull & mfi_rising & flags(cci > cci.shift(1))
    bear_distribution = ~mfi_bear & mfi_falling & flags(cci < cci.shift(1))

    was_oversold = flags(mfi.rolling(lookback_extreme, min_periods=lookback_extreme).min()
                         <= mfi_os)
    was_overbought = flags(mfi.rolling(lookback_extreme, min_periods=lookback_extreme).max()
                           >= mfi_ob)
    bull_cross = crossover(mfi, mfi_signal)
    bear_cross = crossunder(mfi, mfi_signal)
    bull_confirm = flags((not use_cci_confirm) | cci_bullish | (cci > cci.shift(1)))
    bear_confirm = flags((not use_cci_confirm) | cci_bearish | (cci < cci.shift(1)))

    price_trend = ema(close, price_trend_len)
    price_trend_bull = flags((not use_price_trend_filter) | (close > price_trend))
    price_trend_bear = flags((not use_price_trend_filter) | (close < price_trend))

    bull_mfi = (bull_cross & bull_confirm
                & ((not use_extreme_only) | was_oversold) & price_trend_bull)
    bear_mfi = (bear_cross & bear_confirm
                & ((not use_extreme_only) | was_overbought) & price_trend_bear)

    cci_cross_up = crossover(cci, pd.Series(0.0, index=frame.index))
    cci_cross_down = crossunder(cci, pd.Series(0.0, index=frame.index))
    cci_was_os = flags(cci.rolling(lookback_extreme, min_periods=lookback_extreme).min() <= -100.0)
    cci_was_ob = flags(cci.rolling(lookback_extreme, min_periods=lookback_extreme).max() >= 100.0)
    bull_cci = (cci_cross_up & mfi_rising
                & ((not use_extreme_only) | cci_was_os) & price_trend_bull)
    bear_cci = (cci_cross_down & mfi_falling
                & ((not use_extreme_only) | cci_was_ob) & price_trend_bear)

    cci_lead = trigger_mode == "CCI Lead"
    bull_signal = flags(cci_lead & bull_cci) | flags((not cci_lead) & bull_mfi)
    bear_signal = flags(cci_lead & bear_cci) | flags((not cci_lead) & bear_mfi)

    bull_mid_confirm = flags(crossover(mfi, pd.Series(mfi_mid, index=frame.index)) & cci_bullish)
    bear_mid_confirm = flags(crossunder(mfi, pd.Series(mfi_mid, index=frame.index)) & cci_bearish)
    cci_turning_up = flags((cci > cci.shift(1)) & (cci.shift(1) <= cci.shift(2))
                           & (cci.shift(1) < -100.0))
    cci_turning_down = flags((cci < cci.shift(1)) & (cci.shift(1) >= cci.shift(2))
                             & (cci.shift(1) > 100.0))

    return pd.DataFrame({
        "mfi_raw": mfi_raw,
        "mfi": mfi,
        "mfi_signal": mfi_signal,
        "histogram": hist,
        "cci_raw": cci_raw,
        "cci": cci,
        "cci_norm": cci_norm,
        "price_trend": price_trend,
        "bull_expansion": bull_expansion,
        "bear_expansion": bear_expansion,
        "bull_accumulation": bull_accumulation,
        "bear_distribution": bear_distribution,
        "bull_signal": bull_signal,
        "bear_signal": bear_signal,
        "bull_mid_confirm": bull_mid_confirm,
        "bear_mid_confirm": bear_mid_confirm,
        "cci_turning_up": cci_turning_up,
        "cci_turning_down": cci_turning_down,
    }, index=frame.index)
