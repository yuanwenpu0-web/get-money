"""Momentum Profile — 把 WaveTrend 与 MFI 按价格区间（行）做成交量式分布剖面。

Source: pine-collection/indicators/momentum/momentum_profile/momentum_profile.pine
(WavesUnchained, MIT)

每一根 K 线都重算一次剖面：取最近 `lookback` 根的最高/最低价切成 `rows` 个等宽价格行，每根 K 线
按自己与某一行的重叠比例（占该 K 线全幅的比例）把当时的 WaveTrend 与 MFI 摊进这一行，行内除以累计
覆盖度得到该行的平均动量；|平均 WaveTrend| 最大的那一行就是动量 POC（mPOC），其行中心价即
`momentum_poc_price`。另外给出整个窗口的加权平均 WaveTrend / MFI，以及当前收盘价所在行的读数。

WaveTrend 为 `ema((hlc3 - ema(hlc3, n)) / (0.015 * ema(|hlc3 - ema(hlc3, n)|, n)), avg)`；
MFI 用典型价 × max(volume, 1) 的正负流量之比。成交量不可信（累计成交量为 0）时 MFI 整段为空，
剖面里按 Pine 的 `nz(mfi, 50)` 当中性 50 处理。

`barstate.islast` 只在最后一根画一次，这里改成滚动逐根计算；真实窗口不足 `lookback` 根时保持为空。

Dropped: 行方框、mPOC 虚线与标签、lookback 范围框、dashboard 表格、配色与 MFI 参考带着色
（颜色只是展示，`showMfi` 等纯展示开关未进签名）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import ema, highest, integer, lowest, nullable_int, nz, ohlcv


def _wavetrend(typical: pd.Series, channel_length: int, average_length: int) -> pd.Series:
    average = ema(typical, channel_length)
    deviation = ema((typical - average).abs(), channel_length)
    # Pine guards the divisor with syminfo.mintick; a flat window has no scale at
    # all, so the reading stays empty instead of exploding.
    channel = (typical - average) / (0.015 * deviation.where(deviation > 0.0))
    return ema(channel, average_length)


def _money_flow(typical: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    raw = typical * nz(volume).clip(lower=1.0)
    rising, falling = typical > typical.shift(1), typical < typical.shift(1)
    positive = raw.where(rising, 0.0).rolling(length, min_periods=length).sum()
    negative = raw.where(falling, 0.0).rolling(length, min_periods=length).sum()
    values = 100.0 - 100.0 / (1.0 + positive / negative.where(negative > 0.0))
    return values.mask(negative.eq(0.0), 100.0)


def _profile(window_high: np.ndarray, window_low: np.ndarray, weights: np.ndarray,
             flows: np.ndarray, floor: float, ceiling: float, rows: int):
    """Return (row_count, wt_sum, mfi_sum) for one window, mirroring Pine's row loop."""
    step = (ceiling - floor) / rows
    row_low = floor + step * np.arange(rows)
    row_high = row_low + step
    spread = window_high - window_low
    overlap = (np.minimum(window_high, row_high[:, None])
               - np.maximum(window_low, row_low[:, None])).clip(min=0.0)
    usable = spread > 0.0
    fraction = np.where(usable, overlap / np.where(usable, spread, 1.0), 0.0)
    return (fraction.sum(axis=1), fraction @ weights, fraction @ flows)


def momentum_profile(frame: pd.DataFrame, *, lookback: int = 200, rows: int = 25,
                     wt_channel_length: int = 10, wt_average_length: int = 21,
                     mfi_length: int = 14) -> pd.DataFrame:
    integer("lookback", lookback, 10, 1500)
    integer("rows", rows, 10, 100)
    integer("wt_channel_length", wt_channel_length, 1)
    integer("wt_average_length", wt_average_length, 1)
    integer("mfi_length", mfi_length, 1)

    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    typical = (high + low + close) / 3.0
    wavetrend = _wavetrend(typical, wt_channel_length, wt_average_length)
    # A feed without real volume leaves MFI empty; the profile then reads it as neutral 50.
    real = volume.notna() & nz(volume).cumsum().gt(0.0)
    flow = _money_flow(typical, volume, mfi_length).where(real)

    profile_low, profile_high = lowest(low, lookback), highest(high, lookback)
    highs, lows = high.to_numpy(dtype=float), low.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    weights = nz(wavetrend).to_numpy(dtype=float)
    flows = nz(flow, 50.0).to_numpy(dtype=float)

    bars = len(frame)
    poc_price = np.full(bars, np.nan)
    poc_value = np.full(bars, np.nan)
    poc_row = np.full(bars, np.nan)
    average_wt = np.full(bars, np.nan)
    average_flow = np.full(bars, np.nan)
    row_index = np.full(bars, np.nan)
    row_wt = np.full(bars, np.nan)
    row_flow = np.full(bars, np.nan)
    row_count = np.full(bars, np.nan)

    for bar in range(lookback - 1, bars):
        start = bar - lookback + 1
        floor, ceiling = lows[start:bar + 1].min(), highs[start:bar + 1].max()
        if not ceiling > floor:
            continue
        counts, wt_sum, flow_sum = _profile(highs[start:bar + 1], lows[start:bar + 1],
                                            weights[start:bar + 1], flows[start:bar + 1],
                                            floor, ceiling, rows)
        covered = counts > 0.0
        weight_average = np.where(covered, wt_sum / np.where(covered, counts, 1.0), 0.0)
        flow_average = np.where(covered, flow_sum / np.where(covered, counts, 1.0), 50.0)
        strength = np.where(covered, np.abs(weight_average), 0.0)
        best = int(np.argmax(strength))  # Pine keeps row 0 when no row is covered
        step = (ceiling - floor) / rows

        poc_price[bar] = floor + (best + 0.5) * step
        poc_value[bar] = weight_average[best]
        poc_row[bar] = best
        total = counts.sum()
        average_wt[bar] = wt_sum.sum() / total if total > 0.0 else 0.0
        average_flow[bar] = flow_sum.sum() / total if total > 0.0 else 50.0
        position = int(np.clip(np.floor((closes[bar] - floor) / step), 0, rows - 1))
        row_index[bar] = position
        row_wt[bar] = weight_average[position]
        row_flow[bar] = flow_average[position]
        row_count[bar] = counts[position]

    return pd.DataFrame({
        "profile_low": profile_low,
        "profile_high": profile_high,
        "wavetrend": wavetrend,
        "mfi": flow,
        "momentum_poc_price": poc_price,
        "momentum_poc_wavetrend": poc_value,
        "momentum_poc_row": nullable_int(pd.Series(poc_row, index=frame.index)),
        "avg_wavetrend": average_wt,
        "avg_mfi": average_flow,
        "close_row": nullable_int(pd.Series(row_index, index=frame.index)),
        "close_row_wavetrend": row_wt,
        "close_row_mfi": row_flow,
        "close_row_coverage": row_count,
    }, index=frame.index)
