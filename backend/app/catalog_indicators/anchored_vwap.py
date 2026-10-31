"""Anchored VWAP — 从可选锚点起累计的成交量加权均价、加权 σ 带与偏离 σ 数。

Source: pine-collection/indicators/mean_reversion/anchored_vwap/anchored_vwap.pine
(WavesUnchained, MIT)

锚点事件（`evt`）一到就清零三个累计量：Σ(p·v)、Σv、Σ(p²·v)。枢轴模式会从真正的枢轴那根
（确认那根往前 `pivot_right` 根）开始回填，所以均值从真实转折点起算而非确认那根；其余模式
只从事件当根开始。均值 = Σ(p·v)/Σv，方差 = max(0, Σ(p²·v)/Σv − 均值²)，σ = √方差，
偏离 = (close − 均值)/σ。Σv ≤ 0 时（含无真实成交量）全部保持为空。

`Session / Week / Month / Year` 用 `time("D")` 一类的周期起点变化判定事件，这里按 UTC 日历
周期起点复现（周起始为周日，与 TradingView 一致）；`Manual Date` 用「本根开盘 ≥ 锚点且上一根
< 锚点」判定，并且锚点之前不激活。

Dropped: σ 带的四条 plot 与 fill、锚点旗帜、拉伸三角标记、dashboard 表格、颜色与
`alertcondition`（纯展示/告警开关均未进签名）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import choice, flags, integer, number, ohlcv, pivotlow, pivothigh

_ANCHOR_MODES = ("Auto: Last Swing", "Auto: Swing High", "Auto: Swing Low",
                 "Session", "Week", "Month", "Year", "Manual Date")
_SOURCES = ("open", "high", "low", "close", "hl2", "hlc3", "ohlc4")
_PERIOD_FREQUENCY = {"Session": "D", "Week": "W-SAT", "Month": "M", "Year": "Y"}


def _source_series(frame: pd.DataFrame, source: str) -> pd.Series:
    high, low, close = ohlcv(frame, "high", "low", "close")
    if source == "close":
        return close
    if source == "high":
        return high
    if source == "low":
        return low
    if source == "hl2":
        return (high + low) / 2.0
    if source == "hlc3":
        return (high + low + close) / 3.0
    (opening,) = ohlcv(frame, "open")
    if source == "open":
        return opening
    return (opening + high + low + close) / 4.0


def _utc_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("该锚定模式需要 DatetimeIndex 索引")
    return index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")


def _period_starts(index: pd.DatetimeIndex, frequency: str) -> pd.Series:
    naive = index.tz_localize(None)
    starts = pd.DatetimeIndex(naive.to_period(frequency).to_timestamp()).tz_localize("UTC")
    return pd.Series(starts, index=index)


def _anchor_event(frame: pd.DataFrame, anchor_mode: str, pivot_left: int, pivot_right: int,
                  anchor_time) -> tuple[pd.Series, pd.Series]:
    """Return (evt, active) as Pine computes them on every bar."""
    if anchor_mode.startswith("Auto"):
        swing_high = pivothigh(ohlcv(frame, "high")[0], pivot_left, pivot_right)
        swing_low = pivotlow(ohlcv(frame, "low")[0], pivot_left, pivot_right)
        if anchor_mode == "Auto: Swing High":
            event = swing_high.notna()
        elif anchor_mode == "Auto: Swing Low":
            event = swing_low.notna()
        else:
            event = swing_high.notna() | swing_low.notna()
        return event, pd.Series(True, index=frame.index)

    utc = _utc_index(frame)
    times = pd.Series(utc, index=frame.index)
    if anchor_mode == "Manual Date":
        if anchor_time is None:
            raise ValueError("anchor_mode 为 Manual Date 时必须提供 anchor_time")
        anchor = pd.Timestamp(anchor_time)
        anchor = anchor.tz_localize("UTC") if anchor.tz is None else anchor.tz_convert("UTC")
        return (times >= anchor) & (times.shift(1) < anchor), times >= anchor
    starts = _period_starts(utc, _PERIOD_FREQUENCY[anchor_mode])
    return starts.ne(starts.shift(1)), pd.Series(True, index=frame.index)


def _accumulate(source: np.ndarray, volume: np.ndarray, event: np.ndarray, active: np.ndarray,
                pivot_mode: bool, pivot_right: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay Pine's `var` accumulators; NaN seeds stay NaN until the first event."""
    bars = len(source)
    sum_pv = np.full(bars, np.nan)
    sum_v = np.full(bars, np.nan)
    sum_pv2 = np.full(bars, np.nan)
    current_pv = current_v = current_pv2 = np.nan
    for bar in range(bars):
        if active[bar]:
            if event[bar] or np.isnan(current_v):
                current_pv = current_v = current_pv2 = 0.0
                if pivot_mode:
                    for offset in range(pivot_right, -1, -1):
                        position = bar - offset
                        if position < 0:
                            continue  # Pine reads na here and nz() turns it into 0
                        price = 0.0 if np.isnan(source[position]) else source[position]
                        size = 0.0 if np.isnan(volume[position]) else volume[position]
                        current_pv += price * size
                        current_v += size
                        current_pv2 += price * price * size
                else:
                    price, size = source[bar], volume[bar]
                    current_pv, current_v, current_pv2 = price * size, size, price * price * size
            else:
                price, size = source[bar], volume[bar]
                current_pv += price * size
                current_v += size
                current_pv2 += price * price * size
        sum_pv[bar], sum_v[bar], sum_pv2[bar] = current_pv, current_v, current_pv2
    return sum_pv, sum_v, sum_pv2


def anchored_vwap(frame: pd.DataFrame, *, anchor_mode: str = "Auto: Last Swing",
                  pivot_left: int = 5, pivot_right: int = 5, anchor_time=None,
                  source: str = "hlc3", band_1_sigma: float = 1.0, band_2_sigma: float = 2.0,
                  stretch_sigma: float = 2.0) -> pd.DataFrame:
    choice("anchor_mode", anchor_mode, _ANCHOR_MODES)
    integer("pivot_left", pivot_left, 1)
    integer("pivot_right", pivot_right, 1)
    choice("source", source, _SOURCES)
    number("band_1_sigma", band_1_sigma, 0.0)
    number("band_2_sigma", band_2_sigma, 0.0)
    number("stretch_sigma", stretch_sigma, 0.5)

    _, _, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    price = _source_series(frame, source)
    event, active = _anchor_event(frame, anchor_mode, pivot_left, pivot_right, anchor_time)

    # Pine gates the whole reading on a real (non-tick) volume feed: cumulative
    # volume must already be positive and the current bar's volume must exist.
    real = volume.notna() & volume.fillna(0.0).cumsum().gt(0.0)
    traded = volume.where(real)

    sum_pv, sum_v, sum_pv2 = _accumulate(
        price.to_numpy(dtype=float), traded.to_numpy(dtype=float),
        flags(event).to_numpy(dtype=bool), flags(active).to_numpy(dtype=bool),
        anchor_mode.startswith("Auto"), pivot_right)

    defined = pd.Series(sum_v, index=frame.index) > 0.0
    value = pd.Series(np.where(defined, sum_pv / np.where(defined, sum_v, 1.0), np.nan),
                      index=frame.index)
    variance = pd.Series(np.where(defined, np.maximum(0.0, sum_pv2 / np.where(defined, sum_v, 1.0)
                                                      - value.to_numpy() ** 2), np.nan),
                         index=frame.index)
    sigma = variance.pow(0.5)
    distance = (close - value) / sigma.where(sigma > 0.0)

    return pd.DataFrame({
        "avwap": value,
        "sigma": sigma,
        "upper_band_1": value + band_1_sigma * sigma,
        "lower_band_1": value - band_1_sigma * sigma,
        "upper_band_2": value + band_2_sigma * sigma,
        "lower_band_2": value - band_2_sigma * sigma,
        "distance_sigma": distance,
        "anchor_event": flags(event),
        "stretched_up": flags(active & distance.ge(stretch_sigma)),
        "stretched_dn": flags(active & distance.le(-stretch_sigma)),
    }, index=frame.index)
