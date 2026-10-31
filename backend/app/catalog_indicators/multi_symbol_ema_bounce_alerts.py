"""EMA 反弹：EMA 堆叠 + VWAP + MACD 三条件共振，由状态机等待回踩 EMA9 产出 R 倍数计划。

Source: pine-crypto-collection/wlddzuk_ema_bounce/multi_symbol_ema_bounce_alerts.pine

Author: wlddzuk (TradingView); the collection carries no licence file for this script.

The strategy logic is one state machine per bar: an EMA9/EMA20 cross above a rising
EMA200 arms a "waiting for bounce" latch, the latch counts bars, and it fires when a bar
touches the EMA9 band from above (`low <= ema9 * (1 + tol%)`), closes back above the EMA9,
holds above VWAP and keeps MACD above its signal. A signal sets entry = close,
stop = low and target = entry + `rr_multiple` × risk; those three plan columns are filled
on the signal bar only and stay NaN elsewhere, which is the per-bar payload the alert loop
consumes. The latch is cleared when the EMA
stack breaks or the bar budget (`max_wait_bars`, 0 = unlimited) is exhausted.

The script's `request.security` loop over `SYMS × TFS` only builds alert payloads and
labels for *other* instruments and timeframes, so it is not reproducible without fetching
those candles; the same `signalBlock` is evaluated here on the frame the caller passes in.

VWAP uses `ta.vwap(hlc3)`, which restarts with every session: the accumulation is grouped
by the timestamps' own calendar day (a frame without a DatetimeIndex is treated as one
session). `volume` is therefore a required column.

Dropped: the drawing objects (entry/stop/target lines, labels, risk/reward boxes, the R:R
guide length), the status table, the alert payload strings and their sample secret.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (crossunder, crossover, ema, flags, integer, ma, nullable_int, number,
                    ohlcv)


def _macd(series: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series]:
    """`ta.macd`: EMA(fast) - EMA(slow), then an EMA of that line as the signal."""
    line = ema(series, fast) - ema(series, slow)
    return line, ema(line, signal)


def _session_vwap(typical: pd.Series, volume: pd.Series, index) -> pd.Series:
    """Volume-weighted `hlc3` that restarts on every session (calendar day)."""
    key = index.floor("D") if isinstance(index, pd.DatetimeIndex) else pd.Series(
        0, index=typical.index)
    traded = volume.groupby(key).cumsum()
    return (typical * volume).groupby(key).cumsum() / traded.where(traded != 0)


def multi_symbol_ema_bounce_alerts(frame: pd.DataFrame, *, fast_length: int = 9,
                                   medium_length: int = 20, slow_length: int = 200,
                                   macd_fast_length: int = 12, macd_slow_length: int = 26,
                                   macd_signal_length: int = 9,
                                   bounce_tolerance_pct: float = 0.05,
                                   max_wait_bars: int = 30,
                                   rr_multiple: float = 3.0) -> pd.DataFrame:
    integer("fast_length", fast_length, 1)
    integer("medium_length", medium_length, 1)
    integer("slow_length", slow_length, 1)
    integer("macd_fast_length", macd_fast_length, 1)
    integer("macd_slow_length", macd_slow_length, 1)
    integer("macd_signal_length", macd_signal_length, 1)
    number("bounce_tolerance_pct", bounce_tolerance_pct, 0.0)
    integer("max_wait_bars", max_wait_bars, 0)
    number("rr_multiple", rr_multiple, 0.0)

    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    typical = (high + low + close) / 3.0

    fast = ma(close, fast_length, "EMA")
    medium = ma(close, medium_length, "EMA")
    slow = ma(close, slow_length, "EMA")
    vwap = _session_vwap(typical, volume, frame.index)
    macd_line, macd_signal = _macd(close, macd_fast_length, macd_slow_length,
                                   macd_signal_length)

    bull_stack = flags((fast > medium) & (medium > slow))
    cross_up = crossover(fast, medium) & flags(fast > slow)
    tolerance_price = fast * (1.0 + bounce_tolerance_pct / 100.0)
    bounce = flags((low <= tolerance_price) & (close > fast))
    above_vwap = flags(close > vwap)
    macd_bull = flags(macd_line > macd_signal)

    count = len(frame)
    waiting = np.zeros(count, dtype=bool)
    waited = np.full(count, np.nan)
    signals = np.zeros(count, dtype=bool)
    armed = False
    bar_count: int | None = None
    for position in range(count):
        if cross_up.iloc[position]:
            armed, bar_count = True, 0
        elif armed:
            bar_count = None if bar_count is None else bar_count + 1
        if (not bull_stack.iloc[position]
                or (max_wait_bars > 0 and bar_count is not None and bar_count > max_wait_bars)):
            armed, bar_count = False, None
        waiting[position] = armed
        waited[position] = np.nan if bar_count is None else bar_count
        if (armed and bounce.iloc[position] and above_vwap.iloc[position]
                and macd_bull.iloc[position]):
            signals[position] = True
            armed, bar_count = False, None

    signal = pd.Series(signals, index=frame.index)
    entry = close.where(signal)
    stop = low.where(signal)
    risk = entry - stop
    target = (entry + rr_multiple * risk).where(signal & risk.gt(0))

    return pd.DataFrame({
        "ema_fast": fast,
        "ema_medium": medium,
        "ema_slow": slow,
        "vwap": vwap,
        "macd": macd_line,
        "macd_signal": macd_signal,
        "macd_histogram": macd_line - macd_signal,
        "bull_stack": bull_stack,
        "ema_cross_up": cross_up,
        "bounce": bounce,
        "above_vwap": above_vwap,
        "macd_bull": macd_bull,
        "waiting_for_bounce": pd.Series(waiting, index=frame.index),
        "bars_waited": nullable_int(pd.Series(waited, index=frame.index)),
        "long_signal": signal,
        "entry": entry,
        "stop": stop,
        "target": target,
        "ema_cross_down": crossunder(fast, medium),
    }, index=frame.index)
