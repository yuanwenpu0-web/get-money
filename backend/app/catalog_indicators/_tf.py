"""Helpers for projectors that lift a higher-timeframe series onto a lower one.

The catalog scripts ask TradingView for a higher timeframe with
`barmerge.lookahead_on` and freeze the value for the whole lower-timeframe
period, which lets the band drawn at the period open contain that period's
future. These helpers keep the same one-value-per-period shape but only publish
the *previous completed* period, so a caller replaying history never sees data
that had not closed yet. Periods whose lower-timeframe bars are incomplete are
dropped rather than synthesised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import sma, stdev


def aligned_index(frame: pd.DataFrame, step_minutes: int) -> pd.DatetimeIndex:
    """Validate a timezone-aware, increasing, step-aligned bar-open index."""
    index = frame.index
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is None or index.hasnans
            or not index.is_monotonic_increasing or not index.is_unique):
        raise ValueError("索引必须为有时区、递增且唯一的 K 线开盘时间")
    utc = index.tz_convert("UTC")
    misaligned = ((utc.minute.to_numpy() % step_minutes != 0) | (utc.second.to_numpy() != 0)
                  | (utc.microsecond.to_numpy() != 0) | (utc.nanosecond.to_numpy() != 0))
    if misaligned.any():
        raise ValueError(f"K 线时间必须对齐 {step_minutes} 分钟边界")
    return utc


def previous_period_bollinger(close: pd.Series, utc: pd.DatetimeIndex, *, rule: str,
                              step: pd.Timedelta, expected_bars: int, length: int,
                              multiplier: float) -> pd.DataFrame:
    """Bollinger Bands of the previous completed period, one row per input bar."""
    period = utc.floor(rule)
    grouped = close.groupby(period)
    period_close = grouped.last()
    complete = grouped.count().eq(expected_bars)
    middle = sma(period_close, length).where(complete)
    deviation = stdev(period_close, length).where(complete)
    previous = period - step
    columns = {"bb_middle": middle, "bb_upper": middle + multiplier * deviation,
               "bb_lower": middle - multiplier * deviation}
    lifted = {name: series.reindex(previous).to_numpy() for name, series in columns.items()}
    return pd.DataFrame(lifted, index=close.index)


def period_frame(frame: pd.DataFrame, rule: str, step: pd.Timedelta, expected_bars: int) -> pd.DataFrame:
    """Aggregate base bars into higher-timeframe bars, dropping incomplete periods.

    Returns a frame indexed by the period open, with OHLCV columns, so a caller can
    run any HTF indicator on it and then publish it with `lift_previous_period`.
    """
    utc = aligned_index(frame, 1)
    period = utc.floor(rule)
    grouped = frame.groupby(period)
    aggregated = pd.DataFrame({
        "open": grouped["open"].first(),
        "high": grouped["high"].max(),
        "low": grouped["low"].min(),
        "close": grouped["close"].last(),
    })
    if "volume" in frame.columns:
        aggregated["volume"] = grouped["volume"].sum()
    return aggregated[grouped["close"].count().eq(expected_bars)]


def lift_previous_period(values: pd.Series, utc: pd.DatetimeIndex, *, rule: str,
                         step: pd.Timedelta) -> pd.Series:
    """Broadcast HTF values onto base bars using the *previous completed* period.

    `values` is indexed by period open. Every base bar inside period `p` reads the
    value of `p - step`, which is the last period that had already closed when the
    bar opened. This is the lookahead-free reading of
    `request.security(sym, htf, series, lookahead_on)` used throughout the catalog.
    """
    period = utc.floor(rule)
    lifted = values.reindex(period - step)
    return pd.Series(lifted.to_numpy(dtype=float), index=utc)


def base_step_minutes(frame: pd.DataFrame) -> int:
    """Infer the base timeframe in whole minutes from the bar spacing."""
    utc = aligned_index(frame, 1)
    if len(utc) < 2:
        raise ValueError("至少需要两根 K 线才能推断周期")
    deltas = pd.Series(utc[1:] - utc[:-1]).dt.total_seconds().div(60)
    step = int(round(float(deltas.median())))
    if step <= 0:
        raise ValueError("无法推断 K 线周期")
    return step


def htf_frame(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate to a higher timeframe given in minutes, dropping incomplete periods."""
    base = base_step_minutes(frame)
    if minutes % base or minutes < base:
        raise ValueError(f"高周期 {minutes} 分钟必须是基础周期 {base} 分钟的整数倍")
    return period_frame(frame, f"{minutes}min", pd.Timedelta(minutes=minutes), minutes // base)


def lift_series(frame: pd.DataFrame, values: pd.Series, minutes: int) -> pd.Series:
    """Lift an HTF series back onto base bars using the previous completed period."""
    base = base_step_minutes(frame)
    return lift_previous_period(values, aligned_index(frame, 1), rule=f"{minutes}min",
                                step=pd.Timedelta(minutes=minutes))


def _period_starts(periods) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([period.start_time for period in periods]).tz_localize("UTC")


def calendar_frame(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate base bars into calendar periods ('D', 'W', 'M'), dropping partial periods.

    Calendar periods are not fixed-length (weeks are, months are not), so completeness is
    checked against the number of base bars each period should contain.
    """
    utc = aligned_index(frame, 1)
    base = base_step_minutes(frame)
    period = utc.tz_localize(None).to_period(rule)
    grouped = frame.groupby(period.values)
    counts = grouped["close"].count()
    starts = _period_starts(counts.index)
    expected = [round((item.end_time - item.start_time).total_seconds() / 60 / base)
                for item in counts.index]
    aggregated = pd.DataFrame({
        "open": grouped["open"].first(),
        "high": grouped["high"].max(),
        "low": grouped["low"].min(),
        "close": grouped["close"].last(),
    })
    if "volume" in frame.columns:
        aggregated["volume"] = grouped["volume"].sum()
    aggregated.index = starts
    return aggregated[counts.to_numpy() == np.asarray(expected)]


def lift_previous_calendar(frame: pd.DataFrame, values: pd.Series, rule: str) -> pd.Series:
    """Broadcast calendar-period values onto base bars using the previous completed period."""
    utc = aligned_index(frame, 1)
    previous = _period_starts((utc.tz_localize(None).to_period(rule) - 1))
    return pd.Series(values.reindex(previous).to_numpy(dtype=float), index=utc)
