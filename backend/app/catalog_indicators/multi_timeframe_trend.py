"""Multi-Timeframe Trend Detector — five EMAs voted into one bias.

Source: pine-crypto-collection/tamheed_pro/indicators/multi-timeframe-trend.pine

Author: tamheed_pro. The upstream script (Pine v5) carries no licence header; only
the formulas it publishes are reproduced here.

Each timeframe contributes one vote: +1 when its close is above its own EMA of
`ema_length`, -1 when below, 0 when equal. The votes are summed into
`trend_score` (-5..5), rescaled to `trend_strength` (0..100) and bucketed into
`overall_bias`. `strong_bullish` / `strong_bearish` require at least four aligned
timeframes, the `*_start` flags mark the first bar of that state, and the
confluence flags mark the bar a score of 3 is reached.

Timeframes are given in minutes (`5`, `15`, `60`, `240`, `1440` for Pine's "5",
"15", "60", "240", "D") because a bare frame carries no exchange calendar. The
base period is inferred from the finest gap between bars, a timeframe must be a
whole multiple of it, and a higher-timeframe bar is dropped unless all of its base
bars are present.

Only `close` is read, apart from the bar-open index that defines the periods.

Upstream reads the higher timeframe with `request.security` at its default
`barmerge.lookahead_off`, so a base bar inside period `p` only knows the last
closed period `p - 1`; this port reproduces that with the previous completed
period, and both the close and the EMA are taken from that same closed period.

Discarded presentation: the 3x8 dashboard table, its colours, the background tint
and the `plotshape` markers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import ema, flags, integer, ohlcv
from ._tf import aligned_index, lift_previous_period, period_frame


def _utc_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    index = frame.index
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is None or index.hasnans
            or not index.is_monotonic_increasing or not index.is_unique):
        raise ValueError("索引必须为有时区、递增且唯一的 K 线开盘时间")
    return index.tz_convert("UTC")


def _base_minutes(utc: pd.DatetimeIndex) -> int:
    """Infer the base bar period from the finest gap between consecutive bars."""
    if len(utc) < 2:
        raise ValueError("至少需要两根 K 线才能推断基础周期")
    gaps = (utc.to_series().diff().dropna() / pd.Timedelta(minutes=1)).to_numpy()
    if gaps.min() <= 0:
        raise ValueError("K 线时间必须递增")
    return int(round(float(gaps.min())))


def _trend(frame: pd.DataFrame, utc: pd.DatetimeIndex, base: int, minutes: int,
           length: int) -> pd.Series:
    """`getTrend(tf)`: close vs EMA of the previous completed period."""
    rule, step = f"{minutes}min", pd.Timedelta(minutes=minutes)
    higher = period_frame(frame, rule=rule, step=step, expected_bars=minutes // base)
    higher_close = higher["close"]
    higher_ema = ema(higher_close, length)
    close_value = lift_previous_period(higher_close, utc, rule=rule, step=step)
    ema_value = lift_previous_period(higher_ema, utc, rule=rule, step=step)
    close_value.index = ema_value.index = frame.index
    trend = np.where(close_value > ema_value, 1,
                     np.where(close_value < ema_value, -1, 0))
    return pd.Series(trend.astype(int), index=frame.index)


def _bias(score: int) -> str:
    if score > 2:
        return "strong_bullish"
    if score > 0:
        return "bullish"
    if score < -2:
        return "strong_bearish"
    if score < 0:
        return "bearish"
    return "neutral"


def _started(state: pd.Series) -> pd.Series:
    """First bar of a state, i.e. Pine's `state and not state[1]`.

    `state[1]` is na on the first bar, which makes the Pine expression na and
    therefore false, so the flag never fires on bar 0.
    """
    previous = state.shift(1).fillna(True).astype(bool)
    return flags(state & ~previous)


def multi_timeframe_trend(frame: pd.DataFrame, *, tf1_minutes: int = 5, tf2_minutes: int = 15,
                          tf3_minutes: int = 60, tf4_minutes: int = 240,
                          tf5_minutes: int = 1440, ema_length: int = 50) -> pd.DataFrame:
    integer("tf1_minutes", tf1_minutes, 1)
    integer("tf2_minutes", tf2_minutes, 1)
    integer("tf3_minutes", tf3_minutes, 1)
    integer("tf4_minutes", tf4_minutes, 1)
    integer("tf5_minutes", tf5_minutes, 1)
    integer("ema_length", ema_length, 1)
    (close,) = ohlcv(frame, "close")

    utc = _utc_index(frame)
    base = _base_minutes(utc)
    aligned_index(frame, base)
    slots = (("tf1_minutes", tf1_minutes), ("tf2_minutes", tf2_minutes),
             ("tf3_minutes", tf3_minutes), ("tf4_minutes", tf4_minutes),
             ("tf5_minutes", tf5_minutes))
    for name, minutes in slots:
        if minutes < base or minutes % base:
            raise ValueError(f"参数 {name} 必须是基础周期 {base} 分钟的整数倍")

    trends = [_trend(frame, utc, base, minutes, ema_length) for _, minutes in slots]
    score = trends[0].astype(int)
    for trend in trends[1:]:
        score = score + trend.astype(int)

    strength = score.abs() / 5 * 100
    strong_bullish = flags(score >= 4)
    strong_bearish = flags(score <= -4)
    previous_score = score.shift(1)
    return pd.DataFrame({
        "ema": ema(close, ema_length),
        "trend_1": trends[0],
        "trend_2": trends[1],
        "trend_3": trends[2],
        "trend_4": trends[3],
        "trend_5": trends[4],
        "trend_score": score,
        "trend_strength": strength,
        "overall_bias": pd.Series([_bias(int(value)) for value in score], index=frame.index,
                                  dtype=object),
        "strong_bullish": strong_bullish,
        "strong_bearish": strong_bearish,
        "strong_bullish_start": _started(strong_bullish),
        "strong_bearish_start": _started(strong_bearish),
        "bullish_confluence": flags((score >= 3) & (previous_score < 3)),
        "bearish_confluence": flags((score <= -3) & (previous_score > -3)),
    }, index=frame.index)
