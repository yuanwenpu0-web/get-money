"""MTF Strategy Visual — three-timeframe EMA-stack bias with ATR gate.

Source: pine-crypto-collection/gcmishra_mtf/MTF_Strategy_Visual/indicator.pine
(MTF Strategy Visual, author gcmishra; no explicit license header)

Six EMAs (5/8/13/21/34/55) are stacked on three timeframes: the chart, a
medium timeframe and a higher timeframe. A timeframe is bullish when the six
EMAs descend strictly in period order (fastest on top), bearish when inverted.
The HTF bias gates the MTF stack; a buy fires when HTF is bullish, MTF is
aligned bullish, price crosses above the fastest chart EMA (the "cloud") and
ATR exceeds its own average times the threshold (an activity filter). The
mirror produces sells. A sideways state swallows all signals: chart EMAs
converged (fast-vs-slow spread < 0.5%), ATR contracted, or no HTF bias.

The HTF/MTF EMAs follow the repo convention of reading the previous completed
period (lookahead-free `request.security`), published with `_tf.lift_series`.

Dropped: zone shading, signal markers, dashboard, TP/SL level drawings and
the position-tracking `var` block (entry/stop/TP persistence belongs to the
broker, not the indicator).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import atr, boolean, crossover, crossunder, ema, flags, integer, number, ohlcv, sma
from ._tf import base_step_minutes, htf_frame, lift_series

_EMA_PERIODS = ("ema1_period", "ema2_period", "ema3_period", "ema4_period",
                "ema5_period", "ema6_period")


def _stack_bullish(emas: list[pd.Series]) -> pd.Series:
    out = emas[0] > emas[1]
    for faster, slower in zip(emas, emas[1:]):
        out &= faster > slower
    return out


def _stack_bearish(emas: list[pd.Series]) -> pd.Series:
    out = emas[0] < emas[1]
    for faster, slower in zip(emas, emas[1:]):
        out &= faster < slower
    return out


def mtf_strategy_visual(
        frame: pd.DataFrame, *, ema1_period: int = 5, ema2_period: int = 8,
        ema3_period: int = 13, ema4_period: int = 21, ema5_period: int = 34,
        ema6_period: int = 55, atr_period: int = 14, atr_ma_period: int = 14,
        atr_threshold: float = 1.0, atr_sideways_threshold: float = 0.8,
        htf_minutes: int = 60, mtf_minutes: int = 10,
        convergence_threshold: float = 0.5) -> pd.DataFrame:
    for name, value in zip(_EMA_PERIODS, (ema1_period, ema2_period, ema3_period,
                                          ema4_period, ema5_period, ema6_period)):
        integer(name, value, 1)
    integer("atr_period", atr_period, 1)
    integer("atr_ma_period", atr_ma_period, 1)
    number("atr_threshold", atr_threshold, 0.1)
    number("atr_sideways_threshold", atr_sideways_threshold, 0.1)
    integer("htf_minutes", htf_minutes, 1)
    integer("mtf_minutes", mtf_minutes, 1)
    number("convergence_threshold", convergence_threshold, 0.01)
    high, low, close = ohlcv(frame, "high", "low", "close")
    periods = (ema1_period, ema2_period, ema3_period, ema4_period, ema5_period, ema6_period)

    base = base_step_minutes(frame)
    for label, minutes in (("HTF", htf_minutes), ("MTF", mtf_minutes)):
        if minutes % base or minutes < base:
            raise ValueError(f"{label} {minutes} 分钟必须是基础周期 {base} 分钟的整数倍")

    chart_emas = [ema(close, period) for period in periods]

    def stacked_emas(minutes: int) -> list[pd.Series]:
        aggregated = htf_frame(frame, minutes)
        lifted = [lift_series(frame, ema(aggregated["close"], period), minutes)
                  for period in periods]
        return lifted

    htf_emas = stacked_emas(htf_minutes)
    mtf_emas = stacked_emas(mtf_minutes)

    htf_bullish = _stack_bullish(htf_emas)
    htf_bearish = _stack_bearish(htf_emas)
    htf_bias = pd.Series(np.where(htf_bullish, 1, np.where(htf_bearish, -1, 0)),
                         index=frame.index)

    mtf_bullish = _stack_bullish(mtf_emas)
    mtf_bearish = _stack_bearish(mtf_emas)
    mtf_aligned_bullish = flags((htf_bias == 1) & mtf_bullish)
    mtf_aligned_bearish = flags((htf_bias == -1) & mtf_bearish)

    ltf_bullish = _stack_bullish(chart_emas)
    ltf_bearish = _stack_bearish(chart_emas)

    price_above_cloud = flags(close > chart_emas[0])
    price_below_cloud = flags(close < chart_emas[0])
    cloud_break_up = crossover(close, chart_emas[0])
    cloud_break_down = crossunder(close, chart_emas[0])

    average_true_range = atr(frame, atr_period)
    atr_average = sma(average_true_range, atr_ma_period)
    is_atr_active = flags(average_true_range > atr_average * atr_threshold)
    is_atr_contracted = flags(average_true_range < atr_average * atr_sideways_threshold)

    ema_spread = ((chart_emas[0] - chart_emas[5]).abs()
                  / ((chart_emas[0] + chart_emas[5]) / 2) * 100.0)
    ema_converged = flags(ema_spread < convergence_threshold)
    sideways_market = flags(ema_converged | is_atr_contracted | (htf_bias == 0))

    buy_signal = flags(~sideways_market & (htf_bias == 1) & mtf_aligned_bullish
                       & cloud_break_up & is_atr_active)
    sell_signal = flags(~sideways_market & (htf_bias == -1) & mtf_aligned_bearish
                        & cloud_break_down & is_atr_active)

    return pd.DataFrame({
        "ema_fast": chart_emas[0],
        "ema_slow": chart_emas[5],
        "ema_converged": ema_converged,
        "ema_spread_pct": ema_spread,
        "htf_bullish": flags(htf_bullish),
        "htf_bearish": flags(htf_bearish),
        "htf_bias": htf_bias.astype("Int64"),
        "mtf_bullish": flags(mtf_bullish),
        "mtf_bearish": flags(mtf_bearish),
        "mtf_aligned_bullish": mtf_aligned_bullish,
        "mtf_aligned_bearish": mtf_aligned_bearish,
        "ltf_bullish": flags(ltf_bullish),
        "ltf_bearish": flags(ltf_bearish),
        "price_above_cloud": price_above_cloud,
        "price_below_cloud": price_below_cloud,
        "cloud_break_up": cloud_break_up,
        "cloud_break_down": cloud_break_down,
        "atr": average_true_range,
        "atr_average": atr_average,
        "atr_active": is_atr_active,
        "atr_contracted": is_atr_contracted,
        "sideways_market": sideways_market,
        "buy_signal": buy_signal,
        "sell_signal": sell_signal,
    }, index=frame.index)




