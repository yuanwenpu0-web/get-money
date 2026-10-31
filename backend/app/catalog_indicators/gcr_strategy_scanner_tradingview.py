"""GCR Inflection Point Scanner — sentiment/volume/price-action confluence.

Source: pine-crypto-collection/jmragsdale_gcr/GCR_Strategy_Scanner_TradingView.pine
(jmragsdale, license header in the source file)

The scanner combines four families of evidence into inflection signals:
sentiment extremes (RSI or Stochastic beyond their thresholds), volume spikes
and exhaustion against a volume SMA, price action at the Bollinger envelope or
at confirmed pivots, and highest/lowest window divergences where price makes a
new extreme while RSI does not follow. A scalar score in roughly -85..+85 sums
the RSI, Stochastic, volume-direction, divergence and envelope contributions.

Dropped: the strategy-style plotting (EMAs, Bollinger bands, markers), the
info table and all `alertcondition` payloads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (atr, bb, boolean, ema, flags, highest, integer, lowest, number,
                    ohlcv, pivotlow, pivothigh, rsi, sma, stoch, valuewhen)


def gcr_strategy_scanner_tradingview(
        frame: pd.DataFrame, *,
        rsi_length: int = 14, rsi_ob: float = 75.0, rsi_os: float = 25.0,
        stoch_length: int = 14, stoch_ob: float = 80.0, stoch_os: float = 20.0,
        vol_ma_length: int = 20, vol_spike_mult: float = 2.0,
        vol_exhaustion_mult: float = 2.5,
        bb_length: int = 20, bb_mult: float = 2.0, atr_length: int = 14,
        pivot_left: int = 5, pivot_right: int = 2,
        ema_fast: int = 9, ema_slow: int = 21, ema_trend: int = 50,
        div_lookback: int = 30, enable_divergence: bool = True) -> pd.DataFrame:
    integer("rsi_length", rsi_length, 1)
    number("rsi_ob", rsi_ob, 50, 95)
    number("rsi_os", rsi_os, 5, 50)
    integer("stoch_length", stoch_length, 1)
    number("stoch_ob", stoch_ob, 50, 95)
    number("stoch_os", stoch_os, 5, 50)
    integer("vol_ma_length", vol_ma_length, 1)
    number("vol_spike_mult", vol_spike_mult, 1.0)
    number("vol_exhaustion_mult", vol_exhaustion_mult, 1.5)
    integer("bb_length", bb_length, 1)
    number("bb_mult", bb_mult, 0.5)
    integer("atr_length", atr_length, 1)
    integer("pivot_left", pivot_left, 1)
    integer("pivot_right", pivot_right, 1)
    integer("ema_fast", ema_fast, 1)
    integer("ema_slow", ema_slow, 1)
    integer("ema_trend", ema_trend, 1)
    integer("div_lookback", div_lookback, 10)
    boolean("enable_divergence", enable_divergence)
    open_, high, low, close, volume = ohlcv(frame, "open", "high", "low", "close", "volume")

    rsi_line = rsi(close, rsi_length)
    rsi_extreme_high = flags(rsi_line >= rsi_ob)
    rsi_extreme_low = flags(rsi_line <= rsi_os)

    stoch_k = stoch(close, high, low, stoch_length)
    stoch_d = sma(stoch_k, 3)
    stoch_extreme_high = flags(stoch_k >= stoch_ob) & flags(stoch_d >= stoch_ob)
    stoch_extreme_low = flags(stoch_k <= stoch_os) & flags(stoch_d <= stoch_os)

    vol_ma = sma(volume, vol_ma_length)
    vol_spike = flags(volume > vol_ma * vol_spike_mult)
    vol_exhaustion = flags(volume > vol_ma * vol_exhaustion_mult)
    relative_volume = volume / vol_ma.where(vol_ma > 0)

    bb_mid, bb_upper, bb_lower = bb(close, bb_length, bb_mult)
    bb_width = (bb_upper - bb_lower) / bb_mid.where(bb_mid != 0) * 100
    price_at_upper = flags(high >= bb_upper)
    price_at_lower = flags(low <= bb_lower)

    atr_line = atr(frame, atr_length)
    atr_percent = atr_line / close.where(close != 0) * 100

    ema_f = ema(close, ema_fast)
    ema_s = ema(close, ema_slow)
    ema_t = ema(close, ema_trend)
    trend_up = flags(ema_f > ema_s) & flags(close > ema_t)
    trend_down = flags(ema_f < ema_s) & flags(close < ema_t)

    pivot_high = pivothigh(high, pivot_left, pivot_right)
    pivot_low = pivotlow(low, pivot_left, pivot_right)

    # Window divergences: current half-window extreme vs the previous one.
    half = div_lookback // 2
    price_high_1 = highest(high, half)
    price_high_2 = highest(high.shift(half), half)
    price_low_1 = lowest(low, half)
    price_low_2 = lowest(low.shift(half), half)
    rsi_at_high_1 = valuewhen(flags(high.eq(price_high_1)), rsi_line, 0)
    rsi_at_high_2 = valuewhen(flags(high.eq(price_high_2)), rsi_line, 0)
    rsi_at_low_1 = valuewhen(flags(low.eq(price_low_1)), rsi_line, 0)
    rsi_at_low_2 = valuewhen(flags(low.shift(half).eq(price_low_2)), rsi_line, 0)
    bearish_div = flags(enable_divergence & (price_high_1 > price_high_2)
                        & (rsi_at_high_1 < rsi_at_high_2) & (rsi_line > 60))
    bullish_div = flags(enable_divergence & (price_low_1 < price_low_2)
                        & (rsi_at_low_1 > rsi_at_low_2) & (rsi_line < 40))

    bearish_sentiment = rsi_extreme_high | stoch_extreme_high
    bearish_volume_signal = vol_exhaustion & flags(close < open_)
    bearish_price_signal = price_at_upper | flags(pivot_high.notna()
                                                  & pivot_high.eq(high.shift(pivot_right)))
    bearish_inflection = flags(bearish_sentiment & vol_spike & bearish_price_signal
                               & (bearish_div | rsi_extreme_high))
    strong_bearish = bearish_inflection & bearish_div & vol_exhaustion

    bullish_sentiment = rsi_extreme_low | stoch_extreme_low
    bullish_volume_signal = vol_exhaustion & flags(close > open_)
    bullish_price_signal = price_at_lower | flags(pivot_low.notna()
                                                  & pivot_low.eq(low.shift(pivot_right)))
    bullish_inflection = flags(bullish_sentiment & vol_spike & bullish_price_signal
                               & (bullish_div | rsi_extreme_low))
    strong_bullish = bullish_inflection & bullish_div & vol_exhaustion

    vol_score = pd.Series(0.0, index=frame.index)
    vol_score[vol_spike] = np.where(close[vol_spike] > open_[vol_spike], 10.0, -10.0)
    score = ((50 - rsi_line) / 2 + (50 - stoch_k) * 0.3 + vol_score
             + np.where(bearish_div, -25.0, np.where(bullish_div, 25.0, 0.0))
             + np.where(price_at_upper, -25.0, np.where(price_at_lower, 25.0, 0.0)))

    return pd.DataFrame({
        "rsi": rsi_line,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "vol_ma": vol_ma,
        "vol_spike": vol_spike,
        "vol_exhaustion": vol_exhaustion,
        "relative_volume": relative_volume,
        "bb_mid": bb_mid,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "bb_width": bb_width,
        "atr": atr_line,
        "atr_percent": atr_percent,
        "ema_fast": ema_f,
        "ema_slow": ema_s,
        "ema_trend": ema_t,
        "trend_up": trend_up,
        "trend_down": trend_down,
        "pivot_high": pivot_high,
        "pivot_low": pivot_low,
        "price_high_1": price_high_1,
        "price_high_2": price_high_2,
        "price_low_1": price_low_1,
        "price_low_2": price_low_2,
        "bearish_divergence": bearish_div,
        "bullish_divergence": bullish_div,
        "bearish_volume_signal": bearish_volume_signal,
        "bullish_volume_signal": bullish_volume_signal,
        "bearish_inflection": bearish_inflection,
        "bullish_inflection": bullish_inflection,
        "strong_bearish": strong_bearish,
        "strong_bullish": strong_bullish,
        "score": score,
    }, index=frame.index)

