"""Live Market Scanner V7 — MACD/RSI/EMA/成交量/蜡烛五组投票合成多空评分。

Source: pine-crypto-collection/tamheed_pro/indicators/live-market-scanner-v7.pine

Author: tamheed_pro. The upstream script (Pine v5) carries no licence header; only
the formulas it publishes are reproduced here.

每根 K 线独立计算 long/short 两组加分（0..110）：MACD 穿越计 30 分、否则 MACD 顺势 20 分；
RSI 超卖/超买且连续两根回升/回落计 25 分、否则仅回升/回落 15 分；EMA9/21 穿越计 25 分、
否则多头排列 15 分；放量且价动计 15 分、否则量增价动 10 分；强势蜡烛（实体 > 0.3×ATR14）
计 10 分、否则普通价动 5 分；收盘在 EMA50 趋势一侧计 5 分。评分达到敏感度阈值
（High=55 / Medium=65 / Low=75）即给出 LONG/SHORT，两者都未达标时按分差给出
WAIT (Bullish/Bearish) 或 NEUTRAL。

上游图表元素（预测 K 线盒、TP/SL 虚线、右上方仪表盘、plotshape 标记与背景色）为纯展示，
连同 `showPrediction` / `showLevels` / `showDashboard` 开关一并丢弃；敏感度阈值保留。
`ta.macd` 用共享 EMA 原语组合（12/26/9）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (atr, choice, crossover, crossunder, ema, flags, linreg, nullable_int,
                    ohlcv, rsi, sma)


def live_market_scanner_v7(frame: pd.DataFrame, *, sensitivity: str = "Medium") -> pd.DataFrame:
    choice("sensitivity", sensitivity, ("Low", "Medium", "High"))
    min_confidence = {"High": 55, "Medium": 65, "Low": 75}[sensitivity]
    open_, close, volume = ohlcv(frame, "open", "close", "volume")

    macd_line = ema(close, 12) - ema(close, 26)
    macd_signal = ema(macd_line, 9)
    macd_histogram = macd_line - macd_signal
    macd_bullish = flags((macd_line > macd_signal) & (macd_histogram > macd_histogram.shift(1)))
    macd_bearish = flags((macd_line < macd_signal) & (macd_histogram < macd_histogram.shift(1)))
    macd_cross_up = crossover(macd_line, macd_signal)
    macd_cross_down = crossunder(macd_line, macd_signal)

    rsi_line = rsi(close, 14)
    rsi_oversold = flags(rsi_line < 35.0)
    rsi_overbought = flags(rsi_line > 65.0)
    rsi_rising = flags((rsi_line > rsi_line.shift(1)) & (rsi_line > rsi_line.shift(2)))
    rsi_falling = flags((rsi_line < rsi_line.shift(1)) & (rsi_line < rsi_line.shift(2)))

    ema9 = ema(close, 9)
    ema21 = ema(close, 21)
    ema50 = ema(close, 50)
    ema_bullish = flags(ema9 > ema21)
    ema_bearish = flags(ema9 < ema21)
    ema_cross_up = crossover(ema9, ema21)
    ema_cross_down = crossunder(ema9, ema21)

    volume_average = sma(volume, 20)
    volume_spike = flags(volume > volume_average * 1.3)
    volume_rising = flags(volume > volume.shift(1))

    average_range = atr(frame, 14)
    price_up = flags((close > open_) & (close > close.shift(1)))
    price_down = flags((close < open_) & (close < close.shift(1)))
    strong_bull_candle = flags((close - open_) > average_range * 0.3)
    strong_bear_candle = flags((open_ - close) > average_range * 0.3)

    trend = ema50
    bull_trend = flags(close > trend)
    bear_trend = flags(close < trend)

    regression = linreg(close, 20, 0)
    slope = regression - regression.shift(1)

    long_score = (30 * macd_cross_up + 20 * (macd_bullish & ~macd_cross_up)
                  + 25 * (rsi_oversold & rsi_rising)
                  + 15 * (rsi_rising & ~(rsi_oversold & rsi_rising))
                  + 25 * ema_cross_up + 15 * (ema_bullish & ~ema_cross_up)
                  + 15 * (volume_spike & price_up)
                  + 10 * (volume_rising & price_up & ~(volume_spike & price_up))
                  + 10 * strong_bull_candle + 5 * (price_up & ~strong_bull_candle)
                  + 5 * bull_trend).astype(float)
    short_score = (30 * macd_cross_down + 20 * (macd_bearish & ~macd_cross_down)
                   + 25 * (rsi_overbought & rsi_falling)
                   + 15 * (rsi_falling & ~(rsi_overbought & rsi_falling))
                   + 25 * ema_cross_down + 15 * (ema_bearish & ~ema_cross_down)
                   + 15 * (volume_spike & price_down)
                   + 10 * (volume_rising & price_down & ~(volume_spike & price_down))
                   + 10 * strong_bear_candle + 5 * (price_down & ~strong_bear_candle)
                   + 5 * bear_trend).astype(float)

    valid_long = flags(long_score >= min_confidence)
    valid_short = flags(short_score >= min_confidence)
    market_status = pd.Series(
        np.select([valid_long, valid_short, long_score > short_score, short_score > long_score],
                  ["LONG", "SHORT", "WAIT (Bullish)", "WAIT (Bearish)"], "NEUTRAL"),
        index=frame.index)

    return pd.DataFrame({
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_histogram": macd_histogram,
        "macd_bullish": macd_bullish,
        "macd_bearish": macd_bearish,
        "macd_cross_up": macd_cross_up,
        "macd_cross_down": macd_cross_down,
        "rsi": rsi_line,
        "rsi_oversold": rsi_oversold,
        "rsi_overbought": rsi_overbought,
        "rsi_rising": rsi_rising,
        "rsi_falling": rsi_falling,
        "ema9": ema9,
        "ema21": ema21,
        "ema50": ema50,
        "ema_bullish": ema_bullish,
        "ema_bearish": ema_bearish,
        "ema_cross_up": ema_cross_up,
        "ema_cross_down": ema_cross_down,
        "volume_average": volume_average,
        "volume_spike": volume_spike,
        "volume_rising": volume_rising,
        "atr": average_range,
        "price_up": price_up,
        "price_down": price_down,
        "strong_bull_candle": strong_bull_candle,
        "strong_bear_candle": strong_bear_candle,
        "trend": trend,
        "bull_trend": bull_trend,
        "bear_trend": bear_trend,
        "linreg": regression,
        "linreg_slope": slope,
        "long_score": nullable_int(long_score),
        "short_score": nullable_int(short_score),
        "valid_long": valid_long,
        "valid_short": valid_short,
        "market_status": market_status,
    }, index=frame.index)
