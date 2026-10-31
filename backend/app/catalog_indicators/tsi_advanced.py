"""TSI 进阶：双 EMA 平滑的真实强度主线，叠加信号线、趋势背景、停滞识别与背离。

Source: pine-collection/indicators/momentum/tsi_advanced/tsi_advanced.pine
(WavesUnchained, MIT)

TSI 核心与 Pine 的 `f_tsi` 一致：动量的双重 EMA 除以动量绝对值的双重 EMA 再乘 100；
与共享原语 `_pine.tsi` 的唯一差别是上游在分母为 0 时返回 0.0 而不是 NaN，这里按原样
保留（用 `_pine.ema` 组合实现）。信号线由 `smoothing_type` 选择平滑核，其中
SuperSmoother / T3 / KAMA / JMA Approx 是脚本自带递归式，Pine 的 `nz()` 语义
（未定义前驱读作 0）被逐条复刻。

趋势背景是另一组长度的 TSI；停滞检测比较 TSI 与价格在 `stall_len` 根内的位移，价格
位移以 14 周期 ATR 归一。超买/超卖穿越必须落在 `extreme_lookback` 根内触及过极值区
（`extreme_only` 关闭时不要求）；强/弱信号由停滞与背景反向（背离）扣除。情绪分是
TSI 在自身区间内的带符号深度，质量分 = 50% 区间穿透深度 + 25% 背景一致 + 25% 无停滞。

丢弃：全部颜色/填充/柱状图/信号标记/情绪条/质量标签等纯展示开关（showShadow、
showFill、showHist、showSignals、showMidCross、showSentiment、showQuality）、
颜色输入、渐变配色与告警；`src` 固定为 close。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import (atr, boolean, choice, crossover, crossunder, ema, flags, highest, integer,
                    lowest, ma, number, ohlcv)

_SMOOTHERS = ("EMA", "SMA", "RMA", "WMA", "SuperSmoother", "T3", "KAMA", "JMA Approx")


def _super_smoother(series: pd.Series, length: int) -> pd.Series:
    """Ehlers 两极平滑器；两个滞后项的 `nz` 保持递归有限。"""
    values = series.to_numpy(dtype=float)
    out = np.empty(len(values))
    angle = 1.414 * math.pi / length
    a1 = math.exp(-angle)
    c2 = 2.0 * a1 * math.cos(angle)
    c3 = -(a1 * a1)
    c1 = 1.0 - c2 - c3
    lag1 = lag2 = 0.0
    for position, value in enumerate(values):
        previous = values[position - 1] if position else np.nan
        previous = 0.0 if np.isnan(previous) else previous
        current = (c1 * (value + previous) / 2.0
                   + c2 * (0.0 if np.isnan(lag1) else lag1)
                   + c3 * (0.0 if np.isnan(lag2) else lag2))
        out[position] = current
        lag2, lag1 = lag1, current
    return pd.Series(out, index=series.index)


def _t3(series: pd.Series, length: int) -> pd.Series:
    """Tillson T3：六条链式 EMA 以 0.7 体积因子合成。"""
    b = 0.7
    e1 = ema(series, length)
    e2 = ema(e1, length)
    e3 = ema(e2, length)
    e4 = ema(e3, length)
    e5 = ema(e4, length)
    e6 = ema(e5, length)
    c1 = -(b * b * b)
    c2 = 3.0 * b * b + 3.0 * b * b * b
    c3 = -6.0 * b * b - 3.0 * b - 3.0 * b * b * b
    c4 = 1.0 + 3.0 * b + b * b * b + 3.0 * b * b
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def _kama(series: pd.Series, length: int) -> pd.Series:
    """Kaufman 自适应均线：效率比缩放平滑常数。"""
    values = series.to_numpy(dtype=float)
    change = (series - series.shift(length)).abs()
    volatility = series.diff().abs().rolling(length, min_periods=length).sum()
    efficiency = (change / volatility.where(volatility.ne(0))).where(volatility.ne(0), 0.0)
    smoothing = (efficiency * (2.0 / 3.0 - 2.0 / 31.0) + 2.0 / 31.0) ** 2
    out = np.empty(len(values))
    previous = np.nan
    for position, value in enumerate(values):
        base = value if np.isnan(previous) else previous
        current = base + smoothing.iloc[position] * (value - base)
        out[position] = current
        previous = current
    return pd.Series(out, index=series.index)


def _jma(series: pd.Series, length: int, phase: int, power: float) -> pd.Series:
    """脚本的 JMA 近似（对源序列的三极递归滤波）。"""
    values = series.to_numpy(dtype=float)
    out = np.empty(len(values))
    phase_ratio = 0.5 if phase < -100 else 2.5 if phase > 100 else phase / 100.0 + 1.5
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2.0)
    alpha = beta ** power
    e0_previous = e1_previous = e2_previous = jma_previous = np.nan
    for position, value in enumerate(values):
        e0_base = value if np.isnan(e0_previous) else e0_previous
        e0 = (1.0 - alpha) * value + alpha * e0_base
        e1 = (value - e0) * (1.0 - beta) + beta * (0.0 if np.isnan(e1_previous) else e1_previous)
        jma_base = value if np.isnan(jma_previous) else jma_previous
        e2 = ((e0 + phase_ratio * e1 - jma_base) * (1.0 - alpha) ** 2
              + alpha ** 2 * (0.0 if np.isnan(e2_previous) else e2_previous))
        current = jma_base + e2
        out[position] = current
        e0_previous, e1_previous, e2_previous, jma_previous = e0, e1, e2, current
    return pd.Series(out, index=series.index)


def _smooth(series: pd.Series, length: int, mode: str, phase: int, power: float) -> pd.Series:
    if mode == "SuperSmoother":
        return _super_smoother(series, length)
    if mode == "T3":
        return _t3(series, length)
    if mode == "KAMA":
        return _kama(series, length)
    if mode == "JMA Approx":
        return _jma(series, length, phase, power)
    return ma(series, length, mode)


def _tsi_core(close: pd.Series, long_len: int, short_len: int) -> pd.Series:
    """`f_tsi`：分母为 0 时按上游返回 0.0，暖机缺口保持 NaN。"""
    momentum = close.diff()
    double_mom = ema(ema(momentum, long_len), short_len)
    double_abs = ema(ema(momentum.abs(), long_len), short_len)
    values = 100.0 * double_mom / double_abs.replace(0, np.nan)
    return values.mask(double_abs.notna() & double_abs.eq(0), 0.0)


def tsi_advanced(frame: pd.DataFrame, *, long_len: int = 25, short_len: int = 13,
                 signal_len: int = 7, smoothing_type: str = "EMA", jma_phase: int = 50,
                 jma_power: float = 2.0, mid_line: float = 0.0, overbought: float = 25.0,
                 oversold: float = -25.0, detect_stall: bool = True, stall_len: int = 5,
                 stall_flat_atr: float = 0.5, stall_tsi_min: float = 10.0,
                 show_context: bool = True, context_long_len: int = 50,
                 context_short_len: int = 25, extreme_only: bool = True,
                 extreme_lookback: int = 5, show_divergence: bool = True,
                 divergence_len: int = 4, divergence_min_slope: float = 5.0) -> pd.DataFrame:
    integer("long_len", long_len, 1)
    integer("short_len", short_len, 1)
    integer("signal_len", signal_len, 1)
    choice("smoothing_type", smoothing_type, _SMOOTHERS)
    integer("jma_phase", jma_phase, -100, 100)
    number("jma_power", jma_power, 1.0, 4.0)
    number("mid_line", mid_line, float("-inf"), float("inf"))
    number("overbought", overbought, 0.0)
    number("oversold", oversold, float("-inf"), 0.0)
    boolean("detect_stall", detect_stall)
    integer("stall_len", stall_len, 1)
    number("stall_flat_atr", stall_flat_atr, 0.0)
    number("stall_tsi_min", stall_tsi_min, 0.0)
    boolean("show_context", show_context)
    integer("context_long_len", context_long_len, 1)
    integer("context_short_len", context_short_len, 1)
    boolean("extreme_only", extreme_only)
    integer("extreme_lookback", extreme_lookback, 1)
    boolean("show_divergence", show_divergence)
    integer("divergence_len", divergence_len, 1)
    number("divergence_min_slope", divergence_min_slope, 0.0)
    _, _, close = ohlcv(frame, "high", "low", "close")

    tsi = _tsi_core(close, long_len, short_len)
    signal = _smooth(tsi, signal_len, smoothing_type, jma_phase, jma_power)
    histogram = tsi - signal
    context = _tsi_core(close, context_long_len, context_short_len)

    was_oversold = flags(lowest(tsi, extreme_lookback) <= oversold)
    was_overbought = flags(highest(tsi, extreme_lookback) >= overbought)
    bull_extreme = flags(crossover(tsi, signal) & ((not extreme_only) | was_oversold))
    bear_extreme = flags(crossunder(tsi, signal) & ((not extreme_only) | was_overbought))
    bull_zero_cross = crossover(tsi, pd.Series(mid_line, index=frame.index))
    bear_zero_cross = crossunder(tsi, pd.Series(mid_line, index=frame.index))

    average_range = atr(frame, 14)
    tsi_change = tsi - tsi.shift(stall_len)
    price_change = (close - close.shift(stall_len)) / average_range.replace(0, np.nan)
    price_change = price_change.mask(average_range.notna() & average_range.eq(0), 0.0)
    flat_price = flags(price_change.abs() < stall_flat_atr)
    stall_bull = flags(detect_stall & (tsi_change < -stall_tsi_min) & flat_price)
    stall_bear = flags(detect_stall & (tsi_change > stall_tsi_min) & flat_price)

    context_bullish = flags(context > mid_line)
    context_bearish = flags(context < mid_line)
    bear_counter = flags(show_context & bear_extreme & context_bullish)
    bull_counter = flags(show_context & bull_extreme & context_bearish)

    bull_span = overbought - mid_line
    bear_span = mid_line - oversold
    if bull_span != 0.0:
        sent_bull = (tsi - mid_line).clip(lower=0.0) * 100.0 / bull_span
    else:
        sent_bull = pd.Series(0.0, index=frame.index)
    if bear_span != 0.0:
        sent_bear = (mid_line - tsi).clip(lower=0.0) * 100.0 / bear_span
    else:
        sent_bear = pd.Series(0.0, index=frame.index)
    sentiment = (sent_bull - sent_bear).clip(-100.0, 100.0)

    zone_low = lowest(tsi, extreme_lookback)
    zone_high = highest(tsi, extreme_lookback)
    if mid_line != oversold:
        bull_depth = ((oversold - zone_low) / abs(mid_line - oversold)).clip(0.0, 1.0)
    else:
        bull_depth = pd.Series(0.0, index=frame.index)
    if overbought != mid_line:
        bear_depth = ((zone_high - overbought) / abs(overbought - mid_line)).clip(0.0, 1.0)
    else:
        bear_depth = pd.Series(0.0, index=frame.index)
    bull_quality = (bull_depth * 50.0 + np.where(context_bullish, 25.0, 0.0)
                    + np.where(~stall_bear, 25.0, 0.0)).clip(upper=100.0)
    bear_quality = (bear_depth * 50.0 + np.where(context_bearish, 25.0, 0.0)
                    + np.where(~stall_bull, 25.0, 0.0)).clip(upper=100.0)

    fast_dir = tsi - tsi.shift(divergence_len)
    slow_dir = context - context.shift(divergence_len)
    divergence_bull = flags(show_context & show_divergence
                            & (fast_dir < -divergence_min_slope) & (slow_dir > 0.0))
    divergence_bear = flags(show_context & show_divergence
                            & (fast_dir > divergence_min_slope) & (slow_dir < 0.0))

    return pd.DataFrame({
        "tsi": tsi,
        "tsi_signal": signal,
        "tsi_histogram": histogram,
        "trend_context": context,
        "was_oversold": was_oversold,
        "was_overbought": was_overbought,
        "bull_extreme": bull_extreme,
        "bear_extreme": bear_extreme,
        "bull_extreme_strong": bull_extreme & ~stall_bear & ~bull_counter,
        "bull_extreme_weak": bull_extreme & (stall_bear | bull_counter),
        "bear_extreme_strong": bear_extreme & ~stall_bull & ~bear_counter,
        "bear_extreme_weak": bear_extreme & (stall_bull | bear_counter),
        "bull_zero_cross": bull_zero_cross,
        "bear_zero_cross": bear_zero_cross,
        "stall_bull": stall_bull,
        "stall_bear": stall_bear,
        "stalled": flags(stall_bull | stall_bear),
        "context_bullish": context_bullish,
        "context_bearish": context_bearish,
        "sentiment_score": sentiment,
        "bull_quality": bull_quality,
        "bear_quality": bear_quality,
        "trend_divergence_bull": divergence_bull,
        "trend_divergence_bear": divergence_bear,
    }, index=frame.index)
