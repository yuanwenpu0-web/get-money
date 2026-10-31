"""Adaptive Arithmetic Candles — 用效率比驱动自适应平滑重建 K 线，并给出 0–100 趋势质量分。

Source: pine-collection/indicators/trend_direction/adaptive_arithmetic_candles/
adaptive_arithmetic_candles.pine (WavesUnchained, MIT)

核心链路：Kaufman 效率比 ER = |close − close[n]| / Σ|close − close[1]|（n 根），自适应系数
α = (ER·(fast − slow) + slow)²（fast = 2/3，slow = 2/31）；加权价 = close·0.45 + 典型价·0.35 +
线性回归值·0.20，用 α 做 EMA 得到 arith_close，再由上一根的 arith_close 与 open 混合出 arith_open；
叠加 trendPressure = tanh(回归斜率·1.8 + 归一化实体·0.35) 与 ER 得到的 bodyBoost，配合按噪声压缩
缩放的合成影线合成自适应 K 线的四价。质量分 = ER 分·0.35 + 斜率分·0.30 + 实体占比分·0.20 +
延续分·0.15（延续为真 100，否则 35），四舍五入并夹在 0–100；方向质量 = 分数 × 方向。

`barstate.isconfirmed` 相关的脉冲判定在历史回放里按「每根都已收盘」处理；暖机（ATR、回归、ER）
未就绪时质量分保持为空。

Dropped: `plotcandle` 与全部配色（蜡烛/影线/边框/背景）、底部质量分线与高质量圆点、
翻转标签、LONG/SHORT 文字标记、Data Window 调试 plot 与 `alertcondition`。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import atr, flags, highest, integer, linreg, lowest, nullable_int, number, ohlcv, sum_

_FAST = 2.0 / 3.0
_SLOW = 2.0 / 31.0


def _tanh(values: pd.Series) -> pd.Series:
    """Pine's `f_tanh`; numpy's stable version avoids the overflow of exp(2x)."""
    return pd.Series(np.tanh(values.to_numpy(dtype=float)), index=values.index)


def _recurse_close(weighted: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Pine's `arithClose`: seed with the first defined weighted price, then smooth."""
    out = np.full(len(weighted), np.nan)
    previous = np.nan
    for bar, value in enumerate(weighted):
        previous = value if np.isnan(previous) else previous + alpha[bar] * (value - previous)
        out[bar] = previous
    return out


def _recurse_open(opening: np.ndarray, smoothed: np.ndarray) -> np.ndarray:
    """Pine's `arithOpen`: blend the previous synthetic open with the previous close."""
    out = np.full(len(opening), np.nan)
    previous = np.nan
    for bar, value in enumerate(opening):
        previous = value if np.isnan(previous) else previous * 0.45 + smoothed[bar - 1] * 0.55
        out[bar] = previous
    return out


def adaptive_arithmetic_candles_quality_score(
        frame: pd.DataFrame, *, efficiency_length: int = 18, atr_length: int = 14,
        regression_length: int = 34, wick_factor: float = 0.55, signal_length: int = 8,
        min_quality: int = 65) -> pd.DataFrame:
    integer("efficiency_length", efficiency_length, 2)
    integer("atr_length", atr_length, 2)
    integer("regression_length", regression_length, 5)
    number("wick_factor", wick_factor, 0.0, 1.5)
    integer("signal_length", signal_length, 2)
    integer("min_quality", min_quality, 1, 100)

    opening, high, low, close = ohlcv(frame, "open", "high", "low", "close")

    travel = (close - close.shift(efficiency_length)).abs()
    noise = sum_(close.diff().abs(), efficiency_length)
    efficiency = (travel / noise).where(noise != 0.0, 0.0)

    average_range = atr(frame, atr_length)
    body_norm = ((close - opening) / average_range).where(average_range != 0.0, 0.0)
    regression_now = linreg(close, regression_length, 0)
    regression_prev = linreg(close, regression_length, 1)
    regression_slope = ((regression_now - regression_prev) / average_range).where(
        average_range != 0.0, 0.0)

    adaptive_alpha = (efficiency * (_FAST - _SLOW) + _SLOW) ** 2.0
    typical = (opening + high + low + close) / 4.0
    weighted = close * 0.45 + typical * 0.35 + regression_now * 0.20

    smoothed = pd.Series(_recurse_close(weighted.to_numpy(dtype=float),
                                        adaptive_alpha.to_numpy(dtype=float)),
                         index=frame.index)
    synthetic_open = pd.Series(_recurse_open(opening.to_numpy(dtype=float),
                                             smoothed.to_numpy(dtype=float)),
                               index=frame.index)

    trend_pressure = _tanh(regression_slope * 1.8 + body_norm * 0.35)
    body_boost = average_range * trend_pressure * efficiency * 0.18
    synthetic_close = smoothed + body_boost

    upper_wick = (high - np.maximum(opening, close)) * (wick_factor + (1.0 - efficiency) * 0.25)
    lower_wick = (np.minimum(opening, close) - low) * (wick_factor + (1.0 - efficiency) * 0.25)
    synthetic_high = np.maximum(np.maximum(synthetic_open, synthetic_close),
                                synthetic_close + upper_wick)
    synthetic_low = np.minimum(np.minimum(synthetic_open, synthetic_close),
                               synthetic_close - lower_wick)

    direction = pd.Series(0.0, index=frame.index)
    direction = direction.mask(synthetic_close > synthetic_open, 1.0).mask(
        synthetic_close < synthetic_open, -1.0)
    slope_score = (regression_slope.abs() * 140.0).clip(upper=100.0)
    span = synthetic_high - synthetic_low
    body_score = (((synthetic_close - synthetic_open).abs() / span).where(span != 0.0, 0.0)
                  * 140.0).clip(upper=100.0)
    follow = pd.Series(False, index=frame.index)
    follow = follow.mask(direction.eq(1.0), close > close.shift(1)).mask(
        direction.eq(-1.0), close < close.shift(1))
    follow_score = np.where(follow, 100.0, 35.0)

    raw_quality = (efficiency * 100.0 * 0.35 + slope_score * 0.30 + body_score * 0.20
                   + follow_score * 0.15)
    quality = nullable_int(raw_quality.clip(lower=0.0, upper=100.0))
    signed_quality = quality * nullable_int(direction)

    strong_bull = flags(direction.eq(1.0) & quality.ge(min_quality) & regression_slope.gt(0.0))
    strong_bear = flags(direction.eq(-1.0) & quality.ge(min_quality) & regression_slope.lt(0.0))
    break_high = flags(close > highest(high.shift(1), signal_length))
    break_low = flags(close < lowest(low.shift(1), signal_length))

    return pd.DataFrame({
        "efficiency_ratio": efficiency,
        "regression_slope": regression_slope,
        "adaptive_alpha": adaptive_alpha,
        "arith_open": synthetic_open,
        "arith_high": synthetic_high,
        "arith_low": synthetic_low,
        "arith_close": synthetic_close,
        "direction": nullable_int(direction),
        "quality": quality,
        "dir_quality": signed_quality,
        "strong_bull": strong_bull,
        "strong_bear": strong_bear,
        "break_high": break_high,
        "break_low": break_low,
        "trend_flip_up": flags(direction.eq(1.0) & direction.shift(1).eq(-1.0)
                               & quality.ge(55) & regression_slope.gt(0.0)),
        "trend_flip_down": flags(direction.eq(-1.0) & direction.shift(1).eq(1.0)
                                 & quality.ge(55) & regression_slope.lt(0.0)),
        "long_signal": strong_bull & break_high,
        "short_signal": strong_bear & break_low,
    }, index=frame.index)
