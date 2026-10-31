"""Predictability Regime Index — 用方差比、收益自相关、Hurst 与分形效率判断趋势/均值回归环境。

Source: pine-collection/indicators/trend_strength/predictability_regime_index/
predictability_regime_index.pine (WavesUnchained, MIT)

四个传感器各在快/中/慢三个窗口上取值后加权融合成 −100..+100 的 PRI：
· 方差比 VR = Var(log(src/src[k])) / (k · Var(log(src/src[1])))，> 1 偏动量；
· 收益一阶自相关（正 = 动量，负 = 回归）；
· Hurst 近似：窗口内收益对其均值的累计偏差极差 R 除以收益标准差 S，H = log(R/S)/log(窗口长度)；
· 分形效率 ER = |src − src[n]| / Σ|src − src[1]|。
原始分 = VR 分·0.38 + 自相关分·0.24 + Hurst 分·0.23 + 效率分·0.15，×100 后 EMA 成 PRI，
再 EMA 一次成信号线；置信度 = |PRI|/65 × 三传感器一致度 × 100（0..100）；
> 25 且置信度 > 35 为动量环境，< −25 且置信度 > 35 为回归环境，|PRI| ≤ 20 或置信度 < 25 为噪声。

`f_safeDiv` 只在分母为 0/空时退回兜底值（VR 退 1、R/S 退 1、ER 退 0），分子为空时结果仍为空；
因此各窗口首个 VR 读数要等 k − 1 根之后才出现，PRI 在这几个空档按 `_pine.ema` 的既有语义跨越。

Dropped: 直方图与信号线 fill、±25/±60/±100 的水平线与区间填充、渐变配色、背景色、圆点标记、
dashboard 表格与 `alertcondition`（`showHistogram` / `showSignal` / `showZones` / `showDashboard`
均为纯展示开关，未进签名）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (choice, clamp, correlation, crossover, crossunder, edge, ema, flags, integer,
                    nz, ohlcv, sma, stdev, sum_, variance)

_SOURCES = ("open", "high", "low", "close", "hl2", "hlc3", "ohlc4")
_WEIGHTS = (0.45, 0.35, 0.20)


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


def _safe_div(numerator: pd.Series, denominator: pd.Series, fallback: float) -> pd.Series:
    """Pine's `f_safeDiv`: only a zero or empty denominator falls back."""
    usable = denominator.notna() & denominator.ne(0.0)
    return (numerator / denominator.where(usable)).where(usable, fallback)


def _variance_ratio(source: pd.Series, returns: pd.Series, length: int, horizon: int) -> pd.Series:
    stacked = np.log(source / source.shift(horizon))
    return _safe_div(variance(stacked, length), horizon * variance(returns, length), 1.0)


def _hurst(returns: pd.Series, length: int) -> pd.Series:
    """Pine's `f_hurstApprox`: rescaled range of the window's returns."""
    values = returns.to_numpy(dtype=float)
    centre = sma(returns, length).to_numpy(dtype=float)
    scale = stdev(returns, length).to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for bar in range(len(values)):
        window = values[max(0, bar - length + 1):bar + 1][::-1]
        deviations = np.zeros(length)
        deviations[:len(window)] = np.nan_to_num(window, nan=0.0)
        cumulative = np.cumsum(deviations - (0.0 if np.isnan(centre[bar]) else centre[bar]))
        # Pine seeds yMax/yMin at 0, so the range always contains the starting level.
        spread = max(0.0, float(cumulative.max())) - min(0.0, float(cumulative.min()))
        ratio = spread / scale[bar] if not np.isnan(scale[bar]) and scale[bar] != 0.0 else 1.0
        out[bar] = np.log(max(ratio, 1.0)) / np.log(length)
    return pd.Series(out, index=returns.index).clip(lower=0.0, upper=1.0)


def _efficiency(source: pd.Series, length: int) -> pd.Series:
    direct = (source - source.shift(length)).abs()
    path = sum_(source.diff().abs(), length)
    return _safe_div(direct, path, 0.0)


def predictability_regime_index(frame: pd.DataFrame, *, source: str = "close",
                                fast_length: int = 24, medium_length: int = 48,
                                slow_length: int = 96, vr_fast_horizon: int = 3,
                                vr_medium_horizon: int = 5, vr_slow_horizon: int = 8,
                                score_smoothing: int = 8,
                                signal_smoothing: int = 13) -> pd.DataFrame:
    choice("source", source, _SOURCES)
    integer("fast_length", fast_length, 8)
    integer("medium_length", medium_length, 16)
    integer("slow_length", slow_length, 32)
    integer("vr_fast_horizon", vr_fast_horizon, 2)
    integer("vr_medium_horizon", vr_medium_horizon, 2)
    integer("vr_slow_horizon", vr_slow_horizon, 2)
    integer("score_smoothing", score_smoothing, 1)
    integer("signal_smoothing", signal_smoothing, 1)

    price = _source_series(frame, source)
    returns = np.log(price / price.shift(1))
    lengths = (fast_length, medium_length, slow_length)
    horizons = (vr_fast_horizon, vr_medium_horizon, vr_slow_horizon)

    variance_ratio = sum(weight * clamp((_variance_ratio(price, returns, length, horizon) - 1.0)
                                        / 0.45, -1.0, 1.0)
                         for weight, length, horizon in zip(_WEIGHTS, lengths, horizons))
    autocorrelation = clamp(sum(weight * nz(correlation(returns, returns.shift(1), length))
                                for weight, length in zip(_WEIGHTS, lengths)) * 2.5, -1.0, 1.0)
    hurst = sum(weight * _hurst(returns, length)
                for weight, length in zip((0.40, 0.40, 0.20), lengths))
    hurst_score = clamp((hurst - 0.5) * 3.0, -1.0, 1.0)
    efficiency = sum(weight * _efficiency(price, length)
                     for weight, length in zip(_WEIGHTS, lengths))
    efficiency_score = clamp((efficiency - 0.25) * 2.2, -1.0, 1.0)

    raw_score = (variance_ratio * 0.38 + autocorrelation * 0.24 + hurst_score * 0.23
                 + efficiency_score * 0.15)
    index = ema(raw_score * 100.0, score_smoothing)
    signal = ema(index, signal_smoothing)

    agreement = 1.0 - ((variance_ratio - autocorrelation).abs()
                       + (variance_ratio - hurst_score).abs()
                       + (autocorrelation - hurst_score).abs()) / 6.0
    confidence = clamp((index.abs() / 65.0) * agreement * 100.0, 0.0, 100.0)

    momentum = flags(index.gt(25.0) & confidence.gt(35.0))
    reversion = flags(index.lt(-25.0) & confidence.gt(35.0))
    noise = flags(index.abs().le(20.0) | confidence.lt(25.0))
    regime = pd.Series("MIXED", index=frame.index, dtype=object)
    regime = regime.mask(noise, "NOISE").mask(reversion, "REVERSION").mask(momentum, "MOMENTUM")

    threshold = pd.Series(25.0, index=frame.index)
    return pd.DataFrame({
        "variance_ratio_score": variance_ratio,
        "autocorrelation_score": autocorrelation,
        "hurst": hurst,
        "hurst_score": hurst_score,
        "efficiency": efficiency,
        "efficiency_score": efficiency_score,
        "pri": index,
        "signal": signal,
        "agreement": agreement,
        "confidence": confidence,
        "regime": regime,
        "momentum_regime": momentum,
        "reversion_regime": reversion,
        "noise_regime": noise,
        "momentum_shift": flags(crossover(index, threshold) & confidence.gt(35.0)),
        "reversion_shift": flags(crossunder(index, -threshold) & confidence.gt(35.0)),
        "random_walk_cross": flags((index > 0.0) & (index.shift(1) <= 0.0))
        | flags((index < 0.0) & (index.shift(1) >= 0.0)),
    }, index=frame.index)
