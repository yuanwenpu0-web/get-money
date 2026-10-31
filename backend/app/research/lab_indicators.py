"""指标原语：全部为因果计算，bar t 的取值只依赖 ≤t 的数据。

统一返回 numpy 数组（长度 = bar 数，预热期为 NaN），避免 pandas 索引对齐带来的
隐性前视。Wilder 类指标（RSI/ATR/ADX）用「前 n 期 SMA 播种 + 后续 alpha=1/n 平滑」，
与 TA-Lib / 交易平台口径一致；`ewm(adjust=False)` 直接从头累计会低估前期波动。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(values: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(values).rolling(n, min_periods=n).mean().to_numpy()


def ema(values: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(values).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


def rolling_std(values: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(values).rolling(n, min_periods=n).std(ddof=0).to_numpy()


def wilder_smooth(values: np.ndarray, n: int) -> np.ndarray:
    """Wilder 平滑：前 n 个有效值取均值播种，其后 s = s_prev + (x - s_prev)/n。"""
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan)
    if len(arr) <= n:
        return out
    seed = np.nanmean(arr[1:n + 1])
    if not np.isfinite(seed):
        return out
    out[n] = seed
    prev = seed
    for i in range(n + 1, len(arr)):
        if not np.isfinite(arr[i]):
            out[i] = prev
            continue
        prev = prev + (arr[i] - prev) / n
        out[i] = prev
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.concatenate(([np.nan], close[:-1]))
    ranges = np.vstack([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    return np.nanmax(ranges, axis=0)


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    return wilder_smooth(true_range(high, low, close), n)


def rsi_wilder(close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder RSI，返回 0–100；预热期与前 n 个变化之前为 NaN。"""
    close = np.asarray(close, dtype=float)
    out = np.full(close.shape, np.nan)
    if len(close) <= n + 1:
        return out
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    avg_gain = gain[:n].mean()
    avg_loss = loss[:n].mean()
    out[n] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(n, len(diff)):
        avg_gain = (avg_gain * (n - 1) + gain[i]) / n
        avg_loss = (avg_loss * (n - 1) + loss[i]) / n
        out[i + 1] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def macd(close: np.ndarray, fast: int, slow: int, signal: int) -> tuple[np.ndarray, np.ndarray]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False, min_periods=signal).mean().to_numpy()
    return macd_line, signal_line


def bollinger(close: np.ndarray, n: int, k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = sma(close, n)
    std = rolling_std(close, n)
    return mid, mid + k * std, mid - k * std


def donchian(high: np.ndarray, low: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """前 n 根（不含当前）的最高/最低，突破判定必须用 shift(1) 的通道。"""
    upper = pd.Series(high).rolling(n, min_periods=n).max().shift(1).to_numpy()
    lower = pd.Series(low).rolling(n, min_periods=n).min().shift(1).to_numpy()
    return upper, lower


def supertrend(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 10, mult: float = 3.0
) -> np.ndarray:
    """标准 Supertrend：返回方向数组，1 = 多头趋势，-1 = 空头趋势，0 = 预热。"""
    atr = atr_wilder(high, low, close, n)
    hl2 = (high + low) / 2.0
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    length = len(close)
    final_upper = np.full(length, np.nan)
    final_lower = np.full(length, np.nan)
    trend = np.zeros(length, dtype=int)
    for i in range(1, length):
        if not np.isfinite(atr[i]) or not np.isfinite(atr[i - 1]):
            continue
        prev_upper = final_upper[i - 1] if np.isfinite(final_upper[i - 1]) else upper[i]
        prev_lower = final_lower[i - 1] if np.isfinite(final_lower[i - 1]) else lower[i]
        final_upper[i] = upper[i] if (upper[i] < prev_upper or close[i - 1] > prev_upper) else prev_upper
        final_lower[i] = lower[i] if (lower[i] > prev_lower or close[i - 1] < prev_lower) else prev_lower
        if trend[i - 1] <= 0:
            trend[i] = 1 if close[i] > final_upper[i] else -1
        else:
            trend[i] = -1 if close[i] < final_lower[i] else 1
    return trend


def adx_di(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    up = np.concatenate(([np.nan], np.diff(high)))
    down = np.concatenate(([np.nan], -np.diff(low)))
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = atr_wilder(high, low, close, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * wilder_smooth(plus_dm, n) / atr
        minus_di = 100.0 * wilder_smooth(minus_dm, n) / atr
        total = plus_di + minus_di
        dx = np.where(total > 0, 100.0 * np.abs(plus_di - minus_di) / total, 0.0)
    adx = wilder_smooth(dx, n)
    return adx, plus_di, minus_di


def stochastic(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14, smooth: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    lowest = pd.Series(low).rolling(n, min_periods=n).min().to_numpy()
    highest = pd.Series(high).rolling(n, min_periods=n).max().to_numpy()
    span = highest - lowest
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(span > 0, 100.0 * (close - lowest) / span, np.nan)
    return k, sma(k, smooth)


def cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 20) -> np.ndarray:
    typical = (high + low + close) / 3.0
    mean = sma(typical, n)
    mad = pd.Series(typical).rolling(n, min_periods=n).apply(
        lambda window: np.abs(window - window.mean()).mean(), raw=True
    ).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mad > 0, (typical - mean) / (0.015 * mad), np.nan)


def mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, n: int = 14) -> np.ndarray:
    typical = (high + low + close) / 3.0
    flow = typical * volume
    direction = np.concatenate(([np.nan], np.diff(typical)))
    pos = np.where(direction > 0, flow, 0.0)
    neg = np.where(direction < 0, flow, 0.0)
    pos_sum = pd.Series(pos).rolling(n, min_periods=n).sum().to_numpy()
    neg_sum = pd.Series(neg).rolling(n, min_periods=n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(neg_sum > 0, pos_sum / neg_sum, np.nan)
    return 100.0 - 100.0 / (1.0 + ratio)


def roc(close: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(close).pct_change(n).to_numpy() * 100.0


def rolling_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, n: int) -> np.ndarray:
    typical = (high + low + close) / 3.0
    pv = pd.Series(typical * volume).rolling(n, min_periods=n).sum().to_numpy()
    vol = pd.Series(volume).rolling(n, min_periods=n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(vol > 0, pv / vol, np.nan)
