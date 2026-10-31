"""入场信号生成器：每个函数返回 (long, short) 两组布尔数组，长度 = bar 数。

约定：

- 数组在 bar t 为 True 表示「用 t 收盘及之前的信息判定，t+1 开盘入场」，
  因此任何 `shift(-1)` 式的未来信息都是缺陷，测试用 `test_信号不随未来数据改变` 兜住。
- `long` 与 `short` 互斥由构造保证（同一根 bar 不会既判多又判空）。
- 出场不由这里决定：全部候选共用 `rsi50_cross`（见 `lab_sim`）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import lab_indicators as ind


@dataclass(frozen=True, slots=True)
class Signals:
    long: np.ndarray
    short: np.ndarray


def _as_bool(values: np.ndarray) -> np.ndarray:
    """NaN 一律视为不触发；object dtype 也安全转 bool。"""
    arr = np.asarray(values)
    if arr.dtype == bool:
        return arr
    return np.where(np.isfinite(arr.astype(float)), arr.astype(bool), False)


def cross_up(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a 上穿 b：本根 a>b 且上一根 a<=b（任一侧 NaN 视为不穿越）。"""
    prev_a = np.concatenate(([np.nan], a[:-1]))
    prev_b = np.concatenate(([np.nan], b[:-1]))
    return _as_bool((a > b) & (prev_a <= prev_b))


def cross_down(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    prev_a = np.concatenate(([np.nan], a[:-1]))
    prev_b = np.concatenate(([np.nan], b[:-1]))
    return _as_bool((a < b) & (prev_a >= prev_b))


def _ohlc(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        frame["high"].to_numpy(dtype=float),
        frame["low"].to_numpy(dtype=float),
        frame["close"].to_numpy(dtype=float),
        frame["volume"].to_numpy(dtype=float),
    )


def ema_cross(frame: pd.DataFrame, p: dict) -> Signals:
    close = frame["close"].to_numpy(dtype=float)
    fast = ind.ema(close, int(p["fast"]))
    slow = ind.ema(close, int(p["slow"]))
    return Signals(cross_up(fast, slow), cross_down(fast, slow))


def price_sma(frame: pd.DataFrame, p: dict) -> Signals:
    close = frame["close"].to_numpy(dtype=float)
    line = ind.sma(close, int(p["n"]))
    return Signals(cross_up(close, line), cross_down(close, line))


def rsi_reversal(frame: pd.DataFrame, p: dict) -> Signals:
    close = frame["close"].to_numpy(dtype=float)
    rsi = ind.rsi_wilder(close, int(p["n"]))
    return Signals(cross_up(rsi, np.full_like(rsi, p["low"])), cross_down(rsi, np.full_like(rsi, p["high"])))


def macd_cross(frame: pd.DataFrame, p: dict) -> Signals:
    close = frame["close"].to_numpy(dtype=float)
    line, signal = ind.macd(close, int(p["fast"]), int(p["slow"]), int(p["signal"]))
    if int(p.get("mode", 0)) == 1:
        zero = np.zeros_like(line)
        return Signals(cross_up(line, zero), cross_down(line, zero))
    return Signals(cross_up(line, signal), cross_down(line, signal))


def bollinger_break(frame: pd.DataFrame, p: dict) -> Signals:
    close = frame["close"].to_numpy(dtype=float)
    _, upper, lower = ind.bollinger(close, int(p["n"]), float(p["k"]))
    return Signals(cross_up(close, upper), cross_down(close, lower))


def donchian_break(frame: pd.DataFrame, p: dict) -> Signals:
    high, low, close, _ = _ohlc(frame)
    upper, lower = ind.donchian(high, low, int(p["n"]))
    return Signals(_as_bool(close > upper), _as_bool(close < lower))


def supertrend_flip(frame: pd.DataFrame, p: dict) -> Signals:
    high, low, close, _ = _ohlc(frame)
    trend = ind.supertrend(high, low, close, int(p["n"]), float(p["mult"]))
    prev = np.concatenate(([0], trend[:-1]))
    return Signals((trend == 1) & (prev == -1), (trend == -1) & (prev == 1))


def adx_di_cross(frame: pd.DataFrame, p: dict) -> Signals:
    high, low, close, _ = _ohlc(frame)
    adx, plus_di, minus_di = ind.adx_di(high, low, close, int(p["n"]))
    gate = _as_bool(adx >= float(p["adx_min"]))
    return Signals(cross_up(plus_di, minus_di) & gate, cross_down(plus_di, minus_di) & gate)


def stoch_cross(frame: pd.DataFrame, p: dict) -> Signals:
    high, low, close, _ = _ohlc(frame)
    k, d = ind.stochastic(high, low, close, int(p["n"]), int(p["smooth"]))
    long = cross_up(k, d) & _as_bool(k < 50)
    short = cross_down(k, d) & _as_bool(k > 50)
    return Signals(long, short)


def cci_reversal(frame: pd.DataFrame, p: dict) -> Signals:
    high, low, close, _ = _ohlc(frame)
    values = ind.cci(high, low, close, int(p["n"]))
    level = float(p["level"])
    return Signals(cross_up(values, np.full_like(values, -level)), cross_down(values, np.full_like(values, level)))


def mfi_reversal(frame: pd.DataFrame, p: dict) -> Signals:
    high, low, close, volume = _ohlc(frame)
    values = ind.mfi(high, low, close, volume, int(p["n"]))
    low_level = float(p["low"])
    high_level = 100.0 - low_level
    return Signals(cross_up(values, np.full_like(values, low_level)), cross_down(values, np.full_like(values, high_level)))


def volume_break(frame: pd.DataFrame, p: dict) -> Signals:
    high, low, close, volume = _ohlc(frame)
    n = int(p["n"])
    base = ind.sma(volume, n)
    surge = _as_bool(volume > float(p["k"]) * base)
    upper, lower = ind.donchian(high, low, n)
    return Signals(surge & _as_bool(close > upper), surge & _as_bool(close < lower))


def atr_channel_break(frame: pd.DataFrame, p: dict) -> Signals:
    high, low, close, _ = _ohlc(frame)
    n = int(p["n"])
    mid = ind.sma(close, n)
    atr = ind.atr_wilder(high, low, close, n)
    band = float(p["k"]) * atr
    return Signals(cross_up(close, mid + band), cross_down(close, mid - band))


def roc_momentum(frame: pd.DataFrame, p: dict) -> Signals:
    close = frame["close"].to_numpy(dtype=float)
    values = ind.roc(close, int(p["n"]))
    thr = float(p["thr"])
    return Signals(cross_up(values, np.full_like(values, thr)), cross_down(values, np.full_like(values, -thr)))


def vwap_reclaim(frame: pd.DataFrame, p: dict) -> Signals:
    high, low, close, volume = _ohlc(frame)
    line = ind.rolling_vwap(high, low, close, volume, int(p["n"]))
    return Signals(cross_up(close, line), cross_down(close, line))


BUILDERS = {
    "ema_cross": ema_cross,
    "price_sma": price_sma,
    "rsi_reversal": rsi_reversal,
    "macd_cross": macd_cross,
    "bollinger_break": bollinger_break,
    "donchian_break": donchian_break,
    "supertrend_flip": supertrend_flip,
    "adx_di_cross": adx_di_cross,
    "stoch_cross": stoch_cross,
    "cci_reversal": cci_reversal,
    "mfi_reversal": mfi_reversal,
    "volume_break": volume_break,
    "atr_channel_break": atr_channel_break,
    "roc_momentum": roc_momentum,
    "vwap_reclaim": vwap_reclaim,
}
