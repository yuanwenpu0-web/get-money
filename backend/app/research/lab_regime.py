"""市场状态（regime）过滤器：只做**入场门控**，不改统一出场。

三种口径：

- `market_ema`  等权市场指数（全池 close 复利合成）在 4h 上相对 EMA(n) 的位置
- `own_ema`     币自身 1h 收盘相对 EMA(n)
- `own_ema_4h`  币自身 4h 收盘相对 EMA(n)

无未来函数的硬约束：4h 口径一律先 `shift(1)`（只用上一根**已收盘**的 4h），
再按 ffill 映射到 1h。1h 口径用当根收盘计算，入场仍在下一根开盘发生。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .lab_indicators import ema


def market_index(frames: dict[str, pd.DataFrame]) -> pd.Series:
    """等权市场指数：逐币 pct_change 取横截面均值后复利，缺失按 0 处理。"""
    closes = pd.DataFrame({symbol: frame["close"] for symbol, frame in frames.items()})
    returns = closes.pct_change().mean(axis=1).fillna(0.0)
    return (1.0 + returns).cumprod()


def _state_series(series: pd.Series, n: int, freq: str | None) -> pd.Series:
    if freq:
        aggregated = series.resample(freq).last().dropna()
        values = aggregated.to_numpy(dtype=float)
        state = pd.Series((values > ema(values, n)).astype(float), index=aggregated.index)
        state = state.shift(1)
    else:
        values = series.to_numpy(dtype=float)
        state = pd.Series((values > ema(values, n)).astype(float), index=series.index)
    return state.reindex(series.index, method="ffill")


def _align(state: pd.Series, index: pd.Index) -> np.ndarray:
    aligned = state.reindex(index, method="ffill").fillna(0.0).to_numpy(dtype=float)
    return aligned > 0.5


KINDS = {
    "market_ema": ("等权市场指数 4h", "4h"),
    "own_ema": ("币自身 1h", None),
    "own_ema_4h": ("币自身 4h", "4h"),
}


def build(frames: dict[str, pd.DataFrame], kind: str, n: int) -> dict[str, np.ndarray]:
    if kind not in KINDS:
        raise ValueError(f"未知 regime 口径: {kind}")
    _, freq = KINDS[kind]
    if kind == "market_ema":
        state = _state_series(market_index(frames), n, freq)
        return {symbol: _align(state, frame.index) for symbol, frame in frames.items()}
    return {
        symbol: _align(_state_series(frame["close"], n, freq), frame.index)
        for symbol, frame in frames.items()
    }


def describe(kind: str, n: int) -> str:
    label, _ = KINDS[kind]
    return f"{kind}({n}) · {label}"


def on_ratio(masks: dict[str, np.ndarray]) -> float:
    """regime 处于 on 的 bar 占比，用来判断过滤强度是否合理。"""
    total = sum(len(mask) for mask in masks.values())
    if total == 0:
        return 0.0
    return float(sum(int(mask.sum()) for mask in masks.values()) / total)
