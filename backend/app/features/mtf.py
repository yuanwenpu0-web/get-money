"""多周期趋势共振（MTF resonance）特征：用更高周期（4h / 1d）趋势给 1h 信号做上下文确认。

设计要点（反前视为硬红线）：
- 更高周期 OHLC 由 1h 收盘价重采样得到（pandas resample），无需任何新数据源。
- 重采样用 label="left", closed="left"：某周期桶 T 只聚合 [T, T+周期) 的 1h 数据。
- 计算完更高周期方向/ADX 后 **整体 shift(1) 一个桶**：把桶 T 的指标推到 T+周期 才可用，
  保证任意 1h bar 拿到的更高周期值只依赖 ≤ 该 bar 的 1h 数据，绝不用未来。
- 再 reindex 回 1h 索引并前向填充（ffill）。

方向用 sign(ema21/ema50 - 1)（与 1h 合成 side 解耦、确定性强）；ADX 复用 indicators._adx。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import _adx

# 更高周期（分钟）：4h、1d
DEFAULT_TF_MINUTES = (240, 1440)

# 分钟 -> 列名后缀（与 rules.evaluate_signals 读取的列名严格一致）
_TF_SUFFIX = {240: "4h", 1440: "1d", 60: "1h"}


def _tf_label(tf: int) -> str:
    return _TF_SUFFIX.get(tf, f"{tf}m")


def _higher_tf_indicators(frame: pd.DataFrame, rule: str) -> tuple[pd.Series, pd.Series]:
    """在重采样后的更高周期帧上计算方向（+1/-1）与 ADX。

    frame 必须有 DatetimeIndex 且含 open/high/low/close。返回以更高周期索引对齐的
    (direction, adx) 序列——尚未 shift，调用方负责因果移位。
    """
    agg = (
        frame[["open", "high", "low", "close"]]
        .resample(rule, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    )
    if agg.empty:
        empty = pd.Series(dtype=float, index=agg.index)
        return empty, empty
    close = agg["close"]
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    direction = np.sign(ema21 / ema50 - 1.0)
    adx, _ = _adx(agg, 14)
    return direction, adx


def add_mtf_trend(
    frame: pd.DataFrame, tf_minutes: tuple[int, ...] = DEFAULT_TF_MINUTES
) -> pd.DataFrame:
    """给 1h 帧附加多周期趋势共振列。

    新增列（每个 tf）：
      - mtf_dir_{tf}   : 更高周期方向 +1/-1/0（ema21 上穿 ema50 为多，反之为空）
      - mtf_adx_{tf}   : 更高周期 ADX_14（趋势强度过滤用）

    反前视：更高周期指标整体 shift(1) 一个桶后才 reindex 回 1h 并 ffill，确保任意 1h bar
    的更高周期值仅由 ≤ 该 bar 的 1h 数据决定。

    要求 frame 具备 DatetimeIndex（build_dataset / engine.evaluate 产出的帧均满足）。
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("add_mtf_trend 需要 DatetimeIndex（由 1h open_time 转换得到）")
    result = frame.copy()
    for tf in tf_minutes:
        # 注意：pandas 3.x 已移除 "T" 别名，分钟用 "min"（语义等同）。
        rule = f"{tf}min"
        direction, adx = _higher_tf_indicators(frame, rule)
        # 因果移位：桶 T 的指标推到 T+周期 才可用，避免 1h bar 读取含未来的桶。
        direction = direction.shift(1)
        adx = adx.shift(1)
        suffix = _tf_label(tf)
        result[f"mtf_dir_{suffix}"] = direction.reindex(frame.index).ffill()
        result[f"mtf_adx_{suffix}"] = adx.reindex(frame.index).ffill()
    return result


def mtf_columns(tf_minutes: tuple[int, ...] = DEFAULT_TF_MINUTES) -> list[str]:
    """返回 add_mtf_trend 会写入的列名（供调用方提前建列/校验）。"""
    cols: list[str] = []
    for tf in tf_minutes:
        suffix = _tf_label(tf)
        cols.append(f"mtf_dir_{suffix}")
        cols.append(f"mtf_adx_{suffix}")
    return cols
