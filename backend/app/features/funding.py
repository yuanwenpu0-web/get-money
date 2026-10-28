"""资金费特征：asof 合并到每根K线 + 极值统计。全部只用 bar 时间之前的数据（无前视）。

- funding_rate：当前生效费率（bar 时间之前最近一条资金费）
- funding_rate_annualized：rate × 3 × 365（每日 3 次结算的年化）
- funding_zscore：过去 z_window 根K线窗口内 funding_rate 的 z-score（方差 0 或数据不足 → 0）
- funding_24h_sum：过去 24 小时（按时间）累计费率
- funding_available：0/1，无资金费数据时以上特征填 0 且不报错
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FUNDING_FEATURES = [
    "funding_rate",
    "funding_rate_annualized",
    "funding_zscore",
    "funding_24h_sum",
    "funding_cum",           # 3 天滚动累积费率：期限结构/多头拥挤度代理（P6 最强信号）
    "funding_available",
]
FUNDING_Z_WINDOW = 90
FUNDING_FILTER_Z = 2.0
ANNUALIZATION = 3 * 365  # 每日 3 次结算
# 累积费率窗口：与探针验证口径一致（1h 下 72 根 ≈ 3 天）。用时间窗口而非 bar 数，
# 使该特征与 K线周期无关——任意周期都能表达"近期累计资金费"这一拥挤度代理。
FUNDING_CUM_WINDOW = "3D"


def attach_funding(frame: pd.DataFrame, funding: pd.DataFrame | None, z_window: int = FUNDING_Z_WINDOW) -> pd.DataFrame:
    """把资金费序列 asof 合并到K线（fundingTime <= bar.open_time 的最近一条），原地追加特征列。"""
    data = frame
    empty = funding is None or len(funding) == 0
    if empty:
        data["funding_rate"] = 0.0
        data["funding_rate_annualized"] = 0.0
        data["funding_zscore"] = 0.0
        data["funding_24h_sum"] = 0.0
        data["funding_cum"] = 0.0
        data["funding_available"] = 0.0
        return data

    series = funding["funding_rate"].astype(float)
    asof = series.reindex(data.index, method="ffill")
    data["funding_rate"] = asof.fillna(0.0)
    data["funding_available"] = asof.notna().astype(float)
    data["funding_rate_annualized"] = data["funding_rate"] * ANNUALIZATION

    rolling = data["funding_rate"].rolling(z_window, min_periods=z_window)
    std = rolling.std(ddof=0)
    data["funding_zscore"] = ((data["funding_rate"] - rolling.mean()) / std.replace(0, np.nan)).fillna(0.0)
    # 24h / 3D 累计按资金费事件求和（每条事件只计一次），再 asof 映射到 bar；
    # 直接对 bar 级序列求和会把同一事件重复计数数十次。
    # funding_cum = 滚动累积费率：持续正费率意味着多头持续付费、仓位拥挤，
    # 是 P6 验证出的最强独立信号（IC≈-0.09：累积费率越高越不易止盈）。
    event_sum = series.rolling("24h").sum()
    data["funding_24h_sum"] = event_sum.reindex(data.index, method="ffill").fillna(0.0)
    event_cum = series.rolling(FUNDING_CUM_WINDOW).sum()
    data["funding_cum"] = event_cum.reindex(data.index, method="ffill").fillna(0.0)
    return data


def funding_snapshot(funding: pd.DataFrame | None, z_window: int = FUNDING_Z_WINDOW) -> dict:
    """当前资金费快照（API 展示用）：最新费率、年化、z-score（按资金费记录计）、24h 累计。"""
    if funding is None or len(funding) == 0:
        return {
            "available": False,
            "funding_rate": 0.0,
            "funding_rate_annualized": 0.0,
            "funding_zscore": 0.0,
            "funding_24h_sum": 0.0,
            "latest_funding_time": 0,
        }
    series = funding["funding_rate"].astype(float)
    latest_rate = float(series.iloc[-1])
    window = series.iloc[-z_window:]
    std = float(window.std(ddof=0))
    zscore = 0.0 if std == 0 else float((latest_rate - float(window.mean())) / std)
    day_ago = series.index[-1] - pd.Timedelta(hours=24)
    return {
        "available": True,
        "funding_rate": latest_rate,
        "funding_rate_annualized": latest_rate * ANNUALIZATION,
        "funding_zscore": zscore,
        "funding_24h_sum": float(series.loc[series.index >= day_ago].sum()),
        "latest_funding_time": int(series.index[-1].timestamp() * 1000),
    }
