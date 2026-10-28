"""正交衍生品情绪特征（P9）：持仓量/多空比/taker 比 asof 合并到每根K线，全部只用 bar 之前数据。

- oi_value_usd：未平仓名义价值（USD），杠杆总量代理
- oi_change_24h：24h 持仓量变化率（+杠杆流入 / −杠杆撤出）
- long_short_ratio：全市场多空账户比（散户情绪，逆向指标候选）
- top_long_short：大户持仓多空比（聪明钱情绪）
- taker_buy_sell：taker 主动买卖量比（>1 买方主导，<1 卖方主导）
- sentiment_available：0/1 降级标志

数据源 /futures/data/* 仅 ~30 天 1h 历史，因此本模块特征用于：
1) 横截面 IC 探针（500 币 × 500 bar 统计）；2) 近期增强。不做长历史 walk-forward。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SENTIMENT_FEATURES = [
    "oi_value_usd",
    "oi_change_24h",
    "long_short_ratio",
    "top_long_short",
    "taker_buy_sell",
    "sentiment_available",
]


def attach_sentiment(frame: pd.DataFrame, sentiment: pd.DataFrame | None) -> pd.DataFrame:
    """把正交情绪序列 asof 合并到K线（ts <= bar.open_time 的最近一条），原地追加特征列。"""
    data = frame
    empty = sentiment is None or len(sentiment) == 0
    if empty:
        for col in SENTIMENT_FEATURES:
            data[col] = 0.0
        return data

    oi = sentiment["open_interest"].astype(float)
    oiv = sentiment["oi_value"].astype(float)
    asof_oi = oi.reindex(data.index, method="ffill")
    data["oi_value_usd"] = oiv.reindex(data.index, method="ffill").fillna(0.0)
    # 24h 持仓量变化率：事件级 rolling 后再 asof，避免 bar 级重复计数
    oi_change = oi / oi.rolling("24h").mean().replace(0, np.nan) - 1.0
    data["oi_change_24h"] = oi_change.reindex(data.index, method="ffill").fillna(0.0).clip(-5, 5)
    for src, dst in (
        ("long_short_ratio", "long_short_ratio"),
        ("top_long_short", "top_long_short"),
        ("taker_buy_sell", "taker_buy_sell"),
    ):
        series = sentiment[src].astype(float)
        # 0 表示该时刻端点缺失（外对齐合并的补位），映射时改为 NaN 让 ffill 跳过
        series = series.replace(0.0, np.nan)
        data[dst] = series.reindex(data.index, method="ffill").fillna(1.0)
    covered = (
        asof_oi.notna().astype(float)
    )
    data["sentiment_available"] = covered
    return data
