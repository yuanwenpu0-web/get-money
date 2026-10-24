from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from ..features.indicators import FEATURES, build_features
from ..market.data_service import MarketDataService
from ..utils import now_ms


def build_dataset(
    db: Session,
    symbol: str,
    interval: str,
    horizon: int = 5,
    days: int = 30,
    with_label: bool = True,
) -> pd.DataFrame:
    service = MarketDataService(db)
    start_ms = now_ms() - days * 86_400_000
    frame = service.load_frame(symbol, interval, start_ms=start_ms)
    if frame.empty:
        raise ValueError(f"{symbol} {interval} 本地无K线数据，请先拉取历史数据")

    data = build_features(frame, with_label=with_label, horizon=horizon)
    data = data.replace([float("inf"), float("-inf")], pd.NA)
    if with_label:
        data = data.dropna(subset=FEATURES + ["label", "future_return"])
        data["label"] = data["label"].astype(int)
    else:
        data = data.dropna(subset=FEATURES)
    return data
