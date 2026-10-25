from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..features.indicators import FEATURES, build_features
from ..market.data_service import MarketDataService
from ..utils import interval_ms, now_ms


def purged_split(
    data: pd.DataFrame, split: int, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude labels whose future outcome reaches the next time partition."""
    if horizon < 1 or split <= horizon or split >= len(data):
        raise ValueError("时间切分或预测跨度无效")
    return data.iloc[: split - horizon], data.iloc[split:]


def build_dataset(
    db: Session,
    symbol: str,
    interval: str,
    horizon: int = 5,
    days: int = 30,
    with_label: bool = True,
    end_ms: int | None = None,
) -> pd.DataFrame:
    if horizon < 1 or days < 1:
        raise ValueError("预测跨度和历史天数必须为正数")
    service = MarketDataService(db)
    as_of = min(end_ms, now_ms()) if end_ms is not None else now_ms()
    start_ms = as_of - days * 86_400_000
    frame = service.load_frame(symbol, interval, start_ms=start_ms, end_ms=as_of)
    if frame.empty:
        raise ValueError(f"{symbol} {interval} 本地无K线数据，请先拉取历史数据")

    step = interval_ms(interval)
    frame = frame.loc[(frame.index.asi8 // 1_000_000 + step) <= as_of].copy()
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    valid = np.isfinite(frame[["open", "high", "low", "close", "volume"]]).all(axis=1)
    valid &= (frame[["open", "high", "low", "close"]] > 0).all(axis=1) & frame[
        "volume"
    ].ge(0)
    valid &= frame["high"].ge(frame[["open", "close", "low"]].max(axis=1))
    valid &= frame["low"].le(frame[["open", "close", "high"]].min(axis=1))
    invalid_rows = int((~valid).sum())
    frame = frame.loc[valid]
    if frame.empty:
        raise ValueError("没有有效的已收盘K线")
    diffs = frame.index.to_series().diff().dt.total_seconds() * 1000
    gaps = diffs.gt(step)
    segments = []
    for _, segment in frame.groupby(diffs.ne(step).cumsum()):
        if len(segment) >= 60:
            segments.append(
                build_features(segment, with_label=with_label, horizon=horizon)
            )
    if not segments:
        raise ValueError("连续K线不足 60 根，无法计算指标")
    data = pd.concat(segments).replace([float("inf"), float("-inf")], np.nan)
    if with_label:
        data = data.dropna(subset=FEATURES + ["label", "future_return"])
        data["label"] = data["label"].astype(int)
    else:
        data = data.dropna(subset=FEATURES)
    data.attrs["data_quality"] = {
        "raw_bars": len(frame),
        "usable_bars": len(data),
        "invalid_bars": invalid_rows,
        "gap_count": int(gaps.sum()),
        "missing_bars": int(((diffs[gaps] / step) - 1).sum()),
        "last_candle": str(frame.index[-1]),
        "as_of_ms": as_of,
        "stale_hours": round(
            max(0, as_of - (frame.index[-1].value // 1_000_000 + step)) / 3_600_000, 2
        ),
    }
    return data
