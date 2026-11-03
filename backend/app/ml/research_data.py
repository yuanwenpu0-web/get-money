"""Read-only, strict candle validation for offline model research.

Times are UTC Unix milliseconds. No missing observation is imputed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import inspect, text

MINUTE = 60_000
REQUIRED = ("open_time", "close_time", "open", "high", "low", "close", "volume", "quote_volume", "trades")
ORDER_FLOW = ("taker_buy_volume", "taker_buy_quote_volume")


def load_candles(engine, symbol: str, end_ms: int | None = None) -> pd.DataFrame:
    """Load actual table columns, including optional order-flow fields, without writes."""
    columns = {c["name"] for c in inspect(engine).get_columns("klines")}
    missing = (set(REQUIRED) | {"symbol", "interval"}) - columns
    if missing:
        raise ValueError(f"klines 缺少必需字段: {', '.join(sorted(missing))}")
    selected = list(REQUIRED) + [name for name in ORDER_FLOW if name in columns]
    end_filter = " AND open_time + 60000 <= :end_ms" if end_ms is not None else ""
    query = text(f"SELECT {', '.join(selected)} FROM klines WHERE symbol=:symbol AND interval='1m'{end_filter} ORDER BY open_time")
    with engine.connect() as connection:
        return pd.read_sql_query(query, connection, params={"symbol": symbol, "end_ms": end_ms})


def validate_candles(frame: pd.DataFrame, symbol: str, as_of_ms: int) -> tuple[pd.DataFrame, dict]:
    """Return closed candles and a report; any defect makes the report fail."""
    missing = sorted((set(REQUIRED) | set(ORDER_FLOW)) - set(frame.columns))
    if set(REQUIRED) - set(frame.columns):
        raise ValueError("缺少基础 K 线字段")
    data = frame.copy()
    for col in set(REQUIRED + ORDER_FLOW) & set(data.columns):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    # Close time is inclusive in Binance; additionally enforce the whole minute.
    unfinished_mask = (data.open_time + MINUTE > as_of_ms) & (data.open_time <= as_of_ms)
    closed = data[~unfinished_mask].copy()
    unfinished = int(unfinished_mask.sum())
    closed = closed.sort_values("open_time").reset_index(drop=True)
    duplicate = int(closed.open_time.duplicated().sum())
    times = closed.open_time.dropna().drop_duplicates().sort_values()
    diffs = times.diff().dropna()
    missing_minutes = int(((diffs[diffs > MINUTE] // MINUTE) - 1).sum())
    numeric = list(REQUIRED) + [c for c in ORDER_FLOW if c in closed]
    valid = pd.Series(np.isfinite(closed[numeric]).all(axis=1), index=closed.index)
    prices = closed[["open", "high", "low", "close"]]
    valid &= (prices > 0).all(axis=1)
    valid &= closed.high >= closed[["open", "low", "close"]].max(axis=1)
    valid &= closed.low <= closed[["open", "high", "close"]].min(axis=1)
    valid &= (closed.open_time % MINUTE == 0) & (closed.close_time == closed.open_time + MINUTE - 1)
    valid &= (closed.open_time + MINUTE <= as_of_ms) & (closed.close_time < as_of_ms)
    valid &= (closed[["volume", "quote_volume", "trades"]] >= 0).all(axis=1)
    valid &= closed.trades == np.floor(closed.trades)
    if "taker_buy_volume" in closed:
        valid &= closed.taker_buy_volume.between(0, closed.volume)
    if "taker_buy_quote_volume" in closed:
        valid &= closed.taker_buy_quote_volume.between(0, closed.quote_volume)
    invalid = int((~valid).sum())
    good = len(closed) > 0 and not (missing or duplicate or missing_minutes or invalid)
    report = {
        "symbol": symbol, "interval": "1m", "timezone": "UTC", "timestamp_unit": "ms",
        "as_of_ms": as_of_ms, "total_rows": len(data), "closed_rows": len(closed),
        "unfinished_rows": unfinished, "first_open_time_ms": int(times.iloc[0]) if len(times) else None,
        "last_open_time_ms": int(times.iloc[-1]) if len(times) else None,
        "duplicate_count": duplicate, "missing_minutes": missing_minutes,
        "invalid_rows": invalid, "valid_ratio": float(valid.mean()) if len(valid) else 0.0,
        "missing_fields": missing, "passed": bool(good),
    }
    return closed, report


def aggregate_candles(frame: pd.DataFrame, minutes: int, as_of_ms: int) -> pd.DataFrame:
    """Aggregate only complete UTC buckets with all distinct consecutive minutes."""
    if minutes not in (1, 5, 15, 30, 60):
        raise ValueError("仅支持 1、5、15、30、60 分钟")
    if frame.empty:
        return frame.copy()
    step = minutes * MINUTE
    data = frame.sort_values("open_time").copy()
    data["bucket"] = data.open_time // step * step
    rows = []
    for bucket, group in data.groupby("bucket", sort=True):
        expected = np.arange(bucket, bucket + step, MINUTE)
        if bucket + step > as_of_ms or not np.array_equal(group.open_time.to_numpy(), expected):
            continue
        if (group.close_time >= as_of_ms).any():
            continue
        row = {"open_time": int(bucket), "close_time": int(bucket + step - 1),
               "open": group.open.iloc[0], "high": group.high.max(), "low": group.low.min(),
               "close": group.close.iloc[-1]}
        for column in ("volume", "quote_volume", "trades", *ORDER_FLOW):
            if column in group:
                row[column] = group[column].sum()
        rows.append(row)
    return pd.DataFrame(rows, columns=[c for c in frame.columns if c in REQUIRED + ORDER_FLOW])
