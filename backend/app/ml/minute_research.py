"""Leakage-aware 15-minute offline research features and labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import hashlib

HORIZON = 15
FEATURE_VERSION = "minute-v1"
LABEL_VERSION = "return-quantile-v1"


def data_fingerprint(frame):
    columns = ["open_time", "close_time", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"]
    hashed = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(dtype="<u8")
    digest = hashlib.sha256()
    digest.update("|".join(columns).encode())
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def validation_blocks(times, horizon=HORIZON):
    """Four chronological roles, purged so outcomes cannot cross boundaries."""
    times = np.asarray(times)
    if len(times) < 4 or np.any(np.diff(times) <= 0):
        raise ValueError("验证集时间必须严格递增且至少包含四个样本")
    edges = [len(times) * i // 4 for i in range(5)]
    blocks = []
    for i in range(4):
        end = edges[i + 1]
        if i < 3:
            end = min(end, int(np.searchsorted(times, times[end] - horizon * 60000, side="left")))
        if end <= edges[i]:
            raise ValueError("验证集不足以隔离模型选择、校准和融合")
        blocks.append(slice(edges[i], end))
    return blocks


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Every feature at row t uses candles through t only."""
    close = frame["close"]
    returns = close.pct_change(fill_method=None)
    result = pd.DataFrame(index=frame.index)
    for n in (1, 3, 5, 10, 15, 30, 60):
        result[f"return_{n}"] = close.pct_change(n, fill_method=None)
    for n in (5, 10, 20, 50, 100, 200):
        result[f"ema_distance_{n}"] = close / close.ewm(span=n, adjust=False, min_periods=n).mean() - 1
    for n in (5, 15, 30, 60):
        result[f"volatility_{n}"] = returns.rolling(n).std()
    delta = close.diff()
    for n in (7, 14, 21):
        gains = delta.clip(lower=0).rolling(n).mean()
        losses = -delta.clip(upper=0).rolling(n).mean()
        result[f"rsi_{n}"] = gains / (gains + losses).replace(0, np.nan)
    high, low, op = frame["high"], frame["low"], frame["open"]
    spread = (high - low).replace(0, np.nan)
    result["range_pct"] = (high - low) / close
    result["body_pct"] = (close - op) / close
    result["upper_wick_pct"] = (high - pd.concat([op, close], axis=1).max(axis=1)) / close
    result["lower_wick_pct"] = (pd.concat([op, close], axis=1).min(axis=1) - low) / close
    result["close_position"] = ((close - low) / spread).fillna(0.5)
    for field in ("volume", "quote_volume", "trades"):
        series = frame[field]
        result[f"{field}_ratio"] = series / series.rolling(20).mean().replace(0, np.nan)
    result["taker_buy_ratio"] = frame.taker_buy_volume / frame.volume.replace(0, np.nan)
    result["taker_buy_quote_ratio"] = frame.taker_buy_quote_volume / frame.quote_volume.replace(0, np.nan)
    for n in (20, 60):
        result[f"distance_high_{n}"] = close / high.rolling(n).max() - 1
        result[f"distance_low_{n}"] = close / low.rolling(n).min() - 1
    time = pd.to_datetime(frame.open_time, unit="ms", utc=True)
    hour, day = time.dt.hour, time.dt.day_of_week
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["day_sin"] = np.sin(2 * np.pi * day / 7)
    result["day_cos"] = np.cos(2 * np.pi * day / 7)
    result["weekend"] = (day >= 5).astype(float)
    return result.replace([np.inf, -np.inf], np.nan)


def make_labels(frame: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    close = frame.close
    minute_return = close.pct_change(fill_method=None)
    return pd.DataFrame({
        "future_return": close.shift(-horizon) / close - 1,
        "future_volatility": np.sqrt(minute_return.pow(2).rolling(horizon).sum().shift(-horizon)),
    }, index=frame.index)


def time_partitions(length: int, horizon: int = HORIZON):
    """Purge samples whose labels reach into the next partition."""
    train_end, validation_end = int(length * .70), int(length * .85)
    if train_end <= horizon or validation_end - train_end <= horizon or length - validation_end < 1:
        raise ValueError("数据量不足以进行时间切分")
    return slice(0, train_end - horizon), slice(train_end, validation_end - horizon), slice(validation_end, length)


def classify(values, threshold):
    return np.where(values < -threshold, 0, np.where(values > threshold, 2, 1)).astype(int)
