from __future__ import annotations

import pickle
import threading

import pandas as pd
from sqlalchemy.orm import Session

from ..models import ModelRecord

_cache: dict[int, tuple[float, dict]] = {}
_lock = threading.Lock()


def latest_model(db: Session, symbol: str, interval: str, algo: str | None = None) -> ModelRecord | None:
    query = db.query(ModelRecord).filter(
        ModelRecord.symbol == symbol,
        ModelRecord.interval == interval,
        ModelRecord.status == "ready",
    )
    if algo:
        query = query.filter(ModelRecord.algo == algo)
    return query.order_by(ModelRecord.created_at.desc()).first()


def _load(record: ModelRecord) -> dict:
    with _lock:
        cached = _cache.get(record.id)
    if cached and cached[0] == record.id:
        return cached[1]
    with open(record.path, "rb") as handle:
        payload = pickle.load(handle)
    with _lock:
        _cache[record.id] = (record.id, payload)
    return payload


def predict_frame(record: ModelRecord, frame: pd.DataFrame) -> pd.Series:
    payload = _load(record)
    model = payload["model"]
    features = payload.get("features", [])
    matrix = frame[features].fillna(0.0)
    probability = model.predict_proba(matrix)[:, 1]
    return pd.Series(probability, index=frame.index)


def predict_row(record: ModelRecord, row: pd.Series) -> float:
    payload = _load(record)
    model = payload["model"]
    features = payload.get("features", [])
    values = pd.DataFrame([[0.0 if pd.isna(row.get(name)) else float(row.get(name)) for name in features]], columns=features)
    return float(model.predict_proba(values)[0][1])
