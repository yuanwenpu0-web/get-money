import json
from typing import Any

from sqlalchemy.orm import Session

from .config import DEFAULT_STATE
from .models import AppState


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _loads(raw: str, default: Any) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def ensure_default_state(db: Session) -> None:
    existing = {row.key for row in db.query(AppState.key).all()}
    for key, value in DEFAULT_STATE.items():
        if key not in existing:
            db.add(AppState(key=key, value=_dumps(value)))
    db.commit()


def get_state(db: Session, key: str, default: Any = None) -> Any:
    row = db.get(AppState, key)
    if row is None:
        return DEFAULT_STATE.get(key, default) if default is None else default
    return _loads(row.value, DEFAULT_STATE.get(key, default))


def set_state(db: Session, key: str, value: Any) -> None:
    row = db.get(AppState, key)
    payload = _dumps(value)
    if row is None:
        db.add(AppState(key=key, value=payload))
    else:
        row.value = payload
    db.commit()


def all_state(db: Session) -> dict:
    result = dict(DEFAULT_STATE)
    for row in db.query(AppState).all():
        result[row.key] = _loads(row.value, row.value)
    result["data_source"] = "binance"
    return result
