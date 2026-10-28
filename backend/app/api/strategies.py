from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import StrategyConfig
from ..schemas import StrategyUpdate
from ..state import all_state
from ..strategy.defaults import ALLOWED_CODES, DEFAULT_STRATEGIES, seed_strategies
from ..strategy.rules import resolve_params
from ..trading.service import TradingService

router = APIRouter(prefix="/strategies", tags=["strategies"])


def _strategy_dict(row: StrategyConfig) -> dict:
    try:
        params = json.loads(row.params or "{}")
    except ValueError:
        params = {}
    return {
        "id": row.id,
        "code": row.code,
        "name": row.name,
        "description": row.description,
        "enabled": bool(row.enabled),
        "params": params,
        "updated_at": row.updated_at.isoformat() if row.updated_at else "",
    }


@router.get("")
def list_strategies(db: Session = Depends(get_db)):
    seed_strategies(db)
    # 老库可能残留已下线策略的配置行；它们已无规则实现、无法解析参数，
    # 这里只暴露当前仓库真正可执行的策略。
    rows = (
        db.query(StrategyConfig)
        .filter(StrategyConfig.code.in_(ALLOWED_CODES))
        .order_by(StrategyConfig.id.asc())
        .all()
    )
    return {
        "strategies": [{**_strategy_dict(row), 'effective_params': resolve_params(row.code, all_state(db), json.loads(row.params or '{}'))} for row in rows],
        "defaults": DEFAULT_STRATEGIES,
    }


@router.put("/{code}")
def update_strategy(code: str, payload: StrategyUpdate, db: Session = Depends(get_db)):
    row = db.query(StrategyConfig).filter(StrategyConfig.code == code).first()
    if row is None:
        raise HTTPException(404, "策略不存在")
    if payload.params is not None:
        try:
            resolve_params(code, all_state(db), payload.params)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    if payload.enabled is not None:
        row.enabled = 1 if payload.enabled else 0
    if payload.params is not None:
        row.params = json.dumps(payload.params, ensure_ascii=False)
    db.commit()
    db.refresh(row)
    return _strategy_dict(row)


@router.post("/run-once")
def run_once(db: Session = Depends(get_db)):
    result = TradingService(db).run_cycle()
    return result
