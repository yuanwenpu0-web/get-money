from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import StrategyBinding, StrategyConfig
from ..schemas import StrategyUpdate
from ..state import all_state
from ..strategy.bindings import describe_binding
from ..strategy.defaults import ALLOWED_CODES, DEFAULT_STRATEGIES, seed_defaults
from ..strategy.registry import all_specs, get_spec
from ..strategy.rules import resolve_params
from ..trading.service import TradingService

# 只能由专用撮合表达的策略：连锁信号帧方向恒为 0，开启巡检只会得到一个永不交易的空策略
# （simple_zigzag_double 已完成模拟盘集成：挂单触价信号 + 跟随止损 + 同周期反手）
BROKER_ONLY_CODES = ("opening_candle_breakout",)

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
    seed_defaults(db)
    # 老库可能残留已下线策略的配置行；它们已无规则实现、无法解析参数，
    # 这里只暴露当前仓库真正可执行的策略。
    rows = (
        db.query(StrategyConfig)
        .filter(StrategyConfig.code.in_(ALLOWED_CODES))
        .order_by(StrategyConfig.id.asc())
        .all()
    )
    state = all_state(db)
    return {
        "strategies": [
            {
                **_strategy_dict(row),
                "effective_params": resolve_params(row.code, state, json.loads(row.params or "{}")),
                "default_params": get_spec(row.code).defaults,
                "tags": list(get_spec(row.code).tags),
            }
            for row in rows
        ],
        "defaults": DEFAULT_STRATEGIES,
        # 前端「策略 × 池」矩阵需要的元信息
        "specs": [
            {
                "code": spec.code,
                "name": spec.name,
                "tags": list(spec.tags),
                "default_enabled": spec.default_enabled,
            }
            for spec in all_specs()
        ],
    }


@router.put("/{code}")
def update_strategy(code: str, payload: StrategyUpdate, db: Session = Depends(get_db)):
    if code not in ALLOWED_CODES:
        raise HTTPException(404, "策略不存在")
    if code in BROKER_ONLY_CODES and payload.enabled:
        raise HTTPException(400, "此策略目前仅支持回测，模拟盘和实盘尚不支持挂单/会话强平")
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


@router.get("/{code}/bindings")
def strategy_bindings(code: str, db: Session = Depends(get_db)):
    """某个策略下的全部「策略 × 池」绑定（含有效参数与成员数）。"""
    if code not in ALLOWED_CODES:
        raise HTTPException(404, "策略不存在")
    rows = (
        db.query(StrategyBinding)
        .filter(StrategyBinding.strategy_code == code)
        .order_by(StrategyBinding.priority.asc(), StrategyBinding.id.asc())
        .all()
    )
    return {"bindings": [describe_binding(db, row) for row in rows]}


@router.post("/run-once")
def run_once(db: Session = Depends(get_db)):
    result = TradingService(db).run_cycle()
    return result
