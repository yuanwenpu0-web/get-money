"""策略 × 币种池 绑定 API。

一条绑定 = `(策略, 池, 周期)` + 一套参数覆盖。同一策略可以在多个池上跑不同周期与参数，
例如「主流 15m」「山寨 5m」「山寨 1m」三条绑定互不影响。

参数三层合并：全局设置 < 策略默认 < 本条绑定（`rules.resolve_params`）。
保存时会用当前全局设置做一次完整校验，非法参数直接 400，不会等到巡检才炸。
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import StrategyBinding
from ..schemas import BindingPayload, BindingUpdate
from ..state import all_state
from ..strategy.bindings import binding_params, describe_binding, validate_binding

router = APIRouter(prefix="/bindings", tags=["bindings"])


def _get_binding(db: Session, binding_id: int) -> StrategyBinding:
    row = db.get(StrategyBinding, binding_id)
    if row is None:
        raise HTTPException(404, "绑定不存在")
    return row


@router.get("")
def list_bindings(db: Session = Depends(get_db), with_members: bool = False):
    state = all_state(db)
    rows = (
        db.query(StrategyBinding)
        .order_by(StrategyBinding.priority.asc(), StrategyBinding.id.asc())
        .all()
    )
    return {
        "bindings": [describe_binding(db, row, state, with_members=with_members) for row in rows],
        "total": len(rows),
    }


@router.post("")
def create_binding(payload: BindingPayload, db: Session = Depends(get_db)):
    duplicate = (
        db.query(StrategyBinding)
        .filter(
            StrategyBinding.strategy_code == payload.strategy,
            StrategyBinding.pool_code == payload.pool,
            StrategyBinding.interval == payload.interval,
        )
        .first()
    )
    if duplicate is not None:
        raise HTTPException(400, "该「策略 × 池 × 周期」的绑定已存在")
    try:
        validate_binding(db, payload.strategy, payload.pool, payload.interval, payload.params)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    row = StrategyBinding(
        strategy_code=payload.strategy,
        pool_code=payload.pool,
        interval=payload.interval,
        enabled=1 if payload.enabled else 0,
        priority=payload.priority,
        params=json.dumps(payload.params, ensure_ascii=False),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return describe_binding(db, row)


@router.put("/{binding_id}")
def update_binding(binding_id: int, payload: BindingUpdate, db: Session = Depends(get_db)):
    row = _get_binding(db, binding_id)
    interval = payload.interval or row.interval
    params = payload.params if payload.params is not None else json.loads(row.params or "{}")
    try:
        # 全量校验：周期可与参数联动（如 max_hold_bars 按 bar 计）
        validate_binding(db, row.strategy_code, row.pool_code, interval, params)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    if payload.interval is not None:
        row.interval = payload.interval
    if payload.enabled is not None:
        row.enabled = 1 if payload.enabled else 0
    if payload.priority is not None:
        row.priority = payload.priority
    if payload.params is not None:
        row.params = json.dumps(payload.params, ensure_ascii=False)
    db.commit()
    db.refresh(row)
    return describe_binding(db, row)


@router.delete("/{binding_id}")
def delete_binding(binding_id: int, db: Session = Depends(get_db)):
    row = _get_binding(db, binding_id)
    db.delete(row)
    db.commit()
    return {"ok": True, "id": binding_id}


@router.get("/{binding_id}/params")
def binding_effective_params(binding_id: int, db: Session = Depends(get_db)):
    """绑定当前生效的完整参数（合并三层后），用于前端「查看生效值」与排障。"""
    row = _get_binding(db, binding_id)
    try:
        params = binding_params(db, row, all_state(db))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "id": row.id,
        "strategy": row.strategy_code,
        "pool": row.pool_code,
        "interval": row.interval,
        "params": params,
    }
