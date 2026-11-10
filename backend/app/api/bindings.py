"""策略 × 币种池 绑定 API。

一条绑定 = `(策略, 池, 周期)` + 一套参数覆盖 + 一套出场口径。同一策略可以在多个池上
跑不同周期、参数与出场方法，例如「主流 15m，支撑位止损」与「山寨 5m，ATR 止损」。

参数三层合并：全局设置 < 策略默认 < 本条绑定（`rules.resolve_params`）。
出场口径（`exit_method` / `support_indicator` / `exit_params`）以保留键 `exit` 注入
有效参数，回测、模拟盘、巡检、订单管理读到同一份配置。
保存时会用当前全局设置做一次完整校验，非法参数/未知道具直接 400，不会等到巡检才炸。

绑定创建后仍可整体编辑：策略、币种池、周期、优先级、启用状态、参数、出场方法都能改，
不用「删掉重建」（那样会丢参数、换 id）。改身份字段时按 (策略, 池, 周期) 重新查重，
历史订单/信号里记录的是旧池代码，参数解析会自然回落到策略配置，不会被误指向新绑定。
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import StrategyBinding
from ..pagination import DEFAULT_PAGE_SIZE, PageQuery, PageSizeQuery, offset_of, page_metadata
from ..schemas import BindingPayload, BindingUpdate
from ..state import all_state
from ..strategy.bindings import binding_params, describe_binding, exit_config_of, validate_binding, validate_execution
from ..strategy.exits import EXIT_KEY, EXIT_METHODS, normalize as normalize_exit
from ..strategy.support_levels import METHODS

router = APIRouter(prefix="/bindings", tags=["bindings"])


def _get_binding(db: Session, binding_id: int) -> StrategyBinding:
    row = db.get(StrategyBinding, binding_id)
    if row is None:
        raise HTTPException(404, "绑定不存在")
    return row


def _exit_payload(method: str, support: str, params: dict) -> dict:
    return {"method": method, "support": support, "params": params}


@router.get("/exit-options")
def exit_options():
    """出场方法 + 支撑线指标的可选清单，供前端下拉与校验提示。"""
    return {
        "methods": [
            {
                "code": method.code,
                "label": method.label,
                "description": method.description,
                "defaults": method.defaults,
                "ranges": {key: list(value) for key, value in method.ranges.items()},
                "needs_support": method.needs_support,
                "verified": method.verified,
            }
            for method in EXIT_METHODS
        ],
        "support_sources": [
            {"code": spec.key, "label": spec.label, "indicator": spec.indicator}
            for spec in METHODS
        ],
    }


@router.get("")
def list_bindings(
    db: Session = Depends(get_db),
    with_members: bool = False,
    page: PageQuery = 1,
    page_size: PageSizeQuery = DEFAULT_PAGE_SIZE,
):
    state = all_state(db)
    query = db.query(StrategyBinding)
    total = query.count()
    enabled_total = query.filter(StrategyBinding.enabled == 1).count()
    rows = (
        query.order_by(StrategyBinding.priority.asc(), StrategyBinding.id.asc())
        .offset(offset_of(page, page_size)).limit(page_size).all()
    )
    return {
        "bindings": [describe_binding(db, row, state, with_members=with_members) for row in rows],
        # 启用条数是全量口径：分页后前端只看到一页，不能自己数
        "enabled_total": int(enabled_total),
        **page_metadata(total, page, page_size),
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
    exit_payload = _exit_payload(payload.exit_method, payload.support_indicator, payload.exit_params)
    try:
        resolved = validate_binding(
            db, payload.strategy, payload.pool, payload.interval, payload.params, exit_payload
        )
        if payload.enabled:
            validate_execution(db, payload.strategy, payload.interval, resolved)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    normalized = resolved[EXIT_KEY]
    row = StrategyBinding(
        strategy_code=payload.strategy,
        pool_code=payload.pool,
        interval=payload.interval,
        enabled=1 if payload.enabled else 0,
        priority=payload.priority,
        params=json.dumps(payload.params, ensure_ascii=False),
        exit_method=normalized["method"],
        support_indicator=normalized["support"],
        exit_params=json.dumps(normalized["params"], ensure_ascii=False),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return describe_binding(db, row)


@router.put("/{binding_id}")
def update_binding(binding_id: int, payload: BindingUpdate, db: Session = Depends(get_db)):
    row = _get_binding(db, binding_id)
    strategy_code = payload.strategy if payload.strategy is not None else row.strategy_code
    pool_code = payload.pool if payload.pool is not None else row.pool_code
    interval = payload.interval or row.interval
    params = payload.params if payload.params is not None else json.loads(row.params or "{}")

    # 身份字段可变后，唯一性要按「改完后的三元组」重新查（排除自己）。
    duplicate = (
        db.query(StrategyBinding)
        .filter(
            StrategyBinding.strategy_code == strategy_code,
            StrategyBinding.pool_code == pool_code,
            StrategyBinding.interval == interval,
            StrategyBinding.id != binding_id,
        )
        .first()
    )
    if duplicate is not None:
        raise HTTPException(400, "该「策略 × 池 × 周期」的绑定已存在")

    # 出场配置没传时沿用库里的三列（不是「重置为默认」），避免部分更新把它冲掉
    exit_payload = exit_config_of(row) or _exit_payload("strategy", "", {})
    if payload.exit_method is not None:
        exit_payload["method"] = payload.exit_method
        # 换方法后旧方法的参数不再适用（如 support_r 的 buffer_atr 对 atr_r 无意义），
        # 除非调用方同时给出新的 exit_params，否则回到新方法默认值
        if payload.exit_params is None:
            exit_payload["params"] = {}
    if payload.support_indicator is not None:
        exit_payload["support"] = payload.support_indicator
    if payload.exit_params is not None:
        exit_payload["params"] = payload.exit_params

    try:
        # 全量校验：换策略后参数要按新策略的 spec 重新校验；周期可与参数联动
        resolved = validate_binding(db, strategy_code, pool_code, interval, params, exit_payload)
        if payload.enabled if payload.enabled is not None else bool(row.enabled):
            validate_execution(db, strategy_code, interval, resolved)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    normalized = resolved[EXIT_KEY]

    row.strategy_code = strategy_code
    row.pool_code = pool_code
    row.interval = interval
    row.exit_method = normalized["method"]
    row.support_indicator = normalized["support"]
    row.exit_params = json.dumps(normalized["params"], ensure_ascii=False)
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
        "exit": params.get(EXIT_KEY, {}),
        "params": params,
    }


@router.post("/validate-exit")
def validate_exit(payload: dict):
    """干跑一次出场配置校验（前端保存前自检用，不落库）。"""
    try:
        return {"exit": normalize_exit(payload)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
