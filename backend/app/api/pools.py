"""币种池 API：策略运行范围的增删改查 + 成员预览。

池是「策略 × 池」绑定的另一半：同一个策略在不同池上可以跑不同周期、不同参数。
这里只维护池定义；成员由 `strategy/pools.py` 按规则实时解析（成交量走 TTL 缓存）。
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import StrategyBinding, SymbolPool
from ..schemas import PoolPayload, PoolUpdate
from ..strategy.pools import describe_pool, load_pool_spec, pool_symbols, validate_pool

router = APIRouter(prefix="/pools", tags=["pools"])


def _get_pool(db: Session, code: str) -> SymbolPool:
    pool = db.query(SymbolPool).filter(SymbolPool.code == code).first()
    if pool is None:
        raise HTTPException(404, "币种池不存在")
    return pool


@router.get("")
def list_pools(db: Session = Depends(get_db)):
    pools = db.query(SymbolPool).order_by(SymbolPool.id.asc()).all()
    result = []
    for pool in pools:
        try:
            members = pool_symbols(db, pool)
        except Exception:  # 单个池解析失败（如网络问题）不应让整个列表 500
            members = None
        result.append(describe_pool(pool, members))
    return {"pools": result, "total": len(result)}


@router.post("")
def create_pool(payload: PoolPayload, db: Session = Depends(get_db)):
    if db.query(SymbolPool).filter(SymbolPool.code == payload.code).first() is not None:
        raise HTTPException(400, f"币种池已存在: {payload.code}")
    try:
        kind, symbols, rule = validate_pool(payload.kind, payload.symbols, payload.rule)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    pool = SymbolPool(
        code=payload.code,
        name=payload.name,
        description=payload.description,
        kind=kind,
        symbols=json.dumps(symbols, ensure_ascii=False),
        rule=json.dumps(rule, ensure_ascii=False),
        enabled=1 if payload.enabled else 0,
    )
    db.add(pool)
    db.commit()
    db.refresh(pool)
    return describe_pool(pool, pool_symbols(db, pool))


@router.put("/{code}")
def update_pool(code: str, payload: PoolUpdate, db: Session = Depends(get_db)):
    pool = _get_pool(db, code)
    spec = load_pool_spec(pool)
    kind = payload.kind if payload.kind is not None else spec["kind"]
    symbols = payload.symbols if payload.symbols is not None else spec["symbols"]
    if payload.rule is not None:
        rule = payload.rule
    elif payload.kind is not None and payload.kind != spec["kind"]:
        # 只改类型时重置规则：static 池继承 base=major 之类的旧规则会自相矛盾
        rule = {}
    else:
        rule = spec["rule"]
    try:
        kind, symbols, rule = validate_pool(kind, symbols, rule)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    if payload.name is not None:
        pool.name = payload.name
    if payload.description is not None:
        pool.description = payload.description
    if payload.enabled is not None:
        pool.enabled = 1 if payload.enabled else 0
    pool.kind = kind
    pool.symbols = json.dumps(symbols, ensure_ascii=False)
    pool.rule = json.dumps(rule, ensure_ascii=False)
    db.commit()
    db.refresh(pool)
    return describe_pool(pool, pool_symbols(db, pool))


@router.delete("/{code}")
def delete_pool(code: str, db: Session = Depends(get_db)):
    pool = _get_pool(db, code)
    used = db.query(StrategyBinding).filter(StrategyBinding.pool_code == code).count()
    if used:
        raise HTTPException(400, f"该池已被 {used} 条策略绑定引用，请先删除绑定")
    db.delete(pool)
    db.commit()
    return {"ok": True, "code": code}


@router.get("/{code}/members")
def pool_members(code: str, db: Session = Depends(get_db)):
    """成员预览：调参数前先看这个池到底会跑多少个币。"""
    pool = _get_pool(db, code)
    members = pool_symbols(db, pool)
    return {"code": code, "count": len(members), "symbols": members}
