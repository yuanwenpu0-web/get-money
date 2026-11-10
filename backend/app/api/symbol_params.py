"""币种级参数覆盖的读写接口。

覆盖表是「每币种最优参数」的落地形态：`POST /optimizations/{id}/apply` 自动写入，
也可以在这里手工查/改/删。删除即回到绑定参数，不需要动绑定本身。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..pagination import DEFAULT_PAGE_SIZE, PageQuery, PageSizeQuery
from ..schemas import SymbolParamDelete, SymbolParamPayload
from ..strategy.registry import has_spec
from ..strategy.symbol_params import delete_all_overrides, delete_overrides, describe, set_overrides

router = APIRouter(prefix="/symbol-params", tags=["symbol-params"])


@router.get("")
def list_symbol_params(
    strategy: str = "",
    interval: str = "",
    db: Session = Depends(get_db),
    page: PageQuery = 1,
    page_size: PageSizeQuery = DEFAULT_PAGE_SIZE,
):
    if strategy and not has_spec(strategy):
        raise HTTPException(400, f"未知策略: {strategy}")
    return describe(db, strategy or None, interval or None, page=page, page_size=page_size)


@router.put("")
def upsert_symbol_params(payload: SymbolParamPayload, db: Session = Depends(get_db)):
    try:
        written = set_overrides(
            db,
            payload.strategy,
            payload.interval,
            payload.params,
            source="manual",
            note=payload.note,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"written": written}


@router.delete("/all")
def delete_all_symbol_params(
    strategy: str = "", interval: str = "", db: Session = Depends(get_db)
):
    """清空币种级参数覆盖（可按策略/周期收窄），一次请求删完，不受分页影响。"""
    if strategy and not has_spec(strategy):
        raise HTTPException(400, f"未知策略: {strategy}")
    return {"removed": delete_all_overrides(db, strategy or None, interval or None)}


@router.delete("")
def delete_symbol_params(payload: SymbolParamDelete, db: Session = Depends(get_db)):
    if not has_spec(payload.strategy):
        raise HTTPException(400, f"未知策略: {payload.strategy}")
    removed = delete_overrides(db, payload.strategy, payload.interval, payload.symbols)
    return {"removed": removed}
