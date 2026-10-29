"""币种级参数覆盖的读写接口。

覆盖表是「每币种最优参数」的落地形态：`POST /optimizations/{id}/apply` 自动写入，
也可以在这里手工查/改/删。删除即回到绑定参数，不需要动绑定本身。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..schemas import SymbolParamDelete, SymbolParamPayload
from ..strategy.registry import has_spec
from ..strategy.symbol_params import delete_overrides, describe, set_overrides

router = APIRouter(prefix="/symbol-params", tags=["symbol-params"])


@router.get("")
def list_symbol_params(
    strategy: str = "", interval: str = "", db: Session = Depends(get_db)
):
    if strategy and not has_spec(strategy):
        raise HTTPException(400, f"未知策略: {strategy}")
    return describe(db, strategy or None, interval or None)


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


@router.delete("")
def delete_symbol_params(payload: SymbolParamDelete, db: Session = Depends(get_db)):
    if not has_spec(payload.strategy):
        raise HTTPException(400, f"未知策略: {payload.strategy}")
    removed = delete_overrides(db, payload.strategy, payload.interval, payload.symbols)
    return {"removed": removed}
