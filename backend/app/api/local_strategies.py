"""Configuration for TradingView ports used as entry-only filters."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from ..backtest.tradingview_native import tradingview_signals
from ..db import get_db
from ..models import StrategyConfig
from ..state import set_state
from ..strategy.defaults import seed_strategies
from ..strategy.local_filters import CATALOG, config_for, list_filters

router = APIRouter(prefix="/local-strategies", tags=["local-strategies"])


class LocalStrategyUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    enabled: bool | None = None
    bound_strategies: list[str] | None = None
    params: dict | None = None


@router.get("")
def local_strategies(db: Session = Depends(get_db)):
    return {"strategies": list_filters(db)}


@router.put("/{code}")
def update_local_strategy(code: str, payload: LocalStrategyUpdate, db: Session = Depends(get_db)):
    if code not in CATALOG:
        raise HTTPException(404, "本地策略不存在")
    config = {"enabled": False, "bound_strategies": [], "params": {}, **config_for(db, code)}
    update = payload.model_dump(exclude_none=True)
    if "bound_strategies" in update:
        seed_strategies(db)
        allowed = {row.code for row in db.query(StrategyConfig.code).all()}
        if set(update["bound_strategies"]) - allowed:
            raise HTTPException(400, "绑定包含未知系统策略")
        update["bound_strategies"] = list(dict.fromkeys(update["bound_strategies"]))
    if "params" in update:
        params = update["params"]
        try:
            tradingview_signals([], code, params)
            if any(isinstance(value, int) and not isinstance(value, bool) and value > 1000 for value in params.values()):
                raise ValueError("周期不能超过 1000 根")
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc
    config.update(update)
    set_state(db, f"local_filter:{code}", config)
    return next(item for item in list_filters(db) if item["code"] == code)
