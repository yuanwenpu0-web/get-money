from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from ..db import get_db
from ..market.data_service import MarketDataService
from ..models import EquitySnapshot, JobLog, Order, Signal
from ..pagination import page_metadata
from ..schemas import OrderPatch
from ..serializers import order_dict, signal_dict
from ..state import all_state
from ..trading.dispatch import modify_order
from ..trading.portfolio import portfolio_snapshot
from ..trading.scheduler import scheduler
from ..trading.service import TradingService
from ..utils import now_ms

router = APIRouter(prefix="/trading", tags=["trading"])


@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db), include_market: bool = True):
    snapshot = portfolio_snapshot(db, update_marks=True)
    state = all_state(db)
    open_orders = db.query(Order).filter(Order.status == "OPEN").order_by(Order.created_at.desc()).all()
    recent_orders = db.query(Order).order_by(Order.id.desc()).limit(10).all()
    recent_signals = db.query(Signal).order_by(Signal.id.desc()).limit(8).all()
    latest_logs = db.query(JobLog).order_by(JobLog.id.desc()).limit(8).all()
    stats = MarketDataService(db).stats() if include_market else []
    return {
        "portfolio": snapshot,
        "positions": [order_dict(row) for row in open_orders],
        "recent_orders": [order_dict(row) for row in recent_orders],
        "recent_signals": [signal_dict(row) for row in recent_signals],
        "logs": [
            {
                "id": row.id,
                "job": row.job,
                "level": row.level,
                "message": row.message,
                "created_at": row.created_at.isoformat() if row.created_at else "",
            }
            for row in latest_logs
        ],
        "scheduler": scheduler.status(),
        "market": stats,
        "settings": {
            "symbols": state.get("symbols"),
            "interval": state.get("interval"),
            "schedule_enabled": state.get("schedule_enabled"),
            "schedule_seconds": state.get("schedule_seconds"),
            "execution_mode": state.get("execution_mode"),
            "testnet": state.get("testnet"),
        },
    }


@router.get("/orders")
def list_orders(
    status: str | None = None,
    symbol: str | None = None,
    limit: Annotated[int | None, Query(ge=1, le=500)] = None,
    db: Session = Depends(get_db),
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int | None, Query(ge=1, le=100)] = None,
):
    # Explicit page_size wins; legacy callers may still request their old limit.
    page_size = page_size if page_size is not None else (limit if limit is not None else 20)
    query = db.query(Order)
    if status:
        query = query.filter(Order.status == status.upper())
    if symbol:
        query = query.filter(Order.symbol == symbol.upper())
    mark = func.coalesce(func.nullif(Order.mark_price, 0), Order.entry_price)
    pnl = (mark - Order.entry_price) * Order.qty * case((Order.side == "LONG", 1), else_=-1)
    total, open_count, closed_count, realized, unrealized = query.with_entities(
        func.count(Order.id),
        func.coalesce(func.sum(case((Order.status == "OPEN", 1), else_=0)), 0),
        func.coalesce(func.sum(case((Order.status == "CLOSED", 1), else_=0)), 0),
        func.coalesce(func.sum(case((Order.status == "CLOSED", Order.realized_pnl), else_=0.0)), 0.0),
        func.coalesce(func.sum(case((Order.status == "OPEN", pnl), else_=0.0)), 0.0),
    ).one()
    rows = (query.order_by(Order.created_at.desc(), Order.id.desc())
            .offset((page - 1) * page_size).limit(page_size).all())
    return {
        "orders": [order_dict(row) for row in rows],
        **page_metadata(total, page, page_size),
        "summary": {"open": open_count, "closed": closed_count,
                    "realized_pnl": round(realized, 4), "unrealized_pnl": round(unrealized, 4)},
    }


@router.get("/orders/{order_id}")
def get_order(order_id: int, db: Session = Depends(get_db)):
    row = db.get(Order, order_id)
    if row is None:
        raise HTTPException(404, "订单不存在")
    return order_dict(row)


@router.post("/orders/{order_id}/close")
def close_order(order_id: int, db: Session = Depends(get_db)):
    order = TradingService(db).manual_close(order_id)
    if order is None:
        raise HTTPException(400, "订单不存在或已平仓")
    return order_dict(order)


@router.patch("/orders/{order_id}")
def patch_order(order_id: int, payload: OrderPatch, db: Session = Depends(get_db)):
    order = db.get(Order, order_id)
    if order is None:
        raise HTTPException(404, "订单不存在")
    if order.status != "OPEN":
        raise HTTPException(400, "仅未平仓订单可调整")
    mark = order.mark_price or order.entry_price
    if payload.sl is not None:
        if order.side == "LONG" and payload.sl >= mark:
            raise HTTPException(400, "多单止损价必须低于当前价")
        if order.side == "SHORT" and payload.sl <= mark:
            raise HTTPException(400, "空单止损价必须高于当前价")
    if payload.tp is not None:
        if order.side == "LONG" and payload.tp <= mark:
            raise HTTPException(400, "多单止盈价必须高于当前价")
        if order.side == "SHORT" and payload.tp >= mark:
            raise HTTPException(400, "空单止盈价必须低于当前价")
    modify_order(db, order, sl=payload.sl, tp=payload.tp)
    db.commit()
    db.refresh(order)
    db.add(JobLog(job="order", level="info", message=f"订单 #{order.id} 调整 SL={order.sl} TP={order.tp}"))
    db.commit()
    return order_dict(order)


@router.post("/orders/{order_id}/reverse")
def reverse_order(order_id: int, db: Session = Depends(get_db)):
    order = TradingService(db).reverse_order(order_id)
    if order is None:
        raise HTTPException(400, "订单不存在或已平仓")
    return order_dict(order)


@router.get("/equity")
def equity_history(limit: int = Query(1000, ge=10, le=5000), db: Session = Depends(get_db)):
    rows = db.query(EquitySnapshot).order_by(EquitySnapshot.id.desc()).limit(limit).all()
    rows = list(reversed(rows))
    points = [
        {
            "time": row.ts,
            "cash": row.cash,
            "equity": row.equity,
            "unrealized_pnl": row.unrealized_pnl,
            "open_positions": row.open_positions,
        }
        for row in rows
    ]
    return {"points": points, "count": len(points), "now": now_ms()}


@router.get("/logs")
def logs(
    limit: Annotated[int | None, Query(ge=1, le=500)] = None,
    db: Session = Depends(get_db),
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int | None, Query(ge=1, le=100)] = None,
):
    page_size = page_size if page_size is not None else (limit if limit is not None else 20)
    query = db.query(JobLog)
    total = query.count()
    rows = (query.order_by(JobLog.created_at.desc(), JobLog.id.desc())
            .offset((page - 1) * page_size).limit(page_size).all())
    return {
        **page_metadata(total, page, page_size),
        "logs": [
            {
                "id": row.id,
                "job": row.job,
                "level": row.level,
                "message": row.message,
                "created_at": row.created_at.isoformat() if row.created_at else "",
            }
            for row in rows
        ]
    }


@router.post("/run-once")
def run_cycle(db: Session = Depends(get_db)):
    return TradingService(db).run_cycle()
