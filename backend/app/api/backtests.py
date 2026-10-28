from __future__ import annotations

import json
import threading
import traceback
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, defer

from ..backtest.runner import run_backtest
from ..db import SessionLocal, get_db
from ..models import BacktestRun
from ..pagination import page_metadata
from ..schemas import BacktestRequest
from ..serializers import backtest_dict

router = APIRouter(prefix="/backtests", tags=["backtests"])


def _execute(run_id: int) -> None:
    db = SessionLocal()
    try:
        run = db.get(BacktestRun, run_id)
        if run is None:
            return
        params = json.loads(run.params or "{}")
        result = run_backtest(
            db,
            run.symbol,
            run.interval,
            run.strategy,
            algo=run.algo,
            days=int(params.get("days", 30)),
            initial_cash=params.get("initial_cash"),
            fee=params.get("fee"),
            slippage=params.get("slippage"),
            threshold=params.get("threshold"),
            optimize=bool(params.get('optimize', False)),
        )
        run.metrics = json.dumps(result["metrics"], ensure_ascii=False)
        run.equity_curve = json.dumps(result["equity_curve"], ensure_ascii=False)
        run.trades = json.dumps(result["trades"], ensure_ascii=False)
        run.status = "done"
    except Exception as exc:
        run = db.get(BacktestRun, run_id)
        if run is not None:
            run.status = "failed"
            run.error = str(exc)[:1000] + "\n" + traceback.format_exc(limit=2)[:1000]
    finally:
        db.commit()
        db.close()


@router.post("")
def create_backtest(payload: BacktestRequest, db: Session = Depends(get_db)):
    if payload.days < 1 or payload.days > 180:
        raise HTTPException(400, "回测天数允许 1-180")
    run = BacktestRun(
        symbol=payload.symbol.upper(),
        interval=payload.interval,
        strategy=payload.strategy,
        algo=payload.algo,
        status="running",
        params=json.dumps(payload.model_dump(), ensure_ascii=False),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    threading.Thread(target=_execute, args=(run.id,), daemon=True).start()
    return {"id": run.id, "status": "running"}


@router.get("")
def list_backtests(
    db: Session = Depends(get_db),
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
):
    query = db.query(BacktestRun)
    total = query.count()
    rows = (query.options(defer(BacktestRun.params), defer(BacktestRun.equity_curve), defer(BacktestRun.trades))
            .order_by(BacktestRun.created_at.desc(), BacktestRun.id.desc())
            .offset((page - 1) * page_size).limit(page_size).all())
    return {"backtests": [backtest_dict(row) for row in rows], **page_metadata(total, page, page_size)}


@router.get("/{run_id}")
def get_backtest(run_id: int, db: Session = Depends(get_db), include_trades: bool = True):
    query = db.query(BacktestRun).filter(BacktestRun.id == run_id)
    if not include_trades:
        query = query.options(defer(BacktestRun.trades))
    run = query.first()
    if run is None:
        raise HTTPException(404, "回测不存在")
    return backtest_dict(run, full=True, include_trades=include_trades)


@router.get("/{run_id}/trades")
def get_backtest_trades(
    run_id: int,
    db: Session = Depends(get_db),
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
):
    row = db.query(BacktestRun.trades).filter(BacktestRun.id == run_id).first()
    if row is None:
        raise HTTPException(404, "回测不存在")
    # Historical runs store one JSON array; parsing it is still necessary until
    # a future data migration stores trades as individual rows.
    try:
        trades = json.loads(row.trades or "[]")
    except (TypeError, ValueError):
        trades = []
    if not isinstance(trades, list):
        trades = []
    start = (page - 1) * page_size
    return {"trades": trades[start:start + page_size], **page_metadata(len(trades), page, page_size)}
