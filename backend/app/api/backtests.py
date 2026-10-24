from __future__ import annotations

import json
import threading
import traceback

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..backtest.runner import run_backtest
from ..db import SessionLocal, get_db
from ..models import BacktestRun
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
def list_backtests(db: Session = Depends(get_db)):
    rows = db.query(BacktestRun).order_by(BacktestRun.created_at.desc()).limit(40).all()
    return {"backtests": [backtest_dict(row) for row in rows]}


@router.get("/{run_id}")
def get_backtest(run_id: int, db: Session = Depends(get_db)):
    run = db.get(BacktestRun, run_id)
    if run is None:
        raise HTTPException(404, "回测不存在")
    return backtest_dict(run, full=True)
