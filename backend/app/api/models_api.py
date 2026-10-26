from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..ml import auto_trainer
from ..ml.trainer import train_model
from ..models import JobLog, ModelRecord
from ..schemas import TrainRequest
from ..serializers import model_dict

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models(db: Session = Depends(get_db)):
    rows = db.query(ModelRecord).order_by(ModelRecord.created_at.desc()).limit(100).all()
    return {"models": [model_dict(row) for row in rows]}


@router.get("/history")
def training_history(symbol: str = Query(...), interval: str = Query("1h"),
                     limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    """某币种某周期的训练历史（按时间升序）。业务在 auto_trainer，路由只做装配。"""
    history = auto_trainer.list_training_history(db, symbol.upper(), interval, limit=limit)
    return {"symbol": symbol.upper(), "interval": interval, "history": history}


@router.post("/train")
def train(payload: TrainRequest, db: Session = Depends(get_db)):
    if payload.algo not in ("lightgbm", "xgboost"):
        raise HTTPException(400, "algo 仅支持 lightgbm / xgboost")
    try:
        record = train_model(
            db,
            payload.symbol.upper(),
            payload.interval,
            algo=payload.algo,
            horizon=payload.horizon,
            days=payload.days,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        db.add(JobLog(job="train", level="error", message=f"{payload.symbol} {payload.algo}: {exc}"))
        db.commit()
        raise HTTPException(500, f"训练失败: {exc}") from exc
    db.add(JobLog(job="train", level="info", message=f"{record.symbol} {record.interval} {record.algo} 训练完成，样本 {record.rows}"))
    db.commit()
    return model_dict(record)


@router.delete("/{model_id}")
def delete_model(model_id: int, db: Session = Depends(get_db)):
    record = db.get(ModelRecord, model_id)
    if record is None:
        raise HTTPException(404, "模型不存在")
    try:
        Path(record.path).unlink(missing_ok=True)
    except OSError:
        pass
    db.delete(record)
    db.commit()
    return {"ok": True}
