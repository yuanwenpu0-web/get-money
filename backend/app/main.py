from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import backtests, market, models_api, settings, strategies, trading
from .config import BASE_DIR
from .db import SessionLocal, init_db
from .market.nofx_source import bootstrap_defaults
from .state import all_state, ensure_default_state
from .strategy.defaults import seed_strategies
from .trading.scheduler import scheduler, sync_scheduler_with_state


def _auto_train_missing() -> None:
    """首次启动且没有模型时后台训练，避免阻塞服务启动。"""
    from .ml.trainer import train_model
    from .models import JobLog, ModelRecord

    db = SessionLocal()
    try:
        state = all_state(db)
        for symbol in state.get("symbols", []):
            interval = state.get("interval", "1m")
            exists = (
                db.query(ModelRecord)
                .filter(ModelRecord.symbol == symbol, ModelRecord.interval == interval, ModelRecord.status == "ready")
                .first()
            )
            if exists:
                continue
            try:
                record = train_model(
                    db,
                    symbol,
                    interval,
                    algo="lightgbm",
                    horizon=int(state.get("train_horizon", 5)),
                    days=int(state.get("auto_train_days", 30)),
                )
                db.add(JobLog(job="train", level="info", message=f"自动训练 {symbol} {interval} 完成：模型 #{record.id}"))
            except Exception as exc:
                db.add(JobLog(job="train", level="error", message=f"自动训练 {symbol} 失败：{exc}"))
            db.commit()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    db = SessionLocal()
    try:
        ensure_default_state(db)
        seed_strategies(db)
        state = all_state(db)
        if state.get("data_source", "nofx") == "nofx":
            bootstrap_defaults(db, state.get("symbols") or [], state.get("interval", "1m"))
        sync_scheduler_with_state(db)
    finally:
        db.close()
    threading.Thread(target=_auto_train_missing, daemon=True).start()
    yield
    scheduler.stop()


app = FastAPI(title="虚拟币量化交易系统", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(settings.router, prefix="/api")
app.include_router(market.router, prefix="/api")
app.include_router(strategies.router, prefix="/api")
app.include_router(models_api.router, prefix="/api")
app.include_router(backtests.router, prefix="/api")
app.include_router(trading.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"ok": True, "service": "quant-trading", "mode": "paper"}


DIST_DIR = BASE_DIR / "frontend" / "dist"
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="frontend")
