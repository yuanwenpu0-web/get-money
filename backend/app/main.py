from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import backtests, bindings, indicators, market, pools, settings, strategies, trading
from .config import BASE_DIR
from .db import SessionLocal, init_db
from .state import all_state, ensure_default_state
from .strategy.defaults import seed_defaults
from .trading.scheduler import scheduler, sync_scheduler_with_state


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    db = SessionLocal()
    try:
        ensure_default_state(db)
        seed_defaults(db)
        sync_scheduler_with_state(db)
    finally:
        db.close()
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
app.include_router(indicators.router, prefix="/api")
app.include_router(strategies.router, prefix="/api")
app.include_router(pools.router, prefix="/api")
app.include_router(bindings.router, prefix="/api")
app.include_router(backtests.router, prefix="/api")
app.include_router(trading.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"ok": True, "service": "quant-trading", "mode": "paper"}


DIST_DIR = BASE_DIR / "frontend" / "dist"
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="frontend")
