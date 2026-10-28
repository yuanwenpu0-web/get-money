from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import pine
from .config import BASE_DIR
from .db import init_db


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Keep the historical market database available, but never restart the old
    # scheduler or model trainers: they use a different strategy engine.
    init_db()
    try:
        yield
    finally:
        pine.stop_all_paper_runners()


app = FastAPI(title="Pine 本地策略工作台", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(pine.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"ok": True, "service": "pine-strategy-lab", "mode": "research"}


DIST_DIR = BASE_DIR / "frontend" / "dist"
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="frontend")
