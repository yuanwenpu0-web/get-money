"""Pine conversion, closed-bar verification, and local simulation APIs."""
from __future__ import annotations

import json
import re
import threading
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db import get_db
from ..config import DATA_DIR
from ..market.binance import BinanceError, parse_klines
from ..market.data_service import MarketDataService
from ..market.futures import BinanceFuturesClient
from ..pine import compare_csv, compile_pine
from ..pine.backtest import run_backtest
from ..pine.paper import append_candles, create_account, read_account
from ..pine.paper_runner import PaperRunner
from ..state import all_state
from ..utils import interval_ms, now_ms

router = APIRouter(prefix="/pine", tags=["pine"])
PAPER_DIR = DATA_DIR / "pine-paper"
_runner_lock = threading.RLock()
_runners: dict[str, PaperRunner] = {}


class ConvertRequest(BaseModel):
    source: str = Field(min_length=1, max_length=100_000)


class CompareRequest(ConvertRequest):
    candles: list[dict]
    reference: list[dict]
    numeric_columns: list[str] | None = None
    atol: float = 1e-8
    rtol: float = 1e-6


class BacktestRequest(ConvertRequest):
    candles: list[dict] | None = None
    data_source: Literal["local", "binance"] = "local"
    symbol: Literal["BTCUSDT", "ETHUSDT"] = "BTCUSDT"
    interval: str = "1m"
    limit: int = Field(default=3000, ge=1, le=10000)
    initial_equity: float = 10000.0
    quantity: float = 1.0
    fee_rate: float = 0.0004
    slippage: float = 0.0


class PaperCreateRequest(ConvertRequest):
    settings: dict = Field(default_factory=dict)


class PaperAppendRequest(BaseModel):
    candles: list[dict]


class PaperStartRequest(BaseModel):
    symbol: Literal["BTCUSDT", "ETHUSDT"]


def _paper_path(account_id: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", account_id):
        raise HTTPException(status_code=422, detail="账户 ID 只能包含字母、数字、下划线或连字符")
    return PAPER_DIR / f"{account_id}.json"


def _rows(frame: pd.DataFrame) -> list[dict]:
    # Pandas serializes NaN indicator warm-up values as JSON null.
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _market_candles(payload: BacktestRequest, db: Session, step: int) -> pd.DataFrame:
    if payload.candles is not None:
        return pd.DataFrame(payload.candles)
    if payload.data_source == "local":
        frame = MarketDataService(db).load_frame(payload.symbol, payload.interval, limit=payload.limit + 1)
        if frame.empty:
            raise ValueError("本地没有 K 线，请先同步行情或提供 candles")
        frame = frame.reset_index()
        frame["open_time"] = pd.to_datetime(frame.pop("time"), utc=True).astype("datetime64[ns, UTC]").astype("int64") // 1_000_000
        return frame
    client = BinanceFuturesClient(testnet=False, proxy=all_state(db).get("proxy_url") or None)
    server_now = client.server_time()
    cursor = max(0, (server_now // step - payload.limit) * step)
    rows = []
    with client.connection_pool():
        while cursor < server_now:
            page = parse_klines(client.klines(payload.symbol, payload.interval, start_ms=cursor, limit=1000))
            if not page:
                break
            rows.extend(page)
            next_cursor = int(page[-1]["open_time"]) + step
            if next_cursor <= cursor:
                raise ValueError("币安 K 线分页没有前进")
            cursor = next_cursor
    return pd.DataFrame([row for row in rows if row["open_time"] + step <= server_now][-payload.limit:])


@router.post("/convert")
def convert(payload: ConvertRequest):
    try:
        program = compile_pine(payload.source)
        return {"supported": True, "python": program.to_python(),
                "validation": "仅通过语法解析；仍需逐 K 线计算并与 TradingView 数据对比"}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/compare")
def compare(payload: CompareRequest):
    try:
        program = compile_pine(payload.source)
        actual = program.evaluate(pd.DataFrame(payload.candles))
        return compare_csv(pd.DataFrame(payload.reference), actual, payload.numeric_columns, payload.atol, payload.rtol)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/backtest")
def backtest(payload: BacktestRequest, db: Session = Depends(get_db)):
    try:
        program = compile_pine(payload.source)
        step = interval_ms(payload.interval)
        candles = _market_candles(payload, db, step)
        if "open_time" not in candles:
            raise ValueError("K线必须包含 open_time")
        times = pd.to_numeric(candles["open_time"], errors="raise")
        if times.isna().any() or not np.isfinite(times).all() or (times % 1 != 0).any():
            raise ValueError("K线 open_time 必须为有限的整数毫秒时间戳")
        if not times.is_monotonic_increasing or times.duplicated().any():
            raise ValueError("K线时间必须唯一且升序")
        candles = candles.loc[times + step <= now_ms()].copy()
        if payload.candles is None:
            candles = candles.tail(payload.limit)
        candles["open_time"] = times.loc[candles.index].astype("int64")
        candles = candles.reset_index(drop=True)
        signals = program.evaluate(candles)
        result = run_backtest(candles, signals, initial_equity=payload.initial_equity,
                              quantity=payload.quantity, fee_rate=payload.fee_rate,
                              slippage=payload.slippage, interval_ms=step)
        return {"signals": _rows(signals), "result": result}
    except BinanceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/paper/{account_id}")
def paper_create(account_id: str, payload: PaperCreateRequest):
    try:
        return create_account(_paper_path(account_id), payload.source, payload.settings)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail="模拟账户已存在") from exc
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/paper/{account_id}")
def paper_read(account_id: str):
    try:
        return read_account(_paper_path(account_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="模拟账户不存在") from exc


@router.post("/paper/{account_id}/candles")
def paper_append(account_id: str, payload: PaperAppendRequest):
    try:
        with _runner_lock:
            runner = _runners.get(account_id)
            if runner and runner.status()["status"] == "running":
                raise HTTPException(status_code=409, detail="运行期间不能手动追加 K 线")
            return append_candles(_paper_path(account_id), pd.DataFrame(payload.candles))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="模拟账户不存在") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/paper/{account_id}/start")
def paper_start(account_id: str, payload: PaperStartRequest):
    try:
        path = _paper_path(account_id)
        with _runner_lock:
            current = _runners.get(account_id)
            if current and current.status()["status"] == "running":
                raise HTTPException(status_code=409, detail="模拟账户已在运行")
            runner = PaperRunner(path, payload.symbol)
            runner.start()
            _runners[account_id] = runner
            return runner.status()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="模拟账户不存在") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/paper/{account_id}/status")
def paper_status(account_id: str):
    path = _paper_path(account_id)
    try:
        with _runner_lock:
            runner = _runners.get(account_id)
            if runner:
                return runner.status()
            state = read_account(path)
            return {"status": "stopped", "error": None, "symbol": state.get("market_symbol")}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="模拟账户不存在") from exc


@router.post("/paper/{account_id}/stop")
def paper_stop(account_id: str):
    path = _paper_path(account_id)
    try:
        with _runner_lock:
            runner = _runners.get(account_id)
            if not runner:
                state = read_account(path)
                return {"status": "stopped", "error": None, "symbol": state.get("market_symbol")}
            if not runner.stop():
                raise HTTPException(status_code=409, detail="停止请求已发送，等待当前行情请求结束")
            return runner.status()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="模拟账户不存在") from exc


def stop_all_paper_runners():
    """Request shutdown for all workers. Called by the application lifespan."""
    with _runner_lock:
        for runner in _runners.values():
            runner.stop()
