from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..features.indicators import build_features
from ..market import nofx_source
from ..market.data_service import MarketDataService
from ..models import JobLog
from ..schemas import PreviewRequest, SyncRequest
from ..strategy.engine import StrategyEngine
from ..state import all_state
from ..utils import interval_ms, now_ms

router = APIRouter(prefix="/market", tags=["market"])


def _clean(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return round(number, 6)


@router.get("/stats")
def market_stats(db: Session = Depends(get_db)):
    return {"stats": MarketDataService(db).stats()}


@router.get("/sources")
def market_sources(db: Session = Depends(get_db)):
    return {
        "local": {"stats": MarketDataService(db).stats()},
        "nofx": nofx_source.source_status(),
        "binance": {"available": True, "note": "公共行情接口，网络异常时自动多域名重试"},
    }


@router.post("/sync")
def sync_market(payload: SyncRequest, db: Session = Depends(get_db)):
    source = payload.source.lower()
    if source not in ("nofx", "binance"):
        raise HTTPException(400, "source 仅支持 nofx / binance")
    if payload.days < 0 or payload.days > 365:
        raise HTTPException(400, "days 允许范围 0-365（0 表示全部）")

    service = MarketDataService(db)
    try:
        if source == "nofx":
            result = nofx_source.import_klines(db, payload.symbol.upper(), payload.interval, payload.days)
        else:
            if payload.days < 1:
                raise HTTPException(400, "币安同步至少 1 天")
            result = service.sync_history(payload.symbol.upper(), payload.interval, payload.days)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"拉取失败: {exc}") from exc

    db.add(
        JobLog(
            job="sync",
            level="info",
            message=f"{source} {result['symbol']} {result['interval']} 导入 {result.get('imported', result.get('inserted', 0))} 根",
        )
    )
    db.commit()
    return result


@router.get("/klines")
def klines(
    symbol: str = Query(...),
    interval: str = Query("1m"),
    limit: int = Query(500, ge=10, le=1500),
    db: Session = Depends(get_db),
):
    service = MarketDataService(db)
    if service.latest_open_time(symbol, interval) == 0:
        try:
            service.sync_recent(symbol, interval, tail_limit=1000)
        except Exception as exc:
            raise HTTPException(502, f"无本地数据且拉取失败: {exc}") from exc
    frame = service.load_frame(symbol, interval, limit=limit)
    if frame.empty:
        return {"symbol": symbol, "interval": interval, "bars": []}
    bars = []
    for index, row in frame.iterrows():
        bars.append(
            {
                "time": int(index.timestamp() * 1000),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
        )
    return {"symbol": symbol, "interval": interval, "bars": bars}


@router.post("/preview")
def preview_signal(payload: PreviewRequest, db: Session = Depends(get_db)):
    engine = StrategyEngine(db)
    try:
        signal = engine.evaluate(payload.symbol.upper(), payload.interval, payload.strategy, save=False)
    except Exception as exc:
        raise HTTPException(400, f"信号计算失败: {exc}") from exc
    service = MarketDataService(db)
    frame = service.load_frame(payload.symbol.upper(), payload.interval, limit=300)
    if frame.empty:
        raise HTTPException(400, "本地无K线数据，请先同步")
    last_open = int(frame.index[-1].timestamp() * 1000)
    if last_open + interval_ms(payload.interval) > now_ms() + 1000:
        frame = frame.iloc[:-1]
    features = build_features(frame, with_label=False).dropna(subset=["atr_pct"])
    latest = features.iloc[-1] if not features.empty else frame.iloc[-1]
    indicators = {
        "time": str(features.index[-1]) if not features.empty else str(frame.index[-1]),
        "close": _clean(latest.get("close")),
        "rsi_14": _clean(latest.get("rsi_14")),
        "rsi_7": _clean(latest.get("rsi_7")),
        "macd_hist": _clean(latest.get("macd_hist_norm")),
        "atr_pct": _clean(latest.get("atr_pct")),
        "supertrend_dir": _clean(latest.get("supertrend_dir")),
        "supertrend_dist": _clean(latest.get("supertrend_dist")),
        "ema_ratio_9_21": _clean(latest.get("ema_ratio_9_21")),
        "adx_14": _clean(latest.get("adx_14")),
        "bb_pos": _clean(latest.get("bb_pos")),
        "volume_z": _clean(latest.get("volume_z")),
    }
    state = all_state(db)
    return {
        "signal": signal.to_dict() if signal else None,
        "indicators": indicators,
        "risk": {
            "equity": state.get("total_capital"),
            "fee_rate": state.get("fee_rate"),
            "slippage": state.get("slippage"),
            "min_score": state.get("min_score"),
            "heavy_score": state.get("heavy_score"),
        },
    }
