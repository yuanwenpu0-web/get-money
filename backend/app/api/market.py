from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..features.funding import funding_snapshot
from ..features.indicators import build_features
from ..market.data_service import MarketDataService
from ..market import metrics as metrics_mod
from ..market.universe import resolve_universe, universe_symbols
from ..models import JobLog
from ..schemas import MetricsSyncRequest, PreviewRequest, SyncRequest
from ..strategy.bindings import resolve_order_params
from ..strategy.engine import StrategyEngine
from ..state import all_state
from ..utils import interval_ms, now_ms

router = APIRouter(prefix="/market", tags=["market"])


@router.get("/symbols")
def market_symbols(db: Session = Depends(get_db)):
    """U 本位永续合约全量列表（前端下拉选择用）；symbol_scope=custom 时返回用户 symbols。"""
    state = all_state(db)
    if state.get("symbol_scope", "all") == "all":
        info = resolve_universe(db)
        return {"symbols": info["symbols"], "source": info["source"], "cached": info["cached"]}
    symbols = [str(item).strip().upper() for item in (state.get("symbols") or []) if str(item).strip()]
    return {"symbols": symbols, "source": "custom", "cached": True}


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
def market_stats(
    db: Session = Depends(get_db),
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
):
    return MarketDataService(db).stats_page(page, page_size)


@router.get("/sources")
def market_sources(
    db: Session = Depends(get_db),
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
):
    return {
        "local": MarketDataService(db).stats_page(page, page_size),
        "binance": {"available": True, "source_interval": "1m", "note": "仅获取 1m K线，其他周期由本地聚合生成"},
    }


@router.get("/funding")
def market_funding(symbol: str = Query(...), limit: int = Query(50, ge=1, le=1000), db: Session = Depends(get_db)):
    """最近资金费记录 + 当前快照（费率/年化/z-score/24h 累计）。"""
    service = MarketDataService(db)
    frame = service.load_funding_frame(symbol.upper())
    records = [
        {
            "funding_time": int(index.timestamp() * 1000),
            "funding_rate": float(row["funding_rate"]),
            "mark_price": float(row["mark_price"]),
        }
        for index, row in frame.iloc[-limit:].iterrows()
    ]
    return {
        "symbol": symbol.upper(),
        "records": records,
        "current": funding_snapshot(frame),
    }


@router.post("/sync")
def sync_market(payload: SyncRequest, db: Session = Depends(get_db)):
    """同步K线。symbols 为空数组 = 全量 U 本位永续同步（前端"同步全部"），
    单币种契约不变；全量同步使用有界网络并发，HTTP 请求等待同步完成后返回。"""
    source = payload.source.lower()
    if source != "binance":
        raise HTTPException(400, "source 仅支持 binance")
    # 上限 730 天：拉长历史复验需要 400 天以上窗口；sync_history 内部分页拉取，
    # 请求耗时随天数线性增长，由调用方自行承担（前端下拉仍可按需限制）。
    if payload.days < 1 or payload.days > 730:
        raise HTTPException(400, "days 允许范围 1-730")

    service = MarketDataService(db)
    if payload.symbols is None:
        try:
            result = service.sync_history(payload.symbol.upper(), payload.interval, payload.days)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(502, f"拉取失败: {exc}") from exc
        db.add(JobLog(job="sync", level="info",
                      message=f"{source} {result['symbol']} {result['interval']} 导入 {result.get('inserted', 0)} 根"))
        db.commit()
        return result

    if payload.symbols:
        symbols = [str(item).strip().upper() for item in payload.symbols if str(item).strip()]
    else:
        symbols = universe_symbols(db)  # 空数组 = 全量宇宙
    if not symbols:
        raise HTTPException(400, "同步列表为空")
    if payload.funding:
        # 资金费按币种并发拉取，调用线程顺序写库。
        funding_result = service.sync_funding_many(symbols, payload.days)
        db.add(JobLog(job="sync", level="info",
                      message=f"{source} 资金费同步 请求 {funding_result['requested']} 个，"
                              f"导入 {funding_result['inserted']} 条，失败 {len(funding_result['errors'])} 个"))
        db.commit()
        return {"source": source, "funding": funding_result}
    result = service.sync_many(symbols, payload.interval, payload.days)
    result["source"] = source
    db.add(JobLog(job="sync", level="info",
                  message=f"{source} 全量同步 请求 {result['requested']} 个，导入 {result['inserted']} 根，"
                          f"失败 {len(result['errors'])} 个"))
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
    frame = service.load_frame(symbol, interval, limit=limit)
    if frame.empty:
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
        # 预演用「策略 × 池」绑定的参数（没有池或没有绑定就回落策略配置），
        # 且不受策略总开关影响——否则默认停用的策略永远预演不出信号。
        params = resolve_order_params(
            db, payload.strategy, payload.interval, payload.pool or ""
        )
        signal = engine.evaluate(
            payload.symbol.upper(),
            payload.interval,
            payload.strategy,
            save=False,
            params=params,
            pool=payload.pool or "",
            respect_enabled=False,
        )
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


@router.get("/metrics")
def market_metrics(symbol: str = Query(...), db: Session = Depends(get_db)):
    """最近一次多源指标快照（无则返回 404）。"""
    features = metrics_mod.load_latest_metrics(db, symbol.upper())
    if features is None:
        raise HTTPException(404, f"无 {symbol.upper()} 的指标快照，请先 POST /api/market/metrics/sync")
    return {"symbol": symbol.upper(), "metrics": features}


@router.post("/metrics/sync")
def sync_metrics(payload: MetricsSyncRequest, db: Session = Depends(get_db)):
    """采集多源指标快照并落库。symbols 空=全量宇宙；fng + 市值排行全局只拉一次复用。
    业务编排在 metrics.sync_universe_metrics（路由只做装配与鉴权/日志）。"""
    if payload.source.lower() != "binance":
        raise HTTPException(400, "source 仅支持 binance")
    state = all_state(db)
    proxy = state.get("proxy_url") or None
    if payload.symbols:
        symbols = [str(item).strip().upper() for item in payload.symbols if str(item).strip()]
    else:
        symbols = universe_symbols(db)  # 空=全量宇宙
    if not symbols:
        raise HTTPException(400, "同步列表为空")
    result = metrics_mod.sync_universe_metrics(db, symbols, proxy=proxy, period=payload.period)
    db.add(JobLog(job="metrics_sync", level="info",
                  message=f"多源指标同步 {result['synced']} 个成功，{len(result['errors'])} 个失败"))
    db.commit()
    return {"source": payload.source.lower(), "period": payload.period, **result}
