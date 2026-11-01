"""按最近完整小时行情维护研究分组；不创建策略或交易绑定。"""
from __future__ import annotations

import json
import math
import statistics
import threading
import logging
from datetime import datetime, timezone

from sqlalchemy.exc import IntegrityError

from ..market.data_service import MarketDataService
from ..models import Kline, SymbolPool
from ..state import get_state, set_state
from .coin_universe import BLACKLIST

HOUR = 3600000
WINDOW_HOURS = 168
MAX_AGE_HOURS = 2
MIN_ELIGIBLE = 100
POOL_CODES = ("research_high_liquidity_high_volatility", "research_low_liquidity_high_volatility")
OWNER = "rolling_high_volatility_v1"
STATUS_KEY = "research_pool_maintenance"
_maintenance_lock = threading.Lock()
EXCLUDED = {"BTCUSDT", "ETHUSDT", "USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "USDPUSDT", "DAIUSDT", "EURUSDT", "GBPUSDT", "PAXGUSDT"} | BLACKLIST


def research_pool_candidates(db, now_ms: int | None = None) -> dict:
    now = now_ms if now_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
    end = now // HOUR * HOUR
    start = end - WINDOW_HOURS * HOUR
    symbols = (db.query(Kline.symbol).filter(Kline.interval.in_(("1m", "1h")),
               Kline.open_time >= start, Kline.open_time < end).distinct().all())
    market = MarketDataService(db, client=object())
    eligible = []
    for (symbol,) in symbols:
        if (not symbol.endswith("USDT") or symbol in EXCLUDED or
                symbol.endswith(("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"))):
            continue
        try:
            minutes = (db.query(Kline.open_time, Kline.close_time, Kline.close, Kline.quote_volume)
                       .filter(Kline.symbol == symbol, Kline.interval == "1m",
                               Kline.open_time >= start, Kline.open_time < end).yield_per(5000))
            if any(m.open_time % 60000 or m.close_time != m.open_time + 59999
                   or m.close is None or not math.isfinite(m.close) or m.close <= 0
                   or m.quote_volume is None or not math.isfinite(m.quote_volume) or m.quote_volume < 0
                   for m in minutes):
                continue
            candles, _ = market._merged_candle_rows(symbol, "1h", start, end, allow_native=True)
            candles = [c for c in candles if c["open_time"] < end]
            if len(candles) != WINDOW_HOURS:
                continue
            if any(c["open_time"] != start + i * HOUR or c["close_time"] != c["open_time"] + HOUR - 1
                   or c["close"] is None or not math.isfinite(c["close"]) or c["close"] <= 0
                   or c["quote_volume"] is None or not math.isfinite(c["quote_volume"]) or c["quote_volume"] < 0
                   for i, c in enumerate(candles)):
                continue
            if now - (candles[-1]["open_time"] + HOUR) > MAX_AGE_HOURS * HOUR:
                continue
            volume = sum(c["quote_volume"] for c in candles)
            if not math.isfinite(volume) or volume <= 0:
                continue
            returns = [math.log(b["close"]) - math.log(a["close"]) for a, b in zip(candles, candles[1:])]
            eligible.append((symbol, volume, statistics.stdev(returns)))
        except Exception:
            logging.getLogger(__name__).exception("研究币池行情处理失败: %s", symbol)
            continue
    eligible.sort(key=lambda item: (-item[1], item[0]))
    if len(eligible) < MIN_ELIGIBLE:
        return {"updated": False, "reason": f"合格行情不足{MIN_ELIGIBLE}个", "eligible": len(eligible)}
    top = eligible[:100]
    volatile = {x[0] for x in sorted(top, key=lambda item: (-item[2], item[0]))[:50]}
    members = [[x[0] for x in top[:50] if x[0] in volatile],
               [x[0] for x in top[50:] if x[0] in volatile]]
    if not all(members):
        return {"updated": False, "reason": "目标池为空，保留原值", "eligible": len(eligible)}
    return {"updated": True, "eligible": len(eligible), "window_end_ms": end,
            "members": dict(zip(POOL_CODES, members))}


def refresh_research_pools(db, now_ms: int | None = None) -> dict:
    result = research_pool_candidates(db, now_ms)
    if not result["updated"]:
        return result
    end = result["window_end_ms"]
    members = [result["members"][code] for code in POOL_CODES]
    existing = {p.code: p for p in db.query(SymbolPool).filter(SymbolPool.code.in_(POOL_CODES)).all()}
    for pool in existing.values():
        try:
            owned = json.loads(pool.rule or "{}").get("maintenance_owner") == OWNER
        except (ValueError, AttributeError):
            owned = False
        if not owned or pool.kind != "static":
            return {"updated": False, "reason": f"池代码已被其他对象占用: {pool.code}"}
    metadata = json.dumps({"base": "static", "maintenance_owner": OWNER, "window_hours": WINDOW_HOURS,
                           "window_end_ms": end}, ensure_ascii=False)
    for code, name, symbols in zip(POOL_CODES, ("高成交额高波动", "低成交额高波动"), members):
        pool = existing.get(code)
        if pool is None:
            pool = SymbolPool(code=code, name=name, kind="static", enabled=0)
            db.add(pool)
        pool.description = "最近168小时合格币种至少100个，成交额前100与波动率前50交叉分类"
        pool.symbols = json.dumps(symbols, ensure_ascii=False)
        pool.rule = metadata
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return {"updated": False, "reason": "并发创建冲突，等待下次维护"}
    return {"updated": True, "eligible": result["eligible"], "window_end_ms": end,
            "counts": dict(zip(POOL_CODES, map(len, members)))}


def run_research_pool_maintenance(db) -> dict:
    if not _maintenance_lock.acquire(blocking=False):
        return {"updated": False, "reason": "币池维护正在进行中"}
    try:
        try:
            result = refresh_research_pools(db)
        except Exception:
            db.rollback()
            logging.getLogger(__name__).exception("研究币池维护失败")
            result = {"updated": False, "reason": "维护异常，请查看服务日志"}
        status = {**result, "attempted_at": datetime.now(timezone.utc).isoformat()}
        set_state(db, STATUS_KEY, status)
        return status
    finally:
        _maintenance_lock.release()


def research_pool_status(db) -> dict:
    pools = {p.code: p for p in db.query(SymbolPool).filter(SymbolPool.code.in_(POOL_CODES)).all()}
    from .pools import load_pool_spec, pool_symbols
    result = []
    for code, name in zip(POOL_CODES, ("高成交额高波动", "低成交额高波动")):
        pool = pools.get(code)
        if pool is None:
            result.append({"code": code, "name": name, "status": "未创建", "members": [], "window_end_ms": None})
            continue
        spec = load_pool_spec(pool)
        rule = spec["rule"]
        owned = pool.kind == "static" and rule.get("maintenance_owner") == OWNER
        result.append({"code": code, "name": pool.name, "status": ("已维护" if rule.get("window_end_ms") else "待维护") if owned else "代码被占用",
                       "enabled": bool(pool.enabled), "members": pool_symbols(db, pool) if owned else [],
                       "window_end_ms": rule.get("window_end_ms") if owned else None})
    return {"pools": result, "last_attempt": get_state(db, STATUS_KEY, None)}
