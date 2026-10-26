"""U 本位永续合约币种宇宙：从 /fapi/v1/exchangeInfo 过滤全量永续，TTL 内存缓存 + 失败回退。

用户已确认默认范围 = U 本位永续全量（accept 单轮巡检耗时随币种数增长），此处不做范围裁剪。

两层过滤（叠加）：
1. 流动性过滤：按 24h 成交额（/fapi/v1/ticker/24hr，公开免 Key）剔除低于 min_quote_volume_usd
   的币种，让回测/实盘的滑点与成本假设可信；ticker 拉取失败时本轮不过滤（宁贵不漏），
   只记 JobLog；ticker 映射与 exchangeInfo 同 TTL（1h）缓存。
2. 黑名单过滤：symbol_blacklist 中的币种直接排除（负期望/回撤过大的复盘结论落地）。

宇宙缓存键包含过滤配置（流动性门槛 + 黑名单），设置变更立即生效，不必等 TTL。
"""

from __future__ import annotations

import time

from sqlalchemy.orm import Session

from ..config import DEFAULT_SYMBOLS
from ..models import JobLog
from .futures import BinanceFuturesClient

UNIVERSE_TTL_SECONDS = 3600
LEVERAGE_TOKEN_MARKERS = ("UP", "DOWN", "BULL", "BEAR")

_cache: dict[str, list[str]] = {}
_cache_at: dict[str, float] = {}
_ticker_cache: dict[str, dict[str, float]] = {}
_ticker_cache_at: dict[str, float] = {}


def _is_leverage_token(symbol: str) -> bool:
    """杠杆代币形如 BTCUPUSDT / ETHDOWNUSDT / BNBBEARUSDT：去掉 USDT 后以上述标记结尾。
    不用子串包含判断，避免误杀 SUPERUSDT 这类正常合约。"""
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    return base.endswith(LEVERAGE_TOKEN_MARKERS)


def _filter(rows: list[dict]) -> list[str]:
    symbols: list[str] = []
    for row in rows:
        if (
            row.get("contractType") != "PERPETUAL"
            or row.get("quoteAsset") != "USDT"
            or row.get("status") != "TRADING"
        ):
            continue
        symbol = str(row.get("symbol") or "")
        if not symbol or _is_leverage_token(symbol):
            continue
        symbols.append(symbol)
    return sorted(symbols)


def _ticker_volumes(client: BinanceFuturesClient, cache_key: str) -> dict[str, float] | None:
    """ticker 成交额映射，与 exchangeInfo 同 TTL 缓存；拉取失败/空返回 None 表示不过滤。"""
    cached = _ticker_cache.get(cache_key)
    if cached is not None and time.time() - _ticker_cache_at.get(cache_key, 0.0) < UNIVERSE_TTL_SECONDS:
        return cached
    volumes = client.ticker_24hr()
    if not volumes:
        return None
    _ticker_cache[cache_key] = volumes
    _ticker_cache_at[cache_key] = time.time()
    return volumes


def _blacklist(state: dict) -> list[str]:
    """黑名单归一：大写、去空白、去重（保序）；非数组按无黑名单处理。"""
    raw = state.get("symbol_blacklist") or []
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for item in raw:
        symbol = str(item).strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            result.append(symbol)
    return result


def _fetch(testnet: bool, proxy: str | None, cache_key: str, min_volume: float, db: Session,
           blacklist: list[str]) -> list[str]:
    client = BinanceFuturesClient(testnet=testnet, proxy=proxy)
    info = client.exchange_info()
    symbols = _filter(info.get("symbols", []))
    if not symbols:
        raise ValueError("exchangeInfo 未返回可交易的 U 本位永续合约")
    if blacklist:
        blocked = set(blacklist)
        symbols = [symbol for symbol in symbols if symbol not in blocked]
        if not symbols:
            db.add(JobLog(job="market_universe", level="warning",
                          message=f"黑名单清空了可交易宇宙: {', '.join(blacklist)}"))
            db.commit()
            return symbols
    if min_volume <= 0:
        return symbols
    try:
        volumes = _ticker_volumes(client, cache_key)
    except Exception:
        volumes = None
    if not volumes:
        # 宁贵不漏：拿不到成交额就不过滤，避免把可交易合约误杀出宇宙
        db.add(JobLog(job="market_universe", level="error",
                      message="ticker/24hr 拉取失败，本轮不做流动性过滤（宁贵不漏）"))
        db.commit()
        return symbols
    return [symbol for symbol in symbols if volumes.get(symbol, 0.0) >= min_volume]


def resolve_universe(db: Session) -> dict:
    """返回 {"symbols": [...], "cached": bool, "source": "futures-testnet"|"futures-mainnet"}。
    cached=True 表示未发起新请求（缓存命中、过期缓存兜底或回退默认列表）。"""
    from ..state import all_state

    state = all_state(db)
    # 与 K 线源严格同源：live+testnet → 测试网 fapi；live 主网与 paper（固定主网）→ 主网 fapi
    testnet = state.get("execution_mode") == "live" and bool(state.get("testnet", True))
    proxy = state.get("proxy_url") or None
    try:
        min_volume = float(state.get("min_quote_volume_usd") or 0)
    except (TypeError, ValueError):
        min_volume = 0.0
    blacklist = _blacklist(state)
    net_key = "testnet" if testnet else "mainnet"
    source = "futures-testnet" if testnet else "futures-mainnet"
    # 过滤配置进缓存键：黑名单/流动性门槛变更立即生效，不用等 TTL 过期
    cache_key = f"{net_key}|{min_volume:g}|{','.join(blacklist)}"
    cached_symbols = _cache.get(cache_key)
    if cached_symbols is not None and time.time() - _cache_at.get(cache_key, 0.0) < UNIVERSE_TTL_SECONDS:
        return {"symbols": list(cached_symbols), "cached": True, "source": source}
    try:
        symbols = _fetch(testnet, proxy, net_key, min_volume, db, blacklist)
        _cache[cache_key] = symbols
        _cache_at[cache_key] = time.time()
        return {"symbols": list(symbols), "cached": False, "source": source}
    except Exception as exc:
        db.add(JobLog(job="market_universe", level="error", message=f"币种宇宙拉取失败: {exc}"[:4000]))
        db.commit()
        if cached_symbols is not None:
            return {"symbols": list(cached_symbols), "cached": True, "source": source}
        return {"symbols": list(DEFAULT_SYMBOLS), "cached": True, "source": source}


def universe_symbols(db: Session) -> list[str]:
    return resolve_universe(db)["symbols"]


def universe_volumes(db: Session) -> dict[str, float]:
    """24h 成交额映射（与 resolve_universe 同 TTL 缓存）。拿不到返回 {}，
    调用方按原顺序取 TopN，不要因为缺成交额就把币种误杀。"""
    from ..state import all_state

    state = all_state(db)
    testnet = state.get("execution_mode") == "live" and bool(state.get("testnet", True))
    proxy = state.get("proxy_url") or None
    client = BinanceFuturesClient(testnet=testnet, proxy=proxy)
    try:
        return _ticker_volumes(client, "testnet" if testnet else "mainnet") or {}
    except Exception:
        return {}
