"""免费多源币种指标：采集 + 特征 + 存储 + 观测接入。

数据源（均在本环境经代理 127.0.0.1:7890 探明可达）：
- 币安合约 /futures/data：散户多空比、大户多空比、主动买卖量比、未平仓量
- alternative.me 恐惧贪婪指数（全市场情绪）
- CoinGecko 市值排行（市值 / 成交额 / 24h·7d 涨跌）

本模块不依赖 service 层，proxy 由调用方透传；异步 API 与脚本都可直接 import 使用。
本迭代只做采集 + 特征 + 存储 + 观测，不改动任何策略决策逻辑。
"""

from __future__ import annotations

import json
import math
import statistics

import httpx

from ..models import CoinMetric
from .futures import BinanceFuturesClient

FNG_URL = "https://api.alternative.me/fng/?limit=1"
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"


def _safe_float(value, default: float = 0.0) -> float:
    """字符串/None/NaN/inf 统一收敛为有限浮点；不可解析或非法记 default。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _safe_int(value, default: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number


# CoinGecko 的 symbol 是基础资产（BTC/ETH），而本系统交易对为 BTCUSDT 等带报价后缀，
# 二者需用基础资产名对齐才能查到市值排行。
_QUOTE_SUFFIXES = ("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USDP", "USDD", "DAI")


def base_asset(symbol: str) -> str:
    """把交易对符号归一到基础资产大写符号：BTCUSDT->BTC, ETHUSDC->ETH。
    剥离已知报价后缀（长度须大于后缀，避免把 USDT 本身当基础资产）。"""
    sym = symbol.upper()
    for suffix in _QUOTE_SUFFIXES:
        if sym.endswith(suffix) and len(sym) > len(suffix):
            return sym[: -len(suffix)]
    return sym


def fetch_fear_greed(proxy: str | None = None, timeout: float = 20.0) -> dict | None:
    """恐惧贪婪指数（全市场情绪）。失败返回 None，不影响其他指标。"""
    try:
        with httpx.Client(proxy=proxy, timeout=timeout) as client:
            resp = client.get(FNG_URL, timeout=timeout)
        resp.raise_for_status()
        data = resp.json().get("data") or []
        if not data:
            return None
        item = data[0]
        return {
            "value": _safe_int(item.get("value"), 0),
            "classification": str(item.get("value_classification") or ""),
            "timestamp": _safe_int(item.get("timestamp"), 0),
        }
    except Exception:
        return None


def fetch_market_ranks(proxy: str | None = None, timeout: float = 20.0) -> dict[str, dict] | None:
    """CoinGecko 市值排行（前 250）。返回 {SYMBOL(大写): {market_cap_rank, market_cap,
    total_volume, change_24h, change_7d}}；失败返回 None（不阻断其他指标）。"""
    try:
        with httpx.Client(proxy=proxy, timeout=timeout) as client:
            resp = client.get(
                COINGECKO_URL,
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 250,
                    "page": 1,
                    "price_change_percentage": "24h,7d",
                },
                timeout=timeout,
            )
        resp.raise_for_status()
        rows = resp.json()
        result: dict[str, dict] = {}
        for row in rows:
            symbol = str(row.get("symbol") or "").upper()
            if not symbol:
                continue
            rank = _safe_int(row.get("market_cap_rank"), 0) or 0
            change_24h = row.get("price_change_percentage_24h_in_currency")
            if change_24h is None:
                change_24h = row.get("price_change_percentage_24h")
            change_7d = row.get("price_change_percentage_7d_in_currency")
            if change_7d is None:
                change_7d = row.get("price_change_percentage_7d")
            result[symbol] = {
                "market_cap_rank": rank,
                "market_cap": _safe_float(row.get("market_cap"), 0.0),
                "total_volume": _safe_float(row.get("total_volume"), 0.0),
                "change_24h": _safe_float(change_24h, 0.0),
                "change_7d": _safe_float(change_7d, 0.0),
            }
        return result
    except Exception:
        return None


def fetch_symbol_metrics(client, symbol: str, period: str = "5m", limit: int = 1) -> dict:
    """单币种合约情绪快照。聚合 5 个币安公开接口（均免 Key）：
    散户多空比 / 大户多空比 / 主动买卖量比 / 当前未平仓量 / 未平仓量历史（取名义价值）。
    任一接口失败抛错由调用方捕获；本函数不吞异常（保证数据完整性可观测）。"""
    crowd = client.long_short_ratio(symbol, period, limit)
    smart = client.top_trader_long_short(symbol, period, limit)
    taker = client.taker_buy_sell_ratio(symbol, period, limit)
    oi = client.open_interest(symbol)
    oi_hist = client.open_interest_history(symbol, period, max(int(limit), 2))
    latest_crowd = crowd[-1] if crowd else {}
    latest_smart = smart[-1] if smart else {}
    latest_taker = taker[-1] if taker else {}
    latest_oi_hist = oi_hist[-1] if oi_hist else {}
    return {
        "symbol": symbol.upper(),
        "ts": _safe_int(
            latest_crowd.get("timestamp")
            or latest_smart.get("timestamp")
            or latest_taker.get("timestamp")
            or oi.get("time")
            or latest_oi_hist.get("timestamp")
            or 0
        ),
        "crowd_long": _safe_float(latest_crowd.get("long_account")),
        "crowd_short": _safe_float(latest_crowd.get("short_account")),
        "crowd_ls": _safe_float(latest_crowd.get("long_short_ratio")),
        "smart_long": _safe_float(latest_smart.get("long_account")),
        "smart_short": _safe_float(latest_smart.get("short_account")),
        "smart_ls": _safe_float(latest_smart.get("long_short_ratio")),
        "taker_buy_sell": _safe_float(latest_taker.get("buy_sell_ratio")),
        "taker_buy_vol": _safe_float(latest_taker.get("buy_vol")),
        "taker_sell_vol": _safe_float(latest_taker.get("sell_vol")),
        "oi": _safe_float(oi.get("open_interest")),
        "oi_value": _safe_float(latest_oi_hist.get("sum_open_interest_value")),
    }


def _zscore(current: float, history_values: list[float]) -> float:
    """样本 < 8 返回 0.0；否则标准 z-score。NaN/inf 收敛。"""
    finite = [v for v in history_values if math.isfinite(v)]
    if len(finite) < 8 or not math.isfinite(current):
        return 0.0
    mean = statistics.fmean(finite)
    std = statistics.pstdev(finite)
    if std <= 0:
        return 0.0
    return _safe_float((current - mean) / std, 0.0)


def build_metric_features(
    snapshot: dict,
    fng: dict | None = None,
    market_rank: dict | None = None,
    history: list[dict] | None = None,
) -> dict:
    """由单币种快照 + 全市场情绪 + 市值排行 + 历史存储，构造观测特征。
    history 为 load_metric_history 返回的已存特征列表（用于 oi_change_pct 与 z-score）。
    所有输出收敛为有限浮点，NaN/inf → 0.0。"""
    history = history or []
    crowd_ls = _safe_float(snapshot.get("crowd_ls"))
    smart_ls = _safe_float(snapshot.get("smart_ls"))
    taker_buy_sell = _safe_float(snapshot.get("taker_buy_sell"))
    oi = _safe_float(snapshot.get("oi"))
    oi_value = _safe_float(snapshot.get("oi_value"))

    # 未平仓量环比：相对最近一条已存快照
    oi_change_pct = 0.0
    for prev in history:
        prev_oi = _safe_float(prev.get("oi"))
        if prev_oi > 0:
            oi_change_pct = _safe_float((oi - prev_oi) / prev_oi, 0.0)
            break

    # 历史序列（排除当前，因尚未落库）
    hist_crowd = [h.get("crowd_ls", 0.0) for h in history]
    hist_smart = [h.get("smart_ls", 0.0) for h in history]
    hist_taker = [h.get("taker_buy_sell", 0.0) for h in history]
    hist_oi_value = [h.get("oi_value", 0.0) for h in history]

    fng_value = _safe_int(fng.get("value"), 0) if fng else 0
    rank = market_rank or {}

    return {
        "crowd_ls": crowd_ls,
        "smart_ls": smart_ls,
        "taker_buy_sell": taker_buy_sell,
        "oi": oi,
        "oi_value": oi_value,
        "oi_change_pct": oi_change_pct,
        "sentiment_divergence": _safe_float(smart_ls - crowd_ls, 0.0),
        "crowd_ls_z": _zscore(crowd_ls, hist_crowd),
        "smart_ls_z": _zscore(smart_ls, hist_smart),
        "taker_buy_sell_z": _zscore(taker_buy_sell, hist_taker),
        "oi_value_z": _zscore(oi_value, hist_oi_value),
        "fng_value": float(fng_value),
        "fng_classification": (fng.get("classification") or "") if fng else "",
        "fng_timestamp": _safe_int(fng.get("timestamp"), 0) if fng else 0,
        "market_cap_rank": _safe_int(rank.get("market_cap_rank"), 0),
        "market_cap": _safe_float(rank.get("market_cap"), 0.0),
        "vol_24h": _safe_float(rank.get("total_volume"), 0.0),
        "change_24h": _safe_float(rank.get("change_24h"), 0.0),
        "change_7d": _safe_float(rank.get("change_7d"), 0.0),
    }


def save_metrics_snapshot(db, symbol: str, features: dict, ts: int) -> CoinMetric:
    row = CoinMetric(symbol=symbol.upper(), ts=int(ts), metrics=json.dumps(features, ensure_ascii=False))
    db.add(row)
    db.commit()
    return row


def load_latest_metrics(db, symbol: str) -> dict | None:
    row = (
        db.query(CoinMetric)
        .filter(CoinMetric.symbol == symbol.upper())
        .order_by(CoinMetric.ts.desc())
        .first()
    )
    if row is None:
        return None
    return json.loads(row.metrics)


def load_metric_history(db, symbol: str, limit: int = 200) -> list[dict]:
    rows = (
        db.query(CoinMetric)
        .filter(CoinMetric.symbol == symbol.upper())
        .order_by(CoinMetric.ts.desc())
        .limit(int(limit))
        .all()
    )
    return [json.loads(r.metrics) for r in rows]


def sync_universe_metrics(
    db, symbols: list[str], proxy: str | None = None, period: str = "5m"
) -> dict:
    """采集并落库多币种指标快照（API 层业务编排入口，保持路由只做装配）。

    全市场情绪(fng) + 市值排行(ranks) 全局只拉一次并复用；逐币种走
    fetch_symbol_metrics -> build_metric_features -> save_metrics_snapshot。
    任一币种失败不影响其余（错误汇总到 errors）。返回 {synced, errors, results, fng, ranks_ok}。
    """
    fng = fetch_fear_greed(proxy=proxy)
    ranks = fetch_market_ranks(proxy=proxy)
    # 真实市场情绪数据只在主网存在，testnet 无意义
    client = BinanceFuturesClient(testnet=False, proxy=proxy)
    results: list[dict] = []
    errors: list[dict] = []
    for sym in symbols:
        try:
            snapshot = fetch_symbol_metrics(client, sym, period, 1)
            history = load_metric_history(db, sym, limit=200)
            features = build_metric_features(snapshot, fng, ranks.get(base_asset(sym)), history)
            save_metrics_snapshot(db, sym, features, snapshot["ts"])
            results.append({"symbol": sym, "features": features})
        except Exception as exc:
            errors.append({"symbol": sym, "error": str(exc)[:300]})
    return {
        "synced": len(results),
        "errors": errors,
        "results": results,
        "fng": fng,
        "ranks_ok": ranks is not None,
    }
