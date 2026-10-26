"""指标历史回填 + 反前视读取原语（迭代 8）。

背景：迭代 7 只落了一次快照，z-score 无历史，回测无法使用。本模块补齐历史：
- 币安 /futures/data/* 与 openInterestHist 支持 period=1d&limit=500，一次调用覆盖约 500 天
- Fear&Greed 支持 limit=500 日报历史，以哨兵 symbol 落库
- CoinGecko 市值排名无免费历史 → 回测按常量处理（报告显式标注前视风险）

反前视硬红线：metrics_as_of 只取 ts <= ts_ms 的最新一行，绝不读取未来行。
所有加权必须经它取数，不得直接用 load_latest_metrics。
"""

from __future__ import annotations

import json

import httpx

from ..models import CoinMetric
from .metrics import (
    _safe_float,
    _safe_int,
    base_asset,
    build_metric_features,
    fetch_market_ranks,
    load_latest_metrics,
    save_metrics_snapshot,
)

FNG_SENTINEL = "__FNG__"  # 恐惧贪婪指数历史的哨兵 symbol（非真实交易对）
FNG_HISTORY_URL = "https://api.alternative.me/fng/?limit={limit}"
DAY_MS = 86_400_000
DEFAULT_BACKFILL_DAYS = 500


def metrics_as_of(db, symbol: str, ts_ms: int) -> dict:
    """反前视读取：取 ts <= ts_ms 的历史行中最新一条（前向填充）。
    绝不读取 ts 之后的行；无数据返回 {}。"""
    row = (
        db.query(CoinMetric)
        .filter(CoinMetric.symbol == symbol.upper(), CoinMetric.ts <= int(ts_ms))
        .order_by(CoinMetric.ts.desc())
        .first()
    )
    if row is None:
        return {}
    try:
        return json.loads(row.metrics) or {}
    except (TypeError, ValueError):
        return {}


def backfill_fear_greed(db, proxy: str | None = None, limit: int = DEFAULT_BACKFILL_DAYS,
                        timeout: float = 30.0) -> dict:
    """拉取 FNG 日报历史（limit=500 覆盖约 500 天），以哨兵 symbol 落库。
    哨兵行存储原始 FNG 载荷 {value, classification, timestamp}，可直接喂给
    build_metric_features 的 fng 参数。返回 {rows, span_days, error?}。"""
    try:
        with httpx.Client(proxy=proxy, timeout=timeout) as client:
            resp = client.get(FNG_HISTORY_URL.format(limit=int(limit)), timeout=timeout)
        resp.raise_for_status()
        data = resp.json().get("data") or []
    except Exception as exc:  # noqa: BLE001 - 单源失败不阻断
        return {"rows": 0, "span_days": 0, "error": str(exc)[:200]}
    stamps: list[int] = []
    for item in data:
        ts = _safe_int(item.get("timestamp"), 0)
        if ts <= 0:
            continue
        ts_ms = ts * 1000 if ts < 10**12 else ts  # FNG 为秒级时间戳
        payload = {
            "value": _safe_int(item.get("value"), 0),
            "classification": str(item.get("value_classification") or ""),
            "timestamp": ts,
        }
        save_metrics_snapshot(db, FNG_SENTINEL, payload, ts_ms)
        stamps.append(ts_ms)
    if not stamps:
        return {"rows": 0, "span_days": 0}
    return {"rows": len(stamps), "span_days": int((max(stamps) - min(stamps)) / DAY_MS),
            "symbol": FNG_SENTINEL}


def _bucketize(rows: list[dict], field_map: dict[str, str]) -> dict[int, dict]:
    """[{timestamp, 源字段}] → {ts: {目标字段: float}}，timestamp 为对齐键。"""
    buckets: dict[int, dict] = {}
    for row in rows or []:
        ts = _safe_int(row.get("timestamp"), 0)
        if ts <= 0:
            continue
        bucket = buckets.setdefault(ts, {})
        for src, dst in field_map.items():
            if row.get(src) is not None:
                bucket[dst] = _safe_float(row.get(src))
    return buckets


def backfill_symbol_metrics(client, db, symbol: str, period: str = "1d",
                            limit: int = DEFAULT_BACKFILL_DAYS,
                            market_rank: dict | None = None) -> dict:
    """回填单币种日报历史：调 4 类公开方法（3 个 ratio + openInterestHist），
    按 timestamp 对齐合并（缺失字段前向填充），逐条落库。
    z-score 因果构造：每条只用"它之前"的已算特征作历史，绝不掺入未来。
    返回 {symbol, rows, span_days}。"""
    symbol = symbol.upper()
    crowd = client.long_short_ratio(symbol, period, limit)
    smart = client.top_trader_long_short(symbol, period, limit)
    taker = client.taker_buy_sell_ratio(symbol, period, limit)
    oi_hist = client.open_interest_history(symbol, period, limit)

    buckets: dict[int, dict] = {}
    for rows, mapping in (
        (crowd, {"long_short_ratio": "crowd_ls", "long_account": "crowd_long",
                 "short_account": "crowd_short"}),
        (smart, {"long_short_ratio": "smart_ls", "long_account": "smart_long",
                 "short_account": "smart_short"}),
        (taker, {"buy_sell_ratio": "taker_buy_sell", "buy_vol": "taker_buy_vol",
                 "sell_vol": "taker_sell_vol"}),
        (oi_hist, {"sum_open_interest": "oi", "sum_open_interest_value": "oi_value"}),
    ):
        for ts, values in _bucketize(rows, mapping).items():
            buckets.setdefault(ts, {}).update(values)
    if not buckets:
        return {"symbol": symbol, "rows": 0, "span_days": 0}

    ordered = sorted(buckets)
    history: list[dict] = []
    carried: dict = {}
    for ts in ordered:
        merged = dict(carried)  # 前向填充：某日某源缺失时沿用上一日
        merged.update(buckets[ts])
        merged["symbol"], merged["ts"] = symbol, ts
        carried = merged
        fng = metrics_as_of(db, FNG_SENTINEL, ts) or None
        features = build_metric_features(merged, fng, market_rank, history)
        save_metrics_snapshot(db, symbol, features, ts)
        history.append(features)
    return {"symbol": symbol, "rows": len(ordered),
            "span_days": int((ordered[-1] - ordered[0]) / DAY_MS)}


def load_market_rank(db, symbol: str, proxy: str | None = None) -> int:
    """当前市值排名。CoinGecko 无免费历史，回测中按常量使用（前视风险见报告）。
    优先读已存快照的 market_cap_rank；无且给了 proxy 时才现拉一次。"""
    latest = load_latest_metrics(db, symbol.upper()) or {}
    rank = _safe_int(latest.get("market_cap_rank"), 0)
    if rank > 0:
        return rank
    if proxy:
        ranks = fetch_market_ranks(proxy=proxy) or {}
        return _safe_int(ranks.get(base_asset(symbol), {}).get("market_cap_rank"), 0)
    return 0
