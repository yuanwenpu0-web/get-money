"""从 nofx-new 项目的 PostgreSQL 只读导入已有K线，再落到本地 SQLite。"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import NOFX_DB_URL, NOFX_SYMBOL_PREFIX
from ..models import Kline
from ..utils import interval_ms, now_ms
from .data_service import MarketDataService


def _connect():
    try:
        import psycopg2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("缺少 psycopg2-binary，无法读取 nofx 数据库") from exc
    conn = psycopg2.connect(NOFX_DB_URL, connect_timeout=5)
    conn.set_session(readonly=True, autocommit=True)
    return conn


def source_status(symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")) -> dict:
    try:
        conn = _connect()
    except Exception as exc:
        return {"available": False, "error": str(exc)[:300], "database": NOFX_DB_URL.split("@")[-1]}
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT count(DISTINCT symbol) FROM market_klines WHERE symbol LIKE %s",
                (NOFX_SYMBOL_PREFIX + "%",),
            )
            symbol_count = int(cursor.fetchone()[0] or 0)
            cursor.execute(
                "SELECT symbol, interval, count(*), min(open_time), max(open_time) "
                "FROM market_klines WHERE symbol = ANY(%s) GROUP BY symbol, interval "
                "ORDER BY symbol, interval",
                ([NOFX_SYMBOL_PREFIX + symbol for symbol in symbols],),
            )
            rows = cursor.fetchall()
        items = [
            {
                "symbol": row[0].replace(NOFX_SYMBOL_PREFIX, ""),
                "interval": row[1],
                "count": int(row[2]),
                "first_open_time": int(row[3] or 0),
                "last_open_time": int(row[4] or 0),
            }
            for row in rows
        ]
        return {
            "available": True,
            "database": NOFX_DB_URL.split("@")[-1],
            "symbol_count": symbol_count,
            "intervals": sorted({item["interval"] for item in items}),
            "items": items,
        }
    finally:
        conn.close()


def import_klines(db: Session, symbol: str, interval: str, days: int = 0, incremental: bool = False) -> dict:
    service = MarketDataService(db)
    last_local = service.latest_open_time(symbol, interval) if incremental else 0
    start_ms = max(0, now_ms() - days * 86_400_000) if days else 0
    if incremental and last_local:
        start_ms = max(start_ms, last_local - interval_ms(interval) * 3)

    conn = _connect()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT open_time, open, high, low, close, volume, close_time, quote_volume, trade_count "
                "FROM market_klines WHERE symbol=%s AND interval=%s AND open_time >= %s "
                "ORDER BY open_time ASC",
                (NOFX_SYMBOL_PREFIX + symbol, interval, start_ms),
            )
            rows = cursor.fetchall()
    finally:
        conn.close()

    parsed = [
        {
            "open_time": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
            "close_time": int(row[6]),
            "quote_volume": float(row[7]),
            "trades": int(row[8] or 0),
        }
        for row in rows
    ]
    inserted = service.store(symbol, interval, parsed)
    return {
        "symbol": symbol,
        "interval": interval,
        "source": "nofx",
        "imported": inserted,
        "rows": len(parsed),
        "first_open_time": parsed[0]["open_time"] if parsed else 0,
        "last_open_time": parsed[-1]["open_time"] if parsed else last_local,
        "incremental": incremental,
    }


def local_gap_summary(db: Session, symbol: str, interval: str, expected_from: int | None = None) -> dict:
    total = db.execute(
        select(func.count(Kline.id)).where(Kline.symbol == symbol, Kline.interval == interval)
    ).scalar_one()
    first, last = db.execute(
        select(func.min(Kline.open_time), func.max(Kline.open_time)).where(
            Kline.symbol == symbol, Kline.interval == interval
        )
    ).one()
    return {"symbol": symbol, "interval": interval, "rows": int(total), "first": first or 0, "last": last or 0}


def bootstrap_defaults(db: Session, symbols: list[str], interval: str = "1m") -> list[dict]:
    """本地没有数据时，从 nofx 导入默认币种，保证系统开箱可用。"""
    results = []
    for symbol in symbols:
        exists = db.execute(
            select(func.count(Kline.id)).where(Kline.symbol == symbol, Kline.interval == interval)
        ).scalar_one()
        if exists:
            continue
        try:
            results.append(import_klines(db, symbol, interval, days=0))
        except Exception as exc:
            results.append({"symbol": symbol, "interval": interval, "error": str(exc)[:300]})
    return results


def frame_to_readable(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    return [
        {
            "time": int(index.timestamp() * 1000),
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
        }
        for index, row in frame.iterrows()
    ]


def utc_label(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
