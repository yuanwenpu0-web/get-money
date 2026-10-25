"""K 线本地存储。选用 SQLite + WAL：无需额外服务，单文件、按 (symbol, interval, open_time) 唯一索引，
2 个币种 1 分钟线多年数据量级在千万行以内，查询与增量写入足够快，也便于随项目备份。"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from ..models import Kline
from ..utils import interval_ms, now_ms
from .binance import BinanceClient, parse_klines


class MarketDataService:
    def __init__(self, db: Session, client: BinanceClient | None = None):
        self.db = db
        self.client = client or BinanceClient(testnet=False)

    def latest_open_time(self, symbol: str, interval: str) -> int:
        row = self.db.execute(
            select(Kline.open_time)
            .where(Kline.symbol == symbol, Kline.interval == interval)
            .order_by(Kline.open_time.desc())
            .limit(1)
        ).first()
        return int(row[0]) if row else 0

    def store(self, symbol: str, interval: str, rows: Iterable[dict]) -> int:
        payload = []
        for row in rows:
            payload.append({"symbol": symbol, "interval": interval, **row})
        if not payload:
            return 0
        # 分批 upsert，避免 SQLite 参数上限
        for start in range(0, len(payload), 500):
            chunk = payload[start : start + 500]
            stmt = insert(Kline).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol", "interval", "open_time"],
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                    "quote_volume": stmt.excluded.quote_volume,
                    "close_time": stmt.excluded.close_time,
                    "trades": stmt.excluded.trades,
                },
            )
            self.db.execute(stmt)
        self.db.commit()
        return len(payload)

    def sync_recent(self, symbol: str, interval: str, tail_limit: int = 1000) -> dict:
        last = self.latest_open_time(symbol, interval)
        step = interval_ms(interval)
        if last == 0:
            start = now_ms() - step * tail_limit
        else:
            # Refresh the previous tail: it may have been imported before closing.
            start = last

        total = 0
        for _ in range(10):
            rows = self.client.klines(symbol, interval, start_ms=start, limit=1000)
            parsed = parse_klines(rows)
            if not parsed:
                break
            total += self.store(symbol, interval, parsed)
            if len(parsed) < 1000:
                break
            start = parsed[-1]["open_time"] + step

        return {
            "symbol": symbol,
            "interval": interval,
            "inserted": total,
            "latest_open_time": self.latest_open_time(symbol, interval),
        }

    def sync_history(self, symbol: str, interval: str, days: int = 30) -> dict:
        step = interval_ms(interval)
        start = now_ms() - days * 86_400_000
        total = 0
        guard = 0
        while start < now_ms() and guard < 500:
            rows = self.client.klines(symbol, interval, start_ms=start, limit=1000)
            parsed = parse_klines(rows)
            if not parsed:
                break
            total += self.store(symbol, interval, parsed)
            if len(parsed) < 1000:
                break
            start = parsed[-1]["open_time"] + step
            guard += 1
        return {"symbol": symbol, "interval": interval, "inserted": total, "days": days}

    def load_frame(
        self,
        symbol: str,
        interval: str,
        limit: int = 3000,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame:
        query = select(Kline).where(Kline.symbol == symbol, Kline.interval == interval)
        if start_ms:
            query = query.where(Kline.open_time >= start_ms)
        if end_ms:
            query = query.where(Kline.open_time <= end_ms)
        if limit and not start_ms:
            query = query.order_by(Kline.open_time.desc()).limit(limit)
            rows = list(reversed(self.db.execute(query).scalars().all()))
        else:
            query = query.order_by(Kline.open_time.asc())
            rows = self.db.execute(query).scalars().all()
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "open_time", "quote_volume"])

        frame = pd.DataFrame(
            [
                {
                    "open_time": row.open_time,
                    "open": row.open,
                    "high": row.high,
                    "low": row.low,
                    "close": row.close,
                    "volume": row.volume,
                    "quote_volume": row.quote_volume,
                }
                for row in rows
            ]
        )
        index = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        frame.index = index
        frame.index.name = "time"
        return frame.drop(columns=["open_time"])

    def stats(self) -> list[dict]:
        from sqlalchemy import func

        rows = self.db.execute(
            select(
                Kline.symbol,
                Kline.interval,
                func.count(Kline.id),
                func.min(Kline.open_time),
                func.max(Kline.open_time),
            ).group_by(Kline.symbol, Kline.interval)
        ).all()
        result = []
        for symbol, interval, count, first, last in rows:
            result.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "count": count,
                    "first_open_time": int(first or 0),
                    "last_open_time": int(last or 0),
                }
            )
        return result
