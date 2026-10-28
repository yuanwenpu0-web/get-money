"""Store exchange 1m candles; derive every larger interval locally from that source."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from contextlib import nullcontext
from typing import Callable, Iterable, Iterator

import pandas as pd
from sqlalchemy import and_, func, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, aliased

from ..config import MARKET_NATIVE_FALLBACK, MARKET_SYNC_WORKERS
from ..models import FundingRate, Kline, Sentiment
from ..pagination import page_metadata
from ..state import all_state
from ..utils import interval_ms, now_ms
from .binance import BinanceClient, parse_klines
from .futures import BinanceFuturesClient

FUNDING_PAGE = 1000
FUNDING_STEP_MS = 8 * 3_600_000
MINUTE_MS = 60_000
KLINE_PAGE_SIZE = 1000
# 允许用已存储的同周期行补齐的周期。1m 本身就是源，无需回落。
NATIVE_FALLBACK_INTERVALS = ("5m", "15m", "1h", "4h", "1d")


def default_market_client(db: Session | None):
    """行情来源与永续宇宙同源：一律走 fapi 合约公共K线（公开无需签名）。
    live 模式：testnet → 合约测试网，主网 → 合约主网（与真实成交价同源）；
    paper 模式：固定合约主网（价格≈真实行情，合约语义与永续宇宙一致）。
    proxy_url 空字符串表示直连；全局代理对现货/合约与实盘下单链路同时生效。"""
    if db is None:
        return BinanceFuturesClient(testnet=False)
    state = all_state(db)
    proxy = state.get("proxy_url") or None
    if state.get("execution_mode") == "live":
        return BinanceFuturesClient(testnet=bool(state.get("testnet", True)), proxy=proxy)
    return BinanceFuturesClient(testnet=False, proxy=proxy)


class MarketDataService:
    def __init__(self, db: Session, client: BinanceClient | BinanceFuturesClient | None = None):
        self.db = db
        self.client = client if client is not None else default_market_client(db)

    def latest_open_time(self, symbol: str, interval: str) -> int:
        if interval != "1m":
            rows = self._tail_candle_rows(symbol, interval, 1)
            return int(rows[0]["open_time"]) if rows else 0
        row = self.db.execute(
            select(Kline.open_time)
            .where(Kline.symbol == symbol, Kline.interval == interval)
            .order_by(Kline.open_time.desc())
            .limit(1)
        ).first()
        return int(row[0]) if row else 0

    def _insert_fn(self):
        """按当前方言返回 upsert 构造器。

        ON CONFLICT 是各家自己的方言实现，拿 SQLite 的 insert() 去编译 PostgreSQL
        会抛 CompileError，所以必须按 bind 的方言显式分派。
        """
        dialect = self.db.get_bind().dialect.name
        return pg_insert if dialect == "postgresql" else sqlite_insert

    def store(self, symbol: str, interval: str, rows: Iterable[dict]) -> int:
        if interval != "1m":
            raise ValueError("仅存储 1m 原始K线，其他周期由 1m 自动聚合")
        payload = []
        for row in rows:
            payload.append({"symbol": symbol, "interval": interval, **row})
        if not payload:
            return 0
        insert_fn = self._insert_fn()
        # 分批 upsert，SQLite 有参数上限，PG 侧也能避免单条语句过大
        for start in range(0, len(payload), 500):
            chunk = payload[start : start + 500]
            stmt = insert_fn(Kline).values(chunk)
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
        step = interval_ms(interval)
        last = self.latest_open_time(symbol, "1m")
        end = now_ms()
        start = last if last else max(0, (end // step - tail_limit) * step)
        total = self._sync_minutes(symbol, start, end)
        return {
            "symbol": symbol,
            "interval": interval,
            "source_interval": "1m",
            "inserted": total,
            "latest_open_time": self.latest_open_time(symbol, interval),
        }

    def sync_history(self, symbol: str, interval: str, days: int = 30) -> dict:
        step = interval_ms(interval)
        end = now_ms()
        start = max(0, (end - days * 86_400_000) // step * step)
        total = self._sync_minutes(symbol, start, end)
        return {"symbol": symbol, "interval": interval, "source_interval": "1m", "inserted": total, "days": days}

    def _sync_minutes(self, symbol: str, start: int, end: int) -> int:
        """Fetch a fixed range, refreshing its tail, with no silent history page cap."""
        total = 0
        with self._connection_pool(self.client):
            for parsed in self._minute_pages(symbol, start, end):
                total += self.store(symbol, "1m", parsed)
        return total

    def _minute_pages(self, symbol: str, start: int, end: int) -> Iterator[list[dict]]:
        """Network-only iterator: workers never touch the caller's ORM session."""
        while start <= end:
            rows = self.client.klines(symbol, "1m", start_ms=start, limit=KLINE_PAGE_SIZE)
            parsed = sorted(
                (row for row in parse_klines(rows) if start <= row["open_time"] <= end),
                key=lambda row: row["open_time"],
            )
            if not parsed:
                break
            yield parsed
            cursor = parsed[-1]["open_time"] + MINUTE_MS
            if cursor <= start:
                raise RuntimeError("1m K线分页没有前进")
            if len(rows) < KLINE_PAGE_SIZE:
                break
            start = cursor

    @staticmethod
    def _connection_pool(client):
        pool = getattr(client, "connection_pool", None)
        return pool() if pool is not None else nullcontext(client)

    def _sync_pages_many(
        self, symbols: list[str], pages: Callable[[str], Iterator[list[dict]]],
        store: Callable[[str, list[dict]], int], workers: int, client,
    ) -> dict:
        """Bound both network concurrency and buffered data to one page per worker.

        Page commits stay on the calling thread, so this works with SQLite and
        PostgreSQL without sharing a Session or holding a DB connection per worker.
        """
        workers = max(1, min(int(workers), 32, len(symbols)))
        total = 0
        errors: dict[int, str] = {}
        remaining = iter(enumerate(symbols))
        with self._connection_pool(client), ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="market-sync"
        ) as executor:
            pending = {}

            def start_symbol() -> None:
                entry = next(remaining, None)
                if entry is not None:
                    index, symbol = entry
                    iterator = pages(symbol)
                    pending[executor.submit(next, iterator, None)] = (index, symbol, iterator)

            for _ in range(workers):
                start_symbol()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index, symbol, iterator = pending.pop(future)
                    try:
                        page = future.result()
                        if page is not None:
                            total += store(symbol, page)
                            pending[executor.submit(next, iterator, None)] = (index, symbol, iterator)
                            continue
                    except Exception as exc:
                        self.db.rollback()
                        errors[index] = f"{symbol}: {exc}"
                    start_symbol()
        return {"requested": len(symbols), "inserted": total,
                "errors": [errors[index] for index in sorted(errors)]}

    def sync_many(
        self, symbols: list[str], interval: str = "1m", days: int = 30,
        workers: int = MARKET_SYNC_WORKERS,
    ) -> dict:
        """Fetch symbols concurrently while serializing page writes on this session."""
        step = interval_ms(interval)
        end = now_ms()
        start = max(0, (end - days * 86_400_000) // step * step)
        return self._sync_pages_many(
            symbols, lambda symbol: self._minute_pages(symbol, start, end),
            lambda symbol, page: self.store(symbol, "1m", page), workers, self.client,
        )

    def sync_recent_many(
        self, symbols: list[str], interval: str, tail_limit: int = 1000,
        workers: int = MARKET_SYNC_WORKERS,
    ) -> dict:
        """Refresh a shared minute source per symbol without derived-series scans."""
        step = interval_ms(interval)
        end = now_ms()
        fallback = max(0, (end // step - tail_limit) * step)
        # Indexed point lookups stay on the caller thread. Grouping the entire
        # minute table by symbol here would reintroduce a full-history scan.
        starts = {symbol: self.latest_open_time(symbol, "1m") or fallback for symbol in symbols}
        return self._sync_pages_many(
            symbols, lambda symbol: self._minute_pages(symbol, starts[symbol], end),
            lambda symbol, page: self.store(symbol, "1m", page), workers, self.client,
        )

    def sync_funding_many(
        self, symbols: list[str], days: int = 90, workers: int = MARKET_SYNC_WORKERS,
    ) -> dict:
        client = self._futures_client()
        end = now_ms()
        return self._sync_pages_many(
            symbols, lambda symbol: self._funding_pages(client, symbol, end - days * 86_400_000, end),
            self._store_funding, workers, client,
        )

    def _futures_client(self) -> BinanceFuturesClient:
        """资金费一律走 fapi（公开无需签名）：与 K 线客户端保持同源，非 futures 客户端时回退主网。"""
        if isinstance(self.client, BinanceFuturesClient):
            return self.client
        proxy = all_state(self.db).get("proxy_url") or None
        return BinanceFuturesClient(testnet=False, proxy=proxy)

    def _store_funding(self, symbol: str, rows: list[dict]) -> int:
        payload = [
            {
                "symbol": symbol,
                "funding_time": row["funding_time"],
                "funding_rate": row["funding_rate"],
                "mark_price": row["mark_price"],
                "rate_type": row["rate_type"],
            }
            for row in rows
        ]
        insert_fn = self._insert_fn()
        for start in range(0, len(payload), 500):
            chunk = payload[start : start + 500]
            stmt = insert_fn(FundingRate).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol", "funding_time"],
                set_={
                    "funding_rate": stmt.excluded.funding_rate,
                    "mark_price": stmt.excluded.mark_price,
                    "rate_type": stmt.excluded.rate_type,
                },
            )
            self.db.execute(stmt)
        self.db.commit()
        return len(payload)

    def sync_funding(self, symbol: str, days: int = 90) -> dict:
        """分页拉取资金费历史（8 小时 1 条，1000 条约 333 天，一般 1-2 页），幂等 upsert。"""
        end_ms = now_ms()
        start_ms = end_ms - days * 86_400_000
        client = self._futures_client()
        total = 0
        with self._connection_pool(client):
            for rows in self._funding_pages(client, symbol, start_ms, end_ms):
                total += self._store_funding(symbol, rows)
        return {
            "symbol": symbol,
            "requested_days": days,
            "inserted": total,
            "latest_funding_time": self.latest_funding_time(symbol),
        }

    @staticmethod
    def _funding_pages(client, symbol: str, cursor: int, end_ms: int) -> Iterator[list[dict]]:
        for _ in range(10):
            raw = client.funding_rate(symbol, start_ms=cursor, end_ms=end_ms, limit=FUNDING_PAGE)
            rows = sorted(
                (row for row in raw if cursor <= row["funding_time"] <= end_ms),
                key=lambda row: row["funding_time"],
            )
            if not rows:
                break
            yield rows
            if len(raw) < FUNDING_PAGE:
                break
            next_cursor = rows[-1]["funding_time"] + 1
            if next_cursor <= cursor:
                raise RuntimeError("资金费分页没有前进")
            cursor = next_cursor  # startTime INCLUSIVE，+1ms 避免重复页

    def latest_funding_time(self, symbol: str) -> int:
        row = self.db.execute(
            select(FundingRate.funding_time)
            .where(FundingRate.symbol == symbol)
            .order_by(FundingRate.funding_time.desc())
            .limit(1)
        ).first()
        return int(row[0]) if row else 0

    def load_funding_frame(self, symbol: str, start_ms: int | None = None, end_ms: int | None = None) -> pd.DataFrame:
        """资金费序列：索引为 funding_time（UTC DatetimeIndex），列 funding_rate/mark_price。"""
        if self.db is None:
            return pd.DataFrame(columns=["funding_rate", "mark_price"])
        query = select(FundingRate).where(FundingRate.symbol == symbol)
        if start_ms:
            query = query.where(FundingRate.funding_time >= int(start_ms))
        if end_ms:
            query = query.where(FundingRate.funding_time <= int(end_ms))
        rows = self.db.execute(query.order_by(FundingRate.funding_time.asc())).scalars().all()
        if not rows:
            return pd.DataFrame(columns=["funding_rate", "mark_price"])
        frame = pd.DataFrame(
            [
                {
                    "funding_rate": row.funding_rate,
                    "mark_price": row.mark_price,
                }
                for row in rows
            ],
            index=pd.to_datetime([row.funding_time for row in rows], unit="ms", utc=True),
        )
        frame.index.name = "funding_time"
        return frame

    # ---------------- P9 正交衍生品情绪（OI/多空比/taker，免 Key 公开接口） ----------------

    def sync_sentiment_many(
        self, symbols: list[str], workers: int = MARKET_SYNC_WORKERS,
    ) -> dict:
        """拉取每个 symbol 的 OI 历史 + 多空比 + taker 比（1h，各 ~500 条 ≈ 21 天），
        按 timestamp 合并后 upsert 到 sentiment 表。端点不分页，每 symbol 固定 4 个请求。"""
        client = self._futures_client()
        workers = max(1, min(int(workers), 32, len(symbols)))
        total, errors = 0, []
        with self._connection_pool(client), ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="sentiment-sync"
        ) as executor:
            futures = {
                executor.submit(self._fetch_sentiment, client, symbol): symbol
                for symbol in symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    rows = future.result()
                    total += self._store_sentiment(symbol, rows)
                except Exception as exc:
                    self.db.rollback()
                    errors.append(f"{symbol}: {exc}")
        return {"requested": len(symbols), "inserted": total, "errors": errors}

    @staticmethod
    def _fetch_sentiment(client: BinanceFuturesClient, symbol: str) -> list[dict]:
        """四端点各拉 1h×500，按 timestamp 外对齐合并为单行序列（缺端点留 0）。"""
        def to_map(rows: list[dict], key: str) -> dict[int, float]:
            return {int(r["timestamp"]): float(r[key]) for r in rows}

        oi_map = to_map(client.open_interest_history(symbol, "1h", 500), "sum_open_interest")
        oiv_map = to_map(client.open_interest_history(symbol, "1h", 500), "sum_open_interest_value")
        lsr_map = to_map(client.long_short_ratio(symbol, "1h", 500), "long_short_ratio")
        top_map = to_map(client.top_trader_long_short(symbol, "1h", 500), "long_short_ratio")
        taker_map = to_map(client.taker_buy_sell_ratio(symbol, "1h", 500), "buy_sell_ratio")

        stamps = sorted(set(oi_map) | set(oiv_map) | set(lsr_map) | set(top_map) | set(taker_map))
        return [
            {
                "ts": ts,
                "open_interest": oi_map.get(ts, 0.0),
                "oi_value": oiv_map.get(ts, 0.0),
                "long_short_ratio": lsr_map.get(ts, 0.0),
                "top_long_short": top_map.get(ts, 0.0),
                "taker_buy_sell": taker_map.get(ts, 0.0),
            }
            for ts in stamps
        ]

    def _store_sentiment(self, symbol: str, rows: list[dict]) -> int:
        if not rows:
            return 0
        insert_fn = self._insert_fn()
        for start in range(0, len(rows), 500):
            chunk = [{"symbol": symbol, **r} for r in rows[start : start + 500]]
            stmt = insert_fn(Sentiment).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol", "ts"],
                set_={
                    "open_interest": stmt.excluded.open_interest,
                    "oi_value": stmt.excluded.oi_value,
                    "long_short_ratio": stmt.excluded.long_short_ratio,
                    "top_long_short": stmt.excluded.top_long_short,
                    "taker_buy_sell": stmt.excluded.taker_buy_sell,
                },
            )
            self.db.execute(stmt)
        self.db.commit()
        return len(rows)

    def load_sentiment_frame(self, symbol: str, start_ms: int | None = None) -> pd.DataFrame:
        """正交情绪序列：索引为 ts（UTC DatetimeIndex），5 列特征原值。"""
        if self.db is None:
            return pd.DataFrame(columns=["open_interest"])
        query = select(Sentiment).where(Sentiment.symbol == symbol)
        if start_ms:
            query = query.where(Sentiment.ts >= int(start_ms))
        rows = self.db.execute(query.order_by(Sentiment.ts.asc())).scalars().all()
        if not rows:
            return pd.DataFrame(columns=["open_interest"])
        frame = pd.DataFrame(
            [
                {
                    "open_interest": row.open_interest,
                    "oi_value": row.oi_value,
                    "long_short_ratio": row.long_short_ratio,
                    "top_long_short": row.top_long_short,
                    "taker_buy_sell": row.taker_buy_sell,
                }
                for row in rows
            ],
            index=pd.to_datetime([row.ts for row in rows], unit="ms", utc=True),
        )
        frame.index.name = "ts"
        return frame

    def _candle_query(
        self, symbol: str, interval: str,
        start_ms: int | None = None, end_ms: int | None = None,
    ):
        """UTC, left-closed buckets. Completed buckets require every minute.

        The contiguous current bucket is available for chart/mark updates; callers
        already exclude unclosed buckets when calculating signals or backtests.
        Legacy independently downloaded interval rows are deliberately ignored.
        """
        step = interval_ms(interval)
        predicates = [Kline.symbol == symbol, Kline.interval == "1m"]
        if start_ms is not None:
            predicates.append(Kline.open_time >= (start_ms if interval == "1m" else start_ms // step * step))
        if end_ms is not None:
            predicates.append(Kline.open_time <= end_ms)
        if interval == "1m":
            return select(
                Kline.open_time, Kline.close_time, Kline.open, Kline.high, Kline.low,
                Kline.close, Kline.volume, Kline.quote_volume, Kline.trades,
            ).where(*predicates)

        # 用「减余数」对齐桶，而不是 open_time / step * step：
        # SQLite 的 / 对整数是截断，PostgreSQL 遇到 NUMERIC 会转十进制除法，
        # CAST 回 BIGINT 时又是四舍五入（496626.5 -> 496627），桶边界会整体错位。
        # 取模写法不依赖任何一方的除法语义，两侧结果完全一致。
        bucket = (Kline.open_time - Kline.open_time % step).label("open_time")
        grouped = select(
            bucket.label("open_time"),
            func.min(Kline.open_time).label("first_time"),
            func.max(Kline.open_time).label("last_time"),
            func.count().label("minutes"),
            func.max(Kline.high).label("high"),
            func.min(Kline.low).label("low"),
            func.sum(Kline.volume).label("volume"),
            func.sum(Kline.quote_volume).label("quote_volume"),
            func.sum(Kline.trades).label("trades"),
        ).where(*predicates, Kline.open_time % MINUTE_MS == 0).group_by(bucket).subquery()
        first, last = aliased(Kline), aliased(Kline)
        count = grouped.c.minutes
        contiguous = and_(
            grouped.c.first_time == grouped.c.open_time,
            grouped.c.last_time == grouped.c.open_time + (count - 1) * MINUTE_MS,
        )
        available_at = min(end_ms, now_ms()) if end_ms is not None else now_ms()
        query = select(
            grouped.c.open_time, (grouped.c.open_time + step - 1).label("close_time"),
            first.open, grouped.c.high, grouped.c.low,
            last.close, grouped.c.volume, grouped.c.quote_volume, grouped.c.trades,
        ).join(first, and_(first.symbol == symbol, first.interval == "1m",
                           first.open_time == grouped.c.first_time)
        ).join(last, and_(last.symbol == symbol, last.interval == "1m",
                          last.open_time == grouped.c.last_time)
        ).where(contiguous, or_(
            count == step // MINUTE_MS,
            and_(grouped.c.open_time == available_at // step * step,
                 count < step // MINUTE_MS),
        ))
        if start_ms is not None:
            query = query.where(grouped.c.open_time >= start_ms)
        return query

    def _tail_candle_rows(
        self, symbol: str, interval: str, limit: int, end_ms: int | None = None,
    ) -> list:
        """Aggregate only the needed tail; expand for missing or incomplete buckets.

        Limiting after GROUP BY otherwise scans every stored minute on every
        polling cycle. The indexed minute lookup supplies an aligned lower bound;
        sparse history falls back progressively without dropping valid candles.
        """
        minute_query = select(Kline.open_time).where(Kline.symbol == symbol, Kline.interval == "1m")
        if end_ms is not None:
            minute_query = minute_query.where(Kline.open_time <= end_ms)
        latest = self.db.execute(minute_query.order_by(Kline.open_time.desc()).limit(1)).scalar()
        if latest is None:
            return []
        step = interval_ms(interval)
        span = max(1, limit)
        lower = max(0, int(latest) // step * step - (span - 1) * step)
        while True:
            query = self._candle_query(symbol, interval, lower, end_ms).subquery()
            rows = self.db.execute(select(query).order_by(query.c.open_time.desc()).limit(limit)).mappings().all()
            if len(rows) >= limit or lower == 0:
                return rows
            previous = self.db.execute(
                minute_query.where(Kline.open_time < lower).order_by(Kline.open_time.desc()).limit(1)
            ).scalar()
            if previous is None:
                return rows
            span *= 2
            lower = max(0, min(lower - span * step, int(previous) // step * step))

    def _native_candle_query(
        self, symbol: str, interval: str,
        start_ms: int | None = None, end_ms: int | None = None,
    ):
        """已单独存储的同周期行（不经 1m 聚合）。

        末根必须排除：那是「下载那一刻还没走完的周期」的快照，OHLCV 是半截值
        （实测 BNB 1h 末根 volume 1622，而该小时完整值 13332），且永远不会更新。
        """
        step = interval_ms(interval)
        predicates = [Kline.symbol == symbol, Kline.interval == interval]
        if start_ms is not None:
            predicates.append(Kline.open_time >= start_ms // step * step)
        if end_ms is not None:
            predicates.append(Kline.open_time <= end_ms)
        latest = (
            select(func.max(Kline.open_time))
            .where(Kline.symbol == symbol, Kline.interval == interval)
            .scalar_subquery()
        )
        return select(
            Kline.open_time, Kline.close_time, Kline.open, Kline.high, Kline.low,
            Kline.close, Kline.volume, Kline.quote_volume, Kline.trades,
        ).where(*predicates, Kline.open_time < latest)

    def _merged_candle_rows(
        self, symbol: str, interval: str,
        start_ms: int | None = None, end_ms: int | None = None,
        allow_native: bool = True,
    ) -> tuple[list[dict], dict]:
        """聚合优先 + native 补齐。

        重叠桶一律采用聚合结果。
        native 只负责补 1m 根本没覆盖到的历史段，不覆盖任何已有聚合桶。
        """
        query = self._candle_query(symbol, interval, start_ms, end_ms).subquery()
        rows = [
            dict(row)
            for row in self.db.execute(
                select(query).order_by(query.c.open_time.asc())
            ).mappings().all()
        ]
        sources = {"aggregated": len(rows), "native": 0, "overlap": 0, "enabled": False}
        if interval == "1m" or not allow_native or interval not in NATIVE_FALLBACK_INTERVALS:
            return rows, sources
        sources["enabled"] = True
        known = {int(row["open_time"]) for row in rows}
        native = self.db.execute(
            self._native_candle_query(symbol, interval, start_ms, end_ms)
            .order_by(Kline.open_time.asc())
        ).mappings().all()
        for row in native:
            key = int(row["open_time"])
            if key in known:
                sources["overlap"] += 1
                continue
            rows.append(dict(row))
            known.add(key)
            sources["native"] += 1
        rows.sort(key=lambda row: int(row["open_time"]))
        return rows, sources

    def load_frame(
        self, symbol: str, interval: str, limit: int = 3000,
        start_ms: int | None = None, end_ms: int | None = None,
        allow_native: bool | None = None,
    ) -> pd.DataFrame:
        if allow_native is None:
            allow_native = MARKET_NATIVE_FALLBACK
        sources: dict | None = None
        if interval != "1m" and limit and start_ms is None:
            rows = list(reversed(self._tail_candle_rows(symbol, interval, limit, end_ms)))
        elif start_ms is not None and interval != "1m" and allow_native:
            # 只有显式给出 start_ms 的取数（回测/训练）才启用 native 回落；
            # 图表与实盘走尾部路径，不应被历史行影响。
            rows, sources = self._merged_candle_rows(symbol, interval, start_ms, end_ms)
        else:
            query = self._candle_query(symbol, interval, start_ms, end_ms).subquery()
            statement = select(query).order_by(query.c.open_time.asc())
            if limit and start_ms is None:
                statement = select(query).order_by(query.c.open_time.desc()).limit(limit)
                rows = list(reversed(self.db.execute(statement).mappings().all()))
            else:
                rows = self.db.execute(statement).mappings().all()
        if not rows:
            frame = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "quote_volume", "close_time", "trades"])
            frame.attrs["candle_sources"] = sources
            return frame
        frame = pd.DataFrame([dict(row) for row in rows])
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("open_time"), unit="ms", utc=True), name="time")
        frame.attrs["candle_sources"] = sources
        return frame

    def series_stats(self, symbol: str, interval: str) -> dict:
        series = self._candle_query(symbol, interval).subquery()
        count, first, last = self.db.execute(select(
            func.count(), func.min(series.c.open_time), func.max(series.c.open_time),
        ).select_from(series)).one()
        return {"count": count, "first_open_time": int(first or 0), "last_open_time": int(last or 0)}

    def stats(self) -> list[dict]:
        rows = self.db.execute(
            select(
                Kline.symbol,
                Kline.interval,
                func.count(Kline.id),
                func.min(Kline.open_time),
                func.max(Kline.open_time),
            ).where(Kline.interval == "1m").group_by(Kline.symbol, Kline.interval)
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

    def stats_count(self) -> int:
        return int(self.db.execute(
            select(func.count(func.distinct(Kline.symbol))).where(Kline.interval == "1m")
        ).scalar_one())

    def stats_page(self, page: int = 1, page_size: int = 20) -> dict:
        """Paginate the stored minute series in SQL before materializing rows."""
        total = self.stats_count()
        rows = self.db.execute(
            select(
                Kline.symbol, Kline.interval, func.count(Kline.id),
                func.min(Kline.open_time), func.max(Kline.open_time),
            ).where(Kline.interval == "1m")
            .group_by(Kline.symbol, Kline.interval)
            .order_by(Kline.symbol.asc(), Kline.interval.asc())
            .offset((page - 1) * page_size).limit(page_size)
        ).all()
        return {
            "stats": [
                {"symbol": symbol, "interval": interval, "count": count,
                 "first_open_time": int(first or 0), "last_open_time": int(last or 0)}
                for symbol, interval, count, first, last in rows
            ],
            **page_metadata(total, page, page_size),
        }
