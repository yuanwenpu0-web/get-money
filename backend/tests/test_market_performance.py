"""Concurrency and bounded aggregation regressions; no network or production DB."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import httpx
import pandas as pd
import pytest
from sqlalchemy import event

from app.market import data_service
from app.market.binance import BinanceClient
from app.market.data_service import MarketDataService
from app.market.futures import BinanceFuturesClient
from app.models import FundingRate, Kline

START = 1_767_225_600_000
MINUTE = 60_000


def minute_rows(count, start=START):
    return [dict(open_time=start + index * MINUTE, close_time=start + (index + 1) * MINUTE - 1,
                 open=100.0, high=101.0, low=99.0, close=100.5,
                 volume=1.0, quote_volume=100.0, trades=1) for index in range(count)]


class DelayedPages:
    def __init__(self, rows, concurrency=3, failing=None):
        self.rows = rows
        self.barrier = threading.Barrier(concurrency)
        self.lock = threading.Lock()
        self.active = self.peak = self.first_calls = 0
        self.pool_exits = 0
        self.failing = failing

    @contextmanager
    def connection_pool(self):
        try:
            yield self
        finally:
            self.pool_exits += 1

    def klines(self, symbol, interval, start_ms, limit):
        assert interval == "1m"
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.first_calls += 1
            synchronize = self.first_calls <= self.barrier.parties
        try:
            if synchronize:
                self.barrier.wait(timeout=5)
            time.sleep(0.002)
            if symbol == self.failing and start_ms > self.rows[0]["open_time"]:
                raise RuntimeError("page failed")
            selected = [row for row in self.rows if row["open_time"] >= start_ms][:limit]
            # Deliberately return unsorted pages: the next cursor must use the max.
            return [[row["open_time"], row["open"], row["high"], row["low"], row["close"],
                     row["volume"], row["close_time"], row["quote_volume"], row["trades"]]
                    for row in reversed(selected)]
        finally:
            with self.lock:
                self.active -= 1


def test_batch_sync_bounds_concurrency_serializes_writes_and_isolates_page_errors(db, monkeypatch):
    monkeypatch.setattr(data_service, "KLINE_PAGE_SIZE", 2)
    monkeypatch.setattr(data_service, "now_ms", lambda: START + 5 * MINUTE)
    client = DelayedPages(minute_rows(5), failing="BAD")
    service = MarketDataService(db, client=client)
    db_threads = set()
    event.listen(db.get_bind(), "before_cursor_execute", lambda *args: db_threads.add(threading.get_ident()))
    result = service.sync_many(["BTC", "BAD", "ETH", "SOL", "XRP", "DOGE"], days=1, workers=3)
    assert result == {"requested": 6, "inserted": 27, "errors": ["BAD: page failed"]}
    assert db.query(Kline).count() == 27
    assert db_threads == {threading.get_ident()}
    assert client.peak == 3
    assert client.pool_exits == 1
    assert [row.open_time for row in db.query(Kline).filter(Kline.symbol == "BTC")
            .order_by(Kline.open_time)] == [START + index * MINUTE for index in range(5)]


def test_batch_sync_rolls_back_failed_write_before_continuing(db, monkeypatch):
    monkeypatch.setattr(data_service, "now_ms", lambda: START + MINUTE)
    client = DelayedPages(minute_rows(1), concurrency=1)
    service = MarketDataService(db, client=client)
    original_store = service.store

    def store(symbol, interval, rows):
        if symbol == "BAD":
            db.add(Kline(symbol=symbol, interval="1m", **rows[0] | {"close": None}))
            db.flush()  # Invalidate the transaction, so a rollback is required.
        return original_store(symbol, interval, rows)

    monkeypatch.setattr(service, "store", store)
    result = service.sync_many(["BAD", "GOOD"], workers=1)
    assert result["inserted"] == 1
    assert len(result["errors"]) == 1
    assert db.query(Kline).one().symbol == "GOOD"
    assert client.pool_exits == 1


def test_empty_batch_does_not_fetch(db):
    client = DelayedPages([], concurrency=1)
    assert MarketDataService(db, client).sync_many([], workers=0) == {
        "requested": 0, "inserted": 0, "errors": [],
    }
    assert client.first_calls == 0


def test_recent_batch_prefetches_raw_cursors_and_refreshes_each_tail(db, monkeypatch):
    monkeypatch.setattr(data_service, "KLINE_PAGE_SIZE", 2)
    monkeypatch.setattr(data_service, "now_ms", lambda: START + 24 * MINUTE + 30_000)
    client = DelayedPages(minute_rows(25), concurrency=2)
    service = MarketDataService(db, client)
    service.store("EXISTING", "1m", minute_rows(2))
    cursors = {}
    fetch = client.klines

    def klines(symbol, interval, start_ms, limit):
        with client.lock:
            cursors.setdefault(symbol, start_ms)
        return fetch(symbol, interval, start_ms, limit)

    monkeypatch.setattr(client, "klines", klines)
    raw_lookup = service.latest_open_time

    def latest(symbol, interval):
        assert interval == "1m"
        assert threading.current_thread() is threading.main_thread()
        return raw_lookup(symbol, interval)

    monkeypatch.setattr(service, "latest_open_time", latest)
    result = service.sync_recent_many(["EXISTING", "NEW"], "5m", tail_limit=5, workers=2)
    assert result == {"requested": 2, "inserted": 49, "errors": []}
    assert cursors == {"EXISTING": START + MINUTE, "NEW": START - 5 * MINUTE}
    assert db.query(Kline).count() == 50
    assert client.peak == 2


def test_funding_batch_uses_network_workers_and_callers_session(db, monkeypatch):
    monkeypatch.setattr(data_service, "FUNDING_PAGE", 2)
    monkeypatch.setattr(data_service, "now_ms", lambda: START + 24 * 3_600_000)
    barrier = threading.Barrier(2)
    calls = []
    lock = threading.Lock()

    class FundingClient(BinanceFuturesClient):
        def funding_rate(self, symbol, start_ms, end_ms, limit):
            with lock:
                calls.append(threading.get_ident())
                first = len(calls) <= 2
            if first:
                barrier.wait(timeout=5)
            rows = [dict(funding_time=START + index * 8 * 3_600_000, funding_rate=0.0001,
                         mark_price=100.0, rate_type="Regular") for index in range(3)]
            return list(reversed([row for row in rows if row["funding_time"] >= start_ms][:limit]))

    result = MarketDataService(db, FundingClient()).sync_funding_many(["BTC", "ETH"], workers=2)
    assert result == {"requested": 2, "inserted": 6, "errors": []}
    assert db.query(FundingRate).count() == 6
    assert threading.get_ident() not in calls


@pytest.mark.parametrize("client_type", [BinanceClient, BinanceFuturesClient])
def test_scoped_connection_reuse_is_thread_safe_and_closes_on_failure(monkeypatch, client_type):
    opened, closed = [], []

    class FakeHttpClient:
        def __init__(self, **kwargs):
            opened.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(self)

        def request(self, *args, **kwargs):
            return httpx.Response(200, json={"serverTime": START})

    monkeypatch.setattr(httpx, "Client", FakeHttpClient)
    client = client_type()
    with pytest.raises(RuntimeError, match="scope failed"):
        with client.connection_pool():
            with client.connection_pool(), ThreadPoolExecutor(max_workers=4) as pool:
                assert list(pool.map(lambda _: client.server_time(), range(12))) == [START] * 12
            assert len(opened) == 1 and not closed
            raise RuntimeError("scope failed")
    assert len(closed) == 1
    assert client.server_time() == START
    assert len(opened) == len(closed) == 2


@pytest.mark.parametrize("end_offset", [None, 7, 37, 600])
@pytest.mark.parametrize("limit", [1, 2, 7, 100])
def test_bounded_aggregate_matches_full_history_with_gaps(db, monkeypatch, end_offset, limit):
    monkeypatch.setattr(data_service, "now_ms", lambda: START + 610 * MINUTE)
    service = MarketDataService(db)
    # A large missing span, an incomplete old bucket, and an incomplete final
    # bucket force expansion beyond the initial requested number of periods.
    rows = minute_rows(40) + minute_rows(9, START + 600 * MINUTE)
    rows = [row for row in rows if row["open_time"] != START + 32 * MINUTE]
    service.store("BTC", "1m", rows)
    end = None if end_offset is None else START + end_offset * MINUTE
    expected = service.load_frame("BTC", "5m", limit=0, end_ms=end).tail(limit)
    actual = service.load_frame("BTC", "5m", limit=limit, end_ms=end)
    pd.testing.assert_frame_equal(actual, expected)
    if end is None:
        assert service.latest_open_time("BTC", "5m") == int(expected.index[-1].timestamp() * 1000)


def test_complete_tail_queries_have_a_bounded_minute_range(db, monkeypatch):
    service = MarketDataService(db)
    service.store("BTC", "1m", minute_rows(300))
    original = service._candle_query
    lower_bounds = []

    def query(symbol, interval, start_ms=None, end_ms=None):
        lower_bounds.append(start_ms)
        return original(symbol, interval, start_ms, end_ms)

    monkeypatch.setattr(service, "_candle_query", query)
    assert len(service.load_frame("BTC", "5m", limit=2)) == 2
    assert service.latest_open_time("BTC", "5m") == START + 295 * MINUTE
    assert lower_bounds == [START + 290 * MINUTE, START + 295 * MINUTE]
