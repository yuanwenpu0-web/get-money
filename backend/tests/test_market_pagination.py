"""Bounded market group lists and lightweight settings payloads."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import event

from app.api import market, settings
from app.db import get_db
from app.market.data_service import MarketDataService
from app.models import Kline


def seed_series(db, count=25):
    start = 1_767_225_600_000
    rows = []
    for index in range(count):
        for minute in range(2):
            rows.append(Kline(
                symbol=f"S{index:03d}USDT", interval="1m", open_time=start + minute * 60_000,
                close_time=start + (minute + 1) * 60_000 - 1,
                open=100, high=101, low=99, close=100, volume=1, quote_volume=100, trades=1,
            ))
    rows.append(Kline(
        symbol="LEGACYUSDT", interval="1h", open_time=start, close_time=start + 3_599_999,
        open=100, high=101, low=99, close=100, volume=1, quote_volume=100, trades=1,
    ))
    db.add_all(rows)
    db.commit()


def test_market_series_pages_are_bounded_stable_and_exclude_legacy_intervals(db):
    seed_series(db)
    service = MarketDataService(db)
    queries = []

    def record_query(conn, cursor, statement, parameters, context, executemany):
        queries.append(statement)

    event.listen(db.get_bind(), "before_cursor_execute", record_query)
    try:
        first = service.stats_page(1, 20)
        second = service.stats_page(2, 20)
        empty = service.stats_page(3, 20)
    finally:
        event.remove(db.get_bind(), "before_cursor_execute", record_query)
    assert (first["total"], first["pages"], first["page_size"]) == (25, 2, 20)
    assert len(first["stats"]) == 20
    assert len(second["stats"]) == 5
    assert empty["stats"] == [] and empty["total"] == 25
    symbols = [row["symbol"] for row in first["stats"] + second["stats"]]
    assert symbols == [f"S{index:03d}USDT" for index in range(25)]
    assert all(row["count"] == 2 and row["interval"] == "1m" for row in first["stats"])
    group_queries = [query for query in queries if "GROUP BY" in query]
    assert len(group_queries) == 3
    assert all("LIMIT" in query and "OFFSET" in query for query in group_queries)


def test_market_sources_and_stats_share_page_contract(db):
    seed_series(db)
    stats = market.market_stats(db=db, page=2, page_size=10)
    sources = market.market_sources(db=db, page=2, page_size=10)
    assert sources["local"] == stats
    assert stats["total"] == 25 and stats["pages"] == 3
    assert stats["stats"][0]["symbol"] == "S010USDT"


def test_empty_market_groups_have_zero_pages(db):
    assert MarketDataService(db).stats_page() == {
        "stats": [], "total": 0, "page": 1, "page_size": 20, "pages": 0,
    }


def test_settings_can_return_only_group_count_without_loading_all_stats(db, monkeypatch):
    seed_series(db)

    def unnecessary(*args, **kwargs):
        raise AssertionError("lightweight settings must not materialize all stats")

    monkeypatch.setattr(MarketDataService, "stats", unnecessary)
    result = settings.read_settings(db, include_stats=False)
    assert result["data_stats"] == []
    assert result["data_stats_count"] == 25


def test_settings_default_keeps_legacy_stats_payload(db):
    seed_series(db, count=2)
    result = settings.read_settings(db)
    assert len(result["data_stats"]) == result["data_stats_count"] == 2


@pytest.mark.parametrize("path", ["/market/stats", "/market/sources"])
@pytest.mark.parametrize("params", [{"page": 0}, {"page_size": 0}, {"page_size": 101}, {"page": "bad"}])
def test_market_page_query_validation(db, path, params):
    app = FastAPI()
    app.include_router(market.router)
    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as client:
        assert client.get(path, params=params).status_code == 422
