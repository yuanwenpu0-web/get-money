"""K线向左翻页：end 参数只返回更早的K线，翻页也不会触发远端拉取。"""

import pandas as pd
import pytest

from app.api import market
from app.market.data_service import MarketDataService


START = int(pd.Timestamp("2026-01-01T00:00:00Z").timestamp() * 1000)
MINUTE = 60000
BUCKET_15M = 15 * MINUTE


def seed_minutes(db, count=180):
    rows = [
        dict(
            open_time=START + index * MINUTE,
            close_time=START + (index + 1) * MINUTE - 1,
            open=100 + index,
            high=101 + index,
            low=99 + index,
            close=100.5 + index,
            volume=1 + index,
            quote_volume=2 + index,
            trades=1,
        )
        for index in range(count)
    ]
    MarketDataService(db).store("BTCUSDT", "1m", rows)
    return count


def test_tail_page_returns_latest_bars(db):
    seed_minutes(db)
    page = market.klines(symbol="BTCUSDT", interval="15m", limit=6, end=None, db=db)
    times = [bar["time"] for bar in page["bars"]]
    assert len(times) == 6
    assert times == sorted(times)
    # 12 个 15m 桶，首页就是最新的 6 个
    assert times[-1] == START + 11 * BUCKET_15M


def test_end_cursor_walks_backwards_and_stops_at_the_beginning(db):
    seed_minutes(db)
    latest = market.klines(symbol="BTCUSDT", interval="15m", limit=6, end=None, db=db)
    older = market.klines(
        symbol="BTCUSDT", interval="15m", limit=6, end=latest["bars"][0]["time"] - 1, db=db
    )
    older_times = [bar["time"] for bar in older["bars"]]
    latest_times = [bar["time"] for bar in latest["bars"]]
    assert len(older_times) == 6
    # 两页不重叠，且翻页后的时间整体更早
    assert max(older_times) < min(latest_times)
    assert older_times + latest_times == sorted(older_times + latest_times)

    empty = market.klines(
        symbol="BTCUSDT", interval="15m", limit=6, end=older_times[0] - 1, db=db
    )
    assert empty["bars"] == []


def test_paging_request_never_pulls_from_the_exchange(db, monkeypatch):
    seed_minutes(db)

    def boom(*args, **kwargs):
        raise AssertionError("翻页不应该触发远端同步")

    monkeypatch.setattr(MarketDataService, "sync_recent", boom)
    # end 早于本地最早数据 → 没有更早的历史，直接返回空
    result = market.klines(symbol="BTCUSDT", interval="15m", limit=6, end=START - 1, db=db)
    assert result["bars"] == []
    assert result["end"] == START - 1


def test_first_page_still_syncs_when_local_history_is_missing(db, monkeypatch):
    calls = []

    def fake_sync(self, symbol, interval, tail_limit=1000):
        calls.append((symbol, interval, tail_limit))

    monkeypatch.setattr(MarketDataService, "sync_recent", fake_sync)
    result = market.klines(symbol="ETHUSDT", interval="1h", limit=10, end=None, db=db)
    assert result["bars"] == []
    assert calls == [("ETHUSDT", "1h", 1000)]
