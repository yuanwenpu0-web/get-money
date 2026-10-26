from types import SimpleNamespace

import pandas as pd
import pytest

from app.market import data_service
from app.market.data_service import MarketDataService
from app.models import Kline
from app.utils import interval_ms


START = int(pd.Timestamp("2026-01-01T00:00:00Z").timestamp() * 1000)
MINUTE = 60000


def minutes(count, start=START):
    return [dict(open_time=start + i * MINUTE, close_time=start + (i + 1) * MINUTE - 1,
                 open=100 + i, high=103 + i, low=99 + i, close=101 + i,
                 volume=i + 1, quote_volume=10 * (i + 1), trades=i + 2)
            for i in range(count)]


def store(db, rows):
    service = MarketDataService(db)
    service.store("BTCUSDT", "1m", rows)
    return service


@pytest.mark.parametrize("interval", ["5m", "15m", "1h", "4h", "1d"])
def test_supported_intervals_aggregate_from_minutes(db, interval):
    count = interval_ms(interval) // MINUTE
    service = store(db, minutes(count))
    frame = service.load_frame("BTCUSDT", interval)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert int(frame.index[0].timestamp() * 1000) == START
    assert row.open == 100
    assert row.high == 102 + count
    assert row.low == 99
    assert row.close == 100 + count
    assert row.volume == count * (count + 1) / 2
    assert row.quote_volume == 10 * row.volume
    assert row.trades == count * (count + 3) / 2
    assert row.close_time == START + interval_ms(interval) - 1
    assert service.latest_open_time("BTCUSDT", interval) == START
    assert service.series_stats("BTCUSDT", interval) == {
        "count": 1, "first_open_time": START, "last_open_time": START,
    }


def test_legacy_higher_period_rows_are_not_used(db):
    db.add(Kline(symbol="BTCUSDT", interval="5m", **minutes(1)[0]))
    db.commit()
    service = MarketDataService(db)
    assert service.load_frame("BTCUSDT", "5m").empty
    assert service.latest_open_time("BTCUSDT", "5m") == 0
    assert service.stats() == []
    store(db, minutes(5))
    assert service.load_frame("BTCUSDT", "5m").iloc[0].close == 105
    with pytest.raises(ValueError, match="1m"):
        service.store("BTCUSDT", "5m", minutes(1))


@pytest.mark.parametrize("missing", [0, 2, 4])
def test_incomplete_finished_bucket_is_omitted(db, missing):
    rows = minutes(10)
    del rows[missing]
    service = store(db, rows)
    frame = service.load_frame("BTCUSDT", "5m")
    assert list(frame.index.as_unit("ms").asi8) == [START + 5 * MINUTE]


def test_partial_current_bucket_is_live_but_excluded_after_it_ends(db, monkeypatch):
    now = START + 7 * MINUTE + 30000
    monkeypatch.setattr(data_service, "now_ms", lambda: now)
    service = store(db, minutes(8))
    frame = service.load_frame("BTCUSDT", "5m")
    assert len(frame) == 2
    closed = frame.loc[frame.index.as_unit("ms").asi8 + 5 * MINUTE <= now]
    assert len(closed) == 1
    # The tail must not become a complete historical candle merely with time.
    monkeypatch.setattr(data_service, "now_ms", lambda: START + 10 * MINUTE)
    assert len(service.load_frame("BTCUSDT", "5m")) == 1


def test_asof_ranges_limits_and_minute_corrections(db):
    service = store(db, minutes(15))
    frame = service.load_frame("BTCUSDT", "5m", limit=2)
    assert list(frame.index.as_unit("ms").asi8) == [START + 5 * MINUTE, START + 10 * MINUTE]
    frame = service.load_frame("BTCUSDT", "5m", start_ms=START + MINUTE, end_ms=START + 9 * MINUTE)
    assert list(frame.index.as_unit("ms").asi8) == [START + 5 * MINUTE]
    asof = service.load_frame("BTCUSDT", "5m", end_ms=START + 7 * MINUTE)
    assert asof.iloc[-1].close == 108  # No minute later than the as-of boundary.
    corrected = minutes(5)[-1] | {"high": 200, "close": 199, "volume": 100}
    service.store("BTCUSDT", "1m", [corrected])
    updated = service.load_frame("BTCUSDT", "5m").iloc[0]
    assert (updated.high, updated.close, updated.volume) == (200, 199, 110)


def test_bucket_alignment_is_utc_even_when_client_is_in_china(db):
    start = START + 23 * 60 * MINUTE
    service = store(db, minutes(120, start=start))
    frame = service.load_frame("BTCUSDT", "1h")
    assert [ts.hour for ts in frame.index] == [23, 0]
    assert str(frame.index.tz) == "UTC"
    assert service.load_frame("BTCUSDT", "1d").empty


@pytest.mark.parametrize("method", ["sync_recent", "sync_history"])
def test_sync_fetches_only_minutes_and_paginates_past_ten_pages(db, monkeypatch, method):
    rows = minutes(25)
    calls = []
    monkeypatch.setattr(data_service, "KLINE_PAGE_SIZE", 2)
    monkeypatch.setattr(data_service, "now_ms", lambda: START + 24 * MINUTE + 30000)

    def klines(symbol, interval, start_ms, limit):
        calls.append((interval, start_ms))
        return [[r["open_time"], r["open"], r["high"], r["low"], r["close"],
                 r["volume"], r["close_time"], r["quote_volume"], r["trades"]]
                for r in rows if r["open_time"] >= start_ms][:limit]

    service = MarketDataService(db, client=SimpleNamespace(klines=klines))
    if method == "sync_recent":
        result = service.sync_recent("BTCUSDT", "5m", tail_limit=5)
    else:
        result = service.sync_history("BTCUSDT", "5m", days=1)
    assert len(calls) == 13
    assert all(interval == "1m" for interval, _ in calls)
    assert result["inserted"] == 25
    assert result["source_interval"] == "1m"
    assert len(service.load_frame("BTCUSDT", "5m")) == 5
    assert {row.interval for row in db.query(Kline).all()} == {"1m"}
