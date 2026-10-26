import pandas as pd
import pytest
from fastapi import HTTPException
from sqlalchemy import event

from app.api.market import klines
from app.market.data_service import MarketDataService
from app.models import Order
from app.trading.portfolio import portfolio_snapshot


def _bars():
    return pd.DataFrame(
        {"open": [100.0], "high": [103.0], "low": [99.0], "close": [102.0], "volume": [7.0]},
        index=pd.DatetimeIndex(["2026-01-01T00:00:00Z"]),
    )


def test_klines_with_local_data_does_not_scan_for_latest_or_sync(db, monkeypatch):
    calls = []

    def load(self, symbol, interval, limit):
        calls.append((symbol, interval, limit))
        return _bars()

    def unnecessary(*args, **kwargs):
        raise AssertionError("Local data should only be loaded once")

    monkeypatch.setattr(MarketDataService, "load_frame", load)
    monkeypatch.setattr(MarketDataService, "latest_open_time", unnecessary)
    monkeypatch.setattr(MarketDataService, "sync_recent", unnecessary)
    result = klines("BTCUSDT", "1h", 300, db)
    assert calls == [("BTCUSDT", "1h", 300)]
    assert result["bars"] == [
        {"time": 1767225600000, "open": 100.0, "high": 103.0, "low": 99.0, "close": 102.0, "volume": 7.0}
    ]


def test_klines_loads_again_after_syncing_missing_data(db, monkeypatch):
    frames = iter([pd.DataFrame(), _bars()])
    calls = []
    monkeypatch.setattr(MarketDataService, "load_frame", lambda *args, **kwargs: next(frames))
    monkeypatch.setattr(
        MarketDataService, "sync_recent",
        lambda self, symbol, interval, tail_limit: calls.append((symbol, interval, tail_limit)),
    )
    assert len(klines("BTCUSDT", "1h", 300, db)["bars"]) == 1
    assert calls == [("BTCUSDT", "1h", 1000)]


def test_klines_preserves_sync_failure_response(db, monkeypatch):
    monkeypatch.setattr(MarketDataService, "load_frame", lambda *args, **kwargs: pd.DataFrame())

    def fail(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(MarketDataService, "sync_recent", fail)
    with pytest.raises(HTTPException) as exc:
        klines("BTCUSDT", "1h", 300, db)
    assert exc.value.status_code == 502
    assert "offline" in exc.value.detail


def test_portfolio_aggregates_closed_orders_without_loading_their_objects(db):
    for status, pnl in [("CLOSED", 10.0), ("CLOSED", -3.0), ("CLOSED", 0.0), ("CANCELED", 999.0)]:
        db.add(Order(symbol="BTCUSDT", interval="1h", side="LONG", qty=1.0,
                     entry_price=100.0, status=status, realized_pnl=pnl))
    db.add(Order(symbol="ETHUSDT", interval="1h", side="LONG", qty=2.0,
                 entry_price=100.0, mark_price=110.0, margin=100.0, status="OPEN"))
    db.commit()
    db.expunge_all()
    loaded_statuses = []

    def record_load(order, context):
        loaded_statuses.append(order.status)

    event.listen(Order, "load", record_load)
    try:
        snapshot = portfolio_snapshot(db, update_marks=False)
    finally:
        event.remove(Order, "load", record_load)
    assert snapshot["closed_trades"] == 3
    assert snapshot["realized_pnl"] == 7.0
    assert snapshot["win_rate"] == 33.33
    assert snapshot["unrealized_pnl"] == 20.0
    assert snapshot["equity"] == 10120.0
    assert snapshot["open_positions"] == 1
    assert loaded_statuses == ["OPEN"]


def test_portfolio_without_closed_orders_has_zero_statistics(db):
    snapshot = portfolio_snapshot(db, update_marks=False)
    assert snapshot["closed_trades"] == 0
    assert snapshot["realized_pnl"] == 0.0
    assert snapshot["win_rate"] == 0.0
