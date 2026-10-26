import pytest
from fastapi import HTTPException

from app.api.market import sync_market
from app.api.settings import _validate
from app.market.data_service import MarketDataService
from app.schemas import SyncRequest
from app.state import all_state, set_state


def test_existing_source_setting_uses_binance(db):
    set_state(db, "data_source", "legacy_database")
    assert all_state(db)["data_source"] == "binance"


def test_sync_defaults_to_binance(db, monkeypatch):
    calls = []

    def sync_history(self, symbol, interval, days):
        calls.append((symbol, interval, days))
        return {"symbol": symbol, "interval": interval, "inserted": 10}

    monkeypatch.setattr(MarketDataService, "sync_history", sync_history)
    result = sync_market(SyncRequest(symbol="btcusdt"), db)
    assert calls == [("BTCUSDT", "1m", 30)]
    assert result["inserted"] == 10


@pytest.mark.parametrize("days", [0, -1, 731])
def test_sync_rejects_invalid_days(db, days):
    with pytest.raises(HTTPException) as exc:
        sync_market(SyncRequest(symbol="BTCUSDT", days=days), db)
    assert exc.value.status_code == 400


def test_sync_accepts_long_history_window(db, monkeypatch):
    # 400 天拉长复验窗口必须可用（内部按 1000 根/页分页）
    calls = []

    def sync_history(self, symbol, interval, days):
        calls.append((symbol, interval, days))
        return {"symbol": symbol, "interval": interval, "inserted": 9600}

    monkeypatch.setattr(MarketDataService, "sync_history", sync_history)
    result = sync_market(SyncRequest(symbol="BTCUSDT", days=400), db)
    assert calls == [("BTCUSDT", "1m", 400)]
    assert result["inserted"] == 9600


def test_removed_source_is_rejected(db):
    with pytest.raises(HTTPException):
        sync_market(SyncRequest(symbol="BTCUSDT", source="legacy_database"), db)
    with pytest.raises(HTTPException):
        _validate({"data_source": "legacy_database"})
