from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import HTTPException

from app.api.local_strategies import LocalStrategyUpdate, local_strategies, update_local_strategy
from app.models import Order, StrategyConfig
from app.state import set_state
from app.strategy import local_filters
from app.strategy.local_filters import EntryFilters
from app.trading import service as trading_service


def test_defaults_and_configuration(db):
    rows = local_strategies(db)["strategies"]
    assert all(not row["enabled"] and row["bound_strategies"] == [] for row in rows)
    result = update_local_strategy("sma_crossover", LocalStrategyUpdate(
        enabled=True, bound_strategies=["trend_follow"], params={"fast_length": 2, "slow_length": 3}), db)
    assert result["enabled"] and result["bound_strategies"] == ["trend_follow"]
    assert result["source"].endswith("sma-crossover/strategy.pine")
    with pytest.raises(HTTPException):
        update_local_strategy("sma_crossover", LocalStrategyUpdate(bound_strategies=["unknown"]), db)
    with pytest.raises(HTTPException):
        update_local_strategy("sma_crossover", LocalStrategyUpdate(params={"slow_length": 1001}), db)


def _frame(times):
    prices = [3., 2., 1., 2., 3.]
    return pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices, "volume": [1.] * 5},
                        index=pd.to_datetime(times, unit="ms", utc=True))


def test_filter_requires_same_complete_contiguous_candle_and_direction(db, monkeypatch):
    update_local_strategy("sma_crossover", LocalStrategyUpdate(
        enabled=True, bound_strategies=["trend_follow"], params={"fast_length": 2, "slow_length": 3}), db)
    now = 600_000
    monkeypatch.setattr(local_filters, "now_ms", lambda: now)
    frame = _frame([0, 60_000, 120_000, 180_000, 240_000])
    monkeypatch.setattr(local_filters.MarketDataService, "load_frame", lambda *args, **kwargs: frame)
    signal = SimpleNamespace(symbol="AAAUSDT", interval="1m", ts=240_000,
                             strategy="trend_follow", direction="LONG")
    assert EntryFilters(db).allows(signal)
    signal.direction = "SHORT"
    assert not EntryFilters(db).allows(signal)
    signal.direction = "LONG"
    signal.ts = 180_000
    assert not EntryFilters(db).allows(signal)
    signal.ts = 240_000
    frame = _frame([0, 60_000, 120_000, 210_000, 240_000])
    assert not EntryFilters(db).allows(signal)
    signal.strategy = "mean_reversion"
    assert EntryFilters(db).allows(signal)


def test_invalid_bars_restart_history_and_invalid_latest_fails_closed(db, monkeypatch):
    update_local_strategy("sma_crossover", LocalStrategyUpdate(
        enabled=True, bound_strategies=["trend_follow"], params={"fast_length": 2, "slow_length": 3}), db)
    # An invalid old bar splits history; the following five valid bars warm up again.
    frame = pd.concat([_frame([0, 60_000, 120_000, 180_000, 240_000]),
                       _frame([300_000, 360_000, 420_000, 480_000, 540_000])])
    frame.iloc[4, frame.columns.get_loc("volume")] = -1
    monkeypatch.setattr(local_filters, "now_ms", lambda: 660_000)
    monkeypatch.setattr(local_filters.MarketDataService, "load_frame", lambda *args, **kwargs: frame)
    signal = SimpleNamespace(symbol="AAAUSDT", interval="1m", ts=540_000,
                             strategy="trend_follow", direction="LONG")
    assert EntryFilters(db).allows(signal)
    frame.iloc[-1, frame.columns.get_loc("high")] = 0
    assert not EntryFilters(db).allows(signal)
    signal.ts = 480_000
    assert not EntryFilters(db).allows(signal)


def test_all_enabled_bindings_must_agree(db, monkeypatch):
    for code in ("sma_crossover", "donchian_channel"):
        update_local_strategy(code, LocalStrategyUpdate(enabled=True, bound_strategies=["trend_follow"]), db)
    frame = _frame([0, 60_000, 120_000, 180_000, 240_000])
    monkeypatch.setattr(local_filters, "now_ms", lambda: 600_000)
    monkeypatch.setattr(local_filters.MarketDataService, "load_frame", lambda *args, **kwargs: frame)
    monkeypatch.setattr(local_filters, "tradingview_signals", lambda candles, code, params: [{"long": code == "sma_crossover", "short": False}])
    signal = SimpleNamespace(symbol="AAAUSDT", interval="1m", ts=240_000,
                             strategy="trend_follow", direction="LONG")
    assert not EntryFilters(db).allows(signal)
    update_local_strategy("donchian_channel", LocalStrategyUpdate(enabled=False), db)
    assert EntryFilters(db).allows(signal)


def test_rejected_entry_does_not_hide_signal_from_order_manager(db, monkeypatch):
    set_state(db, "symbol_scope", "custom")
    set_state(db, "symbols", ["AAAUSDT"])
    set_state(db, "interval", "1m")
    set_state(db, "use_coin_route", False)
    db.add(StrategyConfig(code="trend_follow", name="trend", enabled=1, params="{}"))
    db.add(Order(symbol="AAAUSDT", interval="1m", side="LONG", qty=1,
                 entry_price=100, mark_price=100, margin=10, strategy="trend_follow", status="OPEN"))
    db.commit()
    signal = SimpleNamespace(direction="SHORT", interval="1m", strategy="trend_follow", score=1)
    monkeypatch.setattr(trading_service, "StrategyEngine", lambda *a, **k: SimpleNamespace(evaluate=lambda *a, **k: signal))
    monkeypatch.setattr(trading_service, "EntryFilters", lambda db: SimpleNamespace(allows=lambda signal: False))
    monkeypatch.setattr(trading_service, "_sync_recent_many", lambda *a, **k: [])
    seen = []
    monkeypatch.setattr(trading_service, "manage_order", lambda db, order, binding: seen.append(binding) or {"order_id": order.id, "actions": []})
    monkeypatch.setattr(trading_service, "snapshot_equity", lambda db: None)
    result = trading_service.TradingService(db).run_cycle()
    assert result["ok"] and seen == [signal] and result["opened"] == []
