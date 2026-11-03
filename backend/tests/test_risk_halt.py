"""旧账户熔断状态不再影响交易，其他入场风控仍然有效。"""

from datetime import timedelta
from types import SimpleNamespace

import pytest

from app.api.settings import read_settings, update_settings
from app.models import JobLog, Order, utcnow
from app.schemas import SettingsUpdate
from app.state import all_state, set_state
from app.trading import dispatch, paper
from app.trading.service import TradingService
from app.utils import interval_ms, now_ms
from fakes import seed_binding


def signal(**overrides):
    values = dict(symbol="BTCUSDT", interval="1m", direction="LONG", price=100.0,
                  sl=99.0, tp=103.0, leverage=10, position_pct=0.25,
                  strategy="absorption_reversal", score=1.0, prob_up=0.8,
                  ts=now_ms() - interval_ms("1m"))
    return SimpleNamespace(**(values | overrides))


def open_order(**overrides):
    values = dict(symbol="BTCUSDT", interval="1m", side="LONG", qty=1.0,
                  entry_price=100.0, mark_price=100.0, margin=0.0, leverage=1,
                  sl=99.0, tp=103.0, status="OPEN", source="paper",
                  strategy="absorption_reversal")
    return Order(**(values | overrides))


@pytest.mark.parametrize("case", ["drawdown", "losses"])
def test_close_does_not_trigger_legacy_halt(db, case):
    set_state(db, "initial_capital", 10000.0)
    set_state(db, "total_capital", 10000.0)
    set_state(db, "cash", 500.0 if case == "drawdown" else 9900.0)
    set_state(db, "max_drawdown_halt_pct", 0.1 if case == "drawdown" else 0)
    set_state(db, "consecutive_loss_halt", 3 if case == "losses" else 0)
    if case == "losses":
        for index in range(3):
            order = open_order(status="CLOSED", realized_pnl=-10.0)
            order.closed_at = utcnow() - timedelta(minutes=5 - index)
            db.add(order)
    order = open_order()
    db.add(order)
    db.commit()
    paper.close_position(db, order, 99.0, "stop_loss")
    assert not all_state(db).get("risk_halt", False)
    assert db.query(JobLog).filter(JobLog.job == "risk_halt").count() == 0
    assert dispatch.open_position(db, signal(symbol="ETHUSDT")) is not None


def test_legacy_halt_ignored_and_other_entry_gates_remain(db):
    set_state(db, "risk_halt", True)
    set_state(db, "risk_halt_reason", "旧回撤熔断")
    assert dispatch.open_position(db, signal(direction="INVALID")) is None
    order = dispatch.open_position(db, signal())
    assert order is not None and order.source == "paper"
    assert dispatch.open_position(db, signal()) is None
    assert all_state(db)["risk_halt"] is True


def test_legacy_settings_are_ignored_without_database_migration(db):
    set_state(db, "risk_halt", True)
    set_state(db, "max_drawdown_halt_pct", 999)
    result = update_settings(SettingsUpdate(risk_halt=True, consecutive_loss_halt=1, max_leverage=7), db)
    for key in ("risk_halt", "risk_halt_reason", "max_drawdown_halt_pct", "consecutive_loss_halt"):
        assert key not in result
    assert result["max_leverage"] == 7
    assert all_state(db)["risk_halt"] is True
    assert all_state(db)["max_drawdown_halt_pct"] == 999
    assert "risk_halt" not in read_settings(db, include_stats=False)


def test_run_cycle_ignores_legacy_halt(db, monkeypatch):
    from app.trading import service as trading_service

    set_state(db, "symbol_scope", "custom")
    set_state(db, "symbols", ["BTCUSDT"])
    set_state(db, "risk_halt", True)
    set_state(db, "risk_halt_reason", "旧熔断")
    seed_binding(db, ["BTCUSDT"])

    class StubService:
        def __init__(self, db, client=None):
            self.db = db

        def sync_recent_many(self, symbols, interval, **kwargs):
            return {"errors": []}

    live_signal = signal(to_dict=lambda: {"symbol": "BTCUSDT"})
    monkeypatch.setattr(trading_service, "MarketDataService", StubService)
    monkeypatch.setattr(trading_service, "StrategyEngine", lambda db, **kwargs:
                        type("Stub", (), {"evaluate": lambda self, *a, **k: live_signal})())
    summary = TradingService(db).run_cycle()
    assert summary["ok"] is True
    assert len(summary["opened"]) == 1
    assert db.query(Order).filter_by(status="OPEN").count() == 1
