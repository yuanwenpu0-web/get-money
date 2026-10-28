"""K线源同源与周期日志分级/摘要测试。全部 mock 网络层，不真实请求。"""

from datetime import datetime, timezone

import pytest

from app.config import FUTURES_MAIN_HOSTS
from app.market.data_service import default_market_client
from app.market.futures import BinanceFuturesClient
from app.models import JobLog, Order
from app.state import set_state
from app.trading import service as trading_service
from app.trading.service import TradingService

from fakes import seed_binding


class _FakeSession:
    def close(self):
        pass


def _stub_cycle(monkeypatch, fail_for=(), evaluate=None):
    """把 run_cycle 的网络/引擎依赖换成桩，只保留 scope 解析与日志行为。"""
    monkeypatch.setattr(trading_service, "SessionLocal", lambda: _FakeSession())

    class _StubService:
        def __init__(self, db, client=None):
            self.db = db

        def sync_recent_many(self, symbols, interval, **kwargs):
            return {"errors": [f"{symbol}: mock sync failure {symbol}"
                               for symbol in symbols if symbol in fail_for]}

    monkeypatch.setattr(trading_service, "MarketDataService", _StubService)
    monkeypatch.setattr(
        trading_service, "StrategyEngine",
        lambda db, **kwargs: type("Stub", (), {"evaluate": evaluate or (lambda self, *a, **k: None)})(),
    )


def test_paper_mode_client_is_futures_mainnet(db):
    set_state(db, "execution_mode", "paper")
    set_state(db, "testnet", True)  # paper 不受 testnet 开关影响，固定主网
    client = default_market_client(db)
    assert isinstance(client, BinanceFuturesClient)
    assert client.hosts == FUTURES_MAIN_HOSTS


def test_live_mode_client_still_follows_testnet_flag(db):
    set_state(db, "execution_mode", "live")
    set_state(db, "testnet", True)
    assert default_market_client(db).hosts == ["https://testnet.binancefuture.com"]
    set_state(db, "testnet", False)
    assert default_market_client(db).hosts == FUTURES_MAIN_HOSTS


def test_db_none_client_is_futures_mainnet():
    assert default_market_client(None).hosts == FUTURES_MAIN_HOSTS


def test_cycle_log_warning_on_partial_sync_failure(db, monkeypatch):
    symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    set_state(db, "symbol_scope", "custom")
    set_state(db, "symbols", symbols)
    seed_binding(db, symbols)
    _stub_cycle(monkeypatch, fail_for={"BBBUSDT"})
    summary = TradingService(db).run_cycle()
    assert summary["ok"] is True and len(summary["sync_errors"]) == 1
    row = db.query(JobLog).filter(JobLog.job == "trading_cycle").order_by(JobLog.id.desc()).first()
    assert row.level == "warning"


def test_cycle_log_error_on_total_sync_failure(db, monkeypatch):
    symbols = ["AAAUSDT", "BBBUSDT"]
    set_state(db, "symbol_scope", "custom")
    set_state(db, "symbols", symbols)
    seed_binding(db, symbols)
    _stub_cycle(monkeypatch, fail_for={"AAAUSDT", "BBBUSDT"})
    TradingService(db).run_cycle()
    row = db.query(JobLog).filter(JobLog.job == "trading_cycle").order_by(JobLog.id.desc()).first()
    assert row.level == "error"


def test_cycle_log_is_compact_without_symbols_array(db, monkeypatch):
    symbols = [f"SYM{index:04d}USDT" for index in range(600)]
    set_state(db, "symbol_scope", "custom")
    set_state(db, "symbols", symbols)
    seed_binding(db, symbols)
    _stub_cycle(monkeypatch)
    summary = TradingService(db).run_cycle()
    # API 返回值仍带全量 symbols
    assert len(summary["symbols"]) == 600
    # 持久化日志是紧凑摘要：无 symbols/decisions 全量数组，含关键指标且远低于 4000 字符
    row = db.query(JobLog).filter(JobLog.job == "trading_cycle").order_by(JobLog.id.desc()).first()
    assert len(row.message) < 4000
    assert '"symbol_count": 600' in row.message
    assert '"elapsed_ms"' in row.message
    assert "SYM0000USDT" not in row.message
    assert "decisions" not in row.message


def test_signal_database_failure_does_not_poison_remaining_cycle(db, monkeypatch):
    from app.models import ModelRecord, StrategyConfig

    set_state(db, "symbol_scope", "custom")
    set_state(db, "symbols", ["BTCUSDT", "ETHUSDT"])
    seed_binding(db, ["BTCUSDT", "ETHUSDT"])
    evaluated = []

    def evaluate(self, symbol, *args, **kwargs):
        evaluated.append(symbol)
        if symbol == "BTCUSDT":
            db.add(ModelRecord(symbol=symbol, interval="1h", algo="lightgbm", path=None))
            db.flush()  # A real constraint violation requires rollback before reuse.
        assert db.query(ModelRecord).count() == 0
        return None

    _stub_cycle(monkeypatch, evaluate=evaluate)
    result = TradingService(db).run_cycle()
    assert result["ok"] is True
    assert evaluated == ["BTCUSDT", "ETHUSDT"]
    assert len(result["signal_errors"]) == 1
    assert db.query(JobLog).filter(JobLog.job == "trading_cycle").count() == 1


def test_cycle_database_failure_can_still_record_error_log(db, monkeypatch):
    from app.models import ModelRecord

    set_state(db, "symbol_scope", "custom")
    set_state(db, "symbols", [])

    def fail(*args, **kwargs):
        db.add(ModelRecord(symbol="BADUSDT", interval="1h", algo="lightgbm", path=None))
        db.flush()

    # 巡检第一步就是解析绑定；这里让它在写库时炸掉，验证错误日志仍能落库
    monkeypatch.setattr(trading_service, "active_bindings", fail)
    result = TradingService(db).run_cycle()
    assert result["ok"] is False
    row = db.query(JobLog).filter(JobLog.job == "trading_cycle").one()
    assert row.level == "error"


def test_cycle_updates_existing_positions_after_interval_and_scope_changes(db, monkeypatch):
    from app.market.data_service import MarketDataService
    from app.utils import now_ms

    set_state(db, "symbol_scope", "custom")
    set_state(db, "symbols", ["AAAUSDT"])
    set_state(db, "interval", "1h")
    seed_binding(db, ["AAAUSDT"], interval="1h")
    for symbol, interval, status in [
        ("AAAUSDT", "1h", "OPEN"),
        ("HELDUSDT", "1h", "OPEN"),
        ("HELDUSDT", "1m", "OPEN"),
        ("CLOSEDUSDT", "1m", "CLOSED"),
    ]:
        db.add(Order(symbol=symbol, interval=interval, side="LONG", qty=1,
                     entry_price=100, mark_price=100, margin=10,
                     strategy="absorption_reversal", status=status))
    db.commit()
    calls = []

    def sync_many(symbols, interval):
        calls.append((symbols, interval))
        timestamp = now_ms() // 3600000 * 3600000
        for symbol in symbols:
            MarketDataService(db).store(symbol, "1m", [{
                "open_time": minute, "close_time": minute + 59999,
                "open": 100, "high": 102, "low": 99, "close": 101,
                "volume": 1, "quote_volume": 100, "trades": 1,
            } for minute in range(timestamp, now_ms() // 60000 * 60000 + 1, 60000)])
        return []

    monkeypatch.setattr(trading_service, "_sync_recent_many", sync_many)
    monkeypatch.setattr(trading_service, "StrategyEngine", lambda db, **kwargs: None)
    summary = TradingService(db).run_cycle()
    assert summary["ok"] is True
    assert calls == [(["AAAUSDT", "HELDUSDT"], "1h")]
    assert summary["symbols"] == ["AAAUSDT"]  # Existing positions don't expand entry scope.
    assert summary["sync_target_count"] == 2
    from app.trading.portfolio import portfolio_snapshot

    snapshot = portfolio_snapshot(db)
    assert snapshot["unrealized_pnl"] == 3
    assert all(row["mark_price"] == 101 for row in snapshot["positions"])


@pytest.mark.parametrize("sync_fails", [False, True])
def test_old_interval_sync_restores_stops_and_isolates_fetch_failures(db, monkeypatch, sync_fails):
    from app.market.data_service import MarketDataService
    from app.utils import now_ms

    set_state(db, "symbol_scope", "custom")
    set_state(db, "symbols", ["AAAUSDT"])
    set_state(db, "interval", "1h")
    seed_binding(db, ["AAAUSDT"], interval="1h")
    created = (now_ms() // 60000 - 3) * 60000
    order = Order(symbol="HELDUSDT", interval="1m", side="LONG", qty=1,
                  entry_price=100, mark_price=100, margin=10, sl=99, tp=105,
                  strategy="absorption_reversal", created_at=datetime.fromtimestamp(created / 1000, timezone.utc))
    db.add(order)
    db.commit()

    def sync_many(symbols, interval):
        if interval == "1m":
            if sync_fails:
                return ["HELDUSDT: unavailable"]
            MarketDataService(db).store("HELDUSDT", "1m", [{
                "open_time": created + 60000, "close_time": created + 119999,
                "open": 100, "high": 101, "low": 98, "close": 99.5,
                "volume": 1, "quote_volume": 100, "trades": 1,
            }])
        return []

    monkeypatch.setattr(trading_service, "_sync_recent_many", sync_many)
    monkeypatch.setattr(trading_service, "StrategyEngine", lambda db, **kwargs: None)
    summary = TradingService(db).run_cycle()
    assert summary["ok"] is True
    if sync_fails:
        assert order.status == "OPEN"
        assert summary["sync_errors"] == ["HELDUSDT: unavailable"]
        log = db.query(JobLog).filter(JobLog.job == "trading_cycle").order_by(JobLog.id.desc()).first()
        assert log.level == "warning"
    else:
        assert order.status == "CLOSED"
        assert order.close_reason == "stop_loss"
        assert order.realized_pnl < 0
        assert summary["managed"] == [{"order_id": order.id, "actions": ["stop_loss"]}]
