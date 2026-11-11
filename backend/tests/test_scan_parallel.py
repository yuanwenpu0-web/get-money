"""巡检并发配置与三段并行的行为测试：全部用桩，不触达真实币安。"""

import pytest
from fastapi import HTTPException

from app.api.settings import EDITABLE_KEYS, _validate
from app.config import DEFAULT_PUBLIC_INTERVAL_MS, MARKET_SYNC_WORKERS
from app.market import futures
from app.trading import live_sync
from app.trading import service as trading_service

from fakes import FakeFuturesClient, live_order


def test_scan_parallelism_defaults_match_legacy_behaviour():
    """上线默认必须与改造前一致：8 线程 / 350ms / 其余串行。"""
    assert trading_service._scan_parallelism({}) == {
        "market_workers": MARKET_SYNC_WORKERS,
        "public_interval_ms": DEFAULT_PUBLIC_INTERVAL_MS,
        "signal_workers": 1,
        "execution_workers": 1,
    }


def test_scan_parallelism_clamps_dirty_state():
    got = trading_service._scan_parallelism({
        "market_sync_workers": 999,
        "market_public_interval_ms": 1,
        "signal_workers": "8",
        "execution_workers": -3,
    })
    assert got == {"market_workers": 32, "public_interval_ms": 20,
                   "signal_workers": 8, "execution_workers": 1}


def test_public_gate_interval_is_configurable_at_runtime():
    original = futures._PUBLIC_GATE.interval
    try:
        assert futures.set_public_request_interval(0.15) == 0.15
        assert futures._PUBLIC_GATE.interval == 0.15
        assert futures.set_public_request_interval(99) == 2.0
        assert futures.set_public_request_interval(0) == 0.02
    finally:
        futures.set_public_request_interval(original)


class _FakeSession:
    def __init__(self):
        self.rollbacks = 0

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        pass


class _StubBinding:
    interval = "1m"
    strategy = "stub"
    pool = "test"

    def params_for(self, symbol):
        return {"symbol": symbol}


class _StubEngine:
    def __init__(self, db, **kwargs):
        self.db = db

    def evaluate(self, symbol, interval, strategy, save=False, params=None, pool=""):
        if symbol == "BADUSDT":
            raise ValueError("boom")
        return {"symbol": symbol, "save": save, "params": params}


def test_evaluate_pairs_parallel_keeps_order_and_isolates_errors(monkeypatch):
    sessions = []

    def make_session():
        session = _FakeSession()
        sessions.append(session)
        return session

    monkeypatch.setattr(trading_service, "SessionLocal", make_session)
    monkeypatch.setattr(trading_service, "StrategyEngine", _StubEngine)
    symbols = ["AAAUSDT", "BADUSDT", "CCCUSDT", "DDDUSDT"]
    pairs = [(symbol, _StubBinding()) for symbol in symbols]
    out = trading_service._evaluate_pairs_parallel(pairs, workers=3)
    # 结果按 pairs 原下标返回，线程完成顺序不能影响择优
    assert [row[0]["symbol"] if row[0] else None for row in out] == [
        "AAAUSDT", None, "CCCUSDT", "DDDUSDT"]
    assert out[1][1] and "ValueError" in out[1][1]
    assert all(row[1] is None for index, row in enumerate(out) if index != 1)
    assert all(row[0]["save"] is True for row in out if row[0])
    assert len(sessions) >= 2, "每个 worker 必须拿到独立的 Session"


def test_settings_validate_scan_worker_ranges():
    assert {"market_sync_workers", "market_public_interval_ms",
            "signal_workers", "execution_workers"}.issubset(EDITABLE_KEYS)
    with pytest.raises(HTTPException):
        _validate({"signal_workers": 2.5})
    with pytest.raises(HTTPException):
        _validate({"market_public_interval_ms": 5})
    with pytest.raises(HTTPException):
        _validate({"market_sync_workers": 0})
    _validate({"signal_workers": 4, "execution_workers": 2,
               "market_sync_workers": 16, "market_public_interval_ms": 150})


def test_prefetch_live_snapshots_returns_readonly_state(db, monkeypatch):
    fake = FakeFuturesClient(position_amt="0.5", mark_price="102.5")
    monkeypatch.setattr(live_sync, "_client", lambda session: fake)
    orders = [live_order(id=1, symbol="BTCUSDT"), live_order(id=2, symbol="ETHUSDT")]
    snaps = live_sync.prefetch_live_snapshots(db, orders, workers=2)
    assert set(snaps) == {1, 2}
    for snap in snaps.values():
        assert snap["mark"] == 102.5
        assert isinstance(snap["positions"], list)
        assert "sl_order_id" in snap["protections"]
        assert "tp_order_id" in snap["protections"]


class _NoNetworkClient(FakeFuturesClient):
    def premium_index(self, symbol):
        raise AssertionError("有快照时不应再拉标记价")

    def position_risk(self, symbol):
        raise AssertionError("有快照时不应再拉仓位")

    def get_algo_order(self, symbol, algo_id):
        raise AssertionError("有快照时不应再查保护单")


def test_sync_live_uses_prefetched_snapshot_without_network(db, monkeypatch):
    fake = _NoNetworkClient()
    monkeypatch.setattr(live_sync, "_client", lambda session: fake)
    order = live_order(symbol="BTCUSDT")
    db.add(order)
    db.commit()
    snapshot = {
        "mark": 102.0,
        "positions": [{"symbol": "BTCUSDT", "positionAmt": "0.5"}],
        "protections": {
            "sl_order_id": {"status": "NEW", "stopPrice": 99.0, "avgPrice": 0.0},
            "tp_order_id": {"status": "NEW", "stopPrice": 103.0, "avgPrice": 0.0},
        },
    }
    assert live_sync.sync_live(db, order, snapshot) is None
    assert order.status == "OPEN"
    assert order.mark_price == 102.0
    assert order.sl == 99.0 and order.tp == 103.0
