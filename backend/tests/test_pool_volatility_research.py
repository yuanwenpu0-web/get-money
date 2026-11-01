from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.research.lab_signals import rsi_reversal, bollinger_break, atr_channel_break
from app.strategy.registry import get_spec
from app.strategy.defaults import DEFAULT_STRATEGIES
from app.strategy.rules import evaluate_signals, prepare_frame, resolve_params
from app.strategy.specs.pool_volatility_research import HIGH_CODE, LOW_CODE
from app.trading import dispatch, live, paper


@pytest.mark.parametrize("code,builder,params", [
    (HIGH_CODE, bollinger_break, {"n": 35, "k": 2.0}),
    (LOW_CODE, atr_channel_break, {"n": 20, "k": 2.0}),
])
def test_research_signal_and_guard(code, builder, params):
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.04, 500)))
    frame = pd.DataFrame({"open": close, "high": close * 1.02, "low": close * 0.98,
                          "close": close, "volume": 10.0},
                         index=pd.date_range("2025-01-01", periods=500, freq="h", tz="UTC"))
    spec = get_spec(code)
    assert not spec.default_enabled
    assert next(row for row in DEFAULT_STRATEGIES if row["code"] == code)["enabled"] == 0
    resolved = resolve_params(code, {})
    prepared = prepare_frame(frame, code, resolved)
    expected = rsi_reversal(frame, {"n": 21, "low": 30, "high": 70}).long | builder(frame, params).long
    expected[:200] = False
    np.testing.assert_array_equal(prepared.research_entry, expected)
    output = evaluate_signals(prepared, code, resolved)
    assert (output.direction == 0).all()
    assert (output.block_reason == "").all()
    assert "execution_blocked" not in spec.tags
    assert "模拟盘" in spec.description and "实盘" in spec.description
    assert "均被屏蔽" not in spec.description
    assert spec.describe(prepared.iloc[-1]) == {"research_entry": bool(prepared.research_entry.iloc[-1])}
    assert expected.any()
    with pytest.raises(ValueError):
        resolve_params(code, {}, {"research_hold_hours": 60})


@pytest.mark.parametrize("code", [HIGH_CODE, LOW_CODE])
def test_live_entrypoints_reject(code, monkeypatch):
    signal = SimpleNamespace(strategy=code, direction="LONG")
    monkeypatch.setattr(dispatch, "halt_status", lambda db: (False, ""))
    monkeypatch.setattr(dispatch, "is_live", lambda db: True)
    monkeypatch.setattr(dispatch, "_log_live_block", lambda *args: None)
    assert dispatch.open_position(None, signal) is None
    assert live.open_live(None, signal) is None


def test_gap_disables_next_200_observed_bars():
    index = pd.date_range("2025-01-01", periods=500, freq="h", tz="UTC").delete(250)
    close = np.linspace(100, 120, len(index))
    frame = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                          "close": close, "volume": 1.0}, index=index)
    prepared = prepare_frame(frame, HIGH_CODE, resolve_params(HIGH_CODE, {}))
    assert not prepared.research_entry.iloc[250:450].any()


@pytest.mark.parametrize("code,hours", [(HIGH_CODE, 120), (LOW_CODE, 48)])
def test_paper_entry_and_exit_follow_actual_open_after_gaps(db, monkeypatch, code, hours):
    from app.state import set_state
    from app.trading import order_manager
    from app.utils import dt_ms

    start = pd.Timestamp("2025-01-01", tz="UTC")
    signal_ts = int(start.timestamp() * 1000)
    index = pd.DatetimeIndex([start + pd.Timedelta(hours=1),
                              start + pd.Timedelta(hours=1 + hours),
                              start + pd.Timedelta(hours=3 + hours)])
    frame = pd.DataFrame({"open": [100., 110., 120.], "high": [101., 111., 121.],
                          "low": [99., 109., 119.], "close": [100., 110., 120.]}, index=index)
    class Market:
        def __init__(self, db):
            pass
        def load_frame(self, symbol, interval, start_ms=None, **kwargs):
            return frame.loc[frame.index.as_unit("ms").asi8 >= start_ms]
    monkeypatch.setattr(paper, "MarketDataService", Market)
    monkeypatch.setattr(order_manager, "MarketDataService", Market)
    now = [signal_ts + 3600000]
    monkeypatch.setattr(paper, "now_ms", lambda: now[0])
    monkeypatch.setattr(order_manager, "now_ms", lambda: now[0])
    set_state(db, "cash", 10000.)
    set_state(db, "fee_rate", 0.)
    set_state(db, "slippage", 0.)
    signal = SimpleNamespace(strategy=code, interval="1h", direction="LONG", ts=signal_ts,
                             symbol="TESTUSDT", pool="pool", reason={"research_entry": True})
    order = paper.open_position(db, signal)
    assert order.entry_price == 100
    assert dt_ms(order.created_at) == int(index[0].timestamp() * 1000)
    assert order.margin == 200 and order.leverage == 1 and order.sl == 0
    now[0] = int((index[1] + pd.Timedelta(hours=1)).timestamp() * 1000)
    assert order_manager.manage_order(db, order)["actions"] == []
    assert order.status == "OPEN"
    now[0] = int(index[2].timestamp() * 1000)
    assert order_manager.manage_order(db, order)["actions"] == ["research_hold"]
    assert order.close_price == 120
    assert paper.open_position(db, signal) is None


@pytest.mark.parametrize("elapsed,allowed", [(3599999, False), (3600000, True),
                                              (7199999, True), (7200000, False),
                                              (14400000, False)])
def test_direct_research_entry_respects_signal_window(db, monkeypatch, elapsed, allowed):
    from app.models import Order
    from app.state import get_state, set_state

    start = pd.Timestamp("2025-01-01", tz="UTC")
    ts = int(start.timestamp() * 1000)
    frame = pd.DataFrame({"open": [100.]}, index=[start + pd.Timedelta(hours=1)])
    monkeypatch.setattr(paper.MarketDataService, "load_frame", lambda *args, **kwargs: frame)
    monkeypatch.setattr(paper, "now_ms", lambda: ts + elapsed)
    set_state(db, "cash", 1000.)
    signal = SimpleNamespace(strategy=HIGH_CODE, interval="1h", direction="LONG", ts=ts,
                             symbol="TESTUSDT", pool="pool", reason={"research_entry": True})
    order = paper.open_research_position(db, signal)
    assert (order is not None) == allowed
    assert db.query(Order).count() == int(allowed)
    if not allowed:
        assert get_state(db, "cash") == 1000.
        assert get_state(db, "last_entry:TESTUSDT:1h", None) is None


def test_research_entry_rejects_wrong_interval_and_missing_observation(db):
    signal = SimpleNamespace(strategy=HIGH_CODE, interval="4h", direction="LONG", ts=1,
                             reason={"research_entry": True})
    assert paper.open_position(db, signal) is None
    signal.interval = "1h"
    signal.reason = {}
    assert paper.open_position(db, signal) is None
