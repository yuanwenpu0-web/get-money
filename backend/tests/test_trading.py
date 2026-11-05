from types import SimpleNamespace

import pandas as pd
import pytest
from app.models import Order
from app.trading import order_manager, paper
from app.trading.risk import position_size


def signal(**overrides):
    return SimpleNamespace(
        **(
            dict(
                symbol="BTCUSDT",
                interval="1m",
                direction="LONG",
                price=100.0,
                sl=99.0,
                tp=103.0,
                leverage=10,
                position_pct=0.25,
                strategy="absorption_reversal",
                score=1.0,
                prob_up=0.8,
            )
            | overrides
        )
    )


def test_research_entry_transaction_rolls_back_together(db, monkeypatch):
    from app.models import AppState
    from app.state import get_state, set_state
    from app.strategy.specs.pool_volatility_research import HIGH_CODE

    start = pd.Timestamp("2025-01-01", tz="UTC")
    ts = int(start.timestamp() * 1000)
    frame = pd.DataFrame({"open": [100.]}, index=[start + pd.Timedelta(hours=1)])
    monkeypatch.setattr(paper.MarketDataService, "load_frame", lambda *a, **k: frame)
    monkeypatch.setattr(paper, "now_ms", lambda: ts + 3600000)
    set_state(db, "cash", 1000.)
    original_commit = db.commit
    calls = 0

    def fail_order_commit():
        nonlocal calls
        calls += 1
        if calls == 2:  # portfolio snapshot commits first
            raise RuntimeError("模拟进程中断")
        original_commit()

    monkeypatch.setattr(db, "commit", fail_order_commit)
    research = signal(strategy=HIGH_CODE, interval="1h", ts=ts,
                      reason={"research_entry": True}, pool="test")
    with pytest.raises(RuntimeError):
        paper.open_position(db, research)
    db.rollback()
    assert get_state(db, "cash") == 1000.
    assert db.query(Order).count() == 0
    assert db.get(AppState, "last_entry:BTCUSDT:1h") is None
    monkeypatch.setattr(db, "commit", original_commit)
    order = paper.open_position(db, research)
    assert order is not None
    assert get_state(db, "last_entry:BTCUSDT:1h") == ts
    assert order.params_snapshot is not None and order.pool == "test"
    paper.close_position(db, order, 100.)
    assert paper.open_position(db, research) is None


def test_paper_close_transaction_rolls_back_together(db, monkeypatch):
    from app.state import get_state, set_state

    set_state(db, "cash", 1000.)
    order = Order(symbol="BTCUSDT", interval="1h", side="LONG", qty=1,
                  entry_price=100, mark_price=100, margin=100, entry_fee=0,
                  strategy="absorption_reversal", status="OPEN", source="paper")
    db.add(order)
    db.commit()
    monkeypatch.setattr(db, "commit", lambda: (_ for _ in ()).throw(RuntimeError("中断")))
    with pytest.raises(RuntimeError):
        paper.close_position(db, order, 110.)
    db.rollback()
    assert get_state(db, "cash") == 1000.
    assert db.get(Order, order.id).status == "OPEN"


def test_paper_stop_loss_respects_fee_inclusive_risk(db):
    from app.models import StrategyConfig

    # 吸收反转默认无冷却（脚本没有该闸门）；本用例单独配置冷却来验证撮合侧闸门
    db.add(StrategyConfig(code="absorption_reversal", name="吸收反转", enabled=1,
                          params='{"cooldown_bars": 10}'))
    db.commit()
    order = paper.open_position(db, signal())
    assert order is not None
    closed = paper.close_position(db, order, order.sl, "stop_loss")
    assert (
        closed.realized_pnl >= -50.01
    )  # default 0.5% of 10,000, including both fees/slips
    assert paper.open_position(db, signal()) is None  # persistent cooldown


def test_sizing_cannot_overspend_cash_at_high_leverage():
    margin, qty = position_size(
        100.0, 100.0, 1.0, 125, 100.0, 99.0, 1, 0.01, 0.001, 0.05
    )
    assert margin + qty * 100 * 0.01 <= 100.0 + 1e-9


def test_stale_signal_cannot_open(db, monkeypatch):
    monkeypatch.setattr(paper, "now_ms", lambda: 1_000_000)
    assert paper.open_position(db, signal(ts=1)) is None


def test_paper_allows_stop_only_signal_without_take_profit(db):
    # ZigZag 反手型：反向通道为止损，无止盈（tp=0）必须能开仓
    order = paper.open_position(db, signal(
        strategy="simple_zigzag_double", tp=0.0, sl=105.0,
        direction="SHORT", leverage=1, position_pct=0.5,
    ))
    assert order is not None
    assert order.tp == 0 and order.sl == 105.0 and order.side == "SHORT"


def test_zigzag_order_follows_reverse_level_and_skips_generic_stops(db, monkeypatch):
    start = pd.Timestamp("2026-01-01T00:00:00Z")

    def make_order(sl):
        return Order(
            symbol="BTCUSDT", interval="1m", side="LONG", qty=1.0,
            entry_price=110.0, mark_price=110.0, margin=110.0, leverage=1,
            sl=sl, tp=0.0, strategy="simple_zigzag_double",
            status="OPEN", created_at=start.to_pydatetime(),
        )

    frame = pd.DataFrame(
        dict(open=[110.0], high=[110.5], low=[109.5], close=[110.0], volume=[1.0]),
        index=pd.date_range(start, periods=1, freq="1min"),
    )
    monkeypatch.setattr(order_manager.MarketDataService, "load_frame", lambda *a, **k: frame)
    monkeypatch.setattr(
        order_manager, "now_ms",
        lambda: int((start + pd.Timedelta(minutes=1)).timestamp() * 1000),
    )

    def zigzag_signal(short_level):
        return SimpleNamespace(
            ts=int(start.timestamp() * 1000), direction="FLAT",
            reason={"zz_levels": {"long": 118.0, "short": short_level}},
        )

    # 多头止损跟随最新空头通道：pivot 下移 → 止损只下调
    order = make_order(105.0)
    db.add(order)
    db.commit()
    result = order_manager.manage_order(db, order, zigzag_signal(102.0))
    assert order.sl == 102.0 and "zz_stop=102.0" in result["actions"]
    assert order.status == "OPEN"  # 未触及，正常持仓

    # 止损只单向松动：新的通道价位更高时不回收紧
    db.add(make_order(102.0))
    db.commit()
    tighter = db.query(Order).filter(Order.id != order.id).first()
    order_manager.manage_order(db, tighter, zigzag_signal(106.0))
    assert tighter.sl == 102.0


def test_order_does_not_replay_pre_entry_candle_or_same_candle(db, monkeypatch):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    order = Order(
        symbol="BTCUSDT",
        interval="1m",
        side="LONG",
        qty=1.0,
        entry_price=100.0,
        mark_price=100.0,
        margin=100.0,
        leverage=1,
        sl=99.0,
        tp=105.0,
        strategy="absorption_reversal",
        status="OPEN",
        created_at=start.to_pydatetime(),
    )
    db.add(order)
    db.commit()
    frame = pd.DataFrame(
        dict(
            open=[100.0, 100.0],
            high=[102.0, 100.2],
            low=[90.0, 99.8],
            close=[100.0, 100.1],
            volume=[1.0, 1.0],
        ),
        index=pd.date_range(start - pd.Timedelta(minutes=1), periods=2, freq="1min"),
    )
    monkeypatch.setattr(
        order_manager.MarketDataService, "load_frame", lambda *a, **k: frame
    )
    monkeypatch.setattr(
        order_manager,
        "now_ms",
        lambda: int((start + pd.Timedelta(minutes=1)).timestamp() * 1000),
    )
    order_manager.manage_order(db, order)
    assert order.status == "OPEN"
    order.sl = (
        100.05  # manual tightening must not apply retroactively to the same candle
    )
    db.commit()
    assert order_manager.manage_order(db, order)["actions"] == []
    assert order.status == "OPEN"


def test_signal_engine_matches_shared_rules_and_blocks_stale(db, monkeypatch):
    import numpy as np
    from app.features.indicators import build_features
    from app.strategy import engine
    from app.strategy.rules import evaluate_signals, resolve_params

    index = pd.date_range("2026-01-01", periods=200, freq="1min", tz="UTC")
    close = 100 + np.arange(200) * 0.04
    frame = pd.DataFrame(
        dict(open=close, high=close + 0.1, low=close - 0.1, close=close, volume=100.0),
        index=index,
    )
    monkeypatch.setattr(engine.MarketDataService, "load_frame", lambda *a, **k: frame)
    timestamp = int(index[-1].timestamp() * 1000) + 60000
    monkeypatch.setattr(engine, "now_ms", lambda: timestamp)
    result = engine.StrategyEngine(db).evaluate("BTCUSDT", "1m", "absorption_reversal")
    shared = evaluate_signals(
        build_features(frame).iloc[-1:],
        "absorption_reversal",
        resolve_params("absorption_reversal", {}),
    ).iloc[0]
    assert result.direction == {0: "FLAT", 1: "LONG", -1: "SHORT"}[shared.direction]
    assert result.position_pct == pytest.approx(shared.position_pct)
    monkeypatch.setattr(engine, "now_ms", lambda: timestamp + 300000)
    stale = engine.StrategyEngine(db).evaluate("BTCUSDT", "1m", "absorption_reversal")
    assert stale.direction == "FLAT"
    assert stale.reason["block_reason"] == "stale_data"


def test_order_snapshot_preserves_binding_and_legacy_resolution(db, monkeypatch):
    import json
    from app.serializers import order_dict
    from app.strategy.rules import resolve_params

    params = resolve_params("absorption_reversal", {}, {"max_hold_bars": 12})
    order = paper.open_position(db, signal(pool="first", binding_id=42, params=params))
    assert order is not None
    params["max_hold_bars"] = 24
    db.expire_all()
    stored = db.get(Order, order.id)
    assert stored.binding_id == 42
    assert json.loads(stored.params_snapshot)["params"]["max_hold_bars"] == 12
    assert order_dict(stored)["pool"] == "first"
    assert "params_snapshot" not in order_dict(stored)
    assert order_dict(stored, include_snapshot=True)["params_snapshot"]["max_hold_bars"] == 12
    monkeypatch.setattr(order_manager.MarketDataService, "load_frame", lambda *a, **k: pd.DataFrame())
    calls = []
    monkeypatch.setattr(order_manager, "resolve_order_params", lambda *a, **k: calls.append(True) or params)
    order_manager.manage_order(db, stored)
    assert calls == []
    legacy = json.loads(stored.params_snapshot)["params"]
    del legacy["exit"]
    stored.params_snapshot = json.dumps(legacy)
    order_manager.manage_order(db, stored)
    assert calls == []
    stored.params_snapshot = None
    order_manager.manage_order(db, stored)
    assert calls == [True]


@pytest.mark.parametrize("raw", ["{", "{}", "[]", '{"fee_rate": 0.0004}', 'null'])
def test_bad_snapshot_detail_and_reverse_preserve_position(db, monkeypatch, raw):
    from fastapi import HTTPException
    from app.api.trading import get_order, list_orders, reverse_order
    from app.trading import service

    order = paper.open_position(db, signal())
    order.params_snapshot = raw
    db.commit()
    listing = list_orders(db=db)
    assert "params_snapshot" not in listing["orders"][0]
    detail = get_order(order.id, db=db)
    assert detail["params_snapshot"] is None
    assert detail["params_snapshot_status"] == "invalid"
    monkeypatch.setattr(service, "close_position", lambda *a, **k: pytest.fail("不能平掉旧仓"))
    monkeypatch.setattr(service, "open_position", lambda *a, **k: pytest.fail("不能创建新仓"))
    with pytest.raises(HTTPException) as exc:
        reverse_order(order.id, db=db)
    assert exc.value.status_code == 409
    assert db.get(Order, order.id).status == "OPEN"
    assert db.query(Order).count() == 1


def test_reverse_valid_snapshot_preserves_binding_and_params(db):
    import json
    from app.trading.service import TradingService

    original = paper.open_position(db, signal(pool="first", binding_id=42))
    snapshot = json.loads(original.params_snapshot)
    reversed_order = TradingService(db).reverse_order(original.id)
    assert original.status == "CLOSED"
    assert reversed_order is not None and reversed_order.side == "SHORT"
    assert reversed_order.binding_id == 42 and reversed_order.pool == "first"
    assert json.loads(reversed_order.params_snapshot) == snapshot


def test_reverse_legacy_invalid_configuration_does_not_close(db, monkeypatch):
    from app.trading import service

    order = paper.open_position(db, signal())
    order.params_snapshot = None
    db.commit()
    monkeypatch.setattr(service, "resolve_order_params", lambda *a, **k: {})
    with pytest.raises(ValueError, match="快照不完整"):
        service.TradingService(db).reverse_order(order.id)
    assert db.get(Order, order.id).status == "OPEN"


def test_legacy_order_schema_adds_nullable_snapshot_columns(monkeypatch):
    from sqlalchemy import create_engine, inspect, text
    from app import db as db_module

    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE orders (id INTEGER PRIMARY KEY, pool VARCHAR(48))"))
        conn.execute(text("INSERT INTO orders (id, pool) VALUES (1, 'old')"))
    monkeypatch.setattr(db_module, "engine", engine)
    db_module._add_missing_columns()
    db_module._add_missing_columns()
    assert {col["name"] for col in inspect(engine).get_columns("orders")} >= {
        "binding_id", "params_snapshot"
    }
    with engine.connect() as conn:
        assert conn.execute(text("SELECT pool, binding_id, params_snapshot FROM orders")).one() == (
            "old", None, None
        )
    engine.dispose()


def test_incomplete_snapshot_rejected_before_paper_entry(db):
    with pytest.raises(ValueError, match="快照不完整"):
        paper.open_position(db, signal(params={"max_hold_bars": 12}))
    assert db.query(Order).count() == 0
