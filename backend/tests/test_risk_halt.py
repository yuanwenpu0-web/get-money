"""账户级熔断测试：回撤触发、连续亏损触发、正常流程不误伤、熔断拒绝开仓与手动恢复。"""

from datetime import timedelta
from types import SimpleNamespace

from app.api.settings import update_settings
from app.models import JobLog, Order, StrategyConfig, utcnow
from app.schemas import SettingsUpdate
from app.state import all_state, set_state
from app.trading import dispatch, paper
from app.trading.service import TradingService
from app.utils import interval_ms, now_ms


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
                strategy="trend_follow",
                score=1.0,
                prob_up=0.8,
                ts=now_ms() - interval_ms("1m"),  # 已收盘 bar，避免被入场闸门判为未收盘
            )
            | overrides
        )
    )


def open_order(**overrides):
    defaults = dict(
        symbol="BTCUSDT",
        interval="1m",
        side="LONG",
        qty=1.0,
        entry_price=100.0,
        mark_price=100.0,
        margin=0.0,
        leverage=1,
        sl=99.0,
        tp=103.0,
        status="OPEN",
        source="paper",
        strategy="trend_follow",
    )
    return Order(**(defaults | overrides))


def closed_order(pnl: float, minutes_ago: float) -> Order:
    order = open_order(status="CLOSED", realized_pnl=pnl, close_reason="stop_loss")
    order.closed_at = utcnow() - timedelta(minutes=minutes_ago)
    return order


def test_drawdown_halt_triggers_after_close(db):
    set_state(db, "initial_capital", 10000.0)
    set_state(db, "total_capital", 10000.0)
    set_state(db, "cash", 500.0)  # 平仓后权益远低于峰值，模拟深度亏损
    order = open_order()
    db.add(order)
    db.commit()
    paper.close_position(db, order, 100.0, "stop_loss")
    state = all_state(db)
    assert state["risk_halt"] is True
    assert "总回撤" in state["risk_halt_reason"]
    row = db.query(JobLog).filter(JobLog.job == "risk_halt", JobLog.level == "error").first()
    assert row is not None and "账户熔断触发" in row.message


def test_consecutive_loss_halt_triggers_after_close(db):
    set_state(db, "max_drawdown_halt_pct", 0)  # 隔离回撤检查，只验证连续亏损
    set_state(db, "consecutive_loss_halt", 3)
    set_state(db, "cash", 9900.0)
    for index in range(3):
        db.add(closed_order(-10.0, minutes_ago=5 - index))
    db.commit()
    order = open_order()
    db.add(order)
    db.commit()
    paper.close_position(db, order, 99.0, "stop_loss")  # 第 4 笔连续亏损
    state = all_state(db)
    assert state["risk_halt"] is True
    assert "连续亏损 4 笔" in state["risk_halt_reason"]


def test_normal_close_does_not_halt(db):
    winner = open_order()
    db.add(winner)
    db.commit()
    paper.close_position(db, winner, 103.0, "take_profit")
    assert all_state(db)["risk_halt"] is False
    assert db.query(JobLog).filter(JobLog.job == "risk_halt").count() == 0

    loser = open_order()
    db.add(loser)
    db.commit()
    paper.close_position(db, loser, 99.5, "stop_loss")  # 单笔亏损：回撤与笔数都未到线
    assert all_state(db)["risk_halt"] is False
    assert db.query(JobLog).filter(JobLog.job == "risk_halt").count() == 0


def test_halt_blocks_open_until_manual_recovery(db):
    dispatch._last_halt_log_at = 0.0  # 重置日志节流，保证断言不受用例顺序影响
    set_state(db, "risk_halt", True)
    set_state(db, "risk_halt_reason", "总回撤超限")
    assert dispatch.open_position(db, signal()) is None
    assert db.query(Order).count() == 0
    warnings = db.query(JobLog).filter(JobLog.job == "risk_halt", JobLog.level == "warning").all()
    assert len(warnings) == 1 and "拒绝开仓" in warnings[0].message

    # 人工复盘后在设置页恢复
    update_settings(SettingsUpdate(risk_halt=False, risk_halt_reason="复盘后恢复"), db)
    assert all_state(db)["risk_halt"] is False
    order = dispatch.open_position(db, signal())
    assert order is not None and order.source == "paper"


def test_run_cycle_skips_open_while_halted(db, monkeypatch):
    from app.trading import service as trading_service

    set_state(db, "symbol_scope", "custom")
    set_state(db, "symbols", ["AAAUSDT"])
    set_state(db, "risk_halt", True)
    set_state(db, "risk_halt_reason", "熔断测试")
    db.add(StrategyConfig(code="trend_follow", name="趋势", params="{}", enabled=1))
    db.commit()

    class _StubService:
        def __init__(self, db, client=None):
            self.db = db

        def sync_recent_many(self, symbols, interval, **kwargs):
            return {"errors": []}

    live_signal = SimpleNamespace(
        symbol="AAAUSDT", interval="1m", direction="LONG", score=0.9,
        strategy="trend_follow", ts=now_ms() - interval_ms("1m"), price=100.0, sl=99.0, tp=103.0,
        leverage=5, position_pct=0.1, prob_up=0.7,
        to_dict=lambda: {"symbol": "AAAUSDT"},
    )
    monkeypatch.setattr(trading_service, "MarketDataService", _StubService)
    monkeypatch.setattr(
        trading_service, "StrategyEngine",
        lambda db, **kwargs: type("Stub", (), {"evaluate": lambda self, *a, **k: live_signal})(),
    )
    summary = TradingService(db).run_cycle()
    assert summary["ok"] is True
    assert summary["opened"] == []
    assert any("账户熔断中" in decision for decision in summary["decisions"])
    assert db.query(Order).count() == 0
