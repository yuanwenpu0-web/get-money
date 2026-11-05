"""一键平仓：一次结清全部持仓，单笔异常不阻断其余订单。"""

from datetime import datetime, timezone

from app.api import trading
from app.models import JobLog, Order


CREATED = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_order(db, symbol="BTCUSDT", side="LONG", status="OPEN", qty=2, entry=100, mark=105):
    row = Order(
        symbol=symbol,
        interval="1h",
        side=side,
        status=status,
        qty=qty,
        entry_price=entry,
        mark_price=mark,
        strategy="absorption_reversal",
        entry_fee=0.1,
        created_at=CREATED,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def test_close_all_closes_every_open_order(db):
    long_order = make_order(db)
    short_order = make_order(db, symbol="ETHUSDT", side="SHORT", mark=95)
    already_closed = make_order(db, symbol="SOLUSDT", status="CLOSED", mark=110)

    result = trading.close_all_orders(db)

    assert result["closed_count"] == 2
    assert result["failed_count"] == 0
    assert result["failed"] == []
    assert {row["symbol"] for row in result["closed"]} == {"BTCUSDT", "ETHUSDT"}
    assert all(row["status"] == "CLOSED" for row in result["closed"])
    assert all(row["close_price"] > 0 and row["closed_at"] for row in result["closed"])
    assert result["realized_pnl"] == round(sum(row["realized_pnl"] for row in result["closed"]), 4)

    assert db.get(Order, long_order.id).status == "CLOSED"
    assert db.get(Order, short_order.id).status == "CLOSED"
    # 已平仓订单保持原样，不重复结算
    untouched = db.get(Order, already_closed.id)
    assert untouched.close_price == 0
    assert untouched.realized_pnl == 0

    logs = db.query(JobLog).filter(JobLog.job == "order").all()
    assert any("一键平仓" in (row.message or "") and "成功 2 笔" in (row.message or "") for row in logs)


def test_close_all_keeps_going_when_one_order_fails(db, monkeypatch):
    broken = make_order(db)
    healthy = make_order(db, symbol="ETHUSDT")
    original = trading.TradingService.manual_close

    def flaky(self, order_id, reason="manual"):
        if order_id == broken.id:
            raise RuntimeError("券商拒绝平仓")
        return original(self, order_id, reason)

    monkeypatch.setattr(trading.TradingService, "manual_close", flaky)

    result = trading.close_all_orders(db)

    assert result["closed_count"] == 1
    assert result["failed_count"] == 1
    assert result["failed"][0]["id"] == broken.id
    assert result["failed"][0]["symbol"] == "BTCUSDT"
    assert "券商拒绝平仓" in result["failed"][0]["error"]
    assert db.get(Order, healthy.id).status == "CLOSED"
    assert db.get(Order, broken.id).status == "OPEN"

    logs = db.query(JobLog).filter(JobLog.job == "order").all()
    assert any(row.level == "warning" and "失败 1 笔" in (row.message or "") for row in logs)


def test_close_all_route_is_not_shadowed_by_order_id():
    # `/orders/{order_id}` 只接受 int，静态路径必须先注册，否则 FastAPI 会先把
    # "close-all" 当成订单 ID 解析。
    paths = [route.path for route in trading.router.routes]
    assert paths.index("/trading/orders/close-all") < paths.index("/trading/orders/{order_id}")


def test_close_all_on_empty_book_reports_zero(db):
    result = trading.close_all_orders(db)

    assert result == {
        "closed": [],
        "failed": [],
        "closed_count": 0,
        "failed_count": 0,
        "realized_pnl": 0.0,
    }
