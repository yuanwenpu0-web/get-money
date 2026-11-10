"""批量勾选平仓：只平用户选中的 ID，且「按筛选全选」与列表口径一致。"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.api import trading
from app.models import JobLog, Order
from app.schemas import BatchClosePayload


CREATED = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_order(db, symbol="BTCUSDT", side="LONG", status="OPEN", strategy="absorption_reversal", mark=105):
    row = Order(
        symbol=symbol,
        interval="1h",
        side=side,
        status=status,
        qty=2,
        entry_price=100,
        mark_price=mark,
        strategy=strategy,
        entry_fee=0.1,
        created_at=CREATED,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def test_batch_close_only_closes_selected_ids(db):
    first = make_order(db)
    untouched = make_order(db, symbol="ETHUSDT")
    third = make_order(db, symbol="SOLUSDT", side="SHORT", mark=95)

    # 重复 ID 只算一次；没勾选的订单必须原样留着
    result = trading.batch_close_orders(BatchClosePayload(order_ids=[third.id, first.id, first.id]), db)

    assert result["requested_count"] == 2
    assert result["closed_count"] == 2
    assert result["failed_count"] == 0
    assert result["skipped_count"] == 0
    assert {row["id"] for row in result["closed"]} == {first.id, third.id}
    assert db.get(Order, first.id).status == "CLOSED"
    assert db.get(Order, third.id).status == "CLOSED"
    assert db.get(Order, untouched.id).status == "OPEN"
    assert any("批量平仓" in (row.message or "") and "成功 2 笔" in (row.message or "")
               for row in db.query(JobLog).filter(JobLog.job == "order").all())


def test_batch_close_skips_non_open_and_keeps_going_after_failure(db, monkeypatch):
    broken = make_order(db)
    healthy = make_order(db, symbol="ETHUSDT")
    already_closed = make_order(db, symbol="SOLUSDT", status="CLOSED")
    original = trading.TradingService.manual_close

    def flaky(self, order_id, reason="manual"):
        if order_id == broken.id:
            raise RuntimeError("券商拒绝平仓")
        return original(self, order_id, reason)

    monkeypatch.setattr(trading.TradingService, "manual_close", flaky)

    result = trading.batch_close_orders(
        BatchClosePayload(order_ids=[broken.id, 999999, already_closed.id, healthy.id]), db
    )

    assert result["requested_count"] == 4
    assert result["closed_count"] == 1
    assert result["failed_count"] == 1
    assert result["skipped_count"] == 2
    assert result["failed"][0]["id"] == broken.id
    assert "券商拒绝平仓" in result["failed"][0]["error"]
    assert {item["id"] for item in result["skipped"]} == {999999, already_closed.id}
    assert db.get(Order, healthy.id).status == "CLOSED"  # 失败不阻断后续
    assert db.get(Order, broken.id).status == "OPEN"


def test_open_ids_uses_the_same_filters_as_the_list(db):
    btc_target = make_order(db, symbol="BTCUSDT")
    make_order(db, symbol="BTCUSDT", strategy="supertrend_current")
    make_order(db, symbol="ETHUSDT")
    make_order(db, symbol="BTCUSDT", status="CLOSED")

    result = trading.list_open_order_ids(
        status="OPEN", symbol="btcusdt", strategy=" Absorption_Reversal ", db=db
    )

    assert result == {"order_ids": [btc_target.id], "count": 1}
    # 列表 summary 与全选接口必须给出同一个数量
    listed = trading.list_orders(status="OPEN", symbol="btcusdt", strategy="absorption_reversal", db=db)
    assert listed["summary"]["selectable_open"] == result["count"]

    # 状态筛到「已平仓」时没有可平仓对象；「全部」时按币种/策略取全部持仓
    assert trading.list_open_order_ids(status="CLOSED", symbol="BTCUSDT", db=db)["count"] == 0
    assert trading.list_open_order_ids(status="", db=db)["count"] == 3


def test_batch_close_payload_rejects_empty_and_oversized():
    with pytest.raises(ValidationError):
        BatchClosePayload(order_ids=[])
    with pytest.raises(ValidationError):
        BatchClosePayload(order_ids=list(range(2001)))
    with pytest.raises(ValidationError):
        BatchClosePayload(order_ids=[1], extra_field=True)


def test_static_batch_routes_are_not_shadowed_by_order_id():
    # 静态路径必须先注册，否则 FastAPI 会把 "batch-close" / "open-ids" 当成订单 ID。
    paths = [route.path for route in trading.router.routes]
    dynamic = paths.index("/trading/orders/{order_id}")
    assert paths.index("/trading/orders/open-ids") < dynamic
    assert paths.index("/trading/orders/batch-close") < dynamic
