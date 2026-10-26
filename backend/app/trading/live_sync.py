"""实盘订单周期同步：更新标记价格；检测交易所侧止盈止损触发导致的仓位消失。"""

from __future__ import annotations

from sqlalchemy.orm import Session

from ..market.futures import BinanceError, BinanceFuturesClient
from ..models import Order
from ..utils import round_price
from .live import load_note, log
from .paper import close_position as settle_close


def sync_live(db: Session, order: Order) -> Order | None:
    """每周期同步。交易所仓位已消失时按保护单成交状态结算本地订单并返回该订单。"""
    if order.status != "OPEN" or order.source != "live":
        return None
    client = _client(db)
    try:
        mark = float(client.premium_index(order.symbol)["markPrice"])
        if mark > 0:
            order.mark_price = round_price(mark)
    except BinanceError as exc:
        log(db, "error", f"标记价格同步失败 #{order.id}: {exc}")
    try:
        positions = client.position_risk(order.symbol)
    except BinanceError as exc:
        log(db, "error", f"仓位同步失败 #{order.id}: {exc}")
        db.commit()
        return None
    still_open = any(
        row.get("symbol") == order.symbol and abs(float(row.get("positionAmt", 0) or 0)) > 0
        for row in positions
    )
    if still_open:
        db.commit()
        return None
    return _settle_exchange_closed(db, client, order, _exchange_close_reason(client, order))


def _client(db: Session) -> BinanceFuturesClient:
    from .live import client_for

    return client_for(db)


def _exchange_close_reason(client: BinanceFuturesClient, order: Order) -> str:
    note = load_note(order)
    for field, reason in (("sl_order_id", "exchange_sl"), ("tp_order_id", "exchange_tp")):
        protection_id = note.get(field)
        if not protection_id:
            continue
        try:
            info = client.get_order(order.symbol, int(protection_id))
        except BinanceError:
            continue
        if info.get("status") == "FILLED":
            return reason
    return "exchange_closed"


def _settle_exchange_closed(db: Session, client: BinanceFuturesClient, order: Order, reason: str) -> Order:
    try:
        mark = float(client.premium_index(order.symbol)["markPrice"])
        if mark > 0:
            order.mark_price = round_price(mark)
    except BinanceError:
        pass
    price = order.mark_price or order.entry_price
    log(db, "info", f"订单 #{order.id} 已在交易所侧平仓（{reason}），按标记价 {price} 结算本地记录")
    return settle_close(db, order, price, reason, slippage_rate=0.0)
