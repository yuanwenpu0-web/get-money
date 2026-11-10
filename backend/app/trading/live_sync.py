"""实盘订单周期同步：标记价、交易所保护单状态、止盈止损触发后的真实成交结算。

以交易所为真相源：保护单的 stopPrice/status 回写本地 sl/tp；保护单意外消失
而仓位仍在时自动重挂并告警；保护单成交后优先用真实成交均价结算。
"""
from __future__ import annotations

import json

from sqlalchemy.orm import Session

from ..market.futures import BinanceError, BinanceFuturesClient, format_amount
from ..models import JobLog, Order
from ..utils import round_price
from . import exchange_gateway as gw
from . import exchange_truth
from .live import _close_side, _LoggedClient, load_note, log
from .paper import close_position as settle_close

PROTECTIONS = (
    ("sl_order_id", "protect_sl", "STOP_MARKET", "sl_status", "exchange_sl"),
    ("tp_order_id", "protect_tp", "TAKE_PROFIT_MARKET", "tp_status", "exchange_tp"),
)
MISSING_STATUSES = ("CANCELED", "EXPIRED", "REJECTED", "PENDING_CANCEL")
# Algo 条件单触发后的状态不叫 FILLED：TRIGGERED 表示已转为实际市价单，
# actualPrice 才是真实成交均价。
FILLED_STATUSES = ("FILLED", "TRIGGERED", "FINISHED", "EXECUTED")


def _f(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def sync_live(db: Session, order: Order) -> Order | None:
    """每周期同步。交易所仓位已消失时按保护单成交状态结算并返回该订单。"""
    if order.status != "OPEN" or order.source != "live":
        return None
    client = _LoggedClient(db, _client(db), local_order_id=order.id)
    try:
        mark = _f(client.premium_index(order.symbol)["markPrice"])
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
        row.get("symbol") == order.symbol and abs(_f(row.get("positionAmt"))) > 0
        for row in positions
    )
    note = load_note(order)
    protections = _sync_protections(db, client, order, note, still_open)
    order.note = json.dumps(note, ensure_ascii=False)
    if still_open:
        db.commit()
        return None
    return _settle_exchange_closed(db, client, order, _close_reason(protections), protections)


def _client(db: Session) -> BinanceFuturesClient:
    from .live import client_for

    return client_for(db)


def _sync_protections(db: Session, client, order: Order, note: dict, still_open: bool) -> dict:
    """回写保护单真实状态与触发价；仓位仍在但保护单消失时自动重挂。"""
    found: dict = {}
    for field, _action, order_type, status_key, _reason in PROTECTIONS:
        order_id = note.get(field)
        if not order_id:
            continue
        try:
            info = _protection_status(client, order, note, order_id)
        except BinanceError as exc:
            log(db, "warning", f"保护单状态查询失败 #{order.id} {field}={order_id}: {exc}")
            continue
        if not isinstance(info, dict):
            continue
        status = str(info.get("status") or "")
        stop_price = _f(info.get("stopPrice"))
        found[field] = {"status": status, "stopPrice": stop_price,
                        "avgPrice": _f(info.get("avgPrice")), "orderId": order_id}
        note[status_key] = status
        if status == "NEW" and stop_price > 0:
            # 交易所侧可能被人工改过触发价，以它为准
            if field == "sl_order_id":
                order.sl = round_price(stop_price)
            else:
                order.tp = round_price(stop_price)
        elif still_open and status in MISSING_STATUSES:
            _restore_protection(db, client, order, note, field, order_type)
    return found


def _protection_status(client, order: Order, note: dict, order_id) -> dict:
    """读 Algo 条件单状态并归一化；迁移前的普通条件单回落 /fapi/v1/order。"""
    if note.get("protection_kind") != "legacy":
        try:
            info = client.get_algo_order(order.symbol, int(order_id))
            return {"status": str(info.get("algoStatus") or ""),
                    "stopPrice": _f(info.get("triggerPrice")),
                    "avgPrice": _f(info.get("actualPrice")),
                    "actualOrderId": info.get("actualOrderId") or ""}
        except BinanceError:
            pass  # 老订单没有 algoId，继续回落普通查询
    info = client.get_order(order.symbol, int(order_id))
    return {"status": str(info.get("status") or ""),
            "stopPrice": _f(info.get("stopPrice")),
            "avgPrice": _f(info.get("avgPrice")), "actualOrderId": ""}


def _restore_protection(db: Session, client, order: Order, note: dict, field: str, order_type: str) -> None:
    """保护单被撤销/过期而仓位还在：按本地 sl/tp 重挂，避免持仓裸奔。"""
    price = order.sl if field == "sl_order_id" else order.tp
    if not price or price <= 0:
        log(db, "error", f"订单 #{order.id} 保护单缺失且本地 {field} 无效，无法自动重挂")
        return
    try:
        resp = client.new_order(
            symbol=order.symbol, side=_close_side(order.side), type=order_type,
            stopPrice=format_amount(price), closePosition="true", newOrderRespType="RESULT",
        )
    except BinanceError as exc:
        log(db, "error", f"订单 #{order.id} 自动重挂 {order_type} 失败: {exc}，请立即人工检查交易所仓位")
        db.add(JobLog(job="live_trading", level="error",
                      message=f"保护单缺失且重挂失败 #{order.id} {order.symbol}: {exc}"[:4000]))
        db.commit()
        return
    note[field] = resp.get("orderId")
    note["sl_status" if field == "sl_order_id" else "tp_status"] = "NEW"
    log(db, "warning", f"订单 #{order.id} {field} 保护单在交易所侧消失，已按本地价 {price} 自动重挂")


def _close_reason(protections: dict) -> str:
    for field, _a, _t, _s, reason in PROTECTIONS:
        if (protections.get(field) or {}).get("status") in FILLED_STATUSES:
            return reason
    return "exchange_closed"


def _fill_price(protections: dict) -> float:
    """优先用已成交保护单的真实成交均价，而不是标记价估算。"""
    for field, _a, _t, _s, _r in PROTECTIONS:
        info = protections.get(field) or {}
        if info.get("status") in FILLED_STATUSES and _f(info.get("avgPrice")) > 0:
            return _f(info["avgPrice"])
    return 0.0


def _settle_exchange_closed(db: Session, client, order: Order, reason: str,
                            protections: dict | None = None) -> Order:
    try:
        mark = _f(client.premium_index(order.symbol)["markPrice"])
        if mark > 0:
            order.mark_price = round_price(mark)
    except BinanceError:
        pass
    price = _fill_price(protections or {}) or order.mark_price or order.entry_price
    log(db, "info", f"订单 #{order.id} 已在交易所侧平仓（{reason}），按 {price} 结算本地记录")
    settled = settle_close(db, order, price, reason, slippage_rate=0.0, update_cash=False)
    if not exchange_truth.backfill_fills(db, settled, client):
        note = load_note(settled)
        note["pnl_source"] = exchange_truth.PNL_ESTIMATED
        settled.note = json.dumps(note, ensure_ascii=False)
        db.commit()
    return settled
