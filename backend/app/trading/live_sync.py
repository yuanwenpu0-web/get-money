"""实盘订单周期同步：标记价、交易所保护单状态、止盈止损触发后的真实成交结算。

以交易所为真相源：保护单的 stopPrice/status 回写本地 sl/tp；保护单意外消失
而仓位仍在时自动重挂并告警；保护单成交后优先用真实成交均价结算。
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from types import SimpleNamespace

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


def sync_live(db: Session, order: Order, snapshot: dict | None = None) -> Order | None:
    """每周期同步。交易所仓位已消失时按保护单成交状态结算并返回该订单。

    snapshot 为 prefetch_live_snapshots 预取的只读状态；缺省/字段缺失时自动回落
    原网络路径，所以预取失败不会让某笔订单失去同步。
    """
    if order.status != "OPEN" or order.source != "live":
        return None
    client = _LoggedClient(db, _client(db), local_order_id=order.id)
    live = snapshot or {}
    mark = _f(live.get("mark"))
    if mark <= 0:
        try:
            mark = _f(client.premium_index(order.symbol)["markPrice"])
        except BinanceError as exc:
            log(db, "error", f"标记价格同步失败 #{order.id}: {exc}")
    if mark > 0:
        order.mark_price = round_price(mark)
    positions = live.get("positions")
    if positions is None:
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
    protections = _sync_protections(db, client, order, note, still_open,
                                    live.get("protections"))
    order.note = json.dumps(note, ensure_ascii=False)
    if still_open:
        db.commit()
        return None
    return _settle_exchange_closed(db, client, order, _close_reason(protections), protections)


def _client(db: Session) -> BinanceFuturesClient:
    from .live import client_for

    return client_for(db)


def prefetch_live_snapshots(db: Session, orders: list[Order], workers: int) -> dict[int, dict]:
    """并发只读预取 live 订单的交易所状态；结算/撤单/重挂仍由主线程串行执行。

    只把纯网络读取放进线程池：不改 ORM 对象、不写库、不下单。某笔请求失败时
    该字段留空，sync_live 会自动回落原网络路径，不会因预取失败丢掉同步。
    """
    if not orders:
        return {}
    # ORM 对象只留在主线程；线程里只用普通 dict，避免跨线程 lazy load
    stubs = [{"id": order.id, "symbol": order.symbol, "note": load_note(order)}
             for order in orders]
    client = _client(db)

    def read_one(stub: dict) -> dict:
        symbol = stub["symbol"]
        out: dict = {"mark": None, "positions": None, "protections": {}}
        try:
            out["mark"] = _f(client.premium_index(symbol)["markPrice"])
        except Exception:  # noqa: BLE001 - 预取失败回落原路径
            pass
        try:
            rows = client.position_risk(symbol)
            out["positions"] = rows if isinstance(rows, list) else None
        except Exception:  # noqa: BLE001
            pass
        note = stub["note"]
        for field, *_ in PROTECTIONS:
            order_id = note.get(field)
            if not order_id:
                continue
            try:
                info = _protection_status(client, SimpleNamespace(symbol=symbol), note, order_id)
                if isinstance(info, dict):
                    out["protections"][field] = info
            except Exception:  # noqa: BLE001
                pass
        return out

    count = min(max(1, int(workers)), len(stubs))
    snapshots: dict[int, dict] = {}
    # connection_pool 作用域内 httpx.Client 在线程间复用，避免每单重建连接；
    # 测试桩/自定义客户端可能没有连接池，用 nullcontext 保持兼容。
    pool_ctx = getattr(client, "connection_pool", None)
    with (pool_ctx() if pool_ctx is not None else nullcontext()), ThreadPoolExecutor(
        max_workers=count, thread_name_prefix="live-prefetch"
    ) as pool:
        futures = {pool.submit(read_one, stub): stub["id"] for stub in stubs}
        for future in as_completed(futures):
            try:
                snapshots[futures[future]] = future.result()
            except Exception:  # noqa: BLE001 - 无快照即走原串行网络路径
                pass
    return snapshots


def _sync_protections(db: Session, client, order: Order, note: dict, still_open: bool,
                      prefetched: dict | None = None) -> dict:
    """回写保护单真实状态与触发价；仓位仍在但保护单消失时自动重挂。"""
    prefetched = prefetched or {}
    found: dict = {}
    for field, _action, order_type, status_key, _reason in PROTECTIONS:
        order_id = note.get(field)
        if not order_id:
            continue
        info = prefetched.get(field)
        if info is None:
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
