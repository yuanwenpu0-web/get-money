"""币安 U 本位合约实盘执行（测试网 / 主网）。与 paper 模块对称：
- 复用 paper.entry_gates 的入场闸门与 paper.close_position 的本地记账
- 下单成功才写本地 Order；保护单挂失败立即市价回滚，不留脏订单
- 交易所订单号/保护单号存入 Order.note 的 JSON 字段，避免改表迁移
"""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from ..market.futures import (
    BinanceApiError,
    BinanceError,
    BinanceFuturesClient,
    floor_to_step,
    format_amount,
    round_to_tick,
    symbol_filters,
)
from ..models import Credential, JobLog, Order
from ..security import decrypt
from ..state import all_state, set_state
from ..strategy.bindings import resolve_order_params
from ..utils import round_price
from .paper import close_position as settle_close
from .paper import entry_gates
from .maker import execute_entry
from .live_rollback import rollback_open
from .portfolio import portfolio_snapshot
from .risk import position_size

CANCEL_IGNORE_CODES = {-2011}
REDUCE_ONLY_REJECT_CODE = -2022
MARGIN_TYPE_UNCHANGED_CODE = -4046
MIN_MARGIN = 10.0


def log(db: Session, level: str, message: str) -> None:
    db.add(JobLog(job="live_trading", level=level, message=message[:4000]))
    db.commit()


def load_credentials(db: Session) -> tuple[str, str]:
    row = db.query(Credential).order_by(Credential.id.desc()).first()
    if row is None:
        return "", ""
    return decrypt(row.api_key_enc), decrypt(row.api_secret_enc)


def client_for(db: Session) -> BinanceFuturesClient:
    state = all_state(db)
    key, secret = load_credentials(db)
    return BinanceFuturesClient(key, secret, testnet=bool(state.get("testnet", True)),
                                proxy=state.get("proxy_url") or None)


def load_note(order: Order) -> dict:
    try:
        data = json.loads(order.note or "{}")
    except ValueError:
        data = {}
    return data if isinstance(data, dict) else {}


def _close_side(direction: str) -> str:
    return "SELL" if direction == "LONG" else "BUY"


def open_live(db: Session, signal) -> Order | None:
    """实盘开仓：与 paper.open_position 对称，交易所市价成交 + 交易所侧保护单。"""
    state = all_state(db)
    key, secret = load_credentials(db)
    if not key or not secret:
        log(db, "error", f"live 开仓被拒绝: 未配置币安 API 凭证 {getattr(signal, 'symbol', '')}")
        return None
    params = getattr(signal, "params", None) or resolve_order_params(
        db,
        signal.strategy,
        signal.interval,
        getattr(signal, "pool", ""),
        symbol=getattr(signal, "symbol", ""),
    )
    if not entry_gates(db, signal, params, ignore_cooldown=False):
        return None

    snapshot = portfolio_snapshot(db, update_marks=False)
    cash = snapshot["cash"]
    equity = snapshot["equity"]
    position_pct = float(signal.position_pct or 0)
    if position_pct <= 0 or equity <= 0:
        return None
    fee_rate = float(state.get("fee_rate", 0.0004))
    side = 1 if signal.direction == "LONG" else -1
    entry = float(signal.price)
    if (entry - signal.sl) * side <= 0 or (signal.tp - entry) * side <= 0:
        return None
    leverage = min(max(1, int(signal.leverage)), int(params["max_leverage"]))
    margin, qty = position_size(equity, cash, min(position_pct, params["max_position_pct"]), leverage,
                                entry, signal.sl, side, fee_rate, 0.0, params["risk_per_trade_pct"])
    if margin < MIN_MARGIN or qty <= 0:
        return None
    try:
        client = BinanceFuturesClient(key, secret, testnet=bool(state.get("testnet", True)), proxy=state.get("proxy_url") or None)
        order_type = str(state.get("order_type", "market") or "market")
        return _place_live(db, client, signal, margin, qty, entry, leverage, fee_rate, cash, order_type)
    except (BinanceError, ValueError, KeyError, TypeError) as exc:
        log(db, "error", f"live 开仓失败 {signal.symbol}: {exc}")
        return None


def _protection_order(client: BinanceFuturesClient, symbol: str, close_side: str, order_type: str, stop_price: float) -> dict:
    return client.new_order(symbol=symbol, side=close_side, type=order_type,
                            stopPrice=format_amount(stop_price), closePosition="true", newOrderRespType="RESULT")


def _line_logger(db: Session):
    """把 live.log(db, level, message) 适配成 maker / rollback 需要的 (level, message) 回调，
    避免下游模块反向依赖 live 造成循环导入。"""
    return lambda level, message: log(db, level, message)


def _place_live(db: Session, client: BinanceFuturesClient, signal, margin, qty, entry, leverage, fee_rate, cash,
                order_type: str = "market") -> Order | None:
    filters = symbol_filters(client.exchange_info(signal.symbol), signal.symbol)
    lot = filters.get("LOT_SIZE", {})
    price_filter = filters.get("PRICE_FILTER", {})
    step = float(lot.get("stepSize", 0) or 0)
    min_qty = float(lot.get("minQty", 0) or 0)
    tick = float(price_filter.get("tickSize", 0) or 0)
    min_notional = float(filters.get("MIN_NOTIONAL", {}).get("notional", 0) or 0)

    adj_qty = floor_to_step(qty, step)
    if adj_qty <= 0 or adj_qty < min_qty:
        log(db, "error", f"live 开仓失败 {signal.symbol}: 计算数量 {qty} 低于最小下单量或步长取整后为 0")
        return None
    if min_notional > 0 and adj_qty * entry < min_notional:
        log(db, "error", f"live 开仓失败 {signal.symbol}: 名义价值 {adj_qty * entry:.4f} 低于最小限制 {min_notional}")
        return None

    _prepare_account(db, client, signal.symbol, leverage)
    exchange_side = "BUY" if signal.direction == "LONG" else "SELL"
    close_side = _close_side(signal.direction)
    line_log = _line_logger(db)
    order_resp = execute_entry(client, signal.symbol, exchange_side, adj_qty, entry,
                               order_type, tick, step, line_log)
    fill_price = float(order_resp.get("avgPrice") or 0) or entry
    placed_protections: list[int] = []
    try:
        sl_order = _protection_order(client, signal.symbol, close_side, "STOP_MARKET", round_to_tick(signal.sl, tick))
        placed_protections.append(int(sl_order["orderId"]))
        tp_order = _protection_order(client, signal.symbol, close_side, "TAKE_PROFIT_MARKET", round_to_tick(signal.tp, tick))
        placed_protections.append(int(tp_order["orderId"]))
    except BinanceError as exc:
        rollback_open(client, signal.symbol, close_side, adj_qty, placed_protections, exc, line_log)
        return None

    fee = adj_qty * fill_price * fee_rate
    set_state(db, "cash", round(cash - margin - fee, 4))
    note = json.dumps({"exchange_order_id": order_resp.get("orderId"), "sl_order_id": sl_order.get("orderId"),
                       "tp_order_id": tp_order.get("orderId"), "order_type": order_type,
                       "score": signal.score, "prob_up": signal.prob_up},
                      ensure_ascii=False)
    order = Order(
        symbol=signal.symbol,
        interval=signal.interval,
        side=signal.direction,
        qty=adj_qty,
        entry_price=round_price(fill_price),
        mark_price=round_price(fill_price),
        margin=round(margin, 4),
        leverage=leverage,
        sl=round_price(round_to_tick(signal.sl, tick)),
        tp=round_price(round_to_tick(signal.tp, tick)),
        status="OPEN",
        source="live",
        strategy=signal.strategy,
        pool=getattr(signal, "pool", "") or "",
        entry_fee=round(fee, 6),
        note=note,
    )
    db.add(order)
    db.commit()
    db.refresh(order)
    timestamp = getattr(signal, "ts", None)
    if timestamp is not None:
        set_state(db, f"last_entry:{signal.symbol}:{signal.interval}", timestamp)
    return order


def _prepare_account(db: Session, client: BinanceFuturesClient, symbol: str, leverage: int) -> None:
    try:
        client.set_position_side(False)
    except BinanceApiError as exc:
        log(db, "info", f"单向持仓设置跳过 {symbol}: code={exc.code}")
    try:
        client.set_margin_type(symbol, "ISOLATED")
    except BinanceApiError as exc:
        if exc.code != MARGIN_TYPE_UNCHANGED_CODE:
            log(db, "info", f"保证金模式设置跳过 {symbol}: code={exc.code}")
    client.set_leverage(symbol, leverage)  # 杠杆设置失败则中断开仓，避免风险与预期不符


def close_live(db: Session, order: Order, reason: str = "manual", fallback_price: float | None = None) -> Order:
    """实盘平仓：撤销两个保护单（忽略已触发 -2011），reduceOnly 市价平仓，用成交均价结算本地记账。"""
    if order.status != "OPEN":
        return order
    from .live_sync import _settle_exchange_closed  # 延迟导入避免循环依赖

    client = client_for(db)
    note = load_note(order)
    _cancel_protection(db, client, order, note)
    close_side = _close_side(order.side)
    try:
        resp = client.new_order(
            symbol=order.symbol, side=close_side, type="MARKET",
            quantity=format_amount(order.qty), reduceOnly="true", newOrderRespType="RESULT",
        )
    except BinanceApiError as exc:
        if exc.code == REDUCE_ONLY_REJECT_CODE:
            # 交易所仓位已不存在（保护单刚触发），按交易所已平仓结算
            log(db, "info", f"订单 #{order.id} reduceOnly 被拒绝(-2022)，交易所仓位已不存在")
            return _settle_exchange_closed(db, client, order, reason)
        log(db, "error", f"live 平仓下单失败 #{order.id}: {exc}")
        raise
    except BinanceError as exc:
        log(db, "error", f"live 平仓下单失败 #{order.id}: {exc}")
        raise
    price = float(resp.get("avgPrice") or 0) or fallback_price or order.mark_price or order.entry_price
    return settle_close(db, order, price, reason, slippage_rate=0.0)


def _cancel_protection(db: Session, client: BinanceFuturesClient, order: Order, note: dict) -> None:
    for field in ("sl_order_id", "tp_order_id"):
        protection_id = note.get(field)
        if not protection_id:
            continue
        try:
            client.cancel_order(order.symbol, int(protection_id))
        except BinanceApiError as exc:
            if exc.code not in CANCEL_IGNORE_CODES:
                log(db, "error", f"撤销保护单失败 #{order.id} {field}={protection_id}: code={exc.code}")
        except BinanceError as exc:
            log(db, "error", f"撤销保护单失败 #{order.id} {field}={protection_id}: {exc}")


def modify_live(db: Session, order: Order, sl: float | None = None, tp: float | None = None) -> Order:
    """实盘改单：撤旧保护单、挂新单；本地 sl/tp 与 note 同步更新。"""
    if order.status != "OPEN":
        return order
    client = client_for(db)
    note = load_note(order)
    close_side = _close_side(order.side)
    tick = _tick_size(client, order.symbol)
    if sl is not None and sl > 0:
        note["sl_order_id"] = _replace_protection(
            db, client, order, note, "sl_order_id", "STOP_MARKET", round_to_tick(sl, tick), close_side
        )
        order.sl = round_price(sl)
    if tp is not None and tp > 0:
        note["tp_order_id"] = _replace_protection(
            db, client, order, note, "tp_order_id", "TAKE_PROFIT_MARKET", round_to_tick(tp, tick), close_side
        )
        order.tp = round_price(tp)
    order.note = json.dumps(note, ensure_ascii=False)
    db.commit()
    return order


def _replace_protection(db: Session, client: BinanceFuturesClient, order: Order, note: dict,
                        field: str, order_type: str, stop_price: float, close_side: str) -> int:
    old_id = note.get(field)
    if old_id:
        try:
            client.cancel_order(order.symbol, int(old_id))
        except BinanceApiError as exc:
            if exc.code not in CANCEL_IGNORE_CODES:
                log(db, "error", f"调整订单 #{order.id} 撤销旧保护单失败: code={exc.code}")
    try:
        resp = _protection_order(client, order.symbol, close_side, order_type, stop_price)
    except BinanceError as exc:
        log(db, "error", f"调整订单 #{order.id} 挂新保护单失败 ({order_type}): {exc}")
        raise
    return int(resp["orderId"])


def _tick_size(client: BinanceFuturesClient, symbol: str) -> float:
    filters = symbol_filters(client.exchange_info(symbol), symbol)
    return float(filters.get("PRICE_FILTER", {}).get("tickSize", 0) or 0)
