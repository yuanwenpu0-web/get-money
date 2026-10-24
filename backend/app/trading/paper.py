"""模拟撮合。前期所有订单只在本地成交，不触达币安真实资金。"""

from __future__ import annotations

from sqlalchemy.orm import Session

from ..models import Order
from ..state import all_state, set_state
from ..utils import round_price
from .portfolio import open_orders, position_unrealized

EPS = 1e-12


def _fee_rate(state: dict) -> float:
    return float(state.get("fee_rate", 0.0004))


def _slippage(state: dict) -> float:
    return float(state.get("slippage", 0.0002))


def open_position(db: Session, signal, equity: float | None = None) -> Order | None:
    if signal.direction not in ("LONG", "SHORT"):
        return None
    state = all_state(db)
    existing = [o for o in open_orders(db) if o.symbol == signal.symbol]
    if existing:
        return None

    from .portfolio import portfolio_snapshot

    snapshot = portfolio_snapshot(db, update_marks=False)
    equity = equity if equity is not None else snapshot["equity"]
    cash = snapshot["cash"]
    position_pct = float(signal.position_pct or 0)
    if position_pct <= 0 or equity <= 0:
        return None

    margin = equity * position_pct
    fee_rate = _fee_rate(state)
    slip = _slippage(state)
    max_cash = cash / (1 + fee_rate)
    margin = min(margin, max_cash * 0.98)
    if margin < 10:
        return None

    entry = signal.price * (1 + slip) if signal.direction == "LONG" else signal.price * (1 - slip)
    leverage = max(1, int(signal.leverage))
    notional = margin * leverage
    qty = notional / entry
    fee = notional * fee_rate
    if cash < margin + fee:
        margin = (cash - fee) / 1.0
        if margin < 10:
            return None
        notional = margin * leverage
        qty = notional / entry
        fee = notional * fee_rate

    set_state(db, "cash", round(cash - margin - fee, 4))
    order = Order(
        symbol=signal.symbol,
        interval=signal.interval,
        side=signal.direction,
        qty=qty,
        entry_price=round_price(entry),
        mark_price=round_price(signal.price),
        margin=round(margin, 4),
        leverage=leverage,
        sl=signal.sl,
        tp=signal.tp,
        status="OPEN",
        source="paper",
        strategy=signal.strategy,
        entry_fee=round(fee, 6),
        note=f"score={signal.score} prob_up={signal.prob_up}",
    )
    db.add(order)
    db.commit()
    db.refresh(order)
    return order


def close_position(db: Session, order: Order, price: float, reason: str = "manual") -> Order:
    if order.status != "OPEN":
        return order
    state = all_state(db)
    slip = _slippage(state)
    fee_rate = _fee_rate(state)
    exit_price = price * (1 - slip) if order.side == "LONG" else price * (1 + slip)
    sign = 1 if order.side == "LONG" else -1
    gross = (exit_price - order.entry_price) * order.qty * sign
    exit_notional = exit_price * order.qty
    exit_fee = exit_notional * fee_rate
    net = gross - order.entry_fee - exit_fee

    cash = float(state.get("cash", 0.0))
    set_state(db, "cash", round(cash + order.margin + gross - exit_fee, 4))

    order.status = "CLOSED"
    order.close_price = round_price(exit_price)
    order.mark_price = round_price(exit_price)
    order.exit_fee = round(exit_fee, 6)
    order.realized_pnl = round(net, 6)
    order.close_reason = reason
    from ..models import utcnow

    order.closed_at = utcnow()
    db.commit()
    db.refresh(order)
    return order


def modify_order(order: Order, sl: float | None = None, tp: float | None = None) -> Order:
    if sl is not None and sl > 0:
        order.sl = round_price(sl)
    if tp is not None and tp > 0:
        order.tp = round_price(tp)
    return order


def unrealized(order: Order) -> float:
    return position_unrealized(order)
