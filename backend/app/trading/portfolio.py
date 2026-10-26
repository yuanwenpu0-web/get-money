from __future__ import annotations

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from ..market.data_service import MarketDataService
from ..models import EquitySnapshot, Order
from ..state import all_state, set_state
from ..utils import now_ms


def open_orders(db: Session) -> list[Order]:
    return db.query(Order).filter(Order.status == "OPEN").order_by(Order.created_at.asc()).all()


def position_unrealized(order: Order) -> float:
    mark = order.mark_price or order.entry_price
    sign = 1 if order.side == "LONG" else -1
    return (mark - order.entry_price) * order.qty * sign


def portfolio_snapshot(db: Session, update_marks: bool = True) -> dict:
    state = all_state(db)
    orders = open_orders(db)
    service = MarketDataService(db)
    unrealized = 0.0
    margin_total = 0.0
    positions = []

    for order in orders:
        if update_marks:
            frame = service.load_frame(order.symbol, order.interval, limit=1)
            if not frame.empty:
                order.mark_price = float(frame["close"].iloc[-1])
        pnl = position_unrealized(order)
        unrealized += pnl
        margin_total += order.margin
        mark = order.mark_price or order.entry_price
        positions.append(
            {
                "id": order.id,
                "symbol": order.symbol,
                "side": order.side,
                "qty": round(order.qty, 6),
                "entry_price": order.entry_price,
                "mark_price": mark,
                "margin": round(order.margin, 4),
                "leverage": order.leverage,
                "sl": order.sl,
                "tp": order.tp,
                "unrealized_pnl": round(pnl, 4),
                "unrealized_pct": round(pnl / order.margin * 100, 2) if order.margin else 0.0,
                "strategy": order.strategy,
                "opened_at": order.created_at.isoformat() if order.created_at else "",
            }
        )
    db.commit()

    cash = float(state.get("cash", 0.0))
    equity = cash + margin_total + unrealized
    closed_count, realized, wins = db.query(
        func.count(Order.id),
        func.coalesce(func.sum(Order.realized_pnl), 0.0),
        func.coalesce(func.sum(case((Order.realized_pnl > 0, 1), else_=0)), 0),
    ).filter(Order.status == "CLOSED").one()
    initial = float(state.get("initial_capital", 0) or 0)
    return {
        "cash": round(cash, 4),
        "margin_used": round(margin_total, 4),
        "unrealized_pnl": round(unrealized, 4),
        "equity": round(equity, 4),
        "realized_pnl": round(realized, 4),
        "total_return_pct": round((equity / initial - 1) * 100, 2) if initial > 0 else 0.0,
        "open_positions": len(orders),
        "closed_trades": closed_count,
        "win_rate": round(wins / closed_count * 100, 2) if closed_count else 0.0,
        "total_capital": round(float(state.get("total_capital", equity)), 4),
        "initial_capital": round(initial, 4),
        "positions": positions,
    }


def snapshot_equity(db: Session) -> EquitySnapshot:
    snapshot = portfolio_snapshot(db, update_marks=True)
    realized_cum = snapshot["realized_pnl"]
    row = EquitySnapshot(
        ts=now_ms(),
        cash=snapshot["cash"],
        equity=snapshot["equity"],
        unrealized_pnl=snapshot["unrealized_pnl"],
        realized_pnl_cum=realized_cum,
        open_positions=snapshot["open_positions"],
    )
    db.add(row)
    db.commit()

    # 盈利自动累加：以当前权益作为新的总资金基准
    if snapshot["equity"] > 0:
        set_state(db, "total_capital", round(snapshot["equity"], 4))
    return row


def adjust_capital(db: Session, amount: float, mode: str = "add") -> dict:
    current = portfolio_snapshot(db, update_marks=True)
    cash = current["cash"]
    equity = current["equity"]
    initial = float(all_state(db).get("initial_capital", equity) or equity)
    if mode == "set":
        delta = amount - equity
        cash += delta
        initial = amount
    elif mode == "withdraw":
        if amount > cash:
            raise ValueError("可提取现金不足")
        cash -= amount
        initial -= amount
    else:
        cash += amount
        initial += amount
    if cash < 0:
        raise ValueError("调整后可用现金为负")
    set_state(db, "cash", round(cash, 4))
    set_state(db, "initial_capital", round(max(initial, 0), 4))
    new_total = round(equity + (cash - current["cash"]), 4)
    set_state(db, "total_capital", new_total)
    return portfolio_snapshot(db, update_marks=True)
