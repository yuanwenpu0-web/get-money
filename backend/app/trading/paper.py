"""模拟撮合。前期所有订单只在本地成交，不触达币安真实资金。"""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from ..models import Order
from ..state import all_state, get_state, set_state
from ..strategy.bindings import resolve_order_params
from ..utils import dt_ms, interval_ms, now_ms, round_price
from .portfolio import open_orders, position_unrealized
from .risk import position_size
from .risk_halt import check_after_close

EPS = 1e-12


def _fee_rate(state: dict) -> float:
    return float(state.get("fee_rate", 0.0004))


def _slippage(state: dict) -> float:
    return float(state.get("slippage", 0.0002))


def entry_gates(db: Session, signal, params: dict, ignore_cooldown: bool = False) -> bool:
    """公共入场闸门：方向、信号时效、重复信号、冷却、已有持仓。paper 与 live 共用。"""
    if signal.direction not in ("LONG", "SHORT"):
        return False
    timestamp = getattr(signal, 'ts', None)
    step = interval_ms(signal.interval)
    if timestamp is not None and (timestamp + step > now_ms() or now_ms() - timestamp - step > params['max_signal_age_bars'] * step):
        return False
    last_entry_key = f'last_entry:{signal.symbol}:{signal.interval}'
    if timestamp is not None and timestamp <= get_state(db, last_entry_key, 0):
        return False
    last_close = db.query(Order).filter(Order.symbol == signal.symbol, Order.status == 'CLOSED').order_by(Order.closed_at.desc()).first()
    if not ignore_cooldown and last_close and last_close.closed_at:
        if now_ms() - dt_ms(last_close.closed_at) < params['cooldown_bars'] * step:
            return False
    if any(o.symbol == signal.symbol for o in open_orders(db)):
        return False
    return True


def open_position(db: Session, signal, equity: float | None = None, ignore_cooldown: bool = False) -> Order | None:
    state = all_state(db)
    # 信号自带参数（来自「策略 × 池」绑定）；否则按订单归属反查，最后才回落策略配置
    params = getattr(signal, "params", None) or resolve_order_params(
        db,
        signal.strategy,
        signal.interval,
        getattr(signal, "pool", ""),
        symbol=getattr(signal, "symbol", ""),
    )
    if not entry_gates(db, signal, params, ignore_cooldown):
        return None

    from .portfolio import portfolio_snapshot

    snapshot = portfolio_snapshot(db, update_marks=False)
    equity = equity if equity is not None else snapshot["equity"]
    cash = snapshot["cash"]
    position_pct = float(signal.position_pct or 0)
    if position_pct <= 0 or equity <= 0:
        return None

    fee_rate = _fee_rate(state)
    slip = _slippage(state)
    side = 1 if signal.direction == 'LONG' else -1
    entry = round_price(signal.price * (1 + side * slip))
    leverage = min(max(1, int(signal.leverage)), int(params['max_leverage']))
    # tp<=0 表示策略无止盈（反手型策略靠反向信号退出），只校验止损方向
    if (entry - signal.sl) * side <= 0 or (signal.tp and (signal.tp - entry) * side <= 0):
        return None
    margin, qty = position_size(equity, cash, min(position_pct, params['max_position_pct']), leverage,
                                entry, signal.sl, side, fee_rate, slip, params['risk_per_trade_pct'])
    if margin < 10:
        return None
    fee = qty * entry * fee_rate

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
        pool=getattr(signal, "pool", "") or "",
        entry_fee=round(fee, 6),
        note=f"score={signal.score} prob_up={signal.prob_up}",
    )
    db.add(order)
    db.commit()
    db.refresh(order)
    timestamp = getattr(signal, 'ts', None)
    if timestamp is not None:
        set_state(db, f'last_entry:{signal.symbol}:{signal.interval}', timestamp)
    return order


def close_position(db: Session, order: Order, price: float, reason: str = "manual", slippage_rate: float | None = None) -> Order:
    """本地记账结算。slippage_rate 默认取全局滑点；live 模式传入 0 用真实成交均价结算。"""
    if order.status != "OPEN":
        return order
    state = all_state(db)
    slip = _slippage(state) if slippage_rate is None else float(slippage_rate)
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
    # 账户级熔断检查：每笔平仓记账后评估回撤/连续亏损，触发即冻结后续开仓
    check_after_close(db)
    return order


def modify_order(order: Order, sl: float | None = None, tp: float | None = None) -> Order:
    if sl is not None and sl > 0:
        order.sl = round_price(sl)
    if tp is not None and tp > 0:
        order.tp = round_price(tp)
    return order


def unrealized(order: Order) -> float:
    return position_unrealized(order)
