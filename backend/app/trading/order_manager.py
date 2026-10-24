from __future__ import annotations

from sqlalchemy.orm import Session

from ..features.indicators import build_features
from ..market.data_service import MarketDataService
from ..models import Order
from ..state import all_state
from ..utils import dt_ms, interval_ms, now_ms, round_price
from .paper import close_position, modify_order


def latest_closed_frame(db: Session, symbol: str, interval: str, limit: int = 3):
    service = MarketDataService(db)
    frame = service.load_frame(symbol, interval, limit=limit)
    if frame.empty:
        return None
    last_open = int(frame.index[-1].timestamp() * 1000)
    if last_open + interval_ms(interval) > now_ms() + 1000:
        frame = frame.iloc[:-1]
    return frame if not frame.empty else None


def _atr_value(db: Session, symbol: str, interval: str, price: float) -> float:
    frame = MarketDataService(db).load_frame(symbol, interval, limit=80)
    if len(frame) < 30:
        return price * 0.002
    data = build_features(frame, with_label=False).dropna(subset=["atr_pct"])
    if data.empty:
        return price * 0.002
    return float(data["atr_pct"].iloc[-1]) * price


def manage_order(db: Session, order: Order, signal=None) -> dict:
    """检查止损/止盈、移动止损、超时与信号弱化，返回本次调整摘要。"""
    state = all_state(db)
    actions = []
    frame = latest_closed_frame(db, order.symbol, order.interval, limit=3)
    if frame is None or order.status != "OPEN":
        return {"order_id": order.id, "actions": actions}

    last = frame.iloc[-1]
    high, low, close = float(last["high"]), float(last["low"]), float(last["close"])

    # 1) 止损/止盈触发（同一根K线同时触发时按止损优先，保守处理）
    sl = float(order.sl or 0)
    tp = float(order.tp or 0)
    if order.side == "LONG":
        if sl > 0 and low <= sl:
            close_position(db, order, sl, "stop_loss")
            return {"order_id": order.id, "actions": ["stop_loss"]}
        if tp > 0 and high >= tp:
            close_position(db, order, tp, "take_profit")
            return {"order_id": order.id, "actions": ["take_profit"]}
    else:
        if sl > 0 and high >= sl:
            close_position(db, order, sl, "stop_loss")
            return {"order_id": order.id, "actions": ["stop_loss"]}
        if tp > 0 and low <= tp:
            close_position(db, order, tp, "take_profit")
            return {"order_id": order.id, "actions": ["take_profit"]}

    # 2) 超时平仓
    max_hold = int(state.get("max_hold_bars", 120))
    if max_hold > 0 and order.created_at:
        held_ms = now_ms() - dt_ms(order.created_at)
        if held_ms > max_hold * interval_ms(order.interval):
            close_position(db, order, close, "time_stop")
            return {"order_id": order.id, "actions": ["time_stop"]}

    # 3) 信号反向或显著弱化（受最小持仓K线数约束，避免高频反复）
    min_hold = int(state.get("min_hold_bars", 30))
    held_bars = 0
    if order.created_at:
        held_bars = (now_ms() - dt_ms(order.created_at)) / interval_ms(order.interval)
    can_reverse = min_hold <= 0 or held_bars >= min_hold
    if signal is not None and can_reverse:
        if signal.direction in ("LONG", "SHORT") and signal.direction != order.side and signal.score >= float(state.get("min_score", 0.55)):
            close_position(db, order, close, "signal_reverse")
            return {"order_id": order.id, "actions": ["signal_reverse"]}
        if signal.direction == "FLAT" and signal.score < 0.5:
            close_position(db, order, close, "signal_weak")
            return {"order_id": order.id, "actions": ["signal_weak"]}

    # 4) 移动止损 + 趋势中扩展止盈
    atr = _atr_value(db, order.symbol, order.interval, close)
    trail_mult = float(state.get("trailing_atr_mult", 1.2))
    tp_mult = float(state.get("tp_atr_mult", 2.5))
    profit = (close - order.entry_price) * (1 if order.side == "LONG" else -1)

    if profit > atr * 0.8:
        if order.side == "LONG":
            new_sl = max(order.sl, close - atr * trail_mult)
            if new_sl > order.sl + 1e-9:
                modify_order(order, sl=round_price(new_sl))
                actions.append(f"trail_sl={round_price(new_sl)}")
            if signal is not None and signal.score >= float(state.get("heavy_score", 0.75)):
                new_tp = max(order.tp, close + atr * tp_mult)
                if new_tp > order.tp + 1e-9:
                    modify_order(order, tp=round_price(new_tp))
                    actions.append(f"extend_tp={round_price(new_tp)}")
        else:
            new_sl = order.sl if order.sl > 0 else close + atr * trail_mult
            candidate = min(order.sl if order.sl > 0 else new_sl, close + atr * trail_mult)
            if order.sl == 0 or candidate < order.sl - 1e-9:
                modify_order(order, sl=round_price(candidate))
                actions.append(f"trail_sl={round_price(candidate)}")
            if signal is not None and signal.score >= float(state.get("heavy_score", 0.75)):
                base_tp = order.tp if order.tp > 0 else close - atr * tp_mult
                new_tp = min(base_tp, close - atr * tp_mult)
                if order.tp == 0 or new_tp < order.tp - 1e-9:
                    modify_order(order, tp=round_price(new_tp))
                    actions.append(f"extend_tp={round_price(new_tp)}")

    order.mark_price = round_price(close)
    db.commit()
    return {"order_id": order.id, "actions": actions}
