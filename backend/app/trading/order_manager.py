from __future__ import annotations

import json

import pandas as pd
from sqlalchemy.orm import Session

from ..features.indicators import build_features
from ..market.data_service import MarketDataService
from ..models import JobLog, Order, StrategyConfig
from ..state import all_state, get_state, set_state
from ..strategy.rules import resolve_params
from ..utils import dt_ms, interval_ms, now_ms, round_price
from . import live as live_exec
from .dispatch import close_position
from .risk import stop_exit, trailing_stop


def manage_order(db: Session, order: Order, signal=None) -> dict:
    """Replay only unseen, fully post-entry candles, chronologically."""
    result = {"order_id": order.id, "actions": []}
    if order.status != "OPEN" or not order.created_at:
        return result
    config = (
        db.query(StrategyConfig).filter(StrategyConfig.code == order.strategy).first()
    )
    params = resolve_params(
        order.strategy,
        all_state(db),
        json.loads(config.params or "{}") if config else None,
    )
    step, created = interval_ms(order.interval), dt_ms(order.created_at)
    key = f"order_cursor:{order.id}"
    cursor = get_state(db, key, 0)
    # Load warmup plus every unseen candle, including after a scheduler interruption.
    start = max(created, cursor) - 400 * step
    frame = MarketDataService(db).load_frame(
        order.symbol, order.interval, start_ms=start
    )
    if frame.empty:
        return result
    stamps = frame.index.as_unit("ms").asi8
    frame = frame.loc[stamps + step <= now_ms()]
    if frame.empty:
        return result
    stamps = frame.index.as_unit("ms").asi8
    unseen = frame.loc[(stamps >= created) & (stamps > cursor)]
    if unseen.empty:
        return result
    # Small frames may occur immediately after import; fixed stops still work.
    atr_values = pd.Series(0.0, index=frame.index)
    groups = (
        frame.index.to_series().diff().dt.total_seconds().mul(1000).ne(step).cumsum()
    )
    for _, segment in frame.groupby(groups):
        if len(segment) >= 60:
            atr_values.loc[segment.index] = (
                build_features(segment)["atr_pct"] * segment["close"]
            )
    side = 1 if order.side == "LONG" else -1
    # live 订单的固定止损止盈由交易所侧保护单执行，本地不做K线止损回放，
    # 否则会在交易所已触发后重复平仓造成不一致；本地只保留移动止损/超时/信号反转。
    replay_stops = order.source != "live"
    for row in unseen.itertuples():
        ts = int(row.Index.timestamp() * 1000)
        held_bars = (ts - created) / step
        hit = None
        if replay_stops:
            hit = stop_exit(
                side, row.open, row.open, row.open, order.sl or 0.0, order.tp or 0.0
            )
            if hit is None and cursor and ts != cursor + step:
                hit = (row.open, "data_gap")
        if hit is None and params["max_hold_bars"] > 0 and held_bars >= params["max_hold_bars"]:
            hit = (row.open, "time_stop")
        if hit is None and replay_stops:
            hit = stop_exit(
                side, row.open, row.high, row.low, order.sl or 0.0, order.tp or 0.0
            )
        if hit:
            close_position(db, order, hit[0], hit[1])
            result["actions"].append(hit[1])
            return result
        atr = float(atr_values.loc[row.Index])
        if pd.notna(atr) and atr > 0:
            new_sl = round_price(
                trailing_stop(
                    side, order.entry_price, order.sl or 0.0, row.close, atr, params
                )
            )
            if new_sl != order.sl:
                if order.source == "live":
                    try:
                        live_exec.modify_live(db, order, sl=new_sl)
                    except Exception as exc:
                        db.add(JobLog(
                            job="live_trading", level="error",
                            message=f"订单 #{order.id} 移动止损失败: {exc}",
                        ))
                        db.commit()
                        result["actions"].append("trail_sl_failed")
                        continue
                else:
                    order.sl = new_sl
                result["actions"].append(f"trail_sl={new_sl}")
        order.mark_price = round_price(row.close)
        cursor = ts
    # Only the order's own strategy may reverse it; no flat signal forces churn.
    if signal is not None and getattr(signal, "ts", 0) == cursor:
        held_bars = (cursor + step - created) / step
        if (
            held_bars >= params["min_hold_bars"]
            and signal.direction in ("LONG", "SHORT")
            and signal.direction != order.side
        ):
            close_position(db, order, order.mark_price, "signal_reverse")
            result["actions"].append("signal_reverse")
    db.commit()
    set_state(db, key, cursor)
    return result
