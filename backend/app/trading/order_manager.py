from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..features.indicators import build_features
from ..market.data_service import MarketDataService
from ..models import JobLog, Order
from ..state import all_state, get_state, set_state
from ..strategy.bindings import resolve_order_params
from ..utils import dt_ms, interval_ms, now_ms, round_price
from . import live as live_exec
from .dispatch import close_position
from .risk import breakeven_stop, stop_exit, trailing_stop


def _initial_risk(db: Session, order: Order) -> float:
    """入场时的初始风险距离 |成交价 − 初始止损|，一次计算后随订单缓存。

    移动止损/保本止损会让 order.sl 单调变化，R 倍数必须以**入场时**的距离为尺，
    否则每抬高一次止损，R 的定义就漂移一次。首次调用时把当前值落库。
    """
    key = f"order_initial_risk:{order.id}"
    cached = get_state(db, key, 0)
    if isinstance(cached, (int, float)) and cached > 0:
        return float(cached)
    risk = abs(order.entry_price - (order.sl or 0.0))
    if risk > 0:
        set_state(db, key, risk)
    return risk


def manage_order(db: Session, order: Order, signal=None) -> dict:
    """Replay only unseen, fully post-entry candles, chronologically."""
    result = {"order_id": order.id, "actions": []}
    if order.status != "OPEN" or not order.created_at:
        return result
    # 订单必须用「它所属那条绑定」的参数管理，否则池级覆盖（止损宽度/时间止损）会失效
    params = resolve_order_params(
        db, order.strategy, order.interval, getattr(order, "pool", "")
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
    # 强回调的 EMA9 失守软出场：用上一根收盘与上一根 EMA9 判定，本根开盘平仓（与回测同口径）
    ema9_prev = pd.Series(np.nan, index=frame.index)
    if params.get("ema_exit"):
        ema9 = pd.Series(np.nan, index=frame.index)
        for _, segment in frame.groupby(groups):
            if len(segment) >= 30:
                ema9.loc[segment.index] = (
                    segment["close"].ewm(span=9, adjust=False).mean()
                )
        ema9_prev = ema9.shift(1)
    close_prev = frame["close"].shift(1)
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
        if hit is None and params.get("ema_exit") and side == 1:
            previous_ema9 = ema9_prev.loc[row.Index]
            previous_close = close_prev.loc[row.Index]
            if pd.notna(previous_ema9) and previous_close < previous_ema9:
                hit = (row.open, "ema_exit")
        if hit is None and replay_stops:
            hit = stop_exit(
                side, row.open, row.high, row.low, order.sl or 0.0, order.tp or 0.0
            )
        if hit:
            close_position(db, order, hit[0], hit[1])
            result["actions"].append(hit[1])
            return result
        be_trigger_r = float(params.get("be_trigger_r", 0.0) or 0.0)
        if be_trigger_r > 0:
            new_be = round_price(
                breakeven_stop(
                    side,
                    order.entry_price,
                    order.sl or 0.0,
                    row.high,
                    row.low,
                    be_trigger_r,
                    _initial_risk(db, order),
                )
            )
            if new_be != order.sl:
                if order.source == "live":
                    try:
                        live_exec.modify_live(db, order, sl=new_be)
                    except Exception as exc:
                        db.add(JobLog(
                            job="live_trading", level="error",
                            message=f"订单 #{order.id} 保本止损失败: {exc}",
                        ))
                        db.commit()
                        result["actions"].append("be_sl_failed")
                        order.mark_price = round_price(row.close)
                        cursor = ts
                        continue
                else:
                    order.sl = new_be
                result["actions"].append(f"be_sl={new_be}")
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
