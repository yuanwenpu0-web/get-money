"""Broker-emulator replay for Simple ZigZag Double (Pine v5).

The default path is a 1:1 Pine-style persistent stop-order replay.  An optional
``exit_policy`` exists only for research: it adds close-confirmed protective
exits to compare an unprotected reversal strategy with bounded-loss variants.
Passing no policy preserves the original behavior exactly.
"""
from __future__ import annotations

import math

import pandas as pd


def simulate_zigzag(data, signals, params, initial, fee, slippage, interval, exit_policy=None):
    """Return mark-to-market equity and completed trades for one symbol.

    Pine timing: orders are evaluated at bar close and become eligible from the
    next tick; the within-bar path follows TradingView's OHLC emulator.

    Research timing: a protective level derived from bar T's high/low can only
    close at T+1 open.  It deliberately cannot use a same-bar stop fill, since
    the source Pine strategy has not submitted such an order.  This makes the
    comparison causal and conservative.
    """
    del interval
    if data.empty:
        return pd.Series(dtype=float, index=data.index), []

    policy = {
        "lock_initial_stop": False,
        "breakeven_r": 0.0,
        "trail_activation_r": 0.0,
        "trail_r": 0.0,
        "time_stop_bars": 0,
    }
    if exit_policy:
        policy.update(exit_policy)

    rows = list(data.itertuples())
    decisions = list(signals.reindex(data.index).itertuples())
    balance, position, pending, market_close = float(initial), None, {}, False
    values, trades = [], []
    lots = {1: None, -1: None}
    previous_size = 0.0

    def mark(price):
        return balance + ((price - position["entry"]) * position["qty"] * position["side"] if position else 0.0)

    def close_at(price, stamp, reason):
        nonlocal balance, position
        p = position
        fill = price * (1 - p["side"] * slippage)
        gross = (fill - p["entry"]) * p["qty"] * p["side"]
        exit_fee = abs(fill * p["qty"]) * fee
        balance += gross - exit_fee
        net = gross - p["fee"] - exit_fee
        margin = abs(p["entry"] * p["qty"])
        trades.append(dict(
            entry_time=str(p["time"]), exit_time=str(stamp), position_id=str(p["time"]),
            side="long" if p["side"] == 1 else "short", entry_price=p["entry"],
            exit_price=fill, size=p["qty"], pnl=net,
            return_pct=net / margin * 100 if margin else 0.0,
            fees=p["fee"] + exit_fee, status=reason, leverage=1, margin=margin,
        ))
        position = None

    def fill_order(order_id, price, stamp, bar_index):
        nonlocal balance, position
        order = pending.pop(order_id, None)
        if order is None:
            return
        side = order["side"]
        if order["kind"] == "exit":
            if position is not None and position["side"] == -side:
                close_at(price, stamp, "stop_loss")
            return
        if position is not None and position["side"] == side:
            return
        if position is not None:
            close_at(price, stamp, "signal_reverse")
        fill = price * (1 + side * slippage)
        qty = order["qty"] if order["qty"] is not None else max(balance, 0.0) / fill
        if not math.isfinite(qty) or qty <= 0 or fill <= 0:
            return
        entry_fee = abs(fill * qty) * fee
        balance -= entry_fee
        initial_stop = order.get("initial_stop")
        risk = abs(fill - initial_stop) if initial_stop is not None and (fill - initial_stop) * side > 0 else 0.0
        position = dict(
            side=side, qty=qty, entry=fill, fee=entry_fee, time=stamp, index=bar_index,
            initial_stop=initial_stop, risk=risk,
            policy_stop=initial_stop if policy["lock_initial_stop"] and risk > 0 else None,
            favorable=fill,
        )

    def crossed_at(order, price):
        return price >= order["price"] if order["side"] == 1 else price <= order["price"]

    def active(order):
        return order["kind"] != "exit" or (position is not None and position["side"] == -order["side"])

    def process_activated_exit(price, stamp, bar_index):
        for order_id in list(pending):
            order = pending.get(order_id)
            if order is not None and order["kind"] == "exit" and active(order) and crossed_at(order, price):
                fill_order(order_id, price, stamp, bar_index)

    def process_at_open(price, stamp, bar_index):
        for order_id in list(pending):
            order = pending.get(order_id)
            if order is not None and active(order) and (order["marketable"] or crossed_at(order, price)):
                fill_order(order_id, price, stamp, bar_index)
                process_activated_exit(price, stamp, bar_index)

    def process_segment(start, end, stamp, bar_index):
        if start == end:
            return
        up = end > start
        while True:
            eligible = [
                (order["price"], key) for key, order in pending.items()
                if active(order) and (order["side"] == 1) == up
                and (start < order["price"] <= end if up else end <= order["price"] < start)
            ]
            if not eligible:
                return
            level, order_id = min(eligible, key=lambda x: x[0]) if up else max(eligible, key=lambda x: x[0])
            fill_order(order_id, level, stamp, bar_index)
            process_activated_exit(level, stamp, bar_index)
            start = level
            for other in list(pending):
                order = pending.get(other)
                if order is not None and active(order) and order["side"] == (1 if up else -1) and order["price"] == level:
                    fill_order(other, level, stamp, bar_index)

    def submit(order_id, kind, side, level, qty, current_close, initial_stop=None):
        if not math.isfinite(level) or level <= 0:
            return
        if kind == "entry" and position is not None and position["side"] == side:
            return
        pending[order_id] = dict(
            kind=kind, side=side, price=level, qty=qty, initial_stop=initial_stop,
            marketable=(current_close >= level if side == 1 else current_close <= level),
        )

    def close_confirmed_policy_exit(row, bar_index):
        """Exit only from a decision that was known at the prior close."""
        if position is None:
            return False
        p = position
        held = bar_index - p["index"]
        if policy["time_stop_bars"] and held >= int(policy["time_stop_bars"]):
            close_at(float(row.open), row.Index, "time_stop")
            return True
        stop = p.get("policy_stop")
        previous_close = float(rows[bar_index - 1].close) if bar_index else float(row.open)
        if stop is not None and (previous_close - stop) * p["side"] <= 0:
            close_at(float(row.open), row.Index, "policy_stop")
            return True
        return False

    def update_policy_after_close(high, low):
        """Move protective stop monotonically; it first becomes active next bar."""
        if position is None or position["risk"] <= 0:
            return
        p = position
        side, risk = p["side"], p["risk"]
        extreme = high if side == 1 else low
        p["favorable"] = max(p["favorable"], extreme) if side == 1 else min(p["favorable"], extreme)
        profit = (p["favorable"] - p["entry"]) * side
        candidate = p.get("policy_stop")
        if policy["breakeven_r"] and profit >= float(policy["breakeven_r"]) * risk:
            candidate = p["entry"] if candidate is None else (max(candidate, p["entry"]) if side == 1 else min(candidate, p["entry"]))
        if policy["trail_activation_r"] and policy["trail_r"] and profit >= float(policy["trail_activation_r"]) * risk:
            trail = p["favorable"] - side * float(policy["trail_r"]) * risk
            candidate = trail if candidate is None else (max(candidate, trail) if side == 1 else min(candidate, trail))
        p["policy_stop"] = candidate

    for bar_index, (row, signal) in enumerate(zip(rows, decisions)):
        stamp = row.Index
        open_price, high, low, close = float(row.open), float(row.high), float(row.low), float(row.close)
        if market_close:
            if position is not None:
                close_at(open_price, stamp, "time_end")
            market_close = False

        # Research policy has priority at the open, but an original Pine replay
        # never enters this branch because its policy is inert.
        closed_by_policy = close_confirmed_policy_exit(row, bar_index)
        if not closed_by_policy:
            process_at_open(open_price, stamp, bar_index)
        path = [open_price, high, low, close] if abs(open_price - high) < abs(open_price - low) else [open_price, low, high, close]
        for start, end in zip(path, path[1:]):
            process_segment(start, end, stamp, bar_index)

        size = position["side"] * position["qty"] if position else 0.0
        if params["fixed_lot"]:
            lots = {1: float(params["lot_long"]), -1: float(params["lot_short"])}
        elif size != previous_size:
            equity = mark(close)
            lots = {1: equity / close * float(params["lot_long"]) / 100, -1: equity / close * float(params["lot_short"]) / 100}
        previous_size = size

        long_level = float(getattr(signal, "zz_long_level", float("nan")))
        short_level = float(getattr(signal, "zz_short_level", float("nan")))
        if getattr(signal, "zz_long_entry", False):
            submit("Long", "entry", 1, long_level, lots[1], close, initial_stop=short_level)
        if getattr(signal, "zz_short_entry", False):
            submit("Short", "entry", -1, short_level, lots[-1], close, initial_stop=long_level)
        if not params["use_long"]:
            pending.pop("Long", None)
            if (position is not None and position["side"] == -1) or "Short" in pending:
                submit("Exit Short", "exit", 1, long_level, None, close)
        if not params["use_short"]:
            pending.pop("Short", None)
            if (position is not None and position["side"] == 1) or "Long" in pending:
                submit("Exit Long", "exit", -1, short_level, None, close)
        if int(pd.Timestamp(stamp).timestamp() * 1000) > int(params["timestamp_end"]):
            market_close = True
            pending.clear()
        update_policy_after_close(high, low)
        values.append(mark(close))

    equity = pd.Series(values, index=data.index, dtype=float)
    equity.attrs["open_position"] = (
        dict(
            side="long" if position["side"] == 1 else "short", qty=position["qty"],
            entry_price=position["entry"], entry_time=str(position["time"]), entry_fee=position["fee"],
            unrealized_pnl=(float(rows[-1].close) - position["entry"]) * position["qty"] * position["side"],
        ) if position else None
    )
    return equity, trades
