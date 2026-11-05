"""Isolated realized outcome of one strategy signal under production exit rules.

Unlike `simulate`, every signal gets its own hypothetical position. This avoids
label censoring where a signal is skipped simply because another position is open.
The bar ordering mirrors app/backtest/execution.py for Supertrend scaled targets:
signal close -> next open fill -> open stop/target checks -> reverse exit ->
intrabar stop (priority) -> intrabar resting targets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..trading.risk import position_size

CASH = 10_000.0


def isolated_trade(
    frame: pd.DataFrame,
    signals: pd.DataFrame,
    i: int,
    params: dict,
    fee: float,
    slippage: float,
    step_ms: int,
) -> dict | None:
    opens = frame.open.to_numpy(float)
    highs = frame.high.to_numpy(float)
    lows = frame.low.to_numpy(float)
    closes = frame.close.to_numpy(float)
    times = frame.index
    directions = signals.direction.to_numpy(int)
    side = int(directions[i])
    n = len(frame)
    if side == 0 or i + 1 >= n:
        return None
    # A data gap on the entry bar blocks the fill in simulate().
    gap_entry = (times[i + 1] - times[i]).value // 1_000_000 != step_ms
    if gap_entry:
        return None
    sig = signals.iloc[i]
    entry = opens[i + 1] * (1 + side * slippage)
    signal_close = closes[i]
    stop = signal_close * (1 - side * float(sig.sl_pct))
    risk = abs(signal_close - stop)
    levels = []
    for r_key, frac_key, name in (
        ("tp1_r", "tp1_fraction", "take_profit_1"),
        ("tp2_r", "tp2_fraction", "take_profit_2"),
        ("target_r", "tp3_fraction", "take_profit_3"),
    ):
        frac = float(params[frac_key])
        if frac > 0:
            levels.append([signal_close + side * risk * float(params[r_key]), frac, name, False])
    first_target = levels[0][0]
    if (entry - stop) * side <= 0 or (first_target - entry) * side <= 0:
        return None
    margin, qty = position_size(
        CASH, CASH, float(sig.position_pct), int(sig.leverage), entry, stop, side,
        fee, slippage, float(params["risk_per_trade_pct"]),
    )
    if margin < 10 or qty <= 0:
        return None

    initial_margin = margin
    net_pnl = -entry * qty * fee
    total_fees = entry * qty * fee
    entry_index = i + 1

    def close(qty_left: float, margin_left: float, price: float, status: str) -> dict:
        fill = price * (1 - side * slippage)
        exit_fee = fill * qty_left * fee
        pnl = net_pnl + (fill - entry) * qty_left * side - exit_fee
        return {
            "entry_time": str(times[entry_index]),
            "exit_time": str(times[j]),
            "side": side,
            "status": status,
            "hold_bars": j - entry_index,
            "pnl": float(pnl),
            "fees": float(total_fees + exit_fee),
            "margin": float(initial_margin),
            "return_pct": float(pnl / initial_margin * 100),
            "censored": status == "close_at_end",
        }

    qty_left, margin_left = qty, margin
    for j in range(entry_index, n):
        if j > entry_index:
            gap = (times[j] - times[j - 1]).value // 1_000_000 != step_ms
        else:
            gap = False
        price_open, high, low = opens[j], highs[j], lows[j]
        # 1) Stop gapped through at the open.
        if (price_open - stop) * side <= 0:
            return close(qty_left, margin_left, price_open, "stop_loss")
        # 2) A gap forces liquidation at the open (mirrors simulate's ordering).
        if gap:
            return close(qty_left, margin_left, price_open, "data_gap")
        # 3) Resting targets already crossed at the open fill at their limit prices.
        for level in levels:
            if level[3]:
                continue
            reached = price_open >= level[0] if side == 1 else price_open <= level[0]
            if not reached:
                continue
            level[3] = True
            fill_qty = min(qty * level[1], qty_left)
            fill = level[0] * (1 - side * slippage)
            exit_fee = fill * fill_qty * fee
            net_pnl += (fill - entry) * fill_qty * side - exit_fee
            total_fees += exit_fee
            qty_left -= fill_qty
            margin_left -= initial_margin * level[1]
            if qty_left <= 1e-10:
                return close(0.0, 0.0, fill, level[2])
        # 4) A previous-bar opposite signal exits at this open.
        held = j - entry_index
        if held >= int(params["min_hold_bars"]) and int(directions[j - 1]) == -side:
            return close(qty_left, margin_left, price_open, "signal_reverse")
        # 4) Intrabar stop has priority over targets (same convention as simulate).
        stopped = low <= stop if side == 1 else high >= stop
        if stopped:
            return close(qty_left, margin_left, stop, "stop_loss")
        for level in levels:
            if level[3]:
                continue
            reached = high >= level[0] if side == 1 else low <= level[0]
            if not reached:
                continue
            level[3] = True
            fill_qty = min(qty * level[1], qty_left)
            fill = level[0] * (1 - side * slippage)
            exit_fee = fill * fill_qty * fee
            net_pnl += (fill - entry) * fill_qty * side - exit_fee
            total_fees += exit_fee
            qty_left -= fill_qty
            margin_left -= initial_margin * level[1]
            if qty_left <= 1e-10:
                return close(0.0, 0.0, fill, level[2])
    return close(qty_left, margin_left, closes[-1], "close_at_end")
