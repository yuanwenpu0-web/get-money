"""Minimal close-decision/next-open execution model for converted strategies.

This is not TradingView's broker emulator: it supports fixed-size market
orders only, with no intrabar fills, funding, margin or liquidation model.
"""
from __future__ import annotations

import math

import pandas as pd


def run_backtest(
    candles: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    initial_equity: float = 10_000.0,
    quantity: float = 1.0,
    fee_rate: float = 0.0004,
    slippage: float = 0.0,
    interval_ms: int | None = 60_000,
) -> dict:
    """Simulate market orders from confirmed-bar decisions at the next open.

    `open_time` is the Binance-style candle open timestamp in milliseconds.
    Quantity is in base-asset units. The final bar's decisions remain pending;
    open positions are marked to the last close rather than forcibly closed.
    """
    values = (initial_equity, quantity, fee_rate, slippage)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("backtest settings must be finite")
    if initial_equity <= 0 or quantity <= 0 or not 0 <= fee_rate < 1 or not 0 <= slippage < 1:
        raise ValueError("invalid equity, quantity, fee rate or slippage")
    if interval_ms is not None and (isinstance(interval_ms, bool) or not isinstance(interval_ms, int) or interval_ms <= 0):
        raise ValueError("interval_ms must be a positive integer or None")
    required = {"open_time", "open", "high", "low", "close", "volume"}
    columns = ("long_entry", "short_entry", "long_close", "short_close")
    if not required.issubset(candles.columns) or not set(columns).issubset(signals.columns):
        raise ValueError("missing candle or signal columns")
    if candles.empty or not candles.index.equals(signals.index):
        raise ValueError("candles and signals must have the same non-empty index")
    times = pd.to_numeric(candles.open_time, errors="coerce")
    prices = candles[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    if not all(math.isfinite(float(x)) and float(x).is_integer() for x in times):
        raise ValueError("open_time must contain finite millisecond timestamps")
    if not all(math.isfinite(float(x)) for x in prices.to_numpy().flat):
        raise ValueError("candle values must be finite")
    if (prices[["open", "high", "low", "close"]] <= 0).any().any() or (prices.volume < 0).any():
        raise ValueError("prices must be positive and volume non-negative")
    if ((prices.high < prices[["open", "close", "low"]].max(axis=1)) | (prices.low > prices[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("invalid candle high/low")
    diffs = times.diff().iloc[1:]
    if (diffs <= 0).any() or (interval_ms is not None and (diffs != interval_ms).any()):
        raise ValueError("candles must be ordered, unique and continuous")
    for col in columns:
        if not signals[col].map(lambda x: isinstance(x, (bool,)) or type(x).__name__ == "bool_").all():
            raise ValueError(f"{col} must contain only boolean decisions")
    if (signals.loc[:, columns].sum(axis=1) > 1).any():
        raise ValueError("simultaneous order decisions are ambiguous")

    balance = float(initial_equity)
    position = None
    trades, equity = [], []
    fee_total = 0.0

    def close_position(price: float, timestamp: int, reason: str) -> None:
        nonlocal position, balance, fee_total
        side = position["side"]
        fill = price * (1 - side * slippage)
        exit_fee = fill * quantity * fee_rate
        gross = (fill - position["entry_price"]) * quantity * side
        balance += gross - exit_fee
        fee_total += exit_fee
        trades.append({
            "side": "long" if side == 1 else "short", "quantity": quantity,
            "entry_time": position["entry_time"], "exit_time": timestamp,
            "entry_price": position["entry_price"], "exit_price": fill,
            "gross_pnl": gross, "fees": position["entry_fee"] + exit_fee,
            "pnl": gross - position["entry_fee"] - exit_fee, "reason": reason,
        })
        position = None

    for i in range(len(candles)):
        price = float(prices.open.iloc[i])
        timestamp = int(times.iloc[i])
        if i:
            decision = signals.iloc[i - 1]
            target = 1 if decision.long_entry else -1 if decision.short_entry else 0
            if position is not None:
                side = position["side"]
                close = bool(decision.long_close if side == 1 else decision.short_close)
                if target == -side or close:
                    close_position(price, timestamp, "reverse" if target == -side else "signal_close")
            if position is None and target:
                fill = price * (1 + target * slippage)
                entry_fee = fill * quantity * fee_rate
                balance -= entry_fee
                fee_total += entry_fee
                position = {"side": target, "quantity": quantity, "entry_time": timestamp,
                            "entry_price": fill, "entry_fee": entry_fee}
        unrealized = 0.0 if position is None else (float(prices.close.iloc[i]) - position["entry_price"]) * quantity * position["side"]
        equity.append({"time": timestamp, "equity": balance + unrealized})

    pending = {name: bool(signals[name].iloc[-1]) for name in columns}
    return {
        "initial_equity": initial_equity, "final_equity": equity[-1]["equity"],
        "balance": balance, "fees": fee_total, "equity_curve": equity,
        "trades": trades, "open_position": position, "pending_signals": pending,
        "execution_model": "confirmed close decision; market fill at next open",
        "limitations": ["fixed quantity", "no intrabar orders", "no funding", "no margin or liquidation model"],
    }
