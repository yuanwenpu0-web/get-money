"""Execution primitives shared by the sequential backtest and paper orders."""

from __future__ import annotations

import math


def stop_exit(
    side: int, open_price: float, high: float, low: float, stop: float, target: float
) -> tuple[float, str] | None:
    # Open is known before the intrabar range. A gap through a stop fills at open.
    if stop > 0 and (open_price - stop) * side <= 0:
        return open_price, "stop_loss"
    if target > 0 and (open_price - target) * side >= 0:
        return target, "take_profit"
    if stop > 0 and (low <= stop if side == 1 else high >= stop):
        return stop, "stop_loss"
    if target > 0 and (high >= target if side == 1 else low <= target):
        return target, "take_profit"
    return None


def position_size(
    equity: float,
    cash: float,
    fraction: float,
    leverage: int,
    entry: float,
    stop: float,
    side: int,
    fee: float,
    slippage: float,
    risk_budget: float,
) -> tuple[float, float]:
    values = (equity, cash, fraction, leverage, entry, stop, fee, slippage, risk_budget)
    if (
        not all(math.isfinite(x) for x in values)
        or min(equity, cash, entry, stop, leverage, fraction) <= 0
    ):
        return 0.0, 0.0
    stop_fill = stop * (1 - side * slippage)
    loss_rate = max(0.0, side * (entry - stop_fill) / entry) + fee * (
        1 + stop_fill / entry
    )
    risk_margin = equity * risk_budget / (leverage * max(loss_rate, 1e-12))
    margin = min(equity * fraction, cash / (1 + leverage * fee), risk_margin)
    return margin, margin * leverage / entry


def breakeven_stop(
    side: int, entry: float, stop: float, high: float, low: float, trigger_r: float,
    risk: float,
) -> float:
    """浮盈达到 trigger_r 个 R 后把止损移到保本价（入场价）。

    实现刻意**无状态**：只问"本根是否触及过 entry ± trigger_r×risk"，触及就把 stop
    提升到 entry（单调、不可回落），因此命中过一次之后无需再记录标志位，
    回测撮合与实盘持仓管理调用同一函数即可保证行为一致。

    risk 是入场时的初始风险距离 |成交价 − 初始止损|；trigger_r<=0 或 risk<=0 时不动作。
    """
    if trigger_r <= 0 or risk <= 0 or stop <= 0:
        return stop
    trigger = entry + side * trigger_r * risk
    reached = high >= trigger if side == 1 else low <= trigger
    if not reached:
        return stop
    target = entry
    if side == 1:
        return max(stop, target)
    if stop > 0:
        return min(stop, target)
    return target


def trailing_stop(
    side: int, entry: float, stop: float, close: float, atr: float, params: dict
) -> float:
    if not params["trailing_enabled"] or atr <= 0:
        return stop
    if (close - entry) * side <= atr * params["trail_activation_atr"]:
        return stop
    candidate = close - side * atr * params["trailing_atr_mult"]
    return (
        max(stop, candidate)
        if side == 1
        else min(stop, candidate)
        if stop > 0
        else candidate
    )
