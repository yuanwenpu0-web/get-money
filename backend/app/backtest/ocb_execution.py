r"""Opening Candle Breakout 专用撮合：按 Pine `strategy` 的默认订单语义逐 bar 回放。

通用撮合（本文件外的 `backtest/execution.py`）假设「信号 + ATR 止损 + 时间止损 + 名义百分比仓位」，
这套假设对 OCB 三条规则都不成立，所以这里单独实现：

**每根 bar 的处理顺序**（严格对齐脚本 + Pine 默认属性）

1. **开盘**：成交上一根收盘提交的订单。`strategy.close_all()` 写在入场之前，
   所以同一根里先平 session close，再开新仓。
2. **盘中**：默认 OHLC 插值路径 `open → 离开盘更近的极值 → 另一个极值`
   （`process_orders_on_close=false` 且不开 bar magnifier 时 TradingView 的口径）。
   止盈/止损是括号单，随入场一起挂出并在成交后立即可用——新仓位在其入场的那一根就能被打止。
3. **收盘**：按脚本顺序执行
   `newDay` 复位 `tradedToday` → 强平窗提交 `close_all` → 开盘首窗提交 `entry`。
   入场数量为「风险预算 ÷ 单位风险距离」，止损/目标价位在**信号那根的收盘**锁定，
   之后不随成交价漂移（脚本原文就是这样，R 的分母是信号收盘，不是实际成交价）。

现金模型与该项目其它撮合一致：开仓冻结保证金（`qty × 成交价`）+ 买入费，
平仓返还保证金 + 净盈亏 − 卖出费；`leverage` 恒为 1（脚本 `margin_long=0`）。
"""

from __future__ import annotations

import math

import pandas as pd

# 低于这个名义值就不重复手续费成本去开一笔（脚本里「不足 1 张合约」的等价物）
MIN_MARGIN = 10.0


def _level(value) -> float:
    """把 prepare 托管出来的价位读成有效正数；缺失/NaN/非正一律当成「不挂这一侧」。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


def simulate_ocb(data, signals, params, initial, fee, slippage, interval):
    """回放一个币种的 OCB 剧本，返回（逐根盯市净值, 成交记录）。

    `interval` 只用于与其它撮合签名保持一致：本策略的出场由**会话时间**触发，
    不由 bar 计数触发。
    """
    del interval
    if data.empty:
        return pd.Series(dtype=float, index=data.index), []

    rows = list(data.itertuples())
    decisions = list(signals.reindex(data.index).itertuples())
    cash = float(initial)
    position = None
    pending_entry = None
    pending_flat = False
    traded_today = False
    values, trades = [], []

    def mark(price: float) -> float:
        if position is None:
            return cash
        return cash + (price - position["entry"]) * position["qty"] * position["side"]

    def close_at(price: float, stamp, reason: str) -> None:
        nonlocal cash, position
        p = position
        fill = price * (1 - p["side"] * slippage)
        gross = (fill - p["entry"]) * p["qty"] * p["side"]
        exit_fee = abs(fill * p["qty"]) * fee
        cash += p["margin"] + gross - exit_fee
        net = gross - p["entry_fee"] - exit_fee
        trades.append(
            {
                "entry_time": str(p["time"]),
                "position_id": str(p["time"]),
                "exit_time": str(stamp),
                "side": "long" if p["side"] == 1 else "short",
                "entry_price": p["entry"],
                "exit_price": fill,
                "size": p["qty"],
                "pnl": net,
                "return_pct": net / p["margin"] * 100 if p["margin"] else 0.0,
                "fees": p["entry_fee"] + exit_fee,
                "status": reason,
                "leverage": 1,
                "margin": p["margin"],
                "risk": p["risk"],
            }
        )
        position = None

    def fill_entry(open_price: float, stamp) -> None:
        """上一根收盘提交的市价单，在本根开盘成交。"""
        nonlocal cash, position
        order = pending_entry
        side = order["side"]
        fill = open_price * (1 + side * slippage)
        qty = order["qty"]
        if not math.isfinite(fill) or fill <= 0 or not math.isfinite(qty) or qty <= 0:
            return
        # 无杠杆：名义 = 保证金。只有在现金不够时才缩量（缩到连最小名义都不够就不开了），
        # 否则按提交时的数量成交——避免因次根开盘价略低而把「已下单」的行情悄悄丢掉。
        affordable = cash / (fill * (1 + fee)) if fill > 0 else 0.0
        if affordable < qty:
            qty = affordable
            if qty * fill < MIN_MARGIN:
                return
        margin = qty * fill
        if not math.isfinite(margin) or margin <= 0:
            return
        entry_fee = abs(margin) * fee
        cash -= margin + entry_fee
        position = {
            "side": side,
            "entry": fill,
            "qty": qty,
            "margin": margin,
            "entry_fee": entry_fee,
            "time": stamp,
            "stop": order["stop"],
            "target": order["target"],
            "risk": order["risk"],
        }

    def intrabar_exit(open_price: float, high: float, low: float, stamp) -> bool:
        """按 OHLC 插值路径检查括号单；返回是否已被平掉。"""
        nonlocal position
        p = position
        side, stop, target = p["side"], p["stop"], p["target"]
        # 开盘就跳空穿价位者，先按跳空处理（止损按开盘价，止盈按约定）
        if stop > 0 and (open_price - stop) * side <= 0:
            close_at(open_price, stamp, "stop_loss")
            return True
        if target > 0 and (open_price - target) * side >= 0:
            price = open_price if params.get("tp_gap_at_open") else target
            close_at(price, stamp, "take_profit")
            return True
        # 两笔极值里先走离开盘更近的那个；距离相等时先高后低（TradingView 未定义平票顺序）
        extremes = (high, low) if abs(open_price - high) < abs(open_price - low) else (low, high)
        for price in extremes:
            if stop > 0 and (price - stop) * side <= 0:
                close_at(stop, stamp, "stop_loss")
                return True
            if target > 0 and (price - target) * side >= 0:
                close_at(target, stamp, "take_profit")
                return True
        return False

    def submit_entry(decision, close_price: float) -> None:
        """脚本在这个收盘时点下的工夫：算数量、锁价位、`tradedToday` 置位。"""
        nonlocal traded_today, pending_entry
        side = int(float(getattr(decision, "ocb_dir", 0.0) or 0.0))
        risk = float(getattr(decision, "ocb_risk", float("nan")))
        if side == 0 or not math.isfinite(risk) or risk <= 0 or close_price <= 0:
            return
        equity = cash  # 走到这里必然空仓：`position is None`
        if int(params["risk_mode"]) == 0:
            budget = float(params["risk_usd"])
        else:
            budget = equity * float(params["risk_per_trade_pct"])
        qty = budget / risk
        cap = equity * float(params["max_position_pct"])
        if cap > 0 and qty * close_price > cap:
            qty = cap / close_price
        notional = qty * close_price
        if notional < MIN_MARGIN:
            if not params["allow_min_position"] or cash < MIN_MARGIN * (1 + fee):
                return  # 脚本：q < 1 且不允许最小手数 → 不下单，`tradedToday` 不置位
            qty = MIN_MARGIN / close_price
        traded_today = True
        pending_entry = {
            "side": side,
            "qty": qty,
            "risk": risk,
            "stop": _level(getattr(decision, "ocb_stop", 0.0)),
            "target": _level(getattr(decision, "ocb_target", 0.0)),
        }

    for row, decision in zip(rows, decisions):
        stamp = row.Index
        open_price = float(row.open)
        high = float(row.high)
        low = float(row.low)
        close_price = float(row.close)

        # ① 开盘：上一根收盘提交的订单在这里成交（脚本里 close_all 写在 entry 之前）
        if pending_flat:
            pending_flat = False
            if position is not None:
                close_at(open_price, stamp, "session_close")
        if pending_entry is not None:
            fill_entry(open_price, stamp)
            pending_entry = None

        # ② 盘中：括号单沿 OHLC 插值路径判定（含刚刚成交的那一根）
        if position is not None:
            intrabar_exit(open_price, high, low, stamp)

        # ③ 收盘：严格按脚本顺序 — newDay 复位 → 强平窗提交 → 开盘首窗下单
        if bool(getattr(decision, "ocb_new_day", False)):
            traded_today = False
        if position is not None and bool(getattr(decision, "ocb_is_flat", False)):
            pending_flat = True
        elif (
            not traded_today
            and position is None
            and pending_entry is None
            and int(float(getattr(decision, "ocb_dir", 0.0) or 0.0)) != 0
        ):
            submit_entry(decision, close_price)

        values.append(mark(close_price))

    equity = pd.Series(values, index=data.index, dtype=float)
    equity.attrs["open_position"] = (
        {
            "side": "long" if position["side"] == 1 else "short",
            "qty": position["qty"],
            "entry_price": position["entry"],
            "entry_time": str(position["time"]),
            "stop": position["stop"],
            "target": position["target"],
            "entry_fee": position["entry_fee"],
            "unrealized_pnl": (float(rows[-1].close) - position["entry"])
            * position["qty"]
            * position["side"],
        }
        if position is not None
        else None
    )
    return equity, trades
