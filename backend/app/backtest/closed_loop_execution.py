"""闭环策略专用撮合：把 `strategy/closed_loop.py` 的事件列翻译成资金曲线与成交。

通用 `execution.simulate` 无法表达「挂单等待 / 成交前撤单 / 保护价随持仓 ratchet /
趋势破位市价平仓」，因此按手册 H10 走专用 broker。状态机只给价格与事件，
手续费与滑点在这里按回测请求的全局口径（fee_rate / slippage）单边各扣一次，
策略参数不得覆盖成本。
"""
from __future__ import annotations

import math

import pandas as pd


def simulate_closed_loop(data, signals, params, initial, fee, slippage, interval):
    """返回 (权益曲线, 成交列表)，与 `zigzag_execution.simulate_zigzag` 同形。"""
    del signals, params, interval  # 事件列已包含全部决策，避免再解释一次信号
    if data.empty:
        return pd.Series(dtype=float, index=data.index), []

    events = data["st_cl_event"].astype(str).to_numpy()
    sides = pd.to_numeric(data["st_cl_side"], errors="coerce").fillna(0).to_numpy(dtype=float)
    fills = pd.to_numeric(data["st_cl_fill_price"], errors="coerce").to_numpy(dtype=float)
    exits = pd.to_numeric(data["st_cl_exit_price"], errors="coerce").to_numpy(dtype=float)
    reasons = data["st_cl_close_reason"].astype(str).to_numpy()
    order_ids = pd.to_numeric(data["st_cl_order_id"], errors="coerce").fillna(0).to_numpy(dtype=float)
    closes = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype=float)
    stamps = list(data.index)

    balance, position = float(initial), None
    values: list[float] = []
    trades: list[dict] = []

    def mark(price: float) -> float:
        if position is None or not math.isfinite(price):
            return balance
        return balance + (price - position["entry"]) * position["qty"] * position["side"]

    for i, stamp in enumerate(stamps):
        event = events[i]
        if event == "fill" and position is None:
            side = int(sides[i])
            price = fills[i]
            if side in (-1, 1) and math.isfinite(price) and price > 0:
                entry = price * (1 + side * slippage)
                qty = max(balance, 0.0) / entry if entry > 0 else 0.0
                if qty > 0:
                    entry_fee = abs(entry * qty) * fee
                    balance -= entry_fee
                    position = dict(side=side, entry=entry, qty=qty, fee=entry_fee,
                                    time=stamp, position_id=int(order_ids[i]) or i)
        elif event.startswith("close_") and position is not None:
            price = exits[i]
            if not math.isfinite(price) or price <= 0:
                price = closes[i] if math.isfinite(closes[i]) else position["entry"]
            p = position
            exit_price = price * (1 - p["side"] * slippage)
            gross = (exit_price - p["entry"]) * p["qty"] * p["side"]
            exit_fee = abs(exit_price * p["qty"]) * fee
            balance += gross - exit_fee
            net = gross - p["fee"] - exit_fee
            margin = abs(p["entry"] * p["qty"])
            trades.append(dict(
                entry_time=str(p["time"]), exit_time=str(stamp),
                position_id=str(p["position_id"]),
                side="long" if p["side"] == 1 else "short",
                entry_price=p["entry"], exit_price=exit_price,
                size=p["qty"], pnl=net,
                return_pct=net / margin * 100 if margin else 0.0,
                fees=p["fee"] + exit_fee,
                status=reasons[i] if reasons[i] else event[len("close_"):],
                leverage=1, margin=margin,
            ))
            position = None
        values.append(mark(closes[i]))

    equity = pd.Series(values, index=data.index, dtype=float)
    equity.attrs["open_position"] = (
        dict(side="long" if position["side"] == 1 else "short", qty=position["qty"],
             entry_price=position["entry"], entry_time=str(position["time"]),
             entry_fee=position["fee"],
             unrealized_pnl=(float(closes[-1]) - position["entry"]) * position["qty"] * position["side"])
        if position else None
    )
    return equity, trades
