"""Small next-open backtest for fixed holding periods; no live order path."""
from __future__ import annotations

import numpy as np


def simulate(frame, positions, probabilities, expected_returns, horizon=15, fee=.0004, slippage=.0002, threshold=.65):
    if fee < 0 or slippage < 0 or horizon < 1:
        raise ValueError("手续费、滑点或持仓周期无效")
    equity = 1.0
    peak = equity
    max_drawdown = 0.0
    trades = []
    next_available = -1
    for pos, probability, expected in zip(positions, probabilities, expected_returns):
        side = 1 if probability[2] > threshold and expected > 2 * (fee + slippage) else (
            -1 if probability[0] > threshold and expected < -2 * (fee + slippage) else 0)
        entry, exit_ = int(pos) + 1, int(pos) + 1 + horizon
        if not side or entry < next_available or exit_ >= len(frame):
            continue
        segment = frame.iloc[int(pos):exit_ + 1]
        if not np.all(np.diff(segment.open_time.to_numpy()) == 60000):
            continue
        entry_price = float(frame.open.iloc[entry]) * (1 + side * slippage)
        exit_price = float(frame.open.iloc[exit_]) * (1 - side * slippage)
        gross = side * (exit_price / entry_price - 1)
        cost = fee * (1 + exit_price / entry_price)
        net = gross - cost
        equity *= 1 + net
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1)
        trades.append({"signal_open_ms": int(frame.open_time.iloc[int(pos)]),
                       "entry_open_ms": int(frame.open_time.iloc[entry]),
                       "exit_open_ms": int(frame.open_time.iloc[exit_]), "side": side,
                       "entry_price": entry_price, "exit_price": exit_price, "net_return": net, "fee": cost})
        next_available = exit_
    profits = [t["net_return"] for t in trades]
    return {"total_return": equity - 1, "closed_trade_max_drawdown": max_drawdown, "trades": len(trades),
            "win_rate": sum(x > 0 for x in profits) / len(profits) if profits else None,
            "fee_sum_notional": sum(t["fee"] for t in trades), "fee_per_side": fee,
            "slippage_per_side": slippage, "horizon_minutes": horizon,
            "limitations": ["每笔使用全部权益的简化收益率", "回撤只在平仓时计算，未按分钟盯市", "未包含资金费率、保证金、强平与止损", "仅作研究比较"],
            "transactions": trades}
