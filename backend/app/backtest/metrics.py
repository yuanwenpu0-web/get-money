"""回测指标与权益曲线下采样（从 runner.py 拆出，保持单文件行数上限）。

指标口径与成交模拟严格一致：收益按权益曲线、回撤按初始资金为基准的滚动峰值、
费用为双边合计；零成交变体同样返回完整指标（不报错），由调用方按成交笔数设门槛。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def metrics_from_equity(
    equity: pd.Series, trades: list[dict], initial: float, interval: str
) -> dict:
    equity = equity.dropna()
    if equity.empty:
        return {}
    drawdown = (equity / equity.cummax().clip(lower=initial) - 1) * 100
    daily = equity.resample("1D").last()
    daily = (
        pd.concat(
            [pd.Series([initial], index=[daily.index[0] - pd.Timedelta(days=1)]), daily]
        )
        .pct_change()
        .dropna()
    )
    sharpe = (
        float(daily.mean() / daily.std() * math.sqrt(365))
        if len(daily) > 2 and daily.std() > 0
        else 0.0
    )
    # A position may have several partial exits. Count its combined result once;
    # legacy trade records without a position ID remain independent trades.
    positions = {}
    for index, trade in enumerate(trades):
        position_id = trade.get("position_id")
        key = ("position", position_id) if position_id is not None else ("trade", index)
        positions[key] = positions.get(key, 0.0) + trade["pnl"]
    pnls = list(positions.values())
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    gross_profit, gross_loss = sum(wins), -sum(losses)
    return {
        "initial_cash": round(initial, 2),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_return_pct": round((float(equity.iloc[-1]) / initial - 1) * 100, 4),
        "max_drawdown_pct": round(float(drawdown.min()), 4),
        "sharpe": round(sharpe, 2) if math.isfinite(sharpe) else 0.0,
        "total_trades": len(pnls),
        "win_rate": round(len(wins) / len(pnls) * 100, 2) if pnls else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
        "profit_factor_infinite": bool(gross_profit > 0 and not gross_loss),
        "expectancy": round(sum(pnls) / len(pnls), 4)
        if pnls
        else 0.0,
        "total_fees": round(sum(t["fees"] for t in trades), 2),
        "bars": len(equity),
        "start": str(equity.index[0]),
        "end": str(equity.index[-1]),
    }


def downsample(equity: pd.Series, limit: int = 1500) -> list[dict]:
    indices = np.unique(
        np.linspace(0, len(equity) - 1, min(limit, len(equity)), dtype=int)
    )
    return [
        {"time": str(equity.index[i]), "equity": round(float(equity.iloc[i]), 2)}
        for i in indices
    ]
