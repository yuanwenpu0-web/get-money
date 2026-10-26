"""One cash account, closed-bar signals, next-open fills and explicit costs."""

from __future__ import annotations

import random

import pandas as pd

from ..trading.risk import position_size, stop_exit, trailing_stop
from ..utils import interval_ms

RANDOM_BASELINE_SEED = 20261025


def randomize_directions(signals: pd.DataFrame, seed: int = RANDOM_BASELINE_SEED) -> pd.DataFrame:
    """证伪基线：固定种子对每个信号方向做 50/50 独立翻转（LONG<->SHORT，0 方向不变）。
    只改 direction 一列：时点、评分、仓位、杠杆与 sl_pct/tp_pct 全部不变——
    simulate 按新方向推导止损止盈，天然保持绝对距离不变（镜像），
    保证"相同入场时点、相同仓位/风控/成本，唯一变量是方向"。"""
    randomized = signals.copy()
    rng = random.Random(seed)
    flipped = []
    for value in randomized["direction"]:
        direction = int(value)
        flipped.append(-direction if direction and rng.random() < 0.5 else direction)
    randomized["direction"] = flipped
    return randomized


def exposure_metrics(equity: pd.Series, trades: list[dict], interval: str) -> dict:
    """曝光时间：持仓 bar 数 / 总 bar 数；平均每笔持仓 bar 数。无成交时为 0。"""
    step = interval_ms(interval)
    held = [
        int((pd.Timestamp(trade["exit_time"]) - pd.Timestamp(trade["entry_time"])).total_seconds() * 1000) // step + 1
        for trade in trades
    ]
    total_bars = len(equity)
    return {
        "exposure_pct": round(sum(held) / total_bars * 100, 4) if total_bars else 0.0,
        "avg_bars_held": round(sum(held) / len(held), 2) if held else 0.0,
    }


def simulate(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    params: dict,
    initial: float,
    fee: float,
    slippage: float,
    interval: str,
) -> tuple[pd.Series, list[dict]]:
    cash, position, next_entry = initial, None, 0
    values, trades = [], []
    bars = list(data.itertuples())
    decisions = list(signals.reindex(data.index).itertuples())
    step = interval_ms(interval)

    def close_position(price, timestamp, reason):
        nonlocal cash, position
        p = position
        fill = price * (1 - p["side"] * slippage)
        gross = (fill - p["entry"]) * p["qty"] * p["side"]
        exit_fee = fill * p["qty"] * fee
        cash += p["margin"] + gross - exit_fee
        net = gross - p["entry_fee"] - exit_fee
        trades.append(
            {
                "entry_time": str(p["time"]),
                "exit_time": str(timestamp),
                "side": "long" if p["side"] == 1 else "short",
                "entry_price": p["entry"],
                "exit_price": fill,
                "size": p["qty"],
                "pnl": net,
                "return_pct": net / p["margin"] * 100,
                "fees": p["entry_fee"] + exit_fee,
                "status": reason,
                "leverage": p["leverage"],
                "margin": p["margin"],
            }
        )
        position = None

    for i, row in enumerate(bars):
        signal = decisions[i - 1] if i else None
        previous = bars[i - 1] if i else None
        gap = i > 0 and (row.Index - previous.Index).total_seconds() * 1000 != step
        exited = False
        if position is not None:
            p = position
            hit = stop_exit(p["side"], row.open, row.open, row.open, p["sl"], p["tp"])
            held = i - p["index"]
            if hit:
                close_position(hit[0], row.Index, hit[1])
            elif gap:
                close_position(row.open, row.Index, "data_gap")
            elif params["max_hold_bars"] > 0 and held >= params["max_hold_bars"]:
                close_position(row.open, row.Index, "time_stop")
            elif held >= params["min_hold_bars"] and signal.direction == -p["side"]:
                close_position(row.open, row.Index, "signal_reverse")
            exited = position is None

        if (
            position is None
            and not exited
            and not gap
            and i >= next_entry
            and signal is not None
            and signal.direction
            and cash > 0
        ):
            side = int(signal.direction)
            entry = row.open * (1 + side * slippage)
            stop = previous.close * (1 - side * signal.sl_pct)
            target = previous.close * (1 + side * signal.tp_pct)
            # Ignore signals whose exit levels were already crossed at the next open.
            if (entry - stop) * side > 0 and (target - entry) * side > 0:
                margin, qty = position_size(
                    cash,
                    cash,
                    signal.position_pct,
                    signal.leverage,
                    entry,
                    stop,
                    side,
                    fee,
                    slippage,
                    params["risk_per_trade_pct"],
                )
                if margin >= 10:
                    entry_fee = entry * qty * fee
                    cash -= margin + entry_fee
                    position = dict(
                        side=side,
                        entry=entry,
                        qty=qty,
                        margin=margin,
                        sl=stop,
                        tp=target,
                        entry_fee=entry_fee,
                        time=row.Index,
                        index=i,
                        leverage=int(signal.leverage),
                    )

        if position is not None:
            p = position
            hit = stop_exit(p["side"], row.open, row.high, row.low, p["sl"], p["tp"])
            if hit:
                close_position(hit[0], row.Index, hit[1])
                exited = True
            else:
                p["sl"] = trailing_stop(
                    p["side"],
                    p["entry"],
                    p["sl"],
                    row.close,
                    row.atr_pct * row.close,
                    params,
                )
        if exited:
            next_entry = i + int(params["cooldown_bars"]) + 1
        if position is None:
            values.append(cash)
        else:
            p = position
            values.append(
                cash + p["margin"] + (row.close - p["entry"]) * p["qty"] * p["side"]
            )
    if position is not None:
        close_position(bars[-1].close, bars[-1].Index, "close_at_end")
        values[-1] = cash
    return pd.Series(values, index=data.index, dtype=float), trades
