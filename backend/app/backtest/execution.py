"""One cash account, closed-bar signals, next-open fills and explicit costs."""

from __future__ import annotations

import random

import pandas as pd

from ..trading.risk import breakeven_stop, position_size, stop_exit, trailing_stop
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
    positions = {}
    for index, trade in enumerate(trades):
        key = trade.get("position_id")
        key = ("position", key) if key is not None else ("trade", index)
        entry = pd.Timestamp(trade["entry_time"])
        exit_time = pd.Timestamp(trade["exit_time"])
        if key in positions:
            old_entry, old_exit = positions[key]
            positions[key] = (min(entry, old_entry), max(exit_time, old_exit))
        else:
            positions[key] = (entry, exit_time)
    held = [int((end - start).total_seconds() * 1000) // step + 1
            for start, end in positions.values()]
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
                "position_id": str(p["time"]),
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

    def partial_close(price, timestamp, frac, reason="partial_tp"):
        """浮盈达标先平一部分落袋：现金/数量/保证金/入场费按比例结转，剩余仓位继续持有。"""
        nonlocal cash
        p = position
        fill = price * (1 - p["side"] * slippage)
        qty = p["qty"] * frac
        margin = p["margin"] * frac
        entry_fee = p["entry_fee"] * frac
        gross = (fill - p["entry"]) * qty * p["side"]
        exit_fee = fill * qty * fee
        cash += margin + gross - exit_fee
        net = gross - entry_fee - exit_fee
        trades.append(
            {
                "entry_time": str(p["time"]),
                "position_id": str(p["time"]),
                "exit_time": str(timestamp),
                "side": "long" if p["side"] == 1 else "short",
                "entry_price": p["entry"],
                "exit_price": fill,
                "size": qty,
                "pnl": net,
                "return_pct": net / margin * 100,
                "fees": entry_fee + exit_fee,
                "status": reason,
                "leverage": p["leverage"],
                "margin": margin,
            }
        )
        p["qty"] -= qty
        p["margin"] -= margin
        p["entry_fee"] -= entry_fee

    def close_reached_targets(high, low, timestamp):
        """Fill reached targets in order; fractions refer to the original position."""
        nonlocal position
        p = position
        for index, target in enumerate(p["targets"]):
            if target["done"]:
                continue
            reached = high >= target["price"] if p["side"] == 1 else low <= target["price"]
            if not reached:
                continue
            target["done"] = True
            quantity = min(p["initial_qty"] * target["fraction"], p["qty"])
            if quantity <= 0:
                continue
            last_active = not any(t["fraction"] > 0 for t in p["targets"][index + 1:])
            if last_active or quantity >= p["qty"] * (1 - 1e-12):
                close_position(target["price"], timestamp, target["status"])
                return
            partial_close(target["price"], timestamp, quantity / p["qty"], target["status"])

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
            elif p.get("targets"):
                # A resting target crossed at the open fills before an open-time exit.
                close_reached_targets(row.open, row.open, row.Index)
            if position is not None and not hit and not gap and params["max_hold_bars"] > 0 and held >= params["max_hold_bars"]:
                close_position(row.open, row.Index, "time_stop")
            elif position is not None and not hit and not gap and (
                params.get("ema_exit")
                and p["side"] == 1
                and signal is not None
                and int(getattr(signal, "exit_long", 0) or 0)
            ):
                # 软出场：上一根收盘已失守 EMA9 → 本根开盘平多（与入场同为「收盘信号、次开成交」）
                close_position(row.open, row.Index, "ema_exit")
            elif position is not None and not hit and not gap and held >= params["min_hold_bars"] and signal.direction == -p["side"]:
                close_position(row.open, row.Index, "signal_reverse")
            exited = position is None

        entry_allowed = True
        if signal is not None and signal.direction:
            attr = "allow_long_entry" if signal.direction == 1 else "allow_short_entry"
            allowed = getattr(signal, attr, True)
            entry_allowed = bool(allowed) if pd.notna(allowed) else False
        if (
            position is None
            and not exited
            and not gap
            and i >= next_entry
            and signal is not None
            and signal.direction
            and entry_allowed
            and cash > 0
        ):
            side = int(signal.direction)
            entry = row.open * (1 + side * slippage)
            stop = previous.close * (1 - side * signal.sl_pct)
            target = previous.close * (1 + side * signal.tp_pct)
            scaled = all(k in params for k in ("tp1_r", "tp2_r", "target_r", "tp1_fraction", "tp2_fraction", "tp3_fraction"))
            target_levels = []
            if scaled:
                risk_from_signal = abs(previous.close - stop)
                for n, r_key in enumerate(("tp1_r", "tp2_r", "target_r"), 1):
                    target_levels.append({"price": previous.close + side * risk_from_signal * float(params[r_key]),
                                          "fraction": float(params[f"tp{n}_fraction"]),
                                          "status": f"take_profit_{n}", "done": False})
                target = next(level["price"] for level in target_levels if level["fraction"] > 0)
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
                        risk=abs(entry - stop),
                        partial_done=False,
                        initial_qty=qty,
                        targets=target_levels,
                    )
                    if scaled:
                        # Individual targets are managed below, not by the single-target exit.
                        position["tp"] = 0.0

        if position is not None:
            p = position
            hit = stop_exit(p["side"], row.open, row.high, row.low, p["sl"], p["tp"])
            if hit:
                close_position(hit[0], row.Index, hit[1])
                exited = True
            else:
                if p.get("targets"):
                    close_reached_targets(row.high, row.low, row.Index)
                    if position is None:
                        exited = True
                if position is None:
                    if exited:
                        next_entry = i + int(params["cooldown_bars"]) + 1
                    values.append(cash)
                    continue
                partial_r = float(params.get("partial_tp_r", 0.0))
                if not p.get("targets") and not p["partial_done"] and partial_r > 0:
                    partial_price = p["entry"] + p["side"] * partial_r * p["risk"]
                    if (row.high - partial_price) * p["side"] >= 0:
                        # 同根 SL/TP 已由上方 stop_exit 先判；到这里才允许落袋。
                        # 剩余仓位止损移到保本价、取消固定目标，交给追踪止损与软出场跑完。
                        partial_close(
                            partial_price, row.Index, float(params.get("partial_tp_frac", 0.5))
                        )
                        p["sl"] = (
                            max(p["sl"], p["entry"])
                            if p["side"] == 1
                            else min(p["sl"], p["entry"])
                        )
                        p["tp"] = 0.0
                        p["partial_done"] = True
                p["sl"] = breakeven_stop(
                    p["side"],
                    p["entry"],
                    p["sl"],
                    row.high,
                    row.low,
                    float(params.get("be_trigger_r", 0.0)),
                    p["risk"],
                )
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
