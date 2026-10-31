"""ATR 助手：按 ATR 倍数给出止损价、三档 R 倍数目标价与仓位规模。

Source: pine-crypto-collection/btankutt_smc/indicators/04-atr-helper/atr-helper.pine

Author: Baris Tankut (@btankutt), https://github.com/btankutt/smc-pine-suite — MPL 2.0.

One trade plan per bar: the average true range (`ta.tr(true)`, so the first bar already
uses its own high-low) smoothed by the chosen family gives the stop distance, which is
`sl_mult` ATRs away from the entry in the direction of the trade. The three targets sit at
`tp1_r` / `tp2_r` / `tp3_r` times that risk, and position sizing divides the account risk
by the risk per unit (`stop distance * point_value`).

The entry is the bar's close unless `entry_mode` is "Manual" *and* `manual_entry` is
positive, which mirrors the script's `entry_mode == "Manual" and manual_entry > 0` guard.
With a trailing close entry the stop and targets follow every bar, so `sl_crossed` /
`tp1_crossed` can only fire on a manual entry — exactly as the upstream comment warns.

Dropped: the box/line/label plan drawing, the summary table, the colours and zone
transparency, and the `tp1_on` / `tp2_on` / `tp3_on` / `sizing_on` / `show_table` display
switches with the two alert conditions.
"""
from __future__ import annotations

import pandas as pd

from ._pine import (choice, cross, integer, ma, nullable_int, number, ohlcv, true_range)

_SMOOTHERS = ("RMA", "SMA", "EMA", "WMA")
_DIRECTIONS = ("Long", "Short")
_ENTRY_MODES = ("Current close", "Manual")


def atr_helper(frame: pd.DataFrame, *, atr_length: int = 14, atr_smoothing: str = "RMA",
               stop_loss_multiple: float = 1.5, direction: str = "Long",
               entry_mode: str = "Current close", manual_entry: float = 0.0,
               tp1_r: float = 1.0, tp2_r: float = 2.0, tp3_r: float = 3.0,
               account_size: float = 10000.0, risk_percent: float = 1.0,
               point_value: float = 1.0) -> pd.DataFrame:
    integer("atr_length", atr_length, 1, 200)
    choice("atr_smoothing", atr_smoothing, _SMOOTHERS)
    number("stop_loss_multiple", stop_loss_multiple, 0.1, 20.0)
    choice("direction", direction, _DIRECTIONS)
    choice("entry_mode", entry_mode, _ENTRY_MODES)
    number("manual_entry", manual_entry, 0.0)
    number("tp1_r", tp1_r, 0.1)
    number("tp2_r", tp2_r, 0.1)
    number("tp3_r", tp3_r, 0.1)
    number("account_size", account_size, 0.0)
    number("risk_percent", risk_percent, 0.01, 100.0)
    number("point_value", point_value, 1e-7)

    _, _, close = ohlcv(frame, "high", "low", "close")
    average_range = ma(true_range(frame), atr_length, atr_smoothing)

    sign = 1.0 if direction == "Long" else -1.0
    if entry_mode == "Manual" and manual_entry > 0:
        entry = pd.Series(manual_entry, index=frame.index)
    else:
        entry = close
    stop_distance = average_range * stop_loss_multiple
    stop_loss = entry - sign * stop_distance

    risk_amount = account_size * risk_percent / 100.0
    risk_per_unit = stop_distance * point_value
    position_size = risk_amount / risk_per_unit.where(risk_per_unit > 0)

    targets = {f"take_profit_{index + 1}": entry + sign * stop_distance * multiple
               for index, multiple in enumerate((tp1_r, tp2_r, tp3_r))}
    columns = {
        "atr": average_range,
        "direction": nullable_int(pd.Series(sign, index=frame.index)),
        "entry": entry,
        "stop_distance": stop_distance,
        "stop_loss": stop_loss,
        **targets,
        "risk_amount": pd.Series(risk_amount, index=frame.index),
        "risk_per_unit": risk_per_unit,
        "position_size": position_size,
        "stop_crossed": cross(close, stop_loss),
        "tp1_crossed": cross(close, targets["take_profit_1"]),
    }
    return pd.DataFrame(columns, index=frame.index)
