"""Simple ZigZag Double: Pine-compatible levels, stop-order fills and signals.

The strategy trades persistent stop orders, so its backtest broker consumes
the zz_* columns on the completed bar.  Paper trading reuses the same timing:
a stop submitted at bar T-1's close (at that bar's level) fills when bar T
touches it, and an opposite fill reverses the open position.  There is no
take-profit: the reverse level IS the exit.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..registry import PrepareContext, StrategySpec, register

CODE = "simple_zigzag_double"
DEFAULTS = {
    "use_long": True,
    "use_short": True,
    "long_length": 5,
    "long_detection": 3,
    "short_length": 5,
    "short_detection": 3,
    "fixed_lot": False,
    "lot_long": 100.0,
    "lot_short": 100.0,
    "timestamp_start": 1388534400000,
    "timestamp_end": 2556057540000,
    "strict_position": False,
    "cost_filter": False,
    "regime_filter": False,
    "trailing_enabled": False,
    "cooldown_bars": 0,
    "min_hold_bars": 0,
    "max_hold_bars": 0,
}
RANGES = {
    "long_length": (2, 10000), "long_detection": (1, 10000),
    "short_length": (2, 10000), "short_detection": (1, 10000),
    "lot_long": (1, 10000), "lot_short": (1, 10000),
    "timestamp_start": (0, 100000000000000),
    "timestamp_end": (0, 100000000000000),
}
PREPARE_KEYS = (
    "use_long", "use_short", "long_length", "long_detection", "short_length",
    "short_detection", "timestamp_start", "timestamp_end", "strict_position",
)


def _levels(frame: pd.DataFrame, length: int, detection: int) -> tuple[pd.Series, pd.Series]:
    """Reproduce the Pine triple EMA and the persistent pivot-derived levels."""
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    # math.round in Pine rounds .5 upward, unlike Python's round-to-even.
    ema1_len = math.floor(length / 3 * 3 + 0.5)
    ema2_len = math.floor(length / 3 * 2 + 0.5)
    ema3_len = math.floor(length / 3 + 0.5)
    ema1 = close.ewm(span=ema1_len, adjust=False, ignore_na=True).mean()
    ema2 = ema1.ewm(span=ema2_len, adjust=False, ignore_na=True).mean()
    ema3 = ema2.ewm(span=ema3_len, adjust=False, ignore_na=True).mean()
    lows = low.rolling(detection, min_periods=1).min()
    highs = high.rolling(detection, min_periods=1).max()
    bottom = (ema3 >= ema3.shift(1)) & (ema3.shift(1) < ema3.shift(2))
    top = (ema3 < ema3.shift(1)) & (ema3.shift(1) >= ema3.shift(2))
    z = pd.Series(np.where(bottom, lows, np.where(top, highs, 0.0)), index=frame.index)
    point = z.where(z.ne(0)).ffill()
    # When point rises/falls, z is the new pivot; otherwise carry the old level.
    long_level = z.where(point.gt(point.shift(1))).ffill()
    short_level = z.where(point.lt(point.shift(1))).ffill()
    return long_level, short_level


def prepare(frame: pd.DataFrame, params: dict, ctx: PrepareContext) -> pd.DataFrame:
    data = frame.copy()
    long_level, _ = _levels(data, int(params["long_length"]), int(params["long_detection"]))
    _, short_level = _levels(data, int(params["short_length"]), int(params["short_detection"]))
    data["zz_long_level"] = long_level
    data["zz_short_level"] = short_level
    close = pd.to_numeric(data["close"], errors="coerce").to_numpy()
    longs, shorts = long_level.to_numpy(), short_level.to_numpy()
    if isinstance(data.index, pd.DatetimeIndex):
        timestamps = data.index.as_unit("ms").asi8
    else:
        raise ValueError("Simple ZigZag Double 需要 DatetimeIndex")
    virtual_pos = 0
    positions, long_entries, short_entries = [], [], []
    start, end = int(params["timestamp_start"]), int(params["timestamp_end"])
    for i, timestamp in enumerate(timestamps):
        prev_close = close[i - 1] if i else np.nan
        # Pine mutates virtual_pos before computing the entry conditions.
        # strict_position only guards the two "virtual exit" resets below; the
        # two direction flips are top-level statements in the source and run on
        # EVERY bar regardless of strict_position. Keeping them inside the
        # guard would leave virtual_pos at 0 by default, making both entry
        # conditions true on every bar and losing Pine's "do not re-place an
        # order once the level is virtually breached" filter.
        if params["strict_position"]:
            if virtual_pos == 1 and close[i] <= shorts[i] and prev_close > shorts[i]:
                virtual_pos = 0
            if virtual_pos == -1 and close[i] >= longs[i] and prev_close < longs[i]:
                virtual_pos = 0
        if virtual_pos != 1 and close[i] >= longs[i]:
            virtual_pos = 1
        if virtual_pos != -1 and close[i] <= shorts[i]:
            virtual_pos = -1
        in_window = timestamp > start and timestamp < end
        long_entries.append(bool(longs[i] > 0 and params["use_long"] and in_window and virtual_pos != 1))
        short_entries.append(bool(shorts[i] > 0 and params["use_short"] and in_window and virtual_pos != -1))
        if timestamp > end:
            virtual_pos = 0
        positions.append(virtual_pos)
    data["zz_virtual_pos"] = positions
    data["zz_long_entry"] = long_entries
    data["zz_short_entry"] = short_entries
    return data


def signals(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """把挂单时序翻译成巡检可执行的方向信号。

    回测撮合（zigzag_execution）的时序是：T-1 收盘按 `zz_long_entry` /
    `zz_short_entry` 提交挂单（价位取 T-1 的通道值，marketable 则 T 开盘成交），
    T 根内触价成交，反向成交自动反手。这里在整帧上复刻同一时序：

    - fill_long[T]  = zz_long_entry[T-1]  且 open/high 触及 long_level[T-1]
                      （或 T-1 收盘已越过价位 → marketable，按开盘成交）
    - 同一根 K 线两个方向都触价时，按开盘价与两档的距离近似触达先后。
    - LONG 的退出即反向通道：sl 取当前 zz_short_level（无止盈，tp=0）。

    仓位沿用系统风控口径（lot/100 再经 risk_per_trade_pct 截断），杠杆 1。
    """
    result = pd.DataFrame(index=data.index)
    close = pd.to_numeric(data["close"], errors="coerce")
    open_ = pd.to_numeric(data["open"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    long_level = pd.to_numeric(data["zz_long_level"], errors="coerce")
    short_level = pd.to_numeric(data["zz_short_level"], errors="coerce")
    long_entry = data["zz_long_entry"].astype(bool)
    short_entry = data["zz_short_entry"].astype(bool)

    place_long = long_entry.shift(1, fill_value=False)
    place_short = short_entry.shift(1, fill_value=False)
    pending_long = long_level.shift(1)
    pending_short = short_level.shift(1)
    prev_close = close.shift(1)
    fill_long = place_long & pending_long.gt(0) & (
        open_.ge(pending_long) | high.ge(pending_long) | prev_close.ge(pending_long)
    )
    fill_short = place_short & pending_short.gt(0) & (
        open_.le(pending_short) | low.le(pending_short) | prev_close.le(pending_short)
    )
    both = fill_long & fill_short
    # 同 bar 双触价：开盘离哪档更近就先成交哪档（回测同款路径假设）。
    long_first = (open_ - pending_long).abs() <= (open_ - pending_short).abs()
    fill_long = fill_long & ~(both & ~long_first)
    fill_short = fill_short & ~(both & long_first)

    signed = pd.Series(0.0, index=data.index)
    signed[fill_long] = 1.0
    signed[fill_short] = -1.0
    # LONG 止损 = 反向（空头挂单）通道；SHORT 止损 = 反向（多头挂单）通道。
    sl_for_long = short_level.where(fill_long)
    sl_for_short = long_level.where(fill_short)
    sl_pct = pd.Series(np.nan, index=data.index)
    sl_pct[fill_long] = 1.0 - sl_for_long[fill_long] / close[fill_long]
    sl_pct[fill_short] = sl_for_short[fill_short] / close[fill_short] - 1.0
    valid = sl_pct.gt(0) & np.isfinite(sl_pct)
    reason = pd.Series("awaiting_stop_fill", index=data.index, dtype=object)
    reason[signed.ne(0) & ~valid] = "no_reverse_level"
    reason[valid] = ""

    lot = np.where(signed.gt(0), float(params["lot_long"]) / 100.0,
                   np.where(signed.lt(0), float(params["lot_short"]) / 100.0, 0.0))
    cost = 2 * (float(params["fee_rate"]) + float(params["slippage"]))
    direction = signed.where(reason.eq(""), 0.0).fillna(0.0).astype(int)
    result["signed"] = signed
    result["score"] = 0.6
    result["prob_up"] = 0.5
    result["direction"] = direction
    result["block_reason"] = reason
    result["position_pct"] = pd.Series(lot, index=data.index).where(direction.ne(0), 0.0)
    result["leverage"] = pd.Series(1, index=data.index).where(direction.ne(0), 0)
    result["sl_pct"] = sl_pct.fillna(0.0)
    result["tp_pct"] = 0.0  # 无止盈：反向通道就是退出
    result["risk_pct"] = (result["position_pct"] * (sl_pct + cost)).fillna(0.0)
    result["net_reward_risk"] = 0.0
    for key in ("zz_long_level", "zz_short_level", "zz_virtual_pos", "zz_long_entry", "zz_short_entry"):
        result[key] = data[key]
    return result


def describe(row: pd.Series) -> dict:
    """最新一根的通道价位，供持仓止损跟随与前端展示。"""
    def _level(key: str) -> float:
        value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
        return round(float(value), 8) if pd.notna(value) and value > 0 else 0.0

    return {
        "zz_levels": {"long": _level("zz_long_level"), "short": _level("zz_short_level")},
        "zz_virtual_pos": int(pd.to_numeric(pd.Series([row.get("zz_virtual_pos")]), errors="coerce").iloc[0]),
    }


SPEC = register(StrategySpec(
    code=CODE,
    name="Simple ZigZag Double",
    description="双重 ZigZag 通道止损单突破策略，按 Pine 模拟挂单和反手。",
    defaults=DEFAULTS, ranges=RANGES,
    lookback=lambda p: max(600, 4 * max(int(p["long_length"]), int(p["short_length"]), int(p["long_detection"]), int(p["short_detection"]))),
    prepare=prepare, signals=signals, describe=describe,
    bool_keys=("use_long", "use_short", "fixed_lot", "strict_position"),
    int_keys=("long_length", "long_detection", "short_length", "short_detection", "timestamp_start", "timestamp_end"),
    prepare_keys=PREPARE_KEYS,
    default_enabled=False, tags=("tradingview", "zigzag", "stop_follow", "reverse_on_stop"),
))
