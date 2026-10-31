"""统一出场撮合器：入场由信号决定，出场一律是 rsi50_cross。

出场规则（全部候选共用，不参与参数搜索）：

- RSI(14, Wilder)，周期固定 14，写死在这里而非搜索维度——否则候选之间不可比。
- 多头：RSI 下穿 50 的那根 bar 发出出场信号，下一根开盘平仓。
- 空头：RSI 上穿 50 的那根 bar 发出出场信号，下一根开盘平仓。

撮合口径：

- 信号在 bar t 收盘确认，t+1 开盘成交（无未来函数）。
- 每笔固定名义本金 `stake`（默认 10 ），不叠加、不加仓，同一币种同时最多一仓。
- 滑点按方向施加（买抬价、卖压价），手续费双边按名义额计。
- 三道兜底：`max_hold_bars` 超时平仓、样本末尾按最后收盘价平仓，两者都记在 `reason` 里。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .lab_indicators import rsi_wilder
from .lab_signals import Signals, cross_down, cross_up

def to_ms(index) -> np.ndarray:
    """DatetimeIndex → 毫秒整数。不要用 `asi8`：pandas 2.x 的索引可能是 us 分辨率。"""
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return np.asarray(index.astype("datetime64[ms]").astype("int64"))


EXIT_RSI_PERIOD = 14
EXIT_RSI_LEVEL = 50.0


@dataclass(frozen=True, slots=True)
class SimConfig:
    stake: float = 10.0
    fee: float = 0.0004
    slippage: float = 0.0002
    max_hold_bars: int = 240
    side: int = 0  # 0 = 多空都做（对称），1 = 只做多，-1 = 只做空

    @property
    def round_trip_cost_pct(self) -> float:
        return 2.0 * (self.fee + self.slippage) * 100.0


@dataclass(slots=True)
class Trade:
    symbol: str
    side: int
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    bars: int
    pnl: float
    net_pct: float
    fees: float
    reason: str


def exit_signals(close: np.ndarray, period: int = EXIT_RSI_PERIOD) -> tuple[np.ndarray, np.ndarray]:
    """返回 (多头出场, 空头出场)：RSI 下穿 50 / 上穿 50。"""
    level = np.full(len(close), EXIT_RSI_LEVEL)
    rsi = rsi_wilder(close, period)
    return cross_down(rsi, level), cross_up(rsi, level)


def _side_trades(
    symbol: str,
    side: int,
    entry_mask: np.ndarray,
    exit_mask: np.ndarray,
    open_: np.ndarray,
    close: np.ndarray,
    ts: np.ndarray,
    cfg: SimConfig,
) -> list[Trade]:
    n = len(close)
    entries = np.flatnonzero(entry_mask)
    exits = np.flatnonzero(exit_mask)
    out: list[Trade] = []
    next_ok = 0
    for e in entries:
        if e < next_ok:
            continue
        fill = e + 1
        if fill >= n:
            break
        pos = int(np.searchsorted(exits, fill))
        natural = int(exits[pos]) + 1 if pos < len(exits) else n
        hard = e + cfg.max_hold_bars
        reason = "rsi50"
        if natural > hard:
            natural, reason = hard, "max_hold"
        if natural > n - 1:
            natural, reason = n - 1, "end_of_sample"
        entry_fill = open_[fill] * (1.0 + cfg.slippage * side)
        if not np.isfinite(entry_fill) or entry_fill <= 0:
            continue
        exit_fill = close[natural] * (1.0 - cfg.slippage * side) if reason == "end_of_sample" else open_[natural] * (1.0 - cfg.slippage * side)
        if not np.isfinite(exit_fill) or exit_fill <= 0:
            continue
        qty = cfg.stake / entry_fill
        gross = qty * (exit_fill - entry_fill) * side
        fees = cfg.fee * cfg.stake + cfg.fee * qty * exit_fill
        pnl = gross - fees
        out.append(
            Trade(
                symbol=symbol,
                side=side,
                entry_ts=int(ts[fill]),
                exit_ts=int(ts[natural]),
                entry_price=float(entry_fill),
                exit_price=float(exit_fill),
                bars=int(natural - fill),
                pnl=float(pnl),
                net_pct=float(pnl / cfg.stake),
                fees=float(fees),
                reason=reason,
            )
        )
        next_ok = natural
    return out


def simulate(
    symbol: str,
    frame: pd.DataFrame,
    signals: Signals,
    exits: tuple[np.ndarray, np.ndarray],
    cfg: SimConfig,
) -> list[Trade]:
    open_ = frame["open"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    ts = to_ms(frame.index)
    trades: list[Trade] = []
    if cfg.side >= 0:
        trades = _side_trades(symbol, 1, signals.long, exits[0], open_, close, ts, cfg)
    if cfg.side <= 0:
        trades.extend(_side_trades(symbol, -1, signals.short, exits[1], open_, close, ts, cfg))
    trades.sort(key=lambda t: t.entry_ts)
    return trades
