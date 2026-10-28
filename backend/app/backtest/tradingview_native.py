"""Local signal ports of Eterna Pine strategies and a small market-order simulator.

Source paths for each port are recorded in SOURCES.

Signals and percent-of-equity order quantities are evaluated at the signal-bar
close; market orders fill at the next bar's open. The last bar's orders cannot
fill. Open positions are marked to the last close, not forcibly liquidated.
Capital, sizing and commission defaults follow the respective source scripts;
webhook leverage and position size are alert fields, not Pine strategy sizing.
This simulator does not implement margin calls,
slippage, bar magnifier, symbol-specific lot increments, or intrabar fills, so it
is not a full Pine emulator. Input bars must already be time-ordered and closed;
the caller is responsible for checking their interval and gaps.
"""

from collections import deque
from math import isfinite

from . import (tv_external_ports, tv_momentum_ports, tv_more_momentum_ports,
               tv_more_trend_ports, tv_trend_ports)
from .pine_compare import gt, ge, lt, le

SOURCES = {
    "sma_crossover": "data/pine-crypto-collection/eterna/strategies/trend-following/sma-crossover/strategy.pine",
    "donchian_channel": "data/pine-crypto-collection/eterna/strategies/trend-following/donchian-channel/strategy.pine",
}
_PORT_MODULES = (tv_trend_ports, tv_momentum_ports, tv_more_trend_ports,
                 tv_more_momentum_ports, tv_external_ports)
for _module in _PORT_MODULES:
    SOURCES.update(_module.SOURCES)


def _cross_up(a, b, old_a, old_b):
    return gt(a, b) and le(old_a, old_b)


def _cross_down(a, b, old_a, old_b):
    return lt(a, b) and ge(old_a, old_b)


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def tradingview_signals(candles, strategy, params=None):
    """Return per-bar entry and position-specific exit signals for the source strategy.

    Each candle is a mapping containing numeric open, high, low and close; an
    optional time value is passed through by the simulator. No lookahead is used.
    """
    if strategy not in SOURCES:
        raise ValueError(f"unsupported TradingView strategy: {strategy}")
    for module in _PORT_MODULES:
        if strategy in module.SOURCES:
            return module.tradingview_signals(candles, strategy, params)
    params = dict(params or {})
    allowed = {"fast_length", "slow_length"} if strategy == "sma_crossover" else {
        "channel_length", "breakout_confirm", "use_middle_line"
    }
    unknown = params.keys() - allowed
    if unknown:
        raise ValueError(f"unsupported parameters: {sorted(unknown)}")
    bars = []
    for candle in candles:
        bar = {key: float(candle[key]) for key in ("open", "high", "low", "close")}
        if not all(isfinite(v) and v > 0 for v in bar.values()):
            raise ValueError("OHLC prices must be finite and positive")
        if bar["high"] < max(bar["open"], bar["close"], bar["low"]) or bar["low"] > min(bar["open"], bar["close"]):
            raise ValueError("invalid OHLC range")
        bars.append(bar)
    signals = [{"long": False, "short": False, "close_long": False, "close_short": False} for _ in bars]
    if strategy == "sma_crossover":
        fast = _positive_int(params.get("fast_length", 50), "fast_length")
        slow = _positive_int(params.get("slow_length", 200), "slow_length")
        def averages(length):
            queue, total, values = deque(), 0.0, []
            for bar in bars:
                queue.append(bar["close"])
                total += bar["close"]
                if len(queue) > length:
                    total -= queue.popleft()
                values.append(total / length if len(queue) == length else None)
            return values
        f, s = averages(fast), averages(slow)
        for i in range(1, len(bars)):
            signals[i]["long"] = _cross_up(f[i], s[i], f[i-1], s[i-1])
            signals[i]["short"] = _cross_down(f[i], s[i], f[i-1], s[i-1])
    else:
        length = _positive_int(params.get("channel_length", 20), "channel_length")
        confirm = _positive_int(params.get("breakout_confirm", 1), "breakout_confirm", 0)
        if confirm > 5:
            raise ValueError("breakout_confirm must be <= 5")
        middle = params.get("use_middle_line", True)
        if not isinstance(middle, bool):
            raise ValueError("use_middle_line must be boolean")
        upper, lower, mid = [], [], []
        for i in range(len(bars)):
            window = bars[max(0, i-length+1):i+1]
            u = max(b["high"] for b in window) if len(window) == length else None
            l = min(b["low"] for b in window) if len(window) == length else None
            upper.append(u); lower.append(l); mid.append((u + l) / 2 if u is not None else None)
        long_breaks, short_breaks = [], []
        for i, bar in enumerate(bars):
            ref = i-confirm
            long_breaks.append(gt(bar["close"], upper[ref]) if ref >= 0 and upper[ref] is not None else None)
            short_breaks.append(lt(bar["close"], lower[ref]) if ref >= 0 and lower[ref] is not None else None)
            if i:
                signals[i]["long"] = long_breaks[i] is True and long_breaks[i-1] is False
                signals[i]["short"] = short_breaks[i] is True and short_breaks[i-1] is False
                exit_long = mid if middle else lower
                exit_short = mid if middle else upper
                signals[i]["close_long"] = _cross_down(bar["close"], exit_long[i], bars[i-1]["close"], exit_long[i-1])
                signals[i]["close_short"] = _cross_up(bar["close"], exit_short[i], bars[i-1]["close"], exit_short[i-1])
    return signals


def run_tradingview_backtest(candles, strategy, params=None, initial_capital=None,
                             position_pct=None, commission_rate=None):
    """Run one of the two ports. Rates and position_pct are fractions, not percent.

    Returns closed trades separately from any open position; equity includes
    mark-to-market P&L and fees. Same-direction entries are ignored (pyramiding=0).
    On reversal, the existing position closes and the new position opens at the
    same next-bar open; commissions apply to each transaction. Entry quantities
    are based on signal-bar equity and close, before the next open and exit fees.
    """
    candles = list(candles)
    signals = tradingview_signals(candles, strategy, params)
    defaults = next((getattr(module, "DEFAULTS", {}).get(strategy, {})
                     for module in _PORT_MODULES if strategy in module.SOURCES), {})
    initial_capital = defaults.get("initial_capital", 10000.0) if initial_capital is None else initial_capital
    position_pct = defaults.get("position_pct", 0.10) if position_pct is None else position_pct
    commission_rate = defaults.get("commission_rate", 0.00075) if commission_rate is None else commission_rate
    flat_only = defaults.get("entry_mode") == "flat_only"
    if not (isfinite(initial_capital) and initial_capital > 0 and isfinite(position_pct) and 0 < position_pct <= 1 and isfinite(commission_rate) and commission_rate >= 0):
        raise ValueError("invalid initial_capital, position_pct or commission_rate")
    cash = float(initial_capital)
    position = None
    pending = []
    trades, equity = [], []
    for i, bar in enumerate(candles):
        price = float(bar["open"])
        time = bar.get("time", i)
        for action, quantity in pending:
            direction = 1 if action == "long" else -1 if action == "short" else None
            if direction is not None and position and position["direction"] == direction:
                continue
            if position and (direction is not None or action == ("close_long" if position["direction"] == 1 else "close_short")):
                gross = position["direction"] * position["quantity"] * (price - position["entry_price"])
                exit_fee = position["quantity"] * price * commission_rate
                cash += gross - exit_fee
                trades.append({**position, "exit_time": time, "exit_price": price,
                               "gross_pnl": gross, "fees": position["entry_fee"] + exit_fee,
                               "pnl": gross - position["entry_fee"] - exit_fee})
                position = None
            if direction is not None and quantity > 0:
                fee = quantity * price * commission_rate
                cash -= fee
                position = {"direction": direction, "quantity": quantity, "entry_time": time,
                            "entry_price": price, "entry_fee": fee}
        mark = cash + (position["direction"] * position["quantity"] * (float(bar["close"]) - position["entry_price"]) if position else 0)
        equity.append({"time": time, "equity": mark})
        signal = signals[i]
        entry_signal = (signal["long"] or signal["short"]) and (not flat_only or position is None)
        if entry_signal and mark < 0:
            raise ValueError("cannot create a percent-of-equity order with negative equity")
        quantity = mark * position_pct / float(bar["close"])
        pending = [(key, quantity) for key in ("long", "short") if signal[key] and quantity > 0 and entry_signal]
        # strategy.close(id) only creates an order while that ID is open.
        if position and signal["close_long" if position["direction"] == 1 else "close_short"]:
            pending.append(("close_long" if position["direction"] == 1 else "close_short", 0))
    return {"strategy": strategy, "source": SOURCES[strategy], "params": dict(params or {}),
            "initial_capital": initial_capital, "position_pct": position_pct,
            "commission_rate": commission_rate, "equity": equity, "trades": trades,
            "open_position": position, "final_equity": equity[-1]["equity"] if equity else initial_capital,
            "unfilled_final_orders": [action for action, _ in pending]}
