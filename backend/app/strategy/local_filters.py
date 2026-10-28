"""TradingView research ports used only as optional system-entry filters."""
from __future__ import annotations

import numpy as np

from ..backtest.tradingview_native import SOURCES, tradingview_signals
from ..market.data_service import MarketDataService
from ..state import get_state
from ..utils import interval_ms, now_ms

CATALOG = {
    "sma_crossover": ("SMA 交叉（50/200）", "快慢 SMA 交叉时产生同方向开仓过滤信号"),
    "donchian_channel": ("Donchian 通道突破", "通道突破时产生同方向开仓过滤信号"),
}


def config_for(db, code):
    value = get_state(db, f"local_filter:{code}", {})
    return value if isinstance(value, dict) else {}


def list_filters(db):
    result = []
    for code, (name, description) in CATALOG.items():
        config = config_for(db, code)
        result.append({"code": code, "name": name, "description": description,
                       "source": SOURCES[code], "enabled": config.get("enabled", False),
                       "bound_strategies": config.get("bound_strategies", []),
                       "params": config.get("params", {})})
    return result


class EntryFilters:
    """Evaluate all enabled bindings on the same completed candle as the signal.

    An unavailable, invalid, stale, or discontinuous history fails closed. This
    class never modifies the strategy signal used to manage an open position.
    """
    def __init__(self, db):
        self.filters = [item for item in list_filters(db) if item["enabled"]]
        self.service = MarketDataService(db)
        self.cache = {}

    def allows(self, signal):
        selected = [f for f in self.filters if signal.strategy in f["bound_strategies"]]
        if not selected:
            return True
        key = (signal.symbol, signal.interval, signal.ts)
        if key not in self.cache:
            step = interval_ms(signal.interval)
            frame = self.service.load_frame(signal.symbol, signal.interval, limit=1010)
            if not frame.empty:
                frame = frame.loc[frame.index.as_unit("ms").asi8 + step <= now_ms()]
                frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
            if frame.empty or int(frame.index[-1].timestamp() * 1000) != signal.ts:
                self.cache[key] = None
            else:
                # Match dataset validation before segmenting. Preserve the initial
                # latest-bar check above so a bad final candle cannot be skipped.
                fields = ["open", "high", "low", "close", "volume"]
                if not set(fields).issubset(frame.columns):
                    self.cache[key] = None
                    return False
                valid = np.isfinite(frame[fields]).all(axis=1)
                valid &= (frame[["open", "high", "low", "close"]] > 0).all(axis=1) & frame["volume"].ge(0)
                valid &= frame["high"].ge(frame[["open", "close", "low"]].max(axis=1))
                valid &= frame["low"].le(frame[["open", "close", "high"]].min(axis=1))
                frame = frame.loc[valid]
                if frame.empty or int(frame.index[-1].timestamp() * 1000) != signal.ts:
                    self.cache[key] = None
                    return False
                # Restart indicators at the most recent gap; never bridge missing candles.
                diffs = frame.index.to_series().diff().dt.total_seconds().mul(1000)
                frame = frame.loc[diffs.ne(step).cumsum().eq(diffs.ne(step).cumsum().iloc[-1])]
                self.cache[key] = frame[["open", "high", "low", "close"]].to_dict("records")
        candles = self.cache[key]
        if not candles:
            return False
        direction = "long" if signal.direction == "LONG" else "short"
        for item in selected:
            signals = tradingview_signals(candles, item["code"], item["params"])
            if not signals or not signals[-1][direction]:
                return False
        return True
