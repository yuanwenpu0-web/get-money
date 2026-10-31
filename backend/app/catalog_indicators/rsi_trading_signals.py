"""RSI Trading Signals, adapted from spustoszenie/tradingview-indicators.

Source: https://github.com/spustoszenie/tradingview-indicators/blob/d14393f824684c37d4a77a48192f2126760f9868/RSI_Trading_Signals.pine
The upstream README declares MIT. Drawing and alert delivery are outside the
calculation; the corresponding numerical and boolean series are returned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi_trading_signals(frame: pd.DataFrame, *, rsi_length: int = 14,
                        rsi_overbought: int = 70, rsi_oversold: int = 30,
                        show_signals: bool = True) -> pd.DataFrame:
    for key, value, low, high in (
        ("rsi_length", rsi_length, 1, None),
        ("rsi_overbought", rsi_overbought, 50, 100),
        ("rsi_oversold", rsi_oversold, 0, 50),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < low or (high is not None and value > high):
            raise ValueError(f"参数 {key} 超出允许范围")
    if not isinstance(show_signals, bool):
        raise ValueError("参数 show_signals 必须为布尔值")
    if "close" not in frame:
        raise ValueError("缺少 close 列")
    close = pd.to_numeric(frame["close"], errors="raise").astype(float)
    change = close.diff()

    def rma(values: pd.Series) -> pd.Series:
        # Pine's RMA starts with the mean of the first length valid changes.
        out = np.full(len(values), np.nan)
        count, total, current = 0, 0.0, np.nan
        for i, value in enumerate(values.to_numpy()):
            if np.isnan(value):
                out[i] = current
                continue
            if count < rsi_length:
                count += 1
                total += value
                if count == rsi_length:
                    current = total / rsi_length
            else:
                current = (current * (rsi_length - 1) + value) / rsi_length
            out[i] = current
        return pd.Series(out, index=values.index)

    gain, loss = rma(change.clip(lower=0)), rma(-change.clip(upper=0))
    rsi = 100 - 100 / (1 + gain / loss)
    previous = rsi.shift(1)
    oversold, overbought = rsi < rsi_oversold, rsi > rsi_overbought
    return pd.DataFrame({
        "rsi": rsi,
        "oversold": oversold,
        "overbought": overbought,
        "buy_signal": oversold & ~oversold.shift(1, fill_value=False),
        "sell_signal": overbought & ~overbought.shift(1, fill_value=False),
        "strong_buy": (rsi > rsi_oversold) & (previous <= rsi_oversold),
        "strong_sell": (rsi < rsi_overbought) & (previous >= rsi_overbought),
    }, index=frame.index)
