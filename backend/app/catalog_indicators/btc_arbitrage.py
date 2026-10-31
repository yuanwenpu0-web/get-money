# SPDX-License-Identifier: MPL-2.0
"""BTC Arbitrage, adapted from syoungs' MPL-2.0 Pine source.

Source: https://github.com/KayK-Pips/tradingview-indicatorz/blob/5df3055fe8a3e7c52993241204a865130f60236b/other/01-btc-arbitrage.pine
The caller supplies aligned one-minute close series; data fetching and the
TradingView chart's lower-timeframe bar merging are outside this function.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def btc_arbitrage(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"cme_close", "binance_close", "upbit_krw_close", "usdkrw_close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"缺少行情列: {', '.join(sorted(missing))}")
    cme, binance, upbit, usdkrw = (pd.to_numeric(frame[col], errors="raise").astype(float)
                                     for col in ("cme_close", "binance_close", "upbit_krw_close", "usdkrw_close"))
    cme_premium = cme - binance
    upbit_premium = upbit / usdkrw.replace(0, np.nan) - binance
    denominator = binance.replace(0, np.nan)
    return pd.DataFrame({"cme_premium": cme_premium,
                         "cme_percent": cme_premium / denominator * 100,
                         "upbit_premium": upbit_premium,
                         "upbit_percent": upbit_premium / denominator * 100}, index=frame.index)
