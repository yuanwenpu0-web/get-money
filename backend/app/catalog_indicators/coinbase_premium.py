"""Asymmetric Coinbase Premium Histogram with SMA, by joemccann.

Source: https://gist.github.com/joemccann/716d647941a4c018eb1f2da0d54a812c/4cb29e6b5a658d45b1203d2becce191bd7244290
The author declares MIT / Apache 2.0 dual licensing at
https://gist.github.com/joemccann/716d647941a4c018eb1f2da0d54a812c?permalink_comment_id=5372099.
The caller provides both price series already aligned to the intended chart bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def coinbase_premium(frame: pd.DataFrame, *, show_percent: bool = False,
                     sma_length: int = 14) -> pd.DataFrame:
    if not isinstance(show_percent, bool):
        raise ValueError("参数 show_percent 必须为布尔值")
    if isinstance(sma_length, bool) or not isinstance(sma_length, int) or sma_length < 1:
        raise ValueError("参数 sma_length 必须为正整数")
    missing = {"coinbase_close", "binance_close"} - set(frame.columns)
    if missing:
        raise ValueError(f"缺少行情列: {', '.join(sorted(missing))}")
    coinbase = pd.to_numeric(frame["coinbase_close"], errors="raise").astype(float)
    binance = pd.to_numeric(frame["binance_close"], errors="raise").astype(float)
    premium = coinbase - binance
    percentage = premium / binance.replace(0, np.nan) * 100
    histogram = percentage if show_percent else premium
    # Pine's ta.sma counts non-na observations, including across gaps.
    positional = pd.Series(histogram.to_numpy())
    valid_sma = positional.dropna().rolling(sma_length).mean()
    sma = valid_sma.reindex(positional.index).ffill().to_numpy()
    return pd.DataFrame({"premium": premium, "percent_premium": percentage,
                         "histogram": histogram, "sma": sma},
                        index=frame.index)
