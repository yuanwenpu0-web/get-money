"""Crypto Intermarket Correlation — BTC/ETH correlation against an outside benchmark.

Source: pine-crypto-collection/realanthony_cic/cic.pine (Crypto Intermarket Correlation, MIT)

The script pulls three extra series (`request.security`) and correlates their log
returns against a TradFi benchmark. Here the caller supplies them as frame columns
`btc_close`, `eth_close`, `bench_close` — already aligned to the same bars — so the
function stays offline. The two correlations are Fisher-z averaged into a single
regime score, optionally smoothed, and classified as Sync / Inverse-Sync / Loose /
Decouple.

Dropped: the right-hand table, its colours and the regime background. Pine's "Auto"
regime smoothing depends on the chart timeframe; pass `regime_smoothing_bars`
explicitly (5-minute 8, 15–30-minute 5, hourly and above 3, daily 3, weekly/monthly 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import choice, correlation, integer, number, ohlcv, sma


def _fisher_average(left: pd.Series, right: pd.Series) -> pd.Series:
    """Average two correlations in Fisher-z space, falling back to whichever exists."""
    a = left.clip(-0.999, 0.999)
    b = right.clip(-0.999, 0.999)
    averaged = np.tanh((np.arctanh(a) + np.arctanh(b)) / 2.0)
    values = pd.Series(averaged, index=left.index)
    values = values.where(left.notna() & right.notna(), left.fillna(right))
    return values


def _state(values: pd.Series, sync: float, decouple: float) -> pd.Series:
    labels = np.where(
        values.isna(), "NA",
        np.where(values >= sync, "Sync",
                 np.where(values <= -sync, "Inverse Sync",
                          np.where(values.abs() >= decouple, "Loose", "Decouple"))))
    return pd.Series(labels, index=values.index, dtype=object)


def cic(frame: pd.DataFrame, *, length: int = 30, mode: str = "Returns",
        sync_threshold: float = 0.50, decouple_threshold: float = 0.25,
        regime_smoothing_bars: int = 1) -> pd.DataFrame:
    integer("length", length, 15)
    choice("mode", mode, ("Returns", "Price"))
    number("sync_threshold", sync_threshold, 0.0, 1.0)
    number("decouple_threshold", decouple_threshold, 0.0, 1.0)
    integer("regime_smoothing_bars", regime_smoothing_bars, 1, 200)

    btc, eth, bench = ohlcv(frame, "btc_close", "eth_close", "bench_close")
    if mode == "Returns":
        def log_return(series: pd.Series) -> pd.Series:
            previous = series.shift(1)
            ratio = (series / previous).where(series.gt(0) & previous.gt(0))
            return np.log(ratio)
        btc, eth, bench = log_return(btc), log_return(eth), log_return(bench)

    btc_corr = correlation(btc, bench, length)
    eth_corr = correlation(eth, bench, length)
    average = _fisher_average(btc_corr, eth_corr)
    regime = sma(average, regime_smoothing_bars) if regime_smoothing_bars > 1 else average

    return pd.DataFrame({
        "btc_corr": btc_corr,
        "eth_corr": eth_corr,
        "average_corr": average,
        "regime_corr": regime,
        "btc_state": _state(btc_corr, sync_threshold, decouple_threshold),
        "eth_state": _state(eth_corr, sync_threshold, decouple_threshold),
        "regime_state": _state(regime, sync_threshold, decouple_threshold),
    }, index=frame.index)
