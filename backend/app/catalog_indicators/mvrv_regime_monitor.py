"""MVRV Regime Monitor — cycle zones driven by Bitcoin's market-value-to-realised-value.

Source: pine-crypto-collection/lonelyobserver0/mvrv_regime_monitor.pine (MVRV Regime Monitor, free tier)

MVRV lives outside OHLCV, so the caller passes it as a `mvrv` column (one value per
bar, already aligned; the upstream script requests the daily `INTOTHEBLOCK:BTC_MVRV`
series and lets TradingView forward-fill it onto intraday charts).

Each bar is labelled ENTRY ZONE / ACCUMULATION / DISTRIBUTION RISK / EXIT ZONE /
EUPHORIA by four thresholds, and the three threshold crossings the alerts fire on are
exported as flags. A missing MVRV prints `—` and never fires.

Dropped: the regime background, the last-bar label, the 2017/2018/2021/2022 cycle
hint lines and alert dialogs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import flags, number, ohlcv

_ZONES = ("ENTRY ZONE", "ACCUMULATION", "DISTRIBUTION RISK", "EXIT ZONE", "EUPHORIA")


def mvrv_regime_monitor(frame: pd.DataFrame, *, entry_threshold: float = 1.0,
                        accumulation_threshold: float = 2.0, exit_threshold: float = 2.5,
                        euphoria_threshold: float = 3.5) -> pd.DataFrame:
    number("entry_threshold", entry_threshold, 0.0, 10.0)
    number("accumulation_threshold", accumulation_threshold, 0.0, 10.0)
    number("exit_threshold", exit_threshold, 0.0, 10.0)
    number("euphoria_threshold", euphoria_threshold, 0.0, 10.0)
    if not entry_threshold < accumulation_threshold < exit_threshold < euphoria_threshold:
        raise ValueError("参数阈值必须严格递增: entry < accumulation < exit < euphoria")

    (mvrv,) = ohlcv(frame, "mvrv")
    bounds = [entry_threshold, accumulation_threshold, exit_threshold, euphoria_threshold]
    regime = np.select([mvrv < bounds[0], mvrv < bounds[1], mvrv < bounds[2], mvrv < bounds[3]],
                       list(_ZONES[:4]), default=_ZONES[4])
    regime = pd.Series(regime, index=frame.index, dtype=object).where(mvrv.notna(), "—")

    previous = mvrv.shift(1)
    valid = mvrv.notna() & previous.notna()
    return pd.DataFrame({
        "mvrv": mvrv,
        "regime": regime,
        "cross_under_entry": flags(valid & (previous >= entry_threshold) & (mvrv < entry_threshold)),
        "cross_over_exit": flags(valid & (previous <= exit_threshold) & (mvrv > exit_threshold)),
        "cross_over_euphoria": flags(valid & (previous <= euphoria_threshold) & (mvrv > euphoria_threshold)),
    }, index=frame.index)
