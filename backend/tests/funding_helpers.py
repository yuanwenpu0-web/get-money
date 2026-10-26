"""资金费测试共享的行情/资金费帧构造helper（非测试用例，pytest 不收集）。"""

import numpy as np
import pandas as pd


def bar_frame(start="2026-01-01T00:00:00Z", periods=200, freq="1min", price=100.0):
    index = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    close = price + np.arange(periods) * 0.01
    return pd.DataFrame(
        dict(open=close, high=close + 1.0, low=close - 1.0, close=close, volume=100.0),
        index=index,
    )


def funding_frame(times, rates):
    return pd.DataFrame(
        {"funding_rate": rates, "mark_price": [100.0] * len(rates)},
        index=pd.to_datetime(times, unit="ms", utc=True),
    )
