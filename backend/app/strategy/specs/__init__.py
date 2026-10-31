"""内置策略清单：导入即注册。

新增 TradingView 迁移过来的策略时，在这里加一行 import 即可（导入副作用完成注册），
不需要改 engine / 巡检 / 回测 / API。
"""

from __future__ import annotations

from . import absorption_reversal  # noqa: F401
from . import opening_candle_breakout  # noqa: F401
from . import rgd_donchian_v2  # noqa: F401
from . import supertrend_current  # noqa: F401
from . import simple_zigzag_double  # noqa: F401

__all__ = [
    "absorption_reversal",
    "opening_candle_breakout",
    "rgd_donchian_v2",
    "supertrend_current",
    "simple_zigzag_double",
]
