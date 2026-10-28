"""内置策略清单：导入即注册。

新增 TradingView 迁移过来的策略时，在这里加一行 import 即可（导入副作用完成注册），
不需要改 engine / 巡检 / 回测 / API。
"""

from __future__ import annotations

from . import absorption_reversal  # noqa: F401

__all__ = ["absorption_reversal"]
