"""MMT Ichi Workflow Overlay **Profiles** — 带符号档案的一目工作流。

与 `ichi_workflow_overlay` 的信号逻辑**完全一致**（四状态记分、四类 setup、
平目标扫描、活跃 setup 状态机），差异只有一处：在预设长度之上再加一层
**符号档案**（`use_symbol_profiles=True` 时按 `syminfo.ticker` 匹配）：

| Ticker | 档案 | Conversion / Base / SpanB / Displacement | 方向 |
|---|---|---|---|
| BTCUSD | BTC Long | 22 / 40 / 104 / 64 | Long Only |
| ETHUSD | ETH Long | 11 / 26 / 44 / 30 | Long Only |
| SOLUSD | SOL Long | 7 / 22 / 52 / 22 | Long Only |
| GLD | GLD Long | 11 / 30 / 44 / 26 | Long Only |
| SPY | SPY Long | 22 / 64 / 104 / 52 | Long Only |
| DBC | DBC Long | 7 / 26 / 44 / 30 | Long Only |
| LQD | LQD Short | 9 / 26 / 60 / 30 | **Short Only** |
| KRE | KRE Long | 24 / 61 / 96 / 36 | Long Only |
| 其他 | Manual | 用预设/自定义 | Both |

命中档案时长度与方向**都**被档案覆盖（方向过滤也一起换），未命中时退回
预设 + 手动方向过滤。符号名做 `upper()` 归一，`BTCUSD`/`BTCUSDT` 需调用方
自行映射成档案键。

丢弃：dashboard/table、markers、label、regime 背景着色（同基础版）。

Source: pine-crypto-collection/memeticmoney_ichimoku/pine/Ichi_Workflow_Overlay_Profiles.pine
License: memeticmoney 公开脚本（随清单收录）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import boolean, choice, flags, integer, ohlcv
from .ichi_workflow_overlay import ichi_workflow_overlay

__all__ = ["ichi_workflow_overlay_profiles", "SYMBOL_PROFILES"]

# ticker -> (label, conversion, base, spanB, displacement, direction)
SYMBOL_PROFILES: dict[str, tuple[str, int, int, int, int, str]] = {
    "BTCUSD": ("BTC Long", 22, 40, 104, 64, "Long Only"),
    "ETHUSD": ("ETH Long", 11, 26, 44, 30, "Long Only"),
    "SOLUSD": ("SOL Long", 7, 22, 52, 22, "Long Only"),
    "GLD": ("GLD Long", 11, 30, 44, 26, "Long Only"),
    "SPY": ("SPY Long", 22, 64, 104, 52, "Long Only"),
    "DBC": ("DBC Long", 7, 26, 44, 30, "Long Only"),
    "LQD": ("LQD Short", 9, 26, 60, 30, "Short Only"),
    "KRE": ("KRE Long", 24, 61, 96, 36, "Long Only"),
}


def ichi_workflow_overlay_profiles(
    frame: pd.DataFrame,
    *,
    preset: str = "Crypto Doubled",
    custom_conversion: int = 20,
    custom_base: int = 61,
    custom_span_b: int = 120,
    custom_displacement: int = 30,
    htf_rule: str = "D",
    use_symbol_profiles: bool = True,
    symbol: str | None = None,
    enable_breakout: bool = True,
    enable_checklist: bool = True,
    enable_kijun: bool = True,
    enable_edge: bool = True,
    direction_filter: str = "Long Only",
    flat_scan_bars: int = 250,
) -> pd.DataFrame:
    """按符号档案覆盖一目长度与方向，再复用基础版信号逻辑。

    Parameters
    ----------
    symbol:
        用于匹配档案的符号键（如 ``"BTCUSD"``）。为 None 时不做匹配，等价于
        未命中档案（Manual）。大小写不敏感。
    use_symbol_profiles:
        False 时完全忽略档案，退回预设 + ``direction_filter``。
    """
    preset = choice("preset", preset,
                    ("Crypto Doubled", "Traditional Standard",
                     "Traditional Doubled", "Custom"))
    custom_conversion = integer("custom_conversion", custom_conversion, 1)
    custom_base = integer("custom_base", custom_base, 2)
    custom_span_b = integer("custom_span_b", custom_span_b, 2)
    custom_displacement = integer("custom_displacement", custom_displacement, 1)
    use_symbol_profiles = boolean("use_symbol_profiles", use_symbol_profiles)
    enable_breakout = boolean("enable_breakout", enable_breakout)
    enable_checklist = boolean("enable_checklist", enable_checklist)
    enable_kijun = boolean("enable_kijun", enable_kijun)
    enable_edge = boolean("enable_edge", enable_edge)
    direction_filter = choice("direction_filter", direction_filter,
                              ("Both", "Long Only", "Short Only"))
    flat_scan_bars = integer("flat_scan_bars", flat_scan_bars, 0, 250)

    # 预设解析（与基础版同一张表）。
    if preset == "Custom":
        manual = (custom_conversion, custom_base, custom_span_b,
                  custom_displacement)
    else:
        manual = {"Crypto Doubled": (20, 61, 120, 30),
                  "Traditional Standard": (9, 26, 52, 26),
                  "Traditional Doubled": (18, 52, 104, 52)}[preset]

    profile = None
    if use_symbol_profiles and symbol:
        profile = SYMBOL_PROFILES.get(symbol.upper())

    if profile is not None:
        label, conv, base, span_b_len, displacement, direction = profile
        using_profile = True
        effective_direction = direction
    else:
        label, conv, base, span_b_len, displacement = ("Manual", *manual)
        using_profile = False
        effective_direction = direction_filter

    core = ichi_workflow_overlay(
        frame,
        preset="Custom",
        custom_conversion=conv,
        custom_base=base,
        custom_span_b=span_b_len,
        custom_displacement=displacement,
        htf_rule=htf_rule,
        enable_breakout=enable_breakout,
        enable_checklist=enable_checklist,
        enable_kijun=enable_kijun,
        enable_edge=enable_edge,
        direction_filter=effective_direction,
        flat_scan_bars=flat_scan_bars,
    )

    index = frame.index
    # setup 编码：Checklist ±1、Kijun ±2、Breakout ±3、Edge ±4，None 为 0。
    setup_codes = {"None": 0, "Checklist Long": 1, "Checklist Short": -1,
                   "Kijun Bounce Long": 2, "Kijun Bounce Short": -2,
                   "Breakout Long": 3, "Breakout Short": -3,
                   "Edge-to-Edge Long": 4, "Edge-to-Edge Short": -4}
    code = core["active_setup"].map(setup_codes).fillna(0).astype("int64")

    enriched = core.copy()
    enriched["profile_label"] = pd.Series(label, index=index, dtype=object)
    enriched["using_profile"] = pd.Series(using_profile, index=index, dtype=bool)
    enriched["effective_conversion"] = pd.Series(conv, index=index, dtype="int64")
    enriched["effective_base"] = pd.Series(base, index=index, dtype="int64")
    enriched["effective_span_b"] = pd.Series(span_b_len, index=index, dtype="int64")
    enriched["effective_displacement"] = pd.Series(displacement, index=index,
                                                   dtype="int64")
    enriched["effective_direction"] = pd.Series(effective_direction, index=index,
                                                dtype=object)
    enriched["setup_code"] = code
    return enriched
