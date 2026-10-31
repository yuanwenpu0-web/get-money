"""MMT Ichi 2D Overlay **Profiles** — 双日（2D）图表专用的符号档案工作流。

与 `ichi_workflow_overlay_profiles` 同一套信号逻辑，专为 **2D 图表** 标定：

1. 档案长度全部按 2D 周期重新拟合（见下表）。
2. **每个档案自带确认周期**（`confirmation_timeframe`）：多数用 `D`，DBC/LQD
   用 `W`；未命中档案时退回输入的 `htf_rule`（默认 `W`）。
3. 纯展示的「非 2D 图表」警告（`chart_is_two_day`）保留为输出列，便于调用方
   判断是否用错了周期。

| Ticker | 档案 | Conv / Base / SpanB / Disp | 确认周期 | 方向 |
|---|---|---|---|---|
| BTCUSD | BTC 2D Long | 20 / 73 / 96 / 36 | D | Long |
| ETHUSD | ETH 2D Long | 18 / 64 / 80 / 52 | D | Long |
| SOLUSD | SOL 2D Long | 9 / 30 / 52 / 30 | D | Long |
| GLD | GLD 2D Long | 14 / 64 / 128 / 40 | D | Long |
| SPY | SPY 2D Long | 14 / 64 / 80 / 52 | D | Long |
| DBC | DBC 2D Long | 7 / 30 / 60 / 26 | **W** | Long |
| LQD | LQD 2D Short | 7 / 30 / 60 / 30 | **W** | **Short** |
| KRE | KRE 2D Long | 11 / 26 / 44 / 22 | D | Long |

丢弃：dashboard/table、markers、label、regime 着色、警告样式（只留布尔列）。

Source: pine-crypto-collection/memeticmoney_ichimoku/pine/Ichi_2D_Overlay_Profiles.pine
License: memeticmoney 公开脚本（随清单收录）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import boolean, choice, integer, ohlcv
from .ichi_workflow_overlay import ichi_workflow_overlay

__all__ = ["ichi_2d_overlay_profiles", "SYMBOL_PROFILES_2D"]

# ticker -> (label, conversion, base, spanB, displacement, htf_rule, direction)
SYMBOL_PROFILES_2D: dict[str, tuple[str, int, int, int, int, str, str]] = {
    "BTCUSD": ("BTC 2D Long", 20, 73, 96, 36, "D", "Long Only"),
    "ETHUSD": ("ETH 2D Long", 18, 64, 80, 52, "D", "Long Only"),
    "SOLUSD": ("SOL 2D Long", 9, 30, 52, 30, "D", "Long Only"),
    "GLD": ("GLD 2D Long", 14, 64, 128, 40, "D", "Long Only"),
    "SPY": ("SPY 2D Long", 14, 64, 80, 52, "D", "Long Only"),
    "DBC": ("DBC 2D Long", 7, 30, 60, 26, "W", "Long Only"),
    "LQD": ("LQD 2D Short", 7, 30, 60, 30, "W", "Short Only"),
    "KRE": ("KRE 2D Long", 11, 26, 44, 22, "D", "Long Only"),
}

_SETUP_CODES = {"None": 0, "Checklist Long": 1, "Checklist Short": -1,
                "Kijun Bounce Long": 2, "Kijun Bounce Short": -2,
                "Breakout Long": 3, "Breakout Short": -3,
                "Edge-to-Edge Long": 4, "Edge-to-Edge Short": -4}


def ichi_2d_overlay_profiles(
    frame: pd.DataFrame,
    *,
    preset: str = "Crypto Doubled",
    custom_conversion: int = 20,
    custom_base: int = 61,
    custom_span_b: int = 120,
    custom_displacement: int = 30,
    htf_rule: str = "W",
    use_symbol_profiles: bool = True,
    symbol: str | None = None,
    enable_breakout: bool = True,
    enable_checklist: bool = True,
    enable_kijun: bool = True,
    enable_edge: bool = True,
    direction_filter: str = "Long Only",
    flat_scan_bars: int = 250,
    chart_is_two_day: bool = True,
) -> pd.DataFrame:
    """按 2D 档案覆盖一目长度/确认周期/方向，再复用基础版信号逻辑。"""
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
    chart_is_two_day = boolean("chart_is_two_day", chart_is_two_day)

    if preset == "Custom":
        manual = (custom_conversion, custom_base, custom_span_b,
                  custom_displacement)
    else:
        manual = {"Crypto Doubled": (20, 61, 120, 30),
                  "Traditional Standard": (9, 26, 52, 26),
                  "Traditional Doubled": (18, 52, 104, 52)}[preset]

    profile = None
    if use_symbol_profiles and symbol:
        profile = SYMBOL_PROFILES_2D.get(symbol.upper())

    if profile is not None:
        label, conv, base, span_b_len, displacement, conf_rule, direction = profile
        using_profile = True
        effective_direction = direction
    else:
        label = "Manual"
        conv, base, span_b_len, displacement = manual
        conf_rule = htf_rule
        using_profile = False
        effective_direction = direction_filter

    core = ichi_workflow_overlay(
        frame,
        preset="Custom",
        custom_conversion=conv,
        custom_base=base,
        custom_span_b=span_b_len,
        custom_displacement=displacement,
        htf_rule=conf_rule,
        enable_breakout=enable_breakout,
        enable_checklist=enable_checklist,
        enable_kijun=enable_kijun,
        enable_edge=enable_edge,
        direction_filter=effective_direction,
        flat_scan_bars=flat_scan_bars,
    )

    index = frame.index
    enriched = core.copy()
    enriched["profile_label"] = pd.Series(label, index=index, dtype=object)
    enriched["using_profile"] = pd.Series(using_profile, index=index, dtype=bool)
    enriched["effective_conversion"] = pd.Series(conv, index=index, dtype="int64")
    enriched["effective_base"] = pd.Series(base, index=index, dtype="int64")
    enriched["effective_span_b"] = pd.Series(span_b_len, index=index, dtype="int64")
    enriched["effective_displacement"] = pd.Series(displacement, index=index,
                                                   dtype="int64")
    enriched["confirmation_rule"] = pd.Series(conf_rule, index=index, dtype=object)
    enriched["effective_direction"] = pd.Series(effective_direction, index=index,
                                                dtype=object)
    enriched["setup_code"] = (core["active_setup"].map(_SETUP_CODES)
                              .fillna(0).astype("int64"))
    enriched["chart_is_two_day"] = pd.Series(chart_is_two_day, index=index,
                                             dtype=bool)
    return enriched
