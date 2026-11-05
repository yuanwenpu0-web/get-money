"""四条生效绑定的寻优配置：每条绑定只有一套参数，所以搜索空间按**整池**定义。

放在这里而不是脚本里，是为了让「跑哪几条、搜什么」与「怎么跑」分开：
`binding_combo_review.py` 只负责执行，报表与复现都引用同一份配置。
"""
from __future__ import annotations

COMBOS = [
    {
        "key": "A", "label": "研究·高成交额高波动（RSI或布林）",
        "code": "research_high_turnover_high_vol",
        "pool": "research_high_liquidity_high_volatility", "interval": "1h", "engine": "research",
        "grid": {"channel_kind": (0, 1), "research_hold_hours": (24, 48, 72, 96, 120, 168)},
        "current": {"channel_kind": 0, "research_hold_hours": 120},
    },
    {
        "key": "B", "label": "研究·低成交额高波动（RSI或ATR通道）",
        "code": "research_low_turnover_high_vol",
        "pool": "research_low_liquidity_high_volatility", "interval": "1h", "engine": "research",
        "grid": {"channel_kind": (0, 1), "research_hold_hours": (24, 48, 72, 96, 120, 168)},
        "current": {"channel_kind": 1, "research_hold_hours": 48},
    },
    {
        "key": "C", "label": "Supertrend 趋势反转 × 主流币",
        "code": "supertrend_current",
        "pool": "trend_reversal_fit", "interval": "15m", "engine": "generic",
        "grid": {"supertrend_factor": (2.0, 2.5, 3.0), "target_r": (3.0, 4.0, 6.0),
                 "sl_atr_mult": (1.5, 2.0), "use_adx_filter": (False, True)},
        "current": {"supertrend_factor": 2.5, "target_r": 3.0, "sl_atr_mult": 1.5,
                    "use_adx_filter": False},
    },
    {
        "key": "D", "label": "订单流净买入极端值 × 山寨",
        "code": "flow_imbalance_long",
        "pool": "alt", "interval": "15m", "engine": "generic", "require_taker": True,
        "grid": {"flow_window": (16, 48, 96), "entry_quantile": (0.90, 0.95, 0.98),
                 "max_hold_bars": (32, 64, 96, 128), "target_r": (2.0, 3.0),
                 "sl_atr_mult": (2.0, 3.0)},
        "current": {"flow_window": 48, "entry_quantile": 0.95, "max_hold_bars": 96,
                    "target_r": 2.0, "sl_atr_mult": 2.0},
    },
]


def by_key(key: str) -> dict:
    for combo in COMBOS:
        if combo["key"] == key:
            return combo
    raise KeyError(f"未知组合 {key}")


def expand_grid(space: dict) -> list[dict]:
    """笛卡尔积展开，空空间返回 [{}]（研究型策略的固定单点）。"""
    out = [{}]
    for key, values in space.items():
        out = [{**item, key: value} for item in out for value in values]
    return out
