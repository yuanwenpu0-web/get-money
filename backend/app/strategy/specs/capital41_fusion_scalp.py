"""Capital41 Fusion Scalp v3（MIT，Copyright 2026 Capital41）的交易适配。

原指标只在已收盘 K 线上给出多空信号，没有出场规则；止损和止盈使用项目通用 ATR 风控。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...catalog_indicators.capital41_fusion_scalp import capital41_fusion_scalp
from ...features.funding import FUNDING_FILTER_Z
from ..registry import PrepareContext, StrategySpec, register
from ..signals import assemble_signal_frame, block, column

CODE = "capital41_fusion_scalp"
DEFAULTS = {
    "ema_length": 50, "rsi_length": 14, "rsi_overbought": 65, "rsi_oversold": 35,
    "wr_length": 14, "wr_overbought": -20.0, "wr_oversold": -80.0,
    "require_turn": True, "atr_length": 14, "atr_average_length": 20,
    "atr_expansion_multiplier": 1.1, "volume_length": 50, "volume_multiplier": 1.0,
    "min_hold_bars": 0, "max_hold_bars": 0, "cooldown_bars": 0,
    "trailing_enabled": False,
}
RANGES = {
    "ema_length": (5, 10000), "rsi_length": (2, 10000),
    "rsi_overbought": (50, 90), "rsi_oversold": (10, 50),
    "wr_length": (2, 10000), "wr_overbought": (-50, 0), "wr_oversold": (-100, -50),
    "atr_length": (1, 10000), "atr_average_length": (5, 10000),
    "atr_expansion_multiplier": (1, 3), "volume_length": (5, 10000),
    "volume_multiplier": (0.5, 3),
}
PREPARE_KEYS = tuple(k for k in DEFAULTS if k not in (
    "min_hold_bars", "max_hold_bars", "cooldown_bars", "trailing_enabled",
))


def lookback(params: dict) -> int:
    return max(600, 4 * max(int(params[k]) for k in (
        "ema_length", "rsi_length", "wr_length", "atr_length", "volume_length",
    )), int(params["atr_length"]) + int(params["atr_average_length"]) + 1)


def prepare(frame: pd.DataFrame, params: dict, ctx: PrepareContext) -> pd.DataFrame:
    if {"fusion_scalp_long", "fusion_scalp_short", "fusion_scalp_atr", "atr_pct"}.issubset(frame.columns):
        return frame
    data = frame.copy()
    indicator = capital41_fusion_scalp(data, **{key: params[key] for key in PREPARE_KEYS})
    data["fusion_scalp_long"] = indicator["long_signal"]
    data["fusion_scalp_short"] = indicator["short_signal"]
    data["fusion_scalp_atr"] = indicator["atr"]
    data["atr_pct"] = indicator["atr"] / column(data, "close").replace(0, np.nan)
    return data


def signals(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    long_signal = column(data, "fusion_scalp_long").fillna(0).ne(0)
    short_signal = column(data, "fusion_scalp_short").fillna(0).ne(0)
    signed = long_signal.astype(float) - short_signal.astype(float)
    close, atr = column(data, "close"), column(data, "fusion_scalp_atr")
    reason = pd.Series("", index=data.index, dtype=object)
    valid = np.isfinite(close) & np.isfinite(atr) & close.gt(0) & atr.gt(0)
    reason = block(reason, ~valid, "invalid_indicators")
    reason = block(reason, signed.eq(0), "no_fusion_scalp_signal")
    atr_pct = atr / close.replace(0, np.nan)
    reason = block(reason, atr_pct.gt(params["max_atr_pct"]), "excess_volatility")
    if params["funding_filter"]:
        funding = column(data, "funding_zscore")
        reason = block(reason, funding.ge(FUNDING_FILTER_Z) & signed.gt(0), "funding_filter_long")
        reason = block(reason, funding.le(-FUNDING_FILTER_Z) & signed.lt(0), "funding_filter_short")
    return assemble_signal_frame(
        data, params, signed=signed, score=pd.Series(1.0, index=data.index),
        prob_up=pd.Series(0.5, index=data.index),
        sl_pct=atr_pct * params["sl_atr_mult"], tp_pct=atr_pct * params["tp_atr_mult"],
        reason=reason,
    )


SPEC = register(StrategySpec(
    code=CODE, name="Capital41 Fusion Scalp",
    description=("EMA 趋势、RSI 与 Williams %R 条件及 ATR 和成交量扩张同时满足时产生信号。"
                 "原指标无出场规则，本地使用可配置的通用 ATR 止损止盈。"),
    defaults=DEFAULTS, ranges=RANGES, lookback=lookback, prepare=prepare, signals=signals,
    bool_keys=("require_turn",),
    int_keys=("ema_length", "rsi_length", "rsi_overbought", "rsi_oversold", "wr_length",
              "atr_length", "atr_average_length", "volume_length"),
    prepare_keys=PREPARE_KEYS, default_enabled=False, tags=("tradingview", "scalp"),
))
