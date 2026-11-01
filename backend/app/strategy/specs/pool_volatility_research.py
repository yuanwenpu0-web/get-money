"""研究组合信号及专用模拟执行规则。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...research.lab_signals import rsi_reversal, bollinger_break, atr_channel_break
from ..registry import StrategySpec, register

HIGH_CODE = "research_high_turnover_high_vol"
LOW_CODE = "research_low_turnover_high_vol"
BLOCKED_CODES = (HIGH_CODE, LOW_CODE)


def prepare(frame, params, ctx):
    if "research_entry" in frame.columns:
        return frame
    data = frame.copy()
    rsi = rsi_reversal(data, {"n": 21, "low": 30, "high": 70}).long
    if params["channel_kind"] == 0:
        channel = bollinger_break(data, {"n": 35, "k": 2.0}).long
    else:
        channel = atr_channel_break(data, {"n": 20, "k": 2.0}).long
    valid = np.arange(len(data)) >= 200
    if isinstance(data.index, pd.DatetimeIndex):
        gaps = np.flatnonzero(data.index.to_series().diff().dt.total_seconds().to_numpy() != 3600)
        for gap in gaps:
            valid[gap:min(gap + 200, len(data))] = False
    else:
        valid[:] = False
    data["research_entry"] = (rsi | channel) & valid
    return data


def signals(data, params):
    entry = data["research_entry"].astype(bool)
    return pd.DataFrame({
        "signed": entry.astype(float), "score": 0.5, "prob_up": 0.5,
        "direction": 0, "block_reason": "",
        "position_pct": 0.0, "leverage": 0, "sl_pct": 0.0,
        "tp_pct": 0.0, "risk_pct": 0.0, "net_reward_risk": 0.0,
    }, index=data.index)


def describe(row):
    return {"research_entry": bool(row.get("research_entry", False))}


def validate_high(params):
    if params["channel_kind"] != 0 or params["research_hold_hours"] != 120:
        raise ValueError("研究策略参数固定为 Bollinger35/2 和 120 小时")


def validate_low(params):
    if params["channel_kind"] != 1 or params["research_hold_hours"] != 48:
        raise ValueError("研究策略参数固定为 ATR通道20/2 和 48 小时")


for code, name, kind, hours, validate in (
    (HIGH_CODE, "研究：高成交额高波动 RSI或布林", 0, 120, validate_high),
    (LOW_CODE, "研究：低成交额高波动 RSI或ATR通道", 1, 48, validate_low),
):
    register(StrategySpec(
        code=code, name=name,
        description=f"1小时收盘产生做多信号，模拟盘在下一根实际小时线开盘价成交，持有{hours}小时后下一开盘出场；实盘开仓暂不支持。",
        defaults={"channel_kind": kind, "research_hold_hours": hours},
        ranges={"channel_kind": (0, 1), "research_hold_hours": (48, 120)},
        int_keys=("channel_kind", "research_hold_hours"),
        lookback=lambda params: 1000, prepare=prepare, signals=signals,
        describe=describe, validate=validate, default_enabled=False,
        tags=("research",),
    ))
