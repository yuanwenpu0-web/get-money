"""Regime-Gated Donchian v2 [RGD]：Pine v6 的本地 spec（契约 §6 逐项落地）。

`defaults` 是脚本 L17-52 的 1:1 默认值；`dir_mode`/`htf_in` 因项目 `resolve_params`
只收数值而编码成 0/1/2 枚举（契约 §8 R2），语义与 Pine 的字符串选项一一对应。

`signals()` 强制 `direction=0` + `block_reason="stop_order_requires_dedicated_broker"`：
本策略的出场是「TP1 限价部分平 + 保本滞后一根 + 接受度加权 trailing」，
通用 `execution.simulate` 无法表达，误接时必须零成交（安全失败），
由 `backtest/rgd_execution.simulate_rgd` 专用撮合负责真实回放（契约 §5.1 / A23）。
"""
from __future__ import annotations

import math

import pandas as pd

from ...features.rgd import DECISION_COLUMNS, build_features
from ..registry import PrepareContext, StrategySpec, register

CODE = "rgd_donchian_v2"
NAME = "Regime-Gated Donchian v2 [RGD]"

DEFAULTS = {
    # —— 脚本 1:1（Pine L17-52）——
    "dir_mode": 0,          # 0=Long & Short 1=Long Only 2=Short Only
    "use_pull": True,
    "cooldown": 5,          # 与项目 cooldown_bars 无关：只门控 pullback
    "brk_fast": 14, "brk_slow": 34, "atr_len": 14, "rank_len": 100,
    "rank_cut": 50.0, "er_len": 10, "er_min": 0.30, "brk_buf": 0.10,
    "use_ten": True, "htf_in": 1, "auto_htf": True,   # htf_in: 0=1h 1=4h 2=1d
    "build_len": 6, "rel_len": 20, "max_ten": 8.25, "ten_th": 0.20,
    "use_corr": True, "corr_len": 30, "corr_min": 0.25,
    "use_vol": True, "vol_min": 1.0,
    "def_fast": 8, "def_slow": 21, "acc_len": 6, "max_depth": 0.70,
    "risk_pct": 1.0, "max_stop_atr": 2.5, "min_stop_atr": 0.5,
    "tp1_r": 2.0, "tp1_pct": 50.0,
    # —— 项目必需、RGD 不用：中和以免通用管线误触发 ——
    "cost_filter": False, "regime_filter": False, "funding_filter": False,
    "trailing_enabled": False,
    "max_hold_bars": 0, "min_hold_bars": 0, "cooldown_bars": 0,
    "max_signal_age_bars": 2, "min_atr_cost_ratio": 0.5,
    "min_net_reward_risk": 1.0, "max_atr_pct": 0.03, "trail_activation_atr": 1.5,
}
# fee_rate / slippage 严禁写入 DEFAULTS 或 overrides：rules.py:105-109 硬禁止，
# 只能走全局设置或回测请求的 --fee / --slippage。

RANGES = {
    "dir_mode": (0, 2), "cooldown": (0, 10000),
    "brk_fast": (2, 500), "brk_slow": (3, 500), "atr_len": (2, 500),
    "rank_len": (20, 2000), "rank_cut": (10, 90), "er_len": (2, 500),
    "er_min": (0.05, 0.95), "brk_buf": (0.0, 20.0),
    "htf_in": (0, 2), "build_len": (1, 100), "rel_len": (1, 100),
    "max_ten": (0.25, 10.0), "ten_th": (0.05, 2.0),
    "corr_len": (5, 500), "corr_min": (0.0, 0.95), "vol_min": (0.1, 100.0),
    "def_fast": (2, 500), "def_slow": (3, 500), "acc_len": (1, 500),
    "max_depth": (0.0, 1.0),
    "risk_pct": (0.1, 10.0), "max_stop_atr": (0.5, 20.0), "min_stop_atr": (0.1, 20.0),
    "tp1_r": (0.5, 50.0), "tp1_pct": (10.0, 90.0),
}
INT_KEYS = (
    "dir_mode", "cooldown", "brk_fast", "brk_slow", "atr_len", "rank_len",
    "er_len", "htf_in", "build_len", "rel_len", "corr_len",
    "def_fast", "def_slow", "acc_len",
)
BOOL_KEYS = ("use_pull", "use_ten", "auto_htf", "use_corr", "use_vol")

# 29 个会改变 prepare 输出的参数（32 业务参数 - 3 个纯 broker 参数）：
# 排除 cooldown（只影响 coolOk）、risk_pct（只影响定仓 q）、tp1_pct（只影响部分平比例）。
# dir_mode 必须在列：G1/G2 只差它，漏列会让 Long Only 复用 Long&Short 的帧静默做空。
PREPARE_KEYS = (
    "dir_mode", "use_pull", "brk_fast", "brk_slow", "atr_len", "rank_len",
    "rank_cut", "er_len", "er_min", "brk_buf", "use_ten", "htf_in",
    "auto_htf", "build_len", "rel_len", "max_ten", "ten_th", "use_corr",
    "corr_len", "corr_min", "use_vol", "vol_min", "def_fast", "def_slow",
    "acc_len", "max_depth", "max_stop_atr", "min_stop_atr", "tp1_r",
)

# prepare 完成标志：这三列齐即视为已富化（幂等短路，照抄 supertrend_current:213-214）。
_PREPARED = ("rgd_state", "rgd_tp1_long", "rgd_tp1_short")
_DIAGNOSTIC = (
    "rgd_ready", "rgd_atr", "rgd_er", "rgd_vol_rank", "rgd_high_vol",
    "rgd_tension", "rgd_rg", "rgd_corr", "rgd_acceptance", "rgd_bull_q",
    "rgd_bear_q", "rgd_ent_up", "rgd_ent_lo", "rgd_mid",
    "rgd_htf_hh", "rgd_htf_ll", "rgd_htf_ok", "rgd_prev_state",
)


def validate(params: dict) -> None:
    if params["min_stop_atr"] > params["max_stop_atr"]:
        raise ValueError("min_stop_atr 不能大于 max_stop_atr（stp 语义会退化）")
    if params["brk_fast"] >= params["brk_slow"]:
        raise ValueError("brk_fast 必须小于 brk_slow（快/慢通道失去意义）")
    if params["def_fast"] >= params["def_slow"]:
        raise ValueError("def_fast 必须小于 def_slow")
    if params["dir_mode"] not in (0, 1, 2) or params["htf_in"] not in (0, 1, 2):
        raise ValueError("dir_mode / htf_in 枚举越界")


def lookback(params: dict) -> int:
    return max(600, 4 * max(
        int(params["rank_len"]), int(params["brk_slow"]), int(params["def_slow"]),
        int(params["corr_len"]), int(params["atr_len"]), int(params["er_len"]), 20,
    ))


def prepare(frame: pd.DataFrame, params: dict, ctx: PrepareContext) -> pd.DataFrame:
    """纯因果富化；HTF 列由调用方预注入（契约 §6.4）。"""
    if set(_PREPARED).issubset(frame.columns):
        return frame
    missing_htf = not {"rgd_htf_hh", "rgd_htf_ll"}.issubset(frame.columns)
    data = build_features(frame, params)
    if missing_htf:
        # 降级可观测（AC-10）：回测脚本必须先 attach 再 prepare，此分支仅为 live 单根兜底。
        data.attrs["rgd_htf_degraded"] = True
    return data


def signals(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """统一信号帧：11 列 neutral（direction≡0）+ broker/诊断列（契约 §6.5）。"""
    del params
    result = pd.DataFrame(index=data.index)
    result["signed"] = 0.0
    result["score"] = 0.5
    result["prob_up"] = 0.5
    result["direction"] = 0
    result["block_reason"] = "stop_order_requires_dedicated_broker"
    for key in ("position_pct", "leverage", "sl_pct", "tp_pct", "risk_pct", "net_reward_risk"):
        result[key] = 0.0
    for key in (*DECISION_COLUMNS, *_DIAGNOSTIC):
        if key not in data.columns:
            raise ValueError(f"{CODE} 的信号帧缺少列 {key}；请先调用 prepare")
        result[key] = data[key]
    return result


def describe(row: pd.Series) -> dict:
    """末根特征快照，纯观测，不参与任何决策。"""
    def num(key: str) -> float | None:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            return None
        return round(value, 6) if math.isfinite(value) else None

    htf_ok = row.get("rgd_htf_ok", False)
    return {
        "tension": num("rgd_tension"),
        "regime": num("rgd_rg"),
        "efficiency": num("rgd_er"),
        "vol_rank": num("rgd_vol_rank"),
        "correlation": num("rgd_corr"),
        "state": num("rgd_state"),
        "defense_long": num("rgd_def_long"),
        "defense_short": num("rgd_def_short"),
        "htf_ok": bool(htf_ok) if not pd.isna(htf_ok) else False,
    }


SPEC = register(StrategySpec(
    code=CODE,
    name=NAME,
    description=(
        "Regime-Gated Donchian v2：自适应 Donchian 突破 + HTF 流动性张力 regime + "
        "价时相关 + 量能接受度 + ER 门控，R 倍数管理与风险定仓。"
        "15m 主 / 4h HTF，需专用 broker（TP1 限价部分平 + 保本 + 动态 trailing）。"
    ),
    defaults=DEFAULTS,
    ranges=RANGES,
    lookback=lookback,
    prepare=prepare,
    signals=signals,
    describe=describe,
    validate=validate,
    bool_keys=BOOL_KEYS,
    int_keys=INT_KEYS,
    grid={},
    prepare_keys=PREPARE_KEYS,
    default_enabled=False,
    tags=("tradingview", "donchian", "mtf", "dedicated_broker"),
))
