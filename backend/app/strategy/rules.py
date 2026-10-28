"""Causal signal and risk rules for the single live strategy (absorption reversal)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import DEFAULT_STATE
from ..features.absorption import ABSORPTION_COLUMNS, add_absorption_context
from ..features.funding import FUNDING_FILTER_Z

RULE_DEFAULTS = {
    "regime_filter": True,
    "cost_filter": True,
    "funding_filter": False,
    "risk_per_trade_pct": 0.005,
    "cooldown_bars": 10,
    "min_atr_cost_ratio": 0.5,
    "min_net_reward_risk": 1.0,
    "max_atr_pct": 0.03,
    "trailing_enabled": True,
    "trail_activation_atr": 1.5,
    "max_signal_age_bars": 2,
}
# Absorption Reversal（本地适配自 TradingView 脚本 Absorption Strat © blitz_locked）。
# 吸收 bar = 放量单边主动成交 + 无价格进展 + 摆动极值 + 拒绝影线 + 单边支配；
# 买方被吸收→做空，卖方被吸收→做多。默认「收盘穿越实体」确认（最多 3 根），
# 期间打穿吸收极值或超时即作废。
# 出场：结构止损（吸收极值 ∓ 0.25 ATR）+ 2R 固定目标 + 1R 保本 + 30 根时间止损，
# 无追踪止损、无平仓后冷却（与脚本一致）。
# 仓位：脚本 default_qty_type = percent_of_equity 30 → 名义 = 权益 30%、1 倍杠杆；
# risk_per_trade_pct 只作为极端宽止损下的兜底（max_sl_atr 已限制 3 ATR 以内）。
STRATEGY_PARAMS = {
    "absorption_reversal": {
        # 吸收识别（Medium 档 = 脚本默认）
        "delta_window": 500,
        "delta_pct": 95.0,
        "body_mult": 0.25,
        "extreme_lookback": 20,
        "wick_mult": 0.20,
        "dominance_pct": 20.0,
        "abs_gap_bars": 10,
        # False = 修正后的「必须先满足吸收条件」节奏；True = 1:1 复现脚本的 and/or 优先级 bug
        "legacy_gap_logic": False,
        # 入场确认
        "conf_mode": 1,          # 0 = 吸收 bar 收盘直接入场；1 = 收盘穿越实体后入场
        "conf_window_bars": 3,
        "direction_mode": 0,     # 0 = 多空都做；1 = 只做多；2 = 只做空
        "trend_mode": 0,         # 0 = 关；1 = 顺 EMA 趋势；2 = 逆 EMA 趋势
        "trend_len": 200,
        "session_start_hour": 0.0,   # UTC 小时（含小数）；0/24 = 全天，等价关闭
        "session_end_hour": 24.0,
        # 出场与风险
        "sl_buffer_atr": 0.25,
        "target_r": 2.0,
        "be_trigger_r": 1.0,     # 浮盈达到该 R 后止损移到保本价（0 = 禁用）
        "min_sl_atr": 0.30,
        "max_sl_atr": 3.00,
        "max_hold_bars": 30,     # 脚本的 Time stop(bars)，按 bar 计（与脚本一致，不换算小时）
        # 脚本在持仓期间允许反手；本地撮合需要 min_hold_bars=0 才允许反向信号先平后开
        "min_hold_bars": 0,
        "max_leverage": 1,
        "base_position_pct": 0.30,
        "max_position_pct": 0.30,
        "risk_per_trade_pct": 0.02,
        "cooldown_bars": 0,
        "trailing_enabled": False,
    },
}
SHARED_KEYS = (
    "min_score",
    "heavy_score",
    "base_position_pct",
    "max_position_pct",
    "max_leverage",
    "sl_atr_mult",
    "tp_atr_mult",
    "trailing_atr_mult",
    "min_sl_pct",
    "min_tp_pct",
    "min_hold_bars",
    "max_hold_bars",
    "fee_rate",
    "slippage",
)


def resolve_params(
    code: str, state: dict, overrides: dict | None = None, interval: str | None = None
) -> dict:
    """合并全局设置、策略参数与调用方覆盖值。

    interval 保留在签名里以兼容既有调用方，但本策略的结构止损按 bar 计（max_hold_bars），
    不再做「小时→bar」换算。
    """
    if code not in STRATEGY_PARAMS:
        raise ValueError(f"未知策略: {code}")
    params = {key: state.get(key, DEFAULT_STATE[key]) for key in SHARED_KEYS}
    params.update({key: state.get(key, value) for key, value in RULE_DEFAULTS.items()})
    params.update(STRATEGY_PARAMS[code])
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError("策略参数必须是 JSON 对象")
        if {"fee_rate", "slippage"} & set(overrides):
            raise ValueError(
                "手续费和滑点请在全局设置或回测请求中配置，策略不能覆盖撮合成本"
            )
        unknown = set(overrides) - set(params)
        if unknown:
            raise ValueError(f"未知策略参数: {', '.join(sorted(unknown))}")
        params.update(overrides)
    ranges = {
        "min_score": (0.5, 1),
        "heavy_score": (0.5, 1),
        "base_position_pct": (0, 1),
        "max_position_pct": (0, 1),
        "risk_per_trade_pct": (0.0001, 0.05),
        "max_leverage": (1, 125),
        "sl_atr_mult": (0.1, 10),
        "tp_atr_mult": (0.2, 20),
        "trailing_atr_mult": (0.1, 10),
        "trail_activation_atr": (0.1, 20),
        "min_sl_pct": (0.0001, 0.2),
        "min_tp_pct": (0.0002, 0.5),
        "fee_rate": (0, 0.01),
        "slippage": (0, 0.01),
        "min_hold_bars": (0, 10000),
        "max_hold_bars": (0, 100000),
        "cooldown_bars": (0, 10000),
        "max_signal_age_bars": (1, 10),
        "min_atr_cost_ratio": (0, 10),
        "min_net_reward_risk": (0, 10),
        "max_atr_pct": (0.0001, 0.2),
        "target_r": (1, 50),
        # --- 吸收反转（脚本 1:1 参数） ---
        "delta_window": (10, 2000),
        "delta_pct": (50, 100),
        "body_mult": (0, 2),
        "extreme_lookback": (2, 200),
        "wick_mult": (0, 1),
        "dominance_pct": (0, 100),
        "abs_gap_bars": (0, 500),
        "conf_mode": (0, 1),
        "conf_window_bars": (1, 20),
        "direction_mode": (0, 2),
        "trend_mode": (0, 2),
        "trend_len": (5, 1000),
        "session_start_hour": (0, 24),
        "session_end_hour": (0, 24),
        "sl_buffer_atr": (0, 5),
        "be_trigger_r": (0, 20),
        "min_sl_atr": (0, 10),
        "max_sl_atr": (0.1, 20),
    }
    for key, value in params.items():
        if key in (
            "regime_filter", "cost_filter", "funding_filter", "trailing_enabled",
            "legacy_gap_logic",
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{key} 必须是布尔值")
        else:
            low, high = ranges[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"{key} 必须在 [{low}, {high}] 范围内")
            if key.endswith("_bars") or key == "max_leverage":
                if int(value) != value:
                    raise ValueError(f"{key} 必须为整数")
    if params["base_position_pct"] > params["max_position_pct"]:
        raise ValueError("基础仓位不能超过单仓上限")
    if 0 < params["max_hold_bars"] < params["min_hold_bars"]:
        raise ValueError("最大持仓K线数不能小于最小持仓K线数")
    return params


def evaluate_signals(data: pd.DataFrame, code: str, params: dict) -> pd.DataFrame:
    """Only reads each row; feature construction must also be causal.

    本系统只剩吸收反转一套规则：调用方把已收盘的整帧（至少含 OHLCV + atr_pct）传入，
    这里补算 delta/百分位/吸收条件/确认状态机（纯因果），再按脚本的 FILTERS 段做门控。
    """
    if code not in STRATEGY_PARAMS:
        raise ValueError(f"未知策略: {code}")
    if not set(ABSORPTION_COLUMNS).issubset(data.columns):
        data = add_absorption_context(data, params)

    def column(name):
        return pd.to_numeric(
            data.get(name, pd.Series(np.nan, index=data.index)), errors="coerce"
        )

    atr = column("atr_pct")
    funding_z = column("funding_zscore")
    # 双向：买方被吸收（delta>0 却被压在极值）→ 做空；卖方被吸收 → 做多。
    # 评分 = 两条「超预期」幅度的均值：|delta| 超出滚动百分位的幅度、单边支配度超出门槛的幅度。
    # 脚本本身不区分信号强弱（一律入场），这里让评分只用于 min_score/heavy_score 分层，
    # 默认 min_score=0.75 时全部通过，行为与原脚本一致。
    raw_signal = column("abs_signal").fillna(0.0)
    percentile = column("abs_percentile").replace(0, np.nan)
    overflow = (column("abs_abs_delta") / percentile - 1.0).clip(lower=0.0, upper=0.5) / 0.5
    overflow = overflow.fillna(0.0)
    floor_dom = float(params["dominance_pct"])
    dominance = column("abs_dominance").fillna(0.0)
    dom_strength = (
        ((dominance - floor_dom) / max(100.0 - floor_dom, 1e-9)).clip(lower=0.0, upper=1.0)
        if floor_dom > 0
        else (dominance / 100.0).clip(lower=0.0, upper=1.0)
    )
    strength = ((overflow + dom_strength) / 2.0).clip(lower=0.0, upper=1.0)
    signed = (0.5 + 0.5 * strength).where(raw_signal.ne(0), 0.0) * np.sign(raw_signal)
    probability = 0.5 + signed / 2
    signed = signed.clip(-1, 1)
    score = 0.5 + 0.5 * signed.abs()
    side = np.sign(signed)
    reason = pd.Series("", index=data.index, dtype=object)

    def block(mask, label):
        reason.loc[mask.fillna(True) & reason.eq("")] = label

    required = ["close", "atr_pct"]
    valid = pd.concat([column(key) for key in required], axis=1)
    block(
        ~pd.Series(np.isfinite(valid).all(axis=1), index=data.index)
        | column("close").le(0)
        | atr.le(0),
        "invalid_indicators",
    )
    # 门控顺序：有无信号 → 止损宽度 → 止损方向 → 方向开关 → 趋势 → 交易时段。
    # 这些过滤全部写在脚本的 FILTERS 段（riskOK / slSide / dirOK / trendOK / inSess）。
    atr_abs = column("abs_atr").replace(0, np.nan)
    risk_dist = column("abs_risk")
    has_signal = column("abs_signal").ne(0) & np.isfinite(risk_dist) & atr_abs.gt(0)
    block(~has_signal, "no_absorption_signal")
    risk_atr = risk_dist / atr_abs
    block(has_signal & risk_atr.gt(params["max_sl_atr"]), "stop_too_wide")
    block(has_signal & risk_atr.lt(params["min_sl_atr"]), "stop_too_tight")
    stop_price = column("abs_sl")
    price = column("close")
    block(has_signal & column("abs_signal").gt(0) & stop_price.ge(price), "stop_wrong_side")
    block(has_signal & column("abs_signal").lt(0) & stop_price.le(price), "stop_wrong_side")
    direction_mode = int(params["direction_mode"])
    if direction_mode == 1:
        block(column("abs_signal").lt(0), "short_disabled")
    elif direction_mode == 2:
        block(column("abs_signal").gt(0), "long_disabled")
    trend_mode = int(params["trend_mode"])
    if trend_mode in (1, 2):
        # 严格大小比较：close 恰好等于 EMA 时两个模式都判不通过（与脚本的 < / > 一致）
        trend_ema = column("abs_trend_ema")
        below_trend = price.lt(trend_ema)
        above_trend = price.gt(trend_ema)
        long_side = column("abs_signal").gt(0)
        with_trend = (long_side & above_trend) | (~long_side & below_trend)
        against_trend = (long_side & below_trend) | (~long_side & above_trend)
        trend_ok = with_trend if trend_mode == 1 else against_trend
        block(has_signal & ~trend_ok, "trend_filter")
    start_hour = float(params["session_start_hour"])
    end_hour = float(params["session_end_hour"])
    if start_hour != end_hour:
        hours = pd.Series(
            data.index.hour + data.index.minute / 60.0, index=data.index
        )
        in_window = (
            hours.ge(start_hour) & hours.lt(end_hour)
            if start_hour < end_hour
            else hours.ge(start_hour) | hours.lt(end_hour)
        )
        block(has_signal & ~in_window, "session_filter")
    block(score.lt(params["min_score"]) | side.eq(0), "below_threshold")
    block(atr.gt(params["max_atr_pct"]), "excess_volatility")
    if params["funding_filter"]:
        # 资金费极端拥挤时禁止顺费开仓：正极端（多头付费）禁多、负极端（空头付费）禁空；
        # 无资金费数据（z 为 NaN 或 available=0）时不拦截
        block(funding_z.ge(FUNDING_FILTER_Z) & side.gt(0), "funding_filter_long")
        block(funding_z.le(-FUNDING_FILTER_Z) & side.lt(0), "funding_filter_short")
    if params["regime_filter"]:
        # 本策略不套用 SuperTrend/EMA/ADX 的趋势一致性规则——吸收的本质就是逆下一根动能
        pass

    cost = 2 * (params["fee_rate"] + params["slippage"])
    # 止损/目标都从结构距离导出：risk = |close − 吸收极值 ∓ buffer×ATR|，目标 = target_r × risk。
    # 注：脚本用实际成交价（次根开盘±滑点）计算 risk 与目标，本地撮合以信号 bar 收盘为参考，
    # 二者差异只有滑点/跳空一个量级，方向语义（保本价、R 倍数）完全一致。
    sl_pct = (column("abs_risk") / price.replace(0, np.nan)).clip(
        lower=params["min_sl_pct"]
    ).fillna(params["min_sl_pct"])
    tp_pct = (sl_pct * params["target_r"]).clip(lower=params["min_tp_pct"])
    net_rr = (tp_pct - cost) / (sl_pct + cost)
    if params["cost_filter"]:
        block(
            atr.lt(cost * params["min_atr_cost_ratio"])
            | net_rr.lt(params["min_net_reward_risk"]),
            "cost_filter",
        )
    confidence = ((score - 0.5) / 0.5).clip(0, 1)
    leverage = (
        (0.015 / atr.clip(lower=0.0001) * (0.6 + 0.4 * confidence))
        .round()
        .clip(1, params["max_leverage"])
        .fillna(1)
        .astype(int)
    )
    span = max(params["heavy_score"] - params["min_score"], 1e-6)
    fraction = ((score - params["min_score"]) / span).clip(0, 1)
    fraction = params["base_position_pct"] + fraction * (
        params["max_position_pct"] - params["base_position_pct"]
    )
    fraction = fraction.where(score < params["heavy_score"], params["max_position_pct"])
    fraction = np.minimum(
        fraction, params["risk_per_trade_pct"] / (leverage * (sl_pct + cost))
    )
    fraction = fraction.where(reason.eq(""), 0.0).fillna(0.0)
    direction = side.where(reason.eq("") & fraction.gt(0), 0).fillna(0).astype(int)
    return pd.DataFrame(
        {
            "signed": signed.fillna(0),
            "score": score.fillna(0.5),
            "prob_up": probability.fillna(0.5),
            "direction": direction,
            "block_reason": reason,
            "position_pct": fraction,
            "leverage": leverage.where(direction.ne(0), 0),
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "risk_pct": fraction * leverage * (sl_pct + cost),
            "net_reward_risk": net_rr,
        },
        index=data.index,
    )
