"""Causal, shared signal and risk calculations for paper trading and research."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import DEFAULT_STATE

RULE_DEFAULTS = {
    "regime_filter": True,
    "cost_filter": True,
    "require_model": False,
    "risk_per_trade_pct": 0.005,
    "cooldown_bars": 10,
    "min_atr_cost_ratio": 0.5,
    "min_net_reward_risk": 1.0,
    "max_atr_pct": 0.03,
    "min_model_edge": 0.05,
    "trailing_enabled": True,
    "trail_activation_atr": 1.5,
    "max_signal_age_bars": 2,
}
STRATEGY_PARAMS = {
    "trend_follow": {"adx_filter": 22.0},
    "mean_reversion": {"rsi_low": 30.0, "rsi_high": 70.0, "max_range_adx": 25.0},
    "ml_ensemble": {"ml_weight": 0.6, "adx_filter": 20.0},
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


def resolve_params(code: str, state: dict, overrides: dict | None = None) -> dict:
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
        "min_model_edge": (0, 0.49),
        "adx_filter": (0, 100),
        "max_range_adx": (0, 100),
        "rsi_low": (1, 49),
        "rsi_high": (51, 99),
        "ml_weight": (0, 1),
    }
    for key, value in params.items():
        if key in ("regime_filter", "cost_filter", "require_model", "trailing_enabled"):
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


def evaluate_signals(
    data: pd.DataFrame, code: str, params: dict, prob_up: pd.Series | None = None
) -> pd.DataFrame:
    """Only reads each row; feature construction must also be causal."""

    def column(name):
        return pd.to_numeric(
            data.get(name, pd.Series(np.nan, index=data.index)), errors="coerce"
        )

    st, ema, macd = (
        column("supertrend_dir"),
        np.sign(column("ema_ratio_9_21")),
        np.sign(column("macd_hist_norm")),
    )
    rsi, adx, atr = column("rsi_14"), column("adx_14"), column("atr_pct")
    rules = 0.5 * st + 0.3 * ema + 0.2 * macd
    if code == "mean_reversion":
        low, high = params["rsi_low"], params["rsi_high"]
        signed = pd.Series(
            np.where(
                rsi < low,
                (low - rsi) / low * 0.9,
                np.where(rsi > high, -(rsi - high) / (100 - high) * 0.9, 0.0),
            ),
            index=data.index,
        )
        probability = 0.5 + signed / 2
    elif code == "trend_follow":
        signed = rules * np.where(adx >= params["adx_filter"], 1.0, 0.5)
        probability = 0.5 + signed / 2
    elif code == "ml_ensemble":
        ml = (
            (prob_up.reindex(data.index) - 0.5) * 2
            if prob_up is not None
            else 0.5 * st + 0.5 * macd
        )
        probability = 0.5 + ml / 2
        signed = params["ml_weight"] * ml + (1 - params["ml_weight"]) * rules
    else:
        raise ValueError(f"未知策略: {code}")
    signed = signed.clip(-1, 1)
    score = 0.5 + 0.5 * signed.abs()
    side = np.sign(signed)
    reason = pd.Series("", index=data.index, dtype=object)

    def block(mask, label):
        reason.loc[mask.fillna(True) & reason.eq("")] = label

    required = [
        "close",
        "atr_pct",
        "rsi_14",
        "adx_14",
        "supertrend_dir",
        "ema_ratio_9_21",
        "ema_ratio_21_50",
        "macd_hist_norm",
    ]
    if code == "mean_reversion":
        required.append("bb_pos")
    valid = pd.concat([column(key) for key in required], axis=1)
    block(
        ~pd.Series(np.isfinite(valid).all(axis=1), index=data.index)
        | column("close").le(0)
        | atr.le(0),
        "invalid_indicators",
    )
    block(score.lt(params["min_score"]) | side.eq(0), "below_threshold")
    block(atr.gt(params["max_atr_pct"]), "excess_volatility")
    if code == "ml_ensemble":
        if prob_up is None and params["require_model"]:
            block(pd.Series(True, index=data.index), "model_required")
        if prob_up is not None:
            block(
                ~np.isfinite(probability) | probability.lt(0) | probability.gt(1),
                "invalid_probability",
            )
    if params["regime_filter"]:
        if code == "mean_reversion":
            block(adx.gt(params["max_range_adx"]), "trending_market")
            block(
                ~(
                    ((side > 0) & (column("bb_pos") <= 0.15))
                    | ((side < 0) & (column("bb_pos") >= 0.85))
                ),
                "band_confirmation",
            )
        else:
            block(adx.lt(params["adx_filter"]), "weak_trend")
            block(
                (st != side)
                | (ema != side)
                | (np.sign(column("ema_ratio_21_50")) != side),
                "trend_conflict",
            )
            if code == "ml_ensemble" and prob_up is not None:
                block(
                    ((probability - 0.5) * side).lt(params["min_model_edge"]),
                    "model_edge",
                )

    cost = 2 * (params["fee_rate"] + params["slippage"])
    sl_pct = (atr * params["sl_atr_mult"]).clip(lower=params["min_sl_pct"])
    tp_pct = (atr * params["tp_atr_mult"]).clip(lower=params["min_tp_pct"])
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
