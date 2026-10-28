"""Causal, shared signal and risk calculations for paper trading and research."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import DEFAULT_STATE
from ..features.funding import FUNDING_FILTER_Z
from ..features.pullback import PULLBACK_COLUMNS, add_pullback_context
from ..utils import interval_ms

RULE_DEFAULTS = {
    "regime_filter": True,
    "cost_filter": True,
    "require_model": False,
    "funding_filter": False,
    "risk_per_trade_pct": 0.005,
    "cooldown_bars": 10,
    "min_atr_cost_ratio": 0.5,
    "min_net_reward_risk": 1.0,
    "max_atr_pct": 0.03,
    "min_model_edge": 0.05,
    "trailing_enabled": True,
    "trail_activation_atr": 1.5,
    "max_signal_age_bars": 2,
    # 软出场：信号帧带 exit_long 时由共享撮合在下一根开盘平多（仅强回调用）
    "ema_exit": False,
    # 浮盈达标先按比例落袋（0=禁用）：剩余仓位止损移到保本价、取消固定目标，
    # 交给追踪止损与软出场跑完（仅强回调默认开启，2R 减半）
    "partial_tp_r": 0.0,
    "partial_tp_frac": 0.5,
}
STRATEGY_PARAMS = {
    "trend_follow": {"adx_filter": 22.0},
    "mean_reversion": {"rsi_low": 30.0, "rsi_high": 70.0, "max_range_adx": 25.0},
    "ml_ensemble": {"ml_weight": 0.6, "adx_filter": 20.0},
    "funding_extreme": {"z_threshold": 2.0, "max_adx": 25.0},
    "mtf_resonance": {"adx_filter": 22.0, "htf_adx_filter": 20.0},
    # Martin Luk 强势回调（本地适配）：4h 强势趋势 → 15m 首次回调 EMA9/AVWAP → 转强确认。
    # 只做多、1 倍杠杆；单笔风险 0.5%，名义仓位上限 35%（max_leverage=1 时仓位比例即名义比例）。
    # 出场：EMA9 失守软出场 + 结构硬止损 + 最长 48 小时 + 10R 固定目标（原版规格）。
    # 注：partial_tp_r=0.0 默认禁用分段落袋；A/B 验证（36 币×4 窗口，持仓 2155+）证明
    #   2R 减半出场并未改善期望值（partial vs fixed 差中位数 = 0，两者 CI 全负，
    #   根因是入场过滤器本身无 edge，非出场结构）。partial_tp_r > 0 的功能代码保留
    #   供将来入场优化后重试。
    "strong_pullback": {
        "htf_adx_filter": 22.0,
        "pullback_min_bars": 8,
        "pullback_confirm_bars": 8,
        "max_leverage": 1,
        "max_position_pct": 0.35,
        "risk_per_trade_pct": 0.005,
        "sl_atr_mult": 1.5,
        "target_r": 10.0,
        "partial_tp_r": 0.0,
        "partial_tp_frac": 0.5,
        "stop_buffer_pct": 0.0015,
        "max_hold_hours": 48.0,
        "max_hold_bars": 192,  # 15m 下的 48 小时；其他周期由 resolve_params(interval=) 换算
        "min_hold_bars": 0,
        "ema_exit": True,
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

    interval 非空时把「最长持仓小时数」换算成该周期的 bar 数（强回调的 48 小时硬约束），
    这样同一套参数在 15m / 1h / 4h 上都表示同样的墙钟时长。
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
    if interval is not None and "max_hold_hours" in params:
        hours = float(params["max_hold_hours"])
        params["max_hold_bars"] = max(1, int(round(hours * 3_600_000 / interval_ms(interval))))
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
        "htf_adx_filter": (0, 100),
        "max_range_adx": (0, 100),
        "max_adx": (0, 100),
        "z_threshold": (0.1, 10),
        "rsi_low": (1, 49),
        "rsi_high": (51, 99),
        "ml_weight": (0, 1),
        "target_r": (1, 50),
        "partial_tp_r": (0, 50),
        "partial_tp_frac": (0.05, 0.95),
        "stop_buffer_pct": (0, 0.05),
        "max_hold_hours": (0.5, 240),
        "pullback_min_bars": (0, 200),
        "pullback_confirm_bars": (1, 200),
    }
    for key, value in params.items():
        if key in (
            "regime_filter", "cost_filter", "require_model", "trailing_enabled",
            "funding_filter", "ema_exit",
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


def evaluate_signals(
    data: pd.DataFrame, code: str, params: dict, prob_up: pd.Series | None = None,
    market_context: dict | None = None,
) -> pd.DataFrame:
    """Only reads each row; feature construction must also be causal.

    market_context 预留给多源指标加权消费（迭代 7 起采集+存储+观测）。当前默认 None，
    且即便传入也不改变任何方向/仓位计算——保证 metrics_assist_enabled 关闭时行为与历史一致。
    """
    if code == "strong_pullback" and not set(PULLBACK_COLUMNS).issubset(data.columns):
        # 实盘只传末根已富化帧；回测/训练传整帧，这里补算 4h 上下文与回调状态机（纯因果）
        data = add_pullback_context(data, params)

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
    funding_z = column("funding_zscore")
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
    elif code == "mtf_resonance":
        # 复用 trend_follow 的 1h 方向逻辑：supertrend_dir/ema_ratio/macd 合成 signed，
        # 用 adx >= adx_filter 缩放（与 trend_follow 完全一致），下面再做多周期共振门控。
        signed = rules * np.where(adx >= params["adx_filter"], 1.0, 0.5)
        probability = 0.5 + signed / 2
        # 共振门控：仅当 1h side 与更高周期（4h/1d）同向且更高周期 ADX 都足够强时保留信号；
        # 否则 signed=0（不开仓，不反向——避免制造额外交易）。更高周期列由 add_mtf_trend 提供。
        side_raw = np.sign(signed)
        mtf_dir_4h = column("mtf_dir_4h")
        mtf_dir_1d = column("mtf_dir_1d")
        mtf_adx_4h = column("mtf_adx_4h")
        mtf_adx_1d = column("mtf_adx_1d")
        htf_filter = params["htf_adx_filter"]
        aligned = (
            (side_raw == mtf_dir_4h)
            & (side_raw == mtf_dir_1d)
            & (mtf_adx_4h >= htf_filter)
            & (mtf_adx_1d >= htf_filter)
        ).fillna(False)
        signed = signed.where(aligned, 0.0)
    elif code == "funding_extreme":
        # 资金费极值逆向：z <= -z_threshold（负费率、空头付费、拥挤做空）→ 多；
        # z >= +z_threshold（正费率、多头付费、拥挤做多）→ 空。未达极值不出信号。
        threshold = params["z_threshold"]
        magnitude = (funding_z.abs() / threshold).clip(upper=1.0)
        signed = pd.Series(
            np.where(funding_z <= -threshold, magnitude,
                     np.where(funding_z >= threshold, -magnitude, 0.0)),
            index=data.index,
        ).fillna(0.0)
        probability = 0.5 + signed / 2
    elif code == "strong_pullback":
        # 只做多：4h 强势趋势下的首次回调转强确认。评分 = 4h 趋势强度（ADX 超出阈值的幅度），
        # 让 min_score 真正参与筛选，而不是一律满分。
        htf_strength = (
            (column("pb_adx_4h") - params["htf_adx_filter"]) / 15.0
        ).clip(lower=0.0, upper=1.0).fillna(0.0)
        confirmed = column("pb_confirm").fillna(0).gt(0)
        signed = (0.25 + 0.75 * htf_strength).where(confirmed, 0.0)
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
    if code == "funding_extreme":
        block(column("funding_available").ne(1), "funding_data_missing")
    if code == "strong_pullback":
        # 门控顺序即「先失败先记」：4h 趋势 → 是否出现首次回调 → 是否转强确认 → 结构位置
        trend_ok = column("pb_trend_ok").fillna(0).gt(0)
        bars_since = column("pb_bars_since_touch")
        confirmed = column("pb_confirm").fillna(0).gt(0)
        block(~trend_ok, "htf_trend_weak")
        block(trend_ok & bars_since.isna(), "no_pullback")
        block(trend_ok & bars_since.notna() & ~confirmed, "no_confirmation")
        block(confirmed & column("close").lt(column("avwap")), "below_avwap")
    block(score.lt(params["min_score"]) | side.eq(0), "below_threshold")
    block(atr.gt(params["max_atr_pct"]), "excess_volatility")
    if params["funding_filter"]:
        # 资金费极端拥挤时禁止顺费开仓：正极端（多头付费）禁多、负极端（空头付费）禁空；
        # 无资金费数据（z 为 NaN 或 available=0）时不拦截
        block(funding_z.ge(FUNDING_FILTER_Z) & side.gt(0), "funding_filter_long")
        block(funding_z.le(-FUNDING_FILTER_Z) & side.lt(0), "funding_filter_short")
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
        elif code == "funding_extreme":
            # 均值回归语义：仅在震荡市（低 ADX）执行，强趋势时资金费极值可能延续
            block(adx.gt(params["max_adx"]), "trending_market")
        elif code == "strong_pullback":
            # 自带 4h 强势趋势 + 回调结构过滤（见上方 block），不套用 1h SuperTrend/EMA 规则
            pass
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
    if code == "strong_pullback":
        # 硬止损放回调段低点下方（结构止损），并以 ATR 止损兜底；目标 = target_r 倍风险（10R）
        close = column("close")
        structure = ((close - column("pb_stop_ref")) / close.replace(0, np.nan)).clip(lower=0.0)
        sl_pct = pd.concat(
            [structure + params["stop_buffer_pct"], atr * params["sl_atr_mult"]], axis=1
        ).max(axis=1).fillna(atr * params["sl_atr_mult"]).clip(lower=params["min_sl_pct"])
        tp_pct = (sl_pct * params["target_r"]).clip(lower=params["min_tp_pct"])
    else:
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
    exit_long = (
        column("pb_exit_long").fillna(0).gt(0)
        if code == "strong_pullback"
        else pd.Series(False, index=data.index)
    )
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
            # EMA9 失守软出场标记：仅强回调用，其他策略恒 False（共享撮合据此平多）
            "exit_long": exit_long.astype(int),
        },
        index=data.index,
    )
