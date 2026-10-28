"""吸收反转（Absorption Reversal）——本地适配自 TradingView 脚本 © blitz_locked。

信号语义：放量单边主动成交（|delta| 位于滚动百分位之上）却推不出价格、在摆动极值
留下拒绝影线 → 进攻方被困。**买方被吸收做空、卖方被吸收做多。**

参数分三类，全部可在「策略 × 币种池」绑定上覆盖：
- 识别：delta_window / delta_pct / body_mult / extreme_lookback / wick_mult /
  dominance_pct / abs_gap_bars / legacy_gap_logic
- 确认：conf_mode / conf_window_bars
- 过滤与风险：direction_mode / trend_mode / trend_len / session_* /
  sl_buffer_atr / target_r / be_trigger_r / min_sl_atr / max_sl_atr /
  max_hold_bars / cooldown_bars / 仓位与杠杆
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ...features.absorption import ABSORPTION_COLUMNS, add_absorption_context, load_lower_tf_delta
from ...features.funding import FUNDING_FILTER_Z
from ...utils import round_price
from ..registry import PrepareContext, StrategySpec, register
from ..signals import assemble_signal_frame, block, column

CODE = "absorption_reversal"

DEFAULTS = {
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
}

RANGES = {
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
    "target_r": (1, 50),
    "be_trigger_r": (0, 20),
    "min_sl_atr": (0, 10),
    "max_sl_atr": (0.1, 20),
}

# 显式声明整数键：除了 `*_bars` 后缀的自动判断，其余整数字段必须写在这里
INT_KEYS = ("conf_mode", "direction_mode", "trend_mode", "extreme_lookback", "conf_window_bars")


def lookback(params: dict) -> int:
    """|delta| 百分位要 delta_window 根窗口，EMA(trend_len) 要 3 倍跨度预热；
    取两者的较大者并留 1200 根底线，避免交易日不足时百分位/EMA 在末根仍失真。"""
    return max(1200, int(params.get("delta_window", 500)) + 3 * int(params.get("trend_len", 200)))


def prepare(frame: pd.DataFrame, params: dict, ctx: PrepareContext) -> pd.DataFrame:
    """补算 delta / 滚动百分位 / c1..c5 / 确认状态机（纯因果）。

    优先用脚本的 lower-tf intrabar delta；拿不到更低周期数据时自动降级为
    `add_absorption_context` 内部的 estDelta 估计（与脚本 `na(ltfDelta) ? estDelta` 一致）。
    """
    if set(ABSORPTION_COLUMNS).issubset(frame.columns):
        return frame
    lower = None
    if ctx.service is not None and ctx.symbol and len(frame.index) >= 2:
        lower = load_lower_tf_delta(ctx.service, ctx.symbol, frame.index)
    return add_absorption_context(frame, params, lower)


def signals(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """按脚本 FILTERS 段顺序逐条门控，最后交给共用拼装层补仓位与杠杆。"""
    atr = column(data, "atr_pct")
    funding_z = column(data, "funding_zscore")
    # 双向：买方被吸收（delta>0 却被压在极值）→ 做空；卖方被吸收 → 做多。
    # 评分 = 两条「超预期」幅度的均值：|delta| 超出滚动百分位的幅度、单边支配度超出门槛的幅度。
    # 脚本本身不区分信号强弱（一律入场），这里评分只用于 min_score/heavy_score 分层。
    raw_signal = column(data, "abs_signal").fillna(0.0)
    percentile = column(data, "abs_percentile").replace(0, np.nan)
    overflow = (column(data, "abs_abs_delta") / percentile - 1.0).clip(lower=0.0, upper=0.5) / 0.5
    overflow = overflow.fillna(0.0)
    floor_dom = float(params["dominance_pct"])
    dominance = column(data, "abs_dominance").fillna(0.0)
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

    valid = pd.concat([column(data, key) for key in ("close", "atr_pct")], axis=1)
    reason = block(
        reason,
        ~pd.Series(np.isfinite(valid).all(axis=1), index=data.index)
        | column(data, "close").le(0)
        | atr.le(0),
        "invalid_indicators",
    )
    # 门控顺序：有无信号 → 止损宽度 → 止损方向 → 方向开关 → 趋势 → 交易时段。
    # 这些过滤全部对应脚本 FILTERS 段的 riskOK / slSide / dirOK / trendOK / inSess。
    atr_abs = column(data, "abs_atr").replace(0, np.nan)
    risk_dist = column(data, "abs_risk")
    has_signal = column(data, "abs_signal").ne(0) & np.isfinite(risk_dist) & atr_abs.gt(0)
    reason = block(reason, ~has_signal, "no_absorption_signal")
    risk_atr = risk_dist / atr_abs
    reason = block(reason, has_signal & risk_atr.gt(params["max_sl_atr"]), "stop_too_wide")
    reason = block(reason, has_signal & risk_atr.lt(params["min_sl_atr"]), "stop_too_tight")
    stop_price = column(data, "abs_sl")
    price = column(data, "close")
    reason = block(reason, has_signal & raw_signal.gt(0) & stop_price.ge(price), "stop_wrong_side")
    reason = block(reason, has_signal & raw_signal.lt(0) & stop_price.le(price), "stop_wrong_side")

    direction_mode = int(params["direction_mode"])
    if direction_mode == 1:
        reason = block(reason, raw_signal.lt(0), "short_disabled")
    elif direction_mode == 2:
        reason = block(reason, raw_signal.gt(0), "long_disabled")

    trend_mode = int(params["trend_mode"])
    if trend_mode in (1, 2):
        # 严格大小比较：close 恰好等于 EMA 时两个模式都判不通过（与脚本的 < / > 一致）
        trend_ema = column(data, "abs_trend_ema")
        below_trend = price.lt(trend_ema)
        above_trend = price.gt(trend_ema)
        long_side = raw_signal.gt(0)
        with_trend = (long_side & above_trend) | (~long_side & below_trend)
        against_trend = (long_side & below_trend) | (~long_side & above_trend)
        trend_ok = with_trend if trend_mode == 1 else against_trend
        reason = block(reason, has_signal & ~trend_ok, "trend_filter")

    start_hour = float(params["session_start_hour"])
    end_hour = float(params["session_end_hour"])
    if start_hour != end_hour:
        hours = pd.Series(data.index.hour + data.index.minute / 60.0, index=data.index)
        in_window = (
            hours.ge(start_hour) & hours.lt(end_hour)
            if start_hour < end_hour
            else hours.ge(start_hour) | hours.lt(end_hour)
        )
        reason = block(reason, has_signal & ~in_window, "session_filter")

    reason = block(reason, score.lt(params["min_score"]) | side.eq(0), "below_threshold")
    reason = block(reason, atr.gt(params["max_atr_pct"]), "excess_volatility")
    if params["funding_filter"]:
        # 资金费极端拥挤时禁止顺费开仓：正极端（多头付费）禁多、负极端（空头付费）禁空；
        # 无资金费数据（z 为 NaN 或 available=0）时不拦截
        reason = block(reason, funding_z.ge(FUNDING_FILTER_Z) & side.gt(0), "funding_filter_long")
        reason = block(reason, funding_z.le(-FUNDING_FILTER_Z) & side.lt(0), "funding_filter_short")

    # 止损/目标都从结构距离导出：risk = |close − 吸收极值 ∓ buffer×ATR|，目标 = target_r × risk。
    # 注：脚本用实际成交价（次根开盘±滑点）计算 risk 与目标，本地撮合以信号 bar 收盘为参考，
    # 二者差异只有滑点/跳空一个量级，方向语义（保本价、R 倍数）完全一致。
    sl_pct = (risk_dist / price.replace(0, np.nan)).clip(
        lower=params["min_sl_pct"]
    ).fillna(params["min_sl_pct"])
    tp_pct = (sl_pct * params["target_r"]).clip(lower=params["min_tp_pct"])

    return assemble_signal_frame(
        data,
        params,
        signed=signed,
        score=score,
        prob_up=probability,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        reason=reason,
    )


def describe(row: pd.Series) -> dict:
    """末根观测快照：delta/支配度/百分位阈值、吸收极值与待穿越关卡（纯展示）。"""
    def value(name):
        try:
            number = float(row[name])
        except (KeyError, TypeError, ValueError):
            return 0.0
        return number if math.isfinite(number) else 0.0

    return {
        "abs_delta": round(value("abs_delta"), 4),
        "abs_dominance": round(value("abs_dominance"), 4),
        "abs_percentile": round(value("abs_percentile"), 4),
        "abs_wick_ratio": round(value("abs_wick_ratio"), 6),
        "abs_atr": round_price(value("abs_atr")),
        "structure_extreme": round_price(value("abs_extreme")),
        "confirm_level": round_price(value("abs_level")),
        "signal_dir": value("abs_signal"),
        "stop_price": round_price(value("abs_sl")),
    }


SPEC = register(
    StrategySpec(
        code=CODE,
        name="吸收反转",
        description=(
            "放量单边主动成交（|delta| 位于滚动百分位之上）却推不出价格、在摆动极值留下拒绝影线 → "
            "进攻方被困：买方被吸收做空、卖方被吸收做多。默认等待后续 3 根内收盘穿越吸收 bar 实体确认，"
            "期间打穿吸收极值或超时作废。结构止损（极值 ∓ 0.25 ATR）、2R 固定目标、1R 保本、"
            "30 根时间止损，名义仓位 30%、1 倍杠杆，多空双向"
        ),
        defaults=DEFAULTS,
        ranges=RANGES,
        lookback=lookback,
        prepare=prepare,
        signals=signals,
        describe=describe,
        bool_keys=("legacy_gap_logic", "funding_filter", "cost_filter", "regime_filter", "trailing_enabled"),
        int_keys=INT_KEYS,
        # 默认**不启用**：docs/absorption-reversal-strategy.md 三窗 × 主流币 15m 检验
        # 池化 88/106 笔的每笔期望 95% CI 跨 0（不可判定），放宽灵敏度后转负。
        # 移植已完整落地、可在「策略 × 池」绑定上手动开启，但在补齐 1m 历史、
        # 做出样本足够的判据之前，不参与自动巡检的资金竞争。
        default_enabled=False,
        tags=("tradingview", "orderflow"),
    )
)
