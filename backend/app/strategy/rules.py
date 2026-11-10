"""参数合并与校验 + 信号评估的统一入口。

仓库按「策略 × 币种池」组织参数，三层合并、后者覆盖前者：

    全局设置（app_state）  <  策略默认（StrategySpec.defaults）  <  绑定参数（binding.params）

本模块只做合并且校验值域/类型；具体信号逻辑在 `strategy/specs/<code>.py`，
通过注册表分派——所以引擎、巡检、回测、API 都不需要知道某个策略的内部细节。
"""

from __future__ import annotations

import math

import pandas as pd

from ..config import DEFAULT_STATE
from . import specs  # noqa: F401  导入即注册内置策略
from .exits import DEFAULT_EXIT, EXIT_KEY, ensure_allowed, normalize as normalize_exit, overlay_distances
from .registry import PrepareContext, available_codes, get_spec
from .signals import SIGNAL_COLUMNS, finalize_signal_frame

# 通用规则开关与默认值（与具体策略无关）
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
    # EMA 快慢线方向过滤（通用，所有策略共享）：
    # 0 = 关闭；>0 时按「快线 EMA(fast) 与慢线 EMA(slow) 的相对位置」做方向门控。
    # 多头信号要求 fast > slow，空头信号要求 fast < slow；fast 或 slow 任一为 0 即关闭。
    # 这只是入场门控，不改变策略本身的方向判断；默认关闭以保持既有口径不变。
    "ema_trend_fast": 0,
    "ema_trend_slow": 0,
    # 绑定层出场覆盖：{method, support, params}，见 strategy/exits.py
    EXIT_KEY: dict(DEFAULT_EXIT),
}

# 允许被全局设置（state）驱动的键：这些是账户级/撮合级参数，不对单个策略开放
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

SHARED_RANGES: dict[str, tuple[float, float]] = {
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
    "ema_trend_fast": (0, 500),
    "ema_trend_slow": (0, 1000),
}

SHARED_BOOL_KEYS = ("regime_filter", "cost_filter", "funding_filter", "trailing_enabled")


def known_codes() -> tuple[str, ...]:
    """当前仓库真正有实现的策略代码（老库残留配置行不在此列）。"""
    return available_codes()


def strategy_defaults(code: str) -> dict:
    return dict(get_spec(code).defaults)


def resolve_params(
    code: str, state: dict, overrides: dict | None = None, interval: str | None = None
) -> dict:
    """合并全局设置、策略默认与调用方覆盖值（绑定参数 / 回测请求）。

    `interval` 保留在签名里以兼容既有调用方：本策略的结构止损按 bar 计
    （`max_hold_bars`），不做「小时→bar」换算；周期已上移到「策略 × 池」绑定上。
    """
    spec = get_spec(code)
    params = {key: state.get(key, DEFAULT_STATE[key]) for key in SHARED_KEYS}
    params.update({key: state.get(key, value) for key, value in RULE_DEFAULTS.items()})
    params.update(spec.defaults)
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

    ranges = {**SHARED_RANGES, **spec.ranges}
    bool_keys = set(SHARED_BOOL_KEYS) | set(spec.bool_keys)
    int_keys = set(spec.int_keys) | {"max_leverage"}
    for key, value in params.items():
        if key == EXIT_KEY:
            # 结构化配置：由出场注册表校验，不参与数值区间检查
            params[key] = normalize_exit(value)
            continue
        if key in bool_keys:
            if not isinstance(value, bool):
                raise ValueError(f"{key} 必须是布尔值")
            continue
        low, high = ranges[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not low <= value <= high
        ):
            raise ValueError(f"{key} 必须在 [{low}, {high}] 范围内")
        if key in int_keys or key.endswith("_bars"):
            if int(value) != value:
                raise ValueError(f"{key} 必须为整数")

    if params["base_position_pct"] > params["max_position_pct"]:
        raise ValueError("基础仓位不能超过单仓上限")
    if 0 < params["max_hold_bars"] < params["min_hold_bars"]:
        raise ValueError("最大持仓K线数不能小于最小持仓K线数")
    ema_fast = int(params.get("ema_trend_fast", 0) or 0)
    ema_slow = int(params.get("ema_trend_slow", 0) or 0)
    if ema_fast and ema_slow and ema_fast >= ema_slow:
        raise ValueError("EMA 方向过滤要求快线周期严格小于慢线周期")
    if spec.validate is not None:
        spec.validate(params)
    ensure_allowed(code, params[EXIT_KEY], params)
    return params


def prepare_frame(
    data: pd.DataFrame,
    code: str,
    params: dict,
    *,
    symbol: str = "",
    service=None,
) -> pd.DataFrame:
    """对整帧补算策略自有特征；已有全部所需列时直接返回（不重复计算）。"""
    return get_spec(code).prepare(data, params, PrepareContext(symbol=symbol, service=service))


def evaluate_signals(
    data: pd.DataFrame,
    code: str,
    params: dict,
    *,
    symbol: str = "",
    service=None,
) -> pd.DataFrame:
    """按策略分派产出统一信号帧（`SIGNAL_COLUMNS`）。

    `symbol` / `service` 只在策略需要更低周期数据时使用（如 intrabar delta）；
    回测与特征测试不传也能跑——策略内部会降级。
    产出后先叠加绑定层的出场覆盖（见 `apply_exit_overlay`），再过 EMA 方向门控。
    覆盖层用 `finalize_signal_frame` 重建 direction / 仓位，门控必须是最后一道，
    否则在 atr_r / support_r 绑定下可能被重建覆盖（与 optimizer worker 同口径，H10）。
    """
    frame = prepare_frame(data, code, params, symbol=symbol, service=service)
    result = get_spec(code).signals(frame, params)
    missing = set(SIGNAL_COLUMNS) - set(result.columns)
    if missing:
        raise ValueError(f"{code} 输出的信号帧缺少列: {', '.join(sorted(missing))}")
    result = apply_exit_overlay(frame, result, params)
    return apply_ema_trend_filter(frame, result, params)


def apply_ema_trend_filter(
    frame: pd.DataFrame, signals: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """通用 EMA 快慢线方向过滤：只做入场门控，不改策略方向判断。

    当 `ema_trend_fast` 与 `ema_trend_slow` 都 > 0 且 fast < slow 时启用：
    - 多头信号要求 EMA(fast) > EMA(slow)
    - 空头信号要求 EMA(fast) < EMA(slow)
    不满足时把方向清零并打 `ema_trend_filter` 标签，仓位/杠杆同步归零。

    默认关闭（两个周期都为 0），行为与改造前完全一致。
    """
    fast = int(params.get("ema_trend_fast", 0) or 0)
    slow = int(params.get("ema_trend_slow", 0) or 0)
    if not fast or not slow:
        return signals
    series = frame.get("close")
    if series is None:
        return signals
    close = pd.to_numeric(series, errors="coerce")
    if close.empty:
        return signals
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    bullish = ema_fast.gt(ema_slow)
    bearish = ema_fast.lt(ema_slow)
    direction = pd.to_numeric(signals["direction"], errors="coerce").fillna(0).astype(int)
    long_blocked = direction.gt(0) & ~bullish
    short_blocked = direction.lt(0) & ~bearish
    blocked = long_blocked | short_blocked
    if not blocked.any():
        return signals
    result = signals.copy()
    reason = result["block_reason"].astype(object)
    reason = reason.mask(blocked & reason.eq(""), "ema_trend_filter")
    result["block_reason"] = reason
    result["direction"] = direction.where(~blocked, 0)
    result["position_pct"] = result["position_pct"].where(~blocked, 0.0)
    result["leverage"] = result["leverage"].where(~blocked, 0)
    result["risk_pct"] = result["risk_pct"].where(~blocked, 0.0)
    return result


def apply_exit_overlay(frame: pd.DataFrame, signals: pd.DataFrame, params: dict) -> pd.DataFrame:
    """按绑定的出场方法重算止损/止盈，并用**新距离**重算仓位与成本过滤。

    `method=strategy`（默认）时原样返回，行为与改造前完全一致。
    支撑位来源需要整帧历史（枢轴确认、滚动池），因此调用方给单行切片时 `support_r`
    会全部回退 ATR —— 实时巡检因此改为整帧评估后取末行（见 `engine.evaluate`）。
    """
    config = params.get(EXIT_KEY) or DEFAULT_EXIT
    if (config.get("method") or "strategy") == "strategy":
        return signals
    if not signals.index.equals(frame.index):
        # 覆盖层按整帧计算距离，索引必须对齐，否则会静默产生 NaN 止损
        signals = signals.reindex(frame.index)

    sl_pct, _, level, fallback = overlay_distances(frame, signals, config)
    target_r = float(config["params"]["target_r"])
    # 显式价格距离：只做下限保护（费用占比过小的止损/止盈没有意义），
    # 距离无效（ATR 缺失、支撑位算不出）的行交给 finalize 标记为 invalid_distance。
    sl_pct = sl_pct.clip(lower=float(params["min_sl_pct"]))
    tp_pct = (sl_pct * target_r).clip(lower=float(params["min_tp_pct"]))

    reason = signals["block_reason"].astype(object)
    # 上一轮基于旧距离打的标签必须清掉，否则新距离明明达标也会被挡住
    reason = reason.mask(reason.isin(("cost_filter", "invalid_distance")), "")
    rebuilt = finalize_signal_frame(
        frame,
        params,
        signed=signals["signed"],
        score=signals["score"],
        prob_up=signals["prob_up"],
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        reason=reason,
        apply_min_distances=False,
    )
    # 保留策略自己的附加列（如 ema_exit 的 exit_long、zigzag 的通道列）
    result = signals.copy()
    for name in rebuilt.columns:
        result[name] = rebuilt[name]
    result["exit_level"] = level.to_numpy(dtype=float)
    result["exit_fallback"] = fallback.to_numpy(dtype=int)
    result["exit_method"] = config.get("method", "")
    return result
