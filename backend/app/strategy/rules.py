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
from .registry import PrepareContext, available_codes, get_spec
from .signals import SIGNAL_COLUMNS

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
    """
    frame = prepare_frame(data, code, params, symbol=symbol, service=service)
    result = get_spec(code).signals(frame, params)
    missing = set(SIGNAL_COLUMNS) - set(result.columns)
    if missing:
        raise ValueError(f"{code} 输出的信号帧缺少列: {', '.join(sorted(missing))}")
    return result
