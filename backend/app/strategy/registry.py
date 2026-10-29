"""策略注册表：把「策略实现」与「公用管线」解耦。

一个策略 = 一个 `StrategySpec`，只描述自己特有的部分：

| 字段 | 作用 |
| --- | --- |
| `defaults` / `ranges` / `bool_keys` | 参数模型，供 `resolve_params` 合并与校验 |
| `lookback(params)` | 指标预热需要多少根K线（决定取数窗口） |
| `prepare(frame, params, ctx)` | 策略自有特征或上下文富化（只加列、纯因果） |
| `signals(frame, params)` | 产出统一信号帧（`signals.SIGNAL_COLUMNS`） |
| `grid` | 参数寻优的候选值（每键一组离散取值，笛卡尔积过大时随机采样） |

新增一个 TradingView 迁移过来的策略 = 在 `specs/` 下加一个模块并调用 `register()`：
引擎、巡检、回测、撮合、参数校验、API 全部自动复用，不需要改任何公用代码。

注册顺序无关；`specs/__init__.py` 负责导入所有内置策略以触发注册。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd


@dataclass(frozen=True)
class PrepareContext:
    """`prepare()` 的上下文：需要更低周期数据（如 intrabar delta）时用。

    回测与特征测试里 `service` 为 None，策略必须能自己在缺数据时降级，
    不能假设一定有数据库可用。
    """

    symbol: str = ""
    service: Any | None = None


LookbackFn = Callable[[dict], int]
PrepareFn = Callable[[pd.DataFrame, dict, PrepareContext], pd.DataFrame]
SignalsFn = Callable[[pd.DataFrame, dict], pd.DataFrame]
DescribeFn = Callable[[pd.Series], dict]
ValidateFn = Callable[[dict], None]


@dataclass(frozen=True)
class StrategySpec:
    code: str
    name: str
    description: str
    defaults: dict[str, Any]
    ranges: dict[str, tuple[float, float]]
    lookback: LookbackFn
    prepare: PrepareFn
    signals: SignalsFn
    # 可选：把末根的特征快照写进信号的 reason，纯观测用途，不影响任何决策
    describe: DescribeFn | None = None
    validate: ValidateFn | None = None
    bool_keys: tuple[str, ...] = ()
    # 只用于整数校验：`*_bars` 这类已在 resolve_params 里按后缀兜底，这里补显式声明的
    int_keys: tuple[str, ...] = ()
    # 首次入库时是否默认启用；未通过统计检验的策略应显式设为 False
    default_enabled: bool = True
    tags: tuple[str, ...] = field(default_factory=tuple)
    # 参数寻优的搜索空间：`{参数名: [候选值, ...]}`。
    # 声明业务上真的会取的离散档位，而不是把 ranges 整个区间铺开——吸收阈值的
    # 90/95/97 分位有实际含义，88.3 分位没有。空 dict 表示该策略不支持寻优。
    grid: dict[str, tuple] = field(default_factory=dict)
    # 会影响 `prepare()` 输出的参数键。寻优时同一币种的组合按这组键分组，
    # 组内复用同一份已富化的帧（prepare 要算滚动百分位，是每币最贵的一步）。
    prepare_keys: tuple[str, ...] = ()


_SPECS: dict[str, StrategySpec] = {}


def register(spec: StrategySpec) -> StrategySpec:
    if spec.code in _SPECS:
        raise ValueError(f"策略重复注册: {spec.code}")
    if not spec.code or not spec.defaults:
        raise ValueError("策略必须有 code 与非空 defaults")
    _SPECS[spec.code] = spec
    return spec


def get_spec(code: str) -> StrategySpec:
    spec = _SPECS.get(code)
    if spec is None:
        raise ValueError(f"未知策略: {code}")
    return spec


def has_spec(code: str) -> bool:
    return code in _SPECS


def all_specs() -> tuple[StrategySpec, ...]:
    return tuple(_SPECS.values())


def available_codes() -> tuple[str, ...]:
    return tuple(_SPECS)
