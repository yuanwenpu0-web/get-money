"""旧策略的指标兼容接口。

显式调用 register_strategy_indicator(code) 将已注册的策略暴露为指标；
指标注册表独立于策略注册表，不会启用策略或修改交易管线。
参数解析复用策略的全局设置、默认值与覆盖值规则。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .registry import PrepareContext, StrategySpec
from .rules import get_spec, resolve_params


@dataclass(frozen=True)
class StrategyIndicator:
    """将已注册的 StrategySpec 作为指标调用。"""

    spec: StrategySpec

    def __post_init__(self) -> None:
        if get_spec(self.spec.code) is not self.spec:
            raise ValueError(f"策略实例与注册表不一致: {self.spec.code}")

    @property
    def code(self) -> str:
        return self.spec.code

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def description(self) -> str:
        return self.spec.description

    @property
    def defaults(self) -> dict[str, Any]:
        return dict(self.spec.defaults)

    @property
    def ranges(self) -> dict[str, tuple[float, float]]:
        return dict(self.spec.ranges)

    @property
    def bool_keys(self) -> tuple[str, ...]:
        return self.spec.bool_keys

    @property
    def int_keys(self) -> tuple[str, ...]:
        return self.spec.int_keys

    @property
    def prepare_keys(self) -> tuple[str, ...]:
        return self.spec.prepare_keys

    def resolve_params(self, state: dict | None = None, overrides: dict | None = None,
                       interval: str | None = None) -> dict:
        return resolve_params(self.code, {} if state is None else state, overrides, interval)

    def validate(self, state: dict | None = None, overrides: dict | None = None,
                 interval: str | None = None) -> dict:
        """校验并返回合并后的完整参数。"""
        return self.resolve_params(state, overrides, interval)

    def lookback(self, params: dict) -> int:
        return self.spec.lookback(params)

    def prepare(self, frame: pd.DataFrame, params: dict,
                ctx: PrepareContext | None = None) -> pd.DataFrame:
        return self.spec.prepare(frame, params, ctx if ctx is not None else PrepareContext())

    def signals(self, frame: pd.DataFrame, params: dict) -> pd.DataFrame:
        return self.spec.signals(frame, params)


_INDICATORS: dict[str, StrategyIndicator] = {}


def register_strategy_indicator(code: str) -> StrategyIndicator:
    """注册一个已存在的策略指标；重复注册会报错。"""
    if code in _INDICATORS:
        raise ValueError(f"指标重复注册: {code}")
    indicator = StrategyIndicator(get_spec(code))
    _INDICATORS[code] = indicator
    return indicator


def get_indicator(code: str) -> StrategyIndicator:
    try:
        return _INDICATORS[code]
    except KeyError:
        raise ValueError(f"未知指标: {code}") from None


def all_indicators() -> tuple[StrategyIndicator, ...]:
    return tuple(_INDICATORS.values())
