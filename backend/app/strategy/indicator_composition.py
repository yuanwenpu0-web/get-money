"""独立的策略指标组合；不注册策略，也不接入交易执行。"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .indicators import StrategyIndicator
from .registry import PrepareContext, get_spec
from .rules import apply_exit_overlay
from .signals import SIGNAL_COLUMNS


@dataclass(frozen=True)
class Component:
    name: str
    indicator: StrategyIndicator
    params: dict


class IndicatorComposition:
    """首项提供信号，其余项须同方向确认；每项独立计算，避免特征列串扰。"""

    def __init__(self, config: dict, *, state: dict | None = None, interval: str | None = None):
        if not isinstance(config, dict) or set(config) != {"components"}:
            raise ValueError("组合配置必须且只能包含 components")
        entries = config["components"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("components 必须是非空列表")
        components = []
        names = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) - {"name", "code", "params"}:
                raise ValueError("指标配置只能包含 name、code、params")
            name, code = entry.get("name"), entry.get("code")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError("指标名称必须非空且唯一")
            if not isinstance(code, str) or not code:
                raise ValueError("指标 code 必须非空")
            indicator = StrategyIndicator(get_spec(code))
            overrides = entry.get("params", {})
            if not isinstance(overrides, dict):
                raise ValueError("指标 params 必须是对象")
            params = indicator.validate(state=state, overrides=overrides, interval=interval)
            components.append(Component(name, indicator, params))
            names.add(name)
        self.components = tuple(components)

    def lookback(self) -> int:
        return max(c.indicator.lookback(c.params) for c in self.components)

    def calculate(self, frame: pd.DataFrame, *, ctx: PrepareContext | None = None) -> dict[str, pd.DataFrame]:
        """返回各项完整信号帧；调用方负责输入数据的时间顺序和连续性。"""
        results = {}
        for component in self.components:
            data = component.indicator.prepare(frame.copy(deep=True), component.params, ctx)
            signals = component.indicator.signals(data, component.params)
            if not signals.index.equals(frame.index) or not data.index.equals(frame.index):
                raise ValueError(f"{component.name} 的索引与输入不一致")
            missing = set(SIGNAL_COLUMNS) - set(signals.columns)
            if missing:
                raise ValueError(f"{component.name} 缺少信号列: {', '.join(sorted(missing))}")
            if not signals["direction"].isin((-1, 0, 1)).all():
                raise ValueError(f"{component.name} 的信号方向无效")
            results[component.name] = apply_exit_overlay(data, signals, component.params)
        return results

    def evaluate(self, frame: pd.DataFrame, *, ctx: PrepareContext | None = None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        results = self.calculate(frame, ctx=ctx)
        primary = results[self.components[0].name].copy()
        allowed = primary["direction"].ne(0)
        for component in self.components[1:]:
            allowed &= results[component.name]["direction"].eq(primary["direction"])
        blocked = primary["direction"].ne(0) & ~allowed
        primary.loc[blocked, "direction"] = 0
        primary.loc[blocked, ["position_pct", "leverage", "risk_pct"]] = 0
        primary.loc[blocked, "block_reason"] = "composition_confirmation"
        return primary, results
