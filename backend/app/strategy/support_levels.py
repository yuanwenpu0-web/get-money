"""支撑/阻力位来源登记簿——绑定层「支撑线指标」的唯一来源。

11 种定义与 `research/support_level_eval.py` 的评估口径**完全一致**：评估脚本从
这里导入 `METHODS`，避免「研究用一套、实盘用另一套」的口径分叉。

因果性：所有来源只用已确认的枢轴（Pine `ta.pivothigh(left, right)` 的值在枢轴后
`right` 根才出现）或上一个完整周期的水平，所以 `level[t]` 只依赖 `t` 及之前的数据。
绑定层的用法是「第 t 根收盘决策、第 t+1 根开盘成交」，`level[t]` 在该时点已知，
不构成未来函数。

`indicator is None` 的来源（如 donchian 基线）走本地 `transform(frame)`；
其余来源的 `transform` 作用于**指标输出帧**，`column` 缺省时按列名直取。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..catalog_indicators import calculate


def _numeric(series: pd.Series) -> np.ndarray:
    value = pd.to_numeric(series, errors="coerce")
    try:
        return value.to_numpy(dtype="float64", na_value=np.nan)
    except TypeError:
        return value.to_numpy(dtype="float64")


def _zone_mid(out: pd.DataFrame) -> np.ndarray:
    return (_numeric(out["srz_support_top"]) + _numeric(out["srz_support_bottom"])) / 2.0


def _zone_mid_above(out: pd.DataFrame) -> np.ndarray:
    return (_numeric(out["srz_resistance_top"]) + _numeric(out["srz_resistance_bottom"])) / 2.0


def _donchian(frame: pd.DataFrame, window: int = 20) -> np.ndarray:
    """不含当前 bar 的滚动最低价；min_periods=window 防止预热期伪支撑。"""
    return frame["low"].shift(1).rolling(window, min_periods=window).min().to_numpy(dtype=float)


def _donchian_high(frame: pd.DataFrame, window: int = 20) -> np.ndarray:
    return frame["high"].shift(1).rolling(window, min_periods=window).max().to_numpy(dtype=float)


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    indicator: str | None
    params: dict
    column: str | None
    transform: Callable[[pd.DataFrame], np.ndarray] | None = None
    # 做空方向的镜像水平（最近上方阻力）。为空时做空不提供水平，由出场层回退。
    resistance_column: str | None = None
    resistance_transform: Callable[[pd.DataFrame], np.ndarray] | None = None


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("pivot_pool", "枢轴池合并·最近下方档", "support_resistance",
               {"strength": 3, "max_levels": 5}, "nearest_support",
               resistance_column="nearest_resistance"),
    MethodSpec("mtf_pivot_1h", "多周期已确认枢轴·1h 档", "mtf_nearest_support_resistance",
               {}, "support_1", resistance_column="resistance_1"),
    MethodSpec("mtf_pivot_4h", "多周期已确认枢轴·4h 档", "mtf_nearest_support_resistance",
               {}, "support_2", resistance_column="resistance_2"),
    MethodSpec("plasmazero", "枢轴区+触及计数", "sr_zones_plasmazero",
               {}, "nearest_support_below", resistance_column="nearest_resistance_above"),
    MethodSpec("sr_zones_wu", "7 探测器打分支撑区中点", "support_and_resistance_zones",
               {}, None, _zone_mid, resistance_transform=_zone_mid_above),
    MethodSpec("sr_zones_mtf", "ATR swing 区·HTF 投影", "s_r_zones_mtf",
               {"htf1_minutes": 240, "htf2_minutes": 1440, "use_htf2": True}, "nearest_support",
               resistance_column="nearest_resistance"),
    MethodSpec("edge_registry", "多源水平登记簿·最近下方", "edge_atlas",
               {}, "ea_nearest_below", resistance_column="ea_nearest_above"),
    MethodSpec("pdh_pdl", "前日低点", "capital41_pdh_pdl",
               {}, "previous_day_low", resistance_column="previous_day_high"),
    MethodSpec("pivot_map", "经典枢轴+confluence·最近下方", "market_structure_pivot_map",
               {}, "near_below", resistance_column="near_above"),
    MethodSpec("trendline", "枢轴连线·动态趋势支撑", "auto_trendlines",
               {"lookback": 120, "max_pivots": 20}, "support_level",
               resistance_column="resistance_level"),
    MethodSpec("donchian20", "20 根滚动最低价（朴素基线）", None,
               {}, None, _donchian, resistance_transform=_donchian_high),
)
METHOD_KEYS = tuple(m.key for m in METHODS)
SOURCE_MAP = {m.key: m for m in METHODS}


def source_keys() -> tuple[str, ...]:
    return METHOD_KEYS


def has_source(key: str) -> bool:
    return key in SOURCE_MAP


def get_source(key: str) -> MethodSpec:
    spec = SOURCE_MAP.get(key)
    if spec is None:
        raise ValueError(f"未知支撑位来源: {key}")
    return spec


def compute_levels(
    frame: pd.DataFrame,
    keys: tuple[str, ...] = METHOD_KEYS,
    errors: dict | None = None,
) -> dict[str, np.ndarray]:
    """逐方法产出与 frame 等长的支撑位价格序列（NaN = 该 bar 无支撑位）。"""
    levels: dict[str, np.ndarray] = {}
    cache: dict[str, pd.DataFrame] = {}
    for spec in METHODS:
        if spec.key not in keys:
            continue
        try:
            if spec.indicator is None:
                value = spec.transform(frame)
            else:
                if spec.indicator not in cache:
                    cache[spec.indicator] = calculate(spec.indicator, frame, **spec.params)
                out = cache[spec.indicator]
                value = spec.transform(out) if spec.transform else _numeric(out[spec.column])
            levels[spec.key] = np.asarray(value, dtype=float)
        except Exception as exc:  # noqa: BLE001 — 单方法失败不拖垮整币
            levels[spec.key] = np.full(len(frame), np.nan)
            if errors is not None:
                errors[spec.key] = f"{type(exc).__name__}: {str(exc)[:180]}"
    return levels


def _series(values, index: pd.Index) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=index, dtype="float64")


def compute_pair(frame: pd.DataFrame, key: str, errors: dict | None = None) -> tuple[pd.Series, pd.Series]:
    """支撑/阻力两条与 frame 对齐的序列；任一侧缺失时该侧为 NaN。

    单侧失败（例如指标缺列）只让该侧变 NaN，不抛错——出场层会按 ATR 回退，
    避免一个可选量具把整条绑定打成「评估失败」。
    """
    spec = get_source(key)
    empty = _series(np.full(len(frame), np.nan), frame.index)
    try:
        if spec.indicator is None:
            support = spec.transform(frame)
            resistance = spec.resistance_transform(frame) if spec.resistance_transform else None
        else:
            out = calculate(spec.indicator, frame, **spec.params)
            support = spec.transform(out) if spec.transform else _numeric(out[spec.column])
            if spec.resistance_transform is not None:
                resistance = spec.resistance_transform(out)
            elif spec.resistance_column and spec.resistance_column in out.columns:
                resistance = _numeric(out[spec.resistance_column])
            else:
                resistance = None
    except Exception as exc:  # noqa: BLE001
        if errors is not None:
            errors[key] = f"{type(exc).__name__}: {str(exc)[:180]}"
        return empty, empty
    return (
        _series(support, frame.index),
        _series(resistance, frame.index) if resistance is not None else empty.copy(),
    )
