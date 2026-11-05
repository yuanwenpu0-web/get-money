"""绑定层出场覆盖：把「止损/止盈怎么算」从策略内部提到绑定层。

一条绑定除了「策略 × 池 × 周期 × 参数」，还带两块出场配置：

- `exit_method`：`strategy`（默认，沿用策略自算的 sl_pct/tp_pct，零行为变化）
  / `atr_r`（ATR 止损 + R 倍数止盈）/ `support_r`（支撑位下方 buffer 止损 + R 倍数止盈）；
- `support_indicator`：`support_r` 使用的支撑位来源，见 `support_levels.METHODS`。

覆盖发生在 `rules.evaluate_signals` 里、策略产出信号之后：先算出新的 sl/tp，
再用同一套 `signals.finalize_signal_frame` 重算净盈亏比、成本过滤、杠杆与仓位——
仓位与风险预算仍然按**新的**止损距离截断，不会出现「止损改了、仓位没改」的口径分裂。

为什么不是「策略参数」：出场方法属于执行口径，跨策略通用；挂在参数里会让每个 spec
都要声明一遍，且与 `resolve_params` 的数值校验冲突。这里用绑定上的独立字段，
解析时以保留键 `exit` 注入有效参数（`resolve_params` 负责校验），
于是回测 / 模拟盘 / 巡检 / 订单管理全部自动拿到同一份配置。

未经统计支持的组合：`support_r` 的支撑位量具在 2026-11-05 的诊断评估里 verdict 为
`inconclusive`（11 种定义都不优于随机价位），因此 UI 与描述统一标注「未经统计支持」，
默认仍为 `strategy`。改动默认值必须按手册 Gate 3–4 出 A/B 报告。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .support_levels import compute_pair, get_source, has_source

# 保留键：绑定层出场配置在有效参数里的存放位置
EXIT_KEY = "exit"

DEFAULT_EXIT: dict = {"method": "strategy", "support": "", "params": {}}


@dataclass(frozen=True)
class ExitMethod:
    code: str
    label: str
    description: str
    defaults: dict
    ranges: dict
    needs_support: bool = False
    # 是否经过统计验证；False 时 API/UI 必须显式标注「未经统计支持」
    verified: bool = False


EXIT_METHODS: tuple[ExitMethod, ...] = (
    ExitMethod(
        "strategy", "策略自算（默认）",
        "沿用策略自己产出的止损/止盈距离。任何改动前的行为都与启用绑定层出场前一致。",
        {}, {},
    ),
    ExitMethod(
        "atr_r", "ATR 止损 + R 倍数止盈",
        "止损距离 = atr_mult × ATR%；止盈距离 = target_r × 止损距离。支撑位不参与。",
        {"atr_mult": 1.5, "target_r": 2.0},
        {"atr_mult": (0.2, 10.0), "target_r": (0.2, 20.0)},
        verified=True,  # 与现行 ATR 止损同族参数，不引入新信息源
    ),
    ExitMethod(
        "support_r", "支撑位止损 + R 倍数止盈",
        "止损放在所选支撑位下方 buffer（buffer_atr × ATR）处；止盈 = target_r × 实际止损距离。"
        "支撑位缺失或距离超过 max_sl_atr × ATR 时回退 ATR 止损。做空用镜像的最近上方阻力。",
        {"atr_mult": 1.5, "target_r": 2.0, "buffer_atr": 0.25, "max_sl_atr": 3.0},
        {
            "atr_mult": (0.2, 10.0),
            "target_r": (0.2, 20.0),
            "buffer_atr": (0.0, 3.0),
            "max_sl_atr": (0.5, 10.0),
        },
        needs_support=True,
        verified=False,  # 支撑位量具 verdict=inconclusive
    ),
)

METHOD_MAP = {method.code: method for method in EXIT_METHODS}
EXIT_METHOD_KEYS = tuple(method.code for method in EXIT_METHODS)

# 专用 broker 的策略：止损止盈由 broker 自己按价格路径推导，通用出场覆盖不会生效，
# 允许保存只会让「回测/实盘看起来改了其实没改」，因此在绑定校验阶段直接拒绝。
BROKER_BLOCKED = ("opening_candle_breakout", "rgd_donchian_v2", "simple_zigzag_double")
# 注册表标签层面的同类拦截：研究池策略的 direction 恒为 0（由专用模拟流程按时长开仓），
# 覆盖层会把「本来不该开仓」的行变成可执行信号，属于伪造 edge。
BLOCKED_TAGS = ("research", "dedicated_broker", "stop_follow")


def get_method(code: str) -> ExitMethod:
    method = METHOD_MAP.get(code)
    if method is None:
        raise ValueError(f"未知出场方法: {code}")
    return method


def normalize(raw: object) -> dict:
    """校验并补全出场配置；非法配置抛 ValueError（绑定保存时直接 400）。"""
    if raw is None or raw == "":
        return {"method": "strategy", "support": "", "params": {}}
    if not isinstance(raw, dict):
        raise ValueError("出场配置必须是 JSON 对象")
    unknown = set(raw) - {"method", "support", "params"}
    if unknown:
        raise ValueError(f"未知出场配置字段: {', '.join(sorted(unknown))}")

    method = get_method(str(raw.get("method") or "strategy"))
    support = str(raw.get("support") or "")
    if support and not has_source(support):
        raise ValueError(f"未知支撑线指标: {support}")
    if method.needs_support and not support:
        raise ValueError(f"{method.label} 必须选择支撑线指标")

    params = raw.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("出场参数必须是 JSON 对象")
    unknown_params = set(params) - set(method.defaults)
    if unknown_params:
        raise ValueError(
            f"{method.code} 不支持参数: {', '.join(sorted(unknown_params))}"
        )

    merged = {**method.defaults, **params}
    for key, value in merged.items():
        low, high = method.ranges[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not low <= value <= high
        ):
            raise ValueError(f"{method.code}.{key} 必须在 [{low}, {high}] 范围内")
        merged[key] = float(value)
    return {"method": method.code, "support": support, "params": merged}


def ensure_allowed(strategy: str, config: dict, params: dict | None = None) -> None:
    """拒绝「专用撮合 / 研究流程」策略上的出场覆盖。

    这些策略的止损止盈不由信号帧的 sl_pct/tp_pct 表达（broker 自推、或方向恒为 0），
    覆盖它们只会造出口径分裂或伪造信号，因此在参数校验阶段就报错。
    """
    if config.get("method", "strategy") == "strategy":
        return
    from .registry import get_spec, has_spec

    if strategy in BROKER_BLOCKED:
        raise ValueError(f"{strategy} 使用专用撮合，出场距离由 broker 推导，不能挂绑定层出场覆盖")
    if has_spec(strategy):
        spec = get_spec(strategy)
        blocked = set(spec.tags) & set(BLOCKED_TAGS)
        if blocked:
            raise ValueError(
                f"{strategy} 的出场由 {'/'.join(sorted(blocked))} 流程决定，不能挂绑定层出场覆盖"
            )
    if strategy == "supertrend_current" and (params or {}).get("cl_enabled"):
        raise ValueError("supertrend_current 开启订单闭环(cl_enabled)后出场由闭环状态机决定，请先关闭闭环")


def label(config: dict) -> str:
    """可读标签：`策略自算` / `ATR 1.5× ・ 2R` / `支撑位(pivot_pool) ・ 2R`。"""
    if not config:
        return "策略自算"
    method = METHOD_MAP.get(config.get("method", "strategy"))
    if method is None or method.code == "strategy":
        return "策略自算"
    params = config.get("params") or {}
    target = params.get("target_r")
    if method.code == "atr_r":
        return f"ATR {params.get('atr_mult')}× · {target}R"
    source = get_source(config.get("support") or "")
    return f"支撑位({source.label}) · {target}R"


def _atr(frame: pd.DataFrame) -> np.ndarray:
    if "atr_pct" not in frame.columns:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame["atr_pct"], errors="coerce").to_numpy(dtype=float)


def _support_distance(
    frame: pd.DataFrame, direction: np.ndarray, config: dict, atr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """支撑位止损：返回 (sl_pct, 使用到的水平价, 是否回退)。

    只在方向非零处计算；`level` 必须站在价格正确一侧（多头在下方、空头在上方），
    且止损距离不超过 `max_sl_atr × ATR`，否则该行回退 ATR 止损。
    """
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    params = config["params"]
    support, resistance = compute_pair(frame, config["support"])
    sup = support.to_numpy(dtype=float)
    res = resistance.to_numpy(dtype=float)
    # 方向为 0 的行不交易，水平价留空，避免把「上方阻力」当成支撑位展示
    level = np.where(direction > 0, sup, np.where(direction < 0, res, np.nan))

    buffer = params["buffer_atr"] * atr
    with np.errstate(invalid="ignore", divide="ignore"):
        stop = np.where(direction > 0, level * (1.0 - buffer), level * (1.0 + buffer))
        distance = np.abs(close - stop) / close

    side_ok = np.where(direction > 0, level < close, level > close)
    valid = (
        np.isfinite(level)
        & (level > 0)
        & np.isfinite(distance)
        & (distance > 0)
        & side_ok
        & (distance <= params["max_sl_atr"] * atr)
    )
    fallback = (direction != 0) & ~valid
    sl_pct = np.where(valid, distance, params["atr_mult"] * atr)
    return sl_pct, np.where(valid, level, np.nan), fallback


def overlay_distances(
    frame: pd.DataFrame, signals: pd.DataFrame, config: dict
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """按绑定出场配置算新的 (sl_pct, tp_pct, 使用的水平价, 是否回退)。

    回退标志用于诊断：`support_r` 找不到合格支撑位时该根改用 ATR 止损。
    """
    method = get_method(config.get("method", "strategy"))
    index = frame.index
    atr = _atr(frame)
    direction = (
        pd.to_numeric(signals.get("direction"), errors="coerce").fillna(0).to_numpy(dtype=float)
        if "direction" in signals.columns
        else np.zeros(len(frame))
    )
    if method.code == "support_r":
        sl_pct, level, fallback = _support_distance(frame, direction, config, atr)
    else:
        sl_pct = np.full(len(frame), config["params"]["atr_mult"]) * atr
        level = np.full(len(frame), np.nan)
        fallback = np.zeros(len(frame), dtype=bool)
    tp_pct = sl_pct * config["params"]["target_r"]
    return (
        pd.Series(sl_pct, index=index, dtype="float64"),
        pd.Series(tp_pct, index=index, dtype="float64"),
        pd.Series(level, index=index, dtype="float64"),
        pd.Series(fallback.astype(int), index=index, dtype="int64"),
    )
