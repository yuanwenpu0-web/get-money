"""趋势方向指标的统一「方向」提取与单币评估（1h K 线）。

契约（避免前视与伪显著）：

* 方向 ``d[t]`` = 指标在 bar t **收盘**时给出的状态，只用 t 及之前的数据。
* 未来收益 ``r[t] = close[t+H] / open[t+1] - 1``：t+1 开盘入场、t+H 收盘出场，
  与项目其它研究管线「bar t 判定、t+1 开盘成交」一致。
* ``H = 48`` 根 1h 即「这两天」；历史评估用 ``stride = H`` 的**不重叠**决策点，
  否则 48 倍重叠样本会把标准误压到虚低、凭空造出显著性。
* 成本按每次往返双边扣（默认 0.12%），只在 ``d != 0``（真正开仓）时扣。

本模块只做「单币 → 每指标 → 样本数组」，跨币/按池聚合与报告在
``backend/scripts/trend_direction_backtest.py``。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.catalog_indicators import calculate

HOURS_PER_DAY = 24
DEFAULT_HORIZON = 48
DEFAULT_COST = 0.0012
WARMUP_BARS = 260  # 最慢的指标（200 根均线 / 一目均衡 120 根）需要的预热


@dataclass(frozen=True)
class DirectionSpec:
    name: str
    label: str
    columns: tuple[str, ...]
    source: str
    mode: str = "sign"          # sign | diff | regime
    params: dict = field(default_factory=dict)
    note: str = ""
    alias_of: str | None = None  # 默认参数下与另一指标逐根同值，直接复用其方向


SPECS: tuple[DirectionSpec, ...] = (
    # --- pine-collection / indicators/trend_direction (9) ---
    DirectionSpec("adaptive_arithmetic_candles_quality_score", "自适应算术K线质量方向",
                  ("direction",), "trend_direction", note="direction 列直接给出 -1/0/1"),
    DirectionSpec("chandelier_flip_radar", "吊灯翻转雷达",
                  ("direction",), "trend_direction", note="ATR 吊灯止损方向 -1/1"),
    DirectionSpec("ma_cascade_engine", "均线级联引擎",
                  ("mce_bias",), "trend_direction", note="mce_bias -2..2 取符号"),
    DirectionSpec("ma_regime_bands", "均线状态带",
                  ("bull", "bear"), "trend_direction", mode="diff", note="bull - bear 取符号"),
    DirectionSpec("market_average_relationship_engine", "价格均线关系引擎",
                  ("mare_trend_direction",), "trend_direction", note="mare_trend_direction -1/0/1"),
    DirectionSpec("modern_trend_regime", "现代趋势状态",
                  ("mtr_regime",), "trend_direction", mode="regime",
                  note="regime: 1=多 / -1=空 / 2=震荡 / 0=过渡，后两者视为中性"),
    DirectionSpec("smooth_trend_radar", "平滑趋势雷达",
                  ("trend",), "trend_direction", note="双重平滑 Supertrend 方向 -1/0/1"),
    DirectionSpec("vein_accumulation_phase_4h", "吸筹派发阶段",
                  ("accum_score", "dist_score"), "trend_direction", mode="diff",
                  note="派生方向：吸筹分 - 派发分 取符号"),
    DirectionSpec("vein_reversal_zones", "反转区域标签",
                  ("reversal_label",), "trend_direction", note="reversal_label -2..2 取符号"),
    # --- pine-crypto-collection 标注为「趋势方向」(8 中可回测的 6) ---
    DirectionSpec("mtf_strategy_visual", "多周期EMA云",
                  ("htf_bias",), "crypto",
                  params={"htf_minutes": 240, "mtf_minutes": 60},
                  note="1h 图上 MTF 必须是 60 的整数倍，故 60/240"),
    DirectionSpec("ichi_workflow_overlay", "一目均衡评分卡",
                  ("price_state",), "crypto",
                  note="价格相对云图 -1/0/1"),
    DirectionSpec("ichi_workflow_overlay_profiles", "一目均衡品种配置",
                  ("price_state",), "crypto", note="默认参数下与 ichi_workflow_overlay 共用同一实现、方向逐根同值，已复用其计算（已用真实数据验证）",
                  alias_of="ichi_workflow_overlay"),
    DirectionSpec("ichi_2d_overlay_profiles", "一目均衡两日云",
                  ("price_state",), "crypto", note="默认参数下与 ichi_workflow_overlay 共用同一实现、方向逐根同值，已复用其计算（已用真实数据验证）",
                  alias_of="ichi_workflow_overlay"),
    DirectionSpec("multi_timeframe_trend", "多周期趋势汇总",
                  ("trend_score",), "crypto",
                  params={"tf1_minutes": 60, "tf2_minutes": 120, "tf3_minutes": 240,
                          "tf4_minutes": 720, "tf5_minutes": 1440},
                  note="1h 图上五个周期都取 60 的整数倍，trend_score -5..5 取符号"),
    DirectionSpec("bullish_lines_four_state", "四态均线排列",
                  ("is_bullish", "is_bearish"), "crypto", mode="diff",
                  params={"fast_length": 20, "mid_warmer_length": 50,
                          "mid_cooler_length": 100, "slow_length": 200},
                  note="上游默认 4 个长度全 0 → 恒 Unconfigured，必须显式配置"),
)

SPECS_BY_NAME = {spec.name: spec for spec in SPECS}

# 分类内但无法在「单币 1h OHLCV」上做方向回测的两个，记录原因而非静默丢弃
EXCLUDED: dict[str, str] = {
    "btc_altcoin_trend": "需要 btc_close / dominance_close 外部行情列（山寨币市值占比），单币 1h OHLCV 无法提供",
    "btc_logarithmic_growth_curves": "输出是 BTC 长期对数增长估值带，无 1h 多空方向列，波段数值比现价高一个数量级",
}


def _numeric(series: pd.Series) -> pd.Series:
    """兼容 bool / Int64(pd.NA) / object 列，统一成 float64 + NaN。"""
    value = pd.to_numeric(series, errors="coerce")
    try:
        array = value.to_numpy(dtype="float64", na_value=np.nan)
    except TypeError:
        array = value.to_numpy(dtype="float64")
    return pd.Series(array, index=series.index)


def extract_direction(output: pd.DataFrame, spec: DirectionSpec) -> pd.Series:
    """把某个指标的输出列折叠成 -1 / 0 / +1 的方向序列。"""
    missing = [col for col in spec.columns if col not in output.columns]
    if missing:
        raise KeyError(f"{spec.name} 缺列 {missing}")
    if spec.mode == "sign":
        raw = _numeric(output[spec.columns[0]])
    elif spec.mode == "diff":
        raw = _numeric(output[spec.columns[0]]) - _numeric(output[spec.columns[1]])
    elif spec.mode == "regime":
        raw = _numeric(output[spec.columns[0]]).map(
            lambda x: 1.0 if x == 1 else (-1.0 if x == -1 else 0.0)
        )
    else:
        raise ValueError(f"未知方向模式 {spec.mode}")
    direction = np.sign(raw.fillna(0.0).to_numpy(dtype="float64"))
    return pd.Series(direction, index=output.index, name=spec.name)


def compute_directions(frame: pd.DataFrame, specs: tuple[DirectionSpec, ...] = SPECS
                       ) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """在整段（因果）K 线上计算所有指标方向。失败指标返回全 NaN 并记录错误。"""
    directions: dict[str, pd.Series] = {}
    errors: dict[str, str] = {}
    for spec in specs:
        if spec.alias_of and spec.alias_of in directions:
            # 已用真实数据逐根验证：默认参数下 price_state 等方向列完全相同
            directions[spec.name] = directions[spec.alias_of].rename(spec.name)
            continue
        try:
            output = calculate(spec.name, frame, **spec.params)
            directions[spec.name] = extract_direction(output, spec)
        except Exception as exc:  # noqa: BLE001 单指标失败不拖垮整币
            errors[spec.name] = f"{type(exc).__name__}: {str(exc)[:180]}"
            directions[spec.name] = pd.Series(
                np.full(len(frame), np.nan), index=frame.index, name=spec.name
            )
    return directions, errors


def decision_points(n_bars: int, horizon: int, eval_days: int,
                    warmup: int = WARMUP_BARS) -> list[int]:
    """不重叠决策点，最后一个恒为「两天前那根」（持有期正好覆盖到最新一根）。"""
    last_t = n_bars - 1 - horizon
    first_t = max(warmup, last_t - eval_days * HOURS_PER_DAY)
    if last_t < first_t:
        return []
    points = list(range(last_t, first_t - 1, -horizon))
    return sorted(points)


def evaluate_symbol(frame: pd.DataFrame, directions: dict[str, pd.Series],
                    errors: dict[str, str] | None = None,
                    horizon: int = DEFAULT_HORIZON,
                    cost: float = DEFAULT_COST,
                    eval_days: int = 180) -> dict | None:
    """单币评估：返回每个指标的样本数组 + 近两日样本 + 当前方向。"""
    if frame is None or len(frame) < WARMUP_BARS + horizon + 2:
        return None
    close = frame["close"].to_numpy(dtype="float64")
    open_ = frame["open"].to_numpy(dtype="float64")
    times = frame.index
    points = decision_points(len(frame), horizon, eval_days)
    if not points:
        return None
    returns = np.array([
        close[t + horizon] / open_[t + 1] - 1.0 if open_[t + 1] > 0 else np.nan
        for t in points
    ], dtype="float64")

    payload: dict = {
        "n_bars": len(frame),
        "first_time": str(times[0]),
        "last_time": str(times[-1]),
        "decision_count": len(points),
        "decision_times": [str(times[t]) for t in points],
        "two_day_return": float(returns[-1]) if np.isfinite(returns[-1]) else None,
        "two_day_decision_time": str(times[points[-1]]),
        "indicators": {},
    }
    for name, series in directions.items():
        values = series.to_numpy(dtype="float64")
        valid = np.isfinite(values[points]) & np.isfinite(returns)
        idx = np.flatnonzero(valid)
        d = values[[points[i] for i in idx]]
        r = returns[idx]
        gross = d * r
        net = gross - np.where(d != 0, cost, 0.0)
        hit = (d != 0) & (gross > 0)
        two_day = None
        # 最后一个决策点 = 近两日样本（若该指标在那根有有效方向）
        if len(points) and np.isfinite(values[points[-1]]) and np.isfinite(returns[-1]):
            d0 = float(values[points[-1]])
            r0 = float(returns[-1])
            n0 = float(d0 * r0 - (cost if d0 != 0 else 0.0))
            two_day = {
                "decision_time": str(times[points[-1]]),
                "direction": d0,
                "realized_return": r0,
                "strategy_return": float(d0 * r0),
                "net_return": n0,
                "hit": bool(d0 != 0 and d0 * r0 > 0),
            }
        payload["indicators"][name] = {
            "d": d.astype("int8"),
            "r": r.astype("float32"),
            "gross": gross.astype("float32"),
            "net": net.astype("float32"),
            "hit": hit,
            "two_day": two_day,
            "current_direction": float(values[-1]) if np.isfinite(values[-1]) else None,
            "error": (errors or {}).get(name),
        }
    return payload
