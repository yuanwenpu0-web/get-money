"""每个 tier 的出入场/仓位参数基线，以及围绕基线的坐标下降寻优网格。

为什么分档要改这些参数
----------------------
* **规模越小 → 噪音越大**：小盘币的 atr_pct 中位数约为 mega 的 3 倍，同样的
  1.5×ATR 止损在大盘只是正常波动，在小盘会被随机噪音反复扫损。故小档
  sl_atr_mult 更大（止损放宽），同时 min_score 更高（只做更干净的信号）。
* **规模越小 → 成本占比越高**：滑点与冲击成本随流动性下降而上升，
  需要更高的 tp_atr_mult（盈亏比）才值得入场。
* **规模越小 → 容量约束越强**：max_position_pct 逐档下调，回测不能假设
  可以无限建仓。
* **vol 维度（档内二次修正）**：volatile 组止损更宽、max_atr_pct 上限更高
  （否则整档会被 excess_volatility 全拦），且 max_hold_bars 更短——
  高波动下风险累积更快。

所有 key 必须是 strategy/rules.py 中 SHARED_KEYS / RULE_DEFAULTS / STRATEGY_PARAMS
已存在的合法键，且取值落在 resolve_params 的 ranges 内，否则会抛 ValueError。
"""

from __future__ import annotations

from typing import Iterable

# tier -> 参数覆盖项（未列出的键沿用全局 state 默认值）
TIER_PARAM_PROFILES: dict[str, dict[str, float]] = {
    # —— mega：BTC/ETH 级别，流动性最深、噪音最低，可承受最紧止损与最长持有 ——
    "mega_stable": {
        "min_score": 0.56, "sl_atr_mult": 1.4, "tp_atr_mult": 2.4,
        "max_hold_bars": 160, "cooldown_bars": 8,
        "base_position_pct": 0.09, "max_position_pct": 0.25, "max_atr_pct": 0.030,
    },
    "mega_volatile": {
        "min_score": 0.57, "sl_atr_mult": 1.8, "tp_atr_mult": 2.8,
        "max_hold_bars": 140, "cooldown_bars": 8,
        "base_position_pct": 0.08, "max_position_pct": 0.22, "max_atr_pct": 0.045,
    },
    # —— large：主流大盘，趋势性较强，接近全局基线 ——
    "large_stable": {
        "min_score": 0.57, "sl_atr_mult": 1.5, "tp_atr_mult": 2.5,
        "max_hold_bars": 140, "cooldown_bars": 10,
        "base_position_pct": 0.08, "max_position_pct": 0.22, "max_atr_pct": 0.035,
    },
    "large_volatile": {
        "min_score": 0.58, "sl_atr_mult": 1.9, "tp_atr_mult": 2.9,
        "max_hold_bars": 120, "cooldown_bars": 10,
        "base_position_pct": 0.07, "max_position_pct": 0.20, "max_atr_pct": 0.050,
    },
    # —— mid：噪音中等，需更宽止损与更高信号门槛 ——
    "mid_stable": {
        "min_score": 0.58, "sl_atr_mult": 1.7, "tp_atr_mult": 2.7,
        "max_hold_bars": 120, "cooldown_bars": 12,
        "base_position_pct": 0.060, "max_position_pct": 0.18, "max_atr_pct": 0.040,
    },
    "mid_volatile": {
        "min_score": 0.60, "sl_atr_mult": 2.1, "tp_atr_mult": 3.2,
        "max_hold_bars": 100, "cooldown_bars": 12,
        "base_position_pct": 0.055, "max_position_pct": 0.16, "max_atr_pct": 0.055,
    },
    # —— small：流动性差、噪音最大；止损最宽、盈亏比要求最高、仓位最小、持有最短 ——
    "small_stable": {
        "min_score": 0.60, "sl_atr_mult": 1.9, "tp_atr_mult": 3.0,
        "max_hold_bars": 100, "cooldown_bars": 14,
        "base_position_pct": 0.050, "max_position_pct": 0.14, "max_atr_pct": 0.045,
    },
    "small_volatile": {
        "min_score": 0.62, "sl_atr_mult": 2.3, "tp_atr_mult": 3.5,
        "max_hold_bars": 80, "cooldown_bars": 14,
        "base_position_pct": 0.045, "max_position_pct": 0.12, "max_atr_pct": 0.060,
    },
}

# 第一轮同时调 (min_score, sl_atr_mult) 这两个影响最大的入场/出场开关，
# 再用结果调 (tp_atr_mult, max_hold_bars)。相比 4 维全网格 81 组，降到 18 组，
# 在 walk-forward × 8 档的场景下才跑得动。
OPTIMIZE_ROUNDS: tuple[tuple[str, ...], ...] = (
    ("min_score", "sl_atr_mult"),
    ("tp_atr_mult", "max_hold_bars"),
)

# 每个维度相对基线的候选倍率/增量
SCORE_OFFSETS = (0.0, 0.02, 0.04)
SL_MULTIPLIERS = (0.85, 1.0, 1.15)
TP_MULTIPLIERS = (0.85, 1.0, 1.2)
HOLD_MULTIPLIERS = (0.7, 1.0, 1.3)

SCORE_CEILING = 0.95          # min_score 上限，避免过滤到几乎无成交
MAX_HOLD_BARS_CAP = 400
HEAVY_SCORE_GAP = 0.15        # heavy_score 至少比 min_score 高多少


def tier_profile(tier: str) -> dict[str, float]:
    """取某档参数；未知 tier 返回空 dict（退化为全局参数）。"""
    return dict(TIER_PARAM_PROFILES.get(tier, {}))


def with_heavy_score(params: dict[str, float]) -> dict[str, float]:
    """保证 heavy_score > min_score，否则仓位映射区间退化（span 被 clamp 到 1e-6）。"""
    result = dict(params)
    min_score = result.get("min_score")
    if min_score is not None:
        heavy = result.get("heavy_score")
        if heavy is None or heavy < min_score + HEAVY_SCORE_GAP:
            result["heavy_score"] = round(min(min_score + HEAVY_SCORE_GAP, 0.99), 4)
    return result


def tier_params(
    base_params: dict, tier: str, overrides: dict | None = None
) -> dict:
    """把档位 profile 叠加到已 resolve 的全局参数上。

    base_params 必须是 resolve_params 输出（含全部合法键）；tier profile 只覆盖
    其中列出的键，其余保持全局值。fee_rate / slippage 永不覆盖（撮合成本全局统一）。
    """
    merged = dict(base_params)
    profile = tier_profile(tier)
    profile.pop("fee_rate", None)
    profile.pop("slippage", None)
    merged.update(profile)
    if overrides:
        for key in ("fee_rate", "slippage"):
            overrides.pop(key, None)
        merged.update(overrides)
    merged = with_heavy_score(merged)
    if merged["base_position_pct"] > merged["max_position_pct"]:
        merged["base_position_pct"] = merged["max_position_pct"]
    if merged["max_hold_bars"] < merged["min_hold_bars"]:
        merged["max_hold_bars"] = merged["min_hold_bars"]
    return merged


def candidate_values(key: str, base: dict) -> list[float]:
    """生成某维度的一组候选值，围绕基线 ±50% 范围内的抖动，并做合法性裁剪。"""
    current = float(base[key])
    if key == "min_score":
        values = [min(current + offset, SCORE_CEILING) for offset in SCORE_OFFSETS]
    elif key == "sl_atr_mult":
        values = [current * m for m in SL_MULTIPLIERS]
    elif key == "tp_atr_mult":
        values = [current * m for m in TP_MULTIPLIERS]
    elif key == "max_hold_bars":
        values = [int(round(current * m)) for m in HOLD_MULTIPLIERS]
    else:
        values = [current]
    unique: list[float] = []
    for value in values:
        value = _clamp(key, value)
        if value not in unique:
            unique.append(value)
    return unique


_BOUNDS = {
    "min_score": (0.5, 0.95),
    "sl_atr_mult": (0.1, 10.0),
    "tp_atr_mult": (0.2, 20.0),
    "max_hold_bars": (30, MAX_HOLD_BARS_CAP),
}


def _clamp(key: str, value: float) -> float:
    low, high = _BOUNDS[key]
    value = max(low, min(high, value))
    return float(int(value)) if key.endswith("_bars") else round(value, 4)


def round_combos(round_keys: tuple[str, ...], base: dict) -> list[list[tuple[str, float]]]:
    """为一轮坐标下降生成候选变更组合。

    必须每轮都用**最新**的 base 重新生成：若提前一次性算好（或用惰性生成器复用），
    第二轮会仍基于首轮更新前的旧基线，导致搜索方向错位。
    """
    axis = [candidate_values(key, base) for key in round_keys]
    combos: list[list[tuple[str, float]]] = [[]]
    for key, values in zip(round_keys, axis):
        combos = [combo + [(key, value)] for combo in combos for value in values]
    return combos
