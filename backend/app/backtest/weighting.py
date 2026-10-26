"""指标加权：动态仓位权重 + regime 自适应策略权重（迭代 8）。

默认关闭（weight_mode=equal）→ 与等权完全一致，现有行为零变更。
所有指标必须经 metrics_as_of 取数（反前视硬红线），本模块只做纯计算，不碰 DB。

仓位公式（写进报告）：
    score = 0.5*tanh(sentiment_divergence)
          - 0.3*min(|crowd_ls_z|, 3)/3
          + 0.2*rank_score(market_cap_rank)
语义：聪明钱相对散户更偏多 → 加分；散户仓位极端拥挤 → 减分（拥挤即反向风险）；
市值靠前 → 加分（流动性与容量）。score → softmax(T=0.5) 保证非负，
再归一化 sum=1，单币上限 clamp 到 2/n 防集中。任一币种指标缺失 → 整体退回等权。

regime 公式：w_trend = clamp(0.5 + A*(fng-50)/50 - B*min(|crowd_ls_z|,3)/3, 0.2, 0.8)
贪婪(fng>50)偏趋势，散户拥挤偏均值回归（逆向）；两者和为 1。
"""

from __future__ import annotations

import math

import pandas as pd

SOFTMAX_TEMPERATURE = 0.5   # score → 权重 的 softmax 温度（越小越集中）
MAX_WEIGHT_MULTIPLIER = 2.0  # 单币权重上限 = 2/n
RANK_DECAY = 50.0            # 市值排名衰减到 0 所需名次跨度
REGIME_A = 0.25              # fng 项系数
REGIME_B = 0.15              # 散户拥挤项系数
REGIME_MIN, REGIME_MAX = 0.2, 0.8


def rank_score(market_cap_rank) -> float:
    """市值排名 → [0,1]：第 1 名 1.0，线性衰减到第 51 名为 0。
    排名未知（0/缺失）→ 中性 0.5，避免把无数据币种判死刑。"""
    rank = int(market_cap_rank or 0)
    if rank <= 0:
        return 0.5
    return max(0.0, min(1.0, 1.0 - (rank - 1) / RANK_DECAY))


def metric_score(metrics: dict) -> float:
    """单币指标分数。非有限值收敛为 0.0。"""
    if not metrics:
        return 0.0
    divergence = _finite(metrics.get("sentiment_divergence"))
    crowd_z = _finite(metrics.get("crowd_ls_z"))
    score = (0.5 * math.tanh(divergence)
             - 0.3 * min(abs(crowd_z), 3.0) / 3.0
             + 0.2 * rank_score(metrics.get("market_cap_rank")))
    return score if math.isfinite(score) else 0.0


def _finite(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _softmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    """softmax（非负）→ 归一 sum=1。"""
    n = len(scores)
    if n == 0:
        return {}
    exponent = {k: math.exp(min(v / SOFTMAX_TEMPERATURE, 30.0)) for k, v in scores.items()}
    total = sum(exponent.values())
    if not total > 0:
        return {k: 1.0 / n for k in scores}
    return {k: v / total for k, v in exponent.items()}


def _clamp_cap(weights: dict[str, float]) -> dict[str, float]:
    """单币上限 clamp 到 2/n：超限者钉在 cap，超出量按剩余币种当前占比重新分配。"""
    n = len(weights)
    if n == 0:
        return {}
    cap = MAX_WEIGHT_MULTIPLIER / n
    weights = dict(weights)
    for _ in range(8):
        over = [k for k, w in weights.items() if w > cap + 1e-12]
        if not over:
            break
        excess = sum(weights[k] - cap for k in over)
        for k in over:
            weights[k] = cap
        free = [k for k in weights if k not in over]
        if free and excess > 0:
            free_total = sum(weights[k] for k in free)
            for k in free:
                share = weights[k] / free_total if free_total > 0 else 1.0 / len(free)
                weights[k] += excess * share
    total = sum(weights.values())
    if total > 0:
        weights = {k: w / total for k, w in weights.items()}
    return weights


def metric_weights(symbols: list[str], metrics_by_symbol: dict) -> dict[str, float]:
    """指标驱动的动态仓位权重：非负、sum=1、单币 <= 2/n。
    任一币种指标缺失（空/None）→ 整体退回等权（不冒险用不完整信息分配资金）。"""
    symbols = [str(s).upper() for s in symbols]
    n = len(symbols)
    if n == 0:
        return {}
    if any(not (metrics_by_symbol.get(s) or {}) for s in symbols):
        return {s: 1.0 / n for s in symbols}
    scores = {s: metric_score(metrics_by_symbol.get(s) or {}) for s in symbols}
    return _clamp_cap(_softmax_normalize(scores))


def regime_weights(metrics: dict) -> dict[str, float]:
    """regime 自适应：返回 {trend_follow: w, mean_reversion: 1-w}，w ∈ [0.2, 0.8]。
    无指标 → 等权 0.5/0.5（"无数据不表态"，与仓位权重回退口径一致）；
    fng 缺失（0 不是真实取值）按中性 50 处理，避免把缺数据误判成极度恐慌。"""
    if not metrics:
        return {"trend_follow": 0.5, "mean_reversion": 0.5}
    fng = _finite(metrics.get("fng_value"))
    if fng <= 0:
        fng = 50.0
    crowd_z = _finite(metrics.get("crowd_ls_z"))
    w = 0.5 + REGIME_A * (fng - 50.0) / 50.0 - REGIME_B * min(abs(crowd_z), 3.0) / 3.0
    if not math.isfinite(w):
        w = 0.5
    w = max(REGIME_MIN, min(REGIME_MAX, w))
    return {"trend_follow": round(w, 6), "mean_reversion": round(1.0 - w, 6)}


def weighted_equity(aligned: pd.DataFrame, initial_cash: float, rebalance_bars: int,
                    weight_lookup) -> pd.Series:
    """按再平衡周期分配资金并汇总组合权益。

    aligned: 列=腿、行=时间，已 ffill 且腿起点前按其初始资金填充
    weight_lookup(ts_ms) -> {列名: 权重}（会自动归一化）或 None（等权）
    每期：期初中权益 × 权重 × 相对该期锚点行的收益指数；期与期之间权益连续。
    """
    columns = list(aligned.columns)
    rows = len(aligned)
    if not columns or rows == 0:
        return pd.Series(dtype=float)
    out = pd.Series(index=aligned.index, dtype=float)
    step = max(int(rebalance_bars), 1)
    starts = list(range(0, rows, step))
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else rows
        anchor = 0 if index == 0 else start - 1
        base = float(initial_cash) if index == 0 else float(out.iloc[anchor])
        anchor_row = aligned.iloc[anchor]
        safe = anchor_row.where(anchor_row > 0)
        relative = aligned.iloc[start:end].divide(safe, axis=1)
        ts_ms = int(aligned.index[start].timestamp() * 1000)
        mapping = weight_lookup(ts_ms) or {c: 1.0 / len(columns) for c in columns}
        weights = pd.Series([_finite(mapping.get(c, 0.0)) for c in columns], index=columns)
        total = float(weights.sum())
        weights = (weights / total) if total > 0 else pd.Series(1.0 / len(columns), index=columns)
        out.iloc[start:end] = base * relative.mul(weights, axis=1).sum(axis=1)
    return out
