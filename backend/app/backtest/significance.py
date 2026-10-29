"""统计显著性：回答"这个收益是真 edge 还是运气"。

为什么必须独立成模块
----------------------
单看 `total_return_pct` 无法区分 edge 与噪声：400 天样本下 63 笔成交跑出 -9%，
和 60 天样本下 7 笔成交跑出 +1.4%，在没有置信区间的情况下都是"一个数"。
本模块把三个判定量做成一等公民：

1. **bootstrap 每笔期望的置信区间**：对成交的 `return_pct` 做有放回重采样。
   CI 下界 > 0 才算有 edge；跨 0 就是没结论（不是"亏了"，是"不知道"）。
2. **IC（信息系数）**：模型输出概率与未来收益的 Spearman 秩相关。
   直接衡量"模型排的序有没有用"，比 AUC 更贴近组合构建。
3. **多种子随机基线**：单一种子的随机翻转本身也是随机变量，
   必须取分布才敢说"策略跑赢随机"。

只读计算，不碰 DB、不写状态。
"""

from __future__ import annotations

import random

import numpy as np

from .trades import aggregate_positions


def bootstrap_expectancy_ci(
    trades: list[dict],
    n_boot: int = 2000,
    seed: int = 20261027,
    confidence: float = 0.95,
    min_trades: int = 0,
) -> dict:
    """每笔收益率均值的 bootstrap 置信区间。

    重采样的单位是「一笔交易」而非「一根 K 线」：交易之间近似独立，
    而相邻 K 线强自相关，按 bar 重采样会低估方差。

    同时给出账户层面的换算：均值 × 成交笔数 ≈ 该窗口的累计收益贡献，
    便于和 total_return_pct 对照。

    `min_trades` > 0 时样本不足就判 `insufficient`（CI 数值照给，但不作结论）。
    池化必须开这个：实测 4h × 365 天全市场只有 4 笔成交，bootstrap 在 n=4 上
    会给出 [0.86%, 10.48%] 这种「显著为正」的假精确读数，只看 CI 会当成金矿。
    """
    if not trades:
        return {"trades": 0, "verdict": "no_trades"}
    returns = np.array([float(t.get("return_pct", 0.0)) for t in aggregate_positions(trades)], dtype=float)
    returns = returns[np.isfinite(returns)]
    n = len(returns)
    if n == 0:
        return {"trades": 0, "verdict": "no_trades"}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = returns[idx].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    point = float(returns.mean())
    positive_share = float((means > 0).mean())
    if min_trades and n < min_trades:
        verdict = "insufficient"
    else:
        verdict = "positive" if lo > 0 else "negative" if hi < 0 else "inconclusive"
    return {
        "trades": n,
        "mean_return_pct": round(point, 4),
        "ci_low_pct": round(float(lo), 4),
        "ci_high_pct": round(float(hi), 4),
        "confidence": confidence,
        "n_boot": n_boot,
        # 正的比例：0.5 表示完全没把握，>0.95 才算统计上站得住
        "positive_share": round(positive_share, 4),
        "verdict": verdict,
    }


def information_coefficient(
    probability, future_return, min_pairs: int = 30
) -> dict:
    """Spearman 秩相关：模型给出的排序与真实收益排序是否一致。

    IC 比 AUC 更适合这里：组合构建关心的是"把币按预测排序后做多做空"，
    而不是"分类对不对"。|IC| > 0.02 在 crypto 1h 上才有实用价值。
    """
    prob = np.asarray(probability, dtype=float)
    fut = np.asarray(future_return, dtype=float)
    mask = np.isfinite(prob) & np.isfinite(fut)
    prob, fut = prob[mask], fut[mask]
    n = len(prob)
    if n < min_pairs:
        return {"pairs": n, "ic": None, "verdict": "insufficient_pairs"}

    def rank(values: np.ndarray) -> np.ndarray:
        order = values.argsort(kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(1, len(values) + 1, dtype=float)
        # 并列取平均秩，避免打散相等值造成虚假相关
        unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks)
        return (sums / counts)[inverse]

    rp, rf = rank(prob), rank(fut)
    rp -= rp.mean()
    rf -= rf.mean()
    denom = float(np.sqrt((rp ** 2).sum() * (rf ** 2).sum()))
    ic = float((rp * rf).sum() / denom) if denom > 0 else 0.0
    return {
        "pairs": n,
        "ic": round(ic, 5),
        "verdict": "usable" if abs(ic) >= 0.02 else "noise",
    }


def random_baseline_distribution(
    data, signals, params, initial, fee, slippage, interval,
    n_seeds: int = 20, base_seed: int = 20261025,
) -> dict:
    """多随机种子的方向翻转基线分布。

    单一种子的随机翻转本身是个随机变量——只比一次，赢了也可能是运气。
    这里跑 n_seeds 个种子取分布，用策略收益在其中的分位数来判定。
    """
    from .execution import randomize_directions, simulate
    from .metrics import metrics_from_equity

    returns = []
    for offset in range(n_seeds):
        randomized = randomize_directions(signals, seed=base_seed + offset * 7919)
        equity, _ = simulate(data, randomized, params, initial, fee, slippage, interval)
        returns.append(float(equity.iloc[-1] / initial - 1) * 100)
    array = np.asarray(returns, dtype=float)
    return {
        "n_seeds": n_seeds,
        "mean_pct": round(float(array.mean()), 4),
        "std_pct": round(float(array.std(ddof=1)), 4) if n_seeds > 1 else 0.0,
        "min_pct": round(float(array.min()), 4),
        "max_pct": round(float(array.max()), 4),
        "p05_pct": round(float(np.percentile(array, 5)), 4),
        "p95_pct": round(float(np.percentile(array, 95)), 4),
    }


def percentile_rank(value: float, samples: list[float]) -> float:
    """策略收益在随机基线分布中的分位数（0~1）。>0.95 才算显著优于随机。"""
    if not samples:
        return 0.0
    array = np.asarray(samples, dtype=float)
    return float((array < value).mean())
