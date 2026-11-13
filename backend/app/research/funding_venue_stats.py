"""H-F Gate 1a 统计层：逐期毛费率、奖金占比、横截面 IC、年化与 D1–D3 判据。

**只算毛费率收益，不算净收益。** 他所现货费率/深度未知，任何净数字都是编造
（假设卡 §5、§6）。H3 的全局成本常量在此仅作参照记录，不参与计算。

毛收益口径 = 空永续收到的资金费 / 名义本金，假设现货腿完全对冲 delta，
因此不含价格损益，也不含保证金占用与清算风险。
"""
from __future__ import annotations

import math

import numpy as np

from app.research.funding_venue import DAY_MS, select_top_k, window_stats
from app.research.trend_direction_report import spearman_ic

LOOKBACK_DAYS = 7
HOLD_DAYS = 3
KS = (10, 20)
MIN_CROSS_SECTION = 30

D1_MIN_SHARE = 0.60
D2_MIN_IC = 0.50
D2_MIN_POSITIVE_SHARE = 0.90
D3_MIN_ANNUAL_PP = 25.0


def _finite(value) -> bool:
    return value is not None and math.isfinite(value)


def period_snapshot(book: dict, t: int, *, lookback_days: int = LOOKBACK_DAYS,
                    hold_days: int = HOLD_DAYS) -> dict:
    """一个重平衡时点上全宇宙的回看均费率与前瞻费率和（按实际结算事件）。"""
    lo, hi = t - lookback_days * DAY_MS, t + hold_days * DAY_MS
    trailing, forward, events = {}, {}, {}
    for symbol, (times, rates) in book.items():
        trailing[symbol] = window_stats(times, rates, lo, t)[0]
        _, total, count = window_stats(times, rates, t, hi)
        forward[symbol], events[symbol] = total, count
    return {"t": t, "trailing": trailing, "forward": forward,
            "forward_events": events}


def gross_period_return(trailing: dict, forward: dict, universe, k: int) -> dict:
    """在给定宇宙内选 TopK，等权平均其前瞻费率和；缺前瞻的币剔除并列出。"""
    selected = select_top_k({s: trailing.get(s) for s in universe}, k)
    missing = [s for s in selected if forward.get(s) is None]
    usable = [forward[s] for s in selected if forward.get(s) is not None]
    return {"selected": selected, "missing_forward": missing, "n": len(usable),
            "return": float(np.mean(usable)) if usable else None}


def bonus_share(selected, forward: dict, non_hedgeable) -> dict:
    """不可对冲币贡献的前瞻费率占该期 TopK 总费率的比例（分母 ≤0 时无意义）。"""
    missing = set(non_hedgeable)
    picked = [s for s in selected if forward.get(s) is not None]
    total = float(sum(forward[s] for s in picked))
    part = float(sum(forward[s] for s in picked if s in missing))
    return {"total": total, "non_hedgeable": part,
            "share": part / total if total > 0 else None,
            "n_selected": len(selected),
            "n_non_hedgeable": sum(1 for s in selected if s in missing)}


def cross_section_ic(trailing: dict, forward: dict,
                     universe) -> tuple[float | None, int]:
    """宇宙内 trailing→forward 的 Spearman 秩 IC；有效币数 <30 不给读数。"""
    pairs = [(trailing[s], forward[s]) for s in universe
             if _finite(trailing.get(s)) and _finite(forward.get(s))]
    if len(pairs) < MIN_CROSS_SECTION:
        return None, len(pairs)
    factor = np.array([p[0] for p in pairs], dtype="float64")
    label = np.array([p[1] for p in pairs], dtype="float64")
    return spearman_ic(factor, label), len(pairs)


def summarize_ic(ics) -> dict:
    """IC 序列的均值、正期占比与跳过期数（None 期是横截面不足，不是 0）。"""
    values = [v for v in ics if _finite(v)]
    skipped = len(ics) - len(values)
    if not values:
        return {"mean": None, "positive_share": None, "n_periods": 0,
                "n_skipped": skipped}
    return {"mean": float(np.mean(values)),
            "positive_share": sum(1 for v in values if v > 0) / len(values),
            "n_periods": len(values), "n_skipped": skipped}


def annualize_gross(returns, *, hold_days: int = HOLD_DAYS) -> float | None:
    """按**实际覆盖天数**复利年化；None 期剔除后天数同步缩短，不假装满窗。"""
    values = [r for r in returns if _finite(r)]
    if not values:
        return None
    growth = math.prod(1.0 + r for r in values)
    if growth <= 0:
        return None
    return float(growth ** (365.0 / (len(values) * hold_days)) - 1.0)


def gate_checks(share, ic: dict, d3_pp) -> dict:
    """D1–D3 事前门槛；任一缺读数或不达标即 gate 不通过。"""
    d1 = _finite(share) and share >= D1_MIN_SHARE
    d2 = (_finite(ic.get("mean")) and ic["mean"] >= D2_MIN_IC
          and _finite(ic.get("positive_share"))
          and ic["positive_share"] >= D2_MIN_POSITIVE_SHARE)
    d3 = _finite(d3_pp) and d3_pp >= D3_MIN_ANNUAL_PP
    return {"D1": bool(d1), "D2": bool(d2), "D3": bool(d3),
            "pass": bool(d1 and d2 and d3)}


BOOT_BLOCK = 8
BOOT_DRAWS = 2000
BOOT_SEED = 20261112
BASELINE_SEEDS = 20


def _clean(values) -> np.ndarray:
    return np.array([v for v in values if _finite(v)], dtype="float64")


def _block_indices(n: int, block: int, draws: int, seed: int) -> np.ndarray:
    """移动块重采样索引。费率强持续（IC≈0.75），iid bootstrap 会低估方差。"""
    rng = np.random.default_rng(seed)
    reps = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(draws, reps))
    idx = starts[:, :, None] + np.arange(block)[None, None, :]
    return idx.reshape(draws, -1)[:, :n]


def block_bootstrap_ci(values, *, block: int = BOOT_BLOCK, n_boot: int = BOOT_DRAWS,
                       seed: int = BOOT_SEED) -> dict:
    """期均值的移动块 bootstrap 95% CI；不足两个整块不给读数。"""
    x = _clean(values)
    meta = {"n": int(x.size), "block": block, "n_boot": n_boot, "seed": seed,
            "method": f"moving_block{block}"}
    if x.size < block * 2:
        return {"ci_low": None, "ci_high": None,
                "mean": float(x.mean()) if x.size else None, **meta}
    means = x[_block_indices(x.size, block, n_boot, seed)].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"ci_low": float(low), "ci_high": float(high),
            "mean": float(x.mean()), **meta}


def paired_increment_ci(full, hedge, *, block: int = BOOT_BLOCK,
                        n_boot: int = BOOT_DRAWS, seed: int = BOOT_SEED,
                        hold_days: int = HOLD_DAYS) -> dict:
    """D3 增量的配对块 bootstrap CI：同一组重采样索引同时作用于两臂。

    估计量是「配对期收益均值年化后作差」，与点估计的逐期复利口径略有差异，
    只用于判断增量是否稳健为正，不与点估计逐位对齐。
    """
    pairs = [(f, h) for f, h in zip(full, hedge) if _finite(f) and _finite(h)]
    meta = {"n_pairs": len(pairs), "block": block, "n_boot": n_boot, "seed": seed,
            "estimator": "mean_period_annualized_difference"}
    if len(pairs) < block * 2:
        return {"ci_low_pp": None, "ci_high_pp": None, "mean_pp": None, **meta}
    a = np.array([p[0] for p in pairs], dtype="float64")
    b = np.array([p[1] for p in pairs], dtype="float64")
    idx = _block_indices(len(pairs), block, n_boot, seed)
    exponent = 365.0 / hold_days
    diff = ((1.0 + a[idx].mean(axis=1)) ** exponent
            - (1.0 + b[idx].mean(axis=1)) ** exponent) * 100.0
    low, high = np.percentile(diff, [2.5, 97.5])
    point = ((1.0 + a.mean()) ** exponent - (1.0 + b.mean()) ** exponent) * 100.0
    return {"ci_low_pp": float(low), "ci_high_pp": float(high),
            "mean_pp": float(point), **meta}


def pooled_share_ci(bonus, *, block: int = BOOT_BLOCK, n_boot: int = BOOT_DRAWS,
                    seed: int = BOOT_SEED) -> dict:
    """D1 池化占比的块 bootstrap CI：重采样期后重新求 Σ部分/Σ全体。"""
    totals = np.array([b["total"] for b in bonus], dtype="float64")
    parts = np.array([b["non_hedgeable"] for b in bonus], dtype="float64")
    meta = {"n": len(bonus), "block": block, "n_boot": n_boot, "seed": seed}
    if len(bonus) < block * 2:
        return {"ci_low": None, "ci_high": None, **meta}
    idx = _block_indices(len(bonus), block, n_boot, seed)
    num, den = parts[idx].sum(axis=1), totals[idx].sum(axis=1)
    ok = den > 0
    if not ok.any():
        return {"ci_low": None, "ci_high": None, **meta}
    low, high = np.percentile(num[ok] / den[ok], [2.5, 97.5])
    return {"ci_low": float(low), "ci_high": float(high), **meta}


def random_baseline(eligible_per_period, forward_per_period, k: int, *,
                    n_seeds: int = BASELINE_SEEDS, seed: int = BOOT_SEED,
                    hold_days: int = HOLD_DAYS) -> dict:
    """同期同 K **随机选币**基线（只从 trailing>0 的合格币里随机）。

    本策略不预测价格方向，H6 的随机方向基线在此无意义；对照必须是随机挑币，
    才能把「排序有用」与「随便拿正费率币就有钱」分开。
    """
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_seeds):
        returns = []
        for eligible, forward in zip(eligible_per_period, forward_per_period):
            pool = [s for s in eligible if _finite(forward.get(s))]
            if not pool:
                returns.append(None)
                continue
            order = rng.permutation(len(pool))[:min(k, len(pool))]
            returns.append(float(np.mean([forward[pool[i]] for i in order])))
        samples.append(annualize_gross(returns, hold_days=hold_days))
    pct = [None if v is None else round(v * 100, 4) for v in samples]
    out = {"k": k, "n_seeds": n_seeds, "seed": seed, "samples_pct": pct,
           "mean_pct": None, "p05_pct": None, "median_pct": None, "p95_pct": None}
    finite = [v for v in pct if v is not None]
    if finite:
        arr = np.array(finite, dtype="float64")
        out.update(mean_pct=round(float(arr.mean()), 4),
                   p05_pct=round(float(np.percentile(arr, 5)), 4),
                   median_pct=round(float(np.median(arr)), 4),
                   p95_pct=round(float(np.percentile(arr, 95)), 4))
    return out