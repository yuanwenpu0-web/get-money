"""入场后观察窗的纯函数：聚合、延续判定、绩效指标。

无 DB 访问、无 app 依赖，便于单测与复用。
"""
from __future__ import annotations

import numpy as np

MINUTE = 60_000
OBS_MINUTES = (1, 5, 10)


def obs_bar(minute_rows: list[dict], k: int, start_ms: int | None = None) -> dict | None:
    """前 k 根连续 1m 聚合成一根 k 分钟观察 bar；不连续或缺根返回 None。

    `start_ms` 是该观察窗应有的第一根 open_time：传了就必须对齐，否则入场
    那一分钟缺数据时会把 t0+1..t0+k 误当成 t0..t0+k-1。
    """
    seg = minute_rows[:k]
    if len(seg) < k:
        return None
    t0 = int(seg[0]["open_time"])
    if start_ms is not None and t0 != int(start_ms):
        return None
    for j, r in enumerate(seg):
        if int(r["open_time"]) != t0 + j * MINUTE:
            return None
    return {"open": float(seg[0]["open"]),
            "high": max(float(r["high"]) for r in seg),
            "low": min(float(r["low"]) for r in seg),
            "close": float(seg[-1]["close"])}


def judge(bar: dict, entry: float, sl: float, base_px: float) -> tuple[float, str]:
    """延续 -> 基准出场价；否则提前离场。观察窗内触 SL 优先（最坏情形）。"""
    if bar["low"] <= sl:
        return float(min(bar["open"], sl)), "sl_in_window"
    if bar["close"] > entry:
        return float(base_px), "continuation"
    return float(bar["close"]), "no_continuation"


def max_drawdown(net_pct) -> float:
    """按笔顺序累计净收益曲线的最大峰谷差（与总收益同单位，≤0）。"""
    v = np.asarray(net_pct, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    curve = np.concatenate([[0.0], np.cumsum(v)])   # 初始净值 0 点也算基准
    peak = np.maximum.accumulate(curve)
    return float((curve - peak).min())


def arm_stats(net_pct) -> dict:
    """总收益 / 每笔期望 / 胜率 / 盈亏比 / profit factor / 最大回撤。"""
    v = np.asarray(net_pct, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"trades": 0}
    win, loss = v[v > 0], v[v < 0]
    avg_win = float(win.mean()) if win.size else 0.0
    avg_loss = float(loss.mean()) if loss.size else 0.0
    return {
        "trades": int(v.size),
        "total_return_pct": float(v.sum()),
        "expectancy_pct": float(v.mean()),
        "win_rate": float((v > 0).mean()),
        "payoff_ratio": float(avg_win / abs(avg_loss)) if avg_loss < 0 else float("inf"),
        "profit_factor": float(win.sum() / abs(loss.sum())) if loss.size else float("inf"),
        "avg_win_pct": avg_win, "avg_loss_pct": avg_loss,
        "max_drawdown_pct": max_drawdown(v),
    }


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 20261110,
                 confidence: float = 0.95) -> list[float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    a = (1 - confidence) / 2
    return [float(np.quantile(means, a)), float(np.quantile(means, 1 - a))]


def paired_perm_p(diffs, rng, n: int = 4000) -> float:
    """符号翻转置换检验：H0 为配对差均值 0。"""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 1.0
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, d.size))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.count_nonzero(null >= obs) + 1) / (n + 1))


def bh_fdr(pvals) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out.tolist()


def segment_means(recs, arm: str, n_segments: int = 4) -> list[float]:
    """按时间四分段的配对差均值（H7 稳健性）。"""
    order = sorted(recs, key=lambda r: r["time_ms"])
    out = []
    for idx in np.array_split(np.arange(len(order)), n_segments):
        if not len(idx):
            continue
        d = [order[i]["arms"][arm]["net"] - order[i]["base_net"] for i in idx]
        out.append(float(np.mean(d)))
    return out


def verdict(ci: list[float], n: int, min_trades: int = 30) -> str:
    lo, hi = ci
    if n < min_trades:
        return "insufficient"
    if lo > 0:
        return "pass"
    if hi < 0:
        return "negative"
    return "inconclusive"
