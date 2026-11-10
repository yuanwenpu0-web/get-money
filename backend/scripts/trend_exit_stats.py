"""趋势反转出场课题的统计判定：Gate 1 条件化 IC 四关、配对 bootstrap、FDR、随机基线。

判定词只来自 strategy-backtest-playbook 第 3 节；本模块只产出数字。
"""
from __future__ import annotations

import math

import numpy as np

from exit_metalabel_stats import spearman
from order_exit_common import bh_fdr, verdict

MIN_POOLED_TRADES = 30
IC_GATE = 0.02
SIGN_GATE = 0.60
Q_GATE = 0.10
BOOT_SEED = 20261027


def boot_ci(values, n_boot: int = 2000, seed: int = BOOT_SEED) -> list[float]:
    """按「一笔交易」重采样；循环版避免 n×n_boot 大数组撑爆内存。"""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    n = len(v)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = v[rng.integers(0, n, size=n)].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired_stats(diffs, n_boot: int = 2000, min_trades: int = MIN_POOLED_TRADES) -> dict:
    v = np.asarray(diffs, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0, "mean_pct": None, "ci_low_pct": None,
                "ci_high_pct": None, "verdict": "no_trades"}
    ci = boot_ci(v, n_boot=n_boot)
    return {"n": int(len(v)), "mean_pct": round(float(v.mean()), 6),
            "ci_low_pct": round(ci[0], 6), "ci_high_pct": round(ci[1], 6),
            "verdict": verdict(ci, len(v), min_trades)}


def segment_means(diffs, times, n_segments: int = 4) -> dict:
    d = np.asarray(diffs, dtype=float)
    t = np.asarray(times, dtype=float)
    ok = np.isfinite(d) & np.isfinite(t)
    order = np.argsort(t[ok], kind="mergesort")
    vals = d[ok][order]
    if len(vals) < n_segments:
        return {"means": [], "n_positive": 0, "n_negative": 0}
    means = [float(x.mean()) for x in np.array_split(vals, n_segments)]
    return {"means": [round(x, 6) for x in means],
            "n_positive": sum(1 for x in means if x > 0),
            "n_negative": sum(1 for x in means if x < 0)}


def percentile_rank(value: float, samples) -> float:
    a = np.asarray(samples, dtype=float)
    return float((a < value).mean())


def _binom_two_sided(k: int, n: int) -> float:
    """精确二项双侧 p（H0: 同号概率 0.5），无 scipy 依赖。"""
    if n <= 0:
        return 1.0
    from math import comb
    probs = [comb(n, i) / 2**n for i in range(n + 1)]
    pk = probs[k]
    return float(min(1.0, sum(p for p in probs if p <= pk + 1e-15)))


def by_symbol_ic(x, y, symbols, min_pairs: int = 10) -> list[tuple[str, float]]:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    out = []
    sa = np.asarray(symbols)
    for sym in sorted(set(symbols)):
        m = ok & (sa == sym)
        if int(m.sum()) >= min_pairs:
            out.append((sym, spearman(a[m], b[m], min_pairs=min_pairs)))
    return out


def self_calibration(y, rng, n_seeds: int = 20) -> dict:
    """正对照（偷看未来收益）／负对照（随机噪声）／半衰对照。"""
    v = np.asarray(y, dtype=float)
    ok = np.isfinite(v)
    v = v[ok]
    if len(v) < 30:
        return {"error": "insufficient pairs"}
    pos = spearman(v, v)
    z = (v - v.mean()) / max(v.std(), 1e-12)
    neg, half = [], []
    for _ in range(n_seeds):
        noise = rng.normal(size=len(v))
        neg.append(spearman(noise, v))
        half.append(spearman(0.5 * z + 0.5 * noise, v))
    neg = np.array([x for x in neg if math.isfinite(x)])
    half = np.array([x for x in half if math.isfinite(x)])
    return {"positive_control_ic_LOOKAHEAD": round(float(pos), 6),
            "negative_control_mean_abs_ic": round(float(np.abs(neg).mean()), 6),
            "negative_control_p95_abs_ic": round(float(np.quantile(np.abs(neg), 0.95)), 6),
            "half_decay_mean_ic": round(float(half.mean()), 6),
            "n_seeds": n_seeds, "n_pairs": int(len(v))}


def gate1_probe(records: list[dict], signals, rng) -> dict:
    """条件化 IC 四关：|IC|>=0.02 ∧ 按币同号>=0.60 ∧ FDR q<0.10 ∧ 4 段>=3 同号。"""
    y = np.array([r["fwd_pct"] for r in records], dtype=float)
    t = np.array([r["time_ms"] for r in records], dtype=float)
    syms = [r["symbol"] for r in records]
    rows = []
    for name in signals:
        x = np.array([r["features"].get(name, float("nan")) for r in records], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        pairs = int(ok.sum())
        uniq = int(len(np.unique(x[ok]))) if pairs else 0
        if pairs < 30 or uniq < 2:
            rows.append({"signal": name, "pairs": pairs, "unique_values": uniq,
                         "ic": None, "coverage": round(pairs / max(1, len(records)), 4),
                         "gate": "insufficient", "gates": {}})
            continue
        ic = spearman(x, y)
        per = by_symbol_ic(x, y, syms)
        same = sum(1 for _, v in per if math.isfinite(v) and (v > 0) == (ic > 0))
        share = same / len(per) if per else float("nan")
        p = _binom_two_sided(same, len(per))
        order = np.argsort(t[ok], kind="mergesort")
        xs, ys = x[ok][order], y[ok][order]
        segs = [spearman(a, b, min_pairs=10)
                for a, b in zip(np.array_split(xs, 4), np.array_split(ys, 4))]
        seg_same = max(sum(1 for v in segs if math.isfinite(v) and v > 0),
                       sum(1 for v in segs if math.isfinite(v) and v < 0))
        rows.append({
            "signal": name, "pairs": pairs, "coverage": round(pairs / len(records), 4),
            "ic": None if not math.isfinite(ic) else round(ic, 6),
            "symbols_with_ic": len(per), "same_sign_symbols": same,
            "sign_share": None if not math.isfinite(share) else round(share, 4),
            "p_binom": round(p, 8), "segment_ics": [None if not math.isfinite(v) else round(v, 6)
                                                    for v in segs],
            "segments_same_sign": seg_same,
            "gate": "pending",
        })
    valid = [r for r in rows if r["gate"] == "pending"]
    qs = bh_fdr([r["p_binom"] for r in valid]) if valid else []
    for r, q in zip(valid, qs):
        r["fdr_q"] = round(float(q), 6)
        g = {"abs_ic_ge_0.02": bool(r["ic"] is not None and abs(r["ic"]) >= IC_GATE),
             "sign_share_ge_0.60": bool(r["sign_share"] is not None
                                        and r["sign_share"] >= SIGN_GATE),
             "fdr_q_below_0.10": bool(q < Q_GATE),
             "segments_same_sign_ge_3of4": r["segments_same_sign"] >= 3}
        r["gates"] = g
        r["gate"] = "pass" if all(g.values()) else "fail"
    passed = [r for r in rows if r["gate"] == "pass"]
    for r in rows:
        # H1 的 |IC| 门槛无方向；作为「卖出」信号还要求分值越高、未来收益越低（IC<0）
        r["sell_direction_ok"] = bool(r["ic"] is not None and r["ic"] < 0)
        r["usable_as_sell"] = bool(r["gate"] == "pass" and r["sell_direction_ok"])
    return {"rows": rows, "self_calibration": self_calibration(y, rng),
            "probe_pass_count": len(passed),
            "sell_direction_pass_count": sum(1 for r in rows if r["usable_as_sell"]),
            "verdict": "pass" if any(r["gate"] == "pass" for r in rows) else "inconclusive"}
