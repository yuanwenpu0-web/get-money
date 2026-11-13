"""H-A 仓位开关组合、成本、随机基线与事前终止条件。

工程预置，非 edge 判定。成本名义本金双边 0.12%，精确按单腿 c*(exit/entry) 计。
随机基线用「精确匹配游程结构」的零分布：只重排 ON/OFF 游程长度的分配位置，
逐序列保持曝光与换手完全不变。不用 iid 伯努利也不用循环位移或拟合马尔可夫链，
那些做法会让基线换手偏离真实信号，靠成本差伪造出 alpha（见报告方法学节）。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

FEE_RATE = 0.0004
SLIPPAGE = 0.0002
ROUND_TRIP = 2 * (FEE_RATE + SLIPPAGE)
MIN_OOS_ABS_RHO = 0.10
MIN_BLOCKS = 30
MIN_SEGMENT_SIGN_SHARE = 0.75
RANDOM_P90 = 90


def switch_from_signal(factor: np.ndarray, threshold: float) -> np.ndarray:
    """预登记方向：OI z 高＝杠杆拥挤→空仓；低于阈值才满仓做多市场。缺失→空仓。"""
    f = np.asarray(factor, float)
    return np.isfinite(f) & (f < threshold)


def switch_net_returns(block_ret: np.ndarray, on: np.ndarray,
                       fee: float = FEE_RATE, slippage: float = SLIPPAGE) -> np.ndarray:
    """逐块净收益。仓位在块边界切换，入场腿按单位名义、出场腿按已增长名义计费。

    一个连续持仓段的总成本恰为 c*(1 + exit/entry)，与项目精确成本口径一致。
    """
    ret = np.asarray(block_ret, float)
    pos = np.asarray(on, bool)
    c, n = fee + slippage, len(ret)
    net = np.zeros(n)
    idx, entry, prev = 1.0, 1.0, False
    for i in range(n):
        cur = bool(pos[i])
        if cur != prev:
            if cur:
                net[i] -= c
                entry = idx
            else:
                net[i] -= c * (idx / entry)
            prev = cur
        if cur:
            net[i] += ret[i]
        idx *= 1.0 + ret[i]
    if prev and n:
        net[-1] -= c * (idx / entry)
    return net


def constant_full_returns(block_ret: np.ndarray, fee: float = FEE_RATE,
                          slippage: float = SLIPPAGE) -> np.ndarray:
    """恒定满仓基线：同一函数、同一成本，只有关开关这一个变量不同。"""
    ret = np.asarray(block_ret, float)
    return switch_net_returns(ret, np.ones(len(ret), bool), fee, slippage)


def fit_markov(on: np.ndarray) -> tuple[float, float]:
    """两状态停留概率 (P(ON|ON), P(OFF|OFF))；无观测的状态视为吸收态。"""
    pos = np.asarray(on, bool)
    if len(pos) < 2:
        return 1.0, 1.0
    prev, cur = pos[:-1], pos[1:]
    n_on, n_off = int(prev.sum()), int((~prev).sum())
    p_on = float((cur & prev).sum() / n_on) if n_on else 1.0
    p_off = float((~cur & ~prev).sum() / n_off) if n_off else 1.0
    return p_on, p_off


def _run_lengths(pos: np.ndarray) -> tuple[list[int], list[int], list[bool]]:
    on_lens: list[int] = []
    off_lens: list[int] = []
    pattern: list[bool] = []
    i, n = 0, len(pos)
    while i < n:
        j = i
        while j < n and pos[j] == pos[i]:
            j += 1
        (on_lens if pos[i] else off_lens).append(j - i)
        pattern.append(bool(pos[i]))
        i = j
    return on_lens, off_lens, pattern


def matched_random_switches(on: np.ndarray, n_seeds: int = 200,
                            seed: int = 20261112) -> np.ndarray:
    """精确匹配零分布：只重排 ON/OFF 游程长度的分配位置。

    逐序列保持曝光块数、游程数与换手次数完全不变，只随机化时点。不用 iid 伯努利
    （换手远高于持续信号），也不用拟合马尔可夫链——持续信号的 OFF→ON 转移可能
    观测为 0，会估出吸收态使基线换手系统性偏低，两者都靠成本差伪造 alpha。
    """
    pos = np.asarray(on, bool)
    n = len(pos)
    on_lens, off_lens, pattern = _run_lengths(pos)
    rng = np.random.default_rng(seed)
    out = np.empty((n_seeds, n), dtype=bool)
    for j in range(n_seeds):
        a, b = rng.permutation(on_lens), rng.permutation(off_lens)
        seq = np.empty(n, dtype=bool)
        ia = ib = k = 0
        for state in pattern:
            length = int(a[ia]) if state else int(b[ib])
            if state:
                ia += 1
            else:
                ib += 1
            seq[k:k + length] = state
            k += length
        out[j] = seq
    return out


def paired_bootstrap_ci(x: np.ndarray, n_boot: int = 2000,
                        seed: int = 20261112, block: int = 1) -> dict:
    """配对差序列的 bootstrap CI。块已非重叠，默认按块 iid 重采样。"""
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return {"ci_low": None, "ci_high": None, "n": 0, "mean": None,
                "n_boot": n_boot, "block": block}
    if n < block * 2:
        block = 1
    rng = np.random.default_rng(seed)
    reps = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, reps))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    means = v[idx.reshape(n_boot, -1)[:, :n]].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"ci_low": float(lo), "ci_high": float(hi), "n": n,
            "mean": float(v.mean()), "n_boot": n_boot, "block": block}


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    x, y = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return None
    ra, rb = rankdata(x[m]), rankdata(y[m])
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else None


def percentile_rank(value: float, dist: np.ndarray) -> float | None:
    d = np.asarray(dist, float)
    d = d[np.isfinite(d)]
    return float((d < value).mean()) if len(d) else None


def time_segment_sign_share(x: np.ndarray, n_segments: int = 4) -> float | None:
    """时间分段同号占比（H7）：整体符号与各段均值符号一致的比例。"""
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    if len(v) < n_segments:
        return None
    overall = np.sign(v.mean())
    parts = [p for p in np.array_split(v, n_segments) if len(p)]
    return float(sum(1 for p in parts if np.sign(p.mean()) == overall) / len(parts))


def termination_checks(*, train_rho, oos_rho, switch_net, baseline_net,
                       ci_low, random_p90, n_blocks, segment_sign_share) -> dict:
    """假设卡 §2 的事前终止条件，逐条落地；任一成立即终止，不许改口径去救。"""
    reasons: list[str] = []
    if train_rho is None or oos_rho is None:
        reasons.append("out-of-sample spearman unavailable")
    else:
        if np.sign(oos_rho) != np.sign(train_rho):
            reasons.append("out-of-sample spearman sign flip")
        if abs(oos_rho) < MIN_OOS_ABS_RHO:
            reasons.append(
                f"out-of-sample |rho| {abs(oos_rho):.4f} < {MIN_OOS_ABS_RHO}")
    if switch_net <= baseline_net:
        reasons.append("switch net <= constant-full baseline")
    if ci_low is None or ci_low <= 0:
        reasons.append("paired bootstrap ci lower bound <= 0")
    if random_p90 is not None and switch_net < random_p90:
        reasons.append("switch net below random-switch p90")
    if n_blocks < MIN_BLOCKS:
        reasons.append(f"non-overlapping blocks {n_blocks} < {MIN_BLOCKS}")
    if segment_sign_share is None or segment_sign_share < MIN_SEGMENT_SIGN_SHARE:
        reasons.append("time-segment same-sign share < 3/4")
    return {
        "terminated": bool(reasons),
        "reasons": reasons,
        "thresholds": {"min_oos_abs_rho": MIN_OOS_ABS_RHO,
                       "min_blocks": MIN_BLOCKS,
                       "min_segment_sign_share": MIN_SEGMENT_SIGN_SHARE,
                       "random_percentile": RANDOM_P90},
    }