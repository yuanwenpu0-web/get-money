"""支撑位判定的可证伪评估口径（诊断性量具对比，非策略 edge 判定）。

核心问题：不同指标给出的「支撑位」哪个更准？把「准」拆成三件可观测的事：
1. 触线后做多的净收益（扣 0.12% 双边成本）是否优于随机价位 / 随机时点；
2. 触线后持有期内是否很少被有效跌破（破位率低）；
3. 触线后能否反弹到目标幅度。

所有支撑位都只用 `level[t-1]`（t-1 收盘时已知）判定 t 根是否触线，禁止未来函数；
前瞻极值一律走反向 rolling（P-01），事件取上升沿且间隔 ≥ H（H7 防重叠计数）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 支撑位定义已上移到 `strategy/support_levels.py`（绑定层出场同样要用），
# 这里只做转出，保证研究与实盘是同一套量具。
from app.strategy.support_levels import (  # noqa: F401
    METHOD_KEYS,
    METHODS,
    MethodSpec,
    _donchian,
    _numeric,
    _zone_mid,
    compute_levels,
)

FEE_RATE = 0.0004
SLIPPAGE = 0.0002
ROUND_TRIP = 2 * (FEE_RATE + SLIPPAGE)  # 0.0012，名义本金口径


def forward_extreme(values: np.ndarray, horizon: int, kind: str = "min") -> np.ndarray:
    """`out[t]` = values[t+1 .. t+H] 的极值；末尾 H 根为 NaN。

    反向 rolling 得 `fwd[t] = extreme[t .. t+H-1]`，再整体左移一根。
    直接写 `shift(-1).rolling(H)` 会得到过去窗口（P-01）。
    """
    series = pd.Series(np.asarray(values, dtype=float))
    rolled = series.iloc[::-1].rolling(horizon, min_periods=horizon)
    fwd = (rolled.min() if kind == "min" else rolled.max()).iloc[::-1].to_numpy(dtype=float)
    return np.concatenate((fwd[1:], [np.nan]))


def touch_mask(level: np.ndarray, close: np.ndarray, low: np.ndarray) -> np.ndarray:
    """t 根触线：前一根收盘在 L[t-1] 上方，且本根最低价触及/跌破 L[t-1]。"""
    prev_level = np.concatenate(([np.nan], np.asarray(level, dtype=float)[:-1]))
    prev_close = np.concatenate(([np.nan], np.asarray(close, dtype=float)[:-1]))
    return (np.isfinite(prev_level) & np.isfinite(prev_close)
            & (prev_close > prev_level) & (np.asarray(low, dtype=float) <= prev_level))


def rising_edges(mask: np.ndarray, min_gap: int) -> np.ndarray:
    edges = np.flatnonzero(mask & ~np.concatenate(([False], mask[:-1])))
    if len(edges) == 0:
        return edges
    kept = [edges[0]]
    for pos in edges[1:]:
        if pos - kept[-1] >= min_gap:
            kept.append(pos)
    return np.asarray(kept, dtype=int)


def event_arrays(frame: pd.DataFrame, level: np.ndarray, horizon: int,
                 cost: float = ROUND_TRIP, break_buffer: float = 0.005,
                 bounce_target: float = 0.01) -> dict[str, np.ndarray]:
    """触线事件的前瞻结果。入场 t+1 开盘、出场 t+H 收盘，与项目其它研究管线一致。"""
    n = len(frame)
    low = frame["low"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    open_ = frame["open"].to_numpy(dtype=float)
    # pandas 3 的 DatetimeIndex 分辨率可能是 us；统一成 ns，避免与 Timestamp.value 混比
    times = frame.index.as_unit("ns").asi8
    edges = rising_edges(touch_mask(level, close, low), horizon)
    edges = edges[(edges + horizon < n) & (edges + 1 < n)]
    if len(edges) == 0:
        return {k: np.array([], dtype=float) for k in
                ("index", "time", "exit_time", "level", "gross", "net", "broke", "bounced")}
    prev_level = np.concatenate(([np.nan], np.asarray(level, dtype=float)[:-1]))[edges]
    entry = open_[edges + 1]
    gross = close[edges + horizon] / entry - 1.0
    future_low = forward_extreme(low, horizon, "min")[edges]
    future_high = forward_extreme(high, horizon, "max")[edges]
    return {
        "index": edges,
        "time": times[edges],
        "exit_time": times[edges + horizon],
        "level": prev_level,
        "gross": gross,
        "net": gross - cost,
        "broke": (future_low < prev_level * (1 - break_buffer)).astype(float),
        "bounced": (future_high / entry - 1.0 > bounce_target).astype(float),
    }


def randomized_baseline(frame: pd.DataFrame, level: np.ndarray, horizon: int,
                        n_seeds: int = 20, base_seed: int = 20261025) -> dict:
    """随机位移基线：把 level 切成时间块，每块内部独立 np.roll 小位移。

    保留「价位尺度 + 覆盖率的时间结构」，只打乱支撑位与具体 K 线的精确对齐。
    不用整段位移：趋势行情里整段位移会把早期低价搬到后期（或反之），
    价位与价格不可比，事件只剩少数暴涨段，基线净收益虚高（实测 2.3% vs 任意时点 0.22%）。
    direction≡0 的策略随机翻转是 no-op（P-16），故用位移而非翻方向。
    """
    n = len(frame)
    net_means: list[float] = []
    break_rates: list[float] = []
    counts: list[int] = []
    shifts: list[int] = []
    if n < 8:
        return {"net_means": np.array([]), "break_rates": np.array([]),
                "event_counts": np.array([]), "shifts": np.array([])}
    block = max(horizon * 4, n // 8)
    arr = np.asarray(level, dtype=float)
    for offset in range(n_seeds):
        rng = np.random.default_rng(base_seed + offset * 7919)
        rolled = arr.copy()
        local_shift = 0
        for start in range(0, n, block):
            chunk = rolled[start:start + block]
            if len(chunk) < 2:
                continue
            local_shift = int(rng.integers(1, max(2, len(chunk) // 2)))
            rolled[start:start + block] = np.roll(chunk, local_shift)
        ev = event_arrays(frame, rolled, horizon)
        shifts.append(local_shift)
        counts.append(len(ev["net"]))
        net_means.append(float(ev["net"].mean()) if len(ev["net"]) else float("nan"))
        break_rates.append(float(ev["broke"].mean()) if len(ev["broke"]) else float("nan"))
    return {"net_means": np.asarray(net_means), "break_rates": np.asarray(break_rates),
            "event_counts": np.asarray(counts), "shifts": np.asarray(shifts)}


def random_price_baseline(frame: pd.DataFrame, level: np.ndarray, horizon: int,
                          n_seeds: int = 20, base_seed: int = 20261105) -> dict:
    """随机价位基线：保留「哪些 bar 有支撑位」，但把价位换成同期 K 线区间内的随机数。

    与随机位移互补：位移检验「时序对齐有没有信息」，随机价位检验「价位选得好不好」。
    """
    n = len(frame)
    low = frame["low"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    valid = np.isfinite(level)
    net_means: list[float] = []
    break_rates: list[float] = []
    for offset in range(n_seeds):
        rng = np.random.default_rng(base_seed + offset * 7919)
        randomized = np.full(n, np.nan)
        u = rng.uniform(0.0, 1.0, n)
        randomized[valid] = low[valid] + u[valid] * (high[valid] - low[valid])
        ev = event_arrays(frame, randomized, horizon)
        net_means.append(float(ev["net"].mean()) if len(ev["net"]) else float("nan"))
        break_rates.append(float(ev["broke"].mean()) if len(ev["broke"]) else float("nan"))
    return {"net_means": np.asarray(net_means), "break_rates": np.asarray(break_rates)}


def oracle_level(frame: pd.DataFrame, horizon: int) -> np.ndarray:
    """正对照（偷看未来）：把未来窗口最低价当作支撑位，量具必须能检出它。"""
    future_low = forward_extreme(frame["low"].to_numpy(dtype=float), horizon, "min")
    return np.concatenate(([np.nan], future_low[:-1]))


def hold_all_net(frame: pd.DataFrame, horizon: int,
                 cost: float = ROUND_TRIP) -> np.ndarray:
    """任意时点买入并持有 H 根的净收益（随机时点基线 / beta 归因的母体）。"""
    close = frame["close"].to_numpy(dtype=float)
    open_ = frame["open"].to_numpy(dtype=float)
    n = len(frame)
    if n <= horizon + 1:
        return np.array([], dtype=float)
    return close[horizon + 1:] / open_[1:n - horizon] - 1.0 - cost


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(len(p))
    q[order] = np.clip(ranked, 0, 1)
    return q


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 20261027) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def segment_consistency(values: np.ndarray, times: np.ndarray, segments: int = 4) -> float:
    """按时间等分后段均值与总体同号的比例（H7 要求 ≥3/4）。"""
    order = np.argsort(np.asarray(times, dtype=float))
    arr = np.asarray(values, dtype=float)[order]
    if len(arr) < segments * 3:
        return float("nan")
    overall = np.sign(arr.mean())
    edges = np.linspace(0, len(arr), segments + 1).astype(int)
    same = valid = 0
    for i in range(segments):
        chunk = arr[edges[i]:edges[i + 1]]
        if len(chunk) == 0:
            continue
        valid += 1
        same += int(np.sign(chunk.mean()) == overall)
    return same / valid if valid else float("nan")


def percentile_rank(value: float, samples: list[float] | np.ndarray) -> float:
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float((arr < value).mean()) if len(arr) else 0.0
