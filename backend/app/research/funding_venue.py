"""H-F Gate 1a 纯函数：现货腿可得性、重平衡时点、窗口费率、选币。

口径由 `research/hypotheses/H-F-funding-multi-venue.md` §5 事前冻结：
7 天回看选币、3 天持有、trailing 均费率 > 0 才可入选。

费率结算周期**逐币不同**（4h/8h 混合），窗口统计一律按实际 `funding_time`
计数，不得假设每天 3 次。
"""
from __future__ import annotations

import math

import numpy as np

from app.market.metrics import base_asset

DAY_MS = 86_400_000
MULTIPLIER_PREFIXES = ("1000000", "100000", "10000", "1000")


def spot_candidates(perp_symbol: str) -> set[str]:
    """永续符号可能对应的现货基础资产：原名 + 剥掉一层 1000X 倍数前缀。

    币安现货用 `1000CATUSDT`、永续也用 `1000CATUSDT` 的情况存在，故两者都留；
    `1000BONKUSDT` 永续对应现货 `BONKUSDT`，靠剥前缀命中。
    """
    base = base_asset(perp_symbol)
    out = {base}
    for prefix in MULTIPLIER_PREFIXES:
        if base.startswith(prefix) and len(base) > len(prefix):
            out.add(base[len(prefix):])
            break
    return out


def hedgeable_split(perp_symbols, spot_bases) -> dict[str, list[str]]:
    """按「币安现货是否有同基础资产在架」把永续宇宙切成两半，各自排序去重。"""
    bases = {str(b).upper() for b in spot_bases}
    hedgeable, missing = [], []
    for symbol in sorted({str(s).upper() for s in perp_symbols}):
        (hedgeable if spot_candidates(symbol) & bases else missing).append(symbol)
    return {"hedgeable": hedgeable, "non_hedgeable": missing}


def rebalance_times(start_ms: int, end_ms: int, *, lookback_days: int = 7,
                    hold_days: int = 3) -> list[int]:
    """重平衡时点 T：留足回看窗，且每期前瞻窗完整落在样本内。"""
    step = hold_days * DAY_MS
    times, t = [], int(start_ms) + lookback_days * DAY_MS
    while t + step <= int(end_ms):
        times.append(t)
        t += step
    return times


def window_stats(times: np.ndarray, rates: np.ndarray, lo: int,
                 hi: int) -> tuple[float | None, float | None, int]:
    """`[lo, hi)` 内实际结算事件的 (均费率, 费率和, 事件数)；空窗返回 None。

    None 与 0.0 必须区分：无事件 = 数据缺失，不是「费率为零」。
    """
    i = int(np.searchsorted(times, lo, side="left"))
    j = int(np.searchsorted(times, hi, side="left"))
    if j <= i:
        return None, None, 0
    chunk = rates[i:j]
    return float(chunk.mean()), float(chunk.sum()), int(chunk.size)


def select_top_k(trailing: dict[str, float | None], k: int) -> list[str]:
    """trailing 均费率降序取前 k，只收 > 0；同值按符号升序，保证可复现。"""
    if k <= 0:
        return []
    eligible = [(s, v) for s, v in trailing.items()
                if v is not None and math.isfinite(v) and v > 0]
    eligible.sort(key=lambda kv: (-kv[1], kv[0]))
    return [s for s, _ in eligible[:k]]