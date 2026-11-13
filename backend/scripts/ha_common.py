"""P13 / H-A 市场级杠杆周期择时量具：因子、等权市场标签、非重叠块与切分。

工程预置，非 edge 判定。预登记见 research/hypotheses/H-A-market-leverage-timing.md。
与 P12 的两处口径差异是刻意的：标签用等权全市场收益（P12 用横截面中位去 beta），
评估只在非重叠 24h 块上做（P12 的逐小时重叠标签 lag1 自相关 0.94，n 虚高）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
HORIZON = 24
ROLL_WINDOW = 168
ROLL_MIN_PERIODS = 72
MIN_CROSS_SECTION = 30
MIN_SPAN_DAYS = 60
TRAIN_DAYS = 40
OOS_DAYS = 20
MIN_BLOCKS = 30


def data_gate(start_ms: int, end_ms: int) -> dict:
    """硬门禁：覆盖不足 60 天一律拒绝计算，只报剩余天数与到期时刻。"""
    span = (int(end_ms) - int(start_ms)) / DAY_MS
    ready = span >= MIN_SPAN_DAYS
    return {
        "ready": bool(ready),
        "span_days": round(span, 4),
        "days_remaining": round(max(0.0, MIN_SPAN_DAYS - span), 4),
        "eta_ms": int(start_ms) + MIN_SPAN_DAYS * DAY_MS,
        "train_days": TRAIN_DAYS,
        "oos_days": OOS_DAYS,
        "min_span_days": MIN_SPAN_DAYS,
        "reason": "ready" if ready else
                  f"sentiment span {span:.2f}d < {MIN_SPAN_DAYS}d required",
    }


def oi_zscore(oi: pd.DataFrame) -> pd.DataFrame:
    """逐币 7 日 z-score；rolling 只回看，std=0 置 NaN 而非 inf。"""
    roll = oi.rolling(ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS)
    return (oi - roll.mean()) / roll.std().replace(0, np.nan)


def market_factor(z: pd.DataFrame,
                  min_cross_section: int = MIN_CROSS_SECTION) -> pd.Series:
    """市场级共同因子 = 逐时点横截面均值；有效币不足门槛的时点置 NaN。"""
    arr = z.to_numpy(float)
    counts = np.isfinite(arr).sum(axis=1)
    level = np.divide(np.nansum(arr, axis=1),
                      np.where(counts > 0, counts, np.nan))
    level = np.where(counts >= min_cross_section, level, np.nan)
    return pd.Series(level, index=z.index, name="market_oi_z")


def market_forward_return(px: pd.DataFrame, horizon: int = HORIZON) -> pd.Series:
    """等权全市场未来 H 根收益；逐币要求 t..t+H 每根 bar 完整，否则该币不计入。

    反向 rolling 数连续非缺失 bar，不引入前视；逐时点只对当时有效的币取均值，
    因此天然 point-in-time，未来才上市的币不会污染早期时点。
    """
    fwd = px.shift(-horizon) / px - 1.0
    complete = px.notna().iloc[::-1].rolling(
        horizon + 1, min_periods=horizon + 1).sum().iloc[::-1] == (horizon + 1)
    return fwd.where(complete).mean(axis=1, skipna=True).rename("market_fwd_ret")


def nonoverlap_block_times(start_ms: int, end_ms: int,
                           horizon: int = HORIZON) -> np.ndarray:
    """非重叠块起点，间距恰为 H 根；末端标签窗口不完整的块一律丢弃。"""
    step = horizon * HOUR_MS
    last = int(end_ms) - step
    if last < int(start_ms):
        return np.empty(0, dtype=np.int64)
    return np.arange(int(start_ms), last + 1, step, dtype=np.int64)


def split_train_oos(block_ts: np.ndarray, train_days: int = TRAIN_DAYS,
                    oos_days: int = OOS_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """取最近 train+oos 个块：前段选阈值、后段只读。块数不足直接拒绝。"""
    need = train_days + oos_days
    blocks = np.asarray(block_ts, dtype=np.int64)
    if len(blocks) < need:
        raise ValueError(
            f"non-overlapping blocks {len(blocks)} < required {need} "
            f"(train {train_days} + oos {oos_days})")
    tail = blocks[-need:]
    return tail[:train_days], tail[train_days:]


def block_readiness(blocks: np.ndarray, factor: np.ndarray, label: np.ndarray,
                    train_days: int = TRAIN_DAYS,
                    oos_days: int = OOS_DAYS) -> dict:
    """按「评估窗内可用块数」判就绪，而不是原始跨度天数。

    72h z-score 预热会吃掉开头若干块，末端 24h 标签窗口又吃掉一块，UTC 日界对齐
    再吃掉不足一天，因此 60 天跨度并不等于 60 个可用块。窗口内部的缺口不能靠
    等更多数据补掉，用 interior_gap 显式区分，绝不静默丢块（丢块会破坏逐块复利）。
    """
    need = train_days + oos_days
    n = len(blocks)
    finite = np.isfinite(np.asarray(factor, float)) & np.isfinite(np.asarray(label, float))
    warm = 0
    while warm < n and not finite[warm]:
        warm += 1
    extra = max(0, need + warm - n)
    if n < need:
        return {"ready": False, "total_blocks": n, "required_blocks": need,
                "window_finite_blocks": 0, "window_invalid_blocks": need,
                "blocks_short": need - n, "warmup_prefix_blocks": warm,
                "extra_days_lower_bound": extra, "interior_gap": False,
                "reason": f"only {n} non-overlapping blocks, need {need} "
                          f"plus {warm} warm-up block(s)"}
    win = finite[-need:]
    bad = int((~win).sum())
    invalid_idx = np.flatnonzero(~win) + (n - need)
    interior = bool(len(invalid_idx) and int(invalid_idx.max()) > warm)
    return {"ready": bad == 0, "total_blocks": n, "required_blocks": need,
            "window_finite_blocks": need - bad, "window_invalid_blocks": bad,
            "blocks_short": 0, "warmup_prefix_blocks": warm,
            "extra_days_lower_bound": extra, "interior_gap": interior,
            "reason": "ready" if bad == 0 else
                      (f"{bad} invalid block(s) inside the evaluation window"
                       + ("; interior gap will NOT be fixed by waiting" if interior else
                          "; warm-up prefix, shifts out as collection advances"))}