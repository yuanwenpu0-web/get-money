"""P12 探针纯计算公共模块：只读 DB、无副作用、无策略状态。

标签做横截面 demean（减去同一时点全市场中位收益）后，市场共同涨跌被剔除，
剩下的才是「按因子选币」的能力。反向 rolling 保证未来窗口不含前视。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.db import make_engine  # noqa: E402

HOUR_MS = 3_600_000
PRIMARY_FACTOR = "oi_value_z7d"
PRIMARY_HORIZON = 24
MIN_CROSS_SECTION = 30
FACTOR_COLS = ("oi_value_usd", "oi_value_z7d", "oi_change_24h",
               "long_short_ratio", "top_long_short", "taker_buy_sell")
HORIZONS = (8, 24, 48)


def fetch_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    engine = make_engine(read_only=True)
    with engine.connect() as conn:
        sent = pd.read_sql_query(text(
            "SELECT symbol, ts, oi_value, long_short_ratio, "
            "top_long_short, taker_buy_sell FROM sentiment"), conn)
        lo, hi = conn.execute(text("SELECT min(ts), max(ts) FROM sentiment")).one()
        klo, khi = conn.execute(text(
            "SELECT min(open_time), max(open_time) FROM klines "
            "WHERE interval='1h'")).one()
        candles = pd.read_sql_query(text(
            "SELECT symbol, open_time, close FROM klines WHERE interval='1h' "
            "AND open_time BETWEEN :a AND :b"), conn,
            params={"a": int(lo) - 2 * HOUR_MS,
                    "b": int(hi) + (max(HORIZONS) + 2) * HOUR_MS})
        ends = dict(conn.execute(text(
            "SELECT interval, max(open_time) FROM klines GROUP BY interval")).all())
        funding_end = conn.execute(text(
            "SELECT max(funding_time) FROM funding_rates")).scalar()
    return sent, candles, {"sentiment_start_ms": int(lo), "sentiment_end_ms": int(hi),
                           "native_1h_start_ms": int(klo), "native_1h_end_ms": int(khi),
                           "kline_ends_ms": {k: int(v) for k, v in ends.items()},
                           "funding_end_ms": int(funding_end)}


def build_factors(sent: pd.DataFrame, grid: np.ndarray) -> dict[str, pd.DataFrame]:
    """0 是采集端外对齐的缺失补位，与 attach_sentiment 一致转成 NaN。"""
    pivots = {}
    for src, name in (("oi_value", "oi_value_usd"),
                      ("long_short_ratio", "long_short_ratio"),
                      ("top_long_short", "top_long_short"),
                      ("taker_buy_sell", "taker_buy_sell")):
        frame = sent.pivot(index="ts", columns="symbol", values=src).sort_index()
        pivots[name] = frame.replace(0.0, np.nan)
    oi = pivots["oi_value_usd"]
    roll = oi.rolling(168, min_periods=72)
    factors = {
        "oi_value_usd": oi,
        "oi_value_z7d": (oi - roll.mean()) / roll.std().replace(0, np.nan),
        "oi_change_24h": oi / oi.shift(24) - 1.0,
        "long_short_ratio": pivots["long_short_ratio"],
        "top_long_short": pivots["top_long_short"],
        "taker_buy_sell": pivots["taker_buy_sell"],
    }
    return {k: v.reindex(grid) for k, v in factors.items()}


def build_labels(candles: pd.DataFrame, grid: np.ndarray, horizon: int,
                 columns=None):
    """close[t+H]/close[t]-1；要求 t..t+H 每根 bar 都存在，否则置 NaN。"""
    px = candles.pivot(index="open_time", columns="symbol", values="close").sort_index()
    px = px.reindex(grid)
    if columns is not None:
        px = px.reindex(columns=columns)
    fwd = px.shift(-horizon) / px - 1.0
    # 反向 rolling：数「从 t 起到 t+H」的连续非缺失 bar，不引入前视
    complete = px.notna().iloc[::-1].rolling(
        horizon + 1, min_periods=horizon + 1).sum().iloc[::-1] == (horizon + 1)
    return fwd.where(complete), fwd.where(complete).sub(
        fwd.where(complete).median(axis=1), axis=0), complete, px


def ic_series(factor: pd.DataFrame, label: pd.DataFrame,
              valid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """逐时点横截面 Spearman rank IC 与每期有效币数。"""
    cols = factor.columns.intersection(label.columns)
    f = factor.reindex(columns=cols).to_numpy(float)
    l = label.reindex(columns=cols).to_numpy(float)
    ok = np.isfinite(f) & np.isfinite(l) & valid.reindex(columns=cols).to_numpy(bool)
    ics, counts = np.full(len(f), np.nan), ok.sum(axis=1)
    for i in range(len(f)):
        if counts[i] < MIN_CROSS_SECTION:
            continue
        mask = ok[i]
        a, b = rankdata(f[i][mask]), rankdata(l[i][mask])
        a, b = a - a.mean(), b - b.mean()
        denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
        ics[i] = float((a * b).sum() / denom) if denom > 0 else 0.0
    return ics, counts


def block_bootstrap_ci(x: np.ndarray, block: int = 24,
                       n_boot: int = 2000, seed: int = 20261111) -> dict:
    """24 期移动块重采样：相邻时点 IC 强自相关，iid bootstrap 会低估方差。"""
    x = x[np.isfinite(x)]
    if len(x) < block * 2:
        return {"ci_low": None, "ci_high": None, "n": int(len(x)), "method": "block24"}
    rng = np.random.default_rng(seed)
    reps = int(np.ceil(len(x) / block))
    starts = rng.integers(0, len(x), size=(n_boot, reps))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % len(x)
    means = x[idx.reshape(n_boot, -1)[:, :len(x)]].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"ci_low": round(float(lo), 5), "ci_high": round(float(hi), 5),
            "n": int(len(x)), "n_boot": n_boot, "method": "block24",
            "mean": round(float(x.mean()), 5)}


def permutation_null(factor: pd.DataFrame, label: pd.DataFrame,
                     valid: pd.DataFrame, n_perm: int, seed: int) -> dict:
    """负对照：每期独立打乱因子在币之间的分配，保留有效币数与边际分布。"""
    f = factor.to_numpy(float)
    rng = np.random.default_rng(seed)
    observed, _ = ic_series(factor, label, valid)
    obs_mean = float(np.nanmean(observed))
    null = np.empty(n_perm)
    for j in range(n_perm):
        shuffled = pd.DataFrame(rng.permuted(f, axis=1),
                                index=factor.index, columns=factor.columns)
        series, _ = ic_series(shuffled, label, valid)
        null[j] = float(np.nanmean(series))
    p_two = float((np.abs(null - null.mean()) >= abs(obs_mean - null.mean())).mean())
    return {"n_permutations": n_perm, "seed": seed,
            "null_mean": round(float(null.mean()), 5),
            "null_p05": round(float(np.percentile(null, 5)), 5),
            "null_p95": round(float(np.percentile(null, 95)), 5),
            "observed_mean": round(obs_mean, 5),
            "empirical_p_two_sided": round(p_two, 4),
            "percentile_in_null": round(float((null < obs_mean).mean()), 4)}


def permutation_p(factor: pd.DataFrame, label: pd.DataFrame,
                  valid: pd.DataFrame, n_perm: int, seed: int) -> float:
    """探索性组合的经验双侧 p，供 BH-FDR 使用（不做判定）。"""
    f, rng = factor.to_numpy(float), np.random.default_rng(seed)
    observed, _ = ic_series(factor, label, valid)
    obs = float(np.nanmean(observed))
    null = np.empty(n_perm)
    for j in range(n_perm):
        shuffled = pd.DataFrame(rng.permuted(f, axis=1),
                                index=factor.index, columns=factor.columns)
        series, _ = ic_series(shuffled, label, valid)
        null[j] = float(np.nanmean(series))
    return round(float((np.abs(null - null.mean())
                        >= abs(obs - null.mean())).mean()), 4)


def bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini–Hochberg q 值，控制批量探索的假阳性。"""
    p = np.asarray(pvals, float)
    order, ranked = np.argsort(p), p[np.argsort(p)]
    q = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty(len(p))
    out[order] = np.minimum(q, 1.0)
    return [round(float(v), 4) for v in out]


def common_factor_diagnostic(factor: pd.DataFrame, fwd: pd.DataFrame) -> dict:
    """量化「共同杠杆周期」：全市场平均 OI z 与全市场未来收益的时序相关。

    若显著为负，说明逐币时序 IC 主要来自时间维度的共同择时成分，
    而非同一时点跨币排序的选币能力（后者正是横截面 IC 测的东西）。
    """
    z = factor.to_numpy(float)
    counts = np.isfinite(z).sum(axis=1)
    # 全 NaN 的时点置 NaN，避免 nanmean/nanstd 的空切片告警
    level = np.divide(np.nansum(z, axis=1), np.where(counts > 0, counts, np.nan))
    stds = [np.nanstd(row) if np.isfinite(row).any() else np.nan for row in z]
    market = fwd.median(axis=1).to_numpy(float)
    m = np.isfinite(level) & np.isfinite(market)
    a, b = rankdata(level[m]), rankdata(market[m])
    a, b = a - a.mean(), b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return {
        "pairs": int(m.sum()),
        "spearman_market_oi_level_vs_market_fwd_24h":
            round(float((a * b).sum() / denom), 5) if denom > 0 else None,
        "mean_cross_section_std_of_oi_z": round(float(np.nanmean(stds)), 5),
        "mean_abs_cross_section_mean_of_oi_z":
            round(float(np.abs(level[np.isfinite(level)]).mean()), 5),
        "note": "rank IC 对逐期减常数是恒等变换，故 raw / demean 标签 IC 相同；"
                "横截面 IC 已天然剔除市场共同平移",
    }


def per_symbol_ic(factor: pd.DataFrame, label: pd.DataFrame) -> dict:
    """复现 P9 的逐币时序 IC 口径，给同号币种占比（H1 门槛之一）。"""
    values = []
    for sym in factor.columns:
        f, l = factor[sym].to_numpy(float), label[sym].to_numpy(float)
        m = np.isfinite(f) & np.isfinite(l)
        if m.sum() < 30:
            continue
        a, b = rankdata(f[m]), rankdata(l[m])
        a, b = a - a.mean(), b - b.mean()
        d = float(np.sqrt((a * a).sum() * (b * b).sum()))
        if d > 0:
            values.append(float((a * b).sum() / d))
    arr = np.asarray(values)
    if not len(arr):
        return {"symbols": 0, "mean_ic": None, "same_sign_share": None}
    sign = float(np.sign(arr.mean()))
    return {"symbols": int(len(arr)), "mean_ic": round(float(arr.mean()), 5),
            "median_ic": round(float(np.median(arr)), 5),
            "same_sign_share": round(float((np.sign(arr) == sign).mean()), 4),
            "direction": "negative" if sign < 0 else "positive",
            "label": "future_H_return_minus_cross_sectional_median"}


def summarize(ics: np.ndarray, label: str) -> dict:
    x = ics[np.isfinite(ics)]
    if not len(x):
        return {"periods": 0, "mean_ic": None, "same_sign_share": None}
    sign = float(np.sign(x.mean()))
    return {"periods": int(len(x)), "mean_ic": round(float(x.mean()), 5),
            "median_ic": round(float(np.median(x)), 5),
            "std_ic": round(float(x.std(ddof=1)), 5),
            "same_sign_share": round(float((np.sign(x) == sign).mean()), 4),
            "first_half_mean_ic": round(float(x[: len(x) // 2].mean()), 5),
            "second_half_mean_ic": round(float(x[len(x) // 2:].mean()), 5),
            "label": label}
