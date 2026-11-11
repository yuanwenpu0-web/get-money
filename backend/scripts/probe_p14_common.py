#!/usr/bin/env python3
"""P14 公共模块：USD-M 逐笔归档解析 + 逐笔「顺序」特征（只读、不落盘）。

只产出每小时桶的少量标量特征，原始逐笔帧在解析完立即释放。特征全部只用
t 收盘及之前的成交，无前视；顺序特异性对照用整行置换（保持每笔
price/quote/direction 配对不变，故 taker_imb 严格不变）。
"""
from __future__ import annotations

import io
import threading
import time
import zipfile
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"
HOUR_MS = 3_600_000
MIN_TRADES_PER_HOUR = 100
FEATURE_COLS = ("seq_flip_dev", "big_run_dev", "impact_rev", "taker_imb")
SHUFFLE_COLS = ("shuf_seq_flip_dev", "shuf_big_run_dev", "shuf_taker_imb")
CONTROL_COLS = ("n_trades", "quote_total")  # 流动性代理，仅作偏相关控制
IMPACT_STEPS = 20
BIG_QUANTILE = 0.90


class PacedClient:
    """共享限速器：归档 GET 之间至少 interval 秒，下载本身可并发。"""

    def __init__(self, interval: float = 0.25):
        self.client = httpx.Client(timeout=90, follow_redirects=True)
        self.interval = max(0.0, interval)
        self.lock = threading.Lock()
        self.next_at = 0.0

    def get(self, url: str):
        with self.lock:
            delay = self.next_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.next_at = time.monotonic() + self.interval
        return self.client.get(url)

    def close(self) -> None:
        self.client.close()


def _flip_dev(dirs: np.ndarray, p: float) -> float:
    """方向切换率相对独立伯努利基准 2p(1-p) 的偏离；负=惯性/聚集。"""
    exp = 2.0 * p * (1.0 - p)
    if len(dirs) < 2 or exp < 1e-6:
        return float("nan")
    flip = float(np.mean(dirs[1:] != dirs[:-1]))
    return flip / exp - 1.0


def _run_dev(dirs: np.ndarray) -> float:
    """大单方向平均游程长度相对独立基准 1/(2p(1-p)) 的偏离。"""
    exp_src = 2.0 * (dirs == 1).mean()
    exp = exp_src * (1.0 - (dirs == 1).mean())
    if len(dirs) < 2 or exp < 1e-6:
        return float("nan")
    runs = 1.0 + float(np.sum(dirs[1:] != dirs[:-1]))
    return (len(dirs) / runs) * exp - 1.0


def _impact(price: np.ndarray, d: np.ndarray, big: np.ndarray) -> float:
    """大单后 20 笔的价格变化 × 冲击方向；正=延续，负=回滚。"""
    idx = np.flatnonzero(big)
    idx = idx[idx + IMPACT_STEPS < len(price)]
    if len(idx) < 5:
        return float("nan")
    chg = (price[idx + IMPACT_STEPS] - price[idx]) / price[idx]
    return float(np.mean(chg * d[idx]))


def _one_hour(hour_ms: int, symbol: str, price: np.ndarray,
              quote: np.ndarray, d: np.ndarray, rng) -> dict:
    n = len(d)
    p = float((d == 1).mean())
    big = quote >= np.quantile(quote, BIG_QUANTILE)
    qb, qs = float(quote[d == 1].sum()), float(quote[d == -1].sum())
    imb = (qb - qs) / (qb + qs) if qb + qs > 0 else float("nan")
    out = {"hour": hour_ms, "symbol": symbol, "n_trades": n,
           "quote_total": float(quote.sum()),
           "seq_flip_dev": _flip_dev(d, p),
           "big_run_dev": _run_dev(d[big]),
           "impact_rev": _impact(price, d, big),
           "taker_imb": imb}
    # 整行置换：每笔 (price, quote, d) 保持配对，只打乱先后顺序
    perm = rng.permutation(n)
    ds, qs_arr, ps_arr = d[perm], quote[perm], price[perm]
    qb2, qs2 = float(qs_arr[ds == 1].sum()), float(qs_arr[ds == -1].sum())
    out["shuf_seq_flip_dev"] = _flip_dev(ds, p)
    out["shuf_big_run_dev"] = _run_dev(ds[big[perm]])
    out["shuf_taker_imb"] = ((qb2 - qs2) / (qb2 + qs2)
                             if qb2 + qs2 > 0 else float("nan"))
    _ = ps_arr  # 重排后的价格路径不用于对照（impact_rev 依赖路径连续性）
    return out


def hour_rows(content: bytes, symbol: str, day: str, seed: int) -> list[dict]:
    """解压单个日归档，按 UTC 小时桶产出特征行；存档字段不做任何持久化。"""
    rng = np.random.default_rng(seed)
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"归档 CSV 数量异常: {symbol} {day}")
        with z.open(names[0]) as f:
            df = pd.read_csv(f, usecols=["price", "quantity",
                                         "transact_time", "is_buyer_maker"])
    price = df["price"].to_numpy(float)
    quote = price * df["quantity"].to_numpy(float)
    ts = df["transact_time"].to_numpy(np.int64)
    d = np.where(df["is_buyer_maker"].astype(str).str.lower().eq("true"),
                 -1, 1).astype(np.int8)
    if len(price) and np.any(ts[1:] < ts[:-1]):
        raise ValueError(f"逐笔时间非递增: {symbol} {day}")
    buckets = (ts // HOUR_MS) * HOUR_MS
    edges = np.r_[0, np.flatnonzero(buckets[1:] != buckets[:-1]) + 1, len(buckets)]
    rows, skipped = [], 0
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < MIN_TRADES_PER_HOUR:
            skipped += 1
            continue
        rows.append(_one_hour(int(buckets[a]), symbol, price[a:b],
                              quote[a:b], d[a:b], rng))
    return rows, {"source_rows": int(len(price)), "skipped_thin_hours": skipped,
                  "kept_hours": len(rows)}


def fetch_one(client: PacedClient, symbol: str, day: date, seed: int) -> dict:
    """下载 + 立即解析；只回传小时级标量，原始逐笔不离开本函数。"""
    tag = day.isoformat()
    url = f"{BASE}/{symbol}/{symbol}-aggTrades-{tag}.zip"
    rec = {"symbol": symbol, "day": tag, "status": "ok", "rows": []}
    try:
        resp = client.get(url)
        if resp.status_code == 404:
            rec.update(status="missing_archive")
            return rec
        resp.raise_for_status()
        rec["bytes"] = len(resp.content)
        rec["rows"], rec["parse"] = hour_rows(resp.content, symbol, tag, seed)
        if not rec["rows"]:
            rec["status"] = "empty_or_thin"
    except Exception as exc:  # noqa: BLE001
        rec.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:150]}")
    return rec


def to_frames(records: list[dict], grid: np.ndarray,
              symbols: list[str]) -> dict[str, pd.DataFrame]:
    rows = [r for rec in records for r in rec["rows"]]
    df = pd.DataFrame(rows)
    cols = FEATURE_COLS + SHUFFLE_COLS + CONTROL_COLS
    if df.empty:
        return {c: pd.DataFrame(index=grid, columns=symbols, dtype=float)
                for c in cols}
    out = {}
    for col in cols:
        out[col] = df.pivot(index="hour", columns="symbol", values=col).reindex(
            index=grid, columns=symbols).astype(float)
    return out


def save_frames(frames, path, meta=None) -> None:
    """缓存小时级标量帧 + 下载元信息（不含原始逐笔），避免重复下载 ~1GB。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"frames": frames, "meta": meta or []}, path)


def load_frames(path):
    obj = pd.read_pickle(path)
    return obj["frames"], obj.get("meta", [])


def mean_xs_corr(a: pd.DataFrame, b: pd.DataFrame, min_cs: int) -> dict:
    """逐期横截面 Spearman 相关均值：检验顺序特征是否为 taker 汇总换皮。"""
    x, y = a.to_numpy(float), b.to_numpy(float)
    vals = []
    for i in range(len(x)):
        m = np.isfinite(x[i]) & np.isfinite(y[i])
        if m.sum() < min_cs:
            continue
        u, v = rankdata(x[i][m]), rankdata(y[i][m])
        u, v = u - u.mean(), v - v.mean()
        den = float(np.sqrt((u * u).sum() * (v * v).sum()))
        if den > 0:
            vals.append(float((u * v).sum() / den))
    if not vals:
        return {"periods": 0, "mean_rho": None, "mean_abs_rho": None}
    return {"periods": len(vals), "mean_rho": round(float(np.mean(vals)), 5),
            "mean_abs_rho": round(float(np.mean(np.abs(vals))), 5)}


def variance_decomp(frame: pd.DataFrame) -> dict:
    """方差拆成「逐期共同水平」与「同期跨币残差」两部分。

    P12 的教训：若信号主要落在时间维度共同平移上，横截面 rank IC 会把它
    对消掉，逐币时序 IC 再漂亮也不是选币 alpha。这里事前量化同一风险。
    """
    x = frame.to_numpy(float)
    cnt = np.isfinite(x).sum(axis=1)
    with np.errstate(invalid="ignore"):
        level = np.divide(np.nansum(x, axis=1), np.where(cnt > 0, cnt, np.nan),
                          out=np.full(len(x), np.nan), where=cnt > 0)
    within = x - level[:, None]
    lv = level[np.isfinite(level)]
    total = float(np.nanvar(x))
    between = float(np.var(lv)) if len(lv) > 1 else 0.0
    inside = float(np.nanvar(within))
    return {
        "total_variance": round(total, 6),
        "time_common_level_variance": round(between, 6),
        "cross_section_within_variance": round(inside, 6),
        "time_common_share": round(between / total, 4) if total > 0 else None,
        "cross_section_share": round(inside / total, 4) if total > 0 else None,
        "interpretation": "time_common_share 高 => 主要是全市场微观结构共同状态，"
                          "横截面 rank IC 会将其对消（P12 同型风险）",
    }


def sequence_diagnostic(frames, min_cs: int) -> dict:
    """整行重排对照 + 共线性检验 + 时间/券维度方差分解。"""
    out = {}
    for orig, shuf in (("seq_flip_dev", "shuf_seq_flip_dev"),
                       ("big_run_dev", "shuf_big_run_dev"),
                       ("taker_imb", "shuf_taker_imb")):
        a, b = frames[orig].to_numpy(float), frames[shuf].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b)
        out[orig] = {
            "pairs": int(m.sum()),
            "mean_original": round(float(np.nanmean(a)), 5),
            "mean_shuffled": round(float(np.nanmean(b)), 5),
            "std_original": round(float(np.nanstd(a)), 5),
            "std_shuffled": round(float(np.nanstd(b)), 5),
            "mean_abs_diff": round(float(np.abs(a[m] - b[m]).mean()), 8) if m.any() else None,
        }
    imb = frames["taker_imb"]
    out["collinearity_vs_taker_imb"] = {
        f: mean_xs_corr(frames[f], imb, min_cs)
        for f in ("seq_flip_dev", "big_run_dev", "impact_rev")}
    out["variance_decomposition"] = {
        f: variance_decomp(frames[f])
        for f in ("seq_flip_dev", "big_run_dev", "impact_rev", "taker_imb")}
    out["interpretation"] = (
        "整行置换保持每笔 price/quote/direction 配对，故 taker_imb 应严格不变；"
        "seq_flip_dev / big_run_dev 应向 0 收敛。若不然则顺序特征不是纯顺序量。")
    return out
