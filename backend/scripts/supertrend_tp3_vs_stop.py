"""Do 3R winners and stop-outs look different BEFORE entry?

Compares take_profit_3 vs stop_loss samples on the 10-bar pre-signal window plus
volatility, volume, ADX and trend-age context. Reports AUC (0.5 = no separation),
standardized mean difference, and whether the direction of the gap repeats OOS.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.db import make_engine  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.strategy.rules import prepare_frame, resolve_params  # noqa: E402

STEP = 15 * 60 * 1000
SPLIT = pd.Timestamp("2026-08-01T00:00:00Z")
CODE = "supertrend_current"


def context(symbol: str, start_ms: int, end_ms: int, params: dict) -> pd.DataFrame:
    engine = make_engine(read_only=True, null_pool=True)
    db = sessionmaker(bind=engine, future=True)()
    try:
        frame = MarketDataService(db).load_frame(symbol, "15m", start_ms=start_ms, end_ms=end_ms)
    finally:
        db.close()
        engine.dispose()
    frame = frame.loc[(frame.index.as_unit("ms").asi8 + STEP) <= end_ms].copy()
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    prepared = prepare_frame(frame, CODE, params, symbol=symbol)
    c = prepared.close
    direction = prepared["st_direction"].fillna(0).to_numpy()
    # Bars since the current Supertrend leg started (trend age at signal close).
    age = np.zeros(len(direction), dtype=int)
    for i in range(1, len(direction)):
        age[i] = age[i - 1] + 1 if direction[i] == direction[i - 1] else 0
    out = pd.DataFrame(index=prepared.index)
    out["atr_pct"] = prepared["atr_pct"]
    out["adx"] = prepared["st_adx"]
    out["trend_age"] = age
    out["ema_dist"] = c / c.ewm(span=200, adjust=False).mean() - 1.0
    out["ret_1d"] = c / c.shift(96) - 1.0
    out["vol_ratio"] = out["atr_pct"] / out["atr_pct"].rolling(672, min_periods=96).median()
    out["dollar_vol"] = (c * prepared.volume).rolling(672, min_periods=96).median()
    out["st_dir"] = direction
    return out.dropna(subset=["atr_pct"])


def auc(part: pd.DataFrame, column: str) -> float | None:
    values = part[column].to_numpy(float)
    label = part.is_tp3.to_numpy(int)
    mask = np.isfinite(values)
    if mask.sum() < 200 or len(np.unique(label[mask])) < 2:
        return None
    return float(roc_auc_score(label[mask], values[mask]))


def describe(part: pd.DataFrame, column: str) -> dict:
    a = part.loc[part.is_tp3 == 1, column].dropna()
    b = part.loc[part.is_tp3 == 0, column].dropna()
    pooled = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)) or 1e-12
    return {
        "tp3_median": round(float(a.median()), 4),
        "stop_median": round(float(b.median()), 4),
        "tp3_mean": round(float(a.mean()), 4),
        "stop_mean": round(float(b.mean()), 4),
        "cohen_d": round(float((a.mean() - b.mean()) / pooled), 3),
    }


def main() -> int:
    samples = pd.read_csv(ROOT / "output/supertrend_ml_realized_full/samples.csv",
                          parse_dates=["time"])
    samples = samples.loc[~samples.censored & samples.status.isin(["take_profit_3", "stop_loss"])].copy()
    samples["is_tp3"] = (samples.status == "take_profit_3").astype(int)
    params = resolve_params(CODE, {}, interval="15m")
    end_ms = int(pd.Timestamp.now(tz="UTC").floor("15min").timestamp() * 1000)
    start_ms = int(pd.Timestamp("2025-12-01T00:00:00Z").timestamp() * 1000)
    frames = []
    for symbol in sorted(samples.symbol.unique()):
        frame = context(symbol, start_ms, end_ms, params)
        frame["symbol"] = symbol
        frames.append(frame.reset_index().rename(columns={"index": "time", "time": "time"}))
    ctx = pd.concat(frames, ignore_index=True)
    merged = pd.merge_asof(samples.sort_values("time"), ctx.sort_values("time"),
                           on="time", by="symbol", direction="backward",
                           tolerance=pd.Timedelta("20min"))
    base = [c for c in merged.columns if c.startswith(("ret_", "body_", "range_", "upper_wick_", "lower_wick_", "volume_"))]
    context_cols = ["atr_pct", "adx", "trend_age", "ema_dist", "ret_1d", "vol_ratio", "dollar_vol"]
    features = base + context_cols
    train, oos = merged[merged.time < SPLIT], merged[merged.time >= SPLIT]
    rows = []
    for column in features:
        tr_auc, oos_auc = auc(train, column), auc(oos, column)
        if tr_auc is None or oos_auc is None:
            continue
        row = {"feature": column, "train_auc": round(tr_auc, 4), "oos_auc": round(oos_auc, 4),
               "train": describe(train, column), "oos": describe(oos, column)}
        row["consistent"] = bool(np.sign(tr_auc - 0.5) == np.sign(oos_auc - 0.5))
        rows.append(row)
    ranked = sorted(rows, key=lambda r: -abs(r["train_auc"] - 0.5))
    print(f"TP3 {int(merged.is_tp3.sum())} vs stop {int((1 - merged.is_tp3).sum())} | "
          f"train {len(train)} | oos {len(oos)}")
    print(f"{'feature':18s} {'trainAUC':>9s} {'oosAUC':>8s} {'d':>7s} {'TP3中位':>10s} {'止损中位':>10s} 一致")
    for r in ranked[:18]:
        print(f"{r['feature']:18s} {r['train_auc']:9.4f} {r['oos_auc']:8.4f} {r['train']['cohen_d']:7.3f} "
              f"{r['train']['tp3_median']:10.4f} {r['train']['stop_median']:10.4f} {r['consistent']}")

    def rate_table(column: str, q: int = 5) -> dict:
        out = {}
        for name, part in (("train", train), ("oos", oos)):
            bins = pd.qcut(part[column], q, duplicates="drop")
            table = part.groupby(bins, observed=True).agg(samples=("is_tp3", "size"), tp3_rate=("is_tp3", "mean"),
                                                           atr=("atr_pct", "median"))
            out[name] = {str(k): {"samples": int(v.samples), "tp3_rate": round(float(v.tp3_rate), 3),
                                  "atr_median_pct": round(float(v.atr * 100), 3)}
                         for k, v in table.iterrows()}
        return out

    buckets = {c: rate_table(c) for c in ("atr_pct", "vol_ratio", "volume_0", "trend_age", "adx")}
    report = {
        "samples": {"tp3": int(merged.is_tp3.sum()), "stop": int((1 - merged.is_tp3).sum())},
        "note": "AUC=0.5 表示该特征对'3R止盈 vs 止损'没有区分力；0.55 以上才算有可用差异",
        "features": rows,
        "tp3_rate_by_bucket": buckets,
    }
    out = ROOT / "output/supertrend_tp3_vs_stop_20261104.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n3R 命中率分层:")
    for name, table in buckets.items():
        print(f"== {name}")
        for key, value in table["oos"].items():
            print(f"  {key:22s} n={value['samples']:5d} tp3={value['tp3_rate']:.1%}")
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
