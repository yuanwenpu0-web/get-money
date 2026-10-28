"""P9 探针：正交衍生品情绪特征（OI/多空比/taker）能否预测「止盈」标签——跨币种 IC 验证。

方法论与 probe_funding 一致：
- 直接复用 build_features 产出的特征列（模型实际会用的口径），
- 标签 = triple-barrier 止盈类（label==1），
- IC = 特征与二值标签的 Pearson 相关，
- 跨币汇总：均值、显著占比（|IC|>0.05 的币占比）、同号占比。

判定：某特征跨币 |IC|均值>0.05 且多数币同号 → 存在正交 alpha 候选；否则证伪。
注意：sentiment 数据仅 ~21 天（API 硬上限），结论限定为「近期横截面」，
不能外推为长期稳健 alpha。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import func, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db import make_engine  # noqa: E402
from app.models import Sentiment  # noqa: E402
from app.features.sentiment import SENTIMENT_FEATURES  # noqa: E402
from app.ml.dataset import build_dataset  # noqa: E402
from app import utils  # noqa: E402

DAYS = 21          # sentiment 数据窗口
MIN_ROWS = 100     # 币种 sentiment 行数下限


def pearson_ic(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 50:
        return float("nan")
    xs, ys = x[mask], y[mask]
    if xs.std() == 0 or ys.std() == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def main() -> None:
    engine = make_engine(read_only=True)
    with Session(engine) as db:
        symbols = [
            s for (s,) in db.execute(
                select(Sentiment.symbol).group_by(Sentiment.symbol)
                .having(func.count() >= MIN_ROWS)
            ).all()
        ]
    print(f"有 sentiment 数据的币种: {len(symbols)}")

    records = []
    end = int(utils.now_ms())
    for i, sym in enumerate(symbols, 1):
        try:
            with Session(engine) as db:
                data = build_dataset(db, sym, "1h", horizon=5, days=DAYS, end_ms=end)
        except Exception:
            continue
        if data is None or len(data) < 200 or "label" not in data.columns:
            continue
        target = (data["label"] == 1).astype(float).to_numpy()
        row = {"symbol": sym, "bars": int(len(data)), "up_ratio": round(float(target.mean()), 4)}
        # 稳定性派生：oi_value 7 天 z-score（去除与价格水平共线的纯规模效应）；
        # 若水平 IC 来自杠杆拥挤而非币价水平，z-score 版应保持负向。
        if "oi_value_usd" in data.columns:
            oiv = data["oi_value_usd"].astype(float)
            m = oiv.rolling(168, min_periods=48).mean()
            s = oiv.rolling(168, min_periods=48).std(ddof=0)
            data["oi_value_z7d"] = ((oiv - m) / s.replace(0, np.nan)).fillna(0.0)
        feats_all = list(SENTIMENT_FEATURES) + ["oi_value_z7d"]
        for feat in feats_all:
            if feat in data.columns:
                ic = pearson_ic(data[feat].to_numpy(dtype=float), target)
                if np.isfinite(ic):
                    row[f"ic_{feat}"] = ic
        # 分半窗一致性：前/后半窗口的 oi_value_usd IC 是否同号（排除单一 regime 偶然）
        if "oi_value_usd" in data.columns:
            half = len(data) // 2
            ic_a = pearson_ic(data["oi_value_usd"].to_numpy(dtype=float)[:half], target[:half])
            ic_b = pearson_ic(data["oi_value_usd"].to_numpy(dtype=float)[half:], target[half:])
            if np.isfinite(ic_a) and np.isfinite(ic_b):
                row["oi_half_a"] = ic_a
                row["oi_half_b"] = ic_b
        records.append(row)
        if i % 100 == 0:
            print(f"  ... {i}/{len(symbols)}", flush=True)

    feats = [f"ic_{f}" for f in SENTIMENT_FEATURES] + ["ic_oi_value_z7d"]
    summary = {}
    for f in feats:
        vals = np.array([r[f] for r in records if f in r and np.isfinite(r[f])], dtype=float)
        if len(vals) == 0:
            continue
        pos = float((vals > 0).mean())
        sig = float((np.abs(vals) > 0.05).mean())
        summary[f] = {
            "n": int(len(vals)), "mean": round(float(vals.mean()), 4),
            "median": round(float(np.median(vals)), 4),
            "sign_frac": round(max(pos, 1 - pos), 3),
            "sig_frac_gt_0.05": round(sig, 3),
        }

    # 分半窗一致性：两半 IC 同为负（或同为正）的币占比
    halves = [r for r in records if "oi_half_a" in r]
    if halves:
        same = sum(1 for r in halves if r["oi_half_a"] * r["oi_half_b"] > 0)
        both_neg = sum(1 for r in halves if r["oi_half_a"] < 0 and r["oi_half_b"] < 0)
        summary["oi_value_halves"] = {
            "n": len(halves), "same_sign_frac": round(same / len(halves), 3),
            "both_negative_frac": round(both_neg / len(halves), 3),
            "mean_a": round(float(np.mean([r["oi_half_a"] for r in halves])), 4),
            "mean_b": round(float(np.mean([r["oi_half_b"] for r in halves])), 4),
        }

    result = {
        "days": DAYS, "symbols_used": len(records),
        "summary": summary,
        "symbols": records,
    }
    out = ROOT / "docs" / "probe-sentiment.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    print("\n=== P9 正交情绪特征跨币 IC ===")
    print(f"{'特征':<22}{'n':>5}{'IC均值':>9}{'中位数':>9}{'同号%':>7}{'|IC|>0.05%':>11}")
    for f, s in summary.items():
        if not f.startswith("ic_") or "mean" not in s:
            continue
        name = f.replace("ic_", "")
        print(f"{name:<22}{s['n']:>5}{s['mean']:>9}{s['median']:>9}"
              f"{s['sign_frac']*100:>7.0f}{s['sig_frac_gt_0.05']*100:>11.0f}")
    h = summary.get("oi_value_halves")
    if h:
        print(f"\noi_value_usd 分半窗一致性: n={h['n']}  两半同号 {h['same_sign_frac']*100:.0f}%  "
              f"两半皆负 {h['both_negative_frac']*100:.0f}%  (前半均值 {h['mean_a']} / 后半均值 {h['mean_b']})")
    print(f"\n样本: {len(records)} 币 × ~{DAYS} 天 1h；报告: {out}")


if __name__ == "__main__":
    main()
