"""资金费率横截面收割探针（非方向性，delta 中性）。

背景：P0~P10 证明现货(永续)OHLCV 无方向性 alpha；本探针换赛道——
不做价格预测，做空高费率永续+多现货对冲，直接收割资金费。

Gate 0（persistence）：过去 7 天均费率 vs 未来 3 天实际费率的横截面 Spearman 相关。
    通过线：平均 rank IC > 0.1 且逐期为正占比 > 60%。
Gate 1（策略模拟）：每 3 天挑过去 7 天均费率最高且 >0 的 K 币做空永续，
    收入=持有期实际费率之和；成本=换仓双腿费用（2腿×2边×(fee+slip)=0.0024/名次换手）。
    对照组：全宇宙等权做空（无选择）。

用法：
    .venv/Scripts/python.exe scripts/probe_funding_harvest.py [--days 400]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.db import make_engine  # noqa: E402
from app.models import FundingRate  # noqa: E402

TRAIL_DAYS = 7      # 挑币用的过去窗口
HOLD_DAYS = 3       # 持有/重平衡周期
TOP_K = (10, 20)    # 两组 K 对照
FEE = 0.0004
SLIP = 0.0002
# delta 中性开/平一个名次 = 现货腿 + 永续腿，各 2 边（开+平）
COST_PER_NAME = 2 * 2 * (FEE + SLIP)  # 0.0024


def load_funding(days: int) -> pd.DataFrame:
    """读全库 funding，返回 (symbol, time, rate) 长表。"""
    engine = make_engine(read_only=True)
    with engine.connect() as conn:
        rows = conn.execute(
            select(FundingRate.symbol, FundingRate.funding_time, FundingRate.funding_rate)
        ).all()
    engine.dispose()
    df = pd.DataFrame(rows, columns=["symbol", "time", "rate"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    cutoff = df["time"].max() - pd.Timedelta(days=days)
    return df[df["time"] >= cutoff].reset_index(drop=True)


def to_grid(df: pd.DataFrame) -> pd.DataFrame:
    """把带毫秒偏移的事件时间 floor 到 8h 桶，桶内 sum（=该 8h 实际费率收入）。

    各币 funding_time 有 ±ms 偏移，直接 pivot 会把同一 8h 事件拆成几十万个
    近似时间戳，行数 rolling 就不再是日历窗口。桶化后网格规则（~1200 行/400 天）。
    """
    df = df.copy()
    df["bucket"] = df["time"].dt.floor("8h")
    wide = df.pivot_table(index="bucket", columns="symbol", values="rate", aggfunc="sum")
    return wide.sort_index()


def persistence(df: pd.DataFrame) -> dict:
    """Gate 0：逐重平衡点算横截面 rank IC（过去 7 天均费率 vs 未来 3 天实际费率）。"""
    wide_rate = to_grid(df)
    # 7 天滚动均费率（按事件 8h 网格，21 根 ≈ 7 天）
    trail = wide_rate.rolling(21, min_periods=10).mean()
    # 未来 3 天实际费率 = 反向滚动和（9 根 ≈ 3 天，含当前根）
    fut = (
        wide_rate.iloc[::-1].rolling(9, min_periods=5).sum().iloc[::-1]
    )
    ics: list[float] = []
    top_spread: list[float] = []  # 最高 7 天均费率 10 币的未来费率均值 − 全宇宙均值
    dates = trail.index
    step = 9  # 每 3 天一个重平衡点（8h 网格）
    for i in range(21, len(dates) - 9, step):
        t = trail.iloc[i].dropna()
        f = fut.iloc[i].reindex(t.index).dropna()
        common = t.index.intersection(f.index)
        if len(common) < 30:
            continue
        tt, ff = t[common], f[common]
        ic = tt.corr(ff, method="spearman")
        if np.isfinite(ic):
            ics.append(float(ic))
        k = min(10, len(common))
        top = tt.nlargest(k).index
        top_spread.append(float(ff[top].mean() - ff.mean()))
    return {
        "n_periods": len(ics),
        "mean_ic": float(np.mean(ics)) if ics else float("nan"),
        "pos_frac": float(np.mean([i > 0 for i in ics])) if ics else float("nan"),
        "top10_future_edge_per_3d": float(np.mean(top_spread)) if top_spread else float("nan"),
    }


def simulate(df: pd.DataFrame, k: int) -> dict:
    """Gate 1：每 3 天重平衡，做空未来费率预期最高的 k 币（用过去 7 天均值选择）。"""
    wide_rate = to_grid(df)
    trail = wide_rate.rolling(21, min_periods=10).mean()
    fut = wide_rate.iloc[::-1].rolling(9, min_periods=5).sum().iloc[::-1]
    dates = trail.index
    step = 9
    held_prev: frozenset[str] = frozenset()
    period_rets: list[float] = []
    turnovers: list[float] = []
    gross_list: list[float] = []
    trail_pick_means: list[float] = []  # 诊断：入选币的平均 trailing 费率（看选择有多极端）
    for i in range(21, len(dates) - 9, step):
        t = trail.iloc[i].dropna()
        t = t[t > 0]  # 只做正费率：负费率环境空仓收不到钱
        pick = set(t.nlargest(k).index) if len(t) >= 3 else set()
        if not pick:
            period_rets.append(0.0)
            turnovers.append(0.0)
            gross_list.append(0.0)
            trail_pick_means.append(0.0)
            held_prev = frozenset()
            continue
        gross = fut.iloc[i].reindex(sorted(pick)).dropna()
        gross_mean = float(gross.mean()) if len(gross) else 0.0
        turnover = 1.0 - len(pick & held_prev) / max(len(pick | held_prev), 1)
        cost = COST_PER_NAME * turnover
        period_rets.append(gross_mean - cost)
        gross_list.append(gross_mean)
        turnovers.append(turnover)
        trail_pick_means.append(float(t.reindex(sorted(pick)).mean()))
        held_prev = frozenset(pick)
    rets = pd.Series(period_rets)
    gross_s = pd.Series(gross_list)
    n_years = len(rets) * HOLD_DAYS / 365
    return {
        "k": k,
        "n_periods": int(len(rets)),
        "flat_frac": float((rets == 0).mean()),
        "avg_turnover": float(np.mean(turnovers)),
        "gross_ann_pct": float(gross_s.mean() / HOLD_DAYS * 365 * 100),
        "net_ann_pct": float((rets.sum() / n_years) * 100) if n_years else float("nan"),
        "win_frac": float((rets > 0).mean()),
        "worst_3d_pct": float(rets.min() * 100),
        "avg_trail_funding_8h_pct": float(
            np.mean(trail_pick_means) * 100
        ) if trail_pick_means else float("nan"),
    }


def main() -> None:
    days = 400
    if "--days" in sys.argv:
        days = int(sys.argv[sys.argv.index("--days") + 1])
    df = load_funding(days)
    n_sym = df["symbol"].nunique()
    print(f"数据：{n_sym} 币 / {len(df):,} 事件 / {days} 天", flush=True)

    p = persistence(df)
    print("\n=== Gate 0: persistence（过去7天均费率 → 未来3天实际费率）===")
    print(f"重平衡点: {p['n_periods']}  平均 rankIC: {p['mean_ic']:.4f}  "
          f"正占比: {p['pos_frac']*100:.0f}%")
    print(f"Top10(按过去7天费率)未来3天费率超额: {p['top10_future_edge_per_3d']*100:.4f}%/3d"
          f"  ≈ 年化 {p['top10_future_edge_per_3d']*122*100:.1f}%")

    gate0 = p["mean_ic"] > 0.1 and p["pos_frac"] > 0.6
    print(f"Gate 0: {'PASS' if gate0 else 'FAIL'}")

    out = {"days": days, "symbols": int(n_sym), "persistence": p, "gate0": bool(gate0)}
    if gate0:
        print("\n=== Gate 1: delta 中性收割模拟（空永续收费用，含双腿换手成本）===")
        sims = [simulate(df, k) for k in TOP_K]
        for s in sims:
            print(f"K={s['k']:>3}: 净年化 {s['net_ann_pct']:.2f}%  "
                  f"毛年化 {s['gross_ann_pct']:.2f}%  平均换手 {s['avg_turnover']:.0%}  "
                  f"正期占比 {s['win_frac']*100:.0f}%  最差3天 {s['worst_3d_pct']:.2f}%  "
                  f"入选币平均trailing {s['avg_trail_funding_8h_pct']:.4f}%/8h")
        # 对照：全宇宙等权做空（无选择，吃平均费率）
        wide = to_grid(df)
        fut_all = wide.iloc[::-1].rolling(9, min_periods=5).sum().iloc[::-1]
        uni = fut_all.resample("3D").mean().mean().mean()
        print(f"对照(全宇宙等权): 平均 3 天费率 {uni*100:.4f}% ≈ 年化 {uni*122*100:.2f}%")
        gate1 = max(s["net_ann_pct"] for s in sims) > 3.0  # 年化 >3% 才值得做
        out["sims"] = sims
        out["gate1"] = bool(gate1)
        print(f"Gate 1(净年化>3%): {'PASS' if gate1 else 'FAIL'}")

    out_path = ROOT / "docs" / "probe-funding-harvest.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {out_path}")


if __name__ == "__main__":
    main()
