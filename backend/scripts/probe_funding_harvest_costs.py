"""P11 资金费收割 · 成本口径与换手缓冲矩阵（诚实宇宙 = Binance 现货可对冲）。

背景：probe_funding_harvest.py 的未过滤宇宙净年化 +41.8%，但接入现货对过滤后
（348/507 币）taker 口径净年化 −2.4%（FAIL）。本脚本回答两个问题：

1. 成本降到多少能翻正？—— 扫 COST_PER_NAME 从 0 到 taker 口径，找 break-even。
   成本构成 = 双腿（现货+永续）× 双边（开+平）× (fee+slip) / 名次换手。
   参考费率（Binance 普通用户）：永续 taker 0.04% / maker 0.02%；
   现货 taker/maker 0.10%（BNB 75 折 0.075%）。
2. 换手缓冲（hysteresis）能省多少？—— 已持有币跌出 Top-K 但仍在 Top-(K+buffer)
   且费率为正时继续持有，降低每 3 天的名次换手。

口径与 probe_funding_harvest.py / app/strategy/funding_harvest.py 完全一致：
8h 桶、trailing 7 天（21 桶）、3 天重平衡（9 桶）。幸存者偏差同前（在架币池）。

用法：
    .venv/Scripts/python.exe scripts/probe_funding_harvest_costs.py [--days 400]
    --no-spot-filter   网络不可用时跳过过滤（结果不可用于 gate 判定）
"""

from __future__ import annotations

import argparse
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
from app.state import all_state  # noqa: E402
from app.strategy.funding_harvest import match_spot_base, spot_base_symbols  # noqa: E402

TRAIL_BUCKETS = 21  # 7 天
HOLD_BUCKETS = 9    # 3 天
HOLD_DAYS = 3
PERIODS_PER_YEAR = 365 / HOLD_DAYS

# 成本场景：每名次开+平双腿总成本（占单名名义比例）
SCENARIOS = {
    # 探针原口径：两腿都按 fee 0.04% + slip 0.02%（对现货腿偏乐观）
    "taker_probe(0.24%)": 4 * (0.0004 + 0.0002),
    # 诚实 taker：永续 2×(0.04%+0.02%) + 现货 2×(0.075%BNB+0.02%)
    "taker_honest(0.31%)": 2 * (0.0004 + 0.0002) + 2 * (0.00075 + 0.0002),
    # 永续 maker(postOnly 0.02%+0.01% 不利偏移) + 现货 taker
    "perp_maker(0.25%)": 2 * (0.0002 + 0.0001) + 2 * (0.00075 + 0.0002),
    # 双腿 maker：永续 2×0.03% + 现货 2×(0.075%+0.01%)
    "both_maker(0.23%)": 2 * (0.0002 + 0.0001) + 2 * (0.00075 + 0.0001),
}
K_VALUES = (10, 20)
BUFFERS = (0, 10)  # hysteresis：0=严格 Top-K；10=持有币跌出 K 但仍在 K+10 内则留


def load_funding(days: int) -> pd.DataFrame:
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
    df = df.copy()
    df["bucket"] = df["time"].dt.floor("8h")
    wide = df.pivot_table(index="bucket", columns="symbol", values="rate", aggfunc="sum")
    return wide.sort_index()


def persistence(wide: pd.DataFrame) -> dict:
    trail = wide.rolling(TRAIL_BUCKETS, min_periods=10).mean()
    fut = wide.iloc[::-1].rolling(HOLD_BUCKETS, min_periods=5).sum().iloc[::-1]
    ics: list[float] = []
    dates = trail.index
    for i in range(TRAIL_BUCKETS, len(dates) - HOLD_BUCKETS, HOLD_BUCKETS):
        t = trail.iloc[i].dropna()
        f = fut.iloc[i].reindex(t.index).dropna()
        common = t.index.intersection(f.index)
        if len(common) < 30:
            continue
        ic = t[common].corr(f[common], method="spearman")
        if np.isfinite(ic):
            ics.append(float(ic))
    return {
        "n_periods": len(ics),
        "mean_ic": float(np.mean(ics)) if ics else float("nan"),
        "pos_frac": float(np.mean([i > 0 for i in ics])) if ics else float("nan"),
    }


def simulate(
    wide: pd.DataFrame, k: int, cost_per_name: float, buffer: int = 0,
) -> dict:
    """3 天重平衡；buffer>0 时持有币仍在 Top-(K+buffer) 且费率>0 则继续持有。"""
    trail = wide.rolling(TRAIL_BUCKETS, min_periods=10).mean()
    fut = wide.iloc[::-1].rolling(HOLD_BUCKETS, min_periods=5).sum().iloc[::-1]
    dates = trail.index
    held_prev: frozenset[str] = frozenset()
    period_rets: list[float] = []
    turnovers: list[float] = []
    gross_list: list[float] = []
    for i in range(TRAIL_BUCKETS, len(dates) - HOLD_BUCKETS, HOLD_BUCKETS):
        t = trail.iloc[i].dropna()
        t = t[t > 0].sort_values(ascending=False)
        if len(t) < 3:
            period_rets.append(0.0)
            turnovers.append(0.0)
            gross_list.append(0.0)
            held_prev = frozenset()
            continue
        pick = set(t.head(k).index)
        if buffer and held_prev:
            keep = {s for s in held_prev if s in t.head(k + buffer).index}
            pick = set(list(pick)[: max(0, k - len(keep))]) | keep
            pick = set(t.index[t.index.isin(pick)][:k])
        gross = fut.iloc[i].reindex(sorted(pick)).dropna()
        gross_mean = float(gross.mean()) if len(gross) else 0.0
        to = 1.0 - len(pick & held_prev) / max(len(pick | held_prev), 1)
        period_rets.append(gross_mean - cost_per_name * to)
        gross_list.append(gross_mean)
        turnovers.append(to)
        held_prev = frozenset(pick)
    rets = pd.Series(period_rets)
    gross_s = pd.Series(gross_list)
    n_years = len(rets) * HOLD_DAYS / 365
    return {
        "k": k,
        "buffer": buffer,
        "n_periods": int(len(rets)),
        "avg_turnover": float(np.mean(turnovers)),
        "gross_ann_pct": float(gross_s.mean() / HOLD_DAYS * 365 * 100),
        "net_ann_pct": float((rets.sum() / n_years) * 100) if n_years else float("nan"),
        "win_frac": float((rets > 0).mean()),
        "worst_3d_pct": float(rets.min() * 100),
        "cost_drag_ann_pct": float(
            cost_per_name * np.mean(turnovers) / HOLD_DAYS * 365 * 100
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--no-spot-filter", action="store_true")
    args = ap.parse_args()

    df = load_funding(args.days)
    n_all = df["symbol"].nunique()
    print(f"数据：{n_all} 币 / {len(df):,} 事件 / {args.days} 天", flush=True)

    # 现货可对冲过滤
    spot_note = "未过滤（--no-spot-filter）"
    if not args.no_spot_filter:
        engine = make_engine(read_only=True)
        with engine.connect() as conn:
            proxy = None
            try:
                from sqlalchemy.orm import Session
                with Session(engine) as db:
                    proxy = all_state(db).get("proxy_url") or None
            except Exception:
                proxy = None
        engine.dispose()
        bases = spot_base_symbols(proxy)
        eligible = {s for s in df["symbol"].unique() if match_spot_base(s, bases)}
        df = df[df["symbol"].isin(eligible)].reset_index(drop=True)
        spot_note = f"现货可对冲 {df['symbol'].nunique()}/{n_all}"
    print(f"宇宙：{spot_note}", flush=True)

    wide = to_grid(df)
    p = persistence(wide)
    print(f"\nGate 0 persistence：rank IC {p['mean_ic']:.4f}，正占比 {p['pos_frac']*100:.0f}%"
          f"（{p['n_periods']} 期）")

    rows: list[dict] = []
    print("\n=== 成本 × 换手矩阵（净年化 %）===")
    header = f"{'场景':<22}" + "".join(f"K={k},buf={b:>2}  " for k in K_VALUES for b in BUFFERS)
    print(header)
    for name, cost in SCENARIOS.items():
        line = f"{name:<22}"
        for k in K_VALUES:
            for b in BUFFERS:
                s = simulate(wide, k, cost, b)
                rows.append({"scenario": name, "cost_per_name": cost, **s})
                line += f"{s['net_ann_pct']:>9.2f}  "
        print(line, flush=True)

    # break-even：毛收益 / 换手 = 能承担的最大每名次成本
    print("\n=== Break-even（净年化=0 时每名次成本上限）===")
    breakevens: list[dict] = []
    for k in K_VALUES:
        for b in BUFFERS:
            s0 = simulate(wide, k, 0.0, b)  # 零成本得毛口径与换手
            if s0["avg_turnover"] > 0 and s0["gross_ann_pct"] > 0:
                be = (s0["gross_ann_pct"] / 100) / (
                    s0["avg_turnover"] * PERIODS_PER_YEAR
                )
            else:
                be = float("nan")
            breakevens.append({"k": k, "buffer": b, "breakeven_cost_per_name": be,
                               "gross_ann_pct": s0["gross_ann_pct"],
                               "avg_turnover": s0["avg_turnover"]})
            print(f"K={k}, buf={b:>2}: 毛 {s0['gross_ann_pct']:.2f}%/年, "
                  f"换手 {s0['avg_turnover']:.0%}/3d → 每名次成本上限 {be*100:.3f}%")

    # gate：任一场景净年化 > 3% 且正期占比 > 60% 视为可推进
    viable = [r for r in rows if r["net_ann_pct"] > 3.0 and r["win_frac"] > 0.6]
    gate = bool(viable)
    print(f"\nGate（存在 净年化>3% 且 正期>60% 的组合）: {'PASS' if gate else 'FAIL'}")
    for r in viable:
        print(f"  ✔ {r['scenario']} K={r['k']} buf={r['buffer']}: "
              f"净 {r['net_ann_pct']:.2f}% 正期 {r['win_frac']*100:.0f}% "
              f"最差3天 {r['worst_3d_pct']:.2f}%")

    out = {
        "days": args.days, "universe": spot_note,
        "persistence": p, "scenarios": SCENARIOS, "rows": rows,
        "breakevens": breakevens, "gate": gate,
    }
    out_path = ROOT / "docs" / "probe-funding-harvest-costs.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {out_path}")


if __name__ == "__main__":
    main()
