"""独立复核 Gate 1a：用 pandas 布尔掩码重算，不复用探针的 searchsorted 纯函数。

目的是抓「同一套代码自证」的盲区：窗口边界、选币、年化、D1 池化比、子集 IC。
只读引擎；退出码 0 = 全部一致，1 = 存在不一致。

用法：.venv/Scripts/python.exe backend/scripts/audit_funding_venue.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.db import make_engine
from app.market.metrics import base_asset

DAY = 86_400_000
LOOKBACK, HOLD, KS = 7, 3, (10, 20)

payload = json.loads((ROOT / "docs/probe-funding-venue.json").read_text(encoding="utf-8"))
spot = json.loads((ROOT / "docs/probe-funding-venue-spot.json").read_text(encoding="utf-8"))
bases = set(spot["bases"])

engine = make_engine(read_only=True)
with engine.connect() as conn:
    frame = pd.read_sql_query(text(
        "SELECT symbol, funding_time, funding_rate FROM funding_rates"), conn)
frame["symbol"] = frame["symbol"].str.upper()

hedge = set()
for sym in frame["symbol"].unique():
    base = base_asset(sym)
    cands = {base}
    for p in ("1000000", "100000", "10000", "1000"):
        if base.startswith(p) and len(base) > len(p):
            cands.add(base[len(p):])
            break
    if cands & bases:
        hedge.add(sym)
nonhedge = set(frame["symbol"].unique()) - hedge
print(f"独立切分：可对冲 {len(hedge)} / 不可对冲 {len(nonhedge)}")
assert len(hedge) == payload["universe"]["hedgeable_count"], "可对冲数不一致"
assert len(nonhedge) == payload["universe"]["non_hedgeable_count"], "不可对冲数不一致"

start, end = int(frame.funding_time.min()), int(frame.funding_time.max())
times = list(range(start + LOOKBACK * DAY, end - HOLD * DAY + 1, HOLD * DAY))
print(f"独立期数 {len(times)} vs 报告 {payload['periods']['count']}")
assert len(times) == payload["periods"]["count"], "期数不一致"

per_k = {k: {"full": [], "hedge": [], "num": 0.0, "den": 0.0, "ic": []} for k in KS}
for t in times:
    back = frame[(frame.funding_time >= t - LOOKBACK * DAY) & (frame.funding_time < t)]
    fwd = frame[(frame.funding_time >= t) & (frame.funding_time < t + HOLD * DAY)]
    trailing = back.groupby("symbol").funding_rate.mean().to_dict()
    forward = fwd.groupby("symbol").funding_rate.sum().to_dict()
    for k in KS:
        bucket = per_k[k]
        for universe, key in ((set(trailing), "full"), (hedge, "hedge")):
            eligible = [(s, trailing[s]) for s in universe
                       if s in trailing and trailing[s] > 0]
            eligible.sort(key=lambda kv: (-kv[1], kv[0]))
            picked = [s for s, _ in eligible[:k]]
            vals = [forward[s] for s in picked if s in forward]
            bucket[key].append(float(np.mean(vals)) if vals else None)
            if key == "full":
                bucket["num"] += sum(forward[s] for s in picked
                                     if s in forward and s in nonhedge)
                bucket["den"] += sum(forward[s] for s in picked if s in forward)
        pairs = [(trailing[s], forward[s]) for s in nonhedge
                 if s in trailing and s in forward]
        if len(pairs) >= 30:
            a = pd.Series([p[0] for p in pairs]).rank().to_numpy()
            b = pd.Series([p[1] for p in pairs]).rank().to_numpy()
            bucket["ic"].append(float(np.corrcoef(a, b)[0, 1]))


def annual(values):
    vals = [v for v in values if v is not None]
    growth = float(np.prod([1 + v for v in vals]))
    return (growth ** (365.0 / (len(vals) * HOLD)) - 1) * 100


print("\n=== 独立重算 vs 报告 ===")
ok = True
for k in KS:
    bucket, arm = per_k[k], payload["arms"][str(k)]
    full_ann, hedge_ann = annual(bucket["full"]), annual(bucket["hedge"])
    d3 = full_ann - hedge_ann
    share = bucket["num"] / bucket["den"]
    ic = float(np.mean(bucket["ic"]))
    checks = [
        ("全宇宙毛年化", full_ann, arm["full"]["annual_gross_pct"]),
        ("仅可对冲毛年化", hedge_ann, arm["hedgeable_only"]["annual_gross_pct"]),
        ("D3 增量pp", d3, arm["d3_increment_pp"]),
        ("D1 池化占比", share, arm["bonus"]["pooled_share"]),
        ("D2 子集IC", ic, arm["ic_non_hedgeable"]["mean"]),
    ]
    for label, mine, reported in checks:
        close = abs(mine - reported) < 5e-4
        ok &= close
        print(f"K={k} {label}: 独立 {mine:.6f} / 报告 {reported} -> "
              f"{'一致' if close else '不一致'}")

print("\n=== 因果性抽查（前瞻窗不得含 T 之前的事件）===")
t = times[50]
fwd = frame[(frame.funding_time >= t) & (frame.funding_time < t + HOLD * DAY)]
back = frame[(frame.funding_time >= t - LOOKBACK * DAY) & (frame.funding_time < t)]
assert fwd.funding_time.min() >= t and back.funding_time.max() < t
print(f"T={pd.to_datetime(t, unit='ms', utc=True)} 回看末事件 "
      f"{pd.to_datetime(back.funding_time.max(), unit='ms', utc=True)} < T ≤ 前瞻首事件 "
      f"{pd.to_datetime(fwd.funding_time.min(), unit='ms', utc=True)}")

print("\n=== 结算周期分布抽查 ===")
counts = frame.groupby("symbol").size()
span_days = (end - start) / DAY
per_day = counts / span_days
print(f"每日事件数分位：p10 {per_day.quantile(0.1):.2f} / 中位 {per_day.median():.2f} "
      f"/ p90 {per_day.quantile(0.9):.2f}（3=8h, 6=4h）")

print("\n结论：" + ("全部一致，独立复核通过" if ok else "存在不一致，须查因"))
sys.exit(0 if ok else 1)