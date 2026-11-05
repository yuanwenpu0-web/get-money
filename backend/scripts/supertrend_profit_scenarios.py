"""Where does the Supertrend reversal strategy actually make money?

Descriptive attribution on the realized-trade sample. Conditions use only
information available at the signal bar. Discovery uses signals before
2026-08-01; everything after is out-of-sample confirmation.
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
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.backtest.significance import bootstrap_expectancy_ci  # noqa: E402
from app.db import make_engine  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402

INTERVAL = "15m"
STEP = 15 * 60 * 1000
SPLIT = pd.Timestamp("2026-08-01T00:00:00Z")


def context_frame(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    engine = make_engine(read_only=True, null_pool=True)
    db = sessionmaker(bind=engine, future=True)()
    try:
        frame = MarketDataService(db).load_frame(symbol, INTERVAL, start_ms=start_ms, end_ms=end_ms)
    finally:
        db.close()
        engine.dispose()
    frame = frame.loc[(frame.index.as_unit("ms").asi8 + STEP) <= end_ms].copy()
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    c, h, l = frame.close, frame.high, frame.low
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    out = pd.DataFrame(index=frame.index)
    out["atr_pct"] = atr / c
    out["vol_ratio"] = out["atr_pct"] / out["atr_pct"].rolling(672, min_periods=96).median()
    out["ema_dist"] = c / ema200 - 1.0
    out["ema_slope"] = ema200 / ema200.shift(20) - 1.0
    out["ret_1d"] = c / c.shift(96) - 1.0
    out["ret_7d"] = c / c.shift(672) - 1.0
    out["dollar_vol"] = (c * frame.volume).rolling(672, min_periods=96).median()
    return out.dropna(subset=["atr_pct"])


def main() -> int:
    samples = pd.read_csv(ROOT / "output/supertrend_ml_realized_full/samples.csv",
                          parse_dates=["time", "entry_time", "exit_time"])
    samples = samples.loc[~samples.censored].copy()
    end_ms = int(pd.Timestamp.now(tz="UTC").floor("15min").timestamp() * 1000)
    start_ms = int(pd.Timestamp("2025-12-01T00:00:00Z").timestamp() * 1000)
    symbols = sorted(samples.symbol.unique())
    frames = []
    for symbol in symbols:
        ctx = context_frame(symbol, start_ms, end_ms)
        ctx["symbol"] = symbol
        frames.append(ctx.reset_index().rename(columns={"time": "time"}))
    context = pd.concat(frames, ignore_index=True)
    merged = pd.merge_asof(
        samples.sort_values("time"), context.sort_values("time").assign(symbol=context.symbol),
        on="time", by="symbol", direction="backward", tolerance=pd.Timedelta("20min"),
    )
    btc = context.loc[context.symbol == "BTCUSDT"].set_index("time")["ret_1d"].rename("btc_ret_1d")
    merged = pd.merge_asof(merged.sort_values("time"), btc.reset_index().sort_values("time"),
                           on="time", direction="backward", tolerance=pd.Timedelta("20min"))
    merged["hour"] = merged.time.dt.hour
    merged["weekday"] = merged.time.dt.weekday
    merged["aligned_trend"] = np.sign(merged.side * merged.ema_dist)
    merged["trend_1d_align"] = np.sign(merged.side * merged.ret_1d)
    merged["market_up"] = (merged.btc_ret_1d > 0).astype(int)
    merged["session"] = pd.cut(merged.hour, [-1, 7, 13, 21, 23],
                               labels=["Asia", "Europe", "US", "Late"])
    merged["vol_bucket"] = pd.qcut(merged.atr_pct, 5, labels=["V1_low", "V2", "V3", "V4", "V5_high"])
    merged["liq_bucket"] = pd.qcut(merged.dollar_vol.rank(method="first"), 3,
                                   labels=["small", "mid", "large"])

    def stats(part: pd.DataFrame) -> dict:
        if len(part) < 30:
            return {"n": int(len(part))}
        ci = bootstrap_expectancy_ci([{"return_pct": x} for x in part.net_return_pct])
        wins = part.net_return_pct[part.net_return_pct > 0]
        losses = part.net_return_pct[part.net_return_pct <= 0]
        return {
            "n": int(len(part)),
            "expectancy_pct": round(float(part.net_return_pct.mean()), 3),
            "win_rate": round(float(len(wins) / len(part)), 3),
            "avg_win_pct": round(float(wins.mean()), 3) if len(wins) else 0.0,
            "avg_loss_pct": round(float(losses.mean()), 3) if len(losses) else 0.0,
            "ci_low": ci.get("ci_low_pct"), "ci_high": ci.get("ci_high_pct"), "verdict": ci.get("verdict"),
        }

    def table(column: str) -> dict:
        out = {}
        for value, part in merged.groupby(column, observed=True):
            out[str(value)] = {
                "train": stats(part[part.time < SPLIT]),
                "oos": stats(part[part.time >= SPLIT]),
            }
        return out

    dimensions = ["side", "status", "session", "vol_bucket", "liq_bucket", "aligned_trend",
                  "trend_1d_align", "market_up", "weekday"]
    report = {
        "overall": {"train": stats(merged[merged.time < SPLIT]), "oos": stats(merged[merged.time >= SPLIT])},
        "by_exit": {k: stats(v) for k, v in merged.groupby("status")},
        "dimensions": {d: table(d) for d in dimensions},
        "hold_bars": {},
    }
    merged["hold_bucket"] = pd.cut(merged.hold_bars, [-1, 2, 5, 12, 30, 10**6],
                                   labels=["1-2", "3-5", "6-12", "13-30", ">30"])
    report["hold_bars"] = table("hold_bucket")
    # Profit concentration: share of gross profit from the top deciles and by exit type.
    winners = merged[merged.net_return_pct > 0].sort_values("net_return_pct", ascending=False)
    total_profit = float(winners.net_return_pct.sum())
    report["profit_concentration"] = {
        "winners": int(len(winners)),
        "top10pct_share": round(float(winners.net_return_pct.head(max(1, len(winners) // 10)).sum()) / total_profit, 3),
        "by_exit_profit_share": {
            str(k): round(float(v.net_return_pct.sum()) / total_profit, 3)
            for k, v in winners.groupby("status")
        },
    }
    # Conditional combos that were positive in BOTH train and OOS, with n >= 200 each.
    combos = []
    for side in (1, -1):
        for vol in ("V3", "V4", "V5_high"):
            for trend in (-1, 0, 1):
                part = merged[(merged.side == side) & (merged.vol_bucket == vol) & (merged.aligned_trend == trend)]
                if len(part) < 500:
                    continue
                tr, oos = part[part.time < SPLIT], part[part.time >= SPLIT]
                if len(tr) < 200 or len(oos) < 150:
                    continue
                if tr.net_return_pct.mean() > 0 and oos.net_return_pct.mean() > 0:
                    combos.append({
                        "side": "long" if side == 1 else "short", "vol": str(vol),
                        "aligned_trend": int(trend),
                        "train": stats(tr), "oos": stats(oos),
                    })
    report["positive_in_both"] = combos
    out = ROOT / "output/supertrend_profit_scenarios_20261104.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"样本 {len(merged)} 笔 | 训练 {int((merged.time < SPLIT).sum())} | 样本外 {int((merged.time >= SPLIT).sum())}")
    print("总体:", report["overall"])
    print("\n出场:", json.dumps(report["by_exit"], ensure_ascii=False))
    for dim in ("side", "session", "vol_bucket", "aligned_trend", "trend_1d_align", "market_up"):
        print(f"\n== {dim} ==")
        for value, row in report["dimensions"][dim].items():
            tr, oos = row["train"], row["oos"]
            if "expectancy_pct" not in tr or "expectancy_pct" not in oos:
                print(f"  {value:12s} n={tr.get('n')}/{oos.get('n')} 样本不足")
                continue
            print(f"  {value:12s} train {tr['expectancy_pct']:+.3f}% (n={tr['n']}, win {tr['win_rate']:.0%}) | "
                  f"OOS {oos['expectancy_pct']:+.3f}% (n={oos['n']}, win {oos['win_rate']:.0%})")
    print("\n持有根数:")
    for value, row in report["hold_bars"].items():
        tr, oos = row["train"], row["oos"]
        if "expectancy_pct" in tr and "expectancy_pct" in oos:
            print(f"  {value:6s} train {tr['expectancy_pct']:+.3f}% | OOS {oos['expectancy_pct']:+.3f}%")
    print("\n双期为正组合:", json.dumps(combos, ensure_ascii=False, indent=2)[:1200])
    print("\n盈利集中度:", report["profit_concentration"])
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
