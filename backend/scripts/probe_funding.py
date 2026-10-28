"""P6 探针：资金费率（FundingRate）能否预测「止盈」标签——跨币种验证 funding_cum 是否真 alpha。

路线图判定：P1~P4 证明现货 OHLCV 序列在 1m~1h 全频率、全特征维度无方向性 alpha；
P5 降换手只是"少亏"不"多赚"。要真正出 edge，必须引入**独立信息源**。

本脚本在 P6 回填（backfill_funding.py 已把资金费扩到全宇宙）之后运行，验证：
- 此前只用 BTC/ETH 2 币得到 f_cum IC≈-0.090；现在用全宇宙复算，确认该信号**跨币种稳健**，
  而非 BTC/ETH 两币的偶然。
- 直接复用 build_features 产出的 funding_cum 列（时间窗口 3D 滚动累积，模型实际会用的特征），
  而不是探针自造的口径，保证 IC 与线上特征严格一致。

方法：
- 自动枚举库内有资金费率的 clean 币种（--symbols 可显式指定子集）；
- 每币 1h K线 + 资金费率对齐到 1h 栅格（前向填充最近一次结算）；
- triple-barrier「止盈」标签（与 P2/P3 同源，1h H=120）；
- 计算 funding_cum 与止盈标签的 IC，跨币均值与显著占比。

判定：若 funding_cum 跨币 |IC|>0.05 且多数币同号 → 出现稳健跨币种独立 alpha，P6 接入成立；
若塌回 ~0 → 资金费率的 alpha 仅限 BTC/ETH，不应作为通用特征。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

from app.db import make_engine  # noqa: E402
from app.features.indicators import _triple_barrier_label, build_features  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.ml.tiers import is_clean_symbol  # noqa: E402
from app.models import FundingRate  # noqa: E402
from app.strategy.rules import resolve_params  # noqa: E402
from app.utils import now_ms  # noqa: E402

DAY_MS = 86_400_000
DEFAULT_SYMBOLS = ""  # 空 = 自动枚举全宇宙有资金费的币种


def corr(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 50 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def funded_symbols(db: Session, min_rows: int = 50) -> list[str]:
    """库内有足够资金费历史的 clean 币种。"""
    rows = db.execute(
        select(FundingRate.symbol, func.count(FundingRate.id).label("n"))
        .group_by(FundingRate.symbol)
        .having(func.count(FundingRate.id) >= min_rows)
    ).all()
    return sorted(s for s, _ in rows if is_clean_symbol(s))


def probe(db: Session, symbols: list[str], days: int) -> list[dict]:
    params = resolve_params("ml_ensemble", {}, {})
    barrier = {
        "sl_atr_mult": params["sl_atr_mult"],
        "tp_atr_mult": params["tp_atr_mult"],
        "max_hold_bars": params["max_hold_bars"],
        "horizon": 5,
    }
    end = now_ms()
    start = end - days * DAY_MS
    svc = MarketDataService(db)
    out = []
    for sym in symbols:
        f = svc.load_frame(sym, "1h", start_ms=start, end_ms=end)
        if f.empty:
            continue
        funding = svc.load_funding_frame(sym, start - 90 * DAY_MS, end)
        if funding is None or len(funding) == 0:
            out.append({"symbol": sym, "status": "no_funding"})
            continue
        feat = build_features(f, with_label=False, funding=funding)
        if feat["funding_available"].sum() == 0:
            out.append({"symbol": sym, "status": "no_funding"})
            continue
        lbl = _triple_barrier_label(feat, feat["close"], barrier)
        # 目标：未来窗口内先触止盈（label==1）
        target = (lbl == 1).astype(float).reindex(f.index).to_numpy()
        up_ratio = float((lbl == 1).mean())

        fcum = feat["funding_cum"].to_numpy()            # 模型实际特征（3D 滚动累积）
        frate = feat["funding_rate"].to_numpy()
        fz = feat["funding_zscore"].to_numpy()
        # 与最初探针一致的 bar-based 72 根累积（1h 下≈3 天），用于口径对照
        f_cum72 = feat["funding_rate"].rolling(72).sum().to_numpy()
        f_diff = feat["funding_rate"].diff().to_numpy()

        out.append({
            "symbol": sym,
            "up_ratio": round(up_ratio, 3),
            "ic_funding_cum": corr(fcum, target),
            "ic_f_rate": corr(frate, target),
            "ic_f_z": corr(fz, target),
            "ic_f_diff": corr(f_diff, target),
            "ic_f_cum72": corr(f_cum72, target),
        })
    return out


def _summarize(records: list[dict], key: str) -> dict:
    vals = [r[key] for r in records if key in r and np.isfinite(r[key])]
    if not vals:
        return {"mean": float("nan"), "n": 0, "sign_frac": float("nan")}
    mean = float(np.mean(vals))
    same_sign = sum(1 for v in vals if (v < 0) == (mean < 0)) / len(vals)
    return {"mean": round(mean, 4), "n": len(vals), "sign_frac": round(same_sign, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS, help="逗号分隔；缺省自动枚举全宇宙")
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--min-rows", type=int, default=50, help="单币资金费最少条数门槛")
    ap.add_argument("--output", type=Path, default=REPO / "docs" / "probe-funding-all.json")
    args = ap.parse_args()

    engine = make_engine(read_only=True)
    with Session(engine) as db:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] \
            if args.symbols else funded_symbols(db, args.min_rows)
        print(f"探针币种：{len(symbols)} 个；days={args.days}")
        started = time.time()
        records = probe(db, symbols, args.days)

    valid = [r for r in records if "ic_funding_cum" in r]
    no_funding = [r for r in records if r.get("status") == "no_funding"]
    feats = ["ic_funding_cum", "ic_f_rate", "ic_f_z", "ic_f_diff", "ic_f_cum72"]
    summary = {feat: _summarize(valid, feat) for feat in feats}

    fc = summary["ic_funding_cum"]
    verdict = (
        f"funding_cum 跨币 IC 均值={fc['mean']}（{fc['n']} 币，{fc['sign_frac']*100:.0f}% 同号）；"
        + ("★ 跨币稳健负向 alpha，P6 接入成立"
           if abs(fc["mean"]) > 0.05 and fc["sign_frac"] >= 0.6
           else "信号不够稳健，需谨慎")
    )
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "params": {"days": args.days, "min_rows": args.min_rows},
        "n_symbols": len(valid),
        "n_no_funding": len(no_funding),
        "summary": summary,
        "symbols": valid,
        "verdict": verdict,
        "elapsed_sec": round(time.time() - started, 1),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 资金费率 vs 止盈标签 跨币 IC（{len(valid)} 币，{len(no_funding)} 无资金费）===")
    for feat in feats:
        s = summary[feat]
        print(f"  {feat:16s} mean={s['mean']:+.4f}  n={s['n']}  same_sign={s['sign_frac']}")
    print(f"\n判定：{verdict}")
    print(f"耗时 {report['elapsed_sec']}s；报告 -> {args.output}")


if __name__ == "__main__":
    main()
