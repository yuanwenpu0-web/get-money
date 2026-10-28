"""P3 探针：横截面相对强弱能否预测 `止盈` 标签（决定 P3 是否值得做）。

只读脚本，不改训练链路。逻辑：
- 加载一批 1h 历史足够长的币，对齐到统一时间索引；
- 每根算各币 20 根动量 ret_20，再做横截面 z-score：
      z[sym,t] = (ret_20[sym,t] - mean_t) / std_t   （同 t 跨所有币）
- 目标 = 该币 triple-barrier +1（止盈）标签（与实盘出场对齐）；
- IC = z[sym,t] 与 (label==+1) 的点二列相关，逐币计算后取均值；
- 对照：单币自身 ret_20（时序动量，无横截面）的 IC，看横截面是否增量有效。

判定：若横截面 z 的 IC 均值显著 > 单币 ret_20 的 IC，则 P3（横截面强弱）方向正确。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/
from app.db import make_engine
from app.features.indicators import _triple_barrier_label, build_features
from app.market.data_service import MarketDataService
from app.strategy.rules import resolve_params
from app.utils import now_ms


def eligible_symbols(db: Session, min_span_days: int = 120, limit: int = 40) -> list[str]:
    rows = db.execute(text("""
        select symbol from klines where interval='1h' group by symbol
        having (max(open_time)-min(open_time))/86400000.0 >= :d
        order by count(*) desc limit :lim
    """), {"d": min_span_days, "lim": limit}).all()
    return [r[0] for r in rows]


def load_close_panel(db: Session, symbols: list[str], start_ms: int, end_ms: int) -> pd.DataFrame:
    svc = MarketDataService(db)
    frames = {}
    for sym in symbols:
        f = svc.load_frame(sym, "1h", start_ms=start_ms, end_ms=end_ms)
        if not f.empty:
            frames[sym] = f["close"]
    if not frames:
        return pd.DataFrame()
    panel = pd.DataFrame(frames)
    # 对齐到共同时间索引（取交集），避免不同币缺口错位
    return panel.dropna(how="any")


def pointbiserial(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 50 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--min-span-days", type=int, default=120)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--momentum-bars", type=int, default=20)
    ap.add_argument("--output", default="docs/probe-cross-section.json")
    args = ap.parse_args()

    engine = make_engine(read_only=True)
    params = resolve_params("ml_ensemble", {}, {})
    barrier = {"sl_atr_mult": params["sl_atr_mult"], "tp_atr_mult": params["tp_atr_mult"],
               "max_hold_bars": params["max_hold_bars"], "horizon": 5}
    end = now_ms()
    start = end - args.days * 86_400_000

    with Session(engine) as db:
        syms = eligible_symbols(db, args.min_span_days, args.limit)
        print(f"[probe] 候选币 {len(syms)} 个")
        panel = load_close_panel(db, syms, start, end)
    if panel.empty or panel.shape[1] < 5:
        print("[probe] 有效面板不足，无法计算横截面")
        return
    print(f"[probe] 对齐后面板 {panel.shape[0]} 根 × {panel.shape[1]} 币")

    ret = panel.pct_change(args.momentum_bars)
    # 横截面 z-score（每根跨币）。用显式 numpy 计算避开 DataFrame-Series 广播对齐陷阱。
    ret_arr = ret.to_numpy(dtype=float)
    mean_r = np.nanmean(ret_arr, axis=1, keepdims=True)
    std_r = np.nanstd(ret_arr, axis=1, keepdims=True)
    std_r[std_r == 0] = np.nan
    xs_arr = (ret_arr - mean_r) / std_r
    xs_z = pd.DataFrame(xs_arr, index=ret.index, columns=ret.columns)

    records = []
    ic_xs_list, ic_ts_list, ic_obv_list, ic_vp_list = [], [], [], []
    for i, sym in enumerate(panel.columns):
        close = panel[sym]
        # 重新构造含 high/low/atr_pct/volume 的单币帧以生成 triple-barrier 标签与量价特征
        with Session(engine) as db:
            f = MarketDataService(db).load_frame(sym, "1h", start_ms=start, end_ms=end)
        if f.empty:
            continue
        feat = build_features(f, with_label=False, horizon=5)
        lbl = _triple_barrier_label(feat, feat["close"], barrier).reindex(close.index)
        target = (lbl == 1).astype(float).to_numpy()
        z = xs_z[sym].to_numpy()
        ts = ret[sym].to_numpy()
        # 量价背离：OBV 累积斜率 vs 价；close 与 volume 的 20 根滚动相关
        c, v = f["close"], f["volume"]
        r1 = c.pct_change(1).fillna(0.0)
        obv = (np.sign(r1) * v).cumsum()
        obv_slope = obv.pct_change(args.momentum_bars).reindex(close.index).to_numpy()
        vp = c.rolling(args.momentum_bars).corr(v).reindex(close.index).to_numpy()
        ic_xs = pointbiserial(z, target)
        ic_ts = pointbiserial(ts, target)
        ic_obv = pointbiserial(obv_slope, target)
        ic_vp = pointbiserial(vp, target)
        if i == 0:
            m = np.isfinite(z) & np.isfinite(target)
            print(f"[debug] {sym}: rows={len(z)} valid={int(m.sum())} "
                  f"up={(target == 1).sum()} ic_xs={ic_xs:.3f} ic_ts={ic_ts:.3f} "
                  f"ic_obv={ic_obv:.3f} ic_vp={ic_vp:.3f}")
        for lst, val in ((ic_xs_list, ic_xs), (ic_ts_list, ic_ts),
                         (ic_obv_list, ic_obv), (ic_vp_list, ic_vp)):
            if np.isfinite(val):
                lst.append(val)
        records.append({"symbol": sym, "ic_cross_section": ic_xs, "ic_time_series": ic_ts,
                        "ic_obv_slope": ic_obv, "ic_volume_price_corr": ic_vp,
                        "up_ratio": float((lbl == 1).mean())})

    def mean(lst):
        return float(np.nanmean(lst)) if lst else None

    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "params": vars(args),
        "panel_bars": int(panel.shape[0]),
        "panel_symbols": int(panel.shape[1]),
        "ic_time_series_mean": mean(ic_ts_list),
        "ic_cross_section_mean": mean(ic_xs_list),
        "ic_obv_slope_mean": mean(ic_obv_list),
        "ic_volume_price_corr_mean": mean(ic_vp_list),
        "symbols": records,
        "verdict": (
            "三类候选特征 IC 均接近 0（<0.02）→ P3 单靠现有频率/数据源的特征扩展翻不了盘；"
            "按路线图 P1 gate，应换频率或数据源，而非继续堆特征"
            if (mean(ic_ts_list) or 0) < 0.02 and (mean(ic_xs_list) or 0) < 0.02
            and (mean(ic_obv_list) or 0) < 0.02
            else "存在 IC>0.02 的候选特征 → 该方向值得正式接入 P3"
        ),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"\nIC 时序动量={report['ic_time_series_mean']} | 横截面排名={report['ic_cross_section_mean']} "
          f"| OBV斜率={report['ic_obv_slope_mean']} | 量价相关={report['ic_volume_price_corr_mean']}")
    print(f"判定：{report['verdict']}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
