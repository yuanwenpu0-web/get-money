"""换频率探针：同一批币在不同采样频率下，时序动量 / 横截面 对 `止盈` 标签的 IC 是否变化。

这是路线图 P1 gate 之后最该验证的一步——之前 P3 探针在 1h 上证明价格序列 IC≈0，
但没排除"只是频率不对"。本脚本在 1m / 15m / 1h 三档上重测，看 IC 是否随频率系统性变化。

与 probe_cross_section.py 的区别：
- 后者只在 1h、含量价结构特征，结论"特征层面无增量"；
- 本脚本扫频率，回答"频率层面是否有救"。

实现要点（不影响生产代码）：
- 内置向量化 triple-barrier 标签（宽松版，无 300 根封顶），标签窗口按**日历时间**对齐，
  保证跨频率"止盈"语义一致：1m=7200根(5天) / 15m=480根 / 1h=120根，动量均 20 小时。
- 横截面 z 在每段时间窗内独立计算（无未来信息泄漏）。
- 只跑 BTC/ETH（库里唯一在三频率都 ≥200 天的币），保证可比。

判定：若低频（1h/15m）IC 显著高于高频（1m），说明往低频走有趋势 alpha；
若三档都≈0，则确认"现货价格序列本身无方向性 alpha"，应换数据源而非频率。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/
from app.db import make_engine
from app.features.indicators import build_features
from app.market.data_service import MarketDataService
from app.strategy.rules import resolve_params
from app.utils import now_ms

INTERVAL_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}


def tb_label_vectorized(close, high, low, atr_pct, sl_m, tp_m, H) -> pd.Series:
    """向量化 triple-barrier（近似版，仅供探针）。

    用未来 H 根窗口的 high 最大值 / low 最小值相对当前价的触达判定。
    '先触 tp 还是 sl' 用 sl 阈值更小（更易触）近似优先标 -1；仅方向性探针用，非生产逻辑。
    """
    n = len(close)
    idx = close.index
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    ap = np.asarray(atr_pct, dtype=float)
    labels = np.full(n, np.nan)
    valid = np.isfinite(ap) & (ap > 0) & (close > 0)
    c = np.where(valid, close, np.nan)
    tp_price = c * (1.0 + tp_m * np.where(valid, ap, np.nan))
    sl_price = c * (1.0 - sl_m * np.where(valid, ap, np.nan))
    # 未来 H 根（i+1 .. i+H）最高/最低：反向 rolling 才是真正向未来看
    fwd_max = pd.Series(high).iloc[::-1].rolling(H, min_periods=1).max().iloc[::-1].to_numpy()
    fwd_min = pd.Series(low).iloc[::-1].rolling(H, min_periods=1).min().iloc[::-1].to_numpy()
    hit_up = np.isfinite(tp_price) & (fwd_max >= tp_price)
    hit_dn = np.isfinite(sl_price) & (fwd_min <= sl_price)
    # 同时触达时 sl 阈值更小 → 视为先触 sl（标 -1）
    labels[hit_up & ~hit_dn] = 1.0
    labels[hit_dn] = -1.0
    labels[~hit_up & ~hit_dn] = 0.0
    return pd.Series(labels, index=idx)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 50 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def freq_bars(interval: str, label_days: int, momentum_hours: int) -> tuple[int, int]:
    mins = INTERVAL_MIN[interval]
    H = max(1, int(round(label_days * 24 * 60 / mins)))
    M = max(1, int(round(momentum_hours * 60 / mins)))
    return H, M


def probe_freq(db: Session, symbols: list[str], interval: str, label_days: int,
               momentum_hours: int, seg_days: int) -> dict:
    params = resolve_params("ml_ensemble", {}, {})
    sl_m, tp_m = params["sl_atr_mult"], params["tp_atr_mult"]
    H, M = freq_bars(interval, label_days, momentum_hours)
    end = now_ms()
    start = end - 400 * 86_400_000  # 足够覆盖各频率最大历史

    closes, labels = {}, {}
    for sym in symbols:
        f = MarketDataService(db).load_frame(sym, interval, start_ms=start, end_ms=end)
        if f.empty:
            continue
        feat = build_features(f, with_label=False)
        lbl = tb_label_vectorized(
            feat["close"], feat["high"], feat["low"], feat["atr_pct"], sl_m, tp_m, H
        )
        closes[sym] = feat["close"]
        labels[sym] = lbl
    if not closes:
        return {"interval": interval, "status": "no_data"}

    panel = pd.DataFrame(closes).dropna(how="any")
    lbl_panel = pd.DataFrame(labels).reindex(panel.index).dropna(how="any")
    panel = panel.reindex(lbl_panel.index)
    if panel.shape[0] < 100:
        return {"interval": interval, "status": "too_short", "bars": panel.shape[0]}

    mins = INTERVAL_MIN[interval]
    seg_bars = max(1, int(round(seg_days * 24 * 60 / mins)))
    actual_days = panel.shape[0] * mins / 1440.0

    # 时序动量
    ret = panel.pct_change(M)

    ic_ts_seg, ic_xs_seg = [], []
    # 分段（无重叠）降方差
    n = panel.shape[0]
    for s0 in range(0, n, seg_bars):
        s1 = min(n, s0 + seg_bars)
        if s1 - s0 < max(M, 30):
            continue
        seg_ret = ret.iloc[s0:s1]
        seg_lbl = lbl_panel.iloc[s0:s1]
        if seg_lbl.shape[1] == 0:
            continue
        # 横截面 z（段内跨币）
        r = seg_ret.to_numpy(dtype=float)
        mu = np.nanmean(r, axis=1, keepdims=True)
        sd = np.nanstd(r, axis=1, keepdims=True)
        sd[sd == 0] = np.nan
        z = (r - mu) / sd
        for j, sym in enumerate(seg_lbl.columns):
            tgt = (seg_lbl[sym] == 1).astype(float).to_numpy()
            ic_ts_seg.append(corr(seg_ret[sym].to_numpy(), tgt))
            ic_xs_seg.append(corr(z[:, j], tgt))

    def agg(lst):
        lst = [x for x in lst if np.isfinite(x)]
        if not lst:
            return None
        return {
            "mean": round(float(np.nanmean(lst)), 4),
            "median": round(float(np.nanmedian(lst)), 4),
            "pos_rate": round(float(np.mean([x > 0 for x in lst])), 3),
            "n": len(lst),
        }

    return {
        "interval": interval,
        "bars": int(panel.shape[0]),
        "actual_days": round(actual_days, 1),
        "symbols": list(panel.columns),
        "label_window_bars": H,
        "momentum_bars": M,
        "ic_time_series": agg(ic_ts_seg),
        "ic_cross_section": agg(ic_xs_seg),
        "up_ratio": round(float((lbl_panel.to_numpy() == 1).mean()), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    ap.add_argument("--intervals", default="1m,15m,1h")
    ap.add_argument("--label-days", type=int, default=5)
    ap.add_argument("--momentum-hours", type=int, default=20)
    ap.add_argument("--seg-days", type=int, default=30)
    ap.add_argument("--output", default="docs/probe-frequency.json")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    intervals = [s.strip() for s in args.intervals.split(",") if s.strip()]

    engine = make_engine(read_only=True)
    results = []
    with Session(engine) as db:
        for iv in intervals:
            print(f"[probe] {iv} ...", flush=True)
            results.append(probe_freq(db, symbols, iv, args.label_days,
                                       args.momentum_hours, args.seg_days))

    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "params": vars(args),
        "symbols": symbols,
        "intervals": results,
        "verdict": "",
    }
    # 判定：低频 IC 是否高于高频
    by_iv = {r["interval"]: r for r in results if "ic_time_series" in r}
    if by_iv:
        ts_means = {iv: (r["ic_time_series"] or {}).get("mean") for iv, r in by_iv.items()}
        best = max(ts_means, key=lambda k: (ts_means[k] or -9))
        best_v = ts_means[best]
        agg_best = by_iv[best]["ic_time_series"] or {}
        pos_rate = agg_best.get("pos_rate")
        n_seg = agg_best.get("n")
        if best_v is None:
            report["verdict"] = "无有效 IC"
        elif abs(best_v) > 0.05 and (pos_rate or 0) > 0.6 and (n_seg or 0) >= 10:
            report["verdict"] = (
                f"时序动量 IC：{' / '.join(f'{iv}={ts_means[iv]}' for iv in intervals if iv in ts_means)}。"
                f" {best} 频率 IC 最高（{best_v}，正率{pos_rate}、{n_seg}段），"
                f"存在可进一步验证的趋势信号，值得往该频率正式做 P3"
            )
        elif abs(best_v) > 0.02:
            report["verdict"] = (
                f"时序动量 IC：{' / '.join(f'{iv}={ts_means[iv]}' for iv in intervals if iv in ts_means)}。"
                f" {best} 频率 IC 微弱正向（{best_v}，但正率{pos_rate}、仅{n_seg}段，不显著），"
                f"不足以定论；先延长 1m 历史或换数据源更划算"
            )
        else:
            report["verdict"] = (
                f"时序动量 IC：{' / '.join(f'{iv}={ts_means[iv]}' for iv in intervals if iv in ts_means)}。"
                f" 全频率 |IC|<0.02，现货价格序列无方向性 alpha，应换数据源（资金费率/订单流/链上）而非频率"
            )

    with open(args.output, "w", encoding="utf-8") as h:
        json.dump(report, h, ensure_ascii=False, indent=2)
    print("\n=== 各频率 IC（时序动量 / 横截面）===")
    for r in results:
        if "ic_time_series" not in r:
            print(f"  {r['interval']}: {r.get('status')}")
            continue
        ts = r["ic_time_series"] or {}
        xs = r["ic_cross_section"] or {}
        print(f"  {r['interval']:4s} 天数={r['actual_days']:6.1f} 样本段={ts.get('n')}  "
              f"时序IC={ts.get('mean')}(正率{ts.get('pos_rate')})  横截面IC={xs.get('mean')}  "
              f"止盈比={r['up_ratio']}")
    print(f"\n判定：{report['verdict']}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
