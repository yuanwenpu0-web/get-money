"""趋势反转 live 订单：信号 bar 重放 + 入场延迟 + 持仓 MFE/MAE 诊断（只读）。

工程诊断，非 edge 判定：不碰策略/绑定/参数，不做参数寻优。
回答两件事：① 方向与信号 bar 是否自洽、成交价偏离信号价多少；
② 持仓期内最大浮盈（MFE）与最终净盈亏的关系。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.db import make_engine  # noqa: E402
from app.features.indicators import build_features  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.strategy import specs as _specs  # noqa: E402,F401  导入即注册
from app.strategy.registry import PrepareContext, get_spec  # noqa: E402
from app.utils import interval_ms  # noqa: E402

ORDER_SQL = """
SELECT id, symbol, interval, side, qty, entry_price, close_price, sl, tp, margin,
       leverage, realized_pnl, close_reason, created_at, closed_at, params_snapshot
FROM orders
WHERE strategy = 'supertrend_current' AND source = 'live' AND status = 'CLOSED'
  AND entry_price > 0 AND close_price > 0 AND closed_at IS NOT NULL
ORDER BY created_at, id
"""


def to_ms(dt) -> int:
    return int(dt.timestamp() * 1000)


def closed_frame(service, symbol, interval, limit, as_of_ms):
    """复刻 engine._prepare_features 的取数与「只留已收盘桶」过滤，但把 now 换成 as_of。"""
    frame = service.load_frame(symbol, interval, limit=limit, end_ms=as_of_ms)
    if frame.empty:
        return frame
    step = interval_ms(interval)
    frame = frame.loc[(frame.index.as_unit("ms").asi8 + step) <= as_of_ms]
    if len(frame) < 120:
        return frame.iloc[0:0]
    disc = frame.index.to_series().diff().dt.total_seconds().mul(1000).ne(step)
    return frame.loc[disc.cumsum().eq(disc.cumsum().iloc[-1])]


def replay_signal(service, spec, order, params):
    """在下单时刻重放策略，取最后一根有方向的 bar 作为信号 bar。"""
    as_of = to_ms(order["created_at"])
    limit = spec.lookback(params)
    frame = closed_frame(service, order["symbol"], order["interval"], limit, as_of)
    if len(frame) < 120:
        return None, "bars_short"
    data = build_features(frame, with_label=False)
    data = spec.prepare(data, params, PrepareContext(symbol=order["symbol"], service=service))
    sig = spec.signals(data, params)
    step = interval_ms(order["interval"])
    dirs = pd.to_numeric(sig["direction"], errors="coerce").fillna(0).to_numpy()
    idx = np.nonzero(dirs != 0)[0]
    if len(idx) == 0:
        return None, "no_signal_in_window"
    j = int(idx[-1])
    bar_ts = int(data.index[j].timestamp() * 1000)
    want = 1 if order["side"] == "LONG" else -1
    return {
        "sig_bar_ts": bar_ts,
        "sig_close": float(data["close"].iloc[j]),
        "sig_dir": int(dirs[j]),
        "dir_match": bool(int(dirs[j]) == want),
        "latency_min": (as_of - (bar_ts + step)) / 60000.0,
        "bars_since_signal": int(len(data) - 1 - j),
        "sig_sl_pct": float(sig["sl_pct"].iloc[j]) if "sl_pct" in sig else None,
        "sig_tp_pct": float(sig["tp_pct"].iloc[j]) if "tp_pct" in sig else None,
        "block_reason": str(sig["block_reason"].iloc[j]),
    }, None


def hold_path(service, order):
    """持仓期 1m 路径：MFE/MAE 用实际成交价与实际止损距离（R_fill）归一。"""
    start, end = to_ms(order["created_at"]), to_ms(order["closed_at"])
    bars = service.load_frame(order["symbol"], "1m", limit=None, start_ms=start, end_ms=end)
    if bars.empty:
        return None, "no_1m"
    side = 1.0 if order["side"] == "LONG" else -1.0
    entry, sl = float(order["entry_price"]), float(order["sl"])
    risk = side * (entry - sl)
    if risk <= 0:
        return None, "bad_sl_side"
    hi = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    fav = side * (hi - entry) if side > 0 else side * (lo - entry)
    adv = side * (lo - entry) if side > 0 else side * (hi - entry)
    mfe, mae = float(np.nanmax(fav)), float(np.nanmin(adv))
    k = int(np.nanargmax(fav))
    return {
        "n_1m": int(len(bars)),
        "mfe_pct": mfe / entry * 100.0,
        "mae_pct": mae / entry * 100.0,
        "mfe_R": mfe / risk,
        "mae_R": mae / risk,
        "mfe_minute": int(k),
        "mfe_ts": int(bars.index[k].timestamp() * 1000),
        "risk_fill_pct": risk / entry * 100.0,
    }, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/trend_reversal_live_diag.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    engine = make_engine(read_only=True)
    spec = get_spec("supertrend_current")
    records, skipped = [], []
    with Session(engine) as db:
        service = MarketDataService(db)
        orders = [dict(r) for r in db.execute(text(ORDER_SQL)).mappings()]
        if args.limit:
            orders = orders[: args.limit]
        for o in orders:
            snap = json.loads(o["params_snapshot"] or "{}")
            params = snap.get("params") or {}
            if not params:
                skipped.append({"id": o["id"], "symbol": o["symbol"], "why": "no_params_snapshot"})
                continue
            info, why = replay_signal(service, spec, o, params)
            path, pwhy = hold_path(service, o)
            if info is None and path is None:
                skipped.append({"id": o["id"], "symbol": o["symbol"], "why": why or pwhy})
                continue
            side = 1.0 if o["side"] == "LONG" else -1.0
            entry, sl, tp = float(o["entry_price"]), float(o["sl"]), float(o["tp"])
            rec = {
                "id": o["id"], "symbol": o["symbol"], "side": o["side"],
                "close_reason": o["close_reason"],
                "entry": entry, "sl": sl, "tp": tp, "close": float(o["close_price"]),
                "realized_pnl": float(o["realized_pnl"]), "margin": float(o["margin"] or 0),
                "hold_min": (to_ms(o["closed_at"]) - to_ms(o["created_at"])) / 60000.0,
                "sl_fill_pct": side * (entry - sl) / entry * 100.0,
                "tp_fill_pct": side * (tp - entry) / entry * 100.0,
                "net_pct": side * (float(o["close_price"]) - entry) / entry * 100.0,
                "skip_signal": why, "skip_path": pwhy,
            }
            rec["rr_fill"] = rec["tp_fill_pct"] / rec["sl_fill_pct"] if rec["sl_fill_pct"] > 0 else None
            if info:
                rec.update(info)
                rec["entry_dev_pct"] = side * (entry - info["sig_close"]) / info["sig_close"] * 100.0
            if path:
                rec.update(path)
            records.append(rec)

    payload = {"n_orders": len(orders), "n_records": len(records),
               "skipped": skipped, "records": records}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}  records={len(records)} skipped={len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())