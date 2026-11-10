"""半年窗口的数据层：币池筛选、native 1h 帧、衍生品横截面分位。

所有查询走 make_engine(read_only=True)；横截面分位按「上一已结算桶」对齐，
避免把结算时刻才可知的费率／持仓量算成决策时刻可得（H8 新鲜度坑）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.db import make_engine  # noqa: E402

BUCKET_8H = 28_800_000
BUCKET_1H = 3_600_000
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")

CANDLE_SQL = """
SELECT open_time, open, high, low, close, volume
FROM klines
WHERE symbol=:sym AND interval=:iv AND open_time >= :a AND open_time < :b
ORDER BY open_time
"""

# native 1h 末端比 1m 落后约 5 天：用 1m 聚合补齐尾桶（只取分钟完整的桶），
# 与 data_service._candle_query 同源口径；native 行优先，不覆盖已有历史。
TAIL_SQL = """
WITH b AS (
  SELECT (open_time - open_time % :step) AS ot, count(*) AS n,
         min(open_time) AS first_t, max(open_time) AS last_t,
         max(high) AS high, min(low) AS low, sum(volume) AS volume
  FROM klines
  WHERE symbol=:sym AND interval='1m' AND open_time >= :a AND open_time < :b
  GROUP BY 1
)
SELECT b.ot AS open_time, f.open, b.high, b.low, l.close, b.volume
FROM b
JOIN klines f ON f.symbol=:sym AND f.interval='1m' AND f.open_time=b.first_t
JOIN klines l ON l.symbol=:sym AND l.interval='1m' AND l.open_time=b.last_t
WHERE b.n = :minutes AND b.first_t = b.ot
  AND b.last_t = b.ot + (b.n - 1) * 60000
ORDER BY b.ot
"""


def iso(ms: int | None) -> str:
    if ms is None:
        return ""
    return pd.Timestamp(ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M UTC")


def to_ms(index: pd.DatetimeIndex) -> np.ndarray:
    return ((index - EPOCH).total_seconds().to_numpy() * 1000).astype(np.int64)


def connect():
    return make_engine(read_only=True, null_pool=True)


def eligible_symbols(conn, interval: str, warm_ms: int, start_ms: int,
                     end_ms: int, min_coverage: float, limit: int = 0) -> tuple[list[str], list[dict]]:
    """按 (币) 分片：要求预热段起点前已有数据，且窗口内覆盖率达标。"""
    per_bar = 3_600_000 if interval == "1h" else 900_000
    expected = (end_ms - start_ms) // per_bar
    rows = conn.execute(text("""
        SELECT symbol, min(open_time) AS first_ms,
               count(*) FILTER (WHERE open_time >= :a AND open_time < :b) AS win_rows
        FROM klines WHERE interval=:iv GROUP BY symbol
    """), {"iv": interval, "a": start_ms, "b": end_ms}).mappings().all()
    ok, skipped = [], []
    for r in rows:
        sym = r["symbol"]
        cov = (r["win_rows"] or 0) / max(1, expected)
        if r["first_ms"] is None or r["first_ms"] > warm_ms:
            skipped.append({"symbol": sym, "reason": "no_warmup_history", "coverage": round(cov, 4)})
        elif cov < min_coverage:
            skipped.append({"symbol": sym, "reason": "window_coverage_below_threshold",
                            "coverage": round(cov, 4)})
        else:
            ok.append(sym)
    ok.sort()
    if limit:
        ok = ok[:limit]
    return ok, skipped


def load_frame(conn, symbol: str, interval: str, warm_ms: int, end_ms: int,
               fill_tail: bool = True) -> pd.DataFrame:
    rows = conn.execute(text(CANDLE_SQL), {"sym": symbol, "iv": interval,
                                           "a": warm_ms, "b": end_ms}).mappings().all()
    if fill_tail and rows and interval in ("1h", "15m"):
        step = 3_600_000 if interval == "1h" else 900_000
        tail_start = int(rows[-1]["open_time"]) + step
        if tail_start < end_ms:
            extra = conn.execute(text(TAIL_SQL), {
                "sym": symbol, "step": step, "a": tail_start, "b": end_ms,
                "minutes": step // 60_000,
            }).mappings().all()
            by_t = {int(r["open_time"]): r for r in rows}
            for r in extra:
                by_t.setdefault(int(r["open_time"]), r)
            rows = [by_t[k] for k in sorted(by_t)]
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.index.name = "time"
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def _xsec(conn, sql: str, value_col: str, bucket_ms: int, start: int, end: int):
    rows = conn.execute(text(sql), {"a": start, "b": end}).mappings().all()
    if not rows:
        return {}, {"first_ms": None, "last_ms": None, "rows": 0, "symbols": 0}
    df = pd.DataFrame(rows)
    df["bucket"] = (df["ts"] // bucket_ms) * bucket_ms
    pivot = df.pivot_table(index="bucket", columns="symbol", values=value_col, aggfunc="last")
    pct = pivot.rank(axis=1, pct=True).shift(1)      # 只用上一已结算桶
    raw = pivot.shift(1)
    out = {}
    for sym in pct.columns:
        s = pct[sym].dropna()
        r = raw[sym].reindex(s.index)
        out[sym] = {int(b): (float(v), float(r.loc[b]) if np.isfinite(r.loc[b]) else float("nan"))
                    for b, v in s.items()}
    return out, {"first_ms": int(df["ts"].min()), "last_ms": int(df["ts"].max()),
                 "rows": int(len(df)), "symbols": int(df["symbol"].nunique())}


def funding_xsec(conn, start: int, end: int):
    sql = ("SELECT symbol, funding_time AS ts, funding_rate FROM funding_rates "
           "WHERE funding_time >= :a AND funding_time < :b")
    return _xsec(conn, sql, "funding_rate", BUCKET_8H, start, end)


def sentiment_xsec(conn, start: int, end: int):
    sql = ("SELECT symbol, ts, oi_value FROM sentiment "
           "WHERE ts >= :a AND ts < :b AND oi_value IS NOT NULL")
    return _xsec(conn, sql, "oi_value", BUCKET_1H, start, end)


def align_xsec(index: pd.DatetimeIndex, mapping: dict, bucket_ms: int,
               with_rate: bool = False):
    """把 (bucket → (pct, rate)) / (bucket → pct) 对齐到 1h bar；缺值 NaN，不做前视填充。"""
    ms = to_ms(index)
    keys = (ms // bucket_ms) * bucket_ms
    if with_rate:
        pct = np.full(len(ms), np.nan)
        rate = np.full(len(ms), np.nan)
        for i, k in enumerate(keys):
            v = mapping.get(int(k))
            if v is not None:
                pct[i], rate[i] = v
        return pct, rate
    out = np.full(len(ms), np.nan)
    for i, k in enumerate(keys):
        v = mapping.get(int(k))
        out[i] = v[0] if v is not None else np.nan   # _xsec 统一存 (pct, raw)
    return out


def btc_change(conn, start_ms: int, end_ms: int) -> tuple[float | None, int]:
    """与回测同源帧的样本期基准；返回 (涨跌 %, 补齐后的窗口内 bar 数)。"""
    df = load_frame(conn, "BTCUSDT", "1h", start_ms, end_ms, fill_tail=True)
    win = df[(df.index >= pd.Timestamp(start_ms, unit="ms", tz="UTC"))]
    if len(win) < 2:
        return None, len(win)
    return round((float(win["close"].iloc[-1]) / float(win["close"].iloc[0]) - 1) * 100, 4), len(win)
