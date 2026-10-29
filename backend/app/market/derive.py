"""把上游周期聚合成下游周期并落库。

为什么需要这层：本地只有 1m 和 1h 的原始采集。1m 只覆盖 94 天，所以
`_candle_query` 的两条路径（1m 聚合 / native 回落）都无法产出 1 年跨度的 4h。
1h 却有 400 天以上的 native 行，据此派生出 4h 就能把可用跨度补齐到与 1h 相同。

派生行写入 `klines` 的 `interval=<target>`，此后对回测而言它就是普通的
native 行（`NATIVE_FALLBACK_INTERVALS` 已包含 4h），取数链路无需任何改动。
重跑幂等：按 (symbol, interval, open_time) upsert。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from ..models import Kline
from ..utils import interval_ms, now_ms

# 只有这些下游周期值得派生（1m 是原始层，其余由回测侧聚合即可）
DERIVABLE = {"5m": ("1m",), "15m": ("1m", "5m"), "1h": ("1m", "15m"), "4h": ("1h", "15m")}


def _insert_fn(db: Session):
    dialect = db.get_bind().dialect.name
    return pg_insert if dialect == "postgresql" else sqlite_insert


def aggregate_frame(
    frame: pd.DataFrame, target: str, source_step: int, min_fill: float = 0.75
) -> pd.DataFrame:
    """把单币的 OHLCV 行（含 open_time 毫秒）聚合到 target 周期。

    `min_fill` 是桶内最少根数占应有根数的比例：不足的桶整桶丢弃，避免用 3 根
    1h 拼出一个 volume 偏小、high/low 可能漏极值的 4h 桶。最后一个未闭合的桶
    也一定丢弃（它的 close 不是终点）。
    """
    if frame.empty:
        return frame.iloc[0:0]
    step = interval_ms(target)
    per_bucket = max(1, step // source_step)
    out = frame.copy()
    out["bucket"] = (out["open_time"] // step) * step
    grouped = out.sort_values("open_time").groupby("bucket", sort=True)
    rows = []
    for bucket, part in grouped:
        if len(part) < per_bucket * min_fill:
            continue
        if bucket + step > now_ms():
            continue
        rows.append(
            {
                "open_time": int(bucket),
                "close_time": int(bucket + step - 1),
                "open": float(part["open"].iloc[0]),
                "high": float(part["high"].max()),
                "low": float(part["low"].min()),
                "close": float(part["close"].iloc[-1]),
                "volume": float(part["volume"].sum()),
                "quote_volume": float(part["quote_volume"].sum()),
                "trades": int(part["trades"].sum()),
                "_bars": len(part),
            }
        )
    return pd.DataFrame(rows)


def derive_interval(
    db: Session,
    source: str = "1h",
    target: str = "4h",
    symbols: Iterable[str] | None = None,
    min_fill: float = 0.75,
    batch: int = 50,
) -> dict:
    """从 `source` 派生 `target` 并 upsert 落库，返回统计。"""
    src_step, dst_step = interval_ms(source), interval_ms(target)
    if dst_step % src_step:
        raise ValueError(f"{target} 不是 {source} 的整数倍，无法派生")

    symbol_sql = (
        select(Kline.symbol)
        .where(Kline.interval == source)
        .distinct()
        .order_by(Kline.symbol)
    )
    universe = [r[0] for r in db.execute(symbol_sql).all()]
    if symbols:
        wanted = {s.upper() for s in symbols}
        universe = [s for s in universe if s in wanted]
    if not universe:
        return {"source": source, "target": target, "symbols": 0, "rows": 0, "skipped": 0}

    insert_fn = _insert_fn(db)
    total_rows = 0
    stat = defaultdict(int)
    for start in range(0, len(universe), batch):
        chunk = universe[start : start + batch]
        raw = db.execute(
            select(
                Kline.symbol, Kline.open_time, Kline.close_time, Kline.open,
                Kline.high, Kline.low, Kline.close, Kline.volume,
                Kline.quote_volume, Kline.trades,
            )
            .where(Kline.interval == source, Kline.symbol.in_(chunk))
            .order_by(Kline.symbol, Kline.open_time)
        ).mappings().all()
        if not raw:
            continue
        frame = pd.DataFrame([dict(r) for r in raw])
        payload = []
        for symbol, part in frame.groupby("symbol", sort=True):
            agg = aggregate_frame(part, target, src_step, min_fill=min_fill)
            if agg.empty:
                stat["skipped"] += 1
                continue
            for row in agg.to_dict("records"):
                row.pop("_bars", None)
                payload.append({"symbol": symbol, "interval": target, **row})
            stat["symbols"] += 1
        for start_at in range(0, len(payload), 500):
            piece = payload[start_at : start_at + 500]
            stmt = insert_fn(Kline).values(piece)
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol", "interval", "open_time"],
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                    "quote_volume": stmt.excluded.quote_volume,
                    "close_time": stmt.excluded.close_time,
                    "trades": stmt.excluded.trades,
                },
            )
            db.execute(stmt)
        db.commit()
        total_rows += len(payload)

    return {
        "source": source,
        "target": target,
        "symbols": stat["symbols"],
        "rows": total_rows,
        "skipped": stat["skipped"],
        "min_fill": min_fill,
    }
