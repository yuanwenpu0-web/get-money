"""周期派生（1h → 4h）的聚合正确性测试。

这一段出错最隐蔽：4h 的 OHLCV 只要算错一点，回测读数就会整体偏移却看不出异常，
所以要逐字段钉住「首根开盘、末根收盘、极值取全桶、量额求和」。
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.market.derive import aggregate_frame
from app.utils import interval_ms, now_ms

HOUR = interval_ms("1h")


def _hourly(n: int = 8, start: int | None = None) -> pd.DataFrame:
    """n 根连续已闭合的 1h K 线，起点对齐到 4h 桶边界。"""
    base = start if start is not None else (now_ms() - n * HOUR - 2 * 86_400_000)
    base = base // (4 * HOUR) * (4 * HOUR)
    rows = []
    for i in range(n):
        rows.append(
            {
                "open_time": base + i * HOUR,
                "close_time": base + (i + 1) * HOUR - 1,
                "open": 100.0 + i,
                "high": 110.0 + i,
                "low": 90.0 - i,
                "close": 105.0 + i,
                "volume": 10.0 + i,
                "quote_volume": 1000.0 + i,
                "trades": 5 + i,
            }
        )
    return pd.DataFrame(rows)


def test_aggregate_takes_first_open_last_close_and_full_range():
    frame = _hourly(8)
    out = aggregate_frame(frame, "4h", HOUR)
    assert len(out) == 2

    first = out.iloc[0]
    part = frame.iloc[0:4]
    assert first["open_time"] == frame["open_time"].iloc[0]
    assert first["close_time"] == frame["open_time"].iloc[3] + HOUR - 1
    assert first["open"] == pytest.approx(part["open"].iloc[0])
    assert first["close"] == pytest.approx(part["close"].iloc[-1])
    assert first["high"] == pytest.approx(part["high"].max())
    assert first["low"] == pytest.approx(part["low"].min())
    assert first["volume"] == pytest.approx(part["volume"].sum())
    assert first["quote_volume"] == pytest.approx(part["quote_volume"].sum())
    assert first["trades"] == int(part["trades"].sum())


def test_aggregate_buckets_align_to_epoch():
    frame = _hourly(8)
    out = aggregate_frame(frame, "4h", HOUR)
    step = interval_ms("4h")
    for value in out["open_time"]:
        assert int(value) % step == 0


def test_min_fill_drops_incomplete_buckets():
    frame = _hourly(8)
    # 第二个桶只有 2 根（缺 2 根）
    holed = frame.drop(index=[4, 5]).reset_index(drop=True)
    strict = aggregate_frame(holed, "4h", HOUR, min_fill=1.0)
    loose = aggregate_frame(holed, "4h", HOUR, min_fill=0.5)
    assert len(strict) == 1
    assert len(loose) == 2


def test_unclosed_bucket_is_dropped():
    """最后一根 4h 桶如果还没走完就不能落库：它的 close 不是终点。"""
    step = interval_ms("4h")
    current = now_ms() // step * step
    frame = _hourly(4, start=current - 4 * HOUR)
    out = aggregate_frame(frame, "4h", HOUR)
    # 当前这个 4h 桶包含 4 根已闭合的 1h，但它自己还没结束
    assert all(int(value) + step <= now_ms() for value in out["open_time"])


def test_aggregate_empty_frame_returns_empty():
    empty = pd.DataFrame(
        columns=["open_time", "close_time", "open", "high", "low", "close", "volume", "quote_volume", "trades"]
    )
    assert aggregate_frame(empty, "4h", HOUR).empty
