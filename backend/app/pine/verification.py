"""Compare exported closed-bar data and signals; does not compare broker fills."""
from __future__ import annotations

import math

import pandas as pd

_PRICES = ["open", "high", "low", "close", "volume"]
_SIGNALS = ["long_entry", "short_entry", "long_close", "short_close"]


def _timestamp(value):
    try:
        text = str(value).strip()
        # Numeric timestamps are UTC epoch milliseconds; ISO 8601 must include a timezone.
        if text.isdigit():
            return pd.to_datetime(int(text), unit="ms", utc=True)
        result = pd.Timestamp(text)
        if result.tzinfo is None:
            raise ValueError("时间必须带时区")
        return result.tz_convert("UTC")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"无效 K 线时间：{value}") from exc


def _bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise ValueError(f"信号必须为 true/false 或 1/0：{value}")


def compare_csv(reference, actual, numeric_columns=None, atol=1e-8, rtol=1e-6):
    """Compare same-source bars and signal columns, plus optional indicator columns.

    Both inputs are DataFrames parsed from CSV (or generated directly). Times must
    be UTC epoch milliseconds or timezone-qualified ISO 8601 strings. At most
    100 detailed mismatches are returned, while mismatch_count is unbounded.
    """
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("数值容差必须是非负有限数")
    indicators = list(numeric_columns or [])
    fields = list(dict.fromkeys(_PRICES + indicators))
    required = {"open_time", *_SIGNALS, *fields}
    frames = []
    for label, frame in (("TradingView", reference), ("Python", actual)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} 缺少列：{', '.join(sorted(missing))}")
        if frame.empty:
            raise ValueError(f"{label} 数据为空")
        frame = frame.copy()
        frame["open_time"] = frame["open_time"].map(_timestamp)
        if frame["open_time"].isna().any() or frame["open_time"].duplicated().any():
            raise ValueError(f"{label} K线时间必须唯一且有效")
        for column in fields:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        for column in _PRICES:
            if frame[column].isna().any() or not frame[column].map(math.isfinite).all():
                raise ValueError(f"{label} {column} 包含无效数值")
        for column in indicators:
            if not frame[column].dropna().map(math.isfinite).all():
                raise ValueError(f"{label} {column} 包含无效数值")
        for column in _SIGNALS:
            frame[column] = frame[column].map(_bool)
        frames.append(frame.set_index("open_time"))
    ref, actual = frames
    mismatches, count = [], 0

    def record(time, field, left, right):
        nonlocal count
        count += 1
        if len(mismatches) < 100:
            mismatches.append({"open_time": time.isoformat(), "field": field, "reference": left, "actual": right})

    for time in sorted(ref.index.union(actual.index)):
        if time not in ref.index or time not in actual.index:
            record(time, "bar", "present" if time in ref.index else None, "present" if time in actual.index else None)
            continue
        for field in fields:
            left, right = ref.at[time, field], actual.at[time, field]
            if pd.isna(left) and pd.isna(right):
                continue
            # Data provenance must match exactly; tolerance only applies to indicators.
            price = field in _PRICES
            if pd.isna(left) or pd.isna(right) or not math.isclose(left, right, rel_tol=0 if price else rtol, abs_tol=0 if price else atol):
                record(time, field, None if pd.isna(left) else float(left), None if pd.isna(right) else float(right))
        for field in _SIGNALS:
            left, right = bool(ref.at[time, field]), bool(actual.at[time, field])
            if left != right:
                record(time, field, left, right)
    return {"passed": count == 0, "rows": {"reference": len(ref), "actual": len(actual)},
            "mismatch_count": count, "mismatches": mismatches, "truncated": count > len(mismatches),
            "numeric_columns": fields, "atol": atol, "rtol": rtol}
