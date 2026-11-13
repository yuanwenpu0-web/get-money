"""Necessary overall-direction gate; it does not certify any other research gate."""
from __future__ import annotations

import math


def _identity(row: dict) -> tuple:
    return (row["key"], row["direction"], row["horizon_bars"], row["benchmark"])


def review_interval(rows: list[dict]) -> list[dict]:
    """Review one interval only, preserving original labels and numbers.

    Mean net excess is already direction-aligned. A positive overall mean is
    necessary, not sufficient: this does not certify significance or OOS edge.
    Missing/nonfinite comparators fail closed. Never borrow another direction,
    horizon, benchmark, interval, or an overall best cell to support a subgroup.
    """
    pooled = {}
    for row in rows:
        if row["scope"] != "all":
            continue
        key = _identity(row)
        if key in pooled:
            raise ValueError(f"duplicate overall coordinate: {key}")
        pooled[key] = row
    reviewed = []
    for row in rows:
        raw = row.get("verdict_cell") or row.get("verdict")
        if row["scope"] == "all" or raw != "candidate":
            continue
        base = pooled.get(_identity(row))
        value = None if base is None else base.get("mean_net_pct")
        finite = isinstance(value, (int, float)) and math.isfinite(value)
        if base is None:
            status, reason = "BLOCKED", "matching_pooled_row_missing"
        elif not finite:
            status, reason = "BLOCKED", "matching_pooled_net_unavailable"
        elif value <= 0:
            status, reason = "FAIL", "pooled_nonpositive_net"
        else:
            status, reason = "PASS", "positive_matching_pooled_net"
        reviewed.append({
            "key": row["key"], "scope": row["scope"],
            "direction": row["direction"], "horizon_bars": row["horizon_bars"],
            "benchmark": row["benchmark"], "cell_mean_net_pct": row.get("mean_net_pct"),
            "pooled_mean_net_pct": value if finite else None,
            "pooled_verdict": base.get("verdict") if base is not None else None,
            "original_verdict": raw,
            "original_verdict_base": row.get("verdict"),
            "original_verdict_cell": row.get("verdict_cell"),
            "review_verdict": raw if status == "PASS" else "exploratory",
            "direction_gate_status": status, "reason": reason,
        })
    return reviewed
