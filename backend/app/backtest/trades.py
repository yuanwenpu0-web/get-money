"""Position-level statistics for executions that may have multiple exits."""

from __future__ import annotations


def aggregate_positions(trades: list[dict]) -> list[dict]:
    """Combine exit fills belonging to one position into a single return sample.

    A fill's return_pct is relative to its own released margin. Summing those
    percentages would overweight small partial fills, so compute the return
    from total net P&L divided by the total original margin. Old standalone
    records without position_id retain their existing return_pct unchanged.
    """
    positions = {}
    for index, trade in enumerate(trades):
        position_id = trade.get("position_id")
        key = ("position", position_id) if position_id is not None else ("trade", index)
        if key not in positions:
            positions[key] = dict(trade)
            continue
        combined = positions[key]
        combined["pnl"] = float(combined.get("pnl", 0)) + float(trade.get("pnl", 0))
        combined["margin"] = float(combined.get("margin", 0)) + float(trade.get("margin", 0))
        combined["fees"] = float(combined.get("fees", 0)) + float(trade.get("fees", 0))
        combined["size"] = float(combined.get("size", 0)) + float(trade.get("size", 0))
        combined["exit_time"] = trade.get("exit_time", combined.get("exit_time"))
        combined["status"] = trade.get("status", combined.get("status"))
        if combined["margin"]:
            combined["return_pct"] = combined["pnl"] / combined["margin"] * 100
    return list(positions.values())
