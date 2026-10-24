from __future__ import annotations

import json
from datetime import datetime, timezone

from .models import BacktestRun, ModelRecord, Order, Signal
from .trading.portfolio import position_unrealized


def dt_iso(value: datetime | None) -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def order_dict(order: Order) -> dict:
    unrealized = position_unrealized(order) if order.status == "OPEN" else 0.0
    return {
        "id": order.id,
        "symbol": order.symbol,
        "interval": order.interval,
        "side": order.side,
        "qty": round(order.qty, 6),
        "entry_price": order.entry_price,
        "mark_price": order.mark_price,
        "margin": round(order.margin, 4),
        "leverage": order.leverage,
        "sl": order.sl,
        "tp": order.tp,
        "status": order.status,
        "source": order.source,
        "strategy": order.strategy,
        "entry_fee": round(order.entry_fee, 4),
        "exit_fee": round(order.exit_fee, 4),
        "realized_pnl": round(order.realized_pnl, 4),
        "unrealized_pnl": round(unrealized, 4),
        "close_price": order.close_price,
        "close_reason": order.close_reason,
        "note": order.note,
        "created_at": dt_iso(order.created_at),
        "closed_at": dt_iso(order.closed_at),
    }


def signal_dict(signal: Signal) -> dict:
    try:
        reason = json.loads(signal.reason or "{}")
    except ValueError:
        reason = {}
    return {
        "id": signal.id,
        "symbol": signal.symbol,
        "interval": signal.interval,
        "strategy": signal.strategy,
        "ts": signal.ts,
        "time": dt_iso(datetime.fromtimestamp(signal.ts / 1000, tz=timezone.utc)),
        "price": signal.price,
        "direction": signal.direction,
        "score": signal.score,
        "prob_up": signal.prob_up,
        "sl": signal.sl,
        "tp": signal.tp,
        "leverage": signal.leverage,
        "position_pct": signal.position_pct,
        "reason": reason,
    }


def model_dict(record: ModelRecord, include_path: bool = False) -> dict:
    try:
        metrics = json.loads(record.metrics or "{}")
    except ValueError:
        metrics = {}
    result = {
        "id": record.id,
        "symbol": record.symbol,
        "interval": record.interval,
        "algo": record.algo,
        "horizon": record.horizon,
        "rows": record.rows,
        "status": record.status,
        "metrics": metrics,
        "train_start": record.train_start,
        "train_end": record.train_end,
        "created_at": dt_iso(record.created_at),
    }
    if include_path:
        result["path"] = record.path
    return result


def backtest_dict(run: BacktestRun, full: bool = False) -> dict:
    def _load(raw: str, default):
        try:
            return json.loads(raw or "null") or default
        except ValueError:
            return default

    result = {
        "id": run.id,
        "symbol": run.symbol,
        "interval": run.interval,
        "strategy": run.strategy,
        "algo": run.algo,
        "status": run.status,
        "error": run.error,
        "metrics": _load(run.metrics, {}),
        "created_at": dt_iso(run.created_at),
    }
    if full:
        result["params"] = _load(run.params, {})
        result["equity_curve"] = _load(run.equity_curve, [])
        result["trades"] = _load(run.trades, [])
    return result
