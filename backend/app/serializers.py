from __future__ import annotations

import json
from datetime import datetime, timezone

from .models import BacktestRun, Order, Signal
from .trading.portfolio import position_unrealized


def dt_iso(value: datetime | None) -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def order_dict(order: Order, *, include_snapshot: bool = False) -> dict:
    unrealized = position_unrealized(order) if order.status == "OPEN" else 0.0
    try:
        note = json.loads(order.note or "{}")
    except ValueError:
        note = {}
    if not isinstance(note, dict):
        note = {}
    result = {
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
        "pool": order.pool or "",
        "binding_id": order.binding_id,
        "entry_fee": round(order.entry_fee, 4),
        "exit_fee": round(order.exit_fee, 4),
        "realized_pnl": round(order.realized_pnl, 4),
        "unrealized_pnl": round(unrealized, 4),
        "close_price": order.close_price,
        "close_reason": order.close_reason,
        "note": order.note,
        # 交易所侧止盈止损状态与盈亏来源：前端据此区分「交易所真实成交」与「本地估算」
        # 新订单写独立列；历史订单列为空时回落 note，保证老数据展示/对账不断档
        "exchange_order_id": order.exchange_order_id or note.get("exchange_order_id"),
        "sl_order_id": note.get("sl_order_id"),
        "tp_order_id": note.get("tp_order_id"),
        "sl_status": note.get("sl_status", ""),
        "tp_status": note.get("tp_status", ""),
        "pnl_source": note.get("pnl_source", "estimated" if order.source == "live" else "ledger"),
        "created_at": dt_iso(order.created_at),
        "closed_at": dt_iso(order.closed_at),
    }
    if include_snapshot:
        from .strategy.bindings import decode_order_params

        result["params_snapshot"] = None
        result["params_snapshot_status"] = "legacy"
        if order.params_snapshot is not None:
            try:
                result["params_snapshot"] = decode_order_params(
                    order.strategy, order.interval, order.params_snapshot
                )
                result["params_snapshot_status"] = "valid"
            except (ValueError, KeyError):
                result["params_snapshot_status"] = "invalid"
    return result


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


def backtest_dict(run: BacktestRun, full: bool = False, include_trades: bool = True) -> dict:
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
        if include_trades:
            result["trades"] = _load(run.trades, [])
    return result
