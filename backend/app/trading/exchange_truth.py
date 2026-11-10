"""「以币安为准」：账户同步、收益覆盖、逐笔真实成交回填、环境切换基准重置。

设计要点：
- 权益/可用/未实现/已实现盈亏来自交易所账户，本地账本只作对账参考；
- 逐笔订单的成交价与手续费来自 /fapi/v1/userTrades，取不到时显式标记 estimated；
- demo 与正式环境共用本模块，切换环境只需重置收益基准，不改统计代码。
"""
from __future__ import annotations

import json

from sqlalchemy.orm import Session

from ..models import ExchangeAccountSnapshot, JobLog, Order
from ..state import all_state, set_state
from ..utils import dt_ms, now_ms, round_price
from . import exchange_gateway as gw

CACHE_KEY = "exchange_account_latest"
SYNC_MIN_INTERVAL_MS = 10_000
PNL_ESTIMATED = "estimated"
PNL_BINANCE = "binance"


def cached(db: Session, state: dict | None = None) -> dict:
    value = (state or all_state(db)).get(CACHE_KEY) or {}
    return value if isinstance(value, dict) else {}


def sync_account(db: Session, force: bool = False) -> dict:
    """每周期同步一次币安账户；失败时保留上次快照并写 warning，不中断交易周期。"""
    state = all_state(db)
    old = cached(db, state)
    if state.get("execution_mode") != "live" or not state.get("exchange_truth_enabled", True):
        return old
    if not force and now_ms() - int(old.get("ts") or 0) < SYNC_MIN_INTERVAL_MS:
        return old
    # 无凭证时直接跳过：既避免无意义的签名请求，也让离线/单测环境不会被网络阻塞
    from .live import load_credentials

    key, secret = load_credentials(db)
    if not key or not secret:
        return old
    try:
        client = gw.client_for(db)
        acc = gw.fetch_account(client)
        income = gw.fetch_income(client, int(state.get("exchange_income_since") or 0))
        env = gw.env_of(db)
        initial = float(state.get("initial_capital") or 0)
        ts = now_ms()
        payload = {
            "ts": ts, "env": env,
            "wallet_balance": round(acc["wallet_balance"], 4),
            "available_balance": round(acc["available_balance"], 4),
            "margin_balance": round(acc["margin_balance"] or acc["wallet_balance"] + acc["unrealized_pnl"], 4),
            "unrealized_pnl": round(acc["unrealized_pnl"], 4),
            "position_margin": round(acc["position_margin"], 4),
            # 已结算进钱包的部分 = 钱包余额 − 收益基准，天然含手续费与资金费
            "realized_pnl": round(acc["wallet_balance"] - initial, 4),
            "commission": round(income["commission"], 4),
            "funding_fee": round(income["funding_fee"], 4),
            "income_rows": income["rows"],
            "income_available": income["available"],
            "account_source": acc["source"],
        }
        db.add(ExchangeAccountSnapshot(
            ts=ts, env=env, wallet_balance=payload["wallet_balance"],
            available_balance=payload["available_balance"], margin_balance=payload["margin_balance"],
            unrealized_pnl=payload["unrealized_pnl"], position_margin=payload["position_margin"],
            realized_pnl=payload["realized_pnl"], commission=payload["commission"],
            funding_fee=payload["funding_fee"],
        ))
        db.commit()
        set_state(db, CACHE_KEY, payload)
        return payload
    except Exception as exc:  # noqa: BLE001 - 账户同步失败不应阻断巡检
        db.rollback()
        db.add(JobLog(job="exchange_sync", level="warning", message=f"币安账户同步失败: {exc}"[:4000]))
        db.commit()
        return old


def overlay(result: dict, state: dict) -> dict:
    """把本地账本快照覆盖成币安账户口径，同时保留本地值用于对账。"""
    acc = cached(None, state)
    if not acc or state.get("execution_mode") != "live" or not state.get("exchange_truth_enabled", True):
        return {**result, "pnl_source": "ledger", "exchange_env": gw.DEMO if state.get("testnet", True) else gw.REAL}
    initial = float(state.get("initial_capital") or 0)
    equity = float(acc.get("margin_balance") or 0)
    # live 下本地 cash 不再随订单增减，「本地权益」已失去可比性；
    # 对账改为逐笔订单盈亏 vs 币安账户盈亏（含手续费/资金费）的口径差。
    ledger_realized = float(result.get("realized_pnl") or 0)
    ledger_unrealized = float(result.get("unrealized_pnl") or 0)
    exchange_net = float(acc.get("realized_pnl") or 0) + float(acc.get("unrealized_pnl") or 0)
    result.update({
        "cash": round(acc.get("available_balance", 0.0), 4),
        "equity": round(equity, 4),
        "margin_used": round(acc.get("position_margin", 0.0), 4),
        "unrealized_pnl": round(acc.get("unrealized_pnl", 0.0), 4),
        "realized_pnl": round(acc.get("realized_pnl", 0.0), 4),
        "wallet_balance": round(acc.get("wallet_balance", 0.0), 4),
        "total_return_pct": round((equity / initial - 1) * 100, 2) if initial > 0 else 0.0,
        "total_capital": round(equity, 4),
        "pnl_source": PNL_BINANCE,
        "exchange_env": acc.get("env", ""),
        "exchange_ts": acc.get("ts"),
        "exchange_commission": round(acc.get("commission", 0.0), 4),
        "exchange_funding_fee": round(acc.get("funding_fee", 0.0), 4),
        "ledger_realized": round(ledger_realized, 4),
        "ledger_unrealized": round(ledger_unrealized, 4),
        "ledger_equity": None,
        "reconcile_diff": round(ledger_realized + ledger_unrealized - exchange_net, 4),
    })
    return result


def rebase(db: Session, force: bool = False) -> dict:
    """切换环境时把收益基准重置为交易所当前钱包余额，并把缓存指向新环境。"""
    state = all_state(db)
    env = gw.env_of(db)
    if not force and state.get("exchange_baseline_env") == env:
        return {"changed": False, "env": env, "initial_capital": float(state.get("initial_capital") or 0)}
    client = gw.client_for(db)
    acc = gw.fetch_account(client)
    wallet = round(acc["wallet_balance"], 4)
    if wallet <= 0:
        return {"changed": False, "env": env, "reason": "账户余额为 0，未重置基准"}
    set_state(db, "initial_capital", wallet)
    set_state(db, "total_capital", wallet)
    set_state(db, "cash", round(acc["available_balance"], 4))
    set_state(db, "exchange_baseline_env", env)
    set_state(db, "exchange_income_since", now_ms())
    # 换环境后可交易对集合不同，重新探测
    set_state(db, UNSUPPORTED_KEY, [])
    set_state(db, CACHE_KEY, {})
    db.add(JobLog(job="exchange_sync", level="info",
                  message=f"切换至 {env}，收益基准按币安钱包余额重置为 {wallet} USDT"))
    db.commit()
    return {"changed": True, "env": env, "initial_capital": wallet}


def archive_paper_positions(db: Session) -> int:
    """切换 live 时归档遗留的本地模拟持仓：不删数据、不计盈亏、不再占用币池。"""
    rows = db.query(Order).filter(Order.status == "OPEN", Order.source != "live").all()
    for order in rows:
        order.status = "CLOSED"
        order.close_reason = "env_switch_archived"
        order.close_price = order.mark_price or order.entry_price
        order.realized_pnl = 0.0
        order.exit_fee = 0.0
        order.note = (order.note or "") + " | 环境切换归档：本地模拟持仓，不计入币安口径收益"
        from ..models import utcnow

        order.closed_at = utcnow()
    if rows:
        db.add(JobLog(job="exchange_sync", level="info",
                      message=f"环境切换归档 {len(rows)} 条本地模拟持仓（保留记录、不计盈亏）"))
        db.commit()
    return len(rows)


UNSUPPORTED_KEY = "exchange_unsupported_symbols"


def refresh_available(db: Session, client=None) -> float:
    """开仓成功后立即刷新可用余额缓存。

    同一周期可能连开多仓，若一直用周期开始时的 availableBalance，
    会重复计算同一笔可用保证金，最终被交易所 -2019 拒单。
    """
    try:
        client = client or gw.client_for(db)
        acc = gw.fetch_account(client)
    except Exception:  # noqa: BLE001 - 刷新失败不阻断已成交的订单
        return 0.0
    state = all_state(db)
    current = dict(state.get(CACHE_KEY) or {})
    if current:
        current["available_balance"] = round(acc["available_balance"], 4)
        set_state(db, CACHE_KEY, current)
    return round(acc["available_balance"], 4)


def is_unsupported(db: Session, symbol: str) -> bool:
    return symbol in set(all_state(db).get(UNSUPPORTED_KEY) or [])


def mark_unsupported(db: Session, symbol: str) -> None:
    """测试网不存在的交易对（-1121）记入短期跳过表，避免每周期重复无效下单。"""
    items = [item for item in (all_state(db).get(UNSUPPORTED_KEY) or []) if item != symbol]
    items.append(symbol)
    set_state(db, UNSUPPORTED_KEY, items[-300:])


def backfill_fills(db: Session, order: Order, client=None) -> bool:
    """用币安真实成交明细回填逐笔订单。成功返回 True，任何缺口回落本地估算。"""
    note = {}
    try:
        note = json.loads(order.note or "{}")
    except ValueError:
        pass
    ids = {int(note[key]) for key in ("exchange_order_id", "sl_order_id", "tp_order_id", "close_order_id")
           if str(note.get(key) or "").isdigit()}
    if not ids or order.created_at is None:
        return False
    try:
        client = client or gw.client_for(db)
        trades = client.user_trades(order.symbol, start_ms=dt_ms(order.created_at) - 60_000, limit=1000)
    except Exception:  # noqa: BLE001 - 成交明细不可用则保留估算并标注
        return False
    if not isinstance(trades, list):
        return False
    related = [row for row in trades if int(row.get("orderId") or 0) in ids]
    if not related:
        return False
    entry_id = int(note.get("exchange_order_id") or 0)
    entry = [row for row in related if int(row.get("orderId") or 0) == entry_id]
    exits = [row for row in related if int(row.get("orderId") or 0) != entry_id]
    if not entry or not exits:
        return False

    def _sum(rows, field):
        return sum(float(row.get(field) or 0) for row in rows)

    def _commission(rows):
        return sum(float(row.get("commission") or 0) for row in rows
                   if str(row.get("commissionAsset") or "USDT") in ("USDT", "USDC", "BUSD"))

    entry_qty = _sum(entry, "qty")
    exit_qty = _sum(exits, "qty")
    if entry_qty <= 0 or exit_qty <= 0:
        return False
    entry_price = sum(float(r.get("price") or 0) * float(r.get("qty") or 0) for r in entry) / entry_qty
    close_price = sum(float(r.get("price") or 0) * float(r.get("qty") or 0) for r in exits) / exit_qty
    fee_in, fee_out = _commission(entry), _commission(exits)
    realized = _sum(exits, "realizedPnl") - fee_in - fee_out
    order.entry_price = round_price(entry_price)
    order.close_price = round_price(close_price)
    order.mark_price = round_price(close_price)
    order.entry_fee = round(fee_in, 6)
    order.exit_fee = round(fee_out, 6)
    order.realized_pnl = round(realized, 6)
    note["pnl_source"] = PNL_BINANCE
    note["fill_count"] = len(related)
    order.note = json.dumps(note, ensure_ascii=False)
    db.commit()
    return True
