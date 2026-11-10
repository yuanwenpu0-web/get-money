from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from ..db import get_db
from ..market.data_service import MarketDataService
from ..models import EquitySnapshot, ExchangeEvent, JobLog, Order, Signal, StrategyConfig
from ..pagination import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    PageQuery,
    PageSizeQuery,
    cached_count,
    offset_of,
    page_metadata,
)
from ..schemas import BatchClosePayload, OrderPatch
from ..serializers import order_dict, signal_dict
from ..state import all_state
from ..trading import exchange_gateway as gw
from ..trading import exchange_truth
from ..trading.dispatch import modify_order
from ..trading.portfolio import portfolio_snapshot
from ..trading.scheduler import scheduler
from ..trading.service import TradingService
from ..utils import now_ms

router = APIRouter(prefix="/trading", tags=["trading"])


def _filtered_orders(db: Session, status: str | None, symbol: str | None, strategy: str | None):
    """订单列表与「按筛选全选」共用的同一套搜索条件，避免两处口径漂移。"""
    query = db.query(Order)
    if status:
        query = query.filter(Order.status == status.upper())
    if symbol:
        query = query.filter(Order.symbol == symbol.upper())
    selected_strategy = strategy.strip().lower() if strategy else ""
    if selected_strategy:
        query = query.filter(func.lower(Order.strategy) == selected_strategy)
    return query


def _open_order_query(db: Session, status: str | None, symbol: str | None, strategy: str | None):
    """当前搜索条件下所有可平仓（OPEN）订单，跨分页；无结果时返回 None。

    status 筛到「已平仓」时没有可平仓对象；筛「全部」时按其他条件取全部持仓。
    """
    if status and status.upper() != "OPEN":
        return None
    return _filtered_orders(db, None, symbol, strategy).filter(Order.status == "OPEN")


@router.get("/exchange")
def exchange_state(db: Session = Depends(get_db)):
    """交易环境、币安账户快照、本地账本对账差异、近期下单成败统计。"""
    state = all_state(db)
    snapshot = portfolio_snapshot(db, update_marks=False)
    recent = db.query(ExchangeEvent).order_by(ExchangeEvent.id.desc()).limit(100).all()
    # blocked 是本地资金闸门主动跳过，从未提交到交易所，不能算作「下单失败」。
    # 成功率只看真正发出去的请求；跳过次数单独返回，便于区分满仓与拒单。
    submitted = [row for row in recent if row.status != "blocked"]
    blocked_rows = [row for row in recent if row.status == "blocked"]
    ok = sum(1 for row in submitted if row.status in ("filled", "accepted", "canceled"))
    failures = [row for row in submitted if row.status in ("rejected", "failed")]
    return {
        "env": gw.env_of(db),
        "env_label": gw.env_label(db),
        "execution_mode": state.get("execution_mode"),
        "truth_enabled": bool(state.get("exchange_truth_enabled", True)),
        "pnl_source": snapshot.get("pnl_source", "ledger"),
        "account": exchange_truth.cached(db, state),
        "equity": snapshot["equity"],
        "reconcile_diff": snapshot.get("reconcile_diff", 0.0),
        "ledger_equity": snapshot.get("ledger_equity") or 0.0,
        "ledger_realized": snapshot.get("ledger_realized", 0.0),
        "ledger_unrealized": snapshot.get("ledger_unrealized", 0.0),
        "market_source": "futures-mainnet",
        "recent_submits": len(submitted),
        "recent_success_rate": round(ok / len(submitted) * 100, 1) if submitted else None,
        "recent_blocked": len(blocked_rows),
        "recent_blocked_reason": blocked_rows[0].message if blocked_rows else "",
        "recent_failures": [
            {"id": row.id, "ts": row.ts, "symbol": row.symbol, "action": row.action,
             "status": row.status, "code": row.code, "message": row.message}
            for row in failures[:10]
        ],
    }


@router.get("/exchange-events")
def exchange_events(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    status: str | None = None,
    action: str | None = None,
    db: Session = Depends(get_db),
):
    """下单/保护单/平仓/撤单的逐笔交易所回执，用于核对「下单是否成功」。"""
    query = db.query(ExchangeEvent)
    if status:
        query = query.filter(ExchangeEvent.status == status)
    if action:
        query = query.filter(ExchangeEvent.action == action)
    rows = query.order_by(ExchangeEvent.id.desc()).limit(limit).all()
    return {
        "events": [
            {
                "id": row.id, "ts": row.ts, "env": row.env, "action": row.action,
                "symbol": row.symbol, "side": row.side, "local_order_id": row.local_order_id,
                "exchange_order_id": row.exchange_order_id, "status": row.status,
                "exchange_status": row.exchange_status, "code": row.code,
                "message": row.message, "avg_price": row.avg_price, "qty": row.qty,
                "created_at": row.created_at.isoformat() if row.created_at else "",
            }
            for row in rows
        ],
        "total": query.count(),
    }


@router.get("/dashboard")
def dashboard(
    db: Session = Depends(get_db),
    include_market: bool = True,
    positions_page: PageQuery = 1,
    orders_page: PageQuery = 1,
    signals_page: PageQuery = 1,
    logs_page: PageQuery = 1,
):
    """账户快照 + 四张列表卡片的首/指定页。

    四个列表各自分页（每页固定不超过 15 条），一次请求就能取到任意一页，前端翻页
    只需换个 page 参数，不必再为每个卡片单开一个接口。
    """
    snapshot = portfolio_snapshot(db, update_marks=True)
    state = all_state(db)
    size = DEFAULT_PAGE_SIZE

    open_query = db.query(Order).filter(Order.status == "OPEN")
    positions_total = open_query.count()
    open_orders = (
        open_query.order_by(Order.created_at.desc(), Order.id.desc())
        .offset(offset_of(positions_page, size)).limit(size).all()
    )

    orders_total = db.query(func.count(Order.id)).scalar() or 0
    recent_orders = (
        db.query(Order).order_by(Order.created_at.desc(), Order.id.desc())
        .offset(offset_of(orders_page, size)).limit(size).all()
    )

    # signals 已到千万级，count(*) 走 PK 索引也要 200ms，而这张卡片只用于「回看最近信号」，
    # 总数按 TTL 缓存，翻页本身仍然是索引扫描 + LIMIT。
    signals_total = cached_count(
        db.get_bind(),
        "trading:signals_total",
        lambda: int(db.query(func.count(Signal.id)).scalar() or 0),
        ttl=60.0,
    )
    recent_signals = (
        db.query(Signal).order_by(Signal.id.desc())
        .offset(offset_of(signals_page, size)).limit(size).all()
    )

    logs_total = db.query(func.count(JobLog.id)).scalar() or 0
    latest_logs = (
        db.query(JobLog).order_by(JobLog.id.desc())
        .offset(offset_of(logs_page, size)).limit(size).all()
    )

    stats = MarketDataService(db).stats() if include_market else []
    return {
        "portfolio": snapshot,
        "positions": [order_dict(row) for row in open_orders],
        "positions_pagination": page_metadata(positions_total, positions_page, size),
        "recent_orders": [order_dict(row) for row in recent_orders],
        "orders_pagination": page_metadata(orders_total, orders_page, size),
        "recent_signals": [signal_dict(row) for row in recent_signals],
        "signals_pagination": page_metadata(signals_total, signals_page, size),
        "logs": [
            {
                "id": row.id,
                "job": row.job,
                "level": row.level,
                "message": row.message,
                "created_at": row.created_at.isoformat() if row.created_at else "",
            }
            for row in latest_logs
        ],
        "logs_pagination": page_metadata(logs_total, logs_page, size),
        "scheduler": scheduler.status(),
        "market": stats,
        "settings": {
            "symbols": state.get("symbols"),
            "interval": state.get("interval"),
            "schedule_enabled": state.get("schedule_enabled"),
            "schedule_seconds": state.get("schedule_seconds"),
            "execution_mode": state.get("execution_mode"),
            "testnet": state.get("testnet"),
            "exchange_env": gw.env_of(db),
            "exchange_env_label": gw.env_label(db),
        },
    }


@router.get("/orders")
def list_orders(
    status: str | None = None,
    symbol: str | None = None,
    limit: Annotated[int | None, Query(ge=1, le=MAX_PAGE_SIZE)] = None,
    db: Session = Depends(get_db),
    page: PageQuery = 1,
    page_size: Annotated[int | None, Query(ge=1, le=MAX_PAGE_SIZE)] = None,
    strategy: str | None = None,
):
    # Explicit page_size wins; legacy callers may still request their old limit.
    page_size = page_size if page_size is not None else (limit if limit is not None else DEFAULT_PAGE_SIZE)
    query = _filtered_orders(db, status, symbol, strategy)
    mark = func.coalesce(func.nullif(Order.mark_price, 0), Order.entry_price)
    pnl = (mark - Order.entry_price) * Order.qty * case((Order.side == "LONG", 1), else_=-1)
    total, open_count, closed_count, realized, unrealized = query.with_entities(
        func.count(Order.id),
        func.coalesce(func.sum(case((Order.status == "OPEN", 1), else_=0)), 0),
        func.coalesce(func.sum(case((Order.status == "CLOSED", 1), else_=0)), 0),
        func.coalesce(func.sum(case((Order.status == "CLOSED", Order.realized_pnl), else_=0.0)), 0.0),
        func.coalesce(func.sum(case((Order.status == "OPEN", pnl), else_=0.0)), 0.0),
    ).one()
    rows = (query.order_by(Order.created_at.desc(), Order.id.desc())
            .offset(offset_of(page, page_size)).limit(page_size).all())
    # open_total：全市场持仓数（不受筛选影响，兼容旧前端）。
    # selectable_open：当前搜索条件下可勾选平掉的持仓数，与 /orders/open-ids 同口径。
    open_total = db.query(func.count(Order.id)).filter(Order.status == "OPEN").scalar() or 0
    selectable_query = _open_order_query(db, status, symbol, strategy)
    selectable_open = selectable_query.count() if selectable_query is not None else 0
    return {
        "orders": [order_dict(row) for row in rows],
        **page_metadata(total, page, page_size),
        "summary": {"open": open_count, "closed": closed_count,
                    "realized_pnl": round(realized, 4), "unrealized_pnl": round(unrealized, 4),
                    "open_total": int(open_total), "selectable_open": selectable_open},
    }


@router.get("/orders/markers")
def list_order_markers(
    symbol: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
    db: Session = Depends(get_db),
):
    """K线图上叠加成交标记用的极简订单列表。

    只回传画图必需的 6 个字段（原来前端为了画几个三角要拉 100 条完整订单，其中包含
    参数快照文本）。这里是「图上的点」，不是「列表」，因此按时间取最近 N 条而不是分页。
    """
    query = db.query(
        Order.id, Order.symbol, Order.side, Order.status,
        Order.created_at, Order.closed_at, Order.entry_price, Order.close_price,
    )
    if symbol:
        query = query.filter(Order.symbol == symbol.upper())
    rows = query.order_by(Order.created_at.desc(), Order.id.desc()).limit(limit).all()
    return {
        "markers": [
            {
                "id": row.id,
                "symbol": row.symbol,
                "side": row.side,
                "status": row.status,
                "created_at": row.created_at.isoformat() if row.created_at else "",
                "closed_at": row.closed_at.isoformat() if row.closed_at else "",
                "entry_price": row.entry_price,
                "close_price": row.close_price,
            }
            for row in rows
        ]
    }


@router.get("/orders/strategies")
def list_order_strategies(db: Session = Depends(get_db)):
    """订单里实际出现过的策略，包括已下线但仍有历史订单的策略。"""
    rows = (
        db.query(Order.strategy, StrategyConfig.name, func.count(Order.id))
        .outerjoin(StrategyConfig, StrategyConfig.code == Order.strategy)
        .filter(Order.strategy.isnot(None), Order.strategy != "")
        .group_by(Order.strategy, StrategyConfig.name)
        .order_by(Order.strategy.asc())
        .all()
    )
    return {
        "strategies": [
            {"code": code, "name": name or code, "count": int(count)}
            for code, name, count in rows
        ]
    }


@router.get("/orders/open-ids")
def list_open_order_ids(
    status: str | None = None,
    symbol: str | None = None,
    strategy: str | None = None,
    db: Session = Depends(get_db),
):
    """按列表的同一套搜索条件，展开所有可平仓订单 ID（跨分页），供「按筛选全选」使用。"""
    query = _open_order_query(db, status, symbol, strategy)
    if query is None:
        return {"order_ids": [], "count": 0}
    ids = [int(row[0]) for row in query.with_entities(Order.id).order_by(Order.id.asc()).all()]
    return {"order_ids": ids, "count": len(ids)}


def _close_order_batch(db: Session, order_ids: list[int], log_label: str) -> dict:
    """按 ID 逐笔市价平仓；只处理 OPEN，单笔异常不中断整批。

    先把 ID 与状态读成普通字典再下单：单笔异常会 rollback，ORM 对象会失效。
    """
    unique_ids = list(dict.fromkeys(int(order_id) for order_id in order_ids))
    portfolio_snapshot(db, update_marks=True)
    snapshots = {}
    if unique_ids:
        snapshots = {
            int(row[0]): (str(row[1]), str(row[2]))
            for row in db.query(Order.id, Order.symbol, Order.status).filter(Order.id.in_(unique_ids)).all()
        }
    service = TradingService(db)
    closed: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    for order_id in unique_ids:
        snapshot = snapshots.get(order_id)
        if snapshot is None:
            skipped.append({"id": order_id, "reason": "订单不存在"})
            continue
        symbol, status = snapshot
        if status != "OPEN":
            skipped.append({"id": order_id, "symbol": symbol, "reason": f"状态为 {status}"})
            continue
        try:
            order = service.manual_close(order_id)
        except Exception as exc:  # noqa: BLE001 - 单笔异常（含实盘下单失败）不应中断整批
            db.rollback()
            failed.append({"id": order_id, "symbol": symbol, "error": f"{exc.__class__.__name__}: {exc}"})
            continue
        if order is None:
            skipped.append({"id": order_id, "symbol": symbol, "reason": "订单不存在或已平仓"})
            continue
        closed.append(order_dict(order))

    realized = round(sum(float(item.get("realized_pnl") or 0) for item in closed), 4)
    db.add(
        JobLog(
            job="order",
            level="warning" if failed else "info",
            message=(f"{log_label}：成功 {len(closed)} 笔，失败 {len(failed)} 笔，"
                     f"跳过 {len(skipped)} 笔，已实现 {realized:.2f}")[:4000],
        )
    )
    db.commit()
    return {
        "closed": closed,
        "failed": failed,
        "skipped": skipped,
        "closed_count": len(closed),
        "failed_count": len(failed),
        "skipped_count": len(skipped),
        "requested_count": len(unique_ids),
        "realized_pnl": realized,
    }


@router.post("/orders/batch-close")
def batch_close_orders(payload: BatchClosePayload, db: Session = Depends(get_db)):
    """批量平掉用户勾选的持仓（按最新标记价市价成交）。

    只认前端提交的 ID 列表：确认框里显示的就是实际会平掉的对象，服务端不会
    重新按筛选条件扩大范围；期间已被止损/止盈平掉的订单会跳过并如实回报。
    """
    return _close_order_batch(db, payload.order_ids, "批量平仓")


@router.post("/orders/close-all")
def close_all_orders(db: Session = Depends(get_db)):
    """兼容旧调用：一次平掉全部持仓。前端已改为勾选 + /orders/batch-close。"""
    ids = [
        int(row[0])
        for row in db.query(Order.id).filter(Order.status == "OPEN").order_by(Order.id.asc()).all()
    ]
    result = _close_order_batch(db, ids, "一键平仓")
    return {key: result[key] for key in ("closed", "failed", "closed_count", "failed_count", "realized_pnl")}


@router.get("/orders/{order_id}")
def get_order(order_id: int, db: Session = Depends(get_db)):
    row = db.get(Order, order_id)
    if row is None:
        raise HTTPException(404, "订单不存在")
    return order_dict(row, include_snapshot=True)


@router.post("/orders/{order_id}/close")
def close_order(order_id: int, db: Session = Depends(get_db)):
    order = TradingService(db).manual_close(order_id)
    if order is None:
        raise HTTPException(400, "订单不存在或已平仓")
    return order_dict(order)


@router.patch("/orders/{order_id}")
def patch_order(order_id: int, payload: OrderPatch, db: Session = Depends(get_db)):
    order = db.get(Order, order_id)
    if order is None:
        raise HTTPException(404, "订单不存在")
    if order.status != "OPEN":
        raise HTTPException(400, "仅未平仓订单可调整")
    mark = order.mark_price or order.entry_price
    if payload.sl is not None:
        if order.side == "LONG" and payload.sl >= mark:
            raise HTTPException(400, "多单止损价必须低于当前价")
        if order.side == "SHORT" and payload.sl <= mark:
            raise HTTPException(400, "空单止损价必须高于当前价")
    if payload.tp is not None:
        if order.side == "LONG" and payload.tp <= mark:
            raise HTTPException(400, "多单止盈价必须高于当前价")
        if order.side == "SHORT" and payload.tp >= mark:
            raise HTTPException(400, "空单止盈价必须低于当前价")
    modify_order(db, order, sl=payload.sl, tp=payload.tp)
    db.commit()
    db.refresh(order)
    db.add(JobLog(job="order", level="info", message=f"订单 #{order.id} 调整 SL={order.sl} TP={order.tp}"))
    db.commit()
    return order_dict(order)


@router.post("/orders/{order_id}/reverse")
def reverse_order(order_id: int, db: Session = Depends(get_db)):
    try:
        order = TradingService(db).reverse_order(order_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    if order is None:
        raise HTTPException(400, "订单不存在或已平仓")
    return order_dict(order)


@router.get("/equity")
def equity_history(limit: int = Query(1000, ge=10, le=5000), db: Session = Depends(get_db)):
    rows = db.query(EquitySnapshot).order_by(EquitySnapshot.id.desc()).limit(limit).all()
    rows = list(reversed(rows))
    points = [
        {
            "time": row.ts,
            "cash": row.cash,
            "equity": row.equity,
            "unrealized_pnl": row.unrealized_pnl,
            "open_positions": row.open_positions,
        }
        for row in rows
    ]
    return {"points": points, "count": len(points), "now": now_ms()}


@router.get("/logs")
def logs(
    limit: Annotated[int | None, Query(ge=1, le=MAX_PAGE_SIZE)] = None,
    db: Session = Depends(get_db),
    page: PageQuery = 1,
    page_size: Annotated[int | None, Query(ge=1, le=MAX_PAGE_SIZE)] = None,
):
    page_size = page_size if page_size is not None else (limit if limit is not None else DEFAULT_PAGE_SIZE)
    query = db.query(JobLog)
    total = query.count()
    rows = (query.order_by(JobLog.created_at.desc(), JobLog.id.desc())
            .offset(offset_of(page, page_size)).limit(page_size).all())
    return {
        **page_metadata(total, page, page_size),
        "logs": [
            {
                "id": row.id,
                "job": row.job,
                "level": row.level,
                "message": row.message,
                "created_at": row.created_at.isoformat() if row.created_at else "",
            }
            for row in rows
        ]
    }


@router.post("/run-once")
def run_cycle(db: Session = Depends(get_db)):
    return TradingService(db).run_cycle()
