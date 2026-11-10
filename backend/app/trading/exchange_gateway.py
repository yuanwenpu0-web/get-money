"""交易所网关：环境标识、客户端复用、请求留痕、账户数据读取。

只负责「与币安对话」，不参与策略判断或本地记账。demo(testnet) 与正式主网
共用同一套代码，差异仅是 client 的 host 与凭据，切换环境不需要改逻辑。
"""
from __future__ import annotations

import json

from sqlalchemy.orm import Session

from ..market.futures import BinanceApiError, BinanceError
from ..models import ExchangeEvent
from ..state import all_state
from ..utils import now_ms

DEMO = "demo"
REAL = "real"
# -2011「订单不存在」通常意味着已成交或已撤销，撤单时按可接受处理
CANCEL_IGNORE_CODES = {-2011}


def env_of(db: Session) -> str:
    return DEMO if bool(all_state(db).get("testnet", True)) else REAL


def env_label(db: Session) -> str:
    return "币安合约 Testnet（DEMO）" if env_of(db) == DEMO else "币安合约主网（正式环境）"


def client_for(db: Session):
    """复用 live.client_for：测试里对 live.BinanceFuturesClient 的替换仍然生效。"""
    from .live import client_for as _client_for

    return _client_for(db)


def _f(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def record(db: Session, action: str, symbol: str = "", *, status: str, message: str = "",
           side: str = "", local_order_id: int | None = None, exchange_order_id="",
           exchange_status: str = "", code: int = 0, avg_price: float = 0.0,
           qty: float = 0.0, payload: dict | None = None) -> None:
    """落一条下单/撤单事件。事件写入失败绝不影响交易主流程。"""
    try:
        db.add(ExchangeEvent(
            ts=now_ms(), env=env_of(db), action=action, symbol=symbol or "",
            side=side or "", local_order_id=local_order_id,
            exchange_order_id=str(exchange_order_id or ""),
            status=status, exchange_status=str(exchange_status or ""),
            code=int(code or 0), message=str(message)[:1000],
            avg_price=_f(avg_price), qty=_f(qty),
            payload=json.dumps(payload or {}, ensure_ascii=False, default=str)[:4000],
        ))
        db.commit()
    except Exception:  # noqa: BLE001 - 留痕失败不能回滚业务
        db.rollback()


def submit(db: Session, client, action: str, params: dict, *, side: str = "",
           local_order_id: int | None = None, extra: dict | None = None,
           call=None) -> dict:
    """包装下单请求：成交、拒单、网络异常三种结果全部落事件，异常原样上抛。

    call 可指定实际的交易所方法（如 Algo 条件单用 new_algo_order），
    让主单与条件单共用同一套事件留痕，不需要第二份实现。
    """
    symbol = str(params.get("symbol") or "")
    payload = {**params, **(extra or {})}
    try:
        resp = call(**params) if call is not None else client.new_order(**params)
    except BinanceApiError as exc:
        record(db, action, symbol, status="rejected", code=exc.code, message=str(exc),
               side=side or str(params.get("side") or ""), local_order_id=local_order_id,
               payload=payload)
        raise
    except BinanceError as exc:
        record(db, action, symbol, status="failed", message=str(exc),
               side=side or str(params.get("side") or ""), local_order_id=local_order_id,
               payload=payload)
        raise
    if not isinstance(resp, dict):
        resp = {}
    status = str(resp.get("status") or "")
    record(db, action, symbol,
           status="filled" if status == "FILLED" else "accepted",
           exchange_order_id=resp.get("orderId", ""), exchange_status=status,
           avg_price=resp.get("avgPrice", 0),
           qty=resp.get("executedQty") or resp.get("origQty") or 0,
           side=side or str(params.get("side") or ""), local_order_id=local_order_id,
           payload=payload)
    return resp


def cancel(db: Session, client, action: str, symbol: str, order_id, *,
           local_order_id: int | None = None, call=None) -> dict:
    """包装撤单；call 可指定 Algo 条件单撤单方法。"""
    try:
        resp = call() if call is not None else client.cancel_order(symbol, int(order_id))
    except BinanceApiError as exc:
        record(db, action, symbol, status="rejected", code=exc.code, message=str(exc),
               exchange_order_id=order_id, local_order_id=local_order_id)
        raise
    except BinanceError as exc:
        record(db, action, symbol, status="failed", message=str(exc),
               exchange_order_id=order_id, local_order_id=local_order_id)
        raise
    record(db, action, symbol, status="canceled", exchange_order_id=order_id,
           local_order_id=local_order_id)
    return resp if isinstance(resp, dict) else {}


def fetch_account(client) -> dict:
    """/fapi/v2/account 是全 USDT 折算口径；测试网若不支持则回落 /fapi/v2/balance。"""
    try:
        acc = client.account()
        if not isinstance(acc, dict):
            raise ValueError("account 返回格式异常")
        usdt = next((a for a in acc.get("assets", []) if a.get("asset") == "USDT"), {})
        return {
            "wallet_balance": _f(acc.get("totalWalletBalance") or usdt.get("walletBalance")),
            "available_balance": _f(acc.get("availableBalance") or usdt.get("availableBalance")),
            "margin_balance": _f(acc.get("totalMarginBalance") or usdt.get("marginBalance")),
            "unrealized_pnl": _f(acc.get("totalUnrealizedProfit") or usdt.get("unrealizedProfit")),
            "position_margin": _f(acc.get("totalPositionInitialMargin") or usdt.get("positionInitialMargin")),
            "source": "account",
        }
    except Exception:  # noqa: BLE001 - 测试网 account 不可用是已知降级路径
        rows = client.balance()
        usdt = next((a for a in rows if a.get("asset") == "USDT"), {}) if isinstance(rows, list) else {}
        wallet = _f(usdt.get("balance") or usdt.get("walletBalance"))
        available = _f(usdt.get("availableBalance"))
        return {
            "wallet_balance": wallet,
            "available_balance": available,
            "margin_balance": wallet,
            "unrealized_pnl": 0.0,
            "position_margin": max(0.0, wallet - available),
            "source": "balance",
        }


def fetch_income(client, since_ms: int = 0) -> dict:
    """账户收益流水明细（手续费/资金费/已实现）。最多翻 10 页，失败返回 available=False。"""
    out = {"realized_pnl": 0.0, "commission": 0.0, "funding_fee": 0.0,
           "rows": 0, "available": True}
    cursor = int(since_ms or 0)
    try:
        for _ in range(10):
            rows = client.income(start_ms=cursor, limit=1000)
            if not isinstance(rows, list) or not rows:
                break
            for row in rows:
                kind = str(row.get("incomeType") or "")
                value = _f(row.get("income"))
                if kind == "REALIZED_PNL":
                    out["realized_pnl"] += value
                elif kind == "COMMISSION":
                    out["commission"] += value
                elif kind == "FUNDING_FEE":
                    out["funding_fee"] += value
            out["rows"] += len(rows)
            last_ts = max(int(row.get("time") or 0) for row in rows)
            if len(rows) < 1000 or last_ts <= cursor:
                break
            cursor = last_ts  # startTime inclusive，下一页从最后一条时间戳起
        return out
    except Exception:  # noqa: BLE001 - 明细不可用不影响主口径
        return {**out, "available": False}
