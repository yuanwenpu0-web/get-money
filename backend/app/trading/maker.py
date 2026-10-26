"""实盘开仓主单执行：market 与 limit_maker 两种模式。

limit_maker 以标记价挂 postOnly（type=LIMIT, timeInForce=GTX）限价单赚取 maker
费率（0.04% → 0.02%），轮询订单状态（每次间隔 MAKER_POLL_INTERVAL_SECONDS，最多
MAKER_POLL_MAX_TIMES 次，约 10s）：
- 成交 → 返回成交回报，由调用方继续挂保护单
- 超时未成交/部分成交 → 撤单后仅对剩余数量转市价单，保证成交确定性且不重复建仓
- 限价单被交易所明确拒绝（如会立即成交）→ 直接转市价单
- 撤单/轮询网络异常 → 向上抛，由上层中止开仓，避免撤单状态不明时重复下单
"""

from __future__ import annotations

import time

from ..market.futures import (
    BinanceApiError,
    BinanceError,
    floor_to_step,
    format_amount,
    round_to_tick,
)

MAKER_POLL_INTERVAL_SECONDS = 1.0
MAKER_POLL_MAX_TIMES = 10
CANCEL_IGNORE_CODES = {-2011}
TERMINAL_STATUSES = ("CANCELED", "EXPIRED", "REJECTED", "PARTIALLY_CANCELED")


def execute_entry(client, symbol: str, side: str, qty: float, fallback_price: float,
                  order_type: str, tick: float, step: float, log) -> dict:
    """按 order_type 下开仓主单，返回带 avgPrice / orderId 的成交回报（market 模式行为不变）。"""
    if order_type != "limit_maker":
        return client.new_order(
            symbol=symbol, side=side, type="MARKET",
            quantity=format_amount(qty), newOrderRespType="RESULT",
        )
    return _maker_entry(client, symbol, side, qty, fallback_price, tick, step, log)


def _maker_entry(client, symbol: str, side: str, qty: float, fallback_price: float,
                 tick: float, step: float, log) -> dict:
    mark = _mark_price(client, symbol)
    price = round_to_tick(mark if mark > 0 else fallback_price, tick)
    try:
        resp = client.new_order(
            symbol=symbol, side=side, type="LIMIT", timeInForce="GTX",
            quantity=format_amount(qty), price=format_amount(price),
            newOrderRespType="RESULT",
        )
    except BinanceApiError as exc:
        # 交易所明确拒单（如 postOnly 会立即成交）：订单未成立，直接转市价保证成交
        log("info", f"limit_maker 限价单被拒 {symbol}: {exc}，转市价单保证成交")
        return _market_entry(client, symbol, side, qty)
    order_id = int(resp.get("orderId") or 0)
    try:
        last = _poll_until_settled(client, symbol, order_id)
    except BinanceError:
        _cancel_quietly(client, symbol, order_id, log)
        raise
    if last is not None and str(last.get("status") or "") == "FILLED":
        return last

    # 未成交（或已终态）：撤单，按交易所回报的最终成交量只补足剩余数量
    cancel_resp = _cancel_quietly(client, symbol, order_id, log)
    filled_qty = float(cancel_resp.get("executedQty") or 0)
    filled_avg = float(cancel_resp.get("avgPrice") or 0)
    remaining = floor_to_step(qty - filled_qty, step)
    if remaining <= 0:
        return cancel_resp or (last or resp)
    market_resp = _market_entry(client, symbol, side, remaining)
    market_avg = float(market_resp.get("avgPrice") or 0) or price
    if filled_qty > 0 and filled_avg > 0:
        # 部分成交 + 市价补足：成交量加权平均价入账，保护单按整笔仓位挂
        blended = (filled_qty * filled_avg + remaining * market_avg) / qty
        log("info", f"limit_maker 部分成交后转市价补足 {symbol}: "
                    f"maker {filled_qty} @ {filled_avg} + taker {remaining} @ {market_avg}")
        return {"orderId": market_resp.get("orderId"), "avgPrice": blended, "status": "FILLED"}
    log("info", f"limit_maker 超时未成交 {symbol}，撤单后转市价单成交")
    return market_resp


def _mark_price(client, symbol: str) -> float:
    """交易所标记价；拉取失败返回 0，由调用方回退到信号价。"""
    try:
        return float(client.premium_index(symbol).get("markPrice") or 0)
    except (BinanceError, AttributeError, TypeError, ValueError):
        return 0.0


def _poll_until_settled(client, symbol: str, order_id: int) -> dict:
    last = client.get_order(symbol, order_id)
    if str(last.get("status") or "") == "FILLED":
        return last
    for _ in range(MAKER_POLL_MAX_TIMES):
        time.sleep(MAKER_POLL_INTERVAL_SECONDS)
        last = client.get_order(symbol, order_id)
        status = str(last.get("status") or "")
        if status == "FILLED" or status in TERMINAL_STATUSES:
            return last
    return last


def _cancel_quietly(client, symbol: str, order_id: int, log) -> dict:
    """撤单并取回最终状态。-2011（订单不存在）可能是已成交或已撤销：
    必须再查一次订单、按最终成交量决定是否转市价，避免把已成交量当未成交而重复建仓；
    仍查不到状态则上抛，由上层中止开仓（宁可不开，不可加倍）。"""
    try:
        resp = client.cancel_order(symbol, order_id)
    except BinanceApiError as exc:
        if exc.code not in CANCEL_IGNORE_CODES:
            log("error", f"limit_maker 撤单失败 {symbol} orderId={order_id}: code={exc.code}")
            raise
        try:
            final = client.get_order(symbol, order_id)
        except BinanceError as fetch_exc:
            log("error", f"limit_maker 撤单返回 -2011 且最终状态查询失败 {symbol} "
                         f"orderId={order_id}: {fetch_exc}，中止开仓防止重复建仓")
            raise
        return final if isinstance(final, dict) else {}
    return resp if isinstance(resp, dict) else {}


def _market_entry(client, symbol: str, side: str, qty: float) -> dict:
    return client.new_order(
        symbol=symbol, side=side, type="MARKET",
        quantity=format_amount(qty), newOrderRespType="RESULT",
    )
