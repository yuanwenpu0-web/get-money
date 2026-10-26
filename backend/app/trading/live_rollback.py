"""实盘开仓回滚：保护单挂失败时先市价平仓保资金安全，再撤销已挂保护单防残留脏单。

从 live.py 拆出（单文件职责与行数约束）；log 由调用方注入，避免与 live 循环依赖。
撤单失败不阻塞平仓，全部留痕 JobLog。
"""

from __future__ import annotations

from ..market.futures import BinanceApiError, BinanceError, format_amount

CANCEL_IGNORE_CODES = {-2011}
REDUCE_ONLY_REJECT_CODE = -2022


def rollback_open(client, symbol: str, close_side: str, qty: float,
                  placed_protections: list[int], exc: BinanceError, log) -> None:
    """保护单挂失败回滚：先市价平仓保资金安全，再撤销已挂出的保护单防止残留脏单。"""
    log("error", f"保护单挂单失败 {symbol}: {exc}，开始回滚刚开的仓")
    try:
        client.new_order(symbol=symbol, side=close_side, type="MARKET",
                         quantity=format_amount(qty), reduceOnly="true")
        log("info", f"保护单失败回滚完成 {symbol}: 已市价平仓，不保留本地订单")
    except BinanceApiError as close_exc:
        level = "info" if close_exc.code == REDUCE_ONLY_REJECT_CODE else "error"
        log(level, f"回滚平仓 {symbol}: {close_exc}（-2022 表示仓位已被交易所保护单平掉），请核对交易所仓位")
    except BinanceError as close_exc:
        log("error", f"回滚平仓失败 {symbol}: {close_exc}，请立即人工检查交易所仓位")
    for order_id in placed_protections:
        try:
            client.cancel_order(symbol, order_id)
        except BinanceApiError as cancel_exc:
            if cancel_exc.code not in CANCEL_IGNORE_CODES:
                log("error", f"回滚撤销保护单失败 {symbol} orderId={order_id}: code={cancel_exc.code}")
        except BinanceError as cancel_exc:
            log("error", f"回滚撤销保护单失败 {symbol} orderId={order_id}: {cancel_exc}")
    if placed_protections:
        log("info", f"回滚清理完成 {symbol}: 已撤销 {len(placed_protections)} 个已挂保护单")
