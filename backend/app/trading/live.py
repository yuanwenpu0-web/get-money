"""币安 U 本位合约实盘执行（测试网 / 主网）。与 paper 模块对称：
- 复用 paper.entry_gates 的入场闸门与 paper.close_position 的本地记账
- 下单成功且三个币安订单号都正常同步才写本地 Order；保护单挂失败或订单号未同步
  立即市价回滚，不留脏订单也不留无记录的交易所裸仓
- 开仓主单 orderId 存 Order.exchange_order_id 独立列（可对账），保护单 algoId /
  平仓单号仍存 Order.note 的 JSON 字段
"""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from ..market.futures import (
    BinanceApiError,
    BinanceError,
    BinanceFuturesClient,
    floor_to_step,
    format_amount,
    round_to_tick,
    symbol_filters,
)
from ..models import Credential, JobLog, Order
from ..security import decrypt
from ..state import all_state, set_state
from ..strategy.bindings import encode_order_params, ensure_execution_supported, resolve_order_params
from ..strategy.specs.pool_volatility_research import BLOCKED_CODES
from ..utils import round_price
from .paper import close_position as settle_close
from .paper import entry_gates
from .maker import execute_entry
from .live_rollback import rollback_open
from .portfolio import portfolio_snapshot
from .risk import position_size
from . import exchange_gateway as gw
from . import exchange_truth

CANCEL_IGNORE_CODES = {-2011}
REDUCE_ONLY_REJECT_CODE = -2022
MARGIN_TYPE_UNCHANGED_CODE = -4046
MIN_MARGIN = 10.0
# 以币安为准时，新仓保证金最多动用交易所可用余额的这一比例。余量留给开仓手续费、
# 逐仓初始保证金取整与价格波动：吃满 100% 可用余额必然被币安 -2019 拒单。
EXCHANGE_MARGIN_BUFFER = 0.90
# 币安口径下的最低开仓保证金还不得低于收益基准的 2%：避免满仓后连续开出
# 10 USDT 级碎仓、把可用余额吃光，正式环境下也留出应对波动的余量。
EXCHANGE_MIN_MARGIN_PCT = 0.02


def _exchange_margin_constraint(db: Session, snapshot: dict) -> dict | None:
    """live + 币安口径下返回本笔开仓的可用保证金约束；非币安口径返回 None。

    刻意不修改 position_size 的 equity 入参：那会改变策略的风险预算口径。
    这里只在「执行层」按交易所真实可用余额裁剪仓位，策略信号与参数保持不变。
    """
    if snapshot.get("pnl_source") != exchange_truth.PNL_BINANCE:
        return None
    acc = exchange_truth.cached(db)
    available = max(0.0, float(acc.get("available_balance") or 0.0))
    return {
        "available": available,
        "position_margin": float(acc.get("position_margin") or 0.0),
        "limit": available * EXCHANGE_MARGIN_BUFFER,
    }


def log(db: Session, level: str, message: str) -> None:
    db.add(JobLog(job="live_trading", level=level, message=message[:4000]))
    db.commit()


# 2025-12-09 起币安 U 本位合约条件单必须走 /fapi/v1/algoOrder
CONDITIONAL_TYPES = ("STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP", "TAKE_PROFIT",
                     "TRAILING_STOP_MARKET")


def _action_of(params: dict) -> str:
    if str(params.get("closePosition")) == "true":
        return "protect_sl" if params.get("type") == "STOP_MARKET" else "protect_tp"
    if str(params.get("reduceOnly")) == "true":
        return "rollback" if params.pop("_rollback", False) else "close"
    if params.get("type") == "LIMIT":
        return "open_maker"
    return "open"


class _LoggedClient:
    """透明包装：所有下单/撤单都过 exchange_gateway 留痕，其余方法原样转发。

    maker.execute_entry 与 rollback_open 内部也会调 new_order，用包装对象即可
    覆盖开仓主单、Maker 转市价、保护单、回滚平仓全部写请求，不需改动下游模块。
    """

    def __init__(self, db: Session, client, local_order_id: int | None = None):
        self._db = db
        self._client = client
        self._local = local_order_id

    def __getattr__(self, name):
        return getattr(self._client, name)

    def new_order(self, **params):
        if params.get("type") in CONDITIONAL_TYPES:
            # stopPrice → triggerPrice，端点切到 Algo；返回 algoId 归一成 orderId
            algo_params = {
                "algoType": "CONDITIONAL",
                "symbol": params["symbol"],
                "side": params["side"],
                "type": params["type"],
                "triggerPrice": format_amount(float(params.get("stopPrice") or 0)),
                "closePosition": str(params.get("closePosition") or "false"),
                "newOrderRespType": params.get("newOrderRespType", "RESULT"),
            }
            resp = gw.submit(self._db, self._client, _action_of(params), algo_params,
                             local_order_id=self._local, extra={"endpoint": "algoOrder"},
                             call=self._client.new_algo_order)
            if "orderId" not in resp and "algoId" in resp:
                resp = {**resp, "orderId": resp["algoId"]}
            return resp
        return gw.submit(self._db, self._client, _action_of(params), params,
                         local_order_id=self._local)

    def cancel_order(self, symbol, order_id):
        return gw.cancel(self._db, self._client, "cancel", symbol, order_id,
                         local_order_id=self._local)

    def cancel_algo_order(self, symbol, algo_id):
        return gw.cancel(self._db, self._client, "cancel_protection", symbol, algo_id,
                         local_order_id=self._local,
                         call=lambda: self._client.cancel_algo_order(symbol, int(algo_id)))

    def get_algo_order(self, symbol, algo_id):
        return self._client.get_algo_order(symbol, int(algo_id))


def load_credentials(db: Session) -> tuple[str, str]:
    row = db.query(Credential).order_by(Credential.id.desc()).first()
    if row is None:
        return "", ""
    return decrypt(row.api_key_enc), decrypt(row.api_secret_enc)


def client_for(db: Session) -> BinanceFuturesClient:
    state = all_state(db)
    key, secret = load_credentials(db)
    return BinanceFuturesClient(key, secret, testnet=bool(state.get("testnet", True)),
                                proxy=state.get("proxy_url") or None)


def load_note(order: Order) -> dict:
    try:
        data = json.loads(order.note or "{}")
    except ValueError:
        data = {}
    return data if isinstance(data, dict) else {}


def _close_side(direction: str) -> str:
    return "SELL" if direction == "LONG" else "BUY"


def open_live(db: Session, signal) -> Order | None:
    """实盘开仓：与 paper.open_position 对称，交易所市价成交 + 交易所侧保护单。"""
    if getattr(signal, "strategy", "") in BLOCKED_CODES:
        return None
    state = all_state(db)
    # 主网宇宙里存在、但当前交易环境不支持的币，已在上轮被 -1121 拒绝，直接跳过
    if exchange_truth.is_unsupported(db, signal.symbol):
        return None
    key, secret = load_credentials(db)
    if not key or not secret:
        log(db, "error", f"live 开仓被拒绝: 未配置币安 API 凭证 {getattr(signal, 'symbol', '')}")
        gw.record(db, "open", signal.symbol, status="rejected", message="未配置币安 API 凭证")
        return None
    params = getattr(signal, "params", None) or resolve_order_params(
        db,
        signal.strategy,
        signal.interval,
        getattr(signal, "pool", ""),
        symbol=getattr(signal, "symbol", ""),
    )
    ensure_execution_supported(signal.strategy, params)
    params_snapshot = encode_order_params(signal.strategy, signal.interval, params)
    if not entry_gates(db, signal, params, ignore_cooldown=False):
        return None

    snapshot = portfolio_snapshot(db, update_marks=False)
    cash = snapshot["cash"]
    equity = snapshot["equity"]
    position_pct = float(signal.position_pct or 0)
    if position_pct <= 0 or equity <= 0:
        gw.record(db, "open", signal.symbol, status="blocked",
                  message=f"本地资金闸门：仓位比例={position_pct} 权益={equity}")
        return None
    fee_rate = float(state.get("fee_rate", 0.0004))
    side = 1 if signal.direction == "LONG" else -1
    entry = float(signal.price)
    if (entry - signal.sl) * side <= 0 or (signal.tp - entry) * side <= 0:
        gw.record(db, "open", signal.symbol, status="blocked", message="止损/止盈方向不合法")
        return None
    leverage = min(max(1, int(signal.leverage)), int(params["max_leverage"]))
    margin, qty = position_size(equity, cash, min(position_pct, params["max_position_pct"]), leverage,
                                entry, signal.sl, side, fee_rate, 0.0, params["risk_per_trade_pct"])
    constraint = _exchange_margin_constraint(db, snapshot)
    min_margin = MIN_MARGIN
    if constraint is not None:
        min_margin = max(MIN_MARGIN, float(state.get("initial_capital") or 0) * EXCHANGE_MIN_MARGIN_PCT)
        cap = constraint["limit"]
        if cap < min_margin:
            # 满仓时安静跳过，不再对每个信号发注定被拒的 maker + 市价两次请求
            gw.record(db, "open", signal.symbol, status="blocked",
                      message=(f"币安可用保证金不足：可用 {constraint['available']:.4f} USDT、"
                               f"已占用 {constraint['position_margin']:.4f} USDT、"
                               f"最低开仓保证金 {min_margin:.4f} USDT，跳过本次信号"),
                      payload={"available_balance": constraint["available"],
                               "position_margin": constraint["position_margin"],
                               "min_margin": min_margin, "margin_before_cap": margin})
            return None
        if margin > cap:
            # 保证金与数量必须同步缩放，保持 qty = margin * leverage / entry 的关系
            margin, qty = cap, cap * leverage / entry
    if margin < min_margin or qty <= 0:
        gw.record(db, "open", signal.symbol, status="blocked",
                  message=f"保证金/数量不足：margin={margin:.4f} qty={qty:.6f} min={min_margin:.4f}")
        return None
    try:
        raw_client = BinanceFuturesClient(key, secret, testnet=bool(state.get("testnet", True)),
                                         proxy=state.get("proxy_url") or None)
        client = _LoggedClient(db, raw_client)
        order_type = str(state.get("order_type", "market") or "market")
        return _place_live(db, client, signal, margin, qty, entry, leverage, fee_rate, cash,
                           order_type, params_snapshot)
    except (BinanceError, ValueError, KeyError, TypeError) as exc:
        log(db, "error", f"live 开仓失败 {signal.symbol}: {exc}")
        gw.record(db, "open", signal.symbol, status="failed", message=str(exc))
        if "-1121" in str(exc) or "Invalid symbol" in str(exc):
            exchange_truth.mark_unsupported(db, signal.symbol)
        return None


def _valid_exchange_id(value) -> int | None:
    """币安订单号必须能解析成正整数；缺失 / 0 / 空串 / 非数字一律视为「ID 未同步」。

    只有三个订单号（开仓主单 + 止损/止盈保护单）都有效，本地订单才具备
    对账、撤单、结算的能力，才允许落库。
    """
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _safe_float(value) -> float:
    """脏回报不能把流程抛到 open_live 的兜底 except（那时仓位已开却不回滚）。"""
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _protection_order(client: BinanceFuturesClient, symbol: str, close_side: str, order_type: str, stop_price: float) -> dict:
    return client.new_order(symbol=symbol, side=close_side, type=order_type,
                            stopPrice=format_amount(stop_price), closePosition="true", newOrderRespType="RESULT")


def reanchor_exits(fill_price: float, signal_price: float, sl: float, tp: float) -> tuple[float, float]:
    """把信号价锚定的止损/止盈平移到实际成交价，保持绝对价格距离不变。

    距离不变 ⇒ sl_atr_mult×ATR 与 target_r 在成交后仍成立，且 qty×距离 恰等于
    开仓前按信号价算出的风险预算；留在信号价上会让追价把实际风险放大数倍。
    """
    offset = fill_price - signal_price
    return sl + offset, tp + offset


def _line_logger(db: Session):
    """把 live.log(db, level, message) 适配成 maker / rollback 需要的 (level, message) 回调，
    避免下游模块反向依赖 live 造成循环导入。"""
    return lambda level, message: log(db, level, message)


def _abort_unsynced_open(db: Session, client, signal, close_side: str, qty: float,
                         placed_protections: list[int], reason: str, responses: dict) -> None:
    """币安订单号未正常同步：先回滚交易所仓位，再放弃本地订单记录。

    顺序不能反：下单请求可能已在交易所成交，若只「不记录」会留下系统看不见、
    sync_live 也不会管理的裸仓。回滚失败时 rollback_open 会写 error 日志，
    这里额外写一条 failed 事件并提示人工核对残留条件单（回报缺 ID 时无法撤销）。
    """
    symbol = signal.symbol
    message = (f"币安订单号同步失败，不记录本地订单：{reason}；已触发回滚，"
               f"请人工核对 {symbol} 在交易所是否残留持仓或条件单")
    log(db, "error", f"live 开仓中止 {symbol}: {message}")
    gw.record(db, "open", symbol, status="failed", message=message,
              payload={"reason": reason, "responses": responses})
    rollback_open(client, symbol, close_side, qty, placed_protections, reason, _line_logger(db))


def _place_live(db: Session, client: BinanceFuturesClient, signal, margin, qty, entry, leverage, fee_rate, cash,
                order_type: str = "market", params_snapshot: str | None = None) -> Order | None:
    filters = symbol_filters(client.exchange_info(signal.symbol), signal.symbol)
    lot = filters.get("LOT_SIZE", {})
    price_filter = filters.get("PRICE_FILTER", {})
    step = float(lot.get("stepSize", 0) or 0)
    min_qty = float(lot.get("minQty", 0) or 0)
    tick = float(price_filter.get("tickSize", 0) or 0)
    min_notional = float(filters.get("MIN_NOTIONAL", {}).get("notional", 0) or 0)

    adj_qty = floor_to_step(qty, step)
    if adj_qty <= 0 or adj_qty < min_qty:
        message = f"计算数量 {qty} 低于最小下单量或步长取整后为 0"
        log(db, "error", f"live 开仓失败 {signal.symbol}: {message}")
        gw.record(db, "open", signal.symbol, status="blocked", message=message,
                  payload={"qty": qty, "step": step, "min_qty": min_qty})
        return None
    if min_notional > 0 and adj_qty * entry < min_notional:
        message = f"名义价值 {adj_qty * entry:.4f} 低于最小限制 {min_notional}"
        log(db, "error", f"live 开仓失败 {signal.symbol}: {message}")
        gw.record(db, "open", signal.symbol, status="blocked", message=message)
        return None

    _prepare_account(db, client, signal.symbol, leverage)
    exchange_side = "BUY" if signal.direction == "LONG" else "SELL"
    close_side = _close_side(signal.direction)
    line_log = _line_logger(db)
    order_resp = execute_entry(client, signal.symbol, exchange_side, adj_qty, entry,
                               order_type, tick, step, line_log)
    fill_price = _safe_float(order_resp.get("avgPrice")) or entry
    side = 1 if signal.direction == "LONG" else -1
    sl_raw, tp_raw = reanchor_exits(fill_price, entry, signal.sl, signal.tp)
    sl_trigger, tp_trigger = round_to_tick(sl_raw, tick), round_to_tick(tp_raw, tick)
    # 币安 ID 同步守卫：三张单的订单号必须全部有效，否则这笔订单无法对账/撤单/结算。
    # 此刻交易所可能已真实成交，所以「不记录订单」之前必须先回滚交易所仓位。
    entry_id = _valid_exchange_id(order_resp.get("orderId"))
    sync_error = "" if entry_id is not None else "开仓主单回报缺少有效 orderId"
    placed_protections: list[int] = []
    sl_order: dict = {}
    tp_order: dict = {}
    sl_id = tp_id = None
    if (fill_price - sl_trigger) * side <= 0 or (tp_trigger - fill_price) * side <= 0:
        reason = (f"成交价 {fill_price} 经 tick 取整后止损/止盈方向不合法"
                  f"（sl={sl_trigger} tp={tp_trigger}），已回滚")
        gw.record(db, "open", signal.symbol, status="failed", message=reason)
        rollback_open(client, signal.symbol, close_side, adj_qty, placed_protections, reason, line_log)
        return None
    try:
        sl_order = _protection_order(client, signal.symbol, close_side, "STOP_MARKET", sl_trigger)
        sl_id = _valid_exchange_id(sl_order.get("orderId"))
        if sl_id is None:
            sync_error = sync_error or "止损保护单回报缺少有效 algoId/orderId"
        else:
            placed_protections.append(sl_id)
        tp_order = _protection_order(client, signal.symbol, close_side, "TAKE_PROFIT_MARKET", tp_trigger)
        tp_id = _valid_exchange_id(tp_order.get("orderId"))
        if tp_id is None:
            sync_error = sync_error or "止盈保护单回报缺少有效 algoId/orderId"
        else:
            placed_protections.append(tp_id)
    except BinanceError as exc:
        gw.record(db, "open", signal.symbol, status="failed",
                  message=f"保护单挂单失败，已触发回滚: {exc}")
        rollback_open(client, signal.symbol, close_side, adj_qty, placed_protections, exc, line_log)
        return None

    if sync_error:
        _abort_unsynced_open(db, client, signal, close_side, adj_qty, placed_protections,
                             sync_error, {"entry": order_resp, "sl": sl_order, "tp": tp_order})
        return None

    fee = adj_qty * fill_price * fee_rate
    # live 模式现金/权益以币安账户为准，本地 cash 不再被模拟账本改动。
    # 平仓时 exchange_truth.backfill_fills 会用真实成交价与手续费覆盖 entry_fee。
    # 三个 ID 都通过守卫后才会走到这里：用规范化后的整数同时写独立列与 note，
    # 让 exchange_truth.backfill_fills / live_sync 的既有取数口径保持不变。
    note = json.dumps({"exchange_order_id": entry_id, "sl_order_id": sl_id,
                       "tp_order_id": tp_id, "order_type": order_type,
                       "score": signal.score, "prob_up": signal.prob_up,
                       "pnl_source": exchange_truth.PNL_ESTIMATED,
                       "protection_kind": "algo", "sl_status": "NEW", "tp_status": "NEW"},
                      ensure_ascii=False)
    order = Order(
        symbol=signal.symbol,
        interval=signal.interval,
        side=signal.direction,
        qty=adj_qty,
        entry_price=round_price(fill_price),
        mark_price=round_price(fill_price),
        margin=round(margin, 4),
        leverage=leverage,
        sl=round_price(sl_trigger),
        tp=round_price(tp_trigger),
        status="OPEN",
        source="live",
        exchange_order_id=str(entry_id),
        strategy=signal.strategy,
        pool=getattr(signal, "pool", "") or "",
        binding_id=getattr(signal, "binding_id", None),
        params_snapshot=params_snapshot,
        entry_fee=round(fee, 6),
        note=note,
    )
    db.add(order)
    db.commit()
    db.refresh(order)
    # 同周期连开多仓时刷新可用余额，避免重复用旧余额导致 -2019 保证金不足
    exchange_truth.refresh_available(db, client)
    timestamp = getattr(signal, "ts", None)
    if timestamp is not None:
        set_state(db, f"last_entry:{signal.symbol}:{signal.interval}", timestamp)
    return order


def _prepare_account(db: Session, client: BinanceFuturesClient, symbol: str, leverage: int) -> None:
    try:
        client.set_position_side(False)
    except BinanceApiError as exc:
        log(db, "info", f"单向持仓设置跳过 {symbol}: code={exc.code}")
    try:
        client.set_margin_type(symbol, "ISOLATED")
    except BinanceApiError as exc:
        if exc.code != MARGIN_TYPE_UNCHANGED_CODE:
            log(db, "info", f"保证金模式设置跳过 {symbol}: code={exc.code}")
    client.set_leverage(symbol, leverage)  # 杠杆设置失败则中断开仓，避免风险与预期不符


def close_live(db: Session, order: Order, reason: str = "manual", fallback_price: float | None = None) -> Order:
    """实盘平仓：撤销两个保护单（忽略已触发 -2011），reduceOnly 市价平仓，用成交均价结算本地记账。"""
    if order.status != "OPEN":
        return order
    from .live_sync import _settle_exchange_closed  # 延迟导入避免循环依赖

    client = _LoggedClient(db, client_for(db), local_order_id=order.id)
    note = load_note(order)
    _cancel_protection(db, client, order, note)
    close_side = _close_side(order.side)
    try:
        resp = client.new_order(
            symbol=order.symbol, side=close_side, type="MARKET",
            quantity=format_amount(order.qty), reduceOnly="true", newOrderRespType="RESULT",
        )
    except BinanceApiError as exc:
        if exc.code == REDUCE_ONLY_REJECT_CODE:
            # 交易所仓位已不存在（保护单刚触发），按交易所已平仓结算
            log(db, "info", f"订单 #{order.id} reduceOnly 被拒绝(-2022)，交易所仓位已不存在")
            return _settle_exchange_closed(db, client, order, reason)
        log(db, "error", f"live 平仓下单失败 #{order.id}: {exc}")
        raise
    except BinanceError as exc:
        log(db, "error", f"live 平仓下单失败 #{order.id}: {exc}")
        raise
    price = float(resp.get("avgPrice") or 0) or fallback_price or order.mark_price or order.entry_price
    note["close_order_id"] = resp.get("orderId")
    order.note = json.dumps(note, ensure_ascii=False)
    db.commit()
    settled = settle_close(db, order, price, reason, slippage_rate=0.0, update_cash=False)
    # 逐笔盈亏以币安真实成交价 + 手续费为准；取不到明细时保留估算并显式标注
    if not exchange_truth.backfill_fills(db, settled, client):
        note["pnl_source"] = exchange_truth.PNL_ESTIMATED
        settled.note = json.dumps(note, ensure_ascii=False)
        db.commit()
    return settled


def _cancel_protection(db: Session, client: BinanceFuturesClient, order: Order, note: dict) -> None:
    for field in ("sl_order_id", "tp_order_id"):
        protection_id = note.get(field)
        if not protection_id:
            continue
        try:
            if note.get("protection_kind") == "legacy":
                client.cancel_order(order.symbol, int(protection_id))
            else:
                client.cancel_algo_order(order.symbol, int(protection_id))
        except BinanceApiError as exc:
            if exc.code not in CANCEL_IGNORE_CODES:
                log(db, "error", f"撤销保护单失败 #{order.id} {field}={protection_id}: code={exc.code}")
        except BinanceError as exc:
            log(db, "error", f"撤销保护单失败 #{order.id} {field}={protection_id}: {exc}")


def modify_live(db: Session, order: Order, sl: float | None = None, tp: float | None = None) -> Order:
    """实盘改单：撤旧保护单、挂新单；本地 sl/tp 与 note 同步更新。"""
    if order.status != "OPEN":
        return order
    client = _LoggedClient(db, client_for(db), local_order_id=order.id)
    note = load_note(order)
    close_side = _close_side(order.side)
    tick = _tick_size(client, order.symbol)
    updates: dict = {}
    # 先全部提交到交易所，全部成功后再一次性写本地，避免「交易所已换单、本地还是旧值」
    if sl is not None and sl > 0:
        updates["sl_order_id"] = _replace_protection(
            db, client, order, note, "sl_order_id", "STOP_MARKET", round_to_tick(sl, tick), close_side
        )
    if tp is not None and tp > 0:
        updates["tp_order_id"] = _replace_protection(
            db, client, order, note, "tp_order_id", "TAKE_PROFIT_MARKET", round_to_tick(tp, tick), close_side
        )
    if "sl_order_id" in updates:
        note["sl_order_id"] = updates["sl_order_id"]
        note["sl_status"] = "NEW"
        order.sl = round_price(sl)
    if "tp_order_id" in updates:
        note["tp_order_id"] = updates["tp_order_id"]
        note["tp_status"] = "NEW"
        order.tp = round_price(tp)
    order.note = json.dumps(note, ensure_ascii=False)
    db.commit()
    return order


def _replace_protection(db: Session, client: BinanceFuturesClient, order: Order, note: dict,
                        field: str, order_type: str, stop_price: float, close_side: str) -> int:
    old_id = note.get(field)
    if old_id:
        try:
            if note.get("protection_kind") == "legacy":
                client.cancel_order(order.symbol, int(old_id))
            else:
                client.cancel_algo_order(order.symbol, int(old_id))
        except BinanceApiError as exc:
            if exc.code not in CANCEL_IGNORE_CODES:
                log(db, "error", f"调整订单 #{order.id} 撤销旧保护单失败: code={exc.code}")
    try:
        resp = _protection_order(client, order.symbol, close_side, order_type, stop_price)
    except BinanceError as exc:
        log(db, "error", f"调整订单 #{order.id} 挂新保护单失败 ({order_type}): {exc}")
        raise
    return int(resp["orderId"])


def _tick_size(client: BinanceFuturesClient, symbol: str) -> float:
    filters = symbol_filters(client.exchange_info(symbol), symbol)
    return float(filters.get("PRICE_FILTER", {}).get("tickSize", 0) or 0)
