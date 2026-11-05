from __future__ import annotations

import hashlib
import json
import threading
import time
import traceback
from types import SimpleNamespace

from sqlalchemy.orm import Session

from ..config import MARKET_SYNC_WORKERS
from ..db import SessionLocal
from ..market.data_service import MarketDataService
from ..models import JobLog, Order
from ..state import all_state, get_state, set_state
from ..strategy.specs.pool_volatility_research import BLOCKED_CODES as RESEARCH_CODES
from ..strategy.bindings import (
    active_bindings, decode_order_params, encode_order_params, expand, resolve_order_params,
)
from ..strategy.engine import StrategyEngine
from ..strategy.registry import get_spec
from ..utils import interval_ms, iso_utc, now_ms
from .order_manager import manage_order
from .dispatch import close_position, open_position
from .live_sync import sync_live
from .portfolio import open_orders, portfolio_snapshot, snapshot_equity

SYNC_WORKERS = MARKET_SYNC_WORKERS


def _research_pending_key(symbol: str, strategy: str, pool: str) -> str:
    key = f"research_pending:{symbol}:{strategy}:{pool}"
    if len(key) <= 64:
        return key
    return "research_pending:" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:47]


def _sync_recent_many(symbols: list[str], interval: str, workers: int = SYNC_WORKERS) -> list[str]:
    """共享连接池并发拉行情，单一会话在调用线程顺序写入。"""
    db = SessionLocal()
    try:
        result = MarketDataService(db).sync_recent_many(symbols, interval, workers=workers)
        return result["errors"]
    finally:
        db.close()


class TradingService:
    _lock = threading.Lock()

    def __init__(self, db: Session):
        self.db = db

    def _log(self, level: str, message: str) -> None:
        self.db.add(JobLog(job="trading_cycle", level=level, message=message[:4000]))
        self.db.commit()

    def run_cycle(self) -> dict:
        if not self._lock.acquire(blocking=False):
            return {"ok": False, "message": "已有交易周期在运行"}
        try:
            started = time.monotonic()
            state = all_state(self.db)
            interval = state.get("interval", "1m")
            # symbol_scope=custom 是全局限制（同步与执行范围），币种池在其内部再筛选
            allowed = (
                list(state.get("symbols") or [])
                if state.get("symbol_scope", "all") == "custom"
                else None
            )
            bindings, binding_errors = active_bindings(
                self.db, state=state, allowed_symbols=allowed
            )
            pairs = expand(bindings)
            symbols = sorted({symbol for symbol, _ in pairs})
            engine = StrategyEngine(self.db, cycle_snapshot=True)

            # 同步窗口按「该币种需要的最大周期」取，保证绑定与既有持仓都能拿到足够预热数据
            required_intervals = dict.fromkeys(symbols, interval)
            for symbol, binding in pairs:
                current = required_intervals.get(symbol)
                if current is None or interval_ms(binding.interval) > interval_ms(current):
                    required_intervals[symbol] = binding.interval
            # Existing positions keep using their own candle series, so an interval
            # change must never stop updating them.
            for order in open_orders(self.db):
                existing = required_intervals.get(order.symbol)
                if existing is None or interval_ms(order.interval) > interval_ms(existing):
                    required_intervals[order.symbol] = order.interval
            sync_groups = {}
            for symbol, required_interval in required_intervals.items():
                sync_groups.setdefault(required_interval, set()).add(symbol)
            sync_errors = []
            for sync_interval, sync_symbols in sync_groups.items():
                sync_errors.extend(_sync_recent_many(sorted(sync_symbols), sync_interval))
            sync_target_count = sum(len(group) for group in sync_groups.values())
            synced_at = time.monotonic()

            # 同一币种可能落进多个池/周期：各自独立评估，再按评分择优只开一仓
            best_signals = {}
            strategy_signals = {}
            initially_held = {order.symbol for order in open_orders(self.db)}
            signal_errors = list(binding_errors)
            for symbol, binding in pairs:
                try:
                    signal = engine.evaluate(
                        symbol,
                        binding.interval,
                        binding.strategy,
                        save=True,
                        params=binding.params_for(symbol),
                        pool=binding.pool,
                    )
                    strategy_signals[(symbol, binding.interval, binding.strategy, binding.pool)] = signal
                    if signal is not None:
                        signal.binding_id = binding.id
                    if signal is None:
                        signal_errors.append(f"{symbol}/{binding.strategy}×{binding.pool}: 数据不足或K线无效，无法生成信号")
                        continue
                    if (signal.strategy in RESEARCH_CODES and signal.interval == "1h"
                            and symbol not in initially_held and signal.reason.get("research_entry")
                            and signal.reason.get("block_reason") != "stale_data"):
                        pending_key = _research_pending_key(symbol, binding.strategy, binding.pool)
                        pending = get_state(self.db, pending_key, None)
                        if not pending:
                            pending = signal.to_dict()
                            pending["params"] = signal.params
                            pending["binding_id"] = binding.id
                            set_state(self.db, pending_key, pending)
                    if signal.direction == "FLAT":
                        continue
                    current = best_signals.get(symbol)
                    if current is None or signal.score > current.score:
                        best_signals[symbol] = signal
                except Exception as exc:
                    self.db.rollback()
                    signal_errors.append(f"{symbol}/{binding.strategy}×{binding.pool}: {exc}")
            signaled_at = time.monotonic()

            # 管理已有订单：live 订单先同步交易所状态（交易所侧止盈止损可能已触发）
            managed = []
            closed_symbols = set()
            for order in open_orders(self.db):
                if order.source == "live":
                    live_order_id = order.id
                    try:
                        if sync_live(self.db, order) is not None:
                            closed_symbols.add(order.symbol)
                            managed.append({"order_id": order.id, "actions": ["exchange_closed_sync"]})
                            continue
                    except Exception as exc:
                        self.db.rollback()
                        sync_errors.append(f"live #{live_order_id}: {exc}")
                        self._log("error", f"live 订单同步失败 #{live_order_id}: {exc}")
                        continue
                key = (order.symbol, order.interval, order.strategy, order.pool or "")
                signal = None
                if order.params_snapshot is not None:
                    order_id = order.id
                    try:
                        params = decode_order_params(order.strategy, order.interval, order.params_snapshot)
                        signal = engine.evaluate(
                            order.symbol, order.interval, order.strategy,
                            params=params, pool=order.pool or "", respect_enabled=False,
                        )
                    except Exception as exc:
                        self.db.rollback()
                        message = f"订单 #{order_id} 信号计算失败: {type(exc).__name__}: {exc}"
                        signal_errors.append(message)
                        self._log("error", message)
                else:
                    signal = strategy_signals.get(key)
                if signal is None and order.params_snapshot is None:
                    # 老订单没有记录 pool：退回该 (币种, 周期, 策略) 上的任意一条信号
                    signal = next(
                        (
                            value
                            for (symbol, interval, strategy, _pool), value in strategy_signals.items()
                            if (symbol, interval, strategy)
                            == (order.symbol, order.interval, order.strategy)
                        ),
                        None,
                    )
                order_id = order.id
                try:
                    result = manage_order(self.db, order, signal)
                except Exception as exc:
                    self.db.rollback()
                    message = f"订单 #{order_id} 管理失败: {type(exc).__name__}: {exc}"
                    signal_errors.append(message)
                    self._log("error", message)
                    continue
                if result["actions"]:
                    managed.append(result)
                if order.status != 'OPEN':
                    # 反手型策略（Pine 止损单系统）：反向成交在同一根K线上同时
                    # 平旧仓开新仓。若按「刚平仓」拦掉本周期开仓，反手后的新仓
                    # 会在后续周期因触价条件消失而永久漏开，这里放行同周期反手。
                    if 'reverse_on_stop' in get_spec(order.strategy).tags and (
                        'signal_reverse' in result['actions'] or 'stop_loss' in result['actions']
                    ):
                        pass
                    else:
                        closed_symbols.add(order.symbol)

            # 无持仓的币种按信号开仓；刚止损/止盈的币种本周期不再追单
            opened = []
            decisions = []
            held_symbols = {order.symbol for order in open_orders(self.db)}
            for symbol, binding in pairs:
                if binding.strategy not in RESEARCH_CODES or binding.interval != "1h":
                    continue
                pending_key = _research_pending_key(symbol, binding.strategy, binding.pool)
                pending = get_state(self.db, pending_key, None)
                if not pending:
                    continue
                if (pending["ts"] + 2 * interval_ms("1h") <= now_ms()
                        or pending["ts"] <= get_state(self.db, f"last_entry:{symbol}:1h", 0)):
                    set_state(self.db, pending_key, None)
                    continue
                if symbol in held_symbols or symbol in closed_symbols:
                    set_state(self.db, pending_key, None)
                    continue
                pending["direction"] = "LONG"
                order = open_position(self.db, SimpleNamespace(**pending))
                if order is not None:
                    set_state(self.db, pending_key, None)
                    held_symbols.add(symbol)
                    opened.append(order.id)
                    decisions.append(f"{symbol} LONG open #{order.id} research")

            for symbol, signal in best_signals.items():
                if symbol in closed_symbols:
                    decisions.append(f"{symbol} {signal.direction} skipped (本周期刚平仓)")
                    continue
                if symbol in held_symbols:
                    decisions.append(f"{symbol} {signal.direction} hold score={signal.score}")
                    continue
                order = open_position(self.db, signal)
                if order is not None:
                    held_symbols.add(symbol)
                    opened.append(order.id)
                    decisions.append(f"{symbol} {signal.direction} open #{order.id} score={signal.score}")
                else:
                    decisions.append(f"{symbol} {signal.direction} skipped (冷却、重复信号或风险/资金限制)")

            snapshot_equity(self.db)
            elapsed_ms = int((time.monotonic() - started) * 1000)
            summary = {
                "ok": True,
                "time": iso_utc(),
                "symbols": symbols,
                "symbol_count": len(symbols),
                "bindings": [
                    {
                        "strategy": binding.strategy,
                        "pool": binding.pool,
                        "interval": binding.interval,
                        "symbol_count": len(binding.symbols),
                        # 出场口径随绑定走，巡检回执里带上可读标签便于排障
                        "exit": binding.exit_label,
                    }
                    for binding in bindings
                ],
                "sync_target_count": sync_target_count,
                "elapsed_ms": elapsed_ms,
                "sync_ms": int((synced_at - started) * 1000),
                "signal_ms": int((signaled_at - synced_at) * 1000),
                "execution_ms": int((time.monotonic() - signaled_at) * 1000),
                "interval": interval,
                "signals": [signal.to_dict() for signal in best_signals.values()],
                "decisions": decisions,
                "opened": opened,
                "managed": managed,
                "sync_errors": sync_errors,
                "signal_errors": signal_errors,
            }
            self._log_summary(summary)
            return summary
        except Exception as exc:
            self.db.rollback()
            self._log("error", f"交易周期异常: {exc}\n{traceback.format_exc(limit=3)}")
            return {"ok": False, "message": str(exc)}
        finally:
            self._lock.release()

    def _log_summary(self, summary: dict) -> None:
        """周期日志存紧凑摘要：全量 symbols/decisions 留在 API 返回值，不进持久化日志，
        避免 4000 字符截断把 symbol_count/elapsed_ms 挤掉。sync 部分失败记 warning，
        全部失败才记 error（526 币种下个别永续无K线属预期，不应制造 error 噪声）。"""
        sync_errors = summary.get("sync_errors", [])
        signal_errors = summary.get("signal_errors", [])
        level = "info"
        if signal_errors:
            level = "warning"
        if sync_errors:
            target_count = summary.get("sync_target_count", summary.get("symbol_count", 0))
            all_failed = bool(target_count) and len(sync_errors) >= target_count
            level = "error" if all_failed else "warning"
        compact = {
            "ok": summary.get("ok"),
            "time": summary.get("time"),
            "symbol_count": summary.get("symbol_count"),
            "sync_target_count": summary.get("sync_target_count"),
            "elapsed_ms": summary.get("elapsed_ms"),
            "sync_ms": summary.get("sync_ms"),
            "signal_ms": summary.get("signal_ms"),
            "execution_ms": summary.get("execution_ms"),
            "opened": len(summary.get("opened", [])),
            "managed": len(summary.get("managed", [])),
            "sync_errors": [f"{len(sync_errors)} 条"] + sync_errors[:2],
            "signal_errors": [f"{len(signal_errors)} 条"] + signal_errors[:2],
        }
        self._log(level, json.dumps(compact, ensure_ascii=False))

    def manual_close(self, order_id: int, reason: str = "manual") -> Order | None:
        order = self.db.get(Order, order_id)
        if order is None or order.status != "OPEN":
            return None
        mark = order.mark_price or order.entry_price
        return close_position(self.db, order, mark, reason)

    def reverse_order(self, order_id: int) -> Order | None:
        order = self.db.get(Order, order_id)
        if order is None or order.status != "OPEN":
            return None
        if order.strategy in RESEARCH_CODES:
            raise ValueError("研究订单不支持手动反向开仓")
        if order.params_snapshot is not None:
            params = decode_order_params(order.strategy, order.interval, order.params_snapshot)
        else:
            params = resolve_order_params(
                self.db, order.strategy, order.interval, order.pool or "", symbol=order.symbol
            )
            encode_order_params(order.strategy, order.interval, params)
        mark = order.mark_price or order.entry_price
        close_position(self.db, order, mark, "reverse")
        if order.side == "LONG":
            direction, sl, tp = "SHORT", mark * 1.01, mark * 0.98
        else:
            direction, sl, tp = "LONG", mark * 0.99, mark * 1.02
        signal = type(
            "Signal",
            (),
            {
                "symbol": order.symbol,
                "interval": order.interval,
                "direction": direction,
                "price": mark,
                "sl": sl,
                "tp": tp,
                "leverage": order.leverage,
                "position_pct": order.margin / max(portfolio_snapshot(self.db)["equity"], 1),
                "strategy": order.strategy,
                "pool": order.pool or "",
                "binding_id": order.binding_id,
                "params": params,
                "score": 0.6,
                "prob_up": 0.5,
            },
        )()
        return open_position(self.db, signal, ignore_cooldown=True)

    def snapshot(self) -> dict:
        return portfolio_snapshot(self.db)
