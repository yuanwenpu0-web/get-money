from __future__ import annotations

import json
import threading
import time
import traceback

from sqlalchemy.orm import Session

from ..config import MARKET_SYNC_WORKERS
from ..db import SessionLocal
from ..market.data_service import MarketDataService
from ..models import JobLog, Order
from ..state import all_state
from ..strategy.bindings import active_bindings, expand
from ..strategy.engine import StrategyEngine
from ..utils import interval_ms, iso_utc
from .order_manager import manage_order
from .dispatch import close_position, open_position
from .live_sync import sync_live
from .portfolio import open_orders, portfolio_snapshot, snapshot_equity
from .risk_halt import halt_status

SYNC_WORKERS = MARKET_SYNC_WORKERS


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
                    if signal is None or signal.direction == "FLAT":
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
                    try:
                        if sync_live(self.db, order) is not None:
                            closed_symbols.add(order.symbol)
                            managed.append({"order_id": order.id, "actions": ["exchange_closed_sync"]})
                            continue
                    except Exception as exc:
                        sync_errors.append(f"live #{order.id}: {exc}")
                        self._log("error", f"live 订单同步失败 #{order.id}: {exc}")
                        continue
                key = (order.symbol, order.interval, order.strategy, order.pool or "")
                signal = strategy_signals.get(key)
                if signal is None:
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
                result = manage_order(self.db, order, signal)
                if result["actions"]:
                    managed.append(result)
                if order.status != 'OPEN':
                    closed_symbols.add(order.symbol)

            # 无持仓的币种按信号开仓；刚止损/止盈的币种本周期不再追单
            # 账户熔断中只拒新仓，已持仓单子的管理在上面照常执行
            halted, halt_reason = halt_status(self.db)
            opened = []
            decisions = []
            held_symbols = {order.symbol for order in open_orders(self.db)}
            for symbol, signal in best_signals.items():
                if halted:
                    decisions.append(f"{symbol} {signal.direction} skipped (账户熔断中: {halt_reason})")
                    continue
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
                "score": 0.6,
                "prob_up": 0.5,
            },
        )()
        return open_position(self.db, signal, ignore_cooldown=True)

    def snapshot(self) -> dict:
        return portfolio_snapshot(self.db)
