from __future__ import annotations

import json
import threading
import traceback

from sqlalchemy.orm import Session

from ..market.data_service import MarketDataService
from ..models import JobLog, Order, StrategyConfig
from ..state import all_state
from ..strategy.engine import StrategyEngine
from ..utils import iso_utc
from .order_manager import manage_order
from .paper import close_position, open_position
from .portfolio import open_orders, portfolio_snapshot, snapshot_equity


class TradingService:
    _lock = threading.Lock()

    def __init__(self, db: Session):
        self.db = db

    def _log(self, level: str, message: str) -> None:
        self.db.add(JobLog(job="trading_cycle", level=level, message=message[:4000]))
        self.db.commit()

    def _enabled_strategies(self) -> list[str]:
        rows = self.db.query(StrategyConfig).filter(StrategyConfig.enabled == 1).all()
        return [row.code for row in rows]

    def run_cycle(self) -> dict:
        if not self._lock.acquire(blocking=False):
            return {"ok": False, "message": "已有交易周期在运行"}
        try:
            state = all_state(self.db)
            symbols = state.get("symbols") or []
            interval = state.get("interval", "1m")
            codes = self._enabled_strategies()
            engine = StrategyEngine(self.db)
            data_service = MarketDataService(self.db)
            sync_errors = []

            for symbol in symbols:
                try:
                    if state.get("data_source", "nofx") == "nofx":
                        from ..market import nofx_source

                        nofx_source.import_klines(self.db, symbol, interval, days=0, incremental=True)
                    else:
                        data_service.sync_recent(symbol, interval)
                except Exception as exc:  # 数据源异常不阻塞已有持仓管理
                    sync_errors.append(f"{symbol}: {exc}")

            # 每个币种选评分最高的策略信号
            best_signals = {}
            strategy_signals = {}
            signal_errors = []
            for symbol in symbols:
                for code in codes:
                    try:
                        signal = engine.evaluate(symbol, interval, code, save=True)
                        strategy_signals[(symbol, interval, code)] = signal
                        if signal is None or signal.direction == "FLAT":
                            continue
                        current = best_signals.get(symbol)
                        if current is None or signal.score > current.score:
                            best_signals[symbol] = signal
                    except Exception as exc:
                        signal_errors.append(f"{symbol}/{code}: {exc}")

            # 管理已有订单
            managed = []
            closed_symbols = set()
            for order in open_orders(self.db):
                binding = strategy_signals.get((order.symbol, order.interval, order.strategy))
                result = manage_order(self.db, order, binding)
                if result["actions"]:
                    managed.append(result)
                if order.status != 'OPEN':
                    closed_symbols.add(order.symbol)

            # 无持仓的币种按信号开仓；刚止损/止盈的币种本周期不再追单
            opened = []
            decisions = []
            for symbol, signal in best_signals.items():
                if symbol in closed_symbols:
                    decisions.append(f"{symbol} {signal.direction} skipped (本周期刚平仓)")
                    continue
                has_open = any(o.symbol == symbol for o in open_orders(self.db))
                if has_open:
                    decisions.append(f"{symbol} {signal.direction} hold score={signal.score}")
                    continue
                order = open_position(self.db, signal)
                if order is not None:
                    opened.append(order.id)
                    decisions.append(f"{symbol} {signal.direction} open #{order.id} score={signal.score}")
                else:
                    decisions.append(f"{symbol} {signal.direction} skipped (冷却、重复信号或风险/资金限制)")

            snapshot_equity(self.db)
            summary = {
                "ok": True,
                "time": iso_utc(),
                "symbols": symbols,
                "interval": interval,
                "signals": [signal.to_dict() for signal in best_signals.values()],
                "decisions": decisions,
                "opened": opened,
                "managed": managed,
                "sync_errors": sync_errors,
                "signal_errors": signal_errors,
            }
            self._log("error" if sync_errors else "info", json.dumps(summary, ensure_ascii=False))
            return summary
        except Exception as exc:
            self._log("error", f"交易周期异常: {exc}\n{traceback.format_exc(limit=3)}")
            return {"ok": False, "message": str(exc)}
        finally:
            self._lock.release()

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
