"""执行模式分发：按 execution_mode 把开仓路由到 paper 或 live；
平仓/改单按订单自身 source 路由，避免模式切换后遗留订单走错通道。

账户级熔断：risk_halt=true 时拒绝一切新开仓（含手动反手），已持仓单子继续管理，
需人工在设置页把 risk_halt 置回 false 恢复。
"""

from __future__ import annotations

import time

from sqlalchemy.orm import Session

from ..models import JobLog, Order
from ..state import get_state
from . import live, paper
from .risk_halt import halt_status

# 熔断中的拒绝日志做节流：run_cycle 会对每个候选币种调一次开仓入口，
# 全量宇宙下逐币种记日志会刷爆 JobLog，按 60s 最多记一条。
_HALT_LOG_INTERVAL_SECONDS = 60.0
_last_halt_log_at = 0.0


def execution_mode(db: Session) -> str:
    return str(get_state(db, "execution_mode", "paper") or "paper")


def is_live(db: Session) -> bool:
    return execution_mode(db) == "live"


def _log_halt(db: Session, reason: str) -> None:
    global _last_halt_log_at
    now = time.monotonic()
    if now - _last_halt_log_at < _HALT_LOG_INTERVAL_SECONDS:
        return
    _last_halt_log_at = now
    try:
        db.add(JobLog(job="risk_halt", level="warning",
                      message=f"账户熔断中，拒绝开仓: {reason or '未记录原因'}"[:4000]))
        db.commit()
    except Exception:  # noqa: BLE001 - 日志失败不影响熔断拒绝本身
        pass


def open_position(db: Session, signal, equity: float | None = None, ignore_cooldown: bool = False) -> Order | None:
    halted, reason = halt_status(db)
    if halted:
        _log_halt(db, reason)
        return None
    if is_live(db):
        return live.open_live(db, signal)
    return paper.open_position(db, signal, equity=equity, ignore_cooldown=ignore_cooldown)


def close_position(db: Session, order: Order, price: float, reason: str = "manual") -> Order:
    if order.source == "live":
        return live.close_live(db, order, reason, fallback_price=price)
    return paper.close_position(db, order, price, reason)


def modify_order(db: Session, order: Order, sl: float | None = None, tp: float | None = None) -> Order:
    if order.source == "live":
        return live.modify_live(db, order, sl=sl, tp=tp)
    return paper.modify_order(order, sl=sl, tp=tp)
