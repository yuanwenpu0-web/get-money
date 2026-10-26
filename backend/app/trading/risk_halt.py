"""账户级熔断（风控底线，实盘前最后一道保险）。

在每笔平仓记账后检查两个条件，任一满足即冻结开仓（已持仓单子继续管理，只是不开新仓）：
1. 总回撤 (peak_equity - equity) / peak_equity >= max_drawdown_halt_pct
   peak_equity 取权益曲线历史峰值，无曲线时用 initial_capital / total_capital 兜底
2. 连续亏损笔数 >= consecutive_loss_halt

触发后写 risk_halt=true + risk_halt_reason + JobLog(error)；需人工复盘后在设置页
把 risk_halt 置回 false 才恢复开仓。
"""

from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models import EquitySnapshot, JobLog, Order
from ..state import all_state, set_state
from .portfolio import portfolio_snapshot


def _current_equity(db: Session) -> float:
    """平仓后的账户权益（不更新标记价，避免熔断检查触发额外行情请求）。"""
    return float(portfolio_snapshot(db, update_marks=False)["equity"])


def _peak_equity(db: Session, equity: float) -> float:
    """历史峰值：优先权益曲线最大值，无曲线时用 initial/total capital 兜底。"""
    state = all_state(db)
    initial = float(state.get("initial_capital", 0) or 0)
    total = float(state.get("total_capital", 0) or 0)
    historical = db.query(func.max(EquitySnapshot.equity)).scalar() or 0.0
    return max(float(historical), initial, total, equity)


def _consecutive_losses(db: Session) -> int:
    """按平仓时间倒序数连续亏损笔数；遇到一笔非亏损（含 0 盈亏）即中断。"""
    rows = (
        db.query(Order.realized_pnl)
        .filter(Order.status == "CLOSED")
        .order_by(Order.closed_at.desc(), Order.id.desc())
        .all()
    )
    streak = 0
    for (pnl,) in rows:
        if pnl is None or pnl >= 0:
            break
        streak += 1
    return streak


def check_after_close(db: Session) -> None:
    """平仓记账后调用：触发条件满足则置 risk_halt 并记录 JobLog(error)。已熔断时跳过。
    检查自身失败只记日志、不上抛——熔断是监控逻辑，不能反过来拖垮交易周期。"""
    try:
        _evaluate_halt(db)
    except Exception as exc:  # noqa: BLE001 - 风控检查失败必须留痕但不阻断周期
        try:
            db.rollback()
            db.add(JobLog(job="risk_halt", level="error", message=f"熔断检查失败: {exc}"[:4000]))
            db.commit()
        except Exception:  # noqa: BLE001 - 连日志都写不出去时只能放弃，避免异常上抛
            pass


def _evaluate_halt(db: Session) -> None:
    state = all_state(db)
    if bool(state.get("risk_halt", False)):
        return
    equity = _current_equity(db)
    reasons: list[str] = []

    drawdown_limit = float(state.get("max_drawdown_halt_pct", 0.10) or 0)
    if drawdown_limit > 0:
        peak = _peak_equity(db, equity)
        if peak > 0:
            drawdown = (peak - equity) / peak
            if drawdown >= drawdown_limit:
                reasons.append(
                    f"总回撤 {drawdown * 100:.2f}% 达到熔断线 {drawdown_limit * 100:.1f}%"
                    f"（峰值 {peak:.2f} → 当前 {equity:.2f}）"
                )

    loss_limit = int(state.get("consecutive_loss_halt", 5) or 0)
    if loss_limit > 0:
        streak = _consecutive_losses(db)
        if streak >= loss_limit:
            reasons.append(f"连续亏损 {streak} 笔达到熔断线 {loss_limit} 笔")

    if not reasons:
        return
    reason = "；".join(reasons)
    set_state(db, "risk_halt", True)
    set_state(db, "risk_halt_reason", reason)
    db.add(JobLog(job="risk_halt", level="error", message=f"账户熔断触发: {reason}"))
    db.commit()


def halt_status(db: Session) -> tuple[bool, str]:
    """返回 (是否熔断中, 熔断原因)。开仓入口据此拒绝新仓。"""
    state = all_state(db)
    return bool(state.get("risk_halt", False)), str(state.get("risk_halt_reason") or "")
