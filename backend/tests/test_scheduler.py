"""交易调度器测试：启动/暂停/恢复/停止的状态机，暂停后恢复不得抛
SchedulerAlreadyRunningError（前序缺陷：stop 只删 job 不关底层 scheduler）。"""

from app.trading.scheduler import TradingScheduler


def _detached_scheduler() -> TradingScheduler:
    """隔离实例：_run 换成空操作，避免测试触发真实交易周期。"""
    instance = TradingScheduler()
    instance._run = lambda: None
    return instance


def test_pause_then_resume_does_not_raise():
    instance = _detached_scheduler()
    try:
        instance.apply(True, 3600)
        assert instance.status()["running"] is True
        instance.apply(False, 3600)
        assert instance.status()["running"] is False
        # 恢复：底层 BackgroundScheduler 仍在运行，必须容忍而不是抛异常
        instance.apply(True, 3600)
        assert instance.status()["running"] is True
        assert instance.status()["next_run"] is not None
    finally:
        instance.stop()


def test_reschedule_keeps_single_job():
    instance = _detached_scheduler()
    try:
        instance.apply(True, 3600)
        instance.apply(True, 120)  # 已启动 → 改周期不新增 job
        jobs = instance.scheduler.get_jobs()
        assert len(jobs) == 1
    finally:
        instance.stop()
        assert instance.status()["running"] is False
