from __future__ import annotations

import threading
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler

from ..db import SessionLocal
from ..state import get_state, set_state
from .service import TradingService


class TradingScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler(timezone="UTC")
        self._started = False
        self._job = None
        self._lock = threading.Lock()

    def _run(self) -> None:
        db = SessionLocal()
        try:
            TradingService(db).run_cycle()
        finally:
            db.close()

    def start(self, seconds: int) -> None:
        with self._lock:
            seconds = max(10, int(seconds))
            if self._started and self._job is not None:
                self.scheduler.reschedule_job(self._job.id, trigger="interval", seconds=seconds)
                return
            self._job = self.scheduler.add_job(
                self._run,
                "interval",
                seconds=seconds,
                id="trading_cycle",
                max_instances=1,
                coalesce=True,
                next_run_time=datetime.now(timezone.utc),
            )
            self.scheduler.start()
            self._started = True

    def stop(self) -> None:
        with self._lock:
            if self._job is not None:
                try:
                    self.scheduler.remove_job(self._job.id)
                except Exception:
                    pass
                self._job = None
            self._started = False

    def apply(self, enabled: bool, seconds: int) -> None:
        if enabled:
            self.start(seconds)
        else:
            self.stop()

    def status(self) -> dict:
        next_run = None
        if self._job is not None:
            job = self.scheduler.get_job(self._job.id)
            if job and job.next_run_time:
                next_run = job.next_run_time.isoformat()
        return {"running": self._started, "next_run": next_run}


scheduler = TradingScheduler()


def sync_scheduler_with_state(db) -> dict:
    enabled = bool(get_state(db, "schedule_enabled", True))
    seconds = int(get_state(db, "schedule_seconds", 60))
    scheduler.apply(enabled, seconds)
    return scheduler.status()


def update_schedule(db, enabled: bool | None = None, seconds: int | None = None) -> dict:
    if enabled is None:
        enabled = bool(get_state(db, "schedule_enabled", True))
    if seconds is None:
        seconds = int(get_state(db, "schedule_seconds", 60))
    seconds = max(10, min(3600, int(seconds)))
    set_state(db, "schedule_enabled", enabled)
    set_state(db, "schedule_seconds", seconds)
    scheduler.apply(enabled, seconds)
    return scheduler.status()
