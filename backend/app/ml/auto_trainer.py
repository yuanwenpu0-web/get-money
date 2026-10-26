"""模型持续训练管线 + 训练历史 + 晋升门禁（迭代 9）。

设计原则（硬约束）：
- 持续训练本身无害（只耗 CPU、不改任何决策）→ 默认开启；
- 但 ml_ensemble 的启用必须过晋升门禁，门禁不过就保持 disabled；
- 绝不因为"AUC 看起来变好了"就放行——迭代 8 已抓到过会伪造改善的缺陷，
  故门禁要求"连续 N 轮 + AUC 阈值 + 样本外下限"三项同时满足，且只记录候选、
  真启用还需 ml_auto_enable 显式打开。

本模块不碰 API 层，路由只做装配。
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ..db import SessionLocal
from ..market.universe import universe_symbols, universe_volumes
from ..models import JobLog, ModelRecord, StrategyConfig
from ..state import all_state
from .predictor import latest_model
from .trainer import train_model

MIN_TEST_ROWS = 200        # 晋升门禁：单轮样本外下限
AUTO_TRAIN_POLL_SECONDS = 600  # 守护线程轮询间隔（真正节流靠 auto_train_interval_hours）
ML_ENSEMBLE_CODE = "ml_ensemble"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _metrics_of(record: ModelRecord) -> dict:
    try:
        return json.loads(record.metrics or "{}") or {}
    except (TypeError, ValueError):
        return {}


def _target_symbols(db, state: dict, top_n: int) -> list[str]:
    """目标币种：universe_symbols 已含流动性过滤 + 黑名单，此处按 24h 成交额取 TopN。"""
    symbols = [str(item).strip().upper() for item in universe_symbols(db) if str(item).strip()]
    volumes = universe_volumes(db)
    if volumes:
        symbols.sort(key=lambda symbol: float(volumes.get(symbol, 0.0)), reverse=True)
    return symbols[: max(int(top_n), 1)]


def _is_fresh(record: ModelRecord, hours: float) -> bool:
    created = record.created_at
    if created is None:
        return False
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return (_now() - created) < timedelta(hours=float(hours))


def _training_workers(db, state: dict, targets: int) -> int:
    """Each estimator already uses two native threads; bound the outer pool too.

    In-memory SQLite and connection-bound sessions cannot provide independent
    worker transactions, so preserve the caller's serial session in those cases.
    """
    bind = db.get_bind()
    if not isinstance(bind, Engine):
        return 1
    if bind.dialect.name == "sqlite" and (
        not bind.url.database or bind.url.database == ":memory:"
        or bind.url.query.get("mode") == "memory" or isinstance(bind.pool, StaticPool)
    ):
        return 1
    requested = max(1, min(int(state.get("auto_train_workers", 2)), 8))
    cpu_budget = max(1, (os.cpu_count() or 2) // 2)
    return max(1, min(requested, targets, cpu_budget))


def _train_symbol(db, symbol: str, interval: str, horizon: int, days: int,
                  fresh_hours: float) -> tuple[str, dict]:
    try:
        existing = latest_model(db, symbol, interval)
        if existing is not None and _is_fresh(existing, fresh_hours):
            return "skipped", {"symbol": symbol, "model_id": existing.id, "reason": "fresh"}
        record = train_model(db, symbol, interval, algo="lightgbm", horizon=horizon, days=days)
        metrics = _metrics_of(record)
        result = {
            "symbol": symbol, "model_id": record.id, "auc": metrics.get("auc"),
            "accuracy": metrics.get("accuracy"), "test_rows": metrics.get("test_rows"),
            "rows": record.rows,
        }
        db.add(JobLog(
            job="auto_train", level="info",
            message=f"{symbol} {interval} 训练完成 #{record.id} "
                    f"AUC={metrics.get('auc')} 样本 {record.rows}",
        ))
        db.commit()
        return "trained", result
    except Exception as exc:  # noqa: BLE001 - one failure must not poison the next transaction
        db.rollback()
        db.add(JobLog(job="auto_train", level="error",
                      message=f"{symbol} {interval} 训练失败: {exc}"[:4000]))
        db.commit()
        return "failed", {"symbol": symbol, "error": str(exc)[:300]}


# ---------- A. 持续训练 ----------

def run_auto_train_cycle(db, top_n: int | None = None, symbols: list[str] | None = None) -> dict:
    """跑一轮自动训练。开关关闭直接返回；单币种失败吞掉并记 JobLog，不中断整轮。

    top_n / symbols 为显式覆盖（脚本按需扩面用），不传则沿用 state 的 auto_train_top_n。
    """
    state = all_state(db)
    if not bool(state.get("auto_train_enabled", True)):
        return {"trained": [], "skipped": [], "failed": [], "enabled": False,
                "reason": "auto_train_enabled=False"}
    interval = str(state.get("interval") or "1h")
    days = int(state.get("auto_train_days") or 60)
    horizon = int(state.get("train_horizon") or 5)
    top_n = int(top_n if top_n is not None else state.get("auto_train_top_n") or 10)
    # 注意：0 是合法语义（无冷却、每次都重训），不能用 `or 24` 否则 0 会被吞成 24
    raw_hours = state.get("auto_train_interval_hours")
    fresh_hours = 24.0 if raw_hours in (None, "") else float(raw_hours)
    try:
        if symbols is not None:
            parsed = [str(item).strip().upper() for item in symbols if str(item).strip()]
        else:
            parsed = _target_symbols(db, state, top_n)
    except Exception as exc:  # noqa: BLE001 - 宇宙解析失败整轮跳过，不逐个重试
        db.add(JobLog(job="auto_train", level="error", message=f"币种宇宙解析失败: {exc}"[:4000]))
        db.commit()
        return {"trained": [], "skipped": [], "failed": [], "enabled": True,
                "error": str(exc)[:300]}

    # A repeated explicit symbol must not race itself and create duplicate models.
    symbols = list(dict.fromkeys(parsed))
    workers = _training_workers(db, state, len(symbols))
    results: dict[str, list[dict]] = {"trained": [], "skipped": [], "failed": []}
    if workers == 1:
        for symbol in symbols:
            kind, result = _train_symbol(db, symbol, interval, horizon, days, fresh_hours)
            results[kind].append(result)
    else:
        sessions = sessionmaker(bind=db.get_bind(), autoflush=False, expire_on_commit=False)

        def train_in_session(symbol: str) -> tuple[str, dict]:
            with sessions() as worker_db:
                return _train_symbol(worker_db, symbol, interval, horizon, days, fresh_hours)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="auto_train") as executor:
            # map preserves target order even when later symbols finish first.
            for kind, result in executor.map(train_in_session, symbols):
                results[kind].append(result)
    return {**results, "enabled": True, "workers": workers,
            "interval": interval, "days": days, "top_n": top_n,
            "fresh_hours": fresh_hours}


def start_auto_train_thread() -> threading.Event:
    """启动守护线程（main.py lifespan 调用）。固定轮询，真正的节流由
    auto_train_interval_hours 的新鲜度判断完成；不触碰 schedule_enabled/schedule_seconds。"""
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            try:
                db = SessionLocal()
                try:
                    run_auto_train_cycle(db)
                finally:
                    db.close()
            except Exception as exc:  # noqa: BLE001 - 守护线程永不因异常退出
                print(f"[auto_train] 循环异常: {exc}"[:300], flush=True)
            stop.wait(AUTO_TRAIN_POLL_SECONDS)

    threading.Thread(target=loop, name="auto_train", daemon=True).start()
    return stop


# ---------- B. 训练历史 ----------

def list_training_history(db, symbol: str, interval: str, limit: int = 50) -> list[dict]:
    """最近 limit 轮训练记录，按 created_at 升序返回（旧→新），便于看趋势。"""
    rows = (
        db.query(ModelRecord)
        .filter(ModelRecord.symbol == symbol.upper(), ModelRecord.interval == interval)
        .order_by(ModelRecord.created_at.desc(), ModelRecord.id.desc())
        .limit(max(int(limit), 1))
        .all()
    )
    history = []
    for record in reversed(rows):
        metrics = _metrics_of(record)
        history.append({
            "id": record.id,
            "created_at": _iso(record.created_at),
            "auc": metrics.get("auc"),
            "accuracy": metrics.get("accuracy"),
            "rows": record.rows,
            "test_rows": metrics.get("test_rows"),
            "train_start": record.train_start,
            "train_end": record.train_end,
            "status": record.status,
            "algo": record.algo,
        })
    return history


# ---------- C. 晋升门禁 ----------

def evaluate_promotion(db, symbol: str, interval: str, state: dict | None = None) -> dict:
    """连续 ml_promote_rounds 轮全部满足 status=ready、auc>=ml_min_auc、
    test_rows>=MIN_TEST_ROWS 才算 promotable。任一条不满足即给具体原因。"""
    state = state or all_state(db)
    rounds = max(int(state.get("ml_promote_rounds") or 2), 1)
    min_auc = float(state.get("ml_min_auc") or 0.55)
    rows = (
        db.query(ModelRecord)
        .filter(ModelRecord.symbol == symbol.upper(), ModelRecord.interval == interval)
        .order_by(ModelRecord.created_at.desc(), ModelRecord.id.desc())
        .limit(rounds)
        .all()
    )
    ordered = list(reversed(rows))  # 统一按时间升序（旧→新）对外，与训练历史口径一致
    verdict = {
        "symbol": symbol.upper(), "interval": interval, "promotable": False,
        "rounds_checked": len(rows), "rounds_required": rounds,
        "aucs": [_metrics_of(row).get("auc") for row in ordered],
        "min_auc": min_auc, "min_test_rows": MIN_TEST_ROWS, "reason": "",
    }
    if len(rows) < rounds:
        verdict["reason"] = f"训练轮数不足 {len(rows)}/{rounds}"
        return verdict
    for row in rows:
        metrics = _metrics_of(row)
        if row.status != "ready":
            verdict["reason"] = f"#{row.id} 状态非 ready（{row.status}）"
            return verdict
        auc = metrics.get("auc")
        if auc is None or float(auc) < min_auc:
            verdict["reason"] = f"#{row.id} AUC={auc} 未达 {min_auc}"
            return verdict
        if int(metrics.get("test_rows") or 0) < MIN_TEST_ROWS:
            verdict["reason"] = f"#{row.id} 测试样本 {metrics.get('test_rows')} < {MIN_TEST_ROWS}"
            return verdict
    verdict["promotable"] = True
    verdict["reason"] = f"连续 {rounds} 轮达标（AUC>={min_auc}、test_rows>={MIN_TEST_ROWS}）"
    return verdict


def _ml_ensemble_config(db) -> StrategyConfig | None:
    return db.query(StrategyConfig).filter(StrategyConfig.code == ML_ENSEMBLE_CODE).first()


def ml_ensemble_enabled(db) -> bool:
    config = _ml_ensemble_config(db)
    return bool(config.enabled) if config is not None else False


def apply_promotion(db, symbols: list[str] | None = None, interval: str | None = None) -> dict:
    """对所有目标币种跑门禁。promotable 且 ml_auto_enable=true 才真正启用 ml_ensemble；
    否则只记录候选，绝不启用（红线）。"""
    state = all_state(db)
    interval = interval or str(state.get("interval") or "1h")
    if symbols is None:
        symbols = _target_symbols(db, state, int(state.get("auto_train_top_n") or 10))
    auto_enable = bool(state.get("ml_auto_enable", False))
    evaluations: list[dict] = []
    candidates: list[str] = []
    promoted: list[str] = []
    for symbol in symbols:
        verdict = evaluate_promotion(db, symbol, interval, state)
        evaluations.append(verdict)
        if not verdict["promotable"]:
            continue
        if not auto_enable:
            candidates.append(symbol)
            continue
        config = _ml_ensemble_config(db)
        if config is not None and not config.enabled:
            config.enabled = 1
            db.commit()
        promoted.append(symbol)
    message = (f"晋升评估：候选 {len(candidates)} 个，已启用 {len(promoted)} 个；"
               f"ml_auto_enable={auto_enable}")
    db.add(JobLog(job="ml_promotion", level="info" if promoted else "warning",
                  message=message))
    db.commit()
    return {"auto_enable": auto_enable, "evaluations": evaluations,
            "candidates": candidates, "promoted": promoted,
            "ml_ensemble_enabled": ml_ensemble_enabled(db)}
