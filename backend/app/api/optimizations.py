"""参数寻优任务：创建、进度、结果、应用。

任务模型
--------
寻优是分钟级的长任务，不能挂在 HTTP 请求里同步跑：
`POST /optimizations` 只做校验 + 落一条任务行 + 起后台线程，立刻返回 id；
前端轮询 `GET /optimizations/{id}` 拿进度。进程池在后台线程里创建，
线程退出即回收（`run_optimization` 的 finally 负责 shutdown）。

同一时刻只允许一个寻优任务：一个任务已经吃掉 CPU-4 个核，再来一个会把
行情同步和巡检挤到无 CPU 可用，回测读数也会因为争抢而失真。
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, defer

from ..backtest.optimizer import (
    MAX_WORKERS,
    OptimizationRequest,
    default_workers,
    grid_size,
    prepare_request,
    run_optimization,
    symbol_universe,
)
from ..backtest.significance import bootstrap_expectancy_ci
from ..config import SUPPORTED_INTERVALS
from ..db import SessionLocal, get_db
from ..models import OptimizationResult, OptimizationRun, StrategyBinding, SymbolPool, utcnow
from ..pagination import page_metadata
from ..schemas import OptimizationApply, OptimizationPayload
from ..serializers import dt_iso
from ..strategy.pools import pool_symbols
from ..strategy.registry import all_specs, has_spec
from ..strategy.symbol_params import delete_overrides, set_overrides

router = APIRouter(prefix="/optimizations", tags=["optimizations"])

_ACTIVE: dict[int, threading.Event] = {}
_LOCK = threading.Lock()
# 扫描范围按 (周期, 天数) 缓存 60 秒：group by + having 在 520 币上要一两秒，
# 而用户调表单时会连续触发
_UNIVERSE_CACHE: dict[tuple[str, int], tuple[float, list[str]]] = {}


def _run_dict(row: OptimizationRun, *, full: bool = False) -> dict:
    payload = {
        "id": row.id,
        "strategy": row.strategy,
        "interval": row.interval,
        "pool": row.pool or "",
        "days": row.days,
        "workers": row.workers,
        "status": row.status,
        "progress": json.loads(row.progress or "{}"),
        "error": row.error or "",
        "created_at": dt_iso(row.created_at),
        "finished_at": dt_iso(row.finished_at),
        "running": row.status in ("pending", "running"),
    }
    if full:
        payload["summary"] = json.loads(row.summary or "{}")
        config = json.loads(row.config or "{}")
        payload["config"] = {
            "request": config.get("request", {}),
            "meta": config.get("meta", {}),
            "symbol_count": len(config.get("symbols") or []),
            "symbols": (config.get("symbols") or [])[:80],
        }
    return payload


def _result_dict(row: OptimizationResult) -> dict:
    metrics = json.loads(row.metrics or "{}")
    return {
        "id": row.id,
        "symbol": row.symbol,
        "params": json.loads(row.params or "{}"),
        "params_before": json.loads(row.params_before or "{}"),
        "score": round(float(row.score or 0.0), 4),
        "verdict": row.verdict or "",
        "delta_source": metrics.get("delta_source", ""),
        "train": metrics.get("train", {}),
        "valid": metrics.get("valid", {}),
        "ci": metrics.get("ci", {}),
        "evaluated": metrics.get("evaluated", 0),
        "eligible": metrics.get("eligible", 0),
        "applied_mode": row.applied_mode or "",
        "applied_at": dt_iso(row.applied_at),
    }


@router.get("/options")
def optimization_options(db: Session = Depends(get_db)):
    """表单需要的元信息：策略、各自搜索空间、并发上限、当前池。"""
    specs = []
    for spec in all_specs():
        specs.append(
            {
                "code": spec.code,
                "name": spec.name,
                "default_enabled": spec.default_enabled,
                "grid": {key: list(values) for key, values in spec.grid.items()},
                "grid_size": grid_size(spec),
            }
        )
    pools = [
        {"code": row.code, "name": row.name, "kind": row.kind}
        for row in db.query(SymbolPool).order_by(SymbolPool.id.asc()).all()
    ]
    return {
        "specs": specs,
        "pools": pools,
        "default_workers": default_workers(),
        "max_workers": MAX_WORKERS,
        "busy": bool(_ACTIVE),
    }


@router.get("/universe")
def universe_preview(
    interval: str = "15m", days: int = 90, db: Session = Depends(get_db)
):
    """预估本次会扫多少币种。

    全量币种表有 500+ 个，但「有足够历史支撑 days 天寻优」的往往只有几十个：
    新上市的币没有 90 天 15m 数据。用户看到「59 个币种」时需要能立刻理解为
    什么不是 520，以及哪些币被排除了。
    """
    if interval not in SUPPORTED_INTERVALS:
        raise HTTPException(400, f"不支持的周期: {interval}")
    cached = _UNIVERSE_CACHE.get((interval, days))
    now = time.time()
    if cached and now - cached[0] < 60:
        symbols = cached[1]
    else:
        symbols = symbol_universe(db, interval, days)
        _UNIVERSE_CACHE[(interval, days)] = (now, symbols)
    return {
        "interval": interval,
        "days": days,
        "count": len(symbols),
        "sample": symbols[:12],
    }

@router.post("")
def create_optimization(payload: OptimizationPayload, db: Session = Depends(get_db)):
    if not has_spec(payload.strategy):
        raise HTTPException(400, f"未知策略: {payload.strategy}")
    with _LOCK:
        if _ACTIVE:
            raise HTTPException(409, "已有寻优任务在运行，请等它结束或先取消")

    if payload.symbols:
        symbols = sorted({str(s).strip().upper() for s in payload.symbols if str(s).strip()})
    elif payload.pool:
        pool = db.query(SymbolPool).filter(SymbolPool.code == payload.pool).first()
        if pool is None:
            raise HTTPException(400, f"未知币种池: {payload.pool}")
        symbols = pool_symbols(db, pool)
    else:
        symbols = symbol_universe(db, payload.interval, payload.days)
    if not symbols:
        raise HTTPException(400, "该周期上没有任何币种有足够历史，先补数据")

    try:
        request, meta, grid = prepare_request(
            db,
            payload.strategy,
            payload.interval,
            days=payload.days,
            train_ratio=payload.train_ratio,
            combos=payload.combos,
            workers=payload.workers,
            min_trades=payload.min_trades,
            objective=payload.objective,
            delta_source=payload.delta_source,
            custom_grid=payload.grid,
            seed=payload.seed,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    workers = request.workers or default_workers()
    config = {
        "request": {
            "strategy": request.strategy,
            "interval": request.interval,
            "days": request.days,
            "train_ratio": request.train_ratio,
            "combos": request.combos,
            "workers": workers,
            "min_trades": request.min_trades,
            "objective": request.objective,
            "delta_source": request.delta_source,
            "custom_grid": payload.grid,
            "seed": request.seed,
            "initial": request.initial,
            "fee": request.fee,
            "slippage": request.slippage,
        },
        "meta": meta,
        "grid": grid,
        "symbols": symbols,
    }
    run = OptimizationRun(
        strategy=request.strategy,
        interval=request.interval,
        pool=payload.pool or "",
        days=request.days,
        windows=0,
        workers=workers,
        status="pending",
        config=json.dumps(config, ensure_ascii=False),
        progress=json.dumps({"total": len(symbols), "done": 0, "elapsed": 0}),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    # 先占住单实例名额再起线程，避免两个请求同时通过 busy 检查（两者都会拉起
    # 一整套进程池，把 CPU 打到 200%）
    cancel = threading.Event()
    with _LOCK:
        _ACTIVE[run.id] = cancel
    try:
        threading.Thread(target=_execute, args=(run.id, cancel), daemon=True).start()
    except Exception:
        with _LOCK:
            _ACTIVE.pop(run.id, None)
        raise
    return {
        "id": run.id,
        "status": "pending",
        "symbol_count": len(symbols),
        "combos": meta["combos"],
        "grid_size": meta["grid_size"],
        "workers": workers,
        "estimated_tasks": len(symbols) * meta["combos"],
    }


def _execute(run_id: int, cancel: threading.Event | None = None) -> None:
    """后台线程：跑完整轮寻优并落库。"""
    db = SessionLocal()
    if cancel is None:
        cancel = threading.Event()
        with _LOCK:
            _ACTIVE[run_id] = cancel
    buffer: list[dict] = []
    valid_returns: list[float] = []
    run: OptimizationRun | None = None
    try:
        run = db.get(OptimizationRun, run_id)
        if run is None:
            return
        config = json.loads(run.config or "{}")
        request = OptimizationRequest(**config["request"])
        meta = config["meta"]
        grid = config["grid"]
        symbols = config["symbols"]

        run.status = "running"
        db.commit()

        def flush() -> None:
            if not buffer:
                return
            for item in buffer:
                metrics = {
                    "train": item.get("train") or {},
                    "valid": item.get("valid") or {},
                    "ci": item.get("ci") or {},
                    "evaluated": item.get("evaluated", 0),
                    "eligible": item.get("eligible", 0),
                    "delta_source": item.get("delta_source", ""),
                    "rows": item.get("rows", 0),
                    "error": item.get("error", ""),
                }
                db.add(
                    OptimizationResult(
                        run_id=run_id,
                        symbol=item["symbol"],
                        params=json.dumps(item.get("best_combo") or {}, ensure_ascii=False),
                        params_before=json.dumps(
                            {
                                key: meta["base_params"].get(key)
                                for key in (meta.get("grid_keys") or [])
                            },
                            ensure_ascii=False,
                        ),
                        metrics=json.dumps(metrics, ensure_ascii=False),
                        score=float(item.get("score") or 0.0),
                        verdict=item.get("verdict") or "",
                    )
                )
            db.commit()
            buffer.clear()

        last_push = [0.0]

        def on_result(item: dict) -> None:
            buffer.append(item)
            valid_returns.extend(item.get("valid_returns") or [])
            if len(buffer) >= 20:
                flush()

        def on_progress(progress: dict) -> None:
            now = time.monotonic()
            if now - last_push[0] < 2.0 and progress["done"] < progress["total"]:
                return
            last_push[0] = now
            run.progress = json.dumps(progress, ensure_ascii=False)
            db.commit()

        summary = run_optimization(
            request=request,
            meta=meta,
            grid=grid,
            symbols=symbols,
            on_progress=on_progress,
            on_result=on_result,
            cancel=cancel,
            workers=request.workers,
        )
        flush()
        pooled = bootstrap_expectancy_ci(
            [{"return_pct": value} for value in valid_returns]
        )
        summary["pooled_ci"] = pooled
        summary["valid_trades_total"] = len(valid_returns)
        run.summary = json.dumps(summary, ensure_ascii=False)
        run.status = "cancelled" if summary.get("cancelled") else "done"
        run.progress = json.dumps(
            {
                "total": summary["total"],
                "done": summary["total"],
                "failed": summary["failed"],
                "skipped": summary["skipped"],
                "elapsed": summary["elapsed"],
                "eta": 0,
                "current": "",
                "verdicts": summary.get("verdicts", {}),
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        db.rollback()
        run = db.get(OptimizationRun, run_id)
        if run is not None:
            run.status = "failed"
            run.error = f"{exc}\n{traceback.format_exc(limit=3)}"[:2000]
    finally:
        try:
            run = db.get(OptimizationRun, run_id)
            if run is not None:
                if run.status == "running":
                    run.status = "failed"
                    run.error = run.error or "任务线程异常退出"
                run.finished_at = utcnow()
                db.commit()
        finally:
            db.close()
            with _LOCK:
                _ACTIVE.pop(run_id, None)


@router.get("")
def list_optimizations(
    db: Session = Depends(get_db),
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=50)] = 20,
):
    query = db.query(OptimizationRun)
    total = query.count()
    rows = (
        query.options(defer(OptimizationRun.config), defer(OptimizationRun.summary))
        .order_by(OptimizationRun.created_at.desc(), OptimizationRun.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return {"optimizations": [_run_dict(row) for row in rows], **page_metadata(total, page, page_size)}


@router.get("/{run_id}")
def get_optimization(run_id: int, db: Session = Depends(get_db)):
    row = db.get(OptimizationRun, run_id)
    if row is None:
        raise HTTPException(404, "寻优任务不存在")
    return _run_dict(row, full=True)


@router.post("/{run_id}/cancel")
def cancel_optimization(run_id: int, db: Session = Depends(get_db)):
    row = db.get(OptimizationRun, run_id)
    if row is None:
        raise HTTPException(404, "寻优任务不存在")
    with _LOCK:
        event = _ACTIVE.get(run_id)
    if event is None:
        raise HTTPException(400, "任务已结束，无法取消")
    event.set()
    return {"ok": True, "message": "已请求取消，等已派发的币种跑完即停"}


@router.delete("/{run_id}")
def delete_optimization(run_id: int, db: Session = Depends(get_db)):
    row = db.get(OptimizationRun, run_id)
    if row is None:
        raise HTTPException(404, "寻优任务不存在")
    with _LOCK:
        if run_id in _ACTIVE:
            raise HTTPException(400, "任务正在运行，先取消再删除")
    db.query(OptimizationResult).filter(OptimizationResult.run_id == run_id).delete(
        synchronize_session=False
    )
    db.delete(row)
    db.commit()
    return {"ok": True}


@router.get("/{run_id}/results")
def list_results(
    run_id: int,
    db: Session = Depends(get_db),
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 50,
    verdict: str = "",
    q: str = "",
    sort: str = "score",
):
    row = db.get(OptimizationRun, run_id)
    if row is None:
        raise HTTPException(404, "寻优任务不存在")
    query = db.query(OptimizationResult).filter(OptimizationResult.run_id == run_id)
    if verdict:
        query = query.filter(OptimizationResult.verdict == verdict)
    if q:
        query = query.filter(OptimizationResult.symbol.like(f"%{q.upper()}%"))
    total = query.count()
    order = (
        OptimizationResult.symbol.asc()
        if sort == "symbol"
        else OptimizationResult.score.desc()
    )
    rows = (
        query.order_by(order)
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return {
        "results": [_result_dict(item) for item in rows],
        **page_metadata(total, page, page_size),
    }


def _selectable(rows: list[OptimizationResult], only: str) -> list[OptimizationResult]:
    """按应用门槛筛结果。

    - pass：验证段每笔期望的 95% CI 下界 > 0（统计上站得住）
    - positive：验证段净收益 > 0 且有成交（不算显著，但至少不是负的）
    - all：全部（含 no_trades，这类没有覆盖参数会被自动跳过）
    """
    if only == "pass":
        return [row for row in rows if row.verdict == "pass"]
    if only == "positive":
        out = []
        for row in rows:
            metrics = json.loads(row.metrics or "{}")
            valid = metrics.get("valid") or {}
            if int(valid.get("total_trades") or 0) > 0 and float(
                valid.get("total_return_pct") or 0.0
            ) > 0:
                out.append(row)
        return out
    return rows


def _pool_code(prefix: str, index: int, used: set[str]) -> str:
    base = (prefix or "opt").strip().lower().replace("-", "_")
    base = "".join(ch for ch in base if ch.isalnum() or ch == "_") or "opt"
    code = f"{base}_{index}"
    suffix = 2
    while code in used:
        code = f"{base}_{index}_{suffix}"
        suffix += 1
    used.add(code)
    return code


@router.post("/{run_id}/apply")
def apply_results(
    run_id: int, payload: OptimizationApply, db: Session = Depends(get_db)
):
    """把寻优结果落到线上。

    mode=symbol：写币种级参数覆盖。绑定不变，巡检下一轮就用新参数评估该币种。
    mode=pool：按「参数完全相同」聚类建池 + 绑定。默认 `enabled=0`——组合是
    按历史最优切出来的，是否上线必须由人决定，不能因为点了一次按钮就自动开跑。
    """
    run = db.get(OptimizationRun, run_id)
    if run is None:
        raise HTTPException(404, "寻优任务不存在")
    with _LOCK:
        if run_id in _ACTIVE:
            raise HTTPException(400, "任务仍在运行，等结束再应用")

    query = db.query(OptimizationResult).filter(OptimizationResult.run_id == run_id)
    if payload.symbols:
        codes = [str(s).strip().upper() for s in payload.symbols]
        query = query.filter(OptimizationResult.symbol.in_(codes))
    rows = query.all()
    selected = _selectable(rows, payload.only)
    items = {
        row.symbol: json.loads(row.params or "{}")
        for row in selected
        if json.loads(row.params or "{}")
    }
    if not items:
        raise HTTPException(
            400,
            "所选条件没有可应用的币种（可能都是「无成交」或未通过筛选；试试把门槛放宽到「全部」）",
        )

    if payload.mode == "symbol":
        config = json.loads(run.config or "{}")
        base = (config.get("meta") or {}).get("base_params") or {}
        try:
            written = set_overrides(
                db,
                run.strategy,
                run.interval,
                items,
                source="optimization",
                run_id=run_id,
                note=f"寻优任务 #{run_id}（{payload.only}）",
                base_params=base,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        _mark_applied(db, run_id, list(items), "symbol")
        return {
            "mode": "symbol",
            "written": written,
            "symbols": sorted(items),
            "message": f"已为 {written} 个币种写入参数覆盖，下一轮巡检即生效",
        }

    # 按参数指纹聚类：同一套参数才值得共用一条绑定
    groups: dict[str, list[str]] = {}
    for symbol, params in items.items():
        groups.setdefault(json.dumps(params, sort_keys=True), []).append(symbol)
    used = {row.code for row in db.query(SymbolPool.code).all()}
    created = []
    for index, (fingerprint, symbols) in enumerate(
        sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])), start=1
    ):
        params = json.loads(fingerprint)
        code = _pool_code(payload.pool_prefix or f"{run.strategy[:6]}opt", index, used)
        pool = SymbolPool(
            code=code,
            name=f"{run.strategy} 寻优组 {index}（{len(symbols)} 币）",
            description=(
                f"由寻优任务 #{run_id} 生成；周期 {run.interval}；"
                f"参数覆盖 {json.dumps(params, ensure_ascii=False)}"
            ),
            kind="static",
            symbols=json.dumps(sorted(symbols), ensure_ascii=False),
            rule="{}",
            enabled=1,
        )
        db.add(pool)
        db.flush()
        binding = StrategyBinding(
            strategy_code=run.strategy,
            pool_code=code,
            interval=run.interval,
            enabled=0,  # 组合需人工确认后再启用
            priority=300 + index,
            params=json.dumps(params, ensure_ascii=False),
        )
        db.add(binding)
        db.flush()
        created.append(
            {
                "pool": code,
                "binding_id": binding.id,
                "symbols": sorted(symbols),
                "params": params,
            }
        )
    db.commit()
    # 这些币种改用新绑定参数，旧的币种级覆盖必须清掉，否则会盖住新组合
    delete_overrides(db, run.strategy, run.interval, list(items))
    _mark_applied(db, run_id, list(items), "pool")
    return {
        "mode": "pool",
        "groups": created,
        "symbols": sorted(items),
        "message": (
            f"已生成 {len(created)} 个池与绑定（默认停用）；"
            "确认参数后到「策略」页启用对应绑定"
        ),
    }


def _mark_applied(db: Session, run_id: int, symbols: list[str], mode: str) -> None:
    now = utcnow()
    (
        db.query(OptimizationResult)
        .filter(
            OptimizationResult.run_id == run_id,
            OptimizationResult.symbol.in_(symbols),
        )
        .update(
            {OptimizationResult.applied_mode: mode, OptimizationResult.applied_at: now},
            synchronize_session=False,
        )
    )
    db.commit()
