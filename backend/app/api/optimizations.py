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
    MIN_POOLED_TRADES,
    OptimizationRequest,
    default_workers,
    grid_size,
    pick_best_per_symbol,
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
# 扫描范围按 (周期, 天数) 缓存 60 秒：group by 在 500 币上要一两秒，
# 而用户调表单时会连续触发
_UNIVERSE_CACHE: dict[tuple[str, int], tuple[float, list[str]]] = {}


def _run_intervals(row: OptimizationRun) -> list[str]:
    """任务实际扫过的周期列表（旧任务没有该字段时回落到单周期列）。"""
    try:
        config = json.loads(row.config or "{}")
    except ValueError:
        return [row.interval]
    intervals = (config.get("request") or {}).get("intervals") or []
    return [str(item) for item in intervals] or [row.interval]


def reap_stale_runs(db: Session | None = None) -> int:
    """把上次进程遗留的 pending/running 任务标成 failed。

    寻优在后台线程里跑，进程一退出线程就没了，但数据库里那行仍是 running。
    不清理的话：前端一直显示「运行中」、`create` 之外的地方看不出异常，
    用户会以为任务还在跑而反复刷新；更糟的是新任务正常创建（`_ACTIVE` 是
    内存字典，重启后为空），历史列表里却永远挂着一个假运行中的任务。

    只在启动时调用，单进程部署下不会误伤真正在跑的任务。`db` 只给测试注入。
    """
    own = db is None
    session = db or SessionLocal()
    try:
        stale = (
            session.query(OptimizationRun)
            .filter(OptimizationRun.status.in_(("pending", "running")))
            .all()
        )
        for row in stale:
            row.status = "failed"
            row.error = f"{row.error}\n进程重启，任务中断" if row.error else "进程重启，任务中断"
            row.finished_at = utcnow()
        if stale:
            session.commit()
        return len(stale)
    except Exception:
        session.rollback()
        return 0
    finally:
        if own:
            session.close()


def _run_dict(row: OptimizationRun, *, full: bool = False) -> dict:
    payload = {
        "id": row.id,
        "strategy": row.strategy,
        "interval": row.interval,
        "intervals": _run_intervals(row),
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
        raw_symbols = config.get("symbols") or {}
        if isinstance(raw_symbols, dict):
            # 多周期任务的 symbols 是 {周期: [币种]}，摊平给「本次扫了哪些币」用
            flat = sorted({symbol for group in raw_symbols.values() for symbol in group})
            per_interval = {item: len(group) for item, group in raw_symbols.items()}
        else:
            flat = list(raw_symbols)
            per_interval = {}
        payload["config"] = {
            "request": config.get("request", {}),
            "meta": config.get("meta", {}),
            "symbol_count": len(flat),
            "symbols": flat[:80],
            "per_interval": per_interval,
        }
    return payload


def _result_dict(row: OptimizationResult) -> dict:
    metrics = json.loads(row.metrics or "{}")
    return {
        "id": row.id,
        "symbol": row.symbol,
        "interval": row.interval or "",
        "params": json.loads(row.params or "{}"),
        "params_before": json.loads(row.params_before or "{}"),
        "score": round(float(row.score or 0.0), 4),
        "verdict": row.verdict or "",
        "delta_source": metrics.get("delta_source", ""),
        "delta_coverage": metrics.get("delta_coverage", 0.0),
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
    interval: str = "15m",
    days: int = 90,
    intervals: str = "",
    db: Session = Depends(get_db),
):
    """预估本次会扫多少币种（逐周期）。

    全量币种表有 500+ 个，但「有足够历史支撑 days 天寻优」的每个周期都不同：
    4h 有 365 天的币，15m 往往只有几十天。用户看到「1h 有 455 个币、15m 只有 31 个」
    时需要能立刻理解这是数据覆盖的必然结果，而不是筛选写错了。

    `intervals` 传逗号分隔的周期列表可一次查多个；留空则只查 `interval`。
    """
    wanted = [item.strip() for item in intervals.split(",") if item.strip()] or [interval]
    for item in wanted:
        if item not in SUPPORTED_INTERVALS:
            raise HTTPException(400, f"不支持的周期: {item}")

    now = time.time()
    per_interval: dict[str, dict] = {}
    for item in wanted:
        cached = _UNIVERSE_CACHE.get((item, days))
        if cached and now - cached[0] < 60:
            symbols = cached[1]
        else:
            symbols = symbol_universe(db, item, days)
            _UNIVERSE_CACHE[(item, days)] = (now, symbols)
        per_interval[item] = {"count": len(symbols), "sample": symbols[:8]}

    primary = wanted[0]
    return {
        "interval": primary,
        "intervals": wanted,
        "days": days,
        "count": per_interval[primary]["count"],
        "sample": per_interval[primary]["sample"],
        "per_interval": per_interval,
        # 多周期时真正要跑的任务数是各周期之和，不是某一个周期的币数
        "total_tasks": sum(item["count"] for item in per_interval.values()),
    }

@router.post("")
def create_optimization(payload: OptimizationPayload, db: Session = Depends(get_db)):
    if not has_spec(payload.strategy):
        raise HTTPException(400, f"未知策略: {payload.strategy}")
    with _LOCK:
        if _ACTIVE:
            raise HTTPException(409, "已有寻优任务在运行，请等它结束或先取消")

    # 周期去重并保持用户给定顺序；第一个作为任务主周期（列表页展示用）
    intervals = list(dict.fromkeys(payload.intervals or [payload.interval]))
    if not intervals:
        raise HTTPException(400, "至少选择一个周期")
    for item in intervals:
        if item not in SUPPORTED_INTERVALS:
            raise HTTPException(400, f"不支持的周期: {item}")

    if payload.symbols:
        picked = sorted({str(s).strip().upper() for s in payload.symbols if str(s).strip()})
        universe = {item: picked for item in intervals}
    elif payload.pool:
        pool = db.query(SymbolPool).filter(SymbolPool.code == payload.pool).first()
        if pool is None:
            raise HTTPException(400, f"未知币种池: {payload.pool}")
        members = pool_symbols(db, pool)
        universe = {item: members for item in intervals}
    else:
        universe = {item: symbol_universe(db, item, payload.days) for item in intervals}
    universe = {item: syms for item, syms in universe.items() if syms}
    if not universe:
        raise HTTPException(400, "所选周期上没有任何币种有足够历史，先补数据")

    try:
        request, meta, grids = prepare_request(
            db,
            payload.strategy,
            intervals=intervals,
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

    total_tasks = sum(len(syms) for syms in universe.values())
    workers = request.workers or default_workers()
    config = {
        "request": {
            "strategy": request.strategy,
            "interval": request.interval,
            "intervals": request.interval_list,
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
        "grids": grids,
        "symbols": universe,
    }
    run = OptimizationRun(
        strategy=request.strategy,
        interval=request.interval_list[0],
        pool=payload.pool or "",
        days=request.days,
        windows=0,
        workers=workers,
        status="pending",
        config=json.dumps(config, ensure_ascii=False),
        progress=json.dumps({"total": total_tasks, "done": 0, "elapsed": 0}),
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
        "intervals": request.interval_list,
        "symbol_count": total_tasks,
        "per_interval": {item: len(syms) for item, syms in universe.items()},
        "combos": meta["combos"],
        "grid_size": meta["grid_size"],
        "workers": workers,
        "estimated_tasks": sum(
            len(universe[item]) * len(grids[item]) for item in universe
        ),
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
    # 逐周期分别累计：不同周期的成交时长差异极大，混在一起算 CI 会把
    # 「4h 每笔 +0.8% 但只有 12 笔」和「15m 每笔 -0.1% 有 900 笔」平均成一个
    # 既不描述 4h 也不描述 15m 的数
    returns_by_interval: dict[str, list[float]] = {}
    run: OptimizationRun | None = None
    try:
        run = db.get(OptimizationRun, run_id)
        if run is None:
            return
        config = json.loads(run.config or "{}")
        request = OptimizationRequest(**config["request"])
        meta = config["meta"]
        # 旧任务把网格存在 "grid" 单键下
        grids = config.get("grids") or {request.interval: config.get("grid") or []}
        symbols = config["symbols"]
        if isinstance(symbols, list):  # 旧任务结构
            symbols = {request.interval: symbols}

        run.status = "running"
        db.commit()

        def flush() -> None:
            if not buffer:
                return
            for item in buffer:
                iv_meta = (meta.get("per_interval") or {}).get(item.get("interval") or "")
                iv_meta = iv_meta or {}
                base = iv_meta.get("base_params") or meta.get("base_params") or {}
                grid_keys = iv_meta.get("grid_keys") or meta.get("grid_keys") or []
                metrics = {
                    "train": item.get("train") or {},
                    "valid": item.get("valid") or {},
                    "ci": item.get("ci") or {},
                    "evaluated": item.get("evaluated", 0),
                    "eligible": item.get("eligible", 0),
                    "delta_source": item.get("delta_source", ""),
                    "delta_coverage": item.get("delta_coverage", 0.0),
                    "rows": item.get("rows", 0),
                    "train_bars": item.get("train_bars", 0),
                    "valid_bars": item.get("valid_bars", 0),
                    "error": item.get("error", ""),
                }
                db.add(
                    OptimizationResult(
                        run_id=run_id,
                        symbol=item["symbol"],
                        interval=item.get("interval") or "",
                        params=json.dumps(item.get("best_combo") or {}, ensure_ascii=False),
                        params_before=json.dumps(
                            {key: base.get(key) for key in grid_keys},
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
            returns = item.get("valid_returns") or []
            valid_returns.extend(returns)
            if returns and item.get("interval"):
                returns_by_interval.setdefault(item["interval"], []).extend(returns)
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
            grids=grids,
            symbols=symbols,
            on_progress=on_progress,
            on_result=on_result,
            cancel=cancel,
            workers=request.workers,
        )
        flush()
        pooled = bootstrap_expectancy_ci(
            [{"return_pct": value} for value in valid_returns],
            min_trades=MIN_POOLED_TRADES,
        )
        summary["pooled_ci"] = pooled
        summary["valid_trades_total"] = len(valid_returns)
        summary["pooled_ci_by_interval"] = {
            item: {
                **bootstrap_expectancy_ci(
                    [{"return_pct": value} for value in values],
                    min_trades=MIN_POOLED_TRADES,
                ),
                "trades": len(values),
            }
            for item, values in sorted(returns_by_interval.items())
        }
        summary["intervals"] = request.interval_list
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
    interval: str = "",
    best_only: bool = True,
):
    """结果列表。

    `best_only=True`（默认）时每个币只留一行：在它跑过的所有周期里选出最佳周期。
    这是「每个币种的最佳参数配置」的直接答案。要看某币在各周期的分别表现
    （比如确认它是不是只在 4h 上可交易）时传 `best_only=false` 拿全量明细。
    """
    row = db.get(OptimizationRun, run_id)
    if row is None:
        raise HTTPException(404, "寻优任务不存在")

    base = db.query(OptimizationResult).filter(OptimizationResult.run_id == run_id)
    if interval:
        base = base.filter(OptimizationResult.interval == interval)

    if best_only:
        # 先收敛再筛选：如果先用 SQL 过滤 verdict，就会拿不到该币其它周期的
        # 结果做比较，"pass" 的语义会退化成"某个周期碰巧 pass"
        picked = pick_best_per_symbol([_result_dict(item) for item in base.all()])
        items = list(picked.values())
        if verdict:
            items = [item for item in items if item["verdict"] == verdict]
        if q:
            needle = q.upper()
            items = [item for item in items if needle in item["symbol"]]
        if sort == "symbol":
            items.sort(key=lambda item: item["symbol"])
        elif sort == "interval":
            items.sort(key=lambda item: (item["interval"], -item["score"]))
        else:
            items.sort(key=lambda item: -float(item["score"] or 0.0))
        total = len(items)
        start = (page - 1) * page_size
        return {
            "results": items[start : start + page_size],
            "best_only": True,
            **page_metadata(total, page, page_size),
        }

    query = base
    if verdict:
        query = query.filter(OptimizationResult.verdict == verdict)
    if q:
        query = query.filter(OptimizationResult.symbol.like(f"%{q.upper()}%"))
    total = query.count()
    if sort == "symbol":
        order = (OptimizationResult.symbol.asc(), OptimizationResult.interval.asc())
    elif sort == "interval":
        order = (OptimizationResult.interval.asc(), OptimizationResult.score.desc())
    else:
        order = (OptimizationResult.score.desc(),)
    rows = (
        query.order_by(*order)
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return {
        "results": [_result_dict(item) for item in rows],
        "best_only": False,
        **page_metadata(total, page, page_size),
    }


def _selectable(items: list[dict], only: str) -> list[dict]:
    """按应用门槛筛结果（输入是 `_result_dict` 的产物）。

    - pass：验证段每笔期望的 95% CI 下界 > 0（统计上站得住）
    - positive：验证段净收益 > 0 且有成交（不算显著，但至少不是负的）
    - all：全部（含 no_trades，这类没有覆盖参数会被自动跳过）
    """
    if only == "pass":
        return [item for item in items if item.get("verdict") == "pass"]
    if only == "positive":
        return [
            item
            for item in items
            if int((item.get("valid") or {}).get("total_trades") or 0) > 0
            and float((item.get("valid") or {}).get("total_return_pct") or 0.0) > 0
        ]
    return items


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
    selected = _selectable([_result_dict(item) for item in query.all()], payload.only)
    # 每币只落「最佳周期」那一套：一个币同时写 4 个周期的覆盖，等于把
    # 选周期的结论又丢回给巡检去碰运气
    picked = pick_best_per_symbol(selected)
    by_interval: dict[str, dict[str, dict]] = {}
    for symbol, item in picked.items():
        params = item.get("params") or {}
        if not params:
            continue
        by_interval.setdefault(item["interval"] or "", {}).setdefault(symbol, params)
    items = {symbol: params for group in by_interval.values() for symbol, params in group.items()}
    if not items:
        raise HTTPException(
            400,
            "所选条件没有可应用的币种（可能都是「无成交」或未通过筛选；试试把门槛放宽到「全部」）",
        )

    config = json.loads(run.config or "{}")
    per_interval = (config.get("meta") or {}).get("per_interval") or {}

    if payload.mode == "symbol":
        written = 0
        for interval, group in by_interval.items():
            base = (per_interval.get(interval) or {}).get("base_params") or {}
            try:
                written += set_overrides(
                    db,
                    run.strategy,
                    interval,
                    group,
                    source="optimization",
                    run_id=run_id,
                    note=f"寻优任务 #{run_id}（{payload.only}）",
                    base_params=base,
                )
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        _mark_applied(db, run_id, by_interval, "symbol")
        detail = "、".join(f"{iv} {len(g)} 币" for iv, g in sorted(by_interval.items()))
        return {
            "mode": "symbol",
            "written": written,
            "symbols": sorted(items),
            "per_interval": {iv: sorted(g) for iv, g in by_interval.items()},
            "message": f"已写入 {written} 条参数覆盖（{detail}），下一轮巡检即生效",
        }

    # 按 (周期, 参数指纹) 聚类：同一套参数才值得共用一条绑定
    used = {row.code for row in db.query(SymbolPool.code).all()}
    created = []
    index = 0
    for interval, group in sorted(by_interval.items()):
        groups: dict[str, list[str]] = {}
        for symbol, params in group.items():
            groups.setdefault(json.dumps(params, sort_keys=True), []).append(symbol)
        for fingerprint, symbols in sorted(
            groups.items(), key=lambda item: (-len(item[1]), item[0])
        ):
            index += 1
            params = json.loads(fingerprint)
            code = _pool_code(
                payload.pool_prefix or f"{run.strategy[:6]}opt", index, used
            )
            pool = SymbolPool(
                code=code,
                name=f"{run.strategy} 寻优组 {index}（{interval} / {len(symbols)} 币）",
                description=(
                    f"由寻优任务 #{run_id} 生成；周期 {interval}；"
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
                interval=interval,
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
                    "interval": interval,
                    "symbols": sorted(symbols),
                    "params": params,
                }
            )
    db.commit()
    # 这些币种改用新绑定参数，旧覆盖必须清掉，否则会盖住新组合
    for interval, group in by_interval.items():
        delete_overrides(db, run.strategy, interval, list(group))
    _mark_applied(db, run_id, by_interval, "pool")
    return {
        "mode": "pool",
        "groups": created,
        "symbols": sorted(items),
        "per_interval": {iv: sorted(g) for iv, g in by_interval.items()},
        "message": (
            f"已生成 {len(created)} 个池与绑定（默认停用）；"
            "确认参数后到「策略」页启用对应绑定"
        ),
    }


def _mark_applied(
    db: Session, run_id: int, applied: dict[str, list[str]], mode: str
) -> None:
    """给已落地的结果打标记（`{周期: [币种]}`，逐周期精确匹配）。

    不能只按 symbol 更新：同一币在多个周期都有结果行，只写 symbol 会把那些
    被挑掉的周期也标成「已应用」，下次打开页面看起来像全都上线了。
    """
    now = utcnow()
    for interval, symbols in applied.items():
        if not symbols:
            continue
        (
            db.query(OptimizationResult)
            .filter(
                OptimizationResult.run_id == run_id,
                OptimizationResult.interval == interval,
                OptimizationResult.symbol.in_(symbols),
            )
            .update(
                {OptimizationResult.applied_mode: mode, OptimizationResult.applied_at: now},
                synchronize_session=False,
            )
        )
    db.commit()
