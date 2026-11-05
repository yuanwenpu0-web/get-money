from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
from typing import Any
from weakref import WeakKeyDictionary

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool, SingletonThreadPool, StaticPool

from .config import (
    DATABASE_URL,
    DB_CONNECT_TIMEOUT,
    DB_MAX_OVERFLOW,
    DB_POOL_SIZE,
)

IS_SQLITE = DATABASE_URL.startswith("sqlite")


@dataclass
class _PoolCounters:
    """一个 Engine 生命周期内的物理连接与借还次数。"""

    _lock: Lock = field(default_factory=Lock, repr=False)
    connections_created: int = 0
    checkouts: int = 0
    checkins: int = 0
    invalidations: int = 0
    connections_closed: int = 0

    def bump(self, name: str) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + 1)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "connections_created": self.connections_created,
                "checkouts": self.checkouts,
                "checkins": self.checkins,
                "invalidations": self.invalidations,
                "connections_closed": self.connections_closed,
            }


@dataclass
class _EngineStats:
    counters: _PoolCounters = field(default_factory=_PoolCounters)
    config: dict[str, Any] = field(default_factory=dict)


# Engine 本身是弱键；事件回调闭包持有 stats，不会反过来阻止 Engine 被回收。
_STATS: WeakKeyDictionary[Engine, _EngineStats] = WeakKeyDictionary()
_STATS_LOCK = Lock()


def _register_engine(eng: Engine, config: dict[str, Any]) -> None:
    """挂连接生命周期事件，用来判断「连接到底有没有被复用」。

    connect/close 计的是 **DBAPI 物理连接**，checkout/checkin 计的是池的借还。
    两者之差就是复用次数：若每次 checkout 都伴随一次 connect，说明调用方在反复
    新建连接（典型成因：误用 NullPool，或在循环里反复 make_engine 建新池）。
    """
    stats = _EngineStats(config=config)
    with _STATS_LOCK:
        _STATS[eng] = stats

    @event.listens_for(eng, "connect")
    def _on_connect(_dbapi_connection, _connection_record):
        stats.counters.bump("connections_created")

    @event.listens_for(eng, "checkout")
    def _on_checkout(_dbapi_connection, _connection_record, _connection_proxy):
        stats.counters.bump("checkouts")

    @event.listens_for(eng, "checkin")
    def _on_checkin(_dbapi_connection, _connection_record):
        stats.counters.bump("checkins")

    @event.listens_for(eng, "invalidate")
    def _on_invalidate(_dbapi_connection, _connection_record, _exception):
        stats.counters.bump("invalidations")

    @event.listens_for(eng, "close")
    def _on_close(_dbapi_connection, _connection_record):
        stats.counters.bump("connections_closed")


def make_engine(
    url: str | None = None,
    read_only: bool = False,
    echo: bool = False,
    *,
    pool_size: int | None = None,
    max_overflow: int | None = None,
    null_pool: bool = False,
) -> Engine:
    """按方言构建 Engine。

    SQLite 需要 check_same_thread=False（FastAPI 多线程共享连接）；
    PostgreSQL 则改走连接池与预检参数，两者互不兼容，必须分支处理。

    read_only=True 时：
    - SQLite 用 mode=ro 的 URI（需要传入普通 sqlite:/// path 形式）
    - PostgreSQL 在连接池层设置 default_transaction_read_only

    `null_pool=True` 给**寻优进程池的 worker** 用：每个 worker 进程一条连接、
    用完即关。默认连接池（pool_size + max_overflow）在 20 个进程下会瞬间打满
    PostgreSQL 的 max_connections，而 worker 的生命周期里只做顺序查询，池化收益为零。

    注意：**不要循环调用本函数**——每次调用都会新建一个独立的池。需要多次访问时
    复用返回的 Engine（后端服务直接用模块级全局 `engine`）。是否真的复用了连接，
    用 `pool_status()` / `probe_pool_reuse()` 判断。
    """
    target = url or DATABASE_URL
    is_local_sqlite = target.startswith("sqlite")

    kwargs: dict = {"future": True, "echo": echo}
    config: dict[str, Any] = {
        "read_only": read_only,
        "null_pool": null_pool,
        "dialect": "sqlite" if is_local_sqlite else "postgresql",
    }

    if is_local_sqlite:
        if read_only and "mode=ro" not in target:
            # sqlite:///C:/path/x.db -> sqlite:///file:C:/path/x.db?mode=ro&uri=true
            path_part = target[len("sqlite:///"):]
            target = f"sqlite:///file:{path_part}?mode=ro&uri=true"
        kwargs["connect_args"] = {"check_same_thread": False}
        if read_only:
            kwargs["connect_args"]["timeout"] = 60
        # SQLAlchemy 2.x 的文件型 SQLite 默认是 QueuePool；显式 null_pool 也要生效，
        # 否则调用方以为关了池、实际仍在池化（或反过来），判断结果会失真。
        if null_pool:
            kwargs["poolclass"] = NullPool
    else:
        # PostgreSQL：只在这里挂排查型参数，SQLite 不接受这些
        kwargs.update(
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={"connect_timeout": DB_CONNECT_TIMEOUT},
        )
        config.update(
            pre_ping=True,
            recycle_seconds=1800,
            connect_timeout=DB_CONNECT_TIMEOUT,
        )
        if null_pool:
            kwargs["poolclass"] = NullPool
        else:
            config["pool_size"] = pool_size if pool_size is not None else DB_POOL_SIZE
            config["max_overflow"] = (
                max_overflow if max_overflow is not None else DB_MAX_OVERFLOW
            )
            kwargs["pool_size"] = config["pool_size"]
            kwargs["max_overflow"] = config["max_overflow"]

    engine = create_engine(target, **kwargs)
    _register_engine(engine, config)

    if is_local_sqlite:
        _attach_sqlite_pragmas(engine)

    if not is_local_sqlite and read_only:
        _enforce_postgres_readonly(engine)

    return engine


def _attach_sqlite_pragmas(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
        finally:
            cursor.close()


def _enforce_postgres_readonly(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_readonly(dbapi_connection, _connection_record):
        # autocommit=False 时 psycopg2 自开事务，这里直接排空结果避免残留
        dbapi_connection.set_session(readonly=True, autocommit=False)


engine = make_engine()

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _pool_number(pool, method: str) -> int | float | None:
    try:
        value = getattr(pool, method)()
    except (AttributeError, NotImplementedError, TypeError):
        return None
    return value if isinstance(value, (int, float)) else None


def _pool_status_text(pool) -> str | None:
    try:
        return str(pool.status())
    except Exception:
        return None


def pool_status(target: Engine | None = None, *, probe: bool = False) -> dict[str, Any]:
    """判断一个 Engine 的连接池状态：是否在复用连接、容量、占用、探活与告警。

    默认判断模块级全局 `engine`（后端服务用的那一个）。`probe=True` 会借一条连接
    执行 `SELECT 1`，PostgreSQL 额外读 `max_connections` 与本库当前连接数，用于判断
    容量余量。异常只回传异常类型名，绝不回传连接串（可能含密码）。

    关键字段：
    - `reuse_mode`: pooled（真连接池）/ single（单连接复用）/ none（每次新建）
    - `counters.connections_created` vs `counters.checkouts`: 物理建连数 vs 借出数
    - `reuse_ratio`: 借出中被复用的比例；接近 0 就是在反复新建连接
    """
    eng = target if target is not None else engine
    pool = eng.pool
    with _STATS_LOCK:
        stats = _STATS.get(eng)
    counters = stats.counters.snapshot() if stats else None
    cfg = dict(stats.config) if stats else {}
    url = eng.url

    if isinstance(pool, NullPool):
        mode, pooled = "none", False
    elif isinstance(pool, (StaticPool, SingletonThreadPool)):
        mode, pooled = "single", True
    elif isinstance(pool, QueuePool):
        mode, pooled = "pooled", True
    else:
        mode, pooled = type(pool).__name__.lower(), None

    size = _pool_number(pool, "size")
    checkedout = _pool_number(pool, "checkedout") or 0
    checkedin = _pool_number(pool, "checkedin")
    # SQLAlchemy 2.1 起 QueuePool.overflow() 在空闲时返回负值（表示距 pool_size
    # 还剩多少空位）。对外统一成「已启用的溢出连接数」，原始值另存 overflow_raw。
    overflow_raw = _pool_number(pool, "overflow")
    overflow = max(0, int(overflow_raw or 0))
    max_overflow = cfg.get("max_overflow")
    if max_overflow is None:
        raw = getattr(pool, "_max_overflow", None)
        max_overflow = raw if isinstance(raw, int) else None
    if mode == "pooled":
        capacity = int(size or 0) + int(max_overflow or 0)
    elif mode == "single":
        capacity = 1
    else:
        capacity = 0
    utilization = round(checkedout / capacity, 4) if capacity else None

    reused: int | None = None
    reuse_ratio: float | None = None
    if counters and counters["checkouts"]:
        reused = max(0, counters["checkouts"] - counters["connections_created"])
        reuse_ratio = round(reused / counters["checkouts"], 4)

    warnings: list[str] = []
    if mode == "none":
        warnings.append(
            "NullPool：每次借连接都会新建物理连接，仅适合回测进程池 worker，不要用于 API / 巡检路径"
        )
    if counters and counters["checkouts"] >= 10 and reuse_ratio is not None and reuse_ratio < 0.8:
        warnings.append(
            f"连接复用率仅 {reuse_ratio:.0%}：疑似反复新建连接，检查是否在循环里 "
            "make_engine 或误用 NullPool"
        )
    if utilization is not None and utilization >= 1:
        warnings.append("连接池已打满，后续请求将阻塞等待 pool_timeout")
    elif utilization is not None and utilization >= 0.8:
        warnings.append(f"连接池利用率 {utilization:.0%} 偏高，考虑调大 pool_size 或减少并发")
    if overflow is not None and overflow > 0:
        warnings.append(f"已启用 {int(overflow)} 条 overflow 连接，超过常驻池大小")

    result: dict[str, Any] = {
        "dialect": eng.dialect.name,
        "driver": url.get_driver_name(),
        "database": url.database,
        "host": url.host,
        "pool_class": type(pool).__name__,
        "reuse_mode": mode,
        "pooled": pooled,
        "read_only": cfg.get("read_only"),
        "pool_size": size,
        "max_overflow": max_overflow,
        "capacity": capacity,
        "pool_timeout": getattr(pool, "_timeout", None),
        "pool_recycle": getattr(pool, "_recycle", None),
        "pre_ping": bool(getattr(pool, "_pre_ping", False)),
        "checkedout": checkedout,
        "checkedin": checkedin,
        "overflow": overflow,
        "overflow_raw": overflow_raw,
        "utilization": utilization,
        "status": _pool_status_text(pool),
        "counters": counters,
        "reused": reused,
        "reuse_ratio": reuse_ratio,
        "instrumented": stats is not None,
        "warnings": warnings,
        "healthy": None,
        "probe": None,
    }

    if probe:
        started = perf_counter()
        try:
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
                if eng.dialect.name == "postgresql":
                    server_max = conn.execute(text("SHOW max_connections")).scalar()
                    used = conn.execute(
                        text(
                            "SELECT count(*) FROM pg_stat_activity "
                            "WHERE datname = current_database()"
                        )
                    ).scalar()
                    if server_max is not None and used is not None:
                        headroom = int(server_max) - int(used)
                        result["server"] = {
                            "max_connections": int(server_max),
                            "database_connections": int(used),
                            "headroom": headroom,
                        }
                        if headroom < capacity:
                            warnings.append(
                                "PG 剩余可用连接数小于本引擎池容量，多进程并发可能被拒绝连接"
                            )
            result["probe"] = {"ok": True, "error": None}
            result["healthy"] = True
        except Exception as exc:
            # 不回传 str(exc)：psycopg2 的报错里可能带主机/用户名，连接串更危险
            result["probe"] = {"ok": False, "error": type(exc).__name__}
            result["healthy"] = False
            warnings.append(f"探活失败：{type(exc).__name__}")
        result["probe"]["ms"] = round((perf_counter() - started) * 1000, 2)

    return result


def probe_pool_reuse(
    target: Engine | None = None, *, times: int = 20, workers: int = 1
) -> dict[str, Any]:
    """跑 `times` 次 `SELECT 1`，用连接事件计数判断是否真的复用了连接。

    判定标准：借出次数 == 查询次数，且新建物理连接数 <= 并发度（首次预热后
    不应再建新连接）。这直接回答「有没有在频繁新建连接」。
    """
    eng = target if target is not None else engine
    times = max(1, int(times))
    workers = max(1, min(int(workers), times))
    with _STATS_LOCK:
        stats = _STATS.get(eng)
    zero = {"connections_created": 0, "checkouts": 0}
    before = stats.counters.snapshot() if stats else dict(zero)

    def _one(_index: int) -> None:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))

    started = perf_counter()
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(_one, range(times)))
    else:
        for index in range(times):
            _one(index)
    elapsed_ms = round((perf_counter() - started) * 1000, 2)

    after = stats.counters.snapshot() if stats else dict(zero)
    created = after["connections_created"] - before["connections_created"]
    checkouts = after["checkouts"] - before["checkouts"]
    expected = min(workers, times)
    ok = bool(stats) and checkouts == times and created <= expected
    return {
        "queries": times,
        "workers": workers,
        "checkouts": checkouts,
        "connections_created": created,
        "expected_new_connections_max": expected,
        "reuse_ratio": round(max(0, checkouts - created) / checkouts, 4) if checkouts else None,
        "elapsed_ms": elapsed_ms,
        "pool_class": type(eng.pool).__name__,
        "instrumented": stats is not None,
        "reuse_ok": ok,
        "verdict": "连接复用正常" if ok else "疑似频繁新建连接或该 Engine 未挂计数",
    }


def init_db():
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _add_missing_columns()
    _ensure_optimization_result_interval()
    from .strategy.strategy_migration import migrate_binding_enablement

    migrate_binding_enablement(engine)


# 轻量补列表：create_all 只建缺失的表，不会给既有表加列。
# 这里只做「ADD COLUMN」（幂等、不重建表、不动数据），够用且对两种方言都安全。
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "klines": {"taker_buy_volume": "FLOAT", "taker_buy_quote_volume": "FLOAT"},
    # 引入币种池后，信号与订单需要记录自己用的是哪套池参数
    "signals": {"pool": "VARCHAR(48) DEFAULT ''"},
    "orders": {"pool": "VARCHAR(48) DEFAULT ''", "binding_id": "INTEGER", "params_snapshot": "TEXT"},
    # 币种池的人工增删层：独立于 symbols，自动维护不会覆盖它
    "symbol_pools": {"manual_override": "TEXT DEFAULT '{}'"},
    # 绑定层的出场覆盖：止损止盈方法与支撑线指标（见 strategy/exits.py）
    "strategy_bindings": {
        "exit_method": "VARCHAR(32) DEFAULT 'strategy'",
        "support_indicator": "VARCHAR(48) DEFAULT ''",
        "exit_params": "TEXT DEFAULT '{}'",
    },
}


def _add_missing_columns() -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table, columns in _ADDED_COLUMNS.items():
            if table not in tables:
                continue
            existing = {column["name"] for column in inspector.get_columns(table)}
            for name, ddl in columns.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


def _ensure_optimization_result_interval() -> None:
    """把 `optimization_results` 的唯一键从 (run_id, symbol) 扩到含 interval。

    周期变成搜索维度后，同一币种一次任务会有多行结果。SQLite 不支持改约束，
    只能重建表——该库是本地回退库，寻优结果可随时重跑，因此直接重建空表；
    PostgreSQL 走标准 ADD COLUMN + 更换约束，不动已有数据。
    """
    inspector = inspect(engine)
    if "optimization_results" not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns("optimization_results")}
    if "interval" in columns:
        return

    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE optimization_results "
                    "ADD COLUMN IF NOT EXISTS interval VARCHAR(8) NOT NULL DEFAULT ''"
                )
            )
            conn.execute(
                text(
                    "ALTER TABLE optimization_results "
                    "DROP CONSTRAINT IF EXISTS uq_optimization_result"
                )
            )
            conn.execute(
                text(
                    "ALTER TABLE optimization_results ADD CONSTRAINT "
                    "uq_optimization_result UNIQUE (run_id, symbol, interval)"
                )
            )
        return

    # SQLite：丢表让 create_all 按新定义重建
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE optimization_results"))
    from . import models  # noqa: F401  确保映射已注册

    Base.metadata.create_all(bind=engine, tables=[models.OptimizationResult.__table__])
