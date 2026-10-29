from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

from .config import (
    DATABASE_URL,
    DB_CONNECT_TIMEOUT,
    DB_MAX_OVERFLOW,
    DB_POOL_SIZE,
)

IS_SQLITE = DATABASE_URL.startswith("sqlite")


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
    用完即关。默认连接池（5+10）在 20 个进程下会瞬间打满 PostgreSQL 的
    max_connections，而 worker 的生命周期里只做顺序查询，池化收益为零。
    """
    target = url or DATABASE_URL
    is_local_sqlite = target.startswith("sqlite")

    kwargs: dict = {"future": True, "echo": echo}

    if is_local_sqlite:
        if read_only and "mode=ro" not in target:
            # sqlite:///C:/path/x.db -> sqlite:///file:C:/path/x.db?mode=ro&uri=true
            path_part = target[len("sqlite:///"):]
            target = f"sqlite:///file:{path_part}?mode=ro&uri=true"
        kwargs["connect_args"] = {"check_same_thread": False}
        if read_only:
            kwargs["connect_args"]["timeout"] = 60
    else:
        # PostgreSQL：只在这里挂排查型参数，SQLite 不接受这些
        kwargs.update(
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={"connect_timeout": DB_CONNECT_TIMEOUT},
        )
        if null_pool:
            kwargs["poolclass"] = NullPool
        else:
            kwargs["pool_size"] = pool_size if pool_size is not None else DB_POOL_SIZE
            kwargs["max_overflow"] = (
                max_overflow if max_overflow is not None else DB_MAX_OVERFLOW
            )

    engine = create_engine(target, **kwargs)

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


def init_db():
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


# 轻量补列表：create_all 只建缺失的表，不会给既有表加列。
# 这里只做「ADD COLUMN」（幂等、不重建表、不动数据），够用且对两种方言都安全。
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    # 引入币种池后，信号与订单需要记录自己用的是哪套池参数
    "signals": {"pool": "VARCHAR(48) DEFAULT ''"},
    "orders": {"pool": "VARCHAR(48) DEFAULT ''"},
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
