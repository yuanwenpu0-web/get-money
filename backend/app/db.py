from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import (
    DATABASE_URL,
    DB_CONNECT_TIMEOUT,
    DB_MAX_OVERFLOW,
    DB_POOL_SIZE,
)

IS_SQLITE = DATABASE_URL.startswith("sqlite")


def make_engine(url: str | None = None, read_only: bool = False, echo: bool = False) -> Engine:
    """按方言构建 Engine。

    SQLite 需要 check_same_thread=False（FastAPI 多线程共享连接）；
    PostgreSQL 则改走连接池与预检参数，两者互不兼容，必须分支处理。

    read_only=True 时：
    - SQLite 用 mode=ro 的 URI（需要传入普通 sqlite:/// path 形式）
    - PostgreSQL 在连接池层设置 default_transaction_read_only
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
            pool_size=DB_POOL_SIZE,
            max_overflow=DB_MAX_OVERFLOW,
            pool_recycle=1800,
            connect_args={"connect_timeout": DB_CONNECT_TIMEOUT},
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
