"""连接池判断的回归测试：不依赖真实 PostgreSQL，只用本地 SQLite 与假 PG URL。"""
from __future__ import annotations

import json

from app import db
from app.config import DB_MAX_OVERFLOW, DB_POOL_SIZE


def _sqlite_url(tmp_path, name: str = "pool.db") -> str:
    return "sqlite:///" + str(tmp_path / name).replace("\\", "/")


def test_pg_engine_reports_expected_pool_config():
    """PG 分支：容量、超时、回收、预检都要能被判断出来（不实际连库）。"""
    eng = db.make_engine(url="postgresql+psycopg2://u:p@127.0.0.1/probe")
    try:
        status = db.pool_status(eng)
        assert status["pool_class"] == "QueuePool"
        assert status["reuse_mode"] == "pooled" and status["pooled"] is True
        assert status["pool_size"] == DB_POOL_SIZE
        assert status["max_overflow"] == DB_MAX_OVERFLOW
        assert status["capacity"] == DB_POOL_SIZE + DB_MAX_OVERFLOW
        assert status["pool_timeout"] == 30.0
        assert status["pool_recycle"] == 1800
        assert status["pre_ping"] is True
        assert status["healthy"] is None  # 未探活时不该假装健康
        assert status["instrumented"] is True
    finally:
        eng.dispose()


def test_pooled_engine_reuses_one_physical_connection(tmp_path):
    """12 次查询只应新建 1 条物理连接——「避免频繁新建连接」的核心断言。"""
    eng = db.make_engine(url=_sqlite_url(tmp_path))
    try:
        reuse = db.probe_pool_reuse(eng, times=12)
        assert reuse["instrumented"] is True
        assert reuse["checkouts"] == 12
        assert reuse["connections_created"] == 1
        assert reuse["reuse_ok"] is True

        status = db.pool_status(eng, probe=True)
        assert status["healthy"] is True
        assert status["counters"]["checkouts"] >= 12
        assert status["reuse_ratio"] >= 0.9
        assert status["overflow"] == 0
        assert status["warnings"] == []
    finally:
        eng.dispose()


def test_null_pool_is_detected_as_no_reuse(tmp_path):
    """NullPool 每次查询都新建连接，必须被判定出来；SQLite 分支也要生效。"""
    eng = db.make_engine(url=_sqlite_url(tmp_path, "null.db"), null_pool=True)
    try:
        assert type(eng.pool).__name__ == "NullPool"
        status = db.pool_status(eng)
        assert status["reuse_mode"] == "none" and status["pooled"] is False
        assert any("NullPool" in item for item in status["warnings"])

        reuse = db.probe_pool_reuse(eng, times=6)
        assert reuse["checkouts"] == 6
        assert reuse["connections_created"] == 6
        assert reuse["reuse_ok"] is False
        assert db.pool_status(eng)["reuse_ratio"] == 0.0
    finally:
        eng.dispose()


def test_probe_failure_never_leaks_connection_string(tmp_path):
    """探活失败只回异常类型名，不把含凭证的连接串带出去。"""
    eng = db.make_engine(url=_sqlite_url(tmp_path, "missing/x.db"))
    try:
        status = db.pool_status(eng, probe=True)
        assert status["healthy"] is False
        assert status["probe"]["ok"] is False
        assert status["probe"]["error"]
        assert any("探活失败" in item for item in status["warnings"])
        assert "://" not in json.dumps(status, ensure_ascii=False)
    finally:
        eng.dispose()


def test_global_engine_status_has_no_credentials():
    """全局 engine（API 用的那个）可被判断，且输出不含 URL / 密码。"""
    status = db.pool_status()
    assert status["instrumented"] is True
    blob = json.dumps(status, ensure_ascii=False)
    assert "://" not in blob
    assert "password" not in blob
