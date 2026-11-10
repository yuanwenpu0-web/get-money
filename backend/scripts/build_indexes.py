"""补建列表页/统计页需要的索引（幂等，可反复执行）。

`app/db.py::init_db` 只会自动补小表的组合索引；`klines` 有 5000 万行量级，
直接 `CREATE INDEX` 会长时间锁表，必须在服务运行时用 `CONCURRENTLY` 建。

用法：
    .venv/Scripts/python.exe backend/scripts/build_indexes.py
    .venv/Scripts/python.exe backend/scripts/build_indexes.py --check   # 只报告，不建
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.db import engine  # noqa: E402

# (索引名, 表, 列定义) —— 列顺序决定能不能走 Index Only Scan + LIMIT 提前结束
INDEXES: tuple[tuple[str, str, str], ...] = (
    ("ix_klines_interval_symbol_time", "klines", '("interval", symbol, open_time)'),
    ("ix_orders_status_created", "orders", "(status, created_at, id)"),
    ("ix_orders_created", "orders", "(created_at, id)"),
    ("ix_orders_symbol_created", "orders", "(symbol, created_at)"),
    ("ix_job_logs_created", "job_logs", "(created_at, id)"),
    ("ix_backtest_runs_created", "backtest_runs", "(created_at, id)"),
    ("ix_optimization_runs_created", "optimization_runs", "(created_at, id)"),
)


def existing_indexes(conn) -> set[str]:
    if engine.dialect.name == "postgresql":
        rows = conn.execute(text("SELECT indexname FROM pg_indexes")).all()
    else:
        rows = conn.execute(text("SELECT name FROM sqlite_master WHERE type = 'index'")).all()
    return {row[0] for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只列出缺失的索引，不创建")
    args = parser.parse_args()

    with engine.connect() as conn:
        present = existing_indexes(conn)
        tables = {
            row[0]
            for row in conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = current_schema()")
                if engine.dialect.name == "postgresql"
                else text("SELECT name FROM sqlite_master WHERE type = 'table'")
            ).all()
        }

    missing = [item for item in INDEXES if item[0] not in present and item[1] in tables]
    if not missing:
        print("所有索引都已存在。")
        return 0
    print("缺失索引：")
    for name, table, columns in missing:
        print(f"  - {name}  ON {table} {columns}")
    if args.check:
        return 0

    concurrent = engine.dialect.name == "postgresql"
    for name, table, columns in missing:
        ddl = f"CREATE INDEX {'CONCURRENTLY ' if concurrent else ''}IF NOT EXISTS {name} ON {table} {columns}"
        started = time.time()
        # CONCURRENTLY 不能在事务里执行，必须用 AUTOCOMMIT
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(ddl))
        print(f"  ✓ {name}  用时 {time.time() - started:.1f}s")
    print("完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
