"""SQLite -> PostgreSQL 全量迁移。

设计要点：
1. 结构由 SQLAlchemy ORM 在 PG 侧 create_all 生成，避免手写 DDL 与模型漂移。
2. 数据走 psycopg2 的 COPY 文本协议批量装载——比逐行 INSERT 快一到两个数量级
   （本机实测 11M 行级别只需数分钟）。
3. 装载前丢掉二级索引、装载后重建：否则每行都要维护 3 个 btree，写入慢数倍。
4. 迁移后必须 setval 所有 id 序列：SQLite 的 rowid 由我们显式写入，
   PG 的 sequence 不会自动前进，漏掉会导致后续 INSERT 主键冲突。
5. 会话强制 UTC：SQLite 里 DateTime(timezone=True) 存的是无偏移字符串，
   由 PG 解释时必须锚定时区，否则整体偏移。

用法：
    python backend/scripts/migrate_sqlite_to_postgres.py
    python backend/scripts/migrate_sqlite_to_postgres.py --truncate      # 覆盖已存在的表
    python backend/scripts/migrate_sqlite_to_postgres.py --tables klines # 只迁移指定表
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import create_engine, inspect, text  # noqa: E402

from app import models  # noqa: F401  导入即注册全部表
from app.config import DB_PATH, DATABASE_URL  # noqa: E402
from app.db import Base  # noqa: E402

CHUNK = 100_000

ESCAPES = {
    "\\": "\\\\",
    "\t": "\\t",
    "\n": "\\n",
    "\r": "\\r",
}


def pg_text(value) -> str:
    """把一个值转成 PG COPY 文本格式的字段。NULL 用 \\N 表示。"""
    if value is None:
        return "\\N"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # NaN / inf 在 PG 里是合法 float8，但 SQLite 库里出现通常意味着脏值
        if math.isnan(value) or math.isinf(value):
            return "\\N"
        return repr(value)
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        text_value = value.decode("utf-8", "replace")
    else:
        text_value = str(value)
    if text_value == "":
        # COPY 文本格式里空串与 NULL 无法区分，这里用零宽不可哑；
        # 业务上 Text 列默认值都是 '{}' / '[]' / ''，空串无语义，按 NULL 处理。
        return "\\N"
    for raw, escaped in ESCAPES.items():
        text_value = text_value.replace(raw, escaped)
    return text_value


def copy_table(sqlite_conn, pg_conn, table: str, columns: list[str], source_rows: int) -> None:
    """用 COPY 将一个表从 SQLite 搬到 PG。"""
    select_sql = "SELECT " + ", ".join(f'"{c}"' for c in columns) + f' FROM "{table}"'
    cursor = sqlite_conn.execute(select_sql)

    buf = io.StringIO()
    written = 0
    started = time.time()

    while True:
        rows = cursor.fetchmany(CHUNK)
        if not rows:
            break
        buf.seek(0)
        buf.truncate(0)
        for row in rows:
            buf.write("\t".join(pg_text(v) for v in row))
            buf.write("\n")
        buf.seek(0)
        with pg_conn.cursor() as pg_cur:
            col_list = ", ".join(f'"{c}"' for c in columns)
            pg_cur.copy_expert(f'COPY "{table}" ({col_list}) FROM STDIN', buf)
        written += len(rows)
        pg_conn.commit()
        rate = written / max(time.time() - started, 1e-6)
        print(f"    {written:,} / {source_rows:,} 行 ({rate:,.0f} 行/秒)", flush=True)

    return None


def drop_secondary_indexes(pg_conn) -> list[tuple[str, str]]:
    """丢掉所有非主键索引/唯一约束，返回 (名称, 重建DDL) 以便装载后还原。

    唯一约束背后也有一张索引，不能直接 DROP INDEX —— PG 会报
    DependentObjectsStillExist，必须改成删约束本身。装载完成后
    pg_indexes.indexdef 里既有 CREATE INDEX 也有 ALTER TABLE ADD CONSTRAINT，
    因此可以直接原样重放。
    """
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            SELECT i.indexname, i.indexdef, c.oid, ix.indisunique
            FROM pg_indexes i
            JOIN pg_class c ON c.relname = i.indexname
            JOIN pg_index ix ON ix.indexrelid = c.oid
            WHERE i.schemaname = 'public' AND NOT ix.indisprimary
            ORDER BY i.indexname
            """
        )
        rows = cur.fetchall()

    dropped: list[tuple[str, str]] = []
    for name, ddl, oid, is_unique in rows:
        with pg_conn.cursor() as cur:
            if is_unique:
                # 唯一约束：查到约束名后删约束（连背后的索引一起删）
                cur.execute("SELECT conname FROM pg_constraint WHERE conindid = %s", (oid,))
                con = cur.fetchone()
                target_sql = None
                if con:
                    cur.execute(
                        "SELECT relname FROM pg_class WHERE oid = "
                        "(SELECT conrelid FROM pg_constraint WHERE conindid = %s)",
                        (oid,),
                    )
                    tbl = cur.fetchone()
                    if tbl:
                        target_sql = f'ALTER TABLE public."{tbl[0]}" DROP CONSTRAINT "{con[0]}"'
                if target_sql:
                    cur.execute(target_sql)
                else:
                    cur.execute(f'DROP INDEX IF EXISTS public."{name}"')
            else:
                cur.execute(f'DROP INDEX IF EXISTS public."{name}"')
        dropped.append((name, ddl))

    pg_conn.commit()
    print(f"  已丢掉 {len(dropped)} 个二级索引（装载后重建）")
    return dropped


def restore_indexes(pg_conn, indexes: list[tuple[str, str]]) -> None:
    if not indexes:
        return
    print(f"  重建 {len(indexes)} 个索引…")
    for i, (name, ddl) in enumerate(indexes, 1):
        started = time.time()
        with pg_conn.cursor() as cur:
            cur.execute(ddl)
        pg_conn.commit()
        print(f"    [{i}/{len(indexes)}] {name} ({time.time() - started:.1f}s)", flush=True)


def fix_sequences(pg_conn, tables: list[str]) -> None:
    """把每张表的 id 序列推进到当前最大值之后。

    不修复的话，迁移完成后第一次 ORM INSERT 会拿到 id=1，与既有数据撞主键。
    没有 id 列的表（如 app_state，主键是 key）直接跳过；
    pg_get_serial_sequence 在列不存在时会抛 UndefinedColumn，故先查元数据。
    """
    for table in tables:
        with pg_conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = %s AND column_name = 'id'",
                (table,),
            )
            if not cur.fetchone():
                continue
            cur.execute("SELECT pg_get_serial_sequence(%s, 'id')", (f"public.{table}",))
            seq = cur.fetchone()
            if not seq or not seq[0]:
                continue
            seq_name = seq[0]
        with pg_conn.cursor() as cur:
            cur.execute(f'SELECT COALESCE(MAX("id"), 0) FROM public."{table}"')
            max_id = cur.fetchone()[0]
            cur.execute("SELECT setval(%s, %s, %s)", (seq_name, max_id + 1, max_id > 0))
        pg_conn.commit()
        print(f"    {table}: sequence -> {max_id + 1}")


def main() -> int:
    parser = argparse.ArgumentParser(description="迁移 SQLite 数据到 PostgreSQL")
    parser.add_argument("--sqlite", type=Path, default=DB_PATH)
    parser.add_argument("--tables", nargs="*", default=None, help="只迁移指定表，默认全部")
    parser.add_argument("--truncate", action="store_true", help="目标表已有数据时清空后重灌")
    parser.add_argument("--skip-index-restore", action="store_true", help="调试用：装载后不重建索引")
    args = parser.parse_args()

    if DATABASE_URL.startswith("sqlite"):
        print("DATABASE_URL 仍指向 SQLite，请先在 .env 里配置 PostgreSQL 后重试。")
        return 2
    if not args.sqlite.exists():
        print(f"源库不存在：{args.sqlite}")
        return 2

    print(f"源库 SQLite: {args.sqlite} ({args.sqlite.stat().st_size / 1024 / 1024:.0f} MB)")
    print(f"目标 PG     : {DATABASE_URL.split('@')[-1]}")

    target_engine = create_engine(DATABASE_URL, future=True)
    inspector = inspect(target_engine)

    sqlite_conn = sqlite3.connect(str(args.sqlite))
    sqlite_conn.text_factory = str

    started_all = time.time()
    try:
        # 1) 建表结构
        print("\n[1/5] 在 PostgreSQL 建表结构…")
        Base.metadata.create_all(bind=target_engine)
        available = inspector.get_table_names()
        print(f"  已就绪表：{len(available)} 个 -> {sorted(available)}")

        # 2) 确定要迁的表
        src_tables = [
            r[0]
            for r in sqlite_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        wanted = [t for t in src_tables if t in available]
        if args.tables:
            wanted = [t for t in wanted if t in set(args.tables)]
        skipped = [t for t in src_tables if t not in available]
        if skipped:
            print(f"  跳过（PG 模型里不存在）：{skipped}")

        raw_pg = target_engine.raw_connection()
        pg_conn = raw_pg.driver_connection  # SQLAlchemy 2.0：.connection 已废弃
        pg_conn.autocommit = False
        with pg_conn.cursor() as cur:
            cur.execute("SET TIME ZONE 'UTC'")  # 见模块注释第 5 点
        pg_conn.commit()

        # 3) 丢索引
        print("\n[2/5] 装载前丢掉二级索引…")
        indexes = drop_secondary_indexes(pg_conn)

        # 4) 逐表 COPY
        print("\n[3/5] 迁移数据…")
        report: list[tuple[str, int, int]] = []
        for table in wanted:
            columns = [c["name"] for c in inspector.get_columns(table)]
            src_cols = [r[1] for r in sqlite_conn.execute(f'PRAGMA table_info("{table}")')]
            missing = [c for c in columns if c not in src_cols]
            if missing:
                print(f"  ! {table}: SQLite 缺少列 {missing}，跳过该表")
                continue
            source_rows = sqlite_conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

            existing = 0
            with pg_conn.cursor() as cur:
                cur.execute(f'SELECT COUNT(*) FROM public."{table}"')
                existing = cur.fetchone()[0]

            if existing and not args.truncate:
                print(f"  - {table}: 目标已有 {existing:,} 行，跳过（用 --truncate 覆盖）")
                report.append((table, source_rows, existing))
                continue
            if existing and args.truncate:
                with pg_conn.cursor() as cur:
                    cur.execute(f'TRUNCATE public."{table}" RESTART IDENTITY CASCADE')
                pg_conn.commit()
                print(f"  - {table}: 已清空")

            print(f"  > {table}: {source_rows:,} 行")
            if source_rows == 0:
                report.append((table, 0, 0))
                continue
            copy_table(sqlite_conn, pg_conn, table, columns, source_rows)
            with pg_conn.cursor() as cur:
                cur.execute(f'SELECT COUNT(*) FROM public."{table}"')
                target_rows = cur.fetchone()[0]
            report.append((table, source_rows, target_rows))

        # 5) 重建索引 + 修序列
        print("\n[4/5] 重建索引…")
        if not args.skip_index_restore:
            restore_indexes(pg_conn, indexes)

        print("\n[5/5] 修正自增序列…")
        fix_sequences(pg_conn, [t for t, _, _ in report])

        pg_conn.commit()
        raw_pg.close()

        # 6) 校验
        print("\n=== 行数核对 ===")
        bad = 0
        for table, src, tgt in report:
            flag = "OK " if src == tgt else "!!!"
            if src != tgt:
                bad += 1
            print(f"  {flag} {table:<20} sqlite={src:>12,}  pg={tgt:>12,}")
        print(f"\n耗时 {time.time() - started_all:.0f}s，不一致表 {bad} 个")
        return 1 if bad else 0
    finally:
        sqlite_conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
