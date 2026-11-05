"""独立的策略版本迁移；调用方显式提供 Engine，不读取默认数据库连接。

版本快照迁移只追加记录；绑定启停迁移另行保存变更前快照并原子更新绑定。
执行期间锁定来源表；数据库所有者仍可移除触发器，记录安全依赖数据库权限与外部备份。
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from ..config import SUPPORTED_INTERVALS
from ..models import AppState, StrategyBinding, StrategyConfig, SymbolParam, SymbolPool
from ..state import all_state
from .bindings import binding_params
from .rules import RULE_DEFAULTS, SHARED_KEYS, resolve_params

SOURCE_MODELS = (StrategyBinding, StrategyConfig, SymbolPool, SymbolParam)
STATE_KEYS = set(SHARED_KEYS) | set(RULE_DEFAULTS) | {"symbol_scope", "symbols", "symbol_blacklist"}


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False, default=lambda v: v.isoformat() if isinstance(v, (date, datetime)) else str(v))


def _hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _row(row):
    return {column.name: getattr(row, column.name) for column in row.__table__.columns}


def _object(raw, label):
    try:
        result = json.loads(raw or "{}")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} JSON 无效") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{label} 必须是对象")
    return result


def _capture(db):
    rows = {model.__tablename__: [_row(r) for r in db.query(model).order_by(model.id).all()]
            for model in SOURCE_MODELS}
    rows["app_state"] = [_row(r) for r in db.query(AppState).filter(AppState.key.in_(STATE_KEYS)).order_by(AppState.key).all()]
    return json.loads(_json(rows))


def _plan(db, raw):
    configs = {r.code: r for r in db.query(StrategyConfig).all()}
    pools = {r.code: r for r in db.query(SymbolPool).all()}
    overrides = db.query(SymbolParam).order_by(SymbolParam.id).all()
    state = all_state(db)
    conn = db.connection()
    migrated = inspect(conn).has_table("binding_enablement_migrations") and bool(
        conn.execute(text("SELECT 1 FROM binding_enablement_migrations WHERE id=:id"),
                     {"id": "legacy_strategy_enablement_v1"}).first()
    )
    plans = []
    for pool in pools.values():
        if pool.kind not in ("static", "rule"):
            raise ValueError(f"池 {pool.code} 类型无效")
        try:
            symbols = json.loads(pool.symbols or "[]")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"池 {pool.code} symbols 无效") from exc
        if not isinstance(symbols, list):
            raise ValueError(f"池 {pool.code} symbols 必须是数组")
        _object(pool.rule, f"池 {pool.code} rule")
        _object(pool.manual_override, f"池 {pool.code} manual_override")
    for row in db.query(StrategyBinding).order_by(StrategyBinding.id).all():
        config, pool = configs.get(row.strategy_code), pools.get(row.pool_code)
        if config is None or pool is None or row.interval not in SUPPORTED_INTERVALS:
            raise ValueError(f"绑定 {row.id} 引用不存在的策略/池或周期无效")
        _object(config.params, f"策略 {config.code} params")
        _object(row.params, f"绑定 {row.id} params")
        _object(row.exit_params, f"绑定 {row.id} exit_params")
        base = binding_params(db, row, state)
        symbol_values = {}
        for item in overrides:
            if item.strategy_code != row.strategy_code or item.interval != row.interval:
                continue
            layer = _object(item.params, f"币种覆盖 {item.id}")
            if {"fee_rate", "slippage"} & layer.keys():
                raise ValueError(f"币种覆盖 {item.id} 不能覆盖成本")
            merged = {k: v for k, v in {**base, **layer}.items() if k not in ("fee_rate", "slippage")}
            resolved = resolve_params(row.strategy_code, state, merged, interval=row.interval)
            resolved.update({k: base[k] for k in ("fee_rate", "slippage")})
            symbol_values[item.symbol] = resolved
        plans.append({"binding_id": row.id, "strategy_code": row.strategy_code,
                      "pool_code": row.pool_code, "interval": row.interval,
                      "enabled": bool(row.enabled and (migrated or config.enabled) and pool.enabled),
                      "binding_enabled": row.enabled, "strategy_enabled": config.enabled,
                      "pool_enabled": pool.enabled, "priority": row.priority,
                      "params": base, "symbol_params": symbol_values,
                      "pool": next(p for p in raw["symbol_pools"] if p["id"] == pool.id)})
    # 保留未绑定的配置及覆盖，并验证其 JSON；不可静默丢弃坏数据。
    for config in configs.values():
        _object(config.params, f"策略 {config.code} params")
    for item in overrides:
        _object(item.params, f"币种覆盖 {item.id}")
    return plans


def _lock(conn):
    dialect = conn.dialect.name
    if dialect == "sqlite":
        conn.exec_driver_sql("BEGIN IMMEDIATE")
    elif dialect == "postgresql":
        conn.exec_driver_sql("LOCK TABLE app_state, strategy_bindings, strategy_configs, symbol_pools, symbol_params IN SHARE ROW EXCLUSIVE MODE")
    else:
        raise ValueError("仅支持 SQLite 和 PostgreSQL")


def _schema(conn):
    conn.exec_driver_sql("CREATE TABLE IF NOT EXISTS strategy_migration_snapshots (id VARCHAR(64) PRIMARY KEY, payload TEXT NOT NULL)")
    conn.exec_driver_sql("CREATE TABLE IF NOT EXISTS strategy_migration_versions (id VARCHAR(64) PRIMARY KEY, snapshot_id VARCHAR(64) NOT NULL, payload TEXT NOT NULL, FOREIGN KEY(snapshot_id) REFERENCES strategy_migration_snapshots(id))")
    conn.exec_driver_sql("CREATE TABLE IF NOT EXISTS strategy_migration_rollbacks (version_id VARCHAR(64) PRIMARY KEY, FOREIGN KEY(version_id) REFERENCES strategy_migration_versions(id))")
    tables = ("strategy_migration_snapshots", "strategy_migration_versions", "strategy_migration_rollbacks")
    if conn.dialect.name == "sqlite":
        for table in tables:
            for op in ("UPDATE", "DELETE"):
                conn.exec_driver_sql(f"CREATE TRIGGER IF NOT EXISTS {table}_no_{op.lower()} BEFORE {op} ON {table} BEGIN SELECT RAISE(ABORT, 'immutable migration record'); END")
    else:
        conn.exec_driver_sql("CREATE OR REPLACE FUNCTION strategy_migration_immutable() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'immutable migration record'; END; $$")
        for table in tables:
            # DDL is protected by the source-table lock and is transactional.
            conn.exec_driver_sql(f"DROP TRIGGER IF EXISTS {table}_immutable ON {table}")
            conn.exec_driver_sql(f"CREATE TRIGGER {table}_immutable BEFORE UPDATE OR DELETE ON {table} FOR EACH ROW EXECUTE FUNCTION strategy_migration_immutable()")
            conn.exec_driver_sql(f"DROP TRIGGER IF EXISTS {table}_no_truncate ON {table}")
            conn.exec_driver_sql(f"CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {table} FOR EACH STATEMENT EXECUTE FUNCTION strategy_migration_immutable()")


def migrate_binding_enablement(engine):
    """一次性将旧总开关造成的禁用固化到绑定；调用方须在种子数据前执行。"""
    with engine.connect() as conn:
        try:
            _lock(conn)
            conn.exec_driver_sql("CREATE TABLE IF NOT EXISTS binding_enablement_migrations (id VARCHAR(64) PRIMARY KEY)")
            conn.exec_driver_sql("CREATE TABLE IF NOT EXISTS binding_enablement_backups (id VARCHAR(64) PRIMARY KEY, payload TEXT NOT NULL, digest VARCHAR(64) NOT NULL)")
            marker = "legacy_strategy_enablement_v1"
            if conn.execute(text("SELECT 1 FROM app_state WHERE key='binding_enablement_rollback'")).first():
                raise ValueError("绑定启停迁移已回滚，禁止新版服务启动")
            if not conn.execute(text("SELECT 1 FROM binding_enablement_migrations WHERE id=:id"), {"id": marker}).first():
                with Session(bind=conn, join_transaction_mode="rollback_only") as db:
                    affected = (db.query(StrategyBinding)
                                .filter(StrategyBinding.enabled == 1,
                                        ~db.query(StrategyConfig.id).filter(
                                            StrategyConfig.code == StrategyBinding.strategy_code,
                                            StrategyConfig.enabled == 1).exists())
                                .order_by(StrategyBinding.id).all())
                    snapshot = _json({
                        "schema": 2,
                        "bindings": [_row(row) for row in db.query(StrategyBinding).order_by(StrategyBinding.id).all()],
                        "configs": [_row(row) for row in db.query(StrategyConfig).order_by(StrategyConfig.id).all()],
                        "affected_ids": [row.id for row in affected],
                    })
                conn.execute(text("INSERT INTO binding_enablement_backups (id,payload,digest) VALUES (:id,:payload,:digest)"),
                             {"id": marker, "payload": snapshot, "digest": _hash(snapshot)})
                conn.execute(text("""
                    UPDATE strategy_bindings SET enabled=0
                    WHERE enabled=1 AND NOT EXISTS (
                        SELECT 1 FROM strategy_configs
                        WHERE strategy_configs.code=strategy_bindings.strategy_code
                          AND strategy_configs.enabled=1
                    )
                """))
                conn.execute(text("INSERT INTO binding_enablement_migrations (id) VALUES (:id)"), {"id": marker})
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def rollback_binding_enablement(engine, *, offline=False):
    """仅离线回滚：调用方必须先停止全部服务、巡检及开仓进程。

    offline=True 是调用方对离线状态的明确声明，并非跨进程锁或状态验证。
    """
    if offline is not True:
        raise ValueError("禁止在线回滚：请先停止所有服务、巡检及开仓进程，再明确声明 offline=True")
    marker = "legacy_strategy_enablement_v1"
    with engine.connect() as conn:
        try:
            _lock(conn)
            if not inspect(conn).has_table("binding_enablement_backups"):
                raise ValueError("迁移前快照不存在，无法安全回滚")
            backup = conn.execute(text("SELECT payload,digest FROM binding_enablement_backups WHERE id=:id"), {"id": marker}).first()
            if backup is None or _hash(backup.payload) != backup.digest:
                raise ValueError("迁移前快照不存在或已损坏，无法安全回滚")
            if not conn.execute(text("SELECT 1 FROM binding_enablement_migrations WHERE id=:id"), {"id": marker}).first():
                raise ValueError("迁移标记不存在，无法安全回滚")
            if conn.execute(text("SELECT 1 FROM app_state WHERE key='binding_enablement_rollback'")).first():
                conn.commit()
                return
            snapshot = json.loads(backup.payload)
            if not isinstance(snapshot, dict) or snapshot.get("schema") != 2:
                raise ValueError("迁移快照缺少完整绑定和策略配置，无法安全回滚")
            affected_ids = set(snapshot["affected_ids"])
            expected = [{**row, "enabled": 0} if row["id"] in affected_ids else row
                        for row in snapshot["bindings"]]
            with Session(bind=conn, join_transaction_mode="rollback_only") as db:
                bindings = json.loads(_json([_row(row) for row in db.query(StrategyBinding).order_by(StrategyBinding.id).all()]))
                configs = json.loads(_json([_row(row) for row in db.query(StrategyConfig).order_by(StrategyConfig.id).all()]))
                if bindings != expected or configs != snapshot["configs"]:
                    raise ValueError("绑定或策略配置已变更或删除，拒绝回滚")
            for original in snapshot["bindings"]:
                if original["id"] in affected_ids:
                    conn.execute(text("UPDATE strategy_bindings SET enabled=:enabled WHERE id=:id"),
                                 {"enabled": original["enabled"], "id": original["id"]})
            conn.execute(text("INSERT INTO app_state (key,value) VALUES ('binding_enablement_rollback','true')"))
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def migrate(engine):
    """原子生成内容寻址的快照和版本；相同输入幂等，已回滚版本不能重建。"""
    with engine.connect() as conn:
        try:
            _lock(conn)
            _schema(conn)
            with Session(bind=conn, join_transaction_mode="rollback_only") as db:
                raw = _capture(db)
                plans = _plan(db, raw)
            snapshot = _json(raw)
            snapshot_id = _hash(snapshot)
            version = _json({"schema_version": 1, "snapshot_id": snapshot_id, "strategies": plans})
            version_id = _hash(version)
            if conn.execute(text("SELECT 1 FROM strategy_migration_rollbacks WHERE version_id=:id"), {"id": version_id}).first():
                raise ValueError("此版本已回滚，禁止重新启用")
            for table, key, payload in (("strategy_migration_snapshots", snapshot_id, snapshot),
                                        ("strategy_migration_versions", version_id, version)):
                existing = conn.execute(text(f"SELECT payload FROM {table} WHERE id=:id"), {"id": key}).scalar_one_or_none()
                if existing is not None and existing != payload:
                    raise ValueError("迁移记录哈希冲突或数据已损坏")
                if existing is not None and table.endswith("versions"):
                    stored_snapshot = conn.execute(text(f"SELECT snapshot_id FROM {table} WHERE id=:id"), {"id": key}).scalar_one()
                    if stored_snapshot != snapshot_id:
                        raise ValueError("迁移记录哈希冲突或数据已损坏")
                if existing is None:
                    if table.endswith("snapshots"):
                        conn.execute(text(f"INSERT INTO {table} (id,payload) VALUES (:id,:payload)"), {"id": key, "payload": payload})
                    else:
                        conn.execute(text(f"INSERT INTO {table} (id,snapshot_id,payload) VALUES (:id,:snapshot,:payload)"), {"id": key, "snapshot": snapshot_id, "payload": payload})
            conn.commit()
            return {"version_id": version_id, "snapshot_id": snapshot_id, "strategies": plans}
        except Exception:
            conn.rollback()
            raise


def rollback(engine, version_id):
    """追加撤销标记；来源未被迁移修改，拒绝来源漂移和损坏的版本。"""
    with engine.connect() as conn:
        try:
            _lock(conn)
            _schema(conn)
            record = conn.execute(text("SELECT snapshot_id,payload FROM strategy_migration_versions WHERE id=:id"), {"id": version_id}).first()
            if record is None or _hash(record.payload) != version_id:
                raise ValueError("版本不存在或已损坏")
            snapshot = conn.execute(text("SELECT payload FROM strategy_migration_snapshots WHERE id=:id"), {"id": record.snapshot_id}).scalar_one_or_none()
            if snapshot is None or _hash(snapshot) != record.snapshot_id or json.loads(record.payload).get("snapshot_id") != record.snapshot_id:
                raise ValueError("快照不存在或已损坏")
            with Session(bind=conn, join_transaction_mode="rollback_only") as db:
                if _hash(_json(_capture(db))) != record.snapshot_id:
                    raise ValueError("来源数据已变更，拒绝回滚")
            if not conn.execute(text("SELECT 1 FROM strategy_migration_rollbacks WHERE version_id=:id"), {"id": version_id}).first():
                conn.execute(text("INSERT INTO strategy_migration_rollbacks (version_id) VALUES (:id)"), {"id": version_id})
            conn.commit()
        except Exception:
            conn.rollback()
            raise
