import json

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import DatabaseError
from sqlalchemy.orm import Session

from app.db import Base
from app.models import StrategyBinding, StrategyConfig, SymbolParam, SymbolPool
from app.strategy.strategy_migration import (_schema, migrate, migrate_binding_enablement,
                                               rollback, rollback_binding_enablement)


@pytest.fixture
def engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add_all([
            StrategyConfig(code="absorption_reversal", name="策略", params='{"cooldown_bars": 5}', enabled=1),
            SymbolPool(code="pool", name="池", kind="static", symbols='["BTCUSDT"]', rule="{}", manual_override='{"include":["ETHUSDT"],"exclude":[]}'),
            StrategyBinding(strategy_code="absorption_reversal", pool_code="pool", interval="1h", params='{"cooldown_bars": 7}', enabled=1),
            SymbolParam(strategy_code="absorption_reversal", interval="1h", symbol="BTCUSDT", params='{"cooldown_bars": 9}', source="manual"),
        ])
        db.commit()
    yield engine
    engine.dispose()


def test_binding_enablement_migration_is_safe_and_runs_only_once(engine):
    with Session(engine) as db:
        db.add_all([
            StrategyConfig(code="disabled", name="关闭", params="{}", enabled=0),
            StrategyBinding(strategy_code="disabled", pool_code="pool", interval="1h", enabled=1),
            StrategyBinding(strategy_code="missing", pool_code="pool", interval="1h", enabled=1),
            StrategyBinding(strategy_code="absorption_reversal", pool_code="pool", interval="15m", enabled=0),
        ])
        db.commit()
    migrate_binding_enablement(engine)
    with Session(engine) as db:
        rows = {r.strategy_code + r.interval: r.enabled for r in db.query(StrategyBinding).all()}
        assert rows == {"absorption_reversal1h": 1, "disabled1h": 0,
                        "missing1h": 0, "absorption_reversal15m": 0}
        db.query(StrategyBinding).filter_by(strategy_code="disabled").one().enabled = 1
        db.commit()
    migrate_binding_enablement(engine)
    with Session(engine) as db:
        assert db.query(StrategyBinding).filter_by(strategy_code="disabled").one().enabled == 1


def test_binding_enablement_migration_rolls_back_on_failure(engine):
    with Session(engine) as db:
        db.query(StrategyConfig).one().enabled = 0
        db.commit()

    def fail_insert(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO binding_enablement_migrations"):
            raise RuntimeError("模拟迁移失败")

    event.listen(engine, "before_cursor_execute", fail_insert)
    try:
        with pytest.raises(RuntimeError, match="模拟迁移失败"):
            migrate_binding_enablement(engine)
    finally:
        event.remove(engine, "before_cursor_execute", fail_insert)
    with Session(engine) as db:
        assert db.query(StrategyBinding).one().enabled == 1
    migrate_binding_enablement(engine)
    with Session(engine) as db:
        assert db.query(StrategyBinding).one().enabled == 0


def test_migrated_binding_can_be_enabled_without_legacy_switch(engine):
    from app.strategy.bindings import active_bindings
    from app.strategy.defaults import seed_strategies
    from app.strategy.engine import StrategyEngine

    with Session(engine) as db:
        db.query(StrategyConfig).one().enabled = 0
        db.commit()
    migrate_binding_enablement(engine)
    with Session(engine) as db:
        assert active_bindings(db)[0] == []
        row = db.query(StrategyBinding).one()
        assert row.enabled == 0
        row.enabled = 1
        db.commit()
        seed_strategies(db)
        resolved, errors = active_bindings(db)
        assert errors == [] and [item.id for item in resolved] == [row.id]
        evaluator = StrategyEngine(db)
        calls = []
        evaluator._frame = lambda *args: calls.append(args) or None
        evaluator.evaluate("BTCUSDT", "1h", row.strategy_code, params=resolved[0].params, pool="pool")
        assert len(calls) == 1
    assert migrate(engine)["strategies"][0]["enabled"] is True


def test_binding_enablement_rollback_restores_and_blocks_new_runtime(engine):
    from app.strategy.bindings import active_bindings

    with Session(engine) as db:
        db.query(StrategyConfig).one().enabled = 0
        db.commit()
    migrate_binding_enablement(engine)
    rollback_binding_enablement(engine, offline=True)
    rollback_binding_enablement(engine, offline=True)
    with Session(engine) as db:
        assert db.query(StrategyBinding).one().enabled == 1
        active, errors = active_bindings(db)
        assert not active and "已回滚" in errors[0]
    with pytest.raises(ValueError, match="禁止新版服务启动"):
        migrate_binding_enablement(engine)


@pytest.mark.parametrize("change", ["params", "enabled", "delete"])
def test_binding_enablement_rollback_rejects_changed_rows(engine, change):
    with Session(engine) as db:
        db.query(StrategyConfig).one().enabled = 0
        db.commit()
    migrate_binding_enablement(engine)
    with Session(engine) as db:
        row = db.query(StrategyBinding).one()
        if change == "delete":
            db.delete(row)
        elif change == "params":
            row.params = '{"cooldown_bars": 8}'
        else:
            row.enabled = 1
        db.commit()
    with pytest.raises(ValueError, match="已变更或删除"):
        rollback_binding_enablement(engine, offline=True)
    with engine.connect() as conn:
        assert not conn.execute(text("SELECT 1 FROM app_state WHERE key='binding_enablement_rollback'")).first()


def test_binding_rollback_conflict_is_atomic_across_rows(engine):
    with Session(engine) as db:
        db.query(StrategyConfig).one().enabled = 0
        db.add(StrategyBinding(strategy_code="missing", pool_code="pool", interval="1h", enabled=1))
        db.commit()
    migrate_binding_enablement(engine)
    with Session(engine) as db:
        db.query(StrategyBinding).filter_by(strategy_code="missing").one().params = '{"changed": true}'
        db.commit()
    with pytest.raises(ValueError, match="已变更或删除"):
        rollback_binding_enablement(engine, offline=True)
    with Session(engine) as db:
        assert [row.enabled for row in db.query(StrategyBinding).order_by(StrategyBinding.id)] == [0, 0]


@pytest.mark.parametrize("change", ["new_binding", "other_binding", "config", "new_config"])
def test_rollback_rejects_any_binding_or_config_drift(engine, change):
    with Session(engine) as db:
        db.query(StrategyConfig).one().enabled = 0
        db.add(StrategyBinding(strategy_code="absorption_reversal", pool_code="pool", interval="15m", enabled=0))
        db.commit()
    migrate_binding_enablement(engine)
    with Session(engine) as db:
        if change == "new_binding":
            db.add(StrategyBinding(strategy_code="absorption_reversal", pool_code="pool", interval="5m", enabled=1))
        elif change == "other_binding":
            db.query(StrategyBinding).filter_by(interval="15m").one().enabled = 1
        elif change == "config":
            db.query(StrategyConfig).one().enabled = 1
        else:
            db.add(StrategyConfig(code="new", name="new", enabled=1, params="{}"))
        db.commit()
    with pytest.raises(ValueError, match="已变更或删除"):
        rollback_binding_enablement(engine, offline=True)
    with Session(engine) as db:
        assert db.query(StrategyBinding).filter_by(interval="1h").one().enabled == 0


def test_rollback_requires_explicit_offline_declaration(engine):
    migrate_binding_enablement(engine)
    with pytest.raises(ValueError, match="禁止在线回滚"):
        rollback_binding_enablement(engine)


def test_old_partial_backup_cannot_be_rolled_back(engine):
    import hashlib

    migrate_binding_enablement(engine)
    payload = "[]"
    with engine.begin() as conn:
        conn.execute(text("UPDATE binding_enablement_backups SET payload=:payload, digest=:digest"),
                     {"payload": payload, "digest": hashlib.sha256(payload.encode()).hexdigest()})
    with pytest.raises(ValueError, match="缺少完整绑定和策略配置"):
        rollback_binding_enablement(engine, offline=True)


def test_legacy_migration_without_backup_cannot_be_rolled_back(engine):
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE binding_enablement_migrations (id VARCHAR(64) PRIMARY KEY)")
        conn.exec_driver_sql("INSERT INTO binding_enablement_migrations VALUES ('legacy_strategy_enablement_v1')")
    migrate_binding_enablement(engine)
    with pytest.raises(ValueError, match="快照不存在"):
        rollback_binding_enablement(engine, offline=True)


def test_mapping_snapshot_idempotency_and_rollback(engine):
    result = migrate(engine)
    assert result == migrate(engine)
    strategy = result["strategies"][0]
    assert strategy["params"]["cooldown_bars"] == 7
    assert strategy["symbol_params"]["BTCUSDT"]["cooldown_bars"] == 9
    assert json.loads(strategy["pool"]["manual_override"])["include"] == ["ETHUSDT"]
    with engine.begin() as conn:
        payload = conn.execute(text("SELECT payload FROM strategy_migration_snapshots")).scalar_one()
        assert json.loads(payload)["strategy_configs"][0]["params"] == '{"cooldown_bars": 5}'
        assert conn.execute(text("SELECT count(*) FROM strategy_migration_versions")).scalar_one() == 1
        with pytest.raises(DatabaseError):
            conn.execute(text("UPDATE strategy_migration_snapshots SET payload='changed'"))
    rollback(engine, result["version_id"])
    rollback(engine, result["version_id"])
    with pytest.raises(ValueError, match="已回滚"):
        migrate(engine)
    with Session(engine) as db:
        assert db.query(StrategyBinding).count() == 1


def test_bad_override_aborts_everything(engine):
    with Session(engine) as db:
        db.query(SymbolParam).one().params = '{"fee_rate": 0.1}'
        db.commit()
    with pytest.raises(ValueError, match="成本"):
        migrate(engine)
    with engine.connect() as conn:
        assert not conn.execute(text("SELECT name FROM sqlite_master WHERE name='strategy_migration_snapshots'")).first()


def test_drift_blocks_rollback(engine):
    result = migrate(engine)
    with Session(engine) as db:
        db.query(SymbolPool).one().name = "已修改"
        db.commit()
    with pytest.raises(ValueError, match="来源数据已变更"):
        rollback(engine, result["version_id"])
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM strategy_migration_rollbacks")).scalar_one() == 0


def test_overlapping_bindings_and_dynamic_pool(engine):
    with Session(engine) as db:
        db.add(SymbolPool(code="dynamic", name="动态", kind="rule", symbols="[]",
                          rule='{"base":"all","max_symbols":10}', enabled=0))
        db.add(StrategyBinding(strategy_code="absorption_reversal", pool_code="dynamic",
                               interval="1h", params='{"cooldown_bars": 3}', enabled=1))
        db.commit()
    strategies = migrate(engine)["strategies"]
    assert len(strategies) == 2
    assert strategies[1]["enabled"] is False
    assert strategies[1]["symbol_params"]["BTCUSDT"]["cooldown_bars"] == 9
    assert json.loads(strategies[1]["pool"]["rule"])["base"] == "all"


def test_existing_version_survives_failed_migration(engine):
    original = migrate(engine)
    with Session(engine) as db:
        db.query(StrategyBinding).one().params = '{"unknown_key": 1}'
        db.commit()
    with pytest.raises(ValueError, match="未知策略参数"):
        migrate(engine)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT id FROM strategy_migration_versions")).scalar_one() == original["version_id"]


def test_invalid_json_and_missing_reference_abort(engine):
    with Session(engine) as db:
        db.query(StrategyBinding).one().params = "[1]"
        db.commit()
    with pytest.raises(ValueError, match="必须是对象"):
        migrate(engine)
    with Session(engine) as db:
        db.query(StrategyBinding).one().params = "{}"
        db.query(StrategyBinding).one().pool_code = "missing"
        db.commit()
    with pytest.raises(ValueError, match="引用不存在"):
        migrate(engine)


def test_corrupted_version_snapshot_reference_is_rejected(engine):
    result = migrate(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TRIGGER strategy_migration_versions_no_update")
        conn.execute(text("UPDATE strategy_migration_versions SET snapshot_id=:id"), {"id": "corrupt"})
    with pytest.raises(ValueError, match="损坏"):
        migrate(engine)
    with pytest.raises(ValueError, match="损坏"):
        rollback(engine, result["version_id"])


def test_postgresql_schema_guards_truncate_without_database():
    class Connection:
        class dialect:
            name = "postgresql"

        def __init__(self):
            self.statements = []

        def exec_driver_sql(self, statement):
            self.statements.append(statement)

    conn = Connection()
    _schema(conn)
    for table in ("strategy_migration_snapshots", "strategy_migration_versions", "strategy_migration_rollbacks"):
        assert any(f"BEFORE TRUNCATE ON {table} FOR EACH STATEMENT" in sql for sql in conn.statements)
        assert any(f"BEFORE UPDATE OR DELETE ON {table} FOR EACH ROW" in sql for sql in conn.statements)
