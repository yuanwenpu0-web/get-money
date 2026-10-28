"""「策略 × 币种池」绑定：池解析、三层参数、巡检按绑定执行、API 契约。"""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.models import StrategyBinding, SymbolPool
from app.state import set_state
from app.strategy import bindings, pools
from app.strategy.defaults import DEFAULT_BINDINGS, DEFAULT_POOLS, seed_defaults
from app.strategy.registry import StrategySpec, get_spec, register
from app.strategy.rules import resolve_params

STRATEGY = "absorption_reversal"


def _pool(db, code="test", symbols=("AAAUSDT", "BBBUSDT"), kind="static", rule=None):
    row = SymbolPool(
        code=code, name=code, kind=kind,
        symbols=json.dumps(list(symbols)),
        rule=json.dumps(rule or {}),
        enabled=1,
    )
    db.add(row)
    db.commit()
    return row


def _binding(db, pool="test", interval="15m", params=None, enabled=1, strategy=STRATEGY, priority=10):
    row = StrategyBinding(
        strategy_code=strategy, pool_code=pool, interval=interval,
        enabled=enabled, priority=priority,
        params=json.dumps(params or {}, ensure_ascii=False),
    )
    db.add(row)
    db.commit()
    return row


def _enable_strategy(db, params=None):
    from app.models import StrategyConfig

    row = db.query(StrategyConfig).filter(StrategyConfig.code == STRATEGY).first()
    if row is None:
        row = StrategyConfig(code=STRATEGY, name=STRATEGY, params=json.dumps(params or {}), enabled=1)
        db.add(row)
    else:
        row.enabled = 1
    db.commit()
    return row


# ---------------------------------------------------------------- 池

def test_static_pool_members_are_normalized(db):
    pool = _pool(db, symbols=("aaausdt", "BBBUSDT", "aaausdt"))
    assert pools.pool_symbols(db, pool) == ["AAAUSDT", "BBBUSDT"]


def test_rule_pool_intersects_custom_scope(db):
    pool = _pool(db, kind="rule", symbols=(), rule={"base": "major"})
    members = pools.pool_symbols(db, pool, allowed=["BTCUSDT", "FOOUSDT"])
    assert members == ["BTCUSDT"]


def test_pool_validation_rejects_unknown_rule_keys():
    with pytest.raises(ValueError, match="未知池规则字段"):
        pools.validate_pool("rule", [], {"base": "all", "bogus": 1})
    with pytest.raises(ValueError, match="static 池必须给出 symbols"):
        pools.validate_pool("static", [], {})
    with pytest.raises(ValueError, match="base"):
        pools.validate_pool("rule", [], {"base": "everything"})


def test_pool_top_n_sorts_by_volume_when_available():
    members = pools.resolve_members(
        "rule", [], {"base": "all", "max_symbols": 2},
        universe=lambda: ["AUSDT", "BUSDT", "CUSDT"],
        volumes=lambda: {"AUSDT": 1.0, "BUSDT": 9.0, "CUSDT": 5.0},
    )
    assert members == ["BUSDT", "CUSDT"]


# ---------------------------------------------------------------- 三层参数

def test_binding_params_override_strategy_defaults(db):
    _enable_strategy(db)
    _pool(db)
    row = _binding(db, params={"target_r": 3.0, "max_hold_bars": 12})

    defaults = resolve_params(STRATEGY, {}, None)
    assert defaults["target_r"] == 2.0 and defaults["max_hold_bars"] == 30

    effective = bindings.binding_params(db, row)
    assert effective["target_r"] == 3.0
    assert effective["max_hold_bars"] == 12
    # 未覆盖的字段仍继承策略默认
    assert effective["delta_pct"] == defaults["delta_pct"]


def test_global_state_sits_below_strategy_defaults(db):
    _enable_strategy(db)
    _pool(db)
    row = _binding(db, params={})
    set_state(db, "max_leverage", 7)

    effective = bindings.binding_params(db, row)
    # 策略默认 max_leverage=1 覆盖全局的 7（策略默认层更高）
    assert effective["max_leverage"] == 1
    # 策略没定义的键才由全局驱动
    set_state(db, "min_score", 0.62)
    assert bindings.binding_params(db, row)["min_score"] == 0.62


def test_same_strategy_two_pools_have_independent_params(db):
    _enable_strategy(db)
    _pool(db, code="major", symbols=("BTCUSDT",), kind="rule", rule={"base": "major"})
    _pool(db, code="alt", symbols=("FOOUSDT",))
    _binding(db, pool="major", interval="15m", params={"delta_pct": 95.0, "target_r": 2.0})
    _binding(db, pool="alt", interval="5m", params={"delta_pct": 99.0, "target_r": 1.5}, priority=20)

    resolved, errors = bindings.active_bindings(db)
    assert errors == []
    by_pool = {binding.pool: binding for binding in resolved}
    assert by_pool["major"].interval == "15m"
    assert by_pool["alt"].interval == "5m"
    assert by_pool["major"].params["delta_pct"] == 95.0
    assert by_pool["alt"].params["delta_pct"] == 99.0
    assert by_pool["alt"].params["target_r"] == 1.5
    # major 池按 base=major 解析出全部主流币；alt 池是静态单币
    assert "BTCUSDT" in by_pool["major"].symbols
    assert by_pool["alt"].symbols == ("FOOUSDT",)


def test_bad_binding_does_not_block_others(db):
    _enable_strategy(db)
    _pool(db, code="good", symbols=("AAAUSDT",))
    _binding(db, pool="good")
    db.add(
        StrategyBinding(
            strategy_code=STRATEGY, pool_code="missing", interval="15m",
            enabled=1, priority=1, params="{}",
        )
    )
    db.commit()

    resolved, errors = bindings.active_bindings(db)
    assert [binding.pool for binding in resolved] == ["good"]
    assert any("币种池不存在" in message for message in errors)


def test_disabled_binding_or_strategy_is_skipped(db):
    _enable_strategy(db)
    _pool(db)
    binding = _binding(db, enabled=0)
    assert bindings.active_bindings(db)[0] == []

    binding.enabled = 1
    db.commit()
    from app.models import StrategyConfig

    db.query(StrategyConfig).filter(StrategyConfig.code == STRATEGY).one().enabled = 0
    db.commit()
    assert bindings.active_bindings(db)[0] == []


def test_expand_repeats_symbol_across_pools(db):
    _enable_strategy(db)
    _pool(db, code="p1", symbols=("BTCUSDT",))
    _pool(db, code="p2", symbols=("BTCUSDT",))
    _binding(db, pool="p1", interval="15m")
    _binding(db, pool="p2", interval="1h", priority=20)

    resolved, _ = bindings.active_bindings(db)
    pairs = bindings.expand(resolved)
    assert [(symbol, binding.pool) for symbol, binding in pairs] == [
        ("BTCUSDT", "p1"), ("BTCUSDT", "p2"),
    ]


def test_resolve_order_params_prefers_binding(db):
    _enable_strategy(db, params={"target_r": 2.5})
    _pool(db)
    _binding(db, interval="15m", params={"target_r": 4.0})

    assert bindings.resolve_order_params(db, STRATEGY, "15m", "test")["target_r"] == 4.0
    # 没有 pool 的老订单回落到策略配置
    assert bindings.resolve_order_params(db, STRATEGY, "15m", "")["target_r"] == 2.5


# ---------------------------------------------------------------- 注册表

def test_registry_rejects_duplicate_and_unknown():
    with pytest.raises(ValueError, match="重复注册"):
        register(get_spec(STRATEGY))
    with pytest.raises(ValueError, match="未知策略"):
        get_spec("nope")


def test_spec_requires_signals_and_lookback():
    spec = StrategySpec(
        code="toy", name="toy", description="", defaults={"x": 1}, ranges={"x": (0, 2)},
        lookback=lambda params: 10, prepare=lambda frame, params, ctx: frame,
        signals=lambda frame, params: frame, default_enabled=False,
    )
    assert spec.lookback({}) == 10 and spec.describe is None


# ---------------------------------------------------------------- 巡检按绑定

def test_run_cycle_picks_best_score_across_pools(db, monkeypatch):
    from app.trading import service as trading_service
    from app.trading.service import TradingService

    _enable_strategy(db)
    _pool(db, code="p1", symbols=("BTCUSDT",))
    _pool(db, code="p2", symbols=("BTCUSDT",))
    _binding(db, pool="p1", interval="15m", priority=1)
    _binding(db, pool="p2", interval="1h", priority=2)
    set_state(db, "symbol_scope", "custom")
    set_state(db, "symbols", ["BTCUSDT"])

    scores = {"p1": 0.70, "p2": 0.90}

    def evaluate(self, symbol, interval, code, save=False, *, params=None, pool="", **kwargs):
        return SimpleNamespace(
            symbol=symbol, interval=interval, strategy=code, pool=pool,
            direction="LONG", score=scores[pool], price=100.0, sl=99.0, tp=103.0,
            leverage=1, position_pct=0.3, prob_up=0.6,
            to_dict=lambda: {"symbol": symbol, "pool": pool, "interval": interval},
        )

    monkeypatch.setattr(trading_service, "StrategyEngine", lambda db, **kwargs: type("S", (), {"evaluate": evaluate})())
    monkeypatch.setattr(trading_service, "_sync_recent_many", lambda symbols, interval: [])
    monkeypatch.setattr(trading_service, "open_position", lambda db, signal, **kwargs: None)

    summary = TradingService(db).run_cycle()
    assert summary["ok"] is True
    assert [(b["pool"], b["interval"]) for b in summary["bindings"]] == [("p1", "15m"), ("p2", "1h")]
    assert len(summary["signals"]) == 1
    assert summary["signals"][0]["pool"] == "p2"  # 评分更高的池胜出


# ---------------------------------------------------------------- API

@pytest.fixture
def client():
    """独立的内存库 + StaticPool：请求在别的线程执行，必须共用同一个连接。"""
    from fastapi import FastAPI
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from sqlalchemy.pool import StaticPool

    from app.api import bindings as bindings_api
    from app.api import pools as pools_api
    from app.db import Base, get_db

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)

    app = FastAPI()
    app.include_router(pools_api.router, prefix="/api")
    app.include_router(bindings_api.router, prefix="/api")
    app.dependency_overrides[get_db] = lambda: session
    try:
        yield TestClient(app), session
    finally:
        session.close()
        engine.dispose()


def test_pool_api_roundtrip(client):
    client, db = client
    created = client.post("/api/pools", json={
        "code": "majors", "name": "主流", "kind": "rule", "rule": {"base": "major"},
    })
    assert created.status_code == 200
    assert created.json()["member_count"] == 16

    members = client.get("/api/pools/majors/members").json()
    assert members["count"] == 16 and "BTCUSDT" in members["symbols"]

    updated = client.put("/api/pools/majors", json={"symbols": ["BTCUSDT"], "kind": "static"})
    assert updated.status_code == 200
    assert updated.json()["members_sample"] == ["BTCUSDT"]

    assert client.delete("/api/pools/majors").status_code == 200
    assert client.get("/api/pools/majors/members").status_code == 404


def test_pool_delete_is_blocked_when_referenced(client):
    client, db = client
    _enable_strategy(db)
    client.post("/api/pools", json={"code": "p1", "name": "p1", "kind": "static", "symbols": ["AAAUSDT"]})
    created = client.post("/api/bindings", json={
        "strategy": STRATEGY, "pool": "p1", "interval": "15m", "params": {},
    })
    assert created.status_code == 200
    blocked = client.delete("/api/pools/p1")
    assert blocked.status_code == 400
    assert "绑定" in blocked.json()["detail"]


def test_binding_api_validates_and_lists_effective_params(client):
    client, db = client
    _enable_strategy(db)
    client.post("/api/pools", json={"code": "p1", "name": "p1", "kind": "static", "symbols": ["AAAUSDT"]})

    bad = client.post("/api/bindings", json={
        "strategy": STRATEGY, "pool": "p1", "interval": "15m", "params": {"target_r": 99},
    })
    assert bad.status_code == 400

    unknown_pool = client.post("/api/bindings", json={
        "strategy": STRATEGY, "pool": "nope", "interval": "15m", "params": {},
    })
    assert unknown_pool.status_code == 400

    created = client.post("/api/bindings", json={
        "strategy": STRATEGY, "pool": "p1", "interval": "15m",
        "params": {"target_r": 3.0},
    })
    assert created.status_code == 200
    payload = created.json()
    assert payload["effective_params"]["target_r"] == 3.0
    assert payload["member_count"] == 1
    assert payload["lookback"] >= 1200

    duplicate = client.post("/api/bindings", json={
        "strategy": STRATEGY, "pool": "p1", "interval": "15m", "params": {},
    })
    assert duplicate.status_code == 400

    listing = client.get("/api/bindings").json()
    assert listing["total"] == 1
    assert listing["bindings"][0]["interval"] == "15m"

    moved = client.put(f"/api/bindings/{payload['id']}", json={"interval": "1h"})
    assert moved.status_code == 200 and moved.json()["interval"] == "1h"

    assert client.delete(f"/api/bindings/{payload['id']}").status_code == 200
    assert client.get("/api/bindings").json()["total"] == 0


def test_seed_defaults_creates_pools_and_bindings_once(db):
    seed_defaults(db)
    codes = {row.code for row in db.query(SymbolPool).all()}
    assert codes == {item["code"] for item in DEFAULT_POOLS}
    seeded = db.query(StrategyBinding).all()
    assert {(row.pool_code, row.interval) for row in seeded} == {
        (item["pool_code"], item["interval"]) for item in DEFAULT_BINDINGS
    }

    # 用户删掉默认绑定后，再次 seed 不应复活它
    db.delete(seeded[0])
    seeds_before = db.query(StrategyBinding).count()
    seed_defaults(db)
    assert db.query(StrategyBinding).count() == seeds_before
