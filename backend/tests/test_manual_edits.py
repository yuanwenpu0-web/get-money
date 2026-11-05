"""绑定整体编辑 + 币种池「人工增删层」的契约测试。

两件事都要抗住自动化：
- 池的手工增删独立于 symbols，研究池维护重写 symbols 后仍然生效；
- 绑定改身份字段（策略/池/周期）不丢参数、不换 id，且仍满足唯一三元组。
"""

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api import bindings as bindings_api
from app.api import pools as pools_api
from app.db import Base, get_db
from app.models import StrategyConfig, SymbolPool
from app.strategy import pools, specs  # noqa: F401  specs 导入即注册策略
from app.strategy.rules import known_codes

STRATEGY = "absorption_reversal"
OTHER_STRATEGY = next(code for code in known_codes() if code != STRATEGY)


@pytest.fixture
def client():
    """独立内存库 + StaticPool：TestClient 在别的线程执行请求，必须共用同一连接。"""
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


def _pool(db, code="p1", symbols=("AAAUSDT", "BBBUSDT"), manual=None):
    row = SymbolPool(
        code=code, name=code, kind="static",
        symbols=json.dumps(list(symbols)), rule="{}",
        manual_override=json.dumps(manual or {}, ensure_ascii=False),
        enabled=1,
    )
    db.add(row)
    db.commit()
    return row


def _create_binding(client, *, strategy=STRATEGY, pool="p1", interval="15m", params=None):
    response = client.post("/api/bindings", json={
        "strategy": strategy, "pool": pool, "interval": interval,
        "params": params or {},
    })
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------- 纯解析层

def test_manual_override_normalizes_and_resolves_conflicts():
    manual = pools.normalize_manual_override(
        ["aaausdt", "AAAUSDT", "BBBUSDT"], ["bbbusdt", "CCCUSDT", "CCCUSDT"],
    )
    assert manual == {"include": ["AAAUSDT"], "exclude": ["BBBUSDT", "CCCUSDT"]}
    with pytest.raises(ValueError, match="不能超过"):
        pools.normalize_manual_override([f"S{i}USDT" for i in range(201)], [])


def test_manual_include_survives_filters_and_topn():
    members = pools.resolve_members(
        "rule", [], {"base": "all", "max_symbols": 1,
                     "filters": {"quote_volume_24h": {"min": 10}}},
        universe=lambda: ["AAAUSDT", "BBBUSDT", "CCCUSDT"],
        volumes=lambda: {"AAAUSDT": 100.0, "BBBUSDT": 50.0, "CCCUSDT": 1.0},
        manual={"include": ["CCCUSDT", "DDDUSDT"], "exclude": ["AAAUSDT"]},
    )
    # 规则只留下成交额最高的 AAA；人工层先剔除它，再加入被门槛和 TopN 刷掉的币
    assert members == ["CCCUSDT", "DDDUSDT"]


def test_manual_include_still_obeys_blacklist_and_global_scope(monkeypatch):
    monkeypatch.setattr(pools, "BLACKLIST", {"BADUSDT"})
    members = pools.resolve_members(
        "static", ["AAAUSDT"], {},
        universe=lambda: [], volumes=lambda: {},
        allowed=["AAAUSDT", "GOODUSDT"],
        manual={"include": ["GOODUSDT", "BADUSDT", "OUTUSDT"], "exclude": []},
    )
    assert members == ["AAAUSDT", "GOODUSDT"]


def test_manual_exclude_wins_over_include_even_after_rewrite(db):
    """同一币两侧都写时剔除优先；且自动维护重写 symbols 不会冲掉人工层。"""
    include = ["CCCUSDT", "AAAUSDT"]
    manual = pools.normalize_manual_override(include, ["AAAUSDT"])
    assert manual["include"] == ["CCCUSDT"]

    pool = _pool(db, manual=manual)
    assert pools.pool_symbols(db, pool) == ["BBBUSDT", "CCCUSDT"]

    # 模拟 pool_maintenance.refresh_research_pools 只重写 symbols
    pool.symbols = json.dumps(["AAAUSDT", "BBBUSDT", "DDDUSDT"])
    db.commit()
    assert pools.pool_symbols(db, pool) == ["BBBUSDT", "DDDUSDT", "CCCUSDT"]


def test_pool_row_without_manual_column_is_tolerated():
    """老库/手工构造的行缺列时按空层处理，不抛 AttributeError。"""
    legacy = SimpleNamespace(kind="static", symbols='["AAAUSDT"]', rule="{}")
    assert pools.load_pool_spec(legacy)["manual"] == {"include": [], "exclude": []}


# ---------------------------------------------------------------- 池 API

def test_pool_api_manual_roundtrip_and_partial_update(client):
    client, db = client
    created = client.post("/api/pools", json={
        "code": "p1", "name": "p1", "kind": "static",
        "symbols": ["AAAUSDT", "BBBUSDT"],
        "manual_include": ["CCCUSDT"], "manual_exclude": ["AAAUSDT"],
    })
    assert created.status_code == 200, created.text
    assert created.json()["members_sample"] == ["BBBUSDT", "CCCUSDT"]

    members = client.get("/api/pools/p1/members").json()
    assert members["count"] == 2
    assert members["manual_include"] == ["CCCUSDT"]
    assert members["manual_exclude"] == ["AAAUSDT"]
    assert members["manual_added"] == ["CCCUSDT"]

    # 只改成员列表，不传手工字段 → 手工层原样保留
    updated = client.put("/api/pools/p1", json={
        "symbols": ["AAAUSDT", "BBBUSDT", "DDDUSDT"],
    })
    assert updated.status_code == 200, updated.text
    assert updated.json()["members_sample"] == ["BBBUSDT", "DDDUSDT", "CCCUSDT"]
    assert updated.json()["manual_include"] == ["CCCUSDT"]

    # 传空数组 = 明确清空人工层
    cleared = client.put("/api/pools/p1", json={"manual_include": [], "manual_exclude": []})
    assert cleared.status_code == 200
    assert cleared.json()["members_sample"] == ["AAAUSDT", "BBBUSDT", "DDDUSDT"]
    assert cleared.json()["manual_include"] == [] and cleared.json()["manual_exclude"] == []


def test_pool_api_rejects_oversized_manual_list(client):
    client, _ = client
    bad = client.post("/api/pools", json={
        "code": "p1", "name": "p1", "kind": "static", "symbols": ["AAAUSDT"],
        "manual_include": [f"S{i}USDT" for i in range(201)],
    })
    assert bad.status_code == 400
    assert "不能超过" in bad.json()["detail"]


# ---------------------------------------------------------------- 绑定 API

def test_binding_api_full_edit_keeps_id_and_params(client):
    client, db = client
    client.post("/api/pools", json={"code": "p1", "name": "p1", "kind": "static", "symbols": ["AAAUSDT"]})
    client.post("/api/pools", json={"code": "p2", "name": "p2", "kind": "static", "symbols": ["BBBUSDT", "CCCUSDT"]})
    created = _create_binding(client, params={"target_r": 3.0})
    binding_id = created["id"]

    edited = client.put(f"/api/bindings/{binding_id}", json={
        "pool": "p2", "interval": "1h", "priority": 7, "enabled": False,
    })
    assert edited.status_code == 200, edited.text
    payload = edited.json()
    assert payload["id"] == binding_id
    assert (payload["strategy"], payload["pool"], payload["interval"]) == (STRATEGY, "p2", "1h")
    assert payload["priority"] == 7 and payload["enabled"] is False
    # 关键：不传 params 时原覆盖不能丢
    assert payload["params"] == {"target_r": 3.0}
    assert payload["effective_params"]["target_r"] == 3.0
    assert payload["member_count"] == 2 and payload["pool_name"] == "p2"

    # 换策略时同步清空旧策略专属参数，避免 resolve_params 拒绝未知键
    switched = client.put(f"/api/bindings/{binding_id}", json={
        "strategy": OTHER_STRATEGY, "params": {}, "enabled": True,
    })
    assert switched.status_code == 200, switched.text
    assert switched.json()["strategy"] == OTHER_STRATEGY
    assert switched.json()["id"] == binding_id
    assert switched.json()["pool"] == "p2" and switched.json()["interval"] == "1h"


def test_binding_edit_rechecks_duplicate_scope(client):
    client, db = client
    client.post("/api/pools", json={"code": "p1", "name": "p1", "kind": "static", "symbols": ["AAAUSDT"]})
    _create_binding(client, interval="15m")
    second = _create_binding(client, interval="1h")

    conflict = client.put(f"/api/bindings/{second['id']}", json={"interval": "15m"})
    assert conflict.status_code == 400
    assert "已存在" in conflict.json()["detail"]
    # 失败不能把行改坏
    assert client.get("/api/bindings").json()["bindings"][1]["interval"] == "1h"


def test_binding_edit_validates_new_identity(client):
    client, db = client
    client.post("/api/pools", json={"code": "p1", "name": "p1", "kind": "static", "symbols": ["AAAUSDT"]})
    created = _create_binding(client, params={"target_r": 3.0})

    unknown_pool = client.put(f"/api/bindings/{created['id']}", json={"pool": "nope"})
    assert unknown_pool.status_code == 400 and "币种池" in unknown_pool.json()["detail"]

    unknown_strategy = client.put(f"/api/bindings/{created['id']}", json={"strategy": "nope"})
    assert unknown_strategy.status_code == 400 and "未知策略" in unknown_strategy.json()["detail"]

    # 换策略但保留旧策略专属参数 → 必须被参数校验挡下，不能写出坏行
    stale_params = client.put(f"/api/bindings/{created['id']}", json={
        "strategy": OTHER_STRATEGY,
    })
    assert stale_params.status_code == 400
    assert client.get("/api/bindings").json()["bindings"][0]["strategy"] == STRATEGY


def test_binding_toggle_only_keeps_identity(client):
    """回归：只改开关时身份与参数全部不动（旧行为不能漂）。"""
    client, db = client
    client.post("/api/pools", json={"code": "p1", "name": "p1", "kind": "static", "symbols": ["AAAUSDT"]})
    created = _create_binding(client, params={"target_r": 3.0})

    toggled = client.put(f"/api/bindings/{created['id']}", json={"enabled": False})
    assert toggled.status_code == 200
    for key in ("strategy", "pool", "interval", "priority", "params"):
        assert toggled.json()[key] == created[key]
    assert toggled.json()["enabled"] is False


def test_manual_pool_layer_feeds_active_bindings(db):
    """巡检看到的成员必须包含人工加入、排除人工剔除的币。"""
    from app.strategy.bindings import active_bindings

    db.add(StrategyConfig(code=STRATEGY, name=STRATEGY, enabled=1, params="{}"))
    _pool(db, code="p1", symbols=("AAAUSDT", "BBBUSDT"),
          manual={"include": ["CCCUSDT"], "exclude": ["AAAUSDT"]})
    from app.models import StrategyBinding
    db.add(StrategyBinding(strategy_code=STRATEGY, pool_code="p1", interval="15m",
                           enabled=1, priority=10, params="{}"))
    db.commit()

    resolved, errors = active_bindings(db)
    assert errors == [], errors
    assert resolved[0].symbols == ("BBBUSDT", "CCCUSDT")
