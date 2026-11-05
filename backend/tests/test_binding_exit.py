"""绑定层出场覆盖：注册表校验、专用流程拦截、支撑位止损与参数贯通。

覆盖的契约：
1. `method=strategy`（默认）时信号帧逐位不变，改造前后零行为差异；
2. `atr_r` / `support_r` 只改止损止盈距离，仓位、杠杆、成本过滤按**新距离**重算；
3. 支撑位缺失/过远时回退 ATR，并在 `exit_fallback` 列留痕；
4. 专用撮合与研究流程的策略拒绝挂覆盖，避免「看起来改了其实没改」或伪造信号。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.models import StrategyBinding, SymbolPool
from app.strategy import bindings
from app.strategy.exits import ensure_allowed, normalize
from app.strategy.rules import apply_exit_overlay, resolve_params

STRATEGY = "absorption_reversal"


def flat_frame(n: int = 60, low: float = 119.5, high: float = 120.5, close: float = 120.0):
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": np.full(n, close),
            "high": np.full(n, high),
            "low": np.full(n, low),
            "close": np.full(n, close),
            "volume": np.full(n, 100.0),
        },
        index=idx,
    )
    frame["atr_pct"] = 0.01
    return frame


def signal_frame(index, direction: float = 1.0):
    n = len(index)
    return pd.DataFrame(
        {
            "signed": np.full(n, direction),
            "score": np.full(n, 0.8),
            "prob_up": np.full(n, 0.6),
            "direction": np.full(n, int(direction)),
            "block_reason": np.full(n, "", dtype=object),
            "position_pct": np.full(n, 0.3),
            "leverage": np.full(n, 2),
            "sl_pct": np.full(n, 0.005),
            "tp_pct": np.full(n, 0.01),
            "risk_pct": np.full(n, 0.03),
            "net_reward_risk": np.full(n, 2.0),
        },
        index=index,
    )


# ---------------------------------------------------------------- 注册表校验

def test_normalize_fills_defaults_and_rejects_bad_input():
    cfg = normalize({"method": "atr_r"})
    assert cfg["params"] == {"atr_mult": 1.5, "target_r": 2.0}

    with pytest.raises(ValueError, match="未知出场方法"):
        normalize({"method": "magic"})
    with pytest.raises(ValueError, match="未知支撑线指标"):
        normalize({"method": "support_r", "support": "nope"})
    with pytest.raises(ValueError, match="必须选择支撑线指标"):
        normalize({"method": "support_r"})
    with pytest.raises(ValueError, match="不支持参数"):
        normalize({"method": "atr_r", "params": {"buffer_atr": 0.5}})
    with pytest.raises(ValueError, match=r"atr_r.atr_mult 必须在"):
        normalize({"method": "atr_r", "params": {"atr_mult": 99}})
    assert normalize(None) == {"method": "strategy", "support": "", "params": {}}


def test_resolve_params_blocks_dedicated_and_research_strategies():
    cfg = {"method": "atr_r"}
    for code in ("rgd_donchian_v2", "opening_candle_breakout", "simple_zigzag_double"):
        with pytest.raises(ValueError, match="专用撮合"):
            resolve_params(code, {}, {"exit": cfg})
    for code in ("research_high_turnover_high_vol", "research_low_turnover_high_vol"):
        with pytest.raises(ValueError, match="流程决定"):
            resolve_params(code, {}, {"exit": cfg})
    # 通用撮合的策略可以挂
    assert resolve_params(STRATEGY, {}, {"exit": cfg})["exit"]["method"] == "atr_r"


def test_ensure_allowed_keeps_strategy_default_open():
    # 默认方法不触发任何拦截，老绑定升级后行为不变
    ensure_allowed("rgd_donchian_v2", {"method": "strategy", "support": "", "params": {}})


# ---------------------------------------------------------------- 覆盖计算

def test_strategy_method_returns_untouched_frame():
    frame = flat_frame()
    signals = signal_frame(frame.index)
    params = resolve_params(STRATEGY, {}, None)
    out = apply_exit_overlay(frame, signals, params)
    assert out is signals
    assert "exit_level" not in out.columns


def test_atr_r_sets_distance_from_atr():
    frame = flat_frame()
    signals = signal_frame(frame.index)
    cfg = normalize({"method": "atr_r", "params": {"atr_mult": 2.0, "target_r": 3.0}})
    params = resolve_params(STRATEGY, {}, {"exit": cfg})
    out = apply_exit_overlay(frame, signals, params)
    assert out.sl_pct.iloc[-1] == pytest.approx(0.02)
    assert out.tp_pct.iloc[-1] == pytest.approx(0.06)
    # 止盈放大后净盈亏比同步重算，仓位按新止损距离放大
    assert out.net_reward_risk.iloc[-1] > signals.net_reward_risk.iloc[-1]
    assert out.position_pct.iloc[-1] >= 0


def test_support_r_places_stop_below_level_and_mirrors_for_short():
    frame = flat_frame()
    long_cfg = normalize({"method": "support_r", "support": "donchian20",
                          "params": {"target_r": 2.0, "buffer_atr": 0.25}})
    params = resolve_params(STRATEGY, {}, {"exit": long_cfg})
    out = apply_exit_overlay(frame, signal_frame(frame.index), params)
    level = out.exit_level.iloc[-1]
    assert level == pytest.approx(119.5)          # 20 根滚动最低价
    expected = (120.0 - level * (1 - 0.25 * 0.01)) / 120.0
    assert out.sl_pct.iloc[-1] == pytest.approx(expected)
    assert out.tp_pct.iloc[-1] == pytest.approx(expected * 2)
    assert int(out.exit_fallback.iloc[-1]) == 0

    short = apply_exit_overlay(frame, signal_frame(frame.index, direction=-1.0), params)
    assert short.exit_level.iloc[-1] == pytest.approx(120.5)   # 滚动最高价
    assert short.sl_pct.iloc[-1] > 0


def test_support_r_falls_back_when_level_is_too_far():
    frame = flat_frame()
    frame.loc[frame.index[-1], "close"] = 400.0        # 水平远离价格，超过 max_sl_atr
    frame.loc[frame.index[-1], ["high", "low", "open"]] = [401.0, 399.0, 400.0]
    cfg = normalize({"method": "support_r", "support": "donchian20"})
    params = resolve_params(STRATEGY, {}, {"exit": cfg})
    out = apply_exit_overlay(frame, signal_frame(frame.index), params)
    assert int(out.exit_fallback.iloc[-1]) == 1
    assert out.sl_pct.iloc[-1] == pytest.approx(1.5 * 0.01)   # 回退 atr_mult × ATR
    assert np.isnan(out.exit_level.iloc[-1])


def test_overlay_respects_min_distance_floor():
    frame = flat_frame()
    frame["atr_pct"] = 0.00001                                  # 极低波动
    cfg = normalize({"method": "atr_r", "params": {"atr_mult": 1.0}})
    params = resolve_params(STRATEGY, {}, {"exit": cfg})
    out = apply_exit_overlay(frame, signal_frame(frame.index), params)
    assert out.sl_pct.iloc[-1] == pytest.approx(params["min_sl_pct"])


def test_overlay_distance_is_causal():
    """第 t 根的距离只能由 t 及之前的数据决定：截断末尾不影响历史行。"""
    frame = flat_frame()
    signals = signal_frame(frame.index)
    cfg = normalize({"method": "support_r", "support": "donchian20"})
    params = resolve_params(STRATEGY, {}, {"exit": cfg})
    full = apply_exit_overlay(frame, signals, params)
    cut = apply_exit_overlay(frame.iloc[:-5], signals.iloc[:-5], params)
    pd.testing.assert_series_equal(full.sl_pct.iloc[:-5], cut.sl_pct)


# ---------------------------------------------------------------- 绑定贯通

def _pool(db, code="test", symbols=("AAAUSDT",)):
    row = SymbolPool(code=code, name=code, kind="static",
                     symbols=json.dumps(list(symbols)), rule="{}", enabled=1)
    db.add(row)
    db.commit()
    return row


def _binding(db, pool="test", interval="15m", **kwargs):
    row = StrategyBinding(
        strategy_code=STRATEGY, pool_code=pool, interval=interval,
        enabled=kwargs.pop("enabled", 1), priority=10,
        params=json.dumps(kwargs.pop("params", {})), **kwargs,
    )
    db.add(row)
    db.commit()
    return row


def test_binding_columns_flow_into_effective_params(db):
    _pool(db)
    row = _binding(db, exit_method="support_r", support_indicator="pivot_pool",
                   exit_params=json.dumps({"target_r": 2.5}))
    params = bindings.binding_params(db, row)
    assert params["exit"]["method"] == "support_r"
    assert params["exit"]["support"] == "pivot_pool"
    assert params["exit"]["params"]["target_r"] == 2.5
    # 默认空缺的参数被补全，下游不需要判断缺键
    assert params["exit"]["params"]["buffer_atr"] == 0.25


def test_default_binding_has_no_exit_overlay(db):
    _pool(db)
    row = _binding(db)
    assert bindings.exit_config_of(row) is None
    assert bindings.binding_params(db, row)["exit"]["method"] == "strategy"


def test_active_bindings_carry_exit_config(db):
    from app.models import StrategyConfig

    db.add(StrategyConfig(code=STRATEGY, name=STRATEGY, params="{}", enabled=1))
    _pool(db)
    _binding(db, exit_method="atr_r", exit_params=json.dumps({"atr_mult": 2.0}))
    db.commit()

    resolved, errors = bindings.active_bindings(db)
    assert errors == []
    assert resolved[0].exit["params"]["atr_mult"] == 2.0
    assert resolved[0].exit_label.startswith("ATR 2.0")


# ---------------------------------------------------------------- API

@pytest.fixture
def client():
    from fastapi import FastAPI
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from sqlalchemy.pool import StaticPool

    from app.api import bindings as bindings_api
    from app.api import pools as pools_api
    from app.db import Base, get_db

    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
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


def test_api_exit_options_lists_methods_and_sources(client):
    client, _ = client
    data = client.get("/api/bindings/exit-options").json()
    codes = [item["code"] for item in data["methods"]]
    assert codes == ["strategy", "atr_r", "support_r"]
    assert any(item["code"] == "support_r" and item["needs_support"] for item in data["methods"])
    assert len(data["support_sources"]) == 11
    assert {item["code"] for item in data["support_sources"]} >= {"pivot_pool", "donchian20"}


def test_api_binding_roundtrip_with_exit(client):
    client, db = client
    from app.models import StrategyConfig

    db.add(StrategyConfig(code=STRATEGY, name=STRATEGY, params="{}", enabled=1))
    db.commit()
    client.post("/api/pools", json={"code": "p1", "name": "p1", "kind": "static", "symbols": ["AAAUSDT"]})

    bad = client.post("/api/bindings", json={
        "strategy": STRATEGY, "pool": "p1", "interval": "15m",
        "exit_method": "support_r", "support_indicator": "nope",
    })
    assert bad.status_code == 400 and "未知支撑线指标" in bad.json()["detail"]

    missing = client.post("/api/bindings", json={
        "strategy": STRATEGY, "pool": "p1", "interval": "15m", "exit_method": "support_r",
    })
    assert missing.status_code == 400

    created = client.post("/api/bindings", json={
        "strategy": STRATEGY, "pool": "p1", "interval": "15m",
        "exit_method": "support_r", "support_indicator": "donchian20",
        "exit_params": {"target_r": 2.5},
    })
    assert created.status_code == 200
    payload = created.json()
    assert payload["exit_method"] == "support_r"
    assert payload["support_indicator"] == "donchian20"
    assert payload["exit_params"]["target_r"] == 2.5
    assert payload["exit"]["verified"] is False          # 支撑位出场未过统计验证
    assert "donchian20" in payload["exit"]["label"] or "支撑位" in payload["exit"]["label"]
    assert payload["effective_params"]["exit"]["method"] == "support_r"

    # 部分更新不带出场字段时不得把它冲回默认
    moved = client.put(f"/api/bindings/{payload['id']}", json={"priority": 5})
    assert moved.status_code == 200
    assert moved.json()["exit_method"] == "support_r"

    reset = client.put(f"/api/bindings/{payload['id']}", json={"exit_method": "strategy",
                                                               "support_indicator": ""})
    assert reset.status_code == 200
    assert reset.json()["exit"]["method"] == "strategy"
    assert reset.json()["exit"]["verified"] is True


def test_api_rejects_overlay_on_broker_strategy(client):
    client, db = client
    from app.models import StrategyConfig

    db.add(StrategyConfig(code="rgd_donchian_v2", name="rgd", params="{}", enabled=1))
    db.commit()
    client.post("/api/pools", json={"code": "p1", "name": "p1", "kind": "static", "symbols": ["AAAUSDT"]})
    response = client.post("/api/bindings", json={
        "strategy": "rgd_donchian_v2", "pool": "p1", "interval": "15m", "exit_method": "atr_r",
    })
    assert response.status_code == 400 and "专用撮合" in response.json()["detail"]
