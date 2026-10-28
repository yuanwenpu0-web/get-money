"""币级路由层测试（基于 8×60d walk-forward 56 币结果）。"""
import json
import tempfile
from pathlib import Path

import pytest

from app.strategy.coin_router import (
    BLACKLIST,
    COIN_TYPES,
    SYMBOL_ROUTES,
    TYPE_ROUTES,
    all_routes,
    get_coin_type,
    get_route,
    save_routes,
)


def test_holdout_verified_symbol_routes():
    """SOL/ARIA/QNT/JCT/BR 是 holdout 验证过的精确路由（训练>0 且 holdout>0）。"""
    for sym, expected_strat in [
        ("SOLUSDT", "strong_pullback"),
        ("ARIAUSDT", "mtf_resonance"),
        ("QNTUSDT", "mtf_resonance"),
        ("JCTUSDT", "mtf_resonance"),
        ("BRUSDT", "trend_follow"),
    ]:
        r = get_route(sym)
        assert r is not None, f"{sym} should have route"
        assert r["strategy"] == expected_strat, f"{sym} expected {expected_strat}, got {r['strategy']}"


def test_holdout_negative_not_routed():
    """ENA/HUMA holdout 分别 −0.20%/−0.19% → 不在 SYMBOL_ROUTES（不路由实盘资金）。"""
    assert get_route("ENAUSDT") is None  # 训练正但 holdout 负 → 不路由
    assert get_route("HUMAUSDT") is None


def test_blacklist_returns_none():
    """黑名单（8/8 全负且 mean<-0.1%）路由返回 None。"""
    for sym in BLACKLIST:
        assert get_route(sym) is None, f"{sym} should be blacklisted"


def test_no_family_routes():
    """8×60d 结论：所有类型族群分化太大，TYPE_ROUTES 应为空。"""
    assert TYPE_ROUTES == {}, "族群路由已证明无效，应为空 dict"


def test_unknown_symbol_returns_none():
    """宇宙外的币返回 None（生产环境会 fallback 到遍历所有 enabled 策略）。"""
    assert get_route("FAKEUSDT") is None
    assert get_coin_type("FAKEUSDT") is None


def test_case_insensitive():
    """symbol 大小写不敏感。"""
    assert get_route("solusdt") == get_route("SOLUSDT")
    assert get_route("SolUsdt")["strategy"] == "strong_pullback"


def test_symbol_routes_not_poisoned_by_family():
    """关键纪律：没有族群路由覆盖/污染精确路由。"""
    # ARIAUSDT 属于 Alt（无族群路由），精确路由 → mtf_resonance
    route = get_route("ARIAUSDT")
    assert route["strategy"] == "mtf_resonance"
    # AAVEUSDT 属于 DeFi（无族群路由），不在 SYMBOL_ROUTES → None（fallback）
    assert get_route("AAVEUSDT") is None


def test_all_routes_returns_mapping():
    """all_routes() 应该：SYMBOL_ROUTES 里有 → 返回路由；BLACKLIST → None；其他 → None。"""
    routes = all_routes()
    assert routes["SOLUSDT"]["strategy"] == "strong_pullback"
    assert routes["ARIAUSDT"]["strategy"] == "mtf_resonance"
    assert routes["AAVEUSDT"] is None
    assert routes["NEARUSDT"] is None  # 黑名单
    assert routes["BTCUSDT"] is None  # 无精确路由，L1 无族群 → None


def test_save_load_roundtrip():
    """路由表 JSON 导出/加载必须能还原结构。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "routes.json"
        save_routes(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["coin_types"]["BTCUSDT"] == "L1"
        assert payload["type_routes"] == {}  # 空族群
        assert "SOLUSDT" in payload["symbol_routes"]
        assert sorted(BLACKLIST) == payload["blacklist"]


def test_no_ml_ensemble_auto_routing():
    """精确路由表里不应该出现 ml_ensemble — 该策略需要训练，路由层只做 rule-based 分发。"""
    for sym, info in SYMBOL_ROUTES.items():
        assert info["strategy"] != "ml_ensemble", f"{sym} routes to ml_ensemble — not allowed"
