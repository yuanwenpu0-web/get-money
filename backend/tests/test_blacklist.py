"""币种黑名单落地测试：排除、归一、与流动性过滤叠加、立即生效、设置校验。全 mock 网络。"""

import pytest
from fastapi import HTTPException

from app.api.settings import _normalize_blacklist, _validate, update_settings
from app.market import universe
from app.models import JobLog
from app.schemas import SettingsUpdate
from app.state import set_state

from universe_helpers import FakeUniverseClient, clear_universe_cache, patch_universe, row


@pytest.fixture(autouse=True)
def _clear_universe_cache():
    clear_universe_cache()
    yield
    clear_universe_cache()


def test_blacklist_excludes_symbols_from_universe(db, monkeypatch):
    fake = FakeUniverseClient(symbols=[row("BTCUSDT"), row("ETHUSDT"), row("SOLUSDT")])
    patch_universe(monkeypatch, fake)
    set_state(db, "symbol_blacklist", ["ETHUSDT"])
    assert universe.universe_symbols(db) == ["BTCUSDT", "SOLUSDT"]


def test_blacklist_normalizes_case_space_and_dedupes(db, monkeypatch):
    fake = FakeUniverseClient(symbols=[row("BTCUSDT"), row("ETHUSDT"), row("SOLUSDT")])
    patch_universe(monkeypatch, fake)
    set_state(db, "symbol_blacklist", [" ethusdt ", "ETHUSDT", "EthUsdt"])
    assert universe.universe_symbols(db) == ["BTCUSDT", "SOLUSDT"]


def test_blacklist_stacks_with_liquidity_filter(db, monkeypatch):
    fake = FakeUniverseClient(
        symbols=[row("BTCUSDT"), row("ETHUSDT"), row("SOLUSDT"), row("XRPUSDT")],
        volumes={"BTCUSDT": 50_000_000.0, "ETHUSDT": 40_000_000.0,
                 "SOLUSDT": 30_000_000.0, "XRPUSDT": 20_000_000.0},
    )
    patch_universe(monkeypatch, fake)
    # 默认门槛 30M：XRP 被流动性剔除；再叠加黑名单去掉 ETH
    set_state(db, "symbol_blacklist", ["ETHUSDT"])
    assert universe.universe_symbols(db) == ["BTCUSDT", "SOLUSDT"]
    assert fake.ticker_calls == 1  # 两层过滤一次拉取完成


def test_blacklist_change_takes_effect_without_ttl(db, monkeypatch):
    fake = FakeUniverseClient(symbols=[row("BTCUSDT"), row("ETHUSDT")])
    patch_universe(monkeypatch, fake)
    assert universe.universe_symbols(db) == ["BTCUSDT", "ETHUSDT"]
    set_state(db, "symbol_blacklist", ["ETHUSDT"])
    # 缓存键含过滤配置：不手动过期也要立即生效
    assert universe.universe_symbols(db) == ["BTCUSDT"]
    assert fake.calls == 2


def test_empty_blacklist_behaves_as_before(db, monkeypatch):
    fake = FakeUniverseClient(symbols=[row("BTCUSDT"), row("ETHUSDT")])
    patch_universe(monkeypatch, fake)
    set_state(db, "symbol_blacklist", [])
    assert universe.universe_symbols(db) == ["BTCUSDT", "ETHUSDT"]
    set_state(db, "symbol_blacklist", None)
    assert universe.universe_symbols(db) == ["BTCUSDT", "ETHUSDT"]


def test_blacklist_emptying_universe_logs_warning(db, monkeypatch):
    fake = FakeUniverseClient(symbols=[row("BTCUSDT")])
    patch_universe(monkeypatch, fake)
    set_state(db, "symbol_blacklist", ["BTCUSDT"])
    assert universe.universe_symbols(db) == []
    rows = db.query(JobLog).filter(JobLog.job == "market_universe").all()
    assert len(rows) == 1 and "黑名单清空" in rows[0].message


def test_settings_validate_blacklist():
    _validate({"symbol_blacklist": []})
    _validate({"symbol_blacklist": ["btcusdt"]})  # 小写在服务层归一，校验不拒绝
    assert _normalize_blacklist([" btcusdt ", "BTCUSDT", "ETHUSDT", "ethusdt"]) == \
        ["BTCUSDT", "ETHUSDT"]
    for bad in ("BTCUSDT", [""], ["  "], [123], [None], {"BTCUSDT"}):
        with pytest.raises(HTTPException) as exc:
            _normalize_blacklist(bad)
        assert exc.value.status_code == 400


def test_settings_update_persists_normalized_blacklist(db):
    result = update_settings(
        SettingsUpdate(symbol_blacklist=[" zecusdt ", "ZECUSDT", "1000pepeusdt"]), db
    )
    assert result["symbol_blacklist"] == ["ZECUSDT", "1000PEPEUSDT"]
    with pytest.raises(HTTPException):
        update_settings(SettingsUpdate(symbol_blacklist=[""]), db)
