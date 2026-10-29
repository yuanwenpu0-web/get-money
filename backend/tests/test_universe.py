"""币种宇宙（U 本位永续全量 + 流动性过滤）测试。全部 mock 网络层，不真实请求。
黑名单专项见 test_blacklist.py；共享 Fake 见 universe_helpers.py。"""

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api import market as market_api
from app.api.settings import _validate, update_settings
from app.config import DEFAULT_SYMBOLS
from app.market import universe
from app.models import JobLog, StrategyBinding
from app.schemas import SettingsUpdate, SyncRequest
from app.state import set_state
from app.strategy import bindings
from app.trading import service as trading_service
from app.trading.service import TradingService

from fakes import seed_binding
from universe_helpers import FakeUniverseClient, clear_universe_cache, patch_universe, row


@pytest.fixture(autouse=True)
def _clear_universe_cache():
    clear_universe_cache()
    yield
    clear_universe_cache()


def test_exchange_info_filters_perpetual_usdt_trading(db, monkeypatch):
    fake = FakeUniverseClient(
        symbols=[
            row("BTCUSDT"),
            row("ETHUSDT"),
            row("SOLUSDT"),
            row("SUPERUSDT"),  # 正常合约：不得被子串误判为杠杆代币
            row("BTCUSDT_230929", contract="CURRENT_QUARTER"),  # 交割合约剔除
            row("ETHUSDT_231231", contract="NEXT_QUARTER"),
            row("BTCUSDC", quote="USDC"),  # 非 USDT 计价剔除
            row("XRPUSDT", status="HALT"),  # 非 TRADING 剔除
            row("LTCUSDT", status="SETTLING"),
            row("BTCUPUSDT"),  # 杠杆代币剔除
            row("ETHDOWNUSDT"),
            row("BNBBULLUSDT"),
            row("ADABEARUSDT"),
        ]
    )
    patch_universe(monkeypatch, fake)
    assert universe.universe_symbols(db) == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "SUPERUSDT"]


def test_universe_cache_hit_avoids_refetch(db, monkeypatch):
    fake = FakeUniverseClient(symbols=[row("BTCUSDT"), row("ETHUSDT")])
    patch_universe(monkeypatch, fake)
    assert universe.universe_symbols(db) == ["BTCUSDT", "ETHUSDT"]
    assert universe.universe_symbols(db) == ["BTCUSDT", "ETHUSDT"]
    assert fake.calls == 1  # 第二次命中 TTL 缓存


def test_universe_cache_expires_after_ttl(db, monkeypatch):
    fake = FakeUniverseClient(symbols=[row("BTCUSDT")])
    patch_universe(monkeypatch, fake)
    universe.universe_symbols(db)
    monkeypatch.setattr(universe, "UNIVERSE_TTL_SECONDS", 0)
    universe.universe_symbols(db)
    assert fake.calls == 2


def test_universe_failure_falls_back_to_stale_cache(db, monkeypatch):
    fake = FakeUniverseClient(symbols=[row("BTCUSDT")])
    patch_universe(monkeypatch, fake)
    universe.universe_symbols(db)
    fake.error = RuntimeError("network down")
    monkeypatch.setattr(universe, "UNIVERSE_TTL_SECONDS", 0)  # 缓存过期后重新拉取
    # 失败时回退旧缓存，不缩小到默认列表
    assert universe.universe_symbols(db) == ["BTCUSDT"]
    assert db.query(JobLog).filter(JobLog.job == "market_universe").count() == 1


def test_universe_failure_without_cache_falls_back_to_defaults(db, monkeypatch):
    fake = FakeUniverseClient(error=RuntimeError("network down"))
    patch_universe(monkeypatch, fake)
    assert universe.universe_symbols(db) == DEFAULT_SYMBOLS
    assert db.query(JobLog).filter(JobLog.job == "market_universe").count() == 1


def test_resolve_universe_reports_source_and_cached(db, monkeypatch):
    fake = FakeUniverseClient(symbols=[row("BTCUSDT")])
    patch_universe(monkeypatch, fake)
    # paper 模式（默认）宇宙与K线同源走主网 fapi，不受 testnet 开关影响
    paper = universe.resolve_universe(db)
    assert paper == {"symbols": ["BTCUSDT"], "cached": False, "source": "futures-mainnet"}
    # live+testnet 才走测试网宇宙，与测试网K线同源
    set_state(db, "execution_mode", "live")
    set_state(db, "testnet", True)
    live_testnet = universe.resolve_universe(db)
    assert live_testnet["source"] == "futures-testnet"
    set_state(db, "testnet", False)
    live_mainnet = universe.resolve_universe(db)
    assert live_mainnet["source"] == "futures-mainnet"


def test_run_cycle_follows_bindings_and_scope(db, monkeypatch):
    """巡检的币种范围来自绑定；symbol_scope=custom 只在其内部再做一次交集。"""
    seed_binding(db, ["AAAUSDT"])

    def members(session, pool, allowed=None):
        return [s for s in ("AAAUSDT", "BBBUSDT") if allowed is None or s in allowed]

    monkeypatch.setattr(bindings, "pool_symbols", members)
    monkeypatch.setattr(
        trading_service, "MarketDataService",
        lambda db, client=None: type("Stub", (), {"sync_recent_many": lambda self, *a, **k: {"errors": []}})(),
    )
    monkeypatch.setattr(
        trading_service, "StrategyEngine",
        lambda db, **kwargs: type("Stub", (), {"evaluate": lambda self, *a, **k: None})(),
    )
    set_state(db, "symbol_scope", "all")
    summary = TradingService(db).run_cycle()
    assert summary["ok"] is True
    assert summary["symbols"] == ["AAAUSDT", "BBBUSDT"]
    assert summary["symbol_count"] == 2
    assert summary["bindings"][0]["pool"] == "test"
    assert "elapsed_ms" in summary

    # 自定义名单是全局限制：池里不在名单内的币种本轮不跑
    set_state(db, "symbol_scope", "custom")
    set_state(db, "symbols", ["BBBUSDT"])
    summary = TradingService(db).run_cycle()
    assert summary["symbols"] == ["BBBUSDT"]

    # 没有启用的绑定 → 一个币种都不跑（不再是「全局周期 × 全宇宙」）
    bindings_row = db.query(StrategyBinding).first()
    bindings_row.enabled = 0
    db.commit()
    summary = TradingService(db).run_cycle()
    assert summary["symbols"] == []
    assert summary["bindings"] == []


def test_settings_scope_validation():
    _validate({"symbol_scope": "all", "symbols": []})  # 全量允许空 symbols
    with pytest.raises(HTTPException) as exc:
        _validate({"symbol_scope": "custom", "symbols": []})
    assert exc.value.status_code == 400
    with pytest.raises(HTTPException) as exc:
        _validate({"symbol_scope": "quarterly"})
    assert exc.value.status_code == 400
    with pytest.raises(HTTPException) as exc:
        _validate({"symbols": []})  # 未声明 scope 时维持旧契约
    assert exc.value.status_code == 400


def test_market_symbols_endpoint_structure(db, monkeypatch):
    monkeypatch.setattr(
        market_api, "resolve_universe",
        lambda session: {"symbols": ["BTCUSDT", "ETHUSDT"], "cached": False, "source": "futures-testnet"},
    )
    result = market_api.market_symbols(db)
    assert result["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert result["source"] == "futures-testnet"
    assert result["cached"] is False
    # 前端要能看出「全部」实际被流动性门槛/黑名单砍成了多少
    assert result["count"] == 2
    assert "min_quote_volume_usd" in result
    assert "blacklist" in result

    set_state(db, "symbol_scope", "custom")
    set_state(db, "symbols", ["BTCUSDT"])
    custom = market_api.market_symbols(db)
    assert custom["symbols"] == ["BTCUSDT"]
    assert custom["source"] == "custom"
    assert custom["cached"] is True
    assert custom["count"] == 1


def test_liquidity_filter_drops_symbols_below_min_quote_volume(db, monkeypatch):
    fake = FakeUniverseClient(
        symbols=[row("BTCUSDT"), row("ETHUSDT"), row("SOLUSDT"), row("XRPUSDT")],
        volumes={
            "BTCUSDT": 50_000_000.0,
            "ETHUSDT": 29_999_999.0,   # 低于 30M 门槛，剔除
            "SOLUSDT": 30_000_000.0,   # 等于门槛，保留
            "XRPUSDT": 1_000_000.0,    # 远低于门槛，剔除
        },
    )
    patch_universe(monkeypatch, fake)
    assert universe.universe_symbols(db) == ["BTCUSDT", "SOLUSDT"]


def test_liquidity_filter_disabled_when_threshold_zero(db, monkeypatch):
    fake = FakeUniverseClient(
        symbols=[row("BTCUSDT"), row("XRPUSDT")],
        volumes={"BTCUSDT": 50_000_000.0, "XRPUSDT": 1_000_000.0},
    )
    patch_universe(monkeypatch, fake)
    set_state(db, "min_quote_volume_usd", 0)
    assert universe.universe_symbols(db) == ["BTCUSDT", "XRPUSDT"]
    assert fake.ticker_calls == 0  # 0 = 不过滤，连 ticker 都不请求


def test_ticker_failure_skips_filtering_and_logs(db, monkeypatch):
    fake = FakeUniverseClient(
        symbols=[row("BTCUSDT"), row("XRPUSDT")],
        ticker_error=RuntimeError("ticker down"),
    )
    patch_universe(monkeypatch, fake)
    # 宁贵不漏：拿不到成交额就不过滤
    assert universe.universe_symbols(db) == ["BTCUSDT", "XRPUSDT"]
    rows = db.query(JobLog).filter(JobLog.job == "market_universe").all()
    assert len(rows) == 1 and "ticker/24hr" in rows[0].message


def test_ticker_map_cached_across_exchange_info_refetch(db, monkeypatch):
    fake = FakeUniverseClient(
        symbols=[row("BTCUSDT"), row("XRPUSDT")],
        volumes={"BTCUSDT": 50_000_000.0, "XRPUSDT": 1_000_000.0},
    )
    patch_universe(monkeypatch, fake)
    assert universe.universe_symbols(db) == ["BTCUSDT"]
    assert fake.ticker_calls == 1
    # 只让 exchangeInfo 结果过期、ticker 映射仍新鲜：不得重复拉 ticker
    cache_key = next(iter(universe._cache_at))
    universe._cache_at[cache_key] = 0.0
    assert universe.universe_symbols(db) == ["BTCUSDT"]
    assert fake.calls == 2 and fake.ticker_calls == 1


def test_settings_validate_liquidity_and_halt_keys():
    _validate({"min_quote_volume_usd": 0})
    _validate({"min_quote_volume_usd": 30_000_000})
    _validate({"order_type": "limit_maker"})
    _validate({"risk_halt": False})
    _validate({"max_drawdown_halt_pct": 0.1})
    _validate({"consecutive_loss_halt": 5})
    for bad in (
        {"min_quote_volume_usd": -1},
        {"min_quote_volume_usd": 1.5},
        {"min_quote_volume_usd": True},
        {"order_type": "post_only"},
        {"risk_halt": "yes"},
        {"max_drawdown_halt_pct": 1.5},
        {"max_drawdown_halt_pct": -0.1},
        {"consecutive_loss_halt": -1},
    ):
        with pytest.raises(HTTPException) as exc:
            _validate(bad)
        assert exc.value.status_code == 400


def test_settings_update_persists_new_keys(db):
    result = update_settings(
        SettingsUpdate(min_quote_volume_usd=50_000_000, order_type="limit_maker",
                       risk_halt=True, risk_halt_reason="人工熔断测试",
                       max_drawdown_halt_pct=0.15, consecutive_loss_halt=3),
        db,
    )
    assert result["min_quote_volume_usd"] == 50_000_000
    assert result["order_type"] == "limit_maker"
    assert result["risk_halt"] is True and result["risk_halt_reason"] == "人工熔断测试"
    assert result["max_drawdown_halt_pct"] == 0.15 and result["consecutive_loss_halt"] == 3
    # 负数在 schema 层即被拒（ge=0），API 返回 422 而非进入业务校验
    with pytest.raises(ValidationError):
        SettingsUpdate(min_quote_volume_usd=-5)


def test_sync_empty_symbols_means_full_universe(db, monkeypatch):
    monkeypatch.setattr(market_api, "universe_symbols", lambda session: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    monkeypatch.setattr(market_api.MarketDataService, "sync_many",
                        lambda self, symbols, interval, days: {"requested": len(symbols), "inserted": 300, "errors": []})
    result = market_api.sync_market(SyncRequest(symbols=[], interval="1m", days=7), db)
    assert result["requested"] == 3 and result["inserted"] == 300
