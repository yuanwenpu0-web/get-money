"""迭代 7：免费多源币种指标采集 + 特征 + 存储 + 观测接入测试。

全部 mock 网络层（httpx.Client / BinanceFuturesClient._request），不发起真实请求；
存储用 conftest 的 :memory: sqlite（已建好 coin_metrics 表）。
"""

import httpx
import pandas as pd
import pytest

from app.config import DEFAULT_STATE
from app.market import metrics as metrics_mod
from app.market.futures import BinanceFuturesClient
from app.market.metrics import (
    build_metric_features,
    fetch_fear_greed,
    fetch_market_ranks,
    fetch_symbol_metrics,
    load_latest_metrics,
    load_metric_history,
    save_metrics_snapshot,
    sync_universe_metrics,
)
from app.strategy.rules import evaluate_signals, resolve_params

# ---------- 桩数据 ----------

LS_ROW = {"symbol": "BTCUSDT", "longAccount": "0.6773", "longShortRatio": "2.0989",
          "shortAccount": "0.3227", "timestamp": 1700000000000}
SMART_ROW = {"symbol": "BTCUSDT", "longAccount": "0.55", "longShortRatio": "1.22",
             "shortAccount": "0.45", "timestamp": 1700000000000}
TAKER_ROW = {"buySellRatio": "1.8679", "sellVol": "163.4040", "buyVol": "305.2210",
             "timestamp": 1700000000000}
OI_ROW = {"symbol": "BTCUSDT", "openInterest": "93619.047", "time": 1700000000000}
OI_HIST = [
    {"symbol": "BTCUSDT", "sumOpenInterest": "90000.00", "sumOpenInterestValue": "7200000000.00",
     "timestamp": 1699999990000},
    {"symbol": "BTCUSDT", "sumOpenInterest": "93263.11", "sumOpenInterestValue": "7508823560.68",
     "timestamp": 1700000000000},
]


def _fake_request(self, method, path, params=None, signed=False):
    captured = _fake_request.captured
    captured.update({"method": method, "path": path, "params": params, "signed": signed})
    if "/futures/data/globalLongShortAccountRatio" in path:
        return [dict(LS_ROW)]
    if "/futures/data/topLongShortPositionRatio" in path:
        return [dict(SMART_ROW)]
    if "/futures/data/takerlongshortRatio" in path:
        return [dict(TAKER_ROW)]
    if "/fapi/v1/openInterest" in path:
        return dict(OI_ROW)
    if "/futures/data/openInterestHist" in path:
        return [dict(r) for r in OI_HIST]
    return []


_fake_request.captured = {}


# ---------- ① 客户端 5 方法解析 ----------

def test_long_short_ratio_parses(monkeypatch):
    monkeypatch.setattr(BinanceFuturesClient, "_request", _fake_request)
    rows = BinanceFuturesClient().long_short_ratio("BTCUSDT", period="5m", limit=1)
    c = _fake_request.captured
    assert c["method"] == "GET" and c["signed"] is False
    assert c["path"] == "/futures/data/globalLongShortAccountRatio"
    assert rows == [{"timestamp": 1700000000000, "long_account": 0.6773,
                     "short_account": 0.3227, "long_short_ratio": 2.0989}]


def test_top_trader_long_short_parses(monkeypatch):
    monkeypatch.setattr(BinanceFuturesClient, "_request", _fake_request)
    rows = BinanceFuturesClient().top_trader_long_short("BTCUSDT", period="5m", limit=1)
    c = _fake_request.captured
    assert c["path"] == "/futures/data/topLongShortPositionRatio"
    assert rows == [{"timestamp": 1700000000000, "long_account": 0.55,
                     "short_account": 0.45, "long_short_ratio": 1.22}]


def test_taker_buy_sell_ratio_parses(monkeypatch):
    monkeypatch.setattr(BinanceFuturesClient, "_request", _fake_request)
    rows = BinanceFuturesClient().taker_buy_sell_ratio("BTCUSDT", period="5m", limit=1)
    c = _fake_request.captured
    assert c["path"] == "/futures/data/takerlongshortRatio"
    assert rows == [{"timestamp": 1700000000000, "buy_sell_ratio": 1.8679,
                     "buy_vol": 305.221, "sell_vol": 163.404}]


def test_open_interest_parses(monkeypatch):
    monkeypatch.setattr(BinanceFuturesClient, "_request", _fake_request)
    row = BinanceFuturesClient().open_interest("BTCUSDT")
    c = _fake_request.captured
    assert c["path"] == "/fapi/v1/openInterest"
    assert row == {"open_interest": 93619.047, "time": 1700000000000}


def test_open_interest_history_parses(monkeypatch):
    monkeypatch.setattr(BinanceFuturesClient, "_request", _fake_request)
    rows = BinanceFuturesClient().open_interest_history("BTCUSDT", period="5m", limit=2)
    c = _fake_request.captured
    assert c["path"] == "/futures/data/openInterestHist"
    # 交易所按时间升序返回，[-1] 为最新
    assert rows[0] == {"timestamp": 1699999990000, "sum_open_interest": 90000.0,
                       "sum_open_interest_value": 7200000000.0}
    assert rows[-1] == {"timestamp": 1700000000000, "sum_open_interest": 93263.11,
                        "sum_open_interest_value": 7508823560.68}


# ---------- ② fng / 市值排行 解析 ----------

class _FakeHttpResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _FakeHttpClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, params=None, timeout=None):
        if "alternative.me/fng" in url:
            return _FakeHttpResponse({"data": [{"value": "54", "value_classification": "Neutral",
                                                "timestamp": "1700000000"}]})
        if "coingecko" in url:
            return _FakeHttpResponse([
                {"symbol": "btc", "market_cap_rank": 1, "market_cap": 1300000000000,
                 "total_volume": 25000000000, "price_change_percentage_24h": 1.5,
                 "price_change_percentage_7d": -2.3,
                 "price_change_percentage_24h_in_currency": 1.5,
                 "price_change_percentage_7d_in_currency": -2.3},
            ])
        return _FakeHttpResponse({})


@pytest.fixture
def fake_httpx(monkeypatch):
    monkeypatch.setattr(httpx, "Client", _FakeHttpClient)


def test_fetch_fear_greed_parses(fake_httpx):
    fng = fetch_fear_greed()
    assert fng == {"value": 54, "classification": "Neutral", "timestamp": 1700000000}


def test_fetch_market_ranks_uppercases_and_picks_24h_7d(fake_httpx):
    ranks = fetch_market_ranks()
    assert "BTC" in ranks
    assert ranks["BTC"]["market_cap_rank"] == 1
    assert ranks["BTC"]["market_cap"] == 1300000000000.0
    assert ranks["BTC"]["total_volume"] == 25000000000.0
    assert ranks["BTC"]["change_24h"] == 1.5
    assert ranks["BTC"]["change_7d"] == -2.3


def test_fetch_fear_greed_returns_none_on_error(monkeypatch):
    def boom(**kwargs):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(httpx, "Client", boom)
    assert fetch_fear_greed() is None


def test_fetch_market_ranks_returns_none_on_error(monkeypatch):
    def boom(**kwargs):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(httpx, "Client", boom)
    assert fetch_market_ranks() is None


# ---------- ③ build_metric_features ----------

def _snapshot(crowd_ls=2.0989, smart_ls=1.22, taker=1.8679, oi=93619.047, oi_value=7508823560.68):
    return {"crowd_ls": crowd_ls, "smart_ls": smart_ls, "taker_buy_sell": taker,
            "oi": oi, "oi_value": oi_value}


FNG = {"value": 54, "classification": "Neutral", "timestamp": 1700000000}
RANK = {"market_cap_rank": 1, "market_cap": 1300000000000.0, "total_volume": 25000000000.0,
        "change_24h": 1.5, "change_7d": -2.3}


def test_build_features_divergence_and_passthrough():
    feats = build_metric_features(_snapshot(), FNG, RANK, history=[])
    assert feats["crowd_ls"] == 2.0989
    assert feats["smart_ls"] == 1.22
    assert feats["taker_buy_sell"] == 1.8679
    assert feats["sentiment_divergence"] == pytest.approx(1.22 - 2.0989)
    assert feats["fng_value"] == 54.0
    assert feats["market_cap_rank"] == 1
    assert feats["change_24h"] == 1.5
    assert feats["change_7d"] == -2.3
    # 无历史：oi_change_pct 与 z-score 均应为 0
    assert feats["oi_change_pct"] == 0.0
    assert feats["crowd_ls_z"] == 0.0


def test_build_features_oi_change_pct_from_history():
    history = [{"oi": 46809.5235}]  # 上一快照 oi = 一半
    feats = build_metric_features(_snapshot(oi=93619.047), FNG, RANK, history=history)
    assert feats["oi_change_pct"] == pytest.approx(1.0)  # 翻倍


def test_build_features_zscore_requires_8_samples():
    # 7 个样本 < 8 → z 仍 0
    history = [{"crowd_ls": 1.0 + 0.01 * i} for i in range(7)]
    feats = build_metric_features(_snapshot(crowd_ls=2.0), FNG, RANK, history=history)
    assert feats["crowd_ls_z"] == 0.0
    # 8 个样本 → z 非 0
    history8 = [{"crowd_ls": 1.0 + 0.01 * i} for i in range(8)]
    feats8 = build_metric_features(_snapshot(crowd_ls=2.0), FNG, RANK, history=history8)
    assert feats8["crowd_ls_z"] != 0.0


def test_build_features_nan_inf_collapse_to_zero():
    snap = _snapshot(crowd_ls=float("nan"), smart_ls=float("inf"), taker=float("-inf"), oi=float("nan"))
    feats = build_metric_features(snap, FNG, RANK, history=[])
    assert feats["crowd_ls"] == 0.0
    assert feats["smart_ls"] == 0.0
    assert feats["taker_buy_sell"] == 0.0
    assert feats["oi"] == 0.0
    assert feats["sentiment_divergence"] == 0.0  # inf - nan → nan → 0
    assert feats["crowd_ls_z"] == 0.0


# ---------- ④ 存储 roundtrip ----------

def test_save_load_roundtrip(db):
    feats = build_metric_features(_snapshot(), FNG, RANK, history=[])
    ts = 1700000000000
    save_metrics_snapshot(db, "BTCUSDT", feats, ts)
    loaded = load_latest_metrics(db, "BTCUSDT")
    assert loaded == feats
    # 大小写不敏感
    assert load_latest_metrics(db, "btcusdt") == feats
    history = load_metric_history(db, "BTCUSDT", limit=10)
    assert len(history) == 1 and history[0] == feats


def test_load_latest_returns_none_when_empty(db):
    assert load_latest_metrics(db, "NONEUSDT") is None


# ---------- ⑤ fetch_symbol_metrics 聚合 ----------

def test_fetch_symbol_metrics_assembles(monkeypatch):
    monkeypatch.setattr(BinanceFuturesClient, "_request", _fake_request)
    snap = fetch_symbol_metrics(BinanceFuturesClient(), "BTCUSDT", "5m", 1)
    assert snap["symbol"] == "BTCUSDT"
    assert snap["crowd_ls"] == 2.0989
    assert snap["smart_ls"] == 1.22
    assert snap["taker_buy_sell"] == 1.8679
    assert snap["oi"] == 93619.047
    assert snap["oi_value"] == 7508823560.68  # 来自 open_interest_history


# ---------- ⑥ 决策不变式：market_context 不改变方向/仓位 ----------

def test_evaluate_signals_ignores_market_context():
    params = resolve_params("trend_follow", dict(DEFAULT_STATE))
    idx = [pd.Timestamp("2024-01-01 00:00:00")]
    data = pd.DataFrame({
        "close": [100.0], "atr_pct": [0.01], "rsi_14": [50.0], "adx_14": [30.0],
        "supertrend_dir": [1.0], "ema_ratio_9_21": [1.02], "ema_ratio_21_50": [1.01],
        "macd_hist_norm": [0.5], "funding_zscore": [0.0], "funding_available": [1],
        "bb_pos": [0.5],
    }, index=idx)
    out_a = evaluate_signals(data, "trend_follow", params, None)
    out_b = evaluate_signals(data, "trend_follow", params, None,
                             market_context={"crowd_ls": 9.9, "fng_value": 99})
    assert out_a.iloc[0].equals(out_b.iloc[0])


# ---------- ⑦ 配置：默认关闭 + 类型校验 ----------

def test_metrics_assist_default_off():
    assert DEFAULT_STATE.get("metrics_assist_enabled") is False


def test_validate_accepts_bool_and_rejects_non_bool():
    from app.api.settings import _validate
    from fastapi import HTTPException
    _validate({"metrics_assist_enabled": False})  # 不应抛
    _validate({"metrics_assist_enabled": True})   # 不应抛
    with pytest.raises(HTTPException):
        _validate({"metrics_assist_enabled": "yes"})


def test_settings_update_schema_accepts_flag():
    from app.schemas import SettingsUpdate
    assert SettingsUpdate(metrics_assist_enabled=True).metrics_assist_enabled is True
    assert SettingsUpdate().metrics_assist_enabled is None


# ---------- ⑧ API 端点（路由只装配，网络全 mock） ----------

@pytest.fixture
def client(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from sqlalchemy.pool import StaticPool

    from app.api.market import router
    from app.db import Base, get_db as real_get_db
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    # 自建 StaticPool 内存库：跨 TestClient 门户线程共享同一连接（避免 :memory: 每连接新建库）
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)

    def _override():
        with Session(engine, expire_on_commit=False) as session:
            yield session

    monkeypatch.setattr(BinanceFuturesClient, "_request", _fake_request)
    monkeypatch.setattr(httpx, "Client", _FakeHttpClient)
    # 最小 app：仅挂载 market 路由，绕过 app.main 的 lifespan/自动训练副作用（离线测试）
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[real_get_db] = _override
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()
    engine.dispose()


def test_metrics_get_404_when_empty(client):
    resp = client.get("/api/market/metrics?symbol=BTCUSDT")
    assert resp.status_code == 404


def test_metrics_sync_then_get_roundtrip(client):
    # 同步落库后再读取，验证端到端管线（同一内存库）
    resp = client.post("/api/market/metrics/sync", json={"symbols": ["BTCUSDT"], "source": "binance"})
    assert resp.status_code == 200
    assert resp.json()["synced"] == 1
    get = client.get("/api/market/metrics?symbol=BTCUSDT")
    assert get.status_code == 200
    body = get.json()
    assert body["symbol"] == "BTCUSDT"
    assert body["metrics"]["crowd_ls"] == 2.0989
    assert body["metrics"]["fng_value"] == 54.0
    assert body["metrics"]["market_cap_rank"] == 1


def test_metrics_sync_endpoint_runs_offline_pipeline(client):
    resp = client.post("/api/market/metrics/sync", json={"symbols": ["BTCUSDT"], "source": "binance"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["synced"] == 1
    # 响应体即落库特征：与桩数据一致（不依赖跨 fixture 读取）
    stored = body["results"][0]["features"]
    assert stored["crowd_ls"] == 2.0989
    assert stored["smart_ls"] == 1.22
    assert stored["taker_buy_sell"] == 1.8679
    assert stored["oi_value"] == 7508823560.68
    assert stored["fng_value"] == 54.0
    assert stored["market_cap_rank"] == 1


def test_metrics_sync_rejects_invalid_source(client):
    resp = client.post("/api/market/metrics/sync", json={"symbols": ["BTCUSDT"], "source": "kraken"})
    assert resp.status_code == 400


def test_sync_universe_metrics_isolates_failures(monkeypatch):
    """逐币种失败不影响其余：用假 client 让一个币种抛错。"""
    monkeypatch.setattr(httpx, "Client", _FakeHttpClient)

    def fake_request(self, method, path, params=None, signed=False):
        if params and params.get("symbol") == "FAILUSDT":
            raise RuntimeError("simulated failure")
        return _fake_request(self, method, path, params, signed)

    monkeypatch.setattr(BinanceFuturesClient, "_request", fake_request)
    from sqlalchemy.orm import Session
    from sqlalchemy import create_engine
    engine = create_engine("sqlite:///:memory:")
    from app.db import Base
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        result = sync_universe_metrics(s, ["BTCUSDT", "FAILUSDT"], proxy=None, period="5m")
        assert result["synced"] == 1
        assert len(result["errors"]) == 1
        assert result["errors"][0]["symbol"] == "FAILUSDT"
