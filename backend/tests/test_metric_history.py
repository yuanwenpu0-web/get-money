"""迭代8-A：指标历史回填 + metrics_as_of 反前视原语的离线单测。

不访问真实网络/真实 DB：BinanceFuturesClient 的 4 类方法与 httpx.Client 全部替换为桩，
存储用 conftest 的 :memory: sqlite。
"""

import httpx
import pytest

from app.market.metrics import save_metrics_snapshot
from app.market.metrics_history import (
    FNG_SENTINEL,
    backfill_fear_greed,
    backfill_symbol_metrics,
    load_market_rank,
    metrics_as_of,
)
from app.models import CoinMetric

DAY_MS = 86_400_000
BASE_TS = 1_700_000_000_000


# ---------- 桩：日报历史 ----------

class FakeFuturesClient:
    """模拟已解析的 futures 输出（含 timestamp 键）。oi_hist 故意少一天，验证前向填充。"""

    def long_short_ratio(self, symbol, period="1d", limit=500):
        return [{"timestamp": BASE_TS + i * DAY_MS, "long_short_ratio": round(1.0 + 0.1 * i, 4),
                 "long_account": 0.5, "short_account": 0.5} for i in range(10)]

    def top_trader_long_short(self, symbol, period="1d", limit=500):
        return [{"timestamp": BASE_TS + i * DAY_MS, "long_short_ratio": 2.0,
                 "long_account": 0.6, "short_account": 0.4} for i in range(10)]

    def taker_buy_sell_ratio(self, symbol, period="1d", limit=500):
        return [{"timestamp": BASE_TS + i * DAY_MS, "buy_sell_ratio": 1.5,
                 "buy_vol": 100.0, "sell_vol": 90.0} for i in range(10)]

    def open_interest_history(self, symbol, period="1d", limit=500):
        # 只到 i=8，最后一天缺失 → 应前向填充为 1008.0
        return [{"timestamp": BASE_TS + i * DAY_MS, "sum_open_interest": 1000.0 + i,
                 "sum_open_interest_value": 1_000_000.0 + i * 1000} for i in range(9)]


class _FakeResponse:
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
        if "alternative.me" in url:
            return _FakeResponse({"data": [
                {"value": "30", "value_classification": "Fear", "timestamp": str(int(BASE_TS / 1000))},
                {"value": "55", "value_classification": "Greed",
                 "timestamp": str(int(BASE_TS / 1000) + DAY_MS // 1000)},
                {"value": "70", "value_classification": "Greed",
                 "timestamp": str(int(BASE_TS / 1000) + 2 * DAY_MS // 1000)},
            ]})
        if "coingecko" in url:
            return _FakeResponse([{"symbol": "btc", "market_cap_rank": 1}])
        return _FakeResponse({})


@pytest.fixture
def fake_httpx(monkeypatch):
    monkeypatch.setattr(httpx, "Client", _FakeHttpClient)


# ---------- ① 回填解析与对齐 ----------

def test_backfill_symbol_metrics_rows_alignment_and_forward_fill(db):
    result = backfill_symbol_metrics(FakeFuturesClient(), db, "BTCUSDT", period="1d", limit=500)
    assert result["rows"] == 10
    assert result["span_days"] == 9
    assert db.query(CoinMetric).filter(CoinMetric.symbol == "BTCUSDT").count() == 10
    # 第 5 天快照：crowd_ls = 1.5（1.0 + 0.1*5）
    assert metrics_as_of(db, "BTCUSDT", BASE_TS + 5 * DAY_MS)["crowd_ls"] == pytest.approx(1.5)
    # 最后一天 oi 缺失 → 前向填充为第 9 天(i=8)的 1008.0
    assert metrics_as_of(db, "BTCUSDT", BASE_TS + 9 * DAY_MS)["oi"] == pytest.approx(1008.0)


def test_backfill_symbol_metrics_empty_returns_zero(db):
    class Empty(FakeFuturesClient):
        def long_short_ratio(self, symbol, period="1d", limit=500):
            return []

        def top_trader_long_short(self, symbol, period="1d", limit=500):
            return []

        def taker_buy_sell_ratio(self, symbol, period="1d", limit=500):
            return []

        def open_interest_history(self, symbol, period="1d", limit=500):
            return []

    assert backfill_symbol_metrics(Empty(), db, "AAAUSDT")["rows"] == 0


def test_backfill_fear_greed_stores_sentinel(db, fake_httpx):
    result = backfill_fear_greed(db, proxy=None, limit=500)
    assert result["rows"] == 3
    assert result["span_days"] == 2
    latest = metrics_as_of(db, FNG_SENTINEL, BASE_TS + 2 * DAY_MS)
    assert latest["value"] == 70 and latest["classification"] == "Greed"


def test_backfill_fear_greed_returns_error_shape_on_failure(db, monkeypatch):
    def boom(**kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "Client", boom)
    result = backfill_fear_greed(db, proxy=None)
    assert result["rows"] == 0 and "error" in result


# ---------- ② metrics_as_of 反前视（硬红线） ----------

def test_metrics_as_of_never_reads_future_rows(db):
    for index, mark in enumerate(["a", "b", "c"]):
        save_metrics_snapshot(db, "SOLUSDT", {"mark": mark}, 100 + index * 100)
    # ts 落在 b 与 c 之间 → 只取 b，绝不取 c
    assert metrics_as_of(db, "SOLUSDT", 250)["mark"] == "b"
    assert metrics_as_of(db, "SOLUSDT", 299)["mark"] == "b"
    # 恰好等于 c 的时间戳才可取 c
    assert metrics_as_of(db, "SOLUSDT", 300)["mark"] == "c"
    assert metrics_as_of(db, "SOLUSDT", 10_000)["mark"] == "c"
    # 早于全部历史 → 空（前向填充无来源）
    assert metrics_as_of(db, "SOLUSDT", 99) == {}


def test_metrics_as_of_is_case_insensitive_and_empty_when_missing(db):
    save_metrics_snapshot(db, "BTCUSDT", {"crowd_ls": 1.25}, BASE_TS)
    assert metrics_as_of(db, "btcusdt", BASE_TS)["crowd_ls"] == 1.25
    assert metrics_as_of(db, "NOPEUSDT", BASE_TS) == {}


# ---------- ③ load_market_rank ----------

def test_load_market_rank_prefers_stored_and_zero_when_absent(db):
    save_metrics_snapshot(db, "BTCUSDT", {"market_cap_rank": 7}, BASE_TS)
    assert load_market_rank(db, "BTCUSDT") == 7
    assert load_market_rank(db, "NOPEUSDT") == 0  # 无 proxy → 不联网
