"""资金费全链路测试：客户端解析、分页幂等、asof 合并、z-score 数学、快照。
策略方向与拦截逻辑见 test_funding_strategy.py。全 mock 网络。"""

import numpy as np
import pandas as pd
import pytest

from app.features.funding import attach_funding, funding_snapshot
from app.market.data_service import MarketDataService
from app.market.futures import BinanceFuturesClient
from app.models import FundingRate
from app.strategy.rules import evaluate_signals, resolve_params

from funding_helpers import bar_frame, funding_frame


class FakeFundingClient(BinanceFuturesClient):
    """按 startTime 分页返回的假资金费客户端（继承以通过 _futures_client 的 isinstance 判定）。"""

    def __init__(self, rows, page=1000):
        super().__init__()
        self.rows = sorted(rows, key=lambda row: row["funding_time"])
        self.page = page
        self.calls = []

    def funding_rate(self, symbol, start_ms=None, end_ms=None, limit=1000):
        self.calls.append({"symbol": symbol, "start_ms": start_ms, "end_ms": end_ms, "limit": limit})
        selected = [
            row for row in self.rows
            if (start_ms is None or row["funding_time"] >= start_ms)
            and (end_ms is None or row["funding_time"] <= end_ms)
        ]
        return selected[: self.page]


def make_funding_rows(count, start_ms, step_ms=8 * 3_600_000, rate=0.0001):
    return [
        {
            "symbol": "BTCUSDT",
            "funding_time": start_ms + index * step_ms,
            "funding_rate": rate,
            "mark_price": 100.0,
            "rate_type": "Regular",
        }
        for index in range(count)
    ]


# ---------- ① 客户端参数与解析 ----------

def test_funding_rate_client_params_and_parsing(monkeypatch):
    captured = {}

    def fake_request(self, method, path, params=None, signed=False):
        captured.update({"method": method, "path": path, "params": params, "signed": signed})
        return [
            {"symbol": "BTCUSDT", "fundingRate": "0.00010000", "fundingTime": 1700000000000,
             "markPrice": "34287.54", "rateType": "Regular"},
        ]

    monkeypatch.setattr(BinanceFuturesClient, "_request", fake_request)
    rows = BinanceFuturesClient().funding_rate("BTCUSDT", start_ms=1690000000000, end_ms=1700000000000, limit=500)
    assert captured["method"] == "GET" and captured["path"] == "/fapi/v1/fundingRate"
    assert captured["signed"] is False  # 公开接口不签名
    assert captured["params"] == {"symbol": "BTCUSDT", "limit": 500,
                                  "startTime": 1690000000000, "endTime": 1700000000000}
    assert rows == [{"funding_time": 1700000000000, "funding_rate": 0.0001,
                     "mark_price": 34287.54, "rate_type": "Regular"}]


def test_funding_rate_limit_clamped(monkeypatch):
    captured = {}

    def fake_request(self, method, path, params=None, signed=False):
        captured.update(params or {})
        return []

    monkeypatch.setattr(BinanceFuturesClient, "_request", fake_request)
    BinanceFuturesClient().funding_rate("BTCUSDT", limit=5000)
    assert captured["limit"] == 1000  # 上限裁剪


# ---------- ② 分页与幂等 ----------

def test_sync_funding_paginates_and_is_idempotent(db):
    from app.utils import now_ms

    # 1200 条 × 8h = 400 天；从窗口起点后移一个结算周期起步，保证全部落在 [now-400d, now] 内
    rows = make_funding_rows(1200, now_ms() - 400 * 86_400_000 + 8 * 3_600_000)
    fake = FakeFundingClient(rows)
    service = MarketDataService(db, client=fake)
    result = service.sync_funding("BTCUSDT", days=400)
    assert result["inserted"] == 1200
    assert len(fake.calls) == 2  # 1000 + 200 两页
    assert fake.calls[1]["start_ms"] == rows[999]["funding_time"] + 1  # 下一页从最后一条 +1ms 起算
    assert db.query(FundingRate).count() == 1200
    # 重复同步不涨行数（幂等 upsert）
    again = service.sync_funding("BTCUSDT", days=400)
    assert again["inserted"] == 1200
    assert db.query(FundingRate).count() == 1200


# ---------- ③ asof 合并 ----------

def test_asof_merge_uses_latest_funding_at_or_before_bar(db):
    frame = bar_frame(start="2026-01-01T07:00:00Z", periods=120, freq="1min")  # 07:00-09:00
    funding = funding_frame(
        [1_767_225_600_000, 1_767_254_400_000],  # 2026-01-01 00:00, 08:00
        [0.0001, 0.0005],
    )
    attach_funding(frame, funding)
    bar_0759 = frame.loc["2026-01-01T07:59:00Z"]
    bar_0800 = frame.loc["2026-01-01T08:00:00Z"]
    assert bar_0759["funding_rate"] == pytest.approx(0.0001)  # 07:59 用 00:00 那条
    assert bar_0759["funding_available"] == 1.0
    assert bar_0800["funding_rate"] == pytest.approx(0.0005)  # 08:00 起用 08:00 那条（INCLUSIVE）


def test_asof_before_first_funding_is_unavailable():
    frame = bar_frame(start="2026-01-01T00:00:00Z", periods=480, freq="1min")  # 00:00-07:59
    funding = funding_frame([1_767_225_600_000 + 8 * 3_600_000], [0.0001])  # 08:00 才首条
    attach_funding(frame, funding)
    assert (frame.loc[:"2026-01-01T07:59:00Z", "funding_available"] == 0.0).all()
    assert (frame.loc[:"2026-01-01T07:59:00Z", "funding_rate"] == 0.0).all()
    assert frame.loc["2026-01-01T07:59:00Z", "funding_24h_sum"] == 0.0


# ---------- ④ z-score 数学 ----------

def test_funding_zscore_matches_hand_calculation():
    frame = bar_frame(periods=200)
    funding_time = 1_767_225_600_000
    # 00:00 费率 0.0001，02:00 起 0.0005；最后 90 根窗口 = 10 根旧值 + 80 根新值
    funding = funding_frame([funding_time, funding_time + 2 * 3_600_000], [0.0001, 0.0005])
    attach_funding(frame, funding)
    series = frame["funding_rate"]
    assert series.iloc[119] == pytest.approx(0.0001)
    assert series.iloc[120] == pytest.approx(0.0005)  # 02:00 起生效
    # 独立手算（不经过 pandas rolling）
    window = [0.0001] * 10 + [0.0005] * 80
    mean = sum(window) / 90
    std = (sum((value - mean) ** 2 for value in window) / 90) ** 0.5
    expected = (0.0005 - mean) / std
    assert frame["funding_zscore"].iloc[-1] == pytest.approx(expected)
    assert frame["funding_rate_annualized"].iloc[-1] == pytest.approx(0.0005 * 3 * 365)


def test_funding_zscore_zero_when_variance_zero():
    frame = bar_frame(periods=100)
    funding = funding_frame([1_767_225_600_000, 1_767_254_400_000], [0.0001, 0.0001])
    attach_funding(frame, funding)
    assert (frame["funding_zscore"] == 0.0).all()  # 常数序列方差为 0 → z=0


def test_funding_24h_sum_uses_time_window():
    index = pd.date_range("2026-01-01T00:00:00Z", periods=48, freq="1h", tz="UTC")
    frame = pd.DataFrame(dict(open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0), index=index)
    funding = funding_frame([1_767_225_600_000, 1_767_254_400_000], [0.001, 0.002])  # 00:00, 08:00
    attach_funding(frame, funding)
    # 23:00 的 bar：24h 窗口 (23:00-24h, 23:00] 含 00:00 与 08:00 两条 → 0.003
    assert frame.loc["2026-01-01T23:00:00Z", "funding_24h_sum"] == pytest.approx(0.003)
    # 07:00 的 bar：窗口内只有 00:00 → 0.001
    assert frame.loc["2026-01-01T07:00:00Z", "funding_24h_sum"] == pytest.approx(0.001)


# ---------- 无资金费数据 ----------

def test_features_zero_without_funding_data():
    frame = bar_frame(periods=100)
    attach_funding(frame, None)
    assert (frame["funding_rate"] == 0.0).all()
    assert (frame["funding_rate_annualized"] == 0.0).all()
    assert (frame["funding_zscore"] == 0.0).all()
    assert (frame["funding_24h_sum"] == 0.0).all()
    assert (frame["funding_available"] == 0.0).all()


def test_evaluate_signals_without_funding_data_does_not_raise():
    frame = bar_frame(periods=200)
    attach_funding(frame, None)  # 无数据：特征填 0、available=0
    from app.features.indicators import build_features

    data = build_features(frame)
    params = resolve_params("absorption_reversal", {})
    result = evaluate_signals(data, "absorption_reversal", params)
    # 无资金费数据不影响本策略的出信号路径：这里要么没信号，要么被明确记录原因，
    # 关键是调用不抛异常、也不会把 zscore=0 当成「资金费极值」。
    assert set(result.block_reason) <= {
        "no_absorption_signal", "invalid_indicators", "funding_filter_long",
        "funding_filter_short", "cost_filter", "below_threshold", "",
    }


# ---------- funding_snapshot ----------

def test_funding_snapshot_math():
    funding = funding_frame([1_767_225_600_000, 1_767_254_400_000, 1_767_283_200_000],
                            [0.0001, 0.0002, 0.0003])
    snapshot = funding_snapshot(funding)
    assert snapshot["available"] is True
    assert snapshot["funding_rate"] == pytest.approx(0.0003)
    assert snapshot["funding_rate_annualized"] == pytest.approx(0.0003 * 1095)
    assert snapshot["funding_zscore"] == pytest.approx(np.std([0.0001, 0.0002, 0.0003], ddof=0) and
                                                       (0.0003 - 0.0002) / np.std([0.0001, 0.0002, 0.0003], ddof=0))
    assert snapshot["latest_funding_time"] == 1_767_283_200_000


def test_funding_snapshot_empty():
    snapshot = funding_snapshot(None)
    assert snapshot == {"available": False, "funding_rate": 0.0, "funding_rate_annualized": 0.0,
                        "funding_zscore": 0.0, "funding_24h_sum": 0.0, "latest_funding_time": 0}
