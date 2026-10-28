import threading
import time

import pandas as pd
import pytest

from app.pine.paper import append_candles, create_account, read_account, bind_market
from app.pine.paper_runner import PaperRunner

SOURCE = '''//@version=6
strategy("paper")
if close > 105
    strategy.entry("L", strategy.long)
'''


def row(t):
    return [t, "110", "112", "108", "111", "1", t + 59999]


class FakeClient:
    def __init__(self, now, rows):
        self.now = now
        self.rows = rows

    def server_time(self):
        return self.now

    def klines(self, symbol, start_ms, limit):
        return [r for r in self.rows if r[0] >= start_ms][:limit]


def test_bootstrap_replay_restart_and_market_binding(tmp_path):
    path = tmp_path / "paper.json"
    create_account(path, SOURCE, {"fee_rate": 0})
    bind_market(path, "BTCUSDT")
    client = FakeClient(120001, [row(60000), row(120000)])
    runner = PaperRunner(path, "BTCUSDT", client=client)
    assert runner.status()["status"] == "stopped"
    assert runner.poll_once() == 1
    assert runner.poll_once() == 0
    assert read_account(path)["result"]["open_position"] is None
    client.now = 180001
    restarted = PaperRunner(path, "BTCUSDT", client=client)
    assert restarted.status()["status"] == "stopped"
    assert restarted.poll_once() == 1
    assert read_account(path)["result"]["open_position"]["entry_time"] == 120000
    with pytest.raises(ValueError, match="bound"):
        bind_market(path, "ETHUSDT")


def test_gap_stops_worker_and_does_not_append_partial_history(tmp_path):
    path = tmp_path / "paper.json"
    create_account(path, SOURCE)
    bind_market(path, "BTCUSDT")
    client = FakeClient(120001, [row(60000)])
    runner = PaperRunner(path, "BTCUSDT", client=client, poll_seconds=0.01)
    assert runner.poll_once() == 1
    before = read_account(path)
    client.now = 300001
    client.rows = [row(120000), row(240000)]
    runner.start()
    deadline = time.monotonic() + 2
    while runner.status()["status"] == "running" and time.monotonic() < deadline:
        threading.Event().wait(0.01)
    assert runner.status()["status"] == "error"
    assert "missing" in runner.status()["error"]
    assert read_account(path) == before
    assert runner.stop()


def test_future_only_and_excess_backlog_fail_without_writes(tmp_path):
    path = tmp_path / "paper.json"
    create_account(path, SOURCE)
    bind_market(path, "ETHUSDT")
    client = FakeClient(120001, [row(120000)])
    runner = PaperRunner(path, "ETHUSDT", client=client, max_backfill=2)
    with pytest.raises(ValueError, match="missing"):
        runner.poll_once()
    assert read_account(path)["candles"] == []
    client.rows = [row(60000)]
    runner.poll_once()
    state = read_account(path)
    client.now = 360001
    with pytest.raises(ValueError, match="backlog"):
        runner.poll_once()
    assert read_account(path) == state


def test_explicit_start_stop_and_duplicate_worker(tmp_path):
    path = tmp_path / "paper.json"
    create_account(path, SOURCE)
    entered = threading.Event()
    release = threading.Event()
    class BlockingClient(FakeClient):
        def server_time(self):
            entered.set()
            assert release.wait(2)
            return self.now
    client = BlockingClient(120001, [row(60000)])
    runner = PaperRunner(path, "BTCUSDT", client=client)
    runner.start()
    assert entered.wait(2)
    manual = pd.DataFrame([dict(open_time=0, open=110., high=112., low=108., close=111., volume=1.)])
    with pytest.raises(RuntimeError, match="running worker"):
        append_candles(path, manual)
    duplicate = PaperRunner(path, "BTCUSDT", client=client)
    with pytest.raises(RuntimeError, match="already"):
        duplicate.start()
    assert not runner.stop(timeout=0)
    release.set()
    assert runner.stop(timeout=2)
    assert runner.status()["status"] == "stopped"


def test_restart_catches_up_in_pages_and_preserves_gap_boundary(tmp_path):
    path = tmp_path / "paper.json"
    create_account(path, SOURCE)
    bind_market(path, "BTCUSDT")
    class PagedClient(FakeClient):
        def klines(self, symbol, start_ms, limit):
            return super().klines(symbol, start_ms, min(limit, 2))
    client = PagedClient(120001, [row(60000)])
    runner = PaperRunner(path, "BTCUSDT", client=client)
    assert runner.poll_once() == 1
    client.now = 360001
    client.rows = [row(t) for t in range(120000, 360000, 60000)]
    assert PaperRunner(path, "BTCUSDT", client=client).poll_once() == 4
    assert [r["open_time"] for r in read_account(path)["candles"]] == [60000, 120000, 180000, 240000, 300000]


def test_public_client_uses_only_market_endpoints(monkeypatch):
    import io
    from app.pine.paper_runner import BinancePublicCandles
    urls = []
    def fake_open(url, timeout):
        urls.append(url)
        return io.BytesIO(b'{"serverTime":120001}' if url.endswith("/time") else b'[]')
    monkeypatch.setattr("app.pine.paper_runner.urllib.request.urlopen", fake_open)
    client = BinancePublicCandles()
    assert client.server_time() == 120001
    assert client.klines("BTCUSDT", 60000, 5) == []
    assert urls[0] == "https://fapi.binance.com/fapi/v1/time"
    assert urls[1].startswith("https://fapi.binance.com/fapi/v1/klines?")
    assert "symbol=BTCUSDT" in urls[1] and "interval=1m" in urls[1]


def test_unbound_csv_history_cannot_be_mixed_with_binance(tmp_path):
    path = tmp_path / "paper.json"
    create_account(path, SOURCE)
    append_candles(path, pd.DataFrame([dict(open_time=0, open=110., high=112., low=108., close=111., volume=1.)]))
    before = read_account(path)
    runner = PaperRunner(path, "BTCUSDT", client=FakeClient(120001, [row(60000)]))
    with pytest.raises(ValueError, match="empty, unbound"):
        runner.start()
    assert read_account(path) == before


def test_market_bound_account_rejects_manual_candles_even_when_stopped(tmp_path):
    path = tmp_path / "paper.json"
    create_account(path, SOURCE)
    bind_market(path, "BTCUSDT")
    before = read_account(path)
    manual = pd.DataFrame([dict(open_time=0, open=110., high=112., low=108., close=111., volume=1.)])
    with pytest.raises(RuntimeError, match="only runner candles"):
        append_candles(path, manual)
    assert read_account(path) == before
