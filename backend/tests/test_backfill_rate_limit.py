import importlib.util
from pathlib import Path
import sys

import httpx
import pytest

scripts = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("backfill_all_history", scripts / "backfill_all_history.py")
backfill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backfill)


def test_429_honours_retry_after_and_retries(monkeypatch):
    seen = []
    responses = iter([httpx.Response(429, headers={"Retry-After": "90"}), httpx.Response(200)])
    with httpx.Client(transport=httpx.MockTransport(lambda request: next(responses))) as raw:
        client = backfill.RateLimitedClient(raw, archive_interval=0, api_interval=0)
        monkeypatch.setattr(client, "_defer", seen.append)
        assert client.get("https://fapi.binance.com/test").status_code == 200
    assert seen[0] >= 90


@pytest.mark.parametrize("status", [403, 418])
def test_block_stops_all_following_requests(status):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status)
    with httpx.Client(transport=httpx.MockTransport(handler)) as raw:
        client = backfill.RateLimitedClient(raw, archive_interval=0, api_interval=0)
        with pytest.raises(backfill.AccessBlocked):
            client.get("https://fapi.binance.com/test")
        with pytest.raises(backfill.AccessBlocked):
            client.get("https://data.binance.vision/test")
    assert len(calls) == 1


def test_missing_archive_is_not_retried():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(404)
    with httpx.Client(transport=httpx.MockTransport(handler)) as raw:
        client = backfill.RateLimitedClient(raw, archive_interval=0, api_interval=0)
        assert client.get("https://data.binance.vision/missing").status_code == 404
    assert len(calls) == 1


def test_major_selection_excludes_other_coins_and_prioritizes_btc_eth():
    records = [{"symbol": symbol, "quoteAsset": "USDT", "contractType": "PERPETUAL"}
               for symbol in ("ETHUSDT", "SOMEALTUSDT", "BTCUSDT")]
    assert [x["symbol"] for x in backfill.major_contracts(records)] == ["BTCUSDT", "ETHUSDT"]


def test_correction_only_applies_to_exact_source_and_valid_volume():
    original = dict(open_time=0, close_time=59999, open=10., high=12., low=9., close=11.,
                    volume=1., quote_volume=20., trades=3, taker_buy_volume=2., taker_buy_quote_volume=10.)
    fixed = {**original, "volume": 3.}
    data = {0: original.copy()}
    entry = {"symbol": "BTCUSDT", "original": original, "corrected": fixed}
    backfill.apply_verified_corrections("BTCUSDT", data, [entry])
    assert data[0]["volume"] == 3.
    with pytest.raises(ValueError):
        backfill.apply_verified_corrections("BTCUSDT", {0: {**original, "close": 10.}}, [entry])


@pytest.mark.parametrize("outcome", ["done", "partial", "error"])
def test_run_checkpoints_and_limits_writes_to_month(tmp_path, monkeypatch, outcome):
    import sqlite3
    from datetime import date, datetime, timezone
    from types import SimpleNamespace

    state = tmp_path / "state.sqlite"
    monkeypatch.setattr(backfill, "STATE", state)
    engine = SimpleNamespace(dispose=lambda: None)
    monkeypatch.setattr(backfill, "make_engine", lambda **kwargs: engine)
    monkeypatch.setattr(backfill, "free_space_check", lambda engine: None)
    monkeypatch.setattr(backfill, "major_contracts", lambda symbols: [{"symbol": "BTCUSDT"}])
    month = date(2024, 1, 1)
    next_month = date(2024, 2, 1)
    to_ms = lambda d: int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)
    task = ("BTCUSDT", month, next_month, to_ms(date(2023, 12, 1)), to_ms(date(2024, 3, 1)), True)
    monkeypatch.setattr(backfill, "tasks_for", lambda *args: iter([task]))
    def fetch(client, task):
        if outcome == "error":
            raise ValueError("下载失败")
        return [], ["2024-01-01"] if outcome == "partial" else []
    monkeypatch.setattr(backfill, "fetch_month", fetch)
    calls = []
    def apply(engine, symbol, rows, start, end):
        calls.append((start, end))
        return {"inserted": 0, "updated": 0, "mismatched": 0, "invalid_times": []}
    monkeypatch.setattr(backfill, "apply_rows", apply)
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"symbols": []})
    monkeypatch.setattr(backfill.RateLimitedClient, "get", lambda *args, **kwargs: response)
    if outcome == "done":
        backfill.run(workers=1)
    else:
        with pytest.raises(RuntimeError, match="进度已保存"):
            backfill.run(workers=1)
    with sqlite3.connect(state) as db:
        status, detail = db.execute("SELECT status, detail FROM months").fetchone()
    assert status == outcome
    if outcome == "error":
        assert "下载失败" in detail and not calls
    else:
        assert calls == [(to_ms(month), to_ms(next_month) - 60000)]
