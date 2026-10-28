import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import pine
from app.db import get_db
from app.pine import compile_pine


SOURCE = '''//@version=6
strategy("Example")
if close > open
    strategy.entry("L", strategy.long)
if close < open
    strategy.close("L")
'''


def client():
    app = FastAPI()
    app.include_router(pine.router, prefix="/api")
    app.dependency_overrides[get_db] = lambda: None
    return TestClient(app)


def bars():
    return [dict(open_time=i * 60000, open=100, high=102, low=98,
                 close=101 if i == 0 else 99, volume=1) for i in range(3)]


def test_convert_and_reject_unsupported():
    http = client()
    result = http.post("/api/pine/convert", json={"source": SOURCE})
    assert result.status_code == 200
    assert "compile_pine" in result.json()["python"]
    assert http.post("/api/pine/convert", json={"source": SOURCE + 'x = request.security()'}).status_code == 422


def test_compare_detects_reference_signal_difference():
    http = client()
    reference = compile_pine(SOURCE).evaluate(pd.DataFrame(bars())).to_dict("records")
    payload = {"source": SOURCE, "candles": bars(), "reference": reference}
    assert http.post("/api/pine/compare", json=payload).json()["passed"]
    reference[0]["long_entry"] = False
    report = http.post("/api/pine/compare", json=payload).json()
    assert not report["passed"]
    assert report["mismatches"][0]["field"] == "long_entry"


def test_backtest_next_open_and_reject_gap(monkeypatch):
    monkeypatch.setattr(pine, "now_ms", lambda: 180000)
    http = client()
    response = http.post("/api/pine/backtest", json={"source": SOURCE, "candles": bars(), "fee_rate": 0})
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["trades"][0]["entry_time"] == 60000
    assert result["trades"][0]["exit_time"] == 120000
    candles = bars()
    candles[1]["open_time"] += 1
    assert http.post("/api/pine/backtest", json={"source": SOURCE, "candles": candles}).status_code == 422


def test_filters_unclosed_and_reads_local(monkeypatch):
    monkeypatch.setattr(pine, "now_ms", lambda: 120000)
    class Service:
        def __init__(self, db):
            pass
        def load_frame(self, symbol, interval, limit):
            frame = pd.DataFrame(bars())
            frame.index = pd.to_datetime(frame.pop("open_time"), unit="ms", utc=True)
            frame.index.name = "time"
            return frame
    monkeypatch.setattr(pine, "MarketDataService", Service)
    response = client().post("/api/pine/backtest", json={"source": SOURCE})
    assert response.status_code == 200
    assert [row["open_time"] for row in response.json()["signals"]] == [0, 60000]


def test_binance_fetch_uses_futures_and_excludes_incomplete(monkeypatch):
    monkeypatch.setattr(pine, "now_ms", lambda: 150000)
    monkeypatch.setattr(pine, "all_state", lambda db: {})
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["testnet"] is False
        def server_time(self):
            return 150000
        def connection_pool(self):
            from contextlib import nullcontext
            return nullcontext()
        def klines(self, symbol, interval, start_ms, limit):
            return [[row["open_time"], row["open"], row["high"], row["low"], row["close"], row["volume"], row["open_time"] + 59999, 1, 1] for row in bars()]
    monkeypatch.setattr(pine, "BinanceFuturesClient", Client)
    response = client().post("/api/pine/backtest", json={"source": SOURCE, "data_source": "binance", "limit": 3})
    assert response.status_code == 200
    assert [row["open_time"] for row in response.json()["signals"]] == [0, 60000]


def test_paper_account_persists_rejects_gaps_and_unconfirmed(monkeypatch, tmp_path):
    monkeypatch.setattr(pine, "PAPER_DIR", tmp_path)
    http = client()
    url = "/api/pine/paper/test-account"
    assert http.get(url).status_code == 404
    assert http.post(url, json={"source": SOURCE, "settings": {"fee_rate": 0}}).status_code == 200
    assert http.post(url, json={"source": SOURCE}).status_code == 409
    result = http.post(url + "/candles", json={"candles": bars()})
    assert result.status_code == 200
    assert result.json()["result"]["trades"][0]["entry_time"] == 60000
    assert http.get(url).json() == result.json()
    assert http.post(url + "/candles", json={"candles": bars()}).json() == result.json()
    gap = dict(bars()[0], open_time=300000)
    assert http.post(url + "/candles", json={"candles": [gap]}).status_code == 422
    future = dict(bars()[0], open_time=9999999999999)
    assert http.post(url + "/candles", json={"candles": [future]}).status_code == 422
    assert http.post("/api/pine/paper/bad.id", json={"source": SOURCE}).status_code == 422


def test_explicit_candles_are_not_silently_limited(monkeypatch):
    monkeypatch.setattr(pine, "now_ms", lambda: 1_000_000_000)
    candles = [dict(open_time=i * 60000, open=100, high=102, low=98, close=100, volume=1) for i in range(3001)]
    response = client().post("/api/pine/backtest", json={"source": SOURCE, "candles": candles, "limit": 2})
    assert response.status_code == 200
    assert len(response.json()["signals"]) == 3001
    candles[1]["open_time"] = None
    assert client().post("/api/pine/backtest", json={"source": SOURCE, "candles": candles}).status_code == 422


def test_paper_runner_api_lifecycle_and_conflict(monkeypatch, tmp_path):
    monkeypatch.setattr(pine, "PAPER_DIR", tmp_path)
    monkeypatch.setattr(pine, "_runners", {})
    class Runner:
        def __init__(self, path, symbol):
            assert path.exists()
            self.symbol, self.state = symbol, "stopped"
        def start(self):
            self.state = "running"
        def status(self):
            return {"status": self.state, "symbol": self.symbol, "error": None}
        def stop(self):
            self.state = "stopped"
            return True
    monkeypatch.setattr(pine, "PaperRunner", Runner)
    http = client()
    url = "/api/pine/paper/runner-test"
    assert http.get(url + "/status").status_code == 404
    assert http.post(url, json={"source": SOURCE}).status_code == 200
    assert http.get(url + "/status").json()["status"] == "stopped"
    assert http.post(url + "/start", json={"symbol": "BTCUSDT"}).json()["status"] == "running"
    assert http.post(url + "/start", json={"symbol": "BTCUSDT"}).status_code == 409
    assert http.post(url + "/candles", json={"candles": bars()}).status_code == 409
    assert http.get(url + "/status").json()["status"] == "running"
    pine.stop_all_paper_runners()
    assert http.post(url + "/stop").json()["status"] == "stopped"


def test_real_app_lifespan_and_local_database(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    script = """
from fastapi.testclient import TestClient
from app.main import app
from app.db import SessionLocal
from app.models import Kline
source = '//@version=6\\nstrategy("Sample")\\nif close > open\\n    strategy.entry("L", strategy.long)'
with TestClient(app) as http:
    assert http.get('/api/health').status_code == 200
    with SessionLocal() as db:
        for i in range(3):
            t = 1700000040000 + i * 60000
            db.add(Kline(symbol='BTCUSDT', interval='1m', open_time=t, close_time=t+59999, open=100, high=102, low=99, close=101, volume=1))
        db.commit()
    response = http.post('/api/pine/backtest', json={'source': source, 'data_source': 'local'})
    assert response.status_code == 200, response.text
    assert [row['open_time'] for row in response.json()['signals']] == [1700000040000 + i * 60000 for i in range(3)]
"""
    env = dict(os.environ, DATABASE_URL="sqlite:///" + (tmp_path / "lifespan.db").as_posix())
    result = subprocess.run([sys.executable, "-c", script], cwd=str(Path(__file__).resolve().parents[1]), env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
