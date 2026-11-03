import sys
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from sqlalchemy import create_engine, text
from compare_model_orders import EIGHT_HOURS, common_window, decisions, load_funding, simulate_orders


def test_decisions_require_confident_direction():
    probabilities = np.array([[.1, .3, .6], [.55, .3, .15], [.3, .4, .3], [.2, .35, .45]])
    assert decisions(probabilities).tolist() == [1, -1, 0, 1]


def test_realistic_orders_charge_fees_slippage_and_funding():
    index = pd.date_range("2026-01-01", periods=25, freq="min", tz="UTC")
    frame = pd.DataFrame({"open": 100., "high": 100.1, "low": 99.9, "close": 100., "atr_pct": 0.}, index=index)
    metrics, trades, _ = simulate_orders(frame, np.array([0]), np.array([1]), {index[10]: (.001, 100.)})
    assert metrics["total_trades"] == 1
    assert trades[0]["entry_time"] == str(index[1]) and trades[0]["exit_time"] == str(index[16])
    assert trades[0]["entry_price"] > 100 and trades[0]["exit_price"] < 100
    assert trades[0]["fees"] > 0 and trades[0]["funding_pnl"] < 0
    assert metrics["final_equity"] < 10000


def test_intrabar_stop_precedes_take_profit():
    index = pd.date_range("2026-01-01", periods=25, freq="min", tz="UTC")
    frame = pd.DataFrame({"open": 100., "high": 100.1, "low": 99.9, "close": 100., "atr_pct": 0.}, index=index)
    frame.loc[index[1], ["high", "low"]] = [102., 99.]
    _, trades, _ = simulate_orders(frame, np.array([0]), np.array([1]), {})
    assert trades[0]["status"] == "stop_loss"


def test_funding_duplicate_is_resolved_by_authoritative_event():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE funding_rates(symbol TEXT, funding_time INTEGER, funding_rate FLOAT, mark_price FLOAT)"))
        conn.execute(text("INSERT INTO funding_rates VALUES ('BTCUSDT',0,0.01,100),('BTCUSDT',1,0.02,101),('BTCUSDT',28800000,0.03,102)"))
    class Client:
        def get(self, url, params):
            assert params["startTime"] == 0
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: [{"symbol": "BTCUSDT", "fundingTime": 1, "fundingRate": "0.02", "markPrice": "101"}])
    events, proxies, resolved = load_funding(engine, "BTCUSDT", 0, EIGHT_HOURS, Client())
    assert events[pd.Timestamp(0, unit="ms", tz="UTC")] == (.02, 101.)
    assert proxies == 0 and resolved == 1


def test_common_window_respects_binary_test_end(tmp_path):
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE funding_rates(symbol TEXT, funding_time INTEGER)"))
        conn.execute(text("INSERT INTO funding_rates VALUES ('BTCUSDT',0),('BTCUSDT',288000000)"))
    model = tmp_path / "models" / "BTCUSDT"
    (model / "transformer").mkdir(parents=True)
    (model / "model_config.json").write_text(json.dumps({"split": {"test": {"first_open_ms": 0, "last_open_ms": 250000000}}}))
    (model / "transformer" / "model_config.json").write_text(json.dumps({"sequence_length": 1}))
    binary = tmp_path / "binary" / "BTCUSDT"
    binary.mkdir(parents=True)
    (binary / "report.json").write_text(json.dumps({"split": {"test": {"first_open_ms": 0, "last_open_ms": 200000000}}}))
    assert common_window(engine, ["BTCUSDT"], tmp_path / "models", tmp_path / "binary")[1] == 200000000
