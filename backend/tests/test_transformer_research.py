import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import importlib.util
import json
import sys
from types import SimpleNamespace

torch = pytest.importorskip("torch")
from app.ml.transformer_research import PriceTransformer, sequence_endpoints
from app.ml.research_backtest import simulate


def test_sequences_stay_inside_partition_and_exclude_gaps():
    positions = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
    assert sequence_endpoints(positions, slice(0, 10), 3, 1).tolist() == [2, 3, 4, 7, 8, 9]
    assert sequence_endpoints(positions, slice(5, 10), 3, 1).tolist() == [7, 8, 9]


def test_transformer_output_shapes_and_finite_probabilities():
    torch.manual_seed(1)
    model = PriceTransformer(5, sequence_length=8, d_model=8, nhead=2, layers=1)
    direction, regression = model(torch.zeros(2, 8, 5))
    assert direction.shape == (2, 3) and regression.shape == (2, 2)
    assert torch.isfinite(direction).all()


def test_backtest_enters_on_next_open_and_deducts_costs():
    frame = pd.DataFrame({"open_time": np.arange(30) * 60000, "open": np.ones(30) * 100})
    p = np.array([[.05, .05, .9]])
    result = simulate(frame, [0], p, [.01], fee=.001, slippage=.001)
    assert result["trades"] == 1 and result["total_return"] < 0
    trade = result["transactions"][0]
    assert trade["entry_open_ms"] == 60000 and trade["exit_open_ms"] == 16 * 60000
    gap = frame.drop(index=5).reset_index(drop=True)
    assert simulate(gap, [0], p, [.01])["trades"] == 0


def test_waiter_ignores_failure_from_an_older_training_session(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    from train_major_transformers import terminal_lightgbm_status
    from train_major_models import register_session
    assert not terminal_lightgbm_status({"status": "failed", "session_id": "old"}, "new")
    assert not terminal_lightgbm_status({"status": "running", "session_id": "new"}, "new")
    assert terminal_lightgbm_status({"status": "failed", "session_id": "new"}, "new")
    assert terminal_lightgbm_status({"status": "candidate", "session_id": "old"}, "new")
    config = {}
    register_session(config, "new")
    with pytest.raises(ValueError, match="唯一"):
        register_session(config, "new")


def test_research_pipeline_trains_compares_and_backtests(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    import train_major_models
    import train_transformer_research
    import predict_research
    from test_minute_research import candles
    frame = candles(2500)
    directory = tmp_path / "BTCUSDT"
    train_major_models.fit_models(frame, directory, trees=3, threads=1,
                                  metadata={"symbol": "BTCUSDT", "as_of_ms": 999999999, "model_version": "test"})
    monkeypatch.setattr(train_transformer_research, "make_engine", lambda **kwargs: SimpleNamespace(dispose=lambda: None))
    monkeypatch.setattr(train_transformer_research, "load_candles", lambda *args: frame)
    monkeypatch.setattr(train_transformer_research, "validate_candles", lambda *args: (frame, {"passed": True}))
    args = SimpleNamespace(run=tmp_path, symbol="BTCUSDT", threads=1, sequence=16, stride=2,
                           width=8, layers=1, epochs=1, batch_size=16, patience=1)
    train_transformer_research.train(args)
    output = directory / "transformer"
    report = json.loads((output / "evaluation_report.json").read_text(encoding="utf8"))
    backtest = json.loads((output / "backtest_report.json").read_text(encoding="utf8"))
    assert (output / "model.pt").exists()
    assert report["sampled_rows"]["test"] > 20 and "ensemble" in report
    assert set(backtest) == {"lightgbm", "transformer", "ensemble", "momentum"}
    cfg = json.loads((directory / "model_config.json").read_text(encoding="utf8"))
    assert predict_research.earliest_availability(cfg, "ensemble") > predict_research.earliest_availability(cfg, "lightgbm")
    predict_research.require_fresh(frame, int(frame.open_time.iloc[-1]) + 60000)
    with pytest.raises(ValueError, match="最新已收盘"):
        predict_research.require_fresh(frame, int(frame.open_time.iloc[-1]) + 120000)
    for model in ("lightgbm", "transformer"):
        result = predict_research.predict_frame(frame, directory, model)
        assert abs(result["up_probability"] + result["down_probability"] + result["sideways_probability"] - 1) < 1e-6
        assert result["expected_volatility"] >= 0
    model_config = output / "model_config.json"
    settings = json.loads(model_config.read_text(encoding="utf8"))
    settings["ensemble_enabled"] = False
    model_config.write_text(json.dumps(settings), encoding="utf8")
    with pytest.raises(ValueError, match="独立测试"):
        predict_research.predict_frame(frame, directory, "ensemble")
