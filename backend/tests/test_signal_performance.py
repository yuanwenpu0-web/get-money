"""Cycle-local feature reuse must preserve every strategy's signal."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from app.api.settings import _validate
from app.models import Signal
from app.schemas import SettingsUpdate
from app.strategy import engine


CODES = ("ml_ensemble", "trend_follow", "mean_reversion", "funding_extreme", "mtf_resonance")


@pytest.fixture
def candle_frame(monkeypatch):
    index = pd.date_range("2026-01-01", periods=400, freq="1h", tz="UTC")
    close = 100 + np.arange(400) * .04 + np.sin(np.arange(400) / 8)
    frame = pd.DataFrame(
        dict(open=close - .02, high=close + .3, low=close - .3,
             close=close, volume=100 + np.arange(400) % 10), index=index,
    )
    monkeypatch.setattr(engine, "now_ms", lambda: int(index[-1].timestamp() * 1000) + 3_600_000)
    monkeypatch.setattr(engine.MarketDataService, "load_frame", lambda *a, **k: frame)
    return frame


def test_cycle_reuses_features_without_changing_or_skipping_saved_signals(db, monkeypatch, candle_frame):
    expected = {
        code: engine.StrategyEngine(db).evaluate("BTCUSDT", "1h", code).to_dict()
        for code in CODES
    }
    calls = dict(features=0, mtf=0, candles=0, funding=0, metrics=0, state=0)

    def counted(name, original):
        def run(*args, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)
        return run

    monkeypatch.setattr(engine, "build_features", counted("features", engine.build_features))
    monkeypatch.setattr(engine, "add_mtf_trend", counted("mtf", engine.add_mtf_trend))
    monkeypatch.setattr(engine, "all_state", counted("state", engine.all_state))
    monkeypatch.setattr(engine.MarketDataService, "load_frame", counted("candles", engine.MarketDataService.load_frame))
    monkeypatch.setattr(engine.MarketDataService, "load_funding_frame", counted("funding", engine.MarketDataService.load_funding_frame))
    monkeypatch.setattr(engine.metrics_mod, "load_latest_metrics", counted("metrics", engine.metrics_mod.load_latest_metrics))

    evaluator = engine.StrategyEngine(db, cycle_snapshot=True)
    for code in CODES:
        assert evaluator.evaluate("BTCUSDT", "1h", code, save=True).to_dict() == expected[code]
    assert db.query(Signal).count() == len(CODES)
    assert calls == dict(features=1, mtf=1, candles=1, funding=1, metrics=1, state=1)
    evaluator.evaluate("ETHUSDT", "1h", "trend_follow")
    assert len(evaluator._features) == 1
    assert calls["features"] == 2
    # A new cycle reloads the data rather than retaining the previous cycle's snapshot.
    engine.StrategyEngine(db, cycle_snapshot=True).evaluate("BTCUSDT", "1h", "trend_follow")
    assert calls["features"] == 3


def test_default_engine_refreshes_data_and_skips_unused_mtf(db, monkeypatch, candle_frame):
    calls = []
    original = engine.build_features

    def build(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "build_features", build)
    monkeypatch.setattr(engine, "add_mtf_trend", lambda *a, **k: pytest.fail("unused MTF calculation"))
    evaluator = engine.StrategyEngine(db)
    first = evaluator.evaluate("BTCUSDT", "1h", "trend_follow")
    candle_frame.iloc[-1, candle_frame.columns.get_loc("close")] += .01
    second = evaluator.evaluate("BTCUSDT", "1h", "trend_follow")
    assert len(calls) == 2
    assert first.price != second.price


@pytest.mark.parametrize("value", [0, 9, True, 1.5])
def test_training_worker_setting_rejects_invalid_values(value):
    with pytest.raises(ValidationError):
        SettingsUpdate(auto_train_workers=value)


def test_training_worker_setting_accepts_and_validates_bounds():
    for value in (1, 2, 8):
        payload = SettingsUpdate(auto_train_workers=value).model_dump(exclude_none=True)
        _validate(payload)
        assert payload["auto_train_workers"] == value
