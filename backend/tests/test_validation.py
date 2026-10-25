import json

import numpy as np
import pandas as pd
import pytest
from app.api.strategies import update_strategy
from app.backtest import runner
from app.features import indicators
from app.ml import dataset
from app.models import StrategyConfig
from app.schemas import BacktestRequest, StrategyUpdate
from app.utils import interval_ms
from fastapi import HTTPException
from pydantic import ValidationError


@pytest.mark.parametrize(
    "changes",
    [
        {"fee": -0.01},
        {"initial_cash": 0},
        {"threshold": float("nan")},
        {"strategy": "typo"},
        {"interval": "2x"},
        {"days": 0},
    ],
)
def test_backtest_rejects_invalid_parameters(changes):
    with pytest.raises(ValidationError):
        BacktestRequest(symbol="BTCUSDT", **changes)


def test_strategy_rejects_invalid_json_values_before_mutation(db):
    row = StrategyConfig(code="trend_follow", name="test", params="{}", enabled=1)
    db.add(row)
    db.commit()
    with pytest.raises(HTTPException):
        update_strategy(
            "trend_follow",
            StrategyUpdate(enabled=False, params={"risk_per_trade_pct": -1}),
            db,
        )
    assert row.enabled == 1
    assert row.params == "{}"


def prices(n=1200):
    index = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    close = 100 + 0.01 * np.arange(n) + 0.8 * np.sin(np.arange(n) / 20)
    return pd.DataFrame(
        dict(open=close, high=close + 0.1, low=close - 0.1, close=close, volume=100.0),
        index=index,
    )


def test_pandas_indicator_fallback_is_finite_and_causal(monkeypatch):
    monkeypatch.setattr(indicators, "HAS_TALIB", False)
    monkeypatch.setattr(indicators, "HAS_PTA", False)
    frame = prices(200)
    original = indicators.build_features(frame)
    mutated = frame.copy()
    mutated.iloc[150:, :4] *= 2
    changed = indicators.build_features(mutated)
    pd.testing.assert_frame_equal(original.iloc[:150], changed.iloc[:150])
    assert np.isfinite(original[indicators.FEATURES].iloc[60:]).all().all()
    monotonic = pd.Series(np.arange(100.0))
    assert indicators._rsi(monotonic, 14).iloc[-1] == 100.0


def test_dataset_excludes_unclosed_candles_and_resets_at_gaps(monkeypatch):
    frame = prices(300)
    frame = frame.drop(frame.index[100:110])
    now = int(frame.index[-1].timestamp() * 1000) + 30000
    monkeypatch.setattr(dataset, "now_ms", lambda: now)
    monkeypatch.setattr(dataset.MarketDataService, "load_frame", lambda *a, **kw: frame)
    result = dataset.build_dataset(None, "BTCUSDT", "1m", days=30)
    assert result.index[-1] < frame.index[-1]
    assert result.attrs["data_quality"]["missing_bars"] == 10
    assert not result.index.isin(
        frame.index[
            (frame.index >= prices(300).index[110])
            & (frame.index < prices(300).index[159])
        ]
    ).any()


def test_test_partition_cannot_influence_training_or_parameter_selection(
    db, monkeypatch
):
    data = indicators.build_features(prices(), with_label=True).dropna()
    fits = []

    class Estimator:
        def fit(self, features, labels):
            fits.append((features.copy(), labels.copy()))

        def predict_proba(self, features):
            return np.tile([0.3, 0.7], (len(features), 1))

    monkeypatch.setattr(runner, "_build_estimator", lambda algo: Estimator())
    monkeypatch.setattr(runner, "build_dataset", lambda *a, **kw: data)
    first = runner.run_backtest(db, "BTCUSDT", "1m", "ml_ensemble", optimize=True)
    split = int(len(data) * 0.75)
    data.iloc[split:, data.columns.get_indexer(["open", "high", "low", "close"])] *= 2
    data.iloc[split:, data.columns.get_loc("label")] = 1
    second = runner.run_backtest(db, "BTCUSDT", "1m", "ml_ensemble", optimize=True)
    assert first["metrics"]["validation"] == second["metrics"]["validation"]
    pd.testing.assert_frame_equal(fits[0][0], fits[1][0])
    pd.testing.assert_series_equal(fits[0][1], fits[1][1])
    assert fits[0][0].index[-1] < data.index[int(len(data) * 0.6) - 5]
    json.dumps(second, allow_nan=False)


def test_supported_long_intervals_have_correct_duration():
    assert interval_ms("4h") == 14_400_000
    assert interval_ms("1d") == 86_400_000
    with pytest.raises(ValueError):
        interval_ms("unknown")


def test_background_job_passes_selection_and_preserves_json_result(db, monkeypatch):
    from app.api import backtests
    from app.models import BacktestRun

    run = BacktestRun(
        symbol="BTCUSDT",
        interval="1m",
        strategy="ml_ensemble",
        algo="rule",
        status="running",
        params=json.dumps({"optimize": True, "fee": 0.0}),
    )
    db.add(run)
    db.commit()
    monkeypatch.setattr(backtests, "SessionLocal", lambda: db)
    captured = {}

    def backtest(*args, **kwargs):
        captured.update(kwargs)
        return {
            "metrics": {"profit_factor": None, "validation": {"enabled": True}},
            "equity_curve": [],
            "trades": [],
        }

    monkeypatch.setattr(backtests, "run_backtest", backtest)
    backtests._execute(run.id)
    saved = db.get(BacktestRun, run.id)
    assert captured["optimize"] is True
    assert captured["fee"] == 0.0
    assert saved.status == "done"
    assert json.loads(saved.metrics)["profit_factor"] is None
