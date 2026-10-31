"""Tests for the ported Market Scenario Projector catalog indicator."""

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.market_scenario_projector import market_scenario_projector


def _random_frame(seed=0, n=280):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"open": c + rng.normal(0, 0.1, n),
                         "high": c + np.abs(rng.normal(0, 0.5, n)),
                         "low": c - np.abs(rng.normal(0, 0.5, n)),
                         "close": c}, index=idx)


def _trend_frame(n=200, slope=0.5):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = 100 + np.arange(n) * slope
    return pd.DataFrame({"open": c, "high": c + 0.3, "low": c - 0.3, "close": c}, index=idx)


def test_columns_and_weights_sum_100():
    out = market_scenario_projector(_random_frame())
    assert out.index.equals(_random_frame().index)
    comb = out["msp_direct_weight"] + out["msp_pullback_weight"] + out["msp_failure_weight"]
    # only defined once total score > 0 (past warm-up)
    tail = comb.iloc[80:]
    assert np.allclose(tail, 100.0, atol=1e-6)
    assert out["msp_setup_quality"].between(0.0, 100.0).all()
    for col in ["msp_direct_score", "msp_pullback_score", "msp_failure_score",
                "msp_proximity", "msp_rejection", "msp_barrier_quality"]:
        assert out[col].between(0.0, 1.0 + 1e-9).all(), col


def test_trend_direction():
    up = market_scenario_projector(_trend_frame(slope=0.5))
    down = market_scenario_projector(_trend_frame(slope=-0.5))
    assert int(up["msp_direction"].iloc[-1]) == 1
    assert int(down["msp_direction"].iloc[-1]) == -1
    assert int(up["msp_impulse"].max() * 100) > 0


def test_valid_flag_and_dtypes():
    out = market_scenario_projector(_random_frame(seed=2))
    assert str(out["msp_valid"].dtype) == "Int64"
    assert set(out["msp_valid"].unique().tolist()).issubset({0, 1})
    assert int(out["msp_valid"].sum()) > 0


def test_parameter_validation():
    df = _random_frame(n=120)
    bad = [
        dict(impulse_length=2),
        dict(atr_length=4),
        dict(analysis_lookback=10),
        dict(max_stored_pivots=5),
        dict(minimum_setup_score=101.0),
        dict(max_level_distance_atr=0.1),
        dict(level_zone_atr=0.01),
        dict(rejection_zone_atr=0.0),
        dict(impulse_retracement_ratio=0.0),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            market_scenario_projector(df, **kwargs)


def test_random_end_to_end():
    out = market_scenario_projector(_random_frame(seed=5, n=240))
    assert len(out) == 240
    assert len(out.columns) == len(set(out.columns))
