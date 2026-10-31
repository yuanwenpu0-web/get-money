"""Tests for the ported Modern Trend Regime catalog indicator."""

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.modern_trend_regime import modern_trend_regime


def _random_frame(seed=0, n=400):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"open": c, "high": c + np.abs(rng.normal(0, 0.5, n)),
                         "low": c - np.abs(rng.normal(0, 0.5, n)), "close": c}, index=idx)


def _trend_frame(n=400, slope=0.5):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = 100 + np.arange(n) * slope
    return pd.DataFrame({"open": c, "high": c + 0.3, "low": c - 0.3, "close": c}, index=idx)


def _range_breakout_frame(n=260):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = np.empty(n)
    c[:160] = 100 + 2 * np.sin(np.arange(160) / 4.0)
    c[160:] = 100 + np.arange(n - 160) * 0.15
    return pd.DataFrame({"open": c, "high": c + 0.6, "low": c - 0.6, "close": c}, index=idx)


def test_columns_dtypes_regimes():
    out = modern_trend_regime(_random_frame())
    assert out.index.equals(_random_frame().index)
    assert len(out.columns) == len(set(out.columns))
    assert str(out["mtr_regime"].dtype) == "Int64"
    assert set(out["mtr_regime"].dropna().unique().tolist()).issubset({-1, 0, 1, 2})
    for col in ["mtr_bull_score", "mtr_bear_score", "mtr_range_score"]:
        assert out[col].between(0, 5).all()
    assert out["mtr_adx"].dropna().between(0, 100).all()


def test_uptrend_is_bull():
    out = modern_trend_regime(_trend_frame())
    assert int(out["mtr_regime"].iloc[-1]) == 1
    assert int(out["mtr_bull_flip"].sum()) >= 1


def test_range_and_breakout():
    out = modern_trend_regime(_range_breakout_frame(), preset="Range Trading")
    assert 2 in set(out["mtr_regime"].dropna().unique().tolist())
    assert (out["mtr_breakout"] == 1).any()


def test_parameter_validation():
    df = _random_frame(n=120)
    bad = [
        dict(preset="Nope"),
        dict(calculation_mode="Nope"),
        dict(custom_fast_length=1),
        dict(custom_regime_confirmation_bars=0),
        dict(custom_range_adx_threshold=1.0),
        dict(custom_range_efficiency_threshold=0.9),
        dict(custom_breakout_validity_bars=0),
        dict(range_reaction_close_position=0.4),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            modern_trend_regime(df, **kwargs)


def test_random_end_to_end():
    out = modern_trend_regime(_random_frame(seed=3, n=260))
    assert len(out) == 260
    assert out["mtr_regime"].isin([-1, 0, 1, 2]).all()
