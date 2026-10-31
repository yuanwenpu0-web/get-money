"""Tests for the ported Harmonic Pattern Recognition catalog indicator."""

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.harmonic_patterns import (
    harmonic_patterns,
    _validate_bat,
    _validate_butterfly,
    _validate_crab,
    _validate_shark,
    _validate_ratio,
    _calculate_ratio,
)


def test_ratio_helpers():
    assert _validate_ratio(0.40, 0.382, 0.05) is True
    assert _validate_ratio(0.50, 0.382, 0.05) is False
    assert _validate_ratio(0.40, 0.382, 0.02) is True   # |0.018| <= 0.02
    assert _validate_ratio(0.45, 0.382, 0.02) is False  # |0.068| > 0.02
    assert math_nan(_calculate_ratio(0.0, 5.0))
    assert _calculate_ratio(10.0, 4.0) == pytest.approx(0.4)


def math_nan(v):
    return isinstance(v, float) and np.isnan(v)


def test_bat_validator():
    ok, d = _validate_bat(100.0, 110.0, 114.0, 116.0, 0.05)
    assert ok is True
    assert d == pytest.approx(110.0 - 10.0 * 0.886)
    # inverted BC leg must invalidate
    ok2, _ = _validate_bat(100.0, 110.0, 114.0, 108.0, 0.05)
    assert ok2 is False


def test_butterfly_validator():
    ok, d = _validate_butterfly(100.0, 110.0, 117.86, 123.0, 0.05)
    assert ok is True
    assert d == pytest.approx(110.0 - 10.0 * 1.272)


def test_crab_validator():
    ok, d = _validate_crab(100.0, 110.0, 114.0, 116.0, 0.05)
    assert ok is True
    assert d == pytest.approx(110.0 - 10.0 * 1.618)


def test_shark_validator():
    ok, d = _validate_shark(100.0, 110.0, 115.0, 121.5, 0.05)
    assert ok is True
    assert d == pytest.approx(110.0 - 10.0 * 0.886)


def test_columns_and_dtypes():
    rng = np.random.default_rng(0)
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c}, index=idx)
    out = harmonic_patterns(df)
    assert out.index.equals(idx)
    assert len(out.columns) == len(set(out.columns))
    assert str(out["harmonic_valid"].dtype) == "Int64"
    assert str(out["harmonic_code"].dtype) == "Int64"
    assert set(out["harmonic_code"].unique().tolist()).issubset({0, 1, 2, 3, 4})
    assert set(out["harmonic_direction"].unique().tolist()).issubset({-1, 0, 1})
    assert out["harmonic_quality"].between(0.0, 1.0).all()
    assert (out["harmonic_pivot_count"].astype(int) >= 0).all()


def test_parameter_validation():
    df = pd.DataFrame({"close": np.linspace(100, 110, 60)})
    assert harmonic_patterns(df).shape[0] == 60  # minimal frame still runs
    bad = [
        dict(lookback=10),
        dict(tolerance=0.001),
        dict(min_quality_for_alert=1.5),
        dict(mode="Nope"),
        dict(enable_bat="yes"),
        dict(risk_percent=2.0),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            harmonic_patterns(df, **kwargs)


def test_missing_close_raises():
    df = pd.DataFrame({"open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3]})
    with pytest.raises(ValueError):
        harmonic_patterns(df)
