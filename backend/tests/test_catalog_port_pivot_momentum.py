"""Tests for the ported Pivot Momentum Structure catalog indicator."""

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.pivot_momentum_structure import pivot_momentum_structure


def _frame(seed=0, n=400):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"open": c, "high": c + np.abs(rng.normal(0, 0.6, n)),
                         "low": c - np.abs(rng.normal(0, 0.6, n)), "close": c,
                         "volume": 1000 + np.abs(rng.normal(0, 100, n))}, index=idx)


def test_columns_dtypes():
    out = pivot_momentum_structure(_frame())
    assert out.index.equals(_frame().index)
    assert len(out.columns) == len(set(out.columns))
    for col in ["pms_structure", "pms_divergence", "pms_regular_bull_div",
                "pms_hidden_bear_div", "pms_momentum_bull"]:
        assert str(out[col].dtype) == "Int64", col
    assert set(out["pms_structure"].dropna().unique().tolist()).issubset({0, 1, 2, 3, 4})
    assert set(out["pms_divergence"].dropna().unique().tolist()).issubset({-1, 0, 1})


def test_all_oscillators_run():
    for osc in ["RSI", "WaveTrend", "Stochastic RSI", "CCI", "Fisher", "TSI",
                "Williams %R", "CMO", "MFI"]:
        out = pivot_momentum_structure(_frame(n=220), oscillator_type=osc)
        assert out["pms_osc"].notna().any(), osc


def test_divergence_and_alignment():
    out = pivot_momentum_structure(_frame(seed=1), oscillator_type="Stochastic RSI")
    assert out["pms_divergence"].isin([-1, 0, 1]).all()
    assert int(out["pms_momentum_bull"].sum()) >= 0
    assert int(out["pms_momentum_bear"].sum()) >= 0


def test_parameter_validation():
    df = _frame(n=160)
    bad = [
        dict(oscillator_type="Nope"),
        dict(structure_preset="Nope"),
        dict(rsi_length=1),
        dict(cci_length=1),
        dict(atr_length=0),
        dict(structure_preset="Custom", pivot_left=0),
        dict(structure_preset="Custom", maximum_pivot_distance=1),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            pivot_momentum_structure(df, **kwargs)


def test_random_end_to_end():
    out = pivot_momentum_structure(_frame(seed=3, n=260))
    assert len(out) == 260
    assert out["pms_osc"].notna().any()
