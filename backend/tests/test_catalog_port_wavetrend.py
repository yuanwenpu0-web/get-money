"""Tests for the ported WaveTrend v3 catalog indicator."""

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.wavetrend_v3 import wavetrend_v3


def _frame(seed=0, n=300):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"open": c, "high": c + np.abs(rng.normal(0, 0.5, n)),
                         "low": c - np.abs(rng.normal(0, 0.5, n)), "close": c,
                         "volume": 1000 + np.abs(rng.normal(0, 200, n))}, index=idx)


def test_signal_plumbing_when_gate_off():
    out = wavetrend_v3(_frame(), use_setup_gate=False, profile_mode="Diagnostic")
    assert int(out["wt3_cross_up"].sum()) > 0
    assert int(out["wt3_long_signal"].sum()) == int(out["wt3_cross_up"].sum())
    assert int(out["wt3_short_signal"].sum()) == int(out["wt3_cross_down"].sum())


def test_columns_dtypes_ranges():
    out = wavetrend_v3(_frame())
    assert out.index.equals(_frame().index)
    assert len(out.columns) == len(set(out.columns))
    for col in ["wt3_long_signal", "wt3_short_signal", "wt3_structure_bias", "wt3_long_family"]:
        assert str(out[col].dtype) == "Int64", col
    assert out["wt3_structure_bias"].dropna().isin([-1, 0, 1]).all()
    assert out["wt3_long_ctx"].between(0.0, 8.0).all()
    assert out["wt3_short_ctx"].between(0.0, 8.0).all()
    assert out["wt3_osc"].iloc[40:].notna().all()


def test_poc_history_scaling():
    short = wavetrend_v3(_frame(n=200))
    assert short["wt3_poc"].isna().all()  # needs 400 bars after 1h auto-scaling
    long_ = wavetrend_v3(_frame(n=800))
    assert long_["wt3_poc"].notna().sum() > 0


def test_parameter_validation():
    df = _frame(n=120)
    bad = [
        dict(ch_length=1),
        dict(min_context_score=8.0),
        dict(ci_const=0.0001),
        dict(setup_edge_pct=0.05),
        dict(profile_mode="Nope"),
        dict(anchor_score_mode="Nope"),
        dict(volume_gate_mode="Nope"),
        dict(struct_piv_len=1),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            wavetrend_v3(df, **kwargs)


def test_random_end_to_end():
    out = wavetrend_v3(_frame(seed=4, n=260))
    assert len(out) == 260
    assert out["wt3_long_signal"].isin([0, 1]).all()
