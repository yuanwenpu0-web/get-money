"""Tests for the ported MTF WaveTrend Opportunity Hunter catalog indicator."""

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.mtf_wavetrend_opportunity_hunter import mtf_wavetrend_opportunity_hunter


def _frame(seed=0, n=400):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"open": c + rng.normal(0, 0.1, n),
                         "high": c + np.abs(rng.normal(0, 0.5, n)),
                         "low": c - np.abs(rng.normal(0, 0.5, n)), "close": c}, index=idx)


def test_columns_dtypes():
    out = mtf_wavetrend_opportunity_hunter(_frame())
    assert out.index.equals(_frame().index)
    assert len(out.columns) == len(set(out.columns))
    for col in ["mwoh_entry_long", "mwoh_entry_short", "mwoh_in_long", "mwoh_in_short",
                "mwoh_tradeable_long", "mwoh_tradeable_short"]:
        assert str(out[col].dtype) == "Int64", col
        assert set(out[col].dropna().unique().tolist()).issubset({0, 1})


def test_scores_and_entries():
    out = mtf_wavetrend_opportunity_hunter(_frame(seed=1))
    assert int(out["mwoh_entry_long"].sum()) > 0
    assert int(out["mwoh_entry_short"].sum()) > 0
    assert out["mwoh_final_long"].dropna().between(0.0, 100.0).all()
    assert out["mwoh_final_short"].dropna().between(0.0, 100.0).all()
    assert out["mwoh_regime_long"].between(0.0, 100.0).all()


def test_position_state_never_both():
    out = mtf_wavetrend_opportunity_hunter(_frame(seed=2))
    both = (out["mwoh_in_long"].astype(int) + out["mwoh_in_short"].astype(int))
    assert (both <= 1).all()


def test_parameter_validation():
    df = _frame(n=160)
    bad = [
        dict(wt_ch_len=0),
        dict(er_len=1),
        dict(entropy_len=5),
        dict(entropy_rank_len=10),
        dict(cooldown_bars=0),
        dict(wt_extreme=5.0),
        dict(min_regime=101.0),
        dict(jma_core_blend=1.5),
        dict(wt_cross_zone=50.0),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            mtf_wavetrend_opportunity_hunter(df, **kwargs)


def test_random_end_to_end():
    out = mtf_wavetrend_opportunity_hunter(_frame(seed=3, n=300))
    assert len(out) == 300
    assert out["mwoh_wt1"].iloc[100:].notna().any()
