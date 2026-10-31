"""Tests for the ported ZigZag Core catalog indicator."""

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.zigzag_core import zigzag_core


def _frame(seed=0, n=400):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"open": c, "high": c + np.abs(rng.normal(0, 0.6, n)),
                         "low": c - np.abs(rng.normal(0, 0.6, n)), "close": c,
                         "volume": 1000.0}, index=idx)


def _trend_frame(n=300, slope=0.35):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = 100 + np.arange(n) * slope + 6.0 * np.sin(np.arange(n) / 5.0)
    return pd.DataFrame({"open": c, "high": c + 0.4, "low": c - 0.4, "close": c,
                         "volume": 1000.0}, index=idx)


def test_columns_dtypes():
    out = zigzag_core(_frame())
    assert out.index.equals(_frame().index)
    assert len(out.columns) == len(set(out.columns))
    for col in ["zz_dir", "zz_new_pivot", "zz_structure", "zz_confirmed_count"]:
        assert str(out[col].dtype) == "Int64", col
    assert set(out["zz_dir"].dropna().unique().tolist()).issubset({-1, 0, 1})
    assert set(out["zz_new_pivot"].dropna().unique().tolist()).issubset({-1, 0, 1})


def test_zigzag_produces_pivots():
    out = zigzag_core(_frame(seed=1))
    assert int((out["zz_new_pivot"] != 0).sum()) > 0
    assert int(out["zz_confirmed_count"].max()) > 0
    assert set(out["zz_structure"].dropna().unique().tolist()).issubset({0, 1, 2, 3, 4})


def test_uptrend_ends_up():
    out = zigzag_core(_trend_frame())
    assert 1 in set(out["zz_dir"].dropna().unique().tolist())
    assert int(out["zz_confirmed_count"].max()) > 0


def test_l2_and_secondary_run():
    out = zigzag_core(_frame(seed=2), show_l2=True, show_secondary=True)
    assert int(out["zz_l2_confirmed_count"].max()) >= 0
    assert int(out["zz_secondary_count"].max()) > 0


def test_parameter_validation():
    df = _frame(n=140)
    bad = [
        dict(pivot_depth=0),
        dict(backstep_bars=-1),
        dict(filter_mode="Nope"),
        dict(atr_length=0),
        dict(deviation_percent=101.0),
        dict(atr_multiplier=21.0),
        dict(l2_deviation_multiplier=0.5),
        dict(secondary_pivot_depth=0),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            zigzag_core(df, **kwargs)


def test_random_end_to_end():
    out = zigzag_core(_frame(seed=3, n=260))
    assert len(out) == 260
    assert out["zz_dir"].isin([-1, 0, 1]).all()
