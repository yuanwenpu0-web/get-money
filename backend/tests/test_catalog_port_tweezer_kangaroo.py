"""Tests for the ported Tweezer & Kangaroo Zones catalog indicator."""

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.tweezer_and_kangaroo_zones import tweezer_and_kangaroo_zones


def _frame(seed=0, n=400):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    o = c + rng.normal(0, 0.2, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.4, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.4, n))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": 1000 + np.abs(rng.normal(0, 200, n))}, index=idx)


def test_patterns_detected():
    out = tweezer_and_kangaroo_zones(_frame())
    assert int(out["tkz_tweezer_top"].sum()) > 0
    assert int(out["tkz_tweezer_bot"].sum()) > 0
    assert int(out["tkz_kangaroo_bear"].sum()) > 0
    assert int(out["tkz_kangaroo_bull"].sum()) > 0


def test_zones_and_scores():
    out = tweezer_and_kangaroo_zones(_frame(seed=1))
    assert int(out["tkz_zone_count"].max()) > 0
    assert out["tkz_supply_score"].dropna().between(0.0, 100.0).all()
    assert out["tkz_demand_score"].dropna().between(0.0, 100.0).all()


def test_columns_dtypes():
    out = tweezer_and_kangaroo_zones(_frame())
    assert out.index.equals(_frame().index)
    assert len(out.columns) == len(set(out.columns))
    for col in ["tkz_tweezer_top", "tkz_tweezer_bot", "tkz_kangaroo_bear",
                "tkz_kangaroo_bull", "tkz_zone_count", "tkz_supply_grade_a", "tkz_demand_grade_a"]:
        assert str(out[col].dtype) == "Int64", col
        assert set(out[col].dropna().unique().tolist()).issubset({0, 1}) or col == "tkz_zone_count"


def test_parameter_validation():
    df = _frame(n=120)
    bad = [
        dict(tweezer_lookback=0),
        dict(kt_body_frac_max=1.5),
        dict(kt_body_position=1.5),
        dict(zone_ttl=0),
        dict(struct_lookback=2),
        dict(grade_a_min=40.0),
        dict(zone_overlap_pct=1.5),
        dict(tf_sensitivity=0.1),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            tweezer_and_kangaroo_zones(df, **kwargs)


def test_random_end_to_end():
    out = tweezer_and_kangaroo_zones(_frame(seed=3, n=260))
    assert len(out) == 260
    assert (out["tkz_zone_count"].astype(int) >= 0).all()
