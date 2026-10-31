"""Tests for the ported Support & Resistance Zones catalog indicator."""

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.support_and_resistance_zones import support_and_resistance_zones


def _gap_frame(n=60):
    o = np.full(n, 100.0)
    c = np.full(n, 100.0)
    h = np.full(n, 102.0)
    l = np.full(n, 98.0)
    o[20], c[20], h[20], l[20] = 100.0, 106.0, 106.5, 105.5
    for k in range(21, n):
        o[k], c[k], h[k], l[k] = 106.0, 106.0, 106.5, 105.5
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": np.full(n, 1000.0)}, index=idx)


def _crash_frame(n=80):
    o = np.full(n, 100.0)
    c = np.full(n, 100.0)
    h = np.full(n, 102.0)
    l = np.full(n, 98.0)
    o[20], c[20], h[20], l[20] = 100.0, 106.0, 106.5, 105.5
    for k in range(21, 40):
        o[k], c[k], h[k], l[k] = 106.0, 106.0, 106.5, 105.5
    for k in range(40, 50):
        o[k], c[k], h[k], l[k] = 95.0, 90.0, 95.5, 89.5
    for k in range(50, n):
        o[k], c[k], h[k], l[k] = 90.0, 90.0, 90.5, 89.5
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": np.full(n, 1000.0)}, index=idx)


def test_gap_creates_support_zone():
    out = support_and_resistance_zones(_gap_frame())
    assert out.index.equals(_gap_frame().index)
    assert int(out["srz_stat_created"].max()) >= 1
    assert out["srz_support_top"].notna().any()
    # merged detectors must have reinforced the zone beyond a single source weight
    assert float(out["srz_support_score"].dropna().max()) >= 18.0


def test_support_break_flips_to_resistance():
    out = support_and_resistance_zones(_crash_frame())
    assert int(out["srz_break_event"].sum()) >= 1
    assert int(out["srz_flip_event"].sum()) >= 1
    assert int(out["srz_flipped_count"].max()) >= 1


def test_counts_are_monotone_consistent():
    out = support_and_resistance_zones(_gap_frame())
    active = out["srz_active_count"].astype(int)
    confirmed = out["srz_confirmed_count"].astype(int)
    weak = out["srz_weak_count"].astype(int)
    flipped = out["srz_flipped_count"].astype(int)
    assert (active >= 0).all()
    assert (confirmed + weak + flipped <= active).all()


def test_flag_columns_int64():
    out = support_and_resistance_zones(_gap_frame())
    int_cols = [c for c in out.columns if c.startswith("srz_")
                and ("count" in c or "event" in c or "stat" in c or "reaction" in c)]
    assert int_cols
    for col in int_cols:
        assert str(out[col].dtype) == "Int64", col


def test_parameter_validation():
    df = _gap_frame()
    bad = [
        dict(atr_length=1),
        dict(pivot_depth=0),
        dict(base_bars=1),
        dict(min_zone_score=101.0),
        dict(max_zone_width_atr=0.05),
        dict(psychological_step=0.0),
        dict(lookback_bars=50),
        dict(reaction_horizon=0),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            support_and_resistance_zones(df, **kwargs)


def test_random_runs_end_to_end():
    rng = np.random.default_rng(2)
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({"open": c + rng.normal(0, 0.1, n),
                       "high": c + np.abs(rng.normal(0, 0.5, n)),
                       "low": c - np.abs(rng.normal(0, 0.5, n)),
                       "close": c,
                       "volume": 1000 + np.abs(rng.normal(0, 100, n))}, index=idx)
    out = support_and_resistance_zones(df)
    assert out.index.equals(idx)
    assert len(out.columns) == len(set(out.columns))
    assert int(out["srz_stat_created"].max()) >= 1
    # cumulative counters never decrease
    assert out["srz_stat_created"].astype(int).diff().fillna(0).ge(0).all()
