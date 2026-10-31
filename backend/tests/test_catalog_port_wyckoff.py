"""Tests for the ported Wyckoff Schematics catalog indicator."""

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.wyckoff_schematics import wyckoff_schematics


def _climax_frame(direction="down", n=220, cb=90):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = np.full(n, np.nan); o = np.full(n, np.nan); h = np.full(n, np.nan); l = np.full(n, np.nan)
    if direction == "down":
        for k in range(0, cb + 1):
            c[k] = 120 - 20 * k / cb
        o[cb], c[cb], h[cb], l[cb] = 100.0, 95.0, 100.5, 94.5
        for k in range(cb + 1, n):
            c[k] = 95.0
    else:
        for k in range(0, cb + 1):
            c[k] = 100 + 20 * k / cb
        o[cb], c[cb], h[cb], l[cb] = 120.0, 125.0, 125.5, 119.5
        for k in range(cb + 1, n):
            c[k] = 125.0
    for k in range(n):
        if np.isnan(o[k]):
            o[k], h[k], l[k] = c[k] + 0.2, c[k] + 0.4, c[k] - 0.4
    h[cb], l[cb] = (100.5, 94.5) if direction == "down" else (125.5, 119.5)
    vol = np.full(n, 1000.0)
    vol[cb] = 6000.0
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": vol}, index=idx)


def test_selling_climax_accumulation():
    out = wyckoff_schematics(_climax_frame("down"))
    assert out["wyk_sc"].iloc[90] == 1
    assert int(out["wyk_phase"].iloc[90]) == 1  # ACCUMULATION
    assert int(out["wyk_phase_ae"].iloc[90]) == 1  # phase A
    assert int(out["wyk_range_score"].max()) > 0


def test_buying_climax_distribution():
    out = wyckoff_schematics(_climax_frame("up"))
    assert out["wyk_bc"].iloc[90] == 1
    assert int(out["wyk_phase"].iloc[90]) == 2  # DISTRIBUTION


def test_columns_and_dtypes():
    out = wyckoff_schematics(_climax_frame("down"))
    assert out.index.equals(_climax_frame("down").index)
    assert len(out.columns) == len(set(out.columns))
    for col in ["wyk_phase", "wyk_phase_ae", "wyk_schematic", "wyk_direction",
                "wyk_setup_score", "wyk_range_score"]:
        assert str(out[col].dtype) == "Int64", col
    for col in ["wyk_sc", "wyk_bc", "wyk_ar", "wyk_st", "wyk_spring", "wyk_test",
                "wyk_sos", "wyk_sow", "wyk_no_supply", "wyk_vdu"]:
        assert str(out[col].dtype) == "Int64", col
        assert set(out[col].dropna().unique().tolist()).issubset({0, 1})
    assert out["wyk_phase"].dropna().isin([0, 1, 2, 3, 4]).all()
    assert out["wyk_direction"].dropna().isin([-1, 0, 1]).all()


def test_parameter_validation():
    df = _climax_frame("down")
    bad = [
        dict(volume_lookback=5),
        dict(range_establish_bars=2),
        dict(pivot_len=1),
        dict(event_cooldown_bars=0),
        dict(markup_min_score=101),
        dict(climax_multiplier=1.0),
        dict(spring_threshold=0.1),
        dict(sos_threshold=40.0),
        dict(invalidation_factor=0.1),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            wyckoff_schematics(df, **kwargs)


def test_random_runs_end_to_end():
    rng = np.random.default_rng(3)
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({"open": c + rng.normal(0, 0.15, n),
                       "high": c + np.abs(rng.normal(0, 0.6, n)),
                       "low": c - np.abs(rng.normal(0, 0.6, n)),
                       "close": c,
                       "volume": 1000 + np.abs(rng.normal(0, 200, n))}, index=idx)
    out = wyckoff_schematics(df)
    assert out.index.equals(idx)
    assert out["wyk_phase"].dropna().isin([0, 1, 2, 3, 4]).all()
