"""Tests for the ported Triangle Compression Scanner Pro catalog indicator."""

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.triangle_compression_scanner_pro import (
    triangle_compression_scanner_pro,
)


def _build(verts, n=200):
    xs = np.arange(n)
    mid = np.zeros(n)
    for k in range(len(verts) - 1):
        (x0, y0), (x1, y1) = verts[k], verts[k + 1]
        seg = (xs >= x0) & (xs < x1)
        mid[seg] = y0 + (y1 - y0) * (xs[seg] - x0) / (x1 - x0)
    mid[xs >= verts[-1][0]] = verts[-1][1]
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({"open": mid, "high": mid + 0.3, "low": mid - 0.3, "close": mid}, index=idx)


_SYMMETRIC = [(10, 110), (20, 90), (30, 107), (40, 92), (50, 104), (60, 94), (70, 102), (80, 96), (90, 100)]
_ASCENDING = [(10, 108), (20, 88), (30, 108), (40, 91), (50, 108), (60, 94), (70, 108), (80, 97), (90, 108)]
_DESCENDING = [(10, 112), (20, 92), (30, 109), (40, 92), (50, 106), (60, 92), (70, 103), (80, 92), (90, 100)]


def test_symmetric_triangle_detected():
    out = triangle_compression_scanner_pro(_build(_SYMMETRIC))
    assert 1 in set(out["tcs_best_type"].dropna().tolist())
    assert int(out["tcs_best_score"].max()) >= 60
    assert int(out["tcs_best_valid"].sum()) > 0
    assert int(out["tcs_tracked_score"].max()) > 0
    # projected compression must be a real contraction
    proj = out["tcs_best_projected_compression"].replace(0, np.nan).dropna()
    assert proj.min() < 0.75


def test_ascending_triangle_classified():
    out = triangle_compression_scanner_pro(_build(_ASCENDING))
    types = set(out["tcs_best_type"].dropna().tolist())
    assert 2 in types, types
    assert int(out["tcs_best_valid"].sum()) > 0


def test_descending_triangle_classified():
    out = triangle_compression_scanner_pro(_build(_DESCENDING))
    types = set(out["tcs_best_type"].dropna().tolist())
    assert 3 in types, types
    assert int(out["tcs_best_valid"].sum()) > 0


def test_bull_breakout_resolved():
    out = triangle_compression_scanner_pro(_build(_SYMMETRIC))
    assert (out["tcs_breakout"] == 1).any()
    assert int(out["tcs_active_count"].max()) >= 1


def test_flag_columns_are_int64_and_ranged():
    out = triangle_compression_scanner_pro(_build(_SYMMETRIC))
    for col in ["tcs_best_type", "tcs_best_valid", "tcs_best_watch",
                "tcs_tracked_type", "tcs_active_count", "tcs_breakout"]:
        assert str(out[col].dtype) == "Int64", col
    assert set(out["tcs_best_type"].dropna().unique().tolist()).issubset({0, 1, 2, 3})
    assert set(out["tcs_tracked_type"].dropna().unique().tolist()).issubset({0, 1, 2, 3})
    assert set(out["tcs_best_valid"].dropna().unique().tolist()).issubset({0, 1})
    assert set(out["tcs_best_watch"].dropna().unique().tolist()).issubset({0, 1})
    assert set(out["tcs_breakout"].dropna().unique().tolist()).issubset({-1, 0, 1})
    assert int(out["tcs_active_count"].min()) >= 0


def test_parameter_validation():
    df = _build(_SYMMETRIC)
    bad = [
        dict(pivot_len=1),
        dict(triangle_window=5),
        dict(search_depth=5),
        dict(min_width_atr=0.1),
        dict(projected_compression_target=0.99),
        dict(max_apex_distance_mult=0.1),
        dict(min_traversal_ratio=0.2),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            triangle_compression_scanner_pro(df, **kwargs)


def test_random_runs_end_to_end():
    rng = np.random.default_rng(7)
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = 100 + np.cumsum(rng.normal(0, 0.8, n))
    df = pd.DataFrame({"open": close, "high": close + np.abs(rng.normal(0, 0.6, n)),
                       "low": close - np.abs(rng.normal(0, 0.6, n)), "close": close}, index=idx)
    out = triangle_compression_scanner_pro(df)
    assert out.index.equals(idx)
    assert len(out.columns) == len(set(out.columns))
    assert out["tcs_best_type"].dropna().isin([0, 1, 2, 3]).all()
