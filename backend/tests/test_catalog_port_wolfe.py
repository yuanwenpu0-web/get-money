"""Tests for the ported Wolfe Wave Scanner Pro catalog indicator."""

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.wolfe_wave_scanner_pro import wolfe_wave_scanner_pro

_BULL_VERTS = [(0, 110), (10, 98), (20, 120), (35, 96), (50, 112), (65, 93), (85, 108)]


def _build(verts, n=120):
    xs = np.arange(n)
    mid = np.zeros(n)
    for k in range(len(verts) - 1):
        (x0, y0), (x1, y1) = verts[k], verts[k + 1]
        seg = (xs >= x0) & (xs < x1)
        mid[seg] = y0 + (y1 - y0) * (xs[seg] - x0) / (x1 - x0)
    mid[xs >= verts[-1][0]] = verts[-1][1]
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({"open": mid, "high": mid + 0.3, "low": mid - 0.3, "close": mid}, index=idx)


def test_bull_wolfe_detected():
    out = wolfe_wave_scanner_pro(_build(_BULL_VERTS))
    assert int(out["ww_best_structure"].max()) == 1
    assert int(out["ww_best_bull"].max()) == 1
    assert int(out["ww_best_score"].max()) >= 60
    assert int(out["ww_new_bull"].sum()) >= 1
    assert int(out["ww_tracked_score"].max()) > 0
    # a wedge apex must be projected ahead of P5
    assert out["ww_best_eta_bar"].notna().any()


def test_bear_wolfe_detected_mirror():
    bear = [(x, 200 - y) for (x, y) in _BULL_VERTS]
    out = wolfe_wave_scanner_pro(_build(bear))
    assert int(out["ww_best_structure"].max()) == 1
    assert int(out["ww_best_bull"].max()) == 0
    assert int(out["ww_new_bear"].sum()) >= 1


def test_columns_and_dtypes():
    out = wolfe_wave_scanner_pro(_build(_BULL_VERTS))
    assert out.index.equals(_build(_BULL_VERTS).index)
    assert len(out.columns) == len(set(out.columns))
    for col in ["ww_best_score", "ww_best_bull", "ww_best_structure", "ww_best_valid",
                "ww_best_watch", "ww_tracked_score", "ww_tracked_bull", "ww_active_count",
                "ww_new_bull", "ww_new_bear", "ww_partial", "ww_target_hit",
                "ww_invalidated", "ww_expired"]:
        assert str(out[col].dtype) == "Int64", col
    assert out["ww_best_score"].between(0, 100).all()
    assert set(out["ww_best_bull"].unique().tolist()).issubset({0, 1})
    assert set(out["ww_best_structure"].unique().tolist()).issubset({0, 1})


def test_parameter_validation():
    df = _build(_BULL_VERTS)
    bad = [
        dict(pivot_len=1),
        dict(search_depth=4),
        dict(max_concurrent_patterns=0),
        dict(min_score=101),
        dict(p5_inside_tol_atr=0.05),
        dict(min_rr=0.1),
        dict(max_leg_purity_ratio=0.05),
        dict(expiry_buffer_bars=-1),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            wolfe_wave_scanner_pro(df, **kwargs)


def test_random_runs_end_to_end():
    rng = np.random.default_rng(3)
    n = 220
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({"open": c + rng.normal(0, 0.1, n),
                       "high": c + np.abs(rng.normal(0, 0.5, n)),
                       "low": c - np.abs(rng.normal(0, 0.5, n)),
                       "close": c}, index=idx)
    out = wolfe_wave_scanner_pro(df)
    assert out.index.equals(idx)
    assert out["ww_best_score"].between(0, 100).all()
    assert (out["ww_active_count"].astype(int) >= 0).all()
