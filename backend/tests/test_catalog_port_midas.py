"""Targeted tests for the midas_2_0 port (v2.7.3 episode-cumulative state machine).

Coverage:
  * core MIDAS curve == expanding mean of source under equal-weight + Manual Time
    anchor (exact, machine-precision) — proves the cumulative episode engine
  * VW-StDev band invariant (upper1 - midas == mult1 * population std, exact)
  * TBF fit path activates on a suitably-shaped series (finite curve, sane states)
  * disabling TBF yields an all-``off`` state
  * Auto Last Swing anchor produces a non-empty MIDAS curve
  * parameter validation raises ValueError
  * flag columns are nullable Int64
  * random / trending frames run end-to-end without error
"""

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.midas_2_0 import midas_2_0


def _frame(close, high=None, low=None, n_extra=None):
    n = len(close)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    high = close + np.abs(np.random.default_rng(0).normal(0, 0.3, n)) if high is None else high
    low = close - np.abs(np.random.default_rng(1).normal(0, 0.3, n)) if low is None else low
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close}, index=idx
    )


def test_midas_equals_expanding_mean():
    rng = np.random.default_rng(1)
    n = 150
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = _frame(close)
    out = midas_2_0(df, anchor_mode="Manual Time", anchor_time=df.index[0],
                   src="close", volume_weight_mode="Equal weight")
    ref = pd.Series(close).expanding().mean().to_numpy()
    np.testing.assert_allclose(out["midas"].to_numpy(), ref, atol=1e-9)


def test_band_invariant():
    rng = np.random.default_rng(2)
    n = 150
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = _frame(close)
    out = midas_2_0(df, anchor_mode="Manual Time", anchor_time=df.index[0],
                   src="close", volume_weight_mode="Equal weight",
                   band_mult1=1.5, band_mult2=2.5)
    ref_std = np.array([np.std(close[: b + 1], ddof=0) for b in range(n)])
    up_err = np.nanmax(np.abs((out["midas_upper1"] - out["midas"]).to_numpy() - 1.5 * ref_std))
    lo_err = np.nanmax(np.abs((out["midas"] - out["midas_lower1"]).to_numpy() - 1.5 * ref_std))
    assert up_err < 1e-9, up_err
    assert lo_err < 1e-9, lo_err
    # upper2 / lower2 use mult2
    up2_err = np.nanmax(np.abs((out["midas_upper2"] - out["midas"]).to_numpy() - 2.5 * ref_std))
    assert up2_err < 1e-9, up2_err


def test_tbf_fit_activates():
    n = 400
    b = np.arange(n)
    rng = np.random.default_rng(0)
    close = 100 + 0.2 * b + 40 * np.sin(2 * np.pi * b / 8.0) + rng.normal(0, 0.1, n)
    # Deterministic ±1 bands (the config that reliably triggers forward fits).
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close}, index=idx)
    out = midas_2_0(df, anchor_mode="Manual Time", anchor_time=df.index[0],
                   src="close", volume_weight_mode="Equal weight")
    assert int(out["tbf"].notna().sum()) > 0, "TBF fit should activate on this shape"
    # Valid phase codes: off=0, no fit=1, developing=2, mature=3, late=4, passed=5
    assert set(out["tbf_state"].dropna().unique()).issubset({0, 1, 2, 3, 4, 5})
    prog = out["tbf_progress"].dropna()
    assert prog.min() > 0 and prog.max() < 1.5


def test_tbf_disabled_state_off():
    rng = np.random.default_rng(3)
    n = 200
    close = 100 + 0.2 * np.arange(n) + 40 * np.sin(2 * np.pi * np.arange(n) / 8.0) + rng.normal(0, 0.1, n)
    df = _frame(close)
    out = midas_2_0(df, anchor_mode="Manual Time", anchor_time=df.index[0],
                   src="close", volume_weight_mode="Equal weight", enable_tbf=False)
    assert (out["tbf_state"] == 0).all()
    assert out["tbf"].isna().all()


def test_auto_last_swing_produces_midas():
    rng = np.random.default_rng(4)
    n = 250
    close = 100 + np.cumsum(rng.normal(0.05, 1, n))
    df = _frame(close)
    out = midas_2_0(df, anchor_mode="Auto Last Swing", src="close",
                   volume_weight_mode="Equal weight", anchor_left=10, anchor_right=10)
    assert out["midas"].notna().sum() > n * 0.5


def test_parameter_validation():
    rng = np.random.default_rng(5)
    df = _frame(100 + np.cumsum(rng.normal(0, 1, 100)))
    with pytest.raises(ValueError):
        midas_2_0(df, anchor_mode="Nonsense")
    with pytest.raises(ValueError):
        midas_2_0(df, band_mode="HybridX")
    with pytest.raises(ValueError):
        midas_2_0(df, anchor_left=0)            # < min 1
    with pytest.raises(ValueError):
        midas_2_0(df, atr_len=0)                # < min 1
    with pytest.raises(ValueError):
        midas_2_0(df, fit_min_progress=0.01)    # < min 0.05
    with pytest.raises(ValueError):
        midas_2_0(df, warn_thresh=0.4)          # < min 0.5


def test_flag_columns_are_int64():
    rng = np.random.default_rng(6)
    df = _frame(100 + np.cumsum(rng.normal(0, 1, 150)))
    out = midas_2_0(df, anchor_mode="Manual Time", anchor_time=df.index[0],
                   src="close", volume_weight_mode="Equal weight")
    for col in ("tbf_state", "zone", "reclaim_up_score", "reclaim_dn_score",
                "tbf_reaction_score", "bias", "tbf_top", "stretch_above", "stretch_below"):
        assert out[col].dtype == "Int64", f"{col} must be nullable Int64"


def test_random_runs_end_to_end():
    rng = np.random.default_rng(7)
    n = 200
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    df = _frame(close)
    out = midas_2_0(df, anchor_mode="Manual Time", anchor_time=df.index[0],
                   src="close", volume_weight_mode="Equal weight")
    assert out.shape == (n, 20)
    # float columns fully populated once the episode warms up
    assert out["midas"].notna().sum() > n * 0.9
    assert out["midas_dev"].notna().sum() > n * 0.9
    # every column present and uniquely named
    assert len(out.columns) == len(set(out.columns))
