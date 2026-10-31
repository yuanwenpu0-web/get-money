"""Tests for the ported Structural Wave Field catalog indicator."""

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.structural_wave_field import structural_wave_field


def _random_frame(seed=0, n=300):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"open": c + rng.normal(0, 0.1, n),
                         "high": c + np.abs(rng.normal(0, 0.5, n)),
                         "low": c - np.abs(rng.normal(0, 0.5, n)),
                         "close": c,
                         "volume": 1000 + np.abs(rng.normal(0, 100, n))}, index=idx)


def _trend_frame(n=250, slope=0.5):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = 100 + np.arange(n) * slope
    return pd.DataFrame({"open": c, "high": c + 0.3, "low": c - 0.3, "close": c,
                         "volume": np.full(n, 1000.0)}, index=idx)


def test_columns_dtypes_ranges():
    out = structural_wave_field(_random_frame())
    assert out.index.equals(_random_frame().index)
    assert len(out.columns) == len(set(out.columns))
    assert str(out["swf_struct_dir"].dtype) == "Int64"
    assert str(out["swf_phase"].dtype) == "Int64"
    assert set(out["swf_struct_dir"].dropna().unique().tolist()).issubset({-1, 0, 1})
    assert out["swf_structural_strength"].between(0.0, 1.0).all()
    assert out["swf_structural_confidence"].between(0.0, 1.0).all()
    assert out["swf_maturity"].between(0.0, 1.0).all()
    assert out["swf_exhaustion"].between(0.0, 1.0).all()
    assert out["swf_compression"].between(0.0, 1.0).all()
    assert out["swf_phase"].between(0, 9).all()
    assert out["swf_resync_stage"].between(0, 4).all()


def test_uptrend_is_bullish():
    out = structural_wave_field(_trend_frame(slope=0.5))
    assert int(out["swf_struct_dir"].iloc[-1]) == 1
    assert float(out["swf_structural_strength"].max()) > 0.5
    assert int(out["swf_structural_initiated"].sum()) >= 1


def test_flags_are_int64_binary():
    out = structural_wave_field(_random_frame(seed=4))
    flag_cols = [c for c in out.columns if c.startswith("swf_") and any(
        t in c for t in ["flip", "initiated", "exit", "confirmed", "failed", "weakening",
                         "exhaustion_event", "context"])]
    assert flag_cols
    for col in flag_cols:
        assert str(out[col].dtype) == "Int64", col
        assert set(out[col].dropna().unique().tolist()).issubset({0, 1})


def test_participation_modes_run():
    for mode in ["Off", "Volume", "Relative Volume", "Money Flow"]:
        out = structural_wave_field(_random_frame(seed=2, n=200), participation_mode=mode)
        assert out["swf_participation"].between(0.0, 1.0).all()


def test_parameter_validation():
    df = _random_frame(n=120)
    bad = [
        dict(atr_len=1),
        dict(struct_base=4),
        dict(wave_len=2),
        dict(quality_len=2),
        dict(medium_ratio=5.0),
        dict(slow_ratio=1.0),
        dict(sensitivity_mode="Nope"),
        dict(participation_mode="Nope"),
        dict(advanced_calibration=True, struct_hold=0.5, struct_entry=0.34),
        dict(advanced_calibration=True, struct_flip=0.1, struct_entry=0.34),
        dict(advanced_calibration=True, medium_ratio=4.0, slow_ratio=2.0),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            structural_wave_field(df, **kwargs)


def test_end_to_end_unique_columns():
    out = structural_wave_field(_random_frame(seed=7, n=260))
    assert len(out) == 260
    assert out["swf_phase"].notna().all()
