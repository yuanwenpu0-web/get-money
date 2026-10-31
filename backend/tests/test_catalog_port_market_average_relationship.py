"""Tests for the ported Market–Average Relationship Engine catalog indicator."""

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.market_average_relationship_engine import market_average_relationship_engine


def _frame(seed=0, n=400):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    o = c + rng.normal(0, 0.1, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.4, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.4, n))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": 1000 + np.abs(rng.normal(0, 100, n))}, index=idx)


def test_columns_dtypes_ranges():
    out = market_average_relationship_engine(_frame())
    assert out.index.equals(_frame().index)
    assert len(out.columns) == len(set(out.columns))
    assert out["mare_trend_quality"].dropna().between(0.0, 100.0).all()
    assert out["mare_relationship"].dropna().between(-100.0, 100.0).all()
    for col in ["mare_respect", "mare_extension", "mare_compression", "mare_acceleration", "mare_exhaustion"]:
        assert out[col].dropna().between(0.0, 100.0).all(), col
    assert str(out["mare_state"].dtype) == "Int64"
    assert out["mare_state"].between(0, 11).all()


def test_ma_families_run():
    for ma_type in ["EMA", "SMA", "WMA", "HMA", "VWMA", "RMA", "DEMA", "TEMA", "ALMA"]:
        out = market_average_relationship_engine(_frame(n=220), ma_type=ma_type)
        assert out["mare_ma"].iloc[-20:].notna().all(), ma_type


def test_events_are_int64():
    out = market_average_relationship_engine(_frame(seed=1))
    event_cols = [c for c in out.columns if any(t in c for t in
                  ["pullback", "momentum", "chase", "exhaustion_", "break_", "divergence", "turn_"])]
    assert event_cols
    for col in event_cols:
        assert str(out[col].dtype) == "Int64", col
        assert set(out[col].dropna().unique().tolist()).issubset({0, 1})


def test_parameter_validation():
    df = _frame(n=160)
    bad = [
        dict(ma_type="Nope"),
        dict(ma_length=1),
        dict(atr_length=1),
        dict(statistics_length=10),
        dict(respect_lookback=5),
        dict(divergence_pivot_left=1),
        dict(turn_lookback=2),
        dict(full_slope_atr=0.001),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            market_average_relationship_engine(df, **kwargs)


def test_random_end_to_end():
    out = market_average_relationship_engine(_frame(seed=3, n=260))
    assert len(out) == 260
    assert out["mare_relationship"].notna().any()
