"""Tests for the ported Klinger Volume Force Map catalog indicator."""

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.klinger_volume_force_map import klinger_volume_force_map
from app.catalog_indicators._pine import ema


def _frame(seed: int = 0, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    vol = 1000 + np.abs(rng.normal(0, 200, n))
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close,
                         "volume": vol}, index=idx)


def _klinger_trend(frame: pd.DataFrame) -> np.ndarray:
    high = frame["high"].to_numpy(float)
    low = frame["low"].to_numpy(float)
    close = frame["close"].to_numpy(float)
    sum_hlc = high + low + close
    diff = np.zeros(len(frame))
    diff[1:] = sum_hlc[1:] - sum_hlc[:-1]
    sign = np.sign(diff)
    sign[sign == 0] = np.nan
    return pd.Series(sign, index=frame.index).ffill().fillna(1.0).to_numpy().astype(int)


def test_kvo_equals_ema_cascade():
    df = _frame(seed=1)
    out = klinger_volume_force_map(df)
    vf = out["vf"].to_numpy(float)
    expected = (ema(pd.Series(vf, index=df.index), 34) - ema(pd.Series(vf, index=df.index), 55)).to_numpy()
    err = np.nanmax(np.abs(out["kvo_raw"].to_numpy(float) - expected))
    assert err == pytest.approx(0.0, abs=1e-9)
    # Signal is ema(KVO, 13)
    sig_expected = ema(out["kvo_raw"], 13).to_numpy()
    assert np.nanmax(np.abs(out["signal_raw"].to_numpy(float) - sig_expected)) == pytest.approx(0.0, abs=1e-9)


def test_relative_kvo_formula():
    df = _frame(seed=2)
    out = klinger_volume_force_map(df)
    kvo = out["kvo_raw"].to_numpy(float)
    den = np.maximum(ema(out["kvo_raw"].abs(), 100).to_numpy(float), 1e-10)
    expected = kvo / den
    assert np.nanmax(np.abs(out["relative_kvo"].to_numpy(float) - expected)) == pytest.approx(0.0, abs=1e-9)


def test_signed_formula_volume_force():
    df = _frame(seed=3)
    out = klinger_volume_force_map(df, formula_mode="Signed Volume Simplified")
    ktr = _klinger_trend(df).astype(float)
    vol = df["volume"].to_numpy(float)
    expected_vf = vol * ktr * 100.0
    assert np.allclose(out["vf"].to_numpy(float), expected_vf, equal_nan=True)


def test_regime_state_in_range():
    out = klinger_volume_force_map(_frame(seed=4))
    states = set(out["regime_state"].dropna().unique().tolist())
    assert states.issubset({-2, -1, 0, 1, 2})
    assert states, "regime engine should produce at least one state"


def test_flag_columns_are_int64_and_binary():
    out = klinger_volume_force_map(_frame(seed=5))
    flag_cols = [c for c in out.columns if c.endswith(("_div", "_cross", "_rejection", "_confirm", "_expansion", "compression"))]
    assert flag_cols
    for col in flag_cols:
        assert str(out[col].dtype) == "Int64", col
        values = set(out[col].dropna().unique().tolist())
        assert values.issubset({0, 1}), (col, values)
    assert str(out["regime_state"].dtype) == "Int64"
    events = set(out["last_event_code"].dropna().unique().tolist())
    assert events.issubset(set(range(13)))


def test_divergence_quality_filter_ordering():
    df = _frame(seed=6, n=600)
    loose = klinger_volume_force_map(df, div_quality_filter="All")
    strict = klinger_volume_force_map(df, div_quality_filter="Strong")
    loose_div = loose[["regular_bear_div", "regular_bull_div", "hidden_bear_div", "hidden_bull_div"]].to_numpy().sum()
    strict_div = strict[["regular_bear_div", "regular_bull_div", "hidden_bear_div", "hidden_bull_div"]].to_numpy().sum()
    assert loose_div >= strict_div
    # at least one divergence somewhere in a 600-bar sample
    assert loose_div >= 1


def test_parameter_validation():
    df = _frame(seed=7, n=120)
    bad_calls = [
        dict(formula_mode="Nope"),
        dict(norm_mode="Nope"),
        dict(div_quality_filter="Nope"),
        dict(fast_len=0),
        dict(slow_len=1),
        dict(norm_len=10),
        dict(pivot_left=0),
        dict(div_min_bars=0),
        dict(volume_is_real="yes"),
    ]
    for kwargs in bad_calls:
        with pytest.raises(ValueError):
            klinger_volume_force_map(df, **kwargs)


def test_random_runs_end_to_end():
    for seed in range(4):
        df = _frame(seed=100 + seed, n=300)
        out = klinger_volume_force_map(df, norm_mode=["Relative Force", "StdDev Scale", "Raw"][seed % 3])
        assert len(out) == len(df)
        assert out.index.equals(df.index)
        assert len(out.columns) == len(set(out.columns))
        # numeric columns should be finite past warm-up
        tail = out["kvo_raw"].iloc[-50:]
        assert tail.notna().all()
