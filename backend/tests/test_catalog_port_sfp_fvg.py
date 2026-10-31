"""Tests for the ported SFP + FVG / IFVG catalog indicator (chart-TF core)."""

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.sfp_fvg_ifvg import sfp_fvg_ifvg


def _sfp_frame(n=120):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    o = np.full(n, np.nan); h = np.full(n, np.nan); l = np.full(n, np.nan); c = np.full(n, np.nan)

    def seg(a, b, y0, y1):
        for k in range(a, b):
            c[k] = y0 + (y1 - y0) * (k - a) / (b - a - 1)

    seg(0, 31, 100, 110.2); seg(31, 51, 110.0, 104); seg(51, 59, 104, 109.2)
    o[59], h[59], l[59], c[59] = 109.2, 110.9, 108.9, 109.0
    o[60], h[60], l[60], c[60] = 109.0, 109.2, 107.5, 107.8
    for k in range(61, n):
        o[k], h[k], l[k], c[k] = 107.8, 108.0, 107.6, 107.8
    for k in range(59):
        if np.isnan(o[k]):
            h[k] = c[k] + 0.3; l[k] = c[k] - 0.3; o[k] = c[k]
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": np.full(n, 1000.0)}, index=idx)


def _gap_frame(n=80):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = np.full(n, 100.0); o = np.full(n, 100.0); h = np.full(n, 100.3); l = np.full(n, 99.7)
    o[31], c[31], h[31], l[31] = 100.0, 105.0, 105.3, 104.5
    for k in range(32, n):
        o[k], c[k], h[k], l[k] = 105.0, 105.0, 105.3, 104.7
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": np.full(n, 1000.0)}, index=idx)


def test_bear_sfp_raw_and_conf():
    out = sfp_fvg_ifvg(_sfp_frame())
    assert out["sfp_bear_raw"].iloc[59] == 1
    assert out["sfp_bear_conf"].iloc[60] == 1


def test_bull_fvg_created():
    out = sfp_fvg_ifvg(_gap_frame())
    assert int(out["sfp_fvg_bull_new"].sum()) >= 1
    assert out["sfp_fvg_above_top"].notna().any() or out["sfp_fvg_below_top"].notna().any()


def test_columns_and_dtypes():
    out = sfp_fvg_ifvg(_sfp_frame())
    assert out.index.equals(_sfp_frame().index)
    assert len(out.columns) == len(set(out.columns))
    for col in ["sfp_bear_raw", "sfp_bull_raw", "sfp_bear_conf", "sfp_bull_conf",
                "sfp_pending_bear", "sfp_pending_bull", "sfp_fvg_bull_new",
                "sfp_fvg_bear_new", "sfp_ifvg_bull_new", "sfp_ifvg_bear_new"]:
        assert str(out[col].dtype) == "Int64", col
        assert set(out[col].dropna().unique().tolist()).issubset({0, 1})
    # EMA warm-up
    assert out["sfp_ema_slow"].iloc[199:].notna().all()
    assert out["sfp_vwap_fast"].notna().sum() > 0


def test_pvp_levels_over_days():
    rng = np.random.default_rng(1)
    n = 240
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c,
                       "volume": 1000 + np.abs(rng.normal(0, 50, n))}, index=idx)
    out = sfp_fvg_ifvg(df)
    assert out["sfp_pvp_daily"].notna().sum() > 0


def test_parameter_validation():
    df = _sfp_frame()
    bad = [
        dict(swiece_swingu=1),
        dict(okres_atr=1),
        dict(maks_wiek=5),
        dict(max_sweep_atr=-1.0),
        dict(min_wielkosc_fvg_atr=-1.0),
        dict(ema_fast_len=0),
        dict(filtr_wolumenu="yes"),
        dict(wlacz_fvg="yes"),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            sfp_fvg_ifvg(df, **kwargs)


def test_random_runs_end_to_end():
    rng = np.random.default_rng(3)
    n = 260
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({"open": c + rng.normal(0, 0.1, n),
                       "high": c + np.abs(rng.normal(0, 0.5, n)),
                       "low": c - np.abs(rng.normal(0, 0.5, n)),
                       "close": c,
                       "volume": 1000 + np.abs(rng.normal(0, 100, n))}, index=idx)
    out = sfp_fvg_ifvg(df)
    assert out.index.equals(idx)
    assert out["sfp_bear_raw"].isin([0, 1]).all()
