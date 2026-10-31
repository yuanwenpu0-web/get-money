"""Port tests for mso_market_stress_oscillator (w9j).

Bi-directional Williams-VIX-Fix stress oscillator: JMA trend context, ADX quality
grade, z-scored WVF long/short, and a peak-hold cluster + forward follow-through
acceptance state machine. Tests lock the 32-column schema, dtype/range invariants,
determinism, edge cases, the JMA recursion against a hand-computed reference, the
f_z zero-stddev branch, and two crafted stress scenarios (rally -> accepted,
continued drop -> rejected) that exercise the full event path.
"""
import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.mso_market_stress_oscillator import (
    _z,
    calculate_jma,
    mso_market_stress_oscillator,
)

EXPECTED_COLS = [
    "z_long", "z_short", "wvf_long", "wvf_short", "jma", "jma_slope_sm", "bias",
    "adx", "adx_grade", "ext_long_raw", "ext_short_raw", "ext_long", "ext_short",
    "trend_long_ok", "trend_short_ok", "regime",
    "long_event", "long_score", "long_follow", "long_peak_z", "long_peak_price",
    "long_peak_atr", "long_peak_idx", "short_event", "short_score", "short_follow",
    "short_peak_z", "short_peak_price", "short_peak_atr", "short_peak_idx",
    "last_event_side", "last_event_accepted",
]


def _frame(seed=0, n=600, trend=None):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    base = 100.0 + np.arange(n) * 0.05
    if trend == "up":
        base = np.linspace(100, 240, n)
    elif trend == "down":
        base = np.linspace(240, 100, n)
    close = base + np.cumsum(rng.standard_normal(n) * 0.3)
    op = close + rng.standard_normal(n) * 0.1
    hi = np.maximum(op, close) + np.abs(rng.standard_normal(n)) * 0.3
    lo = np.minimum(op, close) - np.abs(rng.standard_normal(n)) * 0.3
    vol = rng.uniform(500, 1500, n)
    return pd.DataFrame(
        {"open": op, "high": hi, "low": lo, "close": close, "volume": vol}, index=idx
    )


def _stress_frame(mode: str):
    """Gentle uptrend baseline (so WVF has a non-zero std) with a 2-bar downside
    stress dip at bar 180, then either a rally (accept) or a continued drop (reject)."""
    n = 260
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(42)
    base = 100.0 + np.arange(n) * 0.05
    close = base + rng.standard_normal(n) * 0.3
    op = close + rng.standard_normal(n) * 0.1
    hi = np.maximum(op, close) + np.abs(rng.standard_normal(n)) * 0.3
    lo = np.minimum(op, close) - np.abs(rng.standard_normal(n)) * 0.3
    c = 180
    # 2-bar deep dip (close 90 / low 89) -> consecutive extreme, forms a cluster
    for b in (c, c + 1):
        close[b] = 90.0
        low_ = 89.0
        high_ = 90.0
        op[b] = 90.0
        lo[b] = low_
        hi[b] = high_
    if mode == "rally":
        close[c + 2:c + 20] = np.linspace(93, 110, 18)
        hi[c + 2:c + 20] = np.linspace(94, 112, 18)
    elif mode == "drop":
        # highs never exceed the peak close (90) -> follow-through <= 0 -> reject
        close[c + 2:c + 20] = np.linspace(89, 70, 18)
        hi[c + 2:c + 20] = np.linspace(89, 71, 18)
    else:
        raise ValueError(mode)
    return pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close, "volume": 1000.0}, index=idx)


def test_schema_and_index():
    out = mso_market_stress_oscillator(_frame(seed=1, n=300))
    assert list(out.columns) == EXPECTED_COLS
    assert len(out.columns) == 32
    assert out.index.equals(_frame(seed=1, n=300).index)


def test_invariants():
    for seed in range(4):
        out = mso_market_stress_oscillator(_frame(seed=seed, n=350))
        # bounded / typed columns
        assert out["adx_grade"].dropna().between(0, 1).all()
        assert out["long_score"].dropna().between(0, 1).all()
        assert out["short_score"].dropna().between(0, 1).all()
        assert (out["long_follow"].dropna() >= 0).all()
        assert (out["short_follow"].dropna() >= 0).all()
        assert out["regime"].dropna().isin([-1, 0, 1]).all()
        for c in ["ext_long_raw", "ext_short_raw", "ext_long", "ext_short",
                  "trend_long_ok", "trend_short_ok"]:
            assert out[c].dropna().isin([0, 1]).all()
        for c in ["long_event", "short_event", "long_peak_idx", "short_peak_idx",
                  "last_event_side", "last_event_accepted"]:
            assert out[c].dtype == "Int64"
        assert out["last_event_side"].dropna().isin([-1, 1]).all()
        assert out["last_event_accepted"].dropna().isin([0, 1]).all()
        # peak indexes within frame
        pidx = out["long_peak_idx"].dropna().astype(int)
        assert pidx.between(0, len(out) - 1).all()


def test_deterministic():
    fr = _frame(seed=9, n=300)
    a = mso_market_stress_oscillator(fr)
    b = mso_market_stress_oscillator(fr)
    pd.testing.assert_frame_equal(a, b)


def test_edge_short_frame():
    fr = _frame(seed=2, n=30)
    out = mso_market_stress_oscillator(fr)
    assert list(out.columns) == EXPECTED_COLS


def test_edge_flat_frame():
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    fr = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0},
        index=idx,
    )
    out = mso_market_stress_oscillator(fr)
    assert list(out.columns) == EXPECTED_COLS
    # flat series -> no WVF spikes -> no stress events
    assert out["long_event"].notna().sum() == 0
    assert out["short_event"].notna().sum() == 0


def test_edge_no_volume():
    fr = _frame(seed=5, n=250)
    fr = fr.assign(volume=0.0)
    out = mso_market_stress_oscillator(fr)
    assert list(out.columns) == EXPECTED_COLS


def test_param_validation():
    fr = _frame(seed=1, n=200)
    with pytest.raises(Exception):
        mso_market_stress_oscillator(fr, wvf_len=2)  # min 5
    with pytest.raises(Exception):
        mso_market_stress_oscillator(fr, z_win=10)  # min 20
    with pytest.raises(Exception):
        mso_market_stress_oscillator(fr, tf="XYZ")  # not a preset


def test_jma_recursion_hand_computed():
    src = pd.Series([10.0, 12.0, 11.0, 13.0, 9.0, 10.0])
    L, ph, pw = 3, 0.0, 2.0
    beta = 0.45 * (L - 1) / (0.45 * (L - 1) + 2)
    alpha = beta ** pw
    a1 = 1 - alpha
    a1_2 = a1 * a1
    a2 = alpha * alpha
    e0 = e1 = e2 = jma = np.nan
    ref = []
    for x in src:
        e0 = a1 * x + alpha * (x if np.isnan(e0) else e0)
        e1 = (x - e0) * (1 - beta) + beta * (0.0 if np.isnan(e1) else e1)
        jp = x if np.isnan(jma) else jma
        e2 = (e0 + 1.5 * e1 - jp) * a1_2 + a2 * (0.0 if np.isnan(e2) else e2)
        jma = e2 + jp
        ref.append(jma)
    got = calculate_jma(src, L, ph, pw).tolist()
    assert np.allclose(ref, got, atol=1e-9)
    # first value seeds at the source
    assert np.isclose(got[0], src.iloc[0])


def test_z_zero_stddev_branch():
    # variation then a constant run -> stdev becomes 0 in the constant window
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    z = _z(s, 5)
    # last full window [5,5,5,5,5] has sd==0 -> z-score forced to 0 (Pine f_z)
    assert z.iloc[9] == 0.0
    # earlier windows have sd>0 -> finite, non-zero (except warmup NaN)
    assert not np.isnan(z.iloc[4])


def test_stress_rally_accepted():
    out = mso_market_stress_oscillator(_stress_frame("rally"))
    acc = out[out["long_event"] == 1]
    assert len(acc) >= 1
    # accepted rows respect the acceptance definition
    assert (acc["long_follow"] >= 0.6).all()
    assert (acc["long_peak_z"] >= 2.0).all()  # z_thr default (1h)
    assert acc["long_score"].between(0, 1).all()


def test_stress_drop_rejected():
    out = mso_market_stress_oscillator(_stress_frame("drop"))
    rej = out[out["long_event"] == 0]
    assert len(rej) >= 1
    # the rejection path fires because follow-through stays below minFollowATR
    assert (rej["long_follow"] < 0.6).all()


def test_use_peak_hold_false_runs():
    out = mso_market_stress_oscillator(_frame(seed=3, n=400), use_peak_hold=False)
    assert list(out.columns) == EXPECTED_COLS
    # still deterministic
    a = mso_market_stress_oscillator(_frame(seed=3, n=400), use_peak_hold=False)
    pd.testing.assert_frame_equal(out, a)


def test_tf_selection_runs():
    out = mso_market_stress_oscillator(_frame(seed=4, n=400), tf="4h")
    assert list(out.columns) == EXPECTED_COLS
    # 4h preset uses a different wvf_len / z_win but same schema
    assert out.index.equals(_frame(seed=4, n=400).index)
