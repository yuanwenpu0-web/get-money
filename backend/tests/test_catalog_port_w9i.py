"""Port tests for reversal_type_classifier_v1 (w9i).

Ex-post pivot classifier: each confirmed pivot reversal is tagged with a type
(1=Snapback, 2=Grind, 3=Fake/Pullback, 4=Chop) plus a 0–100 quality score and
the MFE/MAE in R. Tests lock the 28-column schema, invariants (type sets,
score/ R bounds, dtype, forward-filled last_type), determinism, edge cases, and
the core first-hit helper with hand-computed values. Two crafted scenarios
(V-bottom snapback, failed rally -> fake) exercise the full classification path.
"""
import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.reversal_type_classifier_v1 import (
    _first_hit,
    _first_hit_short,
    _pivot_strict,
    reversal_type_classifier_v1,
)

EXPECTED_COLS = [
    "long_type", "long_score", "long_mfe_r", "long_mae_r", "long_wt_at_pivot",
    "long_div", "long_extreme", "long_trend_against", "long_chop",
    "long_atr_rank", "long_adx", "long_srsi", "long_bbw_rank",
    "short_type", "short_score", "short_mfe_r", "short_mae_r", "short_wt_at_pivot",
    "short_div", "short_extreme", "short_trend_against", "short_chop",
    "short_atr_rank", "short_adx", "short_srsi", "short_bbw_rank",
    "last_type", "last_side",
]

TYPE_SET = {1, 2, 3, 4}


def _frame(seed=0, n=600, freq="1h", trend=None, vol=900.0, with_vol=True):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    t = np.arange(n)
    base = 100.0 + 8.0 * np.sin(t / 25.0)
    if trend == "up":
        base = np.linspace(100, 240, n)
    elif trend == "down":
        base = np.linspace(240, 100, n)
    close = base + np.cumsum(rng.standard_normal(n) * 0.15)
    op = close + rng.standard_normal(n) * 0.2
    hi = np.maximum(op, close) + np.abs(rng.standard_normal(n)) * 0.6
    lo = np.minimum(op, close) - np.abs(rng.standard_normal(n)) * 0.6
    vol_arr = rng.uniform(vol * 0.5, vol * 1.5, n) if with_vol else np.zeros(n)
    return pd.DataFrame(
        {"open": op, "high": hi, "low": lo, "close": close, "volume": vol_arr}, index=idx
    )


def test_schema_and_index():
    out = reversal_type_classifier_v1(_frame(seed=1, n=300))
    assert list(out.columns) == EXPECTED_COLS
    assert len(out.columns) == 28
    assert out.index.equals(_frame(seed=1, n=300).index)


def test_invariants():
    for seed in range(4):
        out = reversal_type_classifier_v1(_frame(seed=seed, n=350))
        for c in ["long_type", "short_type"]:
            vals = out[c].dropna()
            assert set(vals.unique()).issubset(TYPE_SET)
        # scores bounded 0..100
        for c in ["long_score", "short_score"]:
            assert out[c].dropna().between(0, 100).all()
        # MFE/MAE non-negative
        assert (out["long_mfe_r"].dropna() >= 0).all()
        assert (out["long_mae_r"].dropna() >= 0).all()
        assert (out["short_mfe_r"].dropna() >= 0).all()
        assert (out["short_mae_r"].dropna() >= 0).all()
        # flag columns are Int64 0/1
        for c in ["long_div", "long_extreme", "long_trend_against", "long_chop",
                  "short_div", "short_extreme", "short_trend_against", "short_chop"]:
            assert out[c].dropna().isin([0, 1]).all()
        # last_type / last_side legal
        assert set(out["last_type"].dropna().unique()).issubset(TYPE_SET)
        assert set(out["last_side"].dropna().unique()).issubset({"long", "short"})


def test_deterministic():
    fr = _frame(seed=9, n=300)
    a = reversal_type_classifier_v1(fr)
    b = reversal_type_classifier_v1(fr)
    pd.testing.assert_frame_equal(a, b)


def test_edge_short_frame():
    fr = _frame(seed=2, n=30)  # < piv_len + eval_bars warmup
    out = reversal_type_classifier_v1(fr)
    assert list(out.columns) == EXPECTED_COLS


def test_edge_flat_frame():
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    fr = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0},
        index=idx,
    )
    out = reversal_type_classifier_v1(fr)
    assert list(out.columns) == EXPECTED_COLS
    # no real pivots on a flat series -> nothing classified, but last_type stays NaN
    assert out["long_type"].notna().sum() == 0


def test_edge_no_volume():
    fr = _frame(seed=5, n=250, with_vol=False)
    out = reversal_type_classifier_v1(fr)
    assert list(out.columns) == EXPECTED_COLS


def test_param_validation():
    fr = _frame(seed=1, n=200)
    with pytest.raises(Exception):  # piv_len min is 2
        reversal_type_classifier_v1(fr, piv_len=1)
    with pytest.raises(Exception):  # eval_bars min is 5
        reversal_type_classifier_v1(fr, eval_bars=2)


def _assert_pivot(got, expected):
    assert len(got) == len(expected)
    for a, b in zip(got, expected):
        if b is None or (isinstance(b, float) and np.isnan(b)):
            assert a is None or (isinstance(a, float) and np.isnan(a))
        else:
            assert a == b


def test_pivot_strict_tiny():
    s = pd.Series([5.0, 4.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    _assert_pivot(_pivot_strict(s, 2, 2, False).tolist(), [None, None, None, None, 3.0, None, None])
    s2 = pd.Series([5.0, 6.0, 7.0, 6.0, 5.0, 4.0, 3.0])
    _assert_pivot(_pivot_strict(s2, 2, 2, True).tolist(), [None, None, None, None, 7.0, None, None])
    # non-strict neighbour must NOT be a pivot
    s3 = pd.Series([5.0, 7.0, 8.0, 7.0, 5.0])
    # value 8 at idx 2; left has 7 (lower) -> not a strict low
    assert np.isnan(_pivot_strict(s3, 2, 2, False).iloc[4])


def test_first_hit_hand_computed():
    # price=100, risk=10 -> target (long)=112, fail=90
    h = np.array([101.0, 103.0, 112.0, 115.0, 108.0])
    lo = np.array([99.0, 98.0, 97.0, 96.0, 95.0])
    # first high>=112 at window index 2 -> first_target = 3 (1-based)
    # no low<=90 -> first_fail None
    ft, ff = _first_hit(h, lo, 100.0, 10.0, 1.2, 1.0)
    assert ft == 3 and ff is None
    # fail hits first: add an early deep low
    lo2 = np.array([99.0, 88.0, 97.0, 96.0, 95.0])
    ft2, ff2 = _first_hit(h, lo2, 100.0, 10.0, 1.2, 1.0)
    assert ff2 == 2 and ft2 == 3  # fail at window idx1 -> 2, target still 3


def test_first_hit_short_hand_computed():
    # short: target on low=80 (price 100, risk 10, targetR 2.0 -> 80), fail on high=110
    lo = np.array([99.0, 98.0, 80.0, 82.0, 85.0])
    hi = np.array([101.0, 102.0, 103.0, 104.0, 112.0])
    ft, ff = _first_hit_short(lo, hi, 100.0, 10.0, 2.0, 1.0)
    assert ft == 3 and ff == 5  # fail hits at window idx4 -> j=5
    hi2 = np.array([101.0, 111.0, 103.0, 104.0, 112.0])
    ft2, ff2 = _first_hit_short(lo, hi2, 100.0, 10.0, 2.0, 1.0)
    assert ff2 == 2 and ft2 == 3


def _vbottom_frame():
    """Downtrend into a V-bottom, then a fast rally that hits the snap target."""
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = np.zeros(n)
    # 0..30 downtrend 120 -> 90
    close[:31] = np.linspace(120, 90, 31)
    # 31..70 fast rally 90 -> 130
    close[31:71] = np.linspace(90, 130, 40)
    # 71.. end drift
    close[71:] = 130 + np.linspace(0, 10, n - 71)
    op = close + 0.1
    hi = np.maximum(op, close) + 0.5
    lo = np.minimum(op, close) - 0.5
    # keep the V-bottom bar (30) clearly the lowest
    lo[30] = 89.0
    hi[30] = 90.5
    vol = np.full(n, 1000.0)
    return pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close, "volume": vol}, index=idx)


def _failed_rally_frame():
    """A strict pivot low at bar 30, followed by a *small* rally that stays just
    below the snap target (so first_snap is never set), then a steep drop that
    breaks the fail level before any target is hit -> Fake (type 3).

    Geometry (piv_len=5): bar 30 is the lowest low; bars 25..29 and 31..35 are
    all strictly higher, so bar 30 is a confirmed pivot low. The rally (31..35)
    reaches ~101.3 (enough for the impulse filter and mfe>=minFollowR) but stays
    below the snap target (~101.9), and the post-window drop (36+) falls through
    the fail level (~98.4), so classify_long returns type 3.
    """
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = np.zeros(n)
    # clean downtrend 120 -> 100, ending exactly at the pivot bar 30
    close[:31] = np.linspace(120, 100, 31)
    # small rally: strictly above the pivot low (100), peaking below snap target
    close[31:36] = np.array([100.5, 100.8, 101.0, 100.8, 100.5])
    # steep drop through the fail level and well below
    close[36:] = np.linspace(95.0, 55.0, n - 36)
    op = close + 0.05
    hi = np.maximum(op, close) + 0.4
    lo = np.minimum(op, close) - 0.4
    # pin the pivot low bar 30 to exactly 100 (strictly below every neighbour)
    hi[30] = 100.4
    lo[30] = 100.0
    vol = np.full(n, 1000.0)
    return pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close, "volume": vol}, index=idx)


def test_snapback_scenario():
    out = reversal_type_classifier_v1(_vbottom_frame())
    # a fast rally after the V-bottom should be classified as Snapback (type 1)
    assert out["long_type"].eq(1).any()
    # the snapback pivot must have MFE >= its snap target (>= 1.2 R) and a real score
    snap = out[out["long_type"] == 1]
    assert (snap["long_mfe_r"] >= 1.0).all()
    assert snap["long_score"].between(0, 100).all()


def test_fake_scenario():
    out = reversal_type_classifier_v1(_failed_rally_frame())
    # a pivot that fails (price drops below) should be classified as Fake (type 3)
    assert out["long_type"].eq(3).any()
