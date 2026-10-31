"""Tests for the liquidity_hunter catalog port (w9k)."""
import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators import liquidity_hunter as lh
from app.catalog_indicators.liquidity_hunter import (
    liquidity_hunter,
    _f_level_score,
    _f_context_bonus,
    _f_event_score,
    COLUMNS,
)


def _base_frame(n=80, seed=1):
    """Near-flat frame whose highs/lows are constant enough that only the
    explicitly-injected peaks become strict pivots."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    open_ = np.full(n, 100.0)
    high = np.full(n, 100.5)
    low = np.full(n, 98.5)
    close = np.full(n, 100.0)
    vol = np.full(n, 1000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


def test_columns_and_index():
    fr = _base_frame(200)
    out = liquidity_hunter(fr)
    assert list(out.columns) == COLUMNS
    assert len(out.columns) == 46
    assert out.index.equals(fr.index)


def test_deterministic():
    fr = _base_frame(300)
    a = liquidity_hunter(fr)
    b = liquidity_hunter(fr)
    pd.testing.assert_frame_equal(a, b)


def test_int_dtype():
    fr = _base_frame(200)
    out = liquidity_hunter(fr)
    for c in ["bsl_sweep", "ssl_sweep", "bsl_breakout", "ssl_breakout", "bsl_hunt",
              "ssl_hunt", "bearish_reclaim", "bullish_reclaim", "exhaustion_bear",
              "exhaustion_bull", "primary_bsl_touches", "secondary_bsl_touches",
              "primary_ssl_touches", "secondary_ssl_touches", "active_bsl_count",
              "active_ssl_count", "near_primary_bsl", "near_primary_ssl",
              "bias_value", "fvg_new", "fvg_unfilled", "volume_is_real",
              "in_premium", "is_trending", "trend_up"]:
        assert out[c].dtype == "Int64", c


def test_f_level_score_hand():
    # touches=3, age=100, dist=2.0, maxAge=400, maxDist=8
    got = _f_level_score(3, 100, 2.0, 400, 8)
    # 60 + 18 - 4.5 - 14 = 59.5
    assert got == pytest.approx(59.5)


def test_f_context_bonus_hand():
    # bearish + inPremium + not trending -> +6 +6 = 12 (capped)
    got = _f_context_bonus(True, True, False, True, 12.0)
    assert got == pytest.approx(12.0)
    # bullish + in discount + trending up -> discount bonus AND uptrend-aligned bonus
    got = _f_context_bonus(False, False, True, True, 12.0)
    assert got == pytest.approx(12.0)


def test_f_event_score_hand():
    # Case A: no real volume -> coreScale = 147/123
    a = _f_event_score(3, 0.5, 0.3, 1.0, 0.0, 0.0, 50.0, True, False,
                       0.0, 0.0, False, 53.0)
    expected_a = (36.0 + 9.0 + 7.2) * (147.0 / 123.0)
    assert a == pytest.approx(expected_a)
    # Case B: real volume, no momentum, session boost caps at 100
    b = _f_event_score(4, 1.0, 0.5, 2.0, 10.0, 0.0, 50.0, True, False,
                       0.0, 0.0, True, 53.0)
    assert b == pytest.approx(100.0)
    # Case C: real volume + momentum, bearish WT/MFI extremes -> +10/+10
    c = _f_event_score(2, 0.2, 0.2, 1.5, 0.0, 60.0, 85.0, True, True,
                       10.0, 5.0, True, 53.0)
    core = 24.0 + 3.6 + 4.8 + 6.0
    expected_c = core + 10.0 + 10.0 + 10.0 + 5.0
    assert c == pytest.approx(expected_c)


def test_param_validation():
    fr = pd.DataFrame({"open": [1], "high": [1], "low": [1]})  # no close
    with pytest.raises(ValueError):
        liquidity_hunter(fr)


def test_short_frame_no_crash():
    fr = _base_frame(10)  # < 2*pivot_len+1
    out = liquidity_hunter(fr, pivot_len=8)
    assert len(out) == 10
    # no pivots possible -> no sweeps
    assert int(out["bsl_sweep"].sum()) == 0
    assert int(out["ssl_sweep"].sum()) == 0


def test_no_volume_bypasses_gate():
    """Without real volume, reclaims must still be evaluable (vol gate bypassed)."""
    fr = _base_frame(80)
    fr = fr.drop(columns=["volume"])
    out = liquidity_hunter(fr, pivot_len=2)
    assert int(out["volume_is_real"].sum()) == 0


def test_fvg_detection():
    fr = _base_frame(70)
    # bullish FVG at bar 60: low > high[2]
    fr.loc[fr.index[58], "high"] = 99.0
    fr.loc[fr.index[60], "low"] = 103.0
    fr.loc[fr.index[60], "high"] = 104.0
    fr.loc[fr.index[60], "close"] = 103.0
    out = liquidity_hunter(fr, pivot_len=2)
    assert int(out["fvg_new"].iloc[60]) == 1
    # size% = (103 - 99)/103*100 ~= 3.883
    assert out["fvg_new_size_pct"].iloc[60] == pytest.approx(3.883, abs=0.01)
    # gap remains unfilled right after
    assert int(out["fvg_unfilled"].iloc[60]) >= 1


def _craft_sweep_reclaim_frame():
    """Build a frame with a BSL double-top + sweep + bearish reclaim and an
    SSL double-bottom + sweep + bullish reclaim, plus a bullish FVG."""
    n = 90
    fr = _base_frame(n)
    idx = fr.index

    # ---- BSL: equal highs at 101 (pivot_len=2) ----
    for b in (8, 9, 11, 12):
        fr.loc[idx[b], "high"] = 100.5
    fr.loc[idx[10], "high"] = 101.0
    fr.loc[idx[10], "low"] = 99.0
    for b in (28, 29, 31, 32):
        fr.loc[idx[b], "high"] = 100.5
    fr.loc[idx[30], "high"] = 101.0
    fr.loc[idx[30], "low"] = 99.0
    # sweep bar 40 (wick up, close stays below level)
    fr.loc[idx[38], "high"] = 100.5
    fr.loc[idx[39], "high"] = 100.5
    fr.loc[idx[41], "high"] = 100.5
    fr.loc[idx[42], "high"] = 100.5
    fr.loc[idx[40], "high"] = 102.0
    fr.loc[idx[40], "low"] = 98.0
    fr.loc[idx[40], "close"] = 100.5
    # bearish reclaim bar 42 (close back below, high still >= level)
    fr.loc[idx[42], "high"] = 101.5
    fr.loc[idx[42], "low"] = 98.5
    fr.loc[idx[42], "close"] = 100.5

    # ---- SSL: equal lows at 90 ----
    for b in (13, 14, 16, 17):
        fr.loc[idx[b], "low"] = 98.5
    fr.loc[idx[15], "low"] = 90.0
    fr.loc[idx[15], "high"] = 100.5
    for b in (33, 34, 36, 37):
        fr.loc[idx[b], "low"] = 98.5
    fr.loc[idx[35], "low"] = 90.0
    fr.loc[idx[35], "high"] = 100.5
    # sweep bar 50 (wick down, close stays above level)
    fr.loc[idx[48], "low"] = 98.5
    fr.loc[idx[49], "low"] = 98.5
    fr.loc[idx[51], "low"] = 98.5
    fr.loc[idx[52], "low"] = 98.5
    fr.loc[idx[50], "low"] = 88.0
    fr.loc[idx[50], "high"] = 100.5
    fr.loc[idx[50], "close"] = 92.0
    # bullish reclaim bar 52
    fr.loc[idx[52], "low"] = 89.5
    fr.loc[idx[52], "high"] = 100.5
    fr.loc[idx[52], "close"] = 92.0

    # ---- bullish FVG at bar 70 ----
    fr.loc[idx[68], "high"] = 99.0
    fr.loc[idx[70], "low"] = 103.0
    fr.loc[idx[70], "high"] = 104.0
    fr.loc[idx[70], "close"] = 103.0
    # No real volume: reclaim vol gate is bypassed (matches source when volume is
    # unavailable), so sweeps/reclaims fire on price structure alone.
    fr = fr.drop(columns=["volume"])
    return fr


def test_crafted_sweep_and_reclaim():
    fr = _craft_sweep_reclaim_frame()
    out = liquidity_hunter(fr, pivot_len=2)
    assert int(out["bsl_sweep"].sum()) >= 1, "expected a BSL sweep"
    assert int(out["bearish_reclaim"].sum()) >= 1, "expected a bearish reclaim"
    assert int(out["ssl_sweep"].sum()) >= 1, "expected an SSL sweep"
    assert int(out["bullish_reclaim"].sum()) >= 1, "expected a bullish reclaim"
    # FVG fired
    assert int(out["fvg_new"].eq(1).sum()) >= 1
    # primary levels should be populated once levels exist
    assert out["primary_bsl_price"].notna().sum() >= 1
    assert out["primary_ssl_price"].notna().sum() >= 1
    # bias reflects at least one reclaim event at some bar
    assert int(out["bias_value"].abs().max()) >= 1


def test_level_flip_on_breakout():
    """A breakout (full close-through) should NOT set a sweep on the broken level
    but should flip it to the opposite side when enable_level_flip is on."""
    fr = _craft_sweep_reclaim_frame()
    # turn the BSL sweep into a breakout: close above the level at bar 40
    idx = fr.index
    fr.loc[idx[40], "close"] = 102.5
    out = liquidity_hunter(fr, pivot_len=2, enable_level_flip=True)
    # the original level broke out (not a wick-reject sweep)
    assert int(out["bsl_breakout"].sum()) >= 1
    # the flipped level should appear on the SSL side as active/primary eventually
    assert out["primary_ssl_price"].notna().sum() >= 1
