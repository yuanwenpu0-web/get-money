"""Port tests for modern_wyckoff_state_machine_lite (w9h).

The Wyckoff state machine is a large composite indicator (accumulation /
distribution / markup / markdown detection). Full per-bar hand-computation of
every score is not tractable, so these tests lock the *contract* and verify the
independently-computable primitives plus structural invariants:

  * exact 41-column schema + index alignment
  * range_high / range_low are wired to rolling highest/lowest of high/low
  * range_w / range_mid identities
  * score / confidence bounds and the NEUTRAL confidence rule
  * legal state / phase value sets, boolean dtypes
  * determinism and edge cases (short / flat / no-volume)
  * parameter validation via _pine guards
"""
import math

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators import _pine
from app.catalog_indicators.modern_wyckoff_state_machine_lite import (
    modern_wyckoff_state_machine_lite,
)

STATE_SET = {"NEUTRAL", "ACCUMULATION", "DISTRIBUTION", "MARKUP", "MARKDOWN"}
PHASE_SET = {"None", "A", "B", "C", "D", "E"}

EXPECTED_COLS = [
    "state", "phase", "phase_text", "acc_score", "dist_score", "cause_score",
    "range_quality", "confidence", "range_high", "range_low", "range_mid",
    "range_w", "range_atr", "pos", "near_low", "near_high", "range_ok",
    "range_locked", "bars_in_phase", "spring", "utad", "sos", "sow",
    "test_spring", "test_utad", "lps", "lpsy", "selling_climax",
    "buying_climax", "absorption_low", "absorption_high", "spring_quality",
    "utad_quality", "markup_started", "markdown_started", "last_spring_bar",
    "last_utad_bar", "last_sos_bar", "last_sow_bar", "last_lps_bar",
    "last_lpsy_bar",
]

BOOL_COLS = [
    "near_low", "near_high", "range_ok", "range_locked", "spring", "utad",
    "sos", "sow", "test_spring", "test_utad", "lps", "lpsy", "selling_climax",
    "buying_climax", "absorption_low", "absorption_high", "markup_started",
    "markdown_started",
]


def _frame(seed=0, n=400, freq="1h", trend=None, vol=900.0, with_vol=True):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    t = np.arange(n)
    base = 100.0 + 8.0 * np.sin(t / 25.0)
    if trend == "up":
        base = np.linspace(100, 240, n)           # monotonic strong uptrend
    elif trend == "down":
        base = np.linspace(240, 100, n)           # monotonic strong downtrend
    close = base + np.cumsum(rng.standard_normal(n) * 0.25)
    op = close + rng.standard_normal(n) * 0.2
    hi = np.maximum(op, close) + np.abs(rng.standard_normal(n)) * 0.6
    lo = np.minimum(op, close) - np.abs(rng.standard_normal(n)) * 0.6
    vol_arr = rng.uniform(vol * 0.5, vol * 1.5, n) if with_vol else np.zeros(n)
    data = {"open": op, "high": hi, "low": lo, "close": close, "volume": vol_arr}
    return pd.DataFrame(data, index=idx)


def test_schema_and_index():
    out = modern_wyckoff_state_machine_lite(_frame(seed=1, n=300))
    assert list(out.columns) == EXPECTED_COLS, out.columns.tolist()
    assert len(out.columns) == 41
    assert out.index.equals(_frame(seed=1, n=300).index)


def test_range_bands_wired_neutral():
    # strong uptrend -> ADX high -> range_ok False -> state stays NEUTRAL,
    # range never locks, so out bands == live rolling highest/lowest.
    fr = _frame(seed=3, n=200, trend="up")
    out = modern_wyckoff_state_machine_lite(fr)
    exp_high = _pine.highest(fr["high"], 80)
    exp_low = _pine.lowest(fr["low"], 80)
    # compare only where expected is not NaN (first range_len bars)
    mask = exp_high.notna() & exp_low.notna()
    assert mask.any()
    pd.testing.assert_series_equal(
        out["range_high"][mask], exp_high[mask], check_names=False
    )
    pd.testing.assert_series_equal(
        out["range_low"][mask], exp_low[mask], check_names=False
    )
    # identities
    wh = out["range_high"]
    wl = out["range_low"]
    both = wh.notna() & wl.notna()
    assert np.allclose(out["range_w"][both], (wh - wl)[both])
    assert np.allclose(out["range_mid"][both], ((wh + wl) / 2.0)[both])


def test_invariants():
    for seed in range(5):
        out = modern_wyckoff_state_machine_lite(_frame(seed=seed, n=350))
        # range ordering / width
        valid = out["range_high"].notna() & out["range_low"].notna()
        assert (out["range_high"][valid] >= out["range_low"][valid]).all()
        assert (out["range_w"][valid] >= 0).all()
        # scores bounded
        for c in ["acc_score", "dist_score", "cause_score",
                  "range_quality", "confidence"]:
            assert out[c].dropna().between(0, 100).all(), c
        # pos bounded where range defined
        pr = out["pos"][valid]
        assert pr.dropna().between(-1.0, 2.0).all()
        # legal value sets
        assert set(out["state"].dropna().unique()).issubset(STATE_SET)
        assert set(out["phase"].dropna().unique()).issubset(PHASE_SET)
        # boolean dtypes
        for c in BOOL_COLS:
            assert out[c].dtype == bool, (c, out[c].dtype)


def test_confidence_rule_neutral():
    out = modern_wyckoff_state_machine_lite(_frame(seed=4, n=400))
    neu = out["state"] == "NEUTRAL"
    expected = out.loc[neu, ["acc_score", "dist_score"]].max(axis=1)
    got = out.loc[neu, "confidence"]
    assert got.dropna().equals(expected.dropna()) or \
        np.allclose(got.dropna().to_numpy(), expected.dropna().to_numpy())


def test_deterministic():
    fr = _frame(seed=9, n=300)
    a = modern_wyckoff_state_machine_lite(fr)
    b = modern_wyckoff_state_machine_lite(fr)
    pd.testing.assert_frame_equal(a, b)


def test_edge_short_frame():
    fr = _frame(seed=2, n=30)  # < default range_len (80)
    out = modern_wyckoff_state_machine_lite(fr)
    assert list(out.columns) == EXPECTED_COLS
    # first range_len bars have no rolling band yet
    assert out["range_high"].iloc[:80].isna().all()


def test_edge_flat_frame():
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    fr = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0,
         "close": 100.0, "volume": 1000.0},
        index=idx,
    )
    out = modern_wyckoff_state_machine_lite(fr)
    assert list(out.columns) == EXPECTED_COLS
    assert out["range_w"].dropna().between(0, 100).all()


def test_edge_no_volume():
    fr = _frame(seed=5, n=250, with_vol=False)
    out = modern_wyckoff_state_machine_lite(fr)
    assert list(out.columns) == EXPECTED_COLS
    assert out["state"].notna().all()


def test_param_validation():
    fr = _frame(seed=1, n=200)
    with pytest.raises(Exception):  # range_len min is 30
        modern_wyckoff_state_machine_lite(fr, range_len=10)
    with pytest.raises(Exception):  # adx_max min is 1.0
        modern_wyckoff_state_machine_lite(fr, adx_max=0.5)


def _wyckoff_schematic(mode):
    """Build a canonical Wyckoff schematic.

    mode == "accumulation":  downtrend -> range with a spring undercut ->
                             markup breakout  (expects ACCUMULATION -> MARKUP)
    mode == "distribution":  uptrend   -> range with a UTAD upthrust ->
                             markdown breakout (expects DISTRIBUTION -> MARKDOWN)
    Returns (frame, n).
    """
    rng = np.random.default_rng(33 if mode == "accumulation" else 44)
    if mode == "accumulation":
        n = 750
        idx = pd.date_range("2024-01-01", periods=n, freq="1h")
        close = np.zeros(n)
        close[:150] = np.linspace(200, 100, 150) + rng.standard_normal(150) * 0.4
        close[150:420] = 101.0 + rng.uniform(-4, 7, 270) + np.sin(np.arange(270) / 9.0)
        close[410] = 97.0          # spring: undercut below range low, close back near low
        close[420:] = np.linspace(105, 235, 330)
        spring_bar = 410
    else:
        n = 420
        idx = pd.date_range("2024-01-01", periods=n, freq="1h")
        close = np.zeros(n)
        close[:90] = np.linspace(100, 200, 90) + rng.standard_normal(90) * 0.4
        close[90:200] = 201.0 + rng.uniform(-6, 5, 110) + np.sin(np.arange(110) / 7.0)
        close[195] = 206.0         # UTAD: upthrust above range high, close back near high
        close[200:] = np.linspace(205, 90, 220)
        spring_bar = 195

    op = close + rng.standard_normal(n) * 0.08
    hi = np.maximum(op, close) + np.abs(rng.standard_normal(n)) * 0.35
    lo = np.minimum(op, close) - np.abs(rng.standard_normal(n)) * 0.35
    if mode == "accumulation":
        hi[spring_bar], lo[spring_bar], op[spring_bar], close[spring_bar] = 103.0, 91.0, 95.0, 97.0
    else:
        hi[spring_bar], lo[spring_bar], op[spring_bar], close[spring_bar] = 216.0, 200.0, 208.0, 206.0
    vol = rng.uniform(700, 1000, n)
    if mode == "accumulation":
        vol[140:145] = 2800
        vol[420:750] = rng.uniform(1300, 2200, 330)
    else:
        vol[80:85] = 2800
        vol[200:420] = rng.uniform(1300, 2200, 220)
    fr = pd.DataFrame(
        {"open": op, "high": hi, "low": lo, "close": close, "volume": vol}, index=idx
    )
    return fr, n


def test_accumulation_reaches_markup():
    fr, _ = _wyckoff_schematic("accumulation")
    out = modern_wyckoff_state_machine_lite(fr)
    # machine leaves NEUTRAL and progresses through the accumulation phases
    assert (out["state"] == "ACCUMULATION").any()
    assert set(out["phase"].unique()).issuperset({"A", "B", "C", "D", "E"})
    # the spring event actually fires and a trend release is flagged
    assert out["spring"].any()
    assert out["markup_started"].any()
    assert (out["state"] == "MARKUP").any()
    # markup_started may only be True once the state is already MARKUP
    assert out.loc[out["markup_started"], "state"].eq("MARKUP").all()


def test_distribution_reaches_markdown():
    fr, _ = _wyckoff_schematic("distribution")
    out = modern_wyckoff_state_machine_lite(fr)
    assert (out["state"] == "DISTRIBUTION").any()
    assert set(out["phase"].unique()).issuperset({"A", "B", "C", "D", "E"})
    assert out["utad"].any()
    assert out["markdown_started"].any()
    assert (out["state"] == "MARKDOWN").any()
    assert out.loc[out["markdown_started"], "state"].eq("MARKDOWN").all()
