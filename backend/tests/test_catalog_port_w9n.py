"""Targeted tests for the broadening_wedge_scanner_pro port (v1.15 state machine).

Pine exposes *no* numeric plot series for this indicator (it only draws), so we
verify the computational core behaviourally:

  * a clean synthetic broadening produces a detected, correctly-classified wedge
  * a rally past the upper boundary resolves a live bullish breakout
  * invalid parameters raise ValueError
  * flag columns are nullable Int64
  * random data runs end-to-end without error
"""

import math

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.broadening_wedge_scanner_pro import (
    broadening_wedge_scanner_pro,
)


# --------------------------------------------------------------------------- #
# Frame builders                                                              #
# --------------------------------------------------------------------------- #
def _build(swings, n, rally_from=None, rally_to=None, rally_end=None):
    """Construct an OHLC frame whose confirmed swings match ``swings``.

    ``swings`` is a list of ``(x, price, dir)`` with ``dir==1`` for high pivots
    and ``dir==-1`` for low pivots, strictly alternating.  ``price`` is the
    exact swing level; the close is linearly interpolated between swing points
    and a tiny band gives high/low so no spurious pivots appear.
    """
    xs = [s[0] for s in swings]
    ys = [s[1] for s in swings]
    close = np.zeros(n)
    for i in range(n):
        if i <= xs[0]:
            close[i] = ys[0]
        elif i >= xs[-1]:
            close[i] = ys[-1]
        else:
            k = next(j for j in range(len(xs) - 1) if xs[j] <= i < xs[j + 1])
            a = (i - xs[k]) / (xs[k + 1] - xs[k])
            close[i] = ys[k] + (ys[k + 1] - ys[k]) * a
    if rally_from is not None:
        for i in range(rally_from, rally_end):
            close[i] = ys[-1] + (i - rally_from) * rally_to
    band = 0.1
    high = close + band
    low = close - band
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close}, index=idx
    )


# Symmetric broadening ("megaphone"): highs rise, lows fall, both expand.
_MEGAPHONE = [
    (30, 110, 1), (50, 90, -1), (70, 115, 1), (90, 85, -1), (110, 120, 1),
    (130, 80, -1), (150, 125, 1), (170, 75, -1), (190, 130, 1), (210, 70, -1),
    (230, 135, 1), (250, 65, -1),
]

# Descending broadening: both fall, lower falls faster (bullish bias). Gentle
# enough that the non-drift gates pass; classification is the thing under test.
_DESCENDING = [
    (30, 120, 1), (50, 100, -1), (70, 117, 1), (90, 95, -1), (110, 114, 1),
    (130, 90, -1), (150, 111, 1), (170, 85, -1), (190, 108, 1), (210, 80, -1),
]

# Ascending broadening: both rise, upper rises faster (bearish bias).
_ASCENDING = [
    (30, 90, -1), (50, 110, 1), (70, 92, -1), (90, 113, 1), (110, 95, -1),
    (130, 117, 1), (150, 99, -1), (170, 122, 1), (190, 104, -1), (210, 128, 1),
]

_TYPE_CODE = {"None": 0, "Descending Broadening": 1,
              "Ascending Broadening": 2, "Symmetric Broadening": 3}


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
def test_symmetric_wedge_detected_and_classified():
    out = broadening_wedge_scanner_pro(_build(_MEGAPHONE, 320))
    valid = out["bws_best_valid"] == 1
    assert int(valid.sum()) > 0, "megaphone should be detected as a valid wedge"
    # When valid, classification must be symmetric (code 3).
    assert (out.loc[valid, "bws_best_type"] == _TYPE_CODE["Symmetric Broadening"]).all()
    assert out["bws_best_score"].max() >= 80


def test_descending_wedge_classified():
    out = broadening_wedge_scanner_pro(_build(_DESCENDING, 320))
    has = out["bws_best_type"] != 0
    assert int(has.sum()) > 0, "a descending candidate should be evaluated"
    # ``bws_best_type`` is populated from the candidate type regardless of the
    # ATR/anchor hard gates, so it verifies the geometric classification branch.
    assert (out.loc[has, "bws_best_type"] == _TYPE_CODE["Descending Broadening"]).all()


def test_ascending_wedge_classified():
    out = broadening_wedge_scanner_pro(_build(_ASCENDING, 320))
    has = out["bws_best_type"] != 0
    assert int(has.sum()) > 0, "an ascending candidate should be evaluated"
    assert (out.loc[has, "bws_best_type"] == _TYPE_CODE["Ascending Broadening"]).all()


def test_bull_breakout_resolved():
    out = broadening_wedge_scanner_pro(
        _build(_MEGAPHONE, 320, rally_from=270, rally_to=3.0, rally_end=300)
    )
    bo = out["bws_breakout"]
    assert int((bo == 1).sum()) > 0, "rally above upper boundary must fire a bull breakout"


def test_parameter_validation():
    good = _build(_MEGAPHONE, 320)
    with pytest.raises(ValueError):
        broadening_wedge_scanner_pro(good, pivot_len=1)        # < min 2
    with pytest.raises(ValueError):
        broadening_wedge_scanner_pro(good, min_score=-1)        # < min 0
    with pytest.raises(ValueError):
        broadening_wedge_scanner_pro(good, touch_tol_atr=0.0)   # < min 0.05
    with pytest.raises(ValueError):
        broadening_wedge_scanner_pro(good, atr_len=1)           # < min 2
    with pytest.raises(ValueError):
        broadening_wedge_scanner_pro(good, max_pattern_age_mult=0.5)  # < min 1.0


def test_flag_columns_are_int64():
    out = broadening_wedge_scanner_pro(_build(_MEGAPHONE, 320))
    for col in ("bws_best_type", "bws_best_valid", "bws_best_watch",
                "bws_tracked_type", "bws_active_count", "bws_breakout"):
        assert out[col].dtype == "Int64", f"{col} must be nullable Int64"


def test_random_runs_end_to_end():
    rng = np.random.default_rng(7)
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    high = close + np.abs(rng.normal(0, 0.15, n))
    low = close - np.abs(rng.normal(0, 0.15, n))
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)
    out = broadening_wedge_scanner_pro(df)
    assert out.shape == (n, 10)
    assert list(out.columns) == [
        "bws_best_score", "bws_best_type", "bws_best_valid",
        "bws_best_projected_expansion", "bws_best_realized_expansion",
        "bws_best_watch", "bws_tracked_score", "bws_tracked_type",
        "bws_active_count", "bws_breakout",
    ]
    assert out["bws_best_projected_expansion"].notna().all()
    assert out["bws_best_realized_expansion"].notna().all()
