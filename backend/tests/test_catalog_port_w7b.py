import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.auto_trendlines import auto_trendlines


def _peaks_frame():
    # Three descending peaks at 110/108/106 and two rising troughs 100/102.
    rows = 120
    index = pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC")
    close = pd.Series(104.0 + 0.0 * np.arange(rows), index=index)
    close.iloc[20] = 110.0
    close.iloc[50] = 108.0
    close.iloc[80] = 106.0
    close.iloc[35] = 100.0
    close.iloc[65] = 102.0
    open_ = close.shift(1).fillna(close.iloc[0])
    # Isolate the pivots: narrow wicks around each spike.
    high = close + 0.2
    low = close - 0.2
    frame = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
    for bar in (20, 50, 80):
        frame.iloc[bar, frame.columns.get_loc("high")] = close.iloc[bar] + 0.5
    for bar in (35, 65):
        frame.iloc[bar, frame.columns.get_loc("low")] = close.iloc[bar] - 0.5
    return frame


def test_descending_resistance_and_rising_support():
    frame = _peaks_frame()
    result = auto_trendlines(frame, lookback=120, pivot_left=4, pivot_right=4,
                             min_touches=2, max_lines=2, min_span=10,
                             relevance_atr=50.0)
    last = result.iloc[-1]
    assert result.resistance_level.notna().any()
    assert result.support_level.notna().any()
    # Resistance slopes down (negative), support slopes up (non-negative).
    if not np.isnan(result.resistance_slope.iloc[-1]):
        assert result.resistance_slope.iloc[-1] <= 0
    if not np.isnan(result.support_slope.iloc[-1]):
        assert result.support_slope.iloc[-1] >= 0
    early = result.iloc[90]
    assert early.support_level < early.resistance_level
    assert result.resistance_touches.dropna().astype(int).ge(2).all()
    with pytest.raises(ValueError, match="violation_source"):
        auto_trendlines(frame, violation_source="Body")
    with pytest.raises(ValueError, match="行情列"):
        auto_trendlines(frame.drop(columns="high"))


def test_line_method_and_fit_mode_change_nothing_fatal():
    frame = _peaks_frame()
    for method in ("Directional", "Regression"):
        for fit in ("Normal", "Outer"):
            result = auto_trendlines(frame, line_method=method, fit_mode=fit,
                                     lookback=120, pivot_left=4, pivot_right=4,
                                     relevance_atr=50.0)
            assert result.resistance_level.notna().any() or \
                result.support_level.notna().any()


def test_retest_flags_and_violation_source():
    frame = _peaks_frame()
    wick = auto_trendlines(frame, violation_source="Wick", lookback=120,
                           pivot_left=4, pivot_right=4, relevance_atr=50.0)
    close = auto_trendlines(frame, violation_source="Close", lookback=120,
                            pivot_left=4, pivot_right=4, relevance_atr=50.0)
    # Both variants produce lines; the retest flags are boolean.
    assert wick.retest_resistance.dtype == bool
    assert close.retest_support.dtype == bool
    with pytest.raises(ValueError, match="show_mode"):
        auto_trendlines(frame, show_mode="Diagonal")
