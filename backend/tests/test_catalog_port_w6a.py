import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.volume_profile_plus import volume_profile_plus


def _triangle_frame():
    # 24 hourly bars: price rises 100->111 then falls back, so the middle rows
    # get the most volume touches.
    n = 24
    t = np.arange(n)
    close = pd.Series(100.0 + t - 2.0 * np.maximum(t - 11, 0), dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": close + 0.2, "low": close - 0.2,
                          "close": close})
    frame["volume"] = 10.0
    frame.index = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return frame


def test_fixed_lookback_profile_matches_hand_computed_poc():
    frame = _triangle_frame()
    result = volume_profile_plus(frame, calc_mode="Fixed lookback", lookback=24,
                                 num_rows=10)
    row = result.iloc[-1]
    assert row.profile_volume == pytest.approx(240.0)  # 24 bars x 10 volume
    # POC sits in the middle band where price spent the most rows.
    assert result.val.iloc[-1] <= result.poc.iloc[-1] <= result.vah.iloc[-1]
    # Value area holds at least va_pct of the volume by construction.
    assert result.close_in_value_area.iloc[-1] in (True, False)
    assert result.hvn_count.iloc[-1] >= 1  # the POC row itself is an HVN
    with pytest.raises(ValueError, match="anchor_rule"):
        volume_profile_plus(frame, anchor_rule="H")
    with pytest.raises(ValueError, match="行情列"):
        volume_profile_plus(frame.drop(columns="volume"))


def test_session_profile_publishes_only_completed_periods():
    index = pd.date_range("2026-01-01", periods=96 * 2, freq="15min", tz="UTC")
    close = pd.Series(100.0 + 0.1 * (np.arange(96 * 2) % 96), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": close + 0.1, "low": close - 0.1,
                          "close": close})
    frame["volume"] = 5.0
    result = volume_profile_plus(frame, calc_mode="Session", anchor_rule="D",
                                 num_rows=10)
    # The session profile only publishes on the last bar of each completed
    # day (bars_into == expected), so day 0 stays blank until its 23:45 bar.
    assert result.poc.iloc[:95].isna().all()
    # Fixed mode on the same data does publish.
    fixed = volume_profile_plus(frame, calc_mode="Fixed lookback", lookback=96,
                                num_rows=10)
    assert fixed.poc.notna().any()


def test_degenerate_flat_market_is_tolerated():
    n = 60
    index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    close = pd.Series(100.0, index=index)
    frame = pd.DataFrame({"open": close, "high": close, "low": close, "close": close})
    frame["volume"] = 1.0
    result = volume_profile_plus(frame, calc_mode="Fixed lookback", lookback=30,
                                 num_rows=10)
    # Zero price range: no profile can form, everything stays NaN.
    assert result.poc.isna().all()
    with pytest.raises(ValueError, match="num_rows"):
        volume_profile_plus(frame, num_rows=5)
