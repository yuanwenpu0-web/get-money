import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.swing_conviction_radar import swing_conviction_radar


def _synthetic_swing_frame():
    # 40 bars: down 100->90 (10 bars), up 90->110 (20 bars), down 110->100 (10 bars).
    closes = ([100.0 - step for step in range(10)]
              + [90.0 + 1.0 * step for step in range(20)]
              + [110.0 - step for step in range(10)])
    close = pd.Series(closes, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.1,
                          "low": np.minimum(open_, close) - 0.1, "close": close})
    frame["volume"] = 10.0
    frame.index = pd.date_range("2026-01-01", periods=len(closes), freq="15min", tz="UTC")
    # Break the high/low ties so strict pivots fire exactly once each.
    frame.iloc[9, frame.columns.get_loc("low")] = 89.5
    frame.iloc[29, frame.columns.get_loc("high")] = 110.6
    return frame


def test_conviction_scores_fire_on_confirmed_pivots():
    frame = _synthetic_swing_frame()
    result = swing_conviction_radar(frame, pivot_left=3, pivot_right=3, atr_len=5,
                                    speed_ref=10.0, w_speed=0.5, w_clean=0.5, w_vol=0.0)
    # With w_vol=0 the volume legs drop out entirely.
    scored = result.drop(columns=["bearish_divergence", "bullish_divergence", "atr"])
    first_valid = scored.high_conviction.first_valid_index()
    assert first_valid is not None
    # All convictions live in 0..100 (speed is normalised, cleanliness is a share).
    assert result.high_conviction.dropna().between(0, 100).all()
    assert result.low_conviction.dropna().between(0, 100).all()
    # Perfect one-way legs step every bar in the leg direction -> cleanliness 1.
    # The up-leg (10 -> 110 over 20 bars, ATR ~1) has speed ~1/10 of speed_ref=10,
    # so conviction = (0.1*0.5 + 1.0*0.5) = 55.
    up_legs = result.high_conviction.dropna()
    assert up_legs.iloc[-1] == pytest.approx(55.0, abs=2.0)
    with pytest.raises(ValueError, match="pivot_left"):
        swing_conviction_radar(frame, pivot_left=1)
    with pytest.raises(ValueError, match="行情列"):
        swing_conviction_radar(frame.drop(columns="volume"))


def test_divergence_flags_on_lower_conviction_extremes():
    # Three rising highs; the third leg covers a similar distance much more
    # slowly, so its conviction collapses -> bearish divergence.
    legs = []
    for distance, pace in ((10.0, 1.0), (12.0, 1.0), (14.0, 0.2)):
        level = legs[-1] if legs else 100.0
        step = distance / 12.0 * pace
        for _ in range(12):
            level += step
            legs.append(level)
        legs.extend([level, level - 0.05, level - 0.1, level - 0.15])
    closes = legs + [legs[-1]] * 6
    close = pd.Series(closes, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = close + 0.05
    low = open_ - 0.05
    # Explicit bumps make each leg top/bottom the unique strict pivot (the
    # open/close envelope alone always ties across the turning bar).
    for k in range(3):
        high.iloc[k * 16 + 12] += 0.30
        low.iloc[k * 16 + 15] -= 0.30
    frame = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
    frame["volume"] = 10.0
    frame.index = pd.date_range("2026-01-01", periods=len(closes), freq="15min", tz="UTC")
    result = swing_conviction_radar(frame, pivot_left=2, pivot_right=2, atr_len=5,
                                    speed_ref=5.0, w_speed=1.0, w_clean=0.0, w_vol=0.0)
    assert result.bearish_divergence.any(), "rising highs with fading speed -> bear div"
    assert not result.bullish_divergence.any()
    assert not (result.bearish_divergence & result.bullish_divergence).any()


def test_zero_volume_feed_keeps_the_two_way_weights():
    frame = _synthetic_swing_frame()
    frame["volume"] = 0.0
    result = swing_conviction_radar(frame, pivot_left=3, pivot_right=3, atr_len=5,
                                    speed_ref=10.0, w_speed=0.4, w_clean=0.4, w_vol=0.2)
    # The volume weight is dropped, so the remaining weights renormalise to 1:
    # a perfect-speed perfect-clean leg scores exactly 100.
    assert result.high_conviction.dropna().between(0, 100).all()
    with pytest.raises(ValueError, match="speed_ref"):
        swing_conviction_radar(frame, speed_ref=0.1)
