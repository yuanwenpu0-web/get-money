import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.capital41_liquidity_void_imbalance_map import \
    capital41_liquidity_void_imbalance_map


def _gapping_frame():
    # One clean bullish FVG at bar 2 (low 11.6 > high[0] 10.0); the later bars
    # overlap the previous bar so no further voids open.
    rows = [
        (9.8, 10.0, 9.5, 9.9),      # 0
        (10.2, 10.5, 10.0, 10.4),   # 1
        (11.8, 12.0, 11.6, 11.9),   # 2  bull void [10.0, 11.6]
        (11.9, 12.1, 11.8, 12.0),   # 3  overlaps bar 2 -> no new void
        (12.0, 12.2, 11.9, 12.1),   # 4
        (12.1, 12.3, 12.0, 12.2),   # 5  low 12.0 stays above the zone top 11.6
    ]
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    frame["volume"] = 10.0
    frame.index = pd.date_range("2026-01-01", periods=len(rows), freq="15min", tz="UTC")
    return frame


def test_bull_void_detected_and_fill_tracks_price():
    frame = _gapping_frame()
    result = capital41_liquidity_void_imbalance_map(
        frame, use_htf=False, merge_overlaps=True, max_zones=10, fresh_bars=80)
    assert result.ltf_bull_void.iloc[2]
    assert result.new_void.iloc[2]
    # Bar 2 opens [10.0, 11.6]; bar 3's gap [10.5, 11.8] overlaps and merges
    # into [10.0, 11.8] (same-direction merge keeps the union).
    assert result.top1_upper.iloc[5] == pytest.approx(11.8)
    assert result.top1_lower.iloc[5] == pytest.approx(10.0)
    assert result.top1_direction.iloc[5] == 1
    # Fill: low 12.7 is above the top -> 0% filled at bar 5.
    assert result.top1_score.iloc[5] > 0
    with pytest.raises(ValueError, match="整数倍"):
        capital41_liquidity_void_imbalance_map(frame, use_htf=True, htf_minutes=7)
    with pytest.raises(ValueError, match="行情列"):
        capital41_liquidity_void_imbalance_map(frame.drop(columns="low"))


def test_bear_void_and_breaker_flag():
    # Bull void then a full sweep down through it: bear breaker.
    rows = [
        (9.8, 10.0, 9.5, 9.9),
        (10.2, 10.5, 10.0, 10.4),
        (11.8, 12.0, 11.6, 11.9),   # low 11.6 > high[0] 10.0 -> bull void
        (11.9, 12.1, 11.8, 12.0),   # merges into [10.0, 11.8]
        (11.0, 11.8, 9.5, 9.8),     # the low sweeps the whole zone (fill=100)
                                    # and the close lands below the bottom
    ]
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    frame["volume"] = 10.0
    frame.index = pd.date_range("2026-01-01", periods=len(rows), freq="15min", tz="UTC")
    result = capital41_liquidity_void_imbalance_map(
        frame, use_htf=False, merge_overlaps=True, max_zones=10, fresh_bars=50)
    assert result.ltf_bull_void.iloc[2]
    # The bull zone spans [10.0, 11.8]. Bar 4's low 9.5 sweeps the whole zone
    # (fill 100%) while closing at 9.8 below the bottom -> breaker-down; the
    # filled zone leaves the top-1 ranking afterwards.
    assert result.top1_lower.iloc[3] == pytest.approx(10.0)
    assert result.top1_upper.iloc[3] == pytest.approx(11.8)
    assert result.breaker_down.iloc[4]
    assert pd.isna(result.top1_score.iloc[4])


def test_htf_voids_carry_extra_weight_and_need_completed_periods():
    index = pd.date_range("2026-01-01", periods=96 * 3, freq="15min", tz="UTC")
    level = 100.0
    closes = []
    for bar in range(len(index)):
        closes.append(level)
        if bar % 24 == 5:
            level += 3.0  # gaps inside the hourly aggregation
        elif bar % 24 == 17:
            level -= 3.0
    close = pd.Series(closes, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.2,
                          "low": np.minimum(open_, close) - 0.2, "close": close})
    result = capital41_liquidity_void_imbalance_map(frame, use_htf=True,
                                                    htf_minutes=60, max_zones=10)
    # The first hour cannot publish a completed HTF void.
    assert not result.htf_bull_void.iloc[:4].any()
    assert result.new_void.any()
    with pytest.raises(ValueError, match="max_zones"):
        capital41_liquidity_void_imbalance_map(frame, max_zones=2)
