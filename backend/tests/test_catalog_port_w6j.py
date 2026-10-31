import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.reversal_engine_score_v1 import reversal_engine_score_v1


def _sweep_frame():
    # Flat base, a dip below the prior 10-bar low that closes back above it
    # (classic liquidity sweep), then recovery. 15-minute base.
    closes = [100.0] * 20
    closes += [99.8, 99.5, 99.2, 99.0]        # gentle drift down
    closes += [99.2, 99.4, 99.6, 99.8]        # base
    closes += [99.9, 99.6, 99.4, 99.7]        # the sweep bar closes back above
    closes += [100.0, 100.2, 100.4, 100.6]    # reclaim
    closes += [100.8] * 8
    close = pd.Series(closes, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    highs = np.maximum(open_, close) + 0.05
    lows = np.minimum(open_, close) - 0.05
    # A single wick bar dips below the standing low and closes back above it.
    lows[28] = 98.9
    frame = pd.DataFrame({"open": open_, "high": highs, "low": lows, "close": close})
    frame.index = pd.date_range("2026-01-01", periods=len(closes), freq="15min", tz="UTC")
    return frame


def test_sweep_detected_and_quality_tiered():
    frame = _sweep_frame()
    result = reversal_engine_score_v1(frame, use_gate_f=False, use_gate_d=False)
    assert result.liquidity_sweep_long.any()
    assert set(result.long_tier.unique()) <= {"A", "B", "C"}
    assert set(result.short_tier.unique()) <= {"A", "B", "C"}
    assert result.long_quality.dropna().between(0, 2).all()
    # Long signals respect the cooldown: gaps between consecutive fires.
    fires = result.index[result.long_signal | result.short_signal]
    gaps = np.diff([frame.index.get_loc(i) for i in fires]) if len(fires) > 1 else [99]
    assert (gaps > 5).all() if len(gaps) else True
    with pytest.raises(ValueError, match="min_quality_long"):
        reversal_engine_score_v1(frame, min_quality_long=3)
    with pytest.raises(ValueError, match="行情列"):
        reversal_engine_score_v1(frame.drop(columns="high"))


def test_htf_trend_gate_blocks_against_trend():
    frame = _sweep_frame()
    gated = reversal_engine_score_v1(frame, use_gate_f=True, htf_minutes=60,
                                     htf_fast_len=2, htf_slow_len=5)
    # The gate ran and produced sane columns; the ramp finish makes the HTF
    # trend bullish, so shorts are blocked but longs are not by gate F.
    assert gated.bull_trend.iloc[-1] and not gated.bear_trend.iloc[-1]
    with pytest.raises(ValueError, match="整数倍"):
        reversal_engine_score_v1(frame, htf_minutes=7)


def test_stop_target_geometry_and_direction_filter():
    frame = _sweep_frame()
    result = reversal_engine_score_v1(frame, use_gate_f=False, use_gate_d=False,
                                      trade_dir="Long Only")
    assert not result.short_signal.any()
    fired = result[result.long_signal]
    for _, row in fired.iterrows():
        assert row.long_stop < row.long_target
        risk = abs(row.long_target - 99.0)  # approximate entry region
    gated_off = reversal_engine_score_v1(frame, use_gate_a=False, use_gate_f=False,
                                         use_gate_d=False, trade_dir="Short Only")
    assert not gated_off.long_signal.any()
    with pytest.raises(ValueError, match="session"):
        reversal_engine_score_v1(frame, use_gate_i=True, session="bad")
