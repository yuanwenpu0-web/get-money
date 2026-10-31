import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.bayesian_trend_factor import bayesian_trend_factor


def _frame(closes, spread=0.5):
    index = pd.date_range("2026-01-01", periods=len(closes), freq="15min", tz="UTC")
    close = pd.Series(np.asarray(closes, dtype=float), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": close + spread,
                         "low": close - spread, "close": close}, index=index)


def _trend(n=300, slope=0.12, amp=3.0, period=14.0):
    t = np.arange(n)
    return _frame(100 + slope * t + amp * np.sin(t / period))


def _pullback_series(dip=0.6, cycles=14):
    """Gentle uptrend (+1 per 30 bars) with a two-bar dip back to the EMA/fair path."""
    closes = []
    base = 100.0
    for _ in range(cycles):
        closes.extend(np.linspace(base, base + 1.0, 30)[1:].tolist())
        base += 1.0
        closes.extend([base - dip, base - dip * 0.4])
        closes.extend(np.linspace(base - dip * 0.4, base + 0.8, 6)[1:].tolist())
        base += 0.8
    return _frame(closes)


def test_btf_factor_and_confidence_ranges():
    frame = _trend()
    out = bayesian_trend_factor(frame)
    factor = out.trend_factor.dropna()
    assert factor.between(-100, 100).all()
    confidence = out.confidence.dropna()
    assert confidence.between(0, 100).all()


def test_btf_uptrend_produces_bull_state():
    frame = _trend(slope=0.25, amp=1.0, period=40.0)
    out = bayesian_trend_factor(frame)
    assert out.trend_factor.iloc[-1] > 55
    assert out.trend_state.iloc[-1] == 1
    assert out.state.iloc[-1] in {"Bull Trend", "Strong Bull"}
    assert out.bull_prob.iloc[-1] > 0.5


def test_btf_downtrend_produces_bear_state():
    frame = _trend(slope=-0.25, amp=1.0, period=40.0)
    out = bayesian_trend_factor(frame)
    assert out.trend_factor.iloc[-1] < -55
    assert out.trend_state.iloc[-1] == -1
    assert out.state.iloc[-1] in {"Bear Trend", "Strong Bear"}
    assert out.bull_prob.iloc[-1] < 0.5


def test_btf_evidence_components_bounded():
    frame = _trend()
    out = bayesian_trend_factor(frame)
    for column in ("direction_evidence", "structure_evidence", "strength_evidence"):
        values = out[column].dropna()
        assert values.between(-1, 1).all(), column
    assert out.vol_optimal.dropna().between(0, 1).all()
    assert out.exhaustion_score.dropna().between(0, 1).all()


def test_btf_confidence_gate_scales_factor():
    frame = _trend()
    gated = bayesian_trend_factor(frame, conf_threshold=95.0)
    ungated = bayesian_trend_factor(frame, conf_threshold=0.0)
    # A high confidence gate shrinks the factor toward zero wherever confidence falls short.
    common = gated.trend_factor.notna() & ungated.trend_factor.notna()
    assert (gated.trend_factor[common].abs() <= ungated.trend_factor[common].abs() + 1e-9).all()


def test_btf_smoothing_length_reduces_jitter():
    frame = _trend(amp=6.0, period=9.0)
    smooth = bayesian_trend_factor(frame, factor_smooth_len=6)
    raw = bayesian_trend_factor(frame, factor_smooth_len=1)
    smooth_jitter = smooth.trend_factor.dropna().diff().abs().mean()
    raw_jitter = raw.trend_factor.dropna().diff().abs().mean()
    assert smooth_jitter < raw_jitter


def test_btf_exhaustion_penalises_extension():
    """A stretched move far above the fair path must raise the exhaustion score."""
    closes = list(np.linspace(100, 101, 40)) + list(np.linspace(101, 118, 8))
    frame = _frame(closes, spread=0.4)
    out = bayesian_trend_factor(frame, conf_threshold=0.0)
    assert out.exhaustion_score.max() > 0
    peak = out.exhaustion_score.idxmax()
    assert out.distance_atr.loc[peak] > 1.0


def test_btf_pullback_state_machine_arms_and_matures():
    frame = _pullback_series()
    out = bayesian_trend_factor(frame, conf_threshold=0.0)
    assert out.bull_pb_start.any()
    assert (out.bull_pb_state == 1).any()
    assert (out.bull_pb_state == 2).any()
    assert out.bull_continuation.any()
    assert set(out.pb_state.unique()) <= {
        "Idle", "Bull PB Armed", "Bear PB Armed", "Bull PB Mature", "Bear PB Mature",
        "Bull Cont", "Bear Cont", "PB Invalidated"}
    mature = out[out.bull_pb_state == 2]
    assert mature.pb_depth_atr.between(0.25, 1.75).all()


def test_btf_pullback_invalidates_on_depth_or_timeout():
    frame = _pullback_series(dip=3.0)
    out = bayesian_trend_factor(frame, conf_threshold=0.0)
    assert out.bull_pb_invalidated.any()
    invalidated = out[out.bull_pb_invalidated]
    assert invalidated.bull_pb_state.eq(4).all()


def test_btf_pullback_disabled_stays_idle():
    frame = _pullback_series()
    out = bayesian_trend_factor(frame, enable_pullback=False)
    assert not out.bull_pb_start.any()
    assert (out.pb_state == "Idle").all()


def test_btf_trend_candles_are_smoothed():
    frame = _trend(amp=6.0, period=9.0)
    out = bayesian_trend_factor(frame)
    raw_range = (frame.high - frame.low).dropna()
    smooth_range = (out.trend_candle_high - out.trend_candle_low).dropna()
    assert smooth_range.mean() < raw_range.mean() * 3
    candles = out[["trend_candle_open", "trend_candle_high", "trend_candle_low",
                   "trend_candle_close"]].dropna()
    assert (candles.trend_candle_high >= candles.trend_candle_close).all()
    assert (candles.trend_candle_low <= candles.trend_candle_close).all()
    assert (candles.trend_candle_high >= candles.trend_candle_open).all()


def test_btf_rejects_bad_params():
    frame = _trend()
    with pytest.raises(ValueError, match="reg_len"):
        bayesian_trend_factor(frame, reg_len=5)
    with pytest.raises(ValueError, match="trend_threshold"):
        bayesian_trend_factor(frame, trend_threshold=10.0)
    with pytest.raises(ValueError, match="candle_open_type"):
        bayesian_trend_factor(frame, candle_open_type="HMA")
