"""Tests for the candle_story_engine port (Candle Story Engine v1.10.2).

Hand-computed expectations come from the source formula block; scenario frames
exercise the stateful path (compression breakout, liquidity sweep, patterns,
signal hysteresis) that vectorised unit checks cannot reach.
"""
import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.candle_story_engine import candle_story_engine, _COLUMNS


def _frame(open_, high, low, close, vol=None):
    n = len(close)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    data = {"open": pd.Series(open_, index=idx), "high": pd.Series(high, index=idx),
            "low": pd.Series(low, index=idx), "close": pd.Series(close, index=idx)}
    if vol is not None:
        data["volume"] = pd.Series(vol, index=idx)
    return pd.DataFrame(data)


def test_schema_and_dtypes():
    rng = np.random.default_rng(0)
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = 100 + np.cumsum(rng.standard_normal(n))
    op = close + rng.standard_normal(n) * 0.2
    hi = np.maximum(op, close) + np.abs(rng.standard_normal(n)) * 0.5
    lo = np.minimum(op, close) - np.abs(rng.standard_normal(n)) * 0.5
    fr = _frame(op, hi, lo, close, vol=rng.uniform(500, 1500, n))
    out = candle_story_engine(fr)
    assert list(out.columns) == _COLUMNS
    assert len(out.columns) == 56
    assert out.index.equals(idx)
    assert all(str(dt) in ("float64", "Int64") for dt in out.dtypes)


def test_determinism():
    rng = np.random.default_rng(1)
    n = 250
    close = 100 + np.cumsum(rng.standard_normal(n))
    op = close + rng.standard_normal(n) * 0.2
    hi = np.maximum(op, close) + np.abs(rng.standard_normal(n)) * 0.5
    lo = np.minimum(op, close) - np.abs(rng.standard_normal(n)) * 0.5
    fr = _frame(op, hi, lo, close, vol=rng.uniform(500, 1500, n))
    a = candle_story_engine(fr)
    b = candle_story_engine(fr)
    pd.testing.assert_frame_equal(a, b)


def test_candle_pressure_and_quality_hand_computed():
    # Identical bullish candles -> every per-bar pressure value is constant.
    op = np.full(60, 100.0)
    cl = np.full(60, 110.0)
    hi = np.full(60, 112.0)
    lo = np.full(60, 98.0)
    fr = _frame(op, hi, lo, cl)  # no volume -> adjustment 1.0
    out = candle_story_engine(fr)
    rng = 14.0
    body_pct = 10.0 / rng
    close_pos = 12.0 / rng
    lower_wick_pct = 2.0 / rng
    expected_pressure = ((close_pos - 0.5) * 2.0 * 45.0
                         + 1.0 * body_pct * 40.0
                         + (lower_wick_pct - lower_wick_pct) * 15.0)
    expected_quality = (body_pct * 35.0 + close_pos * 40.0 + lower_wick_pct * 15.0
                        + (1.0 / 1.5) * 10.0)  # range/atr = 1.0 for constant range
    # candle_pressure has no warm-up (pure per-bar OHLC).
    assert out["candle_pressure"].iloc[10] == pytest.approx(expected_pressure, abs=1e-6)
    assert out["bull_quality"].iloc[40] == pytest.approx(expected_quality, abs=1e-6)


def test_sequence_pressure_and_dominance():
    op = np.full(60, 100.0)
    cl = np.full(60, 110.0)
    hi = np.full(60, 112.0)
    lo = np.full(60, 98.0)
    fr = _frame(op, hi, lo, cl)
    out = candle_story_engine(fr, sequence_length=5)
    # identical candles -> sequence pressure == single-candle pressure
    single = ((12.0 / 14.0 - 0.5) * 2.0 * 45.0 + (10.0 / 14.0) * 40.0)
    assert out["sequence_pressure"].iloc[40] == pytest.approx(single, abs=1e-6)
    body_pct = 10.0 / 14.0
    expected_dom = 5.0 / 5.0 * 40.0 + (5 * body_pct) / 5.0 * 60.0
    assert out["bull_dominance"].iloc[40] == pytest.approx(expected_dom, abs=1e-6)


def test_no_volume_degrades_to_neutral():
    rng = np.random.default_rng(2)
    n = 200
    close = 100 + np.cumsum(rng.standard_normal(n))
    op = close + rng.standard_normal(n) * 0.2
    hi = np.maximum(op, close) + np.abs(rng.standard_normal(n)) * 0.5
    lo = np.minimum(op, close) - np.abs(rng.standard_normal(n)) * 0.5
    fr = _frame(op, hi, lo, close)
    out = candle_story_engine(fr)
    assert bool((out["volume_ratio"] == 1.0).all())


def test_bull_engulfing_detected():
    n = 60
    close = np.linspace(100, 105, n)
    op = close.copy()
    hi = close + 1.0
    lo = close - 1.0
    # bar 29 bearish, bar 30 bullish engulfing
    op[29], close[29], hi[29], lo[29] = 101.0, 95.0, 102.0, 94.0
    op[30], close[30], hi[30], lo[30] = 94.0, 103.0, 104.0, 93.0
    fr = _frame(op, hi, lo, close, vol=np.full(n, 1000.0))
    out = candle_story_engine(fr)
    assert int(out["bull_engulfing"].iloc[30]) == 1
    assert int(out["bull_engulfing"].iloc[29]) == 0


def test_bull_marubozu_role():
    n = 60
    close = np.linspace(100, 105, n)
    op = close.copy()
    hi = close + 1.0
    lo = close - 1.0
    # bar 40 strong bullish marubozu
    op[40], close[40], hi[40], lo[40] = 100.0, 110.0, 110.2, 99.9
    fr = _frame(op, hi, lo, close, vol=np.full(n, 1000.0))
    out = candle_story_engine(fr)
    assert int(out["bull_marubozu"].iloc[40]) == 1
    assert int(out["candle_role_code"].iloc[40]) == 1


def test_bearish_liquidity_sweep():
    n = 40
    close = np.full(n, 100.0)
    op = np.full(n, 100.0)
    hi = np.full(n, 100.0)   # prior 10-bar high = 100
    lo = np.full(n, 99.0)
    # sweep bar: wick above 100, close back below
    hi[25], lo[25], close[25], op[25] = 105.0, 98.0, 99.0, 100.0
    fr = _frame(op, hi, lo, close, vol=np.full(n, 1000.0))
    out = candle_story_engine(fr, sweep_lookback=10)
    assert int(out["bear_liquidity_sweep"].iloc[25]) == 1
    assert int(out["liquidity_sweep"].iloc[25]) == -1


def _compression_breakout_frame():
    """Tight shrinking-range zone (compression) then a range-expansion breakout."""
    n = 160
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = np.full(n, 100.0)
    op = np.full(n, 100.0)
    hi = np.full(n, 100.5)
    lo = np.full(n, 99.5)
    # wide pre-compression bars 0..79 (range 10)
    for b in range(0, 80):
        c = 80.0 + b * 0.25
        close[b], op[b] = c, c
        hi[b], lo[b] = c + 5.0, c - 5.0
    # tight shrinking zone 80..123
    for b in range(80, 124):
        r = 0.3 + 2.7 * (1.0 - (b - 80) / 44.0)  # 3.0 -> 0.3
        hi[b] = 100.0 + r / 2.0
        lo[b] = 100.0 - r / 2.0
        close[b] = 100.0
        op[b] = 100.0
    # breakout attempt bar 124 (range expansion, close above zone)
    op[124], hi[124], lo[124], close[124] = 100.0, 112.0, 99.0, 110.0
    # acceptance bar 125 (extends beyond attempt high)
    op[125], hi[125], lo[125], close[125] = 112.0, 114.0, 109.0, 113.0
    # trailing bars
    for b in range(126, n):
        c = 113.0 + (b - 125) * 0.1
        close[b], op[b] = c, c
        hi[b], lo[b] = c + 1.0, c - 1.0
    fr = pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close}, index=idx)
    return fr


def test_compression_breakout_accepted():
    fr = _compression_breakout_frame()
    out = candle_story_engine(fr, compression_short=3, compression_long=10)
    # compression arms once the shrinking zone makes the short-range average
    # materially smaller than the longer lookback (near the end of the zone)
    assert int(out["compression_active"].iloc[120]) == 1
    # breakout attempt at bar 124
    assert int(out["breakout_attempt"].iloc[124]) == 1
    # accepted (sticky code 2) appears from bar 125 onward
    assert int(out["breakout_story_event"].iloc[125]) == 2
    assert int(out["breakout_story_event"].iloc[140]) == 2


def test_uptrend_triggers_bull_control_signal():
    n = 160
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = np.empty(n)
    op = np.empty(n)
    hi = np.empty(n)
    lo = np.empty(n)
    # growth rate ramps up over time so bullish evidence climbs through the
    # signal threshold (and keeps improving by >= minimum_signal_change).
    price = 100.0
    for b in range(n):
        g = 0.002 + 0.0002 * b
        op[b] = price
        price *= (1.0 + g)
        close[b] = price
        hi[b] = close[b] + 0.002
        lo[b] = op[b] - 0.002
    fr = pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close}, index=idx)
    out = candle_story_engine(fr)
    # strong persistent uptrend -> bull evidence reaches control territory
    assert int(out["bull_control"].sum()) >= 1
    assert float(out["bull_evidence"].max()) >= 65.0
    # the hysteresis armed state is reached once evidence climbs past threshold
    assert int(out["bull_signal_state"].sum()) >= 1
    # no bear control in a clean uptrend
    assert int(out["bear_signal_state"].sum()) == 0


def test_short_frame_does_not_crash():
    op = np.array([100.0, 101.0, 102.0])
    cl = np.array([101.0, 102.0, 103.0])
    hi = np.array([102.0, 103.0, 104.0])
    lo = np.array([99.0, 100.0, 101.0])
    fr = _frame(op, hi, lo, cl)
    out = candle_story_engine(fr)
    assert len(out) == 3
    assert list(out.columns) == _COLUMNS


def test_parameter_validation():
    op = np.full(20, 100.0)
    cl = np.full(20, 101.0)
    hi = np.full(20, 102.0)
    lo = np.full(20, 99.0)
    fr = _frame(op, hi, lo, cl)
    with pytest.raises(ValueError):
        candle_story_engine(fr, sequence_length=1)  # below minval 2
    with pytest.raises(ValueError):
        candle_story_engine(fr, signal_threshold=99.0)  # above maxval 95
