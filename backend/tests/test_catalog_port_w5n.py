import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.stoch_rsi_advanced import stoch_rsi_advanced


def _oscillating_frame(n=300):
    rng = np.random.default_rng(7)
    t = np.arange(n)
    close = pd.Series(100 + 8 * np.sin(t / 5.0) + rng.normal(0, 0.3, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.2,
                          "low": np.minimum(open_, close) - 0.2, "close": close})
    frame.index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    return frame


def test_core_stochrsi_matches_hand_computed_chain():
    frame = _oscillating_frame()
    result = stoch_rsi_advanced(frame, rsi_len=5, stoch_len=6, k_len=2, d_len=2,
                                smoothing="SMA")
    from app.catalog_indicators._pine import rsi as pine_rsi, stoch as pine_stoch
    r = pine_rsi(frame["close"], 5)
    raw = pine_stoch(r, r, r, 6)
    k = raw.rolling(2).mean()
    d = k.rolling(2).mean()
    valid = d.dropna().index
    assert np.allclose(result.k.loc[valid], k.loc[valid])
    assert np.allclose(result.d.loc[valid], d.loc[valid])
    assert result.k.dropna().between(0, 100).all()
    assert result.bull_cross.any() and result.bear_cross.any()
    with pytest.raises(ValueError, match="smoothing"):
        stoch_rsi_advanced(frame, smoothing="T5")


def test_extreme_gate_and_context_counter_trend():
    frame = _oscillating_frame()
    strict = stoch_rsi_advanced(frame, rsi_len=5, stoch_len=6, k_len=2, d_len=2,
                                smoothing="SMA", use_extreme_only=True)
    loose = stoch_rsi_advanced(frame, rsi_len=5, stoch_len=6, k_len=2, d_len=2,
                               smoothing="SMA", use_extreme_only=False)
    # Requiring the OB/OS history can only reduce the extreme count.
    assert loose.bull_extreme.sum() >= strict.bull_extreme.sum()
    assert loose.bear_extreme.sum() >= strict.bear_extreme.sum()
    # Strong + weak partition the extremes exactly.
    assert (strict.bull_extreme == (strict.bull_strong | strict.bull_weak)).all()
    assert (strict.bear_extreme == (strict.bear_strong | strict.bear_weak)).all()
    with pytest.raises(ValueError, match="rsi_len"):
        stoch_rsi_advanced(frame, rsi_len=0)


def test_quality_and_sentiment_bounded():
    frame = _oscillating_frame(n=400)
    result = stoch_rsi_advanced(frame, rsi_len=5, stoch_len=6, k_len=2, d_len=2,
                                smoothing="SMA", ob_level=70.0, os_level=30.0)
    assert result.sentiment_score.dropna().between(-100, 100).all()
    assert result.bull_quality.dropna().between(0, 100).all()
    assert result.bear_quality.dropna().between(0, 100).all()
    # Quality equals its definition on the last bar: depth*50 + context*25 + stall-free*25.
    k = result.k
    from app.catalog_indicators._pine import lowest as p_lowest, highest as p_highest
    depth_bull = min(max((30.0 - p_lowest(k, 5).iloc[-1]) / 30.0, 0.0), 1.0)
    ctx_bull = bool(result.context_line.iloc[-1] > 50.0)
    expected = min(100.0, depth_bull * 50.0 + (25.0 if ctx_bull else 0.0)
                   + (0.0 if result.stall_bear.iloc[-1] else 25.0))
    assert result.bull_quality.iloc[-1] == pytest.approx(expected)
    with pytest.raises(ValueError, match="os_level"):
        stoch_rsi_advanced(frame, os_level=60.0)


def test_stall_flags_need_flat_price_and_big_oscillator_move():
    frame = _oscillating_frame()
    result = stoch_rsi_advanced(frame, rsi_len=5, stoch_len=6, k_len=2, d_len=2,
                                smoothing="SMA", stall_osc_min=0.0, stall_flat_atr=100.0)
    # With a 100-ATR "flat" tolerance and zero oscillator threshold every bar stalls.
    warmed = result.iloc[30:]
    assert warmed.stall_bull.any() or warmed.stall_bear.any()
    off = stoch_rsi_advanced(frame, rsi_len=5, stoch_len=6, k_len=2, d_len=2,
                             smoothing="SMA", use_stall=False)
    assert not off.stall_bull.any() and not off.stall_bear.any()
