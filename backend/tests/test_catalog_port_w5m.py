import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.mtf_strategy_visual import mtf_strategy_visual


def _trend_frame(n=96 * 30):
    # Accelerating uptrend: EMAs stack fast-over-slow on every timeframe.
    index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    t = np.arange(n)
    close = pd.Series(100.0 + 0.05 * t + 0.001 * t * t, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.3,
                          "low": np.minimum(open_, close) - 0.3, "close": close})
    return frame


def test_htf_bias_tracks_the_trend_and_gates_signals():
    frame = _trend_frame()
    result = mtf_strategy_visual(frame, htf_minutes=60, mtf_minutes=15)
    settled = result.iloc[300:]
    assert (settled.htf_bias == 1).all()  # clean uptrend stacks bullish
    assert settled.ltf_bullish.all()
    assert not settled.htf_bearish.any()
    # The full signal needs a cloud cross; a persistent trend only crosses once.
    assert result.buy_signal.sum() <= 3
    assert not result.sell_signal.any()
    assert not result.sideways_market.iloc[300:].any()
    with pytest.raises(ValueError, match="整数倍"):
        mtf_strategy_visual(frame, mtf_minutes=7)
    with pytest.raises(ValueError, match="行情列"):
        mtf_strategy_visual(frame.drop(columns="close"))


def test_sideways_market_suppresses_everything():
    rng = np.random.default_rng(6)
    n = 96 * 20
    index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    close = pd.Series(100 + rng.normal(0, 0.05, n).cumsum(), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.05,
                          "low": np.minimum(open_, close) - 0.05, "close": close})
    result = mtf_strategy_visual(frame, htf_minutes=60, mtf_minutes=15,
                                 atr_threshold=5.0)  # unreachable ATR gate
    assert not result.buy_signal.any()
    assert not result.sell_signal.any()
    # With no HTF stack (random walk) the sideways state covers most bars.
    assert result.sideways_market.mean() > 0.5
    with pytest.raises(ValueError, match="atr_threshold"):
        mtf_strategy_visual(frame, atr_threshold=0.0)


def test_parametric_convergence_threshold_shapes_the_flag():
    frame = _trend_frame()
    tight = mtf_strategy_visual(frame, convergence_threshold=0.01, mtf_minutes=15)
    loose = mtf_strategy_visual(frame, convergence_threshold=50.0, mtf_minutes=15)
    assert loose.ema_converged.iloc[-1]
    assert not tight.ema_converged.iloc[-1]
    with pytest.raises(ValueError, match="ema1_period"):
        mtf_strategy_visual(frame, ema1_period=0)
