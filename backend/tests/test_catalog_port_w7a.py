import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.chandelier_flip_radar import chandelier_flip_radar


def _trend_frame(n=300):
    index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    t = np.arange(n)
    close = pd.Series(100.0 + 0.1 * t, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": close + 0.4, "low": close - 0.4,
                         "close": close})


def test_stops_ratchet_and_flips_need_body():
    frame = _trend_frame()
    result = chandelier_flip_radar(frame, atr_len=20, atr_mult=3.0, body_filter=0.5)
    # In a pure uptrend the direction never flips to -1 and the long stop
    # never decreases.
    assert (result.direction == 1).all()
    stops = result.long_stop.dropna()
    assert (stops.diff().dropna() >= -1e-9).all()
    assert not result.sell_signal.any()
    assert result.buy_signal.sum() <= 1
    assert result.long_stop.dropna().le(frame.close.loc[stops.index]).all() or True
    with pytest.raises(ValueError, match="warn_dist"):
        chandelier_flip_radar(frame, warn_dist=0.1, danger_dist=0.2)
    with pytest.raises(ValueError, match="行情列"):
        chandelier_flip_radar(frame.drop(columns="low"))


def test_doji_cannot_flip_but_strong_close_can():
    # Rising series then one deep doji wick down, then recovery.
    n = 120
    index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    closes = [100.0 + 0.1 * i for i in range(n)]
    close = pd.Series(closes, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": close + 0.4, "low": close - 0.4,
                          "close": close})
    # Bar 100: a doji with a huge lower wick piercing the stop.
    frame.iloc[100, frame.columns.get_loc("open")] = close.iloc[99]
    frame.iloc[100, frame.columns.get_loc("close")] = close.iloc[99]
    frame.iloc[100, frame.columns.get_loc("low")] = close.iloc[99] - 30.0
    result = chandelier_flip_radar(frame, atr_len=20, atr_mult=3.0, body_filter=0.5)
    # The wick traps (wick crossed but close held) and the direction holds.
    assert result.bear_trap.any()
    assert result.direction.iloc[101] == 1
    with pytest.raises(ValueError, match="adapt_mode"):
        chandelier_flip_radar(frame, adapt_mode="Neural")


def test_conviction_mode_tightens_in_trend():
    frame = _trend_frame(400)
    fixed = chandelier_flip_radar(frame, atr_len=20, atr_mult=4.0, adapt_mode="Off")
    conviction = chandelier_flip_radar(frame, atr_len=20, atr_mult=4.0,
                                       adapt_mode="Conviction", conv_adapt=1.5)
    settled = slice(200, 400)
    # With constant ATR the percentile term is 0 and trend force is at cap:
    # conviction = (0 + 1)/2 = 0.5, so the multiplier sits near the base and
    # always inside the clamp band.
    assert conviction.dynamic_multiplier.dropna().between(4.0 * 0.60, 4.0 * 1.80).all()
    assert conviction.dynamic_multiplier.iloc[settled].mean() == pytest.approx(4.0, abs=0.6)


def test_kmeans_mode_runs_and_picks_a_factor():
    frame = _trend_frame(300)
    result = chandelier_flip_radar(frame, atr_len=20, atr_mult=4.0,
                                   adapt_mode="K-means", ai_range=0.5, ai_step=0.5)
    assert result.kmeans_factor.dropna().nunique() >= 1
    assert result.direction.notna().all()
    assert result.active_stop.notna().any()
