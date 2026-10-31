import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.exhaustion_scanner import exhaustion_scanner


def _rally_then_stall():
    # 150-bar rally then 20 bars stalling just below the highs with upper wicks.
    n = 180
    closes = [100.0 + 0.3 * i for i in range(150)]
    closes += [145.0 + (0.4 if i % 2 else 0.0) for i in range(30)]
    close = pd.Series(closes, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    highs = np.maximum(open_, close) + 0.3
    lows = np.minimum(open_, close) - 0.3
    # Stalling bars leave long upper wicks and close below the high.
    highs[150:] = close.iloc[150:].to_numpy() + 3.0
    frame = pd.DataFrame({"open": open_, "high": highs, "low": lows, "close": close})
    frame["volume"] = 100.0
    frame.index = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return frame


def test_scores_bounded_and_thresholds_clamped():
    frame = _rally_then_stall()
    result = exhaustion_scanner(frame, preset="Commodities", structure_len=60,
                                regime_slow=40)
    assert result.top_score.dropna().between(0, 100).all()
    assert result.bot_score.dropna().between(0, 100).all()
    assert result.top_threshold.dropna().between(20, 80).all()
    assert set(result.regime.unique()) <= {"Trend ↑", "Trend ↓", "Range"}
    assert set(result.top_kind.unique()) <= {
        "", "CONTINUATION RISK", "TREND EXHAUSTION", "RANGE FADE"}
    with pytest.raises(ValueError, match="preset"):
        exhaustion_scanner(frame, preset="Bonds")
    with pytest.raises(ValueError, match="行情列"):
        exhaustion_scanner(frame.drop(columns="volume"))


def test_signal_modes_change_the_threshold():
    frame = _rally_then_stall()
    early = exhaustion_scanner(frame, signal_mode="Early", structure_len=60,
                               regime_slow=40)
    conservative = exhaustion_scanner(frame, signal_mode="Conservative",
                                      structure_len=60, regime_slow=40)
    valid = early.top_threshold.notna()
    assert (conservative.top_threshold[valid] > early.top_threshold[valid]).all()
    with pytest.raises(ValueError, match="signal_mode"):
        exhaustion_scanner(frame, signal_mode="Aggressive")
    with pytest.raises(ValueError, match="confirm_mode"):
        exhaustion_scanner(frame, confirm_mode="Never")


def test_presets_change_the_score_mix():
    frame = _rally_then_stall()
    stocks = exhaustion_scanner(frame, preset="Stocks", structure_len=60,
                                regime_slow=40)
    crypto = exhaustion_scanner(frame, preset="Crypto", structure_len=60,
                                regime_slow=40)
    # Different weights produce different scores on the same data.
    assert not stocks.top_score.dropna().equals(crypto.top_score.dropna())
    assert stocks.stretch_top.dropna().equals(crypto.stretch_top.dropna())
    with pytest.raises(ValueError, match="adx_trend"):
        exhaustion_scanner(frame, adx_trend=-1.0)


def test_signals_respect_min_gap():
    frame = _rally_then_stall()
    tight = exhaustion_scanner(frame, min_signal_gap=0, structure_len=60,
                               regime_slow=40)
    wide = exhaustion_scanner(frame, min_signal_gap=100, structure_len=60,
                              regime_slow=40)
    # A wide gap can only reduce the signal count.
    assert wide.top_signal.sum() <= tight.top_signal.sum()
    assert wide.bot_signal.sum() <= tight.bot_signal.sum()
    with pytest.raises(ValueError, match="min_signal_gap"):
        exhaustion_scanner(frame, min_signal_gap=-1)
