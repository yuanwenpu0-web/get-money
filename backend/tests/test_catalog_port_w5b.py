import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.market_pressure_scale import market_pressure_scale


def _frame(closes, volume=None):
    close = pd.Series(closes, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    # Keep open inside [low, high] like real OHLC: extend the range around the
    # wider of open/close instead of blindly around the close.
    top = np.maximum(open_, close)
    bottom = np.minimum(open_, close)
    frame = pd.DataFrame({"open": open_, "high": top + 0.4, "low": bottom - 0.4,
                          "close": close})
    frame["volume"] = 10.0 if volume is None else volume
    return frame


def test_norm100_style_components_on_a_deterministic_range():
    # Flat 100.0 range for 8 bars then a dip to 98 keeps ATR and BB width known.
    closes = [100.0] * 8 + [98.0, 98.2, 98.4]
    frame = _frame(closes)
    result = market_pressure_scale(frame, norm_len=20, atr_len=1, bb_len=3, bb_mult=2.0,
                                   vol_len=3, sr_len=5, bias_type="SMA",
                                   bias_fast_len=2, bias_slow_len=3, smooth_len=1,
                                   smooth_type="SMA", dmi_len=2)
    # range_position = 100 * (close - lowest(low,5)) / (highest(high,5) - lowest(low,5)).
    # Bar 10: open 98.2 close 98.4 -> low 97.8; the 5-bar window still sees
    # bar 8's low 97.6 and the 100.0 closes (range high 100.4).
    assert result.range_position.iloc[10] == pytest.approx(
        100.0 * (98.4 - 97.6) / (100.4 - 97.6))
    assert result.range_position.iloc[:4].isna().all()  # sr_len warm-up
    assert result.range_position.iloc[4:8].eq(50.0).all()  # flat range sits mid-range
    # All pressures stay inside 0..100 once warmed up.
    for column in ("setup_pressure", "impulse_pressure", "move_strength"):
        assert result[column].dropna().between(0.0, 100.0).all()
    # Bar 9 (close 98.2): fast SMA2 = 98.1 < slow SMA3 = 98.733 -> bearish.
    assert result.bear_bias.iloc[9] and not result.bull_bias.iloc[9]
    assert result.bias_text.iloc[9] == "Bearish"
    # The recovery at the last bar flips it back to bullish (98.3 > 98.2).
    assert result.bull_bias.iloc[-1] and result.bias_text.iloc[-1] == "Bullish"
    with pytest.raises(ValueError, match="norm_len"):
        market_pressure_scale(frame, norm_len=10)
    with pytest.raises(ValueError, match="行情列"):
        market_pressure_scale(frame.drop(columns="volume"))


def test_volume_free_feed_rescales_the_weights():
    closes = [100.0 + 0.5 * step for step in range(40)]
    frame = _frame(closes, volume=np.zeros(40))
    quiet = market_pressure_scale(frame, norm_len=20, atr_len=2, bb_len=5, vol_len=5,
                                  sr_len=10, smooth_len=1, smooth_type="SMA",
                                  bias_type="SMA", dmi_len=2)
    loud = market_pressure_scale(_frame(closes), norm_len=20, atr_len=2, bb_len=5,
                                 vol_len=5, sr_len=10, smooth_len=1, smooth_type="SMA",
                                 bias_type="SMA", dmi_len=2)
    # Both variants produce pressures; the rescaled no-volume version compresses
    # the volume legs out, so values differ from the volume-fed run.
    assert quiet.setup_pressure.dropna().shape[0] > 0
    assert not quiet.setup_pressure.dropna().equals(loud.setup_pressure.dropna())
    # Constant volume => vol_rel 1.0 => volume_pressure constant 50 in a
    # well-peppered window; the loud feed's impulse uses the 0.15 leg.
    assert quiet.impulse_pressure.dropna().between(0.0, 100.0).all()


def test_phase_labels_and_flag_families():
    rng = np.random.default_rng(9)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    frame = _frame(closes)
    result = market_pressure_scale(frame, norm_len=100)
    assert set(result.phase.unique()) <= {"Sideways / Chop", "Impulse Running",
                                          "Sideways / Quiet", "Neutral"}
    assert set(result.bias_text.unique()) <= {"Bullish", "Bearish", "Flat"}
    # Chop and valid-setup zones are complementary mid-range/extreme partitions
    # of the chop filter (chop additionally needs ADX/DI agreement).
    both = result.chop_market & result.valid_setup_zone
    assert not both.any()
    # Reversal scores are bounded combinations of bounded inputs.
    assert result.reversal_long_score.dropna().between(-1e-9, 100.0).all()
    assert not (result.reversal_long & ~result.valid_setup_zone).any()
    with pytest.raises(ValueError, match="smooth_len"):
        market_pressure_scale(frame, smooth_len=0)
    with pytest.raises(ValueError, match="bias_type"):
        market_pressure_scale(frame, bias_type="VWAP")
