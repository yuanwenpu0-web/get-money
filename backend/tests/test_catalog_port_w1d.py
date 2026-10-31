import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.capital41_fusion_scalp import capital41_fusion_scalp
from app.catalog_indicators.capital41_pdh_pdl import capital41_pdh_pdl
from app.catalog_indicators.capital41_regime_matrix import capital41_regime_matrix
from app.catalog_indicators.cvd_bias import cvd_bias
from app.catalog_indicators.relative_strength_line import relative_strength_line


def test_capital41_fusion_scalp_needs_the_expansion_gate():
    close = np.array([100.0] * 20 + [101.0, 102.0, 103.0])
    volume = np.array([10.0] * 20 + [40.0, 10.0, 10.0])
    high = close + 1.0
    low = close - 1.0
    high[20], low[20] = 104.0, 98.0  # the spike bar also expands the range
    frame = pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})
    result = capital41_fusion_scalp(frame, ema_length=5, rsi_length=3, wr_length=3,
                                    atr_length=3, atr_average_length=5, volume_length=5,
                                    atr_expansion_multiplier=1.1)
    assert result.trend_long.iloc[20] and result.expansion_ok.iloc[20]
    # The signal is the four-way conjunction; the expansion gate is what opens here.
    assert result.long_signal.equals(result.trend_long & result.rsi_long & result.wr_long
                                    & result.expansion_ok)
    assert not result.long_signal.iloc[21]  # volume falls back, expansion gate closes
    quiet = capital41_fusion_scalp(frame.assign(volume=10.0), ema_length=5, rsi_length=3,
                                   wr_length=3, atr_length=3, atr_average_length=5,
                                   volume_length=5)
    assert not quiet.expansion_ok.any() and not quiet.long_signal.any()
    with pytest.raises(ValueError, match="volume_multiplier"):
        capital41_fusion_scalp(frame, volume_multiplier=5.0)


def test_cvd_bias_normalises_and_stays_bounded():
    close = np.array([100.0, 101.0, 102.0, 101.0, 100.0, 101.0, 102.0, 103.0,
                      102.0, 103.0, 104.0, 105.0])
    frame = pd.DataFrame({"high": close + 0.5, "low": close - 1.0, "close": close,
                          "volume": 10.0})
    result = cvd_bias(frame, accumulation_window=10, normalization_lookback=20,
                      pivot_left=1, pivot_right=1)
    location = (close - (close - 1.0)) / 1.5
    signed = (2.0 * location - 1.0) * 10.0
    assert result.cvd_raw.iloc[-1] == pytest.approx(signed[-10:].sum())  # window of 10
    assert result.signed_volume.iloc[0] == pytest.approx(signed[0])
    assert result.cvd.abs().dropna().le(1.0).all()
    assert result.cvd_roc.iloc[-1] == pytest.approx(result.cvd.iloc[-1] - result.cvd.iloc[-6])  # roc_length 5
    with pytest.raises(ValueError, match="accumulation_window"):
        cvd_bias(frame, accumulation_window=2)
    with pytest.raises(ValueError, match="volume"):
        cvd_bias(frame.drop(columns="volume"))


def test_relative_strength_line_flags_outperformance():
    index = pd.date_range("2026-01-01", periods=80, freq="1h", tz="UTC")
    close = pd.Series(np.linspace(100, 180, 80), index=index)
    frame = pd.DataFrame({"close": close, "bench_close": 100.0}, index=index)
    result = relative_strength_line(frame, mansfield_length=10, trend_length=10, min_gap=5)
    assert result.rs.iloc[-1] == pytest.approx(1.8)
    assert result.mansfield_rs.iloc[-1] > 0
    assert result.long_regime.iloc[-1]
    assert result.loc[result.long_entry, "long_regime"].all()
    assert result.fire_up.sum() <= result.long_entry.sum()
    flat = relative_strength_line(pd.DataFrame({"close": close, "bench_close": close}), min_gap=5)
    assert not flat.long_regime.any() and not flat.short_regime.any()
    with pytest.raises(ValueError, match="display_mode"):
        relative_strength_line(frame, display_mode="Ratio")


def test_capital41_regime_matrix_uses_the_regime_specific_average():
    index = pd.date_range("2026-01-01", periods=400, freq="1h", tz="UTC")
    rng = np.random.default_rng(7)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.6, 400)), index=index)
    frame = pd.DataFrame({"high": close + 0.8, "low": close - 0.8, "close": close})
    result = capital41_regime_matrix(frame, base_ema_length=20, hurst_length=40, adx_length=5,
                                     volatility_length=10, pressure_smoothing=2)
    assert set(result.regime_code.dropna().unique()) <= {1, 2, 3, 4}
    assert result.regime.isin(["TREND", "EXP", "COMP", "MR"]).all()
    last = result.iloc[-1]
    expected_stop = last.adaptive_ema - last.atr * last.stop_multiplier
    assert last.long_stop == pytest.approx(expected_stop)
    assert last.short_stop == pytest.approx(last.adaptive_ema + last.atr * last.stop_multiplier)
    assert 0 <= last.pressure <= 100
    assert result.pressure.dropna().between(0, 100).all()
    with pytest.raises(ValueError, match="hurst_length"):
        capital41_regime_matrix(frame, hurst_length=10)
    with pytest.raises(ValueError, match="trend_hurst"):
        capital41_regime_matrix(frame, trend_hurst=0.4)


def test_capital41_pdh_pdl_tracks_previous_period_extremes():
    index = pd.date_range("2026-01-01", periods=96 * 3, freq="15min", tz="UTC")
    close = np.full(96 * 3, 100.0)
    high = np.full(96 * 3, 100.5)
    low = np.full(96 * 3, 99.5)
    high[10] = 105.0
    low[20] = 95.0
    frame = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=index)
    frame.loc[index[96]:, "close"] = 106.0  # day two closes above yesterday's high
    result = capital41_pdh_pdl(frame, touch_tolerance=0.1)
    assert pd.isna(result.previous_day_high.iloc[0])
    assert result.previous_day_high.iloc[96] == 105.0
    assert result.previous_day_low.iloc[96] == 95.0
    assert result.previous_day_mid.iloc[96] == pytest.approx(100.0)
    assert result.break_day_high.iloc[96]
    assert result.position.iloc[96] == "ABOVE"
    with pytest.raises(ValueError, match="touch_tolerance"):
        capital41_pdh_pdl(frame, touch_tolerance=2.0)
