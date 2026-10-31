import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.ad_current_day_poc_vah_val import ad_current_day_poc_vah_val
from app.catalog_indicators.adaptive_cycle_detector import adaptive_cycle_detector
from app.catalog_indicators.bullish_lines_four_state import bullish_lines_four_state
from app.catalog_indicators.candle_pressure_index import candle_pressure_index
from app.catalog_indicators.capital41_confluence_core import capital41_confluence_core


def test_capital41_confluence_core_needs_all_four_votes():
    closes = [9.2] * 10 + [9.3]
    frame = pd.DataFrame({
        "open": closes,
        "high": [10.0] * 10 + [9.4],
        "low": [9.0] * 10 + [8.5],
        "close": closes,
        "volume": [10.0] * 10 + [30.0],
    })
    result = capital41_confluence_core(frame, ma_length=5, rsi_length=2, wr_length=3,
                                       sweep_lookback=5, volume_lookback=5,
                                       min_volume_ratio=1.2, signal_mode="Loose")
    assert result.prior_low.iloc[10] == 9.0
    assert result.volume_ratio.iloc[10] == pytest.approx(30.0 / 14.0)  # SMA includes this bar
    assert result.bull_sweep.iloc[10]
    assert not result.long_signal.iloc[10]  # RSI stays flat, so the oscillator never confirms
    assert not result.bear_sweep.any()
    with pytest.raises(ValueError, match="signal_mode"):
        capital41_confluence_core(frame, signal_mode="Turbo")
    with pytest.raises(ValueError, match="volume"):
        capital41_confluence_core(frame.drop(columns="volume"))


def test_bullish_lines_four_state_requires_configuration():
    rising = pd.DataFrame({"high": np.arange(60) + 2.0, "low": np.arange(60) + 0.0,
                           "close": np.arange(60) + 1.0})
    unset = bullish_lines_four_state(rising)
    assert unset.state.eq("Unconfigured").all()
    assert unset.fast_smma.isna().all()

    configured = bullish_lines_four_state(rising, fast_length=2, mid_warmer_length=3,
                                          mid_cooler_length=4, slow_length=5)
    assert configured.fast_smma.iloc[-1] > configured.slow_smma.iloc[-1]
    assert configured.is_bullish.iloc[-1] and not configured.mixed.iloc[-1]
    assert configured.state.iloc[-1] == "Bullish"
    assert configured.new_bullish.iloc[4] and configured.new_bullish.sum() == 1

    falling = pd.DataFrame({"high": -np.arange(60) + 62.0, "low": -np.arange(60) + 60.0,
                            "close": -np.arange(60) + 61.0})
    assert bullish_lines_four_state(falling, fast_length=2, mid_warmer_length=3,
                                    mid_cooler_length=4, slow_length=5).state.iloc[-1] == "Bearish"
    with pytest.raises(ValueError, match="source"):
        bullish_lines_four_state(rising, source="OHLC4")


def test_candle_pressure_index_weighs_close_body_and_volume():
    frame = pd.DataFrame({
        "open": [10.0] * 6,
        "high": [11.0] * 6,
        "low": [9.0] * 6,
        "close": [11.0] * 6,
        "volume": [10.0] * 6,
    })
    result = candle_pressure_index(frame, smoothing_length=2, volume_lookback=5,
                                   spike_threshold=0.4)
    assert result.close_location.iloc[1] == pytest.approx(1.0)
    assert result.body_ratio.iloc[1] == pytest.approx(0.5)
    assert result.volume_rank.iloc[4] == pytest.approx(1.0)  # first full lookback
    assert result.raw_pressure.iloc[4] == pytest.approx(0.5)  # rank is complete from here
    assert result.spike_bull.iloc[4] and not result.spike_bear.any()
    assert result.cpi_acceleration.iloc[2] == pytest.approx(result.cpi.iloc[2] - result.cpi.iloc[1])
    silent = candle_pressure_index(frame.assign(volume=0.0), smoothing_length=2, volume_lookback=5)
    assert silent.volume_rank.isna().all() and silent.cpi.isna().all()
    with pytest.raises(ValueError, match="spike_threshold"):
        candle_pressure_index(frame, spike_threshold=1.5)
    with pytest.raises(ValueError, match="行情列"):
        candle_pressure_index(frame.drop(columns="volume"))


def test_current_day_profile_puts_poc_on_the_heaviest_bar():
    index = pd.date_range("2026-01-01", periods=96, freq="15min", tz="UTC")
    close = np.full(96, 100.0)
    volume = np.zeros(96)
    volume[40] = 1000.0
    high = close + 0.1
    low = close - 0.1
    high[40], low[40] = 100.05, 99.95  # the heavy bar covers only the middle of the day's range
    frame = pd.DataFrame({
        "open": close, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=index)
    result = ad_current_day_poc_vah_val(frame, rows=32, value_area_percent=70.0)
    assert result.bars_in_profile.iloc[0] == 1
    assert result.bars_in_profile.iloc[-1] == 96
    assert pd.isna(result.poc.iloc[39])  # no volume yet, so no profile to draw
    assert result.poc.iloc[40] == pytest.approx(100.0, abs=0.05)
    assert result.poc.iloc[95] == pytest.approx(100.0, abs=0.05)
    assert result.val.iloc[95] <= result.poc.iloc[95] <= result.vah.iloc[95]
    # A new UTC day restarts the profile.
    second_day = frame.copy()
    second_day.index = index + pd.Timedelta(days=1)
    combined = pd.concat([frame, second_day])
    rolled = ad_current_day_poc_vah_val(combined)
    assert rolled.bars_in_profile.iloc[96] == 1
    with pytest.raises(ValueError, match="rows"):
        ad_current_day_poc_vah_val(frame, rows=10)
    with pytest.raises(ValueError, match="session_end"):
        ad_current_day_poc_vah_val(frame, session_start="1600", session_end="0930")


def test_adaptive_cycle_detector_stays_inside_its_bounds():
    index = pd.date_range("2026-01-01", periods=600, freq="15min", tz="UTC")
    rng = np.random.default_rng(5)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.4, 600)), index=index)
    frame = pd.DataFrame({"open": close, "high": close + 0.6, "low": close - 0.6,
                          "close": close, "volume": 10.0})
    result = adaptive_cycle_detector(frame, cycle_lookback=50, min_period=8, max_period=40,
                                     channel_length=5, average_length=9, adx_length=5)
    assert result.cycle_period.dropna().between(8, 40).all()
    assert result.adapted_wavetrend_length.dropna().between(5, 50).all()
    assert result.suggested_stochastic_length.dropna().between(8, 30).all()
    assert result.market_mode.isin(["Trend", "Expansion", "Compression", "Normal"]).all()
    assert result.crossing_count.iloc[-1] <= 50
    with pytest.raises(ValueError, match="min_period"):
        adaptive_cycle_detector(frame, min_period=40, max_period=20)
    with pytest.raises(ValueError, match="trend_threshold"):
        adaptive_cycle_detector(frame, trend_threshold=99)
