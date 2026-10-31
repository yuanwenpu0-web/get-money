"""Ports batch W2C: stock_stats, capital41_divergence_detector, atr_advanced,
sr_zones_plasmazero, smart_money_concepts."""
import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.atr_advanced import atr_advanced
from app.catalog_indicators.capital41_divergence_detector import (
    capital41_divergence_detector,
)
from app.catalog_indicators.smart_money_concepts import smart_money_concepts
from app.catalog_indicators.sr_zones_plasmazero import sr_zones_plasmazero
from app.catalog_indicators.stock_stats import stock_stats


def _daily_frame(days, high, low, close, volume):
    index = pd.date_range("2024-01-01", periods=days, freq="1D", tz="UTC")
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close,
                         "volume": volume}, index=index)


def test_stock_stats_adr_lod_and_buzz_hand_computed():
    # 30 constant days: high/low = 1.1 -> ADR% = 10; TR = 10 -> ATR = 10;
    # lod = 100 * (close - low) / atr = 100 * 5 / 10 = 50.
    days = 30
    high = np.full(days, 110.0)
    low = np.full(days, 100.0)
    close = np.full(days, 105.0)
    volume = np.full(days, 100.0)
    frame = _daily_frame(days, high, low, close, volume)
    result = stock_stats(frame, adr_length=20, atr_length=14)

    # Day 20 (index 19) is the first day whose ADR% is defined; each bar reads
    # the previous completed day, so the first defined value shows on day 21.
    assert result.adr_percent.iloc[:20].isna().all()
    assert result.adr_percent.iloc[20] == pytest.approx(10.0)
    assert (result.adr_percent.iloc[20:] - 10.0).abs().max() < 1e-9

    assert result.atr_daily.iloc[:14].isna().all()
    assert result.atr_daily.iloc[14] == pytest.approx(10.0)
    assert result.low_of_day_distance.iloc[14] == pytest.approx(50.0)
    assert result.low_of_day_distance.iloc[:14].isna().all()

    # Volume buzz needs a 100-day average, so it stays NaN on a 30-day feed.
    assert result.volume_buzz.isna().all()
    # 52-week high needs 52 completed weeks.
    assert result.high_52_week.isna().all()
    # On daily bars the closing clock is midnight, so the elapsed fraction is 0
    # and Pine's crypto run rate is undefined.
    assert result.run_rate_percent.isna().all()
    assert result.volume_today.eq(100.0).all()


def test_stock_stats_volume_buzz_and_52_week_hand_computed():
    # 371 days = exactly 53 weeks starting Monday 2024-01-01.
    days = 371
    index = pd.date_range("2024-01-01", periods=days, freq="1D", tz="UTC")
    high = 100.0 + np.arange(days, dtype=float)
    frame = pd.DataFrame({"open": high, "high": high, "low": high - 1.0,
                          "close": high, "volume": np.full(days, 100.0)}, index=index)
    # Day 101 doubles the volume: its buzz is (200/100 - 1) * 100 = 100 and every
    # later bar reads that value from the previous completed day.
    frame.iloc[100, frame.columns.get_loc("volume")] = 200.0
    result = stock_stats(frame)

    # Bar 100 reads day 100's buzz (the average exists there): exactly 0. Bar 101
    # reads day 101's buzz (200/101 - 1) * 100, which stays until the 200-volume
    # day rolls out of the 100-day window; day 201 is back to a flat 0.
    assert result.volume_buzz.iloc[:100].isna().all()
    assert result.volume_buzz.iloc[100] == pytest.approx(0.0)
    assert result.volume_buzz.iloc[101] == pytest.approx((200.0 / 101.0 - 1.0) * 100.0)
    assert result.volume_buzz.iloc[102] == pytest.approx((100.0 / 101.0 - 1.0) * 100.0)
    assert result.volume_buzz.iloc[200] == pytest.approx((100.0 / 101.0 - 1.0) * 100.0)
    assert result.volume_buzz.iloc[201] == pytest.approx(0.0)

    # Week k covers day indexes 7k..7k+6, its high is 100 + 7k + 6; bars in week k
    # (k >= 52) read the previous completed week's high = 100 + 7k - 1.
    assert result.high_52_week.iloc[:364].isna().all()
    # All of week 52 (bars 364..370) reads week 51's high of 100 + 7*51 + 6.
    assert result.high_52_week.iloc[364] == pytest.approx(463.0)
    assert result.high_52_week.iloc[370] == pytest.approx(463.0)


def test_stock_stats_run_rate_flat_volume_is_zero():
    # Hourly bars, 61 days, constant volume: the projected daily volume equals the
    # same-weekday average (240 per day), so the run rate is exactly 0 wherever
    # 56 complete days of history exist.
    periods = 61 * 24
    index = pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")
    close = np.full(periods, 50.0)
    frame = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                          "close": close, "volume": np.full(periods, 10.0)}, index=index)
    result = stock_stats(frame)

    assert result.volume_today.iloc[23] == pytest.approx(240.0)
    assert result.volume_today.iloc[24] == pytest.approx(10.0)
    # same_dow needs 56 complete days before the current one; the 23:00 bar's
    # closing clock is midnight, where the elapsed fraction is 0 and Pine is na.
    nan_mask = (index.hour == 23) | (np.arange(periods) < 56 * 24)
    assert result.run_rate_percent[nan_mask].isna().all()
    defined = result.run_rate_percent[~nan_mask]
    assert defined.notna().all()
    assert defined.abs().max() == pytest.approx(0.0)


def test_stock_stats_rejects_bad_input():
    frame = _daily_frame(30, np.full(30, 110.0), np.full(30, 100.0),
                         np.full(30, 105.0), np.full(30, 100.0))
    with pytest.raises(ValueError, match="adr_length"):
        stock_stats(frame, adr_length=0)
    with pytest.raises(ValueError, match="atr_length"):
        stock_stats(frame, atr_length=-1)
    with pytest.raises(ValueError, match="volume"):
        stock_stats(frame.drop(columns="volume"))


def _osc_frame():
    # 16 bars shaped so that bar 5 is a pivot high (16.0, confirmed on 10) and
    # bar 10 is a pivot low (10.0, confirmed on 15); closes make the price higher
    # at the second high pivot and lower at the second low pivot.
    closes = np.array([12.0, 13.0, 14.0, 13.5, 13.0, 12.5, 12.0, 11.5, 11.0,
                       10.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5])
    highs = closes + np.array([0.0, 1.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                               0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    lows = closes - 0.5
    index = pd.date_range("2026-01-01", periods=16, freq="1h", tz="UTC")
    frame = pd.DataFrame({"open": closes, "high": highs, "low": lows,
                          "close": closes, "volume": np.ones(16)}, index=index)
    return frame


def test_divergence_detector_bearish_regular_rsi_hand_computed():
    # Two pivot highs with pivot_length=2: bar 4 (high 13.0, confirmed bar 6) and
    # bar 8 (high 13.5, confirmed bar 10). Price makes a higher high while RSI(3)
    # falls from 100 to ~87.8 -> a bearish regular RSI divergence on bar 10.
    closes = np.array([10.0, 10.5, 11.0, 11.5, 12.0, 11.5, 11.0, 11.5, 13.0,
                       12.5, 12.0, 11.5, 11.0, 10.5, 10.0, 9.5])
    highs = closes + np.array([0.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5,
                               0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    lows = closes - 0.5
    index = pd.date_range("2026-01-01", periods=16, freq="1h", tz="UTC")
    frame = pd.DataFrame({"open": closes, "high": highs, "low": lows,
                          "close": closes, "volume": np.ones(16)}, index=index)
    result = capital41_divergence_detector(frame, pivot_length=2, rsi_length=3,
                                           wr_length=3, max_pivot_bars=10)
    assert result.pivot_high.iloc[6] == pytest.approx(13.0)
    assert result.pivot_high.iloc[10] == pytest.approx(13.5)

    # RSI(3) hand-rolled: changes (0.5, 0.5, 0.5) seed the gain at 0.5 with zero
    # losses, so bar 4 reads 100; the Wilder recursion at bar 8 gives
    # gain 115/162 and loss 10/81, ratio exactly 5.75.
    assert result.rsi.iloc[4] == pytest.approx(100.0)
    assert result.rsi.iloc[8] == pytest.approx(100.0 - 100.0 / (1.0 + 5.75), abs=1e-6)
    # Williams %R at the pivots: (hh - close) / (hh - ll) * -100.
    assert result.williams_r.iloc[4] == pytest.approx((13.0 - 12.0) / 2.5 * -100.0)
    assert result.williams_r.iloc[8] == pytest.approx((13.5 - 13.0) / 3.0 * -100.0)

    assert result.bear_reg_rsi.iloc[10]
    assert result.bearish_regular.iloc[10]
    assert not result.bear_hidden_rsi.iloc[10]  # price made a higher high
    assert not result.bull_reg_rsi.iloc[10]  # only one pivot low exists
    assert result.last_divergence_direction.iloc[10] == "BEAR"
    assert result.last_divergence_type.iloc[10] == "REG"
    assert result.last_divergence_oscillator.iloc[10] == "RSI"
    # The status sticks until the next divergence fires.
    assert result.last_divergence_direction.iloc[15] == "BEAR"
    assert result.last_divergence_direction.iloc[:10].eq("---").all()


def test_divergence_detector_gate_switches():
    closes = np.array([10.0, 10.5, 11.0, 11.5, 12.0, 11.5, 11.0, 11.5, 13.0,
                       12.5, 12.0, 11.5, 11.0, 10.5, 10.0, 9.5])
    highs = closes + np.array([0.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5,
                               0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    lows = closes - 0.5
    index = pd.date_range("2026-01-01", periods=16, freq="1h", tz="UTC")
    frame = pd.DataFrame({"open": closes, "high": highs, "low": lows,
                          "close": closes, "volume": np.ones(16)}, index=index)
    # Turning the RSI leg off removes the signal and the status stays default.
    no_rsi = capital41_divergence_detector(frame, pivot_length=2, rsi_length=3,
                                           wr_length=3, use_rsi=False)
    assert not no_rsi.bearish_regular.any()
    assert no_rsi.last_divergence_direction.eq("---").all()
    # Hiding the regular divergences removes them as well.
    no_regular = capital41_divergence_detector(frame, pivot_length=2, rsi_length=3,
                                               wr_length=3, show_regular=False)
    assert not no_regular.bearish_regular.any()


def test_divergence_detector_rejects_bad_input():
    frame = _osc_frame()
    with pytest.raises(ValueError, match="pivot_length"):
        capital41_divergence_detector(frame, pivot_length=1)
    with pytest.raises(ValueError, match="max_pivot_bars"):
        capital41_divergence_detector(frame, max_pivot_bars=9)
    with pytest.raises(ValueError, match="use_rsi"):
        capital41_divergence_detector(frame, use_rsi="yes")
    with pytest.raises(ValueError, match="close"):
        capital41_divergence_detector(frame.drop(columns="close"))


def _atr_frame(closes):
    closes = np.asarray(closes, dtype=float)
    index = pd.date_range("2026-01-01", periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame({"open": closes, "high": closes + 0.5, "low": closes - 0.5,
                         "close": closes, "volume": np.ones(len(closes))}, index=index)


def test_atr_advanced_raw_mode_hand_computed():
    # Constant +-0.5 wick -> TR is 1.0 on every bar (bar 0 only has high-low),
    # so the Wilder ATR is exactly 1.0 from bar atr_length - 1 onward.
    frame = _atr_frame(np.full(30, 100.0))
    result = atr_advanced(frame, atr_length=14, signal_length=5,
                          normalization_lookback=10, display_mode="Raw", smoothing="SMA")
    assert result.atr_raw.iloc[13] == pytest.approx(1.0)
    assert result.atr_raw.iloc[:13].isna().all()
    # Raw display passes through: the SMA(5) signal is the mean of the last five
    # ATR readings and the histogram is their difference (0 on a constant feed).
    assert result.atr_signal.iloc[18] == pytest.approx(1.0)
    assert result.atr_histogram.dropna().abs().max() == pytest.approx(0.0)
    # ATR% of close 100 with ATR 1 is 1.0; the z-score of a constant series
    # collapses to 0.0 (Pine's dev > 0 ternary) and the percentile rank of a
    # constant series is 100 (every previous value is <= the current one).
    assert result.atr_percent.dropna().eq(1.0).all()
    assert result.atr_zscore.dropna().eq(0.0).all()
    assert result.atr_percentile.dropna().eq(100.0).all()
    # No expansion or contraction can fire without dispersion.
    assert not result.bull_expansion.any() and not result.bear_contraction.any()
    assert result.histogram_above[result.atr_signal.notna()].all()


def test_atr_advanced_percentile_hand_computed():
    # Flat feed for 13 bars then a 2.0-per-bar climb: TR is 1.0 on bars 0..12
    # and 2.5 on bars 13..19, so ATR(1) = TR exactly (rma with length 1).
    closes = np.concatenate([np.full(13, 100.0), 100.0 + np.arange(7) * 2.0])
    frame = _atr_frame(closes)
    result = atr_advanced(frame, atr_length=1, signal_length=1,
                          normalization_lookback=10, display_mode="Percentile Rank",
                          smoothing="EMA")
    assert result.atr_raw.iloc[0] == pytest.approx(1.0)
    assert result.atr_raw.iloc[13] == pytest.approx(1.0)
    assert result.atr_raw.iloc[14] == pytest.approx(2.5)
    # The percentile rank counts the previous 10 values <= the current one; the
    # TR series never decreases, so every defined window reads 100.
    assert result.atr_percentile.iloc[:10].isna().all()
    assert result.atr_percentile.iloc[10] == pytest.approx(100.0)
    assert result.atr_percentile.eq(100.0).iloc[10:].all()


def test_atr_advanced_normalized_mode_and_smoothing_families():
    # The z-score mode: display = (atr - mean) / stdev over 10 bars, 0.0 while
    # the deviation is not defined; check the hand value on a two-step feed.
    closes = np.concatenate([np.full(12, 100.0), 100.0 + np.arange(8) * 2.0])
    frame = _atr_frame(closes)
    result = atr_advanced(frame, atr_length=1, signal_length=3,
                          normalization_lookback=10, display_mode="Normalized",
                          smoothing="SMA")
    # TR is 1.0 on bars 0..12 and 2.5 on bars 13..19 (the first rising bar keeps
    # the old close, so the jump starts one bar later), so the z-score at the
    # last bar uses the last 10 TR values.
    tr = np.concatenate([np.full(13, 1.0), np.full(7, 2.5)])
    window = tr[-10:]
    expected = (tr[-1] - window.mean()) / window.std(ddof=0)
    assert result.atr_zscore.iloc[-1] == pytest.approx(expected)
    # The SMA(3) signal of the z-scored display is the mean of the last three.
    disp = result.atr_zscore.iloc[-3:]
    assert result.atr_signal.iloc[-1] == pytest.approx(disp.mean())
    # Every smoothing family produces a finite signal at the last bar.
    for kind in ("EMA", "RMA", "WMA", "SuperSmoother", "T3", "KAMA", "JMA Approx"):
        smooth = atr_advanced(frame, atr_length=1, signal_length=3,
                              normalization_lookback=10, smoothing=kind)
        assert np.isfinite(smooth.atr_signal.iloc[-1]), kind


def test_atr_advanced_rejects_bad_input():
    frame = _atr_frame(np.linspace(100.0, 110.0, 30))
    with pytest.raises(ValueError, match="atr_length"):
        atr_advanced(frame, atr_length=0)
    with pytest.raises(ValueError, match="normalization_lookback"):
        atr_advanced(frame, normalization_lookback=9)
    with pytest.raises(ValueError, match="display_mode"):
        atr_advanced(frame, display_mode="Fancy")
    with pytest.raises(ValueError, match="smoothing"):
        atr_advanced(frame, smoothing="GAUSS")
    with pytest.raises(ValueError, match="jma_phase"):
        atr_advanced(frame, jma_phase=101)
    with pytest.raises(ValueError, match="jma_power"):
        atr_advanced(frame, jma_power=0.5)
    with pytest.raises(ValueError, match="close"):
        atr_advanced(frame.drop(columns="close"))


def _zones_frame():
    # 12 bars with pivots: high 12.0 at bar 1 (confirmed bar 2 with right=1),
    # low 8.0 at bar 4 (confirmed bar 5), high 13.0 at bar 7 (confirmed bar 8),
    # low 11.0 at bar 9 (confirmed bar 10), high 12.0 at bar 10 (confirmed 11,
    # merges into the 12 level because 12.0 is within 2 * zone width of it).
    rows = [
        (9.5, 10.0, 9.0, 9.5),
        (11.2, 12.0, 11.0, 11.5),
        (10.6, 11.0, 10.5, 10.8),
        (9.8, 10.0, 9.5, 9.7),
        (8.5, 9.0, 8.0, 8.2),
        (8.8, 9.5, 8.3, 9.0),
        (9.5, 10.0, 9.0, 9.8),
        (12.0, 13.0, 11.8, 12.5),
        (12.1, 12.2, 11.8, 11.995),
        (11.2, 11.5, 11.0, 11.2),
        (11.8, 12.0, 11.6, 11.8),
        (11.6, 11.9, 11.4, 11.7),
    ]
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    frame["volume"] = 1.0
    return frame


def test_sr_zones_plasmazero_levels_touches_and_quirk_breakouts():
    frame = _zones_frame()
    result = sr_zones_plasmazero(frame, pivot_left=1, pivot_right=1, max_levels=5,
                                 zone_width_percent=0.5)
    # Bar 2 confirms the 12.0 resistance, bar 5 the 8.0 support, bar 8 adds
    # 13.0 resistance, bar 10 adds the 11.0 support, and bar 11 confirms a
    # 12.0 pivot high that merges into the existing 12.0 level (tolerance
    # 2 * 0.5% = 1%) and counts as a touch instead of a new zone.
    # ta.pivothigh/pivotlow publish the pivot price on the *confirmation* bar,
    # so the 12.0 high of bar 1 lands on bar 2, the 13.0 high of bar 7 on bar 8, etc.
    assert result.new_resistance.tolist() == [False, False, True, False, False, False,
                                              False, False, True, False, False, False]
    assert result.new_support.tolist() == [False, False, False, False, False, True,
                                           False, False, False, False, True, False]
    assert result.resistance_touched.tolist() == [False] * 11 + [True]
    assert result.active_resistance_levels.tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    assert result.active_support_levels.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
    # The merged pivot does not move the newest level: still 13.0 at bar 11.
    assert result.newest_resistance_level.iloc[11] == pytest.approx(13.0)
    assert result.newest_resistance_upper.iloc[11] == pytest.approx(13.0 * 1.005)
    assert result.newest_resistance_touches.iloc[11] == 1
    assert result.newest_support_level.iloc[11] == pytest.approx(11.0)
    assert result.newest_support_lower.iloc[11] == pytest.approx(11.0 * 0.995)
    # Nearest levels bracket the close.
    assert result.nearest_resistance_above.iloc[8] == pytest.approx(12.0)
    assert result.nearest_support_below.iloc[8] == pytest.approx(8.0)
    assert result.nearest_support_below.iloc[11] == pytest.approx(11.0)
    # Pine quirk kept for fidelity: the breakout windows include the current
    # bar's own high/low, so close can never cross them and both flags stay
    # false forever (upstream included).
    assert not result.break_resistance.any()
    assert not result.break_support.any()
    # Constant volume never clears the 150% threshold -> no liquidity zones.
    assert not result.resistance_liquidity.any()


def _liquidity_frame():
    # 30 bars: flat body with two pivot highs (11.0 at bar 20 confirmed on 21,
    # 12.0 at bar 23 confirmed on 24) and a volume spike on bar 24.
    rows = 30
    close = np.full(rows, 10.0)
    high = np.full(rows, 10.1)
    low = np.full(rows, 9.9)
    high[20], high[23] = 11.0, 12.0
    volume = np.full(rows, 1.0)
    volume[24] = 100.0
    index = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close,
                         "volume": volume}, index=index)


def test_sr_zones_plasmazero_liquidity_needs_spike_and_average():
    frame = _liquidity_frame()
    result = sr_zones_plasmazero(frame, pivot_left=1, pivot_right=1, max_levels=5,
                                 zone_width_percent=0.5)
    # The 11.0 zone is confirmed on bar 21, where the 20-bar volume average is
    # not yet defined (na) -> a plain resistance. The 12.0 zone is confirmed on
    # the spike bar 24 (volume 100 vs an average of ~5.95) -> liquidity zone.
    assert result.new_resistance.iloc[21] and not result.resistance_liquidity.iloc[21]
    assert result.new_resistance.iloc[24] and result.resistance_liquidity.iloc[24]
    # Turning liquidity detection off classifies the same zone as plain resistance.
    plain = sr_zones_plasmazero(frame, pivot_left=1, pivot_right=1, max_levels=5,
                                zone_width_percent=0.5, show_liquidity=False)
    assert not plain.resistance_liquidity.any()
    # A volume column of pure zeros keeps every zone non-liquid.
    dead = _liquidity_frame()
    dead["volume"] = 0.0
    assert not sr_zones_plasmazero(dead, pivot_left=1, pivot_right=1).resistance_liquidity.any()


def test_sr_zones_plasmazero_max_levels_eviction():
    frame = _zones_frame()
    result = sr_zones_plasmazero(frame, pivot_left=1, pivot_right=1, max_levels=1,
                                 zone_width_percent=0.5)
    # The 13.0 zone overflows the single resistance slot and evicts 12.0.
    assert result.active_resistance_levels.tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert result.newest_resistance_level.iloc[8] == pytest.approx(13.0)
    # The 12.0 pivot on bar 10 no longer finds the evicted 12.0 level and opens
    # its own zone, which evicts 13.0 in turn.
    assert result.new_resistance.iloc[11]
    assert result.newest_resistance_level.iloc[11] == pytest.approx(12.0)


def test_sr_zones_plasmazero_rejects_bad_input():
    frame = _zones_frame()
    with pytest.raises(ValueError, match="pivot_left"):
        sr_zones_plasmazero(frame, pivot_left=0)
    with pytest.raises(ValueError, match="pivot_right"):
        sr_zones_plasmazero(frame, pivot_right=21)
    with pytest.raises(ValueError, match="max_levels"):
        sr_zones_plasmazero(frame, max_levels=0)
    with pytest.raises(ValueError, match="lookback"):
        sr_zones_plasmazero(frame, lookback=19)
    with pytest.raises(ValueError, match="zone_width_percent"):
        sr_zones_plasmazero(frame, zone_width_percent=0.05)
    with pytest.raises(ValueError, match="volume_threshold_percent"):
        sr_zones_plasmazero(frame, volume_threshold_percent=99)
    with pytest.raises(ValueError, match="high"):
        sr_zones_plasmazero(frame.drop(columns="high"))
