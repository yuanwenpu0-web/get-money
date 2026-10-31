import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.adx_advanced import adx_advanced
from app.catalog_indicators.adx_advanced import _jma as adx_jma
from app.catalog_indicators.adx_advanced import _kama as adx_kama
from app.catalog_indicators.adx_advanced import _super_smoother as adx_super_smoother
from app.catalog_indicators.adx_advanced import _t3 as adx_t3
from app.catalog_indicators.atr_helper import atr_helper
from app.catalog_indicators.capital41_crypto_perp_stress import capital41_crypto_perp_stress
from app.catalog_indicators.momentum_trajectory import momentum_trajectory
from app.catalog_indicators.multi_symbol_ema_bounce_alerts import multi_symbol_ema_bounce_alerts


def _flat_range(close):
    """high == low == close keeps hlc3 equal to the close series."""
    return pd.DataFrame({"high": close, "low": close, "close": close,
                         "volume": np.ones(len(close))})


def test_momentum_trajectory_hand_computed_wavetrend_and_mfi():
    # hlc3 = 10, 11, 9, 10, 12, 11 with volume 1.
    frame = _flat_range(np.array([10.0, 11.0, 9.0, 10.0, 12.0, 11.0]))
    result = momentum_trajectory(frame, wt_channel_length=2, wt_average_length=2,
                                 rsi_length=2, stoch_length=2, k_smoothing=1,
                                 mfi_length=2, velocity_smoothing=1)

    # ema(hlc3, 2): 10.5, 9.5  (Pine seeds with the mean of the first two bars)
    # ema(|hlc3 - ema|, 2): 0.5, 0.5 -> denominator 0.015 * 0.5
    # ci: 0.5 / 0.0075 = 66.667 then -0.5 / 0.0075 = -66.667
    assert result.wt_osc.iloc[1] == pytest.approx(66.6666667)
    # ema(ci, 2) seeded at 66.667: 66.667 + 2/3 * (-66.667 - 66.667) = -22.222
    assert result.wt_osc.iloc[2] == pytest.approx(-22.2222222)

    # velocity is the 1-bar difference of the oscillator when smoothing is off,
    # and acceleration is the difference of the velocity.
    assert result.wt_velocity.iloc[2] == pytest.approx(-22.2222222 - 66.6666667)
    assert result.wt_acceleration.iloc[3] == pytest.approx(
        result.wt_velocity.iloc[3] - result.wt_velocity.iloc[2])

    # MFI(2) at bar 2: positive flow 11 (bar 1), negative flow 9 (bar 2)
    # 100 - 100 / (1 + 11 / 9) = 55
    assert result.mfi.iloc[1] == pytest.approx(100.0)  # no downside in bars 0..1
    assert result.mfi.iloc[2] == pytest.approx(55.0)

    # stoch_rsi is the %K of the RSI series itself, so it stays inside 0..100.
    assert result.stoch_k.dropna().between(0.0, 100.0).all()


def test_momentum_trajectory_states_and_acceleration_crosses():
    close = np.array([10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0, 11.0, 13.0, 14.0, 15.0, 16.0])
    frame = _flat_range(close)
    result = momentum_trajectory(frame, wt_channel_length=2, wt_average_length=2,
                                 rsi_length=2, stoch_length=2, k_smoothing=1,
                                 mfi_length=2, velocity_smoothing=1)

    def label(velocity, acceleration):
        if velocity > 0 and acceleration > 0:
            return "Building"
        if velocity > 0 and acceleration < 0:
            return "Peaking"
        if velocity < 0 and acceleration < 0:
            return "Collapsing"
        if velocity < 0 and acceleration > 0:
            return "Recovering"
        return "Flat"

    expected = [label(v, a) for v, a in zip(result.wt_velocity, result.wt_acceleration)]
    assert list(result.wt_state) == expected
    # A peaking bar is a zero-cross down of the acceleration while velocity is positive.
    peak = ((result.wt_acceleration < 0)
            & (result.wt_acceleration.shift(1) >= 0)
            & (result.wt_velocity > 0))
    assert result.wt_acceleration_peak.tolist() == peak.tolist()


def test_momentum_trajectory_rejects_bad_input():
    frame = _flat_range(np.linspace(10.0, 20.0, 30))
    with pytest.raises(ValueError, match="velocity_smoothing"):
        momentum_trajectory(frame, velocity_smoothing=11)
    with pytest.raises(ValueError, match="rsi_length"):
        momentum_trajectory(frame, rsi_length=0)
    with pytest.raises(ValueError, match="volume"):
        momentum_trajectory(frame.drop(columns="volume"))


def _perp_frame(start="2026-01-05 00:00", periods=60, basis=0.5, volume=10.0,
                oi_growth=1.02):
    """Rising perp candles, a constant spot leg and a steadily growing open interest."""
    index = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    close = 101.0 + 0.2 * np.arange(periods)
    frame = pd.DataFrame({
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": volume,
        "spot_close": 100.0,
        "perp_close": 100.0 * (1.0 + basis / 100.0),
        "open_interest": 1000.0 * oi_growth ** np.arange(periods),
    }, index=index)
    return frame


def _perp_kwargs(**overrides):
    params = dict(trend_ema_length=5, atr_length=2, atr_average_length=5, volume_length=5,
                  oi_smoothing=1, use_weekend_sensitivity=False)
    params.update(overrides)
    return params


def test_capital41_crypto_perp_stress_scores_the_stress_legs():
    frame = _perp_frame()
    result = capital41_crypto_perp_stress(frame, **_perp_kwargs())
    last = result.iloc[-1]

    # basis 0.5% against the 0.25% threshold -> clamp(0.5 / 0.25 * 35, 0, 35) = 35
    # OI growth 2% against the 1.0% impulse -> clamp(2 / 1 * 30, 0, 30) = 30
    assert last.basis_pct == pytest.approx(0.5)
    assert last.oi_change_pct == pytest.approx(2.0)
    assert last.basis_stress == pytest.approx(35.0)
    assert last.oi_stress == pytest.approx(30.0)
    assert last.wick_stress == pytest.approx(0.0)  # 0.5 wick vs 0.8 * ATR(≈1.0)
    assert last.trend_stress == pytest.approx(10.0)  # crowded long + OI build + uptrend
    assert last.stress_score == pytest.approx(75.0)
    assert last.crowded_long and last.oi_build and last.crowded_trend_up
    assert not last.liquidation_down and not last.long_squeeze_risk


def test_capital41_crypto_perp_stress_weekend_thresholds_and_fallback():
    # Saturday bars scale the basis threshold to 0.25 * 0.85 = 0.2125, so a 0.2% basis
    # is no longer crowded long while its stress score keeps the raw ratio.
    frame = _perp_frame(start="2026-01-03 00:00", periods=12, basis=0.2)
    result = capital41_crypto_perp_stress(frame, **_perp_kwargs(use_weekend_sensitivity=True))
    last = result.iloc[-1]
    assert last.is_weekend
    assert not result.crowded_long.any()
    assert last.basis_stress == pytest.approx(0.2 / 0.2125 * 35.0)
    assert last.trend_stress == pytest.approx(0.0)

    # Without an OI column the volume leg silently stands in; disabling the fallback
    # leaves the OI gates closed but keeps the (volume based) stress reading.
    no_oi = _perp_frame().drop(columns="open_interest")
    blocked = capital41_crypto_perp_stress(no_oi, **_perp_kwargs(use_volume_fallback=False))
    assert not blocked.oi_valid.any()
    assert not blocked.oi_build.any() and not blocked.oi_flush.any()
    assert blocked.oi_change_pct.iloc[-1] == pytest.approx(0.0)  # flat volume
    # A bar whose OI reading is missing falls back to volume even with real OI present.
    sparse = _perp_frame()
    sparse.loc[sparse.index[5], "open_interest"] = np.nan
    partial = capital41_crypto_perp_stress(sparse, **_perp_kwargs(use_volume_fallback=False))
    assert not partial.oi_valid.iloc[5]
    assert partial.oi_valid.drop(sparse.index[5]).all()
    with pytest.raises(ValueError, match="perp_close"):
        capital41_crypto_perp_stress(no_oi.drop(columns="perp_close"), **_perp_kwargs())
    with pytest.raises(ValueError, match="volume_length"):
        capital41_crypto_perp_stress(no_oi, **_perp_kwargs(volume_length=4))


def _adx_frame(highs, lows, closes):
    frame = pd.DataFrame({"high": highs, "low": lows, "close": closes})
    frame["open"] = frame["close"]
    return frame


def test_adx_advanced_hand_computed_dmi_on_three_bars():
    # TR(1) = max(102 - 99, 102 - 100, |99.5 - 100|) = 2.5, +DM = 1, -DM = 0
    # TR(2) = max(3, 0.5, 2.5) = 3, +DM = 0, -DM = 1
    frame = _adx_frame(np.array([101.0, 102.0, 101.5]), np.array([99.0, 99.5, 98.5]),
                       np.array([100.0, 101.0, 99.0]))
    result = adx_advanced(frame, di_length=1, adx_smoothing=1, signal_length=1)
    assert result.di_plus.iloc[1] == pytest.approx(40.0)
    assert result.di_minus.iloc[1] == pytest.approx(0.0)
    assert result.di_plus.iloc[2] == pytest.approx(0.0)
    assert result.di_minus.iloc[2] == pytest.approx(100.0 / 3.0)
    # Both bars are one-sided, so |+DI - -DI| / (+DI + -DI) = 100 and ADX(1) follows it.
    assert result.adx.iloc[1] == pytest.approx(100.0)
    assert result.adx.iloc[2] == pytest.approx(100.0)
    # Bar 0 has both DI at 0, so the first real reading is already a bull cross.
    assert result.di_bull_cross.iloc[1]
    assert result.di_bear_cross.iloc[2]
    assert result.adx_signal.tolist() == pytest.approx(result.adx.tolist(), nan_ok=True)
    assert result.adx_histogram.dropna().eq(0.0).all()


def test_adx_advanced_signal_smoothers_and_crosses():
    # 40 choppy bars (ADX well under 20) followed by a clean uptrend (ADX near 100).
    chop = 100.0 + np.random.default_rng(3).normal(0, 0.5, 40)
    closes = np.concatenate([chop, chop[-1] + np.arange(1, 41)])
    frame = _adx_frame(closes + 0.4, closes - 0.4, closes)
    result = adx_advanced(frame, di_length=5, adx_smoothing=5, signal_length=3,
                          smoothing_type="SMA", level_weak=20.0)
    # The SMA option really is a plain rolling mean of the raw ADX.
    assert result.adx_signal.equals(result.adx.rolling(3, min_periods=3).mean())
    assert result.adx.dropna().between(0.0, 100.0).all()
    # The chop produces repeated DI flips and leaves the ADX below 20 low enough that the
    # trend leg has to cross back up over it.
    assert result.di_bull_cross.any() and result.di_bear_cross.any()
    assert result.adx.min() < 20.0 < result.adx.max()
    assert result.adx_activated.any()

    # Every custom smoother collapses a constant input onto that constant.
    level = pd.Series(50.0, index=range(12))
    for smoother in (adx_super_smoother, adx_t3, adx_kama, adx_jma):
        series = smoother(level, 3) if smoother is not adx_jma else smoother(level, 3, 50, 2.0)
        assert series.iloc[-1] == pytest.approx(50.0)
    # Ehlers' two-pole coefficients on a constant 10: 5.71568 then 10.68051.
    smoothed = adx_super_smoother(pd.Series(10.0, index=range(3)), 2)
    assert smoothed.iloc[0] == pytest.approx(5.7156778)
    assert smoothed.iloc[1] == pytest.approx(10.6805117)


def test_adx_advanced_rejects_bad_input():
    frame = _adx_frame(np.array([101.0, 102.0, 101.5]), np.array([99.0, 99.5, 98.5]),
                       np.array([100.0, 101.0, 99.0]))
    with pytest.raises(ValueError, match="smoothing_type"):
        adx_advanced(frame, smoothing_type="HMA")
    with pytest.raises(ValueError, match="jma_phase"):
        adx_advanced(frame, jma_phase=150)
    with pytest.raises(ValueError, match="high"):
        adx_advanced(frame.drop(columns="high"))


def _atr_frame(close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0,
                         "close": close})


def test_atr_helper_builds_the_long_plan_from_a_two_point_range():
    frame = _atr_frame(np.full(20, 100.0))
    result = atr_helper(frame, atr_length=3)
    last = result.iloc[-1]
    # TR is 2 on every bar (high - low), so ATR(3) is 2 and 1.5 ATR is 3.
    assert last.atr == pytest.approx(2.0)
    assert last.entry == pytest.approx(100.0)
    assert last.stop_distance == pytest.approx(3.0)
    assert last.stop_loss == pytest.approx(97.0)
    assert last.take_profit_1 == pytest.approx(103.0)
    assert last.take_profit_2 == pytest.approx(106.0)
    assert last.take_profit_3 == pytest.approx(109.0)
    # 1% of 10 000 over 3.0 risk per unit.
    assert last.risk_amount == pytest.approx(100.0)
    assert last.position_size == pytest.approx(100.0 / 3.0)
    assert last.direction == 1
    assert not result.stop_crossed.any()  # the entry trails the close, so no cross

    short = atr_helper(frame, atr_length=3, direction="Short")
    assert short.stop_loss.iloc[-1] == pytest.approx(103.0)
    assert short.take_profit_1.iloc[-1] == pytest.approx(97.0)
    assert short.direction.iloc[-1] == -1


def test_atr_helper_manual_entry_arms_the_crosses():
    close = np.concatenate([np.full(20, 100.0), np.arange(98.0, 78.0, -2.0)])
    frame = _atr_frame(close)
    result = atr_helper(frame, atr_length=3, entry_mode="Manual", manual_entry=90.0)
    assert result.entry.eq(90.0).all()
    last = result.iloc[-1]
    assert last.stop_loss == pytest.approx(90.0 - 1.5 * last.atr)
    assert last.take_profit_1 == pytest.approx(90.0 + 1.5 * last.atr)
    assert result.stop_crossed.any()
    # `ta.cross` fires on the bar where price moves through the static manual level, and
    # the target is re-crossed later because the ATR (and so the R distance) grows.
    for flag, level in (("stop_crossed", "stop_loss"), ("tp1_crossed", "take_profit_1")):
        crossed = result[flag]
        assert (frame.loc[crossed, "close"] < result.loc[crossed, level]).all()

    # A manual entry of zero falls back to the close, exactly like the Pine guard.
    fallback = atr_helper(frame, atr_length=3, entry_mode="Manual", manual_entry=0.0)
    assert fallback.entry.equals(frame["close"])

    with pytest.raises(ValueError, match="atr_length"):
        atr_helper(frame, atr_length=201)
    with pytest.raises(ValueError, match="atr_smoothing"):
        atr_helper(frame, atr_smoothing="HMA")
    with pytest.raises(ValueError, match="low"):
        atr_helper(frame.drop(columns="low"))


def _bounce_frame(dip=None, pullback=1.0):
    """10 rising bars, a 3-bar pullback and a 6-bar recovery; `dip` adds a wick."""
    close = np.concatenate([np.arange(100.0, 110.0), 109.0 - np.arange(1, 4) * 1.2,
                            np.arange(1, 7) * 1.5 + 105.4])
    low = close - pullback
    if dip is not None:
        index, depth = dip
        low[index] = close[index] - depth
    return pd.DataFrame({"open": close, "high": close + 1.0, "low": low, "close": close,
                         "volume": np.full(len(close), 100.0)})


def _bounce_kwargs(**overrides):
    params = dict(fast_length=2, medium_length=3, slow_length=5, macd_fast_length=2,
                  macd_slow_length=4, macd_signal_length=2, bounce_tolerance_pct=0.5,
                  max_wait_bars=10, rr_multiple=3.0)
    params.update(overrides)
    return params


def test_multi_symbol_ema_bounce_alerts_hands_over_a_3r_plan():
    result = multi_symbol_ema_bounce_alerts(_bounce_frame(dip=(15, 3.0)), **_bounce_kwargs())
    # Bar 14 is the EMA9/EMA20 cross back above a rising EMA stack, and it already
    # touches the EMA9 band, closes above it, holds over VWAP and keeps MACD bullish,
    # so the latch arms and fires on the same bar.
    assert result.long_signal.sum() == 1
    signal = result.loc[result.long_signal].iloc[0]
    assert signal.entry == pytest.approx(108.4)  # close of that bar
    assert signal.stop == pytest.approx(107.4)   # its low
    assert signal.target == pytest.approx(111.4)  # entry + 3 * (108.4 - 107.4)
    assert result.bars_waited.iloc[14] == 0
    assert not result.waiting_for_bounce.iloc[15]  # the signal clears the latch
    # One bar earlier the bounce was already there, but nothing had armed the latch.
    assert result.bounce.iloc[13] and not result.waiting_for_bounce.iloc[13]
    assert not result.long_signal.iloc[13]


def test_multi_symbol_ema_bounce_alerts_wait_budget():
    # The wick only appears at bar 18; a one-bar budget clears the latch long before it
    # while the unlimited setting (0) keeps waiting.
    patient = multi_symbol_ema_bounce_alerts(_bounce_frame(dip=(18, 3.0), pullback=0.1),
                                             **_bounce_kwargs(bounce_tolerance_pct=0.1,
                                                              max_wait_bars=0))
    assert list(patient.index[patient.long_signal]) == [18]
    signal = patient.loc[patient.long_signal].iloc[0]
    assert signal.entry == pytest.approx(114.4)
    assert signal.stop == pytest.approx(111.4)
    assert signal.target == pytest.approx(114.4 + 3.0 * 3.0)
    assert patient.bars_waited.iloc[18] == 4

    impatient = multi_symbol_ema_bounce_alerts(_bounce_frame(dip=(18, 3.0), pullback=0.1),
                                               **_bounce_kwargs(bounce_tolerance_pct=0.1,
                                                                max_wait_bars=1))
    assert not impatient.long_signal.any()
    assert impatient.bars_waited.dropna().max() == 1


def test_multi_symbol_ema_bounce_alerts_vwap_restarts_each_session():
    index = pd.to_datetime(["2026-01-01 23:30", "2026-01-01 23:45",
                            "2026-01-02 00:00", "2026-01-02 00:15"]).tz_localize("UTC")
    typical = np.array([100.0, 101.0, 102.0, 103.0])
    frame = pd.DataFrame({"open": typical, "high": typical, "low": typical, "close": typical,
                          "volume": np.array([10.0, 30.0, 10.0, 10.0])}, index=index)
    result = multi_symbol_ema_bounce_alerts(frame, **_bounce_kwargs())
    # day one: (100 * 10 + 101 * 30) / 40, then the new session restarts from (102 * 10)
    assert result.vwap.iloc[0] == pytest.approx(100.0)
    assert result.vwap.iloc[1] == pytest.approx(1000.0 / 40.0 + 101.0 * 30.0 / 40.0)
    assert result.vwap.iloc[2] == pytest.approx(102.0)
    assert result.vwap.iloc[3] == pytest.approx(102.5)

    with pytest.raises(ValueError, match="fast_length"):
        multi_symbol_ema_bounce_alerts(frame, fast_length=0)
    with pytest.raises(ValueError, match="max_wait_bars"):
        multi_symbol_ema_bounce_alerts(frame, max_wait_bars=-1)
    with pytest.raises(ValueError, match="volume"):
        multi_symbol_ema_bounce_alerts(frame.drop(columns="volume"))
