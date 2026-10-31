import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.futures_context_module import futures_context_module
from app.catalog_indicators.ma_regime_bands import ma_regime_bands
from app.catalog_indicators.market_memory_and_decay_oscillator import (
    market_memory_and_decay_oscillator as mmdo_port,
)
from app.catalog_indicators.oscillator_topology import _classify, oscillator_topology
from app.catalog_indicators.support_resistance import support_resistance


def _frame(close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": np.ones(len(close))})


def test_mmdo_flat_market_collapses_to_zero():
    # A perfectly flat market: ret is exactly 0, so retSd is 0 and retZ falls back
    # to 0 (Pine f_safeDiv), the impulse line stays 0 and both legs of the score
    # vanish regardless of the decay pressure.
    frame = _frame(np.full(60, 100.0))
    result = mmdo_port(frame, memory_len=20, fast_len=5, slow_len=30, smooth_len=2)
    assert result.mmdo.dropna().abs().max() == pytest.approx(0.0)
    assert result.ret_z.eq(0.0).all()
    # direct == path == 0, so the efficiency falls back to 0 (Pine fallback 0.0).
    assert result.efficiency.eq(0.0).all()
    # TR is 0, so rangeFast/rangeSlow is 0/0 -> the ratio falls back to 1.
    assert result.range_ratio.eq(1.0).all()
    # No impulse ever fires, so the age holds at Pine's 999.0 sentinel.
    assert result.impulse_age.eq(999.0).all()
    assert not result.carry_bull.any() and not result.carry_bear.any()
    assert not result.new_impulse.any() and not result.decaying.any()
    assert result.hurst.eq(0.0).all()  # R/S falls back to 1 -> log(1) = 0


def test_mmdo_hand_computed_impulse_and_half_life():
    closes = 100.0 + np.arange(30) * 0.5
    closes[21] += 3.0
    closes[25] -= 4.0
    frame = _frame(closes)
    result = mmdo_port(frame, memory_len=20, fast_len=5, slow_len=30, smooth_len=1,
                       impulse_z=0.5)

    ret = np.diff(np.log(closes), prepend=np.nan)
    # retZ is the fallback 0.0 until retMean/retSd exist (bar 20 for length 20),
    # because Pine's f_safeDiv collapses the NaN warm-up to 0.
    assert result.ret_z.iloc[:20].eq(0.0).all()
    z = np.zeros(30)
    for t in range(20, 30):
        window = ret[t - 19:t + 1]
        z[t] = (ret[t] - window.mean()) / window.std(ddof=0)
    assert np.allclose(result.ret_z.iloc[20:], z[20:])
    assert result.new_impulse.tolist() == list(np.abs(z) > 0.5)

    # impulseRaw = ema(retZ, 5): SMA-seeded on the fallback zeros, then recursed.
    raw = np.full(30, np.nan)
    raw[4] = z[:5].mean()
    for t in range(5, 30):
        raw[t] = raw[t - 1] + 2 / 6 * (z[t] - raw[t - 1])
    assert np.allclose(result.impulse_raw.iloc[4:], raw[4:], equal_nan=True)

    # phi = corr(impulseRaw, impulseRaw[1], 20) is first defined at bar 24, when 20
    # complete lagged pairs exist; the half-life follows from the clamped phi.
    phi = float(np.corrcoef(raw[5:25], raw[4:24])[0, 1])
    persistent = min(max(phi, 0.0), 0.98)
    expected = -np.log(2.0) / np.log(persistent) if persistent > 0.01 else 0.0
    expected = min(max(expected, 0.0), 48.0)
    assert result.half_life.iloc[24] == pytest.approx(expected, rel=1e-6)
    assert result.half_life.iloc[:24].eq(0.0).all()  # nz(phi) falls back to 0


def test_mmdo_rejects_bad_input():
    frame = _frame(np.linspace(100.0, 110.0, 40))
    with pytest.raises(ValueError, match="memory_len"):
        mmdo_port(frame, memory_len=19)
    with pytest.raises(ValueError, match="slow_len"):
        mmdo_port(frame, slow_len=29)
    with pytest.raises(ValueError, match="max_half_life"):
        mmdo_port(frame, max_half_life=0.0)
    with pytest.raises(ValueError, match="close"):
        mmdo_port(frame.drop(columns="close"))


def _sr_frame():
    rows = [
        # open, high, low, close
        (9.5, 10.0, 9.0, 9.5),     # 0
        (11.2, 12.0, 11.0, 11.5),  # 1 pivot high 12 (confirmed at 2)
        (10.6, 11.0, 10.5, 10.8),  # 2
        (9.8, 10.0, 9.5, 9.7),     # 3
        (8.5, 9.0, 8.0, 8.2),      # 4 pivot low 8 (confirmed at 5)
        (8.8, 9.5, 8.3, 9.0),      # 5
        (9.5, 10.0, 9.0, 9.8),     # 6
        (12.0, 13.0, 11.8, 12.5),  # 7 breakout of 12; pivot high 13 (confirmed at 8)
        (12.1, 12.2, 11.8, 11.995),  # 8 retest of 12; second resistance 13
        (11.2, 11.5, 11.0, 11.2),  # 9 pivot low 11 (confirmed at 10)
        (11.8, 12.0, 11.6, 11.8),  # 10 pivot high 12 -> merges into the 12 level
        (11.6, 11.9, 11.4, 11.7),  # 11
    ]
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    frame["volume"] = 1.0
    return frame


def test_support_resistance_levels_breakouts_and_retests():
    frame = _sr_frame()
    result = support_resistance(frame, strength=1, max_levels=5)

    # Bar 2 confirms the 12 resistance; bar 5 confirms the 8 support.
    assert result.pivot_high.iloc[2] == pytest.approx(12.0)
    assert result.pivot_low.iloc[5] == pytest.approx(8.0)
    assert result.resistance_count.tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    assert result.support_count.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2]

    # Bar 7 closes above the 12 level after closing below it on bar 6.
    assert result.resistance_breakout.tolist() == [False] * 7 + [True] + [False] * 4
    assert not result.support_breakout.any()
    # Bar 8 closes within 2% of the bar range of 12 while staying below it.
    assert result.resistance_retest.tolist() == [False] * 8 + [True] + [False] * 3
    assert not result.support_retest.any()

    # Nearest levels follow the per-bar pool state; bar 7's close sits above the
    # only level it has (12), so no resistance above exists yet.
    assert result.nearest_resistance.iloc[5] == pytest.approx(12.0)
    assert result.nearest_support.iloc[5] == pytest.approx(8.0)
    assert pd.isna(result.nearest_resistance.iloc[7])
    assert result.nearest_support.iloc[7] == pytest.approx(8.0)
    assert result.nearest_resistance.iloc[8] == pytest.approx(12.0)

    # Bar 10's pivot high of 12.0 merges into the existing 12 level (tolerance is
    # 1% of the confirming bar's range = 0.005), so no third level appears.
    assert result.resistance_count.iloc[11] == 2
    assert result.pivot_high.iloc[11] == pytest.approx(12.0)


def test_support_resistance_max_levels_evicts_the_weakest():
    frame = _sr_frame()
    # With only one slot the 13 resistance evicts the 12 level on bar 8 (both carry
    # one hit, the first minimum wins), leaving 13 as the only resistance.
    result = support_resistance(frame, strength=1, max_levels=1)
    assert result.resistance_count.tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert result.nearest_resistance.iloc[10] == pytest.approx(13.0)
    # The breakout of the evicted 12 level on bar 7 still happened while it existed.
    assert result.resistance_breakout.iloc[7]


def test_support_resistance_rejects_bad_input():
    frame = _sr_frame()
    with pytest.raises(ValueError, match="strength"):
        support_resistance(frame, strength=0)
    with pytest.raises(ValueError, match="max_levels"):
        support_resistance(frame, max_levels=11)
    with pytest.raises(ValueError, match="high"):
        support_resistance(frame.drop(columns="high"))


def test_oscillator_topology_shape_classification():
    # V-Spike: a single-bar burst out of a flat line, curvature normalised to 100.
    values = np.array([0.0] * 7 + [10.0, 0.0, 0.0, 0.0])
    shape, curvature, flat = _classify(values, bar=9, pivot_bar=7, is_high=True,
                                       pivot_left=2, spike_threshold=3.0,
                                       flat_bars=3, flat_tolerance=0.15)
    assert shape == "V-Spike"
    assert curvature == pytest.approx(100.0)
    assert flat == 1

    # Flat: a smooth dome whose shoulders hug the peak within 15% of the height.
    values = np.array([4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 4.95, 4.9])
    shape, curvature, flat = _classify(values, bar=7, pivot_bar=5, is_high=True,
                                       pivot_left=2, spike_threshold=3.0,
                                       flat_bars=3, flat_tolerance=0.15)
    assert shape == "Flat"
    assert curvature == pytest.approx(0.0)
    assert flat == 3

    # Asymm: a fast rise and a slow departure, far above the 0.5 asymmetry gate.
    values = np.array([2.0, 4.0, 8.0, 7.5, 7.0, 6.5, 6.0])
    shape, curvature, flat = _classify(values, bar=4, pivot_bar=2, is_high=True,
                                       pivot_left=2, spike_threshold=50.0,
                                       flat_bars=3, flat_tolerance=0.15)
    assert shape == "Asymm"
    # rise (8-2)/2 = 3, fall (8-7)/2 = 0.5 -> |3-0.5| / max(3.5, 0.01) = 0.714
    assert flat == 2

    # Round: a symmetric triangle — zero curvature, no plateau, no asymmetry.
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    shape, curvature, flat = _classify(values, bar=7, pivot_bar=5, is_high=True,
                                       pivot_left=2, spike_threshold=50.0,
                                       flat_bars=3, flat_tolerance=0.15)
    assert shape == "Round"
    assert curvature == pytest.approx(0.0)
    assert flat == 1


def test_oscillator_topology_wavetrend_hand_computed():
    # hlc3 = 10, 11, 9, 10, 12, 11 with high == low == close.
    close = np.array([10.0, 11.0, 9.0, 10.0, 12.0, 11.0])
    frame = pd.DataFrame({"open": close, "high": close, "low": close,
                          "close": close, "volume": np.ones(len(close))})
    result = oscillator_topology(frame, wt_channel_length=2, wt_average_length=2,
                                 pivot_left=2, pivot_right=2)
    # Same WaveTrend recursion as the reference implementation: esa 10.5 then 9.5,
    # d 0.5, so ci is +-66.667 and the oscillator EMA gives 66.667 then -22.222.
    assert result.wt_osc.iloc[1] == pytest.approx(66.6666667)
    assert result.wt_osc.iloc[2] == pytest.approx(-22.2222222)
    assert result.wt_velocity.iloc[2] == pytest.approx(-88.8888889)
    # The trough at bar 2 is confirmed on bar 4; the shape reads only bars <= 4 and
    # the warm-up gap to its left leaves the curvature undefined, so it rounds to
    # the default "Round" classification.
    assert result.pivot_low.iloc[4] == pytest.approx(-22.2222222)
    assert result.shape_low.iloc[4] == "Round"
    # A second trough (bar 3) confirms on bar 5; its curvature (|130.4| / 88.9 * 100)
    # clears the default 3.0 spike threshold, so it classifies as V-Spike.
    assert result.pivot_low.iloc[5] == pytest.approx(19.2592593)
    assert result.shape_low.iloc[5] == "V-Spike"
    assert result.last_low_shape.iloc[5] == "V-Spike"
    # Curvature normalises the second derivative by the 20-bar oscillator range;
    # before that range exists (or when it is 0) Pine's ternary falls back to 0.0.
    osc_range = (result.wt_osc.rolling(20, min_periods=20).max()
                 - result.wt_osc.rolling(20, min_periods=20).min())
    expected_curvature = (result.wt_acceleration / osc_range * 100.0).where(osc_range > 0, 0.0)
    assert result.wt_curvature.tolist() == pytest.approx(expected_curvature.tolist(),
                                                         nan_ok=True)


def test_oscillator_topology_rejects_bad_input():
    close = np.linspace(10.0, 12.0, 30)
    frame = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                          "close": close, "volume": np.ones(len(close))})
    with pytest.raises(ValueError, match="pivot_left"):
        oscillator_topology(frame, pivot_left=1)
    with pytest.raises(ValueError, match="flat_tolerance"):
        oscillator_topology(frame, flat_tolerance=0.6)
    with pytest.raises(ValueError, match="wt_channel_length"):
        oscillator_topology(frame, wt_channel_length=0)
    with pytest.raises(ValueError, match="low"):
        oscillator_topology(frame.drop(columns="low"))


def _futures_frame(periods=25, spread=2.0):
    index = pd.date_range("2026-01-01", periods=periods, freq="1D", tz="UTC")
    front = 100.0 + 0.5 * np.arange(periods)
    return pd.DataFrame({
        "front_close": front,
        "next_close": front - spread,
        "front_open_interest": 1000.0 * 1.02 ** np.arange(periods),
        "front_volume": np.full(periods, 10.0),
    }, index=index)


def test_futures_context_module_hand_computed_states():
    frame = _futures_frame()
    result = futures_context_module(frame, oi_lookback=1, term_ema_fast=2,
                                    term_ema_slow=5)
    # spread is constant at 2.0 -> backwardation, and both EMAs settle on 2.0
    # immediately after their SMA seed (bar 1 for length 2, bar 4 for length 5).
    assert result.term_structure.eq("Backwardation").all()
    assert result.spread.iloc[0] == pytest.approx(2.0)
    assert result.spread_ema_fast.iloc[1] == pytest.approx(2.0)
    assert result.spread_ema_slow.iloc[4] == pytest.approx(2.0)
    # A flat curve with a zero slope can only read "flat".
    assert result.curve_trend.iloc[5] == "flat"
    # OI grows 2% per bar against a rising front price -> new longs.
    assert result.oi_change_pct.iloc[1] == pytest.approx(2.0)
    assert result.oi_price_state.iloc[1] == "OI rising + Price rising -> new longs"
    assert result.front_price_change.iloc[1] == pytest.approx(0.5)
    # Constant volume sits exactly on its 20-bar average -> "below avg" (not above).
    assert result.front_vol_avg.iloc[19] == pytest.approx(10.0)
    assert result.volume_state.iloc[19] == "below avg"
    assert result.volume_state.iloc[18] == "n/a"  # average not defined yet


def test_futures_context_module_missing_legs_and_directions():
    frame = _futures_frame(periods=6)
    # A missing next-month leg kills the spread from that bar on (na propagates
    # through the EMAs, so the curve trend reads "n/a" too).
    gapped = frame.copy()
    gapped.iloc[-1, gapped.columns.get_loc("next_close")] = np.nan
    result = futures_context_module(gapped, oi_lookback=1, term_ema_fast=2,
                                    term_ema_slow=5)
    assert result.term_structure.iloc[-1] == "n/a"
    assert result.curve_trend.iloc[-1] == "n/a"
    assert result.term_structure.iloc[-2] == "Backwardation"

    # Falling price with rising OI -> new shorts; falling OI with falling price
    # -> long liquidation; an unchanged OI prints the Pine "OI flat" fallback.
    falling = frame.copy()
    falling["front_close"] = falling["front_close"].iloc[::-1].to_numpy()
    falling["front_open_interest"] = np.where(np.arange(6) % 2 == 0, 1000.0, 900.0)
    states = futures_context_module(falling, oi_lookback=1, term_ema_fast=2,
                                    term_ema_slow=5).oi_price_state
    assert states.iloc[1] == "OI falling + Price falling -> long liquidation"
    assert states.iloc[2] == "OI rising + Price falling -> new shorts"
    flat_oi = frame.copy()
    flat_oi["front_close"] = 100.0
    flat_oi["next_close"] = 100.0
    flat_oi["front_open_interest"] = 1000.0
    flat = futures_context_module(flat_oi, oi_lookback=1, term_ema_fast=2,
                                  term_ema_slow=5)
    assert flat.oi_price_state.iloc[1] == "OI flat"
    assert flat.term_structure.iloc[0] == "Flat"

    # A feed with no positive volume at all is reported as "no volume".
    dead = _futures_frame(periods=25)
    dead["front_volume"] = 0.0
    assert futures_context_module(dead, oi_lookback=1, term_ema_fast=2,
                                  term_ema_slow=5).volume_state.eq("no volume").all()


def test_futures_context_module_rejects_bad_input():
    frame = _futures_frame()
    with pytest.raises(ValueError, match="oi_lookback"):
        futures_context_module(frame, oi_lookback=0)
    with pytest.raises(ValueError, match="oi_lookback"):
        futures_context_module(frame, oi_lookback=21)
    with pytest.raises(ValueError, match="term_ema_fast"):
        futures_context_module(frame, term_ema_fast=1)
    with pytest.raises(ValueError, match="front_open_interest"):
        futures_context_module(frame.drop(columns="front_open_interest"))


def _bands_frame(closes):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": closes, "high": closes + 0.5, "low": closes - 0.5,
                         "close": closes, "volume": np.ones(len(closes))})


def test_ma_regime_bands_flat_market_is_neutral():
    frame = _bands_frame(np.full(30, 100.0))
    result = ma_regime_bands(frame, ma_type="SMA", len1=2, len2=3, len3=4,
                             atr_len=2, strength_mode="Off")
    # Every MA collapses onto 100, the ATR is 0 and the spread normalisation
    # falls back to 0.0, so the trend can never be strong.
    assert result.ma1.dropna().eq(100.0).all() and result.ma3.dropna().eq(100.0).all()
    assert result.ama.dropna().eq(100.0).all()
    # TR is 1.0 on every bar (the +-0.5 wick), so the ATR is 1 and the spread
    # normalisation is 0/1 = 0.
    assert result.atr.dropna().eq(1.0).all() and result.spread_atr.eq(0.0).all()
    assert not result.strong_trend.any()
    assert result.neutral.all()
    assert not result.bull.any() and not result.bear.any()
    assert not result.bull_flip.any() and not result.bear_flip.any()
    assert result.long_entry.isna().all() and result.short_entry.isna().all()


def test_ma_regime_bands_hand_computed_bull_regime():
    closes = np.concatenate([np.arange(100.0, 112.0), [111.0, 110.5, 110.0],
                             np.arange(109.0, 95.0, -1.0)])
    frame = _bands_frame(closes)
    result = ma_regime_bands(frame, ma_type="SMA", len1=2, len2=3, len3=4,
                             atr_len=2, mult=2.0, strength_mode="By MA Type")
    # Bar 3 is the first bar with all MAs defined: SMA2=102.5, SMA3=102, SMA4=101.5,
    # close 103 above all, stack sorted, and the ATR-normalised spread
    # (1.0 / 1.4375 = 0.696) clears the SMA strength floor of 0.6.
    assert result.bull.iloc[3]
    assert result.bull.iloc[:3].sum() == 0
    assert result.bull_flip.iloc[3]
    assert result.long_entry.iloc[3] == pytest.approx(102.5)
    assert result.long_stop.iloc[3] == pytest.approx(102.5 - 2.0 * 1.4375)
    # The stop level persists while the regime holds.
    assert result.long_stop.iloc[4] == pytest.approx(result.long_stop.iloc[3])
    # Min strength follows the SMA factor of 1.0.
    assert result.min_strength.eq(0.6).all()

    # The mild fall (0.5 steps) breaks the bull regime without sorting the stack
    # far enough for a bear flip, so a neutral exit fires in between; the deep fall
    # afterwards flips bearish with an ATR stop above its entry.
    assert result.exit_flip.any()
    assert result.bear.any() and result.bear_flip.any()
    assert result.short_entry.dropna().le(result.short_stop.dropna()).all()


def test_ma_regime_bands_dema_tema_jma_hand_computed():
    close = np.array([10.0, 11.0, 9.0, 12.0])
    frame = _bands_frame(close)
    dema = ma_regime_bands(frame, ma_type="DEMA", len1=2, len2=3, len3=4,
                           atr_len=1, strength_mode="By MA Type")
    # DEMA(2): ema1 = na, 10.5, 9.5, 11.1667; the nested EMA seeds at bar 2 with
    # mean(10.5, 9.5) = 10.0 (ema1[0] is na), so DEMA at bar 2 = 2*9.5 - 10.0 = 9.0.
    assert dema.ma1.iloc[2] == pytest.approx(9.0)
    assert dema.ma1.iloc[3] == pytest.approx(2 * 11.1666667
                                             - (10.0 + 2 / 3 * (11.1666667 - 10.0)))
    # The DEMA strength factor scales the floor to 0.6 * 1.05.
    assert dema.min_strength.eq(0.63).all()

    tema = ma_regime_bands(frame, ma_type="TEMA", len1=2, len2=3, len3=4,
                           atr_len=1, strength_mode="Off")
    # TEMA stacks three EMA layers, so it stays na until bar 3, where
    # ema2 = [.., 10.0, 10.7778] and ema(ema2, 2) seeds at mean(10.0, 10.7778).
    assert tema.ma1.iloc[:3].isna().all()
    e2_bar3 = 10.0 + 2 / 3 * (11.1666667 - 10.0)
    assert tema.ma1.iloc[3] == pytest.approx(3 * 11.1666667 - 3 * e2_bar3
                                             + (10.0 + e2_bar3) / 2)

    jma = ma_regime_bands(frame, ma_type="JMA", len1=2, len2=3, len3=4,
                          atr_len=1, strength_mode="Off")
    # JMA recursion by hand for length 2: beta = 0.45/2.45, alpha = beta^2, pr = 1.5.
    beta = 0.45 / 2.45
    alpha = beta ** 2
    e0 = (1 - alpha) * 11.0 + alpha * 10.0
    e1 = (11.0 - e0) * (1 - beta)
    e2 = (e0 + 1.5 * e1 - 10.0) * (1 - alpha) ** 2
    assert jma.ma1.iloc[0] == pytest.approx(10.0)
    assert jma.ma1.iloc[1] == pytest.approx(10.0 + e2)


def test_ma_regime_bands_auto_strength_and_bad_input():
    closes = np.concatenate([np.full(40, 100.0) + np.random.default_rng(7).normal(0, 0.3, 40),
                             100.0 + np.arange(1, 41) * 0.8])
    frame = _bands_frame(closes)
    result = ma_regime_bands(frame, ma_type="SMA", len1=2, len2=3, len3=4,
                             atr_len=2, strength_mode="Auto", auto_lookback=50)
    # Before the 50-bar reference exists the factor falls back to 1.0.
    assert result.min_strength.iloc[10] == pytest.approx(0.6)
    # The auto factor is the RMA-smoothed AMA step over its own 50-bar mean.
    from app.catalog_indicators._pine import rma
    step = rma(result.ama.diff().abs(), 2) / result.atr
    reference = step.rolling(50, min_periods=50).mean()
    assert result.min_strength.iloc[-1] == pytest.approx(0.6 * step.iloc[-1]
                                                         / reference.iloc[-1])

    with pytest.raises(ValueError, match="ma_type"):
        ma_regime_bands(frame, ma_type="HMA")
    with pytest.raises(ValueError, match="strength_mode"):
        ma_regime_bands(frame, strength_mode="Fixed")
    with pytest.raises(ValueError, match="auto_lookback"):
        ma_regime_bands(frame, auto_lookback=49)
    with pytest.raises(ValueError, match="mult"):
        ma_regime_bands(frame, mult=0.05)
    with pytest.raises(ValueError, match="len1"):
        ma_regime_bands(frame, len1=0)
    with pytest.raises(ValueError, match="close"):
        ma_regime_bands(frame.drop(columns="close"))
