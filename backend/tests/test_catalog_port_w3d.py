"""Ports batch W3D: crypto_amt_session_bias_v2_companion,
commodity_heat_reversal, gcr_strategy_scanner_tradingview, roc_advanced,
commodity_flowtrend."""
import math

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.crypto_amt_session_bias_v2_companion import (
    crypto_amt_session_bias_v2_companion,
)


def _frame(bars, *, start="2024-01-01", freq="1h", open_=None, high=None, low=None,
           close=None, volume=None):
    index = pd.date_range(start, periods=bars, freq=freq, tz="UTC")
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({
        "open": np.asarray(open_, dtype=float) if open_ is not None else close,
        "high": np.asarray(high, dtype=float) if high is not None else close + 1.0,
        "low": np.asarray(low, dtype=float) if low is not None else close - 1.0,
        "close": close,
        "volume": np.full(bars, 100.0) if volume is None else np.asarray(volume, dtype=float),
    }, index=index)


# ---------------------------------------------------------------------------
# crypto_amt_session_bias_v2_companion
# ---------------------------------------------------------------------------

def _session_frame():
    """29 hourly bars from 2024-01-01 00:00 UTC with three contiguous sessions:
    Asia 0000-0500 (bars 0-4, and again 24-28 on Jan 2), London 0500-1000
    (bars 5-9), New York 1000-1500 (bars 10-14); bars 15-23 sit in no session.
    Prices are engineered so that London sweeps the Asia high on its first bar
    and reclaims the same bar, and New York sweeps the London low on its first
    bar and reclaims the same bar."""
    bars = 29
    open_ = np.full(bars, 100.0)
    high = np.full(bars, 100.0)
    low = np.full(bars, 100.0)
    close = np.full(bars, 100.0)
    # London day 1 (bars 5-9): sweep the Asia high (100) at 101 and close back
    # under it.
    high[5], close[5] = 101.0, 99.5
    close[6:10] = 99.5
    # New York day 1 (bars 10-14): sweep the London low (100) at 99 and close
    # back above it.
    low[10], high[10], close[10] = 99.0, 100.5, 100.5
    close[11:15] = 100.5
    high[11:15] = 100.5
    low[11:15] = 100.0
    # Asia day 2 (bars 24-28): quiet, no sweep of the New York levels.
    close[24:29] = 100.5
    high[24:29] = 100.5
    low[24:29] = 100.0
    return _frame(bars, open_=open_, high=high, low=low, close=close)


def test_session_bias_hand_computed():
    frame = _session_frame()
    result = crypto_amt_session_bias_v2_companion(
        frame, asia_session="0000-0500", london_session="0500-1000",
        new_york_session="1000-1500")

    assert result.in_asia.tolist()[:5] == [True] * 5
    assert result.in_asia.tolist()[5:24] == [False] * 19
    assert result.in_asia.tolist()[24:] == [True] * 5
    assert result.in_london.tolist()[5:10] == [True] * 5
    assert result.in_new_york.tolist()[10:15] == [True] * 5

    # Asia day 1 accumulates the flat 100s; its levels become the previous
    # target from bar 5 on.
    assert result.active_session.tolist()[:5] == ["Asia"] * 5
    assert result.session_high.iloc[3] == pytest.approx(100.0)
    assert result.session_low.iloc[3] == pytest.approx(100.0)
    assert result.previous_session.iloc[5] == "Asia"
    assert result.previous_session_high.iloc[5] == pytest.approx(100.0)
    assert result.previous_session_low.iloc[5] == pytest.approx(100.0)
    assert result.previous_session_mid.iloc[5] == pytest.approx(100.0)

    # Bar 5: London starts, sweeps the Asia high (101 > 100) and reclaims the
    # same bar (close 99.5 < 100) -> bearish bias.
    assert result.sweep_high.iloc[5]
    assert not result.sweep_low.iloc[5]
    assert result.reclaim_high.iloc[5]
    assert result.bias_flip.iloc[5]
    assert result.bias_state.iloc[5] == -1

    # Bar 10: London has ended with high 101 / low 100; those become the target
    # and New York sweeps the London low (99 < 100) and reclaims above it.
    assert result.previous_session.iloc[10] == "London"
    assert result.previous_session_high.iloc[10] == pytest.approx(101.0)
    assert result.previous_session_low.iloc[10] == pytest.approx(100.0)
    assert result.active_session.iloc[10] == "New York"
    assert result.sweep_low.iloc[10]
    assert not result.sweep_high.iloc[10]
    assert result.reclaim_low.iloc[10]
    assert result.bias_state.iloc[10] == 1

    # Bars 15-23 sit in no session; the New York levels (high 100.5, low 99)
    # stay the target and the bias stays neutral after the reset.
    assert result.active_session.tolist()[15:24] == ["None"] * 9
    assert result.session_high.iloc[20] != result.session_high.iloc[20]  # NaN
    assert result.previous_session.iloc[15] == "New York"
    assert result.previous_session_high.iloc[15] == pytest.approx(100.5)
    assert result.previous_session_low.iloc[15] == pytest.approx(99.0)
    assert result.previous_session_mid.iloc[15] == pytest.approx(99.75)

    # Asia day 2: the session change resets the bias and nothing sweeps.
    assert result.active_session.iloc[24] == "Asia"
    assert result.bias_state.iloc[24] == 0
    assert not result.sweep_high.iloc[15:].any()
    assert not result.sweep_low.iloc[15:].any()

    # Opening range (60 minutes) covers only the session's first hourly bar.
    assert result.opening_range_high.iloc[5] == pytest.approx(101.0)
    assert result.opening_range_low.iloc[5] == pytest.approx(100.0)
    assert result.opening_range_high.iloc[6] == pytest.approx(101.0)
    assert result.opening_range_low.iloc[6] == pytest.approx(100.0)
    assert result.opening_range_high.iloc[10] == pytest.approx(100.5)
    assert result.opening_range_low.iloc[10] == pytest.approx(99.0)

    # Disabling a session removes it entirely.
    off = crypto_amt_session_bias_v2_companion(
        frame, show_london=False, asia_session="0000-0500",
        london_session="0500-1000", new_york_session="1000-1500")
    assert not off.in_london.any()
    assert off.active_session.iloc[5] == "None"
    assert off.previous_session.iloc[10] == "Asia"


def test_session_bias_rejects_bad_input():
    frame = _session_frame()
    with pytest.raises(ValueError, match="opening_range_minutes"):
        crypto_amt_session_bias_v2_companion(frame, opening_range_minutes=0)
    with pytest.raises(ValueError, match="opening_range_minutes"):
        crypto_amt_session_bias_v2_companion(frame, opening_range_minutes=241)
    with pytest.raises(ValueError, match="asia_session"):
        crypto_amt_session_bias_v2_companion(frame, asia_session="0800")
    with pytest.raises(ValueError, match="show_london"):
        crypto_amt_session_bias_v2_companion(frame, show_london="yes")
    with pytest.raises(ValueError, match="open"):
        crypto_amt_session_bias_v2_companion(frame.drop(columns="open"))


# ---------------------------------------------------------------------------
# commodity_heat_reversal
# ---------------------------------------------------------------------------

from app.catalog_indicators.commodity_heat_reversal import commodity_heat_reversal


def _wilder_rsi(close, length=14):
    delta = np.diff(close, prepend=close[0])
    gain = np.zeros(len(close))
    loss = np.zeros(len(close))
    gain[1:] = np.clip(delta[1:], 0, None)
    loss[1:] = np.clip(-delta[1:], 0, None)
    out = np.full(len(close), np.nan)

    def rma(values):
        current, count, total = np.nan, 0, 0.0
        result = np.full(len(values), np.nan)
        for position, value in enumerate(values):
            if count < length:
                count += 1
                total += value
                if count == length:
                    current = total / length
            else:
                current = (current * (length - 1) + value) / length
            result[position] = current
        return result

    gains, losses = rma(gain), rma(loss)
    for bar in range(length, len(close)):
        if losses[bar] == 0:
            out[bar] = 100.0 if gains[bar] > 0 else np.nan
        else:
            out[bar] = 100 - 100 / (1 + gains[bar] / losses[bar])
    return out


def _wilder_atr(frame, length=14):
    high, low, close = frame["high"].to_numpy(), frame["low"].to_numpy(), frame["close"].to_numpy()
    tr = np.empty(len(close))
    tr[0] = high[0] - low[0]
    for bar in range(1, len(close)):
        tr[bar] = max(high[bar] - low[bar], abs(high[bar] - close[bar - 1]),
                      abs(low[bar] - close[bar - 1]))
    current, count, total = np.nan, 0, 0.0
    out = np.full(len(close), np.nan)
    for bar in range(len(close)):
        if count < length:
            count += 1
            total += tr[bar]
            if count == length:
                current = total / length
        else:
            current = (current * (length - 1) + tr[bar]) / length
        out[bar] = current
    return out


def _heat_frame():
    """Bars 0-19 flat at 100, bars 20-26 crash 3 points per bar, bars 27-28 are
    two capitulation hammers: each closes a hair above the previous high with a
    deep lower wick, so both satisfy every long-reversal gate."""
    bars = 30
    close = np.full(bars, 100.0)
    close[20:26] = [97, 94, 91, 88, 85, 82]
    close[26] = 78.9  # gap-down open, close just under the open
    close[27], close[28], close[29] = 79.4, 79.8, 80.0
    open_ = np.empty(bars)
    open_[0], open_[1:] = 99.5, close[:-1]
    open_[26] = 79.0  # gap down from 82
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    # The last crash bar and the two hammers carry deep lower wicks; each
    # hammer closes a hair above the previous high.
    low[26] = 70.0
    low[27] = 74.0
    high[28], low[28] = 80.0, 76.0
    high[29], low[29] = 80.1, 79.7
    return _frame(bars, open_=open_, high=high, low=low, close=close)


_HEAT_KW = dict(ma_type="SMA", ma_mean_len=20, atr_distance_mult=0.4,
                atr_expansion_mult=3.0, atr_slow_len=10, confirmation_mode="Early",
                use_htf_bias=False, use_trend_filter=False)


def test_heat_reversal_hand_computed():
    frame = _heat_frame()
    result = commodity_heat_reversal(frame, counter_trend_display="Show", signal_cooldown_bars=0, **_HEAT_KW)

    close = frame["close"].to_numpy()
    open_ = frame["open"].to_numpy()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    ma50 = close.rolling if False else frame["close"].rolling(20, min_periods=20).mean()
    atr = _wilder_atr(frame)
    rsi = _wilder_rsi(close)

    # Intermediates at the two hammer bars.
    for bar in (27, 28):
        assert result.ma_mean.iloc[bar] == pytest.approx(ma50.iloc[bar])
        assert result.atr_fast.iloc[bar] == pytest.approx(atr[bar])
        assert result.rsi.iloc[bar] == pytest.approx(rsi[bar])
        candle_range = high[bar] - low[bar]
        assert result.lower_wick_pct.iloc[bar] == pytest.approx(
            (min(open_[bar], close[bar]) - low[bar]) / candle_range)
        assert result.long_heat.iloc[bar] == pytest.approx(
            1.0 if close[bar] < ma50.iloc[bar] - 0.4 * atr[bar] else 0.0)
        expansion = atr[bar] > atr[bar - 9:bar + 1].mean() * 3.0
        bb_lower = (ma50.iloc[bar] - 2 * frame["close"].rolling(20, min_periods=20)
                    .std(ddof=0).iloc[bar])
        wick_pct = (min(open_[bar], close[bar]) - low[bar]) / candle_range
        expected_score = (2 * bool(close[bar] < ma50.iloc[bar] - 0.4 * atr[bar])
                          + 2 * bool(expansion)
                          + bool(rsi[bar] < 28)
                          + bool(wick_pct > 0.35)
                          + bool(close[bar] < bb_lower))
        assert int(result.long_score.iloc[bar]) == expected_score
        # Stop level: 10-bar lowest low minus 1.2 ATR.
        assert result.long_sl.iloc[bar] == pytest.approx(low[bar - 9:bar + 1].min() - 1.2 * atr[bar])

    # Both hammers are raw signals (rsi far below 28, deep wick, stretch).
    assert result.rsi.iloc[27] < 28 and result.rsi.iloc[28] < 28
    assert result.long_reversal_signal.tolist() == [False] * 27 + [True, True, False]
    assert result.short_reversal_signal.tolist() == [False] * 30

    # Hide (the upstream default) suppresses counter-trend reversals; Mark
    # surfaces them as markers instead.
    hidden = commodity_heat_reversal(frame, counter_trend_display="Hide", signal_cooldown_bars=0, **_HEAT_KW)
    assert not hidden.long_reversal_signal.any()
    marked = commodity_heat_reversal(frame, counter_trend_display="Mark", signal_cooldown_bars=0, **_HEAT_KW)
    assert not marked.long_reversal_signal.any()
    assert marked.long_counter_trend_signal.tolist() == [False] * 27 + [True, True, False]

    # A cooldown of one bar suppresses the second adjacent hammer.
    cooled = commodity_heat_reversal(frame, counter_trend_display="Show",
                                     signal_cooldown_bars=1, **_HEAT_KW)
    assert cooled.long_reversal_signal.tolist() == [False] * 27 + [True, False, False]

    # HTF columns stay empty with the bias disabled; continuation stays off.
    assert result.htf_close.isna().all()
    assert not result.long_continuation_signal.any()
    assert not result.short_continuation_signal.any()
    assert result.htf_up_trend.tolist() == [True] * 30


def test_heat_reversal_rejects_bad_input():
    frame = _heat_frame()
    with pytest.raises(ValueError, match="ma_type"):
        commodity_heat_reversal(frame, ma_type="ZLEMA")
    with pytest.raises(ValueError, match="reversal_heat_threshold"):
        commodity_heat_reversal(frame, reversal_heat_threshold=3)
    with pytest.raises(ValueError, match="confirmation_mode"):
        commodity_heat_reversal(frame, confirmation_mode="Loose")
    with pytest.raises(ValueError, match="counter_trend_display"):
        commodity_heat_reversal(frame, counter_trend_display="Blink")
    with pytest.raises(ValueError, match="jma_power"):
        commodity_heat_reversal(frame, jma_power=5.0)
    with pytest.raises(ValueError, match="open"):
        commodity_heat_reversal(frame.drop(columns="open"))


# ---------------------------------------------------------------------------
# gcr_strategy_scanner_tradingview
# ---------------------------------------------------------------------------

from app.catalog_indicators.gcr_strategy_scanner_tradingview import (
    gcr_strategy_scanner_tradingview,
)


def _pine_ema(values, length):
    out = [float("nan")] * len(values)
    if len(values) < length:
        return out
    alpha = 2.0 / (length + 1.0)
    previous = sum(values[:length]) / length
    out[length - 1] = previous
    for position in range(length, len(values)):
        previous = alpha * values[position] + (1.0 - alpha) * previous
        out[position] = previous
    return out


_GCR_KW = dict(rsi_length=5, rsi_ob=70, rsi_os=30, stoch_length=5, stoch_ob=80,
               stoch_os=20, vol_ma_length=5, vol_spike_mult=2.0,
               vol_exhaustion_mult=2.5, bb_length=5, bb_mult=2.0, atr_length=5,
               pivot_left=2, pivot_right=1, ema_fast=3, ema_slow=5, ema_trend=8,
               div_lookback=10, enable_divergence=True)


def _gcr_blowoff_frame():
    """A gentle 0.5-per-bar rally for 20 bars, then one red blow-off bar on 5x
    volume with its high far above the upper Bollinger band."""
    bars = 25
    close = np.full(bars, 100.0)
    close[:20] = 100 + 0.5 * np.arange(20)
    close[20] = 111.0
    close[21:] = [110.5, 110.0, 109.5, 109.0]
    open_ = close - 0.3
    open_[20] = 110.0
    open_[21:] = close[20:-1]
    high = close + 0.4
    low = close - 0.7
    high[20], low[20] = 116.0, 109.8
    high[21:] = close[21:] + 0.2
    low[21:] = close[21:] - 0.5
    volume = np.full(bars, 100.0)
    volume[20] = 500.0
    return _frame(bars, open_=open_, high=high, low=low, close=close, volume=volume)


def test_gcr_blowoff_hand_computed():
    frame = _gcr_blowoff_frame()
    result = gcr_strategy_scanner_tradingview(frame, **_GCR_KW)
    bar = 20
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]

    expected_rsi = _wilder_rsi(close.to_numpy(), 5)
    assert result.rsi.iloc[bar] == pytest.approx(expected_rsi[bar])
    window_high = high.iloc[bar - 4:bar + 1].max()
    window_low = low.iloc[bar - 4:bar + 1].min()
    expected_stoch_k = 100 * (close.iloc[bar] - window_low) / (window_high - window_low)
    assert result.stoch_k.iloc[bar] == pytest.approx(expected_stoch_k)
    assert result.vol_ma.iloc[bar] == pytest.approx(180.0)  # includes the spike bar
    assert result.relative_volume.iloc[bar] == pytest.approx(500.0 / 180.0)
    bb_basis = close.iloc[bar - 4:bar + 1].mean()
    bb_dev = close.iloc[bar - 4:bar + 1].std(ddof=0)
    assert result.bb_upper.iloc[bar] == pytest.approx(bb_basis + 2 * bb_dev)
    expected_atr = _wilder_atr(frame, 5)
    assert result.atr.iloc[bar] == pytest.approx(expected_atr[bar])
    assert result.atr_percent.iloc[bar] == pytest.approx(100 * expected_atr[bar] / close.iloc[bar])

    # Every bearish gate fires on the blow-off bar; the divergence does not
    # (RSI sat at 100 on both window maxima), so the signal is not "strong".
    assert result.rsi.iloc[bar] >= 70
    assert result.vol_spike.iloc[bar] and result.vol_exhaustion.iloc[bar]
    assert bool(result.bearish_inflection.iloc[bar])
    assert not bool(result.bearish_divergence.iloc[bar])
    assert not bool(result.strong_bearish.iloc[bar])
    assert not bool(result.bearish_volume_signal.iloc[bar])  # close > open
    expected_score = ((50 - expected_rsi[bar]) / 2 + (50 - expected_stoch_k) * 0.3
                      + 10.0 + 0.0 - 25.0)
    assert result.score.iloc[bar] == pytest.approx(expected_score)

    # A quiet rally bar only carries the RSI and Stochastic contributions.
    quiet = 10
    quiet_stoch = 100 * (close.iloc[quiet] - low.iloc[quiet - 4:quiet + 1].min()) / (
        high.iloc[quiet - 4:quiet + 1].max() - low.iloc[quiet - 4:quiet + 1].min())
    expected_quiet = ((50 - expected_rsi[quiet]) / 2 + (50 - quiet_stoch) * 0.3)
    assert result.score.iloc[quiet] == pytest.approx(expected_quiet)
    assert not result.vol_spike.iloc[:20].any()

    # Trend EMAs use the Pine SMA-seeded recursion.
    assert result.ema_fast.iloc[bar] == pytest.approx(_pine_ema(close.to_numpy(), 3)[bar])
    assert result.ema_slow.iloc[bar] == pytest.approx(_pine_ema(close.to_numpy(), 5)[bar])
    assert result.ema_trend.iloc[bar] == pytest.approx(_pine_ema(close.to_numpy(), 8)[bar])
    assert bool(result.trend_up.iloc[bar])


def test_gcr_double_top_divergence():
    """A rally tops out at bar 9 (high 118.4), dips, and bar 14 retests the old
    peak exactly (high == its own window max and the prior window max), so
    `valuewhen(high == price_high_2, rsi, 0)` samples that retest. Price then
    pushes a higher high on fading RSI: exactly one bearish divergence at bar
    19."""
    bars = 22
    close = np.concatenate([
        100 + 2.0 * np.arange(10),            # bars 0-9: 100..118
        [116.5, 114.5, 113.0, 112.0],         # bars 10-13: dip
        [118.0],                              # bar 14: retest of the old peak
        [121.0, 122.0, 119.0, 121.0, 122.5],  # bars 15-19: higher high, fading RSI
        [121.0, 119.5],                       # bars 20-21
    ])
    open_ = np.concatenate([[99.7], close[:-1]])
    high = close + 0.4
    low = close - 0.7
    frame = _frame(bars, open_=open_, high=high, low=low, close=close)
    result = gcr_strategy_scanner_tradingview(frame, **_GCR_KW)
    expected_rsi = _wilder_rsi(close, 5)

    assert result.bearish_divergence.tolist() == [False] * 19 + [True] + [False] * 2
    # RSI at the retest (bar 14) is materially higher than at the new top.
    assert expected_rsi[14] > 60
    assert expected_rsi[19] > 60
    assert expected_rsi[19] < expected_rsi[14]
    assert not result.bullish_divergence.any()

    # Pivot highs (left=2, right=1): bar 9 confirms on 10, bar 16 on 17 and
    # bar 19 on 20.
    assert result.pivot_high.iloc[10] == pytest.approx(118.4)
    assert result.pivot_high.iloc[17] == pytest.approx(122.4)
    assert result.pivot_high.iloc[20] == pytest.approx(122.9)

    # Volume never spikes, so no inflection fires anywhere in this frame.
    assert not result.bearish_inflection.any()
    assert not result.bullish_inflection.any()


def test_gcr_rejects_bad_input():
    frame = _gcr_blowoff_frame()
    with pytest.raises(ValueError, match="rsi_ob"):
        gcr_strategy_scanner_tradingview(frame, rsi_ob=45)
    with pytest.raises(ValueError, match="stoch_os"):
        gcr_strategy_scanner_tradingview(frame, stoch_os=3)
    with pytest.raises(ValueError, match="vol_spike_mult"):
        gcr_strategy_scanner_tradingview(frame, vol_spike_mult=0.5)
    with pytest.raises(ValueError, match="bb_mult"):
        gcr_strategy_scanner_tradingview(frame, bb_mult=0.2)
    with pytest.raises(ValueError, match="div_lookback"):
        gcr_strategy_scanner_tradingview(frame, div_lookback=5)
    with pytest.raises(ValueError, match="rsi_length"):
        gcr_strategy_scanner_tradingview(frame, rsi_length=0)
    with pytest.raises(ValueError, match="volume"):
        gcr_strategy_scanner_tradingview(frame.drop(columns="volume"))
    with pytest.raises(ValueError, match="open"):
        gcr_strategy_scanner_tradingview(frame.drop(columns="open"))


# ---------------------------------------------------------------------------
# roc_advanced
# ---------------------------------------------------------------------------

from app.catalog_indicators.roc_advanced import roc_advanced


_ROC_CLOSES = np.array([100, 101, 102, 101, 100, 100.5, 101, 102, 104, 106, 108,
                        110, 110.2, 110.4, 110.6, 109, 108, 107, 106, 105])
_ROC_KW = dict(source="close", roc_length=3, avg_length=2, signal_length=3,
               smoothing="SMA", midline=0.0, ob_level=5.0, os_level=-5.0,
               use_stall=True, stall_len=3, stall_flat_atr=0.7, stall_roc_min=1.5,
               ctx_len=6, lookback_extreme=3, show_div=True,
               div_len=2, div_min=1.0)


def test_roc_advanced_hand_computed():
    frame = _frame(len(_ROC_CLOSES), close=_ROC_CLOSES)
    result = roc_advanced(frame, use_ctx=True, use_extreme_only=True, **_ROC_KW)
    close = frame["close"]

    # f_roc: 100*(x - x[len])/x[len], warm-up NaN.
    roc_raw = 100 * (close - close.shift(3)) / close.shift(3)
    assert result.roc_raw.iloc[3] == pytest.approx(1.0)
    assert result.roc_raw.iloc[4] == pytest.approx(-100.0 / 101.0)
    assert result.roc_raw.iloc[:3].isna().all()
    assert result.roc_raw.iloc[3:].notna().all()
    roc_line = roc_raw.rolling(2, min_periods=2).mean()
    signal = roc_line.rolling(3, min_periods=3).mean()
    ctx = (100 * (close - close.shift(6)) / close.shift(6)).rolling(2, min_periods=2).mean()

    for bar in range(len(frame)):
        assert result.roc.iloc[bar] == pytest.approx(roc_line.iloc[bar], nan_ok=True)
        assert result.signal.iloc[bar] == pytest.approx(signal.iloc[bar], nan_ok=True)
        assert result.histogram.iloc[bar] == pytest.approx(
            roc_line.iloc[bar] - signal.iloc[bar], nan_ok=True)
        assert result.ctx.iloc[bar] == pytest.approx(ctx.iloc[bar], nan_ok=True)

    # Wilder ATR over the near-constant 2-point true ranges.
    expected_atr = _wilder_atr(frame, 14)
    for bar in range(13, len(frame)):
        assert result.atr_ref.iloc[bar] == pytest.approx(expected_atr[bar])

    # Crosses: ROC leaves the signal line up at bar 7 (and flickers up at 19),
    # crosses it down at 12, and crosses the midline at 5 / 15.
    assert result.bull_cross.tolist() == [False] * 7 + [True] + [False] * 11 + [True]
    assert result.bear_cross.tolist() == [False] * 12 + [True] + [False] * 7
    assert result.bull_mid_cross.tolist() == [False] * 7 + [True] + [False] * 12
    assert result.bear_mid_cross.tolist() == [False] * 5 + [True] + [False] * 9 \
        + [True] + [False] * 4

    # Stretch gates: the ROC never dipped to -5, so the bull extreme needs the
    # gate off; the bear extreme at 12 saw max(roc, 3) = 7.67 >= 5.
    assert not result.bull_extreme.any()
    assert result.bear_extreme.tolist() == [False] * 12 + [True] + [False] * 7
    ungated = roc_advanced(frame, use_ctx=True, use_extreme_only=False, **_ROC_KW)
    assert ungated.bull_extreme.tolist() == [False] * 7 + [True] + [False] * 11 + [True]
    assert ungated.bear_extreme.tolist() == result.bear_extreme.tolist()

    # Stall: ROC collapses while price is flat within 0.7 ATR at bars 14-15.
    assert result.stall_bull.tolist() == [False] * 14 + [True, True] + [False] * 4
    assert not result.stall_bear.any()

    # Sentiment: clamped to +100 deep in the stretch zone, negative on the way
    # down, NaN during warm-up.
    assert math.isnan(result.sentiment_score.iloc[3])
    assert result.sentiment_score.iloc[10] == pytest.approx(100.0)
    expected_sent16 = -(0 - roc_line.iloc[16]) / 5.0 * 100.0
    assert result.sentiment_score.iloc[16] == pytest.approx(expected_sent16)

    # Quality: depth (50%) + context agreement (25%) + stall-freedom (25%).
    roc_min15 = roc_line.rolling(3, min_periods=3).min().iloc[15]
    roc_max15 = roc_line.rolling(3, min_periods=3).max().iloc[15]
    bull_depth15 = min(max((-5.0 - roc_min15) / 5.0, 0.0), 1.0)
    bear_depth15 = min(max((roc_max15 - 5.0) / 5.0, 0.0), 1.0)
    assert result.bull_quality.iloc[15] == pytest.approx(min(100.0, bull_depth15 * 50.0
                                                            + 25.0 * bool(ctx.iloc[15] > 0)
                                                            + 25.0))
    assert result.bear_quality.iloc[15] == pytest.approx(min(100.0, bear_depth15 * 50.0
                                                            + 25.0 * bool(ctx.iloc[15] < 0)
                                                            + 25.0 * (not result.stall_bull.iloc[15])))

    # Divergence: fast ROC collapse needs a still-rising context line, which
    # the shared smoothing never produces on this data.
    assert not result.div_bull.any()
    assert not result.div_bear.any()

    # Without the context toggle only the counter-trend/divergence reads are
    # gated; the context line itself is still reported.
    no_ctx = roc_advanced(frame, use_ctx=False, **_ROC_KW)
    assert not no_ctx.div_bull.any()
    assert not no_ctx.div_bear.any()
    assert no_ctx.ctx_bull.tolist() == result.ctx_bull.tolist()


def test_roc_advanced_rejects_bad_input():
    frame = _frame(len(_ROC_CLOSES), close=_ROC_CLOSES)
    with pytest.raises(ValueError, match="source"):
        roc_advanced(frame, source="vwap")
    with pytest.raises(ValueError, match="smoothing"):
        roc_advanced(frame, smoothing="ZLEMA")
    with pytest.raises(ValueError, match="ob_level"):
        roc_advanced(frame, ob_level=-1.0)
    with pytest.raises(ValueError, match="os_level"):
        roc_advanced(frame, os_level=1.0)
    with pytest.raises(ValueError, match="ctx_len"):
        roc_advanced(frame, ctx_len=1)
    with pytest.raises(ValueError, match="high"):
        roc_advanced(frame.drop(columns="high"))


# ---------------------------------------------------------------------------
# commodity_flowtrend
# ---------------------------------------------------------------------------

from app.catalog_indicators.commodity_flowtrend import commodity_flowtrend


_FLOW_CLOSES = np.array([100, 101, 102, 101, 100, 99, 100, 101, 102, 103, 104,
                         103, 102, 103, 104], dtype=float)
_FLOW_KW = dict(source="close", mfi_len=3, cci_len=4, sig_len=2, smoothing="SMA",
                mfi_ob=70, mfi_os=30, mfi_mid=50, trigger_mode="MFI Lead",
                use_extreme_only=True, lookback_extreme=3, price_trend_len=10)


def test_flowtrend_hand_computed():
    frame = _frame(len(_FLOW_CLOSES), close=_FLOW_CLOSES)
    result = commodity_flowtrend(frame, use_price_trend_filter=False, **_FLOW_KW)
    close = frame["close"]
    volume = frame["volume"]

    # Custom MFI: flow = close * volume signed by the close's own direction.
    flow = close * volume
    positive = flow.where(close > close.shift(1), 0.0)
    negative = flow.where(close < close.shift(1), 0.0)
    pos_sum = positive.rolling(3, min_periods=3).sum()
    neg_sum = negative.rolling(3, min_periods=3).sum()
    expected_raw = 100.0 - 100.0 / (1.0 + pos_sum / neg_sum.where(neg_sum > 0))
    expected_raw = expected_raw.mask(neg_sum.eq(0) & pos_sum.gt(0), 100.0)
    expected_raw = expected_raw.mask(pos_sum.eq(0) & neg_sum.gt(0), 0.0)
    for bar in range(len(frame)):
        assert result.mfi_raw.iloc[bar] == pytest.approx(expected_raw.iloc[bar], nan_ok=True)

    # Hand values: all-positive window -> 100, all-negative -> 0, mixed 2:1.
    assert result.mfi_raw.iloc[2] == pytest.approx(100.0)
    assert result.mfi_raw.iloc[5] == pytest.approx(0.0)
    assert result.mfi_raw.iloc[3] == pytest.approx(100.0 - 100.0 / (1.0 + 20300.0 / 10100.0))

    # Double-smoothed (SMA) MFI, signal and histogram.
    expected_mfi = expected_raw.rolling(2, min_periods=2).mean()
    expected_signal = expected_mfi.rolling(2, min_periods=2).mean()
    for bar in range(len(frame)):
        assert result.mfi.iloc[bar] == pytest.approx(expected_mfi.iloc[bar], nan_ok=True)
        assert result.mfi_signal.iloc[bar] == pytest.approx(expected_signal.iloc[bar],
                                                            nan_ok=True)
        assert result.histogram.iloc[bar] == pytest.approx(
            expected_mfi.iloc[bar] - expected_signal.iloc[bar], nan_ok=True)

    # CCI of the same source, double-smoothed, normalised into 0..100.
    # ta.dev is the mean absolute deviation *inside* the same window, not a
    # rolling MAD of the rolling mean.
    avg = close.rolling(4, min_periods=4).mean()
    dev4 = close.rolling(4, min_periods=4).apply(lambda w: np.abs(w - w.mean()).mean(), raw=True)
    cci_raw = (close - avg) / (0.015 * dev4.where(dev4 > 0))
    expected_cci = cci_raw.rolling(2, min_periods=2).mean()
    # pandas clip keeps NaN as NaN (Python's max() would swallow it into 0.0).
    expected_norm = (50.0 + expected_cci / 4.0).clip(0.0, 100.0)
    for bar in range(len(frame)):
        assert result.cci.iloc[bar] == pytest.approx(expected_cci.iloc[bar], nan_ok=True)
        assert result.cci_norm.iloc[bar] == pytest.approx(expected_norm.iloc[bar], nan_ok=True)

    # MFI path 0 -> 33 -> 67 -> 100: the cross above the signal at bar 7 comes
    # straight out of the oversold zone with CCI positive -> bull signal.
    assert result.mfi.iloc[5] == pytest.approx(16.83, abs=0.01)
    assert result.mfi.iloc[8] == pytest.approx(83.50, abs=0.01)
    assert result.bull_signal.tolist() == [False] * 7 + [True] + [False] * 7
    assert result.bull_expansion.iloc[8]
    assert result.bull_mid_confirm.tolist() == [False] * 7 + [True] + [False] * 6 + [True]

    # The bear cross at bar 11 is CCI-confirmed too (the smoothed CCI is below
    # zero and falling), so both the default and unconfirmed runs fire it.
    assert result.bear_signal.tolist() == [False] * 11 + [True] + [False] * 3
    unconfirmed = commodity_flowtrend(frame, use_price_trend_filter=False,
                                      use_cci_confirm=False, **_FLOW_KW)
    assert unconfirmed.bear_signal.tolist() == [False] * 11 + [True] + [False] * 3

    # Extreme-zone turn markers on the smoothed CCI.
    assert result.cci_turning_up.tolist() == [False] * 6 + [True] + [False] * 8
    assert result.cci_turning_down.tolist() == [False] * 9 + [True] + [False] * 5

    # Price-trend filter: with a short EMA the bar-7 bull signal only survives
    # when the close is above the trend line.
    filtered = commodity_flowtrend(frame, use_price_trend_filter=True,
                                   **{**_FLOW_KW, "price_trend_len": 10})
    trend = _pine_ema(_FLOW_CLOSES, 10)
    for bar in range(len(frame)):
        expected_bull = bool(bar == 7 and not math.isnan(trend[bar])
                             and _FLOW_CLOSES[bar] > trend[bar])
        assert bool(filtered.bull_signal.iloc[bar]) == expected_bull


def test_flowtrend_zero_volume_and_bad_input():
    frame = _frame(len(_FLOW_CLOSES), close=_FLOW_CLOSES, volume=np.zeros(len(_FLOW_CLOSES)))
    result = commodity_flowtrend(frame, **_FLOW_KW)
    assert result.mfi_raw.isna().all()
    assert result.mfi.isna().all()
    assert result.cci.notna().any()  # CCI needs no volume
    assert not result.bull_signal.any()
    assert not result.bear_signal.any()

    good = _frame(len(_FLOW_CLOSES), close=_FLOW_CLOSES)
    with pytest.raises(ValueError, match="source"):
        commodity_flowtrend(good, source="vwap")
    with pytest.raises(ValueError, match="trigger_mode"):
        commodity_flowtrend(good, trigger_mode="RSI Lead")
    with pytest.raises(ValueError, match="price_trend_len"):
        commodity_flowtrend(good, price_trend_len=5)
    with pytest.raises(ValueError, match="mfi_len"):
        commodity_flowtrend(good, mfi_len=0)
    with pytest.raises(ValueError, match="volume"):
        commodity_flowtrend(good.drop(columns="volume"))
