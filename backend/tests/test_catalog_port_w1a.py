"""按 Pine 公式手算的移植验收用例（W1A 批次）。"""
import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.alert_engine_webhook import alert_engine_webhook
from app.catalog_indicators.capital41_duo_oscillator import capital41_duo_oscillator
from app.catalog_indicators.capital41_session_vwap import capital41_session_vwap
from app.catalog_indicators.mtf_nearest_support_resistance import mtf_nearest_support_resistance
from app.catalog_indicators.multi_timeframe_trend import multi_timeframe_trend


def _bars(closes, freq="60min", tz="UTC"):
    index = pd.date_range("2026-01-01", periods=len(closes), freq=freq, tz=tz)
    values = pd.Series(closes, index=index, dtype=float)
    return pd.DataFrame({"open": values, "high": values, "low": values, "close": values}, index=index)


def test_alert_engine_webhook_ema_cross_and_payload():
    # lenFast=2, lenSlow=3: emaFast seeds with the mean of bars 0-1, emaSlow of 0-2.
    frame = _bars([10., 10., 10., 10., 11., 12., 13.])
    result = alert_engine_webhook(frame, fast_len=2, slow_len=3, symbol="BTCUSDT",
                                 exchange="BINANCE", timeframe="60")

    assert result.ema_fast.iloc[0] != result.ema_fast.iloc[0]
    assert result.ema_fast.iloc[1] == pytest.approx(10.)
    assert result.ema_slow.iloc[2] == pytest.approx(10.)
    assert result.ema_fast.iloc[4] == pytest.approx(10. + 2. / 3.)
    assert result.ema_slow.iloc[4] == pytest.approx(10.5)
    # Fast crosses above slow on bar 4 and stays above: one alert only.
    assert result.raw_bull.tolist() == [False, False, False, False, True, False, False]
    assert result.bull_signal.equals(result.raw_bull)
    assert not result.bear_signal.any()
    assert result.message.iloc[4] == (
        '{"strategy":"ema-cross","symbol":"BTCUSDT","exchange":"BINANCE","tf":"60",'
        '"side":"long","price":11,"time":"2026-01-01T04:00:00Z"}')
    assert result.message.drop(result.index[4]).eq("").all()

    falling = _bars([10., 10., 10., 10., 9., 8.])
    plain = alert_engine_webhook(falling, fast_len=2, slow_len=3, message_format="Plain text",
                                symbol="ETHUSDT", timeframe="240")
    assert plain.raw_bear.tolist() == [False, False, False, False, True, False]
    assert plain.bear_signal.equals(plain.raw_bear)
    assert plain.message.iloc[4] == "ETHUSDT 240 - short @ 9 (2026-01-01T04:00:00 UTC)"

    with pytest.raises(ValueError, match="fast_len"):
        alert_engine_webhook(frame, fast_len=0)
    with pytest.raises(ValueError, match="message_format"):
        alert_engine_webhook(frame, message_format="XML")
    with pytest.raises(ValueError, match="行情列"):
        alert_engine_webhook(pd.DataFrame({"high": [1., 2.]}))


def test_mtf_nearest_support_resistance_uses_previous_bar():
    index = pd.date_range("2026-01-01", periods=8, freq="60min", tz="UTC")
    frame = pd.DataFrame({
        "open": [12.] * 8,
        "high": [10., 11., 12., 13., 12., 11., 10., 9.],
        "low": [10., 9., 8., 7., 8., 9., 10., 11.],
        "close": [12.] * 8,
    }, index=index)

    # pivotLeft = pivotRight = 2: the high 13 (bar 3) and the low 7 (bar 3) are
    # confirmed on bar 5, and the snapshot is read one bar late.
    result = mtf_nearest_support_resistance(frame, tf1_minutes=60, use_tf2=False,
                                            use_tf3=False, pivot_left=2, pivot_right=2,
                                            source_memory=2)
    assert result.resistance_1.iloc[:6].isna().all()
    assert result.resistance_1.iloc[6:].tolist() == [13., 13.]
    assert result.support_1.iloc[:6].isna().all()
    assert result.support_1.iloc[6:].tolist() == [7., 7.]
    assert result.resistance_2.isna().all() and result.support_3.isna().all()

    # A timeframe repeating an active one is dropped, as upstream's `tf2 != tf1`.
    repeated = mtf_nearest_support_resistance(frame, tf1_minutes=60, tf2_minutes=60,
                                              use_tf3=False, pivot_left=2, pivot_right=2)
    assert repeated.resistance_2.isna().all()

    with pytest.raises(ValueError, match="pivot_left"):
        mtf_nearest_support_resistance(frame, pivot_left=1)
    with pytest.raises(ValueError, match="行情列"):
        mtf_nearest_support_resistance(frame.drop(columns="low"))


def test_capital41_duo_oscillator_fuses_rsi_and_williams():
    closes = [10., 9., 8., 9., 10.]
    frame = pd.DataFrame({"high": [value + 5 for value in closes],
                          "low": [value - 0.1 for value in closes],
                          "close": closes})
    result = capital41_duo_oscillator(frame, rsi_len=2, wr_len=2, osc_smooth=1,
                                      require_turn=False)
    assert result.rsi.iloc[2:].tolist() == pytest.approx([0., 50., 75.])
    assert result.williams_r.iloc[2] == pytest.approx(-98.3606557, rel=1e-6)
    assert result.williams_r_normalized.iloc[2] == pytest.approx(1.6393443, rel=1e-6)
    # duo_raw = (rsi + wr + 100) / 2
    assert result.duo_raw.iloc[2:].tolist() == pytest.approx(
        [0.8196721, 34.0163934, 46.5163934], rel=1e-6)
    assert result.duo_smoothed.equals(result.duo_raw)  # osc_smooth = 1
    # zones = (rsi threshold + normalized w%r threshold) / 2
    assert result.oversold_zone.iloc[0] == pytest.approx(16.5)
    assert result.overbought_zone.iloc[0] == pytest.approx(85.)
    assert result.both_oversold.tolist() == [False, False, True, False, False]
    assert not result.both_overbought.any()
    assert not result.bull_cross.any() and not result.bear_cross.any()

    turned = capital41_duo_oscillator(frame, rsi_len=2, wr_len=2, osc_smooth=3)
    # Turn confirmation: RSI was below 25 and Williams %R at/below -92 on bar 2
    # and both tick back up on bar 3.
    assert turned.both_oversold.tolist() == [False, False, False, True, False]
    assert turned.duo_smoothed.iloc[4] == pytest.approx(
        (0.8196721 + 34.0163934 + 46.5163934) / 3., rel=1e-6)

    with pytest.raises(ValueError, match="osc_smooth"):
        capital41_duo_oscillator(frame, osc_smooth=0)
    with pytest.raises(ValueError, match="行情列"):
        capital41_duo_oscillator(frame.drop(columns="high"))


def test_multi_timeframe_trend_votes_each_completed_period():
    # 48 one-hour bars rising by 1: ema_length = 2 seeds at the second bar.
    closes = [100. + value for value in range(48)]
    result = multi_timeframe_trend(_bars(closes), tf1_minutes=60, tf2_minutes=120,
                                   tf3_minutes=240, tf4_minutes=480, tf5_minutes=960,
                                   ema_length=2)
    assert result.ema.iloc[1] == pytest.approx(100.5)
    # Every timeframe reads the previous closed period, so each one turns one
    # period after its own EMA becomes defined.
    assert result.trend_1.iloc[:3].tolist() == [0, 0, 1]
    assert result.trend_2.iloc[:5].tolist() == [0, 0, 0, 0, 1]
    assert result.trend_3.iloc[:9].tolist() == [0] * 8 + [1]
    assert result.trend_4.iloc[:17].tolist() == [0] * 16 + [1]
    assert result.trend_5.iloc[:33].tolist() == [0] * 32 + [1]
    assert result.trend_score.iloc[15] == 3 and result.trend_score.iloc[16] == 4
    assert result.trend_score.iloc[-1] == 5
    assert result.trend_strength.iloc[-1] == pytest.approx(100.)
    assert result.overall_bias.iloc[-1] == "strong_bullish"
    assert not result.strong_bullish.iloc[15] and result.strong_bullish.iloc[16]
    assert result.strong_bullish_start.tolist() == [index == 16 for index in range(48)]
    assert result.bullish_confluence.tolist() == [index == 8 for index in range(48)]
    assert not result.strong_bearish.any() and not result.bearish_confluence.any()

    falling = multi_timeframe_trend(_bars([200. - value for value in range(48)]),
                                    tf1_minutes=60, tf2_minutes=120, tf3_minutes=240,
                                    tf4_minutes=480, tf5_minutes=960, ema_length=2)
    assert falling.trend_score.iloc[-1] == -5
    assert falling.overall_bias.iloc[-1] == "strong_bearish"
    assert falling.strong_bearish_start.tolist() == [index == 16 for index in range(48)]

    with pytest.raises(ValueError, match="ema_length"):
        multi_timeframe_trend(_bars(closes), ema_length=0)
    with pytest.raises(ValueError, match="行情列"):
        multi_timeframe_trend(pd.DataFrame({"high": [1., 2.]}))


def test_capital41_session_vwap_resets_and_bands():
    index = pd.DatetimeIndex(["2026-01-05 09:00", "2026-01-05 09:15", "2026-01-05 09:30",
                              "2026-01-05 09:45", "2026-01-05 10:00", "2026-01-05 16:00",
                              "2026-01-06 09:30"], tz="UTC")
    closes = [100., 101., 102., 103., 104., 104., 200.]
    frame = pd.DataFrame({"high": [value + 1 for value in closes],
                          "low": [value - 1 for value in closes],
                          "close": closes,
                          "volume": [10.] * 7}, index=index)

    result = capital41_session_vwap(frame, session_tz="Etc/UTC")
    assert result.in_session.tolist() == [False, False, True, True, True, False, True]
    assert result.session_start.tolist() == [False, False, True, False, False, False, True]
    # hlc3 == close here, so cumPV / cumV is the volume-weighted mean of closes.
    assert result.vwap.iloc[:2].isna().all()
    assert result.vwap.iloc[2:7].tolist() == pytest.approx([102., 102.5, 103., 103., 200.])
    assert result.vwap_upper.iloc[2] == pytest.approx(102.)
    assert result.vwap_upper.iloc[3] == pytest.approx(103.)
    assert result.vwap_lower.iloc[3] == pytest.approx(102.)
    assert result.vwap_upper.iloc[4] == pytest.approx(103.816496581, rel=1e-9)
    assert result.vwap_lower.iloc[4] == pytest.approx(102.183503419, rel=1e-9)
    # Out of session the accumulators keep their last values.
    assert result.vwap.iloc[5] == pytest.approx(103.)
    assert result.vwap_cross_up.tolist() == [False, False, False, True, False, False, False]
    assert result.upper_band_cross_up.tolist() == [False, False, False, False, True, False, False]
    assert not result.vwap_cross_down.any() and not result.lower_band_cross_down.any()

    with pytest.raises(ValueError, match="band_mult"):
        capital41_session_vwap(frame, band_mult=5.)
    with pytest.raises(ValueError, match="session_times"):
        capital41_session_vwap(frame, session_times="0930")
    with pytest.raises(ValueError, match="行情列"):
        capital41_session_vwap(frame.drop(columns="volume"))
