import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.btc_altcoin_trend import btc_altcoin_trend


def _steady_frame(n=96 * 12):
    index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    chart = pd.Series(100.0 + np.arange(n) * 0.05, index=index)
    btc = pd.Series(50.0 + np.arange(n) * 0.1, index=index)
    dom = pd.Series(55.0 - np.arange(n) * 0.02, index=index)
    return pd.DataFrame({"open": chart, "high": chart, "low": chart, "close": chart,
                         "btc_close": btc, "dominance_close": dom}, index=index)


def test_behaviour_matrix_on_steady_series():
    frame = _steady_frame()
    result = btc_altcoin_trend(frame, trend_type="1", timeframes=("1h", "1D"),
                               chart_trend=False)
    # Steady-up BTC against steady-down dominance: the classic alt-2x-up cell.
    late = result.iloc[-1]
    assert late["btc_trend_1h"] == "up"
    assert late["dominance_trend_1h"] == "down"
    assert late["btc_behavior_1h"] == "up_1x"
    assert late["alt_behavior_1h"] == "up_2x"
    # The daily labels need two completed days (period-1 = day 0, whose own
    # label also needs its period-1), so anything before day 2 is "-".
    assert (result["btc_trend_1D"].iloc[:96] == "-").all()
    assert result["btc_trend_1D"].iloc[-1] in ("up", "lateral")
    with pytest.raises(ValueError, match="行情列"):
        btc_altcoin_trend(frame.drop(columns="dominance_close"), trend_type="1")
    with pytest.raises(ValueError, match="未知周期标签"):
        btc_altcoin_trend(frame, timeframes=("2h",))


def test_finer_than_base_timeframes_report_no_data():
    frame = _steady_frame()
    result = btc_altcoin_trend(frame, trend_type="1", timeframes=("5m",),
                               chart_trend=False)
    assert (result["btc_trend_5m"] == "-").all()
    assert (result["alt_behavior_5m"] == "-").all()


def test_mode_two_classifies_by_ema_vs_close():
    # Chart series dips below its own EMAs on the last bars -> lateral/down.
    frame = _steady_frame()
    frame["close"] = frame["close"].astype(float)
    frame.loc[frame.index[-30:], "close"] = 80.0  # crash the tail below the ramp
    result = btc_altcoin_trend(frame, trend_type="2", timeframes=(), chart_trend=True)
    chart = result["btc_trend_chart"]
    # Early bars read lateral (EMA(20) seeded by SMA >= close on the ramp);
    # once the ramp outruns the SMA the trend reads up.
    assert (chart.iloc[:19] == "lateral").all()
    assert (chart.iloc[19:-30] == "up").all()
    # After the crash both EMAs sit above the close -> down.
    assert chart.iloc[-1] == "down"
    with pytest.raises(ValueError, match="ema1"):
        btc_altcoin_trend(frame, ema1=0)
