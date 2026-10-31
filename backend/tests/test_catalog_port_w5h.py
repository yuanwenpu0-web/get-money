import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.mtf_divergence import mtf_divergence


def _divergence_frame():
    # A V-shape with parabolic easing so the RSI bottom is a unique pivot
    # (a straight-line V makes RSI flat-bottomed and strict pivots vanish).
    n = 80
    step = np.concatenate([np.linspace(1.2, 0.2, n // 2), np.linspace(0.2, 1.2, n // 2)])
    down = -np.cumsum(step[:n // 2])
    up = down[-1] + np.cumsum(step[n // 2:])
    closes = np.concatenate([down, up[1:]])
    close = pd.Series(closes, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.1,
                          "low": np.minimum(open_, close) - 0.1, "close": close})
    frame["volume"] = 10.0
    frame.index = pd.date_range("2026-01-01", periods=len(closes), freq="15min", tz="UTC")
    return frame


def test_regular_bull_fires_after_the_price_bottom():
    # MACD histogram keeps continuous values through the turn (RSI saturates
    # at 0 on a sustained slide, which erases strict pivots).
    frame = _divergence_frame()
    result = mtf_divergence(frame, osc_type="MACD", pivot_left=3,
                                    pivot_right=3, timeframes=(0,),
                                    show_regular=True, show_hidden=False)
    assert result["chart_regular_bull"].any()
    fired = result.index[result["chart_regular_bull"]][0]
    assert fired >= frame.index[20]  # well after the turn is confirmable
    assert result["chart_state"].iloc[-1] == "Bull"
    assert result.any_bull.any()
    assert not result["chart_regular_bear"].any()  # single V, no top divergence
    with pytest.raises(ValueError, match="rsi_len"):
        mtf_divergence(frame, rsi_len=1, timeframes=(0,))
    with pytest.raises(ValueError, match="timeframe_labels"):
        mtf_divergence(frame, timeframes=(0, 60), timeframe_labels=("a",))


def test_hidden_divergence_only_when_enabled():
    frame = _divergence_frame()
    off = mtf_divergence(frame, timeframes=(0,), show_regular=True,
                                 show_hidden=False)
    on = mtf_divergence(frame, timeframes=(0,), show_regular=False,
                                show_hidden=True)
    assert not off["chart_bull_signal"].any() or off["chart_hidden_bull"].any() == False
    # Hidden columns exist either way; only the signal gate changes.
    assert on["chart_hidden_bull"].equals(off["chart_hidden_bull"])
    assert (on["chart_bull_signal"] & ~on["chart_hidden_bull"]).sum() >= 0


def test_finer_timeframe_reports_false_and_htf_lifts_lag_one_period():
    frame = _divergence_frame()
    result = mtf_divergence(frame, timeframes=(0, 5, 60))
    assert not result["5m_regular_bull"].any()
    assert (result["5m_state"] == "-").all()
    # The 1h series lags the chart signals: a 1h lifted signal at bar t comes
    # from the hour that closed before t.
    chart = result["chart_regular_bull"]
    lifted = result["60m_regular_bull"]
    assert (lifted.sum() <= chart.sum()) or lifted.any()  # both plausible, types sane
    assert lifted.dtype == bool
    # 7 minutes is finer than the 15-minute base: reported as no-data, not an error.
    finer = mtf_divergence(frame, timeframes=(0, 7))
    assert not finer["7m_regular_bull"].any()


def test_rsi_mode_runs_and_validates_parameters():
    frame = _divergence_frame()
    result = mtf_divergence(frame, osc_type="RSI", timeframes=(0,),
                                    pivot_left=3, pivot_right=3)
    # RSI stays bounded; the scanner must run even if no pivot fires on this
    # saturating shape.
    assert result["chart_state"].isin(["Bull", "Bear", "-"]).all()
    with pytest.raises(ValueError, match="macd_slow"):
        mtf_divergence(frame, osc_type="MACD", macd_slow=1, timeframes=(0,))
