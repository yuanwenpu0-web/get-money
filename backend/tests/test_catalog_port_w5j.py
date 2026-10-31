import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.sessionvwap_sdbands import sessionvwap_sdbands


def _session_frame(days=3):
    # 15-minute bars over full UTC days; the RTH session 0930-1600 keeps 26 bars.
    index = pd.date_range("2026-01-01", periods=96 * days, freq="15min", tz="UTC")
    t = np.arange(len(index))
    close = pd.Series(100 + 0.05 * t + 3.0 * np.sin(t / 8.0), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.2,
                          "low": np.minimum(open_, close) - 0.2, "close": close})
    frame["volume"] = 10.0
    return frame


def test_vwap_resets_each_session_and_bands_scale_with_sd():
    frame = _session_frame()
    result = sessionvwap_sdbands(frame, session="0000-0000", timezone="UTC",
                                  signal_band="1 SD", use_vol_filter=False,
                                  dedup_bars=1)
    in_session = result.in_session
    assert in_session.all()  # the 24h session covers every bar
    resets = result.new_session
    assert resets.sum() == 3  # one per day
    # Band spacing is symmetric around the VWAP and ordered 1 < 2 < 3.
    active = result.dropna(subset=["vwap"])
    spread1 = (active.sd1_upper - active.sd1_lower) / 2
    spread2 = (active.sd2_upper - active.sd2_lower) / 2
    spread3 = (active.sd3_upper - active.sd3_lower) / 2
    assert (spread3 >= spread2).all() and (spread2 >= spread1).all()
    assert ((active.sd1_upper + active.sd1_lower) / 2 == active.vwap).all()
    with pytest.raises(ValueError, match="session"):
        sessionvwap_sdbands(frame, session="bad")
    with pytest.raises(ValueError, match="timezone"):
        sessionvwap_sdbands(frame, timezone="Mars/Olympus")


def test_rth_session_blanks_the_overnight_bars():
    frame = _session_frame()
    result = sessionvwap_sdbands(frame, session="0930-1600", timezone="UTC",
                                  signal_band="2 SD", use_vol_filter=False)
    inside = result.loc[result.in_session]
    outside = result.loc[~result.in_session]
    assert len(inside) == 26 * 3
    assert outside.vwap.isna().all()
    assert (outside.state == "-").all()
    # First bar of each session is a reset, so it has no touch signal.
    starts = inside.index[inside.loc[:, "new_session"]]
    assert len(starts) == 3
    assert not result.loc[starts, "buy_signal"].any()


def test_touch_signals_respect_volume_and_dedup():
    # Force a touch: price spikes above the 2 SD upper band on one bar.
    frame = _session_frame(2)
    spike_at = 96 + 40  # second day, mid-session
    frame.iloc[spike_at, frame.columns.get_loc("high")] = frame.close.iloc[spike_at] + 60.0
    quiet = sessionvwap_sdbands(frame, session="0000-0000", timezone="UTC",
                                 signal_band="2 SD", use_vol_filter=True,
                                 vol_mult=1.2, dedup_bars=1)
    assert not quiet.sell_signal.any()  # constant volume never clears the filter
    loud_volume = frame["volume"].astype(float).copy()
    loud_volume.iloc[spike_at] = 1000.0
    frame["volume"] = loud_volume
    loud = sessionvwap_sdbands(frame, session="0000-0000", timezone="UTC",
                                signal_band="2 SD", use_vol_filter=True,
                                vol_mult=1.0, dedup_bars=1)
    assert loud.sell_touch.any()
    assert loud.sell_signal.any()
    # A long dedup window swallows the follow-up touch.
    chilled = sessionvwap_sdbands(frame, session="0000-0000", timezone="UTC",
                                   signal_band="2 SD", use_vol_filter=False,
                                   dedup_bars=200)
    assert chilled.sell_signal.sum() <= 1
    with pytest.raises(ValueError, match="signal_band"):
        sessionvwap_sdbands(frame, signal_band="4 SD")
