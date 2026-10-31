import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.capital41_adx_filter import capital41_adx_filter
from app.catalog_indicators.capital41_fusion_oscillator import capital41_fusion_oscillator
from app.catalog_indicators.cic import cic
from app.catalog_indicators.mtf_structure_bias import mtf_structure_bias
from app.catalog_indicators.mvrv_regime_monitor import mvrv_regime_monitor


def _frame(close, volume=None, index=None):
    values = pd.Series(close, dtype=float, index=index)
    frame = pd.DataFrame({
        "open": values,
        "high": values + 1.0,
        "low": values - 1.0,
        "close": values,
    }, index=index)
    frame["volume"] = 10.0 if volume is None else volume
    return frame


def test_cic_averages_correlations_in_fisher_space():
    log_returns = np.array([0.02, -0.01, 0.03, 0.005, -0.02, 0.015, -0.004, 0.01, -0.03, 0.02,
                            0.011, -0.007, 0.018, -0.012, 0.006, -0.02, 0.013, 0.004, -0.009, 0.017])
    # Exact log returns keep the mirror series perfectly anti-correlated.
    bench = 100 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))
    btc = bench.copy()
    mirror = 100 * np.exp(np.concatenate([[0.0], np.cumsum(-log_returns)]))
    frame = pd.DataFrame({"btc_close": btc, "eth_close": mirror, "bench_close": bench})
    result = cic(frame, length=15, regime_smoothing_bars=1)
    assert result.btc_corr.iloc[-1] == pytest.approx(1.0)
    assert result.eth_corr.iloc[-1] == pytest.approx(-1.0)
    assert result.average_corr.iloc[-1] == pytest.approx(0.0, abs=1e-6)
    assert result.regime_state.iloc[-1] == "Decouple"
    assert result.btc_state.iloc[-1] == "Sync" and result.eth_state.iloc[-1] == "Inverse Sync"
    assert result.btc_corr.iloc[:15].isna().all()  # one return plus the lookback
    with pytest.raises(ValueError, match="length"):
        cic(frame, length=10)
    with pytest.raises(ValueError, match="行情列"):
        cic(frame.drop(columns="bench_close"), length=15)


def test_capital41_adx_filter_tracks_a_one_way_trend():
    close = pd.Series([100.0 + 0.5 * step for step in range(40)])
    frame = pd.DataFrame({"high": close + 1, "low": close - 1, "close": close})
    result = capital41_adx_filter(frame, di_length=3, adx_smoothing=3,
                                  trend_threshold=20, strong_threshold=35)
    assert result.di_plus.iloc[-1] == pytest.approx(25.0, abs=1e-4)
    assert result.di_minus.iloc[-1] == 0.0
    assert result.adx.iloc[-1] == pytest.approx(100.0)
    assert result.bull_direction.iloc[-1] and not result.bear_direction.iloc[-1]
    assert result.trending.iloc[-1] and result.strong_trend.iloc[-1]
    # ADX jumps from undefined straight to 100 on a one-way trend, so Pine's
    # crossover (which needs a defined previous value) never fires here.
    assert not result.adx_cross_up.any()
    with pytest.raises(ValueError, match="di_length"):
        capital41_adx_filter(frame, di_length=1)


def test_mvrv_regime_monitor_labels_and_crossings():
    frame = pd.DataFrame({"mvrv": [1.2, 0.9, 2.1, 2.6, 3.6, np.nan]})
    result = mvrv_regime_monitor(frame)
    assert result.regime.tolist() == ["ACCUMULATION", "ENTRY ZONE", "DISTRIBUTION RISK",
                                      "EXIT ZONE", "EUPHORIA", "—"]
    assert result.cross_under_entry.tolist() == [False, True, False, False, False, False]
    assert result.cross_over_exit.tolist() == [False, False, False, True, False, False]
    assert result.cross_over_euphoria.tolist() == [False, False, False, False, True, False]
    with pytest.raises(ValueError, match="递增"):
        mvrv_regime_monitor(frame, exit_threshold=1.5)
    with pytest.raises(ValueError, match="mvrv"):
        mvrv_regime_monitor(pd.DataFrame({"close": [1.0]}))


def test_mtf_structure_bias_scores_a_clean_uptrend():
    index = pd.date_range("2026-01-01", periods=2400, freq="15min", tz="UTC")
    close = pd.Series(np.linspace(100, 400, 2400), index=index)
    frame = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close})
    result = mtf_structure_bias(frame, lookback=5, htf1_minutes=60, htf2_minutes=240,
                                htf3_minutes=1440)
    assert result.structure_base.iloc[-1] == 1
    assert result.structure_htf3.iloc[-1] == 1
    assert result.score.iloc[-1] == pytest.approx(100.0)
    assert result.score.iloc[:900].isna().all()  # daily structure needs ten completed days
    # The score jumps from undefined straight to 100, so the 60 crossover never fires.
    assert not result.bull_zone_entry.any() and not result.bear_zone_entry.any()
    descending = pd.Series(close.iloc[::-1].to_numpy(), index=index)
    falling = pd.DataFrame({"open": descending, "high": descending + 0.5,
                            "low": descending - 0.5, "close": descending})
    assert mtf_structure_bias(falling, lookback=5).score.iloc[-1] == pytest.approx(-100.0)
    with pytest.raises(ValueError, match="lookback"):
        mtf_structure_bias(frame, lookback=4)


def test_capital41_fusion_oscillator_scores_trend_vote_only():
    close = pd.Series([100.0 + step for step in range(30)])
    frame = pd.DataFrame({"high": close + 1, "low": close - 1, "close": close, "volume": 10.0})
    result = capital41_fusion_oscillator(frame, ema_length=5, rsi_length=3, wr_length=3,
                                         atr_length=3, atr_average_length=5, volume_length=5,
                                         oscillator_smoothing=1)
    assert result.trend_long.iloc[-1] and not result.trend_short.iloc[-1]
    assert result.rsi.iloc[-1] == 100 and not result.rsi_long.iloc[-1]
    assert result.williams_r.iloc[-1] == pytest.approx(-25.0)  # one above the high in a four-wide range
    assert result.score_long.iloc[-1] == 1  # trend only: RSI and W%R are not oversold
    assert result.fusion.iloc[-1] == pytest.approx(62.5)
    assert result.fusion.between(0, 100).all()
    with pytest.raises(ValueError, match="oscillator_smoothing"):
        capital41_fusion_oscillator(frame, oscillator_smoothing=0)
    with pytest.raises(ValueError, match="volume"):
        capital41_fusion_oscillator(frame.drop(columns="volume"))
