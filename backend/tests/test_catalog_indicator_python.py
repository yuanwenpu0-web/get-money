import pandas as pd
import pytest

from app.catalog_indicators import calculate, get_indicator
from app.api.indicators import _catalog


def test_named_indicator_dynamic_parameters_and_crossings():
    frame = pd.DataFrame({"close": [10., 9., 8., 9., 10., 9., 8.]})
    result = calculate("rsi_trading_signals", frame, rsi_length=2,
                       rsi_oversold=30, rsi_overbought=70)
    assert result.rsi.iloc[:2].isna().all()
    assert result.rsi.iloc[2:].tolist() == pytest.approx([0, 50, 75, 37.5, 18.75])
    assert result.strong_buy.tolist() == [False, False, False, True, False, False, False]
    assert result.strong_sell.tolist() == [False, False, False, False, False, True, False]
    assert result.buy_signal.tolist() == [False, False, True, False, False, False, True]
    changed = calculate("rsi_trading_signals", frame, rsi_length=3)
    assert changed.rsi.iloc[2] != changed.rsi.iloc[2]  # warm-up
    assert changed.rsi.iloc[3] == pytest.approx(100 / 3)
    assert list(frame.columns) == ["close"]


def test_named_indicator_rejects_invalid_inputs():
    frame = pd.DataFrame({"close": [1., 2.]})
    with pytest.raises(ValueError, match="未知"):
        get_indicator("__import__")
    with pytest.raises(ValueError, match="rsi_length"):
        calculate("rsi_trading_signals", frame, rsi_length=0)
    with pytest.raises(ValueError, match="rsi_overbought"):
        calculate("rsi_trading_signals", frame, rsi_overbought=101)
    with pytest.raises(ValueError, match="close"):
        calculate("rsi_trading_signals", pd.DataFrame())
    with pytest.raises(TypeError):
        calculate("rsi_trading_signals", frame, unknown=1)


@pytest.mark.parametrize("name", ["rsi_trading_signals", "volatility_adjusted_momentum", "coinbase_premium", "btc_arbitrage", "btc_scalping_signals", "storm_gauge", "h1_candles_on_m15",
                                  "d1_bollinger_on_m15", "h1_bollinger_on_m15", "bitcoin_market_regime", "capital41_duo",
                                  "capital41_structure_swings", "capital41_atr_volatility_bands", "capital41_oscillator",
                                  "capital41_liquidity_sweep", "volume_flow_anomaly"])
def test_catalog_binds_only_matching_source_version(tmp_path, name):
    import json
    from pathlib import Path

    spec = get_indicator(name)
    root = tmp_path / spec.collection
    path = root / spec.path
    path.parent.mkdir(parents=True)
    source = (Path(__file__).resolve().parents[2] / "data" / spec.collection / spec.path)
    if not source.is_file():
        pytest.skip("本地第三方指标资料未安装")
    path.write_bytes(source.read_bytes())
    (root / "index.json").write_text(json.dumps([{"local_path": spec.path, "kind": "indicator"}]), encoding="utf-8")
    assert _catalog(tmp_path)[0]["python_method"] == spec.name
    path.write_bytes(path.read_bytes() + b"\n// changed")
    assert _catalog(tmp_path)[0]["python_method"] is None


def test_vamo_calculation_and_runtime_parameters():
    frame = pd.DataFrame({"high": [11., 13., 15., 17.], "low": [9., 11., 13., 15.],
                          "close": [10., 12., 14., 16.]})
    result = calculate("volatility_adjusted_momentum", frame, mom_length=1,
                       vol_length=2, signal_length=2, ob_level=800, os_level=700)
    assert result.volatility.iloc[1:].tolist() == pytest.approx([2.5, 2.75, 2.875])
    assert result.momentum.iloc[1:].tolist() == pytest.approx([20, 100 / 6, 100 / 7])
    assert result.vamo.iloc[1:].tolist() == pytest.approx([800, 10000 / 6 / 2.75, 10000 / 7 / 2.875])
    assert result.signal.iloc[2] == pytest.approx(result.vamo.iloc[1] / 3 + result.vamo.iloc[2] * 2 / 3)
    assert result.oversold.tolist() == [False, False, True, True]
    assert pd.isna(calculate("volatility_adjusted_momentum", frame, mom_length=2).momentum.iloc[1])
    with pytest.raises(ValueError, match="vol_length"):
        calculate("volatility_adjusted_momentum", frame, vol_length=0)
    with pytest.raises(ValueError, match="行情列"):
        calculate("volatility_adjusted_momentum", frame.drop(columns="high"))
    flat = pd.DataFrame({"high": [1., 1., 1.], "low": [1., 1., 1.], "close": [1., 1., 1.]})
    assert calculate("volatility_adjusted_momentum", flat, mom_length=1, vol_length=1).vamo.isna().all()


def test_coinbase_premium_uses_supplied_aligned_prices():
    frame = pd.DataFrame({"coinbase_close": [102., 110., 99.], "binance_close": [100., 100., 100.]})
    result = calculate("coinbase_premium", frame, sma_length=2)
    assert result.histogram.tolist() == [2., 10., -1.]
    assert result.sma.iloc[1:].tolist() == [6., 4.5]
    frame.loc[1, "binance_close"] = 200.
    result = calculate("coinbase_premium", frame, show_percent=True, sma_length=1)
    assert result.histogram.tolist() == [2., -45., -1.]
    with pytest.raises(ValueError, match="行情列"):
        calculate("coinbase_premium", frame.drop(columns="coinbase_close"))


def test_btc_arbitrage_external_prices_and_zero_denominators():
    frame = pd.DataFrame({"cme_close": [110., 100.], "binance_close": [100., 0.],
                          "upbit_krw_close": [150000., 150000.], "usdkrw_close": [1000., 0.]})
    result = calculate("btc_arbitrage", frame)
    assert result.loc[0, "cme_premium"] == 10
    assert result.loc[0, "cme_percent"] == 10
    assert result.loc[0, "upbit_premium"] == 50
    assert result.loc[0, "upbit_percent"] == 50
    assert pd.isna(result.loc[1, "cme_percent"])
    assert pd.isna(result.loc[1, "upbit_percent"])
    with pytest.raises(ValueError, match="usdkrw_close"):
        calculate("btc_arbitrage", frame.drop(columns="usdkrw_close"))


def test_btc_scalping_signals_cross_and_filters():
    frame = pd.DataFrame({"open": [10., 9., 8., 11., 8.], "close": [10., 9., 8., 12., 7.]})
    params = dict(ema_fast_len=2, ema_slow_len=3, rsi_len=2,
                  macd_fast=2, macd_slow=3, macd_sig=2)
    result = calculate("btc_scalping_signals", frame, **params)
    assert result.long_signal.tolist() == [False, False, False, True, False]
    assert result.short_signal.tolist() == [False, False, False, False, True]
    assert not calculate("btc_scalping_signals", frame, rsi_buy=90, **params).long_signal.any()
    assert not calculate("btc_scalping_signals", frame, rsi_sell=20, **params).short_signal.any()
    with pytest.raises(ValueError, match="macd_fast"):
        calculate("btc_scalping_signals", frame, macd_fast=0)


def test_storm_gauge_percentile_and_position_size():
    import numpy as np

    returns = np.array([0.] + [(.01 if i % 2 else -.01) * (1 + i / 50) for i in range(1, 81)])
    close = 100 * np.exp(np.cumsum(returns))
    frame = pd.DataFrame({"close": close})
    result = calculate("storm_gauge", frame, vol_len=5, pctile_len=60,
                       periods_year=100, target_vol=10., base_size=2000.)
    expected_vol = np.std(returns[1:6], ddof=0) * 10 * 100
    assert result.annual_vol.iloc[5] == pytest.approx(expected_vol)
    assert result.vol_percentile.iloc[5] == 0
    last = result.annual_vol.iloc[-1]
    expected_rank = sum(last > x for x in result.annual_vol.iloc[-61:-1]) * 100 / 60
    assert result.vol_percentile.iloc[-1] == pytest.approx(expected_rank)
    assert result.dollar_size.iloc[-1] == pytest.approx(2000 * min(max(10 / last, .25), 2))
    hidden = calculate("storm_gauge", frame, vol_len=5, pctile_len=60, show_bands=False)
    assert hidden.upper_band.isna().all()
    with pytest.raises(ValueError, match="pctile_len"):
        calculate("storm_gauge", frame, pctile_len=1)


def test_h1_candles_previous_hour_and_missing_data():
    idx = pd.date_range("2026-01-01", periods=12, freq="15min", tz="UTC")
    frame = pd.DataFrame({"open": range(12), "high": range(1, 13),
                          "low": range(12), "close": range(1, 13)}, index=idx)
    result = calculate("h1_candles_on_m15", frame)
    assert result.h1_close.iloc[:4].isna().all()
    assert result.h1_open.iloc[4:8].tolist() == [0] * 4
    assert result.h1_high.iloc[4:8].tolist() == [4] * 4
    assert result.h1_close.iloc[4:8].tolist() == [4] * 4
    changed = frame.copy()
    changed.iloc[6, changed.columns.get_loc("high")] = 999
    assert calculate("h1_candles_on_m15", changed).h1_high.iloc[4:8].tolist() == [4] * 4
    incomplete = calculate("h1_candles_on_m15", frame.drop(idx[1]))
    assert incomplete.loc[idx[4:8], "h1_close"].isna().all()
    with pytest.raises(ValueError, match="时区"):
        calculate("h1_candles_on_m15", frame.set_axis(idx.tz_localize(None)))


def test_shared_pine_primitives_seed_like_tradingview():
    from app.catalog_indicators._pine import ema, percentrank, pivotlow, pivothigh, rma

    assert ema(pd.Series([1., 2., 3., 4.]), 3).iloc[2:].tolist() == pytest.approx([2., 3.])
    seeded = ema(pd.Series([1., 2., 3., 4.]), 3)
    assert pd.isna(seeded.iloc[1]) and seeded.iloc[2] == 2 and seeded.iloc[3] == 3
    assert rma(pd.Series([1., 2., 3., 4.]), 2).tolist()[1:] == pytest.approx([1.5, 2.25, 3.125])
    assert percentrank(pd.Series([1., 2., 3.]), 3).tolist()[2] == 100
    # A pivot only surfaces `right` bars after the pivot bar.
    highs = pivothigh(pd.Series([1., 2., 3., 2., 1.]), 1, 1)
    assert highs.iloc[3] == 3 and highs.isna().sum() == 4
    lows = pivotlow(pd.Series([3., 2., 1., 2., 3.]), 1, 1)
    assert lows.iloc[3] == 1 and lows.isna().sum() == 4


def _m15_frame(closes, tz="UTC"):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="15min", tz=tz)
    values = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": values, "high": values, "low": values, "close": values}, index=idx)


def test_h1_bollinger_publishes_the_previous_completed_hour():
    hourly = [100, 101, 102, 103]
    frame = _m15_frame([value for value in hourly for _ in range(4)])
    result = calculate("h1_bollinger_on_m15", frame, bb_length=2, bb_std=2.0)
    assert result.iloc[:8].isna().all().all()
    assert result.bb_middle.iloc[8:12].tolist() == pytest.approx([100.5] * 4)
    assert result.bb_upper.iloc[8] == pytest.approx(101.5)
    assert result.bb_lower.iloc[8] == pytest.approx(99.5)
    missing = calculate("h1_bollinger_on_m15", frame.drop(frame.index[5]), bb_length=2)
    assert missing.loc["2026-01-01 02:00":"2026-01-01 02:45", "bb_middle"].isna().all()  # hour 1 lost a bar
    assert missing.loc["2026-01-01 03:00", "bb_middle"] == 101.5  # hour 2 counts again
    with pytest.raises(ValueError, match="15 分钟"):
        calculate("h1_bollinger_on_m15", frame.set_axis(pd.date_range("2026-01-01", periods=len(frame), freq="20min", tz="UTC")))


def test_d1_bollinger_needs_full_days():
    days = [100, 102, 104]
    frame = _m15_frame([value for value in days for _ in range(96)])
    result = calculate("d1_bollinger_on_m15", frame, bb_length=2)
    assert result.bb_middle.iloc[:192].isna().all()
    assert result.bb_middle.iloc[192:].tolist() == pytest.approx([101] * 96)
    partial = calculate("d1_bollinger_on_m15", frame.drop(frame.index[10]), bb_length=2)
    assert partial.bb_middle.loc[:"2026-01-02 23:45"].isna().all()  # day 0 incomplete, so day 1 has no bands
    assert partial.bb_middle.loc["2026-01-03":].tolist() == pytest.approx([101] * 96)  # day 2 still sees day 1


def test_bitcoin_market_regime_labels_trending_series():
    import numpy as np

    rising = pd.DataFrame({
        "high": [100 + i + 0.5 for i in range(60)],
        "low": [100 + i - 0.5 for i in range(60)],
        "close": [100. + i for i in range(60)],
    })
    result = calculate("bitcoin_market_regime", rising, trend_len=10, slope_len=5,
                       vol_rank_len=20, adx_len=5)
    assert result.regime.iloc[-1] == "Bull"
    assert result.is_bull.iloc[-1] and not result.is_bear.iloc[-1]
    assert result.ema_slope.iloc[-1] > 0
    assert result.regime_changed.iloc[0] == False  # noqa: E712 - first bar has no previous regime
    falling = pd.DataFrame({
        # A decelerating decline keeps ATR% out of its top band, so the trend decides.
        "close": 200 - 30 * np.log1p(np.arange(60)),
    })
    falling["high"] = falling["close"] * 1.005
    falling["low"] = falling["close"] * 0.995
    assert calculate("bitcoin_market_regime", falling, trend_len=10, slope_len=5,
                     vol_rank_len=20, adx_len=5).regime.iloc[-1] == "Bear"
    with pytest.raises(ValueError, match="adx_len"):
        calculate("bitcoin_market_regime", rising, adx_len=1)


def test_capital41_duo_turn_confirmation():
    closes = [10., 9., 8., 9., 10.]
    frame = pd.DataFrame({"high": [value + 5 for value in closes],
                          "low": [value - 0.1 for value in closes],
                          "close": closes})
    frame["open"] = frame["close"]
    zone = calculate("capital41_duo", frame, rsi_len=2, wr_len=2, require_turn=False)
    assert zone.rsi.iloc[2:].tolist() == pytest.approx([0, 50, 75])
    assert zone.williams_r.iloc[2] == pytest.approx(-98.36066, rel=1e-4)
    assert zone.both_oversold.tolist() == [False, False, True, False, False]
    turned = calculate("capital41_duo", frame, rsi_len=2, wr_len=2)
    assert turned.both_oversold.tolist() == [False, False, False, True, False]
    with pytest.raises(ValueError, match="wr_oversold"):
        calculate("capital41_duo", frame, wr_oversold=-10.)


def test_capital41_structure_swings_pivot_and_break():
    frame = pd.DataFrame({
        "high": [10., 11, 12, 13, 14, 13, 12, 11, 10, 11, 12, 13, 14],
        "low": [9., 10, 11, 12, 13, 12, 11, 10, 9, 10, 11, 12, 13],
        "close": [10., 11, 12, 13, 13.5, 13, 12, 11, 10, 11, 12, 13, 14.5],
    })
    result = calculate("capital41_structure_swings", frame, pivot_len=2)
    assert result.swing_high.dropna().tolist() == [14.]
    assert result.swing_high.first_valid_index() == frame.index[6]
    assert result.last_swing_high.iloc[:6].isna().all()
    assert result.last_swing_high.iloc[6:].tolist() == [14.] * 7
    assert result.swing_low.first_valid_index() == frame.index[10]
    assert result.swing_low.dropna().tolist() == [9.]
    assert result.bos_up.tolist() == [False] * 12 + [True]
    assert not result.bos_down.any()
    with pytest.raises(ValueError, match="pivot_len"):
        calculate("capital41_structure_swings", frame, pivot_len=30)


def test_capital41_atr_volatility_bands_values():
    closes = [10., 11., 10., 11., 10., 15., 16., 17.]
    frame = pd.DataFrame({"high": [value + 1 for value in closes],
                          "low": [value - 1 for value in closes], "close": closes})
    result = calculate("capital41_atr_volatility_bands", frame, basis_len=5, atr_len=2,
                       atr_mult=1.5, box_lookback=5, exp_factor=1.2)
    assert result.atr.iloc[1] == 2 and result.atr.iloc[5] == 4 and result.atr.iloc[7] == 2.5
    assert result.basis.iloc[4] == pytest.approx(10.4)  # SMA seed of the 5-period EMA
    assert result.basis.iloc[7] == pytest.approx(14.525926, rel=1e-6)
    assert result.upper_band.iloc[7] == pytest.approx(18.275926, rel=1e-6)
    assert result.lower_band.iloc[7] == pytest.approx(10.775926, rel=1e-6)
    assert result.range_high.iloc[7] == 18 and result.range_low.iloc[7] == 9
    assert result.volatility_expansion.tolist() == [False] * 5 + [True, False, False]
    assert result.expansion_start.tolist() == [False] * 5 + [True, False, False]
    with pytest.raises(ValueError, match="basis_type"):
        calculate("capital41_atr_volatility_bands", frame, basis_type="WMA")


def test_capital41_oscillator_three_votes():
    closes = [100. + i for i in range(15)]
    frame = pd.DataFrame({"high": [value + 0.5 for value in closes],
                          "low": [value - 0.5 for value in closes],
                          "close": closes})
    result = calculate("capital41_oscillator", frame, ma_len=5, rsi_len=2, wr_len=2, osc_smooth=3)
    assert result.trend_long.iloc[-1] and not result.trend_short.iloc[-1]
    assert result.rsi.iloc[-1] == 100  # every change is a gain
    assert result.score_long.iloc[-1] == 1  # trend only: RSI/W%R are overbought, not oversold
    assert result.oscillator.iloc[-1] == pytest.approx((1 + 3) * 100 / 6)
    assert result.oscillator_smoothed.iloc[-1] == pytest.approx((1 + 3) * 100 / 6)
    assert not result.long_setup.any()
    with pytest.raises(ValueError, match="osc_smooth"):
        calculate("capital41_oscillator", frame, osc_smooth=0)


def test_capital41_liquidity_sweep_counts_consecutive_sweeps():
    frame = pd.DataFrame({
        "high": [101.] * 8,
        "low": [99.9, 99.9, 99.9, 99.9, 98.8, 99.6, 99.6, 98.7],
        "close": [100.] * 8,
        "volume": [10., 10., 10., 10., 30., 10., 10., 30.],
    })
    result = calculate("capital41_liquidity_sweep", frame, lookback=2, volume_sma_len=5,
                       min_volume_ratio=1.2)
    assert result.bull_sweep.tolist() == [False, False, False, False, True, False, False, True]
    assert result.consecutive_bull.tolist() == [0, 0, 0, 0, 1, 1, 1, 2]
    assert result.high_prob_bull.tolist() == [False] * 7 + [True]
    assert not result.bear_sweep.any()
    quiet = calculate("capital41_liquidity_sweep", frame.assign(volume=10.),
                      lookback=2, volume_sma_len=5)
    assert not quiet.bull_sweep.any()  # volume filter blocks the wick
    with pytest.raises(ValueError, match="volume"):
        calculate("capital41_liquidity_sweep", frame.drop(columns="volume"))


def test_volume_flow_anomaly_matches_its_own_formulas():
    import numpy as np

    rng = np.random.default_rng(11)
    count = 60
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, count)))
    high = close * (1 + np.abs(rng.normal(0, 0.002, count)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, count)))
    open_ = close * (1 + rng.normal(0, 0.001, count))
    volume = np.abs(rng.normal(50., 15., count))
    frame = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})

    result = calculate("volume_flow_anomaly", frame, window_len=3, lookback=30,
                       vol_z_min=1.0, vol_z_early=0.5, clean_min=40., move_min=0.2)
    buy_fraction = (close - low) / (high - low)
    buy_volume = volume * buy_fraction
    window_volume = pd.Series(volume).rolling(3).sum()
    expected_z = ((window_volume - window_volume.rolling(30).mean())
                  / window_volume.rolling(30).std(ddof=0))
    assert result.window_volume.iloc[29] == pytest.approx(window_volume.iloc[29])
    assert result.vol_z.iloc[:31].isna().all()  # window (3) plus lookback (30) of warm-up
    assert result.vol_z.iloc[31:].tolist() == pytest.approx(expected_z.iloc[31:].tolist(), rel=1e-9)
    assert result.buy_fraction.iloc[5] == pytest.approx(buy_fraction[5])
    assert result.window_buy_volume.iloc[5] == pytest.approx(pd.Series(buy_volume).iloc[3:6].sum())
    assert result.trend_cleanliness.iloc[5] == pytest.approx(
        abs(pd.Series(close - open_).iloc[3:6].sum()) / pd.Series(close - open_).abs().iloc[3:6].sum() * 100)
    assert result.window_move.iloc[5] == pytest.approx((close[5] - open_[3]) / open_[3] * 100)
    assert result.vol_z_change.iloc[40] == pytest.approx(result.vol_z.iloc[40] - result.vol_z.iloc[38])
    assert result.extreme_volume.equals(result.vol_z >= 3.5)
    flat = frame.assign(high=frame["close"], low=frame["close"])
    assert calculate("volume_flow_anomaly", flat, lookback=30).buy_fraction.eq(0.5).all()
    with pytest.raises(ValueError, match="window_len"):
        calculate("volume_flow_anomaly", frame, window_len=1)
    with pytest.raises(ValueError, match="行情列"):
        calculate("volume_flow_anomaly", frame.drop(columns="volume"))
