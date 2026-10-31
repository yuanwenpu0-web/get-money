import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.compression_fractal_release import compression_fractal_release


def _frame(closes, index=None):
    close = pd.Series(closes, dtype=float, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.05,
                         "low": np.minimum(open_, close) - 0.05, "close": close},
                        index=index)


def test_break_band_excludes_the_current_bar_and_flags_first_breaks():
    # Flat 20 bars at 100 then a close at 102 clears the previous 4-bar band.
    closes = [100.0] * 20 + [102.0, 102.5, 101.0, 103.0]
    frame = _frame(closes, pd.date_range("2026-01-01", periods=len(closes),
                                         freq="15min", tz="UTC"))
    result = compression_fractal_release(frame, length=10, break_len=4, chop_len=5,
                                         htf_minutes=15, ema_len=10, slope_len=2,
                                         htf_slope_min=0.0)
    # Bar 20: corr_high = highest(high, 4)[1] = 100.05 -> close 102 breaks it.
    assert result.corr_high.iloc[20] == pytest.approx(100.05)
    assert result.break_up.iloc[20] and result.first_up.iloc[20]
    # Bar 21 breaks again (102.5 > band) so it is not a first break.
    assert result.break_up.iloc[21] and not result.first_up.iloc[21]
    # Bar 22 closes back inside -> no break.
    assert not result.break_up.iloc[22]
    # break_impulse on bar 20 = |close - band| / atr(14).
    atr14 = result.break_impulse.iloc[20] * frame["close"].rolling(14).mean()  # not exact
    with pytest.raises(ValueError, match="length"):
        compression_fractal_release(frame, length=5)
    with pytest.raises(ValueError, match="基础周期"):
        compression_fractal_release(frame, htf_minutes=7)


def test_htf_slope_uses_the_previous_completed_period():
    index = pd.date_range("2026-01-01", periods=96 * 3, freq="15min", tz="UTC")
    closes = np.linspace(100, 160, 96 * 3)  # steady 4h uptrend
    frame = _frame(closes.tolist(), index)
    result = compression_fractal_release(frame, length=10, break_len=4, chop_len=5,
                                         htf_minutes=60, ema_len=10, slope_len=3,
                                         htf_slope_min=0.0)
    # The hourly EMA(10) of the previous completed hour, lifted bar by bar.
    from app.catalog_indicators._pine import ema as pine_ema

    htf_close = frame.groupby(frame.index.floor("h"))["close"].last()
    # Every complete hour has exactly 4 bars here, so all are kept.
    htf_ema = pine_ema(htf_close, 10)
    lifted = htf_ema.shift(1).reindex(frame.index.floor("h"))
    lifted.index = frame.index
    compare = result.htf_ema.dropna()
    assert np.allclose(compare.to_numpy(), lifted.loc[compare.index].to_numpy())
    # The very first hour cannot have a previous completed hour.
    assert result.htf_ema.iloc[:4].isna().all()
    # Direction is 0 while the HTF EMA is still warming up, then always up.
    assert (result.htf_dir.iloc[100:] == 1).all()


def test_cooldown_suppresses_the_second_signal_in_the_same_direction():
    # A staircase: 4 flat bars per level, 20 levels. Each level change is a
    # first_up; the flat stretches produce no breaks in between.
    closes = [float(level) for level in range(100, 120) for _ in range(4)]
    index = pd.date_range("2026-01-01", periods=len(closes), freq="15min", tz="UTC")
    frame = _frame(closes, index)
    common = dict(length=10, break_len=4, chop_len=5, htf_minutes=15, ema_len=10,
                  slope_len=2, htf_slope_min=0.0, thr_cont=0.0, break_atr_min=0.0,
                  chop_veto_max=79.9)
    strict = compression_fractal_release(frame, cooldown_bars=0, **common)
    chilled = compression_fractal_release(frame, cooldown_bars=100, **common)
    assert strict.release_long.sum() >= 5
    # Cooldown 100 spans the whole 80-bar series, so only the first long fires.
    assert chilled.release_long.sum() == 1
    # Signals only ever fire on first breaks, and never downward.
    assert not (chilled.release_long & ~chilled.first_up).any()
    assert not chilled.release_short.any()
    with pytest.raises(ValueError, match="cooldown_bars"):
        compression_fractal_release(frame, cooldown_bars=101, **common)


def test_coil_metrics_match_hand_computed_values():
    closes = [100.0] * 12 + [100.2, 100.4, 100.6, 100.8, 101.0]
    frame = _frame(closes, pd.date_range("2026-01-01", periods=len(closes),
                                         freq="15min", tz="UTC"))
    result = compression_fractal_release(frame, length=12, break_len=4, chop_len=5,
                                         htf_minutes=15, ema_len=10, slope_len=2,
                                         htf_slope_min=0.0, symbol_atr_mult=0.9)
    # With a huge ATR step every |diff| below the unit maps to symbols +-1,
    # except the flat stretch which is exactly 0.
    window = result.complexity.dropna()
    assert window.iloc[-1] > 0.0
    # Efficiency: direct displacement / path length * 100.
    eff_last = (101.0 - closes[len(closes) - 12]) / (
        sum(abs(closes[i] - closes[i - 1]) for i in range(len(closes) - 12 + 1,
                                                         len(closes)))) * 100.0
    assert result.efficiency.iloc[-1] == pytest.approx(eff_last)
    # Fractal dimension stays within its clamp.
    assert result.fractal_dimension.dropna().between(1.0, 2.0).all()
    assert result.setup_score.dropna().between(0.0, 1.0).all()
    with pytest.raises(ValueError, match="coil_now_min"):
        compression_fractal_release(frame, coil_now_min=1.5)
