"""Supertrend CURRENT: Pine signal semantics and local strategy integration."""

import numpy as np
import pandas as pd
import pytest

from app.features.indicators import build_features
from app.models import StrategyConfig
from app.strategy.defaults import seed_strategies
from app.strategy.rules import evaluate_signals, known_codes, prepare_frame, resolve_params
from app.strategy.specs.supertrend_current import _setup_signals, signals as supertrend_signals

CODE = "supertrend_current"


def _frame():
    # Long trends separated by sharp turns create unambiguous crossings while
    # providing ample warm-up for the 200-bar EMA and shared indicators.
    close = np.concatenate([
        np.full(220, 100.0),
        np.linspace(100, 120, 40),
        np.linspace(120, 90, 60),
        np.linspace(90, 125, 70),
        np.linspace(125, 92, 65),
    ])
    index = pd.date_range("2026-01-01", periods=len(close), freq="15min", tz="UTC")
    return pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.full(len(close), 1000.0),
    }, index=index)


def _params(**overrides):
    return resolve_params(CODE, {}, {"cost_filter": False, "max_atr_pct": 0.2, **overrides})


def test_staged_exit_defaults_and_custom_values():
    params = _params()
    assert (params["tp1_r"], params["tp2_r"], params["target_r"]) == (1, 2, 3)
    assert [params[k] for k in ("tp1_fraction", "tp2_fraction", "tp3_fraction")] == pytest.approx([1 / 6, 1 / 3, 1 / 2])
    custom = _params(tp1_r=0.5, tp2_r=1.5, target_r=4,
                     tp1_fraction=0.2, tp2_fraction=0.3, tp3_fraction=0.5)
    assert (custom["tp1_r"], custom["tp2_r"], custom["target_r"]) == (0.5, 1.5, 4)
    assert sum(custom[k] for k in ("tp1_fraction", "tp2_fraction", "tp3_fraction")) == pytest.approx(1)


@pytest.mark.parametrize("overrides", [
    {"tp1_r": 2}, {"tp2_r": 3}, {"target_r": 2},
    {"tp1_fraction": -0.1}, {"tp1_fraction": 0.5},
])
def test_reject_invalid_staged_exit_configuration(overrides):
    with pytest.raises(ValueError):
        _params(**overrides)


def test_strategy_is_registered_and_seeded_disabled(db):
    assert CODE in known_codes()
    seed_strategies(db)
    row = db.query(StrategyConfig).filter_by(code=CODE).one()
    assert row.enabled == 0
    seed_strategies(db)
    assert db.query(StrategyConfig).filter_by(code=CODE).count() == 1


def test_supertrend_reversals_produce_both_signal_directions():
    data = prepare_frame(_frame(), CODE, _params())
    cross_up = data.st_direction.eq(1) & data.st_direction.shift(1).eq(-1)
    cross_down = data.st_direction.eq(-1) & data.st_direction.shift(1).eq(1)
    assert cross_up.any() and cross_down.any()
    assert data.loc[cross_up, "st_signal"].eq(1).all()
    assert data.loc[cross_down, "st_signal"].eq(-1).all()
    assert data.loc[~(cross_up | cross_down), "st_signal"].eq(0).all()


def test_wilder_atr_and_supertrend_known_turns():
    # With a 3-bar ATR, the first six bars have TR=1, then TR=3.5,
    # 1.5 and 5.5. The two jumps cross the previous final bands.
    close = np.array([100., 100., 100., 100., 100., 100., 103., 104., 99., 98.])
    data = pd.DataFrame({
        "open": close, "high": close + .5, "low": close - .5,
        "close": close, "volume": 1.,
    }, index=pd.date_range("2026-01-01", periods=len(close), freq="15min", tz="UTC"))
    out = prepare_frame(data, CODE, _params(atr_period=3, supertrend_factor=1.0))
    assert out.st_atr.iloc[5] == pytest.approx(1.0)
    assert out.st_atr.iloc[6] == pytest.approx(11 / 6)
    assert out.st_atr.iloc[7] == pytest.approx(31 / 18)
    assert out.st_atr.iloc[8] == pytest.approx(161 / 54)
    assert out.st_direction.iloc[5] == -1
    assert out.st_signal.iloc[6] == 1
    assert out.st_signal.iloc[8] == -1


def test_filter_uses_strict_adx_and_directional_ema():
    raw = _frame()
    params = _params(use_adx_filter=True, adx_threshold=50.0, use_ema200_filter=True)
    out = prepare_frame(raw, CODE, params)
    raw_signal = out.st_raw_signal
    eligible = (out.st_adx > 50.0) & (
        ((raw_signal == 1) & (out.close > out.st_ema200))
        | ((raw_signal == -1) & (out.close < out.st_ema200))
    )
    expected = raw_signal.where(eligible, 0)
    assert raw_signal.ne(0).any()
    pd.testing.assert_series_equal(out.st_signal, expected, check_names=False, check_dtype=False)


def _setup_frame(close, high, low, direction, line=100.0):
    n = len(close)
    return pd.DataFrame({
        "close": close, "high": high, "low": low, "st_direction": direction,
        "st_line": [line] * n, "st_atr": [1.0] * n,
        "st_adx": [60.0] * n, "st_ema200": [90.0] * n,
    })


def test_adx_threshold_and_ema_equalities_are_rejected():
    frame = _setup_frame([100.] * 4, [100.5] * 4, [99.5] * 4, [-1, 1, -1, 1])
    frame["st_adx"] = [60., 50., 51., 51.]
    frame["st_ema200"] = [90., 90., 100., 99.]
    raw, filtered = _setup_signals(frame, _params(use_adx_filter=True, use_ema200_filter=True))
    assert raw.tolist() == [0, 1, -1, 1]
    assert filtered.tolist() == [0, 0, 0, 1]


def test_reentry_requires_tp3_before_supertrend_touch():
    frame = _setup_frame(
        [100., 100., 104., 103.], [100.5, 100.5, 105., 104.],
        [99.5, 99.5, 103., 99.], [-1, 1, 1, 1],
    )
    _, disabled = _setup_signals(frame, _params())
    _, enabled = _setup_signals(frame, _params(enable_tp3_reentry=True))
    assert disabled.tolist() == [0, 1, 0, 0]
    assert enabled.tolist() == [0, 1, 0, 1]
    before_tp3 = frame.copy()
    before_tp3.loc[2, "high"] = 104.4  # entry 100 + 3 × 1.5 ATR = 104.5
    _, result = _setup_signals(before_tp3, _params(enable_tp3_reentry=True))
    assert result.tolist() == [0, 1, 0, 0]
    frame["st_raw_signal"], frame["st_signal"] = _setup_signals(
        frame, _params(enable_tp3_reentry=True)
    )
    frame["atr_pct"] = 1 / frame.close
    # Re-entry is a stateful signal, but the engine passes only the prepared
    # last row into evaluate_signals.
    decision = evaluate_signals(frame.iloc[-1:], CODE, _params(enable_tp3_reentry=True)).iloc[0]
    assert decision.direction == 1


def test_breakout_requires_confirmed_pivots_and_is_optional():
    frame = _setup_frame(
        [100., 102., 100., 103., 101., 104., 102., 107.],
        [101., 103., 101., 104., 102., 105., 103., 108.],
        [99., 100., 99., 101., 100., 102., 101., 103.],
        [1] * 8,
    )
    _, disabled = _setup_signals(frame, _params(pivot_length=1))
    _, enabled = _setup_signals(frame, _params(pivot_length=1, enable_breakout=True))
    assert disabled.tolist() == [0] * 8
    assert enabled.tolist() == [0, 0, 0, 0, 0, 0, 0, 1]
    # Removing the breakout bar must not change any earlier signal.
    _, prefix = _setup_signals(frame.iloc[:-1], _params(pivot_length=1, enable_breakout=True))
    np.testing.assert_array_equal(prefix, enabled[:-1])


def test_atr_stop_and_three_r_target_use_signal_close():
    raw = _frame()
    params = _params()
    enriched = prepare_frame(build_features(raw, with_label=False), CODE, params)
    signals = evaluate_signals(enriched, CODE, params)
    hit = signals.direction.ne(0)
    assert hit.any()
    expected_stop = 1.5 * enriched.loc[hit, "st_atr"] / enriched.loc[hit, "close"]
    np.testing.assert_allclose(signals.loc[hit, "sl_pct"], expected_stop, rtol=1e-8)
    np.testing.assert_allclose(signals.loc[hit, "tp_pct"], 3 * expected_stop, rtol=1e-8)


def test_supertrend_preserves_small_atr_distances_and_blocks_invalid_values():
    frame = pd.DataFrame({"close": [100.0] * 4, "st_atr": [0.01, -1.0, np.nan, np.inf],
                          "st_raw_signal": [1] * 4, "atr_pct": [0.0001, -0.01, np.nan, np.inf]})
    params = _params(min_sl_pct=0.01, min_tp_pct=0.04)
    result = supertrend_signals(frame, params)
    assert result.loc[0, "direction"] == 1
    assert result.loc[0, "sl_pct"] == pytest.approx(1.5 * 0.01 / 100)
    assert result.loc[0, "tp_pct"] == pytest.approx(3 * 1.5 * 0.01 / 100)
    assert result.loc[1:, "direction"].eq(0).all()
    assert result.loc[1:, "block_reason"].eq("invalid_indicators").all()


def test_prepared_single_row_decisions_match_full_frame():
    # StrategyEngine evaluates only the latest row after preparing the full
    # frame, so this must retain the computed crossing and stateful signals.
    params = _params()
    enriched = prepare_frame(build_features(_frame(), with_label=False), CODE, params)
    full = evaluate_signals(enriched, CODE, params)
    indices = full.index[full.direction.ne(0)]
    assert len(indices) >= 2
    for index in indices:
        single = evaluate_signals(enriched.loc[[index]], CODE, params).iloc[0]
        assert single.direction == full.loc[index, "direction"]
        assert single.sl_pct == pytest.approx(full.loc[index, "sl_pct"])
        assert single.tp_pct == pytest.approx(full.loc[index, "tp_pct"])


@pytest.mark.parametrize("end", [260, 350, 430])
def test_features_and_signals_do_not_depend_on_future_bars(end):
    raw = _frame()
    params = _params()
    full = prepare_frame(raw, CODE, params)
    truncated = prepare_frame(raw.iloc[:end], CODE, params)
    for name in ("st_atr", "st_line", "st_direction", "st_adx", "st_ema200", "st_signal"):
        pd.testing.assert_series_equal(
            full[name].iloc[:end], truncated[name], check_names=False, check_dtype=False,
        )


def test_optional_modes_are_causal_too():
    raw = _frame()
    params = _params(enable_breakout=True, enable_tp3_reentry=True,
                     enable_ema_reentry=True, enable_aggressive_ema_reentry=True)
    full = prepare_frame(raw, CODE, params)
    for end in (260, 350, 430):
        truncated = prepare_frame(raw.iloc[:end], CODE, params)
        pd.testing.assert_series_equal(full.st_signal.iloc[:end], truncated.st_signal)
