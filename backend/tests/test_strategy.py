import numpy as np
import pandas as pd
import pytest
from app.strategy.rules import evaluate_signals, resolve_params


def market_row(**changes):
    row = dict(
        close=100.0,
        atr_pct=0.0015,
        rsi_14=55.0,
        supertrend_dir=1.0,
        ema_ratio_9_21=0.002,
        ema_ratio_21_50=0.003,
        macd_hist_norm=0.001,
        adx_14=30.0,
        bb_pos=0.6,
    )
    row.update(changes)
    return pd.DataFrame([row], index=pd.date_range("2026-01-01", periods=1, tz="UTC"))


def test_trend_rejects_chop_and_conflicting_slow_trend():
    params = resolve_params("trend_follow", {})
    assert evaluate_signals(market_row(), "trend_follow", params).iloc[0].direction == 1
    for changes in [dict(adx_14=10), dict(ema_ratio_21_50=-0.001)]:
        signal = evaluate_signals(market_row(**changes), "trend_follow", params).iloc[0]
        assert signal.direction == 0
        assert signal.block_reason


def test_mean_reversion_rejects_falling_trend():
    params = resolve_params("mean_reversion", {})
    frame = market_row(rsi_14=15, bb_pos=-0.1, adx_14=40)
    assert evaluate_signals(frame, "mean_reversion", params).iloc[0].direction == 0
    frame["adx_14"] = 16
    assert evaluate_signals(frame, "mean_reversion", params).iloc[0].direction == 1


def test_cost_filter_rejects_low_volatility_even_with_full_score():
    signal = evaluate_signals(
        market_row(atr_pct=0.0001), "trend_follow", resolve_params("trend_follow", {})
    ).iloc[0]
    assert signal.score == 1
    assert signal.direction == 0
    assert signal.block_reason == "cost_filter"


@pytest.mark.parametrize(
    "column", ["atr_pct", "adx_14", "close", "supertrend_dir", "rsi_14"]
)
def test_nonfinite_inputs_cannot_open(column):
    signal = evaluate_signals(
        market_row(**{column: np.nan}),
        "trend_follow",
        resolve_params("trend_follow", {}),
    ).iloc[0]
    assert signal.direction == 0
    assert signal.position_pct == 0


def test_risk_cap_and_model_requirement():
    params = resolve_params("trend_follow", {}, {"risk_per_trade_pct": 0.001})
    signal = evaluate_signals(market_row(), "trend_follow", params).iloc[0]
    assert 0 < signal.risk_pct <= 0.001 + 1e-12
    params = resolve_params("ml_ensemble", {}, {"require_model": True})
    assert (
        evaluate_signals(market_row(), "ml_ensemble", params).iloc[0].block_reason
        == "model_required"
    )
    probability = pd.Series([0.49], index=market_row().index)
    assert (
        evaluate_signals(market_row(), "ml_ensemble", params, probability)
        .iloc[0]
        .direction
        == 0
    )


def test_strategy_overrides_and_threshold_boundary():
    params = resolve_params("trend_follow", {"min_score": 0.6}, {"min_score": 1.0})
    signal = evaluate_signals(market_row(), "trend_follow", params).iloc[0]
    assert signal.direction == 1
    assert signal.position_pct > 0
    with pytest.raises(ValueError):
        resolve_params("trend_follow", {}, {"risk_per_trade_pct": -0.1})
    with pytest.raises(ValueError, match="撮合成本"):
        resolve_params("trend_follow", {}, {"fee_rate": 0.0})
