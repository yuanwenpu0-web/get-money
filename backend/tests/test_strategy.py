"""策略层：只保留吸收反转一套规则，这里覆盖它的参数校验与信号门控。"""

import numpy as np
import pandas as pd
import pytest
from app.strategy.defaults import DEFAULT_STRATEGIES
from app.strategy.rules import evaluate_signals, known_codes, resolve_params

CODE = "absorption_reversal"


def signal_row(**changes):
    """一根「已确认吸收」的做多行：结构止损在极值下方、风险宽度落在允许区间内。

    评分靠 |delta| 超出百分位的幅度 + 单边支配度，两个输入都远高于门槛 → 分数 > heavy_score。
    """
    row = dict(
        close=100.0,
        atr_pct=0.0015,          # ATR(价格) = 0.15
        # --- 以下字段凑齐 ABSORPTION_COLUMNS，避免规则层把整帧重算掉 ---
        abs_delta=-1.0,
        abs_abs_delta=100.0,
        abs_dominance=60.0,      # 支配度门槛 20 → 强度 0.5
        abs_percentile=50.0,     # |delta| 是百分位阈值的 2 倍 → overflow 打满
        abs_atr=0.15,
        abs_body_top=100.2,
        abs_body_bottom=99.9,
        abs_wick_ratio=0.3,
        abs_extreme_high=100.6,
        abs_extreme_low=99.5,
        abs_trend_ema=99.0,
        abs_raw=1.0,
        abs_bar=1.0,
        abs_dir=1.0,
        abs_signal=1.0,          # 卖方被吸收 → 做多
        abs_extreme=99.5,
        abs_level=100.2,
        abs_sl=99.6,
        abs_risk=0.4,            # 入场价与止损的绝对距离 → 2.67 ATR（<= max_sl_atr=3）
        funding_zscore=0.0,
    )
    row.update(changes)
    return pd.DataFrame([row], index=pd.date_range("2026-01-01", periods=1, tz="UTC"))


def _out(**changes):
    frame = signal_row(**changes)
    return evaluate_signals(frame, CODE, resolve_params(CODE, {})).iloc[0]


# ---------- 参数层 ----------

def test_builtin_strategies_are_registered():
    codes = {CODE, "supertrend_current", "simple_zigzag_double", "opening_candle_breakout"}
    assert {item["code"] for item in DEFAULT_STRATEGIES} == codes
    assert set(known_codes()) == codes


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="未知策略"):
        resolve_params("trend_follow", {})


def test_unknown_override_and_global_cost_are_rejected():
    with pytest.raises(ValueError, match="未知策略参数"):
        resolve_params(CODE, {}, {"body_mult": 10.0, "adx_filter": 18})
    with pytest.raises(ValueError, match="撮合成本"):
        resolve_params(CODE, {}, {"fee_rate": 0.0})


def test_out_of_range_and_type_errors():
    with pytest.raises(ValueError):
        resolve_params(CODE, {}, {"risk_per_trade_pct": -0.1})
    with pytest.raises(ValueError):
        resolve_params(CODE, {}, {"conf_window_bars": 1.5})  # 必须为整数
    with pytest.raises(ValueError):
        resolve_params(CODE, {}, {"legacy_gap_logic": 1})    # 必须为布尔值


def test_default_params_match_the_script_medium_preset():
    params = resolve_params(CODE, {})
    assert (params["delta_pct"], params["body_mult"], params["extreme_lookback"]) == (
        95.0, 0.25, 20,
    )
    assert (params["target_r"], params["be_trigger_r"], params["max_hold_bars"]) == (
        2.0, 1.0, 30,
    )


# ---------- 信号层 ----------

def test_confirmed_absorption_opens_long():
    signal = _out()
    assert signal.direction == 1
    assert signal.position_pct > 0
    assert signal.sl_pct > 0 and signal.tp_pct > signal.sl_pct


def test_short_side_mirrors_on_the_buyer_absorbed_bar():
    signal = _out(abs_signal=-1.0, abs_delta=1.0, abs_sl=100.4, abs_extreme=100.5)
    assert signal.direction == -1


def test_without_absorption_column_there_is_no_signal():
    assert _out(abs_signal=0.0).block_reason == "no_absorption_signal"


def test_stop_width_gates_use_atr_units():
    assert _out(abs_risk=1.0).block_reason == "stop_too_wide"      # 6.7 ATR > 3
    assert _out(abs_risk=0.02).block_reason == "stop_too_tight"    # 0.13 ATR < 0.3


def test_stop_on_the_wrong_side_is_rejected():
    assert _out(abs_sl=100.4).block_reason == "stop_wrong_side"


def test_direction_mode_can_disable_one_side():
    params = resolve_params(CODE, {}, {"direction_mode": 2})  # 只做空
    assert evaluate_signals(signal_row(), CODE, params).iloc[0].block_reason == "long_disabled"


def test_trend_filter_is_strict_about_price_position():
    params = resolve_params(CODE, {}, {"trend_mode": 1})  # 顺 EMA 趋势：做多需站上 EMA
    assert evaluate_signals(signal_row(), CODE, params).iloc[0].direction == 1
    against = evaluate_signals(
        signal_row(abs_trend_ema=101.0), CODE, params
    ).iloc[0]
    assert against.block_reason == "trend_filter"


def test_session_filter_restricts_entries_to_the_window():
    params = resolve_params(CODE, {}, {"session_start_hour": 9.5, "session_end_hour": 16.0})
    assert evaluate_signals(signal_row(), CODE, params).iloc[0].block_reason == "session_filter"


def test_cost_filter_rejects_low_volatility():
    assert _out(atr_pct=0.00005).block_reason == "cost_filter"


@pytest.mark.parametrize("column", ["atr_pct", "close", "abs_risk"])
def test_nonfinite_inputs_cannot_open(column):
    signal = _out(**{column: np.nan})
    assert signal.direction == 0
    assert signal.position_pct == 0


def test_risk_cap_limits_position_size():
    params = resolve_params(CODE, {}, {"risk_per_trade_pct": 0.001})
    signal = evaluate_signals(signal_row(), CODE, params).iloc[0]
    assert 0 < signal.risk_pct <= 0.001 + 1e-12


def test_threshold_boundary_blocks_weak_signals():
    params = resolve_params(CODE, {}, {"min_score": 1.0})
    assert evaluate_signals(signal_row(), CODE, params).iloc[0].direction == 0
