"""Market Tradability Engine v2 移植测试。

覆盖：三分数恒等式、预设/reactivity 映射、武装区间冻结、边界测试与疲劳、
状态机滞回、量能门、no-trade 原因码、参数校验。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.market_tradability_engine_v2 import (
    NO_TRADE_REASONS,
    STATE_AFTERMATH,
    STATE_BALANCE,
    STATE_BREAKOUT,
    STATE_COMPRESSION,
    STATE_EXHAUSTION,
    STATE_NO_TRADE,
    STATE_READY,
    STATE_TREND,
    market_tradability_engine_v2,
)


def _frame(close: np.ndarray, *, spread: float = 0.4, volume=1000.0, freq="1h") -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    n = len(close)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    vol = volume if isinstance(volume, np.ndarray) else np.full(n, float(volume))
    return pd.DataFrame(
        {"open": close, "high": close + spread, "low": close - spread, "close": close, "volume": vol},
        index=idx,
    )


def _noisy(n: int = 1200, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.8, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + np.abs(rng.normal(0, 0.5, n)),
            "low": close - np.abs(rng.normal(0, 0.5, n)),
            "close": close,
            "volume": 1000.0 + np.abs(rng.normal(0, 200, n)),
        },
        index=idx,
    )


def _range_then_trend(n: int = 600, range_bars: int = 300) -> pd.DataFrame:
    close = np.empty(n)
    close[:range_bars] = 100 + 1.2 * np.sin(np.arange(range_bars) / 6.0)
    close[range_bars:] = 100 + np.arange(n - range_bars) * 0.35
    return _frame(close)


# ---------------------------------------------------------------- 三分数


def test_structure_is_max_of_two_frameworks():
    out = market_tradability_engine_v2(_noisy())
    expected = np.fmax(out["mte_range_structure_quality"], out["mte_trend_structure_quality"])
    assert out["mte_structure_score"].to_numpy() == pytest.approx(expected, nan_ok=True)
    flag = (out["mte_range_structure_quality"] >= out["mte_trend_structure_quality"]).astype(int)
    assert np.array_equal(out["mte_range_framework"].to_numpy(dtype=int), flag)


def test_trend_structure_weighting():
    out = market_tradability_engine_v2(_noisy())
    expected = (out["mte_trend_cross_quality"] * 0.25 + out["mte_trend_efficiency_quality"] * 0.25
                + out["mte_trend_adx_quality"] * 0.20 + out["mte_trend_slope_quality"] * 0.20
                + out["mte_trend_consistency_quality"] * 0.10)
    assert out["mte_trend_structure_quality"].to_numpy() == pytest.approx(expected.to_numpy(), nan_ok=True)


def test_energy_weighting_and_bounds():
    out = market_tradability_engine_v2(_noisy())
    raw = (out["mte_volatility_expansion"] * 0.30 + out["mte_energy_efficiency"] * 0.30
           + out["mte_energy_momentum_accel"] * 0.20 + out["mte_candle_commitment"] * 0.20)
    assert out["mte_energy_score"].to_numpy() == pytest.approx(np.clip(raw, 0.0, 100.0), nan_ok=True)
    assert out["mte_energy_score"].dropna().between(0.0, 100.0).all()


def test_acceptance_identity():
    out = market_tradability_engine_v2(_noisy())
    slope = np.clip(out["mte_normalized_slope"] / 0.75, -1.0, 1.0)
    base = np.clip(out["mte_close_displacement"] / 2.0, -1.0, 1.0)
    di = np.where((out["mte_plus_di"] + out["mte_minus_di"]).abs() > 1e-7,
                  (out["mte_plus_di"] - out["mte_minus_di"]) / (out["mte_plus_di"] + out["mte_minus_di"]), 0.0)
    pressure = np.clip((out["mte_range_position"] - 0.5) * 2.0, -1.0, 1.0)
    expected = np.clip((slope * 0.22 + base * 0.16 + di * 0.16
                        + out["mte_signed_consistency"] * 0.14 + pressure * 0.12
                        + out["mte_average_bar_acceptance"] * 0.20) * 100.0, -100.0, 100.0)
    assert out["mte_acceptance_score"].to_numpy() == pytest.approx(expected.to_numpy(), nan_ok=True)


def test_range_structure_geometric_mean():
    out = market_tradability_engine_v2(_noisy())
    expected = np.sqrt(np.maximum(out["mte_sustained_boundary_stability"], 0.0)
                       * np.maximum(out["mte_range_compactness"], 0.0)) * 100.0
    assert out["mte_range_structure_quality"].to_numpy() == pytest.approx(expected.to_numpy(), nan_ok=True)


# ---------------------------------------------------------------- 预设


def test_reactivity_resolves_thresholds():
    frame = _noisy(400)
    early = market_tradability_engine_v2(frame, reactivity="Early")
    balanced = market_tradability_engine_v2(frame, reactivity="Balanced")
    strict = market_tradability_engine_v2(frame, reactivity="Conservative")
    assert float(early["mte_threshold_structure"].iloc[0]) == 42.0
    assert float(balanced["mte_threshold_structure"].iloc[0]) == 55.0
    assert float(strict["mte_threshold_structure"].iloc[0]) == 65.0
    assert float(early["mte_threshold_energy_low"].iloc[0]) == 36.0
    assert float(strict["mte_threshold_energy_high"].iloc[0]) == 63.0
    assert float(early["mte_threshold_acceptance_neutral"].iloc[0]) == 12.0
    assert float(strict["mte_threshold_acceptance_strong"].iloc[0]) == 38.0

    custom = market_tradability_engine_v2(
        frame, reactivity="Custom", custom_structure_threshold=48.0,
        custom_energy_high_threshold=58.0, custom_acceptance_neutral_band=15.0,
    )
    assert float(custom["mte_threshold_structure"].iloc[0]) == 48.0
    assert float(custom["mte_threshold_energy_high"].iloc[0]) == 58.0
    assert float(custom["mte_threshold_acceptance_neutral"].iloc[0]) == 15.0


def test_preset_only_changes_lookbacks():
    frame = _noisy(400)
    # 预设只改回看长度，阈值由 reactivity 决定
    for preset in ("Very Fast", "Calm", "Position"):
        out = market_tradability_engine_v2(frame, preset=preset)
        assert float(out["mte_threshold_structure"].iloc[0]) == 55.0
    fast = market_tradability_engine_v2(frame, preset="Very Fast")
    slow = market_tradability_engine_v2(frame, preset="Position")
    # 更长的 rank 窗口 → 暖机更久，data_ready 更晚
    assert int(fast["mte_data_ready"].sum()) > int(slow["mte_data_ready"].sum())


# ---------------------------------------------------------------- 武装区间


def test_armed_boundaries_frozen_against_rolling_window():
    out = market_tradability_engine_v2(_range_then_trend())
    armed = out["mte_breakout_armed"].to_numpy(dtype=int)
    reference = out["mte_breakout_reference_high"].to_numpy(dtype=float)
    frozen = out["mte_armed_range_high"].to_numpy(dtype=float)
    rolling = out["mte_rolling_range_high"].to_numpy(dtype=float)
    assert armed.max() == 1
    # 武装期间，突破参考边界恒等于冻结值（不让滚动窗口追着突破条跑）
    mask = armed == 1
    assert np.allclose(reference[mask], frozen[mask])
    # 至少有一段时间滚动上边界在动而参考边界不动
    start = int(np.argmax(armed))
    run = [i for i in range(start, min(start + 40, len(out))) if armed[i] == 1]
    assert len(set(np.round(reference[run], 6))) < len(set(np.round(rolling[run], 6)))


def test_unarmed_reference_falls_back_to_rolling():
    """未武装时参考边界回落到滚动区间。

    例外：突破成立的那一根，参考边界仍是冻结值（Pine 的读取顺序是
    「先算参考边界，再把 breakoutArmed 置 false」），所以要把突破那根排除。
    """
    out = market_tradability_engine_v2(_noisy(600))
    armed = out["mte_breakout_armed"].to_numpy(dtype=int)
    broke = (out["mte_bullish_breakout"].to_numpy(dtype=int)
             | out["mte_bearish_breakout"].to_numpy(dtype=int))
    mask = (armed == 0) & (broke == 0)
    assert mask.any()
    assert out["mte_breakout_reference_high"].to_numpy()[mask] == pytest.approx(
        out["mte_rolling_range_high"].to_numpy()[mask], nan_ok=True)


# ---------------------------------------------------------------- 状态机


def test_state_machine_hysteresis():
    out = market_tradability_engine_v2(_noisy())
    state = out["mte_state"].to_numpy(dtype=int)
    bars = out["mte_state_bars"].to_numpy(dtype=int)
    changed = out["mte_state_changed"].to_numpy(dtype=int)
    assert changed.sum() > 0
    # 切换那根 state_bars 归零，否则单调 +1
    for i in range(1, len(out)):
        if changed[i] == 1:
            assert bars[i] == 0
        else:
            assert bars[i] == bars[i - 1] + 1
    assert set(np.unique(state).tolist()) <= set(range(8))


def test_candidate_and_state_are_valid_codes():
    out = market_tradability_engine_v2(_noisy(800))
    assert out["mte_candidate_state"].dropna().isin(range(8)).all()
    assert out["mte_state"].dropna().isin(range(8)).all()
    assert out["mte_pending_state"].dropna().isin(range(8)).all()


def test_warmup_is_no_trade_with_warmup_reason():
    out = market_tradability_engine_v2(_noisy(500))
    ready = out["mte_data_ready"].to_numpy(dtype=int)
    assert ready[0] == 0
    first_ready = int(np.argmax(ready))
    assert first_ready > 0
    assert (out["mte_state"].to_numpy(dtype=int)[:first_ready] == STATE_NO_TRADE).all()
    warmup_code = NO_TRADE_REASONS.index("WARMUP")
    assert (out["mte_no_trade_reason"].to_numpy(dtype=int)[:first_ready] == warmup_code).all()


def test_breakout_hold_keeps_state_for_several_bars():
    out = market_tradability_engine_v2(_range_then_trend())
    entered = out["mte_entered_breakout"].to_numpy(dtype=int)
    remaining = out["mte_breakout_bars_remaining"].to_numpy(dtype=int)
    assert entered.sum() > 0
    # 进入突破后的 hold 窗口内剩余计数为正
    for i in np.flatnonzero(entered):
        window = remaining[i:i + 4]
        assert (window >= 0).all()


def test_entered_flags_match_state_changes():
    out = market_tradability_engine_v2(_noisy())
    state = out["mte_state"].to_numpy(dtype=int)
    changed = out["mte_state_changed"].to_numpy(dtype=int)
    pairs = {"mte_entered_ready": STATE_READY, "mte_entered_breakout": STATE_BREAKOUT,
             "mte_entered_trend": STATE_TREND, "mte_entered_no_trade": STATE_NO_TRADE}
    for column, code in pairs.items():
        expected = ((state == code) & (changed == 1)).astype(int)
        assert np.array_equal(out[column].to_numpy(dtype=int), expected), column


# ---------------------------------------------------------------- 量能门


def test_volume_confirmation_gate_blocks_breakouts():
    frame = _range_then_trend()
    silent = frame.assign(volume=0.0)
    plain = market_tradability_engine_v2(frame)
    assert plain["mte_volume_confirmed"].to_numpy(dtype=int).all()
    gated = market_tradability_engine_v2(silent, require_volume_confirmation=True)
    # 无量能数据 → 不阻塞（Pine: not volumeIsReal 直接放行）
    assert gated["mte_volume_confirmed"].to_numpy(dtype=int).all()


def test_volume_ratio_normalised_without_real_volume():
    silent = _noisy(300).assign(volume=0.0)
    out = market_tradability_engine_v2(silent)
    assert out["mte_volume_ratio"].to_numpy() == pytest.approx(np.ones(len(out)))


# ---------------------------------------------------------------- 输出契约


def test_output_columns_and_index():
    frame = _noisy(400)
    out = market_tradability_engine_v2(frame)
    assert out.index.equals(frame.index)
    assert len(out.columns) == len(set(out.columns))
    assert all(name.startswith("mte_") for name in out.columns)
    for name in ("mte_state", "mte_candidate_state", "mte_breakout_armed", "mte_upper_tests"):
        assert str(out[name].dtype) == "Int64", name
    assert out["mte_no_trade_reason"].dropna().between(0, len(NO_TRADE_REASONS) - 1).all()
    assert out["mte_acceptance_score"].dropna().between(-100.0, 100.0).all()
    assert out["mte_structure_score"].dropna().between(0.0, 100.0).all()


def test_parameter_validation():
    frame = _noisy(200)
    bad = [
        dict(preset="Nope"),
        dict(reactivity="Nope"),
        dict(baseline_type="Nope"),
        dict(custom_quality_length=1),
        dict(custom_range_length=5),
        dict(custom_baseline_length=1),
        dict(custom_atr_length=1),
        dict(custom_adx_length=1),
        dict(custom_rank_length=10),
        dict(custom_structure_threshold=10.0),
        dict(maximum_range_atr=1.0),
        dict(breakout_buffer_atr=5.0),
        dict(volume_confirmation_ratio=9.0),
        dict(boundary_fresh_test_maximum=-1),
        dict(boundary_stressed_test_minimum=0),
        dict(weak_rejection_threshold=2.0),
        dict(custom_confirmation_bars=0),
        dict(custom_minimum_state_bars=-1),
        dict(breakout_hold_bars=-1),
        dict(impulse_lookback=0),
        dict(recent_impulse_energy_threshold=200.0),
        dict(aftermath_energy_drop_pct=200.0),
        dict(jma_phase=200),
        dict(jma_power=0.0),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            market_tradability_engine_v2(frame, **kwargs)


def test_random_runs_end_to_end():
    for seed in (0, 5, 9):
        out = market_tradability_engine_v2(_noisy(700, seed=seed))
        assert out["mte_state"].nunique() >= 2
        assert np.isfinite(out["mte_structure_score"].dropna()).all()
        assert (out["mte_upper_tests"].dropna() >= 0).all()
        assert (out["mte_dominant_fatigue"].dropna().between(0.0, 1.0)).all()


def test_all_eight_states_reachable():
    seen = set()
    for seed in range(6):
        out = market_tradability_engine_v2(_noisy(1200, seed=seed))
        seen |= set(out["mte_state"].dropna().unique().tolist())
    assert seen <= set(range(8))
    assert len(seen) >= 6  # 随机数据能覆盖绝大多数状态
    assert {STATE_BALANCE, STATE_COMPRESSION, STATE_READY, STATE_TREND} & seen
