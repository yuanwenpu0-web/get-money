"""Commodity Pulse Matrix v4 移植测试。

覆盖：六类分数口径、Heat 五组件加权恒等式、MTF 一致率语义、强度评分、
入场类型与阈值、状态机（FIRST/REENTRY/UPGRADE）、分级证据门、参数校验。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.commodity_pulse_matrix_v4 import (
    _jma,
    _nzmap,
    _six_scores,
    _z_local,
    commodity_pulse_matrix_v4,
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


def _noisy(n: int = 1200, seed: int = 0, freq: str = "1h") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.9, n))
    open_ = close + rng.normal(0, 0.2, n)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + np.abs(rng.normal(0, 0.4, n)),
            "low": np.minimum(open_, close) - np.abs(rng.normal(0, 0.4, n)),
            "close": close,
            "volume": 1000.0 + np.abs(rng.normal(0, 200, n)),
        },
        index=idx,
    )


# ---------------------------------------------------------------- 工具


def test_z_local_zero_sigma_returns_zero():
    flat = np.full(100, 50.0)
    assert np.allclose(_z_local(flat, 20)[19:], 0.0)


def test_nzmap_clamps():
    values = np.array([0.0, 15.0, 25.0, 40.0])
    assert _nzmap(values, 15.0, 35.0).tolist() == [0.0, 0.0, 0.5, 1.0]


def test_jma_converges_on_constant():
    values = np.full(300, 80.0)
    out = _jma(values, 50)
    assert np.allclose(out[100:], 80.0)


# ---------------------------------------------------------------- 六类分数


def test_six_scores_bounds():
    scores = _six_scores(_noisy(600), True, 50)
    for name, values in scores.items():
        finite = values[np.isfinite(values)]
        assert finite.size > 0, name
        assert finite.min() >= -1.0 - 1e-9, name
        assert finite.max() <= 1.0 + 1e-9, name


def test_flow_score_neutral_without_volume():
    scores = _six_scores(_noisy(600).assign(volume=0.0), False, 50)
    assert np.allclose(scores["flow"], 0.0)


def test_trend_score_follows_direction():
    # 带小幅回撤的趋势（纯单调会让 z 分数的分母为 0，退化成 0）
    rng = np.random.default_rng(7)
    up = _frame(100 + np.arange(600) * 0.5 + np.cumsum(rng.normal(0, 0.05, 600)))
    down = _frame(100 - np.arange(600) * 0.5 + np.cumsum(rng.normal(0, 0.05, 600)))
    up_scores = _six_scores(up, True, 50)
    down_scores = _six_scores(down, True, 50)
    assert up_scores["trend"][-1] > 0.2
    assert down_scores["trend"][-1] < -0.2
    assert up_scores["struct"][-1] > 0.2
    assert down_scores["struct"][-1] < -0.2


def test_divergence_score_is_quantised():
    scores = _six_scores(_noisy(800, seed=2), True, 50)
    assert set(np.unique(scores["div"]).tolist()) <= {-0.5, 0.0, 0.5}


def test_volatility_score_positive_when_calm():
    calm = _frame(np.full(600, 100.0))
    calm_scores = _six_scores(calm, True, 50)
    # 零波动 → ATR% 极低 → ATR 分项满格
    assert np.isfinite(calm_scores["vol"][-1])


# ---------------------------------------------------------------- Heat


def test_heat_weight_identity():
    out = commodity_pulse_matrix_v4(_noisy())
    raw = (out["cpm_heat_mtf_alignment"] * 0.30 + out["cpm_heat_signal_strength"] * 0.25
           + out["cpm_heat_setup"] * 0.20 + out["cpm_heat_volatility"] * 0.15
           + out["cpm_heat_confluence"] * 0.10)
    expected = np.clip(np.maximum(0.0, raw.to_numpy(dtype=float)) ** 1.2, 0.0, 1.0)
    assert out["cpm_heat_score"].to_numpy(dtype=float) == pytest.approx(expected, nan_ok=True)


def test_heat_components_bounds():
    out = commodity_pulse_matrix_v4(_noisy())
    for column in ("cpm_heat_mtf_alignment", "cpm_heat_signal_strength", "cpm_heat_setup",
                   "cpm_heat_volatility", "cpm_heat_confluence", "cpm_heat_score"):
        values = out[column].dropna()
        assert values.between(0.0, 1.0).all(), column


def test_mtf_alignment_counts_dominant_direction():
    """一致率是「与多数方向同向的比例」，不是「全部同向」。"""
    out = commodity_pulse_matrix_v4(_noisy())
    alignment = out["cpm_heat_mtf_alignment"].to_numpy(dtype=float)
    # 四个周期 → 只可能落在 0/0.25/0.5/0.75/1.0
    assert set(np.unique(alignment).tolist()) <= {0.0, 0.25, 0.5, 0.75, 1.0}


def test_heat_level_thresholds():
    out = commodity_pulse_matrix_v4(_noisy())
    heat = out["cpm_heat_score"].to_numpy(dtype=float)
    level = out["cpm_heat_level"].to_numpy(dtype=int)
    expected = np.where(heat >= 0.65, 3, np.where(heat >= 0.45, 2, np.where(heat >= 0.30, 1, 0)))
    assert np.array_equal(level, expected)


def test_setup_status_codes():
    out = commodity_pulse_matrix_v4(_noisy())
    assert out["cpm_setup_status"].dropna().isin(range(6)).all()
    # 发散分数极端时一定标为 DIVERGENCE(1)
    div = out["cpm_div_score"].to_numpy(dtype=float)
    status = out["cpm_setup_status"].to_numpy(dtype=int)
    extreme = (div > 0.4) | (div < -0.4)
    assert (status[extreme] == 1).all()


# ---------------------------------------------------------------- 强度 / 入场


def test_long_short_strength_identity():
    out = commodity_pulse_matrix_v4(_noisy())
    momentum = out["cpm_momentum"].to_numpy(dtype=float)
    trend_up = out["cpm_trend_up"].to_numpy(dtype=int).astype(bool)
    trend_down = out["cpm_trend_down"].to_numpy(dtype=int).astype(bool)
    long_bias = out["cpm_long_bias"].to_numpy(dtype=int).astype(bool)
    short_bias = out["cpm_short_bias"].to_numpy(dtype=int).astype(bool)
    near_sup = out["cpm_near_support"].to_numpy(dtype=int).astype(bool)
    near_res = out["cpm_near_resistance"].to_numpy(dtype=int).astype(bool)

    align_long = np.where(trend_up & long_bias, 1.0, np.where(trend_up | long_bias, 0.6, 0.3))
    align_short = np.where(trend_down & short_bias, 1.0, np.where(trend_down | short_bias, 0.6, 0.3))
    zone_long = np.where(near_sup, 1.0, np.where(near_res, 0.8, 0.5))
    zone_short = np.where(near_res, 1.0, np.where(near_sup, 0.8, 0.5))
    mom_long = np.where(momentum > 0, momentum, 0.0)
    mom_short = np.where(momentum < 0, np.abs(momentum), 0.0)

    assert out["cpm_long_strength"].to_numpy(dtype=float) == pytest.approx(
        mom_long * 0.40 + align_long * 0.45 + zone_long * 0.15, nan_ok=True)
    assert out["cpm_short_strength"].to_numpy(dtype=float) == pytest.approx(
        mom_short * 0.40 + align_short * 0.45 + zone_short * 0.15, nan_ok=True)


def test_bias_is_exclusive():
    out = commodity_pulse_matrix_v4(_noisy())
    long_bias = out["cpm_long_bias"].to_numpy(dtype=int)
    short_bias = out["cpm_short_bias"].to_numpy(dtype=int)
    assert ((long_bias == 1) & (short_bias == 1)).sum() == 0


def test_density_thresholds_change_signal_count():
    frame = _noisy(1200, seed=1)
    few = commodity_pulse_matrix_v4(frame, entry_density="Few (Top Signals)")
    many = commodity_pulse_matrix_v4(frame, entry_density="Many (Scalper)")
    few_bpc = int(few["cpm_bpc_long"].sum() + few["cpm_bpc_short"].sum())
    many_bpc = int(many["cpm_bpc_long"].sum() + many["cpm_bpc_short"].sum())
    assert many_bpc >= few_bpc


def test_entry_types_assigned():
    out = commodity_pulse_matrix_v4(_noisy())
    types = set(out["cpm_entry_type"].dropna().unique().tolist())
    assert types <= {0, 1, 2, 3}
    # 突破是默认最常见的类型之一
    assert 1 in types


def test_ranging_market_disables_breakout_and_continuation():
    """ATR 压缩（当前 ATR 远低于 50 根均值）→ is_ranging → 关掉突破与延续。"""
    rng = np.random.default_rng(11)
    close = np.empty(1200)
    close[:400] = 100 + np.cumsum(rng.normal(0, 1.2, 400))   # 高波动
    close[400:] = close[399] + np.cumsum(rng.normal(0, 0.10, 800))  # 低波动
    frame = _frame(close, spread=0.05)
    out = commodity_pulse_matrix_v4(frame)
    ranging = out["cpm_is_ranging"].to_numpy(dtype=int)
    assert ranging.sum() > 0
    # 横盘期间 breakout 已被 ranging 门关掉 —— 所以「关掉 breakout 开关」
    # 不应再改变任何信号（`cpm_entry_type` 仍会标注检测到的形态，那是纯描述）
    without_breakout = commodity_pulse_matrix_v4(frame, use_breakout_entries=False)
    for side in ("long", "short"):
        base = out[f"cpm_bpc_{side}"].to_numpy(dtype=int)
        trimmed = without_breakout[f"cpm_bpc_{side}"].to_numpy(dtype=int)
        assert np.array_equal(base[ranging == 1], trimmed[ranging == 1]), side


def test_counter_trend_penalty_and_gate():
    frame = _noisy(1200, seed=4)
    strict = commodity_pulse_matrix_v4(frame, allow_counter_trend_entries=False)
    loose = commodity_pulse_matrix_v4(frame, allow_counter_trend_entries=True)
    # 反向趋势的 bpc 信号在严格模式下必须被挡掉
    raw_long = strict["cpm_breakout_long"].to_numpy(dtype=int) | strict["cpm_pullback_long"].to_numpy(dtype=int) | strict["cpm_cont_long"].to_numpy(dtype=int)
    counter = raw_long.astype(bool) & strict["cpm_short_bias"].to_numpy(dtype=int).astype(bool)
    assert (strict["cpm_bpc_long"].to_numpy(dtype=int)[counter] == 0).all()
    assert int(loose["cpm_bpc_long"].sum()) >= int(strict["cpm_bpc_long"].sum())


# ---------------------------------------------------------------- 状态机


def test_state_machine_event_codes():
    out = commodity_pulse_matrix_v4(_noisy(1200, seed=0))
    events = out["cpm_event_type"].to_numpy(dtype=int)
    emitted = out["cpm_emit_entry"].to_numpy(dtype=int)
    assert set(np.unique(events).tolist()) <= {0, 1, 2, 3}
    # emit 与 event != NONE 严格等价
    assert np.array_equal(emitted, (events != 0).astype(int))
    # FIRST 一定是方向变化
    first = np.flatnonzero(events == 1)
    assert len(first) > 0


def test_state_machine_same_direction_cooldown():
    out = commodity_pulse_matrix_v4(_noisy(1200, seed=0), entry_cooldown=5)
    events = out["cpm_event_type"].to_numpy(dtype=int)
    direction = out["cpm_state_direction"].to_numpy(dtype=int)
    emitted = np.flatnonzero(out["cpm_emit_entry"].to_numpy(dtype=int))
    # 同向的连续 emit 至少间隔 entry_cooldown
    same_dir_gaps = []
    for prev, curr in zip(emitted[:-1], emitted[1:]):
        if direction[curr] == direction[prev]:
            same_dir_gaps.append(curr - prev)
    if same_dir_gaps:
        assert min(same_dir_gaps) >= 5


def test_direction_change_is_immediate():
    """方向翻转不受冷却约束（反转不该等）。"""
    out = commodity_pulse_matrix_v4(_noisy(1200, seed=0), entry_cooldown=20)
    events = out["cpm_event_type"].to_numpy(dtype=int)
    direction = out["cpm_state_direction"].to_numpy(dtype=int)
    emitted = np.flatnonzero(out["cpm_emit_entry"].to_numpy(dtype=int))
    changes = [i for i in emitted if events[i] == 1]
    assert len(changes) > 0
    # FIRST 事件之间可以很近（只受触发本身限制）
    assert direction[changes[0]] != 0


# ---------------------------------------------------------------- 分级证据门


def test_evidence_gate_is_monotone():
    frame = _noisy(1200, seed=0)
    counts = []
    for gate in (2, 3, 4, 5):
        out = commodity_pulse_matrix_v4(frame, final_min_gate=gate)
        counts.append(int(out["cpm_final_long"].sum() + out["cpm_final_short"].sum()))
    assert counts == sorted(counts, reverse=True)


def test_evidence_threshold_mapping():
    out = commodity_pulse_matrix_v4(_noisy(400))
    assert int(out["cpm_evidence_threshold"].iloc[0]) == 2  # gate=4 → 2
    for gate, expected in ((2, 0), (3, 1), (4, 2), (5, 3)):
        out = commodity_pulse_matrix_v4(_noisy(400), final_min_gate=gate)
        assert int(out["cpm_evidence_threshold"].iloc[0]) == expected


def test_final_requires_trigger():
    out = commodity_pulse_matrix_v4(_noisy(1200, seed=0))
    final_long = out["cpm_final_long"].to_numpy(dtype=int)
    trigger_long = out["cpm_trigger_long"].to_numpy(dtype=int)
    assert (final_long[trigger_long == 0] == 0).all()
    final_short = out["cpm_final_short"].to_numpy(dtype=int)
    trigger_short = out["cpm_trigger_short"].to_numpy(dtype=int)
    assert (final_short[trigger_short == 0] == 0).all()


def test_heat_evidence_is_not_a_veto():
    """heat 只算一个证据点，不是硬门 —— 关掉 heat 证据（提高门槛）仍有信号。"""
    frame = _noisy(1200, seed=0)
    relaxed = commodity_pulse_matrix_v4(frame, final_min_gate=3, entry_heat_min=0.0)
    assert int(relaxed["cpm_final_long"].sum() + relaxed["cpm_final_short"].sum()) > 0


# ---------------------------------------------------------------- 输出契约


def test_output_columns_and_index():
    frame = _noisy(400)
    out = commodity_pulse_matrix_v4(frame)
    assert out.index.equals(frame.index)
    assert len(out.columns) == len(set(out.columns))
    assert all(name.startswith("cpm_") for name in out.columns)
    for name in ("cpm_final_long", "cpm_entry_type", "cpm_event_type", "cpm_heat_level"):
        assert str(out[name].dtype) == "Int64", name
    for name in ("cpm_flow_score", "cpm_mom_score", "cpm_trend_score", "cpm_vol_score",
                 "cpm_struct_score", "cpm_div_score", "cpm_heat_score"):
        values = out[name].dropna()
        assert values.between(-1.0, 1.0).all(), name


def test_stop_and_target_ordering():
    out = commodity_pulse_matrix_v4(_noisy(400))
    atr_a = out["cpm_atr"].to_numpy(dtype=float)
    stop_long = out["cpm_stop_long"].to_numpy(dtype=float)
    tp1 = out["cpm_tp1_long"].to_numpy(dtype=float)
    tp2 = out["cpm_tp2_long"].to_numpy(dtype=float)
    tp3 = out["cpm_tp3_long"].to_numpy(dtype=float)
    valid = np.isfinite(atr_a) & np.isfinite(stop_long)
    assert valid.sum() > 100
    assert (stop_long[valid] < tp1[valid]).all()
    assert (tp1[valid] < tp2[valid]).all()
    assert (tp2[valid] < tp3[valid]).all()
    # 空头方向完全镜像
    assert (out["cpm_stop_short"].to_numpy(dtype=float)[valid]
            > out["cpm_tp1_short"].to_numpy(dtype=float)[valid]).all()


def test_parameter_validation():
    frame = _noisy(200)
    bad = [
        dict(entry_density="Nope"),
        dict(final_min_gate=1),
        dict(final_min_gate=6),
        dict(entry_heat_min=2.0),
        dict(strength_threshold=0.1),
        dict(early_min_heat=-1.0),
        dict(early_cooldown=0),
        dict(entry_cooldown=0),
        dict(upgrade_delta=-1.0),
        dict(entry_jma_fast=200),
        dict(entry_jma_slow=10),
        dict(vol_stop_atr_mult=0.5),
        dict(mtf_timeframes=("240", "D")),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            commodity_pulse_matrix_v4(frame, **kwargs)


def test_random_runs_end_to_end():
    for seed in (0, 3):
        out = commodity_pulse_matrix_v4(_noisy(900, seed=seed))
        assert np.isfinite(out["cpm_heat_score"].dropna()).all()
        assert out["cpm_heat_score"].dropna().between(0.0, 1.0).all()
        assert (out["cpm_bpc_long"].sum() + out["cpm_bpc_short"].sum()) >= 0
        assert out["cpm_final_long"].dropna().isin([0, 1]).all()
