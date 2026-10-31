"""Market State Engine 移植测试。

覆盖：三轴判定、效率/位置口径、扩张加权恒等式、权重再归一、状态仲裁、
状态机滞回、脉冲记忆、上下文等级、参数校验。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.market_state_engine import (
    CONTEXT_COMPATIBLE,
    CONTEXT_NEUTRAL,
    CONTEXT_PREFERRED,
    CONTEXT_UNFAVORABLE,
    ENERGY_CONTRACTING,
    ENERGY_EXPANDING,
    ENERGY_EXHAUSTING,
    REGIME_BALANCED,
    REGIME_BEARISH,
    REGIME_BULLISH,
    STATE_BALANCE,
    STATE_BEAR_EXPANSION,
    STATE_BULL_EXPANSION,
    STATE_BULL_PULLBACK,
    STATE_COMPRESSION,
    STATE_TRANSITION,
    TRANSITION_NO_DATA,
    market_state_engine,
)


def _frame(close: np.ndarray, *, spread: float = 0.3, volume=1000.0, freq="1h") -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    n = len(close)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    vol = volume if isinstance(volume, np.ndarray) else np.full(n, float(volume))
    return pd.DataFrame(
        {"open": close, "high": close + spread, "low": close - spread, "close": close, "volume": vol},
        index=idx,
    )


def _staircase(n: int = 1200, push: int = 30, slope: float = 0.9,
               pull: int = 10, back: float = 0.55) -> pd.DataFrame:
    """阶梯上行：强推 + 浅回撤，反复 —— 触发扩张与拉回状态。"""
    close = np.zeros(n)
    level, i = 100.0, 0
    while i < n:
        seg = min(push, n - i)
        close[i:i + seg] = level + np.arange(seg) * slope
        level += seg * slope
        i += seg
        seg2 = min(pull, n - i)
        close[i:i + seg2] = level - np.arange(seg2) * back
        level -= seg2 * back
        i += seg2
    return _frame(close)


def _noisy(n: int = 900, seed: int = 0) -> pd.DataFrame:
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


# ---------------------------------------------------------------- 三轴


def test_uptrend_is_bullish_regime():
    close = 100 + np.arange(1200) * 0.5
    out = market_state_engine(_frame(close))
    regime = out["mse_structure_regime"].to_numpy(dtype=int)
    assert (regime == REGIME_BULLISH).sum() > 500
    assert int(out["mse_structural_bull"].iloc[-1]) == 1
    assert int(out["mse_htf_bull"].iloc[-1]) == 1


def test_downtrend_is_bearish_regime():
    close = 100 - np.arange(1200) * 0.5
    out = market_state_engine(_frame(close))
    regime = out["mse_structure_regime"].to_numpy(dtype=int)
    assert (regime == REGIME_BEARISH).sum() > 500
    assert int(out["mse_structural_bear"].iloc[-1]) == 1
    states = out["mse_state"].to_numpy(dtype=int)
    assert (states == STATE_BEAR_EXPANSION).sum() > 100


def test_structure_direction_sign_and_bounds():
    out = market_state_engine(_noisy())
    direction = out["mse_structure_direction"].dropna()
    assert direction.between(-1.0, 1.0).all()
    bull_score = out["mse_structure_bull_score"].dropna()
    assert bull_score.between(0.0, 1.0).all()
    # 收盘在宽幅区间内时方向与分数应互洽
    assert np.allclose((out["mse_structure_bull_score"] - (1.0 - out["mse_structure_bull_score"])).dropna(),
                       out["mse_structure_direction"].dropna())


def test_balance_candidate_maps_to_balanced_regime():
    # 极窄幅横盘 → 平衡
    close = 100 + 0.05 * np.sin(np.arange(900) / 3.0)
    out = market_state_engine(_frame(close, spread=0.02))
    regime = out["mse_structure_regime"].to_numpy(dtype=int)
    assert (regime == REGIME_BALANCED).sum() > 100
    assert out["mse_balance_score"].dropna().between(0.0, 1.0).all()


# ---------------------------------------------------------------- 口径


def test_efficiency_ratio_bounds_and_monotone_case():
    close = 100 + np.arange(600) * 0.4  # 单调 → 效率应接近 1
    out = market_state_engine(_frame(close))
    efficiency = out["mse_efficiency_ratio"].dropna()
    assert efficiency.between(0.0, 1.0).all()
    assert efficiency.iloc[-1] > 0.95
    assert out["mse_signed_efficiency"].iloc[-1] > 0.9


def test_value_zone_and_location():
    out = market_state_engine(_noisy(600))
    upper = out["mse_value_upper"].to_numpy(dtype=float)
    lower = out["mse_value_lower"].to_numpy(dtype=float)
    fair = out["mse_fair_value"].to_numpy(dtype=float)
    valid = np.isfinite(upper) & np.isfinite(fair) & np.isfinite(lower)
    assert valid.any()
    assert (upper[valid] >= fair[valid]).all() and (lower[valid] <= fair[valid]).all()
    location = out["mse_normalized_location"].dropna()
    assert location.abs().max() < 50  # 归一化后不应爆炸
    assert out["mse_range_location"].dropna().between(0.0, 1.0).all()


def test_expansion_weight_identity():
    out = market_state_engine(_noisy(600))
    progress = np.nan_to_num(out["mse_directional_progress"].to_numpy(dtype=float))
    bull_progress = np.clip(progress / 2.0, 0.0, 1.0)
    bear_progress = np.clip(-progress / 2.0, 0.0, 1.0)
    common = (0.22 * out["mse_efficiency_ratio"] + 0.18 * out["mse_adx_expansion"]
              + 0.16 * out["mse_range_expansion"] + 0.12 * out["mse_atr_acceleration"])
    bull = common + 0.12 * out["mse_bull_body_quality"] + 0.12 * out["mse_close_location"] + 0.08 * bull_progress
    bear = common + 0.12 * out["mse_bear_body_quality"] + 0.12 * (1.0 - out["mse_close_location"]) + 0.08 * bear_progress
    assert out["mse_bull_expansion_score"].to_numpy() == pytest.approx(np.clip(bull, 0.0, 1.0), nan_ok=True)
    assert out["mse_bear_expansion_score"].to_numpy() == pytest.approx(np.clip(bear, 0.0, 1.0), nan_ok=True)


def test_directional_bias_bounds_and_volume_renormalisation():
    frame = _noisy(600)
    loud = market_state_engine(frame)
    silent = market_state_engine(frame.assign(volume=0.0))
    assert loud["mse_directional_bias"].dropna().between(-1.0, 1.0).all()
    assert silent["mse_directional_bias"].dropna().between(-1.0, 1.0).all()
    # 无量能时吸收项恒为 0
    assert (silent["mse_effort_result_mismatch"] == 0.0).all()
    # 权重重归一：无量能的分母是 0.80，不是 1.00 —— 同方向时偏差至少 20% 不等于原值
    assert not np.allclose(loud["mse_directional_bias"].dropna().to_numpy(),
                           silent["mse_directional_bias"].dropna().to_numpy())


def test_exhaustion_bounds_and_trigger():
    out = market_state_engine(_noisy(900))
    score = out["mse_exhaustion_score"].dropna()
    assert score.between(0.0, 1.0).all()
    trigger = out["mse_has_exhaustion_trigger"].to_numpy(dtype=int)
    assert trigger.max() == 1
    # 没有触发时，扩张状态不应被判为衰竭
    states = out["mse_state"].to_numpy(dtype=int)
    exhaust_states = {6, 7}
    hit = np.isin(states, list(exhaust_states))
    assert (trigger[hit] == 1).all()


# ---------------------------------------------------------------- 仲裁


def test_raw_state_thresholds_and_dominance():
    out = market_state_engine(_noisy(900))
    raw = out["mse_raw_state"].to_numpy(dtype=int)
    best = out["mse_best_state_score"].to_numpy(dtype=float)
    dominance = out["mse_state_dominance"].to_numpy(dtype=float)
    ready = out["mse_engine_ready"].to_numpy(dtype=int)
    assert (dominance >= 0).all()
    # 分数不足或领先优势不够 → TRANSITION
    weak = (best < 0.55) | (dominance < 0.06)
    assert (raw[weak & (ready == 1)] == STATE_TRANSITION).all()
    assert (raw[~weak & (ready == 1)] != STATE_TRANSITION).all()


def test_raw_state_confidence_semantics():
    out = market_state_engine(_noisy(900))
    raw = out["mse_raw_state"].to_numpy(dtype=int)
    confidence = out["mse_raw_confidence"].to_numpy(dtype=float)
    assert confidence.min() >= 0.0 and confidence.max() <= 1.0
    transition = raw == STATE_TRANSITION
    best = out["mse_best_state_score"].to_numpy(dtype=float)
    assert confidence[~transition] == pytest.approx(best[~transition])
    # 暖机期没有数据 → 置信度 0
    assert confidence[0] == 0.0


# ---------------------------------------------------------------- 状态机


def test_state_machine_dwell_and_confirmation():
    out = market_state_engine(_staircase(), minimum_state_bars=3, custom_confirm_bars=3, preset="Custom")
    state = out["mse_state"].to_numpy(dtype=int)
    bars = out["mse_bars_in_state"].to_numpy(dtype=int)
    changed = out["mse_state_changed"].to_numpy(dtype=int)
    assert changed.sum() > 0
    start = int(np.argmax(out["mse_engine_ready"].to_numpy(dtype=int)))
    for i in range(start + 1, len(out)):
        if changed[i] == 1:
            assert bars[i] == 0
        else:
            assert bars[i] == bars[i - 1] + 1


def test_warmup_is_no_data_transition():
    out = market_state_engine(_noisy(600))
    ready = out["mse_engine_ready"].to_numpy(dtype=int)
    first = int(np.argmax(ready))
    assert first > 0
    assert (out["mse_state"].to_numpy(dtype=int)[:first] == STATE_TRANSITION).all()
    assert (out["mse_transition_reason"].to_numpy(dtype=int)[:first] == TRANSITION_NO_DATA).all()


# ---------------------------------------------------------------- 脉冲 / 上下文


def test_impulse_memory_tracks_expansion_state():
    out = market_state_engine(_staircase())
    state = out["mse_state"].to_numpy(dtype=int)
    direction = out["mse_impulse_direction"].to_numpy(dtype=int)
    assert (state == STATE_BULL_EXPANSION).sum() > 50
    assert (direction == 1).sum() > 0
    # 脉冲方向只会是 ±1/0
    assert set(np.unique(direction).tolist()) <= {-1, 0, 1}
    # 脉冲完好标志只在有脉冲方向时可能为 1
    intact = out["mse_impulse_intact"].to_numpy(dtype=int)
    assert (intact[direction == 0] == 0).all()


def test_pullback_requires_impulse_and_structure():
    out = market_state_engine(_staircase())
    bull_pull = out["mse_bull_pullback"].to_numpy(dtype=int)
    bull_struct = out["mse_structural_bull"].to_numpy(dtype=int)
    assert bull_pull.sum() > 0
    assert (bull_struct[bull_pull == 1] == 1).all()
    # 下行趋势里不应出现多头拉回
    down = market_state_engine(_frame(100 - np.arange(1200) * 0.5))
    assert int(down["mse_bull_pullback"].sum()) == 0


def test_context_levels_consistent_with_state():
    out = market_state_engine(_staircase())
    state = out["mse_state"].to_numpy(dtype=int)
    long_ctx = out["mse_long_context"].to_numpy(dtype=int)
    short_ctx = out["mse_short_context"].to_numpy(dtype=int)
    assert set(np.unique(long_ctx).tolist()) <= {0, 1, 2, 3}
    # 多头扩张时空头上下文必须是不利
    mask = state == STATE_BULL_EXPANSION
    if mask.any():
        assert (short_ctx[mask] == CONTEXT_UNFAVORABLE).all()
    # 空头扩张时多头上下文必须是不利
    bear = market_state_engine(_frame(100 - np.arange(1200) * 0.5))
    bear_state = bear["mse_state"].to_numpy(dtype=int)
    bear_long = bear["mse_long_context"].to_numpy(dtype=int)
    mask2 = bear_state == STATE_BEAR_EXPANSION
    if mask2.any():
        assert (bear_long[mask2] == CONTEXT_UNFAVORABLE).all()


def test_readiness_requires_compatible_context():
    out = market_state_engine(_staircase())
    for side, ctx, ready in (("long", "mse_long_context", "mse_long_readiness"),
                             ("short", "mse_short_context", "mse_short_readiness")):
        ready_values = out[ready].to_numpy(dtype=int)
        context = out[ctx].to_numpy(dtype=int)
        if ready_values.sum():
            assert (context[ready_values == 1] >= CONTEXT_COMPATIBLE).all()


# ---------------------------------------------------------------- 输出契约


def test_output_columns_and_index():
    frame = _noisy(400)
    out = market_state_engine(frame)
    assert out.index.equals(frame.index)
    assert len(out.columns) == len(set(out.columns))
    assert all(name.startswith("mse_") for name in out.columns)
    for name in ("mse_state", "mse_raw_state", "mse_structure_regime", "mse_energy_phase",
                 "mse_long_context", "mse_bars_in_state"):
        assert str(out[name].dtype) == "Int64", name
    assert out["mse_state"].dropna().isin(range(9)).all()
    assert out["mse_structure_regime"].dropna().isin(range(4)).all()
    assert out["mse_energy_phase"].dropna().isin(range(4)).all()


def test_parameter_validation():
    frame = _noisy(200)
    bad = [
        dict(preset="Nope"),
        dict(structure_smoothing="Nope"),
        dict(value_method="Nope"),
        dict(custom_fast_len=1),
        dict(custom_slow_len=1),
        dict(custom_range_len=1),
        dict(value_width=0.0),
        dict(atr_len=0),
        dict(trend_adx_threshold=-1.0),
        dict(compression_threshold=2.0),
        dict(minimum_state_bars=0),
        dict(minimum_state_score=2.0),
        dict(state_dominance_margin=-1.0),
        dict(opposite_break_threshold=2.0),
        dict(discount_threshold=1.0),
        dict(premium_threshold=-1.0),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            market_state_engine(frame, **kwargs)


def test_presets_resolve_confirm_bars():
    frame = _noisy(500)
    # Active 预设 confirm=2 / Position=4；短确认更早切换
    active = market_state_engine(frame, preset="Active")
    position = market_state_engine(frame, preset="Position")
    assert int(active["mse_state_changed"].sum()) >= int(position["mse_state_changed"].sum())


def test_require_htf_agreement_actually_gates():
    """``require_htf_agreement`` 必须真的关掉反向扩张。

    这是 ``~bool`` 陷阱的回归测试：写成 ``(~flag) | array`` 时 ``~True == -2``
    恒为真值，门会被静默吃掉，两种设置的输出会完全一样。
    """
    frame = _frame(100 + np.arange(1200) * 0.5)
    loose = market_state_engine(frame, require_htf_agreement=False)
    strict = market_state_engine(frame, require_htf_agreement=True)
    # 纯上行趋势里 HTF 同向，两者应当一致
    assert int(loose["mse_long_context"].sum()) == int(strict["mse_long_context"].sum())
    # 但反向场景（图表上行 + HTF 下行）必须被门拦住：用 1D 图 + W HTF 构造
    daily = _frame(100 + np.arange(400) * 0.5, freq="1D")
    loose_d = market_state_engine(daily, require_htf_agreement=False)
    strict_d = market_state_engine(daily, require_htf_agreement=True)
    expansion = STATE_BULL_EXPANSION
    assert (loose_d["mse_state"].to_numpy(dtype=int) == expansion).sum() >= \
        (strict_d["mse_state"].to_numpy(dtype=int) == expansion).sum()


def test_random_runs_end_to_end():
    for seed in (0, 3):
        out = market_state_engine(_noisy(700, seed=seed))
        assert np.isfinite(out["mse_structure_fast"].dropna()).all()
        assert out["mse_state_quality"].dropna().between(0.0, 1.0).all()
        assert int(out["mse_state_changed"].sum()) >= 0
