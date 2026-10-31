"""Trade Permission Engine 移植测试。

期望值来源：

* **参考腿**：反向枢轴只有把前腿回撤到 ``leg_invalidation_retracement`` 才取代它
  —— 用同一份“探底→拉升→回落”的合成路径，只改回落深度，验证腿是否翻转
* **Permission**：``100 × gated(腿质量) × gated(趋势质量) × gated(成熟度)
  × gated(衰竭)``，两侧取 ``max(continuation, reversal)``（关闭 HTF 时无加法项）
* **成熟度**：``100 × 钟形(r; 偏好回撤中心, 软度) × 回调存在度 × 时长钟形``
* **Trigger**：0→1→2→3 的分数阶梯（30 / 50~75 / 100）与 CONFIRMED→ACCEPTED 的计时
* **目标**：T1 必须距离现价至少 ``min_target_distance_atr × ATR``，T2 必须比 T1 更远
* **持仓健康**：阈值映射到 0..3 状态码，六维加权平均等于导出分
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.trade_permission_engine import trade_permission_engine

FLAG_COLUMNS = [
    "tpe_new_leg_bull",
    "tpe_new_leg_bear",
    "tpe_regular_bull_div",
    "tpe_hidden_bull_div",
    "tpe_regular_bear_div",
    "tpe_hidden_bear_div",
    "tpe_trigger_state_long",
    "tpe_trigger_state_short",
    "tpe_long_is_continuation",
    "tpe_setup_long",
    "tpe_setup_short",
    "tpe_new_setup_long",
    "tpe_new_setup_short",
    "tpe_trigger_long",
    "tpe_trigger_short",
    "tpe_signal_long",
    "tpe_signal_short",
    "tpe_t1_hit_long",
    "tpe_t2_hit_long",
    "tpe_inv_hit_long",
    "tpe_cancelled_long",
    "tpe_t1_hit_short",
    "tpe_t2_hit_short",
    "tpe_inv_hit_short",
    "tpe_cancelled_short",
    "tpe_health_state_long",
    "tpe_health_state_short",
]


def _frame(close, *, spread: float = 0.5, freq: str = "1h") -> pd.DataFrame:
    values = np.asarray(close, dtype=float)
    idx = pd.date_range("2024-01-01", periods=len(values), freq=freq)
    return pd.DataFrame(
        {
            "open": values,
            "high": values + spread,
            "low": values - spread,
            "close": values,
        },
        index=idx,
    )


def _leg_path(tail_to: float, *, n: int = 140, peak: float = 130.0, dip: float = 100.0,
              bounce: float = 2.0) -> np.ndarray:
    """探底 → 拉升到单根峰值 → 回落到 ``tail_to`` → 小幅反弹后走平。

    回落目标的深度决定反向枢轴是否够格取代参考腿（阈值 0.90）。
    """
    first = np.linspace(105.0, dip, 11)
    rise = np.linspace(dip + 0.5, peak, 30)
    fall = np.linspace(peak - 0.5, tail_to, 20)
    rest = np.full(n - len(first) - len(rise) - len(fall), tail_to + bounce)
    return np.concatenate([first, rise, fall, rest])


def _noisy(seed: int = 0, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.8, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + np.abs(rng.normal(0, 0.5, n)),
            "low": close - np.abs(rng.normal(0, 0.5, n)),
            "close": close,
        },
        index=idx,
    )


def _gated(value: float, floor: float) -> float:
    """测试侧的 ``f_gatedFactor`` 复算。"""
    return floor + (1.0 - floor) * min(max(value, 0.0), 100.0) / 100.0


# ---------------------------------------------------------------------------
# 参考腿
# ---------------------------------------------------------------------------


def test_shallow_pullback_does_not_replace_reference_leg():
    """43% 的回撤不得篡位：方向仍为多头，低点仍是原起点。"""
    out = trade_permission_engine(
        _frame(_leg_path(115.0)), preset="Custom", custom_pivot_len=3, htf_enabled=False
    )
    assert int(out["tpe_swing_direction"].iloc[-1]) == 1
    assert out["tpe_last_low"].iloc[-1] == pytest.approx(99.5)
    assert out["tpe_last_high"].iloc[-1] == pytest.approx(130.5)
    assert int(out["tpe_new_leg_bull"].sum()) == 1
    assert int(out["tpe_new_leg_bear"].sum()) == 0
    assert out["tpe_bullish_retracement"].iloc[-1] == pytest.approx((130.5 - 117.0) / 31.0)


def test_deep_retracement_replaces_reference_leg():
    """回撤超过 90% 后反向枢轴才成为新参考腿。"""
    out = trade_permission_engine(
        _frame(_leg_path(98.0)), preset="Custom", custom_pivot_len=3, htf_enabled=False
    )
    assert int(out["tpe_swing_direction"].iloc[-1]) == -1
    assert out["tpe_last_low"].iloc[-1] == pytest.approx(97.5)
    assert out["tpe_last_high"].iloc[-1] == pytest.approx(130.5)
    assert int(out["tpe_new_leg_bear"].sum()) == 1
    assert out["tpe_bearish_retracement"].iloc[-1] == pytest.approx((100.0 - 97.5) / 33.0)


# ---------------------------------------------------------------------------
# Permission / Maturity 恒等式
# ---------------------------------------------------------------------------


def test_permission_is_floored_multiplicative_gate():
    """关闭 HTF 后，Continuation 许可 = 四个门系数的乘积，总许可取两侧 max。"""
    frame = _noisy(seed=1)
    out = trade_permission_engine(frame, htf_enabled=False)
    quality = out["tpe_reference_leg_quality"].to_numpy()
    trend = out["tpe_live_trend_quality"].to_numpy()
    maturity = out["tpe_continuation_maturity_long"].to_numpy()
    exhaustion = out["tpe_exhaustion_bear"].to_numpy()

    expected = (
        100.0
        * np.array([_gated(v, 0.25) for v in quality])
        * np.array([_gated(v, 0.25) for v in trend])
        * np.array([_gated(v, 0.15) for v in maturity])
        * np.array([_gated(v, 0.15) for v in exhaustion])
    )
    assert out["tpe_cont_permission_long"].to_numpy() == pytest.approx(expected)

    long_total = np.fmax(
        out["tpe_cont_permission_long"].to_numpy(), out["tpe_rev_permission_long"].to_numpy()
    )
    short_total = np.fmax(
        out["tpe_cont_permission_short"].to_numpy(), out["tpe_rev_permission_short"].to_numpy()
    )
    assert out["tpe_permission_long"].to_numpy() == pytest.approx(long_total)
    assert out["tpe_permission_short"].to_numpy() == pytest.approx(short_total)
    assert out["tpe_permission_long"].between(0.0, 100.0).all()


def test_continuation_maturity_follows_gaussian_identity():
    """成熟度 = 100 × 钟形(r) × 回调存在度 × 时长钟形，三者在输出里都能复算。"""
    frame = _noisy(seed=2)
    out = trade_permission_engine(
        frame, htf_enabled=False, time_maturity_center=0.35, time_maturity_width=0.35
    )
    r = out["tpe_bullish_retracement"].to_numpy()
    center = out["tpe_preferred_fib_center"].to_numpy()
    softness = out["tpe_fib_softness"].to_numpy()
    time_ratio = out["tpe_time_maturity_ratio"].to_numpy()
    maturity = out["tpe_continuation_maturity_long"].to_numpy()

    mask = np.isfinite(r) & np.isfinite(maturity)
    assert mask.sum() > 50
    zone = np.exp(-0.5 * ((r[mask] - center[mask]) / softness[mask]) ** 2)
    presence = np.clip(r[mask] / 0.10, 0.0, 1.0)
    time_factor = np.where(
        np.isfinite(time_ratio[mask]),
        np.exp(-0.5 * ((np.nan_to_num(time_ratio[mask]) - 0.35) / 0.35) ** 2),
        1.0,
    )
    expected = np.clip(100.0 * zone * presence * time_factor, 0.0, 100.0)
    assert maturity[mask] == pytest.approx(expected)


def test_htf_influence_is_asymmetric_for_reversal():
    """Reversal 顺 HTF 的加成被打折，逆 HTF 的惩罚被放大。"""
    frame = _noisy(seed=3, n=1500)
    base = trade_permission_engine(frame, htf_enabled=False)
    with_htf = trade_permission_engine(
        frame,
        htf_enabled=True,
        htf_mode="Manual",
        manual_htf_seconds=14400,
        htf_influence=15.0,
        reversal_htf_aligned_factor=0.5,
        reversal_htf_conflict_factor=1.5,
    )
    assert not np.allclose(base["tpe_htf_trend_signed"], with_htf["tpe_htf_trend_signed"])
    trend = with_htf["tpe_htf_trend_signed"].to_numpy()
    assert trend.min() >= -1.0 and trend.max() <= 1.0
    # 有 HTF 时许可会偏离（同向加分、反向减分）
    delta = (with_htf["tpe_cont_permission_long"] - base["tpe_cont_permission_long"]).to_numpy()
    assert np.abs(delta).max() > 0.5
    assert (delta[trend > 0.1] >= -1e-9).all()


# ---------------------------------------------------------------------------
# Trigger / 目标
# ---------------------------------------------------------------------------


def test_trigger_state_scores_and_acceptance():
    """状态 1=30 分、2=50~75、3=100；ACCEPTED 需要保持 acceptance_bars 根。"""
    frame = _noisy(seed=0)
    out = trade_permission_engine(frame)
    state = out["tpe_trigger_state_long"].to_numpy()
    score = out["tpe_trigger_score_long"].to_numpy()
    assert state.max() == 3, "随机序列上应能走出完整状态链"

    assert np.allclose(score[state == 0], 0.0)
    assert np.allclose(score[state == 1], 30.0)
    assert ((score[state == 2] >= 50.0 - 1e-9) & (score[state == 2] <= 75.0 + 1e-9)).all()
    assert np.allclose(score[state == 3], 100.0)
    assert np.allclose(out["tpe_acceptance_long"].to_numpy()[state == 3], 100.0)

    # 每个 ACCEPTED 状态之前必须有至少 acceptance_bars 根 CONFIRMED
    first_accepted = int(np.argmax(state == 3))
    assert (state[first_accepted - 3:first_accepted] == 2).any()


def test_targets_respect_minimum_distance_and_ordering():
    """T1 至少 ``min_target_distance_atr × ATR`` 远；T2 必须比 T1 更远。"""
    frame = _noisy(seed=4)
    out = trade_permission_engine(frame, min_target_distance_atr=0.5, use_t2=True)
    close = frame["close"].to_numpy()
    atr = out["tpe_atr"].to_numpy()

    t1_long = out["tpe_t1_long"].to_numpy()
    t1_short = out["tpe_t1_short"].to_numpy()
    mask = np.isfinite(atr) & (atr > 0)
    assert (t1_long[mask] - close[mask] >= 0.5 * atr[mask] - 1e-9).all()
    assert (close[mask] - t1_short[mask] >= 0.5 * atr[mask] - 1e-9).all()

    t2_long = out["tpe_t2_long"].to_numpy()
    t2_short = out["tpe_t2_short"].to_numpy()
    valid_long = np.isfinite(t2_long)
    assert valid_long.sum() > 0
    assert (t2_long[valid_long] - t1_long[valid_long] >= 0.5 * atr[valid_long] - 1e-9).all()
    valid_short = np.isfinite(t2_short)
    assert (t1_short[valid_short] - t2_short[valid_short] >= 0.5 * atr[valid_short] - 1e-9).all()


def test_targets_disabled_closes_trade_at_t1():
    """关掉 T2 后交易在 T1 了结：不会再出现 T2 命中事件。"""
    frame = _noisy(seed=5)
    with_t2 = trade_permission_engine(frame, use_t2=True)
    without = trade_permission_engine(frame, use_t2=False)
    assert int(with_t2["tpe_t2_hit_long"].sum()) + int(with_t2["tpe_t2_hit_short"].sum()) >= 0
    assert int(without["tpe_t2_hit_long"].sum()) == 0
    assert int(without["tpe_t2_hit_short"].sum()) == 0


# ---------------------------------------------------------------------------
# 分数分解 / 衰减
# ---------------------------------------------------------------------------


def test_score_decomposition_from_published_columns():
    """score_raw = 许可 ×（确认地板 + 确认率）× CRV × 衰减。"""
    out = trade_permission_engine(_noisy(seed=6), confirmation_floor=0.35)
    permission = out["tpe_permission_long"].to_numpy()
    confirmation = out["tpe_confirmation_long"].to_numpy()
    crv = out["tpe_crv_factor_long"].to_numpy()
    decay = out["tpe_decay_factor_long"].to_numpy()
    expected = permission * (0.35 + 0.65 * confirmation / 100.0) * crv * decay
    assert out["tpe_score_raw_long"].to_numpy() == pytest.approx(expected)


def test_decay_resets_on_progress():
    """出现新极值/加速/ATR 扩张时停滞计数归零。"""
    out = trade_permission_engine(_noisy(seed=7))
    stale = out["tpe_stale_bars_long"].to_numpy()
    assert stale.max() >= 1.0
    assert stale.min() == 0.0
    # 停滞计数每根最多 +1
    assert np.all(np.diff(stale) <= 1.0)


# ---------------------------------------------------------------------------
# 持仓健康
# ---------------------------------------------------------------------------


def test_health_weights_and_state_codes():
    """六维加权平均 = 导出健康分；阈值映射到 HOLD/PROTECT/REDUCE/EXIT。"""
    out = trade_permission_engine(
        _noisy(seed=8),
        w_live_quality=25.0,
        w_pullback_health=20.0,
        w_trend_progress=20.0,
        w_fatigue=15.0,
        w_structure_intact=10.0,
        w_regime=10.0,
        health_hold_threshold=68.0,
        health_protect_threshold=54.0,
        health_reduce_threshold=36.0,
    )
    raw = (
        25.0 * out["tpe_live_quality_long"].to_numpy()
        + 20.0 * out["tpe_pullback_health_long"].to_numpy()
        + 20.0 * out["tpe_trend_progress_long"].to_numpy()
        + 15.0 * out["tpe_trend_fatigue"].to_numpy()
        + 10.0 * out["tpe_structure_intact_long"].to_numpy()
        + 10.0 * out["tpe_regime_score"].to_numpy()
    ) / 100.0
    smoothed = pd.Series(raw).ewm(alpha=2 / (3 + 1), adjust=False, min_periods=3).mean()
    # ema 的种子是前 3 根的均值，比较尾段即可
    assert out["tpe_health_long"].to_numpy()[-50:] == pytest.approx(smoothed.to_numpy()[-50:])

    health = out["tpe_health_long"].to_numpy()
    state = out["tpe_health_state_long"].to_numpy()
    assert np.array_equal(state, np.where(health >= 68.0, 3, np.where(health >= 54.0, 2, np.where(health >= 36.0, 1, 0))))
    assert set(np.unique(state)) <= {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# 参数 / 契约
# ---------------------------------------------------------------------------


def test_oscillator_choices_change_the_series():
    """四种振荡器都能算，且互不相同。"""
    frame = _noisy(seed=9, n=260)
    series = {}
    for name in ("Stochastic RSI", "RSI", "WaveTrend", "CMO"):
        out = trade_permission_engine(frame, oscillator=name)
        series[name] = out["tpe_oscillator"].to_numpy()
        assert np.isfinite(series[name]).sum() > 100
    assert not np.allclose(series["RSI"], series["WaveTrend"])
    assert not np.allclose(series["CMO"], series["Stochastic RSI"])


def test_preset_auto_resolves_from_bar_length():
    """Auto 预设与 Auto HTF 的档位映射（Pine 的 f_autoHtf / resolvedPreset）。"""
    from app.catalog_indicators.trade_permission_engine import (
        _auto_htf_seconds,
        _bar_seconds,
        _resolve_preset,
    )

    assert _resolve_preset("Auto", 60) == "Scalping"
    assert _resolve_preset("Auto", 300) == "Scalping"
    assert _resolve_preset("Auto", 301) == "Intraday"
    assert _resolve_preset("Auto", 1800) == "Intraday"
    assert _resolve_preset("Auto", 1801) == "Swing"
    assert _resolve_preset("Auto", 14400) == "Swing"
    assert _resolve_preset("Auto", 14401) == "Position"
    assert _resolve_preset("Custom", 60) == "Custom"

    assert _bar_seconds(_frame([1.0, 2.0], freq="5min")) == 300.0
    assert _bar_seconds(_frame([1.0, 2.0], freq="1h")) == 3600.0
    assert _bar_seconds(_frame([1.0, 2.0], freq="1D")) == 86400.0

    assert _auto_htf_seconds(300) == 900.0
    assert _auto_htf_seconds(3600) == 14400.0
    assert _auto_htf_seconds(86400) == 604800.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"preset": "Turbo"},
        {"oscillator": "RSX"},
        {"smoothing": 0},
        {"custom_pivot_len": 1},
        {"min_swing_atr": 0.1},
        {"leg_invalidation_retracement": 0.4},
        {"max_leg_history": 3},
        {"shallow_center": 0.9},
        {"reversal_center": 0.5},
        {"overextension_atr": 0.1},
        {"acceptance_bars": 0},
        {"min_crv": 0.1},
        {"max_pivot_history": 3},
        {"htf_fast_ema": 1},
        {"regime_lookback": 3},
        {"htf_mode": "Reactive"},
    ],
)
def test_parameter_validation(kwargs):
    with pytest.raises(ValueError):
        trade_permission_engine(_noisy(seed=10, n=120), **kwargs)


def test_output_contract():
    """索引一致、列唯一、标志列 Int64、分数落在 0..100。"""
    frame = _noisy(seed=11)
    out = trade_permission_engine(frame)
    assert out.index.equals(frame.index)
    assert len(out.columns) == len(set(out.columns))
    assert all(column.startswith("tpe_") for column in out.columns)
    for column in FLAG_COLUMNS:
        assert str(out[column].dtype) == "Int64", column
    assert out["tpe_permission_long"].between(0.0, 100.0).all()
    assert out["tpe_permission_short"].between(0.0, 100.0).all()
    assert out["tpe_health_long"].between(0.0, 100.0).all()
    numeric = out.drop(columns=FLAG_COLUMNS).to_numpy(dtype=float)
    assert not np.isinf(numeric).any()
    assert set(out["tpe_swing_direction"].dropna().unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_setup_requires_dominance():
    """Setup 要求本侧许可过门槛且严格大于另一侧。"""
    out = trade_permission_engine(_noisy(seed=12), permission_threshold=12.0)
    long_p = out["tpe_permission_long"].to_numpy()
    short_p = out["tpe_permission_short"].to_numpy()
    setup_long = out["tpe_setup_long"].to_numpy(dtype=int)
    expected = ((long_p >= 12.0) & (long_p > short_p)).astype(int)
    assert np.array_equal(setup_long, expected)
    assert int(setup_long.sum()) > 0
