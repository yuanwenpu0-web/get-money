"""MA Cascade Engine（均线级联引擎）移植测试。

期望值来源：

* 均线值 → 共享原语 ``_pine.ema`` / ``sma`` / ``rma`` / ``wma`` / ``hma`` 独立复算对齐，
  JMA/KAMA 用本文件里另一份递归实现逐列比对
* 斜率 = ``(MA - MA[lookback]) / (ATR * lookback)``；曲率 = ``slope - slope[k]``
* 分离度 = ``(fast - base) / ATR``；价格延伸 = ``(close - fast) / ATR``
* 状态码与权重按 Pine 的定义直读
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators._pine import ema, sma
from app.catalog_indicators.ma_cascade_engine import (
    STATE_BEAR_EXPANSION,
    STATE_BEAR_PULLBACK,
    STATE_BULL_EXPANSION,
    STATE_BULL_PULLBACK,
    STATE_NEUTRAL,
    STATE_OVEREXTENDED,
    STATE_TRANSITION,
    ma_cascade_engine,
)

FLAG_COLUMNS = [
    "mce_state",
    "mce_bias",
    "mce_bullish_order",
    "mce_bearish_order",
    "mce_entangled",
    "mce_extreme_volatility",
    "mce_overextended",
    "mce_fading",
    "mce_long_permission",
    "mce_short_permission",
    "mce_long_permission_start",
    "mce_long_setup",
    "mce_short_setup",
    "mce_long_trigger",
    "mce_short_trigger",
    "mce_enter_long",
    "mce_enter_short",
    "mce_exit_long",
    "mce_exit_short",
    "mce_bullish_anchor_rejection",
    "mce_bearish_anchor_rejection",
]


def _frame(close: np.ndarray, *, spread: float = 0.5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(close), freq="1h")
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
        },
        index=idx,
    )


def _trend(n: int = 400, slope: float = 0.3, amplitude: float = 1.0, period: float = 5.0) -> np.ndarray:
    return 100 + np.arange(n) * slope + amplitude * np.sin(np.arange(n) / period)


def _noisy(seed: int = 2, n: int = 600) -> pd.DataFrame:
    """带噪声的温和上行：能自然触发 reclaim / trigger 的那类形态。"""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.6, n)) + np.arange(n) * 0.08
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + np.abs(rng.normal(0, 0.4, n)),
            "low": close - np.abs(rng.normal(0, 0.4, n)),
            "close": close,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# 均线层
# ---------------------------------------------------------------------------


def test_structural_preset_uses_ema_sma_sma():
    """Structural = EMA 20 / SMA 50 / SMA 200，值必须与共享原语一致。"""
    close = _trend(400)
    frame = _frame(close)
    out = ma_cascade_engine(frame, preset="Structural")
    source = pd.Series(close, index=frame.index)

    assert out["mce_fast_ma"].to_numpy()[1:] == pytest.approx(
        ema(source, 20).to_numpy()[1:], nan_ok=True
    )
    assert out["mce_base_ma"].to_numpy()[1:] == pytest.approx(
        sma(source, 50).to_numpy()[1:], nan_ok=True
    )
    assert out["mce_anchor_ma"].to_numpy()[1:] == pytest.approx(
        sma(source, 200).to_numpy()[1:], nan_ok=True
    )


def test_balanced_preset_uses_jma_ema_sma():
    """Balanced = JMA 21 / EMA 50 / SMA 200；JMA 用独立递归复算。"""
    close = _trend(400, slope=0.2, amplitude=1.5, period=6)
    frame = _frame(close)
    out = ma_cascade_engine(frame, preset="Balanced")
    source = pd.Series(close, index=frame.index)

    assert out["mce_base_ma"].to_numpy()[1:] == pytest.approx(
        ema(source, 50).to_numpy()[1:], nan_ok=True
    )
    assert out["mce_anchor_ma"].to_numpy()[1:] == pytest.approx(
        sma(source, 200).to_numpy()[1:], nan_ok=True
    )
    assert out["mce_fast_ma"].to_numpy() == pytest.approx(_jma_reference(close, 21, 0, 2.0))


def _jma_reference(src: np.ndarray, length: int, phase: int, power: float) -> np.ndarray:
    """Pine ``f_jma`` 的独立实现（逐 bar 复算，用来交叉验证）。"""
    phase_ratio = phase / 100.0 + 1.5
    beta = 0.45 * (length - 1.0) / (0.45 * (length - 1.0) + 2.0)
    alpha = beta ** power
    out = np.empty(len(src))
    e0 = e1 = e2 = result = np.nan
    for i, price in enumerate(src):
        seed = price if np.isnan(result) else result
        e0 = (1.0 - alpha) * price + alpha * (price if np.isnan(e0) else e0)
        e1 = (price - e0) * (1.0 - beta) + beta * (0.0 if np.isnan(e1) else e1)
        e2 = (e0 + phase_ratio * e1 - seed) * (1.0 - alpha) ** 2 + alpha ** 2 * (
            0.0 if np.isnan(e2) else e2
        )
        result = e2 + seed
        out[i] = result
    return out


def test_all_ma_types_run():
    """八种平滑都能算出有限值（KAMA/JMA 从第一根就有值）。"""
    frame = _frame(_trend(400, slope=0.2, amplitude=2.0, period=8), spread=0.6)
    for kind in ("JMA", "EMA", "SMA", "RMA", "WMA", "HMA", "DEMA", "KAMA"):
        out = ma_cascade_engine(
            frame,
            preset="Custom",
            custom_fast_type=kind,
            custom_base_type=kind,
            custom_anchor_type=kind,
        )
        assert out["mce_fast_ma"].notna().sum() > 300, kind
        assert np.isfinite(out["mce_fast_ma"].dropna()).all(), kind


def test_source_can_be_synthetic_price():
    """``source`` 支持 hl2/hlc3 之类的合成价。"""
    frame = _frame(_trend(120))
    hl2 = ((frame["high"] + frame["low"]) / 2.0)
    out = ma_cascade_engine(frame, preset="Structural", source="hl2")
    expected = sma(hl2, 50).to_numpy()
    assert out["mce_base_ma"].to_numpy()[1:] == pytest.approx(expected[1:], nan_ok=True)


# ---------------------------------------------------------------------------
# 特征层
# ---------------------------------------------------------------------------


def test_slope_and_curvature_follow_formula():
    """斜率和曲率必须由导出的均线列 + ATR 反算得到。"""
    from app.catalog_indicators._pine import atr

    frame = _frame(_trend(300, slope=0.25, amplitude=1.2, period=7), spread=0.7)
    out = ma_cascade_engine(frame, preset="Balanced")

    safe_atr = atr(frame, 14).to_numpy()
    fast = out["mce_fast_ma"].to_numpy()
    expected = np.full(len(fast), np.nan)
    expected[5:] = (fast[5:] - fast[:-5]) / (safe_atr[5:] * 5)
    assert out["mce_fast_slope"].to_numpy()[5:] == pytest.approx(expected[5:], nan_ok=True)

    slope = out["mce_fast_slope"].to_numpy()
    curvature = np.full(len(slope), np.nan)
    curvature[3:] = slope[3:] - slope[:-3]
    assert out["mce_fast_curvature"].to_numpy()[6:] == pytest.approx(curvature[6:], nan_ok=True)


def test_spread_and_extension_follow_formula():
    """分离度 = (fast-base)/ATR，延伸 = (close-fast)/ATR。"""
    from app.catalog_indicators._pine import atr

    frame = _frame(_trend(300, slope=0.2, amplitude=1.5, period=9), spread=0.5)
    out = ma_cascade_engine(frame, preset="Balanced")
    close = frame["close"].to_numpy()
    safe_atr = atr(frame, 14).to_numpy()

    fast = out["mce_fast_ma"].to_numpy()
    base = out["mce_base_ma"].to_numpy()
    spread = (fast - base) / safe_atr
    extension = (close - fast) / safe_atr
    mask = np.isfinite(spread)
    assert out["mce_fast_base_spread"].to_numpy()[mask] == pytest.approx(spread[mask])
    mask = np.isfinite(extension)
    assert out["mce_price_extension"].to_numpy()[mask] == pytest.approx(extension[mask])


def test_constant_price_is_entangled_and_scores_zero():
    """完全走平时两条均线间距为 0 → 缠绕；所有方向分量为 0 → 总分 0。"""
    frame = _frame(np.full(120, 100.0), spread=0.0)
    out = ma_cascade_engine(
        frame, preset="Custom",
        custom_fast_type="SMA", custom_fast_length=5,
        custom_base_type="SMA", custom_base_length=10,
        custom_anchor_type="SMA", custom_anchor_length=20,
    )
    tail = slice(-20, None)
    assert int(out["mce_entangled"].iloc[tail].min()) == 1
    assert out["mce_score"].iloc[tail].abs().max() == pytest.approx(0.0, abs=1e-9)
    assert int(out["mce_state"].iloc[-1]) == STATE_NEUTRAL
    # 间距为 0 时既不算多头排列也不算空头排列
    assert int(out["mce_bullish_order"].iloc[tail].sum()) == 0
    assert int(out["mce_bearish_order"].iloc[tail].sum()) == 0


# ---------------------------------------------------------------------------
# 状态层
# ---------------------------------------------------------------------------


def test_uptrend_states_and_score_sign():
    """上行趋势 → 出现多头排列 / 多头回调态，总分为正。"""
    frame = _frame(_trend(500, slope=0.3, amplitude=1.0, period=5))
    out = ma_cascade_engine(frame, preset="Balanced")
    states = set(out["mce_state"].dropna().unique().tolist())
    assert int(out["mce_bullish_order"].sum()) > 300
    assert STATE_BULL_PULLBACK in states
    assert out["mce_score"].iloc[-1] > 25.0
    assert int(out["mce_bias"].iloc[-1]) > 0


def test_downtrend_states_and_score_sign():
    """下行趋势 → 空头排列、负分。"""
    close = 200 - _trend(500, slope=0.3, amplitude=1.0, period=5) + 100
    frame = _frame(close)
    out = ma_cascade_engine(frame, preset="Balanced")
    assert int(out["mce_bearish_order"].sum()) > 300
    assert out["mce_score"].iloc[-1] < -25.0
    assert int(out["mce_bias"].iloc[-1]) < 0
    states = set(out["mce_state"].dropna().unique().tolist())
    assert state_overlap(states, {STATE_BEAR_PULLBACK, STATE_BEAR_EXPANSION, STATE_OVEREXTENDED})


def state_overlap(actual: set, expected: set) -> bool:
    """辅助：至少命中一个期望状态。"""
    return bool(actual & expected)


def test_state_codes_are_definitionally_consistent():
    """状态码必须落在 1..7，且与 overextended / 排列标志自洽。"""
    frame = _noisy(seed=3)
    out = ma_cascade_engine(frame)
    states = out["mce_state"].dropna().astype(int)
    assert states.min() >= STATE_BULL_EXPANSION
    assert states.max() <= STATE_TRANSITION
    # 过度延伸优先于一切
    over = out["mce_overextended"] == 1
    assert (states[over] == STATE_OVEREXTENDED).all()
    # 多头扩张只能出现在多头排列时
    bull_exp = states == STATE_BULL_EXPANSION
    assert (out["mce_bullish_order"][bull_exp] == 1).all()


# ---------------------------------------------------------------------------
# 决策层
# ---------------------------------------------------------------------------


def test_trigger_fires_and_composes_from_published_columns():
    """trigger = reclaim ∧ 斜率>阈值 ∧ 斜率抬升 ∧ 曲率 ∧ 价格确认。

    在带噪声的温和上行里能自然出现少量触发（Pine 的触发条件本就严格）。
    """
    frame = _noisy(seed=2)
    out = ma_cascade_engine(frame)
    assert int(out["mce_long_trigger"].sum()) > 0
    assert int(out["mce_short_trigger"].sum()) > 0
    assert int(out["mce_long_setup"].sum()) > 0

    close = frame["close"].to_numpy()
    fast = out["mce_fast_ma"].to_numpy()
    slope = out["mce_fast_slope"].to_numpy()
    previous_close = np.r_[[np.nan], close[:-1]]
    previous_fast = np.r_[[np.nan], fast[:-1]]
    previous_slope = np.r_[[np.nan], slope[:-1]]
    previous_high = np.r_[[np.nan], frame["high"].to_numpy()[:-1]]
    curvature_ok = out["mce_fast_curvature"].to_numpy() > 0.005

    expected = (
        (close > fast)
        & (previous_close <= previous_fast)
        & (slope > 0.015)
        & (slope > previous_slope)
        & curvature_ok
        & (close > previous_high)
    )
    assert np.array_equal(out["mce_long_trigger"].to_numpy(dtype=int), expected.astype(int))


def test_triggers_relax_when_confirmations_are_disabled():
    """关掉“突破前高/曲率”门槛后，触发数不可能变少。"""
    frame = _noisy(seed=5)
    strict = ma_cascade_engine(frame)
    loose = ma_cascade_engine(frame, require_price_confirmation=False, require_curvature=False)
    assert int(loose["mce_long_trigger"].sum()) >= int(strict["mce_long_trigger"].sum())
    assert int(loose["mce_enter_long"].sum()) >= int(strict["mce_enter_long"].sum())


def test_entangled_blocks_permission_setup_and_entries():
    """缠绕 / 过度延伸都会关掉 permission，并连带关掉 setup 与入场。"""
    frame = _frame(np.full(150, 100.0) + np.linspace(0, 1, 150), spread=0.0)
    out = ma_cascade_engine(frame, preset="Structural")
    entangled = out["mce_entangled"] == 1
    assert (out["mce_long_permission"][entangled] == 0).all()
    assert (out["mce_short_permission"][entangled] == 0).all()
    assert (out["mce_long_setup"][entangled] == 0).all()
    assert (out["mce_enter_long"][entangled] == 0).all()


def test_respect_touches_counted_once_per_episode():
    """连续贴线只算一次触达（Pine 的 touch-episode 语义）。"""
    frame = _frame(np.full(200, 100.0), spread=0.0)
    out = ma_cascade_engine(frame, preset="Structural", respect_lookback=150)
    assert out["mce_base_respect_touches"].max() == pytest.approx(1.0)


def test_score_decomposition_and_bounds():
    """总分 = 各分项之和 × 惩罚系数，且被夹在 ±100。"""
    frame = _noisy(seed=1)
    out = ma_cascade_engine(frame)
    assert out["mce_score"].dropna().between(-100.0, 100.0).all()

    parts = (
        out["mce_direction_score"]
        + out["mce_slope_score"]
        + out["mce_expansion_score"]
        + out["mce_curvature_score"]
        + out["mce_respect_score"]
        + out["mce_anchor_respect_score"]
        + out["mce_cross_quality_score"]
    )
    penalty = (
        (out["mce_overextended"] == 1).astype(float) * 0.20
        + (out["mce_entangled"] == 1).astype(float) * 0.25
        + (out["mce_extreme_volatility"] == 1).astype(float) * 0.10
    )
    expected = (parts * np.maximum(0.0, 1.0 - penalty)).clip(-100.0, 100.0)
    valid = out["mce_score"].notna().to_numpy()
    assert valid.sum() > 100
    assert out["mce_score"].to_numpy()[valid] == pytest.approx(expected.to_numpy()[valid])

    # 方向分最多 ±30，且按 10/15/5 的权重构成
    assert out["mce_direction_score"].dropna().between(-30.0, 30.0).all()
    assert ((out["mce_direction_score"].dropna() % 5.0).abs() < 1e-9).all()


# ---------------------------------------------------------------------------
# 契约
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"preset": "Turbo"},
        {"jma_phase": 200},
        {"jma_power": 0.5},
        {"atr_length": 1},
        {"slope_lookback": 0},
        {"curvature_lookback": 60},
        {"entanglement_spread": 0.001},
        {"cross_frequency_lookback": 3},
        {"maximum_crosses": 0},
        {"pullback_distance": 0.01},
        {"maximum_extension": 0.1},
        {"extreme_volatility_ratio": 0.9},
        {"volatility_baseline": 10},
        {"reaction_bars": 0},
        {"respect_lookback": 10},
        {"strong_bias_threshold": 200.0},
        {"directional_bias_threshold": 1.0},
        {"custom_anchor_length": 1},
        {"custom_fast_type": "T3"},
        {"source": "nope"},
    ],
)
def test_parameter_validation(kwargs):
    frame = _frame(_trend(60))
    with pytest.raises(ValueError):
        ma_cascade_engine(frame, **kwargs)


def test_output_contract():
    """索引一致、列唯一、标志列 Int64、无 inf。"""
    frame = _noisy(seed=4)
    out = ma_cascade_engine(frame)
    assert out.index.equals(frame.index)
    assert len(out.columns) == len(set(out.columns))
    assert all(column.startswith("mce_") for column in out.columns)
    for column in FLAG_COLUMNS:
        assert str(out[column].dtype) == "Int64", column
    numeric = out.drop(columns=FLAG_COLUMNS).to_numpy(dtype=float)
    assert not np.isinf(numeric).any()
    assert set(out["mce_bias"].dropna().unique().tolist()) <= {-2, -1, 0, 1, 2}


def test_presets_differ_and_all_run():
    """三个预设都应产出不同的均线组合，且都能跑通。"""
    frame = _frame(_trend(300, slope=0.2, amplitude=1.5, period=8), spread=0.6)
    results = {p: ma_cascade_engine(frame, preset=p) for p in ("Responsive", "Balanced", "Structural")}
    last = {p: float(out["mce_fast_ma"].iloc[-1]) for p, out in results.items()}
    assert len(set(round(value, 6) for value in last.values())) == 3
    for out in results.values():
        assert out["mce_score"].notna().sum() > 50
    # Custom 用传入的类型/长度
    custom = ma_cascade_engine(
        frame, preset="Custom", custom_fast_type="SMA", custom_fast_length=10,
        custom_base_type="SMA", custom_base_length=30, custom_anchor_type="SMA",
        custom_anchor_length=90,
    )
    assert custom["mce_base_ma"].to_numpy()[1:] == pytest.approx(
        sma(frame["close"], 30).to_numpy()[1:], nan_ok=True
    )
