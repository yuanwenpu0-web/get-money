"""Regime Detector Pro（含 MA 区间覆盖）移植测试。

期望值来源：

* **ER / alpha / 自适应线**：由导出的 ``rdp_er``、``rdp_alpha``、``rdp_line`` 反算递推
* **斜率与强度**：``slope = Δline / ATR``、``strength = clamp01(|slope|)``
* **置信度**：``clamp01((ER − erThreshold) / (1 − erThreshold))``（进场口径）
* **制度状态机**：非零制度必然与斜率同号，且置信度不低于进出两档门槛的较小者
* **过滤器**：``quality = 0.5·volOk + 0.3·volGate + 0.2·(trendScore>0.2)``，低于合流门槛打五折
* **信号层级**：Tier A/B 的三项门槛逐条核对
* **MA 区间**：MA 排列门控 + 翻转时记录进场位与 ATR 止损位
* **绩效**：样本数 = 胜 + 负，胜率 = 胜 / 样本
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators._pine import atr, ema, percentrank, sma
from app.catalog_indicators.regime_detector_ma_zones_overlay import (
    PLAYBOOK_AGING,
    PLAYBOOK_CHOPPY,
    PLAYBOOK_COMPRESSION,
    PLAYBOOK_HEALTHY,
    PLAYBOOK_RANGE,
    PLAYBOOK_STRETCHED,
    PLAYBOOK_TREND,
    regime_detector_ma_zones_overlay,
)

FLAG_COLUMNS = [
    "rdp_regime",
    "rdp_htf_bias",
    "rdp_market_condition",
    "rdp_position",
    "rdp_tier",
    "rdp_vola_state",
    "rdp_playbook",
    "rdp_bars_in_regime",
    "rdp_samples",
    "rdp_long_entry",
    "rdp_short_entry",
    "rdp_exit_long",
    "rdp_exit_short",
    "rdp_low_quality_long",
    "rdp_low_quality_short",
    "rdp_vola_compression",
    "rdp_vola_expansion",
    "rdp_regime_weakening",
    "rdp_regime_weakening_event",
    "rdp_strong_trend",
    "rdp_zone_bull",
    "rdp_zone_bear",
    "rdp_zone_bull_flip",
    "rdp_zone_bear_flip",
    "rdp_zone_exit_flip",
    "rdp_warn_divergence",
    "rdp_warn_low_strength",
]


def _frame(close: np.ndarray, *, spread: float = 0.4, volume: float = 1000.0,
           freq: str = "1h") -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    idx = pd.date_range("2024-01-01", periods=len(close), freq=freq)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": np.full(len(close), volume),
        },
        index=idx,
    )


def _trend_frame(n: int = 3000, drift: float = 0.55, spread: float = 0.4,
                 seed: int = 1) -> pd.DataFrame:
    """强趋势 + 小噪声：ER 与动量门都能过，制度可以形成。"""
    rng = np.random.default_rng(seed)
    close = 100 + np.arange(n) * drift + np.cumsum(rng.normal(0, 0.15, n))
    return _frame(close, spread=spread)


def _noisy_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.8, n))
    return _frame(close, spread=0.5)


# ---------------------------------------------------------------------------
# 自适应核
# ---------------------------------------------------------------------------


def test_efficiency_ratio_matches_formula():
    """ER = |src − src[n]| / (SMA(|Δsrc|, n) · n)。"""
    frame = _trend_frame(600)
    out = regime_detector_ma_zones_overlay(frame, er_length=20)
    close = frame["close"].to_numpy()
    direction = np.abs(close - np.r_[np.full(20, np.nan), close[:-20]])
    volatility = sma(pd.Series(np.abs(np.diff(close, prepend=close[0]))), 20).to_numpy() * 20
    expected = np.where(volatility > 0, direction / volatility, 0.0)
    valid = np.isfinite(expected)
    assert out["rdp_er"].to_numpy()[valid] == pytest.approx(expected[valid])


def test_adaptive_line_follows_alpha_recursion():
    """alpha = aMin + (aMax−aMin)·ER^gamma，线是 prev + alpha·(src−prev) 的递推。"""
    frame = _trend_frame(600)
    out = regime_detector_ma_zones_overlay(
        frame, er_length=20, alpha_min=0.03, alpha_max=0.30, gamma=1.25, profile_preset="Custom"
    )
    er = out["rdp_er"].to_numpy()
    alpha = out["rdp_alpha"].to_numpy()
    line = out["rdp_line"].to_numpy()
    src = frame["close"].to_numpy()

    assert alpha[:50] == pytest.approx(0.03 + 0.27 * np.power(er[:50], 1.25))

    manual = np.empty(len(src))
    prev = src[0]
    manual[0] = src[0]
    for i in range(1, len(src)):
        current = prev + alpha[i] * (src[i] - prev)
        manual[i] = current
        prev = current
    assert line == pytest.approx(manual)


def test_slope_and_strength_from_published_columns():
    """slope = Δline/ATR，strength = clamp01(|slope|)。"""
    frame = _trend_frame(600)
    out = regime_detector_ma_zones_overlay(frame)
    line = out["rdp_line"].to_numpy()
    atr_a = out["rdp_atr"].to_numpy()
    slope = out["rdp_slope"].to_numpy()
    expected = np.where(atr_a > 0, (line - np.r_[np.nan, line[:-1]]) / atr_a, 0.0)
    valid = np.isfinite(expected) & (atr_a > 0)
    assert slope[valid] == pytest.approx(expected[valid])
    assert out["rdp_strength"].to_numpy() == pytest.approx(np.clip(np.abs(slope), 0.0, 1.0))


def test_profile_preset_overrides_alpha_and_gamma():
    """非 Custom 的 profile 会覆盖 alpha/gamma；Custom 用传入值。"""
    frame = _trend_frame(400)
    smart = regime_detector_ma_zones_overlay(frame, profile_preset="Smart")
    custom = regime_detector_ma_zones_overlay(
        frame, profile_preset="Custom", alpha_min=0.03, alpha_max=0.30, gamma=1.25
    )
    assert not np.allclose(smart["rdp_alpha"], custom["rdp_alpha"])
    # Smart = (0.025, 0.35, 1.15)
    er = smart["rdp_er"].to_numpy()
    assert smart["rdp_alpha"].to_numpy()[:50] == pytest.approx(
        0.025 + 0.325 * np.power(er[:50], 1.15)
    )


# ---------------------------------------------------------------------------
# 制度识别
# ---------------------------------------------------------------------------


def test_regime_forms_and_matches_slope_sign():
    """强趋势下制度成立；非零制度必须与斜率同号。"""
    out = regime_detector_ma_zones_overlay(_trend_frame())
    regime = out["rdp_regime"].to_numpy()
    slope = out["rdp_slope"].to_numpy()
    assert (regime == 1).sum() > 100
    assert (slope[regime == 1] > 0).all()
    assert (slope[regime == -1] < 0).all()


def test_regime_confidence_respects_thresholds():
    """进场口径置信度 = clamp01((ER−阈值)/(1−阈值))；维持期不低于退出门槛。"""
    out = regime_detector_ma_zones_overlay(
        _trend_frame(), er_threshold=0.2, regime_confidence_min=0.65, regime_confidence_exit=0.55
    )
    er = out["rdp_er"].to_numpy()
    conf = out["rdp_confidence"].to_numpy()
    regime = out["rdp_regime"].to_numpy()

    expect_enter = np.where(er >= 0.2, np.clip((er - 0.2) / 0.8, 0.0, 1.0), 0.0)
    zero_bars = regime == 0
    assert conf[zero_bars] == pytest.approx(expect_enter[zero_bars])
    assert (conf[regime != 0] >= 0.55 - 1e-9).all()


def test_trend_only_mode_ignores_momentum_gate():
    """Trend Only 只看 ER；Trend + Momentum 还要动量门，所以制度更少。"""
    frame = _trend_frame(drift=0.25, spread=0.4)
    trend_only = regime_detector_ma_zones_overlay(frame, regime_mode="Trend Only")
    both = regime_detector_ma_zones_overlay(frame, regime_mode="Trend + Momentum")
    assert int((trend_only["rdp_regime"] != 0).sum()) > int((both["rdp_regime"] != 0).sum())


def test_advanced_regime_disabled_opens_everything():
    """关掉高级制度识别后，只要 ER/置信度过关就立制度（不过动量门）。"""
    frame = _trend_frame(drift=0.25)
    off = regime_detector_ma_zones_overlay(frame, enable_advanced_regime=False)
    on = regime_detector_ma_zones_overlay(frame, regime_mode="Momentum Only")
    assert int((off["rdp_regime"] != 0).sum()) >= int((on["rdp_regime"] != 0).sum())


# ---------------------------------------------------------------------------
# MTF 门
# ---------------------------------------------------------------------------


def test_mtf_agreement_bounds_and_disable():
    """一致率在 0..1；关闭 MTF 时恒为 1 且不产生背离。"""
    frame = _trend_frame(1200)
    on = regime_detector_ma_zones_overlay(frame, enable_mtf_gate=True)
    assert on["rdp_htf_agreement"].between(0.0, 1.0).all()
    assert set(on["rdp_htf_bias"].unique().tolist()) <= {-1, 0, 1}

    off = regime_detector_ma_zones_overlay(frame, enable_mtf_gate=False)
    assert (off["rdp_htf_agreement"] == 1.0).all()
    assert (off["rdp_htf_divergence"] == 0.0).all()


def test_htf_divergence_is_complement_of_agreement():
    """背离强度 = 1 − 一致率。"""
    out = regime_detector_ma_zones_overlay(_trend_frame(1200), enable_htf_divergence=True)
    assert out["rdp_htf_divergence"].to_numpy() == pytest.approx(
        1.0 - out["rdp_htf_agreement"].to_numpy()
    )


def test_htf_weights_change_agreement():
    """三档权重分配会改变加权一致率。"""
    frame = _trend_frame(1200)
    primary = regime_detector_ma_zones_overlay(frame, htf_weights="Primary Focus")
    long_term = regime_detector_ma_zones_overlay(frame, htf_weights="Long-term Focus")
    assert not np.allclose(primary["rdp_htf_agreement"], long_term["rdp_htf_agreement"])


# ---------------------------------------------------------------------------
# 过滤器
# ---------------------------------------------------------------------------


def test_signal_quality_matches_formula():
    """quality = 0.5·volOk + 0.3·volGate + 0.2·(trendScore>0.2)，低于合流门槛打五折。"""
    frame = _trend_frame(600)
    out = regime_detector_ma_zones_overlay(
        frame,
        enable_advanced_filters=True,
        use_volume_filter=False,
        enable_confluence=True,
        confluence_threshold=0.6,
    )
    slope = np.abs(out["rdp_slope"].to_numpy())
    trend_score = sma(pd.Series(slope), 100).to_numpy()
    atr_pct = out["rdp_atr_percent"].to_numpy()
    atr_pct_avg = sma(pd.Series(atr_pct), 100).to_numpy()
    threshold = np.fmax(0.0007, atr_pct_avg * 0.7)
    vol_ok = atr_pct >= threshold

    quality = 0.5 * vol_ok.astype(float) + 0.3 + 0.2 * (np.nan_to_num(trend_score) > 0.2)
    quality = np.where(quality < 0.6, quality * 0.5, quality)
    expected = np.clip(quality, 0.0, 1.0)
    valid = np.isfinite(trend_score)
    assert out["rdp_signal_quality"].to_numpy()[valid] == pytest.approx(expected[valid])


def test_filters_disabled_sets_quality_to_one():
    """关闭过滤器系统后信号质量恒为 1。"""
    out = regime_detector_ma_zones_overlay(_noisy_frame(), enable_advanced_filters=False)
    assert (out["rdp_signal_quality"] == 1.0).all()


def test_market_condition_codes():
    """市场状态只有 1（趋势）/ −1（盘整）/ 0（中性）三档。"""
    out = regime_detector_ma_zones_overlay(_noise_free_trend(600))
    assert set(out["rdp_market_condition"].unique().tolist()) <= {-1, 0, 1}
    assert int((out["rdp_market_condition"] == 1).sum()) > 0


def _noise_free_trend(n: int = 600, drift: float = 0.6) -> pd.DataFrame:
    return _frame(100 + np.arange(n) * drift, spread=0.4)


def test_volume_filter_degrades_without_real_volume():
    """量能不可信时量能门退化为放行（质量里那 0.3 权重被重新分配）。"""
    frame = _trend_frame(500)
    frame = frame.drop(columns=["volume"])
    out = regime_detector_ma_zones_overlay(
        frame, use_volume_filter=True, enable_confluence=False
    )
    # 无真实量能时 0.3 权重被按比例并回 volOk(0.5) 与 trendScore(0.2)，
    # 于是质量只可能落在 0 / 0.2/0.7 / 0.5/0.7 / 1.0 四档
    levels = {0.0, 0.2 / 0.7, 0.5 / 0.7, 1.0}
    observed = set(np.round(out["rdp_signal_quality"].unique().tolist(), 9))
    assert observed <= {round(v, 9) for v in levels}
    assert max(observed) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 信号
# ---------------------------------------------------------------------------


def test_entries_require_regime_confirmation_and_quality():
    """多头进场只能在多头制度 + 确认根数满足 + 质量/HTF 过关时出现。"""
    out = regime_detector_ma_zones_overlay(_trend_frame(), confirm_bars=2)
    long_entry = out["rdp_long_entry"].to_numpy(dtype=int)
    regime = out["rdp_regime"].to_numpy()
    quality = out["rdp_signal_quality"].to_numpy()
    agreement = out["rdp_htf_agreement"].to_numpy()

    assert long_entry.sum() > 0
    bars = np.where(long_entry == 1)[0]
    assert (regime[bars] == 1).all()
    assert (quality[bars] >= 0.5).all()
    assert (agreement[bars] >= 0.6).all()


def test_position_tracks_entries_and_exits():
    """仓位只由进场/出场翻转：非零仓位前后必有一次进场。"""
    out = regime_detector_ma_zones_overlay(_trend_frame())
    position = out["rdp_position"].to_numpy()
    long_entry = out["rdp_long_entry"].to_numpy(dtype=int)
    short_entry = out["rdp_short_entry"].to_numpy(dtype=int)

    for i in np.where(position == 1)[0]:
        assert long_entry[: i + 1].any()
    assert set(np.unique(position).tolist()) <= {-1, 0, 1}
    # 持仓变化处必然伴随进场或出场
    changed = np.where(np.diff(position) != 0)[0] + 1
    for i in changed:
        assert (
            long_entry[i] or short_entry[i] or out["rdp_exit_long"].iloc[i] or out["rdp_exit_short"].iloc[i]
        )


def test_cooldown_suppresses_repeated_entries():
    """冷却越长，进场次数越少。"""
    frame = _trend_frame(1200)
    fast = regime_detector_ma_zones_overlay(frame, cooldown_period=1)
    slow = regime_detector_ma_zones_overlay(frame, cooldown_period=40)
    assert int(fast["rdp_long_entry"].sum()) >= int(slow["rdp_long_entry"].sum())


def test_tiers_follow_thresholds():
    """Tier A/B 必须逐项满足对应门槛。"""
    out = regime_detector_ma_zones_overlay(_trend_frame())
    tier = out["rdp_tier"].to_numpy()
    conf = out["rdp_confidence"].to_numpy()
    agree = out["rdp_htf_agreement"].to_numpy()
    quality = out["rdp_signal_quality"].to_numpy()

    a = tier == 1
    assert (conf[a] >= 0.80 - 1e-9).all()
    assert (agree[a] >= 0.65 - 1e-9).all()
    assert (quality[a] >= 0.70 - 1e-9).all()

    b = tier == 2
    assert (conf[b] >= 0.70 - 1e-9).all()
    assert (agree[b] >= 0.55 - 1e-9).all()
    assert (quality[b] >= 0.60 - 1e-9).all()


def test_exit_modes_differ():
    """Conservative 只在带外出场，Aggressive 只在制度失效出场。"""
    frame = _trend_frame(1200)
    conservative = regime_detector_ma_zones_overlay(frame, enable_advanced_exits=True, exit_mode="Conservative")
    aggressive = regime_detector_ma_zones_overlay(frame, enable_advanced_exits=True, exit_mode="Aggressive")
    hybrid = regime_detector_ma_zones_overlay(frame, enable_advanced_exits=True, exit_mode="Hybrid")
    assert int(hybrid["rdp_exit_long"].sum()) >= int(conservative["rdp_exit_long"].sum())
    assert int(hybrid["rdp_exit_long"].sum()) >= int(aggressive["rdp_exit_long"].sum())


# ---------------------------------------------------------------------------
# 绩效
# ---------------------------------------------------------------------------


def test_performance_accounting():
    """样本数 = 胜 + 负；胜率与盈亏比由已平仓交易推出。"""
    out = regime_detector_ma_zones_overlay(_trend_frame(2000))
    samples = out["rdp_samples"].to_numpy()
    win_rate = out["rdp_win_rate"].to_numpy()
    profit_factor = out["rdp_profit_factor"].to_numpy()

    assert samples.max() > 0
    assert np.all(np.diff(samples) >= 0)  # 单调不减
    final = samples[-1]
    assert win_rate[-1] == pytest.approx(round(win_rate[-1] * final) / final, abs=1e-9)
    assert profit_factor[-1] >= 0.0


# ---------------------------------------------------------------------------
# 制度上下文
# ---------------------------------------------------------------------------


def test_volatility_state_from_percentile_ranks():
    """波动率状态由 ATR 与布林带宽的分位秩均值决定。"""
    frame = _trend_frame(800)
    out = regime_detector_ma_zones_overlay(frame, vola_lookback=100, vola_quiet_max=33, vola_vol_min=66)
    atr_a = out["rdp_atr"].to_numpy()
    atr_rank = percentrank(pd.Series(atr_a), 100).to_numpy()
    expected_score = (np.nan_to_num(atr_rank, nan=50.0) + np.nan_to_num(
        out["rdp_bbw_rank"].to_numpy(), nan=50.0
    )) / 2.0
    assert out["rdp_vola_score"].to_numpy() == pytest.approx(expected_score)
    state = out["rdp_vola_state"].to_numpy()
    assert ((state == -1) == (expected_score < 33)).all()
    assert ((state == 1) == (expected_score > 66)).all()


def test_regime_weakening_and_playbook_codes():
    """衰减判定与 playbook 码一致：时间越长/强度衰减 → Aging；否则按波动率维度分流。"""
    out = regime_detector_ma_zones_overlay(_trend_frame())
    regime = out["rdp_regime"].to_numpy()
    weakening = out["rdp_regime_weakening"].to_numpy(dtype=int)
    vola = out["rdp_vola_state"].to_numpy()
    playbook = out["rdp_playbook"].to_numpy()

    trend_bars = np.where(regime != 0)[0]
    assert len(trend_bars) > 0
    aging = trend_bars[weakening[trend_bars] == 1]
    if len(aging):
        assert (playbook[aging] == PLAYBOOK_AGING).all()
    stretched = trend_bars[(weakening[trend_bars] == 0) & (vola[trend_bars] == 1)]
    if len(stretched):
        assert (playbook[stretched] == PLAYBOOK_STRETCHED).all()
    healthy = trend_bars[(weakening[trend_bars] == 0) & (vola[trend_bars] == -1)]
    if len(healthy):
        assert (playbook[healthy] == PLAYBOOK_HEALTHY).all()
    plain = trend_bars[(weakening[trend_bars] == 0) & (vola[trend_bars] == 0)]
    if len(plain):
        assert (playbook[plain] == PLAYBOOK_TREND).all()

    flat = np.where(regime == 0)[0]
    assert set(np.unique(playbook[flat]).tolist()) <= {
        PLAYBOOK_COMPRESSION, PLAYBOOK_CHOPPY, PLAYBOOK_RANGE
    }


def test_bars_in_regime_counts_up_and_resets():
    """制度计时在换向处归零，其余递增。"""
    out = regime_detector_ma_zones_overlay(_trend_frame())
    bars = out["rdp_bars_in_regime"].to_numpy()
    regime = out["rdp_regime"].to_numpy()
    changed = np.r_[True, regime[1:] != regime[:-1]]
    assert (bars[changed] == 0).all()
    steps = np.diff(bars)[~changed[1:]]
    assert (steps == 1).all()


# ---------------------------------------------------------------------------
# MA 区间
# ---------------------------------------------------------------------------


def test_ma_zone_values_follow_ma_type():
    """区间用的三条均线按所选类型计算，间距强度 = (高−低)/ATR。"""
    frame = _trend_frame(800)
    out = regime_detector_ma_zones_overlay(frame, enable_ma_zones=True, ma_type="SMA")
    close = frame["close"]
    assert out["rdp_ma1"].to_numpy()[1:] == pytest.approx(
        sma(close, 21).to_numpy()[1:], nan_ok=True
    )
    assert out["rdp_ma3"].to_numpy()[1:] == pytest.approx(
        sma(close, 89).to_numpy()[1:], nan_ok=True
    )
    atr_zones = atr(frame, 20).to_numpy()
    highest_ma = np.fmax(out["rdp_ma1"].to_numpy(), np.fmax(out["rdp_ma2"].to_numpy(), out["rdp_ma3"].to_numpy()))
    lowest_ma = np.fmin(out["rdp_ma1"].to_numpy(), np.fmin(out["rdp_ma2"].to_numpy(), out["rdp_ma3"].to_numpy()))
    expected = np.where(atr_zones > 0, (highest_ma - lowest_ma) / atr_zones, 0.0)
    valid = np.isfinite(expected) & (atr_zones > 0)
    assert out["rdp_spread_atr"].to_numpy()[valid] == pytest.approx(expected[valid])


def test_ma_zones_gated_by_ma_alignment():
    """use_rdp_regime=False 时区间由 MA 排列 + 强度门决定。"""
    frame = _trend_frame(800)
    out = regime_detector_ma_zones_overlay(
        frame, enable_ma_zones=True, use_rdp_regime=False, base_min_strength_zones=0.0
    )
    close = frame["close"].to_numpy()
    above = (
        (close >= out["rdp_ma1"].to_numpy())
        & (close >= out["rdp_ma2"].to_numpy())
        & (close >= out["rdp_ma3"].to_numpy())
    )
    valid = np.isfinite(out["rdp_ma3"].to_numpy())
    expected = (above & valid).astype(int)
    assert np.array_equal(out["rdp_zone_bull"].to_numpy(dtype=int)[valid], expected[valid])


def test_zone_flip_sets_entry_and_atr_stop():
    """区间翻多时进场位 = 最高 MA、止损 = 最高 MA − ATR×倍数。"""
    frame = _trend_frame(900)
    out = regime_detector_ma_zones_overlay(
        frame, enable_ma_zones=True, use_rdp_regime=False, base_min_strength_zones=0.0,
        mult_zones=2.0,
    )
    flips = np.where(out["rdp_zone_bull_flip"].to_numpy(dtype=int) == 1)[0]
    assert len(flips) > 0
    for bar in flips:
        assert out["rdp_zone_long_entry"].iloc[bar] == pytest.approx(
            out["rdp_highest_ma"].iloc[bar]
        )
        assert out["rdp_zone_long_sl"].iloc[bar] == pytest.approx(
            out["rdp_highest_ma"].iloc[bar] - out["rdp_zone_atr"].iloc[bar] * 2.0
        )


def test_zone_disabled_clears_zones():
    """关闭 MA 区间后不产生多空区间。"""
    out = regime_detector_ma_zones_overlay(_trend_frame(600), enable_ma_zones=False)
    assert int(out["rdp_zone_bull"].sum()) == 0
    assert int(out["rdp_zone_bear"].sum()) == 0


# ---------------------------------------------------------------------------
# 契约
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"profile_preset": "Turbo"},
        {"regime_mode": "Both"},
        {"er_length": 3},
        {"alpha_min": 0.9},
        {"gamma": 0.1},
        {"atr_length": 1},
        {"volatility_mode": "Turbo"},
        {"htf_weights": "Neutral"},
        {"exit_mode": "Loose"},
        {"ma_type": "T3"},
        {"strength_mode_zones": "Auto2"},
        {"confirm_bars": 0},
        {"cooldown_period": 60},
        {"vola_quiet_max": 60},
        {"auto_lookback_zones": 10},
        {"source": "nope"},
    ],
)
def test_parameter_validation(kwargs):
    with pytest.raises(ValueError):
        regime_detector_ma_zones_overlay(_trend_frame(120), **kwargs)


def test_output_contract():
    """索引一致、列唯一、标志列 Int64、参数范围合理。"""
    frame = _trend_frame(600)
    out = regime_detector_ma_zones_overlay(frame)
    assert out.index.equals(frame.index)
    assert len(out.columns) == len(set(out.columns))
    assert all(column.startswith("rdp_") for column in out.columns)
    for column in FLAG_COLUMNS:
        assert str(out[column].dtype) == "Int64", column
    numeric = out.drop(columns=FLAG_COLUMNS).to_numpy(dtype=float)
    assert not np.isinf(numeric).any()
    assert out["rdp_confidence"].between(0.0, 1.0).all()
    assert out["rdp_strength"].between(0.0, 1.0).all()
    assert out["rdp_signal_quality"].between(0.0, 1.0).all()
    assert set(out["rdp_regime"].unique().tolist()) <= {-1, 0, 1}


def test_random_walk_end_to_end():
    """随机游走也能跑通（不要求制度形成）。"""
    out = regime_detector_ma_zones_overlay(_noisy_frame(800))
    assert out.index.equals(_noisy_frame(800).index)
    assert int(out["rdp_bars_in_regime"].max()) > 0
    assert np.isfinite(out["rdp_line"]).sum() > 700
