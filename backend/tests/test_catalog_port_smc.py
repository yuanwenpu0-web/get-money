"""SMC Structure & Expectation 移植测试。

期望值来源：

* **结构**：HH/HL 序列必须把结构判成多头；且**新高不单独**改结构（必须由 HL 触发）
* **预期**：HH + 多头结构 → Expect HL（参考位=上一个摆动低）；HL → Expect Continuation；
  超时 → 过期；跌破参考位 → Expect Failure
* **BOS**：需要「预期一致 + 结构一致 + Range-Mismatch 守卫（HL 必须在 HH 之后）」
* **OB**：区间取自「被破摆动 → BOS」段内最强冲量 bar 之前的最后一根反向 K；
  分数 = 15 底分 + 位移档 + FVG + 量能档 + 位置档 − 尺寸惩罚；优先级再叠加结构对齐/腿强度
* **生命周期**：渗透 ≥30% 判 Tap、≥70% 判 Failure、破位直接失效、TTL 过期删除、
  缓解扣除 15 分

所有网格都满足两个前提，否则 strict 枢轴根本不会确认：

1. ``high = close + s`` / ``low = close - s``（高低点只由收盘价决定 → 极值唯一）
2. 实体方向由收盘价变化方向决定（``open = close − body·sign(Δclose)``）→ 阴阳烛存在
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.smc_structure_and_expectation import (
    smc_structure_and_expectation,
)

FLAG_COLUMNS = [
    "smc_structure_code",
    "smc_internal_structure_code",
    "smc_last_high_type",
    "smc_last_low_type",
    "smc_swing_high_bar",
    "smc_swing_low_bar",
    "smc_structure_aligned",
    "smc_structure_conflict",
    "smc_expectation_code",
    "smc_expectation_active",
    "smc_expectation_expired",
    "smc_expect_hl",
    "smc_expect_lh",
    "smc_expect_continuation",
    "smc_expect_failure",
    "smc_bullish_bos",
    "smc_bearish_bos",
    "smc_pending_bull",
    "smc_pending_bear",
    "smc_has_bull_fvg",
    "smc_has_bear_fvg",
    "smc_bull_ob_state",
    "smc_bull_ob_mitigated",
    "smc_bull_ob_active",
    "smc_bear_ob_state",
    "smc_bear_ob_mitigated",
    "smc_bear_ob_active",
    "smc_confirmed_bullish",
    "smc_confirmed_bearish",
    "smc_reject_range_mismatch",
    "smc_reject_displacement",
    "smc_reject_location",
    "smc_reject_volume",
    "smc_reject_size",
    "smc_reject_fvg",
    "smc_reject_segment",
    "smc_bos_attempts",
    "smc_swing_break_bull",
    "smc_swing_break_bear",
    "smc_early_top",
    "smc_early_bottom",
]

# 干净的 HH/HL 上升摆动（收盘价路径）—— 拐点必须在两侧都成立才是真极值
BULL_POINTS = [
    (0, 100.0), (10, 110.0), (18, 104.0), (26, 116.0), (34, 108.0),
    (44, 124.0), (52, 116.0), (62, 130.0), (70, 126.0),
]
# 只走完第一个 BOS（收盘停在 124）—— 用来单独观察第一个 OB
ONE_BOS_POINTS = [(0, 100.0), (10, 110.0), (18, 104.0), (26, 116.0), (34, 108.0), (44, 124.0)]


def _path(points: list[tuple[int, float]], n: int) -> np.ndarray:
    """按折线插值出收盘价路径（段内不含右端点，保证极值唯一）。"""
    close = np.empty(n)
    for k in range(len(points) - 1):
        x0, y0 = points[k]
        x1, y1 = points[k + 1]
        span = np.arange(x0, x1)
        close[span] = np.linspace(y0, y1, len(span), endpoint=False)
    close[points[-1][0]:] = points[-1][1]
    return close


def _frame(close: np.ndarray, *, spread: float = 0.5, body: float = 0.2,
           volume: float = 1000.0, freq: str = "1h") -> pd.DataFrame:
    direction = np.sign(np.r_[0.0, np.diff(close)])
    open_ = close - body * direction
    idx = pd.date_range("2024-01-01", periods=len(close), freq=freq)
    return pd.DataFrame(
        {
            "open": open_,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": np.full(len(close), volume),
        },
        index=idx,
    )


def _bull_frame(n: int = 90, tail: float | None = None) -> pd.DataFrame:
    close = _path(BULL_POINTS, n)
    if tail is not None:
        close[len(close) - 20:] = np.linspace(close[len(close) - 21], tail, 20)
    return _frame(close)


SMC_KWARGS = dict(structure_lookback=2, internal_lookback=2)


# ---------------------------------------------------------------------------
# Layer 1 · 结构
# ---------------------------------------------------------------------------


def test_structure_becomes_bullish_on_higher_low():
    """HH 不改结构，HL 才把结构判成多头；低点类型依次为 HL。"""
    out = smc_structure_and_expectation(_bull_frame(), **SMC_KWARGS)
    assert int(out["smc_structure_code"].iloc[-1]) == 1
    codes = set(out["smc_last_low_type"].unique().tolist())
    assert 2 in codes  # HL
    assert 1 in set(out["smc_last_high_type"].unique().tolist())  # HH


def test_first_pivots_keep_structure_neutral():
    """只有一个高点和低点时结构仍是中性（Pine 的 First High / First Low）。"""
    close = _path([(0, 100.0), (10, 110.0), (20, 105.0), (30, 106.0)], 40)
    out = smc_structure_and_expectation(_frame(close), **SMC_KWARGS)
    # 首个高、首个低都属于 First 类型（码 5），结构保持中性
    assert int(out["smc_last_high_type"].iloc[14]) == 5
    assert int(out["smc_structure_code"].iloc[14]) == 0


def test_lower_high_flips_structure_to_bearish():
    """LH 会把结构切成空头（新高/新低单独不切）。"""
    bear = [(0, 130.0), (10, 120.0), (18, 126.0), (26, 112.0), (34, 118.0), (42, 104.0)]
    close = _path(bear, 55)
    out = smc_structure_and_expectation(_frame(close), **SMC_KWARGS)
    assert int(out["smc_structure_code"].iloc[-1]) == -1
    assert 3 in set(out["smc_last_high_type"].unique().tolist())  # LH


# ---------------------------------------------------------------------------
# Layer 2 · 预期
# ---------------------------------------------------------------------------


def test_expectation_follows_structure():
    """HH@多头 → Expect HL；HL@多头 → Expect Continuation。"""
    out = smc_structure_and_expectation(_bull_frame(), **SMC_KWARGS)
    codes = set(out["smc_expectation_code"].unique().tolist())
    assert 1 in codes   # Expect HL
    assert 2 in codes   # Expect Continuation

    hl_bars = [i for i in range(len(out)) if int(out["smc_expect_hl"].iloc[i]) == 1]
    assert hl_bars
    # 预期 HL 的参考位就是当时的摆动低
    reference = out["smc_expectation_price"].iloc[hl_bars[0]]
    assert reference == pytest.approx(out["smc_swing_low"].iloc[hl_bars[0]])


def test_expectation_expires_after_window():
    """超过 ``expectation_expiry`` 没有确认 → 预期作废。"""
    # 走完一个 HH 后长时间横盘：预期 HL 迟迟得不到确认
    close = _path([(0, 100.0), (8, 110.0), (16, 102.0), (24, 114.0), (32, 106.0), (40, 120.0)], 70)
    frame = _frame(close)
    out = smc_structure_and_expectation(frame, structure_lookback=2, internal_lookback=2,
                                        expectation_expiry=10)
    assert int(out["smc_expectation_expired"].sum()) >= 1
    assert int(out["smc_expectation_active"].iloc[-1]) == 0
    assert int(out["smc_expectation_code"].iloc[-1]) == 0


def test_expectation_downgrades_on_break():
    """跌破 Expect HL 的参考低 → 降级为 Expect Failure。"""
    points = [(0, 100.0), (8, 110.0), (16, 102.0), (24, 114.0), (32, 106.0), (40, 120.0), (52, 100.0)]
    frame = _frame(_path(points, 64))
    out = smc_structure_and_expectation(frame, structure_lookback=2, internal_lookback=2)
    assert int(out["smc_expect_failure"].sum()) >= 1
    failure_bars = [i for i in range(len(out)) if int(out["smc_expect_failure"].iloc[i]) == 1]
    assert int(out["smc_expectation_code"].iloc[failure_bars[0]]) == 3


# ---------------------------------------------------------------------------
# Layer 3 · BOS
# ---------------------------------------------------------------------------


def test_bullish_bos_fires_with_range_guard():
    """收盘上破摆动高 + 预期一致 + HL 在 HH 之后 → 确认 BOS 并记录上下文区间。"""
    out = smc_structure_and_expectation(_bull_frame(), **SMC_KWARGS)
    bos_bars = [i for i in range(len(out)) if int(out["smc_bullish_bos"].iloc[i]) == 1]
    assert len(bos_bars) >= 2

    first = bos_bars[0]
    # 区间 = 被破的 HH 与最近 HL
    assert out["smc_bos_range_high"].iloc[first] == pytest.approx(116.5)
    assert out["smc_bos_range_low"].iloc[first] == pytest.approx(107.5)
    # 腿强度 = 冲量 / 修正
    assert out["smc_leg_strength"].iloc[first] == pytest.approx((116.5 - 103.5) / (116.5 - 107.5))
    # BOS 后摆动位被消费，下一根不再有同一个可破位
    assert not np.isfinite(out["smc_bos_swing_high"].iloc[first])


def test_bos_requires_directional_expectation():
    """结构为空头时，上破摆动高不会被确认为多头 BOS。"""
    bear = [(0, 130.0), (10, 120.0), (18, 126.0), (26, 112.0), (34, 118.0), (42, 130.0)]
    close = _path(bear, 55)
    out = smc_structure_and_expectation(_frame(close), **SMC_KWARGS)
    assert int(out["smc_bullish_bos"].sum()) == 0


def test_aggro_mode_requires_close_confirmation():
    """激进模式先进入 pending，收盘确认后才算 BOS。"""
    close = _path(BULL_POINTS, 90)
    frame = _frame(close)
    aggro = smc_structure_and_expectation(
        frame, bos_mode="High/Low-cross", require_confirm=True, confirm_bars=2, **SMC_KWARGS
    )
    assert int(aggro["smc_pending_bull"].sum()) >= 1
    strict = smc_structure_and_expectation(
        frame, bos_mode="High/Low-cross", require_confirm=False, **SMC_KWARGS
    )
    # 不要求确认时没有 pending 状态
    assert int(strict["smc_pending_bull"].sum()) == 0
    assert int(strict["smc_bullish_bos"].sum()) >= int(aggro["smc_bullish_bos"].sum())


# ---------------------------------------------------------------------------
# Layer 3 · Order Block
# ---------------------------------------------------------------------------


def _ob_frame(n: int = 90, retest: float | None = None) -> pd.DataFrame:
    """只触发第一个 BOS 的网格；``retest`` 给定时在末尾回踩到该价。"""
    points = list(ONE_BOS_POINTS)
    if retest is None:
        return _frame(_path(points, n))
    points += [(50, 124.0), (64, retest)]
    return _frame(_path(points, 80))


def _ob_kwargs(**extra) -> dict:
    kwargs = dict(SMC_KWARGS)
    kwargs.update(use_auto_displacement=False, min_displacement=0.5, min_ob_range=0.1)
    kwargs.update(extra)
    return kwargs


def test_order_block_geometry_and_score_identity():
    """OB 取「最强冲量前最后一根反向 K」；分数 = 底分 + 位移 + FVG + 量能 + 位置 − 尺寸惩罚。"""
    out = smc_structure_and_expectation(_ob_frame(), **_ob_kwargs())
    active = [i for i in range(len(out)) if int(out["smc_bull_ob_active"].iloc[i]) == 1]
    assert active, "应创建多头 OB"
    bar = active[0]

    ob_top = out["smc_bull_ob_top"].iloc[bar]
    ob_bottom = out["smc_bull_ob_bottom"].iloc[bar]
    assert ob_top > ob_bottom
    # 本轮形态里反向 K 就是 108 那根低点 bar
    assert ob_top == pytest.approx(108.5)
    assert ob_bottom == pytest.approx(107.5)

    # 逐项复算分数（该网格量能为常数 → 相对量能 1.0，不加分）
    disp = out["smc_bar_displacement"].iloc[bar]
    assert out["smc_relative_volume"].iloc[bar] == pytest.approx(1.0)
    score = 15.0
    score += 30.0 if disp >= 2.0 else 20.0 if disp >= 1.5 else 10.0
    if int(out["smc_has_bull_fvg"].iloc[bar]) == 1:
        score += 20.0
    low_range = out["smc_bos_range_low"].iloc[bar]
    high_range = out["smc_bos_range_high"].iloc[bar]
    span = high_range - low_range
    if ob_bottom < low_range + span * 0.38:
        score += 20.0
    elif ob_bottom < (high_range + low_range) / 2.0:
        score += 10.0
    assert out["smc_bull_ob_score"].iloc[bar] == pytest.approx(score)

    # 优先级 = 分数 + 对齐 30 / 冲突 −15 + 腿强度档，封顶 100
    priority = score
    if int(out["smc_structure_aligned"].iloc[bar]) == 1:
        priority += 30.0
    elif int(out["smc_structure_conflict"].iloc[bar]) == 1:
        priority -= 15.0
    strength = out["smc_leg_strength"].iloc[bar]
    if np.isfinite(strength):
        if strength > 1.5:
            priority += 25.0
        elif strength > 1.0:
            priority += 15.0
    assert out["smc_bull_ob_priority"].iloc[bar] == pytest.approx(min(priority, 100.0))


def test_order_block_tap_and_failure_states():
    """浅回踩 → Tap；深回踩（>70% 渗透）→ Failure 并缓解扣 15 分。"""
    shallow = smc_structure_and_expectation(_ob_frame(retest=108.8), **_ob_kwargs())
    assert 2 in set(shallow["smc_bull_ob_state"].unique().tolist())  # Tap

    deep = smc_structure_and_expectation(_ob_frame(retest=108.2), **_ob_kwargs())
    assert 4 in set(deep["smc_bull_ob_state"].unique().tolist())  # Failure
    mitigated = [i for i in range(len(deep)) if int(deep["smc_bull_ob_mitigated"].iloc[i]) == 1]
    assert mitigated, "深度渗透应触发缓解"
    first = mitigated[0]
    before = deep["smc_bull_ob_score"].iloc[first - 1]
    after = deep["smc_bull_ob_score"].iloc[first]
    assert after == pytest.approx(before - 15.0)


def test_order_block_invalidated_when_broken():
    """跌破 OB 下沿 → OB 失效（active 归零）。"""
    out = smc_structure_and_expectation(_ob_frame(retest=106.0), **_ob_kwargs())
    active = [i for i in range(len(out)) if int(out["smc_bull_ob_active"].iloc[i]) == 1]
    assert active
    assert int(out["smc_bull_ob_active"].iloc[-1]) == 0


def test_order_block_ttl_expiry():
    """超过 TTL 后 OB 被删除。"""
    out = smc_structure_and_expectation(_ob_frame(), **_ob_kwargs(ob_ttl=20))
    active = [i for i in range(len(out)) if int(out["smc_bull_ob_active"].iloc[i]) == 1]
    assert active
    # 第一段连续存活长度受 TTL 约束（后面还可能因新的 BOS 重建 OB）
    first_run_end = active[0]
    for bar in active:
        if bar != first_run_end + 1:
            break
        first_run_end = bar
    assert first_run_end - active[0] <= 20
    assert int(out["smc_bull_ob_active"].iloc[-1]) == 0


def test_displacement_filter_rejects_weak_impulse():
    """自动位移门槛（1h → 1.0 ATR）下，这个平缓网格的 OB 会被拒绝并计数。"""
    out = smc_structure_and_expectation(_bull_frame(), **SMC_KWARGS)
    assert int(out["smc_bull_ob_active"].max()) == 0
    assert int(out["smc_reject_displacement"].iloc[-1]) >= 1
    assert int(out["smc_bos_attempts"].iloc[-1]) >= 1


# ---------------------------------------------------------------------------
# Layer 4 · 警告
# ---------------------------------------------------------------------------


def test_early_top_warning_needs_wick_and_rsi():
    """长上影 + 收盘显著低于高点 → 早期顶部警告（同时受冷却约束）。"""
    n = 120
    close = np.linspace(100.0, 140.0, n)
    frame = _frame(close, spread=0.5)
    # 人为制造一根长上影 + 收在低位
    frame.iloc[-1, frame.columns.get_loc("high")] = close[-1] + 6.0
    frame.iloc[-1, frame.columns.get_loc("close")] = close[-1]
    out = smc_structure_and_expectation(frame, **SMC_KWARGS)
    # RSI 在单边上涨末端应偏高，若不足则用更低阈值再验一次
    if float(out["smc_rsi"].iloc[-1]) > 70:
        assert int(out["smc_early_top"].iloc[-1]) == 1
    else:
        relaxed = smc_structure_and_expectation(frame, rsi_overbought=60, **SMC_KWARGS)
        assert int(relaxed["smc_early_top"].sum()) >= 1


def test_swing_level_break_warning_uses_cooldown():
    """多头结构里跌破摆动低 → 警告，且受冷却窗口限制。"""
    # HH → HL（结构转多、摆动低抬到 115.5）→ 跌破该 HL
    points = [(0, 100.0), (10, 110.0), (18, 104.0), (26, 116.0), (34, 108.0), (44, 124.0), (52, 116.0), (64, 104.0)]
    frame = _frame(_path(points, 74))
    out = smc_structure_and_expectation(frame, warning_cooldown=100, **SMC_KWARGS)
    total = int(out["smc_swing_break_bull"].sum())
    assert total >= 1
    assert total <= 1  # 冷却 100 根内只报一次


# ---------------------------------------------------------------------------
# 契约
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"structure_lookback": 1},
        {"structure_lookback": 21},
        {"internal_lookback": 11},
        {"expectation_expiry": 5},
        {"bos_mode": "Wick-cross"},
        {"confirm_bars": 6},
        {"min_displacement": 0.1},
        {"size_filter_mode": "Soft"},
        {"ob_location_threshold": 0.9},
        {"ob_ttl": 10},
        {"mitigation_penetration": 0.05},
        {"wick_threshold": 0.9},
        {"rsi_length": 3},
        {"rsi_overbought": 95},
        {"warning_cooldown": 5},
        {"displacement_window": 11},
    ],
)
def test_parameter_validation(kwargs):
    with pytest.raises(ValueError):
        smc_structure_and_expectation(_frame(_path(BULL_POINTS, 90)), **kwargs)


def test_output_contract():
    """索引一致、列唯一、标志列 Int64、无 inf、状态码取值合法。"""
    frame = _ob_frame()
    out = smc_structure_and_expectation(frame, **_ob_kwargs())
    assert out.index.equals(frame.index)
    assert len(out.columns) == len(set(out.columns))
    assert all(column.startswith("smc_") for column in out.columns)
    for column in FLAG_COLUMNS:
        assert str(out[column].dtype) == "Int64", column
    numeric = out.drop(columns=FLAG_COLUMNS).to_numpy(dtype=float)
    assert not np.isinf(numeric).any()
    assert set(out["smc_structure_code"].unique().tolist()) <= {-1, 0, 1}
    assert set(out["smc_bull_ob_state"].unique().tolist()) <= {0, 1, 2, 3, 4}
    assert out["smc_bull_ob_priority"].dropna().between(0.0, 100.0).all()


def test_internal_structure_alignment_flag():
    """内外部结构都用同一网格时应当对齐。"""
    out = smc_structure_and_expectation(_bull_frame(), **SMC_KWARGS)
    assert int(out["smc_structure_aligned"].sum()) >= 1
    off = smc_structure_and_expectation(_bull_frame(), structure_lookback=2, enable_internal_structure=False)
    assert int(off["smc_structure_aligned"].sum()) == 0
    assert int(off["smc_structure_conflict"].sum()) == 0
