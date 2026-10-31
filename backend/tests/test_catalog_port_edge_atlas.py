"""Edge Atlas（价格水平图谱）移植测试。

期望值全部按 Pine v1.7.2 的公式手算，或由构造数据直接推出：

* 周期水平 = 上一根已完成 HTF bar 的 H/L/C + 当前 HTF bar 的 O
* 五种枢轴公式逐项手算
* 斐波回撤/扩展位 = 摆动腿端点 ± 腿长 × 比例
* VWAP 在每个周期首根重置（等比价格时 VWAP == 价格）
* 会话区间区分“已完成”与“正在形成”
* 开盘区间在窗口走完前不登记
* 整数关口按 1/2/5×10ⁿ 自动步长
* registry：优先级压过距离、ATR 最小间距、方向/优先级/数量过滤
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.edge_atlas import _session_mask, _to_local_index, edge_atlas

FLAG_COLUMNS = [
    "ea_levels_registered",
    "ea_levels_selected",
    "ea_above_count",
    "ea_below_count",
    "ea_nearest_above_priority",
    "ea_nearest_above_category",
    "ea_nearest_below_priority",
    "ea_nearest_below_category",
    "ea_fib_leg",
    "ea_support_touches_1",
    "ea_support_touches_2",
    "ea_resistance_touches_1",
    "ea_resistance_touches_2",
]


def _frame(close: np.ndarray, *, freq: str = "1h", spread: float = 1.0,
           volume: float = 1000.0, start: str = "2024-01-01",
           gapped: bool = False) -> pd.DataFrame:
    """构造 OHLCV。

    ``gapped=False``（默认）让 ``open == close``，high/low 对称张开 —— 这样拐点处
    只有一根 bar 是极值，`ta.pivothigh` 才能确认它。``gapped=True`` 用“上一根收盘
    作开盘”的真实形态，用于需要区分 open 的场景（周期水平 / 枢轴）。
    """
    idx = pd.date_range(start, periods=len(close), freq=freq)
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]]) if gapped else close
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": np.full(len(close), volume),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# 周期水平
# ---------------------------------------------------------------------------


def test_daily_period_levels_match_hand_calc():
    """两个完整 UTC 日：PDH/PDL/PDC 取前一日，DO 取当日开盘。"""
    day1 = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                     112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101], dtype=float)
    day2 = np.array([120, 119, 118, 117, 116, 115, 114, 113, 122, 123, 124, 125,
                     126, 127, 128, 129, 128, 127, 126, 125, 124, 123, 122, 121], dtype=float)
    close = np.concatenate([day1, day2])
    frame = _frame(close, gapped=True)
    out = edge_atlas(frame, preset="Clean")

    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    open_ = frame["open"].to_numpy()

    # 第一天没有“前一日”，PDH 必须是 na
    assert np.isnan(out["ea_pdh"].iloc[23])
    # 第二天开始，前一日（bar 0..23）的极值
    assert out["ea_pdh"].iloc[24] == pytest.approx(high[:24].max())
    assert out["ea_pdl"].iloc[24] == pytest.approx(low[:24].min())
    assert out["ea_pdc"].iloc[24] == pytest.approx(close[23])
    assert out["ea_pdo"].iloc[24] == pytest.approx(open_[0])
    assert out["ea_do"].iloc[24] == pytest.approx(open_[24])
    assert out["ea_pdm"].iloc[24] == pytest.approx((high[:24].max() + low[:24].min()) / 2.0)


def test_higher_period_opens_available_from_first_bar():
    """当前周/月/年的开盘价在周期第一根就可登记（Pine 用 lookahead_on 读 open）。"""
    close = np.linspace(100, 120, 48)
    frame = _frame(close)
    out = edge_atlas(frame, preset="Everything")
    assert out["ea_wo"].iloc[0] == pytest.approx(frame["open"].iloc[0])
    assert out["ea_mo"].iloc[0] == pytest.approx(frame["open"].iloc[0])
    # 周/月高点是“上一周期”，第一根必然为空
    assert np.isnan(out["ea_pwh"].iloc[0])
    assert np.isnan(out["ea_pmh"].iloc[0])


# ---------------------------------------------------------------------------
# 枢轴点
# ---------------------------------------------------------------------------


def test_pivot_methods_match_hand_calc():
    """五种枢轴公式全部对手算值。"""
    day1 = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                     112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101], dtype=float)
    day2 = np.linspace(120, 121, 24)
    frame = _frame(np.concatenate([day1, day2]), gapped=True)

    high = frame["high"].to_numpy()[:24].max()
    low = frame["low"].to_numpy()[:24].min()
    close = day1[-1]
    open_ = frame["open"].to_numpy()[0]
    span = high - low
    pp_trad = (high + low + close) / 3.0
    i = 24

    out = edge_atlas(frame, preset="Custom", enable_pivots=True)
    assert out["ea_pp"].iloc[i] == pytest.approx(pp_trad)
    assert out["ea_r1"].iloc[i] == pytest.approx(2.0 * pp_trad - low)
    assert out["ea_s1"].iloc[i] == pytest.approx(2.0 * pp_trad - high)
    assert out["ea_r2"].iloc[i] == pytest.approx(pp_trad + span)
    assert out["ea_s2"].iloc[i] == pytest.approx(pp_trad - span)
    assert out["ea_r3"].iloc[i] == pytest.approx(high + 2.0 * (pp_trad - low))
    assert out["ea_s3"].iloc[i] == pytest.approx(low - 2.0 * (high - pp_trad))

    out = edge_atlas(frame, preset="Custom", enable_pivots=True, pivot_method="Fibonacci")
    assert out["ea_r1"].iloc[i] == pytest.approx(pp_trad + 0.382 * span)
    assert out["ea_s2"].iloc[i] == pytest.approx(pp_trad - 0.618 * span)
    assert out["ea_r3"].iloc[i] == pytest.approx(pp_trad + span)

    out = edge_atlas(frame, preset="Custom", enable_pivots=True, pivot_method="Camarilla")
    assert out["ea_r1"].iloc[i] == pytest.approx(close + span * 1.1 / 12.0)
    assert out["ea_r2"].iloc[i] == pytest.approx(close + span * 1.1 / 6.0)
    assert out["ea_s3"].iloc[i] == pytest.approx(close - span * 1.1 / 4.0)

    out = edge_atlas(frame, preset="Custom", enable_pivots=True, pivot_method="Woodie")
    pp_woodie = (high + low + 2.0 * close) / 4.0
    assert out["ea_pp"].iloc[i] == pytest.approx(pp_woodie)
    assert out["ea_r1"].iloc[i] == pytest.approx(2.0 * pp_woodie - low)
    assert out["ea_s1"].iloc[i] == pytest.approx(2.0 * pp_woodie - high)
    assert out["ea_r3"].iloc[i] == pytest.approx(high + 2.0 * (pp_woodie - low))


def test_demark_pivot_has_only_three_levels():
    """DeMark 只定义 PP/R1/S1；收盘 > 开盘时用 2H+L+C。"""
    day1 = np.linspace(100, 110, 24)
    frame = _frame(np.concatenate([day1, np.linspace(110, 112, 24)]), gapped=True)
    high = frame["high"].to_numpy()[:24].max()
    low = frame["low"].to_numpy()[:24].min()
    close = day1[-1]
    open_ = frame["open"].to_numpy()[0]
    assert close > open_  # 走 2H + L + C 分支
    x = 2.0 * high + low + close

    out = edge_atlas(frame, preset="Custom", enable_pivots=True, pivot_method="DeMark")
    assert out["ea_pp"].iloc[24] == pytest.approx(x / 4.0)
    assert out["ea_r1"].iloc[24] == pytest.approx(x / 2.0 - low)
    assert out["ea_s1"].iloc[24] == pytest.approx(x / 2.0 - high)
    for column in ("ea_r2", "ea_s2", "ea_r3", "ea_s3"):
        assert np.isnan(out[column].iloc[24]), column


# ---------------------------------------------------------------------------
# 摆动斐波
# ---------------------------------------------------------------------------


def test_fib_levels_match_hand_calc():
    """低点 90 → 高点 110 的多头腿：回撤自高点、扩展自低点。"""
    close = np.concatenate([
        np.linspace(100, 90, 11),
        np.linspace(90, 110, 20),
        np.linspace(109.0, 104, 29),
    ])
    frame = _frame(close, spread=0.4)
    out = edge_atlas(
        frame,
        preset="Custom",
        enable_fib=True,
        fib_pivot_left=3,
        fib_pivot_right=3,
        fib_236=True,
        fib_786=True,
        fib_100=True,
        fib_1272=True,
        fib_1618=True,
    )
    last = -1
    leg_high = float(out["ea_fib_high"].iloc[last])
    leg_low = float(out["ea_fib_low"].iloc[last])
    assert leg_high == pytest.approx(110.4)
    assert leg_low == pytest.approx(89.6)
    assert int(out["ea_fib_leg"].iloc[last]) == 1  # 低点在前 → 多头腿

    size = leg_high - leg_low
    assert out["ea_fib_382"].iloc[last] == pytest.approx(leg_high - size * 0.382)
    assert out["ea_fib_618"].iloc[last] == pytest.approx(leg_high - size * 0.618)
    assert out["ea_fib_236"].iloc[last] == pytest.approx(leg_high - size * 0.236)
    assert out["ea_fib_786"].iloc[last] == pytest.approx(leg_high - size * 0.786)
    assert out["ea_fib_100"].iloc[last] == pytest.approx(leg_low)
    assert out["ea_fib_1272"].iloc[last] == pytest.approx(leg_low + size * 1.272)
    assert out["ea_fib_1618"].iloc[last] == pytest.approx(leg_low + size * 1.618)


def test_fib_leg_invalid_when_too_short():
    """腿长不足 ``fib_min_leg_atr`` 倍 ATR 时不登记任何斐波水平。"""
    close = 100 + np.sin(np.arange(120) / 4.0)  # 小幅快摆
    frame = _frame(close, spread=0.8)
    out = edge_atlas(
        frame,
        preset="Custom",
        enable_fib=True,
        fib_pivot_left=2,
        fib_pivot_right=2,
        fib_min_leg_atr=50.0,
        max_distance_atr=500.0,
        min_label_spacing_atr=0.0,
    )
    assert out["ea_fib_382"].notna().any()  # 水平本身算了
    # 但 registry 里不应出现 FIB 类别
    assert not (out["ea_nearest_above_category"] == 8).any()
    assert not (out["ea_nearest_below_category"] == 8).any()


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------


def test_vwap_resets_each_period():
    """恒定价格下 VWAP == 价格；跨周期时在首根重置。"""
    close = np.array([100.0] * 24 + [200.0] * 24)
    frame = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": np.full(len(close), 10.0)},
        index=pd.date_range("2024-01-01", periods=len(close), freq="1h"),
    )
    out = edge_atlas(frame, preset="Custom", enable_vwap=True)
    assert out["ea_vwap_session"].iloc[23] == pytest.approx(100.0)
    assert out["ea_vwap_session"].iloc[24] == pytest.approx(200.0)
    assert out["ea_vwap_session"].iloc[47] == pytest.approx(200.0)


def test_vwap_is_volume_weighted():
    """同一周期内两根不同价格 + 不同成交量 → 成交量加权均值。"""
    close = np.array([100.0, 200.0])
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.array([3.0, 1.0]),
        },
        index=pd.date_range("2024-01-01 00:00", periods=2, freq="1h"),
    )
    out = edge_atlas(frame, preset="Custom", enable_vwap=True)
    # (100*3 + 200*1) / 4
    assert out["ea_vwap_session"].iloc[1] == pytest.approx(125.0)


# ---------------------------------------------------------------------------
# 会话区间
# ---------------------------------------------------------------------------


def test_session_completed_vs_developing():
    """Asia 0000-0800：已完成区间稳定，正在形成区间随 bar 扩张。"""
    close = np.concatenate([np.linspace(100, 108, 24), np.linspace(200, 208, 24)])
    frame = _frame(close)

    done = edge_atlas(frame, preset="Custom", enable_sessions=True, asia_enable=True, asia_mid=True)
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    day1_high = high[:8].max()
    day2_high = high[24:32].max()
    assert done["ea_asia_high"].iloc[8] == pytest.approx(day1_high)
    assert done["ea_asia_high"].iloc[24] == pytest.approx(day1_high)  # 新日 Asia 中仍是前一日
    assert done["ea_asia_high"].iloc[32] == pytest.approx(day2_high)
    assert done["ea_asia_mid"].iloc[32] == pytest.approx(
        (high[24:32].max() + low[24:32].min()) / 2.0
    )

    dev = edge_atlas(
        frame, preset="Custom", enable_sessions=True, asia_enable=True, use_completed_session=False
    )
    assert dev["ea_asia_high"].iloc[30] == pytest.approx(high[24:31].max())


def test_session_timezone_boundaries():
    """纽约 0930-1600 在 1 月（EST=UTC-5）对应 UTC 14:30-21:00。"""
    idx = pd.date_range("2024-01-15", periods=24, freq="1h")
    local = _to_local_index(idx, "America/New_York")
    mask = _session_mask(local, "0930-1600")
    inside = [int(idx[i].hour) for i in range(24) if mask[i]]
    assert inside == [15, 16, 17, 18, 19, 20]


# ---------------------------------------------------------------------------
# 开盘区间
# ---------------------------------------------------------------------------


def test_opening_range_waits_until_window_closes():
    """15m + 30 分钟窗口 = 每日前 2 根；窗口未走完不登记。"""
    close = 100 + np.arange(192) * 0.1
    frame = _frame(close, freq="15min")
    out = edge_atlas(
        frame, preset="Custom", enable_opening_ranges=True, daily_or_minutes=30
    )
    day1_high = (close[:2] + 1.0).max()
    day2_high = (close[96:98] + 1.0).max()
    assert np.isnan(out["ea_dor_high"].iloc[1])  # 第二根仍在窗口内
    assert out["ea_dor_high"].iloc[2] == pytest.approx(day1_high)
    assert out["ea_dor_high"].iloc[96] == pytest.approx(day1_high)
    assert out["ea_dor_high"].iloc[98] == pytest.approx(day2_high)
    # 周窗口默认 1440 分钟（首日）→ 第 96 根收口，值等于首日极值
    weekly = frame["high"].to_numpy()[:96].max()
    assert np.isnan(out["ea_wor_high"].iloc[95])
    assert out["ea_wor_high"].iloc[96] == pytest.approx(weekly)


# ---------------------------------------------------------------------------
# 整数关口
# ---------------------------------------------------------------------------


def test_round_number_steps():
    """价格 100 → 自动步长 2（1/2/5×10ⁿ），手动步长生效，Half 档收窄到 1。"""
    frame = _frame(np.full(24, 100.0), spread=0.0)
    auto = edge_atlas(frame, preset="Custom", enable_round=True)
    assert auto["ea_round_major_above"].iloc[0] == pytest.approx(102.0)
    assert auto["ea_round_major_below"].iloc[0] == pytest.approx(98.0)

    manual = edge_atlas(
        frame, preset="Custom", enable_round=True,
        round_auto_step=False, round_manual_step=10.0,
    )
    assert manual["ea_round_major_above"].iloc[0] == pytest.approx(110.0)
    assert manual["ea_round_major_below"].iloc[0] == pytest.approx(90.0)

    half = edge_atlas(frame, preset="Custom", enable_round=True, round_half=True)
    assert half["ea_round_major_above"].iloc[0] == pytest.approx(101.0)
    assert half["ea_round_major_below"].iloc[0] == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# 枢轴聚类 S/R
# ---------------------------------------------------------------------------


def _zigzag_frame(n: int = 110) -> np.ndarray:
    """四次往返 100↔110，之后横盘在 100。"""
    segs = []
    for _ in range(4):
        segs.append(np.linspace(100.0, 110.0, 7))
        segs.append(np.linspace(110.0, 100.0, 7)[1:])
    close = np.concatenate(segs)
    return np.concatenate([close[:52], np.full(n - 52, close[51])])


def test_sr_clusters_count_touches():
    """同价位反复确认 → 阻力 4 次触达、支撑 6 次触达。"""
    frame = _frame(_zigzag_frame(), spread=0.5)
    out = edge_atlas(
        frame, preset="Custom", enable_sr=True,
        sr_pivot_left=2, sr_pivot_right=2, sr_min_touches=2, sr_max_age=2000,
    )
    assert out["ea_resistance_1"].iloc[-1] == pytest.approx(110.5)
    assert int(out["ea_resistance_touches_1"].iloc[-1]) == 4
    assert out["ea_support_1"].iloc[-1] == pytest.approx(99.5)
    assert int(out["ea_support_touches_1"].iloc[-1]) == 6


def test_sr_break_resets_touch_count():
    """收盘越过簇中心超过容差 → 触达计数归零，阻力不再登记。"""
    close = _zigzag_frame().copy()
    close[52:] = np.linspace(115, 120, len(close) - 52)
    frame = _frame(close, spread=0.5)
    out = edge_atlas(
        frame, preset="Custom", enable_sr=True,
        sr_pivot_left=2, sr_pivot_right=2, sr_min_touches=2, sr_max_age=2000,
    )
    assert np.isnan(out["ea_resistance_1"].iloc[-1])
    assert out["ea_support_1"].iloc[-1] == pytest.approx(99.5)


def test_sr_min_touches_filters_single_confirmations():
    """触达门槛提到 5 后，只有 6 次确认的支撑留下。"""
    frame = _frame(_zigzag_frame(), spread=0.5)
    out = edge_atlas(
        frame, preset="Custom", enable_sr=True,
        sr_pivot_left=2, sr_pivot_right=2, sr_min_touches=5, sr_max_age=2000,
    )
    assert out["ea_support_1"].iloc[-1] == pytest.approx(99.5)
    assert np.isnan(out["ea_resistance_1"].iloc[-1])


# ---------------------------------------------------------------------------
# registry 过滤
# ---------------------------------------------------------------------------


def _manual_frame(n: int = 40) -> pd.DataFrame:
    return _frame(np.full(n, 100.0), spread=1.0)


def _manual_kwargs(**overrides) -> dict:
    kwargs = dict(
        preset="Custom",
        enable_manual=True,
        manual1_enable=True, manual1_price=100.5,
        manual2_enable=True, manual2_price=100.9,
        manual3_enable=True, manual3_price=95.0,
        manual4_enable=True, manual4_price=106.0,
    )
    kwargs.update(overrides)
    return kwargs


def test_registry_priority_dominates_distance():
    """PDH（优先级 9）比更近的手动位（6）先被接受，间距把后者挤掉。"""
    frame = _manual_frame()
    out = edge_atlas(frame, **_manual_kwargs())
    last = -1
    assert out["ea_nearest_above"].iloc[last] == pytest.approx(101.0)
    assert int(out["ea_nearest_above_priority"].iloc[last]) == 9

    # 关闭间距后，最近的 100.0 反而胜出（优先级 8 的 DO）
    loose = edge_atlas(frame, **_manual_kwargs(min_label_spacing_atr=0.0))
    assert loose["ea_nearest_above"].iloc[last] == pytest.approx(100.0)
    assert int(loose["ea_nearest_above_priority"].iloc[last]) == 8
    assert int(loose["ea_levels_selected"].iloc[last]) > int(out["ea_levels_selected"].iloc[last])


def test_registry_spacing_keeps_levels_apart():
    """ATR 最小间距会挤掉过近的水平；间距放大到 6×ATR 时只剩一个。"""
    frame = _manual_frame()
    loose = edge_atlas(frame, **_manual_kwargs(min_label_spacing_atr=1.5))
    assert loose["ea_nearest_above"].iloc[-1] == pytest.approx(101.0)
    assert loose["ea_nearest_below"].iloc[-1] == pytest.approx(95.0)
    assert int(loose["ea_levels_selected"].iloc[-1]) >= 3

    tight = edge_atlas(frame, **_manual_kwargs(min_label_spacing_atr=6.0))
    assert int(tight["ea_levels_selected"].iloc[-1]) == 1
    assert tight["ea_nearest_above"].iloc[-1] == pytest.approx(101.0)
    assert np.isnan(tight["ea_nearest_below"].iloc[-1])


def test_registry_limits_and_sides():
    """上限、方向开关、最低优先级三条硬过滤。"""
    frame = _manual_frame()
    capped = edge_atlas(frame, **_manual_kwargs(max_levels=1))
    assert int(capped["ea_levels_selected"].iloc[-1]) == 1

    no_above = edge_atlas(frame, **_manual_kwargs(show_above=False))
    assert int(no_above["ea_above_count"].iloc[-1]) == 0
    assert np.isnan(no_above["ea_nearest_above"].iloc[-1])
    assert int(no_above["ea_below_count"].iloc[-1]) > 0

    strict = edge_atlas(frame, **_manual_kwargs(min_priority=10))
    assert int(strict["ea_levels_selected"].iloc[-1]) == 0
    assert int(strict["ea_levels_registered"].iloc[-1]) > 0

    tight = edge_atlas(frame, **_manual_kwargs(max_distance_atr=0.25))
    assert int(tight["ea_levels_selected"].iloc[-1]) <= 1


def test_master_switch_disables_registry():
    """总开关关闭时 registry 不登记任何水平，但模块列仍算出数值。"""
    frame = _manual_frame()
    out = edge_atlas(frame, **_manual_kwargs(master_enable=False))
    assert int(out["ea_levels_selected"].iloc[-1]) == 0
    assert int(out["ea_levels_registered"].iloc[-1]) == 0
    assert not np.isnan(out["ea_manual_1"].iloc[-1])


def test_presets_change_registered_counts():
    """预设叠加的模块越多，登记的水平越多，选中数受上限约束。"""
    close = 100 + np.cumsum(np.sin(np.arange(300) / 7.0) * 0.4)
    frame = _frame(close)
    counts = {}
    for preset in ("Clean", "Intraday", "Structure", "Everything"):
        out = edge_atlas(frame, preset=preset)
        counts[preset] = int(out["ea_levels_registered"].max())
        assert int(out["ea_levels_selected"].max()) <= 24
    assert counts["Clean"] < counts["Intraday"]
    assert counts["Intraday"] < counts["Everything"]
    assert counts["Structure"] < counts["Everything"]


# ---------------------------------------------------------------------------
# 参数校验与输出契约
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"preset": "Nope"},
        {"max_levels": 0},
        {"max_levels": 121},
        {"max_distance_atr": 0.1},
        {"min_priority": 0},
        {"min_priority": 11},
        {"sr_cluster_atr": 0.01},
        {"sr_pivot_left": 0},
        {"fib_min_leg_atr": 0.1},
        {"pivot_method": "Sine"},
        {"pivot_period": "Hourly"},
        {"trend_preset": "Fast"},
        {"round_levels_each_side": 0},
        {"atr_length": 0},
        {"asia_session": "nonsense"},
    ],
)
def test_parameter_validation(kwargs):
    frame = _manual_frame(30)
    with pytest.raises(ValueError):
        edge_atlas(frame, **{"preset": "Custom", **kwargs})


def test_output_contract():
    """索引一致、列名唯一、标志列为 Int64、无数值溢出。"""
    close = 100 + np.cumsum(np.sin(np.arange(300) / 5.0) * 0.5)
    frame = _frame(close)
    out = edge_atlas(frame, preset="Everything")

    assert out.index.equals(frame.index)
    assert len(out.columns) == len(set(out.columns))
    assert all(column.startswith("ea_") for column in out.columns)
    for column in FLAG_COLUMNS:
        assert str(out[column].dtype) == "Int64", column
    numeric = out.drop(columns=FLAG_COLUMNS).to_numpy(dtype=float)
    assert not np.isinf(numeric).any()
    assert int(out["ea_levels_selected"].max()) <= int(out["ea_levels_registered"].max())


def test_random_walks_end_to_end():
    """随机游走在所有预设下都能跑通，且选出数不超过上限。"""
    rng = np.random.default_rng(7)
    n = 260
    close = 100 + np.cumsum(rng.normal(0, 0.7, n))
    frame = _frame(close, spread=0.6)
    for preset in ("Clean", "Intraday", "Structure", "Everything"):
        out = edge_atlas(frame, preset=preset)
        assert out.index.equals(frame.index)
        assert int(out["ea_levels_selected"].max()) <= 24
        assert out["ea_levels_registered"].notna().all()

    custom = edge_atlas(
        frame,
        preset="Custom",
        enable_sessions=True,
        asia_enable=True,
        london_enable=True,
        ny_enable=True,
        enable_pivots=True,
        enable_fib=True,
        enable_sr=True,
        enable_round=True,
        enable_vwap=True,
        enable_manual=True,
        manual1_enable=True,
        manual1_price=float(close[-1]),
        enable_opening_ranges=True,
        show_trend_curves=True,
        trend_as_levels=True,
    )
    assert custom["ea_trend_fast"].notna().sum() > 0
    assert int(custom["ea_levels_selected"].max()) <= 24
    assert custom["ea_asia_high"].notna().any()
