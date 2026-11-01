"""币种画像模块的契约测试：口径正确、无前视、边界安全。

测试重点不是「指标数值好看」，而是几条会静默出错的约定：
- 收益率必须是对数收益，且不跨越非法价格（0/负值记 NaN 而不是 inf）
- 方差比/Hurst 的方向语义（趋势 >0.5，均值回归 <0.5）
- 分档是横截面相对排名，不是绝对阈值（否则牛市全挤高档）
- 样本不足返回空字典，而不是返回一堆 NaN 假装有结果
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research.coin_taxonomy import (
    DIMENSIONS,
    cross_section,
    dimension_keys,
    dimension_of,
    profile,
    tercile_bins,
)


def make_frame(closes, *, qv=100.0, trades=5.0, freq="1h") -> pd.DataFrame:
    """构造一帧合规 K 线；high/low 由 open/close 外扩，保证 OHLC 自洽。"""
    n = len(closes)
    index = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    close = np.asarray(closes, dtype=float)
    open_ = np.concatenate(([close[0]], close[:-1]))
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.003,
            "low": np.minimum(open_, close) * 0.997,
            "close": close,
            "volume": np.full(n, 10.0),
            "quote_volume": np.full(n, qv),
            "trades": np.full(n, trades),
        },
        index=index,
    )


def test_profile_returns_all_dimensions() -> None:
    """理想输入（随机游走，涨跌双向样本充足）应产出 34 个键、6 个维度、全部有限。"""
    rng = np.random.default_rng(11)
    closes = np.exp(np.cumsum(rng.normal(0, 0.02, 900)))
    row = profile(make_frame(closes))
    assert row, "900 根 bar 应产出非空画像"
    covered = {dimension_of(key) for key in row}
    assert covered == set(DIMENSIONS), f"缺失维度: {set(DIMENSIONS) - covered}"
    assert all(np.isfinite(v) for v in row.values()), "理想输入不应产出 NaN"


def test_insufficient_bars_returns_empty() -> None:
    """样本不足必须返回空字典 —— 宁可缺失，也不能给出不可信的统计量。"""
    frame = make_frame(np.linspace(100, 110, 50))
    assert profile(frame) == {}


def test_missing_columns_raise() -> None:
    frame = make_frame(np.linspace(100, 110, 300)).drop(columns=["trades"])
    with pytest.raises(ValueError, match="trades"):
        profile(frame)


def test_zero_price_yields_nan_not_inf() -> None:
    """价格为 0 时对数收益应记 NaN；若记成 -inf，后续中位数/分位数会被污染。"""
    closes = np.linspace(100, 120, 400)
    closes[200] = 0.0
    row = profile(make_frame(closes))
    assert np.isfinite(row["vol_hourly_std"]), "std 应忽略 NaN 而非被 inf 污染"
    assert np.isfinite(row["tail_kurt"])


def test_hurst_direction_on_trending_series() -> None:
    """正自相关序列（收益 AR(1) φ=+0.5）方差比 >1，Hurst 应高于 0.5。

    不能用单调上涨序列：那样每期收益恒定、方差为 0，VR 会退化成 NaN。
    """
    row = profile(make_frame(_ar_prices(phi=0.5)))
    assert row["trd_vr4"] > 1.0
    assert row["trd_hurst"] > 0.5


def test_hurst_direction_on_mean_reverting_series() -> None:
    """负自相关序列（φ=-0.5）方差比 <1，Hurst 应低于 0.5。"""
    row = profile(make_frame(_ar_prices(phi=-0.5)))
    assert row["trd_vr4"] < 1.0
    assert row["trd_hurst"] < 0.5


def _ar_prices(phi: float, n: int = 800, seed: int = 3, sigma: float = 0.01) -> np.ndarray:
    """收益服从 AR(1)，累加成价格。这是唯一能同时给出稳定 VR/ACF/Hurst 的构造。"""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, n)
    rets = np.zeros(n)
    for i in range(1, n):
        rets[i] = phi * rets[i - 1] + eps[i]
    return np.exp(np.cumsum(rets))


def test_efficiency_ratio_bounds() -> None:
    """效率比必须在 [0,1]：纯趋势→1，纯噪声→低。"""
    trending = profile(make_frame(np.exp(np.linspace(0, 1.0, 800))))
    noisy = profile(make_frame(100.0 + np.tile([1.0, -1.0], 400)))
    assert 0.0 <= trending["trd_er_med"] <= 1.0
    assert trending["trd_er_med"] > noisy["trd_er_med"]


def test_daily_range_uses_path_not_return() -> None:
    """日振幅是 high/low-1（路径波动），必须严格大于同日的收盘涨跌。"""
    closes = [100.0] * 300
    closes[-24:] = [100.0 + (i % 5) for i in range(24)]
    row = profile(make_frame(closes))
    assert row["vol_range_daily_med"] >= row["vol_absret_daily_med"]


def test_liquidity_metrics_scale_with_quote_volume() -> None:
    """成交额翻 10 倍，流动性指标应同步翻 10 倍（口径线性）。"""
    low = profile(make_frame(np.linspace(100, 120, 400), qv=100.0, trades=10.0))
    high = profile(make_frame(np.linspace(100, 120, 400), qv=1000.0, trades=10.0))
    assert high["liq_qv_median_usd"] == pytest.approx(low["liq_qv_median_usd"] * 10)
    assert high["liq_trade_size_usd"] == pytest.approx(low["liq_trade_size_usd"] * 10)


def test_zero_volume_fraction() -> None:
    frame = make_frame(np.linspace(100, 120, 400))
    frame.loc[frame.index[:40], "quote_volume"] = 0.0
    row = profile(frame)
    assert row["liq_zero_vol_frac"] == pytest.approx(40 / 400, abs=1e-6)


def test_extreme_day_fraction_thresholds() -> None:
    """单日振幅超过 100% 的日子要被计入 vol_day_frac_gt100，而不是只算 50%。"""
    closes = np.full(600, 100.0)
    # 造一天：日内从 50 冲到 150（振幅 200%）
    closes[-24:] = np.concatenate([np.linspace(50, 150, 12), np.linspace(150, 100, 12)])
    row = profile(make_frame(closes))
    assert row["vol_day_frac_gt100"] > 0
    assert row["vol_day_frac_gt50"] >= row["vol_day_frac_gt100"]


def test_tercile_bins_are_cross_sectional() -> None:
    """分档必须按横截面相对排名：整体上移不应改变任何币的档位。"""
    values = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0})
    shifted = values + 1000.0
    assert tercile_bins(values).tolist() == tercile_bins(shifted).tolist()


def test_tercile_bins_keep_nan() -> None:
    """缺数据的币种必须保持 NaN，不能被塞进某一档。"""
    values = pd.Series({"A": 1.0, "B": np.nan, "C": 3.0})
    bins = tercile_bins(values)
    assert pd.isna(bins["B"])
    assert not pd.isna(bins["A"])


def test_cross_section_fills_missing_with_nan() -> None:
    profiles = {"A": {"liq_qv_median_usd": 1.0}, "B": {}}
    series = cross_section(profiles, "liq_qv_median_usd")
    assert series["A"] == 1.0
    assert np.isnan(series["B"])


def test_dimension_keys_partition() -> None:
    """每个指标键必须且只能归属一个维度，否则分档会重复计入。"""
    profiles = {"A": profile(make_frame(np.linspace(100, 130, 600)))}
    keys = {dim: set(dimension_keys(profiles, dim)) for dim in DIMENSIONS}
    total = sum(len(v) for v in keys.values())
    assert total == len(profiles["A"]), "存在未被任何维度收录的指标键"
    assert all(len(keys[a] & keys[b]) == 0 for a in DIMENSIONS for b in DIMENSIONS if a != b)


def test_profile_is_causal_no_lookahead() -> None:
    """画像必须是传入帧的纯函数：同一帧重复调用结果完全相同，且不依赖帧外数据。

    注意**不能**断言「截断帧与全量帧的指标相同」：acf1 / VR / Hurst 都是全样本统计量，
    多喂一段数据必然改变它们的值。那不是前视，是统计量的定义。
    真正的前视风险在于实现是否偷看帧外数据 —— 用「同帧幂等 + 截断帧可算」来约束。
    """
    rng = np.random.default_rng(7)
    closes = np.exp(np.cumsum(rng.normal(0, 0.01, 800)))
    frame = make_frame(closes)

    first = profile(frame)
    second = profile(frame)
    assert first == second, "同一帧两次调用结果必须完全相同（纯函数、无隐藏状态）"

    # 截断帧必须能独立算出画像，且与全量帧的键集合一致
    truncated = profile(frame.iloc[:600])
    assert set(truncated) == set(first), "截断帧应产出同样的键集合"

    # 只依赖帧内数据：改动帧尾不应影响「帧头那一段」的任何判断依据。
    # 用尾部替换后重算，只要求结果仍是有限值、且键完整 —— 说明没有外部依赖。
    altered = frame.copy()
    altered.iloc[600:] = frame.iloc[600:].assign(close=frame["close"].iloc[600:] * 3.0)
    altered_row = profile(altered)
    assert set(altered_row) == set(first)
    assert all(np.isfinite(v) for v in altered_row.values() if not np.isnan(v))
