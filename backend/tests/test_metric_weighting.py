"""迭代8-B/C：动态仓位加权 + regime 自适应策略权重 + portfolio 接线的离线单测。

不访问真实网络/真实 DB：组合回测的 run_backtest 用合成曲线替换，存储用 :memory: sqlite。
默认参数（weight_mode=equal）行为必须与加权前完全一致，由不变式测试守住。
"""

import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest

from app.backtest import portfolio as portfolio_mod
from app.backtest.weighting import (
    metric_score,
    metric_weights,
    rank_score,
    regime_weights,
    weighted_equity,
)
from app.market.metrics import save_metrics_snapshot

BASE_TS = 1_700_000_000_000
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rolling_validation.py"
_spec = importlib.util.spec_from_file_location("rolling_validation_iter8", SCRIPT)
rolling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rolling)


def _feats(divergence=0.0, crowd_z=0.0, rank=1):
    return {"sentiment_divergence": divergence, "crowd_ls_z": crowd_z,
            "market_cap_rank": rank, "fng_value": 50.0}


# ---------- ① 仓位权重 ----------

def test_metric_weights_normalized_non_negative_and_capped():
    metrics = {"A": _feats(divergence=2.0, crowd_z=0.0, rank=1),      # 高分
               "B": _feats(divergence=-2.0, crowd_z=3.0, rank=50),    # 低分
               "C": _feats(divergence=-2.0, crowd_z=3.0, rank=50),
               "D": _feats(divergence=-2.0, crowd_z=3.0, rank=50)}
    weights = metric_weights(list(metrics), metrics)
    assert len(weights) == 4
    assert all(w >= 0 for w in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)
    assert max(weights.values()) <= 2.0 / 4 + 1e-9  # 单币上限 clamp
    assert weights["A"] > weights["B"]              # 高分币种权重更高


def test_metric_weights_falls_back_to_equal_when_any_missing():
    weights = metric_weights(["A", "B"], {"A": _feats(divergence=3.0), "B": {}})
    assert weights == {"A": 0.5, "B": 0.5}


def test_metric_weights_equal_scores_give_equal_weights():
    weights = metric_weights(["A", "B", "C"], {k: _feats() for k in "ABC"})
    assert all(math.isclose(w, 1 / 3, abs_tol=1e-9) for w in weights.values())


def test_rank_score_and_metric_score_edges():
    assert rank_score(1) == 1.0
    assert rank_score(51) == 0.0
    assert rank_score(0) == 0.5      # 未知 → 中性
    assert rank_score(None) == 0.5
    assert metric_score({}) == 0.0
    assert metric_score({"sentiment_divergence": float("nan")}) != float("nan")  # 非有限收敛


# ---------- ② regime 权重 ----------

def test_regime_weights_bounds_and_sum():
    for metrics in ({"fng_value": 100.0, "crowd_ls_z": 0.0},
                    {"fng_value": 0.0, "crowd_ls_z": 3.0},
                    {"fng_value": 50.0, "crowd_ls_z": 0.0},
                    {"fng_value": 100.0, "crowd_ls_z": 3.0}):
        pair = regime_weights(metrics)
        assert pair["trend_follow"] + pair["mean_reversion"] == pytest.approx(1.0)
        assert 0.2 <= pair["trend_follow"] <= 0.8


def test_regime_weights_greed_favors_trend_fear_favors_mean_reversion():
    assert regime_weights({"fng_value": 90.0, "crowd_ls_z": 0.0})["trend_follow"] > 0.5
    assert regime_weights({"fng_value": 10.0, "crowd_ls_z": 0.0})["trend_follow"] < 0.5
    crowded = regime_weights({"fng_value": 50.0, "crowd_ls_z": 3.0})["trend_follow"]
    calm = regime_weights({"fng_value": 50.0, "crowd_ls_z": 0.0})["trend_follow"]
    assert crowded < calm  # 散户极端拥挤 → 更偏均值回归（逆向）


def test_regime_weights_clamps_to_floor():
    # fng=10（真实低值）且极端拥挤：0.5+0.25*(-0.8)-0.15=0.15 → 下限 0.2
    assert regime_weights({"fng_value": 10.0, "crowd_ls_z": 3.0})["trend_follow"] == 0.2


def test_regime_weights_neutral_when_no_metrics():
    """无指标必须等权：不得把缺失数据误判成极端情绪（否则会凭空产生倾斜）。"""
    assert regime_weights({})["trend_follow"] == 0.5
    assert regime_weights(None)["trend_follow"] == 0.5


def test_regime_weights_treats_missing_fng_as_neutral():
    # fng 缺失（0 不是真实取值）按中性 50 → 无拥挤时即 0.5
    assert regime_weights({"crowd_ls_z": 0.0})["trend_follow"] == pytest.approx(0.5)


def test_regime_weights_ceiling_and_absolute_crowd():
    # fng=100 且无拥挤 → 0.75（A=0.25 下实际达不到 0.8 上限，0.8 仅为安全边界）
    assert regime_weights({"fng_value": 100.0, "crowd_ls_z": 0.0})["trend_follow"] == pytest.approx(0.75)
    # 拥挤度取绝对值：z=-3 与 z=+3 效果一致（0.5+0.25-0.15=0.60）
    assert regime_weights({"fng_value": 100.0, "crowd_ls_z": -3.0})["trend_follow"] == pytest.approx(0.60)
    assert regime_weights({"fng_value": 100.0, "crowd_ls_z": 3.0})["trend_follow"] == pytest.approx(0.60)


# ---------- ③ weighted_equity 不变式 ----------

def test_weighted_equity_equals_buy_and_hold_when_single_period_equal_weights():
    index = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z"])
    aligned = pd.DataFrame({"leg0": [500.0, 550.0, 605.0], "leg1": [500.0, 500.0, 500.0]}, index=index)
    out = weighted_equity(aligned, 1000.0, rebalance_bars=10_000, weight_lookup=lambda ts: None)
    pd.testing.assert_series_equal(out, aligned.sum(axis=1).rename(None), check_names=False)


def test_weighted_equity_applies_skewed_weights():
    index = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"])
    aligned = pd.DataFrame({"leg0": [500.0, 600.0], "leg1": [500.0, 500.0]}, index=index)
    skewed = weighted_equity(aligned, 1000.0, 10_000, lambda ts: {"leg0": 0.9, "leg1": 0.1})
    # leg0 涨 20%、leg1 持平 → 0.9*1.2 + 0.1*1.0 = 1.18
    assert skewed.iloc[-1] == pytest.approx(1180.0)


# ---------- ④ portfolio 接线 ----------

def _fake_backtest(db, symbol, interval, strategy, days=None, initial_cash=500.0,
                   fee=0.0, slippage=0.0, optimize=False, end_ms=None):
    base = float(initial_cash)
    bars = 10
    growth = 0.01 if symbol == "BTCUSDT" else 0.0  # BTC 每根 +1%，ETH 持平
    equities = [base * (1 + growth * i) for i in range(bars)]
    times = [f"2026-01-{day:02d}T00:00:00+00:00" for day in range(1, bars + 1)]
    return {"equity_curve": [{"time": t, "equity": e} for t, e in zip(times, equities)],
            "trades": [], "metrics": {"total_return_pct": 0.0, "max_drawdown_pct": 0.0,
                                      "total_trades": 0, "expectancy": 0.0}}


@pytest.fixture
def patched_backtest(monkeypatch):
    monkeypatch.setattr(portfolio_mod, "run_backtest", _fake_backtest)


def _seed(db, btc_features, eth_features=None):
    save_metrics_snapshot(db, "BTCUSDT", btc_features, BASE_TS)
    if eth_features is not None:
        save_metrics_snapshot(db, "ETHUSDT", eth_features, BASE_TS)


def _kwargs():
    return dict(symbols=["BTCUSDT", "ETHUSDT"], interval="1h", strategies=["trend_follow"],
                days=200, fee=0.0002, slippage=0.0002, initial_cash=1000.0)


def test_default_weight_mode_equals_explicit_equal(patched_backtest, db):
    _seed(db, _feats(divergence=2.0, rank=1), _feats(divergence=-2.0, crowd_z=3.0, rank=50))
    default = portfolio_mod.run_portfolio_backtest(db, **_kwargs())
    explicit = portfolio_mod.run_portfolio_backtest(db, weight_mode="equal", **_kwargs())
    assert default["params"]["weight_mode"] == "equal"
    assert default["portfolio"]["total_return_pct"] == explicit["portfolio"]["total_return_pct"]


def test_metric_mode_outperforms_equal_when_metrics_favor_winner(patched_backtest, db):
    _seed(db, _feats(divergence=2.0, rank=1), _feats(divergence=-2.0, crowd_z=3.0, rank=50))
    equal = portfolio_mod.run_portfolio_backtest(db, **_kwargs())
    metric = portfolio_mod.run_portfolio_backtest(db, weight_mode="metric", rebalance_bars=24,
                                                  **_kwargs())
    assert metric["params"]["weight_mode"] == "metric"
    # BTC 上涨、ETH 持平；指标把权重压向 BTC → 加权组合收益高于等权
    assert metric["portfolio"]["total_return_pct"] > equal["portfolio"]["total_return_pct"]


def test_metric_mode_falls_back_to_equal_when_metrics_missing(patched_backtest, db):
    _seed(db, _feats(divergence=2.0, rank=1), eth_features=None)  # ETH 无指标
    equal = portfolio_mod.run_portfolio_backtest(db, **_kwargs())
    metric = portfolio_mod.run_portfolio_backtest(db, weight_mode="metric", **_kwargs())
    assert metric["portfolio"]["total_return_pct"] == pytest.approx(
        equal["portfolio"]["total_return_pct"])


def test_metric_weight_lookup_applies_regime_weights(db):
    save_metrics_snapshot(db, "BTCUSDT", {"fng_value": 90.0, "crowd_ls_z": 0.0}, BASE_TS)
    legs = [{"symbol": "BTCUSDT", "strategy": "trend_follow"},
            {"symbol": "BTCUSDT", "strategy": "mean_reversion"}]
    strategies = ["trend_follow", "mean_reversion"]
    weights = portfolio_mod._metric_weight_lookup(db, legs, strategies, regime_weighting=True)(BASE_TS)
    assert weights["leg0"] > weights["leg1"]          # 贪婪 → 趋势权重更高
    assert sum(weights.values()) == pytest.approx(1.0)
    off = portfolio_mod._metric_weight_lookup(db, legs, strategies, regime_weighting=False)(BASE_TS)
    assert off["leg0"] == pytest.approx(off["leg1"])  # 关闭时两策略等权


def test_lookup_skips_regime_tilt_when_any_symbol_missing_metrics(db):
    """部分币种无指标时，策略维度也必须等权（不能只对有数据的币种倾斜）。"""
    save_metrics_snapshot(db, "BTCUSDT", {"fng_value": 90.0, "crowd_ls_z": 0.0}, BASE_TS)
    legs = [{"symbol": "BTCUSDT", "strategy": "trend_follow"},
            {"symbol": "BTCUSDT", "strategy": "mean_reversion"},
            {"symbol": "ETHUSDT", "strategy": "trend_follow"},
            {"symbol": "ETHUSDT", "strategy": "mean_reversion"}]
    strategies = ["trend_follow", "mean_reversion"]
    weights = portfolio_mod._metric_weight_lookup(db, legs, strategies, regime_weighting=True)(BASE_TS)
    assert all(math.isclose(w, 0.25, abs_tol=1e-9) for w in weights.values())


def test_invalid_weight_mode_rejected():
    with pytest.raises(ValueError):
        portfolio_mod.run_portfolio_backtest(
            None, ["BTCUSDT"], "1h", ["trend_follow"], 200, 0.0002, 0.0002, 1000.0,
            weight_mode="bogus")


# ---------- ⑤ 滚动汇总统计（复用迭代6脚本） ----------

def test_summarize_mean_and_positive_window_count():
    windows = [
        {"index": 0, "end_date": "2026-10-25", "total_return_pct": 1.0, "sharpe": 0.5,
         "expectancy": 0.1, "max_drawdown_pct": 1.0},
        {"index": 1, "end_date": "2026-09-25", "total_return_pct": -0.5, "sharpe": -0.2,
         "expectancy": -0.1, "max_drawdown_pct": 1.0},
        {"index": 2, "end_date": "2026-08-25", "total_return_pct": 2.0, "sharpe": 0.9,
         "expectancy": 0.2, "max_drawdown_pct": 1.0},
    ]
    summary = rolling.summarize(windows)
    assert summary["total_windows"] == 3
    assert summary["positive_windows"] == 2
    assert summary["mean_return_pct"] == pytest.approx(round((1.0 - 0.5 + 2.0) / 3, 4))
    assert summary["best_window"]["index"] == 2
    assert summary["worst_window"]["index"] == 1
