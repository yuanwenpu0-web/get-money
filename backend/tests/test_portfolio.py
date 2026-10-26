"""组合回测引擎测试：曲线对齐、等权数学、失败跳过、指标口径、黑名单规则。
全部 mock run_backtest，不访问网络、不读真实行情。"""

import pandas as pd
import pytest

from app.backtest import portfolio as portfolio_engine
from app.backtest.metrics import metrics_from_equity


def daily_curve(equities, start="2026-01-01"):
    index = pd.date_range(start, periods=len(equities), freq="1D", tz="UTC")
    return [{"time": str(ts), "equity": float(value)} for ts, value in zip(index, equities)]


def curve_series(curve):
    index = pd.to_datetime([point["time"] for point in curve], utc=True)
    return pd.Series([float(point["equity"]) for point in curve], index=index)


def fake_run_backtest(plans, failures=None):
    """plans: {(symbol, strategy): [equity...]}; failures: {(symbol, strategy): Exception}"""
    failures = failures or {}

    def run(db, symbol, interval, strategy, days=30, initial_cash=None, fee=None,
            slippage=None, train_ratio=0.75, threshold=None, optimize=False, end_ms=None):
        key = (symbol, strategy)
        if key in failures:
            raise failures[key]
        curve = daily_curve(plans[key])
        metrics = metrics_from_equity(curve_series(curve), [], float(initial_cash or 0), interval)
        return {"metrics": metrics, "equity_curve": curve, "trades": []}

    return run


def run_engine(monkeypatch, plans, symbols, strategies, initial_cash=1000.0, failures=None):
    monkeypatch.setattr(portfolio_engine, "run_backtest", fake_run_backtest(plans, failures))
    return portfolio_engine.run_portfolio_backtest(
        None, symbols, "1h", strategies, days=200, fee=0.0002, slippage=0.0002,
        initial_cash=initial_cash,
    )


def test_curve_alignment_outer_join_ffill_and_pre_start_fill():
    legs = [
        {"initial": 100.0, "curve": daily_curve([100, 101, 102, 103, 104], start="2026-01-01")},
        {"initial": 50.0, "curve": daily_curve([50, 60, 70, 80], start="2026-01-03")},
    ]
    aligned = portfolio_engine.align_portfolio_equity(legs)
    # legB 起点（01-03）之前按初始 50 填充；legA 在 01-06 无数据按 104 前向填充
    assert list(aligned.values) == pytest.approx([150.0, 151.0, 152.0, 163.0, 174.0, 184.0])
    assert list(aligned.index.strftime("%m-%d")) == ["01-01", "01-02", "01-03", "01-04", "01-05", "01-06"]


def test_equal_weight_portfolio_return_equals_mean_symbol_return(monkeypatch):
    plans = {
        ("AAAUSDT", "trend_follow"): [1000 * (1 + 0.10 * i / 9) for i in range(10)],
        ("BBBUSDT", "trend_follow"): [1000 * (1 - 0.05 * i / 9) for i in range(10)],
        ("CCCUSDT", "trend_follow"): [1000 * (1 + 0.20 * i / 9) for i in range(10)],
    }
    result = run_engine(monkeypatch, plans, ["AAAUSDT", "BBBUSDT", "CCCUSDT"], ["trend_follow"],
                        initial_cash=3000.0)
    assert result["ok"] is True
    assert result["params"]["leg_cash"] == 1000.0  # 3000 / (3 币种 × 1 策略)
    # 单币种收益 10% / -5% / 20% → 均值 8.3333%，等权组合收益应与之相等
    assert result["diversification"]["mean_symbol_return_pct"] == pytest.approx(8.3333, abs=0.01)
    assert result["portfolio"]["total_return_pct"] == pytest.approx(8.3333, abs=0.01)
    assert [row["symbol"] for row in result["symbols"]] == ["CCCUSDT", "AAAUSDT", "BBBUSDT"]


def test_single_symbol_failure_skipped_without_blocking(monkeypatch):
    plans = {("AAAUSDT", "trend_follow"): [1000, 1100], ("CCCUSDT", "trend_follow"): [1000, 1050]}
    failures = {("BBBUSDT", "trend_follow"): ValueError("回测样本不足（100 行），至少需要 800 根有效K线")}
    result = run_engine(monkeypatch, plans, ["AAAUSDT", "BBBUSDT", "CCCUSDT"], ["trend_follow"],
                        initial_cash=3000.0, failures=failures)
    assert result["ok"] is True
    assert result["errors"] == [{
        "symbol": "BBBUSDT", "strategy": "trend_follow",
        "error": "回测样本不足（100 行），至少需要 800 根有效K线",
    }]
    # 失败腿资金不计入组合：2000 / 3000
    assert result["params"]["portfolio_initial"] == 2000.0
    assert [row["symbol"] for row in result["symbols"]] == ["AAAUSDT", "CCCUSDT"]


def test_all_legs_fail_returns_ok_false(monkeypatch):
    failures = {("AAAUSDT", "trend_follow"): ValueError("本地无K线数据")}
    result = run_engine(monkeypatch, {}, ["AAAUSDT"], ["trend_follow"], failures=failures)
    assert result["ok"] is False
    assert result["errors"] and "本地无K线数据" in result["errors"][0]["error"]


def test_portfolio_metrics_match_single_symbol_convention(monkeypatch):
    # 单腿组合：1000 → 1200 → 900 → 1000；峰值 1200，回撤 (900-1200)/1200 = -25%
    plans = {("AAAUSDT", "trend_follow"): [1000.0, 1200.0, 900.0, 1000.0]}
    result = run_engine(monkeypatch, plans, ["AAAUSDT"], ["trend_follow"], initial_cash=1000.0)
    assert result["portfolio"]["max_drawdown_pct"] == -25.0
    assert result["portfolio"]["total_return_pct"] == 0.0
    # 口径一致：与直接对同一条对齐曲线调用 metrics_from_equity 的结果相同
    series = portfolio_engine.align_portfolio_equity(
        [{"initial": 1000.0, "curve": daily_curve(plans[("AAAUSDT", "trend_follow")])}]
    )
    expected = metrics_from_equity(series, [], 1000.0, "1h")
    assert result["portfolio"]["sharpe"] == expected["sharpe"]
    assert result["portfolio"]["max_drawdown_pct"] == expected["max_drawdown_pct"]
    assert result["portfolio"]["total_fees"] == expected["total_fees"]


def test_multi_strategy_splits_symbol_allocation(monkeypatch):
    plans = {("AAAUSDT", "trend_follow"): [500, 550], ("AAAUSDT", "mean_reversion"): [500, 510]}
    result = run_engine(monkeypatch, plans, ["AAAUSDT"], ["trend_follow", "mean_reversion"],
                        initial_cash=1000.0)
    assert result["params"]["leg_cash"] == 500.0  # 1000 / (1 币种 × 2 策略)
    row = result["symbols"][0]
    assert row["initial"] == 1000.0 and row["final_equity"] == 1060.0  # 550 + 510
    assert row["return_pct"] == pytest.approx(6.0)
    assert result["portfolio"]["total_return_pct"] == pytest.approx(6.0)
    assert set(row["strategies"]) == {"trend_follow", "mean_reversion"}


def test_blacklist_flag_requires_negative_expectancy_and_enough_trades(monkeypatch):
    plans = {
        ("BADUSDT", "trend_follow"): [1000, 980],
        ("SMALLLOSS", "trend_follow"): [1000, 995],
        ("GOODUSDT", "trend_follow"): [1000, 1020],
    }

    def run(db, symbol, interval, strategy, **kwargs):
        curve = daily_curve(plans[(symbol, strategy)])
        count = 25 if symbol == "BADUSDT" else (5 if symbol == "SMALLLOSS" else 25)
        pnl = -1.0 if symbol in ("BADUSDT", "SMALLLOSS") else 1.0
        trades = [
            {"pnl": pnl, "fees": 0.1, "entry_time": "2026-01-01 00:00:00+00:00",
             "exit_time": "2026-01-02 00:00:00+00:00"}
        ] * count
        metrics = metrics_from_equity(curve_series(curve), trades, kwargs["initial_cash"], interval)
        return {"metrics": metrics, "equity_curve": curve, "trades": trades}

    monkeypatch.setattr(portfolio_engine, "run_backtest", run)
    result = portfolio_engine.run_portfolio_backtest(
        None, ["BADUSDT", "SMALLLOSS", "GOODUSDT"], "1h", ["trend_follow"],
        days=200, fee=0.0002, slippage=0.0002, initial_cash=3000.0,
    )
    rows = {row["symbol"]: row for row in result["symbols"]}
    assert rows["BADUSDT"]["blacklist_candidate"] is True   # 负期望 + 25 笔
    assert rows["SMALLLOSS"]["blacklist_candidate"] is False  # 负期望但仅 5 笔，样本不足
    assert rows["GOODUSDT"]["blacklist_candidate"] is False
    assert rows["BADUSDT"]["expectancy"] == pytest.approx(-1.0)


def test_diversification_reports_concentration_and_positive_share(monkeypatch):
    plans = {
        ("AAAUSDT", "trend_follow"): [1000, 1200],
        ("BBBUSDT", "trend_follow"): [1000, 1100],
        ("CCCUSDT", "trend_follow"): [1000, 900],
        ("DDDUSDT", "trend_follow"): [1000, 1000],
    }
    result = run_engine(monkeypatch, plans, [symbol for symbol, _ in plans], ["trend_follow"],
                        initial_cash=4000.0)
    diversification = result["diversification"]
    assert diversification["symbol_count"] == 4
    assert diversification["positive_share_pct"] == 50.0  # +20% / +10% / -10% / 0%
    assert diversification["return_std"] > 0
    top3 = {item["symbol"] for item in diversification["top3"]}
    bottom3 = {item["symbol"] for item in diversification["bottom3"]}
    assert "AAAUSDT" in top3 and "CCCUSDT" in bottom3
    # 盈亏贡献：+200 / +100 / -100 / 0，毛量 400 → AAA 占 50%
    aaa = next(row for row in result["symbols"] if row["symbol"] == "AAAUSDT")
    assert aaa["contribution_pct"] == pytest.approx(50.0)


def test_empty_inputs_rejected():
    with pytest.raises(ValueError):
        portfolio_engine.run_portfolio_backtest(
            None, [], "1h", ["trend_follow"], days=200, fee=0.0002, slippage=0.0002, initial_cash=1000.0
        )
    with pytest.raises(ValueError):
        portfolio_engine.run_portfolio_backtest(
            None, ["AAAUSDT"], "1h", [], days=200, fee=0.0002, slippage=0.0002, initial_cash=1000.0
        )
    with pytest.raises(ValueError):
        portfolio_engine.run_portfolio_backtest(
            None, ["AAAUSDT"], "1h", ["trend_follow"], days=200, fee=0.0002, slippage=0.0002, initial_cash=0.0
        )
