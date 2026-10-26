"""迭代6滚动复验脚本的离线单测：窗口终点生成、失败跳过、汇总统计、鲁棒性判定。

不访问真实网络/DB：run_portfolio_backtest 经 monkeypatch 替换，脚本以文件方式加载。"""

import importlib.util
import statistics
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rolling_validation.py"
_spec = importlib.util.spec_from_file_location("rolling_validation_script", _SCRIPT)
rolling_validation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rolling_validation)


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _args(symbols, windows=3, window_days=200, step_days=30, interval="1h",
          strategies=("trend_follow",), fee=0.0002, slippage=0.0002, initial_cash=10000.0):
    return _Args(symbols=symbols, windows=windows, window_days=window_days, step_days=step_days,
                 interval=interval, strategies=list(strategies), fee=fee, slippage=slippage,
                 initial_cash=initial_cash)


def test_window_ends_are_decreasing_and_count_correct():
    end_ms = 1_700_000_000_000  # 任意确定值
    ends = rolling_validation.build_window_ends(end_ms, 200, 30, 7)
    assert len(ends) == 7
    # 严格递减（越早的窗口 end 越小）
    assert ends == sorted(ends, reverse=True)
    # i=0 即原始 end；每步递减 step_days 天
    assert ends[0] == end_ms
    assert ends[1] == end_ms - 30 * 86_400_000
    assert ends[6] == end_ms - 6 * 30 * 86_400_000
    diffs = [ends[i] - ends[i + 1] for i in range(6)]
    assert all(d == 30 * 86_400_000 for d in diffs)


def test_run_windows_skips_data_insufficient_symbols(monkeypatch):
    # 模拟引擎内部逐腿跳过：整批传入，结果里排除数据不足币种并记入 errors
    def fake(db, symbols, interval, strategies, days, fee, slippage, initial_cash, end_ms=None):
        skipped = [s for s in symbols if s == "DOGEUSDT"]
        active = [s for s in symbols if s != "DOGEUSDT"]
        return {
            "ok": True,
            "portfolio": {"total_return_pct": 1.0, "max_drawdown_pct": -2.0,
                          "sharpe": 1.5, "expectancy": 0.05, "total_trades": 10},
            "diversification": {"positive_share_pct": 60.0},
            "symbols": [
                {"symbol": s, "return_pct": 1.0, "max_drawdown_pct": -2.0,
                 "total_trades": 5, "expectancy": 0.05, "win_rate": 50.0}
                for s in active
            ],
            "errors": [{"symbol": s, "strategy": "trend_follow", "error": "回测样本不足"}
                       for s in skipped],
        }

    monkeypatch.setattr(rolling_validation.portfolio_engine, "run_portfolio_backtest", fake)
    symbols = ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]
    windows, accum = rolling_validation.run_windows(None, 1_700_000_000_000, _args(symbols, windows=3))
    assert len(windows) == 3
    for w in windows:
        assert w["ok"] is True
        assert w["n_symbols"] == 2  # DOGE 跳过
        assert w["n_errors"] == 1
    summary = rolling_validation.summarize(windows)
    assert summary["total_windows"] == 3 and summary["positive_windows"] == 3


def test_run_windows_all_fail_yields_empty_ok_false(monkeypatch):
    # 模拟引擎：全部腿失败 → 返回 ok=False（run_windows 不阻塞、记空窗口）
    def fake(db, symbols, interval, strategies, days, fee, slippage, initial_cash, end_ms=None):
        return {"ok": False, "errors": [{"symbol": symbols[0], "error": "全失败"}],
                "message": "全部币种回测失败，无组合结果"}

    monkeypatch.setattr(rolling_validation.portfolio_engine, "run_portfolio_backtest", fake)
    windows, accum = rolling_validation.run_windows(
        None, 1_700_000_000_000, _args(["BTCUSDT"], windows=2))
    assert len(windows) == 2 and all(not w["ok"] for w in windows)
    assert all(w["n_symbols"] == 0 for w in windows)


def test_summarize_statistics_use_known_arrays():
    windows = [
        {"index": i, "end_date": f"2026-{i:02d}-01", "total_return_pct": r,
         "sharpe": s, "expectancy": e, "n_symbols": 10, "n_errors": 0,
         "max_drawdown_pct": -1.0, "total_trades": 100, "positive_symbol_pct": 50.0, "per_symbol": []}
        for i, (r, s, e) in enumerate(
            [(1.0, 1.2, 0.05), (2.0, 1.5, 0.06), (-0.5, -0.3, -0.01), (0.8, 1.0, 0.04),
             (1.5, 1.3, 0.05), (0.2, 0.4, 0.02), (1.1, 1.1, 0.05)])
    ]
    summary = rolling_validation.summarize(windows)
    returns = [1.0, 2.0, -0.5, 0.8, 1.5, 0.2, 1.1]
    # 脚本对均值/标准差四舍五入到 4 位小数（JSON 输出口径），测试对齐该口径
    assert summary["mean_return_pct"] == pytest.approx(round(statistics.mean(returns), 4))
    assert summary["std_return_pct"] == pytest.approx(round(statistics.pstdev(returns), 4))
    assert summary["positive_windows"] == 6
    assert summary["sharpe_positive_windows"] == 6
    assert summary["expectancy_positive_windows"] == 6
    assert summary["total_windows"] == 7
    assert summary["best_window"]["index"] == 1   # 2.0
    assert summary["worst_window"]["index"] == 2  # -0.5


def test_robustness_verdict_thresholds():
    robust = {"positive_windows": 6, "total_windows": 7,
              "expectancy_positive_windows": 6, "sharpe_positive_windows": 6}
    fragile = {"positive_windows": 3, "total_windows": 7,
               "expectancy_positive_windows": 2, "sharpe_positive_windows": 4}
    assert "稳健" in rolling_validation.robustness_verdict(robust)
    assert "脆弱" in rolling_validation.robustness_verdict(fragile)


def test_symbol_stability_tracks_cross_window():
    accum = {
        "BTCUSDT": {"returns": [1.0, 2.0, -0.5, 1.5, 0.8, 1.1, 1.0],
                    "expectancies": [0.05] * 7, "trades": 70},
        "DOGEUSDT": {"returns": [-1.0, -0.5, 0.2, -0.8, 0.1, -0.3, -0.6],
                     "expectancies": [-0.04] * 7, "trades": 35},
    }
    rows = rolling_validation.symbol_stability(accum)
    by_symbol = {row["symbol"]: row for row in rows}
    assert by_symbol["BTCUSDT"]["positive_expectancy_windows"] == 7
    assert by_symbol["DOGEUSDT"]["positive_expectancy_windows"] == 0
    assert rows[0]["symbol"] == "BTCUSDT"  # 期望全正者排前
