"""组合级回测引擎的纯函数与聚合口径。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.research import combo_engine as ce


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")


def test_segment_bounds_split_by_bar_ratio():
    bounds = ce.segment_bounds(_index(1000))
    assert bounds["train"][0] == _index(1000)[0]
    assert bounds["train"][1] == _index(1000)[600]
    assert bounds["valid"][1] == _index(1000)[800]
    assert bounds["test"][1] == _index(1000)[999]


def test_by_segment_drops_trades_spanning_boundaries():
    index = _index(100)
    bounds = ce.segment_bounds(index)
    crossing = {"t0": index[50], "t1": index[95], "ret": 1.0, "reason": ""}
    inside = {"t0": index[10], "t1": index[20], "ret": 2.0, "reason": ""}
    buckets = ce.by_segment([crossing, inside], bounds)
    assert buckets["train"] == [2.0]
    assert buckets["valid"] == [] and buckets["test"] == []
    assert sum(len(values) for values in buckets.values()) == 1, "跨段样本必须被清除"


def test_pooled_marks_insufficient_below_threshold():
    few = ce.pooled([0.5] * 10 + [-0.2] * 5)
    assert few["trades"] == 15 and few["verdict"] == "insufficient"
    many = ce.pooled([0.5] * 60)
    assert many["verdict"] == "positive" and many["ci_low"] > 0
    assert set(many) >= {"trades", "mean", "ci_low", "ci_high", "verdict"}


def test_pick_skips_sparse_cells():
    rows = [{"train": [1.0] * 400, "idx": 0}, {"train": [0.5] * 400, "idx": 1},
            {"train": [20.0] * 12, "idx": 2}]
    chosen = ce.pick(rows)
    assert chosen is not None, "中位数 400 时 12 笔的格子应被剔除，剩下两个仍可选"
    assert chosen["idx"] == 0, "稀疏格子即使均值最高也不能当冠军"


def test_pick_returns_none_when_all_cells_sparse():
    assert ce.pick([{"train": [1.0] * 5, "idx": 0}]) is None


def test_generic_trades_aggregates_partial_legs(monkeypatch):
    """分批止盈的三条腿必须合并成一笔仓位，否则 1/6 仓位的止盈腿与 5/6 仓位的止损腿等权。"""
    frame = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "atr_pct": 0.01, "volume": 1.0},
        index=_index(3),
    )
    legs = [
        {"entry_time": "2026-01-01 00:00:00+00:00", "position_id": "p1", "exit_time": "2026-01-01 01:00:00+00:00",
         "entry_price": 100.0, "size": 1.0, "pnl": 10.0, "return_pct": 10.0, "status": "take_profit_1", "margin": 100.0},
        {"entry_time": "2026-01-01 00:00:00+00:00", "position_id": "p1", "exit_time": "2026-01-01 02:00:00+00:00",
         "entry_price": 100.0, "size": 5.0, "pnl": -20.0, "return_pct": -4.0, "status": "stop_loss", "margin": 500.0},
    ]
    monkeypatch.setattr(ce, "simulate", lambda *a, **k: (pd.Series([1.0]), legs))
    monkeypatch.setattr(ce, "apply_exit_overlay", lambda frame, signals, params: signals)
    monkeypatch.setattr(ce, "get_spec", lambda code: type("S", (), {"signals": staticmethod(lambda d, p: d)}))

    trades = ce.generic_trades(frame, "fake", {}, fee=0.0, slippage=0.0, initial=1000.0, interval="1h")
    assert len(trades) == 1, "两条腿必须合并为一笔"
    assert abs(trades[0]["ret"] - (-10.0 / 600.0 * 100)) < 1e-9, "按总名义本金折算，不是各腿等权平均"
    assert trades[0]["reason"] == "stop_loss"


def test_random_entry_returns_is_pooled_and_deterministic():
    opens = [100.0 * (1 + 0.001 * np.sin(i / 7.0)) for i in range(800)]
    first = ce.random_entry_returns([np.asarray(opens)], 48, 5, n_seeds=4, warmup=200)
    second = ce.random_entry_returns([np.asarray(opens)], 48, 5, n_seeds=4, warmup=200)
    assert first == second, "同种子必须可复现"
    assert len(first) == 4 and all(np.isfinite(value) for value in first)


def test_buy_hold_matches_segment_return():
    index = _index(100)
    frame = pd.DataFrame({"close": np.linspace(100, 200, 100)}, index=index)
    bounds = ce.segment_bounds(index)
    out = ce.buy_hold(frame, bounds)
    assert out["train"] > 0 and out["test"] > 0
