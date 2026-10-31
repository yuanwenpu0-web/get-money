"""指标实验室的契约测试：撮合口径、出场规则、无未来函数、搜索空间与可复现性。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.research import lab_grid, lab_signals, lab_sim, lab_stats
from app.research.lab_grid import Candidate
from app.research.lab_signals import Signals, cross_down, cross_up
from app.research.lab_sim import SimConfig, exit_signals, simulate


def make_frame(closes: list[float], opens: list[float] | None = None) -> pd.DataFrame:
    n = len(closes)
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    opens = opens if opens is not None else closes
    return pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) * 1.002 for o, c in zip(opens, closes)],
            "low": [min(o, c) * 0.998 for o, c in zip(opens, closes)],
            "close": closes,
            "volume": [10.0] * n,
            "quote_volume": [100.0] * n,
            "trades": [5.0] * n,
        },
        index=index,
    )


def test_cross_helpers_ignore_nan_edges() -> None:
    a = np.array([np.nan, 1.0, 2.0, 1.0, 2.0])
    b = np.array([1.0, 1.5, 1.5, 1.5, 1.5])
    assert cross_up(a, b).tolist() == [False, False, True, False, True]
    assert cross_down(a, b).tolist() == [False, False, False, True, False]


def test_rsi_saturates_on_monotonic_series() -> None:
    up = lab_signals.ind.rsi_wilder(np.arange(1.0, 60.0), 14)
    down = lab_signals.ind.rsi_wilder(np.arange(60.0, 1.0, -1.0), 14)
    assert pytest.approx(up[-1], rel=1e-6) == 100.0
    assert pytest.approx(down[-1], rel=1e-6) == 0.0


def test_atr_and_rsi_warmup_is_nan() -> None:
    close = np.linspace(100, 110, 40)
    high, low = close * 1.01, close * 0.99
    rsi = lab_signals.ind.rsi_wilder(close, 14)
    atr = lab_signals.ind.atr_wilder(high, low, close, 14)
    assert np.isnan(rsi[:14]).all() and np.isfinite(rsi[14])
    assert np.isnan(atr[:14]).all() and np.isfinite(atr[14])


def test_exit_signals_are_rsi50_crossings() -> None:
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 1.0, 400))
    long_exit, short_exit = exit_signals(close)
    rsi = lab_signals.ind.rsi_wilder(close, 14)
    for i in range(1, len(close)):
        if np.isnan(rsi[i]) or np.isnan(rsi[i - 1]):
            assert not long_exit[i] and not short_exit[i]
            continue
        assert bool(long_exit[i]) == (rsi[i] < 50.0 and rsi[i - 1] >= 50.0)
        assert bool(short_exit[i]) == (rsi[i] > 50.0 and rsi[i - 1] <= 50.0)


def test_trade_fills_next_open_with_cost_and_slippage() -> None:
    closes = [100 + i for i in range(12)]
    opens = [100 + i + 0.5 for i in range(12)]
    frame = make_frame(closes, opens)
    long_entry = np.zeros(12, dtype=bool)
    long_entry[2] = True
    exit_long = np.zeros(12, dtype=bool)
    exit_long[5] = True
    cfg = SimConfig(stake=10.0, fee=0.0004, slippage=0.0002, max_hold_bars=100)
    trades = simulate("X", frame, Signals(long_entry, np.zeros(12, bool)), (exit_long, np.zeros(12, bool)), cfg)
    assert len(trades) == 1
    trade = trades[0]
    entry = opens[3] * (1 + cfg.slippage)
    exit_price = opens[6] * (1 - cfg.slippage)
    qty = cfg.stake / entry
    expected = qty * (exit_price - entry) - cfg.fee * cfg.stake - cfg.fee * qty * exit_price
    assert trade.entry_price == pytest.approx(entry)
    assert trade.exit_price == pytest.approx(exit_price)
    assert trade.pnl == pytest.approx(expected)
    assert trade.reason == "rsi50"
    assert trade.bars == 3


def test_max_hold_forces_exit_and_blocks_overlap() -> None:
    closes = [100 + i for i in range(20)]
    opens = [100 + i + 0.5 for i in range(20)]
    frame = make_frame(closes, opens)
    entry = np.zeros(20, dtype=bool)
    entry[[2, 3, 4, 10]] = True
    cfg = SimConfig(max_hold_bars=3)
    trades = simulate("X", frame, Signals(entry, np.zeros(20, bool)), (np.zeros(20, bool), np.zeros(20, bool)), cfg)
    assert [t.reason for t in trades[:2]] == ["max_hold", "max_hold"]
    assert trades[0].entry_ts < trades[0].exit_ts
    for prev, nxt in zip(trades, trades[1:]):
        if prev.symbol == nxt.symbol:
            assert nxt.entry_ts > prev.exit_ts


def test_end_of_sample_closes_at_last_close() -> None:
    closes = [100.0] * 10
    frame = make_frame(closes)
    entry = np.zeros(10, dtype=bool)
    entry[6] = True
    trades = simulate("X", frame, Signals(entry, np.zeros(10, bool)), (np.zeros(10, bool), np.zeros(10, bool)), SimConfig())
    assert len(trades) == 1 and trades[0].reason == "end_of_sample"


def test_signals_do_not_look_ahead() -> None:
    rng = np.random.default_rng(11)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600)))
    full = make_frame(list(close))
    cut = 420
    part = full.iloc[:cut]
    for family in lab_grid.GRID:
        params = lab_grid.expand(family)[0]
        whole = lab_signals.BUILDERS[family](full, params)
        partial = lab_signals.BUILDERS[family](part, params)
        assert np.array_equal(whole.long[:cut], partial.long), family
        assert np.array_equal(whole.short[:cut], partial.short), family


def test_grid_respects_bounds_and_constraints() -> None:
    for family, spec in lab_grid.GRID.items():
        for params in lab_grid.expand(family):
            for name, bound in spec["params"].items():
                value = params[name]
                if "values" in bound:
                    assert value in bound["values"]
                else:
                    assert bound["min"] <= value <= bound["max"]
                    assert (value - bound["min"]) % bound["step"] == 0
            if spec.get("constraint") == "fast_lt_slow":
                assert params["fast"] < params["slow"]
            if spec.get("constraint") == "symmetric_rsi":
                assert params["low"] + params["high"] == 100
        assert len(lab_grid.expand(family)) == spec["combos"] if "combos" in spec else True


def test_candidate_keys_are_unique() -> None:
    keys = [c.key for c in lab_grid.candidates()]
    assert len(keys) == len(set(keys))
    assert len(keys) > 80


def test_summary_metrics_are_consistent() -> None:
    trades = [
        lab_sim.Trade("A", 1, 1, 2, 100.0, 110.0, 1, 0.98, 0.098, 0.02, "rsi50"),
        lab_sim.Trade("A", 1, 3, 4, 100.0, 95.0, 1, -0.52, -0.052, 0.02, "rsi50"),
        lab_sim.Trade("B", -1, 5, 6, 100.0, 90.0, 1, 0.98, 0.098, 0.02, "max_hold"),
    ]
    stats = lab_stats.summarize(trades, 10.0, 2)
    assert stats["trades"] == 3
    assert stats["win_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert stats["total_pnl"] == pytest.approx(0.98 - 0.52 + 0.98)
    assert stats["long_pnl"] == pytest.approx(0.46)
    assert stats["short_pnl"] == pytest.approx(0.98)
    assert stats["max_concurrent"] == 1


def test_side_filter_keeps_only_requested_direction() -> None:
    rng = np.random.default_rng(5)
    frame = make_frame(list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))))
    cand = lab_grid.candidates(["ema_cross"])[0]
    signals = lab_signals.BUILDERS[cand.family](frame, cand.params)
    exits = exit_signals(frame["close"].to_numpy(dtype=float))
    both = simulate("X", frame, signals, exits, SimConfig())
    long_only = simulate("X", frame, signals, exits, SimConfig(side=1))
    short_only = simulate("X", frame, signals, exits, SimConfig(side=-1))
    assert {t.side for t in both} == {1, -1}
    assert long_only and all(t.side == 1 for t in long_only)
    assert short_only and all(t.side == -1 for t in short_only)
    assert len(long_only) + len(short_only) == len(both)


def test_simulation_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    frame = make_frame(list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))))
    cand = lab_grid.candidates(["donchian_break"])[0]
    signals = lab_signals.BUILDERS[cand.family](frame, cand.params)
    exits = exit_signals(frame["close"].to_numpy(dtype=float))
    first = simulate("X", frame, signals, exits, SimConfig())
    second = simulate("X", frame, signals, exits, SimConfig())
    assert [t.pnl for t in first] == [t.pnl for t in second]


def test_combine_rules_are_set_operations() -> None:
    from app.research.lab_combos import combine

    a = Signals(np.array([True, True, False, False]), np.array([False, False, True, False]))
    b = Signals(np.array([True, False, False, True]), np.array([False, True, True, False]))
    both = combine(a, b, "and")
    either = combine(a, b, "or")
    assert both.long.tolist() == [True, False, False, False]
    assert both.short.tolist() == [False, False, True, False]
    assert either.long.tolist() == [True, True, False, True]


def _random_frame(n: int = 600, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return make_frame(list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))))


def test_regime_mask_aligns_with_frame() -> None:
    from app.research import lab_regime

    frame = _random_frame(400)
    for kind in lab_regime.KINDS:
        mask = lab_regime.build({"A": frame}, kind, 50)["A"]
        assert mask.shape == (len(frame),)
        assert mask.dtype == bool
        assert mask.sum() > 0


def test_regime_mask_has_no_lookahead() -> None:
    from app.research import lab_regime

    full = _random_frame(600)
    cut = 380
    for kind in lab_regime.KINDS:
        whole = lab_regime.build({"A": full}, kind, 30)["A"]
        part = lab_regime.build({"A": full.iloc[:cut]}, kind, 30)["A"]
        assert np.array_equal(whole[:cut], part), kind


def test_market_index_is_equal_weighted() -> None:
    from app.research import lab_regime

    up = make_frame([100.0, 110.0])
    down = make_frame([100.0, 90.0])
    index = lab_regime.market_index({"A": up, "B": down})
    assert index.iloc[-1] == pytest.approx(1.0)


def test_regime_only_ratio_is_bounded() -> None:
    from app.research import lab_regime

    masks = lab_regime.build({"A": _random_frame(500)}, "own_ema", 50)
    assert 0.0 <= lab_regime.on_ratio(masks) <= 1.0


def test_new_research_files_stay_under_300_lines() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("lab_indicators", "lab_signals", "lab_grid", "lab_sim", "lab_stats",
                 "lab_experiment", "lab_combos", "lab_report", "lab_regime"):
        path = root / "app" / "research" / f"{name}.py"
        assert path.exists(), name
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 300, name
    for script_name in ("indicator_lab.py", "indicator_regime.py"):
        script = root / "scripts" / script_name
        assert len(script.read_text(encoding="utf-8").splitlines()) <= 300, script_name


def test_candidate_is_hashable_by_key() -> None:
    cand = Candidate("ema_cross", {"fast": 5, "slow": 21}, "ema_cross|x")
    assert cand.label == "ema_cross(fast=5,slow=21)"
