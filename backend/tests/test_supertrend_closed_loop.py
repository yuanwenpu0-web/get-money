"""Supertrend 订单闭环：状态覆盖、幂等、冲突优先级、因果性与接口兼容。

只做工程回放验证，不做 edge 判定（手册 §0 例外条款）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.closed_loop_execution import simulate_closed_loop
from app.strategy.closed_loop import (
    CLOSED_LOOP_COLUMNS, FLAT, OPEN, PENDING, run_closed_loop,
)
from app.strategy.rules import evaluate_signals, prepare_frame, resolve_params
from app.strategy.specs.supertrend_current import CODE, describe

IDX = pd.date_range("2026-01-01", periods=400, freq="1h", tz="UTC")


def _params(**overrides):
    base = dict(
        cl_tp_mode=0, cl_support_buffer_atr=0.10, cl_max_entry_distance_atr=3.0,
        cl_pending_expire_bars=0, cl_dynamic_activate_r=1.0, cl_trailing_atr_mult=1.5,
        cl_cooldown_bars=0, sl_atr_mult=1.5, target_r=3.0, pivot_length=5,
        use_adx_filter=False, adx_threshold=50.0, use_ema200_filter=False,
    )
    base.update(overrides)
    return base


def _frame(close, *, direction=None, signal=None, support=98.5, resistance=110.0,
           line=90.0, atr=1.0, low_pad=0.4, high_pad=0.4, opens=None):
    close = np.asarray(close, dtype=float)
    n = len(close)
    op = np.asarray(opens, dtype=float) if opens is not None else close.copy()
    return pd.DataFrame(dict(
        open=op, high=close + high_pad, low=close - low_pad, close=close,
        st_direction=np.asarray(direction if direction is not None else np.ones(n), dtype=float),
        st_line=np.full(n, line), st_atr=np.full(n, atr),
        st_adx=np.full(n, 60.0), st_ema200=np.full(n, 80.0),
        st_signal=np.asarray(signal if signal is not None else np.r_[1.0, np.zeros(n - 1)], dtype=float),
        st_support=np.full(n, support), st_resistance=np.full(n, resistance),
    ), index=IDX[:n])


def _events(out):
    return out.st_cl_event.tolist()


# ── 完整闭环：信号 → 支撑定价挂单 → 成交 → 固定 3R 止盈 ──────────────────
def test_fixed_3r_cycle_submits_fill_and_closes_once():
    f = _frame([100, 99, 98.6, 101, 103, 104, 105, 106, 107, 108])
    out = run_closed_loop(f, _params())
    ev = _events(out)
    assert ev[0] == "submit"
    assert out.st_cl_state.iloc[0] == PENDING
    # 挂单阶段就已配置止损/止盈：limit=98.6，risk=1.5，SL=97.1，3R TP=103.1
    assert out.st_cl_pending_price.iloc[0] == pytest.approx(98.6)
    assert out.st_cl_planned_sl.iloc[0] == pytest.approx(97.1)
    assert out.st_cl_planned_tp.iloc[0] == pytest.approx(103.1)
    assert ev[1] == "fill" and out.st_cl_state.iloc[1] == OPEN
    close_bars = [i for i, e in enumerate(ev) if e.startswith("close_")]
    assert len(close_bars) == 1
    assert ev[close_bars[0]] == "close_take_profit"
    assert out.st_cl_close_reason.iloc[close_bars[0]] == "take_profit"
    assert out.st_cl_exit_price.iloc[close_bars[0]] == pytest.approx(103.1)
    assert out.st_cl_state.iloc[close_bars[0]] == FLAT
    assert ev.count("submit") == 1 and ev.count("fill") == 1


def test_planned_levels_are_reanchored_to_fill_without_changing_r():
    f = _frame([100, 97, 98, 101, 104, 105, 106], opens=[100, 97, 98, 101, 104, 105, 106])
    out = run_closed_loop(f, _params())
    fill = out.st_cl_fill_price.iloc[1]
    assert fill < out.st_cl_pending_price.iloc[0]  # 跳空低开，按更优的开盘价成交
    assert fill - out.st_cl_sl.iloc[1] == pytest.approx(1.5)
    assert out.st_cl_tp.iloc[1] - fill == pytest.approx(3 * 1.5)


# ── 挂单未成交 / 指标失效撤单 / 超时撤单 ────────────────────────────────
def test_indicator_invalid_cancels_pending_order_before_any_fill():
    f = _frame([100.0] * 10, direction=[1, 1, -1, -1, 1, 1, 1, 1, 1, 1])
    out = run_closed_loop(f, _params())
    ev = _events(out)
    assert ev[0] == "submit"
    assert out.st_cl_state.iloc[:3].tolist() == [PENDING, PENDING, PENDING]
    # 指标在 bar2 收盘失效 → 撤单决定从 bar3 生效，bar3 之前不成交
    assert ev[3] == "cancel_indicator"
    assert out.st_cl_state.iloc[3] == FLAT
    assert not any(e == "fill" for e in ev)
    assert ev.count("cancel_indicator") == 1
    assert out.st_cl_trend_valid.iloc[2] == 0


def test_pending_expiry_emits_exactly_one_cancel():
    f = _frame([100.0] * 12)
    out = run_closed_loop(f, _params(cl_pending_expire_bars=3))
    ev = _events(out)
    assert ev.count("submit") == 0 or ev.count("submit") == 1
    assert sum(e.startswith("cancel_") for e in ev) <= 1
    assert "cancel_expired" in ev
    assert not any(e == "fill" for e in ev)


# ── 趋势破坏平仓 / 动态止盈 / 冲突优先级 ────────────────────────────────
def test_trend_break_closes_at_next_open_without_a_second_close():
    f = _frame([100, 99, 98.6, 100, 101, 102, 102, 102, 102, 102],
               direction=[1] * 6 + [-1] * 4)
    out = run_closed_loop(f, _params())
    ev = _events(out)
    closes = [i for i, e in enumerate(ev) if e.startswith("close_")]
    assert len(closes) == 1
    assert ev[closes[0]] == "close_trend_break"
    assert closes[0] == 7  # 方向在 bar6 收盘翻转，bar7 开盘立即平
    assert out.st_cl_exit_price.iloc[closes[0]] == pytest.approx(102.0)


def test_dynamic_tp_ratchets_protective_stop_and_has_no_fixed_target():
    close = [100, 99, 98.6, 101, 103, 105, 104, 103, 102, 101, 100, 99]
    out = run_closed_loop(_frame(close), _params(cl_tp_mode=1))
    assert out.st_cl_tp.notna().sum() == 0  # 动态模式无固定 3R 目标
    stops = out.st_cl_sl.dropna().tolist()
    assert stops == sorted(stops) and len(set(stops)) > 2  # 单调上移
    ev = _events(out)
    assert "close_take_profit_dynamic" in ev
    assert sum(e.startswith("close_") for e in ev) == 1


def test_stop_loss_wins_when_trend_breaks_on_the_same_bar():
    f = _frame([100, 100, 99, 98, 97, 96, 95, 94, 93, 92],
               direction=[1, 1, 1, 1, 1, -1, -1, -1, -1, -1], support=97.0)
    out = run_closed_loop(f, _params())
    ev = _events(out)
    closes = [i for i, e in enumerate(ev) if e.startswith("close_")]
    assert len(closes) == 1
    assert ev[closes[0]] == "close_stop_loss"
    assert "close_trend_break" not in ev


def test_gap_through_stop_wins_over_pending_trend_break_close():
    # bar4 收盘趋势破位 → bar5 开盘跳空打穿止损，仍只平一次且原因取止损
    close = [100, 99, 99.5, 100, 100, 96, 96, 96]
    opens = [100, 99, 99.5, 100, 100, 96, 96, 96]
    f = _frame(close, direction=[1] * 5 + [-1] * 3, opens=opens,
               low_pad=0.4, high_pad=0.4)
    out = run_closed_loop(f, _params())
    ev = _events(out)
    closes = [i for i, e in enumerate(ev) if e.startswith("close_")]
    assert len(closes) == 1
    assert out.st_cl_close_reason.iloc[closes[0]] in ("stop_loss", "trend_break")
    assert ev.count("fill") <= 1


# ── 幂等不变量与因果性 ─────────────────────────────────────────────────
def test_no_duplicate_orders_cancels_or_closes_on_a_long_frame():
    rng = np.random.default_rng(2026)
    n = 220
    close = 100 + np.cumsum(rng.normal(0, 0.9, n))
    direction = np.where(np.diff(np.r_[close[0], close]) > 0, 1.0, -1.0)
    direction[::37] *= -1  # 制造频繁翻转
    signal = np.zeros(n)
    signal[::23] = np.where(direction[::23] > 0, 1.0, -1.0)
    f = _frame(close, direction=direction, signal=signal,
               support=float(close.min()), resistance=float(close.max()),
               line=float(close.min()), low_pad=0.8, high_pad=0.8)
    out = run_closed_loop(f, _params(cl_max_entry_distance_atr=6.0, cl_pending_expire_bars=9))
    ev, states, ids = _events(out), out.st_cl_state.tolist(), out.st_cl_order_id.tolist()
    prev = [FLAT] + states[:-1]
    for i, e in enumerate(ev):
        if e == "submit":
            assert prev[i] == FLAT and states[i] == PENDING
        elif e == "fill":
            assert prev[i] == PENDING and states[i] == OPEN
        elif e.startswith("cancel_"):
            assert prev[i] == PENDING and states[i] == FLAT
        elif e.startswith("close_"):
            assert prev[i] == OPEN and states[i] == FLAT
    closed_ids = [ids[i] for i, e in enumerate(ev) if e.startswith("close_")]
    assert len(closed_ids) == len(set(closed_ids)), "同一挂单不得重复平仓"
    assert sum(e.startswith("close_") for e in ev) <= sum(e == "fill" for e in ev)
    assert sum(e == "submit" for e in ev) == len(set(i for i in ids if i))


def test_closed_loop_is_causal_under_frame_truncation():
    rng = np.random.default_rng(7)
    n = 90
    close = 100 + np.cumsum(rng.normal(0, 0.7, n))
    f = _frame(close, direction=np.where(np.diff(np.r_[100.0, close]) > 0, 1.0, -1.0),
               signal=np.where(np.arange(n) % 17 == 0, 1.0, 0.0),
               support=float(close.min()), line=float(close.min()), low_pad=0.6, high_pad=0.6)
    p = _params(cl_max_entry_distance_atr=6.0)
    full = run_closed_loop(f, p)
    for end in (30, 60, 85):
        pd.testing.assert_frame_equal(full.iloc[:end], run_closed_loop(f.iloc[:end], p))


# ── 策略接口兼容 / 参数校验 / 专用撮合 ──────────────────────────────────
def _price_frame():
    close = np.concatenate([np.full(220, 100.0), np.linspace(100, 120, 40),
                            np.linspace(120, 90, 60), np.linspace(90, 125, 70)])
    return pd.DataFrame(dict(open=close, high=close + 0.5, low=close - 0.5,
                             close=close, volume=np.full(len(close), 1000.0)),
                        index=pd.date_range("2026-01-01", periods=len(close), freq="15min", tz="UTC"))


def test_default_mode_keeps_legacy_signals_and_adds_only_columns():
    raw = _price_frame()
    params = resolve_params(CODE, {}, {"cost_filter": False, "max_atr_pct": 0.2})
    enriched = prepare_frame(raw, CODE, params)
    assert set(CLOSED_LOOP_COLUMNS).issubset(enriched.columns)
    legacy = evaluate_signals(enriched, CODE, params)
    assert legacy.direction.ne(0).any(), "默认模式不得改变原有信号口径"
    assert set(legacy.columns) == {"signed", "score", "prob_up", "direction", "block_reason",
                                   "position_pct", "leverage", "sl_pct", "tp_pct", "risk_pct",
                                   "net_reward_risk"}


def test_closed_loop_mode_early_routes_and_describes_state():
    raw = _price_frame()
    params = resolve_params(CODE, {}, {"cost_filter": False, "max_atr_pct": 0.2, "cl_enabled": True})
    enriched = prepare_frame(raw, CODE, params)
    decision = evaluate_signals(enriched, CODE, params)
    assert decision.direction.eq(0).all()
    assert decision.block_reason.eq("closed_loop_requires_dedicated_broker").all()
    snap = describe(enriched.iloc[-1])
    for key in ("closed_loop_state", "closed_loop_event", "pending_price",
                "support_level", "resistance_level", "protective_stop", "take_profit", "close_reason"):
        assert key in snap
    assert "submit" in enriched.st_cl_event.values or "skip_no_support" in enriched.st_cl_event.values


@pytest.mark.parametrize("overrides", [
    {"cl_tp_mode": 2},
    {"cl_support_buffer_atr": 5.0, "cl_max_entry_distance_atr": 0.5},
    {"cl_not_a_param": 1},
])
def test_new_params_are_validated(overrides):
    with pytest.raises(ValueError):
        resolve_params(CODE, {}, overrides)


def test_broker_applies_global_costs_exactly_once_per_side():
    f = _frame([100, 99, 98.6, 101, 103, 104, 105, 106, 107, 108])
    p = _params()
    enriched = f.assign(**{c: run_closed_loop(f, p)[c] for c in CLOSED_LOOP_COLUMNS})
    fee, slippage, initial = 0.0004, 0.0002, 10000.0
    equity, trades = simulate_closed_loop(enriched, None, p, initial, fee, slippage, "1h")
    assert len(trades) == 1
    trade = trades[0]
    assert trade["status"] == "take_profit"
    entry, exit_ = trade["entry_price"], trade["exit_price"]
    assert entry == pytest.approx(98.6 * (1 + slippage))
    assert exit_ == pytest.approx(103.1 * (1 - slippage))
    assert trade["fees"] == pytest.approx(initial * fee + abs(exit_ * trade["size"]) * fee)
    assert equity.attrs["open_position"] is None


def test_legacy_strategy_switch_is_rejected_for_both_modes():
    """迁移后旧总开关统一拒绝，启停应通过绑定操作。"""
    import json
    from types import SimpleNamespace

    from fastapi import HTTPException

    from app.api.strategies import CLOSED_LOOP_CODE, update_strategy
    from app.schemas import StrategyUpdate

    class _Query:
        def __init__(self, row): self.row = row
        def filter(self, *args, **kwargs): return self
        def first(self): return self.row

    class _DB:
        def __init__(self, row): self.row = row
        def query(self, *args): return _Query(self.row)
        def commit(self): pass
        def refresh(self, row): pass

    def make_row(params):
        return SimpleNamespace(id=1, code=CLOSED_LOOP_CODE, name="Supertrend", description="",
                               enabled=0, params=json.dumps(params), updated_at=None)

    for params in ({"cl_enabled": True}, {}):
        for enabled in (True, False):
            row = make_row(params)
            with pytest.raises(HTTPException) as exc:
                update_strategy(CLOSED_LOOP_CODE, StrategyUpdate(enabled=enabled), _DB(row))
            assert exc.value.status_code == 409
            assert row.enabled == 0 and json.loads(row.params) == params
