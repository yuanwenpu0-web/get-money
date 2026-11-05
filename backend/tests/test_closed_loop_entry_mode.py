"""闭环进场价模式：市价 / 反向回踩支撑位 / 对侧电平突破。

只做工程回放验证，不做 edge 判定（手册 §0 例外条款）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.closed_loop_execution import simulate_closed_loop
from app.strategy.closed_loop import (
    CLOSED_LOOP_COLUMNS, ENTRY_BREAKOUT, ENTRY_MARKET, ENTRY_PULLBACK,
    FLAT, OPEN, PENDING, run_closed_loop,
)
from app.strategy.rules import prepare_frame, resolve_params
from app.strategy.specs.supertrend_current import CODE, describe

IDX = pd.date_range("2026-02-01", periods=400, freq="1h", tz="UTC")


def _params(**overrides):
    base = dict(
        cl_entry_mode=1, cl_tp_mode=0, cl_support_buffer_atr=0.10,
        cl_max_entry_distance_atr=3.0, cl_pending_expire_bars=3,
        cl_dynamic_activate_r=1.0, cl_trailing_atr_mult=1.5, cl_cooldown_bars=0,
        sl_atr_mult=1.5, target_r=3.0, pivot_length=5,
        use_adx_filter=False, adx_threshold=50.0, use_ema200_filter=False,
    )
    base.update(overrides)
    return base


def _frame(close, *, support=98.5, resistance=110.0, direction=None, signal=None,
           line=90.0, atr=1.0, opens=None, low_pad=0.4, high_pad=0.4):
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


# ── 默认（1 反向回踩）必须逐位等于改造前口径 ────────────────────────────
def test_default_mode_is_bitwise_identical_to_explicit_pullback():
    f = _frame([100, 99, 98.6, 101, 103, 104, 105, 106, 107, 108])
    pd.testing.assert_frame_equal(run_closed_loop(f, _params()),
                                  run_closed_loop(f, _params(cl_entry_mode=ENTRY_PULLBACK)))
    out = run_closed_loop(f, _params())
    assert out.st_cl_entry_mode.dropna().eq(ENTRY_PULLBACK).all()


# ── 0 市价：不依赖任何电平，下一根开盘成交 ──────────────────────────────
def test_market_entry_fills_at_next_open_without_any_support_level():
    # 支撑位远到回踩模式必然 skip_no_support；市价模式必须照常提交并成交
    f = _frame([100, 100.5, 102, 103, 104, 105, 106, 107, 108, 109],
               support=1.0, resistance=110.0, opens=[100, 100.5, 102, 103, 104, 105, 106, 107, 108, 109])
    assert _events(run_closed_loop(f, _params()))[0] == "skip_no_support"
    out = run_closed_loop(f, _params(cl_entry_mode=ENTRY_MARKET))
    ev = _events(out)
    assert ev[0] == "submit" and ev[1] == "fill"
    assert pd.isna(out.st_cl_pending_price.iloc[0])   # 市价单没有挂单价
    assert pd.isna(out.st_cl_level.iloc[0])           # 也不做支撑位判定
    assert out.st_cl_planned_sl.iloc[0] == pytest.approx(98.5)   # 按信号收盘 100 预配
    assert out.st_cl_planned_tp.iloc[0] == pytest.approx(104.5)
    assert out.st_cl_fill_price.iloc[1] == pytest.approx(100.5)   # s+1 开盘
    assert out.st_cl_sl.iloc[1] == pytest.approx(99.0)             # 成交后按实际价重锚
    assert out.st_cl_tp.iloc[1] == pytest.approx(105.0)
    assert out.st_cl_entry_mode.iloc[:2].eq(ENTRY_MARKET).all()
    assert ev.count("fill") == 1 and sum(e.startswith("close_") for e in ev) == 1


def test_market_entry_broker_applies_global_costs_once_per_side():
    f = _frame([100, 100.5, 102, 103, 104, 105, 106, 107, 108, 109],
               support=1.0, opens=[100, 100.5, 102, 103, 104, 105, 106, 107, 108, 109])
    p = _params(cl_entry_mode=ENTRY_MARKET)
    enriched = f.assign(**{c: run_closed_loop(f, p)[c] for c in CLOSED_LOOP_COLUMNS})
    fee, slippage, initial = 0.0004, 0.0002, 10000.0
    _, trades = simulate_closed_loop(enriched, None, p, initial, fee, slippage, "1h")
    assert len(trades) == 1
    assert trades[0]["entry_price"] == pytest.approx(100.5 * (1 + slippage))
    assert trades[0]["exit_price"] == pytest.approx(105.0 * (1 - slippage))
    assert trades[0]["fees"] > 0


def test_market_order_never_hangs_when_next_bar_has_no_price():
    f = _frame([100, np.nan, 102, 103, 104, 105], support=1.0,
               opens=[100, np.nan, 102, 103, 104, 105])
    out = run_closed_loop(f, _params(cl_entry_mode=ENTRY_MARKET))
    ev = _events(out)
    assert ev.count("submit") == 1
    assert "fill" not in ev
    assert ev.count("cancel_market_unfilled") == 1
    assert out.st_cl_state.iloc[3] == FLAT


def test_market_entry_is_causal_under_frame_truncation():
    rng = np.random.default_rng(11)
    n = 90
    close = 100 + np.cumsum(rng.normal(0, 0.7, n))
    f = _frame(close, support=1.0,
               signal=np.where(np.arange(n) % 19 == 0, 1.0, 0.0))
    p = _params(cl_entry_mode=ENTRY_MARKET)
    full = run_closed_loop(f, p)
    for end in (30, 60, 85):
        pd.testing.assert_frame_equal(full.iloc[:end], run_closed_loop(f.iloc[:end], p))


# ── 2 对侧电平突破：多头取阻力、空头取支撑，stop 语义成交 ────────────────
def test_breakout_long_uses_resistance_and_conservative_gap_fill():
    f = _frame([100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
               support=50.0, resistance=100.4,
               opens=[100, 100.8, 102, 103, 104, 105, 106, 107, 108, 109])
    assert _events(run_closed_loop(f, _params()))[0] == "skip_no_support"
    out = run_closed_loop(f, _params(cl_entry_mode=ENTRY_BREAKOUT))
    ev = _events(out)
    assert ev[0] == "submit" and ev[1] == "fill"
    assert out.st_cl_level.iloc[0] == pytest.approx(100.4)            # 用的是阻力，不是支撑
    assert out.st_cl_pending_price.iloc[0] == pytest.approx(100.5)    # +0.1ATR 缓冲
    assert out.st_cl_planned_sl.iloc[0] == pytest.approx(99.0)
    # 开盘 100.8 高于触发价 100.5 → 按更差的 100.8 成交（不是更优的 100.5）
    assert out.st_cl_fill_price.iloc[1] == pytest.approx(100.8)
    assert out.st_cl_sl.iloc[1] == pytest.approx(99.3)
    assert out.st_cl_entry_mode.iloc[:2].eq(ENTRY_BREAKOUT).all()
    assert ev.count("fill") == 1 and sum(e.startswith("close_") for e in ev) == 1


def test_breakout_short_uses_support_level():
    f = _frame([100, 99, 98, 97, 96, 95, 94, 93, 92, 91],
               support=99.6, resistance=200.0, direction=-np.ones(10),
               signal=np.r_[-1.0, np.zeros(9)], line=200.0,
               opens=[100, 99, 98, 97, 96, 95, 94, 93, 92, 91])
    assert _events(run_closed_loop(f, _params()))[0] == "skip_no_support"
    out = run_closed_loop(f, _params(cl_entry_mode=ENTRY_BREAKOUT))
    ev = _events(out)
    assert ev[0] == "submit" and ev[1] == "fill"
    assert out.st_cl_level.iloc[0] == pytest.approx(99.6)             # 用的是支撑
    assert out.st_cl_pending_price.iloc[0] == pytest.approx(99.5)     # −0.1ATR 缓冲
    assert out.st_cl_fill_price.iloc[1] == pytest.approx(99.0)
    assert out.st_cl_side.iloc[1] == -1
    assert ev.count("fill") == 1


def test_breakout_does_not_fill_when_price_never_touches_the_level():
    f = _frame([100, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7, 100.8, 100.9],
               support=50.0, resistance=102.0)  # 触发价 102.1，全帧最高价仅 101.3
    out = run_closed_loop(f, _params(cl_entry_mode=ENTRY_BREAKOUT))
    ev = _events(out)
    assert ev[0] == "submit"
    assert "fill" not in ev
    assert "cancel_expired" in ev or "cancel_indicator" in ev
    assert out.st_cl_state.iloc[-1] == FLAT


# ── 三档共用退出、幂等与参数校验 ────────────────────────────────────────
@pytest.mark.parametrize("mode", [ENTRY_MARKET, ENTRY_PULLBACK, ENTRY_BREAKOUT])
def test_all_modes_keep_single_fill_and_single_close_invariants(mode):
    rng = np.random.default_rng(2026 + mode)
    n = 200
    close = 100 + np.cumsum(rng.normal(0, 0.8, n))
    direction = np.where(np.diff(np.r_[close[0], close]) > 0, 1.0, -1.0)
    direction[::31] *= -1
    signal = np.zeros(n)
    signal[::23] = np.where(direction[::23] > 0, 1.0, -1.0)
    f = _frame(close, direction=direction, signal=signal,
               support=float(close.min() - 2), resistance=float(close.max() + 2),
               line=float(close.min()))
    out = run_closed_loop(f, _params(cl_entry_mode=mode, cl_pending_expire_bars=6))
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
    assert len(closed_ids) == len(set(closed_ids))
    assert sum(e.startswith("close_") for e in ev) <= sum(e == "fill" for e in ev)
    assert out.st_cl_entry_mode.dropna().isin([float(mode)]).all()


@pytest.mark.parametrize("value", [3, -1, 0.5, 1.5])
def test_invalid_entry_mode_is_rejected_by_resolve_params(value):
    with pytest.raises(ValueError):
        resolve_params(CODE, {}, {"cl_entry_mode": value})


def test_market_mode_ignores_buffer_vs_distance_conflict():
    params = resolve_params(CODE, {}, {
        "cl_entry_mode": ENTRY_MARKET,
        "cl_support_buffer_atr": 5.0,
        "cl_max_entry_distance_atr": 0.5,
    })
    assert params["cl_entry_mode"] == ENTRY_MARKET
    with pytest.raises(ValueError):
        resolve_params(CODE, {}, {
            "cl_entry_mode": ENTRY_PULLBACK,
            "cl_support_buffer_atr": 5.0,
            "cl_max_entry_distance_atr": 0.5,
        })


def test_describe_exposes_entry_mode_and_entry_price():
    def _price_frame():
        close = np.concatenate([np.full(220, 100.0), np.linspace(100, 120, 40),
                                np.linspace(120, 90, 60), np.linspace(90, 125, 70)])
        return pd.DataFrame(dict(open=close, high=close + 0.5, low=close - 0.5,
                                 close=close, volume=np.full(len(close), 1000.0)),
                            index=pd.date_range("2026-01-01", periods=len(close), freq="15min", tz="UTC"))

    params = resolve_params(CODE, {}, {"cost_filter": False, "max_atr_pct": 0.2,
                                       "cl_enabled": True, "cl_entry_mode": ENTRY_MARKET})
    enriched = prepare_frame(_price_frame(), CODE, params)
    snap = describe(enriched.iloc[-1])
    assert "entry_mode" in snap and "entry_price" in snap
    assert "submit" in enriched.st_cl_event.values
