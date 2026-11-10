"""通用 EMA 快慢线方向过滤：参数校验、门控语义、顺序一致性、纯因果。

过滤只做入场门控（把逆趋势方向清零），不改变策略自身的方向判断。
默认关闭（fast/slow 任一为 0），行为与改造前完全一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategy import specs  # noqa: F401  导入即注册内置策略
from app.strategy.registry import all_specs, get_spec
from app.strategy.rules import (
    EXIT_KEY,
    apply_ema_trend_filter,
    apply_exit_overlay,
    evaluate_signals,
    prepare_frame,
    resolve_params,
)
from app.strategy.signals import SIGNAL_COLUMNS

CODE = "supertrend_current"
ON = {"ema_trend_fast": 10, "ema_trend_slow": 30}


def _signals(idx, directions, reasons=None):
    """构造最小可执行信号帧，其余列给合法占位值。"""
    frame = pd.DataFrame(index=idx)
    for name in SIGNAL_COLUMNS:
        frame[name] = 0.0
    frame["direction"] = np.asarray(directions, dtype=int)
    frame["block_reason"] = "" if reasons is None else list(reasons)
    frame["position_pct"] = np.where(frame["direction"].ne(0), 0.3, 0.0)
    frame["leverage"] = np.where(frame["direction"].ne(0), 1, 0)
    frame["risk_pct"] = np.where(frame["direction"].ne(0), 0.01, 0.0)
    frame["signed"] = frame["direction"].astype(float)
    return frame


def _trend(idx, start=100.0, end=200.0):
    close = pd.Series(np.linspace(start, end, len(idx)), index=idx)
    return pd.DataFrame({"close": close})


# ---------- 参数层 ----------

def test_defaults_off_for_every_spec():
    """所有策略默认关闭 EMA 过滤，既有口径不变。"""
    for spec in all_specs():
        params = resolve_params(spec.code, {})
        assert params["ema_trend_fast"] == 0, spec.code
        assert params["ema_trend_slow"] == 0, spec.code


@pytest.mark.parametrize("fast,slow", [(50, 20), (20, 20), (100, 50)])
def test_reject_fast_not_less_than_slow(fast, slow):
    with pytest.raises(ValueError, match="快线周期严格小于慢线周期"):
        resolve_params(CODE, {}, {"ema_trend_fast": fast, "ema_trend_slow": slow})


@pytest.mark.parametrize("fast,slow", [(0, 0), (0, 50), (12, 0), (12, 50)])
def test_legal_pairs(fast, slow):
    params = resolve_params(CODE, {}, {"ema_trend_fast": fast, "ema_trend_slow": slow})
    assert (params["ema_trend_fast"], params["ema_trend_slow"]) == (fast, slow)


# ---------- 门控层 ----------

@pytest.mark.parametrize("params", [
    {"ema_trend_fast": 0, "ema_trend_slow": 0},
    {"ema_trend_fast": 12, "ema_trend_slow": 0},
    {"ema_trend_fast": 0, "ema_trend_slow": 50},
])
def test_any_zero_leg_is_exact_noop(params):
    idx = pd.date_range("2026-01-01", periods=120, freq="1h")
    sig = _signals(idx, [1, -1, 0, 1, -1] + [0] * 115)
    out = apply_ema_trend_filter(_trend(idx), sig, params)
    pd.testing.assert_frame_equal(out, sig)


def test_uptrend_allows_long_blocks_short():
    idx = pd.date_range("2026-01-01", periods=200, freq="1h")
    frame = _trend(idx)  # 线性上升：EMA(fast) > EMA(slow)
    sig = _signals(idx, [0] * 198 + [-1, 1])
    out = apply_ema_trend_filter(frame, sig, ON)
    assert out["direction"].iloc[-1] == 1, "顺势多头必须放行"
    assert out["direction"].iloc[-2] == 0, "逆势空头必须被挡"
    assert out.loc[idx[-2], "block_reason"] == "ema_trend_filter"
    assert out.loc[idx[-2], "position_pct"] == 0.0
    assert out.loc[idx[-2], "leverage"] == 0
    assert out.loc[idx[-2], "risk_pct"] == 0.0
    # 通过的那一行所有风控列保持原值
    assert out.loc[idx[-1], "position_pct"] == 0.3


def test_downtrend_allows_short_blocks_long():
    idx = pd.date_range("2026-01-01", periods=200, freq="1h")
    frame = _trend(idx, start=200.0, end=100.0)
    sig = _signals(idx, [0] * 198 + [1, -1])
    out = apply_ema_trend_filter(frame, sig, ON)
    assert out["direction"].iloc[-2] == 0, "逆势多头必须被挡"
    assert out["direction"].iloc[-1] == -1, "顺势空头必须放行"
    assert out.loc[idx[-2], "block_reason"] == "ema_trend_filter"


def test_existing_block_reason_is_not_overwritten():
    idx = pd.date_range("2026-01-01", periods=200, freq="1h")
    sig = _signals(idx, [0] * 198 + [-1, 0], [""] * 198 + ["cost_filter", ""])
    out = apply_ema_trend_filter(_trend(idx), sig, ON)
    assert out.loc[idx[-2], "direction"] == 0
    assert out.loc[idx[-2], "block_reason"] == "cost_filter"


def test_missing_close_column_is_noop():
    idx = pd.date_range("2026-01-01", periods=60, freq="1h")
    frame = pd.DataFrame({"volume": np.ones(len(idx))}, index=idx)
    sig = _signals(idx, [1, -1, 0] + [0] * 57)
    out = apply_ema_trend_filter(frame, sig, ON)
    pd.testing.assert_frame_equal(out, sig)


def test_causality_future_close_does_not_change_past():
    """EMA 用 ewm 递推：改动最后一根 close 不得影响此前任何一根的门控结果。"""
    idx = pd.date_range("2026-01-01", periods=200, freq="1h")
    frame_a = _trend(idx)
    frame_b = frame_a.copy()
    frame_b.loc[idx[-1], "close"] = 1000.0
    sig = _signals(idx, [0] * 198 + [1, -1])
    out_a = apply_ema_trend_filter(frame_a, sig.copy(), ON)
    out_b = apply_ema_trend_filter(frame_b, sig.copy(), ON)
    pd.testing.assert_series_equal(
        out_a["direction"].iloc[:-1], out_b["direction"].iloc[:-1]
    )


# ---------- 与出场覆盖的顺序（H10：回测/寻优/实盘同口径）----------

def _ohlcv(n=600, seed=7):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    return pd.DataFrame(
        {"open": close, "high": close * 1.005, "low": close * 0.995,
         "close": close, "volume": 1000.0},
        index=idx,
    )


@pytest.mark.parametrize("exit_cfg", [
    {"method": "strategy"},
    {"method": "atr_r", "params": {"atr_mult": 1.5, "target_r": 2.0}},
])
def test_ema_gate_is_last_after_exit_overlay(exit_cfg):
    """覆盖层会用 finalize_signal_frame 重建 direction；门控放最后一道，
    evaluate_signals 与 optimizer worker 的手动链路必须产出同一份 direction。

    同时验证两种顺序在真策略帧上等价（回归保护）。
    """
    data = _ohlcv()
    params = resolve_params(CODE, {}, {
        "ema_trend_fast": 12, "ema_trend_slow": 50,
        EXIT_KEY: exit_cfg,
    })
    frame = prepare_frame(data, CODE, params)
    raw = get_spec(CODE).signals(frame, params)
    # 顺序 A：门控在前、覆盖在后（改造前的错误顺序）
    a = apply_exit_overlay(frame, apply_ema_trend_filter(frame, raw.copy(), params), params)
    # 顺序 B：覆盖在前、门控最后（optimizer worker 与改造后的 evaluate_signals）
    b = apply_ema_trend_filter(frame, apply_exit_overlay(frame, raw.copy(), params), params)
    pd.testing.assert_series_equal(a["direction"], b["direction"])
    via_entry = evaluate_signals(data, CODE, params)
    pd.testing.assert_series_equal(
        b["direction"], via_entry["direction"], check_names=False
    )


def test_evaluate_signals_filter_changes_output():
    """同一份行情下开/关 EMA 门控必须产出不同信号（证明过滤确实接到了统一入口）。"""
    data = _ohlcv()
    off = evaluate_signals(data, CODE, resolve_params(CODE, {}))
    on = evaluate_signals(data, CODE, resolve_params(
        CODE, {}, {"ema_trend_fast": 12, "ema_trend_slow": 50}
    ))
    assert not off["direction"].equals(on["direction"])
    assert on["direction"].ne(0).sum() <= off["direction"].ne(0).sum()
    assert "ema_trend_filter" in set(on["block_reason"])
