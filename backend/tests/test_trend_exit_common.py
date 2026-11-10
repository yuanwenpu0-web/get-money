"""趋势反转出场课题的纯函数单测：因果性、成本、撮合时序、随机基线。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

from trend_exit_common import (  # noqa: E402
    DECISION_BAR, FORWARD_H, HORIZON, MIN_GAP, SELL_SIGNALS,
    attach_features, build_entries, decision_point, first_trigger,
    mirror_barriers, net_return_pct, replay, replay_from_barrier, wilder,
)


def _frame(n=260, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-05-11", periods=n, freq="1h", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.4, n))
    high = close + rng.uniform(0.1, 0.8, n)
    low = close - rng.uniform(0.1, 0.8, n)
    return pd.DataFrame({"open": close, "high": high, "low": low,
                         "close": close, "volume": 1.0}, index=idx)


def test_wilder_sma_seed():
    v = pd.Series([1.0] * 20)
    out = wilder(v, 14)
    assert np.isnan(out.iloc[12])
    assert out.iloc[13] == pytest.approx(1.0)
    assert out.iloc[-1] == pytest.approx(1.0)
    w = wilder(pd.Series([1.0] * 14 + [2.0]), 14)
    assert w.iloc[14] == pytest.approx(15.0 / 14.0)


def test_net_return_cost_matches_project_convention():
    # 与 order_exit_common.net_return_pct 同口径：fee*(1+exit/entry) + 2*slip
    from order_exit_common import net_return_pct as ref
    assert net_return_pct(100.0, 101.0, 1) == pytest.approx(ref(100.0, 101.0, "LONG"))
    assert net_return_pct(100.0, 99.0, -1) > 0
    assert np.isnan(net_return_pct(0.0, 101.0, 1))
    # 双边成本约 0.12%，零毛收益时应为负
    assert net_return_pct(100.0, 100.0, 1) == pytest.approx(-0.12, abs=0.001)


def test_attach_features_is_causal():
    df = _frame()
    score, trig = attach_features(df)
    assert list(trig.columns) == list(SELL_SIGNALS)
    assert set(SELL_SIGNALS).issubset(score.columns)
    # 改掉未来 50 根，历史窗口的分值/触发必须逐点不变
    df2 = df.copy()
    cut = 180
    df2.iloc[cut:, df2.columns.get_loc("close")] *= 3.0
    df2.iloc[cut:, df2.columns.get_loc("high")] *= 3.0
    s2, t2 = attach_features(df2)
    for name in SELL_SIGNALS:
        assert np.allclose(score[name].iloc[:cut].to_numpy(dtype=float),
                           s2[name].iloc[:cut].to_numpy(dtype=float), equal_nan=True), name
        assert (trig[name].iloc[:cut].to_numpy()
                == t2[name].iloc[:cut].to_numpy()).all(), name


def test_attach_features_xsec_threshold():
    df = _frame(200)
    fp = np.full(len(df), 0.5)
    fp[100:] = 0.9
    op = np.full(len(df), np.nan)
    score, trig = attach_features(df, funding_pct=fp, funding_rate=-fp, oi_pct=op)
    assert not trig["funding_crowd"].iloc[99]
    assert trig["funding_crowd"].iloc[100]
    assert not trig["oi_crowd"].any()          # 无数据不得读成「无信号」再当负例
    assert trig["funding_neg"].all()


def test_build_entries_rising_edge_and_gap():
    df = _frame(400)
    n = len(df)
    d = np.zeros(n)
    for t in (20, 21, 90, 200, 240):
        d[t] = 1
    sig = pd.DataFrame({"direction": d, "sl_pct": np.full(n, 0.01),
                        "tp_pct": np.full(n, 0.03)}, index=df.index)
    entries = build_entries(df, sig)
    assert [e["t"] for e in entries] == [20, 90, 200]   # 240 落在 MIN_GAP 内
    assert entries[0]["entry_i"] == 21
    assert entries[0]["time_ms"] == int(df.index[21].timestamp() * 1000)
    assert MIN_GAP == HORIZON


def test_replay_same_bar_sl_first_and_gap():
    # 第一根 high 触 TP、low 触 SL：SL 优先
    bars = np.array([[100, 103, 97, 101]], dtype=float)
    px, j, why = replay(bars, sl=98.0, tp=102.0, side=1)
    assert (px, j, why) == (98.0, 0, "sl")
    # 跳空低开穿过 SL：按开盘价
    bars = np.array([[96, 97, 95, 96]], dtype=float)
    assert replay(bars, sl=98.0, tp=102.0, side=1)[::2] == (96.0, "sl_gap")
    # 跳空高开穿过 TP：按开盘价
    bars = np.array([[105, 106, 104, 105]], dtype=float)
    assert replay(bars, sl=98.0, tp=102.0, side=1)[::2] == (105.0, "tp_gap")


def test_replay_signal_exits_next_bar_open():
    bars = np.array([[100, 100.5, 99.5, 100.2],
                     [100.3, 100.8, 100.1, 100.6],
                     [100.7, 101.0, 100.5, 100.9]], dtype=float)
    px, j, why = replay(bars, sl=90.0, tp=200.0, side=1, trigger_j=0)
    assert (px, j, why) == (100.3, 1, "signal")
    px, j, why = replay(bars, sl=90.0, tp=200.0, side=1, trigger_j=2)
    assert (px, j, why) == (100.9, 2, "signal_end")
    assert replay(bars, sl=90.0, tp=200.0, side=1)[2] == "time"


def test_replay_short_mirror():
    bars = np.array([[100, 103, 97, 101]], dtype=float)
    e = {"entry_price": 100.0, "sl": 99.0, "tp": 97.0,
         "sl_pct": 0.01, "tp_pct": 0.03}
    m = mirror_barriers(e)
    assert m["side"] == -1 and m["sl"] > 100 and m["tp"] < 100
    px, _, why = replay(bars, sl=m["sl"], tp=m["tp"], side=-1)
    assert why == "sl" and px == pytest.approx(101.0)


def test_replay_from_barrier_matches_full_replay():
    rng = np.random.default_rng(11)
    bars = np.column_stack([
        100 + np.cumsum(rng.normal(0, 0.3, 48)),
        100 + np.cumsum(rng.normal(0, 0.3, 48)) + 0.5,
        100 + np.cumsum(rng.normal(0, 0.3, 48)) - 0.5,
        100 + np.cumsum(rng.normal(0, 0.3, 48)),
    ])
    barrier = replay(bars, sl=90.0, tp=200.0, side=1)
    for j in [None, 0, 1, 5, barrier[1] - 1, barrier[1], 46, 47]:
        a = replay_from_barrier(bars, barrier, j)
        b = replay(bars, sl=90.0, tp=200.0, side=1, trigger_j=j)
        assert a == b, (j, a, b)


def test_first_trigger_and_decision_point():
    col = np.array([False, False, True, False, True])
    assert first_trigger(col, 0) == 2
    assert first_trigger(np.zeros(5, dtype=bool), 0) is None
    df = _frame(120)
    score, _ = attach_features(df)
    close = df["close"].to_numpy(dtype=float)
    dp = decision_point(score, 10, close)
    assert dp is not None
    assert dp["fwd_pct"] == pytest.approx(
        (close[10 + DECISION_BAR + FORWARD_H] / close[10 + DECISION_BAR] - 1) * 100)
    assert decision_point(score, 100, close) is None
