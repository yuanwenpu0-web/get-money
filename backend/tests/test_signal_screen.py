"""signal_screen 的统计口径测试：锁住「重叠计数」「退化列」「FDR」三个易错点。

这些不是形式测试。每个断言对应一次会改变结论的统计错误：
- 若 rising_edges 退化成「为真就计数」，事件数会虚高数倍，t 值虚高，假信号通过。
- 若 bh_qvalues 写成 p*n/rank 的朴素版，排序后不会单调，最小 q 会算错。
- 若 forward_return 用了 shift(-1) 的窗口，会变成「过去窗口」，正对照立刻失效。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / ".." / "scripts"))

import numpy as np
import pytest

from signal_screen import bh_qvalues, forward_return, rising_edges


def test_rising_edges_takes_onset_not_every_true_bar():
    """连续为真的 5 根只算 1 个事件——这是重叠计数的根源。"""
    mask = np.array([False, True, True, True, True, True, False, False, True, False])
    edges = rising_edges(mask, min_gap=1)
    assert list(edges) == [1, 8]


def test_rising_edges_enforces_min_gap_to_avoid_overlap():
    """间隔小于 horizon 的第二次触发被丢弃，保证前瞻窗口不重叠。"""
    mask = np.zeros(20, dtype=bool)
    mask[[0, 3, 10, 12]] = True
    edges = rising_edges(mask, min_gap=5)
    # 0 保留；3 距 0 只有 3 < 5 丢弃；10 距 0 为 10 保留；12 距 10 为 2 丢弃
    assert list(edges) == [0, 10]


def test_rising_edges_empty_mask():
    assert len(rising_edges(np.zeros(10, dtype=bool), min_gap=3)) == 0


def test_forward_return_is_future_not_past():
    """关键：收益必须朝未来看。末根无法计算，应为 nan。"""
    close = np.array([100.0, 101.0, 102.0, 103.0])
    fut = forward_return(close, horizon=1)
    assert fut[0] == pytest.approx(0.01)  # 100 → 101
    assert np.isnan(fut[-1])


def test_forward_return_horizon_two():
    close = np.array([100.0, 101.0, 110.0, 111.0])
    fut = forward_return(close, horizon=2)
    assert fut[0] == pytest.approx(0.10)  # 100 → 110
    assert fut[1] == pytest.approx((111 - 101) / 101)
    assert np.isnan(fut[2]) and np.isnan(fut[3])


def test_bh_qvalues_monotone_and_bounded():
    p = np.array([0.001, 0.004, 0.02, 0.03, 0.5])
    q = bh_qvalues(p)
    assert np.all(q >= 0) and np.all(q <= 1)
    # q 值按 p 的排序应单调不减
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= 0)


def test_bh_qvalues_corrects_upward_for_many_tests():
    """同样 p 值，测 100 个比测 1 个的 q 值更大——多重比较必须惩罚。"""
    p = np.array([0.01])
    assert bh_qvalues(p)[0] == pytest.approx(0.01)
    p_many = np.array([0.01] + [0.9] * 99)
    assert bh_qvalues(p_many)[0] > 0.01


def test_bh_qvalues_empty():
    assert len(bh_qvalues(np.array([]))) == 0
