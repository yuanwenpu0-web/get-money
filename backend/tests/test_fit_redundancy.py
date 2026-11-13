"""`redundancy_pairs` 的归堆测试：形状、阈值与「不许把无关列连成一堆」。

这个函数出过一次静默事故：`covered[cand_idx]` 已经是 (列, 币) 两维，条件张量却写成
`cov[:, None]`，`np.where` 把逐币均值广播成 (列, 列, 币)，于是「两列的相关」实际上是拿
一列对**所有列**的切片去算——1h 一轮 70 列报了 1858 对、并成 1 堆，看着像结论，其实
等于没归堆。形状错了不会抛异常，只会给出**听起来更严重**的数字，所以这里用不变量守：
每对报告的共同币数不可能超过参评币数。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
from indicator_fit_matrix import REDUNDANCY_R_MIN, redundancy_pairs  # noqa: E402

from app.research.fit_events import FIELDS, FI  # noqa: E402

NH = 1
N_SYM = 40


def build(means: list[np.ndarray], events: int = 100) -> tuple:
    """把「每列逐币平均超额」塞进聚合张量：n 恒为 events，sum_x = 均值 × n。"""
    cols = len(means)
    matrix = np.zeros((cols, N_SYM, NH, len(FIELDS)), dtype=float)
    for i, m in enumerate(means):
        matrix[i, :, 0, FI["n"]] = events
        matrix[i, :, 0, FI["sum_x"]] = m * events
    covered = np.ones((cols, N_SYM), dtype=bool)
    keys = [f"ind{i}|col" for i in range(cols)]
    return matrix, covered, keys


def test_identical_columns_form_one_pair():
    rng = np.random.default_rng(7)
    base = rng.normal(0.0, 0.002, N_SYM)
    matrix, covered, keys = build([base, base.copy()])
    pairs = redundancy_pairs(matrix, covered, keys, 0, [0, 1], threshold=REDUNDANCY_R_MIN)
    assert len(pairs) == 1
    assert pairs[0]["r"] > 0.999
    # 形状事故的不变量：共同币数不可能超过参评币数
    assert pairs[0]["symbols"] <= N_SYM


def test_unrelated_columns_are_not_pooled():
    rng = np.random.default_rng(11)
    a = rng.normal(0.0, 0.002, N_SYM)
    b = rng.normal(0.0, 0.002, N_SYM)
    c = rng.normal(0.0, 0.002, N_SYM)
    matrix, covered, keys = build([a, b, c])
    pairs = redundancy_pairs(matrix, covered, keys, 0, [0, 1, 2], threshold=REDUNDANCY_R_MIN)
    assert all(abs(p["r"]) < REDUNDANCY_R_MIN for p in pairs)


def test_shape_bug_would_inflate_pairs():
    """反证：把广播后的错误写法跟正确结果并排跑，错的会虚报冗余对。

    这里不 import 私有实现，直接在测试里复现当年那行 `cov[:, None]`，用它算一遍同样的
    列对；一旦有人把生产代码改回广播写法，前两个用例会挂，这个用例说明为什么必须挂。
    """
    rng = np.random.default_rng(3)
    means = [rng.normal(0.0, 0.002, N_SYM) for _ in range(6)]
    matrix, covered, keys = build(means)
    idx = list(range(6))
    good = redundancy_pairs(matrix, covered, keys, 0, idx, threshold=REDUNDANCY_R_MIN)
    n = matrix[idx, :, 0, FI["n"]]
    x = matrix[idx, :, 0, FI["sum_x"]]
    cov = covered[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        bad = np.where(cov[:, None] & (n > 0), x / np.maximum(n, 1.0), np.nan)
    assert bad.shape == (len(idx), len(idx), N_SYM), "广播一发生就是三维，没有任何异常会提醒你"
    inflated = 0
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            va, vb = bad[a].ravel(), bad[b].ravel()
            ok = np.isfinite(va) & np.isfinite(vb)
            if np.corrcoef(va[ok], vb[ok])[0, 1] >= REDUNDANCY_R_MIN:
                inflated += 1
    assert inflated > len(good)


def test_columns_without_events_are_not_pooled():
    rng = np.random.default_rng(5)
    a = rng.normal(0.0, 0.002, N_SYM)
    matrix, covered, keys = build([a, a.copy()])
    matrix[:, :, 0, FI["n"]] = 0  # 有覆盖但一个事件都没触发
    assert redundancy_pairs(matrix, covered, keys, 0, [0, 1], threshold=REDUNDANCY_R_MIN) == []


def test_single_candidate_returns_empty():
    matrix, covered, keys = build([np.zeros(N_SYM)])
    assert redundancy_pairs(matrix, covered, keys, 0, [0]) == []
