"""RGD v2 指标原语：Wilder ATR / 前置窗口百分位秩 / 滚动 Pearson / tension 状态机。

本模块只做纯数值计算：不读数据库、不持有跨调用状态、不依赖 pandas 之外的库。
口径出处 `docs/rgd-porting-contract.md`：

- §3 V1 / §8 R4：`ta.atr` 用 Wilder RMA（与 `supertrend_current._rma` 同一算法），
  不用 `indicators._atr` 的 `ewm(alpha=1/L)`——后者首根即出值、seed 不同。
- §4.3：`ta.percentrank` 的比较集是「当前 bar 之前的 length 根」，`<=` 计入相等，
  分母是 length；首个有效位置为 length（0-based）。
- §4.4：`ta.correlation` 用 `rolling(length).corr`；Pearson 对任一变量仿射变换不变，
  所以 `bar_index` 用段内 0-based 与 Pine 绝对索引逐位等价。
- §3 H2 / T5：`tensionStep` 必须显式循环（自引用 `tension[1]` 无法向量化）。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _rma(values: np.ndarray, length: int) -> np.ndarray:
    """Wilder 平滑，按前 length 个有效值的简单均值初始化。

    与 `supertrend_current._rma` 逐行同构（契约 A4）：遇到非有限值沿用前值，
    这样 `ta.atr` 在预热段的输出与项目既有策略完全一致。
    """
    result = np.full(len(values), np.nan)
    count = 0
    total = 0.0
    prev = np.nan
    for i, x in enumerate(values):
        if not np.isfinite(x):
            result[i] = prev
            continue
        if count < length:
            count += 1
            total += x
            if count == length:
                prev = total / length
        else:
            prev = (prev * (length - 1) + x) / length
        result[i] = prev
    return result


def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Pine `ta.atr(length)`：Wilder RMA(TR)，前 length-1 根为 NaN。

    首根 TR 取 `high - low`（Pine `ta.tr(true)` 的 handle_na 分支），
    之后 `TR = max(high-low, |high-prev_close|, |low-prev_close|)`。
    """
    h = pd.to_numeric(high, errors="coerce").to_numpy(dtype=float)
    l = pd.to_numeric(low, errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    prev = np.concatenate(([np.nan], c[:-1]))
    tr = np.fmax(h - l, np.fmax(np.abs(h - prev), np.abs(l - prev)))
    return pd.Series(_rma(tr, int(length)), index=high.index, dtype=float)


def percentrank_preceding(src: np.ndarray, length: int) -> np.ndarray:
    """Pine `ta.percentrank(source, length)`（契约 §4.3，定义已核验）。

        percentrank[i] = 100 * |{ j in [i-length, i-1] : src[j] <= src[i] }| / length

    三个不许：含当前 bar、用 `<` 而漏掉相等、分母写成 length±1。
    窗口内任一 na 或当前值为 na -> na；首个有效位置 = length（0-based）。
    """
    array = np.asarray(src, dtype=float)
    n = array.size
    out = np.full(n, np.nan, dtype=float)
    if length <= 0 or n < length + 1:
        return out
    window = np.lib.stride_tricks.sliding_window_view(array, length + 1)
    history = window[:, :-1]          # src[k .. k+length-1]：不含当前
    current = window[:, -1]           # src[k+length]
    finite = np.isfinite(history).all(axis=1) & np.isfinite(current)
    rank = (history <= current[:, None]).sum(axis=1)
    out[length:] = np.where(finite, 100.0 * rank / float(length), np.nan)
    return out


def rolling_pearson(x: pd.Series, y: pd.Series, length: int) -> pd.Series:
    """Pine `ta.correlation(x, y, length)`：滚动 Pearson，min_periods=length。"""
    xs = pd.to_numeric(x, errors="coerce").astype(float)
    ys = pd.to_numeric(y, errors="coerce").astype(float)
    return xs.rolling(int(length), min_periods=int(length)).corr(ys)


def sign(value: float) -> int:
    """Pine `math.sign`：Python 的 `math.copysign` 对 0 返回 1，这里必须返回 0。

    显式分支而不是 `(x>0)-(x<0)`：后者在 numpy 标量上会做 bool 相减并直接报错。
    """
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def tension_step(
    prev: float, target: float, build_len: int, release_len: int, cap: float
) -> float:
    """Pine `tensionStep(prev, tgt, b, r, cap)`（契约 §3 H2）。

    `sameDir = prev==0 or tgt==0 or sign(prev)==sign(tgt)`；
    `building = sameDir and |tgt|>|prev|`；
    `a = 2/(b+1)` if building else `2/(r+1)`；结果 clamp 到 ±cap。
    target 为 na 时返回 na（Pine 的 na 传播），由调用方用 `nz(...,0)` 自愈。
    """
    if not math.isfinite(target):
        return math.nan
    same_dir = prev == 0 or target == 0 or sign(prev) == sign(target)
    building = same_dir and abs(target) > abs(prev)
    alpha = 2.0 / (build_len + 1.0) if building else 2.0 / (release_len + 1.0)
    return max(-cap, min(cap, prev + alpha * (target - prev)))


def tension_series(
    targets: np.ndarray, build_len: int, release_len: int, cap: float
) -> np.ndarray:
    """显式状态机：`tension := tensionStep(nz(tension[1],0), tg, b, r, cap)`。

    `nz(tension[1], 0)` 是自愈点：即使某根因 tg 为 na 使 tension 变 na，
    下一根的前值也归 0 而不是把 na 永久传播下去（契约 A2）。
    """
    values = np.asarray(targets, dtype=float)
    out = np.full(values.size, np.nan, dtype=float)
    prev = 0.0
    for i in range(values.size):
        base = prev if math.isfinite(prev) else 0.0
        out[i] = tension_step(base, values[i], build_len, release_len, cap)
        prev = out[i]
    return out


def prior_extreme(series: pd.Series, length: int, kind: str) -> np.ndarray:
    """`ta.highest/lowest(x[1], N)`：先 shift(+1) 再 rolling，窗口 [i-1 .. i-N]，不含当前。"""
    rolled = series.shift(1).rolling(int(length), min_periods=int(length))
    return (rolled.max() if kind == "max" else rolled.min()).to_numpy(dtype=float)


def state_machine(
    brk_long: np.ndarray, brk_short: np.ndarray,
    close: np.ndarray, def_long: np.ndarray, def_short: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pine L141-155 的纯状态机，逐条 else-if 顺序照抄。

    返回 `(state, prev_state, long_break, short_break)`。`long_break` 已含
    `prevState != 1` 门控、`short_break` 已含 `prevState != -1` 门控。
    比较 na（NaN）一律为 False，等价 Pine 的 na 条件取 else 分支。
    """
    n = len(close)
    states = np.zeros(n, dtype=int)
    prevs = np.zeros(n, dtype=int)
    longs = np.zeros(n, dtype=bool)
    shorts = np.zeros(n, dtype=bool)
    state = 0
    for i in range(n):
        prev = state
        prevs[i] = prev
        long_brk = bool(brk_long[i]) and prev != 1
        short_brk = bool(brk_short[i]) and prev != -1
        longs[i] = long_brk
        shorts[i] = short_brk
        if long_brk:
            state = 1
        elif short_brk:
            state = -1
        elif prev == 1 and close[i] < def_long[i]:
            state = 0
        elif prev == -1 and close[i] > def_short[i]:
            state = 0
        states[i] = state
    return states, prevs, longs, shorts
