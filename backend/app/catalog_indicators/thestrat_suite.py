"""TheStrat Suite v3.1.1 的 Python 移植。

TheStrat 的核心是**把一个 K 线约简成四元组**，其余都是它的投影：

* ``structure`` —— ``"1"`` / ``"2u"`` / ``"2d"`` / ``"3"``：这根 K 线对**前一根区间**
  做了什么。u/d 在这里是**破位侧**，不是阴阳。
* ``above_open`` —— 收盘严格高于开盘（收在开盘价上的算跌，归到下一档）。
* ``failed`` —— 方向性破位被拒绝（只有 2u/2d 会 fail）。
* ``live`` —— 这根还在形成。

三条容易读错的规则（原库文档明确写死）：

1. **相等不算破位**：高点恰好等于前高，没有越过去。
2. **缺口不特殊处理**：分类只看区间，不看开盘价；跳空开盘在前高之上，本身就已经
   至少是一个 2u。
3. **u/d 在 2 上是破位侧，在 1/3 上是阴阳** —— 这是整个记号里最容易被误读的地方。
   本移植另外输出 ``*_above_open`` 把符号通道单独暴露出来。

模块覆盖
--------
K 线分类（CC/C1/C2/C3 四档）、锤子/射手（Broad / Classic / Pin Bar 三档 + 颜色门）、
Failed 2（Reclaim / Open / Reclaim+Open / Reclaim OR Open 四种口径 + pre-F2 预信号）、
全周期连续性（FTFC）、多米诺（连续 inside 的周期数）、幅度目标（C1 区间）、
信号在效判定（``shouldDrawC1Level`` 的九分支决策树）、止损位（CC 或 C1 反向极值）。

未移植：绘图/线/箱/表格/标签池、告警、预览模式（依赖交易所日历与休市判定）、
Exhaustion 枢轴扫描（需早期扫描窗口与显示门槛，纯视觉）、Lead 信号过滤的绘制优先级。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import boolean, choice, integer, number, ohlcv

__all__ = ["thestrat_suite"]

# 结构码：0 = 无法分类（暖机期），1/2/3/4 对应 "1"/"2u"/"2d"/"3"
_STRUCT_NONE, _STRUCT_1, _STRUCT_2U, _STRUCT_2D, _STRUCT_3 = 0, 1, 2, 3, 4

_HAMMER_METHODS = ("Broad (Loose)", "Classic", "Pin Bar (Strict)")
_FAILED2_METHODS = ("Reclaim", "Open", "Reclaim + Open", "Reclaim OR Open")

_PRESET_TFS = {
    # (tf1..tf6, enabled flags)
    "TheStrat Classic": (("30", "60", "D", "W", "M", "M"), (True, True, True, True, True, False)),
    "Scalp": (("5", "15", "30", "60", "60", "60"), (True, True, True, True, True, False)),
    "Day Trade": (("15", "30", "60", "240", "D", "D"), (True, True, True, True, True, False)),
    "Futures/Crypto": (("60", "240", "720", "D", "W", "W"), (True, True, True, True, True, False)),
    "Swing Trade": (("240", "D", "W", "M", "M", "M"), (True, True, True, True, True, False)),
    "Investing": (("W", "M", "3M", "12M", "12M", "12M"), (True, True, True, True, True, False)),
}


def _period_size(token: str) -> int:
    text = str(token).strip().upper()
    fixed = {"D": 86400, "W": 604800, "M": 2592000, "3M": 7776000, "12M": 31536000}
    if text in fixed:
        return fixed[text]
    return max(1, int(float(text))) * 60


def _period_key(index: pd.DatetimeIndex, seconds: int):
    if seconds >= 31536000:
        return pd.PeriodIndex(index, freq="Y")
    if seconds >= 7776000:
        return pd.PeriodIndex(index, freq="Q")
    if seconds >= 2592000:
        return pd.PeriodIndex(index, freq="M")
    if seconds >= 604800:
        return pd.PeriodIndex(index, freq="W")
    if seconds >= 86400:
        return pd.PeriodIndex(index, freq="D")
    return index.floor(pd.Timedelta(seconds=int(seconds)))


def structure_of(prev_high, prev_low, high, low) -> np.ndarray:
    """``sg.structureOf``：对前一根区间做了什么（相等不算破位）。"""
    prev_high = np.asarray(prev_high, dtype=float)
    prev_low = np.asarray(prev_low, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    valid = (np.isfinite(prev_high) & np.isfinite(prev_low) & np.isfinite(high) & np.isfinite(low)
             & (high >= low) & (prev_high >= prev_low))
    broke_high = high > prev_high
    broke_low = low < prev_low
    code = np.where(broke_high & broke_low, _STRUCT_3,
                    np.where(broke_high, _STRUCT_2U,
                             np.where(broke_low, _STRUCT_2D, _STRUCT_1)))
    return np.where(valid, code, _STRUCT_NONE).astype(np.int64)


def is_failed(prev_high, prev_low, open_px, close_px, structure, method: str) -> np.ndarray:
    """``sg.isFailed``：方向性破位是否被拒绝（只有 2u/2d 可能 fail）。"""
    prev_high = np.asarray(prev_high, dtype=float)
    prev_low = np.asarray(prev_low, dtype=float)
    open_px = np.asarray(open_px, dtype=float)
    close_px = np.asarray(close_px, dtype=float)
    reclaimed = (close_px <= prev_high) & (close_px >= prev_low)
    against_open = np.where(structure == _STRUCT_2U, close_px <= open_px, close_px > open_px)
    if method == "Reclaim":
        rejected = reclaimed
    elif method == "Open":
        rejected = against_open
    elif method == "Reclaim + Open":
        rejected = reclaimed & against_open
    else:  # Reclaim OR Open
        rejected = reclaimed | against_open
    return np.isin(structure, [_STRUCT_2U, _STRUCT_2D]) & rejected


def is_hammer(open_px, high, low, close_px, method: str, require_color: bool) -> np.ndarray:
    """``sg.isHammer``：下方被拒绝的 K 线。"""
    o = np.asarray(open_px, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close_px, dtype=float)
    r = h - l
    open_from_low = np.where(r != 0, (o - l) / np.where(r != 0, r, 1.0), 0.0)
    close_from_low = np.where(r != 0, (c - l) / np.where(r != 0, r, 1.0), 0.0)
    body = np.abs(c - o)
    body_pct = np.where(r != 0, body / np.where(r != 0, r, 1.0), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        wick_ratio = np.where(body_pct > 0.001, (np.minimum(o, c) - l) / np.where(body > 0, body, 1.0), 0.0)
    upper_wick_pct = np.where(r != 0, (h - np.maximum(o, c)) / np.where(r != 0, r, 1.0), 0.0)
    body_center_from_high = np.where(r != 0, (h - (o + c) / 2.0) / np.where(r != 0, r, 1.0), 0.0)
    close_from_high = np.where(r != 0, (h - c) / np.where(r != 0, r, 1.0), 0.0)
    if method == "Classic":
        shape = ((body_pct <= 0.30) & (wick_ratio >= 3.0) & (upper_wick_pct <= 0.35)
                 & (body_center_from_high <= 0.33) & (close_from_high <= 0.25))
    elif method == "Pin Bar (Strict)":
        shape = (close_from_high <= 0.25) & ((h - o) / np.where(r != 0, r, 1.0) <= 0.25)
    else:  # Broad (Loose)：开收都在中点严格之上
        shape = (open_from_low > 0.50) & (close_from_low > 0.50)
    valid = np.isfinite(r) & (r != 0)
    if require_color:
        shape = shape & (c > o)
    return valid & shape


def is_shooter(open_px, high, low, close_px, method: str, require_color: bool) -> np.ndarray:
    """``sg.isShooter``：上方被拒绝的 K 线（isHammer 的镜像）。"""
    o = np.asarray(open_px, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close_px, dtype=float)
    r = h - l
    safe_r = np.where(r != 0, r, 1.0)
    open_from_low = np.where(r != 0, (o - l) / safe_r, 0.0)
    close_from_low = np.where(r != 0, (c - l) / safe_r, 0.0)
    body = np.abs(c - o)
    body_pct = np.where(r != 0, body / safe_r, 0.0)
    wick_ratio = np.where(body_pct > 0.001, (h - np.maximum(o, c)) / np.where(body > 0, body, 1.0), 0.0)
    lower_wick_pct = np.where(r != 0, (np.minimum(o, c) - l) / safe_r, 0.0)
    body_center_from_low = np.where(r != 0, ((o + c) / 2.0 - l) / safe_r, 0.0)
    if method == "Classic":
        shape = ((body_pct <= 0.30) & (wick_ratio >= 3.0) & (lower_wick_pct <= 0.35)
                 & (body_center_from_low <= 0.33) & (close_from_low <= 0.25))
    elif method == "Pin Bar (Strict)":
        shape = (close_from_low <= 0.25) & ((o - l) / safe_r <= 0.25)
    else:  # Broad (Loose)
        shape = (open_from_low < 0.50) & (close_from_low < 0.50)
    valid = np.isfinite(r) & (r != 0)
    if require_color:
        shape = shape & (c < o)
    return valid & shape


def _tf_slots(frame: pd.DataFrame, token: str) -> dict[str, np.ndarray]:
    """取某个周期的 CC（形成中）/ C1 / C2 / C3 四档 OHLC。

    ``CC`` 是**当前周期到本根为止**的聚合（Pine 里形成中的那根），``C1`` 是上一根
    已完成的周期。
    """
    open_s, high_s, low_s, close_s = ohlcv(frame, "open", "high", "low", "close")
    if token == "chart":
        return {
            "cc_open": open_s.to_numpy(dtype=float),
            "cc_high": high_s.to_numpy(dtype=float),
            "cc_low": low_s.to_numpy(dtype=float),
            "cc_close": close_s.to_numpy(dtype=float),
            "c1_open": open_s.shift(1).to_numpy(dtype=float),
            "c1_high": high_s.shift(1).to_numpy(dtype=float),
            "c1_low": low_s.shift(1).to_numpy(dtype=float),
            "c1_close": close_s.shift(1).to_numpy(dtype=float),
            "c2_open": open_s.shift(2).to_numpy(dtype=float),
            "c2_high": high_s.shift(2).to_numpy(dtype=float),
            "c2_low": low_s.shift(2).to_numpy(dtype=float),
            "c2_close": close_s.shift(2).to_numpy(dtype=float),
            "c3_high": high_s.shift(3).to_numpy(dtype=float),
            "c3_low": low_s.shift(3).to_numpy(dtype=float),
        }
    key = _period_key(pd.DatetimeIndex(frame.index), _period_size(token))
    grouped = frame.groupby(key)
    bars = pd.Series(np.asarray(key))
    aggregates = grouped.agg(open=("open", "first"), high=("high", "max"),
                             low=("low", "min"), close=("close", "last"))

    def mapped(series: pd.Series) -> np.ndarray:
        return bars.map(series).to_numpy(dtype=float)

    return {
        "cc_open": grouped["open"].transform("first").to_numpy(dtype=float),
        "cc_high": grouped["high"].cummax().to_numpy(dtype=float),
        "cc_low": grouped["low"].cummin().to_numpy(dtype=float),
        "cc_close": close_s.to_numpy(dtype=float),
        "c1_open": mapped(aggregates["open"].shift(1)),
        "c1_high": mapped(aggregates["high"].shift(1)),
        "c1_low": mapped(aggregates["low"].shift(1)),
        "c1_close": mapped(aggregates["close"].shift(1)),
        "c2_open": mapped(aggregates["open"].shift(2)),
        "c2_high": mapped(aggregates["high"].shift(2)),
        "c2_low": mapped(aggregates["low"].shift(2)),
        "c2_close": mapped(aggregates["close"].shift(2)),
        "c3_high": mapped(aggregates["high"].shift(3)),
        "c3_low": mapped(aggregates["low"].shift(3)),
    }


_STRUCT_NAMES = {0: "", 1: "1", 2: "2u", 3: "2d", 4: "3"}


def thestrat_suite(
    frame: pd.DataFrame,
    *,
    preset: str = "TheStrat Classic",
    timeframes: tuple[str, ...] | None = None,
    hammer_method: str = "Broad (Loose)",
    hammer_require_color: bool = False,
    failed2_method: str = "Reclaim",
    show_inside_reversals: bool = True,
    inside_rev_only_momo: bool = False,
    inside_rev_require_ftfc: bool = False,
    show_all_reversals: bool = True,
    reversals_require_actionable: bool = False,
    reversals_require_ftfc: bool = False,
    show_inside_continuations: bool = True,
    inside_cont_only_momo: bool = False,
    inside_cont_require_ftfc: bool = False,
    show_continuations: bool = False,
    two_two_cont_only_momo: bool = False,
    continuations_require_ftfc: bool = False,
    show_failing2: bool = False,
    p3_require_ftfc: bool = False,
    show_range_expansions: bool = False,
    range_exp_only_momo: bool = False,
    range_expansions_require_ftfc: bool = False,
    show_3_expansions: bool = False,
    three_exp_require_ftfc: bool = False,
    show_magnitude: bool = True,
    min_domino_tfs: int = 2,
) -> pd.DataFrame:
    """TheStrat Suite：逐 bar 的分型 / 形态 / Failed2 / FTFC / 信号在效。"""
    preset = choice("preset", preset, tuple(_PRESET_TFS) + ("Custom",))
    hammer_method = choice("hammer_method", hammer_method, _HAMMER_METHODS)
    failed2_method = choice("failed2_method", failed2_method, _FAILED2_METHODS)
    hammer_require_color = boolean("hammer_require_color", hammer_require_color)
    show_inside_reversals = boolean("show_inside_reversals", show_inside_reversals)
    inside_rev_only_momo = boolean("inside_rev_only_momo", inside_rev_only_momo)
    inside_rev_require_ftfc = boolean("inside_rev_require_ftfc", inside_rev_require_ftfc)
    show_all_reversals = boolean("show_all_reversals", show_all_reversals)
    reversals_require_actionable = boolean("reversals_require_actionable", reversals_require_actionable)
    reversals_require_ftfc = boolean("reversals_require_ftfc", reversals_require_ftfc)
    show_inside_continuations = boolean("show_inside_continuations", show_inside_continuations)
    inside_cont_only_momo = boolean("inside_cont_only_momo", inside_cont_only_momo)
    inside_cont_require_ftfc = boolean("inside_cont_require_ftfc", inside_cont_require_ftfc)
    show_continuations = boolean("show_continuations", show_continuations)
    two_two_cont_only_momo = boolean("two_two_cont_only_momo", two_two_cont_only_momo)
    continuations_require_ftfc = boolean("continuations_require_ftfc", continuations_require_ftfc)
    show_failing2 = boolean("show_failing2", show_failing2)
    p3_require_ftfc = boolean("p3_require_ftfc", p3_require_ftfc)
    show_range_expansions = boolean("show_range_expansions", show_range_expansions)
    range_exp_only_momo = boolean("range_exp_only_momo", range_exp_only_momo)
    range_expansions_require_ftfc = boolean("range_expansions_require_ftfc", range_expansions_require_ftfc)
    show_3_expansions = boolean("show_3_expansions", show_3_expansions)
    three_exp_require_ftfc = boolean("three_exp_require_ftfc", three_exp_require_ftfc)
    show_magnitude = boolean("show_magnitude", show_magnitude)
    min_domino_tfs = integer("min_domino_tfs", min_domino_tfs, 2, 6)

    if preset == "Custom":
        tokens = tuple(timeframes) if timeframes else ("30", "60", "D", "W", "M", "M")
        enabled = tuple(True for _ in tokens)
    else:
        tokens, enabled = _PRESET_TFS[preset]
    if len(tokens) != 6:
        raise ValueError("timeframes must provide exactly 6 entries")
    if timeframes is not None and preset != "Custom" and len(timeframes) == 6:
        tokens = tuple(timeframes)

    open_s, high_s, low_s, close_s = ohlcv(frame, "open", "high", "low", "close")
    open_a = open_s.to_numpy(dtype=float)
    high = high_s.to_numpy(dtype=float)
    low = low_s.to_numpy(dtype=float)
    close = close_s.to_numpy(dtype=float)
    idx = frame.index
    n = len(frame)

    # --- 主周期（图表本身作为一个 slot） ------------------------------------
    chart = _tf_slots(frame, "chart")
    cc_structure = structure_of(chart["c1_high"], chart["c1_low"], chart["cc_high"], chart["cc_low"])
    c1_structure = structure_of(chart["c2_high"], chart["c2_low"], chart["c1_high"], chart["c1_low"])
    c2_structure = structure_of(chart["c3_high"], chart["c3_low"], chart["c2_high"], chart["c2_low"])

    cc_above_open = chart["cc_close"] > chart["cc_open"]
    c1_above_open = chart["c1_close"] > chart["c1_open"]
    cc_inside = (chart["cc_high"] <= chart["c1_high"]) & (chart["cc_low"] >= chart["c1_low"])

    c1_hammer = is_hammer(chart["c1_open"], chart["c1_high"], chart["c1_low"], chart["c1_close"],
                          hammer_method, hammer_require_color)
    c1_shooter = is_shooter(chart["c1_open"], chart["c1_high"], chart["c1_low"], chart["c1_close"],
                            hammer_method, hammer_require_color)

    # Failed 2：CC 相对 C1 区间
    cc_is_2u = cc_structure == _STRUCT_2U
    cc_is_2d = cc_structure == _STRUCT_2D
    cc_is_3 = cc_structure == _STRUCT_3
    cc_inside_c1 = (chart["cc_close"] <= chart["c1_high"]) & (chart["cc_close"] >= chart["c1_low"])
    if failed2_method == "Open":
        is_f2u = cc_is_2u & (~cc_above_open)
        is_f2d = cc_is_2d & cc_above_open
    elif failed2_method == "Reclaim":
        is_f2u = cc_is_2u & cc_inside_c1
        is_f2d = cc_is_2d & cc_inside_c1
    elif failed2_method == "Reclaim + Open":
        is_f2u = cc_is_2u & (~cc_above_open) & cc_inside_c1
        is_f2d = cc_is_2d & cc_above_open & cc_inside_c1
    else:
        is_f2u = cc_is_2u & ((~cc_above_open) | cc_inside_c1)
        is_f2d = cc_is_2d & (cc_above_open | cc_inside_c1)
    pre_f2u = np.zeros(n, dtype=bool)
    pre_f2d = np.zeros(n, dtype=bool)
    if failed2_method in ("Open", "Reclaim + Open"):
        pre_f2u = cc_is_2u & cc_inside_c1 & cc_above_open & ~is_f2u & ~is_f2d
        pre_f2d = cc_is_2d & cc_inside_c1 & (~cc_above_open) & ~is_f2u & ~is_f2d

    # C1 / C2 的相对分类
    c1_was_2u = c1_structure == _STRUCT_2U
    c1_was_2d = c1_structure == _STRUCT_2D
    c1_is_3 = c1_structure == _STRUCT_3
    c2_was_2u = c2_structure == _STRUCT_2U
    c2_was_2d = c2_structure == _STRUCT_2D
    c2_closed_up = chart["c2_close"] > chart["c2_open"]
    c2_closed_down = chart["c2_close"] < chart["c2_open"]
    c1_was_f2d = c1_was_2d & (chart["c1_close"] > chart["c2_low"])
    c1_was_f2u = c1_was_2u & (chart["c1_close"] < chart["c2_high"])

    # --- 各周期 slot 与 FTFC ------------------------------------------------
    tf_structures: list[np.ndarray] = []
    tf_inside: list[np.ndarray] = []
    tf_sign: list[np.ndarray] = []
    tf_c1_high: list[np.ndarray] = []
    tf_c1_low: list[np.ndarray] = []
    for k in range(6):
        if not enabled[k]:
            tf_structures.append(np.full(n, _STRUCT_NONE, dtype=np.int64))
            tf_inside.append(np.zeros(n, dtype=bool))
            tf_sign.append(np.full(n, np.nan))
            tf_c1_high.append(np.full(n, np.nan))
            tf_c1_low.append(np.full(n, np.nan))
            continue
        slots = _tf_slots(frame, tokens[k])
        tf_structures.append(structure_of(slots["c1_high"], slots["c1_low"], slots["cc_high"], slots["cc_low"]))
        tf_inside.append((slots["cc_high"] <= slots["c1_high"]) & (slots["cc_low"] >= slots["c1_low"]))
        tf_sign.append(slots["cc_close"] > slots["cc_open"])
        tf_c1_high.append(slots["c1_high"])
        tf_c1_low.append(slots["c1_low"])

    ftfc_up = np.ones(n, dtype=bool)
    ftfc_down = np.ones(n, dtype=bool)
    for k in range(6):
        if not enabled[k]:
            continue
        sign = tf_sign[k]
        valid = np.isfinite(np.asarray(sign, dtype=float)) if sign.dtype != bool else np.ones(n, dtype=bool)
        sign_bool = np.asarray(sign, dtype=bool)
        ftfc_up &= ~(valid & ~sign_bool)
        ftfc_down &= ~(valid & sign_bool)

    domino_count = np.zeros(n, dtype=np.int64)
    for k in range(6):
        domino_count += np.asarray(tf_inside[k], dtype=bool).astype(np.int64)
    domino = domino_count >= min_domino_tfs

    # --- 幅度目标（C1 区间） -------------------------------------------------
    mag_high = chart["c1_high"] if show_magnitude else np.full(n, np.nan)
    mag_low = chart["c1_low"] if show_magnitude else np.full(n, np.nan)

    # --- 信号在效判定（shouldDrawC1Level 的九分支） -------------------------
    def draw_level(is_bullish: bool, i: int) -> tuple[bool, bool]:
        c2_same = c2_closed_up[i] if is_bullish else c2_closed_down[i]
        c2_opp = c2_closed_down[i] if is_bullish else c2_closed_up[i]
        momo = c1_hammer[i] if is_bullish else c1_shooter[i]
        cc_in_force = cc_is_2u[i] if is_bullish else cc_is_2d[i]
        cc_opposite = cc_is_2d[i] if is_bullish else cc_is_2u[i]
        c1_was_same = c1_was_2u[i] if is_bullish else c1_was_2d[i]
        c1_was_opp = c1_was_2d[i] if is_bullish else c1_was_2u[i]
        failed_same = is_f2u[i] if is_bullish else is_f2d[i]
        failed_opp = is_f2d[i] if is_bullish else is_f2u[i]
        ftfc_blocks = ftfc_down[i] if is_bullish else ftfc_up[i]

        inside_rev_blocks = inside_rev_require_ftfc and ftfc_blocks
        reversals_blocks = reversals_require_ftfc and ftfc_blocks
        continuations_blocks = continuations_require_ftfc and ftfc_blocks
        inside_cont_blocks = inside_cont_require_ftfc and ftfc_blocks
        expansions_blocks = range_expansions_require_ftfc and ftfc_blocks
        three_exp_blocks = three_exp_require_ftfc and ftfc_blocks
        p3_blocks = p3_require_ftfc and ftfc_blocks

        cc_is_inside_now = (not cc_is_2u[i]) and (not cc_is_2d[i]) and (not cc_is_3[i])
        inside_is_continuation = cc_inside[i] and c2_same
        inside_is_reversal = cc_inside[i] and (c2_opp or ((not c2_closed_up[i]) and (not c2_closed_down[i])))

        if cc_is_3[i]:
            return (show_3_expansions and not three_exp_blocks), False
        if show_failing2 and failed_same:
            return (not p3_blocks), False
        if show_failing2 and failed_opp:
            return (not p3_blocks), False
        if cc_inside[i]:
            if inside_is_reversal and show_inside_reversals:
                if inside_rev_only_momo and not momo:
                    return False, False
                return (not inside_rev_blocks), False
            if inside_is_reversal and not show_inside_reversals:
                return False, False
            if inside_is_continuation and show_inside_continuations:
                if inside_cont_only_momo and not momo:
                    return False, False
                return (not inside_cont_blocks), True
            if inside_is_continuation and not show_inside_continuations:
                return False, False
            if show_inside_reversals:
                if inside_rev_only_momo and not momo:
                    return False, False
                return (not inside_rev_blocks), False
            return False, False
        if c1_is_3[i] and cc_in_force and show_range_expansions:
            if range_exp_only_momo and not momo:
                return False, False
            return (not expansions_blocks), True
        if c1_is_3[i] and cc_is_inside_now and show_range_expansions:
            if range_exp_only_momo and not momo:
                return False, False
            return (not expansions_blocks), True
        if momo and not (c1_is_3[i] and not show_range_expansions):
            is_momo_reversal = c1_was_opp or (c2_opp and not c1_was_same) or \
                ((not c1_was_2u[i]) and (not c1_was_2d[i]) and not c2_same)
            is_momo_cont = c1_was_same or (c2_same and not c1_was_opp)
            if cc_opposite and c1_was_opp:
                return False, False
            if is_momo_reversal and not cc_opposite:
                return (not reversals_blocks), False
            if is_momo_cont and show_continuations:
                return (not continuations_blocks), True
            # 落到后面的通用分支
        if c1_was_opp and cc_in_force and show_all_reversals:
            if momo:
                return (not reversals_blocks), False
            if reversals_require_actionable:
                return False, False
            return (not reversals_blocks), False
        if c1_was_opp and show_all_reversals and not cc_opposite:
            if momo:
                return (not reversals_blocks), False
            if reversals_require_actionable:
                return False, False
            return (not reversals_blocks), False
        if c1_was_same and cc_in_force and show_continuations:
            if two_two_cont_only_momo and not momo:
                return False, False
            return (not continuations_blocks), True
        if c1_was_same and cc_is_inside_now and show_continuations:
            if two_two_cont_only_momo and not momo:
                return False, False
            return (not continuations_blocks), True
        return False, False

    high_in_force = np.zeros(n, dtype=bool)
    low_in_force = np.zeros(n, dtype=bool)
    high_continuation = np.zeros(n, dtype=bool)
    low_continuation = np.zeros(n, dtype=bool)
    for i in range(n):
        high_in_force[i], high_continuation[i] = draw_level(True, i)
        low_in_force[i], low_continuation[i] = draw_level(False, i)

    # --- 止损位 -------------------------------------------------------------
    stop_high = np.where(high_in_force, chart["c1_low"], np.nan)
    stop_low = np.where(low_in_force, chart["c1_high"], np.nan)

    # --- 各周期信号在效计数（不含图表 slot） --------------------------------
    bull_count = np.zeros(n, dtype=np.int64)
    bear_count = np.zeros(n, dtype=np.int64)
    for k in range(6):
        bull_count += (tf_structures[k] == _STRUCT_2U).astype(np.int64)
        bear_count += (tf_structures[k] == _STRUCT_2D).astype(np.int64)

    lead_direction = np.zeros(n, dtype=np.int64)
    for k in range(5, -1, -1):
        if not enabled[k]:
            continue
        structure = tf_structures[k]
        lead_direction = np.where((lead_direction == 0) & (structure == _STRUCT_2U), 1, lead_direction)
        lead_direction = np.where((lead_direction == 0) & (structure == _STRUCT_2D), -1, lead_direction)

    def as_int(values) -> pd.Series:
        return pd.Series(np.asarray(values).astype(np.int64), index=idx).astype("Int64")

    data = {
        "thestrat_cc_structure": as_int(cc_structure),
        "thestrat_c1_structure": as_int(c1_structure),
        "thestrat_c2_structure": as_int(c2_structure),
        "thestrat_cc_above_open": as_int(cc_above_open),
        "thestrat_c1_above_open": as_int(c1_above_open),
        "thestrat_cc_inside": as_int(cc_inside),
        "thestrat_c1_hammer": as_int(c1_hammer),
        "thestrat_c1_shooter": as_int(c1_shooter),
        "thestrat_cc_is_2u": as_int(cc_is_2u),
        "thestrat_cc_is_2d": as_int(cc_is_2d),
        "thestrat_cc_is_3": as_int(cc_is_3),
        "thestrat_c1_was_2u": as_int(c1_was_2u),
        "thestrat_c1_was_2d": as_int(c1_was_2d),
        "thestrat_c1_is_3": as_int(c1_is_3),
        "thestrat_c2_closed_up": as_int(c2_closed_up),
        "thestrat_c2_closed_down": as_int(c2_closed_down),
        "thestrat_c1_was_f2u": as_int(c1_was_f2u),
        "thestrat_c1_was_f2d": as_int(c1_was_f2d),
        "thestrat_f2u": as_int(is_f2u),
        "thestrat_f2d": as_int(is_f2d),
        "thestrat_pre_f2u": as_int(pre_f2u),
        "thestrat_pre_f2d": as_int(pre_f2d),
        "thestrat_ftfc_up": as_int(ftfc_up),
        "thestrat_ftfc_down": as_int(ftfc_down),
        "thestrat_domino_count": as_int(domino_count),
        "thestrat_domino": as_int(domino),
        "thestrat_mag_high": mag_high,
        "thestrat_mag_low": mag_low,
        "thestrat_signal_high_in_force": as_int(high_in_force),
        "thestrat_signal_low_in_force": as_int(low_in_force),
        "thestrat_high_continuation": as_int(high_continuation),
        "thestrat_low_continuation": as_int(low_continuation),
        "thestrat_stop_bull": stop_high,
        "thestrat_stop_bear": stop_low,
        "thestrat_tf_bull_count": as_int(bull_count),
        "thestrat_tf_bear_count": as_int(bear_count),
        "thestrat_lead_direction": as_int(lead_direction),
    }
    for k in range(6):
        suffix = str(k + 1)
        data["thestrat_tf" + suffix + "_structure"] = as_int(tf_structures[k])
        data["thestrat_tf" + suffix + "_inside"] = as_int(tf_inside[k])
        data["thestrat_tf" + suffix + "_c1_high"] = tf_c1_high[k]
        data["thestrat_tf" + suffix + "_c1_low"] = tf_c1_low[k]
    return pd.DataFrame(data, index=idx)
