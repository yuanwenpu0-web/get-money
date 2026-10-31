"""SMC Structure & Expectation（结构 + 预期 + BOS/OB 确认）移植。

Port of ``data/pine-collection/indicators/market_structure/smc_structure_expectation/
smc_structure_expectation.pine``.

Pine 原脚本是四层结构，本移植逐层保留：

**Layer 1 · 市场结构**：确认摆动枢轴（外部主摆动 + 内部择时），按 HH/HL/LH/LL 分类，
维护「外部结构」与「内部结构」两个状态，并给出对齐 / 冲突判定。
注意结构只在**单侧**改变：新高不单独把结构改成多头，新低也不单独改成空头 ——
必须由 HL 或 LH 触发，这是原脚本刻意的保守设计。

**Layer 2 · 预期**：把结构翻译成「假设」而不是信号 ——
HH → 等 HL、LL → 等 LH、HL → 等延续、LH → 等延续；带超时过期，
并在价格反向击穿参考位时降级为 Expect Failure。

**Layer 3 · 确认**：只在「与预期一致」时确认 BOS（Break of Structure），
带 Range-Mismatch 守卫（HL 必须在 HH 之后）、相对腿强度（冲量/修正比）、
Aggro 模式的待确认状态机；随后从「被破摆动 → BOS」这段区间里提取最强冲量 bar
之前的最后一根反向 K 线作为 Order Block，经 判定位（premium/discount）→ 尺寸 →
量能 → FVG 六级过滤后打分（displacement / FVG / 量能 / 位置，再加结构对齐与腿强度），
并跟踪 Tap/Deep/Failure 反应类型、30% 渗透缓解、破位失效与 TTL 过期。

**Layer 4 · 警告**：摆动位击穿、长上影 + RSI 超买、长下影 + RSI 超卖 ——
纯提示，不改变预期。

未移植：绘图、dashboard、``log.info`` 调试日志、告警。日志里那些「潜在评分」
只用于过滤日志噪声，不影响任何状态。

输出 ``smc_`` 前缀列：结构/预期/BOS/OB 的数值与状态码、拒绝计数、警告标志。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    boolean,
    choice,
    integer,
    number,
    ohlcv,
    pivothigh,
    pivotlow,
    rsi,
    sma,
)

_STRUCTURE_NEUTRAL = 0
_STRUCTURE_BULL = 1
_STRUCTURE_BEAR = -1

_SWING_NONE = 0
_SWING_FIRST = 5
_SWING_HH = 1
_SWING_HL = 2
_SWING_LH = 3
_SWING_LL = 4

_EXPECT_NONE = 0
_EXPECT_HL = 1
_EXPECT_LH = -1
_EXPECT_CONTINUATION = 2
_EXPECT_FAILURE = 3

_OB_NONE = 0
_OB_UNTOUCHED = 1
_OB_TAP = 2
_OB_DEEP = 3
_OB_FAILURE = 4


def _cross_over_level(series: np.ndarray, level: float, prev_level: float, i: int) -> bool:
    """``ta.crossover(series, level)``：位上穿。

    ``prev_level`` 必须是**上一根 bar 收盘时**该位的取值（Pine 的 ``level[1]``）——
    位在当根被刷新时，用同一个值去比较前一根会漏掉/误判交叉。
    """
    if i <= 0 or not (np.isfinite(level) and np.isfinite(prev_level)):
        return False
    now, prev = series[i], series[i - 1]
    if not (np.isfinite(now) and np.isfinite(prev)):
        return False
    return now > level and prev <= prev_level


def _cross_under_level(series: np.ndarray, level: float, prev_level: float, i: int) -> bool:
    """``ta.crossunder(series, level)``：位下穿（同上，需要上一根的位值）。"""
    if i <= 0 or not (np.isfinite(level) and np.isfinite(prev_level)):
        return False
    now, prev = series[i], series[i - 1]
    if not (np.isfinite(now) and np.isfinite(prev)):
        return False
    return now < level and prev >= prev_level


def _bar_seconds(frame: pd.DataFrame) -> float:
    """图表周期（秒），用于位移参数与 OB 上限的自适应。"""
    if not isinstance(frame.index, pd.DatetimeIndex) or len(frame.index) < 2:
        return 3600.0
    deltas = pd.Series(frame.index).diff().dt.total_seconds().dropna()
    median = float(deltas.median()) if len(deltas) else 3600.0
    return median if median > 0 else 3600.0


def smc_structure_and_expectation(
    frame: pd.DataFrame,
    *,
    structure_lookback: int = 5,
    internal_lookback: int = 3,
    enable_internal_structure: bool = True,
    expectation_expiry: int = 30,
    bos_mode: str = "Close-cross",
    require_confirm: bool = True,
    confirm_bars: int = 2,
    enable_order_blocks: bool = True,
    use_auto_displacement: bool = True,
    min_displacement: float = 1.0,
    displacement_window: int = 3,
    min_ob_range: float = 0.3,
    size_filter_mode: str = "Score Penalty",
    use_auto_max_range: bool = True,
    manual_max_ob_range: float = 1.0,
    require_fvg: bool = False,
    ob_location_threshold: float = 0.58,
    ob_ttl: int = 100,
    mitigation_penetration: float = 0.25,
    use_volume_filter: bool = False,
    volume_floor: float = 0.6,
    enable_swing_level_break: bool = True,
    enable_early_top: bool = True,
    enable_early_bottom: bool = True,
    wick_threshold: float = 0.5,
    rsi_length: int = 14,
    rsi_overbought: int = 70,
    rsi_oversold: int = 30,
    warning_cooldown: int = 30,
    atr_length: int = 14,
    volume_length: int = 20,
    show_layer4: bool = True,
    volume_is_real: bool | None = None,
) -> pd.DataFrame:
    """SMC 结构 / 预期 / BOS+OB 确认 / 警告。

    Args:
        frame: OHLCV（``open``/``high``/``low``/``close``，``volume`` 可选）。
        structure_lookback: 外部主摆动的枢轴腿长（2..20）。
        internal_lookback: 内部择时枢轴的腿长（2..10）。
        enable_internal_structure: 是否跟踪内部结构并做对齐/冲突判定。
        expectation_expiry: 预期过期 bar 数（10..100）。
        bos_mode: ``Close-cross``（保守，收盘破位）或 ``High/Low-cross``（激进，影线破位）。
        require_confirm: 激进模式下是否要求 N 根内收盘确认。
        confirm_bars: 激进模式的确认窗口（1..5）。
        enable_order_blocks: 是否启用 Order Block 检测。
        use_auto_displacement: 按周期自动推导位移窗口与最小位移。
        min_displacement: 手动最小位移（ATR 倍数）。
        displacement_window: 手动位移回看窗口。
        min_ob_range: OB 最小高度（ATR 倍数）。
        size_filter_mode: ``Hard Limit`` / ``Score Penalty`` / ``Disabled``。
        use_auto_max_range: 按周期自动推导 OB 最大高度。
        manual_max_ob_range: 手动的 OB 最大高度（ATR 倍数）。
        require_fvg: 是否要求 FVG 重叠。
        ob_location_threshold: Premium/Discount 分界（0.5..0.8）。
        ob_ttl: OB 存活 bar 数（20..300）。
        mitigation_penetration: 缓解所需的渗透比例（0.1..1.0）。
        use_volume_filter: 是否要求量能 > SMA 的 ``volume_floor`` 倍。
        volume_floor: 量能地板（相对 SMA）。
        enable_swing_level_break: 是否启用摆动位击穿警告。
        enable_early_top: 是否启用早期顶部警告。
        enable_early_bottom: 是否启用早期底部警告。
        wick_threshold: 早期信号的影线占比阈值（0.3..0.8）。
        rsi_length: 警告用 RSI 长度（5..50）。
        rsi_overbought: RSI 超买阈值（60..90）。
        rsi_oversold: RSI 超卖阈值（10..40）。
        warning_cooldown: 警告冷却 bar 数（10..100）。
        atr_length: ATR 长度（5..50）。
        volume_length: 量能 SMA 长度（10..100）。
        show_layer4: Layer 4 总开关（关闭时警告不再产生）。
        volume_is_real: ``False`` 表示量能不可信，此时相对量能为 na（不参与打分）。

    Returns:
        ``pd.DataFrame``，索引与 ``frame`` 一致，列为 ``smc_*``。

    Raises:
        ValueError: 参数越界或取值不在允许集合内。
    """
    structure_lookback = integer("structure_lookback", structure_lookback, 2, 20)
    internal_lookback = integer("internal_lookback", internal_lookback, 2, 10)
    enable_internal_structure = boolean("enable_internal_structure", enable_internal_structure)
    expectation_expiry = integer("expectation_expiry", expectation_expiry, 10, 100)
    bos_mode = choice("bos_mode", bos_mode, ("Close-cross", "High/Low-cross"))
    require_confirm = boolean("require_confirm", require_confirm)
    confirm_bars = integer("confirm_bars", confirm_bars, 1, 5)
    enable_order_blocks = boolean("enable_order_blocks", enable_order_blocks)
    use_auto_displacement = boolean("use_auto_displacement", use_auto_displacement)
    min_displacement = number("min_displacement", min_displacement, 0.5, 5.0)
    displacement_window = integer("displacement_window", displacement_window, 1, 10)
    min_ob_range = number("min_ob_range", min_ob_range, 0.1, 2.0)
    size_filter_mode = choice("size_filter_mode", size_filter_mode, ("Hard Limit", "Score Penalty", "Disabled"))
    use_auto_max_range = boolean("use_auto_max_range", use_auto_max_range)
    manual_max_ob_range = number("manual_max_ob_range", manual_max_ob_range, 0.5, 3.0)
    require_fvg = boolean("require_fvg", require_fvg)
    ob_location_threshold = number("ob_location_threshold", ob_location_threshold, 0.5, 0.8)
    ob_ttl = integer("ob_ttl", ob_ttl, 20, 300)
    mitigation_penetration = number("mitigation_penetration", mitigation_penetration, 0.1, 1.0)
    use_volume_filter = boolean("use_volume_filter", use_volume_filter)
    volume_floor = number("volume_floor", volume_floor, 0.3, 1.5)
    enable_swing_level_break = boolean("enable_swing_level_break", enable_swing_level_break)
    enable_early_top = boolean("enable_early_top", enable_early_top)
    enable_early_bottom = boolean("enable_early_bottom", enable_early_bottom)
    wick_threshold = number("wick_threshold", wick_threshold, 0.3, 0.8)
    rsi_length = integer("rsi_length", rsi_length, 5, 50)
    rsi_overbought = integer("rsi_overbought", rsi_overbought, 60, 90)
    rsi_oversold = integer("rsi_oversold", rsi_oversold, 10, 40)
    warning_cooldown = integer("warning_cooldown", warning_cooldown, 10, 100)
    atr_length = integer("atr_length", atr_length, 5, 50)
    volume_length = integer("volume_length", volume_length, 10, 100)
    show_layer4 = boolean("show_layer4", show_layer4)

    open_s, high_s, low_s, close_s = ohlcv(frame, "open", "high", "low", "close")
    open_ = open_s.to_numpy(dtype=float)
    high = high_s.to_numpy(dtype=float)
    low = low_s.to_numpy(dtype=float)
    close = close_s.to_numpy(dtype=float)
    n = len(frame)

    bar_seconds = _bar_seconds(frame)
    if use_auto_displacement:
        if bar_seconds >= 3600:
            disp_window, min_disp = 3, 1.0
        elif bar_seconds >= 900:
            disp_window, min_disp = 4, 0.9
        else:
            disp_window, min_disp = 5, 0.8
    else:
        disp_window, min_disp = displacement_window, min_displacement
    if use_auto_max_range:
        if bar_seconds >= 604800:
            max_ob_range = 1.5
        elif bar_seconds >= 86400:
            max_ob_range = 1.0
        elif bar_seconds >= 60:
            max_ob_range = 1.3
        else:
            max_ob_range = 1.0
    else:
        max_ob_range = manual_max_ob_range

    atr_a = atr(frame, atr_length).to_numpy(dtype=float)
    rsi_a = rsi(close_s, rsi_length).to_numpy(dtype=float)

    if "volume" in frame.columns and volume_is_real is not False:
        volume = np.nan_to_num(frame["volume"].to_numpy(dtype=float), nan=0.0)
        volume_real = bool(np.nansum(volume) > 0)
    else:
        volume = np.zeros(n, dtype=float)
        volume_real = False
    volume_sma = sma(pd.Series(volume), volume_length).to_numpy(dtype=float)

    pivot_high = pivothigh(high_s, structure_lookback, structure_lookback).to_numpy(dtype=float)
    pivot_low = pivotlow(low_s, structure_lookback, structure_lookback).to_numpy(dtype=float)
    if enable_internal_structure:
        int_pivot_high = pivothigh(high_s, internal_lookback, internal_lookback).to_numpy(dtype=float)
        int_pivot_low = pivotlow(low_s, internal_lookback, internal_lookback).to_numpy(dtype=float)
    else:
        int_pivot_high = np.full(n, np.nan)
        int_pivot_low = np.full(n, np.nan)

    # --- 输出容器 -----------------------------------------------------------
    cols: dict[str, np.ndarray] = {name: np.full(n, np.nan) for name in (
        "swing_high", "swing_low", "expectation_price", "bos_swing_high", "bos_swing_low",
        "bos_prior_high", "bos_prior_low", "bos_range_high", "bos_range_low", "leg_strength",
        "bar_displacement", "relative_volume", "rsi", "swing_break_level", "atr",
        "bull_ob_top", "bull_ob_bottom", "bull_ob_score", "bull_ob_priority",
        "bull_ob_penetration", "bear_ob_top", "bear_ob_bottom", "bear_ob_score",
        "bear_ob_priority", "bear_ob_penetration",
    )}
    flags: dict[str, np.ndarray] = {name: np.zeros(n, dtype=np.int64) for name in (
        "structure_code", "internal_structure_code", "last_high_type", "last_low_type",
        "swing_high_bar", "swing_low_bar", "structure_aligned", "structure_conflict",
        "expectation_code", "expectation_active", "expectation_expired",
        "expect_hl", "expect_lh", "expect_continuation", "expect_failure",
        "bullish_bos", "bearish_bos", "pending_bull", "pending_bear",
        "has_bull_fvg", "has_bear_fvg", "bull_ob_state", "bull_ob_mitigated", "bull_ob_active",
        "bear_ob_state", "bear_ob_mitigated", "bear_ob_active",
        "confirmed_bullish", "confirmed_bearish",
        "reject_range_mismatch", "reject_displacement", "reject_location", "reject_volume",
        "reject_size", "reject_fvg", "reject_segment", "bos_attempts",
        "swing_break_bull", "swing_break_bear", "early_top", "early_bottom",
    )}

    # --- Layer 1 状态 -------------------------------------------------------
    last_swing_high = math.nan
    last_swing_low = math.nan
    last_high_bar = -1
    last_low_bar = -1
    structure = _STRUCTURE_NEUTRAL
    last_swing_type = 0
    last_high_type = 0
    last_low_type = 0
    int_swing_high = math.nan
    int_swing_low = math.nan
    int_structure = _STRUCTURE_NEUTRAL
    int_high_type = 0
    int_low_type = 0

    # --- Layer 2 状态 -------------------------------------------------------
    expectation = _EXPECT_NONE
    expectation_price = math.nan
    expectation_bar = -1
    expectation_active = False

    # --- Layer 3 状态 -------------------------------------------------------
    bos_swing_high = math.nan
    bos_swing_low = math.nan
    bos_prior_low = math.nan
    bos_prior_high = math.nan
    prev_bos_high = math.nan
    prev_bos_low = math.nan
    bos_trigger_high_bar = -1
    bos_trigger_low_bar = -1
    leg_strength = math.nan
    bos_range_high = math.nan
    bos_range_low = math.nan
    pending_bull = False
    pending_bear = False
    pending_until_bull = -1
    pending_until_bear = -1
    pending_level_bull = math.nan
    pending_level_bear = math.nan
    pending_swing_bar_bull = -1
    pending_swing_bar_bear = -1

    bull_ob: dict | None = None
    bear_ob: dict | None = None
    rejects = {
        "reject_range_mismatch": 0,
        "reject_displacement": 0,
        "reject_location": 0,
        "reject_volume": 0,
        "reject_size": 0,
        "reject_fvg": 0,
        "reject_segment": 0,
        "bos_attempts": 0,
    }

    # --- Layer 4 状态 -------------------------------------------------------
    last_swing_break_bar = -1
    last_early_top_bar = -1
    last_early_bottom_bar = -1

    for i in range(n):
        price = close[i]
        atr_i = atr_a[i] if np.isfinite(atr_a[i]) and atr_a[i] > 0 else math.nan

        # ================= Layer 1：结构 ====================================
        swing_high_before = last_swing_high
        swing_low_before = last_swing_low

        if not np.isnan(pivot_high[i]):
            previous_high = last_swing_high
            if np.isnan(previous_high):
                # Pine 把首个枢轴标成 "First High"，它不属于 HH/LH，因此不更新 BOS 摆动位
                last_swing_type = _SWING_FIRST
                last_high_type = _SWING_FIRST
                structure = _STRUCTURE_NEUTRAL
            elif pivot_high[i] > previous_high:
                last_swing_type = _SWING_HH
                last_high_type = _SWING_HH
            elif pivot_high[i] < previous_high:
                last_swing_type = _SWING_LH
                last_high_type = _SWING_LH
                if structure != _STRUCTURE_BEAR:
                    structure = _STRUCTURE_BEAR
            last_swing_high = float(pivot_high[i])
            last_high_bar = i - structure_lookback

        if not np.isnan(pivot_low[i]):
            previous_low = last_swing_low
            if np.isnan(previous_low):
                last_swing_type = _SWING_FIRST
                last_low_type = _SWING_FIRST
                structure = _STRUCTURE_NEUTRAL
            elif pivot_low[i] > previous_low:
                last_swing_type = _SWING_HL
                last_low_type = _SWING_HL
                if structure != _STRUCTURE_BULL:
                    structure = _STRUCTURE_BULL
            elif pivot_low[i] < previous_low:
                last_swing_type = _SWING_LL
                last_low_type = _SWING_LL
            last_swing_low = float(pivot_low[i])
            last_low_bar = i - structure_lookback

        if enable_internal_structure:
            if not np.isnan(int_pivot_high[i]):
                if not np.isnan(int_swing_high):
                    if int_pivot_high[i] > int_swing_high:
                        int_high_type = _SWING_HH
                    else:
                        int_high_type = _SWING_LH
                        if int_structure != _STRUCTURE_BEAR:
                            int_structure = _STRUCTURE_BEAR
                int_swing_high = float(int_pivot_high[i])
            if not np.isnan(int_pivot_low[i]):
                if not np.isnan(int_swing_low):
                    if int_pivot_low[i] > int_swing_low:
                        int_low_type = _SWING_HL
                        if int_structure != _STRUCTURE_BULL:
                            int_structure = _STRUCTURE_BULL
                    else:
                        int_low_type = _SWING_LL
                int_swing_low = float(int_pivot_low[i])

        aligned = False
        conflict = False
        if enable_internal_structure:
            ext_bull, ext_bear = structure == _STRUCTURE_BULL, structure == _STRUCTURE_BEAR
            int_bull, int_bear = int_structure == _STRUCTURE_BULL, int_structure == _STRUCTURE_BEAR
            aligned = (ext_bull and int_bull) or (ext_bear and int_bear)
            conflict = (ext_bull and int_bear) or (ext_bear and int_bull)

        is_bull_structure = structure == _STRUCTURE_BULL
        is_bear_structure = structure == _STRUCTURE_BEAR

        # ================= Layer 2：预期 ====================================
        expectation_expired = False
        if not np.isnan(pivot_high[i]) and last_swing_type == _SWING_HH and is_bull_structure:
            if not expectation_active or expectation != _EXPECT_HL:
                expectation = _EXPECT_HL
                expectation_price = last_swing_low
                expectation_bar = i
                expectation_active = True
        if not np.isnan(pivot_low[i]) and last_swing_type == _SWING_LL and is_bear_structure:
            if not expectation_active or expectation != _EXPECT_LH:
                expectation = _EXPECT_LH
                expectation_price = last_swing_high
                expectation_bar = i
                expectation_active = True
        if not np.isnan(pivot_low[i]) and last_swing_type == _SWING_HL and is_bull_structure:
            if not expectation_active or expectation != _EXPECT_CONTINUATION:
                expectation = _EXPECT_CONTINUATION
                expectation_price = last_swing_high
                expectation_bar = i
                expectation_active = True
        if not np.isnan(pivot_high[i]) and last_swing_type == _SWING_LH and is_bear_structure:
            if not expectation_active or expectation != _EXPECT_CONTINUATION:
                expectation = _EXPECT_CONTINUATION
                expectation_price = last_swing_low
                expectation_bar = i
                expectation_active = True

        if expectation_active and expectation_bar >= 0:
            if i - expectation_bar > expectation_expiry:
                expectation_active = False
                expectation = _EXPECT_NONE
                expectation_price = math.nan
                expectation_expired = True

        expect_hl = expectation_active and expectation == _EXPECT_HL
        expect_lh = expectation_active and expectation == _EXPECT_LH
        expect_continuation = expectation_active and expectation == _EXPECT_CONTINUATION
        expect_failure = expectation_active and expectation == _EXPECT_FAILURE

        if expect_hl and np.isfinite(expectation_price) and price < expectation_price:
            expectation = _EXPECT_FAILURE
            expect_hl = False
            expect_failure = True
        if expect_lh and np.isfinite(expectation_price) and price > expectation_price:
            expectation = _EXPECT_FAILURE
            expect_lh = False
            expect_failure = True

        # ================= Layer 3：BOS =====================================
        if not np.isnan(pivot_high[i]) and last_high_type in (_SWING_HH, _SWING_LH):
            bos_swing_high = float(pivot_high[i])
            bos_prior_low = swing_low_before
        if not np.isnan(pivot_low[i]) and last_low_type in (_SWING_HL, _SWING_LL):
            bos_swing_low = float(pivot_low[i])
            bos_prior_high = swing_high_before

        bull_bos_event = False
        bear_bos_event = False
        bull_bos = False
        bear_bos = False

        aggro = bos_mode == "High/Low-cross"
        cross_close_over = _cross_over_level(close, bos_swing_high, prev_bos_high, i)
        cross_high_over = _cross_over_level(high, bos_swing_high, prev_bos_high, i)
        cross_close_under = _cross_under_level(close, bos_swing_low, prev_bos_low, i)
        cross_low_under = _cross_under_level(low, bos_swing_low, prev_bos_low, i)

        if np.isfinite(bos_swing_high):
            if not aggro:
                if cross_close_over:
                    bull_bos_event = True
                    bos_trigger_high_bar = last_high_bar
            elif require_confirm:
                if cross_high_over and not pending_bull:
                    pending_bull = True
                    pending_until_bull = i + confirm_bars
                    pending_level_bull = bos_swing_high
                    pending_swing_bar_bull = last_high_bar
            else:
                if cross_high_over:
                    bull_bos_event = True
                    bos_trigger_high_bar = last_high_bar

        if np.isfinite(bos_swing_low):
            if not aggro:
                if cross_close_under:
                    bear_bos_event = True
                    bos_trigger_low_bar = last_low_bar
            elif require_confirm:
                if cross_low_under and not pending_bear:
                    pending_bear = True
                    pending_until_bear = i + confirm_bars
                    pending_level_bear = bos_swing_low
                    pending_swing_bar_bear = last_low_bar
            else:
                if cross_low_under:
                    bear_bos_event = True
                    bos_trigger_low_bar = last_low_bar

        if pending_bull:
            if i > pending_until_bull:
                pending_bull = False
                pending_until_bull = -1
                pending_level_bull = math.nan
                pending_swing_bar_bull = -1
            elif price > pending_level_bull:
                bull_bos_event = True
                pending_bull = False
                pending_until_bull = -1
                bos_trigger_high_bar = pending_swing_bar_bull
                pending_level_bull = math.nan
                pending_swing_bar_bull = -1

        if pending_bear:
            if i > pending_until_bear:
                pending_bear = False
                pending_until_bear = -1
                pending_level_bear = math.nan
                pending_swing_bar_bear = -1
            elif price < pending_level_bear:
                bear_bos_event = True
                pending_bear = False
                pending_until_bear = -1
                bos_trigger_low_bar = pending_swing_bar_bear
                pending_level_bear = math.nan
                pending_swing_bar_bear = -1

        current_leg_strength = leg_strength
        if bull_bos_event:
            if (expect_hl or expect_continuation) and is_bull_structure:
                range_ok = last_low_bar >= 0 and bos_trigger_high_bar >= 0 and last_low_bar > bos_trigger_high_bar
                if range_ok:
                    bull_bos = True
                    bos_range_high = bos_swing_high
                    bos_range_low = last_swing_low
                    impulse = bos_swing_high - bos_prior_low
                    correction = bos_swing_high - last_swing_low
                    if np.isfinite(impulse) and correction > 0:
                        current_leg_strength = impulse / correction
                    bos_swing_high = math.nan
                else:
                    rejects["reject_range_mismatch"] += 1

        if bear_bos_event:
            if (expect_lh or expect_continuation) and is_bear_structure:
                range_ok = last_high_bar >= 0 and bos_trigger_low_bar >= 0 and last_high_bar > bos_trigger_low_bar
                if range_ok:
                    bear_bos = True
                    bos_range_low = bos_swing_low
                    bos_range_high = last_swing_high
                    impulse = bos_prior_high - bos_swing_low
                    correction = last_swing_high - bos_swing_low
                    if np.isfinite(impulse) and correction > 0:
                        current_leg_strength = impulse / correction
                    bos_swing_low = math.nan
                else:
                    rejects["reject_range_mismatch"] += 1
        leg_strength = current_leg_strength

        # ================= 位移 / FVG / OB 创建 =============================
        displacement = 0.0
        if i >= disp_window - 1 and np.isfinite(atr_i):
            max_range = 0.0
            for k in range(i - disp_window + 1, i + 1):
                span = high[k] - low[k]
                if span > max_range:
                    max_range = span
            displacement = max_range / atr_i

        has_bull_fvg = i >= 2 and low[i] > high[i - 2]
        has_bear_fvg = i >= 2 and high[i] < low[i - 2]

        def priority_of(base: float) -> float:
            """Pine ``calculateOBPriority``：结构对齐 + 腿强度的加权。"""
            value = base
            if enable_internal_structure:
                if aligned:
                    value += 30.0
                elif conflict:
                    value -= 15.0
            if np.isfinite(current_leg_strength):
                if current_leg_strength > 1.5:
                    value += 25.0
                elif current_leg_strength > 1.0:
                    value += 15.0
            return min(value, 100.0)

        def extract_segment(is_bull: bool, swing_bar: int) -> tuple[float, float, int, bool]:
            """从「摆动 → BOS」这段里找最强冲量 bar 之前的最后一根反向 K。"""
            swing_offset = i - swing_bar
            if swing_offset <= 0:
                return math.nan, math.nan, -1, False
            strongest_offset = 0
            max_impulse = 0.0
            for k in range(0, swing_offset):
                bar = i - k
                impulse = (close[bar] - open_[bar]) if is_bull else (open_[bar] - close[bar])
                if impulse > max_impulse:
                    max_impulse = impulse
                    strongest_offset = k
            for k in range(strongest_offset, swing_offset):
                bar = i - k
                opposite = (close[bar] < open_[bar]) if is_bull else (close[bar] > open_[bar])
                if opposite:
                    return float(high[bar]), float(low[bar]), k, True
            return math.nan, math.nan, -1, False

        def score_of(is_bull: bool, ob_top: float, ob_bottom: float, rel_vol: float,
                     fvg_ok: bool, ob_range: float, size_ok: bool) -> float:
            """OB 新鲜度评分（位移 / FVG / 量能 / 位置 + 尺寸惩罚）。"""
            score = 15.0
            if displacement >= 2.0:
                score += 30.0
            elif displacement >= 1.5:
                score += 20.0
            else:
                score += 10.0
            if fvg_ok:
                score += 20.0
            if np.isfinite(rel_vol):
                if rel_vol > 2.0:
                    score += 15.0
                elif rel_vol > 1.5:
                    score += 10.0
                elif rel_vol > 1.0:
                    score += 5.0
            if np.isfinite(bos_range_high) and np.isfinite(bos_range_low):
                mid = (bos_range_high + bos_range_low) / 2.0
                span = bos_range_high - bos_range_low
                if is_bull:
                    if ob_bottom < bos_range_low + span * 0.38:
                        score += 20.0
                    elif ob_bottom < mid:
                        score += 10.0
                else:
                    if ob_top > bos_range_low + span * 0.62:
                        score += 20.0
                    elif ob_top > mid:
                        score += 10.0
            if size_filter_mode == "Score Penalty" and not size_ok and np.isfinite(atr_i):
                size_ratio = ob_range / (max_ob_range * atr_i)
                penalty = (size_ratio - 1.0) * 20.0
                score -= min(penalty, 30.0)
            return score

        if enable_order_blocks:
            if bull_bos:
                rejects["bos_attempts"] += 1
                range_ok = last_low_bar >= 0 and bos_trigger_high_bar >= 0 and last_low_bar > bos_trigger_high_bar
                if not range_ok:
                    rejects["reject_range_mismatch"] += 1
                else:
                    ob_top, ob_bottom, ob_offset, found = extract_segment(True, bos_trigger_high_bar)
                    if not found:
                        rejects["reject_segment"] += 1
                    elif displacement < min_disp:
                        rejects["reject_displacement"] += 1
                    else:
                        location_ok = True
                        if np.isfinite(bos_range_high) and np.isfinite(bos_range_low):
                            threshold = bos_range_low + (bos_range_high - bos_range_low) * ob_location_threshold
                            location_ok = ob_bottom < threshold
                        if not location_ok:
                            rejects["reject_location"] += 1
                        else:
                            ob_range = ob_top - ob_bottom
                            size_ok = (
                                np.isfinite(atr_i)
                                and ob_range <= max_ob_range * atr_i
                                and ob_range >= min_ob_range * atr_i
                            )
                            proceed = True
                            if size_filter_mode == "Hard Limit" and not size_ok:
                                rejects["reject_size"] += 1
                                proceed = False
                            if proceed:
                                rel_vol = (
                                    volume[i - ob_offset] / volume_sma[i - ob_offset]
                                    if volume_real and np.isfinite(volume_sma[i - ob_offset])
                                    and volume_sma[i - ob_offset] > 0
                                    else math.nan
                                )
                                volume_ok = (not use_volume_filter) or (not volume_real) or (
                                    np.isfinite(rel_vol) and rel_vol > volume_floor
                                )
                                if not volume_ok:
                                    rejects["reject_volume"] += 1
                                elif require_fvg and not has_bull_fvg:
                                    rejects["reject_fvg"] += 1
                                else:
                                    score = score_of(True, ob_top, ob_bottom, rel_vol, has_bull_fvg, ob_range, size_ok)
                                    bull_ob = {
                                        "creation_bar": i - ob_offset,
                                        "top": ob_top,
                                        "bottom": ob_bottom,
                                        "bos_bar": i,
                                        "mitigated": False,
                                        "score": score,
                                        "priority": priority_of(score),
                                        "state": _OB_UNTOUCHED,
                                        "max_penetration": 0.0,
                                        "first_touch_bar": -1,
                                        "relative_volume": rel_vol,
                                        "has_fvg": has_bull_fvg,
                                    }

            if bear_bos:
                rejects["bos_attempts"] += 1
                range_ok = last_high_bar >= 0 and bos_trigger_low_bar >= 0 and last_high_bar > bos_trigger_low_bar
                if not range_ok:
                    rejects["reject_range_mismatch"] += 1
                else:
                    ob_top, ob_bottom, ob_offset, found = extract_segment(False, bos_trigger_low_bar)
                    if not found:
                        rejects["reject_segment"] += 1
                    elif displacement < min_disp:
                        rejects["reject_displacement"] += 1
                    else:
                        location_ok = True
                        if np.isfinite(bos_range_high) and np.isfinite(bos_range_low):
                            threshold = bos_range_low + (bos_range_high - bos_range_low) * ob_location_threshold
                            location_ok = ob_top > threshold
                        if not location_ok:
                            rejects["reject_location"] += 1
                        else:
                            ob_range = ob_top - ob_bottom
                            size_ok = (
                                np.isfinite(atr_i)
                                and ob_range <= max_ob_range * atr_i
                                and ob_range >= min_ob_range * atr_i
                            )
                            proceed = True
                            if size_filter_mode == "Hard Limit" and not size_ok:
                                rejects["reject_size"] += 1
                                proceed = False
                            if proceed:
                                rel_vol = (
                                    volume[i - ob_offset] / volume_sma[i - ob_offset]
                                    if volume_real and np.isfinite(volume_sma[i - ob_offset])
                                    and volume_sma[i - ob_offset] > 0
                                    else math.nan
                                )
                                volume_ok = (not use_volume_filter) or (not volume_real) or (
                                    np.isfinite(rel_vol) and rel_vol > volume_floor
                                )
                                if not volume_ok:
                                    rejects["reject_volume"] += 1
                                elif require_fvg and not has_bear_fvg:
                                    rejects["reject_fvg"] += 1
                                else:
                                    score = score_of(False, ob_top, ob_bottom, rel_vol, has_bear_fvg, ob_range, size_ok)
                                    bear_ob = {
                                        "creation_bar": i - ob_offset,
                                        "top": ob_top,
                                        "bottom": ob_bottom,
                                        "bos_bar": i,
                                        "mitigated": False,
                                        "score": score,
                                        "priority": priority_of(score),
                                        "state": _OB_UNTOUCHED,
                                        "max_penetration": 0.0,
                                        "first_touch_bar": -1,
                                        "relative_volume": rel_vol,
                                        "has_fvg": has_bear_fvg,
                                    }

        # ================= OB 生命周期 ======================================
        if enable_order_blocks and bull_ob is not None and i > bull_ob["bos_bar"]:
            ob_top, ob_bottom = bull_ob["top"], bull_ob["bottom"]
            ob_range = ob_top - ob_bottom
            if bull_ob["state"] == _OB_UNTOUCHED and ob_range > 0:
                if ob_bottom <= low[i] <= ob_top:
                    penetration = (ob_top - low[i]) / ob_range * 100.0
                    if bull_ob["first_touch_bar"] < 0:
                        bull_ob["first_touch_bar"] = i
                    bull_ob["max_penetration"] = max(bull_ob["max_penetration"], penetration)
                    if bull_ob["max_penetration"] <= 30:
                        bull_ob["state"] = _OB_TAP
                    elif bull_ob["max_penetration"] <= 70:
                        bull_ob["state"] = _OB_DEEP
                    else:
                        bull_ob["state"] = _OB_FAILURE
            if low[i] < ob_bottom:
                bull_ob = None
            else:
                if not bull_ob["mitigated"]:
                    penetration = ob_top - low[i] if ob_bottom <= low[i] <= ob_top else -1.0
                    if penetration >= ob_range * mitigation_penetration:
                        bull_ob["mitigated"] = True
                        bull_ob["score"] -= 15.0
                        bull_ob["priority"] = priority_of(bull_ob["score"])
                if bull_ob is not None and i - bull_ob["creation_bar"] > ob_ttl:
                    bull_ob = None

        if enable_order_blocks and bear_ob is not None and i > bear_ob["bos_bar"]:
            ob_top, ob_bottom = bear_ob["top"], bear_ob["bottom"]
            ob_range = ob_top - ob_bottom
            if bear_ob["state"] == _OB_UNTOUCHED and ob_range > 0:
                if ob_bottom <= high[i] <= ob_top:
                    penetration = (high[i] - ob_bottom) / ob_range * 100.0
                    if bear_ob["first_touch_bar"] < 0:
                        bear_ob["first_touch_bar"] = i
                    bear_ob["max_penetration"] = max(bear_ob["max_penetration"], penetration)
                    if bear_ob["max_penetration"] <= 30:
                        bear_ob["state"] = _OB_TAP
                    elif bear_ob["max_penetration"] <= 70:
                        bear_ob["state"] = _OB_DEEP
                    else:
                        bear_ob["state"] = _OB_FAILURE
            if high[i] > ob_top:
                bear_ob = None
            else:
                if not bear_ob["mitigated"]:
                    penetration = high[i] - ob_bottom if ob_bottom <= high[i] <= ob_top else -1.0
                    if penetration >= ob_range * mitigation_penetration:
                        bear_ob["mitigated"] = True
                        bear_ob["score"] -= 15.0
                        bear_ob["priority"] = priority_of(bear_ob["score"])
                if bear_ob is not None and i - bear_ob["creation_bar"] > ob_ttl:
                    bear_ob = None

        confirmed_bull = bull_bos and bull_ob is not None and not bull_ob["mitigated"]
        confirmed_bear = bear_bos and bear_ob is not None and not bear_ob["mitigated"]

        # ================= Layer 4：警告 =====================================
        swing_break_bull = False
        swing_break_bear = False
        swing_break_level = math.nan
        early_top = False
        early_bottom = False

        if enable_swing_level_break and show_layer4 and np.isfinite(last_swing_high) and np.isfinite(last_swing_low):
            if last_low_bar >= 0 and i > last_low_bar + 2 and is_bull_structure:
                if _cross_under_level(close, last_swing_low * 0.995, swing_low_before * 0.995, i):
                    if last_swing_break_bar < 0 or i - last_swing_break_bar > warning_cooldown:
                        swing_break_bull = True
                        swing_break_level = last_swing_low
                        last_swing_break_bar = i
            if last_high_bar >= 0 and i > last_high_bar + 2 and is_bear_structure:
                if _cross_over_level(close, last_swing_high * 1.005, swing_high_before * 1.005, i):
                    if last_swing_break_bar < 0 or i - last_swing_break_bar > warning_cooldown:
                        swing_break_bear = True
                        swing_break_level = last_swing_high
                        last_swing_break_bar = i

        candle_span = high[i] - low[i] + 1e-10
        rsi_i = rsi_a[i]
        if enable_early_top and show_layer4 and np.isfinite(rsi_i):
            wick_ratio = (high[i] - max(open_[i], price)) / candle_span
            if wick_ratio >= wick_threshold and rsi_i > rsi_overbought and price < high[i] * 0.98:
                if last_early_top_bar < 0 or i - last_early_top_bar > warning_cooldown:
                    early_top = True
                    last_early_top_bar = i
        if enable_early_bottom and show_layer4 and np.isfinite(rsi_i):
            wick_ratio = (min(open_[i], price) - low[i]) / candle_span
            if wick_ratio >= wick_threshold and rsi_i < rsi_oversold and price > low[i] * 1.02:
                if last_early_bottom_bar < 0 or i - last_early_bottom_bar > warning_cooldown:
                    early_bottom = True
                    last_early_bottom_bar = i

        # ================= 落盘 =============================================
        cols["swing_high"][i] = last_swing_high
        cols["swing_low"][i] = last_swing_low
        cols["expectation_price"][i] = expectation_price
        cols["bos_swing_high"][i] = bos_swing_high
        cols["bos_swing_low"][i] = bos_swing_low
        cols["bos_prior_high"][i] = bos_prior_high
        cols["bos_prior_low"][i] = bos_prior_low
        cols["bos_range_high"][i] = bos_range_high
        cols["bos_range_low"][i] = bos_range_low
        cols["leg_strength"][i] = leg_strength
        cols["bar_displacement"][i] = displacement if displacement > 0 else math.nan
        cols["relative_volume"][i] = (
            volume[i] / volume_sma[i]
            if volume_real and np.isfinite(volume_sma[i]) and volume_sma[i] > 0
            else math.nan
        )
        cols["rsi"][i] = rsi_i
        cols["atr"][i] = atr_i
        cols["swing_break_level"][i] = swing_break_level
        for prefix, ob in (("bull", bull_ob), ("bear", bear_ob)):
            if ob is not None:
                cols[prefix + "_ob_top"][i] = ob["top"]
                cols[prefix + "_ob_bottom"][i] = ob["bottom"]
                cols[prefix + "_ob_score"][i] = ob["score"]
                cols[prefix + "_ob_priority"][i] = ob["priority"]
                cols[prefix + "_ob_penetration"][i] = ob["max_penetration"]

        flags["structure_code"][i] = structure
        flags["internal_structure_code"][i] = int_structure if enable_internal_structure else 0
        flags["last_high_type"][i] = last_high_type
        flags["last_low_type"][i] = last_low_type
        flags["swing_high_bar"][i] = last_high_bar if last_high_bar >= 0 else -1
        flags["swing_low_bar"][i] = last_low_bar if last_low_bar >= 0 else -1
        flags["structure_aligned"][i] = int(aligned)
        flags["structure_conflict"][i] = int(conflict)
        flags["expectation_code"][i] = expectation
        flags["expectation_active"][i] = int(expectation_active)
        flags["expectation_expired"][i] = int(expectation_expired)
        flags["expect_hl"][i] = int(expect_hl)
        flags["expect_lh"][i] = int(expect_lh)
        flags["expect_continuation"][i] = int(expect_continuation)
        flags["expect_failure"][i] = int(expect_failure)
        flags["bullish_bos"][i] = int(bull_bos)
        flags["bearish_bos"][i] = int(bear_bos)
        flags["pending_bull"][i] = int(pending_bull)
        flags["pending_bear"][i] = int(pending_bear)
        flags["has_bull_fvg"][i] = int(has_bull_fvg)
        flags["has_bear_fvg"][i] = int(has_bear_fvg)
        flags["bull_ob_state"][i] = bull_ob["state"] if bull_ob is not None else _OB_NONE
        flags["bear_ob_state"][i] = bear_ob["state"] if bear_ob is not None else _OB_NONE
        flags["bull_ob_mitigated"][i] = int(bool(bull_ob is not None and bull_ob["mitigated"]))
        flags["bear_ob_mitigated"][i] = int(bool(bear_ob is not None and bear_ob["mitigated"]))
        flags["bull_ob_active"][i] = int(bull_ob is not None)
        flags["bear_ob_active"][i] = int(bear_ob is not None)
        flags["confirmed_bullish"][i] = int(confirmed_bull)
        flags["confirmed_bearish"][i] = int(confirmed_bear)
        for key, value in rejects.items():
            flags[key][i] = value
        flags["swing_break_bull"][i] = int(swing_break_bull)
        flags["swing_break_bear"][i] = int(swing_break_bear)
        flags["early_top"][i] = int(early_top)
        flags["early_bottom"][i] = int(early_bottom)

        prev_bos_high = bos_swing_high
        prev_bos_low = bos_swing_low

    out: dict[str, object] = {}
    for name, values in cols.items():
        out["smc_" + name] = values
    for name, values in flags.items():
        out["smc_" + name] = pd.array(values, dtype="Int64")
    return pd.DataFrame(out, index=frame.index)
