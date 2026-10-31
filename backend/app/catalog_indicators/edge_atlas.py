"""Edge Atlas [WavesUnchained] v1.7.2 — right-edge price-level atlas.

Port of ``data/pine-collection/indicators/market_structure/edge_atlas/edge_atlas.pine``.

Pine 原脚本是一个层级登记簿：各计算模块产出归一化的价格水平（``Level``），
再由同一个 registry 统一过滤和渲染。本移植保留 **全部计算模块** 与 **registry
的筛选语义**，只丢弃绘图（line/label/box）、面板和告警 —— 那些在单帧 DataFrame
上下文里没有数值意义。

已移植的计算模块（对应 Pine 的输入分组）：

* 周期水平（日/周/月/年）：PDH/PDL/PDC/PDO/DO/PDM 及其周/月/年镜像
* 会话区间（Asia/London/NY/两个自定义）：可切换“已完成区间”或“正在形成区间”
* 开盘区间（日/周/月）：首个 N 分钟的高低点，只在窗口收盘后登记
* 枢轴点：Traditional / Fibonacci / Camarilla / Woodie / DeMark 五种公式
* 摆动斐波：最新确认的反向摆动对 + 回撤/扩展位
* 枢轴聚类支撑阻力：ATR 容差合并、触达计数、破位归零、取最近 N 档
* 整数关口：自动 1/2/5×10ⁿ 步长，Major/Half/Quarter 三档
* VWAP：会话/周/月三种锚定（含成交量加权标准差带）
* 手动价位：4 个槽位（价格 0 视为未设置）
* 趋势曲线：JMA-style / EMA / SMA 三线，可选喂入 registry

未移植：``request.security`` 的多周期 FVG/PVP、所有绘图、表格、告警。
周期水平用 **按本地时区重采样的已完结周期** 复刻 ``request.security(..., "D", high[1],
lookahead_on)`` 的语义（即“上一根 HTF bar”的值），趋势曲线的 MTF 选项同理退化为图表
周期本身。

输出为逐 bar 的 DataFrame（``ea_`` 前缀）：各模块的水平价 + registry 过滤后的
最近上下方水平/距离/优先级/类别，以及登记数与选中数。

约定：``ea_*`` 模块列始终输出**计算值**，模块开关（``enable_*`` / ``show_*``）只决定
该水平是否进入 registry（即是否可能出现在 ``ea_nearest_*`` / 计数里，与 Pine 的
“是否绘制”对应）。尚未形成的值（如第一根的 PDH）保持 ``na``。

Pine 的 ``Level`` 优先级（1-10）在这里以 ``priority`` 字段保留，registry 的排序键
与 Pine 完全一致：``(10 - priority) * 1e6 + distance_atr``，再按 ATR 最小间距贪心去重。
"""

from __future__ import annotations

import math
from zoneinfo import ZoneInfo

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
    sma,
)

# 类别编码（输出列用，顺序固定）
CAT_DAY = 1
CAT_WEEK = 2
CAT_MONTH = 3
CAT_YEAR = 4
CAT_SESSION = 5
CAT_OR = 6
CAT_PIVOT = 7
CAT_FIB = 8
CAT_SR = 9
CAT_ROUND = 10
CAT_VWAP = 11
CAT_MANUAL = 12
CAT_TREND = 13

_MAX_CLUSTERS = 80

_PRESETS = ("Clean", "Intraday", "Structure", "Everything", "Custom")
_TREND_PRESETS = ("Responsive", "Balanced", "Structural", "Custom")
_MA_TYPES = ("JMA-style", "EMA", "SMA")
_PIVOT_METHODS = ("Traditional", "Fibonacci", "Camarilla", "Woodie", "DeMark")
_PIVOT_PERIODS = ("Daily", "Weekly", "Monthly")


# ---------------------------------------------------------------------------
# 时间 / 会话工具
# ---------------------------------------------------------------------------


def _to_local_index(idx: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    """把 bar 开盘时间换算到 ``tz`` 的本地墙钟（去掉 tzinfo）。

    Pine 的 ``time(period, session, tz)`` 与 ``time("D")`` 都基于交易所/指定时区
    的墙钟时间，所以先转换再按本地时间做周期切分与会话判定。
    """
    if idx.tz is None:
        localized = idx.tz_localize("UTC")
    else:
        localized = idx
    return localized.tz_convert(ZoneInfo(tz)).tz_localize(None)


def _split_window(spec: str) -> list[tuple[int, int]]:
    """解析 ``"HHMM-HHMM"`` / ``"HH:MM-HH:MM"``（可逗号分隔多段）为分钟数区间。"""
    windows: list[tuple[int, int]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            raise ValueError(f"session window must look like 'HHMM-HHMM': {spec!r}")
        left, right = chunk.split("-", 1)

        def minutes(text: str) -> int:
            text = text.strip().replace(":", "")
            if not text.isdigit():
                raise ValueError(f"bad session time {text!r}")
            if len(text) == 4:
                hours, mins = int(text[:2]), int(text[2:])
            elif len(text) <= 2:
                hours, mins = int(text), 0
            else:
                raise ValueError(f"bad session time {text!r}")
            if not (0 <= hours <= 24 and 0 <= mins < 60):
                raise ValueError(f"bad session time {text!r}")
            return min(hours * 60 + mins, 24 * 60)

        windows.append((minutes(left), minutes(right)))
    if not windows:
        raise ValueError(f"empty session window: {spec!r}")
    return windows


def _session_mask(local: pd.DatetimeIndex, spec: str) -> np.ndarray:
    """Pine ``time(period, session, tz)`` 的近似：按 bar **开盘**时间判定是否在窗口内。"""
    minutes = local.hour.to_numpy() * 60 + local.minute.to_numpy()
    mask = np.zeros(len(local), dtype=bool)
    for start, end in _split_window(spec):
        if start == end:  # Pine: 起止相同 = 全天
            mask[:] = True
        elif start < end:
            mask |= (minutes >= start) & (minutes < end)
        else:  # 跨午夜
            mask |= (minutes >= start) | (minutes < end)
    return mask


def _period_key(local: pd.DatetimeIndex, freq: str) -> pd.PeriodIndex:
    """按本地墙钟时间取周期键（D/W/M/Y）。"""
    if freq == "D":
        return pd.PeriodIndex(local, freq="D")
    if freq == "W":
        return pd.PeriodIndex(local, freq="W")
    if freq == "M":
        return pd.PeriodIndex(local, freq="M")
    if freq == "Y":
        return pd.PeriodIndex(local, freq="Y")
    raise ValueError(f"unsupported period freq {freq!r}")


def _period_levels(
    local: pd.DatetimeIndex,
    frame: pd.DataFrame,
    freq: str,
) -> dict[str, np.ndarray]:
    """当前/上一周期的 OHLC。

    对应 Pine ``request.security(tickerid, freq, high[1], lookahead_on)`` ——
    即上一根**已完成** HTF bar 的值（``last_*``）；以及
    ``request.security(..., open, lookahead_on)`` —— 即当前 HTF bar 的开盘价
    （``current_open``，在新的周期 bar 第一根就已知）。
    """
    key = _period_key(local, freq)
    grouped = frame.groupby(key).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    previous = grouped.shift(1)
    bars = pd.Series(np.asarray(key))
    out: dict[str, np.ndarray] = {}
    for column in ("open", "high", "low", "close"):
        out["current_" + column] = bars.map(grouped[column]).to_numpy(dtype=float)
        out["last_" + column] = bars.map(previous[column]).to_numpy(dtype=float)
    return out


def _period_start(local: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """周期起始的墙钟时间（Pine 的 ``time("D"/"W"/"M")``），用于开盘区间判定。"""
    key = _period_key(local, freq)
    starts = key.start_time
    return np.asarray(starts)


def _anchored_vwap(
    src: np.ndarray,
    vol: np.ndarray,
    anchor: np.ndarray,
    mult: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pine ``ta.vwap(src, anchor, mult)`` —— 锚点处重置的成交量加权均价 + 标准差带。

    方差用 ``Σv·s² / Σv − vwap²`` 的闭式（Pine 的 vwap band 即成交量加权标准差）。
    """
    n = len(src)
    value = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    sum_pv = 0.0
    sum_v = 0.0
    sum_pv2 = 0.0
    for i in range(n):
        if anchor[i]:
            sum_pv = 0.0
            sum_v = 0.0
            sum_pv2 = 0.0
        price = src[i]
        weight = vol[i]
        if not np.isfinite(price) or not np.isfinite(weight):
            continue
        sum_pv += price * weight
        sum_v += weight
        sum_pv2 += price * price * weight
        if sum_v <= 0:
            continue
        vwap = sum_pv / sum_v
        variance = max(sum_pv2 / sum_v - vwap * vwap, 0.0)
        stdev = math.sqrt(variance)
        value[i] = vwap
        upper[i] = vwap + mult * stdev
        lower[i] = vwap - mult * stdev
    return value, upper, lower


def _jma(src: np.ndarray, length: int, phase: float, power: float) -> np.ndarray:
    """Pine 的 ``f_jma`` —— Jurik 启发的开源近似（非 Mark Jurik 的专利原算法）。"""
    phase_ratio = 0.5 if phase < -100 else 2.5 if phase > 100 else phase / 100.0 + 1.5
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2.0)
    alpha = beta ** power
    n = len(src)
    out = np.full(n, np.nan)
    e0 = e1 = e2 = jma_value = math.nan
    for i in range(n):
        price = src[i]
        seed = price if not np.isfinite(jma_value) else jma_value
        prev_e0 = price if not np.isfinite(e0) else e0
        prev_e2 = 0.0 if not np.isfinite(e2) else e2
        e0 = (1.0 - alpha) * price + alpha * prev_e0
        e1 = (price - e0) * (1.0 - beta) + beta * (0.0 if not np.isfinite(e1) else e1)
        e2 = (e0 + phase_ratio * e1 - seed) * (1.0 - alpha) ** 2 + alpha ** 2 * prev_e2
        jma_value = e2 + seed
        out[i] = jma_value
    return out


def _ma_value(kind: str, src: np.ndarray, length: int, phase: float, power: float) -> np.ndarray:
    """Pine ``f_maValue``：三种平滑的派发。"""
    if kind == "JMA-style":
        return _jma(src, length, phase, power)
    series = pd.Series(src)
    if kind == "EMA":
        return series.ewm(span=length, adjust=False, min_periods=length).mean().to_numpy(dtype=float)
    return sma(series, length).to_numpy(dtype=float)


def _auto_round_step(price: float, mintick: float) -> float:
    """Pine ``f_autoRoundStep``：按价格量级取 1/2/5×10ⁿ 的“视觉友好”步长。"""
    safe_price = max(abs(price), mintick)
    exponent = math.floor(math.log10(safe_price))
    magnitude = 10.0 ** exponent
    normalized = safe_price / magnitude
    if normalized < 2.0:
        factor = 0.02
    elif normalized < 5.0:
        factor = 0.05
    else:
        factor = 0.10
    return max(magnitude * factor, mintick)


def _update_cluster(
    centers: list[float],
    touches: list[int],
    last_bars: list[int],
    price: float,
    bar: int,
    tolerance: float,
) -> None:
    """Pine ``f_updateCluster``：在容差内合并到最近的已有簇（加权中心），否则新建。"""
    best_index = -1
    best_distance = 1e100
    for i, center in enumerate(centers):
        distance = abs(center - price)
        if distance <= tolerance and distance < best_distance:
            best_distance = distance
            best_index = i
    if best_index >= 0:
        old_touches = touches[best_index]
        old_center = centers[best_index]
        centers[best_index] = (old_center * old_touches + price) / (old_touches + 1)
        touches[best_index] = old_touches + 1
        last_bars[best_index] = bar
    else:
        centers.append(price)
        touches.append(1)
        last_bars.append(bar)


def _cluster_scan(
    close: np.ndarray,
    safe_atr: np.ndarray,
    piv_high: np.ndarray,
    piv_low: np.ndarray,
    right: int,
    tolerance_atr: float,
    min_touches: int,
    max_age: int,
    n_sup: int,
    n_res: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """枢轴聚类 S/R 的逐 bar 状态机（合并 → 破位归零 → 取最近 N 档）。"""
    n = len(close)
    width = max(n_sup, 1)
    sup_price = np.full((n, width), np.nan)
    sup_touch = np.zeros((n, width), dtype=np.int64)
    res_price = np.full((n, max(n_res, 1)), np.nan)
    res_touch = np.zeros((n, max(n_res, 1)), dtype=np.int64)

    sup_centers: list[float] = []
    sup_touches: list[int] = []
    sup_bars: list[int] = []
    res_centers: list[float] = []
    res_touches: list[int] = []
    res_bars: list[int] = []

    for i in range(n):
        tolerance = tolerance_atr * safe_atr[i]
        if not np.isnan(piv_high[i]):
            _update_cluster(res_centers, res_touches, res_bars, float(piv_high[i]), i - right, tolerance)
        if not np.isnan(piv_low[i]):
            _update_cluster(sup_centers, sup_touches, sup_bars, float(piv_low[i]), i - right, tolerance)
        while len(sup_centers) > _MAX_CLUSTERS:
            sup_centers.pop(0)
            sup_touches.pop(0)
            sup_bars.pop(0)
        while len(res_centers) > _MAX_CLUSTERS:
            res_centers.pop(0)
            res_touches.pop(0)
            res_bars.pop(0)

        # 收盘价越过簇中心超过容差 → 视为破位，触达计数归零
        for k in range(len(res_centers)):
            if close[i] > res_centers[k] + tolerance:
                res_touches[k] = 0
        for k in range(len(sup_centers)):
            if close[i] < sup_centers[k] - tolerance:
                sup_touches[k] = 0

        sup_candidates = [
            (close[i] - sup_centers[k], k)
            for k in range(len(sup_centers))
            if sup_centers[k] < close[i]
            and sup_touches[k] >= min_touches
            and (i - sup_bars[k]) <= max_age
        ]
        sup_candidates.sort()
        for rank, (_, k) in enumerate(sup_candidates[:n_sup]):
            sup_price[i, rank] = sup_centers[k]
            sup_touch[i, rank] = sup_touches[k]

        res_candidates = [
            (res_centers[k] - close[i], k)
            for k in range(len(res_centers))
            if res_centers[k] > close[i]
            and res_touches[k] >= min_touches
            and (i - res_bars[k]) <= max_age
        ]
        res_candidates.sort()
        for rank, (_, k) in enumerate(res_candidates[:n_res]):
            res_price[i, rank] = res_centers[k]
            res_touch[i, rank] = res_touches[k]

    return sup_price, sup_touch, res_price, res_touch


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def edge_atlas(
    frame: pd.DataFrame,
    *,
    preset: str = "Clean",
    master_enable: bool = True,
    max_levels: int = 24,
    max_distance_atr: float = 12.0,
    show_above: bool = True,
    show_below: bool = True,
    min_priority: int = 1,
    min_label_spacing_atr: float = 3.0,
    enable_day_levels: bool = True,
    show_pdh: bool = True,
    show_pdl: bool = True,
    show_pdc: bool = True,
    show_pdo: bool = False,
    show_do: bool = True,
    show_pdm: bool = False,
    enable_week_levels: bool = True,
    show_pwh: bool = True,
    show_pwl: bool = True,
    show_pwc: bool = False,
    show_pwo: bool = False,
    show_wo: bool = True,
    show_pwm: bool = False,
    enable_month_levels: bool = True,
    show_pmh: bool = False,
    show_pml: bool = False,
    show_pmc: bool = False,
    show_pmo: bool = False,
    show_mo: bool = True,
    show_pmm: bool = False,
    enable_year_levels: bool = False,
    show_pyh: bool = False,
    show_pyl: bool = False,
    show_pyc: bool = False,
    show_pyo: bool = False,
    show_yo: bool = False,
    show_pym: bool = False,
    enable_sessions: bool = False,
    session_timezone: str = "Etc/UTC",
    asia_enable: bool = False,
    asia_session: str = "0000-0800",
    asia_high: bool = True,
    asia_low: bool = True,
    asia_mid: bool = False,
    london_enable: bool = False,
    london_session: str = "0700-1600",
    london_high: bool = True,
    london_low: bool = True,
    london_mid: bool = False,
    ny_enable: bool = False,
    ny_session: str = "1330-2100",
    ny_high: bool = True,
    ny_low: bool = True,
    ny_mid: bool = False,
    custom1_enable: bool = False,
    custom1_session: str = "0800-1200",
    custom1_high: bool = True,
    custom1_low: bool = True,
    custom1_mid: bool = False,
    custom2_enable: bool = False,
    custom2_session: str = "1200-1600",
    custom2_high: bool = True,
    custom2_low: bool = True,
    custom2_mid: bool = False,
    use_completed_session: bool = True,
    enable_pivots: bool = False,
    pivot_method: str = "Traditional",
    pivot_period: str = "Daily",
    show_pp: bool = True,
    show_r1: bool = True,
    show_s1: bool = True,
    show_r2: bool = True,
    show_s2: bool = True,
    show_r3: bool = False,
    show_s3: bool = False,
    enable_fib: bool = False,
    fib_pivot_left: int = 5,
    fib_pivot_right: int = 5,
    fib_min_leg_atr: float = 2.0,
    fib_236: bool = False,
    fib_382: bool = True,
    fib_500: bool = True,
    fib_618: bool = True,
    fib_786: bool = False,
    fib_100: bool = False,
    fib_1272: bool = False,
    fib_1618: bool = False,
    enable_sr: bool = True,
    sr_pivot_left: int = 5,
    sr_pivot_right: int = 5,
    sr_cluster_atr: float = 0.35,
    sr_zone_width_atr: float = 0.25,
    sr_min_touches: int = 2,
    sr_max_age: int = 1500,
    sr_supports: int = 2,
    sr_resistances: int = 2,
    enable_round: bool = False,
    round_auto_step: bool = True,
    round_manual_step: float = 100.0,
    round_major: bool = True,
    round_half: bool = False,
    round_quarter: bool = False,
    round_levels_each_side: int = 3,
    enable_vwap: bool = False,
    vwap_session: bool = True,
    vwap_weekly: bool = False,
    vwap_monthly: bool = False,
    enable_manual: bool = False,
    manual1_enable: bool = False,
    manual1_price: float = 0.0,
    manual2_enable: bool = False,
    manual2_price: float = 0.0,
    manual3_enable: bool = False,
    manual3_price: float = 0.0,
    manual4_enable: bool = False,
    manual4_price: float = 0.0,
    enable_opening_ranges: bool = False,
    daily_or_enable: bool = True,
    daily_or_minutes: int = 30,
    daily_or_high: bool = True,
    daily_or_low: bool = True,
    weekly_or_enable: bool = False,
    weekly_or_minutes: int = 1440,
    weekly_or_high: bool = True,
    weekly_or_low: bool = True,
    monthly_or_enable: bool = False,
    monthly_or_minutes: int = 1440,
    monthly_or_high: bool = True,
    monthly_or_low: bool = True,
    show_trend_curves: bool = False,
    trend_preset: str = "Balanced",
    trend_as_levels: bool = False,
    trend_fast_type: str = "JMA-style",
    trend_fast_length: int = 21,
    trend_fast_phase: float = 0.0,
    trend_fast_power: float = 2.0,
    trend_base_type: str = "EMA",
    trend_base_length: int = 50,
    trend_base_phase: float = 0.0,
    trend_base_power: float = 2.0,
    trend_anchor_type: str = "SMA",
    trend_anchor_length: int = 200,
    trend_anchor_phase: float = 0.0,
    trend_anchor_power: float = 2.0,
    atr_length: int = 14,
    mintick: float | None = None,
    volume_is_real: bool | None = None,
) -> pd.DataFrame:
    """Edge Atlas —— 右缘价格水平图谱。

    对应 Pine ``Edge Atlas [WavesUnchained] v1.7.2``。各模块产出归一化价格水平，
    再由统一 registry 按「距离 ≤ N·ATR、方向开关、最低优先级、ATR 最小间距、上限」
    筛选，输出逐 bar 的最近上下方水平与各模块水平价。

    Args:
        frame: 含 ``open`` / ``high`` / ``low`` / ``close``（``volume`` 可选）的 OHLCV，
            索引必须可转 ``DatetimeIndex``。
        preset: ``Clean`` / ``Intraday`` / ``Structure`` / ``Everything`` / ``Custom``。
            非 Custom 时各模块开关由预设决定（Custom 才读下面的 ``enable_*``）。
        master_enable: 总开关，关闭时 registry 不登记任何水平。
        max_levels: 渲染上限（registry 贪心接受后的硬上限）。
        max_distance_atr: 距收盘价的 ATR 距离上限，超出的水平不参与筛选。
        show_above: 是否允许登记价格 ≥ 收盘价的水平。
        show_below: 是否允许登记价格 < 收盘价的水平。
        min_priority: 最低优先级（1 保留全部；越高越先剔除不重要的）。
        min_label_spacing_atr: 已选中水平之间的最小 ATR 间距，0 表示关闭去重。
        enable_day_levels: Custom 预设下的日线周期水平总开关。
        show_pdh: 登记 PDH（前日高）。
        show_pdl: 登记 PDL（前日低）。
        show_pdc: 登记 PDC（前日收）。
        show_pdo: 登记 PDO（前日开）。
        show_do: 登记 DO（当日开）。
        show_pdm: 登记 PDM（前日中间价）。
        enable_week_levels: Custom 预设下的周线周期水平总开关。
        show_pwh: 登记 PWH（前周高）。
        show_pwl: 登记 PWL（前周低）。
        show_pwc: 登记 PWC（前周收）。
        show_pwo: 登记 PWO（前周开）。
        show_wo: 登记 WO（本周开）。
        show_pwm: 登记 PWM（前周中间价）。
        enable_month_levels: Custom 预设下的月线周期水平总开关。
        show_pmh: 登记 PMH（前月高）。
        show_pml: 登记 PML（前月低）。
        show_pmc: 登记 PMC（前月收）。
        show_pmo: 登记 PMO（前月开）。
        show_mo: 登记 MO（本月开）。
        show_pmm: 登记 PMM（前月中间价）。
        enable_year_levels: Custom 预设下的年线周期水平总开关。
        show_pyh: 登记 PYH（去年高）。
        show_pyl: 登记 PYL（去年低）。
        show_pyc: 登记 PYC（去年收）。
        show_pyo: 登记 PYO（去年开）。
        show_yo: 登记 YO（今年开）。
        show_pym: 登记 PYM（去年中间价）。
        enable_sessions: Custom 预设下的会话区间总开关。
        session_timezone: IANA 时区名，会话与周期切分都用它。
        asia_enable: 是否启用 Asia 会话区间。
        asia_session: Asia 会话窗口，``"HHMM-HHMM"``。
        asia_high: 登记 ASH。
        asia_low: 登记 ASL。
        asia_mid: 登记 ASM。
        london_enable: 是否启用 London 会话区间。
        london_session: London 会话窗口。
        london_high: 登记 LDH。
        london_low: 登记 LDL。
        london_mid: 登记 LDM。
        ny_enable: 是否启用 New York 会话区间。
        ny_session: New York 会话窗口。
        ny_high: 登记 NYH。
        ny_low: 登记 NYL。
        ny_mid: 登记 NYM。
        custom1_enable: 是否启用自定义会话 1。
        custom1_session: 自定义会话 1 窗口。
        custom1_high: 登记自定义会话 1 高点。
        custom1_low: 登记自定义会话 1 低点。
        custom1_mid: 登记自定义会话 1 中间价。
        custom2_enable: 是否启用自定义会话 2。
        custom2_session: 自定义会话 2 窗口。
        custom2_high: 登记自定义会话 2 高点。
        custom2_low: 登记自定义会话 2 低点。
        custom2_mid: 登记自定义会话 2 中间价。
        use_completed_session: True 用上一段已完成区间（稳定），False 用正在形成的区间。
        enable_pivots: Custom 预设下的枢轴点总开关。
        pivot_method: ``Traditional`` / ``Fibonacci`` / ``Camarilla`` / ``Woodie`` / ``DeMark``。
        pivot_period: 枢轴参考周期 ``Daily`` / ``Weekly`` / ``Monthly``。
        show_pp: 登记 PP（中枢）。
        show_r1: 登记 R1。
        show_s1: 登记 S1。
        show_r2: 登记 R2。
        show_s2: 登记 S2。
        show_r3: 登记 R3。
        show_s3: 登记 S3。
        enable_fib: Custom 预设下的摆动斐波总开关。
        fib_pivot_left: 摆动枢轴左侧 bar 数。
        fib_pivot_right: 摆动枢轴右侧 bar 数。
        fib_min_leg_atr: 有效摆动腿的最小长度（ATR 倍数）。
        enable_sr: Custom 预设下的枢轴聚类 S/R 总开关。
        sr_pivot_left: S/R 枢轴左侧 bar 数。
        sr_pivot_right: S/R 枢轴右侧 bar 数。
        sr_cluster_atr: 聚类容差（ATR 倍数）。
        sr_zone_width_atr: 区域宽度（ATR 倍数）。
        sr_min_touches: 最少触达次数。
        sr_max_age: 簇的最大年龄（bar 数）。
        sr_supports: 最近支撑档数。
        sr_resistances: 最近阻力档数。
        enable_round: Custom 预设下的整数关口总开关。
        round_auto_step: 自动步长（否则用 ``round_manual_step``）。
        round_manual_step: 手动主步长。
        round_major: 登记主步长关口。
        round_half: 登记半步关口。
        round_quarter: 登记四分之一关口。
        round_levels_each_side: 每侧关口数。
        enable_vwap: Custom 预设下的 VWAP 总开关。
        vwap_session: 登记会话 VWAP。
        vwap_weekly: 登记周 VWAP。
        vwap_monthly: 登记月 VWAP。
        enable_manual: Custom 预设下的手动价位总开关。
        manual1_enable: 槽位 1 开关。
        manual1_price: 槽位 1 价格（0 视为未设置）。
        manual2_enable: 槽位 2 开关。
        manual2_price: 槽位 2 价格。
        manual3_enable: 槽位 3 开关。
        manual3_price: 槽位 3 价格。
        manual4_enable: 槽位 4 开关。
        manual4_price: 槽位 4 价格。
        enable_opening_ranges: Custom 预设下的开盘区间总开关。
        daily_or_enable: 启用日开盘区间。
        daily_or_minutes: 日开盘区间窗口（分钟）。
        daily_or_high: 登记 ORH。
        daily_or_low: 登记 ORL。
        weekly_or_enable: 启用周开盘区间。
        weekly_or_minutes: 周开盘区间窗口（分钟）。
        weekly_or_high: 登记 WORH。
        weekly_or_low: 登记 WORL。
        monthly_or_enable: 启用月开盘区间。
        monthly_or_minutes: 月开盘区间窗口（分钟）。
        monthly_or_high: 登记 MORH。
        monthly_or_low: 登记 MORL。
        show_trend_curves: 是否计算趋势曲线。
        trend_preset: ``Responsive`` / ``Balanced`` / ``Structural`` / ``Custom``。
        trend_as_levels: 是否把三线的当前值也喂入 registry。
        trend_fast_type: 快线类型（Custom 预设下生效）。
        trend_fast_length: 快线长度。
        trend_fast_phase: 快线相位（JMA-style）。
        trend_fast_power: 快线幂次（JMA-style）。
        trend_base_type: 基线类型。
        trend_base_length: 基线长度。
        trend_base_phase: 基线相位。
        trend_base_power: 基线幂次。
        trend_anchor_type: 锚线类型。
        trend_anchor_length: 锚线长度。
        trend_anchor_phase: 锚线相位。
        trend_anchor_power: 锚线幂次。
        atr_length: ATR 长度（Pine 固定 14，这里可覆盖以便测试）。
        mintick: 最小跳动，``None`` 时按价格量级推断（Pine 用 ``syminfo.mintick``）。
        volume_is_real: ``False`` 表示成交量不可信，此时 VWAP 退化为等权累积均值。

    Returns:
        ``pd.DataFrame``，索引与 ``frame`` 一致，列为 ``ea_*`` 的浮点/Int64。

    Raises:
        ValueError: 任一参数越界或取值不在允许集合内。
    """
    idx = frame.index
    n = len(frame)
    high_s, low_s, close_s = ohlcv(frame, "high", "low", "close")

    # --- 参数校验（Pine input.* 的 minval/maxval/options） ------------------
    preset = choice("preset", preset, _PRESETS)
    master_enable = boolean("master_enable", master_enable)
    max_levels = integer("max_levels", max_levels, 1, 120)
    max_distance_atr = number("max_distance_atr", max_distance_atr, 0.25, 500.0)
    show_above = boolean("show_above", show_above)
    show_below = boolean("show_below", show_below)
    min_priority = integer("min_priority", min_priority, 1, 10)
    min_label_spacing_atr = number("min_label_spacing_atr", min_label_spacing_atr, 0.0, 20.0)
    fib_pivot_left = integer("fib_pivot_left", fib_pivot_left, 1, 50)
    fib_pivot_right = integer("fib_pivot_right", fib_pivot_right, 1, 50)
    fib_min_leg_atr = number("fib_min_leg_atr", fib_min_leg_atr, 0.25, 100.0)
    sr_pivot_left = integer("sr_pivot_left", sr_pivot_left, 1, 50)
    sr_pivot_right = integer("sr_pivot_right", sr_pivot_right, 1, 50)
    sr_cluster_atr = number("sr_cluster_atr", sr_cluster_atr, 0.05, 5.0)
    sr_zone_width_atr = number("sr_zone_width_atr", sr_zone_width_atr, 0.05, 3.0)
    sr_min_touches = integer("sr_min_touches", sr_min_touches, 1, 20)
    sr_max_age = integer("sr_max_age", sr_max_age, 20, 20000)
    sr_supports = integer("sr_supports", sr_supports, 0, 10)
    sr_resistances = integer("sr_resistances", sr_resistances, 0, 10)
    round_levels_each_side = integer("round_levels_each_side", round_levels_each_side, 1, 20)
    round_manual_step = number("round_manual_step", round_manual_step, 1e-7, 1e15)
    pivot_method = choice("pivot_method", pivot_method, _PIVOT_METHODS)
    pivot_period = choice("pivot_period", pivot_period, _PIVOT_PERIODS)
    trend_preset = choice("trend_preset", trend_preset, _TREND_PRESETS)
    trend_fast_type = choice("trend_fast_type", trend_fast_type, _MA_TYPES)
    trend_base_type = choice("trend_base_type", trend_base_type, _MA_TYPES)
    trend_anchor_type = choice("trend_anchor_type", trend_anchor_type, _MA_TYPES)
    trend_fast_length = integer("trend_fast_length", trend_fast_length, 1, 1000)
    trend_base_length = integer("trend_base_length", trend_base_length, 1, 1000)
    trend_anchor_length = integer("trend_anchor_length", trend_anchor_length, 1, 2000)
    atr_length = integer("atr_length", atr_length, 1, 500)
    daily_or_minutes = integer("daily_or_minutes", daily_or_minutes, 1, 1000)
    weekly_or_minutes = integer("weekly_or_minutes", weekly_or_minutes, 1, 10000)
    monthly_or_minutes = integer("monthly_or_minutes", monthly_or_minutes, 1, 44640)

    for name, value in (
        ("preset", preset),
        ("pivot_method", pivot_method),
        ("pivot_period", pivot_period),
        ("trend_preset", trend_preset),
    ):
        if not isinstance(value, str):  # pragma: no cover - choice 已保证
            raise ValueError(f"{name} must be a string")

    # --- 预设解析（Pine “Preset resolution”） --------------------------------
    is_custom = preset == "Custom"
    is_intraday = preset == "Intraday"
    is_structure = preset == "Structure"
    is_everything = preset == "Everything"

    use_day = boolean("enable_day_levels", enable_day_levels) if is_custom else True
    use_week = boolean("enable_week_levels", enable_week_levels) if is_custom else True
    use_month = boolean("enable_month_levels", enable_month_levels) if is_custom else is_everything
    use_year = boolean("enable_year_levels", enable_year_levels) if is_custom else is_everything
    use_sessions = (
        boolean("enable_sessions", enable_sessions) if is_custom else (is_intraday or is_everything)
    )
    use_pivots = (
        boolean("enable_pivots", enable_pivots) if is_custom else (is_intraday or is_everything)
    )
    use_fib = boolean("enable_fib", enable_fib) if is_custom else (is_structure or is_everything)
    use_sr = boolean("enable_sr", enable_sr) if is_custom else True
    use_round = boolean("enable_round", enable_round) if is_custom else is_everything
    use_vwap = boolean("enable_vwap", enable_vwap) if is_custom else (is_intraday or is_everything)
    use_manual = is_custom and boolean("enable_manual", enable_manual)
    use_or = is_custom and boolean("enable_opening_ranges", enable_opening_ranges)

    # --- 基础序列 -----------------------------------------------------------
    close = close_s.to_numpy(dtype=float)
    high = high_s.to_numpy(dtype=float)
    low = low_s.to_numpy(dtype=float)
    hlc3 = (high + low + close) / 3.0
    atr_a = atr(frame, atr_length).to_numpy(dtype=float)
    if mintick is None:
        magnitude = np.nanmedian(np.abs(close)) if n else 1.0
        magnitude = magnitude if np.isfinite(magnitude) and magnitude > 0 else 1.0
        tick = 10.0 ** (math.floor(math.log10(magnitude)) - 5)
    else:
        tick = number("mintick", mintick, 1e-12, 1e15)
    safe_atr = np.fmax(np.nan_to_num(atr_a, nan=tick), tick)

    if "volume" in frame.columns and volume_is_real is not False:
        vol = frame["volume"].to_numpy(dtype=float)
        vol = np.nan_to_num(vol, nan=0.0)
    else:
        vol = np.ones(n, dtype=float)

    local = _to_local_index(pd.DatetimeIndex(idx), str(session_timezone))

    out: dict[str, np.ndarray] = {}
    static: list[tuple[np.ndarray, int, int, np.ndarray]] = []  # (price, priority, category, mask)

    def register(
        name: str,
        price: np.ndarray | float,
        priority: int,
        category: int,
        enabled: bool | np.ndarray,
    ) -> None:
        """登记一个模块水平：写入输出列并加入 registry 候选池。

        ``enabled`` 可以是标量，也可以是逐 bar 的布尔掩码（斐波水平就是后者：
        有效腿的条件本身是 series）。
        """
        values = (
            np.full(n, float(price), dtype=float)
            if np.isscalar(price)
            else np.asarray(price, dtype=float)
        )
        if isinstance(enabled, np.ndarray):
            mask = enabled.astype(bool)
        else:
            mask = np.full(n, bool(enabled), dtype=bool)
        out["ea_" + name] = values
        if mask.any():
            static.append((values, priority, category, mask))

    # --- 周期水平 -----------------------------------------------------------
    daily = _period_levels(local, frame, "D")
    weekly = _period_levels(local, frame, "W")
    monthly = _period_levels(local, frame, "M")
    yearly = _period_levels(local, frame, "Y")

    pdh, pdl = daily["last_high"], daily["last_low"]
    pdc, pdo, do_ = daily["last_close"], daily["last_open"], daily["current_open"]
    pwh, pwl = weekly["last_high"], weekly["last_low"]
    pwc, pwo, wo_ = weekly["last_close"], weekly["last_open"], weekly["current_open"]
    pmh, pml = monthly["last_high"], monthly["last_low"]
    pmc, pmo, mo_ = monthly["last_close"], monthly["last_open"], monthly["current_open"]
    pyh, pyl = yearly["last_high"], yearly["last_low"]
    pyc, pyo, yo_ = yearly["last_close"], yearly["last_open"], yearly["current_open"]

    # 非 Custom 预设忽略细分的 show_* 开关（Pine 的 `presetCustom ? showX : true`）
    def pick(flag: bool) -> bool:
        return flag if is_custom else True

    def pick_opt(flag: bool) -> bool:  # PDO/PDM 之类只在 Custom/Everything 打开
        return bool(flag) and (is_custom or is_everything)

    register("pdh", pdh, 9, CAT_DAY, use_day and pick(boolean("show_pdh", show_pdh)))
    register("pdl", pdl, 9, CAT_DAY, use_day and pick(boolean("show_pdl", show_pdl)))
    register("pdc", pdc, 7, CAT_DAY, use_day and pick(boolean("show_pdc", show_pdc)))
    register("pdo", pdo, 5, CAT_DAY, use_day and pick_opt(boolean("show_pdo", show_pdo)))
    register("do", do_, 8, CAT_DAY, use_day and pick(boolean("show_do", show_do)))
    register("pdm", (pdh + pdl) / 2.0, 4, CAT_DAY, use_day and pick_opt(boolean("show_pdm", show_pdm)))

    register("pwh", pwh, 9, CAT_WEEK, use_week and pick(boolean("show_pwh", show_pwh)))
    register("pwl", pwl, 9, CAT_WEEK, use_week and pick(boolean("show_pwl", show_pwl)))
    register("pwc", pwc, 6, CAT_WEEK, use_week and pick_opt(boolean("show_pwc", show_pwc)))
    register("pwo", pwo, 5, CAT_WEEK, use_week and pick_opt(boolean("show_pwo", show_pwo)))
    register("wo", wo_, 8, CAT_WEEK, use_week and pick(boolean("show_wo", show_wo)))
    register("pwm", (pwh + pwl) / 2.0, 4, CAT_WEEK, use_week and pick_opt(boolean("show_pwm", show_pwm)))

    register("pmh", pmh, 8, CAT_MONTH, use_month and pick(boolean("show_pmh", show_pmh)))
    register("pml", pml, 8, CAT_MONTH, use_month and pick(boolean("show_pml", show_pml)))
    register("pmc", pmc, 5, CAT_MONTH, use_month and pick_opt(boolean("show_pmc", show_pmc)))
    register("pmo", pmo, 5, CAT_MONTH, use_month and pick_opt(boolean("show_pmo", show_pmo)))
    register("mo", mo_, 7, CAT_MONTH, use_month and pick(boolean("show_mo", show_mo)))
    register("pmm", (pmh + pml) / 2.0, 4, CAT_MONTH, use_month and pick_opt(boolean("show_pmm", show_pmm)))

    register("pyh", pyh, 8, CAT_YEAR, use_year and pick(boolean("show_pyh", show_pyh)))
    register("pyl", pyl, 8, CAT_YEAR, use_year and pick(boolean("show_pyl", show_pyl)))
    register("pyc", pyc, 5, CAT_YEAR, use_year and pick(boolean("show_pyc", show_pyc)))
    register("pyo", pyo, 5, CAT_YEAR, use_year and pick_opt(boolean("show_pyo", show_pyo)))
    register("yo", yo_, 6, CAT_YEAR, use_year and pick(boolean("show_yo", show_yo)))
    register("pym", (pyh + pyl) / 2.0, 4, CAT_YEAR, use_year and pick_opt(boolean("show_pym", show_pym)))

    # --- 会话区间 -----------------------------------------------------------
    def session_range(
        enable: bool,
        spec: str,
        flag_high: bool,
        flag_low: bool,
        flag_mid: bool,
        name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """会话高/低的 developing + done 状态机（Pine 的 devXxx/doneXxx 变量）。"""
        in_session = _session_mask(local, spec)
        dev_high = np.full(n, np.nan)
        dev_low = np.full(n, np.nan)
        done_high = np.full(n, np.nan)
        done_low = np.full(n, np.nan)
        cur_high = math.nan
        cur_low = math.nan
        last_high = math.nan
        last_low = math.nan
        for i in range(n):
            started = in_session[i] and (i == 0 or not in_session[i - 1])
            ended = (not in_session[i]) and i > 0 and in_session[i - 1]
            if started:
                cur_high = high[i]
                cur_low = low[i]
            elif in_session[i]:
                cur_high = high[i] if not np.isfinite(cur_high) else max(cur_high, high[i])
                cur_low = low[i] if not np.isfinite(cur_low) else min(cur_low, low[i])
            if ended:
                last_high = cur_high
                last_low = cur_low
            dev_high[i] = cur_high
            dev_low[i] = cur_low
            done_high[i] = last_high
            done_low[i] = last_low
        if use_completed_session:
            exposed_high, exposed_low = done_high, done_low
        else:
            exposed_high = np.where(in_session, dev_high, done_high)
            exposed_low = np.where(in_session, dev_low, done_low)
        register(name + "_high", exposed_high, 7, CAT_SESSION, enable and flag_high)
        register(name + "_low", exposed_low, 7, CAT_SESSION, enable and flag_low)
        register(name + "_mid", (exposed_high + exposed_low) / 2.0, 4, CAT_SESSION,
                 enable and is_custom and flag_mid)
        return exposed_high, exposed_low

    asia_active = use_sessions and (
        boolean("asia_enable", asia_enable) if is_custom else is_everything
    )
    london_active = use_sessions and (
        boolean("london_enable", london_enable) if is_custom else (is_intraday or is_everything)
    )
    ny_active = use_sessions and (
        boolean("ny_enable", ny_enable) if is_custom else (is_intraday or is_everything)
    )
    # 自定义会话没有合理的默认名称/时段，只在 Custom 预设下激活
    custom1_active = use_sessions and is_custom and boolean("custom1_enable", custom1_enable)
    custom2_active = use_sessions and is_custom and boolean("custom2_enable", custom2_enable)

    session_range(asia_active, asia_session, boolean("asia_high", asia_high),
                  boolean("asia_low", asia_low), boolean("asia_mid", asia_mid), "asia")
    session_range(london_active, london_session, boolean("london_high", london_high),
                  boolean("london_low", london_low), boolean("london_mid", london_mid), "london")
    session_range(ny_active, ny_session, boolean("ny_high", ny_high),
                  boolean("ny_low", ny_low), boolean("ny_mid", ny_mid), "ny")
    session_range(custom1_active, custom1_session, boolean("custom1_high", custom1_high),
                  boolean("custom1_low", custom1_low), boolean("custom1_mid", custom1_mid), "c1")
    session_range(custom2_active, custom2_session, boolean("custom2_high", custom2_high),
                  boolean("custom2_low", custom2_low), boolean("custom2_mid", custom2_mid), "c2")

    # --- 开盘区间（只在窗口收盘后登记） --------------------------------------
    def opening_range(
        enable: bool,
        minutes: int,
        flag_high: bool,
        flag_low: bool,
        freq: str,
        name: str,
    ) -> None:
        """首个 N 分钟内的高低点；窗口未走完不登记（保持 na）。"""
        starts = _period_start(local, freq)
        in_window = (local.to_numpy() - starts) < np.timedelta64(minutes, "m")
        new_period = np.zeros(n, dtype=bool)
        if n:
            new_period[0] = True
            new_period[1:] = starts[1:] != starts[:-1]
        done_high = np.full(n, np.nan)
        done_low = np.full(n, np.nan)
        cur_high = math.nan
        cur_low = math.nan
        last_high = math.nan
        last_low = math.nan
        for i in range(n):
            if new_period[i]:
                cur_high = high[i]
                cur_low = low[i]
            elif in_window[i]:
                cur_high = high[i] if not np.isfinite(cur_high) else max(cur_high, high[i])
                cur_low = low[i] if not np.isfinite(cur_low) else min(cur_low, low[i])
            if i > 0 and in_window[i - 1] and not in_window[i]:
                last_high = cur_high
                last_low = cur_low
            done_high[i] = last_high
            done_low[i] = last_low
        register(name + "_high", done_high, 7 if freq == "D" else (6 if freq == "W" else 5),
                 CAT_OR, enable and flag_high)
        register(name + "_low", done_low, 7 if freq == "D" else (6 if freq == "W" else 5),
                 CAT_OR, enable and flag_low)

    opening_range(use_or and boolean("daily_or_enable", daily_or_enable), daily_or_minutes,
                  boolean("daily_or_high", daily_or_high), boolean("daily_or_low", daily_or_low),
                  "D", "dor")
    opening_range(use_or and boolean("weekly_or_enable", weekly_or_enable), weekly_or_minutes,
                  boolean("weekly_or_high", weekly_or_high), boolean("weekly_or_low", weekly_or_low),
                  "W", "wor")
    opening_range(use_or and boolean("monthly_or_enable", monthly_or_enable), monthly_or_minutes,
                  boolean("monthly_or_high", monthly_or_high), boolean("monthly_or_low", monthly_or_low),
                  "M", "mor")

    # --- 枢轴点 -------------------------------------------------------------
    pivot_freq = "W" if pivot_period == "Weekly" else "M" if pivot_period == "Monthly" else "D"
    piv = _period_levels(local, frame, pivot_freq)
    p_high, p_low = piv["last_high"], piv["last_low"]
    p_close, p_open = piv["last_close"], piv["last_open"]
    p_range = p_high - p_low

    demark_x = np.where(
        p_close < p_open,
        p_high + 2.0 * p_low + p_close,
        np.where(p_close > p_open, 2.0 * p_high + p_low + p_close, p_high + p_low + 2.0 * p_close),
    )

    pp = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)

    if pivot_method == "Fibonacci":
        pp = (p_high + p_low + p_close) / 3.0
        r1, s1 = pp + 0.382 * p_range, pp - 0.382 * p_range
        r2, s2 = pp + 0.618 * p_range, pp - 0.618 * p_range
        r3, s3 = pp + 1.000 * p_range, pp - 1.000 * p_range
    elif pivot_method == "Camarilla":
        pp = (p_high + p_low + p_close) / 3.0
        r1, s1 = p_close + p_range * 1.1 / 12.0, p_close - p_range * 1.1 / 12.0
        r2, s2 = p_close + p_range * 1.1 / 6.0, p_close - p_range * 1.1 / 6.0
        r3, s3 = p_close + p_range * 1.1 / 4.0, p_close - p_range * 1.1 / 4.0
    elif pivot_method == "Woodie":
        pp = (p_high + p_low + 2.0 * p_close) / 4.0
        r1 = 2.0 * pp - p_low
        s1 = 2.0 * pp - p_high
        r2, s2 = pp + p_range, pp - p_range
        r3 = p_high + 2.0 * (pp - p_low)
        s3 = p_low - 2.0 * (p_high - pp)
    elif pivot_method == "DeMark":
        pp = demark_x / 4.0
        r1 = demark_x / 2.0 - p_low
        s1 = demark_x / 2.0 - p_high
        # DeMark 自身没有 R2/R3/S2/S3 —— 保持 na，register 会跳过
    else:  # Traditional
        pp = (p_high + p_low + p_close) / 3.0
        r1 = 2.0 * pp - p_low
        s1 = 2.0 * pp - p_high
        r2, s2 = pp + p_range, pp - p_range
        r3 = p_high + 2.0 * (pp - p_low)
        s3 = p_low - 2.0 * (p_high - pp)

    register("pp", pp, 8, CAT_PIVOT, use_pivots and pick(boolean("show_pp", show_pp)))
    register("r1", r1, 7, CAT_PIVOT, use_pivots and pick(boolean("show_r1", show_r1)))
    register("s1", s1, 7, CAT_PIVOT, use_pivots and pick(boolean("show_s1", show_s1)))
    register("r2", r2, 6, CAT_PIVOT, use_pivots and pick(boolean("show_r2", show_r2)))
    register("s2", s2, 6, CAT_PIVOT, use_pivots and pick(boolean("show_s2", show_s2)))
    register("r3", r3, 5, CAT_PIVOT, use_pivots and is_custom and boolean("show_r3", show_r3))
    register("s3", s3, 5, CAT_PIVOT, use_pivots and is_custom and boolean("show_s3", show_s3))

    # --- 摆动斐波 -----------------------------------------------------------
    fib_ph = pivothigh(high_s, fib_pivot_left, fib_pivot_right).to_numpy(dtype=float)
    fib_pl = pivotlow(low_s, fib_pivot_left, fib_pivot_right).to_numpy(dtype=float)
    last_high = np.full(n, np.nan)
    last_low = np.full(n, np.nan)
    last_high_bar = np.full(n, -1, dtype=np.int64)
    last_low_bar = np.full(n, -1, dtype=np.int64)
    cur_high_val = math.nan
    cur_low_val = math.nan
    cur_high_bar = -1
    cur_low_bar = -1
    for i in range(n):
        if not np.isnan(fib_ph[i]):
            cur_high_val = fib_ph[i]
            cur_high_bar = i - fib_pivot_right
        if not np.isnan(fib_pl[i]):
            cur_low_val = fib_pl[i]
            cur_low_bar = i - fib_pivot_right
        last_high[i] = cur_high_val
        last_low[i] = cur_low_val
        last_high_bar[i] = cur_high_bar
        last_low_bar[i] = cur_low_bar

    has_both = (last_high_bar >= 0) & (last_low_bar >= 0)
    fib_bull = has_both & (last_low_bar < last_high_bar)
    fib_bear = has_both & (last_high_bar < last_low_bar)
    fib_leg_size = np.abs(last_high - last_low)
    valid_leg = fib_leg_size >= fib_min_leg_atr * safe_atr
    fib_active = use_fib & valid_leg

    def fib_price(ratio: float) -> np.ndarray:
        """Pine ``f_fibPrice``：多头腿自高点回撤，空头腿自低点回撤。"""
        return np.where(fib_bull, last_high - fib_leg_size * ratio,
                        np.where(fib_bear, last_low + fib_leg_size * ratio, np.nan))

    def fib_ext(ratio: float) -> np.ndarray:
        """Pine ``f_extensionPrice``：自腿起点向反方向的扩展位。"""
        return np.where(fib_bull, last_low + fib_leg_size * ratio,
                        np.where(fib_bear, last_high - fib_leg_size * ratio, np.nan))

    out["ea_fib_high"] = last_high
    out["ea_fib_low"] = last_low
    out["ea_fib_leg"] = pd.array(
        np.where(fib_bull, 1, np.where(fib_bear, -1, 0)), dtype="Int64"
    )
    register("fib_236", fib_price(0.236), 5, CAT_FIB,
             fib_active & np.full(n, pick_opt(boolean("fib_236", fib_236))))
    register("fib_382", fib_price(0.382), 7, CAT_FIB,
             fib_active & np.full(n, pick(boolean("fib_382", fib_382))))
    register("fib_500", fib_price(0.500), 7, CAT_FIB,
             fib_active & np.full(n, pick(boolean("fib_500", fib_500))))
    register("fib_618", fib_price(0.618), 8, CAT_FIB,
             fib_active & np.full(n, pick(boolean("fib_618", fib_618))))
    register("fib_786", fib_price(0.786), 5, CAT_FIB,
             fib_active & np.full(n, pick_opt(boolean("fib_786", fib_786))))
    register("fib_100", fib_price(1.000), 5, CAT_FIB,
             fib_active & np.full(n, pick_opt(boolean("fib_100", fib_100))))
    register("fib_1272", fib_ext(1.272), 5, CAT_FIB,
             fib_active & np.full(n, pick_opt(boolean("fib_1272", fib_1272))))
    register("fib_1618", fib_ext(1.618), 6, CAT_FIB,
             fib_active & np.full(n, pick_opt(boolean("fib_1618", fib_1618))))

    # --- 枢轴聚类 S/R -------------------------------------------------------
    sr_ph = pivothigh(high_s, sr_pivot_left, sr_pivot_right).to_numpy(dtype=float)
    sr_pl = pivotlow(low_s, sr_pivot_left, sr_pivot_right).to_numpy(dtype=float)
    sup_price, sup_touch, res_price, res_touch = _cluster_scan(
        close, safe_atr, sr_ph, sr_pl, sr_pivot_right, sr_cluster_atr,
        sr_min_touches, sr_max_age, sr_supports, sr_resistances,
    )
    for rank in range(sup_price.shape[1]):
        out[f"ea_support_{rank + 1}"] = sup_price[:, rank]
        out[f"ea_support_touches_{rank + 1}"] = pd.array(sup_touch[:, rank], dtype="Int64")
    for rank in range(res_price.shape[1]):
        out[f"ea_resistance_{rank + 1}"] = res_price[:, rank]
        out[f"ea_resistance_touches_{rank + 1}"] = pd.array(res_touch[:, rank], dtype="Int64")

    # --- 整数关口 -----------------------------------------------------------
    # 与其它模块一致：列始终输出计算值，开关只决定是否进入 registry
    auto_step = boolean("round_auto_step", round_auto_step)
    major_step = np.array(
        [_auto_round_step(close[i], tick) if auto_step else round_manual_step for i in range(n)],
        dtype=float,
    )
    if boolean("round_quarter", round_quarter):
        smallest = major_step * 0.25
    elif boolean("round_half", round_half):
        smallest = major_step * 0.5
    else:
        smallest = major_step
    base_index = np.floor(close / smallest) if n else np.zeros(0)
    # Pine 只登记 offset != 0 的关口；这里同时暴露“最近的上下主关口”
    out["ea_round_major_above"] = (base_index + 1.0) * smallest
    out["ea_round_major_below"] = (base_index - 1.0) * smallest

    # --- VWAP ---------------------------------------------------------------
    day_key = _period_key(local, "D")
    week_key = _period_key(local, "W")
    month_key = _period_key(local, "M")

    def anchor_flags(key: pd.PeriodIndex) -> np.ndarray:
        """Pine ``timeframe.change(freq)``：新周期第一根 bar 为 True。"""
        codes = key.asi8 if hasattr(key, "asi8") else np.asarray(key.astype("int64"))
        flags = np.zeros(n, dtype=bool)
        if n:
            flags[0] = True
            flags[1:] = codes[1:] != codes[:-1]
        return flags

    vwap_sess, _, _ = _anchored_vwap(hlc3, vol, anchor_flags(day_key))
    vwap_week, _, _ = _anchored_vwap(hlc3, vol, anchor_flags(week_key))
    vwap_month, _, _ = _anchored_vwap(hlc3, vol, anchor_flags(month_key))
    register("vwap_session", vwap_sess, 7, CAT_VWAP,
             use_vwap and pick(boolean("vwap_session", vwap_session)))
    register("vwap_weekly", vwap_week, 6, CAT_VWAP,
             use_vwap and is_custom and boolean("vwap_weekly", vwap_weekly))
    register("vwap_monthly", vwap_month, 5, CAT_VWAP,
             use_vwap and is_custom and boolean("vwap_monthly", vwap_monthly))

    # --- 手动价位 -----------------------------------------------------------
    for slot, (enabled, price) in enumerate(
        (
            (manual1_enable, manual1_price),
            (manual2_enable, manual2_price),
            (manual3_enable, manual3_price),
            (manual4_enable, manual4_price),
        ),
        start=1,
    ):
        value = number(f"manual{slot}_price", price, -1e15, 1e15)
        active = use_manual and boolean(f"manual{slot}_enable", enabled) and value != 0.0
        register(f"manual_{slot}", value, 6, CAT_MANUAL, active)

    # --- 趋势曲线 -----------------------------------------------------------
    src = close
    if trend_preset == "Custom":
        fast_type, fast_len = trend_fast_type, trend_fast_length
        fast_phase, fast_power = float(trend_fast_phase), float(trend_fast_power)
        base_type, base_len = trend_base_type, trend_base_length
        base_phase, base_power = float(trend_base_phase), float(trend_base_power)
        anchor_type, anchor_len = trend_anchor_type, trend_anchor_length
        anchor_phase, anchor_power = float(trend_anchor_phase), float(trend_anchor_power)
    elif trend_preset == "Responsive":
        fast_type, fast_len, fast_phase, fast_power = "JMA-style", 9, 0.0, 2.0
        base_type, base_len, base_phase, base_power = "JMA-style", 21, 0.0, 2.0
        anchor_type, anchor_len, anchor_phase, anchor_power = "JMA-style", 50, 0.0, 2.0
    elif trend_preset == "Structural":
        fast_type, fast_len, fast_phase, fast_power = "EMA", 20, 0.0, 2.0
        base_type, base_len, base_phase, base_power = "SMA", 50, 0.0, 2.0
        anchor_type, anchor_len, anchor_phase, anchor_power = "SMA", 200, 0.0, 2.0
    else:  # Balanced
        fast_type, fast_len, fast_phase, fast_power = "JMA-style", 21, 0.0, 2.0
        base_type, base_len, base_phase, base_power = "EMA", 50, 0.0, 2.0
        anchor_type, anchor_len, anchor_phase, anchor_power = "SMA", 200, 0.0, 2.0

    trend_on = boolean("show_trend_curves", show_trend_curves)
    fast_value = _ma_value(fast_type, src, fast_len, fast_phase, fast_power)
    base_value = _ma_value(base_type, src, base_len, base_phase, base_power)
    anchor_value = _ma_value(anchor_type, src, anchor_len, anchor_phase, anchor_power)
    out["ea_trend_fast"] = fast_value
    out["ea_trend_base"] = base_value
    out["ea_trend_anchor"] = anchor_value
    as_levels = trend_on and boolean("trend_as_levels", trend_as_levels)
    if as_levels:
        trend_mask = np.full(n, True, dtype=bool)
        static.append((fast_value, 6, CAT_TREND, trend_mask))
        static.append((base_value, 6, CAT_TREND, trend_mask))
        static.append((anchor_value, 6, CAT_TREND, trend_mask))

    # --- registry 过滤 ------------------------------------------------------
    registered = np.zeros(n, dtype=np.int64)
    selected = np.zeros(n, dtype=np.int64)
    above_count = np.zeros(n, dtype=np.int64)
    below_count = np.zeros(n, dtype=np.int64)
    near_above = np.full(n, np.nan)
    near_above_dist = np.full(n, np.nan)
    near_above_priority = np.zeros(n, dtype=np.int64)
    near_above_category = np.zeros(n, dtype=np.int64)
    near_below = np.full(n, np.nan)
    near_below_dist = np.full(n, np.nan)
    near_below_priority = np.zeros(n, dtype=np.int64)
    near_below_category = np.zeros(n, dtype=np.int64)

    spacing = min_label_spacing_atr * safe_atr
    round_major_on = boolean("round_major", round_major)
    round_half_on = boolean("round_half", round_half)
    round_quarter_on = boolean("round_quarter", round_quarter)

    for i in range(n):
        if not master_enable:
            continue
        price = close[i]
        unit = safe_atr[i]
        candidates: list[tuple[float, int, int]] = []
        for values, priority, category, mask in static:
            if not mask[i]:
                continue
            value = values[i]
            if value == value:
                candidates.append((float(value), priority, category))
        # S/R 区域（Pine 在 registry 里是 isZone=true 的条目，优先级 8）
        for rank in range(sup_price.shape[1]):
            value = sup_price[i, rank]
            if value == value:
                candidates.append((float(value), 8, CAT_SR))
        for rank in range(res_price.shape[1]):
            value = res_price[i, rank]
            if value == value:
                candidates.append((float(value), 8, CAT_SR))
        # 整数关口：Pine 逐 offset 登记 (baseIndex + offset) * smallestStep
        if use_round:
            step = smallest[i]
            if step == step and step > 0:
                major = major_step[i]
                centre = base_index[i]
                for offset in range(-round_levels_each_side, round_levels_each_side + 1):
                    if offset == 0:
                        continue
                    level = (centre + offset) * step
                    normalized = level / major if major > 0 else 0.0
                    fraction = abs(normalized - round(normalized))
                    if fraction < 1e-6:
                        priority = 5 if round_major_on else None
                    elif abs(fraction - 0.5) < 1e-6:
                        priority = 3 if round_half_on else None
                    else:
                        priority = 3 if round_quarter_on else None
                    if priority is not None:
                        candidates.append((float(level), priority, CAT_ROUND))

        registered[i] = len(candidates)
        eligible: list[tuple[float, float, int, int]] = []
        for value, priority, category in candidates:
            side_ok = show_above if value >= price else show_below
            if not side_ok:
                continue
            if priority < min_priority:
                continue
            distance = abs(value - price) / unit
            if distance > max_distance_atr:
                continue
            eligible.append(((10 - priority) * 1_000_000.0 + distance, value, priority, category))
        eligible.sort(key=lambda item: item[0])

        accepted: list[tuple[float, int, int]] = []
        accepted_prices: list[float] = []
        for _key, value, priority, category in eligible:
            if spacing[i] > 0 and any(abs(existing - value) < spacing[i] for existing in accepted_prices):
                continue
            accepted.append((value, priority, category))
            accepted_prices.append(value)
            if len(accepted) >= max_levels:
                break

        selected[i] = len(accepted)
        best_above: tuple[float, int, int] | None = None
        best_below: tuple[float, int, int] | None = None
        for value, priority, category in accepted:
            if value >= price:
                above_count[i] += 1
                if best_above is None or value < best_above[0]:
                    best_above = (value, priority, category)
            else:
                below_count[i] += 1
                if best_below is None or value > best_below[0]:
                    best_below = (value, priority, category)
        if best_above is not None:
            near_above[i] = best_above[0]
            near_above_dist[i] = (best_above[0] - price) / unit
            near_above_priority[i] = best_above[1]
            near_above_category[i] = best_above[2]
        if best_below is not None:
            near_below[i] = best_below[0]
            near_below_dist[i] = (price - best_below[0]) / unit
            near_below_priority[i] = best_below[1]
            near_below_category[i] = best_below[2]

    out["ea_levels_registered"] = pd.array(registered, dtype="Int64")
    out["ea_levels_selected"] = pd.array(selected, dtype="Int64")
    out["ea_above_count"] = pd.array(above_count, dtype="Int64")
    out["ea_below_count"] = pd.array(below_count, dtype="Int64")
    out["ea_nearest_above"] = near_above
    out["ea_nearest_above_dist_atr"] = near_above_dist
    out["ea_nearest_above_priority"] = pd.array(near_above_priority, dtype="Int64")
    out["ea_nearest_above_category"] = pd.array(near_above_category, dtype="Int64")
    out["ea_nearest_below"] = near_below
    out["ea_nearest_below_dist_atr"] = near_below_dist
    out["ea_nearest_below_priority"] = pd.array(near_below_priority, dtype="Int64")
    out["ea_nearest_below_category"] = pd.array(near_below_category, dtype="Int64")

    return pd.DataFrame(out, index=idx)
