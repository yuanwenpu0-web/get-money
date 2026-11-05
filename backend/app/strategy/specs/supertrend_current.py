# SPDX-License-Identifier: MPL-2.0
"""Supertrend CURRENT 的本地交易适配（原脚本 MPL-2.0）。

原脚本是指标而非交易策略：信号时标出 1/2/3R 目标，没有指定仓位或分批成交。
本地按可配置比例在 TP1/TP2/TP3 分批止盈，并使用项目统一的资金、成本和撮合规则。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ...features.funding import FUNDING_FILTER_Z
from ..closed_loop import CLOSED_LOOP_COLUMNS, run_closed_loop
from ..registry import PrepareContext, StrategySpec, register
from ..signals import assemble_signal_frame, block, column

CODE = "supertrend_current"

DEFAULTS = {
    "supertrend_factor": 2.5,
    "atr_period": 14,
    "adx_length": 14,
    "adx_smoothing": 14,
    "use_adx_filter": False,
    "adx_threshold": 50.0,
    "use_ema200_filter": False,
    "enable_breakout": False,
    "pivot_length": 5,
    "enable_tp3_reentry": False,
    "enable_ema_reentry": False,
    "enable_aggressive_ema_reentry": False,
    "reentry_ema_length": 13,
    "sl_atr_mult": 1.5,
    "tp1_r": 1.0,
    "tp2_r": 2.0,
    "target_r": 3.0,
    "tp1_fraction": 1 / 6,
    "tp2_fraction": 1 / 3,
    "tp3_fraction": 1 / 2,
    # 指标没有时间止损、追踪止损和固定仓位；使用本地风险预算确定仓位。
    "max_hold_bars": 0,
    "min_hold_bars": 0,
    "cooldown_bars": 0,
    "trailing_enabled": False,
    # 订单闭环：默认关闭，保持原有「信号即成交 + 1R/2R/3R 分批止盈」口径不变。
    # 打开后由 strategy/closed_loop.py 状态机 + backtest/closed_loop_execution.py
    # 专用撮合接管，signals() 方向恒 0 做早路由（手册 H10）。
    "cl_enabled": False,
    # 进场价模式：0 市价（下一根开盘）/ 1 反向回踩支撑位挂单（默认，旧口径）/
    # 2 对侧电平突破挂单（多头取阻力、空头取支撑）。三种模式共用同一套退出逻辑。
    "cl_entry_mode": 1,
    "cl_tp_mode": 0,                   # 0 = 固定 3R 止盈；1 = 动态追踪止盈
    "cl_support_buffer_atr": 0.10,    # 挂单价 = 支撑/阻力 ∓ 该 ATR 缓冲
    "cl_max_entry_distance_atr": 3.0, # 现价到挂单价的最大距离（ATR 倍数）
    "cl_pending_expire_bars": 8,      # 0 = 挂单不超时；>0 = 最多等待 N 根（防挂单长期悬空）
    "cl_dynamic_activate_r": 1.0,     # 动态止盈在浮盈达到该 R 后启动追踪
    "cl_trailing_atr_mult": 1.5,      # 追踪保护价 = 最优价 − N × ATR
    "cl_cooldown_bars": 0,            # 平仓后冷却根数，防止同轮反复挂单
}
RANGES = {
    "supertrend_factor": (0.01, 50),
    "atr_period": (1, 500),
    "adx_length": (1, 500),
    "adx_smoothing": (1, 500),
    "adx_threshold": (1, 100),
    "pivot_length": (1, 50),
    "reentry_ema_length": (1, 500),
    "tp1_r": (0.1, 50),
    "tp2_r": (0.1, 50),
    "target_r": (0.1, 50),
    "tp1_fraction": (0, 1),
    "tp2_fraction": (0, 1),
    "tp3_fraction": (0, 1),
    "cl_entry_mode": (0, 2),
    "cl_tp_mode": (0, 1),
    "cl_support_buffer_atr": (0, 5),
    "cl_max_entry_distance_atr": (0.1, 20),
    "cl_pending_expire_bars": (0, 500),
    "cl_dynamic_activate_r": (0, 20),
    "cl_trailing_atr_mult": (0.1, 10),
    "cl_cooldown_bars": (0, 500),
}
PREPARE_KEYS = (
    "supertrend_factor", "atr_period", "adx_length", "adx_smoothing", "use_adx_filter",
    "adx_threshold", "use_ema200_filter", "enable_breakout", "pivot_length",
    "enable_tp3_reentry", "enable_ema_reentry", "enable_aggressive_ema_reentry",
    "reentry_ema_length", "sl_atr_mult", "tp1_r", "target_r",
    "cl_enabled", "cl_entry_mode", "cl_tp_mode", "cl_support_buffer_atr", "cl_max_entry_distance_atr",
    "cl_pending_expire_bars", "cl_dynamic_activate_r", "cl_trailing_atr_mult",
    "cl_cooldown_bars",
)


def validate(params: dict) -> None:
    if not params["tp1_r"] < params["tp2_r"] < params["target_r"]:
        raise ValueError("止盈目标必须满足 TP1 < TP2 < TP3")
    total = sum(params[key] for key in ("tp1_fraction", "tp2_fraction", "tp3_fraction"))
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("三级止盈比例之和必须为 100%")
    if int(params["cl_tp_mode"]) not in (0, 1):
        raise ValueError("cl_tp_mode 只能是 0（固定 3R）或 1（动态追踪止盈）")
    if int(params["cl_entry_mode"]) not in (0, 1, 2):
        raise ValueError("cl_entry_mode 只能是 0（市价）/ 1（反向回踩支撑位）/ 2（对侧电平突破）")
    # 市价模式不使用电平与挂单距离，这两个键不该变成无效配置的拦路虎
    if (int(params["cl_entry_mode"]) != 0
            and params["cl_support_buffer_atr"] > params["cl_max_entry_distance_atr"]):
        raise ValueError("支撑缓冲不能大于最大挂单距离")


def lookback(params: dict) -> int:
    return max(600, 4 * max(int(params["atr_period"]), int(params["adx_length"]) + int(params["adx_smoothing"]), int(params["reentry_ema_length"]), int(params["pivot_length"])))


def _rma(values: pd.Series, length: int) -> pd.Series:
    """Wilder 平滑，按前 length 个有效值的简单均值初始化。"""
    array = values.to_numpy(dtype=float)
    result = np.full(len(array), np.nan)
    count = 0
    total = 0.0
    prev = np.nan
    for i, x in enumerate(array):
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
    return pd.Series(result, index=values.index)


def _setup_signals(data: pd.DataFrame, params: dict, levels_out: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """重放原指标的 setup 和重入状态；不读取未来或实际账户持仓。

    `levels_out` 是可选的旁路输出：把每根已确认的枢轴支撑/阻力线交给订单闭环
    做挂单定价。两参数调用（既有单测与生产代码）的行为完全不变。
    """
    n = len(data)
    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)
    close = data["close"].to_numpy(dtype=float)
    atr = data["st_atr"].to_numpy(dtype=float)
    line = data["st_line"].to_numpy(dtype=float)
    direction = data["st_direction"].to_numpy(dtype=float)
    adx = data["st_adx"].to_numpy(dtype=float)
    ema200 = data["st_ema200"].to_numpy(dtype=float)
    ema = data["close"].ewm(span=int(params["reentry_ema_length"]), adjust=False).mean().to_numpy()
    raw = np.zeros(n)
    filtered = np.zeros(n)
    sup_levels = np.full(n, np.nan)
    res_levels = np.full(n, np.nan)
    pivot = int(params["pivot_length"])
    supports, resistances = [], []
    armed_long = armed_short = pull_long = pull_short = False
    trade_dir = 0
    stop = tp1 = tp3 = np.nan
    resolved = reached_tp3 = False

    def line_value(points, i):
        if len(points) < 2:
            return np.nan
        (x1, y1), (x2, y2) = points
        return y1 + (y2 - y1) / (x2 - x1) * (i - x1)

    for i in range(n):
        # pivot 在右侧 pivot 根已收盘之后才确认，平值选最右侧极值。
        j = i - pivot
        if j >= pivot:
            left, right = slice(j - pivot, j), slice(j + 1, i + 1)
            if (np.isfinite(low[j]) and np.all(low[j] <= low[left]) and np.all(low[j] < low[right])):
                supports = (supports + [(j, low[j])])[-2:]
            if (np.isfinite(high[j]) and np.all(high[j] >= high[left]) and np.all(high[j] > high[right])):
                resistances = (resistances + [(j, high[j])])[-2:]

        sup, sup_prev = line_value(supports, i), line_value(supports, i - 1)
        res, res_prev = line_value(resistances, i), line_value(resistances, i - 1)
        if levels_out is not None:
            sup_levels[i], res_levels[i] = sup, res
        up_width = down_width = np.nan
        if len(supports) == 2 and resistances:
            up_width = resistances[-1][1] - line_value(supports, resistances[-1][0])
        if len(resistances) == 2 and supports:
            down_width = line_value(resistances, supports[-1][0]) - supports[-1][1]
        upper = sup + up_width if up_width > 0 else res
        upper_prev = sup_prev + up_width if up_width > 0 else res_prev
        lower = res - down_width if down_width > 0 else sup
        lower_prev = res_prev - down_width if down_width > 0 else sup_prev
        prev_direction = direction[i - 1] if i else np.nan
        prev_close = close[i - 1] if i else np.nan
        prev_ema = ema[i - 1] if i else np.nan
        cross_up = close[i] > ema[i] and prev_close <= prev_ema
        cross_down = close[i] < ema[i] and prev_close >= prev_ema
        buy = direction[i] > 0 and prev_direction < 0
        sell = direction[i] < 0 and prev_direction > 0
        if params["enable_breakout"]:
            buy |= close[i] > upper and prev_close <= upper_prev
            sell |= close[i] < lower and prev_close >= lower_prev
        if params["enable_tp3_reentry"]:
            buy |= armed_long and direction[i] > 0 and low[i] <= line[i] and close[i] > line[i]
            sell |= armed_short and direction[i] < 0 and high[i] >= line[i] and close[i] < line[i]
        if params["enable_ema_reentry"]:
            buy |= armed_long and direction[i] > 0 and pull_long and cross_up
            sell |= armed_short and direction[i] < 0 and pull_short and cross_down
        if params["enable_aggressive_ema_reentry"]:
            buy |= armed_long and direction[i] > 0 and low[i] <= ema[i] and close[i] > ema[i]
            sell |= armed_short and direction[i] < 0 and high[i] >= ema[i] and close[i] < ema[i]
        # 原指标先检查信号，再根据本根 EMA 下穿/上穿武装后续重入。
        if params["enable_ema_reentry"]:
            pull_long |= armed_long and direction[i] > 0 and cross_down
            pull_short |= armed_short and direction[i] < 0 and cross_up
        adx_ok = not params["use_adx_filter"] or adx[i] > params["adx_threshold"]
        buy_ok = bool(buy and adx_ok and (not params["use_ema200_filter"] or close[i] > ema200[i]))
        sell_ok = bool(sell and adx_ok and (not params["use_ema200_filter"] or close[i] < ema200[i]))
        raw[i] = -1 if sell else 1 if buy else 0
        filtered[i] = -1 if sell_ok else 1 if buy_ok else 0
        if filtered[i]:
            # 极少数同根同时多空时，Pine 先 buy 后 sell，因此 sell 覆盖 setup。
            raw[i] = filtered[i]
            trade_dir = int(filtered[i])
            risk = atr[i] * float(params["sl_atr_mult"])
            stop = close[i] - trade_dir * risk
            tp1 = close[i] + trade_dir * risk * float(params["tp1_r"])
            tp3 = close[i] + trade_dir * risk * float(params["target_r"])
            resolved = reached_tp3 = False
            armed_long = armed_short = pull_long = pull_short = False
        elif trade_dir:
            stop_hit = low[i] <= stop if trade_dir == 1 else high[i] >= stop
            tp1_hit = high[i] >= tp1 if trade_dir == 1 else low[i] <= tp1
            tp3_hit = high[i] >= tp3 if trade_dir == 1 else low[i] <= tp3
            if not resolved:
                if stop_hit:
                    trade_dir = 0
                    resolved = True
                elif tp1_hit:
                    resolved = True
            if trade_dir and not reached_tp3 and tp3_hit:
                reached_tp3 = True
                if trade_dir == 1:
                    armed_long, pull_long = True, False
                else:
                    armed_short, pull_short = True, False
    if levels_out is not None:
        levels_out["st_support"] = sup_levels
        levels_out["st_resistance"] = res_levels
    return raw, filtered


def _attach_closed_loop(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """只补闭环状态列，绝不重算已有 st_* 列（单行 evaluate 不能重放历史）。

    若调用方给的是已富化但没有枢轴电平的帧，`run_closed_loop` 会因支撑位
    判定失败而保持中性状态，既有信号口径不受影响。
    """
    if set(CLOSED_LOOP_COLUMNS).issubset(data.columns):
        return data
    out = data.copy()
    loop = run_closed_loop(out, params)
    for col in CLOSED_LOOP_COLUMNS:
        out[col] = loop[col]
    return out


def prepare(frame: pd.DataFrame, params: dict, ctx: PrepareContext) -> pd.DataFrame:
    """计算当前周期指标，方向约定 +1 看多、-1 看空；所有计算只使用历史数据。"""
    # 引擎会在全帧预处理后取末行再次调用 evaluate_signals；不要丢弃历史状态。
    # 早退判据只认原有 7 列：老代码/单测会手工注入 st_* 后只传单行，
    # 若因为新增列缺失而重算，setup 状态机丢历史、方向判定就会变。
    indicator_cols = (
        "st_atr", "st_line", "st_direction", "st_adx", "st_ema200",
        "st_raw_signal", "st_signal",
    )
    if set(indicator_cols).issubset(frame.columns):
        return _attach_closed_loop(frame, params)
    data = frame.copy()
    for col in (*indicator_cols, "st_support", "st_resistance"):
        data[col] = np.nan
    if data.empty:
        return _attach_closed_loop(data, params)
    high, low, close = (pd.to_numeric(data[k], errors="coerce").astype(float) for k in ("high", "low", "close"))
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = _rma(tr, int(params["atr_period"]))
    hl2 = (high + low) / 2
    upper = (hl2 + float(params["supertrend_factor"]) * atr).to_numpy()
    lower = (hl2 - float(params["supertrend_factor"]) * atr).to_numpy()
    prices = close.to_numpy()
    out_line = np.full(len(data), np.nan)
    out_dir = np.full(len(data), np.nan)
    final_up = np.nan
    final_down = np.nan
    prev_line = np.nan
    for i in range(len(data)):
        if not np.isfinite(upper[i]) or not np.isfinite(lower[i]) or not np.isfinite(prices[i]):
            continue
        last_up = final_up if np.isfinite(final_up) else 0.0
        last_down = final_down if np.isfinite(final_down) else 0.0
        last_close = prices[i - 1] if i else np.nan
        final_down = lower[i] if lower[i] > last_down or last_close < last_down else last_down
        final_up = upper[i] if upper[i] < last_up or last_close > last_up else last_up
        # Pine 的首个有效 ATR 采用下跌方向，随后依据上根 Supertrend 所在带切换。
        if i == 0 or not np.isfinite(atr.iloc[i - 1]):
            direction = -1
        elif prev_line == last_up:
            direction = 1 if prices[i] > final_up else -1
        else:
            direction = -1 if prices[i] < final_down else 1
        prev_line = final_down if direction == 1 else final_up
        out_line[i], out_dir[i] = prev_line, direction

    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    # ta.dmi 的首根方向运动无前值，不参与平均。
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan
    di_len = int(params["adx_length"])
    di_tr = _rma(tr.where(prev_close.notna()), di_len).replace(0, np.nan)
    plus_di = 100 * _rma(plus_dm, di_len) / di_tr
    minus_di = 100 * _rma(minus_dm, di_len) / di_tr
    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum.where(di_sum.ne(0), 1)
    data["st_atr"] = atr
    data["st_line"] = out_line
    data["st_direction"] = out_dir
    data["st_adx"] = _rma(dx, int(params["adx_smoothing"]))
    data["st_ema200"] = close.ewm(span=200, adjust=False).mean()
    levels: dict = {}
    data["st_raw_signal"], data["st_signal"] = _setup_signals(data, params, levels_out=levels)
    data["st_support"] = levels["st_support"]
    data["st_resistance"] = levels["st_resistance"]
    # 公用资金/过滤层使用 atr_pct，与本策略配置的 ATR 周期保持一致。
    data["atr_pct"] = atr / close.replace(0, np.nan)
    # 订单闭环状态机：只加列、纯因果，不改动任何既有 st_* 列。
    return _attach_closed_loop(data, params)


def signals(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    side = column(data, "st_raw_signal").fillna(0.0)
    close = column(data, "close")
    atr = column(data, "st_atr")
    reason = pd.Series("", index=data.index, dtype=object)
    valid = pd.Series(np.isfinite(close) & np.isfinite(atr), index=data.index) & close.gt(0) & atr.gt(0)
    reason = block(reason, ~valid, "invalid_indicators")
    reason = block(reason, side.eq(0), "no_supertrend_signal")
    if params["use_adx_filter"]:
        reason = block(reason, side.ne(0) & ~column(data, "st_adx").gt(params["adx_threshold"]), "adx_filter")
    if params["use_ema200_filter"]:
        ema = column(data, "st_ema200")
        passes = (side.gt(0) & close.gt(ema)) | (side.lt(0) & close.lt(ema))
        reason = block(reason, side.ne(0) & ~passes, "ema200_filter")
    reason = block(reason, (atr / close).gt(params["max_atr_pct"]), "excess_volatility")
    if params["funding_filter"]:
        funding_z = column(data, "funding_zscore")
        reason = block(reason, funding_z.ge(FUNDING_FILTER_Z) & side.gt(0), "funding_filter_long")
        reason = block(reason, funding_z.le(-FUNDING_FILTER_Z) & side.lt(0), "funding_filter_short")
    # indicator 无强度概率，固定分数仅用于本地统一风险及仓位接口。
    signed = side
    score = pd.Series(1.0, index=data.index)
    # 原指标不提供概率预测，0.5 表示未建模，而不是声称必胜。
    probability = pd.Series(0.5, index=data.index)
    sl_pct = atr * float(params["sl_atr_mult"]) / close.replace(0, np.nan)
    tp_pct = sl_pct * float(params["target_r"])
    result = assemble_signal_frame(data, params, signed=signed, score=score, prob_up=probability,
                                   sl_pct=sl_pct, tp_pct=tp_pct, reason=reason,
                                   apply_min_distances=False)
    if params.get("cl_enabled", False):
        # 闭环模式：挂单/撤单/平仓由专用 broker 表达。通用撮合或巡检若按信号
        # 直接开仓，会与状态机里的挂单重复建仓，因此方向恒 0 做早路由（H10）。
        result["direction"] = 0
        result["block_reason"] = "closed_loop_requires_dedicated_broker"
        result["position_pct"] = 0.0
        result["leverage"] = 0
        result["risk_pct"] = 0.0
    return result


def describe(row: pd.Series) -> dict:
    def finite(key: str) -> float | None:
        try:
            value = float(row[key])
            return value if math.isfinite(value) else None
        except (KeyError, TypeError, ValueError):
            return None
    return {
        "supertrend": finite("st_line"), "supertrend_direction": finite("st_direction"),
        "adx": finite("st_adx"), "ema200": finite("st_ema200"), "atr": finite("st_atr"),
        "closed_loop_state": finite("st_cl_state"),
        "closed_loop_event": str(row.get("st_cl_event", "") or ""),
        "entry_mode": finite("st_cl_entry_mode"),
        "entry_price": finite("st_cl_entry"),
        "pending_price": finite("st_cl_pending_price"),
        "support_level": finite("st_support"), "resistance_level": finite("st_resistance"),
        "protective_stop": finite("st_cl_sl"), "take_profit": finite("st_cl_tp"),
        "close_reason": str(row.get("st_cl_close_reason", "") or ""),
    }


SPEC = register(StrategySpec(
    code=CODE,
    name="Supertrend 趋势反转",
    description=("Supertrend 方向翻转时做多或做空，可启用 ADX 强度和 EMA200 方向过滤；"
                 "ATR 止损，默认在 1R、2R、3R 分批止盈。"
                 "开启 cl_enabled 后改走订单闭环：进场价由 cl_entry_mode 选择（0 市价下一根开盘 / "
                 "1 反向回踩支撑位挂单（默认）/ 2 对侧电平突破挂单）→ 提交阶段同步配置止盈止损"
                 "（固定 3R 或动态追踪）→ 成交前指标复核撤单 → 持仓趋势破位平仓；"
                 "该模式只能由专用撮合回放，默认关闭。"),
    defaults=DEFAULTS,
    ranges=RANGES,
    lookback=lookback,
    prepare=prepare,
    signals=signals,
    describe=describe,
    validate=validate,
    bool_keys=("use_adx_filter", "use_ema200_filter", "enable_breakout", "enable_tp3_reentry",
               "enable_ema_reentry", "enable_aggressive_ema_reentry", "cl_enabled"),
    int_keys=("atr_period", "adx_length", "adx_smoothing", "pivot_length", "reentry_ema_length",
              "cl_entry_mode", "cl_tp_mode", "cl_pending_expire_bars", "cl_cooldown_bars"),
    grid={"supertrend_factor": (2.0, 2.5, 3.0), "target_r": (3.0, 4.0, 6.0)},
    prepare_keys=PREPARE_KEYS,
    default_enabled=False,
    tags=("tradingview", "trend", "closed_loop"),
))
