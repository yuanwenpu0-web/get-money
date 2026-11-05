"""策略订单闭环状态机：信号 → 支撑位定价挂单 → 成交前复核 → 持仓管理 → 退出。

纯因果、无资金/成本概念（成本由 `backtest/closed_loop_execution.py` 按全局口径扣）。
时序约定（与项目 §2.4 一致，禁止当根信号当根成交）：

- 在 bar s **收盘**做出的决定（挂单 / 撤单 / 平仓）从 bar s+1 起生效；
- bar t 的 OHLC 只用来撮合「在 t-1 收盘时已经挂着」的单；
- 指标有效性、趋势破位都用 bar t 收盘时**已确认**的数据判断，不读未来。

状态只有三个：`FLAT` / `PENDING`（挂单未成交）/ `OPEN`（已成交持仓）。
指标失效、趋势破坏、止盈止损触发都是**事件**，不新增可重入的歧义状态。
所有迁移都带守卫：一个周期只有一个活动挂单、一次撤单、一次平仓。

进场价由 `cl_entry_mode` 决定（只影响定价/撮合，退出三档共用）：`0` 市价＝信号收盘提交、
`s+1` 开盘成交；`1` 反向回踩（默认）＝多头下方支撑／空头上方阻力等回撤；
`2` 对侧电平突破＝多头上方阻力／空头下方支撑挂突破单，触价按更保守价成交。
输出列由 `CLOSED_LOOP_COLUMNS` 声明，供专用 broker 回放；前缀 `st_cl_` 与
`supertrend_current` 的 `st_*` 命名一致，不覆盖任何既有列。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FLAT, PENDING, OPEN = 0, 1, 2
# 进场价模式：0 市价 / 1 反向回踩（默认，旧口径）/ 2 对侧电平突破。
ENTRY_MARKET, ENTRY_PULLBACK, ENTRY_BREAKOUT = 0, 1, 2
ORDER_KINDS = {ENTRY_MARKET: "market", ENTRY_PULLBACK: "pullback", ENTRY_BREAKOUT: "breakout"}
CLOSE_REASONS = ("stop_loss", "take_profit", "take_profit_dynamic", "trend_break")
CANCEL_EVENTS = ("cancel_indicator", "cancel_expired", "cancel_market_unfilled")
CLOSED_LOOP_COLUMNS = (
    "st_cl_state", "st_cl_event", "st_cl_side", "st_cl_order_id", "st_cl_cycle_id",
    "st_cl_level", "st_cl_pending_price", "st_cl_planned_sl", "st_cl_planned_tp",
    "st_cl_sl", "st_cl_tp", "st_cl_entry", "st_cl_fill_price", "st_cl_exit_price",
    "st_cl_close_reason", "st_cl_trend_valid", "st_cl_bars_pending", "st_cl_bars_held", "st_cl_entry_mode",
)


def _numeric(data: pd.DataFrame, name: str, fallback: str | None = None) -> np.ndarray:
    """缺失列一律退化为 NaN 或回退列，保证不依赖调用方是否做过特征富化。"""
    source = name if name in data.columns else fallback
    if source is None or source not in data.columns:
        return np.full(len(data), np.nan)
    return pd.to_numeric(data[source], errors="coerce").to_numpy(dtype=float)


def run_closed_loop(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """逐 bar 回放完整订单生命周期，返回与 `data` 同索引的闭环状态帧。"""
    n = len(data)
    if n == 0:
        return pd.DataFrame({key: pd.Series(dtype=float) for key in CLOSED_LOOP_COLUMNS})

    op = _numeric(data, "open", "close")
    hi = _numeric(data, "high", "close")
    lo = _numeric(data, "low", "close")
    cl = _numeric(data, "close")
    direction = _numeric(data, "st_direction")
    line = _numeric(data, "st_line")
    atr = _numeric(data, "st_atr")
    adx = _numeric(data, "st_adx")
    ema200 = _numeric(data, "st_ema200")
    signal = _numeric(data, "st_signal")
    support = _numeric(data, "st_support")
    resistance = _numeric(data, "st_resistance")

    # 枢轴连线要两个已确认枢轴，信号根常常还没有；回退到近 2×pivot_length+1 根的
    # 摆动低/高点（只含已收盘 bar，纯因果），两者都判不出来才放弃本轮挂单。
    window = max(3, 2 * int(params.get("pivot_length", 5)) + 1)
    min_periods = min(3, window)
    fb_support = pd.Series(lo).rolling(window, min_periods=min_periods).min().to_numpy(dtype=float)
    fb_resistance = pd.Series(hi).rolling(window, min_periods=min_periods).max().to_numpy(dtype=float)

    tp_mode = int(params.get("cl_tp_mode", 0))          # 0 = 固定 3R，1 = 动态追踪
    entry_mode = int(params.get("cl_entry_mode", ENTRY_PULLBACK))
    buffer_atr = float(params.get("cl_support_buffer_atr", 0.10))
    max_dist_atr = float(params.get("cl_max_entry_distance_atr", 3.0))
    expire = int(params.get("cl_pending_expire_bars", 0))
    activate_r = float(params.get("cl_dynamic_activate_r", 1.0))
    trail_atr = float(params.get("cl_trailing_atr_mult", 1.5))
    cooldown = int(params.get("cl_cooldown_bars", 0))
    sl_atr = float(params.get("sl_atr_mult", 1.5))
    target_r = float(params.get("target_r", 3.0))
    use_adx = bool(params.get("use_adx_filter", False))
    adx_th = float(params.get("adx_threshold", 50.0))
    use_ema = bool(params.get("use_ema200_filter", False))

    def indicator_ok(t: int, side: int) -> bool:
        """「指标仍满足」：趋势方向 + 可选 ADX / EMA200 过滤，且未被反向信号取代。"""
        if not np.isfinite(direction[t]) or direction[t] != side or signal[t] == -side:
            return False
        if use_adx and (not np.isfinite(adx[t]) or adx[t] <= adx_th):
            return False
        if use_ema and (not np.isfinite(ema200[t]) or (cl[t] - ema200[t]) * side <= 0):
            return False
        return True

    def trend_broken(t: int, side: int) -> bool:
        """「趋势已破」：Supertrend 方向翻转，或收盘跌破/升破趋势带。"""
        if not np.isfinite(direction[t]) or direction[t] != side:
            return True
        return bool(np.isfinite(line[t]) and (cl[t] - line[t]) * side <= 0)

    def quote_entry(t: int, side: int, mode: int):
        """按进场模式给出「参考价 / 触发价 / 风险距离」。

        `reference` 用于提交当根预配 SL/TP（市价取信号收盘、限价取触发价），
        成交后按实际价重锚保持 R 不变。回踩单触发价须在现价**有利**侧、
        突破单在**不利**侧，两者距离都不超过 `cl_max_entry_distance_atr` 倍 ATR。
        """
        if not (np.isfinite(atr[t]) and atr[t] > 0):
            return None
        risk = sl_atr * atr[t]
        if risk <= 0:
            return None
        if mode == ENTRY_MARKET:
            if not (np.isfinite(cl[t]) and cl[t] > 0):
                return None
            return dict(kind="market", level=np.nan, limit=np.nan,
                        reference=cl[t], risk=risk, touch="", mode=mode)
        if mode == ENTRY_PULLBACK:
            primary = support[t] if side == 1 else resistance[t]
            fallback = fb_support[t] if side == 1 else fb_resistance[t]
            touch = "down" if side == 1 else "up"
            favorable = True
        elif mode == ENTRY_BREAKOUT:
            primary = resistance[t] if side == 1 else support[t]
            fallback = fb_resistance[t] if side == 1 else fb_support[t]
            touch = "up" if side == 1 else "down"
            favorable = False
        else:
            return None
        for level in (primary, fallback):
            if not (np.isfinite(level) and level > 0):
                continue
            limit = level + side * buffer_atr * atr[t]
            distance = (cl[t] - limit) * side
            if not np.isfinite(distance):
                continue
            ok = (0 < distance <= max_dist_atr * atr[t]) if favorable else (
                -max_dist_atr * atr[t] <= distance < 0
            )
            if ok:
                return dict(kind=ORDER_KINDS[mode], level=level, limit=limit,
                            reference=limit, risk=risk, touch=touch, mode=mode)
        return None

    state, order, pos, decision = FLAT, None, None, None
    cooldown_left = 0
    order_seq = cycle_seq = 0
    rows: list[dict] = []

    for t in range(n):
        event, close_reason = "", ""
        fill_price = exit_price = level_used = np.nan
        closed_this_bar = False
        closed_id = closed_cycle = 0  # 平仓行仍要带上被平订单号，便于审计幂等

        # ── 阶段 1：执行 t-1 收盘做出的决定，早于本根任何撮合 ──
        if decision is not None:
            kind = decision[0]
            if kind == "cancel" and state == PENDING:
                event, order, state = decision[1], None, FLAT
            elif kind == "close" and state == OPEN:
                side, protective, tp = pos["side"], pos["protective"], pos["tp"]
                price = op[t] if np.isfinite(op[t]) else cl[t]
                reason = "trend_break"
                # 跳空越过保护价时按更保守的原因记账，但依旧只平一次
                if np.isfinite(protective) and (price - protective) * side <= 0:
                    reason = "take_profit_dynamic" if pos["dyn_on"] else "stop_loss"
                elif np.isfinite(tp) and (price - tp) * side >= 0:
                    reason = "take_profit"
                event, close_reason, exit_price = "close_" + reason, reason, price
                closed_id, closed_cycle = pos["id"], pos["cycle"]
                pos, state, closed_this_bar = None, FLAT, True
                cooldown_left = cooldown
            decision = None

        # ── 阶段 2：本根价格路径，只撮合 t-1 收盘时已挂的单 / 已开的仓 ──
        if state == PENDING and order is not None and t >= order["active_from"]:
            side = order["side"]
            price = None
            if order["kind"] == "market":
                # 市价单：只认下一根开盘价，开盘缺失才回退到该根收盘，绝不悬空到更后面的 bar
                if np.isfinite(op[t]) and op[t] > 0:
                    price = op[t]
                elif np.isfinite(cl[t]) and cl[t] > 0:
                    price = cl[t]
            else:
                limit = order["limit"]
                touched = (lo[t] <= limit) if order["touch"] == "down" else (hi[t] >= limit)
                if touched and np.isfinite(limit) and limit > 0:
                    if order["kind"] == "breakout":
                        # 突破单是 stop 语义：跳空越过触发价时按对交易者更差的价成交
                        price = max(op[t], limit) if side == 1 else min(op[t], limit)
                    else:
                        price = min(op[t], limit) if side == 1 else max(op[t], limit)
                    if not np.isfinite(price) or price <= 0:
                        price = limit
            if price is not None and np.isfinite(price) and price > 0:
                # 保护价在挂单阶段已按参考价配置；成交后按实际成交价重锚，保持 R 不变
                risk = order["risk"]
                sl = price - side * risk
                tp = price + side * risk * target_r if tp_mode == 0 else np.nan
                pos = dict(side=side, entry=price, sl=sl, tp=tp, protective=sl,
                           risk=risk, best=price, dyn_on=False, index=t,
                           id=order["id"], cycle=order["cycle"],
                           entry_mode=order["entry_mode"])
                state, fill_price, order = OPEN, price, None
                if not event:
                    event = "fill"
        elif state == OPEN and not closed_this_bar:
            side, protective, tp = pos["side"], pos["protective"], pos["tp"]
            price, reason = None, ""
            # 单一平仓出口：保护价（止损/动态追踪）优先于固定止盈，避免同根双平
            if np.isfinite(protective) and ((lo[t] <= protective) if side == 1 else (hi[t] >= protective)):
                reason = "take_profit_dynamic" if pos["dyn_on"] else "stop_loss"
                candidate = min(op[t], protective) if side == 1 else max(op[t], protective)
                price = protective if not np.isfinite(candidate) else candidate
            if price is None and np.isfinite(tp) and ((hi[t] >= tp) if side == 1 else (lo[t] <= tp)):
                reason = "take_profit"
                candidate = max(op[t], tp) if side == 1 else min(op[t], tp)
                price = tp if not np.isfinite(candidate) else candidate
            if price is not None:
                event, close_reason, exit_price = "close_" + reason, reason, price
                closed_id, closed_cycle = pos["id"], pos["cycle"]
                pos, state, closed_this_bar = None, FLAT, True
                cooldown_left = cooldown
            else:
                # 动态止盈：收盘更新追踪保护价，下一根才生效（不读未来）
                fav = hi[t] if side == 1 else lo[t]
                if np.isfinite(fav):
                    pos["best"] = max(pos["best"], fav) if side == 1 else min(pos["best"], fav)
                if tp_mode == 1 and np.isfinite(atr[t]) and (pos["best"] - pos["entry"]) * side >= activate_r * pos["risk"]:
                    pos["dyn_on"] = True
                    trail = pos["best"] - side * trail_atr * atr[t]
                    pos["protective"] = max(pos["sl"], trail) if side == 1 else min(pos["sl"], trail)

        # ── 阶段 3：t 收盘复核，决定只对 t+1 生效 ──
        if state == PENDING and order is not None:
            active = t >= order["active_from"]
            if active:
                order["bars_pending"] += 1
            if order["kind"] == "market" and active:
                # 市价单只在生效那一根有效；开盘与收盘都缺价时不许拖成挂单
                decision = ("cancel", "cancel_market_unfilled")
            elif not indicator_ok(t, order["side"]):
                decision = ("cancel", "cancel_indicator")
            elif active and expire > 0 and order["bars_pending"] >= expire:
                decision = ("cancel", "cancel_expired")
        elif state == OPEN and not closed_this_bar and t > pos["index"]:
            if trend_broken(t, pos["side"]):
                decision = ("close", "trend_break")
        elif state == FLAT and not closed_this_bar and not event and cooldown_left <= 0:
            sig = signal[t]
            prev = signal[t - 1] if t else 0.0
            # 只取信号上升沿：同一轮信号不得重复挂单
            if np.isfinite(sig) and sig in (-1.0, 1.0) and sig != prev:
                side = int(sig)
                if indicator_ok(t, side):
                    quote = quote_entry(t, side, entry_mode)
                    if quote is None:
                        event = "skip_invalid_atr" if entry_mode == ENTRY_MARKET else "skip_no_support"
                    else:
                        level_used, risk, reference = quote["level"], quote["risk"], quote["reference"]
                        order_seq += 1
                        cycle_seq += 1
                        order = dict(
                            id=order_seq, cycle=cycle_seq, side=side,
                            kind=quote["kind"], touch=quote["touch"], mode=quote["mode"],
                            level=level_used, limit=quote["limit"], reference=reference,
                            risk=risk, entry_mode=entry_mode,
                            planned_sl=reference - side * risk,
                            planned_tp=reference + side * risk * target_r if tp_mode == 0 else np.nan,
                            active_from=t + 1, bars_pending=0,
                        )
                        state, event = PENDING, "submit"

        cooldown_left = max(0, cooldown_left - 1)
        live = order if state == PENDING else (pos if state == OPEN else None)
        side_now = live["side"] if live else 0
        rows.append(dict(
            st_cl_state=state, st_cl_event=event, st_cl_side=side_now,
            st_cl_order_id=live["id"] if live else (closed_id if closed_this_bar else 0),
            st_cl_cycle_id=live["cycle"] if live else (closed_cycle if closed_this_bar else 0),
            st_cl_level=level_used,
            st_cl_pending_price=order["limit"] if state == PENDING else np.nan,
            st_cl_planned_sl=order["planned_sl"] if state == PENDING else np.nan,
            st_cl_planned_tp=order["planned_tp"] if state == PENDING else np.nan,
            st_cl_sl=(order["planned_sl"] if state == PENDING else pos["protective"] if state == OPEN else np.nan),
            st_cl_tp=(order["planned_tp"] if state == PENDING else pos["tp"] if state == OPEN else np.nan),
            st_cl_entry=pos["entry"] if state == OPEN else np.nan,
            st_cl_fill_price=fill_price, st_cl_exit_price=exit_price,
            st_cl_close_reason=close_reason,
            st_cl_trend_valid=int(indicator_ok(t, side_now)) if side_now else 0,
            st_cl_bars_pending=order["bars_pending"] if state == PENDING else 0,
            st_cl_bars_held=(t - pos["index"]) if state == OPEN else 0,
            st_cl_entry_mode=live["entry_mode"] if live else np.nan,
        ))

    return pd.DataFrame(rows, index=data.index)[list(CLOSED_LOOP_COLUMNS)]
