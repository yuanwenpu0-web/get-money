"""RGD 专用撮合：按 Pine 默认订单语义逐 bar 回放（契约 §5 全节）。

每根 bar 的处理顺序严格照抄脚本的语句顺序（`calc_on_every_tick=false`、
`process_orders_on_close=false`、`pyramiding=0`）：

1. **开盘**：成交上一根收盘提交的市价单（`strategy.entry` / `strategy.close`）。
   反向 entry 先平旧仓再开新仓（成交数量 = 旧 + 新，两腿费用分别计）；同向 entry
   在 `pyramiding=0` 下忽略。
2. **盘中**：只有 `exit_armed_bar < i` 的 exit 生效。信号 bar 成交当根**裸奔**
   （契约 N8 / R12）——不得「顺手」在成交瞬间挂止损。
3. **收盘**：先镜像 `closedtrades`（只计全平，部分平不 bump），再按
   「交易管理 → 入场 → trend flip」的源码顺序执行。管理在入场之前，所以新开仓
   当根不挂 exit；重挂的 stop 最早下一根生效（保本滞后一根，R12）。

TP1 有两个状态，混用会静默漏掉 50% 部分平仓：`tp1_seen` 是 Pine 的 `tp1Hit`
（本 bar 的 high/low 曾触目标，只把 stop 提到 entryRef 保本），`tp1_filled` 是
本地真实部分成交（决定是否撤掉 TP1 挂单）。Pine 在 `tp1Hit=True` 之后**仍会**
每 bar 重挂 `strategy.exit("L-TP1", limit=tp1Lvl)`，所以「价格曾见但当时没成交」
（例如入场 bar 裸奔）之后必须继续挂单等待成交。

`rgd_path` 与 `zigzag_execution.py:139` / `ocb_execution.py:140` 同源：近极值先走，
**平票先低后高**。注意 `ocb_execution.py:139` 的注释文字写成「先高后低」，与其
`:140` 代码矛盾，以代码为准（契约 §9 红线 12 / A29）。
"""
from __future__ import annotations

import math

import pandas as pd

from .rgd_orders import RgdBook


def rgd_path(open_price: float, high: float, low: float, close: float) -> tuple:
    """OHLC 插值路径：离开盘更近的极值先走，平票走 else 分支 → 先低后高。"""
    if abs(open_price - high) < abs(open_price - low):
        return (open_price, high, low, close)
    return (open_price, low, high, close)


def simulate_rgd(data, signals, params, initial, fee, slippage, interval):
    """返回 `(逐 bar 盯市权益, 分腿成交记录)`；末根未平仓写入 `equity.attrs`。

    `interval` 只用于与其它撮合签名对齐：本策略的时序由 bar 序号与 Pine 订单语义
    决定，不做「小时→bar」换算。
    """
    del interval
    if data.empty:
        return pd.Series(dtype=float, index=data.index), []

    rows = list(data.itertuples())
    decisions = list(signals.reindex(data.index).itertuples())
    book = RgdBook(initial, fee, slippage)
    pending: dict[str, dict] = {}      # 已挂 exit（持久，成交/取消才移除）
    market: dict[str, dict] = {}       # 上一根收盘提交的市价单，本根开盘成交
    closed_count, last_ct, last_exit = 0, 0, -9999
    values: list[float] = []
    cooldown = int(params["cooldown"])
    tp1_frac = float(params["tp1_pct"]) / 100.0
    risk_pct = float(params["risk_pct"])

    def cancel(prefix: str) -> None:
        for key in [key for key in pending if key.startswith(prefix)]:
            pending.pop(key, None)

    def close_position(price, stamp, reason: str) -> None:
        nonlocal closed_count
        p = book.position
        if p is None:
            return
        cancel("L-" if p["side"] == 1 else "S-")
        book.close_at(price, stamp, reason)
        closed_count += 1

    def tp1_fill(price, stamp) -> None:
        """真实 TP1 部分成交：只有这里置 `tp1_filled`，与 Pine 的 `tp1Hit` 区分。"""
        p = book.position
        if p is None or p["tp1_filled"]:
            return
        book.partial_close(price, stamp, tp1_frac, "take_profit_1")
        p["tp1_filled"] = True        # R13：实际成交后不再 re-arm
        p["tp1_seen"] = True          # 实际成交必然意味着价格已见
        cancel("L-TP1" if p["side"] == 1 else "S-TP1")

    def protect(open_price, high, low, stamp, i) -> None:
        """盘中止损/止盈判定；R14 的 stop 优先在这里落地。"""
        p = book.position
        if p is None or p["exit_armed_bar"] is None or i <= p["exit_armed_bar"]:
            return
        side, stop, target = p["side"], p["stop_lvl"], p["tp1_lvl"]
        if not math.isfinite(stop) or stop <= 0:
            return
        # (1) 开盘跳空：stop 优先；穿 limit 是既成部分平（更优价），不受 R14 约束。
        if (open_price - stop) * side <= 0:
            close_position(open_price, stamp, "stop_loss")
            return
        tp1_at_open = False
        if target > 0 and not p["tp1_filled"] and (open_price - target) * side >= 0:
            tp1_fill(open_price, stamp)
            tp1_at_open = True
            p = book.position
            if p is None:
                return
        # (2) R14（团队点名裁决）：盘中同根双触 -> stop 全平，不给 TP1 部分成交机会。
        #     这一条压过路径顺序，平票先低后高也不会让空头先吃到 TP1。
        if (low <= stop) if side == 1 else (high >= stop):
            close_position(stop, stamp, "stop_loss")
            return
        # (3) 仅 TP1 盘中触及 -> 部分平 tp1_pct%，余仓由 Trail 继续。
        if target > 0 and not p["tp1_filled"] and not tp1_at_open:
            if (high >= target) if side == 1 else (low <= target):
                tp1_fill(target, stamp)

    def fill_market(open_price, stamp) -> None:
        """开盘成交上一根收盘提交的市价单，按提交顺序处理。"""
        for order_id in list(market):
            order = market.pop(order_id)
            p = book.position
            if order["kind"] == "close":
                if p is not None and p["side"] == -order["side"]:
                    close_position(open_price, stamp, "trend_flip")
                continue
            side = order["side"]
            if p is not None and p["side"] == side:
                continue                       # pyramiding=0：同向 entry 忽略
            if p is not None:
                close_position(open_price, stamp, "signal_reverse")
            book.open_position(
                side, order["qty"], open_price, stamp,
                order["stop_lvl"], order["entry_ref"], order["tp1_lvl"],
            )

    def manage(decision, high, low, i) -> None:
        """L158-174：用本根 high/low 更新 tp1_seen 与 stop_lvl，并按 tp1_filled 重挂。"""
        p = book.position
        if p is None:
            return
        if p["side"] == 1:
            if math.isfinite(p["tp1_lvl"]) and high >= p["tp1_lvl"]:
                p["tp1_seen"] = True       # Pine tp1Hit：只驱动保本，不撤单
            defense = float(getattr(decision, "rgd_def_long", float("nan")))
            if math.isfinite(defense):
                p["stop_lvl"] = max(p["stop_lvl"], defense)
            if p["tp1_seen"]:
                p["stop_lvl"] = max(p["stop_lvl"], p["entry_ref"])
            trail_id, tp1_id = "L-Trail", "L-TP1"
        else:
            if math.isfinite(p["tp1_lvl"]) and low <= p["tp1_lvl"]:
                p["tp1_seen"] = True       # Pine tp1Hit：只驱动保本，不撤单
            defense = float(getattr(decision, "rgd_def_short", float("nan")))
            if math.isfinite(defense):
                p["stop_lvl"] = min(p["stop_lvl"], defense)
            if p["tp1_seen"]:
                p["stop_lvl"] = min(p["stop_lvl"], p["entry_ref"])
            trail_id, tp1_id = "S-Trail", "S-TP1"
        if p["exit_armed_bar"] is None:
            p["exit_armed_bar"] = i            # 首次挂 exit 的 bar：下一根才生效
        # 同 id 重挂（Pine strategy.exit 每 bar 覆盖参数），新 stop 生效于下一根。
        # 只有**实际成交过** TP1 才不再挂：`tp1_seen` 只是价格曾见，Pine 仍会重挂
        # L-TP1 等 limit 成交（契约 X4/X5/R12；漏掉这条会静默丢失 50% 部分平仓）。
        pending[trail_id] = {"stop": p["stop_lvl"], "armed_bar": i}
        if not p["tp1_filled"]:
            pending[tp1_id] = {
                "limit": p["tp1_lvl"], "stop": p["stop_lvl"],
                "qty_percent": tp1_frac, "armed_bar": i,
            }
        else:
            pending.pop(tp1_id, None)

    def submit_entry(decision, close_price, side: int) -> None:
        """L177-199：信号 bar 收盘定 q 与三档价位，市价单下一根开盘成交。"""
        suffix = "long" if side == 1 else "short"
        stp = float(getattr(decision, f"rgd_stp_{suffix}", float("nan")))
        rd = float(getattr(decision, f"rgd_rd_{suffix}", float("nan")))
        tp1 = float(getattr(decision, f"rgd_tp1_{suffix}", float("nan")))
        if not (math.isfinite(rd) and rd > 0 and math.isfinite(stp) and math.isfinite(tp1)):
            return                            # Pine `if rd > 0` 才下单
        equity = book.mark(close_price)
        qty = min(equity * risk_pct / 100.0 / rd, equity / close_price)
        if not math.isfinite(qty) or qty <= 0:
            return
        market["Long" if side == 1 else "Short"] = {
            "kind": "entry", "side": side, "qty": qty,
            # entryRef 取信号 bar 收盘价（不是成交价），tp1Lvl 以它为基准（契约 N3）
            "stop_lvl": stp, "entry_ref": close_price, "tp1_lvl": tp1,
        }

    for i, (row, decision) in enumerate(zip(rows, decisions)):
        stamp = row.Index
        open_price = float(row.open)
        high, low, close_price = float(row.high), float(row.low), float(row.close)

        fill_market(open_price, stamp)                    # (1) 开盘
        protect(open_price, high, low, stamp, i)          # (2) 盘中
        # (3) 收盘
        if closed_count > last_ct:                        # L132-134
            last_ct, last_exit = closed_count, i
        flat = book.position is None                      # L137
        cool_ok = (i - last_exit) > cooldown              # L140，严格大于
        long_break = bool(getattr(decision, "rgd_long_break", False))
        short_break = bool(getattr(decision, "rgd_short_break", False))
        long_pull = bool(getattr(decision, "rgd_long_pull_core", False)) and flat and cool_ok
        short_pull = bool(getattr(decision, "rgd_short_pull_core", False)) and flat and cool_ok
        allow_long = bool(getattr(decision, "rgd_allow_long", False))
        allow_short = bool(getattr(decision, "rgd_allow_short", False))
        manage(decision, high, low, i)                    # ③-a：先管理
        if (long_break or long_pull) and allow_long:      # ③-b：后入场
            submit_entry(decision, close_price, 1)
        if (short_break or short_pull) and allow_short:
            submit_entry(decision, close_price, -1)
        state = int(getattr(decision, "rgd_state", 0))    # ③-c：trend flip 纯平仓
        position = book.position
        # order 的 `side` 是**成交方向**（平多 = 卖出 -1、平空 = 买入 +1），
        # 与持仓方向相反；fill_market 用 `p["side"] == -order["side"]` 匹配。
        if state == -1 and not allow_short and position is not None and position["side"] == 1:
            market["CloseLong"] = {"kind": "close", "side": -1}
        if state == 1 and not allow_long and position is not None and position["side"] == -1:
            market["CloseShort"] = {"kind": "close", "side": 1}
        values.append(book.mark(close_price))

    equity = pd.Series(values, index=data.index, dtype=float)
    # 契约 §5.7 / A26：Pine 结束时持仓保留（盯市计入净值），不做 close_at_end 强平。
    equity.attrs["open_position"] = book.open_position_summary(float(rows[-1].close))
    return equity, book.trades
