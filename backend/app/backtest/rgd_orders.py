"""RGD 专用 broker 的成交原语：现金账户、开仓、全平、部分平、盯市。

口径与 `execution.py:68-126` 一致：一个来回 = 2 次手续费 + 2 次滑点，
开仓 `fill = price*(1+side*slip)`、平仓 `fill = price*(1-side*slip)`。

与 `ocb_execution.py` 的差别只有一处，但很关键：**不冻结保证金**。
`cash` 只扣已发生的费用，`mark = cash + 毛浮动`，等价于 Pine
`equity = initial_capital + netprofit(已平仓净额) + openprofit(毛浮动)`
（契约 §5.6 / §8 R11）。冻结保证金的写法会让持仓期净值凭空少一个 margin。
"""
from __future__ import annotations

import math


class RgdBook:
    """单币种、1× 无杠杆的现金账户 + 唯一持仓。

    `position` 字段与契约 §5.2 的 broker 状态一致：side/qty/entry/entry_fee/time/
    initial_qty/stop_lvl/entry_ref/tp1_lvl/tp1_seen/tp1_filled/exit_armed_bar。

    两个 TP1 标志必须分开，混用会静默漏掉 50% 部分平仓：
    - `tp1_seen` 对应 Pine `tp1Hit`：本 bar 的 high/low **曾触及** tp1Lvl，只驱动
      `stopLvl -> entryRef` 的保本，不表示 `strategy.exit("L-TP1")` 已成交。
    - `tp1_filled` 对应本地真实的部分成交：只有 `partial_close` 成功后才置位，
      只有它才允许撤掉 TP1 挂单（R13 的「一次性实际成交后不 re-arm」）。
    """

    def __init__(self, initial: float, fee: float, slippage: float) -> None:
        self.initial = float(initial)
        self.cash = float(initial)
        self.fee = float(fee)
        self.slippage = float(slippage)
        self.position: dict | None = None
        self.trades: list[dict] = []

    def mark(self, close: float) -> float:
        """逐 bar 盯市净值：已实现现金 + 持仓毛浮动（不扣未发生的平仓费）。"""
        p = self.position
        if p is None:
            return self.cash
        return self.cash + (close - p["entry"]) * p["qty"] * p["side"]

    def open_position(
        self, side: int, qty: float, price: float, stamp,
        stop_lvl: float, entry_ref: float, tp1_lvl: float,
    ) -> bool:
        """市价开仓。`qty` 是信号 bar 收盘算出的币数量，成交时按现金夹逼。

        夹逼沿用 `ocb_execution.py:102-104` 的保守写法：只在新仓买不起时缩量，
        不因次根开盘价略低就把「已下的单」丢掉（偏差 ≤ 滑点 + 费量级，§8 R11）。
        """
        fill = float(price) * (1.0 + side * self.slippage)
        affordable = self.cash / (fill * (1.0 + self.fee)) if fill > 0 else 0.0
        qty = min(float(qty), affordable)
        if not math.isfinite(fill) or fill <= 0 or not math.isfinite(qty) or qty <= 0:
            return False
        entry_fee = abs(fill * qty) * self.fee
        self.cash -= entry_fee
        self.position = {
            "side": int(side), "qty": float(qty), "entry": fill,
            "entry_fee": entry_fee, "time": stamp, "initial_qty": float(qty),
            "stop_lvl": float(stop_lvl), "entry_ref": float(entry_ref),
            "tp1_lvl": float(tp1_lvl),
            # tp1_seen = Pine tp1Hit（价格曾见，只驱动保本）
            # tp1_filled = 本地真实部分成交（决定是否撤掉 TP1 挂单）
            "tp1_seen": False, "tp1_filled": False,
            # None = 尚未挂过 exit：入场当根裸奔（契约 N8 / R12）
            "exit_armed_bar": None,
        }
        return True

    def close_at(self, price: float, stamp, reason: str) -> float:
        """全平当前持仓，返回该腿净盈亏。`position_id = entry_time` 供分腿合并。"""
        p = self.position
        fill = float(price) * (1.0 - p["side"] * self.slippage)
        gross = (fill - p["entry"]) * p["qty"] * p["side"]
        exit_fee = abs(fill * p["qty"]) * self.fee
        self.cash += gross - exit_fee
        net = gross - p["entry_fee"] - exit_fee
        margin = abs(p["entry"] * p["qty"])
        self.trades.append({
            "entry_time": str(p["time"]), "position_id": str(p["time"]), "exit_time": str(stamp),
            "side": "long" if p["side"] == 1 else "short",
            "entry_price": p["entry"], "exit_price": fill, "size": p["qty"],
            "pnl": net, "return_pct": net / margin * 100 if margin else 0.0,
            "fees": p["entry_fee"] + exit_fee, "status": reason,
            "leverage": 1, "margin": margin,
        })
        self.position = None
        return net

    def partial_close(self, price: float, stamp, frac: float, reason: str) -> float:
        """按当前持仓的比例部分平仓；入场费按同一比例结转，剩余仓继续持有。"""
        p = self.position
        frac = min(max(float(frac), 0.0), 1.0)
        qty = p["qty"] * frac
        entry_fee = p["entry_fee"] * frac
        fill = float(price) * (1.0 - p["side"] * self.slippage)
        gross = (fill - p["entry"]) * qty * p["side"]
        exit_fee = abs(fill * qty) * self.fee
        self.cash += gross - exit_fee
        net = gross - entry_fee - exit_fee
        margin = abs(p["entry"] * qty)
        self.trades.append({
            "entry_time": str(p["time"]), "position_id": str(p["time"]), "exit_time": str(stamp),
            "side": "long" if p["side"] == 1 else "short",
            "entry_price": p["entry"], "exit_price": fill, "size": qty,
            "pnl": net, "return_pct": net / margin * 100 if margin else 0.0,
            "fees": entry_fee + exit_fee, "status": reason,
            "leverage": 1, "margin": margin,
        })
        p["qty"] -= qty
        p["entry_fee"] -= entry_fee
        return net

    def open_position_summary(self, last_close: float) -> dict | None:
        """末根未平仓的收尾快照（契约 §5.7 / A26：不强平，只记录盯市）。"""
        p = self.position
        if p is None:
            return None
        return {
            "side": "long" if p["side"] == 1 else "short",
            "qty": p["qty"],
            "entry_price": p["entry"],
            "entry_time": str(p["time"]),
            "stop": p["stop_lvl"],
            "target": p["tp1_lvl"],
            "entry_fee": p["entry_fee"],
            "unrealized_pnl": (last_close - p["entry"]) * p["qty"] * p["side"],
        }
