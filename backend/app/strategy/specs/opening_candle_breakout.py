r"""Opening Candle Breakout [OCB] —— 开盘首根突破：本地适配自 TradingView 脚本 © Swissy 2026（MPL-2.0）。

原脚本的思路来自 Zarattini & Aziz (2023)《Can Day Trading Really Be Profitable?》：
一天只在**开盘第一根K线**上做一次方向判断——

1. **入场条件**（收盘判定）：该K线落在 `09:30-09:35` 会话窗内 **且** 收阳/收阴明确
   （严格 `close > open` / `close < open`）**且** 实体够大（`|close-open| >= atr * minBodyATR`）。
2. **风险距离**：从收盘价量到这根K线的反侧影线（`close-low` / `high-close`），
   再按 ATR 夹到 `[minStopATR, maxStopATR] × ATR` 区间——这就是这笔的 1R。
3. **出场**：`close - risk` 止损、`close + risk * targetR` 止盈（上限10R），
   **或者**在 `15:55-16:00` 强平窗把还在手的仓位平掉，先到先算。
4. **仓位**：不是「固定名义百分比」，而是「每笔风险预算 ÷ 每单位风险距离」——
   止损越宽，仓位越小，每笔亏损恒定。
5. **每天最多一笔**（`tradedToday`）：同一交易日内出现第二个 setup 也不再下单。

为什么必须走专用撮合
--------------------
上面第 3、4、5 条通用撮合（`backtest/execution.py`）一条都表达不了：
它只有「按 bar 计时的时间止损」、ATR 反比定杠杆的仓位、以及统一的风险百分比截断，
既没有**会话强平**，也没有**按风险距离定手数**和**每日一笔**。
因此这里和 `simple_zigzag_double` 走同一条路：`signals()` 只负责把脚本的判定逐条算出来
并把全部中间量托管给信号帧，真正的回放由 `backtest/ocb_execution.py` 完成，
回测入口会自动路由过去（见 `backtest/runner.run_ocb_backtest`）。

已知取舍
--------
- 原脚本按合约张数下单（`floor` 取整 + `maxQty` + `allowMin`），加密货币可碎股交易，
  因此数量保留小数；名义上限改用 `max_position_pct`（名义 ≤ 权益的这一比例），
  `allow_min_position` 对应脚本的 `allowMin`。
- 原脚本的 `commission_type=cash_per_contract`、`slippage=2 ticks` 是交易所口径；
  本地统一走双向 `fee_rate`/`slippage` 比例成本。

移植说明与逐条语义对照见 `docs/opening-candle-breakout-port.md`，
本地历史检验（含为什么默认不启用）见 `docs/opening-candle-breakout-strategy.md`。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...features.session import local_day, local_minutes, session_flags
from ...utils import round_price
from ..registry import PrepareContext, StrategySpec, register
from ..signals import block

CODE = "opening_candle_breakout"

# 脚本 1:1 默认值：会话 09:30-09:35 / 15:55-16:00（America/New_York）,
# minBodyATR=0.5, atrLen=14, minStopATR=0.10, maxStopATR=2.0, tpR=10, useTP=true。
# riskMode 脚本默认是「Fixed USD + 250」；在加密货币账户上「%$ 权益」更稳，
# 需要 1:1 复刻时用 risk_mode=0 + risk_usd=250。
DEFAULTS = {
    # ── 会话（分钟数 = hour*60+minute，会话时区墙上时间）
    "first_session_start_min": 570,   # 09:30
    "first_session_end_min": 575,     # 09:35
    "flat_session_start_min": 955,    # 15:55
    "flat_session_end_min": 960,      # 16:00
    "session_tz": 1,                  # 1 = America/New_York
    "tz_offset_minutes": -300,        # tzdata 缺失时的固定偏移兜底（EST）
    "trade_longs": True,
    "trade_shorts": True,
    # ── 形态过滤
    "min_body_atr": 0.5,
    "atr_len": 14,
    # ── 止损夹逼
    "min_stop_atr": 0.10,
    "max_stop_atr": 2.0,
    # ── 出场
    "target_r": 10.0,
    "use_target": True,
    # 开盘跳空直接越过止盈价时的成交价约定：False = 保守按限价成交，
    # True = 按 TradingView 的「限价单以更优价成交」取开盘价。
    "tp_gap_at_open": False,
    # ── 仓位与风险
    "risk_mode": 1,              # 0 = 固定金额 risk_usd；1 = 权益百分比 risk_per_trade_pct
    "risk_usd": 250.0,
    "allow_min_position": False,
    "max_position_pct": 0.25,
    "base_position_pct": 0.25,
    "max_leverage": 1,
    # ── 通用管线的开关：本策略一律全关，成本/趋势/资金费/追踪都不参与
    "regime_filter": False,
    "cost_filter": False,
    "funding_filter": False,
    "trailing_enabled": False,
    "cooldown_bars": 0,
    "min_hold_bars": 0,
    "max_hold_bars": 0,
}

RANGES = {
    "first_session_start_min": (0, 1439),
    "first_session_end_min": (1, 1440),
    "flat_session_start_min": (0, 1439),
    "flat_session_end_min": (0, 1440),
    "session_tz": (0, 4),
    "tz_offset_minutes": (-720, 840),
    "min_body_atr": (0, 5),
    "atr_len": (5, 200),
    "min_stop_atr": (0.01, 5),
    "max_stop_atr": (0.25, 10),
    "target_r": (0.5, 50),
    "risk_mode": (0, 1),
    "risk_usd": (0.01, 1_000_000),
}

INT_KEYS = (
    "atr_len",
    "first_session_start_min",
    "first_session_end_min",
    "flat_session_start_min",
    "flat_session_end_min",
    "session_tz",
    "tz_offset_minutes",
    "risk_mode",
)

BOOL_KEYS = (
    "trade_longs",
    "trade_shorts",
    "use_target",
    "tp_gap_at_open",
    "allow_min_position",
)

# 只要这些键一变，`prepare()` 的产出就得重算
PREPARE_KEYS = (
    "first_session_start_min",
    "first_session_end_min",
    "flat_session_start_min",
    "flat_session_end_min",
    "session_tz",
    "tz_offset_minutes",
    "trade_longs",
    "trade_shorts",
    "min_body_atr",
    "atr_len",
    "min_stop_atr",
    "max_stop_atr",
    "target_r",
    "use_target",
)

# 交给专用撮合托管的中间量
OCB_COLUMNS = (
    "ocb_local_minute",
    "ocb_is_first",
    "ocb_is_flat",
    "ocb_new_day",
    "ocb_atr",
    "ocb_body_atr",
    "ocb_dir",
    "ocb_risk",
    "ocb_risk_atr",
    "ocb_stop",
    "ocb_target",
)

# 参数寻优空间：只放真的会改变读数的档位。
# target_r / min_body_atr / stop 夹逼是这套结构的三根杠杆；
# 会话窗本身不参与寻优（改了就等于换策略语义）。
GRID = {
    "min_body_atr": (0.0, 0.5, 1.0),
    "target_r": (4.0, 10.0, 20.0),
    "min_stop_atr": (0.10, 0.25),
    "max_stop_atr": (1.0, 2.0, 3.0),
    "risk_per_trade_pct": (0.005, 0.01, 0.02),
}


def wilder_atr(frame: pd.DataFrame, length: int) -> pd.Series:
    """Wilder ATR = RMA(TR, length)，首个有效值用前 `length` 根的 SMA 播种。

    这一步刻意不复用 `features.indicators._atr`：那里的 `ewm(alpha=1/n)` 从第一根就开始递推，
    而 Pine 的 `ta.rma` 是先取 SMA(length) 播种（`sum := na(sum[1]) ? sma(src, length) : ...`）。
    预热区间的差异会直接改变止损宽度，不能省。
    """
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return rma(true_range, length)


def rma(series: pd.Series, length: int) -> pd.Series:
    """Pine `ta.rma`：SMA(length) 播种 + 之后 `alpha*x + (1-alpha)*prev`。"""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    if len(values) < length:
        return pd.Series(out, index=series.index)
    alpha = 1.0 / length
    acc = float(np.nanmean(values[:length])) if np.isfinite(values[:length]).all() else np.nan
    out[length - 1] = acc
    for i in range(length, len(values)):
        current = values[i]
        if np.isfinite(current):
            acc = alpha * current + (1 - alpha) * acc
        elif np.isnan(acc):
            acc = np.nan
        out[i] = acc
    return pd.Series(out, index=series.index)


def lookback(params: dict) -> int:
    """ATR 预热：Wilder RMA 需要 atr_len 根播种，再留 3 倍跨度让读数稳定。"""
    return max(600, 3 * int(params.get("atr_len", 14)) + 200)


def prepare(frame: pd.DataFrame, params: dict, ctx: PrepareContext) -> pd.DataFrame:
    """补齐 OCB 判定所需的全部列：会话布尔、Wilder ATR、夹逼后的风险距离与止损/目标价位。

    全部只依赖当前及过去的 bar（因果）；会话时区换算依赖索引为 DatetimeIndex。
    """
    if set(OCB_COLUMNS).issubset(frame.columns):
        return frame
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Opening Candle Breakout 需要 DatetimeIndex 才能判定交易时段")
    data = frame.copy()
    index = data.index
    tz_id = int(params["session_tz"])
    offset = int(params["tz_offset_minutes"])

    data["ocb_local_minute"] = local_minutes(index, tz_id, offset)
    data["ocb_is_first"] = session_flags(
        index, params["first_session_start_min"], params["first_session_end_min"], tz_id, offset
    )
    data["ocb_is_flat"] = session_flags(
        index, params["flat_session_start_min"], params["flat_session_end_min"], tz_id, offset
    )
    day = local_day(index, tz_id, offset)
    # 脚本：`dayofmonth(time, tz) != dayofmonth(time[1], tz)`；首根天然为 True
    data["ocb_new_day"] = day.ne(day.shift(1))

    atr = wilder_atr(data, int(params["atr_len"]))
    data["ocb_atr"] = atr
    open_, high, low, close = (
        pd.to_numeric(data[key], errors="coerce") for key in ("open", "high", "low", "close")
    )
    body = (close - open_).abs()
    data["ocb_body_atr"] = body / atr.replace(0, np.nan)

    min_body = float(params["min_body_atr"])
    body_ok = pd.Series(True, index=index) if min_body <= 0 else body.ge(atr * min_body).fillna(False)
    # 严格大小比较：十字星（close == open）即使关掉实体过滤也没有方向
    long_setup = data["ocb_is_first"] & close.gt(open_) & body_ok & bool(params["trade_longs"])
    short_setup = data["ocb_is_first"] & close.lt(open_) & body_ok & bool(params["trade_shorts"])
    direction = pd.Series(
        np.where(long_setup, 1, np.where(short_setup, -1, 0)), index=index, dtype=int
    )
    data["ocb_dir"] = direction

    raw_risk = pd.Series(
        np.where(long_setup, close - low, np.where(short_setup, high - close, np.nan)),
        index=index,
    )
    # 脚本：risk = min(max(rawRisk, atr*minStopATR), atr*maxStopATR)
    risk = raw_risk.clip(lower=atr * float(params["min_stop_atr"]), upper=atr * float(params["max_stop_atr"]))
    data["ocb_risk"] = risk
    data["ocb_risk_atr"] = risk / atr.replace(0, np.nan)

    signed = direction.astype(float).replace(0, np.nan)
    data["ocb_stop"] = close - signed * risk
    data["ocb_target"] = (
        close + signed * risk * float(params["target_r"]) if params["use_target"] else np.nan
    )
    return data


def signals(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """按脚本判定顺序逐条门控，并把所有中间量托管给专用撮合。

    统一信号帧里 `direction` 恒为 0：会话强平与「每笔风险定手数」表达不出来，
    不许它偷偷走通用撮合（否则出场规则就不再是脚本里的那套）。
    """
    index = data.index
    close = pd.to_numeric(data["close"], errors="coerce")
    open_ = pd.to_numeric(data["open"], errors="coerce")
    atr = pd.to_numeric(data.get("ocb_atr"), errors="coerce")
    direction = pd.to_numeric(data.get("ocb_dir"), errors="coerce").fillna(0)
    is_first = data.get("ocb_is_first", pd.Series(False, index=index)).astype(bool)
    setup = direction.ne(0)
    valid = (
        pd.concat([open_, close, atr], axis=1).gt(0).all(axis=1)
        & np.isfinite(open_)
        & np.isfinite(close)
        & np.isfinite(atr)
    )

    body = (close - open_).abs()
    min_body = float(params["min_body_atr"])
    body_ok = pd.Series(True, index=index) if min_body <= 0 else body.ge(atr * min_body).fillna(False)
    raw_dir = pd.Series(np.where(close.gt(open_), 1, np.where(close.lt(open_), -1, 0)), index=index)

    reason = pd.Series("", index=index, dtype=object)
    reason = block(reason, ~valid, "invalid_indicators")
    reason = block(reason, ~is_first, "not_opening_candle")
    reason = block(reason, raw_dir.eq(1) & (not params["trade_longs"]), "long_disabled")
    reason = block(reason, raw_dir.eq(-1) & (not params["trade_shorts"]), "short_disabled")
    reason = block(reason, ~body_ok, "body_too_small")
    reason = reason.mask(reason.eq(""), "requires_session_broker")

    risk = pd.to_numeric(data.get("ocb_risk"), errors="coerce")
    target_r = float(params["target_r"]) if params["use_target"] else 0.0
    sl_pct = (risk / close.replace(0, np.nan)).where(setup).fillna(0.0)
    tp_pct = (sl_pct * target_r) if target_r else pd.Series(0.0, index=index)
    target_r_ratio = pd.Series(np.where(setup, target_r, 0.0), index=index, dtype=float)

    result = pd.DataFrame(
        {
            # signed 只是观察量（脚本的方向判断），不代表可以走通用撮合
            "signed": direction.where(setup, 0.0),
            "score": pd.Series(0.5, index=index),
            "prob_up": pd.Series(0.5, index=index),
            "direction": 0,
            "block_reason": reason,
            "position_pct": 0.0,
            "leverage": 0,
            "sl_pct": sl_pct.fillna(0.0),
            "tp_pct": tp_pct.fillna(0.0),
            "risk_pct": 0.0,
            "net_reward_risk": target_r_ratio,
        },
        index=index,
    )
    for key in OCB_COLUMNS:
        result[key] = data[key] if key in data.columns else np.nan
    return result


def describe(row: pd.Series) -> dict:
    """末根观测快照（纯展示，不参与任何决策）。"""

    def value(name, default=0.0):
        try:
            number = float(row[name])
        except (KeyError, TypeError, ValueError):
            return default
        return number if np.isfinite(number) else default

    minute = int(value("ocb_local_minute"))
    return {
        "local_time": f"{minute // 60:02d}:{minute % 60:02d}",
        "atr": round_price(value("ocb_atr")),
        "body_atr": round(value("ocb_body_atr"), 4),
        "risk_atr": round(value("ocb_risk_atr"), 4),
        "risk": round_price(value("ocb_risk")),
        "stop_price": round_price(value("ocb_stop")),
        "target_price": round_price(value("ocb_target")),
        "setup_dir": int(value("ocb_dir")),
        "in_session": bool(row.get("ocb_is_first", False)) or bool(row.get("ocb_is_flat", False)),
    }


SPEC = register(
    StrategySpec(
        code=CODE,
        name="开盘首根突破",
        description=(
            "一天只在开盘第一根 K 线（NY 09:30-09:35）上下一次注：实体方向定多空、"
            "实体需达 0.5 ATR，风险距离 = 收盘到反向影线并按 [0.1, 2]×ATR 夹逼，"
            "10R 固定止盈或 15:55-16:00 会话强平，每天最多一笔；"
            "仓位按「每笔风险预算 ÷ 风险距离」定，不是固定名义百分比"
        ),
        defaults=DEFAULTS,
        ranges=RANGES,
        lookback=lookback,
        prepare=prepare,
        signals=signals,
        describe=describe,
        bool_keys=BOOL_KEYS,
        int_keys=INT_KEYS,
        grid=GRID,
        prepare_keys=PREPARE_KEYS,
        # 尚未在本地数据上做过统计检验：会话强平与「每日一笔」只能由专用撮合表达，
        # 也不参与自动巡检的资金竞争。验证结论见 docs/opening-candle-breakout-strategy.md。
        default_enabled=False,
        tags=("tradingview", "session", "breakout"),
    )
)
