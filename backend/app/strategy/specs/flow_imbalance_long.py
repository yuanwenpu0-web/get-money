"""订单流净买入策略：真实主动买入量的极端不平衡作为择时信号。

## 与「估算 delta」的区别

本策略只用交易所 K 线自带的 `taker_buy_volume`（币安周期汇总的真实主动成交），
净主动买入量 = `2 × taker_buy_volume − volume`，等价于「主动买 − 主动卖」。
仓库里 `features/absorption.py` 与 `catalog_indicators/cvd_bias.py` 的 delta/CVD
是按 OHLC 位置**估算**方向，与本策略不是同一口径，不能互相对照。

## 为什么是「极端值」而不是线性打分

400 天 / 19 币探针（`scripts/probe_flow_imbalance.py`）显示线性 IC 全部 ≤ 0.008，
但分组检验（`scripts/probe_flow_quantile.py`）显示只有尾部才有方向性：
做多侧超额随持仓周期单调上升，做空侧几乎无 edge。因此策略只做**单边做多**，
且入场条件是「不平衡进入自身历史的极端分位」而不是连续打分。

## 前视约束

探针用全样本分位（含未来信息）只是上限估计；本策略的分位阈值一律用
`rolling(quantile_window)` 的**历史分位**，且 `shift(1)` 保证只用已收盘 bar。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import PrepareContext, StrategySpec, register
from ..signals import assemble_signal_frame, block, column

CODE = "flow_imbalance_long"

DEFAULTS = {
    # 净买入不平衡的累计窗口（bar 数）：探针里 fi_48/fi_96 优于 fi_1
    "flow_window": 48,
    # 分位阈值：不平衡进入自身历史的该分位才入场。尾部越极端样本越少但 edge 越强
    "entry_quantile": 0.95,
    # 计算滚动分位用的历史长度（bar 数）。必须显著长于 flow_window
    "quantile_window": 480,
    # 附加过滤：要求累计不平衡为正（与分位条件同向，避免低波动期误触发）
    "require_positive_flow": True,
    # 附加过滤：只在放量时入场（成交量相对自身历史的 z 分数）
    "min_volume_z": 0.0,
    # 附加过滤：趋势方向，要求收盘在 EMA 之上才做多（0 表示关闭）
    "trend_ema": 0,
    # 反向信号：不平衡跌破低分位时平仓（只平不开空）
    "exit_quantile": 0.5,
    # 结构止损与目标，按 ATR 倍数
    "sl_atr_mult": 2.0,
    "target_r": 2.0,
    # 最长持仓，0 表示不限制（靠止损/目标/反向信号出场）
    "max_hold_bars": 96,
}

RANGES = {
    "flow_window": (1, 480),
    "entry_quantile": (0.5, 0.999),
    "quantile_window": (60, 5000),
    "min_volume_z": (-5, 10),
    "trend_ema": (0, 500),
    "exit_quantile": (0.0, 0.999),
    "sl_atr_mult": (0.1, 10),
    "target_r": (0.2, 20),
}

PREPARE_KEYS = (
    "flow_window", "entry_quantile", "quantile_window", "require_positive_flow",
    "min_volume_z", "trend_ema", "exit_quantile",
)

COLUMNS = (
    "fi_net", "fi_ratio", "fi_threshold", "fi_exit_threshold", "fi_upper",
    "fi_volume_z", "fi_trend_ok", "fi_signal", "fi_exit",
)


def validate(params: dict) -> None:
    if int(params["quantile_window"]) <= int(params["flow_window"]):
        raise ValueError("分位窗口必须大于不平衡累计窗口，否则分位由同一批样本决定")
    if float(params["entry_quantile"]) <= float(params["exit_quantile"]):
        raise ValueError("入场分位必须高于出场分位")


def lookback(params: dict) -> int:
    return max(
        600,
        int(params["quantile_window"]) + int(params["flow_window"]) + 60,
        int(params["trend_ema"]) * 4,
    )


def prepare(frame: pd.DataFrame, params: dict, ctx: PrepareContext) -> pd.DataFrame:
    """计算净买入不平衡与滚动分位阈值；全部只使用已收盘 bar。"""
    if set(COLUMNS).issubset(frame.columns):
        return frame
    data = frame.copy()
    for name in COLUMNS:
        data[name] = np.nan
    if data.empty:
        return data

    volume = pd.to_numeric(data["volume"], errors="coerce").astype(float)
    taker = pd.to_numeric(data.get("taker_buy_volume"), errors="coerce").astype(float)
    # 缺失主动买入量的 bar 不能当成「主动买入为 0」，整根作废
    net = (2 * taker - volume).where(taker.notna() & volume.gt(0))
    data["fi_net"] = net

    window = int(params["flow_window"])
    total = volume.where(net.notna()).rolling(window, min_periods=window).sum()
    net_sum = net.rolling(window, min_periods=window).sum()
    ratio = net_sum / total.replace(0, np.nan)
    data["fi_ratio"] = ratio.clip(-1, 1)

    q_window = int(params["quantile_window"])
    # shift(1)：本根的分位阈值只由之前 q_window 根已收盘 bar 决定，避免同根自指
    history = ratio.shift(1)
    data["fi_threshold"] = history.rolling(q_window, min_periods=int(q_window * 0.5)).quantile(
        float(params["entry_quantile"])
    )
    data["fi_exit_threshold"] = history.rolling(q_window, min_periods=int(q_window * 0.5)).quantile(
        float(params["exit_quantile"])
    )
    # 历史分布的上沿：用于把「超出多少」换算成 0~1 的强度，供仓位分层
    data["fi_upper"] = history.rolling(q_window, min_periods=int(q_window * 0.5)).quantile(0.995)

    vol_mean = volume.rolling(q_window, min_periods=int(q_window * 0.5)).mean()
    vol_std = volume.rolling(q_window, min_periods=int(q_window * 0.5)).std()
    data["fi_volume_z"] = (volume - vol_mean) / vol_std.replace(0, np.nan)

    ema_length = int(params["trend_ema"])
    if ema_length > 0:
        ema = pd.to_numeric(data["close"], errors="coerce").ewm(span=ema_length, adjust=False).mean()
        data["fi_trend_ok"] = (pd.to_numeric(data["close"], errors="coerce") > ema).astype(float)
    else:
        data["fi_trend_ok"] = 1.0

    entry = ratio.gt(data["fi_threshold"]) & data["fi_threshold"].notna()
    if params["require_positive_flow"]:
        entry &= ratio.gt(0)
    if float(params["min_volume_z"]) > 0:
        entry &= data["fi_volume_z"].ge(float(params["min_volume_z"]))
    if ema_length > 0:
        entry &= data["fi_trend_ok"].gt(0)
    data["fi_signal"] = entry.astype(float)
    # 出场条件：不平衡跌破历史中位（只平多，不做空）
    data["fi_exit"] = (ratio.lt(data["fi_exit_threshold"]) & data["fi_exit_threshold"].notna()).astype(float)

    # 公用资金/过滤层 + 撮合层追踪止损都需要 atr_pct
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    close = pd.to_numeric(data["close"], errors="coerce")
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    data["atr_pct"] = atr / close.replace(0, np.nan)
    data["atr_abs"] = atr
    return data


def _prepare_atr_only(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """保证撮合层所需的 atr_pct 存在：不重复跑滚动分位（幂等短路在 prepare 入口）。"""
    spec = get_spec(CODE)
    full = prepare(frame, params, PrepareContext())
    if "atr_pct" not in full.columns:
        raise ValueError("prepare() 未产出 atr_pct")
    return full


def signals(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """只做多：信号即入场方向；不平衡转弱时给出 exit_long 供撮合软出场。"""
    ratio = column(data, "fi_ratio")
    threshold = column(data, "fi_threshold")
    upper = column(data, "fi_upper")
    atr = column(data, "atr_pct")
    close = column(data, "close")
    entry = column(data, "fi_signal").fillna(0.0).gt(0)
    exit_flag = column(data, "fi_exit").fillna(0.0).gt(0)

    valid = np.isfinite(close) & np.isfinite(atr) & close.gt(0) & atr.gt(0) & np.isfinite(ratio)
    reason = pd.Series("", index=data.index, dtype=object)
    reason = block(reason, ~valid, "invalid_flow_data")
    reason = block(reason, valid & ~entry, "flow_below_threshold")

    # 分位越极端，评分越高：把「阈值 → 历史上沿」映射成 0.6~1.0，供仓位分层。
    # 超出上沿即满分，避免极端值把仓位推到上限之外。
    span = (upper - threshold).where(np.isfinite(upper) & np.isfinite(threshold))
    strength = ((ratio - threshold) / span.replace(0, np.nan)).clip(0, 1)
    score = (0.6 + 0.4 * strength).clip(0.6, 1.0).fillna(0.6)
    signed = entry.astype(float)
    sl_pct = atr * float(params["sl_atr_mult"])
    tp_pct = sl_pct * float(params["target_r"])
    result = assemble_signal_frame(
        data, params, signed=signed, score=score,
        prob_up=pd.Series(0.5, index=data.index),
        sl_pct=sl_pct, tp_pct=tp_pct, reason=reason,
        apply_min_distances=False,
    )
    result["exit_long"] = exit_flag.astype(int)
    return result


def describe(row: pd.Series) -> dict:
    def finite(key: str) -> float | None:
        try:
            value = float(row[key])
            return value if math_isfinite(value) else None
        except (KeyError, TypeError, ValueError):
            return None
    return {"flow_ratio": finite("fi_ratio"), "flow_threshold": finite("fi_threshold"),
            "volume_z": finite("fi_volume_z")}


def math_isfinite(value: float) -> bool:
    return bool(np.isfinite(value))


SPEC = register(StrategySpec(
    code=CODE,
    name="订单流净买入极端值（只做多）",
    description=(
        "用交易所真实主动买入量计算净买入不平衡（2×主动买−总量），"
        "当其进入自身历史高分位时做多；不平衡跌破历史中位或触及 ATR 止损/目标时出场。"
        "只做多：探针显示做空侧无 edge。"
    ),
    defaults=DEFAULTS,
    ranges=RANGES,
    lookback=lookback,
    prepare=prepare,
    signals=signals,
    describe=describe,
    validate=validate,
    bool_keys=("require_positive_flow",),
    int_keys=("flow_window", "quantile_window", "trend_ema"),
    grid={
        # 探针已证明：短持有期必被成本吃掉，长持有期才可能摊薄成本；
        # 因此寻优必须同时搜索持有上限，而不是只调阈值。
        "flow_window": (16, 48, 96),
        "entry_quantile": (0.90, 0.95, 0.98),
        "max_hold_bars": (32, 64, 128),
        "target_r": (2.0, 3.0),
        "sl_atr_mult": (2.0, 3.0),
    },
    prepare_keys=PREPARE_KEYS,
    default_enabled=False,
    tags=("order_flow", "long_only"),
))
