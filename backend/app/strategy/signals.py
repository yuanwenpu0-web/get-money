"""信号帧的共用拼装：把「策略自己的判断」补成统一可执行的信号列。

策略只负责回答四件事：方向（signed）、强度（score）、止损/止盈距离（sl_pct / tp_pct）、
以及为什么不做（block_reason）。剩下的全部在这里统一处理，保证模拟盘、巡检、
回测三条路径用的是同一套成本、杠杆与仓位口径：

- 双边成本与净盈亏比过滤（`cost_filter`）
- 波动率倒数定杠杆（ATR 越小杠杆越高，受 `max_leverage` 夹逼）
- 评分分层的仓位占比（`base_position_pct` → `max_position_pct`）
- `risk_per_trade_pct` 的风险预算截断（极端宽止损时自动缩仓）
- 方向清零（有拦截原因或仓位为 0 → `direction=0`）

新增策略不需要复制这段逻辑，调用 `assemble_signal_frame()` 即可。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIGNAL_COLUMNS = (
    "signed",
    "score",
    "prob_up",
    "direction",
    "block_reason",
    "position_pct",
    "leverage",
    "sl_pct",
    "tp_pct",
    "risk_pct",
    "net_reward_risk",
)

# 杠杆公式的波动率基准：ATR/价格 = 1.5% 时给 1 倍基准强度，随 ATR 缩小而放大。
LEVERAGE_ATR_REFERENCE = 0.015


def column(data: pd.DataFrame, name: str, fill: float = np.nan) -> pd.Series:
    """按列名取数值列；缺失或非数值一律当 NaN，避免策略里到处写 try/except。"""
    if name not in data.columns:
        return pd.Series(fill, index=data.index, dtype=float)
    return pd.to_numeric(data[name], errors="coerce")


def block(reason: pd.Series, mask: pd.Series, label: str) -> pd.Series:
    """只给「还没有原因」的行打标签：先到先得，保持与 Pine 脚本的判定顺序一致。"""
    hit = mask.reindex(reason.index).fillna(True).astype(bool) & reason.eq("")
    return reason.mask(hit, label)


def assemble_signal_frame(
    data: pd.DataFrame,
    params: dict,
    *,
    signed: pd.Series,
    score: pd.Series,
    prob_up: pd.Series,
    sl_pct: pd.Series,
    tp_pct: pd.Series,
    reason: pd.Series,
    apply_min_distances: bool = True,
) -> pd.DataFrame:
    """输入策略判断，输出 `SIGNAL_COLUMNS` 定义的可执行信号帧。"""
    index = data.index
    signed = pd.to_numeric(signed, errors="coerce").fillna(0.0)
    score = pd.to_numeric(score, errors="coerce").fillna(0.5).clip(0.5, 1.0)
    prob_up = pd.to_numeric(prob_up, errors="coerce").fillna(0.5)
    reason = reason.reindex(index).fillna("").astype(object)

    cost = 2 * (float(params["fee_rate"]) + float(params["slippage"]))
    sl_pct = pd.to_numeric(sl_pct, errors="coerce")
    tp_pct = pd.to_numeric(tp_pct, errors="coerce")
    if apply_min_distances:
        sl_pct = sl_pct.clip(lower=float(params["min_sl_pct"])).fillna(float(params["min_sl_pct"]))
        tp_pct = tp_pct.clip(lower=float(params["min_tp_pct"])).fillna(float(params["min_tp_pct"]))
    else:
        # Strategies whose price levels are explicitly defined must not have
        # their ATR distances silently widened by generic minimums.
        valid = np.isfinite(sl_pct) & np.isfinite(tp_pct) & sl_pct.gt(0) & tp_pct.gt(0)
        reason = block(reason, ~valid, "invalid_distance")
        sl_pct = sl_pct.where(valid, 0.0)
        tp_pct = tp_pct.where(valid, 0.0)
    net_rr = (tp_pct - cost) / (sl_pct + cost)

    atr = column(data, "atr_pct")
    if params.get("cost_filter", True):
        too_small = atr.lt(cost * float(params["min_atr_cost_ratio"]))
        bad_rr = net_rr.lt(float(params["min_net_reward_risk"]))
        reason = block(reason, too_small | bad_rr, "cost_filter")

    side = np.sign(signed)

    confidence = ((score - 0.5) / 0.5).clip(0, 1)
    leverage = (
        (LEVERAGE_ATR_REFERENCE / atr.clip(lower=0.0001) * (0.6 + 0.4 * confidence))
        .round()
        .clip(1, int(params["max_leverage"]))
        .fillna(1)
        .astype(int)
    )

    base_pct = float(params["base_position_pct"])
    max_pct = float(params["max_position_pct"])
    min_score = float(params["min_score"])
    heavy_score = float(params["heavy_score"])
    span = max(heavy_score - min_score, 1e-6)
    fraction = ((score - min_score) / span).clip(0, 1)
    fraction = base_pct + fraction * (max_pct - base_pct)
    fraction = fraction.where(score < heavy_score, max_pct)
    fraction = np.minimum(
        fraction, float(params["risk_per_trade_pct"]) / (leverage * (sl_pct + cost))
    )
    fraction = fraction.where(reason.eq(""), 0.0).fillna(0.0)

    direction = side.where(reason.eq("") & fraction.gt(0), 0).fillna(0).astype(int)
    return pd.DataFrame(
        {
            "signed": signed.fillna(0),
            "score": score.fillna(0.5),
            "prob_up": prob_up.fillna(0.5),
            "direction": direction,
            "block_reason": reason,
            "position_pct": fraction,
            "leverage": leverage.where(direction.ne(0), 0),
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "risk_pct": (fraction * leverage * (sl_pct + cost)).fillna(0.0),
            "net_reward_risk": net_rr,
        },
        index=index,
    )
