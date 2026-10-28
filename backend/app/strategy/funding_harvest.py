"""资金费率横截面收割（delta 中性）选币模块。

P11 探针（scripts/probe_funding_harvest.py）的产品化提炼：做空 trailing 费率
最高的 K 个永续 + 买现货对冲，收资金费。本模块只产出"目标持仓建议"，
不下单、不管理真实仓位——执行与风险控制在上层。

关键口径（勿改，探针 gate 结论依赖这些常数）：
- funding_time 带 ±ms 偏移，必须 floor 到 8h 桶再 pivot（桶内 sum=该 8h 费率收入），
  否则行数滚动 ≠ 日历窗口。
- trailing 窗口 = 过去 7 天均费率（21 个 8h 桶）；持有/重平衡周期 = 3 天（9 桶）。
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models import FundingRate

BUCKET = "8h"
TRAIL_BUCKETS = 21  # 7 天
HOLD_BUCKETS = 9  # 3 天


def funding_grid(db: Session, days: int = 30) -> pd.DataFrame:
    """近 N 天资金费 → 8h 桶 × symbol 矩阵（桶内 sum = 该 8h 实际费率收入）。"""
    last_ms = db.execute(select(func.max(FundingRate.funding_time))).scalar()
    if last_ms is None:
        return pd.DataFrame()
    start = pd.to_datetime(last_ms, unit="ms", utc=True) - pd.Timedelta(days=days)
    rows = db.execute(
        select(FundingRate.symbol, FundingRate.funding_time, FundingRate.funding_rate).where(
            FundingRate.funding_time >= int(start.timestamp() * 1000)
        )
    ).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["symbol", "time", "rate"])
    df["bucket"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.floor(BUCKET)
    grid = df.pivot_table(index="bucket", columns="symbol", values="rate", aggfunc="sum")
    return grid.sort_index()


def select_from_grid(
    grid: pd.DataFrame, k: int = 10, min_rate: float = 0.0,
    eligible: set[str] | None = None,
) -> list[dict]:
    """核心选币（纯函数）：trailing 均费率最高且 > min_rate 的 K 币。

    eligible：可选白名单（如现货对过滤后的集合）；None 表示不过滤。
    """
    if grid is None or grid.empty:
        return []
    trail = grid.rolling(TRAIL_BUCKETS, min_periods=TRAIL_BUCKETS // 2).mean().iloc[-1].dropna()
    trail = trail[trail > min_rate]
    if eligible is not None:
        trail = trail[trail.index.isin(eligible)]
    out: list[dict] = []
    for symbol, rate in trail.sort_values(ascending=False).head(k).items():
        out.append({
            "symbol": str(symbol),
            "trail_rate_8h": round(float(rate), 6),
            "annualized_pct": round(float(rate) * 3 * 365 * 100, 2),
        })
    return out


def select_targets(db: Session, k: int = 10, min_rate: float = 0.0,
                   eligible: set[str] | None = None) -> list[dict]:
    """从库里选当前收割目标（DB 包装层）。"""
    return select_from_grid(funding_grid(db), k=k, min_rate=min_rate, eligible=eligible)


def turnover(prev: list[str] | None, cur: list[str]) -> float:
    """名次换手率（Jaccard 距离）：1 − |交集| / |并集|。首次建仓 = 1。"""
    prev_set = set(prev or [])
    cur_set = set(cur)
    union = prev_set | cur_set
    if not union:
        return 0.0
    return 1.0 - len(prev_set & cur_set) / len(union)


# 每名次开/平双腿成本：现货腿 + 永续腿，各 2 边（开+平）；fee/slippage 与回测全局一致
COST_PER_NAME = 2 * 2 * (0.0004 + 0.0002)  # 0.0024


def rebalance_cost(prev: list[str] | None, cur: list[str]) -> float:
    """本次重平衡的组合成本（占组合名义比例）。"""
    return COST_PER_NAME * turnover(prev, cur)


def match_spot_base(symbol: str, spot_bases: set[str]) -> str | None:
    """永续 symbol → 现货 baseAsset（支持 1000X 前缀换算），无可对冲现货返回 None。

    例：1000PEPEUSDT → PEPE（现货 PEPEUSDT）；BROCCOLIF3BUSDT → None（纯永续）。
    """
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    if base in spot_bases:
        return base
    if base.startswith("1000") and base[4:] in spot_bases:
        return base[4:]
    return None


def spot_base_symbols(proxy: str | None = None, timeout: int = 15) -> set[str]:
    """Binance 现货 exchangeInfo 中 TRADING 状态的 USDT 对的 baseAsset 集合。

    永续 symbol（如 BROCCOLIF3BUSDT）去掉 USDT 后与现货 base 对比；
    注意 1000X 系前缀可能与现货 base 不一致，匹配不上会被过滤掉（宁可漏选）。
    """
    import requests

    proxies = {"http": proxy, "https": proxy} if proxy else None
    resp = requests.get(
        "https://api.binance.com/api/v3/exchangeInfo", timeout=timeout, proxies=proxies,
    )
    resp.raise_for_status()
    out: set[str] = set()
    for s in resp.json().get("symbols", []):
        if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING":
            out.add(s.get("baseAsset", ""))
    out.discard("")
    return out
