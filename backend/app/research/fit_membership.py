"""P18 币种池成员的时点（PIT）口径核验。

为什么需要这一层
----------------
`fit_prep.prepare_scope` 用**整个评估窗**的月度成交额中位数给每个币定一个固定档
（`fit_data.assign_fixed_tiers`），再把这批成员写进 `symbol_pools` 供统计阶段使用。
固定档是必要的——事件缓存按币聚合，统计阶段拿不到「这个事件发生在哪个月」——但它
用的是窗口内**全部**月份，包含事件发生之后的月份。对 8 月的事件来说，9/10 月的成交额
是未来信息。预登记（`docs/indicator-evaluation-plan.md` 第 5.1 节）要求分组用事件时点
可得的属性，所以这一层负责把「事后固定档」与「时点可得档」的差距量化出来。

本模块只做测量，不做选择
------------------------
- 不重算任何收益/统计，不改 A 轮标签，不重选指标/周期/池/h；
- PIT 档只用**严格早于事件月**且**完整**的月份，绝不读事件月自身或之后的月；
- 没有可用 prior 月时显式返回 `NO_PRIOR_COMPLETE_MONTH`，而不是拿未来月补齐。
"""
from __future__ import annotations

import pandas as pd

from . import fit_data as fd

NO_PRIOR = "NO_PRIOR_COMPLETE_MONTH"
OK = "PIT_AVAILABLE"


def _median_tiers(frame: pd.DataFrame, groups: int) -> dict[str, str]:
    """在给定币种集内部按成交额中位数切固定档，复用 A 轮的同一套口径。"""
    return fd.assign_fixed_tiers(frame, groups=groups)


def pit_tiers_by_month(frame: pd.DataFrame, groups: int = 4) -> dict[str, dict]:
    """逐事件月给出「只用更早完整月」能定出的档位。

    入参是 `fit_data.monthly_quote_volume` / `mark_partial_months` 的结果，必须带
    `ym / symbol / quote_volume / partial`。返回 {ym: {...}}，每项含：
    - status：`PIT_AVAILABLE` 或 `NO_PRIOR_COMPLETE_MONTH`
    - prior_months：实际用到的月份（严格早于事件月、且非 partial）
    - tiers：{symbol: 'Q1'..}，只覆盖在 prior 月里有成交额的币
    - unavailable：事件月出现、但 prior 月无量的币（新上市币）——它们无法 PIT 定档
    """
    if frame.empty:
        return {}
    need = {"ym", "symbol", "quote_volume", "partial"}
    missing = need - set(frame.columns)
    if missing:
        raise ValueError(f"pit_tiers_by_month 缺少列: {sorted(missing)}")
    complete = frame[~frame["partial"].astype(bool)]
    months = sorted(frame["ym"].astype(str).unique())
    out: dict[str, dict] = {}
    for ym in months:
        prior_months = [m for m in months if m < ym
                        and m in set(complete["ym"].astype(str))]
        prior = complete[complete["ym"].astype(str).isin(prior_months)]
        event_symbols = set(frame.loc[frame["ym"].astype(str) == ym, "symbol"].astype(str))
        if prior.empty:
            out[ym] = {"status": NO_PRIOR, "prior_months": [], "tiers": {},
                       "unavailable": sorted(event_symbols)}
            continue
        tiers = _median_tiers(prior, groups)
        out[ym] = {
            "status": OK, "prior_months": prior_months, "tiers": tiers,
            "unavailable": sorted(event_symbols - set(tiers)),
        }
    return out


def tier_disagreement(fixed: dict[str, str], pit: dict[str, dict]) -> dict:
    """固定档 vs PIT 档的逐月不一致率。

    只比「两边都有档」的币：PIT 无从定档的币单独计数，不能算成一致，也不能算成不一致。
    """
    per_month: dict[str, dict] = {}
    for ym, blk in sorted(pit.items()):
        tiers = blk.get("tiers") or {}
        both = sorted(set(tiers) & set(fixed))
        diff = [s for s in both if tiers[s] != fixed[s]]
        per_month[ym] = {
            "status": blk.get("status"),
            "prior_months": blk.get("prior_months", []),
            "comparable": len(both),
            "disagree": len(diff),
            "disagree_rate": round(len(diff) / len(both), 4) if both else None,
            "no_pit_tier": len(blk.get("unavailable") or []),
            "disagree_symbols": diff,
        }
    comparable = sum(m["comparable"] for m in per_month.values())
    disagree = sum(m["disagree"] for m in per_month.values())
    return {
        "per_month": per_month,
        "comparable_total": comparable,
        "disagree_total": disagree,
        "disagree_rate": round(disagree / comparable, 4) if comparable else None,
        "months_without_pit": [ym for ym, m in per_month.items() if m["status"] == NO_PRIOR],
    }


def membership_drift(frozen: dict[str, list[str]], live: dict[str, list[str]]) -> dict[str, dict]:
    """逐池比成员**集合**，不是比数量。

    数量相同但换了一个币也是漂移：A 轮统计用的篮子必须能原样复现，否则报告里的
    「major 池」和重跑时的「major 池」不是同一批币。
    """
    out: dict[str, dict] = {}
    for code in sorted(set(frozen) | set(live)):
        was = set(frozen.get(code) or [])
        now = set(live.get(code) or [])
        if code not in live:
            status = "MISSING_NOW"
        elif code not in frozen:
            status = "NEW_NOW"
        elif was == now:
            status = "MATCH"
        else:
            status = "DRIFT"
        out[code] = {
            "status": status,
            "frozen_count": len(was), "live_count": len(now),
            "added": sorted(now - was), "removed": sorted(was - now),
        }
    return out