"""P17 / H-E 臂构造与汇总 —— 纯函数，不碰 DB、不重新撮合。

铁律：**所有臂共用同一份逐币成交**。`build_arms` 只做子集选择，绝不调用
`simulate`。否则 `posthoc − pit` 的差里会混进撮合差异，第 5 节的泄漏诊断就废了。

选币口径
--------
排名先按 `meta["eligible"]`（覆盖率过滤后的回测宇宙）裁剪，再取 TopN。
这样「固定规模」才真的是固定规模：不会因为某个币当期切片太短就少一个成员。
宇宙过滤对**所有臂一视同仁**（含 `all` 基线），所以臂间比较是公平的；
但它本身用了全窗口覆盖信息，属于绝对可交易性声明的削弱项，须在报告局限里写明。
"""
from __future__ import annotations

import statistics

import numpy as np

from ..backtest.significance import bootstrap_expectancy_ci, percentile_rank
from .pit_universe import select


def build_arms(fold_meta: list[dict], results: list[dict], sizes: tuple, seeds: int) -> dict:
    """成员构成是唯一变量：所有臂从同一份逐币成交里取子集。"""
    by_symbol = {r["symbol"]: {f["index"]: f for f in r["folds"]}
                 for r in results if r["status"] == "ok"}
    arms: dict[str, dict] = {}
    for meta in fold_meta:
        index = meta["index"]
        per = {s: by_symbol[s][index] for s in meta["eligible"]
               if s in by_symbol and index in by_symbol[s]}
        mean_ret = {s: (statistics.fmean([t[0] for t in per[s]["trades"]])
                        if per[s]["trades"] else None) for s in per}
        post = sorted([s for s in per if mean_ret[s] is not None],
                      key=lambda s: (-mean_ret[s], s))
        ranked = [item for item in meta["ranked"] if item["symbol"] in per]
        members = {"all": sorted(per)}
        for n in sizes:
            members[f"pit_liq_{n}"] = select(ranked, "liquidity", n)
            members[f"pit_vol_{n}"] = select(ranked, "volatility", n)
            members[f"posthoc_{n}"] = post[:n]
        for name, mem in members.items():
            arm = arms.setdefault(name, {"folds": [], "trades": [], "acct": [],
                                         "rnd": [[] for _ in range(seeds)]})
            # 聚合按 symbol 字典序，不按排名序：成员集合相同而排序不同的两个臂
            # （15m 只有 30 币时的 liq_30 / vol_30）必须给出逐位一致的 bootstrap CI。
            ordered = sorted(mem)
            trades = [t for s in ordered for t in per[s]["trades"]]
            arm["trades"].extend(trades)
            arm["folds"].append({
                "index": index, "start": meta["start_iso"], "end": meta["end_iso"],
                "members": mem, "requested": len(mem) if name == "all" else int(name.split("_")[-1]),
                "trades": len(trades),
                "mean": round(statistics.fmean([t[0] for t in trades]), 4) if trades else None,
            })
            for s in ordered:
                arm["acct"].append(per[s]["account_return_pct"])
                for k in range(seeds):
                    arm["rnd"][k].append(per[s]["random_samples"][k])
    return arms


def summarise(arms: dict, min_trades: int = 30) -> dict:
    """池化 CI + 多空分解 + 逐 fold 均值 + 组合账户收益的随机分位。"""
    out = {}
    for name, arm in arms.items():
        ci = bootstrap_expectancy_ci([{"return_pct": v} for v, _ in arm["trades"]],
                                     min_trades=min_trades)
        means = [f["mean"] for f in arm["folds"] if f["mean"] is not None]
        strategy = statistics.fmean(arm["acct"]) if arm["acct"] else 0.0
        samples = [statistics.fmean(col) for col in arm["rnd"] if col]
        longs = [v for v, side in arm["trades"] if side == "long"]
        shorts = [v for v, side in arm["trades"] if side == "short"]
        out[name] = {
            "pooled": ci,
            "side_split": {
                "long": bootstrap_expectancy_ci([{"return_pct": v} for v in longs],
                                                min_trades=min_trades),
                "short": bootstrap_expectancy_ci([{"return_pct": v} for v in shorts],
                                                 min_trades=min_trades),
            },
            "folds": arm["folds"], "fold_means": means,
            "positive_folds": sum(1 for m in means if m > 0),
            "account_mean_pct": round(strategy, 4),
            "random_mean_pct": round(statistics.fmean(samples), 4) if samples else None,
            "random_p95_pct": round(float(np.percentile(samples, 95)), 4) if samples else None,
            "percentile_rank": round(percentile_rank(strategy, samples), 4) if samples else None,
            "symbols_unique": sorted({s for f in arm["folds"] for s in f["members"]}),
        }
    return out