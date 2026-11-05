"""趋势方向回测的按池聚合、显著性与报告生成。

统计口径刻意保守：

* 主统计单位是**币种**而非「笔」。同一币的 45 笔 2 日持有样本高度相关，
  把它们当独立样本会把标准误缩小好几倍；这里对「每币平均净收益」做
  cluster bootstrap（按币有放回重采样），得到 CI 与 p 值。
* 池内 15 个指标做 Benjamini–Hochberg FDR 校正，防止多重比较挑出假阳性。
* 判定只用扣成本后的净收益，毛收益仅作对照。
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype="float64")
    return array[np.isfinite(array)]


def spearman_ic(d: np.ndarray, r: np.ndarray) -> float | None:
    """逐币 Spearman 秩 IC（秩的 Pearson，不依赖 scipy）；方向无变化时返回 None。"""
    mask = np.isfinite(d) & np.isfinite(r)
    if mask.sum() < 8 or np.unique(d[mask]).size < 2:
        return None
    rank_d = pd.Series(d[mask]).rank().to_numpy()
    rank_r = pd.Series(r[mask]).rank().to_numpy()
    if rank_d.std() == 0 or rank_r.std() == 0:
        return None
    value = float(np.corrcoef(rank_d, rank_r)[0, 1])
    return value if math.isfinite(value) else None


def cluster_bootstrap(per_symbol_means: list[float], draws: int = 2000,
                      seed: int = 20261105) -> dict:
    """按币重采样，返回均值的 95% CI 与双侧 p。"""
    values = _finite(np.asarray(per_symbol_means, dtype="float64"))
    if values.size == 0:
        return {"ci_low": None, "ci_high": None, "p_value": None, "mean": None}
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, values.size, size=(draws, values.size))
    means = values[picks].mean(axis=1)
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    tail = min(float((means <= 0).mean()), float((means >= 0).mean()))
    return {
        "mean": float(values.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(max(2.0 * tail, 1.0 / draws)),
    }


def bh_qvalues(p_values: list[float]) -> list[float]:
    """Benjamini–Hochberg，None 视为 1。"""
    clean = [1.0 if p is None or not math.isfinite(p) else float(p) for p in p_values]
    n = len(clean)
    if n == 0:
        return []
    order = np.argsort(clean)
    ranked = np.asarray(clean, dtype="float64")[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(n, dtype="float64")
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return [float(x) for x in result]


def _verdict(stats: dict, q: float | None) -> str:
    low, high = stats.get("ci_low"), stats.get("ci_high")
    if low is None or high is None:
        return "insufficient"
    if q is not None and q < 0.05:
        if low > 0:
            return "positive"
        if high < 0:
            return "negative"
    return "inconclusive"


def summarize_indicator(rows: dict, name: str, seed: int) -> dict:
    """rows: {symbol: evaluate_symbol(...)}；汇总单个指标在一个池内的表现。"""
    net_all: list[np.ndarray] = []
    gross_all: list[np.ndarray] = []
    d_all: list[np.ndarray] = []
    r_all: list[np.ndarray] = []
    per_symbol_means: list[float] = []
    per_symbol_hits: list[float] = []
    ics: list[float] = []
    two_day: list[dict] = []
    current: list[float] = []
    symbols: list[str] = []
    errors: list[str] = []

    for symbol, payload in rows.items():
        rec = payload["indicators"].get(name)
        if rec is None:
            continue
        if rec.get("error"):
            errors.append(f"{symbol}: {rec['error']}")
            continue
        net = _finite(rec["net"])
        if net.size == 0:
            continue
        d = np.asarray(rec["d"], dtype="float64")
        r = _finite(rec["r"])
        net_all.append(net)
        gross_all.append(_finite(rec["gross"]))
        d_all.append(d)
        r_all.append(r)
        per_symbol_means.append(float(net.mean()))
        opened = d != 0
        if opened.any():
            per_symbol_hits.append(float(np.asarray(rec["hit"])[opened].mean()))
        ic = spearman_ic(d, r)
        if ic is not None:
            ics.append(ic)
        if rec.get("two_day"):
            item = dict(rec["two_day"])
            item["symbol"] = symbol
            two_day.append(item)
        if rec.get("current_direction") is not None:
            current.append(float(rec["current_direction"]))
        symbols.append(symbol)

    if not net_all:
        return {
            "name": name, "n_symbols": 0, "n_trades": 0,
            "direction_coverage": None, "hit_rate": None, "hit_rate_pooled": None,
            "ic_mean": None, "ic_positive_share": None,
            "mean_gross": None, "mean_net": None, "per_symbol_mean_net": None,
            "long_mean_return": None, "short_mean_return": None,
            "long_trades": 0, "short_trades": 0,
            "bootstrap": {"ci_low": None, "ci_high": None, "p_value": None, "mean": None},
            "p_value": None, "q_value": None, "verdict": "insufficient",
            "two_day": {"n_symbols": 0, "n_decided": 0, "hit_rate": None,
                        "mean_strategy_return": None, "mean_net_return": None,
                        "vote_direction": 0.0, "long_votes": 0, "short_votes": 0,
                        "neutral_votes": 0, "items": []},
            "current": {"n_symbols": 0, "net_bias": None, "long_share": None},
            "errors": errors[:5],
        }
    net = np.concatenate(net_all)
    gross = np.concatenate(gross_all)
    d = np.concatenate(d_all)
    r = np.concatenate(r_all)
    boot = cluster_bootstrap(per_symbol_means, seed=seed)
    opened = d != 0
    long_r = r[(d > 0) & np.isfinite(r)]
    short_r = -r[(d < 0) & np.isfinite(r)]
    decided = [x for x in two_day if x["direction"] != 0]
    current_nonzero = [x for x in current if x != 0]
    return {
        "name": name,
        "n_symbols": len(symbols),
        "n_trades": int(net.size),
        "direction_coverage": float(opened.mean()),
        "hit_rate": float(np.asarray([h for h in per_symbol_hits]).mean()) if per_symbol_hits else None,
        "hit_rate_pooled": float((np.sign(r[opened]) == d[opened]).mean()) if opened.any() else None,
        "ic_mean": float(np.mean(ics)) if ics else None,
        "ic_positive_share": float(np.mean([x > 0 for x in ics])) if ics else None,
        "mean_gross": float(gross.mean()),
        "mean_net": float(net.mean()),
        "per_symbol_mean_net": float(np.mean(per_symbol_means)),
        "long_mean_return": float(long_r.mean()) if long_r.size else None,
        "short_mean_return": float(short_r.mean()) if short_r.size else None,
        "long_trades": int((d > 0).sum()),
        "short_trades": int((d < 0).sum()),
        "bootstrap": boot,
        "p_value": boot["p_value"],
        "two_day": {
            "n_symbols": len(two_day),
            "n_decided": len(decided),
            "hit_rate": float(np.mean([x["hit"] for x in decided])) if decided else None,
            "mean_strategy_return": float(np.mean([x["strategy_return"] for x in two_day])) if two_day else None,
            "mean_net_return": float(np.mean([x["net_return"] for x in two_day])) if two_day else None,
            "vote_direction": float(np.sign(np.sum([x["direction"] for x in two_day]))),
            "long_votes": int(sum(x["direction"] > 0 for x in two_day)),
            "short_votes": int(sum(x["direction"] < 0 for x in two_day)),
            "neutral_votes": int(sum(x["direction"] == 0 for x in two_day)),
            "items": two_day,
        },
        "current": {
            "n_symbols": len(current),
            "net_bias": float(np.mean(current)) if current else None,
            "long_share": float(np.mean([x > 0 for x in current_nonzero])) if current_nonzero else None,
        },
        "errors": errors[:5],
    }


def aggregate_pool(code: str, members: list[str], rows: dict, names: list[str],
                   seed: int) -> dict:
    """一个池的完整结果：逐指标统计 + FDR + 排名 + 近两日池级基准。"""
    available = [s for s in members if s in rows]
    stats = [summarize_indicator({s: rows[s] for s in available}, name, seed)
             for name in names]
    q_values = bh_qvalues([s.get("p_value") or 1.0 for s in stats])
    for item, q in zip(stats, q_values):
        item["q_value"] = q
        item["verdict"] = _verdict(item.get("bootstrap", {}), q)
    ranked = sorted(
        [s for s in stats if s["n_symbols"] > 0],
        key=lambda s: s["mean_net"], reverse=True,
    )
    for rank, item in enumerate(ranked, 1):
        item["rank"] = rank
    realized = [
        rows[s]["two_day_return"] for s in available
        if rows[s].get("two_day_return") is not None
        and math.isfinite(rows[s]["two_day_return"])
    ]
    return {
        "code": code,
        "member_count": len(members),
        "evaluated_symbols": len(available),
        "skipped_symbols": [s for s in members if s not in rows],
        "two_day_market": {
            "n_symbols": len(realized),
            "mean_return": float(np.mean(realized)) if realized else None,
            "median_return": float(np.median(realized)) if realized else None,
            "up_share": float(np.mean([x > 0 for x in realized])) if realized else None,
        },
        "indicators": stats,
        "ranking": [
            {"rank": s["rank"], "name": s["name"], "mean_net": s["mean_net"],
             "verdict": s["verdict"], "q_value": s["q_value"]}
            for s in ranked
        ],
        "best": ranked[0] if ranked else None,
    }


def _fmt(value, digits: int = 4, pct: bool = False, sign: bool = True) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if not pct:
        return f"{value:.{digits}f}"
    return f"{value * 100:+.{digits - 2}f}%" if sign else f"{value * 100:.{digits - 2}f}%"


def build_markdown(payload: dict) -> str:
    cfg = payload["config"]
    lines: list[str] = [
        "# 趋势方向指标横评：1h K 线 / 近两日方向验证 / 分币种池最优",
        "",
        f"- 生成时间：{payload['generated_at']} UTC",
        f"- 数据：`klines` 表 `interval='1h'`（native 行），最后一根完整 K 线 **{payload['data']['last_complete']}**",
        f"- 指标：分类「趋势方向」共 {payload['data']['catalog_total']} 个 → 可回测 **{len(cfg['indicators'])}** 个，"
        f"另 {len(payload['excluded'])} 个因数据/输出限制剔除",
        f"- 评估窗口：{cfg['days']} 天，持有 {cfg['horizon']} 根 1h（= 两天），不重叠决策点，"
        f"双边成本 {cfg['cost'] * 100:.2f}%",
        f"- 参评币数：{payload['data']['evaluated_symbols']} / {payload['data']['target_symbols']}"
        "（池成员并集 ∩ 1h 数据可用），跳过原因："
        + (", ".join(f"{k}={v}" for k, v in payload['data']['skip_reasons'].items()) or "无"),
        "",
        "## 一、结论速览",
        "",
        "| 币种池 | 成员/参评 | 净收益最优 | 扣费后每笔 | 95% CI | BH q | 判定 | 死多头基准 | 超额(vs 死多头) | 显著为正的指标（q<0.05 且 CI>0） |",
        "| --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for pool in payload["pools"]:
        best = pool["best"]
        significant = [
            item for item in pool["indicators"]
            if item.get("verdict") == "positive" and item.get("mean_net") is not None
        ]
        significant.sort(key=lambda item: item["mean_net"], reverse=True)
        sig_text = "—"
        if significant:
            top = significant[0]
            sig_text = (f"{payload['labels'][top['name']]} "
                        f"({_fmt(top['mean_net'], 4, True)}, q={_fmt(top['q_value'], 3)})")
        bench = (payload["data"].get("benchmarks") or {}).get(pool["code"])
        bench_text = _fmt(bench["always_long_net"], 6, True) if bench else "—"
        excess_text = (_fmt(best["mean_net"] - bench["always_long_net"], 6, True)
                       if bench and best else "—")
        if not best:
            lines.append(f"| `{pool['code']}` | {pool['member_count']}/{pool['evaluated_symbols']} | — | — | — | — | 无有效样本 | {bench_text} | — | {sig_text} |")
            continue
        boot = best["bootstrap"]
        ci = f"[{_fmt(boot['ci_low'], 6, True)}, {_fmt(boot['ci_high'], 6, True)}]"
        lines.append(
            f"| `{pool['code']}` | {pool['member_count']}/{pool['evaluated_symbols']} | "
            f"{payload['labels'][best['name']]} | "
            f"{_fmt(best['mean_net'], 6, True)} | {ci} | {_fmt(best['q_value'], 4)} | "
            f"`{best['verdict']}` | {bench_text} | {excess_text} | {sig_text} |"
        )
    lines += [
        "",
        "> 判定口径：`positive`/`negative` 要求 CI 不跨 0 **且** 池内 15 指标 BH-FDR q<0.05；"
        "否则一律 `inconclusive`。排名第一只代表**相对最优**，不等于有可用 alpha。",
        "",
    ]
    baseline = next((p for p in payload["pools"] if p["code"] == "all"), payload["pools"][0] if payload["pools"] else None)
    if baseline and baseline["two_day_market"].get("mean_return") is not None:
        lines += [
            f"> **读法**：本次「近两日」窗口 `{baseline['code']}` 平均 "
            f"{_fmt(baseline['two_day_market']['mean_return'], 4, True)}、上涨币占比 "
            f"{_fmt(baseline['two_day_market']['up_share'], 4, True, False)}。普涨（或普跌）行情里，"
            "天生偏多（偏空）的指标更容易「蒙对」，单看命中率会高估能力；"
            "请以 180 天扣费净收益与 BH q 为准。净收益最优与显著为正可能不是同一个指标，上表两列都列出。",
            "",
            "> **基准**：「死多头基准」= 每个决策点无条件做多、持有两天并扣同样成本。"
            "「超额」为负说明该池所谓最优指标还不如一直做多（本次 `research_high_liquidity_high_volatility` 就是这种情况）。",
            "",
        ]
    lines += [
        "## 二、近两日方向验证（用户指定的样本外窗口）",
        "",
        f"决策点 = **{payload['data']['two_day_decision']}**（两天前那根 1h 收盘），"
        f"入场 = 下一根开盘，出场 = {payload['data']['last_complete']} 收盘；"
        "所有指标都只用决策点及之前的数据计算方向。",
        "",
        "| 币种池 | 实际两日平均涨跌 | 上涨币占比 | 最优指标近两日命中率 | 该指标两日净收益 | 该指标当时方向票数(多/空/中性) |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for pool in payload["pools"]:
        market = pool["two_day_market"]
        best = pool["best"]
        td = best["two_day"] if best else {}
        votes = f"{td.get('long_votes', '—')}/{td.get('short_votes', '—')}/{td.get('neutral_votes', '—')}"
        lines.append(
            f"| `{pool['code']}` | {_fmt(market['mean_return'], 4, True)} | "
            f"{_fmt(market['up_share'], 4, True, False)} | {_fmt(td.get('hit_rate'), 4, True, False)} | "
            f"{_fmt(td.get('mean_net_return'), 4, True)} | {votes} |"
        )
    lines += [
        "",
        "## 三、分池回测明细（按扣费净收益降序）",
        "",
    ]
    for pool in payload["pools"]:
        lines += [f"### `{pool['code']}`（{pool['evaluated_symbols']} 币）", "",
                  "| # | 指标 | 笔数 | 方向覆盖 | 命中率 | 平均IC | 毛收益/笔 | 净收益/笔 | CI | q | 判定 |",
                  "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |"]
        for item in sorted(pool["indicators"], key=lambda x: x.get("rank", 999)):
            boot = item.get("bootstrap", {})
            ci = f"[{_fmt(boot.get('ci_low'), 6, True)},{_fmt(boot.get('ci_high'), 6, True)}]"
            lines.append(
                f"| {item.get('rank', '—')} | {payload['labels'][item['name']]} | "
                f"{item.get('n_trades', 0)} | {_fmt(item.get('direction_coverage'), 4, True, False)} | "
                f"{_fmt(item.get('hit_rate'), 4, True, False)} | {_fmt(item.get('ic_mean'), 4)} | "
                f"{_fmt(item.get('mean_gross'), 6, True)} | {_fmt(item.get('mean_net'), 6, True)} | "
                f"{ci} | {_fmt(item.get('q_value'), 3)} | `{item.get('verdict')}` |"
            )
        lines.append("")
    lines += [
        "## 四、指标方向契约",
        "",
        "| 指标 | 来源 | 方向列 / 处理 | 说明 |",
        "| --- | --- | --- | --- |",
    ]
    for spec in payload["contracts"]:
        lines.append(f"| `{spec['name']}` ({spec['label']}) | {spec['source']} | "
                     f"`{spec['columns']}` / {spec['mode']} | {spec['note']} |")
    lines += ["", "**未参评（分类内但无法在单币 1h OHLCV 上回测）**：", ""]
    for name, reason in payload["excluded"].items():
        lines.append(f"- `{name}`：{reason}")
    lines += [
        "",
        "## 五、方法与局限",
        "",
        "1. **无前视**：方向取 bar t 收盘；收益用 `close[t+H]/open[t+1]-1`，t+1 开盘才成交。",
        "2. **不重叠采样**：历史评估的决策点间隔 = H（48 根），避免 48 倍重叠样本伪造显著性；"
        "最后一个决策点恒为「两天前」，正好覆盖最近两日。",
        "3. **显著性按币聚类**：对每币平均净收益做 2000 次有放回 bootstrap，池内再对 15 指标做 BH-FDR。",
        "4. **成本**：每笔往返双边 0.12%；方向为 0（中性）不开仓、不扣成本。",
        "5. 这些移植指标本质是**描述当前趋势状态**，不是概率预测器；本回测检验的是"
        "「按当前趋势方向持有两天」是否有 edge，指标原作者的入场/止损/过滤条件未参与。",
        "6. 池成员按 `symbol_pools` 规则 ∩ 1h 数据可用性解析；数据新鲜度不足的币被剔除。",
        "7. 1h 数据来自 native 行；项目已验证 native 与 1m 聚合口径在重叠区间零差异"
        "（`docs/native-consistency.md`）。",
        "",
        f"复现命令：`{payload['config']['command']}`",
        "",
    ]
    return "\n".join(lines)


def write_reports(payload: dict, prefix: Path) -> tuple[Path, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    return md_path, json_path
