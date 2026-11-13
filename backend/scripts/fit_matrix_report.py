"""把 S3–S5 的统计产物渲染成成对交付：`docs/indicator-fit-matrix.md` + 同名 `.json`。

为什么要两份：md 是给人读的结论，json 是这些结论的数字底稿。两份由**同一次运行**从
`docs/indicator-fit-stats-<scope>.json` 算出，md 里出现的每个数都能在 json 里找到同名字段
——所以 md 不许手改数字，要改就改口径再重跑。页面（`fit_matrix_page.py`）读同一份 stats，
三者可互相核对。

用法：
    .venv\\Scripts\\python.exe backend/scripts/fit_matrix_report.py --scope A
    # 自检（单周期调试产物，不覆盖正式交付）：
    ... --stats docs/indicator-fit-stats-A-dev.json --suffix -dev
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
sys.path.insert(0, str(ROOT / "backend"))

from app.research.fit_stats import (  # noqa: E402
    BENCH_KEYS, PROMOTABLE, VERDICT_RANK, best_cell, intervals_sorted, level,
)

SCOPE_ORDER = ["all", "common", "major", "alt", "liq_q1", "liq_q2", "liq_q3", "liq_q4"]
# 台账编号写死，不做「扫最大号 +1」：experiment_id 会被 OPEN-DECISIONS、verdicts-history
# 之类的文档引用，自动递增会让同一份引用在某次运行之后指向别的实验。
LEDGER_EXPERIMENT_ID = "P18-001"
LEDGER_PATH = ROOT / "research" / "experiments.jsonl"
SCOPE_LABEL = {
    "all": "全样本", "common": "四周期交集", "major": "主流", "alt": "山寨",
    "liq_q1": "Q1 最高流动", "liq_q2": "Q2", "liq_q3": "Q3", "liq_q4": "Q4 最低流动",
}
# 周期次序与判定词表都只在 fit_stats 里定义；这里派生展示用的东西，别在报告里另写一份。
VERDICT_ORDER = sorted(VERDICT_RANK, key=lambda v: VERDICT_RANK[v])


def load(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_interval(stats: dict) -> dict[str, list[dict]]:
    return {iv: (v or {}).get("rows") or [] for iv, v in (stats.get("intervals") or {}).items()}


def cluster_count(pairs: list[dict]) -> tuple[int, int, list[dict]]:
    """冗余对 → 并查集堆数。返回 (参与归堆的列数, 堆数, 最大的几个堆)。"""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        a, b = find(p["a"]), find(p["b"])
        if a != b:
            parent[a] = b
    groups: dict[str, list[str]] = {}
    for k in parent:
        groups.setdefault(find(k), []).append(k)
    big = sorted(({"size": len(v), "members": sorted(v)[:8]} for v in groups.values()),
                 key=lambda g: -g["size"])[:6]
    return len(parent), len(groups), big


def summarize(stats: dict, dirs: dict, events: dict) -> dict:
    """报告的全部数字都在这一个函数里算——md 只是它的排版，不引入新口径。"""
    intervals = intervals_sorted(stats.get("intervals") or {})
    rows_iv = rows_by_interval(stats)
    per_interval: dict[str, dict] = {}
    all_rows: list[dict] = []
    for iv in intervals:
        rows = rows_iv[iv]
        for r in rows:
            r["interval"] = iv  # 行本身不带周期，周期是外层键；不补上就没法回答「适合哪个 K 线」
        all_rows += rows
        counts = (stats["intervals"][iv] or {}).get("verdict_counts") or {}
        red = (stats["intervals"][iv] or {}).get("redundancy") or {}
        pairs = [p for lst in red.values() for p in lst]
        n_in, n_groups, big = cluster_count(pairs)
        cells = {}
        for r in rows:
            cells.setdefault(r["key"], []).append(r)
        promotable = [k for k, rs in cells.items()
                      if any((r.get("verdict_cell") or r["verdict"]) in PROMOTABLE for r in rs)]
        # 「格」= 池 × 预测长度 × 基准。池数从**全量**判定分布取（裁剪后的行会漏池），
        # 基准从行里观测到的取值数取（口径只跑一个基准时不许硬按两个算）。
        n_bench = len({r.get("benchmark") for r in rows if r.get("benchmark")}) or len(BENCH_KEYS)
        n_cells = max(len(counts) * len((stats.get("horizons_bars") or [0])) * n_bench, 1)
        per_interval[iv] = {
            "columns": (stats["intervals"][iv] or {}).get("columns"),
            # 「声明的列」和「真进统计的列」是两个数：后者扣掉了触发率退化的列，
            # 两者混在一列里会让「每格行数」看起来对不上（1h：885 声明 / 737 参评）。
            "columns_evaluated": len({r["key"] for r in rows}),
            "symbols": (stats["intervals"][iv] or {}).get("symbols"),
            "scope_sizes": (stats["intervals"][iv] or {}).get("scope_sizes") or {},
            "rows_evaluated": (stats["intervals"][iv] or {}).get("rows_total"),
            "cells": n_cells,
            "rows_kept": (stats["intervals"][iv] or {}).get("rows_kept"),
            "verdict_counts_all": counts.get("all") or {},
            "verdict_counts_by_scope": counts,
            "columns_promotable": len(promotable),
            "redundancy_pairs": len(pairs),
            "redundancy_members": n_in,
            "redundancy_clusters": n_groups,
            "redundancy_largest": big,
        }

    # 每列的「最强格」：跨周期跨池取最强——这就是「适合哪个周期 / 哪个池」的答案本体。
    by_key: dict[str, list[dict]] = {}
    for r in all_rows:
        by_key.setdefault(r["key"], []).append(r)
    best_rows = []
    for key, rs in by_key.items():
        b = best_cell(rs)
        if b is None:
            continue
        b = dict(b)
        b["key"] = key
        # 同列在 scope=all 的最强格：用来判断「池级结论是不是只是整体结论的复读」
        pooled = best_cell([r for r in rs if r["scope"] == "all"])
        if pooled:
            b["pooled_verdict"] = pooled["verdict"]
            b["pooled_mean_net_pct"] = pooled.get("mean_net_pct")
            b["pooled_ci_low_pct"] = pooled.get("ci_low_pct")
            b["pooled_events"] = pooled.get("events")
        best_rows.append(b)
    best_rows.sort(key=lambda r: (level(r), -(r.get("ci_low_pct") or -9e9)))

    promotable_best = [r for r in best_rows if (r.get("verdict_cell") or r["verdict"]) in PROMOTABLE]
    by_interval: dict[str, int] = {}
    by_scope: dict[str, int] = {}
    for r in promotable_best:
        by_interval[r["interval"]] = by_interval.get(r["interval"], 0) + 1
        by_scope[r["scope"]] = by_scope.get(r["scope"], 0) + 1
    # 「适合这个池」的硬定义：池级格子自己拿到 candidate（内含交互检验 p<α 与收缩），
    # 而同一列在整体池拿不到 candidate。少了后半句，「适合 Q3」就只是把「整体有效」复读
    # 一遍，池这个维度没有增加任何信息。
    pool_specific = [r for r in promotable_best
                     if r["scope"] != "all"
                     and (r.get("verdict_cell") or "") == "candidate"
                     and r.get("pooled_verdict") != "candidate"]
    pool_by_scope: dict[str, int] = {}
    for r in pool_specific:
        pool_by_scope[r["scope"]] = pool_by_scope.get(r["scope"], 0) + 1
    cand_best = [r for r in promotable_best
                 if (r.get("verdict_cell") or r["verdict"]) == "candidate"]
    # 指标级汇总：「这一族指标适合哪个 K 线」只能按「该指标说得出口的列落在哪个周期」数出来。
    # 判定等级数字越小越强（VERDICT_RANK 的本义），这里取每族最强的一格当代表。
    rollup: dict[str, dict] = {}
    for r in promotable_best:
        ind = r.get("indicator") or (r.get("key") or "").split("|")[0]
        e = rollup.setdefault(ind, {"indicator": ind, "columns": 0, "by_interval": {},
                                   "by_scope": {}, "best_verdict": ""})
        e["columns"] += 1
        for field, value in (("by_interval", r.get("interval")), ("by_scope", r.get("scope"))):
            e[field][value] = e[field].get(value, 0) + 1
        v = r.get("verdict_cell") or r.get("verdict")
        if VERDICT_RANK.get(v, 99) < VERDICT_RANK.get(e["best_verdict"], 99):
            e["best_verdict"] = v
    indicator_rollup = sorted(rollup.values(), key=lambda e: (-e["columns"], e["indicator"]))
    acc = dirs.get("indicator_accounting") or {}
    skipped_inds = (int(acc.get("registered") or 0) - int(acc.get("with_boolean_columns") or 0)
                    or None)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": stats.get("scope"), "sha": stats.get("sha"),
        "inference_status": "EXPLORATORY_ONLY",
        "audit_report": "docs/indicator-fit-readiness.json",
        "stats_generated_at": stats.get("generated_at"),
        "horizons_bars": stats.get("horizons_bars"),
        "config": stats.get("config"),
        "directions_sha": events.get("directions_sha"),
        "directions_count": dirs.get("count"),
        "directions_by_source": dirs.get("by_source"),
        "directions_by_direction": dirs.get("by_direction"),
        # 指标注册账本（方向表枚举时结算）：报告里「跳过多少个注册项」只能引用这里。
        # 探针产物 run_stats.skipped 是**单币**计数，拿来当全量口径就是假数。
        "indicator_accounting": acc,
        "indicators_skipped": skipped_inds,
        "intervals": intervals,
        "per_interval": per_interval,
        "rows_evaluated_total": sum((stats["intervals"][iv] or {}).get("rows_total") or 0
                                    for iv in intervals),
        "columns_declared": dirs.get("count"),
        "columns_evaluated": len(by_key),
        "columns_degenerate_excluded": (dirs.get("count") or 0) - len(by_key),
        "columns_total": len(by_key),
        "columns_promotable": len(promotable_best),
        "indicators_promotable": len(indicator_rollup),
        "columns_candidate": len(cand_best),
        "promotable_by_interval": by_interval,
        "promotable_by_scope": by_scope,
        "pool_specific_columns": len(pool_specific),
        "pool_specific_by_scope": pool_by_scope,
        "candidate_cells": [_trim(r) for r in cand_best],
        "pool_specific_cells": [_trim(r) for r in pool_specific],
        "indicator_rollup": indicator_rollup,
        "promotable_cells": [_trim(r) for r in promotable_best],
        "best_cells": [_trim(r) for r in best_rows],
        "verdict_order": VERDICT_ORDER,
        "benchmarks": list(BENCH_KEYS),
        "promotable_labels": list(PROMOTABLE),
    }


ROW_FIELDS = (
    "key", "indicator", "column", "interval", "scope", "horizon_bars", "benchmark",
    "direction", "direction_source", "events", "symbols_with_events", "symbols_in_stat",
    "mean_gross_pct", "mean_net_pct", "ci_low_pct", "ci_high_pct", "q_value", "p_bootstrap",
    "seg_consistency", "seg_valid", "symbol_sign_rate", "cost_stress_pct", "cost_ratio",
    "shrunk_net_pct", "shrink_weight", "interaction", "hit_rate", "win_rate", "payoff_ratio",
    "cs_coverage", "verdict", "verdict_cell", "pooled_verdict", "pooled_mean_net_pct",
    "pooled_ci_low_pct", "pooled_events",
)


def _trim(r: dict) -> dict:
    """只留报告要用的字段：整行 30 多个键里，`direction_matched` 这类长数组对读者没信息量。"""
    return {k: r.get(k) for k in ROW_FIELDS if k in r}


def pct(v, digits: int = 2) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def table(rows: list[list[str]], head: list[str]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def size_text(s: dict, ivs: list[str], sc: str) -> str:
    """把某个池在各周期下的合格币数拼成一句（池成员是按周期各自过数据门的，不是全局一个数）。"""
    parts = []
    for iv in ivs:
        n = (s["per_interval"][iv]["scope_sizes"] or {}).get(sc)
        if n is not None:
            parts.append(f"{iv} {n}")
    return " / ".join(parts) if parts else "—"


def render_md(s: dict, events: dict, probe: dict, quality: dict, prep: dict) -> str:
    """把 summarize() 的字典排成 markdown。这里只允许「取数 + 排版」，不许新增口径。"""
    cfg = s.get("config") or {}
    r_min = cfg.get("redundancy_r_min")
    if r_min is None:
        # 老产物里没写阈值：宁可让读者看到「未记录」，也不要在报告里补一个猜测值。
        r_min = "未记录（该轮产物没写阈值）"
    ivs = s["intervals"]
    L: list[str] = []
    A = L.append
    PROM = tuple(s["promotable_labels"])

    # 名单全部来自 summarize()，渲染层不再重算筛选条件——否则「表里的行数」和
    # 「结论里说的列数」会是两套口径。
    promotable_best = s["promotable_cells"]
    pool_specific = s["pool_specific_cells"]

    A(f"# 全指标 × K 线周期 × 币种池 适配测评（轮次 {s.get('scope')}）")
    A("")
    A(f"口径 sha `{s.get('sha')}` · 统计生成 {s.get('stats_generated_at')} · 报告生成 "
      f"{s.get('generated_at')} · 方向表 {s.get('directions_count')} 个布尔条件列"
      f"（sha `{s.get('directions_sha')}`）· 成本 {pct((cfg.get('round_trip') or 0) * 100)}% 往返"
      f"（固定，不按策略覆盖）")
    A("")
    A("本报告是 `docs/indicator-evaluation-plan.md` 的 S0–S5 落地结果。md 里每个数都来自同一次运行"
      "写入的同名 `indicator-fit-matrix.json`，手改数字无效——要改结论得改口径重跑。")
    A("")
    A("## 1. 结论")
    A("")
    A("> 审计修正：本轮仅为探索性筛选，candidate 是当前实现标签。max-T 尚未实施；"
      "交互检验使用未校正多重比较的池内外 bootstrap，并非预登记的整体交互；"
      "Q 池是事后固定分档，major/alt 成员未完全冻结；子集标签未检查整体方向一致性。"
      "跨周期最强格还混合了币种构成和实际持有时间，不能单独证明周期适配。"
      "原始统计保留，后续验证 BLOCKED；详见 `docs/indicator-fit-readiness.md`。")
    A("")
    # 行数只报实测值 + 可推导的分解，不做「列数 × 格数」的乘法断言：退化列不进事件表、
    # 方向声明为 ambiguous 的列两个方向各计一行，简单乘积两头都对不上（曾经就在这行骗人）。
    # 格数按**该周期实际有到的池**算，不是拿 SCOPE_ORDER 的长度硬乘（5m 可能凑不齐 8 个池）。
    per_cell = []
    for iv in ivs:
        rows_n = (s["per_interval"][iv]["rows_evaluated"] or 0)
        cells_n = (s["per_interval"][iv]["cells"] or 1)
        per_cell.append(f"{iv} {rows_n:,}÷{cells_n:,} 格 = 每格 {round(rows_n / cells_n)} 行")
    A(f"- 评测量：方向表声明 {s['columns_declared']} 个布尔条件列，其中 {s['columns_total']} 列进入"
      f"统计（{s['columns_degenerate_excluded']} 列因触发率退化在筛选阶段就被排除）；逐"
      f"「列 × 池 × 预测长度 × 基准」出判定行，方向声明为 ambiguous 的列两个方向各计一行，"
      f"合计 {s['rows_evaluated_total']:,} 行（{'；'.join(per_cell)}。每格的行数 = 这一格里参评的"
      "「列 × 方向」组合数，所以会高于参评列数，多出来的是双向列，不是重复计数）。")
    A(f"- 说得出口（{'/'.join(PROM)}）的列：**{s['columns_promotable']} / {s['columns_total']}**；"
      f"其中当前实现标记为 `candidate` 的有 {s['columns_candidate']} 列，均待协议复核。")
    A("- 各列在本次样本中的最强格周期分布（探索性排名）："
      + "、".join(f"{iv} {s['promotable_by_interval'].get(iv, 0)} 列" for iv in ivs) + "。")
    A("- 最强格落在哪个池：" + "、".join(
        f"{SCOPE_LABEL.get(sc, sc)} {s['promotable_by_scope'].get(sc, 0)} 列"
        for sc in SCOPE_ORDER if s["promotable_by_scope"].get(sc)) + "。")
    A(f"- 原实现列为「池特有」的有 {len(pool_specific)} 列；由于事后分池、交互检验选择偏差"
      "及整体方向门缺失，该数量仅表示复核线索，不是已证实的池适配发现。")
    A("")
    A("## 2. 口径（S0 冻结）")
    A("")
    rows = [[iv, str(s["per_interval"][iv]["symbols"]), str(s["per_interval"][iv]["columns"]),
             str(s["per_interval"][iv]["columns_evaluated"]),
             str(s["per_interval"][iv]["rows_evaluated"]), str(s["per_interval"][iv]["rows_kept"]),
             f"{(prep.get('window_days') or {}).get(iv, '—')}d",
             str(((prep.get('bars_total_by_interval') or {}).get(iv)
                  or (probe.get('bars_total_by_interval') or {}).get(iv) or "—"))]
            for iv in ivs]
    L += table(rows, ["周期", "币数", "声明列数", "进统计列数", "判定行数", "落盘行数", "窗口",
                      "bars"])
    A("")
    # 同一个池在不同周期下的成员不一样：币是按周期各自过数据门的，5m 凑不齐 1h 的 119 个。
    # 之前只报 ivs[0] 的规模，读者拿「Q1 20 币」去对 30m 的格子数就发现对不上。逐周期列。
    A("- 币种池规模（按周期分别统计；`—` = 这个池在该周期下没有合格币，判定分布里也不会有它）：")
    A("")
    pool_rows = []
    for sc in SCOPE_ORDER:
        sizes = [(s["per_interval"][iv]["scope_sizes"] or {}).get(sc) for iv in ivs]
        if not any(v for v in sizes):
            continue
        pool_rows.append([SCOPE_LABEL.get(sc, sc)]
                         + [str(v) if v is not None else "—" for v in sizes])
    L += table(pool_rows, ["币种池"] + [f"{iv} 币数" for iv in ivs])
    ps = "、".join(f"{sc.replace('liq_', '').upper()} {n} 币"
                   for sc, n in (s.get("pool_sizes") or {}).items())
    A("")
    A(f"- Q1–Q4 为本轮新建池（全样本口径 {ps}），按近 {s.get('liquidity_months')} 个月月成交额"
      f"四分位划分，分档稳定性中位数 {s.get('tier_stability_median')}——"
      "「这个币属于哪一档」本身会漂，池级结论必须连同这个漂移一起读。")
    A(f"- 数据门：密度 ≥{(prep.get('data_gates') or {}).get('min_density')}、历史 ≥"
      f"{(prep.get('data_gates') or {}).get('min_history_bars')} bar、缺口率 ≤"
      f"{(prep.get('data_gates') or {}).get('max_gap_rate')}、零量占比 ≤"
      f"{(prep.get('data_gates') or {}).get('max_zero_volume_rate')}。S1 实查：")
    L += table([[iv, str((quality.get(iv) or {}).get("checked", "—")),
                 str((quality.get(iv) or {}).get("passed", "—")),
                 str(len((quality.get(iv) or {}).get("reject_reasons") or {}))] for iv in ivs],
               ["周期", "送检币数", "通过", "被拒币数"])
    A("")
    A(f"- 出场口径：{prep.get('entry') or '信号在 t 收盘确认，t+1 开盘成交，t+h 收盘出场'}；"
      f"预测长度 h ∈ {s['horizons_bars']} bar（各周期折算成各自的真实时间）。")
    A("- 方向口径：`excess = dir × (r − bench) − cost`，`dir` 只来自列名语义表，不用被测收益反推。"
      "来源分布：" + "、".join(f"{k} {v}" for k, v in (s.get("directions_by_source") or {}).items()) + "。")
    A("- 基准不减 0 不行：同时报 `unconditional`（同币同周期全体 bar 期望）与 `cross_section`"
      "（同一时刻一篮子等权期望），90 天普涨窗口里有多少「净为正」其实是 beta，只有减法知道。")
    A("")
    A("### 2.1 判定门")
    A("")
    L += table([
        ["1 效应", f"净均值 > 0 且 {cfg.get('n_boot')} 次按币 block bootstrap 的 95% CI 下界 > 0"],
        ["2 显著性", f"BH-FDR q < {cfg.get('q_alpha')}（家族 = 同周期 × 同池 × 同 h × 同基准，"
                    f"逐周期单独校正，跨周期不共享）"],
        ["3 稳定性", f"窗内 4 段同号率 ≥ {cfg.get('segment_sign_min')}，币间同号率 ≥ {cfg.get('symbol_sign_min')}"],
        ["4 可执行", f"2× 成本后净超额仍 > 0；格子事件 ≥{cfg.get('cell_events_min')}、"
                    f"覆盖币 ≥{cfg.get('cell_symbols_min')}、触发率落在 "
                    f"{(prep.get('gates') or {}).get('hit_rate_range', ['—', '—'])}"],
        ["5 币块下限", f"`symbols_in_stat ≥ {cfg.get('blocks_for_candidate')}`：币块少时 p 打到离散下界 "
                    f"1/(n_boot+1)，q<0.10 是免费的，降级为 `underpowered_blocks`"],
        ["分层附加", f"格子须过交互检验 p < {cfg.get('interaction_alpha')}，并按 "
                  f"w=n/(n+{cfg.get('shrink_k')}) 向整体收缩后才许引用"],
    ], ["关卡", "判据"])
    A("")
    A(f"判定词表（强→弱）：{', '.join(VERDICT_ORDER)}。`candidate` 会进台账，其余都不算发现。")
    A("")
    A("## 3. 周期适配：每列最强格")
    A("")
    if promotable_best:
        A("只列「说得出口」的列，按判定等级 → CI 下界排序。`池` 是最强格所在币种池；"
          "`整体` 是同列在 `all` 池的最强判定，用来对照池级结论是否只是整体复读。")
        A("")
        rows = []
        for r in promotable_best[:60]:
            cs = r.get("cost_stress_pct") or {}
            it = r.get("interaction") or {}
            rows.append([
                f"`{r.get('indicator')}.{r.get('column')}`",
                "多" if (r.get("direction") or 0) > 0 else ("空" if (r.get("direction") or 0) < 0 else "双向"),
                str(r.get("interval")), SCOPE_LABEL.get(r.get("scope"), str(r.get("scope"))),
                f"h{r.get('horizon_bars')}",
                "截面" if r.get("benchmark") == "cross_section" else "无条件",
                f"{r.get('events'):,}", pct(r.get("mean_gross_pct")), pct(r.get("mean_net_pct")),
                pct(r.get("ci_low_pct")),
                "—" if r.get("q_value") is None else f"{r['q_value']:.4f}",
                pct(r.get("seg_consistency")), pct(r.get("symbol_sign_rate")),
                pct(cs.get("x2")),
                f"{it.get('p'):.3f}" if it.get("p") is not None else "—",
                str(r.get("verdict_cell") or r.get("verdict")),
                str(r.get("pooled_verdict") or "—"),
            ])
        L += table(rows, ["指标.列", "向", "周期", "池", "h", "基准", "事件", "毛%", "净%",
                          "CI下界%", "q", "段同号", "币同号", "2×成本%", "交互p", "判定", "整体判定"])
        A("")
        A(f"（前 {min(60, len(promotable_best))} / {len(promotable_best)} 行；"
          "全部行见交互页与 `indicator-fit-stats-*.json`。）")
    else:
        A("**本轮没有任何一列在任何 (周期, 池, h, 基准) 组合上说得出口。** 这本身就是结论："
          "全部注册指标的布尔条件列，在 4 个周期、8 个币种池、去 beta 与方向对齐之后，"
          "没有一列能过前四道关——连「币块不足但方向已声明」的 `underpowered_blocks` 都没有。")
    A("")
    A("### 3.1 判定分布（全量行统计，不受落盘裁剪影响）")
    A("")
    rows = []
    for iv in ivs:
        vc = s["per_interval"][iv]["verdict_counts_by_scope"] or {}
        for sc in SCOPE_ORDER:
            if sc not in vc:
                continue
            rows.append([iv, SCOPE_LABEL.get(sc, sc)]
                        + [str(vc[sc].get(v, 0)) for v in VERDICT_ORDER])
    L += table(rows, ["周期", "池"] + VERDICT_ORDER)
    A("")
    A("### 3.2 指标级汇总：每个指标适合哪个 K 线")
    A("")
    A(f"{s['columns_promotable']} 个说得出口的列分布在 {s['indicators_promotable']} 个指标上"
      "（指标按列数排序，最多列 40 行）。下表按**每一列自己的最强格**落在哪个周期计数："
      "同一指标的不同列可以落在不同周期，`0` 表示这个指标在该周期没有一列说得出口。"
      "这是本次样本的探索性最强格计数，不能单独归因为周期适配；它统计的是**列**，同一指标多列同堆时"
      "（见 §5）算不了独立证据，读列数之前先读堆数。`最强判定` 取该指标所有列里最强的一级。")
    A("")
    rows = [[f"`{e['indicator']}`", str(e["columns"])]
            + [str(e["by_interval"].get(iv, 0)) for iv in ivs]
            + [str(e["best_verdict"]),
               "、".join(f"{SCOPE_LABEL.get(sc, sc)} {n}" for sc, n in
                         sorted(e["by_scope"].items(), key=lambda kv: (-kv[1], str(kv[0]))))
               or "—"]
            for e in s["indicator_rollup"][:40]]
    L += table(rows, ["指标", "列数"] + [f"{iv} 列" for iv in ivs] + ["最强判定", "最强格落点（池）"])
    A("")
    if len(s["indicator_rollup"]) > 40:
        A(f"（只列前 40 / {len(s['indicator_rollup'])} 个指标，按列数排序；"
          "整表在 `indicator-fit-matrix.json` 的 `indicator_rollup` 里，交互页则按列展开、可按指标名搜索。）")
        A("")
    A("## 4. 币种池适配")
    A("")
    A("下表保留原实现的池级统计。当前交互 p 未作跨格校正，Q 池含事后信息，"
      "且子集 candidate 未检查整体方向；不能将表中的「池特有」标签当作通过预登记的发现。")
    A("")
    rows = []
    for sc in SCOPE_ORDER:
        rows.append([
            SCOPE_LABEL.get(sc, sc), size_text(s, ivs, sc),
            str(sum(1 for r in promotable_best if r.get("scope") == sc)),
            str(sum((s["per_interval"][iv]["verdict_counts_by_scope"] or {}).get(sc, {}).get("candidate", 0)
                    for iv in ivs)),
            str(sum(1 for r in pool_specific if r.get("scope") == sc)),
        ])
    L += table(rows, ["币种池", "币数（分周期）", "最强格落此池的列数",
                      "此池 candidate 格数", "池特有结论数"])
    A("")
    if pool_specific:
        A("原池特有标签明细（按判定等级 → 收缩后净超额排序，仅供复核）。`整体` 两列是同列"
          "在 `all` 池跨周期的最强格，并非同周期、同 h、同基准配对；该比较不能替代整体方向门。")
        A("")
        L += table([[f"`{(r.get('key') or '').replace('|', '.')}`", str(r.get("interval")),
                     SCOPE_LABEL.get(r.get("scope"), str(r.get("scope"))),
                     f"h{r.get('horizon_bars')}", f"{r.get('events'):,}",
                     pct(r.get("mean_net_pct")), pct(r.get("shrunk_net_pct")),
                     pct(r.get("pooled_mean_net_pct")), str(r.get("pooled_verdict") or "—"),
                     str(r.get("verdict_cell")),
                     f"{(r.get('interaction') or {}).get('p'):.3f}"
                     if (r.get("interaction") or {}).get("p") is not None else "—"]
                    for r in pool_specific[:40]],
                   ["列", "周期", "池", "h", "事件", "格子净%", "收缩后净%", "整体净%",
                    "整体判定", "格子判定", "交互p"])
    else:
        A("本轮没有池特有结论。")
    A("")
    A("## 5. 冗余归堆")
    A("")
    A(f"同堆的列在**哪些币上有效**这件事高度一致——按逐币平均超额的币间 Pearson 相关衡量，"
      f"r ≥ {r_min} 连成一堆（并查集，传递闭包）。这不是逐事件重合率：聚合缓存里没有逐事件"
      "掩码，为了算重合率要再跑一遍几小时的事件抽取，不划算；而且币间相关会漏掉「在不同币上"
      "各说各话、但触发时刻几乎相同」的同族列——那一层留给后续小样本复核。"
      "**同堆只能算一份证据**：把 3 个同族指标当 3 个独立发现会直接放大 FDR 的欺骗性。"
      "归堆只在「说得出口」的列之间做，且按各自最强 h 分组，跨 h 的同族列不会连起来。")
    A("")
    L += table([[iv, str(s["per_interval"][iv]["redundancy_pairs"]),
                 str(s["per_interval"][iv]["redundancy_members"]),
                 str(s["per_interval"][iv]["redundancy_clusters"]),
                 " / ".join(f"{g['size']}列" for g in s["per_interval"][iv]["redundancy_largest"][:4])]
                for iv in ivs],
               ["周期", "冗余对", "参与归堆列数", "独立堆数", "最大几堆规模"])
    A("")
    # 只给堆的规模等于没给证据：读者要能核对「这几列真是同一个信号吗」。列名截到 8 个
    # （`cluster_count` 里就截了），更大的堆去 json 的 redundancy 字段看全量对。
    for iv in ivs:
        multi = [g for g in s["per_interval"][iv]["redundancy_largest"] if g["size"] >= 3]
        if not multi:
            continue
        A(f"{iv} 上 ≥3 列的同堆（最多列 6 堆，成员按列名排序截断）：")
        A("")
        for g in multi[:6]:
            mem = "`. `".join(m.replace("|", ".") for m in g["members"])
            A(f"- {g['size']} 列：`{mem}`")
        A("")
    A("## 6. 成本与可执行性")
    A("")
    L += table([[iv,
                 str(sum(1 for r in s["best_cells"] if r.get("interval") == iv)),
                 str(sum(1 for r in s["best_cells"] if r.get("interval") == iv
                         and ((r.get("cost_stress_pct") or {}).get("x2") or 0) > 0)),
                 str(sum(1 for r in s["best_cells"] if r.get("interval") == iv
                         and ((r.get("cost_stress_pct") or {}).get("x3") or 0) > 0))]
                for iv in ivs],
               ["周期", "最强格数", "2×成本后仍>0", "3×成本后仍>0"])
    A("")
    A(f"往返成本固定 {pct((cfg.get('round_trip') or 0) * 100)}%（单边手续费 0.04% + 滑点 0.02%），"
      "**不允许按策略覆盖**。`cost_fragile` = 只靠成本假设活着，一律不算发现。")
    A("")
    A("## 7. 算力与运行证据")
    A("")
    L += table([[iv, str(((events.get("intervals") or {}).get(iv) or {}).get("symbols", "—")),
                 str(((events.get("intervals") or {}).get(iv) or {}).get("computed", "—")),
                 str(((events.get("intervals") or {}).get(iv) or {}).get("cached", "—")),
                 pct(((events.get("intervals") or {}).get(iv) or {}).get("seconds"), 0),
                 str(len(((events.get("intervals") or {}).get(iv) or {}).get("failures") or []))]
                for iv in ivs],
               ["周期", "币数", "本轮计算", "缓存命中", "耗时 s", "失败币数"])
    A("")
    if probe:
        wh = probe.get("estimate_wall_hours_at_workers") or {}
        acc = s.get("indicator_accounting") or {}
        bars_probe = probe.get("bars_total_by_interval") or {}
        bars_here = sum(v for iv in ivs if (v := bars_probe.get(iv)))
        ratio = (bars_here / sum(bars_probe.values())) if bars_probe else None
        A(f"- 探针实测中位 {probe.get('median_seconds_per_bar')} s/bar"
          f"（{probe.get('interval')}、{len(probe.get('symbols') or [])} 币）。算力模型是"
          f"「代价 ∝ bar 数」：探针按全量 {len(bars_probe)} 周期总 bar 数 "
          f"{sum(bars_probe.values()):,} 外推 = {probe.get('estimate_cpu_hours')} CPU 小时，"
          f"10 worker ≈ {wh.get(10, wh.get('10', '—'))} h、12 worker ≈ "
          f"{wh.get(12, wh.get('12', '—'))} h。本报告覆盖 {'、'.join(ivs)}，"
          + (f"其总 bar 数 {bars_here:,}（占探针口径 {ratio:.0%}）；"
             if ratio is not None and ratio < 0.995 else
             f"其总 bar 数 {bars_here:,}，即探针外推口径的全部；")
          + "s/bar 只在单个周期上探针过，跨周期按 bar 数线性外推、没有实测第二个周期——"
            "真正的逐周期耗时看上表。")
        # 外推值要和实测对账，否则§7 只是一句「我们估计要 5 小时」——本轮真有 4 个周期的实测。
        elapsed = events.get("elapsed_s")
        proj10 = wh.get(10, wh.get("10"))
        if elapsed and isinstance(proj10, (int, float)) and proj10:
            wall = elapsed / 3600.0
            A(f"- 实测对账：四周期事件抽取墙钟合计 {elapsed:,.0f} s ≈ {wall:.2f} h，"
              f"探针在 10 worker 下外推 {proj10} h，误差 {wall / proj10 - 1:+.0%}"
              "（本轮 worker 数记在 `output/fit-events-A.log` 首行命令里，产物 json 不存这个字段）。"
              "「代价 ∝ bar 数」在本轮没被证伪，但它只有这一组实测——s/bar 仍只在一个周期上量过，"
              "别当周期性规律外推。")
        A(f"- 事件抽取跳过 {s.get('indicators_skipped')} 个指标注册项（注册 {acc.get('registered')} 个，"
          f"其中 {acc.get('with_boolean_columns')} 个产出布尔列；{acc.get('no_boolean_columns')} 个"
          f"一列都不出，{acc.get('raised')} 个在探测帧上必然报错）。跳过它们不改变列集合——列由 dtype "
          f"决定，与币种/周期无关；`fit_events.run_interval` 对首币自检产出列数是否等于方向表的 "
          f"{probe.get('columns_expected_from_directions')} 列，不等就打报警。")
    A("")
    A("## 8. 判负与判不出的列（REJECTED 登记）")
    A("")
    A(f"- {s['columns_total'] - s['columns_promotable']} 列在任何周期、任何池、任何 h、任何基准下"
      "都没有说得出口的一格，全部按 REJECTED 登记，不做「换个参数再来一遍」。")
    A(f"- 判定分布（表 3.1）合计 {s['rows_evaluated_total']:,} 行，全部按上面五道关逐行给出，"
      "没有「先筛掉不显著的再调参」这一步。")
    A(f"- {s['columns_degenerate_excluded']} 列在筛选阶段就被排除：触发率不在 "
      f"{(prep.get('gates') or {}).get('hit_rate_range', ['—', '—'])} 区间——恒真列等于「每根 bar "
      "都开仓」，恒假列一个事件都没有。它们是「无法测」，不是「测过无效」，两者不许混写。")
    A("- `no_edge` = 净超额 ≤ 0（不是「没测到」）；`inconclusive` = 均值为正但 CI 下界 ≤ 0；"
      "`likely_noise` = CI 撑住了但 FDR q 没过。三者都不许写成「可能有效，待样本更多」。")
    A(f"- `underpowered_blocks` 是本轮新加的第 5 道门：因参与自助法的币块不足 "
      f"{cfg.get('blocks_for_candidate')} 个而降级，数字照给、不上候选台账。")
    A("")
    A("## 9. 局限")
    A("")
    A("- 窗口只有 90 天（四周期同窗可比是本轮主口径），是单一市场状态下的结论；"
      "`seg_consistency` 只覆盖窗内 4 段，跨年 regime 变化测不到。")
    A("- 事件研究只回答「信号后价格怎么走」，不回答「能不能赚钱」。`candidate` 接进生产前必须再过"
      "一遍 `runner.run_backtest`（真实仓位、分批止盈、止损、资金占用），不一致时以回测为准。")
    A(f"- bar 来源规则：{prep.get('provenance_rule') or '同一轮同一周期内某币只用一个来源：'
       'native 满密度就用 native，否则派生'}；5m/15m/30m 由 1m 聚合、1h 由 15m 聚合。")
    A("- 横截面基准要求同一时刻 ≥5 个币可对齐，`cs_coverage` 明示样本被砍掉多少；"
      "覆盖率低的列，其截面结论本来就弱。")
    A("- 判定只用到「列名语义声明的方向」。声明为 ambiguous 的列两个方向都试，最好的一格也封顶在 "
      "`exploratory`——这是刻意的：事后挑方向不算发现。")
    pools_iv = {iv: len(s["per_interval"][iv]["verdict_counts_by_scope"] or {}) for iv in ivs}
    A(f"- 校正只做到族内：BH-FDR 的家族 = 同周期 × 同池 × 同 h × 同基准，本轮共 "
      f"{s['rows_evaluated_total']:,} 条判定行（{'、'.join(f'{iv} {n} 池' for iv, n in pools_iv.items())}）。"
      "跨周期、跨池这些「层」没有计入 DSR/PBO 的有效试验数——那是 S8 汇总要补的账。"
      "所以这里的 q 值回答「同一格这么多列里谁运气好」，不回答「整个项目一共试了多少次」。")
    A("- 本轮只落 S0–S5：组合筛选（S6）、真实仓位回测（S7）、台账汇总与试验数校正（S8）都没跑。"
      "候选名单要接进任何策略之前，后两步不是可选项（承接上文「事件研究只回答信号后价格怎么走」"
      "那条，已登记为未决项：`docs/decisions/OPEN-DECISIONS.md`）。")
    diagnosis = s.get("log_diagnosis") or {}
    if diagnosis.get("diagnostic_status") == "PASS":
        m = diagnosis["metrics"]
        A(f"- 5m log 告警已定位并完成工程复核：trend_persistence_score 在平收盘窗口中提前求值了"
          f"未选中的 log(0) 分支。重放 {m['symbols_checked']} 币，涉及 {m['zero_log_input_bars']} 根K线；"
          "修复保留原回退值，完整输出与旧函数精确一致，选定事件统计与原缓存一致。"
          "见 `docs/indicator-fit-log-diagnosis.md`；该工程结论不解除其余方法或样本外验证阻塞。")
    else:
        A("- 本轮抽取过程中出现过 1 次未定位的 `divide by zero encountered in log`（pandas 告警，落在 5m 段，"
          "见 `output/fit-events-A.log`：`grep -c 'divide by zero' output/fit-events-A.log` = 1）。"
          "告警来源尚未定位，影响方向与范围尚未确认。无穷值比较可能为 True，NaN 的不等比较也可能为 True；"
          "不能由告警或逐币零失败推断信号安全。候选进入交易化验证前须定位指标、币种与时间，并复核相关信号。")
    A("")
    A("## 10. 产物与复现")
    A("")
    # 名字一律取本次真实写出的文件名：`--stats` / `--suffix` 是自检通道，
    # 表里还写正式名就是把自检产物登记成交付物。
    stats_name = s.get("stats_file") or f"indicator-fit-stats-{s.get('scope')}.json"
    matrix_md_name = s.get("out_md") or "indicator-fit-matrix.md"
    matrix_json_name = s.get("out_json") or "indicator-fit-matrix.json"
    L += table([
        [f"`docs/{stats_name}`",
         "判定明细：每列每池最强格 + 说得出口的列的全明细（紧凑写，分布按全量行统计）。"
         + (f"本轮实测 {s['stats_bytes'] / 1e6:.1f} MB，是原始产物不是交付物——"
            "报告与页面只读它，不重算。" if s.get("stats_bytes") else "")],
        [f"`docs/{matrix_md_name}`", "本报告本身（与 json 同一次运行写出，md 的数都能在 json 里对上）"],
        [f"`docs/{matrix_json_name}`", "本报告数字底稿"],
        ["`docs/indicator-fit-matrix.html`", "交互矩阵页：列 × 周期 × 池，可筛可排可下钻（由 `fit_matrix_page.py` 写出，不是本脚本）"],
        ["`docs/indicator-directions.json`", "方向声明表（列名语义 → dir）"],
        ["`docs/data-quality-<scope>-<interval>.json`", "S1 数据门逐币检查结果"],
        ["`docs/fit-universe.json`", "币种池、流动性分档与分档稳定性"],
    ], ["文件", "内容"])
    A("")
    A("```")
    A("python backend/scripts/fit_prep.py                        # S0 冻结口径（写 scope sha）")
    A("python backend/scripts/indicator_fit_matrix.py --scope A --stage directions")
    A("python backend/scripts/indicator_fit_matrix.py --scope A --stage probe")
    A("python backend/scripts/indicator_fit_matrix.py --scope A --stage events --workers 12")
    A("python backend/scripts/indicator_fit_matrix.py --scope A --stage stats")
    A("python backend/scripts/fit_matrix_report.py --scope A     # 本 md + 同名 json")
    A("python backend/scripts/fit_matrix_page.py --scope A       # 交互页")
    A("python backend/scripts/fit_matrix_report.py --scope A --ledger \\")
    A("    --verification '<测试计数>'   # 追加一条 research/experiments.jsonl 台账（数字全部取自本次产物）")
    A("python -m pytest backend/tests/test_fit_stats.py backend/tests/test_fit_events.py \\")
    A("                   backend/tests/test_fit_redundancy.py backend/tests/test_fit_report.py \\")
    A("                   backend/tests/test_fit_ledger.py backend/tests/test_fit_page.py -q")
    A("```")
    A("")
    return "\n".join(L) + "\n"


def attach_prep(s: dict, spec: dict, uni: dict, probe: dict) -> dict:
    """把 S0 冻结口径与 S1 币种池的元数据并进报告字典（md 与台账共用一份，别各拼一遍）。"""
    stab = sorted((uni.get("tier_stability") or {}).values())
    s.update({
        "liquidity_months": len(uni.get("monthly_liquidity_bins") or {}),
        "pool_sizes": uni.get("pool_sizes") or {},
        "window_days": spec.get("window_days") or {},
        "data_gates": spec.get("data_gates") or {},
        "gates": spec.get("gates") or {},
        "entry": spec.get("entry"),
        "provenance_rule": (spec.get("provenance") or {}).get("rule"),
        "pools": spec.get("pools") or {},
        "bars_total_by_interval": probe.get("bars_total_by_interval") or {},
        "universe_size": uni.get("universe_size") or {},
        "tier_stability_median": stab[len(stab) // 2] if stab else None,
    })
    return s


def git_state() -> str:
    """台账必须记下「哪份代码跑出来的」。工作区脏就写明脏，别假装可复现。

    只说「脏、几个路径」，不猜是哪些改动——本轮的产物和别人的半成品混在同一个工作区时，
    写死「fit pipeline uncommitted」就是一句可能为假的陈述。
    """
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=30).stdout.strip()
        porcelain = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                   capture_output=True, text=True, timeout=60).stdout
    except OSError:
        return "unknown"
    if not head:
        return "unknown"
    n_dirty = len([ln for ln in porcelain.splitlines() if ln.strip()])
    return head if n_dirty == 0 else f"{head} (dirty worktree, {n_dirty} 个路径)"


def ledger_record(s: dict, events: dict, parent_id: str, verification: str,
                  suffix: str = "") -> dict:
    """把本轮筛选写成一条台账。数字全部取自 `s`（= 报告 json 的同一份字典），不新造口径。"""
    ivs = s["intervals"]
    per = s["per_interval"]
    cfg = s.get("config") or {}
    # 台账记的文件名必须是这次真实读到的那份：`--stats` 覆盖过就是自检产物，
    # 挂正式名等于让后人以为台账那一行来自全量 stats。
    stats_name = s.get("stats_file") or f"indicator-fit-stats-{s.get('scope')}.json"
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment_id": LEDGER_EXPERIMENT_ID,
        "parent_experiment_id": parent_id,
        "strategy_id": None,
        "code_commit": git_state(),
        "dataset_version": (
            f"local PostgreSQL get_money (read_only); scope {s.get('scope')} sha {s.get('sha')}; "
            f"windows days={s.get('window_days')}; provenance={s.get('provenance_rule')}; "
            f"bars per interval={s.get('bars_total_by_interval')}"),
        "universe_snapshot": (
            f"per-interval qualified symbols {[per[iv]['symbols'] for iv in ivs]}; "
            f"pool sizes={s.get('pool_sizes')}; liquidity months={s.get('liquidity_months')}; "
            f"tier stability median={s.get('tier_stability_median')}"),
        "symbols": {iv: per[iv]["symbols"] for iv in ivs},
        "interval": ivs,
        "hypothesis_card": ("docs/indicator-evaluation-plan.md (S0–S5 预登记，含判据 5 与 S5 "
                            "口径偏离登记); 本轮是筛选而非单假设检验，无 hypotheses 卡"),
        "stage": ("S0–S5 全量事件研究筛选（触发时刻对齐的方向化超额 + 双基准 + BH-FDR + "
                  "分段/币间一致性 + 交互检验 + 收缩 + 成本压力）；只读，无 paper/live，无下单"),
        "parameters": {
            "source": "docs/indicator-eval-scope.json (spec key A)",
            "scope_sha": s.get("sha"),
            "directions_sha": s.get("directions_sha"),
            "grid_search_performed": False,
            "parameter_trials": 0,
            "horizons_bars": s.get("horizons_bars"),
            "benchmarks": s.get("benchmarks"),
            **cfg,
        },
        "fee_model": f"round_trip {cfg.get('round_trip')} applied to every cell (fixed, not per-strategy)",
        "slippage_model": "included in round_trip (0.0004 fee + 0.0002 slip per side)",
        "funding_model": "not modelled (event study on price path only)",
        "cost_basis": ("mean net excess per event, %; cost stress at 2x and 3x round_trip; "
                       "cost_fragile 明确不算发现"),
        "metrics": {
            "columns_declared": s.get("columns_declared"),
            "columns_evaluated": s.get("columns_evaluated"),
            "columns_degenerate_excluded": s.get("columns_degenerate_excluded"),
            "columns_promotable": s.get("columns_promotable"),
            "columns_candidate": s.get("columns_candidate"),
            "indicators_promotable": s.get("indicators_promotable"),
            "pool_specific_columns": s.get("pool_specific_columns"),
            "pool_specific_by_scope": s.get("pool_specific_by_scope"),
            "promotable_by_interval": s.get("promotable_by_interval"),
            "rows_evaluated_total": s.get("rows_evaluated_total"),
            "verdict_counts_all_by_interval": {iv: per[iv]["verdict_counts_all"] for iv in ivs},
            "redundancy": {iv: {"pairs": per[iv]["redundancy_pairs"],
                                "members": per[iv]["redundancy_members"],
                                "clusters": per[iv]["redundancy_clusters"]} for iv in ivs},
            "events_run": {iv: {k: (v or {}).get(k) for k in ("computed", "cached", "seconds")}
                           for iv, v in (events.get("intervals") or {}).items()},
        },
        "decision": ("screen_only: 事件研究给出「哪列在哪个周期/池说得出口」的登记，"
                     "不构成可部署策略；candidate 必须先过 runner.run_backtest"),
        "failure_reason": None,
        "interpretation": (
            f"{s.get('columns_declared')} 列声明、{s.get('columns_evaluated')} 列进入统计，"
            f"{s.get('columns_promotable')} 列至少有一格说得出口，{s.get('columns_candidate')} 列"
            f"最强格达 candidate；{s.get('pool_specific_columns')} 列的结论是**池特有**的"
            "（池级 candidate 且整体池不 candidate）。其余全部按 REJECTED 登记，不换参数重试。"),
        "controls_registered_not_executed": [
            "逐事件重合率的冗余复核（本轮 S5 只做币间均值相关的近似，见 plan §6.1 偏离登记）",
            "candidate 接进 runner.run_backtest 的真实仓位/分批止盈/止损复核",
            "跨年 regime 的分段一致性（本轮窗口仅 90 天）",
        ],
        "stop_conditions": [
            "同一列换 h / 换基准 / 换池再试一遍不算新发现（FDR 分母已按全格子计入）",
            "cost_fragile 不升级；underpowered_blocks 不进候选台账",
        ],
        "next_action": ("把 candidate 列按池分组交给回测复核（S7），或按用户指定的周期/池先做小样本"
                        "逐事件重合率复核（S5 补强）"),
        "runtime_cost": {
            "events_seconds_by_interval": {
                iv: (v or {}).get("seconds")
                for iv, v in (events.get("intervals") or {}).items()},
            "events_total_seconds": events.get("elapsed_s"),
            "db_access": "read_only engine",
        },
        "artifacts": [
            f"docs/indicator-fit-matrix{suffix}.md",
            f"docs/indicator-fit-matrix{suffix}.json",
            f"docs/{stats_name}",
            "docs/indicator-fit-matrix.html",
            "docs/indicator-directions.json",
            "docs/indicator-eval-scope.json",
            "docs/fit-universe.json",
        ],
        "verification": verification,
    }


def append_ledger(rec: dict, path: Path) -> str:
    """追加一条台账。同 id + 同 sha = 同一轮重跑，幂等跳过；同 id + 不同 sha 直接报错。

    后半个分支是刻意的：换口径重跑必须换 `experiment_id`，否则台账里同一个编号对应两套
    数字，`OPEN-DECISIONS` 之类的引用就说不清指的是哪一次。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new_sha = (rec.get("parameters") or {}).get("scope_sha")
    for line in (path.read_text(encoding="utf-8").splitlines() if path.exists() else []):
        try:
            old = json.loads(line)
        except json.JSONDecodeError:
            continue
        if old.get("experiment_id") != rec["experiment_id"]:
            continue
        old_sha = (old.get("parameters") or {}).get("scope_sha")
        if old_sha == new_sha:
            return f"{rec['experiment_id']} 已在台账里（sha {new_sha}），幂等跳过"
        raise SystemExit(f"{rec['experiment_id']} 已登记在 sha {old_sha} 上，本轮是 {new_sha}："
                         "换口径重跑要换 experiment_id，别复用编号")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    try:
        label = str(path.relative_to(ROOT))
    except ValueError:  # 单测写临时目录：路径不在仓库里也要能报出写到哪了
        label = str(path)
    return f"append -> {label}"


def last_ledger_id(path: Path) -> str | None:
    """返回台账里**最后写入**的那条编号。

    这是顺序前驱，不是逻辑父子——本轮是筛选，跟前一格（可能是任何假设卡）没有派生关系。
    真正的假设来源记在 `hypothesis_card` 里；读 `parent_experiment_id` 时按「上一条」理解。
    """
    if not path.exists():
        return None
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            exp = json.loads(line).get("experiment_id")
        except json.JSONDecodeError:
            continue
        if exp:
            return exp
    return None


def norm_suffix(raw: str) -> str:
    """把 `--suffix dev` 归一成 `-dev`。

    坑：文件名是 `indicator-fit-matrix{suffix}.md` 直接拼的，传 `dev` 会写出
    `indicator-fit-matrixdev.md`——既不是正式产物也不是约定的自检产物，还容易误当成
    正式交付（少了那个连字符）。这里统一补上，空串保持空（正式产物）。
    """
    if not raw:
        return ""
    return raw if raw.startswith("-") else f"-{raw}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="A")
    ap.add_argument("--stats", default="", help="覆盖统计产物路径（自检用，正式渲染不要传）")
    ap.add_argument("--suffix", default="", help="交付文件名后缀（自检用，别盖正式产物）")
    ap.add_argument("--ledger", action="store_true",
                    help="追加一条 research/experiments.jsonl 台账（数字全部取自本轮产物）")
    ap.add_argument("--verification", default="",
                    help="台账里记一句「本次交付前实际跑过什么」——必须是真跑过的输出，不许留空凑数")
    args = ap.parse_args()
    if args.ledger and args.suffix:
        # 带后缀的是单周期调试产物，写进台账就是拿局部结果冒充全量结论。
        print("[report ] --ledger 不能和 --suffix 同时用：自检产物不进台账", flush=True)
        return 1
    if args.ledger and not args.verification:
        print("[report ] --ledger 需要 --verification：台账里「验证」一栏空着等于没记", flush=True)
        return 1
    stats_path = Path(args.stats) if args.stats else DOCS / f"indicator-fit-stats-{args.scope}.json"
    stats = load(stats_path)
    if stats is None:
        print(f"[report ] 缺统计产物 {stats_path.name}——先跑 --stage stats", flush=True)
        return 1
    dirs = load(DOCS / "indicator-directions.json", {"columns": [], "count": 0})
    events = load(DOCS / f"fit-events-{args.scope}.json", {})
    probe = load(DOCS / f"fit-probe-{args.scope}.json", {})
    # 口径文件只读不猜：缺这一轮直接失败，和 stage_stats 用同一份 S0 冻结产物。
    spec = (load(DOCS / "indicator-eval-scope.json", {}) or {}).get(args.scope)
    if spec is None:
        print(f"[report ] docs/indicator-eval-scope.json 里没有轮次 {args.scope}——先跑 fit_prep.py",
              flush=True)
        return 1
    uni = (load(DOCS / "fit-universe.json", {}).get("scopes") or {}).get(args.scope) or {}
    quality = {iv: load(DOCS / f"data-quality-{args.scope}-{iv}.json") or {}
               for iv in (stats.get("intervals") or {})}
    s = summarize(stats, dirs, events)
    attach_prep(s, spec, uni, probe)
    diagnosis = load(DOCS / "indicator-fit-log-diagnosis.json", {}) or {}
    source = ROOT / "backend/app/catalog_indicators/trend_persistence_score.py"
    if (diagnosis.get("scope_sha") == s["sha"]
            and diagnosis.get("diagnostic_status") == "PASS"
            and diagnosis.get("source_sha256") == hashlib.sha256(source.read_bytes()).hexdigest()):
        s["log_diagnosis"] = diagnosis
    # §10 要说清「原始产物有多大」：四周期全量的 stats json 到几十 MB，别让人以为交付物是这个。
    # 名字也必须写实：`--stats` 覆盖过就是自检产物，写个正式名等于伪造来源。
    s["stats_bytes"] = stats_path.stat().st_size if stats_path.exists() else None
    s["stats_file"] = stats_path.name
    suffix = norm_suffix(args.suffix)
    out_json = DOCS / f"indicator-fit-matrix{suffix}.json"
    out_md = DOCS / f"indicator-fit-matrix{suffix}.md"
    s["out_md"], s["out_json"] = out_md.name, out_json.name
    md = render_md(s, events, probe, quality, s)
    out_json.write_text(json.dumps(s, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    out_md.write_text(md, encoding="utf-8")
    print(f"[report ] {out_md.name} {out_md.stat().st_size / 1024:.0f} KB + "
          f"{out_json.name} {out_json.stat().st_size / 1024:.0f} KB；"
          f"{s['columns_total']} 列，说得出口 {s['columns_promotable']} 列，"
          f"池特有 {s['pool_specific_columns']} 列", flush=True)
    if args.ledger:
        rec = ledger_record(s, events, last_ledger_id(LEDGER_PATH), args.verification, suffix)
        # 不打印 parent：幂等跳过时台账尾巴就是本轮自己，「P18（parent P18）」是自指；
        # 真正写进去的值在记录里，看那一行就行。
        print(f"[ledger ] {LEDGER_EXPERIMENT_ID}：{append_ledger(rec, LEDGER_PATH)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
