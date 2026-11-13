"""P18 币种池成员 PIT 口径审计（只读，不改任何已冻结产物）。

做四件事，全部是测量，不是选择：
1. 复现 A 轮 Q 档，确认 `expected_bars` int32 溢出是否真的影响了已交付的分档；
2. 用修好的 partial 判定重算同一窗口的档位，量化差异——**只报告，不落库、不覆盖产物**；
3. 用「严格早于事件月的完整月」算 PIT 档，量化事后固定档的时点泄漏，并指出哪些事件月
   根本无从 PIT 定档；
4. 冻结 major/alt 与 Q 池的当前成员，逐集合比漂移。major/alt 是 rule 池、A 轮从未落盘
   成员名单，所以只能比计数，且本次冻结是**事后快照**，不能冒充 A 轮篮子。

不重算收益/统计，不改 A 轮标签，不重选指标/周期/池/h，参数试验 0。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
from app.db import make_engine  # noqa: E402
from app.research import fit_data as fd  # noqa: E402
from app.research.fit_membership import (  # noqa: E402
    membership_drift, pit_tiers_by_month, tier_disagreement,
)
from app.strategy.pools import pool_symbols_by_code  # noqa: E402
from fit_prep import SCOPES  # noqa: E402

DOCS = ROOT / "docs"
INPUTS = ("indicator-eval-scope.json", "fit-universe.json", "indicator-fit-stats-A.json")
Q_CODES = ("liq_q1", "liq_q2", "liq_q3", "liq_q4")
RULE_CODES = ("major", "alt")
# 与 fit_readiness.history_gate 同一条预登记历史要求，不新造旋钮
HISTORY_MONTHS = 15
GROUPS = 4


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_window_end(uni: dict) -> dict:
    """从冻结 Q 池 rule 里的 window_end_ms 反推 A 轮窗口末端。

    prep 用当时的 `max(open_time)` 当末端，而线上表一直在长；要复现冻结窗口就必须
    用产物里留下的时点锚，而不是今天的末端。锚取整到小时（1h 柱的开盘时刻）。
    """
    ends = []
    for code in Q_CODES:
        rule = (uni.get("pool_rules") or {}).get(code) or {}
        if rule.get("window_end_ms"):
            ends.append(int(rule["window_end_ms"]))
    if not ends:
        return {"status": "NO_FROZEN_ANCHOR", "end_ms": None}
    anchor = min(ends)
    floored = anchor - anchor % 3_600_000
    return {"status": "OK", "end_ms": floored, "source": "min(liq_q*.rule.window_end_ms)",
            "raw_min_ms": anchor,
            "iso": pd.Timestamp(floored, unit="ms", tz="UTC").isoformat()}


def month_window_days(lo_ms: int, hi_ms: int, months: list[str]) -> dict[str, float]:
    """评估窗 [lo, hi] 落在每个自然月里的天数——用来按时间加权，而不是按月数平均。"""
    out: dict[str, float] = {}
    for ym in months:
        start = pd.Timestamp(f"{ym}-01", tz="UTC")
        end = start + pd.DateOffset(months=1)
        a = max(lo_ms, int(start.value // 1_000_000))
        b = min(hi_ms, int(end.value // 1_000_000))
        out[ym] = max(0.0, (b - a) / 86_400_000)
    return out


def tier_delta(frozen: dict[str, str], recomputed: dict[str, str]) -> dict:
    both = sorted(set(frozen) & set(recomputed))
    changed = [{"symbol": s, "frozen": frozen[s], "recomputed": recomputed[s]}
               for s in both if frozen[s] != recomputed[s]]
    return {
        "frozen_count": len(frozen), "recomputed_count": len(recomputed),
        "unchanged": len(both) - len(changed), "changed": changed,
        "dropped": sorted(set(frozen) - set(recomputed)),
        "added": sorted(set(recomputed) - set(frozen)),
    }


def sizes(tiers: dict[str, str]) -> dict[str, int]:
    out = {f"liq_q{i}": 0 for i in range(1, GROUPS + 1)}
    for tier in tiers.values():
        code = f"liq_q{int(str(tier)[1:])}"
        out[code] = out.get(code, 0) + 1
    return out


def main() -> int:
    raise SystemExit("P18-007 审计口径已更正；禁止重跑并覆盖历史产物，请使用 fit_membership_correction.py")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", action="store_true")
    args = ap.parse_args()

    paths = {name: DOCS / name for name in INPUTS}
    for name, path in paths.items():
        if not path.exists():
            raise SystemExit(f"缺少冻结产物 {name}，先跑 fit_prep / indicator_fit_matrix")
    spec_doc = json.loads(paths["indicator-eval-scope.json"].read_text(encoding="utf-8"))
    uni_doc = json.loads(paths["fit-universe.json"].read_text(encoding="utf-8"))
    stats = json.loads(paths["indicator-fit-stats-A.json"].read_text(encoding="utf-8"))
    spec, uni = spec_doc["A"], uni_doc["scopes"]["A"]
    if spec["sha"] != stats["sha"]:
        raise SystemExit(f"scope sha 不一致: spec={spec['sha']} stats={stats['sha']}")
    ref = list(uni["common_universe"])
    frozen_tiers = dict(uni.get("fixed_tiers") or {})
    frozen_snapshot = {c: list((uni.get("pool_snapshot") or {}).get(c, [])) for c in Q_CODES}

    engine = make_engine(read_only=True)
    db = sessionmaker(bind=engine)()
    try:
        read_only = db.execute(text("SHOW transaction_read_only")).scalar()
        if read_only != "on":
            raise RuntimeError("research connection must be read-only")
        db.execute(text("SET LOCAL statement_timeout = '120s'"))

        pool_rows = db.execute(text(
            "SELECT code, kind, enabled, symbols, rule, updated_at FROM symbol_pools "
            "WHERE code IN :codes ORDER BY code"
        ).bindparams(**{"codes": tuple(Q_CODES + RULE_CODES)})).mappings().all()
        pools: dict[str, dict] = {}
        for r in pool_rows:
            pools[r["code"]] = {
                "kind": r["kind"], "enabled": bool(r["enabled"]),
                "symbols": json.loads(r["symbols"]) if r["symbols"] else [],
                "rule": json.loads(r["rule"]) if r["rule"] else {},
                "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
            }
        uni["pool_rules"] = {c: p["rule"] for c, p in pools.items()}
        # 锚来自可变的 symbol_pools 表，所以必须先确认 Q 池成员与冻结快照逐集合一致，
        # 否则这个 window_end_ms 未必还是 A 轮那一次的时点。
        q_drift = membership_drift(
            frozen_snapshot, {c: pools.get(c, {}).get("symbols", []) for c in Q_CODES})
        anchor_trusted = all(d["status"] == "MATCH" for d in q_drift.values())
        anchor = frozen_window_end(uni)
        anchor["q_snapshot_trusted"] = anchor_trusted
        if anchor["end_ms"] is None:
            raise SystemExit("冻结产物里没有 window_end_ms，无法复现 A 轮窗口")
        if not anchor_trusted:
            raise SystemExit(f"Q 池成员已漂移，window_end_ms 不可信: {q_drift}")
        end_ms = int(anchor["end_ms"])
        pool_days = int(SCOPES["A"]["pool_window_days"])
        lo_ms = end_ms - pool_days * fd.DAY_MS
        hist_lo = int((pd.Timestamp(end_ms, unit="ms", tz="UTC")
                       - pd.DateOffset(months=HISTORY_MONTHS)).value // 1_000_000)
        hist_days = int((end_ms - hist_lo) / fd.DAY_MS) + 1

        window = fd.monthly_quote_volume(db, "1h", ref, days=pool_days, end_ms=end_ms)
        history = fd.monthly_quote_volume(db, "1h", ref, days=hist_days, end_ms=end_ms)

        live_members: dict[str, dict[str, list[str]]] = {}
        for iv in spec["intervals"]:
            allowed = list(uni["universe"][iv])
            live_members[iv] = {c: sorted(pool_symbols_by_code(db, c, allowed=allowed))
                                for c in RULE_CODES}
    finally:
        db.close()
        engine.dispose()

    # 1) 溢出复现：partial 恒 False 时能否原样复现冻结档位
    buggy = fd.assign_fixed_tiers(window.assign(partial=False), groups=GROUPS)
    corrected = fd.assign_fixed_tiers(window, groups=GROUPS)
    repro = tier_delta(frozen_tiers, buggy)
    repro_status = "MATCH" if not (repro["changed"] or repro["dropped"] or repro["added"]) else "MISMATCH"
    impact = tier_delta(frozen_tiers, corrected)

    # 2) PIT 档：只用严格早于事件月的完整月。
    #    必须把**完整历史帧**喂进去：prior 月早于事件月，先把帧裁到事件月会把它们全删掉，
    #    结果每个月都变成 NO_PRIOR，看起来像「历史不足」其实是自己把历史切没了。
    event_months = sorted(window["ym"].astype(str).unique())
    pit_all = pit_tiers_by_month(history, groups=GROUPS)
    pit = {ym: pit_all[ym] for ym in event_months if ym in pit_all}
    disagree = tier_disagreement(frozen_tiers, pit)
    # 分离两种效应：上面的 PIT 档用 15 个月长回看，与 A 轮的 2~4 个月短回看本身就不同，
    # 所以 36% 的不一致里混着「回看长度」和「用了未来月」两件事，不能当纯泄漏估计。
    # 这一路对照保持 A 轮**实际使用**的月份集合（残月因溢出也在内），只去掉事件月之后的月，
    # 于是差异就只来自未来信息。
    as_run_all = pit_tiers_by_month(window.assign(partial=False), groups=GROUPS)
    as_run = {ym: as_run_all[ym] for ym in event_months if ym in as_run_all}
    lookahead = tier_disagreement(frozen_tiers, as_run)
    days = month_window_days(lo_ms, end_ms, event_months)
    total_days = sum(days.values())
    no_pit_days = sum(days.get(m, 0.0) for m in disagree["months_without_pit"])

    # 3) 成员漂移（Q 池的比对已在取锚前做过，这里直接复用，避免两处口径分叉）
    rule_drift = {}
    for iv, members in live_members.items():
        sizes_now = stats["intervals"][iv]["scope_sizes"]
        rule_drift[iv] = {
            c: {"live_count": len(members[c]),
                "a_round_count": sizes_now.get(c),
                "count_status": "MATCH" if len(members[c]) == sizes_now.get(c) else "DRIFT",
                "members_frozen_in_a_round": False}
            for c in RULE_CODES
        }

    payload = {
        "generated_at": stamp(), "experiment_id": "P18-007", "parent_experiment_id": "P18-002",
        "supersedes_interpretation_of": "P18-006",
        "registration_note": "P18-006 登记的是同一份测量，但当时尚未分离「回看长度」与「未来信息」两种效应；本条携带完整口径与分离后的泄漏率。台账 append-only，不改写前条。",
        "stage": "S1/S5 pool membership point-in-time audit",
        "decision": "BLOCKED", "db_access": "read_only", "transaction_read_only": read_only,
        "scope_sha": spec["sha"],
        "source_hashes": {name: sha256(p) for name, p in paths.items()},
        "window_anchor": anchor, "pool_window_days": pool_days,
        "window_lo_ms": lo_ms, "history_lookback_months": HISTORY_MONTHS,
        "history_lookback_days": hist_days,
        "reference_symbols": len(ref), "pool_interval": "1h",
        "overflow_bug": {
            "id": "expected_bars_int32_overflow",
            "root_cause": "pandas 3.x .dt.days_in_month 为 int32，乘 DAY_MS(86400000) 对 >=25 天的月份溢出为负",
            "consequence": "partial 恒为 False，窗口首尾残月从未被剔除，直接进入 Q 档中位数与逐月档位",
            "fixed_in": "backend/app/research/fit_data.py:month_expected_bars/mark_partial_months",
            "frozen_artifacts_unchanged": True,
            "a_round_reproduction": repro_status,
            "a_round_reproduction_detail": repro,
        },
        "corrected_recompute": {
            "note": "只用修好的 partial 判定重算同一窗口档位；不落库、不覆盖 fit-universe.json、不重算统计",
            "tier_delta": impact,
            "frozen_pool_sizes": sizes(frozen_tiers),
            "recomputed_pool_sizes": sizes(corrected),
            "window_months": [
                {"ym": str(r.ym), "symbols": int(r.n), "partial_months": int(r.p),
                 "expected_bars_min": int(r.e)}
                for r in window.groupby("ym").agg(
                    n=("symbol", "size"), p=("partial", "sum"), e=("expected_bars", "min")
                ).reset_index().itertuples()
            ],
        },
        "pit": {
            "rule": "只用严格早于事件月且非 partial 的月份定档；不读事件月自身或之后的月",
            "lookback_months_available": HISTORY_MONTHS,
            "confound_note": "长回看 PIT 档与 A 轮短回看固定档的差异同时含「回看长度」与「未来信息」两种效应，不是纯泄漏估计",
            "per_month": disagree["per_month"],
            "comparable_total": disagree["comparable_total"],
            "disagree_total": disagree["disagree_total"],
            "disagree_rate": disagree["disagree_rate"],
            "months_without_pit": disagree["months_without_pit"],
            "window_days_by_month": {k: round(v, 4) for k, v in days.items()},
            "window_days_total": round(total_days, 4),
            "window_days_without_pit": round(no_pit_days, 4),
            "window_share_without_pit": round(no_pit_days / total_days, 4) if total_days else None,
        },
        "lookahead_isolated": {
            "rule": "保持 A 轮实际使用的月份集合（残月因溢出也在内），只去掉事件月之后的月",
            "purpose": "把「未来信息」这一项单独量出来，与回看长度效应分开",
            "per_month": lookahead["per_month"],
            "comparable_total": lookahead["comparable_total"],
            "disagree_total": lookahead["disagree_total"],
            "disagree_rate": lookahead["disagree_rate"],
            "months_without_pit": lookahead["months_without_pit"],
        },
        "membership": {
            "q_pools_static": True,
            "q_drift_vs_frozen_snapshot": q_drift,
            "rule_pools": {c: {"kind": pools.get(c, {}).get("kind"),
                               "rule": pools.get(c, {}).get("rule"),
                               "updated_at": pools.get(c, {}).get("updated_at")}
                           for c in RULE_CODES},
            "rule_pool_a_round_members_persisted": False,
            "rule_pool_count_check": rule_drift,
            "frozen_now": live_members,
            "freeze_caveat": "本次 major/alt 成员是审计时点的事后快照；A 轮成员名单从未落盘，无法逐币复现",
        },
        "costs": {"fee_rate": 0.0004, "slippage": 0.0002, "round_trip": 0.0012,
                  "basis": "notional", "applied_in_this_audit": False,
                  "note": "沿用原统计名义本金往返0.12%，每边fee_rate=0.0004/slippage=0.0002；本审计不新增净收益"},
        "window": {"start_ms": lo_ms, "end_ms": end_ms, "days": pool_days,
                   "anchor_source": anchor["source"],
                   "start_iso": pd.Timestamp(lo_ms, unit="ms", tz="UTC").isoformat(),
                   "end_iso": pd.Timestamp(end_ms, unit="ms", tz="UTC").isoformat()},
        "universe": {"reference_symbols": len(ref), "pool_interval": "1h",
                     "intervals": spec["intervals"],
                     "universe_size": uni["universe_size"],
                     "common_universe": len(ref)},
        "verdict": "insufficient",
        "trades": {"status": "NOT_RUN", "count": 0},
        "ci": {"status": "NOT_APPLICABLE",
               "reason": "本审计不估计任何收益期望，未新增置信区间；档位不一致率是计数比例，不是收益估计"},
        "limitations": [
            "只测量成员口径，不产生净收益数字，不改 A 轮统计与标签。",
            "溢出修复改变此后 prep 的分档口径；A 轮冻结档位与池成员保持原样，未按修复重选。",
            "长回看 PIT 不一致率混合回看长度与未来信息两种效应，不是纯泄漏估计。",
            "PIT 不一致率是档位标签的不一致，不是收益偏差的估计，两者不能互相替代。",
            "残月判定修复后 tier_stability 与逐月档位的历史产物同样受影响，本轮未重写它们。",
            "major/alt 的 A 轮成员名单从未落盘，只能比计数；本次冻结是事后快照。",
            "max-T、先整体交互的校正、整体显著性/稳定性、独立样本外历史仍未补齐。",
        ],
        "backtest_status": "NOT_RUN", "parameter_trials": 0, "stats_recomputed": False,
        "a_round_labels_unchanged": True, "live_orders": False,
        "verification": {},
    }
    for name in ("fit-membership-focused-pytest.log", "fit-membership-full-pytest.log",
                 "fit-membership-independent-verify.log"):
        log = ROOT / "output" / name
        if log.exists():
            payload["verification"][f"output/{name}"] = log.read_text(
                encoding="utf-8").strip().splitlines()[-1]

    lines = [
        "# P18 币种池成员 PIT 口径审计", "",
        f"生成：{payload['generated_at']}。结论：**BLOCKED**（成员口径不满足预登记的时点要求）。", "",
        "只读数据库；不重算统计、不改 A 轮标签、不重选指标/周期/池/h，参数试验 0。",
        f"窗口末端锚取自冻结 Q 池 rule 的 `window_end_ms`：{anchor['iso']}，回看 {pool_days} 天；"
        f"PIT 历史回看 {HISTORY_MONTHS} 个自然月（{hist_days} 天），与就绪审计同一条预登记要求。", "",
        "## 1. 发现一个真实缺陷：残月标记因 int32 溢出恒为 False", "",
        "`fit_data.monthly_quote_volume` 内联算应有柱数时用 `.dt.days_in_month`（pandas 3.x 为 int32）"
        "乘 `DAY_MS=86400000`。30 天 → 2.59e9 超过 int32 上限，结果翻负（-474 根），"
        "于是 `bars < 0.7 × 负数` 恒为 False，**partial 全库失效**。", "",
        "后果：窗口首尾只盖到十几天的残月从未被剔除，直接进入 Q 档的月度成交额中位数，"
        "也进入逐月档位与档位稳定率。这与 `assign_fixed_tiers` docstring 声明的口径相反。", "",
        f"- 已修：`month_expected_bars` 先升 int64 再乘；`mark_partial_months` 统一判定。",
        f"- 用溢出仍生效的口径复现 A 轮冻结档位：**{repro_status}**"
        f"（不变 {repro['unchanged']}、改档 {len(repro['changed'])}、"
        f"掉出 {len(repro['dropped'])}、新增 {len(repro['added'])}）。",
        "- 冻结产物一律未改写；修复只影响此后的 prep 运行。", "",
        "### 同一窗口用修好的判定重算（仅测量，不落库）", "",
        f"- 档位不变 {impact['unchanged']} 币；改档 {len(impact['changed'])} 币；"
        f"掉出 {len(impact['dropped'])} 币；新增 {len(impact['added'])} 币。",
        f"- 冻结池规模 {payload['corrected_recompute']['frozen_pool_sizes']}；"
        f"重算池规模 {payload['corrected_recompute']['recomputed_pool_sizes']}。",
        "- 池规模相同是**结构性的**：`assign_fixed_tiers` 按 `ceil(n/4)` 切档，80 币必然 20/20/20/20。"
        "规模不变不代表成员不变——上面 28 个币换了档，等于四个池的成员都被重排过。", "",
        "| 事件月 | 币数 | 残月数 | 应有柱数(最小) |", "|---|---:|---:|---:|",
    ]
    for m in payload["corrected_recompute"]["window_months"]:
        lines.append(f"| {m['ym']} | {m['symbols']} | {m['partial_months']} | {m['expected_bars_min']} |")
    if impact["changed"]:
        lines += ["", "| 币 | 冻结档 | 重算档 |", "|---|---|---|"]
        for c in impact["changed"]:
            lines.append(f"| {c['symbol']} | {c['frozen']} | {c['recomputed']} |")
    if impact["dropped"]:
        lines += ["", f"重算后掉出分档的币（无完整月）：{', '.join(impact['dropped'])}。"]
    lines += [
        "", "## 2. 时点泄漏：事后固定档 vs PIT 档", "",
        "PIT 档只用**严格早于事件月**且完整的月份；没有可用 prior 月时显式记 "
        "`NO_PRIOR_COMPLETE_MONTH`，不拿未来月补齐。", "",
        "| 事件月 | 窗口天数 | prior 完整月 | 可比币 | 不一致 | 不一致率 | 无 PIT 档币 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for ym, m in disagree["per_month"].items():
        prior = ", ".join(m["prior_months"]) or "—"
        rate = "—" if m["disagree_rate"] is None else f"{m['disagree_rate']:.2%}"
        lines.append(f"| {ym} | {days.get(ym, 0):.2f} | {prior} | {m['comparable']} | "
                     f"{m['disagree']} | {rate} | {m['no_pit_tier']} |")
    share = payload["pit"]["window_share_without_pit"]
    lines += [
        "", f"- 合计可比 {disagree['comparable_total']} 币月，不一致 {disagree['disagree_total']}，"
        f"不一致率 {'—' if disagree['disagree_rate'] is None else format(disagree['disagree_rate'], '.2%')}。",
        f"- 评估窗共 {total_days:.2f} 天，其中 {no_pit_days:.2f} 天"
        f"（{'—' if share is None else format(share, '.2%')}）落在**整月无从 PIT 定档**的月份；"
        "另有逐币的新上市币在事件月之前无量，单独计在「无 PIT 档币」列。",
        "", "**这个不一致率不是纯泄漏估计**：长回看 PIT 档用 10–13 个完整月，A 轮固定档只用窗口内"
        "2 个完整月＋2 个残月，两者回看长度本身就不同。差异里同时含「回看长度」与「用了未来月」"
        "两种效应，不能把它读成未来信息单独造成的偏差。",
        "", "### 2.1 分离未来信息：同窗口、只去掉事件月之后的月", "",
        "保持 A 轮**实际使用**的月份集合（残月因溢出也在内），只删掉事件月之后的月。"
        "这样回看口径与 A 轮一致，差异就只来自未来信息。", "",
        "| 事件月 | 可用 prior 月 | 可比币 | 不一致 | 不一致率 | 无档币 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for ym, m in lookahead["per_month"].items():
        prior = ", ".join(m["prior_months"]) or "—"
        rate = "—" if m["disagree_rate"] is None else f"{m['disagree_rate']:.2%}"
        lines.append(f"| {ym} | {prior} | {m['comparable']} | {m['disagree']} | {rate} | "
                     f"{m['no_pit_tier']} |")
    lines += [
        "", f"- 合计可比 {lookahead['comparable_total']} 币月，不一致 {lookahead['disagree_total']}，"
        f"不一致率 {'—' if lookahead['disagree_rate'] is None else format(lookahead['disagree_rate'], '.2%')}。",
        f"- 整月无 prior 可用的月份：{', '.join(lookahead['months_without_pit']) or '无'}。",
        "", "两路对照合起来说明：Q 池分层不是「事件时点可得的属性」。A 轮档位确实依赖事件之后的月份，"
        "而在 A 轮自己的窗口口径下，最早的事件月连一个 prior 月都没有。",
        "补足窗口之前的历史可以让 PIT 分档可得，但那是新的数据边界，不能倒灌成 A 轮已通过。",
        "", "## 3. 成员冻结与漂移", "",
        "Q1–Q4 是 static 池，成员已随 `fit-universe.json` 冻结；逐集合比对结果：", "",
        "| 池 | 冻结币数 | 当前币数 | 状态 | 新增 | 移除 |", "|---|---:|---:|---|---|---|",
    ]
    for code, d in q_drift.items():
        lines.append(f"| {code} | {d['frozen_count']} | {d['live_count']} | {d['status']} | "
                     f"{', '.join(d['added']) or '—'} | {', '.join(d['removed']) or '—'} |")
    lines += [
        "", "major/alt 是 **rule 池**（`base: major` / `base: all + exclude_major`），"
        "统计阶段实时解析数据库宇宙与成交额，成员随线上状态变化。",
        "**A 轮从未落盘这两个池的成员名单**，产物里只有 `scope_sizes` 计数，"
        "因此无法逐币复现 A 轮篮子——这是可复现性缺口，不是「已核对一致」。", "",
        "| 周期 | 池 | A 轮计数 | 当前计数 | 计数状态 | A 轮成员已落盘 |",
        "|---|---|---:|---:|---|---|",
    ]
    for iv, per in rule_drift.items():
        for code, d in per.items():
            lines.append(f"| {iv} | {code} | {d['a_round_count']} | {d['live_count']} | "
                         f"{d['count_status']} | 否 |")
    lines += [
        "", "本次已把 major/alt 的当前解析成员写进本产物 `membership.frozen_now`，"
        "供此后复现使用；它是**审计时点的事后快照**，不能冒充 A 轮篮子。", "",
        "## 4. 局限与启用决策", "",
        "判定词：**insufficient**——现有证据不足以支持「Q 池成员是事件时点可得的属性」，"
        "不是收益为负的判定。", "",
    ]
    lines += [f"- {item}" for item in payload["limitations"]]
    lines += [
        "", "成本沿用原统计名义本金往返 0.12%（fee_rate=0.0004 / slippage=0.0002，每边），"
        "本审计不新增净收益，也未对该口径做任何策略级覆盖。",
        "未跑交易回测、未新增置信区间 CI 或随机基线（本审计不估计收益期望，CI 不适用）；"
        "default_enabled / 绑定不改，未启用 paper/live。", "",
        "## 5. 验证", "",
    ]
    lines += [f"- `{name}`：{result}" for name, result in payload["verification"].items()]
    lines += ["", "复现：`.venv/Scripts/python.exe backend/scripts/fit_membership_audit.py`（只读数据库）。",
              "独立复算（不复用本脚本的实现，直接 SQL + 手写中位数/切档）："
              "`.venv/Scripts/python.exe backend/scripts/fit_membership_verify.py`。", ""]

    (DOCS / "indicator-fit-membership.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (DOCS / "indicator-fit-membership.md").write_text("\n".join(lines), encoding="utf-8")

    if args.ledger:
        from fit_matrix_report import append_ledger, git_state
        record = {
            "timestamp": payload["generated_at"], "experiment_id": "P18-007",
            "parent_experiment_id": "P18-002", "strategy_id": None,
            "supersedes_interpretation_of": "P18-006",
            "code_commit": git_state(), "stage": payload["stage"],
            "decision": "BLOCKED_MEMBERSHIP_NOT_PIT",
            "dataset_version": "local PostgreSQL klines read_only; frozen A-round window anchor",
            "universe_snapshot": "docs/fit-universe.json (A); major/alt frozen post-hoc in this artifact",
            "interval": spec["intervals"],
            "parameters": {"scope_sha": spec["sha"], "parameter_trials": 0,
                           "grid_search_performed": False, "stats_recomputed": False,
                           "history_lookback_months": HISTORY_MONTHS,
                           "pool_window_days": pool_days},
            "fee_model": "0.0004 per side", "slippage_model": "0.0002 per side",
            "funding_model": "NOT_MODELLED; no backtest executed",
            "metrics": {
                "a_round_tier_reproduction": repro_status,
                "corrected_tier_changed": len(impact["changed"]),
                "corrected_tier_dropped": len(impact["dropped"]),
                "pit_disagree_rate": disagree["disagree_rate"],
                "pit_disagree_confound": "lookback length + future months; not a pure lookahead estimate",
                "lookahead_isolated_disagree_rate": lookahead["disagree_rate"],
                "lookahead_isolated_months_without_prior": lookahead["months_without_pit"],
                "window_share_without_pit": share,
                "q_pool_drift": {c: d["status"] for c, d in q_drift.items()},
            },
            "failure_reason": "pool tiers use post-hoc window months; int32 overflow disabled partial-month filter; major/alt members never persisted",
            "next_action": [
                "把 major/alt 成员名单纳入 scope 冻结，统计阶段只读冻结快照，不再实时解析 rule 池。",
                "补足事件月之前的完整月历史，使 PIT 分档在评估窗内可得；不得用未来月补齐。",
                "在新 scope 下用修复后的 partial 判定重新 prep，并重新走全部统计门；A 轮产物保持冻结。",
            ],
            "verification": payload["verification"],
            "artifacts": ["docs/indicator-fit-membership.md", "docs/indicator-fit-membership.json"],
        }
        print(append_ledger(record, ROOT / "research" / "experiments.jsonl"))
    print(f"membership=BLOCKED; a_round_repro={repro_status}; "
          f"corrected_changed={len(impact['changed'])} dropped={len(impact['dropped'])}; "
          f"pit_disagree_rate={disagree['disagree_rate']}; "
          f"window_share_without_pit={share}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())