"""报告渲染层的测试：排版不许新增口径，也不许把实测数字写成乘法结论。

起因是 §1 那行「列数 × 池 × h × 基准 = 行数」：退化列不进事件表、双向列各计一行，
简单乘积两头都对不上，但 md 里读起来像真的一样。这里把「渲染出的数必须等于产物里的数」
钉成用例，顺手钉住池特有表的筛选条件（漏了后半句就成了复读）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
import fit_matrix_report as fmr  # noqa: E402

from app.research.fit_stats import PROMOTABLE  # noqa: E402

IV = "1h"


def row(key: str, scope: str, verdict_name: str, **over) -> dict:
    r = {
        "key": key, "indicator": key.split("|")[0], "column": key.split("|")[1],
        "scope": scope, "horizon_bars": 24, "benchmark": "unconditional",
        "direction": 1, "direction_source": "declared", "events": 500,
        "symbols_with_events": 30, "symbols_in_stat": 30,
        "mean_gross_pct": 1.0, "mean_net_pct": 0.9, "ci_low_pct": 0.3, "ci_high_pct": 1.5,
        "q_value": 0.02, "seg_consistency": 1.0, "symbol_sign_rate": 0.8,
        # 与 `fit_stats` 的实际产物一致：成本压力是 {x1,x2,x3} 字典，不是单个数字
        "cost_stress_pct": {"x1": 0.9, "x2": 0.8, "x3": 0.7},
        "hit_rate": 0.05, "verdict": verdict_name,
        "verdict_cell": verdict_name,
    }
    r.update(over)
    return r


def stats_doc(rows: list[dict], **over) -> dict:
    iv_doc = {
        "columns": 3, "symbols": 30, "scope_sizes": {"all": 30, "liq_q2": 10},
        "rows": rows, "rows_total": len(rows), "rows_kept": len(rows),
        "columns_promotable": 1,
        "verdict_counts": {}, "redundancy": {},
    }
    for r in rows:
        iv_doc["verdict_counts"].setdefault(r["scope"], {})
        iv_doc["verdict_counts"][r["scope"]][r["verdict"]] = (
            iv_doc["verdict_counts"][r["scope"]].get(r["verdict"], 0) + 1)
    doc = {
        "generated_at": "2026-11-13T00:00:00+00:00", "scope": "T", "sha": "sha-t",
        "horizons_bars": [24],
        "config": {"round_trip": 0.0012, "blocks_for_candidate": 15, "q_alpha": 0.1,
                   "interaction_alpha": 0.1, "shrink_k": 30.0, "redundancy_r_min": 0.8},
        "intervals": {IV: iv_doc},
    }
    doc.update(over)
    return doc


DIRS = {"count": 3, "by_source": {"declared": 3}, "by_direction": {"bull": 3},
        "indicator_accounting": {"registered": 10, "with_boolean_columns": 8,
                                 "no_boolean_columns": 1, "raised": 1},
        "columns": [{"key": "a|x", "direction": 1, "direction_source": "declared"}]}


PROBE = {"interval": IV, "symbols": ["S1USDT", "S2USDT"], "median_seconds_per_bar": 0.048,
         "bars_total_by_interval": {IV: 1000}, "estimate_cpu_hours": 1.0,
         "estimate_wall_hours_at_workers": {10: 0.1, 12: 0.08},
         "columns_expected_from_directions": 3}


def render(rows, stats=None, extra=None, events=None):
    s = fmr.summarize(stats or stats_doc(rows), DIRS, {"directions_sha": "d", "intervals": {}})
    fmr.attach_prep(s, {"window_days": {IV: 90}, "gates": {"hit_rate_range": [0.002, 0.6]},
                        "provenance": {"rule": "native 优先"}, "data_gates": {},
                        "pools": {}, "entry": "t+1 open"},
                    {"pool_sizes": {"liq_q2": 10}, "monthly_liquidity_bins": {1: 1},
                     "tier_stability": {"liq_q2": 0.8}, "universe_size": {IV: 30}}, PROBE)
    s.update(extra or {})
    return s, fmr.render_md(s, events or {"intervals": {}}, PROBE, {}, s)


def test_row_counts_are_measured_not_multiplied():
    rows = [row("a|x", "all", "candidate"), row("a|x", "liq_q2", "candidate",
                                                shrunk_net_pct=0.7, pooled_verdict="candidate")]
    s, md = render(rows)
    # 声明 3 列、实际只有 1 列进统计、实测 2 行：渲染层必须照抄这三个实测值，
    # 不许写「3 列 × 2 池 × 1 h × 2 基准 = 12 行」这类看着自洽的乘法。
    assert "方向表声明 3 个布尔条件列，其中 1 列进入统计" in md
    assert "合计 2 行" in md and "每格 1 行" in md
    for wrong in ("= 12 行", "12 行判定", "× 2 周期 × 8 池"):
        assert wrong not in md


def test_pool_specific_requires_pooled_not_candidate():
    rows = [
        # 池级 candidate + 整体 candidate：只是复读，不算池特有
        row("a|x", "all", "candidate"),
        row("a|x", "liq_q2", "candidate", pooled_verdict="candidate",
            pooled_mean_net_pct=0.9),
    ]
    s, md = render(rows)
    assert s["pool_specific_columns"] == 0 and "本轮没有池特有结论" in md
    rows = [
        row("b|y", "all", "inconclusive", mean_net_pct=0.2, ci_low_pct=-0.1,
            verdict="inconclusive", verdict_cell="inconclusive"),
        row("b|y", "liq_q2", "candidate", pooled_verdict="inconclusive",
            pooled_mean_net_pct=0.2, pooled_ci_low_pct=-0.1),
    ]
    s, md = render(rows)
    assert s["pool_specific_columns"] == 1
    assert "b.y" in md and "inconclusive" in md


def test_missing_redundancy_threshold_is_reported_as_missing():
    rows = [row("a|x", "all", "candidate")]
    stats = stats_doc(rows)
    stats["config"].pop("redundancy_r_min")
    _, md = render(rows, stats=stats)
    assert "未记录" in md


def test_indicator_skipped_comes_from_directions_accounting():
    rows = [row("a|x", "all", "candidate")]
    s, md = render(rows)
    assert s["indicators_skipped"] == 2           # 10 注册 - 8 出布尔列
    assert "跳过 2 个指标注册项" in md
    assert "None 个" not in md                    # 曾经的读数：探针里没有 skipped 键


def test_interval_tag_is_attached_to_rows():
    """每列的最强格是跨周期取的——行上没有 interval 就答不出「适合哪个 K 线」。"""
    rows = [row("a|x", "all", "candidate"),
            row("c|z", "all", "no_edge", mean_net_pct=-0.5, ci_low_pct=-1.0,
                verdict="no_edge", verdict_cell="no_edge")]
    s, _ = render(rows)
    assert {r["interval"] for r in s["best_cells"]} == {IV}
    assert s["per_interval"][IV]["verdict_counts_by_scope"]["all"] == {"candidate": 1,
                                                                       "no_edge": 1}


def two_interval_stats() -> dict:
    """两个周期、池规模不同的产物：30m 的合格币数本来就少于 1h（各自过数据门）。"""
    doc = stats_doc([row("a|x", "all", "candidate")])
    doc["intervals"][IV] = {**doc["intervals"][IV], "symbols": 119,
                            "scope_sizes": {"all": 119, "liq_q2": 10}}
    doc["intervals"]["30m"] = {**doc["intervals"][IV], "symbols": 99,
                               "scope_sizes": {"all": 99, "liq_q2": 9},
                               # summarize() 会往行上盖 interval 标签，两个周期不能共用同一批对象
                               "rows": [row("a|x", "all", "candidate")]}
    return doc


def test_pool_sizes_are_per_interval_not_first_interval_only():
    """池成员是按周期各自过数据门的，只报 ivs[0] 会让另一个周期对不上号。"""
    s, md = render([], stats=two_interval_stats())
    assert s["intervals"] == ["30m", "1h"]
    assert "| 币种池 | 30m 币数 | 1h 币数 |" in md
    assert "| 全样本 | 99 | 119 |" in md
    assert "| 全样本 | 30m 99 / 1h 119 |" in md          # §4 同一口径
    assert "币种池：全样本 99 币" not in md               # 旧写法：只有第一个周期的数
    assert "30m 币数" in md and "1h 币数" in md


def test_suffix_is_normalized_with_a_dash():
    """`--suffix dev` 直接拼名会写出 indicator-fit-matrixdev.md，看着像正式交付。"""
    assert fmr.norm_suffix("dev") == "-dev"
    assert fmr.norm_suffix("-dev") == "-dev"
    assert fmr.norm_suffix("") == ""


def test_indicator_rollup_counts_only_promotable_columns():
    """§3.2 是「哪个指标适合哪个周期」的直接答案：判负的指标不许混进列数。"""
    rows = [row("adx_advanced|x", "all", "candidate"),
            row("adx_advanced|y", "liq_q2", "exploratory"),
            row("noise_ind|z", "all", "no_edge", mean_net_pct=-0.5, ci_low_pct=-1.0,
                verdict="no_edge", verdict_cell="no_edge")]
    s, md = render(rows)
    assert s["columns_promotable"] == 2 and s["indicators_promotable"] == 1
    e = s["indicator_rollup"][0]
    assert e["indicator"] == "adx_advanced" and e["columns"] == 2
    assert e["by_interval"] == {IV: 2}
    assert e["by_scope"] == {"all": 1, "liq_q2": 1}
    assert e["best_verdict"] == "candidate"        # 数字越小越强，取最强那一格
    assert "| 指标 | 列数 | 1h 列 | 最强判定 | 最强格落点（池） |" in md
    assert "`adx_advanced`" in md and "noise_ind" not in md




def test_artifact_table_names_the_files_actually_written():
    """`--stats` / `--suffix` 是自检通道：§10 若还印正式文件名，就是把自检产物登记成交付物。"""
    s, md = render([row("a|x", "all", "candidate")], extra={
        "stats_file": "indicator-fit-stats-A-pre3.json",
        "out_md": "indicator-fit-matrix-pre3.md",
        "out_json": "indicator-fit-matrix-pre3.json"})
    for name in ("indicator-fit-stats-A-pre3.json", "indicator-fit-matrix-pre3.md",
                 "indicator-fit-matrix-pre3.json"):
        assert name in md
    assert "`docs/indicator-fit-stats-A.json`" not in md
    # 没覆盖时回到「按 scope 拼出的正式名」，且绝不残留上一轮的自检名
    s0, md0 = render([row("a|x", "all", "candidate")])
    assert f"`docs/indicator-fit-stats-{s0['scope']}.json`" in md0
    assert "indicator-fit-stats-A-pre3.json" not in md0
    rec = fmr.ledger_record(s, {"intervals": {}}, "P17-001", "x", "")
    assert "docs/indicator-fit-stats-A-pre3.json" in rec["artifacts"]


def test_cpu_estimate_is_reconciled_with_measured_wall_clock():
    """§7 不许只留一句外推：产物里有实测墙钟就得对账，并把「只在一个周期实测过」讲清楚。"""
    rows = [row("a|x", "all", "candidate")]
    _, md = render(rows, events={"elapsed_s": 396.0, "intervals": {}})
    assert "实测对账" in md and "396 s" in md and "+10%" in md
    assert "即探针外推口径的全部" in md          # 覆盖满口径时不再写「占 100%」这种废话
    # 没有实测耗时（单周期调试产物）时不能硬造一句对账
    _, md0 = render(rows)
    assert "实测对账" not in md0


def test_audit_limits_candidate_and_interval_claims():
    s, md = render([row("a|x", "all", "candidate")])
    assert s["inference_status"] == "EXPLORATORY_ONLY"
    assert "max-T 尚未实施" in md
    assert "事后固定分档" in md
    assert "不能单独证明周期适配" in md
    assert "过全部五道关的" not in md


def test_verified_log_diagnosis_closes_only_the_warning_issue():
    _, md = render([row("a|x", "all", "candidate")], extra={
        "log_diagnosis": {"diagnostic_status": "PASS", "metrics": {
            "symbols_checked": 80, "zero_log_input_bars": 7}}})
    assert "log 告警已定位并完成工程复核" in md
    assert "indicator-fit-log-diagnosis.md" in md
    assert "不能单独证明周期适配" in md
    assert "影响方向与范围尚未确认" not in md


def test_unattributed_log_warning_does_not_assert_signal_safety():
    _, md = render([row("a|x", "all", "candidate")])
    assert "影响方向与范围尚未确认" in md
    assert "恒为 False" not in md
    assert "不会凭空造出信号" not in md
