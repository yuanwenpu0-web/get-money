"""H-F Gate 1a 成对报告渲染（判据装配见 funding_venue_report.py）。

报告只呈现毛超额；全宇宙毛年化在标题下方即标注为**不可执行的反事实**，
防止读者把 D3 增量当成收益预期。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat(timespec="seconds")


def _gate_text(gate: dict) -> str:
    marks = " · ".join(f"{k} {'过' if gate[k] else '挂'}" for k in ("D1", "D2", "D3"))
    return f"{marks} → {'pass' if gate['pass'] else 'not pass'}"


def _fmt(value, suffix=""):
    return "—" if value is None else f"{value}{suffix}"


def _interval(ci, low, high):
    if ci.get(low) is None or ci.get(high) is None:
        return "—"
    return f"[{ci[low]}, {ci[high]}]"


def render(payload: dict) -> str:
    data, uni, spot = payload["data"], payload["universe"], payload["spot_snapshot"]
    lines = ["# P11 / H-F · 资金费率多所路径 Gate 1a（毛超额可行性）", ""]
    add = lines.append
    add(f"- 判定：`{payload['verdict']}`")
    add(f"- 判定范围：{payload['verdict_scope']}")
    add(f"- 生成时间：{payload['generated_at']} · 代码基线 `{payload['code_commit']}`")
    add(f"- 假设卡：`{payload['hypothesis_card']}`（判据事前冻结，未事后改口径）")
    add("- 授权边界：只读回测/探针，不下单、不接真实 API key、`default_enabled=False`、无绑定。")
    add("")
    add("> **警告：本报告的全宇宙毛年化是不可执行的反事实，不是收益预期。**")
    add("> 它假设那些不可对冲币能在另一家交易所以币安级成本买到现货腿——而这正是 Gate 1b "
        "要验证的未知量。单所口径的同一反事实（净 +41.8%）已被终判为不可执行。")
    add("> Gate 1a 只回答一个问题：**奖金够不够大，值不值得去接第二家所**。")
    add("")
    add("## 1. 数据与宇宙")
    add("")
    add(f"- `funding_rates` {data['rows']:,} 行 · {data['symbols']} 币 · "
        f"{_iso(data['start_ms'])} → {_iso(data['end_ms'])}（{data['days']:.1f} 天）· "
        f"`rate_type` = {'/'.join(data['rate_types'])} · 只读连接")
    ev = data["events_per_symbol"]
    add(f"- 逐币结算事件数 min {ev['min']} / 中位 {ev['median']} / max {ev['max']}"
        " → 结算周期非统一 8h，前瞻窗按实际事件计数")
    add(f"- 现货快照：`{spot['source']}` · {spot['fetched_at']} · "
        f"{spot['symbols_total']} 交易对，TRADING {spot['trading']}，"
        f"去重基础资产 {spot['distinct_trading_bases']}"
        f"{'（读缓存）' if spot.get('from_cache') else ''}")
    hist = uni["historical_reference"]
    add(f"- 可对冲 **{uni['hedgeable_count']}** / 不可对冲 **{uni['non_hedgeable_count']}**"
        f"（合计 {data['symbols']}）；{hist['as_of']} 历史口径为 {hist['hedgeable']}"
        f"/{hist['total']}，差 {uni['spot_drift_vs_2026_10_27']} 币来自现货上架漂移")
    add(f"- 重平衡 {payload['periods']['count']} 期 · "
        f"{_iso(payload['periods']['first_ms'])} → {_iso(payload['periods']['last_ms'])}")
    add("")
    add("## 2. 冻结口径")
    add("")
    proto = payload["protocol"]
    add(f"- 回看 {proto['lookback_days']} 天 · 持有 {proto['hold_days']} 天 · "
        f"K ∈ {proto['ks']} · {proto['gate_rule']}")
    add(f"- 选币：{proto['selection']}")
    add(f"- 前瞻：{proto['forward']} · {proto['weighting']}")
    add(f"- **不报净收益**：`costs_applied={proto['costs_applied']}`，"
        f"参照 fee_rate={proto['fee_rate_reference']} / slippage={proto['slippage_reference']}"
        "（他所真实成本未知，报净就是编数）")
    add("")
    add("## 3. D1–D3 判据")
    add("")
    th = payload["thresholds"]
    add("| 判据 | 门槛 | K=10 | K=20 | 通过 |")
    add("|---|---|---:|---:|---|")
    a10, a20 = payload["arms"]["10"], payload["arms"]["20"]
    rows = [
        ("D1 不可对冲币奖金占比", f"≥ {th['D1_min_share']:.0%}",
         _fmt(a10["bonus"]["pooled_share"]), _fmt(a20["bonus"]["pooled_share"]), "D1"),
        ("D2 子集 rank IC 均值", f"≥ {th['D2_min_ic']}",
         _fmt(a10["ic_non_hedgeable"]["mean"]), _fmt(a20["ic_non_hedgeable"]["mean"]), "D2"),
        ("D2 子集 IC 正期占比", f"≥ {th['D2_min_positive_share']:.0%}",
         _fmt(a10["ic_non_hedgeable"]["positive_share"]),
         _fmt(a20["ic_non_hedgeable"]["positive_share"]), "D2"),
        ("D3 加回子集毛年化增量", f"≥ {th['D3_min_annual_pp']}pp",
         _fmt(a10["d3_increment_pp"], "pp"), _fmt(a20["d3_increment_pp"], "pp"), "D3"),
    ]
    for label, gate, v10, v20, key in rows:
        ok = payload["gates"]["10"][key] and payload["gates"]["20"][key]
        add(f"| {label} | {gate} | {v10} | {v20} | {'是' if ok else '**否**'} |")
    add("")
    add(f"- 综合：K=10 {_gate_text(payload['gates']['10'])} · "
        f"K=20 {_gate_text(payload['gates']['20'])}")
    add("")
    add("95% CI（移动块 bootstrap，见第 7 节口径说明）：")
    add("")
    add("| 量 | K=10 | K=20 |")
    add("|---|---|---|")
    for label, getter in (
        ("D1 奖金占比 CI", lambda a: _interval(a["bonus"]["share_ci"], "ci_low", "ci_high")),
        ("D2 子集 IC 均值 CI", lambda a: _interval(a["ic_non_hedgeable"]["ci"], "ci_low", "ci_high")),
        ("D3 增量 CI（pp）", lambda a: _interval(a["d3_ci"], "ci_low_pp", "ci_high_pp")),
    ):
        add(f"| {label} | {getter(a10)} | {getter(a20)} |")
    add("")
    add("## 4. 毛年化与奖金分解")
    add("")
    add("| 臂 | K | 有效期数 | 毛年化 | 期均毛费率 | 期均 95% CI | 正期占比 |")
    add("|---|---:|---:|---:|---:|---|---:|")
    for key in ("10", "20"):
        for arm, label in (("full", "全宇宙 TopK"), ("hedgeable_only", "仅可对冲 TopK")):
            row = payload["arms"][key][arm]
            add(f"| {label} | {key} | {row['n_periods']} | "
                f"{_fmt(row['annual_gross_pct'], '%')} | {_fmt(row['mean_period_pct'], '%')} | "
                f"{_interval(row['ci'], 'ci_low_pct', 'ci_high_pct')} | "
                f"{_fmt(row['positive_period_share'])} |")
    add("")
    cross = []
    for key in ("10", "20"):
        for arm, label in (("full", "全宇宙 TopK"), ("hedgeable_only", "仅可对冲 TopK")):
            ci = payload["arms"][key][arm]["ci"]
            low, high = ci["ci_low_pct"], ci["ci_high_pct"]
            if low is not None and high is not None and low <= 0 <= high:
                cross.append(f"{label}（K={key}）")
    if cross:
        add(f"- **期均毛费率 95% CI 跨 0**：{'、'.join(cross)}"
            " —— 这些臂的毛费率与 0 不可区分，点估计为正不构成证据。")
    for key in ("10", "20"):
        arm = payload["arms"][key]
        bonus, full = arm["bonus"], arm["full"]
        add(f"- K={key}：奖金池化合计 {bonus['pooled_non_hedgeable']} / {bonus['pooled_total']}"
            f" = {_fmt(bonus['pooled_share'])}，逐期中位 {_fmt(bonus['period_median_share'])}，"
            f"TopK 中不可对冲币均值 {_fmt(bonus['mean_non_hedgeable_in_topk'])} 个")
        add(f"- K={key}：选中币缺前瞻 {full['missing_forward']} 币·期（已剔除，不计 0）；"
            f"分母非正的期数 {bonus['periods_nonpositive_total']}")
    add("")
    prior = payload["historical_single_venue"]
    add(f"- 量级对照（**不可直接比较**）：历史单所口径 {prior['as_of']} K={prior['k']} 为"
        f"未过滤毛年化 {prior['unfiltered_annual_pct']}%、现货可对冲毛年化 "
        f"{prior['hedgeable_annual_pct']}%，可对冲**净**年化 −2.4% 终判 FAIL"
        f"（见 `{prior['source']}`）。本次窗口 {data['days']} 天、回看 "
        f"{payload['protocol']['lookback_days']} 天/持有 {payload['protocol']['hold_days']} 天"
        "与当时口径不完全一致，此处只作量级参照，不是复现。")
    add("")
    add("## 5. 随机选币基线（不是随机方向）")
    add("")
    add("同期同 K，从 trailing>0 的合格币里**随机**挑，重复多种子；"
        "用于分离「排序有用」与「随便拿正费率币就有钱」。")
    add("")
    add("| K | 种子数 | 基线毛年化均值 | p05 | 中位 | p95 | 策略分位 |")
    add("|---:|---:|---:|---:|---:|---:|---:|")
    for key in ("10", "20"):
        base = payload["arms"][key]["random_baseline"]
        add(f"| {key} | {base['n_seeds']} | {_fmt(base['mean_pct'], '%')} | "
            f"{_fmt(base['p05_pct'], '%')} | {_fmt(base['median_pct'], '%')} | "
            f"{_fmt(base['p95_pct'], '%')} | {_fmt(base['percentile_of_strategy'])} |")
    add("")
    add(f"- 基线种子基准 {payload['arms']['10']['random_baseline']['seed']}，"
        "逐种子毛年化全量落在 JSON `arms.*.random_baseline.samples_pct`。")
    add("- 分位 = 随机样本中严格低于策略毛年化的比例；>0.95 才算跑赢随机选币（H6）。")
    add("")
    add("## 6. 横截面 rank IC")
    add("")
    add("| 宇宙 | K | 均值 | 正期占比 | 有效期数 | 最小横截面 |")
    add("|---|---:|---:|---:|---:|---:|")
    for key in ("10", "20"):
        arm = payload["arms"][key]
        for field, label in (("ic_non_hedgeable", "不可对冲子集"),
                             ("ic_hedgeable", "仅可对冲"), ("ic_full", "全宇宙")):
            ic = arm[field]
            add(f"| {label} | {key} | {_fmt(ic['mean'])} | {_fmt(ic['positive_share'])} | "
                f"{ic['n_periods']} | {ic.get('min_cross_section', '—')} |")
    add("")
    for key in ("10", "20"):
        ic = payload["arms"][key]["ic_non_hedgeable"]
        add(f"- K={key} 子集 IC：有效 {ic['n_periods']} 期、跳过 {ic['n_skipped']} 期"
            f"（横截面 < {payload['protocol']['min_cross_section']} 币不给读数）")
    add("")
    add("## 7. 局限与诚实边界")
    add("")
    for item in payload["limitations"]:
        add(f"- {item}")
    add("")
    add("## 8. 复现")
    add("")
    add("```powershell")
    add(".\\.venv\\Scripts\\python.exe -m pytest backend/tests/test_funding_venue.py "
        "backend/tests/test_funding_venue_stats.py backend/tests/test_probe_funding_venue.py -q")
    add(".\\.venv\\Scripts\\python.exe backend/scripts/probe_funding_venue.py "
        "--out docs/probe-funding-venue")
    add(".\\.venv\\Scripts\\python.exe "
        ".workbuddy/skills/strategy-backtest-playbook/scripts/preflight.py "
        "--prefix docs/probe-funding-venue")
    add("```")
    add("")
    return "\n".join(lines)


def write_reports(payload: dict, prefix: str) -> tuple[Path, Path]:
    md_path, json_path = Path(f"{prefix}.md"), Path(f"{prefix}.json")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    return md_path, json_path
