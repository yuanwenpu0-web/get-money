"""把 S3–S5 的统计产物渲染成单文件交互矩阵页（docs/indicator-fit-matrix.html）。

页面只做「看」：所有数字都来自 `docs/indicator-fit-stats-<scope>.json`，这里不重新
计算、不修改任何判定。口径以 json 为准，页面只是它的切片浏览器——所以产物里必须同时
留下 md 报告（结论）与 json（数据）与这个 html（交互查看），三者可互相核对。

用法：
    .venv\\Scripts\\python.exe backend/scripts/fit_matrix_page.py --scope A
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
sys.path.insert(0, str(ROOT / "backend"))

# 判定分级与「最能用格子」的取法是判定器的一部分，不在页面里另写一份，否则页面和统计
# 会各自漂移。数字与标签一律来自 app.research.fit_stats。
from app.research.fit_stats import (  # noqa: E402
    PROMOTABLE, VERDICT_RANK, best_cell, intervals_sorted,
)

SCOPE_ORDER = ["all", "common", "major", "alt", "liq_q1", "liq_q2", "liq_q3", "liq_q4"]
SCOPE_LABEL = {
    "all": "全样本", "common": "四周期交集", "major": "主流", "alt": "山寨",
    "liq_q1": "Q1 最高流动", "liq_q2": "Q2", "liq_q3": "Q3", "liq_q4": "Q4 最低流动",
}


def build_payload(scope: str, stats_file: Path | None = None,
                  dirs_file: Path | None = None) -> dict:
    stats_path = stats_file or DOCS / f"indicator-fit-stats-{scope}.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    dirs = json.loads((dirs_file or DOCS / "indicator-directions.json").read_text(encoding="utf-8"))
    probe_path = DOCS / f"fit-probe-{scope}.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.exists() else {}
    # directions_sha 只写在事件产物里：它证明「统计用的方向表」和「抽事件时冻结的那份」
    # 是同一个文件，页面顶部必须显示，否则换过方向表也看不出来。
    events_path = DOCS / f"fit-events-{scope}.json"
    events = json.loads(events_path.read_text(encoding="utf-8")) if events_path.exists() else {}

    # 方向/冗余是「列」这一级的属性，先建成索引，页面里按 key 取。
    col_meta = {c["key"]: c for c in dirs["columns"]}
    # 页面是单文件、数据内联（file:// 下 fetch 同目录 json 会被 CORS 挡住），所以体积
    # 必须压：所有格子按「定长数组」编码而不是对象，判定词表和基准名做字典。
    # 885 列 × 4 周期 × 8 池 = 2.8 万格，用对象写会到 20MB+，压完约 1MB。
    verdicts: list[str] = []
    benches: list[str] = []

    def vi(v: str | None) -> int:
        s = v or "insufficient"
        return verdicts.index(s) if s in verdicts else (verdicts.append(s) or len(verdicts) - 1)

    def bi(v: str | None) -> int:
        s = v or "unconditional"
        return benches.index(s) if s in benches else (benches.append(s) or len(benches) - 1)

    def num(v, nd=4):
        return None if v is None else round(float(v), nd)

    CF = ["v", "h", "bm", "net", "shr", "lo", "q", "n", "seg", "sgn", "dir"]
    DF = ["sc", "h", "bm", "v", "cv", "dir", "n", "gross", "net", "lo", "hi", "q",
          "seg", "sgn", "x2", "ip", "w", "shr"]
    cells: dict[str, dict[str, dict[str, list]]] = {}
    all_rows: dict[str, dict[str, list]] = {}
    blocks = stats.get("intervals") or {}
    # 列次序按 K 线从快到慢（fit_stats.intervals_sorted）。之前用字典序，1h 排在 30m 前面，
    # 「这列适合哪个周期」的矩阵要读者自己在脑子里重排——报告与页面也就会给出两套次序。
    intervals = intervals_sorted(blocks.keys())
    for interval in intervals:
        blk = blocks[interval] or {}
        by_key: dict[tuple, list[dict]] = {}
        for r in blk["rows"]:
            by_key.setdefault((r["key"], r["scope"]), []).append(r)
            it = r.get("interaction") or {}
            cs = r.get("cost_stress_pct") or {}
            all_rows.setdefault(r["key"], {}).setdefault(interval, []).append([
                r["scope"], r["horizon_bars"], bi(r["benchmark"]), vi(r.get("verdict")),
                vi(r.get("verdict_cell") or r.get("verdict")), r.get("direction"),
                r.get("events"), num(r.get("mean_gross_pct")), num(r.get("mean_net_pct")),
                num(r.get("ci_low_pct")), num(r.get("ci_high_pct")), num(r.get("q_value"), 5),
                num(r.get("seg_consistency"), 3), num(r.get("symbol_sign_rate"), 3),
                num(cs.get("x2")), num(it.get("p"), 4), num(r.get("shrink_weight"), 3),
                num(r.get("shrunk_net_pct")),
            ])
        for (key, sc), rows in by_key.items():
            b = best_cell(rows)
            if b is None:
                continue
            cells.setdefault(key, {}).setdefault(interval, {})[sc] = [
                vi(b.get("verdict_cell") or b.get("verdict")), b["horizon_bars"], bi(b["benchmark"]),
                num(b.get("mean_net_pct")), num(b.get("shrunk_net_pct")), num(b.get("ci_low_pct")),
                num(b.get("q_value"), 5), b.get("events"), num(b.get("seg_consistency"), 3),
                num(b.get("symbol_sign_rate"), 3), b.get("direction"),
            ]
    keys = sorted(cells)
    # 明细只给「至少有一个说得出口的格子」的列：no_edge / inconclusive 的链路对读者没有
    # 额外信息（矩阵格里已经写了判定），全留会让页面涨到几十 MB。
    positive = {k for k in keys if any(
        verdicts[c[0]] in PROMOTABLE
        for per_iv in cells[k].values() for c in per_iv.values())}
    detail = {k: all_rows.get(k, {}) for k in sorted(positive)}
    cfg = stats.get("config") or {}
    # 指标族计数：列数会骗人（同一指标 6 个条件列 ≠ 6 份证据），页面顶部先给分母。
    ind_all = {k.split("|")[0] for k in keys}
    ind_pos = {k.split("|")[0] for k in positive}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": scope,
        # 数据源文件名要跟着 `--stats` 走：写死成 `indicator-fit-stats-<scope>.json` 会在
        # 用调试产物生成页面时骗人（顶部那行「数据源」是判断「这页看的是哪一轮」的唯一线索）。
        "stats_source": stats_path.name,
        "stats_generated_at": stats.get("generated_at"),
        "sha": stats.get("sha"),
        "horizons_bars": stats.get("horizons_bars"),
        "gates": cfg,
        "verdict_rank": VERDICT_RANK,
        "promotable": list(PROMOTABLE),
        "cost_round_trip_pct": cfg.get("round_trip"),
        "intervals": intervals,
        "scopes": [s for s in SCOPE_ORDER if any(
            s in (cells[k].get(iv) or {}) for k in keys for iv in cells[k])],
        "scope_label": SCOPE_LABEL,
        "scope_sizes": {iv: (stats["intervals"][iv].get("scope_sizes") or {}) for iv in intervals},
        # `verdict_counts` 在产物里是**按池**分层的 {池: {判定: 行数}}。页面顶部的卡片要的是
        # 一个周期的判定分布，直接喂嵌套结构会让 Object.values 相加得 0、把对象印成
        # [object Object]（实测踩过）。分池分布由 cells[key][iv][scope] 承担，这里只取全样本池。
        "verdict_counts": {iv: ((stats["intervals"][iv].get("verdict_counts") or {}).get("all") or {})
                           for iv in intervals},
        "columns": len(keys),
        "indicators_total": len(ind_all),
        "indicators_promotable": len(ind_pos),
        "directions_sha": events.get("directions_sha"),
        "event_run": {iv: {k: v.get(k) for k in ("computed", "cached", "seconds", "failures")}
                      for iv, v in (events.get("intervals") or {}).items()},
        "probe": {k: probe.get(k) for k in (
            "median_seconds_per_bar", "estimate_cpu_hours", "estimate_wall_hours_at_workers",
            "bars_total_by_interval")} if probe else {},
        "redundancy": {iv: {str(h): sorted(
            (stats["intervals"][iv].get("redundancy") or {}).get(str(h)) or [],
            key=lambda p: -(p.get("r") or 0))[:400]
            for h in (stats["intervals"][iv].get("redundancy") or {})} for iv in intervals},
        # 表格主体：每个 key 一行，cells[key][interval][scope] = 最佳格子（定长数组，
        # 字段名见 cf；判定词与基准名是 verdicts / benches 的下标）
        "cf": CF, "df": DF, "verdicts": verdicts, "benches": benches,
        "keys": keys,
        "meta": {k: [col_meta.get(k, {}).get("direction", 0),
                     col_meta.get(k, {}).get("source", "")] for k in keys},
        "cells": cells,
        "detail": detail,
        "detail_keys": len(detail),
    }


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>指标 × 周期 × 币种池 适配矩阵</title>
<style>
:root{--bg:#0f1115;--fg:#e6e8ee;--mut:#9aa3b2;--line:#232732;--card:#161a22;
--ok:#2ea043;--warn:#d29922;--bad:#6e7681;--info:#388bfd;--danger:#db6d28}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,"Segoe UI",
"Microsoft YaHei",sans-serif}
header{padding:18px 20px 12px;border-bottom:1px solid var(--line)}
h1{margin:0 0 6px;font-size:19px}
.sub{color:var(--mut);font-size:12px}
.wrap{padding:14px 20px 60px}
.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:12px}
.ctrl{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:7px 10px;
display:flex;gap:7px;align-items:center;flex-wrap:wrap}
.ctrl b{font-size:12px;color:var(--mut);font-weight:600}
.chip{border:1px solid var(--line);background:#11151c;color:var(--fg);border-radius:999px;
padding:3px 10px;font-size:12px;cursor:pointer;user-select:none}
.chip.on{border-color:var(--info);background:#132a4d;color:#cfe3ff}
input[type=search],select{background:#11151c;border:1px solid var(--line);color:var(--fg);
border-radius:6px;padding:5px 8px;font-size:13px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:right;white-space:nowrap}
th{position:sticky;top:0;background:#0f1115;color:var(--mut);font-weight:600;font-size:12px;
cursor:pointer;z-index:2}
th.l,td.l{text-align:left}
td.name{max-width:420px;overflow:hidden;text-overflow:ellipsis}
.cell{border-radius:5px;padding:2px 6px;font-size:12px;display:inline-flex;gap:5px;
align-items:center;justify-content:right;min-width:78px}
.candidate{background:rgba(46,160,67,.22);color:#7ee2a8;border:1px solid rgba(46,160,67,.5)}
.cost_fragile{background:rgba(210,153,34,.18);color:#e3c56d;border:1px solid rgba(210,153,34,.45)}
.exploratory{background:rgba(56,139,253,.16);color:#9ecbff;border:1px solid rgba(56,139,253,.4)}
.underpowered_blocks{background:rgba(154,107,214,.16);color:#cfb7f0;border:1px solid rgba(154,107,214,.42)}
.unstable,.not_universal,.inconclusive,.likely_noise{background:rgba(110,118,129,.16);color:#b6bec9;
border:1px solid rgba(110,118,129,.35)}
.no_edge,.no_heterogeneity,.insufficient,.underpowered_cell{background:rgba(110,118,129,.07);color:#6b7280;
border:1px solid rgba(110,118,129,.2)}
.rowdet{background:#0b0d12}
.rowdet td{text-align:left;white-space:normal;padding:10px 14px}
.small{font-size:12px;color:var(--mut)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px;margin:10px 0 16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.card h4{margin:0 0 6px;font-size:13px}
.kv{display:flex;justify-content:space-between;font-size:12px;color:var(--mut);gap:8px}
.kv span:last-child{color:var(--fg)}
.bar{height:5px;border-radius:3px;background:#20242d;display:flex;overflow:hidden;margin-top:6px}
.bar i{display:block;height:100%}
.legend{display:flex;gap:10px;flex-wrap:wrap;font-size:12px;color:var(--mut);margin:4px 0 10px}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:4px}
tr.click:hover{background:#131720;cursor:pointer}
.note{border-left:3px solid var(--info);background:#111722;padding:8px 12px;border-radius:0 8px 8px 0;
font-size:13px;margin:10px 0}
a{color:#8ab8ff}
.scroll{overflow:auto;max-height:72vh;border:1px solid var(--line);border-radius:10px}
</style></head>
<body>
<header>
<h1>指标 × K线周期 × 币种池 适配矩阵</h1>
<div class="sub" id="hdr"></div>
</header>
<div class="wrap">
<div class="card" role="note"><b>审计修正：仅供探索性复核</b><p>candidate 是当前实现标签，尚未通过完整预登记要求：max-T 未实施，交互 p 未作跨格校正，Q 池为事后固定分档，子集判定缺少整体方向门。最强格排名混合币种构成及持有时间，不能证明周期或币池适配。后续验证 BLOCKED；原始数字保留。</p><a href="indicator-fit-readiness.md">查看方法与数据边界审计</a></div>
<div id="summary"></div>
<div class="controls">
  <div class="ctrl"><b>搜索</b><input type="search" id="q" placeholder="指标名 / 列名"></div>
  <div class="ctrl" id="fVerdict"><b>判定</b></div>
  <div class="ctrl" id="fDir"><b>方向</b>
    <span class="chip on" data-v="any">全部</span>
    <span class="chip" data-v="1">多头</span>
    <span class="chip" data-v="-1">空头</span>
    <span class="chip" data-v="0">未声明</span></div>
  <div class="ctrl" id="fSrc"><b>来源</b>
    <span class="chip on" data-v="any">全部</span>
    <span class="chip" data-v="declared">已声明</span>
    <span class="chip" data-v="weak">弱词</span>
    <span class="chip" data-v="ambiguous">方向不明</span></div>
  <div class="ctrl"><b>主判定基准</b><select id="fBench">
    <option value="unconditional">无条件（超额 vs 该币自身）</option>
    <option value="cross_section">横截面（超额 vs 同时刻篮子）</option></select></div>
  <div class="ctrl"><b>排序</b><select id="sortBy">
    <option value="score">按最强格子强度</option>
    <option value="name">按列名</option>
    <option value="events">按事件数</option></select></div>
  <div class="ctrl"><b>只看有结论的列</b><span class="chip" id="onlyPos">开</span></div>
</div>
<div class="legend" id="legend"></div>
<div class="scroll"><table id="tbl"><thead></thead><tbody></tbody></table></div>
<div class="small" id="foot" style="margin-top:10px"></div>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const IV = D.intervals, SC = D.scopes.length ? D.scopes : ['all'];
// 数据在内联时压成定长数组（见 fit_matrix_page.build_payload 的 cf/df），这里还原成
// 带字段名的对象——页面只读不重算，判定词与基准名从字典下标取回。
function obj(fields, arr, dict){
  const o = {};
  fields.forEach((f,i)=>{ o[f] = arr[i]; if (dict && dict[f]!==undefined) o[f] = dict[f][arr[i]]; });
  return o;
}
const CD = {v:D.verdicts, bm:D.benches}, DD = {v:D.verdicts, cv:D.verdicts, bm:D.benches};
const cellOf = (k,iv,sc) => { const a = ((D.cells[k]||{})[iv]||{})[sc]; return a ? obj(D.cf,a,CD) : null; };
const detOf = k => { const out = {}; for (const iv of IV) out[iv] = ((D.detail[k]||{})[iv]||[]).map(a=>obj(D.df,a,DD)); return out; };
// key 形如 `指标名|条件列`，一次解析后所有行复用，避免每格 split。
const INFO = {};
for (const k of D.keys){ const p = k.indexOf('|'); INFO[k] = {ind:k.slice(0,p), col:k.slice(p+1),
  dir:(D.meta[k]||[])[0]||0, src:(D.meta[k]||[])[1]||''}; }
const DET = {};  // 明细懒加载：只有点开某行才解码它那一份
const POS = new Set(D.promotable || ['candidate','cost_fragile','exploratory','underpowered_blocks']);
// 判定等级表由后端判定器给出（D.verdict_rank），页面不维护第二份词表；fallback 只为兼容旧产物。
const RANK = D.verdict_rank || {candidate:0,cost_fragile:1,exploratory:2,underpowered_blocks:3,
  unstable:4,not_universal:5,likely_noise:6,inconclusive:7,no_edge:8,no_heterogeneity:9,
  underpowered_cell:10,insufficient:11};
const VLIST = Object.keys(RANK).sort((a,b)=>RANK[a]-RANK[b]);
const rankOf = v => (RANK[v]===undefined ? 99 : RANK[v]);
const COLORS = {candidate:'#2ea043',cost_fragile:'#d29922',exploratory:'#388bfd',
  underpowered_blocks:'#9a6bd6',unstable:'#8b949e',not_universal:'#8b949e',
  likely_noise:'#6e7681',inconclusive:'#6e7681',no_edge:'#4d5560',
  no_heterogeneity:'#4d5560',underpowered_cell:'#4d5560',insufficient:'#4d5560'};
const esc = s => String(s ?? '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const f3 = v => v==null ? '—' : (Math.abs(v)>=100 ? v.toFixed(0) : v.toFixed(v>=1?2:3));

document.getElementById('hdr').innerHTML =
  `轮次 <b>${esc(D.scope)}</b> · 口径 sha <code>${esc(D.sha)}</code> · 方向表 sha <code>${esc(D.directions_sha)}</code>`
  + ` · 列数 ${D.columns} · 统计产物 ${esc(D.stats_generated_at)} · 页面生成 ${esc(D.generated_at)}`
  // 「数据源」必须在页面上：同一目录里可能同时躺着全量产物和自检产物，光看 sha 分不出来。
  + ` · 数据源 <code>${esc(D.stats_source||'—')}</code>`
  + ` · 成本 ${D.cost_round_trip_pct!=null ? (D.cost_round_trip_pct*100).toFixed(2)+'%' : '—'} 往返（固定，不按策略覆盖）`;

// ---- 顶部小结：每周期判定分布 + 成本/耗时证据
function summary(){
  let h = '<div class="grid">';
  for(const iv of IV){
    const vc = D.verdict_counts[iv] || {};
    const tot = Object.values(vc).reduce((a,b)=>a+b,0) || 1;
    const order = Object.keys(vc).sort((a,b)=>(vc[b]||0)-(vc[a]||0));
    const bar = order.map(k=>`<i style="width:${100*(vc[k]||0)/tot}%;background:${COLORS[k]||'#555'}"></i>`).join('');
    const sz = D.scope_sizes[iv] || {};
    h += `<div class="card"><h4>${esc(iv)} · ${sz.all||'?'} 币 · 全样本池判定</h4>`
      + `<div class="kv"><span>candidate</span><span>${vc.candidate||0}</span></div>`
      + `<div class="kv"><span>cost_fragile</span><span>${vc.cost_fragile||0}</span></div>`
      + `<div class="kv"><span>exploratory</span><span>${vc.exploratory||0}</span></div>`
      + `<div class="kv"><span>参与判定的行（全样本池）</span><span>${tot}</span></div>`
      + `<div class="bar">${bar}</div>`
      + `<div class="small" style="margin-top:6px">池规模 ` + SC.map(s=>`${esc(D.scope_label[s]||s)}:${sz[s]||0}`).join(' ') + `</div>`
      + `</div>`;
  }
  h += `<div class="card"><h4>指标族</h4>`
    + `<div class="kv"><span>说得出口的列</span><span>${D.detail_keys||0} / ${D.columns} 列</span></div>`
    + `<div class="kv"><span>说得出口的指标</span><span>${D.indicators_promotable||0} / `
    + `${D.indicators_total||'?'} 个</span></div>`
    + `<div class="small" style="margin-top:6px">同一指标多个条件列不算多份证据（冗余归堆见报告 §5）；`
    + `「哪个指标适合哪个周期」的逐周期计数在报告 §3.2 与 matrix json 的 <code>indicator_rollup</code>。`
    + `本页可按指标名搜索、按周期排序。</div></div>`;
  const run = D.event_run || {};
  const nFail = Object.values(run).reduce((a, v) => a + ((v.failures || []).length), 0);
  h += `<div class="card"><h4>事件覆盖</h4>`
    + IV.map(iv=>`<div class="kv"><span>${esc(iv)}</span><span>${run[iv]?`${run[iv].computed} 算 / ${run[iv].cached} 缓存命中 / ${D.scope_sizes[iv]?.all||'?'} 币`:'—'}</span></div>`).join('')
    + `<div class="kv"><span>失败币</span><span style="color:${nFail?'#e3c56d':'#7ee2a8'}">${nFail}</span></div>`
    + `<div class="small" style="margin-top:6px">失败=取数或计算异常，该币整列不进池化；缺缓存不等于 0 收益。</div></div>`;
  if (D.probe && D.probe.estimate_cpu_hours != null){
    h += `<div class="card"><h4>算力证据</h4>`
      + `<div class="kv"><span>中位耗时</span><span>${D.probe.median_seconds_per_bar} s/bar</span></div>`
      + `<div class="kv"><span>全量 CPU</span><span>${D.probe.estimate_cpu_hours} h</span></div>`
      + `<div class="kv"><span>12 worker 墙钟</span><span>${D.probe.estimate_wall_hours_at_workers?.['12']||'—'} h</span></div>`
      + `<div class="bar"><i style="width:100%;background:#388bfd"></i></div>`
      + `<div class="small" style="margin-top:6px">bars ${IV.map(iv=>esc(iv)+':'+(D.probe.bars_total_by_interval||{})[iv]).join(' ')}</div></div>`;
  }
  h += '</div>';
  const red = Object.entries(D.redundancy||{}).map(([iv,m])=>{
    const n = Object.values(m).reduce((a,l)=>a+l.length,0);
    return `${esc(iv)}:${n}对`;
  }).join(' ');
  h += `<div class="note"><b>怎么读这张表。</b> 每格取「3 个预测长度 × 2 个基准」里最强的一条：先比判定等级，`
    + `同级别比置信下界。绿色 <span class="cell candidate">candidate</span> 表示当前实现的五项筛选标签，仍须审计复核——`
    + `自助法 95% 下界&gt;0、BH-FDR q&lt;0.10、四段同号≥0.75、两倍成本后仍&gt;0、币块数够。`
    + `<span class="cell exploratory">exploratory</span> 只是「方向不明的列」的上限，不许直接接进策略。`
    + `<span class="cell underpowered_blocks">underpowered_blocks</span> = 前四道关全过，但参与自助法的币块少于 `
    + `${D.gates.blocks_for_candidate ?? 15} 个：块数少时 p 值会打到离散下界（约 2^(1-k)），q&lt;0.10 不代表 FDR 成立，`
    + `所以不上「候选」——数字照给，当线索用。`
    + ` 冗余归堆（逐币均值的币间相关 r≥${D.gates.redundancy_r_min ?? '—'}，是近似，不是逐事件重合率；`
    + `每周期每 h 最多列 400 对）：${red||'—'}。`
    + ` 数字全部来自 <code>${esc(D.stats_source || ('indicator-fit-stats-' + D.scope + '.json'))}</code>，页面不重算。</div>`;
  return h;
}
document.getElementById('summary').innerHTML = summary();
document.getElementById('legend').innerHTML = VLIST.map(k =>
  `<span><i class="dot" style="background:${COLORS[k]||'#555'}"></i>${k}</span>`).join('');

// ---- 过滤条（判定）
document.getElementById('fVerdict').insertAdjacentHTML('beforeend',
  `<span class="chip on" data-v="any">全部</span>` + VLIST.map(v=>`<span class="chip" data-v="${v}">${v}</span>`).join(''));
function wireSingle(id, cb){
  const box = document.getElementById(id);
  box.addEventListener('click', e=>{
    const c = e.target.closest('.chip'); if(!c) return;
    box.querySelectorAll('.chip').forEach(x=>x.classList.toggle('on', x===c));
    cb(c.dataset.v);
  });
}
const state = {verdict:'any', dir:'any', src:'any', bench:'unconditional', sort:'score',
  onlyPos:false, q:''};
wireSingle('fVerdict', v=>{state.verdict=v; render();});
wireSingle('fDir', v=>{state.dir=v; render();});
wireSingle('fSrc', v=>{state.src=v; render();});
document.getElementById('fBench').onchange = e=>{state.bench=e.target.value; render();};
document.getElementById('sortBy').onchange = e=>{state.sort=e.target.value; render();};
document.getElementById('q').oninput = e=>{state.q=e.target.value.trim(); render();};
document.getElementById('onlyPos').addEventListener('click', e=>{
  state.onlyPos = !state.onlyPos; e.target.classList.toggle('on', state.onlyPos); render();});

// 每格的判定等级与置信下界只算一次（与筛选状态无关），后面排序/过滤都读这份缓存。
const CELLS = {};   // key -> [{iv,sc,c}]
const SCORE = {};   // key -> {best, val, ev, pos}
for (const k of D.keys){
  const list = []; let best = 99, bestV = '', val = -9e9, ev = 0, pos = false;
  for (const iv of IV) for (const sc of SC){
    const c = cellOf(k, iv, sc); if (!c) continue;
    list.push({iv, sc, c});
    const rr = rankOf(c.v);
    if (POS.has(c.v)) pos = true;
    const lo = c.lo==null ? -9e9 : c.lo;
    if (rr < best || (rr === best && lo > val)) { best = rr; bestV = c.v; val = lo; }
    ev += c.n || 0;
  }
  CELLS[k] = list; SCORE[k] = {best, bestV, val, ev, pos};
}
// 主基准：格子的显示值按选中的 benchmark 取（判定等级仍用最强格，页面不重算判定）
function det(k, iv){
  if (!DET[k]) DET[k] = detOf(k);
  return DET[k][iv] || [];
}
function cellVal(k, iv, sc, bench){
  const c = cellOf(k, iv, sc); if(!c) return null;
  const d = det(k, iv).find(r=>r.sc===sc && r.h===c.h && r.bm===bench);
  return {c, d};
}
function render(){
  const tb = document.querySelector('#tbl tbody');
  const th = document.querySelector('#tbl thead');
  th.innerHTML = `<tr><th class="l">指标 | 条件列</th><th class="l">方向</th>`
    + IV.map(iv=>`<th class="l">${esc(iv)}</th>`).join('') + `<th>最强档</th><th>事件</th></tr>`;
  const scored = D.keys.filter(k=>{
    const m = INFO[k];
    if(state.q && m.ind.indexOf(state.q)<0 && m.col.indexOf(state.q)<0) return false;
    if(state.src!=='any' && (m.src||'ambiguous')!==state.src) return false;
    if(state.dir!=='any'){
      const want = Number(state.dir);
      if(want===0 ? m.dir!==0 : (m.dir!==want && !CELLS[k].some(x=>x.c.dir===want))) return false;
    }
    const s = SCORE[k];
    if(state.onlyPos && !s.pos) return false;
    if(state.verdict!=='any' && !CELLS[k].some(x=>x.c.v===state.verdict)) return false;
    return true;
  }).map(k=>({k, s: SCORE[k]}));
  if(state.sort==='score') scored.sort((a,b)=>a.s.best-b.s.best || b.s.val-a.s.val || a.k.localeCompare(b.k));
  else if(state.sort==='events') scored.sort((a,b)=>b.s.ev-a.s.ev);
  else scored.sort((a,b)=>a.k.localeCompare(b.k));
  const cap = 1200;
  let html = '';
  for(const {k, s} of scored.slice(0, cap)){
    const m = INFO[k];
    const byIv = {}; for(const x of CELLS[k]) (byIv[x.iv] = byIv[x.iv] || []).push(x);
    const cells = IV.map(iv=>{
      let best = null, bestSc = null;
      for(const x of (byIv[iv]||[])){
        const rr = rankOf(x.c.v);
        const lo = x.c.lo==null?-9e9:x.c.lo;
        if(!best || rr<best.rank || (rr===best.rank && lo>best.lo)) best={c:x.c,rank:rr,lo}, bestSc=x.sc;
      }
      if(!best) return '<td class="l"><span class="small">无覆盖</span></td>';
      const c = best.c;
      const d = cellVal(k, iv, bestSc, state.bench);
      const dd = d ? d.d : null;
      const net = (dd && dd.net!=null) ? dd.net : c.net;
      const shr = dd ? dd.shr : null;
      const bmTag = dd ? (dd.bm==='cross_section'?'截面':'无条件') : '';
      return `<td class="l"><span class="cell ${esc(c.v)}" title="${esc(c.v)} · h=${c.h} · ${esc(D.scope_label[bestSc]||bestSc)}`
        + ` · ${esc(bmTag)} net=${f3(net)}% · CI下界=${f3(dd?dd.lo:c.lo)}% · q=${(dd?dd.q:c.q)==null?'—':(dd?dd.q:c.q).toFixed(3)}`
        + ` · 事件=${c.n} · 段同号=${f3(c.seg)} · 币同号=${f3(c.sgn)}">`
        + `${f3(net)}%</span> <span class="small">h${c.h}/${esc((D.scope_label[bestSc]||bestSc).slice(0,4))}`
        + (shr!=null?` 收缩${f3(shr)}`:'')+`</span></td>`;
    }).join('');
    html += `<tr class="click" data-k="${esc(k)}"><td class="l name" title="${esc(k)}">${esc(m.ind)} <span class="small">|</span> ${esc(m.col)}</td>`
      + `<td class="l small">${m.dir>0?'多':(m.dir<0?'空':'—')}<span class="small">/${esc(m.src.slice(0,4))}</span></td>`
      + cells
      + `<td>${s.pos?`<span class="cell ${esc(s.bestV)}" title="最强格判定">${esc(s.bestV)}</span>`:'—'}</td>`
      + `<td>${s.ev}</td></tr>`;
  }
  tb.innerHTML = html || `<tr><td colspan="${IV.length+4}" class="l small">没有符合条件的列</td></tr>`;
  document.getElementById('foot').textContent =
    `显示 ${Math.min(scored.length,cap)} / 命中 ${scored.length} / 全部 ${D.keys.length} 列`
    + (scored.length>cap ? `（超过 ${cap} 行截断，请用搜索或筛选收窄）` : '')
    + ` · 明细只为 ${D.detail_keys} 个「至少有一格说得出口」的列内联，其余列点开创空`
    + ` · 单元格数字 = 该周期该池该 horizon 扣掉 0.12% 往返成本后的平均超额 %，悬停看完整统计`;
}
// 行展开：列出这一列在所有 (周期, 池, horizon, 基准) 上的完整判定链
document.querySelector('#tbl tbody').addEventListener('click', e=>{
  const tr = e.target.closest('tr.click'); if(!tr) return;
  const nxt = tr.nextElementSibling;
  if(nxt && nxt.classList.contains('rowdet')){ nxt.remove(); return; }
  document.querySelectorAll('tr.rowdet').forEach(x=>x.remove());
  const k = tr.dataset.k;
  const rows = [];
  for(const iv of IV) for(const r of det(k, iv)) rows.push([iv,r]);
  const body = rows.map(([iv,r])=>`<tr><td>${esc(iv)}</td><td>${esc(D.scope_label[r.sc]||r.sc)}</td>
    <td>h${r.h}</td><td>${esc(r.bm)}</td><td>${r.dir>0?'多':'空'}</td>
    <td>${r.n}</td><td>${f3(r.gross)}</td><td>${f3(r.net)}</td>
    <td>${f3(r.lo)}..${f3(r.hi)}</td><td>${r.q==null?'—':r.q.toFixed(4)}</td>
    <td>${f3(r.seg)}</td><td>${f3(r.sgn)}</td>
    <td>${f3(r.x2)}</td><td>${r.ip==null?'—':'p'+r.ip}</td>
    <td>${r.w!=null?`w=${r.w}→${f3(r.shr)}`:'—'}</td>
    <td class="cell ${esc(r.cv)}">${esc(r.cv)}</td></tr>`).join('');
  const t = document.createElement('tr');
  t.className = 'rowdet';
  t.innerHTML = `<td colspan="${IV.length+4}"><div class="small" style="margin-bottom:6px">`
    + `<b>${esc(k)}</b> 判定链（gross/net/CI 单位 %；「2×成本」= 两倍往返成本后的净超额，仍&gt;0 才算扛得住滑点）`
    + ` · 交互 p = 该池相对其余样本的差异检验，p≥${D.gates.interaction_alpha} 判 no_heterogeneity`
    + ` · 收缩 w=n/(n+${D.gates.shrink_k})，格子均值向全样本均值收</div>`
    + (body ? `<table><thead><tr><th>周期</th><th>池</th><th>horizon</th><th>基准</th><th>向</th><th>事件</th>`
        + `<th>毛</th><th>净</th><th>95%CI</th><th>q</th><th>段</th><th>币</th><th>2×成本</th>`
        + `<th>交互</th><th>收缩</th><th>判定</th></tr></thead><tbody>${body}</tbody></table>`
      : `<div class="small">该列所有格子都是 ${'no_edge / inconclusive'} 级，明细未内联（数字见 json 产物）。</div>`)
    + `</td>`;
  tr.after(t);
});
render();
</script>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="A")
    ap.add_argument("--out", default="indicator-fit-matrix.html")
    ap.add_argument("--stats", default="", help="覆盖统计产物路径（自检用，正式渲染不要传）")
    ap.add_argument("--directions", default="", help="覆盖方向表路径（自检用）")
    args = ap.parse_args()
    payload = build_payload(
        args.scope,
        Path(args.stats) if args.stats else None,
        Path(args.directions) if args.directions else None,
    )
    html = HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    dst = DOCS / args.out
    dst.write_text(html, encoding="utf-8")
    print(f"[page   ] {dst.name} {dst.stat().st_size/1024:.0f} KB，"
          f"{payload['columns']} 列 × {len(payload['intervals'])} 周期 × {len(payload['scopes'])} 池"
          f"（数据源 {payload['stats_source']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
