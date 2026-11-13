"""P17 / H-E 成对报告的 markdown 渲染 —— 纯函数，不碰 DB、不写文件。

为什么报告本身是判定的一部分
--------------------------
10 个臂的收益数字单独看会骗人：`posthoc_N` 一定漂亮，因为它是按**当期已实现收益**
挑币的作弊臂。报告必须把 `posthoc − pit` 的差（pp/笔）写在同一页上，读者才知道
「缩小池能赚钱」这个直觉里有多少是泄漏、多少是真实可交易的信息。

同理，被跳过的币、覆盖率门槛、排名窗口口径都必须落纸面：零成交不等于「没信号」，
也可能等于「这个币根本没进样本」（手册 H8）。
"""
from __future__ import annotations

import statistics

from .pit_pool_stats import arm_names, jaccard

CHECK_LABELS = (
    ("C1_ci_low_positive", "C1 池化每笔期望 bootstrap 95% CI 下界 > 0"),
    ("C2_min_trades", "C2 池化成交笔数 ≥ 30"),
    ("C3_fold_signs", "C3 正收益 fold 数 ≥ ceil(0.8 × folds)"),
    ("C4_beats_random", "C4 组合账户收益在多种子随机方向基线中的分位 > 0.95"),
    ("C5_fdr", "C5 六个 PIT 臂 BH-FDR q < 0.10"),
)


def _num(value, spec: str = ".4f") -> str:
    return "—" if value is None else format(value, spec)


def _signed(value, spec: str = "+.4f") -> str:
    return "—" if value is None else format(value, spec)


def _ci_text(ci: dict) -> str:
    if not ci.get("trades"):
        return "—"
    return f"[{_signed(ci.get('ci_low_pct'))}, {_signed(ci.get('ci_high_pct'))}]"


def _overlap(folds: list[dict]) -> tuple[str, str]:
    pairs = [jaccard(a["members"], b["members"]) for a, b in zip(folds, folds[1:])]
    if not pairs:
        return "—", "—"
    return f"{statistics.median(pairs):.3f}", f"{min(pairs):.3f}"


def limitations(payload: dict) -> list[str]:
    """局限清单：md 与 json 共用同一份，避免两处口径漂移。"""
    sizes = tuple(payload["protocol"]["sizes"])
    universe = payload["universe"]
    items = [
        "**无上市／下架事件表**：PIT 合格性只能由在库 1h bar 的存在性与新鲜度推断，"
        "是「当时可交易」的**下界**，不是交易所事实。",
        "排名候选先按**全窗口覆盖率**裁剪再取 TopN。该过滤对所有臂（含 `all`）一视同仁，"
        "臂间比较公平；但它用了全窗口信息，因此本报告的 PIT 是「相对 PIT」，"
        "不是严格逐时点可交易宇宙。",
        "排名只用成交额与波动率，**不含 spread／盘口深度／资金费率**；"
        "真实可成交性比本口径更严格。",
        "每币独立账户等权，**不是组合层面的资金分配**；账户均值% 只用于随机基线分位比较，"
        "不能与每笔期望% 混比（H3）。",
        "随机基线只翻转方向，**不随机化选币**；因此 C4 检验的是「方向有没有信息」，"
        "不是「选币有没有信息」。选币的信息量由第 5 节的泄漏差与 C1–C3 共同回答。",
        "15m 全池 365 天在库不可得（0 币达标），故 15m 臂只能跑 180 天，"
        "两臂样本期不同，**不可直接比大小**。",
        "币池由运行时刻的在库 bar 数决定；采集任务仍在增量同步，"
        "不同时刻重跑达标币数可能变化，报告已记录 `generated_at` 与实际币数。",
    ]
    if max(sizes) >= universe["evaluated"]:
        items.append(
            f"**池规模退化**：达标宇宙只有 {universe['evaluated']} 币，而最大 N={max(sizes)}。"
            f"N={max(sizes)} 的「缩小池」等于不缩小，该臂与 `all` 基线实质同构，"
            f"**不能用来回答「池该取多大」**；本周期只有 N<{universe['evaluated']} 的臂有区分度。")
    short = []
    for name in arm_names(sizes):
        arm = payload["summary"].get(name)
        if not arm or name == "all":
            continue
        gaps = [(len(f["members"]), f["requested"]) for f in arm["folds"]
                if len(f["members"]) < f["requested"]]
        if gaps:
            worst = min(gaps, key=lambda g: g[0])
            short.append(f"`{name}` 最少 {worst[0]}/{worst[1]}")
    if short:
        items.append(
            "**池规模凑不满**：排名候选另有门槛（168h 回看窗内 ≥160 根 1h bar、末根距今 ≤2h、"
            "非黑名单／非杠杆代币），交易周期覆盖率达标不等于排名合格，TopN 只能取到候选上限。"
            f"实测 {'、'.join(short)}。这些臂**不是严格固定规模**，"
            "跨期比较须对照分报告「成员稳定性与每期规模」表一起读。")
    return items


def render(payload: dict) -> str:
    summary = payload["summary"]
    verdicts = payload["verdicts"]
    sizes = tuple(payload["protocol"]["sizes"])
    names = [n for n in arm_names(sizes) if n in summary]
    missing = [n for n in arm_names(sizes) if n not in summary]
    universe = payload["universe"]
    costs = payload["costs"]
    params = payload["parameters"]
    protocol = payload["protocol"]
    lines: list[str] = []
    add = lines.append

    add(f"# P17 / H-E · Supertrend「缩小币种池」PIT 可行性判定 · {payload['interval']}")
    add("")
    add(f"- 预登记假设卡：`{payload['hypothesis_card']}`（判据 C1–C5 **跑之前**写死）")
    add(f"- 生成时间：{payload['generated_at']} · 策略 `{payload['strategy']}` · "
        f"{payload['folds']} folds × {payload['days']} 天 · {payload['elapsed_seconds']}s "
        f"/ {payload['processes']} 进程")
    add(f"- **课题判定：`{verdicts['overall']}`**")
    add("- `verdict_scope`：固定规模 PIT 币池下、冻结参数的净收益统计判定；"
        "**不是**「换个指标再试一次」的许可。")
    add("- 授权范围：**只读回测 / 探针，不下真实订单**、不启用 live API key、不切换交易环境、"
        "不改 spec 默认参数。趋势反转绑定 #7 维持 `enabled=0`。")
    add("- 启用决策：`default_enabled=False`，本轮**不启用**任何绑定（H11：无 `pass` 不得启用）。")
    add("")

    add("## 1. 预登记判据（与关系，任一不过即不得写 pass）")
    add("")
    add("| 编号 | 条件 |")
    add("|---|---|")
    for _, label in CHECK_LABELS:
        add(f"| {label.split(' ', 1)[0]} | {label.split(' ', 1)[1]} |")
    add("")
    add(f"`all` 与 `posthoc_*` 是**诊断臂**，不参与 gate、不进 FDR 家族；"
        f"FDR 家族 = {', '.join(verdicts['fdr_family'])}。")
    add("")

    add("## 2. 数据边界与样本口径（H8）")
    add("")
    add(f"- 窗口：{payload['window']['start']} → {payload['window']['end']}（UTC）")
    add(f"- 入选宇宙：{payload['interval']} bar 数 ≥ {universe['coverage_threshold']:.0%} × "
        f"{universe['expected_bars']} 根；窗口内有 bar {universe['requested']} 币，"
        f"覆盖率达标 {universe.get('qualified', universe['requested'])} 币，"
        f"实际产出 {universe['evaluated']} 币")
    add(f"- 排名口径：`{protocol['ranking_interval']}` 上 `[T−{protocol['ranking_window_hours']}h, T)` "
        f"的事前窗口，**两个周期臂共用同一份 1h 排名**（避免用 15m 噪声排序）")
    add(f"- 预热 {protocol['warmup_bars']} 根 · 单 fold 最短 {protocol['min_fold_bars']} 根 · "
        f"单段最短 {protocol['min_segment_bars']} 根 · 随机基线 {protocol['seeds']} 种子"
        f"（base {protocol['base_seed']}，步长 7919）")
    add(f"- 排名黑名单（{len(universe['excluded'])} 币）：{', '.join(universe['excluded']) or '无'}")
    add(f"- 跳过 {len(universe['skipped'])} 币：" if universe["skipped"] else "- 跳过 0 币（无）")
    for item in universe["skipped"][:40]:
        add(f"  - {item}")
    if len(universe["skipped"]) > 40:
        add(f"  - …其余 {len(universe['skipped']) - 40} 币见 JSON `universe.skipped`")
    add("")

    add("## 3. 冻结参数与成本（H3：全局统一，不得按臂覆盖）")
    add("")
    shown = ("supertrend_factor", "atr_period", "sl_atr_mult", "tp1_r", "tp2_r", "target_r",
             "min_score", "heavy_score", "max_signal_age_bars", "regime_filter", "cost_filter",
             "use_adx_filter", "use_ema200_filter", "ema_trend_fast", "ema_trend_slow")
    add("- 关键参数：" + " / ".join(f"{k}={params[k]}" for k in shown if k in params))
    add("- 完整参数（含未启用的开关）见 JSON `parameters`；本轮**未改动任何默认值**。")
    add(f"- 成本：fee_rate={costs['fee_rate']} / slippage={costs['slippage']} / "
        f"双边 {costs['round_trip']:.2%} · 每币独立账户 {costs['cash_per_symbol']} · "
        f"口径 **{costs['basis']}（名义本金）**")
    add("- 所有臂共用同一份逐币成交与同一套参数，**成员构成是唯一变量**。")
    add("")

    add(f"## 4. {len(names)} 臂结果")
    add("")
    add("| 臂 | 笔数 | 每笔均值% | CI 下界% | CI 上界% | 正 fold | 账户均值% | 随机分位 | q | 判定 |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for name in names:
        row = summary[name]
        ci = row["pooled"]
        gate = verdicts["arms"][name]
        add(f"| {name} | {ci.get('trades', 0)} | {_signed(ci.get('mean_return_pct'))} | "
            f"{_signed(ci.get('ci_low_pct'))} | {_signed(ci.get('ci_high_pct'))} | "
            f"{row['positive_folds']}/{len(row['fold_means'])} | "
            f"{_signed(row['account_mean_pct'])} | {_num(row['percentile_rank'], '.3f')} | "
            f"{_num(gate['q'], '.4f')} | **{gate['verdict']}** |")
    add("")
    if missing:
        add(f"未产出的臂（该口径下无成员或无成交）：{', '.join(missing)}")
        add("")
    add("### 逐关明细")
    add("")
    add("| 臂 | C1 | C2 | C3 | C4 | C5 |")
    add("|---|---|---|---|---|---|")
    for name in names:
        checks = verdicts["arms"][name]["checks"]
        cells = " | ".join("✓" if checks[key] else "✗" for key, _ in CHECK_LABELS)
        add(f"| {name} | {cells} |")
    add("")

    add("## 5. 事后选币泄漏（本课题的核心诊断）")
    add("")
    add("`posthoc_N` 按**当期已实现收益**挑币，是作弊臂；它与 PIT 臂的差就是"
        "「用未来信息选币」能凭空造出的 pp/笔。")
    add("")
    add("| N | posthoc 均值% | pit_liq 均值% | pit_vol 均值% | 泄漏 vs liq (pp/笔) | 泄漏 vs vol (pp/笔) |")
    add("|---|---|---|---|---|---|")
    for n in sizes:
        post = summary.get(f"posthoc_{n}", {}).get("pooled", {})
        liq = summary.get(f"pit_liq_{n}", {}).get("pooled", {})
        vol = summary.get(f"pit_vol_{n}", {}).get("pooled", {})
        pm, lm, vm = post.get("mean_return_pct"), liq.get("mean_return_pct"), vol.get("mean_return_pct")
        gap_l = None if pm is None or lm is None else round(pm - lm, 4)
        gap_v = None if pm is None or vm is None else round(pm - vm, 4)
        add(f"| {n} | {_signed(pm)} | {_signed(lm)} | {_signed(vm)} | "
            f"{_signed(gap_l, '+.2f')} | {_signed(gap_v, '+.2f')} |")
    add("")

    add("## 6. 成员稳定性与每期规模")
    add("")
    add("| 臂 | 期数 | 每期成员数 | Jaccard 中位 | Jaccard 最小 | 唯一币数 |")
    add("|---|---|---|---|---|---|")
    for name in names:
        folds = summary[name]["folds"]
        counts = ", ".join(str(len(f["members"])) for f in folds)
        med, low = _overlap(folds)
        add(f"| {name} | {len(folds)} | {counts} | {med} | {low} | "
            f"{len(summary[name]['symbols_unique'])} |")
    add("")
    add("Jaccard = 相邻两期成员交并比；1.000 = 完全没换血，0.000 = 整批换掉。"
        "两期都为空时约定 1.000，须对照「每期成员数」一起读。")
    add("")
    add("| fold | 区间 | PIT 合格币数 |")
    add("|---|---|---|")
    for meta in payload["fold_meta"]:
        add(f"| {meta['index']} | {meta['start_iso']} → {meta['end_iso']} | "
            f"{meta.get('eligible_count', '—')} |")
    add("")

    add("## 7. 多空分解")
    add("")
    add("| 臂 | 做多笔数 | 做多均值% | 做多 CI | 做多判定 | 做空笔数 | 做空均值% | 做空 CI | 做空判定 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for name in names:
        split = summary[name]["side_split"]
        cells = []
        for side in ("long", "short"):
            ci = split[side]
            cells += [str(ci.get("trades", 0)), _signed(ci.get("mean_return_pct")),
                      _ci_text(ci), ci.get("verdict", "—")]
        add(f"| {name} | " + " | ".join(cells) + " |")
    add("")

    add("## 8. 局限与边界")
    add("")
    for item in limitations(payload):
        add(f"- {item}")
    add("")

    add("## 9. 复现")
    add("")
    add("```powershell")
    add(r".\.venv\Scripts\python.exe -m pytest backend/tests/test_pit_universe.py "
        r"backend/tests/test_pit_pool_stats.py backend/tests/test_pit_pool_arms.py "
        r"backend/tests/test_pit_pool_report.py -q")
    add(rf".\.venv\Scripts\python.exe backend/scripts/supertrend_pit_pool.py "
        rf"--interval {payload['interval']} --days {payload['days']} --folds {payload['folds']} "
        rf"--out docs/supertrend-pit-pool-{payload['interval']}")
    add(r".\.venv\Scripts\python.exe .workbuddy/skills/strategy-backtest-playbook/scripts/preflight.py "
        r"--prefix docs/supertrend-pit-pool")
    add("```")
    add("")
    return "\n".join(lines)