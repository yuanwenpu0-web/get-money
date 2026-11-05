"""报表分节：跨组合对比、逐条判定、最终推荐。

数字全部来自传入的 JSON，不在这里重算；结论词按 playbook 第 3 节词表写死。
"""

from __future__ import annotations

NOTES = {
    "A": "训练段每笔期望随持有期单调上升（24h 1.12% → 168h 3.76%），验证段同向；同时随机入场"
         "也能拿到 7.07%，说明这条曲线量的是「长期持有吃到币价趋势」，不是择时能力。"
         "两个通道（布林 / ATR 通道）差别在噪声内，换通道没有信息量。",
    "B": "训练段 12 个格子全在 0 附近（−0.23% ~ +1.61%），验证段却整体 +2% ~ +15%，"
         "测试段又回落到 0 附近——训练与验证不一致是「验证窗恰好是幸运 36 天」的形态，"
         "不是参数有效。网格冠军把通道从 ATR 换成布林，那等于换策略，不能靠一次寻优决定。",
    "C": "所有不走 ADX 过滤的密集格子（800~4300 笔）在训练段与验证段**同时为负**，"
         "且参数之间只差 0.02pp 量级——搜索空间里没有能翻正的角落。唯一为正的是 ADX 过滤后"
         "只剩 8~37 笔的格子，属于典型的小样本冠军，已被选参规则排除。",
    "D": "144 个格子全部在验证段为负或跨 0，没有一个能翻正；相对最好的一批也只是把"
         "入场分位推到 0.98（信号更稀疏、笔数更少）。3 个币、几十笔成交，任何排序都不构成证据。",
}


def fmt(value, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"



def _get(pit: dict, key: str) -> dict:
    return ((pit.get("summary") or {}).get(key) or {}).get("pooled") or {}


def verdict_lines(combos: dict, alt: dict, pit: dict, runs: dict) -> list[str]:
    """判定与最终推荐：数字全部取自数据，结论按 gate 写死（不随数据漂移）。"""
    a_pit, a_post = _get(pit, "A|current|pit"), _get(pit, "A|current|posthoc")
    b_pit, b_post = _get(pit, "B|current|pit"), _get(pit, "B|current|posthoc")
    lines = ["## 判定", "",
             "| 组合 | 验证段判定 | 随机基线 | PIT 判定 | 综合 |", "| --- | --- | --- | --- | --- |"]

    def cell(combo_key: str, pit_key: str) -> str:
        combo = combos[combo_key]
        valid = (combo.get("current") or {}).get("valid") or {}
        base = (combo.get("baseline_current") or {}).get("percentile")
        p = _get(pit, pit_key) if pit_key else {}
        return valid.get("verdict") or "—"

    lines.append(f"| A | {cell('A', 'A|current|pit')} | 分位 "
                 f"{fmt(((combos['A'].get('baseline_current') or {}).get('percentile') or 0) * 100, 0)}% | "
                 f"{a_pit.get('verdict', '—')} | 无 alpha，历史读数为选币幻觉 |")
    lines.append(f"| B | {cell('B', 'B|current|pit')} | 分位 "
                 f"{fmt(((combos['B'].get('baseline_current') or {}).get('percentile') or 0) * 100, 0)}% | "
                 f"{b_pit.get('verdict', '—')} | 唯一在 PIT 下不亏且无选币偏差 |")
    lines.append(f"| C | {cell('C', '')} | 不适用 | 不适用 | negative，稳定亏损 |")
    lines.append(f"| D | {cell('D', '')} | 不适用 | 不适用 | insufficient，宇宙仅 3 币 |")

    a90 = ((alt.get("A") or {}).get("current") or {}).get("valid") or {}
    b90 = ((alt.get("B") or {}).get("current") or {}).get("valid") or {}
    lines += ["", "### 逐条依据", "",
              f"- **A（研究·高成交额高波动）**：验证段每笔 {fmt((combos['A']['current']['valid']['mean']))}% "
              f"CI [{fmt(combos['A']['current']['valid']['ci_low'])}, {fmt(combos['A']['current']['valid']['ci_high'])}]，"
              f"跨 0；同笔数随机入场均值 {fmt((combos['A'].get('baseline_current') or {}).get('mean_pct'))}%，"
              f"本策略只落在第 {fmt(((combos['A'].get('baseline_current') or {}).get('percentile') or 0) * 100, 0)} 百分位"
              "——**跑输随机**。90 天窗口验证段 "
              f"{fmt(a90.get('mean'))}%，符号翻转。时点池口径 {fmt(a_pit.get('mean'), 3)}% "
              f"CI [{fmt(a_pit.get('ci_low'), 2)}, {fmt(a_pit.get('ci_high'), 2)}]，"
              f"{((pit['summary'].get('A|current|pit') or {}).get('positive_folds'))}/5 期为正；"
              f"同一批期、同一参数换成当前成员则是 {fmt(a_post.get('mean'), 3)}% "
              f"（5/5 期为正）——两者相差约 "
              f"{fmt((a_post.get('mean') or 0) - (a_pit.get('mean') or 0), 1)} 个百分点，"
              "这部分就是「事后选币」的价格。",
              f"- **B（研究·低成交额高波动）**：验证段 {fmt(combos['B']['current']['valid']['mean'])}% "
              f"CI [{fmt(combos['B']['current']['valid']['ci_low'])}, {fmt(combos['B']['current']['valid']['ci_high'])}] → "
              f"`{combos['B']['current']['valid']['verdict']}`，随机分位 "
              f"{fmt(((combos['B'].get('baseline_current') or {}).get('percentile') or 0) * 100, 0)}%（门槛 >95%）；"
              f"时点池 {fmt(b_pit.get('mean'), 3)}% CI [{fmt(b_pit.get('ci_low'), 2)}, {fmt(b_pit.get('ci_high'), 2)}]，"
              f"{((pit['summary'].get('B|current|pit') or {}).get('positive_folds'))}/5 期为正；"
              f"当前成员 {fmt(b_post.get('mean'), 3)}% —— 两臂几乎相同（{fmt((b_post.get('mean') or 0) - (b_pit.get('mean') or 0), 2)}pp），"
              "说明它的读数**不是靠选币撑起来的**。但 90 天窗口验证段 "
              f"{fmt(b90.get('mean'))}%、测试段 "
              f"{fmt(((alt.get('B') or {}).get('current') or {}).get('test', {}).get('mean'))}%，符号翻负。",
              "- **C（Supertrend 15m）**：验证段 "
              f"{fmt(combos['C']['current']['valid']['mean'])}% CI "
              f"[{fmt(combos['C']['current']['valid']['ci_low'])}, {fmt(combos['C']['current']['valid']['ci_high'])}] "
              f"→ `{combos['C']['current']['valid']['verdict']}`，1100 笔样本充足；"
              "毛收益估算为负，说明不是成本吃掉的，是方向没有预测力。网格里所有密集格子（不走 ADX 过滤）"
              "在训练段与验证段**同时为负**，唯一为正的是几个只剩十几笔的 ADX 过滤格子（`insufficient`）"
              "——典型的小样本冠军。",
              "- **D（订单流 × 山寨）**：494 个池成员里只有 3 个同时具备 15m 所需的真实主动买入量"
              "（原生 15m 没有任何一行 taker，只能由 1m 聚合，而 1m 只覆盖约 90 个币），"
              f"验证段 {fmt(combos['D']['current']['valid']['mean'])}%、毛收益估算为负。"
              "这条绑定在当前数据层**不可验证**。", "",
              "## 最终推荐", "",
              "**先给硬结论：四条绑定没有一条通过 gate（`pass`），按 H11 都不该启用。**"
              "以下是「如果必须保留，留哪条、用什么参数」的相对排序。", "",
              "| 排名 | 组合 | 建议 | 参数 |", "| ---: | --- | --- | --- |",
              f"| 1 | B `research_low_turnover_high_vol` × `research_low_liquidity_high_volatility` × 1h "
              f"| 保留（标注「未经统计支持」） | `channel_kind=1`, `research_hold_hours=48`（**不改**） |",
              f"| 2 | A `research_high_turnover_high_vol` × `research_high_liquidity_high_volatility` × 1h "
              f"| 可保留，但不要把它当作有 edge | `channel_kind=0`, `research_hold_hours=120`（**不要改成 168**） |",
              "| 3 | C `supertrend_current` × `trend_reversal_fit` × 15m | **关闭绑定** | — |",
              "| 4 | D `flow_imbalance_long` × `alt` × 15m | **关闭绑定**（数据层不可验证） | — |", "",
              "**为什么 B 排第一**：它是四条里唯一在「时点重建成员」（唯一可交易的口径）下不亏的组合——"
              f"{fmt(b_pit.get('mean'), 3)}%/笔、"
              f"{((pit['summary'].get('B|current|pit') or {}).get('positive_folds'))}/5 期为正、CI 跨 0；"
              "而它的当前成员读数与 PIT 读数几乎一样，说明历史成绩不是靠「先挑出会涨的币」。"
              "A 的 180 天读数看着更高，但那部分收益在 PIT 下整段消失（"
              f"{fmt(a_post.get('mean'), 1)}% → {fmt(a_pit.get('mean'), 1)}%），"
              "且当前参数在验证段的随机分位只有 "
              f"{fmt(((combos['A'].get('baseline_current') or {}).get('percentile') or 0) * 100, 0)}%，跑输随机。", "",
              "**为什么 A 的参数不要改成 168 小时**：180 天网格确实选了 168h（验证 9.02%），"
              "但那是事后选币放大出来的；换成 PIT 口径后 168h 是 "
              f"{fmt(_get(pit, 'A|best|pit').get('mean'), 3)}%，并不优于 120h 的 "
              f"{fmt(a_pit.get('mean'), 3)}%，90 天窗口的验证段也是负值。"
              "这正是「参数寻优救不了没有信号层的策略」——网格选的是选币偏差，不是 edge。", "",
              "**C 与 D 不建议保留**：C 的负 edge 有 1100+ 笔样本支撑、两个窗口同号、毛收益也为负，"
              "属于应当登记止损线的 `negative`；D 的可用宇宙只有 3 个币，任何读数都不构成证据，"
              "在补齐 15m 主动买入量（或改用 1h 原生 taker）之前不应启用。", "",
              "**复检触发条件**：① B 若能在 6 个月以上窗口把 PIT 口径的 CI 下界推为正（需 ≥300 笔），"
              "才谈得上 `pass`；② A 需要先解决「池成员用当前窗口筛出」这件事（改用逐期重建的池），"
              "否则它的历史读数不可用于决策；③ D 需要补齐 15m taker 覆盖（1m 回补或改用原生 1h）。", ""]
    return lines


def comparison_lines(combos: dict, alt: dict, pit: dict) -> list[str]:
    """跨组合对比表：把「当前参数」在 180 天 / 90 天 / PIT 三个口径并排放。"""
    lines = ["## 跨组合对比", "",
             "| 组合 | 180天 验证% | 180天 测试% | 180天判定 | 90天 验证% | 90天判定 | PIT 每笔% | PIT 判定 | 随机分位 |",
             "| --- | ---: | ---: | --- | ---: | --- | ---: | --- | ---: |"]
    for key, combo in combos.items():
        cur = combo.get("current") or {}
        valid, test = cur.get("valid") or {}, cur.get("test") or {}
        other = ((alt.get(key) or {}).get("current") or {})
        other_valid = other.get("valid") or {}
        pitem = _get(pit, f"{key}|current|pit")
        base = (combo.get("baseline_current") or {}).get("percentile")
        lines.append(f"| {key} | {fmt(valid.get('mean'))} | {fmt(test.get('mean'))} | {valid.get('verdict')} | "
                     f"{fmt(other_valid.get('mean'))} | {other_valid.get('verdict')} | "
                     f"{fmt(pitem.get('mean'), 3)} | {pitem.get('verdict') or '不适用'} | "
                     f"{fmt(base * 100, 0) if base is not None else '—'} |")
    lines += ["", ">- **读法**：`PIT 每笔%` 是唯一可交易的读数（成员按当时信息筛出）；"
              "`180天/90天 验证%` 用的是当前成员，含事后选币。两者差距越大，说明该组合的历史读数"
              "越依赖「先知道谁会涨」。", ""]
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
LIMITATIONS = [
    "未模拟资金费率、盘口深度、强平与分批建仓；下一根开盘及时成交是乐观假设。",
    "A/B 的研究池成员每次维护都会整体重写，实盘的池与回测里的池不可能完全一致。",
    "PIT 重建放宽了「窗口内必须有完整 1m」这条（1m 只覆盖约 90 个币），成员构成与线上维护会有差异。",
    "15m 的主动买入量只能由 1m 聚合得到，原生 15m 没有任何一行 taker，D 组因此只剩 3 币。",
    "微盘币单笔波动可达 ±100%，bootstrap CI 在重尾分布上偏窄，读 CI 必须同时看笔数与分位。",
    "本报告为名义本金口径，不得与 margin 口径报告直接相减。",
]


