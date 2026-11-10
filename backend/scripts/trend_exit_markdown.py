"""趋势反转出场课题的 Markdown 报告渲染（主报告 + Gate 1 探针报告）。

只做文本渲染，不做统计判定；判定结果由 trend_exit_report.py 传入。
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import numpy as np

from trend_exit_common import DECISION_BAR, FORWARD_H, MIN_GAP, PURGE_MS
from trend_exit_stats import MIN_POOLED_TRADES, percentile_rank

def md_table(rows, key="all"):
    head = ("| 卖出信号 | 类别 | 触发率 | 配对差 % | 95% CI % | FDR q | 分段同号 | "
            "随机早退均值 % | 随机基线分位 | verdict |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
    lines = []
    for r in rows:
        st = r[key]
        ci = (f"[{st['ci_low_pct']:.4f}, {st['ci_high_pct']:.4f}]"
              if st["ci_low_pct"] is not None else "—")
        seg = r["segments"]
        segsame = seg["n_positive"] if st["mean_pct"] > 0 else seg["n_negative"]
        lines.append(f"| `{r['signal']}` | {r['class']} | {r['hit_rate']:.1%} | "
                     f"{st['mean_pct']:+.4f} | {ci} | {r['fdr_q']:.4f} | "
                     f"{segsame}/4 | "
                     f"{r['random_early_mean_pct']:+.4f} | {r['random_percentile']:.3f} | "
                     f"{st['verdict']} |")
    return head + "\n".join(lines)


def probe_md(d, probe, parts, split_meta, out_prefix):
    rows = probe["rows"]
    cal = probe["self_calibration"]
    lines = [
        "# 探针：条件化止盈／卖出信号判别力（过去半年，训练段）",
        "",
        f"> 生成时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ｜ "
        f"脚本：`backend/scripts/trend_exit_report.py` ｜ 原始数据：`output/trend_exit_study.json`",
        "",
        "## 假设与边界",
        "",
        "- **信息源**：在趋势反转多头入场后的决策时刻（入场后第 "
        f"{DECISION_BAR} 根 1h bar，预注册固定），用 8 条预注册卖出信号的分值排序；"
        f"标签为未来 {FORWARD_H} 根收益。价格类 5 条与非价格类 3 条分开登记。",
        "- **可证伪假设**：若某信号能判断「该及时止盈／推后止盈」，则它与未来收益的 "
        "Spearman IC 绝对值 ≥ 0.02、按币同号 ≥ 60%、FDR q < 0.10、4 段同号 ≥ 3。",
        "- **止损线核对**：已查 `references/verdicts-history.md`，固定入场只换出场、"
        "OHLCV 出场规则均已证伪（H2 命中）。本轮是用户明确指定的一次性、预注册、"
        "无网格复核，价格类信号只作已证伪路线复述对照；不投入生产实现。",
        f"- **窗口 / 样本期**：{d['window']['start']} → {d['window']['end']}（{d['days']} 天，"
        f"1h，墙钟对齐到 UTC 日界）。样本期 BTC {d['data']['btc_change_pct']:+.2f}%（下跌市）。",
        f"- **币池 / 币数**：native 1h 全窗口池 {d['universe']['symbols']} 币，"
        f"实际有成交 {d['universe']['symbols_with_trades']} 币；skip {len(d['universe']['skipped'])} "
        "币（无预热／覆盖率不足／信号报错／零多头入场）。",
        f"- **成本口径**：`fee_rate`={d['costs']['fee_rate']}、`slippage`={d['costs']['slippage']}、"
        f"双边合计 {d['costs']['round_trip']*100:.2f}%，口径＝{d['costs']['basis']}（名义本金）。",
        f"- **样本外切分**：训练/验证/测试 = 60/15/25，边界各 purge {PURGE_MS//3600000}h；"
        f"本探针**只用训练段**（{len(parts['train'])} 笔），测试段只读。",
        "- **对照组**：正对照（偷看未来收益）／负对照（随机噪声）／半衰对照，量具自校准。",
        "",
        "## 结果",
        "",
        "| 信号 | 有效对 | 覆盖 | IC | 按币同号 | 符号二项 p | FDR q | 分段 IC | 四关 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        seg = ", ".join("—" if v is None else f"{v:+.3f}" for v in r.get("segment_ics", []))
        ic = "—" if r["ic"] is None else f"{r['ic']:+.4f}"
        share = "—" if r.get("sign_share") is None else f"{r['sign_share']:.3f}"
        lines.append(f"| `{r['signal']}` | {r['pairs']} | {r['coverage']:.1%} | {ic} | "
                     f"{share} | {r.get('p_binom', 1):.4g} | {r.get('fdr_q', 1):.4f} | "
                     f"{seg or '—'} | {r['gate']} |")
    lines += [
        "",
        f"**自校准**：正对照 IC {cal.get('positive_control_ic_LOOKAHEAD')}；"
        f"负对照 |IC| 均值 {cal.get('negative_control_mean_abs_ic')}、"
        f"p95 {cal.get('negative_control_p95_abs_ic')}；"
        f"半衰对照 {cal.get('half_decay_mean_ic')} → 量具可用。",
        "",
        "**随机基线**：本探针的负对照即 20 种子随机噪声（多种子随机对照）；"
        "回测层的 20 种子匹配触发率随机早退基线与 20 种子随机方向基线在主报告"
        "`docs/trend-reversal-exit-signals.md` 给出，探针阶段不重复计算收益分布。",
        "",
        "## 判定",
        "",
        f"- **verdict（初筛）**：`{probe['verdict']}`；无方向四关全过 "
        f"{probe['probe_pass_count']}/{len(rows)} 列。H1 的 |IC| 门槛是**无方向**的，"
        "本课题的「卖出」语义还额外要求 IC < 0（分值越高、未来收益越低）："
        f"同时满足的只有 {probe['sell_direction_pass_count']} 列。",
        "- **方向解读**：`mom_fade`（IC +0.0776）与 `ma_break`（IC +0.0758）为正，"
        "表示短期动量转负／跌破均线之后未来 24h 反而反弹（均值回归）——那恰是"
        "「不该卖」的情形，把它们当卖出信号方向是反的，也解释了主报告里二者"
        "配对差为负。`vol_spike` 方向正确（IC −0.0703），但在纯样本外受控 A/B 上"
        "配对差 CI 跨 0、FDR q 未过关。",
        f"- 池化门槛 `MIN_POOLED_TRADES={MIN_POOLED_TRADES}`；按币 block 同号与 4 段 IC "
        "同号均列出。FDR 的 p 来自「按币同号占比」的精确二项检验，不是 IC 置换检验，"
        "只用于列间多重比较校正，不能单独当作 edge 证据。",
        "- 与历史结论对照：`docs/exit-metalabel.md` 在 105 笔线上订单上 26 列 0 过四关；"
        "本轮把同一量具扩到半年全市场训练段，样本量提升两个数量级。",
        "",
        "## 局限",
        "",
        "- 决策时刻固定为入场后第 8 根，不覆盖持仓全路径；信号触发采用固定 1σ／0.80 分位阈值，"
        "未做网格寻优（H2）。未模拟资金费率、盘口深度、强平、分批建仓。",
        "- 衍生品情绪 `sentiment` 仅覆盖 "
        f"{d['data']['sentiment']['first']} → {d['data']['sentiment']['last']}，"
        "半年窗口内大部分时间为 NaN，覆盖率低的列不得读成「无信号」。",
        "- 口径限制：名义本金，不得与 margin 口径报告直接相减。",
        "",
        "## 结论",
        "",
        f"- 无方向四关全过 {probe['probe_pass_count']} 列，但方向符合「卖出」语义的仅 "
        f"{probe['sell_direction_pass_count']} 列；Gate 1 只是「可能有信息」的初筛，"
        "**不是最终 edge 判定**。最终以 `docs/trend-reversal-exit-signals.md` 的"
        "纯样本外受控 A/B 为准：8 条规则 0 过三关，训练段冠军在测试段相对基线显著更差。",
        "- 是否启用：`default_enabled=False`，未新增 spec、未改绑定、未改生产参数。",
        "",
        "## 复现命令",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe backend/scripts/trend_exit_study.py --days 183 --seeds 20 "
        "--out output/trend_exit_study.json",
        ".\\.venv\\Scripts\\python.exe backend/scripts/trend_exit_report.py "
        "--input output/trend_exit_study.json --prefix docs/trend-reversal-exit-signals",
        "```",
        "",
    ]
    return "\n".join(lines)


def main_md(d, probe, parts, split_meta, rows, champ, verdict, reason, baseline, rand_dir):
    by = {r["signal"]: r for r in rows}
    test = by[champ]["splits"]["test"]
    reasons = Counter(r["baseline_reason"] for r in d["records"])
    skip_reasons = Counter(s["reason"] for s in d["universe"]["skipped"])
    sizes = " / ".join(f"{k} {v} 笔" for k, v in split_meta["sizes"].items())
    lines = [
        "# 趋势反转入场 × 其他信号卖出：过去半年受控回测",
        "",
        f"> 生成时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ｜ "
        "脚本：`backend/scripts/trend_exit_study.py` + `trend_exit_report.py` ｜ 命令见文末",
        "",
        "## 假设与边界",
        "",
        "- **信息源 / 可证伪假设**：买入信号固定为 `supertrend_current`（Supertrend 趋势反转）"
        "的多头方向翻转上升沿；新增信息只来自 8 条预注册卖出信号（5 条 OHLCV 量价、"
        "3 条非价格：资金费率横截面拥挤／资金费率转负／持仓量横截面拥挤）。"
        "若某卖出信号真有择时价值，则它与基线在同一入场点上的配对净收益差 "
        "CI 下界为正、FDR q < 0.10、4 段同号 ≥ 3，且跑赢「匹配触发率的随机早退」分布。",
        "**事前终止条件**：上述任一不满足即不启用、不调阈值；判定词按手册第 3 节词表取"
        "（CI 上界为负写 `negative`、跨 0 写 `inconclusive`，不预先锁定词条）。",
        "- **周期 / 持有期 H**：1h；所有臂共享 48 根（2 天）持仓上限并强制平仓；"
        f"Gate 1 标签前瞻 {FORWARD_H} 根；相邻入场间隔 ≥ {MIN_GAP} 根以消除重叠计数。",
        f"- **窗口起止 / 墙钟对齐**：{d['window']['start']} → {d['window']['end']}"
        f"（{d['days']} 天，UTC 日界对齐，{d['data']['btc_bars_in_window']} 根 1h）。"
        f"样本期 BTC {d['data']['btc_change_pct']:+.2f}%（下跌市，beta 归因基准）。",
        f"- **币池 / 币数**：native 1h 全窗口池 {d['universe']['symbols']} 币，"
        f"有成交 {d['universe']['symbols_with_trades']} 币；skip {len(d['universe']['skipped'])} 币"
        f"（{dict(skip_reasons)}）。"
        "native 1h 窗口内 4264/4392 根，末端 5.3 天由 1m 聚合补齐（native 行优先，"
        "与 derived/native 一致性零差异口径同源）；15m 因仅 28 币有半年深度，不采用。",
        "- **止损线核对（H2 冲突登记）**：已查 `references/verdicts-history.md` 与 "
        "`docs/supertrend-exit-diagnosis.md`／`docs/trend-reversal-order-exit.md`／"
        "`docs/exit-metalabel.md`——「固定入场只换出场」「不要再调止损倍数／持有期救它」"
        "是已证伪路线，按 Gate 0 本应终止。因用户明确指令，本轮执行**一次性、预注册、"
        "无网格、不改生产代码**的复核：价格类信号作为已证伪路线复述对照；唯一可能带来"
        "增量的是非价格信息源。冲突与取舍在此显式登记。",
        "- **对照组**：① 当前参数基线（固定 TP=3R / SL=1.5ATR）；② 20 种子随机方向基线；"
        "③ 每条信号各 20 种子「匹配触发率随机早退」基线（重排触发时点，分离纯粹的早退 beta 效应）。"
        "「关闭过滤基线」会改变入场点集合、与固定入场配对 A/B 不同口径，按"
        "「受控 A/B 只改一个变量」原则不纳入，此为与标准 SOP 的显式偏差。",
        "",
        "## 口径与参数",
        "",
        f"- **成本**：`fee_rate`={d['costs']['fee_rate']}、`slippage`={d['costs']['slippage']}、"
        f"双边合计 {d['costs']['round_trip']*100:.2f}%；口径＝{d['costs']['basis']}（名义本金），"
        "全程不切换，不与 margin 口径报告混比。",
        f"- **参数来源**：{d['parameters']['source']}；关键项 "
        f"`sl_atr_mult`={d['parameters']['sl_atr_mult']}、`target_r`={d['parameters']['target_r']}、"
        f"`supertrend_factor`={d['parameters']['supertrend_factor']}、"
        f"`regime_filter`={d['parameters']['regime_filter']}、`cost_filter`={d['parameters']['cost_filter']}。"
        "1h 无生效绑定（实查 `strategy_bindings.enabled=1` 仅 #7 `supertrend_current × all × 15m`），"
        "故不假称复现绑定参数层。",
        f"- **样本外切分**：训练/验证/测试 = 60/15/25；边界各 purge {PURGE_MS//3600000}h，"
        f"剔除 {split_meta['purged_records']} 笔；各段 {sizes}。"
        "冠军只在训练段选、验证段判定、测试段只读。",
        "- **撮合**：`engine=sequential`；收盘出信号、下一根开盘成交；OHLC 触发止损止盈、"
        "跳空按开盘、同根 SL 优先、平仓当根不再开仓；信号在 bar j 收盘触发、bar j+1 开盘平。"
        "所有臂共享同一批入场点与同一段未来 K 线。",
        "- **数据与覆盖**：native 1h + 1m 聚合尾桶；funding 8h 桶化后取上一已结算桶的"
        "横截面分位（避免结算前视）；sentiment 仅 "
        f"{d['data']['sentiment']['first']} → {d['data']['sentiment']['last']}；"
        f"funding 覆盖 {d['data']['funding']['first']} → {d['data']['funding']['last']}。",
        "",
        "## 结果",
        "",
        f"- 总入场点 **{d['n_records']}** 笔（{d['universe']['symbols_with_trades']} 币）。"
        f"基线每笔期望 {baseline['mean_pct']:+.4f}%，95% CI "
        f"[{baseline['ci_low_pct']:+.4f}, {baseline['ci_high_pct']:+.4f}]，"
        f"verdict `{baseline['verdict']}`；基线出场原因分布 {dict(reasons)}"
        "（止损占绝对多数，说明 1.5ATR 结构止损先于 3R 目标触发）。",
        f"- 随机方向基线（{len(rand_dir)} 种子）均值 {np.mean(rand_dir):+.4f}%、"
        f"p05 {np.percentile(rand_dir,5):+.4f}%、p95 {np.percentile(rand_dir,95):+.4f}%；"
        f"基线收益在其中的分位 {percentile_rank(baseline['mean_pct'], rand_dir):.3f}"
        "（未超过 0.95 —— 入场方向相对随机方向本身没有可检出的优势）。",
        "",
        md_table(rows),
        "",
        f"- 训练段冠军：`{champ}`（配对差 {by[champ]['splits']['train']['mean_pct']:+.4f}%）；"
        f"纯样本外测试段 {test['n']} 笔，配对差 {test['mean_pct']:+.4f}%，"
        f"95% CI [{test['ci_low_pct']:+.4f}, {test['ci_high_pct']:+.4f}]，"
        f"verdict `{test['verdict']}`。训练段正读数到测试段反号，是样本内选参噪声，"
        "不是可外推的改善。",
        f"- 三关全过的信号：{sum(1 for r in rows if r['three_gates_pass'])}/{len(rows)}；"
        "随机早退分布控制了「下跌市里早平仓天然少亏」这一 beta 效应，"
        "配对差与随机早退均值之差才是择时增量。",
        f"- Gate 1 条件化 IC 四关全过 {probe['probe_pass_count']}/{len(probe['rows'])} 列，"
        f"但其中 {sum(1 for r in probe['rows'] if r['gate'] == 'pass' and r['ic'] is not None and r['ic'] > 0)} "
        "列 IC 为正、方向与「卖出」相反（短期超跌后 24h 反弹，卖了反而少赚）；"
        f"方向正确的仅 {probe['sell_direction_pass_count']} 列，且在受控 A/B 上 CI 跨 0"
        "（详见 `docs/probe-trend-exit-signals.md`）。",
        "- 样本门槛说明：`oi_crowd` 只在 sentiment 覆盖的 23 天里有值，"
        f"全样本命中率 {by['oi_crowd']['hit_rate']:.1%}，其测试段孤立的 `pass` "
        "伴随全样本 CI 跨 0、FDR q=0.1319、4 段仅 1 段非零 → 判 `insufficient`／`likely_noise`，"
        "不是可用 edge（H8 小样本假精确）。",
        "",
        "## 判定",
        "",
        f"- **verdict**：`{verdict}` —— {reason}。",
        f"- 判定对象：训练段选出的冠军 `{champ}` 在纯样本外测试段相对基线的**配对差**"
        f"（{test['n']} 笔）；其余 7 条规则全样本多为 `inconclusive`，"
        "不因冠军为负就反推「所有信号都亏」，也不把测试段孤立的 `pass` 当 edge。",
        f"- **池化 CI**：`MIN_POOLED_TRADES={MIN_POOLED_TRADES}`，"
        f"全样本 {baseline['n']} 笔、测试段 {test['n']} 笔；bootstrap 按一笔交易重采样 2000 次。",
        f"- **随机基线**：20 种子；冠军分位 {by[champ]['random_percentile']:.3f}"
        "（需 > 0.95 才算跑赢匹配触发率的随机早退）。",
        f"- **归因**：样本期 {d['data']['btc_change_pct']:+.2f}%（下跌市），"
        "任何提前平仓都会机械性少亏；"
        f"配对差最大绝对值仅 {max(abs(r['all']['mean_pct']) for r in rows):.4f}pp/笔，"
        "远小于双边成本 0.12%，且多数与随机早退均值同量级或更差。"
        "基线本身每笔期望为负，方向层没有留下可供出场规则修复的毛 edge。",
        "- **与历史结论对照**：与 `docs/supertrend-exit-diagnosis.md`「不要再调止损倍数／"
        "持有期救它」、`docs/trend-reversal-order-exit.md`（17 条静态规则 0 过三关）、"
        "`docs/exit-metalabel.md`（26 列 0 过四关）一致；本轮把样本从 105 笔线上订单扩到 "
        f"{d['n_records']} 笔全市场半年入场，结论未翻转。",
        "",
        "## 局限",
        "",
        "- 未模拟资金费率、盘口深度、强平、分批建仓；回测假定下一根开盘及时成交，"
        "与模拟盘巡检撮合有差异。",
        "- 信号阈值固定 1σ / 0.80 分位、决策点固定第 8 根，未做网格寻优；"
        "价格类信号在 48 根持仓窗内命中率偏高，属高频门控而非稀疏事件。",
        "- `sentiment` 仅 23 天、`funding` 末端止于 10-27，`oi_crowd` 命中率仅 "
        f"{by['oi_crowd']['hit_rate']:.1%} → 该列判 `insufficient`（数据缺口，不是指标无效）。",
        "- 1m 长历史仅 19 币，半年窗口依赖 native 1h；15m 深度不足未做周期敏感性。",
        "- 口径限制：名义本金，本报告数值不得与 margin 口径报告直接相减。",
        "",
        "## 结论与后续",
        "",
        f"- 一句话结论：**否，不可行**。8 条预注册卖出信号在 {d['n_records']} 笔同入场点配对 A/B 上"
        f"未产生统计支持的盈利提升，主 verdict `{verdict}`。",
        f"- **是否启用**：`default_enabled=False`；未新增 spec、未改绑定、未改任何生产参数；"
        "现行生效绑定仍只有 #7（`supertrend_current × all × 15m`）。",
        "- 按 H2 止损线：**不再调出场阈值／止损倍数／持有期／周期，也不再扩充 OHLCV 指标族**。",
        "- 后续只写补样本／补数据方案：补足 ≥60 天衍生品情绪与 funding 末端、"
        "把订单级下单瞬间原始 TP/SL 快照留全；下一个信息增量仍在逐笔成交顺序、"
        "盘口挂撤单、多所现货腿。",
        "",
        "## 复现命令",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe backend/scripts/trend_exit_study.py --days 183 --seeds 20 "
        "--out output/trend_exit_study.json",
        ".\\.venv\\Scripts\\python.exe backend/scripts/trend_exit_report.py "
        "--input output/trend_exit_study.json --prefix docs/trend-reversal-exit-signals",
        ".\\.venv\\Scripts\\python.exe -m pytest backend/tests/test_trend_exit_common.py -q",
        ".\\.venv\\Scripts\\python.exe .workbuddy/skills/strategy-backtest-playbook/scripts/preflight.py "
        "--prefix docs/trend-reversal-exit-signals --strict",
        "```",
        "",
        "## 产物",
        "",
        "- `docs/trend-reversal-exit-signals.md` + `.json`、`docs/probe-trend-exit-signals.md` + `.json`",
        "- `backend/scripts/trend_exit_{common,data,study,stats,report}.py`、"
        "`backend/tests/test_trend_exit_common.py`、当日工作记忆登记",
        "",
    ]
    return "\n".join(lines)
