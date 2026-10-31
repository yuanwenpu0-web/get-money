"""RGD 回测报告渲染：只做 markdown 字符串拼装，不读数据、不改状态。"""
from __future__ import annotations


def _ci_cell(ci: dict) -> str:
    if "mean_return_pct" not in ci:
        return "无成交（no_trades）"
    return (
        f"{ci['mean_return_pct']:+.4f}%　[{ci['ci_low_pct']:+.4f}%, "
        f"{ci['ci_high_pct']:+.4f}%]　**{ci['verdict']}**（{ci['trades']} 笔）"
    )


def _verdict_text(ci: dict, min_trades: int) -> str:
    verdict = ci.get("verdict")
    if verdict == "positive":
        return (
            f"池化 CI 下界为正（{ci['ci_low_pct']:+.4f}%）→ 存在可判定的正 edge。"
            "下一步是换一段未参与的时间窗做 holdout 复检，通过后再谈启用。"
        )
    if verdict == "negative":
        return (
            f"池化 CI 上界为负（{ci['ci_high_pct']:+.4f}%）→ 在当前口径下是负贡献，"
            "不要靠微调阈值救它：样本内变好大概率是噪声。"
        )
    if verdict == "insufficient":
        return (
            f"成交 {ci['trades']} 笔低于池化门槛 {min_trades} 笔 → **不给结论**。"
            "笔数越少区间越宽，把「判不了」读成「有希望」是这类回测最常见的事故。"
        )
    return (
        "池化 CI 跨 0 → **不可判定**：既不能说有 edge，也不能说没有。"
        "样本不足时的默认值是「不启用」。"
    )


def render_markdown(payload: dict) -> str:
    base = payload["baseline"]
    ci = base["expectancy_ci"]
    min_trades = payload["min_pooled_trades"]
    total_exit = max(sum(base["status_counts"].values()), 1)
    lines = [
        "# Regime-Gated Donchian v2 [RGD] — 15m 主周期 / 4h HTF 回测报告",
        "",
        f"生成时间：{payload['generated_at']}　主周期：**{payload['interval']}**　"
        f"HTF：**{payload['htf_interval']}**　"
        f"窗口：{payload['days']} 天（{payload['window']['start'][:10]} → {payload['window']['end'][:10]}）",
        f"币种：{payload['universe']['evaluated']}/{payload['universe']['requested']} 个有可用连续K线　"
        f"进程：{payload['processes']}　耗时：{payload['elapsed_seconds']}s",
        "",
    ]
    thin = payload["universe"].get("insufficient_history") or []
    if thin:
        lines.append(
            f"覆盖度不足、未参与评估的币（{len(thin)} 个）：{', '.join(thin)}。"
            "这些币的 15m 历史达不到本次窗口要求，**跳过不等于没有信号**。"
        )
        lines.append("")
    lines.extend([
        "## 四组矩阵（方向 × 成本）",
        "",
        "| 组 | 方向 | fee | slippage | 有成交币 | 成交 | 池化每笔期望 | 95% CI | 判定 | 账户收益中位数 | 超随机占比 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for group in payload["groups"]:
        gci = group["expectancy_ci"]
        slip = group["slippage_used"]
        slip_cell = f"{slip:.6g}" if isinstance(slip, (int, float)) else "每币标量"
        lines.append(
            f"| **{group['id']}** | {'Long & Short' if group['dir_mode'] == 0 else 'Long Only'} | "
            f"{group['fee_rate']} | {slip_cell} | "
            f"{group['symbols_with_trades']}/{group['symbols_total']} | {group['trades']} | "
            f"{gci.get('mean_return_pct', 0.0):+.4f}% | "
            f"[{gci.get('ci_low_pct', 0.0):+.4f}%, {gci.get('ci_high_pct', 0.0):+.4f}%] | "
            f"{gci.get('verdict', '')} | {group['account_return_median_pct']:+.2f}% | "
            f"{group['beats_random_ratio']:.0%} |"
        )
    lines.extend([
        "",
        "G1 是主报告（合约多空 + 项目全局成本）；G2 是现货仅多对照；"
        "G3/G4 用脚本原样的 commission 0.05% + slippage 2 ticks。"
        "2 ticks 没有每币 tickSize 数据，按 `utils.round_price` 的精度约定做每币标量代理（契约 §8 R15），"
        "**属于近似、不是逐位复制**，只作 TradingView 方向性对齐。",
        "",
        "## 主报告 G1：结论速览",
        "",
        f"- 有成交的币种 **{base['symbols_with_trades']}/{base['symbols_total']}** 个，"
        f"合计 **{base['trades']} 笔**成交（突破信号根数 {base['signal_bars']}）",
        f"- 单币账户收益中位数 **{base['account_return_median_pct']:+.2f}%**，"
        f"均值 {base['account_return_mean_pct']:+.2f}%",
        f"- 正收益币种 {base['positive_symbols']}/{base['symbols_with_trades']}　"
        f"平均最大回撤 {base['max_drawdown_mean_pct']:.2f}%　"
        f"平均持仓占比 {base['exposure_mean_pct']:.1f}%",
        "",
        f"### 池化判据（{base['trades']} 笔，门槛 {min_trades} 笔）",
        "",
        f"- {_ci_cell(ci)}",
        f"- 重采样为正的比例 {ci.get('positive_share', 0.0):.1%}（>95% 才算统计上站得住）",
        f"- 随机基线（{payload['random_baseline_seeds']} 种子，循环位移决策列）"
        f"账户收益均值 {base['random_mean_pct']:+.3f}%",
        f"- 账户收益超过自身随机基线 95 分位的币种占比 {base['beats_random_ratio']:.0%}",
        "",
        "### 多空分解",
        "",
    ])
    for side, label in (("long", "做多（向上突破 / 回踩）"), ("short", "做空（向下突破 / 反抽）")):
        split = base["side_split"][side]
        cell = _ci_cell(split) if "mean_return_pct" in split else "无成交"
        lines.append(f"- **{label}**：{cell}")
    lines.extend([
        "",
        "判读规则与全项目一致：CI 下界 > 0 才算有正 edge；跨 0 不可判定；上界 < 0 是负贡献。",
        "随机基线在币种之间离散度极大，所以「是否跑赢随机」只能逐币比它自己的分布。",
        "",
        "## 结论与处置",
        "",
        _verdict_text(ci, min_trades),
        "",
        "1. 策略已完整实现并可手动绑定（`rgd_donchian_v2`，`default_enabled=False`），"
        "不参与自动巡检的资金竞争。",
        "2. 本回测不做 train/test split：脚本的 32 个参数是固定默认值检验，`spec.grid` 为空，"
        "没有可寻优的自由度，因此也不存在「验证集选参」这一步。",
        "3. 实盘持仓管理（TP1 限价挂单 + 动态 trailing 状态机）不在本次交付范围（契约 §2.2 / OPEN-10）。",
        "",
        "## 出场结构（G1）：腿级成交 vs 仓位级",
        "",
    ])
    leg_status = base.get("fill_status_counts") or {}
    leg_total = max(sum(leg_status.values()), 1)
    if leg_status:
        lines.extend([
            f"下表是**腿级**成交（一个仓位可能有多腿：TP1 部分平 + 后续 trailing 出场），"
            f"因此合计 {leg_total} 腿 ≠ {base['trades']} 个仓位。",
            "",
            "| 腿级 status | 笔数 | 占比 |",
            "| --- | --- | --- |",
        ])
        for reason, count in leg_status.items():
            lines.append(f"| {reason} | {count} | {count / leg_total:.0%} |")
        lines.extend([
            "",
            f"TP1 触发率 {base.get('tp1_positions', 0)}/{base['trades']} = "
            f"{base.get('tp1_position_ratio', 0.0):.1%}",
        ])
    else:
        lines.append(
            "本次 JSON 未采集腿级成交字段（`fill_status_counts`），无法拆分部分平仓；"
            "以下仅给出仓位级最终出场。用当前脚本重跑一次回测即可获得腿级分布。"
        )
    lines.extend([
        "",
        "仓位级统计（`aggregate_positions` 合并多腿、后一腿 status 覆盖前一腿）"
        "在 JSON 的 per_symbol / expectancy 口径中使用，故仓位级只看得到最终 `stop_loss`。",
        "",
        "| 仓位级 status | 笔数 | 占比 |",
        "| --- | --- | --- |",
    ])
    for reason, count in base["status_counts"].items():
        lines.append(f"| {reason} | {count} | {count / total_exit:.0%} |")
    lines.extend([
        "",
        "### 信号被拦截原因（G1，池化）", "", "| 原因 | 根数 |", "| --- | --- |",
    ])
    for reason, count in list(base["blocked"].items())[:10]:
        lines.append(f"| {reason} | {count} |")
    lines.extend([
        "",
        "`stop_order_requires_dedicated_broker` 是 `signals()` 的固定拦截原因："
        "RGD 的 direction 恒为 0，通用撮合即使被误调也只会零成交（安全失败），"
        "真实成交由专用 broker 产出。",
        "",
        "## 逐币种（G1，按账户收益排序）",
        "",
        "| 币种 | 成交 | 账户收益 | 胜率 | 最大回撤 | 随机基线均值 |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    ranked = sorted(base["per_symbol"], key=lambda item: -(item["return_pct"] or 0.0))
    for item in ranked[:25]:
        lines.append(
            f"| {item['symbol']} | {item['trades']} | {item['return_pct']:+.2f}% | "
            f"{item['win_rate']:.1f}% | {item['max_drawdown_pct']:.2f}% | "
            f"{(item['random_mean_pct'] or 0.0):+.2f}% |"
        )
    if len(ranked) > 25:
        lines.append(f"| … | 其余 {len(ranked) - 25} 币见 `rgd-backtest.json` | | | | |")
    lines.extend([
        "",
        "## 口径与数据限制（必须先读）",
        "",
        "1. **撮合口径**：完全复刻 Pine 默认订单语义——信号收盘下市价单、次根开盘成交；"
        "信号 bar 成交当根无止损（入场 bar 裸奔）；TP1 限价部分平 50%；"
        "保本止损滞后一根生效；接受度加权 trailing 逐根 ratchet 不回退；"
        "不强制收盘平仓（末根未平仓按盯市计入净值）。",
        "2. **HTF 因果**：15m 帧只看到「上一根已完成」的 4h bar（`searchsorted` 于 `t - 4h`），"
        "同一 4h 桶内 16 根 15m 的 hh/ll 完全相同，未完成桶已剔除。",
        "3. **跨缺口**：K 线缺口把序列切成多段，各段独立撮合、账户权益按段复利结转。",
        f"4. **最短段长**：单段不足 {payload['min_bars']} 根的片段直接丢弃（指标预热需要 600 根）。",
        "5. **样本口径**：每个币种是独立的 10,000 账户；全市场数字是截面统计，"
        "不等于一个组合账户的收益。15m 历史只覆盖部分币种，`universe.skipped` 已列明被跳过的币，"
        "**零成交不能读成没有信号**。",
        "",
        f"耗时 {payload['elapsed_seconds']}s，{payload['processes']} 进程。",
        "",
    ])
    return "\n".join(lines)
