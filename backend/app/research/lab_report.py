"""实验报告输出：markdown 给人看，json 给机器复核，两份由同一次运行写出。"""

from __future__ import annotations

import json
from pathlib import Path


def _num(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    text = f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return text.replace("|", "\\|")


def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_num(cell) for cell in row) + " |")
    return "\n".join(lines)


def _single_row(index: int, row: dict) -> list:
    train, test = row["train"], row["test"]
    return [
        index,
        row["label"],
        train["trades"],
        _num(train["win_rate"] * 100 if train["win_rate"] is not None else None),
        _num(train["payoff"]),
        _num(train["total_pnl"], 3),
        _num(train["total_return_pct"], 2),
        _num(train["max_drawdown_pct"], 2),
        _num(row.get("q_value"), 4),
        _num(test["total_pnl"], 3),
        row.get("verdict", ""),
    ]


SINGLE_HEADERS = [
    "#", "候选（指标 + 参数）", "交易数", "胜率%", "盈亏比", "累计盈亏",
    "收益率%", "回撤%", "q 值", "样本外盈亏", "判定",
]


def family_table(singles: list[dict]) -> str:
    families: dict[str, dict] = {}
    for row in singles:
        bucket = families.setdefault(row["family"], {"best_train": row, "best_test": row})
        if row["train"]["total_pnl"] > bucket["best_train"]["train"]["total_pnl"]:
            bucket["best_train"] = row
        if row["test"]["total_pnl"] > bucket["best_test"]["test"]["total_pnl"]:
            bucket["best_test"] = row
    rows = []
    for name, bucket in sorted(families.items(), key=lambda kv: kv[1]["best_train"]["train"]["total_pnl"], reverse=True):
        best_train, best_test = bucket["best_train"], bucket["best_test"]
        rows.append([
            name,
            len([r for r in singles if r["family"] == name]),
            best_train["label"].split("(", 1)[-1].rstrip(")"),
            best_train["train"]["trades"],
            _num(best_train["train"]["total_pnl"], 3),
            _num(best_train["test"]["total_pnl"], 3),
            _num(best_test["test"]["total_pnl"], 3),
        ])
    return _table(
        ["指标", "参数组合数", "训练集最优参数", "交易数", "训练盈亏", "该组样本外盈亏", "本组样本外最佳"],
        rows,
    )


def build_markdown(payload: dict) -> str:
    cfg, meta = payload["config"], payload["dataset"]
    summary = payload["summary"]
    lines: list[str] = []
    lines.append("# 指标信号检测与优化实验")
    lines.append("")
    lines.append(f"> 生成时间 {cfg['generated_at']}；配置哈希 `{cfg['config_sha256'][:16]}`；"
                 f"随机种子 {cfg['seed']}。本文件与同名 json 由同一次运行写出，数值不得手工修改。")
    lines.append("")
    lines.append("## 0. 复现")
    lines.append("")
    lines.append("```bash")
    lines.append(f".venv\\Scripts\\python.exe backend/scripts/indicator_lab.py --interval {cfg['interval']} "
                 f"--days {cfg['days']} --limit {cfg['limit']} --top {cfg['top']} "
                 f"--draws {cfg['draws']} --seed {cfg['seed']}")
    lines.append(".venv\\Scripts\\python.exe -m pytest backend/tests/test_indicator_lab.py -q")
    lines.append("```")
    lines.append("")
    lines.append("换周期只改 `--interval`（15m / 1h / 4h / 1d 均支持，超时按 bar 数计）；"
                 "换股票池改 `--limit`，指标集合改 `--families`。所有随机性来自 `--seed`，同参数重跑数值完全一致。")
    lines.append("")
    lines.append("## 1. 实验口径")
    lines.append("")
    lines.append(f"- 数据：{meta['interval']} **native** 直查，{meta['symbols']} 个币，"
                 f"{meta['start'][:16]} → {meta['end'][:16]}，共 {meta['bars_total']} 根 bar")
    lines.append(f"- 出场：统一 `rsi50_cross`（RSI14 Wilder 穿越 50，下一根开盘平仓），"
                 f"超时兜底 {cfg['max_hold_bars']} 根，样本末尾按收盘平仓")
    lines.append(f"- 入场：指标信号在 bar t 收盘确认，t+1 开盘成交；单币同时最多一仓，不加仓")
    side_label = {"both": "多空对称（两个方向都开仓）", "long": "**只做多**（空头信号全部忽略）",
                  "short": "只做空（多头信号全部忽略）"}.get(cfg.get("side", "both"), "多空对称")
    lines.append(f"- 方向：{side_label}")
    lines.append(f"- 资金：每笔固定 {cfg['stake']} 名义本金；本金基数 = 名义本金 × 最大并发持仓数")
    lines.append(f"- 成本：手续费 {cfg['fee'] * 100:.3f}%/边 + 滑点 {cfg['slippage'] * 100:.3f}%/边，"
                 f"往返 {cfg['round_trip_pct']:.3f}%")
    lines.append(f"- 切分：训练集 {cfg['train_ratio']:.0%} / 样本外 {1 - cfg['train_ratio']:.0%}，"
                 f"切点 {meta['split'][:16]}；跨越切点的持仓两边都剔除（purge）")
    lines.append(f"- 搜索规模：单指标 {summary['single_tests']} 组参数 + 组合 {summary['combo_tests']} 组，"
                 f"合计 {summary['total_tests']} 次检验")
    lines.append("")
    lines.append("## 2. 结论摘要")
    lines.append("")
    best = summary.get("best_single")
    best_combo = summary.get("best_combo")
    lines.append(f"- 训练集最优单指标：**{best['label'] if best else '无'}**"
                 + (f"，训练盈亏 {best['train']['total_pnl']:.3f}，样本外 {best['test']['total_pnl']:.3f}，"
                    f"判定 `{best['verdict']}`" if best else ""))
    lines.append(f"- 训练集最优组合：**{best_combo['label'] if best_combo else '无'}**"
                 + (f"，训练盈亏 {best_combo['train']['total_pnl']:.3f}，样本外 {best_combo['test']['total_pnl']:.3f}，"
                    f"判定 `{best_combo['verdict']}`" if best_combo else ""))
    lines.append(f"- 通过全部关卡（q<0.10 且样本外为正）的候选：**{summary['passed']} / {summary['total_tests']}**")
    lines.append(f"- 训练集为正 {summary['train_positive']} 个，样本外为正 {summary['test_positive']} 个，"
                 f"两者同时为正 {summary['consistent']} 个（样本外按盈亏排序前 5："
                 + "、".join(f"`{name}`" for name in summary.get("test_top", []))
                 + "；仅作诊断，不用于选择）")
    bench = payload["benchmarks"]
    lines.append(f"- 基准（同口径 {cfg['stake']} 名义本金/币，全期）：始终做多 {bench['always_long']['full']['total_pnl']:.3f}，"
                 f"始终做空 {bench['always_short']['full']['total_pnl']:.3f}，"
                 f"买入持有 {bench['buy_hold']['full']['total_pnl']:.3f}")
    lines.append(f"- 结构提示：买入持有训练期 {bench['buy_hold']['train']['total_pnl']:.3f}、"
                 f"样本外 {bench['buy_hold']['test']['total_pnl']:.3f}，两段市场方向"
                 + ("相反" if bench['buy_hold']['train']['total_pnl'] * bench['buy_hold']['test']['total_pnl'] < 0 else "一致")
                 + "；训练期为正的候选只有 "
                 f"{summary['train_positive']} 个而样本外为正的有 {summary['test_positive']} 个，"
                 "说明结论对市场状态高度敏感，不能跨 regime 外推")
    lines.append("")
    lines.append("## 3. 可搜索指标清单")
    lines.append("")
    lines.append("参数一律按闭区间等差取值；`约束` 表示展开时的额外过滤。")
    lines.append("")
    grid_rows = []
    for name, spec in payload["grid"].items():
        params = "; ".join(
            f"{key}∈[{val['min']},{val['max']}]步{val['step']}" if "min" in val else f"{key}∈{{{','.join(map(str, val['values']))}}}"
            for key, val in spec["params"].items()
        )
        grid_rows.append([name, spec["label"], params, spec["constraint"] or "—", spec["combos"], spec["rule"]])
    lines.append(_table(["指标", "含义", "参数区间/步长", "约束", "组合数", "入场规则"], grid_rows))
    lines.append("")
    lines.append("## 4. 单指标结果（按训练集累计盈亏排序，前 15）")
    lines.append("")
    lines.append(_table(SINGLE_HEADERS, [_single_row(i, r) for i, r in enumerate(payload["singles_top"], 1)]))
    lines.append("")
    best_full = sorted(payload["singles_all"], key=lambda r: r["all"]["total_pnl"], reverse=True)[:5]
    lines.append("### 4.1 全期累计盈亏前 5（仅描述，不作选择依据）")
    lines.append("")
    lines.append("按全期排序已经用到了样本外数据，因此只能用来描述「历史上谁赚得多」，"
                 "不能作为选择依据；选择依据始终是训练集 + 样本外判定。")
    lines.append("")
    lines.append(_table(
        ["#", "候选", "交易数", "训练期盈亏", "样本外盈亏", "全期盈亏", "全期回撤%", "判定"],
        [[i, r["label"], r["all"]["trades"], _num(r["train"]["total_pnl"], 3),
          _num(r["test"]["total_pnl"], 3), _num(r["all"]["total_pnl"], 3),
          _num(r["all"]["max_drawdown_pct"], 2), r["verdict"]]
         for i, r in enumerate(best_full, 1)],
    ))
    lines.append("")
    lines.append("## 5. 家族汇总")
    lines.append("")
    lines.append(family_table(payload["singles_all"]))
    lines.append("")
    best = payload["summary"].get("best_single")
    if best:
        family_rows = [r for r in payload["singles_all"] if r["family"] == best["family"]]
        family_rows = sorted(family_rows, key=lambda r: r["train"]["total_pnl"], reverse=True)[:10]
        lines.append(f"### 5.1 最优指标所在家族 `{best['family']}` 的参数敏感性")
        lines.append("")
        lines.append("如果最优参数是孤立的尖峰、而邻域全为负，说明它是噪声而不是结构。")
        lines.append("")
        lines.append(_table(SINGLE_HEADERS, [_single_row(i, r) for i, r in enumerate(family_rows, 1)]))
        lines.append("")
    lines.append("## 6. 组合结果（前 10）")
    lines.append("")
    combo_rows = []
    for index, row in enumerate(payload["combos_top"], 1):
        train, test = row["train"], row["test"]
        combo_rows.append([
            index, row["rule"], row["label"], train["trades"], _num(train["win_rate"] * 100),
            _num(train["payoff"]), _num(train["total_pnl"], 3), _num(train["max_drawdown_pct"], 2),
            _num(row.get("q_value"), 4), _num(test["total_pnl"], 3), row.get("verdict", ""),
        ])
    lines.append(_table(
        ["#", "规则", "组合", "交易数", "胜率%", "盈亏比", "训练盈亏", "回撤%", "q 值", "样本外盈亏", "判定"],
        combo_rows,
    ))
    lines.append("")
    lines.append("## 7. 基准对比")
    lines.append("")
    bench_rows = [[name, row["full"]["trades"], _num(row["train"]["total_pnl"], 3),
                   _num(row["test"]["total_pnl"], 3), _num(row["full"]["total_pnl"], 3),
                   _num(row["full"]["win_rate"] * 100 if row["full"]["win_rate"] is not None else None),
                   _num(row["full"]["max_drawdown_pct"], 2)]
                  for name, row in payload["benchmarks"].items()]
    lines.append(_table(["基准", "交易数", "训练期盈亏", "样本外盈亏", "全期盈亏", "胜率%", "全期回撤%"], bench_rows))
    lines.append("")
    lines.append("买入持有按两段分别计（各自 10 本金、各自一次往返），全期列是单笔持有到期末，两者不可相加；"
                 "两段方向相反正是本实验最关键的结构性事实。")
    lines.append("")
    passed_rows = [r for r in payload["singles_all"] + payload.get("combos_all", []) if r.get("verdict") == "candidate"]
    bench_long, bench_hold = bench["always_long"], bench["buy_hold"]
    lines.append("### 7.1 通过关卡的候选 vs 基准（同口径收益率）")
    lines.append("")
    if passed_rows:
        lines.append("收益率 = 累计盈亏 / (10 × 该段最大并发)，即「占用资金回报」，可与基准直接比。"
                     "超额为负表示跑不过「什么都不筛、一直做多」的基准。")
        lines.append("")
        lines.append(_table(
            ["候选", "训练收益%", "样本外收益%", "全期收益%", "样本外回撤%", "样本外超额%", "全期超额%"],
            [[r["label"], _num(r["train"]["total_return_pct"], 2), _num(r["test"]["total_return_pct"], 2),
              _num(r["all"]["total_return_pct"], 2), _num(r["test"]["max_drawdown_pct"], 2),
              _num(r["test"]["total_return_pct"] - bench_long["test"]["total_return_pct"], 2),
              _num(r["all"]["total_return_pct"] - bench_hold["full"]["total_return_pct"], 2)]
             for r in sorted(passed_rows, key=lambda r: r["test"]["total_return_pct"], reverse=True)],
        ))
        lines.append("")
        lines.append(f"- 同期基准：always-long 样本外 {bench_long['test']['total_return_pct']:.2f}%，"
                     f"买入持有全期 {bench_hold['full']['total_return_pct']:.2f}%、样本外 {bench_hold['test']['total_return_pct']:.2f}%")
        lines.append("- 只用样本外为正作为关卡时，单边行情会让大多数候选「通过」；必须看相对基准的增量，"
                     "否则测的是 beta 不是 alpha。")
    else:
        lines.append("无候选通过统计关卡，无需比较。")
    lines.append("")
    lines.append("## 8. 滚动验证（锚定式）")
    lines.append("")
    fold_rows = [[f["fold"], f["train_end"][:10], f["test_end"][:10], f["selected"],
                  _num(f["train"]["total_pnl"], 3), f["test"]["trades"], _num(f["test"]["total_pnl"], 3)]
                 for f in payload["walk_forward"]["folds"]]
    lines.append(_table(["折", "训练截止", "验证截止", "该折选中", "训练盈亏", "验证交易数", "验证盈亏"], fold_rows))
    lines.append("")
    lines.append("## 9. 随机基线")
    lines.append("")
    rb = payload["random_baseline"]
    lines.append(f"- 对象：{rb['target']}；{rb['draws']} 次同频率随机入场（seed={rb['seed']}）")
    if rb.get("draws"):
        lines.append(f"- 实际 {rb['actual_total_pnl']:.3f} vs 随机均值 {rb['baseline_mean']:.3f}、"
                     f"95 分位 {rb['baseline_p95']:.3f}，分位 {rb['percentile']:.2%}，p={rb['p_value']:.3f}")
    else:
        lines.append("- 本轮跳过（--skip-baseline 或没有可评估候选）")
    lines.append("")
    lines.append("## 10. 过拟合控制与可复现")
    lines.append("")
    for item in payload["controls"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 11. 限制")
    lines.append("")
    for item in payload["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_reports(prefix: Path, payload: dict) -> tuple[Path, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    return md_path, json_path
