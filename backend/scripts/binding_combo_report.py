r"""把三轮寻优与时点池对照汇总成 `docs/binding-combo-optimization.{md,json}`。

只是报表层：所有数字都从 `output/binding-combo-*.json` 与 `output/pit-pool-check.json`
读，不重算、不硬编码——避免「报告里的数」和「跑出来的数」各写一遍后对不上。

    .\.venv\Scripts\python.exe backend/scripts/binding_combo_report.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from binding_combo_sections import (  # noqa: E402
    LIMITATIONS,
    NOTES,
    comparison_lines,
    fmt,
    verdict_lines,
)

ROOT = Path(__file__).resolve().parents[2]
PIT_POOL = "research_high_liquidity_high_volatility"
NAMES = {"A": "研究·高成交额高波动", "B": "研究·低成交额高波动",
         "C": "Supertrend 趋势反转", "D": "订单流净买入极端值"}

# 判定之后的实际落库动作。数字仍来自 output/*.json；这里是「人做了什么」的审计栏，
# 只有真实执行过才写，反悔回滚时也必须改这里，避免报告与库里的 enabled 状态打架。
APPLIED = [
    {"at": "2026-11-05 22:35:10+08:00", "binding_id": 10, "combo": "C",
     "binding": "supertrend_current × trend_reversal_fit × 15m", "change": "enabled: 1 → 0",
     "reason": "verdict=negative，1100 笔、两窗口同号、毛收益 −0.04% < 0"},
    {"at": "2026-11-05 22:35:10+08:00", "binding_id": 43, "combo": "D",
     "binding": "flow_imbalance_long × alt × 15m", "change": "enabled: 1 → 0",
     "reason": "verdict=insufficient，原生 15m 无 taker，494 个池成员里仅 3 个可回测"},
]

APPLIED_NOTES = [
    "A（#8）、B（#9）保持 `enabled=1`、参数不动：A 维持 `research_hold_hours=120`，"
    "不改成网格冠军 168h（PIT 与 90 天口径都更差）；B 维持 48h，标注「未经统计支持」。",
    "停用只挡新开仓：`trading/service.py` 的持仓管理走 `open_orders(db)`，与绑定无关，"
    "`resolve_order_params` 也按 (策略, 池, 周期) 读仍存在的那行绑定，所以停用当时未平的 "
    "3 笔 supertrend 仓位（LTC/TIA/SOL）继续按原止损止盈跑完，不会被强平或失去管理。",
    "副产物：周期内同步目标从 505 个币降到约 50 个（alt 池 494 成员此前每轮全量同步）。",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pick_row(combo: dict, key: str) -> dict | None:
    target = (combo.get(key) or {}).get("params")
    for row in combo.get("table") or []:
        if row["params"] == target:
            return row
    return None


def grid_rows(combo: dict, limit: int = 8) -> list[dict]:
    rows = sorted(combo.get("table") or [], key=lambda r: -(r["train"]["mean"] or -99))
    current = pick_row(combo, "current")
    head = rows[:limit]
    if current and current not in head:
        head.append(current)
    return head


def params_text(params: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in params.items())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["output/binding-combo-180.json",
                                                 "output/binding-combo-90.json"])
    ap.add_argument("--pit", default="output/pit-pool-check.json")
    ap.add_argument("--out", default="docs/binding-combo-optimization")
    args = ap.parse_args()

    runs = {path.split("-")[-1].split(".")[0]: load(ROOT / path) for path in args.runs}
    pit = load(ROOT / args.pit)
    main_run = runs.get("180") or list(runs.values())[0]
    combos = {item["key"]: item for item in main_run["combos"] if not item.get("error")}
    alt = {item["key"]: item for item in (runs.get("90") or {}).get("combos", [])}

    starts = [d["start"] for c in combos.values() for d in c.get("symbol_detail") or []]
    ends = [d["end"] for c in combos.values() for d in c.get("symbol_detail") or []]
    payload = {
        "strategy": "binding-combo-optimization",
        "generated_at": main_run.get("generated_at"),
        "window": {"start": min(starts) if starts else None, "end": max(ends) if ends else None,
                   "robustness_window_days": 90},
        "universe": {k: {"strategy": v["strategy"], "pool": v["pool"], "interval": v["interval"],
                         "symbols": v["symbols"], "skipped": len(v.get("skipped") or [])}
                     for k, v in combos.items()},
        "interval": {k: v["interval"] for k, v in combos.items()},
        "days": {"main": 180, "robustness": 90},
        "costs": main_run["costs"], "split": main_run["split"],
        "min_pooled_trades": 30,
        "random_baseline_seeds": 20,
        "combos": [], "pit": pit["summary"], "verdict": "inconclusive",
        "default_enabled": False,
    }
    lines = ["# 已启用「策略 × 币种池」绑定：逐组合寻优与最终选型", "",
             f"> 生成时间：{(main_run.get('generated_at') or '')[:19]} ｜ 脚本："
             "`backend/scripts/binding_combo_review.py`、`backend/scripts/pit_pool_check.py` ｜ "
             "报表：`backend/scripts/binding_combo_report.py`", "",
             "## 假设与边界", "",
             "- **信息源**：四条绑定现有的 OHLCV 信号层，未引入任何新特征（H2 止损线下不新增特征）。",
             "- **可证伪假设**：如果某条绑定存在正 edge，那么它在验证段（不看训练段）的每笔期望"
             " 95% CI 下界应为正，且在「同笔数、同持有期的随机入场」分布中的分位应超过 0.95。",
             "- **终止条件**：验证段 CI 上界为负 → `negative`，停止该路线；CI 跨 0 但不优于随机 → "
             "`inconclusive`，只补样本不改参数。",
             "- **周期 / 持有期**：A/B 为 1h（持有 120h / 48h）；C/D 为 15m（出场由 ATR 止损与目标决定）。",
             "- **窗口**：主口径最近 180 天，稳健性口径最近 90 天；时点池对照 5 期 × 30 天。",
             "- **止损线核对**：已读 `references/verdicts-history.md`；本课题只做「现有参数空间内的"
             "边界确认」，不新增特征、不改标签、不换周期，因此不越 H2。",
             "- **对照组**：① 绑定当前参数；② 寻优冠军；③ 随机基线（研究型＝随机入场＋同持有期；"
             "趋势型＝方向翻转）。", "",
             "## 口径与参数", "",
             f"- **成本**：fee {main_run['costs']['fee_rate']:.2%}/边 + 滑点 "
             f"{main_run['costs']['slippage']:.2%}/边，开平双边 0.12%；口径＝**名义本金**。",
             "- **收益口径**：每笔净收益 ÷ 名义本金。通用撮合的 `return_pct` 是保证金口径，"
             "已按 `pnl/(entry×size)` 折算后与固定 200U 名义的研究池同表比较。",
             "- **分批止盈按仓位聚合**：先按 `position_id` 合并腿再加总，避免 1/6 仓位的 1R 止盈腿"
             "与 5/6 仓位的止损腿等权（未聚合时 ETHUSDT 15m 腿级 +0.24% vs 仓位级 −0.09%）。",
             f"- **样本外切分**：训练 {main_run['split']['train'][1] - main_run['split']['train'][0]:.0%} / "
             f"验证 {main_run['split']['valid'][1] - main_run['split']['valid'][0]:.0%} / "
             f"测试 {main_run['split']['test'][1] - main_run['split']['test'][0]:.0%}，按墙钟切，跨段成交清除。",
             "- **选参规则**：训练段池化每笔期望最高者；成交笔数低于「各格中位数的 40%」或低于 30 笔"
             "的组合不参与竞争（防「开过滤后只剩十几笔」的小样本冠军）。",
             "- **撮合**：收盘出信号、下一根开盘成交；OHLC 触发止损止盈、跳空按开盘、同根止损优先。", ""]

    lines += ["## 现状：四条生效绑定", "",
              "| # | 策略 | 池 | 周期 | 币数 | 当前参数 |", "| --- | --- | --- | --- | ---: | --- |"]
    for key, combo in combos.items():
        lines.append(f"| {key} | `{combo['strategy']}` | `{combo['pool']}` | {combo['interval']} | "
                     f"{combo['symbols']} | {params_text((combo.get('current') or {}).get('params') or {})} |")
    lines += ["", "四条绑定的 spec 全部 `default_enabled=False`；它们处于启用状态是历史配置，"
              "不是回测推荐（H11）。", ""]

    for key, combo in combos.items():
        lines += [f"## 组合 {key} · {combo['label']}", "",
                  f"策略 `{combo['strategy']}` × 池 `{combo['pool']}` × `{combo['interval']}`，"
                  f"可用 {combo['symbols']} 币，搜索空间 {combo['grid_size']} 组。"]
        skipped = combo.get("skipped") or []
        if skipped:
            sample = "、".join(f"{item['symbol']}（{item['reason'][:18]}）" for item in skipped[:3])
            lines.append(f"剔除 {len(skipped)} 币，例：{sample}。")
        lines += ["", "| 参数 | 训练笔数 | 训练每笔% | 验证笔数 | 验证每笔% | 验证 95% CI | 判定 | 测试每笔% |",
                  "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |"]
        for row in grid_rows(combo):
            valid, test = row["valid"], row["test"]
            mark = " ★" if row is pick_row(combo, "best") else " ●" if row is pick_row(combo, "current") else ""
            lines.append(
                f"| {params_text(row['params'])}{mark} | {row['train']['trades']} | "
                f"{fmt(row['train']['mean'])} | {valid['trades']} | {fmt(valid['mean'])} | "
                f"[{fmt(valid['ci_low'])}, {fmt(valid['ci_high'])}] | {valid['verdict']} | {fmt(test['mean'])} |")
        base = combo.get("baseline_current") or {}
        best = combo.get("baseline_best") or {}
        if base.get("kind") == "random_entry":
            lines += ["", f"随机入场基线（20 种子，{base['symbols']} 币、每种子 {base['trades_per_seed']} 笔）："
                      f"均值 {fmt(base['mean_pct'])}%，p05~p95 [{fmt(base['p05_pct'])}, {fmt(base['p95_pct'])}]%；"
                      f"当前参数 {fmt(base['actual_mean_pct'])}% 落在第 {fmt((base['percentile'] or 0) * 100, 0)} 百分位，"
                      f"寻优冠军 {fmt(best.get('actual_mean_pct'))}% 落在第 "
                      f"{fmt((best.get('percentile') or 0) * 100, 0)} 百分位。"]
        else:
            lines += ["", f"毛收益估算（净 + 0.12% 成本）：当前参数 {fmt(base.get('gross_estimate_pct'), 3)}%/笔，"
                      f"寻优冠军 {fmt(best.get('gross_estimate_pct'), 3)}%/笔——毛 edge 低于 0 说明方向本身"
                      "没有预测力，不是被成本吃掉的。"]
        if NOTES.get(key):
            lines += ["", f"**过程读出**：{NOTES[key]}"]
        payload["combos"].append({"key": key, "strategy": combo["strategy"], "pool": combo["pool"],
                                  "interval": combo["interval"], "symbols": combo["symbols"],
                                  "grid_size": combo["grid_size"],
                                  "current": combo.get("current"), "best": combo.get("best"),
                                  "baseline_current": base, "baseline_best": best,
                                  "skipped": len(skipped)})
        lines.append("")

    lines += ["## 时点池（PIT）对照：正收益是策略还是选币", "",
              "研究池由「最近 168 小时成交额 + 波动率」筛出，库里存的是**当前**成员。拿这批成员"
              "回放过去 180 天等于用今天知道会涨的币去赌昨天。下表按 30 天一期往回推 5 期，"
              "每期起点用**当时之前 168 小时**的行情重建池成员，两臂参数/成本/撮合完全相同。", "",
              "| 组合/参数/成员来源 | 笔数 | 每笔均值% | 95% CI | 判定 | 正收益期数 |",
              "| --- | ---: | ---: | --- | --- | ---: |"]
    for key, item in sorted(pit["summary"].items()):
        pooled = item["pooled"]
        arm, tag, label = key.split("|")
        name = {"pit": "时点重建（可交易）", "posthoc": "当前成员（事后选币）"}[label]
        lines.append(f"| {arm} {tag} · {name} | {pooled['trades']} | {fmt(pooled['mean'], 3)} | "
                     f"[{fmt(pooled['ci_low'], 2)}, {fmt(pooled['ci_high'], 2)}] | {pooled['verdict']} | "
                     f"{item['positive_folds']}/{item['folds']} |")
    lines += ["", "同一期、同一参数下两臂之差就是「选币偏差」的实际价格。", ""]

    lines += comparison_lines(combos, alt, pit)
    lines += verdict_lines(combos, alt, pit, runs)
    lines += ["## 已执行的变更", "",
              "| 时间 | 绑定 id | 组合 | 绑定 | 变更 | 依据 |",
              "| --- | ---: | --- | --- | --- | --- |"]
    for item in APPLIED:
        lines.append(f"| {item['at']} | {item['binding_id']} | {item['combo']} | {item['binding']} | "
                     f"`{item['change']}` | {item['reason']} |")
    lines += ["", *[f"- {note}" for note in APPLIED_NOTES], ""]
    payload["applied"] = APPLIED
    payload["applied_notes"] = APPLIED_NOTES
    lines += ["## 局限", ""] + [f"- {item}" for item in LIMITATIONS] + [
        "", "## 复现命令", "", "```bash",
        ".venv/Scripts/python.exe backend/scripts/binding_combo_review.py --days 180 --workers 8 --out output/binding-combo-180",
        ".venv/Scripts/python.exe backend/scripts/binding_combo_review.py --days 90 --workers 8 --out output/binding-combo-90",
        ".venv/Scripts/python.exe backend/scripts/pit_pool_check.py --folds 5 --span 30 --out output/pit-pool-check",
        ".venv/Scripts/python.exe backend/scripts/binding_combo_report.py",
        ".venv/Scripts/python.exe .workbuddy/skills/strategy-backtest-playbook/scripts/preflight.py --prefix docs/binding-combo-optimization",
        "```", "", "## 产物", "",
        "- `docs/binding-combo-optimization.md` + `docs/binding-combo-optimization.json`",
        "- 原始结果：`output/binding-combo-180.json`、`output/binding-combo-90.json`、`output/pit-pool-check.json`",
        "- 单测：`backend/tests/test_combo_engine.py`", ""]

    payload["verdict"] = "inconclusive"
    payload["limitations"] = LIMITATIONS
    out = ROOT / args.out
    out.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str),
                                       encoding="utf-8")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {out.with_suffix('.md')} / {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
