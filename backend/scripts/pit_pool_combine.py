r"""P17 / H-E 双臂合并报告：把 1h 与 15m 两份判定并列成一页结论。

**只并列，不池化**：两臂样本期不同（365 天 vs 180 天），把成交混在一起算一个 CI
就是手册 H3 禁止的混比。合并层只做三件事：课题级 verdict、泄漏差汇总、局限声明。

    .\.venv\Scripts\python.exe backend/scripts/pit_pool_combine.py \
        --inputs docs/supertrend-pit-pool-1h.json docs/supertrend-pit-pool-15m.json \
        --out docs/supertrend-pit-pool
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.research.pit_pool_stats import arm_names  # noqa: E402

CARD = "research/hypotheses/H-E-supertrend-pit-pool.md"
NEXT_CARD = "research/hypotheses/H-F-funding-multi-venue.md"

LIMITATIONS = [
    "无上市／下架事件表：PIT 合格性由在库 1h bar 的存在性与新鲜度推断，是「当时可交易」的下界，不是交易所事实。",
    "排名候选先按全窗口覆盖率裁剪再取 TopN；该过滤对所有臂（含 all）一视同仁，臂间比较公平，但用了全窗口信息，故为「相对 PIT」。",
    "排名只用成交额与波动率，不含 spread／盘口深度／资金费率；真实可成交性比本口径更严格。",
    "每币独立账户等权，不是组合层面资金分配；账户均值% 只用于随机基线分位比较，不能与每笔期望% 混比（H3）。",
    "随机基线只翻转方向、不随机化选币；C4 检验的是方向信息量，选币信息量由泄漏差与 C1–C3 回答。",
    "1h 与 15m 样本期不同（365 天 vs 180 天），两周期数字不可直接比大小，本报告不做跨周期池化。",
    "15m 达标宇宙仅 30 币，N=30 的臂与 all 基线实质同构（池规模退化），不能用来回答「池该取多大」。",
    "posthoc_* 是按当期已实现收益选币的作弊臂，只用于量化泄漏，不参与 gate、不进 FDR 家族。",
]


def combine_verdict(verdicts: list[str]) -> str:
    """课题级判定：任一 pass 才 pass；全 negative 才 negative；其余一律降级。"""
    if "pass" in verdicts:
        return "pass"
    if verdicts and all(v == "no_trades" for v in verdicts):
        return "no_trades"
    if verdicts and all(v in ("insufficient", "no_trades") for v in verdicts):
        return "insufficient"
    if verdicts and all(v == "negative" for v in verdicts):
        return "negative"
    return "inconclusive"


def combine(payloads: dict[str, dict]) -> dict:
    leakage = []
    for interval, payload in payloads.items():
        for n in payload["protocol"]["sizes"]:
            post = payload["summary"].get(f"posthoc_{n}", {}).get("pooled", {})
            liq = payload["summary"].get(f"pit_liq_{n}", {}).get("pooled", {})
            pm, lm = post.get("mean_return_pct"), liq.get("mean_return_pct")
            if pm is None or lm is None:
                continue
            leakage.append({"interval": interval, "n": n, "posthoc_mean_pct": pm,
                            "pit_liq_mean_pct": lm, "gap_pp": round(pm - lm, 4)})
    first = next(iter(payloads.values()))
    limits = list(LIMITATIONS)
    for interval, payload in payloads.items():
        limits += [f"{interval}：{item}" for item in payload.get("limitations", [])
                   if "凑不满" in item]
    return {
        "name": "supertrend-pit-pool", "strategy": first["strategy"],
        "hypothesis_card": CARD, "next_hypothesis_card": NEXT_CARD,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verdict": combine_verdict([p["verdicts"]["overall"] for p in payloads.values()]),
        "verdict_scope": "固定规模 PIT 币池下、冻结参数的净收益统计判定；非「换指标再试一次」的许可。",
        "default_enabled": False,
        "binding_state": "趋势反转绑定 #7 维持 enabled=0；本轮不改任何绑定启用状态、不改 spec 默认参数。",
        "authorization": "只读回测／探针；不下真实订单、不启用 live API key、不切换交易环境。",
        "costs": first["costs"], "parameters": first["parameters"],
        "window": {iv: p["window"] for iv, p in payloads.items()},
        "universe": {iv: p["universe"] for iv, p in payloads.items()},
        "intervals": payloads, "leakage": leakage, "limitations": limits,
    }


def _signed(value, spec: str = "+.4f") -> str:
    return "—" if value is None else format(value, spec)


def render_combined(payload: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add("# P17 / H-E · Supertrend「缩小币种池」PIT 可行性判定 · 双臂合并结论")
    add("")
    add(f"- 预登记假设卡：`{payload['hypothesis_card']}`（判据 C1–C5 **跑之前**写死）")
    add(f"- 生成时间：{payload['generated_at']} · 策略 `{payload['strategy']}`")
    add(f"- **课题判定：`{payload['verdict']}`**")
    add(f"- `verdict_scope`：{payload['verdict_scope']}")
    add(f"- 授权范围：{payload['authorization']}")
    add("")
    add("## 1. 结论与启用决策")
    add("")
    add(f"**一句话结论：在事前可得的成交额／波动率排名下，把 Supertrend 趋势反转限制在 "
        f"N∈{{10,20,30}} 的固定规模 PIT 币池，1h 与 15m 两个周期**全部判 `{payload['verdict']}`**；"
        f"「缩小币种池」不能让这套参数转正。**")
    add("")
    add(f"- `default_enabled={payload['default_enabled']}`；{payload['binding_state']}")
    add("- **不启用**任何新绑定，**不改**任何 spec 默认参数（H11：无 pass 不得启用）。")
    add("- 预登记终止条件已触发：**不得换分档指标、不得换 N、不得换排名窗口重试**。"
        "本路线在现货 OHLCV 选币口径上到此为止（H2 止损线）。")
    add(f"- 唯一有强先验信号的剩余路线是资金费率（非价格预测），已立项：`{NEXT_CARD}`。")
    add("")
    add("## 2. 预登记判据（与关系，任一不过即不得写 pass）")
    add("")
    add("| 编号 | 条件 |")
    add("|---|---|")
    add("| C1 | 池化每笔期望 bootstrap 95% CI 下界 > 0 |")
    add("| C2 | 池化成交笔数 ≥ 30 |")
    add("| C3 | 正收益 fold 数 ≥ ceil(0.8 × folds) |")
    add("| C4 | 组合账户收益在多种子随机方向基线中的分位 > 0.95 |")
    add("| C5 | 六个 PIT 臂 BH-FDR q < 0.10 |")
    add("")
    add("## 3. 两周期并列结果（不池化）")
    add("")
    for interval, sub in payload["intervals"].items():
        add(f"### {interval} · 样本期 {sub['days']} 天 · {sub['folds']} folds · "
            f"判定 `{sub['verdicts']['overall']}`")
        add("")
        add(f"- 窗口：{sub['window']['start']} → {sub['window']['end']}（UTC）")
        uni = sub["universe"]
        add(f"- 币池：覆盖率 ≥{uni.get('coverage_threshold', 0.95):.0%}；窗口内有 bar "
            f"{uni['requested']} 币，达标 {uni.get('qualified', uni['requested'])} 币，"
            f"实际产出 {uni['evaluated']} 币；跳过 {len(uni['skipped'])} 币"
            f"（清单见 `{interval}` 分报告）")
        add("")
        add("| 臂 | 笔数 | 每笔均值% | CI 下界% | CI 上界% | 正 fold | 随机分位 | q | 判定 |")
        add("|---|---|---|---|---|---|---|---|---|")
        for name in arm_names(tuple(sub["protocol"]["sizes"])):
            if name not in sub["summary"]:
                continue
            row = sub["summary"][name]
            ci = row["pooled"]
            gate = sub["verdicts"]["arms"][name]
            add(f"| {name} | {ci.get('trades', 0)} | {_signed(ci.get('mean_return_pct'))} | "
                f"{_signed(ci.get('ci_low_pct'))} | {_signed(ci.get('ci_high_pct'))} | "
                f"{row['positive_folds']}/{len(row['fold_means'])} | "
                f"{'—' if row['percentile_rank'] is None else format(row['percentile_rank'], '.3f')} | "
                f"{'—' if gate['q'] is None else format(gate['q'], '.4f')} | "
                f"**{gate['verdict']}** |")
        add("")
    add("## 4. 事后选币泄漏（核心诊断）")
    add("")
    add("`posthoc_N` 按当期已实现收益挑币，是作弊臂。它与 PIT 臂的差就是「用未来信息选币」"
        "凭空造出的 pp/笔——**这正是「缩小池看起来能赚钱」的来源**。")
    add("")
    add("| 周期 | N | posthoc 均值% | pit_liq 均值% | 泄漏 (pp/笔) |")
    add("|---|---|---|---|---|")
    for row in payload["leakage"]:
        add(f"| {row['interval']} | {row['n']} | {_signed(row['posthoc_mean_pct'])} | "
            f"{_signed(row['pit_liq_mean_pct'])} | {_signed(row['gap_pp'], '+.2f')} |")
    add("")
    add("## 5. 成本口径（H3：全局统一，未按臂覆盖）")
    add("")
    costs = payload["costs"]
    add(f"- `fee_rate={costs['fee_rate']}` / `slippage={costs['slippage']}` / "
        f"双边 {costs['round_trip']:.2%} · 每币独立账户 {costs['cash_per_symbol']} · "
        f"口径 **{costs['basis']}（名义本金）**，不是保证金口径，两者不可混比。")
    add(f"- 冻结参数：factor={payload['parameters'].get('supertrend_factor')} / "
        f"target_r={payload['parameters'].get('target_r')}（完整参数见各分报告 JSON）。")
    add("- 所有臂共用同一份逐币成交与同一套参数，**成员构成是唯一变量**。")
    add("")
    add("## 6. 局限与数据边界")
    add("")
    for item in payload["limitations"]:
        add(f"- {item}")
    add("")
    add("## 7. 复现")
    add("")
    add("```powershell")
    add(r".\.venv\Scripts\python.exe -m pytest backend/tests/test_pit_universe.py "
        r"backend/tests/test_pit_pool_stats.py backend/tests/test_pit_pool_arms.py "
        r"backend/tests/test_pit_pool_report.py backend/tests/test_pit_pool_combine.py -q")
    add(r".\.venv\Scripts\python.exe backend/scripts/supertrend_pit_pool.py --interval 1h "
        r"--days 365 --folds 5 --seeds 10 --sizes 10,20,30 --out docs/supertrend-pit-pool-1h")
    add(r".\.venv\Scripts\python.exe backend/scripts/supertrend_pit_pool.py --interval 15m "
        r"--days 180 --folds 5 --seeds 10 --sizes 10,20,30 --out docs/supertrend-pit-pool-15m")
    add(r".\.venv\Scripts\python.exe backend/scripts/pit_pool_combine.py "
        r"--out docs/supertrend-pit-pool")
    add(r".\.venv\Scripts\python.exe .workbuddy/skills/strategy-backtest-playbook/scripts/preflight.py "
        r"--prefix docs/supertrend-pit-pool")
    add("```")
    add("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P17 双臂合并报告")
    ap.add_argument("--inputs", nargs="*",
                    default=["docs/supertrend-pit-pool-1h.json", "docs/supertrend-pit-pool-15m.json"])
    ap.add_argument("--out", default="docs/supertrend-pit-pool")
    args = ap.parse_args(argv)
    payloads = {}
    for path in args.inputs:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        payloads[data["interval"]] = data
    payload = combine(payloads)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str),
                                        encoding="utf-8")
    out.with_suffix(".md").write_text(render_combined(payload), encoding="utf-8")
    print(f"课题判定 = {payload['verdict']}")
    for row in payload["leakage"]:
        print(f"  {row['interval']:4s} N={row['n']:2d} 泄漏 {row['gap_pp']:+.2f} pp/笔")
    print(f"[输出] {out.with_suffix('.md')} / {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())