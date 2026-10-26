r"""迭代9：模型持续训练报告 —— 训练轮次、AUC 趋势、晋升门禁结果。

支持两种模式：
- `--train`：先跑一轮持续训练（写库，耗时长可后台跑），再汇总
- `--dry-run`：只读已有记录汇总，不训练、不联网取宇宙

诚实结论优先：AUC 普遍≈0.5 即判定"未展现出可交易 edge"，明确不建议启用 ml_ensemble，
并给出继续观察所需轮数与大致天数。门禁不过绝不启用（ml_auto_enable 默认 False）。

用法（项目根）：
    PYTHONPATH=backend ./.venv/Scripts/python.exe backend/scripts/model_training_report.py --train --symbols-limit 10
    PYTHONPATH=backend ./.venv/Scripts/python.exe backend/scripts/model_training_report.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.db import make_engine  # noqa: E402
from app.ml import auto_trainer  # noqa: E402
from app.models import ModelRecord  # noqa: E402
from app.state import all_state  # noqa: E402


def _fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def existing_symbols(db, interval: str, limit: int) -> list[str]:
    """dry-run 用：从已有 ModelRecord 取币种，避免联网拉宇宙。"""
    rows = db.query(ModelRecord.symbol).filter(ModelRecord.interval == interval).distinct().all()
    return sorted({str(row[0]) for row in rows})[: max(int(limit), 1)]


def target_symbols(db, args) -> list[str]:
    if args.symbols:
        return [item.strip().upper() for item in args.symbols if item.strip()]
    if args.train:
        return auto_trainer._target_symbols(db, all_state(db), args.symbols_limit)
    return existing_symbols(db, args.interval, args.symbols_limit)


def trend_of(history: list[dict]) -> dict:
    """按 created_at 升序的 AUC 序列给出趋势（<2 轮无法判断）。"""
    aucs = [item["auc"] for item in history if item.get("auc") is not None]
    if not aucs:
        return {"latest": None, "delta": None, "label": "无记录", "rounds": len(history)}
    if len(aucs) < 2:
        return {"latest": aucs[-1], "delta": None, "label": "轮数不足", "rounds": len(aucs)}
    delta = round(aucs[-1] - aucs[0], 4)
    label = "改善" if delta > 0.005 else ("恶化" if delta < -0.005 else "持平")
    return {"latest": aucs[-1], "delta": delta, "label": label, "rounds": len(aucs)}


def collect(db, symbols, interval):
    """返回 (逐轮明细, {symbol: history}, 门禁结论列表)。"""
    rows, histories, verdicts = [], {}, []
    for symbol in symbols:
        history = auto_trainer.list_training_history(db, symbol, interval, limit=50)
        histories[symbol] = history
        for item in history:
            rows.append({"symbol": symbol, **item})
        verdicts.append(auto_trainer.evaluate_promotion(db, symbol, interval))
    return rows, histories, verdicts


def build_report(rows, histories, verdicts, state, args, cycle, as_of_iso, ml_enabled):
    min_auc = float(state.get("ml_min_auc") or 0.55)
    required = int(state.get("ml_promote_rounds") or 2)
    interval_hours = float(state.get("auto_train_interval_hours") or 24)
    lines = [
        f"# 模型持续训练报告 — as-of {as_of_iso}", "",
        "## 0. 参数与口径", "",
        f"- 模式：{'--train（实际训练）' if args.train else '--dry-run（只读汇总）'}；"
        f"interval={args.interval}；训练窗口 {int(state.get('auto_train_days') or 40)} 天；"
        f"horizon={int(state.get('train_horizon') or 5)}",
        f"- 持续训练 auto_train_enabled={state.get('auto_train_enabled')}；"
        f"最小重训间隔 {interval_hours:g}h；本轮目标币种数 {len(histories)}"
        f"（命令行覆盖 {args.symbols_limit}，state 默认 {int(state.get('auto_train_top_n') or 10)}）",
        f"- 晋升门禁：连续 {required} 轮，每轮需 status=ready 且 AUC>={min_auc} "
        f"且 test_rows>={auto_trainer.MIN_TEST_ROWS}",
        f"- ml_auto_enable={state.get('ml_auto_enable')}；"
        f"ml_ensemble 当前 enabled={ml_enabled}", ""]
    if cycle is not None:
        lines += [
            "## 1. 本轮训练执行结果", "",
            f"- 训练 {len(cycle.get('trained') or [])} 个、跳过（模型新鲜）"
            f"{len(cycle.get('skipped') or [])} 个、失败 {len(cycle.get('failed') or [])} 个", ""]
        for item in cycle.get("failed") or []:
            lines.append(f"- 失败：{item['symbol']} — {item['error'][:160]}")
        if cycle.get("failed"):
            lines.append("")
    lines += ["## 2. 训练轮次明细", "",
              "| 币种 | 模型 | 时间(UTC) | AUC | accuracy | test_rows | 样本 | 状态 |",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['symbol']} | #{row['id']} | {str(row['created_at'])[:19]} "
                     f"| {_fmt(row['auc'])} | {_fmt(row['accuracy'])} "
                     f"| {row['test_rows']} | {row['rows']} | {row['status']} |")
    lines += ["", "## 3. AUC 趋势（每币种，旧→新）", "",
              "| 币种 | 轮数 | 各轮 AUC | 最新 | 变化 | 趋势 |", "|---|---:|---|---:|---:|---|"]
    for symbol, history in histories.items():
        info = trend_of(history)
        aucs = "、".join(_fmt(item["auc"], 3) for item in history) or "-"
        lines.append(f"| {symbol} | {info['rounds']} | {aucs} | {_fmt(info['latest'], 3)} "
                     f"| {_fmt(info['delta'], 3)} | {info['label']} |")
    lines += ["", "## 4. 晋升门禁结果", "",
              "| 币种 | 达标 | 已检查/需要轮数 | 各轮 AUC | 原因 |", "|---|---|---:|---|---|"]
    promotable = [v for v in verdicts if v["promotable"]]
    for verdict in verdicts:
        aucs = "、".join(_fmt(a, 3) for a in verdict["aucs"]) or "-"
        lines.append(f"| {verdict['symbol']} | {'是' if verdict['promotable'] else '否'} "
                     f"| {verdict['rounds_checked']}/{verdict['rounds_required']} "
                     f"| {aucs} | {verdict['reason']} |")
    latest = [trend_of(h)["latest"] for h in histories.values()]
    latest = [value for value in latest if value is not None]
    mean_auc = round(statistics.mean(latest), 4) if latest else None
    max_checked = max((v["rounds_checked"] for v in verdicts), default=0)
    need_rounds = max(0, required - max_checked)
    need_days = round(need_rounds * interval_hours / 24, 1)
    lines += ["", "## 5. 结论", ""]
    if promotable and bool(state.get("ml_auto_enable")):
        lines.append(f"门禁通过 {len(promotable)}/{len(verdicts)} 个币种，且 ml_auto_enable=True → "
                     "可启用 ml_ensemble（仍建议小仓位观察）。")
    elif promotable:
        lines.append(f"门禁通过 {len(promotable)}/{len(verdicts)} 个币种，"
                     "但 ml_auto_enable=False → **仅记录候选，不启用 ml_ensemble**。")
    else:
        lines.append(f"**门禁未通过（0/{len(verdicts)} 达标）→ ml_ensemble 保持 disabled。**")
    if mean_auc is not None and mean_auc < 0.55:
        lines.append(f"- 最新一轮 AUC 均值 {_fmt(mean_auc)} 接近 0.5（随机水平）："
                     "**模型未展现出可交易 edge，不建议启用 ml_ensemble**。")
    elif mean_auc is not None:
        lines.append(f"- 最新一轮 AUC 均值 {_fmt(mean_auc)}，略高于随机，"
                     "但仍需连续多轮稳定达标才谈得上启用。")
    if need_rounds:
        lines.append(f"- 继续训练建议：按当前 {interval_hours:g}h 重训间隔，再累积 {need_rounds} 轮"
                     f"（约 {need_days} 天）后重跑本报告再评估；"
                     "期间不要因单轮 AUC 波动就调低阈值。")
    lines.append("- 口径提醒：AUC 为样本外（purged split 后的测试段）指标，"
                "不等同于交易盈利；即便达标也需先过组合层回测再谈实盘。")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="先跑一轮持续训练再汇总")
    parser.add_argument("--dry-run", action="store_true", help="只汇总已有记录（默认）")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--symbols-limit", type=int, default=10)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "model-training-report.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "docs" / "model-training-report.json")
    args = parser.parse_args()

    engine = make_engine(read_only=True)
    cycle = None
    with Session(engine) as db:
        state = all_state(db)
        if args.train:
            cycle = auto_trainer.run_auto_train_cycle(
                db, top_n=args.symbols_limit,
                symbols=([s.strip().upper() for s in args.symbols] if args.symbols else None),
            )
            print(json.dumps({"trained": len(cycle.get("trained") or []),
                              "skipped": len(cycle.get("skipped") or []),
                              "failed": len(cycle.get("failed") or []),
                              "failed_symbols": [f["symbol"] for f in cycle.get("failed") or []]},
                             ensure_ascii=False), flush=True)
        try:
            symbols = target_symbols(db, args)
        except Exception as exc:  # noqa: BLE001 - 宇宙不可用则回退已有记录
            symbols = existing_symbols(db, args.interval, args.symbols_limit)
            print(f"宇宙解析失败，回退已有记录: {str(exc)[:160]}", flush=True)
        rows, histories, verdicts = collect(db, symbols, args.interval)
        ml_enabled = auto_trainer.ml_ensemble_enabled(db)
        as_of = datetime.now(timezone.utc)
        report = build_report(rows, histories, verdicts, state, args, cycle,
                              as_of.isoformat(), ml_enabled)
        args.output.write_text(report, encoding="utf-8")
        args.json_output.write_text(json.dumps({
            "as_of": as_of.isoformat(), "interval": args.interval, "symbols": symbols,
            "trained": (cycle or {}).get("trained"), "skipped": (cycle or {}).get("skipped"),
            "failed": (cycle or {}).get("failed"), "verdicts": verdicts,
            "histories": histories, "ml_ensemble_enabled": ml_enabled,
        }, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    engine.dispose()
    print(f"报告已生成: {args.output} | 门禁达标 "
          f"{sum(1 for v in verdicts if v['promotable'])}/{len(verdicts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
