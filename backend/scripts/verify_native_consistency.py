r"""native 1h 回落的一致性校验：证明「库里躺着的 400 天」能安全并入取数链路。

为什么必须先做这一步
----------------------
`_candle_query` 目前强制从 1m 聚合出所有周期，并刻意忽略已单独存储的
non-1m 行。库里 BTC/ETH/DOGE 各有 9622 根 native 1h（400.9 天），
但引擎只用得上 1m 覆盖的 91 天 —— 浪费 77% 历史。

问题在于：native 行是「历史遗留的独立下载」，与 1m 聚合结果**是否一致未知**。
若两者 OHLC 有偏差却直接混合，会在训练样本里注入假信号，比没数据更糟。

本脚本在两者重叠区间内逐根比对，产出可判定的结论：
  - 完全一致的桶占比（close/high/low/open/volume 相对偏差 < 1e-9）
  - 最大相对偏差
  - 不一致桶明细（前 20 根）

只有「不一致桶占比 = 0」才允许放开 native 回落。

只读运行，不写任何数据。

用法：
    .\.venv\Scripts\python.exe backend/scripts/verify_native_consistency.py
    .\.venv\Scripts\python.exe backend/scripts/verify_native_consistency.py --symbols BTCUSDT ETHUSDT
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.db import make_engine  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.models import Kline  # noqa: E402
from app.utils import interval_ms, now_ms  # noqa: E402

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT"]
REL_TOL = 1e-9


def compare_symbol(db: Session, symbol: str, interval: str) -> dict:
    """在一个币种上比对「1m 聚合」与「native 行」的重叠区间。"""
    service = MarketDataService(db)

    # 1) 聚合口径：现有引擎路径，只出「分钟齐全」的桶
    agg_q = service._candle_query(symbol, interval).subquery()
    agg = pd.DataFrame(
        [dict(r) for r in db.execute(select(agg_q).order_by(agg_q.c.open_time.asc())).mappings().all()]
    )
    if agg.empty:
        return {"symbol": symbol, "interval": interval, "status": "no_aggregated_rows"}

    # 2) native 口径：已单独存储的同周期行
    native_rows = db.execute(
        select(
            Kline.open_time, Kline.open, Kline.high, Kline.low, Kline.close,
            Kline.volume, Kline.quote_volume,
        )
        .where(Kline.symbol == symbol, Kline.interval == interval)
        .order_by(Kline.open_time.asc())
    ).mappings().all()
    native = pd.DataFrame([dict(r) for r in native_rows])
    if native.empty:
        return {"symbol": symbol, "interval": interval, "status": "no_native_rows",
                "aggregated_rows": len(agg)}

    # native 序列的**最后一根**必须排除：它是「下载那一刻还没走完的小时」的快照，
    # OHLCV 全是半截值（实测 BNB 最后一根 volume 1622，而该小时完整值是 13332）。
    # 这与当前时间无关——1m 后来补齐了，但 native 那根快照永远不会更新。
    # 取数链路同样会丢弃它，所以这里不参与一致性判定。
    unclosed = 1
    native = native.iloc[:-1]

    merged = agg.merge(native, on="open_time", suffixes=("_agg", "_nat"))
    if merged.empty:
        return {"symbol": symbol, "interval": interval, "status": "no_overlap",
                "aggregated_rows": len(agg), "native_rows": len(native)}

    # 逐字段相对偏差；分母为 0 时用绝对差
    report: dict = {
        "symbol": symbol,
        "interval": interval,
        "aggregated_rows": len(agg),
        "native_rows": len(native),
        "excluded_native_tail": unclosed,
        "overlap_rows": len(merged),
        "fields": {},
    }
    mismatched = pd.Series(False, index=merged.index)
    for field in ("open", "high", "low", "close", "volume", "quote_volume"):
        a, b = merged[f"{field}_agg"].astype(float), merged[f"{field}_nat"].astype(float)
        denom = a.abs().clip(lower=1e-12)
        rel = ((a - b).abs() / denom).fillna(0.0)
        bad = rel > REL_TOL
        report["fields"][field] = {
            "max_rel_diff": float(rel.max()),
            "mismatch_rows": int(bad.sum()),
        }
        mismatched |= bad

    report["mismatch_rows"] = int(mismatched.sum())
    report["mismatch_ratio"] = round(mismatched.sum() / len(merged), 6)
    report["consistent"] = report["mismatch_rows"] == 0
    if report["mismatch_rows"]:
        sample = merged.loc[mismatched, ["open_time"]].head(20)
        report["mismatch_samples"] = [
            str(datetime.fromtimestamp(int(t) / 1000, timezone.utc)) for t in sample["open_time"]
        ]
    return report


def to_markdown(report: dict) -> str:
    lines = ["# native 周期行一致性校验\n"]
    lines.append(
        f"- 生成时间：{report['generated_at']}；比对周期：{report['interval']}\n"
        f"- 判定口径：open/high/low/close/volume/quote_volume 相对偏差均 < {REL_TOL:g}\n"
        "- 已排除 native 序列的**最后一根**：那是「下载那一刻还没走完的小时」的快照，"
        "OHLCV 是半截值（实测 BNB 末根 volume 1622，而该小时完整值 13332）。"
        "取数链路同样丢弃它，故不参与判定\n"
    )
    lines.append(
        "\n| 币种 | 聚合行 | native 行 | 排除末根 | 重叠行 | 不一致行 | 不一致占比 | close 最大相对偏差 | 结论 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for r in report["symbols"]:
        if "overlap_rows" not in r:
            lines.append(
                f"| {r['symbol']} | {r.get('aggregated_rows', 0)} | {r.get('native_rows', 0)} | "
                f"{r.get('excluded_native_tail', 0)} | 0 | - | - | - | {r['status']} |\n"
            )
            continue
        close = r["fields"]["close"]["max_rel_diff"]
        verdict = "一致，可回落" if r["consistent"] else "不一致，禁止回落"
        lines.append(
            f"| {r['symbol']} | {r['aggregated_rows']} | {r['native_rows']} | "
            f"{r['excluded_native_tail']} | {r['overlap_rows']} | "
            f"{r['mismatch_rows']} | {r['mismatch_ratio']} | {close:.2e} | {verdict} |\n"
        )
    ok = [r for r in report["symbols"] if r.get("consistent")]
    lines.append(
        f"\n## 结论\n\n- {len(ok)}/{len(report['symbols'])} 个币种的重叠区间完全一致。\n"
    )
    lines.append(
        "- 只有不一致行数为 0 时，才允许放开 native 回落；"
        "否则必须先修复 native 数据或维持现状。\n"
        "- 本报告只做只读比对，不修改任何数据。\n"
    )
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="native 周期行一致性校验")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--output", default="docs/native-consistency.json")
    parser.add_argument("--markdown-output", default="docs/native-consistency.md")
    parser.add_argument("--report-only", action="store_true", help="用已有 JSON 重渲染 Markdown")
    args = parser.parse_args()

    if args.report_only:
        with open(args.output, encoding="utf-8") as handle:
            report = json.load(handle)
        with open(args.markdown_output, "w", encoding="utf-8") as handle:
            handle.write(to_markdown(report))
        print(f"re-rendered: {args.markdown_output}")
        return

    engine = make_engine(read_only=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "interval": args.interval,
        "symbols": [],
    }
    with Session(engine) as db:
        for symbol in args.symbols:
            print(f"[verify] {symbol} ...", flush=True)
            report["symbols"].append(compare_symbol(db, symbol, args.interval))
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with open(args.markdown_output, "w", encoding="utf-8") as handle:
        handle.write(to_markdown(report))
    print(f"written: {args.output} / {args.markdown_output}")


if __name__ == "__main__":
    main()
