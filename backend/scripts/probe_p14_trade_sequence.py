#!/usr/bin/env python3
"""H-B / P14 逐笔成交顺序 Gate 1 探针（只读、可行性筛查，非 edge 判定）。

预注册主判据：seq_flip_dev × H=24 的横截面 rank IC。核心对照是「整行重排」：
保持每笔 price/quote/direction 配对不变、只打乱先后，K 线级 taker_imb 严格
不变，而顺序特征应向 0 收敛。若顺序特征与 taker_imb 共线或重排不归零，则
判定为已证伪路线的换皮。复用 P12 统计量具，仅显式放宽横截面宽度阈值。

不入库、不落盘原始逐笔、不改 spec/绑定/参数、不下单。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
import probe_p12_common as p12  # noqa: E402
from app.db import make_engine  # noqa: E402
from probe_p14_common import (  # noqa: E402
    BASE, HOUR_MS, PacedClient, fetch_one, load_frames, save_frames,
    sequence_diagnostic, to_frames,
)
from probe_p14_robust import decision_summary, robustness  # noqa: E402

SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "UNIUSDT", "APTUSDT", "NEARUSDT",
    "ARBUSDT", "SUIUSDT", "DOTUSDT", "LTCUSDT", "TRXUSDT", "ETCUSDT",
    "FILUSDT", "ATOMUSDT", "INJUSDT", "OPUSDT", "TIAUSDT", "SEIUSDT",
)
HORIZONS = (8, 24, 48)
PRIMARY_FACTOR, PRIMARY_HORIZON = "seq_flip_dev", 24
COSTS = {"fee_rate": 0.0004, "slippage": 0.0002, "round_trip": 0.0012,
         "basis": "notional", "applied_to_ic": False}
MIN_EFFECTIVE_BLOCKS = 20


def day_ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def load_candles(start: int, end: int) -> pd.DataFrame:
    engine = make_engine(read_only=True)
    with engine.connect() as conn:
        return pd.read_sql_query(text(
            "SELECT symbol, open_time, close FROM klines WHERE interval='1h' "
            "AND open_time BETWEEN :a AND :b"), conn,
            params={"a": start - 2 * HOUR_MS,
                    "b": end + (max(HORIZONS) + 2) * HOUR_MS})


def download(symbols, days, workers, seed, pace):
    client = PacedClient(pace)
    tasks = [(s, d) for s in symbols for d in days]
    records = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for s, d in tasks:
                h = int(hashlib.md5(f"{s}{d}{seed}".encode()).hexdigest()[:8], 16)
                futures.append(pool.submit(fetch_one, client, s, d, seed + h))
            for fut in futures:
                records.append(fut.result())
    finally:
        client.close()
    return records


def run(args) -> dict:
    end_day = date.fromisoformat(args.end_day)
    day_list = [end_day - timedelta(days=i) for i in range(args.days)][::-1]
    symbols = args.symbols.split(",") if args.symbols else list(SYMBOLS)
    start, end = day_ms(day_list[0]), day_ms(day_list[-1]) + 23 * HOUR_MS
    grid = np.arange(start, end + HOUR_MS, HOUR_MS)
    p12.MIN_CROSS_SECTION = args.min_cross_section  # 量具逻辑不变，仅放宽宽度门槛

    cache = Path(args.cache)
    if cache.exists() and not args.refresh_cache:
        frames, records = load_frames(cache)
        from_cache = True
    else:
        records = download(symbols, day_list, args.workers, args.seed, args.pace)
        frames = to_frames(records, grid, symbols)
        save_frames(frames, cache,
                    [{k: v for k, v in r.items() if k != "rows"} for r in records])
        from_cache = False
    candles = load_candles(start, end)
    labels = {h: p12.build_labels(candles, grid, h, list(symbols)) for h in HORIZONS}
    _, excess, valid, _ = labels[PRIMARY_HORIZON]
    factor = frames[PRIMARY_FACTOR]

    ics, counts = p12.ic_series(factor, excess, valid)
    primary = p12.summarize(ics, "future_24h_return_minus_cross_sectional_median")
    primary.update({
        "factor": PRIMARY_FACTOR, "horizon": PRIMARY_HORIZON,
        "ci": p12.block_bootstrap_ci(ics, seed=args.seed),
        "per_symbol_ic": p12.per_symbol_ic(factor, excess),
        "permutation_null": (
            p12.permutation_null(factor, excess, valid, args.permutations, args.seed)
            if np.isfinite(ics).any() else
            {"empirical_p_two_sided": 1.0, "null_mean": None,
             "note": "无有效横截面期，未评估"}),
        "mean_valid_symbols_per_period": round(float(np.nanmean(counts)), 2),
    })
    pos_ics, _ = p12.ic_series(
        factor, factor.copy(),
        pd.DataFrame(True, index=grid, columns=symbols))
    primary["positive_control"] = {
        "mean_ic": round(float(np.nanmean(pos_ics)), 5),
        "note": "以主因子本身作伪标签，rank IC 应≈1，量具自校准"}

    exploratory = []
    for feat in ("seq_flip_dev", "big_run_dev", "impact_rev"):
        for h in HORIZONS:
            if feat == PRIMARY_FACTOR and h == PRIMARY_HORIZON:
                continue
            _, ex, ok, _ = labels[h]
            s, _ = p12.ic_series(frames[feat], ex, ok)
            row = p12.summarize(s, f"H{h}_excess")
            row.update({"factor": feat, "horizon": h,
                        "ci": p12.block_bootstrap_ci(s, seed=args.seed)})
            exploratory.append(row)
    pvals = []
    for row in exploratory:
        evaluable = (row["periods"] > 0 and row["mean_ic"] is not None
                     and np.isfinite(row["mean_ic"]))
        # 标签全空时 permutation_p 会因 NaN 比较返回假的 p=0，必须显式排除
        pvals.append(p12.permutation_p(
            frames[row["factor"]], labels[row["horizon"]][1],
            labels[row["horizon"]][2], args.permutations, args.seed)
            if evaluable else 1.0)
        row["not_evaluable"] = not evaluable
    qvals = p12.bh_fdr(pvals)
    for row, p, q in zip(exploratory, pvals, qvals):
        row.update({"empirical_p": p, "fdr_q": q})

    control = []
    for h in HORIZONS:
        _, ex, ok, _ = labels[h]
        s, _ = p12.ic_series(frames["taker_imb"], ex, ok)
        control.append({"factor": "taker_imb", "horizon": h,
                        "mean_ic": p12.summarize(s, f"H{h}")["mean_ic"],
                        "note": "已证伪的 K 线级主动成交汇总，仅作换皮参照"})
    diag = sequence_diagnostic(frames, args.min_cross_section)
    null_mean = primary["permutation_null"].get("null_mean")
    calib_ok = (primary["positive_control"]["mean_ic"] is not None
                and primary["positive_control"]["mean_ic"] > 0.99
                and null_mean is not None and abs(null_mean) < 0.02
                and diag["taker_imb"]["mean_abs_diff"] < 1e-6)
    gate = {
        "abs_mean_ic_gt_0.02": abs(primary["mean_ic"] or 0) > 0.02,
        "ci_excludes_zero": (primary["ci"]["ci_low"] is not None
                             and (primary["ci"]["ci_low"] > 0
                                  or primary["ci"]["ci_high"] < 0)),
        "same_sign_periods_ge_0.60": (primary["same_sign_share"] or 0) >= 0.60,
        "same_sign_symbols_ge_0.60":
            (primary["per_symbol_ic"]["same_sign_share"] or 0) >= 0.60,
        "permutation_p_lt_0.05":
            primary["permutation_null"]["empirical_p_two_sided"] < 0.05,
        "calibration_ok": bool(calib_ok),
    }
    finite_periods = int(primary["periods"])
    blocks = int(np.ceil(finite_periods / PRIMARY_HORIZON))
    verdict = "insufficient" if blocks < MIN_EFFECTIVE_BLOCKS else "inconclusive"
    status = {}
    for r in records:
        status[r["status"]] = status.get(r["status"], 0) + 1
    downloads = {
        "archive_base": BASE, "requested_files": len(records),
        "status_counts": status, "total_bytes": sum(r.get("bytes", 0) for r in records),
        "total_megabytes": round(sum(r.get("bytes", 0) for r in records) / 1048576, 2),
        "source_trade_rows": sum(r.get("parse", {}).get("source_rows", 0) for r in records),
        "skipped_thin_hours": sum(r.get("parse", {}).get("skipped_thin_hours", 0)
                                  for r in records),
        "kept_hour_rows": sum(r.get("parse", {}).get("kept_hours", 0) for r in records),
        "skip_list": [{"symbol": r["symbol"], "day": r["day"],
                       "reason": r.get("error", r["status"])}
                      for r in records if r["status"] != "ok"],
        "checksum": "未请求 .CHECKSUM，依赖 zipfile 的 CRC 校验",
        "from_cache": from_cache,
    }
    robust = robustness(frames, labels, grid)
    decision = decision_summary(robust, all(gate.values()))
    return {
        "topic": "H-B / P14 逐笔成交顺序 Gate 1 可行性探针",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "backend/scripts/probe_p14_trade_sequence.py",
        "stage": "Gate 1（IC 探针，非回测、非 edge 判定）",
        "hypothesis_card": "research/hypotheses/H-B-trade-sequence.md",
        "interval": "1h", "primary": primary, "gate": gate,
        "gate_passed": all(gate.values()), "exploratory": exploratory,
        "exploratory_multiple_testing": {
            "method": "BH-FDR on 200-permutation empirical p", "n_tests": len(qvals),
            "q_below_0.10": sum(1 for q in qvals if q < 0.10),
            "note": "置换零分布假设各期独立，零分布偏窄；以 block24 CI 为准"},
        "control_feature_ic": control, "sequence_specificity_diagnostic": diag,
        "robustness_diagnostics": robust, "decision": decision,
        "download": downloads,
        "costs": COSTS, "trades": 0,
        "trades_note": "本轮无交易；IC 为毛价格关联，未扣 0.12% 双边成本",
        "window": {"start_utc": iso(start), "end_utc": iso(end),
                   "days": args.days, "hourly_periods": int(len(grid)),
                   "effective_nonoverlapping_blocks_at_H24": blocks,
                   "min_effective_blocks_required": MIN_EFFECTIVE_BLOCKS},
        "universe": {"size": len(symbols), "symbols": symbols,
                     "selection": "事前 HEAD 验证归档存在的 USD-M 主流/中流动性币",
                     "min_cross_section": args.min_cross_section,
                     "p12_min_cross_section": 30,
                     "note": "横截面宽度低于 P12，仅支持可行性筛查"},
        "ci": primary["ci"], "verdict": verdict,
        "verdict_scope": ("Gate 1 探针不是净收益判定；样本不足以支撑 edge 结论，"
                          "数值照给但按 H4 判 insufficient"),
        "default_enabled": False,
        "limitations": [
            "14 天窗口且 H=24 标签高度重叠，有效独立时间块不足 20",
            "横截面仅 24 币，低于 P12 的 30；已显式放宽 MIN_CROSS_SECTION",
            "aggTrades 是毫秒级聚合逐笔，不含真实逐笔委托/撤单与盘口",
            "未请求 .CHECKSUM，仅依赖 zip CRC；未入库、未落盘原始逐笔",
            "IC 未扣 0.12% 双边成本，不得解读为净收益 edge",
            "impact_rev 与桶内价格路径相关，未纳入顺序重排对照",
            "固定币池与单一市场，未覆盖多所现货腿与盘口信息",
            "样本期单一普跌状态（24 币中位 −9.86%、仅 8.3% 上涨），无样本外切分",
            "H=48 标签重叠 48 期而 block bootstrap 块长仅 24，长 H 的 CI 可能过窄",
        ],
        "seed": args.seed, "permutations": args.permutations,
        "workers": args.workers, "request_pace_seconds": args.pace,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", help="逗号分隔；缺省为预注册 24 币池")
    ap.add_argument("--end-day", default="2026-11-09")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pace", type=float, default=0.25)
    ap.add_argument("--min-cross-section", type=int, default=18)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--out", default="docs/probe-p14-trade-sequence.json")
    ap.add_argument("--cache", default="output/probe_p14_frames.pkl",
                    help="小时级标量帧缓存，避免重复下载约 1GB 归档")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="忽略缓存重新下载归档")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"输出已存在：{out}；如需覆盖加 --force")
    result = run(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "verdict": result["verdict"],
                      "gate_passed": result["gate_passed"],
                      "primary_mean_ic": result["primary"]["mean_ic"],
                      "download": result["download"]["status_counts"],
                      "total_megabytes": result["download"]["total_megabytes"]},
                     ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
