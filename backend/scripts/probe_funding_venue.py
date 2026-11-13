r"""P11 / H-F · 资金费率多所路径 Gate 1a（只读可行性探针，只报毛超额）。

预登记假设卡：`research/hypotheses/H-F-funding-multi-venue.md`（D1–D3 事前写死）。

问题
----
单所口径已终判 FAIL：现货可对冲宇宙 348/507，K=10 净年化 −2.4%，maker 改造天花板
+1.43%。失败归因是**对冲容量**不是信号——高费率集中在币安只上永续的 memecoin。
本脚本回答：**把那些币加回来，奖金够不够大到值得去接第二家交易所**。

做法
----
同一份币安 funding 历史上切两个宇宙（今日现货 exchangeInfo 判定可对冲性），
每 3 天重平衡、过去 7 天均费率 >0 降序取 TopK、前瞻 3 天**实际结算事件**费率和：
- 全宇宙 TopK vs 仅可对冲 TopK 的毛年化差 = D3 增量
- 全宇宙 TopK 中不可对冲币贡献的费率占比 = D1 奖金
- 不可对冲子集内部 trailing→forward 秩 IC = D2 信号强度

**不报净收益**：他所费率/深度未知，报净就是编数。不下单、不接 API key、无绑定。

    .\.venv\Scripts\python.exe backend/scripts/probe_funding_venue.py --out docs/probe-funding-venue
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import httpx  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.db import make_engine  # noqa: E402
from app.market.binance import PUBLIC_HOSTS  # noqa: E402
from app.research.funding_venue import (  # noqa: E402
    DAY_MS, hedgeable_split, rebalance_times,
)
from app.research.funding_venue_render import write_reports  # noqa: E402
from app.research.funding_venue_report import assemble  # noqa: E402
from app.research.funding_venue_stats import (  # noqa: E402
    HOLD_DAYS, KS, LOOKBACK_DAYS, bonus_share, cross_section_ic,
    gross_period_return, period_snapshot, random_baseline,
)

SPOT_PATH = "/api/v3/exchangeInfo"
HISTORICAL = {"hedgeable": 348, "total": 507, "as_of": "2026-10-27"}


def _commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"],
                                        cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha} (dirty worktree)" if dirty else sha


def load_book() -> tuple[dict, dict]:
    """只读连接一次性取全表，按币切成 (times, rates) 有序数组。"""
    engine = make_engine(read_only=True)
    with engine.connect() as conn:
        frame = pd.read_sql_query(text(
            "SELECT symbol, funding_time, funding_rate FROM funding_rates "
            "ORDER BY symbol, funding_time"), conn)
        types = sorted(str(r[0]) for r in conn.execute(text(
            "SELECT DISTINCT rate_type FROM funding_rates")))
    book = {str(sym): (g["funding_time"].to_numpy("int64"),
                       g["funding_rate"].to_numpy("float64"))
            for sym, g in frame.groupby("symbol", sort=True)}
    start, end = int(frame["funding_time"].min()), int(frame["funding_time"].max())
    counts = frame.groupby("symbol").size()
    data = {
        "table": "funding_rates", "rows": int(len(frame)), "symbols": len(book),
        "start_ms": start, "end_ms": end,
        "days": round((end - start) / DAY_MS, 1), "rate_types": types,
        "events_per_symbol": {"min": int(counts.min()),
                              "median": int(counts.median()),
                              "max": int(counts.max())},
        "read_only": True,
    }
    return book, data


def fetch_spot(cache: Path, refresh: bool) -> tuple[set[str], dict]:
    """公开现货 exchangeInfo（无 key）；落缓存供复现，失败即停，不编数据。"""
    if cache.exists() and not refresh:
        stored = json.loads(cache.read_text(encoding="utf-8"))
        meta = dict(stored["meta"], from_cache=True)
        return set(stored["bases"]), meta
    last = None
    for host in PUBLIC_HOSTS:
        try:
            with httpx.Client(timeout=40.0, trust_env=False) as client:
                resp = client.get(host + SPOT_PATH)
            resp.raise_for_status()
            symbols = resp.json()["symbols"]
            used = host + SPOT_PATH
            break
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            last = exc
    else:
        raise SystemExit(f"现货 exchangeInfo 全部公开端点失败，Gate 1a 无法执行：{last}")
    trading = [s for s in symbols
               if s.get("status") == "TRADING" and s.get("isSpotTradingAllowed")]
    bases = sorted({str(s["baseAsset"]).upper() for s in trading})
    meta = {"source": used,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "symbols_total": len(symbols), "trading": len(trading),
            "distinct_trading_bases": len(bases), "from_cache": False,
            "cache_path": str(cache)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"meta": meta, "bases": bases}, ensure_ascii=False),
                     encoding="utf-8")
    return set(bases), meta


def measure(book: dict, hedge: list[str], nonhedge: list[str],
            times: list[int]) -> dict:
    symbols = sorted(book)
    missing = set(nonhedge)
    per_k = {str(k): {"full_returns": [], "hedgeable_returns": [], "bonus": [],
                      "ic_non_hedgeable": [], "ic_full": [], "ic_hedgeable": [],
                      "cross_counts_non_hedgeable": [], "cross_counts_full": [],
                      "cross_counts_hedgeable": [], "missing_forward": 0,
                      "n_symbol_periods": 0, "random_baseline": None}
             for k in KS}
    eligible_periods, forward_periods = [], []
    for t in times:
        snap = period_snapshot(book, t)
        trailing, forward = snap["trailing"], snap["forward"]
        eligible_periods.append(sorted(s for s, v in trailing.items()
                                       if v is not None and v > 0))
        forward_periods.append(forward)
        ic_nh, n_nh = cross_section_ic(trailing, forward, nonhedge)
        ic_full, n_full = cross_section_ic(trailing, forward, symbols)
        ic_hedge, n_hedge = cross_section_ic(trailing, forward, hedge)
        for k in KS:
            bucket = per_k[str(k)]
            full = gross_period_return(trailing, forward, symbols, k)
            hedged = gross_period_return(trailing, forward, hedge, k)
            bucket["full_returns"].append(full["return"])
            bucket["hedgeable_returns"].append(hedged["return"])
            bucket["missing_forward"] += len(full["missing_forward"])
            bucket["n_symbol_periods"] += len(full["selected"])
            bucket["bonus"].append(bonus_share(full["selected"], forward, missing))
            bucket["ic_non_hedgeable"].append(ic_nh)
            bucket["cross_counts_non_hedgeable"].append(n_nh)
            bucket["ic_full"].append(ic_full)
            bucket["cross_counts_full"].append(n_full)
            bucket["ic_hedgeable"].append(ic_hedge)
            bucket["cross_counts_hedgeable"].append(n_hedge)
    for k in KS:
        per_k[str(k)]["random_baseline"] = random_baseline(
            eligible_periods, forward_periods, k)
    return per_k


def run(args) -> dict:
    started = time.perf_counter()
    book, data = load_book()
    bases, spot = fetch_spot(Path(args.spot_cache), args.refresh_spot)
    split = hedgeable_split(book.keys(), bases)
    times = rebalance_times(data["start_ms"], data["end_ms"],
                            lookback_days=LOOKBACK_DAYS, hold_days=HOLD_DAYS)
    if not times:
        raise SystemExit("样本跨度不足以容纳一个完整回看+前瞻窗，Gate 1a 无法执行")
    per_k = measure(book, split["hedgeable"], split["non_hedgeable"], times)
    payload = assemble({
        "data": data, "spot_snapshot": spot,
        "universe": {"hedgeable": split["hedgeable"],
                     "non_hedgeable": split["non_hedgeable"],
                     "historical_reference": HISTORICAL},
        "periods": {"count": len(times), "first_ms": times[0], "last_ms": times[-1]},
        "per_k": per_k, "code_commit": _commit(),
    })
    md_path, json_path = write_reports(payload, args.out)
    elapsed = round(time.perf_counter() - started, 1)
    print(f"funding {data['rows']:,} 行 / {data['symbols']} 币 / {data['days']} 天 / "
          f"{len(times)} 期 · 可对冲 {len(split['hedgeable'])} / "
          f"不可对冲 {len(split['non_hedgeable'])} · {elapsed}s")
    for key in ("10", "20"):
        arm, gate = payload["arms"][key], payload["gates"][key]
        print(f"K={key}: D1 奖金占比 {arm['bonus']['pooled_share']} · "
              f"D2 IC {arm['ic_non_hedgeable']['mean']} "
              f"(正期 {arm['ic_non_hedgeable']['positive_share']}) · "
              f"D3 增量 {arm['d3_increment_pp']}pp · "
              f"全宇宙毛年化 {arm['full']['annual_gross_pct']}% / "
              f"仅可对冲 {arm['hedgeable_only']['annual_gross_pct']}% · {gate}")
    print(f"判定 {payload['verdict']}（{payload['verdict_scope']}）")
    print(f"报告 {md_path} / {json_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="H-F Gate 1a 资金费率多所可行性（只读）")
    parser.add_argument("--out", default="docs/probe-funding-venue",
                        help="成对报告前缀（生成 .md 与 .json）")
    parser.add_argument("--spot-cache", default="docs/probe-funding-venue-spot.json",
                        help="现货 exchangeInfo 快照缓存路径")
    parser.add_argument("--refresh-spot", action="store_true",
                        help="忽略缓存重新拉取公开现货端点")
    run(parser.parse_args())


if __name__ == "__main__":
    main()