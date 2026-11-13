r"""P16 固定规模 PIT 币池可构造性 / 池规模敏感性（只读，非 edge 判定）。

只用 klines(1h) 的 open_time 与 quote_volume 做样本选择，不读 close／不算收益，
不重复 docs/binding-combo-optimization.md 的 8.1pp 收益对照，不触碰 H2。

    .\.venv\Scripts\python.exe backend/scripts/probe_p16_fixed_pool.py --out docs/probe-p16-fixed-pool.json
"""
from __future__ import annotations

import argparse, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db import make_engine  # noqa: E402
from app.strategy.pool_maintenance import EXCLUDED, WINDOW_HOURS  # noqa: E402

HOUR_MS, DAY_MS = 3_600_000, 86_400_000
LOOKBACK_HOURS = int(WINDOW_HOURS)          # 168，与线上池维护一致
MIN_BARS, STALE_HOURS, FORWARD_HOURS = 160, 2, 24  # 与 pit_pool_check.rebuild 一致
POOL_SIZES = (10, 20, 30, 50, 100)
MIN_CROSS_SECTION, STEP_DAYS = 30, 7        # P12 横截面探针门槛
LEVERAGED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
GATE_THRESHOLDS = {"C1_constructible_share": 0.90, "C2_jaccard_median": 0.60,
                   "C3_width_ge_30_share": 0.90}


def iso(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat() if ms else None


def symbol_allowed(symbol: str) -> bool:
    """沿用线上池规则：USDT 后缀、排除黑名单与杠杆代币。"""
    s = str(symbol)
    return (s.endswith("USDT") and s not in EXCLUDED
            and not s.endswith(LEVERAGED_SUFFIXES))


def eligible_ranking(window: np.ndarray, symbols: list[str], t_ms: int,
                     window_start_ms: int, min_bars: int = MIN_BARS,
                     stale_ms: int = STALE_HOURS * HOUR_MS) -> list[tuple[str, float]]:
    """按事前窗口成交额降序给出合格币；零成交额单根仍计有效 bar。"""
    finite = np.isfinite(window)
    bars = finite.sum(axis=0)
    vol = np.where(finite, window, 0.0).sum(axis=0)
    rows = np.arange(window.shape[0], dtype=np.int64)[:, None]
    last_row = np.where(finite, rows, -1).max(axis=0)
    last_ts = window_start_ms + np.maximum(last_row, 0) * HOUR_MS
    last_ok = (last_row >= 0) & (last_ts >= t_ms - stale_ms)
    ok = (bars >= min_bars) & (vol > 0) & last_ok
    cand = [(float(vol[j]), symbols[j]) for j in range(len(symbols)) if ok[j]]
    cand.sort(key=lambda item: (-item[0], item[1]))
    return [(s, v) for v, s in cand]


def forward_width(members: list[str], fwd: np.ndarray,
                  index: dict[str, int]) -> int:
    """PIT TopN 成员中在 [T,T+24h) 具备全部 24 根 1h bar 的币数。"""
    if fwd.shape[0] == 0:
        return 0
    complete = np.isfinite(fwd).sum(axis=0) >= fwd.shape[0]
    return int(sum(1 for s in members if complete[index[s]]))


def jaccard(a: list[str], b: list[str]) -> float:
    x, y = set(a), set(b)
    union = x | y
    return round(len(x & y) / len(union), 6) if union else 1.0


def rank_percentile(rank: int | None, eligible_count: int) -> float | None:
    """1.0 = 当期最活跃；None = 当期不在合格集合。"""
    if rank is None or eligible_count <= 1 or rank >= eligible_count:
        return None
    return round(1.0 - rank / (eligible_count - 1), 4)


def build_matrix(conn) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    raw = pd.read_sql_query(text("select symbol, open_time, quote_volume from klines "
                                 "where interval='1h'"), conn)
    raw = raw[raw["symbol"].map(symbol_allowed)].copy()
    raw["quote_volume"] = pd.to_numeric(raw["quote_volume"], errors="coerce")
    pivot = raw.pivot_table(index="open_time", columns="symbol",
                            values="quote_volume", aggfunc="last").sort_index()
    grid = np.arange(int(pivot.index.min()), int(pivot.index.max()) + HOUR_MS,
                     HOUR_MS, dtype=np.int64)
    mat = pivot.reindex(grid).to_numpy(float)
    symbols = [str(s) for s in pivot.columns]
    dq = {"source_rows": int(len(raw)), "symbols_after_filter": len(symbols),
          "hourly_grid_points": int(len(grid)),
          "quote_volume_null_rows": int(raw["quote_volume"].isna().sum()),
          "quote_volume_zero_rows": int((raw["quote_volume"] == 0).sum()),
          "window_start_ms": int(grid[0]), "window_end_ms": int(grid[-1])}
    return mat, grid, symbols, dq


def make_periods(grid: np.ndarray, step_days: int = STEP_DAYS) -> list[int]:
    g0, gend = int(grid[0]), int(grid[-1])
    first = ((g0 + LOOKBACK_HOURS * HOUR_MS) // DAY_MS + 1) * DAY_MS
    out, t = [], first
    while t + FORWARD_HOURS * HOUR_MS <= gend + HOUR_MS:
        out.append(int(t))
        t += step_days * DAY_MS
    return out


def collect(mat: np.ndarray, grid: np.ndarray, symbols: list[str],
            periods: list[int], sizes=POOL_SIZES) -> list[dict]:
    g0, h = int(grid[0]), HOUR_MS
    index = {s: j for j, s in enumerate(symbols)}
    look = LOOKBACK_HOURS * h
    records = []
    for t in periods:
        i0, i1 = (t - look - g0) // h, (t - g0) // h
        ranking = eligible_ranking(mat[i0:i1], symbols, t, t - look)
        ranks = {s: i for i, (s, _) in enumerate(ranking)}
        fwd = mat[i1:i1 + FORWARD_HOURS]
        rec = {"t": t, "eligible_count": len(ranking),
               "members": {}, "width": {}, "ranks": ranks}
        for n in sizes:
            members = [s for s, _ in ranking[:n]]
            rec["members"][n] = members
            rec["width"][n] = forward_width(members, fwd, index)
        records.append(rec)
    return records


def summarize(records: list[dict], sizes=POOL_SIZES) -> dict:
    n_p = len(records)
    out = {}
    for n in sizes:
        constructible = sum(r["eligible_count"] >= n for r in records)
        js = [jaccard(records[i]["members"][n], records[i + 1]["members"][n])
              for i in range(n_p - 1)]
        # 只读诊断：剔除任一臂凑不够 N 的相邻对（首期数据边界会造成伪低 Jaccard）
        both = [records[i]["eligible_count"] >= n
                and records[i + 1]["eligible_count"] >= n for i in range(n_p - 1)]
        clean = [j for j, ok in zip(js, both) if ok]
        turns = [1 - len(set(records[i]["members"][n])
                         & set(records[i + 1]["members"][n])) / n
                 for i in range(n_p - 1)]
        widths = [r["width"][n] for r in records]
        s1 = constructible / n_p if n_p else 0.0
        s3 = sum(w >= MIN_CROSS_SECTION for w in widths) / n_p if n_p else 0.0
        med_j = float(np.median(js)) if js else 0.0
        c1, c2 = s1 >= 0.90, med_j >= 0.60
        c3 = s3 >= 0.90 if n >= MIN_CROSS_SECTION else True
        out[str(n)] = {
            "constructible_share": round(s1, 4),
            "constructible_periods": int(constructible),
            "failed_periods": [r["t"] for r in records if r["eligible_count"] < n],
            "jaccard_median": round(med_j, 4),
            "jaccard_min": round(float(np.min(js)), 4) if js else None,
            "jaccard_median_constructible_pairs":
                round(float(np.median(clean)), 4) if clean else None,
            "jaccard_min_constructible_pairs":
                round(float(np.min(clean)), 4) if clean else None,
            "excluded_unconstructible_pairs": int(len(js) - len(clean)),
            "mean_turnover": round(float(np.mean(turns)), 4) if turns else None,
            "width_median": round(float(np.median(widths)), 2),
            "width_min": int(np.min(widths)), "width_ge_30_share": round(s3, 4),
            "supports_cross_section_30": bool(s3 >= 0.90),
            "gate": {"C1": bool(c1), "C2": bool(c2), "C3": bool(c3),
                     "passed": bool(c1 and c2 and c3)}}
    return out


def posthoc_diagnostic(records: list[dict], sizes=POOL_SIZES) -> dict:
    """当前成员池在历史各期的成交额排名分位（事后选币的非价格代理）。"""
    current = {n: records[-1]["members"][n] for n in sizes}
    out = {}
    for n in sizes:
        series = []
        for r in records:
            prs = [rank_percentile(r["ranks"].get(s), r["eligible_count"])
                   for s in current[n]]
            valid = [p for p in prs if p is not None]
            series.append({"t": r["t"],
                           "median_percentile":
                               round(float(np.median(valid)), 4) if valid else None,
                           "not_eligible_share": round(1 - len(valid) / n, 4)})
        k = max(1, len(series) // 3)
        early, late = series[:k], series[-k:]
        med = lambda items, key: (  # noqa: E731
            round(float(np.median([x[key] for x in items if x[key] is not None])), 4)
            if any(x[key] is not None for x in items) else None)
        out[str(n)] = {
            "current_pool_size": n,
            "early_median_percentile": med(early, "median_percentile"),
            "late_median_percentile": med(late, "median_percentile"),
            "early_not_eligible_share": med(early, "not_eligible_share"),
            "late_not_eligible_share": med(late, "not_eligible_share"),
            "all_periods_not_eligible_share_max":
                round(max(x["not_eligible_share"] for x in series), 4),
            "series": series}
    return out


def run(args) -> dict:
    engine = make_engine(read_only=True)
    with engine.connect() as conn:
        mat, grid, symbols, dq = build_matrix(conn)
    periods = make_periods(grid, args.step_days)
    records = collect(mat, grid, symbols, periods)
    summary, diag = summarize(records), posthoc_diagnostic(records)
    feasible = [n for n in POOL_SIZES if summary[str(n)]["gate"]["passed"]]
    cs_feasible = [n for n in feasible if summary[str(n)]["supports_cross_section_30"]]
    margin = (round(summary[str(max(feasible))]["jaccard_median"] - 0.60, 4)
              if feasible else None)
    return {
        "topic": "H-D / P16 固定规模 PIT 币池可构造性与池规模敏感性",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "backend/scripts/probe_p16_fixed_pool.py",
        "hypothesis_card": "research/hypotheses/H-D-fixed-size-universe.md",
        "stage": "Gate 0 + 数据口径可行性诊断（非回测、非 edge 判定）",
        "window": {"start_utc": iso(dq["window_start_ms"]), "end_utc": iso(dq["window_end_ms"]),
                   "days": round((dq["window_end_ms"] - dq["window_start_ms"]) / DAY_MS, 2),
                   "rebuild_periods": len(periods), "step_days": args.step_days,
                   "first_period_utc": iso(periods[0]) if periods else None,
                   "last_period_utc": iso(periods[-1]) if periods else None},
        "universe": {"source": "klines interval=1h, quote_volume + bar presence",
                     "symbols_after_filter": len(symbols), "pool_sizes": list(POOL_SIZES),
                     "selection": "per-period pre-window 168h quote_volume TopN",
                     "note": "不读 close / 不算收益 / 不构造特征，纯样本选择口径"},
        "parameters": {"lookback_hours": LOOKBACK_HOURS, "min_bars_per_window": MIN_BARS,
                       "stale_hours": STALE_HOURS, "forward_hours": FORWARD_HOURS,
                       "min_cross_section": MIN_CROSS_SECTION,
                       "gate_thresholds": GATE_THRESHOLDS,
                       "sort": "quote_volume desc, symbol asc tie-break"},
        "data_quality": dq,
        "eligible_count_series": [{"t": r["t"], "eligible_count": r["eligible_count"]}
                                  for r in records],
        "results_by_pool_size": summary, "posthoc_selection_diagnostic": diag,
        "costs": {"fee_rate": 0.0004, "slippage": 0.0002, "round_trip": 0.0012,
                  "basis": "notional", "applied": False,
                  "reason": "trades=0, no PnL evaluation this round"},
        "ci": {"confidence": 0.95, "method": None, "applicable": False,
               "reason": "no return estimate; data-schema feasibility check"},
        "random_baseline": {"applicable": False,
                            "reason": "no strategy evaluation this round"},
        "trades": 0, "default_enabled": False,
        "decision": {
            "feasible_pool_sizes": feasible,
            "pool_sizes_supporting_min_cs_30": cs_feasible,
            "recommended_pool_size": min(cs_feasible) if cs_feasible else None,
            "c2_margin_at_largest_feasible_n": margin,
            "action": "FEASIBLE_WITH_BOUNDS" if cs_feasible else "NOT_FEASIBLE",
            "caveat": ("C2 门槛 0.60 为事前设定，本轮不得调整；通过者的 margin 很小，"
                       "不构成该 N 最优的证据。周均换手 26–33% 说明『固定规模』"
                       "不等于『固定成分』，PIT 池成员会逐期重选。"),
            "note": ("若后续要跑 MIN_CROSS_SECTION=30 的横截面探针，N 须同时满足 gate "
                     "与 width_ge_30_share>=0.90；N<30 的 C3 按预注册豁免，"
                     "但不支持 30 币横截面判据。")},
        "verdict": "operational_pass" if cs_feasible else "operational_fail",
        "verdict_scope": ("工程可行性判定，非手册第 3 节 edge verdict；"
                          "无收益估计、无 CI 下界判定、无启用建议"),
        "limitations": [
            "无上市/下架事件表与历史 ticker 快照，无法验证当时真实可交易性；结论基于在库 1h bar 下界",
            "首期（2025-09-28）仅 6 币有 168 根 bar：394 币的历史从 09-22 才起，属回填起点不齐，非量具缺陷",
            "quote_volume=0 单根按有效 bar 计（不读成无信号），窗口总和 >0 才合格",
            "合格门槛沿用 pit_pool_check.rebuild 的放宽口径（1h >=160 根、末根不早于 T-2h）",
            "只验证样本选择口径，不做策略收益/IC/净 edge 判定，不重复 8.1pp 收益对照；成交额排序不含点差/盘口/执行成本",
            "成员稳定性按 7 天一期测量，未覆盖日内换手；有效宽度要求 [T,T+24h) 全 24 根 bar，属最严格口径",
            "本轮不改 StrategySpec / strategy_bindings / 参数 / 前端，不部署模拟盘、不下单"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/probe-p16-fixed-pool.json")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--step-days", type=int, default=STEP_DAYS)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"输出已存在：{out}；如需覆盖加 --force")
    result = run(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                              allow_nan=False), encoding="utf-8")
    print(json.dumps({"out": str(out), "verdict": result["verdict"],
                      "decision": result["decision"]["action"],
                      "recommended_pool_size": result["decision"]["recommended_pool_size"],
                      "feasible": result["decision"]["feasible_pool_sizes"],
                      "periods": result["window"]["rebuild_periods"]},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
