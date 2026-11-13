"""P18 成员口径审计的独立复算（只读）。

刻意**不复用** fit_data / fit_membership 的实现：直接 SQL 取月度成交额，用纯 Python
`calendar.monthrange` 算应有柱数、手写中位数与 ceil(n/4) 切档。目的是避免「用同一套
代码自证同一套数字」——审计脚本改了口径而这里没跟着改，才对得上才算独立验证。

同时核验：冻结产物指纹未变、台账编号无重复。
"""
from __future__ import annotations

import calendar
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.db import make_engine  # noqa: E402

DOCS = ROOT / "docs"
GROUPS = 4
HISTORY_MONTHS = 15


def manual_expected_bars(ym: str, interval_hours: int = 1) -> int:
    y, m = map(int, ym.split("-"))
    return calendar.monthrange(y, m)[1] * (24 // interval_hours)


def fetch_monthly(db, ref, lo_ms, hi_ms):
    sql = text(
        """select symbol, to_char(to_timestamp(open_time/1000) at time zone 'UTC','YYYY-MM') ym,
                  sum(coalesce(quote_volume,0)) v, count(*) bars
           from klines
          where interval='1h' and open_time>=:lo and open_time<:hi and symbol in :s
          group by symbol, ym"""
    ).bindparams(**{"s": tuple(ref)})
    rows = db.execute(sql, {"lo": lo_ms, "hi": hi_ms, "s": tuple(ref)}).mappings().all()
    out: dict[str, dict] = {}
    for r in rows:
        out.setdefault(r["ym"], {})[r["symbol"]] = (
            float(r["v"]), int(r["bars"]), manual_expected_bars(r["ym"]))
    return out


def hand_tiers(months, use, ref, groups=GROUPS, force_complete=False):
    """手写中位数 + ceil(n/groups) 切档；force_complete 模拟溢出使 partial 恒 False。"""
    med = {}
    for sym in ref:
        vals = []
        for ym in use:
            cell = months.get(ym, {}).get(sym)
            if cell is None:
                continue
            v, bars, exp = cell
            if force_complete or bars >= 0.7 * exp:
                vals.append(v)
        if vals:
            vals.sort()
            n = len(vals)
            med[sym] = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    order = sorted(med, key=lambda s: (-med[s], s))
    per = max(1, -(-len(order) // groups))
    return {s: f"Q{min(i // per, groups - 1) + 1}" for i, s in enumerate(order)}


def main() -> int:
    pay = json.loads((DOCS / "indicator-fit-membership.json").read_text(encoding="utf-8"))
    uni = json.loads((DOCS / "fit-universe.json").read_text(encoding="utf-8"))["scopes"]["A"]
    end_ms = int(pay["window"]["end_ms"])
    lo = int(pay["window"]["start_ms"])
    ref = list(uni["common_universe"])
    frozen = dict(uni["fixed_tiers"])
    print(f"anchor end_ms={end_ms} ({pd.Timestamp(end_ms, unit='ms', tz='UTC').isoformat()}) lo={lo}")
    for ym in ("2026-08", "2026-09", "2026-10", "2026-11", "2026-02"):
        print(f"  manual expected_bars {ym} = {manual_expected_bars(ym)}")

    engine = make_engine(read_only=True)
    db = sessionmaker(bind=engine)()
    checks: dict[str, bool] = {}
    try:
        ro = db.execute(text("SHOW transaction_read_only")).scalar()
        checks["read_only"] = ro == "on"
        print(f"transaction_read_only={ro}")
        db.execute(text("SET LOCAL statement_timeout = '120s'"))
        win = fetch_monthly(db, ref, lo, end_ms + 3_600_000)
        wmonths = sorted(win)
        print("window months", wmonths)
        buggy = hand_tiers(win, wmonths, ref, force_complete=True)
        corrected = hand_tiers(win, wmonths, ref)
        repro_diff = sum(1 for s in frozen if buggy.get(s) != frozen[s])
        changed = [s for s in frozen if corrected.get(s) != frozen[s]]
        dropped = sorted(set(frozen) - set(corrected))
        print(f"buggy-vs-frozen changed={repro_diff} missing={len(set(frozen) - set(buggy))}")
        print(f"corrected changed={len(changed)} dropped={len(dropped)}")
        checks["a_round_repro_match"] = repro_diff == 0 and not (set(frozen) - set(buggy))
        checks["corrected_changed_matches"] = len(changed) == len(
            pay["corrected_recompute"]["tier_delta"]["changed"])
        checks["corrected_dropped_matches"] = len(dropped) == len(
            pay["corrected_recompute"]["tier_delta"]["dropped"])

        hist_lo = int((pd.Timestamp(end_ms, unit="ms", tz="UTC")
                       - pd.DateOffset(months=HISTORY_MONTHS)).value // 1_000_000)
        hist = fetch_monthly(db, ref, hist_lo, end_ms + 3_600_000)
        hmonths = sorted(hist)
        print(f"history months {hmonths[0]}..{hmonths[-1]} n={len(hmonths)}")
        tc = td = 0
        for ym in wmonths:
            prior = [m for m in hmonths if m < ym]
            t = hand_tiers(hist, prior, ref)
            both = sorted(set(t) & set(frozen))
            d = [s for s in both if t[s] != frozen[s]]
            tc += len(both)
            td += len(d)
            print(f"  PIT {ym}: prior_raw={len(prior)} comparable={len(both)} disagree={len(d)}")
        rate = round(td / tc, 4) if tc else None
        print(f"PIT total comparable={tc} disagree={td} rate={rate}")
        checks["pit_matches"] = (tc == pay["pit"]["comparable_total"]
                                 and td == pay["pit"]["disagree_total"]
                                 and rate == pay["pit"]["disagree_rate"])
        ac = ad = 0
        for ym in wmonths:
            prior = [m for m in wmonths if m < ym]
            t = hand_tiers(win, prior, ref, force_complete=True)
            both = sorted(set(t) & set(frozen))
            d = [s for s in both if t[s] != frozen[s]]
            ac += len(both)
            ad += len(d)
            print(f"  as-run {ym}: prior={prior} comparable={len(both)} disagree={len(d)}")
        arate = round(ad / ac, 4) if ac else None
        print(f"isolated total comparable={ac} disagree={ad} rate={arate}")
        checks["lookahead_isolated_matches"] = (
            ac == pay["lookahead_isolated"]["comparable_total"]
            and ad == pay["lookahead_isolated"]["disagree_total"]
            and arate == pay["lookahead_isolated"]["disagree_rate"])
    finally:
        db.close()
        engine.dispose()

    prev = json.loads((DOCS / "indicator-fit-readiness.json").read_text(encoding="utf-8"))["source_hashes"]
    for name, h in prev.items():
        now = hashlib.sha256((DOCS / name).read_bytes()).hexdigest()
        checks[f"hash_unchanged:{name}"] = now == h
        print(f"hash {name}: {'UNCHANGED' if now == h else 'CHANGED!!'}")
    for name in ("fit-universe.json", "indicator-fit-stats-A.json"):
        now = hashlib.sha256((DOCS / name).read_bytes()).hexdigest()
        checks[f"hash_unchanged:{name}"] = now == pay["source_hashes"][name]
        print(f"hash {name}: {'UNCHANGED' if now == pay['source_hashes'][name] else 'CHANGED!!'}")

    ids = [json.loads(ln)["experiment_id"]
           for ln in (ROOT / "research/experiments.jsonl").read_text(encoding="utf-8").splitlines()
           if ln.strip()]
    counts = Counter(ids)
    dupes = [k for k, v in counts.items() if v > 1]
    checks["ledger_no_duplicates"] = not dupes
    print("ledger P18 ids:", {k: v for k, v in counts.items() if k.startswith("P18")})
    print("duplicates:", dupes)
    failed = [k for k, v in checks.items() if not v]
    print(f"INDEPENDENT VERIFY: {len(checks) - len(failed)}/{len(checks)} checks passed"
          + (f"; FAILED={failed}" if failed else "; all matched"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())