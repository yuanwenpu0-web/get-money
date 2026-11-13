"""测评前置：建短周期数据 → 质量门 → 币种池分档 → 冻结口径（S0+S1）。

两轮 scope（同一份 klines，两种窗口，各自独立哈希与缓存目录）：
  A `comparable-90d`：四周期同币种、同 90 天窗，主结论用。币种数受限于 1m 的覆盖
    （只有约 73 个币有满密度 1m），所以 5m/15m/30m 都从 1m 派生，1h 用 native。
  B `long-window`：1h 400 天 + 15m/30m 300 天，外部效度用——90 天只切成 4 个时间分段，
    「稳定」必须在更长的窗口上再判一次。这一轮币种数受限于 15m native 的历史长度。

为什么 A 轮不直接用 native 15m：实测 native 15m 在 90 天内只有 31 个币达到密度门槛，
且与 1m 覆盖的 73 币只重叠 5 个。若 A 轮用 native 15m，「5m 比 15m 好」这类结论就永远
分不清是周期效应还是换了批币。

派生只补「该周期本身不合格」的币；已有满密度 native 的币不动，避免用聚合行去覆盖
交易所原始行。

用法：
    .venv\\Scripts\\python.exe backend/scripts/fit_prep.py                  # 两轮全流程（会写库）
    .venv\\Scripts\\python.exe backend/scripts/fit_prep.py --scope A        # 只做 A 轮
    .venv\\Scripts\\python.exe backend/scripts/fit_prep.py --skip-derive    # 只重跑门与池
    .venv\\Scripts\\python.exe backend/scripts/fit_prep.py --dry-run        # 只读，产物写 docs/tmp-fit

产出（全部由同一次运行写出，时间戳一致）：
    docs/indicator-eval-scope.json      S0 口径冻结，后续产物引用各自的 sha
    docs/data-quality-<scope>-<iv>.json 每轮每周期逐币质量门明细
    docs/fit-universe.json              两轮入选币种、池分档、档位稳定率、ATR% 代理
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from sqlalchemy import bindparam, select, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.db import make_engine  # noqa: E402
from app.market.derive import derive_interval  # noqa: E402
from app.models import SymbolPool  # noqa: E402
from app.research import fit_data as fd  # noqa: E402
from app.research.fit_stats import GateConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
POOL_CODES = {1: "liq_q1", 2: "liq_q2", 3: "liq_q3", 4: "liq_q4"}
POOL_NAMES = {
    "liq_q1": "流动性 Q1（成交额最高档）",
    "liq_q2": "流动性 Q2",
    "liq_q3": "流动性 Q3",
    "liq_q4": "流动性 Q4（成交额最低档）",
}
MIN_FILL = 0.75
# 派生决策用的「窗口已铺满」阈值，比评估门槛 MIN_WINDOW_COVERAGE 严格：
# 覆盖率 0.5–0.9 之间的币（典型是 native 只采到最近一段）不剔除，但要把头部补齐，
# 否则同一轮里 5m 有 90 天、1h 只有 59 天，四周期交集与分段同号都会被上市时间错位污染。
COVERAGE_FOR_DERIVE = 0.9

# lookback = 窗口天数 + 暖机余量：派生要盖住整个评估窗，首尾未闭合的桶还会被丢掉
SCOPES: dict[str, dict] = {
    "A": {
        "label": "comparable-90d",
        "purpose": "四周期同币种同窗口：周期 × 池矩阵的主结论",
        "anchor_interval": "1m",
        "window_days": {"5m": 90, "15m": 90, "30m": 90, "1h": 90},
        "intervals": ("5m", "15m", "30m", "1h"),
        "derive": {"5m": ("1m", 95), "15m": ("1m", 95), "30m": ("1m", 95),
                   "1h": ("15m", 95)},
        "pool_window_days": 90,
    },
    "B": {
        "label": "long-window",
        "purpose": "长窗外部效度：1h 400d、15m/30m 300d（受 native 历史长度限制）",
        "anchor_interval": "1h",
        "window_days": {"15m": 300, "30m": 300, "1h": 400},
        "intervals": ("15m", "30m", "1h"),
        "derive": {"30m": ("15m", 320)},
        "pool_window_days": 400,
    },
}


def sha(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def scope_spec(name: str) -> dict:
    """S0：把这一轮所有会影响结论的口径写死，之后只读不改。"""
    cfg = SCOPES[name]
    tc = GateConfig()
    return {
        "scope": name,
        "label": cfg["label"],
        "purpose": cfg["purpose"],
        "version": "indicator-fit-2026-11-12",
        "intervals": list(cfg["intervals"]),
        "window_days": dict(cfg["window_days"]),
        "max_symbols": fd.MAX_SYMBOLS,
        "horizons_bars": [6, 24, 96],
        "cost": {"fee_side": 0.0004, "slip_side": 0.0002, "round_trip": 0.0012,
                 "stress": list(tc.cost_stress), "override_allowed": False},
        "direction_mode": "aligned",
        "direction_source": "docs/indicator-directions.json",
        "benchmarks": ["unconditional", "cross_section"],
        "gates": {
            "bootstrap_ci_low_gt": 0.0,
            "fdr_q_lt": tc.q_alpha,
            "segments": 4,
            "segment_sign_min": tc.seg_sign_min,
            "symbol_sign_min": tc.sym_sign_min,
            "cell_events_min": tc.cell_events_min,
            "cell_symbols_min": tc.cell_symbols_min,
            "interaction_p_lt": tc.interaction_alpha,
            "shrink_k": tc.shrink_k,
            "pooled_events_min": tc.pooled_events_min,
            "sym_events_min": tc.sym_events_min,
            "hit_rate_range": [tc.min_hit, tc.max_hit],
        },
        "data_gates": {
            "min_density": fd.MIN_DENSITY,
            "min_history_bars": fd.MIN_HISTORY_BARS,
            "max_gap_rate": 0.02,
            "max_zero_volume_rate": 0.05,
        },
        "pools": ["all", "major", "alt"] + [POOL_CODES[i] for i in (1, 2, 3, 4)],
        "entry": "signal 在 t 收盘确认，t+1 开盘成交，t+h 收盘出场",
        "seed": tc.seed,
        "n_boot": tc.n_boot,
        "provenance": {
            "rule": "同一轮同一周期内某币只用一个来源：native 满密度就用 native，否则派生",
            "derive": {iv: {"source": src, "lookback_days": lb, "min_fill": MIN_FILL}
                       for iv, (src, lb) in cfg["derive"].items()},
        },
    }


def dense_symbols(db, interval: str, days: int, limit: int | None = None,
                  min_coverage: float = fd.MIN_WINDOW_COVERAGE) -> list[str]:
    """该周期达到评估门槛的币种；给了 limit 就按窗口成交额截断，避免门跑在 500 个币上。

    `min_coverage` 分成两档用：候选集用评估门槛（0.5），派生决策用 `COVERAGE_FOR_DERIVE`
    （0.9）——后者决定「要不要给这个币补派生头部」，前者决定「这个币够不够格进评估」。
    """
    frame = fd.coverage_rows(db, interval, days)
    if frame.empty:
        return []
    ok = frame[
        (frame["density"] >= fd.MIN_DENSITY)
        & (frame["n"] >= fd.MIN_HISTORY_BARS)
        & (frame["win_coverage"] >= min_coverage)
    ]
    ok = ok.sort_values("quote_volume", ascending=False)
    if limit:
        ok = ok.head(limit)
    return ok["symbol"].tolist()


def anchor_universe(db, cfg: dict, limit: int) -> list[str]:
    """币种集由本轮「最短木板」的锚周期决定，保证各周期尽量拿到同一批币。"""
    anchor = cfg["anchor_interval"]
    days = cfg["window_days"].get(anchor, max(cfg["window_days"].values()))
    return fd.select_candidates(db, anchor, days=days, limit=limit)


def run_gate(db, scope: str, interval: str, symbols: list[str], days: int,
             prefix: str = "docs") -> list[dict]:
    rows = fd.quality_gate(db, interval, symbols, days=days)
    kept = [r for r in rows if not r.reject]
    reasons: dict[str, int] = {}
    for r in rows:
        for reason in r.reject:
            reasons[reason] = reasons.get(reason, 0) + 1
    payload = {
        "scope": scope,
        "interval": interval,
        "window_days": days,
        "checked": len(rows),
        "passed": len(kept),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reject_reasons": reasons,
        "symbols": [r.as_dict() for r in sorted(rows, key=lambda x: x.symbol)],
    }
    (ROOT / f"{prefix}/data-quality-{scope}-{interval}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload["symbols"]


def atr_proxy(db, interval: str, symbols: list[str], days: int) -> dict[str, float]:
    """每币窗口内平均 (high-low)/close，作为波动率代理用于成本占比。

    为什么不调 ATR 指标：这里只需要一个跨币可比的量级，`(H-L)/C` 的窗口均值与 ATR% 同阶，
    一条聚合 SQL 就能算完，不必为 90 币 × 4 周期再跑一遍指标库。
    """
    if not symbols:
        return {}
    hi = db.execute(
        text("select max(open_time) from klines where interval=:iv"), {"iv": interval}
    ).scalar()
    if hi is None:
        return {}
    lo = int(hi - days * fd.DAY_MS)
    out: dict[str, float] = {}
    sql = text(
        "select symbol, avg((high - low) / nullif(close, 0)) v from klines "
        "where interval=:iv and open_time>=:lo and symbol in :syms group by symbol"
    ).bindparams(bindparam("syms", expanding=True))
    chunk = 40
    for start in range(0, len(symbols), chunk):
        rows = db.execute(
            sql, {"iv": interval, "lo": lo, "syms": symbols[start : start + chunk]}
        ).mappings().all()
        for row in rows:
            if row["v"]:
                out[str(row["symbol"])] = round(float(row["v"]), 6)
    return out


def persist_pools(db, members: dict[str, list[str]], note: str) -> None:
    """把 Q1–Q4 写成 static 池，沿用研究池的 maintenance_owner 约定。

    `symbols` / `rule` 在 `symbol_pools` 里是 **Text 列存 JSON 文本**（见 models.SymbolPool
    与 api/pools.create_pool），直接塞 dict/list 会在 autoflush 时报
    `can't adapt type 'dict'`，所以这里必须 json.dumps，与线上写池的口径保持一致。
    """
    for rank, code in POOL_CODES.items():
        symbols = sorted(members.get(code, []))
        if not symbols:
            continue
        row = db.execute(select(SymbolPool).where(SymbolPool.code == code)).scalar_one_or_none()
        payload = {
            "name": POOL_NAMES[code],
            "description": f"{note}；成员为评估币种集按窗口内月度成交额中位数的 Q{rank} 档",
            "kind": "static",
            "symbols": json.dumps(symbols, ensure_ascii=False),
            "rule": json.dumps({
                "base": "static",
                "maintenance_owner": "liquidity_quartiles_v1",
                "window_hours": 24 * 90,
                "window_end_ms": int(time.time() * 1000),
            }, ensure_ascii=False),
            "enabled": 1,
        }
        if row is None:
            db.add(SymbolPool(code=code, **payload))
        else:
            for key, value in payload.items():
                setattr(row, key, value)
    db.commit()


def prepare_scope(db, name: str, args, prefix: str) -> dict:
    cfg = SCOPES[name]
    spec = scope_spec(name)
    spec["sha"] = sha(spec)
    print(f"[scope {name}] {spec['sha']} {cfg['label']} 窗口={cfg['window_days']}", flush=True)

    anchor = anchor_universe(db, cfg, args.limit)
    print(f"[anchor {name}] {cfg['anchor_interval']}: {len(anchor)} 币", flush=True)

    # 1) 派生：只补「该周期本身没铺满窗口」的币，native 达标的币一律不覆盖
    dense: dict[str, list[str]] = {
        iv: dense_symbols(db, iv, cfg["window_days"][iv], limit=args.limit)
        for iv in cfg["intervals"]
    }
    # 派生决策看覆盖率（比评估门槛严）：59/90 天的 1h 够格进评估，但头部该补齐
    covered: dict[str, set[str]] = {
        iv: set(dense_symbols(db, iv, cfg["window_days"][iv], min_coverage=COVERAGE_FOR_DERIVE))
        for iv in cfg["intervals"]
    }
    derive_stats: dict[str, dict] = {}
    for interval, (src, lookback) in cfg["derive"].items():
        need = [s for s in anchor if s not in covered.get(interval, set())]
        if not need:
            derive_stats[interval] = {"source": src, "symbols": 0, "rows": 0, "skipped_all": True}
            print(f"[derive {name}] {src}->{interval}: 候选币都已满密度，跳过", flush=True)
            continue
        since = int(time.time() * 1000) - lookback * fd.DAY_MS
        if args.dry_run or args.skip_derive:
            print(f"[derive {name}] {src}->{interval}: {len(need)} 币待派生（本次未写库）", flush=True)
            derive_stats[interval] = {"source": src, "pending_symbols": len(need), "written": False}
            continue
        # 1m 是四千多万行的大表：一次读 50 个币 × 95 天 = 680 万行进 pandas 会把内存
        # 顶到 6GB 以上，派生 1m 上游时把批次压到 20 个币
        res = derive_interval(db, source=src, target=interval, symbols=need,
                              since_ms=since, min_fill=MIN_FILL,
                              batch=20 if src == "1m" else 50)
        derive_stats[interval] = {**res, "written": True}
        print(f"[derive {name}] {src}->{interval}: {res}", flush=True)

    # 2) 质量门：候选集 = 锚定币种 ∪ 该周期已密集的币种（截断到 limit，避免门跑在 500 币上）
    universes: dict[str, list[str]] = {}
    rejects: dict[str, dict] = {}
    for interval in cfg["intervals"]:
        days = cfg["window_days"][interval]
        pool = sorted(set(anchor) | set(dense.get(interval, [])))
        detail = run_gate(db, name, interval, pool, days, prefix=prefix)
        kept = sorted(d["symbol"] for d in detail if not d["reject"])
        universes[interval] = kept
        rejects[interval] = {d["symbol"]: d["reject"] for d in detail if d["reject"]}
        print(f"[gate {name}] {interval}: {len(kept)}/{len(detail)} 通过（{days}d）", flush=True)

    common = sorted(set.intersection(*[set(v) for v in universes.values()])) if universes else []
    print(f"[common {name}] 交集 {len(common)} 币（跨周期归因只用这批）", flush=True)

    # 3) 池：用 1h 的月度成交额（覆盖最全、月度口径跨周期可比）分档。
    #    分档必须在**本轮评估币种集内部**切：拿 500 币的四分位去切 73 币，档位会全部
    #    挤在同一档里，「指标适合哪种池」这个问题就没法回答了。
    all_syms = sorted(set().union(*[set(v) for v in universes.values()])) if universes else []
    ref = common or all_syms
    monthly = fd.monthly_quote_volume(db, "1h", ref, days=cfg["pool_window_days"]) \
        if ref else pd.DataFrame()
    bins = fd.pool_members_from_frame(monthly, groups=4)
    assigned = fd.assign_fixed_tiers(monthly, groups=4)
    members: dict[str, list[str]] = {}
    for symbol, tier in assigned.items():
        # 档位标签是 "Q1".."Q4"，池 code 是 "liq_q1".."liq_q4"：这里必须换算，
        # 上一轮直接拿标签当 key，persist_pools 查 code 查不到，四个池全空。
        rank = int(str(tier)[1:])
        members.setdefault(POOL_CODES[rank], []).append(symbol)
    if not args.dry_run and name == "A":
        persist_pools(db, members, note=f"A 轮口径 {spec['sha']}")

    return {
        "spec": spec,
        "derive": derive_stats,
        "anchor": cfg["anchor_interval"],
        "anchor_size": len(anchor),
        "universe": universes,
        "universe_size": {iv: len(v) for iv, v in universes.items()},
        "rejected": rejects,
        "pairwise_overlap": {
            f"{a}&{b}": len(set(universes.get(a, [])) & set(universes.get(b, [])))
            for a in cfg["intervals"] for b in cfg["intervals"] if a < b
        },
        "common_universe": common,
        "monthly_liquidity_bins": bins,
        "fixed_tiers": assigned,
        "tier_stability": fd.tier_stability(bins, assigned),
        "pool_snapshot": {POOL_CODES[i]: sorted(members.get(POOL_CODES[i], [])) for i in (1, 2, 3, 4)},
        "pool_sizes": {POOL_CODES[i]: len(members.get(POOL_CODES[i], [])) for i in (1, 2, 3, 4)},
        "atr_pct": {iv: atr_proxy(db, iv, common, cfg["window_days"][iv]) for iv in cfg["intervals"]},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="A,B", help="A / B / A,B")
    ap.add_argument("--skip-derive", action="store_true", help="不派生（数据已就绪时）")
    ap.add_argument("--dry-run", action="store_true", help="只读：不写 klines、不写 symbol_pools")
    ap.add_argument("--limit", type=int, default=fd.MAX_SYMBOLS, help="每轮最多币种数")
    args = ap.parse_args()

    engine = make_engine(read_only=args.dry_run)
    db = sessionmaker(bind=engine)()
    started = time.time()
    prefix = "docs/tmp-fit" if args.dry_run else "docs"
    (ROOT / prefix).mkdir(parents=True, exist_ok=True)

    names = [s.strip().upper() for s in args.scope.split(",") if s.strip()]
    unknown = [s for s in names if s not in SCOPES]
    if unknown:
        raise SystemExit(f"未知 scope: {unknown}，可选 {sorted(SCOPES)}")

    scopes: dict[str, dict] = {}
    for name in names:
        scopes[name] = prepare_scope(db, name, args, prefix)

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for name, s in scopes.items():
        print(f"[{name}] 周期币数={s['universe_size']} 交集={len(s['common_universe'])} "
              f"池={s['pool_sizes']}", flush=True)
    if args.dry_run:
        print(f"[dry-run] {time.time() - started:.0f}s，未写库，产物在 {prefix}/")
        return 0

    (ROOT / "docs/indicator-eval-scope.json").write_text(
        json.dumps({n: s["spec"] for n, s in scopes.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    uni_doc = {n: {k: v for k, v in s.items() if k != "spec"} for n, s in scopes.items()}
    (ROOT / "docs/fit-universe.json").write_text(
        json.dumps({"generated_at": stamp, "table": "klines", "scopes": uni_doc},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done ] {time.time() - started:.0f}s -> indicator-eval-scope.json + fit-universe.json",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
