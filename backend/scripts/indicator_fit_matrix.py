"""S3–S5 主调度：全指标 × 四周期 × 币种池 的「适配矩阵」。

这是把测评真正跑起来的那根线，本身不做判定，只负责编排与产物：

    probe   小样本端到端（缓存、方向表、统计全链路），用来估时与接线自检
    events  逐周期抽事件（可断点续跑，缓存按口径哈希分目录）
    stats   从缓存出统计、FDR、分层判定与收缩，落 docs/indicator-fit-stats-<scope>.json

前置：先跑 `backend/scripts/fit_prep.py --scope A`（S0 口径冻结 + S1 数据门 + Q1–Q4 池）。
本脚本只读那两个产物，不在这里重新决定窗口、门槛或池成员——口径变了必须重跑 prep，
否则哈希对不上、缓存目录也是新的，不会串味。

用法：
    .venv\\Scripts\\python.exe backend/scripts/indicator_fit_matrix.py --scope A --stage probe
    .venv\\Scripts\\python.exe backend/scripts/indicator_fit_matrix.py --scope A --stage events --workers 10
    .venv\\Scripts\\python.exe backend/scripts/indicator_fit_matrix.py --scope A --stage stats
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.db import make_engine  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.research import fit_events as fe  # noqa: E402
from app.research.fit_events import CacheLayout, load_cached, run_interval  # noqa: E402
from app.research.fit_stats import (  # noqa: E402
    PROMOTABLE, Evaluator, GateConfig, apply_fdr, best_cell, shrink, verdict,
)
from app.strategy.pools import pool_symbols_by_code  # noqa: E402
from app.utils import interval_ms  # noqa: E402

DAY_MS = 86_400_000
# 矩阵原子行（每列 × 每池的最强格）只带判定与展示要用的字段：全字段是 30 多个键，
# 885 列 × 8 池 × 4 周期落一遍就是几十 MB，而读者在这些格子上只会看这几个数。
ATOM_FIELDS = (
    "key", "scope", "horizon_bars", "benchmark", "direction", "direction_source",
    "verdict", "verdict_cell", "events", "symbols_with_events", "symbols_in_stat",
    "mean_gross_pct", "mean_net_pct", "ci_low_pct", "ci_high_pct", "q_value",
    "seg_consistency", "symbol_sign_rate", "cost_stress_pct", "shrunk_net_pct",
    "shrink_weight", "hit_rate",
)

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
CACHE = ROOT / "output" / "cache" / "fit"
POOL_CODES = {1: "liq_q1", 2: "liq_q2", 3: "liq_q3", 4: "liq_q4"}


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_prep(scope: str) -> tuple[dict, dict]:
    """读 S0 口径与 S1 币种池产物。缺文件直接失败——不许在这里「默认」任何口径。"""
    spec_doc = json.loads((DOCS / "indicator-eval-scope.json").read_text(encoding="utf-8"))
    uni_doc = json.loads((DOCS / "fit-universe.json").read_text(encoding="utf-8"))
    if scope not in spec_doc or scope not in uni_doc["scopes"]:
        raise SystemExit(f"scope {scope} 未准备，请先跑 fit_prep.py --scope {scope}")
    return spec_doc[scope], uni_doc["scopes"][scope]


def stat_scopes(db, uni: dict, interval: str, limit_symbols: int | None = None) -> dict[str, list[str]]:
    """统计阶段的币篮子：本周期 universe ∩ 各池。

    `all` 与 `common` 是两个必须都在的对照组：前者是「这个周期能评的所有币」，
    后者是「四个周期都合格的币」——跨周期比的就是这一列，不是 all。
    major/alt 走线上池定义解析（`pool_symbols_by_code`），保证「测评里的主流币」
    和「实盘绑定的主流币」是同一个口径，而不是我在脚本里再抄一份名单。
    """
    universe = list(uni["universe"][interval])
    if limit_symbols:
        universe = universe[:limit_symbols]
    wanted = set(universe)
    out: dict[str, list[str]] = {"all": universe}
    common = [s for s in uni.get("common_universe", []) if s in wanted]
    if len(common) >= 2:
        out["common"] = common
    for code in ("major", "alt"):
        members = pool_symbols_by_code(db, code, allowed=universe)
        if len(members) >= 2:
            out[code] = members
    snapshot = uni.get("pool_snapshot") or {}
    for rank in (1, 2, 3, 4):
        code = POOL_CODES[rank]
        members = [s for s in snapshot.get(code, []) if s in wanted]
        if len(members) >= 2:
            out[code] = members
    return out


def evaluator_for(uni: dict, interval: str, layout: CacheLayout, horizons: list[int]):
    """把一个 (轮, 周期) 的缓存拼成 Evaluator，并带回统计要用的辅助量。"""
    keys, syms, matrix, covered, key_meta, sym_meta = load_cached(
        layout, list(uni["universe"][interval]), horizons
    )
    if not keys or not syms:
        return None
    bars = {s: int(sym_meta.get(s, {}).get("bars", 0)) for s in syms}
    ev = Evaluator(keys, syms, matrix, covered, key_meta, bars)
    return ev, keys, syms, covered, matrix, sym_meta


# --------------------------------------------------------------------- directions


def stage_directions(db, args) -> int:
    """枚举全指标布尔列并写方向声明表 `docs/indicator-directions.json`。

    只需要**一帧真数据**：列集合由指标代码决定，与币种无关，所以跑一遍 156 个指标
    把布尔列收齐即可，不必按币跑。方向判定完全不看收益（`fit_directions` 只读列名与
    指标名），因此这张表可以先于事件抽取冻结，作为 S0 里 `direction_source` 引用的产物。
    指标名会带着 `calculate()` 的异常一起写出来——移植过来跑不动的指标必须显式暴露，
    而不是在事件阶段被 `except` 悄悄吞掉。
    """
    from app.catalog_indicators import list_indicators
    from app.research import fit_directions as fdir
    from app.research.fit_events import bool_column_positions

    svc = MarketDataService(db)
    now_ms = int(time.time() * 1000)
    frame = svc.load_frame(args.dir_symbol, "1h", limit=0,
                           start_ms=now_ms - args.dir_bars * 3_600_000, end_ms=now_ms)
    if frame.empty:
        raise SystemExit(f"{args.dir_symbol} 1h 取不到数据，无法枚举列")
    sample: dict[str, list[str]] = {}
    errors: dict[str, str] = {}
    t0 = time.time()
    for ind in list_indicators():
        try:
            out = ind.calculate(frame)
        except Exception as exc:  # noqa: BLE001
            errors[ind.name] = f"{type(exc).__name__}: {exc}"[:200]
            continue
        cols = [name for _, name in bool_column_positions(out)]
        if cols:
            sample[ind.name] = cols
    index = fdir.build_column_index(sample)
    index["generated_at"] = stamp()
    total = len(list_indicators())
    index["indicator_accounting"] = {
        "registered": total,
        "with_boolean_columns": len(sample),
        "raised": len(errors),
        # 跑通但没有布尔条件列的指标：它们只出数值线（overlay 类），不参与事件评估。
        # 这个数必须写出来，否则「156 个指标」和「885 列」之间的关系会被读成丢了大半。
        "no_boolean_columns": total - len(sample) - len(errors),
    }
    index["probe"] = {"symbol": args.dir_symbol, "bars": int(len(frame)),
                      "indicators": len(sample) + len(errors),
                      "indicator_errors": errors, "elapsed_s": round(time.time() - t0, 1)}
    fdir.write_index(index, DOCS / "indicator-directions.json")
    acc = index["indicator_accounting"]
    print(f"[dirs   ] 注册 {acc['registered']}：有布尔列 {acc['with_boolean_columns']}、"
          f"报错 {acc['raised']}、无布尔列 {acc['no_boolean_columns']}；"
          f"{index['count']} 列 declared={index['by_source']['declared']} "
          f"weak={index['by_source']['weak']} ambiguous={index['by_source']['ambiguous']} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return 0


# --------------------------------------------------------------------- probe


def keep_index() -> tuple[set[str], int]:
    """方向表 → （有布尔列的指标名集合, 列总数）。

    事件阶段按这份名单跳过其余指标：它们一列都不产出，却占掉 12.6% 的 `calculate()`
    时间（`docs/fit-speed-1h.json` 里逐指标计时可查）。列总数用来在第一个币上自检
    ——如果跳过后列数对不上，说明「列集合与币种/周期无关」这个前提不成立，立即报警。
    """
    dirs = json.loads((DOCS / "indicator-directions.json").read_text(encoding="utf-8"))
    return {c["indicator"] for c in dirs["columns"]}, int(dirs["count"])


def stage_probe(db, args) -> int:
    """端到端小样本：几个币 × 一个周期，验证接线并估时。

    缓存写到 `<sha>-probe` 命名空间，不写正式目录：探测只有 3 个币，横截面基准
    （要 ≥5 个可对齐币）必然全是 nan，混进正式缓存就会被全量运行当成有效产物。
    """
    spec, uni = load_prep(args.scope)
    horizons = spec["horizons_bars"]
    keep, expect_keys = keep_index()
    interval = (args.intervals or spec["intervals"])[0]
    symbols = uni["universe"][interval][: args.probe_symbols]
    layout = CacheLayout(root=CACHE, scope_sha=spec["sha"] + "-probe", interval=interval)
    svc = MarketDataService(db)
    t0 = time.time()
    res = run_interval(db, svc, interval, symbols, spec["window_days"][interval], layout,
                       workers=args.workers, horizons=horizons, resume=False, progress_every=1,
                       keep_names=keep, expect_keys=expect_keys)
    elapsed = time.time() - t0
    pack = evaluator_for(uni, interval, layout, horizons)
    sym_meta = pack[5] if pack else {}
    per_symbol = sorted(float(v.get("elapsed") or 0.0) for v in sym_meta.values())
    med = float(np.median(per_symbol)) if per_symbol else float("nan")
    per_bar = sorted(
        float(v["elapsed"]) / float(v["bars"]) for v in sym_meta.values()
        if v.get("elapsed") and v.get("bars")
    )
    spb = float(np.median(per_bar)) if per_bar else float("nan")
    jobs_total = sum(len(v) for v in uni["universe"].values())
    # 估算按「秒/bar」而不是「秒/币」：一次测评的工作量就是 bar 数 × 指标数，
    # 5m 一币 2.6 万根、1h 一币 2 千根，按币取中位会把 5m 的代价平摊到全部周期上。
    bars_total = {
        iv: len(syms) * (spec["window_days"][iv] * DAY_MS // interval_ms(iv))
        for iv, syms in uni["universe"].items()
    }
    est_cpu = spb * sum(bars_total.values()) / 3600 if spb == spb else float("nan")
    payload = {
        "generated_at": stamp(), "scope": args.scope, "sha": spec["sha"],
        "interval": interval, "symbols": symbols, "elapsed_s": round(elapsed, 1),
        "run_stats": res, "per_symbol_elapsed": per_symbol,
        "cache_keys": len(pack[1]) if pack else 0,
        "cache_symbols": len(pack[2]) if pack else 0,
        "columns_expected_from_directions": expect_keys,
        "median_seconds_per_symbol": round(med, 2),
        "median_seconds_per_bar": round(spb, 5) if spb == spb else None,
        "bars_total_by_interval": bars_total,
        "universe_size": {iv: len(v) for iv, v in uni["universe"].items()},
        "symbol_jobs_total": jobs_total,
        "stat_scopes": {k: len(v) for k, v in stat_scopes(db, uni, interval).items()},
        "estimate_cpu_hours": round(est_cpu, 1) if est_cpu == est_cpu else None,
        # 墙钟 = CPU / worker 数，再除以并行效率。效率不猜：真跑全量时用事件阶段的
        # 实际耗时回填。
        "estimate_wall_hours_at_workers": {
            w: round(est_cpu / w, 1) for w in (6, 10, 12)
        } if est_cpu == est_cpu else None,
    }
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / f"fit-probe-{args.scope}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[probe] {interval} {len(symbols)} 币 {elapsed:.0f}s，中位 {med:.1f}s/币 "
          f"{spb:.4f}s/bar；全量 {sum(bars_total.values())} bars "
          f"≈ {payload['estimate_cpu_hours']} CPU 小时（10 worker ≈ "
          f"{payload['estimate_wall_hours_at_workers'][10]} h 墙钟）；"
          f"scopes={payload['stat_scopes']}", flush=True)
    return 0


# --------------------------------------------------------------------- events


def stage_events(db, args) -> int:
    spec, uni = load_prep(args.scope)
    dirs_path = DOCS / "indicator-directions.json"
    if not dirs_path.exists():
        # 方向表决定每列按多头还是空头计分，必须先冻结再抽事件；缺文件就停，
        # 不许退回到「运行时现算」——那样口径哈希里看不到它，换一次词表就跑出不同结论。
        raise SystemExit("先跑 --stage directions 生成 docs/indicator-directions.json")
    dirs_sha = hashlib.sha256(dirs_path.read_bytes()).hexdigest()[:16]
    horizons = spec["horizons_bars"]
    keep, expect_keys = keep_index()
    svc = MarketDataService(db)
    out: dict[str, dict] = {}
    total_t0 = time.time()
    for interval in (args.intervals or spec["intervals"]):
        symbols = list(uni["universe"][interval])
        if args.limit_symbols:
            symbols = symbols[: args.limit_symbols]
        layout = CacheLayout(root=CACHE, scope_sha=spec["sha"], interval=interval)
        print(f"[events ] {args.scope}/{interval}: {len(symbols)} 币 "
              f"窗口={spec['window_days'][interval]}d 缓存={layout.dir.relative_to(ROOT)}",
              flush=True)
        res = run_interval(db, svc, interval, symbols, spec["window_days"][interval], layout,
                           workers=args.workers, horizons=horizons,
                           resume=not args.no_resume, progress_every=args.progress_every,
                           keep_names=keep, expect_keys=expect_keys)
        out[interval] = res
        # 每个周期跑完就落一次盘：全量要跑几小时，中途崩了也要留下已完成证据
        (DOCS / f"fit-events-{args.scope}.json").write_text(
            json.dumps({"generated_at": stamp(), "scope": args.scope, "sha": spec["sha"],
                        "directions_sha": dirs_sha,
                        "intervals": out, "elapsed_s": round(time.time() - total_t0, 1)},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"[events ] 合计 {time.time() - total_t0:.0f}s", flush=True)
    return 0


# --------------------------------------------------------------------- stats


# 归堆阈值写在产物里，报告才能说清「几算同堆」——阈值散在函数默认参数里，换一次数字
# 报告里的表格就成了无来源的断言。
REDUNDANCY_R_MIN = 0.8


def redundancy_pairs(matrix: np.ndarray, covered: np.ndarray, keys: list[str], hi: int,
                     cand_idx: list[int], threshold: float = REDUNDANCY_R_MIN) -> list[dict]:
    """按「逐币平均超额」的币间相关给候选列归堆，识别同一个信号的重复列。

    这不是逐事件重合率：聚合缓存里没有保留逐事件掩码，为了算重合率再跑一遍事件抽取
    不划算。币间相关回答的是「这些列在哪些币上有效这件事是否一致」，作为冗余的第一层
    筛子够用；报告里会写明它是近似，逐事件重合率留给后续小样本复核。

    坑（2026-11-13 修）：`covered` 已经是 (列, 币) 两维，之前写成 `cov[:, None]` 让条件
    张量炸成 (列, 列, 币)，`np.where` 把均值广播成三维，取出来的 `mean[a]` 是「a 对每个
    列」的切片，相关值全是自相关的复读——1h 一轮 70 列报了 1858 对、并成 1 堆，等于没归堆。
    修完用同一份缓存重跑：形状对的时候 r 中位数量级在 0.03，≥0.8 的是少数对。归堆结果只有
    在形状对的时候才有意义，所以配了单测（tests/test_fit_redundancy.py），别只信肉眼看过的表格。
    """
    if len(cand_idx) < 2:
        return []
    n = matrix[cand_idx, :, hi, fe.FI["n"]]
    x = matrix[cand_idx, :, hi, fe.FI["sum_x"]]
    cov = covered[cand_idx]
    if not (n.shape == x.shape == cov.shape):
        raise AssertionError(f"冗余归堆形状不一致: {n.shape} {x.shape} {cov.shape}")
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(cov & (n > 0), x / np.maximum(n, 1.0), np.nan)
    out: list[dict] = []
    for a in range(len(cand_idx)):
        for b in range(a + 1, len(cand_idx)):
            ok = np.isfinite(mean[a]) & np.isfinite(mean[b])
            if int(ok.sum()) < 5:
                continue
            va, vb = mean[a][ok], mean[b][ok]
            if va.std() <= 0 or vb.std() <= 0:
                continue
            r = float(np.corrcoef(va, vb)[0, 1])
            if r >= threshold:
                out.append({
                    "a": keys[cand_idx[a]], "b": keys[cand_idx[b]], "r": round(r, 4),
                    "symbols": int(ok.sum()),
                    "mean_a_pct": round(float(va.mean()) * 100, 4),
                    "mean_b_pct": round(float(vb.mean()) * 100, 4),
                })
    return out


def stage_stats(db, args) -> int:
    spec, uni = load_prep(args.scope)
    horizons = spec["horizons_bars"]
    cfg = GateConfig(n_boot=int(spec.get("n_boot") or GateConfig().n_boot))
    per_interval: dict[str, dict] = {}
    for interval in (args.intervals or spec["intervals"]):
        layout = CacheLayout(root=CACHE, scope_sha=spec["sha"], interval=interval)
        pack = evaluator_for(uni, interval, layout, horizons)
        if pack is None:
            print(f"[stats  ] {interval}: 无可用缓存，跳过", flush=True)
            continue
        ev, keys, syms, covered, matrix, _ = pack
        scopes = stat_scopes(db, uni, interval, args.limit_symbols or None)
        atr = (uni.get("atr_pct") or {}).get(interval) or {}
        rows = apply_fdr(ev.evaluate(horizons, scopes, atr_pct=atr))
        # 分层结论必须相对整池收缩后再看：小样本格子单独报出来的均值不可信
        pooled = {(r["key"], r["horizon_bars"], r["benchmark"]): r
                  for r in rows if r["scope"] == "all"}
        for r in rows:
            r["verdict_cell"] = verdict(r, cell=(r["scope"] != "all"))
            base = pooled.get((r["key"], r["horizon_bars"], r["benchmark"]))
            if r["scope"] != "all" and base is not None:
                r.update(shrink(r, base, k=cfg.shrink_k))
        pos = {k: i for i, k in enumerate(keys)}
        cand_by_h: dict[int, list[int]] = {}
        for r in rows:
            if r["scope"] == "all" and r["benchmark"] == "unconditional" \
                    and r["verdict"] in PROMOTABLE:
                cand_by_h.setdefault(horizons.index(r["horizon_bars"]), []).append(pos[r["key"]])
        # 键是**预测长度（bar 数）**，不是候选下标：产物里要能直接读出「哪个 h 的冗余对」。
        red = {str(horizons[hi]): redundancy_pairs(matrix, covered, keys, hi, sorted(set(idx)))
               for hi, idx in cand_by_h.items()}
        counts: dict[str, dict[str, int]] = {}
        for r in rows:
            bucket = counts.setdefault(r["scope"], {})
            bucket[r["verdict"]] = bucket.get(r["verdict"], 0) + 1
        # 落盘只留两样：每个 (列, 池) 的最强格（矩阵的原子，885×8 条，只带判定要用的字段）
        # +「说得出口」的列的全部明细。判定分布按**全量** rows 统计后单独存一份，所以砍掉
        # 的只是重复数字，不是结论——四周期全量 19 万行原样落 json 是上百 MB，git 放不动。
        keep = {r["key"] for r in rows
                if (r.get("verdict_cell") or r["verdict"]) in PROMOTABLE}
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            groups[(r["key"], r["scope"])].append(r)
        slim = [r for r in rows if r["key"] in keep]
        slim += [{k: val for k, val in (best_cell(rs) or {}).items() if k in ATOM_FIELDS}
                 for g, rs in groups.items() if g[0] not in keep]
        per_interval[interval] = {
            "scope_sizes": {k: len(v) for k, v in scopes.items()},
            "columns": len(keys), "symbols": len(syms),
            "rows": slim, "rows_total": len(rows), "rows_kept": len(slim),
            "columns_promotable": len(keep),
            "verdict_counts": counts, "redundancy": red,
        }
        print(f"[stats  ] {interval}: {len(keys)} 列 × {len(syms)} 币，{len(rows)} 行"
              f"（落盘 {len(slim)}）；all/candidate={counts.get('all', {}).get('candidate', 0)} "
              f"exploratory={counts.get('all', {}).get('exploratory', 0)} "
              f"underpowered_blocks={counts.get('all', {}).get('underpowered_blocks', 0)}",
              flush=True)
    payload = {
        "generated_at": stamp(), "scope": args.scope, "sha": spec["sha"],
        "horizons_bars": horizons,
        "config": {
            "n_boot": cfg.n_boot, "q_alpha": cfg.q_alpha,
            "interaction_alpha": cfg.interaction_alpha,
            "segment_sign_min": cfg.seg_sign_min, "symbol_sign_min": cfg.sym_sign_min,
            "cell_events_min": cfg.cell_events_min, "cell_symbols_min": cfg.cell_symbols_min,
            "blocks_for_candidate": cfg.blocks_for_candidate,
            "redundancy_r_min": REDUNDANCY_R_MIN,
            "shrink_k": cfg.shrink_k, "round_trip": 0.0012,
        },
        "intervals": per_interval,
    }
    out = DOCS / f"indicator-fit-stats-{args.scope}{getattr(args, 'out_suffix', '')}.json"
    # 紧凑写（实测 1h 一轮：indent=2 22.2 MB → 紧凑 13.2 MB，四周期约 50 MB）。这份产物是
    # 「数据」不是「读物」：人读的是 indicator-fit-matrix.md 与 .html，数字全从这里来。
    # 行数已按「每列每池最强格 + 说得出口的列的全部明细」裁剪，判定分布仍按全量行统计。
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    print(f"[stats  ] -> {out.name}（{out.stat().st_size / 1e6:.1f} MB）", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="A", help="fit_prep 里冻结的轮次：A / B")
    ap.add_argument("--stage", default="all",
                    help="directions / probe / events / stats / all")
    ap.add_argument("--interval", default="", help="只处理某些周期（逗号分隔）")
    ap.add_argument("--out-suffix", default="",
                    help="统计产物文件名后缀：调试时用 --out-suffix=-dev 别盖掉正式产物。"
                         "必须带等号——裸写 `-dev` 会被 argparse 当成一个新选项而报"
                         "「expected one argument」")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit-symbols", type=int, default=0)
    ap.add_argument("--probe-symbols", type=int, default=3)
    ap.add_argument("--dir-symbol", default="BTCUSDT", help="枚举布尔列用的探针币")
    ap.add_argument("--dir-bars", type=int, default=1500, help="探针帧的 1h bar 数")
    ap.add_argument("--progress-every", type=int, default=2)
    ap.add_argument("--no-resume", action="store_true", help="忽略已有事件缓存，重算")
    args = ap.parse_args()
    args.intervals = [s.strip() for s in args.interval.split(",") if s.strip()] or None

    engine = make_engine(read_only=True)  # 本脚本只读 klines；写产物只落 docs/ 与缓存目录
    db = sessionmaker(bind=engine)()
    if args.stage == "directions":
        return stage_directions(db, args)
    if args.stage in ("probe", "all"):
        stage_probe(db, args)
    if args.stage in ("events", "all"):
        stage_events(db, args)
    if args.stage in ("stats", "all"):
        stage_stats(db, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
