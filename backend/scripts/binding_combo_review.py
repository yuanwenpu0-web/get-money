r"""已启用「策略 × 币种池」绑定的逐个回测与组合级参数寻优。

为什么另起一个脚本
------------------
`optimize_all.py` 的任务单位是 (币, 周期)，选参在单币内完成——它回答的是「每个币
各自该用哪套参数」。而绑定层只有**一份** `params` 对整池生效，所以「这套绑定该用
哪套参数」必须在池层面聚合选参。此外两条研究池策略 `direction≡0`、无止损止盈，
通用撮合根本跑不出成交（`simulate` 只在 `signal.direction` 非零时开仓），必须走
专用持有期回放。

口径
----
- 成本：全局 `fee_rate=0.0004` + `slippage=0.0002`，开平双边 0.12%（研究池沿用其
  模拟盘口径 0.04%/边 + 0.02% 滑点，同为双边 0.12%）。
- 收益口径：**每笔净收益 / 名义本金**。通用撮合的 `return_pct` 是保证金口径，
  这里按 `pnl/(entry*size)` 换算，避免与固定 200U 名义的研究池混比。
- 切分：训练 60% / 验证 20% / 测试 20%，按**墙钟**切（不是按 bar 数），
  跨段成交一律清除；测试段只在选出冠军后读一次。
- 判定：池化 bootstrap 95% CI，`MIN_POOLED_TRADES=30`。

用法
----
    .\.venv\Scripts\python.exe backend/scripts/binding_combo_review.py --days 180 --workers 6
    .\.venv\Scripts\python.exe backend/scripts/binding_combo_review.py --only A,B --days 240
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.db import SessionLocal, make_engine  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.research import combo_engine as ce  # noqa: E402
from app.research.combo_config import COMBOS, expand_grid  # noqa: E402
from app.state import all_state  # noqa: E402
from app.strategy.bindings import active_bindings  # noqa: E402
from app.strategy.registry import get_spec  # noqa: E402
from app.strategy.rules import resolve_params  # noqa: E402
from app.utils import interval_ms, now_ms  # noqa: E402
from sqlalchemy import text  # noqa: E402

_DB = None


def _db():
    global _DB
    if _DB is None:
        _DB = sessionmaker(bind=make_engine(read_only=True, null_pool=True))()
    return _DB


def _window(days: int) -> tuple[int, int]:
    end = now_ms()
    return end - days * 86_400_000, end


def _bounds(start: int, end: int) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    span = end - start
    marks = [start, start + 0.6 * span, start + 0.8 * span, end]
    names = ("train", "valid", "test")
    stamps = [pd.Timestamp(int(ms), unit="ms", tz="UTC") for ms in marks]
    return {name: (stamps[i], stamps[i + 1]) for i, name in enumerate(names)}


# ------------------------------------------------------------------ worker


def _symbol_task(task: dict) -> dict:
    symbol, interval = task["symbol"], task["interval"]
    start, end = task["window"]
    started = time.perf_counter()
    try:
        frame = MarketDataService(_db()).load_frame(
            symbol, interval, start_ms=start - task["warmup_ms"], end_ms=end
        )
        if len(frame) < task["min_bars"]:
            return {"symbol": symbol, "error": f"样本不足 {len(frame)} 根", "trades": []}
        first = int(frame.index[0].timestamp() * 1000)
        if first > start + 2 * 86_400_000:
            return {"symbol": symbol, "error": f"历史不足，起于 {str(frame.index[0])[:10]}", "trades": []}
        out: list[list] = []
        if task["engine"] == "research":
            for idx, params in enumerate(task["combos"]):
                hold = int(params["research_hold_hours"] * 3_600_000 // interval_ms(interval))
                for trade in ce.research_trades(frame, params["channel_kind"], hold):
                    out.append([idx, int(trade["t0"].timestamp() * 1000),
                                int(trade["t1"].timestamp() * 1000), trade["ret"]])
        else:
            cache: dict = {}
            keys = tuple(task["prepare_keys"])
            for idx, params in enumerate(task["combos"]):
                prepared = ce.prepare_cached(cache, frame, task["code"], params, keys)
                try:
                    trades = ce.generic_trades(
                        prepared, task["code"], params, fee=task["fee"],
                        slippage=task["slippage"], initial=task["initial"], interval=interval,
                    )
                except Exception:  # 该组合参数在本币上不可跑，跳过而不毁整轮
                    continue
                for trade in trades:
                    out.append([idx, int(trade["t0"].timestamp() * 1000),
                                int(trade["t1"].timestamp() * 1000), trade["ret"]])
        return {"symbol": symbol, "error": "", "trades": out, "bars": len(frame),
                "start": str(frame.index[0])[:10], "end": str(frame.index[-1])[:10],
                "buy_hold_pct": float((frame["close"].iloc[-1] / frame["close"].iloc[0] - 1) * 100),
                "elapsed": round(time.perf_counter() - started, 1)}
    except Exception as exc:  # noqa: BLE001
        return {"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"[:200], "trades": []}


# ------------------------------------------------------------------ 主流程


def build_combos(combo: dict, state: dict) -> tuple[list[dict], list[dict]]:
    """返回 (给 worker 的参数列表, 记录用标签)。generic 组合走 resolve_params 校验。"""
    grid = expand_grid(combo["grid"])
    if combo["engine"] == "research":
        return grid, [dict(item) for item in grid]
    resolved = []
    for item in grid:
        try:
            params = resolve_params(combo["code"], state, dict(item), interval=combo["interval"])
        except ValueError:
            continue
        resolved.append(params)
    return resolved, resolved


def run_combo(combo: dict, symbols: list[str], args, state: dict, extra_skip: list[dict]) -> dict:
    start, end = _window(args.days)
    combos, labels = build_combos(combo, state)
    if not combos:
        return {"key": combo["key"], "error": "搜索空间为空"}
    step = interval_ms(combo["interval"])
    warmup_ms = 420 * step
    prepare_keys = tuple(get_spec(combo["code"]).prepare_keys) if combo["engine"] == "generic" else ()
    tasks = [{
        "symbol": symbol, "interval": combo["interval"], "window": (start, end),
        "warmup_ms": warmup_ms, "engine": combo["engine"], "code": combo["code"],
        "combos": combos, "prepare_keys": prepare_keys,
        "fee": float(state["fee_rate"]), "slippage": float(state["slippage"]),
        "initial": float(state.get("initial_capital", 10000.0)), "min_bars": 400,
    } for symbol in symbols]
    print(f"[{combo['key']}] {combo['code']} × {combo['pool']} × {combo['interval']}"
          f" · {len(symbols)} 币 × {len(combos)} 组参数", flush=True)
    rows = []
    workers = max(1, min(args.workers, len(tasks)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(_symbol_task, tasks):
            rows.append(row)
    result = summarize(combo, rows, labels, _bounds(start, end))
    result["skipped"] = extra_skip + result["skipped"]
    return result


def summarize(combo: dict, rows: list[dict], labels: list[dict], bounds: dict) -> dict:
    ok = [row for row in rows if not row.get("error")]
    skipped = [{"symbol": row["symbol"], "reason": row["error"]} for row in rows if row.get("error")]
    buckets = [{name: [] for name in bounds} for _ in labels]
    for row in ok:
        for idx, t0, t1, ret in row["trades"]:
            for name, (lo, hi) in bounds.items():
                lo_ms, hi_ms = int(lo.timestamp() * 1000), int(hi.timestamp() * 1000)
                if lo_ms <= t0 and t1 < hi_ms:
                    buckets[idx][name].append(ret)
                    break
    table = []
    for idx, label in enumerate(labels):
        seg = buckets[idx]
        table.append({
            "params": {k: label[k] for k in sorted(label)
                       if k in combo["grid"] or k in combo["current"]},
            "train": {"trades": len(seg["train"]), "mean": ce.mean_or_none(seg["train"])},
            "valid": {**ce.pooled(seg["valid"])},
            "test": {**ce.pooled(seg["test"])},
        })
    best = ce.pick([{"train": buckets[i]["train"], "idx": i} for i in range(len(labels))])
    best_row = table[best["idx"]] if best else None
    current_idx = _match_current(combo, labels)
    result = {
        "key": combo["key"], "label": combo["label"], "strategy": combo["code"],
        "pool": combo["pool"], "interval": combo["interval"], "engine": combo["engine"],
        "symbols": len(ok), "skipped": skipped,
        "symbol_detail": [{k: v for k, v in row.items() if k != "trades"} for row in ok],
        "grid_size": len(labels), "table": table,
        "best": best_row, "current": table[current_idx] if current_idx is not None else None,
        "baseline_best": {}, "baseline_current": {},
    }
    # 随机基线必须同时给「当前参数」和「寻优冠军」：只说冠军跑赢随机，无法判断
    # 改动本身有没有价值——线上跑的是当前参数，不是冠军。
    if best_row:
        result["baseline_best"] = _baseline(combo, ok, labels[best["idx"]], bounds, best_row)
    if current_idx is not None:
        result["baseline_current"] = _baseline(combo, ok, labels[current_idx], bounds, table[current_idx])
    return result


def _match_current(combo: dict, labels: list[dict]) -> int | None:
    for idx, label in enumerate(labels):
        if all(label.get(k) == v for k, v in combo["current"].items()):
            return idx
    return None


def _baseline(combo: dict, rows: list[dict], label: dict, bounds: dict, best_row: dict) -> dict:
    """随机基线：research 用「随机入场、同样持有期」，generic 记方向翻转口径。

    研究池是**只做多**，方向翻转等于变成做空，那是另一个 beta，不是同一策略的
    对照。真正要问的是「不定时、随便什么时候买、持有同样久」能不能拿到一样的
    每笔期望——所以基线建在入场时点上，并且必须**按池内所有币池化**，只取一个
    币的随机分布会把个别微盘币的波动当成基线。
    """
    actual = float(best_row["valid"]["mean"] or 0.0)
    if combo["engine"] != "research":
        return {"kind": "direction_flip",
                "note": "generic 组合的方向翻转基线：负 verdict 已先失败，翻转后为正只说明方向反了，不算 edge",
                "actual_mean_pct": actual, "percentile": None,
                "gross_estimate_pct": round(actual + 0.12, 4)}
    step = interval_ms(combo["interval"])
    hold = int(label["research_hold_hours"] * 3_600_000 // step)
    per_symbol = max(1, round(best_row["valid"]["trades"] / max(1, len(rows))))
    samples, used = ce.research_baseline(
        MarketDataService(_db()), [row["symbol"] for row in rows], combo["interval"], hold,
        int(bounds["valid"][0].timestamp() * 1000), int(bounds["valid"][1].timestamp() * 1000),
        per_symbol,
    )
    return {"kind": "random_entry", "seeds": 20, "symbols": used,
            "trades_per_seed": per_symbol * used,
            "mean_pct": round(float(np.mean(samples)), 4),
            "p05_pct": round(float(np.percentile(samples, 5)), 4),
            "p95_pct": round(float(np.percentile(samples, 95)), 4),
            "actual_mean_pct": actual,
            "percentile": ce.percentile_rank(actual, samples)}


def main() -> int:
    ap = argparse.ArgumentParser(description="已启用绑定的组合级寻优与判定")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default="", help="只跑指定组合，如 A,B")
    ap.add_argument("--max-symbols", type=int, default=0, help="每池最多取多少币（0=不限）")
    ap.add_argument("--out", default=str(ROOT / "output" / "binding-combo-review"))
    args = ap.parse_args()

    db = SessionLocal()
    state = all_state(db)
    pairs, _ = active_bindings(db)
    enabled = {(p.strategy, p.pool, p.interval): p for p in pairs}
    members = {}
    for combo in COMBOS:
        bound = enabled.get((combo["code"], combo["pool"], combo["interval"]))
        members[combo["key"]] = list(bound.symbols) if bound else []
    start, end = _window(args.days)

    picked = [c for c in COMBOS if not args.only or c["key"] in args.only.split(",")]
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "days": args.days, "costs": {"fee_rate": float(state["fee_rate"]),
                                            "slippage": float(state["slippage"]),
                                            "round_trip": 0.0012, "basis": "notional"},
               "split": ce.SPLIT, "combos": []}
    for combo in picked:
        symbols, extra = members[combo["key"]], []
        if combo.get("require_taker"):
            symbols, extra = ce.taker_ready_symbols(db, symbols, start, end)
        symbols = symbols[: args.max_symbols or None]
        if not symbols:
            payload["combos"].append({"key": combo["key"], "error": "绑定未解析出可用币种",
                                      "skipped": extra})
            continue
        payload["combos"].append(run_combo(combo, symbols, args, state, extra))
    db.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n已写出 {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
