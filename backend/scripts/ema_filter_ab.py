r"""EMA 方向过滤的受控 A/B 池化判定（手册 Gate 3–4）。

只改 `ema_trend_fast` / `ema_trend_slow` 一个变量：关闭过滤 (0,0) 为基线，
若干开启组合为实验臂，同一币池、同一窗口、同一撮合成本、同一策略默认参数。

设计要点
--------
* 4 个**不重叠** 90 天窗口，墙钟对齐；寻优只用最近 180 天（≈W1+W2），
  因此 W3+W4 是未被寻优读取过的样本外，单独给一份池化读数（H5）。
* 每个 (策略, 币, 窗口) 作为一个进程分片，帧只建一次（H9）；EMA 在含预热的
  完整帧上计算 signals，再裁到窗口段撮合，避免慢线在窗口起点重新播种。
* 逐片先 `aggregate_positions` 再取 `return_pct`，避免跨币 position_id 撞车。
* 池化 bootstrap 95% CI（min_trades=30）、同币配对差、20 种子方向翻转随机基线、
  BH-FDR 校正、时间分段同号数。
* 成本全局统一（fee_rate / slippage 从 app_state 读），不允许臂级覆盖（H3）。

输出 `output/ema-filter-ab.json`（原始读数），报告由 docs/ 成对文件引用。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.backtest.execution import randomize_directions, simulate  # noqa: E402
from app.backtest.significance import bootstrap_expectancy_ci, percentile_rank  # noqa: E402
from app.backtest.trades import aggregate_positions  # noqa: E402
from app.db import SessionLocal, make_engine  # noqa: E402
from app.ml.dataset import build_dataset  # noqa: E402
from app.models import SymbolPool  # noqa: E402
from app.state import all_state  # noqa: E402
from app.strategy import specs  # noqa: F401  导入即注册
from app.strategy.pools import pool_symbols  # noqa: E402
from app.strategy.registry import get_spec  # noqa: E402
from app.strategy.rules import evaluate_signals, resolve_params  # noqa: E402
from app.utils import interval_ms, now_ms  # noqa: E402

# 1h 有交易的两条策略；其余 6 条或 1h 上 no_trades，或走专用 broker（direction≡0），
# 通用 EMA 门控对它们不产生可判定样本，见报告「币池与适用性」。
STRATEGIES = ("supertrend_current", "flow_imbalance_long")

ARMS = {
    "off": {"ema_trend_fast": 0, "ema_trend_slow": 0},
    "e12_50": {"ema_trend_fast": 12, "ema_trend_slow": 50},    # 首轮 12/25 抽样最优
    "e50_100": {"ema_trend_fast": 50, "ema_trend_slow": 100},  # 全网格最常选中
    "e20_26": {"ema_trend_fast": 20, "ema_trend_slow": 26},    # 全网格第三
}
ON_ARMS = tuple(k for k in ARMS if k != "off")

MIN_POOLED_TRADES = 30
WINDOW_DAYS = 90
N_WINDOWS = 4
N_SEEDS = 20
BASE_SEED = 20261025
DAY_MS = 86_400_000

_SESSION_FACTORY = None


def _session():
    """每个 worker 进程只建一次只读 engine（H9）。"""
    global _SESSION_FACTORY
    if _SESSION_FACTORY is None:
        from sqlalchemy.orm import sessionmaker
        _SESSION_FACTORY = sessionmaker(
            bind=make_engine(read_only=True, null_pool=True),
            autoflush=False, expire_on_commit=False, future=True,
        )
    return _SESSION_FACTORY()


def _returns(trades: list[dict]) -> list[float]:
    agg = aggregate_positions(trades)
    return [float(t["return_pct"]) for t in agg
            if np.isfinite(float(t.get("return_pct", 0.0)))]


def _worker(task: dict) -> dict:
    code = task["code"]
    symbol = task["symbol"]
    wkey = task["window"]
    arms = task["arms"]
    fee, slip, initial = task["fee"], task["slippage"], task["initial"]
    interval = task["interval"]
    end_ms = task["end_ms"]

    db = _session()
    try:
        full = build_dataset(
            db, symbol, interval, horizon=5,
            days=WINDOW_DAYS + math.ceil(task["warmup_days"]) + 1,
            with_label=False, end_ms=end_ms,
        )
    except Exception as exc:
        return {"code": code, "symbol": symbol, "window": wkey,
                "status": "skipped", "error": str(exc)}
    finally:
        db.close()

    cutoff = pd.Timestamp(end_ms - WINDOW_DAYS * DAY_MS, unit="ms", tz="UTC")
    # EMA 在含预热的完整帧上计算，随后只裁撮合窗口
    segment = full[full.index >= cutoff]
    span = (segment.index[-1] - segment.index[0]).days if len(segment) else 0
    if len(segment) < 500 or span < WINDOW_DAYS - 10:
        return {"code": code, "symbol": symbol, "window": wkey,
                "status": "skipped", "error": f"窗口内仅 {len(segment)} 根 / {span} 天",
                "rows": len(segment)}

    out: dict[str, list[float]] = {}
    signal_bars: dict[str, int] = {}
    try:
        for name, params in arms.items():
            signals = evaluate_signals(full, code, params)
            sig_w = signals.reindex(segment.index)
            _, trades = simulate(segment, sig_w, params, initial, fee, slip, interval)
            out[name] = _returns(trades)
            signal_bars[name] = int(sig_w["direction"].ne(0).sum())
        base_signals = evaluate_signals(full, code, arms["off"])
        base_sig_w = base_signals.reindex(segment.index)
        # 保留每个种子的逐笔收益：主进程按种子把所有币池化成 20 个分布样本（H6），
        # 而不是把「每币 × 每种子」当成几百个独立样本。
        random_returns = []
        for offset in range(N_SEEDS):
            randomized = randomize_directions(
                base_sig_w, seed=BASE_SEED + offset * 7919
            )
            _, rand_trades = simulate(
                segment, randomized, arms["off"], initial, fee, slip, interval
            )
            random_returns.append(_returns(rand_trades))
    except Exception as exc:
        return {"code": code, "symbol": symbol, "window": wkey,
                "status": "skipped", "error": f"评估失败: {exc}"}

    return {
        "code": code, "symbol": symbol, "window": wkey, "status": "ok",
        "rows": len(segment),
        "start": str(segment.index[0]), "end": str(segment.index[-1]),
        "returns": out, "signal_bars": signal_bars,
        "random_returns": random_returns,
    }


def _ci(returns: list[float], min_trades: int = MIN_POOLED_TRADES) -> dict:
    """把已逐片聚合的 return_pct 包成伪 trades，复用项目唯一判据实现。"""
    pseudo = [{"return_pct": float(r)} for r in returns]
    return bootstrap_expectancy_ci(pseudo, min_trades=min_trades)


def _bh_q(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q_sorted = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0.0, 1.0)
    q = np.empty(n)
    q[order] = q_sorted
    return q.tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description="EMA 方向过滤受控 A/B 池化判定")
    ap.add_argument("--pool", default="major")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="只跑前 N 个币（冒烟测试用，0=全池）")
    ap.add_argument("--out", default="output/ema-filter-ab")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        state = all_state(db)
        fee = float(state["fee_rate"])
        slip = float(state["slippage"])
        initial = float(state.get("initial_capital", 10000.0))
        pool = db.query(SymbolPool).filter(SymbolPool.code == args.pool).first()
        if pool is None:
            print(f"未知币种池: {args.pool}")
            return 2
        symbols = pool_symbols(db, pool)
        if args.limit:
            symbols = symbols[:args.limit]
    finally:
        db.close()

    step = interval_ms(args.interval)
    last = now_ms()
    windows = {
        f"W{i + 1}": last - i * WINDOW_DAYS * DAY_MS
        for i in range(N_WINDOWS)
    }

    arms_by_code: dict[str, dict] = {}
    tasks = []
    for code in STRATEGIES:
        spec = get_spec(code)
        base = resolve_params(code, state, {}, interval=args.interval)
        warmup = float(spec.lookback(base)) * step / DAY_MS
        arms_by_code[code] = {
            name: resolve_params(code, state, dict(over), interval=args.interval)
            for name, over in ARMS.items()
        }
        for symbol in symbols:
            for wkey, end_ms in windows.items():
                tasks.append({
                    "code": code, "symbol": symbol, "window": wkey,
                    "arms": arms_by_code[code], "fee": fee, "slippage": slip,
                    "initial": initial, "interval": args.interval,
                    "end_ms": end_ms, "warmup_days": warmup,
                })

    print(f"{len(tasks)} 个分片（{len(STRATEGIES)} 策略 × {len(symbols)} 币 × "
          f"{N_WINDOWS} 窗口），{args.workers} 进程；成本 fee={fee} slip={slip}")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool_ex:
        futures = [pool_ex.submit(_worker, t) for t in tasks]
        for done, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if done % 20 == 0 or done == len(futures):
                print(f"\r  {done}/{len(futures)}", end="", flush=True)
    print()

    ok = [r for r in rows if r["status"] == "ok"]
    skipped = [{"code": r["code"], "symbol": r["symbol"], "window": r["window"],
                "reason": r.get("error", "")} for r in rows if r["status"] != "ok"]
    print(f"完成 {len(ok)} 片，跳过 {len(skipped)} 片")

    groups = {}
    for code in STRATEGIES:
        for name in ARMS:
            for scope, wkeys in (("all", list(windows)), ("oos_only", ["W3", "W4"])):
                pooled = [r for x in ok if x["code"] == code and x["window"] in wkeys
                          for r in x["returns"][name]]
                groups[f"{code}|{name}|{scope}"] = {
                    "strategy": code, "arm": name, "scope": scope,
                    "windows": wkeys, "trades": len(pooled),
                    "ci": _ci(pooled),
                }

    # 同币配对差：每个 (策略, 币, 窗口) 上实验臂与基线每笔期望之差
    paired = {}
    for code in STRATEGIES:
        for name in ON_ARMS:
            diffs, per_window = [], {}
            for x in ok:
                if x["code"] != code:
                    continue
                a, b = x["returns"]["off"], x["returns"][name]
                if not a or not b:
                    continue
                d = float(np.mean(b) - np.mean(a))
                diffs.append(d)
                per_window.setdefault(x["window"], []).append(d)
            paired[f"{code}|{name}"] = {
                "strategy": code, "arm": name, "pairs": len(diffs),
                "ci": _ci(diffs, min_trades=MIN_POOLED_TRADES),
                "per_window": {
                    w: {"n": len(v), "mean_pct": round(float(np.mean(v)), 4)}
                    for w, v in sorted(per_window.items())
                },
                "positive_windows": sum(
                    1 for v in per_window.values() if np.mean(v) > 0
                ),
                "window_count": len(per_window),
            }

    # BH-FDR：配对差 bootstrap 的 positive_share 近似双侧 p
    keys = list(paired)
    pvals = []
    for k in keys:
        ps = float(paired[k]["ci"].get("positive_share", 0.5))
        pvals.append(float(np.clip(2 * min(ps, 1 - ps), 1e-6, 1.0)))
    qvals = _bh_q(pvals)
    for k, p, q in zip(keys, pvals, qvals):
        paired[k]["p_two_sided_approx"] = round(p, 4)
        paired[k]["fdr_q"] = round(q, 4)
        paired[k]["fdr_pass"] = bool(q < 0.10 and
                                     paired[k]["ci"].get("verdict") == "positive")

    # 随机基线：每策略每窗口把 20 个种子的「每笔期望」汇总成分布
    random_baseline = {}
    for code in STRATEGIES:
        for wkey in windows:
            pieces = [x for x in ok if x["code"] == code and x["window"] == wkey]
            samples = []
            for seed_idx in range(N_SEEDS):
                pooled = [r for x in pieces for r in x["random_returns"][seed_idx]]
                samples.append(float(np.mean(pooled)) if pooled else 0.0)
            pooled_off = [r for x in pieces for r in x["returns"]["off"]]
            observed = float(np.mean(pooled_off)) if pooled_off else 0.0
            random_baseline[f"{code}|{wkey}"] = {
                "strategy": code, "window": wkey, "n_seeds": N_SEEDS,
                "n_samples": len(samples),
                "mean_pct": round(float(np.mean(samples)), 4) if samples else None,
                "p05_pct": round(float(np.percentile(samples, 5)), 4) if samples else None,
                "p95_pct": round(float(np.percentile(samples, 95)), 4) if samples else None,
                "off_observed_pct": round(observed, 4),
                "off_percentile": round(percentile_rank(observed, samples), 4),
            }

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "topic": "EMA 快慢线方向过滤 · 全策略通用门控 · 受控 A/B",
        "interval": args.interval,
        "window_days": WINDOW_DAYS,
        "windows": {k: str(pd.Timestamp(v, unit="ms", tz="UTC"))
                    for k, v in windows.items()},
        "seek_window": "W1+W2（最近 180 天，寻优所用）",
        "oos_window": "W3+W4（寻优未读取的样本外）",
        "universe": {"pool": args.pool, "symbols": symbols,
                     "skipped": skipped, "rows_ok": len(ok)},
        "costs": {"fee_rate": fee, "slippage": slip,
                  "round_trip": round(2 * (fee + slip), 6),
                  "basis": "notional", "source": "app_state（臂级禁止覆盖）"},
        "arms": {k: dict(v) for k, v in ARMS.items()},
        "strategies": list(STRATEGIES),
        "min_pooled_trades": MIN_POOLED_TRADES,
        "random_baseline_seeds": N_SEEDS,
        "groups": groups,
        "paired_difference": paired,
        "random_baseline": random_baseline,
        "windows_detail": {
            w: next(({"start": r["start"], "end": r["end"], "rows": r["rows"]}
                     for r in ok if r["window"] == w), None)
            for w in windows
        },
        "limitations": [
            "W1/W2 与寻优窗口重叠，只有 W3/W4 是未被选参读取的样本外；重叠只会让读数偏乐观。",
            "未模拟资金费率、盘口深度、强平、分批建仓；撮合假定下一根开盘及时成交。",
            "EMA 为收盘价衍生，落在 H2 止损线禁区内；本轮为按用户明确要求的受控证伪，不构成新特征投入。",
            "BNBUSDT 1h 数据仅约 210 天，较早窗口因样本不足被剔除并列 skip 清单。",
        ],
    }

    out = Path(args.out).with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 池化每笔期望（全窗口）===")
    for code in STRATEGIES:
        for name in ARMS:
            g = groups[f"{code}|{name}|all"]
            c = g["ci"]
            print(f"  {code:22s} {name:8s} n={g['trades']:5d} "
                  f"mean={c.get('mean_return_pct', 0):8.4f}% "
                  f"CI=[{c.get('ci_low_pct', 0):7.4f},{c.get('ci_high_pct', 0):7.4f}] "
                  f"{c.get('verdict')}")
    print("\n=== 同币配对差（实验臂 - 关闭过滤）===")
    for k, v in paired.items():
        c = v["ci"]
        print(f"  {k:36s} pairs={v['pairs']:3d} diff={c.get('mean_return_pct', 0):8.4f}% "
              f"CI=[{c.get('ci_low_pct', 0):7.4f},{c.get('ci_high_pct', 0):7.4f}] "
              f"q={v['fdr_q']:.4f} {c.get('verdict')}")
    print(f"\n已写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
