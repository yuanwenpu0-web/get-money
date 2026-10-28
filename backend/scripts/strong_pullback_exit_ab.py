"""strong_pullback 出场重构 A/B walk-forward 验证（15m，只读）。

对照设计（同一批币、同一批窗口、同一撮合口径）：
- partial ：默认参数（partial_tp_r=2.0，2R 减半 + 余仓保本/追踪/EMA9 软出场）
- fixed   ：param_overrides={"partial_tp_r": 0.0}（旧结构：10R 固定目标）
- trend_follow / mtf_resonance：同窗口控制组（默认参数），避免"只和随机基线比"。

窗口：120 天（训练 90 / 测试 30），步长 30 天互不重叠，终点由 --end 回退 --windows 个。
成交口径：partial 下一笔持仓产生 partial_tp + 末段两条记录；统计 gate 用「持仓级」
（按 entry_time 合并两腿、pnl 求和、return_pct=pnl/Σmargin）的 bootstrap CI。

Gate（写进文档的判定标准）：池化全部窗口×币种的持仓数 ≥30 且持仓级 CI 下界 >0 → 有 edge；
同时要求 partial 相对 fixed 的同窗口收益差中位数 > 0 才算「重构有效」。

用法（只读 trading.db，不启服务）：
    .\\.venv\\Scripts\\python.exe backend/scripts/strong_pullback_exit_ab.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy.orm import Session  # noqa: E402

from app.backtest.runner import run_backtest  # noqa: E402
from app.backtest.significance import bootstrap_expectancy_ci  # noqa: E402
from app.db import make_engine  # noqa: E402

DAY_MS = 86_400_000
VARIANTS = {
    "partial": ("strong_pullback", None),
    "fixed": ("strong_pullback", {"partial_tp_r": 0.0}),
    "trend_follow": ("trend_follow", None),
    "mtf_resonance": ("mtf_resonance", None),
}


def merge_positions(trades: list[dict]) -> list[dict]:
    """把同一 entry_time 的 partial_tp + 末段合并成一条「持仓级」记录。"""
    grouped: dict[str, dict] = {}
    for t in trades:
        key = t["entry_time"]
        slot = grouped.setdefault(key, {"entry_time": key, "pnl": 0.0, "margin": 0.0})
        slot["pnl"] += float(t["pnl"])
        slot["margin"] += float(t["margin"])
    return [
        {**p, "return_pct": p["pnl"] / p["margin"] * 100 if p["margin"] else 0.0}
        for p in grouped.values()
    ]


def eligible_symbols(db: Session, min_days: float, end_ms: int) -> list[str]:
    """15m 覆盖 >= min_days 且数据尾部接近 --end 的币种。

    注意：MarketDataService.stats() 只统计 1m，不能用作 15m 宇宙的遍历集，
    这里直接按 15m 聚合查询。
    """
    from sqlalchemy import func, select

    from app.models import Kline

    rows = db.execute(
        select(Kline.symbol, func.min(Kline.open_time), func.max(Kline.open_time))
        .where(Kline.interval == "15m")
        .group_by(Kline.symbol)
    ).all()
    out = []
    for symbol, first, last in rows:
        if not first or not last:
            continue
        span = (int(last) - int(first)) / DAY_MS
        lag = (end_ms - int(last)) / DAY_MS
        if span >= min_days and lag <= 7:
            out.append(symbol)
    return sorted(out)


def run_one(engine, symbol, strategy, overrides, end_ms, args) -> dict:
    with Session(engine) as db:
        try:
            result = run_backtest(
                db, symbol, args.interval, strategy, algo="rule", days=args.window_days,
                optimize=True, end_ms=end_ms, fee=args.fee, slippage=args.slippage,
                param_overrides=overrides,
            )
        except Exception as exc:
            return {"symbol": symbol, "end_ms": end_ms, "error": str(exc)[:200]}
    m = result["metrics"]
    positions = merge_positions(result.get("trades", []))
    statuses = Counter(t["status"] for t in result.get("trades", []))
    return {
        "symbol": symbol,
        "end_ms": end_ms,
        "return_pct": m["total_return_pct"],
        "max_drawdown_pct": m["max_drawdown_pct"],
        "sharpe": m["sharpe"],
        "trades": m["total_trades"],
        "positions": len(positions),
        "buy_hold_pct": m["buy_hold_return_pct"],
        "random_pct": m["random_baseline"].get("total_return_pct"),
        "position_legs": positions,
        "exits": dict(statuses),
        "warnings": m.get("warnings", []),
    }


def summarize_variant(records: list[dict]) -> dict:
    ok = [r for r in records if "error" not in r]
    legs = [p for r in ok for p in r["position_legs"]]
    returns = [r["return_pct"] for r in ok]
    ci = bootstrap_expectancy_ci([
        {"return_pct": p["return_pct"]} for p in legs
    ])
    exits = Counter()
    for r in ok:
        exits.update(r["exits"])
    return {
        "runs": len(ok),
        "errors": len(records) - len(ok),
        "positions": len(legs),
        "mean_return_pct": round(statistics.mean(returns), 4) if returns else None,
        "positive_runs": sum(1 for x in returns if x > 0),
        "position_ci": ci,
        "exit_distribution": dict(exits),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--end", default=datetime.now(timezone.utc).isoformat())
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--window-days", type=int, default=120)
    ap.add_argument("--step-days", type=int, default=30)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--min-days", type=float, default=205.0)
    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--slippage", type=float, default=0.0002)
    ap.add_argument("--symbols", default="", help="逗号分隔白名单（调试）")
    ap.add_argument("--output", type=Path, default=ROOT / "docs" / "strong-pullback-exit-ab.json")
    args = ap.parse_args()
    end0 = int(datetime.fromisoformat(args.end.replace("Z", "+00:00")).timestamp() * 1000)
    ends = [end0 - i * args.step_days * DAY_MS for i in range(args.windows)]

    engine = make_engine(read_only=True)
    with Session(engine) as db:
        symbols = (
            [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
            or eligible_symbols(db, args.min_days, end0)
        )
    print(f"宇宙: {len(symbols)} 币 | 窗口终点: "
          + ", ".join(datetime.fromtimestamp(e / 1000, timezone.utc).strftime("%m-%d") for e in ends),
          flush=True)

    records: dict[str, list[dict]] = defaultdict(list)
    started = time.time()
    for name, (strategy, overrides) in VARIANTS.items():
        for end_ms in ends:
            for symbol in symbols:
                rec = run_one(engine, symbol, strategy, overrides, end_ms, args)
                rec["variant"] = name
                records[name].append(rec)
                if "error" not in rec:
                    print(json.dumps({
                        "variant": name, "symbol": symbol,
                        "end": datetime.fromtimestamp(end_ms / 1000, timezone.utc).strftime("%Y-%m-%d"),
                        "return_pct": rec["return_pct"], "positions": rec["positions"],
                        "exits": rec["exits"],
                    }, ensure_ascii=False), flush=True)
                else:
                    print(json.dumps({"variant": name, "symbol": symbol, "error": rec["error"]},
                                     ensure_ascii=False), flush=True)

    summary = {name: summarize_variant(recs) for name, recs in records.items()}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": {k: str(v) for k, v in vars(args).items()},
        "symbols": symbols,
        "window_ends": [datetime.fromtimestamp(e / 1000, timezone.utc).isoformat() for e in ends],
        "summary": summary,
        "records": {k: [{kk: vv for kk, vv in r.items() if kk != "position_legs"} for r in v]
                    for k, v in records.items()},
        "elapsed_sec": round(time.time() - started, 1),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n== 汇总 ==", flush=True)
    for name, s in summary.items():
        ci = s["position_ci"]
        print(f"{name}: runs={s['runs']} 持仓={s['positions']} "
              f"均值={s['mean_return_pct']}% CI=[{ci.get('ci_low_pct')}, {ci.get('ci_high_pct')}] "
              f"verdict={ci.get('verdict')}", flush=True)
    print(f"报告: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
