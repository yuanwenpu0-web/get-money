"""Supertrend 反向交易检验：把 `direction` 整列取反，其余一模一样。

只改方向一列——入场时点、评分、仓位、杠杆、`sl_pct`/`tp_pct` 的**绝对距离**都不变，
`simulate` 按新方向镜像推导止损止盈（与 `randomize_directions` 同一机制，只是这里
100% 翻转而不是 50/50 随机）。因此：

- 若「反着买」能盈利 → 说明信号方向系统性反了，是可以直接利用的 bug；
- 若反向也亏 → 说明方向这一维没有信息，正反两侧都在付同一笔成本。

对照口径：与 `supertrend_backtest.py` 完全一致（每币独立 10,000 账户、次根开盘成交、
双边费率与滑点、跨缺口分段复利结转）。每笔期望是 **margin 口径**，可与 baseline
的 -0.4585%（15m）/ -0.1902%（1h）直接比较。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.backtest.execution import simulate  # noqa: E402
from app.backtest.significance import bootstrap_expectancy_ci  # noqa: E402
from app.backtest.trades import aggregate_positions  # noqa: E402
from app.db import make_engine  # noqa: E402
from app.models import StrategyConfig  # noqa: E402
from app.state import all_state  # noqa: E402
from app.strategy.bindings import binding_overrides  # noqa: E402
from app.strategy.pools import pool_symbols_by_code  # noqa: E402
from app.strategy.rules import evaluate_signals, prepare_frame, resolve_params  # noqa: E402
from app.utils import interval_ms, now_ms  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "st_bt", ROOT / "backend/scripts/supertrend_backtest.py"
)
st_bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st_bt)

CODE = "supertrend_current"
CASH = 10_000.0
MIN_TRADES = 30

_SESSION = None


def _db():
    global _SESSION
    if _SESSION is None:
        engine = make_engine(read_only=True, null_pool=True)
        _SESSION = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)()
    return _SESSION


def resolve_for(db, pool: str, interval: str) -> dict:
    cfg = db.query(StrategyConfig).filter(StrategyConfig.code == CODE).first()
    overrides = json.loads(cfg.params or "{}") if cfg else {}
    bound = binding_overrides(db, CODE, pool, interval)
    if bound:
        overrides.update(bound)
    return resolve_params(CODE, all_state(db), overrides, interval=interval)


def run_symbol(symbol: str, interval: str, days: int, params: dict, as_of: int) -> dict:
    """同一份数据跑两遍：原方向 / 反方向。返回两边的账户与逐笔样本。"""
    try:
        segments = st_bt.load_segments(symbol, interval, days, as_of, st_bt.MIN_SEGMENT_BARS)
    except Exception as exc:
        return {"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"[:160]}
    if not segments:
        return {"symbol": symbol, "error": "no_segments"}

    fee, slip = float(params["fee_rate"]), float(params["slippage"])
    step = interval_ms(interval)
    result = {"symbol": symbol}
    for mode in ("normal", "inverse"):
        account = CASH
        legs: list[dict] = []
        for segment in segments:
            segment = segment.loc[(segment.index.as_unit("ms").asi8 + step) <= as_of]
            if segment.empty:
                continue
            frame = prepare_frame(segment, CODE, params)
            signals = evaluate_signals(frame, CODE, params)
            if mode == "inverse":
                signals = signals.copy()
                signals["direction"] = -signals["direction"].astype(int)
            equity, trades = simulate(frame, signals, params, account, fee, slip, interval)
            if trades:
                legs.extend(trades)
            # 段间复利结转：本段结束权益即下一段起始资金（无成交时权益不变）
            if equity is not None and len(equity):
                account = float(equity.iloc[-1])
        positions = aggregate_positions(legs) if legs else []
        returns = [float(item.get("return_pct", 0.0)) for item in positions]
        result[mode] = {
            "account_return_pct": round((account / CASH - 1) * 100, 4),
            "trades": len(positions),
            "mean_pct": round(float(np.mean(returns)), 4) if returns else None,
            "returns": returns,
            "long": sum(1 for leg in legs if leg.get("side") == "long"),
            "short": sum(1 for leg in legs if leg.get("side") == "short"),
            "status": dict(Counter(str(leg.get("status")) for leg in legs)),
            "long_returns": [float(item.get("return_pct", 0.0)) for item in positions
                             if str(item.get("side")) == "long"],
            "short_returns": [float(item.get("return_pct", 0.0)) for item in positions
                              if str(item.get("side")) == "short"],
        }
    return result


def pool_stats(rows: list[dict], mode: str, key: str = "returns") -> dict:
    samples = [value for row in rows if row.get(mode) for value in row[mode][key]]
    ci = bootstrap_expectancy_ci([{"return_pct": value} for value in samples], min_trades=MIN_TRADES)
    accounts = [row[mode]["account_return_pct"] for row in rows if row.get(mode)]
    return {
        "trades": ci.get("trades", 0),
        "mean_pct": ci.get("mean_return_pct"),
        "ci_low_pct": ci.get("ci_low_pct"),
        "ci_high_pct": ci.get("ci_high_pct"),
        "positive_share": ci.get("positive_share"),
        "verdict": ci.get("verdict"),
        "account_median_pct": round(float(np.median(accounts)), 4) if accounts else None,
        "account_mean_pct": round(float(np.mean(accounts)), 4) if accounts else None,
        "positive_symbols": int(sum(1 for value in accounts if value > 0)),
        "symbols": len(accounts),
    }


def run_interval(pool: str, interval: str, days: int) -> dict:
    db = _db()
    params = resolve_for(db, pool, interval)
    members = pool_symbols_by_code(db, pool)
    as_of = now_ms()
    rows = [run_symbol(symbol, interval, days, params, as_of) for symbol in members]
    rows = [row for row in rows if "error" not in row]
    payload = {
        "interval": interval,
        "days": days,
        "members": [row["symbol"] for row in rows],
        "normal": pool_stats(rows, "normal"),
        "inverse": pool_stats(rows, "inverse"),
        "normal_long": pool_stats(rows, "normal", "long_returns"),
        "normal_short": pool_stats(rows, "normal", "short_returns"),
        "inverse_long": pool_stats(rows, "inverse", "long_returns"),
        "inverse_short": pool_stats(rows, "inverse", "short_returns"),
        "per_symbol": [
            {
                "symbol": row["symbol"],
                "normal_pct": row["normal"]["account_return_pct"],
                "inverse_pct": row["inverse"]["account_return_pct"],
                "trades": row["normal"]["trades"],
            }
            for row in rows
        ],
        "status_normal": dict(sum((Counter(row["normal"]["status"]) for row in rows), Counter())),
        "status_inverse": dict(sum((Counter(row["inverse"]["status"]) for row in rows), Counter())),
    }
    return payload


def _row(label: str, stats: dict) -> str:
    ci = "—"
    if stats["ci_low_pct"] is not None:
        ci = f"[{stats['ci_low_pct']:+.3f}, {stats['ci_high_pct']:+.3f}]"
    mean = "—" if stats["mean_pct"] is None else f"{stats['mean_pct']:+.4f}%"
    median = "—" if stats["account_median_pct"] is None else f"{stats['account_median_pct']:+.2f}%"
    return (f"| {label} | {stats['trades']} | {mean} | {ci} | {stats['verdict']} | "
            f"{median} | {stats['positive_symbols']}/{stats['symbols']} |")


def render(payload: dict) -> str:
    lines = [
        f"## {payload['interval']} · {payload['days']} 天 · {len(payload['members'])} 币",
        "",
        "| 方向 | 成交 | 每笔期望（margin 口径） | 95% CI | 判定 | 账户收益中位数 | 正收益币 |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: |",
        _row("原方向", payload["normal"]),
        _row("**反向**", payload["inverse"]),
        "",
        "### 多空分解（每笔期望）",
        "",
        "| 方向 | 做多 | 做空 |",
        "| --- | ---: | ---: |",
        f"| 原方向 | {payload['normal_long']['mean_pct']:+.4f}% | {payload['normal_short']['mean_pct']:+.4f}% |",
        f"| 反向 | {payload['inverse_long']['mean_pct']:+.4f}% | {payload['inverse_short']['mean_pct']:+.4f}% |",
        "",
        "### 出场原因（腿级）",
        "",
        "| 方向 | stop_loss | take_profit_3 | signal_reverse |",
        "| --- | ---: | ---: | ---: |",
        f"| 原方向 | {payload['status_normal'].get('stop_loss', 0)} | "
        f"{payload['status_normal'].get('take_profit_3', 0)} | "
        f"{payload['status_normal'].get('signal_reverse', 0)} |",
        f"| 反向 | {payload['status_inverse'].get('stop_loss', 0)} | "
        f"{payload['status_inverse'].get('take_profit_3', 0)} | "
        f"{payload['status_inverse'].get('signal_reverse', 0)} |",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Supertrend 反向交易检验")
    parser.add_argument("--pool", default="trend_reversal_fit")
    parser.add_argument("--intervals", default="15m,1h")
    parser.add_argument("--days", type=int, default=300)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--out", default="docs/supertrend-inverse-check")
    args = parser.parse_args(argv)

    payloads = []
    for interval in [item.strip() for item in args.intervals.split(",") if item.strip()]:
        payload = run_interval(args.pool, interval, args.days)
        payloads.append(payload)
        normal, inverse = payload["normal"], payload["inverse"]
        print(f"[{interval}] 原方向 {normal['mean_pct']:+.4f}% ({normal['verdict']}) · "
              f"反向 {inverse['mean_pct']:+.4f}% ({inverse['verdict']}) · "
              f"账户中位 原 {normal['account_median_pct']:+.2f}% / 反 {inverse['account_median_pct']:+.2f}%")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": CODE,
        "pool": args.pool,
        "days": args.days,
        "periods": payloads,
    }
    out.with_suffix(".json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    header = [
        "# Supertrend 反向交易检验",
        "",
        f"生成时间：{document['generated_at'][:19]}Z　策略：`{CODE}`　池：`{args.pool}`　窗口：{args.days} 天",
        "",
        "只把 `direction` 整列取反，入场时点/仓位/杠杆/止损止盈绝对距离/成本全部不变。",
        "",
    ]
    body = "\n\n".join(render(payload) for payload in payloads)
    out.with_suffix(".md").write_text("\n".join(header) + body + "\n", encoding="utf-8")
    print(f"[输出] {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
