"""Supertrend 出场归因：方向判断错，还是止损/成本的问题？

做法：**入场点完全复用 baseline**（ATR 止损 + 1R/2R/3R 分批 + 反向翻转），
只把出场规则换掉，同一批样本重新结算：

1. `baseline`     —— 生产规则，已知每笔约 -0.46%（15m）。
2. `hold_H`       —— 无止损、无止盈、无反向出场，进场的次根开盘买入，持有 H 根收盘卖出。
3. `inverse_H`    —— 与 2 完全相同的时点与成本，只把方向取反。

判读：
- `hold_H` 的**毛收益**（未扣费）若显著为正 → 方向本身有信息，亏损来自成本/止损；
  若接近 0 → 方向没有预测力；若显著为负 → 方向是反的（再看 `inverse_H`）。
- 再按 baseline 的出场原因分组：`stop_loss` 那批「如果不止损、持有 H 根」是正是负，
  直接回答「止损有没有冤枉它们」。

口径：收益按**名义本金**计（不做杠杆/仓位换算），与 baseline 的 margin 口径不同，
只用于比较符号与量级。成本 = 双边手续费 + 双边滑点，与生产参数一致。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
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
HOLDS = (2, 4, 8, 16, 24, 32, 48, 96)

_SESSION = None


def _db():
    global _SESSION
    if _SESSION is None:
        engine = make_engine(read_only=True, null_pool=True)
        _SESSION = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)()
    return _SESSION


def resolve_for(db, pool: str, interval: str) -> dict:
    """与实盘/回测同一套三层参数：全局 < 策略配置 < 绑定覆盖。"""
    cfg = db.query(StrategyConfig).filter(StrategyConfig.code == CODE).first()
    overrides = json.loads(cfg.params or "{}") if cfg else {}
    bound = binding_overrides(db, CODE, pool, interval)
    if bound:
        overrides.update(bound)
    return resolve_params(CODE, all_state(db), overrides, interval=interval)


def _status_of(legs: list[dict]) -> str:
    """仓位级出场原因：止损优先（分批止盈的腿会覆盖早期 status，这里显式取并集）。"""
    names = {str(leg.get("status", "")) for leg in legs}
    if "stop_loss" in names:
        return "stop_loss"
    if "signal_reverse" in names:
        return "signal_reverse"
    if any(name.startswith("take_profit") for name in names):
        return "take_profit"
    return "other"


def analyze_segment(seg: pd.DataFrame, params: dict, interval: str) -> list[dict]:
    """跑一次 baseline 撮合，再对每个入场点做固定持有 / 反向结算。"""
    frame = prepare_frame(seg, CODE, params)
    signals = evaluate_signals(frame, CODE, params)
    # simulate 逐 bar 读 `row.atr_pct`，必须喂已富化的帧而不是原始 OHLCV
    _, legs = simulate(frame, signals, params, CASH, params["fee_rate"],
                       params["slippage"], interval)
    if not legs:
        return []

    fee = float(params["fee_rate"])
    slip = float(params["slippage"])
    index = frame.index
    close = frame["close"].to_numpy(dtype=float)

    grouped: dict[object, list[dict]] = defaultdict(list)
    for leg in legs:
        grouped[leg.get("position_id")].append(leg)

    rows: list[dict] = []
    for _, group in grouped.items():
        first = min(group, key=lambda item: item["entry_time"])
        entry_time = pd.Timestamp(first["entry_time"])
        side = {"long": 1, "short": -1}.get(str(first.get("side", "")), 0)
        entry_price = float(first.get("entry_price") or 0.0)
        if side == 0 or entry_price <= 0:
            continue
        i = int(index.searchsorted(entry_time))
        if i >= len(index) or index[i] != entry_time:
            continue
        status = _status_of(group)
        for hold in HOLDS:
            j = i + hold
            if j >= len(close):
                continue
            exit_fill = close[j] * (1 - side * slip)
            ratio = exit_fill / entry_price
            cost = fee * (1 + ratio)
            rows.append({
                "hold": hold,
                "side": side,
                "status": status,
                "gross": side * (ratio - 1),
                "net": side * (ratio - 1) - cost,
                "inverse": -side * (ratio - 1) - cost,
            })
    return rows


def summarize(rows: list[dict], key: str) -> list[dict]:
    """按 hold 汇总：均值 / 胜率 / 毛收益 / bootstrap CI。"""
    out = []
    for hold in HOLDS:
        subset = [row for row in rows if row["hold"] == hold]
        if not subset:
            continue
        values = np.array([row[key] for row in subset], dtype=float)
        values = values[np.isfinite(values)]
        ci = bootstrap_expectancy_ci(
            [{"return_pct": float(value) * 100} for value in values], min_trades=MIN_TRADES
        )
        out.append({
            "hold": hold,
            "trades": int(values.size),
            "mean_pct": round(float(values.mean()) * 100, 4),
            "median_pct": round(float(np.median(values)) * 100, 4),
            "win_rate": round(float((values > 0).mean()), 4),
            "ci_low_pct": ci.get("ci_low_pct"),
            "ci_high_pct": ci.get("ci_high_pct"),
            "verdict": ci.get("verdict"),
        })
    return out


def group_by_status(rows: list[dict], hold: int) -> list[dict]:
    bucket: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["hold"] == hold:
            bucket[row["status"]].append(row)
    result = []
    for status, subset in bucket.items():
        values = np.array([row["net"] for row in subset], dtype=float)
        ci = bootstrap_expectancy_ci(
            [{"return_pct": float(value) * 100} for value in values], min_trades=MIN_TRADES
        )
        result.append({
            "status": status,
            "trades": int(values.size),
            "mean_pct": round(float(values.mean()) * 100, 4),
            "win_rate": round(float((values > 0).mean()), 4),
            "ci_low_pct": ci.get("ci_low_pct"),
            "ci_high_pct": ci.get("ci_high_pct"),
            "verdict": ci.get("verdict"),
        })
    order = {"stop_loss": 0, "take_profit": 1, "signal_reverse": 2, "other": 3}
    return sorted(result, key=lambda item: order.get(item["status"], 9))


def run_interval(pool: str, interval: str, days: int, symbols: list[str] | None) -> dict:
    db = _db()
    params = resolve_for(db, pool, interval)
    members = symbols or pool_symbols_by_code(db, pool)
    as_of = now_ms()
    step = interval_ms(interval)

    rows: list[dict] = []
    per_symbol: dict[str, int] = {}
    for symbol in members:
        try:
            segments = st_bt.load_segments(symbol, interval, days, as_of, st_bt.MIN_SEGMENT_BARS)
        except Exception:  # 单币取数失败不拖垮整轮
            continue
        count = 0
        for segment in segments:
            segment = segment.loc[(segment.index.as_unit("ms").asi8 + step) <= as_of]
            for row in analyze_segment(segment, params, interval):
                row["symbol"] = symbol
                rows.append(row)
                count += 1
        per_symbol[symbol] = count // len(HOLDS)

    long_rows = [row for row in rows if row["side"] > 0]
    short_rows = [row for row in rows if row["side"] < 0]
    payload = {
        "interval": interval,
        "days": days,
        "members": members,
        "holders": list(HOLDS),
        "trades": len(rows) // max(len(HOLDS), 1),
        "net": summarize(rows, "net"),
        "gross": summarize(rows, "gross"),
        "inverse": summarize(rows, "inverse"),
        "long": summarize(long_rows, "net"),
        "short": summarize(short_rows, "net"),
        "by_status": {
            str(hold): group_by_status(rows, hold) for hold in (8, 16, 32)
        },
        "status_share": dict(sorted(
            ((status, sum(1 for row in rows if row["hold"] == 16 and row["status"] == status))
             for status in {row["status"] for row in rows}),
            key=lambda item: -item[1],
        )),
        "per_symbol": per_symbol,
    }
    return payload


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}%"


def render(payload: dict) -> str:
    lines = [
        f"## {payload['interval']} · {payload['days']} 天 · {len(payload['members'])} 币",
        "",
        f"入场点 {payload['trades']} 个（与 baseline 完全同一批）。收益为**名义本金口径**，",
        "已扣双边手续费与滑点；baseline 的每笔期望是 margin 口径，两者只比较符号与量级。",
        "",
        "### 固定持有 H 根（无止损 / 无止盈 / 无反向出场）",
        "",
        "| 持有 | 笔数 | 净收益均值 | 胜率 | 95% CI | 判定 | 毛收益均值 | 反向净收益 |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    gross = {item["hold"]: item["mean_pct"] for item in payload["gross"]}
    inverse = {item["hold"]: item["mean_pct"] for item in payload["inverse"]}
    for item in payload["net"]:
        ci = f"[{item['ci_low_pct']:+.3f}, {item['ci_high_pct']:+.3f}]" if item["ci_low_pct"] is not None else "—"
        lines.append(
            f"| {item['hold']} 根 | {item['trades']} | {_fmt(item['mean_pct'])} | "
            f"{item['win_rate']:.1%} | {ci} | {item['verdict']} | "
            f"{_fmt(gross.get(item['hold']))} | {_fmt(inverse.get(item['hold']))} |"
        )

    lines += ["", "### 多空分解（净收益）", "", "| 持有 | 做多 | 做空 |", "| --- | ---: | ---: |"]
    long_map = {item["hold"]: item["mean_pct"] for item in payload["long"]}
    short_map = {item["hold"]: item["mean_pct"] for item in payload["short"]}
    for hold in payload["holders"]:
        if hold in long_map or hold in short_map:
            lines.append(f"| {hold} 根 | {_fmt(long_map.get(hold))} | {_fmt(short_map.get(hold))} |")

    lines += ["", "### 按 baseline 出场原因分组（不止损、持有 H 根的净收益）", ""]
    for hold, groups in payload["by_status"].items():
        lines += [f"**持有 {hold} 根**", "", "| baseline 出场 | 笔数 | 净收益均值 | 胜率 | 95% CI | 判定 |",
                  "| --- | ---: | ---: | ---: | --- | --- |"]
        for item in groups:
            ci = f"[{item['ci_low_pct']:+.3f}, {item['ci_high_pct']:+.3f}]" if item["ci_low_pct"] is not None else "—"
            lines.append(
                f"| {item['status']} | {item['trades']} | {_fmt(item['mean_pct'])} | "
                f"{item['win_rate']:.1%} | {ci} | {item['verdict']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Supertrend 出场归因：方向 vs 止损 vs 成本")
    parser.add_argument("--pool", default="trend_reversal_fit")
    parser.add_argument("--intervals", default="15m,1h")
    parser.add_argument("--days", type=int, default=300)
    parser.add_argument("--symbols", default="", help="逗号分隔，覆盖 --pool")
    parser.add_argument("--out", default="docs/supertrend-exit-diagnosis")
    args = parser.parse_args(argv)

    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    payloads = []
    for interval in [item.strip() for item in args.intervals.split(",") if item.strip()]:
        payload = run_interval(args.pool, interval, args.days, symbols or None)
        payloads.append(payload)
        print(f"[{interval}] 入场点 {payload['trades']} 个 · 成员 {len(payload['members'])}")
        for item in payload["net"]:
            if item["hold"] == 16:
                print(f"  持有 16 根：净 {item['mean_pct']:+.4f}% "
                      f"CI[{item['ci_low_pct']:+.3f},{item['ci_high_pct']:+.3f}] {item['verdict']}")

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
        "# Supertrend 出场归因：方向判断 vs 止损 vs 成本",
        "",
        f"生成时间：{document['generated_at'][:19]}Z　策略：`{CODE}`　池：`{args.pool}`　窗口：{args.days} 天",
        "",
        "同一批入场点只换出场规则。净收益已扣双边手续费（0.04%×2）与双边滑点（0.02%×2）= 0.12%。",
        "",
    ]
    body = "\n\n".join(render(payload) for payload in payloads)
    out.with_suffix(".md").write_text("\n".join(header) + body + "\n", encoding="utf-8")
    print(f"[输出] {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
