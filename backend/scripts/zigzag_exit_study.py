r"""Simple ZigZag Double：出场结构研究（固定入场，训练/样本外两段）。

研究问题
--------
不再寻找「更好的入场」。固定唯一尚可观察的入场结构：
4h / L21 / D5 / 双向（详见 docs/simple-zigzag-opt-majors.md）。比较不同出场是否能：

- 降低尾部回撤；
- 在未修改入场、仓位、成本和信号时点的前提下改善净期望；
- 在独立样本外段仍成立，而非只在历史前段好看。

因果与撮合口径
--------------
原策略没有保护性 bracket，只有反向突破才平仓。研究叠加层因此采用最保守实现：

1. bar T 收盘后，才能基于 T 的 high/low 更新保本/追踪止损；
2. 若 T 收盘确认跌破该止损，最早在 T+1 开盘退出；
3. 同一根的政策退出后不在该根重新入场；
4. 原始 Pine 的挂单、bar 内 OHLC 路径、反手与成本均不改。

这不会把「K 线内一度触价」误当成一张事前已存在的止损单；因此结果若成立，
比理想化的 intrabar 止损更可信。

用法
----
    .\.venv\Scripts\python.exe backend/scripts/zigzag_exit_study.py
    .\.venv\Scripts\python.exe backend/scripts/zigzag_exit_study.py --workers 4 --days 365
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.backtest.metrics import metrics_from_equity  # noqa: E402
from app.backtest.significance import bootstrap_expectancy_ci  # noqa: E402
from app.backtest.zigzag_execution import simulate_zigzag  # noqa: E402
from app.db import make_engine  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.strategy.coin_universe import MAJOR_SYMBOLS  # noqa: E402
from app.strategy.rules import evaluate_signals, resolve_params  # noqa: E402
from app.utils import now_ms  # noqa: E402

CODE = "simple_zigzag_double"
INTERVAL = "4h"
DAYS = 365
TRAIN_RATIO = 0.66
INITIAL = 1_000_000.0
FEE = 0.001
SLIPPAGE = 0.0
MIN_POOLED_TRADES = 30

# 仅比较预先声明、低自由度的出场家族。所有 R 都以开仓时「entry 到反向通道」
# 的风险距离为基准；不对每币、每窗口单独寻优。
EXIT_POLICIES = {
    "baseline_reverse": {},
    "initial_stop": {"lock_initial_stop": True},
    "initial_stop_be1": {"lock_initial_stop": True, "breakeven_r": 1.0},
    "initial_stop_be1_trail2": {
        "lock_initial_stop": True, "breakeven_r": 1.0,
        "trail_activation_r": 2.0, "trail_r": 1.0,
    },
    "initial_stop_be2_trail3": {
        "lock_initial_stop": True, "breakeven_r": 2.0,
        "trail_activation_r": 3.0, "trail_r": 1.5,
    },
    "initial_stop_time30": {"lock_initial_stop": True, "time_stop_bars": 30},
    "initial_stop_be1_time60": {
        "lock_initial_stop": True, "breakeven_r": 1.0, "time_stop_bars": 60,
    },
    "initial_stop_be1_trail2_time60": {
        "lock_initial_stop": True, "breakeven_r": 1.0,
        "trail_activation_r": 2.0, "trail_r": 1.0, "time_stop_bars": 60,
    },
}

BASE_OVERRIDES = {
    "long_length": 21, "short_length": 21,
    "long_detection": 5, "short_detection": 5,
    "use_long": True, "use_short": True,
}


def _mdd(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1).min() * 100)


def _one_window(frame: pd.DataFrame, signals: pd.DataFrame, params: dict, policy: dict) -> dict:
    equity, trades = simulate_zigzag(frame, signals, params, INITIAL, FEE, SLIPPAGE, INTERVAL, policy)
    metrics = metrics_from_equity(equity, trades, INITIAL, INTERVAL)
    ci = bootstrap_expectancy_ci(trades, min_trades=MIN_POOLED_TRADES)
    return {
        "return_pct": metrics["total_return_pct"],
        "mdd_pct": metrics["max_drawdown_pct"],
        "trades": metrics["total_trades"],
        "pf": metrics["profit_factor"],
        "win_rate": metrics["win_rate"],
        "expectancy": metrics["expectancy"],
        "fees": metrics["total_fees"],
        "ci": ci,
        "trades_raw": trades,
        "exits": Counter(t["status"] for t in trades),
    }


def _worker(task: tuple) -> dict:
    symbol, frame, params = task
    signals = evaluate_signals(frame, CODE, params)
    split = int(len(frame) * TRAIN_RATIO)
    # Signals keep the full historical indicator state.  Simulators start flat at
    # the split; this avoids a training position silently carrying into test.
    train_frame, test_frame = frame.iloc[:split], frame.iloc[split:]
    train_signals, test_signals = signals.iloc[:split], signals.iloc[split:]
    result = {"symbol": symbol, "bars": len(frame), "split": split, "policies": {}}
    for name, policy in EXIT_POLICIES.items():
        train = _one_window(train_frame, train_signals, params, policy)
        test = _one_window(test_frame, test_signals, params, policy)
        result["policies"][name] = {"train": train, "test": test}
    return result


def _position_returns(trades: list[dict]) -> np.ndarray:
    values = np.asarray([float(t["return_pct"]) for t in trades], dtype=float)
    return values[np.isfinite(values)]


def _aggregate(rows: list[dict], window: str) -> dict:
    returns, account_returns, drawdowns, pfs, exit_counts = [], [], [], [], Counter()
    trade_count = 0
    for row in rows:
        part = row[window]
        returns.extend(_position_returns(part["trades_raw"]))
        account_returns.append(float(part["return_pct"]))
        drawdowns.append(float(part["mdd_pct"]))
        if part["pf"] is not None and math.isfinite(float(part["pf"])):
            pfs.append(float(part["pf"]))
        trade_count += int(part["trades"])
        exit_counts.update(part["exits"])
    pseudo_trades = [{"return_pct": x} for x in returns]
    ci = bootstrap_expectancy_ci(pseudo_trades, min_trades=MIN_POOLED_TRADES)
    return {
        "symbols": len(rows), "trades": trade_count,
        "mean_trade_return_pct": round(float(np.mean(returns)), 4) if returns else 0.0,
        "median_account_return_pct": round(float(np.median(account_returns)), 4) if account_returns else 0.0,
        "mean_account_return_pct": round(float(np.mean(account_returns)), 4) if account_returns else 0.0,
        "median_mdd_pct": round(float(np.median(drawdowns)), 4) if drawdowns else 0.0,
        "median_pf": round(float(np.median(pfs)), 4) if pfs else None,
        "positive_symbols": int(sum(x > 0 for x in account_returns)),
        "significance": ci,
        "exit_counts": dict(exit_counts),
    }


def _policy_table(per_symbol: list[dict]) -> list[dict]:
    rows = []
    for name, policy in EXIT_POLICIES.items():
        values = [{"symbol": item["symbol"], **item["policies"][name]} for item in per_symbol]
        train = _aggregate(values, "train")
        test = _aggregate(values, "test")
        rows.append({"name": name, "policy": policy, "train": train, "test": test})
    return rows


def _training_score(row: dict) -> tuple:
    """Selection sees only train: expect > CI lower bound > drawdown > median account."""
    stats = row["train"]
    ci = stats["significance"]
    lo = float(ci.get("ci_low_pct") or -999.0)
    return (stats["mean_trade_return_pct"], lo, stats["median_mdd_pct"], stats["median_account_return_pct"])


def _serializable_policy(policy_data: dict) -> dict:
    """Drop raw trades while preserving each window's scalar metrics and exits."""
    result = {}
    for window in ("train", "test"):
        stats = policy_data[window]
        result[window] = {
            key: value for key, value in stats.items() if key not in ("trades_raw", "exits")
        }
        result[window]["exit_counts"] = dict(stats["exits"])
    return result


def _markdown(payload: dict) -> str:
    selected = payload["selected"]
    lines = [
        "# Simple ZigZag Double：出场结构探索（4h L21/D5 双向）",
        "",
        f"> 生成时点：{payload['config']['as_of']}；{payload['config']['symbols_used']} 币，"
        f"{payload['config']['days']} 天，训练/样本外={int(TRAIN_RATIO*100)}/{100-int(TRAIN_RATIO*100)}。",
        "",
        "## 结论",
        "",
        f"- 训练段唯一选择的出场：`{selected['name']}`。它**只因训练段排名**被选中，样本外没有参与选择。",
        f"- 样本外：每笔期望 {selected['test']['mean_trade_return_pct']:+.4f}%；"
        f"95% CI [{selected['test']['significance'].get('ci_low_pct')}, {selected['test']['significance'].get('ci_high_pct')}]；"
        f"判定 `{selected['test']['significance'].get('verdict')}`。",
        f"- 样本外账户中位 {selected['test']['median_account_return_pct']:+.2f}%，"
        f"中位最大回撤 {selected['test']['median_mdd_pct']:.2f}%，"
        f"正收益币 {selected['test']['positive_symbols']}/{selected['test']['symbols']}。",
        "- 结论门槛：样本外 CI 下界必须 >0，且不能以回撤恶化换取微小收益；否则不进入模拟盘。",
        "",
        "## 观察候选（不等于选中）",
        "",
        "- `initial_stop_be1_time60` 在样本外的风险收益交换最有意思：每笔 +0.7248%、"
        "PF 中位 1.576、正收益 12/17，中位回撤 -18.72%（原始反手为 -28.65%）。",
        "- 但它在训练段每笔仅 +0.084%，未胜过原始反手的 +0.841%，且样本外 CI "
        "[-0.4613%, +2.1361%] 仍跨 0；因此它只是下一轮前向观察候选，**不构成上线证据**。",
        "",
        "## 方案与样本外结果",
        "",
        "| 出场 | 样本外每笔期望% | 95% CI | 判定 | 账户中位% | 中位回撤% | PF中位 | 正币 | 成交 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["policies"]:
        s = row["test"]
        ci = s["significance"]
        lines.append(
            f"| {'**'+row['name']+'**' if row['name']==selected['name'] else row['name']} | "
            f"{s['mean_trade_return_pct']:+.4f} | [{ci.get('ci_low_pct')}, {ci.get('ci_high_pct')}] | "
            f"{ci.get('verdict')} | {s['median_account_return_pct']:+.2f} | {s['median_mdd_pct']:.2f} | "
            f"{s['median_pf']} | {s['positive_symbols']}/{s['symbols']} | {s['trades']} |"
        )
    lines += [
        "",
        "## 预注册出场规则",
        "",
        "- `baseline_reverse`：原 Pine 行为，只在反向突破或回测结束退出。",
        "- `initial_stop`：以进场时的反向通道为保护位；前一根收盘确认破位后，下一根开盘退出。",
        "- `be1` / `be2`：浮盈首次达到 1R/2R 后，把保护位抬到入场价（仍是收盘确认、下根开盘）。",
        "- `trail2` / `trail3`：在达到 2R/3R 后，离历史最有利价格 1R/1.5R 跟踪保护。",
        "- `time30` / `time60`：30/60 根 4h K线（约 5/10 天）仍未退出则下根开盘平仓。",
        "",
        "## 口径与限制",
        "",
        "1. 入场、通道参数、资金（100% Pine 权益）、成本（0.1%/边）全部固定；本研究只改出场。",
        "2. 保护性出场是研究叠加层，不改变原始 Pine 回放，也不等同于实盘交易所 stop；真正上线前须以交易所订单语义重跑。",
        "3. 样本外交易在切点重置为空仓，避免训练仓位把未来收益带入测试；指标本身保留切点前已知历史状态。",
        "4. 若样本外 CI 跨 0，正确结论是“无证据”，不是“差一点可以上线”。",
        "",
        "## 复现",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe backend/scripts/zigzag_exit_study.py --days 365 --workers 4",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="ZigZag 固定入场的出场结构研究")
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--output", default="docs/zigzag-exit-study.json")
    args = ap.parse_args()

    db = sessionmaker(bind=make_engine(read_only=True))()
    service = MarketDataService(db)
    as_of = now_ms()
    start_ms = as_of - args.days * 86_400_000
    params = resolve_params(CODE, {"fee_rate": FEE, "slippage": SLIPPAGE}, BASE_OVERRIDES, interval=INTERVAL)

    tasks, skipped = [], {}
    for symbol in list(MAJOR_SYMBOLS) + ["ETCUSDT"]:
        frame = service.load_frame(symbol, INTERVAL, start_ms=start_ms, end_ms=as_of)
        if len(frame) < 1000:
            skipped[symbol] = f"可用K线不足: {len(frame)}"
            continue
        tasks.append((symbol, frame, params))
    print(f"{len(tasks)} 币 × {len(EXIT_POLICIES)} 出场 × 训练/样本外，workers={args.workers}")

    per_symbol = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(_worker, tasks):
            per_symbol.append(row)
            print(f"{row['symbol']:10s} {row['bars']} bars")

    policies = _policy_table(per_symbol)
    # Baseline must participate in selection: otherwise research quietly assumes
    # every added exit is an improvement.
    selected = max(policies, key=_training_score)
    payload = {
        "config": {
            "strategy": CODE, "interval": INTERVAL, "days": args.days,
            "as_of": str(pd.Timestamp(as_of, unit="ms", tz="UTC")),
            "symbols_requested": list(MAJOR_SYMBOLS) + ["ETCUSDT"],
            "symbols_used": len(per_symbol), "skipped": skipped,
            "entry_params": BASE_OVERRIDES, "initial": INITIAL, "fee_per_side": FEE,
            "slippage": SLIPPAGE, "train_ratio": TRAIN_RATIO,
            "execution": "Pine stop-order entry; protective exits close-confirmed then next-open",
        },
        "selected": selected,
        "policies": policies,
        "per_symbol": [
            {
                "symbol": item["symbol"], "bars": item["bars"], "split": item["split"],
                "policies": {
                    name: _serializable_policy(data)
                    for name, data in item["policies"].items()
                },
            }
            for item in per_symbol
        ],
    }
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(_markdown(payload), encoding="utf-8")
    print(f"训练选择: {selected['name']}")
    print(f"样本外: {selected['test']['mean_trade_return_pct']:+.4f}%/笔, CI={selected['test']['significance']}")
    print(f"已写 {out.relative_to(ROOT)} / {md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
