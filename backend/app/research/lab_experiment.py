"""实验编排：加载数据、遍历单指标候选、切分样本、排序与判定。

数据口径固定为 **native 直查**（`klines` 表里 `interval` 等于目标周期的原始行），
不走 1m 聚合。原因：400 天 × 40 币的 1h 聚合要扫 2300 万条 1m 行，成本不可接受；
且本实验只做指标横评，native 1h 与聚合 1h 的价格序列在已有校验脚本下零差异
（见 `docs/native-consistency.md`）。两种口径不得在同一轮结果里混用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from .lab_grid import Candidate
from .lab_signals import BUILDERS
from .lab_sim import SimConfig, Trade, exit_signals, simulate, to_ms
from .lab_stats import bh_qvalues, summarize

QUERY = text(
    """
    select open_time, open, high, low, close, volume, quote_volume, trades
    from klines
    where symbol = :symbol and interval = :interval and open_time >= :start
    order by open_time
    """
)

INTERVAL_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


@dataclass
class Dataset:
    interval: str
    frames: dict[str, pd.DataFrame]
    exits: dict[str, tuple[np.ndarray, np.ndarray]]
    split_ts: int
    bar_ms: int
    quality: list[dict] = field(default_factory=list)

    @property
    def symbols(self) -> list[str]:
        return list(self.frames)

    def meta(self) -> dict:
        starts = [int(to_ms(frame.index)[0]) for frame in self.frames.values()]
        ends = [int(to_ms(frame.index)[-1]) for frame in self.frames.values()]
        return {
            "interval": self.interval,
            "source": "native",
            "symbols": len(self.frames),
            "bars_total": int(sum(len(f) for f in self.frames.values())),
            "start": str(pd.to_datetime(min(starts), unit="ms", utc=True)),
            "end": str(pd.to_datetime(max(ends), unit="ms", utc=True)),
            "split": str(pd.to_datetime(self.split_ts, unit="ms", utc=True)),
            "bar_ms": self.bar_ms,
        }


def list_symbols(db: Session, interval: str, days: int, limit: int, end_ms: int) -> list[str]:
    start = end_ms - days * 86_400_000
    rows = db.execute(
        text(
            "select symbol, count(*) n from klines where interval=:iv and open_time>=:s "
            "group by symbol order by n desc limit :lim"
        ),
        {"iv": interval, "s": start, "lim": limit},
    ).fetchall()
    return [row[0] for row in rows]


def _frame(db: Session, symbol: str, interval: str, start: int) -> pd.DataFrame:
    rows = db.execute(QUERY, {"symbol": symbol, "interval": interval, "start": start}).mappings().all()
    if len(rows) < 300:
        return pd.DataFrame()
    frame = pd.DataFrame([dict(row) for row in rows])
    frame["open_time"] = frame["open_time"].astype("int64")
    frame = frame.drop_duplicates("open_time").sort_values("open_time")
    frame.index = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame = frame[["open", "high", "low", "close", "volume", "quote_volume", "trades"]].astype(float)
    return frame[frame["close"] > 0]


def load_dataset(
    db: Session,
    interval: str = "1h",
    days: int = 400,
    limit: int = 40,
    end_ms: int | None = None,
    train_ratio: float = 0.7,
    min_coverage: float = 0.85,
) -> Dataset:
    bar_ms = INTERVAL_MS[interval]
    end_ms = int(end_ms or time.time() * 1000)
    start = end_ms - days * 86_400_000
    symbols = list_symbols(db, interval, days, limit, end_ms)
    frames: dict[str, pd.DataFrame] = {}
    quality: list[dict] = []
    for symbol in symbols:
        frame = _frame(db, symbol, interval, start)
        if frame.empty:
            continue
        stamps = to_ms(frame.index)
        expected = int((stamps[-1] - stamps[0]) // bar_ms) + 1
        gaps = expected - len(frame)
        coverage = len(frame) / expected if expected else 0.0
        quality.append(
            {
                "symbol": symbol,
                "bars": len(frame),
                "gaps": gaps,
                "gap_pct": round(gaps / expected * 100, 3) if expected else None,
                "start": str(frame.index[0]),
                "end": str(frame.index[-1]),
                "coverage": round(coverage, 4),
                "zero_volume_pct": round(float((frame["volume"] <= 0).mean() * 100), 3),
            }
        )
        if coverage < min_coverage:
            quality[-1]["dropped"] = True
            continue
        frames[symbol] = frame
    if not frames:
        raise SystemExit("没有可取的数据，检查 symbol/interval/days")
    lo = min(int(to_ms(f.index)[0]) for f in frames.values())
    hi = max(int(to_ms(f.index)[-1]) for f in frames.values())
    split_ts = int(lo + (hi - lo) * train_ratio)
    exits = {s: exit_signals(f["close"].to_numpy(dtype=float)) for s, f in frames.items()}
    return Dataset(interval, frames, exits, split_ts, bar_ms, quality)


def sign(cand: Candidate, frame: pd.DataFrame):
    return BUILDERS[cand.family](frame, cand.params)


def strategy_trades(ds: Dataset, cand: Candidate, cfg: SimConfig) -> list[Trade]:
    out: list[Trade] = []
    for symbol, frame in ds.frames.items():
        out.extend(simulate(symbol, frame, sign(cand, frame), ds.exits[symbol], cfg))
    return out


def split_trades(trades: list[Trade], split_ts: int) -> tuple[list, list, int]:
    """按平仓时间切样本：跨切分点的持仓两边都不算（purge），避免标签泄漏。"""
    train = [t for t in trades if t.exit_ts < split_ts]
    test = [t for t in trades if t.entry_ts >= split_ts]
    return train, test, len(trades) - len(train) - len(test)


def evaluate(ds: Dataset, cand: Candidate, trades: list[Trade], cfg: SimConfig) -> dict:
    train, test, purged = split_trades(trades, ds.split_ts)
    return {
        "family": cand.family,
        "params": cand.params,
        "key": cand.key,
        "label": cand.label,
        "train": summarize(train, cfg.stake, len(ds.frames)),
        "test": summarize(test, cfg.stake, len(ds.frames)),
        "all": summarize(trades, cfg.stake, len(ds.frames)),
        "purged": purged,
    }


def run_singles(
    ds: Dataset,
    cands: list[Candidate],
    cfg: SimConfig,
    min_train_trades: int = 30,
    progress: bool = True,
) -> list[dict]:
    rows: list[dict] = []
    started = time.time()
    for index, cand in enumerate(cands, 1):
        rows.append(evaluate(ds, cand, strategy_trades(ds, cand, cfg), cfg))
        if progress and index % 20 == 0:
            print(f"  单指标 {index}/{len(cands)} 用时 {time.time() - started:.0f}s", flush=True)
    _attach_verdict(rows, min_train_trades)
    return rows


def _attach_verdict(rows: list[dict], min_train_trades: int) -> None:
    if not rows:
        return
    qvalues = bh_qvalues(np.array([row["train"]["p_value"] for row in rows]))
    for row, q in zip(rows, qvalues):
        row["q_value"] = round(float(q), 5)
        train, test = row["train"], row["test"]
        if train["trades"] < min_train_trades:
            verdict = "insufficient"
        elif train["total_pnl"] <= 0:
            verdict = "no_edge"
        elif q >= 0.10:
            verdict = "likely_noise"
        elif test["total_pnl"] <= 0:
            verdict = "unstable_oos"
        else:
            verdict = "candidate"
        row["verdict"] = verdict


def rank(rows: list[dict], key: str = "train", metric: str = "total_pnl") -> list[dict]:
    return sorted(rows, key=lambda r: (r[key][metric], r[key]["trades"]), reverse=True)
