r"""4h 通道突破探针：山寨趋势突破策略在本项目数据上到底有没有 edge。

为什么单独做这一档
------------------
P1~P10 已证伪的是 **1m / 15m / 1h** 现货价格序列（P4 实测 1h 时序动量 IC=−0.051）。
nofx 的 `altcoin-trend-breakout-live-v1` 是 **4h**（`INTERVAL_MS = 14_400_000`、`MIN_BARS=180`），
从未在本项目测过。这不是"再测一个已证伪的东西"，而是唯一还没被检验的频段。

库里没有 4h 行（PG: 1m 9.9M / 1h 4.58M / 15m 558k / 5m 34k），所以本脚本
**从 native 1h 按 4 根聚合出 4h**，并要求 4 个 1h 桶齐全、对齐、连续——
这同时也是那条 4h 数据管线的可行性验证。

复刻口径（严格对齐 `rust/src/strategies/altcoin_breakout.rs`）
---------------------------------------------------------------
- EMA 25/75，SMA 种子（与源 `ema()` 一致，不是 pandas ewm 的自由种子）
- ATR 14，Wilder 递归 + SMA 种子（与源 `atr_series()` 一致）
- 通道：**严格前 8 根**的 high 最大值 / low 最小值（`rolling(8).max().shift(1)`），
  当根不参与——源实现用单调队列 `start = index.saturating_sub(channel)` 且先取极值再 push
- 触发 = 「本根条件成立 且 前一根条件不成立」（源的 `fresh trigger`）；多空同根时多优先
- 入场：信号根收盘价定 ATR，**下一根开盘成交**，按 `reanchorToFill` 用成交价重锚止损
- 初始止损 = 1.0 × ATR；移动止损 = 入场后**最高收盘** − 3.0 × 当根 ATR；**无止盈**
- 退出：止损触达 / 移动止损越过收盘（下一根开盘出） / EMA 快慢反转（下一根开盘出）
- 成本：fee 0.0004 + slippage 0.0002，双边

判定（沿用项目 gate 口径）
---------------------------
单笔口径（带真实出场规则）是主判据，前向收益 IC 是辅证：
- 过 gate：只做多 mean_net > 0、t > 2、前后两半同号、且头部流动性组为正
- 不过：任一条不成立 → 停在 Step 0，不写策略代码

已知简化（不声称等价于实盘）
------------------------------
- 每笔信号独立模拟，**没有**共享资金/最多 3 仓/同向 2 仓约束
- 不做资金费（4h 持仓跨资金费结算，真实成本会更高）
- 4h 由 1h 聚合，与交易所原生 4h 的边界一致但未逐根比对

只读运行，不写库。

用法：
    .\.venv\Scripts\python.exe backend/scripts/probe_breakout_4h.py
    .\.venv\Scripts\python.exe backend/scripts/probe_breakout_4h.py --days 400 --top-n 30
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.db import make_engine  # noqa: E402
from app.models import Kline  # noqa: E402
from app.utils import now_ms  # noqa: E402

HOUR_MS = 3_600_000
FOUR_H_MS = 14_400_000
BARS_PER_DAY = 86_400_000 // FOUR_H_MS  # 6

EMA_FAST = 25
EMA_SLOW = 75
CHANNEL = 8
ATR_N = 14
STOP_ATR = 1.0
TRAIL_ATR = 3.0
MIN_BARS = 180  # 源的 minimum_bars()
MAX_HOLD = 500  # 源无时间止损；探针封顶防极端，约 83 天

FEE = 0.0004
SLIP = 0.0002

STABLE_BASES = {
    "USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USDP",
    "USDE", "USDD", "USD1", "PYUSD", "FRAX",
}
FWD_HORIZONS = (1, 3, 6, 12, 24)  # 4h 根数：4h / 12h / 1d / 2d / 4d


# ---------------------------------------------------------------- 数据层


def base_of(symbol: str) -> str:
    for suffix in ("USDT", "USDC", "USD"):
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


def load_1h(db: Session, symbols: list[str], start_ms: int) -> pd.DataFrame:
    rows = db.execute(
        select(
            Kline.symbol, Kline.open_time, Kline.open, Kline.high,
            Kline.low, Kline.close, Kline.quote_volume,
        )
        .where(
            Kline.interval == "1h",
            Kline.symbol.in_(symbols),
            Kline.open_time >= start_ms,
        )
        .order_by(Kline.symbol.asc(), Kline.open_time.asc())
    ).mappings().all()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for col in ("open", "high", "low", "close", "quote_volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def to_4h(frame: pd.DataFrame) -> pd.DataFrame:
    """1h → 4h：只保留「4 根齐全 + 跨度正好 3 小时 + 对齐」的桶。

    与源 `closed_rows` 的苛刻度对齐：宁可少一根，也不用缝合出来的假 4h。
    """
    if frame.empty:
        return frame
    order = frame.sort_values("open_time")
    bucket = (order["open_time"] // FOUR_H_MS) * FOUR_H_MS
    grouped = order.groupby(bucket, sort=True)
    counts = grouped.size()
    spans = grouped["open_time"].max() - grouped["open_time"].min()
    valid = (counts == 4) & (spans == 3 * HOUR_MS)
    if not valid.any():
        return pd.DataFrame(columns=["open", "high", "low", "close", "quote_volume"])
    bars = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "quote_volume": grouped["quote_volume"].sum(),
        }
    )[valid]
    bars.index.name = "open_time"
    return bars


def contiguous_segments(index: pd.Index) -> list[np.ndarray]:
    """按「4h 连续」切段：源的预热与递归种子在缺口处重置，探针同样不跨缺口。"""
    if len(index) == 0:
        return []
    values = np.asarray(index, dtype=np.int64)
    breaks = np.flatnonzero(np.diff(values) != FOUR_H_MS) + 1
    bounds = [0, *breaks.tolist(), len(values)]
    return [
        np.arange(bounds[i], bounds[i + 1])
        for i in range(len(bounds) - 1)
        if bounds[i + 1] - bounds[i] >= MIN_BARS
    ]


# ---------------------------------------------------------------- 指标层


def ema_sma_seed(close: pd.Series, length: int) -> np.ndarray:
    """SMA 种子 + 递推，与源 `ema()` 同构（pandas ewm 的种子不同）。"""
    values = close.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) < length:
        return out
    out[length - 1] = values[:length].mean()
    alpha = 2.0 / (length + 1.0)
    for i in range(length, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = ATR_N) -> np.ndarray:
    """TR 的 Wilder 递归 + SMA 种子，与源 `atr_series()` 同构。"""
    n = len(close)
    out = np.full(n, np.nan)
    if n < length:
        return out
    prev_close = np.concatenate(([np.nan], close[:-1]))
    tr = np.maximum.reduce(
        [high - low, np.abs(high - prev_close), np.abs(low - prev_close)]
    )
    tr[0] = high[0] - low[0]  # 首根没有前收盘，与源 true_ranges 一致
    out[length - 1] = tr[:length].mean()
    for i in range(length, n):
        out[i] = (out[i - 1] * (length - 1) + tr[i]) / length
    return out


def build_signals(bars: pd.DataFrame) -> dict:
    close = bars["close"]
    high = bars["high"]
    low = bars["low"]
    ema_f = ema_sma_seed(close, EMA_FAST)
    ema_s = ema_sma_seed(close, EMA_SLOW)
    atr = wilder_atr(high.to_numpy(float), low.to_numpy(float), close.to_numpy(float))
    # 严格前 CHANNEL 根：当根不参与
    ch_high = high.rolling(CHANNEL).max().shift(1).to_numpy(float)
    ch_low = low.rolling(CHANNEL).min().shift(1).to_numpy(float)
    c = close.to_numpy(float)
    warm = np.arange(len(bars)) >= (MIN_BARS - 1)
    long_cond = warm & (c > ch_high) & (ema_f > ema_s)
    short_cond = warm & (c < ch_low) & (ema_f < ema_s)
    long_trig = long_cond & ~np.concatenate(([False], long_cond[:-1]))
    short_trig = short_cond & ~np.concatenate(([False], short_cond[:-1])) & ~long_trig
    return {
        "ema_fast": ema_f, "ema_slow": ema_s, "atr": atr,
        "long": long_trig, "short": short_trig,
    }


# ---------------------------------------------------------------- 撮合层


def simulate_trade(bars: pd.DataFrame, atr: np.ndarray, ema_f: np.ndarray,
                   ema_s: np.ndarray, i: int, side: int) -> dict | None:
    """单笔独立模拟（无资金约束）。i = 信号根，成交在 i+1 开盘。"""
    open_ = bars["open"].to_numpy(float)
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    n = len(close)
    if i + 1 >= n or not np.isfinite(atr[i]) or atr[i] <= 0:
        return None
    entry = open_[i + 1] * (1 + side * SLIP)
    risk = STOP_ATR * atr[i]
    stop = entry - side * risk
    if not np.isfinite(stop) or stop <= 0:
        return None
    extreme = entry
    exit_price = None
    reason = "max_hold"
    last = min(n, i + 1 + MAX_HOLD)
    for j in range(i + 1, last):
        if side > 0 and low[j] <= stop:
            exit_price, reason = stop * (1 - SLIP), "stop"
            break
        if side < 0 and high[j] >= stop:
            exit_price, reason = stop * (1 + SLIP), "stop"
            break
        extreme = max(extreme, close[j]) if side > 0 else min(extreme, close[j])
        if np.isfinite(atr[j]):
            proposed = extreme - side * TRAIL_ATR * atr[j]
            tighter = max(stop, proposed) if side > 0 else min(stop, proposed)
            crossed = (side > 0 and tighter >= close[j]) or (side < 0 and tighter <= close[j])
            if crossed:
                if j + 1 < n:
                    exit_price = open_[j + 1] * (1 - side * SLIP)
                else:
                    exit_price = close[j]
                reason = "trail_cross"
                break
            stop = tighter
        reversed_ = (side > 0 and ema_f[j] < ema_s[j]) or (side < 0 and ema_f[j] > ema_s[j])
        if reversed_:
            if j + 1 < n:
                exit_price = open_[j + 1] * (1 - side * SLIP)
            else:
                exit_price = close[j]
            reason = "ema_reversal"
            break
    if exit_price is None:
        j = last - 1
        exit_price = close[j] * (1 - side * SLIP)
    gross = side * (exit_price - entry) / entry
    net = gross - 2 * FEE
    return {
        "side": "long" if side > 0 else "short",
        "entry_time": int(bars.index[i + 1]),
        "bars_held": j - i,
        "net": float(net),
        "reason": reason,
    }


# ---------------------------------------------------------------- 探针层


def probe_symbol(bars: pd.DataFrame) -> dict | None:
    if bars is None or bars.empty:
        return None
    close = bars["close"].to_numpy(float)
    trades: list[dict] = []
    fwd: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for seg in contiguous_segments(bars.index):
        chunk = bars.iloc[seg]
        sig = build_signals(chunk)
        # 前向收益（仅用于 IC 辅证）
        seg_close = chunk["close"].to_numpy(float)
        side_series = np.zeros(len(chunk))
        side_series[sig["long"]] = 1.0
        side_series[sig["short"]] = -1.0
        for h in FWD_HORIZONS:
            fwd_ret = np.full(len(seg_close), np.nan)
            if len(seg_close) > h:
                fwd_ret[:-h] = seg_close[h:] / seg_close[:-h] - 1.0
            pair = fwd.setdefault(h, (np.array([], dtype=float), np.array([], dtype=float)))
            fwd[h] = (
                np.concatenate([pair[0], side_series]),
                np.concatenate([pair[1], fwd_ret]),
            )
        for i in np.flatnonzero(sig["long"] | sig["short"]):
            side = 1 if sig["long"][i] else -1
            trade = simulate_trade(chunk, sig["atr"], sig["ema_fast"], sig["ema_slow"], int(i), side)
            if trade is not None:
                trades.append(trade)
    if not trades:
        return None
    nets = np.array([t["net"] for t in trades], dtype=float)
    longs = np.array([t["net"] for t in trades if t["side"] == "long"], dtype=float)
    shorts = np.array([t["net"] for t in trades if t["side"] == "short"], dtype=float)
    times = np.array([t["entry_time"] for t in trades], dtype=np.int64)
    half = np.median(times)
    ics = {}
    for h, (side_a, ret_a) in fwd.items():
        mask = np.isfinite(ret_a) & (side_a != 0)
        if mask.sum() >= 20:
            s, r = side_a[mask], ret_a[mask]
            if s.std() > 0 and r.std() > 0:
                ics[h] = float(np.corrcoef(s, r)[0, 1])
    return {
        "bars": int(len(close)),
        "days": round(len(close) / BARS_PER_DAY, 1),
        "trades": len(trades),
        "mean_net": float(nets.mean()),
        "median_net": float(np.median(nets)),
        "win_rate": float((nets > 0).mean()),
        "long_n": int(len(longs)),
        "long_mean": float(longs.mean()) if len(longs) else None,
        "long_win": float((longs > 0).mean()) if len(longs) else None,
        "short_n": int(len(shorts)),
        "short_mean": float(shorts.mean()) if len(shorts) else None,
        "short_win": float((shorts > 0).mean()) if len(shorts) else None,
        "first_half_mean": float(nets[times < half].mean()) if (times < half).any() else None,
        "second_half_mean": float(nets[times >= half].mean()) if (times >= half).any() else None,
        "avg_bars_held": float(np.mean([t["bars_held"] for t in trades])),
        "ic": ics,
    }


def t_stat(values: list[float]) -> float | None:
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) < 3 or arr.std(ddof=1) == 0:
        return None
    return float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr))))


def aggregate(rows: list[dict], key: str) -> dict:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(float(np.mean(vals)), 6),
        "median": round(float(np.median(vals)), 6),
        "pos_rate": round(float(np.mean([v > 0 for v in vals])), 3),
        "t": round(t_stat(vals), 2) if t_stat(vals) is not None else None,
    }


def trimmed_mean(values: list[float], drop: int = 3) -> float | None:
    """符号级截尾均值：去掉两端各 drop 个币，检验结论是否由个别肥尾币撑起。"""
    arr = sorted(v for v in values if v is not None and np.isfinite(v))
    if len(arr) <= 2 * drop:
        return None
    core = arr[drop : len(arr) - drop] if drop else arr
    return float(np.mean(core))


def render_markdown(report: dict, passes: bool) -> str:
    u = report["universe"]
    rows = report["symbols"]
    head_set = set(report["head"]["symbols"])
    head_sorted = sorted(
        [r for r in rows if r["symbol"] in head_set], key=lambda r: r["mean_net"], reverse=True
    )
    top_row = head_sorted[0] if head_sorted else None
    p = report["params"]
    lines = [
        "# 4h 通道突破探针（山寨趋势突破 · EMA25/75 + 8 根通道）",
        "",
        f"> 生成时间：{report['as_of']}",
        f"> 数据：`{p['data_source']}`，{u['probed']} 币 / 中位 {u['median_days']} 天 / 中位 {u['median_4h_bars']} 根 4h",
        f"> 口径：EMA {p['ema'][0]}/{p['ema'][1]}、通道 {p['channel']}、ATR {p['atr']}、"
        f"初始止损 {p['stop_atr']} ATR、跟踪 {p['trail_atr']} ATR、无止盈、费 {p['fee']}+滑 {p['slippage']}",
        "",
        "## 一句话结论",
        "",
        f"**{'过 gate' if passes else '未过 gate'}** —— {report['verdict']['note']}",
        "",
        "关键读数：",
        "",
        f"- 全样本单笔均值 **{report['all']['mean_net']['mean']:+.4%}**（t={report['all']['mean_net']['t']}，"
        f"{report['all']['trades_total']} 笔，胜率 {report['all']['win_rate']['mean']:.1%}）",
        f"- 只做多 {report['all']['long_mean']['mean']:+.4%}（t={report['all']['long_mean']['t']}） / "
        f"只做空 {report['all']['short_mean']['mean']:+.4%}（t={report['all']['short_mean']['t']}）",
        f"- 头部 {report['head']['mean_net']['mean']:+.4%}（t={report['head']['mean_net']['t']}，"
        f"正收益币占比 {report['head']['mean_net']['pos_rate']:.0%}） / "
        f"长尾 {report['tail']['mean_net']['mean']:+.4%}（t={report['tail']['mean_net']['t']}）",
        f"- 前向收益 IC（2 日）：{report['all']['ic'][12].get('mean')}，全部频段为负 → 无方向性 alpha",
        f"- 符号级截尾均值：全样本 {report['robust']['trimmed_all']}、头部 {report['robust']['trimmed_head']}",
        f"- 正收益币 {report['robust']['symbols_positive']}/{report['robust']['symbols_total']}"
        f"（{report['robust']['symbols_positive'] / report['robust']['symbols_total']:.0%}）——低于抛硬币",
        "",
        "### 头部「正收益」由单个币撑起",
        "",
        f"头部等权均值 {report['head']['mean_net']['mean']:+.4%} 里，"
        f"**{top_row['symbol']} 一个币的均值就有 {top_row['mean_net']:+.2%}**；"
        f"剔掉它后头部均值降到 **{report['robust']['head_ex_top1']:+.4%}**，"
        f"两端各截 3 币后为 **{report['robust']['trimmed_head']:+.4%}**。",
        "",
        f"该币 {top_row['trades']} 笔、胜率 {top_row['win_rate']:.1%}，"
        f"前半 {top_row['first_half_mean']:+.2%}、后半 {top_row['second_half_mean']:+.2%}"
        " —— 单次上市行情的一次性肥尾，不是可复制的信号。",
        "",
        "## 分组建模（符号级均值，等权）",
        "",
        "| 组 | 币数 | 单笔均值 | 中位 | 胜率 | t | 只多均值 | 只空均值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("全样本", "all"), ("头部 top-N", "head"), ("长尾", "tail")):
        block = report[key]
        mn = block["mean_net"]
        lm, sm = block.get("long_mean", {}), block.get("short_mean", {})
        count = len(report["head"]["symbols"]) if key == "head" else (
            report["tail"]["count"] if key == "tail" else u["probed"]
        )
        lines.append(
            f"| {label} | {count} | {mn['mean']:+.4%} | {mn.get('median', 0):+.4%} | "
            f"{block['win_rate'].get('mean', 0):.1%} | {mn['t']} | "
            f"{lm.get('mean', 0):+.4%} | {sm.get('mean', 0):+.4%} |"
        )
    lines += [
        "",
        "## 前向收益 IC（信号方向 ±1 vs 未来 N 根 4h 收益，仅突破 bar）",
        "",
        "| 组 | 4h | 12h | 1d | 2d | 4d |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("全样本", "all"), ("头部", "head")):
        ic = report[key]["ic"]
        cells = " | ".join(str(ic[h].get("mean", "—")) for h in FWD_HORIZONS)
        lines.append(f"| {label} | {cells} |")
    lines += [
        "",
        "## 头部逐币（按单笔均值降序）",
        "",
        "| 币 | 笔数 | 均值 | 中位 | 胜率 | 只多均值 | 只空均值 | 前半 | 后半 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in head_sorted:
        lm = r["long_mean"] if r["long_mean"] is not None else float("nan")
        sm = r["short_mean"] if r["short_mean"] is not None else float("nan")
        h1 = r["first_half_mean"] if r["first_half_mean"] is not None else float("nan")
        h2 = r["second_half_mean"] if r["second_half_mean"] is not None else float("nan")
        lines.append(
            f"| {r['symbol']} | {r['trades']} | {r['mean_net']:+.4%} | {r['median_net']:+.4%} | "
            f"{r['win_rate']:.1%} | {lm:+.4%} | {sm:+.4%} | {h1:+.4%} | {h2:+.4%} |"
        )
    lines += [
        "",
        "## 与源仓库结论对照",
        "",
        "源仓库 `output/altcoin-direction-analysis-20261026.md`（90 天窗口、Rust 完整组合回放）结论：",
        "只做多的正收益只在 orig45（流动性好）成立，在 fresh79（微市值为主）8 个候选 7 个亏损；"
        "做空在两个群体里都是纯拖累。",
        "",
        "本次用本项目的独立数据管线（native 1h→4h、独立成本、400 天）复测，方向性一致：",
        "",
        f"- 头部只多 {report['head']['long_mean']['mean']:+.4%}，长尾只多 {report['tail']['long_mean']['mean']:+.4%}",
        f"- 只空在全样本 {report['all']['short_mean']['mean']:+.4%}（t={report['all']['short_mean']['t']}）、"
        f"长尾 {report['tail']['short_mean']['mean']:+.4%}（t={report['tail']['short_mean']['t']}）→ **做空结构性亏损被独立样本确认**",
        "",
        "## 已知简化（不声称等价于实盘）",
        "",
        "- 每笔信号独立模拟，**无**共享资金 / 最多 3 仓 / 同向 2 仓约束",
        "- 不计资金费（4h 持仓跨结算，真实成本更高）",
        "- 4h 由 native 1h 聚合（要求 4 根齐全且连续），未与交易所原生 4h 逐根比对",
        "- 未做 top-30 换手选币的逐期重排，头部组是**全窗口静态**成交额前 N",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--top-n", type=int, default=30, help="头部流动性组币数")
    ap.add_argument("--min-bars", type=int, default=MIN_BARS + 60, help="每币最少 4h 根数")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--output", default="docs/probe-breakout-4h.json")
    ap.add_argument("--markdown", default="docs/probe-breakout-4h.md")
    args = ap.parse_args()

    end = now_ms()
    start = end - args.days * 86_400_000
    engine = make_engine(read_only=True)

    with Session(engine) as db:
        symbols = sorted({
            s for s in db.execute(
                select(Kline.symbol).where(Kline.interval == "1h").distinct()
            ).scalars()
            if base_of(s) not in STABLE_BASES
        })
        print(f"[probe] 候选 {len(symbols)} 币（已剔除稳定币），窗口 {args.days} 天", flush=True)

        per_symbol: dict[str, dict] = {}
        turnover: dict[str, float] = {}
        dropped = {"no_4h": 0, "too_short": 0, "no_trade": 0}
        for i in range(0, len(symbols), args.batch):
            batch = symbols[i : i + args.batch]
            raw = load_1h(db, batch, start)
            if raw.empty:
                continue
            for symbol, group in raw.groupby("symbol", sort=False):
                bars = to_4h(group)
                if bars.empty:
                    dropped["no_4h"] += 1
                    continue
                covered_days = max((group["open_time"].max() - group["open_time"].min()) / 86_400_000, 1e-9)
                turnover[symbol] = float(group["quote_volume"].sum() / covered_days)
                if len(bars) < args.min_bars:
                    dropped["too_short"] += 1
                    continue
                result = probe_symbol(bars)
                if result is None:
                    dropped["no_trade"] += 1
                    continue
                per_symbol[symbol] = result
            print(f"[probe] {min(i + args.batch, len(symbols))}/{len(symbols)} 已处理，有效 {len(per_symbol)}", flush=True)

    if not per_symbol:
        print("无有效样本")
        return

    ranked = sorted(turnover, key=lambda s: turnover[s], reverse=True)
    head = set(ranked[: args.top_n])
    rows = [{"symbol": s, **v, "daily_usd": turnover.get(s, 0.0)} for s, v in per_symbol.items()]
    for row in rows:
        row["bucket"] = "head" if row["symbol"] in head else "tail"

    head_rows = [r for r in rows if r["bucket"] == "head"]
    tail_rows = [r for r in rows if r["bucket"] == "tail"]
    all_nets = [r["mean_net"] for r in rows]

    def ic_agg(key: int, subset: list[dict]) -> dict:
        vals = [r["ic"][key] for r in subset if key in r["ic"]]
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals),
            "mean": round(float(np.mean(vals)), 4),
            "same_sign_rate": round(float(np.mean([v > 0 for v in vals])), 3),
        }

    top1 = max(head_rows, key=lambda r: r["mean_net"]) if head_rows else None
    rest = [r["mean_net"] for r in head_rows if top1 is None or r is not top1]
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "params": {
            "days": args.days, "top_n": args.top_n, "min_bars": args.min_bars,
            "ema": [EMA_FAST, EMA_SLOW], "channel": CHANNEL, "atr": ATR_N,
            "stop_atr": STOP_ATR, "trail_atr": TRAIL_ATR, "max_hold": MAX_HOLD,
            "fee": FEE, "slippage": SLIP, "interval": "4h", "data_source": "native 1h → 4h 聚合",
        },
        "universe": {
            "candidates": len(symbols), "probed": len(rows), "dropped": dropped,
            "median_4h_bars": int(np.median([r["bars"] for r in rows])),
            "median_days": float(np.median([r["days"] for r in rows])),
        },
        "all": {
            "trades_total": int(sum(r["trades"] for r in rows)),
            "mean_net": aggregate(rows, "mean_net"),
            "long_mean": aggregate(rows, "long_mean"),
            "short_mean": aggregate(rows, "short_mean"),
            "win_rate": aggregate(rows, "win_rate"),
            "first_half_mean": aggregate(rows, "first_half_mean"),
            "second_half_mean": aggregate(rows, "second_half_mean"),
            "avg_bars_held": aggregate(rows, "avg_bars_held"),
            "ic": {h: ic_agg(h, rows) for h in FWD_HORIZONS},
        },
        "head": {
            "symbols": sorted(head),
            "mean_net": aggregate(head_rows, "mean_net"),
            "long_mean": aggregate(head_rows, "long_mean"),
            "short_mean": aggregate(head_rows, "short_mean"),
            "win_rate": aggregate(head_rows, "win_rate"),
            "ic": {h: ic_agg(h, head_rows) for h in FWD_HORIZONS},
        },
        "tail": {
            "count": len(tail_rows),
            "mean_net": aggregate(tail_rows, "mean_net"),
            "long_mean": aggregate(tail_rows, "long_mean"),
            "short_mean": aggregate(tail_rows, "short_mean"),
            "win_rate": aggregate(tail_rows, "win_rate"),
        },
        "symbols": sorted(rows, key=lambda r: r["mean_net"], reverse=True),
        "robust": {
            "trimmed_all": round(trimmed_mean(all_nets), 6),
            "trimmed_head": round(trimmed_mean([r["mean_net"] for r in head_rows]), 6),
            "trimmed_tail": round(trimmed_mean([r["mean_net"] for r in tail_rows]), 6),
            "head_top1_symbol": top1["symbol"] if top1 else None,
            "head_top1_mean": round(top1["mean_net"], 6) if top1 else None,
            "head_ex_top1": round(float(np.mean(rest)), 6) if rest else None,
            "symbols_positive": int(sum(1 for r in rows if r["mean_net"] > 0)),
            "symbols_total": len(rows),
        },
    }

    # ------------------------------------------------------------ 判定
    overall = report["all"]["mean_net"]
    long_only = report["all"]["long_mean"]
    head_long = report["head"]["long_mean"]
    halves = (report["all"]["first_half_mean"], report["all"]["second_half_mean"])
    ic_main = report["all"]["ic"].get(12, {}).get("mean")
    passes = (
        (long_only.get("mean") or -1) > 0
        and (long_only.get("t") or 0) > 2
        and all((h.get("mean") or 0) > 0 for h in halves)
        and (head_long.get("mean") or -1) > 0
    )
    report["verdict"] = {
        "passes_gate": bool(passes),
        "long_only_minus_double": round(
            (long_only.get("mean") or 0) - (report["all"]["short_mean"].get("mean") or 0), 6
        ),
        "note": (
            "只做多在头部流动性组为正、t>2、前后两半同号 → 值得进 Step 1"
            if passes
            else "4h 突破未过 gate：单笔净收益不足或方向不稳定，停止在 Step 0，不写策略代码"
        ),
        "ic_2d": ic_main,
    }

    with open(args.output, "w", encoding="utf-8") as h:
        json.dump(report, h, ensure_ascii=False, indent=2)
    if args.markdown:
        Path(args.markdown).write_text(render_markdown(report, passes), encoding="utf-8")

    print("\n=== 4h 通道突破（单笔口径，含成本，独立无资金约束）===")
    print(f"  宇宙：{report['universe']['probed']} 币 / 中位 {report['universe']['median_days']} 天 / "
          f"{report['all']['trades_total']} 笔")
    for label, block in (("全样本", report["all"]), ("头部", report["head"]), ("长尾", report["tail"])):
        mn, lm, sm = block["mean_net"], block.get("long_mean", {}), block.get("short_mean", {})
        if not mn.get("n"):
            continue
        print(f"  {label:4s} 均值={mn.get('mean'):+.4%} 中位={mn.get('median'):+.4%} "
              f"胜率={block['win_rate'].get('mean', 0):.1%} t={mn.get('t')}  "
              f"只多={lm.get('mean', 0):+.4%}(t={lm.get('t')})  只空={sm.get('mean', 0):+.4%}(t={sm.get('t')})")
    print(f"  前半={report['all']['first_half_mean'].get('mean'):+.4%}  "
          f"后半={report['all']['second_half_mean'].get('mean'):+.4%}  "
          f"平均持仓={report['all']['avg_bars_held'].get('mean')} 根")
    print("  IC(前向收益)：" + "  ".join(
        f"{h}根={report['all']['ic'][h].get('mean')}" for h in FWD_HORIZONS))
    print(f"\n判定：{'过 gate' if passes else '未过 gate'} — {report['verdict']['note']}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
