"""筛选器自校准：证明 signal_screen 能抓到真信号，不是「永远输出 0」的死工具。

为什么要做
----------
「132 个条件 0 个通过」这个结论的可信度，取决于检测器本身有没有灵敏度。
如果没有正对照，无法区分「市场上没有 edge」和「筛选器坏了」。

三个对照：
1. **正对照（泄漏信号）**：用 shift 后的未来收益方向当信号。它必然有极强 edge
   （因为直接偷看了答案）。筛选器若报它 candidate，说明检测链路是通的。
   这是标准做法——用一个已知答案的输入校准量具。
2. **负对照（随机信号）**：触发率与真实信号相同的随机布尔序列。它必然不显著。
   若筛选器把它报成 candidate，说明显著性判据太松（假阳性率失控）。
3. **半衰对照**：正对照的前瞻窗口缩短到与事件间隔相同，edge 应下降但仍显著，
   用来确认「重叠计数」没有把 p 值虚高。

只读，不写 DB。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from app.db import make_engine
from app.market.data_service import MarketDataService

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signal_screen import (  # noqa: E402
    ROUND_TRIP,
    bh_qvalues,
    bootstrap_ci,
    forward_return,
    rising_edges,
    segment_sign_consistency,
)

HORIZON = 24
DAYS = 400


def evaluate(mask: np.ndarray, close: np.ndarray, times: np.ndarray, horizon: int) -> dict:
    fut = forward_return(close, horizon)
    pos = rising_edges(mask, horizon)
    pos = pos[np.isfinite(fut[pos])]
    if len(pos) < 5:
        return {"events": len(pos), "verdict": "insufficient"}
    net = fut[pos] - ROUND_TRIP
    n = len(net)
    mean_net = float(net.mean())
    std = float(net.std(ddof=1))
    t_stat = mean_net / (std / np.sqrt(n)) if std > 0 else 0.0
    from math import erfc, sqrt

    p = float(erfc(abs(t_stat) / sqrt(2)))
    lo, hi = bootstrap_ci(net)
    seg = segment_sign_consistency(net, times[pos])
    return {
        "events": n,
        "mean_net_pct": round(mean_net * 100, 4),
        "t_stat": round(t_stat, 3),
        "p_value": p,
        "ci_low_pct": round(lo * 100, 4),
        "ci_high_pct": round(hi * 100, 4),
        "seg": round(seg, 3) if np.isfinite(seg) else None,
    }


def main() -> None:
    engine = make_engine(read_only=True)
    db = sessionmaker(bind=engine)()
    svc = MarketDataService(db)
    now = int(time.time() * 1000)

    with engine.connect() as conn:
        symbols = [
            row[0]
            for row in conn.execute(
                text(
                    "select symbol, count(*) n from klines where interval='1h' "
                    "group by symbol having count(*) > 4800 order by n desc limit 12"
                )
            ).fetchall()
        ]
    print(f"校准池 {len(symbols)} 币，前瞻 {HORIZON} 根\n")

    rng = np.random.default_rng(20261030)
    leak_results, rand_results, half_results = [], [], []

    for symbol in symbols:
        frame = svc.load_frame(symbol, "1h", start_ms=now - DAYS * 86_400_000, end_ms=now)
        if len(frame) < HORIZON * 4:
            continue
        close = frame["close"].to_numpy(float)
        times = frame.index.asi8
        fut = forward_return(close, HORIZON)

        # 正对照：直接偷看未来（必然有 edge）
        leak = np.zeros(len(close), bool)
        leak[:-HORIZON] = fut[:-HORIZON] > 0
        leak_results.append(evaluate(leak, close, times, HORIZON))

        # 负对照：同触发率的随机信号（必然无 edge）
        hit = float(leak.mean())
        rand = rng.random(len(close)) < hit
        rand_results.append(evaluate(rand, close, times, HORIZON))

        # 半衰对照：前瞻窗口砍半，edge 应仍显著（验证重叠计数口径）
        half_results.append(evaluate(leak, close, times, HORIZON // 2))

    def summarize(items, label):
        vals = [r for r in items if r.get("events", 0) >= 30]
        if not vals:
            print(f"{label}: 样本不足")
            return
        mean_net = np.mean([r["mean_net_pct"] for r in vals])
        pv = np.array([r["p_value"] for r in vals])
        q = bh_qvalues(pv)
        ci_pos = sum(1 for r in vals if (r["ci_low_pct"] or -1) > 0)
        print(
            f"{label:22s} 均值净超额={mean_net:+7.3f}%  "
            f"CI下界>0 的币数={ci_pos}/{len(vals)}  "
            f"最小q={q.min():.2e}  中位p={np.median(pv):.2e}"
        )

    print("=== 校准结果 ===")
    summarize(leak_results, "正对照(偷看未来)")
    summarize(rand_results, "负对照(随机)")
    summarize(half_results, "半衰(前瞻砍半)")

    print("\n=== 正对照明细（前5币）===")
    for s, r in list(zip(symbols, leak_results))[:5]:
        print(f"  {s:12s} n={r['events']:5d} 净={r['mean_net_pct']:+7.3f}% "
              f"t={r['t_stat']:6.2f} CI=[{r['ci_low_pct']:6.2f},{r['ci_high_pct']:6.2f}]")
    print("\n=== 负对照明细（前5币）===")
    for s, r in list(zip(symbols, rand_results))[:5]:
        print(f"  {s:12s} n={r['events']:5d} 净={r['mean_net_pct']:+7.3f}% "
              f"t={r['t_stat']:6.2f} CI=[{r['ci_low_pct']:6.2f},{r['ci_high_pct']:6.2f}]")


if __name__ == "__main__":
    main()
