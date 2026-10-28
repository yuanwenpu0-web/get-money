"""P7 探针：定量 funding_filter 叠加层的实际触发率。

对代表币分别用 funding_filter=on/off 跑 evaluate_signals，比对 direction，
统计被拦截的多头信号数，以及 funding_zscore>=2σ 的 bar 占比。
"""
from __future__ import annotations
import sys
sys.path.insert(0, ".")
from sqlalchemy.orm import Session
from app.db import make_engine
from app.ml.dataset import build_dataset
from app.strategy.rules import evaluate_signals, resolve_params
from app.config import DEFAULT_STATE
from app import utils

eng = make_engine(read_only=True)
strat = "ml_ensemble"
base = resolve_params(strat, {**DEFAULT_STATE, "fee_rate": 0.001, "slippage": 0.001})
on = dict(base)
on["funding_filter"] = True
syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT"]
end = int(utils.now_ms())

total_block = 0
total_long = 0
total_extreme = 0
total_bars = 0
for s in syms:
    with Session(eng) as db:
        d = build_dataset(db, s, "1h", horizon=5, days=120, end_ms=end)
    if d is None or len(d) < 800:
        print(s, "skip")
        continue
    sig_off = evaluate_signals(d, strat, base, None)
    sig_on = evaluate_signals(d, strat, on, None)
    off_long = int(sig_off.direction.eq(1).sum())
    on_long = int(sig_on.direction.eq(1).sum())
    blocked = off_long - on_long
    extreme = int((d["funding_zscore"] >= 2.0).sum())
    total_block += blocked
    total_long += off_long
    total_extreme += extreme
    total_bars += len(d)
    print(f"{s}: long_off={off_long} long_on={on_long} blocked={blocked} | bars_z>=2sigma={extreme}/{len(d)}")

print(
    f"--- TOTAL blocked longs / total longs = {total_block}/{total_long} "
    f"= {total_block / max(total_long, 1) * 100:.2f}% ---"
)
print(
    f"--- TOTAL bars z>=2sigma / total bars = {total_extreme}/{total_bars} "
    f"= {total_extreme / max(total_bars, 1) * 100:.2f}% ---"
)
