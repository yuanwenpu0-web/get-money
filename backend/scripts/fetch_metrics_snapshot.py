"""一次性脚本：拉取多币种多源指标快照并打印（迭代 7 采集验证用）。

直接 import app.market.futures + app.market.metrics，不经过 API / service 层。
用法：
    python backend/scripts/fetch_metrics_snapshot.py --symbols BTCUSDT ETHUSDT SOLUSDT --proxy http://127.0.0.1:7890
不传 --symbols 时默认取配置里的 DEFAULT_SYMBOLS（BTCUSDT/ETHUSDT）。
CoinGecko 限流时 ranks 为 None，仅影响市值/涨跌类字段，不阻断其余指标。
"""

from __future__ import annotations

import argparse
import sys

from app.config import DEFAULT_SYMBOLS
from app.market import metrics as metrics_mod
from app.market.futures import BinanceFuturesClient
from app.market.metrics import base_asset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="拉取多源币种指标快照")
    parser.add_argument("--symbols", nargs="*", default=None, help="币种列表（默认 DEFAULT_SYMBOLS）")
    parser.add_argument("--proxy", default=None, help="HTTP/SOCKS5 代理，如 http://127.0.0.1:7890")
    parser.add_argument("--period", default="5m", help="多空比/主动买卖量/未平仓量聚合周期")
    parser.add_argument("--top", type=int, default=10, help="打印前 N 个币种")
    args = parser.parse_args(argv)

    symbols = [str(s).strip().upper() for s in (args.symbols or DEFAULT_SYMBOLS) if str(s).strip()]
    if not symbols:
        print("无可用币种", file=sys.stderr)
        return 2

    proxy = args.proxy or None
    client = BinanceFuturesClient(testnet=False, proxy=proxy)
    fng = metrics_mod.fetch_fear_greed(proxy=proxy)
    ranks = metrics_mod.fetch_market_ranks(proxy=proxy)
    print(f"代理: {proxy or '直连'} | 恐惧贪婪指数: {fng['value'] if fng else '获取失败'} "
          f"({fng['classification'] if fng else '-'}) | 市值排行: {'已获取' if ranks else '获取失败(限流?)'}")

    snapshots: list[dict] = []
    for sym in symbols:
        try:
            snapshot = metrics_mod.fetch_symbol_metrics(client, sym, args.period, 1)
        except Exception as exc:
            print(f"  {sym}: 采集失败 {exc}", file=sys.stderr)
            continue
        history = []  # 脚本仅即时快照，不读库；z-score/oi_change 此处为 0（属预期）
        features = metrics_mod.build_metric_features(snapshot, fng, ranks.get(base_asset(sym)), history)
        snapshots.append({"symbol": sym, "snapshot": snapshot, "features": features})

    for item in snapshots[: max(args.top, 0)]:
        sym = item["symbol"]
        snap = item["snapshot"]
        feats = item["features"]
        print(f"\n=== {sym} ===")
        print(f"  散户多空比 crowd_ls={snap['crowd_ls']:.4f}  大户多空比 smart_ls={snap['smart_ls']:.4f}")
        print(f"  主动买卖比 taker_buy_sell={snap['taker_buy_sell']:.4f}")
        print(f"  未平仓量 oi={snap['oi']:.2f}  名义价值 oi_value={snap['oi_value']:.2f}")
        rank = ranks.get(base_asset(sym)) if ranks else None
        change_24h = feats["change_24h"] if rank else float('nan')
        mcr = feats["market_cap_rank"] if rank else 0
        print(f"  24h涨跌 change_24h={change_24h}  市值排名 market_cap_rank={mcr}")
        print(f"  特征 crowd_ls={feats['crowd_ls']:.4f} smart_ls={feats['smart_ls']:.4f} "
              f"divergence={feats['sentiment_divergence']:.4f} fng={feats['fng_value']:.0f}")

    print(f"\n共成功 {len(snapshots)} / {len(symbols)} 个币种")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
