"""Write a readable comparison of completed independent order simulations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.strategy.coin_universe import MAJOR_SYMBOLS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    config = json.loads((args.directory / "config.json").read_text(encoding="utf-8"))
    results = json.loads((args.directory / "summary.json").read_text(encoding="utf-8"))
    if set(results) != set(config.get("symbols", MAJOR_SYMBOLS)) or any(item["status"] != "complete" for item in results.values()):
        raise ValueError("存在未完成币种，不能生成完整汇总")
    schemes = ("lightgbm", "transformer", "ensemble_research", "momentum")
    lines = ["# 模型逐分钟模拟下单比较", "", "各币种、各方案分别使用独立 10,000 USDT 账户；本表不代表共享资金的组合收益。",
             "", f"共同区间 UTC 毫秒：{config['start_ms']} 至 {config['end_ms']}。每分钟评估信号、下一根开盘成交。",
             "", "| 币种 | LightGBM 收益 / 笔数 / 最大回撤 | Transformer 收益 / 笔数 / 最大回撤 | 融合研究 收益 / 笔数 / 最大回撤 | 动量基线 收益 / 笔数 / 最大回撤 |",
             "| --- | --- | --- | --- | --- |"]
    for symbol, item in results.items():
        cells = []
        for name in schemes:
            metrics = item["strategies"][name]
            cells.append(f"{metrics['total_return_pct']:.2f}% / {metrics['total_trades']} / {metrics['max_drawdown_pct']:.2f}%")
        lines.append(f"| {symbol} | " + " | ".join(cells) + " |")
    lines.extend(["", "## 分方案统计", "", "下表收益为各币种独立账户收益率的算术平均，不是共享资金组合的真实收益。", "",
                  "| 方案 | 币种数 | 平均收益率 | 盈利币种数 | 平均交易笔数 |", "| --- | ---: | ---: | ---: | ---: |"])
    for name in schemes:
        metrics = [item["strategies"][name] for item in results.values()]
        lines.append(f"| {name} | {len(metrics)} | {sum(m['total_return_pct'] for m in metrics)/len(metrics):.2f}% | {sum(m['total_return_pct'] > 0 for m in metrics)} | {sum(m['total_trades'] for m in metrics)/len(metrics):.1f} |")
    lines.extend(["", "## BTC / ETH 二分类补充", "", "二分类模型以 15 分钟抽样训练，本次逐分钟推断可能存在分布差异。", "",
                  "| 币种 | 方案 | 收益率 | 笔数 | 胜率 | 最大回撤 | 手续费 USDT | 资金费盈亏 USDT |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for symbol, item in results.items():
        for name in ("binary", "binary_filtered"):
            if name not in item["strategies"]:
                continue
            m = item["strategies"][name]
            lines.append(f"| {symbol} | {name} | {m['total_return_pct']:.2f}% | {m['total_trades']} | {m['win_rate']:.2f}% | {m['max_drawdown_pct']:.2f}% | {m['total_fees']:.2f} | {m['funding_pnl']:.2f} |")
    lines.extend(["", "每笔使用 1 倍杠杆、8% 权益作为基础保证金，单笔风险预算上限 0.5%；以信号 K 线收盘价为参照设置 0.5% 止损、1% 止盈，最长持有 15 分钟，平仓后冷却 5 分钟。",
                  "", "单边手续费 0.04%、单边滑点 0.02%；资金费率来自本地，重复结算窗口通过币安官方历史接口核对。结算价缺失时以对应分钟开盘价近似。",
                  "", "融合权重只由验证集选择，此处仅作假设性样本外比较，不依据同一测试期的融合门槛择优。未模拟深度、排队、网络延迟、强平、交易所最小步长；权益回撤按分钟收盘计算。"])
    (args.directory / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"汇总完成: {args.directory / 'comparison.md'}")


if __name__ == "__main__":
    main()
