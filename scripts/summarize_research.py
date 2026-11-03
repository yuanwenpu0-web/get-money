"""Build a compact summary from completed offline research reports."""
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
    parser.add_argument("--run", type=Path, default=ROOT / "data" / "model_research" / "verified_20261103")
    args = parser.parse_args()
    rows = []
    for symbol in MAJOR_SYMBOLS:
        path = args.run / symbol / "transformer"
        report = json.loads((path / "evaluation_report.json").read_text(encoding="utf-8"))
        backtest = json.loads((path / "backtest_report.json").read_text(encoding="utf-8"))
        rows.append({"symbol": symbol, "test_samples": report["sampled_rows"]["test"],
                     "lightgbm_log_loss": report["lightgbm"]["log_loss"],
                     "transformer_log_loss": report["transformer"]["log_loss"],
                     "ensemble_log_loss": report["ensemble"]["log_loss"],
                     "ensemble_passes_holdout_gate": report["ensemble_passes_holdout_gate"],
                     "lightgbm_trades": backtest["lightgbm"]["trades"],
                     "transformer_trades": backtest["transformer"]["trades"],
                     "ensemble_trades": backtest["ensemble"]["trades"]})
    (args.run / "comparison_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 主流币 15 分钟模型研究汇总", "", "所有模型均为离线候选模型。测试时点每 120 分钟抽样一次，三个方案使用相同时点。",
             "", "| 币种 | 测试样本 | LightGBM Log Loss | Transformer Log Loss | 融合 Log Loss | 融合通过测试门槛 | 模型策略交易数 L/T/融合 |",
             "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['symbol']} | {row['test_samples']} | {row['lightgbm_log_loss']:.4f} | {row['transformer_log_loss']:.4f} | {row['ensemble_log_loss']:.4f} | {'是' if row['ensemble_passes_holdout_gate'] else '否'} | {row['lightgbm_trades']} / {row['transformer_trades']} / {row['ensemble_trades']} |")
    lines.extend(["", "融合权重由验证集选择；测试门槛只检查已选权重是否优于两个单模型，不能解释为统计显著或实盘资格。",
                  "", "回测按下一根开盘价、单边手续费 0.04%、单边滑点 0.02%、持仓 15 分钟计算；回撤仅在平仓时计算。未含资金费率、保证金、强平和止损，也不是逐分钟全信号回测。"])
    (args.run / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已汇总 {len(rows)} 个币种: {args.run / 'comparison_summary.md'}")


if __name__ == "__main__":
    main()
