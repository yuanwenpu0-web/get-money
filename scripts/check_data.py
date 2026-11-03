"""Check historical candles without changing the trading database."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.db import make_engine  # noqa: E402
from app.ml.research_data import load_candles, validate_candles  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="只读检查历史 1m K 线")
    parser.add_argument("--symbol", action="append", help="交易对，可重复；默认 BTCUSDT、ETHUSDT")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "data_quality_report.json")
    args = parser.parse_args()
    reports = []
    engine = make_engine(read_only=True, null_pool=True)
    try:
        as_of = int(time.time() * 1000)
        for symbol in args.symbol or ["BTCUSDT", "ETHUSDT"]:
            try:
                _, report = validate_candles(load_candles(engine, symbol.upper()), symbol.upper(), as_of)
            except Exception as exc:
                # Avoid printing DB URLs, which may contain credentials.
                report = {"symbol": symbol.upper(), "passed": False, "error": type(exc).__name__}
            reports.append(report)
    finally:
        engine.dispose()
    result = {"passed": all(r["passed"] for r in reports), "symbols": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
