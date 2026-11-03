"""Train CPU Transformer research experiments sequentially for the major pool."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.strategy.coin_universe import MAJOR_SYMBOLS  # noqa: E402

RUN = ROOT / "data" / "model_research" / "verified_20261103"
REQUIRED = ("model.pt", "model_config.json", "training_history.json", "evaluation_report.json", "backtest_report.json")


def terminal_lightgbm_status(entry, session_id):
    return entry.get("status") == "candidate" or (entry.get("status") == "failed" and entry.get("session_id") == session_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--wait-for-lightgbm", action="store_true")
    parser.add_argument("--session-id")
    parser.add_argument("--threads", type=int, choices=range(1, 65), default=2)
    args = parser.parse_args()
    if args.wait_for_lightgbm and not args.session_id:
        parser.error("--wait-for-lightgbm 需要与 LightGBM 相同的 --session-id")
    run = args.run
    status = {}
    for symbol in MAJOR_SYMBOLS:
        output = run / symbol / "transformer"
        if args.wait_for_lightgbm:
            deadline = time.monotonic() + 7200
            while time.monotonic() < deadline:
                status_file = run / "run_status.json"
                lightgbm = json.loads(status_file.read_text(encoding="utf-8")) if status_file.exists() else {}
                if terminal_lightgbm_status(lightgbm.get(symbol, {}), args.session_id):
                    break
                time.sleep(5)
            if symbol not in lightgbm or lightgbm[symbol].get("status") != "candidate":
                status[symbol] = {"status": "failed", "detail": "LightGBM 未完成或等待超时"}
                path = run / "transformer_run_status.json"
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(path)
                print(f"{symbol}: {status[symbol]}", flush=True)
                continue
        if all((output / name).is_file() for name in REQUIRED):
            status[symbol] = {"status": "completed_existing"}
        elif output.exists() and any(output.iterdir()):
            status[symbol] = {"status": "incomplete_existing", "detail": "保留已有文件，请核查"}
        else:
            started = time.monotonic()
            print(f"{symbol}: 开始 Transformer 实验", flush=True)
            result = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "train_transformer_research.py"),
                                     "--run", str(run), "--symbol", symbol, "--threads", str(args.threads)], cwd=ROOT, check=False)
            status[symbol] = {"status": "completed" if result.returncode == 0 and all((output / name).is_file() for name in REQUIRED) else "failed",
                              "returncode": result.returncode, "elapsed_seconds": round(time.monotonic() - started, 1)}
        path = run / "transformer_run_status.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
        print(f"{symbol}: {status[symbol]}", flush=True)
    return int(any(v["status"] not in ("completed", "completed_existing") for v in status.values()))


if __name__ == "__main__":
    raise SystemExit(main())
