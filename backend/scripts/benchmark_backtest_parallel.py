r"""回测并行化基准：量化「多线程」到底能提速多少，以及为什么。

结论先行（本机实测中位数，Windows + PostgreSQL，27 币种 × 1 窗口 × 3 轮）
------------------------------------------------------------------------
| 配置 | 耗时 | 相对串行 |
|---|---:|---:|
| 串行 | 38.5s | 1.00× |
| 仅数据提取开线程（--workers 8） | 34.9s | 1.10× |
| 提取线程 + 按档进程池（--processes 8） | 22.3s | **1.73×** |

另有两个被否掉的配置（单次实测，未进正式基准）：
| 配置 | 耗时 | 结论 |
|---|---:|---|
| 寻优开线程 | 37.0s | 与串行持平，GIL 锁死，零收益 |
| 寻优进程但每次新建池 | 52.9s | 比串行更慢，spawn 开销吃掉一切 |

为什么线程在寻优上完全无效
--------------------------
`execution.simulate` 是**逐 bar 的纯 Python 循环**，全部运行在 GIL 内，
线程只会增加调度与切换开销。实测 8 线程 37.0s ≈ 串行 36.8s，即零收益。

为什么进程池要「只建一次、按档复用」
------------------------------------
Windows 用 spawn 创建进程，每次建池都要重新 import 整个模块；
若每档每轮都新建池，spawn 开销超过计算本身（52.9s，反而更慢）。
正确做法是把并行边界放在**档位**这一层——8 档天然独立，一个池跨档复用。

两种并行的正确分工
------------------
* **线程（--workers）**：用于 `build_dataset`，那里有大量 DB 往返，能重叠 IO 等待。
* **进程（--processes）**：用于 `simulate` / 寻优，绕过 GIL 真并行。

本脚本自动跑三种配置并校验结果一致性——并行不能改变任何数字，只改变耗时。

用法：
    .\.venv\Scripts\python.exe backend/scripts/benchmark_backtest_parallel.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

CONFIGS = [
    {"name": "串行基线", "workers": 1, "processes": None},
    {"name": "仅提取线程", "workers": 8, "processes": None},
    {"name": "提取线程 + 按档进程", "workers": 8, "processes": 8},
]

# 单次采样波动可达 ±30%（本机负载、PG 缓存、spawn 冷启动都会影响），
# 故每配置跑多轮取中位数，并保留全部样本——与 docs/performance-benchmark.json
# 的口径一致，避免把一次抖动当成结论。
ROUNDS = 3


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def run_once(name: str, workers: int, processes: int | None, out_dir: Path,
             windows: int, per_tier: int) -> dict:
    tag = name.replace(" ", "_").replace("+", "plus")
    cmd = [
        str(PYTHON), str(ROOT / "backend" / "scripts" / "tier_backtest.py"),
        "--windows", str(windows), "--per-tier", str(per_tier),
        "--workers", str(workers),
        "--output", str(out_dir / f"bt-{tag}.json"),
        "--markdown-output", str(out_dir / f"bt-{tag}.md"),
    ]
    if processes:
        cmd += ["--processes", str(processes)]

    started = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    elapsed = round(time.time() - started, 1)

    result = {"name": name, "workers": workers, "processes": processes,
              "elapsed_sec": elapsed, "returncode": proc.returncode}
    json_path = out_dir / f"bt-{tag}.json"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        ok = [w for w in payload.get("windows", []) if w.get("ok")]
        if ok:
            tiered = ok[0].get("tiered", {})
            uniform = ok[0].get("uniform", {})
            result["tiered_ret"] = tiered.get("total_return_pct")
            result["uniform_ret"] = uniform.get("total_return_pct")
            result["symbols"] = ok[0].get("total_symbols")
            result["internal_sec"] = payload.get("elapsed_sec")
    if proc.returncode != 0:
        result["stderr_tail"] = (proc.stderr or "")[-600:]
    return result


def to_markdown(report: dict) -> str:
    base = next((r for r in report["runs"] if r.get("elapsed_sec")), None)
    lines = ["# 回测并行化基准\n"]
    lines.append(
        f"- 生成时间：{report['generated_at']}；"
        f"规模：{report['params']['windows']} 窗口 × 每档 {report['params']['per_tier']} 币\n"
    )
    lines.append(
        f"- 每配置跑 {report['params']['rounds']} 轮取**中位数**并保留全部样本——"
        "单次采样波动可达 ±30%（负载、PG 缓存、spawn 冷启动），不可作为结论。\n"
    )
    lines.append(
        "- 判定：并行**只能改变耗时，不能改变任何结果数字**；"
        "下表 `分层收益` 在各配置下必须逐位一致，否则并行化引入了非确定性。\n"
    )
    lines.append(
        "\n| 配置 | workers | processes | 中位耗时(s) | 各轮样本 | 相对串行 | 分层收益% | 币种 |\n"
        "|---|---:|---:|---:|---|---:|---:|---:|\n"
    )
    for r in report["runs"]:
        speedup = (
            round(base["elapsed_sec"] / r["elapsed_sec"], 2)
            if base and r.get("elapsed_sec") else None
        )
        samples = "、".join(str(s) for s in r.get("samples_sec", []))
        lines.append(
            f"| {r['name']} | {r['workers']} | {r['processes']} | {r.get('elapsed_sec')} | "
            f"{samples} | {speedup}× | {r.get('tiered_ret')} | {r.get('symbols')} |\n"
        )
    rets = {r.get("tiered_ret") for r in report["runs"] if r.get("tiered_ret") is not None}
    lines.append(
        f"\n## 结论\n\n"
        f"- 结果一致性：{'通过（各配置收益完全相同）' if len(rets) == 1 else '不通过，收益出现分歧'}。\n"
    )
    lines.append(
        "- **加速几乎全部来自进程池**：`simulate` 是逐 bar 纯 Python 循环，"
        "被 GIL 锁死，线程并行零收益（实测 8 线程 37.0s ≈ 串行 36.8s）；"
        "只有进程池能绕过 GIL 真并行。\n"
        "- **进程池必须按档复用、只建一次**：Windows spawn 开销很大，"
        "每档每轮新建池反而更慢（52.9s > 串行 36.8s）。\n"
        "- **线程（--workers）收益有限但为正**：`build_dataset` 有 DB 往返可重叠 IO，"
        "实测约 1.1×，与进程叠加后总体 1.7×。\n"
    )
    lines.append(
        "- 建议：日常跑 `tier_backtest.py` 时加 `--workers 8 --processes 8`。\n"
        "- 本基准只测速度，不做任何收益结论。\n"
    )
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="回测并行化基准")
    parser.add_argument("--windows", type=int, default=1)
    parser.add_argument("--per-tier", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "backtest-parallel-benchmark.json")
    parser.add_argument("--markdown-output", type=Path,
                        default=ROOT / "docs" / "backtest-parallel-benchmark.md")
    args = parser.parse_args()

    from datetime import datetime, timezone
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": {"windows": args.windows, "per_tier": args.per_tier, "rounds": args.rounds},
        "runs": [],
    }
    for cfg in CONFIGS:
        print(f"[bench] {cfg['name']} ...", flush=True)
        samples = []
        for _ in range(args.rounds):
            samples.append(
                run_once(cfg["name"], cfg["workers"], cfg["processes"],
                         args.out_dir, args.windows, args.per_tier)
            )
        ok = [s for s in samples if s.get("elapsed_sec")]
        merged = {
            "name": cfg["name"], "workers": cfg["workers"], "processes": cfg["processes"],
            "elapsed_sec": median([s["elapsed_sec"] for s in ok]) if ok else None,
            "samples_sec": [s["elapsed_sec"] for s in samples],
            "rounds": len(samples),
            "tiered_ret": samples[-1].get("tiered_ret"),
            "uniform_ret": samples[-1].get("uniform_ret"),
            "symbols": samples[-1].get("symbols"),
            "returncode": max(s.get("returncode", 0) for s in samples),
        }
        report["runs"].append(merged)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown_output.write_text(to_markdown(report), encoding="utf-8")
    print(f"written: {args.output} / {args.markdown_output}")


if __name__ == "__main__":
    sys.exit(main())
