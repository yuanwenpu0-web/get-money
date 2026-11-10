"""Verify that ONNX Runtime DirectML really executes on the AMD Radeon iGPU.

Usage (from project root):
    .venv-dml\\Scripts\\python.exe scripts\\verify_directml.py

This script is read-only with respect to the trading system. It builds two
tiny fixed-shape ONNX graphs in a temp directory, runs them on the DML
execution provider (CPU fallback explicitly disabled) and on the CPU provider,
then compares numerics and latency.

Why it matters: onnxruntime-directml silently falls back to CPU for unsupported
operators. A successful import is NOT proof that the iGPU is being used.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover - environment broken
    print(json.dumps({"status": "IMPORT_FAILED", "error": repr(exc)}, indent=2))
    raise SystemExit(1)

# batch = Kronos sample_count, 512 = Kronos-small/base context length
B, S, H = 8, 512, 64
INV_SQRT2 = 0.7071067811865476


def _const(name: str, value) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)


def _build_model(use_erf: bool) -> onnx.ModelProto:
    """MatMul -> Add -> (GELU via Erf | Relu) -> Softmax -> MatMul."""
    rng = np.random.default_rng(20261110)
    w0 = rng.normal(0, 0.02, (H, H)).astype(np.float32)
    b0 = rng.normal(0, 0.01, (H,)).astype(np.float32)
    w1 = rng.normal(0, 0.02, (H, 1)).astype(np.float32)

    initializers = [_const("W0", w0), _const("b0", b0), _const("W1", w1)]
    nodes = [
        helper.make_node("MatMul", ["X", "W0"], ["t1"]),
        helper.make_node("Add", ["t1", "b0"], ["t2"]),
    ]

    if use_erf:
        initializers += [
            _const("half", 0.5),
            _const("one", 1.0),
            _const("inv_sqrt2", INV_SQRT2),
        ]
        nodes += [
            helper.make_node("Mul", ["t2", "inv_sqrt2"], ["scaled"]),
            helper.make_node("Erf", ["scaled"], ["erfed"]),
            helper.make_node("Add", ["erfed", "one"], ["one_plus"]),
            helper.make_node("Mul", ["t2", "one_plus"], ["times"]),
            helper.make_node("Mul", ["times", "half"], ["t3"]),
        ]
    else:
        nodes.append(helper.make_node("Relu", ["t2"], ["t3"]))

    nodes += [
        helper.make_node("Softmax", ["t3"], ["sm"], axis=-1),
        helper.make_node("MatMul", ["sm", "W1"], ["Y"]),
    ]

    graph = helper.make_graph(
        nodes,
        "dml_probe_erf" if use_erf else "dml_probe_relu",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [B, S, H])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [B, S, 1])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    # ORT 1.24.x (including the DirectML wheel) supports ONNX IR <= 13, while
    # onnx >= 1.23 emits IR 14 by default. Clamp it, otherwise the graph is
    # rejected at load time -- the same fix is needed for torch.onnx exports.
    model.ir_version = min(model.ir_version, 13)
    onnx.checker.check_model(model)
    return model


def _make_session(path: Path, providers: list, disable_fallback: bool = False):
    opts = ort.SessionOptions()
    # DML owns its allocator; mem pattern and parallel execution must be off.
    opts.enable_mem_pattern = False
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.log_severity_level = 2
    if disable_fallback:
        # Force a hard error instead of a silent CPU fallback.
        opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(str(path), sess_options=opts, providers=providers)


def _time_run(session, x: np.ndarray, warmup: int = 3, repeat: int = 20):
    for _ in range(warmup):
        session.run(None, {"X": x})
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = session.run(None, {"X": x})[0]
        samples.append((time.perf_counter() - t0) * 1000.0)
    return out, statistics.median(samples), min(samples)


def _installed_ort() -> dict[str, str]:
    """Detect a standard onnxruntime wheel shadowing the DirectML wheel."""
    import importlib.metadata as md

    found = {}
    for name in ("onnxruntime", "onnxruntime-directml", "onnxruntime-gpu"):
        try:
            found[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    return found


def _profile_kernel_providers(path: Path, x: np.ndarray, tmp: Path) -> dict:
    """Use ORT's own profiler as hard evidence of which EP owned each kernel."""
    opts = ort.SessionOptions()
    opts.enable_mem_pattern = False
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_profiling = True
    opts.profile_file_prefix = str(tmp / "dml_profile")
    opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    sess = ort.InferenceSession(
        str(path), sess_options=opts, providers=["DmlExecutionProvider"]
    )
    sess.run(None, {"X": x})
    profile_file = sess.end_profiling()
    data = json.loads(Path(profile_file).read_text(encoding="utf-8"))
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    counts: dict[str, int] = {}
    for ev in events:
        args = ev.get("args") or {}
        provider = args.get("provider") or args.get("ep") or ""
        if provider:
            counts[provider] = counts.get(provider, 0) + 1
    blob = json.dumps(data)
    return {
        "kernel_events_by_provider": counts,
        "trace_mentions_dml": "DmlExecutionProvider" in blob,
        "trace_events": len(events),
    }


def main() -> int:
    global B, S, H
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=B, help="sample_count dimension")
    parser.add_argument("--seq", type=int, default=S, help="context length")
    parser.add_argument("--hidden", type=int, default=H, help="d_model")
    parser.add_argument("--repeat", type=int, default=20, help="timing repeats")
    args, _ = parser.parse_known_args()
    B, S, H = args.batch, args.seq, args.hidden

    available = ort.get_available_providers()
    report = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "onnxruntime_file": getattr(ort, "__file__", None),
        "available_providers": available,
        "shape": {"batch": B, "seq": S, "hidden": H},
        "installed_ort_distributions": _installed_ort(),
        "status": "FAIL",
    }
    try:
        report["ort_device"] = ort.get_device()
    except Exception:
        report["ort_device"] = "unavailable"

    print("== environment ==")
    for k in (
        "python", "numpy", "onnx", "onnxruntime", "ort_device",
        "available_providers", "installed_ort_distributions",
    ):
        print(f"  {k}: {report[k]}")

    if not any("dml" in p.lower() for p in available):
        report["error"] = (
            "DmlExecutionProvider absent. The standard 'onnxruntime' wheel is "
            "probably shadowing onnxruntime-directml in this interpreter."
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, (B, S, H)).astype(np.float32)

    with tempfile.TemporaryDirectory(prefix="dml_probe_") as tmp:
        tmp = Path(tmp)
        # --- model A: strict numeric parity, CPU fallback disabled ---
        path_a = tmp / "probe_relu.onnx"
        onnx.save(_build_model(use_erf=False), path_a)
        try:
            dml = _make_session(path_a, ["DmlExecutionProvider"], disable_fallback=True)
            report["dml_no_fallback_session"] = True
        except Exception as exc:
            report["dml_no_fallback_session"] = False
            report["dml_no_fallback_error"] = repr(exc)[:400]
            dml = _make_session(path_a, ["DmlExecutionProvider"])

        cpu = _make_session(path_a, ["CPUExecutionProvider"])
        report["dml_active_providers"] = dml.get_providers()
        report["cpu_active_providers"] = cpu.get_providers()

        y_dml, dml_ms, dml_min = _time_run(dml, x, repeat=args.repeat)
        y_cpu, cpu_ms, cpu_min = _time_run(cpu, x, repeat=args.repeat)
        max_diff = float(np.max(np.abs(y_dml - y_cpu)))
        parity = bool(np.allclose(y_dml, y_cpu, rtol=1e-3, atol=1e-5))
        report["probe_relu"] = {
            "max_abs_diff_vs_cpu": max_diff,
            "numeric_parity": parity,
            "dml_median_ms": round(dml_ms, 3),
            "dml_min_ms": round(dml_min, 3),
            "cpu_median_ms": round(cpu_ms, 3),
            "cpu_min_ms": round(cpu_min, 3),
            "finite": bool(np.isfinite(y_dml).all()),
        }

        # --- model B: Erf-based GELU, the activation Kronos exports ---
        path_b = tmp / "probe_erf.onnx"
        onnx.save(_build_model(use_erf=True), path_b)
        try:
            dml_b = _make_session(path_b, ["DmlExecutionProvider"], disable_fallback=True)
            y_b, b_ms, _ = _time_run(dml_b, x, warmup=2, repeat=min(8, args.repeat))
            report["probe_gelu_erf"] = {
                "supported_on_dml_without_cpu_fallback": True,
                "finite": bool(np.isfinite(y_b).all()),
                "median_ms": round(b_ms, 3),
            }
        except Exception as exc:
            report["probe_gelu_erf"] = {
                "supported_on_dml_without_cpu_fallback": False,
                "error": repr(exc)[:400],
            }

        # --- ORT profiler: which execution provider actually owned the kernels ---
        try:
            report["profiling"] = _profile_kernel_providers(path_a, x, tmp)
        except Exception as exc:
            report["profiling"] = {"unavailable": repr(exc)[:400]}

    ok = (
        report.get("dml_no_fallback_session") is True
        and parity
        and report["probe_relu"]["finite"]
        and "DmlExecutionProvider" in report["dml_active_providers"]
    )
    report["status"] = "PASS" if ok else "FAIL"
    print("\n== result ==")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
