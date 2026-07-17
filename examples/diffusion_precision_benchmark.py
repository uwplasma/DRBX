"""Float64 vs float32 runtime benchmark of the native diffusion run path.

Starting from the restartable-diffusion TOML template, the script writes one
precision-specific input deck per precision (rewriting only the ``[runtime]``
precision entry with the public ``dkx.config.rewrite_input_precision``
helper), runs ``python -m dkx`` on each deck in a fresh subprocess
(``REPEATS`` times per precision, each precision with its own compilation
cache directory so the first repeat shows compile cost), and records wall-clock
times plus any float64-truncation warnings emitted by the float32 path.

It prints per-repeat timings and writes (relative to the current working
directory)
``docs/data/runtime_precision_benchmark_artifacts/data/diffusion_precision_analysis.json``
and
``docs/data/runtime_precision_benchmark_artifacts/images/diffusion_precision_elapsed.png``.

Run from the repository root:

    PYTHONPATH=src python examples/diffusion_precision_benchmark.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from dkx.config import rewrite_input_precision

# --- PARAMETERS ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_TEMPLATE = REPO_ROOT / "examples" / "inputs" / "restartable_diffusion.toml"  # deck to benchmark
OUTPUT_ROOT = Path("docs/data/runtime_precision_benchmark_artifacts")  # artifact root (cwd-relative)
CASE_NAME = "diffusion_precision"  # artifact/case prefix
PRECISIONS = ("float64", "float32")  # runtime precisions to compare
REPEATS = 2                          # runs per precision; repeat 1 includes compile time


def run_precision_case(precision: str, repeat_index: int, input_path: Path, cache_dir: Path) -> dict[str, Any]:
    """Run one dkx subprocess and collect timing plus warning counts."""

    run_dir = OUTPUT_ROOT / f"{precision}_run_{repeat_index + 1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "dkx",
        str(input_path),
        "--case-name",
        f"{CASE_NAME}_{precision}",
        "--output-dir",
        str(run_dir),
        "--quiet",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["DKX_CACHE_DIR"] = str(cache_dir)

    started = time.perf_counter()
    completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"precision run failed for {precision} repeat {repeat_index + 1} with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )

    warning_lines = [line for line in completed.stderr.splitlines() if "requested dtype float64" in line]
    run_log_path = run_dir / f"{CASE_NAME}_{precision}_run_log.json"
    run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
    return {
        "precision": precision,
        "repeat": repeat_index + 1,
        "elapsed_seconds": elapsed,
        "warning_count": len(warning_lines),
        "run_log": run_log,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "run_dir": str(run_dir),
    }


# --- run every precision in its own subprocess ------------------------------------
print("Requested precision benchmark")
print(f"  input_template: {INPUT_TEMPLATE}")
print(f"  output_root:    {OUTPUT_ROOT}")
print(f"  repeats:        {REPEATS}")

template_text = INPUT_TEMPLATE.read_text(encoding="utf-8")
input_dir = OUTPUT_ROOT / "input"
input_dir.mkdir(parents=True, exist_ok=True)

results: list[dict[str, Any]] = []
for precision in PRECISIONS:
    input_path = input_dir / f"{CASE_NAME}_{precision}.toml"
    input_path.write_text(rewrite_input_precision(template_text, precision), encoding="utf-8")
    print(f"\nBenchmarking {precision}")
    print(f"  input_file: {input_path}")
    with tempfile.TemporaryDirectory(prefix=f"dkx_precision_{precision}_") as cache_dir_raw:
        cache_dir = Path(cache_dir_raw)
        for repeat_index in range(REPEATS):
            result = run_precision_case(precision, repeat_index, input_path, cache_dir)
            results.append(result)
            print(f"  repeat {result['repeat']}: {result['elapsed_seconds']:.3f} s, "
                  f"{result['warning_count']} float64-truncation warning(s), "
                  f"run_dir {result['run_dir']}")

# --- write the analysis JSON ------------------------------------------------------
data_dir = OUTPUT_ROOT / "data"
data_dir.mkdir(parents=True, exist_ok=True)
grouped: dict[str, list[dict[str, Any]]] = {precision: [] for precision in PRECISIONS}
for item in results:
    grouped[item["precision"]].append(item)
summary: dict[str, Any] = {}
for precision, entries in grouped.items():
    elapsed = [float(entry["elapsed_seconds"]) for entry in entries]
    warning_counts = [int(entry["warning_count"]) for entry in entries]
    summary[precision] = {
        "elapsed_seconds": elapsed,
        "best_seconds": min(elapsed),
        "last_seconds": elapsed[-1],
        "warning_counts": warning_counts,
        "last_warning_count": warning_counts[-1],
        "runtime_precision": entries[-1]["run_log"]["run_configuration"]["runtime"]["precision"],
    }
analysis_payload = {
    "case_name": CASE_NAME,
    "input_template": str(INPUT_TEMPLATE),
    "repeats": REPEATS,
    "summary": summary,
    "speedup_last_float32_vs_float64": (
        summary["float64"]["last_seconds"] / summary["float32"]["last_seconds"]
        if summary["float32"]["last_seconds"] > 0.0
        else None
    ),
}
analysis_path = data_dir / f"{CASE_NAME}_analysis.json"
analysis_path.write_text(json.dumps(analysis_payload, indent=2, sort_keys=True), encoding="utf-8")

# --- timing plot ------------------------------------------------------------------
images_dir = OUTPUT_ROOT / "images"
images_dir.mkdir(parents=True, exist_ok=True)
figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
for precision, color in (("float64", "tab:blue"), ("float32", "tab:orange")):
    entries = [item for item in results if item["precision"] == precision]
    x_values = [item["repeat"] for item in entries]
    y_values = [item["elapsed_seconds"] for item in entries]
    axis.plot(x_values, y_values, marker="o", linewidth=2.0, color=color, label=precision)
axis.set_xlabel("repeat")
axis.set_ylabel("elapsed seconds")
axis.set_title("Diffusion runtime precision benchmark")
axis.grid(alpha=0.3)
axis.legend()
plot_path = images_dir / f"{CASE_NAME}_elapsed.png"
figure.savefig(plot_path, dpi=180)
plt.close(figure)

print("\nGenerated artifacts")
print(f"  analysis_json: {analysis_path}")
print(f"  elapsed_plot:  {plot_path}")
