from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PrecisionBenchmarkSettings:
    input_template: Path
    output_root: Path
    case_name: str
    repeats: int
    quiet: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the current native diffusion run path in float64 and float32. "
            "The script writes precision-specific TOML decks, runs jax_drb in separate "
            "subprocesses, saves JSON analysis, and renders a small Matplotlib timing plot."
        )
    )
    parser.add_argument(
        "--input-template",
        type=Path,
        default=repo_root / "examples" / "inputs" / "restartable_diffusion.toml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "docs" / "data" / "runtime_precision_benchmark_artifacts",
    )
    parser.add_argument("--case-name", default="diffusion_precision")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> PrecisionBenchmarkSettings:
    return PrecisionBenchmarkSettings(
        input_template=args.input_template,
        output_root=args.output_root,
        case_name=args.case_name,
        repeats=args.repeats,
        quiet=args.quiet,
    )


def print_section(settings: PrecisionBenchmarkSettings, title: str) -> None:
    if settings.quiet:
        return
    print(f"\n{title}")
    print("-" * len(title))


def print_mapping(settings: PrecisionBenchmarkSettings, mapping: dict[str, Any]) -> None:
    if settings.quiet:
        return
    for key, value in mapping.items():
        print(f"  {key}: {value}")


def build_precision_input(template_text: str, precision: str) -> str:
    lines = template_text.splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("precision ="):
            lines[index] = f'precision = "{precision}"'
            return "\n".join(lines) + "\n"
    raise ValueError("Template TOML is missing a [runtime] precision entry")


def write_precision_input(settings: PrecisionBenchmarkSettings, precision: str) -> Path:
    input_dir = settings.output_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    template_text = settings.input_template.read_text(encoding="utf-8")
    input_path = input_dir / f"{settings.case_name}_{precision}.toml"
    input_path.write_text(build_precision_input(template_text, precision), encoding="utf-8")
    return input_path


def run_precision_case(
    settings: PrecisionBenchmarkSettings,
    precision: str,
    repeat_index: int,
    input_path: Path,
    *,
    cache_dir: Path,
) -> dict[str, Any]:
    run_dir = settings.output_root / f"{precision}_run_{repeat_index + 1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "jax_drb",
        str(input_path),
        "--case-name",
        f"{settings.case_name}_{precision}",
        "--output-dir",
        str(run_dir),
        "--quiet",
    ]
    env = dict(os.environ)
    repo_root = _repo_root()
    env["PYTHONPATH"] = str(repo_root / "src")
    env["JAX_DRB_CACHE_DIR"] = str(cache_dir)

    started = time.perf_counter()
    completed = subprocess.run(command, cwd=repo_root, env=env, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"precision run failed for {precision} repeat {repeat_index + 1} with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )

    warning_lines = [line for line in completed.stderr.splitlines() if "requested dtype float64" in line]
    run_log_path = run_dir / f"{settings.case_name}_{precision}_run_log.json"
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


def write_analysis_json(settings: PrecisionBenchmarkSettings, results: list[dict[str, Any]]) -> Path:
    data_dir = settings.output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {"float64": [], "float32": []}
    for item in results:
        grouped[item["precision"]].append(item)
    summary = {}
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
    payload = {
        "case_name": settings.case_name,
        "input_template": str(settings.input_template),
        "repeats": settings.repeats,
        "summary": summary,
        "speedup_last_float32_vs_float64": (
            summary["float64"]["last_seconds"] / summary["float32"]["last_seconds"]
            if summary["float32"]["last_seconds"] > 0.0
            else None
        ),
    }
    path = data_dir / f"{settings.case_name}_analysis.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def plot_elapsed_times(settings: PrecisionBenchmarkSettings, results: list[dict[str, Any]]) -> Path:
    images_dir = settings.output_root / "images"
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
    path = images_dir / f"{settings.case_name}_elapsed.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> int:
    settings = build_settings(parse_args())
    print_section(settings, "Requested Precision Benchmark")
    print_mapping(
        settings,
        {
            "input_template": settings.input_template,
            "output_root": settings.output_root,
            "repeats": settings.repeats,
        },
    )

    results: list[dict[str, Any]] = []
    for precision in ("float64", "float32"):
        input_path = write_precision_input(settings, precision)
        print_section(settings, f"Benchmarking {precision}")
        print_mapping(settings, {"input_file": input_path})
        with tempfile.TemporaryDirectory(prefix=f"jax_drb_precision_{precision}_") as cache_dir_raw:
            cache_dir = Path(cache_dir_raw)
            for repeat_index in range(settings.repeats):
                result = run_precision_case(
                    settings,
                    precision,
                    repeat_index,
                    input_path,
                    cache_dir=cache_dir,
                )
                results.append(result)
                print_mapping(
                    settings,
                    {
                        "repeat": result["repeat"],
                        "elapsed_seconds": f"{result['elapsed_seconds']:.3f}",
                        "warning_count": result["warning_count"],
                        "run_dir": result["run_dir"],
                    },
                )

    analysis_path = write_analysis_json(settings, results)
    plot_path = plot_elapsed_times(settings, results)
    print_section(settings, "Generated Artifacts")
    print_mapping(settings, {"analysis_json": analysis_path, "elapsed_plot": plot_path})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
