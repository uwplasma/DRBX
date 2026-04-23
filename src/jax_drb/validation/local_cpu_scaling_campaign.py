from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE, Popen
import sys
from textwrap import dedent
from time import perf_counter

from matplotlib import pyplot as plt
import numpy as np

from .publication_plotting import annotate_bars, save_publication_figure, style_axis


@dataclass(frozen=True)
class LocalCpuScalingCampaignArtifacts:
    summary_json_path: Path
    summary_plot_png_path: Path


def create_local_cpu_scaling_campaign_package(
    *,
    output_root: str | Path,
    case_name: str = "tokamak_recycling_dthene_one_step",
    worker_counts: tuple[int, ...] = (1, 2, 4, 8),
    total_runs: int = 16,
    case_label: str = "local_cpu_scaling_campaign",
) -> LocalCpuScalingCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report = build_local_cpu_scaling_campaign_report(
        case_name=case_name,
        worker_counts=worker_counts,
        total_runs=total_runs,
    )
    summary_json_path = data_dir / f"{case_label}.json"
    summary_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary_plot_png_path = save_local_cpu_scaling_campaign_plot(report, images_dir / f"{case_label}.png")
    return LocalCpuScalingCampaignArtifacts(
        summary_json_path=summary_json_path,
        summary_plot_png_path=summary_plot_png_path,
    )


def build_local_cpu_scaling_campaign_report(
    *,
    case_name: str,
    worker_counts: tuple[int, ...],
    total_runs: int,
) -> dict[str, object]:
    ensemble = _benchmark_steady_state_ensemble_sweep(
        case_name=case_name,
        worker_counts=worker_counts,
        total_runs=total_runs,
    )
    return {
        "case": "local_cpu_scaling_campaign",
        "benchmark_case_name": case_name,
        "benchmark_case_label": "direct tokamak recycling one-step transient with neon",
        "steady_state_ensemble_sweep": ensemble,
        "profiling_note": (
            "The latest cProfile pass on the same heavy solve shows the dominant "
            "remaining costs in the production path are sparse finite-difference "
            "Jacobian assembly and recycling RHS/source assembly in the implicit "
            "residual path, not the older per-cell transport loops or the "
            "previously unvectorized parallel-gradient kernel."
        ),
        "recommendations": [
            "Use one Jacobian thread per worker and multiple local worker processes for batched heavy solves such as UQ, optimization, and parameter scans.",
            "Treat the steady-state ensemble speedup as the reviewer-facing local strong-scaling result on MacBook-class CPUs.",
            "Do not treat single-solve Jacobian threading as the main local scaling story on this hardware.",
            "Keep the committed ensemble moderate on thermally limited laptops; the current 16-solve artifact was retained because heavier local sweeps did not improve the curve on this machine.",
        ],
    }


def save_local_cpu_scaling_campaign_plot(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    ensemble = dict(report["steady_state_ensemble_sweep"])
    worker_counts = np.asarray(ensemble["worker_counts"], dtype=np.float64)
    ensemble_speedups = np.asarray(ensemble["steady_state_speedups"], dtype=np.float64)
    ideal_ensemble = worker_counts / worker_counts[0]
    efficiencies = 100.0 * ensemble_speedups / ideal_ensemble

    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)

    axes[0].plot(worker_counts, ensemble_speedups, marker="o", linewidth=2.8, color="#bb3e03", label="measured")
    axes[0].plot(worker_counts, ideal_ensemble, linestyle="--", linewidth=1.8, color="#ee9b00", label="ideal")
    style_axis(
        axes[0],
        title="Fixed-work steady-state speedup",
        xlabel="local worker processes",
        ylabel="speedup vs 1 worker",
        grid="both",
    )
    axes[0].legend(frameon=False, loc="upper left")
    for x_value, y_value in zip(worker_counts, ensemble_speedups, strict=False):
        axes[0].annotate(
            f"{y_value:.2f}x",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=9.5,
        )
    axes[0].text(
        0.03,
        0.04,
        f"Case: {report['benchmark_case_name']}\n"
        f"Total heavy solves: {ensemble['total_runs']}\n"
        f"1-worker steady-state baseline: {ensemble['steady_state_baseline_seconds']:.2f} s",
        transform=axes[0].transAxes,
        fontsize=9.2,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "#cccccc"},
    )

    x = np.arange(worker_counts.size, dtype=np.float64)
    bars = axes[1].bar(x, efficiencies, color="#0a9396", width=0.62)
    style_axis(
        axes[1],
        title="Parallel efficiency",
        xlabel="local worker processes",
        ylabel="efficiency (%)",
        grid="y",
    )
    axes[1].set_xticks(x, [str(int(value)) for value in worker_counts])
    axes[1].set_ylim(0.0, 110.0)
    annotate_bars(axes[1], x, efficiencies, fmt="{:.0f}%", fontsize=9.0)
    for bar in bars:
        bar.set_alpha(0.92)

    save_publication_figure(figure, target)
    return target

def _benchmark_steady_state_ensemble_sweep(
    *,
    case_name: str,
    worker_counts: tuple[int, ...],
    total_runs: int,
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for worker_count in worker_counts:
        counts = _distribute_runs(total_runs=total_runs, worker_count=worker_count)
        wall_start = perf_counter()
        worker_results = _run_ensemble_worker_group(case_name=case_name, worker_run_counts=counts)
        wall_elapsed = perf_counter() - wall_start
        steady_state_elapsed = max(float(result["elapsed_seconds"]) for result in worker_results)
        entries.append(
            {
                "worker_count": worker_count,
                "worker_run_counts": counts,
                "steady_state_seconds": steady_state_elapsed,
                "wall_seconds": wall_elapsed,
                "launch_overhead_seconds": max(0.0, wall_elapsed - steady_state_elapsed),
                "worker_results": worker_results,
            }
        )

    baseline_seconds = float(entries[0]["steady_state_seconds"])
    return {
        "worker_counts": [int(entry["worker_count"]) for entry in entries],
        "total_runs": total_runs,
        "entries": entries,
        "steady_state_baseline_seconds": baseline_seconds,
        "steady_state_speedups": [baseline_seconds / float(entry["steady_state_seconds"]) for entry in entries],
    }


def _distribute_runs(*, total_runs: int, worker_count: int) -> list[int]:
    counts = [total_runs // worker_count for _ in range(worker_count)]
    for index in range(total_runs % worker_count):
        counts[index] += 1
    return [count for count in counts if count > 0]


def _run_ensemble_worker_group(*, case_name: str, worker_run_counts: list[int]) -> list[dict[str, object]]:
    worker_code = dedent(
        """
        import json
        import os
        import sys
        import time
        from jax_drb.native.runner import run_curated_case
        from jax_drb.reference.paths import require_reference_root

        count = int(sys.argv[1])
        case_name = sys.argv[2]
        reference_root = require_reference_root()
        os.environ["JAX_DRB_FD_JACOBIAN_THREADS"] = "1"
        run_curated_case(case_name, reference_root=reference_root)
        start = time.perf_counter()
        for _ in range(count):
            run_curated_case(case_name, reference_root=reference_root)
        print(json.dumps({"count": count, "elapsed_seconds": time.perf_counter() - start}, sort_keys=True))
        """
    ).strip()
    env = dict(os.environ)
    repo_root = Path(__file__).resolve().parents[3]
    existing_pythonpath = env.get("PYTHONPATH", "")
    path_prefix = str(repo_root / "src")
    env["PYTHONPATH"] = path_prefix if not existing_pythonpath else f"{path_prefix}{os.pathsep}{existing_pythonpath}"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    processes: list[Popen[str]] = []
    try:
        for count in worker_run_counts:
            processes.append(
                Popen(
                    [sys.executable, "-c", worker_code, str(count), case_name],
                    stdout=PIPE,
                    stderr=PIPE,
                    text=True,
                    env=env,
                    cwd=repo_root,
                )
            )
        results: list[dict[str, object]] = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=1800)
            if process.returncode != 0:
                raise RuntimeError(
                    "local cpu scaling worker failed\n"
                    f"stdout:\n{stdout}\n"
                    f"stderr:\n{stderr}"
                )
            results.append(json.loads(stdout.strip().splitlines()[-1]))
        return results
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
