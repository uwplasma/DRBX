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

from ..native.runner import run_curated_case
from ..reference.paths import require_reference_root


@dataclass(frozen=True)
class LocalCpuScalingCampaignArtifacts:
    summary_json_path: Path
    summary_plot_png_path: Path


def create_local_cpu_scaling_campaign_package(
    *,
    output_root: str | Path,
    case_name: str = "tokamak_recycling_dthe_one_step",
    thread_counts: tuple[int, ...] = (1, 2, 4, 8),
    worker_counts: tuple[int, ...] = (1, 2, 4, 8),
    total_runs: int = 8,
    repeats: int = 3,
    case_label: str = "local_cpu_scaling_campaign",
) -> LocalCpuScalingCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report = build_local_cpu_scaling_campaign_report(
        case_name=case_name,
        thread_counts=thread_counts,
        worker_counts=worker_counts,
        total_runs=total_runs,
        repeats=repeats,
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
    thread_counts: tuple[int, ...],
    worker_counts: tuple[int, ...],
    total_runs: int,
    repeats: int,
) -> dict[str, object]:
    reference_root = require_reference_root()
    single_solve = _benchmark_thread_sweep(
        case_name=case_name,
        reference_root=reference_root,
        thread_counts=thread_counts,
        repeats=repeats,
    )
    ensemble = _benchmark_steady_state_ensemble_sweep(
        case_name=case_name,
        worker_counts=worker_counts,
        total_runs=total_runs,
    )
    return {
        "case": "local_cpu_scaling_campaign",
        "benchmark_case_name": case_name,
        "benchmark_case_label": "direct tokamak recycling one-step transient",
        "single_solve_thread_sweep": single_solve,
        "steady_state_ensemble_sweep": ensemble,
        "recommendations": [
            "Use JAX_DRB_FD_JACOBIAN_THREADS=2 or 4 for a single heavy recycling solve on this MacBook-class CPU.",
            "Use one Jacobian thread per worker and multiple local worker processes for batched heavy solves such as UQ, optimization, and parameter scans.",
            "Treat the ensemble speedup as the reviewer-facing local strong-scaling result; treat the single-solve thread curve as a bounded local acceleration knob.",
        ],
    }


def save_local_cpu_scaling_campaign_plot(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    single = dict(report["single_solve_thread_sweep"])
    ensemble = dict(report["steady_state_ensemble_sweep"])
    thread_counts = np.asarray(single["thread_counts"], dtype=np.float64)
    single_speedups = np.asarray(single["speedups"], dtype=np.float64)
    worker_counts = np.asarray(ensemble["worker_counts"], dtype=np.float64)
    ensemble_speedups = np.asarray(ensemble["steady_state_speedups"], dtype=np.float64)
    ideal_single = thread_counts / thread_counts[0]
    ideal_ensemble = worker_counts / worker_counts[0]

    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), constrained_layout=True)

    axes[0].plot(thread_counts, single_speedups, marker="o", linewidth=2.2, color="#005f73", label="measured")
    axes[0].plot(thread_counts, ideal_single, linestyle="--", linewidth=1.4, color="#94d2bd", label="ideal")
    axes[0].set_xlabel("Jacobian worker threads")
    axes[0].set_ylabel("speedup vs 1 thread")
    axes[0].set_title("Single heavy solve on local CPUs")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].text(
        0.03,
        0.04,
        f"Case: {report['benchmark_case_name']}\n"
        f"Best 1-thread steady solve: {single['baseline_seconds']:.2f} s",
        transform=axes[0].transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "#cccccc"},
    )

    axes[1].plot(worker_counts, ensemble_speedups, marker="o", linewidth=2.2, color="#bb3e03", label="measured")
    axes[1].plot(worker_counts, ideal_ensemble, linestyle="--", linewidth=1.4, color="#ee9b00", label="ideal")
    axes[1].set_xlabel("Local worker processes")
    axes[1].set_ylabel("steady-state speedup vs 1 worker")
    axes[1].set_title("Fixed-work local ensemble scaling")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].text(
        0.03,
        0.04,
        f"Total heavy solves: {ensemble['total_runs']}\n"
        f"1-worker steady-state baseline: {ensemble['steady_state_baseline_seconds']:.2f} s",
        transform=axes[1].transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "#cccccc"},
    )

    figure.suptitle("Local CPU scaling on a heavy direct-tokamak recycling workload", fontsize=14)
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def _benchmark_thread_sweep(
    *,
    case_name: str,
    reference_root: Path,
    thread_counts: tuple[int, ...],
    repeats: int,
) -> dict[str, object]:
    original_threads = os.environ.get("JAX_DRB_FD_JACOBIAN_THREADS")
    entries: list[dict[str, object]] = []
    try:
        for thread_count in thread_counts:
            os.environ["JAX_DRB_FD_JACOBIAN_THREADS"] = str(thread_count)
            run_curated_case(case_name, reference_root=reference_root)
            timings: list[float] = []
            for _ in range(repeats):
                start = perf_counter()
                run_curated_case(case_name, reference_root=reference_root)
                timings.append(perf_counter() - start)
            entries.append(
                {
                    "thread_count": thread_count,
                    "timings_seconds": timings,
                    "best_seconds": float(min(timings)),
                    "mean_seconds": float(np.mean(np.asarray(timings, dtype=np.float64))),
                }
            )
    finally:
        if original_threads is None:
            os.environ.pop("JAX_DRB_FD_JACOBIAN_THREADS", None)
        else:
            os.environ["JAX_DRB_FD_JACOBIAN_THREADS"] = original_threads

    baseline_seconds = float(entries[0]["best_seconds"])
    return {
        "thread_counts": [int(entry["thread_count"]) for entry in entries],
        "entries": entries,
        "baseline_seconds": baseline_seconds,
        "speedups": [baseline_seconds / float(entry["best_seconds"]) for entry in entries],
    }


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
