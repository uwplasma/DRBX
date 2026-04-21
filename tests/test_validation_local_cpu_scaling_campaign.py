from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation.local_cpu_scaling_campaign import (
    create_local_cpu_scaling_campaign_package,
    save_local_cpu_scaling_campaign_plot,
)


def _synthetic_report() -> dict[str, object]:
    return {
        "case": "local_cpu_scaling_campaign",
        "benchmark_case_name": "tokamak_recycling_dthe_one_step",
        "single_solve_thread_sweep": {
            "thread_counts": [1, 2, 4],
            "baseline_seconds": 2.0,
            "speedups": [1.0, 1.3, 1.35],
            "entries": [
                {"thread_count": 1, "timings_seconds": [2.0], "best_seconds": 2.0, "mean_seconds": 2.0},
                {"thread_count": 2, "timings_seconds": [1.54], "best_seconds": 1.54, "mean_seconds": 1.54},
                {"thread_count": 4, "timings_seconds": [1.48], "best_seconds": 1.48, "mean_seconds": 1.48},
            ],
        },
        "steady_state_ensemble_sweep": {
            "worker_counts": [1, 2, 4],
            "total_runs": 8,
            "steady_state_baseline_seconds": 12.0,
            "steady_state_speedups": [1.0, 1.9, 3.5],
            "entries": [
                {"worker_count": 1, "worker_run_counts": [8], "steady_state_seconds": 12.0, "wall_seconds": 13.0, "launch_overhead_seconds": 1.0, "worker_results": [{"count": 8, "elapsed_seconds": 12.0}]},
                {"worker_count": 2, "worker_run_counts": [4, 4], "steady_state_seconds": 6.3, "wall_seconds": 7.0, "launch_overhead_seconds": 0.7, "worker_results": [{"count": 4, "elapsed_seconds": 6.3}, {"count": 4, "elapsed_seconds": 6.2}]},
                {"worker_count": 4, "worker_run_counts": [2, 2, 2, 2], "steady_state_seconds": 3.43, "wall_seconds": 4.0, "launch_overhead_seconds": 0.57, "worker_results": [{"count": 2, "elapsed_seconds": 3.43}]},
            ],
        },
        "recommendations": ["synthetic"],
    }


def test_save_local_cpu_scaling_campaign_plot_writes_png(tmp_path: Path) -> None:
    plot_path = save_local_cpu_scaling_campaign_plot(_synthetic_report(), tmp_path / "scaling.png")
    assert plot_path.exists()


def test_create_local_cpu_scaling_campaign_package_writes_summary_and_plot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "jax_drb.validation.local_cpu_scaling_campaign.build_local_cpu_scaling_campaign_report",
        lambda **_: _synthetic_report(),
    )
    artifacts = create_local_cpu_scaling_campaign_package(output_root=tmp_path / "artifacts")
    assert artifacts.summary_json_path.exists()
    assert artifacts.summary_plot_png_path.exists()
    payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert payload["case"] == "local_cpu_scaling_campaign"
    assert payload["benchmark_case_name"] == "tokamak_recycling_dthe_one_step"
