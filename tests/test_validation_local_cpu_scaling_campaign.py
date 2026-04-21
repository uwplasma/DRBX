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
        "profiling_note": "synthetic",
        "steady_state_ensemble_sweep": {
            "worker_counts": [1, 2, 4],
            "total_runs": 24,
            "steady_state_baseline_seconds": 36.0,
            "steady_state_speedups": [1.0, 1.95, 3.65],
            "entries": [
                {"worker_count": 1, "worker_run_counts": [24], "steady_state_seconds": 36.0, "wall_seconds": 38.0, "launch_overhead_seconds": 2.0, "worker_results": [{"count": 24, "elapsed_seconds": 36.0}]},
                {"worker_count": 2, "worker_run_counts": [12, 12], "steady_state_seconds": 18.46, "wall_seconds": 20.2, "launch_overhead_seconds": 1.74, "worker_results": [{"count": 12, "elapsed_seconds": 18.46}, {"count": 12, "elapsed_seconds": 18.4}]},
                {"worker_count": 4, "worker_run_counts": [6, 6, 6, 6], "steady_state_seconds": 9.86, "wall_seconds": 12.2, "launch_overhead_seconds": 2.34, "worker_results": [{"count": 6, "elapsed_seconds": 9.86}]},
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
