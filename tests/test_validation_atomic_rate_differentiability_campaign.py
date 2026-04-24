from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jax_drb.validation.atomic_rate_differentiability_campaign import (
    build_atomic_rate_differentiability_campaign_report,
    create_atomic_rate_differentiability_campaign_package,
)


def test_atomic_rate_differentiability_campaign_passes_derivative_gates() -> None:
    report, arrays = build_atomic_rate_differentiability_campaign_report(point_count=48)

    assert report["case"] == "atomic_rate_differentiability_campaign"
    assert report["passed_metric_count"] == report["metric_count"]
    assert np.all(np.isfinite(arrays["temperature_ev"]))
    for metric in report["metrics"]:
        assert metric["passed"] is True
        assert metric["max_abs_derivative_error"] < 1.0e-6


def test_create_atomic_rate_differentiability_campaign_package_writes_artifacts(tmp_path: Path) -> None:
    artifacts = create_atomic_rate_differentiability_campaign_package(output_root=tmp_path / "artifacts")

    assert artifacts.summary_json_path.exists()
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    arrays = np.load(artifacts.arrays_npz_path)
    assert payload["passed_metric_count"] == payload["metric_count"]
    assert "amjuel_d_ionisation_rate" in arrays
    assert "openadas_ne_ionisation_autodiff_dlograte_dlogte" in arrays
    assert "hydrogen_charge_exchange_derivative_abs_error" in arrays
