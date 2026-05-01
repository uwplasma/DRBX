from __future__ import annotations

import json
from pathlib import Path

import pytest

from jax_drb.validation import (
    ImpurityRadiationCampaignMetric,
    build_impurity_radiation_campaign,
    create_impurity_radiation_campaign_package,
)
from tests.ci_reference_fixtures import reference_root_or_ci_fixture


def test_create_impurity_radiation_campaign_package_writes_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "jax_drb.validation.impurity_radiation_campaign.build_impurity_radiation_campaign",
        lambda *, reference_root=None: (
            ImpurityRadiationCampaignMetric(
                name="demo_gate",
                kind="fraction",
                value=1.0,
                target=1.0,
                passed=True,
                notes="demo",
            ),
        ),
    )
    artifacts = create_impurity_radiation_campaign_package(output_root=tmp_path / "output")
    assert artifacts.summary_json_path.exists()
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert payload["family"] == "impurity_radiation_and_detachment_control"
    assert payload["passed_metric_count"] == 1


def test_build_impurity_radiation_campaign_reports_expected_neon_metrics(tmp_path: Path) -> None:
    reference_root = reference_root_or_ci_fixture(tmp_path)
    metrics = build_impurity_radiation_campaign(reference_root=reference_root)
    names = {metric.name for metric in metrics}
    assert "openadas_neon_full_bundle_finite_fraction" in names
    assert "tokamak_dthene_rhs_neon_density_exact" in names
    assert "tokamak_dthene_rhs_neon_pressure_exact" in names
    assert all(metric.passed for metric in metrics)
