from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jax_drb.validation import (
    ImpurityRadiationCampaignMetric,
    build_impurity_radiation_campaign,
    create_impurity_radiation_campaign_package,
)
import jax_drb.validation.impurity_radiation_campaign as impurity_campaign_mod
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


def test_build_impurity_radiation_campaign_uses_committed_neon_baseline_without_external_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_variables = {
        "Nne+": np.asarray([[[1.0, 1.25]]], dtype=np.float64),
        "Pne+": np.asarray([[[2.0, 2.5]]], dtype=np.float64),
        "Pe": np.asarray([[[3.0, 3.5]]], dtype=np.float64),
    }
    actual_payload = SimpleNamespace(variables={name: value.copy() for name, value in expected_variables.items()})

    monkeypatch.setattr(
        impurity_campaign_mod,
        "_load_openadas_rate",
        lambda *_args: (
            np.ones((2, 2), dtype=np.float64),
            np.ones((2, 2), dtype=np.float64),
            np.ones(2, dtype=np.float64),
            np.ones(2, dtype=np.float64),
            1.0,
        ),
    )
    monkeypatch.setattr(
        impurity_campaign_mod,
        "_openadas_reaction_rate",
        lambda *_args: np.asarray([2.0], dtype=np.float64),
    )
    monkeypatch.setattr(
        impurity_campaign_mod,
        "_openadas_energy_loss",
        lambda *_args, **_kwargs: np.asarray([0.5], dtype=np.float64),
    )
    monkeypatch.setattr(
        impurity_campaign_mod,
        "load_portable_array_payload",
        lambda _path: {"variables": expected_variables},
    )
    monkeypatch.setattr(
        impurity_campaign_mod,
        "run_curated_case",
        lambda _case_name, *, reference_root: actual_payload,
    )

    metrics = build_impurity_radiation_campaign(reference_root=tmp_path)

    names = [metric.name for metric in metrics]
    assert names == [
        "openadas_neon_full_bundle_finite_fraction",
        "neon_ionisation_rate_positive",
        "neon_recombination_rate_positive",
        "neon_openadas_radiation_terms_finite",
        "tokamak_dthene_rhs_neon_density_exact",
        "tokamak_dthene_rhs_neon_pressure_exact",
        "tokamak_dthene_rhs_electron_pressure_exact",
    ]
    assert all(metric.passed for metric in metrics)
    assert {metric.kind for metric in metrics} >= {"fraction", "scalar", "max_abs_error"}
