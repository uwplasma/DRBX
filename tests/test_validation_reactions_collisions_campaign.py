from __future__ import annotations

import json
from pathlib import Path

import pytest

from jax_drb.reference.paths import default_reference_root
from jax_drb.validation.reactions_collisions_campaign import (
    ReactionsCollisionsCampaignMetric,
    build_reactions_collisions_campaign,
    create_reactions_collisions_campaign_package,
)


def test_create_reactions_collisions_campaign_package_writes_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jax_drb.validation.reactions_collisions_campaign.build_reactions_collisions_campaign",
        lambda **kwargs: (
            ReactionsCollisionsCampaignMetric(
                name="metric_a",
                kind="ratio",
                value=1.0,
                target=1.0,
                passed=True,
                notes="ok",
            ),
            ReactionsCollisionsCampaignMetric(
                name="metric_b",
                kind="relative_error",
                value=0.0,
                target=1.0e-12,
                passed=True,
                notes="ok",
            ),
        ),
    )
    artifacts = create_reactions_collisions_campaign_package(
        output_root=tmp_path / "output",
        single_species_input=tmp_path / "single.inp",
        multispecies_input=tmp_path / "multi.inp",
    )
    assert artifacts.summary_json_path.exists()
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert payload["family"] == "reactions_collisions_and_atomic_data"
    assert payload["passed_metric_count"] == 2


def test_build_reactions_collisions_campaign_passes_reference_checks() -> None:
    reference_root = default_reference_root()
    if reference_root is None:
        pytest.skip("external reference decks are not available")
    metrics = build_reactions_collisions_campaign(
        single_species_input=reference_root / "tests" / "integrated" / "1D-recycling" / "data" / "BOUT.inp",
        multispecies_input=reference_root / "tests" / "integrated" / "1D-recycling-dthe" / "data" / "BOUT.inp",
    )
    assert [metric.name for metric in metrics] == [
        "single_species_atom_cx_matches_same_species_formula",
        "multispecies_cross_isotope_cx_fraction",
        "species_rate_multiplier_ratio",
        "ionisation_rate_matches_reaction_diagnostic",
        "ion_parallel_viscosity_collisionality_closure",
        "openadas_neon_rate_bundle_finite_fraction",
    ]
    assert all(metric.passed for metric in metrics)
