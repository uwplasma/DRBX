from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from jax_drb.reference.paths import default_reference_root
from jax_drb.validation.reactions_collisions_campaign import (
    ReactionsCollisionsCampaignMetric,
    build_reactions_collisions_campaign,
    create_reactions_collisions_campaign_package,
)
import jax_drb.validation.reactions_collisions_campaign as reactions_campaign_mod


def _synthetic_reactions_context(*, multispecies: bool = False) -> dict[str, object]:
    mesh = SimpleNamespace(
        xstart=0,
        ystart=1,
        yend=3,
        y=np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64),
    )
    shape = (1, 5, 1)

    def field(value: float) -> np.ndarray:
        return np.full(shape, value, dtype=np.float64)

    species = {
        "d": SimpleNamespace(atomic_mass=2.0, charge=0.0, density=field(5.0)),
        "d+": SimpleNamespace(atomic_mass=2.0, charge=1.0, density=field(3.0)),
    }
    prepared = {
        "d": SimpleNamespace(density=field(5.0), temperature=field(1.0)),
        "d+": SimpleNamespace(density=field(3.0), temperature=field(1.0)),
    }
    if multispecies:
        species.update(
            {
                "t": SimpleNamespace(atomic_mass=3.0, charge=0.0, density=field(4.0)),
                "t+": SimpleNamespace(atomic_mass=3.0, charge=1.0, density=field(1.0)),
            }
        )
        prepared.update(
            {
                "t": SimpleNamespace(density=field(4.0), temperature=field(1.0)),
                "t+": SimpleNamespace(density=field(1.0), temperature=field(1.0)),
            }
        )
    return {
        "config": SimpleNamespace(scaled=False),
        "mesh": mesh,
        "metrics": SimpleNamespace(),
        "dataset_scalars": {"Tnorm": 1.0},
        "species": species,
        "prepared": prepared,
    }


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
    monkeypatch.setattr(
        "jax_drb.validation.reactions_collisions_campaign.build_reactions_collisions_context",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "jax_drb.validation.reactions_collisions_campaign._build_reactions_collisions_profiles",
        lambda **kwargs: {},
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
    assert payload["profiles"] == {}


def test_reactions_collisions_campaign_runs_synthetic_physics_gates_without_external_decks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    single_context = _synthetic_reactions_context(multispecies=False)
    multispecies_context = _synthetic_reactions_context(multispecies=True)
    scaled_context = _synthetic_reactions_context(multispecies=True)
    scaled_context["config"] = SimpleNamespace(scaled=True)

    def fake_context(path: Path, *, config=None):
        if config is not None:
            return scaled_context
        return single_context if "single" in str(path) else multispecies_context

    def fake_hydrogen_cx_sigmav(teff, _dataset_scalars):
        return np.full_like(np.asarray(teff, dtype=np.float64), 2.0, dtype=np.float64)

    def fake_charge_exchange_rates(config, *, species, prepared, dataset_scalars):
        base = np.ones_like(next(iter(prepared.values())).density, dtype=np.float64)
        if "t+" in prepared:
            d_total = 8.0 * base
            d_plus_total = 4.0 * base
            if getattr(config, "scaled", False):
                d_total = 24.0 * base
                d_plus_total = 12.0 * base
            return {"d": d_total, "d+": d_plus_total, "t": 3.0 * base, "t+": base}
        return {"d": 6.0 * base, "d+": 2.0 * base}

    def fake_collision_frequencies(_config, _species, prepared, *, dataset_scalars):
        base = np.ones_like(prepared["d+"].density, dtype=np.float64)
        return {("d+", "d+"): base, ("d+", "d"): 2.0 * base}

    def fake_viscosity_inputs(*, species_name, species, prepared, collision_rates, cx_rates):
        base = np.ones_like(prepared[species_name].density, dtype=np.float64)
        return SimpleNamespace(total_collisionality=7.0 * base)

    def fake_ionisation_rates(_config, *, species, prepared, dataset_scalars):
        return {"d": np.full_like(prepared["d"].density, 0.5, dtype=np.float64)}

    def fake_reaction_sources(_config, *, species, electron_density, dataset_scalars):
        return SimpleNamespace(diagnostics={"Sd+_iz": 0.5 * species["d"].density})

    monkeypatch.setattr(reactions_campaign_mod, "build_reactions_collisions_context", fake_context)
    monkeypatch.setattr(reactions_campaign_mod, "load_bout_input", lambda _path: SimpleNamespace(scaled=False))
    monkeypatch.setattr(
        reactions_campaign_mod,
        "apply_bout_overrides",
        lambda _config, _overrides: SimpleNamespace(scaled=True),
    )
    monkeypatch.setattr(reactions_campaign_mod, "hydrogen_cx_sigmav", fake_hydrogen_cx_sigmav)
    monkeypatch.setattr(reactions_campaign_mod, "charge_exchange_collision_rates", fake_charge_exchange_rates)
    monkeypatch.setattr(reactions_campaign_mod, "compute_collision_frequencies", fake_collision_frequencies)
    monkeypatch.setattr(reactions_campaign_mod, "ion_parallel_viscosity_inputs", fake_viscosity_inputs)
    monkeypatch.setattr(reactions_campaign_mod, "neutral_ionisation_collision_rates", fake_ionisation_rates)
    monkeypatch.setattr(reactions_campaign_mod, "reaction_sources", fake_reaction_sources)
    monkeypatch.setattr(
        reactions_campaign_mod,
        "electron_density",
        lambda ions: np.full_like(next(iter(ions)).density, 3.0, dtype=np.float64),
    )
    monkeypatch.setattr(
        reactions_campaign_mod,
        "load_openadas_rate",
        lambda *_args: (
            np.ones((2, 2), dtype=np.float64),
            np.ones((2, 2), dtype=np.float64),
            np.ones(2, dtype=np.float64),
            np.ones(2, dtype=np.float64),
            np.ones(1, dtype=np.float64),
        ),
    )

    metrics = build_reactions_collisions_campaign(
        single_species_input=Path("single/BOUT.inp"),
        multispecies_input=Path("multi/BOUT.inp"),
    )
    profiles = reactions_campaign_mod._build_reactions_collisions_profiles(
        single_species=single_context,
        multispecies=multispecies_context,
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
    assert profiles["ionisation_profile"]["series"]["assembled_collision_rate"].shape == (3,)
    assert profiles["d_atom_charge_exchange_profile"]["series"]["cross_isotope_t_plus"].mean() > 0.0
    assert np.allclose(
        profiles["d_plus_collisionality_profile"]["series"]["assembled_total_collisionality"],
        profiles["d_plus_collisionality_profile"]["series"]["expected_collision_stack"],
    )

    with tempfile.TemporaryDirectory() as tmp_root:
        artifacts = create_reactions_collisions_campaign_package(
            output_root=tmp_root,
            single_species_input=Path("single/BOUT.inp"),
            multispecies_input=Path("multi/BOUT.inp"),
        )
        payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
        assert sorted(payload["profiles"]) == [
            "d_atom_charge_exchange_profile",
            "d_plus_collisionality_profile",
            "ionisation_profile",
        ]
        assert artifacts.arrays_npz_path.exists()
        assert artifacts.plot_png_path.exists()


def test_reactions_collisions_context_builder_uses_prepared_open_field_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {"mesh": "synthetic"}
    run_config = SimpleNamespace()
    mesh = SimpleNamespace(xstart=0, ystart=0, yend=0, y=np.asarray([0.0], dtype=np.float64))
    metrics = SimpleNamespace()
    species = {"d+": SimpleNamespace(charge=1.0)}
    prepared = {"d+": SimpleNamespace(density=np.ones((1, 1, 1), dtype=np.float64))}

    monkeypatch.setattr(reactions_campaign_mod, "load_bout_input", lambda path: config)
    monkeypatch.setattr(reactions_campaign_mod.RunConfiguration, "from_config", lambda _config: run_config)
    monkeypatch.setattr(reactions_campaign_mod, "build_structured_mesh", lambda _config, _run_config: mesh)
    monkeypatch.setattr(reactions_campaign_mod, "build_structured_metrics", lambda _config, _run_config, _mesh: metrics)
    monkeypatch.setattr(reactions_campaign_mod, "resolved_dataset_scalars", lambda _run_config: {"Tnorm": 1.0})
    monkeypatch.setattr(reactions_campaign_mod, "_initialize_species", lambda _config, *, mesh: species)
    monkeypatch.setattr(
        reactions_campaign_mod,
        "_prepare_open_field_states",
        lambda _species, **_kwargs: (prepared, {}, {}),
    )

    context = reactions_campaign_mod.build_reactions_collisions_context(Path("synthetic/BOUT.inp"))

    assert context["config"] is config
    assert context["mesh"] is mesh
    assert context["metrics"] is metrics
    assert context["dataset_scalars"] == {"Tnorm": 1.0}
    assert context["species"] is species
    assert context["prepared"] is prepared


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


def test_create_reactions_collisions_campaign_package_writes_profile_arrays() -> None:
    reference_root = default_reference_root()
    if reference_root is None:
        pytest.skip("external reference decks are not available")
    with tempfile.TemporaryDirectory() as tmp_root:
        artifacts = create_reactions_collisions_campaign_package(
            output_root=tmp_root,
            single_species_input=reference_root / "tests" / "integrated" / "1D-recycling" / "data" / "BOUT.inp",
            multispecies_input=reference_root / "tests" / "integrated" / "1D-recycling-dthe" / "data" / "BOUT.inp",
        )
        payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
        assert sorted(payload["profiles"]) == [
            "d_atom_charge_exchange_profile",
            "d_plus_collisionality_profile",
            "ionisation_profile",
        ]
        arrays = np.load(artifacts.arrays_npz_path)
        assert "ionisation_profile_coordinate" in arrays
        assert "ionisation_profile_diagnostic_per_density" in arrays
        assert "ionisation_profile_assembled_collision_rate" in arrays
        assert "d_atom_charge_exchange_profile_same_isotope_d_plus" in arrays
        assert "d_atom_charge_exchange_profile_cross_isotope_t_plus" in arrays
        assert "d_atom_charge_exchange_profile_assembled_total" in arrays
        assert "d_plus_collisionality_profile_expected_collision_stack" in arrays
        assert "d_plus_collisionality_profile_assembled_total_collisionality" in arrays
