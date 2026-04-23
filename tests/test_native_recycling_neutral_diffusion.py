from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from jax_drb.config.boutinp import apply_bout_overrides, load_bout_input
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.recycling_collisions import compute_collision_frequencies
from jax_drb.native.recycling_neutral_diffusion import apply_neutral_parallel_diffusion
from jax_drb.native.recycling_reactions import (
    neutral_charge_exchange_collision_rates,
    neutral_ionisation_collision_rates,
)
from jax_drb.native.recycling_setup import initialize_species
from jax_drb.native.recycling_state import prepare_species_state
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.native.units import resolved_dataset_scalars


_INPUT_1D = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling/data/BOUT.inp")


def _build_prepared_case(*, overrides: tuple[str, ...] = ()) -> tuple[object, object, object, dict[str, object], dict[str, object]]:
    config = apply_bout_overrides(load_bout_input(_INPUT_1D), overrides)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    species = initialize_species(config, mesh=mesh, dataset_scalars=scalars)
    prepared = {name: prepare_species_state(sp, mesh=mesh) for name, sp in species.items()}
    return config, mesh, metrics, species, prepared


def test_neutral_parallel_diffusion_returns_zero_when_component_disabled() -> None:
    config, mesh, metrics, species, prepared = _build_prepared_case()

    terms = apply_neutral_parallel_diffusion(
        config,
        species=species,
        prepared=prepared,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(RunConfiguration.from_config(config)),
    )

    assert all(np.allclose(value, 0.0) for value in terms.density_source.values())
    assert all(np.allclose(value, 0.0) for value in terms.energy_source.values())
    assert all(np.allclose(value, 0.0) for value in terms.momentum_source.values())
    assert terms.diagnostics == {}


def test_neutral_parallel_diffusion_raises_on_unsupported_mode() -> None:
    config, mesh, metrics, species, prepared = _build_prepared_case(
        overrides=(
            "model:components=neutral_parallel_diffusion",
            "neutral_parallel_diffusion:dneut=1.0",
            "neutral_parallel_diffusion:diffusion_collisions_mode=unsupported_mode",
        )
    )

    with pytest.raises(NotImplementedError, match="Unsupported neutral_parallel_diffusion"):
        apply_neutral_parallel_diffusion(
            config,
            species=species,
            prepared=prepared,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=resolved_dataset_scalars(RunConfiguration.from_config(config)),
        )


def test_neutral_parallel_diffusion_diagnose_emits_profile_fields() -> None:
    config, mesh, metrics, species, prepared = _build_prepared_case(
        overrides=(
            "model:components=neutral_parallel_diffusion",
            "neutral_parallel_diffusion:dneut=1.0",
            "neutral_parallel_diffusion:diagnose=true",
        )
    )

    terms = apply_neutral_parallel_diffusion(
        config,
        species=species,
        prepared=prepared,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(RunConfiguration.from_config(config)),
    )

    assert "Dd_Dpar" in terms.diagnostics
    assert "Sd_Dpar" in terms.diagnostics
    assert "Ed_Dpar" in terms.diagnostics
    assert "Fd_Dpar" in terms.diagnostics
    assert np.isfinite(terms.diagnostics["Dd_Dpar"]).all()


def test_neutral_parallel_diffusion_multispecies_mode_produces_finite_terms() -> None:
    config, mesh, metrics, species, prepared = _build_prepared_case(
        overrides=(
            "model:components=neutral_parallel_diffusion",
            "neutral_parallel_diffusion:dneut=1.0",
            "neutral_parallel_diffusion:diffusion_collisions_mode=multispecies",
            "neutral_parallel_diffusion:diagnose=true",
        )
    )
    prepared_d = prepared["d"]
    density = np.asarray(prepared_d.density, dtype=np.float64, copy=True)
    pressure = np.asarray(prepared_d.pressure, dtype=np.float64, copy=True)
    temperature = np.asarray(prepared_d.temperature, dtype=np.float64, copy=True)
    momentum = np.asarray(prepared_d.momentum, dtype=np.float64, copy=True)
    velocity = np.asarray(prepared_d.velocity, dtype=np.float64, copy=True)
    density[:, mesh.ystart : mesh.yend + 1, :] *= np.linspace(1.0, 1.3, mesh.yend - mesh.ystart + 1)[None, :, None]
    pressure[:, mesh.ystart : mesh.yend + 1, :] *= np.linspace(1.0, 1.6, mesh.yend - mesh.ystart + 1)[None, :, None]
    temperature = pressure / np.maximum(density, 1.0e-8)
    prepared["d"] = replace(
        prepared_d,
        density=density,
        pressure=pressure,
        temperature=temperature,
        momentum=momentum,
        velocity=velocity,
    )

    terms = apply_neutral_parallel_diffusion(
        config,
        species=species,
        prepared=prepared,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(RunConfiguration.from_config(config)),
    )

    assert np.isfinite(terms.density_source["d"]).all()
    assert np.isfinite(terms.energy_source["d"]).all()
    assert np.isfinite(terms.momentum_source["d"]).all()
    assert float(np.nanmax(np.abs(terms.density_source["d"]))) > 0.0


def test_neutral_parallel_diffusion_accepts_precomputed_rates() -> None:
    config, mesh, metrics, species, prepared = _build_prepared_case(
        overrides=(
            "model:components=neutral_parallel_diffusion",
            "neutral_parallel_diffusion:dneut=1.0",
            "neutral_parallel_diffusion:diffusion_collisions_mode=multispecies",
        )
    )
    scalars = resolved_dataset_scalars(RunConfiguration.from_config(config))
    collision_rates = compute_collision_frequencies(config, species, prepared, dataset_scalars=scalars)
    ionisation_rates = neutral_ionisation_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=scalars,
    )
    cx_rates = neutral_charge_exchange_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=scalars,
    )

    baseline = apply_neutral_parallel_diffusion(
        config,
        species=species,
        prepared=prepared,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )
    reused = apply_neutral_parallel_diffusion(
        config,
        species=species,
        prepared=prepared,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        collision_rates=collision_rates,
        ionisation_rates=ionisation_rates,
        charge_exchange_rates=cx_rates,
    )

    for name in baseline.density_source:
        np.testing.assert_allclose(reused.density_source[name], baseline.density_source[name])
    for name in baseline.energy_source:
        np.testing.assert_allclose(reused.energy_source[name], baseline.energy_source[name])
    for name in baseline.momentum_source:
        np.testing.assert_allclose(reused.momentum_source[name], baseline.momentum_source[name])
