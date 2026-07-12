from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.recycling_1d import (
    _charge_exchange_collision_rates,
    _initialize_species,
    _prepare_open_field_states,
)
from jax_drb.native.recycling_collisions import (
    compute_collision_frequencies,
    electron_density,
    ion_parallel_viscosity_inputs,
)
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.reference.paths import default_reference_root


_REFERENCE_ROOT = default_reference_root()
_REFERENCE_BASE = _REFERENCE_ROOT if _REFERENCE_ROOT is not None else Path("/nonexistent-reference-root")
_DTHE_INPUT = _REFERENCE_BASE / "tests/integrated/1D-recycling-dthe/data/BOUT.inp"


def test_electron_density_sums_ion_charge_weighted_fields() -> None:
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = load_bout_input(_DTHE_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    species = _initialize_species(config, mesh=mesh)
    ions = tuple(sp for sp in species.values() if sp.charge > 0.0)

    actual = electron_density(ions)
    expected = np.zeros_like(species["d+"].density, dtype=np.float64)
    for ion in ions:
        expected = expected + ion.charge * ion.density

    np.testing.assert_allclose(actual, expected)


def test_collision_frequencies_cover_asymmetric_ion_neutral_pairs() -> None:
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = load_bout_input(_DTHE_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    species = _initialize_species(config, mesh=mesh)
    prepared, _, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )
    collision_rates = compute_collision_frequencies(
        config,
        species,
        prepared,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )

    for pair in (("d", "t+"), ("t+", "d"), ("t", "d+"), ("d+", "t")):
        assert pair in collision_rates

    active = (0, mesh.ystart, 0)
    assert np.isfinite(float(collision_rates[("t", "d+")][active]))
    assert np.isfinite(float(collision_rates[("d", "t+")][active]))
    assert float(collision_rates[("t", "d+")][active]) != float(collision_rates[("d", "t+")][active])


def test_ion_parallel_viscosity_inputs_match_pressure_tau_formula() -> None:
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = load_bout_input(_DTHE_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    dataset_scalars = resolved_dataset_scalars(run_config)
    species = _initialize_species(config, mesh=mesh)
    prepared, _, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        apply_sheath_boundaries=True,
    )

    collision_rates = compute_collision_frequencies(
        config,
        species,
        prepared,
        dataset_scalars=dataset_scalars,
    )
    cx_rates = _charge_exchange_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=dataset_scalars,
    )

    inputs = ion_parallel_viscosity_inputs(
        species_name="d+",
        species=species,
        prepared=prepared,
        collision_rates=collision_rates,
        cx_rates=cx_rates,
    )

    expected_collisionality = np.zeros_like(prepared["d+"].density, dtype=np.float64)
    for other_name in species:
        rate = collision_rates.get(("d+", other_name))
        if rate is not None:
            expected_collisionality = expected_collisionality + rate
    expected_collisionality = expected_collisionality + cx_rates["d+"]
    expected_collisionality = np.maximum(expected_collisionality, 1.0e-12)
    expected_tau = 1.0 / expected_collisionality
    expected_eta = 1.28 * np.asarray(prepared["d+"].pressure, dtype=np.float64) * expected_tau

    np.testing.assert_allclose(inputs.total_collisionality, expected_collisionality)
    np.testing.assert_allclose(inputs.tau, expected_tau)
    np.testing.assert_allclose(inputs.eta, expected_eta)
