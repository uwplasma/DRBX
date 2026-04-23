from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from jax_drb.config.boutinp import apply_bout_overrides, load_bout_input
from jax_drb.native.mesh import StructuredMesh, build_structured_mesh
from jax_drb.native.metrics import StructuredMetrics, build_structured_metrics
from jax_drb.native.neutral_mixed import _grad_par_open
from jax_drb.native.recycling_collision_closure import (
    apply_collision_closure,
    conduction_kappa_coefficient,
    ion_thermal_force_pair,
    parallel_ion_viscous_stress_open,
)
from jax_drb.native.recycling_collisions import compute_collision_frequencies
from jax_drb.native.recycling_reactions import charge_exchange_collision_rates
from jax_drb.native.recycling_setup import initialize_species
from jax_drb.native.recycling_1d import _prepare_open_field_states
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.native.units import resolved_dataset_scalars


_DTHE_INPUT = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling-dthe/data/BOUT.inp")


def test_parallel_ion_viscous_stress_matches_braginskii_formula() -> None:
    mesh = StructuredMesh(
        nx=1,
        ny=1,
        nz=1,
        mxg=0,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=0,
        jyseps1_2=0,
        jyseps2_2=0,
        ny_inner=1,
        has_lower_y_target=True,
        has_upper_y_target=True,
        x=np.array([0.0], dtype=np.float64),
        y=np.array([-1.0, 0.0, 1.0], dtype=np.float64),
        z=np.array([0.0], dtype=np.float64),
    )
    ones = np.ones((1, 3, 1), dtype=np.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g22=ones,
        g33=ones,
        g_22=ones,
        g23=np.zeros_like(ones),
        Bxy=np.array([[[2.0], [4.0], [8.0]]], dtype=np.float64),
    )
    pressure = np.array([[[3.0], [5.0], [7.0]]], dtype=np.float64)
    tau = np.array([[[0.5], [0.25], [0.125]]], dtype=np.float64)
    velocity = np.array([[[1.0], [2.0], [4.0]]], dtype=np.float64)

    stress = parallel_ion_viscous_stress_open(
        pressure,
        tau,
        velocity,
        mesh=mesh,
        metrics=metrics,
    )

    expected = -0.96 * pressure * tau * (
        2.0 * _grad_par_open(velocity, mesh=mesh, metrics=metrics)
        + velocity * _grad_par_open(np.log(metrics.Bxy), mesh=mesh, metrics=metrics)
    )
    np.testing.assert_allclose(stress, expected)


def test_ion_thermal_force_pair_is_enabled_for_dt_when_mass_override_is_set() -> None:
    config = load_bout_input(_DTHE_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    species = initialize_species(config, mesh=mesh)
    prepared, _, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )

    pair = ion_thermal_force_pair(
        "d+",
        "t+",
        species=species,
        prepared=prepared,
        mesh=mesh,
        metrics=metrics,
        override_mass_restrictions=True,
    )

    assert pair is not None
    light_name, heavy_name, heavy_force = pair
    active = (mesh.xstart, mesh.yend, 0)

    assert light_name == "d+"
    assert heavy_name == "t+"
    assert np.isfinite(float(heavy_force[active]))
    assert heavy_force.shape == species["t+"].density.shape


def test_conduction_kappa_coefficient_uses_species_defaults_and_override() -> None:
    config = load_bout_input(_DTHE_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    species = initialize_species(config, mesh=mesh)

    assert conduction_kappa_coefficient(config, species["e"]) == pytest.approx(3.16 / math.sqrt(2.0))
    assert conduction_kappa_coefficient(config, species["d+"]) == pytest.approx(3.9)
    assert conduction_kappa_coefficient(config, species["d"]) == pytest.approx(2.5)

    override_config = apply_bout_overrides(load_bout_input(_DTHE_INPUT), ("d:kappa_coefficient=7.25",))
    assert conduction_kappa_coefficient(override_config, species["d"]) == pytest.approx(7.25)


def test_collision_closure_accepts_precomputed_collision_and_cx_rates() -> None:
    config = load_bout_input(_DTHE_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    species = initialize_species(config, mesh=mesh)
    prepared, _, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )

    collision_rates = compute_collision_frequencies(config, species, prepared, dataset_scalars=scalars)
    cx_rates = charge_exchange_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=scalars,
    )

    baseline = apply_collision_closure(
        config,
        species,
        prepared,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )
    reused = apply_collision_closure(
        config,
        species,
        prepared,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        collision_rates=collision_rates,
        cx_rates=cx_rates,
    )

    for name in baseline.energy_source:
        np.testing.assert_allclose(reused.energy_source[name], baseline.energy_source[name])
    for name in baseline.momentum_source:
        np.testing.assert_allclose(reused.momentum_source[name], baseline.momentum_source[name])
