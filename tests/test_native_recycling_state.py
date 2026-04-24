from __future__ import annotations

import numpy as np
import pytest

from jax_drb.native.mesh import StructuredMesh
from jax_drb.native.recycling_setup import OpenFieldSpecies
from jax_drb.native.recycling_state import (
    merge_target_guard_cells,
    prepare_species_state,
    safe_temperature,
)


def test_prepare_species_state_reconstructs_neutral_target_guards_from_active_cells() -> None:
    mesh = StructuredMesh(
        nx=1,
        ny=2,
        nz=1,
        mxg=0,
        myg=2,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=1,
        jyseps1_2=1,
        jyseps2_2=1,
        ny_inner=2,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=np.array([0.0], dtype=np.float64),
        y=np.arange(6, dtype=np.float64),
        z=np.array([0.0], dtype=np.float64),
    )
    density = np.zeros((1, 6, 1), dtype=np.float64)
    pressure = np.zeros((1, 6, 1), dtype=np.float64)
    momentum = np.zeros((1, 6, 1), dtype=np.float64)
    density[0, 2, 0] = 2.0
    density[0, 3, 0] = 0.5
    pressure[0, 2, 0] = 6.0
    pressure[0, 3, 0] = 1.0
    momentum[0, 2, 0] = -4.0
    momentum[0, 3, 0] = 0.25
    species = OpenFieldSpecies(
        name="d",
        density=density,
        pressure=pressure,
        momentum=momentum,
        charge=0.0,
        atomic_mass=2.0,
        density_floor=1.0e-8,
        has_pressure=True,
        has_momentum=True,
        noflow_lower_y=False,
        noflow_upper_y=False,
        target_recycle=False,
        recycle_as=None,
        target_recycle_multiplier=0.0,
        target_recycle_energy=0.0,
        target_fast_recycle_fraction=0.0,
        target_fast_recycle_energy_factor=0.0,
    )

    prepared = prepare_species_state(species, mesh=mesh)

    assert prepared.density[0, 1, 0] == pytest.approx(3.5)
    assert prepared.pressure[0, 1, 0] == pytest.approx(6.0)
    assert prepared.momentum[0, 1, 0] == pytest.approx(4.0)
    assert prepared.velocity[0, 1, 0] == pytest.approx(4.0 / (2.0 * 3.5))


def test_prepare_species_state_only_applies_noflow_guards_on_local_targets() -> None:
    mesh = StructuredMesh(
        nx=1,
        ny=2,
        nz=1,
        mxg=0,
        myg=2,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=1,
        jyseps1_2=1,
        jyseps2_2=1,
        ny_inner=2,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=np.array([0.0], dtype=np.float64),
        y=np.arange(6, dtype=np.float64),
        z=np.array([0.0], dtype=np.float64),
    )
    density = np.ones((1, 6, 1), dtype=np.float64)
    pressure = np.full((1, 6, 1), 2.0, dtype=np.float64)
    momentum = np.zeros((1, 6, 1), dtype=np.float64)
    momentum[0, 2, 0] = -0.5
    momentum[0, 3, 0] = 0.25
    momentum[0, 4, 0] = 0.75
    species = OpenFieldSpecies(
        name="d+",
        density=density,
        pressure=pressure,
        momentum=momentum,
        charge=1.0,
        atomic_mass=2.0,
        density_floor=1.0e-8,
        has_pressure=True,
        has_momentum=True,
        noflow_lower_y=False,
        noflow_upper_y=True,
        target_recycle=False,
        recycle_as=None,
        target_recycle_multiplier=0.0,
        target_recycle_energy=0.0,
        target_fast_recycle_fraction=0.0,
        target_fast_recycle_energy_factor=0.0,
    )

    prepared = prepare_species_state(species, mesh=mesh)

    assert prepared.velocity[0, 4, 0] == pytest.approx(0.75 / 2.0)
    assert prepared.momentum[0, 4, 0] == pytest.approx(0.75)


def test_merge_target_guard_cells_overwrites_only_local_target_guards() -> None:
    mesh = StructuredMesh(
        nx=1,
        ny=2,
        nz=1,
        mxg=0,
        myg=2,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=1,
        jyseps1_2=1,
        jyseps2_2=1,
        ny_inner=2,
        has_lower_y_target=True,
        has_upper_y_target=True,
        x=np.array([0.0], dtype=np.float64),
        y=np.arange(6, dtype=np.float64),
        z=np.array([0.0], dtype=np.float64),
    )
    base = np.zeros((1, 6, 1), dtype=np.float64)
    boundary = np.arange(6, dtype=np.float64).reshape(1, 6, 1)

    merged = merge_target_guard_cells(base, boundary, mesh=mesh)

    assert merged[0, 1, 0] == pytest.approx(boundary[0, 1, 0])
    assert merged[0, 4, 0] == pytest.approx(boundary[0, 4, 0])
    assert merged[0, 2, 0] == pytest.approx(0.0)
    assert merged[0, 3, 0] == pytest.approx(0.0)


def test_merge_target_guard_cells_preserves_jax_backend() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    mesh = StructuredMesh(
        nx=1,
        ny=2,
        nz=1,
        mxg=0,
        myg=2,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=1,
        jyseps1_2=1,
        jyseps2_2=1,
        ny_inner=2,
        has_lower_y_target=True,
        has_upper_y_target=True,
        x=np.array([0.0], dtype=np.float64),
        y=np.arange(6, dtype=np.float64),
        z=np.array([0.0], dtype=np.float64),
    )
    base = np.zeros((1, 6, 1), dtype=np.float64)
    boundary = np.arange(6, dtype=np.float64).reshape(1, 6, 1)

    def qoi(scale):
        merged = merge_target_guard_cells(jnp.asarray(base), scale * jnp.asarray(boundary), mesh=mesh)
        return jnp.sum(merged)

    value, tangent = jax.jvp(qoi, (jnp.array(1.0),), (jnp.array(1.0),))

    assert value == pytest.approx(1.0 + 4.0)
    assert tangent == pytest.approx(1.0 + 4.0)


def test_prepare_species_state_is_jax_jvp_transformable_for_neutral_guards() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    mesh = StructuredMesh(
        nx=1,
        ny=2,
        nz=1,
        mxg=0,
        myg=2,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=1,
        jyseps1_2=1,
        jyseps2_2=1,
        ny_inner=2,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=np.array([0.0], dtype=np.float64),
        y=np.arange(6, dtype=np.float64),
        z=np.array([0.0], dtype=np.float64),
    )
    density = np.zeros((1, 6, 1), dtype=np.float64)
    pressure = np.zeros((1, 6, 1), dtype=np.float64)
    momentum = np.zeros((1, 6, 1), dtype=np.float64)
    density[0, 2, 0] = 2.0
    density[0, 3, 0] = 0.5
    pressure[0, 2, 0] = 6.0
    pressure[0, 3, 0] = 1.0
    momentum[0, 2, 0] = -4.0
    momentum[0, 3, 0] = 0.25

    def qoi(scale):
        species = OpenFieldSpecies(
            name="d",
            density=jnp.asarray(density) * scale,
            pressure=jnp.asarray(pressure) * scale,
            momentum=jnp.asarray(momentum) * scale,
            charge=0.0,
            atomic_mass=2.0,
            density_floor=1.0e-8,
            has_pressure=True,
            has_momentum=True,
            noflow_lower_y=False,
            noflow_upper_y=False,
            target_recycle=False,
            recycle_as=None,
            target_recycle_multiplier=0.0,
            target_recycle_energy=0.0,
            target_fast_recycle_fraction=0.0,
            target_fast_recycle_energy_factor=0.0,
        )
        prepared = prepare_species_state(species, mesh=mesh)
        return jnp.sum(prepared.density) + 0.1 * jnp.sum(prepared.pressure) + 0.01 * jnp.sum(prepared.momentum)

    _, tangent = jax.jvp(qoi, (jnp.array(1.0),), (jnp.array(1.0),))
    eps = 1.0e-5
    finite_difference = (qoi(1.0 + eps) - qoi(1.0 - eps)) / (2.0 * eps)

    np.testing.assert_allclose(np.asarray(tangent), np.asarray(finite_difference), rtol=2.0e-6, atol=2.0e-8)


def test_safe_temperature_jax_branch_matches_numpy() -> None:
    jnp = pytest.importorskip("jax.numpy")
    pressure = np.array([[[0.0], [2.0], [6.0]]], dtype=np.float64)
    density = np.array([[[0.0], [1.0], [3.0]]], dtype=np.float64)

    np.testing.assert_allclose(
        np.asarray(safe_temperature(jnp.asarray(pressure), jnp.asarray(density), 1.0e-6)),
        safe_temperature(pressure, density, 1.0e-6),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
