from __future__ import annotations

import numpy as np
import pytest

from jax_drb.config.boutinp import parse_bout_input
from jax_drb.native.array_backend import use_jax_backend
from jax_drb.native.mesh import StructuredMesh
from jax_drb.native.metrics import StructuredMetrics
from jax_drb.native.recycling_1d import _prepare_open_field_states
from jax_drb.native.recycling_setup import OpenFieldSpecies


def _mesh_and_metrics() -> tuple[StructuredMesh, StructuredMetrics]:
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
    return mesh, StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g22=ones,
        g33=ones,
        g_22=ones,
        g23=np.zeros_like(ones),
        Bxy=ones,
    )


def _species(field, pressure, momentum) -> dict[str, OpenFieldSpecies]:
    return {
        "e": OpenFieldSpecies(
            name="e",
            density=field,
            pressure=pressure,
            momentum=momentum * 0.0,
            charge=-1.0,
            atomic_mass=1.0 / 1836.0,
            density_floor=1.0e-8,
            has_pressure=True,
            has_momentum=False,
            noflow_lower_y=False,
            noflow_upper_y=False,
            target_recycle=False,
            recycle_as=None,
            target_recycle_multiplier=0.0,
            target_recycle_energy=0.0,
            target_fast_recycle_fraction=0.0,
            target_fast_recycle_energy_factor=0.0,
        ),
        "d+": OpenFieldSpecies(
            name="d+",
            density=2.0 * field,
            pressure=1.5 * pressure,
            momentum=momentum,
            charge=1.0,
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
        ),
    }


def test_prepare_open_field_states_no_sheath_preserves_jax_backend_and_jvp() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    mesh, metrics = _mesh_and_metrics()
    config = parse_bout_input("")
    field = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float64)
    pressure = np.array([[[2.0], [4.0], [6.0]]], dtype=np.float64)
    momentum = np.array([[[0.2], [0.7], [1.1]]], dtype=np.float64)
    weights = jnp.asarray([[[0.2], [0.5], [0.9]]], dtype=jnp.float64)

    numpy_prepared, numpy_ion_boundary, numpy_electron_boundary = _prepare_open_field_states(
        _species(field, pressure, momentum),
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars={},
        apply_sheath_boundaries=False,
    )

    def qoi(scale):
        prepared, ion_boundary, electron_boundary = _prepare_open_field_states(
            _species(scale * jnp.asarray(field), scale * jnp.asarray(pressure), scale * jnp.asarray(momentum)),
            config=config,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars={},
            apply_sheath_boundaries=False,
        )
        assert use_jax_backend(prepared["e"].density)
        assert use_jax_backend(prepared["d+"].momentum)
        assert use_jax_backend(ion_boundary.energy_source["d+"])
        assert use_jax_backend(electron_boundary.energy_source)
        return (
            jnp.sum(prepared["e"].velocity * weights)
            + 0.1 * jnp.sum(prepared["d+"].velocity * weights)
            + 0.01 * jnp.sum(ion_boundary.density["d+"] * weights)
            + 0.001 * jnp.sum(electron_boundary.density * weights)
        )

    value, tangent = jax.jvp(qoi, (jnp.array(1.0),), (jnp.array(1.0),))
    eps = 1.0e-5
    finite_difference = (qoi(1.0 + eps) - qoi(1.0 - eps)) / (2.0 * eps)

    numpy_value = (
        np.sum(numpy_prepared["e"].velocity * np.asarray(weights))
        + 0.1 * np.sum(numpy_prepared["d+"].velocity * np.asarray(weights))
        + 0.01 * np.sum(numpy_ion_boundary.density["d+"] * np.asarray(weights))
        + 0.001 * np.sum(numpy_electron_boundary.density * np.asarray(weights))
    )
    np.testing.assert_allclose(np.asarray(value), numpy_value, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(tangent), np.asarray(finite_difference), rtol=2.0e-6, atol=2.0e-8)
