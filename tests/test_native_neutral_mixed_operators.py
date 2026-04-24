from __future__ import annotations

import numpy as np
import pytest

from jax_drb.native.mesh import StructuredMesh
from jax_drb.native.metrics import StructuredMetrics
from jax_drb.native.neutral_mixed_operators import div_par_fvv_open, div_par_mod_open


def _mesh_and_metrics() -> tuple[StructuredMesh, StructuredMetrics]:
    mesh = StructuredMesh(
        nx=2,
        ny=3,
        nz=2,
        mxg=0,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=0,
        jyseps1_2=0,
        jyseps2_2=0,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=True,
        x=np.linspace(0.0, 1.0, 2),
        y=np.linspace(-0.5, 3.5, 5),
        z=np.linspace(0.0, 1.0, 2),
    )
    shape = (mesh.nx, mesh.local_ny, mesh.nz)
    ones = np.ones(shape, dtype=np.float64)
    y_scale = np.linspace(0.9, 1.2, shape[1], dtype=np.float64)[None, :, None]
    return mesh, StructuredMetrics(
        dx=ones,
        dy=ones * y_scale,
        dz=ones,
        J=ones * np.linspace(1.0, 1.15, shape[1], dtype=np.float64)[None, :, None],
        g11=ones,
        g22=ones,
        g33=ones,
        g_22=ones * np.linspace(1.0, 1.3, shape[1], dtype=np.float64)[None, :, None],
        g23=np.zeros(shape, dtype=np.float64),
        Bxy=ones,
    )


def _sample_fields(mesh: StructuredMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (mesh.nx, mesh.local_ny, mesh.nz)
    field = np.linspace(1.0, 2.2, np.prod(shape), dtype=np.float64).reshape(shape)
    velocity = np.linspace(-0.35, 0.45, np.prod(shape), dtype=np.float64).reshape(shape)
    wave_speed = np.full(shape, 0.8, dtype=np.float64) + 0.1 * np.abs(velocity)
    return field, velocity, wave_speed


def test_parallel_advection_jax_branch_matches_numpy() -> None:
    jnp = pytest.importorskip("jax.numpy")
    mesh, metrics = _mesh_and_metrics()
    field, velocity, wave_speed = _sample_fields(mesh)

    numpy_result = div_par_mod_open(field, velocity, wave_speed, mesh=mesh, metrics=metrics)
    jax_result = div_par_mod_open(
        jnp.asarray(field),
        jnp.asarray(velocity),
        jnp.asarray(wave_speed),
        mesh=mesh,
        metrics=metrics,
    )

    np.testing.assert_allclose(np.asarray(jax_result), numpy_result, rtol=1.0e-12, atol=1.0e-12)


def test_parallel_inertia_jax_branch_matches_numpy_for_both_flux_modes() -> None:
    jnp = pytest.importorskip("jax.numpy")
    mesh, metrics = _mesh_and_metrics()
    density, velocity, wave_speed = _sample_fields(mesh)

    for fix_flux in (True, False):
        numpy_result = div_par_fvv_open(density, velocity, wave_speed, mesh=mesh, metrics=metrics, fix_flux=fix_flux)
        jax_result = div_par_fvv_open(
            jnp.asarray(density),
            jnp.asarray(velocity),
            jnp.asarray(wave_speed),
            mesh=mesh,
            metrics=metrics,
            fix_flux=fix_flux,
        )
        np.testing.assert_allclose(np.asarray(jax_result), numpy_result, rtol=1.0e-12, atol=1.0e-12)


def test_parallel_advection_jvp_matches_centered_finite_difference() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    mesh, metrics = _mesh_and_metrics()
    field, velocity, wave_speed = _sample_fields(mesh)
    weights = jnp.linspace(0.2, 1.1, field.size, dtype=jnp.float64).reshape(field.shape)

    def qoi(scale):
        result = div_par_mod_open(
            jnp.asarray(field) * scale,
            jnp.asarray(velocity),
            jnp.asarray(wave_speed),
            mesh=mesh,
            metrics=metrics,
        )
        return jnp.sum(result * weights)

    _, tangent = jax.jvp(qoi, (jnp.array(1.0),), (jnp.array(1.0),))
    eps = 1.0e-5
    finite_difference = (qoi(1.0 + eps) - qoi(1.0 - eps)) / (2.0 * eps)

    np.testing.assert_allclose(np.asarray(tangent), np.asarray(finite_difference), rtol=2.0e-6, atol=2.0e-8)
