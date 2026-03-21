from __future__ import annotations

import numpy as np

from jax import grad
import jax.numpy as jnp

from jax_drb.native.mesh import StructuredMesh
from jax_drb.native.open_field import (
    apply_noflow_flow_guards,
    apply_noflow_scalar_guards,
    apply_parallel_electric_force,
    compute_electron_force_balance,
    compute_target_recycling_sources,
    limit_free,
)


def _mesh(*, nx: int = 1, ny: int = 4, nz: int = 1, myg: int = 2) -> StructuredMesh:
    return StructuredMesh(
        nx=nx,
        ny=ny,
        nz=nz,
        mxg=0,
        myg=myg,
        symmetric_global_x=True,
        symmetric_global_y=True,
        jyseps1_1=-1,
        jyseps2_1=ny // 2,
        jyseps1_2=ny // 2,
        jyseps2_2=ny - 1,
        ny_inner=ny // 2,
        x=jnp.arange(nx, dtype=jnp.float64),
        y=jnp.arange(ny + 2 * myg, dtype=jnp.float64),
        z=jnp.arange(nz, dtype=jnp.float64),
    )


def test_limit_free_matches_reference_modes() -> None:
    assert float(limit_free(jnp.array(2.0), jnp.array(3.0), 0)) == 3.0
    assert float(limit_free(jnp.array(4.0), jnp.array(2.0), 0)) == 1.0
    assert float(limit_free(jnp.array(4.0), jnp.array(2.0), 1)) == 1.0
    assert float(limit_free(jnp.array(4.0), jnp.array(2.0), 2)) == 0.0


def test_noflow_guards_copy_scalars_and_reflect_flows() -> None:
    mesh = _mesh()
    field = jnp.arange(mesh.nx * mesh.local_ny * mesh.nz, dtype=jnp.float64).reshape((mesh.nx, mesh.local_ny, mesh.nz))

    scalar = apply_noflow_scalar_guards(field, mesh=mesh, lower_y=True, upper_y=True)
    flow = apply_noflow_flow_guards(field, mesh=mesh, lower_y=True, upper_y=True)

    np.testing.assert_allclose(np.asarray(scalar[:, mesh.ystart - 1, :]), np.asarray(field[:, mesh.ystart, :]))
    np.testing.assert_allclose(np.asarray(scalar[:, mesh.yend + 1, :]), np.asarray(field[:, mesh.yend, :]))
    np.testing.assert_allclose(np.asarray(flow[:, mesh.ystart - 1, :]), -np.asarray(field[:, mesh.ystart, :]))
    np.testing.assert_allclose(np.asarray(flow[:, mesh.yend + 1, :]), -np.asarray(field[:, mesh.yend, :]))


def test_electron_force_balance_applies_parallel_pressure_force_to_species() -> None:
    mesh = _mesh()
    pe = jnp.array([[[0.0], [0.0], [2.0], [4.0], [8.0], [8.0], [0.0], [0.0]]], dtype=jnp.float64)
    ne = jnp.ones_like(pe)
    dy = jnp.ones_like(pe)
    result = compute_electron_force_balance(pe, ne, mesh=mesh, dy=dy)
    ion_source = apply_parallel_electric_force(jnp.full_like(pe, 2.0), charge=1.0, epar=result.epar)

    expected_gradient = np.zeros_like(np.asarray(pe))
    expected_gradient[:, mesh.ystart : mesh.yend + 1, :] = -0.5 * (
        np.asarray(pe[:, mesh.ystart + 1 : mesh.yend + 2, :]) - np.asarray(pe[:, mesh.ystart - 1 : mesh.yend, :])
    )
    np.testing.assert_allclose(np.asarray(result.force_density), expected_gradient)
    np.testing.assert_allclose(np.asarray(ion_source), 2.0 * expected_gradient)


def test_target_recycling_sources_match_reference_formula() -> None:
    mesh = _mesh()
    shape = (mesh.nx, mesh.local_ny, mesh.nz)
    density = jnp.ones(shape, dtype=jnp.float64)
    velocity = jnp.zeros(shape, dtype=jnp.float64)
    temperature = 2.0 * jnp.ones(shape, dtype=jnp.float64)
    velocity = velocity.at[:, mesh.ystart - 1, :].set(-3.0)
    velocity = velocity.at[:, mesh.ystart, :].set(-1.0)
    velocity = velocity.at[:, mesh.yend, :].set(2.0)
    velocity = velocity.at[:, mesh.yend + 1, :].set(4.0)
    J = jnp.ones(shape, dtype=jnp.float64)
    dy = 2.0 * jnp.ones(shape, dtype=jnp.float64)
    dx = 3.0 * jnp.ones(shape, dtype=jnp.float64)
    dz = 5.0 * jnp.ones(shape, dtype=jnp.float64)
    g_22 = 4.0 * jnp.ones(shape, dtype=jnp.float64)

    result = compute_target_recycling_sources(
        density,
        velocity,
        temperature,
        mesh=mesh,
        J=J,
        dy=dy,
        dx=dx,
        dz=dz,
        g_22=g_22,
        target_multiplier=0.5,
        target_energy=3.0,
        gamma_i=3.5,
    )

    lower_flux = max(-0.25 * (1.0 + 1.0) * (-1.0 - 3.0), 0.0)
    upper_flux = max(0.25 * (1.0 + 1.0) * (2.0 + 4.0), 0.0)
    dapar = 0.25 * (1.0 + 1.0) / (np.sqrt(4.0) + np.sqrt(4.0)) * (3.0 + 3.0) * (5.0 + 5.0)
    volume = 1.0 * 3.0 * 2.0 * 5.0
    lower_density = 0.5 * lower_flux * dapar / volume
    upper_density = 0.5 * upper_flux * dapar / volume
    lower_heat = abs(3.5 * 1.0 * 2.0 * (-2.0) * dapar / volume)
    upper_heat = abs(3.5 * 1.0 * 2.0 * 3.0 * dapar / volume)
    lower_energy = (0.5 * lower_flux * dapar) * 3.0 / volume
    upper_energy = (0.5 * upper_flux * dapar) * 3.0 / volume

    assert float(result.density_source[0, mesh.ystart, 0]) == lower_density
    assert float(result.density_source[0, mesh.yend, 0]) == upper_density
    assert float(result.energy_source[0, mesh.ystart, 0]) == lower_energy
    assert float(result.energy_source[0, mesh.yend, 0]) == upper_energy
    assert lower_heat > 0.0
    assert upper_heat > 0.0


def test_open_field_utilities_are_differentiable() -> None:
    mesh = _mesh()
    shape = (mesh.nx, mesh.local_ny, mesh.nz)
    dy = jnp.ones(shape, dtype=jnp.float64)

    def loss(scale: jnp.ndarray) -> jnp.ndarray:
        pe = scale * jnp.linspace(0.0, 1.0, mesh.local_ny, dtype=jnp.float64)[None, :, None]
        pe = jnp.broadcast_to(pe, shape)
        ne = jnp.ones(shape, dtype=jnp.float64)
        result = compute_electron_force_balance(pe, ne, mesh=mesh, dy=dy)
        source = apply_parallel_electric_force(ne, charge=1.0, epar=result.epar)
        return jnp.sum(source * source)

    gradient = grad(loss)(jnp.array(2.0, dtype=jnp.float64))
    assert np.isfinite(float(gradient))
