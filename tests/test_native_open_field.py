from __future__ import annotations

import numpy as np
import pytest

import jax
from jax import grad
import jax.numpy as jnp

from jax_drb.native.mesh import StructuredMesh
from jax_drb.native.open_field import (
    apply_noflow_flow_guards,
    apply_noflow_scalar_guards,
    apply_parallel_electric_force,
    build_target_boundary_geometry,
    compute_electron_force_balance,
    compute_full_electron_sheath_boundary,
    compute_full_ion_sheath_boundary,
    compute_simple_ion_sheath_boundary,
    compute_target_recycling_sources,
    grad_par_y,
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
        has_lower_y_target=False,
        has_upper_y_target=True,
        x=jnp.arange(nx, dtype=jnp.float64),
        y=jnp.arange(ny + 2 * myg, dtype=jnp.float64),
        z=jnp.arange(nz, dtype=jnp.float64),
    )


def test_limit_free_matches_reference_modes() -> None:
    assert float(limit_free(jnp.array(2.0), jnp.array(3.0), 0)) == 3.0
    assert float(limit_free(jnp.array(4.0), jnp.array(2.0), 0)) == 1.0
    assert float(limit_free(jnp.array(4.0), jnp.array(2.0), 1)) == 1.0
    assert float(limit_free(jnp.array(4.0), jnp.array(2.0), 2)) == 0.0


def test_limit_free_numpy_reference_modes_and_invalid_modes() -> None:
    fm = np.asarray([2.0, 4.0], dtype=np.float64)
    fc = np.asarray([3.0, 2.0], dtype=np.float64)

    np.testing.assert_allclose(limit_free(fm, fc, 0), np.asarray([3.0, 1.0]))
    np.testing.assert_allclose(limit_free(fm, fc, 1), np.asarray([4.5, 1.0]))
    np.testing.assert_allclose(limit_free(fm, fc, 2), np.asarray([4.0, 0.0]))

    with pytest.raises(ValueError, match="Unsupported boundary mode"):
        limit_free(fm, fc, 99)
    with pytest.raises(ValueError, match="Unsupported boundary mode"):
        limit_free(jnp.asarray(fm), jnp.asarray(fc), 99)


def test_noflow_guards_copy_scalars_and_reflect_flows() -> None:
    mesh = _mesh()
    field = jnp.arange(mesh.nx * mesh.local_ny * mesh.nz, dtype=jnp.float64).reshape((mesh.nx, mesh.local_ny, mesh.nz))

    scalar = apply_noflow_scalar_guards(field, mesh=mesh, lower_y=True, upper_y=True)
    flow = apply_noflow_flow_guards(field, mesh=mesh, lower_y=True, upper_y=True)

    np.testing.assert_allclose(np.asarray(scalar[:, mesh.ystart - 1, :]), np.asarray(field[:, mesh.ystart, :]))
    np.testing.assert_allclose(np.asarray(scalar[:, mesh.yend + 1, :]), np.asarray(field[:, mesh.yend, :]))
    np.testing.assert_allclose(np.asarray(flow[:, mesh.ystart - 1, :]), -np.asarray(field[:, mesh.ystart, :]))
    np.testing.assert_allclose(np.asarray(flow[:, mesh.yend + 1, :]), -np.asarray(field[:, mesh.yend, :]))


def test_noflow_guards_use_numpy_fast_path_without_changing_values() -> None:
    mesh = _mesh()
    field = np.arange(mesh.nx * mesh.local_ny * mesh.nz, dtype=np.float64).reshape((mesh.nx, mesh.local_ny, mesh.nz))

    scalar = apply_noflow_scalar_guards(field, mesh=mesh, lower_y=True, upper_y=True)
    flow = apply_noflow_flow_guards(field, mesh=mesh, lower_y=True, upper_y=True)

    assert isinstance(scalar, np.ndarray)
    assert isinstance(flow, np.ndarray)
    np.testing.assert_allclose(scalar[:, mesh.ystart - 1, :], field[:, mesh.ystart, :])
    np.testing.assert_allclose(scalar[:, mesh.yend + 1, :], field[:, mesh.yend, :])
    np.testing.assert_allclose(flow[:, mesh.ystart - 1, :], -field[:, mesh.ystart, :])
    np.testing.assert_allclose(flow[:, mesh.yend + 1, :], -field[:, mesh.yend, :])


def test_noflow_guards_are_noops_without_guard_cells() -> None:
    mesh = _mesh(myg=0)
    field = jnp.arange(mesh.nx * mesh.local_ny * mesh.nz, dtype=jnp.float64).reshape((mesh.nx, mesh.local_ny, mesh.nz))
    field_np = np.asarray(field)

    np.testing.assert_allclose(
        np.asarray(apply_noflow_scalar_guards(field, mesh=mesh, lower_y=True, upper_y=True)),
        field_np,
    )
    np.testing.assert_allclose(
        np.asarray(apply_noflow_flow_guards(field, mesh=mesh, lower_y=True, upper_y=True)),
        field_np,
    )
    np.testing.assert_allclose(
        apply_noflow_scalar_guards(field_np, mesh=mesh, lower_y=True, upper_y=True),
        field_np,
    )
    np.testing.assert_allclose(
        apply_noflow_flow_guards(field_np, mesh=mesh, lower_y=True, upper_y=True),
        field_np,
    )


def test_noflow_guards_can_disable_each_target_side_independently() -> None:
    mesh = _mesh()
    field = jnp.arange(mesh.nx * mesh.local_ny * mesh.nz, dtype=jnp.float64).reshape((mesh.nx, mesh.local_ny, mesh.nz))

    scalar = apply_noflow_scalar_guards(field, mesh=mesh, lower_y=False, upper_y=False)
    flow = apply_noflow_flow_guards(field, mesh=mesh, lower_y=False, upper_y=False)

    np.testing.assert_allclose(np.asarray(scalar), np.asarray(field))
    np.testing.assert_allclose(np.asarray(flow), np.asarray(field))


def test_grad_par_y_returns_zero_for_single_active_cell() -> None:
    mesh = _mesh(ny=1)
    shape = (mesh.nx, mesh.local_ny, mesh.nz)
    field = jnp.linspace(0.0, 1.0, mesh.local_ny, dtype=jnp.float64).reshape(shape)
    dy = jnp.ones(shape, dtype=jnp.float64)

    np.testing.assert_allclose(np.asarray(grad_par_y(field, mesh=mesh, dy=dy)), np.zeros(shape))
    np.testing.assert_allclose(grad_par_y(np.asarray(field), mesh=mesh, dy=np.asarray(dy)), np.zeros(shape))


def test_grad_par_y_numpy_path_matches_centered_difference() -> None:
    mesh = _mesh()
    field = np.asarray([[[0.0], [0.0], [2.0], [4.0], [8.0], [8.0], [0.0], [0.0]]], dtype=np.float64)
    dy = np.ones_like(field)

    gradient = grad_par_y(field, mesh=mesh, dy=dy)

    expected = np.zeros_like(field)
    expected[:, mesh.ystart : mesh.yend + 1, :] = (
        field[:, mesh.ystart + 1 : mesh.yend + 2, :] - field[:, mesh.ystart - 1 : mesh.yend, :]
    ) / 2.0
    np.testing.assert_allclose(gradient, expected)


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


def test_electron_force_balance_numpy_path_adds_momentum_source_and_density_floor() -> None:
    mesh = _mesh()
    pe = np.asarray([[[0.0], [0.0], [2.0], [4.0], [8.0], [8.0], [0.0], [0.0]]], dtype=np.float64)
    ne = np.ones_like(pe)
    ne[:, mesh.ystart, :] = 1.0e-12
    dy = np.ones_like(pe)
    momentum_source = 0.25 * np.ones_like(pe)

    result = compute_electron_force_balance(
        pe,
        ne,
        mesh=mesh,
        dy=dy,
        electron_momentum_source=momentum_source,
        density_floor=1.0e-3,
    )

    expected_force = -grad_par_y(pe, mesh=mesh, dy=dy) + momentum_source
    np.testing.assert_allclose(result.force_density, expected_force)
    assert float(result.epar[0, mesh.ystart, 0]) == pytest.approx(float(expected_force[0, mesh.ystart, 0]) / 1.0e-3)


def test_parallel_electric_force_numpy_path_accumulates_existing_source() -> None:
    density = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    epar = np.asarray([0.5, -0.25, 0.125], dtype=np.float64)
    existing = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)

    source = apply_parallel_electric_force(density, charge=-2.0, epar=epar, existing_source=existing)

    np.testing.assert_allclose(source, -2.0 * density * epar + existing)


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


def test_target_recycling_sources_numpy_fast_path_matches_reference_formula() -> None:
    mesh = _mesh()
    shape = (mesh.nx, mesh.local_ny, mesh.nz)
    density = np.ones(shape, dtype=np.float64)
    velocity = np.zeros(shape, dtype=np.float64)
    temperature = 2.0 * np.ones(shape, dtype=np.float64)
    velocity[:, mesh.ystart - 1, :] = -3.0
    velocity[:, mesh.ystart, :] = -1.0
    velocity[:, mesh.yend, :] = 2.0
    velocity[:, mesh.yend + 1, :] = 4.0
    J = np.ones(shape, dtype=np.float64)
    dy = 2.0 * np.ones(shape, dtype=np.float64)
    dx = 3.0 * np.ones(shape, dtype=np.float64)
    dz = 5.0 * np.ones(shape, dtype=np.float64)
    g_22 = 4.0 * np.ones(shape, dtype=np.float64)

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

    assert isinstance(result.density_source, np.ndarray)
    assert isinstance(result.energy_source, np.ndarray)
    assert float(result.density_source[0, mesh.ystart, 0]) == pytest.approx(0.25)
    assert float(result.density_source[0, mesh.yend, 0]) == pytest.approx(0.375)
    assert float(result.energy_source[0, mesh.ystart, 0]) == pytest.approx(0.75)
    assert float(result.energy_source[0, mesh.yend, 0]) == pytest.approx(1.125)


def test_target_recycling_sources_fast_fraction_uses_reference_fixed_energy_branch() -> None:
    mesh = _mesh()
    shape = (mesh.nx, mesh.local_ny, mesh.nz)
    density = np.ones(shape, dtype=np.float64)
    velocity = np.zeros(shape, dtype=np.float64)
    temperature = 2.0 * np.ones(shape, dtype=np.float64)
    velocity[:, mesh.ystart - 1, :] = -3.0
    velocity[:, mesh.ystart, :] = -1.0
    J = np.ones(shape, dtype=np.float64)
    dy = 2.0 * np.ones(shape, dtype=np.float64)
    dx = 3.0 * np.ones(shape, dtype=np.float64)
    dz = 5.0 * np.ones(shape, dtype=np.float64)
    g_22 = 4.0 * np.ones(shape, dtype=np.float64)

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
        target_fast_recycle_fraction=0.8,
        target_fast_recycle_energy_factor=0.48,
        upper_y=False,
    )

    assert float(result.density_source[0, mesh.ystart, 0]) == pytest.approx(0.25)
    assert float(result.energy_source[0, mesh.ystart, 0]) == pytest.approx(0.25 * 0.2 * 3.0)


def test_target_recycling_sources_return_zero_without_y_guard_cells() -> None:
    mesh = _mesh(myg=0)
    shape = (mesh.nx, mesh.local_ny, mesh.nz)
    density = jnp.ones(shape, dtype=jnp.float64)
    velocity = jnp.ones(shape, dtype=jnp.float64)
    temperature = jnp.ones(shape, dtype=jnp.float64)
    unit_metric = jnp.ones(shape, dtype=jnp.float64)

    result = compute_target_recycling_sources(
        density,
        velocity,
        temperature,
        mesh=mesh,
        J=unit_metric,
        dy=unit_metric,
        dx=unit_metric,
        dz=unit_metric,
        g_22=unit_metric,
        target_multiplier=1.0,
        target_energy=3.0,
        gamma_i=3.5,
    )

    np.testing.assert_allclose(np.asarray(result.density_source), np.zeros(shape))
    np.testing.assert_allclose(np.asarray(result.energy_source), np.zeros(shape))


def test_numpy_target_boundary_geometry_matches_finite_volume_scale() -> None:
    mesh = _mesh()
    shape = (mesh.nx, mesh.local_ny, mesh.nz)
    J = np.ones(shape, dtype=np.float64)
    dy = 2.0 * np.ones(shape, dtype=np.float64)
    dx = 3.0 * np.ones(shape, dtype=np.float64)
    dz = 5.0 * np.ones(shape, dtype=np.float64)
    g_22 = 4.0 * np.ones(shape, dtype=np.float64)

    geometry = build_target_boundary_geometry(
        J=J,
        dy=dy,
        dx=dx,
        dz=dz,
        g_22=g_22,
        y_index=mesh.yend,
        guard_index=mesh.yend + 1,
    )

    dapar = 0.25 * (1.0 + 1.0) / (np.sqrt(4.0) + np.sqrt(4.0)) * (3.0 + 3.0) * (5.0 + 5.0)
    volume = 1.0 * 3.0 * 2.0 * 5.0
    np.testing.assert_allclose(geometry.source_scale, dapar / volume)


def test_target_recycling_sources_precomputed_geometry_matches_direct_geometry_path() -> None:
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

    direct = compute_target_recycling_sources(
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
    cached = compute_target_recycling_sources(
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
        lower_geometry=build_target_boundary_geometry(
            J=J,
            dy=dy,
            dx=dx,
            dz=dz,
            g_22=g_22,
            y_index=mesh.ystart,
            guard_index=mesh.ystart - 1,
        ),
        upper_geometry=build_target_boundary_geometry(
            J=J,
            dy=dy,
            dx=dx,
            dz=dz,
            g_22=g_22,
            y_index=mesh.yend,
            guard_index=mesh.yend + 1,
        ),
    )

    np.testing.assert_allclose(np.asarray(cached.density_source), np.asarray(direct.density_source), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(cached.energy_source), np.asarray(direct.energy_source), rtol=0.0, atol=0.0)


def test_simple_ion_sheath_boundary_matches_numpy_and_is_jvp_transformable() -> None:
    sheath_density = np.asarray([[1.2, 1.4]], dtype=np.float64)
    sheath_temperature = np.asarray([[2.0, 2.5]], dtype=np.float64)
    electron_temperature = np.asarray([[3.0, 3.5]], dtype=np.float64)
    interior_velocity = np.asarray([[0.1, 2.2]], dtype=np.float64)
    interior_momentum = np.asarray([[0.25, 0.4]], dtype=np.float64)
    source_scale = np.asarray([[0.5, 0.75]], dtype=np.float64)

    numpy_result = compute_simple_ion_sheath_boundary(
        sheath_density=sheath_density,
        sheath_temperature=sheath_temperature,
        electron_sheath_temperature=electron_temperature,
        interior_velocity=interior_velocity,
        interior_momentum=interior_momentum,
        atomic_mass=2.0,
        charge=1.0,
        gamma_i=3.5,
        sheath_ion_polytropic=1.0,
        direction=1.0,
        source_scale=source_scale,
    )
    jax_result = compute_simple_ion_sheath_boundary(
        sheath_density=jnp.asarray(sheath_density),
        sheath_temperature=jnp.asarray(sheath_temperature),
        electron_sheath_temperature=jnp.asarray(electron_temperature),
        interior_velocity=jnp.asarray(interior_velocity),
        interior_momentum=jnp.asarray(interior_momentum),
        atomic_mass=2.0,
        charge=1.0,
        gamma_i=3.5,
        sheath_ion_polytropic=1.0,
        direction=1.0,
        source_scale=jnp.asarray(source_scale),
    )

    np.testing.assert_allclose(np.asarray(jax_result.guard_velocity), numpy_result.guard_velocity, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(jax_result.guard_momentum), numpy_result.guard_momentum, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(jax_result.energy_source_delta),
        numpy_result.energy_source_delta,
        rtol=0.0,
        atol=0.0,
    )

    def qoi(scale: jnp.ndarray) -> jnp.ndarray:
        result = compute_simple_ion_sheath_boundary(
            sheath_density=jnp.asarray(sheath_density) * scale,
            sheath_temperature=jnp.asarray(sheath_temperature),
            electron_sheath_temperature=jnp.asarray(electron_temperature),
            interior_velocity=jnp.asarray(interior_velocity),
            interior_momentum=jnp.asarray(interior_momentum),
            atomic_mass=2.0,
            charge=1.0,
            gamma_i=3.5,
            sheath_ion_polytropic=1.0,
            direction=1.0,
            source_scale=jnp.asarray(source_scale),
        )
        return jnp.sum(result.energy_source_delta)

    _, tangent = jax.jvp(qoi, (jnp.array(1.0),), (jnp.array(1.0),))
    eps = 1.0e-5
    finite_difference = (qoi(jnp.array(1.0 + eps)) - qoi(jnp.array(1.0 - eps))) / (2.0 * eps)
    np.testing.assert_allclose(np.asarray(tangent), np.asarray(finite_difference), rtol=2.0e-7, atol=2.0e-9)


def test_simple_ion_sheath_boundary_lower_target_no_flow_sign() -> None:
    result = compute_simple_ion_sheath_boundary(
        sheath_density=np.asarray([[1.5]], dtype=np.float64),
        sheath_temperature=np.asarray([[2.0]], dtype=np.float64),
        electron_sheath_temperature=np.asarray([[3.0]], dtype=np.float64),
        interior_velocity=np.asarray([[-0.2]], dtype=np.float64),
        interior_momentum=np.asarray([[0.1]], dtype=np.float64),
        atomic_mass=2.0,
        charge=1.0,
        gamma_i=3.5,
        sheath_ion_polytropic=1.0,
        direction=-1.0,
        no_flow=True,
        source_scale=np.asarray([[0.25]], dtype=np.float64),
    )

    assert float(result.sheath_velocity[0, 0]) == pytest.approx(0.0)
    assert float(result.guard_velocity[0, 0]) == pytest.approx(0.2)
    assert float(result.guard_momentum[0, 0]) == pytest.approx(-0.1)
    assert float(result.energy_source_delta[0, 0]) == pytest.approx(0.0)


def test_full_electron_sheath_boundary_matches_numpy_and_is_jvp_transformable() -> None:
    sheath_density = np.asarray([[1.2, 1.4]], dtype=np.float64)
    sheath_temperature = np.asarray([[2.0, 2.5]], dtype=np.float64)
    raw_potential = np.asarray([[5.0, 6.0]], dtype=np.float64)
    wall_potential = np.asarray([[0.25, 0.25]], dtype=np.float64)
    interior_velocity = np.asarray([[0.1, 0.2]], dtype=np.float64)
    interior_momentum = np.asarray([[0.01, 0.02]], dtype=np.float64)
    source_scale = np.asarray([[0.5, 0.75]], dtype=np.float64)

    numpy_result = compute_full_electron_sheath_boundary(
        sheath_density=sheath_density,
        sheath_temperature=sheath_temperature,
        sheath_potential_raw=raw_potential,
        wall_potential=wall_potential,
        interior_velocity=interior_velocity,
        interior_momentum=interior_momentum,
        electron_mass=1.0 / 1836.0,
        electron_thermal_mass=1.0 / 1836.0,
        secondary_electron_coef=0.2,
        electron_adiabatic=5.0 / 3.0,
        direction=1.0,
        source_scale=source_scale,
    )
    jax_result = compute_full_electron_sheath_boundary(
        sheath_density=jnp.asarray(sheath_density),
        sheath_temperature=jnp.asarray(sheath_temperature),
        sheath_potential_raw=jnp.asarray(raw_potential),
        wall_potential=jnp.asarray(wall_potential),
        interior_velocity=jnp.asarray(interior_velocity),
        interior_momentum=jnp.asarray(interior_momentum),
        electron_mass=1.0 / 1836.0,
        electron_thermal_mass=1.0 / 1836.0,
        secondary_electron_coef=0.2,
        electron_adiabatic=5.0 / 3.0,
        direction=1.0,
        source_scale=jnp.asarray(source_scale),
    )

    np.testing.assert_allclose(np.asarray(jax_result.sheath_velocity), numpy_result.sheath_velocity, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(jax_result.guard_velocity), numpy_result.guard_velocity, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(jax_result.guard_momentum), numpy_result.guard_momentum, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(jax_result.energy_source_delta),
        numpy_result.energy_source_delta,
        rtol=0.0,
        atol=0.0,
    )

    def qoi(scale: jnp.ndarray) -> jnp.ndarray:
        result = compute_full_electron_sheath_boundary(
            sheath_density=jnp.asarray(sheath_density) * scale,
            sheath_temperature=jnp.asarray(sheath_temperature),
            sheath_potential_raw=jnp.asarray(raw_potential),
            wall_potential=jnp.asarray(wall_potential),
            interior_velocity=jnp.asarray(interior_velocity),
            interior_momentum=jnp.asarray(interior_momentum),
            electron_mass=1.0 / 1836.0,
            electron_thermal_mass=1.0 / 1836.0,
            secondary_electron_coef=0.2,
            electron_adiabatic=5.0 / 3.0,
            direction=1.0,
            source_scale=jnp.asarray(source_scale),
        )
        return jnp.sum(result.energy_source_delta)

    _, tangent = jax.jvp(qoi, (jnp.array(1.0),), (jnp.array(1.0),))
    eps = 1.0e-5
    finite_difference = (qoi(jnp.array(1.0 + eps)) - qoi(jnp.array(1.0 - eps))) / (2.0 * eps)
    np.testing.assert_allclose(np.asarray(tangent), np.asarray(finite_difference), rtol=2.0e-7, atol=2.0e-9)


def test_full_electron_sheath_boundary_lower_target_sign_and_potential_floor() -> None:
    result = compute_full_electron_sheath_boundary(
        sheath_density=np.asarray([[1.5]], dtype=np.float64),
        sheath_temperature=np.asarray([[2.0]], dtype=np.float64),
        sheath_potential_raw=np.asarray([[0.1]], dtype=np.float64),
        wall_potential=np.asarray([[0.5]], dtype=np.float64),
        interior_velocity=np.asarray([[0.2]], dtype=np.float64),
        interior_momentum=np.asarray([[0.1]], dtype=np.float64),
        electron_mass=1.0 / 1836.0,
        electron_thermal_mass=1.0 / 1836.0,
        secondary_electron_coef=0.1,
        electron_adiabatic=5.0 / 3.0,
        direction=-1.0,
        source_scale=np.asarray([[0.25]], dtype=np.float64),
    )

    assert float(result.sheath_potential[0, 0]) == pytest.approx(0.5)
    assert float(result.sheath_velocity[0, 0]) < 0.0
    assert float(result.energy_source_delta[0, 0]) <= 0.0


def test_full_ion_sheath_boundary_matches_numpy_and_is_jvp_transformable() -> None:
    sheath_density = np.asarray([[1.2, 1.4]], dtype=np.float64)
    sheath_temperature = np.asarray([[2.0, 2.5]], dtype=np.float64)
    electron_density = np.asarray([[2.0, 2.2]], dtype=np.float64)
    electron_temperature = np.asarray([[3.0, 3.5]], dtype=np.float64)
    grad_ne = np.asarray([[0.4, 0.5]], dtype=np.float64)
    grad_ni = np.asarray([[0.3, 0.6]], dtype=np.float64)
    interior_velocity = np.asarray([[0.1, 0.2]], dtype=np.float64)
    interior_momentum = np.asarray([[0.25, 0.4]], dtype=np.float64)
    source_scale = np.asarray([[0.5, 0.75]], dtype=np.float64)

    numpy_result = compute_full_ion_sheath_boundary(
        sheath_density=sheath_density,
        sheath_temperature=sheath_temperature,
        electron_sheath_density=electron_density,
        electron_sheath_temperature=electron_temperature,
        electron_density_gradient=grad_ne,
        ion_density_gradient=grad_ni,
        interior_velocity=interior_velocity,
        interior_momentum=interior_momentum,
        atomic_mass=2.0,
        charge=1.0,
        direction=1.0,
        source_scale=source_scale,
    )
    jax_result = compute_full_ion_sheath_boundary(
        sheath_density=jnp.asarray(sheath_density),
        sheath_temperature=jnp.asarray(sheath_temperature),
        electron_sheath_density=jnp.asarray(electron_density),
        electron_sheath_temperature=jnp.asarray(electron_temperature),
        electron_density_gradient=jnp.asarray(grad_ne),
        ion_density_gradient=jnp.asarray(grad_ni),
        interior_velocity=jnp.asarray(interior_velocity),
        interior_momentum=jnp.asarray(interior_momentum),
        atomic_mass=2.0,
        charge=1.0,
        direction=1.0,
        source_scale=jnp.asarray(source_scale),
    )

    np.testing.assert_allclose(np.asarray(jax_result.guard_velocity), numpy_result.guard_velocity, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(jax_result.guard_momentum), numpy_result.guard_momentum, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(jax_result.energy_source_delta),
        numpy_result.energy_source_delta,
        rtol=0.0,
        atol=0.0,
    )

    def qoi(scale: jnp.ndarray) -> jnp.ndarray:
        result = compute_full_ion_sheath_boundary(
            sheath_density=jnp.asarray(sheath_density) * scale,
            sheath_temperature=jnp.asarray(sheath_temperature),
            electron_sheath_density=jnp.asarray(electron_density),
            electron_sheath_temperature=jnp.asarray(electron_temperature),
            electron_density_gradient=jnp.asarray(grad_ne),
            ion_density_gradient=jnp.asarray(grad_ni),
            interior_velocity=jnp.asarray(interior_velocity),
            interior_momentum=jnp.asarray(interior_momentum),
            atomic_mass=2.0,
            charge=1.0,
            direction=1.0,
            source_scale=jnp.asarray(source_scale),
        )
        return jnp.sum(result.energy_source_delta)

    _, tangent = jax.jvp(qoi, (jnp.array(1.0),), (jnp.array(1.0),))
    eps = 1.0e-5
    finite_difference = (qoi(jnp.array(1.0 + eps)) - qoi(jnp.array(1.0 - eps))) / (2.0 * eps)
    np.testing.assert_allclose(np.asarray(tangent), np.asarray(finite_difference), rtol=3.0e-7, atol=3.0e-9)


def test_full_ion_sheath_boundary_lower_target_sign_and_gradient_floor() -> None:
    result = compute_full_ion_sheath_boundary(
        sheath_density=np.asarray([[1.5]], dtype=np.float64),
        sheath_temperature=np.asarray([[2.0]], dtype=np.float64),
        electron_sheath_density=np.asarray([[2.5]], dtype=np.float64),
        electron_sheath_temperature=np.asarray([[3.0]], dtype=np.float64),
        electron_density_gradient=np.asarray([[0.2]], dtype=np.float64),
        ion_density_gradient=np.asarray([[0.0]], dtype=np.float64),
        interior_velocity=np.asarray([[0.2]], dtype=np.float64),
        interior_momentum=np.asarray([[0.1]], dtype=np.float64),
        atomic_mass=2.0,
        charge=1.0,
        direction=-1.0,
        source_scale=np.asarray([[0.25]], dtype=np.float64),
    )

    assert float(result.sound_speed_squared[0, 0]) > 0.0
    assert float(result.sheath_velocity[0, 0]) < 0.0
    assert float(result.guard_velocity[0, 0]) < 0.0
    assert float(result.energy_source_delta[0, 0]) <= 0.0


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
