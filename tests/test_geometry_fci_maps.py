from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import jax

from drbx.geometry import MetricTensor3D, build_metric_report, build_synthetic_stellarator_geometry, identity_fci_maps
from drbx.native.fci import (
    conservative_parallel_diffusion_fci,
    conservative_perp_diffusion_xz,
    fci_yup,
    grad_parallel_fci,
    logical_exb_bracket_xz,
    metric_weighted_scalar_laplacian_3d,
)
from drbx.native.fci_sheath_recycling import compute_fci_sheath_recycling
from drbx.native.fci_drb_rhs import FciDrbRhsParameters, FciDrbState, compute_fci_drb_rhs


def test_synthetic_stellarator_geometry_metric_and_maps_are_valid() -> None:
    geometry = build_synthetic_stellarator_geometry(nx=10, ny=8, nz=16)

    report = build_metric_report(geometry.metric)

    assert report["passed"] is True
    assert geometry.maps.shape == (10, 8, 16)
    assert float(jnp.max(geometry.metric.Bxy)) > float(jnp.min(geometry.metric.Bxy))
    assert float(jnp.min(geometry.connection_length)) > 0.0


def test_identity_fci_maps_reproduce_neighboring_plane_values() -> None:
    maps = identity_fci_maps(nx=4, ny=5, nz=6, dphi=0.2)
    field = jnp.arange(4 * 5 * 6, dtype=jnp.float64).reshape((4, 5, 6))

    actual = fci_yup(field, maps)
    expected = jnp.roll(field, -1, axis=1)

    assert jnp.allclose(actual, expected)
    assert jnp.allclose(grad_parallel_fci(jnp.ones_like(field), maps), 0.0)


def test_fci_sheath_recycling_closes_particle_and_current_balance() -> None:
    geometry = build_synthetic_stellarator_geometry(nx=10, ny=8, nz=16)
    density = 1.0 + 0.1 * geometry.radial
    electron_temperature = 0.1 + 0.02 * geometry.radial
    ion_temperature = 0.08 + 0.01 * geometry.radial

    result = compute_fci_sheath_recycling(
        density,
        electron_temperature,
        ion_temperature,
        geometry.maps,
        recycling_fraction=0.95,
    )

    assert float(result.total_ion_particle_loss) > 0.0
    assert float(jnp.abs(result.particle_recycling_residual)) < 1.0e-12
    assert float(jnp.abs(result.current_balance_residual)) < 1.0e-12
    assert float(jnp.max(result.target_heat_load)) > 0.0


def test_conservative_fci_diffusion_annihilates_constants_and_dissipates() -> None:
    geometry = build_synthetic_stellarator_geometry(nx=10, ny=8, nz=16)
    coefficient = jnp.ones(geometry.shape, dtype=jnp.float64) * 0.07
    constant = jnp.ones(geometry.shape, dtype=jnp.float64)
    field = jnp.sin(jnp.pi * geometry.radial) * jnp.cos(2.0 * geometry.poloidal_angle - 5.0 * geometry.toroidal_angle)

    parallel_constant = conservative_parallel_diffusion_fci(
        constant,
        coefficient,
        geometry.maps,
        jacobian=geometry.metric.J,
    )
    perpendicular_constant = conservative_perp_diffusion_xz(constant, coefficient, geometry.metric)
    parallel_energy_rate = jnp.mean(field * conservative_parallel_diffusion_fci(field, coefficient, geometry.maps, jacobian=geometry.metric.J))
    perpendicular_energy_rate = jnp.mean(field * conservative_perp_diffusion_xz(field, coefficient, geometry.metric))

    assert float(jnp.max(jnp.abs(parallel_constant))) < 1.0e-12
    assert float(jnp.max(jnp.abs(perpendicular_constant))) < 1.0e-12
    assert float(parallel_energy_rate) < 0.0
    assert float(perpendicular_energy_rate) < 0.0


def test_metric_weighted_scalar_laplacian_3d_matches_cartesian_mms() -> None:
    errors = []
    resolutions = np.asarray([12, 16, 24], dtype=np.int64)
    for resolution in resolutions:
        metric = _identity_metric_3d(nx=resolution, ny=resolution, nz=2 * resolution)
        x = jnp.arange(resolution, dtype=jnp.float64) / float(resolution)
        y = 2.0 * jnp.pi * jnp.arange(resolution, dtype=jnp.float64) / float(resolution)
        z = 2.0 * jnp.pi * jnp.arange(2 * resolution, dtype=jnp.float64) / float(2 * resolution)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        field = jnp.sin(2.0 * jnp.pi * X) * jnp.cos(3.0 * Y) * jnp.sin(2.0 * Z)
        exact = -((2.0 * jnp.pi) ** 2 + 3.0**2 + 2.0**2) * field

        actual = metric_weighted_scalar_laplacian_3d(
            field,
            metric,
            periodic_axes=(True, True, True),
        )
        errors.append(float(jnp.sqrt(jnp.mean(jnp.square(actual - exact)))))

    slope, _ = np.polyfit(np.log(1.0 / resolutions.astype(np.float64)), np.log(np.asarray(errors)), 1)
    assert float(slope) > 1.7
    assert errors[-1] < 0.35 * errors[0]


def test_logical_exb_bracket_xz_matches_periodic_cartesian_mms() -> None:
    errors = []
    resolutions = np.asarray([16, 24, 32], dtype=np.int64)
    for resolution in resolutions:
        metric = _identity_metric_3d(nx=resolution, ny=4, nz=2 * resolution)
        x = jnp.arange(resolution, dtype=jnp.float64) / float(resolution)
        y = jnp.arange(4, dtype=jnp.float64)
        z = 2.0 * jnp.pi * jnp.arange(2 * resolution, dtype=jnp.float64) / float(2 * resolution)
        X, _, Z = jnp.meshgrid(x, y, z, indexing="ij")
        phi = jnp.sin(2.0 * jnp.pi * X) * jnp.cos(Z)
        field = jnp.cos(2.0 * jnp.pi * X) * jnp.sin(2.0 * Z)
        exact = (
            2.0 * jnp.pi * jnp.square(jnp.sin(2.0 * jnp.pi * X)) * jnp.sin(Z) * jnp.sin(2.0 * Z)
            - 4.0 * jnp.pi * jnp.square(jnp.cos(2.0 * jnp.pi * X)) * jnp.cos(Z) * jnp.cos(2.0 * Z)
        )

        actual = logical_exb_bracket_xz(phi, field, metric, periodic_x=True, periodic_z=True)
        errors.append(float(jnp.sqrt(jnp.mean(jnp.square(actual - exact)))))

        assert float(jnp.abs(jnp.mean(actual))) < 1.0e-12
        assert (
            float(
                jnp.max(
                    jnp.abs(
                        logical_exb_bracket_xz(
                            phi,
                            jnp.ones_like(field),
                            metric,
                            periodic_x=True,
                            periodic_z=True,
                        )
                    )
                )
            )
            < 1.0e-12
        )

    slope, _ = np.polyfit(np.log(1.0 / resolutions.astype(np.float64)), np.log(np.asarray(errors)), 1)
    assert float(slope) > 1.6
    assert errors[-1] < 0.45 * errors[0]


def test_fci_drb_pytree_rhs_is_jvp_transformable() -> None:
    geometry = build_synthetic_stellarator_geometry(nx=6, ny=5, nz=10)
    radial = geometry.radial
    theta = geometry.poloidal_angle
    phi = geometry.toroidal_angle
    state = FciDrbState(
        ion_density=1.0 + 0.15 * radial,
        electron_density=1.0 + 0.15 * radial,
        neutral_density=0.2 + 0.1 * radial,
        ion_pressure=0.08 + 0.02 * radial,
        electron_pressure=0.10 + 0.03 * radial,
        neutral_pressure=0.01 + 0.003 * radial,
        ion_momentum=0.02 * jnp.cos(2.0 * theta - 5.0 * phi),
        neutral_momentum=0.01 * jnp.sin(theta - 5.0 * phi),
        vorticity=0.04 * jnp.sin(2.0 * theta - 5.0 * phi),
    )
    tangent = jax.tree_util.tree_map(lambda value: jnp.ones_like(value) * 1.0e-3, state)

    def objective(candidate: FciDrbState) -> jnp.ndarray:
        result = compute_fci_drb_rhs(
            candidate,
            maps=geometry.maps,
            metric=geometry.metric,
            parameters=FciDrbRhsParameters(potential_iterations=8),
        )
        return jnp.sum(result.rhs.ion_density) + jnp.sum(result.rhs.neutral_density) + result.potential_residual_l2

    value, derivative = jax.jvp(objective, (state,), (tangent,))

    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(derivative))


def test_fci_drb_rhs_potential_boussinesq_switch_changes_potential_only() -> None:
    geometry = build_synthetic_stellarator_geometry(nx=6, ny=5, nz=10)
    radial = geometry.radial
    theta = geometry.poloidal_angle
    phi = geometry.toroidal_angle
    state = FciDrbState(
        ion_density=1.0 + 0.35 * radial + 0.08 * jnp.cos(theta - 2.0 * phi),
        electron_density=1.0 + 0.35 * radial + 0.08 * jnp.cos(theta - 2.0 * phi),
        neutral_density=0.2 + 0.1 * radial,
        ion_pressure=0.08 + 0.02 * radial,
        electron_pressure=0.10 + 0.03 * radial,
        neutral_pressure=0.01 + 0.003 * radial,
        ion_momentum=0.02 * jnp.cos(2.0 * theta - 5.0 * phi),
        neutral_momentum=0.01 * jnp.sin(theta - 5.0 * phi),
        vorticity=0.04 * jnp.sin(2.0 * theta - 5.0 * phi),
    )
    boussinesq = compute_fci_drb_rhs(
        state,
        maps=geometry.maps,
        metric=geometry.metric,
        parameters=FciDrbRhsParameters(
            potential_iterations=10,
            potential_boussinesq=True,
        ),
    )
    non_boussinesq = compute_fci_drb_rhs(
        state,
        maps=geometry.maps,
        metric=geometry.metric,
        parameters=FciDrbRhsParameters(
            potential_iterations=10,
            potential_boussinesq=False,
        ),
    )
    potential_relative_difference = jnp.linalg.norm(
        non_boussinesq.potential - boussinesq.potential
    ) / jnp.maximum(jnp.linalg.norm(boussinesq.potential), 1.0e-30)
    rhs_difference = max(
        float(jnp.max(jnp.abs(left - right)))
        for left, right in zip(
            jax.tree_util.tree_leaves(boussinesq.rhs),
            jax.tree_util.tree_leaves(non_boussinesq.rhs),
            strict=True,
        )
    )

    assert float(potential_relative_difference) > 1.0e-4
    assert rhs_difference < 1.0e-12


def test_fci_drb_rhs_potential_feedback_changes_plasma_rhs_and_is_jvp_safe() -> None:
    geometry = build_synthetic_stellarator_geometry(nx=6, ny=5, nz=10)
    radial = geometry.radial
    theta = geometry.poloidal_angle
    phi = geometry.toroidal_angle
    state = FciDrbState(
        ion_density=1.0 + 0.35 * radial + 0.08 * jnp.cos(theta - 2.0 * phi),
        electron_density=1.0 + 0.32 * radial + 0.06 * jnp.sin(theta + phi),
        neutral_density=0.2 + 0.1 * radial,
        ion_pressure=(0.08 + 0.02 * radial) * (1.0 + 0.04 * jnp.cos(2.0 * theta)),
        electron_pressure=(0.10 + 0.03 * radial) * (1.0 + 0.03 * jnp.sin(theta - phi)),
        neutral_pressure=0.01 + 0.003 * radial,
        ion_momentum=0.02 * jnp.cos(2.0 * theta - 5.0 * phi),
        neutral_momentum=0.01 * jnp.sin(theta - 5.0 * phi),
        vorticity=0.04 * jnp.sin(2.0 * theta - 5.0 * phi) + 0.01 * jnp.cos(theta + phi),
    )
    common = {
        "potential_iterations": 10,
        "plasma_exb_advection_strength": 0.04,
    }
    boussinesq = compute_fci_drb_rhs(
        state,
        maps=geometry.maps,
        metric=geometry.metric,
        parameters=FciDrbRhsParameters(potential_boussinesq=True, **common),
    )
    non_boussinesq = compute_fci_drb_rhs(
        state,
        maps=geometry.maps,
        metric=geometry.metric,
        parameters=FciDrbRhsParameters(potential_boussinesq=False, **common),
    )
    plasma_difference = max(
        float(jnp.max(jnp.abs(left - right)))
        for left, right in (
            (boussinesq.rhs.ion_density, non_boussinesq.rhs.ion_density),
            (boussinesq.rhs.electron_density, non_boussinesq.rhs.electron_density),
            (boussinesq.rhs.ion_pressure, non_boussinesq.rhs.ion_pressure),
            (boussinesq.rhs.electron_pressure, non_boussinesq.rhs.electron_pressure),
            (boussinesq.rhs.ion_momentum, non_boussinesq.rhs.ion_momentum),
            (boussinesq.rhs.vorticity, non_boussinesq.rhs.vorticity),
        )
    )
    neutral_difference = max(
        float(jnp.max(jnp.abs(left - right)))
        for left, right in (
            (boussinesq.rhs.neutral_density, non_boussinesq.rhs.neutral_density),
            (boussinesq.rhs.neutral_pressure, non_boussinesq.rhs.neutral_pressure),
            (boussinesq.rhs.neutral_momentum, non_boussinesq.rhs.neutral_momentum),
        )
    )

    assert plasma_difference > 1.0e-8
    assert neutral_difference < 1.0e-12

    tangent = jax.tree_util.tree_map(lambda value: jnp.ones_like(value) * 1.0e-3, state)

    def objective(candidate: FciDrbState) -> jnp.ndarray:
        result = compute_fci_drb_rhs(
            candidate,
            maps=geometry.maps,
            metric=geometry.metric,
            parameters=FciDrbRhsParameters(
                potential_iterations=8,
                plasma_exb_advection_strength=0.04,
                potential_boussinesq=False,
            ),
        )
        return (
            jnp.sum(result.rhs.ion_density)
            + jnp.sum(result.rhs.electron_pressure)
            + jnp.sum(result.rhs.vorticity)
            + result.potential_residual_l2
        )

    value, derivative = jax.jvp(objective, (state,), (tangent,))

    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(derivative))


def _identity_metric_3d(*, nx: int, ny: int, nz: int) -> MetricTensor3D:
    shape = (nx, ny, nz)
    ones = jnp.ones(shape, dtype=jnp.float64)
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    return MetricTensor3D(
        dx=ones * (1.0 / float(nx)),
        dy=ones * (2.0 * np.pi / float(ny)),
        dz=ones * (2.0 * np.pi / float(nz)),
        J=ones,
        Bxy=ones,
        g11=ones,
        g22=ones,
        g33=ones,
        g12=zeros,
        g13=zeros,
        g23=zeros,
        g_11=ones,
        g_22=ones,
        g_33=ones,
        g_12=zeros,
        g_13=zeros,
        g_23=zeros,
    )
