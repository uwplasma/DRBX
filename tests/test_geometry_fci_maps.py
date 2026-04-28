from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import jax

from jax_drb.geometry import MetricTensor3D, build_metric_report, build_synthetic_stellarator_geometry, identity_fci_maps
from jax_drb.native.fci import (
    conservative_parallel_diffusion_fci,
    conservative_perp_diffusion_xz,
    fci_yup,
    grad_parallel_fci,
    metric_weighted_scalar_laplacian_3d,
)
from jax_drb.native.fci_sheath_recycling import compute_fci_sheath_recycling, fci_sheath_recycling_field_rhs
from jax_drb.native.fci_drb_rhs import FciDrbRhsParameters, FciDrbState, compute_fci_drb_rhs
from jax_drb.native.recycling_fixed_residual import (
    build_fixed_array_rhs,
    build_fixed_backward_euler_residual,
    fixed_residual_jvp_action,
    fixed_state_from_fields,
    linearize_fixed_residual_action,
    pack_fixed_state,
)
from jax_drb.native.recycling_layout import RecyclingPackedStateLayout


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


def test_fci_sheath_recycling_promotes_to_fixed_layout_rhs() -> None:
    geometry = build_synthetic_stellarator_geometry(nx=8, ny=6, nz=12)
    shape = geometry.shape
    fields = {
        "Ni": np.asarray(1.0 + 0.1 * geometry.radial, dtype=np.float64),
        "Ne": np.asarray(1.0 + 0.1 * geometry.radial, dtype=np.float64),
        "Nn": np.asarray(0.2 + 0.05 * geometry.radial, dtype=np.float64),
        "Pi": np.asarray(0.08 + 0.02 * geometry.radial, dtype=np.float64),
        "Pe": np.asarray(0.10 + 0.03 * geometry.radial, dtype=np.float64),
        "Pn": np.asarray(0.01 + 0.002 * geometry.radial, dtype=np.float64),
    }
    field_names = tuple(fields)
    layout = RecyclingPackedStateLayout(
        field_names=field_names,
        feedback_names=(),
        active_slices=(slice(None), slice(None), slice(None)),
        active_shape=shape,
        field_size=int(np.prod(shape)) * len(field_names),
        field_templates=tuple(fields[name] for name in field_names),
    )
    state = fixed_state_from_fields(fields, feedback_integrals={}, layout=layout)
    rhs_function = build_fixed_array_rhs(
        lambda active_fields, _feedback: fci_sheath_recycling_field_rhs(
            active_fields,
            geometry.maps,
            recycling_fraction=0.95,
        ),
        layout=layout,
    )

    rhs_state = rhs_function(state)
    rhs_fields = {name: value for name, value in zip(field_names, rhs_state.field_values, strict=True)}

    assert float(jnp.sum(rhs_fields["Ni"])) < 0.0
    assert float(jnp.sum(rhs_fields["Nn"])) > 0.0
    assert float(jnp.abs(jnp.sum(rhs_fields["Ne"] - rhs_fields["Ni"]))) < 1.0e-12
    assert float(jnp.abs(jnp.sum(rhs_fields["Nn"]) + 0.95 * jnp.sum(rhs_fields["Ni"]))) < 1.0e-12


def test_fixed_residual_jvp_action_matches_finite_difference() -> None:
    geometry = build_synthetic_stellarator_geometry(nx=6, ny=5, nz=10)
    shape = geometry.shape
    fields = {
        "Ni": np.asarray(1.0 + 0.1 * geometry.radial, dtype=np.float64),
        "Ne": np.asarray(1.0 + 0.1 * geometry.radial, dtype=np.float64),
        "Nn": np.asarray(0.2 + 0.05 * geometry.radial, dtype=np.float64),
        "Pi": np.asarray(0.08 + 0.02 * geometry.radial, dtype=np.float64),
        "Pe": np.asarray(0.10 + 0.03 * geometry.radial, dtype=np.float64),
        "Pn": np.asarray(0.01 + 0.002 * geometry.radial, dtype=np.float64),
    }
    field_names = tuple(fields)
    layout = RecyclingPackedStateLayout(
        field_names=field_names,
        feedback_names=(),
        active_slices=(slice(None), slice(None), slice(None)),
        active_shape=shape,
        field_size=int(np.prod(shape)) * len(field_names),
        field_templates=tuple(fields[name] for name in field_names),
    )
    state = fixed_state_from_fields(fields, feedback_integrals={}, layout=layout)
    packed = pack_fixed_state(state)
    rhs_function = build_fixed_array_rhs(
        lambda active_fields, _feedback: fci_sheath_recycling_field_rhs(active_fields, geometry.maps),
        layout=layout,
    )
    residual = build_fixed_backward_euler_residual(
        rhs_function,
        layout=layout,
        previous_packed_state=packed,
        timestep=1.0e-3,
    )
    tangent = jnp.linspace(-0.2, 0.3, packed.size, dtype=jnp.float64)

    jvp_action = fixed_residual_jvp_action(residual, packed, tangent)
    residual_value, linear_action = linearize_fixed_residual_action(residual, packed)
    finite_difference = (residual(packed + 1.0e-5 * tangent) - residual(packed - 1.0e-5 * tangent)) / 2.0e-5

    assert float(jnp.max(jnp.abs(residual_value - residual(packed)))) < 1.0e-12
    assert float(jnp.max(jnp.abs(jvp_action - linear_action(tangent)))) < 1.0e-12
    assert float(jnp.max(jnp.abs(jvp_action - finite_difference))) < 1.0e-6


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
