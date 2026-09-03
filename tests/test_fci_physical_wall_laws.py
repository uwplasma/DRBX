"""Focused tests for stage-local physical wall bundles."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.geometry import HaloLayout3D
from drbx.native.fci_boundaries import BC_DIRICHLET, BC_NEUMANN
from drbx.native.fci_physical_wall import (
    LegacyParallelVelocityPhysicalWallModel,
    NoFlowPhysicalWallModel,
    PHYSICAL_WALL_MODEL_NAMES,
    SimpleConductingSheathPhysicalWallModel,
    physical_wall_model_from_name,
    resolve_fci_material_wall_endpoint_state,
)

jax.config.update("jax_enable_x64", True)


def _inputs(*, phi=0.0, vi=0.0, ve=0.0, b_sign=1.0):
    shape = (3, 3, 3)
    layout = HaloLayout3D(shape, 1)
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    state = SimpleNamespace(
        density=jnp.ones(shape, dtype=jnp.float64),
        phi=jnp.full(shape, phi, dtype=jnp.float64),
        Te=jnp.full(shape, 4.0, dtype=jnp.float64),
        Ti=jnp.ones(shape, dtype=jnp.float64),
        Vi=jnp.full(shape, vi, dtype=jnp.float64),
        Ve=jnp.full(shape, ve, dtype=jnp.float64),
        vorticity=zeros,
    )

    def face_bfield(axis):
        face_shape = list(shape)
        face_shape[axis] += 1
        values = jnp.zeros(tuple(face_shape) + (3,), dtype=jnp.float64)
        values = values.at[..., axis].set(b_sign)
        return SimpleNamespace(B_contra_owned=values)

    geometry = SimpleNamespace(
        layout=layout,
        face_bfield=SimpleNamespace(
            axes=tuple(face_bfield(axis) for axis in range(3))
        ),
    )
    domain = SimpleNamespace(
        runtime_has_physical_lower=lambda axis: True,
        runtime_has_physical_upper=lambda axis: True,
    )
    parameters = SimpleNamespace(
        Te0=4.0,
        Ti0=1.0,
        tau=2.0,
        mi_over_me=10.0,
    )
    return state, geometry, domain, parameters


def _wall_values(bundle, axis=0, side="upper"):
    return getattr(bundle, "Vi").value_x[-1 if side == "upper" else 0]


def test_model_names_are_the_four_rung_stage_local_choices():
    assert PHYSICAL_WALL_MODEL_NAMES == (
        "legacy-velocity-trace",
        "no-flow",
        "simple-conducting-sheath",
    )
    assert isinstance(
        physical_wall_model_from_name("legacy-velocity-trace"),
        LegacyParallelVelocityPhysicalWallModel,
    )
    assert isinstance(physical_wall_model_from_name("no-flow"), NoFlowPhysicalWallModel)
    assert isinstance(
        physical_wall_model_from_name("simple-conducting-sheath"),
        SimpleConductingSheathPhysicalWallModel,
    )
    with pytest.raises(ValueError):
        physical_wall_model_from_name("ion-bohm-chodura")


def test_no_flow_preserves_scalar_bc_kinds_and_zero_values():
    state, geometry, domain, parameters = _inputs(vi=0.4, ve=-0.7)
    bundle = NoFlowPhysicalWallModel()(state, geometry, domain, parameters)
    for field in (bundle.Vi, bundle.Ve, bundle.phi, bundle.vorticity):
        np.testing.assert_array_equal(jnp.stack((field.kind_x[0], field.kind_x[-1])), BC_DIRICHLET)
    np.testing.assert_allclose(jnp.stack((bundle.Vi.value_x[0], bundle.Vi.value_x[-1])), 0.0)
    np.testing.assert_allclose(jnp.stack((bundle.Ve.value_x[0], bundle.Ve.value_x[-1])), 0.0)
    np.testing.assert_allclose(jnp.stack((bundle.phi.value_x[0], bundle.phi.value_x[-1])), 0.0)
    np.testing.assert_allclose(jnp.stack((bundle.vorticity.value_x[0], bundle.vorticity.value_x[-1])), 0.0)
    for field in (bundle.density, bundle.Te, bundle.Ti):
        np.testing.assert_array_equal(jnp.stack((field.kind_x[0], field.kind_x[-1])), BC_NEUMANN)


def test_legacy_model_keeps_neumann_or_zero_dirichlet_velocity_trace():
    state, geometry, domain, parameters = _inputs()
    neumann = LegacyParallelVelocityPhysicalWallModel("neumann")(
        state, geometry, domain, parameters
    )
    zero = LegacyParallelVelocityPhysicalWallModel("dirichlet-zero")(
        state, geometry, domain, parameters
    )
    np.testing.assert_array_equal(jnp.stack((neumann.Vi.kind_x[0], neumann.Vi.kind_x[-1])), BC_NEUMANN)
    np.testing.assert_array_equal(jnp.stack((neumann.Ve.kind_x[0], neumann.Ve.kind_x[-1])), BC_NEUMANN)
    np.testing.assert_array_equal(jnp.stack((zero.Vi.kind_x[0], zero.Vi.kind_x[-1])), BC_DIRICHLET)
    np.testing.assert_array_equal(jnp.stack((zero.Ve.kind_x[0], zero.Ve.kind_x[-1])), BC_DIRICHLET)


def test_conducting_sheath_default_is_equilibrium_compatible_both_orientations():
    for b_sign in (1.0, -1.0):
        state, geometry, domain, parameters = _inputs(b_sign=b_sign, phi=0.0)
        bundle = SimpleConductingSheathPhysicalWallModel()(
            state, geometry, domain, parameters
        )
        c_b = np.sqrt(4.0 + 2.0)
        expected_upper = b_sign * c_b
        expected_lower = -b_sign * c_b
        np.testing.assert_allclose(bundle.Vi.value_x[-1], expected_upper)
        np.testing.assert_allclose(bundle.Ve.value_x[-1], expected_upper)
        np.testing.assert_allclose(bundle.Vi.value_x[0], expected_lower)
        np.testing.assert_allclose(bundle.Ve.value_x[0], expected_lower)
        np.testing.assert_allclose(
            bundle.Vi.value_x[-1] - bundle.Ve.value_x[-1], 0.0, atol=1e-13
        )


def test_conducting_sheath_passes_supersonic_owner_flow_and_rejects_subsonic():
    # At the upper wall B.n>0.  Outward owner flow passes through; inward or
    # subsonic owner flow is replaced by the sonic target.
    state, geometry, domain, parameters = _inputs(vi=3.0, ve=0.0)
    bundle = SimpleConductingSheathPhysicalWallModel()(
        state, geometry, domain, parameters
    )
    np.testing.assert_allclose(bundle.Vi.value_x[-1], 3.0)

    state, geometry, domain, parameters = _inputs(vi=0.2, ve=0.0)
    bundle = SimpleConductingSheathPhysicalWallModel()(
        state, geometry, domain, parameters
    )
    np.testing.assert_allclose(bundle.Vi.value_x[-1], np.sqrt(6.0))
    np.testing.assert_allclose(bundle.Vi.value_x[0], -np.sqrt(6.0))


def test_conducting_sheath_nonzero_current_for_perturbed_plasma_potential():
    state, geometry, domain, parameters = _inputs(phi=0.25)
    bundle = SimpleConductingSheathPhysicalWallModel()(
        state, geometry, domain, parameters
    )
    c_b = np.sqrt(6.0)
    np.testing.assert_allclose(bundle.Vi.value_x[-1], c_b)
    assert not np.allclose(bundle.Ve.value_x[-1], c_b)
    current = np.asarray(state.density[-1]) * (
        np.asarray(bundle.Vi.value_x[-1]) - np.asarray(bundle.Ve.value_x[-1])
    )
    assert np.max(np.abs(current)) > 1e-8


def test_conducting_sheath_fixed_wall_potential_override_changes_electron_loss():
    state, geometry, domain, parameters = _inputs(phi=0.0)
    default = SimpleConductingSheathPhysicalWallModel()(
        state, geometry, domain, parameters
    )
    override = physical_wall_model_from_name(
        "simple-conducting-sheath",
        conducting_sheath_wall_potential=0.0,
    )(state, geometry, domain, parameters)
    assert not np.allclose(default.Ve.value_x[-1], override.Ve.value_x[-1])
    np.testing.assert_allclose(override.phi.value_x, 0.0)


def test_conducting_sheath_preserves_neumann_thermodynamics_and_provisional_vorticity():
    state, geometry, domain, parameters = _inputs()
    bundle = physical_wall_model_from_name("simple-conducting-sheath")(
        state, geometry, domain, parameters
    )
    for field in (bundle.density, bundle.Te, bundle.Ti):
        np.testing.assert_array_equal(jnp.stack((field.kind_x[0], field.kind_x[-1])), BC_NEUMANN)
    np.testing.assert_array_equal(
        jnp.stack((bundle.Vi.kind_x[0], bundle.Vi.kind_x[-1])), BC_DIRICHLET
    )
    np.testing.assert_array_equal(
        jnp.stack((bundle.Ve.kind_x[0], bundle.Ve.kind_x[-1])), BC_DIRICHLET
    )
    np.testing.assert_array_equal(jnp.stack((bundle.vorticity.kind_x[0], bundle.vorticity.kind_x[-1])), BC_DIRICHLET)
    np.testing.assert_allclose(jnp.stack((bundle.vorticity.value_x[0], bundle.vorticity.value_x[-1])), 0.0)
    np.testing.assert_array_equal(jnp.stack((bundle.phi.kind_x[0], bundle.phi.kind_x[-1])), BC_DIRICHLET)
    np.testing.assert_allclose(jnp.stack((bundle.phi.value_x[0], bundle.phi.value_x[-1])), 0.0)


def test_conducting_sheath_uses_owner_values_on_grazing_faces():
    state, geometry, domain, parameters = _inputs(vi=0.31, ve=-0.27, b_sign=0.0)
    bundle = SimpleConductingSheathPhysicalWallModel()(
        state, geometry, domain, parameters
    )
    np.testing.assert_allclose(bundle.Vi.value_x[0], 0.31)
    np.testing.assert_allclose(bundle.Vi.value_x[-1], 0.31)
    np.testing.assert_allclose(bundle.Ve.value_x[0], -0.27)
    np.testing.assert_allclose(bundle.Ve.value_x[-1], -0.27)


def test_nonpositive_thermodynamics_are_rejected_without_clamping():
    state, geometry, domain, parameters = _inputs()
    state.Te = state.Te.at[0, 0, 0].set(0.0)
    with pytest.raises(ValueError, match="finite positive"):
        SimpleConductingSheathPhysicalWallModel()(state, geometry, domain, parameters)


def test_inverse_sheath_exponential_is_not_clipped():
    state, geometry, domain, parameters = _inputs(phi=-0.1)
    bundle = SimpleConductingSheathPhysicalWallModel()(
        state, geometry, domain, parameters
    )
    # A negative sheath drop is not clipped to zero; the electron speed is
    # correspondingly larger than the equilibrium value.
    assert float(bundle.Ve.value_x[-1][0, 0]) > np.sqrt(6.0)


def test_endpoint_native_sheath_selects_branch_after_interpolating_b_normal():
    """Regression for the HSX theta=34 wrong-sign wall target."""

    plasma = jnp.asarray([[1.0, 1.0, 1.0, 0.0, 0.0]])
    parameters = SimpleNamespace(Te0=1.0, Ti0=1.0, tau=1.0, mi_over_me=1836.0)
    resolved = resolve_fci_material_wall_endpoint_state(
        "simple-conducting-sheath",
        plasma,
        jnp.asarray([0.0]),
        # Continuous MetricEvaluator result at the traced backward endpoint.
        # It is not recoverable accurately by interpolating the coarse
        # coordinate-face samples below.
        jnp.asarray([-0.28590450084598473]),
        jnp.asarray([0.8775673487952317]),
        parameters,
    )
    np.testing.assert_allclose(resolved[0, 3], -np.sqrt(2.0), rtol=1e-13)
    np.testing.assert_allclose(resolved[0, 4], -np.sqrt(2.0), rtol=1e-13)
    assert float((-0.28590450084598473) * resolved[0, 3]) > 0.0

    # Evaluating sign(B.n)*cs on the four regular wall nodes first and then
    # interpolating gives +1.30463: a convex mixture dominated by the wrong
    # wall branch.  Even interpolating B itself gives +0.05123, also the wrong
    # sign compared with the continuous endpoint evaluation above.
    weights = np.asarray(
        [0.25268600122069357, 0.6944722707003752,
         0.014097290137977177, 0.038744437940954106]
    )
    nodal_b = np.asarray(
        [0.10622818693644548, 0.046367322079636265,
         0.1359753339941373, -0.25108642634927064]
    )
    old_target = np.sum(weights * np.sign(nodal_b) * np.sqrt(2.0))
    np.testing.assert_allclose(old_target, 1.3046277431678552, rtol=1e-13)
    assert float(np.sum(weights * nodal_b)) > 0.0
    assert old_target * (-0.28590450084598473) < 0.0


def test_endpoint_native_no_flow_commutes_with_interpolation():
    plasma = jnp.asarray([[1.2, 0.9, 1.1, 3.0, -4.0]])
    resolved = resolve_fci_material_wall_endpoint_state(
        "no-flow",
        plasma,
        jnp.asarray([0.2]),
        jnp.asarray([-0.4]),
        jnp.asarray([1.3]),
        SimpleNamespace(tau=1.0, mi_over_me=1836.0),
    )
    np.testing.assert_allclose(resolved[0, :3], plasma[0, :3])
    np.testing.assert_allclose(resolved[0, 3:], 0.0)
