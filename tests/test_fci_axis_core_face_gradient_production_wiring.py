"""Regression test for production Cartesian-core face-gradient wiring."""

from __future__ import annotations

from pathlib import Path
import sys

import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from axis_regular_operator_support import polar_fixture, scalar_field_halo  # noqa: E402
from drbx.geometry import build_local_conservative_stencil_from_field  # noqa: E402
from drbx.native.fci_drb_EB_rhs import (  # noqa: E402
    FciDrbEBRhsParameters,
    LocalFciDrbEBRhs,
)


def _expected_gradient(geometry, family: str) -> jnp.ndarray:
    if family == "x":
        radius = geometry.grid.x.faces_owned[:, None, None]
        theta = geometry.grid.y.centers_owned[None, :, None]
        eta = geometry.grid.z.centers_owned[None, None, :]
    elif family == "y":
        radius = geometry.grid.x.centers_owned[:, None, None]
        theta = geometry.grid.y.faces_owned[None, :, None]
        eta = geometry.grid.z.centers_owned[None, None, :]
    elif family == "z":
        radius = geometry.grid.x.centers_owned[:, None, None]
        theta = geometry.grid.y.centers_owned[None, :, None]
        eta = geometry.grid.z.faces_owned[None, None, :]
    else:
        raise ValueError(f"unknown face family {family!r}")

    x = radius * jnp.cos(theta)
    y = radius * jnp.sin(theta)
    eta_factor = 1.0 + 0.37 * eta
    full_shape = (radius.shape[0], theta.shape[1], eta.shape[2])
    return jnp.stack(
        (
            2.0 * radius * jnp.cos(theta) * jnp.sin(theta) * eta_factor,
            radius**2 * jnp.cos(2.0 * theta) * eta_factor,
            jnp.broadcast_to(0.37 * x * y, full_shape),
        ),
        axis=-1,
    )


def _production_disabled_rhs(
    geometry,
    domain,
    halo_exchange,
    topology_filler,
    *,
    axis_core_gradient_polynomial_degree=3,
    axis_core_gradient_observation_ring_count=6,
    axis_core_gradient_target_ring_count=3,
):
    return LocalFciDrbEBRhs(
        geometry=geometry,
        domain=domain,
        halo_exchange=halo_exchange,
        topology_filler=topology_filler,
        physical_ghost_filler=None,
        parameters=FciDrbEBRhsParameters(),
        curvature_coefficients_owned=None,
        face_projectors=(None, None, None),
        gmres_config=None,
        face_bc_builder=lambda *_args: None,
        diffusion_only=True,
        axis_regular_axes=(True, False, False),
        curvature_scheme="disabled",
        axis_core_gradient_polynomial_degree=axis_core_gradient_polynomial_degree,
        axis_core_gradient_observation_ring_count=axis_core_gradient_observation_ring_count,
        axis_core_gradient_target_ring_count=axis_core_gradient_target_ring_count,
    )


def test_rhs_axis_core_face_gradient_policy_is_propagated_and_applied():
    """The production RHS context reconstructs exact Cartesian face gradients."""

    geometry, domain, _context, (r, theta, eta), halo_exchange, topology_filler, *_ = (
        polar_fixture(shape=(8, 16, 8))
    )
    polynomial_degree = 2
    observation_ring_count = 5
    target_ring_count = 2

    rhs = _production_disabled_rhs(
        geometry,
        domain,
        halo_exchange,
        topology_filler,
        axis_core_gradient_polynomial_degree=polynomial_degree,
        axis_core_gradient_observation_ring_count=observation_ring_count,
        axis_core_gradient_target_ring_count=target_ring_count,
    )

    context = rhs._stencil_builder_context()
    cell_policy = context.axis_core_cell_gradient_reconstruction
    face_value_policy = context.axis_core_face_reconstruction
    face_policy = context.axis_core_face_gradient_reconstruction
    assert cell_policy is not None
    assert face_value_policy is not None
    assert face_policy is not None
    assert cell_policy.polynomial_degree == polynomial_degree
    assert cell_policy.observation_ring_count == observation_ring_count
    assert cell_policy.target_ring_count == target_ring_count
    assert face_policy.polynomial_degree == polynomial_degree
    assert face_policy.observation_ring_count == observation_ring_count
    assert face_policy.target_ring_count == target_ring_count
    assert face_policy.x_face_count == target_ring_count + 1
    assert face_policy.y_radial_count == target_ring_count
    assert face_policy.z_radial_count == target_ring_count
    assert face_value_policy is face_policy.reconstruction
    assert face_value_policy.radial_ring_count == observation_ring_count
    assert face_value_policy.x_face_count == target_ring_count + 1
    assert face_value_policy.y_radial_count == target_ring_count
    assert face_value_policy.z_radial_count == target_ring_count

    field_halo = scalar_field_halo(
        r,
        theta,
        eta,
        lambda r, theta, eta: (
            r * jnp.cos(theta) * r * jnp.sin(theta) * (1.0 + 0.37 * eta)
        ),
    )
    stencil = build_local_conservative_stencil_from_field(
        field_halo,
        geometry,
        context,
    )

    assert stencil.face_grad.x.shape == (9, 16, 8, 3)
    assert stencil.face_grad.y.shape == (8, 17, 8, 3)
    assert stencil.face_grad.z.shape == (8, 16, 9, 3)
    assert jnp.allclose(
        stencil.face_grad.x[: target_ring_count + 1],
        _expected_gradient(geometry, "x")[: target_ring_count + 1],
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    assert jnp.allclose(
        stencil.face_grad.y[:target_ring_count],
        _expected_gradient(geometry, "y")[:target_ring_count],
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    assert jnp.allclose(
        stencil.face_grad.z[:target_ring_count],
        _expected_gradient(geometry, "z")[:target_ring_count],
        atol=2.0e-11,
        rtol=2.0e-11,
    )


def test_rhs_default_axis_core_face_gradients_are_exact_and_filter_m6_at_32_cubed():
    """The persistent default policy is exact and suppresses nonregular m=6."""

    geometry, domain, _context, (r, theta, eta), halo_exchange, topology_filler, *_ = (
        polar_fixture(shape=(32, 32, 32))
    )
    rhs = _production_disabled_rhs(
        geometry,
        domain,
        halo_exchange,
        topology_filler,
    )
    context = rhs._stencil_builder_context()
    cell_policy = context.axis_core_cell_gradient_reconstruction
    face_value_policy = context.axis_core_face_reconstruction
    face_policy = context.axis_core_face_gradient_reconstruction
    assert cell_policy is not None
    assert face_value_policy is not None
    assert face_policy is not None
    assert (cell_policy.polynomial_degree, cell_policy.observation_ring_count,
            cell_policy.target_ring_count) == (3, 6, 3)
    assert (face_policy.polynomial_degree, face_policy.observation_ring_count,
            face_policy.target_ring_count) == (3, 6, 3)
    assert face_policy.x_face_count == 4
    assert face_policy.y_radial_count == 3
    assert face_policy.z_radial_count == 3
    assert face_value_policy is face_policy.reconstruction
    assert face_value_policy.radial_ring_count == 6
    assert face_value_policy.x_face_count == 4
    assert face_value_policy.y_radial_count == 3
    assert face_value_policy.z_radial_count == 3

    polynomial_halo = scalar_field_halo(
        r,
        theta,
        eta,
        lambda r, theta, eta: (
            r * jnp.cos(theta) * r * jnp.sin(theta) * (1.0 + 0.37 * eta)
        ),
    )
    polynomial_stencil = build_local_conservative_stencil_from_field(
        polynomial_halo,
        geometry,
        context,
    )
    protected = {
        "x": (polynomial_stencil.face_grad.x[:4], _expected_gradient(geometry, "x")[:4]),
        "y": (polynomial_stencil.face_grad.y[:3], _expected_gradient(geometry, "y")[:3]),
        "z": (polynomial_stencil.face_grad.z[:3], _expected_gradient(geometry, "z")[:3]),
    }
    for actual, expected in protected.values():
        assert jnp.all(jnp.isfinite(actual))
        assert jnp.allclose(actual, expected, atol=5.0e-11, rtol=5.0e-11)

    m6_halo = scalar_field_halo(
        r,
        theta,
        eta,
        lambda r, theta, eta: (
            (1.0 + 0.8 * r) * jnp.cos(6.0 * theta) * (1.0 + 0.37 * eta)
        ),
    )
    m6_stencil = build_local_conservative_stencil_from_field(
        m6_halo,
        geometry,
        context,
    )
    m6_protected = (
        m6_stencil.face_grad.x[:4],
        m6_stencil.face_grad.y[:3],
        m6_stencil.face_grad.z[:3],
    )
    for gradient in m6_protected:
        assert jnp.all(jnp.isfinite(gradient))
        assert jnp.max(jnp.abs(gradient)) < 1.0e-10

    m6_face_values = (
        m6_stencil.face_values.x[:4],
        m6_stencil.face_values.y[:3],
        m6_stencil.face_values.z[:3],
    )
    for face_values in m6_face_values:
        assert jnp.all(jnp.isfinite(face_values))
        assert jnp.max(jnp.abs(face_values)) < 1.0e-10
