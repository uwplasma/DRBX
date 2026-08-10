"""Axis-regular numerical checks for the non-FCI conservative operators."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drbx.geometry import (  # noqa: E402
    AxisCoreFaceReconstruction3D,
    HaloLayout3D,
    LocalCurvatureFaceCoefficients3D,
    LocalDomain3D,
    ShardSpec3D,
    SIDE_AXIS_REGULAR,
    SIDE_PHYSICAL,
    SIDE_SIMPLE_PERIODIC,
    StencilBuilderContext,
    build_local_curvature_face_coefficients,
    build_local_conservative_stencil_from_field,
)

try:  # Added by the forthcoming face-gradient production implementation.
    from drbx.geometry import (  # noqa: E402
        AxisCoreFaceGradientReconstruction3D,
        build_axis_core_face_gradient_reconstruction,
    )
except ImportError:  # pragma: no cover - expected before that API lands.
    AxisCoreFaceGradientReconstruction3D = None
    build_axis_core_face_gradient_reconstruction = None

from drbx.native.fci_operators import (  # noqa: E402
    build_local_parallel_laplacian_face_projectors,
    build_local_perp_laplacian_face_projectors,
    build_local_perp_laplacian_stencil,
    build_local_projected_laplacian_flux_stencil,
    local_curvature_conservative_op,
    local_curvature_upwind_conservative_op,
    local_divergence_conservative_op,
    local_grad_parallel_op_conservative,
    local_parallel_div_b_op,
    local_parallel_flux_div_op,
    local_parallel_laplacian_conservative_op,
    local_perp_laplacian_conservative_op,
)

try:  # Keep the focused file runnable on the pre-compatible-flux branch.
    from drbx.native.fci_operators import (  # noqa: E402
        local_poisson_bracket_compatible_flux_op,
    )
except ImportError:  # pragma: no cover - compatibility fallback.
    local_poisson_bracket_compatible_flux_op = None
from drbx.native.fci_boundaries import (  # noqa: E402
    BC_NEUMANN,
    ConservativeStencil3D,
    LocalBoundaryFaceBC3D,
)
from drbx.native.fci_drb_EB_rhs import LocalFciDrbEBRhs  # noqa: E402

from axis_regular_operator_support import (  # noqa: E402
    owned,
    polar_fixture,
    scalar_field_halo,
)
from test_fci_operators_domain_decomp import (  # noqa: E402
    _build_local_geometry,
    make_mesh_for_shard_counts,
    put_scalar_field_on_mesh,
)
from test_fci_geometry_axis_regular_curvature import _build_polar_geometry  # noqa: E402


def _stencil(field, geometry, context):
    return build_local_conservative_stencil_from_field(field, geometry, context)


def _cartesian_core_api_available():
    """Whether the Cartesian-core face reconstruction API is installed."""

    context_fields = getattr(StencilBuilderContext, "__dataclass_fields__", {})
    stencil_fields = getattr(ConservativeStencil3D, "__dataclass_fields__", {})
    return (
        AxisCoreFaceReconstruction3D is not None
        and "axis_core_face_reconstruction" in context_fields
        and "face_values" in stencil_fields
        and not hasattr(StencilBuilderContext, "axis_regular_face_" + "polynomial_degree")
    )


def _require_cartesian_core_api():
    if not _cartesian_core_api_available():
        pytest.skip("Cartesian axis-core face reconstruction API is not installed")


def _axis_core_context(context):
    _require_cartesian_core_api()
    if context.axis_core_face_reconstruction is None:
        pytest.skip("axis-regular domain did not install a Cartesian core payload")
    return context


def _face_gradient_api_available():
    context_fields = getattr(StencilBuilderContext, "__dataclass_fields__", {})
    return (
        AxisCoreFaceGradientReconstruction3D is not None
        and build_axis_core_face_gradient_reconstruction is not None
        and "axis_core_face_gradient_reconstruction" in context_fields
    )


def _require_face_gradient_api():
    if not _face_gradient_api_available():
        pytest.skip("Cartesian axis-core face-gradient API is not installed")


def _face_gradient_context(context, *, degree=3):
    """Build the public face-gradient payload with the CLI fit settings."""

    _require_face_gradient_api()
    payload = build_axis_core_face_gradient_reconstruction(
        context.layout,
        context.domain,
        polynomial_degree=degree,
        observation_ring_count=6,
        target_ring_count=3,
    )
    return replace(context, axis_core_face_gradient_reconstruction=payload)


def _non_axis_domain(domain):
    """Disable only the lower-radial axis closure for an ordinary reference."""

    return replace(
        domain,
        shard_spec=replace(
            domain.shard_spec,
            axis_regular_axes=(False, False, False),
            side_kind_lower=(
                SIDE_PHYSICAL,
                domain.shard_spec.lower_side_kind(1),
                domain.shard_spec.lower_side_kind(2),
            ),
        ),
    )


def _nonperiodic_eta_domain(domain):
    """Use analytic linear eta halos while retaining the lower radial axis."""

    return replace(
        domain,
        shard_spec=replace(
            domain.shard_spec,
            periodic_axes=(False, True, False),
            side_kind_lower=(
                domain.shard_spec.lower_side_kind(0),
                domain.shard_spec.lower_side_kind(1),
                SIDE_PHYSICAL,
            ),
            side_kind_upper=(
                domain.shard_spec.upper_side_kind(0),
                domain.shard_spec.upper_side_kind(1),
                SIDE_PHYSICAL,
            ),
        ),
    )


def _eta_factor(z):
    return 1.0 + 0.37 * z


def _eta_factor_prime(z):
    return jnp.full_like(z, 0.37)


def _cartesian_monomial_with_eta(r, theta, z, p, q):
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    return x**p * y**q * _eta_factor(z)


def _nonregular_m6_with_eta(r, theta, z):
    """O(1) m=6 contamination with nonzero radial, angular, and eta traces."""

    return (1.0 + 0.8 * r) * jnp.cos(6.0 * theta) * _eta_factor(z)


def _expected_face_gradient_with_eta(geometry, family, p, q):
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
    dx = p * x ** (p - 1) * y**q if p else jnp.zeros_like(x)
    dy = q * x**p * y ** (q - 1) if q else jnp.zeros_like(y)
    cartesian_value = x**p * y**q
    return jnp.stack(
        (
            (jnp.cos(theta) * dx + jnp.sin(theta) * dy) * _eta_factor(eta),
            (-radius * jnp.sin(theta) * dx + radius * jnp.cos(theta) * dy)
            * _eta_factor(eta),
            cartesian_value * _eta_factor_prime(eta),
        ),
        axis=-1,
    )


def _protected_face_gradient_components(local, payload):
    """Return x faces 0..3 and y/z radial rows 0..2."""

    assert payload.target_ring_count == 3
    return (
        local.face_grad.x[: payload.target_ring_count + 1],
        local.face_grad.y[: payload.target_ring_count],
        local.face_grad.z[: payload.target_ring_count],
    )


def _ordinary_face_values(local):
    """Reference arithmetic cell-to-face values for all three directions."""

    def axis_faces(stencil, axis):
        center = jnp.asarray(stencil.center)
        minus = jnp.asarray(stencil.minus)
        plus = jnp.asarray(stencil.plus)
        lower = 0.5 * (
            jnp.take(center, 0, axis=axis)
            + jnp.take(minus, 0, axis=axis)
        )
        upper = 0.5 * (center + plus)
        return jnp.concatenate((jnp.expand_dims(lower, axis=axis), upper), axis=axis)

    return (
        axis_faces(local.x, 0),
        axis_faces(local.y, 1),
        axis_faces(local.z, 2),
    )


def _regular_mode_field(r, theta, z, m):
    del z
    return r**m * jnp.cos(m * theta)


def _cartesian_monomial_field(r, theta, z, p, q):
    del z
    return (r * jnp.cos(theta))**p * (r * jnp.sin(theta))**q


def _x_face_cartesian_coordinates(geometry):
    radius = geometry.grid.x.faces_owned[:, None, None]
    theta = geometry.grid.y.centers_owned[None, :, None]
    return radius * jnp.cos(theta), radius * jnp.sin(theta)


def _y_face_cartesian_coordinates(geometry):
    radius = geometry.grid.x.centers_owned[:, None, None]
    theta = geometry.grid.y.faces_owned[None, :, None]
    return radius * jnp.cos(theta), radius * jnp.sin(theta)


def _assert_core_face_matches(local, geometry, p, q, *, atol=5.0e-11):
    x_coord, y_coord = _x_face_cartesian_coordinates(geometry)
    x_expected = x_coord**p * y_coord**q
    x_actual = local.face_values.x[:3]
    assert jnp.allclose(x_actual, x_expected[:3], atol=atol, rtol=atol)

    x_coord, y_coord = _y_face_cartesian_coordinates(geometry)
    y_expected = x_coord**p * y_coord**q
    y_actual = local.face_values.y[:2]
    assert jnp.allclose(y_actual, y_expected[:2], atol=atol, rtol=atol)


def _logical_gradient_of_cartesian_monomial(radius, theta, p, q):
    x = radius * jnp.cos(theta)
    y = radius * jnp.sin(theta)
    dx = p * x ** (p - 1) * y**q if p else jnp.zeros_like(x)
    dy = q * x**p * y ** (q - 1) if q else jnp.zeros_like(y)
    du = jnp.cos(theta) * dx + jnp.sin(theta) * dy
    dtheta = -radius * jnp.sin(theta) * dx + radius * jnp.cos(theta) * dy
    return jnp.stack((du, dtheta, jnp.zeros_like(du)), axis=-1)


def _expected_face_gradient(geometry, family, p, q):
    if family == "x":
        radius = geometry.grid.x.faces_owned[:, None, None]
        theta = geometry.grid.y.centers_owned[None, :, None]
    elif family == "y":
        radius = geometry.grid.x.centers_owned[:, None, None]
        theta = geometry.grid.y.faces_owned[None, :, None]
    elif family == "z":
        radius = geometry.grid.x.centers_owned[:, None, None]
        theta = geometry.grid.y.centers_owned[None, :, None]
    else:
        raise ValueError(f"unknown face family {family!r}")
    return _logical_gradient_of_cartesian_monomial(radius, theta, p, q)


def _assert_pytree_leaves_equal(left_leaves, right_leaves):
    assert len(left_leaves) == len(right_leaves)
    for left, right in zip(left_leaves, right_leaves):
        if hasattr(left, "shape") and hasattr(right, "shape"):
            assert bool(jnp.all(jnp.asarray(left) == jnp.asarray(right)))
        else:
            assert left == right


def test_materialized_face_values_preserve_ordinary_averages():
    """Materialized faces retain arithmetic averages away from the core."""

    geometry, domain, context, (r, theta, z), *_ = polar_fixture()
    field = scalar_field_halo(r, theta, z, lambda r, t, z: r + 0.25 * t + z)

    non_axis_domain = replace(
        domain,
        shard_spec=replace(
            domain.shard_spec,
            axis_regular_axes=(False, False, False),
            side_kind_lower=(SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_SIMPLE_PERIODIC),
        ),
    )
    axis_local = _stencil(
        field,
        geometry,
        StencilBuilderContext(layout=geometry.layout, domain=non_axis_domain),
    )
    expected = _ordinary_face_values(axis_local)
    for actual, reference in zip(
        (axis_local.face_values.x, axis_local.face_values.y, axis_local.face_values.z),
        expected,
    ):
        assert jnp.allclose(actual, reference, atol=1.0e-13, rtol=1.0e-13)

    # In particular, ordinary values are unchanged away from the first x-face.
    assert jnp.allclose(
        axis_local.face_values.x[1:], expected[0][1:], atol=1.0e-13, rtol=1.0e-13
    )


def test_axis_core_payload_is_installed_and_signed_degree_api_is_absent():
    _require_cartesian_core_api()
    geometry, domain, _context, *_ = polar_fixture(shape=(8, 32, 2))
    context = StencilBuilderContext(layout=geometry.layout, domain=domain)
    assert isinstance(context.axis_core_face_reconstruction, AxisCoreFaceReconstruction3D)
    assert not hasattr(context, "axis_regular_face_" + "polynomial_degree")
    payload = context.axis_core_face_reconstruction
    assert payload.polynomial_degree == 3
    assert payload.radial_ring_count == 3
    assert payload.x_face_count == 3
    assert payload.y_radial_count == 2
    assert payload.z_radial_count == 2


@pytest.mark.parametrize(
    "p,q",
    [(p, q) for total in range(4) for p in range(total + 1) for q in (total - p,)],
)
def test_cartesian_core_reproduces_complete_cubic_monomials(p, q):
    geometry, domain, context, (r, theta, z), *_ = polar_fixture(
        shape=(8, 32, 2), halo_width=1
    )
    context = _axis_core_context(context)
    field = scalar_field_halo(
        r, theta, z, lambda r, t, z: _cartesian_monomial_field(r, t, z, p, q)
    )
    local = _stencil(field, geometry, context)
    _assert_core_face_matches(local, geometry, p, q)


@pytest.mark.parametrize(
    "p,q",
    [(p, q) for total in range(4) for p in range(total + 1) for q in (total - p,)],
)
def test_cartesian_core_face_gradients_reproduce_complete_cubic_monomials(p, q):
    geometry, _domain, context, (r, theta, z), *_ = polar_fixture(
        shape=(8, 32, 2), halo_width=1
    )
    context = _axis_core_context(context)
    field = scalar_field_halo(
        r, theta, z, lambda r, t, z: _cartesian_monomial_field(r, t, z, p, q)
    )
    local = _stencil(field, geometry, context)
    payload = context.axis_core_face_reconstruction
    assert payload is not None

    expected_x = _expected_face_gradient(geometry, "x", p, q)
    expected_y = _expected_face_gradient(geometry, "y", p, q)
    expected_z = _expected_face_gradient(geometry, "z", p, q)
    assert jnp.allclose(
        local.face_grad.x[: payload.x_face_count],
        expected_x[: payload.x_face_count],
        atol=8.0e-10,
        rtol=8.0e-10,
    )
    assert jnp.allclose(
        local.face_grad.y[: payload.y_radial_count],
        expected_y[: payload.y_radial_count],
        atol=8.0e-10,
        rtol=8.0e-10,
    )
    assert jnp.allclose(
        local.face_grad.z[: payload.z_radial_count],
        expected_z[: payload.z_radial_count],
        atol=8.0e-10,
        rtol=8.0e-10,
    )


def test_cartesian_core_face_eta_gradient_uses_scalar_target_functional():
    geometry, domain, context, (r, theta, z), *_ = polar_fixture(
        shape=(8, 32, 8), halo_width=1
    )
    context = _axis_core_context(context)
    field = scalar_field_halo(r, theta, z, lambda r, t, z: jnp.sin(z))
    reconstructed = _stencil(field, geometry, context)

    non_axis_domain = replace(
        domain,
        shard_spec=replace(
            domain.shard_spec,
            axis_regular_axes=(False, False, False),
            side_kind_lower=(
                SIDE_PHYSICAL,
                SIDE_SIMPLE_PERIODIC,
                SIDE_SIMPLE_PERIODIC,
            ),
        ),
    )
    ordinary = _stencil(
        field,
        geometry,
        StencilBuilderContext(layout=geometry.layout, domain=non_axis_domain),
    )
    payload = context.axis_core_face_reconstruction
    assert payload is not None
    for actual, reference, count in (
        (reconstructed.face_grad.x, ordinary.face_grad.x, payload.x_face_count),
        (reconstructed.face_grad.y, ordinary.face_grad.y, payload.y_radial_count),
        (reconstructed.face_grad.z, ordinary.face_grad.z, payload.z_radial_count),
    ):
        assert jnp.max(jnp.abs(actual[:count, ..., :2])) < 2.0e-12
        assert jnp.allclose(
            actual[:count, ..., 2],
            reference[:count, ..., 2],
            atol=2.0e-12,
            rtol=2.0e-12,
        )


def test_cartesian_core_face_gradients_filter_unrepresented_high_mode():
    geometry, _domain, context, (r, theta, z), *_ = polar_fixture(
        shape=(8, 32, 2), halo_width=1
    )
    context = _axis_core_context(context)
    field = scalar_field_halo(
        r, theta, z, lambda r, t, z: _regular_mode_field(r, t, z, 8)
    )
    local = _stencil(field, geometry, context)
    payload = context.axis_core_face_reconstruction
    assert payload is not None
    assert jnp.max(jnp.abs(local.face_grad.x[: payload.x_face_count])) < 5.0e-9
    assert jnp.max(jnp.abs(local.face_grad.y[: payload.y_radial_count])) < 5.0e-9
    assert jnp.max(jnp.abs(local.face_grad.z[: payload.z_radial_count])) < 5.0e-9


@pytest.mark.parametrize("m", range(17))
def test_cartesian_core_resolved_mode_sweep_reproduces_low_modes_and_filters_high_modes(m):
    geometry, _domain, context, (r, theta, z), *_ = polar_fixture(
        shape=(8, 32, 2), halo_width=1
    )
    context = _axis_core_context(context)
    field = scalar_field_halo(r, theta, z, lambda r, t, z: _regular_mode_field(r, t, z, m))
    local = _stencil(field, geometry, context)

    x_coord, y_coord = _x_face_cartesian_coordinates(geometry)
    x_radius = jnp.hypot(x_coord, y_coord)
    x_expected = x_radius**m * jnp.cos(m * jnp.arctan2(y_coord, x_coord))
    x_expected = jnp.where(m == 0, jnp.ones_like(x_expected), x_expected)
    y_coord_x, y_coord_y = _y_face_cartesian_coordinates(geometry)
    y_radius = jnp.hypot(y_coord_x, y_coord_y)
    y_expected = y_radius**m * jnp.cos(m * jnp.arctan2(y_coord_y, y_coord_x))
    y_expected = jnp.where(m == 0, jnp.ones_like(y_expected), y_expected)

    if m <= 3:
        assert jnp.allclose(local.face_values.x[:3], x_expected[:3], atol=5.0e-10, rtol=5.0e-10)
        assert jnp.allclose(local.face_values.y[:2], y_expected[:2], atol=5.0e-10, rtol=5.0e-10)
    else:
        assert jnp.max(jnp.abs(local.face_values.x[:3])) < 5.0e-10
        assert jnp.max(jnp.abs(local.face_values.y[:2])) < 5.0e-10


def test_face_gradient_context_installs_default_payload_and_round_trips():
    _require_face_gradient_api()
    geometry, domain, _context, *_ = polar_fixture(shape=(10, 32, 8))
    context = StencilBuilderContext(layout=geometry.layout, domain=domain)
    payload = context.axis_core_face_gradient_reconstruction
    assert isinstance(payload, AxisCoreFaceGradientReconstruction3D)
    assert payload.polynomial_degree == 3
    assert payload.observation_ring_count == 6
    assert payload.target_ring_count == 3

    leaves, treedef = jax.tree_util.tree_flatten(context)
    copied = jax.tree_util.tree_unflatten(treedef, leaves)
    copied_payload = copied.axis_core_face_gradient_reconstruction
    assert isinstance(copied_payload, AxisCoreFaceGradientReconstruction3D)
    assert copied_payload.polynomial_degree == 3
    assert copied_payload.observation_ring_count == 6
    assert copied_payload.target_ring_count == 3
    assert jnp.array_equal(
        copied_payload.observation_to_coefficient_weights,
        payload.observation_to_coefficient_weights,
    )


@pytest.mark.parametrize(
    "p,q",
    [(p, q) for total in range(4) for p in range(total + 1) for q in (total - p,)],
)
def test_face_gradients_reproduce_eta_dependent_cartesian_cubics_on_all_faces(p, q):
    """The protected x/y/z face targets reproduce every logical component."""

    geometry, domain, _context, (r, theta, z), *_ = polar_fixture(
        shape=(10, 32, 8), halo_width=1
    )
    domain = _nonperiodic_eta_domain(domain)
    context = _face_gradient_context(
        StencilBuilderContext(layout=geometry.layout, domain=domain), degree=3
    )
    field = scalar_field_halo(
        r,
        theta,
        z,
        lambda r, t, z: _cartesian_monomial_with_eta(r, t, z, p, q),
    )
    local = _stencil(field, geometry, context)
    payload = context.axis_core_face_gradient_reconstruction
    assert payload is not None
    actual_x, actual_y, actual_z = _protected_face_gradient_components(local, payload)
    expected_x = _expected_face_gradient_with_eta(geometry, "x", p, q)
    expected_y = _expected_face_gradient_with_eta(geometry, "y", p, q)
    expected_z = _expected_face_gradient_with_eta(geometry, "z", p, q)
    for actual, expected in zip(
        (actual_x, actual_y, actual_z),
        (expected_x[:4], expected_y[:3], expected_z[:3]),
    ):
        assert actual.shape == expected.shape
        assert jnp.allclose(actual, expected, atol=2.0e-9, rtol=2.0e-9)


def test_face_gradient_payload_degree_four_is_not_silently_capped_at_cubic():
    geometry, domain, _context, (r, theta, z), *_ = polar_fixture(
        shape=(10, 32, 8), halo_width=1
    )
    domain = _nonperiodic_eta_domain(domain)
    context = _face_gradient_context(
        StencilBuilderContext(layout=geometry.layout, domain=domain), degree=4
    )
    payload = context.axis_core_face_gradient_reconstruction
    assert payload is not None
    assert payload.polynomial_degree == 4
    field = scalar_field_halo(
        r,
        theta,
        z,
        lambda r, t, z: _cartesian_monomial_with_eta(r, t, z, 4, 0),
    )
    local = _stencil(field, geometry, context)
    expected_x = _expected_face_gradient_with_eta(geometry, "x", 4, 0)[:4]
    expected_y = _expected_face_gradient_with_eta(geometry, "y", 4, 0)[:3]
    expected_z = _expected_face_gradient_with_eta(geometry, "z", 4, 0)[:3]
    for actual, expected in zip(
        _protected_face_gradient_components(local, payload),
        (expected_x, expected_y, expected_z),
    ):
        assert jnp.allclose(actual, expected, atol=3.0e-9, rtol=3.0e-9)


def test_face_gradient_reconstruction_filters_o1_m6_in_every_face_component():
    """A nonregular O(1) m=6 trace is removed from all protected targets."""

    geometry, domain, _context, (r, theta, z), *_ = polar_fixture(
        shape=(10, 32, 8), halo_width=1
    )
    domain = _nonperiodic_eta_domain(domain)
    context = _face_gradient_context(
        StencilBuilderContext(layout=geometry.layout, domain=domain), degree=3
    )
    ordinary_context = StencilBuilderContext(
        layout=geometry.layout, domain=_non_axis_domain(domain)
    )
    field = scalar_field_halo(r, theta, z, _nonregular_m6_with_eta)
    reconstructed = _stencil(field, geometry, context)
    ordinary = _stencil(field, geometry, ordinary_context)
    payload = context.axis_core_face_gradient_reconstruction
    assert payload is not None
    for actual, reference in zip(
        _protected_face_gradient_components(reconstructed, payload),
        _protected_face_gradient_components(ordinary, payload),
    ):
        for component in range(3):
            assert jnp.max(jnp.abs(actual[..., component])) < 2.0e-8
            assert jnp.max(jnp.abs(reference[..., component])) > 1.0e-3


def test_face_gradient_is_ordinary_outside_three_target_radial_cells():
    geometry, domain, _context, (r, theta, z), *_ = polar_fixture(
        shape=(10, 32, 8), halo_width=1
    )
    domain = _nonperiodic_eta_domain(domain)
    context = _face_gradient_context(
        StencilBuilderContext(layout=geometry.layout, domain=domain), degree=3
    )
    ordinary_context = StencilBuilderContext(
        layout=geometry.layout, domain=_non_axis_domain(domain)
    )
    field = scalar_field_halo(
        r, theta, z, lambda r, t, z: jnp.sin(1.7 * r + 0.2 * t) * _eta_factor(z)
    )
    reconstructed = _stencil(field, geometry, context)
    ordinary = _stencil(field, geometry, ordinary_context)
    assert jnp.allclose(
        reconstructed.face_grad.x[4:], ordinary.face_grad.x[4:], atol=1.0e-13, rtol=1.0e-13
    )
    assert jnp.allclose(
        reconstructed.face_grad.y[3:], ordinary.face_grad.y[3:], atol=1.0e-13, rtol=1.0e-13
    )
    assert jnp.allclose(
        reconstructed.face_grad.z[3:], ordinary.face_grad.z[3:], atol=1.0e-13, rtol=1.0e-13
    )


def test_face_gradient_reconstruction_reduces_operator_m6_core_leakage():
    """Projected Laplacian and compatible bracket consume reconstructed faces."""

    if local_poisson_bracket_compatible_flux_op is None:
        pytest.skip("compatible-flux bracket is not installed")
    geometry, domain, _context, (r, theta, z), *_ = polar_fixture(
        shape=(10, 32, 8), halo_width=1
    )
    domain = _nonperiodic_eta_domain(domain)
    context = _face_gradient_context(
        StencilBuilderContext(layout=geometry.layout, domain=domain), degree=3
    )
    ordinary_context = StencilBuilderContext(
        layout=geometry.layout, domain=_non_axis_domain(domain)
    )
    contaminated = scalar_field_halo(r, theta, z, _nonregular_m6_with_eta)
    regular = scalar_field_halo(
        r, theta, z, lambda r, t, z: r * jnp.cos(t) * _eta_factor(z)
    )
    reconstructed_contaminated = _stencil(contaminated, geometry, context)
    reconstructed_regular = _stencil(regular, geometry, context)
    ordinary_contaminated = _stencil(contaminated, geometry, ordinary_context)
    ordinary_regular = _stencil(regular, geometry, ordinary_context)

    projectors = build_local_perp_laplacian_face_projectors(
        geometry, domain, axis_regular_axes=(True, False, False)
    )
    reconstructed_laplacian = local_perp_laplacian_conservative_op(
        reconstructed_contaminated,
        geometry,
        domain,
        face_projectors=projectors,
        axis_regular_axes=(True, False, False),
    )
    ordinary_laplacian = local_perp_laplacian_conservative_op(
        ordinary_contaminated,
        geometry,
        domain,
        face_projectors=projectors,
        axis_regular_axes=(True, False, False),
    )
    reconstructed_bracket = local_poisson_bracket_compatible_flux_op(
        reconstructed_contaminated,
        reconstructed_regular,
        geometry,
        domain=domain,
        axis_regular_axes=(True, False, False),
    )
    ordinary_bracket = local_poisson_bracket_compatible_flux_op(
        ordinary_contaminated,
        ordinary_regular,
        geometry,
        domain=domain,
        axis_regular_axes=(True, False, False),
    )
    for filtered, reference in (
        (reconstructed_laplacian[:3], ordinary_laplacian[:3]),
        (reconstructed_bracket[:3], ordinary_bracket[:3]),
    ):
        assert jnp.all(jnp.isfinite(filtered))
        assert jnp.max(jnp.abs(reference)) > 1.0e-2
        # The face-gradient reconstruction must materially suppress the
        # singular core drive.  The projected Laplacian retains ordinary
        # finite-volume truncation error, so require a strong reduction rather
        # than exact zero.
        assert jnp.max(jnp.abs(filtered)) < 0.25 * jnp.max(jnp.abs(reference))


def test_theta_sharded_face_gradient_reconstruction_matches_serial():
    """The global coefficient fit and local face targets agree across theta shards."""

    _require_face_gradient_api()
    if jax.local_device_count() < 2:
        pytest.skip("requires at least two local devices for theta sharding")

    shard_counts = (1, 2, 1)
    global_shape = (10, 32, 4)
    local_shape = (10, 16, 4)
    halo_width = 1
    geometry, domain = _build_polar_geometry(
        shape=local_shape,
        global_shape=global_shape,
        halo_width=halo_width,
        shard_counts=shard_counts,
        coordinate_shard_index=(0, 0, 0),
        mesh_axis_names=(None, "y", None),
    )
    context = _face_gradient_context(
        StencilBuilderContext(layout=geometry.layout, domain=domain), degree=3
    )
    global_r = (jnp.arange(global_shape[0], dtype=jnp.float64) + 0.5) / global_shape[0]
    global_theta = 2.0 * jnp.pi * (
        jnp.arange(global_shape[1], dtype=jnp.float64) + 0.5
    ) / global_shape[1]
    global_z = 2.0 * jnp.pi * (
        jnp.arange(global_shape[2], dtype=jnp.float64) + 0.5
    ) / global_shape[2]
    rr, tt, zz = jnp.meshgrid(global_r, global_theta, global_z, indexing="ij")
    # Keep eta constant so the comparison isolates theta-sharded Cartesian
    # fitting rather than requiring a separate distributed eta halo exchange.
    global_field = (rr * jnp.cos(tt)) ** 2

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        def kernel(field_owned):
            field_halo = jnp.pad(
                field_owned,
                ((halo_width, halo_width),) * 3,
                mode="edge",
            )
            local = _stencil(field_halo, geometry, context)
            return (
                local.face_grad.x[:4],
                local.face_grad.y[:3, :-1],
                local.face_grad.z[:3],
            )

        mapped = shard_map(
            kernel,
            mesh=mesh,
            in_specs=P("x", "y", "z"),
            out_specs=(
                P(None, "y", None, None),
                P(None, "y", None, None),
                P(None, "y", None, None),
            ),
            check_rep=False,
        )
        sharded_x, sharded_y, sharded_z = mapped(global_field)

    serial_geometry, serial_domain, _serial_context, (r, theta, z), *_ = polar_fixture(
        shape=global_shape, halo_width=halo_width
    )
    serial_context = _face_gradient_context(
        StencilBuilderContext(layout=serial_geometry.layout, domain=serial_domain),
        degree=3,
    )
    serial_field = scalar_field_halo(
        r, theta, z,
        lambda r, t, z: (r * jnp.cos(t)) ** 2,
    )
    serial_local = _stencil(serial_field, serial_geometry, serial_context)
    assert jnp.allclose(
        sharded_x, serial_local.face_grad.x[:4], atol=2.0e-9, rtol=2.0e-9
    )
    assert jnp.allclose(
        sharded_y, serial_local.face_grad.y[:3, :-1], atol=2.0e-9, rtol=2.0e-9
    )
    assert jnp.allclose(
        sharded_z, serial_local.face_grad.z[:3], atol=2.0e-9, rtol=2.0e-9
    )


def _unit_x_face_bfield(face_bfield):
    contra = jnp.zeros_like(face_bfield.B_contra_halo).at[..., 0].set(1.0)
    return replace(
        face_bfield,
        B_contra_halo=contra,
        Bmag_halo=jnp.ones_like(face_bfield.Bmag_halo),
    )


def test_centered_curvature_and_parallel_flux_consume_cartesian_core_faces():
    geometry, domain, context, (r, theta, z), *_ = polar_fixture(
        shape=(8, 32, 2), halo_width=1
    )
    context = _axis_core_context(context)
    field = scalar_field_halo(r, theta, z, lambda r, t, z: _regular_mode_field(r, t, z, 8))
    local = _stencil(field, geometry, context)
    coefficients = _constant_coefficients(geometry, value=0.0)
    coefficients = replace(coefficients, x=jnp.ones_like(coefficients.x))
    curvature = local_curvature_conservative_op(
        local, geometry, coefficients, domain=domain, axis_regular_axes=(True, False, False)
    )
    geometry_x = replace(
        geometry,
        face_bfield=replace(
            geometry.face_bfield,
            x=_unit_x_face_bfield(geometry.face_bfield.x),
            y=_unit_x_face_bfield(geometry.face_bfield.y),
            z=_unit_x_face_bfield(geometry.face_bfield.z),
        ),
    )
    parallel = local_parallel_flux_div_op(
        local, geometry_x, domain, axis_regular_axes=(True, False, False)
    )
    assert jnp.all(jnp.isfinite(curvature))
    assert jnp.all(jnp.isfinite(parallel))
    # A cubic Cartesian fit has no m=8 component.  The first two cell rings
    # only see reconstructed x faces 0..2, so neither consumer may leak m=8
    # through the core materialized faces.
    assert jnp.max(jnp.abs(curvature[:2])) < 5.0e-9
    assert jnp.max(jnp.abs(parallel[:2])) < 5.0e-9


def test_axis_core_context_and_payload_round_trip_through_pytree():
    _require_cartesian_core_api()
    geometry, domain, context, *_ = polar_fixture(shape=(8, 32, 2))
    context = _axis_core_context(context)
    payload = context.axis_core_face_reconstruction
    payload_leaves, payload_treedef = jax.tree_util.tree_flatten(payload)
    payload_copy = jax.tree_util.tree_unflatten(payload_treedef, payload_leaves)
    assert isinstance(payload_copy, AxisCoreFaceReconstruction3D)
    copied_leaves, copied_treedef = jax.tree_util.tree_flatten(payload_copy)
    assert payload_treedef == copied_treedef
    _assert_pytree_leaves_equal(payload_leaves, copied_leaves)

    context_leaves, context_treedef = jax.tree_util.tree_flatten(context)
    context_copy = jax.tree_util.tree_unflatten(context_treedef, context_leaves)
    copied_context_leaves, copied_context_treedef = jax.tree_util.tree_flatten(context_copy)
    assert context_treedef == copied_context_treedef
    _assert_pytree_leaves_equal(context_leaves, copied_context_leaves)


def test_theta_sharded_cartesian_core_matches_serial_reconstruction():
    """The coefficient fit is globally reduced while targets remain local."""

    if jax.local_device_count() < 2:
        pytest.skip("requires at least two local devices for theta sharding")
    shard_counts = (1, 2, 1)
    global_shape = (8, 32, 2)
    local_shape = (8, 16, 2)
    halo_width = 1
    geometry, domain = _build_polar_geometry(
        shape=local_shape,
        global_shape=global_shape,
        halo_width=halo_width,
        shard_counts=shard_counts,
        coordinate_shard_index=(0, 0, 0),
        mesh_axis_names=(None, "y", None),
    )
    context = StencilBuilderContext(layout=geometry.layout, domain=domain)
    _require_cartesian_core_api()
    context = _axis_core_context(context)

    global_r = (jnp.arange(global_shape[0], dtype=jnp.float64) + 0.5) / global_shape[0]
    global_theta = 2.0 * jnp.pi * (jnp.arange(global_shape[1], dtype=jnp.float64) + 0.5) / global_shape[1]
    global_z = 2.0 * jnp.pi * (jnp.arange(global_shape[2], dtype=jnp.float64) + 0.5) / global_shape[2]
    rr, tt, zz = jnp.meshgrid(global_r, global_theta, global_z, indexing="ij")
    global_field = _regular_mode_field(rr, tt, zz, 3) + 0.25 * _regular_mode_field(rr, tt, zz, 8)

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        def kernel(field_owned):
            field_halo_xy = jnp.pad(
                field_owned,
                (
                    (halo_width, halo_width),
                    (halo_width, halo_width),
                    (0, 0),
                ),
                mode="constant",
            )
            # This focused theta-sharding test does not run the halo exchange.
            # Materialize the simple-periodic z halo explicitly so the eta
            # face-gradient target sees the same data as the serial fixture.
            field_halo = jnp.pad(
                field_halo_xy,
                ((0, 0), (0, 0), (halo_width, halo_width)),
                mode="wrap",
            )
            local = _stencil(field_halo, geometry, context)
            return (
                local.face_values.x[:3],
                local.face_values.y[:2, :-1],
                local.face_grad.x[:3],
                local.face_grad.y[:2, :-1],
                local.face_grad.z[:2],
            )

        mapped = shard_map(
            kernel,
            mesh=mesh,
            in_specs=P(None, "y", None),
            out_specs=(
                P(None, "y", None),
                P(None, "y", None),
                P(None, "y", None, None),
                P(None, "y", None, None),
                P(None, "y", None, None),
            ),
            check_rep=False,
        )
        (
            sharded_x,
            sharded_y,
            sharded_grad_x,
            sharded_grad_y,
            sharded_grad_z,
        ) = mapped(global_field)

    serial_geometry, serial_domain, serial_context, (r, theta, z), *_ = polar_fixture(
        shape=global_shape, halo_width=halo_width
    )
    serial_context = _axis_core_context(serial_context)
    serial_field = scalar_field_halo(r, theta, z, lambda r, t, z: _regular_mode_field(r, t, z, 3) + 0.25 * _regular_mode_field(r, t, z, 8))
    serial_local = _stencil(serial_field, serial_geometry, serial_context)
    assembled_x = sharded_x
    assembled_y = sharded_y
    assert jnp.allclose(assembled_x, serial_local.face_values.x[:3], atol=5.0e-10, rtol=5.0e-10)
    assert jnp.allclose(assembled_y, serial_local.face_values.y[:2, :-1], atol=5.0e-10, rtol=5.0e-10)
    assert jnp.allclose(
        sharded_grad_x,
        serial_local.face_grad.x[:3],
        atol=5.0e-10,
        rtol=5.0e-10,
    )
    assert jnp.allclose(
        sharded_grad_y,
        serial_local.face_grad.y[:2, :-1],
        atol=5.0e-10,
        rtol=5.0e-10,
    )
    assert jnp.allclose(
        sharded_grad_z,
        serial_local.face_grad.z[:2],
        atol=5.0e-10,
        rtol=5.0e-10,
    )


def test_centered_curvature_consumes_materialized_face_values():
    """Centered curvature must use the builder-owned face payload."""

    geometry, domain, context, (r, theta, z), *_ = polar_fixture()
    field = scalar_field_halo(r, theta, z, lambda r, t, z: r + 0.2 * t + z)
    local = _stencil(field, geometry, context)
    face_values = replace(
        local.face_values,
        x=jnp.broadcast_to(
            jnp.arange(geometry.layout.face_control_shape(0)[0], dtype=jnp.float64)[:, None, None],
            geometry.layout.face_control_shape(0),
        ),
        y=jnp.zeros(geometry.layout.face_control_shape(1), dtype=jnp.float64),
        z=jnp.zeros(geometry.layout.face_control_shape(2), dtype=jnp.float64),
    )
    local = replace(local, face_values=face_values)
    coefficients = _constant_coefficients(geometry, value=0.0)
    coefficients = replace(
        coefficients,
        x=jnp.ones_like(coefficients.x),
        y=jnp.zeros_like(coefficients.y),
        z=jnp.zeros_like(coefficients.z),
    )
    result = local_curvature_conservative_op(
        local, geometry, coefficients, domain=domain, axis_regular_axes=(False, False, False)
    )
    expected = (
        geometry.cell_bfield.Bmag_owned
        / geometry.cell_metric.J_owned
        / geometry.spacing.dx_owned
    )
    assert jnp.allclose(result, expected, atol=1.0e-12, rtol=1.0e-12)


def test_axis_regular_upwind_curvature_is_explicitly_unsupported():
    geometry, domain, context, (r, theta, z), *_ = polar_fixture()
    field = scalar_field_halo(r, theta, z, lambda r, t, z: _regular_mode_field(r, t, z, 6))
    local = _stencil(field, geometry, context)
    coefficients = _constant_coefficients(geometry, value=0.0)
    with pytest.raises(NotImplementedError, match="upwind"):
        local_curvature_upwind_conservative_op(
            local,
            geometry,
            coefficients,
            domain=domain,
            axis_regular_axes=(True, False, False),
        )


def test_axis_regular_eb_model_accepts_wall_only_upwind_curvature():
    """The wall closure does not require upwind states on the collapsed axis."""

    model = LocalFciDrbEBRhs(
        geometry=None,
        domain=None,
        halo_exchange=None,
        topology_filler=None,
        physical_ghost_filler=None,
        parameters=None,
        curvature_coefficients_owned=None,
        face_projectors=None,
        gmres_config=None,
        face_bc_builder=None,
        axis_regular_axes=(True, False, False),
        curvature_face_coefficients=object(),
        upwind_equilibrium_wall_projectors=object(),
        curvature_scheme="conservative",
        curvature_inflow_closure="upwind-equilibrium",
    )
    assert model.curvature_inflow_closure == "upwind-equilibrium"


def test_axis_regular_wall_upwind_skips_collapsed_lower_radial_face():
    """Only physical wall masks may receive characteristic wall states."""

    import inspect

    implementation = inspect.getsource(
        LocalFciDrbEBRhs._upwind_equilibrium_boundary_face_bcs
    )
    assert "side == 0 and self.axis_regular_axes[axis]" in implementation
    assert "continue" in implementation


def _constant_coefficients(geometry, value=0.0):
    faces = tuple(geometry.layout.face_control_shape(axis) for axis in range(3))
    return LocalCurvatureFaceCoefficients3D(
        layout=geometry.layout,
        x=jnp.full(faces[0], value, dtype=jnp.float64),
        y=jnp.full(faces[1], value, dtype=jnp.float64),
        z=jnp.full(faces[2], value, dtype=jnp.float64),
    )


def test_axis_projectors_are_finite_and_zero_on_collapsed_lower_x_face():
    geometry, domain, *_ = polar_fixture()
    perp = build_local_perp_laplacian_face_projectors(
        geometry, domain, axis_regular_axes=(True, False, False)
    )
    parallel = build_local_parallel_laplacian_face_projectors(
        geometry, domain, axis_regular_axes=(True, False, False)
    )
    for projector in (*perp, *parallel):
        assert jnp.all(jnp.isfinite(projector))
    # The x-family owns the collapsed lower-radial face.  The y/z families'
    # first index is a cell index, so those faces are not axis faces.
    assert jnp.allclose(perp[0][0], 0.0)
    assert jnp.allclose(parallel[0][0], 0.0)


def test_projected_flux_stencil_and_perp_laplacian_are_regular_for_r_squared():
    geometry, domain, context, (r, theta, z), *_ = polar_fixture()
    del theta, z
    field = scalar_field_halo(r, 0.0 * r, 0.0 * r, lambda r, t, z: r**2)
    local = _stencil(field, geometry, context)
    projectors = build_local_perp_laplacian_face_projectors(
        geometry, domain, axis_regular_axes=(True, False, False)
    )
    flux = build_local_projected_laplacian_flux_stencil(
        local,
        geometry,
        domain,
        face_projectors=projectors,
        axis_regular_axes=(True, False, False),
    )
    assert jnp.all(jnp.isfinite(flux.regular_flux.x))
    assert jnp.allclose(flux.regular_flux.x[0], 0.0)
    explicit_stencil = build_local_perp_laplacian_stencil(
        local,
        geometry,
        domain,
        face_projectors=projectors,
        axis_regular_axes=(True, False, False),
    )
    assert jnp.allclose(explicit_stencil.regular_flux.x[0], 0.0)
    result = local_perp_laplacian_conservative_op(
        local, geometry, domain,
        face_projectors=projectors,
        axis_regular_axes=(True, False, False),
    )
    # The implementation's documented convention is +div(P grad).
    assert jnp.all(jnp.isfinite(result))
    assert jnp.max(jnp.abs(result - 4.0)) < 0.45
    assert jnp.max(jnp.abs(result[0] - 4.0)) < 0.45


def test_physical_neumann_closure_cancels_full_normal_gradient_flux():
    geometry, domain, context, (r, theta, z), *_ = polar_fixture()
    field = scalar_field_halo(r, theta, z, lambda r, t, z: jnp.sin(t))
    local = _stencil(field, geometry, context)

    def cross_metric(metric):
        return replace(
            metric,
            g12_halo=jnp.full_like(metric.g12_halo, 0.25),
        )

    geometry = replace(
        geometry,
        face_metric=replace(
            geometry.face_metric,
            x=cross_metric(geometry.face_metric.x),
            y=cross_metric(geometry.face_metric.y),
            z=cross_metric(geometry.face_metric.z),
        ),
    )
    face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    face_bc = replace(
        face_bc,
        kind_x=face_bc.kind_x.at[-1].set(BC_NEUMANN),
        mask_x=face_bc.mask_x.at[-1].set(True),
    )
    full_gradient_projectors = tuple(
        metric.g_contra_owned
        for metric in (
            geometry.face_metric.x,
            geometry.face_metric.y,
            geometry.face_metric.z,
        )
    )

    logical = build_local_projected_laplacian_flux_stencil(
        local,
        geometry,
        domain,
        face_projectors=full_gradient_projectors,
        face_bc=face_bc,
        axis_regular_axes=(True, False, False),
        neumann_normal_scheme="logical",
    )
    physical = build_local_projected_laplacian_flux_stencil(
        local,
        geometry,
        domain,
        face_projectors=full_gradient_projectors,
        face_bc=face_bc,
        axis_regular_axes=(True, False, False),
        neumann_normal_scheme="physical",
    )
    assert jnp.max(jnp.abs(logical.regular_flux.x[-1])) > 1.0e-2
    assert jnp.max(jnp.abs(physical.regular_flux.x[-1])) < 1.0e-12


def test_parallel_laplacian_matches_dzz_and_axis_flux_is_zero():
    geometry, domain, context, (r, theta, z), *_ = polar_fixture()
    field = scalar_field_halo(
        r, theta, z, lambda r, t, z: r**2 * jnp.cos(2.0 * t) * jnp.sin(z)
    )
    local = _stencil(field, geometry, context)
    projectors = build_local_parallel_laplacian_face_projectors(
        geometry, domain, axis_regular_axes=(True, False, False)
    )
    result = local_parallel_laplacian_conservative_op(
        local, geometry, domain,
        face_projectors=projectors,
        axis_regular_axes=(True, False, False),
    )
    assert jnp.all(jnp.isfinite(result))
    assert jnp.max(jnp.abs(result + owned(field, geometry))) < 0.45


def test_parallel_flux_and_compatible_gradient_match_z_derivative_first_ring():
    geometry, domain, context, (r, theta, z), *_ = polar_fixture()
    field = scalar_field_halo(
        r, theta, z, lambda r, t, z: r * jnp.cos(t) * jnp.sin(z)
    )
    local = _stencil(field, geometry, context)
    div_b = local_parallel_div_b_op(
        _stencil(jnp.ones_like(field), geometry, context),
        geometry,
        domain,
        axis_regular_axes=(True, False, False),
    )
    flux_div = local_parallel_flux_div_op(
        local, geometry, domain, axis_regular_axes=(True, False, False)
    )
    compatible = local_grad_parallel_op_conservative(
        local,
        geometry,
        domain,
        div_b=div_b,
        axis_regular_axes=(True, False, False),
    )
    expected = owned(r * jnp.cos(theta) * jnp.cos(z), geometry)
    assert jnp.all(jnp.isfinite(flux_div))
    assert jnp.all(jnp.isfinite(compatible))
    assert jnp.max(jnp.abs(flux_div - expected)) < 0.20
    assert jnp.max(jnp.abs(compatible - expected)) < 0.20
    assert jnp.max(jnp.abs(flux_div[0] - expected[0])) < 0.20
    assert jnp.max(jnp.abs(compatible[0] - expected[0])) < 0.20


def test_parallel_flux_div_and_compatible_gradient_cancel_constants():
    geometry, domain, context, (r, theta, z), *_ = polar_fixture()
    del theta, z
    one = scalar_field_halo(r, 0.0 * r, 0.0 * r, lambda r, t, z: jnp.ones_like(r))
    one_stencil = _stencil(one, geometry, context)
    div_b = local_parallel_div_b_op(
        one_stencil, geometry, domain, axis_regular_axes=(True, False, False)
    )
    div_one_b = local_parallel_flux_div_op(
        one_stencil, geometry, domain, axis_regular_axes=(True, False, False)
    )
    compatible = local_grad_parallel_op_conservative(
        one_stencil,
        geometry,
        domain,
        div_b=div_b,
        axis_regular_axes=(True, False, False),
    )
    assert jnp.all(jnp.isfinite(div_b))
    assert jnp.allclose(div_one_b, div_b, atol=1.0e-12)
    assert jnp.allclose(compatible, 0.0, atol=1.0e-12)
    assert jnp.allclose(div_b[0], 0.0, atol=1.0e-12)


def test_generic_divergence_requires_caller_to_zero_axis_flux():
    geometry, domain, context, (r, theta, z), *_ = polar_fixture()
    del theta, z
    field = scalar_field_halo(r, 0.0 * r, 0.0 * r, lambda r, t, z: r**2)
    local = _stencil(field, geometry, context)
    projectors = build_local_perp_laplacian_face_projectors(
        geometry, domain, axis_regular_axes=(True, False, False)
    )
    flux = build_local_projected_laplacian_flux_stencil(
        local,
        geometry,
        domain,
        face_projectors=projectors,
        axis_regular_axes=(True, False, False),
    )
    assert jnp.allclose(flux.regular_flux.x[0], 0.0)
    result = local_divergence_conservative_op(flux, geometry)
    assert jnp.all(jnp.isfinite(result))


def test_straight_field_curvature_is_zero_for_centered_and_upwind_paths():
    geometry, domain, context, (r, theta, z), *_ = polar_fixture()
    field = scalar_field_halo(r, theta, z, lambda r, t, z: r * jnp.cos(t) + jnp.sin(z))
    local = _stencil(field, geometry, context)
    coefficients = _constant_coefficients(geometry, 0.0)
    centered = local_curvature_conservative_op(
        local, geometry, coefficients, domain=domain,
        axis_regular_axes=(True, False, False)
    )
    assert jnp.all(jnp.isfinite(centered))
    assert jnp.allclose(centered, 0.0)
    with pytest.raises(NotImplementedError, match="upwind"):
        local_curvature_upwind_conservative_op(
            local, geometry, coefficients, domain=domain,
            axis_regular_axes=(True, False, False)
        )


def test_axis_regular_curvature_face_coefficients_are_compatible_with_constants():
    geometry, domain = _build_polar_geometry(shape=(8, 32, 4))
    coefficients = build_local_curvature_face_coefficients(geometry, domain)
    for value in coefficients.axes:
        assert jnp.all(jnp.isfinite(value))

    # The smooth helical field has no radial curl flux at the collapsed face.
    assert jnp.max(jnp.abs(coefficients.x[0])) < 1.0e-13

    def incidence_divergence(qx):
        return (
            (qx[1:] - qx[:-1]) / geometry.spacing.dx_owned
            + (coefficients.y[:, 1:] - coefficients.y[:, :-1])
            / geometry.spacing.dy_owned
            + (coefficients.z[:, :, 1:] - coefficients.z[:, :, :-1])
            / geometry.spacing.dz_owned
        )

    divergence_before = incidence_divergence(coefficients.x)
    divergence_after = incidence_divergence(coefficients.x.at[0].set(0.0))
    assert jnp.max(jnp.abs(divergence_before)) < 1.0e-13
    assert jnp.max(jnp.abs(divergence_after)) < 1.0e-13

    constant = jnp.ones(geometry.halo_shape, dtype=jnp.float64)
    stencil = build_local_conservative_stencil_from_field(
        constant,
        geometry,
        StencilBuilderContext(layout=geometry.layout, domain=domain),
    )
    result = local_curvature_conservative_op(
        stencil,
        geometry,
        coefficients,
        domain=domain,
        axis_regular_axes=(True, False, False),
    )
    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result, 0.0, atol=1.0e-13)
    assert jnp.allclose(result[0], 0.0, atol=1.0e-13)
    with pytest.raises(NotImplementedError, match="upwind"):
        local_curvature_upwind_conservative_op(
            stencil,
            geometry,
            coefficients,
            domain=domain,
            axis_regular_axes=(True, False, False),
        )


def test_curvature_operator_does_not_patch_completed_divergence_free_complex():
    geometry, domain, context, (r, theta, z), *_ = polar_fixture()
    field = scalar_field_halo(r, theta, z, lambda r, t, z: jnp.ones_like(r))
    local = _stencil(field, geometry, context)
    faces = tuple(geometry.layout.face_control_shape(axis) for axis in range(3))
    # A constant radial face coefficient has zero incidence divergence.  It is
    # deliberately not a physically axis-regular coefficient; this probes the
    # operator/coefficient contract.  Post-hoc zeroing of qx[0] would destroy
    # its cancellation and create a first-ring source.  Axis regularity must
    # instead be imposed upstream while constructing the edge curl.
    qx = jnp.ones(faces[0], dtype=jnp.float64)
    coefficients = LocalCurvatureFaceCoefficients3D(
        layout=geometry.layout,
        x=qx,
        y=jnp.zeros(faces[1], dtype=jnp.float64),
        z=jnp.zeros(faces[2], dtype=jnp.float64),
    )
    centered = local_curvature_conservative_op(
        local, geometry, coefficients, domain=domain,
        axis_regular_axes=(True, False, False)
    )
    assert jnp.allclose(centered, 0.0, atol=1.0e-12)
    with pytest.raises(NotImplementedError, match="upwind"):
        local_curvature_upwind_conservative_op(
            local, geometry, coefficients, domain=domain,
            axis_regular_axes=(True, False, False)
        )


def _distributed_axis_face_probes():
    if jax.local_device_count() < 2:
        pytest.skip("requires at least two local devices")

    global_shape = (4, 4, 4)
    shard_counts = (2, 1, 1)
    owned_shape = (2, 4, 4)
    halo_width = 1
    layout = HaloLayout3D(owned_shape, halo_width)
    # Static shard metadata is intentionally identical on every device.  The
    # runtime mesh axis index must be what distinguishes the true axis owner.
    domain = LocalDomain3D(
        shard_spec=ShardSpec3D(
            global_shape=global_shape,
            owned_start=(0, 0, 0),
            owned_stop=owned_shape,
            shard_index=(0, 0, 0),
            shard_counts=shard_counts,
            periodic_axes=(False, True, True),
            axis_regular_axes=(True, False, False),
            halo_width=halo_width,
            side_kind_lower=(
                SIDE_AXIS_REGULAR,
                SIDE_SIMPLE_PERIODIC,
                SIDE_SIMPLE_PERIODIC,
            ),
            side_kind_upper=(
                SIDE_PHYSICAL,
                SIDE_SIMPLE_PERIODIC,
                SIDE_SIMPLE_PERIODIC,
            ),
        ),
        layout=layout,
        mesh_axis_names=("x", "y", "z"),
    )

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        dummy = put_scalar_field_on_mesh(jnp.ones(global_shape), mesh)

        def kernel(_field_owned):
            shard_index = tuple(lax.axis_index(name) for name in ("x", "y", "z"))
            geometry = _build_local_geometry(
                owned_shape,
                halo_width,
                global_shape=global_shape,
                shard_index=shard_index,
            )
            context = StencilBuilderContext(layout=layout, domain=domain)

            constant = jnp.ones(geometry.halo_shape, dtype=jnp.float64)
            curvature_stencil = build_local_conservative_stencil_from_field(
                constant, geometry, context
            )
            face_shapes = tuple(layout.face_control_shape(axis) for axis in range(3))
            # Preserve a divergence-free supplied coefficient complex on every
            # shard. Axis regularization belongs to the coefficient builder,
            # not to a shard-local post-curl flux patch in the operator.
            qx = jnp.ones(face_shapes[0], dtype=jnp.float64)
            coefficients = LocalCurvatureFaceCoefficients3D(
                layout=layout,
                x=qx,
                y=jnp.zeros(face_shapes[1], dtype=jnp.float64),
                z=jnp.zeros(face_shapes[2], dtype=jnp.float64),
            )
            centered = local_curvature_conservative_op(
                curvature_stencil,
                geometry,
                coefficients,
                domain=domain,
                axis_regular_axes=(True, False, False),
            )

            radial = jnp.broadcast_to(
                geometry.grid.x.centers[:, None, None], geometry.halo_shape
            )
            projected_stencil = build_local_conservative_stencil_from_field(
                radial, geometry, context
            )
            supplied_projectors = (
                jnp.zeros(face_shapes[0] + (3, 3), dtype=jnp.float64)
                .at[..., 0, 0]
                .set(1.0),
                jnp.zeros(face_shapes[1] + (3, 3), dtype=jnp.float64),
                jnp.zeros(face_shapes[2] + (3, 3), dtype=jnp.float64),
            )
            completed_flux = build_local_projected_laplacian_flux_stencil(
                projected_stencil,
                geometry,
                domain,
                face_projectors=supplied_projectors,
                axis_regular_axes=(True, False, False),
            )
            local_face_zero_probe = jnp.broadcast_to(
                completed_flux.regular_flux.x[0][None, ...], owned_shape
            )
            return centered, local_face_zero_probe

        mapped = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"),),
            out_specs=(P("x", "y", "z"),) * 2,
            check_rep=False,
        )
        return tuple(jnp.asarray(value) for value in mapped(dummy))


def test_distributed_curvature_preserves_supplied_compatible_complex():
    centered, _projected_flux = _distributed_axis_face_probes()
    assert jnp.allclose(centered, 0.0, atol=1.0e-12)


def test_supplied_projector_axis_flux_uses_runtime_axis_owner_only():
    _centered, projected_flux = _distributed_axis_face_probes()
    shard_split = 2
    assert jnp.allclose(projected_flux[:shard_split], 0.0, atol=1.0e-12)
    assert jnp.min(jnp.abs(projected_flux[shard_split:])) > 1.0e-8
