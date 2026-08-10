"""Numerical axis-regular tests for the non-FCI direct operator family."""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drbx.geometry import (
    AxisCoreCellGradientReconstruction3D,
    StencilBuilderContext,
    build_axis_core_cell_gradient_reconstruction,
    build_local_cell_gradient_from_field,
    build_local_stencil_from_field,
)
from drbx.native.fci_operators import (
    local_curvature_op,
    local_curvature_op_from_gradient,
    local_grad_parallel_op_direct,
    local_grad_parallel_op_from_gradient,
    local_grad_perp_op_direct,
    local_parallel_laplacian_direct_op,
    local_perp_laplacian_local_op,
    local_poisson_bracket_op,
    local_poisson_bracket_op_from_gradients,
)

from axis_regular_operator_support import analytic_gradient, gradient, owned, polar_fixture, scalar_field_halo


def _stencil(field, geometry, context):
    return build_local_stencil_from_field(field, geometry, context)


def _cell_gradient(field, geometry, context):
    return build_local_cell_gradient_from_field(field, geometry, context)


def _cartesian_monomial(r, theta, z, a, b):
    return (r * jnp.cos(theta))**a * (r * jnp.sin(theta))**b * (1.0 + 0.37 * z)


def _cartesian_monomial_gradient(r, theta, z, a, b):
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    coefficient = 1.0 + 0.37 * z
    value_x = a * x**(a - 1) * y**b * coefficient if a else jnp.zeros_like(r)
    value_y = b * x**a * y**(b - 1) * coefficient if b else jnp.zeros_like(r)
    return jnp.stack(
        (jnp.cos(theta) * value_x + jnp.sin(theta) * value_y,
         r * (-jnp.sin(theta) * value_x + jnp.cos(theta) * value_y),
         0.37 * x**a * y**b),
        axis=-1,
    )


def _ordinary_centered_gradient(field, geometry):
    core = geometry.layout.owned_slices_cell
    steps = (geometry.spacing.dx_halo, geometry.spacing.dy_halo, geometry.spacing.dz_halo)
    result = []
    for axis, step in enumerate(steps):
        plus = list(core)
        minus = list(core)
        plus[axis] = slice(plus[axis].start + 1, plus[axis].stop + 1)
        minus[axis] = slice(minus[axis].start - 1, minus[axis].stop - 1)
        numerator = field[tuple(plus)] - field[tuple(minus)]
        # The fixture has uniform spacing; retaining the explicit spacing makes
        # this reference test the ordinary centered-difference contract.
        step_core = step[core]
        result.append(numerator / (2.0 * step_core))
    return jnp.stack(result, axis=-1)


def _regular_mode(r, theta, m):
    return r**m * jnp.cos(m * theta)


def _regular_mode_gradient(r, theta, m):
    if m == 0:
        return jnp.zeros(r.shape + (3,), dtype=r.dtype)
    return jnp.stack(
        (m * r**(m - 1) * jnp.cos(m * theta),
         -m * r**m * jnp.sin(m * theta),
         jnp.zeros_like(r)),
        axis=-1,
    )


def test_scalar_axis_half_turn_and_parallel_gradient():
    geometry, domain, context, (r, theta, z), exchange, scalar, _, _ = polar_fixture()
    field = scalar_field_halo(r, theta, z, lambda r, t, z: r * jnp.cos(t) * jnp.sin(z))
    closed = scalar(field, domain)
    # r cos(theta) is x and is odd under the polar half-turn.
    assert jnp.allclose(closed[0, 1:-1, 1:-1], -closed[1, 1:-1, 1:-1], atol=1e-12)
    result = local_grad_parallel_op_direct(_stencil(closed, geometry, context), geometry)
    expected = owned(r * jnp.cos(theta) * jnp.cos(z), geometry)
    assert jnp.max(jnp.abs(result - expected)) < 3.0e-2


def test_parallel_laplacian_is_regular_at_axis_adjacent_ring():
    geometry, domain, context, (r, theta, z), exchange, scalar, _, _ = polar_fixture()
    field = scalar_field_halo(r, theta, z, lambda r, t, z: r**2 * jnp.cos(2*t) * jnp.sin(z))
    closed = scalar(field, domain)
    result = local_parallel_laplacian_direct_op(
        closed, geometry, domain, context=context,
        halo_exchange=exchange, topology_filler=scalar,
    )
    expected = -owned(r**2 * jnp.cos(2*theta) * jnp.sin(z), geometry)
    assert jnp.max(jnp.abs(result - expected)) < 5.0e-2


def test_perpendicular_gradient_has_polar_regular_cartesian_limits():
    geometry, domain, context, (r, theta, z), *_ = polar_fixture()
    field = scalar_field_halo(r, theta, z, lambda r, t, z: r**2 * jnp.cos(2*t))
    result = local_grad_perp_op_direct(_stencil(field, geometry, context), geometry)
    expected = jnp.stack(
        (2.0*r*jnp.cos(2.0*theta), -2.0*jnp.sin(2.0*theta), jnp.zeros_like(r)),
        axis=-1,
    )[geometry.layout.owned_slices_cell]
    assert result.shape == geometry.owned_shape + (3,)
    # The angular derivative is second-order centered; this tolerance covers
    # its O(dtheta^2) error while checking both the first ring and interior.
    assert jnp.max(jnp.abs(result - expected)) < 0.20
    assert jnp.max(jnp.abs(result[0] - expected[0])) < 0.20


def test_perpendicular_laplacian_requires_flux_density_axis_parity():
    geometry, domain, context, (r, theta, z), exchange, scalar, ordinary_vector, flux_density = polar_fixture()
    del ordinary_vector
    field = scalar_field_halo(r, theta, z, lambda r, t, z: r**2)
    # The intermediate F^i = J P^{ij} partial_j f is a contravariant vector
    # density. A scalar polar rule cannot close this three-component field.
    with pytest.raises(ValueError, match="scalar|shape|cell_halo"):
        local_perp_laplacian_local_op(
            scalar(field, domain), geometry, domain, context=context,
            intermediate_stencil_builder=build_local_stencil_from_field,
            halo_exchange=exchange, topology_filler=scalar,
        )

    result = local_perp_laplacian_local_op(
        scalar(field, domain), geometry, domain, context=context,
        intermediate_stencil_builder=build_local_stencil_from_field,
        halo_exchange=exchange, topology_filler=flux_density,
    )
    assert result.shape == geometry.owned_shape
    # The upper physical ghost is intentionally unclosed in this fixture, so
    # only the outermost ring is excluded. Axis through penultimate rings must
    # reproduce nabla_perp^2(r^2) = 4 to roundoff.
    assert jnp.allclose(result[:-1], 4.0, atol=1.0e-12, rtol=1.0e-12)
    assert jnp.allclose(result[0], 4.0, atol=1.0e-12, rtol=1.0e-12)


def _expected_vector_axis_ghost(vector_field, domain, diagonal):
    source = vector_field[domain.layout.owned_slices_cell][0]
    return jnp.einsum(
        "ij,klj->kli",
        jnp.diag(jnp.asarray(diagonal)),
        jnp.roll(source, shift=-(domain.shard_spec.global_shape[1] // 2), axis=0),
    )


def test_ordinary_vector_axis_rule_applies_exact_component_parity():
    geometry, domain, context, (r, theta, z), exchange, scalar, ordinary_vector, flux_density = polar_fixture()
    del geometry, context, exchange, scalar, flux_density
    vector_field = jnp.stack((r*jnp.cos(theta), r*jnp.sin(theta), jnp.sin(z)), axis=-1)
    closed = ordinary_vector(vector_field, domain)
    expected = _expected_vector_axis_ghost(vector_field, domain, (-1.0, 1.0, 1.0))
    assert jnp.allclose(closed[0, 1:-1, 1:-1], expected, atol=1.0e-12)


def test_flux_density_axis_rule_applies_exact_component_parity():
    geometry, domain, context, (r, theta, z), exchange, scalar, ordinary_vector, flux_density = polar_fixture()
    del geometry, context, exchange, scalar, ordinary_vector
    vector_density = jnp.stack((r**2, r*jnp.cos(theta), r*jnp.sin(theta)), axis=-1)
    closed = flux_density(vector_density, domain)
    expected = _expected_vector_axis_ghost(vector_density, domain, (1.0, -1.0, -1.0))
    assert jnp.allclose(closed[0, 1:-1, 1:-1], expected, atol=1.0e-12)


def test_parallel_gradient_from_manufactured_gradient():
    geometry, domain, context, (r, theta, z), *_ = polar_fixture()
    coordinate_gradient = jnp.stack(
        (jnp.cos(theta)*jnp.sin(z),
         -r*jnp.sin(theta)*jnp.sin(z),
         r*jnp.cos(theta)*jnp.cos(z)),
        axis=-1,
    )
    reconstructed = gradient(coordinate_gradient[geometry.layout.owned_slices_cell], geometry)
    result = local_grad_parallel_op_from_gradient(reconstructed, geometry)
    expected = r[geometry.layout.owned_slices_cell]*jnp.cos(theta[geometry.layout.owned_slices_cell])*jnp.cos(z[geometry.layout.owned_slices_cell])
    assert result.shape == geometry.owned_shape
    assert jnp.max(jnp.abs(result - expected)) < 1.0e-12
    assert jnp.max(jnp.abs(result[0] - expected[0])) < 1.0e-12


def test_poisson_bracket_and_gradient_form_are_one_on_first_ring_and_interior():
    geometry, domain, context, (r, theta, z), *_ = polar_fixture()
    f = scalar_field_halo(r, theta, z, lambda r, t, z: r*jnp.cos(t))
    g = scalar_field_halo(r, theta, z, lambda r, t, z: r*jnp.sin(t))
    bracket = local_poisson_bracket_op(_stencil(f, geometry, context), _stencil(g, geometry, context), geometry)
    expected = jnp.ones(geometry.owned_shape)
    assert bracket.shape == geometry.owned_shape
    assert jnp.all(jnp.isfinite(bracket))
    assert jnp.max(jnp.abs(bracket - expected)) < 0.15
    assert jnp.max(jnp.abs(bracket[0] - expected[0])) < 0.15
    fg = gradient(owned(analytic_gradient(r, theta, z, "x"), geometry), geometry)
    gg = gradient(owned(analytic_gradient(r, theta, z, "y"), geometry), geometry)
    bracket_grad = local_poisson_bracket_op_from_gradients(fg, gg, geometry)
    assert jnp.allclose(bracket_grad, 1.0, atol=1e-12)


def test_curvature_direct_zero_field_and_gradient_form():
    geometry, domain, context, (r, theta, z), *_ = polar_fixture()
    field = scalar_field_halo(r, theta, z, lambda r, t, z: r**2)
    zero = jnp.zeros(geometry.owned_shape + (3,))
    direct = local_curvature_op(_stencil(field, geometry, context), geometry, curvature_coefficients=zero)
    assert jnp.allclose(direct, 0.0)
    grad = gradient(jnp.zeros(geometry.owned_shape + (3,)), geometry)
    from_grad = local_curvature_op_from_gradient(grad, geometry, curvature_coefficients=zero)
    assert jnp.allclose(from_grad, 0.0)


def test_cell_gradient_axis_core_payload_installs_with_metadata_and_round_trips():
    geometry, domain, context, *_ = polar_fixture(shape=(8, 32, 4))
    assert isinstance(context.axis_core_cell_gradient_reconstruction, AxisCoreCellGradientReconstruction3D)
    payload = context.axis_core_cell_gradient_reconstruction
    assert payload.layout == geometry.layout
    assert payload.global_shape == tuple(domain.shard_spec.global_shape)
    assert payload.polynomial_degree == 3
    assert payload.observation_ring_count == 6
    assert payload.target_ring_count == 3
    explicit = build_axis_core_cell_gradient_reconstruction(
        geometry.layout, domain, polynomial_degree=3,
        observation_ring_count=6, target_ring_count=3,
    )
    assert explicit.polynomial_degree == 3
    if hasattr(payload, "normalized_design_condition_number"):
        assert jnp.isfinite(payload.normalized_design_condition_number)

    rebuilt = StencilBuilderContext(layout=geometry.layout, domain=domain)
    assert isinstance(rebuilt.axis_core_cell_gradient_reconstruction, AxisCoreCellGradientReconstruction3D)
    leaves, treedef = jax.tree_util.tree_flatten(context)
    copied = jax.tree_util.tree_unflatten(treedef, leaves)
    copied_payload = copied.axis_core_cell_gradient_reconstruction
    assert copied_payload.observation_ring_count == 6
    assert copied_payload.target_ring_count == 3
    assert jnp.array_equal(copied_payload.observation_to_coefficient_weights, payload.observation_to_coefficient_weights)


def test_cell_gradient_reconstruction_mask_is_owned_and_only_target_rings():
    geometry, _domain, context, (r, theta, z), *_ = polar_fixture(shape=(8, 32, 4))
    field = scalar_field_halo(r, theta, z, lambda r, t, z: (r * jnp.cos(t))**2)
    result = _cell_gradient(field, geometry, context)
    target = context.axis_core_cell_gradient_reconstruction.target_ring_count
    expected = jnp.zeros(geometry.owned_shape, dtype=bool).at[:target].set(True)
    assert result.reconstruction_mask.shape == geometry.owned_shape
    assert jnp.array_equal(result.reconstruction_mask, expected)


@pytest.mark.parametrize(
    "a,b", [(a, b) for total in range(4) for a in range(total + 1) for b in (total - a,)]
)
def test_cell_gradient_reproduces_cartesian_monomials_through_cubic_degree(a, b):
    geometry, _domain, context, (r, theta, z), *_ = polar_fixture(shape=(8, 32, 4))
    field = scalar_field_halo(r, theta, z, lambda r, t, z: _cartesian_monomial(r, t, z, a, b))
    actual = _cell_gradient(field, geometry, context)
    expected = _cartesian_monomial_gradient(r, theta, z, a, b)[geometry.layout.owned_slices_cell]
    rings = context.axis_core_cell_gradient_reconstruction.target_ring_count
    assert jnp.allclose(actual.gradient[:rings], expected[:rings], atol=2.0e-11, rtol=2.0e-11)


def test_cell_gradient_patches_all_components_and_eta_is_projected():
    geometry, _domain, context, (r, theta, z), *_ = polar_fixture(shape=(8, 32, 4))
    field = scalar_field_halo(r, theta, z, lambda r, t, z: (1.0 + 0.37 * z) * (r * jnp.cos(t))**2)
    actual = _cell_gradient(field, geometry, context).gradient
    expected = _cartesian_monomial_gradient(r, theta, z, 2, 0)[geometry.layout.owned_slices_cell]
    rings = context.axis_core_cell_gradient_reconstruction.target_ring_count
    assert jnp.allclose(actual[:rings], expected[:rings], atol=2.0e-11, rtol=2.0e-11)
    assert jnp.max(jnp.abs(actual[:rings, ..., 2])) > 1.0e-5


def test_cell_gradient_is_ordinary_centered_difference_outside_target_rings():
    geometry, _domain, context, (r, theta, z), *_ = polar_fixture(shape=(8, 32, 4))
    field = scalar_field_halo(r, theta, z, lambda r, t, z: jnp.sin(1.7 * r + 0.2 * t + z))
    actual = _cell_gradient(field, geometry, context).gradient
    expected = _ordinary_centered_gradient(field, geometry)
    rings = context.axis_core_cell_gradient_reconstruction.target_ring_count
    assert jnp.allclose(actual[rings:], expected[rings:], atol=1.0e-13, rtol=1.0e-13)


def test_cell_gradient_constants_and_poisson_bracket_are_preserved_and_antisymmetric():
    geometry, _domain, context, (r, theta, z), *_ = polar_fixture(shape=(8, 32, 4))
    constant = scalar_field_halo(r, theta, z, lambda r, t, z: jnp.ones_like(r))
    zero = _cell_gradient(constant, geometry, context)
    assert jnp.allclose(zero.gradient, 0.0, atol=1.0e-13, rtol=1.0e-13)

    f = scalar_field_halo(r, theta, z, lambda r, t, z: _cartesian_monomial(r, t, z, 2, 0))
    g = scalar_field_halo(r, theta, z, lambda r, t, z: _cartesian_monomial(r, t, z, 1, 1))
    fg = _cell_gradient(f, geometry, context)
    gg = _cell_gradient(g, geometry, context)
    bracket_fg = local_poisson_bracket_op_from_gradients(fg, gg, geometry)
    bracket_gf = local_poisson_bracket_op_from_gradients(gg, fg, geometry)
    assert jnp.allclose(bracket_fg, -bracket_gf, atol=2.0e-13, rtol=2.0e-13)


def test_cell_gradient_mode_order_sweep_uses_actual_degree_and_filters_higher_modes():
    geometry, domain, _context, (r, theta, z), *_ = polar_fixture(shape=(8, 32, 4))
    for degree in (2, 3, 4, 6):
        payload = build_axis_core_cell_gradient_reconstruction(
            geometry.layout, domain, polynomial_degree=degree,
            observation_ring_count=6, target_ring_count=3,
        )
        context = StencilBuilderContext(
            layout=geometry.layout, domain=domain,
            axis_core_cell_gradient_reconstruction=payload,
        )
        if hasattr(payload, "normalized_design_condition_number"):
            assert jnp.isfinite(payload.normalized_design_condition_number)
        for m in range(8):
            field = scalar_field_halo(r, theta, z, lambda r, t, z: _regular_mode(r, t, m))
            actual = _cell_gradient(field, geometry, context).gradient
            if m <= payload.polynomial_degree:
                expected = _regular_mode_gradient(r, theta, m)[geometry.layout.owned_slices_cell]
                assert jnp.allclose(actual[:payload.target_ring_count], expected[:payload.target_ring_count], atol=2.0e-11, rtol=2.0e-11)
            else:
                assert jnp.max(jnp.abs(actual[:payload.target_ring_count])) < 2.0e-10


def test_cell_gradient_does_not_regularize_an_order_one_cosine_with_o_one_radial_amplitude():
    geometry, _domain, context, (r, theta, z), *_ = polar_fixture(shape=(8, 32, 4))
    field = scalar_field_halo(r, theta, z, lambda r, t, z: jnp.cos(t))
    actual = _cell_gradient(field, geometry, context).gradient
    rings = context.axis_core_cell_gradient_reconstruction.target_ring_count
    smooth_axis_mode = _regular_mode_gradient(r, theta, 1)[geometry.layout.owned_slices_cell]
    assert jnp.max(jnp.abs(actual[:rings] - smooth_axis_mode[:rings])) > 0.1


def test_cell_gradient_without_domain_or_payload_is_ordinary_and_unmasked():
    geometry, _domain, _axis_context, (r, theta, z), *_ = polar_fixture(shape=(8, 32, 4))
    context = StencilBuilderContext(layout=geometry.layout)
    field = scalar_field_halo(r, theta, z, lambda r, t, z: jnp.sin(1.7 * r + 0.2 * t + z))
    actual = _cell_gradient(field, geometry, context)
    expected = _ordinary_centered_gradient(field, geometry)
    assert jnp.allclose(actual.gradient, expected, atol=1.0e-13, rtol=1.0e-13)
    assert actual.reconstruction_mask.shape == geometry.owned_shape
    assert not bool(jnp.any(actual.reconstruction_mask))
