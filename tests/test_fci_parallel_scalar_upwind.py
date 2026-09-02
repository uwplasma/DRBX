"""Focused tests for the scalar mapped parallel-vorticity upwind kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.native.fci_parallel_production_flux import (
    parallel_vorticity_upwind_residual,
)


def test_positive_speed_uses_only_backward_upstream_leg():
    result = parallel_vorticity_upwind_residual(
        omega_center=3.0,
        omega_minus=1.0,
        omega_plus=100.0,
        Vi=2.0,
        dx_minus=2.0,
        dx_plus=4.0,
    )
    # -(2 * (3 - 1) / 2)
    np.testing.assert_array_equal(result, -2.0)


def test_negative_speed_uses_only_forward_upstream_leg():
    result = parallel_vorticity_upwind_residual(
        omega_center=3.0,
        omega_minus=-100.0,
        omega_plus=1.0,
        Vi=-2.0,
        dx_minus=2.0,
        dx_plus=4.0,
    )
    # -((-2) * (1 - 3) / 4)
    np.testing.assert_array_equal(result, -1.0)


def test_zero_speed_has_zero_residual_independent_of_endpoint_values():
    result = parallel_vorticity_upwind_residual(
        omega_center=3.0,
        omega_minus=-100.0,
        omega_plus=100.0,
        Vi=0.0,
        dx_minus=2.0,
        dx_plus=4.0,
    )
    np.testing.assert_array_equal(result, 0.0)


def test_unequal_leg_lengths_are_used_independently():
    result = parallel_vorticity_upwind_residual(
        omega_center=4.0,
        omega_minus=1.0,
        omega_plus=10.0,
        Vi=2.0,
        dx_minus=3.0,
        dx_plus=7.0,
    )
    np.testing.assert_allclose(result, -2.0)

    result = parallel_vorticity_upwind_residual(
        omega_center=4.0,
        omega_minus=1.0,
        omega_plus=10.0,
        Vi=-2.0,
        dx_minus=3.0,
        dx_plus=7.0,
    )
    np.testing.assert_allclose(result, 12.0 / 7.0)


def test_constant_omega_is_exactly_preserved_for_all_flow_signs():
    result = parallel_vorticity_upwind_residual(
        omega_center=jnp.full((3, 2), 7.25),
        omega_minus=jnp.full((3, 2), 7.25),
        omega_plus=jnp.full((3, 2), 7.25),
        Vi=jnp.asarray([[2.0, -2.0], [0.0, 4.0], [-3.0, 0.0]]),
        dx_minus=jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        dx_plus=jnp.asarray([[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]),
    )
    np.testing.assert_array_equal(np.asarray(result), 0.0)


def test_outflow_endpoint_trace_is_ignored_but_inflow_trace_is_used():
    # Positive flow exits through the forward endpoint: changing its wall
    # trace cannot affect the backward upwind residual.
    positive_a = parallel_vorticity_upwind_residual(2.0, 1.0, 3.0, 4.0, 2.0, 1.0)
    positive_b = parallel_vorticity_upwind_residual(2.0, 1.0, 999.0, 4.0, 2.0, 1.0)
    np.testing.assert_array_equal(positive_a, positive_b)

    # Negative flow exits through the backward endpoint: changing its wall
    # trace cannot affect the forward upwind residual.
    negative_a = parallel_vorticity_upwind_residual(2.0, 3.0, 1.0, -4.0, 2.0, 1.0)
    negative_b = parallel_vorticity_upwind_residual(2.0, -999.0, 1.0, -4.0, 2.0, 1.0)
    np.testing.assert_array_equal(negative_a, negative_b)

    # Conversely, the physically incoming trace is selected for each sign.
    positive_in_a = parallel_vorticity_upwind_residual(2.0, 1.0, 3.0, 4.0, 2.0, 1.0)
    positive_in_b = parallel_vorticity_upwind_residual(2.0, 9.0, 3.0, 4.0, 2.0, 1.0)
    assert positive_in_a != positive_in_b
    negative_in_a = parallel_vorticity_upwind_residual(2.0, 3.0, 1.0, -4.0, 2.0, 1.0)
    negative_in_b = parallel_vorticity_upwind_residual(2.0, 3.0, 9.0, -4.0, 2.0, 1.0)
    assert negative_in_a != negative_in_b


def test_inputs_broadcast_to_float64_and_jit_without_special_arguments():
    center = jnp.asarray([[2.0], [4.0]], dtype=jnp.float32)
    minus = jnp.asarray([[1.0, 0.0, -1.0]], dtype=jnp.float32)
    plus = jnp.asarray(8.0, dtype=jnp.float32)
    speed = jnp.asarray([[2.0, -2.0, 0.0]], dtype=jnp.float32)
    dx_minus = jnp.asarray(2.0, dtype=jnp.float32)
    dx_plus = jnp.asarray([[1.0], [3.0]], dtype=jnp.float32)
    result = jax.jit(parallel_vorticity_upwind_residual)(
        center, minus, plus, speed, dx_minus, dx_plus
    )
    assert result.shape == (2, 3)
    assert result.dtype == jnp.float64
    assert bool(jnp.all(jnp.isfinite(result)))


def test_nonpositive_leg_lengths_are_replaced_by_safe_positive_floor():
    result = parallel_vorticity_upwind_residual(
        omega_center=1.0,
        omega_minus=0.0,
        omega_plus=0.0,
        Vi=1.0,
        dx_minus=0.0,
        dx_plus=-2.0,
    )
    np.testing.assert_allclose(result, -1.0e30, rtol=2.0e-15, atol=0.0)
    assert bool(jnp.isfinite(result))


def test_incompatible_shapes_raise_during_broadcast_validation():
    with pytest.raises(ValueError):
        parallel_vorticity_upwind_residual(
            omega_center=jnp.zeros((2,)),
            omega_minus=jnp.zeros((3,)),
            omega_plus=0.0,
            Vi=1.0,
            dx_minus=1.0,
            dx_plus=1.0,
        )


@pytest.mark.parametrize("speed", (1.25, -1.25))
def test_periodic_smooth_wave_has_first_order_convergence(speed):
    """Nearest-leg upwinding converges at first order for either flow sign."""

    errors = []
    for n in (32, 64, 128):
        dx = 2.0 * np.pi / n
        x = jnp.arange(n, dtype=jnp.float64) * dx
        omega = jnp.sin(x)
        omega_minus = jnp.roll(omega, 1)
        omega_plus = jnp.roll(omega, -1)
        numerical = parallel_vorticity_upwind_residual(
            omega, omega_minus, omega_plus, speed, dx, dx
        )
        exact = -speed * jnp.cos(x)
        errors.append(float(jnp.max(jnp.abs(numerical - exact))))

    coarse_to_medium = errors[0] / errors[1]
    medium_to_fine = errors[1] / errors[2]
    assert coarse_to_medium > 1.8
    assert medium_to_fine > 1.8
