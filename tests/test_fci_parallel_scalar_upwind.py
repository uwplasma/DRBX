"""Focused tests for the scalar mapped parallel-vorticity upwind kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.native.fci_parallel_production_flux import (
    parallel_vorticity_second_order_upwind_residual,
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


@pytest.mark.parametrize("speed", (1.25, -1.25))
def test_periodic_smooth_wave_has_second_order_convergence(speed):
    """Two same-direction mapped legs recover the selected order-two action."""

    errors = []
    for n in (32, 64, 128):
        dx = 2.0 * np.pi / n
        x = jnp.arange(n, dtype=jnp.float64) * dx
        omega = jnp.sin(x)
        numerical = parallel_vorticity_second_order_upwind_residual(
            omega,
            jnp.roll(omega, 1),
            jnp.roll(omega, -1),
            jnp.roll(omega, 2),
            jnp.roll(omega, -2),
            speed,
            dx,
            dx,
            dx,
            dx,
        )
        exact = -speed * jnp.cos(x)
        errors.append(float(jnp.max(jnp.abs(numerical - exact))))

    coarse_to_medium = errors[0] / errors[1]
    medium_to_fine = errors[1] / errors[2]
    assert coarse_to_medium > 3.8
    assert medium_to_fine > 3.8


@pytest.mark.parametrize("speed", (1.25, -1.25))
def test_nonuniform_second_order_upwind_is_quadratic_exact(speed):
    """Consecutive unequal mapped legs reproduce the center derivative."""

    dm, dm2 = 0.3, 0.55
    dp, dp2 = 0.4, 0.2

    def field(x):
        return 2.0 + 1.7 * x - 0.6 * x * x

    actual = parallel_vorticity_second_order_upwind_residual(
        field(0.0),
        field(-dm),
        field(dp),
        field(-(dm + dm2)),
        field(dp + dp2),
        speed,
        dm,
        dp,
        dm2,
        dp2,
    )
    np.testing.assert_allclose(actual, -speed * 1.7, rtol=0.0, atol=2.0e-14)


def test_second_order_invalid_row_falls_back_to_complete_first_order_action():
    inputs = dict(
        omega_center=jnp.asarray((2.0, 3.0)),
        omega_minus=jnp.asarray((1.0, 1.5)),
        omega_plus=jnp.asarray((4.0, 5.0)),
        Vi=jnp.asarray((1.25, -0.75)),
        dx_minus=jnp.asarray((0.3, 0.4)),
        dx_plus=jnp.asarray((0.5, 0.6)),
    )
    expected = parallel_vorticity_upwind_residual(**inputs)
    actual = parallel_vorticity_second_order_upwind_residual(
        **inputs,
        omega_minus2=jnp.asarray((-7.0, -9.0)),
        omega_plus2=jnp.asarray((11.0, 13.0)),
        dx_minus2=jnp.asarray((0.7, 0.8)),
        dx_plus2=jnp.asarray((0.9, 1.0)),
        backward_second_valid=False,
        forward_second_valid=False,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("speed", (1.25, -0.75))
def test_invalid_downwind_second_endpoint_does_not_poison_selected_direction(
    speed,
):
    inputs = dict(
        omega_center=2.0,
        omega_minus=1.3,
        omega_plus=2.9,
        omega_minus2=0.2 if speed > 0.0 else jnp.nan,
        omega_plus2=jnp.nan if speed > 0.0 else 4.1,
        Vi=speed,
        dx_minus=0.3,
        dx_plus=0.5,
        dx_minus2=0.7,
        dx_plus2=0.9,
        backward_second_valid=speed > 0.0,
        forward_second_valid=speed < 0.0,
    )
    actual = parallel_vorticity_second_order_upwind_residual(**inputs)
    assert bool(jnp.isfinite(actual))

    if speed > 0.0:
        expected = parallel_vorticity_second_order_upwind_residual(
            **{**inputs, "omega_plus2": 123.0}
        )
    else:
        expected = parallel_vorticity_second_order_upwind_residual(
            **{**inputs, "omega_minus2": -123.0}
        )
    np.testing.assert_array_equal(actual, expected)


def test_second_order_constant_is_exact_and_jittable():
    constant = jnp.full((3, 2), 4.25)
    speed = jnp.asarray([[2.0, -2.0], [0.0, 4.0], [-3.0, 0.0]])
    result = jax.jit(parallel_vorticity_second_order_upwind_residual)(
        constant,
        constant,
        constant,
        constant,
        constant,
        speed,
        0.3,
        0.4,
        0.5,
        0.6,
    )
    np.testing.assert_array_equal(np.asarray(result), 0.0)
