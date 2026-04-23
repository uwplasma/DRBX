from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jax_drb.native.limiters import (
    minmod3_jax,
    minmod3_numpy,
    minmod3_scalar,
    monotonic_centered_edges_jax,
    monotonic_centered_edges_numpy,
    monotonic_centered_edges_scalar,
    periodic_monotonic_centered_edges_jax,
)


def _reference_minmod3(a, b, c):
    same_sign = (a * b > 0.0) & (a * c > 0.0)
    magnitude = np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c)))
    return np.where(same_sign, np.sign(a) * magnitude, 0.0)


def test_minmod3_jax_matches_numpy_reference() -> None:
    a = np.array([-2.0, -1.0, 1.0, 2.0, 2.0])
    b = np.array([-1.0, 2.0, 0.5, 4.0, -1.0])
    c = np.array([-3.0, 1.0, 3.0, 1.0, 1.0])

    expected = _reference_minmod3(a, b, c)

    np.testing.assert_allclose(np.asarray(minmod3_jax(jnp.asarray(a), jnp.asarray(b), jnp.asarray(c))), expected)
    np.testing.assert_allclose(minmod3_numpy(a, b, c), expected)
    assert minmod3_scalar(-2.0, -1.0, -3.0) == pytest.approx(-1.0)
    assert minmod3_scalar(-2.0, 1.0, -3.0) == pytest.approx(0.0)


def test_monotonic_centered_edges_match_reference_formula() -> None:
    center = np.array([1.0, 2.0, 2.5, 2.6])
    minus = np.array([0.5, 1.0, 2.0, 3.0])
    plus = np.array([2.0, 2.5, 2.4, 2.2])
    slope = _reference_minmod3(2.0 * (plus - center), 0.5 * (plus - minus), 2.0 * (center - minus))
    expected_left = center - 0.5 * slope
    expected_right = center + 0.5 * slope

    jax_left, jax_right = monotonic_centered_edges_jax(
        jnp.asarray(center),
        jnp.asarray(minus),
        jnp.asarray(plus),
    )
    np_left, np_right = monotonic_centered_edges_numpy(center, minus, plus)

    np.testing.assert_allclose(np.asarray(jax_left), expected_left)
    np.testing.assert_allclose(np.asarray(jax_right), expected_right)
    np.testing.assert_allclose(np_left, expected_left)
    np.testing.assert_allclose(np_right, expected_right)
    assert monotonic_centered_edges_scalar(2.0, 1.0, 2.5) == pytest.approx((1.625, 2.375))


def test_periodic_monotonic_centered_edges_selects_requested_axis() -> None:
    field = jnp.arange(12, dtype=jnp.float64).reshape(3, 4)

    left_axis0, right_axis0 = periodic_monotonic_centered_edges_jax(field, axis=0)
    left_axis1, right_axis1 = periodic_monotonic_centered_edges_jax(field, axis=1)

    ref_axis0 = monotonic_centered_edges_jax(field, jnp.roll(field, shift=1, axis=0), jnp.roll(field, shift=-1, axis=0))
    ref_axis1 = monotonic_centered_edges_jax(field, jnp.roll(field, shift=1, axis=1), jnp.roll(field, shift=-1, axis=1))

    np.testing.assert_allclose(np.asarray(left_axis0), np.asarray(ref_axis0[0]))
    np.testing.assert_allclose(np.asarray(right_axis0), np.asarray(ref_axis0[1]))
    np.testing.assert_allclose(np.asarray(left_axis1), np.asarray(ref_axis1[0]))
    np.testing.assert_allclose(np.asarray(right_axis1), np.asarray(ref_axis1[1]))
