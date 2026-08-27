"""Tests for the reusable matrix-free weighted support adjoint."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.native.fci_support_pair import build_weighted_negative_adjoint


def _gradient(matrix: jnp.ndarray):
    return lambda values: matrix @ values


def test_random_vectors_satisfy_weighted_negative_adjoint_closure():
    matrix = jnp.array(
        [[-1.0, 0.2, 0.8], [0.4, -0.9, 0.5], [0.3, 0.1, -0.4], [0.7, -0.6, -0.1]]
    )
    primal_mass = jnp.array([2.0, 3.5, 1.25])
    dual_mass = jnp.array([0.8, 1.1, 2.2, 0.6])
    divergence = build_weighted_negative_adjoint(
        _gradient(matrix), primal_mass, dual_mass
    )
    primal = jnp.array([1.3, -0.6, 2.1])
    dual = jnp.array([-1.2, 0.4, 2.5, -0.8])

    lhs = jnp.vdot(primal_mass * primal, divergence(dual))
    rhs = -jnp.vdot(dual_mass * _gradient(matrix)(primal), dual)
    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), atol=2e-12)


def test_constant_exact_gradient_produces_conservative_divergence():
    # Every row sums to zero, so this is exact on a primal constant.
    matrix = jnp.array(
        [[-1.0, 1.0, 0.0], [0.25, -0.75, 0.5], [0.5, 0.0, -0.5], [1.0, -0.2, -0.8]]
    )
    primal_mass = jnp.array([1.5, 2.0, 0.7])
    dual_mass = jnp.array([0.8, 1.1, 1.3, 0.9])
    gradient = _gradient(matrix)
    divergence = build_weighted_negative_adjoint(gradient, primal_mass, dual_mass)

    np.testing.assert_allclose(np.asarray(gradient(jnp.ones(3))), 0.0, atol=2e-12)
    dual = jnp.array([0.7, -1.1, 0.4, 2.0])
    np.testing.assert_allclose(
        np.asarray(jnp.sum(primal_mass * divergence(dual))), 0.0, atol=2e-12
    )


def test_matches_explicit_weighted_matrix_formula():
    matrix = jnp.array([[2.0, -1.0], [0.5, 3.0], [-4.0, 1.5]])
    primal_mass = jnp.array([2.0, 5.0])
    dual_mass = jnp.array([1.5, 0.75, 3.0])
    divergence = build_weighted_negative_adjoint(
        _gradient(matrix), primal_mass, dual_mass
    )
    dual = jnp.array([1.2, -0.4, 0.8])

    expected = -jnp.diag(1.0 / primal_mass) @ matrix.T @ jnp.diag(dual_mass) @ dual
    np.testing.assert_allclose(np.asarray(divergence(dual)), np.asarray(expected), atol=2e-12)


def test_inactive_entries_are_excluded_and_zeroed():
    matrix = jnp.array(
        [[2.0, -1.0, 0.5], [1.0, 3.0, -2.0], [-4.0, 0.2, 2.0], [0.5, -0.3, 1.0]]
    )
    primal_mass = jnp.array([2.0, 0.0, 4.0])
    dual_mass = jnp.array([1.0, -2.0, 3.0, 0.0])
    primal_active = jnp.array([True, False, True])
    dual_active = jnp.array([True, False, True, False])
    divergence = build_weighted_negative_adjoint(
        _gradient(matrix),
        primal_mass,
        dual_mass,
        primal_active=primal_active,
        dual_active=dual_active,
    )
    dual = jnp.array([1.2, -99.0, 0.4, 13.0])

    weighted_dual = jnp.array([1.0 * 1.2, 0.0, 3.0 * 0.4, 0.0])
    expected = -matrix.T @ weighted_dual / jnp.array([2.0, 1.0, 4.0])
    expected = jnp.where(primal_active, expected, 0.0)
    np.testing.assert_allclose(np.asarray(divergence(dual)), np.asarray(expected), atol=2e-12)
    assert float(divergence(dual)[1]) == 0.0


def test_jit_and_leading_batch_lanes():
    matrix = jnp.array([[1.0, -2.0], [0.3, 0.7], [-0.5, 1.5]])
    divergence = build_weighted_negative_adjoint(
        _gradient(matrix), jnp.array([2.0, 4.0]), jnp.array([1.0, 3.0, 0.5])
    )
    dual_batch = jnp.array([[1.0, -0.2, 0.5], [-2.0, 0.3, 1.1]])

    actual = jax.jit(divergence)(dual_batch)
    expected = jnp.stack([divergence(dual_batch[index]) for index in range(2)])
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-12)


def test_builder_accepts_traced_geometry_masses():
    matrix = jnp.array([[1.0, -2.0], [0.3, 0.7], [-0.5, 1.5]])
    gradient = _gradient(matrix)

    @jax.jit
    def build_and_apply(primal_mass, dual_mass, dual_values):
        divergence = build_weighted_negative_adjoint(
            gradient, primal_mass, dual_mass
        )
        return divergence(dual_values)

    primal_mass = jnp.array([2.0, 4.0])
    dual_mass = jnp.array([1.0, 3.0, 0.5])
    dual_values = jnp.array([1.0, -0.2, 0.5])
    actual = build_and_apply(primal_mass, dual_mass, dual_values)
    expected = build_weighted_negative_adjoint(
        gradient, primal_mass, dual_mass
    )(dual_values)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-12)


def test_builder_rejects_bad_mass_shapes_dtypes_and_active_masses():
    gradient = _gradient(jnp.eye(2))
    with pytest.raises(TypeError, match="floating-point"):
        build_weighted_negative_adjoint(gradient, jnp.array([1, 2]), jnp.ones(2))
    with pytest.raises(ValueError, match="dual_mass"):
        build_weighted_negative_adjoint(gradient, jnp.ones(2), jnp.ones(3))
    with pytest.raises(ValueError, match="positive on active"):
        build_weighted_negative_adjoint(
            gradient, jnp.array([1.0, 0.0]), jnp.ones(2)
        )
