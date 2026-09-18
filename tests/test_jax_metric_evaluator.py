"""Focused parity checks for the JAX toroidal metric runtime state."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.geometry.MetricEvaluator import MetricEvaluator
from drbx.geometry.jax_metric_evaluator import JaxMetricEvaluator


jax.config.update("jax_enable_x64", True)


def _evaluator():
    period = 2.0 * np.pi
    u = np.linspace(0.0, 1.0, 9)
    theta = 2.0 * np.pi * np.arange(16) / 16.0
    eta = period * np.arange(20) / 20.0
    ug, tg, eg = np.meshgrid(u, theta, eta, indexing="ij")
    angle = tg + 0.15 * np.sin(eg)
    radius = 3.0 + 0.7 * ug * (np.cos(angle) + 0.08 * np.cos(2.0 * angle))
    positions = np.stack(
        (radius * np.cos(eg), radius * np.sin(eg), -0.7 * ug * np.sin(angle)), axis=-1
    )
    return MetricEvaluator(
        u, theta, eta, positions, period=period, topology="toroidal",
        radial_degree=7, poloidal_modes=3, toroidal_modes=2,
    )


def test_from_payload_is_immutable_pytree_and_matches_eager_and_jit():
    evaluator = _evaluator()
    runtime = JaxMetricEvaluator.from_metric_evaluator(evaluator)
    leaves, _ = jax.tree_util.tree_flatten(runtime)
    assert leaves
    assert runtime.topology == "toroidal"

    rng = np.random.default_rng(1234)
    points = np.column_stack((rng.uniform(1.0e-6, 1.0, 31), rng.uniform(-2.0, 8.0, 31), rng.uniform(-0.5, 2.0 * np.pi + 0.5, 31)))
    for got, expected in (
        (runtime.position(points), evaluator.position(points)),
        (runtime.jacobian_matrix(points), evaluator.jacobian_matrix(points)),
        (runtime.evaluate(points).J, evaluator.evaluate(points).J),
    ):
        np.testing.assert_allclose(np.asarray(got), expected, rtol=2.0e-12, atol=2.0e-12)

    jit_position = jax.jit(runtime.position)
    jit_jacobian = jax.jit(runtime.jacobian_matrix)
    np.testing.assert_allclose(np.asarray(jit_position(jnp.asarray(points))), evaluator.position(points), rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(np.asarray(jit_jacobian(jnp.asarray(points))), evaluator.jacobian_matrix(points), rtol=2.0e-12, atol=2.0e-12)
    jit_state_position = jax.jit(lambda state, query: state.position(query))
    np.testing.assert_allclose(np.asarray(jit_state_position(runtime, jnp.asarray(points))), evaluator.position(points), rtol=2.0e-12, atol=2.0e-12)
    jit_evaluate = jax.jit(runtime.evaluate)
    np.testing.assert_allclose(np.asarray(jit_evaluate(jnp.asarray(points)).jacobian_matrix), evaluator.jacobian_matrix(points), rtol=2.0e-12, atol=2.0e-12)


def test_seam_and_near_axis_position_and_jacobian_parity():
    evaluator = _evaluator()
    runtime = JaxMetricEvaluator.from_metric_evaluator(evaluator)
    points = np.array(
        [[1.0e-8, 0.0, 0.0], [2.0e-7, 2.0 * np.pi, 2.0 * np.pi], [0.31, 0.5 * np.pi, 2.0 * np.pi + 1.0e-12]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(runtime.position(points), evaluator.position(points), rtol=3.0e-12, atol=3.0e-12)
    np.testing.assert_allclose(runtime.jacobian_matrix(points), evaluator.jacobian_matrix(points), rtol=3.0e-11, atol=3.0e-11)


def test_regularized_frame_parity_including_axis():
    evaluator = _evaluator()
    runtime = JaxMetricEvaluator.from_metric_evaluator(evaluator)
    points = np.array([[0.0, 0.0, 0.0], [0.0, 1.1, 2.4], [0.35, 0.8, 1.1]])
    expected = evaluator.evaluate_regularized(points)
    got = runtime.evaluate_regularized(points)
    np.testing.assert_allclose(got.position, expected.position, rtol=3.0e-12, atol=3.0e-12)
    np.testing.assert_allclose(got.jacobian_matrix, expected.jacobian_matrix, rtol=4.0e-11, atol=4.0e-11)
    np.testing.assert_allclose(got.J, expected.J, rtol=4.0e-11, atol=4.0e-11)


def test_rejects_square_payload():
    period = 2.0 * np.pi
    u = np.linspace(0.0, 1.0, 3)
    v = np.linspace(0.0, 1.0, 4)
    eta = period * np.arange(8) / 8.0
    ug, vg, eg = np.meshgrid(u, v, eta, indexing="ij")
    positions = np.stack(((3.0 + ug) * np.cos(eg), (3.0 + ug) * np.sin(eg), vg), axis=-1)
    evaluator = MetricEvaluator(u, v, eta, positions, period=period)
    with pytest.raises(ValueError, match="toroidal"):
        JaxMetricEvaluator.from_metric_evaluator(evaluator)
