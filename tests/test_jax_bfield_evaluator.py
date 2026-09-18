"""Parity tests for the standalone JAX component-spline consumer."""

from __future__ import annotations

import dataclasses
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from drbx.geometry.Bfield_evaluator import ComponentSplineBFieldEvaluator
from drbx.geometry.jax_bfield_evaluator import (
    JaxComponentSplineBFieldEvaluator,
    jax_bfield_evaluator_from_component_spline,
)


def _evaluator(method: str = "cubic") -> ComponentSplineBFieldEvaluator:
    R = np.linspace(1.2, 2.0, 8, dtype=np.float64)
    Z = np.linspace(-0.7, 0.65, 9, dtype=np.float64)
    nfp = 3
    period = 2.0 * np.pi / nfp
    phi = np.arange(10, dtype=np.float64) * period / 10.0
    pp, zz, rr = np.meshgrid(phi, Z, R, indexing="ij")
    field = np.stack(
        (
            0.12 + 0.03 * np.sin(2.0 * pp) + 0.02 * (rr - 1.6) * zz,
            -0.2 + 0.04 * np.cos(pp + zz) + 0.01 * (rr - 1.6) ** 2,
            0.05 * np.sin(pp - 0.4 * zz) + 0.03 * (rr - 1.6),
        ),
        axis=-1,
    )
    return ComponentSplineBFieldEvaluator(
        R, phi, Z, field, nfp=nfp, method=method, extrapolate=False
    )


@pytest.mark.parametrize("method", ["cubic", "linear"])
def test_eager_and_jitted_cylindrical_parity_random(method: str) -> None:
    scipy_evaluator = _evaluator(method)
    jax_evaluator = jax_bfield_evaluator_from_component_spline(scipy_evaluator)
    rng = np.random.default_rng(1234)
    points = np.column_stack(
        (
            rng.uniform(scipy_evaluator.R[0], scipy_evaluator.R[-1], 257),
            rng.uniform(-4.0 * scipy_evaluator.period, 5.0 * scipy_evaluator.period, 257),
            rng.uniform(scipy_evaluator.Z[0], scipy_evaluator.Z[-1], 257),
        )
    ).reshape(257, 1, 3)

    expected = scipy_evaluator.evaluate_cylindrical(points)
    np.testing.assert_allclose(
        np.asarray(jax_evaluator.evaluate_cylindrical(points)),
        expected,
        rtol=2e-13,
        atol=2e-13,
    )
    compiled = jax.jit(jax_evaluator.evaluate_cylindrical)(jnp.asarray(points))
    np.testing.assert_allclose(np.asarray(compiled), expected, rtol=2e-13, atol=2e-13)


def test_cubic_phi_seam_and_reflected_rz_boundaries() -> None:
    scipy_evaluator = _evaluator("cubic")
    jax_evaluator = JaxComponentSplineBFieldEvaluator.from_evaluator(scipy_evaluator)
    eps = 1.0e-11
    # The first/last points probe both sides of the periodic seam.  The other
    # points are just inside each reflected R/Z boundary.
    points = np.array(
        [
            [scipy_evaluator.R[0] + eps, scipy_evaluator.phi[0] - eps, scipy_evaluator.Z[0] + eps],
            [scipy_evaluator.R[-1] - eps, scipy_evaluator.phi[0] + scipy_evaluator.period - eps, scipy_evaluator.Z[-1] - eps],
            [scipy_evaluator.R[0] + 0.25 * (scipy_evaluator.R[1] - scipy_evaluator.R[0]), scipy_evaluator.phi[0] + 0.5 * scipy_evaluator.period, scipy_evaluator.Z[-1] - 0.25 * (scipy_evaluator.Z[-1] - scipy_evaluator.Z[-2])],
        ],
        dtype=np.float64,
    )
    expected = scipy_evaluator.evaluate_cylindrical(points)
    np.testing.assert_allclose(
        np.asarray(jax.jit(jax_evaluator.evaluate_cylindrical)(points)),
        expected,
        rtol=2e-13,
        atol=2e-13,
    )


def test_cartesian_parity_and_arbitrary_leading_shape() -> None:
    scipy_evaluator = _evaluator("cubic")
    jax_evaluator = JaxComponentSplineBFieldEvaluator.from_evaluator(scipy_evaluator)
    points_rphiz = np.array(
        [
            [[1.25, -0.2, -0.5], [1.65, 0.3, 0.1]],
            [[1.9, 1.0, 0.6], [1.4, -2.0, -0.2]],
        ],
        dtype=np.float64,
    )
    R, phi, Z = np.moveaxis(points_rphiz, -1, 0)
    points_xyz = np.stack((R * np.cos(phi), R * np.sin(phi), Z), axis=-1)
    expected = scipy_evaluator.evaluate_cartesian(points_xyz)
    np.testing.assert_allclose(
        np.asarray(jax.jit(jax_evaluator.evaluate_cartesian)(points_xyz)),
        expected,
        rtol=2e-13,
        atol=2e-13,
    )


def test_state_is_immutable_pytree_and_replicated_sharding_compatible() -> None:
    scipy_evaluator = _evaluator("cubic")
    state = JaxComponentSplineBFieldEvaluator.from_evaluator(scipy_evaluator)
    leaves, _ = jax.tree_util.tree_flatten(state)
    assert len(leaves) == 7
    for expected, actual in zip(scipy_evaluator._coefficients, leaves[3:]):
        np.testing.assert_array_equal(np.asarray(actual), expected)
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        state.method = "linear"  # type: ignore[misc]

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("device",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    replicated = jax.device_put(state, sharding)
    points = jnp.asarray([[1.35, 0.2, -0.1], [1.8, 1.1, 0.4]], dtype=jnp.float64)
    expected = scipy_evaluator.evaluate_cylindrical(np.asarray(points))
    actual = jax.jit(lambda s, p: s.evaluate_cylindrical(p))(replicated, points)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-13, atol=2e-13)
