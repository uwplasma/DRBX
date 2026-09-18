"""Parity and sharding checks for the continuous-field JAX FCI tracer."""

import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from drbx.geometry.fci_geometry import (
    CellCenteredGrid3D,
    Grid1D,
    trace_fci_points_to_plane_from_callbacks,
)
from drbx.geometry.fci_trace_jax import trace_fci_points_to_plane_jax


BOUNDS = np.asarray([[0.0, 1.0], [0.0, 2.0 * np.pi], [0.0, 1.0]])


def _grid() -> CellCenteredGrid3D:
    return CellCenteredGrid3D(
        Grid1D(jnp.asarray([0.125, 0.375, 0.625, 0.875]), jnp.linspace(0.0, 1.0, 5)),
        Grid1D(
            jnp.asarray([np.pi / 4.0, 3.0 * np.pi / 4.0, 5.0 * np.pi / 4.0, 7.0 * np.pi / 4.0]),
            jnp.linspace(0.0, 2.0 * np.pi, 5),
        ),
        Grid1D(jnp.asarray([0.125, 0.375, 0.625, 0.875]), jnp.linspace(0.0, 1.0, 5)),
    )


def _constant_numpy(bx, by, bmag=1.0):
    def callback(points):
        points = np.asarray(points)
        return np.broadcast_to([bx, by, 1.0], (points.shape[0], 3)).copy(), np.full(points.shape[0], bmag)

    return callback


def _constant_jax(bx, by, bmag=1.0):
    def callback(points):
        return jnp.broadcast_to(jnp.asarray([bx, by, 1.0]), (points.shape[0], 3)), jnp.full((points.shape[0],), bmag)

    return callback


def _compare(points, eta, numpy_callback, jax_callback, *, substeps=4, axis_regular=False, mask=None):
    grid = _grid()
    kwargs = {
        "substeps": substeps,
        "periodic_axes": (False, True, True),
        "axis_regular_axes": (axis_regular, False, False),
    }
    if np.ndim(eta) == 0:
        reference = trace_fci_points_to_plane_from_callbacks(
            grid, numpy_callback, points, eta, return_numpy=True, **kwargs
        )
    else:
        # The existing NumPy callback API has a scalar eta_step, so establish
        # the per-seed reference one trajectory at a time.
        pieces = [
            trace_fci_points_to_plane_from_callbacks(
                grid, numpy_callback, points[i : i + 1], float(eta[i]), return_numpy=True, **kwargs
            )
            for i in range(points.shape[0])
        ]
        reference = {
            name: np.concatenate([piece[name] for piece in pieces], axis=0)
            for name in pieces[0]
        }

    # Keep the callback in the transformed closure: JAX callbacks are static
    # Python callables, while seed/step/mask arrays remain transform inputs.
    def run(seed_values, step_values, valid_values):
        return trace_fci_points_to_plane_jax(
            BOUNDS,
            jax_callback,
            seed_values,
            step_values,
            valid_mask=valid_values,
            **kwargs,
        )

    valid = np.ones(points.shape[0], dtype=bool) if mask is None else np.asarray(mask)
    args = (jnp.asarray(points), jnp.asarray(eta), jnp.asarray(valid))
    eager = run(*args)
    actual = jax.jit(run)(*args)
    for result in (eager, actual):
        for name in ("endpoint", "length", "endpoint_b_contravariant", "endpoint_bmag"):
            np.testing.assert_allclose(np.asarray(result[name]), np.asarray(reference[name]), rtol=2.0e-13, atol=2.0e-13)
        np.testing.assert_array_equal(np.asarray(result["boundary"]), np.asarray(reference["boundary"]))


def test_identity_field_eager_and_jit_match_numpy():
    points = np.asarray([[0.2, 0.1, 0.2], [0.8, 6.2, 0.7]])
    _compare(points, 0.25, _constant_numpy(0.0, 0.0), _constant_jax(0.0, 0.0), substeps=3)


def test_transverse_drift_with_per_seed_signed_steps_matches_numpy():
    points = np.asarray([[0.2, 0.1, 0.2], [0.7, 6.1, 0.7], [0.4, 3.0, 0.4]])

    def numpy_callback(values):
        values = np.asarray(values)
        return np.column_stack((0.35 + 0.1 * values[:, 0], -0.2 + 0.03 * values[:, 1], np.ones(values.shape[0]))), np.ones(values.shape[0])

    def jax_callback(values):
        return jnp.stack((0.35 + 0.1 * values[:, 0], -0.2 + 0.03 * values[:, 1], jnp.ones(values.shape[0])), axis=-1), jnp.ones(values.shape[0])

    _compare(points, np.asarray([0.25, -0.2, 0.15]), numpy_callback, jax_callback, substeps=5)


def test_axis_crossing_signed_radius_matches_numpy():
    points = np.asarray([[0.125, np.pi / 4.0, 0.125], [0.375, 3.0 * np.pi / 4.0, 0.125]])

    def numpy_callback(values):
        values = np.asarray(values)
        radial = -0.75 * np.cos(values[:, 1]) / np.cos(np.pi / 4.0)
        return np.column_stack((radial, np.zeros(values.shape[0]), np.ones(values.shape[0]))), np.ones(values.shape[0])

    def jax_callback(values):
        radial = -0.75 * jnp.cos(values[:, 1]) / jnp.cos(np.pi / 4.0)
        return jnp.stack((radial, jnp.zeros(values.shape[0]), jnp.ones(values.shape[0])), axis=-1), jnp.ones(values.shape[0])

    _compare(points, 0.25, numpy_callback, jax_callback, substeps=1, axis_regular=True)


def test_wall_exit_chord_and_signed_steps_match_numpy():
    points = np.asarray([[0.875, 0.2, 0.2], [0.125, 6.1, 0.2]])
    _compare(points, np.asarray([0.25, -0.25]), _constant_numpy(4.0, 0.0), _constant_jax(4.0, 0.0), substeps=1)


def test_named_sharding_is_batch_local_and_padding_mask_is_preserved():
    devices = jax.devices()
    pytest.importorskip("jax.sharding")
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    points = np.asarray([[0.2, 0.1, 0.2], [0.8, 6.2, 0.7], [0.4, 3.0, 0.4], [0.3, 1.0, 0.1]])
    eta = np.asarray([0.25, 0.15, -0.2, 0.25])
    valid = np.asarray([True, True, True, False])
    mesh = Mesh(np.asarray(devices[:1]), ("trace",))
    point_sharding = NamedSharding(mesh, PartitionSpec("trace", None))
    vector_sharding = NamedSharding(mesh, PartitionSpec("trace"))

    def callback(values):
        return jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0]), (values.shape[0], 3)), jnp.ones(values.shape[0])

    run = jax.jit(
        lambda p, h, m: trace_fci_points_to_plane_jax(
            BOUNDS, callback, p, h, substeps=3, valid_mask=m
        )
    )
    actual = run(
        jax.device_put(points, point_sharding),
        jax.device_put(eta, vector_sharding),
        jax.device_put(valid, vector_sharding),
    )
    np.testing.assert_allclose(np.asarray(actual["endpoint"][:3, 2]), points[:3, 2] + eta[:3])
    np.testing.assert_allclose(np.asarray(actual["length"]), np.asarray([0.25, 0.15, 0.2, 0.0]))
    np.testing.assert_array_equal(np.asarray(actual["boundary"]), valid & False)
