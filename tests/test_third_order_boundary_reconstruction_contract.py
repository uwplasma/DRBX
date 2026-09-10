"""Focused contracts for the third-order physical-face reconstruction repair."""

from dataclasses import replace
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drbx.native.fci_boundaries import LocalBoundaryFaceTrace3D
from drbx.native.fci_operators import (
    _axis_face_samples_from_halo,
    _third_order_scalar_face_states_from_halo,
)
from test_fci_operators_domain_decomp import _build_local_geometry
from axis_regular_operator_support import polar_fixture


def _x_trace(geometry, *, lower=None, upper=None):
    trace = LocalBoundaryFaceTrace3D.empty(geometry.layout)
    if lower is not None:
        trace = replace(
            trace,
            mask_x=trace.mask_x.at[0].set(lower[0]),
            value_x=trace.value_x.at[0].set(lower[1]),
        )
    if upper is not None:
        trace = replace(
            trace,
            mask_x=trace.mask_x.at[-1].set(upper[0]),
            value_x=trace.value_x.at[-1].set(upper[1]),
        )
    return trace


def test_first_interior_physical_face_is_exact_quadratic_and_ghost_independent():
    """The repaired face uses q0,q1,q2 and exactly reproduces a quadratic."""
    shape = (5, 3, 2)
    geometry = _build_local_geometry(shape, 2, global_shape=shape)
    h = geometry.layout.halo_width
    x = geometry.grid.x.centers_halo[:, None, None]
    values = jnp.broadcast_to((x + 2.0) ** 2, geometry.halo_shape)
    mask = jnp.ones((shape[1], shape[2]), dtype=bool)
    trace = _x_trace(geometry, lower=(mask, jnp.full(mask.shape, 91.0)))
    changed_values = values.at[h - 1].set(-1.0e7)
    changed_values = changed_values.at[h + shape[0]].set(2.0e7)
    baseline = _third_order_scalar_face_states_from_halo(
        values, geometry, boundary_trace=trace,
        axis_regular_axes=(False, False, False), positivity_floor=None,
    )
    changed = _third_order_scalar_face_states_from_halo(
        changed_values, geometry, boundary_trace=trace,
        axis_regular_axes=(False, False, False), positivity_floor=None,
    )
    samples = _axis_face_samples_from_halo(values, geometry, axis=0)
    q0, q1, q2 = samples[1][1], samples[2][1], samples[3][1]
    expected = (2.0 * q0 + 5.0 * q1 - q2) / 6.0
    np.testing.assert_allclose(baseline[0].x[1], expected, rtol=0.0, atol=1e-13)
    np.testing.assert_array_equal(baseline[0].x[1], baseline[1].x[1])
    np.testing.assert_array_equal(baseline[0].x[1], changed[0].x[1])
    np.testing.assert_array_equal(baseline[1].x[1], changed[1].x[1])


def test_masked_physical_wall_repairs_only_selected_columns():
    shape = (5, 3, 2)
    geometry = _build_local_geometry(shape, 2, global_shape=shape)
    h = geometry.layout.halo_width
    values = jnp.arange(np.prod(geometry.halo_shape), dtype=jnp.float64).reshape(geometry.halo_shape)
    lower_mask = jnp.array([[True, False], [False, True], [False, False]])
    upper_mask = jnp.array([[False, True], [True, False], [False, False]])
    trace = _x_trace(geometry, lower=(lower_mask, jnp.full(lower_mask.shape, 17.0)), upper=(upper_mask, jnp.full(upper_mask.shape, -23.0)))
    left, right, fallback = _third_order_scalar_face_states_from_halo(
        values, geometry, boundary_trace=trace,
        axis_regular_axes=(False, False, False), positivity_floor=None,
    )
    np.testing.assert_array_equal(left.x[1][lower_mask], right.x[1][lower_mask])
    np.testing.assert_array_equal(left.x[-2][upper_mask], right.x[-2][upper_mask])
    assert bool(jnp.all(fallback.x[1][lower_mask]))
    assert bool(jnp.all(fallback.x[-2][upper_mask]))
    # An unmasked column keeps the canonical high-order value.
    qm, q0, q1, qp = (sample[2, 0, 0] for sample in _axis_face_samples_from_halo(values, geometry, axis=0))
    np.testing.assert_allclose(left.x[2, 0, 0], (-qm + 5*q0 + 2*q1) / 6)


def test_axis_regular_lower_radial_side_ignores_physical_trace_mask():
    geometry, _domain, _context, _coords, *_ = polar_fixture(shape=(6, 8, 3), halo_width=2)
    values = jnp.ones(geometry.halo_shape, dtype=jnp.float64)
    mask = jnp.ones((geometry.owned_shape[1], geometry.owned_shape[2]), dtype=bool)
    trace = _x_trace(geometry, lower=(mask, jnp.full(mask.shape, 1234.0)))
    with_trace = _third_order_scalar_face_states_from_halo(
        values, geometry, boundary_trace=trace,
        axis_regular_axes=(True, False, False), positivity_floor=None,
    )
    without_trace = _third_order_scalar_face_states_from_halo(
        values, geometry, boundary_trace=None,
        axis_regular_axes=(True, False, False), positivity_floor=None,
    )
    np.testing.assert_array_equal(with_trace[0].x[0], without_trace[0].x[0])
    np.testing.assert_array_equal(with_trace[1].x[0], without_trace[1].x[0])
    np.testing.assert_array_equal(with_trace[0].x[1], without_trace[0].x[1])
    np.testing.assert_array_equal(with_trace[1].x[1], without_trace[1].x[1])


def test_periodic_and_shard_seams_keep_canonical_shared_values():
    # Periodic y seam: both local faces see the same periodic halo support.
    shape = (4, 8, 3)
    geometry = _build_local_geometry(shape, 2, global_shape=shape)
    y = geometry.grid.y.centers_halo[None, :, None]
    values = jnp.broadcast_to(jnp.sin(y), geometry.halo_shape)
    left, right, fallback = _third_order_scalar_face_states_from_halo(
        values, geometry, boundary_trace=None,
        axis_regular_axes=(False, False, False), positivity_floor=None,
    )
    np.testing.assert_allclose(left.y[:, 0], left.y[:, -1], atol=1e-14)
    np.testing.assert_allclose(right.y[:, 0], right.y[:, -1], atol=1e-14)
    np.testing.assert_array_equal(fallback.y[:, 0], fallback.y[:, -1])

    # A two-way y shard seam must agree on the shared physical face.
    global_shape, local_shape = (4, 8, 4), (4, 4, 4)
    seam = []
    for shard_y in (0, 1):
        g = _build_local_geometry(local_shape, 2, global_shape=global_shape, shard_index=(0, shard_y, 0))
        x, y, z = jnp.meshgrid(g.grid.x.centers_halo, g.grid.y.centers_halo, g.grid.z.centers_halo, indexing="ij")
        f = 1.5 + 0.1*x + 0.2*jnp.sin(y) + 0.05*jnp.cos(z)
        seam.append(_third_order_scalar_face_states_from_halo(f, g, boundary_trace=None, axis_regular_axes=(False,False,False), positivity_floor=None))
    np.testing.assert_allclose(seam[0][0].y[:, -1], seam[1][0].y[:, 0], atol=1e-14)
    np.testing.assert_allclose(seam[0][1].y[:, -1], seam[1][1].y[:, 0], atol=1e-14)


def test_short_axis_uses_adjacent_owner_fallback():
    shape = (2, 3, 2)
    geometry = _build_local_geometry(shape, 2, global_shape=shape)
    values = jnp.arange(np.prod(geometry.halo_shape), dtype=jnp.float64).reshape(geometry.halo_shape)
    mask = jnp.ones((shape[1], shape[2]), dtype=bool)
    trace = _x_trace(geometry, lower=(mask, jnp.zeros(mask.shape)), upper=(mask, jnp.zeros(mask.shape)))
    left, right, fallback = _third_order_scalar_face_states_from_halo(values, geometry, boundary_trace=trace, axis_regular_axes=(False,False,False), positivity_floor=None)
    h = geometry.layout.halo_width
    q0 = values[h, h:h+shape[1], h:h+shape[2]]
    q1 = values[h+1, h:h+shape[1], h:h+shape[2]]
    np.testing.assert_array_equal(left.x[1], q0)
    np.testing.assert_array_equal(right.x[1], q1)
    assert bool(jnp.all(fallback.x[1]))
