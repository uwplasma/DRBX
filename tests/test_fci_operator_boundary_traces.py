from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from drbx.geometry import HaloLayout3D, LocalDomain3D, LocalFciGeometry3D, ShardSpec3D
from drbx.geometry.fci_geometry import SIDE_AXIS_REGULAR, SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC
from drbx.native.fci_boundaries import (
    BC_DIRICHLET,
    BC_NEUMANN,
    LocalBoundaryFaceBC3D,
    build_local_boundary_face_trace_from_halo,
)
from drbx.native.fci_halo import (
    GhostFillWeights1D,
    MetricAwarePhysicalGhostCellFiller3D,
)


def _domain(layout, *, axis_regular_x=False):
    side_lower = (
        SIDE_AXIS_REGULAR if axis_regular_x else SIDE_PHYSICAL,
        SIDE_SIMPLE_PERIODIC,
        SIDE_SIMPLE_PERIODIC,
    )
    side_upper = (SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_SIMPLE_PERIODIC)
    return LocalDomain3D(
        shard_spec=ShardSpec3D(
            global_shape=layout.owned_shape,
            owned_start=(0, 0, 0),
            owned_stop=layout.owned_shape,
            shard_index=(0, 0, 0),
            shard_counts=(1, 1, 1),
            periodic_axes=(False, True, True),
            axis_regular_axes=(axis_regular_x, False, False),
            side_kind_lower=side_lower,
            side_kind_upper=side_upper,
            halo_width=layout.halo_width,
        ),
        layout=layout,
        mesh_axis_names=(None, None, None),
    )


def _geometry(layout, x_centers, *, x_faces=None):
    h = layout.halo_width
    if x_faces is None:
        x_faces = jnp.arange(-h, layout.owned_shape[0] + h + 1, dtype=jnp.float64)
    axis_grid = lambda centers, faces: SimpleNamespace(
        centers_halo=jnp.asarray(centers, dtype=jnp.float64),
        faces_halo=jnp.asarray(faces, dtype=jnp.float64),
    )
    y = jnp.arange(-h, layout.owned_shape[1] + h, dtype=jnp.float64) + 0.5
    z = jnp.arange(-h, layout.owned_shape[2] + h, dtype=jnp.float64) + 0.5
    geometry = object.__new__(LocalFciGeometry3D)
    object.__setattr__(geometry, "layout", layout)
    object.__setattr__(
        geometry,
        "grid",
        SimpleNamespace(
            x=axis_grid(x_centers, x_faces),
            y=axis_grid(y, jnp.arange(-h, layout.owned_shape[1] + h + 1)),
            z=axis_grid(z, jnp.arange(-h, layout.owned_shape[2] + h + 1)),
        ),
    )
    return geometry


def _orthogonal_metric_geometry(layout, x_centers):
    geometry = _geometry(layout, x_centers)
    metric = jnp.broadcast_to(
        jnp.eye(3, dtype=jnp.float64),
        layout.face_halo_shape(0) + (3, 3),
    )
    object.__setattr__(
        geometry,
        "face_metric",
        SimpleNamespace(
            axes=tuple(
                SimpleNamespace(
                    g_contra=jnp.broadcast_to(
                        jnp.eye(3, dtype=jnp.float64),
                        layout.face_halo_shape(axis) + (3, 3),
                    )
                )
                for axis in range(3)
            )
        ),
    )
    del metric
    return geometry


def test_neumann_trace_uses_all_two_halo_samples_and_lagrange_interpolation():
    layout = HaloLayout3D((4, 3, 2), 2)
    domain = _domain(layout)
    centers = jnp.arange(-2, 6, dtype=jnp.float64) + 0.5
    geometry = _geometry(layout, centers)
    x = centers[:, None, None]
    field = jnp.broadcast_to(x**3 + x**2 + 0.3 * x + 1.0, layout.cell_halo_shape)
    bc = LocalBoundaryFaceBC3D.empty(layout)
    bc = replace(
        bc,
        kind_x=bc.kind_x.at[0].set(BC_NEUMANN).at[-1].set(BC_DIRICHLET),
        value_x=bc.value_x.at[-1].set(7.0),
        mask_x=bc.mask_x.at[0].set(True).at[-1].set(True),
    )

    trace = build_local_boundary_face_trace_from_halo(field, geometry, domain, bc)
    expected_lower = 1.0
    first_pair_average = float((field[1, 2, 2] + field[2, 2, 2]) / 2.0)
    np.testing.assert_allclose(trace.value_x[0], expected_lower, atol=1.0e-12)
    assert not np.isclose(expected_lower, first_pair_average)
    np.testing.assert_allclose(trace.value_x[-1], 7.0, atol=0.0, rtol=0.0)
    assert bool(trace.mask_x[0, 0, 0])
    assert bool(trace.mask_x[-1, 0, 0])
    np.testing.assert_allclose(trace.value_y, 0.0)
    np.testing.assert_allclose(trace.value_z, 0.0)


def test_axis_regular_face_is_inactive_even_with_physics_payload():
    layout = HaloLayout3D((4, 3, 2), 2)
    domain = _domain(layout, axis_regular_x=True)
    centers = jnp.arange(-2, 6, dtype=jnp.float64) + 0.5
    geometry = _geometry(layout, centers)
    field = jnp.ones(layout.cell_halo_shape, dtype=jnp.float64)
    bc = LocalBoundaryFaceBC3D.empty(layout)
    bc = replace(
        bc,
        kind_x=bc.kind_x.at[0].set(BC_NEUMANN),
        value_x=bc.value_x.at[0].set(3.0),
        mask_x=bc.mask_x.at[0].set(True),
    )
    trace = build_local_boundary_face_trace_from_halo(field, geometry, domain, bc)
    assert not bool(jnp.any(trace.mask_x[0]))
    np.testing.assert_allclose(trace.value_x[0], 0.0)


def test_short_periodic_axis_does_not_block_other_axis_trace():
    layout = HaloLayout3D((4, 1, 2), 2)
    domain = _domain(layout)
    centers = jnp.arange(-2, 6, dtype=jnp.float64) + 0.5
    geometry = _geometry(layout, centers)
    x = centers[:, None, None]
    field = jnp.broadcast_to(x**2 + 1.0, layout.cell_halo_shape)
    bc = LocalBoundaryFaceBC3D.empty(layout)
    bc = replace(
        bc,
        kind_x=bc.kind_x.at[0].set(BC_NEUMANN),
        mask_x=bc.mask_x.at[0].set(True),
    )

    trace = build_local_boundary_face_trace_from_halo(field, geometry, domain, bc)

    np.testing.assert_allclose(trace.value_x[0], 1.0, atol=1.0e-12)
    assert bool(jnp.all(trace.mask_x[0]))
    assert not bool(jnp.any(trace.mask_y))


def test_metric_neumann_pairs_each_ghost_with_matching_owner_on_nonuniform_grid():
    layout = HaloLayout3D((4, 2, 2), 2)
    domain = _domain(layout)
    centers = jnp.asarray((-1.7, -0.6, 0.5, 1.7, 3.1, 4.8, 6.8, 9.1))
    geometry = _orthogonal_metric_geometry(layout, centers)
    zero = jnp.zeros((2, 1), dtype=jnp.float64)
    dirichlet = GhostFillWeights1D(
        owned_weights=-jnp.ones((2, 1), dtype=jnp.float64),
        bc_weights=2.0 * jnp.ones((2,), dtype=jnp.float64),
    )
    neutral = GhostFillWeights1D(
        owned_weights=jnp.ones((2, 1), dtype=jnp.float64),
        bc_weights=zero[:, 0],
    )
    filler = MetricAwarePhysicalGhostCellFiller3D(
        dirichlet=(dirichlet, dirichlet, dirichlet),
        neumann_lower=(neutral, neutral, neutral),
        neumann_upper=(neutral, neutral, neutral),
        geometry=geometry,
    )
    field = jnp.zeros(layout.cell_halo_shape, dtype=jnp.float64)
    owned_values = jnp.asarray((10.0, 20.0, 30.0, 40.0))[:, None, None]
    field = field.at[layout.owned_slices_cell].set(
        jnp.broadcast_to(owned_values, layout.owned_shape)
    )
    bc = LocalBoundaryFaceBC3D.empty(layout)
    bc = replace(
        bc,
        kind_x=bc.kind_x.at[0].set(BC_NEUMANN).at[-1].set(BC_NEUMANN),
        value_x=bc.value_x.at[0].set(2.0).at[-1].set(3.0),
        mask_x=bc.mask_x.at[0].set(True).at[-1].set(True),
    )
    filled = filler(field, domain, bc)

    # Orthogonal metric: lower logical derivative is -2 and upper is +3.
    np.testing.assert_allclose(filled[1, 2, 2], 10.0 + (-2.0) * (-0.6 - 0.5))
    np.testing.assert_allclose(filled[0, 2, 2], 20.0 + (-2.0) * (-1.7 - 1.7))
    np.testing.assert_allclose(filled[6, 2, 2], 40.0 + 3.0 * (6.8 - 4.8))
    np.testing.assert_allclose(filled[7, 2, 2], 30.0 + 3.0 * (9.1 - 3.1))


def test_boundary_trace_builder_is_jittable():
    layout = HaloLayout3D((4, 3, 2), 2)
    domain = _domain(layout)
    centers = jnp.arange(-2, 6, dtype=jnp.float64) + 0.5
    geometry = _geometry(layout, centers)
    field = jnp.arange(np.prod(layout.cell_halo_shape), dtype=jnp.float64).reshape(
        layout.cell_halo_shape
    )
    bc = LocalBoundaryFaceBC3D.empty(layout)
    bc = replace(
        bc,
        kind_x=bc.kind_x.at[0].set(BC_NEUMANN),
        mask_x=bc.mask_x.at[0].set(True),
    )
    eager = build_local_boundary_face_trace_from_halo(field, geometry, domain, bc)
    compiled = jax.jit(
        lambda value: build_local_boundary_face_trace_from_halo(
            value, geometry, domain, bc
        )
    )(field)
    np.testing.assert_allclose(compiled.value_x, eager.value_x)
    np.testing.assert_array_equal(compiled.mask_x, eager.mask_x)
