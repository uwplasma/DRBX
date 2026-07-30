from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from drbx.geometry import (
    HaloLayout3D,
    LocalDomain3D,
    LocalFciGeometry3D,
    ShardSpec3D,
)
from drbx.native.fci_boundaries import (
    BC_NEUMANN,
    LocalBoundaryFaceBC3D,
)
from drbx.native.fci_halo import (
    GhostFillWeights1D,
    MetricAwarePhysicalGhostCellFiller3D,
)


def _metric_aware_fixture():
    owned_shape = (4, 5, 3)
    halo_width = 2
    layout = HaloLayout3D(owned_shape, halo_width)
    domain = LocalDomain3D(
        shard_spec=ShardSpec3D(
            global_shape=owned_shape,
            owned_start=(0, 0, 0),
            owned_stop=owned_shape,
            shard_index=(0, 0, 0),
            shard_counts=(1, 1, 1),
            periodic_axes=(False, False, True),
            halo_width=halo_width,
        ),
        layout=layout,
        mesh_axis_names=(None, None, None),
    )

    def axis_grid(axis: int):
        n = owned_shape[axis]
        centers = jnp.arange(-halo_width, n + halo_width, dtype=jnp.float64) + 0.5
        return SimpleNamespace(centers_halo=centers)

    skew = 0.5
    inverse_metric = jnp.asarray(
        (
            (1.0, skew, 0.0),
            (skew, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=jnp.float64,
    )

    def face_metric(axis: int):
        return SimpleNamespace(
            g_contra=jnp.broadcast_to(
                inverse_metric,
                layout.face_halo_shape(axis) + (3, 3),
            )
        )

    geometry = object.__new__(LocalFciGeometry3D)
    object.__setattr__(geometry, "layout", layout)
    object.__setattr__(
        geometry,
        "grid",
        SimpleNamespace(x=axis_grid(0), y=axis_grid(1), z=axis_grid(2)),
    )
    object.__setattr__(
        geometry,
        "face_metric",
        SimpleNamespace(
            axes=(face_metric(0), face_metric(1), face_metric(2)),
        ),
    )

    dirichlet = GhostFillWeights1D(
        owned_weights=-jnp.ones((halo_width, 1), dtype=jnp.float64),
        bc_weights=2.0 * jnp.ones((halo_width,), dtype=jnp.float64),
    )
    neutral = GhostFillWeights1D(
        owned_weights=jnp.ones((halo_width, 1), dtype=jnp.float64),
        bc_weights=jnp.zeros((halo_width,), dtype=jnp.float64),
    )
    filler = MetricAwarePhysicalGhostCellFiller3D(
        dirichlet=(dirichlet, dirichlet, dirichlet),
        neumann_lower=(neutral, neutral, neutral),
        neumann_upper=(neutral, neutral, neutral),
        geometry=geometry,
    )

    slope_x = 0.7
    slope_y = -0.4
    x = geometry.grid.x.centers_halo
    y = geometry.grid.y.centers_halo
    z = geometry.grid.z.centers_halo
    xx, yy, _zz = jnp.meshgrid(x, y, z, indexing="ij")
    expected = slope_x * xx + slope_y * yy
    field = jnp.zeros(layout.cell_halo_shape, dtype=jnp.float64)
    field = field.at[layout.owned_slices_cell].set(expected[layout.owned_slices_cell])

    bc = LocalBoundaryFaceBC3D.empty(layout)
    x_normal = slope_x + skew * slope_y
    y_normal = skew * slope_x + slope_y
    bc = replace(
        bc,
        kind_x=bc.kind_x.at[0].set(BC_NEUMANN).at[-1].set(BC_NEUMANN),
        kind_y=bc.kind_y.at[:, 0, :].set(BC_NEUMANN).at[:, -1, :].set(BC_NEUMANN),
        value_x=bc.value_x.at[0].set(-x_normal).at[-1].set(x_normal),
        value_y=bc.value_y.at[:, 0, :].set(-y_normal).at[:, -1, :].set(y_normal),
        mask_x=bc.mask_x.at[0].set(True).at[-1].set(True),
        mask_y=bc.mask_y.at[:, 0, :].set(True).at[:, -1, :].set(True),
    )
    return filler, field, expected, domain, bc


def test_metric_aware_neumann_reproduces_linear_physical_normal_data():
    filler, field, expected, domain, bc = _metric_aware_fixture()
    filled = filler(field, domain, bc)
    h = domain.layout.halo_width
    nx, ny, nz = domain.layout.owned_shape

    np.testing.assert_allclose(
        filled[:h, h : h + ny, h : h + nz],
        expected[:h, h : h + ny, h : h + nz],
        atol=2.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        filled[h + nx :, h : h + ny, h : h + nz],
        expected[h + nx :, h : h + ny, h : h + nz],
        atol=2.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        filled[h : h + nx, :h, h : h + nz],
        expected[h : h + nx, :h, h : h + nz],
        atol=2.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        filled[h : h + nx, h + ny :, h : h + nz],
        expected[h : h + nx, h + ny :, h : h + nz],
        atol=2.0e-12,
        rtol=0.0,
    )


def test_metric_aware_neumann_is_jittable():
    filler, field, _expected, domain, bc = _metric_aware_fixture()
    eager = filler(field, domain, bc)
    compiled = jax.jit(lambda value: filler(value, domain, bc))(field)
    np.testing.assert_allclose(compiled, eager, atol=1.0e-12, rtol=0.0)
