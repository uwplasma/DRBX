"""Reusable analytic polar-axis fixture for local non-FCI operator tests."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys

import jax.numpy as jnp

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from drbx.geometry import (  # noqa: E402
    HaloLayout3D,
    LocalBFieldGeometry,
    LocalCellCenteredGrid3D,
    LocalCellVolumeGeometry3D,
    LocalDomain3D,
    LocalFaceBFieldGeometry,
    LocalFaceMetricGeometry,
    LocalFciGeometry3D,
    LocalGrid1D,
    LocalMetricGeometry,
    LocalSpacing3D,
    SIDE_AXIS_REGULAR,
    SIDE_PHYSICAL,
    SIDE_SIMPLE_PERIODIC,
    ShardSpec3D,
    StencilBuilderContext,
)
from drbx.native.fci_boundaries import LocalCellGradient3D  # noqa: E402
from drbx.native.fci_halo import (  # noqa: E402
    HaloExchange3D,
    PolarAxisRegularScalarRule3D,
    PolarAxisRegularVectorRule3D,
    TopologyHaloFiller3D,
    LocalPeriodicTopologyRule3D,
)

# The existing fixture supplies the complete, valid local maps/face containers.
from test_fci_operators_domain_decomp import _build_local_geometry  # noqa: E402


def _axis_metric(layout, location, r, theta, z):
    del z
    r, theta = jnp.broadcast_arrays(jnp.asarray(r, dtype=jnp.float64), jnp.asarray(theta, dtype=jnp.float64))
    del theta
    rr = jnp.abs(r)
    inv_r2 = jnp.where(rr > 1.0e-14, 1.0 / rr**2, 0.0)
    zeros = jnp.zeros_like(rr)
    return LocalMetricGeometry(
        layout=layout, location=location,
        J_halo=rr, g11_halo=jnp.ones_like(rr), g22_halo=inv_r2, g33_halo=jnp.ones_like(rr),
        g12_halo=zeros, g13_halo=zeros, g23_halo=zeros,
        g_11_halo=jnp.ones_like(rr), g_22_halo=rr**2, g_33_halo=jnp.ones_like(rr),
        g_12_halo=zeros, g_13_halo=zeros, g_23_halo=zeros,
    )


def _axis_bfield(metric):
    shape = metric.shape
    b = jnp.zeros(shape + (3,), dtype=jnp.float64).at[..., 2].set(1.0)
    return LocalBFieldGeometry(
        layout=metric.layout, B_contra_halo=b,
        Bmag_halo=jnp.ones(shape, dtype=jnp.float64), location=metric.location,
    )


def polar_fixture(shape=(8, 16, 16), halo_width=1):
    """Return ``(geometry, domain, context, coordinates, fillers)``.

    The radial cell centers are positive; the lower radial face is the polar
    axis. Scalar halos are filled by the exact half-turn rule. Two vector
    fillers are returned because an ordinary contravariant vector transforms
    with ``diag(-1, +1, +1)``, while the contravariant vector density
    ``F^i = J P^{ij} partial_j f`` transforms with ``diag(+1, -1, -1)``.
    """
    shape = tuple(int(v) for v in shape)
    nx, ny, nz = shape
    base = _build_local_geometry(shape, halo_width, global_shape=shape)
    layout = base.layout
    h = int(halo_width)
    dr = 1.0 / nx
    dtheta = 2.0 * math.pi / ny
    dz = 2.0 * math.pi / nz

    def axis_grid(axis, n, step, periodic):
        idx = jnp.arange(-h, n + h, dtype=jnp.float64)
        centers = (idx + 0.5) * step
        if periodic:
            faces = jnp.arange(-h, n + h + 1, dtype=jnp.float64) * step
        else:
            faces = jnp.arange(-h, n + h + 1, dtype=jnp.float64) * step
        return LocalGrid1D(layout, axis, centers, faces, 0, n)

    grid = LocalCellCenteredGrid3D(
        layout=layout,
        x=axis_grid(0, nx, dr, False),
        y=axis_grid(1, ny, dtheta, True),
        z=axis_grid(2, nz, dz, True),
    )
    r, theta, z = jnp.meshgrid(grid.x.centers, grid.y.centers, grid.z.centers, indexing="ij")
    metric = _axis_metric(layout, "cell", r, theta, z)
    face_metric = LocalFaceMetricGeometry(
        layout=layout,
        x=_axis_metric(layout, "x_face", *jnp.meshgrid(grid.x.faces, grid.y.centers, grid.z.centers, indexing="ij")),
        y=_axis_metric(layout, "y_face", *jnp.meshgrid(grid.x.centers, grid.y.faces, grid.z.centers, indexing="ij")),
        z=_axis_metric(layout, "z_face", *jnp.meshgrid(grid.x.centers, grid.y.centers, grid.z.faces, indexing="ij")),
    )
    cell_b = _axis_bfield(metric)
    face_b = LocalFaceBFieldGeometry(
        layout=layout,
        x=_axis_bfield(face_metric.x), y=_axis_bfield(face_metric.y), z=_axis_bfield(face_metric.z),
    )
    spacing = LocalSpacing3D(
        layout=layout,
        dx_halo=jnp.full(layout.cell_halo_shape, dr),
        dy_halo=jnp.full(layout.cell_halo_shape, dtheta),
        dz_halo=jnp.full(layout.cell_halo_shape, dz),
    )
    regular = replace(
        base.regular_face_geometry,
        x_area=jnp.ones(layout.face_control_shape(0)),
        y_area=jnp.ones(layout.face_control_shape(1)),
        z_area=jnp.ones(layout.face_control_shape(2)),
    )
    geometry = LocalFciGeometry3D(
        layout=layout, grid=grid, maps=base.maps, spacing=spacing,
        cell_metric=metric, face_metric=face_metric, cell_bfield=cell_b,
        face_bfield=face_b, regular_face_geometry=regular,
        cell_volume_geometry=LocalCellVolumeGeometry3D(
            layout=layout, volume=metric.J_owned, volume_fraction=jnp.ones(shape)
        ),
    )
    domain = LocalDomain3D(
        shard_spec=ShardSpec3D(
            global_shape=shape, owned_start=(0, 0, 0), owned_stop=shape,
            shard_index=(0, 0, 0), shard_counts=(1, 1, 1),
            periodic_axes=(False, True, True), axis_regular_axes=(True, False, False),
            halo_width=h,
            side_kind_lower=(SIDE_AXIS_REGULAR, SIDE_SIMPLE_PERIODIC, SIDE_SIMPLE_PERIODIC),
            side_kind_upper=(SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_SIMPLE_PERIODIC),
        ), layout=layout, mesh_axis_names=(None, None, None)
    )
    context = StencilBuilderContext(layout=layout, domain=domain)
    scalar_filler = TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(), PolarAxisRegularScalarRule3D(angle_axis_name=None)))
    def vector_filler(diagonal):
        return TopologyHaloFiller3D(rules=(
            LocalPeriodicTopologyRule3D(),
            PolarAxisRegularVectorRule3D(
                axis=0, side="lower", angular_axis=1, mesh_axis_name=None,
                source_shard_offset=0, local_shift_cells=ny // 2,
                component_transform=jnp.diag(jnp.asarray(diagonal)),
            ),
        ))

    ordinary_vector_filler = vector_filler((-1.0, 1.0, 1.0))
    flux_density_filler = vector_filler((1.0, -1.0, -1.0))
    return (
        geometry, domain, context, (r, theta, z), HaloExchange3D(),
        scalar_filler, ordinary_vector_filler, flux_density_filler,
    )


def scalar_field_halo(r, theta, z, expression):
    """Evaluate a smooth scalar expression on the halo coordinate mesh."""
    return jnp.asarray(expression(r, theta, z), dtype=jnp.float64)


def analytic_gradient(r, theta, z, name):
    if name == "x":
        return jnp.stack((jnp.cos(theta), -r*jnp.sin(theta), jnp.zeros_like(r)), axis=-1)
    if name == "y":
        return jnp.stack((jnp.sin(theta), r*jnp.cos(theta), jnp.zeros_like(r)), axis=-1)
    if name == "r2":
        return jnp.stack((2.0 * r, jnp.zeros_like(r), jnp.zeros_like(r)), axis=-1)
    raise ValueError(name)


def owned(values, geometry):
    return values[geometry.layout.owned_slices_cell]


def gradient(value, geometry):
    shape = geometry.owned_shape
    return LocalCellGradient3D(
        gradient=value, valid=jnp.ones(shape, dtype=bool),
        reconstruction_mask=jnp.ones(shape, dtype=bool)
    )
