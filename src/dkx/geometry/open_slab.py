"""Open-field-line slab SOL flux-tube geometry.

This module provides :func:`build_open_slab_geometry`, a Cartesian scrape-off-layer
(SOL) flux tube with a straight magnetic field along the parallel coordinate
``z`` and **open** field lines: every field line terminates on a material target
plate at ``z = 0`` and ``z = L_parallel``. The endpoint masks
(``forward_boundary`` at the ``z = L`` target, ``backward_boundary`` at the
``z = 0`` target) are exactly the open-field-line target masks the sheath /
recycling closure (:mod:`dkx.native.fci_sheath_recycling`) consumes.

The logical coordinates are ``(x, y, z)`` with ``x`` the radial SOL coordinate,
``y`` the periodic binormal, and ``z`` the parallel (connection-length)
coordinate bounded by the two targets. The metric is Cartesian/identity — the
point of this geometry is the *open* parallel topology, not curvature — so the
FCI parallel gradient reduces to ``d/dz`` and the target plates carry the Bohm
sheath. It is the open-field-line counterpart to the closed flux tubes
(rotating ellipse, shifted torus) elsewhere in the package.
"""

from __future__ import annotations

import jax.numpy as jnp

from .fci_geometry import (
    BFieldGeometry,
    CellCenteredGrid3D,
    FaceBFieldGeometry,
    FaceMetricGeometry,
    FciGeometry3D,
    FciMaps3D,
    Grid1D,
    MetricGeometry,
    Spacing3D,
)

__all__ = ["build_open_slab_geometry"]


def _identity_metric(shape: tuple[int, int, int]) -> MetricGeometry:
    ones = jnp.ones(shape, dtype=jnp.float64)
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    return MetricGeometry(
        J=ones,
        g11=ones,
        g22=ones,
        g33=ones,
        g12=zeros,
        g13=zeros,
        g23=zeros,
        g_11=ones,
        g_22=ones,
        g_33=ones,
        g_12=zeros,
        g_13=zeros,
        g_23=zeros,
    )


def _parallel_bfield(shape: tuple[int, int, int]) -> BFieldGeometry:
    return BFieldGeometry(
        B_contra=jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64), shape + (3,)),
        Bmag=jnp.ones(shape, dtype=jnp.float64),
    )


def build_open_slab_geometry(
    shape: tuple[int, int, int],
    *,
    radial_extent: float = 1.0,
    binormal_extent: float = 1.0,
    parallel_length: float = 40.0,
) -> FciGeometry3D:
    """Build an open-field-line slab SOL flux tube of the requested ``shape``.

    Args:
        shape: ``(nx, ny, nz)`` cell-centered grid. ``nz`` is the number of cells
            along the field between the two targets; ``nx``/``ny`` are the radial
            and binormal extents (use small values for a flux-tube study).
        radial_extent, binormal_extent: physical sizes of the ``x`` and ``y``
            axes.
        parallel_length: connection length ``L_parallel`` between the ``z = 0``
            and ``z = L`` target plates.

    Returns:
        A :class:`FciGeometry3D` whose field lines are open: ``forward_boundary``
        is set on the ``z = L`` target plane and ``backward_boundary`` on the
        ``z = 0`` target plane, so :func:`dkx.native.fci_sheath_recycling`.
        `build_fci_target_masks` marks exactly the two target planes.
    """

    nx, ny, nz = shape
    x_faces = jnp.linspace(0.0, float(radial_extent), nx + 1, dtype=jnp.float64)
    y_faces = jnp.linspace(0.0, float(binormal_extent), ny + 1, dtype=jnp.float64)
    z_faces = jnp.linspace(0.0, float(parallel_length), nz + 1, dtype=jnp.float64)
    grid = CellCenteredGrid3D(
        x=Grid1D(centers=0.5 * (x_faces[:-1] + x_faces[1:]), faces=x_faces),
        y=Grid1D(centers=0.5 * (y_faces[:-1] + y_faces[1:]), faces=y_faces),
        z=Grid1D(centers=0.5 * (z_faces[:-1] + z_faces[1:]), faces=z_faces),
    )
    target_shape = (nx, ny, nz)
    zeros = jnp.zeros(target_shape, dtype=jnp.float64)
    ones = jnp.ones(target_shape, dtype=jnp.float64)

    cell_metric = _identity_metric(target_shape)
    face_metric = FaceMetricGeometry(
        x=_identity_metric((nx + 1, ny, nz)),
        y=_identity_metric((nx, ny + 1, nz)),
        z=_identity_metric((nx, ny, nz + 1)),
    )
    cell_bfield = _parallel_bfield(target_shape)
    face_bfield = FaceBFieldGeometry(
        x=_parallel_bfield((nx + 1, ny, nz)),
        y=_parallel_bfield((nx, ny + 1, nz)),
        z=_parallel_bfield((nx, ny, nz + 1)),
    )

    # Open field lines: the forward map exits the domain at the z = L target and
    # the backward map exits at the z = 0 target. Interior planes stay closed.
    forward_boundary = jnp.zeros(target_shape, dtype=bool).at[:, :, -1].set(True)
    backward_boundary = jnp.zeros(target_shape, dtype=bool).at[:, :, 0].set(True)
    maps = FciMaps3D(
        forward_x=zeros,
        forward_y=zeros,
        backward_x=zeros,
        backward_y=zeros,
        forward_endpoint_x=zeros,
        forward_endpoint_y=zeros,
        forward_endpoint_z=zeros,
        backward_endpoint_x=zeros,
        backward_endpoint_y=zeros,
        backward_endpoint_z=zeros,
        forward_length=ones,
        backward_length=ones,
        forward_boundary=forward_boundary,
        backward_boundary=backward_boundary,
    )
    spacing = Spacing3D(
        dx=jnp.broadcast_to(grid.x.widths[:, None, None], target_shape),
        dy=jnp.broadcast_to(grid.y.widths[None, :, None], target_shape),
        dz=jnp.broadcast_to(grid.z.widths[None, None, :], target_shape),
    )
    return FciGeometry3D(
        grid=grid,
        maps=maps,
        spacing=spacing,
        cell_metric=cell_metric,
        face_metric=face_metric,
        cell_bfield=cell_bfield,
        face_bfield=face_bfield,
    )
