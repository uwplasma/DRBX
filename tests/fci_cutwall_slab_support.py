"""Shared analytic and fixture support for the cut-wall slab tests.

This module intentionally has no ``test_`` prefix: pytest should only collect
the focused assertions in ``test_fci_cutwall_slab_operators``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from drbx.geometry import (
    HaloLayout3D,
    LocalBFieldGeometry,
    LocalCellCenteredGrid3D,
    LocalCellVolumeGeometry3D,
    LocalCoordinateStencilDependencyMap3D,
    LocalCoordinateStencilLocalDependencyTable,
    LocalDomain3D,
    LocalFaceBFieldGeometry,
    LocalFaceMetricGeometry,
    LocalFciDirectionMap,
    LocalFciGeometry3D,
    LocalFciLocalDependencyTable,
    LocalFciMaps3D,
    LocalGrid1D,
    LocalMetricGeometry,
    LocalRegularFaceGeometry3D,
    LocalSpacing3D,
    ShardSpec3D,
    SIDE_PHYSICAL,
    SIDE_SIMPLE_PERIODIC,
)
from drbx.native.fci_boundaries import (
    BC_DIRICHLET,
    LocalCutWallBC3D,
    LocalCutWallGeometry3D,
)

WALL_ALPHA = 0.25
WALL_C = 0.72
FIELD_EPS = 0.03
_MESH_AXIS_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class _SlabCase:
    geometry: LocalFciGeometry3D
    domain: LocalDomain3D
    field_halo: jnp.ndarray
    stencil_cut_wall_geometry: LocalCutWallGeometry3D
    stencil_cut_wall_values: jnp.ndarray
    flux_cut_wall_geometry: LocalCutWallGeometry3D
    cut_wall_bc: LocalCutWallBC3D
    regular_face_geometry: LocalRegularFaceGeometry3D
    owner_mask: jnp.ndarray
    valid_mask: jnp.ndarray
    interior_mask: jnp.ndarray


def _wall_x(y: np.ndarray | jnp.ndarray) -> np.ndarray | jnp.ndarray:
    return WALL_C - WALL_ALPHA * y


def _field_value(x, y, z):
    del z
    return (
        x * x
        + 0.5 * y * y
        + 0.1 * x * y
        + FIELD_EPS * jnp.sin(2.0 * jnp.pi * x) * jnp.cos(2.0 * jnp.pi * y)
    )


def _exact_grad_perp(x, y, z):
    del z
    dfdx = (
        2.0 * x
        + 0.1 * y
        + FIELD_EPS
        * 2.0
        * jnp.pi
        * jnp.cos(2.0 * jnp.pi * x)
        * jnp.cos(2.0 * jnp.pi * y)
    )
    dfdy = (
        y
        + 0.1 * x
        - FIELD_EPS
        * 2.0
        * jnp.pi
        * jnp.sin(2.0 * jnp.pi * x)
        * jnp.sin(2.0 * jnp.pi * y)
    )
    return jnp.stack((dfdx, dfdy, jnp.zeros_like(dfdx)), axis=-1)


def _linear_field_value(x, y, z):
    return 0.2 + 1.7 * x - 0.8 * y + 0.3 * z


def _linear_field_gradient() -> jnp.ndarray:
    return jnp.asarray((1.7, -0.8, 0.3), dtype=jnp.float64)


def _exact_perp_laplacian(x, y, z):
    del z
    return 3.0 - 2.0 * FIELD_EPS * (2.0 * jnp.pi) ** 2 * (
        jnp.sin(2.0 * jnp.pi * x) * jnp.cos(2.0 * jnp.pi * y)
    )


def _axis_grid(
    layout: HaloLayout3D,
    *,
    axis: int,
    local_n: int,
    global_n: int,
    shard_id: int | jax.Array = 0,
) -> LocalGrid1D:
    h = int(layout.halo_width)
    spacing = 1.0 / float(global_n)
    start = jnp.asarray(shard_id, dtype=jnp.int32) * int(local_n)
    center_indices = start + jnp.arange(-h, local_n + h, dtype=jnp.float64)
    face_indices = start + jnp.arange(-h, local_n + h + 1, dtype=jnp.float64)
    return LocalGrid1D(
        layout=layout,
        axis=axis,
        centers_halo=(center_indices + 0.5) * spacing,
        faces_halo=face_indices * spacing,
        owned_start_global=0,
        owned_stop_global=local_n,
    )


def _unit_metric(layout: HaloLayout3D, location: str) -> LocalMetricGeometry:
    shape = layout.location_halo_shape(location)
    ones = jnp.ones(shape, dtype=jnp.float64)
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    return LocalMetricGeometry(
        layout=layout,
        J_halo=ones,
        g11_halo=ones,
        g22_halo=ones,
        g33_halo=ones,
        g12_halo=zeros,
        g13_halo=zeros,
        g23_halo=zeros,
        g_11_halo=ones,
        g_22_halo=ones,
        g_33_halo=ones,
        g_12_halo=zeros,
        g_13_halo=zeros,
        g_23_halo=zeros,
        location=location,
    )


def _unit_z_bfield(layout: HaloLayout3D, location: str) -> LocalBFieldGeometry:
    shape = layout.location_halo_shape(location)
    b = jnp.broadcast_to(
        jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
        shape + (3,),
    )
    return LocalBFieldGeometry(
        layout=layout,
        B_contra_halo=b,
        Bmag_halo=jnp.ones(shape, dtype=jnp.float64),
        location=location,
    )


def _empty_fci_local_rows() -> LocalFciLocalDependencyTable:
    return LocalFciLocalDependencyTable(
        target_flat=jnp.zeros((0,), dtype=jnp.int32),
        source_i=jnp.zeros((0,), dtype=jnp.int32),
        source_j=jnp.zeros((0,), dtype=jnp.int32),
        source_k=jnp.zeros((0,), dtype=jnp.int32),
        weight=jnp.zeros((0,), dtype=jnp.float64),
        active=jnp.zeros((0,), dtype=bool),
    )


def _empty_maps(layout: HaloLayout3D) -> LocalFciMaps3D:
    direction = LocalFciDirectionMap(
        layout=layout,
        local=_empty_fci_local_rows(),
        connection_length=jnp.ones(layout.owned_shape, dtype=jnp.float64),
    )
    return LocalFciMaps3D(
        layout=layout,
        forward=direction,
        backward=direction,
        mode="local_halo_only",
    )


def _build_domain(
    shape: tuple[int, int, int],
    halo_width: int,
    *,
    global_shape: tuple[int, int, int] | None = None,
    shard_counts: tuple[int, int, int] = (1, 1, 1),
) -> LocalDomain3D:
    global_shape = tuple(shape if global_shape is None else global_shape)
    spec = ShardSpec3D(
        global_shape=global_shape,
        owned_start=(0, 0, 0),
        owned_stop=shape,
        shard_index=(0, 0, 0),
        shard_counts=tuple(int(value) for value in shard_counts),
        periodic_axes=(False, True, True),
        halo_width=halo_width,
        side_kind_lower=(SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_SIMPLE_PERIODIC),
        side_kind_upper=(SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_SIMPLE_PERIODIC),
    )
    return LocalDomain3D(
        shard_spec=spec,
        layout=HaloLayout3D(shape, halo_width),
        mesh_axis_names=("x", "y", "z"),
    )


def _build_geometry(
    shape: tuple[int, int, int],
    halo_width: int,
    *,
    global_shape: tuple[int, int, int] | None = None,
    shard_index: tuple[int | jax.Array, int | jax.Array, int | jax.Array] = (0, 0, 0),
) -> LocalFciGeometry3D:
    nx, ny, nz = shape
    global_nx, global_ny, global_nz = tuple(shape if global_shape is None else global_shape)
    layout = HaloLayout3D(shape, halo_width)
    grid = LocalCellCenteredGrid3D(
        layout=layout,
        x=_axis_grid(
            layout,
            axis=0,
            local_n=nx,
            global_n=global_nx,
            shard_id=shard_index[0],
        ),
        y=_axis_grid(
            layout,
            axis=1,
            local_n=ny,
            global_n=global_ny,
            shard_id=shard_index[1],
        ),
        z=_axis_grid(
            layout,
            axis=2,
            local_n=nz,
            global_n=global_nz,
            shard_id=shard_index[2],
        ),
    )
    spacing = LocalSpacing3D(
        layout=layout,
        dx_halo=jnp.full(layout.cell_halo_shape, 1.0 / global_nx, dtype=jnp.float64),
        dy_halo=jnp.full(layout.cell_halo_shape, 1.0 / global_ny, dtype=jnp.float64),
        dz_halo=jnp.full(layout.cell_halo_shape, 1.0 / global_nz, dtype=jnp.float64),
    )
    face_shapes = (
        layout.face_control_shape(0),
        layout.face_control_shape(1),
        layout.face_control_shape(2),
    )
    regular = LocalRegularFaceGeometry3D(
        layout=layout,
        x_area=jnp.ones(face_shapes[0], dtype=jnp.float64),
        y_area=jnp.ones(face_shapes[1], dtype=jnp.float64),
        z_area=jnp.ones(face_shapes[2], dtype=jnp.float64),
        x_area_fraction=jnp.ones(face_shapes[0], dtype=jnp.float64),
        y_area_fraction=jnp.ones(face_shapes[1], dtype=jnp.float64),
        z_area_fraction=jnp.ones(face_shapes[2], dtype=jnp.float64),
        x_open_mask=jnp.ones(face_shapes[0], dtype=bool),
        y_open_mask=jnp.ones(face_shapes[1], dtype=bool),
        z_open_mask=jnp.ones(face_shapes[2], dtype=bool),
    )
    return LocalFciGeometry3D(
        layout=layout,
        grid=grid,
        maps=_empty_maps(layout),
        spacing=spacing,
        cell_metric=_unit_metric(layout, "cell"),
        face_metric=LocalFaceMetricGeometry(
            layout=layout,
            x=_unit_metric(layout, "x_face"),
            y=_unit_metric(layout, "y_face"),
            z=_unit_metric(layout, "z_face"),
        ),
        cell_bfield=_unit_z_bfield(layout, "cell"),
        face_bfield=LocalFaceBFieldGeometry(
            layout=layout,
            x=_unit_z_bfield(layout, "x_face"),
            y=_unit_z_bfield(layout, "y_face"),
            z=_unit_z_bfield(layout, "z_face"),
        ),
        regular_face_geometry=regular,
        cell_volume_geometry=LocalCellVolumeGeometry3D(
            layout=layout,
            volume=jnp.ones(shape, dtype=jnp.float64),
            volume_fraction=jnp.ones(shape, dtype=jnp.float64),
        ),
    )


def _owned_coordinates(geometry: LocalFciGeometry3D):
    x = geometry.grid.x.centers_owned[:, None, None]
    y = geometry.grid.y.centers_owned[None, :, None]
    z = geometry.grid.z.centers_owned[None, None, :]
    return jnp.broadcast_arrays(x, y, z)


def _halo_coordinates(geometry: LocalFciGeometry3D):
    x = geometry.grid.x.centers_halo[:, None, None]
    y = geometry.grid.y.centers_halo[None, :, None]
    z = geometry.grid.z.centers_halo[None, None, :]
    return jnp.broadcast_arrays(x, y, z)


def _make_cut_wall_geometry(
    *,
    owner_i: list[int],
    owner_j: list[int],
    owner_k: list[int],
    centers: list[tuple[float, float, float]],
    distances: list[float],
    area_covector: np.ndarray,
    normal: np.ndarray,
    stencil_axis: list[int],
    stencil_side: list[int],
) -> LocalCutWallGeometry3D:
    max_wall_faces = len(owner_i)
    return LocalCutWallGeometry3D(
        owner_i=jnp.asarray(owner_i, dtype=jnp.int32),
        owner_j=jnp.asarray(owner_j, dtype=jnp.int32),
        owner_k=jnp.asarray(owner_k, dtype=jnp.int32),
        center=jnp.asarray(centers, dtype=jnp.float64),
        normal_contra=jnp.broadcast_to(jnp.asarray(normal), (max_wall_faces, 3)),
        area_covector=jnp.broadcast_to(
            jnp.asarray(area_covector),
            (max_wall_faces, 3),
        ),
        distance=jnp.asarray(distances, dtype=jnp.float64),
        J=jnp.ones((max_wall_faces,), dtype=jnp.float64),
        g_contra=jnp.broadcast_to(
            jnp.eye(3, dtype=jnp.float64),
            (max_wall_faces, 3, 3),
        ),
        g_cov=jnp.broadcast_to(
            jnp.eye(3, dtype=jnp.float64),
            (max_wall_faces, 3, 3),
        ),
        B_contra=jnp.broadcast_to(
            jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
            (max_wall_faces, 3),
        ),
        Bmag=jnp.ones((max_wall_faces,), dtype=jnp.float64),
        sign=jnp.ones((max_wall_faces,), dtype=jnp.float64),
        active=jnp.ones((max_wall_faces,), dtype=bool),
        max_wall_faces=max_wall_faces,
        stencil_axis=jnp.asarray(stencil_axis, dtype=jnp.int32),
        stencil_side=jnp.asarray(stencil_side, dtype=jnp.int32),
        stencil_distance=jnp.asarray(distances, dtype=jnp.float64),
    )


def _unchecked_local_cut_wall_geometry(
    *,
    owner_i: jnp.ndarray,
    owner_j: jnp.ndarray,
    owner_k: jnp.ndarray,
    center: jnp.ndarray,
    normal_contra: jnp.ndarray,
    area_covector: jnp.ndarray,
    distance: jnp.ndarray,
    sign: jnp.ndarray,
    active: jnp.ndarray,
    stencil_axis: jnp.ndarray,
    stencil_side: jnp.ndarray,
    stencil_distance: jnp.ndarray,
) -> LocalCutWallGeometry3D:
    max_wall_faces = int(owner_i.size)
    obj = object.__new__(LocalCutWallGeometry3D)
    object.__setattr__(obj, "owner_i", jnp.asarray(owner_i, dtype=jnp.int32))
    object.__setattr__(obj, "owner_j", jnp.asarray(owner_j, dtype=jnp.int32))
    object.__setattr__(obj, "owner_k", jnp.asarray(owner_k, dtype=jnp.int32))
    object.__setattr__(obj, "center", jnp.asarray(center, dtype=jnp.float64))
    object.__setattr__(obj, "normal_contra", jnp.asarray(normal_contra, dtype=jnp.float64))
    object.__setattr__(obj, "area_covector", jnp.asarray(area_covector, dtype=jnp.float64))
    object.__setattr__(obj, "distance", jnp.asarray(distance, dtype=jnp.float64))
    object.__setattr__(obj, "J", jnp.ones((max_wall_faces,), dtype=jnp.float64))
    object.__setattr__(
        obj,
        "g_contra",
        jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), (max_wall_faces, 3, 3)),
    )
    object.__setattr__(
        obj,
        "g_cov",
        jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), (max_wall_faces, 3, 3)),
    )
    object.__setattr__(
        obj,
        "B_contra",
        jnp.broadcast_to(
            jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
            (max_wall_faces, 3),
        ),
    )
    object.__setattr__(obj, "Bmag", jnp.ones((max_wall_faces,), dtype=jnp.float64))
    object.__setattr__(obj, "sign", jnp.asarray(sign, dtype=jnp.float64))
    object.__setattr__(obj, "active", jnp.asarray(active, dtype=bool))
    object.__setattr__(obj, "max_wall_faces", max_wall_faces)
    object.__setattr__(obj, "stencil_axis", jnp.asarray(stencil_axis, dtype=jnp.int32))
    object.__setattr__(obj, "stencil_side", jnp.asarray(stencil_side, dtype=jnp.int32))
    object.__setattr__(
        obj,
        "stencil_distance",
        jnp.asarray(stencil_distance, dtype=jnp.float64),
    )
    return obj


def _unchecked_coordinate_dependencies(
    layout: HaloLayout3D,
    cut_wall_geometry: LocalCutWallGeometry3D,
) -> LocalCoordinateStencilDependencyMap3D:
    nx, ny, nz = layout.owned_shape
    target_flat = (
        jnp.asarray(cut_wall_geometry.owner_i, dtype=jnp.int32) * ny
        + jnp.asarray(cut_wall_geometry.owner_j, dtype=jnp.int32)
    ) * nz + jnp.asarray(cut_wall_geometry.owner_k, dtype=jnp.int32)

    local = object.__new__(LocalCoordinateStencilLocalDependencyTable)
    object.__setattr__(local, "target_flat", target_flat)
    object.__setattr__(local, "axis", jnp.asarray(cut_wall_geometry.stencil_axis, dtype=jnp.int32))
    object.__setattr__(local, "side", jnp.asarray(cut_wall_geometry.stencil_side, dtype=jnp.int32))
    object.__setattr__(
        local,
        "value_slot",
        jnp.arange(int(cut_wall_geometry.max_wall_faces), dtype=jnp.int32),
    )
    object.__setattr__(
        local,
        "distance",
        jnp.asarray(cut_wall_geometry.stencil_distance, dtype=jnp.float64),
    )
    object.__setattr__(local, "active", jnp.asarray(cut_wall_geometry.active, dtype=bool))

    dependencies = object.__new__(LocalCoordinateStencilDependencyMap3D)
    object.__setattr__(dependencies, "layout", layout)
    object.__setattr__(dependencies, "local", local)
    object.__setattr__(dependencies, "remote", None)
    return dependencies


def _shard_start_from_centers(centers: jnp.ndarray, global_n: int) -> jnp.ndarray:
    return jnp.rint(jnp.asarray(centers[0]) * float(global_n) - 0.5).astype(jnp.int32)


def _active_wall_index(
    wall_coordinate: jnp.ndarray,
    *,
    global_n: int,
    local_start: jnp.ndarray,
    local_n: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    global_owner = jnp.floor(wall_coordinate * float(global_n) - 0.5).astype(jnp.int32)
    local_owner = global_owner - local_start
    active = (
        (wall_coordinate > 0.0)
        & (wall_coordinate < 1.0)
        & (local_owner >= 0)
        & (local_owner < int(local_n))
    )
    safe_local_owner = jnp.clip(local_owner, 0, int(local_n) - 1)
    return global_owner, safe_local_owner, active


def _build_sharded_cut_wall_fixture(
    geometry: LocalFciGeometry3D,
    *,
    global_shape: tuple[int, int, int],
) -> tuple[
    LocalCutWallGeometry3D,
    jnp.ndarray,
    LocalCutWallGeometry3D,
    LocalCutWallBC3D,
    jnp.ndarray,
    LocalRegularFaceGeometry3D,
]:
    nx, ny, nz = geometry.owned_shape
    global_nx, global_ny, global_nz = tuple(int(value) for value in global_shape)
    dx = 1.0 / float(global_nx)
    dy = 1.0 / float(global_ny)
    dz = 1.0 / float(global_nz)
    normal = jnp.array([1.0, WALL_ALPHA, 0.0], dtype=jnp.float64)
    normal = normal / jnp.linalg.norm(normal)
    area_covector = jnp.array([1.0, WALL_ALPHA, 0.0], dtype=jnp.float64) * dy * dz

    x_centers = jnp.asarray(geometry.grid.x.centers_owned, dtype=jnp.float64)
    y_centers = jnp.asarray(geometry.grid.y.centers_owned, dtype=jnp.float64)
    z_centers = jnp.asarray(geometry.grid.z.centers_owned, dtype=jnp.float64)
    x_start = _shard_start_from_centers(x_centers, global_nx)
    y_start = _shard_start_from_centers(y_centers, global_ny)

    y_grid, z_grid = jnp.meshgrid(y_centers, z_centers, indexing="ij")
    j_grid, k_grid = jnp.meshgrid(
        jnp.arange(ny, dtype=jnp.int32),
        jnp.arange(nz, dtype=jnp.int32),
        indexing="ij",
    )
    wall_x = _wall_x(y_grid)
    global_owner_i, owner_i, active_x = _active_wall_index(
        wall_x,
        global_n=global_nx,
        local_start=x_start,
        local_n=nx,
    )
    owner_x = (global_owner_i.astype(jnp.float64) + 0.5) * dx
    center_distance_x = jnp.maximum(wall_x - owner_x, 1.0e-30)
    normal_distance = jnp.maximum(
        (WALL_C - owner_x - WALL_ALPHA * y_grid)
        / jnp.sqrt(1.0 + WALL_ALPHA * WALL_ALPHA),
        1.0e-30,
    )
    flux_center = jnp.stack(
        (
            owner_x + normal_distance * normal[0],
            y_grid + normal_distance * normal[1],
            z_grid,
        ),
        axis=-1,
    )
    flux_size = ny * nz
    flux_cut_wall_geometry = _unchecked_local_cut_wall_geometry(
        owner_i=owner_i.reshape((flux_size,)),
        owner_j=j_grid.reshape((flux_size,)),
        owner_k=k_grid.reshape((flux_size,)),
        center=flux_center.reshape((flux_size, 3)),
        normal_contra=jnp.broadcast_to(normal, (flux_size, 3)),
        area_covector=jnp.broadcast_to(area_covector, (flux_size, 3)),
        distance=jnp.where(active_x, normal_distance, 1.0).reshape((flux_size,)),
        sign=jnp.ones((flux_size,), dtype=jnp.float64),
        active=active_x.reshape((flux_size,)),
        stencil_axis=jnp.zeros((flux_size,), dtype=jnp.int32),
        stencil_side=jnp.ones((flux_size,), dtype=jnp.int32),
        stencil_distance=jnp.where(active_x, center_distance_x, 1.0).reshape((flux_size,)),
    )
    flux_wall_values = _field_value(
        flux_center[..., 0],
        flux_center[..., 1],
        flux_center[..., 2],
    ).reshape((flux_size,))

    x_i_grid, z_for_y_grid = jnp.meshgrid(
        jnp.arange(nx, dtype=jnp.int32),
        jnp.arange(nz, dtype=jnp.int32),
        indexing="ij",
    )
    x_grid, z_y_grid = jnp.meshgrid(x_centers, z_centers, indexing="ij")
    wall_y = (WALL_C - x_grid) / WALL_ALPHA
    global_owner_j, owner_j, active_y = _active_wall_index(
        wall_y,
        global_n=global_ny,
        local_start=y_start,
        local_n=ny,
    )
    owner_y = (global_owner_j.astype(jnp.float64) + 0.5) * dy
    center_distance_y = jnp.maximum(wall_y - owner_y, 1.0e-30)
    active_y = active_y & (x_grid < _wall_x(owner_y))
    y_size = nx * nz

    x_stencil_center = jnp.stack((wall_x, y_grid, z_grid), axis=-1)
    y_stencil_center = jnp.stack((x_grid, wall_y, z_y_grid), axis=-1)
    stencil_cut_wall_geometry = _unchecked_local_cut_wall_geometry(
        owner_i=jnp.concatenate(
            (owner_i.reshape((flux_size,)), x_i_grid.reshape((y_size,))),
            axis=0,
        ),
        owner_j=jnp.concatenate(
            (j_grid.reshape((flux_size,)), owner_j.reshape((y_size,))),
            axis=0,
        ),
        owner_k=jnp.concatenate(
            (k_grid.reshape((flux_size,)), z_for_y_grid.reshape((y_size,))),
            axis=0,
        ),
        center=jnp.concatenate(
            (
                x_stencil_center.reshape((flux_size, 3)),
                y_stencil_center.reshape((y_size, 3)),
            ),
            axis=0,
        ),
        normal_contra=jnp.broadcast_to(normal, (flux_size + y_size, 3)),
        area_covector=jnp.broadcast_to(area_covector, (flux_size + y_size, 3)),
        distance=jnp.concatenate(
            (
                jnp.where(active_x, center_distance_x, 1.0).reshape((flux_size,)),
                jnp.where(active_y, center_distance_y, 1.0).reshape((y_size,)),
            ),
            axis=0,
        ),
        sign=jnp.ones((flux_size + y_size,), dtype=jnp.float64),
        active=jnp.concatenate(
            (active_x.reshape((flux_size,)), active_y.reshape((y_size,))),
            axis=0,
        ),
        stencil_axis=jnp.concatenate(
            (
                jnp.zeros((flux_size,), dtype=jnp.int32),
                jnp.ones((y_size,), dtype=jnp.int32),
            ),
            axis=0,
        ),
        stencil_side=jnp.ones((flux_size + y_size,), dtype=jnp.int32),
        stencil_distance=jnp.concatenate(
            (
                jnp.where(active_x, center_distance_x, 1.0).reshape((flux_size,)),
                jnp.where(active_y, center_distance_y, 1.0).reshape((y_size,)),
            ),
            axis=0,
        ),
    )
    stencil_cut_wall_values = _field_value(
        stencil_cut_wall_geometry.center[:, 0],
        stencil_cut_wall_geometry.center[:, 1],
        stencil_cut_wall_geometry.center[:, 2],
    )
    cut_wall_bc = LocalCutWallBC3D(
        kind=jnp.full((flux_size,), BC_DIRICHLET, dtype=jnp.int32),
        value=jnp.where(flux_cut_wall_geometry.active, flux_wall_values, 0.0),
        active=flux_cut_wall_geometry.active,
        max_wall_faces=flux_size,
    )

    x_face_i = jnp.arange(nx + 1, dtype=jnp.int32)[:, None, None]
    owner_face = owner_i[None, :, :] + 1
    x_open_mask = ~((x_face_i == owner_face) & active_x[None, :, :])
    cell_i = jnp.arange(nx, dtype=jnp.int32)[:, None, None]
    owner_mask = (cell_i == owner_i[None, :, :]) & active_x[None, :, :]

    regular = LocalRegularFaceGeometry3D(
        layout=geometry.layout,
        x_area=geometry.regular_face_geometry.x_area,
        y_area=geometry.regular_face_geometry.y_area,
        z_area=geometry.regular_face_geometry.z_area,
        x_area_fraction=geometry.regular_face_geometry.x_area_fraction,
        y_area_fraction=geometry.regular_face_geometry.y_area_fraction,
        z_area_fraction=geometry.regular_face_geometry.z_area_fraction,
        x_open_mask=x_open_mask,
        y_open_mask=geometry.regular_face_geometry.y_open_mask,
        z_open_mask=geometry.regular_face_geometry.z_open_mask,
    )
    return (
        stencil_cut_wall_geometry,
        stencil_cut_wall_values,
        flux_cut_wall_geometry,
        cut_wall_bc,
        owner_mask,
        regular,
    )


def _build_cut_wall_fixture(geometry: LocalFciGeometry3D) -> tuple[
    LocalCutWallGeometry3D,
    jnp.ndarray,
    LocalCutWallGeometry3D,
    LocalCutWallBC3D,
    jnp.ndarray,
    LocalRegularFaceGeometry3D,
]:
    nx, ny, nz = geometry.owned_shape
    x_centers = np.asarray(geometry.grid.x.centers_owned)
    y_centers = np.asarray(geometry.grid.y.centers_owned)
    z_centers = np.asarray(geometry.grid.z.centers_owned)
    dy = 1.0 / float(ny)
    dz = 1.0 / float(nz)

    flux_owner_i: list[int] = []
    flux_owner_j: list[int] = []
    flux_owner_k: list[int] = []
    flux_centers: list[tuple[float, float, float]] = []
    flux_distances: list[float] = []
    flux_wall_values: list[float] = []
    stencil_owner_i: list[int] = []
    stencil_owner_j: list[int] = []
    stencil_owner_k: list[int] = []
    stencil_centers: list[tuple[float, float, float]] = []
    stencil_distances: list[float] = []
    stencil_values: list[float] = []
    stencil_axis: list[int] = []
    stencil_side: list[int] = []
    owner_mask = np.zeros((nx, ny, nz), dtype=bool)
    x_open_mask = np.ones(geometry.layout.face_control_shape(0), dtype=bool)

    for j, y in enumerate(y_centers):
        wall_x = float(_wall_x(y))
        i = int(np.searchsorted(x_centers, wall_x, side="right") - 1)
        i = max(1, min(nx - 2, i))
        distance = wall_x - float(x_centers[i])
        if distance <= 0.0:
            continue
        for k, z in enumerate(z_centers):
            flux_owner_i.append(i)
            flux_owner_j.append(j)
            flux_owner_k.append(k)
            owner_mask[i, j, k] = True
            x_open_mask[i + 1, j, k] = False
            stencil_owner_i.append(i)
            stencil_owner_j.append(j)
            stencil_owner_k.append(k)
            stencil_centers.append((wall_x, float(y), float(z)))
            stencil_distances.append(distance)
            stencil_values.append(float(_field_value(wall_x, y, z)))
            stencil_axis.append(0)
            stencil_side.append(1)

    for i, x in enumerate(x_centers):
        wall_y = float((WALL_C - x) / WALL_ALPHA)
        if wall_y <= 0.0 or wall_y >= 1.0:
            continue
        j = int(np.searchsorted(y_centers, wall_y, side="right") - 1)
        j = max(1, min(ny - 2, j))
        distance = wall_y - float(y_centers[j])
        if distance <= 0.0:
            continue
        if not (float(x) < float(_wall_x(y_centers[j]))):
            continue
        for k, z in enumerate(z_centers):
            stencil_owner_i.append(i)
            stencil_owner_j.append(j)
            stencil_owner_k.append(k)
            stencil_centers.append((float(x), wall_y, float(z)))
            stencil_distances.append(distance)
            stencil_values.append(float(_field_value(x, wall_y, z)))
            stencil_axis.append(1)
            stencil_side.append(1)

    normal = np.array([1.0, WALL_ALPHA, 0.0], dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    area_covector = np.array([1.0, WALL_ALPHA, 0.0], dtype=np.float64) * dy * dz

    flux_centers = []
    flux_distances = []
    flux_wall_values = []
    for i, j, k in zip(flux_owner_i, flux_owner_j, flux_owner_k):
        x_owner = float(x_centers[i])
        y_owner = float(y_centers[j])
        z_owner = float(z_centers[k])
        normal_distance = (
            WALL_C - x_owner - WALL_ALPHA * y_owner
        ) / math.sqrt(1.0 + WALL_ALPHA * WALL_ALPHA)
        wall_center = (
            x_owner + normal_distance * normal[0],
            y_owner + normal_distance * normal[1],
            z_owner,
        )
        flux_centers.append(wall_center)
        flux_distances.append(normal_distance)
        flux_wall_values.append(float(_field_value(*wall_center)))

    flux_cut_wall_geometry = _make_cut_wall_geometry(
        owner_i=flux_owner_i,
        owner_j=flux_owner_j,
        owner_k=flux_owner_k,
        centers=flux_centers,
        distances=flux_distances,
        area_covector=area_covector,
        normal=normal,
        stencil_axis=[0] * len(flux_owner_i),
        stencil_side=[1] * len(flux_owner_i),
    )
    stencil_cut_wall_geometry = _make_cut_wall_geometry(
        owner_i=stencil_owner_i,
        owner_j=stencil_owner_j,
        owner_k=stencil_owner_k,
        centers=stencil_centers,
        distances=stencil_distances,
        area_covector=area_covector,
        normal=normal,
        stencil_axis=stencil_axis,
        stencil_side=stencil_side,
    )
    flux_cut_wall_values = jnp.asarray(flux_wall_values, dtype=jnp.float64)
    stencil_cut_wall_values = jnp.asarray(stencil_values, dtype=jnp.float64)
    cut_wall_bc = LocalCutWallBC3D(
        kind=jnp.full((len(flux_owner_i),), BC_DIRICHLET, dtype=jnp.int32),
        value=flux_cut_wall_values,
        active=jnp.ones((len(flux_owner_i),), dtype=bool),
        max_wall_faces=len(flux_owner_i),
    )
    regular = LocalRegularFaceGeometry3D(
        layout=geometry.layout,
        x_area=geometry.regular_face_geometry.x_area,
        y_area=geometry.regular_face_geometry.y_area,
        z_area=geometry.regular_face_geometry.z_area,
        x_area_fraction=geometry.regular_face_geometry.x_area_fraction,
        y_area_fraction=geometry.regular_face_geometry.y_area_fraction,
        z_area_fraction=geometry.regular_face_geometry.z_area_fraction,
        x_open_mask=jnp.asarray(x_open_mask, dtype=bool),
        y_open_mask=geometry.regular_face_geometry.y_open_mask,
        z_open_mask=geometry.regular_face_geometry.z_open_mask,
    )
    return (
        stencil_cut_wall_geometry,
        stencil_cut_wall_values,
        flux_cut_wall_geometry,
        cut_wall_bc,
        jnp.asarray(owner_mask, dtype=bool),
        regular,
    )


def _build_case(shape: tuple[int, int, int]) -> _SlabCase:
    halo_width = 1
    geometry = _build_geometry(shape, halo_width)
    domain = _build_domain(shape, halo_width)
    x_halo, y_halo, z_halo = _halo_coordinates(geometry)
    field_halo = _field_value(x_halo, y_halo, z_halo)
    (
        stencil_cut_wall_geometry,
        stencil_cut_wall_values,
        flux_cut_wall_geometry,
        cut_wall_bc,
        owner_mask,
        regular_face_geometry,
    ) = _build_cut_wall_fixture(geometry)

    x, y, z = _owned_coordinates(geometry)
    del z
    inside = x < _wall_x(y)
    i = jnp.arange(shape[0])[:, None, None]
    j = jnp.arange(shape[1])[None, :, None]
    k = jnp.arange(shape[2])[None, None, :]
    margin = (
        (i > 0)
        & (i < shape[0] - 1)
        & (j > 0)
        & (j < shape[1] - 1)
        & (k > 0)
        & (k < shape[2] - 1)
    )
    valid_mask = inside & margin
    interior_mask = valid_mask & (~owner_mask)
    return _SlabCase(
        geometry=geometry,
        domain=domain,
        field_halo=field_halo,
        stencil_cut_wall_geometry=stencil_cut_wall_geometry,
        stencil_cut_wall_values=stencil_cut_wall_values,
        flux_cut_wall_geometry=flux_cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
        regular_face_geometry=regular_face_geometry,
        owner_mask=owner_mask,
        valid_mask=valid_mask,
        interior_mask=interior_mask,
    )


def _build_sharded_case(
    global_shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int],
) -> _SlabCase:
    halo_width = 1
    local_shape = tuple(
        int(size) // int(count)
        for size, count in zip(global_shape, shard_counts)
    )
    shard_index = tuple(lax.axis_index(name) for name in _MESH_AXIS_NAMES)
    geometry = _build_geometry(
        local_shape,
        halo_width,
        global_shape=global_shape,
        shard_index=shard_index,
    )
    domain = _build_domain(
        local_shape,
        halo_width,
        global_shape=global_shape,
        shard_counts=shard_counts,
    )
    x_halo, y_halo, z_halo = _halo_coordinates(geometry)
    field_halo = _field_value(x_halo, y_halo, z_halo)
    (
        stencil_cut_wall_geometry,
        stencil_cut_wall_values,
        flux_cut_wall_geometry,
        cut_wall_bc,
        owner_mask,
        regular_face_geometry,
    ) = _build_sharded_cut_wall_fixture(geometry, global_shape=global_shape)

    x, y, z = _owned_coordinates(geometry)
    del z
    inside = x < _wall_x(y)
    global_nx, global_ny, global_nz = tuple(int(value) for value in global_shape)
    x_start = _shard_start_from_centers(geometry.grid.x.centers_owned, global_nx)
    y_start = _shard_start_from_centers(geometry.grid.y.centers_owned, global_ny)
    z_start = _shard_start_from_centers(geometry.grid.z.centers_owned, global_nz)
    i = x_start + jnp.arange(local_shape[0], dtype=jnp.int32)[:, None, None]
    j = y_start + jnp.arange(local_shape[1], dtype=jnp.int32)[None, :, None]
    k = z_start + jnp.arange(local_shape[2], dtype=jnp.int32)[None, None, :]
    margin = (
        (i > 0)
        & (i < global_nx - 1)
        & (j > 0)
        & (j < global_ny - 1)
        & (k > 0)
        & (k < global_nz - 1)
    )
    valid_mask = inside & margin
    interior_mask = valid_mask & (~owner_mask)
    return _SlabCase(
        geometry=geometry,
        domain=domain,
        field_halo=field_halo,
        stencil_cut_wall_geometry=stencil_cut_wall_geometry,
        stencil_cut_wall_values=stencil_cut_wall_values,
        flux_cut_wall_geometry=flux_cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
        regular_face_geometry=regular_face_geometry,
        owner_mask=owner_mask,
        valid_mask=valid_mask,
        interior_mask=interior_mask,
    )
