"""Cut-wall operator tests on a Cartesian slab with an oblique embedded wall."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import math
import time

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from drbx.geometry import (
    HaloLayout3D,
    LocalBFieldGeometry,
    LocalCellCenteredGrid3D,
    LocalControlVolumeCellGeometry3D,
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
    StencilBuilderContext,
    build_conservative_stencil_from_field,
    build_local_control_volume_cell_geometry,
    build_local_coordinate_stencil_dependency_map_from_cut_wall_geometry,
    build_local_stencil_from_field,
)
from drbx.native.fci_model import inject_owned_field_to_halo
from drbx.native.fci_boundaries import (
    BC_DIRICHLET,
    CV_FACE_CUT_WALL,
    CV_FACE_INTERIOR,
    CV_FACE_PARTIAL,
    CV_FACE_PHYSICAL_BOUNDARY,
    CV_RECONSTRUCTION_EQUATION_DIRICHLET,
    LocalControlVolumeBoundaryBC3D,
    LocalControlVolumeFaceRows3D,
    LocalControlVolumePolynomial3D,
    LocalBoundaryFaceBC3D,
    LocalEmbeddedControlVolumeGeometry3D,
    LocalMomentReconstruction3D,
    LocalRegularBoundaryMomentClosure3D,
    LocalCutWallBC3D,
    LocalCutWallGeometry3D,
    LocalStencil1D,
)
from drbx.native.fci_halo import (
    GhostFillWeights1D,
    LocalHaloClosure3D,
    LocalPeriodicTopologyRule3D,
    PhysicalGhostCellFiller3D,
    TopologyHaloFiller3D,
)
from drbx.native.fci_operators import (
    build_local_control_volume_polynomial_from_field,
    build_local_perp_laplacian_stencil,
    evaluate_local_control_volume_polynomial,
    expand_local_control_volume_owner_field,
    local_control_volume_product_average,
    local_grad_perp_op_direct,
    local_parallel_flux_div_op,
    local_perp_laplacian_conservative_op,
)
from drbx.native.fci_control_volume_operators import (
    precompute_local_moment_reconstruction,
)


jax.config.update("jax_enable_x64", True)


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


def _masked_l2(error: jnp.ndarray, mask: jnp.ndarray) -> float:
    values = jnp.asarray(error)[jnp.asarray(mask, dtype=bool)]
    return float(jnp.sqrt(jnp.mean(values * values)))


def _masked_sumsq_and_count(error: jnp.ndarray, mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    error = jnp.asarray(error, dtype=jnp.float64)
    mask = jnp.asarray(mask, dtype=bool)
    return (
        jnp.sum(jnp.where(mask, error * error, 0.0)),
        jnp.sum(mask).astype(jnp.float64),
    )


def _local_grad_error_stats(
    case: _SlabCase,
    *,
    unchecked_dependencies: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if unchecked_dependencies:
        dependencies = _unchecked_coordinate_dependencies(
            case.geometry.layout,
            case.stencil_cut_wall_geometry,
        )
    else:
        dependencies = build_local_coordinate_stencil_dependency_map_from_cut_wall_geometry(
            case.geometry.layout,
            case.stencil_cut_wall_geometry,
        )
    stencil = build_local_stencil_from_field(
        case.field_halo,
        case.geometry,
        StencilBuilderContext(
            layout=case.geometry.layout,
            cut_wall_stencil_dependencies=dependencies,
            cut_wall_values=case.stencil_cut_wall_values,
        ),
    )
    actual = local_grad_perp_op_direct(stencil, case.geometry)
    expected = _exact_grad_perp(*_owned_coordinates(case.geometry))
    return _masked_sumsq_and_count(
        jnp.linalg.norm(actual - expected, axis=-1),
        case.valid_mask,
    )


def _local_grad_error(shape: tuple[int, int, int]) -> float:
    sumsq, count = _local_grad_error_stats(_build_case(shape))
    return float(jnp.sqrt(sumsq / count))


def _conservative_laplacian_error_stats(
    case: _SlabCase,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    context = StencilBuilderContext(layout=case.geometry.layout, domain=case.domain)
    stencil = build_conservative_stencil_from_field(
        case.field_halo,
        case.geometry,
        context,
    )
    actual = local_perp_laplacian_conservative_op(
        stencil,
        case.geometry,
        case.domain,
        regular_face_geometry=case.regular_face_geometry,
        cell_volume=case.geometry.cell_volume_geometry,
        cut_wall_geometry=case.flux_cut_wall_geometry,
        cut_wall_bc=case.cut_wall_bc,
    )
    expected = _exact_perp_laplacian(*_owned_coordinates(case.geometry))
    sumsq, count = _masked_sumsq_and_count(actual - expected, case.interior_mask)
    cv_flux = build_local_perp_laplacian_stencil(
        stencil,
        case.geometry,
        case.domain,
        regular_face_geometry=case.regular_face_geometry,
        cell_volume=case.geometry.cell_volume_geometry,
        cut_wall_geometry=case.flux_cut_wall_geometry,
        cut_wall_bc=case.cut_wall_bc,
    )
    cut_wall_delta = jnp.zeros(case.geometry.owned_shape, dtype=jnp.float64).at[
        case.flux_cut_wall_geometry.owner_i,
        case.flux_cut_wall_geometry.owner_j,
        case.flux_cut_wall_geometry.owner_k,
    ].add(cv_flux.cut_wall_flux)
    outside_linf = jnp.max(jnp.abs(jnp.where(case.owner_mask, 0.0, cut_wall_delta)))
    owner_linf = jnp.max(jnp.abs(jnp.where(case.owner_mask, cut_wall_delta, 0.0)))
    return sumsq, count, owner_linf, outside_linf


def _conservative_laplacian_error(shape: tuple[int, int, int]) -> tuple[float, float, float]:
    sumsq, count, owner_linf, outside_linf = _conservative_laplacian_error_stats(
        _build_case(shape)
    )
    interior_l2 = float(jnp.sqrt(sumsq / count))
    owner_linf = float(owner_linf)
    outside_linf = float(outside_linf)
    return interior_l2, owner_linf, outside_linf


def test_oblique_cutwall_geometry_has_active_non_axis_aligned_faces() -> None:
    case = _build_case((8, 12, 8))

    assert case.flux_cut_wall_geometry.max_wall_faces > 0
    assert case.stencil_cut_wall_geometry.max_wall_faces > case.flux_cut_wall_geometry.max_wall_faces
    assert bool(jnp.all(case.flux_cut_wall_geometry.active))
    assert bool(jnp.all(case.stencil_cut_wall_geometry.active))
    assert bool(jnp.any(case.stencil_cut_wall_geometry.stencil_axis == 0))
    assert bool(jnp.any(case.stencil_cut_wall_geometry.stencil_axis == 1))
    assert float(jnp.max(jnp.abs(case.flux_cut_wall_geometry.normal_contra[:, 1]))) > 0.0
    assert int(jnp.sum(case.owner_mask)) == case.flux_cut_wall_geometry.max_wall_faces


def test_local_stencil_replace_recomputes_weights_when_cutwall_distance_changes() -> None:
    stencil = LocalStencil1D(
        center=jnp.asarray([[[0.0]]], dtype=jnp.float64),
        minus=jnp.asarray([[[-1.0]]], dtype=jnp.float64),
        plus=jnp.asarray([[[0.25]]], dtype=jnp.float64),
        dx_min=jnp.asarray([[[1.0]]], dtype=jnp.float64),
        dx_plus=jnp.asarray([[[1.0]]], dtype=jnp.float64),
    )

    patched = stencil.replace(dx_plus=jnp.asarray([[[0.25]]], dtype=jnp.float64))

    assert not jnp.allclose(
        patched.derivative_plus_weight,
        stencil.derivative_plus_weight,
    )
    np.testing.assert_allclose(
        np.asarray(
            patched.derivative_minus_weight * patched.minus
            + patched.derivative_center_weight * patched.center
            + patched.derivative_plus_weight * patched.plus
        ),
        np.asarray([[[1.0]]], dtype=np.float64),
        atol=1.0e-12,
    )



def _uniform_control_volume_cells(
    geometry: LocalFciGeometry3D,
    *,
    merged_source: tuple[int, int, int] | None = None,
    merge_target: tuple[int, int, int] | None = None,
) -> LocalControlVolumeCellGeometry3D:
    shape = geometry.owned_shape
    x, y, z = _owned_coordinates(geometry)
    centroid = jnp.stack(jnp.broadcast_arrays(x, y, z), axis=-1)
    dx = jnp.asarray(geometry.spacing.dx_owned, dtype=jnp.float64)
    dy = jnp.asarray(geometry.spacing.dy_owned, dtype=jnp.float64)
    dz = jnp.asarray(geometry.spacing.dz_owned, dtype=jnp.float64)
    volume = dx * dy * dz
    second_moment = jnp.zeros(shape + (3, 3), dtype=jnp.float64)
    second_moment = second_moment.at[..., 0, 0].set(dx * dx / 12.0)
    second_moment = second_moment.at[..., 1, 1].set(dy * dy / 12.0)
    second_moment = second_moment.at[..., 2, 2].set(dz * dz / 12.0)
    source_active = jnp.zeros(shape, dtype=bool)
    i, j, k = jnp.meshgrid(
        jnp.arange(shape[0], dtype=jnp.int32),
        jnp.arange(shape[1], dtype=jnp.int32),
        jnp.arange(shape[2], dtype=jnp.int32),
        indexing="ij",
    )
    target_i, target_j, target_k = i, j, k
    if merged_source is not None:
        if merge_target is None:
            raise ValueError("merge_target is required with merged_source")
        source_active = source_active.at[merged_source].set(True)
        target_i = target_i.at[merged_source].set(int(merge_target[0]))
        target_j = target_j.at[merged_source].set(int(merge_target[1]))
        target_k = target_k.at[merged_source].set(int(merge_target[2]))
    return build_local_control_volume_cell_geometry(
        geometry.layout,
        raw_volume=volume,
        raw_centroid=centroid,
        raw_second_moment=second_moment,
        source_active=source_active,
        target_i=target_i,
        target_j=target_j,
        target_k=target_k,
        retained_active=jnp.ones(shape, dtype=bool),
    )


def _unit_control_volume_face_rows(
    geometry: LocalFciGeometry3D,
    row_specs: tuple[
        tuple[
            int,
            tuple[int, int, int],
            tuple[int, int, int] | None,
            int,
            tuple[float, float, float],
            float,
        ],
        ...,
    ],
) -> LocalControlVolumeFaceRows3D:
    """Build compact Cartesian 2x2 face quadrature for focused CLI checks."""

    max_rows = len(row_specs)
    max_patches = 4
    qshape = (max_rows, max_patches, 4)
    kind = np.zeros((max_rows,), dtype=np.int32)
    minus = np.zeros((max_rows, 3), dtype=np.int32)
    plus = np.zeros((max_rows, 3), dtype=np.int32)
    has_plus = np.zeros((max_rows,), dtype=bool)
    points = np.zeros(qshape + (3,), dtype=np.float64)
    area = np.zeros_like(points)
    J = np.ones(qshape, dtype=np.float64)
    identity = np.broadcast_to(
        np.eye(3, dtype=np.float64),
        qshape + (3, 3),
    ).copy()
    b_contra = np.zeros(qshape + (3,), dtype=np.float64)
    b_contra[..., 2] = 1.0
    Bmag = np.ones(qshape, dtype=np.float64)
    projector = identity.copy()
    projector[..., 2, 2] = 0.0
    patch_active = np.zeros((max_rows, max_patches), dtype=bool)
    spacing = np.asarray(
        (
            float(np.asarray(geometry.spacing.dx_owned).flat[0]),
            float(np.asarray(geometry.spacing.dy_owned).flat[0]),
            float(np.asarray(geometry.spacing.dz_owned).flat[0]),
        ),
        dtype=np.float64,
    )
    gauss = 1.0 / np.sqrt(3.0)

    for row, (
        row_kind,
        minus_owner,
        plus_owner,
        axis,
        center,
        orientation,
    ) in enumerate(row_specs):
        kind[row] = int(row_kind)
        minus[row] = minus_owner
        if plus_owner is not None:
            has_plus[row] = True
            plus[row] = plus_owner
        tangential = [candidate for candidate in range(3) if candidate != axis]
        q = 0
        for node_a in (-gauss, gauss):
            for node_b in (-gauss, gauss):
                point = np.asarray(center, dtype=np.float64).copy()
                point[tangential[0]] += 0.5 * spacing[tangential[0]] * node_a
                point[tangential[1]] += 0.5 * spacing[tangential[1]] * node_b
                points[row, 0, q] = point
                area[row, 0, q, axis] = (
                    float(orientation)
                    * spacing[tangential[0]]
                    * spacing[tangential[1]]
                    / 4.0
                )
                q += 1
        patch_active[row, 0] = True

    return LocalControlVolumeFaceRows3D(
        layout=geometry.layout,
        kind=jnp.asarray(kind),
        minus_owner_i=jnp.asarray(minus[:, 0]),
        minus_owner_j=jnp.asarray(minus[:, 1]),
        minus_owner_k=jnp.asarray(minus[:, 2]),
        plus_owner_i=jnp.asarray(plus[:, 0]),
        plus_owner_j=jnp.asarray(plus[:, 1]),
        plus_owner_k=jnp.asarray(plus[:, 2]),
        has_plus_owner=jnp.asarray(has_plus),
        quadrature_points=jnp.asarray(points),
        area_covector_weight=jnp.asarray(area),
        J=jnp.asarray(J),
        g_contra=jnp.asarray(identity),
        g_cov=jnp.asarray(identity),
        B_contra=jnp.asarray(b_contra),
        Bmag=jnp.asarray(Bmag),
        projector=jnp.asarray(projector),
        patch_active=jnp.asarray(patch_active),
        active=jnp.ones((max_rows,), dtype=bool),
        max_rows=max_rows,
        max_patches=max_patches,
    )


def test_control_volume_face_rows_activate_all_gauss_points() -> None:
    geometry = _build_geometry((5, 5, 5), 2)
    rows = _unit_control_volume_face_rows(
        geometry,
        (
            (
                CV_FACE_INTERIOR,
                (2, 2, 2),
                (3, 2, 2),
                0,
                (0.5, 0.5, 0.5),
                1.0,
            ),
        ),
    )

    assert rows.quadrature_active.shape == (1, 4, 4)
    np.testing.assert_array_equal(
        np.asarray(rows.quadrature_active[0, 0]),
        np.ones((4,), dtype=bool),
    )
    np.testing.assert_array_equal(
        np.asarray(rows.quadrature_active[0, 1:]),
        np.zeros((3, 4), dtype=bool),
    )


def _all_closed_regular_faces(
    geometry: LocalFciGeometry3D,
) -> LocalRegularFaceGeometry3D:
    regular = geometry.regular_face_geometry
    return replace(
        regular,
        x_open_mask=jnp.zeros_like(regular.x_open_mask, dtype=bool),
        y_open_mask=jnp.zeros_like(regular.y_open_mask, dtype=bool),
        z_open_mask=jnp.zeros_like(regular.z_open_mask, dtype=bool),
    )


def _quadratic_point_value(
    points: jnp.ndarray,
    *,
    constant: float,
    gradient: jnp.ndarray,
    hessian: jnp.ndarray,
) -> jnp.ndarray:
    points = jnp.asarray(points, dtype=jnp.float64)
    return (
        float(constant)
        + jnp.einsum("...i,i->...", points, gradient)
        + 0.5 * jnp.einsum("...i,ij,...j->...", points, hessian, points)
    )


def _quadratic_cell_average_halo(
    geometry: LocalFciGeometry3D,
    *,
    constant: float,
    gradient: jnp.ndarray,
    hessian: jnp.ndarray,
) -> jnp.ndarray:
    x, y, z = _halo_coordinates(geometry)
    points = jnp.stack(jnp.broadcast_arrays(x, y, z), axis=-1)
    dx = float(np.asarray(geometry.spacing.dx_owned).flat[0])
    dy = float(np.asarray(geometry.spacing.dy_owned).flat[0])
    dz = float(np.asarray(geometry.spacing.dz_owned).flat[0])
    moment = jnp.diag(
        jnp.asarray((dx * dx, dy * dy, dz * dz), dtype=jnp.float64)
        / 12.0
    )
    return _quadratic_point_value(
        points,
        constant=constant,
        gradient=gradient,
        hessian=hessian,
    ) + 0.5 * jnp.einsum("ij,ij->", hessian, moment)


def test_control_volume_aggregate_moments_are_conservative() -> None:
    geometry = _build_geometry((7, 7, 7), 2)
    source = (3, 3, 3)
    target = (2, 3, 3)
    identity = _uniform_control_volume_cells(geometry)
    merged = _uniform_control_volume_cells(
        geometry,
        merged_source=source,
        merge_target=target,
    )
    raw_volume = float(identity.raw_volume[target])
    spacing = float(geometry.spacing.dx_owned[target])

    assert bool(merged.is_merged_source[source])
    assert bool(merged.is_aggregate_target[target])
    assert int(merged.received_source_count[target]) == 1
    assert int(merged.member_count[target]) == 2
    np.testing.assert_allclose(
        float(merged.aggregate_volume[target]),
        2.0 * raw_volume,
        rtol=0.0,
        atol=1.0e-14,
    )
    expected_centroid = 0.5 * (
        np.asarray(identity.centroid[source])
        + np.asarray(identity.centroid[target])
    )
    np.testing.assert_allclose(
        np.asarray(merged.centroid[target]),
        expected_centroid,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        float(merged.second_moment[target][0, 0]),
        spacing * spacing / 3.0,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.asarray(merged.third_moment[target]),
        np.zeros((3, 3, 3)),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        float(jnp.sum(merged.aggregate_volume)),
        float(jnp.sum(identity.raw_volume)),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_control_volume_identity_geometry_is_noop() -> None:
    geometry = _build_geometry((7, 7, 7), 2)
    cells = _uniform_control_volume_cells(geometry)
    i, j, k = jnp.meshgrid(
        jnp.arange(geometry.owned_shape[0], dtype=jnp.int32),
        jnp.arange(geometry.owned_shape[1], dtype=jnp.int32),
        jnp.arange(geometry.owned_shape[2], dtype=jnp.int32),
        indexing="ij",
    )

    np.testing.assert_array_equal(np.asarray(cells.owner_i), np.asarray(i))
    np.testing.assert_array_equal(np.asarray(cells.owner_j), np.asarray(j))
    np.testing.assert_array_equal(np.asarray(cells.owner_k), np.asarray(k))
    assert not bool(jnp.any(cells.is_merged_source))
    assert not bool(jnp.any(cells.is_aggregate_target))
    np.testing.assert_array_equal(
        np.asarray(cells.member_count),
        np.asarray(cells.is_active_owner, dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.asarray(cells.aggregate_volume),
        np.asarray(cells.raw_volume),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(cells.centroid),
        np.asarray(cells.raw_centroid),
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        np.asarray(cells.second_moment),
        np.asarray(cells.raw_second_moment),
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        np.asarray(cells.third_moment),
        np.asarray(cells.raw_third_moment),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_control_volume_product_average_includes_moment_covariance() -> None:
    geometry = _build_geometry((7, 7, 7), 2)
    source = (3, 3, 3)
    target = (2, 3, 3)
    cells = _uniform_control_volume_cells(
        geometry,
        merged_source=source,
        merge_target=target,
    )
    left_gradient = jnp.asarray((1.2, -0.4, 0.3), dtype=jnp.float64)
    right_gradient = jnp.asarray((-0.7, 0.8, 0.2), dtype=jnp.float64)
    left = jnp.where(
        cells.is_active_owner,
        0.6 + jnp.einsum("...i,i->...", cells.centroid, left_gradient),
        0.0,
    )
    right = jnp.where(
        cells.is_active_owner,
        -0.2 + jnp.einsum("...i,i->...", cells.centroid, right_gradient),
        0.0,
    )
    zeros_hessian = jnp.zeros(cells.shape + (3, 3), dtype=jnp.float64)
    valid = jnp.asarray(cells.is_active_owner, dtype=bool)

    def polynomial(gradient: jnp.ndarray) -> LocalControlVolumePolynomial3D:
        return LocalControlVolumePolynomial3D(
            gradient=jnp.where(valid[..., None], gradient, 0.0),
            hessian=zeros_hessian,
            valid=valid,
            polynomial_order=jnp.where(valid, 1, 0),
            condition_number=jnp.where(valid, 1.0, jnp.inf),
        )

    actual = local_control_volume_product_average(
        left,
        right,
        polynomial(left_gradient),
        polynomial(right_gradient),
        cells,
    )
    covariance = jnp.einsum(
        "...i,...ij,...j->...",
        left_gradient,
        cells.second_moment,
        right_gradient,
    )
    expected = jnp.where(
        cells.is_active_owner,
        left * right + covariance,
        0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=0.0,
        atol=1.0e-14,
    )
    assert abs(float(actual[target] - left[target] * right[target])) > 1.0e-6
    assert float(actual[source]) == 0.0




def test_regular_boundary_moment_closure_reproduces_cubic() -> None:
    shape = (7, 5, 5)
    geometry = _build_geometry(shape, 2)
    domain = _build_domain(shape, 2)
    cells = _uniform_control_volume_cells(geometry)
    h = 1.0 / shape[0]
    intervals = np.asarray(
        ((0.0, h), (h, 2.0 * h), (2.0 * h, 3.0 * h)),
        dtype=np.float64,
    )
    moment_rows = [[1.0, 0.0, 0.0, 0.0]]
    for lower, upper in intervals:
        moment_rows.append(
            [
                1.0,
                0.5 * (lower + upper),
                (lower**2 + lower * upper + upper**2) / 3.0,
                (
                    lower**3
                    + lower**2 * upper
                    + lower * upper**2
                    + upper**3
                )
                / 4.0,
            ]
        )
    moment_matrix = np.asarray(moment_rows, dtype=np.float64)
    face_weights_1d = np.linalg.solve(
        moment_matrix.T,
        np.asarray((0.0, 1.0, 0.0, 0.0), dtype=np.float64),
    )
    first_centroid = 0.5 * h
    owner_weights_1d = np.linalg.solve(
        moment_matrix.T,
        np.asarray(
            (
                0.0,
                1.0,
                2.0 * first_centroid,
                3.0 * first_centroid**2,
            ),
            dtype=np.float64,
        ),
    )
    face_shapes = tuple(
        geometry.layout.face_control_shape(axis) for axis in range(3)
    )
    face_weights = [
        np.zeros(face_shape + (4,), dtype=np.float64)
        for face_shape in face_shapes
    ]
    owner_weights = [
        np.zeros(face_shape + (4,), dtype=np.float64)
        for face_shape in face_shapes
    ]
    valid = [
        np.zeros(face_shape, dtype=bool) for face_shape in face_shapes
    ]
    face_weights[0][0] = face_weights_1d
    owner_weights[0][0] = owner_weights_1d
    valid[0][0] = True
    closure = LocalRegularBoundaryMomentClosure3D(
        layout=geometry.layout,
        x_face_weights=jnp.asarray(face_weights[0]),
        y_face_weights=jnp.asarray(face_weights[1]),
        z_face_weights=jnp.asarray(face_weights[2]),
        x_owner_weights=jnp.asarray(owner_weights[0]),
        y_owner_weights=jnp.asarray(owner_weights[1]),
        z_owner_weights=jnp.asarray(owner_weights[2]),
        x_valid=jnp.asarray(valid[0]),
        y_valid=jnp.asarray(valid[1]),
        z_valid=jnp.asarray(valid[2]),
    )
    bundle = LocalEmbeddedControlVolumeGeometry3D(
        cells=cells,
        regular_faces=geometry.regular_face_geometry,
        irregular_faces=LocalControlVolumeFaceRows3D.empty(
            geometry.layout
        ),
        reconstruction=LocalMomentReconstruction3D.empty(
            geometry.layout
        ),
        regular_boundary_closure=closure,
    )
    # A reconstruction row may target a first physical owner cell.  Its
    # candidate is deliberately zero here: the final boundary patch must
    # restore the cubic moment derivative after row assembly.
    target_row_for_cell = -jnp.ones(shape, dtype=jnp.int32).at[0, 0, 0].set(0)
    ordering_row = LocalMomentReconstruction3D(
        layout=geometry.layout,
        target_i=jnp.asarray((0,), dtype=jnp.int32),
        target_j=jnp.asarray((0,), dtype=jnp.int32),
        target_k=jnp.asarray((0,), dtype=jnp.int32),
        equation_kind=jnp.zeros((1, 1), dtype=jnp.int32),
        sample_i=jnp.zeros((1, 1), dtype=jnp.int32),
        sample_j=jnp.zeros((1, 1), dtype=jnp.int32),
        sample_k=jnp.zeros((1, 1), dtype=jnp.int32),
        boundary_face_row=jnp.zeros((1, 1), dtype=jnp.int32),
        equation_active=jnp.zeros((1, 1), dtype=bool),
        rhs_transform=jnp.zeros((1, 9, 1), dtype=jnp.float64),
        active=jnp.asarray((True,)),
        target_row_for_cell=target_row_for_cell,
        polynomial_order=jnp.asarray((2,), dtype=jnp.int32),
        rank=jnp.asarray((0,), dtype=jnp.int32),
        condition_number=jnp.asarray((1.0,), dtype=jnp.float64),
        max_rows=1,
        max_equations=1,
    )
    bundle = replace(bundle, reconstruction=ordering_row)
    coefficients = np.asarray((0.4, -0.7, 0.9, -0.35), dtype=np.float64)
    x_faces = np.arange(shape[0] + 1, dtype=np.float64) * h
    cell_average = np.empty((shape[0],), dtype=np.float64)
    for index, (lower, upper) in enumerate(
        zip(x_faces[:-1], x_faces[1:])
    ):
        cell_average[index] = (
            coefficients[0]
            + coefficients[1] * 0.5 * (lower + upper)
            + coefficients[2]
            * (lower**2 + lower * upper + upper**2)
            / 3.0
            + coefficients[3]
            * (
                lower**3
                + lower**2 * upper
                + lower * upper**2
                + upper**3
            )
            / 4.0
        )
    values_owned = jnp.broadcast_to(
        jnp.asarray(cell_average)[:, None, None],
        shape,
    )
    field_halo = inject_owned_field_to_halo(
        values_owned,
        geometry.layout,
    )
    empty_face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    face_bc = LocalBoundaryFaceBC3D(
        kind_x=empty_face_bc.kind_x.at[0].set(BC_DIRICHLET),
        kind_y=empty_face_bc.kind_y,
        kind_z=empty_face_bc.kind_z,
        value_x=empty_face_bc.value_x.at[0].set(coefficients[0]),
        value_y=empty_face_bc.value_y,
        value_z=empty_face_bc.value_z,
        mask_x=empty_face_bc.mask_x.at[0].set(True),
        mask_y=empty_face_bc.mask_y,
        mask_z=empty_face_bc.mask_z,
        layout=geometry.layout,
    )
    polynomial = build_local_control_volume_polynomial_from_field(
        field_halo,
        geometry,
        domain,
        StencilBuilderContext(layout=geometry.layout, domain=domain),
        bundle,
        LocalControlVolumeBoundaryBC3D.empty(),
        face_bc,
    )
    expected = (
        coefficients[1]
        + 2.0 * coefficients[2] * first_centroid
        + 3.0 * coefficients[3] * first_centroid**2
    )
    np.testing.assert_allclose(
        np.asarray(polynomial.gradient[0, ..., 0]),
        expected,
        rtol=0.0,
        atol=2.0e-13,
    )


def test_regular_boundary_moment_closure_reproduces_metric_weighted_cubic() -> None:
    """Verify moment weights when finite-volume measures vary radially."""

    intervals = np.asarray(((0.0, 0.2), (0.2, 0.4), (0.4, 0.6)))
    nodes, weights = np.polynomial.legendre.leggauss(12)

    def jacobian(x: np.ndarray) -> np.ndarray:
        return 1.0 + 0.45 * x + 0.2 * x * x

    def average_power(lower: float, upper: float, power: int) -> float:
        x = 0.5 * (lower + upper) + 0.5 * (upper - lower) * nodes
        measure = weights * 0.5 * (upper - lower) * jacobian(x)
        return float(np.sum(measure * x**power) / np.sum(measure))

    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = 1.0
    for sample, (lower, upper) in enumerate(intervals, start=1):
        for power in range(4):
            matrix[sample, power] = average_power(lower, upper, power)
    face_weights = np.linalg.solve(
        matrix.T,
        np.asarray((0.0, 1.0, 0.0, 0.0)),
    )
    first_centroid = average_power(*intervals[0], 1)
    owner_weights = np.linalg.solve(
        matrix.T,
        np.asarray(
            (
                0.0,
                1.0,
                2.0 * first_centroid,
                3.0 * first_centroid**2,
            )
        ),
    )
    coefficients = np.asarray((0.7, -1.2, 0.9, -0.3))
    data = matrix @ coefficients
    expected_face = coefficients[1]
    expected_owner = (
        coefficients[1]
        + 2.0 * coefficients[2] * first_centroid
        + 3.0 * coefficients[3] * first_centroid**2
    )
    np.testing.assert_allclose(
        np.dot(face_weights, data),
        expected_face,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.dot(owner_weights, data),
        expected_owner,
        rtol=0.0,
        atol=2.0e-12,
    )


def test_control_volume_quadratic_reconstruction_reproduces_polynomials() -> None:
    shape = (7, 7, 7)
    geometry = _build_geometry(shape, 2)
    domain = _build_domain(shape, 2)
    cells = _uniform_control_volume_cells(geometry)
    target_one_wall = (3, 2, 3)
    target_multi_wall = (3, 4, 3)
    h = float(geometry.spacing.dx_owned[target_one_wall])
    one_center = np.asarray(cells.centroid[target_one_wall], dtype=np.float64)
    multi_center = np.asarray(cells.centroid[target_multi_wall], dtype=np.float64)
    faces = _unit_control_volume_face_rows(
        geometry,
        (
            (
                CV_FACE_CUT_WALL,
                target_one_wall,
                None,
                0,
                tuple(one_center + np.asarray((0.45 * h, 0.0, 0.0))),
                1.0,
            ),
            (
                CV_FACE_CUT_WALL,
                target_multi_wall,
                None,
                0,
                tuple(multi_center + np.asarray((0.45 * h, 0.0, 0.0))),
                1.0,
            ),
            (
                CV_FACE_CUT_WALL,
                target_multi_wall,
                None,
                1,
                tuple(multi_center + np.asarray((0.0, -0.4 * h, 0.0))),
                -1.0,
            ),
        ),
    )
    spacing = jnp.stack(
        (
            geometry.spacing.dx_owned,
            geometry.spacing.dy_owned,
            geometry.spacing.dz_owned,
        ),
        axis=-1,
    )
    reconstruction = precompute_local_moment_reconstruction(
        cells,
        faces,
        spacing_owned=spacing,
    )
    bundle = LocalEmbeddedControlVolumeGeometry3D(
        cells=cells,
        regular_faces=geometry.regular_face_geometry,
        irregular_faces=faces,
        reconstruction=reconstruction,
    )
    assert bool(jnp.all(reconstruction.polynomial_order[reconstruction.active] == 3))
    one_row = int(reconstruction.target_row_for_cell[target_one_wall])
    one_dirichlet = np.asarray(
        reconstruction.equation_active[one_row]
        & (
            reconstruction.equation_kind[one_row]
            == CV_RECONSTRUCTION_EQUATION_DIRICHLET
        )
    )
    # A wall constraint is collocated with each active face quadrature point,
    # rather than reduced to one boundary-centroid value.
    assert set(
        zip(
            np.asarray(reconstruction.boundary_patch[one_row])[one_dirichlet],
            np.asarray(reconstruction.boundary_quadrature[one_row])[one_dirichlet],
        )
    ) == {(0, 0), (0, 1), (0, 2), (0, 3)}

    coefficient_sets = (
        (
            1.25,
            jnp.zeros((3,), dtype=jnp.float64),
            jnp.zeros((3, 3), dtype=jnp.float64),
        ),
        (
            -0.4,
            jnp.asarray((1.7, -0.8, 0.3), dtype=jnp.float64),
            jnp.zeros((3, 3), dtype=jnp.float64),
        ),
        (
            0.2,
            jnp.asarray((0.7, -0.35, 0.2), dtype=jnp.float64),
            jnp.asarray(
                (
                    (1.2, 0.15, -0.1),
                    (0.15, -0.8, 0.25),
                    (-0.1, 0.25, 0.6),
                ),
                dtype=jnp.float64,
            ),
        ),
    )
    for constant, gradient, hessian in coefficient_sets:
        field_halo = _quadratic_cell_average_halo(
            geometry,
            constant=constant,
            gradient=gradient,
            hessian=hessian,
        )
        wall_values = _quadratic_point_value(
            faces.quadrature_points,
            constant=constant,
            gradient=gradient,
            hessian=hessian,
        )
        face_measure = jnp.linalg.norm(
            faces.area_covector_weight,
            axis=-1,
        )
        measure = jnp.sum(face_measure, axis=(1, 2))
        boundary_centroid = jnp.sum(
            face_measure[..., None] * faces.quadrature_points,
            axis=(1, 2),
        ) / jnp.maximum(measure[:, None], 1.0e-30)
        centroid_value = _quadratic_point_value(
            boundary_centroid,
            constant=constant,
            gradient=gradient,
            hessian=hessian,
        )
        boundary_bc = LocalControlVolumeBoundaryBC3D(
            kind=jnp.full((faces.max_rows,), BC_DIRICHLET, dtype=jnp.int32),
            centroid_value=centroid_value,
            quadrature_value=wall_values,
            active=jnp.ones((faces.max_rows,), dtype=bool),
            max_rows=faces.max_rows,
            max_patches=faces.max_patches,
        )
        polynomial = build_local_control_volume_polynomial_from_field(
            field_halo,
            geometry,
            domain,
            StencilBuilderContext(layout=geometry.layout, domain=domain),
            bundle,
            boundary_bc,
        )
        for target in (target_one_wall, target_multi_wall):
            expected_gradient = gradient + hessian @ cells.centroid[target]
            np.testing.assert_allclose(
                np.asarray(polynomial.gradient[target]),
                np.asarray(expected_gradient),
                rtol=0.0,
                atol=2.0e-11,
            )
            np.testing.assert_allclose(
                np.asarray(polynomial.hessian[target]),
                np.asarray(hessian),
                rtol=0.0,
                atol=3.0e-10,
            )
            np.testing.assert_allclose(
                np.asarray(polynomial.third_derivative[target]),
                np.zeros((3, 3, 3)),
                rtol=0.0,
                atol=3.0e-9,
            )
            target_rows = np.flatnonzero(
                np.asarray(faces.active, dtype=bool)
                & (np.asarray(faces.minus_owner_i) == target[0])
                & (np.asarray(faces.minus_owner_j) == target[1])
                & (np.asarray(faces.minus_owner_k) == target[2])
            )
            for row in target_rows:
                value, _gradient, valid = (
                    evaluate_local_control_volume_polynomial(
                        field_halo[geometry.layout.owned_slices_cell],
                        polynomial,
                        cells,
                        jnp.full(
                            faces.quadrature_points[row].shape[:-1],
                            target[0],
                            dtype=jnp.int32,
                        ),
                        jnp.full(
                            faces.quadrature_points[row].shape[:-1],
                            target[1],
                            dtype=jnp.int32,
                        ),
                        jnp.full(
                            faces.quadrature_points[row].shape[:-1],
                            target[2],
                            dtype=jnp.int32,
                        ),
                        faces.quadrature_points[row],
                    )
                )
                mask = np.asarray(faces.quadrature_active[row], dtype=bool)
                assert bool(jnp.all(valid[mask]))
                np.testing.assert_allclose(
                    np.asarray(value)[mask],
                    np.asarray(wall_values[row])[mask],
                    rtol=0.0,
                    atol=5.0e-10,
                )


def test_control_volume_padded_rows_do_not_overwrite_real_target() -> None:
    shape = (7, 7, 7)
    geometry = _build_geometry(shape, 2)
    domain = _build_domain(shape, 2)
    cells = _uniform_control_volume_cells(geometry)
    target = (0, 0, 0)
    h = float(geometry.spacing.dx_owned[target])
    target_center = np.asarray(cells.centroid[target], dtype=np.float64)
    faces = _unit_control_volume_face_rows(
        geometry,
        (
            (
                CV_FACE_CUT_WALL,
                target,
                None,
                0,
                tuple(target_center + np.asarray((0.45 * h, 0.0, 0.0))),
                1.0,
            ),
        ),
    )
    spacing = jnp.stack(
        (
            geometry.spacing.dx_owned,
            geometry.spacing.dy_owned,
            geometry.spacing.dz_owned,
        ),
        axis=-1,
    )
    reconstruction = precompute_local_moment_reconstruction(
        cells,
        faces,
        spacing_owned=spacing,
        requested_order=2,
    )
    assert reconstruction.max_rows == 1
    extra_rows = 3

    def pad_rows(array, value=0):
        padding = ((0, extra_rows),) + tuple(
            (0, 0) for _ in range(array.ndim - 1)
        )
        return jnp.pad(array, padding, constant_values=value)

    padded = LocalMomentReconstruction3D(
        layout=reconstruction.layout,
        target_i=pad_rows(reconstruction.target_i),
        target_j=pad_rows(reconstruction.target_j),
        target_k=pad_rows(reconstruction.target_k),
        equation_kind=pad_rows(reconstruction.equation_kind),
        sample_i=pad_rows(reconstruction.sample_i),
        sample_j=pad_rows(reconstruction.sample_j),
        sample_k=pad_rows(reconstruction.sample_k),
        boundary_face_row=pad_rows(reconstruction.boundary_face_row),
        boundary_patch=pad_rows(reconstruction.boundary_patch),
        boundary_quadrature=pad_rows(reconstruction.boundary_quadrature),
        equation_active=pad_rows(
            reconstruction.equation_active,
            value=False,
        ),
        rhs_transform=pad_rows(reconstruction.rhs_transform),
        active=pad_rows(reconstruction.active, value=False),
        target_row_for_cell=reconstruction.target_row_for_cell,
        polynomial_order=pad_rows(reconstruction.polynomial_order),
        rank=pad_rows(reconstruction.rank),
        condition_number=pad_rows(
            reconstruction.condition_number,
            value=jnp.inf,
        ),
        max_rows=reconstruction.max_rows + extra_rows,
        max_equations=reconstruction.max_equations,
    )
    bundle = LocalEmbeddedControlVolumeGeometry3D(
        cells=cells,
        regular_faces=geometry.regular_face_geometry,
        irregular_faces=faces,
        reconstruction=padded,
    )
    constant = 0.3
    gradient = jnp.asarray((0.8, -0.4, 0.25), dtype=jnp.float64)
    hessian = jnp.asarray(
        (
            (0.6, 0.1, -0.05),
            (0.1, -0.3, 0.08),
            (-0.05, 0.08, 0.4),
        ),
        dtype=jnp.float64,
    )
    field_halo = _quadratic_cell_average_halo(
        geometry,
        constant=constant,
        gradient=gradient,
        hessian=hessian,
    )
    wall_values = _quadratic_point_value(
        faces.quadrature_points,
        constant=constant,
        gradient=gradient,
        hessian=hessian,
    )
    face_measure = jnp.linalg.norm(
        faces.area_covector_weight,
        axis=-1,
    )
    measure = jnp.sum(face_measure, axis=(1, 2))
    boundary_centroid = jnp.sum(
        face_measure[..., None] * faces.quadrature_points,
        axis=(1, 2),
    ) / jnp.maximum(measure[:, None], 1.0e-30)
    boundary_bc = LocalControlVolumeBoundaryBC3D(
        kind=jnp.full((faces.max_rows,), BC_DIRICHLET, dtype=jnp.int32),
        centroid_value=_quadratic_point_value(
            boundary_centroid,
            constant=constant,
            gradient=gradient,
            hessian=hessian,
        ),
        quadrature_value=wall_values,
        active=jnp.ones((faces.max_rows,), dtype=bool),
        max_rows=faces.max_rows,
        max_patches=faces.max_patches,
    )
    polynomial = build_local_control_volume_polynomial_from_field(
        field_halo,
        geometry,
        domain,
        StencilBuilderContext(layout=geometry.layout, domain=domain),
        bundle,
        boundary_bc,
    )
    assert bool(polynomial.valid[target])
    np.testing.assert_allclose(
        np.asarray(polynomial.gradient[target]),
        np.asarray(gradient + hessian @ cells.centroid[target]),
        rtol=0.0,
        atol=3.0e-11,
    )
    np.testing.assert_allclose(
        np.asarray(polynomial.hessian[target]),
        np.asarray(hessian),
        rtol=0.0,
        atol=4.0e-10,
    )


def test_control_volume_partial_aggregate_flux_is_conservative_and_source_safe() -> None:
    shape = (7, 7, 7)
    geometry = _build_geometry(shape, 2)
    domain = _build_domain(shape, 2)
    source = (3, 3, 3)
    minus_owner = (2, 3, 3)
    plus_owner = (2, 3, 4)
    cells = _uniform_control_volume_cells(
        geometry,
        merged_source=source,
        merge_target=minus_owner,
    )
    center = 0.5 * (
        np.asarray(cells.centroid[minus_owner])
        + np.asarray(cells.centroid[plus_owner])
    )
    faces = _unit_control_volume_face_rows(
        geometry,
        (
            (
                CV_FACE_PARTIAL,
                minus_owner,
                plus_owner,
                2,
                tuple(center),
                1.0,
            ),
        ),
    )
    spacing = jnp.stack(
        (
            geometry.spacing.dx_owned,
            geometry.spacing.dy_owned,
            geometry.spacing.dz_owned,
        ),
        axis=-1,
    )
    reconstruction = precompute_local_moment_reconstruction(
        cells,
        faces,
        spacing_owned=spacing,
        requested_order=2,
    )
    bundle = LocalEmbeddedControlVolumeGeometry3D(
        cells=cells,
        regular_faces=_all_closed_regular_faces(geometry),
        irregular_faces=faces,
        reconstruction=reconstruction,
    )
    boundary_bc = LocalControlVolumeBoundaryBC3D.empty(
        max_rows=faces.max_rows,
        max_patches=faces.max_patches,
    )
    owner_values = jnp.where(cells.is_active_owner, 1.25, 0.0)

    def evaluate(poison_value: float) -> tuple[jnp.ndarray, object]:
        storage = expand_local_control_volume_owner_field(owner_values, cells)
        storage = storage.at[source].set(poison_value)
        field_halo = inject_owned_field_to_halo(storage, geometry.layout)
        local = build_conservative_stencil_from_field(
            field_halo,
            geometry,
            StencilBuilderContext(layout=geometry.layout, domain=domain),
        )
        polynomial = build_local_control_volume_polynomial_from_field(
            field_halo,
            geometry,
            domain,
            StencilBuilderContext(layout=geometry.layout, domain=domain),
            bundle,
            boundary_bc,
        )
        divergence = local_parallel_flux_div_op(
            local,
            geometry,
            domain,
            control_volume_geometry=bundle,
            boundary_bc=boundary_bc,
            field_reconstruction=polynomial,
        )
        return divergence, polynomial

    reference, polynomial = evaluate(1.25)
    poisoned, poisoned_polynomial = evaluate(jnp.nan)
    active = np.asarray(cells.is_active_owner, dtype=bool)
    np.testing.assert_allclose(
        np.asarray(poisoned)[active],
        np.asarray(reference)[active],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert bool(jnp.all(jnp.isfinite(poisoned[cells.is_active_owner])))
    assert float(poisoned[source]) == 0.0
    assert float(jnp.linalg.norm(reference[minus_owner])) > 0.0
    integrated = (
        reference[minus_owner] * cells.aggregate_volume[minus_owner]
        + reference[plus_owner] * cells.aggregate_volume[plus_owner]
    )
    np.testing.assert_allclose(float(integrated), 0.0, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(
        np.asarray(polynomial.gradient[source]),
        np.zeros((3,), dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(poisoned_polynomial.gradient[source]),
        np.zeros((3,), dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_control_volume_projected_flux_is_conservative_and_source_safe() -> None:
    shape = (7, 7, 7)
    geometry = _build_geometry(shape, 2)
    domain = _build_domain(shape, 2)
    source = (3, 3, 3)
    minus_owner = (2, 3, 3)
    plus_owner = (2, 4, 3)
    cells = _uniform_control_volume_cells(
        geometry,
        merged_source=source,
        merge_target=minus_owner,
    )
    center = 0.5 * (
        np.asarray(cells.centroid[minus_owner])
        + np.asarray(cells.centroid[plus_owner])
    )
    faces = _unit_control_volume_face_rows(
        geometry,
        (
            (
                CV_FACE_PARTIAL,
                minus_owner,
                plus_owner,
                1,
                tuple(center),
                1.0,
            ),
        ),
    )
    spacing = jnp.stack(
        (
            geometry.spacing.dx_owned,
            geometry.spacing.dy_owned,
            geometry.spacing.dz_owned,
        ),
        axis=-1,
    )
    reconstruction = precompute_local_moment_reconstruction(
        cells,
        faces,
        spacing_owned=spacing,
        requested_order=2,
    )
    bundle = LocalEmbeddedControlVolumeGeometry3D(
        cells=cells,
        regular_faces=_all_closed_regular_faces(geometry),
        irregular_faces=faces,
        reconstruction=reconstruction,
    )
    boundary_bc = LocalControlVolumeBoundaryBC3D.empty(
        max_rows=faces.max_rows,
        max_patches=faces.max_patches,
    )
    owner_values = jnp.where(
        cells.is_active_owner,
        0.7 + 0.4 * cells.centroid[..., 1],
        0.0,
    )

    def evaluate(poison_value: float) -> tuple[jnp.ndarray, object]:
        storage = expand_local_control_volume_owner_field(owner_values, cells)
        storage = storage.at[source].set(poison_value)
        field_halo = inject_owned_field_to_halo(storage, geometry.layout)
        context = StencilBuilderContext(
            layout=geometry.layout,
            domain=domain,
        )
        local = build_conservative_stencil_from_field(
            field_halo,
            geometry,
            context,
        )
        polynomial = build_local_control_volume_polynomial_from_field(
            field_halo,
            geometry,
            domain,
            context,
            bundle,
            boundary_bc,
        )
        divergence = local_perp_laplacian_conservative_op(
            local,
            geometry,
            domain,
            control_volume_geometry=bundle,
            boundary_bc=boundary_bc,
            field_reconstruction=polynomial,
        )
        return divergence, polynomial

    reference, polynomial = evaluate(float(owner_values[source]))
    poisoned, poisoned_polynomial = evaluate(jnp.nan)
    active = np.asarray(cells.is_active_owner, dtype=bool)
    np.testing.assert_allclose(
        np.asarray(poisoned)[active],
        np.asarray(reference)[active],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert bool(jnp.all(jnp.isfinite(poisoned[cells.is_active_owner])))
    assert float(poisoned[source]) == 0.0
    assert float(jnp.linalg.norm(reference[minus_owner])) > 0.0
    integrated = (
        reference[minus_owner] * cells.aggregate_volume[minus_owner]
        + reference[plus_owner] * cells.aggregate_volume[plus_owner]
    )
    np.testing.assert_allclose(float(integrated), 0.0, rtol=0.0, atol=1.0e-13)
    for candidate in (polynomial, poisoned_polynomial):
        np.testing.assert_allclose(
            np.asarray(candidate.gradient[source]),
            np.zeros((3,), dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )


def test_control_volume_physical_boundary_uses_quadratic_gradient() -> None:
    shape = (7, 7, 7)
    geometry = _build_geometry(shape, 2)
    domain = _build_domain(shape, 2)
    cells = _uniform_control_volume_cells(geometry)
    owner = (0, 3, 3)
    wall_x = float(geometry.grid.x.faces_owned[0])
    owner_centroid = np.asarray(cells.centroid[owner], dtype=np.float64)
    face_center = owner_centroid.copy()
    face_center[0] = wall_x
    faces = _unit_control_volume_face_rows(
        geometry,
        (
            (
                CV_FACE_PHYSICAL_BOUNDARY,
                owner,
                None,
                0,
                tuple(face_center),
                -1.0,
            ),
        ),
    )
    spacing = jnp.stack(
        (
            geometry.spacing.dx_owned,
            geometry.spacing.dy_owned,
            geometry.spacing.dz_owned,
        ),
        axis=-1,
    )
    reconstruction = precompute_local_moment_reconstruction(
        cells,
        faces,
        spacing_owned=spacing,
        requested_order=2,
    )
    bundle = LocalEmbeddedControlVolumeGeometry3D(
        cells=cells,
        regular_faces=_all_closed_regular_faces(geometry),
        irregular_faces=faces,
        reconstruction=reconstruction,
    )

    x = cells.centroid[..., 0]
    y = cells.centroid[..., 1]
    z = cells.centroid[..., 2]
    moment = cells.second_moment
    owner_values = jnp.where(
        cells.is_active_owner,
        (
            x**3
            + 3.0 * x * moment[..., 0, 0]
            + 0.5 * (y**3 + 3.0 * y * moment[..., 1, 1])
            - 0.25 * (z**3 + 3.0 * z * moment[..., 2, 2])
            + 0.7
            * (
                x * y * z
                + x * moment[..., 1, 2]
                + y * moment[..., 0, 2]
                + z * moment[..., 0, 1]
            )
            + 2.0 * x
            - y
            + 0.3 * z
        ),
        0.0,
    )
    gradient = jnp.stack(
        (
            3.0 * x**2 + 0.7 * y * z + 2.0,
            1.5 * y**2 + 0.7 * x * z - 1.0,
            -0.75 * z**2 + 0.7 * x * y + 0.3,
        ),
        axis=-1,
    )
    hessian = jnp.zeros(cells.shape + (3, 3), dtype=jnp.float64)
    polynomial = LocalControlVolumePolynomial3D(
        gradient=gradient,
        hessian=hessian,
        valid=cells.is_active_owner,
        polynomial_order=jnp.where(cells.is_active_owner, 2, 0),
        condition_number=jnp.where(cells.is_active_owner, 1.0, jnp.inf),
    )
    quadrature_points = faces.quadrature_points
    qx = quadrature_points[..., 0]
    qy = quadrature_points[..., 1]
    qz = quadrature_points[..., 2]
    quadrature_value = (
        qx**3
        + 0.5 * qy**3
        - 0.25 * qz**3
        + 0.7 * qx * qy * qz
        + 2.0 * qx
        - qy
        + 0.3 * qz
    )
    wall_value = float(quadrature_value[0, 0, 0])
    boundary_bc = LocalControlVolumeBoundaryBC3D(
        kind=jnp.asarray((BC_DIRICHLET,), dtype=jnp.int32),
        centroid_value=jnp.asarray((wall_value,), dtype=jnp.float64),
        quadrature_value=quadrature_value,
        active=jnp.asarray((True,), dtype=bool),
        max_rows=faces.max_rows,
        max_patches=faces.max_patches,
    )
    storage = expand_local_control_volume_owner_field(owner_values, cells)
    field_halo = inject_owned_field_to_halo(storage, geometry.layout)
    local = build_conservative_stencil_from_field(
        field_halo,
        geometry,
        StencilBuilderContext(layout=geometry.layout, domain=domain),
    )
    quadratic_face_gradient = jnp.broadcast_to(
        gradient[owner],
        faces.quadrature_points[0].shape,
    )
    divergence = local_perp_laplacian_conservative_op(
        local,
        geometry,
        domain,
        control_volume_geometry=bundle,
        boundary_bc=boundary_bc,
        field_reconstruction=polynomial,
    )
    expected_flux = jnp.sum(
        faces.J[0]
        * jnp.einsum(
            "pqi,pqij,pqj->pq",
            faces.area_covector_weight[0],
            faces.projector[0],
            quadratic_face_gradient,
        )
    )
    expected_divergence = (
        expected_flux / cells.aggregate_volume[owner]
    )
    np.testing.assert_allclose(
        float(divergence[owner]),
        float(expected_divergence),
        rtol=0.0,
        atol=2.0e-11,
    )


def run_control_volume_reconstruction_checks() -> dict[str, object]:
    """Run the unified control-volume moment/reconstruction checks."""

    print()
    print("=" * 80)
    print("Unified control-volume reconstruction checks")
    print("=" * 80)
    checks = (
        (
            "face_rows_activate_all_gauss_points",
            test_control_volume_face_rows_activate_all_gauss_points,
        ),
        (
            "identity_geometry_is_noop",
            test_control_volume_identity_geometry_is_noop,
        ),
        (
            "aggregate_moments_are_conservative",
            test_control_volume_aggregate_moments_are_conservative,
        ),
        (
            "product_average_includes_moment_covariance",
            test_control_volume_product_average_includes_moment_covariance,
        ),
        (
            "regular_boundary_moment_closure_reproduces_cubic",
            test_regular_boundary_moment_closure_reproduces_cubic,
        ),
        (
            "regular_boundary_moment_closure_reproduces_metric_weighted_cubic",
            test_regular_boundary_moment_closure_reproduces_metric_weighted_cubic,
        ),
        (
            "quadratic_reconstruction_reproduces_polynomials",
            test_control_volume_quadratic_reconstruction_reproduces_polynomials,
        ),
        (
            "padded_rows_do_not_overwrite_real_target",
            test_control_volume_padded_rows_do_not_overwrite_real_target,
        ),
        (
            "partial_aggregate_flux_is_conservative_and_source_safe",
            test_control_volume_partial_aggregate_flux_is_conservative_and_source_safe,
        ),
        (
            "projected_flux_is_conservative_and_source_safe",
            test_control_volume_projected_flux_is_conservative_and_source_safe,
        ),
        (
            "physical_boundary_uses_quadratic_gradient",
            test_control_volume_physical_boundary_uses_quadratic_gradient,
        ),
    )
    passed: list[str] = []
    start = time.perf_counter()
    for name, check in checks:
        check_start = time.perf_counter()
        check()
        elapsed = time.perf_counter() - check_start
        passed.append(name)
        print(f"PASS {name}  time={elapsed:.3f}s")
    total_elapsed = time.perf_counter() - start
    print(
        f"Completed {len(passed)} control-volume checks in "
        f"{total_elapsed:.3f}s"
    )
    return {"passed": passed, "elapsed": total_elapsed}


def _halo_closure_domain(
    *,
    periodic_axes: tuple[bool, bool, bool],
) -> LocalDomain3D:
    shape = (3, 4, 3)
    return LocalDomain3D(
        shard_spec=ShardSpec3D(
            global_shape=shape,
            owned_start=(0, 0, 0),
            owned_stop=shape,
            shard_index=(0, 0, 0),
            shard_counts=(1, 1, 1),
            periodic_axes=periodic_axes,
            halo_width=1,
        ),
        layout=HaloLayout3D(shape, 1),
        mesh_axis_names=(None, None, None),
    )


def _linear_halo_closure_inputs(
    domain: LocalDomain3D,
) -> tuple[jnp.ndarray, LocalBoundaryFaceBC3D, PhysicalGhostCellFiller3D]:
    layout = domain.layout
    nx, ny, nz = layout.owned_shape
    x = jnp.arange(nx, dtype=jnp.float64)[:, None, None] + 0.5
    y = jnp.arange(ny, dtype=jnp.float64)[None, :, None] + 0.5
    z = jnp.arange(nz, dtype=jnp.float64)[None, None, :] + 0.5
    field = jnp.full(layout.cell_halo_shape, -99.0)
    field = field.at[layout.owned_slices_cell].set(x + y + z)

    x_centers = jnp.arange(nx, dtype=jnp.float64) + 0.5
    y_centers = jnp.arange(ny, dtype=jnp.float64) + 0.5
    z_centers = jnp.arange(nz, dtype=jnp.float64) + 0.5
    bc = LocalBoundaryFaceBC3D.empty(layout)
    value_x = bc.value_x.at[0].set(
        y_centers[:, None] + z_centers[None, :]
    ).at[-1].set(
        float(nx) + y_centers[:, None] + z_centers[None, :]
    )
    value_y = bc.value_y.at[:, 0, :].set(
        x_centers[:, None] + z_centers[None, :]
    ).at[:, -1, :].set(
        x_centers[:, None] + float(ny) + z_centers[None, :]
    )
    value_z = bc.value_z.at[:, :, 0].set(
        x_centers[:, None] + y_centers[None, :]
    ).at[:, :, -1].set(
        x_centers[:, None] + y_centers[None, :] + float(nz)
    )

    physical = tuple(not periodic for periodic in domain.periodic_axes)
    kind_x = bc.kind_x
    value_mask_x = bc.mask_x
    kind_y = bc.kind_y
    value_mask_y = bc.mask_y
    kind_z = bc.kind_z
    value_mask_z = bc.mask_z
    if physical[0]:
        kind_x = kind_x.at[0].set(BC_DIRICHLET).at[-1].set(BC_DIRICHLET)
        value_mask_x = value_mask_x.at[0].set(True).at[-1].set(True)
    if physical[1]:
        kind_y = kind_y.at[:, 0, :].set(BC_DIRICHLET).at[:, -1, :].set(
            BC_DIRICHLET
        )
        value_mask_y = value_mask_y.at[:, 0, :].set(True).at[:, -1, :].set(
            True
        )
    if physical[2]:
        kind_z = kind_z.at[:, :, 0].set(BC_DIRICHLET).at[:, :, -1].set(
            BC_DIRICHLET
        )
        value_mask_z = value_mask_z.at[:, :, 0].set(True).at[:, :, -1].set(
            True
        )
    bc = replace(
        bc,
        kind_x=kind_x,
        kind_y=kind_y,
        kind_z=kind_z,
        value_x=value_x,
        value_y=value_y,
        value_z=value_z,
        mask_x=value_mask_x,
        mask_y=value_mask_y,
        mask_z=value_mask_z,
    )

    dirichlet = GhostFillWeights1D(
        owned_weights=jnp.array([[-1.0]], dtype=jnp.float64),
        bc_weights=jnp.array([2.0], dtype=jnp.float64),
    )
    neutral = GhostFillWeights1D(
        owned_weights=jnp.array([[1.0]], dtype=jnp.float64),
        bc_weights=jnp.array([0.0], dtype=jnp.float64),
    )
    filler = PhysicalGhostCellFiller3D(
        dirichlet=(dirichlet, dirichlet, dirichlet),
        neumann_lower=(neutral, neutral, neutral),
        neumann_upper=(neutral, neutral, neutral),
    )
    return field, bc, filler


def test_halo_closure_physical_periodic_corner() -> None:
    domain = _halo_closure_domain(periodic_axes=(False, True, False))
    field, bc, filler = _linear_halo_closure_inputs(domain)
    closed = LocalHaloClosure3D(
        physical_ghost_filler=filler,
        topology_filler=TopologyHaloFiller3D(
            rules=(LocalPeriodicTopologyRule3D(),)
        ),
    )(field, domain, bc)
    h = domain.layout.halo_width
    ny = domain.owned_shape[1]
    np.testing.assert_allclose(
        closed[0, 0, h:-h],
        closed[0, h + ny - 1, h:-h],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_halo_closure_physical_edges_and_corners() -> None:
    domain = _halo_closure_domain(periodic_axes=(False, False, False))
    field, bc, filler = _linear_halo_closure_inputs(domain)
    closed = LocalHaloClosure3D(physical_ghost_filler=filler)(
        field,
        domain,
        bc,
    )
    coordinates = [
        jnp.arange(-0.5, extent + 0.5 + 1.0e-12, 1.0)
        for extent in domain.owned_shape
    ]
    exact = (
        coordinates[0][:, None, None]
        + coordinates[1][None, :, None]
        + coordinates[2][None, None, :]
    )
    np.testing.assert_allclose(closed, exact, rtol=0.0, atol=1.0e-12)


def run_halo_closure_checks() -> dict[str, object]:
    print()
    print("=" * 80)
    print("Complete halo closure checks")
    print("=" * 80)
    checks = (
        ("physical_periodic_corner", test_halo_closure_physical_periodic_corner),
        (
            "physical_edges_and_corners",
            test_halo_closure_physical_edges_and_corners,
        ),
    )
    start = time.perf_counter()
    passed: list[str] = []
    for name, check in checks:
        check_start = time.perf_counter()
        check()
        elapsed = time.perf_counter() - check_start
        passed.append(name)
        print(f"PASS {name}  time={elapsed:.3f}s")
    total_elapsed = time.perf_counter() - start
    print(f"Completed {len(passed)} halo closure checks in {total_elapsed:.3f}s")
    return {"passed": passed, "elapsed": total_elapsed}



def test_oblique_cutwall_local_grad_perp_converges() -> None:
    coarse = _local_grad_error((8, 12, 8))
    fine = _local_grad_error((16, 24, 16))

    assert math.isfinite(coarse)
    assert math.isfinite(fine)
    assert fine < 0.95 * coarse


def test_oblique_cutwall_conservative_perp_laplacian_converges() -> None:
    coarse, coarse_owner_flux, coarse_outside_flux = _conservative_laplacian_error(
        (8, 12, 8)
    )
    fine, fine_owner_flux, fine_outside_flux = _conservative_laplacian_error(
        (16, 24, 16)
    )

    assert math.isfinite(coarse)
    assert math.isfinite(fine)
    assert fine < 0.95 * coarse
    assert coarse_owner_flux > 1.0e-10
    assert fine_owner_flux > 1.0e-10
    assert coarse_outside_flux < 1.0e-12
    assert fine_outside_flux < 1.0e-12


def _shape_from_resolution(n: int) -> tuple[int, int, int]:
    n = int(n)
    return (n, int(round(1.5 * n)), n)


def _normalize_shard_counts(shard_counts: tuple[int, int, int]) -> tuple[int, int, int]:
    shard_counts = tuple(int(value) for value in shard_counts)
    if len(shard_counts) != 3:
        raise ValueError(f"shard_counts must have length 3, got {shard_counts}")
    if any(value <= 0 for value in shard_counts):
        raise ValueError(f"shard_counts must be positive, got {shard_counts}")
    return shard_counts


def _assert_shape_divisible_by_shards(
    shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int],
) -> None:
    for axis, (size, count) in enumerate(zip(shape, shard_counts)):
        if int(size) % int(count):
            raise ValueError(
                f"shape axis {axis} with size {size} is not divisible by "
                f"shard count {count}; shape={shape}, shard_counts={shard_counts}"
            )


def _local_shape_for_shards(
    shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int],
) -> tuple[int, int, int]:
    _assert_shape_divisible_by_shards(shape, shard_counts)
    return tuple(int(size) // int(count) for size, count in zip(shape, shard_counts))


def _require_supported_sharding(
    shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int],
) -> tuple[int, int, int]:
    shard_counts = _normalize_shard_counts(shard_counts)
    _assert_shape_divisible_by_shards(shape, shard_counts)
    local_shape = _local_shape_for_shards(shape, shard_counts)
    if any(size < 3 for size in local_shape):
        raise ValueError(
            "each local shard must have at least 3 cells on every axis for the "
            f"cut-wall operator fixture; got local_shape={local_shape}, "
            f"shape={shape}, shard_counts={shard_counts}"
        )
    return shard_counts


def _make_mesh_for_shard_counts(shard_counts: tuple[int, int, int]) -> Mesh:
    shard_counts = _normalize_shard_counts(shard_counts)
    ndevices = math.prod(shard_counts)
    devices = np.asarray(jax.devices()[:ndevices], dtype=object)
    if devices.size < ndevices:
        raise RuntimeError(
            f"shard_counts={shard_counts} requires {ndevices} devices, "
            f"but only {devices.size} are available"
        )
    return Mesh(devices.reshape(shard_counts), _MESH_AXIS_NAMES)


def _psum_all(value: jnp.ndarray) -> jnp.ndarray:
    result = value
    for axis_name in _MESH_AXIS_NAMES:
        result = lax.psum(result, axis_name)
    return result


def _pmax_all(value: jnp.ndarray) -> jnp.ndarray:
    result = value
    for axis_name in _MESH_AXIS_NAMES:
        result = lax.pmax(result, axis_name)
    return result


def _sharded_local_grad_error(
    shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int],
) -> float:
    shard_counts = _require_supported_sharding(shape, shard_counts)
    if shard_counts == (1, 1, 1):
        return _local_grad_error(shape)

    with _make_mesh_for_shard_counts(shard_counts) as mesh:
        def kernel(_dummy: jax.Array) -> jax.Array:
            case = _build_sharded_case(shape, shard_counts)
            sumsq, count = _local_grad_error_stats(
                case,
                unchecked_dependencies=True,
            )
            global_sumsq = _psum_all(sumsq)
            global_count = _psum_all(count)
            return jnp.sqrt(global_sumsq / global_count)

        mapped = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P(),),
            out_specs=P(),
            check_rep=True,
        )
        result = jax.jit(mapped)(jnp.asarray(0.0, dtype=jnp.float64))
        return float(jax.device_get(result))


def _sharded_conservative_laplacian_error(
    shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int],
) -> tuple[float, float, float]:
    shard_counts = _require_supported_sharding(shape, shard_counts)
    if shard_counts == (1, 1, 1):
        return _conservative_laplacian_error(shape)

    with _make_mesh_for_shard_counts(shard_counts) as mesh:
        def kernel(_dummy: jax.Array) -> jax.Array:
            case = _build_sharded_case(shape, shard_counts)
            sumsq, count, owner_linf, outside_linf = _conservative_laplacian_error_stats(
                case
            )
            global_sumsq = _psum_all(sumsq)
            global_count = _psum_all(count)
            return jnp.asarray(
                [
                    jnp.sqrt(global_sumsq / global_count),
                    _pmax_all(owner_linf),
                    _pmax_all(outside_linf),
                ],
                dtype=jnp.float64,
            )

        mapped = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P(),),
            out_specs=P(),
            check_rep=True,
        )
        result = jax.device_get(jax.jit(mapped)(jnp.asarray(0.0, dtype=jnp.float64)))
        return (float(result[0]), float(result[1]), float(result[2]))


def estimate_orders(errors: list[float], resolutions: list[int]) -> list[float]:
    if len(errors) != len(resolutions):
        raise ValueError("errors and resolutions must have the same length")
    if len(errors) < 2:
        return []

    orders: list[float] = []
    for i in range(1, len(errors)):
        e0 = errors[i - 1]
        e1 = errors[i]
        n0 = resolutions[i - 1]
        n1 = resolutions[i]
        orders.append(float(jnp.log(e0 / e1) / jnp.log(float(n1) / float(n0))))
    return orders


def _print_cutwall_resolution_result(
    *,
    n: int,
    shape: tuple[int, int, int],
    error_l2: float,
    elapsed: float,
    owner_flux_linf: float | None = None,
    outside_flux_linf: float | None = None,
) -> None:
    extra = ""
    if owner_flux_linf is not None and outside_flux_linf is not None:
        extra = (
            f"  owner_flux_linf={owner_flux_linf:.6e}"
            f"  outside_flux_linf={outside_flux_linf:.6e}"
        )
    print(
        f"N={n:4d}  "
        f"shape={shape!s:>14s}  "
        f"L2={error_l2:.6e}  "
        f"time={elapsed:.3f}s"
        f"{extra}"
    )


def run_cutwall_local_grad_perp_convergence(
    *,
    resolutions: tuple[int, ...] = (8, 16, 24),
    shard_counts: tuple[int, int, int] = (1, 1, 1),
) -> dict[str, object]:
    """Run the oblique-cutwall local grad_perp convergence sweep."""

    shard_counts = _normalize_shard_counts(shard_counts)
    if len(resolutions) == 0:
        raise ValueError("resolutions must contain at least one value")
    if any(int(n) < 4 for n in resolutions):
        raise ValueError("each resolution must be at least 4")

    l2_errors: list[float] = []
    case_times: list[float] = []

    print()
    print("=" * 80)
    print("Slab oblique cut-wall local grad_perp convergence")
    print("=" * 80)
    print(f"wall plane: x + {WALL_ALPHA:g} y = {WALL_C:g}")
    print(f"shard_counts = {shard_counts}")
    print()

    for n in resolutions:
        shape = _shape_from_resolution(int(n))
        _require_supported_sharding(shape, shard_counts)
        start = time.perf_counter()
        l2 = _sharded_local_grad_error(shape, shard_counts)
        elapsed = time.perf_counter() - start
        l2_errors.append(l2)
        case_times.append(elapsed)
        _print_cutwall_resolution_result(
            n=int(n),
            shape=shape,
            error_l2=l2,
            elapsed=elapsed,
        )

    l2_orders = estimate_orders(l2_errors, list(resolutions))
    print()
    print("Estimated orders")
    print("-" * 80)
    for i, order in enumerate(l2_orders):
        print(f"N={resolutions[i]} -> {resolutions[i + 1]}: L2 order={order:.3f}")

    return {
        "resolutions": resolutions,
        "shard_counts": shard_counts,
        "l2_errors": l2_errors,
        "case_times": case_times,
        "l2_orders": l2_orders,
    }


def run_cutwall_conservative_perp_laplacian_convergence(
    *,
    resolutions: tuple[int, ...] = (8, 16, 24),
    shard_counts: tuple[int, int, int] = (1, 1, 1),
) -> dict[str, object]:
    """Run the oblique-cutwall conservative perp-laplacian convergence sweep."""

    shard_counts = _normalize_shard_counts(shard_counts)
    if len(resolutions) == 0:
        raise ValueError("resolutions must contain at least one value")
    if any(int(n) < 4 for n in resolutions):
        raise ValueError("each resolution must be at least 4")

    l2_errors: list[float] = []
    owner_flux_linf: list[float] = []
    outside_flux_linf: list[float] = []
    case_times: list[float] = []

    print()
    print("=" * 80)
    print("Slab oblique cut-wall conservative perp_laplacian convergence")
    print("=" * 80)
    print(f"wall plane: x + {WALL_ALPHA:g} y = {WALL_C:g}")
    print(f"shard_counts = {shard_counts}")
    print()

    for n in resolutions:
        shape = _shape_from_resolution(int(n))
        _require_supported_sharding(shape, shard_counts)
        start = time.perf_counter()
        l2, owner_flux, outside_flux = _sharded_conservative_laplacian_error(
            shape,
            shard_counts,
        )
        elapsed = time.perf_counter() - start
        l2_errors.append(l2)
        owner_flux_linf.append(owner_flux)
        outside_flux_linf.append(outside_flux)
        case_times.append(elapsed)
        _print_cutwall_resolution_result(
            n=int(n),
            shape=shape,
            error_l2=l2,
            elapsed=elapsed,
            owner_flux_linf=owner_flux,
            outside_flux_linf=outside_flux,
        )

    l2_orders = estimate_orders(l2_errors, list(resolutions))
    print()
    print("Estimated orders")
    print("-" * 80)
    for i, order in enumerate(l2_orders):
        print(f"N={resolutions[i]} -> {resolutions[i + 1]}: L2 order={order:.3f}")

    return {
        "resolutions": resolutions,
        "shard_counts": shard_counts,
        "l2_errors": l2_errors,
        "owner_flux_linf": owner_flux_linf,
        "outside_flux_linf": outside_flux_linf,
        "case_times": case_times,
        "l2_orders": l2_orders,
    }


def print_jax_runtime_info() -> None:
    print("=" * 80)
    print("JAX runtime")
    print("=" * 80)
    print("default backend:", jax.default_backend())
    print("local_device_count:", jax.local_device_count())
    print("devices:")
    for i, device in enumerate(jax.devices()):
        print(f"  [{i}] {device}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cut-wall slab operator convergence tests."
    )
    parser.add_argument(
        "--operator",
        type=str,
        default="both",
        choices=(
            "both",
            "grad_perp",
            "perp_laplacian_conservative",
            "control_volume_reconstruction",
            "halo_closure",
        ),
        help="Which cut-wall operator convergence sweep to run.",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[8, 16, 24],
        help="Base x/z resolutions. The y resolution is round(1.5*N).",
    )
    parser.add_argument(
        "--shard-counts",
        type=int,
        nargs=3,
        metavar=("PX", "PY", "PZ"),
        default=(1, 1, 1),
        help=(
            "Number of shards along x, y, and z. Each resolution must be "
            "divisible by the corresponding shard count."
        ),
    )
    parser.add_argument(
        "--skip-runtime-info",
        action="store_true",
        help="Do not print JAX device/runtime information before the sweep.",
    )
    args = parser.parse_args()

    if not args.skip_runtime_info:
        print_jax_runtime_info()

    resolutions = tuple(int(n) for n in args.resolutions)
    shard_counts = tuple(int(n) for n in args.shard_counts)
    if args.operator == "control_volume_reconstruction":
        run_control_volume_reconstruction_checks()
        return
    if args.operator == "halo_closure":
        run_halo_closure_checks()
        return
    if args.operator in ("both", "grad_perp"):
        run_cutwall_local_grad_perp_convergence(
            resolutions=resolutions,
            shard_counts=shard_counts,
        )
    if args.operator in ("both", "perp_laplacian_conservative"):
        run_cutwall_conservative_perp_laplacian_convergence(
            resolutions=resolutions,
            shard_counts=shard_counts,
        )


if __name__ == "__main__":
    main()
