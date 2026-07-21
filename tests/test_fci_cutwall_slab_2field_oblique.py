"""Oblique cut-wall slab two-field MMS tests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time as time_module

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
import numpy as np

from jax_drb.geometry import (
    FciGeometry3D,
    LocalCoordinateStencilDependencyMap3D,
    LocalCoordinateStencilLocalDependencyTable,
    LocalDomain3D,
    LocalFciGeometry3D,
    LocalRegularFaceGeometry3D,
    StencilBuilderContext,
    build_conservative_stencil_from_field,
    build_local_stencil_from_field,
)
from jax_drb.native import Rk4Stepper
from jax_drb.native.fci_2_field_rhs import (
    Fci2FieldRhsParameters,
    Fci2FieldState,
)
from jax_drb.native.fci_boundaries import (
    BC_DIRICHLET,
    LocalCutWallBC3D,
    LocalCutWallGeometry3D,
)
from jax_drb.native.fci_halo import HaloExchange3D, PhysicalGhostCellFiller3D
from jax_drb.native.fci_operators import (
    local_curvature_op,
    local_grad_parallel_op_direct,
    local_perp_laplacian_conservative_op,
    local_poisson_bracket_op,
)


jax.config.update("jax_enable_x64", True)

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))
import test_mms_slab_2_field as slab_mms  # noqa: E402


WALL_ALPHA = 0.25
WALL_C = 0.72


@dataclass(frozen=True)
class _ObliqueCutWallFixture:
    stencil_dependencies: LocalCoordinateStencilDependencyMap3D
    stencil_geometry: LocalCutWallGeometry3D
    flux_geometry: LocalCutWallGeometry3D
    owner_mask: jnp.ndarray
    inside_mask: jnp.ndarray
    valid_mask: jnp.ndarray
    regular_face_geometry: LocalRegularFaceGeometry3D


def _shape_from_resolution(resolution: int) -> tuple[int, int, int]:
    n = int(resolution)
    return (n, n, n)


def _wall_x(y):
    return WALL_C - WALL_ALPHA * y


def _inside_from_coordinates(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return x < _wall_x(y)


def _owned_coordinates(geometry: LocalFciGeometry3D):
    x = geometry.grid.x.centers_owned[:, None, None]
    y = geometry.grid.y.centers_owned[None, :, None]
    z = geometry.grid.z.centers_owned[None, None, :]
    return jnp.broadcast_arrays(x, y, z)


def _unchecked_coordinate_dependencies(
    layout,
    *,
    target_flat: jnp.ndarray,
    axis: jnp.ndarray,
    side: jnp.ndarray,
    distance: jnp.ndarray,
    active: jnp.ndarray,
) -> LocalCoordinateStencilDependencyMap3D:
    local = object.__new__(LocalCoordinateStencilLocalDependencyTable)
    object.__setattr__(local, "target_flat", jnp.asarray(target_flat, dtype=jnp.int32))
    object.__setattr__(local, "axis", jnp.asarray(axis, dtype=jnp.int32))
    object.__setattr__(local, "side", jnp.asarray(side, dtype=jnp.int32))
    object.__setattr__(
        local,
        "value_slot",
        jnp.arange(int(target_flat.size), dtype=jnp.int32),
    )
    object.__setattr__(local, "distance", jnp.asarray(distance, dtype=jnp.float64))
    object.__setattr__(local, "active", jnp.asarray(active, dtype=bool))

    dependencies = object.__new__(LocalCoordinateStencilDependencyMap3D)
    object.__setattr__(dependencies, "layout", layout)
    object.__setattr__(dependencies, "local", local)
    object.__setattr__(dependencies, "remote", None)
    return dependencies


def _unchecked_cut_wall_geometry(
    *,
    owner_i: jnp.ndarray,
    owner_j: jnp.ndarray,
    owner_k: jnp.ndarray,
    center: jnp.ndarray,
    normal: jnp.ndarray,
    area_covector: jnp.ndarray,
    distance: jnp.ndarray,
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
    object.__setattr__(
        obj,
        "normal_contra",
        jnp.broadcast_to(jnp.asarray(normal, dtype=jnp.float64), (max_wall_faces, 3)),
    )
    object.__setattr__(
        obj,
        "area_covector",
        jnp.broadcast_to(
            jnp.asarray(area_covector, dtype=jnp.float64),
            (max_wall_faces, 3),
        ),
    )
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
            jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64),
            (max_wall_faces, 3),
        ),
    )
    object.__setattr__(obj, "Bmag", jnp.ones((max_wall_faces,), dtype=jnp.float64))
    object.__setattr__(obj, "sign", jnp.ones((max_wall_faces,), dtype=jnp.float64))
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


def _build_oblique_cut_wall_fixture(
    geometry: LocalFciGeometry3D,
    *,
    global_shape: tuple[int, int, int],
) -> _ObliqueCutWallFixture:
    nx, ny, nz = geometry.owned_shape
    global_nx, global_ny, global_nz = tuple(int(value) for value in global_shape)
    dx = 1.0 / float(global_nx - 1)
    dy = 1.0 / float(global_ny - 1)
    dz = 1.0 / float(global_nz - 1)

    x_centers = jnp.asarray(geometry.grid.x.centers_owned, dtype=jnp.float64)
    y_centers = jnp.asarray(geometry.grid.y.centers_owned, dtype=jnp.float64)
    z_centers = jnp.asarray(geometry.grid.z.centers_owned, dtype=jnp.float64)
    normal = jnp.asarray([1.0, WALL_ALPHA, 0.0], dtype=jnp.float64)
    normal = normal / jnp.linalg.norm(normal)
    area_covector = jnp.asarray([1.0, WALL_ALPHA, 0.0], dtype=jnp.float64) * dy * dz

    x, y, _z = _owned_coordinates(geometry)
    inside_mask = _inside_from_coordinates(x, y)
    i = jnp.arange(nx, dtype=jnp.int32)[:, None, None]
    j = jnp.arange(ny, dtype=jnp.int32)[None, :, None]
    k = jnp.arange(nz, dtype=jnp.int32)[None, None, :]
    margin = (
        (i > 0)
        & (i < nx - 1)
        & (j > 0)
        & (j < ny - 1)
        & (k > 0)
        & (k < nz - 1)
    )

    y_grid, z_grid = jnp.meshgrid(y_centers, z_centers, indexing="ij")
    j_grid, k_grid = jnp.meshgrid(
        jnp.arange(ny, dtype=jnp.int32),
        jnp.arange(nz, dtype=jnp.int32),
        indexing="ij",
    )
    wall_x = _wall_x(y_grid)
    x_index = jnp.searchsorted(x_centers, wall_x, side="right") - 1
    x_active = (
        (wall_x > x_centers[0])
        & (wall_x < x_centers[-1])
        & (x_index >= 0)
        & (x_index < nx)
    )
    x_owner_i = jnp.clip(x_index, 0, nx - 1).astype(jnp.int32)
    owner_x = x_centers[x_owner_i]
    x_distance = jnp.maximum(wall_x - owner_x, 1.0e-30)
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
    flux_geometry = _unchecked_cut_wall_geometry(
        owner_i=x_owner_i.reshape((flux_size,)),
        owner_j=j_grid.reshape((flux_size,)),
        owner_k=k_grid.reshape((flux_size,)),
        center=flux_center.reshape((flux_size, 3)),
        normal=normal,
        area_covector=area_covector,
        distance=jnp.where(x_active, normal_distance, 1.0).reshape((flux_size,)),
        active=x_active.reshape((flux_size,)),
        stencil_axis=jnp.zeros((flux_size,), dtype=jnp.int32),
        stencil_side=jnp.ones((flux_size,), dtype=jnp.int32),
        stencil_distance=jnp.where(x_active, x_distance, 1.0).reshape((flux_size,)),
    )

    x_grid, z_grid_y = jnp.meshgrid(x_centers, z_centers, indexing="ij")
    i_grid, k_grid_y = jnp.meshgrid(
        jnp.arange(nx, dtype=jnp.int32),
        jnp.arange(nz, dtype=jnp.int32),
        indexing="ij",
    )
    wall_y = (WALL_C - x_grid) / WALL_ALPHA
    y_index = jnp.searchsorted(y_centers, wall_y, side="right") - 1
    y_owner_j = jnp.clip(y_index, 0, ny - 1).astype(jnp.int32)
    owner_y = y_centers[y_owner_j]
    y_active = (
        (wall_y > y_centers[0])
        & (wall_y < y_centers[-1])
        & (y_index >= 0)
        & (y_index < ny)
        & (x_grid < _wall_x(owner_y))
    )
    y_distance = jnp.maximum(wall_y - owner_y, 1.0e-30)
    y_size = nx * nz

    x_center = jnp.stack((wall_x, y_grid, z_grid), axis=-1)
    y_center = jnp.stack((x_grid, wall_y, z_grid_y), axis=-1)
    stencil_owner_i = jnp.concatenate(
        (x_owner_i.reshape((flux_size,)), i_grid.reshape((y_size,))),
        axis=0,
    )
    stencil_owner_j = jnp.concatenate(
        (j_grid.reshape((flux_size,)), y_owner_j.reshape((y_size,))),
        axis=0,
    )
    stencil_owner_k = jnp.concatenate(
        (k_grid.reshape((flux_size,)), k_grid_y.reshape((y_size,))),
        axis=0,
    )
    stencil_axis = jnp.concatenate(
        (
            jnp.zeros((flux_size,), dtype=jnp.int32),
            jnp.ones((y_size,), dtype=jnp.int32),
        ),
        axis=0,
    )
    stencil_side = jnp.ones((flux_size + y_size,), dtype=jnp.int32)
    stencil_distance = jnp.concatenate(
        (
            jnp.where(x_active, x_distance, 1.0).reshape((flux_size,)),
            jnp.where(y_active, y_distance, 1.0).reshape((y_size,)),
        ),
        axis=0,
    )
    stencil_active = jnp.concatenate(
        (x_active.reshape((flux_size,)), y_active.reshape((y_size,))),
        axis=0,
    )
    stencil_geometry = _unchecked_cut_wall_geometry(
        owner_i=stencil_owner_i,
        owner_j=stencil_owner_j,
        owner_k=stencil_owner_k,
        center=jnp.concatenate(
            (x_center.reshape((flux_size, 3)), y_center.reshape((y_size, 3))),
            axis=0,
        ),
        normal=normal,
        area_covector=area_covector,
        distance=stencil_distance,
        active=stencil_active,
        stencil_axis=stencil_axis,
        stencil_side=stencil_side,
        stencil_distance=stencil_distance,
    )

    target_flat = (stencil_owner_i * ny + stencil_owner_j) * nz + stencil_owner_k
    dependencies = _unchecked_coordinate_dependencies(
        geometry.layout,
        target_flat=target_flat,
        axis=stencil_axis,
        side=stencil_side,
        distance=stencil_distance,
        active=stencil_active,
    )

    x_face_i = jnp.arange(nx + 1, dtype=jnp.int32)[:, None, None]
    x_open_mask = ~((x_face_i == (x_owner_i[None, :, :] + 1)) & x_active[None, :, :])
    owner_mask = (i == x_owner_i[None, :, :]) & x_active[None, :, :]
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
    return _ObliqueCutWallFixture(
        stencil_dependencies=dependencies,
        stencil_geometry=stencil_geometry,
        flux_geometry=flux_geometry,
        owner_mask=owner_mask,
        inside_mask=inside_mask,
        valid_mask=inside_mask & margin,
        regular_face_geometry=regular,
    )


def _mms_wall_values(
    geometry: LocalCutWallGeometry3D,
    stage_time: float | jax.Array,
) -> Fci2FieldState:
    x = geometry.center[:, 0]
    y = geometry.center[:, 1]
    z = geometry.center[:, 2]
    return slab_mms._mms_local_exact_state_from_coordinates(x, y, z, stage_time)


def _apply_exact_outside_state(
    state_owned: Fci2FieldState,
    exact_owned: Fci2FieldState,
    inside_mask: jnp.ndarray,
) -> Fci2FieldState:
    return Fci2FieldState(
        density=jnp.where(inside_mask, state_owned.density, exact_owned.density),
        v_parallel=jnp.where(inside_mask, state_owned.v_parallel, exact_owned.v_parallel),
        density_background=jnp.where(
            inside_mask,
            state_owned.density_background,
            exact_owned.density_background,
        ),
    )


@dataclass(frozen=True)
class LocalSlab2FieldObliqueCutWallRhs:
    geometry: LocalFciGeometry3D
    domain: LocalDomain3D
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    halo_exchange: HaloExchange3D
    physical_ghost_filler: PhysicalGhostCellFiller3D
    parameters: Fci2FieldRhsParameters
    curvature_coefficients_owned: jnp.ndarray
    global_shape: tuple[int, int, int]

    def __call__(
        self,
        state_owned: Fci2FieldState,
        stage_time: float | jax.Array,
        carry: None,
    ) -> tuple[Fci2FieldState, None, jnp.ndarray]:
        del carry
        prepared_stage = slab_mms._prepare_local_slab_stage_state(
            state_owned,
            self.coordinates_halo,
            self.domain,
            halo_exchange=self.halo_exchange,
            physical_ghost_filler=self.physical_ghost_filler,
            stage_time=stage_time,
        )
        state_halo = prepared_stage.state_halo
        density_halo = jnp.asarray(state_halo.density, dtype=jnp.float64)
        v_parallel_halo = jnp.asarray(state_halo.v_parallel, dtype=jnp.float64)
        density_background_halo = jnp.asarray(
            state_halo.density_background,
            dtype=jnp.float64,
        )
        phi_halo = jnp.log(
            jnp.maximum(density_halo, 1.0e-30)
            / jnp.maximum(density_background_halo, 1.0e-30)
        )
        fixture = _build_oblique_cut_wall_fixture(
            self.geometry,
            global_shape=self.global_shape,
        )
        wall_state = _mms_wall_values(fixture.stencil_geometry, stage_time)
        phi_wall = jnp.log(
            jnp.maximum(wall_state.density, 1.0e-30)
            / jnp.maximum(wall_state.density_background, 1.0e-30)
        )

        def build_stencil(field_halo: jnp.ndarray, wall_values: jnp.ndarray):
            return build_local_stencil_from_field(
                field_halo,
                self.geometry,
                StencilBuilderContext(
                    layout=self.domain.layout,
                    domain=self.domain,
                    cut_wall_stencil_dependencies=fixture.stencil_dependencies,
                    cut_wall_values=wall_values,
                ),
            )

        density_stencil = build_stencil(density_halo, wall_state.density)
        phi_stencil = build_stencil(phi_halo, phi_wall)
        v_parallel_stencil = build_stencil(v_parallel_halo, wall_state.v_parallel)

        density_owned = density_halo[self.domain.layout.owned_slices_cell]
        magnetic_field = jnp.maximum(
            jnp.asarray(self.geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
            1.0e-30,
        )
        rho_star_value = jnp.asarray(self.parameters.rho_star, dtype=jnp.float64)
        poisson_density = local_poisson_bracket_op(
            phi_stencil,
            density_stencil,
            self.geometry,
        )
        curvature_density = local_curvature_op(
            density_stencil,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
        )
        curvature_phi = local_curvature_op(
            phi_stencil,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
        )
        parallel_velocity_gradient = local_grad_parallel_op_direct(
            v_parallel_stencil,
            self.geometry,
        )
        poisson_v_parallel = local_poisson_bracket_op(
            phi_stencil,
            v_parallel_stencil,
            self.geometry,
        )

        density_rhs = (
            -(poisson_density / (rho_star_value * magnetic_field))
            + (2.0 / magnetic_field) * curvature_density
            - (2.0 * density_owned / magnetic_field) * curvature_phi
            - density_owned * parallel_velocity_gradient
        )
        v_parallel_rhs = -(poisson_v_parallel / (rho_star_value * magnetic_field))
        density_rhs = density_rhs + slab_mms._mms_local_density_source_from_coordinates(
            *self.coordinates_halo,
            stage_time,
            rho_star_value=rho_star_value,
        )[self.domain.layout.owned_slices_cell]
        v_parallel_rhs = (
            v_parallel_rhs
            + slab_mms._mms_local_v_parallel_source_from_coordinates(
                *self.coordinates_halo,
                stage_time,
                rho_star_value=rho_star_value,
            )[self.domain.layout.owned_slices_cell]
        )
        active_mask = self.geometry.active_cell_mask_owned & fixture.inside_mask
        return (
            Fci2FieldState(
                density=jnp.where(
                    active_mask,
                    jnp.asarray(density_rhs, dtype=jnp.float64),
                    0.0,
                ),
                v_parallel=jnp.where(
                    active_mask,
                    jnp.asarray(v_parallel_rhs, dtype=jnp.float64),
                    0.0,
                ),
                density_background=jnp.zeros(
                    self.domain.layout.owned_shape,
                    dtype=jnp.float64,
                ),
            ),
            None,
            jnp.zeros((3,), dtype=jnp.float64),
        )


def _masked_error_statistics(
    final_state: Fci2FieldState,
    geometry: FciGeometry3D,
    time: float,
    mask: jnp.ndarray,
) -> tuple[float, float, float]:
    exact = slab_mms._mms_exact_state(geometry, time)
    error = jnp.concatenate(
        [
            jnp.ravel(jnp.where(mask, jnp.abs(final_state.density - exact.density), 0.0)),
            jnp.ravel(
                jnp.where(mask, jnp.abs(final_state.v_parallel - exact.v_parallel), 0.0)
            ),
        ]
    )
    active = jnp.concatenate([jnp.ravel(mask), jnp.ravel(mask)])
    active_error = error[active]
    return (
        float(jnp.sqrt(jnp.mean(active_error**2))),
        float(jnp.median(active_error)),
        float(jnp.max(active_error)),
    )


def _global_active_mask(shape: tuple[int, int, int]) -> jnp.ndarray:
    geometry = slab_mms.build_slab_2field_geometry(*shape)
    x = geometry.grid.x.centers[:, None, None]
    y = geometry.grid.y.centers[None, :, None]
    z = geometry.grid.z.centers[None, None, :]
    xx, yy, zz = jnp.broadcast_arrays(x, y, z)
    del zz
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
    return _inside_from_coordinates(xx, yy) & margin


def simulate_mms_2field_slab_oblique_cutwall(
    geometry: FciGeometry3D,
    *,
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
    timestep: float | None = None,
    final_time: float = slab_mms.tf,
    rho_star_value: float = slab_mms.rho_star,
    show_progress: bool = False,
) -> tuple[Fci2FieldState, dict[str, float]]:
    slab_mms._assert_mms_slab_geometry(geometry)
    shard_counts = tuple(int(value) for value in shard_counts)
    slab_mms.assert_shape_divisible_by_shards(geometry.shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(geometry.shape, shard_counts)
    )
    if any(size < 4 for size in owned_shape):
        raise ValueError(f"local shard shape is too small for oblique wall: {owned_shape}")
    domain = slab_mms._build_local_domain(geometry.shape, halo_width, shard_counts)
    ghost_filler = slab_mms._build_ghost_filler(halo_width)
    parameters = Fci2FieldRhsParameters(rho_star=rho_star_value)
    curvature_coefficients_owned = jnp.zeros(owned_shape + (3,), dtype=jnp.float64)
    dt = (
        float(final_time) / float(slab_mms.num_steps)
        if timestep is None
        else float(timestep)
    )
    steps = int(round(float(final_time) / dt))
    dt = float(final_time) / float(steps)
    initial_state = slab_mms._mms_exact_state(geometry, 0.0)
    prebuilt_local_geometry = None
    prebuilt_coordinates_halo = None
    if shard_counts == (1, 1, 1):
        prebuilt_local_geometry = slab_mms.build_local_slab_2field_geometry(
            owned_shape,
            halo_width,
            global_shape=geometry.shape,
            shard_index=(0, 0, 0),
        )
        prebuilt_coordinates_halo = slab_mms._mms_local_coordinates(
            prebuilt_local_geometry
        )
    total_runtime = 0.0
    wall_step_times: list[float] = []

    with slab_mms.make_mesh_for_shard_counts(shard_counts) as mesh:
        state = slab_mms._put_state_on_mesh(initial_state, mesh)
        state_spec = slab_mms._state_partition_spec()

        def kernel(
            state_owned: Fci2FieldState,
            step_time: jax.Array,
            step_timestep: jax.Array,
        ) -> Fci2FieldState:
            if prebuilt_local_geometry is None:
                shard_index = tuple(lax.axis_index(name) for name in slab_mms._MESH_AXIS_NAMES)
                local_geometry = slab_mms.build_local_slab_2field_geometry(
                    owned_shape,
                    halo_width,
                    global_shape=geometry.shape,
                    shard_index=shard_index,
                )
                coordinates_halo = slab_mms._mms_local_coordinates(local_geometry)
            else:
                local_geometry = prebuilt_local_geometry
                coordinates_halo = prebuilt_coordinates_halo
            rhs = LocalSlab2FieldObliqueCutWallRhs(
                geometry=local_geometry,
                domain=domain,
                coordinates_halo=coordinates_halo,
                halo_exchange=HaloExchange3D(),
                physical_ghost_filler=ghost_filler,
                parameters=parameters,
                curvature_coefficients_owned=curvature_coefficients_owned,
                global_shape=geometry.shape,
            )
            step_result = Rk4Stepper(rhs)(
                state_owned,
                time=step_time,
                timestep=step_timestep,
                carry=None,
            )
            next_exact_halo = slab_mms._mms_local_exact_state_from_coordinates(
                *coordinates_halo,
                step_time + step_timestep,
            )
            next_exact_owned = Fci2FieldState(
                density=next_exact_halo.density[domain.layout.owned_slices_cell],
                v_parallel=next_exact_halo.v_parallel[domain.layout.owned_slices_cell],
                density_background=next_exact_halo.density_background[
                    domain.layout.owned_slices_cell
                ],
            )
            next_state = slab_mms._apply_local_owned_dirichlet_to_state(
                step_result.state,
                next_exact_owned,
                domain,
            )
            local_fixture = _build_oblique_cut_wall_fixture(
                local_geometry,
                global_shape=geometry.shape,
            )
            return _apply_exact_outside_state(
                next_state,
                next_exact_owned,
                local_fixture.inside_mask,
            )

        step_kernel = jax.jit(
            shard_map(
                kernel,
                mesh=mesh,
                in_specs=(state_spec, P(), P()),
                out_specs=state_spec,
                check_rep=False,
            )
        )
        progress_start = time_module.perf_counter()
        time_value = 0.0
        if show_progress:
            print(
                "slab_2field oblique cut-wall RK4 progress: "
                f"{slab_mms._format_progress_bar(0, steps, start_time=progress_start)}",
                end="",
                flush=True,
            )
        for step_index in range(steps):
            step_start = time_module.perf_counter()
            state = step_kernel(
                state,
                jnp.asarray(time_value, dtype=jnp.float64),
                jnp.asarray(dt, dtype=jnp.float64),
            )
            jax.block_until_ready(state.density)
            elapsed = time_module.perf_counter() - step_start
            total_runtime += elapsed
            wall_step_times.append(elapsed)
            time_value += dt
            if show_progress:
                print(
                    "\r"
                    "slab_2field oblique cut-wall RK4 progress: "
                    f"{slab_mms._format_progress_bar(step_index + 1, steps, start_time=progress_start)}",
                    end="",
                    flush=True,
                )
        if show_progress:
            print()
        final_state = slab_mms._gather_state_from_mesh(state)

    mean_step_runtime = total_runtime / float(steps) if steps else 0.0
    return final_state, {
        "total_runtime": float(total_runtime),
        "avg_step_runtime": float(mean_step_runtime),
        "prep_time": 0.0,
        "stencil_time": 0.0,
        "operator_time": 0.0,
    }


def run_slab_2field_oblique_cutwall_convergence(
    *,
    resolutions: list[int],
    shard_counts: tuple[int, int, int],
    halo_width: int,
    final_time: float = slab_mms.tf,
    base_steps: int = slab_mms.num_steps,
    plot: bool = False,
    plot_path: str | None = None,
    show_progress: bool = False,
) -> dict[str, object]:
    successful_resolutions: list[int] = []
    l2_errors: list[float] = []
    linf_errors: list[float] = []

    print()
    print("=" * 80)
    print("Slab 2-field MMS oblique cut-wall convergence")
    print("=" * 80)
    print(f"wall plane: x + {WALL_ALPHA:g} y = {WALL_C:g}")
    print(f"shard_counts = {tuple(int(value) for value in shard_counts)}")
    print()

    for resolution in resolutions:
        shape = _shape_from_resolution(resolution)
        slab_mms.assert_shape_divisible_by_shards(shape, shard_counts)
        geometry = slab_mms.build_slab_2field_geometry(*shape)
        steps = slab_mms._resolution_step_count(resolution, base_steps=base_steps)
        dt = float(final_time) / float(steps)
        start = time_module.perf_counter()
        final_state, timing_summary = simulate_mms_2field_slab_oblique_cutwall(
            geometry,
            shard_counts=shard_counts,
            halo_width=halo_width,
            final_time=final_time,
            timestep=dt,
            rho_star_value=slab_mms.rho_star,
            show_progress=show_progress,
        )
        elapsed = time_module.perf_counter() - start
        l2_error, median_error, linf_error = _masked_error_statistics(
            final_state,
            geometry,
            final_time,
            _global_active_mask(shape),
        )
        successful_resolutions.append(int(resolution))
        l2_errors.append(l2_error)
        linf_errors.append(linf_error)
        print(
            f"N={resolution}: shard_counts={shard_counts}, "
            f"steps={steps}, total_runtime={timing_summary['total_runtime']:.6e} s, "
            f"avg_step_runtime={timing_summary['avg_step_runtime']:.6e} s, "
            f"wall={elapsed:.6e} s, L2={l2_error:.6e}, "
            f"median={median_error:.6e}, Linf={linf_error:.6e}"
        )

    l2_order = slab_mms._estimate_convergence_order(successful_resolutions, l2_errors)
    linf_order = slab_mms._estimate_convergence_order(successful_resolutions, linf_errors)
    if l2_order is not None:
        print(f"slab_2field oblique cut-wall L2 convergence order: {l2_order:.6f}")
    if linf_order is not None:
        print(f"slab_2field oblique cut-wall Linf convergence order: {linf_order:.6f}")
    if plot and successful_resolutions:
        slab_mms._save_convergence_plot(
            successful_resolutions,
            l2_errors,
            linf_errors,
            title=f"2-field slab oblique cut-wall MMS convergence ({shard_counts})",
            output_path=Path(plot_path or "slab_2field_oblique_cutwall_convergence.png"),
        )
    return {
        "resolutions": successful_resolutions,
        "l2_errors": l2_errors,
        "linf_errors": linf_errors,
        "l2_order": l2_order,
        "linf_order": linf_order,
    }


def _build_single_local_fixture(shape: tuple[int, int, int], halo_width: int = 2):
    geometry = slab_mms.build_local_slab_2field_geometry(
        shape,
        halo_width,
        global_shape=shape,
    )
    domain = slab_mms._build_local_domain(
        shape,
        halo_width,
        (1, 1, 1),
        mesh_axis_names=(None, None, None),
    )
    return geometry, domain, _build_oblique_cut_wall_fixture(geometry, global_shape=shape)


def test_oblique_slab_2field_cutwall_geometry_is_active_and_non_axis_aligned() -> None:
    _geometry, _domain, fixture = _build_single_local_fixture((12, 12, 12))

    assert int(jnp.sum(fixture.stencil_geometry.active)) > 0
    assert int(jnp.sum(fixture.flux_geometry.active)) > 0
    assert bool(jnp.any(fixture.stencil_geometry.stencil_axis == 0))
    assert bool(jnp.any(fixture.stencil_geometry.stencil_axis == 1))
    assert float(jnp.max(jnp.abs(fixture.flux_geometry.normal_contra[:, 1]))) > 0.0
    assert bool(jnp.all(fixture.stencil_geometry.distance > 0.0))
    assert int(jnp.sum(fixture.inside_mask)) > 0
    assert int(jnp.sum(~fixture.inside_mask)) > 0


def test_oblique_slab_2field_cutwall_rhs_isolates_inside_from_outside_values() -> None:
    shape = (12, 12, 12)
    halo_width = 2
    global_geometry = slab_mms.build_slab_2field_geometry(*shape)
    local_geometry, domain, fixture = _build_single_local_fixture(shape, halo_width)
    coordinates_halo = slab_mms._mms_local_coordinates(local_geometry)
    rhs = LocalSlab2FieldObliqueCutWallRhs(
        geometry=local_geometry,
        domain=domain,
        coordinates_halo=coordinates_halo,
        halo_exchange=HaloExchange3D(),
        physical_ghost_filler=slab_mms._build_ghost_filler(halo_width),
        parameters=Fci2FieldRhsParameters(rho_star=slab_mms.rho_star),
        curvature_coefficients_owned=jnp.zeros(shape + (3,), dtype=jnp.float64),
        global_shape=shape,
    )
    exact = slab_mms._mms_exact_state(global_geometry, 0.0)
    perturbed = Fci2FieldState(
        density=jnp.where(fixture.inside_mask, exact.density, exact.density + 7.0),
        v_parallel=jnp.where(
            fixture.inside_mask,
            exact.v_parallel,
            exact.v_parallel - 3.0,
        ),
        density_background=jnp.where(
            fixture.inside_mask,
            exact.density_background,
            exact.density_background + 5.0,
        ),
    )
    base_rhs, _, _ = rhs(exact, 0.0, None)
    perturbed_rhs, _, _ = rhs(perturbed, 0.0, None)
    mask = fixture.valid_mask
    assert float(jnp.max(jnp.abs(jnp.where(mask, base_rhs.density - perturbed_rhs.density, 0.0)))) < 1.0e-10
    assert float(jnp.max(jnp.abs(jnp.where(mask, base_rhs.v_parallel - perturbed_rhs.v_parallel, 0.0)))) < 1.0e-10


def test_oblique_slab_2field_mms_converges() -> None:
    results = run_slab_2field_oblique_cutwall_convergence(
        resolutions=[10, 14],
        shard_counts=(1, 1, 1),
        halo_width=2,
        final_time=0.02,
        base_steps=6,
    )
    l2_errors = list(results["l2_errors"])
    linf_errors = list(results["linf_errors"])
    assert len(l2_errors) == 2
    assert all(np.isfinite(l2_errors))
    assert all(np.isfinite(linf_errors))
    assert l2_errors[-1] < l2_errors[0]


def test_oblique_slab_2field_conservative_wall_diagnostic_is_localized() -> None:
    shape = (12, 12, 12)
    geometry, domain, fixture = _build_single_local_fixture(shape)
    exact = slab_mms._mms_local_exact_state(geometry, 0.0)
    context = StencilBuilderContext(layout=geometry.layout, domain=domain)
    stencil = build_conservative_stencil_from_field(exact.density, geometry, context)
    wall_state = _mms_wall_values(fixture.flux_geometry, 0.0)
    cut_wall_bc = LocalCutWallBC3D(
        kind=jnp.full((fixture.flux_geometry.max_wall_faces,), BC_DIRICHLET, dtype=jnp.int32),
        value=wall_state.density,
        active=fixture.flux_geometry.active,
        max_wall_faces=fixture.flux_geometry.max_wall_faces,
    )
    actual = local_perp_laplacian_conservative_op(
        stencil,
        geometry,
        domain,
        regular_face_geometry=fixture.regular_face_geometry,
        cell_volume=geometry.cell_volume_geometry,
        cut_wall_geometry=fixture.flux_geometry,
        cut_wall_bc=cut_wall_bc,
    )
    baseline = local_perp_laplacian_conservative_op(
        stencil,
        geometry,
        domain,
        regular_face_geometry=fixture.regular_face_geometry,
        cell_volume=geometry.cell_volume_geometry,
        cut_wall_geometry=None,
        cut_wall_bc=None,
    )
    delta = actual - baseline
    owner_linf = float(jnp.max(jnp.abs(jnp.where(fixture.owner_mask, delta, 0.0))))
    outside_linf = float(jnp.max(jnp.abs(jnp.where(fixture.owner_mask, 0.0, delta))))
    assert owner_linf > 1.0e-12
    assert outside_linf < 1.0e-12


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
        description="Slab 2-field oblique cut-wall MMS convergence harness"
    )
    parser.add_argument("--resolutions", nargs="+", type=int, default=[20, 40, 80])
    parser.add_argument(
        "--shard-counts",
        nargs=3,
        type=int,
        metavar=("PX", "PY", "PZ"),
        default=(1, 1, 1),
    )
    parser.add_argument("--halo-width", type=int, default=2)
    parser.add_argument("--final-time", type=float, default=slab_mms.tf)
    parser.add_argument("--base-steps", type=int, default=slab_mms.num_steps)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--skip-runtime-info", action="store_true")
    args = parser.parse_args()

    if not args.skip_runtime_info:
        print_jax_runtime_info()
    run_slab_2field_oblique_cutwall_convergence(
        resolutions=[int(value) for value in args.resolutions],
        shard_counts=tuple(int(value) for value in args.shard_counts),
        halo_width=int(args.halo_width),
        final_time=float(args.final_time),
        base_steps=int(args.base_steps),
        plot=bool(args.plot),
        plot_path=args.plot_path,
        show_progress=bool(args.show_progress),
    )


if __name__ == "__main__":
    main()
