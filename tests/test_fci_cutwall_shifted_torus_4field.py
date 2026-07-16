"""Shifted-torus four-field MMS tests with a closed embedded cut-wall box."""

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
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

from jax_drb.geometry import (
    FciGeometry3D,
    LocalCoordinateStencilDependencyMap3D,
    LocalCoordinateStencilLocalDependencyTable,
    LocalDomain3D,
    LocalFciGeometry3D,
    LocalRegularFaceGeometry3D,
    StencilBuilderContext,
    build_local_conservative_stencil_from_field,
    build_local_stencil_from_field,
)
from jax_drb.native import Fci4FieldRhsParameters, Fci4FieldState, SpmdGmresConfig
from jax_drb.native.fci_boundaries import (
    BC_DIRICHLET,
    LocalBoundaryFaceBC3D,
    LocalCutWallBC3D,
    LocalCutWallGeometry3D,
)
from jax_drb.native.fci_halo import (
    HaloExchange3D,
    PhysicalGhostCellFiller3D,
    TopologyHaloFiller3D,
    LocalPeriodicTopologyRule3D,
)
from jax_drb.native.fci_model import inject_owned_state_to_halo
from jax_drb.native.fci_operators import (
    LocalPerpLaplacianInverseSolver,
    local_curvature_op,
    local_grad_parallel_op_direct,
    local_poisson_bracket_op,
)


jax.config.update("jax_enable_x64", True)

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))
import shifted_torus_4field_mms_helpers as shifted_mms  # noqa: E402
MESH_AXIS_NAMES = ("x", "y", "z")


def assert_shape_divisible_by_shards(*args, **kwargs):
    from mms_domain_decomp_helpers import assert_shape_divisible_by_shards as impl

    return impl(*args, **kwargs)


def build_shifted_torus_local_domain(*args, **kwargs):
    from mms_domain_decomp_helpers import build_shifted_torus_local_domain as impl

    return impl(*args, **kwargs)


def build_shifted_torus_local_geometry(*args, **kwargs):
    from mms_domain_decomp_helpers import build_shifted_torus_local_geometry as impl

    return impl(*args, **kwargs)


def expand_local_shard_pytree(*args, **kwargs):
    from mms_domain_decomp_helpers import expand_local_shard_pytree as impl

    return impl(*args, **kwargs)


def extract_local_shard_pytree(*args, **kwargs):
    from mms_domain_decomp_helpers import extract_local_shard_pytree as impl

    return impl(*args, **kwargs)


def local_shard_pytree_partition_spec(*args, **kwargs):
    from mms_domain_decomp_helpers import local_shard_pytree_partition_spec as impl

    return impl(*args, **kwargs)


def make_mesh_for_shard_counts(*args, **kwargs):
    from mms_domain_decomp_helpers import make_mesh_for_shard_counts as impl

    return impl(*args, **kwargs)


BOX_X_FRACTION_RANGE = (0.25, 0.75)
BOX_THETA_CENTER = 1.5 * np.pi
BOX_THETA_HALF_WIDTH = 0.35
BOX_ZETA_RANGE = (0.45, 4.25)


@dataclass(frozen=True)
class _ShiftedTorusCutWallFixture:
    dependencies: LocalCoordinateStencilDependencyMap3D
    geometry: LocalCutWallGeometry3D
    regular_face_geometry: LocalRegularFaceGeometry3D
    active_owner_mask: jnp.ndarray
    plane_id: jnp.ndarray
    wall_x: jnp.ndarray
    wall_theta_shift: jnp.ndarray
    wall_theta: jnp.ndarray
    wall_zeta: jnp.ndarray


def _wrap_angle(value: jnp.ndarray) -> jnp.ndarray:
    return (value + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def _shape_from_resolution(resolution: int) -> tuple[int, int, int]:
    n = int(resolution)
    return (n, n, n)


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
    normal_contra: jnp.ndarray,
    area_covector: jnp.ndarray,
    distance: jnp.ndarray,
    J: jnp.ndarray,
    g_contra: jnp.ndarray,
    g_cov: jnp.ndarray,
    B_contra: jnp.ndarray,
    Bmag: jnp.ndarray,
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
    object.__setattr__(obj, "J", jnp.asarray(J, dtype=jnp.float64))
    object.__setattr__(obj, "g_contra", jnp.asarray(g_contra, dtype=jnp.float64))
    object.__setattr__(obj, "g_cov", jnp.asarray(g_cov, dtype=jnp.float64))
    object.__setattr__(obj, "B_contra", jnp.asarray(B_contra, dtype=jnp.float64))
    object.__setattr__(obj, "Bmag", jnp.asarray(Bmag, dtype=jnp.float64))
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


def _local_cut_wall_bc(value: jnp.ndarray, active: jnp.ndarray) -> LocalCutWallBC3D:
    value = jnp.asarray(value, dtype=jnp.float64)
    active = jnp.asarray(active, dtype=bool)
    return LocalCutWallBC3D(
        kind=jnp.full(value.shape, int(BC_DIRICHLET), dtype=jnp.int32),
        value=value,
        active=active,
        max_wall_faces=int(value.size),
    )


def _shifted_torus_cartesian_from_logical(
    x: jnp.ndarray,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x_mid = 0.5 * (float(shifted_mms.x_min) + float(shifted_mms.x_max))
    theta_shift = theta + float(shifted_mms.sigma) * (x - x_mid)
    radius = (
        float(shifted_mms.r0)
        + float(shifted_mms.alpha_value) * x
        + x * jnp.cos(theta_shift)
    )
    return radius * jnp.cos(zeta), radius * jnp.sin(zeta), x * jnp.sin(theta_shift)


def _build_shifted_torus_cut_wall_fixture(
    geometry: LocalFciGeometry3D,
    *,
    global_shape: tuple[int, int, int],
) -> _ShiftedTorusCutWallFixture:
    del global_shape
    nx, ny, nz = geometry.owned_shape
    x_values = jnp.asarray(geometry.grid.x.centers_owned, dtype=jnp.float64)
    theta_values = jnp.asarray(geometry.grid.y.centers_owned, dtype=jnp.float64)
    z_values = jnp.asarray(geometry.grid.z.centers_owned, dtype=jnp.float64)
    radial_span = float(shifted_mms.x_max) - float(shifted_mms.x_min)
    # This test uses a closed volume in logical computational coordinates.
    # ``theta`` here is the unshifted logical poloidal coordinate; metric and
    # exact-field sampling below convert it to the shifted physical angle.
    box_ranges = (
        (
            float(shifted_mms.x_min) + BOX_X_FRACTION_RANGE[0] * radial_span,
            float(shifted_mms.x_min) + BOX_X_FRACTION_RANGE[1] * radial_span,
        ),
        (
            float(BOX_THETA_CENTER - BOX_THETA_HALF_WIDTH),
            float(BOX_THETA_CENTER + BOX_THETA_HALF_WIDTH),
        ),
        (float(BOX_ZETA_RANGE[0]), float(BOX_ZETA_RANGE[1])),
    )
    axis_values = (x_values, theta_values, z_values)
    axis_sizes = (nx, ny, nz)

    def _axis_spacing(axis: int, owner_i, owner_j, owner_k):
        if axis == 0:
            return jnp.asarray(geometry.spacing.dx_owned, dtype=jnp.float64)[owner_i, owner_j, owner_k]
        if axis == 1:
            return jnp.asarray(geometry.spacing.dy_owned, dtype=jnp.float64)[owner_i, owner_j, owner_k]
        return jnp.asarray(geometry.spacing.dz_owned, dtype=jnp.float64)[owner_i, owner_j, owner_k]

    def _wall_metric_payload(wall_x, wall_theta):
        x_mid = 0.5 * (float(shifted_mms.x_min) + float(shifted_mms.x_max))
        theta_shift = wall_theta + float(shifted_mms.sigma) * (wall_x - x_mid)
        cos_theta = jnp.cos(theta_shift)
        sin_theta = jnp.sin(theta_shift)
        radius = (
            float(shifted_mms.r0)
            + float(shifted_mms.alpha_value) * wall_x
            + wall_x * cos_theta
        )
        q_value = 1.0 + float(shifted_mms.alpha_value) * cos_theta
        jacobian = radius * wall_x * q_value
        zeros = jnp.zeros_like(wall_x)
        g11 = 1.0 / q_value**2
        g12 = float(shifted_mms.alpha_value) * sin_theta / (wall_x * q_value**2)
        g22 = (1.0 + 2.0 * float(shifted_mms.alpha_value) * cos_theta + float(shifted_mms.alpha_value) ** 2) / (
            wall_x**2 * q_value**2
        )
        g33 = 1.0 / radius**2
        g_11 = 1.0 + 2.0 * float(shifted_mms.alpha_value) * cos_theta + float(shifted_mms.alpha_value) ** 2
        g_12 = -float(shifted_mms.alpha_value) * wall_x * sin_theta
        g_22 = wall_x**2
        g_33 = radius**2
        g_contra = jnp.stack(
            (
                jnp.stack((g11, g12, zeros), axis=-1),
                jnp.stack((g12, g22, zeros), axis=-1),
                jnp.stack((zeros, zeros, g33), axis=-1),
            ),
            axis=-2,
        )
        g_cov = jnp.stack(
            (
                jnp.stack((g_11, g_12, zeros), axis=-1),
                jnp.stack((g_12, g_22, zeros), axis=-1),
                jnp.stack((zeros, zeros, g_33), axis=-1),
            ),
            axis=-2,
        )
        b_contra = jnp.stack(
            (
                zeros,
                float(shifted_mms.iota) * float(shifted_mms.c_phi) / jacobian,
                float(shifted_mms.c_phi) / jacobian,
            ),
            axis=-1,
        )
        bmag = jnp.sqrt(jnp.einsum("...i,...ij,...j->...", b_contra, g_cov, b_contra))
        return theta_shift, jacobian, g_contra, g_cov, b_contra, bmag

    rows: dict[str, list[jnp.ndarray]] = {name: [] for name in (
        "owner_i", "owner_j", "owner_k", "center", "normal", "area", "distance",
        "jacobian", "g_contra", "g_cov", "b_contra", "bmag", "sign", "active",
        "axis", "side", "stencil_distance", "plane_id", "wall_x",
        "wall_theta_shift", "wall_theta", "wall_zeta",
    )}
    face_closed_counts = [
        jnp.zeros((nx + 1, ny, nz), dtype=jnp.int32),
        jnp.zeros((nx, ny + 1, nz), dtype=jnp.int32),
        jnp.zeros((nx, ny, nz + 1), dtype=jnp.int32),
    ]

    def _append_surface(axis: int, surface_index: int, plane_value: float) -> None:
        axis_array = axis_values[axis]
        lower_index = jnp.searchsorted(axis_array, plane_value, side="right") - 1
        upper_index = lower_index + 1
        local_active = (
            (plane_value > axis_array[0])
            & (plane_value < axis_array[-1])
            & (lower_index >= 0)
            & (upper_index < axis_sizes[axis])
        )
        tangential_axes = [candidate for candidate in range(3) if candidate != axis]
        tangential_grids = jnp.meshgrid(
            jnp.arange(axis_sizes[tangential_axes[0]], dtype=jnp.int32),
            jnp.arange(axis_sizes[tangential_axes[1]], dtype=jnp.int32),
            indexing="ij",
        )
        for side_value, owner_axis_index, normal_sign in (
            (1, lower_index, 1.0),
            (0, upper_index, -1.0),
        ):
            owner_axis_index = jnp.clip(owner_axis_index, 0, axis_sizes[axis] - 1).astype(jnp.int32)
            owner_axes = [None, None, None]
            owner_axes[axis] = owner_axis_index
            owner_axes[tangential_axes[0]] = tangential_grids[0]
            owner_axes[tangential_axes[1]] = tangential_grids[1]
            owner_i, owner_j, owner_k = jnp.broadcast_arrays(*owner_axes)
            coords = [x_values[owner_i], theta_values[owner_j], z_values[owner_k]]
            coords[axis] = jnp.full_like(coords[axis], float(plane_value), dtype=jnp.float64)
            wall_x, wall_theta, wall_zeta = coords
            active = jnp.ones_like(wall_x, dtype=bool) & local_active
            for tangential_axis in tangential_axes:
                lower, upper = box_ranges[tangential_axis]
                active = active & (coords[tangential_axis] >= lower) & (coords[tangential_axis] <= upper)

            owner_coordinate = axis_array[owner_axis_index]
            coordinate_distance = jnp.broadcast_to(
                jnp.maximum(jnp.abs(float(plane_value) - owner_coordinate), 1.0e-12),
                wall_x.shape,
            )
            theta_shift, jacobian, g_contra, g_cov, b_contra, bmag = _wall_metric_payload(wall_x, wall_theta)
            normal_covector = jnp.zeros(wall_x.shape + (3,), dtype=jnp.float64).at[..., axis].set(float(normal_sign))
            normal_scale = jnp.sqrt(
                jnp.maximum(
                    jnp.einsum("...i,...ij,...j->...", normal_covector, g_contra, normal_covector),
                    1.0e-30,
                )
            )
            normal_contra = jnp.einsum("...ij,...j->...i", g_contra, normal_covector) / normal_scale[..., None]
            normal_distance = coordinate_distance / normal_scale
            area_covector = jnp.zeros_like(normal_contra).at[..., axis].set(
                1.0 / jnp.maximum(_axis_spacing(axis, owner_i, owner_j, owner_k), 1.0e-30)
            )
            row_count = int(owner_i.size)
            rows["owner_i"].append(owner_i.reshape((-1,)))
            rows["owner_j"].append(owner_j.reshape((-1,)))
            rows["owner_k"].append(owner_k.reshape((-1,)))
            rows["center"].append(jnp.stack((wall_x, wall_theta, wall_zeta), axis=-1).reshape((-1, 3)))
            rows["normal"].append(normal_contra.reshape((-1, 3)))
            rows["area"].append(area_covector.reshape((-1, 3)))
            rows["distance"].append(normal_distance.reshape((-1,)))
            rows["jacobian"].append(jacobian.reshape((-1,)))
            rows["g_contra"].append(g_contra.reshape((-1, 3, 3)))
            rows["g_cov"].append(g_cov.reshape((-1, 3, 3)))
            rows["b_contra"].append(b_contra.reshape((-1, 3)))
            rows["bmag"].append(bmag.reshape((-1,)))
            rows["sign"].append(jnp.full((row_count,), float(normal_sign), dtype=jnp.float64))
            rows["active"].append(active.reshape((-1,)))
            rows["axis"].append(jnp.full((row_count,), axis, dtype=jnp.int32))
            rows["side"].append(jnp.full((row_count,), side_value, dtype=jnp.int32))
            rows["stencil_distance"].append(coordinate_distance.reshape((-1,)))
            rows["plane_id"].append(jnp.full((row_count,), axis * 2 + surface_index, dtype=jnp.int32))
            rows["wall_x"].append(wall_x.reshape((-1,)))
            rows["wall_theta_shift"].append(theta_shift.reshape((-1,)))
            rows["wall_theta"].append(wall_theta.reshape((-1,)))
            rows["wall_zeta"].append(wall_zeta.reshape((-1,)))

            face_index = owner_axis_index + int(side_value)
            if axis == 0:
                face_closed_counts[0] = face_closed_counts[0].at[face_index, owner_j, owner_k].add(active.astype(jnp.int32))
            elif axis == 1:
                face_closed_counts[1] = face_closed_counts[1].at[owner_i, face_index, owner_k].add(active.astype(jnp.int32))
            else:
                face_closed_counts[2] = face_closed_counts[2].at[owner_i, owner_j, face_index].add(active.astype(jnp.int32))

    for axis, (lower, upper) in enumerate(box_ranges):
        _append_surface(axis, 0, lower)
        _append_surface(axis, 1, upper)

    owner_i = jnp.concatenate(rows["owner_i"], axis=0)
    owner_j = jnp.concatenate(rows["owner_j"], axis=0)
    owner_k_flat = jnp.concatenate(rows["owner_k"], axis=0)
    active_flat = jnp.concatenate(rows["active"], axis=0)
    geometry_payload = _unchecked_cut_wall_geometry(
        owner_i=owner_i,
        owner_j=owner_j,
        owner_k=owner_k_flat,
        center=jnp.concatenate(rows["center"], axis=0),
        normal_contra=jnp.concatenate(rows["normal"], axis=0),
        area_covector=jnp.concatenate(rows["area"], axis=0),
        distance=jnp.concatenate(rows["distance"], axis=0),
        J=jnp.concatenate(rows["jacobian"], axis=0),
        g_contra=jnp.concatenate(rows["g_contra"], axis=0),
        g_cov=jnp.concatenate(rows["g_cov"], axis=0),
        B_contra=jnp.concatenate(rows["b_contra"], axis=0),
        Bmag=jnp.concatenate(rows["bmag"], axis=0),
        sign=jnp.concatenate(rows["sign"], axis=0),
        active=active_flat,
        stencil_axis=jnp.concatenate(rows["axis"], axis=0),
        stencil_side=jnp.concatenate(rows["side"], axis=0),
        stencil_distance=jnp.concatenate(rows["stencil_distance"], axis=0),
    )
    target_flat = (owner_i * ny + owner_j) * nz + owner_k_flat
    dependencies = _unchecked_coordinate_dependencies(
        geometry.layout,
        target_flat=target_flat,
        axis=jnp.concatenate(rows["axis"], axis=0),
        side=jnp.concatenate(rows["side"], axis=0),
        distance=jnp.concatenate(rows["stencil_distance"], axis=0),
        active=active_flat,
    )
    regular_face_geometry = LocalRegularFaceGeometry3D(
        layout=geometry.layout,
        x_area=geometry.regular_face_geometry.x_area,
        y_area=geometry.regular_face_geometry.y_area,
        z_area=geometry.regular_face_geometry.z_area,
        x_area_fraction=geometry.regular_face_geometry.x_area_fraction,
        y_area_fraction=geometry.regular_face_geometry.y_area_fraction,
        z_area_fraction=geometry.regular_face_geometry.z_area_fraction,
        x_open_mask=geometry.regular_face_geometry.x_open_mask & (face_closed_counts[0] == 0),
        y_open_mask=geometry.regular_face_geometry.y_open_mask & (face_closed_counts[1] == 0),
        z_open_mask=geometry.regular_face_geometry.z_open_mask & (face_closed_counts[2] == 0),
    )
    active_owner_count = jnp.zeros(geometry.owned_shape, dtype=jnp.int32).at[
        owner_i,
        owner_j,
        owner_k_flat,
    ].add(active_flat.astype(jnp.int32))
    active_owner_mask = active_owner_count > 0
    return _ShiftedTorusCutWallFixture(
        dependencies=dependencies,
        geometry=geometry_payload,
        regular_face_geometry=regular_face_geometry,
        active_owner_mask=active_owner_mask,
        plane_id=jnp.concatenate(rows["plane_id"], axis=0),
        wall_x=jnp.concatenate(rows["wall_x"], axis=0),
        wall_theta_shift=jnp.concatenate(rows["wall_theta_shift"], axis=0),
        wall_theta=jnp.concatenate(rows["wall_theta"], axis=0),
        wall_zeta=jnp.concatenate(rows["wall_zeta"], axis=0),
    )


def _cut_wall_exact_values(
    fixture: _ShiftedTorusCutWallFixture,
    stage_time: float | jax.Array,
) -> tuple[Fci4FieldState, jnp.ndarray]:
    coords = (
        fixture.wall_x,
        fixture.wall_theta_shift,
        fixture.wall_theta,
        fixture.wall_zeta,
    )
    density = shifted_mms._shifted_torus_local_density_derivatives(coords, stage_time)[0]
    omega = shifted_mms._shifted_torus_local_omega_and_derivatives(coords, stage_time)[0]
    v_ion = shifted_mms._shifted_torus_local_v_ion_parallel_derivatives(coords, stage_time)[0]
    v_electron = shifted_mms._shifted_torus_local_v_electron_parallel_derivatives(coords, stage_time)[0]
    phi = shifted_mms._shifted_torus_local_phi_derivatives(coords, stage_time)[0]
    return (
        Fci4FieldState(
            density=jnp.asarray(density, dtype=jnp.float64),
            omega=jnp.asarray(omega, dtype=jnp.float64),
            v_ion_parallel=jnp.asarray(v_ion, dtype=jnp.float64),
            v_electron_parallel=jnp.asarray(v_electron, dtype=jnp.float64),
        ),
        jnp.asarray(phi, dtype=jnp.float64),
    )


@dataclass(frozen=True)
class LocalShiftedTorus4FieldCutWallRhs:
    geometry: LocalFciGeometry3D
    domain: LocalDomain3D
    halo_exchange: HaloExchange3D
    topology_filler: TopologyHaloFiller3D
    physical_ghost_filler: PhysicalGhostCellFiller3D
    parameters: Fci4FieldRhsParameters
    curvature_coefficients_owned: jnp.ndarray
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    gmres_config: SpmdGmresConfig
    global_shape: tuple[int, int, int]

    def _prepare_phi_halo(
        self,
        phi_owned: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
    ) -> jnp.ndarray:
        phi_halo = inject_owned_state_to_halo(
            Fci4FieldState(
                density=phi_owned,
                omega=phi_owned,
                v_ion_parallel=phi_owned,
                v_electron_parallel=phi_owned,
            ),
            self.domain.layout,
        ).density
        phi_halo = self.topology_filler(
            self.halo_exchange(phi_halo, self.domain),
            self.domain,
        )
        return self.physical_ghost_filler(phi_halo, self.domain, face_bc)

    def evaluate_stage(
        self,
        state_owned: Fci4FieldState,
        stage_data: shifted_mms._ShiftedTorus4FieldStageData,
        phi_guess_owned: jnp.ndarray | None,
        *,
        phi_wall_offset: float | jax.Array = 0.0,
        return_wall_diagnostic: bool = False,
    ) -> tuple[Fci4FieldState, jnp.ndarray, jnp.ndarray]:
        prepared = shifted_mms._prepare_local_shifted_torus_4field_stage_state(
            state_owned,
            stage_data,
            self.domain,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
        )
        face_bc = prepared.boundary_data.face_bc
        state_halo = prepared.state_halo
        fixture = _build_shifted_torus_cut_wall_fixture(
            self.geometry,
            global_shape=self.global_shape,
        )
        stage_time = _infer_stage_time_from_phi(stage_data, self.geometry)
        wall_state, phi_wall = _cut_wall_exact_values(fixture, stage_time)
        phi_wall = phi_wall + jnp.asarray(phi_wall_offset, dtype=jnp.float64)
        phi_cut_wall_bc = _local_cut_wall_bc(phi_wall, fixture.geometry.active)

        omega_owned = jnp.asarray(
            state_halo.omega[self.domain.layout.owned_slices_cell],
            dtype=jnp.float64,
        )
        phi_solver = LocalPerpLaplacianInverseSolver(
            geometry=self.geometry,
            domain=self.domain,
            stencil_builder=build_local_conservative_stencil_from_field,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
            face_projectors=self.face_projectors,
            regular_face_geometry=fixture.regular_face_geometry,
            cut_wall_geometry=fixture.geometry,
            cut_wall_bc=phi_cut_wall_bc,
            face_bc=face_bc.phi,
            config=self.gmres_config,
        )
        phi_owned = phi_solver(
            -omega_owned,
            guess_owned=phi_guess_owned,
            phi_lift_owned=stage_data.phi_halo[self.domain.layout.owned_slices_cell],
        )
        phi_halo = self._prepare_phi_halo(phi_owned, face_bc.phi)
        _, phi_wall = _cut_wall_exact_values(fixture, stage_time)
        phi_wall = phi_wall + jnp.asarray(phi_wall_offset, dtype=jnp.float64)

        context = StencilBuilderContext(
            layout=self.domain.layout,
            domain=self.domain,
            cut_wall_stencil_dependencies=fixture.dependencies,
        )

        def build_stencil(field_halo: jnp.ndarray, values: jnp.ndarray):
            return build_local_stencil_from_field(
                field_halo,
                self.geometry,
                StencilBuilderContext(
                    layout=context.layout,
                    domain=context.domain,
                    cut_wall_stencil_dependencies=fixture.dependencies,
                    cut_wall_values=values,
                ),
            )

        density_stencil = build_stencil(state_halo.density, wall_state.density)
        omega_stencil = build_stencil(state_halo.omega, wall_state.omega)
        v_ion_stencil = build_stencil(state_halo.v_ion_parallel, wall_state.v_ion_parallel)
        v_electron_stencil = build_stencil(
            state_halo.v_electron_parallel,
            wall_state.v_electron_parallel,
        )
        phi_stencil = build_stencil(phi_halo, phi_wall)

        density_owned = jnp.asarray(
            state_halo.density[self.domain.layout.owned_slices_cell],
            dtype=jnp.float64,
        )
        density_safe = jnp.maximum(density_owned, 1.0e-30)
        bmag_owned = jnp.maximum(
            jnp.asarray(self.geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
            1.0e-30,
        )
        rho_star_value = jnp.asarray(self.parameters.rho_star, dtype=jnp.float64)
        te = jnp.asarray(self.parameters.Te, dtype=jnp.float64)
        mi_over_me_value = jnp.asarray(self.parameters.mi_over_me, dtype=jnp.float64)

        poisson_density = local_poisson_bracket_op(phi_stencil, density_stencil, self.geometry)
        poisson_omega = local_poisson_bracket_op(phi_stencil, omega_stencil, self.geometry)
        poisson_v_ion = local_poisson_bracket_op(phi_stencil, v_ion_stencil, self.geometry)
        poisson_v_electron = local_poisson_bracket_op(phi_stencil, v_electron_stencil, self.geometry)
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
        grad_parallel_density = local_grad_parallel_op_direct(density_stencil, self.geometry)
        grad_parallel_phi = local_grad_parallel_op_direct(phi_stencil, self.geometry)
        grad_parallel_v_ion = local_grad_parallel_op_direct(v_ion_stencil, self.geometry)
        grad_parallel_v_electron = local_grad_parallel_op_direct(v_electron_stencil, self.geometry)

        density_poisson_term = -(poisson_density / (rho_star_value * bmag_owned))
        density_curvature_term = (2.0 * te / bmag_owned) * curvature_density
        density_phi_curvature_term = -(2.0 * density_owned / bmag_owned) * curvature_phi
        density_parallel_term = -density_owned * grad_parallel_v_electron
        density_rhs = (
            density_poisson_term
            + density_curvature_term
            + density_phi_curvature_term
            + density_parallel_term
        )
        omega_poisson_term = -(poisson_omega / (rho_star_value * bmag_owned))
        omega_parallel_term = (
            (bmag_owned * bmag_owned / density_safe)
            * (grad_parallel_v_ion - grad_parallel_v_electron)
        )
        omega_curvature_term = (2.0 * bmag_owned * te / density_safe) * curvature_density
        omega_rhs = (
            omega_poisson_term
            + omega_parallel_term
            + omega_curvature_term
        )
        v_ion_poisson_term = -(poisson_v_ion / (rho_star_value * bmag_owned))
        v_ion_pressure_term = -(te / density_safe) * grad_parallel_density
        v_ion_rhs = (
            v_ion_poisson_term
            + v_ion_pressure_term
        )
        v_electron_poisson_term = -(poisson_v_electron / (rho_star_value * bmag_owned))
        v_electron_parallel_phi_term = mi_over_me_value * grad_parallel_phi
        v_electron_pressure_term = -mi_over_me_value * (te / density_safe) * grad_parallel_density
        v_electron_rhs = (
            v_electron_poisson_term
            + v_electron_parallel_phi_term
            + v_electron_pressure_term
        )
        owned = self.domain.layout.owned_slices_cell
        source_density = stage_data.source_halo.density[owned]
        source_omega = stage_data.source_halo.omega[owned]
        source_v_ion = stage_data.source_halo.v_ion_parallel[owned]
        source_v_electron = stage_data.source_halo.v_electron_parallel[owned]
        rhs = Fci4FieldState(
            density=density_rhs + source_density,
            omega=omega_rhs + source_omega,
            v_ion_parallel=v_ion_rhs + source_v_ion,
            v_electron_parallel=v_electron_rhs + source_v_electron,
        )

        def term_stats(value: jnp.ndarray) -> jnp.ndarray:
            finite = jnp.isfinite(value)
            nonfinite_count = jnp.sum(~finite).astype(jnp.float64)
            maxabs = jnp.max(jnp.where(finite, jnp.abs(value), 0.0))
            return jnp.stack((nonfinite_count, maxabs))

        diagnostic = jnp.asarray(
            (
                term_stats(phi_owned),
                term_stats(density_owned),
                term_stats(1.0 / density_safe),
                term_stats(poisson_density),
                term_stats(poisson_omega),
                term_stats(poisson_v_ion),
                term_stats(poisson_v_electron),
                term_stats(curvature_density),
                term_stats(curvature_phi),
                term_stats(grad_parallel_density),
                term_stats(grad_parallel_phi),
                term_stats(grad_parallel_v_ion),
                term_stats(grad_parallel_v_electron),
                term_stats(density_poisson_term),
                term_stats(density_curvature_term),
                term_stats(density_phi_curvature_term),
                term_stats(density_parallel_term),
                term_stats(omega_poisson_term),
                term_stats(omega_parallel_term),
                term_stats(omega_curvature_term),
                term_stats(v_ion_poisson_term),
                term_stats(v_ion_pressure_term),
                term_stats(v_electron_poisson_term),
                term_stats(v_electron_parallel_phi_term),
                term_stats(v_electron_pressure_term),
                term_stats(source_density),
                term_stats(source_omega),
                term_stats(source_v_ion),
                term_stats(source_v_electron),
                term_stats(rhs.density),
                term_stats(rhs.omega),
                term_stats(rhs.v_ion_parallel),
                term_stats(rhs.v_electron_parallel),
            )
        )
        if return_wall_diagnostic:
            return rhs, phi_owned, diagnostic
        return rhs, phi_owned, diagnostic


def _infer_stage_time_from_phi(
    stage_data: shifted_mms._ShiftedTorus4FieldStageData,
    geometry: LocalFciGeometry3D,
) -> jnp.ndarray:
    # The stage object intentionally mirrors the existing MMS helper and does
    # not carry time.  For wall sampling, the exact phi halo is the source of
    # truth; the MMS phi time dependence is a scalar cos(Omega*t).  Recovering
    # it keeps wall values synchronized with the stage without changing the
    # shared stage-data type.
    coords = shifted_mms._shifted_torus_local_coordinates(geometry)
    phi_at_zero = shifted_mms._shifted_torus_local_phi_derivatives(coords, 0.0)[0]
    owned = geometry.layout.owned_slices_cell
    denom = jnp.sum(phi_at_zero[owned] * phi_at_zero[owned]) + 1.0e-30
    cos_time = jnp.clip(jnp.sum(stage_data.phi_halo[owned] * phi_at_zero[owned]) / denom, -1.0, 1.0)
    return jnp.arccos(cos_time) / float(shifted_mms.Omega)


def _state_error_statistics(
    actual: Fci4FieldState,
    expected: Fci4FieldState,
) -> dict[str, tuple[float, float, float]]:
    return shifted_mms._state_error_statistics(actual, expected)


def _solid_box_center_mask(geometry: FciGeometry3D, *, margin_cells: int = 1) -> jnp.ndarray:
    """Cell-center mask for the embedded solid obstacle interior."""

    radial_span = float(shifted_mms.x_max) - float(shifted_mms.x_min)
    x_min = float(shifted_mms.x_min) + BOX_X_FRACTION_RANGE[0] * radial_span
    x_max = float(shifted_mms.x_min) + BOX_X_FRACTION_RANGE[1] * radial_span
    theta_min = BOX_THETA_CENTER - BOX_THETA_HALF_WIDTH
    theta_max = BOX_THETA_CENTER + BOX_THETA_HALF_WIDTH
    zeta_min, zeta_max = BOX_ZETA_RANGE
    x_centers = jnp.asarray(geometry.grid.x.centers, dtype=jnp.float64)
    theta_centers = jnp.asarray(geometry.grid.y.centers, dtype=jnp.float64)
    zeta_centers = jnp.asarray(geometry.grid.z.centers, dtype=jnp.float64)
    x, theta, zeta = jnp.meshgrid(
        x_centers,
        theta_centers,
        zeta_centers,
        indexing="ij",
    )
    mask = (
        (x > x_min)
        & (x < x_max)
        & (theta > theta_min)
        & (theta < theta_max)
        & (zeta > zeta_min)
        & (zeta < zeta_max)
    )
    if margin_cells <= 0:
        return mask
    i, j, k = jnp.meshgrid(
        jnp.arange(geometry.shape[0], dtype=jnp.int32),
        jnp.arange(geometry.shape[1], dtype=jnp.int32),
        jnp.arange(geometry.shape[2], dtype=jnp.int32),
        indexing="ij",
    )
    coordinate_margin = (
        (i >= margin_cells)
        & (i < geometry.shape[0] - margin_cells)
        & (j >= margin_cells)
        & (j < geometry.shape[1] - margin_cells)
        & (k >= margin_cells)
        & (k < geometry.shape[2] - margin_cells)
    )
    dx = jnp.abs(x_centers[1] - x_centers[0])
    dtheta = jnp.abs(theta_centers[1] - theta_centers[0])
    dzeta = jnp.abs(zeta_centers[1] - zeta_centers[0])
    face_margin = (
        (x > x_min + margin_cells * dx)
        & (x < x_max - margin_cells * dx)
        & (theta > theta_min + margin_cells * dtheta)
        & (theta < theta_max - margin_cells * dtheta)
        & (zeta > zeta_min + margin_cells * dzeta)
        & (zeta < zeta_max - margin_cells * dzeta)
    )
    return mask & coordinate_margin & face_margin


def _masked_field_error_statistics(
    actual: jnp.ndarray,
    expected: jnp.ndarray,
    mask: jnp.ndarray,
) -> tuple[float, float, float]:
    error = jnp.asarray(actual - expected, dtype=jnp.float64)
    expected_array = jnp.asarray(expected, dtype=jnp.float64)
    mask_f = jnp.asarray(mask, dtype=jnp.float64)
    count = jnp.maximum(jnp.sum(mask_f), 1.0)
    masked_error = jnp.where(mask, error, 0.0)
    masked_expected = jnp.where(mask, expected_array, 0.0)
    l2 = float(jnp.sqrt(jnp.sum(jnp.square(masked_error)) / count))
    linf = float(jnp.max(jnp.abs(masked_error)))
    rel_l2 = float(
        jnp.sqrt(jnp.sum(jnp.square(masked_error)))
        / (jnp.sqrt(jnp.sum(jnp.square(masked_expected))) + 1.0e-30)
    )
    return l2, linf, rel_l2


def _masked_state_error_statistics(
    actual: Fci4FieldState,
    expected: Fci4FieldState,
    mask: jnp.ndarray,
) -> dict[str, tuple[float, float, float]]:
    return {
        "density": _masked_field_error_statistics(actual.density, expected.density, mask),
        "omega": _masked_field_error_statistics(actual.omega, expected.omega, mask),
        "v_ion_parallel": _masked_field_error_statistics(actual.v_ion_parallel, expected.v_ion_parallel, mask),
        "v_electron_parallel": _masked_field_error_statistics(
            actual.v_electron_parallel,
            expected.v_electron_parallel,
            mask,
        ),
    }


def _combined_error_statistics(
    final_state: Fci4FieldState,
    geometry: FciGeometry3D,
    time: float,
) -> tuple[float, float, float]:
    return shifted_mms._combined_error_statistics(final_state, geometry, time)


def _print_state_error_statistics(label: str, stats: dict[str, tuple[float, float, float]]) -> None:
    shifted_mms._print_state_error_statistics(label, stats)


def _resolution_step_count(resolution: int, *, base_steps: int) -> int:
    return shifted_mms._resolution_step_count(resolution, base_steps=base_steps)


def _make_parameters(rho_star_value: float) -> Fci4FieldRhsParameters:
    return Fci4FieldRhsParameters(
        rho_star=float(rho_star_value),
        Te=float(shifted_mms.Te),
        mi_over_me=float(shifted_mms.mi_over_me),
        phi_inversion_tol=1.0e-4,
        phi_inversion_maxiter=100,
        phi_inversion_restart=100,
    )


def _make_gmres_config(parameters: Fci4FieldRhsParameters) -> SpmdGmresConfig:
    return SpmdGmresConfig(
        tol=1.0e-10,
        atol=1.0e-10,
        maxiter=int(parameters.phi_inversion_maxiter),
        restart=int(parameters.phi_inversion_restart),
        acceptance_tol=float(parameters.phi_inversion_tol),
        acceptance_atol=float(parameters.phi_inversion_tol),
        regularization_epsilon=float(parameters.phi_inversion_regularization),
    )


def simulate_mms_shifted_torus_4field_cutwall(
    geometry: FciGeometry3D,
    *,
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
    timestep: float | None = None,
    final_time: float = shifted_mms.tf,
    rho_star_value: float = shifted_mms.rho_star,
    show_progress: bool = False,
    debug_nonfinite: bool = False,
) -> tuple[Fci4FieldState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    shard_counts = tuple(int(value) for value in shard_counts)
    assert_shape_divisible_by_shards(geometry.shape, shard_counts)
    owned_shape = tuple(int(size) // int(count) for size, count in zip(geometry.shape, shard_counts))
    domain = build_shifted_torus_local_domain(geometry.shape, halo_width, shard_counts)
    ghost_filler = shifted_mms._build_ghost_filler(halo_width)
    topology_filler = TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),))
    parameters = _make_parameters(rho_star_value)
    gmres_config = _make_gmres_config(parameters)
    dt = float(final_time) / float(shifted_mms.num_steps) if timestep is None else float(timestep)
    steps = int(round(float(final_time) / dt))
    dt = float(final_time) / float(steps)
    initial_state = shifted_mms._shifted_torus_exact_state(geometry, 0.0)
    times: list[float] = [0.0]
    density_history: list[jnp.ndarray] = [jnp.asarray(initial_state.density, dtype=jnp.float32)]
    omega_history: list[jnp.ndarray] = [jnp.asarray(initial_state.omega, dtype=jnp.float32)]
    v_ion_history: list[jnp.ndarray] = [jnp.asarray(initial_state.v_ion_parallel, dtype=jnp.float32)]
    v_electron_history: list[jnp.ndarray] = [jnp.asarray(initial_state.v_electron_parallel, dtype=jnp.float32)]
    wall_step_times: list[float] = []

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        state = shifted_mms._put_state_on_mesh(initial_state, mesh)
        phi_guess = jax.device_put(
            jnp.asarray(shifted_mms._shifted_torus_phi(geometry, 0.0), dtype=jnp.float64),
            NamedSharding(mesh, P(*MESH_AXIS_NAMES)),
        )
        state_spec = shifted_mms._state_partition_spec()
        field_spec = P(*MESH_AXIS_NAMES)
        host_invariant_domain = LocalDomain3D(
            shard_spec=domain.shard_spec,
            layout=domain.layout,
            mesh_axis_names=(None, None, None),
        )
        sample_invariants = expand_local_shard_pytree(
            shifted_mms._build_local_4field_invariants(
                (0, 0, 0),
                owned_shape=owned_shape,
                halo_width=halo_width,
                global_shape=geometry.shape,
                domain=host_invariant_domain,
            )
        )
        invariant_spec = local_shard_pytree_partition_spec(sample_invariants)
        sample_stage_data = shifted_mms._build_local_4field_rk4_stage_data(
            shifted_mms._build_local_4field_invariants(
                (0, 0, 0),
                owned_shape=owned_shape,
                halo_width=halo_width,
                global_shape=geometry.shape,
                domain=host_invariant_domain,
            ),
            0.0,
            dt,
            parameters=parameters,
        )
        stage_data_spec = local_shard_pytree_partition_spec(
            expand_local_shard_pytree(sample_stage_data)
        )

        def invariant_kernel() -> shifted_mms._ShiftedTorus4FieldInvariantBundle:
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            return expand_local_shard_pytree(
                shifted_mms._build_local_4field_invariants(
                    shard_index,
                    owned_shape=owned_shape,
                    halo_width=halo_width,
                    global_shape=geometry.shape,
                    domain=domain,
                )
            )

        def source_kernel(
            local_invariants: shifted_mms._ShiftedTorus4FieldInvariantBundle,
            step_time: jax.Array,
            step_timestep: jax.Array,
        ) -> shifted_mms._ShiftedTorus4FieldRk4StageData:
            local_invariants = extract_local_shard_pytree(local_invariants)
            return expand_local_shard_pytree(
                shifted_mms._build_local_4field_rk4_stage_data(
                    local_invariants,
                    step_time,
                    step_timestep,
                    parameters=parameters,
                )
            )

        def kernel(
            state_owned: Fci4FieldState,
            phi_guess_owned: jnp.ndarray,
            local_invariants: shifted_mms._ShiftedTorus4FieldInvariantBundle,
            rk_stage_data: shifted_mms._ShiftedTorus4FieldRk4StageData,
            step_time: jax.Array,
            step_timestep: jax.Array,
        ) -> tuple[Fci4FieldState, jnp.ndarray, jnp.ndarray]:
            del step_time
            local_invariants = extract_local_shard_pytree(local_invariants)
            rk_stage_data = extract_local_shard_pytree(rk_stage_data)
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            local_geometry = build_shifted_torus_local_geometry(
                owned_shape,
                halo_width,
                global_shape=geometry.shape,
                shard_index=shard_index,
                x_min=shifted_mms.x_min,
                x_max=shifted_mms.x_max,
                r0=shifted_mms.r0,
                alpha_value=shifted_mms.alpha_value,
                iota=shifted_mms.iota,
                c_phi=shifted_mms.c_phi,
                sigma=shifted_mms.sigma,
            )
            rhs = LocalShiftedTorus4FieldCutWallRhs(
                geometry=local_geometry,
                domain=domain,
                halo_exchange=HaloExchange3D(),
                topology_filler=topology_filler,
                physical_ghost_filler=ghost_filler,
                parameters=parameters,
                curvature_coefficients_owned=local_invariants.curvature_coefficients_owned,
                face_projectors=(
                    local_invariants.face_projector_x,
                    local_invariants.face_projector_y,
                    local_invariants.face_projector_z,
                ),
                gmres_config=gmres_config,
                global_shape=geometry.shape,
            )

            def global_sum(value: jnp.ndarray) -> jnp.ndarray:
                total = value
                for axis_name in MESH_AXIS_NAMES:
                    total = lax.psum(total, axis_name)
                return total

            def global_max(value: jnp.ndarray) -> jnp.ndarray:
                total = value
                for axis_name in MESH_AXIS_NAMES:
                    total = lax.pmax(total, axis_name)
                return total

            def array_nonfinite_count(value: jnp.ndarray) -> jnp.ndarray:
                return global_sum(jnp.sum(~jnp.isfinite(value)).astype(jnp.int32))

            def state_nonfinite_count(value: Fci4FieldState) -> jnp.ndarray:
                return global_sum(
                    (
                        jnp.sum(~jnp.isfinite(value.density))
                        + jnp.sum(~jnp.isfinite(value.omega))
                        + jnp.sum(~jnp.isfinite(value.v_ion_parallel))
                        + jnp.sum(~jnp.isfinite(value.v_electron_parallel))
                    ).astype(jnp.int32)
                )

            def global_term_diagnostics(local_diagnostic: jnp.ndarray) -> jnp.ndarray:
                nonfinite_count = global_sum(local_diagnostic[:, 0])
                maxabs = global_max(local_diagnostic[:, 1])
                return jnp.stack((nonfinite_count, maxabs), axis=-1)

            k1, carry_1, diag_1 = rhs.evaluate_stage(state_owned, rk_stage_data.stage_1, phi_guess_owned)
            stage_1 = state_owned.axpy(k1, scale=0.5 * step_timestep)
            k2, carry_2, diag_2 = rhs.evaluate_stage(stage_1, rk_stage_data.stage_2, carry_1)
            stage_2 = state_owned.axpy(k2, scale=0.5 * step_timestep)
            k3, carry_3, diag_3 = rhs.evaluate_stage(stage_2, rk_stage_data.stage_3, carry_2)
            stage_3 = state_owned.axpy(k3, scale=step_timestep)
            k4, carry_4, diag_4 = rhs.evaluate_stage(stage_3, rk_stage_data.stage_4, carry_3)
            next_state = state_owned.axpy(
                k1.axpy(k2, scale=2.0).axpy(k3, scale=2.0).axpy(k4, scale=1.0),
                scale=step_timestep / 6.0,
            )
            nonfinite_diagnostics = jnp.asarray(
                [
                    state_nonfinite_count(state_owned),
                    array_nonfinite_count(carry_1),
                    state_nonfinite_count(k1),
                    state_nonfinite_count(stage_1),
                    array_nonfinite_count(carry_2),
                    state_nonfinite_count(k2),
                    state_nonfinite_count(stage_2),
                    array_nonfinite_count(carry_3),
                    state_nonfinite_count(k3),
                    state_nonfinite_count(stage_3),
                    array_nonfinite_count(carry_4),
                    state_nonfinite_count(k4),
                    state_nonfinite_count(next_state),
                ],
                dtype=jnp.int32,
            )
            if debug_nonfinite:
                term_diagnostics = jnp.stack(
                    (
                        global_term_diagnostics(diag_1),
                        global_term_diagnostics(diag_2),
                        global_term_diagnostics(diag_3),
                        global_term_diagnostics(diag_4),
                    ),
                    axis=0,
                )
                diagnostics = jnp.concatenate(
                    (
                        nonfinite_diagnostics.astype(jnp.float64),
                        term_diagnostics.reshape((-1,)),
                    ),
                    axis=0,
                )
            else:
                diagnostics = nonfinite_diagnostics.astype(jnp.float64)
            return next_state, carry_4, diagnostics

        invariants = jax.jit(
            shard_map(
                invariant_kernel,
                mesh=mesh,
                in_specs=(),
                out_specs=invariant_spec,
                check_rep=False,
            )
        )()
        compiled_source_kernel = jax.jit(
            shard_map(
                source_kernel,
                mesh=mesh,
                in_specs=(invariant_spec, P(), P()),
                out_specs=stage_data_spec,
                check_rep=False,
            )
        )
        step_kernel = jax.jit(
            shard_map(
                kernel,
                mesh=mesh,
                in_specs=(state_spec, field_spec, invariant_spec, stage_data_spec, P(), P()),
                out_specs=(state_spec, field_spec, P()),
                check_rep=False,
            )
        )
        diagnostic_labels = (
            "state_in",
            "phi_1",
            "k1",
            "stage_1",
            "phi_2",
            "k2",
            "stage_2",
            "phi_3",
            "k3",
            "stage_3",
            "phi_4",
            "k4",
            "next_state",
        )

        time_value = 0.0
        progress_start = time_module.perf_counter()
        if show_progress:
            print(
                f"shifted_torus_4field_cutwall RK4 progress: "
                f"{shifted_mms._format_progress_bar(0, steps, start_time=progress_start)}",
                end="",
                flush=True,
            )
        for step_index in range(steps):
            step_start = time_module.perf_counter()
            rk_stage_data = compiled_source_kernel(
                invariants,
                jnp.asarray(time_value, dtype=jnp.float64),
                jnp.asarray(dt, dtype=jnp.float64),
            )
            state, phi_guess, step_diagnostics = step_kernel(
                state,
                phi_guess,
                invariants,
                rk_stage_data,
                jnp.asarray(time_value, dtype=jnp.float64),
                jnp.asarray(dt, dtype=jnp.float64),
            )
            jax.block_until_ready(state.density)
            if debug_nonfinite:
                diag = np.asarray(jax.device_get(step_diagnostics), dtype=np.int64)
                nonzero = [(label, int(value)) for label, value in zip(diagnostic_labels, diag) if int(value) != 0]
                if nonzero:
                    print()
                    print(
                        "shifted_torus_4field_cutwall nonfinite diagnostics: "
                        f"step={step_index + 1}, t={time_value + dt:.6e}"
                    )
                    for label, value in nonzero:
                        print(f"  {label}: {value}")
            wall_step_times.append(time_module.perf_counter() - step_start)
            time_value += dt
            times.append(time_value)
            gathered_state = shifted_mms._gather_state_from_mesh(state)
            density_history.append(jnp.asarray(gathered_state.density, dtype=jnp.float32))
            omega_history.append(jnp.asarray(gathered_state.omega, dtype=jnp.float32))
            v_ion_history.append(jnp.asarray(gathered_state.v_ion_parallel, dtype=jnp.float32))
            v_electron_history.append(jnp.asarray(gathered_state.v_electron_parallel, dtype=jnp.float32))
            if show_progress:
                print(
                    "\r"
                    f"shifted_torus_4field_cutwall RK4 progress: "
                    f"{shifted_mms._format_progress_bar(step_index + 1, steps, start_time=progress_start)}",
                    end="",
                    flush=True,
                )
        if show_progress:
            print()
        final_state = shifted_mms._gather_state_from_mesh(state)

    if wall_step_times:
        print(
            "shifted_torus_4field_cutwall mean timings per RK step: "
            f"wall={np.mean(np.asarray(wall_step_times, dtype=np.float64)):.6e} s"
        )

    return (
        final_state,
        jnp.asarray(times, dtype=jnp.float64),
        jnp.stack(density_history, axis=0),
        jnp.stack(omega_history, axis=0),
        jnp.stack(v_ion_history, axis=0),
        jnp.stack(v_electron_history, axis=0),
    )


def run_shifted_torus_4field_cutwall_convergence(
    *,
    resolutions: list[int],
    shard_counts: tuple[int, int, int],
    halo_width: int,
    final_time: float = shifted_mms.tf,
    base_steps: int = shifted_mms.num_steps,
    rho_star_value: float = shifted_mms.rho_star,
    plot: bool = False,
    plot_path: str | None = None,
    show_progress: bool = False,
    debug_nonfinite: bool = False,
) -> dict[str, object]:
    successful_resolutions: list[int] = []
    l2_errors: list[float] = []
    max_errors: list[float] = []
    per_resolution_stats: list[tuple[int, dict[str, tuple[float, float, float]]]] = []

    for resolution in resolutions:
        shape = _shape_from_resolution(int(resolution))
        assert_shape_divisible_by_shards(shape, shard_counts)
        geometry = shifted_mms.build_shifted_torus_4field_geometry(shape)
        steps = _resolution_step_count(int(resolution), base_steps=base_steps)
        dt = float(final_time) / float(steps)
        print(
            f"Starting shifted_torus_4field_cutwall MMS run: resolution={int(resolution)}, "
            f"shard_counts={shard_counts}, steps={steps}, dt={dt:.6e}"
        )
        start = time_module.perf_counter()
        final_state, *_ = simulate_mms_shifted_torus_4field_cutwall(
            geometry,
            shard_counts=shard_counts,
            halo_width=halo_width,
            final_time=final_time,
            timestep=dt,
            rho_star_value=rho_star_value,
            show_progress=show_progress,
            debug_nonfinite=debug_nonfinite,
        )
        elapsed = time_module.perf_counter() - start
        exact_state = shifted_mms._shifted_torus_exact_state(geometry, final_time)
        solid_mask = _solid_box_center_mask(geometry, margin_cells=0)
        active_mask = ~solid_mask
        abs_errors = [
            jnp.abs(final_state.density - exact_state.density),
            jnp.abs(final_state.omega - exact_state.omega),
            jnp.abs(final_state.v_ion_parallel - exact_state.v_ion_parallel),
            jnp.abs(final_state.v_electron_parallel - exact_state.v_electron_parallel),
        ]
        count = jnp.maximum(jnp.sum(active_mask.astype(jnp.float64)) * float(len(abs_errors)), 1.0)
        active_errors = [jnp.where(active_mask, error, 0.0) for error in abs_errors]
        sumsq_error = sum(jnp.sum(jnp.square(error)) for error in active_errors)
        mean_error = float(jnp.sqrt(sumsq_error / count))
        active_mask_host = np.asarray(active_mask, dtype=bool)
        solid_mask_host = np.asarray(solid_mask, dtype=bool)
        masked_values = np.concatenate(
            [
                np.asarray(error, dtype=np.float64)[active_mask_host].ravel()
                for error in abs_errors
            ]
        )
        active_nonfinite = int(np.count_nonzero(~np.isfinite(masked_values)))
        solid_values = np.concatenate(
            [
                np.asarray(error, dtype=np.float64)[solid_mask_host].ravel()
                for error in abs_errors
            ]
        )
        solid_nonfinite = int(np.count_nonzero(~np.isfinite(solid_values)))
        finite_masked_values = masked_values[np.isfinite(masked_values)]
        median_error = float(np.median(finite_masked_values)) if finite_masked_values.size else float("nan")
        max_error = float(np.max(finite_masked_values)) if finite_masked_values.size else float("nan")
        per_field_stats = _masked_state_error_statistics(final_state, exact_state, active_mask)
        successful_resolutions.append(int(resolution))
        l2_errors.append(mean_error)
        max_errors.append(max_error)
        per_resolution_stats.append((int(resolution), per_field_stats))
        print(
            f"N={int(resolution)}: shard_counts={shard_counts}, steps={steps}, "
            f"total_runtime={elapsed:.6e} s, avg_step_runtime={elapsed / float(steps):.6e} s, "
            f"L2={mean_error:.6e}, median={median_error:.6e}, Linf={max_error:.6e}, "
            f"active_nonfinite={active_nonfinite}, solid_nonfinite={solid_nonfinite}"
        )
        _print_state_error_statistics(f"N={int(resolution)} per-field final errors", per_field_stats)

    l2_order: float | None = None
    max_order: float | None = None
    if len(successful_resolutions) >= 2:
        plotted_resolutions = np.asarray(successful_resolutions, dtype=np.float64)
        l2_log_errors = np.log(np.asarray(l2_errors, dtype=np.float64))
        max_log_errors = np.log(np.asarray(max_errors, dtype=np.float64))
        l2_slope, l2_intercept = np.polyfit(np.log(plotted_resolutions), l2_log_errors, 1)
        max_slope, max_intercept = np.polyfit(np.log(plotted_resolutions), max_log_errors, 1)
        l2_order = float(-l2_slope)
        max_order = float(-max_slope)
        print(f"shifted_torus_4field_cutwall L2 convergence order: {l2_order:.6f}")
        print(f"shifted_torus_4field_cutwall Linf convergence order: {max_order:.6f}")
        if plot:
            import matplotlib.pyplot as plt

            output_path = Path(plot_path or "shifted_torus_4field_cutwall_convergence.png")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6.8, 4.8))
            ax.loglog(plotted_resolutions, l2_errors, "o-", label=f"L2, order {l2_order:.2f}")
            ax.loglog(plotted_resolutions, max_errors, "^-", label=f"Linf, order {max_order:.2f}")
            ax.loglog(
                plotted_resolutions,
                np.exp(l2_intercept) * plotted_resolutions**l2_slope,
                "--",
                color=ax.lines[0].get_color(),
            )
            ax.loglog(
                plotted_resolutions,
                np.exp(max_intercept) * plotted_resolutions**max_slope,
                "--",
                color=ax.lines[1].get_color(),
            )
            ax.set_xlabel("resolution")
            ax.set_ylabel("absolute error")
            ax.set_title(f"Shifted-torus 4-field cut-wall MMS ({shard_counts})")
            ax.grid(True, which="both", linestyle=":", alpha=0.45)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_path, dpi=200)
            plt.close(fig)

    return {
        "resolutions": successful_resolutions,
        "l2_errors": l2_errors,
        "linf_errors": max_errors,
        "l2_order": l2_order,
        "linf_order": max_order,
        "per_field": per_resolution_stats,
    }


def _single_local_case(
    shape: tuple[int, int, int] = (12, 12, 12),
    *,
    halo_width: int = 2,
) -> tuple[LocalFciGeometry3D, LocalDomain3D, shifted_mms._ShiftedTorus4FieldInvariantBundle]:
    domain = build_shifted_torus_local_domain(shape, halo_width, (1, 1, 1))
    geometry = build_shifted_torus_local_geometry(
        shape,
        halo_width,
        global_shape=shape,
        shard_index=(0, 0, 0),
        x_min=shifted_mms.x_min,
        x_max=shifted_mms.x_max,
        r0=shifted_mms.r0,
        alpha_value=shifted_mms.alpha_value,
        iota=shifted_mms.iota,
        c_phi=shifted_mms.c_phi,
        sigma=shifted_mms.sigma,
    )
    invariants = shifted_mms._build_local_4field_invariants(
        (0, 0, 0),
        owned_shape=shape,
        halo_width=halo_width,
        global_shape=shape,
        domain=domain,
    )
    return geometry, domain, invariants


def test_shifted_torus_4field_cutwall_geometry_has_closed_box_faces() -> None:
    geometry, _domain, _invariants = _single_local_case((12, 12, 12))
    fixture = _build_shifted_torus_cut_wall_fixture(geometry, global_shape=(12, 12, 12))
    active = np.asarray(fixture.geometry.active)
    assert int(np.sum(active)) > 0
    active_plane_ids = set(np.asarray(fixture.plane_id[fixture.geometry.active], dtype=np.int64).tolist())
    assert active_plane_ids == {0, 1, 2, 3, 4, 5}

    radial_span = float(shifted_mms.x_max) - float(shifted_mms.x_min)
    radial_min = float(shifted_mms.x_min) + BOX_X_FRACTION_RANGE[0] * radial_span
    radial_max = float(shifted_mms.x_min) + BOX_X_FRACTION_RANGE[1] * radial_span
    theta_min = BOX_THETA_CENTER - BOX_THETA_HALF_WIDTH
    theta_max = BOX_THETA_CENTER + BOX_THETA_HALF_WIDTH
    zeta_min, zeta_max = BOX_ZETA_RANGE
    assert bool(jnp.all(fixture.wall_x[fixture.geometry.active] >= radial_min - 1.0e-12))
    assert bool(jnp.all(fixture.wall_x[fixture.geometry.active] <= radial_max + 1.0e-12))
    assert bool(jnp.all(fixture.wall_theta[fixture.geometry.active] >= theta_min - 1.0e-12))
    assert bool(jnp.all(fixture.wall_theta[fixture.geometry.active] <= theta_max + 1.0e-12))
    assert bool(jnp.all(fixture.wall_zeta[fixture.geometry.active] >= zeta_min - 1.0e-12))
    assert bool(jnp.all(fixture.wall_zeta[fixture.geometry.active] <= zeta_max + 1.0e-12))

    cart_x, cart_y, _cart_z = _shifted_torus_cartesian_from_logical(
        fixture.wall_x,
        fixture.wall_theta,
        fixture.wall_zeta,
    )
    zeta_wall = fixture.geometry.active & (fixture.geometry.stencil_axis == 2)
    cartesian_plane_residual = (
        -jnp.sin(fixture.wall_zeta) * cart_x
        + jnp.cos(fixture.wall_zeta) * cart_y
    )
    assert float(jnp.max(jnp.where(zeta_wall, jnp.abs(cartesian_plane_residual), 0.0))) < 1.0e-10

    assert bool(jnp.all(jnp.isfinite(fixture.geometry.normal_contra[fixture.geometry.active])))
    normal_norm_squared = jnp.einsum(
        "...i,...ij,...j->...",
        fixture.geometry.normal_contra,
        fixture.geometry.g_cov,
        fixture.geometry.normal_contra,
    )
    assert float(jnp.max(jnp.where(fixture.geometry.active, jnp.abs(normal_norm_squared - 1.0), 0.0))) < 1.0e-10

    active_axes = set(np.asarray(fixture.geometry.stencil_axis[fixture.geometry.active], dtype=np.int64).tolist())
    assert active_axes == {0, 1, 2}
    assert int(jnp.sum(~fixture.regular_face_geometry.x_open_mask)) > 0
    assert int(jnp.sum(~fixture.regular_face_geometry.y_open_mask)) > 0
    assert int(jnp.sum(~fixture.regular_face_geometry.z_open_mask)) > 0


def test_shifted_torus_4field_cutwall_geometry_respects_z_shards() -> None:
    global_shape = (20, 20, 20)
    shard_counts = (1, 1, 4)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(global_shape, shard_counts)
    )
    active_counts_by_shard: list[int] = []
    active_plane_ids_by_shard: list[set[int]] = []
    for shard_z in range(shard_counts[2]):
        local_geometry = build_shifted_torus_local_geometry(
            owned_shape,
            2,
            global_shape=global_shape,
            shard_index=(0, 0, shard_z),
            x_min=shifted_mms.x_min,
            x_max=shifted_mms.x_max,
            r0=shifted_mms.r0,
            alpha_value=shifted_mms.alpha_value,
            iota=shifted_mms.iota,
            c_phi=shifted_mms.c_phi,
            sigma=shifted_mms.sigma,
        )
        fixture = _build_shifted_torus_cut_wall_fixture(
            local_geometry,
            global_shape=global_shape,
        )
        active_plane_ids = set(
            np.asarray(
                fixture.plane_id[fixture.geometry.active],
                dtype=np.int64,
            ).tolist()
        )
        active_counts_by_shard.append(int(jnp.sum(fixture.geometry.active)))
        active_plane_ids_by_shard.append(active_plane_ids)
    assert all(count > 0 for count in active_counts_by_shard[:3])
    assert active_counts_by_shard[3] == 0
    assert active_plane_ids_by_shard[0] >= {0, 1, 2, 3, 4}
    assert active_plane_ids_by_shard[1] >= {0, 1, 2, 3}
    assert active_plane_ids_by_shard[2] >= {0, 1, 2, 3, 5}
    assert active_plane_ids_by_shard[3] == set()


def test_shifted_torus_4field_cutwall_rhs_isolates_blocked_stencil_values() -> None:
    shape = (8, 8, 8)
    halo_width = 2
    geometry, domain, invariants = _single_local_case(shape, halo_width=halo_width)
    parameters = _make_parameters(shifted_mms.rho_star)
    stage = shifted_mms._build_local_4field_stage_data(invariants, 0.0, parameters=parameters)
    exact_owned = Fci4FieldState(
        density=stage.exact_halo.density[domain.layout.owned_slices_cell],
        omega=stage.exact_halo.omega[domain.layout.owned_slices_cell],
        v_ion_parallel=stage.exact_halo.v_ion_parallel[domain.layout.owned_slices_cell],
        v_electron_parallel=stage.exact_halo.v_electron_parallel[domain.layout.owned_slices_cell],
    )
    rhs = LocalShiftedTorus4FieldCutWallRhs(
        geometry=geometry,
        domain=domain,
        halo_exchange=HaloExchange3D(),
        topology_filler=TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),)),
        physical_ghost_filler=shifted_mms._build_ghost_filler(halo_width),
        parameters=parameters,
        curvature_coefficients_owned=invariants.curvature_coefficients_owned,
        face_projectors=(invariants.face_projector_x, invariants.face_projector_y, invariants.face_projector_z),
        gmres_config=_make_gmres_config(parameters),
        global_shape=shape,
    )
    fixture = _build_shifted_torus_cut_wall_fixture(geometry, global_shape=shape)
    plus_rows = fixture.geometry.active & (fixture.geometry.stencil_side == 1)
    plus_owner_mask = jnp.zeros(shape, dtype=jnp.int32).at[
        fixture.geometry.owner_i,
        fixture.geometry.owner_j,
        fixture.geometry.owner_k,
    ].add(plus_rows.astype(jnp.int32)) > 0
    plus_neighbor_i = jnp.where(
        fixture.geometry.stencil_axis == 0,
        jnp.minimum(fixture.geometry.owner_i + 1, shape[0] - 1),
        fixture.geometry.owner_i,
    )
    plus_neighbor_j = jnp.where(
        fixture.geometry.stencil_axis == 1,
        jnp.minimum(fixture.geometry.owner_j + 1, shape[1] - 1),
        fixture.geometry.owner_j,
    )
    plus_neighbor_k = jnp.where(
        fixture.geometry.stencil_axis == 2,
        jnp.minimum(fixture.geometry.owner_k + 1, shape[2] - 1),
        fixture.geometry.owner_k,
    )
    blocked_plus_mask = jnp.zeros(shape, dtype=jnp.int32).at[
        plus_neighbor_i,
        plus_neighbor_j,
        plus_neighbor_k,
    ].add(plus_rows.astype(jnp.int32)) > 0
    perturbed = Fci4FieldState(
        density=jnp.where(blocked_plus_mask, exact_owned.density + 0.17, exact_owned.density),
        omega=exact_owned.omega,
        v_ion_parallel=jnp.where(blocked_plus_mask, exact_owned.v_ion_parallel - 0.11, exact_owned.v_ion_parallel),
        v_electron_parallel=jnp.where(
            blocked_plus_mask,
            exact_owned.v_electron_parallel + 0.09,
            exact_owned.v_electron_parallel,
        ),
    )
    base_rhs, phi_owned, _ = rhs.evaluate_stage(exact_owned, stage, stage.phi_halo[domain.layout.owned_slices_cell])
    pert_rhs, _, _ = rhs.evaluate_stage(perturbed, stage, phi_owned)
    assert int(jnp.sum(plus_owner_mask)) > 0
    assert float(jnp.max(jnp.where(plus_owner_mask, jnp.abs(base_rhs.density - pert_rhs.density), 0.0))) < 1.0e-8
    assert float(jnp.max(jnp.where(plus_owner_mask, jnp.abs(base_rhs.v_ion_parallel - pert_rhs.v_ion_parallel), 0.0))) < 1.0e-8


def test_shifted_torus_4field_cutwall_phi_solve_uses_wall_bc() -> None:
    shape = (8, 8, 8)
    halo_width = 2
    geometry, domain, invariants = _single_local_case(shape, halo_width=halo_width)
    parameters = _make_parameters(shifted_mms.rho_star)
    stage = shifted_mms._build_local_4field_stage_data(invariants, 0.0, parameters=parameters)
    exact_owned = Fci4FieldState(
        density=stage.exact_halo.density[domain.layout.owned_slices_cell],
        omega=stage.exact_halo.omega[domain.layout.owned_slices_cell],
        v_ion_parallel=stage.exact_halo.v_ion_parallel[domain.layout.owned_slices_cell],
        v_electron_parallel=stage.exact_halo.v_electron_parallel[domain.layout.owned_slices_cell],
    )
    rhs = LocalShiftedTorus4FieldCutWallRhs(
        geometry=geometry,
        domain=domain,
        halo_exchange=HaloExchange3D(),
        topology_filler=TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),)),
        physical_ghost_filler=shifted_mms._build_ghost_filler(halo_width),
        parameters=parameters,
        curvature_coefficients_owned=invariants.curvature_coefficients_owned,
        face_projectors=(invariants.face_projector_x, invariants.face_projector_y, invariants.face_projector_z),
        gmres_config=_make_gmres_config(parameters),
        global_shape=shape,
    )
    _, phi_base, diagnostic_base = rhs.evaluate_stage(
        exact_owned,
        stage,
        stage.phi_halo[domain.layout.owned_slices_cell],
        return_wall_diagnostic=True,
    )
    _, phi_offset, diagnostic_offset = rhs.evaluate_stage(
        exact_owned,
        stage,
        phi_base,
        phi_wall_offset=1.0e-3,
        return_wall_diagnostic=True,
    )
    assert bool(jnp.all(jnp.isfinite(phi_base)))
    assert float(diagnostic_base[1]) > 0.0
    assert float(jnp.max(jnp.abs(phi_offset - phi_base))) > 0.0
    assert float(diagnostic_offset[0]) > float(diagnostic_base[0])


def test_shifted_torus_4field_cutwall_mms_converges() -> None:
    results = run_shifted_torus_4field_cutwall_convergence(
        resolutions=[6, 8],
        shard_counts=(1, 1, 1),
        halo_width=2,
        final_time=0.01,
        base_steps=4,
        rho_star_value=shifted_mms.rho_star,
        show_progress=False,
    )
    l2_errors = results["l2_errors"]
    linf_errors = results["linf_errors"]
    assert len(l2_errors) == 2
    assert np.isfinite(np.asarray(l2_errors)).all()
    assert np.isfinite(np.asarray(linf_errors)).all()
    assert float(l2_errors[-1]) < float(l2_errors[0])
    assert float(linf_errors[-1]) < float(linf_errors[0])


def _print_runtime_info() -> None:
    print("=" * 80)
    print("JAX runtime")
    print("=" * 80)
    print(f"default backend: {jax.default_backend()}")
    print(f"local_device_count: {jax.local_device_count()}")
    print("devices:")
    for index, device in enumerate(jax.local_devices()):
        print(f"  [{index}] {device}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shifted-torus 4-field cut-wall MMS convergence harness")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[10, 14])
    parser.add_argument("--shard-counts", nargs=3, type=int, metavar=("PX", "PY", "PZ"), default=(1, 1, 1))
    parser.add_argument("--halo-width", type=int, default=2)
    parser.add_argument("--final-time", type=float, default=shifted_mms.tf)
    parser.add_argument("--base-steps", type=int, default=shifted_mms.num_steps)
    parser.add_argument("--rho-star", type=float, default=shifted_mms.rho_star)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--debug-nonfinite", action="store_true")
    parser.add_argument("--skip-runtime-info", action="store_true")
    args = parser.parse_args()

    if not args.skip_runtime_info:
        _print_runtime_info()
    run_shifted_torus_4field_cutwall_convergence(
        resolutions=[int(value) for value in args.resolutions],
        shard_counts=tuple(int(value) for value in args.shard_counts),
        halo_width=int(args.halo_width),
        final_time=float(args.final_time),
        base_steps=int(args.base_steps),
        rho_star_value=float(args.rho_star),
        plot=bool(args.plot),
        plot_path=args.plot_path,
        show_progress=bool(args.show_progress),
        debug_nonfinite=bool(args.debug_nonfinite),
    )


if __name__ == "__main__":
    main()
