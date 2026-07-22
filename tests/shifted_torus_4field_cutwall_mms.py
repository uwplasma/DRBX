"""Manufactured-solution projection and boundary data for cut-wall torus tests."""

from __future__ import annotations

from dataclasses import replace as dataclass_replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from drbx.geometry import (
    FciGeometry3D,
    LocalControlVolumeCellGeometry3D,
    LocalFciGeometry3D,
)
from drbx.native import Fci4FieldRhsParameters, Fci4FieldState
from drbx.native.fci_boundaries import (
    BC_DIRICHLET,
    BC_NONE,
    CV_FACE_CUT_WALL,
    CV_FACE_PHYSICAL_BOUNDARY,
    LocalBoundaryFaceBC3D,
    LocalControlVolumeBoundaryBC3D,
    LocalEmbeddedControlVolumeGeometry3D,
)


_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))


import shifted_torus_4field_mms_helpers as shifted_mms  # noqa: E402
from mms_domain_decomp_helpers import build_shifted_torus_local_geometry  # noqa: E402
from shifted_torus_4field_cutwall_geometry import (  # noqa: E402
    _GAUSS3_NODES,
    _GAUSS3_WEIGHTS,
    _box_bounds,
    _shifted_torus_curvature_at_logical_points,
    _shifted_torus_metric_payload_jax,
)


def _shifted_torus_regular_radial_face_average(
    geometry: LocalFciGeometry3D,
    stage_time: float | jax.Array,
    field_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return J-weighted Dirichlet averages on the two radial face patches."""

    theta_faces = jnp.asarray(
        geometry.grid.y.faces_owned,
        dtype=jnp.float64,
    )
    zeta_faces = jnp.asarray(
        geometry.grid.z.faces_owned,
        dtype=jnp.float64,
    )
    theta_mid = 0.5 * (theta_faces[:-1] + theta_faces[1:])
    zeta_mid = 0.5 * (zeta_faces[:-1] + zeta_faces[1:])
    theta_half = 0.5 * (theta_faces[1:] - theta_faces[:-1])
    zeta_half = 0.5 * (zeta_faces[1:] - zeta_faces[:-1])
    theta_mid = theta_mid[:, None]
    zeta_mid = zeta_mid[None, :]
    theta_half = theta_half[:, None]
    zeta_half = zeta_half[None, :]

    def one_side(x_face: jnp.ndarray) -> jnp.ndarray:
        numerator = jnp.zeros(
            (geometry.owned_shape[1], geometry.owned_shape[2]),
            dtype=jnp.float64,
        )
        denominator = jnp.zeros_like(numerator)
        for node_y, weight_y in zip(_GAUSS3_NODES, _GAUSS3_WEIGHTS):
            theta = theta_mid + theta_half * float(node_y)
            for node_z, weight_z in zip(_GAUSS3_NODES, _GAUSS3_WEIGHTS):
                zeta = zeta_mid + zeta_half * float(node_z)
                x = jnp.broadcast_to(x_face, numerator.shape)
                theta_value = jnp.broadcast_to(theta, numerator.shape)
                zeta_value = jnp.broadcast_to(zeta, numerator.shape)
                points = jnp.stack((x, theta_value, zeta_value), axis=-1)
                jacobian = _shifted_torus_metric_payload_jax(points)[0]
                measure = (
                    float(weight_y)
                    * float(weight_z)
                    * theta_half
                    * zeta_half
                    * jacobian
                )
                value = _shifted_torus_exact_field_at_logical_points(
                    points,
                    stage_time,
                    field_name,
                )
                numerator = numerator + measure * value
                denominator = denominator + measure
        return numerator / jnp.maximum(denominator, 1.0e-30)

    x_faces = jnp.asarray(
        geometry.grid.x.faces_owned,
        dtype=jnp.float64,
    )
    return one_side(x_faces[0]), one_side(x_faces[-1])


def _with_shifted_torus_regular_radial_face_averages(
    stage: shifted_mms._ShiftedTorus4FieldStageData,
    geometry: LocalFciGeometry3D,
    stage_time: float | jax.Array,
) -> shifted_mms._ShiftedTorus4FieldStageData:
    """Use finite-volume face averages for regular radial Dirichlet data."""

    phi_lower, phi_upper = _shifted_torus_regular_radial_face_average(
        geometry,
        stage_time,
        "phi",
    )
    density_lower, density_upper = _shifted_torus_regular_radial_face_average(
        geometry,
        stage_time,
        "density",
    )
    omega_lower, omega_upper = _shifted_torus_regular_radial_face_average(
        geometry,
        stage_time,
        "omega",
    )
    v_ion_lower, v_ion_upper = _shifted_torus_regular_radial_face_average(
        geometry,
        stage_time,
        "v_ion_parallel",
    )
    v_electron_lower, v_electron_upper = (
        _shifted_torus_regular_radial_face_average(
            geometry,
            stage_time,
            "v_electron_parallel",
        )
    )
    return dataclass_replace(
        stage,
        phi_face_lower=phi_lower,
        phi_face_upper=phi_upper,
        density_face_lower=density_lower,
        density_face_upper=density_upper,
        omega_face_lower=omega_lower,
        omega_face_upper=omega_upper,
        v_ion_face_lower=v_ion_lower,
        v_ion_face_upper=v_ion_upper,
        v_electron_face_lower=v_electron_lower,
        v_electron_face_upper=v_electron_upper,
    )


def _integrate_local_exact_state_over_fluid(
    geometry: LocalFciGeometry3D,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
    time: float | jax.Array,
) -> tuple[Fci4FieldState, jnp.ndarray]:
    """Project exact state and phi to true three-point fluid-volume averages."""

    faces = (
        jnp.asarray(geometry.grid.x.faces_owned, dtype=jnp.float64),
        jnp.asarray(geometry.grid.y.faces_owned, dtype=jnp.float64),
        jnp.asarray(geometry.grid.z.faces_owned, dtype=jnp.float64),
    )
    lower = (
        faces[0][:-1, None, None],
        faces[1][None, :-1, None],
        faces[2][None, None, :-1],
    )
    upper = (
        faces[0][1:, None, None],
        faces[1][None, 1:, None],
        faces[2][None, None, 1:],
    )
    bounds = _box_bounds()
    solid_lower = tuple(
        jnp.maximum(value, float(bounds[axis][0]))
        for axis, value in enumerate(lower)
    )
    solid_upper = tuple(
        jnp.minimum(value, float(bounds[axis][1]))
        for axis, value in enumerate(upper)
    )

    def integrate_region(
        region_lower: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        region_upper: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        region_lower = tuple(
            jnp.broadcast_to(value, geometry.owned_shape)
            for value in region_lower
        )
        region_upper = tuple(
            jnp.broadcast_to(value, geometry.owned_shape)
            for value in region_upper
        )
        half = tuple(
            0.5 * (hi - lo)
            for lo, hi in zip(region_lower, region_upper)
        )
        midpoint = tuple(
            0.5 * (hi + lo)
            for lo, hi in zip(region_lower, region_upper)
        )
        valid = (half[0] > 0.0) & (half[1] > 0.0) & (half[2] > 0.0)
        integrals = [
            jnp.zeros(geometry.owned_shape, dtype=jnp.float64)
            for _ in range(5)
        ]
        for node_x, weight_x in zip(_GAUSS3_NODES, _GAUSS3_WEIGHTS):
            x = midpoint[0] + half[0] * float(node_x)
            for node_y, weight_y in zip(_GAUSS3_NODES, _GAUSS3_WEIGHTS):
                theta = midpoint[1] + half[1] * float(node_y)
                x_mid = 0.5 * (
                    float(shifted_mms.x_min) + float(shifted_mms.x_max)
                )
                theta_shift = theta + float(shifted_mms.sigma) * (x - x_mid)
                radius = (
                    float(shifted_mms.r0)
                    + float(shifted_mms.alpha_value) * x
                    + x * jnp.cos(theta_shift)
                )
                q_value = 1.0 + float(shifted_mms.alpha_value) * jnp.cos(
                    theta_shift
                )
                J = radius * x * q_value
                for node_z, weight_z in zip(_GAUSS3_NODES, _GAUSS3_WEIGHTS):
                    zeta = midpoint[2] + half[2] * float(node_z)
                    quadrature_weight = (
                        float(weight_x)
                        * float(weight_y)
                        * float(weight_z)
                        * half[0]
                        * half[1]
                        * half[2]
                        * J
                    )
                    quadrature_weight = jnp.where(
                        valid,
                        quadrature_weight,
                        0.0,
                    )
                    points = jnp.stack((x, theta, zeta), axis=-1)
                    values = (
                        _shifted_torus_exact_field_at_logical_points(
                            points,
                            time,
                            "density",
                        ),
                        _shifted_torus_exact_field_at_logical_points(
                            points,
                            time,
                            "omega",
                        ),
                        _shifted_torus_exact_field_at_logical_points(
                            points,
                            time,
                            "v_ion_parallel",
                        ),
                        _shifted_torus_exact_field_at_logical_points(
                            points,
                            time,
                            "v_electron_parallel",
                        ),
                        _shifted_torus_exact_field_at_logical_points(
                            points,
                            time,
                            "phi",
                        ),
                    )
                    integrals = [
                        integral + quadrature_weight * value
                        for integral, value in zip(integrals, values)
                    ]
        return tuple(integrals)

    full_integrals = integrate_region(lower, upper)
    solid_integrals = integrate_region(solid_lower, solid_upper)
    fluid_integrals = tuple(
        full - solid for full, solid in zip(full_integrals, solid_integrals)
    )
    cells = control_volume_geometry.cells
    raw_averages = tuple(
        integral / jnp.maximum(cells.raw_volume, 1.0e-30)
        for integral in fluid_integrals
    )
    aggregate = tuple(
        _agglomerate_control_volume_average(value, cells)
        for value in raw_averages
    )
    return (
        Fci4FieldState(
            density=aggregate[0],
            omega=aggregate[1],
            v_ion_parallel=aggregate[2],
            v_electron_parallel=aggregate[3],
        ),
        aggregate[4],
    )


def _project_global_exact_state_to_control_volumes(
    geometry: FciGeometry3D,
    stacked_control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
    *,
    shard_counts: tuple[int, int, int],
    halo_width: int,
    time: float,
) -> tuple[Fci4FieldState, jnp.ndarray]:
    """Assemble global exact control-volume averages from local-owned bundles."""

    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(geometry.shape, shard_counts)
    )
    fields = [
        jnp.zeros(geometry.shape, dtype=jnp.float64)
        for _ in range(5)
    ]
    for shard_i in range(shard_counts[0]):
        for shard_j in range(shard_counts[1]):
            for shard_k in range(shard_counts[2]):
                shard_index = (shard_i, shard_j, shard_k)
                local_control_volume_geometry = jax.tree_util.tree_map(
                    lambda leaf: leaf[shard_index],
                    stacked_control_volume_geometry,
                )
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
                local_state, local_phi = _integrate_local_exact_state_over_fluid(
                    local_geometry,
                    local_control_volume_geometry,
                    jnp.asarray(time, dtype=jnp.float64),
                )
                starts = tuple(
                    shard_index[axis] * owned_shape[axis]
                    for axis in range(3)
                )
                slices = tuple(
                    slice(start, start + owned_shape[axis])
                    for axis, start in enumerate(starts)
                )
                local_values = (
                    local_state.density,
                    local_state.omega,
                    local_state.v_ion_parallel,
                    local_state.v_electron_parallel,
                    local_phi,
                )
                fields = [
                    field.at[slices].set(local_value)
                    for field, local_value in zip(fields, local_values)
                ]
    return (
        Fci4FieldState(
            density=fields[0],
            omega=fields[1],
            v_ion_parallel=fields[2],
            v_electron_parallel=fields[3],
        ),
        fields[4],
    )


def _assemble_global_control_volume_cell_data(
    global_shape: tuple[int, int, int],
    stacked_control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
    *,
    shard_counts: tuple[int, int, int],
) -> dict[str, jnp.ndarray]:
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(global_shape, shard_counts)
    )
    names = (
        "aggregate_volume",
        "raw_volume",
        "is_active_owner",
        "is_aggregate_target",
        "is_merged_source",
        "received_source_count",
        "member_count",
        "boundary_face_count",
        "irregular_face_count",
        "remote_face_count",
        "reconstruction_row_count",
    )
    dtypes = (
        jnp.float64,
        jnp.float64,
        bool,
        bool,
        bool,
        jnp.int32,
        jnp.int32,
        jnp.int32,
        jnp.int32,
        jnp.int32,
        jnp.int32,
    )
    result = {
        name: jnp.zeros(global_shape, dtype=dtype)
        for name, dtype in zip(names, dtypes)
    }
    for shard_i in range(shard_counts[0]):
        for shard_j in range(shard_counts[1]):
            for shard_k in range(shard_counts[2]):
                shard_index = (shard_i, shard_j, shard_k)
                local = jax.tree_util.tree_map(
                    lambda leaf: leaf[shard_index],
                    stacked_control_volume_geometry,
                )
                cells = local.cells
                faces = local.irregular_faces
                starts = tuple(
                    shard_index[axis] * owned_shape[axis]
                    for axis in range(3)
                )
                slices = tuple(
                    slice(start, start + owned_shape[axis])
                    for axis, start in enumerate(starts)
                )
                local_values = {
                    name: getattr(cells, name)
                    for name in (
                        "aggregate_volume",
                        "raw_volume",
                        "is_active_owner",
                        "is_aggregate_target",
                        "is_merged_source",
                        "received_source_count",
                        "member_count",
                    )
                }
                boundary_face_count = jnp.zeros(
                    owned_shape,
                    dtype=jnp.int32,
                ).at[
                    faces.minus_owner_i,
                    faces.minus_owner_j,
                    faces.minus_owner_k,
                ].add(
                    (
                        faces.active
                        & (
                            (faces.kind == CV_FACE_CUT_WALL)
                            | (
                                faces.kind
                                == CV_FACE_PHYSICAL_BOUNDARY
                            )
                        )
                    ).astype(jnp.int32)
                )
                irregular_face_count = jnp.zeros(
                    owned_shape,
                    dtype=jnp.int32,
                ).at[
                    faces.minus_owner_i,
                    faces.minus_owner_j,
                    faces.minus_owner_k,
                ].add(faces.active.astype(jnp.int32))
                irregular_face_count = irregular_face_count.at[
                    faces.plus_owner_i,
                    faces.plus_owner_j,
                    faces.plus_owner_k,
                ].add(
                    (
                        faces.active
                        & faces.has_plus_owner
                    ).astype(jnp.int32)
                )
                remote_face_count = jnp.zeros(
                    owned_shape,
                    dtype=jnp.int32,
                ).at[
                    faces.minus_owner_i,
                    faces.minus_owner_j,
                    faces.minus_owner_k,
                ].add(
                    (
                        faces.active
                        & faces.has_remote_owner
                    ).astype(jnp.int32)
                )
                reconstruction = local.reconstruction
                reconstruction_row_count = jnp.zeros(
                    owned_shape,
                    dtype=jnp.int32,
                ).at[
                    reconstruction.target_i,
                    reconstruction.target_j,
                    reconstruction.target_k,
                ].add(reconstruction.active.astype(jnp.int32))
                local_values["boundary_face_count"] = boundary_face_count
                local_values["irregular_face_count"] = irregular_face_count
                local_values["remote_face_count"] = remote_face_count
                local_values["reconstruction_row_count"] = (
                    reconstruction_row_count
                )
                for name, value in local_values.items():
                    result[name] = result[name].at[slices].set(
                        value
                    )
    return result
def _shifted_torus_exact_field_at_logical_points(
    points: jnp.ndarray,
    stage_time: float | jax.Array,
    field_name: str,
) -> jnp.ndarray:
    points = jnp.asarray(points, dtype=jnp.float64)
    x = points[..., 0]
    theta = points[..., 1]
    zeta = points[..., 2]
    x_mid = 0.5 * (float(shifted_mms.x_min) + float(shifted_mms.x_max))
    theta_shift = theta + float(shifted_mms.sigma) * (x - x_mid)
    coords = (x, theta_shift, theta, zeta)
    if field_name == "density":
        return shifted_mms._shifted_torus_local_density_derivatives(
            coords,
            stage_time,
        )[0]
    if field_name == "omega":
        return shifted_mms._shifted_torus_local_omega_and_derivatives(
            coords,
            stage_time,
        )[0]
    if field_name == "v_ion_parallel":
        return shifted_mms._shifted_torus_local_v_ion_parallel_derivatives(
            coords,
            stage_time,
        )[0]
    if field_name == "v_electron_parallel":
        return shifted_mms._shifted_torus_local_v_electron_parallel_derivatives(
            coords,
            stage_time,
        )[0]
    if field_name == "phi":
        return shifted_mms._shifted_torus_local_phi_derivatives(
            coords,
            stage_time,
        )[0]
    if field_name == "density_v_electron":
        density = shifted_mms._shifted_torus_local_density_derivatives(
            coords,
            stage_time,
        )[0]
        v_electron = (
            shifted_mms._shifted_torus_local_v_electron_parallel_derivatives(
                coords,
                stage_time,
            )[0]
        )
        return density * v_electron
    raise ValueError(f"unsupported shifted-torus exact field {field_name!r}")


def _shifted_torus_exact_field_and_gradient_at_logical_points(
    points: jnp.ndarray,
    stage_time: float | jax.Array,
    field_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate one exact primitive field and its logical gradient."""

    points = jnp.asarray(points, dtype=jnp.float64)
    x = points[..., 0]
    theta = points[..., 1]
    zeta = points[..., 2]
    x_mid = 0.5 * (
        float(shifted_mms.x_min) + float(shifted_mms.x_max)
    )
    theta_shift = theta + float(shifted_mms.sigma) * (x - x_mid)
    coordinates = (x, theta_shift, theta, zeta)
    if field_name == "density":
        derivatives = shifted_mms._shifted_torus_local_density_derivatives(
            coordinates,
            stage_time,
        )
    elif field_name == "omega":
        derivatives = shifted_mms._shifted_torus_local_omega_and_derivatives(
            coordinates,
            stage_time,
        )
    elif field_name == "v_ion_parallel":
        derivatives = (
            shifted_mms._shifted_torus_local_v_ion_parallel_derivatives(
                coordinates,
                stage_time,
            )
        )
    elif field_name == "v_electron_parallel":
        derivatives = (
            shifted_mms._shifted_torus_local_v_electron_parallel_derivatives(
                coordinates,
                stage_time,
            )
        )
    elif field_name == "phi":
        derivatives = shifted_mms._shifted_torus_local_phi_derivatives(
            coordinates,
            stage_time,
        )
    elif field_name == "density_v_electron":
        density = shifted_mms._shifted_torus_local_density_derivatives(
            coordinates,
            stage_time,
        )
        v_electron = (
            shifted_mms._shifted_torus_local_v_electron_parallel_derivatives(
                coordinates,
                stage_time,
            )
        )
        density_value = density[0]
        v_electron_value = v_electron[0]
        density_gradient = jnp.stack(
            (density[1], density[2], density[3]),
            axis=-1,
        )
        v_electron_gradient = jnp.stack(
            (v_electron[1], v_electron[2], v_electron[3]),
            axis=-1,
        )
        return (
            density_value * v_electron_value,
            density_gradient * v_electron_value[..., None]
            + v_electron_gradient * density_value[..., None],
        )
    else:
        raise ValueError(
            f"unsupported shifted-torus primitive field {field_name!r}"
        )
    return (
        derivatives[0],
        jnp.stack(
            (derivatives[1], derivatives[2], derivatives[3]),
            axis=-1,
        ),
    )




def _shifted_torus_exact_time_derivative_at_logical_points(
    points: jnp.ndarray,
    stage_time: float | jax.Array,
) -> Fci4FieldState:
    points = jnp.asarray(points, dtype=jnp.float64)
    x = points[..., 0]
    theta = points[..., 1]
    zeta = points[..., 2]
    x_mid = 0.5 * (float(shifted_mms.x_min) + float(shifted_mms.x_max))
    theta_shift = theta + float(shifted_mms.sigma) * (x - x_mid)
    coords = (x, theta_shift, theta, zeta)
    return Fci4FieldState(
        density=shifted_mms._shifted_torus_local_density_derivatives(
            coords,
            stage_time,
        )[-1],
        omega=shifted_mms._shifted_torus_local_omega_and_derivatives(
            coords,
            stage_time,
        )[-1],
        v_ion_parallel=(
            shifted_mms._shifted_torus_local_v_ion_parallel_derivatives(
                coords,
                stage_time,
            )[-1]
        ),
        v_electron_parallel=(
            shifted_mms._shifted_torus_local_v_electron_parallel_derivatives(
                coords,
                stage_time,
            )[-1]
        ),
    )


def _shifted_torus_analytic_rhs_at_logical_points(
    points: jnp.ndarray,
    stage_time: float | jax.Array,
    parameters: Fci4FieldRhsParameters,
) -> Fci4FieldState:
    points = jnp.asarray(points, dtype=jnp.float64)
    x = points[..., 0]
    theta = points[..., 1]
    zeta = points[..., 2]
    x_mid = 0.5 * (float(shifted_mms.x_min) + float(shifted_mms.x_max))
    theta_shift = theta + float(shifted_mms.sigma) * (x - x_mid)
    coords = (x, theta_shift, theta, zeta)
    density, density_x, density_theta, density_zeta, _density_t = (
        shifted_mms._shifted_torus_local_density_derivatives(
            coords,
            stage_time,
        )
    )
    _omega, omega_x, omega_theta, omega_zeta, _omega_t = (
        shifted_mms._shifted_torus_local_omega_and_derivatives(
            coords,
            stage_time,
        )
    )
    _v_ion, v_ion_x, v_ion_theta, v_ion_zeta, _v_ion_t = (
        shifted_mms._shifted_torus_local_v_ion_parallel_derivatives(
            coords,
            stage_time,
        )
    )
    _v_electron, v_e_x, v_e_theta, v_e_zeta, _v_e_t = (
        shifted_mms._shifted_torus_local_v_electron_parallel_derivatives(
            coords,
            stage_time,
        )
    )
    _phi, phi_x, phi_theta, phi_zeta, _phi_t = (
        shifted_mms._shifted_torus_local_phi_derivatives(
            coords,
            stage_time,
        )
    )
    density_gradient = jnp.stack(
        (density_x, density_theta, density_zeta),
        axis=-1,
    )
    omega_gradient = jnp.stack(
        (omega_x, omega_theta, omega_zeta),
        axis=-1,
    )
    v_ion_gradient = jnp.stack(
        (v_ion_x, v_ion_theta, v_ion_zeta),
        axis=-1,
    )
    v_electron_gradient = jnp.stack(
        (v_e_x, v_e_theta, v_e_zeta),
        axis=-1,
    )
    phi_gradient = jnp.stack(
        (phi_x, phi_theta, phi_zeta),
        axis=-1,
    )
    J, _g_contra, g_cov, B_contra, Bmag, _projector = (
        _shifted_torus_metric_payload_jax(points)
    )
    b_contra = B_contra / jnp.maximum(Bmag[..., None], 1.0e-30)
    b_cov = jnp.einsum("...ij,...j->...i", g_cov, b_contra)

    def poisson(first: jnp.ndarray, second: jnp.ndarray) -> jnp.ndarray:
        return (
            jnp.sum(b_cov * jnp.cross(first, second), axis=-1)
            / jnp.maximum(J, 1.0e-30)
        )

    curvature = _shifted_torus_curvature_at_logical_points(points)
    curvature_density = jnp.einsum(
        "...i,...i->...",
        curvature,
        density_gradient,
    )
    curvature_phi = jnp.einsum(
        "...i,...i->...",
        curvature,
        phi_gradient,
    )
    grad_parallel_density = jnp.einsum(
        "...i,...i->...",
        b_contra,
        density_gradient,
    )
    grad_parallel_phi = jnp.einsum(
        "...i,...i->...",
        b_contra,
        phi_gradient,
    )
    grad_parallel_v_ion = jnp.einsum(
        "...i,...i->...",
        b_contra,
        v_ion_gradient,
    )
    grad_parallel_v_electron = jnp.einsum(
        "...i,...i->...",
        b_contra,
        v_electron_gradient,
    )
    exact_density = _shifted_torus_exact_field_at_logical_points(
        points,
        stage_time,
        "density",
    )
    exact_v_electron = _shifted_torus_exact_field_at_logical_points(
        points,
        stage_time,
        "v_electron_parallel",
    )
    parallel_density_flux_divergence = (
        shifted_mms._parallel_density_flux_divergence(
            x=x,
            theta_shift=theta_shift,
            density=exact_density,
            v_electron_parallel=exact_v_electron,
            density_grad=density_gradient,
            v_electron_grad=v_electron_gradient,
            b_contra=b_contra,
            jacobian=J,
        )
    )
    density_safe = jnp.maximum(density, 1.0e-30)
    rho_star = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    te = jnp.asarray(parameters.Te, dtype=jnp.float64)
    mi_over_me = jnp.asarray(parameters.mi_over_me, dtype=jnp.float64)
    return Fci4FieldState(
        density=(
            -poisson(phi_gradient, density_gradient) / (rho_star * Bmag)
            + (2.0 * te / Bmag) * curvature_density
            - (2.0 * density / Bmag) * curvature_phi
            - parallel_density_flux_divergence
        ),
        omega=(
            -poisson(phi_gradient, omega_gradient) / (rho_star * Bmag)
            + (Bmag * Bmag / density_safe)
            * (grad_parallel_v_ion - grad_parallel_v_electron)
            + (2.0 * Bmag * te / density_safe) * curvature_density
        ),
        v_ion_parallel=(
            -poisson(phi_gradient, v_ion_gradient) / (rho_star * Bmag)
            - (te / density_safe) * grad_parallel_density
        ),
        v_electron_parallel=(
            -poisson(phi_gradient, v_electron_gradient) / (rho_star * Bmag)
            + mi_over_me * grad_parallel_phi
            - mi_over_me * (te / density_safe) * grad_parallel_density
        ),
    )


def _shifted_torus_mms_source_at_logical_points(
    points: jnp.ndarray,
    stage_time: float | jax.Array,
    parameters: Fci4FieldRhsParameters,
) -> Fci4FieldState:
    exact_t = _shifted_torus_exact_time_derivative_at_logical_points(
        points,
        stage_time,
    )
    analytic_rhs = _shifted_torus_analytic_rhs_at_logical_points(
        points,
        stage_time,
        parameters,
    )
    return Fci4FieldState(
        density=exact_t.density - analytic_rhs.density,
        omega=exact_t.omega - analytic_rhs.omega,
        v_ion_parallel=(
            exact_t.v_ion_parallel - analytic_rhs.v_ion_parallel
        ),
        v_electron_parallel=(
            exact_t.v_electron_parallel
            - analytic_rhs.v_electron_parallel
        ),
    )


def _integrate_local_four_field_over_fluid(
    geometry: LocalFciGeometry3D,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
    evaluator,
) -> Fci4FieldState:
    """Project a pointwise four-field evaluator with compact 3x3x3 quadrature."""

    faces = (
        jnp.asarray(geometry.grid.x.faces_owned, dtype=jnp.float64),
        jnp.asarray(geometry.grid.y.faces_owned, dtype=jnp.float64),
        jnp.asarray(geometry.grid.z.faces_owned, dtype=jnp.float64),
    )
    lower = (
        jnp.broadcast_to(faces[0][:-1, None, None], geometry.owned_shape),
        jnp.broadcast_to(faces[1][None, :-1, None], geometry.owned_shape),
        jnp.broadcast_to(faces[2][None, None, :-1], geometry.owned_shape),
    )
    upper = (
        jnp.broadcast_to(faces[0][1:, None, None], geometry.owned_shape),
        jnp.broadcast_to(faces[1][None, 1:, None], geometry.owned_shape),
        jnp.broadcast_to(faces[2][None, None, 1:], geometry.owned_shape),
    )
    box = _box_bounds()
    solid_lower = tuple(
        jnp.maximum(value, float(box[axis][0]))
        for axis, value in enumerate(lower)
    )
    solid_upper = tuple(
        jnp.minimum(value, float(box[axis][1]))
        for axis, value in enumerate(upper)
    )

    # The trailing axis holds the full-cell and clipped-solid regions. Their
    # signed integrals produce the fluid-cell integral in one traced loop.
    region_lower = tuple(
        jnp.stack((full, solid), axis=-1)
        for full, solid in zip(lower, solid_lower)
    )
    region_upper = tuple(
        jnp.stack((full, solid), axis=-1)
        for full, solid in zip(upper, solid_upper)
    )
    midpoint = tuple(
        0.5 * (hi + lo)
        for lo, hi in zip(region_lower, region_upper)
    )
    half_width = tuple(
        0.5 * (hi - lo)
        for lo, hi in zip(region_lower, region_upper)
    )
    valid = (
        (half_width[0] > 0.0)
        & (half_width[1] > 0.0)
        & (half_width[2] > 0.0)
    )
    nodes = jnp.asarray(_GAUSS3_NODES, dtype=jnp.float64)
    weights = jnp.asarray(_GAUSS3_WEIGHTS, dtype=jnp.float64)
    region_sign = jnp.asarray((1.0, -1.0), dtype=jnp.float64)
    zeros = jnp.zeros(geometry.owned_shape, dtype=jnp.float64)
    initial = Fci4FieldState(
        density=zeros,
        omega=zeros,
        v_ion_parallel=zeros,
        v_electron_parallel=zeros,
    )

    def body(index: int, accumulated: Fci4FieldState) -> Fci4FieldState:
        node_i = index // 9
        node_j = (index // 3) % 3
        node_k = index % 3
        x = midpoint[0] + half_width[0] * nodes[node_i]
        theta = midpoint[1] + half_width[1] * nodes[node_j]
        zeta = midpoint[2] + half_width[2] * nodes[node_k]
        points = jnp.stack((x, theta, zeta), axis=-1)
        values = evaluator(points)
        jacobian = _shifted_torus_metric_payload_jax(points)[0]
        measure = (
            weights[node_i]
            * weights[node_j]
            * weights[node_k]
            * half_width[0]
            * half_width[1]
            * half_width[2]
            * jacobian
        )
        signed_measure = jnp.where(
            valid,
            measure * region_sign,
            0.0,
        )

        def integrate(value: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(
                signed_measure * jnp.asarray(value, dtype=jnp.float64),
                axis=-1,
            )

        return Fci4FieldState(
            density=accumulated.density + integrate(values.density),
            omega=accumulated.omega + integrate(values.omega),
            v_ion_parallel=(
                accumulated.v_ion_parallel
                + integrate(values.v_ion_parallel)
            ),
            v_electron_parallel=(
                accumulated.v_electron_parallel
                + integrate(values.v_electron_parallel)
            ),
        )

    raw_integrals = lax.fori_loop(0, 27, body, initial)
    cells = control_volume_geometry.cells

    def to_aggregate_average(integral: jnp.ndarray) -> jnp.ndarray:
        raw_average = integral / jnp.maximum(cells.raw_volume, 1.0e-30)
        return _agglomerate_control_volume_average(raw_average, cells)

    return Fci4FieldState(
        density=to_aggregate_average(raw_integrals.density),
        omega=to_aggregate_average(raw_integrals.omega),
        v_ion_parallel=to_aggregate_average(raw_integrals.v_ion_parallel),
        v_electron_parallel=to_aggregate_average(
            raw_integrals.v_electron_parallel
        ),
    )


def _project_local_mms_source_to_control_volumes(
    geometry: LocalFciGeometry3D,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
    stage_time: float | jax.Array,
    parameters: Fci4FieldRhsParameters,
) -> Fci4FieldState:
    return _integrate_local_four_field_over_fluid(
        geometry,
        control_volume_geometry,
        lambda points: _shifted_torus_mms_source_at_logical_points(
            points,
            stage_time,
            parameters,
        ),
    )


def _project_local_exact_time_derivative_to_control_volumes(
    geometry: LocalFciGeometry3D,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
    stage_time: float | jax.Array,
) -> Fci4FieldState:
    return _integrate_local_four_field_over_fluid(
        geometry,
        control_volume_geometry,
        lambda points: _shifted_torus_exact_time_derivative_at_logical_points(
            points,
            stage_time,
        ),
    )


def _integrate_local_scalar_over_fluid(
    geometry: LocalFciGeometry3D,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
    evaluator,
) -> jnp.ndarray:
    def four_field_evaluator(points: jnp.ndarray) -> Fci4FieldState:
        value = evaluator(points)
        return Fci4FieldState(
            density=value,
            omega=value,
            v_ion_parallel=value,
            v_electron_parallel=value,
        )

    return _integrate_local_four_field_over_fluid(
        geometry,
        control_volume_geometry,
        four_field_evaluator,
    ).density


def _shifted_torus_operator_reference_at_logical_points(
    points: jnp.ndarray,
    stage_time: float | jax.Array,
    operator_name: str,
) -> jnp.ndarray:
    points = jnp.asarray(points, dtype=jnp.float64)
    x = points[..., 0]
    theta = points[..., 1]
    zeta = points[..., 2]
    x_mid = 0.5 * (float(shifted_mms.x_min) + float(shifted_mms.x_max))
    theta_shift = theta + float(shifted_mms.sigma) * (x - x_mid)
    coordinates = (x, theta_shift, theta, zeta)
    density, density_gradient = (
        _shifted_torus_exact_field_and_gradient_at_logical_points(
            points,
            stage_time,
            "density",
        )
    )
    v_electron, v_electron_gradient = (
        _shifted_torus_exact_field_and_gradient_at_logical_points(
            points,
            stage_time,
            "v_electron_parallel",
        )
    )
    _phi, phi_gradient = (
        _shifted_torus_exact_field_and_gradient_at_logical_points(
            points,
            stage_time,
            "phi",
        )
    )
    J, _g_contra, g_cov, B_contra, Bmag, _projector = (
        _shifted_torus_metric_payload_jax(points)
    )
    unit_b = B_contra / jnp.maximum(Bmag[..., None], 1.0e-30)
    field_suffixes = {
        "density": "density",
        "omega": "omega",
        "v_ion": "v_ion_parallel",
        "v_electron": "v_electron_parallel",
        "phi": "phi",
    }
    if operator_name.startswith("grad_parallel_"):
        suffix = operator_name.removeprefix("grad_parallel_")
        field_name = field_suffixes.get(suffix)
        if field_name is None:
            raise ValueError(
                f"unsupported parallel-gradient field {suffix!r}"
            )
        _value, gradient = (
            _shifted_torus_exact_field_and_gradient_at_logical_points(
                points,
                stage_time,
                field_name,
            )
        )
        return jnp.einsum("...i,...i->...", unit_b, gradient)
    if operator_name == "parallel_density_flux_divergence":
        return shifted_mms._parallel_density_flux_divergence(
            x=x,
            theta_shift=theta_shift,
            density=density,
            v_electron_parallel=v_electron,
            density_grad=density_gradient,
            v_electron_grad=v_electron_gradient,
            b_contra=unit_b,
            jacobian=J,
        )
    if operator_name.startswith("poisson_"):
        suffix = operator_name.removeprefix("poisson_")
        field_name = field_suffixes.get(suffix)
        if field_name is None or field_name == "phi":
            raise ValueError(f"unsupported Poisson-bracket field {suffix!r}")
        _value, gradient = (
            _shifted_torus_exact_field_and_gradient_at_logical_points(
                points,
                stage_time,
                field_name,
            )
        )
        unit_b_cov = jnp.einsum(
            "...ij,...j->...i",
            g_cov,
            unit_b,
        )
        return (
            jnp.sum(
                unit_b_cov
                * jnp.cross(phi_gradient, gradient),
                axis=-1,
            )
            / jnp.maximum(J, 1.0e-30)
        )
    if operator_name.startswith("curvature_"):
        suffix = operator_name.removeprefix("curvature_")
        field_name = field_suffixes.get(suffix)
        if field_name is None:
            raise ValueError(f"unsupported curvature field {suffix!r}")
        _value, gradient = (
            _shifted_torus_exact_field_and_gradient_at_logical_points(
                points,
                stage_time,
                field_name,
            )
        )
        return jnp.einsum(
            "...i,...i->...",
            _shifted_torus_curvature_at_logical_points(points),
            gradient,
        )
    raise ValueError(f"unsupported shifted-torus operator {operator_name!r}")


def _control_volume_exact_boundary_bc(
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
    stage_time: float | jax.Array,
    field_name: str,
    *,
    value_offset: float | jax.Array = 0.0,
) -> LocalControlVolumeBoundaryBC3D:
    """Collocate exact Dirichlet data with compact physical wall rows."""

    faces = control_volume_geometry.irregular_faces
    boundary_active = (
        jnp.asarray(faces.active, dtype=bool)
        & (
            (jnp.asarray(faces.kind, dtype=jnp.int32) == CV_FACE_CUT_WALL)
            | (
                jnp.asarray(faces.kind, dtype=jnp.int32)
                == CV_FACE_PHYSICAL_BOUNDARY
            )
        )
    )
    quadrature_value = _shifted_torus_exact_field_at_logical_points(
        faces.quadrature_points,
        stage_time,
        field_name,
    )
    quadrature_value = quadrature_value + jnp.asarray(
        value_offset,
        dtype=jnp.float64,
    )
    quadrature_measure = (
        jnp.asarray(faces.J, dtype=jnp.float64)
        * jnp.linalg.norm(
            jnp.asarray(faces.area_covector_weight, dtype=jnp.float64),
            axis=-1,
        )
        * jnp.asarray(faces.quadrature_active, dtype=jnp.float64)
    )
    measure = jnp.sum(quadrature_measure, axis=(1, 2))
    centroid = jnp.sum(
        quadrature_measure[..., None] * faces.quadrature_points,
        axis=(1, 2),
    ) / jnp.maximum(measure[:, None], 1.0e-30)
    centroid_value = _shifted_torus_exact_field_at_logical_points(
        centroid,
        stage_time,
        field_name,
    )
    centroid_value = centroid_value + jnp.asarray(
        value_offset,
        dtype=jnp.float64,
    )
    return LocalControlVolumeBoundaryBC3D(
        kind=jnp.where(boundary_active, BC_DIRICHLET, BC_NONE),
        centroid_value=jnp.where(boundary_active, centroid_value, 0.0),
        quadrature_value=jnp.where(
            boundary_active[:, None, None],
            quadrature_value,
            0.0,
        ),
        active=boundary_active,
        max_rows=faces.max_rows,
        max_patches=faces.max_patches,
    )


def _multiply_local_dirichlet_face_bc(
    left: LocalBoundaryFaceBC3D,
    right: LocalBoundaryFaceBC3D,
) -> LocalBoundaryFaceBC3D:
    """Multiply collocated Dirichlet data for one derived scalar field."""

    if left.layout != right.layout:
        raise ValueError("face BC operands must share one HaloLayout3D")

    def axis_payload(
        left_kind: jnp.ndarray,
        right_kind: jnp.ndarray,
        left_value: jnp.ndarray,
        right_value: jnp.ndarray,
        left_mask: jnp.ndarray,
        right_mask: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mask = (
            left_mask
            & right_mask
            & (left_kind == BC_DIRICHLET)
            & (right_kind == BC_DIRICHLET)
        )
        return (
            jnp.where(mask, BC_DIRICHLET, BC_NONE),
            jnp.where(mask, left_value * right_value, 0.0),
            mask,
        )

    x = axis_payload(
        left.kind_x,
        right.kind_x,
        left.value_x,
        right.value_x,
        left.mask_x,
        right.mask_x,
    )
    y = axis_payload(
        left.kind_y,
        right.kind_y,
        left.value_y,
        right.value_y,
        left.mask_y,
        right.mask_y,
    )
    z = axis_payload(
        left.kind_z,
        right.kind_z,
        left.value_z,
        right.value_z,
        left.mask_z,
        right.mask_z,
    )
    return LocalBoundaryFaceBC3D(
        kind_x=x[0],
        kind_y=y[0],
        kind_z=z[0],
        value_x=x[1],
        value_y=y[1],
        value_z=z[1],
        mask_x=x[2],
        mask_y=y[2],
        mask_z=z[2],
        layout=left.layout,
    )


def _agglomerate_control_volume_average(
    values_owned: jnp.ndarray,
    cells: LocalControlVolumeCellGeometry3D,
) -> jnp.ndarray:
    """Map raw cell averages into their unique active control-volume owners."""

    values = jnp.asarray(values_owned, dtype=jnp.float64)
    weighted = cells.raw_volume * values
    aggregate_weighted = jnp.zeros(cells.shape, dtype=jnp.float64).at[
        cells.owner_i,
        cells.owner_j,
        cells.owner_k,
    ].add(weighted)
    result = aggregate_weighted / jnp.maximum(
        cells.aggregate_volume,
        1.0e-30,
    )
    return jnp.where(cells.is_active_owner, result, 0.0)


def _expand_control_volume_owner_values(
    owner_values: jnp.ndarray,
    cells: LocalControlVolumeCellGeometry3D,
) -> jnp.ndarray:
    """Fill every positive-volume storage cell from its mapped owner value."""

    values = jnp.asarray(owner_values, dtype=jnp.float64)
    expanded = values[cells.owner_i, cells.owner_j, cells.owner_k]
    return jnp.where(cells.raw_volume > 0.0, expanded, 0.0)
