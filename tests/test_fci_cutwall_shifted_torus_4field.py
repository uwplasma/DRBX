"""Shifted-torus four-field MMS tests with a closed embedded cut-wall box."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path
import sys
import time as time_module

from drbx.runtime import configure_jax_runtime

_JAX_COMPILATION_CACHE_DIR = configure_jax_runtime(precision="float64")

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

from drbx.geometry import (
    FciGeometry3D,
    LocalControlVolumeCellGeometry3D,
    LocalCellVolumeGeometry3D,
    LocalDomain3D,
    LocalFciGeometry3D,
    LocalRegularFaceGeometry3D,
    StencilBuilderContext,
    build_local_control_volume_cell_geometry,
    build_local_conservative_stencil_from_field,
    build_local_stencil_from_field,
)
from drbx.native import (
    Fci4FieldRhsParameters,
    Fci4FieldState,
    SpmdGmresConfig,
    precompute_local_moment_reconstruction,
)
from drbx.native.fci_boundaries import (
    BC_DIRICHLET,
    BC_NONE,
    CV_FACE_CUT_WALL,
    CV_FACE_INTERIOR,
    CV_FACE_PARTIAL,
    CV_FACE_PHYSICAL_BOUNDARY,
    CV_RECONSTRUCTION_EQUATION_CELL,
    CV_RECONSTRUCTION_EQUATION_DIRICHLET,
    CV_RECONSTRUCTION_EQUATION_REMOTE_CELL,
    LocalBoundaryFaceBC3D,
    LocalCellGradient3D,
    LocalControlVolumeBoundaryBC3D,
    LocalControlVolumeFaceRows3D,
    LocalEmbeddedControlVolumeGeometry3D,
    LocalMomentReconstruction3D,
    LocalRegularBoundaryMomentClosure3D,
    LocalRegularTransitionFaceRows3D,
)
from drbx.native.fci_halo import (
    HaloExchange3D,
    LocalHaloClosure3D,
    PhysicalGhostCellFiller3D,
    TopologyHaloFiller3D,
    LocalPeriodicTopologyRule3D,
)
from drbx.native.fci_model import inject_owned_field_to_halo, inject_owned_state_to_halo
from drbx.native.fci_operators import (
    LocalPerpLaplacianInverseSolver,
    _apply_local_face_flux_bc,
    _apply_local_face_value_dirichlet_bc,
    _axis_slice_nd,
    _lift_cell_field_to_faces,
    _local_axis_face_values_from_stencil,
    _evaluate_local_regular_transition_functional,
    _take_stencil_finite_difference,
    build_local_control_volume_polynomial_from_field,
    build_local_perp_laplacian_stencil,
    _local_control_volume_irregular_parallel_flux,
    _local_control_volume_irregular_projected_flux,
    evaluate_local_control_volume_polynomial,
    local_curvature_op_from_gradient,
    local_grad_parallel_op_from_gradient,
    local_parallel_flux_div_op,
    local_control_volume_product_average,
    local_perp_laplacian_conservative_op,
    local_poisson_bracket_op_from_gradients,
)

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


def stack_local_shard_pytree(*args, **kwargs):
    from mms_domain_decomp_helpers import stack_local_shard_pytree as impl

    return impl(*args, **kwargs)


def make_mesh_for_shard_counts(*args, **kwargs):
    from mms_domain_decomp_helpers import make_mesh_for_shard_counts as impl

    return impl(*args, **kwargs)


BOX_X_FRACTION_RANGE = (0.25, 0.75)
BOX_THETA_CENTER = 1.5 * np.pi
BOX_THETA_HALF_WIDTH = 0.35
BOX_ZETA_RANGE = (0.45, 4.25)


def _shape_from_resolution(resolution: int) -> tuple[int, int, int]:
    n = int(resolution)
    return (n, n, n)


def _box_bounds(
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    radial_span = float(shifted_mms.x_max) - float(shifted_mms.x_min)
    return (
        (
            float(shifted_mms.x_min) + BOX_X_FRACTION_RANGE[0] * radial_span,
            float(shifted_mms.x_min) + BOX_X_FRACTION_RANGE[1] * radial_span,
        ),
        (
            BOX_THETA_CENTER - BOX_THETA_HALF_WIDTH,
            BOX_THETA_CENTER + BOX_THETA_HALF_WIDTH,
        ),
        BOX_ZETA_RANGE,
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


_GAUSS3_NODES = np.asarray((-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)))
_GAUSS3_WEIGHTS = np.asarray((5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0))
_GAUSS2_NODES = np.asarray((-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)))


def _shifted_torus_metric_payload_numpy(
    points: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Evaluate shifted-torus metric, field, and perpendicular projector."""

    points = np.asarray(points, dtype=np.float64)
    x = points[..., 0]
    theta = points[..., 1]
    x_mid = 0.5 * (float(shifted_mms.x_min) + float(shifted_mms.x_max))
    theta_shift = theta + float(shifted_mms.sigma) * (x - x_mid)
    cos_theta = np.cos(theta_shift)
    sin_theta = np.sin(theta_shift)
    alpha = float(shifted_mms.alpha_value)
    radius = float(shifted_mms.r0) + alpha * x + x * cos_theta
    q_value = 1.0 + alpha * cos_theta
    jacobian = radius * x * q_value
    zeros = np.zeros_like(jacobian)

    g_contra = np.stack(
        (
            np.stack(
                (
                    1.0 / q_value**2,
                    alpha * sin_theta / (x * q_value**2),
                    zeros,
                ),
                axis=-1,
            ),
            np.stack(
                (
                    alpha * sin_theta / (x * q_value**2),
                    (1.0 + 2.0 * alpha * cos_theta + alpha**2)
                    / (x**2 * q_value**2),
                    zeros,
                ),
                axis=-1,
            ),
            np.stack((zeros, zeros, 1.0 / radius**2), axis=-1),
        ),
        axis=-2,
    )
    g_cov = np.stack(
        (
            np.stack(
                (
                    1.0 + 2.0 * alpha * cos_theta + alpha**2,
                    -alpha * x * sin_theta,
                    zeros,
                ),
                axis=-1,
            ),
            np.stack((-alpha * x * sin_theta, x**2, zeros), axis=-1),
            np.stack((zeros, zeros, radius**2), axis=-1),
        ),
        axis=-2,
    )
    B_contra = np.stack(
        (
            zeros,
            float(shifted_mms.iota) * float(shifted_mms.c_phi) / jacobian,
            float(shifted_mms.c_phi) / jacobian,
        ),
        axis=-1,
    )
    Bmag = np.sqrt(
        np.einsum("...i,...ij,...j->...", B_contra, g_cov, B_contra)
    )
    unit_b = B_contra / np.maximum(Bmag[..., None], 1.0e-30)
    projector = g_contra - unit_b[..., :, None] * unit_b[..., None, :]
    return jacobian, g_contra, g_cov, B_contra, Bmag, projector


def _integrate_shifted_torus_rectangular_moments(
    lower: tuple[np.ndarray, np.ndarray, np.ndarray],
    upper: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate ``J``, first moments, and origin second moments on rectangles."""

    shape = np.broadcast_shapes(
        np.shape(lower[0]),
        np.shape(lower[1]),
        np.shape(lower[2]),
    )
    lower_b = tuple(np.broadcast_to(np.asarray(value, dtype=np.float64), shape) for value in lower)
    upper_b = tuple(np.broadcast_to(np.asarray(value, dtype=np.float64), shape) for value in upper)
    half = tuple(0.5 * (hi - lo) for lo, hi in zip(lower_b, upper_b))
    midpoint = tuple(0.5 * (hi + lo) for lo, hi in zip(lower_b, upper_b))
    valid = (half[0] > 0.0) & (half[1] > 0.0) & (half[2] > 0.0)
    volume = np.zeros(shape, dtype=np.float64)
    first = np.zeros(shape + (3,), dtype=np.float64)
    second_origin = np.zeros(shape + (3, 3), dtype=np.float64)
    third_origin = np.zeros(shape + (3, 3, 3), dtype=np.float64)
    for ax, wx in zip(_GAUSS3_NODES, _GAUSS3_WEIGHTS):
        x = midpoint[0] + half[0] * ax
        for ay, wy in zip(_GAUSS3_NODES, _GAUSS3_WEIGHTS):
            theta = midpoint[1] + half[1] * ay
            x_mid = 0.5 * (
                float(shifted_mms.x_min) + float(shifted_mms.x_max)
            )
            theta_shift = theta + float(shifted_mms.sigma) * (x - x_mid)
            radius = (
                float(shifted_mms.r0)
                + float(shifted_mms.alpha_value) * x
                + x * np.cos(theta_shift)
            )
            q_value = 1.0 + float(shifted_mms.alpha_value) * np.cos(theta_shift)
            jacobian = radius * x * q_value
            for az, wz in zip(_GAUSS3_NODES, _GAUSS3_WEIGHTS):
                zeta = midpoint[2] + half[2] * az
                weight = (
                    wx
                    * wy
                    * wz
                    * half[0]
                    * half[1]
                    * half[2]
                    * jacobian
                )
                weight = np.where(valid, weight, 0.0)
                point = np.stack((x, theta, zeta), axis=-1)
                volume += weight
                first += weight[..., None] * point
                second_origin += (
                    weight[..., None, None]
                    * point[..., :, None]
                    * point[..., None, :]
                )
                third_origin += (
                    weight[..., None, None, None]
                    * point[..., :, None, None]
                    * point[..., None, :, None]
                    * point[..., None, None, :]
                )
    return volume, first, second_origin, third_origin


def _closed_box_fluid_moments_3point(
    geometry: LocalFciGeometry3D,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return true local fluid moments from full-cell minus solid integration."""

    x_faces = np.asarray(geometry.grid.x.faces_owned, dtype=np.float64)
    y_faces = np.asarray(geometry.grid.y.faces_owned, dtype=np.float64)
    z_faces = np.asarray(geometry.grid.z.faces_owned, dtype=np.float64)
    lower = (
        x_faces[:-1, None, None],
        y_faces[None, :-1, None],
        z_faces[None, None, :-1],
    )
    upper = (
        x_faces[1:, None, None],
        y_faces[None, 1:, None],
        z_faces[None, None, 1:],
    )
    full_volume, full_first, full_second, full_third = (
        _integrate_shifted_torus_rectangular_moments(lower, upper)
    )
    bounds = _box_bounds()
    solid_lower = tuple(
        np.maximum(value, float(bounds[axis][0]))
        for axis, value in enumerate(lower)
    )
    solid_upper = tuple(
        np.minimum(value, float(bounds[axis][1]))
        for axis, value in enumerate(upper)
    )
    solid_volume, solid_first, solid_second, solid_third = (
        _integrate_shifted_torus_rectangular_moments(
            solid_lower,
            solid_upper,
        )
    )
    raw_volume = np.maximum(full_volume - solid_volume, 0.0)
    raw_first = full_first - solid_first
    raw_second_origin = full_second - solid_second
    raw_third_origin = full_third - solid_third
    x_center, y_center, z_center = np.meshgrid(
        np.asarray(geometry.grid.x.centers_owned, dtype=np.float64),
        np.asarray(geometry.grid.y.centers_owned, dtype=np.float64),
        np.asarray(geometry.grid.z.centers_owned, dtype=np.float64),
        indexing="ij",
    )
    ordinary_center = np.stack((x_center, y_center, z_center), axis=-1)
    safe_volume = np.maximum(raw_volume, 1.0e-30)
    centroid = raw_first / safe_volume[..., None]
    centroid = np.where(
        (raw_volume > 1.0e-24)[..., None],
        centroid,
        ordinary_center,
    )
    second_moment = raw_second_origin / safe_volume[..., None, None]
    second_moment -= centroid[..., :, None] * centroid[..., None, :]
    second_moment = 0.5 * (
        second_moment + np.swapaxes(second_moment, -1, -2)
    )
    diagonal = np.diagonal(second_moment, axis1=-2, axis2=-1)
    clipped_diagonal = np.maximum(diagonal, 0.0)
    for axis in range(3):
        second_moment[..., axis, axis] = clipped_diagonal[..., axis]
    second_moment = np.where(
        (raw_volume > 1.0e-24)[..., None, None],
        second_moment,
        0.0,
    )
    second_origin = raw_second_origin / safe_volume[..., None, None]
    third_origin = raw_third_origin / safe_volume[..., None, None, None]
    third_moment = third_origin - (
        centroid[..., :, None, None] * second_origin[..., None, :, :]
        + centroid[..., None, :, None] * second_origin[..., :, None, :]
        + centroid[..., None, None, :] * second_origin[..., :, :, None]
    ) + 2.0 * (
        centroid[..., :, None, None]
        * centroid[..., None, :, None]
        * centroid[..., None, None, :]
    )
    third_moment = np.where(
        (raw_volume > 1.0e-24)[..., None, None, None], third_moment, 0.0
    )
    return raw_volume, centroid, second_moment, third_moment, full_volume


def _build_shifted_torus_regular_boundary_closure(
    geometry: LocalFciGeometry3D,
    cells: LocalControlVolumeCellGeometry3D,
) -> LocalRegularBoundaryMomentClosure3D:
    """Build radial Dirichlet face/owner derivatives from FV moments."""

    face_shapes = tuple(
        geometry.layout.face_control_shape(axis) for axis in range(3)
    )
    face_weights = [
        np.zeros(shape + (4,), dtype=np.float64) for shape in face_shapes
    ]
    owner_weights = [
        np.zeros(shape + (4,), dtype=np.float64) for shape in face_shapes
    ]
    valid = [np.zeros(shape, dtype=bool) for shape in face_shapes]
    if geometry.owned_shape[0] < 3:
        return LocalRegularBoundaryMomentClosure3D(
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

    x_faces = np.asarray(geometry.grid.x.faces_owned, dtype=np.float64)
    theta_faces = np.asarray(geometry.grid.y.faces_owned, dtype=np.float64)
    zeta_faces = np.asarray(geometry.grid.z.faces_owned, dtype=np.float64)
    raw_volume = np.asarray(cells.raw_volume, dtype=np.float64)
    raw_centroid = np.asarray(cells.raw_centroid, dtype=np.float64)
    raw_second_moment = np.asarray(
        cells.raw_second_moment,
        dtype=np.float64,
    )
    active_owner = np.asarray(cells.is_active_owner, dtype=bool)
    merged_source = np.asarray(cells.is_merged_source, dtype=bool)

    def side_weights(
        side: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower_side = side == 0
        wall = x_faces[0] if lower_side else x_faces[-1]
        inward_sign = 1.0 if lower_side else -1.0
        indices = (
            np.arange(3, dtype=np.int32)
            if lower_side
            else np.arange(geometry.owned_shape[0] - 1, geometry.owned_shape[0] - 4, -1)
        )
        x_lower = x_faces[indices]
        x_upper = x_faces[indices + 1]
        shape = (3, geometry.owned_shape[1], geometry.owned_shape[2])
        midpoint = (
            0.5 * (x_lower + x_upper)[:, None, None],
            0.5 * (theta_faces[:-1] + theta_faces[1:])[None, :, None],
            0.5 * (zeta_faces[:-1] + zeta_faces[1:])[None, None, :],
        )
        half_width = (
            0.5 * (x_upper - x_lower)[:, None, None],
            0.5 * (theta_faces[1:] - theta_faces[:-1])[None, :, None],
            0.5 * (zeta_faces[1:] - zeta_faces[:-1])[None, None, :],
        )
        third_integral = np.zeros(shape, dtype=np.float64)
        for node_x, weight_x in zip(_GAUSS3_NODES, _GAUSS3_WEIGHTS):
            x = midpoint[0] + half_width[0] * node_x
            for node_y, weight_y in zip(_GAUSS3_NODES, _GAUSS3_WEIGHTS):
                theta = midpoint[1] + half_width[1] * node_y
                for node_z, weight_z in zip(
                    _GAUSS3_NODES,
                    _GAUSS3_WEIGHTS,
                ):
                    zeta = midpoint[2] + half_width[2] * node_z
                    points = np.stack(
                        np.broadcast_arrays(x, theta, zeta),
                        axis=-1,
                    )
                    jacobian = _shifted_torus_metric_payload_numpy(points)[0]
                    measure = (
                        weight_x
                        * weight_y
                        * weight_z
                        * half_width[0]
                        * half_width[1]
                        * half_width[2]
                        * jacobian
                    )
                    inward_coordinate = inward_sign * (x - wall)
                    third_integral += measure * inward_coordinate**3

        volume = raw_volume[indices]
        centroid_offset = inward_sign * (
            raw_centroid[indices, ..., 0] - wall
        )
        second_origin = (
            raw_second_moment[indices, ..., 0, 0]
            + centroid_offset**2
        )
        third_origin = third_integral / np.maximum(volume, 1.0e-30)
        matrix = np.zeros(shape[1:] + (4, 4), dtype=np.float64)
        matrix[..., 0, 0] = 1.0
        matrix[..., 1:, 0] = 1.0
        matrix[..., 1:, 1] = np.moveaxis(centroid_offset, 0, -1)
        matrix[..., 1:, 2] = np.moveaxis(second_origin, 0, -1)
        matrix[..., 1:, 3] = np.moveaxis(third_origin, 0, -1)
        face_derivative = np.zeros(shape[1:] + (4,), dtype=np.float64)
        face_derivative[..., 1] = inward_sign
        face_result = np.linalg.solve(
            np.swapaxes(matrix, -1, -2),
            face_derivative[..., None],
        )[..., 0]
        owner_coordinate = centroid_offset[0]
        owner_derivative = np.stack(
            (
                np.zeros_like(owner_coordinate),
                np.full_like(owner_coordinate, inward_sign),
                2.0 * inward_sign * owner_coordinate,
                3.0 * inward_sign * owner_coordinate**2,
            ),
            axis=-1,
        )
        owner_result = np.linalg.solve(
            np.swapaxes(matrix, -1, -2),
            owner_derivative[..., None],
        )[..., 0]
        face_reproduced = np.einsum(
            "...rc,...r->...c",
            matrix,
            face_result,
        )
        owner_reproduced = np.einsum(
            "...rc,...r->...c",
            matrix,
            owner_result,
        )
        if not np.allclose(
            face_reproduced,
            face_derivative,
            rtol=1.0e-11,
            atol=1.0e-11,
        ):
            raise ValueError(
                "regular radial face derivative weights do not reproduce "
                "the cubic moment basis"
            )
        if not np.allclose(
            owner_reproduced,
            owner_derivative,
            rtol=1.0e-11,
            atol=1.0e-11,
        ):
            raise ValueError(
                "regular radial owner derivative weights do not reproduce "
                "the cubic moment basis"
            )
        side_valid = np.all(
            active_owner[indices]
            & (~merged_source[indices])
            & (volume > 1.0e-24),
            axis=0,
        )
        return face_result, owner_result, side_valid

    if np.isclose(x_faces[0], float(shifted_mms.x_min)):
        (
            face_weights[0][0],
            owner_weights[0][0],
            valid[0][0],
        ) = side_weights(0)
    if np.isclose(x_faces[-1], float(shifted_mms.x_max)):
        (
            face_weights[0][-1],
            owner_weights[0][-1],
            valid[0][-1],
        ) = side_weights(1)
    return LocalRegularBoundaryMomentClosure3D(
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


def _print_shifted_torus_radial_moment_reproduction(
    *,
    global_shape: tuple[int, int, int],
    owned_shape: tuple[int, int, int],
    halo_width: int,
    enable_merging: bool,
) -> None:
    """Report what the radial closure reconstructs on actual torus moments."""

    geometry = build_shifted_torus_local_geometry(
        owned_shape,
        halo_width,
        global_shape=global_shape,
        shard_index=(0, 0, 0),
        x_min=shifted_mms.x_min,
        x_max=shifted_mms.x_max,
        r0=shifted_mms.r0,
        alpha_value=shifted_mms.alpha_value,
        iota=shifted_mms.iota,
        c_phi=shifted_mms.c_phi,
        sigma=shifted_mms.sigma,
    )
    cells = _build_closed_box_control_volume_cells(
        geometry,
        enable_merging=enable_merging,
    )
    closure = _build_shifted_torus_regular_boundary_closure(geometry, cells)
    face_weights, owner_weights, valid = closure.axis_payload(0)
    valid_lower = np.argwhere(np.asarray(valid[0], dtype=bool))
    if valid_lower.size == 0:
        print(
            "  shifted-torus radial moment reproduction: no valid lower "
            "patches (physical_face={:.8e}, configured_x_min={:.8e}); "
            "the regular moment closure is inactive.".format(
                float(np.asarray(geometry.grid.x.faces_owned)[0]),
                float(shifted_mms.x_min),
            )
        )
        return
    center = np.asarray(geometry.owned_shape[1:], dtype=np.float64) / 2.0
    chosen = valid_lower[
        int(
            np.argmin(
                np.sum((valid_lower.astype(np.float64) - center) ** 2, axis=1)
            )
        )
    ]
    j_index, k_index = (int(chosen[0]), int(chosen[1]))
    x_faces = np.asarray(geometry.grid.x.faces_owned, dtype=np.float64)
    theta_faces = np.asarray(geometry.grid.y.faces_owned, dtype=np.float64)
    zeta_faces = np.asarray(geometry.grid.z.faces_owned, dtype=np.float64)
    radial_span = float(shifted_mms.x_max) - float(shifted_mms.x_min)
    owner_x = float(np.asarray(cells.raw_centroid)[0, j_index, k_index, 0])
    face_weights = np.asarray(face_weights[0, j_index, k_index])
    owner_weights = np.asarray(owner_weights[0, j_index, k_index])

    def average_at_x(
        x_lower: float,
        x_upper: float,
        *,
        degree: int,
        fourier: bool,
    ) -> float:
        numerator = 0.0
        denominator = 0.0
        is_volume = x_upper > x_lower
        x_nodes = _GAUSS3_NODES if is_volume else np.asarray((0.0,))
        x_weights = _GAUSS3_WEIGHTS if is_volume else np.asarray((1.0,))
        x_measure = 0.5 * (x_upper - x_lower) if is_volume else 1.0
        for node_x, weight_x in zip(x_nodes, x_weights):
            x = 0.5 * (x_lower + x_upper) + 0.5 * (x_upper - x_lower) * node_x
            for node_y, weight_y in zip(_GAUSS3_NODES, _GAUSS3_WEIGHTS):
                theta = 0.5 * (theta_faces[j_index] + theta_faces[j_index + 1]) + 0.5 * (
                    theta_faces[j_index + 1] - theta_faces[j_index]
                ) * node_y
                for node_z, weight_z in zip(_GAUSS3_NODES, _GAUSS3_WEIGHTS):
                    zeta = 0.5 * (zeta_faces[k_index] + zeta_faces[k_index + 1]) + 0.5 * (
                        zeta_faces[k_index + 1] - zeta_faces[k_index]
                    ) * node_z
                    point = np.asarray((x, theta, zeta), dtype=np.float64)
                    jacobian = float(_shifted_torus_metric_payload_numpy(point)[0])
                    modulation = (
                        1.0
                        if not fourier
                        else 1.0 + 0.25 * np.cos(theta) + 0.15 * np.sin(zeta)
                    )
                    value = ((x - x_faces[0]) / radial_span) ** degree * modulation
                    measure = (
                        weight_x
                        * weight_y
                        * weight_z
                        * x_measure
                        * 0.5
                        * (theta_faces[j_index + 1] - theta_faces[j_index])
                        * 0.5
                        * (zeta_faces[k_index + 1] - zeta_faces[k_index])
                        * jacobian
                    )
                    numerator += measure * value
                    denominator += measure
        return numerator / max(denominator, 1.0e-30)

    def radial_derivative(x: float, *, degree: int, fourier: bool, side: str) -> float:
        step = radial_span * 1.0e-5
        if side == "lower":
            values = [
                average_at_x(x + offset * step, x + offset * step, degree=degree, fourier=fourier)
                for offset in range(5)
            ]
            return (-25.0 * values[0] + 48.0 * values[1] - 36.0 * values[2] + 16.0 * values[3] - 3.0 * values[4]) / (12.0 * step)
        return (
            average_at_x(x + step, x + step, degree=degree, fourier=fourier)
            - average_at_x(x - step, x - step, degree=degree, fourier=fourier)
        ) / (2.0 * step)

    print(
        "  shifted-torus radial moment reproduction: patch=(theta={}, zeta={}), owner_x={:.6e}".format(
            j_index,
            k_index,
            owner_x,
        )
    )
    for fourier, label in ((False, "constant"), (True, "fourier")):
        max_owner_error = 0.0
        max_face_error = 0.0
        for degree in range(4):
            samples = np.asarray(
                [
                    average_at_x(
                        x_faces[index],
                        x_faces[index + 1],
                        degree=degree,
                        fourier=fourier,
                    )
                    for index in range(3)
                ]
            )
            face_value = average_at_x(
                x_faces[0],
                x_faces[0],
                degree=degree,
                fourier=fourier,
            )
            owner_value = float(np.dot(owner_weights, np.concatenate(((face_value,), samples))))
            face_value_derivative = float(np.dot(face_weights, np.concatenate(((face_value,), samples))))
            owner_expected = radial_derivative(
                owner_x,
                degree=degree,
                fourier=fourier,
                side="owner",
            )
            face_expected = radial_derivative(
                x_faces[0],
                degree=degree,
                fourier=fourier,
                side="lower",
            )
            max_owner_error = max(max_owner_error, abs(owner_value - owner_expected))
            max_face_error = max(max_face_error, abs(face_value_derivative - face_expected))
        print(
            "    {} radial basis max derivative error: owner={:.6e} face={:.6e}".format(
                label,
                max_owner_error,
                max_face_error,
            )
        )


def _open_face_rectangles_numpy(
    *,
    axis: int,
    face_coordinate: float,
    tangential_bounds: tuple[tuple[float, float], tuple[float, float]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Decompose one open box-clipped face into at most four rectangles."""

    bounds = _box_bounds()
    full_a, full_b = tangential_bounds
    if not (
        float(bounds[axis][0]) <= face_coordinate <= float(bounds[axis][1])
    ):
        return [(full_a, full_b)]
    tangential_axes = [candidate for candidate in range(3) if candidate != axis]
    block_a = (
        max(full_a[0], float(bounds[tangential_axes[0]][0])),
        min(full_a[1], float(bounds[tangential_axes[0]][1])),
    )
    block_b = (
        max(full_b[0], float(bounds[tangential_axes[1]][0])),
        min(full_b[1], float(bounds[tangential_axes[1]][1])),
    )
    if block_a[1] <= block_a[0] or block_b[1] <= block_b[0]:
        return [(full_a, full_b)]
    rectangles: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def append(a0: float, a1: float, b0: float, b1: float) -> None:
        if a1 > a0 + 1.0e-15 and b1 > b0 + 1.0e-15:
            rectangles.append(((a0, a1), (b0, b1)))

    append(full_a[0], block_a[0], full_b[0], full_b[1])
    append(block_a[1], full_a[1], full_b[0], full_b[1])
    append(block_a[0], block_a[1], full_b[0], block_b[0])
    append(block_a[0], block_a[1], block_b[1], full_b[1])
    return rectangles


def _face_patch_quadrature_numpy(
    *,
    axis: int,
    face_coordinate: float,
    rectangle: tuple[tuple[float, float], tuple[float, float]],
    orientation: float,
) -> tuple[np.ndarray, np.ndarray]:
    tangential_axes = [candidate for candidate in range(3) if candidate != axis]
    points = np.zeros((4, 3), dtype=np.float64)
    area_weights = np.zeros((4, 3), dtype=np.float64)
    half_a = 0.5 * (rectangle[0][1] - rectangle[0][0])
    half_b = 0.5 * (rectangle[1][1] - rectangle[1][0])
    midpoint_a = 0.5 * (rectangle[0][1] + rectangle[0][0])
    midpoint_b = 0.5 * (rectangle[1][1] + rectangle[1][0])
    q = 0
    for node_a in _GAUSS2_NODES:
        for node_b in _GAUSS2_NODES:
            points[q, axis] = float(face_coordinate)
            points[q, tangential_axes[0]] = midpoint_a + half_a * node_a
            points[q, tangential_axes[1]] = midpoint_b + half_b * node_b
            area_weights[q, axis] = float(orientation) * half_a * half_b
            q += 1
    return points, area_weights


def _select_closed_box_control_volume_owners(
    geometry: LocalFciGeometry3D,
    *,
    raw_volume: np.ndarray,
    raw_centroid: np.ndarray,
    full_volume: np.ndarray,
    enable_merging: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose direct local merge targets by open-face measure and distance."""

    shape = geometry.owned_shape
    i_grid, j_grid, k_grid = np.meshgrid(
        np.arange(shape[0], dtype=np.int32),
        np.arange(shape[1], dtype=np.int32),
        np.arange(shape[2], dtype=np.int32),
        indexing="ij",
    )
    target_i = i_grid.copy()
    target_j = j_grid.copy()
    target_k = k_grid.copy()
    source = np.zeros(shape, dtype=bool)
    if not enable_merging:
        return source, target_i, target_j, target_k

    x, y, z = np.meshgrid(
        np.asarray(geometry.grid.x.centers_owned, dtype=np.float64),
        np.asarray(geometry.grid.y.centers_owned, dtype=np.float64),
        np.asarray(geometry.grid.z.centers_owned, dtype=np.float64),
        indexing="ij",
    )
    bounds = _box_bounds()
    center_in_solid = (
        (x > bounds[0][0])
        & (x < bounds[0][1])
        & (y > bounds[1][0])
        & (y < bounds[1][1])
        & (z > bounds[2][0])
        & (z < bounds[2][1])
    )
    positive = raw_volume > 1.0e-14 * max(float(np.max(full_volume)), 1.0)
    fluid_fraction = raw_volume / np.maximum(full_volume, 1.0e-30)
    candidate_source = positive & (
        center_in_solid | (fluid_fraction < 0.5)
    )
    axis_faces = (
        np.asarray(geometry.grid.x.faces_owned, dtype=np.float64),
        np.asarray(geometry.grid.y.faces_owned, dtype=np.float64),
        np.asarray(geometry.grid.z.faces_owned, dtype=np.float64),
    )
    directions = (
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 1),
        (2, -1),
        (2, 1),
    )
    for source_index_array in np.argwhere(candidate_source):
        source_index = tuple(int(value) for value in source_index_array)
        candidates: list[
            tuple[float, float, int, tuple[int, int, int]]
        ] = []
        for direction, (axis, sign) in enumerate(directions):
            neighbor = list(source_index)
            neighbor[axis] += sign
            if not all(0 <= neighbor[d] < shape[d] for d in range(3)):
                continue
            neighbor_index = tuple(neighbor)
            if candidate_source[neighbor_index] or not positive[neighbor_index]:
                continue
            face_index = source_index[axis] + (1 if sign > 0 else 0)
            face_coordinate = float(axis_faces[axis][face_index])
            tangential_axes = [candidate for candidate in range(3) if candidate != axis]
            tangential = tuple(
                (
                    float(axis_faces[t][source_index[t]]),
                    float(axis_faces[t][source_index[t] + 1]),
                )
                for t in tangential_axes
            )
            rectangles = _open_face_rectangles_numpy(
                axis=axis,
                face_coordinate=face_coordinate,
                tangential_bounds=tangential,
            )
            shared_measure = 0.0
            for rectangle in rectangles:
                points, area_weight = _face_patch_quadrature_numpy(
                    axis=axis,
                    face_coordinate=face_coordinate,
                    rectangle=rectangle,
                    orientation=1.0,
                )
                J, *_rest = _shifted_torus_metric_payload_numpy(points)
                shared_measure += float(
                    np.sum(J * np.linalg.norm(area_weight, axis=-1))
                )
            distance = float(
                np.linalg.norm(
                    raw_centroid[neighbor_index] - raw_centroid[source_index]
                )
            )
            if shared_measure > 1.0e-20:
                candidates.append(
                    (-shared_measure, distance, direction, neighbor_index)
                )
        if candidates:
            _measure, _distance, _direction, target = min(candidates)
            source[source_index] = True
            target_i[source_index] = target[0]
            target_j[source_index] = target[1]
            target_k[source_index] = target[2]
    return source, target_i, target_j, target_k


def _closed_box_irregular_storage_mask(
    geometry: LocalFciGeometry3D,
    cells: LocalControlVolumeCellGeometry3D,
) -> np.ndarray:
    """Storage cells whose dense face stencils do not represent one CV."""

    _raw_volume, _raw_centroid, _raw_m2, _raw_m3, full_volume = (
        _closed_box_fluid_moments_3point(geometry)
    )
    raw_volume = np.asarray(cells.raw_volume, dtype=np.float64)
    cut_storage = (raw_volume > 1.0e-20) & (
        raw_volume
        < np.asarray(full_volume, dtype=np.float64) * (1.0 - 1.0e-12)
    )
    return (
        np.asarray(cells.is_merged_source, dtype=bool)
        | np.asarray(cells.is_aggregate_target, dtype=bool)
        | cut_storage
    )


def _build_closed_box_control_volume_faces(
    geometry: LocalFciGeometry3D,
    cells: LocalControlVolumeCellGeometry3D,
    *,
    remote_boundary_payloads: dict[
        tuple[int, int],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]
    | None = None,
    reconstruction_owner_mask: np.ndarray | None = None,
    compact_owner_mask: np.ndarray | None = None,
) -> tuple[LocalRegularFaceGeometry3D, LocalControlVolumeFaceRows3D]:
    """Build dense ordinary faces and compact unique irregular interfaces."""

    if remote_boundary_payloads is None:
        remote_boundary_payloads = {}
    shape = geometry.owned_shape
    owner_i = np.asarray(cells.owner_i, dtype=np.int32)
    owner_j = np.asarray(cells.owner_j, dtype=np.int32)
    owner_k = np.asarray(cells.owner_k, dtype=np.int32)
    active_owner = np.asarray(cells.is_active_owner, dtype=bool)
    source = np.asarray(cells.is_merged_source, dtype=bool)
    aggregate_target = np.asarray(cells.is_aggregate_target, dtype=bool)
    aggregate_centroid = np.asarray(cells.centroid, dtype=np.float64)
    axis_faces = (
        np.asarray(geometry.grid.x.faces_owned, dtype=np.float64),
        np.asarray(geometry.grid.y.faces_owned, dtype=np.float64),
        np.asarray(geometry.grid.z.faces_owned, dtype=np.float64),
    )
    bounds = _box_bounds()
    axis_overlap = tuple(
        np.maximum(
            np.minimum(faces[1:], float(bounds[axis][1]))
            - np.maximum(faces[:-1], float(bounds[axis][0])),
            0.0,
        )
        / np.maximum(faces[1:] - faces[:-1], 1.0e-30)
        for axis, faces in enumerate(axis_faces)
    )
    solid_face_fraction = (
        np.where(
            (axis_faces[0][:, None, None] >= bounds[0][0])
            & (axis_faces[0][:, None, None] <= bounds[0][1]),
            axis_overlap[1][None, :, None] * axis_overlap[2][None, None, :],
            0.0,
        ),
        np.where(
            (axis_faces[1][None, :, None] >= bounds[1][0])
            & (axis_faces[1][None, :, None] <= bounds[1][1]),
            axis_overlap[0][:, None, None] * axis_overlap[2][None, None, :],
            0.0,
        ),
        np.where(
            (axis_faces[2][None, None, :] >= bounds[2][0])
            & (axis_faces[2][None, None, :] <= bounds[2][1]),
            axis_overlap[0][:, None, None] * axis_overlap[1][None, :, None],
            0.0,
        ),
    )
    open_fraction = tuple(
        np.clip(1.0 - fraction, 0.0, 1.0)
        for fraction in solid_face_fraction
    )
    open_masks = [fraction > 1.0e-12 for fraction in open_fraction]

    face_candidates: set[tuple[int, int, int, int]] = set()
    for axis, fraction in enumerate(open_fraction):
        partial = (fraction > 1.0e-12) & (fraction < 1.0 - 1.0e-12)
        for face_index in np.argwhere(partial):
            face_candidates.add(
                (axis, *(int(value) for value in face_index))
            )
    touched_storage = _closed_box_irregular_storage_mask(geometry, cells)
    for storage_array in np.argwhere(touched_storage):
        storage = tuple(int(value) for value in storage_array)
        for axis in range(3):
            for side in (0, 1):
                face = list(storage)
                face[axis] += side
                face_candidates.add((axis, face[0], face[1], face[2]))
    if reconstruction_owner_mask is not None:
        selected_owner = np.asarray(reconstruction_owner_mask, dtype=bool)
        if selected_owner.shape != shape:
            raise ValueError(
                "reconstruction_owner_mask must match the owned cell shape"
            )
        compact_owner = (
            selected_owner
            if compact_owner_mask is None
            else np.asarray(compact_owner_mask, dtype=bool)
        )
        if compact_owner.shape != shape:
            raise ValueError("compact_owner_mask must match the owned cell shape")
        selected_storage = compact_owner[owner_i, owner_j, owner_k]
        # A compact full face must own the entire dense stencil support, not
        # merely touch a reconstructed owner.  Otherwise the dense projected
        # stencil can still read one poisoned/source sample tangentially while
        # the compact row supplies the other part of the interface flux.
        identity_storage = (
            (owner_i == np.indices(shape)[0])
            & (owner_j == np.indices(shape)[1])
            & (owner_k == np.indices(shape)[2])
        )
        nonordinary_storage = (
            (~active_owner[owner_i, owner_j, owner_k])
            | source
            | (~identity_storage)
            | selected_storage
        )

        def _wrap_support_index(index: list[int]) -> tuple[int, int, int] | None:
            for candidate_axis in range(3):
                if candidate_axis in (1, 2):
                    index[candidate_axis] %= shape[candidate_axis]
                elif not 0 <= index[candidate_axis] < shape[candidate_axis]:
                    return None
            return tuple(index)

        for axis in range(3):
            face_shape = open_fraction[axis].shape
            for face_array in np.ndindex(face_shape):
                face_axis_index = face_array[axis]
                if not 0 < face_axis_index < shape[axis]:
                    continue
                minus = list(face_array)
                minus[axis] -= 1
                plus = list(face_array)
                support: set[tuple[int, int, int]] = set()
                for center in (minus, plus):
                    for derivative_axis in range(3):
                        for direction in (-1, 1):
                            sample = list(center)
                            sample[derivative_axis] += direction
                            sample_index = _wrap_support_index(sample)
                            if sample_index is not None:
                                support.add(sample_index)
                if any(nonordinary_storage[sample] for sample in support):
                    face_candidates.add((axis, *face_array))
    for (axis, side), (
        remote_touched,
        _remote_fluid,
        _remote_centroid,
        _remote_second_moment,
        _remote_third_moment,
    ) in (
        remote_boundary_payloads.items()
    ):
        local_boundary = np.take(
            touched_storage,
            0 if side == 0 else shape[axis] - 1,
            axis=axis,
        )
        for tangential_index in np.argwhere(remote_touched | local_boundary):
            face = [0, 0, 0]
            face[axis] = 0 if side == 0 else shape[axis]
            tangential_axes = [candidate for candidate in range(3) if candidate != axis]
            face[tangential_axes[0]] = int(tangential_index[0])
            face[tangential_axes[1]] = int(tangential_index[1])
            face_candidates.add((axis, face[0], face[1], face[2]))

    rows: list[dict[str, object]] = []


    def add_row(
        *,
        kind: int,
        minus_owner: tuple[int, int, int],
        plus_owner: tuple[int, int, int] | None,
        axis: int,
        coordinate: float,
        rectangles: list[tuple[tuple[float, float], tuple[float, float]]],
        orientation: float,
        remote_halo: tuple[int, int, int] | None = None,
        remote_centroid: np.ndarray | None = None,
        remote_second_moment: np.ndarray | None = None,
        remote_third_moment: np.ndarray | None = None,
    ) -> None:
        if not rectangles:
            return
        patches = []
        for rectangle in rectangles[:4]:
            points, area_weight = _face_patch_quadrature_numpy(
                axis=axis,
                face_coordinate=coordinate,
                rectangle=rectangle,
                orientation=orientation,
            )
            metric = _shifted_torus_metric_payload_numpy(points)
            patches.append((points, area_weight, metric))
        rows.append(
            {
                "kind": int(kind),
                "minus": minus_owner,
                "plus": plus_owner,
                "remote_halo": remote_halo,
                "remote_centroid": remote_centroid,
                "remote_second_moment": remote_second_moment,
                "remote_third_moment": remote_third_moment,
                "patches": patches,
            }
        )

    for key in sorted(face_candidates):
        axis, fi, fj, fk = key
        face_index = (fi, fj, fk)
        if not all(
            0 <= face_index[d] < open_fraction[axis].shape[d]
            for d in range(3)
        ):
            continue
        axis_face_index = face_index[axis]
        if axis_face_index in (0, shape[axis]):
            side = 0 if axis_face_index == 0 else 1
            remote_payload = remote_boundary_payloads.get((axis, side))
            if remote_payload is None:
                if axis != 0:
                    continue
                # A full coordinate-aligned physical face stays on the dense
                # regular path. Compact physical rows are only needed when an
                # embedded boundary leaves a partial domain-boundary face.
                if (
                    float(open_fraction[axis][face_index])
                    >= 1.0 - 1.0e-12
                ):
                    continue
                local_storage = list(face_index)
                local_storage[axis] = (
                    0 if side == 0 else shape[axis] - 1
                )
                local_storage_index = tuple(local_storage)
                local_owner = (
                    int(owner_i[local_storage_index]),
                    int(owner_j[local_storage_index]),
                    int(owner_k[local_storage_index]),
                )
                if not active_owner[local_owner]:
                    open_masks[axis][face_index] = False
                    continue
                face_coordinate = float(
                    axis_faces[axis][axis_face_index]
                )
                tangential_axes = [
                    candidate for candidate in range(3)
                    if candidate != axis
                ]
                tangential = tuple(
                    (
                        float(axis_faces[t][face_index[t]]),
                        float(axis_faces[t][face_index[t] + 1]),
                    )
                    for t in tangential_axes
                )
                rectangles = _open_face_rectangles_numpy(
                    axis=axis,
                    face_coordinate=face_coordinate,
                    tangential_bounds=tangential,
                )
                add_row(
                    kind=CV_FACE_PHYSICAL_BOUNDARY,
                    minus_owner=local_owner,
                    plus_owner=None,
                    axis=axis,
                    coordinate=face_coordinate,
                    rectangles=rectangles,
                    orientation=-1.0 if side == 0 else 1.0,
                )
                open_masks[axis][face_index] = False
                continue
            (
                remote_touched,
                remote_fluid,
                remote_centroid,
                remote_second_moment,
                remote_third_moment,
            ) = remote_payload
            tangential_axes = [
                candidate for candidate in range(3) if candidate != axis
            ]
            tangential_index = tuple(face_index[d] for d in tangential_axes)
            local_storage = list(face_index)
            local_storage[axis] = 0 if side == 0 else shape[axis] - 1
            local_storage_index = tuple(local_storage)
            if not bool(remote_fluid[tangential_index]):
                open_masks[axis][face_index] = False
                continue
            local_owner = (
                int(owner_i[local_storage_index]),
                int(owner_j[local_storage_index]),
                int(owner_k[local_storage_index]),
            )
            if not active_owner[local_owner]:
                open_masks[axis][face_index] = False
                continue
            face_coordinate = float(axis_faces[axis][axis_face_index])
            tangential = tuple(
                (
                    float(axis_faces[t][face_index[t]]),
                    float(axis_faces[t][face_index[t] + 1]),
                )
                for t in tangential_axes
            )
            rectangles = _open_face_rectangles_numpy(
                axis=axis,
                face_coordinate=face_coordinate,
                tangential_bounds=tangential,
            )
            if not rectangles:
                open_masks[axis][face_index] = False
                continue
            halo_width = int(geometry.layout.halo_width)
            remote_halo = [
                halo_width + face_index[0],
                halo_width + face_index[1],
                halo_width + face_index[2],
            ]
            remote_halo[axis] = (
                halo_width - 1
                if side == 0
                else halo_width + shape[axis]
            )
            is_partial = (
                open_fraction[axis][face_index] < 1.0 - 1.0e-12
            )
            add_row(
                kind=CV_FACE_PARTIAL if is_partial else CV_FACE_INTERIOR,
                minus_owner=local_owner,
                plus_owner=None,
                remote_halo=tuple(remote_halo),
                remote_centroid=remote_centroid[tangential_index],
                remote_second_moment=remote_second_moment[tangential_index],
                remote_third_moment=remote_third_moment[tangential_index],
                axis=axis,
                coordinate=face_coordinate,
                rectangles=rectangles,
                orientation=-1.0 if side == 0 else 1.0,
            )
            open_masks[axis][face_index] = False
            continue
        if axis_face_index < 0 or axis_face_index > shape[axis]:
            continue
        minus_storage = [fi, fj, fk]
        plus_storage = [fi, fj, fk]
        minus_storage[axis] -= 1
        minus_storage = tuple(minus_storage)
        plus_storage = tuple(plus_storage)
        minus_owner = (
            int(owner_i[minus_storage]),
            int(owner_j[minus_storage]),
            int(owner_k[minus_storage]),
        )
        plus_owner = (
            int(owner_i[plus_storage]),
            int(owner_j[plus_storage]),
            int(owner_k[plus_storage]),
        )
        if not active_owner[minus_owner] or not active_owner[plus_owner]:
            open_masks[axis][face_index] = False
            continue
        if minus_owner == plus_owner:
            open_masks[axis][face_index] = False
            continue
        face_coordinate = float(axis_faces[axis][axis_face_index])
        tangential_axes = [candidate for candidate in range(3) if candidate != axis]
        tangential = tuple(
            (
                float(axis_faces[t][face_index[t]]),
                float(axis_faces[t][face_index[t] + 1]),
            )
            for t in tangential_axes
        )
        rectangles = _open_face_rectangles_numpy(
            axis=axis,
            face_coordinate=face_coordinate,
            tangential_bounds=tangential,
        )
        if not rectangles:
            open_masks[axis][face_index] = False
            continue
        is_partial = open_fraction[axis][face_index] < 1.0 - 1.0e-12
        add_row(
            kind=CV_FACE_PARTIAL if is_partial else CV_FACE_INTERIOR,
            minus_owner=minus_owner,
            plus_owner=plus_owner,
            axis=axis,
            coordinate=face_coordinate,
            rectangles=rectangles,
            orientation=1.0,
        )
        open_masks[axis][face_index] = False

    for axis, axis_bounds in enumerate(bounds):
        tangential_axes = [candidate for candidate in range(3) if candidate != axis]
        for surface, coordinate in enumerate(axis_bounds):
            if surface == 0:
                owner_axis_index = (
                    int(np.searchsorted(axis_faces[axis], coordinate, side="left"))
                    - 1
                )
                orientation = 1.0
            else:
                owner_axis_index = (
                    int(np.searchsorted(axis_faces[axis], coordinate, side="right"))
                    - 1
                )
                orientation = -1.0
            if not 0 <= owner_axis_index < shape[axis]:
                continue
            for first in range(shape[tangential_axes[0]]):
                first_bounds = (
                    max(
                        float(axis_faces[tangential_axes[0]][first]),
                        float(bounds[tangential_axes[0]][0]),
                    ),
                    min(
                        float(axis_faces[tangential_axes[0]][first + 1]),
                        float(bounds[tangential_axes[0]][1]),
                    ),
                )
                if first_bounds[1] <= first_bounds[0] + 1.0e-15:
                    continue
                for second in range(shape[tangential_axes[1]]):
                    second_bounds = (
                        max(
                            float(axis_faces[tangential_axes[1]][second]),
                            float(bounds[tangential_axes[1]][0]),
                        ),
                        min(
                            float(axis_faces[tangential_axes[1]][second + 1]),
                            float(bounds[tangential_axes[1]][1]),
                        ),
                    )
                    if second_bounds[1] <= second_bounds[0] + 1.0e-15:
                        continue
                    storage = [0, 0, 0]
                    storage[axis] = owner_axis_index
                    storage[tangential_axes[0]] = first
                    storage[tangential_axes[1]] = second
                    storage_index = tuple(storage)
                    owner = (
                        int(owner_i[storage_index]),
                        int(owner_j[storage_index]),
                        int(owner_k[storage_index]),
                    )
                    if not active_owner[owner]:
                        continue
                    add_row(
                        kind=CV_FACE_CUT_WALL,
                        minus_owner=owner,
                        plus_owner=None,
                        axis=axis,
                        coordinate=float(coordinate),
                        rectangles=[(first_bounds, second_bounds)],
                        orientation=orientation,
                    )

    max_rows = len(rows)
    max_patches = 4
    row_shape = (max_rows,)
    patch_shape = (max_rows, max_patches)
    quadrature_shape = (max_rows, max_patches, 4)
    kind = np.zeros(row_shape, dtype=np.int32)
    minus_i = np.zeros(row_shape, dtype=np.int32)
    minus_j = np.zeros(row_shape, dtype=np.int32)
    minus_k = np.zeros(row_shape, dtype=np.int32)
    plus_i = np.zeros(row_shape, dtype=np.int32)
    plus_j = np.zeros(row_shape, dtype=np.int32)
    plus_k = np.zeros(row_shape, dtype=np.int32)
    has_plus = np.zeros(row_shape, dtype=bool)
    has_remote = np.zeros(row_shape, dtype=bool)
    remote_halo_i = np.zeros(row_shape, dtype=np.int32)
    remote_halo_j = np.zeros(row_shape, dtype=np.int32)
    remote_halo_k = np.zeros(row_shape, dtype=np.int32)
    remote_centroid = np.zeros(row_shape + (3,), dtype=np.float64)
    remote_second_moment = np.zeros(row_shape + (3, 3), dtype=np.float64)
    remote_third_moment = np.zeros(row_shape + (3, 3, 3), dtype=np.float64)
    points = np.zeros(quadrature_shape + (3,), dtype=np.float64)
    area = np.zeros(quadrature_shape + (3,), dtype=np.float64)
    J = np.zeros(quadrature_shape, dtype=np.float64)
    g_contra = np.zeros(quadrature_shape + (3, 3), dtype=np.float64)
    g_cov = np.zeros_like(g_contra)
    B_contra = np.zeros(quadrature_shape + (3,), dtype=np.float64)
    Bmag = np.ones(quadrature_shape, dtype=np.float64)
    projector = np.zeros_like(g_contra)
    patch_active = np.zeros(patch_shape, dtype=bool)
    for row_index, row in enumerate(rows):
        kind[row_index] = int(row["kind"])
        minus = row["minus"]
        minus_i[row_index], minus_j[row_index], minus_k[row_index] = minus
        plus = row["plus"]
        if plus is not None:
            has_plus[row_index] = True
            plus_i[row_index], plus_j[row_index], plus_k[row_index] = plus
        remote_halo = row["remote_halo"]
        if remote_halo is not None:
            has_remote[row_index] = True
            (
                remote_halo_i[row_index],
                remote_halo_j[row_index],
                remote_halo_k[row_index],
            ) = remote_halo
            remote_centroid[row_index] = row["remote_centroid"]
            remote_second_moment[row_index] = row["remote_second_moment"]
            remote_third_moment[row_index] = row.get("remote_third_moment", 0.0)
        for patch_index, (q_points, q_area, metric) in enumerate(row["patches"]):
            patch_active[row_index, patch_index] = True
            points[row_index, patch_index] = q_points
            area[row_index, patch_index] = q_area
            (
                J[row_index, patch_index],
                g_contra[row_index, patch_index],
                g_cov[row_index, patch_index],
                B_contra[row_index, patch_index],
                Bmag[row_index, patch_index],
                projector[row_index, patch_index],
            ) = metric

    irregular = LocalControlVolumeFaceRows3D(
        layout=geometry.layout,
        kind=jnp.asarray(kind),
        minus_owner_i=jnp.asarray(minus_i),
        minus_owner_j=jnp.asarray(minus_j),
        minus_owner_k=jnp.asarray(minus_k),
        plus_owner_i=jnp.asarray(plus_i),
        plus_owner_j=jnp.asarray(plus_j),
        plus_owner_k=jnp.asarray(plus_k),
        has_plus_owner=jnp.asarray(has_plus),
        quadrature_points=jnp.asarray(points),
        area_covector_weight=jnp.asarray(area),
        J=jnp.asarray(J),
        g_contra=jnp.asarray(g_contra),
        g_cov=jnp.asarray(g_cov),
        B_contra=jnp.asarray(B_contra),
        Bmag=jnp.asarray(Bmag),
        projector=jnp.asarray(projector),
        patch_active=jnp.asarray(patch_active),
        active=jnp.ones((max_rows,), dtype=bool),
        max_rows=max_rows,
        max_patches=max_patches,
        has_remote_owner=jnp.asarray(has_remote),
        remote_halo_i=jnp.asarray(remote_halo_i),
        remote_halo_j=jnp.asarray(remote_halo_j),
        remote_halo_k=jnp.asarray(remote_halo_k),
        remote_centroid=jnp.asarray(remote_centroid),
        remote_second_moment=jnp.asarray(remote_second_moment),
        remote_third_moment=jnp.asarray(remote_third_moment),
    )
    base_regular = geometry.regular_face_geometry
    regular = LocalRegularFaceGeometry3D(
        layout=geometry.layout,
        x_area=base_regular.x_area,
        y_area=base_regular.y_area,
        z_area=base_regular.z_area,
        x_area_fraction=jnp.asarray(open_fraction[0]),
        y_area_fraction=jnp.asarray(open_fraction[1]),
        z_area_fraction=jnp.asarray(open_fraction[2]),
        x_open_mask=jnp.asarray(open_masks[0]),
        y_open_mask=jnp.asarray(open_masks[1]),
        z_open_mask=jnp.asarray(open_masks[2]),
        x_centroid_offset=base_regular.x_centroid_offset,
        y_centroid_offset=base_regular.y_centroid_offset,
        z_centroid_offset=base_regular.z_centroid_offset,
    )
    for side, face_axis_index in ((0, 0), (1, shape[0])):
        if (0, side) in remote_boundary_payloads:
            continue
        boundary_fraction = np.asarray(
            open_fraction[0][face_axis_index],
            dtype=np.float64,
        )
        boundary_open = np.asarray(
            open_masks[0][face_axis_index],
            dtype=bool,
        )
        full_face = boundary_fraction >= 1.0 - 1.0e-12
        if np.any(full_face & (~boundary_open)):
            raise ValueError(
                "full coordinate-aligned radial boundary faces must remain "
                "on the dense regular path"
            )
    return regular, irregular


def _build_closed_box_control_volume_cells(
    geometry: LocalFciGeometry3D,
    *,
    enable_merging: bool,
) -> LocalControlVolumeCellGeometry3D:
    raw_volume, raw_centroid, raw_second_moment, raw_third_moment, full_volume = (
        _closed_box_fluid_moments_3point(geometry)
    )
    source, target_i, target_j, target_k = (
        _select_closed_box_control_volume_owners(
            geometry,
            raw_volume=raw_volume,
            raw_centroid=raw_centroid,
            full_volume=full_volume,
            enable_merging=enable_merging,
        )
    )
    cells = build_local_control_volume_cell_geometry(
        geometry.layout,
        raw_volume=jnp.asarray(raw_volume),
        raw_centroid=jnp.asarray(raw_centroid),
        raw_second_moment=jnp.asarray(raw_second_moment),
        raw_third_moment=jnp.asarray(raw_third_moment),
        source_active=jnp.asarray(source),
        target_i=jnp.asarray(target_i),
        target_j=jnp.asarray(target_j),
        target_k=jnp.asarray(target_k),
        retained_active=jnp.asarray(raw_volume > 1.0e-20),
    )
    source_volume = float(np.sum(np.asarray(cells.raw_volume, dtype=np.float64)))
    owner_volume = float(
        np.sum(np.asarray(cells.aggregate_volume, dtype=np.float64))
    )
    if not np.isclose(
        source_volume,
        owner_volume,
        rtol=5.0e-13,
        atol=5.0e-14,
    ):
        raise ValueError(
            "control-volume merging did not conserve local fluid volume: "
            f"raw={source_volume:.16e}, aggregate={owner_volume:.16e}"
        )
    owner_i_np = np.asarray(cells.owner_i, dtype=np.int32)
    owner_j_np = np.asarray(cells.owner_j, dtype=np.int32)
    owner_k_np = np.asarray(cells.owner_k, dtype=np.int32)
    mapped_owner = (
        owner_i_np[owner_i_np, owner_j_np, owner_k_np],
        owner_j_np[owner_i_np, owner_j_np, owner_k_np],
        owner_k_np[owner_i_np, owner_j_np, owner_k_np],
    )
    if not (
        np.array_equal(mapped_owner[0], owner_i_np)
        and np.array_equal(mapped_owner[1], owner_j_np)
        and np.array_equal(mapped_owner[2], owner_k_np)
    ):
        raise ValueError("control-volume owner mapping must be idempotent")
    return cells


def _intrinsic_reconstruction_owner_mask(
    cells: LocalControlVolumeCellGeometry3D,
    irregular_faces: LocalControlVolumeFaceRows3D,
) -> np.ndarray:
    """Select owners needing polynomial data from embedded geometry alone."""

    active_owner = np.asarray(cells.is_active_owner, dtype=bool)
    target = np.asarray(cells.is_aggregate_target, dtype=bool).copy()
    face_active = np.asarray(irregular_faces.active, dtype=bool)
    face_kind = np.asarray(irregular_faces.kind, dtype=np.int32)
    intrinsic_face = face_active & (
        (face_kind == CV_FACE_PARTIAL)
        | (face_kind == CV_FACE_CUT_WALL)
        | (face_kind == CV_FACE_PHYSICAL_BOUNDARY)
    )
    for row in np.flatnonzero(intrinsic_face):
        minus = (
            int(np.asarray(irregular_faces.minus_owner_i)[row]),
            int(np.asarray(irregular_faces.minus_owner_j)[row]),
            int(np.asarray(irregular_faces.minus_owner_k)[row]),
        )
        target[minus] = True
        if bool(np.asarray(irregular_faces.has_plus_owner)[row]):
            plus = (
                int(np.asarray(irregular_faces.plus_owner_i)[row]),
                int(np.asarray(irregular_faces.plus_owner_j)[row]),
                int(np.asarray(irregular_faces.plus_owner_k)[row]),
            )
            target[plus] = True
    return target & active_owner


def _dilate_reconstruction_owner_mask(
    cells: LocalControlVolumeCellGeometry3D,
    intrinsic_mask: np.ndarray,
    *,
    periodic_axes: tuple[bool, bool, bool],
) -> np.ndarray:
    """Add one face-neighbour guard ring around intrinsic CV owners.

    The transition functional reads cell-centred derivatives from both sides
    of a full face.  Reconstructing only the owner directly touching a wall
    leaves that support incomplete one logical cell away.
    """

    shape = cells.shape
    active = np.asarray(cells.is_active_owner, dtype=bool)
    owner_i = np.asarray(cells.owner_i, dtype=np.int32)
    owner_j = np.asarray(cells.owner_j, dtype=np.int32)
    owner_k = np.asarray(cells.owner_k, dtype=np.int32)
    result = np.asarray(intrinsic_mask, dtype=bool).copy()
    for storage_array in np.argwhere(intrinsic_mask):
        storage = [int(value) for value in storage_array]
        for axis in range(3):
            for direction in (-1, 1):
                neighbor = list(storage)
                neighbor[axis] += direction
                if periodic_axes[axis]:
                    neighbor[axis] %= shape[axis]
                elif not 0 <= neighbor[axis] < shape[axis]:
                    continue
                neighbor_index = tuple(neighbor)
                owner = (
                    int(owner_i[neighbor_index]),
                    int(owner_j[neighbor_index]),
                    int(owner_k[neighbor_index]),
                )
                if active[owner]:
                    result[owner] = True
    return result & active


def _transition_face_owner_mask(
    cells: LocalControlVolumeCellGeometry3D,
    irregular_faces: LocalControlVolumeFaceRows3D,
) -> np.ndarray:
    """Return active owners directly attached to local full transition rows."""

    result = np.zeros(cells.shape, dtype=bool)
    active = np.asarray(irregular_faces.active, dtype=bool)
    interior = np.asarray(irregular_faces.kind, dtype=np.int32) == CV_FACE_INTERIOR
    has_plus = np.asarray(irregular_faces.has_plus_owner, dtype=bool)
    for row in np.flatnonzero(active & interior & has_plus):
        minus = (
            int(np.asarray(irregular_faces.minus_owner_i)[row]),
            int(np.asarray(irregular_faces.minus_owner_j)[row]),
            int(np.asarray(irregular_faces.minus_owner_k)[row]),
        )
        plus = (
            int(np.asarray(irregular_faces.plus_owner_i)[row]),
            int(np.asarray(irregular_faces.plus_owner_j)[row]),
            int(np.asarray(irregular_faces.plus_owner_k)[row]),
        )
        result[minus] = True
        result[plus] = True
    return result & np.asarray(cells.is_active_owner, dtype=bool)


def _build_regular_transition_face_rows(
    geometry: LocalFciGeometry3D,
    cells: LocalControlVolumeCellGeometry3D,
    irregular_faces: LocalControlVolumeFaceRows3D,
    reconstruction_owner_mask: np.ndarray,
    *,
    local_periodic_axes: tuple[bool, bool, bool] = (False, True, True),
) -> LocalRegularTransitionFaceRows3D:
    """Compile dense-compatible scalar and gradient stencils for full rows."""

    max_rows = int(irregular_faces.max_rows)
    max_samples = 16
    shape = geometry.owned_shape
    face_active = np.asarray(irregular_faces.active, dtype=bool)
    interior = np.asarray(irregular_faces.kind, dtype=np.int32) == CV_FACE_INTERIOR
    has_plus = np.asarray(irregular_faces.has_plus_owner, dtype=bool)
    remote = np.asarray(irregular_faces.has_remote_owner, dtype=bool)
    active = face_active & interior & has_plus & (~remote)
    owner_i = np.asarray(cells.owner_i, dtype=np.int32)
    owner_j = np.asarray(cells.owner_j, dtype=np.int32)
    owner_k = np.asarray(cells.owner_k, dtype=np.int32)
    is_active_owner = np.asarray(cells.is_active_owner, dtype=bool)
    source = np.asarray(cells.is_merged_source, dtype=bool)
    aggregate_target = np.asarray(cells.is_aggregate_target, dtype=bool)
    centroid = np.asarray(cells.centroid, dtype=np.float64)
    second_moment = np.asarray(cells.second_moment, dtype=np.float64)
    axis_faces = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            geometry.grid.x.faces_owned,
            geometry.grid.y.faces_owned,
            geometry.grid.z.faces_owned,
        )
    )
    lower = (
        axis_faces[0][:-1, None, None],
        axis_faces[1][None, :-1, None],
        axis_faces[2][None, None, :-1],
    )
    upper = (
        axis_faces[0][1:, None, None],
        axis_faces[1][None, 1:, None],
        axis_faces[2][None, None, 1:],
    )
    full_volume, full_first, full_second_origin, full_third_origin = (
        _integrate_shifted_torus_rectangular_moments(lower, upper)
    )
    full_centroid = full_first / np.maximum(full_volume[..., None], 1.0e-30)
    full_m2 = full_second_origin / np.maximum(
        full_volume[..., None, None], 1.0e-30
    ) - full_centroid[..., :, None] * full_centroid[..., None, :]
    full_m2 = 0.5 * (full_m2 + np.swapaxes(full_m2, -1, -2))
    full_second_origin = full_second_origin / np.maximum(
        full_volume[..., None, None], 1.0e-30
    )
    full_m3 = full_third_origin / np.maximum(
        full_volume[..., None, None, None], 1.0e-30
    ) - (
        full_centroid[..., :, None, None] * full_second_origin[..., None, :, :]
        + full_centroid[..., None, :, None] * full_second_origin[..., :, None, :]
        + full_centroid[..., None, None, :] * full_second_origin[..., :, :, None]
    ) + 2.0 * (
        full_centroid[..., :, None, None]
        * full_centroid[..., None, :, None]
        * full_centroid[..., None, None, :]
    )
    spacing = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            geometry.spacing.dx_owned,
            geometry.spacing.dy_owned,
            geometry.spacing.dz_owned,
        )
    )
    row_int = lambda: np.zeros((max_rows,), dtype=np.int32)
    row_samples_int = lambda: np.zeros((max_rows, max_samples), dtype=np.int32)
    face_axis = row_int(); face_i = row_int(); face_j = row_int(); face_k = row_int()
    sample_storage_i = row_samples_int(); sample_storage_j = row_samples_int(); sample_storage_k = row_samples_int()
    sample_owner_i = row_samples_int(); sample_owner_j = row_samples_int(); sample_owner_k = row_samples_int()
    sample_center_owner_i = row_samples_int(); sample_center_owner_j = row_samples_int(); sample_center_owner_k = row_samples_int()
    sample_active = np.zeros((max_rows, max_samples), dtype=bool)
    sample_direct = np.zeros((max_rows, max_samples), dtype=bool)
    sample_displacement = np.zeros((max_rows, max_samples, 3), dtype=np.float64)
    sample_moment_delta = np.zeros((max_rows, max_samples, 3, 3), dtype=np.float64)
    sample_third_moment_delta = np.zeros((max_rows, max_samples, 3, 3, 3), dtype=np.float64)
    scalar_coefficients = np.zeros((max_rows, max_samples), dtype=np.float64)
    gradient_coefficients = np.zeros((max_rows, 3, max_samples), dtype=np.float64)
    sample_count = row_int()
    valid = active.copy()

    def wrap(index: list[int]) -> tuple[int, int, int] | None:
        """Map a dense support sample into this shard's owned storage.

        A periodic global coordinate is only locally periodic when it has one
        shard.  Wrapping a zeta support sample from the last local plane to
        local zero on a decomposed mesh silently substitutes an unrelated
        control volume.  Remote transition samples need a dedicated packed
        polynomial exchange; until that path exists, leave such faces on the
        regular irregular-row flux path below.
        """
        for axis in range(3):
            if 0 <= index[axis] < shape[axis]:
                continue
            if local_periodic_axes[axis]:
                index[axis] %= shape[axis]
                continue
            return None
        return tuple(index)

    points = np.asarray(irregular_faces.quadrature_points, dtype=np.float64)
    area = np.asarray(irregular_faces.area_covector_weight, dtype=np.float64)
    patch_active = np.asarray(irregular_faces.patch_active, dtype=bool)
    for row in np.flatnonzero(active):
        first_patch = np.argwhere(patch_active[row])
        if not first_patch.size:
            valid[row] = False
            continue
        patch = int(first_patch[0, 0])
        q_point = points[row, patch, 0]
        q_area = area[row, patch, 0]
        axis = int(np.argmax(np.abs(q_area)))
        index = []
        for coordinate_axis in range(3):
            if coordinate_axis == axis:
                index.append(int(np.argmin(np.abs(axis_faces[axis] - q_point[axis]))))
            else:
                index.append(
                    int(np.clip(
                        np.searchsorted(axis_faces[coordinate_axis], q_point[coordinate_axis], side="right") - 1,
                        0,
                        shape[coordinate_axis] - 1,
                    ))
                )
        if not 0 < index[axis] < shape[axis]:
            valid[row] = False
            continue
        minus = list(index); minus[axis] -= 1
        plus = list(index)
        # Do not deduplicate by storage location.  One logical unavailable
        # sample can be used by two distinct centered derivatives; each use
        # must retain the control-volume owner of the derivative center.
        occurrences: list[tuple[tuple[int, int, int], tuple[int, int, int], float, int | None, float]] = []

        def add(
            sample: tuple[int, int, int] | None,
            center: tuple[int, int, int],
            *,
            scalar: float = 0.0,
            gradient_axis: int | None = None,
            gradient: float = 0.0,
        ) -> None:
            if sample is None:
                valid[row] = False
                return
            occurrences.append((sample, center, scalar, gradient_axis, gradient))

        add(tuple(minus), tuple(minus), scalar=0.5)
        add(tuple(plus), tuple(plus), scalar=0.5)
        for center in (minus, plus):
            for derivative_axis in range(3):
                h = float(spacing[derivative_axis][tuple(center)])
                if not np.isfinite(h) or abs(h) <= 1.0e-14:
                    valid[row] = False
                    continue
                lower_sample = list(center); lower_sample[derivative_axis] -= 1
                upper_sample = list(center); upper_sample[derivative_axis] += 1
                add(
                    wrap(lower_sample),
                    tuple(center),
                    gradient_axis=derivative_axis,
                    gradient=-0.25 / h,
                )
                add(
                    wrap(upper_sample),
                    tuple(center),
                    gradient_axis=derivative_axis,
                    gradient=0.25 / h,
                )
        if len(occurrences) > max_samples:
            valid[row] = False
            continue
        if not valid[row]:
            # The underlying irregular row remains active and computes its
            # polynomial flux normally.  Only the local dense-compatible
            # transition functional is deferred until remote support is
            # represented explicitly.
            active[row] = False
            continue
        face_axis[row] = axis
        face_i[row], face_j[row], face_k[row] = index
        for column, (storage, center, scalar, derivative_axis, gradient_coefficient) in enumerate(occurrences):
            center_owner = (
                int(owner_i[center]),
                int(owner_j[center]),
                int(owner_k[center]),
            )
            storage_owner = (
                int(owner_i[storage]),
                int(owner_j[storage]),
                int(owner_k[storage]),
            )
            direct = (
                storage_owner == storage
                and is_active_owner[storage_owner]
                and not source[storage]
                and not aggregate_target[storage]
            )
            # A virtual sample is the regular-cell average predicted from the
            # centered stencil's active owner, never an arbitrary nearby cell.
            owner = storage_owner if direct else center_owner
            if not is_active_owner[owner] or (
                not direct and not reconstruction_owner_mask[owner]
            ):
                valid[row] = False
                continue
            sample_storage_i[row, column], sample_storage_j[row, column], sample_storage_k[row, column] = storage
            sample_owner_i[row, column], sample_owner_j[row, column], sample_owner_k[row, column] = owner
            sample_center_owner_i[row, column], sample_center_owner_j[row, column], sample_center_owner_k[row, column] = center_owner
            sample_active[row, column] = True
            sample_direct[row, column] = direct
            sample_displacement[row, column] = full_centroid[storage] - centroid[owner]
            sample_moment_delta[row, column] = (
                full_m2[storage]
                + np.outer(sample_displacement[row, column], sample_displacement[row, column])
                - second_moment[owner]
            )
            delta = sample_displacement[row, column]
            sample_third_moment_delta[row, column] = (
                full_m3[storage]
                + delta[:, None, None] * full_m2[storage][None, :, :]
                + delta[None, :, None] * full_m2[storage][:, None, :]
                + delta[None, None, :] * full_m2[storage][:, :, None]
                + delta[:, None, None] * delta[None, :, None] * delta[None, None, :]
                - cells.third_moment[owner]
            )
            scalar_coefficients[row, column] = scalar
            if derivative_axis is not None:
                gradient_coefficients[row, derivative_axis, column] = gradient_coefficient
        sample_count[row] = len(occurrences)

    for row in np.flatnonzero(active & valid):
        count = int(sample_count[row])
        if (
            count <= 0
            or not np.isclose(
                np.sum(scalar_coefficients[row, :count]),
                1.0,
                rtol=0.0,
                atol=1.0e-13,
            )
            or not np.allclose(
                np.sum(gradient_coefficients[row, :, :count], axis=-1),
                0.0,
                rtol=0.0,
                atol=1.0e-13,
            )
        ):
            valid[row] = False

    return LocalRegularTransitionFaceRows3D(
        layout=geometry.layout,
        irregular_face_row=jnp.arange(max_rows, dtype=jnp.int32),
        active=jnp.asarray(active), valid=jnp.asarray(valid),
        has_remote_owner=jnp.asarray(remote), sample_count=jnp.asarray(sample_count),
        face_axis=jnp.asarray(face_axis), face_i=jnp.asarray(face_i), face_j=jnp.asarray(face_j), face_k=jnp.asarray(face_k),
        sample_storage_i=jnp.asarray(sample_storage_i), sample_storage_j=jnp.asarray(sample_storage_j), sample_storage_k=jnp.asarray(sample_storage_k),
        sample_owner_i=jnp.asarray(sample_owner_i), sample_owner_j=jnp.asarray(sample_owner_j), sample_owner_k=jnp.asarray(sample_owner_k),
        sample_center_owner_i=jnp.asarray(sample_center_owner_i), sample_center_owner_j=jnp.asarray(sample_center_owner_j), sample_center_owner_k=jnp.asarray(sample_center_owner_k),
        sample_active=jnp.asarray(sample_active), sample_direct=jnp.asarray(sample_direct),
        sample_remote=jnp.zeros((max_rows, max_samples), dtype=bool),
        sample_displacement=jnp.asarray(sample_displacement), sample_moment_delta=jnp.asarray(sample_moment_delta),
        sample_third_moment_delta=jnp.asarray(sample_third_moment_delta),
        scalar_coefficients=jnp.asarray(scalar_coefficients), gradient_coefficients=jnp.asarray(gradient_coefficients),
        max_rows=max_rows, max_samples=max_samples,
    )


def _build_closed_box_embedded_control_volume_geometry(
    geometry: LocalFciGeometry3D,
    *,
    enable_merging: bool,
    cells: LocalControlVolumeCellGeometry3D | None = None,
    remote_boundary_payloads: dict[
        tuple[int, int],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]
    | None = None,
    remote_reconstruction_samples: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
    | None = None,
    local_periodic_axes: tuple[bool, bool, bool] = (False, True, True),
) -> LocalEmbeddedControlVolumeGeometry3D:
    if cells is None:
        cells = _build_closed_box_control_volume_cells(
            geometry,
            enable_merging=enable_merging,
        )
    # First discover intrinsic embedded-boundary owners, then rebuild face
    # ownership so every face touching one of those owners is compact.
    _regular_faces, seed_irregular_faces = _build_closed_box_control_volume_faces(
        geometry,
        cells,
        remote_boundary_payloads=remote_boundary_payloads,
    )
    intrinsic_reconstruction_owner_mask = _intrinsic_reconstruction_owner_mask(
        cells,
        seed_irregular_faces,
    )
    reconstruction_owner_mask = _dilate_reconstruction_owner_mask(
        cells,
        intrinsic_reconstruction_owner_mask,
        periodic_axes=local_periodic_axes,
    )
    regular_faces, irregular_faces = _build_closed_box_control_volume_faces(
        geometry,
        cells,
        remote_boundary_payloads=remote_boundary_payloads,
        reconstruction_owner_mask=reconstruction_owner_mask,
        compact_owner_mask=intrinsic_reconstruction_owner_mask,
    )
    reconstruction_owner_mask = _dilate_reconstruction_owner_mask(
        cells,
        reconstruction_owner_mask
        | _transition_face_owner_mask(cells, irregular_faces),
        periodic_axes=local_periodic_axes,
    )
    spacing = jnp.stack(
        (
            geometry.spacing.dx_owned,
            geometry.spacing.dy_owned,
            geometry.spacing.dz_owned,
        ),
        axis=-1,
    )
    if remote_reconstruction_samples is None:
        remote_halo_indices = None
        remote_centroids = None
        remote_second_moments = None
        remote_third_moments = None
    else:
        (
            remote_halo_indices,
            remote_centroids,
            remote_second_moments,
            remote_third_moments,
        ) = remote_reconstruction_samples
    reconstruction = precompute_local_moment_reconstruction(
        cells,
        irregular_faces,
        spacing_owned=spacing,
        remote_sample_halo_indices=remote_halo_indices,
        remote_sample_centroids=remote_centroids,
        remote_sample_second_moments=remote_second_moments,
        remote_sample_third_moments=remote_third_moments,
        periodic_axes=local_periodic_axes,
        coordinate_periodic_axes=(False, True, True),
        coordinate_periods=(
            float(shifted_mms.x_max) - float(shifted_mms.x_min),
            2.0 * np.pi,
            2.0 * np.pi,
        ),
        target_mask=jnp.asarray(reconstruction_owner_mask),
        max_samples=48,
        max_equations=64,
    )
    reconstruction_active = np.asarray(reconstruction.active, dtype=bool)
    reconstruction_order = np.asarray(
        reconstruction.polynomial_order,
        dtype=np.int32,
    )
    fallback_count = int(
        np.sum(reconstruction_active & (reconstruction_order < 3))
    )
    invalid_count = int(np.sum(~reconstruction_active))
    if fallback_count:
        raise ValueError(
            "shifted-torus control-volume fixture requires cubic "
            f"reconstruction on every active row; found {fallback_count} "
            "lower-order fallbacks"
        )
    if invalid_count:
        raise ValueError(
            "shifted-torus control-volume fixture produced "
            f"{invalid_count} invalid reconstruction rows"
        )
    centroid_points = np.asarray(cells.centroid, dtype=np.float64)
    (
        centroid_J,
        _centroid_g_contra,
        centroid_g_cov,
        centroid_B_contra,
        centroid_Bmag,
        _centroid_projector,
    ) = _shifted_torus_metric_payload_numpy(centroid_points)
    centroid_curvature = np.asarray(
        _shifted_torus_curvature_at_logical_points(
            jnp.asarray(centroid_points, dtype=jnp.float64)
        ),
        dtype=np.float64,
    )
    transition_faces = _build_regular_transition_face_rows(
        geometry,
        cells,
        irregular_faces,
        reconstruction_owner_mask,
        local_periodic_axes=local_periodic_axes,
    )
    invalid_transition_count = int(
        np.sum(
            np.asarray(transition_faces.active, dtype=bool)
            & ~np.asarray(transition_faces.valid, dtype=bool)
        )
    )
    if invalid_transition_count:
        raise ValueError(
            "shifted-torus control-volume fixture produced "
            f"{invalid_transition_count} invalid regular transition rows"
        )
    transition_active = np.asarray(transition_faces.active, dtype=bool)
    transition_rows = np.flatnonzero(transition_active)
    target_row = np.asarray(
        reconstruction.target_row_for_cell,
        dtype=np.int32,
    )
    for row in transition_rows:
        owners = (
            (
                int(np.asarray(irregular_faces.minus_owner_i)[row]),
                int(np.asarray(irregular_faces.minus_owner_j)[row]),
                int(np.asarray(irregular_faces.minus_owner_k)[row]),
            ),
            (
                int(np.asarray(irregular_faces.plus_owner_i)[row]),
                int(np.asarray(irregular_faces.plus_owner_j)[row]),
                int(np.asarray(irregular_faces.plus_owner_k)[row]),
            ),
        )
        if any(target_row[owner] < 0 for owner in owners):
            raise ValueError(
                "each local compact transition face owner must have a "
                "cubic reconstruction row"
            )
    return LocalEmbeddedControlVolumeGeometry3D(
        cells=cells,
        regular_faces=regular_faces,
        irregular_faces=irregular_faces,
        reconstruction=reconstruction,
        regular_transition_faces=transition_faces,
        centroid_J=jnp.asarray(centroid_J),
        centroid_g_cov=jnp.asarray(centroid_g_cov),
        centroid_B_contra=jnp.asarray(centroid_B_contra),
        centroid_Bmag=jnp.asarray(centroid_Bmag),
        centroid_curvature=jnp.asarray(centroid_curvature),
        regular_boundary_closure=(
            _build_shifted_torus_regular_boundary_closure(
                geometry,
                cells,
            )
        ),
    )



def _pad_control_volume_face_rows(
    rows: LocalControlVolumeFaceRows3D,
    max_rows: int,
) -> LocalControlVolumeFaceRows3D:
    max_rows = int(max_rows)
    pad = max_rows - int(rows.max_rows)
    if pad < 0:
        raise ValueError("max_rows must be at least rows.max_rows")
    if pad == 0:
        return rows

    def row_pad(value: jnp.ndarray, fill_value=0):
        widths = ((0, pad),) + tuple((0, 0) for _ in range(value.ndim - 1))
        return jnp.pad(value, widths, constant_values=fill_value)

    return LocalControlVolumeFaceRows3D(
        layout=rows.layout,
        kind=row_pad(rows.kind),
        minus_owner_i=row_pad(rows.minus_owner_i),
        minus_owner_j=row_pad(rows.minus_owner_j),
        minus_owner_k=row_pad(rows.minus_owner_k),
        plus_owner_i=row_pad(rows.plus_owner_i),
        plus_owner_j=row_pad(rows.plus_owner_j),
        plus_owner_k=row_pad(rows.plus_owner_k),
        has_plus_owner=row_pad(rows.has_plus_owner, False),
        quadrature_points=row_pad(rows.quadrature_points),
        area_covector_weight=row_pad(rows.area_covector_weight),
        J=row_pad(rows.J),
        g_contra=row_pad(rows.g_contra),
        g_cov=row_pad(rows.g_cov),
        B_contra=row_pad(rows.B_contra),
        Bmag=row_pad(rows.Bmag, 1.0),
        projector=row_pad(rows.projector),
        patch_active=row_pad(rows.patch_active, False),
        active=row_pad(rows.active, False),
        max_rows=max_rows,
        max_patches=rows.max_patches,
        has_remote_owner=row_pad(rows.has_remote_owner, False),
        remote_halo_i=row_pad(rows.remote_halo_i),
        remote_halo_j=row_pad(rows.remote_halo_j),
        remote_halo_k=row_pad(rows.remote_halo_k),
        remote_centroid=row_pad(rows.remote_centroid),
        remote_second_moment=row_pad(rows.remote_second_moment),
        remote_third_moment=row_pad(rows.remote_third_moment),
    )


def _pad_quadratic_reconstruction(
    rows: LocalMomentReconstruction3D,
    max_rows: int,
) -> LocalMomentReconstruction3D:
    max_rows = int(max_rows)
    pad = max_rows - int(rows.max_rows)
    if pad < 0:
        raise ValueError("max_rows must be at least rows.max_rows")
    if pad == 0:
        return rows

    def row_pad(value: jnp.ndarray, fill_value=0):
        widths = ((0, pad),) + tuple((0, 0) for _ in range(value.ndim - 1))
        return jnp.pad(value, widths, constant_values=fill_value)

    return LocalMomentReconstruction3D(
        layout=rows.layout,
        target_i=row_pad(rows.target_i),
        target_j=row_pad(rows.target_j),
        target_k=row_pad(rows.target_k),
        equation_kind=row_pad(rows.equation_kind),
        sample_i=row_pad(rows.sample_i),
        sample_j=row_pad(rows.sample_j),
        sample_k=row_pad(rows.sample_k),
        boundary_face_row=row_pad(rows.boundary_face_row),
        boundary_patch=row_pad(rows.boundary_patch),
        boundary_quadrature=row_pad(rows.boundary_quadrature),
        equation_active=row_pad(rows.equation_active, False),
        rhs_transform=row_pad(rows.rhs_transform),
        active=row_pad(rows.active, False),
        target_row_for_cell=rows.target_row_for_cell,
        polynomial_order=row_pad(rows.polynomial_order),
        rank=row_pad(rows.rank),
        condition_number=row_pad(rows.condition_number, jnp.inf),
        max_rows=max_rows,
        max_equations=rows.max_equations,
    )


def _pad_embedded_control_volume_geometry(
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
    *,
    max_face_rows: int,
    max_reconstruction_rows: int,
) -> LocalEmbeddedControlVolumeGeometry3D:
    transition = control_volume_geometry.regular_transition_faces
    transition_pad = max_face_rows - int(transition.max_rows)
    if transition_pad < 0:
        raise ValueError(
            "max_face_rows must be at least regular transition max_rows"
        )

    def transition_row_pad(value: jnp.ndarray, fill_value=0):
        widths = ((0, transition_pad),) + tuple(
            (0, 0) for _ in range(value.ndim - 1)
        )
        return jnp.pad(value, widths, constant_values=fill_value)

    return LocalEmbeddedControlVolumeGeometry3D(
        cells=control_volume_geometry.cells,
        regular_faces=control_volume_geometry.regular_faces,
        irregular_faces=_pad_control_volume_face_rows(
            control_volume_geometry.irregular_faces,
            max_face_rows,
        ),
        reconstruction=_pad_quadratic_reconstruction(
            control_volume_geometry.reconstruction,
            max_reconstruction_rows,
        ),
        regular_transition_faces=LocalRegularTransitionFaceRows3D(
            layout=transition.layout,
            irregular_face_row=transition_row_pad(
                transition.irregular_face_row
            ),
            active=transition_row_pad(transition.active, False),
            valid=transition_row_pad(transition.valid, False),
            has_remote_owner=transition_row_pad(
                transition.has_remote_owner,
                False,
            ),
            sample_count=transition_row_pad(transition.sample_count),
            face_axis=transition_row_pad(transition.face_axis),
            face_i=transition_row_pad(transition.face_i),
            face_j=transition_row_pad(transition.face_j),
            face_k=transition_row_pad(transition.face_k),
            sample_storage_i=transition_row_pad(transition.sample_storage_i),
            sample_storage_j=transition_row_pad(transition.sample_storage_j),
            sample_storage_k=transition_row_pad(transition.sample_storage_k),
            sample_owner_i=transition_row_pad(transition.sample_owner_i),
            sample_owner_j=transition_row_pad(transition.sample_owner_j),
            sample_owner_k=transition_row_pad(transition.sample_owner_k),
            sample_center_owner_i=transition_row_pad(
                transition.sample_center_owner_i
            ),
            sample_center_owner_j=transition_row_pad(
                transition.sample_center_owner_j
            ),
            sample_center_owner_k=transition_row_pad(
                transition.sample_center_owner_k
            ),
            sample_active=transition_row_pad(transition.sample_active, False),
            sample_direct=transition_row_pad(transition.sample_direct, False),
            sample_remote=transition_row_pad(transition.sample_remote, False),
            sample_displacement=transition_row_pad(transition.sample_displacement),
            sample_moment_delta=transition_row_pad(transition.sample_moment_delta),
            sample_third_moment_delta=transition_row_pad(transition.sample_third_moment_delta),
            scalar_coefficients=transition_row_pad(transition.scalar_coefficients),
            gradient_coefficients=transition_row_pad(transition.gradient_coefficients),
            max_rows=max_face_rows,
            max_samples=transition.max_samples,
        ),
        centroid_J=control_volume_geometry.centroid_J,
        centroid_g_cov=control_volume_geometry.centroid_g_cov,
        centroid_B_contra=control_volume_geometry.centroid_B_contra,
        centroid_Bmag=control_volume_geometry.centroid_Bmag,
        centroid_curvature=control_volume_geometry.centroid_curvature,
        regular_boundary_closure=(
            control_volume_geometry.regular_boundary_closure
        ),
    )


def _build_stacked_embedded_control_volume_geometry(
    *,
    global_shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int],
    halo_width: int,
    enable_merging: bool,
) -> LocalEmbeddedControlVolumeGeometry3D:
    """Precompute compact geometry and reconstruction transforms on the host."""

    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(global_shape, shard_counts)
    )
    local_geometry_and_cells: dict[
        tuple[int, int, int],
        tuple[LocalFciGeometry3D, LocalControlVolumeCellGeometry3D],
    ] = {}
    for shard_i in range(int(shard_counts[0])):
        for shard_j in range(int(shard_counts[1])):
            for shard_k in range(int(shard_counts[2])):
                shard_index = (shard_i, shard_j, shard_k)
                local_geometry = build_shifted_torus_local_geometry(
                    owned_shape,
                    halo_width,
                    global_shape=global_shape,
                    shard_index=shard_index,
                    x_min=shifted_mms.x_min,
                    x_max=shifted_mms.x_max,
                    r0=shifted_mms.r0,
                    alpha_value=shifted_mms.alpha_value,
                    iota=shifted_mms.iota,
                    c_phi=shifted_mms.c_phi,
                    sigma=shifted_mms.sigma,
                )
                local_geometry_and_cells[shard_index] = (
                    local_geometry,
                    _build_closed_box_control_volume_cells(
                        local_geometry,
                        enable_merging=enable_merging,
                    ),
                )

    periodic_axes = (False, True, True)

    def neighbor_index(
        shard_index: tuple[int, int, int],
        axis: int,
        side: int,
    ) -> tuple[int, int, int] | None:
        index = list(shard_index)
        offset = -1 if side == 0 else 1
        candidate = index[axis] + offset
        if 0 <= candidate < int(shard_counts[axis]):
            index[axis] = candidate
            return tuple(index)
        if periodic_axes[axis] and int(shard_counts[axis]) > 1:
            index[axis] = candidate % int(shard_counts[axis])
            return tuple(index)
        return None

    local_bundles: list[LocalEmbeddedControlVolumeGeometry3D] = []
    for shard_index, (local_geometry, cells) in local_geometry_and_cells.items():
        local_touched_storage = _closed_box_irregular_storage_mask(
            local_geometry,
            cells,
        )
        remote_boundary_payloads: dict[
            tuple[int, int],
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        remote_sample_halo_indices: list[tuple[int, int, int]] = []
        remote_sample_centroids: list[np.ndarray] = []
        remote_sample_second_moments: list[np.ndarray] = []
        remote_sample_third_moments: list[np.ndarray] = []
        remote_sample_owners: set[tuple[int, ...]] = set()

        for axis in range(3):
            for side in (0, 1):
                remote_index = neighbor_index(shard_index, axis, side)
                if remote_index is None:
                    continue
                remote_geometry, remote_cells = local_geometry_and_cells[remote_index]
                remote_raw_volume = np.asarray(
                    remote_cells.raw_volume,
                    dtype=np.float64,
                )
                remote_touched_storage = _closed_box_irregular_storage_mask(
                    remote_geometry,
                    remote_cells,
                )
                remote_owner_index = (
                    np.asarray(remote_cells.owner_i, dtype=np.int32),
                    np.asarray(remote_cells.owner_j, dtype=np.int32),
                    np.asarray(remote_cells.owner_k, dtype=np.int32),
                )
                remote_storage_centroid = np.asarray(
                    remote_cells.centroid,
                    dtype=np.float64,
                )[remote_owner_index]
                remote_storage_second_moment = np.asarray(
                    remote_cells.second_moment,
                    dtype=np.float64,
                )[remote_owner_index]
                remote_storage_third_moment = np.asarray(
                    remote_cells.third_moment,
                    dtype=np.float64,
                )[remote_owner_index]
                wraps_periodic_boundary = periodic_axes[axis] and (
                    (side == 0 and shard_index[axis] == 0)
                    or (
                        side == 1
                        and shard_index[axis] == int(shard_counts[axis]) - 1
                    )
                )
                if wraps_periodic_boundary:
                    period = 2.0 * np.pi
                    remote_storage_centroid = remote_storage_centroid.copy()
                    remote_storage_centroid[..., axis] += (
                        -period if side == 0 else period
                    )
                remote_boundary_index = (
                    remote_cells.shape[axis] - 1 if side == 0 else 0
                )
                remote_boundary_payloads[(axis, side)] = (
                    np.take(
                        remote_touched_storage,
                        remote_boundary_index,
                        axis=axis,
                    ),
                    np.take(
                        remote_raw_volume > 1.0e-20,
                        remote_boundary_index,
                        axis=axis,
                    ),
                    np.take(
                        remote_storage_centroid,
                        remote_boundary_index,
                        axis=axis,
                    ),
                    np.take(
                        remote_storage_second_moment,
                        remote_boundary_index,
                        axis=axis,
                    ),
                    np.take(
                        remote_storage_third_moment,
                        remote_boundary_index,
                        axis=axis,
                    ),
                )
                local_boundary_index = 0 if side == 0 else cells.shape[axis] - 1
                # Reconstruction targets include physical-boundary owners as
                # well as cut-wall/aggregate owners.  Export the complete
                # active face-halo layer so a target near a decomposed axis
                # sees the same radius-one/radius-two neighborhood as the
                # equivalent single-shard reconstruction.
                needed_tangential = np.ones_like(
                    np.take(
                        local_touched_storage,
                        local_boundary_index,
                        axis=axis,
                    ),
                    dtype=bool,
                )
                remote_extent = remote_cells.shape[axis]
                layer_width = min(int(halo_width), int(remote_extent))
                remote_layer_indices = (
                    np.arange(remote_extent - layer_width, remote_extent)
                    if side == 0
                    else np.arange(0, layer_width)
                )
                remote_layer_fluid = np.take(
                    remote_raw_volume > 1.0e-20,
                    remote_layer_indices,
                    axis=axis,
                )
                needed_layer = np.expand_dims(needed_tangential, axis=axis)
                sample_storage_rows = np.argwhere(
                    remote_layer_fluid & needed_layer
                )
                for layer_storage_array in sample_storage_rows:
                    layer_storage = [
                        int(value) for value in layer_storage_array
                    ]
                    remote_storage = list(layer_storage)
                    remote_storage[axis] = int(
                        remote_layer_indices[layer_storage[axis]]
                    )
                    remote_storage_index = tuple(remote_storage)
                    remote_owner = tuple(
                        int(owner_component[remote_storage_index])
                        for owner_component in remote_owner_index
                    )
                    owner_key = (axis, side, *remote_index, *remote_owner)
                    if owner_key in remote_sample_owners:
                        continue
                    remote_sample_owners.add(owner_key)
                    halo_index = [
                        int(halo_width) + remote_storage[d]
                        for d in range(3)
                    ]
                    halo_index[axis] = (
                        layer_storage[axis]
                        if side == 0
                        else int(halo_width)
                        + cells.shape[axis]
                        + layer_storage[axis]
                    )
                    remote_sample_halo_indices.append(tuple(halo_index))
                    remote_sample_centroids.append(
                        remote_storage_centroid[remote_storage_index]
                    )
                    remote_sample_second_moments.append(
                        remote_storage_second_moment[remote_storage_index]
                    )
                    remote_sample_third_moments.append(
                        remote_storage_third_moment[remote_storage_index]
                    )
        remote_reconstruction_samples = (
            np.asarray(remote_sample_halo_indices, dtype=np.int32).reshape((-1, 3)),
            np.asarray(remote_sample_centroids, dtype=np.float64).reshape((-1, 3)),
            np.asarray(
                remote_sample_second_moments,
                dtype=np.float64,
            ).reshape((-1, 3, 3)),
            np.asarray(
                remote_sample_third_moments,
                dtype=np.float64,
            ).reshape((-1, 3, 3, 3)),
        )
        local_bundles.append(
            _build_closed_box_embedded_control_volume_geometry(
                local_geometry,
                enable_merging=enable_merging,
                cells=cells,
                remote_boundary_payloads=remote_boundary_payloads,
                remote_reconstruction_samples=remote_reconstruction_samples,
                local_periodic_axes=tuple(
                    periodic_axes[axis] and int(shard_counts[axis]) == 1
                    for axis in range(3)
                ),
            )
        )
    max_face_rows = max(
        (
            int(bundle.irregular_faces.max_rows)
            for bundle in local_bundles
        ),
        default=0,
    )
    max_reconstruction_rows = max(
        (
            int(bundle.reconstruction.max_rows)
            for bundle in local_bundles
        ),
        default=0,
    )
    padded = iter(
        _pad_embedded_control_volume_geometry(
            bundle,
            max_face_rows=max_face_rows,
            max_reconstruction_rows=max_reconstruction_rows,
        )
        for bundle in local_bundles
    )

    def builder(_shard_index):
        return next(padded)

    return stack_local_shard_pytree(shard_counts, builder)


def _with_embedded_control_volume_geometry(
    geometry: LocalFciGeometry3D,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
) -> LocalFciGeometry3D:
    """Align legacy geometry masks/measures with the authoritative CV cells."""

    cells = control_volume_geometry.cells
    logical_volume = (
        jnp.asarray(geometry.spacing.dx_owned, dtype=jnp.float64)
        * jnp.asarray(geometry.spacing.dy_owned, dtype=jnp.float64)
        * jnp.asarray(geometry.spacing.dz_owned, dtype=jnp.float64)
    )
    base_J = jnp.maximum(
        jnp.asarray(geometry.cell_volume_geometry.volume, dtype=jnp.float64),
        1.0e-30,
    )
    volume_fraction = cells.aggregate_volume / jnp.maximum(
        base_J * logical_volume,
        1.0e-30,
    )
    return dataclass_replace(
        geometry,
        active_cell_mask=cells.is_active_owner,
        cell_volume_geometry=LocalCellVolumeGeometry3D(
            layout=geometry.layout,
            volume=geometry.cell_volume_geometry.volume,
            volume_fraction=volume_fraction,
        ),
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


def _print_control_volume_geometry_summary(
    stacked: LocalEmbeddedControlVolumeGeometry3D,
) -> None:
    cells = stacked.cells
    faces = stacked.irregular_faces
    reconstruction = stacked.reconstruction
    transitions = stacked.regular_transition_faces
    face_active = np.asarray(faces.active, dtype=bool)
    face_kind = np.asarray(faces.kind, dtype=np.int32)
    reconstruction_active = np.asarray(reconstruction.active, dtype=bool)
    reconstruction_order = np.asarray(
        reconstruction.polynomial_order,
        dtype=np.int32,
    )
    condition = np.asarray(
        reconstruction.condition_number,
        dtype=np.float64,
    )
    finite_condition = condition[
        reconstruction_active & np.isfinite(condition)
    ]
    transition_active = np.asarray(transitions.active, dtype=bool)
    transition_sample_active = (
        transition_active[..., None]
        & np.asarray(transitions.sample_active, dtype=bool)
    )
    transition_direct = int(
        np.sum(
            transition_sample_active
            & np.asarray(transitions.sample_direct, dtype=bool)
        )
    )
    transition_virtual = int(np.sum(transition_sample_active) - transition_direct)
    print(
        "embedded control volumes: "
        f"active_owners={int(np.sum(np.asarray(cells.is_active_owner)))}, "
        f"merged_sources={int(np.sum(np.asarray(cells.is_merged_source)))}, "
        f"aggregate_targets={int(np.sum(np.asarray(cells.is_aggregate_target)))}, "
        f"irregular_faces={int(np.sum(face_active))}, "
        "interior/partial/cutwall="
        f"{int(np.sum(face_active & (face_kind == CV_FACE_INTERIOR)))}/"
        f"{int(np.sum(face_active & (face_kind == CV_FACE_PARTIAL)))}/"
        f"{int(np.sum(face_active & (face_kind == CV_FACE_CUT_WALL)))}, "
        "physical_boundary="
        f"{int(np.sum(face_active & (face_kind == CV_FACE_PHYSICAL_BOUNDARY)))}, "
        "transitions/invalid/remote/max_samples="
        f"{int(np.sum(np.asarray(transitions.active, dtype=bool)))}/"
        f"{int(np.sum(np.asarray(transitions.active, dtype=bool) & ~np.asarray(transitions.valid, dtype=bool)))}/"
        f"{int(np.sum(np.asarray(transitions.has_remote_owner, dtype=bool)))}/"
        f"{int(np.max(np.asarray(transitions.sample_count, dtype=np.int32))) if int(transitions.max_rows) else 0}, "
        "transition_direct/virtual="
        f"{transition_direct}/{transition_virtual}, "
        f"cubic_rows={int(np.sum(reconstruction_active & (reconstruction_order == 3)))}, "
        f"quadratic_fallbacks={int(np.sum(reconstruction_active & (reconstruction_order == 2)))}, "
        f"linear_fallbacks={int(np.sum(reconstruction_active & (reconstruction_order == 1)))}, "
        "max_condition="
        f"{float(np.max(finite_condition)) if finite_condition.size else 0.0:.6e}"
    )



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


def _shifted_torus_metric_payload_jax(
    points: jnp.ndarray,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    points = jnp.asarray(points, dtype=jnp.float64)
    x = points[..., 0]
    theta = points[..., 1]
    x_mid = 0.5 * (float(shifted_mms.x_min) + float(shifted_mms.x_max))
    theta_shift = theta + float(shifted_mms.sigma) * (x - x_mid)
    cos_theta = jnp.cos(theta_shift)
    sin_theta = jnp.sin(theta_shift)
    alpha = float(shifted_mms.alpha_value)
    radius = float(shifted_mms.r0) + alpha * x + x * cos_theta
    q_value = 1.0 + alpha * cos_theta
    J = radius * x * q_value
    zeros = jnp.zeros_like(J)
    g_contra = jnp.stack(
        (
            jnp.stack(
                (
                    1.0 / q_value**2,
                    alpha * sin_theta / (x * q_value**2),
                    zeros,
                ),
                axis=-1,
            ),
            jnp.stack(
                (
                    alpha * sin_theta / (x * q_value**2),
                    (1.0 + 2.0 * alpha * cos_theta + alpha**2)
                    / (x**2 * q_value**2),
                    zeros,
                ),
                axis=-1,
            ),
            jnp.stack((zeros, zeros, 1.0 / radius**2), axis=-1),
        ),
        axis=-2,
    )
    g_cov = jnp.stack(
        (
            jnp.stack(
                (
                    1.0 + 2.0 * alpha * cos_theta + alpha**2,
                    -alpha * x * sin_theta,
                    zeros,
                ),
                axis=-1,
            ),
            jnp.stack((-alpha * x * sin_theta, x**2, zeros), axis=-1),
            jnp.stack((zeros, zeros, radius**2), axis=-1),
        ),
        axis=-2,
    )
    B_contra = jnp.stack(
        (
            zeros,
            float(shifted_mms.iota) * float(shifted_mms.c_phi) / J,
            float(shifted_mms.c_phi) / J,
        ),
        axis=-1,
    )
    Bmag = jnp.sqrt(
        jnp.einsum("...i,...ij,...j->...", B_contra, g_cov, B_contra)
    )
    unit_b = B_contra / jnp.maximum(Bmag[..., None], 1.0e-30)
    projector = g_contra - unit_b[..., :, None] * unit_b[..., None, :]
    return J, g_contra, g_cov, B_contra, Bmag, projector


def _shifted_torus_curvature_at_logical_points(
    points: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the continuum curvature coefficients at arbitrary points."""

    points = jnp.asarray(points, dtype=jnp.float64)
    x = points[..., 0]
    theta = points[..., 1]

    def covariant_field(
        radial: jnp.ndarray,
        poloidal: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        sample = jnp.stack(
            (radial, poloidal, jnp.zeros_like(radial)),
            axis=-1,
        )
        J, _g_contra, g_cov, B_contra, Bmag, _projector = (
            _shifted_torus_metric_payload_jax(sample)
        )
        covariant = jnp.einsum(
            "...ij,...j->...i",
            g_cov,
            B_contra,
        ) / jnp.maximum(Bmag[..., None] ** 2, 1.0e-30)
        return (
            covariant[..., 0],
            covariant[..., 1],
            covariant[..., 2],
            J,
            Bmag,
        )

    ones_x = jnp.ones_like(x)
    ones_theta = jnp.ones_like(theta)

    def component(component: int, radial, poloidal):
        return covariant_field(radial, poloidal)[component]

    _, dA_zeta_dtheta = jax.jvp(
        lambda poloidal: component(2, x, poloidal),
        (theta,),
        (ones_theta,),
    )
    _, dA_zeta_dx = jax.jvp(
        lambda radial: component(2, radial, theta),
        (x,),
        (ones_x,),
    )
    _, dA_theta_dx = jax.jvp(
        lambda radial: component(1, radial, theta),
        (x,),
        (ones_x,),
    )
    _, dA_x_dtheta = jax.jvp(
        lambda poloidal: component(0, x, poloidal),
        (theta,),
        (ones_theta,),
    )
    _A_x, _A_theta, _A_zeta, J, Bmag = covariant_field(x, theta)
    curl = jnp.stack(
        (
            dA_zeta_dtheta,
            -dA_zeta_dx,
            dA_theta_dx - dA_x_dtheta,
        ),
        axis=-1,
    )
    return (
        Bmag[..., None]
        * curl
        / jnp.maximum(2.0 * J[..., None], 1.0e-30)
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
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D

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
        return LocalHaloClosure3D(
            physical_ghost_filler=self.physical_ghost_filler,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
        )(phi_halo, self.domain, face_bc)

    def evaluate_stage(
        self,
        state_owned: Fci4FieldState,
        stage_data: shifted_mms._ShiftedTorus4FieldStageData,
        phi_guess_owned: jnp.ndarray | None,
        *,
        phi_wall_offset: float | jax.Array = 0.0,
        solve_phi: bool = True,
    ) -> tuple[Fci4FieldState, jnp.ndarray]:
        prepared = shifted_mms._prepare_local_shifted_torus_4field_stage_state(
            state_owned,
            stage_data,
            self.domain,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
        )
        face_bc = prepared.boundary_data.face_bc
        control_volume_geometry = self.control_volume_geometry
        cells = control_volume_geometry.cells
        stage_time = jnp.asarray(
            stage_data.stage_time,
            dtype=jnp.float64,
        )
        phi_control_volume_bc = _control_volume_exact_boundary_bc(
            control_volume_geometry,
            stage_time,
            "phi",
            value_offset=phi_wall_offset,
        )
        aggregate_cell_positions_owned = cells.centroid

        def prepare_owner_field(
            values_owned: jnp.ndarray,
            field_face_bc: LocalBoundaryFaceBC3D,
        ) -> jnp.ndarray:
            return self._prepare_phi_halo(
                _expand_control_volume_owner_values(values_owned, cells),
                field_face_bc,
            )

        state_halo = Fci4FieldState(
            density=prepare_owner_field(state_owned.density, face_bc.density),
            omega=prepare_owner_field(state_owned.omega, face_bc.omega),
            v_ion_parallel=prepare_owner_field(
                state_owned.v_ion_parallel,
                face_bc.v_ion_parallel,
            ),
            v_electron_parallel=prepare_owner_field(
                state_owned.v_electron_parallel,
                face_bc.v_electron_parallel,
            ),
        )
        omega_owned = jnp.asarray(state_owned.omega, dtype=jnp.float64)
        raw_phi_lift_owned = jnp.asarray(
            stage_data.phi_halo[self.domain.layout.owned_slices_cell],
            dtype=jnp.float64,
        )
        phi_lift_owned = _agglomerate_control_volume_average(
            raw_phi_lift_owned,
            cells,
        )
        if solve_phi:
            phi_solver = LocalPerpLaplacianInverseSolver(
                geometry=self.geometry,
                domain=self.domain,
                stencil_builder=build_local_conservative_stencil_from_field,
                halo_exchange=self.halo_exchange,
                topology_filler=self.topology_filler,
                physical_ghost_filler=self.physical_ghost_filler,
                face_projectors=self.face_projectors,
                control_volume_geometry=control_volume_geometry,
                control_volume_boundary_bc=phi_control_volume_bc,
                face_bc=face_bc.phi,
                config=self.gmres_config,
            )
            phi_owned = phi_solver(
                -omega_owned,
                guess_owned=phi_guess_owned,
                phi_lift_owned=phi_lift_owned,
            )
        else:
            if phi_guess_owned is None:
                raise ValueError(
                    "solve_phi=False requires prescribed phi in "
                    "phi_guess_owned"
                )
            phi_owned = jnp.asarray(phi_guess_owned, dtype=jnp.float64)
        phi_owned = jnp.where(self.geometry.active_cell_mask_owned, phi_owned, 0.0)
        phi_halo = prepare_owner_field(phi_owned, face_bc.phi)

        context = StencilBuilderContext(
            layout=self.domain.layout,
            domain=self.domain,
        )

        def build_stencil(field_halo: jnp.ndarray):
            return build_local_stencil_from_field(
                field_halo,
                self.geometry,
                context,
            )

        def build_gradient(
            field_halo: jnp.ndarray,
            field_name: str,
            field_boundary_bc: LocalControlVolumeBoundaryBC3D | None = None,
            field_face_bc: LocalBoundaryFaceBC3D | None = None,
        ) -> tuple[LocalCellGradient3D, object, LocalControlVolumeBoundaryBC3D]:
            if field_boundary_bc is None:
                field_boundary_bc = _control_volume_exact_boundary_bc(
                    control_volume_geometry,
                    stage_time,
                    field_name,
                )
            if field_face_bc is None:
                field_face_bc = getattr(face_bc, field_name)
            polynomial = build_local_control_volume_polynomial_from_field(
                field_halo,
                self.geometry,
                self.domain,
                StencilBuilderContext(
                    layout=context.layout,
                    domain=context.domain,
                ),
                control_volume_geometry,
                field_boundary_bc,
                field_face_bc,
                halo_exchange=self.halo_exchange,
                topology_filler=self.topology_filler,
            )
            return polynomial.as_cell_gradient(), polynomial, field_boundary_bc

        density_gradient, density_polynomial, density_control_volume_bc = build_gradient(
            state_halo.density,
            "density",
        )
        omega_gradient, omega_polynomial, omega_control_volume_bc = build_gradient(
            state_halo.omega,
            "omega",
        )
        phi_gradient, phi_polynomial, _phi_control_volume_bc = build_gradient(
            phi_halo,
            "phi",
            phi_control_volume_bc,
        )
        (
            v_ion_gradient,
            v_ion_polynomial,
            v_ion_control_volume_bc,
        ) = build_gradient(
            state_halo.v_ion_parallel,
            "v_ion_parallel",
        )
        (
            v_electron_gradient,
            v_electron_polynomial,
            v_electron_control_volume_bc,
        ) = build_gradient(
            state_halo.v_electron_parallel,
            "v_electron_parallel",
        )

        density_owned = jnp.asarray(state_owned.density, dtype=jnp.float64)
        v_electron_owned = jnp.asarray(
            state_owned.v_electron_parallel,
            dtype=jnp.float64,
        )
        density_v_electron_owned = local_control_volume_product_average(
            density_owned,
            v_electron_owned,
            density_polynomial,
            v_electron_polynomial,
            cells,
        )
        density_v_electron_face_bc = _multiply_local_dirichlet_face_bc(
            face_bc.density,
            face_bc.v_electron_parallel,
        )
        density_v_electron_halo = prepare_owner_field(
            density_v_electron_owned,
            density_v_electron_face_bc,
        )
        density_v_electron_stencil = build_stencil(
            density_v_electron_halo
        )
        density_v_electron_conservative_stencil = (
            build_local_conservative_stencil_from_field(
                density_v_electron_halo,
                self.geometry,
                context,
            )
        )
        (
            density_v_electron_gradient,
            density_v_electron_polynomial,
            density_v_electron_control_volume_bc,
        ) = build_gradient(
            density_v_electron_halo,
            "density_v_electron",
            field_face_bc=density_v_electron_face_bc,
        )
        density_safe = jnp.maximum(density_owned, 1.0e-30)
        bmag_owned = jnp.maximum(
            jnp.asarray(
                (
                    control_volume_geometry.centroid_Bmag
                    if control_volume_geometry.has_centroid_operator_geometry
                    else self.geometry.cell_bfield.Bmag_owned
                ),
                dtype=jnp.float64,
            ),
            1.0e-30,
        )
        rho_star_value = jnp.asarray(self.parameters.rho_star, dtype=jnp.float64)
        te = jnp.asarray(self.parameters.Te, dtype=jnp.float64)
        mi_over_me_value = jnp.asarray(self.parameters.mi_over_me, dtype=jnp.float64)

        poisson_density = local_poisson_bracket_op_from_gradients(
            phi_gradient,
            density_gradient,
            self.geometry,
            control_volume_geometry=control_volume_geometry,
        )
        poisson_omega = local_poisson_bracket_op_from_gradients(
            phi_gradient,
            omega_gradient,
            self.geometry,
            control_volume_geometry=control_volume_geometry,
        )
        poisson_v_ion = local_poisson_bracket_op_from_gradients(
            phi_gradient,
            v_ion_gradient,
            self.geometry,
            control_volume_geometry=control_volume_geometry,
        )
        poisson_v_electron = local_poisson_bracket_op_from_gradients(
            phi_gradient,
            v_electron_gradient,
            self.geometry,
            control_volume_geometry=control_volume_geometry,
        )
        curvature_density = local_curvature_op_from_gradient(
            density_gradient,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
            control_volume_geometry=control_volume_geometry,
        )
        curvature_phi = local_curvature_op_from_gradient(
            phi_gradient,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
            control_volume_geometry=control_volume_geometry,
        )
        grad_parallel_density = local_grad_parallel_op_from_gradient(
            density_gradient,
            self.geometry,
            control_volume_geometry=control_volume_geometry,
        )
        grad_parallel_phi = local_grad_parallel_op_from_gradient(
            phi_gradient,
            self.geometry,
            control_volume_geometry=control_volume_geometry,
        )
        grad_parallel_v_ion = local_grad_parallel_op_from_gradient(
            v_ion_gradient,
            self.geometry,
            control_volume_geometry=control_volume_geometry,
        )
        grad_parallel_v_electron = local_grad_parallel_op_from_gradient(
            v_electron_gradient,
            self.geometry,
            control_volume_geometry=control_volume_geometry,
        )
        pressure_grad_parallel_density = grad_parallel_density
        pressure_density_safe = density_safe
        parallel_density_flux_divergence = local_parallel_flux_div_op(
            density_v_electron_conservative_stencil,
            self.geometry,
            self.domain,
            control_volume_geometry=control_volume_geometry,
            boundary_bc=density_v_electron_control_volume_bc,
            field_reconstruction=density_v_electron_polynomial,
        )

        density_poisson_term = -(poisson_density / (rho_star_value * bmag_owned))
        density_curvature_term = (2.0 * te / bmag_owned) * curvature_density
        density_phi_curvature_term = -(2.0 * density_owned / bmag_owned) * curvature_phi
        density_parallel_term = -parallel_density_flux_divergence
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
        v_ion_pressure_term = -(te / pressure_density_safe) * pressure_grad_parallel_density
        v_ion_rhs = (
            v_ion_poisson_term
            + v_ion_pressure_term
        )
        v_electron_poisson_term = -(poisson_v_electron / (rho_star_value * bmag_owned))
        v_electron_parallel_phi_term = mi_over_me_value * grad_parallel_phi
        v_electron_pressure_term = -mi_over_me_value * (
            te / pressure_density_safe
        ) * pressure_grad_parallel_density
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
        source_owned = Fci4FieldState(
            density=_agglomerate_control_volume_average(source_density, cells),
            omega=_agglomerate_control_volume_average(source_omega, cells),
            v_ion_parallel=_agglomerate_control_volume_average(
                source_v_ion,
                cells,
            ),
            v_electron_parallel=_agglomerate_control_volume_average(
                source_v_electron,
                cells,
            ),
        )
        rhs = Fci4FieldState(
            density=density_rhs + source_owned.density,
            omega=omega_rhs + source_owned.omega,
            v_ion_parallel=v_ion_rhs + source_owned.v_ion_parallel,
            v_electron_parallel=v_electron_rhs + source_owned.v_electron_parallel,
        )
        rhs = _mask_4field_state_inactive(rhs, self.geometry)

        return rhs, phi_owned


def _state_error_statistics(
    actual: Fci4FieldState,
    expected: Fci4FieldState,
) -> dict[str, tuple[float, float, float]]:
    return shifted_mms._state_error_statistics(actual, expected)



def _mask_4field_state_inactive(
    state: Fci4FieldState,
    geometry: LocalFciGeometry3D,
) -> Fci4FieldState:
    """Zero inactive owned cells for test-local 4-field RHS/update payloads."""

    active = geometry.active_cell_mask_owned
    return Fci4FieldState(
        density=jnp.where(active, state.density, 0.0),
        omega=jnp.where(active, state.omega, 0.0),
        v_ion_parallel=jnp.where(active, state.v_ion_parallel, 0.0),
        v_electron_parallel=jnp.where(active, state.v_electron_parallel, 0.0),
    )



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


def _volume_weighted_field_error_statistics(
    actual: jnp.ndarray,
    expected: jnp.ndarray,
    volume: jnp.ndarray,
    active_owner: jnp.ndarray,
) -> tuple[float, float, float]:
    actual = jnp.asarray(actual, dtype=jnp.float64)
    expected = jnp.asarray(expected, dtype=jnp.float64)
    weight = jnp.where(
        active_owner,
        jnp.asarray(volume, dtype=jnp.float64),
        0.0,
    )
    error = jnp.where(active_owner, actual - expected, 0.0)
    weight_sum = jnp.maximum(jnp.sum(weight), 1.0e-30)
    l2 = jnp.sqrt(jnp.sum(weight * error * error) / weight_sum)
    linf = jnp.max(jnp.where(active_owner, jnp.abs(error), 0.0))
    expected_norm = jnp.sqrt(
        jnp.sum(weight * expected * expected) / weight_sum
    )
    return (
        float(l2),
        float(linf),
        float(l2 / jnp.maximum(expected_norm, 1.0e-30)),
    )


def _volume_weighted_state_error_statistics(
    actual: Fci4FieldState,
    expected: Fci4FieldState,
    volume: jnp.ndarray,
    active_owner: jnp.ndarray,
) -> dict[str, tuple[float, float, float]]:
    return {
        name: _volume_weighted_field_error_statistics(
            getattr(actual, name),
            getattr(expected, name),
            volume,
            active_owner,
        )
        for name in (
            "density",
            "omega",
            "v_ion_parallel",
            "v_electron_parallel",
        )
    }


def _print_state_error_statistics(
    label: str,
    statistics: dict[str, tuple[float, float, float]],
) -> None:
    print(label)
    for field_name, (l2, linf, relative_l2) in statistics.items():
        print(
            f"  {field_name}: L2={l2:.6e}, Linf={linf:.6e}, "
            f"rel_L2={relative_l2:.6e}"
        )



def _resolution_step_count(resolution: int, *, base_steps: int) -> int:
    return max(
        1,
        int(
            round(
                float(base_steps)
                * float(resolution)
                / 20.0
            )
        ),
    )


def _make_parameters(rho_star_value: float) -> Fci4FieldRhsParameters:
    return Fci4FieldRhsParameters(
        rho_star=float(rho_star_value),
        Te=float(shifted_mms.Te),
        mi_over_me=float(shifted_mms.mi_over_me),
        phi_inversion_tol=1.0e-11,
        phi_inversion_maxiter=500,
        phi_inversion_restart=100,
    )


def _make_gmres_config(parameters: Fci4FieldRhsParameters) -> SpmdGmresConfig:
    return SpmdGmresConfig(
        tol=float(parameters.phi_inversion_tol),
        atol=float(parameters.phi_inversion_tol),
        maxiter=int(parameters.phi_inversion_maxiter),
        restart=int(parameters.phi_inversion_restart),
        stagnation_iters=0,
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
    enable_agglomeration: bool = False,
    stacked_control_volume_geometry: (
        LocalEmbeddedControlVolumeGeometry3D | None
    ) = None,
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
    if stacked_control_volume_geometry is None:
        stacked_control_volume_geometry = (
            _build_stacked_embedded_control_volume_geometry(
                global_shape=geometry.shape,
                shard_counts=shard_counts,
                halo_width=halo_width,
                enable_merging=enable_agglomeration,
            )
        )
    initial_state, initial_phi_guess = (
        _project_global_exact_state_to_control_volumes(
            geometry,
            stacked_control_volume_geometry,
            shard_counts=shard_counts,
            halo_width=halo_width,
            time=0.0,
        )
    )
    times: list[float] = [0.0]
    density_history: list[jnp.ndarray] = [jnp.asarray(initial_state.density, dtype=jnp.float32)]
    omega_history: list[jnp.ndarray] = [jnp.asarray(initial_state.omega, dtype=jnp.float32)]
    v_ion_history: list[jnp.ndarray] = [jnp.asarray(initial_state.v_ion_parallel, dtype=jnp.float32)]
    v_electron_history: list[jnp.ndarray] = [jnp.asarray(initial_state.v_electron_parallel, dtype=jnp.float32)]
    wall_step_times: list[float] = []

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        state = shifted_mms._put_state_on_mesh(initial_state, mesh)
        phi_guess = jax.device_put(
            jnp.asarray(initial_phi_guess, dtype=jnp.float64),
            NamedSharding(mesh, P(*MESH_AXIS_NAMES)),
        )
        state_spec = shifted_mms._state_partition_spec()
        field_spec = P(*MESH_AXIS_NAMES)
        control_volume_geometry_spec = local_shard_pytree_partition_spec(
            stacked_control_volume_geometry
        )
        control_volume_geometry_sharding = jax.tree_util.tree_map(
            lambda spec: NamedSharding(mesh, spec),
            control_volume_geometry_spec,
        )
        control_volume_geometry = jax.device_put(
            stacked_control_volume_geometry,
            control_volume_geometry_sharding,
        )
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

        def _local_geometry_for_shard(
            local_control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
        ) -> LocalFciGeometry3D:
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
            return _with_embedded_control_volume_geometry(
                local_geometry,
                local_control_volume_geometry,
            )

        def source_kernel(
            local_invariants: shifted_mms._ShiftedTorus4FieldInvariantBundle,
            local_control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
            step_time: jax.Array,
            step_timestep: jax.Array,
        ) -> shifted_mms._ShiftedTorus4FieldRk4StageData:
            local_invariants = extract_local_shard_pytree(local_invariants)
            local_control_volume_geometry = extract_local_shard_pytree(
                local_control_volume_geometry
            )
            local_geometry = _local_geometry_for_shard(
                local_control_volume_geometry
            )
            stage_data = shifted_mms._build_local_4field_rk4_stage_data(
                local_invariants,
                step_time,
                step_timestep,
                parameters=parameters,
            )

            def project_stage(
                stage: shifted_mms._ShiftedTorus4FieldStageData,
                stage_time: jax.Array,
            ) -> shifted_mms._ShiftedTorus4FieldStageData:
                stage = _with_shifted_torus_regular_radial_face_averages(
                    stage,
                    local_geometry,
                    stage_time,
                )
                owner_source = _project_local_mms_source_to_control_volumes(
                    local_geometry,
                    local_control_volume_geometry,
                    stage_time,
                    parameters,
                )
                cells = local_control_volume_geometry.cells
                expanded_source = Fci4FieldState(
                    density=_expand_control_volume_owner_values(
                        owner_source.density,
                        cells,
                    ),
                    omega=_expand_control_volume_owner_values(
                        owner_source.omega,
                        cells,
                    ),
                    v_ion_parallel=_expand_control_volume_owner_values(
                        owner_source.v_ion_parallel,
                        cells,
                    ),
                    v_electron_parallel=_expand_control_volume_owner_values(
                        owner_source.v_electron_parallel,
                        cells,
                    ),
                )
                return dataclass_replace(
                    stage,
                    stage_time=jnp.asarray(stage_time, dtype=jnp.float64),
                    source_halo=inject_owned_state_to_halo(
                        expanded_source,
                        domain.layout,
                    ),
                )

            half_step = 0.5 * step_timestep
            projected = shifted_mms._ShiftedTorus4FieldRk4StageData(
                stage_1=project_stage(stage_data.stage_1, step_time),
                stage_2=project_stage(
                    stage_data.stage_2,
                    step_time + half_step,
                ),
                stage_3=project_stage(
                    stage_data.stage_3,
                    step_time + half_step,
                ),
                stage_4=project_stage(
                    stage_data.stage_4,
                    step_time + step_timestep,
                ),
            )
            return expand_local_shard_pytree(projected)

        def kernel(
            state_owned: Fci4FieldState,
            phi_guess_owned: jnp.ndarray,
            local_invariants: shifted_mms._ShiftedTorus4FieldInvariantBundle,
            rk_stage_data: shifted_mms._ShiftedTorus4FieldRk4StageData,
            local_control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
            step_time: jax.Array,
            step_timestep: jax.Array,
        ) -> tuple[Fci4FieldState, jnp.ndarray]:
            local_invariants = extract_local_shard_pytree(local_invariants)
            rk_stage_data = extract_local_shard_pytree(rk_stage_data)
            local_control_volume_geometry = extract_local_shard_pytree(
                local_control_volume_geometry
            )
            local_geometry = _local_geometry_for_shard(
                local_control_volume_geometry
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
                control_volume_geometry=local_control_volume_geometry,
            )

            k1, carry_1 = rhs.evaluate_stage(
                state_owned,
                rk_stage_data.stage_1,
                phi_guess_owned,
            )
            stage_1 = state_owned.axpy(k1, scale=0.5 * step_timestep)
            k2, carry_2 = rhs.evaluate_stage(
                stage_1,
                rk_stage_data.stage_2,
                carry_1,
            )
            stage_2 = state_owned.axpy(k2, scale=0.5 * step_timestep)
            k3, carry_3 = rhs.evaluate_stage(
                stage_2,
                rk_stage_data.stage_3,
                carry_2,
            )
            stage_3 = state_owned.axpy(k3, scale=step_timestep)
            k4, carry_4 = rhs.evaluate_stage(
                stage_3,
                rk_stage_data.stage_4,
                carry_3,
            )
            next_state = state_owned.axpy(
                k1.axpy(k2, scale=2.0).axpy(k3, scale=2.0).axpy(k4, scale=1.0),
                scale=step_timestep / 6.0,
            )
            return next_state, carry_4

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
                in_specs=(
                    invariant_spec,
                    control_volume_geometry_spec,
                    P(),
                    P(),
                ),
                out_specs=stage_data_spec,
                check_rep=False,
            )
        )
        step_kernel = jax.jit(
            shard_map(
                kernel,
                mesh=mesh,
                in_specs=(
                    state_spec,
                    field_spec,
                    invariant_spec,
                    stage_data_spec,
                    control_volume_geometry_spec,
                    P(),
                    P(),
                ),
                out_specs=(state_spec, field_spec),
                check_rep=False,
            )
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
                control_volume_geometry,
                jnp.asarray(time_value, dtype=jnp.float64),
                jnp.asarray(dt, dtype=jnp.float64),
            )
            state, phi_guess = step_kernel(
                state,
                phi_guess,
                invariants,
                rk_stage_data,
                control_volume_geometry,
                jnp.asarray(time_value, dtype=jnp.float64),
                jnp.asarray(dt, dtype=jnp.float64),
            )
            jax.block_until_ready(state.density)
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



def _control_volume_operator_category_masks(
    cell_data: dict[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    active = jnp.asarray(cell_data["is_active_owner"], dtype=bool)
    boundary_count = jnp.asarray(
        cell_data["boundary_face_count"],
        dtype=jnp.int32,
    )
    irregular_count = jnp.asarray(
        cell_data["irregular_face_count"],
        dtype=jnp.int32,
    )
    aggregate_target = jnp.asarray(
        cell_data["is_aggregate_target"],
        dtype=bool,
    )
    remote_count = jnp.asarray(
        cell_data["remote_face_count"],
        dtype=jnp.int32,
    )
    reconstruction_count = jnp.asarray(
        cell_data["reconstruction_row_count"],
        dtype=jnp.int32,
    )

    def neighbor_band(mask: jnp.ndarray) -> jnp.ndarray:
        mask = jnp.asarray(mask, dtype=bool)
        result = jnp.zeros_like(mask)
        result = result | jnp.zeros_like(mask).at[1:, :, :].set(
            mask[:-1, :, :]
        )
        result = result | jnp.zeros_like(mask).at[:-1, :, :].set(
            mask[1:, :, :]
        )
        result = result | jnp.roll(mask, 1, axis=1)
        result = result | jnp.roll(mask, -1, axis=1)
        result = result | jnp.roll(mask, 1, axis=2)
        result = result | jnp.roll(mask, -1, axis=2)
        return result

    compact_core = active & (
        (irregular_count > 0)
        | (reconstruction_count > 0)
        | aggregate_target
    )
    dense_compact_d1 = (
        active
        & (~compact_core)
        & neighbor_band(compact_core)
    )
    dense_compact_d2 = (
        active
        & (~compact_core)
        & (~dense_compact_d1)
        & neighbor_band(compact_core | dense_compact_d1)
    )
    dense_far = active & (
        ~(compact_core | dense_compact_d1 | dense_compact_d2)
    )
    radial_index = jnp.arange(active.shape[0], dtype=jnp.int32)[:, None, None]
    radial_lower_owner = active & (radial_index == 0)
    radial_upper_owner = active & (radial_index == active.shape[0] - 1)
    radial_interior = active & (radial_index >= 2) & (
        radial_index < active.shape[0] - 2
    )
    return {
        "all_active": active,
        "bulk": active & (irregular_count == 0) & (~aggregate_target),
        "one_wall": active & (boundary_count == 1),
        "multi_wall": active & (boundary_count >= 2),
        "aggregate_target": active & aggregate_target,
        "remote_interface": active & (remote_count > 0),
        "reconstruction_row": active & (reconstruction_count > 0),
        "retained_cut_cell": (
            active
            & (boundary_count > 0)
            & (~aggregate_target)
        ),
        "dense_compact_d1": dense_compact_d1,
        "dense_compact_d2": dense_compact_d2,
        "dense_far": dense_far,
        "radial_lower_owner": radial_lower_owner,
        "radial_upper_owner": radial_upper_owner,
        "radial_interior_2plus": radial_interior,
    }


def _operator_category_statistics(
    actual: jnp.ndarray,
    expected: jnp.ndarray,
    volume: jnp.ndarray,
    categories: dict[str, jnp.ndarray],
) -> dict[str, tuple[float, float, float, int]]:
    result: dict[str, tuple[float, float, float, int]] = {}
    for category, mask in categories.items():
        count = int(jnp.sum(jnp.asarray(mask, dtype=jnp.int32)))
        if count == 0:
            result[category] = (
                float("nan"),
                float("nan"),
                float("nan"),
                0,
            )
            continue
        l2, linf, relative = _volume_weighted_field_error_statistics(
            actual,
            expected,
            volume,
            mask,
        )
        result[category] = (l2, linf, relative, count)
    return result


def _fit_operator_order(
    resolutions: list[int],
    errors: list[float],
) -> float | None:
    resolution_array = np.asarray(resolutions, dtype=np.float64)
    error_array = np.asarray(errors, dtype=np.float64)
    valid = (
        np.isfinite(resolution_array)
        & np.isfinite(error_array)
        & (resolution_array > 0.0)
        & (error_array > 0.0)
    )
    if int(np.sum(valid)) < 2:
        return None
    slope = np.polyfit(
        np.log(resolution_array[valid]),
        np.log(error_array[valid]),
        1,
    )[0]
    return float(-slope)


def run_shifted_torus_control_volume_operator_convergence(
    *,
    resolutions: list[int],
    shard_counts: tuple[int, int, int],
    halo_width: int,
    rho_star_value: float = shifted_mms.rho_star,
    enable_agglomeration: bool = True,
    minimum_order: float = 1.8,
    check_phi_solve: bool = True,
    debug_operator_failures: bool = False,
) -> dict[str, object]:
    """Run low-memory spatial convergence kernels for the unified CV path."""

    shard_counts = tuple(int(value) for value in shard_counts)
    parameters = _make_parameters(rho_star_value)
    gmres_config = _make_gmres_config(parameters)
    operator_names = (
        "grad_parallel_density",
        "grad_parallel_phi",
        "grad_parallel_v_ion",
        "grad_parallel_v_electron",
        "parallel_density_flux_divergence",
        "poisson_density",
        "poisson_omega",
        "poisson_v_ion",
        "poisson_v_electron",
        "curvature_density",
        "curvature_phi",
        "perp_laplacian_phi",
    )
    records: dict[
        str,
        dict[str, list[tuple[int, float, float]]],
    ] = {}
    phi_residuals: list[tuple[int, float]] = []

    for resolution in resolutions:
        resolution = int(resolution)
        shape = _shape_from_resolution(resolution)
        assert_shape_divisible_by_shards(shape, shard_counts)
        owned_shape = tuple(
            int(size) // int(count)
            for size, count in zip(shape, shard_counts)
        )
        geometry = shifted_mms.build_shifted_torus_4field_geometry(shape)
        print(
            "Preparing shifted_torus control-volume geometry: "
            f"N={resolution}, shape={shape}, shards={shard_counts}",
            flush=True,
        )
        geometry_start = time_module.perf_counter()
        stacked_control_volume_geometry = (
            _build_stacked_embedded_control_volume_geometry(
                global_shape=shape,
                shard_counts=shard_counts,
                halo_width=halo_width,
                enable_merging=enable_agglomeration,
            )
        )
        print(
            "Prepared shifted_torus control-volume geometry: "
            f"N={resolution}, elapsed="
            f"{time_module.perf_counter() - geometry_start:.3f}s",
            flush=True,
        )
        if debug_operator_failures and resolution == int(resolutions[0]):
            _print_shifted_torus_radial_moment_reproduction(
                global_shape=shape,
                owned_shape=owned_shape,
                halo_width=halo_width,
                enable_merging=enable_agglomeration,
            )
        exact_state, exact_phi = _project_global_exact_state_to_control_volumes(
            geometry,
            stacked_control_volume_geometry,
            shard_counts=shard_counts,
            halo_width=halo_width,
            time=0.0,
        )
        cell_data = _assemble_global_control_volume_cell_data(
            shape,
            stacked_control_volume_geometry,
            shard_counts=shard_counts,
        )
        categories = _control_volume_operator_category_masks(cell_data)
        volume = cell_data["aggregate_volume"]
        _print_control_volume_geometry_summary(
            stacked_control_volume_geometry
        )
        print(
            "Starting shifted_torus control-volume operator sweep: "
            f"N={resolution}, shape={shape}, shards={shard_counts}"
        )
        print(
            "  radial boundary contract: x-low=physical Dirichlet, "
            "x-high=physical Dirichlet, axis_regular_x=False"
        )

        domain = build_shifted_torus_local_domain(
            shape,
            halo_width,
            shard_counts,
        )
        topology_filler = TopologyHaloFiller3D(
            rules=(LocalPeriodicTopologyRule3D(),)
        )
        physical_ghost_filler = shifted_mms._build_ghost_filler(halo_width)

        with make_mesh_for_shard_counts(shard_counts) as mesh:
            state_spec = shifted_mms._state_partition_spec()
            field_spec = P(*MESH_AXIS_NAMES)
            state_mesh = shifted_mms._put_state_on_mesh(exact_state, mesh)
            phi_mesh = jax.device_put(
                jnp.asarray(exact_phi, dtype=jnp.float64),
                NamedSharding(mesh, field_spec),
            )
            host_domain = LocalDomain3D(
                shard_spec=domain.shard_spec,
                layout=domain.layout,
                mesh_axis_names=(None, None, None),
            )
            sample_invariants = expand_local_shard_pytree(
                shifted_mms._build_local_4field_invariants(
                    (0, 0, 0),
                    owned_shape=owned_shape,
                    halo_width=halo_width,
                    global_shape=shape,
                    domain=host_domain,
                )
            )
            invariant_spec = local_shard_pytree_partition_spec(
                sample_invariants
            )
            control_volume_spec = local_shard_pytree_partition_spec(
                stacked_control_volume_geometry
            )
            control_volume_sharding = jax.tree_util.tree_map(
                lambda spec: NamedSharding(mesh, spec),
                control_volume_spec,
            )
            control_volume_mesh = jax.device_put(
                stacked_control_volume_geometry,
                control_volume_sharding,
            )

            def invariant_kernel():
                shard_index = tuple(
                    lax.axis_index(name)
                    for name in MESH_AXIS_NAMES
                )
                return expand_local_shard_pytree(
                    shifted_mms._build_local_4field_invariants(
                        shard_index,
                        owned_shape=owned_shape,
                        halo_width=halo_width,
                        global_shape=shape,
                        domain=domain,
                    )
                )

            invariants_mesh = jax.jit(
                shard_map(
                    invariant_kernel,
                    mesh=mesh,
                    in_specs=(),
                    out_specs=invariant_spec,
                    check_rep=False,
                )
            )()

            def local_geometry(
                control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
            ) -> LocalFciGeometry3D:
                shard_index = tuple(
                    lax.axis_index(name)
                    for name in MESH_AXIS_NAMES
                )
                base = build_shifted_torus_local_geometry(
                    owned_shape,
                    halo_width,
                    global_shape=shape,
                    shard_index=shard_index,
                    x_min=shifted_mms.x_min,
                    x_max=shifted_mms.x_max,
                    r0=shifted_mms.r0,
                    alpha_value=shifted_mms.alpha_value,
                    iota=shifted_mms.iota,
                    c_phi=shifted_mms.c_phi,
                    sigma=shifted_mms.sigma,
                )
                return _with_embedded_control_volume_geometry(
                    base,
                    control_volume_geometry,
                )

            def regular_face_bc(
                local_geometry_value: LocalFciGeometry3D,
                stage_time: jax.Array,
                field_name: str,
            ) -> LocalBoundaryFaceBC3D:
                lower, upper = _shifted_torus_regular_radial_face_average(
                    local_geometry_value,
                    stage_time,
                    field_name,
                )
                return (
                    shifted_mms._build_local_radial_dirichlet_face_bc_from_values(
                        lower,
                        upper,
                        domain,
                    )
                )

            def prepare_field(
                values_owned: jnp.ndarray,
                field_name: str,
                local_invariants,
                local_geometry_value: LocalFciGeometry3D,
                control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
                stage_time: jax.Array,
            ):
                face_bc = regular_face_bc(
                    local_geometry_value,
                    stage_time,
                    field_name,
                )
                storage = _expand_control_volume_owner_values(
                    values_owned,
                    control_volume_geometry.cells,
                )
                field_halo = inject_owned_field_to_halo(
                    storage,
                    domain.layout,
                )
                field_halo = LocalHaloClosure3D(
                    physical_ghost_filler=physical_ghost_filler,
                    halo_exchange=HaloExchange3D(),
                    topology_filler=topology_filler,
                )(
                    field_halo,
                    domain,
                    face_bc,
                )
                boundary_bc = _control_volume_exact_boundary_bc(
                    control_volume_geometry,
                    stage_time,
                    field_name,
                )
                polynomial = build_local_control_volume_polynomial_from_field(
                    field_halo,
                    local_geometry_value,
                    domain,
                    StencilBuilderContext(
                        layout=domain.layout,
                        domain=domain,
                    ),
                    control_volume_geometry,
                    boundary_bc,
                    face_bc,
                    halo_exchange=HaloExchange3D(),
                    topology_filler=topology_filler,
                )
                return field_halo, polynomial, boundary_bc, face_bc

            def evaluate_scalar_operator(
                operator_name: str,
                state_owned: Fci4FieldState,
                phi_owned: jnp.ndarray,
                local_invariants,
                control_volume_geometry,
                stage_time: jax.Array,
            ) -> tuple[
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
            ]:
                local_invariants = extract_local_shard_pytree(
                    local_invariants
                )
                control_volume_geometry = extract_local_shard_pytree(
                    control_volume_geometry
                )
                local_geometry_value = local_geometry(
                    control_volume_geometry
                )
                active = control_volume_geometry.cells.is_active_owner
                remote_flux_sum = jnp.asarray(0.0, dtype=jnp.float64)
                remote_flux_abs_sum = jnp.asarray(0.0, dtype=jnp.float64)
                invalid_remote_quadrature = jnp.asarray(
                    0,
                    dtype=jnp.int32,
                )
                invalid_reconstruction_rows = jnp.asarray(
                    0,
                    dtype=jnp.int32,
                )
                reconstruction_target = (
                    control_volume_geometry.reconstruction.target_row_for_cell
                    >= 0
                )

                field_suffixes = {
                    "density": "density",
                    "omega": "omega",
                    "v_ion": "v_ion_parallel",
                    "v_electron": "v_electron_parallel",
                    "phi": "phi",
                }

                if operator_name.startswith("grad_parallel_"):
                    suffix = operator_name.removeprefix(
                        "grad_parallel_"
                    )
                    field_name = field_suffixes[suffix]
                    field_owned = (
                        phi_owned
                        if field_name == "phi"
                        else getattr(state_owned, field_name)
                    )
                    _field_halo, field_poly, _bc, _face_bc = prepare_field(
                        field_owned,
                        field_name,
                        local_invariants,
                        local_geometry_value,
                        control_volume_geometry,
                        stage_time,
                    )
                    actual = local_grad_parallel_op_from_gradient(
                        field_poly.as_cell_gradient(),
                        local_geometry_value,
                        control_volume_geometry=control_volume_geometry,
                    )
                    invalid_reconstruction_rows = jnp.sum(
                        (
                            reconstruction_target
                            & (~field_poly.valid)
                        ).astype(jnp.int32)
                    )
                    reference = (
                        _shifted_torus_operator_reference_at_logical_points(
                            control_volume_geometry.cells.centroid,
                            stage_time,
                            operator_name,
                        )
                    )
                elif operator_name == "parallel_density_flux_divergence":
                    (
                        _density_halo,
                        density_polynomial,
                        _density_bc,
                        _density_face_bc,
                    ) = prepare_field(
                        state_owned.density,
                        "density",
                        local_invariants,
                        local_geometry_value,
                        control_volume_geometry,
                        stage_time,
                    )
                    (
                        _v_electron_halo,
                        v_electron_polynomial,
                        _v_electron_bc,
                        _v_electron_face_bc,
                    ) = prepare_field(
                        state_owned.v_electron_parallel,
                        "v_electron_parallel",
                        local_invariants,
                        local_geometry_value,
                        control_volume_geometry,
                        stage_time,
                    )
                    density_v_electron = local_control_volume_product_average(
                        state_owned.density,
                        state_owned.v_electron_parallel,
                        density_polynomial,
                        v_electron_polynomial,
                        control_volume_geometry.cells,
                    )
                    field_halo, polynomial, boundary_bc, face_bc = (
                        prepare_field(
                            density_v_electron,
                            "density_v_electron",
                            local_invariants,
                            local_geometry_value,
                            control_volume_geometry,
                            stage_time,
                        )
                    )
                    local = build_local_conservative_stencil_from_field(
                        field_halo,
                        local_geometry_value,
                        StencilBuilderContext(
                            layout=domain.layout,
                            domain=domain,
                        ),
                    )
                    actual = local_parallel_flux_div_op(
                        local,
                        local_geometry_value,
                        domain,
                        face_bc=face_bc,
                        control_volume_geometry=control_volume_geometry,
                        boundary_bc=boundary_bc,
                        field_reconstruction=polynomial,
                    )
                    invalid_reconstruction_rows = jnp.sum(
                        (
                            reconstruction_target
                            & (
                                (~density_polynomial.valid)
                                | (~v_electron_polynomial.valid)
                                | (~polynomial.valid)
                            )
                        ).astype(jnp.int32)
                    )
                    irregular_flux = (
                        _local_control_volume_irregular_parallel_flux(
                            jnp.asarray(local.x.center, dtype=jnp.float64),
                            polynomial,
                            control_volume_geometry,
                            boundary_bc,
                            b_floor=1.0e-30,
                        )
                    )
                    remote_row = (
                        control_volume_geometry.irregular_faces.active
                        & control_volume_geometry.irregular_faces.has_remote_owner
                    )
                    remote_flux_sum = jnp.sum(
                        jnp.where(remote_row, irregular_flux, 0.0)
                    )
                    remote_flux_abs_sum = jnp.sum(
                        jnp.where(
                            remote_row,
                            jnp.abs(irregular_flux),
                            0.0,
                        )
                    )
                    remote_quadrature = (
                        remote_row[:, None, None]
                        & control_volume_geometry.irregular_faces.quadrature_active
                    )
                    invalid_remote_quadrature = jnp.sum(
                        (
                            remote_quadrature
                            & (~polynomial.remote_face_valid)
                        ).astype(jnp.int32)
                    )
                    for mesh_axis_name in MESH_AXIS_NAMES:
                        remote_flux_sum = lax.psum(
                            remote_flux_sum,
                            mesh_axis_name,
                        )
                        remote_flux_abs_sum = lax.psum(
                            remote_flux_abs_sum,
                            mesh_axis_name,
                        )
                        invalid_remote_quadrature = lax.psum(
                            invalid_remote_quadrature,
                            mesh_axis_name,
                        )
                    reference = _integrate_local_scalar_over_fluid(
                        local_geometry_value,
                        control_volume_geometry,
                        lambda points: (
                            _shifted_torus_operator_reference_at_logical_points(
                                points,
                                stage_time,
                                "parallel_density_flux_divergence",
                            )
                        ),
                    )
                elif operator_name.startswith("poisson_"):
                    suffix = operator_name.removeprefix("poisson_")
                    field_name = field_suffixes[suffix]
                    field_owned = getattr(state_owned, field_name)
                    _phi_halo, phi_poly, _bc, _face_bc = prepare_field(
                        phi_owned,
                        "phi",
                        local_invariants,
                        local_geometry_value,
                        control_volume_geometry,
                        stage_time,
                    )
                    _v_electron_halo, v_electron_poly, _bc, _face_bc = (
                        prepare_field(
                            field_owned,
                            field_name,
                            local_invariants,
                            local_geometry_value,
                            control_volume_geometry,
                            stage_time,
                        )
                    )
                    actual = local_poisson_bracket_op_from_gradients(
                        phi_poly.as_cell_gradient(),
                        v_electron_poly.as_cell_gradient(),
                        local_geometry_value,
                        control_volume_geometry=control_volume_geometry,
                    )
                    invalid_reconstruction_rows = jnp.sum(
                        (
                            reconstruction_target
                            & (
                                (~phi_poly.valid)
                                | (~v_electron_poly.valid)
                            )
                        ).astype(jnp.int32)
                    )
                    reference = _integrate_local_scalar_over_fluid(
                        local_geometry_value,
                        control_volume_geometry,
                        lambda points: (
                            _shifted_torus_operator_reference_at_logical_points(
                                points,
                                stage_time,
                                operator_name,
                            )
                        ),
                    )
                elif operator_name.startswith("curvature_"):
                    suffix = operator_name.removeprefix("curvature_")
                    field_name = field_suffixes[suffix]
                    field_owned = (
                        phi_owned
                        if field_name == "phi"
                        else getattr(state_owned, field_name)
                    )
                    _field_halo, field_poly, _bc, _face_bc = (
                        prepare_field(
                            field_owned,
                            field_name,
                            local_invariants,
                            local_geometry_value,
                            control_volume_geometry,
                            stage_time,
                        )
                    )
                    actual = local_curvature_op_from_gradient(
                        field_poly.as_cell_gradient(),
                        local_geometry_value,
                        curvature_coefficients=(
                            local_invariants.curvature_coefficients_owned
                        ),
                        control_volume_geometry=control_volume_geometry,
                    )
                    invalid_reconstruction_rows = jnp.sum(
                        (
                            reconstruction_target
                            & (~field_poly.valid)
                        ).astype(jnp.int32)
                    )
                    reference = _integrate_local_scalar_over_fluid(
                        local_geometry_value,
                        control_volume_geometry,
                        lambda points: (
                            _shifted_torus_operator_reference_at_logical_points(
                                points,
                                stage_time,
                                operator_name,
                            )
                        ),
                    )
                elif operator_name == "perp_laplacian_phi":
                    phi_halo, phi_poly, boundary_bc, face_bc = (
                        prepare_field(
                            phi_owned,
                            "phi",
                            local_invariants,
                            local_geometry_value,
                            control_volume_geometry,
                            stage_time,
                        )
                    )
                    local = build_local_conservative_stencil_from_field(
                        phi_halo,
                        local_geometry_value,
                        StencilBuilderContext(
                            layout=domain.layout,
                            domain=domain,
                        ),
                    )
                    actual = local_perp_laplacian_conservative_op(
                        local,
                        local_geometry_value,
                        domain,
                        face_projectors=(
                            local_invariants.face_projector_x,
                            local_invariants.face_projector_y,
                            local_invariants.face_projector_z,
                        ),
                        face_bc=face_bc,
                        control_volume_geometry=control_volume_geometry,
                        boundary_bc=boundary_bc,
                        field_reconstruction=phi_poly,
                    )
                    invalid_reconstruction_rows = jnp.sum(
                        (
                            reconstruction_target
                            & (~phi_poly.valid)
                        ).astype(jnp.int32)
                    )
                    reference = state_owned.omega
                else:
                    raise ValueError(
                        f"unsupported operator kernel {operator_name!r}"
                    )
                for mesh_axis_name in MESH_AXIS_NAMES:
                    invalid_reconstruction_rows = lax.psum(
                        invalid_reconstruction_rows,
                        mesh_axis_name,
                    )
                return (
                    jnp.where(active, actual, 0.0),
                    jnp.where(active, reference, 0.0),
                    remote_flux_sum,
                    remote_flux_abs_sum,
                    invalid_remote_quadrature,
                    invalid_reconstruction_rows,
                )

            def make_scalar_kernel(operator_name: str):
                def kernel(
                    state_owned,
                    phi_owned,
                    local_invariants,
                    control_volume_geometry,
                    stage_time,
                ):
                    return evaluate_scalar_operator(
                        operator_name,
                        state_owned,
                        phi_owned,
                        local_invariants,
                        control_volume_geometry,
                        stage_time,
                    )

                return kernel

            for operator_name in operator_names:
                compiled = jax.jit(
                    shard_map(
                        make_scalar_kernel(operator_name),
                        mesh=mesh,
                        in_specs=(
                            state_spec,
                            field_spec,
                            invariant_spec,
                            control_volume_spec,
                            P(),
                        ),
                        out_specs=(
                            field_spec,
                            field_spec,
                            P(),
                            P(),
                            P(),
                            P(),
                        ),
                        check_rep=False,
                    )
                )
                start = time_module.perf_counter()
                (
                    actual_mesh,
                    reference_mesh,
                    remote_flux_sum,
                    remote_flux_abs_sum,
                    invalid_remote_quadrature,
                    invalid_reconstruction_rows,
                ) = compiled(
                    state_mesh,
                    phi_mesh,
                    invariants_mesh,
                    control_volume_mesh,
                    jnp.asarray(0.0, dtype=jnp.float64),
                )
                jax.block_until_ready(actual_mesh)
                elapsed = time_module.perf_counter() - start
                actual = jnp.asarray(jax.device_get(actual_mesh))
                reference = jnp.asarray(jax.device_get(reference_mesh))
                statistics = _operator_category_statistics(
                    actual,
                    reference,
                    volume,
                    categories,
                )
                print(
                    f"N={resolution} operator={operator_name} "
                    f"compile+run={elapsed:.3f}s "
                    "invalid_reconstruction_rows="
                    f"{int(np.asarray(jax.device_get(invalid_reconstruction_rows)))}"
                )
                if int(
                    np.asarray(jax.device_get(invalid_reconstruction_rows))
                ):
                    raise AssertionError(
                        f"{operator_name} produced invalid active "
                        "quadratic reconstruction rows"
                    )
                if operator_name == "parallel_density_flux_divergence":
                    remote_flux_sum_value = float(
                        np.asarray(jax.device_get(remote_flux_sum))
                    )
                    remote_flux_abs_sum_value = float(
                        np.asarray(jax.device_get(remote_flux_abs_sum))
                    )
                    invalid_remote_quadrature_value = int(
                        np.asarray(
                            jax.device_get(invalid_remote_quadrature)
                        )
                    )
                    print(
                        "  mirrored_remote_flux signed_sum="
                        f"{remote_flux_sum_value:.6e} "
                        f"abs_sum={remote_flux_abs_sum_value:.6e} "
                        "relative_imbalance="
                        f"{abs(remote_flux_sum_value) / max(remote_flux_abs_sum_value, 1.0e-30):.6e} "
                        "invalid_quadrature="
                        f"{invalid_remote_quadrature_value}"
                    )
                    remote_relative_imbalance = (
                        abs(remote_flux_sum_value)
                        / max(remote_flux_abs_sum_value, 1.0e-30)
                    )
                    if invalid_remote_quadrature_value:
                        raise AssertionError(
                            "mirrored remote interfaces contain invalid "
                            f"quadrature samples: {invalid_remote_quadrature_value}"
                        )
                    if (
                        remote_flux_abs_sum_value > 1.0e-14
                        and remote_relative_imbalance > 1.0e-12
                    ):
                        raise AssertionError(
                            "mirrored remote interface fluxes do not cancel: "
                            f"relative imbalance={remote_relative_imbalance:.6e}"
                        )
                for category, (
                    l2,
                    linf,
                    relative,
                    count,
                ) in statistics.items():
                    print(
                        f"  {category:18s} count={count:8d} "
                        f"volume_L2={l2:.6e} Linf={linf:.6e} "
                        f"rel_L2={relative:.6e}"
                    )
                    records.setdefault(operator_name, {}).setdefault(
                        category,
                        [],
                    ).append((resolution, l2, linf))
                active_mask = np.asarray(
                    cell_data["is_active_owner"],
                    dtype=bool,
                )
                absolute_error = np.where(
                    active_mask,
                    np.abs(
                        np.asarray(actual, dtype=np.float64)
                        - np.asarray(reference, dtype=np.float64)
                    ),
                    -np.inf,
                )
                top_flat = int(np.argmax(absolute_error))
                top_index = tuple(
                    int(value)
                    for value in np.unravel_index(top_flat, shape)
                )
                top_is_compact = (
                    int(
                        np.asarray(
                            cell_data["irregular_face_count"]
                        )[top_index]
                    )
                    > 0
                    or int(
                        np.asarray(
                            cell_data["reconstruction_row_count"]
                        )[top_index]
                    )
                    > 0
                    or bool(
                        np.asarray(
                            cell_data["is_aggregate_target"]
                        )[top_index]
                    )
                )
                if top_is_compact:
                    top_compact_distance = 0
                elif bool(
                    np.asarray(categories["dense_compact_d1"])[top_index]
                ):
                    top_compact_distance = 1
                elif bool(
                    np.asarray(categories["dense_compact_d2"])[top_index]
                ):
                    top_compact_distance = 2
                else:
                    top_compact_distance = 3
                print(
                    "  top_error index={} error={:.6e} actual={:.6e} "
                    "reference={:.6e} regular_physical_boundary={} "
                    "embedded_cutwall_faces={} irregular_faces={} "
                    "remote_faces={} reconstruction_rows={} aggregate={} "
                    "compact_distance={}".format(
                        top_index,
                        float(absolute_error[top_index]),
                        float(np.asarray(actual)[top_index]),
                        float(np.asarray(reference)[top_index]),
                        bool(
                            top_index[0] == 0
                            or top_index[0] == shape[0] - 1
                        ),
                        int(np.asarray(cell_data["boundary_face_count"])[top_index]),
                        int(np.asarray(cell_data["irregular_face_count"])[top_index]),
                        int(np.asarray(cell_data["remote_face_count"])[top_index]),
                        int(
                            np.asarray(
                                cell_data["reconstruction_row_count"]
                            )[top_index]
                        ),
                        bool(
                            np.asarray(
                                cell_data["is_aggregate_target"]
                            )[top_index]
                        ),
                        top_compact_distance,
                    )
                )
                if (
                    debug_operator_failures
                    and operator_name == "poisson_omega"
                    and top_index[0] == 0
                ):
                    target_shard = tuple(
                        int(top_index[axis]) // int(owned_shape[axis])
                        for axis in range(3)
                    )
                    target_local = tuple(
                        int(top_index[axis]) % int(owned_shape[axis])
                        for axis in range(3)
                    )

                    def radial_owner_gradient_diagnostic_kernel(
                        state_owned,
                        phi_owned,
                        local_invariants,
                        control_volume_geometry,
                        stage_time,
                    ):
                        local_invariants = extract_local_shard_pytree(
                            local_invariants
                        )
                        control_volume_geometry = extract_local_shard_pytree(
                            control_volume_geometry
                        )
                        local_geometry_value = local_geometry(
                            control_volume_geometry
                        )
                        phi_halo, phi_poly, _phi_bc, phi_face_bc = prepare_field(
                            phi_owned,
                            "phi",
                            local_invariants,
                            local_geometry_value,
                            control_volume_geometry,
                            stage_time,
                        )
                        omega_halo, omega_poly, _omega_bc, omega_face_bc = prepare_field(
                            state_owned.omega,
                            "omega",
                            local_invariants,
                            local_geometry_value,
                            control_volume_geometry,
                            stage_time,
                        )
                        context = StencilBuilderContext(
                            layout=domain.layout,
                            domain=domain,
                        )
                        phi_stencil = build_local_stencil_from_field(
                            phi_halo,
                            local_geometry_value,
                            context,
                        )
                        omega_stencil = build_local_stencil_from_field(
                            omega_halo,
                            local_geometry_value,
                            context,
                        )
                        phi_baseline = jnp.stack(
                            (
                                _take_stencil_finite_difference(phi_stencil.x),
                                _take_stencil_finite_difference(phi_stencil.y),
                                _take_stencil_finite_difference(phi_stencil.z),
                            ),
                            axis=-1,
                        )
                        omega_baseline = jnp.stack(
                            (
                                _take_stencil_finite_difference(omega_stencil.x),
                                _take_stencil_finite_difference(omega_stencil.y),
                                _take_stencil_finite_difference(omega_stencil.z),
                            ),
                            axis=-1,
                        )
                        local_i, local_j, local_k = target_local
                        cells = control_volume_geometry.cells
                        centroid = cells.centroid[local_i, local_j, local_k]
                        phi_exact_value, phi_exact_gradient = (
                            _shifted_torus_exact_field_and_gradient_at_logical_points(
                                centroid,
                                stage_time,
                                "phi",
                            )
                        )
                        omega_exact_value, omega_exact_gradient = (
                            _shifted_torus_exact_field_and_gradient_at_logical_points(
                                centroid,
                                stage_time,
                                "omega",
                            )
                        )
                        del phi_exact_value, omega_exact_value
                        closure = control_volume_geometry.regular_boundary_closure
                        face_weights, owner_weights, closure_valid = closure.axis_payload(0)
                        phi_owned_values = phi_halo[
                            local_geometry_value.layout.owned_slices_cell
                        ]
                        omega_owned_values = omega_halo[
                            local_geometry_value.layout.owned_slices_cell
                        ]
                        phi_samples = phi_owned_values[:3, local_j, local_k]
                        omega_samples = omega_owned_values[:3, local_j, local_k]
                        phi_face_value = phi_face_bc.value_x[0, local_j, local_k]
                        omega_face_value = omega_face_bc.value_x[0, local_j, local_k]
                        phi_owner_weights = owner_weights[0, local_j, local_k]
                        omega_owner_weights = owner_weights[0, local_j, local_k]
                        phi_face_weights = face_weights[0, local_j, local_k]
                        omega_face_weights = face_weights[0, local_j, local_k]
                        phi_owner_derivative = (
                            phi_owner_weights[0] * phi_face_value
                            + jnp.dot(phi_owner_weights[1:], phi_samples)
                        )
                        omega_owner_derivative = (
                            omega_owner_weights[0] * omega_face_value
                            + jnp.dot(omega_owner_weights[1:], omega_samples)
                        )
                        phi_face_derivative = (
                            phi_face_weights[0] * phi_face_value
                            + jnp.dot(phi_face_weights[1:], phi_samples)
                        )
                        omega_face_derivative = (
                            omega_face_weights[0] * omega_face_value
                            + jnp.dot(omega_face_weights[1:], omega_samples)
                        )
                        theta_lower = local_geometry_value.grid.y.faces_owned[local_j]
                        theta_upper = local_geometry_value.grid.y.faces_owned[local_j + 1]
                        zeta_lower = local_geometry_value.grid.z.faces_owned[local_k]
                        zeta_upper = local_geometry_value.grid.z.faces_owned[local_k + 1]

                        def patch_average(x, field_name):
                            numerator = jnp.asarray(0.0, dtype=jnp.float64)
                            denominator = jnp.asarray(0.0, dtype=jnp.float64)
                            for node_y, weight_y in zip(_GAUSS3_NODES, _GAUSS3_WEIGHTS):
                                theta = 0.5 * (theta_lower + theta_upper) + 0.5 * (
                                    theta_upper - theta_lower
                                ) * float(node_y)
                                for node_z, weight_z in zip(_GAUSS3_NODES, _GAUSS3_WEIGHTS):
                                    zeta = 0.5 * (zeta_lower + zeta_upper) + 0.5 * (
                                        zeta_upper - zeta_lower
                                    ) * float(node_z)
                                    point = jnp.stack((x, theta, zeta))
                                    measure = (
                                        float(weight_y)
                                        * float(weight_z)
                                        * 0.25
                                        * (theta_upper - theta_lower)
                                        * (zeta_upper - zeta_lower)
                                        * _shifted_torus_metric_payload_jax(point)[0]
                                    )
                                    numerator = numerator + measure * (
                                        _shifted_torus_exact_field_at_logical_points(
                                            point,
                                            stage_time,
                                            field_name,
                                        )
                                    )
                                    denominator = denominator + measure
                            return numerator / jnp.maximum(denominator, 1.0e-30)

                        phi_patch_derivative = jax.jacfwd(
                            lambda x: patch_average(x, "phi")
                        )(centroid[0])
                        omega_patch_derivative = jax.jacfwd(
                            lambda x: patch_average(x, "omega")
                        )(centroid[0])
                        g_cov = control_volume_geometry.centroid_g_cov[
                            local_i, local_j, local_k
                        ]
                        b_contra = (
                            control_volume_geometry.centroid_B_contra[
                                local_i, local_j, local_k
                            ]
                            / jnp.maximum(
                                control_volume_geometry.centroid_Bmag[
                                    local_i, local_j, local_k
                                ],
                                1.0e-30,
                            )
                        )
                        b_cov = jnp.einsum("ij,j->i", g_cov, b_contra)
                        J = control_volume_geometry.centroid_J[
                            local_i, local_j, local_k
                        ]

                        def bracket(phi_gradient, omega_gradient):
                            cross = jnp.cross(phi_gradient, omega_gradient)
                            return jnp.dot(b_cov, cross) / jnp.maximum(J, 1.0e-30), cross

                        bracket_baseline, cross_baseline = bracket(
                            phi_baseline[local_i, local_j, local_k],
                            omega_baseline[local_i, local_j, local_k],
                        )
                        bracket_reconstructed, cross_reconstructed = bracket(
                            phi_poly.gradient[local_i, local_j, local_k],
                            omega_poly.gradient[local_i, local_j, local_k],
                        )
                        bracket_exact_phi, _ = bracket(
                            phi_exact_gradient,
                            omega_poly.gradient[local_i, local_j, local_k],
                        )
                        bracket_exact_omega, _ = bracket(
                            phi_poly.gradient[local_i, local_j, local_k],
                            omega_exact_gradient,
                        )
                        bracket_exact, cross_exact = bracket(
                            phi_exact_gradient,
                            omega_exact_gradient,
                        )
                        owns_target = jnp.asarray(True)
                        for axis, mesh_axis_name in enumerate(MESH_AXIS_NAMES):
                            owns_target = owns_target & (
                                lax.axis_index(mesh_axis_name)
                                == int(target_shard[axis])
                            )

                        def reduce_target(value):
                            value = jnp.where(
                                owns_target,
                                value,
                                jnp.zeros_like(value),
                            )
                            for mesh_axis_name in MESH_AXIS_NAMES:
                                value = lax.psum(value, mesh_axis_name)
                            return value

                        return (
                            reduce_target(centroid),
                            reduce_target(phi_baseline[local_i, local_j, local_k]),
                            reduce_target(omega_baseline[local_i, local_j, local_k]),
                            reduce_target(phi_poly.gradient[local_i, local_j, local_k]),
                            reduce_target(omega_poly.gradient[local_i, local_j, local_k]),
                            reduce_target(phi_exact_gradient),
                            reduce_target(omega_exact_gradient),
                            reduce_target(phi_samples),
                            reduce_target(omega_samples),
                            reduce_target(phi_face_value),
                            reduce_target(omega_face_value),
                            reduce_target(phi_owner_weights),
                            reduce_target(omega_owner_weights),
                            reduce_target(phi_face_weights),
                            reduce_target(omega_face_weights),
                            reduce_target(jnp.stack((
                                phi_owner_derivative,
                                omega_owner_derivative,
                                phi_face_derivative,
                                omega_face_derivative,
                                phi_patch_derivative,
                                omega_patch_derivative,
                                phi_exact_gradient[0],
                                omega_exact_gradient[0],
                                closure_valid[0, local_j, local_k].astype(jnp.float64),
                            ))),
                            reduce_target(jnp.stack((
                                bracket_baseline,
                                bracket_reconstructed,
                                bracket_exact_phi,
                                bracket_exact_omega,
                                bracket_exact,
                            ))),
                            reduce_target(cross_baseline),
                            reduce_target(cross_reconstructed),
                            reduce_target(cross_exact),
                            reduce_target(b_cov),
                        )

                    radial_diagnostic = jax.jit(
                        shard_map(
                            radial_owner_gradient_diagnostic_kernel,
                            mesh=mesh,
                            in_specs=(
                                state_spec,
                                field_spec,
                                invariant_spec,
                                control_volume_spec,
                                P(),
                            ),
                            out_specs=(P(),) * 21,
                            check_rep=False,
                        )
                    )(
                        state_mesh,
                        phi_mesh,
                        invariants_mesh,
                        control_volume_mesh,
                        jnp.asarray(0.0, dtype=jnp.float64),
                    )
                    radial_diagnostic = tuple(
                        np.asarray(jax.device_get(value))
                        for value in radial_diagnostic
                    )
                    (
                        radial_centroid,
                        phi_baseline,
                        omega_baseline,
                        phi_reconstructed,
                        omega_reconstructed,
                        phi_exact,
                        omega_exact,
                        phi_samples,
                        omega_samples,
                        phi_face_value,
                        omega_face_value,
                        phi_owner_weights,
                        omega_owner_weights,
                        phi_face_weights,
                        omega_face_weights,
                        radial_derivatives,
                        bracket_variants,
                        cross_baseline,
                        cross_reconstructed,
                        cross_exact,
                        b_cov,
                    ) = radial_diagnostic
                    print("  radial_owner_gradient_diagnostic")
                    print(
                        "    centroid={} b_cov={} closure_valid={}".format(
                            np.array2string(radial_centroid, precision=8),
                            np.array2string(b_cov, precision=8),
                            bool(radial_derivatives[8]),
                        )
                    )
                    for field_name, baseline, reconstructed, exact, samples, face_value, owner_weights, face_weights in (
                        ("phi", phi_baseline, phi_reconstructed, phi_exact, phi_samples, phi_face_value, phi_owner_weights, phi_face_weights),
                        ("omega", omega_baseline, omega_reconstructed, omega_exact, omega_samples, omega_face_value, omega_owner_weights, omega_face_weights),
                    ):
                        print(
                            "    {} gradient baseline={} reconstructed={} exact={}".format(
                                field_name,
                                np.array2string(baseline, precision=8),
                                np.array2string(reconstructed, precision=8),
                                np.array2string(exact, precision=8),
                            )
                        )
                        print(
                            "      face_average={:.8e} inward_averages={} owner_weights={} face_weights={}".format(
                                float(face_value),
                                np.array2string(samples, precision=8),
                                np.array2string(owner_weights, precision=8),
                                np.array2string(face_weights, precision=8),
                            )
                        )
                    print(
                        "    radial_derivatives phi(owner/face/patch/exact)={:.8e}/{:.8e}/{:.8e}/{:.8e} "
                        "omega(owner/face/patch/exact)={:.8e}/{:.8e}/{:.8e}/{:.8e}".format(
                            float(radial_derivatives[0]),
                            float(radial_derivatives[2]),
                            float(radial_derivatives[4]),
                            float(radial_derivatives[6]),
                            float(radial_derivatives[1]),
                            float(radial_derivatives[3]),
                            float(radial_derivatives[5]),
                            float(radial_derivatives[7]),
                        )
                    )
                    print(
                        "    poisson variants baseline={:.8e} reconstructed={:.8e} "
                        "exact_phi={:.8e} exact_omega={:.8e} exact_both={:.8e}".format(
                            *[float(value) for value in bracket_variants]
                        )
                    )
                    print(
                        "    poisson cross baseline={} reconstructed={} exact={}".format(
                            np.array2string(cross_baseline, precision=8),
                            np.array2string(cross_reconstructed, precision=8),
                            np.array2string(cross_exact, precision=8),
                        )
                    )
                if (
                    debug_operator_failures
                    and operator_name == "perp_laplacian_phi"
                ):
                    radial_error = np.where(
                        np.asarray(categories["radial_lower_owner"], dtype=bool),
                        absolute_error,
                        -np.inf,
                    )
                    radial_flat = int(np.argmax(radial_error))
                    radial_index = tuple(
                        int(value)
                        for value in np.unravel_index(radial_flat, shape)
                    )
                    radial_shard = tuple(
                        int(radial_index[axis]) // int(owned_shape[axis])
                        for axis in range(3)
                    )
                    radial_local = tuple(
                        int(radial_index[axis]) % int(owned_shape[axis])
                        for axis in range(3)
                    )

                    def radial_projected_flux_diagnostic_kernel(
                        phi_owned,
                        local_invariants,
                        control_volume_geometry,
                        stage_time,
                    ):
                        local_invariants = extract_local_shard_pytree(
                            local_invariants
                        )
                        control_volume_geometry = extract_local_shard_pytree(
                            control_volume_geometry
                        )
                        local_geometry_value = local_geometry(
                            control_volume_geometry
                        )
                        phi_halo, _poly, _boundary_bc, face_bc = prepare_field(
                            phi_owned,
                            "phi",
                            local_invariants,
                            local_geometry_value,
                            control_volume_geometry,
                            stage_time,
                        )
                        local = build_local_conservative_stencil_from_field(
                            phi_halo,
                            local_geometry_value,
                            StencilBuilderContext(
                                layout=domain.layout,
                                domain=domain,
                            ),
                        )
                        cv_flux = build_local_perp_laplacian_stencil(
                            local,
                            local_geometry_value,
                            domain,
                            face_projectors=(
                                local_invariants.face_projector_x,
                                local_invariants.face_projector_y,
                                local_invariants.face_projector_z,
                            ),
                            face_bc=face_bc,
                            regular_face_geometry=(
                                control_volume_geometry.regular_faces
                            ),
                            regular_boundary_closure=(
                                control_volume_geometry.regular_boundary_closure
                            ),
                        )
                        local_i, local_j, local_k = radial_local
                        face_gradient = local.face_grad.x[0, local_j, local_k]
                        face_projector = local_invariants.face_projector_x[
                            0, local_j, local_k, 0
                        ]
                        x_metric = local_geometry_value.face_metric.x
                        face_J = x_metric.J_owned[0, local_j, local_k]
                        regular_faces = control_volume_geometry.regular_faces
                        open_measure = (
                            local_geometry_value.spacing.dy_owned[
                                local_i, local_j, local_k
                            ]
                            * local_geometry_value.spacing.dz_owned[
                                local_i, local_j, local_k
                            ]
                            * regular_faces.x_area[0, local_j, local_k]
                            * regular_faces.x_area_fraction[0, local_j, local_k]
                        )
                        numerical_density = cv_flux.regular_flux.x[
                            0, local_j, local_k
                        ]
                        numerical_integrated = numerical_density * open_measure
                        x_face = local_geometry_value.grid.x.faces_owned[0]
                        theta_center = local_geometry_value.grid.y.centers_owned[local_j]
                        zeta_center = local_geometry_value.grid.z.centers_owned[local_k]
                        dy = local_geometry_value.spacing.dy_owned[
                            local_i, local_j, local_k
                        ]
                        dz = local_geometry_value.spacing.dz_owned[
                            local_i, local_j, local_k
                        ]
                        signs = jnp.asarray(
                            ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)),
                            dtype=jnp.float64,
                        )
                        points = jnp.stack(
                            (
                                jnp.full((4,), x_face),
                                theta_center + signs[:, 0] * dy / (2.0 * jnp.sqrt(3.0)),
                                zeta_center + signs[:, 1] * dz / (2.0 * jnp.sqrt(3.0)),
                            ),
                            axis=-1,
                        )
                        (
                            exact_J,
                            _g_contra,
                            _g_cov,
                            _B_contra,
                            _Bmag,
                            exact_projector,
                        ) = _shifted_torus_metric_payload_jax(points)
                        _exact_value, exact_gradient_q = (
                            _shifted_torus_exact_field_and_gradient_at_logical_points(
                                points,
                                stage_time,
                                "phi",
                            )
                        )
                        exact_density_q = exact_J * jnp.einsum(
                            "qi,qi->q",
                            exact_projector[:, 0, :],
                            exact_gradient_q,
                        )
                        exact_density = jnp.mean(exact_density_q)
                        exact_integrated = exact_density * open_measure
                        center_point = jnp.asarray(
                            (x_face, theta_center, zeta_center),
                            dtype=jnp.float64,
                        )
                        _center_value, exact_center_gradient = (
                            _shifted_torus_exact_field_and_gradient_at_logical_points(
                                center_point,
                                stage_time,
                                "phi",
                            )
                        )
                        owns_target = jnp.asarray(True)
                        for axis, mesh_axis_name in enumerate(MESH_AXIS_NAMES):
                            owns_target = owns_target & (
                                lax.axis_index(mesh_axis_name)
                                == int(radial_shard[axis])
                            )

                        def reduce_target(value):
                            value = jnp.where(
                                owns_target,
                                value,
                                jnp.zeros_like(value),
                            )
                            for mesh_axis_name in MESH_AXIS_NAMES:
                                value = lax.psum(value, mesh_axis_name)
                            return value

                        return (
                            reduce_target(face_gradient),
                            reduce_target(exact_center_gradient),
                            reduce_target(face_projector),
                            reduce_target(face_projector * face_gradient),
                            reduce_target(face_J),
                            reduce_target(numerical_density),
                            reduce_target(exact_density),
                            reduce_target(open_measure),
                            reduce_target(numerical_integrated),
                            reduce_target(exact_integrated),
                            reduce_target(
                                control_volume_geometry.cells.aggregate_volume[
                                    local_i, local_j, local_k
                                ]
                            ),
                        )

                    radial_flux = jax.jit(
                        shard_map(
                            radial_projected_flux_diagnostic_kernel,
                            mesh=mesh,
                            in_specs=(
                                field_spec,
                                invariant_spec,
                                control_volume_spec,
                                P(),
                            ),
                            out_specs=(P(),) * 11,
                            check_rep=False,
                        )
                    )(
                        phi_mesh,
                        invariants_mesh,
                        control_volume_mesh,
                        jnp.asarray(0.0, dtype=jnp.float64),
                    )
                    radial_flux = tuple(
                        np.asarray(jax.device_get(value))
                        for value in radial_flux
                    )
                    (
                        radial_face_gradient,
                        radial_exact_gradient,
                        radial_projector,
                        radial_component_flux,
                        radial_J,
                        radial_numerical_density,
                        radial_exact_density,
                        radial_measure,
                        radial_numerical_integrated,
                        radial_exact_integrated,
                        radial_volume,
                    ) = radial_flux
                    print(
                        "  radial_projected_face_diagnostic index={} operator_error={:.6e}".format(
                            radial_index,
                            float(radial_error[radial_index]),
                        )
                    )
                    print(
                        "    face_gradient={} exact_center_gradient={} projector={} "
                        "projected_components={} J={:.8e}".format(
                            np.array2string(radial_face_gradient, precision=8),
                            np.array2string(radial_exact_gradient, precision=8),
                            np.array2string(radial_projector, precision=8),
                            np.array2string(radial_component_flux, precision=8),
                            float(radial_J),
                        )
                    )
                    print(
                        "    x-low flux density numerical/exact={:.8e}/{:.8e} "
                        "integrated={:.8e}/{:.8e} measure={:.8e} volume={:.8e} "
                        "operator_delta={:.8e}".format(
                            float(radial_numerical_density),
                            float(radial_exact_density),
                            float(radial_numerical_integrated),
                            float(radial_exact_integrated),
                            float(radial_measure),
                            float(radial_volume),
                            float(
                                (radial_numerical_integrated - radial_exact_integrated)
                                / max(float(radial_volume), 1.0e-30)
                            ),
                        )
                    )
                if operator_name in (
                    "parallel_density_flux_divergence",
                    "perp_laplacian_phi",
                ):
                    target_shard = tuple(
                        int(top_index[axis]) // int(owned_shape[axis])
                        for axis in range(3)
                    )
                    target_local = tuple(
                        int(top_index[axis]) % int(owned_shape[axis])
                        for axis in range(3)
                    )
                    max_reported_rows = 32

                    def top_cell_face_diagnostic_kernel(
                        state_owned,
                        phi_owned,
                        local_invariants,
                        control_volume_geometry,
                        stage_time,
                    ):
                        local_invariants = extract_local_shard_pytree(
                            local_invariants
                        )
                        control_volume_geometry = (
                            extract_local_shard_pytree(
                                control_volume_geometry
                            )
                        )
                        local_geometry_value = local_geometry(
                            control_volume_geometry
                        )
                        cells = control_volume_geometry.cells
                        faces = control_volume_geometry.irregular_faces

                        if (
                            operator_name
                            == "parallel_density_flux_divergence"
                        ):
                            (
                                _density_halo,
                                density_polynomial,
                                _density_bc,
                                _density_face_bc,
                            ) = prepare_field(
                                state_owned.density,
                                "density",
                                local_invariants,
                                local_geometry_value,
                                control_volume_geometry,
                                stage_time,
                            )
                            (
                                _electron_halo,
                                electron_polynomial,
                                _electron_bc,
                                _electron_face_bc,
                            ) = prepare_field(
                                state_owned.v_electron_parallel,
                                "v_electron_parallel",
                                local_invariants,
                                local_geometry_value,
                                control_volume_geometry,
                                stage_time,
                            )
                            values_owned = (
                                local_control_volume_product_average(
                                    state_owned.density,
                                    state_owned.v_electron_parallel,
                                    density_polynomial,
                                    electron_polynomial,
                                    cells,
                                )
                            )
                            (
                                field_halo,
                                polynomial,
                                boundary_bc,
                                face_bc,
                            ) = prepare_field(
                                values_owned,
                                "density_v_electron",
                                local_invariants,
                                local_geometry_value,
                                control_volume_geometry,
                                stage_time,
                            )
                            exact_field_name = "density_v_electron"
                        else:
                            values_owned = phi_owned
                            (
                                field_halo,
                                polynomial,
                                boundary_bc,
                                face_bc,
                            ) = prepare_field(
                                values_owned,
                                "phi",
                                local_invariants,
                                local_geometry_value,
                                control_volume_geometry,
                                stage_time,
                            )
                            exact_field_name = "phi"

                        local = build_local_conservative_stencil_from_field(
                            field_halo,
                            local_geometry_value,
                            StencilBuilderContext(
                                layout=domain.layout,
                                domain=domain,
                            ),
                        )
                        if (
                            operator_name
                            == "parallel_density_flux_divergence"
                        ):
                            face_values = []
                            for axis, axis_stencil in enumerate(
                                (local.x, local.y, local.z)
                            ):
                                kind = (
                                    face_bc.kind_x,
                                    face_bc.kind_y,
                                    face_bc.kind_z,
                                )[axis]
                                value = (
                                    face_bc.value_x,
                                    face_bc.value_y,
                                    face_bc.value_z,
                                )[axis]
                                mask = (
                                    face_bc.mask_x,
                                    face_bc.mask_y,
                                    face_bc.mask_z,
                                )[axis]
                                face_values.append(
                                    _apply_local_face_value_dirichlet_bc(
                                        _local_axis_face_values_from_stencil(
                                            axis_stencil,
                                            axis=axis,
                                        ),
                                        axis=axis,
                                        axis_kind=kind,
                                        axis_value=value,
                                        axis_mask=mask,
                                        axis_regular_axes=(
                                            False,
                                            False,
                                            False,
                                        ),
                                    )
                                )
                            regular_flux = []
                            face_bfield = (
                                local_geometry_value.face_bfield.x,
                                local_geometry_value.face_bfield.y,
                                local_geometry_value.face_bfield.z,
                            )
                            face_metric = (
                                local_geometry_value.face_metric.x,
                                local_geometry_value.face_metric.y,
                                local_geometry_value.face_metric.z,
                            )
                            for axis in range(3):
                                B_contra = jnp.asarray(
                                    face_bfield[axis].B_contra_owned,
                                    dtype=jnp.float64,
                                )
                                Bmag = jnp.maximum(
                                    jnp.asarray(
                                        face_bfield[axis].Bmag_owned,
                                        dtype=jnp.float64,
                                    ),
                                    1.0e-30,
                                )
                                flux = (
                                    jnp.asarray(
                                        face_metric[axis].J_owned,
                                        dtype=jnp.float64,
                                    )
                                    * B_contra[..., axis]
                                    / Bmag
                                    * face_values[axis]
                                )
                                regular_flux.append(
                                    _apply_local_face_flux_bc(
                                        flux,
                                        axis=axis,
                                        axis_kind=(
                                            face_bc.kind_x,
                                            face_bc.kind_y,
                                            face_bc.kind_z,
                                        )[axis],
                                        axis_value=(
                                            face_bc.value_x,
                                            face_bc.value_y,
                                            face_bc.value_z,
                                        )[axis],
                                        axis_mask=(
                                            face_bc.mask_x,
                                            face_bc.mask_y,
                                            face_bc.mask_z,
                                        )[axis],
                                        axis_regular_axes=(
                                            False,
                                            False,
                                            False,
                                        ),
                                    )
                                )
                            irregular_flux = (
                                _local_control_volume_irregular_parallel_flux(
                                    jnp.asarray(
                                        local.x.center,
                                        dtype=jnp.float64,
                                    ),
                                    polynomial,
                                    control_volume_geometry,
                                    boundary_bc,
                                    b_floor=1.0e-30,
                                )
                            )
                        else:
                            cv_flux = build_local_perp_laplacian_stencil(
                                local,
                                local_geometry_value,
                                domain,
                                face_projectors=(
                                    local_invariants.face_projector_x,
                                    local_invariants.face_projector_y,
                                    local_invariants.face_projector_z,
                                ),
                                face_bc=face_bc,
                                regular_face_geometry=(
                                    control_volume_geometry.regular_faces
                                ),
                                regular_boundary_closure=(
                                    control_volume_geometry.regular_boundary_closure
                                ),
                            )
                            regular_flux = [
                                cv_flux.regular_flux.x,
                                cv_flux.regular_flux.y,
                                cv_flux.regular_flux.z,
                            ]
                            irregular_flux = (
                                _local_control_volume_irregular_projected_flux(
                                    jnp.asarray(
                                        local.x.center,
                                        dtype=jnp.float64,
                                    ),
                                    polynomial,
                                    control_volume_geometry,
                                    boundary_bc,
                                )
                            )

                        owns_target = jnp.asarray(True)
                        for axis, mesh_axis_name in enumerate(
                            MESH_AXIS_NAMES
                        ):
                            owns_target = owns_target & (
                                lax.axis_index(mesh_axis_name)
                                == int(target_shard[axis])
                            )
                        local_i, local_j, local_k = target_local
                        minus_target = (
                            faces.active
                            & (faces.minus_owner_i == local_i)
                            & (faces.minus_owner_j == local_j)
                            & (faces.minus_owner_k == local_k)
                        )
                        plus_target = (
                            faces.active
                            & faces.has_plus_owner
                            & (faces.plus_owner_i == local_i)
                            & (faces.plus_owner_j == local_j)
                            & (faces.plus_owner_k == local_k)
                        )
                        selected = owns_target & (
                            minus_target | plus_target
                        )
                        selected_count = jnp.sum(
                            selected.astype(jnp.int32)
                        )
                        selected_index = jnp.nonzero(
                            selected,
                            size=max_reported_rows,
                            fill_value=0,
                        )[0]
                        slot_active = (
                            jnp.arange(max_reported_rows)
                            < selected_count
                        )
                        row_sign_all = (
                            minus_target.astype(jnp.float64)
                            - plus_target.astype(jnp.float64)
                        )

                        points = faces.quadrature_points
                        minus_value, minus_gradient, minus_valid = (
                            evaluate_local_control_volume_polynomial(
                                jnp.asarray(
                                    local.x.center,
                                    dtype=jnp.float64,
                                ),
                                polynomial,
                                cells,
                                faces.minus_owner_i[:, None, None],
                                faces.minus_owner_j[:, None, None],
                                faces.minus_owner_k[:, None, None],
                                points,
                            )
                        )
                        plus_value, plus_gradient, plus_valid = (
                            evaluate_local_control_volume_polynomial(
                                jnp.asarray(
                                    local.x.center,
                                    dtype=jnp.float64,
                                ),
                                polynomial,
                                cells,
                                faces.plus_owner_i[:, None, None],
                                faces.plus_owner_j[:, None, None],
                                faces.plus_owner_k[:, None, None],
                                points,
                            )
                        )
                        has_neighbor = (
                            faces.has_plus_owner
                            | faces.has_remote_owner
                        )
                        neighbor_value = jnp.where(
                            faces.has_remote_owner[:, None, None],
                            polynomial.remote_face_value,
                            plus_value,
                        )
                        neighbor_gradient = jnp.where(
                            faces.has_remote_owner[:, None, None, None],
                            polynomial.remote_face_gradient,
                            plus_gradient,
                        )
                        neighbor_valid = jnp.where(
                            faces.has_remote_owner[:, None, None],
                            polynomial.remote_face_valid,
                            plus_valid,
                        )
                        (
                            transition_value,
                            transition_gradient,
                            transition_valid,
                        ) = _evaluate_local_regular_transition_functional(
                            jnp.asarray(local.x.center, dtype=jnp.float64),
                            polynomial,
                            control_volume_geometry,
                        )
                        transition_rows = (
                            control_volume_geometry.regular_transition_faces
                        )
                        transition_active = (
                            transition_rows.active
                            if int(transition_rows.max_rows)
                            == int(faces.max_rows)
                            else jnp.zeros(
                                (int(faces.max_rows),), dtype=bool
                            )
                        )
                        transition_sample_storage = jnp.stack(
                            (
                                transition_rows.sample_storage_i,
                                transition_rows.sample_storage_j,
                                transition_rows.sample_storage_k,
                            ),
                            axis=-1,
                        )
                        transition_sample_center = jnp.stack(
                            (
                                transition_rows.sample_center_owner_i,
                                transition_rows.sample_center_owner_j,
                                transition_rows.sample_center_owner_k,
                            ),
                            axis=-1,
                        )
                        transition_sample_owner = jnp.stack(
                            (
                                transition_rows.sample_owner_i,
                                transition_rows.sample_owner_j,
                                transition_rows.sample_owner_k,
                            ),
                            axis=-1,
                        )
                        transition_storage_value = jnp.asarray(
                            local.x.center, dtype=jnp.float64
                        )[
                            transition_rows.sample_storage_i,
                            transition_rows.sample_storage_j,
                            transition_rows.sample_storage_k,
                        ]
                        transition_owner_value = jnp.asarray(
                            local.x.center, dtype=jnp.float64
                        )[
                            transition_rows.sample_owner_i,
                            transition_rows.sample_owner_j,
                            transition_rows.sample_owner_k,
                        ]
                        transition_owner_gradient = polynomial.gradient[
                            transition_rows.sample_owner_i,
                            transition_rows.sample_owner_j,
                            transition_rows.sample_owner_k,
                        ]
                        transition_owner_hessian = polynomial.hessian[
                            transition_rows.sample_owner_i,
                            transition_rows.sample_owner_j,
                            transition_rows.sample_owner_k,
                        ]
                        transition_virtual_value = (
                            transition_owner_value
                            + jnp.einsum(
                                "rsi,rsi->rs",
                                transition_owner_gradient,
                                transition_rows.sample_displacement,
                            )
                            + 0.5 * jnp.einsum(
                                "rsij,rsij->rs",
                                transition_owner_hessian,
                                transition_rows.sample_moment_delta,
                            )
                        )
                        transition_sample_value = jnp.where(
                            transition_rows.sample_direct,
                            transition_storage_value,
                            transition_virtual_value,
                        )
                        transition_sample_coefficients = jnp.concatenate(
                            (
                                transition_rows.scalar_coefficients[..., None],
                                jnp.swapaxes(
                                    transition_rows.gradient_coefficients,
                                    1,
                                    2,
                                ),
                            ),
                            axis=-1,
                        )
                        if (
                            operator_name
                            == "parallel_density_flux_divergence"
                        ):
                            face_value = jnp.where(
                                has_neighbor[:, None, None],
                                0.5 * (minus_value + neighbor_value),
                                jnp.where(
                                    boundary_bc.kind[:, None, None]
                                    == BC_DIRICHLET,
                                    boundary_bc.quadrature_value,
                                    minus_value,
                                ),
                            )
                            face_value = jnp.where(
                                transition_active[:, None, None],
                                transition_value[:, None, None],
                                face_value,
                            )
                            unit_b = faces.B_contra / jnp.maximum(
                                faces.Bmag[..., None],
                                1.0e-30,
                            )
                            numerical_quadrature_flux = (
                                faces.J
                                * jnp.einsum(
                                    "rpqi,rpqi->rpq",
                                    faces.area_covector_weight,
                                    unit_b,
                                )
                                * face_value
                            )
                            exact_value = (
                                _shifted_torus_exact_field_at_logical_points(
                                    points,
                                    stage_time,
                                    exact_field_name,
                                )
                            )
                            exact_quadrature_flux = (
                                faces.J
                                * jnp.einsum(
                                    "rpqi,rpqi->rpq",
                                    faces.area_covector_weight,
                                    unit_b,
                                )
                                * exact_value
                            )
                        else:
                            face_gradient = jnp.where(
                                has_neighbor[:, None, None, None],
                                0.5
                                * (
                                    minus_gradient
                                    + neighbor_gradient
                                ),
                                minus_gradient,
                            )
                            face_gradient = jnp.where(
                                transition_active[:, None, None, None],
                                transition_gradient[:, None, None, :],
                                face_gradient,
                            )
                            applied_face_gradient = face_gradient
                            cut_wall_normal_closure_valid = jnp.zeros_like(
                                faces.quadrature_active
                            )
                            numerical_quadrature_flux = (
                                faces.J
                                * jnp.einsum(
                                    "rpqi,rpqij,rpqj->rpq",
                                    faces.area_covector_weight,
                                    faces.projector,
                                    applied_face_gradient,
                                )
                            )
                            _exact_value, exact_gradient = (
                                _shifted_torus_exact_field_and_gradient_at_logical_points(
                                    points,
                                    stage_time,
                                    exact_field_name,
                                )
                            )
                            exact_quadrature_flux = (
                                faces.J
                                * jnp.einsum(
                                    "rpqi,rpqij,rpqj->rpq",
                                    faces.area_covector_weight,
                                    faces.projector,
                                    exact_gradient,
                                )
                            )
                        quadrature_valid = (
                            faces.quadrature_active
                            & minus_valid
                            & (
                                (~has_neighbor[:, None, None])
                                | neighbor_valid
                            )
                        )
                        quadrature_valid = quadrature_valid & (
                            (~transition_active[:, None, None])
                            | transition_valid[:, None, None]
                        )
                        if operator_name == "parallel_density_flux_divergence":
                            diagnostic_face_gradient = jnp.zeros_like(
                                faces.quadrature_points
                            )
                            diagnostic_applied_face_gradient = (
                                diagnostic_face_gradient
                            )
                            diagnostic_exact_gradient = jnp.zeros_like(
                                faces.quadrature_points
                            )
                            diagnostic_closure_valid = jnp.zeros_like(
                                faces.quadrature_active
                            )
                            diagnostic_closure_axis = jnp.zeros_like(
                                faces.kind
                            )
                            diagnostic_exact_input_derivative = jnp.zeros_like(
                                faces.quadrature_active,
                                dtype=jnp.float64,
                            )
                        else:
                            diagnostic_face_gradient = face_gradient
                            diagnostic_applied_face_gradient = face_gradient
                            diagnostic_exact_gradient = exact_gradient
                            diagnostic_closure_valid = jnp.zeros_like(
                                faces.quadrature_active
                            )
                            diagnostic_closure_axis = jnp.zeros_like(faces.kind)
                            diagnostic_exact_input_derivative = jnp.zeros_like(
                                faces.quadrature_active,
                                dtype=jnp.float64,
                            )
                        diagnostic_projected_covector = jnp.einsum(
                            "rpqij,rpqi->rpqj",
                            faces.projector,
                            faces.area_covector_weight,
                        )
                        # Cut-wall rows have no neighboring control volume.
                        # Separate the polynomial trace and its metric-normal
                        # component from the tangential component so a wrong
                        # projected flux can be attributed without changing
                        # the production functional.
                        normal_length = jnp.sqrt(
                            jnp.maximum(
                                jnp.einsum(
                                    "rpqi,rpqij,rpqj->rpq",
                                    faces.area_covector_weight,
                                    faces.g_contra,
                                    faces.area_covector_weight,
                                ),
                                1.0e-30,
                            )
                        )
                        diagnostic_normal_covector = (
                            faces.area_covector_weight / normal_length[..., None]
                        )
                        diagnostic_normal_contra = jnp.einsum(
                            "rpqij,rpqj->rpqi",
                            faces.g_contra,
                            diagnostic_normal_covector,
                        )
                        diagnostic_normal_derivative = jnp.einsum(
                            "rpqi,rpqi->rpq",
                            diagnostic_normal_contra,
                            diagnostic_face_gradient,
                        )
                        diagnostic_exact_normal_derivative = jnp.einsum(
                            "rpqi,rpqi->rpq",
                            diagnostic_normal_contra,
                            diagnostic_exact_gradient,
                        )
                        diagnostic_tangent_gradient = (
                            diagnostic_face_gradient
                            - diagnostic_normal_derivative[..., None]
                            * diagnostic_normal_covector
                        )
                        diagnostic_exact_tangent_gradient = (
                            diagnostic_exact_gradient
                            - diagnostic_exact_normal_derivative[..., None]
                            * diagnostic_normal_covector
                        )
                        diagnostic_tangent_exact_normal_flux = faces.J * jnp.einsum(
                            "rpqi,rpqi->rpq",
                            diagnostic_projected_covector,
                            diagnostic_tangent_gradient
                            + diagnostic_exact_normal_derivative[..., None]
                            * diagnostic_normal_covector,
                        )
                        diagnostic_exact_tangent_normal_flux = faces.J * jnp.einsum(
                            "rpqi,rpqi->rpq",
                            diagnostic_projected_covector,
                            diagnostic_exact_tangent_gradient
                            + diagnostic_normal_derivative[..., None]
                            * diagnostic_normal_covector,
                        )

                        # Audit precisely the reconstruction row belonging to
                        # the top-error owner.  This is intentionally a single
                        # row, keeping the failure diagnostic small enough to
                        # run beside the operator kernels.
                        reconstruction = control_volume_geometry.reconstruction
                        audit_row = reconstruction.target_row_for_cell[
                            local_i, local_j, local_k
                        ]
                        audit_has_row = audit_row >= 0
                        audit_safe_row = jnp.clip(
                            audit_row,
                            0,
                            max(0, int(reconstruction.max_rows) - 1),
                        )
                        audit_target = jnp.stack(
                            (
                                reconstruction.target_i[audit_safe_row],
                                reconstruction.target_j[audit_safe_row],
                                reconstruction.target_k[audit_safe_row],
                            )
                        )
                        audit_target_value = jnp.asarray(
                            local.x.center, dtype=jnp.float64
                        )[audit_target[0], audit_target[1], audit_target[2]]
                        audit_target_centroid = cells.centroid[
                            audit_target[0], audit_target[1], audit_target[2]
                        ]
                        audit_target_m2 = cells.second_moment[
                            audit_target[0], audit_target[1], audit_target[2]
                        ]
                        audit_kind = reconstruction.equation_kind[audit_safe_row]
                        audit_active = (
                            reconstruction.equation_active[audit_safe_row]
                            & audit_has_row
                        )
                        audit_sample_i = reconstruction.sample_i[audit_safe_row]
                        audit_sample_j = reconstruction.sample_j[audit_safe_row]
                        audit_sample_k = reconstruction.sample_k[audit_safe_row]
                        audit_sample_value = jnp.asarray(
                            local.x.center, dtype=jnp.float64
                        )[
                            jnp.clip(audit_sample_i, 0, cells.shape[0] - 1),
                            jnp.clip(audit_sample_j, 0, cells.shape[1] - 1),
                            jnp.clip(audit_sample_k, 0, cells.shape[2] - 1),
                        ]
                        audit_boundary_row = jnp.clip(
                            reconstruction.boundary_face_row[audit_safe_row],
                            0,
                            max(0, int(boundary_bc.max_rows) - 1),
                        )
                        audit_boundary_patch = jnp.clip(
                            reconstruction.boundary_patch[audit_safe_row],
                            0,
                            max(0, int(faces.max_patches) - 1),
                        )
                        audit_boundary_quadrature = jnp.clip(
                            reconstruction.boundary_quadrature[audit_safe_row],
                            0,
                            3,
                        )
                        audit_boundary_value = boundary_bc.quadrature_value[
                            audit_boundary_row,
                            audit_boundary_patch,
                            audit_boundary_quadrature,
                        ]
                        audit_rhs = jnp.where(
                            audit_kind == CV_RECONSTRUCTION_EQUATION_CELL,
                            audit_sample_value - audit_target_value,
                            audit_boundary_value - audit_target_value,
                        )
                        audit_sample_centroid = cells.centroid[
                            jnp.clip(audit_sample_i, 0, cells.shape[0] - 1),
                            jnp.clip(audit_sample_j, 0, cells.shape[1] - 1),
                            jnp.clip(audit_sample_k, 0, cells.shape[2] - 1),
                        ]
                        audit_sample_m2 = cells.second_moment[
                            jnp.clip(audit_sample_i, 0, cells.shape[0] - 1),
                            jnp.clip(audit_sample_j, 0, cells.shape[1] - 1),
                            jnp.clip(audit_sample_k, 0, cells.shape[2] - 1),
                        ]
                        audit_wall_point = faces.quadrature_points[
                            audit_boundary_row,
                            audit_boundary_patch,
                            audit_boundary_quadrature,
                        ]
                        audit_displacement = jnp.where(
                            (
                                audit_kind
                                == CV_RECONSTRUCTION_EQUATION_DIRICHLET
                            )[:, None],
                            audit_wall_point - audit_target_centroid,
                            audit_sample_centroid - audit_target_centroid,
                        )
                        audit_moment_delta = jnp.where(
                            (
                                audit_kind
                                == CV_RECONSTRUCTION_EQUATION_DIRICHLET
                            )[:, None, None],
                            audit_displacement[..., :, None]
                            * audit_displacement[..., None, :]
                            - audit_target_m2,
                            audit_sample_m2
                            + audit_displacement[..., :, None]
                            * audit_displacement[..., None, :]
                            - audit_target_m2,
                        )
                        audit_gradient = polynomial.gradient[
                            audit_target[0], audit_target[1], audit_target[2]
                        ]
                        audit_hessian = polynomial.hessian[
                            audit_target[0], audit_target[1], audit_target[2]
                        ]
                        audit_predicted_rhs = (
                            jnp.einsum(
                                "i,ei->e",
                                audit_gradient,
                                audit_displacement,
                            )
                            + 0.5
                            * jnp.einsum(
                                "ij,eij->e",
                                audit_hessian,
                                audit_moment_delta,
                            )
                        )
                        audit_residual = jnp.where(
                            audit_active
                            & (
                                audit_kind
                                != CV_RECONSTRUCTION_EQUATION_REMOTE_CELL
                            ),
                            audit_predicted_rhs - audit_rhs,
                            0.0,
                        )

                        # Replay the same transform with an exact quadratic
                        # field.  Failure is a geometry/metadata/transform bug;
                        # success isolates the MMS error to approximation rather
                        # than the algebraic reconstruction machinery.
                        synthetic_gradient0 = jnp.asarray(
                            (0.31, -0.27, 0.19), dtype=jnp.float64
                        )
                        synthetic_hessian = jnp.asarray(
                            (
                                (0.41, -0.07, 0.05),
                                (-0.07, -0.33, 0.08),
                                (0.05, 0.08, 0.29),
                            ),
                            dtype=jnp.float64,
                        )

                        def synthetic_average(position, moment):
                            return (
                                0.37
                                + jnp.einsum(
                                    "i,...i->...",
                                    synthetic_gradient0,
                                    position,
                                )
                                + 0.5
                                * jnp.einsum(
                                    "...i,ij,...j->...",
                                    position,
                                    synthetic_hessian,
                                    position,
                                )
                                + 0.5
                                * jnp.einsum(
                                    "ij,...ij->...",
                                    synthetic_hessian,
                                    moment,
                                )
                            )

                        def synthetic_point(position):
                            return (
                                0.37
                                + jnp.einsum(
                                    "i,...i->...",
                                    synthetic_gradient0,
                                    position,
                                )
                                + 0.5
                                * jnp.einsum(
                                    "...i,ij,...j->...",
                                    position,
                                    synthetic_hessian,
                                    position,
                                )
                            )

                        synthetic_target = synthetic_average(
                            audit_target_centroid,
                            audit_target_m2,
                        )
                        synthetic_sample = synthetic_average(
                            audit_sample_centroid,
                            audit_sample_m2,
                        )
                        synthetic_boundary = synthetic_point(audit_wall_point)
                        synthetic_rhs = jnp.where(
                            audit_kind == CV_RECONSTRUCTION_EQUATION_CELL,
                            synthetic_sample - synthetic_target,
                            jnp.where(
                                audit_kind
                                == CV_RECONSTRUCTION_EQUATION_DIRICHLET,
                                synthetic_boundary - synthetic_target,
                                0.0,
                            ),
                        )
                        audit_synthetic_valid = audit_has_row & (~jnp.any(
                            audit_active
                            & (
                                audit_kind
                                == CV_RECONSTRUCTION_EQUATION_REMOTE_CELL
                            )
                        ))
                        synthetic_coefficients = jnp.einsum(
                            "ie,e->i",
                            reconstruction.rhs_transform[audit_safe_row],
                            jnp.where(audit_active, synthetic_rhs, 0.0),
                        )
                        synthetic_expected_quadratic = jnp.concatenate(
                            (
                                synthetic_gradient0
                                + synthetic_hessian @ audit_target_centroid,
                                jnp.asarray(
                                    (
                                        synthetic_hessian[0, 0],
                                        synthetic_hessian[1, 1],
                                        synthetic_hessian[2, 2],
                                        synthetic_hessian[0, 1],
                                        synthetic_hessian[0, 2],
                                        synthetic_hessian[1, 2],
                                    ),
                                    dtype=jnp.float64,
                                ),
                            )
                        )
                        coefficient_count = int(
                            reconstruction.rhs_transform.shape[1]
                        )
                        synthetic_expected = jnp.pad(
                            synthetic_expected_quadratic,
                            (0, coefficient_count - 9),
                        )
                        numerical_quadrature_flux = jnp.where(
                            quadrature_valid,
                            numerical_quadrature_flux,
                            0.0,
                        )
                        exact_quadrature_flux = jnp.where(
                            faces.quadrature_active,
                            exact_quadrature_flux,
                            0.0,
                        )
                        exact_irregular_flux = jnp.sum(
                            exact_quadrature_flux,
                            axis=(1, 2),
                        )

                        regular_faces = (
                            control_volume_geometry.regular_faces
                        )
                        spacing = (
                            local_geometry_value.spacing.dx_owned,
                            local_geometry_value.spacing.dy_owned,
                            local_geometry_value.spacing.dz_owned,
                        )
                        logical_measure = (
                            spacing[1] * spacing[2],
                            spacing[0] * spacing[2],
                            spacing[0] * spacing[1],
                        )
                        area = (
                            regular_faces.x_area,
                            regular_faces.y_area,
                            regular_faces.z_area,
                        )
                        fraction = (
                            regular_faces.x_area_fraction,
                            regular_faces.y_area_fraction,
                            regular_faces.z_area_fraction,
                        )
                        open_mask = (
                            regular_faces.x_open_mask,
                            regular_faces.y_open_mask,
                            regular_faces.z_open_mask,
                        )
                        dense_flux = jnp.zeros(
                            (3, 2),
                            dtype=jnp.float64,
                        )
                        dense_measure = jnp.zeros(
                            (3, 2),
                            dtype=jnp.float64,
                        )
                        dense_open = jnp.zeros((3, 2), dtype=bool)
                        dense_exact_flux = jnp.zeros(
                            (3, 2),
                            dtype=jnp.float64,
                        )
                        dense_compact_owned = jnp.zeros(
                            (3, 2),
                            dtype=bool,
                        )
                        dense_neighbor_reconstructed = jnp.zeros(
                            (3, 2),
                            dtype=bool,
                        )
                        grid_centers = (
                            local_geometry_value.grid.x.centers_owned,
                            local_geometry_value.grid.y.centers_owned,
                            local_geometry_value.grid.z.centers_owned,
                        )
                        grid_faces = (
                            local_geometry_value.grid.x.faces_owned,
                            local_geometry_value.grid.y.faces_owned,
                            local_geometry_value.grid.z.faces_owned,
                        )
                        gauss_offset = 1.0 / (
                            2.0 * jnp.sqrt(3.0)
                        )
                        quadrature_signs = jnp.asarray(
                            (
                                (-1.0, -1.0),
                                (-1.0, 1.0),
                                (1.0, -1.0),
                                (1.0, 1.0),
                            ),
                            dtype=jnp.float64,
                        )
                        for axis in range(3):
                            open_measure = (
                                _lift_cell_field_to_faces(
                                    logical_measure[axis],
                                    axis=axis,
                                    periodic=False,
                                )
                                * jnp.asarray(area[axis], dtype=jnp.float64)
                                * jnp.asarray(
                                    fraction[axis],
                                    dtype=jnp.float64,
                                )
                            )
                            integrated_face = jnp.where(
                                jnp.asarray(open_mask[axis], dtype=bool)
                                & (open_measure > 0.0),
                                jnp.asarray(
                                    regular_flux[axis],
                                    dtype=jnp.float64,
                                )
                                * open_measure,
                                0.0,
                            )
                            for side in range(2):
                                face_index = [
                                    local_i,
                                    local_j,
                                    local_k,
                                ]
                                face_index[axis] += side
                                face_index = tuple(face_index)
                                orientation = -1.0 if side == 0 else 1.0
                                dense_flux = dense_flux.at[
                                    axis,
                                    side,
                                ].set(
                                    orientation
                                    * integrated_face[face_index]
                                )
                                dense_measure = dense_measure.at[
                                    axis,
                                    side,
                                ].set(open_measure[face_index])
                                dense_open = dense_open.at[
                                    axis,
                                    side,
                                ].set(
                                    jnp.asarray(
                                        open_mask[axis],
                                        dtype=bool,
                                    )[face_index]
                                )
                                dense_compact_owned = (
                                    dense_compact_owned.at[
                                        axis,
                                        side,
                                    ].set(
                                        ~jnp.asarray(
                                            open_mask[axis],
                                            dtype=bool,
                                        )[face_index]
                                    )
                                )
                                neighbor_index = [
                                    local_i,
                                    local_j,
                                    local_k,
                                ]
                                neighbor_index[axis] += (
                                    -1 if side == 0 else 1
                                )
                                neighbor_in_bounds = all(
                                    0
                                    <= int(neighbor_index[candidate_axis])
                                    < int(
                                        local_geometry_value.owned_shape[
                                            candidate_axis
                                        ]
                                    )
                                    for candidate_axis in range(3)
                                )
                                if neighbor_in_bounds:
                                    neighbor_reconstructed = (
                                        polynomial.polynomial_order[
                                            tuple(neighbor_index)
                                        ]
                                        > 0
                                    )
                                else:
                                    neighbor_reconstructed = jnp.asarray(
                                        False
                                    )
                                dense_neighbor_reconstructed = (
                                    dense_neighbor_reconstructed.at[
                                        axis,
                                        side,
                                    ].set(neighbor_reconstructed)
                                )

                                tangential_axes = tuple(
                                    candidate_axis
                                    for candidate_axis in range(3)
                                    if candidate_axis != axis
                                )
                                face_center = jnp.stack(
                                    tuple(
                                        jnp.asarray(
                                            grid_faces[candidate_axis][
                                                face_index[candidate_axis]
                                            ]
                                            if candidate_axis == axis
                                            else grid_centers[candidate_axis][
                                                face_index[candidate_axis]
                                            ],
                                            dtype=jnp.float64,
                                        )
                                        for candidate_axis in range(3)
                                    )
                                )
                                face_points = jnp.broadcast_to(
                                    face_center,
                                    (4, 3),
                                )
                                for (
                                    tangential_slot,
                                    tangential_axis,
                                ) in enumerate(tangential_axes):
                                    face_points = face_points.at[
                                        :,
                                        tangential_axis,
                                    ].add(
                                        quadrature_signs[
                                            :,
                                            tangential_slot,
                                        ]
                                        * spacing[tangential_axis][
                                            local_i,
                                            local_j,
                                            local_k,
                                        ]
                                        * gauss_offset
                                    )
                                (
                                    exact_face_J,
                                    _exact_face_g_contra,
                                    _exact_face_g_cov,
                                    exact_face_B_contra,
                                    exact_face_Bmag,
                                    exact_face_projector,
                                ) = _shifted_torus_metric_payload_jax(
                                    face_points
                                )
                                if (
                                    operator_name
                                    == "parallel_density_flux_divergence"
                                ):
                                    exact_face_value = (
                                        _shifted_torus_exact_field_at_logical_points(
                                            face_points,
                                            stage_time,
                                            exact_field_name,
                                        )
                                    )
                                    exact_flux_density = (
                                        exact_face_J
                                        * exact_face_B_contra[:, axis]
                                        / jnp.maximum(
                                            exact_face_Bmag,
                                            1.0e-30,
                                        )
                                        * exact_face_value
                                    )
                                else:
                                    (
                                        _exact_face_value,
                                        exact_face_gradient,
                                    ) = (
                                        _shifted_torus_exact_field_and_gradient_at_logical_points(
                                            face_points,
                                            stage_time,
                                            exact_field_name,
                                        )
                                    )
                                    exact_flux_density = (
                                        exact_face_J
                                        * jnp.einsum(
                                            "qi,qi->q",
                                            exact_face_projector[
                                                :,
                                                axis,
                                                :,
                                            ],
                                            exact_face_gradient,
                                        )
                                    )
                                dense_exact_flux = dense_exact_flux.at[
                                    axis,
                                    side,
                                ].set(
                                    jnp.where(
                                        dense_open[axis, side],
                                        orientation
                                        * jnp.mean(
                                            exact_flux_density
                                        )
                                        * open_measure[face_index],
                                        0.0,
                                    )
                                )

                        def target_reduce(value):
                            result = jnp.where(
                                owns_target,
                                value,
                                jnp.zeros_like(value),
                            )
                            for mesh_axis_name in MESH_AXIS_NAMES:
                                result = lax.psum(
                                    result,
                                    mesh_axis_name,
                                )
                            return result

                        def selected_rows(value):
                            gathered = value[selected_index]
                            active_shape = (
                                (max_reported_rows,)
                                + (1,) * (gathered.ndim - 1)
                            )
                            return target_reduce(
                                jnp.where(
                                    slot_active.reshape(active_shape),
                                    gathered,
                                    jnp.zeros_like(gathered),
                                )
                            )

                        # Diagnose the actual transition functional using
                        # exact J-weighted regular-cell averages only for the
                        # handful of attached rows selected above.  This
                        # separates virtual-average reconstruction error from
                        # truncation in the structured face functional.
                        selected_transition_storage = (
                            transition_sample_storage[selected_index]
                        )
                        selected_transition_value = (
                            transition_sample_value[selected_index]
                        )
                        selected_transition_coefficients = (
                            transition_sample_coefficients[selected_index]
                        )
                        selected_transition_owner = (
                            transition_sample_owner[selected_index]
                        )
                        selected_owner_centroid = cells.centroid[
                            selected_transition_owner[..., 0],
                            selected_transition_owner[..., 1],
                            selected_transition_owner[..., 2],
                        ]
                        selected_owner_value = jnp.asarray(
                            local.x.center, dtype=jnp.float64
                        )[
                            selected_transition_owner[..., 0],
                            selected_transition_owner[..., 1],
                            selected_transition_owner[..., 2],
                        ]
                        selected_owner_gradient = polynomial.gradient[
                            selected_transition_owner[..., 0],
                            selected_transition_owner[..., 1],
                            selected_transition_owner[..., 2],
                        ]
                        selected_owner_order = polynomial.polynomial_order[
                            selected_transition_owner[..., 0],
                            selected_transition_owner[..., 1],
                            selected_transition_owner[..., 2],
                        ]
                        selected_owner_condition = polynomial.condition_number[
                            selected_transition_owner[..., 0],
                            selected_transition_owner[..., 1],
                            selected_transition_owner[..., 2],
                        ]
                        (
                            selected_owner_exact_value,
                            selected_owner_exact_gradient,
                        ) = _shifted_torus_exact_field_and_gradient_at_logical_points(
                            selected_owner_centroid,
                            stage_time,
                            exact_field_name,
                        )
                        transition_axis_faces = (
                            local_geometry_value.grid.x.faces_owned,
                            local_geometry_value.grid.y.faces_owned,
                            local_geometry_value.grid.z.faces_owned,
                        )
                        selected_lower = jnp.stack(
                            tuple(
                                transition_axis_faces[axis][
                                    selected_transition_storage[..., axis]
                                ]
                                for axis in range(3)
                            ),
                            axis=-1,
                        )
                        selected_upper = jnp.stack(
                            tuple(
                                transition_axis_faces[axis][
                                    selected_transition_storage[..., axis] + 1
                                ]
                                for axis in range(3)
                            ),
                            axis=-1,
                        )
                        quadrature_nodes = jnp.asarray(
                            (-jnp.sqrt(3.0 / 5.0), 0.0, jnp.sqrt(3.0 / 5.0)),
                            dtype=jnp.float64,
                        )
                        quadrature_weights_1d = jnp.asarray(
                            (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0),
                            dtype=jnp.float64,
                        )
                        qx, qy, qz = jnp.meshgrid(
                            quadrature_nodes,
                            quadrature_nodes,
                            quadrature_nodes,
                            indexing="ij",
                        )
                        selected_offsets = jnp.stack((qx, qy, qz), axis=-1)
                        selected_weight = (
                            quadrature_weights_1d[:, None, None]
                            * quadrature_weights_1d[None, :, None]
                            * quadrature_weights_1d[None, None, :]
                        )
                        selected_midpoint = 0.5 * (
                            selected_lower + selected_upper
                        )
                        selected_halfwidth = 0.5 * (
                            selected_upper - selected_lower
                        )
                        selected_points = (
                            selected_midpoint[..., None, None, None, :]
                            + selected_halfwidth[..., None, None, None, :]
                            * selected_offsets
                        )
                        selected_exact_value = (
                            _shifted_torus_exact_field_at_logical_points(
                                selected_points,
                                stage_time,
                                exact_field_name,
                            )
                        )
                        selected_exact_J = _shifted_torus_metric_payload_jax(
                            selected_points
                        )[0]
                        selected_exact_average = jnp.sum(
                            selected_weight * selected_exact_J * selected_exact_value,
                            axis=(-3, -2, -1),
                        ) / jnp.maximum(
                            jnp.sum(
                                selected_weight * selected_exact_J,
                                axis=(-3, -2, -1),
                            ),
                            1.0e-30,
                        )
                        selected_structured_value = jnp.einsum(
                            "rs,rs->r",
                            selected_transition_coefficients[..., 0],
                            selected_exact_average,
                        )
                        selected_structured_gradient = jnp.einsum(
                            "rsc,rs->rc",
                            selected_transition_coefficients[..., 1:],
                            selected_exact_average,
                        )
                        selected_faces_J = faces.J[selected_index]
                        selected_faces_area = (
                            faces.area_covector_weight[selected_index]
                        )
                        selected_faces_active = (
                            faces.quadrature_active[selected_index]
                        )
                        if operator_name == "parallel_density_flux_divergence":
                            selected_unit_b = faces.B_contra[selected_index] / jnp.maximum(
                                faces.Bmag[selected_index, ..., None],
                                1.0e-30,
                            )
                            selected_structured_quadrature_flux = (
                                selected_faces_J
                                * jnp.einsum(
                                    "rpqi,rpqi->rpq",
                                    selected_faces_area,
                                    selected_unit_b,
                                )
                                * selected_structured_value[:, None, None]
                            )
                        else:
                            selected_structured_quadrature_flux = (
                                selected_faces_J
                                * jnp.einsum(
                                    "rpqi,rpqij,rj->rpq",
                                    selected_faces_area,
                                    faces.projector[selected_index],
                                    selected_structured_gradient,
                                )
                            )
                        selected_structured_flux = jnp.sum(
                            jnp.where(
                                selected_faces_active,
                                selected_structured_quadrature_flux,
                                0.0,
                            ),
                            axis=(1, 2),
                        )

                        owner_indices = jnp.stack(
                            (
                                faces.minus_owner_i,
                                faces.minus_owner_j,
                                faces.minus_owner_k,
                                faces.plus_owner_i,
                                faces.plus_owner_j,
                                faces.plus_owner_k,
                            ),
                            axis=-1,
                        )
                        full_irregular_numerical_sum = jnp.sum(
                            row_sign_all * irregular_flux
                        )
                        full_irregular_exact_sum = jnp.sum(
                            row_sign_all * exact_irregular_flux
                        )
                        return (
                            target_reduce(
                                slot_active.astype(jnp.int32)
                            ),
                            target_reduce(
                                jnp.where(
                                    slot_active,
                                    selected_index,
                                    jnp.zeros_like(selected_index),
                                )
                            ),
                            selected_rows(faces.kind),
                            selected_rows(row_sign_all),
                            selected_rows(
                                faces.has_plus_owner.astype(jnp.int32)
                            ),
                            selected_rows(
                                faces.has_remote_owner.astype(jnp.int32)
                            ),
                            selected_rows(owner_indices),
                            selected_rows(irregular_flux),
                            selected_rows(exact_irregular_flux),
                            selected_rows(numerical_quadrature_flux),
                            selected_rows(exact_quadrature_flux),
                            selected_rows(
                                faces.quadrature_active.astype(jnp.int32)
                            ),
                            selected_rows(points),
                            target_reduce(dense_flux),
                            target_reduce(dense_exact_flux),
                            target_reduce(dense_measure),
                            target_reduce(dense_open.astype(jnp.int32)),
                            target_reduce(
                                dense_compact_owned.astype(
                                    jnp.int32
                                )
                            ),
                            target_reduce(
                                dense_neighbor_reconstructed.astype(
                                    jnp.int32
                                )
                            ),
                            target_reduce(selected_count),
                            target_reduce(
                                full_irregular_numerical_sum
                            ),
                            target_reduce(full_irregular_exact_sum),
                            selected_rows(transition_active.astype(jnp.int32)),
                            selected_rows(transition_valid.astype(jnp.int32)),
                            selected_rows(transition_sample_storage),
                            selected_rows(transition_sample_center),
                            selected_rows(transition_sample_owner),
                            selected_rows(
                                transition_rows.sample_direct.astype(jnp.int32)
                            ),
                            selected_rows(transition_sample_value),
                            selected_rows(transition_sample_coefficients),
                            target_reduce(selected_exact_average),
                            target_reduce(selected_structured_value),
                            target_reduce(selected_structured_gradient),
                            target_reduce(selected_structured_flux),
                            target_reduce(selected_owner_centroid),
                            target_reduce(selected_owner_value),
                            target_reduce(selected_owner_gradient),
                            target_reduce(selected_owner_exact_value),
                            target_reduce(selected_owner_exact_gradient),
                            target_reduce(selected_owner_order),
                            target_reduce(selected_owner_condition),
                            selected_rows(faces.kind),
                            selected_rows(boundary_bc.kind),
                            selected_rows(boundary_bc.quadrature_value),
                            selected_rows(diagnostic_face_gradient),
                            selected_rows(diagnostic_applied_face_gradient),
                            selected_rows(diagnostic_exact_gradient),
                            selected_rows(
                                diagnostic_closure_valid.astype(jnp.int32)
                            ),
                            selected_rows(diagnostic_closure_axis),
                            selected_rows(diagnostic_exact_input_derivative),
                            selected_rows(diagnostic_projected_covector),
                            selected_rows(minus_value),
                            selected_rows(
                                minus_value - boundary_bc.quadrature_value
                            ),
                            selected_rows(diagnostic_normal_contra),
                            selected_rows(diagnostic_normal_covector),
                            selected_rows(diagnostic_normal_derivative),
                            selected_rows(
                                diagnostic_exact_normal_derivative
                            ),
                            selected_rows(
                                diagnostic_tangent_exact_normal_flux
                            ),
                            selected_rows(
                                diagnostic_exact_tangent_normal_flux
                            ),
                            target_reduce(audit_has_row.astype(jnp.int32)),
                            target_reduce(audit_target),
                            target_reduce(audit_kind),
                            target_reduce(audit_active.astype(jnp.int32)),
                            target_reduce(audit_rhs),
                            target_reduce(audit_predicted_rhs),
                            target_reduce(audit_residual),
                            target_reduce(
                                reconstruction.rank[audit_safe_row]
                            ),
                            target_reduce(
                                reconstruction.condition_number[audit_safe_row]
                            ),
                            target_reduce(synthetic_coefficients),
                            target_reduce(synthetic_expected),
                            target_reduce(
                                audit_synthetic_valid.astype(jnp.int32)
                            ),
                        )

                    face_diagnostic = jax.jit(
                        shard_map(
                            top_cell_face_diagnostic_kernel,
                            mesh=mesh,
                            in_specs=(
                                state_spec,
                                field_spec,
                                invariant_spec,
                                control_volume_spec,
                                P(),
                            ),
                            out_specs=(P(),) * 71,
                            check_rep=False,
                        )
                    )(
                        state_mesh,
                        phi_mesh,
                        invariants_mesh,
                        control_volume_mesh,
                        jnp.asarray(0.0, dtype=jnp.float64),
                    )
                    face_diagnostic = tuple(
                        np.asarray(jax.device_get(value))
                        for value in face_diagnostic
                    )
                    (
                        slot_active,
                        row_index,
                        row_kind,
                        row_sign,
                        row_has_plus,
                        row_has_remote,
                        row_owners,
                        row_flux,
                        row_exact_flux,
                        row_quadrature_flux,
                        row_exact_quadrature_flux,
                        row_quadrature_active,
                        row_points,
                        dense_flux,
                        dense_exact_flux,
                        dense_measure,
                        dense_open,
                        dense_compact_owned,
                        dense_neighbor_reconstructed,
                        selected_count,
                        irregular_numerical_sum,
                        irregular_exact_sum,
                        row_transition_active,
                        row_transition_valid,
                        row_transition_storage,
                        row_transition_center,
                        row_transition_owner,
                        row_transition_direct,
                        row_transition_value,
                        row_transition_coefficients,
                        row_transition_exact_average,
                        row_transition_exact_value,
                        row_transition_exact_gradient,
                        row_transition_exact_structured_flux,
                        row_transition_owner_centroid,
                        row_transition_owner_value,
                        row_transition_owner_gradient,
                        row_transition_owner_exact_value,
                        row_transition_owner_exact_gradient,
                        row_transition_owner_order,
                        row_transition_owner_condition,
                        row_boundary_face_kind,
                        row_boundary_bc_kind,
                        row_boundary_bc_value,
                        row_boundary_raw_gradient,
                        row_boundary_applied_gradient,
                        row_boundary_exact_gradient,
                        row_boundary_closure_valid,
                        row_boundary_closure_axis,
                        row_boundary_exact_input_derivative,
                        row_boundary_projected_covector,
                        row_boundary_polynomial_value,
                        row_boundary_trace_residual,
                        row_boundary_normal_contra,
                        row_boundary_normal_covector,
                        row_boundary_normal_derivative,
                        row_boundary_exact_normal_derivative,
                        row_boundary_numerical_tangent_exact_normal_flux,
                        row_boundary_exact_tangent_numerical_normal_flux,
                        audit_has_row,
                        audit_target,
                        audit_kind,
                        audit_active,
                        audit_rhs,
                        audit_predicted_rhs,
                        audit_residual,
                        audit_rank,
                        audit_condition,
                        audit_synthetic_coefficients,
                        audit_synthetic_expected,
                        audit_synthetic_valid,
                    ) = face_diagnostic
                    kind_names = {
                        CV_FACE_INTERIOR: "interior",
                        CV_FACE_PARTIAL: "partial",
                        CV_FACE_CUT_WALL: "cut-wall",
                        CV_FACE_PHYSICAL_BOUNDARY: "physical",
                    }
                    irregular_numerical_sum = float(
                        irregular_numerical_sum
                    )
                    irregular_exact_sum = float(irregular_exact_sum)
                    aggregate_volume = float(
                        np.asarray(volume)[top_index]
                    )
                    actual_integrated = (
                        float(np.asarray(actual)[top_index])
                        * aggregate_volume
                    )
                    reference_integrated = (
                        float(np.asarray(reference)[top_index])
                        * aggregate_volume
                    )
                    dense_inferred = (
                        actual_integrated
                        - irregular_numerical_sum
                    )
                    target_storage_dense_sum = float(
                        np.sum(dense_flux)
                    )
                    print(
                        "  top_face_balance aggregate_volume={:.6e} "
                        "actual_integrated={:.6e} "
                        "owner_dense={:.6e} target_storage_dense={:.6e} "
                        "routed_member_dense={:.6e} irregular={:.6e} "
                        "reference_integrated={:.6e} "
                        "reference_remainder={:.6e} "
                        "exact_irregular={:.6e} attached_rows={}/{}".format(
                            aggregate_volume,
                            actual_integrated,
                            dense_inferred,
                            target_storage_dense_sum,
                            dense_inferred - target_storage_dense_sum,
                            irregular_numerical_sum,
                            reference_integrated,
                            reference_integrated
                            - irregular_exact_sum,
                            irregular_exact_sum,
                            int(selected_count),
                            max_reported_rows,
                        )
                    )
                    if debug_operator_failures:
                        dense_exact_sum = float(np.sum(dense_exact_flux))
                        attached_numerical = (
                            dense_inferred + irregular_numerical_sum
                        )
                        exact_irregular_variant = (
                            dense_inferred + irregular_exact_sum
                        )
                        exact_dense_variant = (
                            dense_exact_sum + irregular_numerical_sum
                        )
                        exact_attached_variant = (
                            dense_exact_sum + irregular_exact_sum
                        )
                        print(
                            "  attached_face_substitutions (same target scatter): "
                            "numerical={:.6e} exact_irregular={:.6e} "
                            "exact_dense={:.6e} exact_all={:.6e} "
                            "reference={:.6e}".format(
                                attached_numerical / aggregate_volume,
                                exact_irregular_variant / aggregate_volume,
                                exact_dense_variant / aggregate_volume,
                                exact_attached_variant / aggregate_volume,
                                float(np.asarray(reference)[top_index]),
                            )
                        )
                        print(
                            "    integrated deltas irregular={:.6e} dense={:.6e} "
                            "all_attached={:.6e}; dense values exclude routed "
                            "merged-source storage contributions.".format(
                                irregular_exact_sum - irregular_numerical_sum,
                                dense_exact_sum - dense_inferred,
                                exact_attached_variant - attached_numerical,
                            )
                        )
                        if bool(audit_has_row):
                            synthetic_error = (
                                audit_synthetic_coefficients
                                - audit_synthetic_expected
                            )
                            print(
                                "  reconstruction_row_audit target={} rank={} "
                                "condition={:.6e} synthetic_quadratic_replay={} "
                                "max_error={:.6e}".format(
                                    tuple(int(value) for value in audit_target),
                                    int(audit_rank),
                                    float(audit_condition),
                                    bool(audit_synthetic_valid),
                                    float(np.max(np.abs(synthetic_error))),
                                )
                            )
                            equation_names = {
                                CV_RECONSTRUCTION_EQUATION_CELL: "cell",
                                CV_RECONSTRUCTION_EQUATION_DIRICHLET: "dirichlet",
                                CV_RECONSTRUCTION_EQUATION_REMOTE_CELL: "remote",
                            }
                            for equation in range(audit_active.shape[0]):
                                if not bool(audit_active[equation]):
                                    continue
                                print(
                                    "    equation[{}] kind={} rhs={:+.6e} "
                                    "predicted={:+.6e} residual={:+.6e}".format(
                                        equation,
                                        equation_names.get(
                                            int(audit_kind[equation]),
                                            str(int(audit_kind[equation])),
                                        ),
                                        float(audit_rhs[equation]),
                                        float(audit_predicted_rhs[equation]),
                                        float(audit_residual[equation]),
                                    )
                                )
                    axis_names = ("x", "theta", "zeta")
                    side_names = ("low", "high")
                    for axis in range(3):
                        for side in range(2):
                            print(
                                "    dense {}-{}: open={} "
                                "measure={:.6e} signed_flux={:.6e} "
                                "exact={:.6e} error={:.6e} "
                                "compact_owned={} "
                                "neighbor_reconstructed={}".format(
                                    axis_names[axis],
                                    side_names[side],
                                    bool(dense_open[axis, side]),
                                    float(dense_measure[axis, side]),
                                    float(dense_flux[axis, side]),
                                    float(
                                        dense_exact_flux[axis, side]
                                    ),
                                    float(
                                        dense_flux[axis, side]
                                        - dense_exact_flux[axis, side]
                                    ),
                                    bool(
                                        dense_compact_owned[
                                            axis,
                                            side,
                                        ]
                                    ),
                                    bool(
                                        dense_neighbor_reconstructed[
                                            axis,
                                            side,
                                        ]
                                    ),
                                )
                            )
                    for slot in range(max_reported_rows):
                        if not bool(slot_active[slot]):
                            continue
                        owners = tuple(
                            int(value)
                            for value in row_owners[slot]
                        )
                        quadrature_mask = (
                            row_quadrature_active[slot].astype(bool)
                        )
                        quadrature_error = np.where(
                            quadrature_mask,
                            np.abs(
                                row_quadrature_flux[slot]
                                - row_exact_quadrature_flux[slot]
                            ),
                            -np.inf,
                        )
                        worst_patch, worst_quadrature = np.unravel_index(
                            int(np.argmax(quadrature_error)),
                            quadrature_error.shape,
                        )
                        print(
                            "    irregular row={} kind={} sign={:+.0f} "
                            "owners=({},{},{})/({},{},{}) "
                            "plus={} remote={} flux={:.6e} "
                            "exact={:.6e} error={:.6e} active_q={}".format(
                                int(row_index[slot]),
                                kind_names.get(
                                    int(row_kind[slot]),
                                    str(int(row_kind[slot])),
                                ),
                                float(row_sign[slot]),
                                *owners,
                                bool(row_has_plus[slot]),
                                bool(row_has_remote[slot]),
                                float(row_flux[slot]),
                                float(row_exact_flux[slot]),
                                float(
                                    row_flux[slot]
                                    - row_exact_flux[slot]
                                ),
                                int(np.sum(quadrature_mask)),
                            )
                        )
                        if np.any(quadrature_mask):
                            point = row_points[
                                slot,
                                worst_patch,
                                worst_quadrature,
                            ]
                            print(
                                "      worst_q patch={} q={} "
                                "point=({:.6e},{:.6e},{:.6e}) "
                                "numerical={:.6e} exact={:.6e} "
                                "error={:.6e}".format(
                                    worst_patch,
                                    worst_quadrature,
                                    float(point[0]),
                                    float(point[1]),
                                    float(point[2]),
                                    float(
                                        row_quadrature_flux[
                                            slot,
                                            worst_patch,
                                            worst_quadrature,
                                        ]
                                    ),
                                    float(
                                        row_exact_quadrature_flux[
                                            slot,
                                            worst_patch,
                                            worst_quadrature,
                                        ]
                                    ),
                                    float(
                                        row_quadrature_flux[
                                            slot,
                                            worst_patch,
                                            worst_quadrature,
                                        ]
                                        - row_exact_quadrature_flux[
                                            slot,
                                            worst_patch,
                                            worst_quadrature,
                                        ]
                                    ),
                                )
                            )
                        if bool(row_transition_active[slot]):
                            print(
                                "      transition functional: valid={} "
                                "production_q_sum={:.6e} row_delta={:.6e} "
                                "exact_support_flux={:.6e} "
                                "support_gradient=({:+.6e},{:+.6e},{:+.6e})".format(
                                    bool(row_transition_valid[slot]),
                                    float(np.sum(row_quadrature_flux[slot])),
                                    float(
                                        np.sum(row_quadrature_flux[slot])
                                        - row_flux[slot]
                                    ),
                                    float(
                                        row_transition_exact_structured_flux[slot]
                                    ),
                                    float(row_transition_exact_gradient[slot, 0]),
                                    float(row_transition_exact_gradient[slot, 1]),
                                    float(row_transition_exact_gradient[slot, 2]),
                                )
                            )
                            for sample_slot in range(
                                row_transition_value.shape[1]
                            ):
                                coefficients = row_transition_coefficients[
                                    slot,
                                    sample_slot,
                                ]
                                if not np.any(coefficients):
                                    continue
                                exact_average = row_transition_exact_average[
                                    slot,
                                    sample_slot,
                                ]
                                value = row_transition_value[
                                    slot,
                                    sample_slot,
                                ]
                                gradient_contribution = (
                                    coefficients[1:] * value
                                )
                                exact_gradient_contribution = (
                                    coefficients[1:] * exact_average
                                )
                                print(
                                    "        support[{}]: storage={} center={} "
                                    "owner={} direct={} value={:.6e} exact_avg={:.6e} "
                                    "avg_error={:+.6e} coeff=(scalar={:+.6e}, "
                                    "grad=({:+.6e},{:+.6e},{:+.6e})) "
                                    "grad_contrib=({:+.6e},{:+.6e},{:+.6e}) "
                                    "exact_contrib=({:+.6e},{:+.6e},{:+.6e})".format(
                                        sample_slot,
                                        tuple(int(v) for v in row_transition_storage[slot, sample_slot]),
                                        tuple(int(v) for v in row_transition_center[slot, sample_slot]),
                                        tuple(int(v) for v in row_transition_owner[slot, sample_slot]),
                                        bool(row_transition_direct[slot, sample_slot]),
                                        float(value),
                                        float(exact_average),
                                        float(value - exact_average),
                                        float(coefficients[0]),
                                        float(coefficients[1]),
                                        float(coefficients[2]),
                                        float(coefficients[3]),
                                        float(gradient_contribution[0]),
                                        float(gradient_contribution[1]),
                                        float(gradient_contribution[2]),
                                        float(exact_gradient_contribution[0]),
                                        float(exact_gradient_contribution[1]),
                                        float(exact_gradient_contribution[2]),
                                    )
                                )
                                if not bool(
                                    row_transition_direct[slot, sample_slot]
                                ):
                                    owner_gradient = (
                                        row_transition_owner_gradient[
                                            slot,
                                            sample_slot,
                                        ]
                                    )
                                    exact_owner_gradient = (
                                        row_transition_owner_exact_gradient[
                                            slot,
                                            sample_slot,
                                        ]
                                    )
                                    print(
                                        "          owner polynomial: centroid={} "
                                        "order={} condition={:.3e} stored={:.6e} "
                                        "exact_centroid={:.6e} grad=({:+.6e},{:+.6e},{:+.6e}) "
                                        "exact_grad=({:+.6e},{:+.6e},{:+.6e})".format(
                                            tuple(
                                                float(v)
                                                for v in row_transition_owner_centroid[
                                                    slot,
                                                    sample_slot,
                                                ]
                                            ),
                                            int(
                                                row_transition_owner_order[
                                                    slot,
                                                    sample_slot,
                                                ]
                                            ),
                                            float(
                                                row_transition_owner_condition[
                                                    slot,
                                                    sample_slot,
                                                ]
                                            ),
                                            float(
                                                row_transition_owner_value[
                                                    slot,
                                                    sample_slot,
                                                ]
                                            ),
                                            float(
                                                row_transition_owner_exact_value[
                                                    slot,
                                                    sample_slot,
                                                ]
                                            ),
                                            float(owner_gradient[0]),
                                            float(owner_gradient[1]),
                                            float(owner_gradient[2]),
                                            float(exact_owner_gradient[0]),
                                            float(exact_owner_gradient[1]),
                                            float(exact_owner_gradient[2]),
                                        )
                                    )
                        if int(row_boundary_face_kind[slot]) == CV_FACE_CUT_WALL:
                            print(
                                "      cut-wall projected-flux diagnostics: "
                                "bc_kind={}".format(
                                    int(row_boundary_bc_kind[slot])
                                )
                            )
                            for patch in range(
                                row_quadrature_active.shape[1]
                            ):
                                for quadrature in range(
                                    row_quadrature_active.shape[2]
                                ):
                                    if not bool(
                                        row_quadrature_active[
                                            slot,
                                            patch,
                                            quadrature,
                                        ]
                                    ):
                                        continue
                                    projected_covector = (
                                        row_boundary_projected_covector[
                                            slot,
                                            patch,
                                            quadrature,
                                        ]
                                    )
                                    raw_gradient = row_boundary_raw_gradient[
                                        slot,
                                        patch,
                                        quadrature,
                                    ]
                                    applied_gradient = (
                                        row_boundary_applied_gradient[
                                            slot,
                                            patch,
                                            quadrature,
                                        ]
                                    )
                                    exact_gradient = (
                                        row_boundary_exact_gradient[
                                            slot,
                                            patch,
                                            quadrature,
                                        ]
                                    )
                                    normal_contra = row_boundary_normal_contra[
                                        slot,
                                        patch,
                                        quadrature,
                                    ]
                                    normal_covector = row_boundary_normal_covector[
                                        slot,
                                        patch,
                                        quadrature,
                                    ]
                                    print(
                                        "        patch={} q={} bc_value={:.6e} "
                                        "poly_value={:.6e} trace_residual={:+.6e} "
                                        "projected_covector=({:+.6e},{:+.6e},{:+.6e}) "
                                        "normal_contra=({:+.6e},{:+.6e},{:+.6e}) "
                                        "normal_covector=({:+.6e},{:+.6e},{:+.6e}) "
                                        "raw_grad=({:+.6e},{:+.6e},{:+.6e}) "
                                        "applied_grad=({:+.6e},{:+.6e},{:+.6e}) "
                                        "exact_grad=({:+.6e},{:+.6e},{:+.6e}) "
                                        "closure=(valid={}, axis={}, exact_input_dxi={:+.6e}) "
                                        "normal={:+.6e} exact_normal={:+.6e} "
                                        "projected_derivative={:+.6e} "
                                        "exact_projected_derivative={:+.6e} "
                                        "flux={:+.6e} exact_tangent_num_normal={:+.6e} "
                                        "num_tangent_exact_normal={:+.6e} exact_flux={:+.6e}".format(
                                            patch,
                                            quadrature,
                                            float(
                                                row_boundary_bc_value[
                                                    slot,
                                                    patch,
                                                    quadrature,
                                                ]
                                            ),
                                            float(
                                                row_boundary_polynomial_value[
                                                    slot,
                                                    patch,
                                                    quadrature,
                                                ]
                                            ),
                                            float(
                                                row_boundary_trace_residual[
                                                    slot,
                                                    patch,
                                                    quadrature,
                                                ]
                                            ),
                                            float(projected_covector[0]),
                                            float(projected_covector[1]),
                                            float(projected_covector[2]),
                                            float(normal_contra[0]),
                                            float(normal_contra[1]),
                                            float(normal_contra[2]),
                                            float(normal_covector[0]),
                                            float(normal_covector[1]),
                                            float(normal_covector[2]),
                                            float(raw_gradient[0]),
                                            float(raw_gradient[1]),
                                            float(raw_gradient[2]),
                                            float(applied_gradient[0]),
                                            float(applied_gradient[1]),
                                            float(applied_gradient[2]),
                                            float(exact_gradient[0]),
                                            float(exact_gradient[1]),
                                            float(exact_gradient[2]),
                                            bool(
                                                row_boundary_closure_valid[
                                                    slot,
                                                    patch,
                                                    quadrature,
                                                ]
                                            ),
                                            int(
                                                row_boundary_closure_axis[slot]
                                            ),
                                            float(
                                                row_boundary_exact_input_derivative[
                                                    slot,
                                                    patch,
                                                    quadrature,
                                                ]
                                            ),
                                            float(
                                                row_boundary_normal_derivative[
                                                    slot,
                                                    patch,
                                                    quadrature,
                                                ]
                                            ),
                                            float(
                                                row_boundary_exact_normal_derivative[
                                                    slot,
                                                    patch,
                                                    quadrature,
                                                ]
                                            ),
                                            float(
                                                np.dot(
                                                    projected_covector,
                                                    raw_gradient,
                                                )
                                            ),
                                            float(
                                                np.dot(
                                                    projected_covector,
                                                    exact_gradient,
                                                )
                                            ),
                                            float(
                                                row_quadrature_flux[
                                                    slot,
                                                    patch,
                                                    quadrature,
                                                ]
                                            ),
                                            float(
                                                row_boundary_exact_tangent_numerical_normal_flux[
                                                    slot,
                                                    patch,
                                                    quadrature,
                                                ]
                                            ),
                                            float(
                                                row_boundary_numerical_tangent_exact_normal_flux[
                                                    slot,
                                                    patch,
                                                    quadrature,
                                                ]
                                            ),
                                            float(
                                                row_exact_quadrature_flux[
                                                    slot,
                                                    patch,
                                                    quadrature,
                                                ]
                                            ),
                                        )
                                    )

            def full_rhs_kernel(
                state_owned,
                phi_owned,
                local_invariants,
                control_volume_geometry,
                stage_time,
            ):
                local_invariants = extract_local_shard_pytree(
                    local_invariants
                )
                control_volume_geometry = extract_local_shard_pytree(
                    control_volume_geometry
                )
                local_geometry_value = local_geometry(
                    control_volume_geometry
                )
                stage = shifted_mms._build_local_4field_stage_data(
                    local_invariants,
                    stage_time,
                    parameters=parameters,
                )
                stage = _with_shifted_torus_regular_radial_face_averages(
                    stage,
                    local_geometry_value,
                    stage_time,
                )
                source_owner = _project_local_mms_source_to_control_volumes(
                    local_geometry_value,
                    control_volume_geometry,
                    stage_time,
                    parameters,
                )
                cells = control_volume_geometry.cells
                source_storage = Fci4FieldState(
                    density=_expand_control_volume_owner_values(
                        source_owner.density,
                        cells,
                    ),
                    omega=_expand_control_volume_owner_values(
                        source_owner.omega,
                        cells,
                    ),
                    v_ion_parallel=_expand_control_volume_owner_values(
                        source_owner.v_ion_parallel,
                        cells,
                    ),
                    v_electron_parallel=_expand_control_volume_owner_values(
                        source_owner.v_electron_parallel,
                        cells,
                    ),
                )
                phi_storage = _expand_control_volume_owner_values(
                    phi_owned,
                    cells,
                )
                stage = dataclass_replace(
                    stage,
                    source_halo=inject_owned_state_to_halo(
                        source_storage,
                        domain.layout,
                    ),
                    phi_halo=inject_owned_field_to_halo(
                        phi_storage,
                        domain.layout,
                    ),
                )
                rhs = LocalShiftedTorus4FieldCutWallRhs(
                    geometry=local_geometry_value,
                    domain=domain,
                    halo_exchange=HaloExchange3D(),
                    topology_filler=topology_filler,
                    physical_ghost_filler=physical_ghost_filler,
                    parameters=parameters,
                    curvature_coefficients_owned=(
                        local_invariants.curvature_coefficients_owned
                    ),
                    face_projectors=(
                        local_invariants.face_projector_x,
                        local_invariants.face_projector_y,
                        local_invariants.face_projector_z,
                    ),
                    gmres_config=gmres_config,
                    global_shape=shape,
                    control_volume_geometry=control_volume_geometry,
                )
                actual, _phi = rhs.evaluate_stage(
                    state_owned,
                    stage,
                    phi_owned,
                    solve_phi=bool(check_phi_solve),
                )
                reference = (
                    _project_local_exact_time_derivative_to_control_volumes(
                        local_geometry_value,
                        control_volume_geometry,
                        stage_time,
                    )
                )
                source_roundtrip = Fci4FieldState(
                    density=_agglomerate_control_volume_average(
                        source_storage.density,
                        cells,
                    ),
                    omega=_agglomerate_control_volume_average(
                        source_storage.omega,
                        cells,
                    ),
                    v_ion_parallel=_agglomerate_control_volume_average(
                        source_storage.v_ion_parallel,
                        cells,
                    ),
                    v_electron_parallel=_agglomerate_control_volume_average(
                        source_storage.v_electron_parallel,
                        cells,
                    ),
                )
                active = cells.is_active_owner

                def max_active(value: jnp.ndarray) -> jnp.ndarray:
                    result = jnp.max(
                        jnp.where(active, jnp.abs(value), 0.0)
                    )
                    for mesh_axis_name in MESH_AXIS_NAMES:
                        result = lax.pmax(result, mesh_axis_name)
                    return result

                source_diagnostics = jnp.stack(
                    (
                        max_active(
                            source_owner.v_ion_parallel
                            - reference.v_ion_parallel
                        ),
                        max_active(
                            source_roundtrip.v_ion_parallel
                            - source_owner.v_ion_parallel
                        ),
                        max_active(
                            actual.v_ion_parallel
                            - source_roundtrip.v_ion_parallel
                        ),
                        max_active(
                            actual.v_ion_parallel
                            - reference.v_ion_parallel
                        ),
                    )
                )
                return actual, reference, source_diagnostics

            compiled_full_rhs = jax.jit(
                shard_map(
                    full_rhs_kernel,
                    mesh=mesh,
                    in_specs=(
                        state_spec,
                        field_spec,
                        invariant_spec,
                        control_volume_spec,
                        P(),
                    ),
                    out_specs=(state_spec, state_spec, P()),
                    check_rep=False,
                )
            )
            start = time_module.perf_counter()
            (
                actual_rhs_mesh,
                reference_rhs_mesh,
                source_diagnostics,
            ) = compiled_full_rhs(
                state_mesh,
                phi_mesh,
                invariants_mesh,
                control_volume_mesh,
                jnp.asarray(0.0, dtype=jnp.float64),
            )
            jax.block_until_ready(actual_rhs_mesh.density)
            elapsed = time_module.perf_counter() - start
            actual_rhs = shifted_mms._gather_state_from_mesh(
                actual_rhs_mesh
            )
            reference_rhs = shifted_mms._gather_state_from_mesh(
                reference_rhs_mesh
            )
            print(
                f"N={resolution} operator=full_rhs "
                f"phi_mode={'solved' if check_phi_solve else 'projected_exact'} "
                f"compile+run={elapsed:.3f}s"
            )
            source_diagnostics_host = np.asarray(
                jax.device_get(source_diagnostics),
                dtype=np.float64,
            )
            print(
                "  v_ion source consistency: "
                "source_vs_exact_t={:.6e} "
                "roundtrip_vs_source={:.6e} "
                "operator_without_source={:.6e} "
                "full_residual={:.6e}".format(
                    *source_diagnostics_host,
                )
            )
            for field_name in (
                "density",
                "omega",
                "v_ion_parallel",
                "v_electron_parallel",
            ):
                operator_name = f"full_rhs_{field_name}"
                statistics = _operator_category_statistics(
                    getattr(actual_rhs, field_name),
                    getattr(reference_rhs, field_name),
                    volume,
                    categories,
                )
                print(f"  field={field_name}")
                for category, (
                    l2,
                    linf,
                    relative,
                    count,
                ) in statistics.items():
                    print(
                        f"    {category:16s} count={count:8d} "
                        f"volume_L2={l2:.6e} Linf={linf:.6e} "
                        f"rel_L2={relative:.6e}"
                    )
                    records.setdefault(operator_name, {}).setdefault(
                        category,
                        [],
                    ).append((resolution, l2, linf))

            if not bool(check_phi_solve):
                print(f"N={resolution} phi algebraic solve skipped")
                continue

            def phi_solve_kernel(
                state_owned,
                phi_owned,
                local_invariants,
                control_volume_geometry,
                stage_time,
            ):
                local_invariants = extract_local_shard_pytree(
                    local_invariants
                )
                control_volume_geometry = extract_local_shard_pytree(
                    control_volume_geometry
                )
                local_geometry_value = local_geometry(
                    control_volume_geometry
                )
                phi_face_bc = regular_face_bc(
                    local_geometry_value,
                    stage_time,
                    "phi",
                )
                phi_boundary_bc = _control_volume_exact_boundary_bc(
                    control_volume_geometry,
                    stage_time,
                    "phi",
                )
                solver = LocalPerpLaplacianInverseSolver(
                    geometry=local_geometry_value,
                    domain=domain,
                    stencil_builder=(
                        build_local_conservative_stencil_from_field
                    ),
                    halo_exchange=HaloExchange3D(),
                    topology_filler=topology_filler,
                    physical_ghost_filler=physical_ghost_filler,
                    face_projectors=(
                        local_invariants.face_projector_x,
                        local_invariants.face_projector_y,
                        local_invariants.face_projector_z,
                    ),
                    control_volume_geometry=control_volume_geometry,
                    control_volume_boundary_bc=phi_boundary_bc,
                    face_bc=phi_face_bc,
                    config=gmres_config,
                )
                solution, info = solver(
                    -state_owned.omega,
                    guess_owned=phi_owned,
                    phi_lift_owned=phi_owned,
                    return_diagnostics=True,
                )
                return (
                    solution,
                    info.final_residual_rel_l2,
                    info.converged,
                    info.failed,
                    info.num_steps,
                    info.initial_residual_l2,
                    info.final_residual_l2,
                    info.rhs_l2,
                )

            compiled_phi_solve = jax.jit(
                shard_map(
                    phi_solve_kernel,
                    mesh=mesh,
                    in_specs=(
                        state_spec,
                        field_spec,
                        invariant_spec,
                        control_volume_spec,
                        P(),
                    ),
                    out_specs=(
                        field_spec,
                        P(),
                        P(),
                        P(),
                        P(),
                        P(),
                        P(),
                        P(),
                    ),
                    check_rep=False,
                )
            )
            (
                solved_phi,
                relative_residual,
                converged,
                phi_failed,
                phi_num_steps,
                phi_initial_residual,
                phi_final_residual,
                phi_rhs_l2,
            ) = compiled_phi_solve(
                state_mesh,
                phi_mesh,
                invariants_mesh,
                control_volume_mesh,
                jnp.asarray(0.0, dtype=jnp.float64),
            )
            jax.block_until_ready(solved_phi)
            relative_residual_value = float(
                np.asarray(jax.device_get(relative_residual))
            )
            converged_value = bool(
                np.asarray(jax.device_get(converged))
            )
            failed_value = bool(
                np.asarray(jax.device_get(phi_failed))
            )
            num_steps_value = int(
                np.asarray(jax.device_get(phi_num_steps))
            )
            initial_residual_value = float(
                np.asarray(jax.device_get(phi_initial_residual))
            )
            final_residual_value = float(
                np.asarray(jax.device_get(phi_final_residual))
            )
            rhs_l2_value = float(
                np.asarray(jax.device_get(phi_rhs_l2))
            )
            phi_residuals.append(
                (resolution, relative_residual_value)
            )
            print(
                f"N={resolution} phi algebraic residual="
                f"{relative_residual_value:.6e}, converged={converged_value}, "
                f"failed={failed_value}, steps={num_steps_value}, "
                f"initial={initial_residual_value:.6e}, "
                f"final={final_residual_value:.6e}, rhs_l2={rhs_l2_value:.6e}"
            )
            if (
                not np.isfinite(relative_residual_value)
                or relative_residual_value > 5.0e-5
            ):
                raise AssertionError(
                    "phi solve failed operator-convergence acceptance: "
                    f"N={resolution}, residual={relative_residual_value:.6e}, "
                    f"converged={converged_value}, failed={failed_value}"
                )

    order_results: dict[
        str,
        dict[str, tuple[float | None, float | None]],
    ] = {}
    failed_orders: list[str] = []
    for operator_name, category_records in records.items():
        order_results[operator_name] = {}
        for category, values in category_records.items():
            category_resolutions = [value[0] for value in values]
            exact_to_roundoff = bool(values) and all(
                np.isfinite(value[1])
                and np.isfinite(value[2])
                and abs(value[1]) <= 1.0e-12
                and abs(value[2]) <= 1.0e-12
                for value in values
            )
            l2_order = _fit_operator_order(
                category_resolutions,
                [value[1] for value in values],
            )
            linf_order = _fit_operator_order(
                category_resolutions,
                [value[2] for value in values],
            )
            order_results[operator_name][category] = (
                l2_order,
                linf_order,
            )
            l2_text = "n/a" if l2_order is None else f"{l2_order:.6f}"
            linf_text = (
                "n/a" if linf_order is None else f"{linf_order:.6f}"
            )
            if exact_to_roundoff:
                l2_text = "exact"
                linf_text = "exact"
            print(
                f"operator order {operator_name} {category}: "
                f"volume_L2={l2_text}, Linf={linf_text}"
            )
            if len(resolutions) >= 2 and category == "all_active":
                if (
                    not exact_to_roundoff
                    and (
                        l2_order is None
                        or l2_order < float(minimum_order)
                    )
                ):
                    failed_orders.append(
                        f"{operator_name}/{category} L2={l2_text}"
                    )
                if (
                    not exact_to_roundoff
                    and (
                        linf_order is None
                        or linf_order < float(minimum_order)
                    )
                ):
                    failed_orders.append(
                        f"{operator_name}/{category} Linf={linf_text}"
                    )
    if failed_orders:
        raise AssertionError(
            "operator convergence acceptance failed (minimum order "
            f"{float(minimum_order):.3f}): "
            + "; ".join(failed_orders)
        )
    return {
        "records": records,
        "orders": order_results,
        "phi_residuals": phi_residuals,
    }


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
    enable_agglomeration: bool = False,
    minimum_order: float | None = None,
) -> dict[str, object]:
    successful_resolutions: list[int] = []
    l2_errors: list[float] = []
    max_errors: list[float] = []
    per_resolution_stats: list[tuple[int, dict[str, tuple[float, float, float]]]] = []

    for resolution in resolutions:
        shape = _shape_from_resolution(int(resolution))
        assert_shape_divisible_by_shards(shape, shard_counts)
        geometry = shifted_mms.build_shifted_torus_4field_geometry(shape)
        stacked_control_volume_geometry = (
            _build_stacked_embedded_control_volume_geometry(
                global_shape=shape,
                shard_counts=shard_counts,
                halo_width=halo_width,
                enable_merging=enable_agglomeration,
            )
        )
        steps = _resolution_step_count(int(resolution), base_steps=base_steps)
        dt = float(final_time) / float(steps)
        print(
            f"Starting shifted_torus_4field_cutwall MMS run: resolution={int(resolution)}, "
            f"shard_counts={shard_counts}, steps={steps}, dt={dt:.6e}, "
            f"enable_agglomeration={enable_agglomeration}"
        )
        _print_control_volume_geometry_summary(
            stacked_control_volume_geometry
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
            enable_agglomeration=enable_agglomeration,
            stacked_control_volume_geometry=stacked_control_volume_geometry,
        )
        elapsed = time_module.perf_counter() - start
        exact_state, _exact_phi = _project_global_exact_state_to_control_volumes(
            geometry,
            stacked_control_volume_geometry,
            shard_counts=shard_counts,
            halo_width=halo_width,
            time=final_time,
        )
        control_volume_cells = _assemble_global_control_volume_cell_data(
            geometry.shape,
            stacked_control_volume_geometry,
            shard_counts=shard_counts,
        )
        active_mask = control_volume_cells["is_active_owner"]
        aggregate_volume = control_volume_cells["aggregate_volume"]
        solid_mask = ~active_mask
        abs_errors = [
            jnp.abs(final_state.density - exact_state.density),
            jnp.abs(final_state.omega - exact_state.omega),
            jnp.abs(final_state.v_ion_parallel - exact_state.v_ion_parallel),
            jnp.abs(final_state.v_electron_parallel - exact_state.v_electron_parallel),
        ]
        active_errors = [jnp.where(active_mask, error, 0.0) for error in abs_errors]
        weight_sum = jnp.maximum(jnp.sum(aggregate_volume), 1.0e-30)
        sumsq_error = sum(
            jnp.sum(aggregate_volume * jnp.square(error))
            for error in active_errors
        )
        mean_error = float(
            jnp.sqrt(
                sumsq_error
                / (weight_sum * float(len(active_errors)))
            )
        )
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
        per_field_stats = _volume_weighted_state_error_statistics(
            final_state,
            exact_state,
            aggregate_volume,
            active_mask,
        )
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
        if active_nonfinite or solid_nonfinite:
            raise AssertionError(
                "shifted-torus control-volume state contains nonfinite values: "
                f"N={int(resolution)}, active={active_nonfinite}, "
                f"inactive_or_source={solid_nonfinite}"
            )
        _print_state_error_statistics(f"N={int(resolution)} per-field final errors", per_field_stats)

    l2_order: float | None = None
    max_order: float | None = None
    per_field_orders: dict[str, tuple[float | None, float | None]] = {}
    per_field_exact_to_roundoff: dict[str, bool] = {}
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
        for field_name in (
            "density",
            "omega",
            "v_ion_parallel",
            "v_electron_parallel",
        ):
            field_l2 = np.asarray(
                [
                    statistics[field_name][0]
                    for _resolution, statistics in per_resolution_stats
                ],
                dtype=np.float64,
            )
            field_linf = np.asarray(
                [
                    statistics[field_name][1]
                    for _resolution, statistics in per_resolution_stats
                ],
                dtype=np.float64,
            )
            exact_to_roundoff = bool(field_l2.size) and bool(field_linf.size) and (
                bool(np.all(np.isfinite(field_l2)))
                and bool(np.all(np.isfinite(field_linf)))
                and bool(np.all(np.abs(field_l2) <= 1.0e-12))
                and bool(np.all(np.abs(field_linf) <= 1.0e-12))
            )
            per_field_exact_to_roundoff[field_name] = exact_to_roundoff
            field_l2_order: float | None = None
            field_linf_order: float | None = None
            if not exact_to_roundoff:
                field_l2_order = float(
                    -np.polyfit(
                        np.log(plotted_resolutions),
                        np.log(field_l2),
                        1,
                    )[0]
                )
                field_linf_order = float(
                    -np.polyfit(
                        np.log(plotted_resolutions),
                        np.log(field_linf),
                        1,
                    )[0]
                )
            per_field_orders[field_name] = (field_l2_order, field_linf_order)
            l2_text = "exact" if exact_to_roundoff else f"{field_l2_order:.6f}"
            linf_text = "exact" if exact_to_roundoff else f"{field_linf_order:.6f}"
            print(
                "shifted_torus_4field_cutwall "
                f"{field_name} orders: volume_L2={l2_text}, "
                f"active_owner_Linf={linf_text}"
            )
        if minimum_order is not None:
            failed_orders = [
                (
                    f"{field_name}: volume_L2="
                    f"{'n/a' if orders[0] is None else f'{orders[0]:.6f}'}, "
                    f"active_owner_Linf="
                    f"{'n/a' if orders[1] is None else f'{orders[1]:.6f}'}"
                )
                for field_name, orders in per_field_orders.items()
                if (
                    not per_field_exact_to_roundoff[field_name]
                    and (
                        orders[0] is None
                        or orders[1] is None
                        or not np.isfinite(orders[0])
                        or not np.isfinite(orders[1])
                        or orders[0] < float(minimum_order)
                        or orders[1] < float(minimum_order)
                    )
                )
            ]
            if failed_orders:
                raise AssertionError(
                    "shifted-torus convergence acceptance failed "
                    f"(minimum order {float(minimum_order):.3f}): "
                    + "; ".join(failed_orders)
                )
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
        "per_field_orders": per_field_orders,
    }



def _print_runtime_info() -> None:
    print("=" * 80)
    print("JAX runtime")
    print("=" * 80)
    print(f"default backend: {jax.default_backend()}")
    print(f"local_device_count: {jax.local_device_count()}")
    print(f"compilation_cache_dir: {_JAX_COMPILATION_CACHE_DIR}")
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
    parser.add_argument(
        "--minimum-order",
        type=float,
        default=1.8,
        help=(
            "Minimum accepted per-field volume-L2 and active-owner Linf "
            "order for operator and full convergence sweeps."
        ),
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument(
        "--operator-convergence-only",
        action="store_true",
        help=(
            "Run separate unified control-volume spatial operator kernels and "
            "skip the RK convergence sweep."
        ),
    )
    parser.add_argument(
        "--skip-operator-phi-solve",
        action="store_true",
        help=(
            "Diagnostic-only: use projected exact phi in the full-RHS kernel "
            "and skip the separate phi inversion check while retaining all "
            "spatial operator kernels."
        ),
    )
    parser.add_argument(
        "--debug-operator-failures",
        action="store_true",
        help=(
            "Print focused radial-boundary, compact-face, and reconstruction "
            "failure attribution diagnostics during --operator-convergence-only."
        ),
    )
    parser.add_argument(
        "--enable-agglomeration",
        action="store_true",
        help=(
            "Merge sub-threshold fluid cut cells into a face-connected "
            "control-volume owner."
        ),
    )
    parser.add_argument("--skip-runtime-info", action="store_true")
    args = parser.parse_args()

    if not args.skip_runtime_info:
        _print_runtime_info()
    if bool(args.operator_convergence_only):
        run_shifted_torus_control_volume_operator_convergence(
            resolutions=[int(value) for value in args.resolutions],
            shard_counts=tuple(int(value) for value in args.shard_counts),
            halo_width=int(args.halo_width),
            rho_star_value=float(args.rho_star),
            enable_agglomeration=bool(args.enable_agglomeration),
            minimum_order=float(args.minimum_order),
            check_phi_solve=not bool(args.skip_operator_phi_solve),
            debug_operator_failures=bool(args.debug_operator_failures),
        )
        return
    if bool(args.debug_operator_failures):
        parser.error(
            "--debug-operator-failures requires --operator-convergence-only"
        )
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
        enable_agglomeration=bool(args.enable_agglomeration),
        minimum_order=float(args.minimum_order),
    )


if __name__ == "__main__":
    main()
