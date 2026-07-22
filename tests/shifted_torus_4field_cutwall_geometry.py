"""Closed-box embedded-control-volume geometry for shifted-torus MMS tests.

The module has no ``test_`` prefix so pytest collects only the assertions and
CLI wrappers in the public test module.
"""

from __future__ import annotations

from dataclasses import replace as dataclass_replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from drbx.geometry import (
    LocalCellVolumeGeometry3D,
    LocalControlVolumeCellGeometry3D,
    LocalFciGeometry3D,
    LocalRegularFaceGeometry3D,
    build_local_control_volume_cell_geometry,
)
from drbx.geometry.fci_control_volumes import (
    GlobalControlVolumeTopology3D,
    LocalControlVolumeGeometry3D,
    build_global_control_volume_topology,
    compile_local_control_volume_geometry,
)
from drbx.native import precompute_local_moment_reconstruction
from drbx.native.fci_boundaries import (
    CV_FACE_CUT_WALL,
    CV_FACE_INTERIOR,
    CV_FACE_PARTIAL,
    CV_FACE_PHYSICAL_BOUNDARY,
    LocalControlVolumeFaceRows3D,
    LocalEmbeddedControlVolumeGeometry3D,
    LocalMomentReconstruction3D,
    LocalRegularBoundaryMomentClosure3D,
)


_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))


import shifted_torus_4field_mms_helpers as shifted_mms  # noqa: E402
from mms_domain_decomp_helpers import (  # noqa: E402
    build_shifted_torus_local_geometry,
    stack_local_shard_pytree,
)


MESH_AXIS_NAMES = ("x", "y", "z")


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
    global_topology: GlobalControlVolumeTopology3D | None = None,
    local_topology: LocalControlVolumeGeometry3D | None = None,
    canonical_compact_face_ids: set[int] | None = None,
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
    global_face_lookup: dict[tuple[int, int, int, int], int] = {}
    evaluator_face_ids: set[int] | None = None
    if global_topology is not None:
        if local_topology is None:
            raise ValueError("global face topology needs a local shard compilation")
        global_face_lookup = {
            (int(axis), *(int(v) for v in storage)): int(face_id)
            for face_id, axis, storage in zip(
                global_topology.face_id,
                global_topology.face_axis,
                global_topology.face_storage_index,
            )
        }
        evaluator_face_ids = set(int(value) for value in local_topology.local_face_id)
        shard_extent = tuple(
            global_topology.shape[axis] // local_topology.shard_counts[axis]
            for axis in range(3)
        )
        shard_start = tuple(
            local_topology.shard_index[axis] * shard_extent[axis]
            for axis in range(3)
        )

        def global_face_id(axis: int, face: tuple[int, int, int]) -> int:
            storage = [shard_start[d] + int(face[d]) for d in range(3)]
            # Global topology stores periodic seams at the low image.
            if axis in (1, 2) and storage[axis] == global_topology.shape[axis]:
                storage[axis] = 0
            return global_face_lookup.get((axis, *storage), -1)
    else:
        def global_face_id(axis: int, face: tuple[int, int, int]) -> int:
            return -1

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

    if (
        global_topology is not None
        and local_topology is not None
        and canonical_compact_face_ids is not None
        and evaluator_face_ids is not None
    ):
        # A shard halo can discover fewer compact candidates than the global
        # build.  Insert every globally selected face owned by this evaluator
        # shard, converting the canonical periodic low seam to the local high
        # image when its minus owner lives on the final periodic shard.
        records = {
            int(face_id): (int(axis), tuple(int(v) for v in storage))
            for face_id, axis, storage in zip(
                global_topology.face_id,
                global_topology.face_axis,
                global_topology.face_storage_index,
            )
        }
        for face_id in evaluator_face_ids & canonical_compact_face_ids:
            axis, storage = records[face_id]
            local_face = [storage[d] - shard_start[d] for d in range(3)]
            if axis in (1, 2) and storage[axis] == 0 and shard_start[axis] > 0:
                local_face[axis] = shape[axis]
            tangential_axes = [d for d in range(3) if d != axis]
            if not all(0 <= local_face[d] < shape[d] for d in tangential_axes):
                continue
            if 0 <= local_face[axis] <= shape[axis]:
                face_candidates.add((axis, *local_face))

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
        global_face_id: int = -1,
        remote_residual_halo: tuple[int, int, int] | None = None,
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
                "global_face_id": global_face_id,
                "remote_residual_halo": remote_residual_halo,
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
        row_global_face_id = global_face_id(axis, face_index)
        # Keep dense masks closed on every shard, but only the canonical
        # evaluator shard receives the compact logical-face row.
        if (
            row_global_face_id >= 0
            and (
                (evaluator_face_ids is not None and row_global_face_id not in evaluator_face_ids)
                or (canonical_compact_face_ids is not None and row_global_face_id not in canonical_compact_face_ids)
            )
        ):
            open_masks[axis][face_index] = False
            continue
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
                    global_face_id=row_global_face_id,
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
                global_face_id=row_global_face_id,
                remote_residual_halo=tuple(remote_halo),
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
            global_face_id=row_global_face_id,
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
    global_face_ids = np.full(row_shape, -1, dtype=np.int64)
    has_remote_residual = np.zeros(row_shape, dtype=bool)
    remote_residual_halo_i = np.zeros(row_shape, dtype=np.int32)
    remote_residual_halo_j = np.zeros(row_shape, dtype=np.int32)
    remote_residual_halo_k = np.zeros(row_shape, dtype=np.int32)
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
        global_face_ids[row_index] = int(row.get("global_face_id", -1))
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
        residual_halo = row.get("remote_residual_halo")
        if residual_halo is not None:
            has_remote_residual[row_index] = True
            remote_residual_halo_i[row_index], remote_residual_halo_j[row_index], remote_residual_halo_k[row_index] = residual_halo
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
        global_face_id=jnp.asarray(global_face_ids),
        has_remote_residual=jnp.asarray(has_remote_residual),
        remote_residual_halo_i=jnp.asarray(remote_residual_halo_i),
        remote_residual_halo_j=jnp.asarray(remote_residual_halo_j),
        remote_residual_halo_k=jnp.asarray(remote_residual_halo_k),
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


def _build_global_closed_box_control_volume_topology(
    *, global_shape: tuple[int, int, int], halo_width: int
) -> tuple[GlobalControlVolumeTopology3D, tuple[np.ndarray, ...]]:
    """Build the one canonical host topology for every shifted-torus shard."""
    geometry = build_shifted_torus_local_geometry(
        global_shape, halo_width, global_shape=global_shape, shard_index=(0, 0, 0),
        x_min=shifted_mms.x_min, x_max=shifted_mms.x_max, r0=shifted_mms.r0,
        alpha_value=shifted_mms.alpha_value, iota=shifted_mms.iota,
        c_phi=shifted_mms.c_phi, sigma=shifted_mms.sigma,
    )
    raw_volume, centroid, second, third, full_volume = _closed_box_fluid_moments_3point(geometry)
    grids = (geometry.grid.x, geometry.grid.y, geometry.grid.z)
    axis_faces = tuple(np.asarray(grid.faces_owned, dtype=np.float64) for grid in grids)
    measures: list[np.ndarray] = []
    for axis in range(3):
        face_shape = list(global_shape); face_shape[axis] += 1
        measure = np.zeros(tuple(face_shape), dtype=np.float64)
        for face in np.ndindex(*face_shape):
            tangential = tuple((float(axis_faces[t][face[t]]), float(axis_faces[t][face[t] + 1])) for t in range(3) if t != axis)
            for rectangle in _open_face_rectangles_numpy(axis=axis, face_coordinate=float(axis_faces[axis][face[axis]]), tangential_bounds=tangential):
                points, area_weight = _face_patch_quadrature_numpy(axis=axis, face_coordinate=float(axis_faces[axis][face[axis]]), rectangle=rectangle, orientation=1.0)
                measure[face] += float(np.sum(_shifted_torus_metric_payload_numpy(points)[0] * np.linalg.norm(area_weight, axis=-1)))
        measures.append(measure)
    x, y, z = np.meshgrid(*(np.asarray(grid.centers_owned, dtype=np.float64) for grid in grids), indexing="ij")
    bounds = _box_bounds()
    center_in_solid = ((x > bounds[0][0]) & (x < bounds[0][1]) & (y > bounds[1][0]) & (y < bounds[1][1]) & (z > bounds[2][0]) & (z < bounds[2][1]))
    floor = 1.0e-14 * max(float(np.max(full_volume)), 1.0)
    fraction = raw_volume / np.maximum(full_volume, 1.0e-30)
    # Legacy selection is ``fraction < .5 OR center_in_solid``.
    fraction = np.where(center_in_solid & (raw_volume > floor), 0.0, fraction)
    topology = build_global_control_volume_topology(
        raw_volume=raw_volume, raw_centroid=centroid, raw_second_moment=second, raw_third_moment=third,
        fluid_volume_fraction=fraction, face_open_measure=tuple(measures), periodic_axes=(False, True, True),
        coordinate_periods=(float(shifted_mms.x_max - shifted_mms.x_min), 2.0 * np.pi, 2.0 * np.pi),
        positive_volume_floor=floor,
    )
    return topology, (raw_volume, centroid, second, third)


def _lower_global_control_volume_cells(geometry: LocalFciGeometry3D, local: LocalControlVolumeGeometry3D) -> LocalControlVolumeCellGeometry3D:
    """Lower canonical data only when this legacy runtime can represent it."""
    if np.any(local.owner_is_remote):
        raise NotImplementedError("legacy LocalControlVolumeCellGeometry3D cannot lower remote aggregate owners")
    owner = local.owner_local_index
    active = local.local_active_owner
    return LocalControlVolumeCellGeometry3D(
        layout=geometry.layout, owner_i=jnp.asarray(owner[..., 0]), owner_j=jnp.asarray(owner[..., 1]), owner_k=jnp.asarray(owner[..., 2]),
        is_merged_source=jnp.asarray(local.local_merge_source), is_active_owner=jnp.asarray(active),
        is_aggregate_target=jnp.asarray(active & (local.local_received_source_count > 0)),
        received_source_count=jnp.asarray(local.local_received_source_count), member_count=jnp.asarray(local.local_member_count),
        raw_volume=jnp.asarray(local.local_raw_volume), aggregate_volume=jnp.asarray(local.local_aggregate_volume),
        raw_centroid=jnp.asarray(local.local_raw_centroid), centroid=jnp.asarray(local.local_aggregate_centroid),
        raw_second_moment=jnp.asarray(local.local_raw_second_moment), second_moment=jnp.asarray(local.local_aggregate_second_moment),
        raw_third_moment=jnp.asarray(local.local_raw_third_moment), third_moment=jnp.asarray(local.local_aggregate_third_moment),
    )


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

    Compact-face flux reconstruction needs polynomial support on both sides
    of the irregular band. Reconstructing only the owner directly touching a
    wall leaves that support incomplete one logical cell away.
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


def _compact_face_reconstruction_owner_mask(
    cells: LocalControlVolumeCellGeometry3D,
    irregular_faces: LocalControlVolumeFaceRows3D,
) -> np.ndarray:
    """Return active local owners attached to every compact face.

    Remote neighbors are supplied by the reconstruction halo exchange; this
    mask deliberately marks only local minus/plus owners.
    """

    result = np.zeros(cells.shape, dtype=bool)
    active = np.asarray(irregular_faces.active, dtype=bool)
    has_plus = np.asarray(irregular_faces.has_plus_owner, dtype=bool)
    for row in np.flatnonzero(active):
        minus = (
            int(np.asarray(irregular_faces.minus_owner_i)[row]),
            int(np.asarray(irregular_faces.minus_owner_j)[row]),
            int(np.asarray(irregular_faces.minus_owner_k)[row]),
        )
        result[minus] = True
        if has_plus[row]:
            plus = (
                int(np.asarray(irregular_faces.plus_owner_i)[row]),
                int(np.asarray(irregular_faces.plus_owner_j)[row]),
                int(np.asarray(irregular_faces.plus_owner_k)[row]),
            )
            result[plus] = True
    return result & np.asarray(cells.is_active_owner, dtype=bool)


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
    global_topology: GlobalControlVolumeTopology3D | None = None,
    local_topology: LocalControlVolumeGeometry3D | None = None,
    canonical_compact_face_ids: set[int] | None = None,
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
        global_topology=global_topology,
        local_topology=local_topology,
        canonical_compact_face_ids=canonical_compact_face_ids,
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
        global_topology=global_topology,
        local_topology=local_topology,
        canonical_compact_face_ids=canonical_compact_face_ids,
    )
    reconstruction_owner_mask = _dilate_reconstruction_owner_mask(
        cells,
        reconstruction_owner_mask
        | _compact_face_reconstruction_owner_mask(cells, irregular_faces),
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
    compact_rows = np.flatnonzero(
        np.asarray(irregular_faces.active, dtype=bool)
    )
    target_row = np.asarray(
        reconstruction.target_row_for_cell,
        dtype=np.int32,
    )
    has_plus = np.asarray(irregular_faces.has_plus_owner, dtype=bool)
    for row in compact_rows:
        owners = [
            (
                int(np.asarray(irregular_faces.minus_owner_i)[row]),
                int(np.asarray(irregular_faces.minus_owner_j)[row]),
                int(np.asarray(irregular_faces.minus_owner_k)[row]),
            )
        ]
        if has_plus[row]:
            owners.append(
                (
                    int(np.asarray(irregular_faces.plus_owner_i)[row]),
                    int(np.asarray(irregular_faces.plus_owner_j)[row]),
                    int(np.asarray(irregular_faces.plus_owner_k)[row]),
                )
            )
        if any(target_row[owner] < 0 for owner in owners):
            raise ValueError(
                "each local compact-face owner must have a "
                "cubic reconstruction row"
            )
    return LocalEmbeddedControlVolumeGeometry3D(
        cells=cells,
        regular_faces=regular_faces,
        irregular_faces=irregular_faces,
        reconstruction=reconstruction,
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
        global_face_id=row_pad(rows.global_face_id, -1),
        has_remote_residual=row_pad(rows.has_remote_residual, False),
        remote_residual_halo_i=row_pad(rows.remote_residual_halo_i),
        remote_residual_halo_j=row_pad(rows.remote_residual_halo_j),
        remote_residual_halo_k=row_pad(rows.remote_residual_halo_k),
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
    """Pad only live compact-face and reconstruction payloads for SPMD."""

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
        centroid_J=control_volume_geometry.centroid_J,
        centroid_g_cov=control_volume_geometry.centroid_g_cov,
        centroid_B_contra=control_volume_geometry.centroid_B_contra,
        centroid_Bmag=control_volume_geometry.centroid_Bmag,
        centroid_curvature=control_volume_geometry.centroid_curvature,
        regular_boundary_closure=control_volume_geometry.regular_boundary_closure,
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
    global_topology: GlobalControlVolumeTopology3D | None = None
    global_raw: tuple[np.ndarray, ...] | None = None
    canonical_compact_face_ids: set[int] | None = None
    if enable_merging:
        global_topology, global_raw = _build_global_closed_box_control_volume_topology(global_shape=global_shape, halo_width=halo_width)
        # Define compact logical interfaces once on the unsplit global
        # geometry.  Local face discovery can be more conservative near a
        # shard halo; this set prevents those extra interfaces from changing
        # the canonical compact/dense partition.
        global_geometry = build_shifted_torus_local_geometry(
            global_shape, halo_width, global_shape=global_shape, shard_index=(0, 0, 0),
            x_min=shifted_mms.x_min, x_max=shifted_mms.x_max, r0=shifted_mms.r0,
            alpha_value=shifted_mms.alpha_value, iota=shifted_mms.iota,
            c_phi=shifted_mms.c_phi, sigma=shifted_mms.sigma,
        )
        whole_topology = compile_local_control_volume_geometry(
            global_topology, shard_index=(0, 0, 0), shard_counts=(1, 1, 1),
            raw_volume=global_raw[0], raw_centroid=global_raw[1],
            raw_second_moment=global_raw[2], raw_third_moment=global_raw[3],
        )
        whole_cells = _lower_global_control_volume_cells(global_geometry, whole_topology)
        whole_bundle = _build_closed_box_embedded_control_volume_geometry(
            global_geometry, enable_merging=True, cells=whole_cells,
            global_topology=global_topology, local_topology=whole_topology,
        )
        whole_ids = np.asarray(whole_bundle.irregular_faces.global_face_id, dtype=np.int64)
        whole_active = np.asarray(whole_bundle.irregular_faces.active, dtype=bool)
        canonical_compact_face_ids = set(int(value) for value in whole_ids[whole_active & (whole_ids >= 0)])
    local_geometry_and_cells: dict[
        tuple[int, int, int],
        tuple[LocalFciGeometry3D, LocalControlVolumeCellGeometry3D, LocalControlVolumeGeometry3D | None],
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
                if global_topology is None:
                    cells = _build_closed_box_control_volume_cells(local_geometry, enable_merging=False)
                    local_topology = None
                else:
                    local_topology = compile_local_control_volume_geometry(global_topology, shard_index=shard_index, shard_counts=shard_counts, raw_volume=global_raw[0], raw_centroid=global_raw[1], raw_second_moment=global_raw[2], raw_third_moment=global_raw[3])
                    cells = _lower_global_control_volume_cells(local_geometry, local_topology)
                local_geometry_and_cells[shard_index] = (local_geometry, cells, local_topology)

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
    for shard_index, (local_geometry, cells, local_topology) in local_geometry_and_cells.items():
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
                remote_geometry, remote_cells, _remote_topology = local_geometry_and_cells[remote_index]
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
                global_topology=global_topology,
                local_topology=local_topology,
                canonical_compact_face_ids=canonical_compact_face_ids,
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
