"""Boundary-functional constraints for field-independent cubic reconstruction.

The host-side routines in this module compile a weighted polynomial fit and a
linear boundary relation into two fixed maps,

``coefficients = owner_map @ owner_data + boundary_map @ boundary_data``.

Geometry preparation contains only polynomial evaluation rows, metric-derived
normal/tangential rows, and physical surface measures.  Runtime boundary data
remain separate, so a sidecar cannot accidentally contain a manufactured
solution or a particular wall-law value.  The runtime application is a pair of
small dense matrix products and is therefore suitable for eager JAX, ``jit``
and automatic differentiation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np


BOUNDARY_FUNCTIONAL_RECONSTRUCTION_VERSION = (
    "drbx.boundary-functional-reconstruction-v1"
)


def half_open_periodic_patch_index(
    faces: np.ndarray, value: float, period: float
) -> int:
    """Return the deterministic half-open patch containing a periodic value."""

    face_array = np.asarray(faces, dtype=np.float64)
    if (
        face_array.ndim != 1
        or face_array.size < 2
        or not np.all(np.isfinite(face_array))
        or np.any(np.diff(face_array) <= 0.0)
    ):
        raise ValueError("faces must be a finite strictly increasing rank-1 array")
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("period must be finite and positive")
    lower = float(face_array[0])
    wrapped = lower + (float(value) - lower) % float(period)
    if np.isclose(
        wrapped,
        face_array[-1],
        rtol=0.0,
        atol=16 * np.finfo(np.float64).eps,
    ):
        wrapped = lower
    index = int(np.searchsorted(face_array, wrapped, side="right") - 1)
    return min(max(index, 0), face_array.size - 2)


def _as_float64(name: str, value: np.ndarray, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite rank-{ndim} array")
    return array


@dataclass(frozen=True)
class BoundaryFunctionalGeometry:
    """Geometry-only coefficient rows on one physical-wall patch.

    ``gradient_rows[q, alpha, :]`` maps polynomial coefficients to logical
    derivatives at wall node ``q``.  ``normal_covector`` is the oriented
    physical-normal derivative covector in logical coordinates.  The two
    tangential rows are the derivatives along the wall coordinates; they are
    retained for diagnostics and future coupled boundary relations.
    """

    value_rows: np.ndarray
    gradient_rows: np.ndarray
    normal_covector: np.ndarray
    normal_derivative_rows: np.ndarray
    tangential_derivative_rows: np.ndarray
    surface_measure: np.ndarray
    moment_modes: np.ndarray
    version: str = BOUNDARY_FUNCTIONAL_RECONSTRUCTION_VERSION

    def __post_init__(self) -> None:
        value = _as_float64("value_rows", self.value_rows, 2)
        gradient = _as_float64("gradient_rows", self.gradient_rows, 3)
        normal = _as_float64("normal_covector", self.normal_covector, 2)
        normal_rows = _as_float64(
            "normal_derivative_rows", self.normal_derivative_rows, 2
        )
        tangent_rows = _as_float64(
            "tangential_derivative_rows", self.tangential_derivative_rows, 3
        )
        measure = _as_float64("surface_measure", self.surface_measure, 1)
        modes = _as_float64("moment_modes", self.moment_modes, 2)
        node_count, coefficient_count = value.shape
        if gradient.shape != (node_count, 3, coefficient_count):
            raise ValueError("gradient_rows must have shape (nodes,3,coefficients)")
        if normal.shape != (node_count, 3):
            raise ValueError("normal_covector must have shape (nodes,3)")
        if normal_rows.shape != value.shape:
            raise ValueError("normal_derivative_rows must match value_rows")
        if tangent_rows.shape != (node_count, 2, coefficient_count):
            raise ValueError(
                "tangential_derivative_rows must have shape (nodes,2,coefficients)"
            )
        if measure.shape != (node_count,) or np.any(measure <= 0.0):
            raise ValueError("surface_measure must be positive with one entry per node")
        if modes.shape[0] != node_count or modes.shape[1] < 1:
            raise ValueError("moment_modes must have one or more columns")
        if self.version != BOUNDARY_FUNCTIONAL_RECONSTRUCTION_VERSION:
            raise ValueError("unsupported boundary-functional geometry version")


@dataclass(frozen=True)
class BoundaryRelation:
    """Linear, dynamic boundary relation ``constraint_rows c = rhs_map b``."""

    constraint_rows: np.ndarray
    rhs_map: np.ndarray
    labels: tuple[str, ...]
    functional: Literal["value", "normal_derivative"]
    version: str = BOUNDARY_FUNCTIONAL_RECONSTRUCTION_VERSION

    def __post_init__(self) -> None:
        rows = _as_float64("constraint_rows", self.constraint_rows, 2)
        rhs = _as_float64("rhs_map", self.rhs_map, 2)
        if rows.shape[0] != rhs.shape[0] or len(self.labels) != rows.shape[0]:
            raise ValueError("boundary relation row counts are inconsistent")
        if self.functional not in ("value", "normal_derivative"):
            raise ValueError("unsupported boundary functional")
        if self.version != BOUNDARY_FUNCTIONAL_RECONSTRUCTION_VERSION:
            raise ValueError("unsupported boundary-relation version")


@dataclass(frozen=True)
class PreparedBoundaryReconstruction:
    """Fixed maps and rank diagnostics for one constrained polynomial fit."""

    owner_map: np.ndarray
    boundary_map: np.ndarray
    observation_rank: int
    constraint_rank: int
    reduced_rank: int
    observation_condition: float
    constraint_condition: float
    reduced_condition: float
    version: str = BOUNDARY_FUNCTIONAL_RECONSTRUCTION_VERSION

    def __post_init__(self) -> None:
        owner = _as_float64("owner_map", self.owner_map, 2)
        boundary = _as_float64("boundary_map", self.boundary_map, 2)
        if owner.shape[0] != boundary.shape[0]:
            raise ValueError("owner_map and boundary_map coefficient counts differ")
        if self.version != BOUNDARY_FUNCTIONAL_RECONSTRUCTION_VERSION:
            raise ValueError("unsupported prepared-reconstruction version")


def build_boundary_functional_geometry(
    *,
    value_rows: np.ndarray,
    gradient_rows: np.ndarray,
    g_contravariant: np.ndarray,
    jacobian: np.ndarray,
    logical_quadrature_weight: np.ndarray,
    tangential_coordinates: np.ndarray,
    outward_sign: float = 1.0,
    wall_axis: int = 0,
) -> BoundaryFunctionalGeometry:
    """Build metric-aware wall rows and the ``{1,s_theta,s_eta}`` modes.

    The physical-normal derivative is
    ``sign*g^{wall_axis,alpha} d_alpha/sqrt(g^{wall_axis,wall_axis})`` and
    the physical surface measure is
    ``|J|*sqrt(g^{wall_axis,wall_axis}) d(theta)d(eta)``.
    """

    values = _as_float64("value_rows", value_rows, 2)
    gradients = _as_float64("gradient_rows", gradient_rows, 3)
    metric = _as_float64("g_contravariant", g_contravariant, 3)
    jac = _as_float64("jacobian", jacobian, 1)
    quadrature = _as_float64(
        "logical_quadrature_weight", logical_quadrature_weight, 1
    )
    coordinates = _as_float64("tangential_coordinates", tangential_coordinates, 2)
    node_count = values.shape[0]
    if gradients.shape[:2] != (node_count, 3) or gradients.shape[2] != values.shape[1]:
        raise ValueError("gradient_rows must have shape (nodes,3,coefficients)")
    if metric.shape != (node_count, 3, 3):
        raise ValueError("g_contravariant must have shape (nodes,3,3)")
    if jac.shape != (node_count,) or quadrature.shape != (node_count,):
        raise ValueError("jacobian and quadrature weights require one value per node")
    if coordinates.shape != (node_count, 2):
        raise ValueError("tangential_coordinates must have shape (nodes,2)")
    wall_axis = int(wall_axis)
    if wall_axis not in (0, 1, 2):
        raise ValueError("wall_axis must be 0, 1, or 2")
    diagonal = metric[:, wall_axis, wall_axis]
    if np.any(diagonal <= 0.0) or not np.isfinite(outward_sign) or outward_sign == 0.0:
        raise ValueError("wall metric and outward orientation must be nondegenerate")
    normal = float(np.sign(outward_sign)) * metric[:, wall_axis, :] / np.sqrt(diagonal)[:, None]
    normal_rows = np.einsum("qa,qac->qc", normal, gradients)
    tangent_axes = [axis for axis in range(3) if axis != wall_axis]
    centered = coordinates - np.average(
        coordinates, axis=0, weights=np.abs(jac) * np.sqrt(diagonal) * quadrature
    )
    scale = np.max(np.abs(centered), axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    modes = np.column_stack((np.ones(node_count), centered / scale))
    return BoundaryFunctionalGeometry(
        value_rows=values,
        gradient_rows=gradients,
        normal_covector=normal,
        normal_derivative_rows=normal_rows,
        tangential_derivative_rows=gradients[:, tangent_axes, :],
        surface_measure=np.abs(jac) * np.sqrt(diagonal) * quadrature,
        moment_modes=modes,
    )


def build_moment_boundary_relation(
    geometry: BoundaryFunctionalGeometry,
    *,
    functional: Literal["value", "normal_derivative"],
) -> BoundaryRelation:
    """Integrate three low-order boundary moments into a linear relation.

    Runtime ``boundary_data`` are the corresponding normalized moments.  For
    homogeneous data this is simply a length-three zero vector.
    """

    rows = (
        geometry.value_rows
        if functional == "value"
        else geometry.normal_derivative_rows
        if functional == "normal_derivative"
        else None
    )
    if rows is None:
        raise ValueError("functional must be 'value' or 'normal_derivative'")
    weighted_modes = geometry.surface_measure[:, None] * geometry.moment_modes
    normalization = np.sum(weighted_modes * geometry.moment_modes, axis=0)
    if np.any(normalization <= 0.0):
        raise ValueError("boundary moment normalization is singular")
    constraint = np.einsum("qm,qc->mc", weighted_modes, rows) / normalization[:, None]
    labels = tuple(f"{functional}:{name}" for name in ("1", "s_theta", "s_eta"))
    return BoundaryRelation(
        constraint_rows=constraint,
        rhs_map=np.eye(len(labels), dtype=np.float64),
        labels=labels,
        functional=functional,
    )


def _svd_pinv_and_nullspace(
    matrix: np.ndarray, *, rcond: float
) -> tuple[np.ndarray, np.ndarray, int, float]:
    rows, columns = matrix.shape
    u, singular, vt = np.linalg.svd(matrix, full_matrices=True)
    threshold = rcond * singular[0] if len(singular) and singular[0] > 0.0 else 0.0
    rank = int(np.count_nonzero(singular > threshold))
    pinv = np.zeros((columns, rows), dtype=np.float64)
    if rank:
        pinv = (vt[:rank].T / singular[:rank]) @ u[:, :rank].T
        condition = float(singular[0] / singular[rank - 1])
    else:
        condition = float("inf")
    nullspace = vt[rank:].T
    return pinv, nullspace, rank, condition


def prepare_boundary_reconstruction(
    observation_matrix: np.ndarray,
    observation_weight: np.ndarray,
    relation: BoundaryRelation,
    *,
    rcond: float = 1.0e-12,
) -> PreparedBoundaryReconstruction:
    """Compile a weighted constrained least-squares fit into fixed maps.

    Redundant boundary moments are removed by a scaled rank-revealing SVD.
    The weighted fit is then solved in the constraint nullspace.  No boundary
    values are consumed at preparation time.
    """

    observation = _as_float64("observation_matrix", observation_matrix, 2)
    weight = _as_float64("observation_weight", observation_weight, 1)
    constraints = np.asarray(relation.constraint_rows, dtype=np.float64)
    rhs_map = np.asarray(relation.rhs_map, dtype=np.float64)
    if weight.shape != (observation.shape[0],) or np.any(weight <= 0.0):
        raise ValueError("observation_weight must be positive per observation")
    if constraints.shape[1] != observation.shape[1]:
        raise ValueError("constraint and observation coefficient counts differ")
    if not np.isfinite(rcond) or not 0.0 < rcond < 1.0:
        raise ValueError("rcond must lie strictly between zero and one")
    root_weight = np.sqrt(weight)
    weighted = observation * root_weight[:, None]
    column_scale = np.sqrt(
        np.sum(weighted * weighted, axis=0) + np.sum(constraints * constraints, axis=0)
    )
    column_scale = np.where(column_scale > 0.0, column_scale, 1.0)
    scaled_observation = weighted / column_scale[None, :]
    scaled_constraints = constraints / column_scale[None, :]
    constraint_pinv, nullspace, constraint_rank, constraint_condition = (
        _svd_pinv_and_nullspace(scaled_constraints, rcond=rcond)
    )
    constraint_projected_rhs = scaled_constraints @ constraint_pinv @ rhs_map
    if not np.allclose(
        constraint_projected_rhs, rhs_map, rtol=50.0 * rcond, atol=50.0 * rcond
    ):
        raise ValueError("boundary rhs map is inconsistent with retained constraints")
    reduced = scaled_observation @ nullspace
    reduced_pinv, _unused, reduced_rank, reduced_condition = _svd_pinv_and_nullspace(
        reduced, rcond=rcond
    )
    if reduced_rank != nullspace.shape[1]:
        raise ValueError("observations do not determine the constrained polynomial")
    observation_pinv, _unused, observation_rank, observation_condition = (
        _svd_pinv_and_nullspace(scaled_observation, rcond=rcond)
    )
    del observation_pinv, _unused
    particular = constraint_pinv @ rhs_map
    correction = nullspace @ reduced_pinv
    inverse_scale = 1.0 / column_scale
    owner_map = inverse_scale[:, None] * (correction * root_weight[None, :])
    boundary_map = inverse_scale[:, None] * (
        particular - correction @ scaled_observation @ particular
    )
    return PreparedBoundaryReconstruction(
        owner_map=owner_map,
        boundary_map=boundary_map,
        observation_rank=observation_rank,
        constraint_rank=constraint_rank,
        reduced_rank=reduced_rank,
        observation_condition=observation_condition,
        constraint_condition=constraint_condition,
        reduced_condition=reduced_condition,
    )


def apply_boundary_reconstruction(
    prepared: PreparedBoundaryReconstruction,
    owner_data: jnp.ndarray,
    boundary_data: jnp.ndarray,
) -> jnp.ndarray:
    """Apply fixed maps to scalar or batched dynamic data."""

    owner = jnp.asarray(owner_data)
    boundary = jnp.asarray(boundary_data)
    return (
        jnp.asarray(prepared.owner_map, dtype=owner.dtype) @ owner
        + jnp.asarray(prepared.boundary_map, dtype=boundary.dtype) @ boundary
    )


__all__ = [
    "BOUNDARY_FUNCTIONAL_RECONSTRUCTION_VERSION",
    "BoundaryFunctionalGeometry",
    "BoundaryRelation",
    "PreparedBoundaryReconstruction",
    "apply_boundary_reconstruction",
    "build_boundary_functional_geometry",
    "build_moment_boundary_relation",
    "prepare_boundary_reconstruction",
]
