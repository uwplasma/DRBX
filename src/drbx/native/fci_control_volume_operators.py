"""Canonical moment-reconstruction and direct face-functional primitives.

The module is intentionally narrow: it owns moment-fit metadata and its
runtime evaluation.  Geometry construction and sharding compilation live in
``drbx.geometry.fci_control_volumes``.  Legacy FCI modules can delegate here
while the experimental embedded-boundary path is migrated in stages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from .fci_boundaries import (
    CV_RECONSTRUCTION_EQUATION_CELL,
    CV_RECONSTRUCTION_EQUATION_DIRICHLET,
    CV_RECONSTRUCTION_EQUATION_REMOTE_CELL,
    LocalMomentReconstruction3D,
)


CUBIC_MONOMIAL_EXPONENTS: tuple[tuple[int, int, int], ...] = tuple(
    (px, py, degree - px - py)
    for degree in range(4)
    for px in range(degree, -1, -1)
    for py in range(degree - px, -1, -1)
)


def cubic_monomial_basis(points: np.ndarray) -> np.ndarray:
    """Evaluate the 20 monomials through total degree three at points."""

    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1:] != (3,):
        raise ValueError("points must have a trailing logical-coordinate axis")
    return np.stack(
        [
            points[..., 0] ** power[0]
            * points[..., 1] ** power[1]
            * points[..., 2] ** power[2]
            for power in CUBIC_MONOMIAL_EXPONENTS
        ],
        axis=-1,
    )


def cubic_control_volume_average_basis(
    centroid: np.ndarray,
    second_moment: np.ndarray,
    third_moment: np.ndarray,
    *,
    origin: np.ndarray | None = None,
    scale: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Return exact cubic basis averages from central control-volume moments.

    Coordinates are translated by ``origin`` and scaled componentwise before
    evaluating the basis.  This is the common moment row used by cell-average
    observations in both reconstruction and direct face fitting.
    """

    centroid = np.asarray(centroid, dtype=np.float64)
    second = np.asarray(second_moment, dtype=np.float64)
    third = np.asarray(third_moment, dtype=np.float64)
    if centroid.shape[-1:] != (3,) or second.shape[-2:] != (3, 3) or third.shape[-3:] != (3, 3, 3):
        raise ValueError("centroid, second_moment, and third_moment need 3D trailing shapes")
    if centroid.shape[:-1] != second.shape[:-2] or centroid.shape[:-1] != third.shape[:-3]:
        raise ValueError("control-volume moment batch shapes must match")
    origin_value = np.zeros((3,), dtype=np.float64) if origin is None else np.asarray(origin, dtype=np.float64)
    scale_value = np.asarray(scale, dtype=np.float64)
    if origin_value.shape != (3,):
        raise ValueError("origin must have shape (3,)")
    if scale_value.ndim == 0:
        scale_value = np.full((3,), float(scale_value), dtype=np.float64)
    if scale_value.shape != (3,) or np.any(~np.isfinite(scale_value)) or np.any(scale_value <= 0.0):
        raise ValueError("scale must be one positive scalar or three positive values")
    displacement = centroid - origin_value
    raw_second = second + displacement[..., :, None] * displacement[..., None, :]
    raw_third = (
        third
        + displacement[..., :, None, None] * second[..., None, :, :]
        + displacement[..., None, :, None] * second[..., :, None, :]
        + displacement[..., None, None, :] * second[..., :, :, None]
        + displacement[..., :, None, None]
        * displacement[..., None, :, None]
        * displacement[..., None, None, :]
    )
    result = np.empty(centroid.shape[:-1] + (20,), dtype=np.float64)
    for column, power in enumerate(CUBIC_MONOMIAL_EXPONENTS):
        degree = sum(power)
        if degree == 0:
            value = np.ones(centroid.shape[:-1], dtype=np.float64)
        elif degree == 1:
            axis = int(np.flatnonzero(power)[0])
            value = displacement[..., axis]
        elif degree == 2:
            axes = np.repeat(np.arange(3), np.asarray(power, dtype=np.int32))
            value = raw_second[..., axes[0], axes[1]]
        else:
            axes = np.repeat(np.arange(3), np.asarray(power, dtype=np.int32))
            value = raw_third[..., axes[0], axes[1], axes[2]]
        denominator = np.prod(scale_value ** np.asarray(power, dtype=np.float64))
        result[..., column] = value / denominator
    return result


def cubic_dense_face_targets(
    regular_sample_centroid: np.ndarray,
    regular_sample_second_moment: np.ndarray,
    regular_sample_third_moment: np.ndarray,
    *,
    scalar_coefficients: np.ndarray,
    gradient_coefficients: np.ndarray,
    origin: np.ndarray | None = None,
    scale: np.ndarray | float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build direct cubic targets equivalent to a dense face functional.

    The dense structured operator is a linear combination of logical regular
    cell averages.  Applying its stored coefficients to exact cubic average
    rows yields the target value and coordinate gradients for a direct compact
    face fit.  This is the required compatibility condition at a
    dense/compact interface.
    """

    basis = cubic_control_volume_average_basis(
        regular_sample_centroid,
        regular_sample_second_moment,
        regular_sample_third_moment,
        origin=origin,
        scale=scale,
    )
    if basis.ndim != 2:
        raise ValueError("regular sample moments must describe one sample axis")
    scalar = np.asarray(scalar_coefficients, dtype=np.float64).reshape((-1,))
    gradient = np.asarray(gradient_coefficients, dtype=np.float64)
    if scalar.shape != (basis.shape[0],) or gradient.shape != (3, basis.shape[0]):
        raise ValueError("dense coefficients must align with regular samples")
    return scalar @ basis, gradient @ basis


def cubic_projected_face_flux_target(
    points: np.ndarray,
    jacobian: np.ndarray,
    area_covector_weight: np.ndarray,
    projector: np.ndarray,
    active: np.ndarray,
    *,
    origin: np.ndarray,
    scale: np.ndarray | float,
) -> np.ndarray:
    """Return the direct cubic target for the projected face-flux oracle.

    This is the compact-face runtime expression:
    ``J * a_weight^T P grad_x(phi)`` summed over active quadrature points.
    ``area_covector_weight`` already contains the two-dimensional Gauss weight;
    ``J`` is intentionally multiplied separately here (never double counted).
    """
    points = np.asarray(points, dtype=np.float64)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    area = np.asarray(area_covector_weight, dtype=np.float64)
    projector = np.asarray(projector, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    origin = np.asarray(origin, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if scale.ndim == 0:
        scale = np.full((3,), float(scale))
    if (
        points.shape[-1:] != (3,)
        or area.shape != points.shape
        or projector.shape != points.shape[:-1] + (3, 3)
        or jacobian.shape != active.shape
        or jacobian.shape != points.shape[:-1]
    ):
        raise ValueError("quadrature points/J/area/projector/active shapes are incompatible")
    if (
        origin.shape != (3,)
        or scale.shape != (3,)
        or np.any(~np.isfinite(scale))
        or np.any(scale <= 0)
    ):
        raise ValueError("origin and positive componentwise scale are required")
    if any(np.any(~np.isfinite(x)) for x in (points, jacobian, area, projector, origin)):
        raise ValueError("projected face target inputs must be finite")
    xi = (points - origin) / scale
    target = np.zeros((20,), dtype=np.float64)
    for column, power in enumerate(CUBIC_MONOMIAL_EXPONENTS):
        grad = np.zeros_like(points)
        for axis in range(3):
            if power[axis]:
                reduced = list(power)
                reduced[axis] -= 1
                grad[..., axis] = (
                    power[axis]
                    * np.prod(
                        [xi[..., q] ** reduced[q] for q in range(3)], axis=0,
                    )
                    / scale[axis]
                )
        integrand = jacobian * np.einsum("...i,...ij,...j->...", area, projector, grad)
        target[column] = np.sum(np.where(active, integrand, 0.0))
    return target


def cubic_parallel_face_flux_target(
    points: np.ndarray,
    jacobian: np.ndarray,
    area_covector_weight: np.ndarray,
    B_contra: np.ndarray,
    Bmag: np.ndarray,
    active: np.ndarray,
    *,
    origin: np.ndarray,
    scale: np.ndarray | float,
    b_floor: float = 1.0e-30,
) -> np.ndarray:
    """Return the direct cubic target for the conservative parallel flux.

    The target follows the runtime face quadrature exactly: for each cubic
    monomial ``phi``, sum ``J * dot(a_weight, B_contra / max(Bmag, b_floor))
    * phi`` over active points.  ``area_covector_weight`` contains the
    two-dimensional Gauss weight, while ``jacobian`` is deliberately applied
    once here.
    """
    points = np.asarray(points, dtype=np.float64)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    area = np.asarray(area_covector_weight, dtype=np.float64)
    b_contra = np.asarray(B_contra, dtype=np.float64)
    bmag = np.asarray(Bmag, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    origin = np.asarray(origin, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    b_floor = float(b_floor)
    if scale.ndim == 0:
        scale = np.full((3,), float(scale), dtype=np.float64)
    if (
        points.shape[-1:] != (3,)
        or area.shape != points.shape
        or b_contra.shape != points.shape
        or jacobian.shape != active.shape
        or bmag.shape != active.shape
        or jacobian.shape != points.shape[:-1]
    ):
        raise ValueError("quadrature points/J/area/B_contra/Bmag/active shapes are incompatible")
    if (
        origin.shape != (3,)
        or scale.shape != (3,)
        or np.any(~np.isfinite(scale))
        or np.any(scale <= 0.0)
        or not np.isfinite(b_floor)
        or b_floor <= 0.0
    ):
        raise ValueError("origin, positive componentwise scale, and positive b_floor are required")
    if any(np.any(~np.isfinite(item)) for item in (points, jacobian, area, b_contra, bmag, origin)):
        raise ValueError("parallel face target inputs must be finite")
    xi = (points - origin) / scale
    flux_scale = jacobian * np.einsum(
        "...i,...i->...", area, b_contra / np.maximum(bmag, b_floor)[..., None],
    )
    basis = cubic_monomial_basis(xi)
    return np.sum(np.where(active[..., None], flux_scale[..., None] * basis, 0.0), axis=tuple(range(active.ndim)))


def cubic_parallel_gradient_face_flux_target(
    points: np.ndarray,
    jacobian: np.ndarray,
    area_covector_weight: np.ndarray,
    B_contra: np.ndarray,
    Bmag: np.ndarray,
    active: np.ndarray,
    *,
    origin: np.ndarray,
    scale: np.ndarray | float,
    b_floor: float = 1.0e-30,
) -> np.ndarray:
    """Return the cubic target for ``J a^T P_parallel grad(phi)``.

    This is the direct face functional required by the conservative parallel
    Laplacian.  It is intentionally distinct from ``cubic_parallel_face_flux_target``,
    whose integrand contains the scalar value ``phi`` rather than its gradient.
    """
    b_contra = np.asarray(B_contra, dtype=np.float64)
    bmag = np.asarray(Bmag, dtype=np.float64)
    b_floor = float(b_floor)
    if b_floor <= 0.0 or not np.isfinite(b_floor):
        raise ValueError("b_floor must be positive and finite")
    if b_contra.shape[-1:] != (3,) or bmag.shape != b_contra.shape[:-1]:
        raise ValueError("B_contra and Bmag shapes are incompatible")
    if np.any(~np.isfinite(b_contra)) or np.any(~np.isfinite(bmag)):
        raise ValueError("parallel-gradient face target magnetic inputs must be finite")
    b = b_contra / np.maximum(bmag, b_floor)[..., None]
    projector = np.einsum("...i,...j->...ij", b, b)
    return cubic_projected_face_flux_target(
        points,
        jacobian,
        area_covector_weight,
        projector,
        active,
        origin=origin,
        scale=scale,
    )


@dataclass(frozen=True)
class LocalMomentFittedFaceFunctional3D:
    """One direct compact-face functional with static observation weights."""

    equation_kind: np.ndarray
    sample_reference: np.ndarray
    active: np.ndarray
    value_weights: np.ndarray
    gradient_weights: np.ndarray
    polynomial_order: int
    rank: int
    condition_number: float
    reproduction_residual: float
    normalized_weight_norm: float
    face_id: int = -1
    face_sign: int = 1
    projected_flux_weights: np.ndarray | None = None
    parallel_flux_weights: np.ndarray | None = None
    parallel_gradient_flux_weights: np.ndarray | None = None
    normalized_projected_weight_norm: float | None = None
    normalized_parallel_weight_norm: float | None = None
    normalized_parallel_gradient_weight_norm: float | None = None

    def __post_init__(self) -> None:
        kind = np.asarray(self.equation_kind, dtype=np.int32).reshape((-1,))
        reference = np.asarray(self.sample_reference, dtype=np.int64).reshape((-1,))
        active = np.asarray(self.active, dtype=bool).reshape((-1,))
        value = np.asarray(self.value_weights, dtype=np.float64).reshape((-1,))
        gradient = np.asarray(self.gradient_weights, dtype=np.float64)
        count = kind.size
        projected = (
            np.zeros((count,), dtype=np.float64)
            if self.projected_flux_weights is None
            else np.asarray(self.projected_flux_weights, dtype=np.float64).reshape((-1,))
        )
        parallel = (
            np.zeros((count,), dtype=np.float64)
            if self.parallel_flux_weights is None
            else np.asarray(self.parallel_flux_weights, dtype=np.float64).reshape((-1,))
        )
        parallel_gradient = (
            np.zeros((count,), dtype=np.float64)
            if self.parallel_gradient_flux_weights is None
            else np.asarray(
                self.parallel_gradient_flux_weights, dtype=np.float64
            ).reshape((-1,))
        )
        if not (
            reference.size == active.size == value.size == count
            and gradient.shape == (3, count)
            and projected.shape == parallel.shape == parallel_gradient.shape == (count,)
        ):
            raise ValueError("face-functional observation arrays must align")
        valid_kind = {
            CV_RECONSTRUCTION_EQUATION_CELL,
            CV_RECONSTRUCTION_EQUATION_REMOTE_CELL,
            CV_RECONSTRUCTION_EQUATION_DIRICHLET,
        }
        if any(int(item) not in valid_kind for item in kind[active]):
            raise ValueError("face functional has an unsupported equation kind")
        if np.any(active & (reference < 0)):
            raise ValueError("active face-functional observations need nonnegative references")
        object.__setattr__(self, "equation_kind", kind)
        object.__setattr__(self, "sample_reference", reference)
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "value_weights", value)
        object.__setattr__(self, "gradient_weights", gradient)
        object.__setattr__(self, "projected_flux_weights", projected)
        object.__setattr__(self, "parallel_flux_weights", parallel)
        object.__setattr__(
            self, "parallel_gradient_flux_weights", parallel_gradient
        )
        object.__setattr__(
            self, "normalized_projected_weight_norm",
            float(np.linalg.norm(projected)) if self.normalized_projected_weight_norm is None
            else float(self.normalized_projected_weight_norm),
        )
        object.__setattr__(
            self, "normalized_parallel_weight_norm",
            float(np.linalg.norm(parallel)) if self.normalized_parallel_weight_norm is None
            else float(self.normalized_parallel_weight_norm),
        )
        object.__setattr__(
            self, "normalized_parallel_gradient_weight_norm",
            float(np.linalg.norm(parallel_gradient))
            if self.normalized_parallel_gradient_weight_norm is None
            else float(self.normalized_parallel_gradient_weight_norm),
        )
        diagnostics = (
            self.normalized_projected_weight_norm,
            self.normalized_parallel_weight_norm,
            self.normalized_parallel_gradient_weight_norm,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in diagnostics):
            raise ValueError("normalized face-functional weight norms must be finite and nonnegative")


@dataclass(frozen=True)
class LocalMomentFittedFaceFunctionals3D:
    """Packed direct functionals for a set of unique compact faces.

    This host-side representation is intentionally independent of the legacy
    transition-row layout.  A later JAX compiler lowers its observation
    references into owned/halo/BC gathers; keeping the rows packed here makes
    global face ordering and mirrored-shard validation testable now.
    """

    face_id: np.ndarray
    face_sign: np.ndarray
    equation_kind: np.ndarray
    sample_reference: np.ndarray
    observation_active: np.ndarray
    value_weights: np.ndarray
    gradient_weights: np.ndarray
    projected_flux_weights: np.ndarray
    parallel_flux_weights: np.ndarray
    parallel_gradient_flux_weights: np.ndarray
    rank: np.ndarray
    condition_number: np.ndarray
    reproduction_residual: np.ndarray
    normalized_weight_norm: np.ndarray
    normalized_projected_weight_norm: np.ndarray
    normalized_parallel_weight_norm: np.ndarray
    normalized_parallel_gradient_weight_norm: np.ndarray

    def __post_init__(self) -> None:
        face_id = np.asarray(self.face_id, dtype=np.int64).reshape((-1,))
        count = face_id.size
        face_sign = np.asarray(self.face_sign, dtype=np.int8).reshape((-1,))
        kind = np.asarray(self.equation_kind, dtype=np.int32)
        reference = np.asarray(self.sample_reference, dtype=np.int64)
        active = np.asarray(self.observation_active, dtype=bool)
        value = np.asarray(self.value_weights, dtype=np.float64)
        gradient = np.asarray(self.gradient_weights, dtype=np.float64)
        projected = np.asarray(self.projected_flux_weights, dtype=np.float64)
        parallel = np.asarray(self.parallel_flux_weights, dtype=np.float64)
        parallel_gradient = np.asarray(
            self.parallel_gradient_flux_weights, dtype=np.float64
        )
        rank = np.asarray(self.rank, dtype=np.int32).reshape((-1,))
        condition = np.asarray(self.condition_number, dtype=np.float64).reshape((-1,))
        residual = np.asarray(self.reproduction_residual, dtype=np.float64).reshape((-1,))
        norm = np.asarray(self.normalized_weight_norm, dtype=np.float64).reshape((-1,))
        projected_norm = np.asarray(self.normalized_projected_weight_norm, dtype=np.float64).reshape((-1,))
        parallel_norm = np.asarray(self.normalized_parallel_weight_norm, dtype=np.float64).reshape((-1,))
        parallel_gradient_norm = np.asarray(
            self.normalized_parallel_gradient_weight_norm, dtype=np.float64
        ).reshape((-1,))
        if not (
            kind.ndim == reference.ndim == active.ndim == value.ndim == 2
            and kind.shape == reference.shape == active.shape == value.shape
            and gradient.shape == (count, 3, kind.shape[1])
            and projected.shape == parallel.shape == parallel_gradient.shape == kind.shape
            and face_sign.shape == rank.shape == condition.shape == residual.shape == norm.shape == projected_norm.shape == parallel_norm.shape == parallel_gradient_norm.shape == (count,)
        ):
            raise ValueError("packed face-functional arrays must have compatible shapes")
        if np.unique(face_id).size != count:
            raise ValueError("packed face functional IDs must be unique per shard")
        if np.any((face_sign != -1) & (face_sign != 1)):
            raise ValueError("packed face signs must be either -1 or +1")
        if np.any(active & (reference < 0)):
            raise ValueError("active functional observations need nonnegative references")
        object.__setattr__(self, "face_id", face_id)
        object.__setattr__(self, "face_sign", face_sign)
        object.__setattr__(self, "equation_kind", kind)
        object.__setattr__(self, "sample_reference", reference)
        object.__setattr__(self, "observation_active", active)
        object.__setattr__(self, "value_weights", value)
        object.__setattr__(self, "gradient_weights", gradient)
        object.__setattr__(self, "projected_flux_weights", projected)
        object.__setattr__(self, "parallel_flux_weights", parallel)
        object.__setattr__(self, "parallel_gradient_flux_weights", parallel_gradient)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "condition_number", condition)
        object.__setattr__(self, "reproduction_residual", residual)
        object.__setattr__(self, "normalized_weight_norm", norm)
        object.__setattr__(self, "normalized_projected_weight_norm", projected_norm)
        object.__setattr__(self, "normalized_parallel_weight_norm", parallel_norm)
        object.__setattr__(
            self,
            "normalized_parallel_gradient_weight_norm",
            parallel_gradient_norm,
        )


def pack_local_face_functionals(
    functionals: list[LocalMomentFittedFaceFunctional3D],
) -> LocalMomentFittedFaceFunctionals3D:
    """Pack equal-length direct functionals for deterministic inspection.

    Production face builders select a fixed observation capacity, so rejecting
    variable lengths here catches an accidental geometry-dependent runtime
    layout before JAX sees it.
    """

    if not functionals:
        return LocalMomentFittedFaceFunctionals3D(
            face_id=np.zeros((0,), dtype=np.int64),
            face_sign=np.zeros((0,), dtype=np.int8),
            equation_kind=np.zeros((0, 0), dtype=np.int32),
            sample_reference=np.zeros((0, 0), dtype=np.int64),
            observation_active=np.zeros((0, 0), dtype=bool),
            value_weights=np.zeros((0, 0), dtype=np.float64),
            gradient_weights=np.zeros((0, 3, 0), dtype=np.float64),
            projected_flux_weights=np.zeros((0, 0), dtype=np.float64),
            parallel_flux_weights=np.zeros((0, 0), dtype=np.float64),
            parallel_gradient_flux_weights=np.zeros((0, 0), dtype=np.float64),
            rank=np.zeros((0,), dtype=np.int32),
            condition_number=np.zeros((0,), dtype=np.float64),
            reproduction_residual=np.zeros((0,), dtype=np.float64),
            normalized_weight_norm=np.zeros((0,), dtype=np.float64),
            normalized_projected_weight_norm=np.zeros((0,), dtype=np.float64),
            normalized_parallel_weight_norm=np.zeros((0,), dtype=np.float64),
            normalized_parallel_gradient_weight_norm=np.zeros((0,), dtype=np.float64),
        )
    count = functionals[0].equation_kind.size
    if any(item.equation_kind.size != count for item in functionals):
        raise ValueError("packed face functionals require one observation capacity")
    return LocalMomentFittedFaceFunctionals3D(
        face_id=np.asarray([item.face_id for item in functionals]),
        face_sign=np.asarray([item.face_sign for item in functionals]),
        equation_kind=np.stack([item.equation_kind for item in functionals]),
        sample_reference=np.stack([item.sample_reference for item in functionals]),
        observation_active=np.stack([item.active for item in functionals]),
        value_weights=np.stack([item.value_weights for item in functionals]),
        gradient_weights=np.stack([item.gradient_weights for item in functionals]),
        projected_flux_weights=np.stack([item.projected_flux_weights for item in functionals]),
        parallel_flux_weights=np.stack([item.parallel_flux_weights for item in functionals]),
        parallel_gradient_flux_weights=np.stack(
            [item.parallel_gradient_flux_weights for item in functionals]
        ),
        rank=np.asarray([item.rank for item in functionals]),
        condition_number=np.asarray([item.condition_number for item in functionals]),
        reproduction_residual=np.asarray([item.reproduction_residual for item in functionals]),
        normalized_weight_norm=np.asarray([item.normalized_weight_norm for item in functionals]),
        normalized_projected_weight_norm=np.asarray([item.normalized_projected_weight_norm for item in functionals]),
        normalized_parallel_weight_norm=np.asarray([item.normalized_parallel_weight_norm for item in functionals]),
        normalized_parallel_gradient_weight_norm=np.asarray(
            [item.normalized_parallel_gradient_weight_norm for item in functionals]
        ),
    )


def precompute_local_moment_reconstruction(
    cells,
    irregular_faces,
    *,
    spacing_owned,
    requested_order: int = 3,
    max_radius: int = 2,
    **kwargs,
) -> LocalMomentReconstruction3D:
    """Build canonical local moment reconstruction metadata.

    The temporary delegate preserves tested numerical behavior while callers
    migrate.  Radius three is intentionally rejected: it exceeds the standard
    halo contract and introduces decomposition-dependent support.
    """

    requested_order = int(requested_order)
    if requested_order not in (1, 2, 3):
        raise ValueError("requested_order must be one, two, or three")
    if int(max_radius) != 2:
        raise ValueError("max_radius must match the two-cell halo contract")
    from .fci_operators import (
        _precompute_local_cubic_reconstruction,
        _precompute_local_degree_two_reconstruction,
    )

    if requested_order < 3:
        return _precompute_local_degree_two_reconstruction(
            cells,
            irregular_faces,
            spacing_owned=spacing_owned,
            **kwargs,
        )

    return _precompute_local_cubic_reconstruction(
        cells,
        irregular_faces,
        spacing_owned=spacing_owned,
        **kwargs,
    )


def precompute_local_face_functional(
    observation_matrix: np.ndarray,
    *,
    equation_kind: np.ndarray,
    sample_reference: np.ndarray,
    value_target: np.ndarray,
    gradient_target: np.ndarray,
    projected_flux_target: np.ndarray | None = None,
    parallel_flux_target: np.ndarray | None = None,
    parallel_gradient_flux_target: np.ndarray | None = None,
    observation_weight: np.ndarray | None = None,
    requested_order: int = 3,
    svd_cutoff: float = 1.0e-12,
    condition_limit: float = 1.0e6,
    max_derivative_l1: float = 100.0,
    max_projected_flux_l1: float = 100.0,
    max_parallel_flux_l1: float = 100.0,
    max_parallel_gradient_flux_l1: float = 100.0,
    max_normalized_projected_weight_norm: float = np.inf,
    max_normalized_parallel_weight_norm: float = np.inf,
    max_normalized_parallel_gradient_weight_norm: float = np.inf,
    face_id: int = -1,
    face_sign: int = 1,
) -> LocalMomentFittedFaceFunctional3D:
    """Fit one direct value/gradient functional from moment observations.

    ``observation_matrix`` has one row per control-volume average or
    independent boundary trace moment and one column per polynomial basis
    term.  The returned weights make runtime flux evaluation a pair of dot
    products; no owner-centered virtual average is materialized.
    """

    if int(requested_order) != 3:
        raise ValueError("only the 20-term cubic compact functional is supported")
    matrix = np.asarray(observation_matrix, dtype=np.float64)
    kind = np.asarray(equation_kind, dtype=np.int32).reshape((-1,))
    reference = np.asarray(sample_reference, dtype=np.int64).reshape((-1,))
    value_target = np.asarray(value_target, dtype=np.float64).reshape((-1,))
    gradient_target = np.asarray(gradient_target, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (kind.size, 20):
        raise ValueError("cubic observation_matrix must have shape (observations, 20)")
    if np.any(~np.isfinite(matrix)) or np.any(~np.isfinite(value_target)) or np.any(~np.isfinite(gradient_target)):
        raise ValueError("face-functional matrix and targets must be finite")
    if reference.size != kind.size or value_target.shape != (20,):
        raise ValueError("face-functional targets must align with the cubic basis")
    if gradient_target.shape != (3, 20):
        raise ValueError("gradient_target must have shape (3, 20)")
    projected_target = np.zeros((20,), dtype=np.float64) if projected_flux_target is None else np.asarray(projected_flux_target, dtype=np.float64).reshape((-1,))
    if projected_target.shape != (20,) or np.any(~np.isfinite(projected_target)):
        raise ValueError("projected_flux_target must have shape (20,) and be finite")
    parallel_target = np.zeros((20,), dtype=np.float64) if parallel_flux_target is None else np.asarray(parallel_flux_target, dtype=np.float64).reshape((-1,))
    if parallel_target.shape != (20,) or np.any(~np.isfinite(parallel_target)):
        raise ValueError("parallel_flux_target must have shape (20,) and be finite")
    parallel_gradient_target = (
        np.zeros((20,), dtype=np.float64)
        if parallel_gradient_flux_target is None
        else np.asarray(parallel_gradient_flux_target, dtype=np.float64).reshape((-1,))
    )
    if parallel_gradient_target.shape != (20,) or np.any(
        ~np.isfinite(parallel_gradient_target)
    ):
        raise ValueError(
            "parallel_gradient_flux_target must have shape (20,) and be finite"
        )
    if observation_weight is None:
        weight = np.ones((kind.size,), dtype=np.float64)
    else:
        weight = np.asarray(observation_weight, dtype=np.float64).reshape((-1,))
        if weight.shape != (kind.size,) or np.any(~np.isfinite(weight)) or np.any(weight <= 0.0):
            raise ValueError("observation_weight must be positive and align with observations")
    weighted_matrix = np.sqrt(weight)[:, None] * matrix
    try:
        u, singular, vh = np.linalg.svd(weighted_matrix, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("face-functional SVD failed") from exc
    tolerance = float(svd_cutoff) * singular[0] if singular.size else np.inf
    rank = int(np.sum(singular > tolerance))
    condition = (
        float(singular[0] / singular[19]) if rank >= 20 else np.inf
    )
    if rank < 20 or condition > float(condition_limit):
        raise ValueError(
            f"cubic face functional is rank deficient/ill conditioned: "
            f"rank={rank}, condition={condition:.3e}"
        )
    inverse = (vh[:20].T / singular[:20]) @ u[:, :20].T
    weighted_value_weights = value_target @ inverse
    weighted_gradient_weights = gradient_target @ inverse
    weighted_projected_weights = projected_target @ inverse
    weighted_parallel_weights = parallel_target @ inverse
    weighted_parallel_gradient_weights = parallel_gradient_target @ inverse
    value_weights = weighted_value_weights * np.sqrt(weight)
    gradient_weights = weighted_gradient_weights * np.sqrt(weight)[None, :]
    projected_weights = weighted_projected_weights * np.sqrt(weight)
    parallel_weights = weighted_parallel_weights * np.sqrt(weight)
    parallel_gradient_weights = weighted_parallel_gradient_weights * np.sqrt(weight)
    reproduction = max(
        float(np.max(np.abs(value_weights @ matrix - value_target))),
        float(np.max(np.abs(gradient_weights @ matrix - gradient_target))),
        float(np.max(np.abs(projected_weights @ matrix - projected_target))),
        float(np.max(np.abs(parallel_weights @ matrix - parallel_target))),
        float(
            np.max(
                np.abs(
                    parallel_gradient_weights @ matrix
                    - parallel_gradient_target
                )
            )
        ),
    )
    derivative_l1 = float(np.max(np.sum(np.abs(gradient_weights), axis=1)))
    projected_l1 = float(np.sum(np.abs(projected_weights)))
    parallel_l1 = float(np.sum(np.abs(parallel_weights)))
    parallel_gradient_l1 = float(np.sum(np.abs(parallel_gradient_weights)))
    # Observation rows use the nondimensional normalized cubic basis.  Divide
    # by the corresponding target coefficient norm so these diagnostics are
    # dimensionless amplification factors rather than mesh-scaled flux norms.
    projected_target_norm = float(np.linalg.norm(projected_target))
    parallel_target_norm = float(np.linalg.norm(parallel_target))
    parallel_gradient_target_norm = float(np.linalg.norm(parallel_gradient_target))
    projected_norm = (
        0.0
        if projected_target_norm == 0.0
        else float(np.linalg.norm(projected_weights)) / projected_target_norm
    )
    parallel_norm = (
        0.0
        if parallel_target_norm == 0.0
        else float(np.linalg.norm(parallel_weights)) / parallel_target_norm
    )
    parallel_gradient_norm = (
        0.0
        if parallel_gradient_target_norm == 0.0
        else float(np.linalg.norm(parallel_gradient_weights))
        / parallel_gradient_target_norm
    )
    if not np.isfinite(reproduction) or reproduction > 1.0e-10:
        raise ValueError(f"cubic face functional reproduction failed: {reproduction:.3e}")
    if derivative_l1 > float(max_derivative_l1):
        raise ValueError(
            f"cubic face functional derivative norm {derivative_l1:.3e} exceeds limit"
        )
    if projected_l1 > float(max_projected_flux_l1):
        raise ValueError(
            "cubic face functional projected-flux norm "
            f"{projected_l1:.3e} exceeds limit"
        )
    if parallel_l1 > float(max_parallel_flux_l1):
        raise ValueError(
            "cubic face functional parallel-flux norm "
            f"{parallel_l1:.3e} exceeds limit"
        )
    if parallel_gradient_l1 > float(max_parallel_gradient_flux_l1):
        raise ValueError(
            "cubic face functional parallel-gradient-flux norm "
            f"{parallel_gradient_l1:.3e} exceeds limit"
        )
    if projected_norm > float(max_normalized_projected_weight_norm):
        raise ValueError(
            "cubic face functional normalized projected weight norm "
            f"{projected_norm:.3e} exceeds limit"
        )
    if parallel_norm > float(max_normalized_parallel_weight_norm):
        raise ValueError(
            "cubic face functional normalized parallel weight norm "
            f"{parallel_norm:.3e} exceeds limit"
        )
    if parallel_gradient_norm > float(
        max_normalized_parallel_gradient_weight_norm
    ):
        raise ValueError(
            "cubic face functional normalized parallel-gradient weight norm "
            f"{parallel_gradient_norm:.3e} exceeds limit"
        )
    return LocalMomentFittedFaceFunctional3D(
        equation_kind=kind,
        sample_reference=reference,
        active=np.ones((kind.size,), dtype=bool),
        value_weights=value_weights,
        gradient_weights=gradient_weights,
        polynomial_order=3,
        rank=rank,
        condition_number=condition,
        reproduction_residual=reproduction,
        normalized_weight_norm=max(
            float(np.linalg.norm(value_weights)),
            float(np.max(np.linalg.norm(gradient_weights, axis=1))),
        ),
        face_id=int(face_id),
        face_sign=int(face_sign),
        projected_flux_weights=projected_weights,
        parallel_flux_weights=parallel_weights,
        parallel_gradient_flux_weights=parallel_gradient_weights,
        normalized_projected_weight_norm=projected_norm,
        normalized_parallel_weight_norm=parallel_norm,
        normalized_parallel_gradient_weight_norm=parallel_gradient_norm,
    )


def evaluate_local_face_functional(
    functional: LocalMomentFittedFaceFunctional3D,
    *,
    local_values: np.ndarray,
    remote_values: np.ndarray | None = None,
    boundary_values: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Evaluate a direct compact face functional from gathered observations."""

    local_values = np.asarray(local_values, dtype=np.float64).reshape((-1,))
    remote_values = (
        np.asarray(remote_values, dtype=np.float64).reshape((-1,))
        if remote_values is not None
        else np.zeros((0,), dtype=np.float64)
    )
    boundary_values = (
        np.asarray(boundary_values, dtype=np.float64).reshape((-1,))
        if boundary_values is not None
        else np.zeros((0,), dtype=np.float64)
    )
    observation = np.zeros_like(functional.value_weights)
    for row, (kind, reference, active) in enumerate(
        zip(
            functional.equation_kind,
            functional.sample_reference,
            functional.active,
        )
    ):
        if not active:
            continue
        values = (
            local_values
            if kind == CV_RECONSTRUCTION_EQUATION_CELL
            else (
                remote_values
                if kind == CV_RECONSTRUCTION_EQUATION_REMOTE_CELL
                else boundary_values
            )
        )
        if not 0 <= int(reference) < values.size:
            raise ValueError("face-functional observation reference is unavailable")
        observation[row] = values[int(reference)]
    return (
        float(functional.value_weights @ observation),
        np.asarray(functional.gradient_weights @ observation),
    )


def evaluate_local_projected_face_flux(
    functional: LocalMomentFittedFaceFunctional3D, *, local_values: np.ndarray,
    remote_values: np.ndarray | None = None, boundary_values: np.ndarray | None = None,
) -> float:
    """Evaluate only the precompiled scalar projected flux observation row."""
    local_values = np.asarray(local_values, dtype=np.float64).reshape((-1,))
    remote_values = np.zeros((0,), dtype=np.float64) if remote_values is None else np.asarray(remote_values, dtype=np.float64).reshape((-1,))
    boundary_values = np.zeros((0,), dtype=np.float64) if boundary_values is None else np.asarray(boundary_values, dtype=np.float64).reshape((-1,))
    observation = np.zeros_like(functional.projected_flux_weights)
    for row, (kind, ref, active) in enumerate(zip(functional.equation_kind, functional.sample_reference, functional.active)):
        if active:
            values = local_values if kind == CV_RECONSTRUCTION_EQUATION_CELL else remote_values if kind == CV_RECONSTRUCTION_EQUATION_REMOTE_CELL else boundary_values
            if not 0 <= int(ref) < values.size: raise ValueError("face-functional observation reference is unavailable")
            observation[row] = values[int(ref)]
    return float(functional.projected_flux_weights @ observation)


def evaluate_local_parallel_face_flux(
    functional: LocalMomentFittedFaceFunctional3D, *, local_values: np.ndarray,
    remote_values: np.ndarray | None = None, boundary_values: np.ndarray | None = None,
) -> float:
    """Evaluate only the precompiled scalar parallel flux observation row."""
    local_values = np.asarray(local_values, dtype=np.float64).reshape((-1,))
    remote_values = np.zeros((0,), dtype=np.float64) if remote_values is None else np.asarray(remote_values, dtype=np.float64).reshape((-1,))
    boundary_values = np.zeros((0,), dtype=np.float64) if boundary_values is None else np.asarray(boundary_values, dtype=np.float64).reshape((-1,))
    observation = np.zeros_like(functional.parallel_flux_weights)
    for row, (kind, ref, active) in enumerate(zip(functional.equation_kind, functional.sample_reference, functional.active)):
        if active:
            values = local_values if kind == CV_RECONSTRUCTION_EQUATION_CELL else remote_values if kind == CV_RECONSTRUCTION_EQUATION_REMOTE_CELL else boundary_values
            if not 0 <= int(ref) < values.size: raise ValueError("face-functional observation reference is unavailable")
            observation[row] = values[int(ref)]
    return float(functional.parallel_flux_weights @ observation)


def evaluate_local_parallel_gradient_face_flux(
    functional: LocalMomentFittedFaceFunctional3D, *, local_values: np.ndarray,
    remote_values: np.ndarray | None = None,
    boundary_values: np.ndarray | None = None,
) -> float:
    """Evaluate the precompiled ``P_parallel grad(field)`` face flux."""
    local_values = np.asarray(local_values, dtype=np.float64).reshape((-1,))
    remote_values = (
        np.zeros((0,), dtype=np.float64)
        if remote_values is None
        else np.asarray(remote_values, dtype=np.float64).reshape((-1,))
    )
    boundary_values = (
        np.zeros((0,), dtype=np.float64)
        if boundary_values is None
        else np.asarray(boundary_values, dtype=np.float64).reshape((-1,))
    )
    observation = np.zeros_like(functional.parallel_gradient_flux_weights)
    for row, (kind, ref, active) in enumerate(
        zip(
            functional.equation_kind,
            functional.sample_reference,
            functional.active,
        )
    ):
        if active:
            values = (
                local_values
                if kind == CV_RECONSTRUCTION_EQUATION_CELL
                else remote_values
                if kind == CV_RECONSTRUCTION_EQUATION_REMOTE_CELL
                else boundary_values
            )
            if not 0 <= int(ref) < values.size:
                raise ValueError(
                    "face-functional observation reference is unavailable"
                )
            observation[row] = values[int(ref)]
    return float(functional.parallel_gradient_flux_weights @ observation)


__all__ = [
    "CUBIC_MONOMIAL_EXPONENTS",
    "cubic_control_volume_average_basis",
    "cubic_dense_face_targets",
    "cubic_projected_face_flux_target",
    "cubic_parallel_face_flux_target",
    "cubic_parallel_gradient_face_flux_target",
    "cubic_monomial_basis",
    "LocalMomentFittedFaceFunctional3D",
    "LocalMomentFittedFaceFunctionals3D",
    "LocalMomentReconstruction3D",
    "evaluate_local_face_functional",
    "evaluate_local_projected_face_flux",
    "evaluate_local_parallel_face_flux",
    "evaluate_local_parallel_gradient_face_flux",
    "pack_local_face_functionals",
    "precompute_local_face_functional",
    "precompute_local_moment_reconstruction",
]
