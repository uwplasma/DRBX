"""Canonical moment-reconstruction and direct face-functional primitives.

The module is intentionally narrow: it owns moment-fit metadata and its
runtime evaluation.  Geometry construction and sharding compilation live in
``drbx.geometry.fci_control_volumes``.  Legacy FCI modules can delegate here
while the experimental embedded-boundary path is migrated in stages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


def monomial_exponents(
    total_degree: int | None = None,
    *,
    exponents: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]] | None = None,
) -> tuple[tuple[int, int, int], ...]:
    """Return a validated, ordered set of three-coordinate exponents.

    ``total_degree`` selects all monomials through that degree using the
    historical ordering.  ``exponents`` permits a caller to select an
    arbitrary subset, which is useful for regular charts whose reconstruction
    basis is deliberately smaller than the full cubic basis.
    """
    if exponents is not None:
        if total_degree is not None:
            raise ValueError("provide either total_degree or exponents, not both")
        selected = tuple(tuple(int(power) for power in item) for item in exponents)
    else:
        if total_degree is None:
            total_degree = 3
        total_degree = int(total_degree)
        if total_degree not in (0, 1, 2, 3):
            raise ValueError("total_degree must be between zero and three")
        selected = tuple(
            power for power in CUBIC_MONOMIAL_EXPONENTS
            if sum(power) <= total_degree
        )
    if not selected:
        raise ValueError("at least one monomial exponent is required")
    if len(set(selected)) != len(selected):
        raise ValueError("monomial exponents must be unique")
    for power in selected:
        if len(power) != 3 or any(item < 0 for item in power):
            raise ValueError("monomial exponents must be three nonnegative integers")
        if sum(power) > 3:
            raise ValueError("moments through third order support degree at most three")
    return selected


def monomial_basis(
    points: np.ndarray,
    *,
    total_degree: int | None = None,
    exponents: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Evaluate selected monomials at three-coordinate points."""

    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1:] != (3,):
        raise ValueError("points must have a trailing logical-coordinate axis")
    selected = monomial_exponents(total_degree, exponents=exponents)
    return np.stack(
        [
            np.prod(
                [points[..., axis] ** power[axis] for axis in range(3)], axis=0
            )
            for power in selected
        ],
        axis=-1,
    )


def control_volume_average_basis(
    centroid: np.ndarray,
    second_moment: np.ndarray,
    third_moment: np.ndarray,
    *,
    origin: np.ndarray | None = None,
    scale: np.ndarray | float = 1.0,
    total_degree: int | None = None,
    exponents: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Return exact selected monomial averages from central moments.

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
    selected = monomial_exponents(total_degree, exponents=exponents)
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
    result = np.empty(centroid.shape[:-1] + (len(selected),), dtype=np.float64)
    for column, power in enumerate(selected):
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
    polynomial_exponents: tuple[tuple[int, int, int], ...] | None = None

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
        exponents = (
            tuple(tuple(int(value) for value in power) for power in self.polynomial_exponents)
            if self.polynomial_exponents is not None
            else None
        )
        if exponents is not None:
            monomial_exponents(exponents=exponents)
        object.__setattr__(self, "polynomial_exponents", exponents)
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
    polynomial_exponents: tuple[tuple[int, int, int], ...] | None = None

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
        exponents = (
            tuple(tuple(int(value) for value in power) for power in self.polynomial_exponents)
            if self.polynomial_exponents is not None
            else None
        )
        if exponents is not None:
            monomial_exponents(exponents=exponents)
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
        object.__setattr__(self, "polynomial_exponents", exponents)
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


__all__ = [
    "CUBIC_MONOMIAL_EXPONENTS",
    "monomial_exponents",
    "monomial_basis",
    "control_volume_average_basis",
    "LocalMomentFittedFaceFunctional3D",
    "LocalMomentFittedFaceFunctionals3D",
    "LocalMomentReconstruction3D",
]
