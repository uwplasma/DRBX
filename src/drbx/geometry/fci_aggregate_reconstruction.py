"""Experimental polynomial reconstruction from aggregate cell averages.

This module is deliberately a small, host-side experiment.  It turns compact
volume and moment data for distinct active owners into a sparse matrix that
recovers a value at a point in the regular planar chart
``X = u*cos(theta)``, ``Y = u*sin(theta)``.  Donors are selected independently
inside the same eta plane.  Aggregate moments are volume averages, and are
used as such; no wall closure, sharding, or deconvolution of an eta-averaged
quantity is implied by this API.

The matrix is built once with NumPy/SciPy and can then be applied to one or
more compact owner fields with :meth:`AggregateEvaluation.evaluate`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree


_SVD_RELATIVE_CUTOFF = 1.0e-12
_REPRODUCTION_TOLERANCE = 1.0e-9


def _as_float_array(value: Any, name: str) -> np.ndarray:
    """Convert an input to float64 and reject nonfinite data."""

    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class AggregateGeometry:
    """Compact geometry for one independent aggregate owner per row.

    ``owner_flat_ids`` are sorted indices in the original cell-shaped host
    arrays.  ``owner_plane`` is the eta-plane index for each compact owner.
    ``central_second_xy`` and ``central_third_xy`` contain volume-averaged
    central moments, with the latter optional because degree two needs no
    third-order information.
    """

    owner_flat_ids: np.ndarray
    owner_plane: np.ndarray
    volumes: np.ndarray
    centroid_xy: np.ndarray
    central_second_xy: np.ndarray
    central_third_xy: np.ndarray | None = None

    def __post_init__(self) -> None:
        owner_flat_ids = np.asarray(self.owner_flat_ids, dtype=np.int64).reshape(-1)
        owner_plane = np.asarray(self.owner_plane, dtype=np.int64).reshape(-1)
        volumes = _as_float_array(self.volumes, "volumes").reshape(-1)
        centroid_xy = _as_float_array(self.centroid_xy, "centroid_xy")
        central_second_xy = _as_float_array(
            self.central_second_xy, "central_second_xy"
        )
        count = owner_flat_ids.size
        if owner_plane.shape != (count,):
            raise ValueError("owner_plane must have one entry per compact owner")
        if volumes.shape != (count,):
            raise ValueError("volumes must have one entry per compact owner")
        if centroid_xy.shape != (count, 2):
            raise ValueError("centroid_xy must have shape (n_owner, 2)")
        if central_second_xy.shape != (count, 2, 2):
            raise ValueError("central_second_xy must have shape (n_owner, 2, 2)")
        if count == 0:
            raise ValueError("at least one active aggregate owner is required")
        if np.any(owner_flat_ids < 0) or np.any(np.diff(owner_flat_ids) <= 0):
            raise ValueError("owner_flat_ids must be nonnegative and strictly sorted")
        if np.any(owner_plane < 0):
            raise ValueError("owner_plane must be nonnegative")
        if np.any(volumes <= 0.0):
            raise ValueError("aggregate volumes must be strictly positive")

        third = self.central_third_xy
        if third is not None:
            third = _as_float_array(third, "central_third_xy")
            if third.shape != (count, 2, 2, 2):
                raise ValueError(
                    "central_third_xy must have shape (n_owner, 2, 2, 2)"
                )

        # Frozen dataclasses do not make mutable arrays immutable.  Defensive
        # copies prevent a caller from invalidating a built evaluation later.
        for name, value in (
            ("owner_flat_ids", owner_flat_ids),
            ("owner_plane", owner_plane),
            ("volumes", volumes),
            ("centroid_xy", centroid_xy),
            ("central_second_xy", central_second_xy),
            ("central_third_xy", third),
        ):
            if value is not None:
                value = np.array(value, copy=True)
                value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def n_owner(self) -> int:
        """Number of compact independent owners."""

        return int(self.owner_flat_ids.size)

    @property
    def owner_count(self) -> int:
        """Alias for :attr:`n_owner`."""

        return self.n_owner

    @classmethod
    def from_host_payload(cls, payload: Mapping[str, Any]) -> "AggregateGeometry":
        """Compact aggregate moments from a host/NPZ cache payload.

        Required keys are ``topology_is_active_owner``,
        ``host_aggregate_chart_volume``, ``host_aggregate_chart_centroid``, and
        ``host_aggregate_chart_second_moment``.  A third moment is consumed if
        ``host_aggregate_chart_third_moment`` is present.  Host chart arrays may
        have trailing size two or three; for size three only X/Y are retained.
        The final axis of the active-owner shape is interpreted as eta.
        """

        required = (
            "topology_is_active_owner",
            "host_aggregate_chart_volume",
            "host_aggregate_chart_centroid",
            "host_aggregate_chart_second_moment",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError("missing aggregate payload key(s): " + ", ".join(missing))

        active = np.asarray(payload[required[0]])
        if active.ndim == 0:
            raise ValueError("topology_is_active_owner must be an array")
        if active.dtype.kind == "f" and not np.all(np.isfinite(active)):
            raise ValueError("topology_is_active_owner must be finite")
        active = np.asarray(active, dtype=bool)
        shape = active.shape
        owner_flat_ids = np.flatnonzero(active.reshape(-1))
        if owner_flat_ids.size == 0:
            raise ValueError("payload contains no active aggregate owners")
        if active.ndim == 1:
            owner_plane = np.zeros(owner_flat_ids.size, dtype=np.int64)
        else:
            owner_plane = np.unravel_index(owner_flat_ids, shape)[-1].astype(np.int64)

        volume = _as_float_array(
            payload["host_aggregate_chart_volume"],
            "host_aggregate_chart_volume",
        )
        if volume.shape != shape:
            raise ValueError(
                "host_aggregate_chart_volume must match active-owner shape "
                f"{shape}, got {volume.shape}"
            )

        # The production host uses 3D chart moments.  Accepting 2D here makes
        # small planar cache fixtures convenient without changing semantics.
        centroid_name = "host_aggregate_chart_centroid"
        centroid_raw = _as_float_array(payload[centroid_name], centroid_name)
        centroid_tail = centroid_raw.shape[len(shape) :]
        if centroid_tail not in ((2,), (3,)):
            raise ValueError(
                f"{centroid_name} must end in (2,) or (3,), got {centroid_raw.shape}"
            )
        centroid = centroid_raw.reshape(shape + centroid_tail).reshape(
            (-1,) + centroid_tail
        )[owner_flat_ids][..., :2]

        second_name = "host_aggregate_chart_second_moment"
        second_raw = _as_float_array(payload[second_name], second_name)
        second_tail = second_raw.shape[len(shape) :]
        if second_tail not in ((2, 2), (3, 3)):
            raise ValueError(
                f"{second_name} must end in (2,2) or (3,3), got {second_raw.shape}"
            )
        second_compact = second_raw.reshape(shape + second_tail).reshape(
            (-1,) + second_tail
        )[owner_flat_ids]
        second = second_compact[..., :2, :2]

        third = None
        third_key = "host_aggregate_chart_third_moment"
        if third_key in payload and payload[third_key] is not None:
            third_name = third_key
            third_raw = _as_float_array(payload[third_name], third_name)
            third_tail = third_raw.shape[len(shape) :]
            if third_tail not in ((2, 2, 2), (3, 3, 3)):
                raise ValueError(
                    f"{third_name} must end in (2,2,2) or (3,3,3), got {third_raw.shape}"
                )
            third = third_raw.reshape(shape + third_tail).reshape(
                (-1,) + third_tail
            )[owner_flat_ids][..., :2, :2, :2]

        return cls(
            owner_flat_ids=owner_flat_ids,
            owner_plane=owner_plane,
            volumes=volume.reshape(-1)[owner_flat_ids],
            centroid_xy=centroid,
            central_second_xy=second,
            central_third_xy=third,
        )


@dataclass(frozen=True)
class AggregateEvaluationDiagnostics:
    """Per-row stability and polynomial-reproduction diagnostics."""

    rank: np.ndarray
    condition: np.ndarray
    reproduction_residual: np.ndarray
    l1_norm: np.ndarray
    donor_counts: np.ndarray
    degree: int
    basis_size: int
    rank_deficient_count: int = 0
    ill_conditioned_count: int = 0
    reproduction_failure_count: int = 0
    l1_exceeded_count: int = 0
    max_condition: float = float("nan")
    max_reproduction_residual: float = float("nan")
    max_l1_norm: float = float("nan")
    initial_donor_count: int = 0
    expansion_steps: np.ndarray | None = None

    @property
    def condition_number(self) -> np.ndarray:
        """Alias for the per-row condition array."""

        return self.condition

    @property
    def L1(self) -> np.ndarray:
        """Alias for :attr:`l1_norm`."""

        return self.l1_norm

    @property
    def l1(self) -> np.ndarray:
        """Alias for :attr:`l1_norm`."""

        return self.l1_norm

    def __getitem__(self, key: str) -> Any:
        """Permit lightweight mapping-style access in diagnostics tooling."""

        return getattr(self, key)

    @property
    def min_rank(self) -> int:
        return int(np.min(self.rank)) if self.rank.size else 0

    @property
    def max_condition_number(self) -> float:
        return self.max_condition

    @property
    def max_reproduction(self) -> float:
        return self.max_reproduction_residual

    @property
    def max_l1(self) -> float:
        return self.max_l1_norm

    @property
    def final_donor_count(self) -> np.ndarray:
        """Alias for the per-row final donor count."""

        return self.donor_counts


@dataclass(frozen=True)
class AggregateEvaluation:
    """Sparse point-evaluation rows and their reconstruction diagnostics."""

    matrix: csr_matrix
    diagnostics: AggregateEvaluationDiagnostics
    points_xy: np.ndarray
    eta_indices: np.ndarray

    @property
    def csr_matrix(self) -> csr_matrix:
        """Alias for :attr:`matrix`."""

        return self.matrix

    @property
    def evaluation_matrix(self) -> csr_matrix:
        """Alias for :attr:`matrix`."""

        return self.matrix

    @property
    def weights(self) -> csr_matrix:
        """Alias for :attr:`matrix`."""

        return self.matrix

    @property
    def rank(self) -> np.ndarray:
        return self.diagnostics.rank

    @property
    def condition(self) -> np.ndarray:
        return self.diagnostics.condition

    @property
    def reproduction_residual(self) -> np.ndarray:
        return self.diagnostics.reproduction_residual

    @property
    def l1_norm(self) -> np.ndarray:
        return self.diagnostics.l1_norm

    def evaluate(self, values_compact: Any) -> np.ndarray:
        """Apply all CSR rows to compact owner values."""

        values = np.asarray(values_compact)
        if values.ndim == 0 or values.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                "values_compact must have first dimension equal to the owner count "
                f"({self.matrix.shape[1]})"
            )
        return np.asarray(self.matrix @ values)

    def __call__(self, values_compact: Any) -> np.ndarray:
        return self.evaluate(values_compact)


def _monomial_exponents(degree: int) -> tuple[tuple[int, int], ...]:
    if int(degree) != degree or degree < 0 or degree > 3:
        raise ValueError("degree must be an integer from zero through three")
    # Constant first, then increasing total degree.  Within a degree, X comes
    # first; only the span matters for the constraints.
    return tuple(
        (px, total - px)
        for total in range(int(degree) + 1)
        for px in range(total, -1, -1)
    )


def _average_basis(
    target_xy: np.ndarray,
    donor_centroid: np.ndarray,
    donor_second: np.ndarray,
    donor_third: np.ndarray | None,
    degree: int,
    scale: np.ndarray,
) -> np.ndarray:
    """Evaluate volume-average monomials in target-centered coordinates."""

    displacement = donor_centroid - target_xy[:, None, :]
    second_raw = donor_second + displacement[..., :, None] * displacement[..., None, :]
    # Shape is (batch, donor, basis).
    columns: list[np.ndarray] = [np.ones(displacement.shape[:2], dtype=np.float64)]
    if degree >= 1:
        columns.extend(
            [displacement[..., axis] / scale[:, None] for axis in range(2)]
        )
    if degree >= 2:
        columns.extend(
            [
                second_raw[..., 0, 0] / scale[:, None]**2,
                second_raw[..., 0, 1] / scale[:, None]**2,
                second_raw[..., 1, 1] / scale[:, None]**2,
            ]
        )
    if degree >= 3:
        if donor_third is None:
            raise ValueError("degree three reconstruction requires central_third_xy")
        third_raw = (
            donor_third
            + displacement[..., :, None, None] * donor_second[:, :, None, :, :]
            + displacement[..., None, :, None] * donor_second[:, :, :, None, :]
            + displacement[..., None, None, :] * donor_second[:, :, :, :, None]
            + displacement[..., :, None, None]
            * displacement[..., None, :, None]
            * displacement[..., None, None, :]
        )
        columns.extend(
            [
                third_raw[..., 0, 0, 0] / scale[:, None]**3,
                third_raw[..., 0, 0, 1] / scale[:, None]**3,
                third_raw[..., 0, 1, 1] / scale[:, None]**3,
                third_raw[..., 1, 1, 1] / scale[:, None]**3,
            ]
        )
    return np.stack(columns, axis=-1)


def _query_tree(
    tree: cKDTree,
    points: np.ndarray,
    donor_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    distances, indices = tree.query(points, k=donor_count)
    if donor_count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    return np.asarray(distances, dtype=np.float64), np.asarray(indices, dtype=np.int64)


def _fit_weight_chunk(
    target: np.ndarray,
    tree: cKDTree,
    compact_owner: np.ndarray,
    geometry: AggregateGeometry,
    degree: int,
    donor_count: int,
    weight_power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one vectorized chunk for one fixed donor count."""

    distance, tree_indices = _query_tree(tree, target, donor_count)
    donor = compact_owner[tree_indices]
    if np.any(~np.isfinite(distance)):
        raise ValueError("nonfinite donor distance")
    # A per-row positive floor keeps coincident-target weights finite and makes
    # the result independent of the chunk partition.
    positive_min = np.min(np.where(distance > 0.0, distance, np.inf), axis=1)
    max_distance = np.max(distance, axis=1)
    floor = np.where(np.isfinite(positive_min), 0.25 * positive_min, 1.0e-12)
    # The absolute floor also guards the dimensionless power from overflow
    # for coordinates whose magnitudes are close to machine underflow.
    floor = np.maximum(floor, 1.0e-12 * max_distance)
    floor = np.maximum(floor, 1.0e-14)
    scale = np.maximum(max_distance, floor)
    basis = _average_basis(
        target,
        geometry.centroid_xy[donor],
        geometry.central_second_xy[donor],
        None
        if geometry.central_third_xy is None
        else geometry.central_third_xy[donor],
        int(degree),
        scale,
    )
    normalized_distance = distance / scale[:, None]
    sqrt_weight = np.power(
        np.maximum(normalized_distance, floor[:, None] / scale[:, None]),
        -0.5 * float(weight_power),
    )
    if np.any(~np.isfinite(sqrt_weight)) or np.any(sqrt_weight <= 0.0):
        raise ValueError("nonfinite donor weight")
    weighted_basis = sqrt_weight[..., None] * basis
    try:
        u, singular, vh = np.linalg.svd(weighted_basis, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("SVD failed for aggregate reconstruction") from exc
    basis_size = basis.shape[-1]
    cutoff = _SVD_RELATIVE_CUTOFF * singular[:, :1]
    row_rank = np.sum(singular > cutoff, axis=1).astype(np.int32)
    row_condition = np.full(target.shape[0], np.inf, dtype=np.float64)
    full_rank = row_rank >= basis_size
    row_condition[full_rank] = (
        singular[full_rank, 0] / singular[full_rank, basis_size - 1]
    )
    inverse = np.einsum(
        "mrb,mr,mkr->mbk",
        vh[:, :basis_size, :],
        np.divide(
            1.0,
            singular[:, :basis_size],
            out=np.zeros_like(singular[:, :basis_size]),
            where=singular[:, :basis_size] > 0.0,
        ),
        u[:, :, :basis_size],
        optimize=True,
    )
    weights = inverse[:, 0, :] * sqrt_weight
    target_constant = np.eye(1, basis_size, 0)[0]
    residual = np.max(
        np.abs(
            np.einsum("mk,mkb->mb", weights, basis, optimize=True)
            - target_constant
        ),
        axis=1,
    )
    l1_norm = np.sum(np.abs(weights), axis=1)
    return donor, weights, row_rank, row_condition, residual, l1_norm


def build_aggregate_evaluation(
    geometry: AggregateGeometry,
    points_xy: Any,
    eta_indices: Any,
    *,
    degree: int = 2,
    donor_count: int = 12,
    condition_limit: float = 1.0e8,
    weight_power: float = 4.0,
    chunk_size: int = 1024,
    row_l1_limit: float = 8.0,
    max_donor_count: int | None = 96,
) -> AggregateEvaluation:
    """Build stable polynomial-reproducing sparse point-evaluation rows.

    Each row starts with ``donor_count`` nearest distinct aggregate owners in
    the row's eta plane.  If its conditioning or L1 amplification is outside
    the requested bounds, the same polynomial degree is retried with a larger
    stencil up to ``max_donor_count`` (the schedule includes 12, 18, 24, 32,
    48, 64, and 96).  The constrained weights reproduce every planar
    monomial through ``degree`` exactly to numerical tolerance.  A locally
    scaled SVD computes the minimum distance-weighted norm solution; rank,
    condition, residual, and L1 amplification are checked before any CSR row
    is accepted.  Failure is explicit when the requested polynomial basis
    cannot be supported.
    """

    if not isinstance(geometry, AggregateGeometry):
        raise TypeError("geometry must be an AggregateGeometry")
    exponents = _monomial_exponents(degree)
    basis_size = len(exponents)
    if int(donor_count) != donor_count or donor_count < basis_size:
        raise ValueError(
            f"donor_count must be an integer at least the degree-{degree} basis size "
            f"({basis_size})"
        )
    donor_count = int(donor_count)
    if max_donor_count is None:
        max_donor_count = donor_count
    if int(max_donor_count) != max_donor_count or max_donor_count < donor_count:
        raise ValueError("max_donor_count must be an integer at least donor_count")
    max_donor_count = int(max_donor_count)
    if not np.isfinite(condition_limit) or condition_limit <= 0.0:
        raise ValueError("condition_limit must be positive and finite")
    if not np.isfinite(weight_power) or weight_power < 0.0:
        raise ValueError("weight_power must be finite and nonnegative")
    if int(chunk_size) != chunk_size or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    chunk_size = int(chunk_size)
    if not np.isfinite(row_l1_limit) or row_l1_limit <= 0.0:
        raise ValueError("row_l1_limit must be positive and finite")

    points = _as_float_array(points_xy, "points_xy")
    if points.ndim == 1:
        if points.shape != (2,):
            raise ValueError("points_xy must have shape (n, 2)")
        points = points.reshape(1, 2)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape (n, 2)")
    if points.shape[0] == 0:
        raise ValueError("at least one evaluation point is required")
    eta_raw = np.asarray(eta_indices)
    if eta_raw.size and eta_raw.dtype.kind in "fc" and (
        not np.all(np.isfinite(eta_raw))
        or not np.all(np.equal(eta_raw, np.floor(eta_raw)))
    ):
        raise ValueError("eta_indices must contain finite integers")
    eta = np.asarray(eta_raw, dtype=np.int64).reshape(-1)
    if eta.shape != (points.shape[0],):
        raise ValueError("eta_indices must have one entry per point")
    if np.any(eta < 0):
        raise ValueError("eta_indices must be nonnegative")

    # Build one tree per plane.  The compact geometry already contains one
    # row per distinct owner, so cKDTree query results are distinct owners.
    trees: dict[int, tuple[cKDTree, np.ndarray]] = {}
    for plane in np.unique(geometry.owner_plane):
        owner = np.flatnonzero(geometry.owner_plane == plane)
        if owner.size < basis_size:
            # Leave the explicit row/plane error to the build below, where it
            # can identify a requested point as well.
            continue
        trees[int(plane)] = (cKDTree(geometry.centroid_xy[owner]), owner)

    n_rows = points.shape[0]
    rank = np.empty(n_rows, dtype=np.int32)
    condition = np.empty(n_rows, dtype=np.float64)
    residual = np.empty(n_rows, dtype=np.float64)
    l1_norm = np.empty(n_rows, dtype=np.float64)
    donor_counts = np.empty(n_rows, dtype=np.int32)
    expansion_steps = np.empty(n_rows, dtype=np.int32)
    row_indices: list[np.ndarray] = []
    col_indices: list[np.ndarray] = []
    data_values: list[np.ndarray] = []

    for plane in np.unique(eta):
        rows = np.flatnonzero(eta == plane)
        if int(plane) not in trees:
            raise ValueError(
                f"eta plane {int(plane)} has fewer than {basis_size} aggregate owners; "
                "cannot support the requested polynomial degree"
            )
        tree, compact_owner = trees[int(plane)]
        if compact_owner.size < basis_size:
            raise ValueError(
                f"eta plane {int(plane)} has only {compact_owner.size} donors, "
                f"fewer than degree-{degree} basis size {basis_size}"
            )
        available = min(max_donor_count, compact_owner.size)
        schedule = [donor_count]
        standard_schedule = (12, 18, 24, 32, 48, 64, 96)
        for candidate in standard_schedule:
            if donor_count < candidate <= available:
                schedule.append(candidate)
        while schedule[-1] < available:
            candidate = min(
                available,
                max(schedule[-1] + 1, int(np.ceil(1.5 * schedule[-1]))),
            )
            if candidate <= schedule[-1]:
                break
            schedule.append(candidate)
        for start in range(0, rows.size, chunk_size):
            row_chunk = rows[start : start + chunk_size]
            pending = np.arange(row_chunk.size, dtype=np.int64)
            last_reason = ""
            for step, k_requested in enumerate(schedule):
                if pending.size == 0:
                    break
                k = min(k_requested, compact_owner.size)
                target = points[row_chunk[pending]]
                donor, weights, row_rank, row_condition, residual_chunk, l1_chunk = (
                    _fit_weight_chunk(
                        target,
                        tree,
                        compact_owner,
                        geometry,
                        int(degree),
                        k,
                        float(weight_power),
                    )
                )
                active_rows = row_chunk[pending]
                rank[active_rows] = row_rank
                condition[active_rows] = row_condition
                residual[active_rows] = residual_chunk
                l1_norm[active_rows] = l1_chunk
                bad_rank = row_rank < basis_size
                bad_condition = ~np.isfinite(row_condition) | (
                    row_condition > condition_limit
                )
                bad_residual = ~np.isfinite(residual_chunk) | (
                    residual_chunk > _REPRODUCTION_TOLERANCE
                )
                bad_l1 = ~np.isfinite(l1_chunk) | (l1_chunk > row_l1_limit)
                bad = bad_rank | bad_condition | bad_residual | bad_l1
                good = ~bad
                if np.any(good):
                    good_rows = active_rows[good]
                    donor_counts[good_rows] = k
                    expansion_steps[good_rows] = step
                    row_indices.append(np.repeat(good_rows, k))
                    col_indices.append(donor[good].reshape(-1))
                    data_values.append(weights[good].reshape(-1))
                pending = pending[bad]
                if pending.size == 0:
                    break
                last_reason = (
                    f"rank={int(np.min(row_rank))}, "
                    f"condition={float(np.max(row_condition)):.6e}, "
                    f"reproduction={float(np.max(residual_chunk)):.6e}, "
                    f"L1={float(np.max(l1_chunk)):.6e}"
                )
            if pending.size:
                bad_row = int(row_chunk[pending[0]])
                raise ValueError(
                    f"aggregate reconstruction failed at row {bad_row} in eta plane "
                    f"{int(plane)} after donor expansion to {schedule[-1]}: {last_reason}"
                )

    matrix = csr_matrix(
        (
            np.concatenate(data_values) if data_values else np.empty(0),
            (
                np.concatenate(row_indices) if row_indices else np.empty(0, dtype=np.int64),
                np.concatenate(col_indices) if col_indices else np.empty(0, dtype=np.int64),
            ),
        ),
        shape=(n_rows, geometry.n_owner),
        dtype=np.float64,
    )
    matrix.sum_duplicates()
    diagnostics = AggregateEvaluationDiagnostics(
        rank=rank,
        condition=condition,
        reproduction_residual=residual,
        l1_norm=l1_norm,
        donor_counts=donor_counts,
        degree=int(degree),
        basis_size=basis_size,
        rank_deficient_count=int(np.sum(rank < basis_size)),
        ill_conditioned_count=int(
            np.sum(~np.isfinite(condition) | (condition > condition_limit))
        ),
        reproduction_failure_count=int(
            np.sum(~np.isfinite(residual) | (residual > _REPRODUCTION_TOLERANCE))
        ),
        l1_exceeded_count=int(
            np.sum(~np.isfinite(l1_norm) | (l1_norm > row_l1_limit))
        ),
        max_condition=float(np.max(condition)),
        max_reproduction_residual=float(np.max(residual)),
        max_l1_norm=float(np.max(l1_norm)),
        initial_donor_count=donor_count,
        expansion_steps=expansion_steps,
    )
    return AggregateEvaluation(
        matrix=matrix,
        diagnostics=diagnostics,
        points_xy=np.array(points, copy=True),
        eta_indices=np.array(eta, copy=True),
    )


__all__ = [
    "AggregateGeometry",
    "AggregateEvaluation",
    "AggregateEvaluationDiagnostics",
    "build_aggregate_evaluation",
]
