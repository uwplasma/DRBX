"""Sparse cell-average reconstruction for polar angular RLP diffusion.

This module compiles the production linear map ``H`` from values stored at
active RLP owners to smooth fine raw-cell averages.  Diffusion and
polarization use it in the conservative matched-space action
``M_owner^-1 H.T M_raw A_f H``.

Rows on ordinary ``q == 1`` rings are implicit identities.  Only rows on
agglomerated rings are stored.  Every stored row is a weighted sum of nearby
active-owner averages, and the donor eta coordinate is represented by an
offset from the target row.  The runtime evaluator performs the required
width-two periodic eta exchange even when a shard owns only one eta plane.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ..geometry import LocalDomain3D
from ..geometry.fci_control_volumes import PolarAngularAgglomerationGeometry3D
from .fci_control_volume_operators import (
    control_volume_average_basis,
    monomial_exponents,
)


Array = jax.Array


@dataclass(frozen=True)
class RLPCellAverageReconstructionDiagnostics:
    """Host diagnostics for a compiled prolongation.

    Residuals measure reproduction of all chart monomials through
    ``polynomial_degree`` after the conservative correction.  The
    conservation residual measures ``R H - I`` coefficientwise, and the eta
    line residual measures exact preservation of arbitrary fields that are
    constant within each perpendicular plane.  The maximum row L1 norm makes
    target-evaluation amplification visible even when those algebraic
    residuals and the fit condition limit pass.
    """

    requested_polynomial_degree: int
    polynomial_degree: int
    eta_polynomial_degree: int
    fallback_reason: str | None
    basis_size: int
    minimum_rank: int
    maximum_condition_number: float
    maximum_reproduction_residual: float
    maximum_constant_residual: float
    maximum_conservation_residual: float
    maximum_eta_line_residual: float = 0.0
    maximum_row_weight_l1_norm: float = 1.0


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RLPCellAverageProlongation:
    """Compact sparse owner-to-raw-cell-average map.

    ``raw_i/raw_j/raw_k`` identify the fine output rows.  ``owner_i`` and
    ``owner_j`` identify donor owners, while ``owner_eta_offset`` is relative
    to ``raw_k`` and lies in ``[-eta_radius, eta_radius]``.  All donor arrays
    and ``weights`` have shape ``(row_count, max_observations)``.

    Rows absent from this payload are exact identities.  Stored raw-row index
    triples are unique, including inactive rows; this invariant permits the
    indexed update to declare ``unique_indices=True`` so JAX can transpose the
    linear map.  ``observation_active`` permits padded donor columns, while
    neither the host compiler nor the packed sharded lowering pads raw rows.
    """

    raw_i: Array
    raw_j: Array
    raw_k: Array
    raw_row_active: Array
    owner_i: Array
    owner_j: Array
    owner_eta_offset: Array
    observation_active: Array
    weights: Array
    reconstructed_raw_mask: Array
    raw_shape: tuple[int, int, int]
    eta_radius: int
    diagnostics: RLPCellAverageReconstructionDiagnostics

    def __post_init__(self) -> None:
        row_shape = tuple(self.raw_i.shape)
        if len(row_shape) != 1:
            raise ValueError("raw row indices must be one-dimensional")
        for name in ("raw_j", "raw_k", "raw_row_active"):
            if tuple(getattr(self, name).shape) != row_shape:
                raise ValueError(f"{name} must have shape {row_shape}")
        observation_shape = tuple(self.weights.shape)
        if len(observation_shape) != 2 or observation_shape[0] != row_shape[0]:
            raise ValueError("weights must have shape (row_count, max_observations)")
        for name in (
            "owner_i",
            "owner_j",
            "owner_eta_offset",
            "observation_active",
        ):
            if tuple(getattr(self, name).shape) != observation_shape:
                raise ValueError(f"{name} must have shape {observation_shape}")
        if tuple(self.reconstructed_raw_mask.shape) != tuple(self.raw_shape):
            raise ValueError("reconstructed_raw_mask must have raw_shape")
        raw_indices = (self.raw_i, self.raw_j, self.raw_k)
        if not any(isinstance(index, jax.core.Tracer) for index in raw_indices):
            concrete_rows = np.column_stack(
                tuple(np.asarray(index, dtype=np.int64) for index in raw_indices)
            )
            if np.unique(concrete_rows, axis=0).shape[0] != row_shape[0]:
                raise ValueError("stored raw-row index triples must be unique")

    @property
    def row_count(self) -> int:
        return int(self.raw_i.shape[0])

    @property
    def max_observations(self) -> int:
        return int(self.weights.shape[1])

    @property
    def owner_k(self) -> Array:
        """Return global periodic donor-k indices for host/single-device use."""

        return (self.raw_k[:, None] + self.owner_eta_offset) % self.raw_shape[2]

    def tree_flatten(self):
        children = (
            self.raw_i,
            self.raw_j,
            self.raw_k,
            self.raw_row_active,
            self.owner_i,
            self.owner_j,
            self.owner_eta_offset,
            self.observation_active,
            self.weights,
            self.reconstructed_raw_mask,
        )
        auxiliary = (self.raw_shape, self.eta_radius, self.diagnostics)
        return children, auxiliary

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        raw_shape, eta_radius, diagnostics = auxiliary
        return cls(*children, raw_shape, eta_radius, diagnostics)


def _periodic_eta_offsets(neta: int, radius: int) -> tuple[int, ...]:
    """Return unique signed offsets in a deterministic nearest-first order."""

    chosen: list[int] = []
    seen: set[int] = set()
    for distance in range(radius + 1):
        candidates = (0,) if distance == 0 else (-distance, distance)
        for offset in candidates:
            residue = offset % neta
            if residue not in seen:
                seen.add(residue)
                chosen.append(offset)
    return tuple(chosen)


def _planar_owner_stencil(
    host: PolarAngularAgglomerationGeometry3D,
    owner_i: int,
    owner_j: int,
    *,
    planar_count: int,
    radial_radius: int,
) -> np.ndarray:
    """Select a k-independent stencil of active planar owners."""

    active = np.asarray(host.topology.is_active_owner, dtype=bool)
    if not np.all(active == active[..., :1]):
        raise ValueError("RLP active-owner topology must be identical in every eta plane")
    planar = np.argwhere(active[..., 0])
    radial_near = np.abs(planar[:, 0] - owner_i) <= radial_radius
    candidates = planar[radial_near]
    if candidates.shape[0] < planar_count:
        candidates = planar

    # Geometry chooses the nearest chart-space owners, averaged over eta so
    # the selected donor topology is identical in every eta plane.
    centroid = np.asarray(host.aggregate_chart_centroid, dtype=np.float64)
    representative = np.mean(centroid[candidates[:, 0], candidates[:, 1]], axis=1)
    target = np.mean(centroid[owner_i, owner_j], axis=0)
    distance2 = np.sum((representative[:, :2] - target[None, :2]) ** 2, axis=1)
    is_self = (candidates[:, 0] == owner_i) & (candidates[:, 1] == owner_j)
    order = np.lexsort(
        (candidates[:, 1], candidates[:, 0], ~is_self, distance2)
    )
    selected = candidates[order[:planar_count]]
    if not np.any((selected[:, 0] == owner_i) & (selected[:, 1] == owner_j)):
        raise AssertionError("internal donor selection omitted the target owner")
    return selected.astype(np.int32)


def _fit_average_weights(
    observation_basis: np.ndarray,
    target_basis: np.ndarray,
    observation_centroid: np.ndarray,
    *,
    origin: np.ndarray,
    condition_limit: float,
    minimum_rank: int | None = None,
) -> tuple[np.ndarray, int, float, float]:
    displacement = observation_centroid - origin[None, :]
    distance2 = np.einsum("ni,ni->n", displacement, displacement)
    positive = distance2[distance2 > 1.0e-28]
    distance_floor = 0.25 * float(np.min(positive)) if positive.size else 1.0
    observation_weight = 1.0 / np.maximum(distance2, distance_floor)
    sqrt_weight = np.sqrt(observation_weight)
    weighted_basis = sqrt_weight[:, None] * observation_basis
    singular = np.linalg.svd(weighted_basis, compute_uv=False)
    tolerance = 1.0e-12 * singular[0] if singular.size else np.inf
    rank = int(np.sum(singular > tolerance))
    basis_size = int(observation_basis.shape[1])
    required_rank = basis_size if minimum_rank is None else int(minimum_rank)
    condition = (
        float(singular[0] / singular[rank - 1])
        if rank >= required_rank and rank > 0
        else np.inf
    )
    if rank < required_rank:
        raise ValueError(
            "RLP cell-average reconstruction is rank deficient: "
            f"rank={rank}, required_rank={required_rank}, columns={basis_size}"
        )
    if not np.isfinite(condition) or condition > condition_limit:
        raise ValueError(
            "RLP cell-average reconstruction is ill conditioned: "
            f"condition={condition:.6g}, limit={condition_limit:.6g}"
        )
    inverse = np.linalg.pinv(weighted_basis, rcond=1.0e-12)
    weights = (target_basis @ inverse) * sqrt_weight[None, :]
    residual = float(np.max(np.abs(weights @ observation_basis - target_basis)))
    if residual > 1.0e-9:
        raise ValueError(
            "RLP cell-average reconstruction constraints are inconsistent: "
            f"residual={residual:.6g}"
        )
    return weights, rank, condition, residual


def compile_rlp_cell_average_prolongation(
    host: PolarAngularAgglomerationGeometry3D,
    *,
    max_observations: int = 120,
    eta_radius: int = 2,
    radial_radius: int = 4,
    polynomial_degree: int = 3,
    condition_limit: float = 1.0e12,
) -> RLPCellAverageProlongation:
    """Compile a conservative sparse owner-to-raw-average prolongation.

    The fit uses exact owner and raw control-volume average rows constructed
    from their centroid and central moments.  After fitting, every
    aggregate's volume-weighted mean coefficient defect is subtracted from
    each member row.  Thus the returned map obeys ``R H = I`` independently
    of the downstream fine-grid operator.  Eta-block constraints additionally
    preserve the straight-field plane-wise nullspace exactly.  Resolved
    planar grids use the
    requested cubic basis.  Grids smaller than eight cells in both planar
    directions use the highest supported linear or quadratic basis and report
    that choice in ``diagnostics.fallback_reason``.  Eta degree is limited by
    the number of distinct periodic eta planes, so ``neta == 1`` uses the full
    two-dimensional planar basis.
    """

    if not isinstance(host, PolarAngularAgglomerationGeometry3D):
        raise TypeError("host must be PolarAngularAgglomerationGeometry3D")
    max_observations = int(max_observations)
    eta_radius = int(eta_radius)
    radial_radius = int(radial_radius)
    polynomial_degree = int(polynomial_degree)
    condition_limit = float(condition_limit)
    if max_observations < 1:
        raise ValueError("max_observations must be positive")
    if eta_radius < 0 or eta_radius > 2:
        raise ValueError("eta_radius must lie between zero and the halo2 limit")
    if radial_radius < 0:
        raise ValueError("radial_radius must be nonnegative")
    if polynomial_degree != 3:
        raise ValueError("production RLP reconstruction requires polynomial_degree=3")
    if not np.isfinite(condition_limit) or condition_limit <= 1.0:
        raise ValueError("condition_limit must be finite and greater than one")

    shape = tuple(int(value) for value in host.topology.shape)
    nx, ny, nz = shape
    eta_offsets = _periodic_eta_offsets(nz, eta_radius)
    # A periodic stencil with fewer than four distinct eta cells cannot
    # identify all cubic eta powers.  Retain the full cubic x/y basis and the
    # highest eta degree supported by the available planes; neta == 1 is the
    # exact two-dimensional cubic reconstruction used by cheap matrix probes.
    requested_polynomial_degree = polynomial_degree
    fallback_reason = None
    if max(nx, ny) < 8:
        polynomial_degree = 2 if max(nx, ny) > 4 else 1
        fallback_reason = (
            "tiny planar grid has fewer than eight cells in both directions; "
            f"using degree {polynomial_degree}"
        )
    eta_polynomial_degree = min(polynomial_degree, nz - 1)
    basis_exponents = tuple(
        power
        for power in monomial_exponents(polynomial_degree)
        if power[2] <= eta_polynomial_degree
    )
    basis_size = len(basis_exponents)
    planar_count = max_observations // len(eta_offsets)
    planar_basis_size = (polynomial_degree + 1) * (polynomial_degree + 2) // 2
    if (
        planar_count < planar_basis_size
        or planar_count * len(eta_offsets) < basis_size
    ):
        raise ValueError(
            "max_observations is too small for a cubic tensor-spanning stencil"
        )

    q = np.asarray(host.angular_group_size, dtype=np.int32)
    topology = host.topology
    active_owner = np.asarray(topology.is_active_owner, dtype=bool)
    raw_volume = np.asarray(host.raw_volume, dtype=np.float64)
    raw_centroid = np.asarray(host.raw_chart_centroid, dtype=np.float64)
    raw_second = np.asarray(host.raw_chart_second_moment, dtype=np.float64)
    raw_third = np.asarray(host.raw_chart_third_moment, dtype=np.float64)
    owner_centroid = np.asarray(host.aggregate_chart_centroid, dtype=np.float64)
    owner_second = np.asarray(host.aggregate_chart_second_moment, dtype=np.float64)
    owner_third = np.asarray(host.aggregate_chart_third_moment, dtype=np.float64)

    reconstructed_mask = np.broadcast_to(q[:, None, None] > 1, shape).copy()
    raw_rows = np.argwhere(reconstructed_mask).astype(np.int32)
    row_count = int(raw_rows.shape[0])
    if row_count == 0:
        empty = jnp.zeros((0,), dtype=jnp.int32)
        empty_observation = jnp.zeros((0, max_observations), dtype=jnp.int32)
        diagnostics = RLPCellAverageReconstructionDiagnostics(
            requested_polynomial_degree=requested_polynomial_degree,
            polynomial_degree=polynomial_degree,
            eta_polynomial_degree=eta_polynomial_degree,
            fallback_reason=fallback_reason,
            basis_size=basis_size,
            minimum_rank=basis_size,
            maximum_condition_number=1.0,
            maximum_reproduction_residual=0.0,
            maximum_constant_residual=0.0,
            maximum_conservation_residual=0.0,
            maximum_eta_line_residual=0.0,
            maximum_row_weight_l1_norm=1.0,
        )
        return RLPCellAverageProlongation(
            empty,
            empty,
            empty,
            jnp.zeros((0,), dtype=bool),
            empty_observation,
            empty_observation,
            empty_observation,
            jnp.zeros((0, max_observations), dtype=bool),
            jnp.zeros((0, max_observations), dtype=jnp.float64),
            jnp.asarray(reconstructed_mask),
            shape,
            eta_radius,
            diagnostics,
        )

    donor_i = np.zeros((row_count, max_observations), dtype=np.int32)
    donor_j = np.zeros_like(donor_i)
    donor_offset = np.zeros_like(donor_i)
    observation_active = np.zeros_like(donor_i, dtype=bool)
    weights = np.zeros((row_count, max_observations), dtype=np.float64)
    row_lookup = {
        tuple(int(value) for value in raw): row
        for row, raw in enumerate(raw_rows)
    }
    ranks: list[int] = []
    conditions: list[float] = []

    planar_owners = np.argwhere(active_owner[..., 0]).astype(np.int32)
    for planar_owner in planar_owners:
        oi, oj = (int(value) for value in planar_owner)
        if q[oi] == 1:
            continue
        planar_donors = _planar_owner_stencil(
            host,
            oi,
            oj,
            planar_count=planar_count,
            radial_radius=radial_radius,
        )
        donor_pattern = np.asarray(
            [
                (int(di), int(dj), int(offset))
                for di, dj in planar_donors
                for offset in eta_offsets
            ],
            dtype=np.int32,
        )
        donor_count = int(donor_pattern.shape[0])
        self_mask = (
            (donor_pattern[:, 0] == oi)
            & (donor_pattern[:, 1] == oj)
            & (donor_pattern[:, 2] == 0)
        )
        if np.count_nonzero(self_mask) != 1:
            raise AssertionError("target owner must occur exactly once in its stencil")
        self_column = int(np.flatnonzero(self_mask)[0])

        for k in range(nz):
            members = np.column_stack(
                (
                    np.full((int(q[oi]),), oi, dtype=np.int32),
                    np.arange(oj, oj + int(q[oi]), dtype=np.int32),
                    np.full((int(q[oi]),), k, dtype=np.int32),
                )
            )
            rows = np.asarray(
                [row_lookup[tuple(int(value) for value in member)] for member in members],
                dtype=np.int32,
            )
            dk = (k + donor_pattern[:, 2]) % nz
            di = donor_pattern[:, 0]
            dj = donor_pattern[:, 1]
            centroid = owner_centroid[di, dj, dk].copy()
            origin = owner_centroid[oi, oj, k].copy()
            # Choose the intended signed periodic image, including the
            # otherwise ambiguous half-period offset on even eta grids.
            nominal_deta = host.eta_period / nz
            branch_target = origin[2] + donor_pattern[:, 2] * nominal_deta
            centroid[:, 2] += host.eta_period * np.round(
                (branch_target - centroid[:, 2]) / host.eta_period
            )
            target_centroid = raw_centroid[
                members[:, 0], members[:, 1], members[:, 2]
            ].copy()
            target_centroid[:, 2] += host.eta_period * np.round(
                (origin[2] - target_centroid[:, 2]) / host.eta_period
            )

            displacement = centroid - origin[None, :]
            scale = np.sqrt(np.mean(displacement * displacement, axis=0))
            local_floor = np.asarray(
                (
                    host.radial_widths[oi],
                    host.radial_widths[oi],
                    nominal_deta,
                ),
                dtype=np.float64,
            )
            scale = np.maximum(scale, local_floor)
            observation_basis = control_volume_average_basis(
                centroid,
                owner_second[di, dj, dk],
                owner_third[di, dj, dk],
                origin=origin,
                scale=scale,
                exponents=basis_exponents,
            )
            target_basis = control_volume_average_basis(
                target_centroid,
                raw_second[members[:, 0], members[:, 1], members[:, 2]],
                raw_third[members[:, 0], members[:, 1], members[:, 2]],
                origin=origin,
                scale=scale,
                exponents=basis_exponents,
            )
            # Perpendicular diffusion must retain its field-line nullspace.
            # In the straight-field limit, any owner field that is constant
            # over each perpendicular plane (with arbitrary eta dependence)
            # must reconstruct to that same plane-wise constant.  Polynomial
            # reproduction alone only guarantees this for low eta modes and
            # lets the reconstructed-space energy form spuriously couple the
            # remaining modes.  Enforce zero total weight in every nonlocal
            # eta-offset block; constant reproduction then fixes the offset-0
            # block sum to one.  These constraints are compatible with the
            # mixed chart-polynomial moments and leave cross-eta donors
            # available for genuinely mixed spatial variation.
            nonzero_offsets = tuple(offset for offset in eta_offsets if offset != 0)
            if nonzero_offsets:
                eta_constraints = np.column_stack(
                    tuple(
                        donor_pattern[:, 2] == offset
                        for offset in nonzero_offsets
                    )
                ).astype(np.float64)
                fit_basis = np.column_stack((observation_basis, eta_constraints))
                fit_target = np.column_stack(
                    (
                        target_basis,
                        np.zeros(
                            (target_basis.shape[0], len(nonzero_offsets)),
                            dtype=np.float64,
                        ),
                    )
                )
            else:
                fit_basis = observation_basis
                fit_target = target_basis
            fitted, rank, condition, _ = _fit_average_weights(
                fit_basis,
                fit_target,
                centroid,
                origin=origin,
                condition_limit=condition_limit,
                minimum_rank=basis_size,
            )
            # First make the constant row explicit, then enforce R H = I by
            # subtracting the same coefficient defect from every member.
            fitted[:, self_column] += 1.0 - np.sum(fitted, axis=1)
            member_volume = raw_volume[
                members[:, 0], members[:, 1], members[:, 2]
            ]
            alpha = member_volume / np.sum(member_volume)
            mean_weights = alpha @ fitted
            desired = np.zeros((donor_count,), dtype=np.float64)
            desired[self_column] = 1.0
            fitted += (desired - mean_weights)[None, :]
            # A second round removes only floating-point residue left by the
            # two exact linear constraints.
            fitted[:, self_column] += 1.0 - np.sum(fitted, axis=1)
            fitted += (desired - alpha @ fitted)[None, :]

            donor_i[rows, :donor_count] = di[None, :]
            donor_j[rows, :donor_count] = dj[None, :]
            donor_offset[rows, :donor_count] = donor_pattern[:, 2][None, :]
            observation_active[rows, :donor_count] = True
            weights[rows, :donor_count] = fitted
            ranks.append(rank)
            conditions.append(condition)

    reproduction_residual = 0.0
    conservation_residual = 0.0
    eta_line_residual = 0.0
    for planar_owner in planar_owners:
        oi, oj = (int(value) for value in planar_owner)
        if q[oi] == 1:
            continue
        for k in range(nz):
            members = np.column_stack(
                (
                    np.full((int(q[oi]),), oi, dtype=np.int32),
                    np.arange(oj, oj + int(q[oi]), dtype=np.int32),
                    np.full((int(q[oi]),), k, dtype=np.int32),
                )
            )
            rows = np.asarray(
                [row_lookup[tuple(int(value) for value in member)] for member in members]
            )
            active_columns = observation_active[rows[0]]
            di = donor_i[rows[0], active_columns]
            dj = donor_j[rows[0], active_columns]
            offsets = donor_offset[rows[0], active_columns]
            dk = (k + offsets) % nz
            origin = owner_centroid[oi, oj, k].copy()
            centroid = owner_centroid[di, dj, dk].copy()
            branch_target = origin[2] + offsets * (host.eta_period / nz)
            centroid[:, 2] += host.eta_period * np.round(
                (branch_target - centroid[:, 2]) / host.eta_period
            )
            target_centroid = raw_centroid[
                members[:, 0], members[:, 1], members[:, 2]
            ].copy()
            target_centroid[:, 2] += host.eta_period * np.round(
                (origin[2] - target_centroid[:, 2]) / host.eta_period
            )
            displacement = centroid - origin[None, :]
            scale = np.maximum(
                np.sqrt(np.mean(displacement * displacement, axis=0)),
                (host.radial_widths[oi], host.radial_widths[oi], host.eta_period / nz),
            )
            observation_basis = control_volume_average_basis(
                centroid,
                owner_second[di, dj, dk],
                owner_third[di, dj, dk],
                origin=origin,
                scale=scale,
                exponents=basis_exponents,
            )
            target_basis = control_volume_average_basis(
                target_centroid,
                raw_second[members[:, 0], members[:, 1], members[:, 2]],
                raw_third[members[:, 0], members[:, 1], members[:, 2]],
                origin=origin,
                scale=scale,
                exponents=basis_exponents,
            )
            local_weights = weights[rows][:, active_columns]
            reproduction_residual = max(
                reproduction_residual,
                float(np.max(np.abs(local_weights @ observation_basis - target_basis))),
            )
            alpha = raw_volume[
                members[:, 0], members[:, 1], members[:, 2]
            ]
            alpha = alpha / np.sum(alpha)
            desired = np.zeros((active_columns.sum(),), dtype=np.float64)
            desired[
                np.flatnonzero((di == oi) & (dj == oj) & (offsets == 0))[0]
            ] = 1.0
            conservation_residual = max(
                conservation_residual,
                float(np.max(np.abs(alpha @ local_weights - desired))),
            )
            for offset in eta_offsets:
                expected = 1.0 if offset == 0 else 0.0
                eta_line_residual = max(
                    eta_line_residual,
                    float(
                        np.max(
                            np.abs(
                                np.sum(local_weights[:, offsets == offset], axis=1)
                                - expected
                            )
                        )
                    ),
                )

    constant_residual = float(
        np.max(np.abs(np.sum(weights, axis=1) - 1.0))
    )
    diagnostics = RLPCellAverageReconstructionDiagnostics(
        requested_polynomial_degree=requested_polynomial_degree,
        polynomial_degree=polynomial_degree,
        eta_polynomial_degree=eta_polynomial_degree,
        fallback_reason=fallback_reason,
        basis_size=basis_size,
        minimum_rank=min(ranks),
        maximum_condition_number=max(conditions),
        maximum_reproduction_residual=reproduction_residual,
        maximum_constant_residual=constant_residual,
        maximum_conservation_residual=conservation_residual,
        maximum_eta_line_residual=eta_line_residual,
        maximum_row_weight_l1_norm=float(
            np.max(np.sum(np.abs(weights), axis=1))
        ),
    )
    return RLPCellAverageProlongation(
        raw_i=jnp.asarray(raw_rows[:, 0]),
        raw_j=jnp.asarray(raw_rows[:, 1]),
        raw_k=jnp.asarray(raw_rows[:, 2]),
        raw_row_active=jnp.ones((row_count,), dtype=bool),
        owner_i=jnp.asarray(donor_i),
        owner_j=jnp.asarray(donor_j),
        owner_eta_offset=jnp.asarray(donor_offset),
        observation_active=jnp.asarray(observation_active),
        weights=jnp.asarray(weights, dtype=jnp.float64),
        reconstructed_raw_mask=jnp.asarray(reconstructed_mask),
        raw_shape=shape,
        eta_radius=eta_radius,
        diagnostics=diagnostics,
    )


def apply_rlp_cell_average_prolongation(
    owner_field: Array,
    prolongation: RLPCellAverageProlongation,
    *,
    domain: LocalDomain3D | None = None,
) -> Array:
    """Apply ``H`` using only JAX gather, weighted sum, and indexed update.

    ``owner_field`` has leading shape ``raw_shape`` and may have any trailing
    component axes.  Its non-owner values are ignored on reconstructed rows.
    The implementation is linear and its transpose is available through JAX
    automatic differentiation.  Angular RLP assumes periodic eta: the global
    evaluator wraps locally, while an eta-sharded ``domain`` exchanges the
    required planes through the periodic shard permutation.
    """

    if not isinstance(prolongation, RLPCellAverageProlongation):
        raise TypeError("prolongation must be RLPCellAverageProlongation")
    field = jnp.asarray(owner_field)
    if tuple(field.shape[:3]) != prolongation.raw_shape:
        raise ValueError(
            f"owner_field must have leading shape {prolongation.raw_shape}, "
            f"got {field.shape}"
        )
    if prolongation.row_count == 0:
        return field
    if domain is not None and not bool(domain.shard_spec.periodic_axes[2]):
        raise ValueError("RLP diffusion reconstruction requires periodic eta")
    if domain is None or int(domain.shard_spec.shard_counts[2]) == 1:
        donor_k = prolongation.owner_k
        gathered = field[
            prolongation.owner_i,
            prolongation.owner_j,
            donor_k,
        ]
    else:
        if tuple(domain.owned_shape) != prolongation.raw_shape:
            raise ValueError("domain.owned_shape must match prolongation.raw_shape")
        axis_name = domain.mesh_axis_names[2]
        shard_count = int(domain.shard_spec.shard_counts[2])
        if axis_name is None:
            raise ValueError("eta-sharded prolongation requires an eta mesh axis name")

        def shift_one(values: Array, direction: int) -> Array:
            lower = (slice(None), slice(None), slice(0, 1)) + (slice(None),) * (
                values.ndim - 3
            )
            upper = (slice(None), slice(None), slice(-1, None)) + (
                slice(None),
            ) * (values.ndim - 3)
            without_lower = (slice(None), slice(None), slice(1, None)) + (
                slice(None),
            ) * (values.ndim - 3)
            without_upper = (slice(None), slice(None), slice(None, -1)) + (
                slice(None),
            ) * (values.ndim - 3)
            if direction > 0:
                received = lax.ppermute(
                    values[lower],
                    axis_name=axis_name,
                    perm=[(source, (source - 1) % shard_count) for source in range(shard_count)],
                )
                return jnp.concatenate((values[without_lower], received), axis=2)
            received = lax.ppermute(
                values[upper],
                axis_name=axis_name,
                perm=[(source, (source + 1) % shard_count) for source in range(shard_count)],
            )
            return jnp.concatenate((received, values[without_upper]), axis=2)

        shifted = {0: field}
        for offset in range(1, prolongation.eta_radius + 1):
            shifted[offset] = shift_one(shifted[offset - 1], 1)
            shifted[-offset] = shift_one(shifted[-offset + 1], -1)
        stacked = jnp.stack(
            tuple(shifted[offset] for offset in range(-prolongation.eta_radius, prolongation.eta_radius + 1)),
            axis=0,
        )
        offset_index = prolongation.owner_eta_offset + prolongation.eta_radius
        gathered = stacked[
            offset_index,
            prolongation.owner_i,
            prolongation.owner_j,
            jnp.broadcast_to(prolongation.raw_k[:, None], offset_index.shape),
        ]
    trailing = (1,) * (field.ndim - 3)
    coefficients = jnp.where(
        prolongation.observation_active,
        prolongation.weights,
        0.0,
    ).reshape(prolongation.weights.shape + trailing)
    # Evaluate the affine combination in difference form.  The compiled rows
    # have unit weight sum, so this is mathematically identical to ``H u``;
    # unlike a raw weighted sum it preserves a constant field algebraically.
    # That distinction matters at the polar axis, where a 1e-15 constant-mode
    # residue can be amplified by the inverse radial metric.
    reference = gathered[:, :1, ...]
    reconstructed = reference[:, 0, ...] + jnp.sum(
        coefficients * (gathered - reference),
        axis=1,
    )
    current = field[
        prolongation.raw_i,
        prolongation.raw_j,
        prolongation.raw_k,
    ]
    active = prolongation.raw_row_active.reshape(
        prolongation.raw_row_active.shape + trailing
    )
    reconstructed = jnp.where(active, reconstructed, current)
    return field.at[
        prolongation.raw_i,
        prolongation.raw_j,
        prolongation.raw_k,
    ].set(reconstructed, unique_indices=True)


__all__ = [
    "RLPCellAverageProlongation",
    "RLPCellAverageReconstructionDiagnostics",
    "apply_rlp_cell_average_prolongation",
    "compile_rlp_cell_average_prolongation",
]
