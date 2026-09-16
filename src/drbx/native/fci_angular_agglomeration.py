"""Native lowering for production polar angular RLP.

Diffusion and polarization use the conservative matched-space fine-grid action
``M_owner^-1 H.T M_raw A_f H``.  The sparse map ``H`` reconstructs fine
raw-cell averages from active-owner averages while preserving every aggregate
mean and every straight-field plane-wise constant.  Its planar donor topology
is static under eta sharding; eta-varying weights travel in the ordinary
cell-shaped payload.

The generic radial coarse--fine transition-face lowering fits scalar traces
for operators such as the Poisson bracket.  Curvature additionally has a lean
all-radial descriptor: it retains only value traces, quadrature weights, and
``Bmag``, avoiding the generic four-patch gradient/flux allocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import PartitionSpec as P

from ..geometry import (
    LocalControlVolumeCellGeometry3D,
    LocalDomain3D,
    LocalFciGeometry3D,
    compile_local_control_volume_geometry,
)
from ..geometry.fci_control_volumes import (
    PolarAngularAgglomerationGeometry3D,
    polar_regular_chart,
)
from .fci_boundaries import (
    CV_FACE_INTERIOR,
    CV_RECONSTRUCTION_EQUATION_CELL,
    CV_RECONSTRUCTION_EQUATION_REMOTE_CELL,
    LocalControlVolumeBoundaryBC3D,
    LocalControlVolumeFaceRows3D,
    LocalEmbeddedControlVolumeGeometry3D,
    LocalMomentFittedFaceRows3D,
    LocalMomentReconstruction3D,
    LocalRadialCurvatureFaceRows3D,
    LocalRegularBoundaryMomentClosure3D,
)
from .fci_control_volume_operators import (
    control_volume_average_basis,
    monomial_basis,
    monomial_exponents,
)
from .fci_rlp_diffusion import (
    RLPCellAverageProlongation,
    RLPCellAverageReconstructionDiagnostics,
    compile_rlp_cell_average_prolongation,
)


RLP_DIFFUSION_MAX_OBSERVATIONS = 160
# Match the established direct-face trace gate: retain exact lower-order
# moments instead of accepting a formally cubic but strongly amplifying row.
_RADIAL_CURVATURE_WEIGHT_L1_LIMIT = 8.0
# Keep adaptive compilation bounded.  The first pass is still the established
# nearest-64 stencil; these limits only apply after that stencil fails the
# resolved quadratic quality gate.
_RADIAL_CURVATURE_ADAPTIVE_CANDIDATE_BUDGET = 32
_RADIAL_CURVATURE_ADAPTIVE_TRIAL_BUDGET = 192


# The packed payload is deliberately cell-shaped.  This lets callers place it
# on the same eta-only mesh as the ordinary FCI geometry with
# ``NamedSharding(mesh, P("x", "y", "z", None))``.  Volumes support the
# existing projection/restriction path; first through third physical moments
# support the optional moment-fitted transition-face traces.
_RLP_CHANNEL_WIDTHS = (
    ("raw_volume", 1),
    ("aggregate_volume", 1),
    ("raw_centroid", 3),
    ("aggregate_centroid", 3),
    ("raw_second_moment", 9),
    ("aggregate_second_moment", 9),
    ("raw_third_moment", 27),
    ("aggregate_third_moment", 27),
    ("radial_boundary_face_weights", 4),
    ("radial_boundary_owner_weights", 4),
    ("radial_boundary_valid", 1),
    ("diffusion_weights", RLP_DIFFUSION_MAX_OBSERVATIONS),
)
_RLP_CHANNEL_SLICES: dict[str, slice] = {}
_rlp_channel_start = 0
for _rlp_channel_name, _rlp_channel_width in _RLP_CHANNEL_WIDTHS:
    _RLP_CHANNEL_SLICES[_rlp_channel_name] = slice(
        _rlp_channel_start,
        _rlp_channel_start + _rlp_channel_width,
    )
    _rlp_channel_start += _rlp_channel_width
RLP_PACKED_FIELD_COUNT = _rlp_channel_start


@dataclass(frozen=True)
class _PolarAngularCompactTransitionPayload:
    """Concrete one-device rows compiled once from the global RLP geometry."""

    faces: dict[str, np.ndarray]
    functionals: dict[str, np.ndarray]

    @property
    def max_rows(self) -> int:
        return int(self.faces["active"].size)

    @property
    def max_equations(self) -> int:
        return int(self.functionals["observation_active"].shape[1])


@dataclass(frozen=True)
class _PolarAngularRadialCurvaturePayload:
    """Host-side lean direct-face data for all interior radial faces."""

    rows: dict[str, np.ndarray]

    @property
    def max_rows(self) -> int:
        return int(self.rows["active"].size)

    @property
    def max_equations(self) -> int:
        return int(self.rows["observation_active"].shape[1])


@dataclass(frozen=True)
class ShardedPolarAngularAgglomerationDescriptor:
    """Static metadata for an eta-shardable production angular RLP payload.

    The separate packed array returned by the builder is global and
    cell-shaped.  Under an eta-only ``shard_map`` each shard receives a
    contiguous ``(nx, ny, nz_local, 2)`` block.  The radial/angular owner map
    is reconstructed locally from the static profile; because the owner map
    never changes eta, the local eta coordinate is also the local owner-k
    coordinate.
    """

    domain: LocalDomain3D
    angular_group_sizes: tuple[int, ...]
    diffusion_owner_i: np.ndarray = field(repr=False, compare=False)
    diffusion_owner_j: np.ndarray = field(repr=False, compare=False)
    diffusion_owner_eta_offset: np.ndarray = field(repr=False, compare=False)
    diffusion_observation_active: np.ndarray = field(repr=False, compare=False)
    diffusion_diagnostics: RLPCellAverageReconstructionDiagnostics = field(
        repr=False, compare=False
    )
    compile_regular_boundary_closure: bool = False
    compact_transition_payload: _PolarAngularCompactTransitionPayload | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    radial_curvature_payload: _PolarAngularRadialCurvaturePayload | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.domain, LocalDomain3D):
            raise TypeError("domain must be a LocalDomain3D")
        counts = tuple(
            int(value) for value in self.domain.shard_spec.shard_counts
        )
        if counts[0] != 1 or counts[1] != 1:
            raise ValueError(
                "eta-sharded angular agglomeration supports only "
                f"shard_counts=(1, 1, Sz), got {counts}"
            )
        expected = self.domain.shard_spec.global_shape
        profile = tuple(int(value) for value in self.angular_group_sizes)
        if len(profile) != expected[0]:
            raise ValueError(
                "angular_group_sizes must have one entry per radial ring"
            )
        object.__setattr__(self, "angular_group_sizes", profile)
        object.__setattr__(
            self,
            "compile_regular_boundary_closure",
            bool(self.compile_regular_boundary_closure),
        )
        topology_shape = expected[:2] + (RLP_DIFFUSION_MAX_OBSERVATIONS,)
        for name in (
            "diffusion_owner_i",
            "diffusion_owner_j",
            "diffusion_owner_eta_offset",
            "diffusion_observation_active",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != topology_shape:
                raise ValueError(f"{name} must have shape {topology_shape}")
            object.__setattr__(self, name, value)

    @property
    def global_shape(self) -> tuple[int, int, int]:
        return self.domain.shard_spec.global_shape

    @property
    def shard_counts(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.domain.shard_spec.shard_counts)

    @property
    def cell_partition_spec(self):
        """Partitioning for ``cell_fields`` on the eta-only execution mesh."""

        return P("x", "y", "z", None)

    @property
    def packed_cell_shape(self) -> tuple[int, int, int, int]:
        return self.global_shape + (RLP_PACKED_FIELD_COUNT,)

    @property
    def compact_face_count(self) -> int:
        """Number of transition rows owned by one eta shard."""

        payload = self.compact_transition_payload
        if payload is None:
            return 0
        eta_shards = self.shard_counts[2]
        if payload.max_rows % eta_shards != 0:
            raise ValueError("compact transition rows do not divide eta shards")
        return payload.max_rows // eta_shards

    @property
    def global_compact_face_count(self) -> int:
        payload = self.compact_transition_payload
        return 0 if payload is None else payload.max_rows

    @property
    def radial_curvature_face_count(self) -> int:
        """Number of lean all-radial curvature rows owned by one eta shard."""

        payload = self.radial_curvature_payload
        if payload is None:
            return 0
        eta_shards = self.shard_counts[2]
        if payload.max_rows % eta_shards != 0:
            raise ValueError("radial curvature rows do not divide eta shards")
        return payload.max_rows // eta_shards

    @property
    def global_radial_curvature_face_count(self) -> int:
        payload = self.radial_curvature_payload
        return 0 if payload is None else payload.max_rows


def _resolve_direct_face_functionals_flag(
    preferred: bool | None,
    legacy: bool,
    *,
    legacy_name: str,
) -> bool:
    """Resolve the neutral direct-face switch and its legacy alias.

    The old flags remain accepted because geometry builders are used by
    cached/replay paths.  A caller may opt into the neutral spelling without
    changing those paths; the only invalid combination is explicitly turning
    the preferred switch off while retaining a true legacy switch.
    """

    legacy_flag = bool(legacy)
    if preferred is None:
        return legacy_flag
    preferred_flag = bool(preferred)
    if legacy_flag and preferred_flag != legacy_flag:
        raise ValueError(
            "compile_direct_face_functionals disagrees with "
            f"{legacy_name}"
        )
    return preferred_flag


def _validate_direct_face_band_radius(value: int) -> int:
    """Validate and normalize the radial direct-face coverage radius."""

    radius = int(value)
    if radius < 0:
        raise ValueError("direct_face_band_radius must be non-negative")
    return radius


def _fit_transition_face_value_weights(
    centroid: np.ndarray,
    second_moment: np.ndarray,
    third_moment: np.ndarray,
    points: np.ndarray,
    *,
    origin: np.ndarray,
    scale: np.ndarray,
    observation_weight: np.ndarray | None = None,
    max_weight_l1: float | None = None,
) -> tuple[np.ndarray, int, int, float, float]:
    """Fit moment-preserving point traces, with controlled order fallback."""

    displacement = (centroid - origin[None, :]) / scale[None, :]
    distance2 = np.einsum("ni,ni->n", displacement, displacement)
    if observation_weight is None:
        observation_weight = 1.0 / np.maximum(distance2, 0.25)
    else:
        observation_weight = np.asarray(observation_weight, dtype=np.float64)
        if observation_weight.shape != distance2.shape:
            raise ValueError("transition face observation weights have wrong shape")
        if np.any(~np.isfinite(observation_weight)) or np.any(
            observation_weight <= 0.0
        ):
            raise ValueError(
                "transition face observation weights must be finite and positive"
            )
    sqrt_weight = np.sqrt(observation_weight)
    weight_l1_limit = (
        np.inf if max_weight_l1 is None else float(max_weight_l1)
    )
    if max_weight_l1 is not None and (
        not np.isfinite(weight_l1_limit) or weight_l1_limit <= 0.0
    ):
        raise ValueError(
            "transition face weight L1 limit must be finite and positive"
        )
    for order in (3, 2, 1):
        exponents = monomial_exponents(order)
        matrix = control_volume_average_basis(
            centroid,
            second_moment,
            third_moment,
            origin=origin,
            scale=scale,
            exponents=exponents,
        )
        weighted = sqrt_weight[:, None] * matrix
        singular = np.linalg.svd(weighted, compute_uv=False)
        tolerance = 1.0e-12 * singular[0] if singular.size else np.inf
        rank = int(np.sum(singular > tolerance))
        basis_size = len(exponents)
        condition = (
            float(singular[0] / singular[basis_size - 1])
            if rank >= basis_size
            else np.inf
        )
        if rank < basis_size or condition > 1.0e10:
            continue
        inverse = np.linalg.pinv(weighted, rcond=1.0e-12)
        target = monomial_basis(
            (points - origin[None, :]) / scale[None, :],
            exponents=exponents,
        )
        weights = (target @ inverse) * sqrt_weight[None, :]
        residual = float(np.max(np.abs(weights @ matrix - target)))
        weight_l1 = float(np.max(np.sum(np.abs(weights), axis=-1)))
        if (
            np.isfinite(residual)
            and residual <= 1.0e-9
            and weight_l1 <= weight_l1_limit
        ):
            return weights, order, rank, condition, residual
    raise ValueError("RLP transition face trace fit is rank deficient")


def _fit_radial_curvature_face_weights(
    centroid: np.ndarray,
    second_moment: np.ndarray,
    third_moment: np.ndarray,
    points: np.ndarray,
    *,
    origin: np.ndarray,
    scale: np.ndarray,
    outward: np.ndarray,
    max_equations: int = 64,
) -> tuple[
    np.ndarray,
    tuple[
        tuple[np.ndarray, int, int, float, float],
        tuple[np.ndarray, int, int, float, float],
        tuple[np.ndarray, int, int, float, float],
    ],
]:
    """Select a fixed-size resolved donor stencil for one radial face.

    The donor arrays must already be sorted by normalized centroid distance,
    with deterministic tie breaking performed by the caller.  The initial
    trial is exactly the nearest ``max_equations`` donors.  If any of the
    centered, inward-biased, or outward-biased fits is below quadratic order
    or exceeds the radial-curvature weight bound, bounded replacements are
    tried in deterministic order while retaining the same number of slots.
    """

    centroid = np.asarray(centroid, dtype=np.float64)
    second_moment = np.asarray(second_moment, dtype=np.float64)
    third_moment = np.asarray(third_moment, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    outward = np.asarray(outward, dtype=np.float64)
    if centroid.ndim != 2 or centroid.shape[1] != 3:
        raise ValueError("radial curvature donor centroids must have shape (n, 3)")
    donor_count = centroid.shape[0]
    if second_moment.shape != (donor_count, 3, 3):
        raise ValueError("radial curvature donor second moments have wrong shape")
    if third_moment.shape != (donor_count, 3, 3, 3):
        raise ValueError("radial curvature donor third moments have wrong shape")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("radial curvature trace points must have shape (n, 3)")
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("radial curvature origin must be a finite vector")
    if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("radial curvature scale must be finite and positive")
    if outward.shape != (3,) or not np.all(np.isfinite(outward)):
        raise ValueError("radial curvature outward must be a finite vector")
    if not np.any(outward):
        raise ValueError("radial curvature outward must be nonzero")
    max_equations = int(max_equations)
    if max_equations < 1:
        raise ValueError("radial curvature max_equations must be positive")
    slot_count = min(max_equations, donor_count)
    if slot_count == 0:
        raise ValueError(
            "RLP radial curvature donor quality exhaustion: no donors available"
        )

    displacement = (centroid - origin[None, :]) / scale[None, :]
    distance2 = np.einsum("ni,ni->n", displacement, displacement)
    base_weight = 1.0 / np.maximum(distance2, 0.25)
    signed_normal = (
        (centroid - origin[None, :]) @ outward
    ) / max(float(np.linalg.norm(scale)), 1.0e-30)
    observation_weights = (
        base_weight,
        base_weight * np.where(signed_normal <= 0.0, 4.0, 1.0),
        base_weight * np.where(signed_normal >= 0.0, 4.0, 1.0),
    )

    def fit_selection(indices: np.ndarray):
        fit_values = []
        for observation_weight in observation_weights:
            fit = _fit_transition_face_value_weights(
                centroid[indices],
                second_moment[indices],
                third_moment[indices],
                points,
                origin=origin,
                scale=scale,
                observation_weight=observation_weight[indices],
                max_weight_l1=_RADIAL_CURVATURE_WEIGHT_L1_LIMIT,
            )
            fit_values.append(fit)
        fits = tuple(fit_values)
        if any(fit[1] < 2 for fit in fits):
            return None
        if any(
            not np.isfinite(fit[0]).all()
            or float(np.max(np.sum(np.abs(fit[0]), axis=-1)))
            > _RADIAL_CURVATURE_WEIGHT_L1_LIMIT
            for fit in fits
        ):
            return None
        return fits

    def try_selection(indices: np.ndarray):
        try:
            return fit_selection(indices)
        except ValueError:
            return None

    nearest = np.arange(slot_count, dtype=np.intp)
    fits = try_selection(nearest)
    if fits is not None:
        return nearest, fits

    # Candidate 65, 66, ... replace the last slot first.  This is the common
    # lean-face failure mode and preserves every other nearest donor exactly.
    extra = np.arange(slot_count, donor_count, dtype=np.intp)
    extra = extra[:_RADIAL_CURVATURE_ADAPTIVE_CANDIDATE_BUDGET]
    trials = 0
    for candidate in extra:
        indices = nearest.copy()
        indices[-1] = candidate
        fits = try_selection(indices)
        trials += 1
        if fits is not None:
            return indices, fits

    # Broaden support by replacing a deterministic suffix with a contiguous
    # next-nearest block.  The trial cap makes the host compilation cost
    # independent of the total candidate pool size.
    for width in (2, 4, 8, 16, 32, 64):
        if width > slot_count:
            continue
        for start in range(
            slot_count,
            min(donor_count - width + 1, slot_count + 8),
        ):
            indices = np.concatenate(
                (
                    np.arange(slot_count - width, dtype=np.intp),
                    np.arange(start, start + width, dtype=np.intp),
                )
            )
            fits = try_selection(indices)
            trials += 1
            if fits is not None:
                return indices, fits
            if trials >= _RADIAL_CURVATURE_ADAPTIVE_TRIAL_BUDGET:
                break
        if trials >= _RADIAL_CURVATURE_ADAPTIVE_TRIAL_BUDGET:
            break

    # Finally cover individual donor replacements throughout the stencil,
    # starting at the nearest candidate that was not tried in the last slot.
    # This remains a fixed-budget search and has no mesh-index special cases.
    for candidate in extra:
        for position in range(slot_count - 2, -1, -1):
            if trials >= _RADIAL_CURVATURE_ADAPTIVE_TRIAL_BUDGET:
                break
            indices = nearest.copy()
            indices[position] = candidate
            if np.unique(indices).size != slot_count:
                continue
            fits = try_selection(indices)
            trials += 1
            if fits is not None:
                return indices, fits
        if trials >= _RADIAL_CURVATURE_ADAPTIVE_TRIAL_BUDGET:
            break

    raise ValueError(
        "RLP radial curvature donor quality exhaustion: bounded donor "
        "replacements could not provide quadratic centered, inward, and "
        "outward fits with weight L1 <= 8"
    )


def _compile_compact_radial_transition_payload(
    host: PolarAngularAgglomerationGeometry3D,
    *,
    max_equations: int = 64,
    face_band_radius: int = 0,
    direct_face_geometry_sampler=None,
    direct_face_angular_origins: tuple[float, float] | None = None,
) -> _PolarAngularCompactTransitionPayload:
    """Compile moment-fitted rows on radial q transitions and an optional band."""

    topology = host.topology
    shape = topology.shape
    nx, ny, nz = shape
    q = np.asarray(host.angular_group_size, dtype=np.int32)
    face_axis = np.asarray(topology.face_axis, dtype=np.int32)
    storage = np.asarray(topology.face_storage_index, dtype=np.int32)
    minus_id = np.asarray(topology.face_minus_aggregate_id, dtype=np.int64)
    plus_id = np.asarray(topology.face_plus_aggregate_id, dtype=np.int64)
    transition = np.zeros(face_axis.shape, dtype=bool)
    radial = face_axis == 0
    interior = radial & (storage[:, 0] > 0) & (storage[:, 0] < nx)
    transition[interior] = (
        q[storage[interior, 0] - 1] != q[storage[interior, 0]]
    )
    face_band_radius = _validate_direct_face_band_radius(face_band_radius)
    selected_faces = transition.copy()
    if face_band_radius:
        transition_i = np.unique(storage[transition, 0])
        for offset in range(-face_band_radius, face_band_radius + 1):
            selected_faces |= interior & np.isin(
                storage[:, 0], transition_i + offset
            )
    selected = np.flatnonzero(
        selected_faces & (minus_id >= 0) & (plus_id >= 0)
    )
    # Eta-major row order makes each contiguous eta shard a contiguous slice
    # of the compact payload.  Within a plane, preserve deterministic radial
    # then poloidal ordering.
    selected_storage = storage[selected]
    selected = selected[
        np.lexsort(
            (
                selected_storage[:, 1],
                selected_storage[:, 0],
                selected_storage[:, 2],
            )
        )
    ]
    row_count = int(selected.size)
    if row_count == 0:
        raise ValueError("angular RLP profile has no radial coarse-fine transitions")

    radial_faces = np.empty((nx + 1,), dtype=np.float64)
    radial_faces[:-1] = host.radial_centers - 0.5 * host.radial_widths
    radial_faces[-1] = host.radial_centers[-1] + 0.5 * host.radial_widths[-1]
    dtheta = float(host.theta_period) / ny
    deta = float(host.eta_period) / nz
    if direct_face_angular_origins is not None:
        origins = np.asarray(direct_face_angular_origins, dtype=np.float64)
        if origins.shape != (2,) or not np.all(np.isfinite(origins)):
            raise ValueError(
                "direct_face_angular_origins must be a finite (theta, eta) pair"
            )
        theta_start, eta_start = (float(value) for value in origins)
    else:
        # The production toroidal logical grid starts each periodic angle at zero.
        # Recover eta's branch from the integrated chart centroids so cached host
        # geometries with a shifted eta origin retain the same branch.
        eta_centers = np.mean(host.raw_chart_centroid[..., 2], axis=(0, 1))
        eta_start = float(eta_centers[0] - 0.5 * deta)
        outer_first_centroid = np.asarray(host.raw_chart_centroid)[-1, 0, 0]
        theta_start = float(
            np.arctan2(outer_first_centroid[1], outer_first_centroid[0])
            - 0.5 * dtheta
        )
    gauss = 1.0 / np.sqrt(3.0)
    nodes = np.asarray((-gauss, gauss), dtype=np.float64)

    active_owner = np.asarray(topology.is_active_owner, dtype=bool)
    owner_coordinates = np.argwhere(active_owner)
    owner_centroid_all = np.asarray(host.aggregate_chart_centroid)[active_owner]
    owner_second_all = np.asarray(host.aggregate_chart_second_moment)[active_owner]
    owner_third_all = np.asarray(host.aggregate_chart_third_moment)[active_owner]

    max_patches = 4
    quadrature_points = np.zeros((row_count, max_patches, 4, 3))
    logical_quadrature_points = np.zeros((row_count, 4, 3))
    area = np.zeros_like(quadrature_points)
    patch_active = np.zeros((row_count, max_patches), dtype=bool)
    owner_minus = np.zeros((row_count, 3), dtype=np.int32)
    owner_plus = np.zeros((row_count, 3), dtype=np.int32)
    logical_index = np.zeros((row_count, 3), dtype=np.int32)
    face_ids = np.asarray(topology.face_id, dtype=np.int64)[selected]

    observation_active = np.zeros((row_count, max_equations), dtype=bool)
    observation_owner = np.zeros((row_count, max_equations, 3), dtype=np.int32)
    value_weights = np.zeros(
        (row_count, max_patches, 4, max_equations), dtype=np.float64
    )
    upwind_minus_value_weights = np.zeros_like(value_weights)
    upwind_plus_value_weights = np.zeros_like(value_weights)
    polynomial_order = np.zeros((row_count,), dtype=np.int32)
    polynomial_basis_size = np.zeros((row_count,), dtype=np.int32)
    rank = np.zeros((row_count,), dtype=np.int32)
    condition = np.zeros((row_count,), dtype=np.float64)
    residual = np.zeros((row_count,), dtype=np.float64)

    for row, global_row in enumerate(selected):
        fi, fj, fk = (int(value) for value in storage[global_row])
        owner_minus[row] = np.asarray(
            np.unravel_index(int(minus_id[global_row]), shape), dtype=np.int32
        )
        owner_plus[row] = np.asarray(
            np.unravel_index(int(plus_id[global_row]), shape), dtype=np.int32
        )
        logical_index[row] = (fi, fj, fk)

        theta_bounds = theta_start + dtheta * np.asarray((fj, fj + 1))
        eta_bounds = eta_start + deta * np.asarray((fk, fk + 1))
        logical_q = []
        for theta_node in nodes:
            for eta_node in nodes:
                logical_q.append(
                    (
                        radial_faces[fi],
                        np.mean(theta_bounds)
                        + 0.5 * dtheta * theta_node,
                        np.mean(eta_bounds) + 0.5 * deta * eta_node,
                    )
                )
        logical_q = np.asarray(logical_q, dtype=np.float64)
        chart_q = polar_regular_chart(logical_q)
        logical_quadrature_points[row] = logical_q
        quadrature_points[row, 0] = chart_q
        area[row, 0, :, 0] = 0.25 * dtheta * deta
        patch_active[row, 0] = True

        # Candidate discovery uses complete aggregate owners, not fine storage
        # aliases.  Keep nearby radial/eta layers, then retain the closest
        # chart-space observations.  Eta centroids are moved to the face's
        # periodic image before fitting.
        radial_near = np.abs(owner_coordinates[:, 0] - fi) <= 4
        eta_delta_index = np.minimum(
            (owner_coordinates[:, 2] - fk) % nz,
            (fk - owner_coordinates[:, 2]) % nz,
        )
        candidate_mask = radial_near & (eta_delta_index <= 2)
        candidate_owner = owner_coordinates[candidate_mask]
        candidate_centroid = owner_centroid_all[candidate_mask].copy()
        candidate_second = owner_second_all[candidate_mask]
        candidate_third = owner_third_all[candidate_mask]
        face_origin = np.mean(chart_q, axis=0)
        candidate_centroid[:, 2] += host.eta_period * np.round(
            (face_origin[2] - candidate_centroid[:, 2]) / host.eta_period
        )
        local_dr = 0.5 * (
            host.radial_widths[fi - 1] + host.radial_widths[fi]
        )
        scale = np.asarray((local_dr, local_dr, deta), dtype=np.float64)
        distance = np.sum(
            ((candidate_centroid - face_origin[None, :]) / scale[None, :]) ** 2,
            axis=1,
        )
        candidate_order = np.lexsort(
            (
                candidate_owner[:, 2],
                candidate_owner[:, 1],
                candidate_owner[:, 0],
                distance,
            )
        )[:max_equations]
        candidate_owner = candidate_owner[candidate_order]
        candidate_centroid = candidate_centroid[candidate_order]
        candidate_second = candidate_second[candidate_order]
        candidate_third = candidate_third[candidate_order]
        weights, order, fitted_rank, fitted_condition, fitted_residual = (
            _fit_transition_face_value_weights(
                candidate_centroid,
                candidate_second,
                candidate_third,
                chart_q,
                origin=face_origin,
                scale=scale,
            )
        )
        displacement = (candidate_centroid - face_origin[None, :]) / scale[None, :]
        distance2 = np.einsum("ni,ni->n", displacement, displacement)
        base_weight = 1.0 / np.maximum(distance2, 0.25)
        theta_mid = 0.5 * (theta_bounds[0] + theta_bounds[1])
        outward = np.asarray(
            (np.cos(theta_mid), np.sin(theta_mid), 0.0), dtype=np.float64
        )
        signed_normal = (
            (candidate_centroid - face_origin[None, :]) @ outward
        ) / max(local_dr, 1.0e-30)
        # This is the irregular-control-volume analogue of the two ordinary
        # third-order biased stencils.  Both rows use one face-centred moment
        # matrix and the same trace targets.  Only the least-squares norm is
        # biased: the minus row favors inward donors and the plus row favors
        # outward donors, while retaining observations on both sides.
        minus_observation_weight = base_weight * np.where(
            signed_normal <= 0.0, 4.0, 1.0
        )
        plus_observation_weight = base_weight * np.where(
            signed_normal >= 0.0, 4.0, 1.0
        )
        (
            minus_weights,
            minus_order,
            minus_rank,
            minus_condition,
            minus_residual,
        ) = _fit_transition_face_value_weights(
            candidate_centroid,
            candidate_second,
            candidate_third,
            chart_q,
            origin=face_origin,
            scale=scale,
            observation_weight=minus_observation_weight,
        )
        (
            plus_weights,
            plus_order,
            plus_rank,
            plus_condition,
            plus_residual,
        ) = _fit_transition_face_value_weights(
            candidate_centroid,
            candidate_second,
            candidate_third,
            chart_q,
            origin=face_origin,
            scale=scale,
            observation_weight=plus_observation_weight,
        )
        count = candidate_owner.shape[0]
        observation_active[row, :count] = True
        observation_owner[row, :count] = candidate_owner
        value_weights[row, 0, :, :count] = weights
        upwind_minus_value_weights[row, 0, :, :count] = minus_weights
        upwind_plus_value_weights[row, 0, :, :count] = plus_weights
        common_order = min(order, minus_order, plus_order)
        polynomial_order[row] = common_order
        polynomial_basis_size[row] = len(monomial_exponents(common_order))
        rank[row] = min(fitted_rank, minus_rank, plus_rank)
        condition[row] = max(
            fitted_condition, minus_condition, plus_condition
        )
        residual[row] = max(
            fitted_residual, minus_residual, plus_residual
        )

    qshape = (row_count, max_patches, 4)
    identity = np.broadcast_to(np.eye(3), qshape + (3, 3)).copy()
    b_contra = np.zeros(qshape + (3,), dtype=np.float64)
    b_contra[..., 2] = 1.0
    jacobian = np.ones(qshape, dtype=np.float64)
    bmag = np.ones(qshape, dtype=np.float64)
    if direct_face_geometry_sampler is not None:
        sampled = direct_face_geometry_sampler(logical_quadrature_points.copy())

        def sampled_field(name):
            if isinstance(sampled, Mapping):
                if name not in sampled:
                    raise ValueError(
                        f"direct_face_geometry_sampler result is missing {name!r}"
                    )
                value = sampled[name]
            else:
                try:
                    value = getattr(sampled, name)
                except AttributeError as exc:
                    raise ValueError(
                        f"direct_face_geometry_sampler result is missing {name!r}"
                    ) from exc
            return np.asarray(value, dtype=np.float64)

        expected = (row_count, 4)
        sampled_j = sampled_field("J")
        sampled_g_contra = sampled_field("g_contra")
        sampled_g_cov = sampled_field("g_cov")
        sampled_b_contra = sampled_field("B_contra")
        sampled_bmag = sampled_field("Bmag")
        if sampled_j.shape != expected or sampled_bmag.shape != expected:
            raise ValueError(
                "direct_face_geometry_sampler J and Bmag must have shape "
                f"{expected}"
            )
        tensor_expected = expected + (3, 3)
        vector_expected = expected + (3,)
        if sampled_g_contra.shape != tensor_expected or sampled_g_cov.shape != tensor_expected:
            raise ValueError(
                "direct_face_geometry_sampler g_contra and g_cov must have shape "
                f"{tensor_expected}"
            )
        if sampled_b_contra.shape != vector_expected:
            raise ValueError(
                "direct_face_geometry_sampler B_contra must have shape "
                f"{vector_expected}"
            )
        fields = (sampled_j, sampled_g_contra, sampled_g_cov, sampled_b_contra, sampled_bmag)
        if not all(np.all(np.isfinite(value)) for value in fields):
            raise ValueError("direct_face_geometry_sampler returned non-finite geometry")
        if np.any(np.abs(sampled_j) <= 0.0):
            raise ValueError("direct_face_geometry_sampler J magnitude must be positive")
        if np.any(sampled_bmag <= 0.0):
            raise ValueError("direct_face_geometry_sampler Bmag must be positive")
        metric_product = np.einsum("...ik,...kj->...ij", sampled_g_cov, sampled_g_contra)
        if not np.allclose(metric_product, np.eye(3), rtol=1.0e-8, atol=1.0e-10):
            raise ValueError(
                "direct_face_geometry_sampler g_cov and g_contra are not inverses"
            )
        jacobian[:, 0] = sampled_j
        bmag[:, 0] = sampled_bmag
        b_contra[:, 0] = sampled_b_contra
        identity[:, 0] = sampled_g_contra
        g_cov = identity.copy()
        g_cov[:, 0] = sampled_g_cov
        unit_b = sampled_b_contra / sampled_bmag[..., None]
        sampled_projector = sampled_g_contra - np.einsum(
            "...i,...j->...ij", unit_b, unit_b
        )
        projector = identity.copy()
        projector[:, 0] = sampled_projector
    else:
        g_cov = identity.copy()
        projector = identity.copy()
    faces = {
        "kind": np.full((row_count,), CV_FACE_INTERIOR, dtype=np.int32),
        "minus_owner_i": owner_minus[:, 0],
        "minus_owner_j": owner_minus[:, 1],
        "minus_owner_k": owner_minus[:, 2],
        "plus_owner_i": owner_plus[:, 0],
        "plus_owner_j": owner_plus[:, 1],
        "plus_owner_k": owner_plus[:, 2],
        "has_plus_owner": np.ones((row_count,), dtype=bool),
        "quadrature_points": quadrature_points,
        "area_covector_weight": area,
        "J": jacobian,
        "g_contra": identity,
        "g_cov": g_cov,
        "B_contra": b_contra,
        "Bmag": bmag,
        "projector": projector,
        "patch_active": patch_active,
        "active": np.ones((row_count,), dtype=bool),
        "global_face_id": face_ids,
        "logical_axis": np.zeros((row_count,), dtype=np.int32),
        "logical_face_i": logical_index[:, 0],
        "logical_face_j": logical_index[:, 1],
        "logical_face_k": logical_index[:, 2],
    }
    observation_kind = np.where(
        observation_active,
        CV_RECONSTRUCTION_EQUATION_CELL,
        0,
    ).astype(np.int32)
    zero_observation_int = np.zeros(
        (row_count, max_equations), dtype=np.int32
    )
    zero_observation_float = np.zeros(
        (row_count, max_equations), dtype=np.float64
    )
    functionals = {
        "functional_face_id": face_ids,
        "observation_kind": observation_kind,
        "owned_i": observation_owner[..., 0],
        "owned_j": observation_owner[..., 1],
        "owned_k": observation_owner[..., 2],
        "halo_i": zero_observation_int,
        "halo_j": zero_observation_int,
        "halo_k": zero_observation_int,
        "boundary_face_row": zero_observation_int,
        "boundary_patch": zero_observation_int,
        "boundary_quadrature": zero_observation_int,
        "boundary_source_shard": zero_observation_int,
        "observation_active": observation_active,
        "projected_flux_weights": zero_observation_float,
        "parallel_flux_weights": zero_observation_float,
        "parallel_gradient_flux_weights": zero_observation_float,
        "value_weights": value_weights,
        "upwind_minus_value_weights": upwind_minus_value_weights,
        "upwind_plus_value_weights": upwind_plus_value_weights,
        "logical_gradient_weights": np.zeros(
            (row_count, max_patches, 4, 3, max_equations),
            dtype=np.float64,
        ),
        "polynomial_order": polynomial_order,
        "polynomial_basis_size": polynomial_basis_size,
        "rank": rank,
        "condition_number": condition,
        "reproduction_residual": residual,
        "normalized_projected_weight_norm": np.zeros(row_count),
        "normalized_parallel_weight_norm": np.zeros(row_count),
        "normalized_parallel_gradient_weight_norm": np.zeros(row_count),
        "active": np.ones((row_count,), dtype=bool),
    }
    return _PolarAngularCompactTransitionPayload(faces, functionals)


def _compile_radial_curvature_payload(
    host: PolarAngularAgglomerationGeometry3D,
    *,
    max_equations: int = 64,
    direct_face_geometry_sampler=None,
    direct_face_angular_origins: tuple[float, float] | None = None,
) -> _PolarAngularRadialCurvaturePayload:
    """Compile lean cubic traces on every interior radial RLP face.

    The generic compact payload reserves four patches plus gradient and flux
    functionals. Curvature needs only one logical patch, three scalar trace
    fits, a transverse quadrature measure, and ``Bmag``. Keeping a dedicated
    payload makes full-radial coverage practical without changing Poisson's
    transition-only descriptor.
    """

    topology = host.topology
    shape = topology.shape
    nx, ny, nz = shape
    face_axis = np.asarray(topology.face_axis, dtype=np.int32)
    storage = np.asarray(topology.face_storage_index, dtype=np.int32)
    minus_id = np.asarray(topology.face_minus_aggregate_id, dtype=np.int64)
    plus_id = np.asarray(topology.face_plus_aggregate_id, dtype=np.int64)
    selected = np.flatnonzero(
        (face_axis == 0)
        & (storage[:, 0] > 0)
        & (storage[:, 0] < nx)
        & (minus_id >= 0)
        & (plus_id >= 0)
    )
    selected_storage = storage[selected]
    selected = selected[
        np.lexsort(
            (
                selected_storage[:, 1],
                selected_storage[:, 0],
                selected_storage[:, 2],
            )
        )
    ]
    row_count = int(selected.size)
    if row_count == 0:
        raise ValueError("angular RLP profile has no interior radial faces")

    radial_faces = np.empty((nx + 1,), dtype=np.float64)
    radial_faces[:-1] = host.radial_centers - 0.5 * host.radial_widths
    radial_faces[-1] = host.radial_centers[-1] + 0.5 * host.radial_widths[-1]
    dtheta = float(host.theta_period) / ny
    deta = float(host.eta_period) / nz
    if direct_face_angular_origins is not None:
        origins = np.asarray(direct_face_angular_origins, dtype=np.float64)
        if origins.shape != (2,) or not np.all(np.isfinite(origins)):
            raise ValueError(
                "direct_face_angular_origins must be a finite (theta, eta) pair"
            )
        theta_start, eta_start = (float(value) for value in origins)
    else:
        eta_centers = np.mean(host.raw_chart_centroid[..., 2], axis=(0, 1))
        eta_start = float(eta_centers[0] - 0.5 * deta)
        outer_first_centroid = np.asarray(host.raw_chart_centroid)[-1, 0, 0]
        theta_start = float(
            np.arctan2(outer_first_centroid[1], outer_first_centroid[0])
            - 0.5 * dtheta
        )
    gauss = 1.0 / np.sqrt(3.0)
    nodes = np.asarray((-gauss, gauss), dtype=np.float64)

    active_owner = np.asarray(topology.is_active_owner, dtype=bool)
    owner_coordinates = np.argwhere(active_owner)
    owner_centroid_all = np.asarray(host.aggregate_chart_centroid)[active_owner]
    owner_second_all = np.asarray(host.aggregate_chart_second_moment)[active_owner]
    owner_third_all = np.asarray(host.aggregate_chart_third_moment)[active_owner]

    logical_index = np.zeros((row_count, 3), dtype=np.int32)
    owner_minus = np.zeros((row_count, 3), dtype=np.int32)
    owner_plus = np.zeros((row_count, 3), dtype=np.int32)
    observation_active = np.zeros((row_count, max_equations), dtype=bool)
    observation_owner = np.zeros((row_count, max_equations, 3), dtype=np.int32)
    centered_weights = np.zeros((row_count, 4, max_equations), dtype=np.float64)
    minus_weights = np.zeros_like(centered_weights)
    plus_weights = np.zeros_like(centered_weights)
    logical_quadrature_points = np.zeros((row_count, 4, 3), dtype=np.float64)

    for row, global_row in enumerate(selected):
        fi, fj, fk = (int(value) for value in storage[global_row])
        logical_index[row] = (fi, fj, fk)
        owner_minus[row] = np.asarray(
            np.unravel_index(int(minus_id[global_row]), shape), dtype=np.int32
        )
        owner_plus[row] = np.asarray(
            np.unravel_index(int(plus_id[global_row]), shape), dtype=np.int32
        )
        theta_bounds = theta_start + dtheta * np.asarray((fj, fj + 1))
        eta_bounds = eta_start + deta * np.asarray((fk, fk + 1))
        logical_q = np.asarray(
            [
                (
                    radial_faces[fi],
                    np.mean(theta_bounds) + 0.5 * dtheta * theta_node,
                    np.mean(eta_bounds) + 0.5 * deta * eta_node,
                )
                for theta_node in nodes
                for eta_node in nodes
            ],
            dtype=np.float64,
        )
        chart_q = polar_regular_chart(logical_q)
        logical_quadrature_points[row] = logical_q

        radial_near = np.abs(owner_coordinates[:, 0] - fi) <= 4
        eta_delta_index = np.minimum(
            (owner_coordinates[:, 2] - fk) % nz,
            (fk - owner_coordinates[:, 2]) % nz,
        )
        candidate_mask = radial_near & (eta_delta_index <= 2)
        candidate_indices = np.flatnonzero(candidate_mask)
        candidate_owner = owner_coordinates[candidate_indices]
        candidate_centroid = owner_centroid_all[candidate_indices].copy()
        face_origin = np.mean(chart_q, axis=0)
        candidate_centroid[:, 2] += host.eta_period * np.round(
            (face_origin[2] - candidate_centroid[:, 2]) / host.eta_period
        )
        local_dr = 0.5 * (
            host.radial_widths[fi - 1] + host.radial_widths[fi]
        )
        scale = np.asarray((local_dr, local_dr, deta), dtype=np.float64)
        distance = np.sum(
            ((candidate_centroid - face_origin[None, :]) / scale[None, :]) ** 2,
            axis=1,
        )
        # Bound the sorted adaptive pool before indexing the moment tensors.
        # Two max-equation windows cover the nearest-64 fast path, candidate
        # 65 replacement, and the nearest-96 support observed in the audit;
        # donors beyond this deterministic bound are deliberately exhausted.
        candidate_pool = min(2 * max_equations, candidate_indices.size)
        candidate_order = np.lexsort(
            (
                candidate_owner[:, 2],
                candidate_owner[:, 1],
                candidate_owner[:, 0],
                distance,
            )
        )[:candidate_pool]
        sorted_candidate_indices = candidate_indices[candidate_order]
        candidate_owner = owner_coordinates[sorted_candidate_indices]
        # Preserve the periodic eta image applied above; re-indexing the
        # original centroid array here would put seam donors back on the wrong
        # branch before distance fitting.
        candidate_centroid = candidate_centroid[candidate_order]
        candidate_second = owner_second_all[sorted_candidate_indices]
        candidate_third = owner_third_all[sorted_candidate_indices]
        theta_mid = 0.5 * (theta_bounds[0] + theta_bounds[1])
        outward = np.asarray(
            (np.cos(theta_mid), np.sin(theta_mid), 0.0), dtype=np.float64
        )
        selected_indices, fits = (
            _fit_radial_curvature_face_weights(
                candidate_centroid,
                candidate_second,
                candidate_third,
                chart_q,
                origin=face_origin,
                scale=scale,
                outward=outward,
                max_equations=max_equations,
            )
        )
        centered_fit = fits[0][0]
        minus_fit = fits[1][0]
        plus_fit = fits[2][0]
        candidate_owner = candidate_owner[selected_indices]
        candidate_centroid = candidate_centroid[selected_indices]
        count = candidate_owner.shape[0]
        observation_active[row, :count] = True
        observation_owner[row, :count] = candidate_owner
        centered_weights[row, :, :count] = centered_fit
        minus_weights[row, :, :count] = minus_fit
        plus_weights[row, :, :count] = plus_fit

    bmag = np.ones((row_count, 4), dtype=np.float64)
    if direct_face_geometry_sampler is not None:
        sampled = direct_face_geometry_sampler(logical_quadrature_points.copy())
        if isinstance(sampled, Mapping):
            if "Bmag" not in sampled:
                raise ValueError(
                    "direct_face_geometry_sampler result is missing 'Bmag'"
                )
            sampled_bmag = sampled["Bmag"]
        else:
            try:
                sampled_bmag = sampled.Bmag
            except AttributeError as exc:
                raise ValueError(
                    "direct_face_geometry_sampler result is missing 'Bmag'"
                ) from exc
        sampled_bmag = np.asarray(sampled_bmag, dtype=np.float64)
        if sampled_bmag.shape != bmag.shape:
            raise ValueError(
                "direct_face_geometry_sampler Bmag must have shape "
                f"{bmag.shape}"
            )
        if not np.all(np.isfinite(sampled_bmag)) or np.any(sampled_bmag <= 0.0):
            raise ValueError(
                "direct_face_geometry_sampler Bmag must be finite and positive"
            )
        bmag = sampled_bmag

    zero_observation = np.zeros(
        (row_count, max_equations), dtype=np.int32
    )
    rows = {
        "logical_face_i": logical_index[:, 0],
        "logical_face_j": logical_index[:, 1],
        "logical_face_k": logical_index[:, 2],
        "minus_owner_i": owner_minus[:, 0],
        "minus_owner_j": owner_minus[:, 1],
        "minus_owner_k": owner_minus[:, 2],
        "plus_owner_i": owner_plus[:, 0],
        "plus_owner_j": owner_plus[:, 1],
        "plus_owner_k": owner_plus[:, 2],
        "observation_kind": np.where(
            observation_active, CV_RECONSTRUCTION_EQUATION_CELL, 0
        ).astype(np.int32),
        "owned_i": observation_owner[..., 0],
        "owned_j": observation_owner[..., 1],
        "owned_k": observation_owner[..., 2],
        "halo_i": zero_observation.copy(),
        "halo_j": zero_observation.copy(),
        "halo_k": zero_observation.copy(),
        "observation_active": observation_active,
        "centered_weights": centered_weights,
        "minus_weights": minus_weights,
        "plus_weights": plus_weights,
        "quadrature_weight": np.full(
            (row_count, 4), 0.25 * dtheta * deta, dtype=np.float64
        ),
        "Bmag": bmag,
        "active": np.ones((row_count,), dtype=bool),
    }
    return _PolarAngularRadialCurvaturePayload(rows)


def _lower_compact_transition_payload(
    compact: _PolarAngularCompactTransitionPayload,
    local_geometry: LocalFciGeometry3D,
    *,
    global_eta_count: int | None = None,
    eta_shard_count: int = 1,
    eta_axis_name: str | None = None,
) -> tuple[LocalControlVolumeFaceRows3D, LocalMomentFittedFaceRows3D]:
    """Lower the eta-local slice of face-fit rows into local containers."""

    local_eta_count = int(local_geometry.owned_shape[2])
    if global_eta_count is None:
        global_eta_count = local_eta_count
    global_eta_count = int(global_eta_count)
    eta_shard_count = int(eta_shard_count)
    if global_eta_count < 1 or eta_shard_count < 1:
        raise ValueError("eta counts must be positive")
    if global_eta_count != local_eta_count * eta_shard_count:
        raise ValueError(
            "global eta extent must equal local extent times shard count"
        )
    if compact.max_rows % global_eta_count != 0:
        raise ValueError("compact transition rows must be uniform in eta")
    rows_per_eta = compact.max_rows // global_eta_count
    local_row_count = rows_per_eta * local_eta_count
    if eta_shard_count > 1:
        if not eta_axis_name:
            raise ValueError("eta-sharded transition rows require a mesh axis name")
        eta_shard_index = lax.axis_index(eta_axis_name)
    else:
        eta_shard_index = jnp.asarray(0, dtype=jnp.int32)
    global_eta_start = eta_shard_index * local_eta_count

    def eta_local(value):
        array = jnp.asarray(value)
        shaped = array.reshape(
            (global_eta_count, rows_per_eta) + tuple(array.shape[1:])
        )
        selected = lax.dynamic_slice_in_dim(
            shaped,
            global_eta_start,
            local_eta_count,
            axis=0,
        )
        return selected.reshape((local_row_count,) + tuple(array.shape[1:]))

    face_kwargs = {
        name: eta_local(value) for name, value in compact.faces.items()
    }
    face_global_k = face_kwargs["logical_face_k"]
    face_kwargs["logical_face_k"] = face_global_k - global_eta_start
    face_kwargs["minus_owner_k"] = (
        face_kwargs["minus_owner_k"] - global_eta_start
    )
    face_kwargs["plus_owner_k"] = (
        face_kwargs["plus_owner_k"] - global_eta_start
    )
    irregular_faces = LocalControlVolumeFaceRows3D(
        layout=local_geometry.layout,
        max_rows=local_row_count,
        max_patches=4,
        **face_kwargs,
    )
    functional_kwargs = {
        name: eta_local(value) for name, value in compact.functionals.items()
    }
    observation_active = functional_kwargs["observation_active"]
    observation_global_k = functional_kwargs["owned_k"]
    observation_local_k = observation_global_k - global_eta_start
    observation_is_local = (
        (observation_local_k >= 0)
        & (observation_local_k < local_eta_count)
    )
    periodic_delta_k = jnp.mod(
        observation_global_k
        - face_global_k[:, None]
        + global_eta_count // 2,
        global_eta_count,
    ) - global_eta_count // 2
    halo_width = int(local_geometry.layout.halo_width)
    functional_kwargs["observation_kind"] = jnp.where(
        observation_active,
        jnp.where(
            observation_is_local,
            CV_RECONSTRUCTION_EQUATION_CELL,
            CV_RECONSTRUCTION_EQUATION_REMOTE_CELL,
        ),
        0,
    )
    functional_kwargs["owned_k"] = jnp.where(
        observation_is_local, observation_local_k, 0
    )
    functional_kwargs["halo_i"] = functional_kwargs["owned_i"] + halo_width
    functional_kwargs["halo_j"] = functional_kwargs["owned_j"] + halo_width
    functional_kwargs["halo_k"] = (
        face_kwargs["logical_face_k"][:, None]
        + periodic_delta_k
        + halo_width
    )
    face_functionals = LocalMomentFittedFaceRows3D(
        layout=local_geometry.layout,
        max_rows=local_row_count,
        max_equations=compact.max_equations,
        max_patches=4,
        **functional_kwargs,
    )
    return irregular_faces, face_functionals


def _lower_radial_curvature_payload(
    payload: _PolarAngularRadialCurvaturePayload,
    local_geometry: LocalFciGeometry3D,
    *,
    global_eta_count: int | None = None,
    eta_shard_count: int = 1,
    eta_axis_name: str | None = None,
) -> LocalRadialCurvatureFaceRows3D:
    """Lower one eta-local slice of the lean radial curvature descriptor."""

    local_eta_count = int(local_geometry.owned_shape[2])
    if global_eta_count is None:
        global_eta_count = local_eta_count
    global_eta_count = int(global_eta_count)
    eta_shard_count = int(eta_shard_count)
    if global_eta_count != local_eta_count * eta_shard_count:
        raise ValueError(
            "global eta extent must equal local extent times shard count"
        )
    if payload.max_rows % global_eta_count != 0:
        raise ValueError("radial curvature rows must be uniform in eta")
    rows_per_eta = payload.max_rows // global_eta_count
    local_row_count = rows_per_eta * local_eta_count
    if eta_shard_count > 1:
        if not eta_axis_name:
            raise ValueError(
                "eta-sharded radial curvature rows require a mesh axis name"
            )
        eta_shard_index = lax.axis_index(eta_axis_name)
    else:
        eta_shard_index = jnp.asarray(0, dtype=jnp.int32)
    global_eta_start = eta_shard_index * local_eta_count

    def eta_local(value):
        array = jnp.asarray(value)
        shaped = array.reshape(
            (global_eta_count, rows_per_eta) + tuple(array.shape[1:])
        )
        selected = lax.dynamic_slice_in_dim(
            shaped, global_eta_start, local_eta_count, axis=0
        )
        return selected.reshape(
            (local_row_count,) + tuple(array.shape[1:])
        )

    row_kwargs = {name: eta_local(value) for name, value in payload.rows.items()}
    face_global_k = row_kwargs["logical_face_k"]
    row_kwargs["logical_face_k"] = face_global_k - global_eta_start
    row_kwargs["minus_owner_k"] = (
        row_kwargs["minus_owner_k"] - global_eta_start
    )
    row_kwargs["plus_owner_k"] = (
        row_kwargs["plus_owner_k"] - global_eta_start
    )
    observation_active = row_kwargs["observation_active"]
    observation_global_k = row_kwargs["owned_k"]
    observation_local_k = observation_global_k - global_eta_start
    observation_is_local = (
        (observation_local_k >= 0)
        & (observation_local_k < local_eta_count)
    )
    periodic_delta_k = jnp.mod(
        observation_global_k
        - face_global_k[:, None]
        + global_eta_count // 2,
        global_eta_count,
    ) - global_eta_count // 2
    halo_width = int(local_geometry.layout.halo_width)
    row_kwargs["observation_kind"] = jnp.where(
        observation_active,
        jnp.where(
            observation_is_local,
            CV_RECONSTRUCTION_EQUATION_CELL,
            CV_RECONSTRUCTION_EQUATION_REMOTE_CELL,
        ),
        0,
    )
    row_kwargs["owned_k"] = jnp.where(
        observation_is_local, observation_local_k, 0
    )
    row_kwargs["halo_i"] = row_kwargs["owned_i"] + halo_width
    row_kwargs["halo_j"] = row_kwargs["owned_j"] + halo_width
    row_kwargs["halo_k"] = (
        row_kwargs["logical_face_k"][:, None]
        + periodic_delta_k
        + halo_width
    )
    return LocalRadialCurvatureFaceRows3D(
        layout=local_geometry.layout,
        max_rows=local_row_count,
        max_equations=payload.max_equations,
        **row_kwargs,
    )


def _pack_rlp_channels(
    host_geometry: PolarAngularAgglomerationGeometry3D,
    diffusion_prolongation: RLPCellAverageProlongation,
    radial_boundary_channels: Mapping[str, np.ndarray],
) -> jnp.ndarray:
    """Pack host volumes and moments into one eta-shardable cell array."""

    diffusion_weights = np.zeros(
        host_geometry.topology.shape + (RLP_DIFFUSION_MAX_OBSERVATIONS,),
        dtype=np.float64,
    )
    diffusion_weights[
        np.asarray(diffusion_prolongation.raw_i),
        np.asarray(diffusion_prolongation.raw_j),
        np.asarray(diffusion_prolongation.raw_k),
    ] = np.asarray(diffusion_prolongation.weights)
    fields = (
        np.asarray(host_geometry.raw_volume)[..., None],
        np.asarray(host_geometry.aggregate_chart_volume)[..., None],
        np.asarray(host_geometry.raw_chart_centroid),
        np.asarray(host_geometry.aggregate_chart_centroid),
        np.asarray(host_geometry.raw_chart_second_moment).reshape(
            host_geometry.topology.shape + (9,)
        ),
        np.asarray(host_geometry.aggregate_chart_second_moment).reshape(
            host_geometry.topology.shape + (9,)
        ),
        np.asarray(host_geometry.raw_chart_third_moment).reshape(
            host_geometry.topology.shape + (27,)
        ),
        np.asarray(host_geometry.aggregate_chart_third_moment).reshape(
            host_geometry.topology.shape + (27,)
        ),
        np.asarray(radial_boundary_channels["face_weights"]),
        np.asarray(radial_boundary_channels["owner_weights"]),
        np.asarray(radial_boundary_channels["valid"])[..., None],
        diffusion_weights,
    )
    packed = np.concatenate(fields, axis=-1)
    if packed.shape[-1] != RLP_PACKED_FIELD_COUNT:
        raise AssertionError("internal RLP channel packing mismatch")
    return jnp.asarray(packed, dtype=jnp.float64)


def _compile_radial_regular_boundary_channels(
    host_geometry: PolarAngularAgglomerationGeometry3D,
    domain: LocalDomain3D,
    *,
    enabled: bool,
) -> dict[str, np.ndarray]:
    """Compile cubic logical-radial wall derivatives into cell channels."""

    shape = host_geometry.topology.shape
    face_weights = np.zeros(shape + (4,), dtype=np.float64)
    owner_weights = np.zeros(shape + (4,), dtype=np.float64)
    valid = np.zeros(shape, dtype=bool)
    if not enabled or shape[0] < 3:
        return {
            "face_weights": face_weights,
            "owner_weights": owner_weights,
            "valid": valid,
        }

    radial_faces = np.empty((shape[0] + 1,), dtype=np.float64)
    radial_faces[:-1] = (
        np.asarray(host_geometry.radial_centers)
        - 0.5 * np.asarray(host_geometry.radial_widths)
    )
    radial_faces[-1] = (
        host_geometry.radial_centers[-1]
        + 0.5 * host_geometry.radial_widths[-1]
    )
    centroid = np.asarray(host_geometry.raw_radial_centroid)
    central_second = np.asarray(host_geometry.raw_radial_second_moment)
    central_third = np.asarray(host_geometry.raw_radial_third_moment)
    raw_volume = np.asarray(host_geometry.raw_volume)
    periodic_x = bool(domain.periodic_axes[0])
    axis_regular_x = bool(domain.axis_regular_axes[0])

    for side, side_enabled in (
        ("lower", not periodic_x and not axis_regular_x),
        ("upper", not periodic_x),
    ):
        if not side_enabled:
            continue
        if side == "lower":
            indices = np.arange(3, dtype=np.int32)
            storage_index = 0
            wall = radial_faces[0]
        else:
            indices = np.arange(shape[0] - 1, shape[0] - 4, -1, dtype=np.int32)
            storage_index = -1
            wall = radial_faces[-1]
        delta = centroid[indices] - wall
        second_about_wall = central_second[indices] + delta**2
        third_about_wall = (
            central_third[indices]
            + 3.0 * delta * central_second[indices]
            + delta**3
        )
        cell_rows = np.stack(
            (
                np.ones_like(delta),
                delta,
                second_about_wall,
                third_about_wall,
            ),
            axis=-1,
        )
        cell_rows = np.moveaxis(cell_rows, 0, -2)
        boundary_row = np.zeros(cell_rows.shape[:-2] + (1, 4), dtype=np.float64)
        boundary_row[..., 0, 0] = 1.0
        matrix = np.concatenate((boundary_row, cell_rows), axis=-2)
        face_target = np.zeros(matrix.shape[:-2] + (4,), dtype=np.float64)
        face_target[..., 1] = 1.0
        first_delta = delta[0]
        owner_target = np.stack(
            (
                np.zeros_like(first_delta),
                np.ones_like(first_delta),
                2.0 * first_delta,
                3.0 * first_delta**2,
            ),
            axis=-1,
        )
        transposed = np.swapaxes(matrix, -1, -2)
        side_face = np.linalg.solve(transposed, face_target[..., None])[..., 0]
        side_owner = np.linalg.solve(transposed, owner_target[..., None])[..., 0]
        side_valid = (
            np.all(raw_volume[indices] > 1.0e-30, axis=0)
            & np.all(np.isfinite(matrix), axis=(-2, -1))
            & np.all(np.isfinite(side_face), axis=-1)
            & np.all(np.isfinite(side_owner), axis=-1)
        )
        face_weights[storage_index] = np.where(
            side_valid[..., None], side_face, 0.0
        )
        owner_weights[storage_index] = np.where(
            side_valid[..., None], side_owner, 0.0
        )
        valid[storage_index] = side_valid
    return {
        "face_weights": face_weights,
        "owner_weights": owner_weights,
        "valid": valid,
    }


def build_sharded_polar_angular_agglomeration_payload(
    host_geometry: PolarAngularAgglomerationGeometry3D,
    domain: LocalDomain3D,
    *,
    compile_compact_transition_faces: bool = False,
    compile_direct_face_functionals: bool | None = None,
    compile_radial_curvature_faces: bool = False,
    compile_regular_boundary_closure: bool = False,
    direct_face_band_radius: int = 0,
    direct_face_geometry_sampler=None,
    direct_face_angular_origins: tuple[float, float] | None = None,
) -> tuple[ShardedPolarAngularAgglomerationDescriptor, jnp.ndarray]:
    """Build an eta-only sharded payload from host RLP geometry.

    ``compile_direct_face_functionals`` is the preferred operator-neutral
    spelling.  ``compile_compact_transition_faces`` remains a backwards-
    compatible alias.  A positive ``direct_face_band_radius`` includes
    neighboring radial faces around every transition. Setting
    ``compile_radial_curvature_faces`` independently builds the lean
    all-interior-radial payload consumed only by production curvature.

    ``domain`` supplies the global shape and decomposition contract (normally
    ``ShardedFciGeometry3D.domain``).  X/theta decomposition is rejected
    explicitly because an angular aggregate is allowed to span theta cells,
    while eta aggregates are single-cell in this topology.
    """

    compile_direct_face_functionals = _resolve_direct_face_functionals_flag(
        compile_direct_face_functionals,
        compile_compact_transition_faces,
        legacy_name="compile_compact_transition_faces",
    )
    direct_face_band_radius = _validate_direct_face_band_radius(
        direct_face_band_radius
    )
    if not compile_direct_face_functionals and direct_face_band_radius:
        raise ValueError(
            "direct_face_band_radius > 0 requires direct face compilation"
        )
    if not (compile_direct_face_functionals or compile_radial_curvature_faces) and (
        direct_face_geometry_sampler is not None
        or direct_face_angular_origins is not None
    ):
        raise ValueError(
            "direct_face_geometry_sampler and direct_face_angular_origins "
            "require compile_direct_face_functionals=True or "
            "compile_radial_curvature_faces=True"
        )
    if not isinstance(host_geometry, PolarAngularAgglomerationGeometry3D):
        raise TypeError("host_geometry must be PolarAngularAgglomerationGeometry3D")
    if not isinstance(domain, LocalDomain3D):
        raise TypeError("domain must be a LocalDomain3D")
    counts = tuple(int(value) for value in domain.shard_spec.shard_counts)
    if counts[0] != 1 or counts[1] != 1:
        raise ValueError(
            "angular RLP supports eta sharding only; x/theta sharding is not "
            f"supported, got shard_counts={counts}"
        )
    if tuple(domain.shard_spec.global_shape) != tuple(host_geometry.topology.shape):
        raise ValueError("host geometry and sharded domain global shapes do not match")
    compact_payload = None
    if compile_direct_face_functionals:
        compact_payload = _compile_compact_radial_transition_payload(
            host_geometry,
            face_band_radius=direct_face_band_radius,
            direct_face_geometry_sampler=direct_face_geometry_sampler,
            direct_face_angular_origins=direct_face_angular_origins,
        )
    radial_curvature_payload = None
    if compile_radial_curvature_faces:
        radial_curvature_payload = _compile_radial_curvature_payload(
            host_geometry,
            direct_face_geometry_sampler=direct_face_geometry_sampler,
            direct_face_angular_origins=direct_face_angular_origins,
        )
    diffusion = compile_rlp_cell_average_prolongation(
        host_geometry,
        max_observations=RLP_DIFFUSION_MAX_OBSERVATIONS,
    )
    nx, ny, _ = host_geometry.topology.shape
    topology_shape = (nx, ny, RLP_DIFFUSION_MAX_OBSERVATIONS)
    diffusion_owner_i = np.zeros(topology_shape, dtype=np.int32)
    diffusion_owner_j = np.zeros(topology_shape, dtype=np.int32)
    diffusion_owner_eta_offset = np.zeros(topology_shape, dtype=np.int32)
    diffusion_observation_active = np.zeros(topology_shape, dtype=bool)
    raw_i = np.asarray(diffusion.raw_i)
    raw_j = np.asarray(diffusion.raw_j)
    raw_k = np.asarray(diffusion.raw_k)
    plane = raw_k == 0
    plane_i = raw_i[plane]
    plane_j = raw_j[plane]
    diffusion_owner_i[plane_i, plane_j] = np.asarray(diffusion.owner_i)[plane]
    diffusion_owner_j[plane_i, plane_j] = np.asarray(diffusion.owner_j)[plane]
    diffusion_owner_eta_offset[plane_i, plane_j] = np.asarray(
        diffusion.owner_eta_offset
    )[plane]
    diffusion_observation_active[plane_i, plane_j] = np.asarray(
        diffusion.observation_active
    )[plane]
    descriptor = ShardedPolarAngularAgglomerationDescriptor(
        domain=domain,
        angular_group_sizes=tuple(
            int(value) for value in host_geometry.angular_group_size
        ),
        diffusion_owner_i=diffusion_owner_i,
        diffusion_owner_j=diffusion_owner_j,
        diffusion_owner_eta_offset=diffusion_owner_eta_offset,
        diffusion_observation_active=diffusion_observation_active,
        diffusion_diagnostics=diffusion.diagnostics,
        compile_regular_boundary_closure=bool(
            compile_regular_boundary_closure
        ),
        compact_transition_payload=compact_payload,
        radial_curvature_payload=radial_curvature_payload,
    )
    radial_boundary_channels = _compile_radial_regular_boundary_channels(
        host_geometry,
        domain,
        enabled=bool(compile_regular_boundary_closure),
    )
    return descriptor, _pack_rlp_channels(
        host_geometry,
        diffusion,
        radial_boundary_channels,
    )


def _unpack_rlp_channels(cell_fields_owned: jnp.ndarray):
    if (
        cell_fields_owned.ndim != 4
        or cell_fields_owned.shape[-1] != RLP_PACKED_FIELD_COUNT
    ):
        raise ValueError(
            "eta-local RLP cell payload has the wrong channel count"
        )

    def take(name, shape_tail):
        value = cell_fields_owned[..., _RLP_CHANNEL_SLICES[name]]
        return value.reshape(cell_fields_owned.shape[:-1] + shape_tail)

    return {
        "raw_volume": take("raw_volume", ()),
        "aggregate_volume": take("aggregate_volume", ()),
        "raw_centroid": take("raw_centroid", (3,)),
        "aggregate_centroid": take("aggregate_centroid", (3,)),
        "raw_second_moment": take("raw_second_moment", (3, 3)),
        "aggregate_second_moment": take("aggregate_second_moment", (3, 3)),
        "raw_third_moment": take("raw_third_moment", (3, 3, 3)),
        "aggregate_third_moment": take("aggregate_third_moment", (3, 3, 3)),
        "radial_boundary_face_weights": take(
            "radial_boundary_face_weights", (4,)
        ),
        "radial_boundary_owner_weights": take(
            "radial_boundary_owner_weights", (4,)
        ),
        "radial_boundary_valid": take("radial_boundary_valid", ()),
        "diffusion_weights": take(
            "diffusion_weights", (RLP_DIFFUSION_MAX_OBSERVATIONS,)
        ),
    }


def _assemble_radial_regular_boundary_moment_closure(
    unpacked: Mapping[str, jnp.ndarray],
    local_geometry: LocalFciGeometry3D,
) -> LocalRegularBoundaryMomentClosure3D:
    """Lower precompiled logical-radial boundary channels to face arrays."""

    closure = LocalRegularBoundaryMomentClosure3D.empty(
        local_geometry.layout
    )
    face_weights = closure.x_face_weights
    owner_weights = closure.x_owner_weights
    valid = closure.x_valid
    packed_face = jnp.asarray(unpacked["radial_boundary_face_weights"])
    packed_owner = jnp.asarray(unpacked["radial_boundary_owner_weights"])
    packed_valid = jnp.asarray(unpacked["radial_boundary_valid"], dtype=bool)
    for cell_index, face_index in ((0, 0), (-1, -1)):
        face_weights = face_weights.at[face_index].set(
            packed_face[cell_index]
        )
        owner_weights = owner_weights.at[face_index].set(
            packed_owner[cell_index]
        )
        valid = valid.at[face_index].set(packed_valid[cell_index])

    return LocalRegularBoundaryMomentClosure3D(
        layout=local_geometry.layout,
        x_face_weights=face_weights,
        y_face_weights=closure.y_face_weights,
        z_face_weights=closure.z_face_weights,
        x_owner_weights=owner_weights,
        y_owner_weights=closure.y_owner_weights,
        z_owner_weights=closure.z_owner_weights,
        x_valid=valid,
        y_valid=closure.y_valid,
        z_valid=closure.z_valid,
    )


def assemble_local_polar_angular_agglomeration_geometry(
    sharded_geometry: ShardedPolarAngularAgglomerationDescriptor,
    cell_fields_owned: jnp.ndarray,
    local_geometry: LocalFciGeometry3D,
) -> LocalEmbeddedControlVolumeGeometry3D:
    """Assemble one eta shard's embedded RLP geometry.

    This function is suitable for use inside ``shard_map``.  It does not
    inspect a dynamic shard index: eta ownership is local by construction,
    and the only nontrivial owner coordinates are the local radial/theta
    coordinates generated from the static angular profile.
    """

    if not isinstance(
        sharded_geometry, ShardedPolarAngularAgglomerationDescriptor
    ):
        raise TypeError(
            "sharded_geometry must be an eta-sharded angular RLP descriptor"
        )
    if not isinstance(local_geometry, LocalFciGeometry3D):
        raise TypeError("local_geometry must be LocalFciGeometry3D")
    counts = sharded_geometry.shard_counts
    if counts[0] != 1 or counts[1] != 1:
        raise ValueError("eta-sharded RLP requires shard_counts=(1, 1, Sz)")
    shape = tuple(int(value) for value in local_geometry.owned_shape)
    expected = shape + (RLP_PACKED_FIELD_COUNT,)
    if tuple(cell_fields_owned.shape) != expected:
        raise ValueError(
            f"cell_fields_owned must have shape {expected}, got "
            f"{cell_fields_owned.shape}"
        )
    nx, ny, nz = shape
    q = np.asarray(sharded_geometry.angular_group_sizes, dtype=np.int32)
    if q.shape != (nx,):
        raise ValueError("local radial extent must match the global angular profile")
    # These are static topology metadata, not runtime field data.  Building
    # them with NumPy also keeps LocalControlVolumeCellGeometry3D's eager
    # metadata validation legal when this assembler is traced by shard_map.
    ii = np.arange(nx, dtype=np.int32)[:, None, None]
    jj = np.arange(ny, dtype=np.int32)[None, :, None]
    kk = np.arange(nz, dtype=np.int32)[None, None, :]
    q_cell = q[:, None, None]
    owner_i = np.broadcast_to(ii, shape)
    owner_j = np.broadcast_to((jj // q_cell) * q_cell, shape)
    owner_k = np.broadcast_to(kk, shape)
    active = jj == owner_j
    merged = ~active
    active = np.broadcast_to(active, shape)
    merged = np.broadcast_to(merged, shape)
    received = np.where(active, q_cell - 1, 0).astype(np.int32)
    members = np.where(active, q_cell, 0).astype(np.int32)
    # Spell out the C-order flattening so this remains legal under JAX
    # tracing; ravel_multi_index(mode="raise") requires concrete indices.
    local_aggregate_id = owner_i * (ny * nz) + owner_j * nz + owner_k
    unpacked = _unpack_rlp_channels(
        jnp.asarray(cell_fields_owned, dtype=jnp.float64)
    )
    cells = LocalControlVolumeCellGeometry3D(
        layout=local_geometry.layout,
        owner_i=jnp.asarray(owner_i),
        owner_j=jnp.asarray(owner_j),
        owner_k=jnp.asarray(owner_k),
        is_merged_source=jnp.asarray(merged),
        is_active_owner=jnp.asarray(active),
        is_aggregate_target=jnp.asarray(active & (received > 0)),
        received_source_count=jnp.asarray(received),
        member_count=jnp.asarray(members),
        raw_volume=unpacked["raw_volume"],
        aggregate_volume=unpacked["aggregate_volume"],
        raw_centroid=unpacked["raw_centroid"],
        centroid=unpacked["aggregate_centroid"],
        raw_second_moment=unpacked["raw_second_moment"],
        second_moment=unpacked["aggregate_second_moment"],
        raw_third_moment=unpacked["raw_third_moment"],
        third_moment=unpacked["aggregate_third_moment"],
        aggregate_id=jnp.asarray(local_aggregate_id),
        owner_is_remote=jnp.zeros(shape, dtype=bool),
    )
    reconstructed_mask = np.broadcast_to(q[:, None, None] > 1, shape).copy()
    raw_rows = np.argwhere(reconstructed_mask).astype(np.int32)
    row_i = raw_rows[:, 0]
    row_j = raw_rows[:, 1]
    row_k = raw_rows[:, 2]
    diffusion_prolongation = RLPCellAverageProlongation(
        raw_i=jnp.asarray(row_i),
        raw_j=jnp.asarray(row_j),
        raw_k=jnp.asarray(row_k),
        raw_row_active=jnp.ones((raw_rows.shape[0],), dtype=bool),
        owner_i=jnp.asarray(sharded_geometry.diffusion_owner_i[row_i, row_j]),
        owner_j=jnp.asarray(sharded_geometry.diffusion_owner_j[row_i, row_j]),
        owner_eta_offset=jnp.asarray(
            sharded_geometry.diffusion_owner_eta_offset[row_i, row_j]
        ),
        observation_active=jnp.asarray(
            sharded_geometry.diffusion_observation_active[row_i, row_j]
        ),
        weights=unpacked["diffusion_weights"][row_i, row_j, row_k],
        reconstructed_raw_mask=jnp.asarray(reconstructed_mask),
        raw_shape=shape,
        eta_radius=2,
        diagnostics=sharded_geometry.diffusion_diagnostics,
    )
    compact = sharded_geometry.compact_transition_payload
    if compact is None:
        regular_faces = local_geometry.regular_face_geometry
        irregular_faces = LocalControlVolumeFaceRows3D.empty(
            local_geometry.layout
        )
        face_functionals = None
    else:
        irregular_faces, face_functionals = (
            _lower_compact_transition_payload(
                compact,
                local_geometry,
                global_eta_count=sharded_geometry.global_shape[2],
                eta_shard_count=counts[2],
                eta_axis_name=sharded_geometry.domain.mesh_axis_names[2],
            )
        )
        # These rows replace scalar traces in the ordinary radial face array;
        # they are not additional embedded faces.  Keep the regular face open
        # masks unchanged so the established fine-grid metric/divergence path
        # remains authoritative.
        regular_faces = local_geometry.regular_face_geometry
    radial_curvature_faces = None
    radial_curvature_payload = sharded_geometry.radial_curvature_payload
    if radial_curvature_payload is not None:
        radial_curvature_faces = _lower_radial_curvature_payload(
            radial_curvature_payload,
            local_geometry,
            global_eta_count=sharded_geometry.global_shape[2],
            eta_shard_count=counts[2],
            eta_axis_name=sharded_geometry.domain.mesh_axis_names[2],
        )
    regular_boundary_closure = None
    if sharded_geometry.compile_regular_boundary_closure:
        regular_boundary_closure = (
            _assemble_radial_regular_boundary_moment_closure(
                unpacked,
                local_geometry,
            )
        )
    return LocalEmbeddedControlVolumeGeometry3D(
        cells=cells,
        regular_faces=regular_faces,
        irregular_faces=irregular_faces,
        reconstruction=LocalMomentReconstruction3D.empty(
            local_geometry.layout, max_rows=0, max_equations=1
        ),
        face_functionals=face_functionals,
        radial_curvature_faces=radial_curvature_faces,
        regular_boundary_closure=regular_boundary_closure,
        angular_group_sizes=tuple(
            int(value) for value in sharded_geometry.angular_group_sizes
        ),
        diffusion_prolongation=diffusion_prolongation,
    )


def _cell_geometry_with_layout(host, local, layout):
    return LocalControlVolumeCellGeometry3D(
        layout=layout,
        owner_i=local.local_owner_index[..., 0],
        owner_j=local.local_owner_index[..., 1],
        owner_k=local.local_owner_index[..., 2],
        is_merged_source=local.local_merge_source,
        is_active_owner=local.local_active_owner,
        is_aggregate_target=local.local_received_source_count > 0,
        received_source_count=local.local_received_source_count,
        member_count=local.local_member_count,
        raw_volume=host.raw_volume,
        aggregate_volume=host.aggregate_chart_volume,
        raw_centroid=host.raw_chart_centroid,
        centroid=host.aggregate_chart_centroid,
        raw_second_moment=host.raw_chart_second_moment,
        second_moment=host.aggregate_chart_second_moment,
        raw_third_moment=host.raw_chart_third_moment,
        third_moment=host.aggregate_chart_third_moment,
        aggregate_id=local.local_aggregate_id,
        owner_is_remote=local.owner_is_remote,
    )


def lower_polar_angular_agglomeration_geometry(
    host_geometry: PolarAngularAgglomerationGeometry3D,
    local_geometry: LocalFciGeometry3D,
    *,
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    compile_direct_poisson_faces: bool = False,
    compile_direct_face_functionals: bool | None = None,
    compile_radial_curvature_faces: bool = False,
    direct_face_band_radius: int = 0,
    direct_face_geometry_sampler=None,
    direct_face_angular_origins: tuple[float, float] | None = None,
) -> LocalEmbeddedControlVolumeGeometry3D:
    """Lower one complete production RLP owner/volume geometry.

    ``compile_direct_face_functionals`` is the preferred operator-neutral
    spelling; ``compile_direct_poisson_faces`` remains a backwards-compatible
    alias.  The switch prepares cubic, face-centred minus/plus scalar rows on
    radial RLP transition faces.  A positive
    ``direct_face_band_radius`` additionally compiles neighboring radial faces
    for derivative operators that require one reconstruction family on both
    sides of a transition-adjacent owner.  The ordinary
    diffusion/polarization lowering remains unchanged. The independent
    ``compile_radial_curvature_faces`` switch builds the lean full-radial
    curvature descriptor without widening the generic transition payload.
    """

    compile_direct_face_functionals = _resolve_direct_face_functionals_flag(
        compile_direct_face_functionals,
        compile_direct_poisson_faces,
        legacy_name="compile_direct_poisson_faces",
    )
    direct_face_band_radius = _validate_direct_face_band_radius(
        direct_face_band_radius
    )
    if not compile_direct_face_functionals and direct_face_band_radius:
        raise ValueError(
            "direct_face_band_radius > 0 requires direct face compilation"
        )
    if not (compile_direct_face_functionals or compile_radial_curvature_faces) and (
        direct_face_geometry_sampler is not None
        or direct_face_angular_origins is not None
    ):
        raise ValueError(
            "direct_face_geometry_sampler and direct_face_angular_origins "
            "require compile_direct_face_functionals=True or "
            "compile_radial_curvature_faces=True"
        )
    if not isinstance(host_geometry, PolarAngularAgglomerationGeometry3D):
        raise TypeError("host_geometry must be PolarAngularAgglomerationGeometry3D")
    if not isinstance(local_geometry, LocalFciGeometry3D):
        raise TypeError("local_geometry must be LocalFciGeometry3D")
    if tuple(int(value) for value in shard_counts) != (1, 1, 1):
        raise ValueError("angular RLP lowering currently supports one device/subdomain")
    if tuple(int(value) for value in local_geometry.owned_shape) != host_geometry.topology.shape:
        raise ValueError("host and local geometry shapes do not match")

    local = compile_local_control_volume_geometry(
        host_geometry.topology,
        shard_index=(0, 0, 0),
        shard_counts=(1, 1, 1),
        raw_volume=host_geometry.raw_volume,
        raw_centroid=host_geometry.raw_chart_centroid,
        raw_second_moment=host_geometry.raw_chart_second_moment,
        raw_third_moment=host_geometry.raw_chart_third_moment,
    )
    cells = _cell_geometry_with_layout(
        host_geometry, local, local_geometry.layout
    )
    diffusion_prolongation = compile_rlp_cell_average_prolongation(
        host_geometry,
        max_observations=RLP_DIFFUSION_MAX_OBSERVATIONS,
    )
    reconstruction = LocalMomentReconstruction3D.empty(
        local_geometry.layout, max_rows=0, max_equations=1
    )
    irregular_faces = LocalControlVolumeFaceRows3D.empty(local_geometry.layout)
    face_functionals = None
    if compile_direct_face_functionals:
        compact = _compile_compact_radial_transition_payload(
            host_geometry,
            face_band_radius=direct_face_band_radius,
            direct_face_geometry_sampler=direct_face_geometry_sampler,
            direct_face_angular_origins=direct_face_angular_origins,
        )
        irregular_faces, face_functionals = (
            _lower_compact_transition_payload(compact, local_geometry)
        )
    radial_curvature_faces = None
    if compile_radial_curvature_faces:
        radial_curvature_faces = _lower_radial_curvature_payload(
            _compile_radial_curvature_payload(
                host_geometry,
                direct_face_geometry_sampler=direct_face_geometry_sampler,
                direct_face_angular_origins=direct_face_angular_origins,
            ),
            local_geometry,
        )
    return LocalEmbeddedControlVolumeGeometry3D(
        cells=cells,
        regular_faces=local_geometry.regular_face_geometry,
        irregular_faces=irregular_faces,
        reconstruction=reconstruction,
        face_functionals=face_functionals,
        radial_curvature_faces=radial_curvature_faces,
        angular_group_sizes=tuple(
            int(value) for value in np.asarray(host_geometry.angular_group_size)
        ),
        diffusion_prolongation=diffusion_prolongation,
    )


def empty_angular_agglomeration_boundary_bc(
    *, max_rows: int = 0, max_patches: int = 4
) -> LocalControlVolumeBoundaryBC3D:
    """Return the empty irregular-boundary payload required by the container."""

    return LocalControlVolumeBoundaryBC3D.empty(
        max_rows=max_rows, max_patches=max_patches
    )


__all__ = [
    "RLP_PACKED_FIELD_COUNT",
    "RLP_DIFFUSION_MAX_OBSERVATIONS",
    "ShardedPolarAngularAgglomerationDescriptor",
    "build_sharded_polar_angular_agglomeration_payload",
    "assemble_local_polar_angular_agglomeration_geometry",
    "lower_polar_angular_agglomeration_geometry",
    "empty_angular_agglomeration_boundary_bc",
]
