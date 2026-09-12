"""Native lowering for production polar angular RLP.

Diffusion and polarization use the conservative matched-space fine-grid action
``M_owner^-1 H.T M_raw A_f H``.  The sparse map ``H`` reconstructs fine
raw-cell averages from active-owner averages while preserving every aggregate
mean and every straight-field plane-wise constant.  Its planar donor topology
is static under eta sharding; eta-varying weights travel in the ordinary
cell-shaped payload.

The optional radial coarse--fine transition-face lowering remains available
for explicit experiments.  It fits scalar traces from aggregate averages and
physical moments, then inserts canonical shared fluxes into the fine radial
face array.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    compact_transition_payload: _PolarAngularCompactTransitionPayload | None = field(
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


def _fit_transition_face_value_weights(
    centroid: np.ndarray,
    second_moment: np.ndarray,
    third_moment: np.ndarray,
    points: np.ndarray,
    *,
    origin: np.ndarray,
    scale: np.ndarray,
    observation_weight: np.ndarray | None = None,
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
        if np.isfinite(residual) and residual <= 1.0e-9:
            return weights, order, rank, condition, residual
    raise ValueError("RLP transition face trace fit is rank deficient")


def _compile_compact_radial_transition_payload(
    host: PolarAngularAgglomerationGeometry3D,
    *,
    max_equations: int = 64,
) -> _PolarAngularCompactTransitionPayload:
    """Compile canonical moment-fitted rows only at radial q transitions."""

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
    selected = np.flatnonzero(transition & (minus_id >= 0) & (plus_id >= 0))
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
        "J": np.ones(qshape, dtype=np.float64),
        "g_contra": identity,
        "g_cov": identity,
        "B_contra": b_contra,
        "Bmag": np.ones(qshape, dtype=np.float64),
        "projector": identity,
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


def _pack_rlp_channels(
    host_geometry: PolarAngularAgglomerationGeometry3D,
    diffusion_prolongation: RLPCellAverageProlongation,
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
        diffusion_weights,
    )
    packed = np.concatenate(fields, axis=-1)
    if packed.shape[-1] != RLP_PACKED_FIELD_COUNT:
        raise AssertionError("internal RLP channel packing mismatch")
    return jnp.asarray(packed, dtype=jnp.float64)


def build_sharded_polar_angular_agglomeration_payload(
    host_geometry: PolarAngularAgglomerationGeometry3D,
    domain: LocalDomain3D,
    *,
    compile_compact_transition_faces: bool = False,
) -> tuple[ShardedPolarAngularAgglomerationDescriptor, jnp.ndarray]:
    """Build an eta-only sharded payload from host RLP geometry.

    ``domain`` supplies the global shape and decomposition contract (normally
    ``ShardedFciGeometry3D.domain``).  X/theta decomposition is rejected
    explicitly because an angular aggregate is allowed to span theta cells,
    while eta aggregates are single-cell in this topology.
    """

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
    if compile_compact_transition_faces:
        compact_payload = _compile_compact_radial_transition_payload(
            host_geometry
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
        compact_transition_payload=compact_payload,
    )
    return descriptor, _pack_rlp_channels(host_geometry, diffusion)


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
        "diffusion_weights": take(
            "diffusion_weights", (RLP_DIFFUSION_MAX_OBSERVATIONS,)
        ),
    }


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
    return LocalEmbeddedControlVolumeGeometry3D(
        cells=cells,
        regular_faces=regular_faces,
        irregular_faces=irregular_faces,
        reconstruction=LocalMomentReconstruction3D.empty(
            local_geometry.layout, max_rows=0, max_equations=1
        ),
        face_functionals=face_functionals,
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
) -> LocalEmbeddedControlVolumeGeometry3D:
    """Lower one complete production RLP owner/volume geometry.

    ``compile_direct_poisson_faces`` additionally prepares cubic, face-centred
    minus/plus scalar rows on radial RLP transition faces.  Those rows support
    the experimental direct owner-to-face Poisson-bracket path; the ordinary
    diffusion/polarization lowering remains unchanged.
    """

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
    if compile_direct_poisson_faces:
        compact = _compile_compact_radial_transition_payload(host_geometry)
        irregular_faces, face_functionals = (
            _lower_compact_transition_payload(compact, local_geometry)
        )
    return LocalEmbeddedControlVolumeGeometry3D(
        cells=cells,
        regular_faces=local_geometry.regular_face_geometry,
        irregular_faces=irregular_faces,
        reconstruction=reconstruction,
        face_functionals=face_functionals,
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
