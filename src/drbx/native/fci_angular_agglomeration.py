"""Native lowering for production polar angular RLP.

The host geometry owns the nested angular owner map and physical volume
moments.  Native lowering merely packs that data into the common owner-space
container.  No compact interfaces, moment-fitted face functionals, or
alternate Cartesian reconstruction are constructed: operators act through
``R A_f P`` on the ordinary fine polar grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from ..geometry import (
    LocalControlVolumeCellGeometry3D,
    LocalDomain3D,
    LocalFciGeometry3D,
    compile_local_control_volume_geometry,
)
from ..geometry.fci_control_volumes import PolarAngularAgglomerationGeometry3D
from .fci_boundaries import (
    LocalControlVolumeBoundaryBC3D,
    LocalControlVolumeFaceRows3D,
    LocalEmbeddedControlVolumeGeometry3D,
    LocalMomentReconstruction3D,
)


# The packed payload is deliberately cell-shaped.  This lets callers place it
# on the same eta-only mesh as the ordinary FCI geometry with
# ``NamedSharding(mesh, P("x", "y", "z", None))``.  Only volumes are runtime
# data: projected fine-grid RLP and the owner-space solver do not consume
# moments.  The local container receives finite zero moment placeholders.
_RLP_CHANNEL_SLICES = {
    "raw_volume": slice(0, 1),
    "aggregate_volume": slice(1, 2),
}
RLP_PACKED_FIELD_COUNT = 2


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


def _pack_rlp_channels(
    host_geometry: PolarAngularAgglomerationGeometry3D,
) -> jnp.ndarray:
    """Pack host raw and aggregate volumes into a cell-shaped array."""

    fields = (
        np.asarray(host_geometry.raw_volume)[..., None],
        np.asarray(host_geometry.aggregate_chart_volume)[..., None],
    )
    packed = np.concatenate(fields, axis=-1)
    if packed.shape[-1] != RLP_PACKED_FIELD_COUNT:
        raise AssertionError("internal RLP channel packing mismatch")
    return jnp.asarray(packed, dtype=jnp.float64)


def build_sharded_polar_angular_agglomeration_payload(
    host_geometry: PolarAngularAgglomerationGeometry3D,
    domain: LocalDomain3D,
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
    descriptor = ShardedPolarAngularAgglomerationDescriptor(
        domain=domain,
        angular_group_sizes=tuple(
            int(value) for value in host_geometry.angular_group_size
        ),
    )
    return descriptor, _pack_rlp_channels(host_geometry)


def _unpack_rlp_channels(cell_fields_owned: jnp.ndarray):
    if (
        cell_fields_owned.ndim != 4
        or cell_fields_owned.shape[-1] != RLP_PACKED_FIELD_COUNT
    ):
        raise ValueError(
            "eta-local RLP cell payload must have shape (nx, ny, nz, 2)"
        )

    def take(name, shape_tail):
        value = cell_fields_owned[..., _RLP_CHANNEL_SLICES[name]]
        return value.reshape(cell_fields_owned.shape[:-1] + shape_tail)

    return take("raw_volume", ()), take("aggregate_volume", ())


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
    raw_v, agg_v = _unpack_rlp_channels(
        jnp.asarray(cell_fields_owned, dtype=jnp.float64)
    )
    # Moments are intentionally not part of the runtime payload.  The
    # projected fine-grid RLP action and owner-space solver consume only raw
    # and aggregate volumes; provide finite placeholders for the shared
    # container so all existing metadata contracts remain valid.
    raw_c = jnp.zeros(shape + (3,), dtype=jnp.float64)
    agg_c = jnp.zeros(shape + (3,), dtype=jnp.float64)
    raw_m2 = jnp.zeros(shape + (3, 3), dtype=jnp.float64)
    agg_m2 = jnp.zeros(shape + (3, 3), dtype=jnp.float64)
    raw_m3 = jnp.zeros(shape + (3, 3, 3), dtype=jnp.float64)
    agg_m3 = jnp.zeros(shape + (3, 3, 3), dtype=jnp.float64)
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
        raw_volume=raw_v,
        aggregate_volume=agg_v,
        raw_centroid=raw_c,
        centroid=agg_c,
        raw_second_moment=raw_m2,
        second_moment=agg_m2,
        raw_third_moment=raw_m3,
        third_moment=agg_m3,
        aggregate_id=jnp.asarray(local_aggregate_id),
        owner_is_remote=jnp.zeros(shape, dtype=bool),
    )
    return LocalEmbeddedControlVolumeGeometry3D(
        cells=cells,
        regular_faces=local_geometry.regular_face_geometry,
        irregular_faces=LocalControlVolumeFaceRows3D.empty(local_geometry.layout),
        reconstruction=LocalMomentReconstruction3D.empty(
            local_geometry.layout, max_rows=0, max_equations=1
        ),
        face_functionals=None,
        angular_group_sizes=tuple(
            int(value) for value in sharded_geometry.angular_group_sizes
        ),
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
) -> LocalEmbeddedControlVolumeGeometry3D:
    """Lower one complete production RLP owner/volume geometry."""

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
    return LocalEmbeddedControlVolumeGeometry3D(
        cells=cells,
        regular_faces=local_geometry.regular_face_geometry,
        irregular_faces=LocalControlVolumeFaceRows3D.empty(local_geometry.layout),
        reconstruction=LocalMomentReconstruction3D.empty(
            local_geometry.layout, max_rows=0, max_equations=1
        ),
        face_functionals=None,
        angular_group_sizes=tuple(
            int(value) for value in np.asarray(host_geometry.angular_group_size)
        ),
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
    "ShardedPolarAngularAgglomerationDescriptor",
    "build_sharded_polar_angular_agglomeration_payload",
    "assemble_local_polar_angular_agglomeration_geometry",
    "lower_polar_angular_agglomeration_geometry",
    "empty_angular_agglomeration_boundary_bc",
]
