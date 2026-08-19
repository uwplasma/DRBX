"""Native lowering for arbitrary eta-plane-local owner-map agglomeration.

This is the topology-neutral counterpart to the polar angular RLP lowering.
The host supplies a direct owner map for every fine cell; native code only
turns the eta-local payload into the shared control-volume container.  The
operator remains projected fine grid, ``R A_f P``.  In particular this module
does not create irregular faces or reconstruction rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from ..geometry import LocalControlVolumeCellGeometry3D, LocalDomain3D, LocalFciGeometry3D
from .fci_boundaries import (
    LocalControlVolumeFaceRows3D,
    LocalEmbeddedControlVolumeGeometry3D,
    LocalMomentReconstruction3D,
)


# Keep this compatible with the driver's existing single cell-shaped
# ``control_volume_fields`` argument.  i/j indices are exactly representable
# in float64 and are converted back to int32 during lowering.
CORNER_EDGE_PACKED_FIELD_COUNT = 4


@dataclass(frozen=True)
class ShardedPlaneLocalOwnerMapDescriptor:
    """Static metadata for an arbitrary eta-sharded direct owner-map payload."""

    domain: LocalDomain3D

    def __post_init__(self) -> None:
        if not isinstance(self.domain, LocalDomain3D):
            raise TypeError("domain must be a LocalDomain3D")
        counts = self.shard_counts
        if counts[0] != 1 or counts[1] != 1:
            raise ValueError(
                "plane-local owner-map agglomeration supports only "
                f"shard_counts=(1, 1, Sz), got {counts}"
            )

    @property
    def global_shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.domain.shard_spec.global_shape)

    @property
    def shard_counts(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.domain.shard_spec.shard_counts)

    @property
    def cell_partition_spec(self):
        return P("x", "y", "z", None)

    @property
    def packed_cell_shape(self) -> tuple[int, int, int, int]:
        return self.global_shape + (CORNER_EDGE_PACKED_FIELD_COUNT,)


def _validate_host_owner_map(owner_i, owner_j, raw_volume, aggregate_volume, shape):
    """Validate the direct map before placing it in a device payload."""

    oi = np.asarray(owner_i, dtype=np.int32)
    oj = np.asarray(owner_j, dtype=np.int32)
    raw = np.asarray(raw_volume, dtype=np.float64)
    agg = np.asarray(aggregate_volume, dtype=np.float64)
    if oi.shape != shape or oj.shape != shape:
        raise ValueError("owner_i and owner_j must match the domain global shape")
    if raw.shape != shape or agg.shape != shape:
        raise ValueError("raw_volume and aggregate_volume must match the domain global shape")
    nx, ny, nz = shape
    ii, jj, kk = np.indices(shape, dtype=np.int32)
    if np.any(oi < 0) or np.any(oi >= nx) or np.any(oj < 0) or np.any(oj >= ny):
        raise ValueError("owner-map coordinates must be in the global i/j bounds")
    # Owner coordinates identify a cell in the *same* eta plane.  The owner
    # cell must map to itself; this catches dangling maps and gives a unique,
    # deterministic active owner for every aggregate.
    if not np.all(oi[oi, oj, kk] == oi) or not np.all(oj[oi, oj, kk] == oj):
        raise ValueError("each plane-local owner map must point to a self-owning cell")
    if not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError("raw_volume must be finite and positive")
    active = (oi == ii) & (oj == jj)
    if np.any(agg[active] <= 0.0) or not np.all(np.isfinite(agg)):
        raise ValueError("aggregate_volume must be finite and positive on owners")


def build_sharded_plane_local_owner_map_payload(
    owner_i,
    owner_j,
    raw_volume,
    aggregate_volume,
    domain: LocalDomain3D,
) -> tuple[ShardedPlaneLocalOwnerMapDescriptor, jnp.ndarray]:
    """Pack arbitrary direct owner-map and volume arrays for eta sharding.

    ``owner_i`` and ``owner_j`` are global i/j owner coordinates.  The eta
    coordinate is intentionally omitted: an aggregate is always restricted to
    its local eta plane, so lowering constructs ``owner_k = arange(nz_local)``.
    """

    descriptor = ShardedPlaneLocalOwnerMapDescriptor(domain=domain)
    shape = descriptor.global_shape
    _validate_host_owner_map(owner_i, owner_j, raw_volume, aggregate_volume, shape)
    payload = jnp.asarray(
        np.stack((
            np.asarray(owner_i, dtype=np.float64), np.asarray(owner_j, dtype=np.float64),
            np.asarray(raw_volume, dtype=np.float64), np.asarray(aggregate_volume, dtype=np.float64),
        ), axis=-1),
        dtype=jnp.float64,
    )
    return descriptor, payload


def _unpack_payload(packed_cells: jnp.ndarray, shape):
    expected = shape + (CORNER_EDGE_PACKED_FIELD_COUNT,)
    if tuple(packed_cells.shape) != expected:
        raise ValueError(f"packed_cells must have shape {expected}, got {packed_cells.shape}")
    return (
        packed_cells[..., 0].astype(jnp.int32),
        packed_cells[..., 1].astype(jnp.int32),
        packed_cells[..., 2].astype(jnp.float64),
        packed_cells[..., 3].astype(jnp.float64),
    )


def assemble_local_plane_local_owner_map_geometry(
    sharded_geometry: ShardedPlaneLocalOwnerMapDescriptor,
    cell_fields_owned: jnp.ndarray,
    local_geometry: LocalFciGeometry3D,
) -> LocalEmbeddedControlVolumeGeometry3D:
    """Lower an arbitrary owner-map payload inside an eta-only ``shard_map``."""

    if not isinstance(sharded_geometry, ShardedPlaneLocalOwnerMapDescriptor):
        raise TypeError("sharded_geometry must be a plane-local owner-map descriptor")
    if not isinstance(local_geometry, LocalFciGeometry3D):
        raise TypeError("local_geometry must be LocalFciGeometry3D")
    shape = tuple(int(v) for v in local_geometry.owned_shape)
    nx, ny, nz = shape
    owner_i, owner_j, raw_volume, aggregate_volume = _unpack_payload(
        jnp.asarray(cell_fields_owned), shape
    )
    ii = jnp.arange(nx, dtype=jnp.int32)[:, None, None]
    jj = jnp.arange(ny, dtype=jnp.int32)[None, :, None]
    kk = jnp.arange(nz, dtype=jnp.int32)[None, None, :]
    owner_k = jnp.broadcast_to(kk, shape)
    active = (owner_i == ii) & (owner_j == jj)
    aggregate_id = owner_i.astype(jnp.int64) * (ny * nz) + owner_j.astype(jnp.int64) * nz + owner_k.astype(jnp.int64)
    # Counts are computed from the map, so arbitrary connected (including
    # L-shaped) groups need no profile-specific metadata.  For an eta shard,
    # aggregate ids are local by construction.
    flat_ids = aggregate_id.reshape((-1,))
    counts = jnp.zeros((nx * ny * nz,), dtype=jnp.int32).at[flat_ids].add(1)
    member_total = counts[flat_ids].reshape(shape)
    member_count = jnp.where(active, member_total, 0)
    received = jnp.where(active, member_total - 1, 0)
    merged = ~active
    zeros_c = jnp.zeros(shape + (3,), dtype=jnp.float64)
    zeros_m2 = jnp.zeros(shape + (3, 3), dtype=jnp.float64)
    zeros_m3 = jnp.zeros(shape + (3, 3, 3), dtype=jnp.float64)
    cells = LocalControlVolumeCellGeometry3D(
        layout=local_geometry.layout,
        owner_i=owner_i, owner_j=owner_j, owner_k=owner_k,
        is_merged_source=merged, is_active_owner=active,
        is_aggregate_target=received > 0,
        received_source_count=received, member_count=member_count,
        raw_volume=raw_volume, aggregate_volume=aggregate_volume,
        raw_centroid=zeros_c, centroid=zeros_c,
        raw_second_moment=zeros_m2, second_moment=zeros_m2,
        raw_third_moment=zeros_m3, third_moment=zeros_m3,
        aggregate_id=aggregate_id,
        owner_is_remote=jnp.zeros(shape, dtype=bool),
    )
    return LocalEmbeddedControlVolumeGeometry3D(
        cells=cells,
        regular_faces=local_geometry.regular_face_geometry,
        irregular_faces=LocalControlVolumeFaceRows3D.empty(local_geometry.layout),
        reconstruction=LocalMomentReconstruction3D.empty(local_geometry.layout, max_rows=0, max_equations=1),
        face_functionals=None,
        angular_group_sizes=None,
        agglomeration_kind="corner-edge",
    )


__all__ = [
    "CORNER_EDGE_PACKED_FIELD_COUNT",
    "ShardedPlaneLocalOwnerMapDescriptor",
    "build_sharded_plane_local_owner_map_payload",
    "assemble_local_plane_local_owner_map_geometry",
]
