"""Single-device lowering for production polar angular RLP.

The host geometry owns the nested angular owner map and physical volume
moments.  Native lowering merely packs that data into the common owner-space
container.  No compact interfaces, moment-fitted face functionals, or
alternate Cartesian reconstruction are constructed: operators act through
``R A_f P`` on the ordinary fine polar grid.
"""

from __future__ import annotations

import numpy as np

from ..geometry import (
    LocalControlVolumeCellGeometry3D,
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
    "lower_polar_angular_agglomeration_geometry",
    "empty_angular_agglomeration_boundary_bc",
]
