"""Host-side corner/edge control volumes for square FCI charts.

This deliberately builds only a *direct owner map*.  The seven-field
production operators can consequently continue to use their projected
fine-grid ``R A_f P`` form; no cut-face reconstruction is implied here.
The chart moments below are intentionally zero placeholders because that
path consumes aggregate volumes and ownership only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fci_control_volumes import (
    GlobalControlVolumeTopology3D,
    build_global_control_volume_topology_from_owner_map,
)


@dataclass(frozen=True)
class CornerEdgeAgglomerationGeometry3D:
    """A plane-local square-chart agglomeration and its diagnostics.

    ``aggregate_chart_volume`` is cell-shaped, with a nonzero value only at
    active owners.  This is the convention used by the existing RLP host
    geometries and makes the object suitable for a future common lowering.
    """

    topology: GlobalControlVolumeTopology3D
    raw_volume: np.ndarray
    aggregate_chart_volume: np.ndarray
    parallel_rate: np.ndarray
    projected_parallel_rate: np.ndarray
    seed_mask: np.ndarray
    target_volume_lower: float
    target_volume_upper: float
    maximum_volume_upper: float
    max_wall_normal_depth: int
    max_tangential_corner_reach: int


def _as_numpy(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _face_cell_average(values: np.ndarray, axis: int) -> np.ndarray:
    """Average cell data to faces, using one-sided values at physical walls."""

    lower = np.take(values, 0, axis=axis)
    upper = np.take(values, -1, axis=axis)
    interior = 0.5 * (
        np.take(values, np.arange(values.shape[axis] - 1), axis=axis)
        + np.take(values, np.arange(1, values.shape[axis]), axis=axis)
    )
    return np.concatenate((np.expand_dims(lower, axis), interior, np.expand_dims(upper, axis)), axis=axis)


def _physical_face_measures(geometry: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx, dy, dz = (_as_numpy(value) for value in (geometry.spacing.dx, geometry.spacing.dy, geometry.spacing.dz))
    return (
        np.abs(_as_numpy(geometry.face_metric.x.J)) * _face_cell_average(dy, 0) * _face_cell_average(dz, 0),
        np.abs(_as_numpy(geometry.face_metric.y.J)) * _face_cell_average(dx, 1) * _face_cell_average(dz, 1),
        np.abs(_as_numpy(geometry.face_metric.z.J)) * _face_cell_average(dx, 2) * _face_cell_average(dy, 2),
    )


def _parallel_rate(geometry: Any) -> np.ndarray:
    b = _as_numpy(geometry.cell_bfield.B_contra)
    bmag = _as_numpy(geometry.cell_bfield.Bmag)
    if np.any(~np.isfinite(bmag)) or np.any(bmag <= 0.0):
        raise ValueError("cell Bmag must be finite and positive")
    spacings = tuple(_as_numpy(value) for value in (geometry.spacing.dx, geometry.spacing.dy, geometry.spacing.dz))
    if any(np.any(value <= 0.0) or np.any(~np.isfinite(value)) for value in spacings):
        raise ValueError("logical spacings must be finite and positive")
    return sum(np.abs(b[..., axis] / bmag) / spacings[axis] for axis in range(3))


def _is_corner_attached_edge_cell(
    i: int,
    j: int,
    nx: int,
    ny: int,
    *,
    max_wall_normal_depth: int,
    max_tangential_corner_reach: int,
) -> bool:
    """Whether a cell lies in one of the four corner-attached edge strips."""

    # The defaults are calibrated to a 64-cell square chart.  Expressing
    # them as fractions of the cell-index span retains the intended footprint
    # on small synthetic grids and on non-64 production exploratory meshes.
    ui = min(i, nx - 1 - i) / max(nx - 1, 1)
    vj = min(j, ny - 1 - j) / max(ny - 1, 1)
    normal_fraction = max_wall_normal_depth / 63.0
    tangential_fraction = max_tangential_corner_reach / 63.0
    near_u_wall = ui <= normal_fraction
    near_v_wall = vj <= normal_fraction
    near_v_corner_end = vj <= tangential_fraction
    near_u_corner_end = ui <= tangential_fraction
    return (near_u_wall and near_v_corner_end) or (near_v_wall and near_u_corner_end)


def _face_conductance(geometry: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y face ordering weights ``abs(J_face b^axis_face)``."""

    x = np.abs(_as_numpy(geometry.face_metric.x.J) * _as_numpy(geometry.face_bfield.x.B_contra)[..., 0] /
               _as_numpy(geometry.face_bfield.x.Bmag))
    y = np.abs(_as_numpy(geometry.face_metric.y.J) * _as_numpy(geometry.face_bfield.y.B_contra)[..., 1] /
               _as_numpy(geometry.face_bfield.y.Bmag))
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("face parallel conductances must be finite")
    return x, y


def _parallel_face_transport(
    geometry: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return absolute integrated ``J b^axis`` transport on coordinate faces."""

    measures = _physical_face_measures(geometry)
    fields = (
        geometry.face_bfield.x,
        geometry.face_bfield.y,
        geometry.face_bfield.z,
    )
    result = []
    for axis, (measure, field) in enumerate(zip(measures, fields)):
        unit_b = _as_numpy(field.B_contra)[..., axis] / _as_numpy(field.Bmag)
        value = np.abs(unit_b) * measure
        if np.any(~np.isfinite(value)):
            raise ValueError("face parallel transport must be finite")
        result.append(value)
    return tuple(result)


def _singleton_projected_parallel_rate(
    raw_volume: np.ndarray,
    face_transport: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Conservative centered-flux absolute row sum for singleton cells."""

    x, y, z = face_transport
    numerator = 0.5 * (x[:-1] + x[1:])
    numerator = numerator + 0.5 * (y[:, :-1] + y[:, 1:])
    numerator = numerator + 0.5 * (z[:, :, :-1] + z[:, :, 1:])
    # A physical characteristic wall may retain the complete outgoing owner
    # perturbation rather than the centered half contribution.
    numerator[0] += 0.5 * x[0]
    numerator[-1] += 0.5 * x[-1]
    numerator[:, 0] += 0.5 * y[:, 0]
    numerator[:, -1] += 0.5 * y[:, -1]
    return numerator / np.maximum(raw_volume, 1.0e-300)


def build_corner_edge_agglomeration(
    geometry: Any,
    *,
    rate_threshold: float,
    volume_ratio: float = 1.2,
    max_wall_normal_depth: int = 13,
    max_tangential_corner_reach: int = 63,
    maximum_volume_ratio: float = 2.0,
) -> CornerEdgeAgglomerationGeometry3D:
    """Build connected, eta-local aggregates around square-chart CFL seeds.

    Seeds are processed from largest conservative face-transport/volume rate
    to smallest.
    A group grows over unclaimed four-neighbours, choosing the largest
    ``abs(J_face b^axis_face)`` interface first, until it has at least two
    members and preferably lies in ``[median(volume)/volume_ratio,
    median(volume)*volume_ratio]``.  When no two-cell combination can satisfy
    that preferred upper edge, the least-overshooting connected choice may
    extend to ``maximum_volume_ratio * median(volume)``; this is necessary to
    change the CFL of a seed that is already near the preferred lower edge.
    The maximum band is a soft guard: a seed is never left as a singleton
    merely because its own volume or the last available connected pairing
    already exceeds that band.
    """

    if not np.isfinite(rate_threshold) or rate_threshold <= 0.0:
        raise ValueError("rate_threshold must be finite and positive")
    if not np.isfinite(volume_ratio) or volume_ratio <= 1.0:
        raise ValueError("volume_ratio must be greater than one")
    if not np.isfinite(maximum_volume_ratio) or maximum_volume_ratio < volume_ratio:
        raise ValueError("maximum_volume_ratio must be finite and at least volume_ratio")
    if max_wall_normal_depth < 0 or max_tangential_corner_reach < 0:
        raise ValueError("corner-strip depths must be nonnegative")

    shape = tuple(int(v) for v in geometry.shape)
    if len(shape) != 3:
        raise ValueError("geometry must be three dimensional")
    dx, dy, dz = (_as_numpy(value) for value in (geometry.spacing.dx, geometry.spacing.dy, geometry.spacing.dz))
    raw_volume = np.abs(_as_numpy(geometry.cell_metric.J)) * dx * dy * dz
    if raw_volume.shape != shape or np.any(~np.isfinite(raw_volume)) or np.any(raw_volume <= 0.0):
        raise ValueError("square corner agglomeration requires finite positive raw cell volumes")
    median_volume = float(np.median(raw_volume))
    lower, upper = median_volume / volume_ratio, median_volume * volume_ratio
    maximum_upper = median_volume * maximum_volume_ratio
    nx, ny, nz = shape
    face_transport = _parallel_face_transport(geometry)
    rate = _singleton_projected_parallel_rate(raw_volume, face_transport)
    seed_mask = rate > rate_threshold

    outside = [
        (int(i), int(j), int(k))
        for i, j, k in np.argwhere(seed_mask)
        if not _is_corner_attached_edge_cell(
            int(i), int(j), nx, ny,
            max_wall_normal_depth=max_wall_normal_depth,
            max_tangential_corner_reach=max_tangential_corner_reach,
        )
    ]
    if outside:
        preview = ", ".join(map(str, outside[:8]))
        raise ValueError(
            f"{len(outside)} of {int(np.count_nonzero(seed_mask))} CFL seed(s) lie "
            "outside configured corner-attached edge strips; "
            f"first: {preview}"
        )

    self_index = np.stack(np.indices(shape, dtype=np.int32), axis=-1)
    owner = self_index.copy()
    claimed = np.zeros(shape, dtype=bool)
    group_members: dict[tuple[int, int, int], set[tuple[int, int, int]]] = {}
    group_volumes: dict[tuple[int, int, int], float] = {}
    gx, gy = _face_conductance(geometry)

    def projected_group_rate(
        members: set[tuple[int, int, int]],
        group_volume: float,
    ) -> float:
        """Conservative centered-flux row-sum proxy for one aggregate."""

        total = 0.0
        for i, j, k in members:
            # Interior centered faces contribute half their transport to the
            # row.  A physical wall can retain the full outgoing owner mode,
            # so use the full boundary transport as the safe closure bound.
            for axis, coordinate in ((0, i), (1, j)):
                n_axis = shape[axis]
                lower = [i, j, k]
                lower[axis] -= 1
                upper = [i, j, k]
                upper[axis] += 1
                lower_weight = 1.0 if coordinate == 0 else 0.5
                upper_weight = 1.0 if coordinate == n_axis - 1 else 0.5
                if coordinate == 0 or tuple(lower) not in members:
                    total += lower_weight * float(
                        face_transport[axis][i, j, k]
                    )
                upper_face = [i, j, k]
                upper_face[axis] += 1
                if coordinate == n_axis - 1 or tuple(upper) not in members:
                    total += upper_weight * float(
                        face_transport[axis][tuple(upper_face)]
                    )
            # Eta-local agglomeration never internalizes a parallel eta face.
            total += 0.5 * float(face_transport[2][i, j, k])
            total += 0.5 * float(face_transport[2][i, j, k + 1])
        return total / max(group_volume, 1.0e-300)

    def neighbours(cell: tuple[int, int, int]):
        i, j, k = cell
        if i > 0:
            yield (i - 1, j, k), float(gx[i, j, k])
        if i + 1 < nx:
            yield (i + 1, j, k), float(gx[i + 1, j, k])
        if j > 0:
            yield (i, j - 1, k), float(gy[i, j, k])
        if j + 1 < ny:
            yield (i, j + 1, k), float(gy[i, j + 1, k])

    seeds = sorted(
        (tuple(int(value) for value in row) for row in np.argwhere(seed_mask)),
        key=lambda cell: (-float(rate[cell]), cell),
    )
    for seed in seeds:
        if claimed[seed]:
            continue
        members = {seed}
        claimed[seed] = True
        group_volume = float(raw_volume[seed])
        group_rate = projected_group_rate(members, group_volume)
        while (
            len(members) < 2
            or group_volume < lower
            or group_rate > rate_threshold
        ):
            candidate_edges: dict[tuple[int, int, int], float] = {}
            for member in members:
                for candidate, conductance in neighbours(member):
                    if not claimed[candidate] and candidate not in members:
                        candidate_edges[candidate] = max(
                            conductance,
                            candidate_edges.get(candidate, 0.0),
                        )
            candidates: list[
                tuple[float, float, float, tuple[int, int, int]]
            ] = []
            for candidate, conductance in candidate_edges.items():
                candidate_volume = float(raw_volume[candidate])
                trial_volume = group_volume + candidate_volume
                trial_rate = projected_group_rate(
                    members | {candidate}, trial_volume
                )
                candidates.append(
                    (
                        trial_rate,
                        abs(trial_volume - median_volume),
                        conductance,
                        candidate,
                    )
                )
            # First avoid consuming a neighbor that would leave a badly
            # under/over-sized remainder once the projected row is safe.
            # While it is still rate-limited, minimize that row first.
            candidates.sort(
                key=lambda item: (
                    item[0] if group_rate > rate_threshold else 0.0,
                    item[1],
                    -item[2],
                    item[3],
                )
            )
            selected = None
            for _, _, _, candidate in candidates:
                if group_volume + float(raw_volume[candidate]) <= upper:
                    selected = candidate
                    break
            if selected is None:
                # A CFL seed whose raw volume is already near the lower edge
                # can have no two-cell combination inside the preferred
                # 1.2x band.  It still must be merged to change the projected
                # operator, so take the least-overshooting connected choice
                # under a separate hard guard and report the achieved tail.
                for _, _, _, candidate in candidates:
                    if group_volume + float(raw_volume[candidate]) <= maximum_upper:
                        selected = candidate
                        break
            if selected is None and candidates:
                # A rate-limited cell can itself already exceed the soft
                # maximum-volume band.  It still needs at least one neighbor
                # to alter the projected upwind operator, so take the least
                # volume-overshooting connected choice.
                selected = candidates[0][3]
            if selected is None:
                adjacent_groups: set[tuple[int, int, int]] = set()
                for member in members:
                    for candidate, _ in neighbours(member):
                        if claimed[candidate] and candidate not in members:
                            adjacent_groups.add(tuple(int(v) for v in owner[candidate]))
                compatible_groups = [
                    canonical for canonical in adjacent_groups
                    if canonical in group_volumes
                ]
                if compatible_groups:
                    canonical = min(
                        compatible_groups,
                        key=lambda value: (
                            projected_group_rate(
                                members | group_members[value],
                                group_volumes[value] + group_volume,
                            ),
                            abs(
                                group_volumes[value] + group_volume
                                - median_volume
                            ),
                            value,
                        ),
                    )
                    existing_members = group_members.pop(canonical)
                    existing_volume = group_volumes.pop(canonical)
                    members.update(existing_members)
                    group_volume += existing_volume
                    group_rate = projected_group_rate(members, group_volume)
                    continue
                raise ValueError(
                    "cannot form a connected corner/edge aggregate inside the requested "
                    f"volume guards [{lower:.6e}, {maximum_upper:.6e}] for seed {seed}; "
                    f"current members={len(members)}, volume={group_volume:.6e}"
                )
            members.add(selected)
            claimed[selected] = True
            group_volume += float(raw_volume[selected])
            group_rate = projected_group_rate(members, group_volume)

        # The smallest storage index is a deterministic direct owner.
        canonical = min(members)
        for member in members:
            owner[member] = canonical
        group_members[canonical] = members
        group_volumes[canonical] = group_volume

    raw_centroid = np.stack(np.meshgrid(
        _as_numpy(geometry.grid.x.centers), _as_numpy(geometry.grid.y.centers),
        _as_numpy(geometry.grid.z.centers), indexing="ij",
    ), axis=-1)
    zeros2 = np.zeros(shape + (3, 3), dtype=np.float64)
    zeros3 = np.zeros(shape + (3, 3, 3), dtype=np.float64)
    topology = build_global_control_volume_topology_from_owner_map(
        owner_index=owner,
        positive_mask=np.ones(shape, dtype=bool),
        raw_volume=raw_volume,
        raw_centroid=raw_centroid,
        raw_second_moment=zeros2,
        raw_third_moment=zeros3,
        face_open_measure=_physical_face_measures(geometry),
        periodic_axes=(False, False, True),
    )
    if not np.all(claimed[seed_mask]):  # Defensive: prior groups may cover later seeds.
        raise AssertionError("corner/edge agglomeration left an uncovered CFL seed")
    projected_rate = np.asarray(rate, dtype=np.float64).copy()
    for canonical, members in group_members.items():
        projected_rate[canonical] = projected_group_rate(
            members, group_volumes[canonical]
        )
        for member in members:
            if member != canonical:
                projected_rate[member] = 0.0
    return CornerEdgeAgglomerationGeometry3D(
        topology=topology,
        raw_volume=raw_volume,
        aggregate_chart_volume=np.asarray(topology.aggregate_volume),
        parallel_rate=rate,
        projected_parallel_rate=projected_rate,
        seed_mask=seed_mask,
        target_volume_lower=lower,
        target_volume_upper=upper,
        maximum_volume_upper=maximum_upper,
        max_wall_normal_depth=max_wall_normal_depth,
        max_tangential_corner_reach=max_tangential_corner_reach,
    )


__all__ = ["CornerEdgeAgglomerationGeometry3D", "build_corner_edge_agglomeration"]
