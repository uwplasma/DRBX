"""Global, decomposition-invariant embedded control-volume topology.

This module is deliberately NumPy based.  It runs while constructing static
geometry, before JAX tracing and before the global mesh is split into local
shards.  Runtime JAX payloads are compiled from these records elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def nearest_periodic_image_delta(
    displacement: np.ndarray,
    *,
    periodic_axes: tuple[bool, bool, bool],
    periods: tuple[float, float, float],
) -> np.ndarray:
    """Return the nearest-image logical displacement for periodic axes."""

    result = np.asarray(displacement, dtype=np.float64).copy()
    for axis, periodic in enumerate(periodic_axes):
        if periodic:
            period = float(periods[axis])
            if not np.isfinite(period) or period <= 0.0:
                raise ValueError("periodic coordinate periods must be positive")
            result[..., axis] -= period * np.round(result[..., axis] / period)
    return result


def combine_volume_moments(
    volume: np.ndarray,
    centroid: np.ndarray,
    second_moment: np.ndarray,
    third_moment: np.ndarray,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
    periods: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Combine central moments of positive-volume control volumes.

    ``second_moment`` and ``third_moment`` are normalized central moments,
    rather than raw integrals.  The returned values use the same convention.
    Periodic centroids are first unwrapped relative to the first member.
    """

    volume = np.asarray(volume, dtype=np.float64).reshape((-1,))
    centroid = np.asarray(centroid, dtype=np.float64).reshape((-1, 3))
    second_moment = np.asarray(second_moment, dtype=np.float64).reshape(
        (-1, 3, 3)
    )
    third_moment = np.asarray(third_moment, dtype=np.float64).reshape(
        (-1, 3, 3, 3)
    )
    if not (
        volume.shape[0]
        == centroid.shape[0]
        == second_moment.shape[0]
        == third_moment.shape[0]
    ):
        raise ValueError("volume and moment members must have matching lengths")
    keep = volume > 0.0
    if not np.any(keep):
        return (
            0.0,
            np.zeros((3,), dtype=np.float64),
            np.zeros((3, 3), dtype=np.float64),
            np.zeros((3, 3, 3), dtype=np.float64),
        )
    volume = volume[keep]
    centroid = centroid[keep]
    second_moment = second_moment[keep]
    third_moment = third_moment[keep]
    reference = centroid[0]
    centroid = reference + nearest_periodic_image_delta(
        centroid - reference,
        periodic_axes=periodic_axes,
        periods=periods,
    )
    total_volume = float(np.sum(volume))
    aggregate_centroid = np.einsum("n,ni->i", volume, centroid) / total_volume
    displacement = centroid - aggregate_centroid
    aggregate_second = np.einsum(
        "n,nij->ij",
        volume,
        second_moment
        + displacement[..., :, None] * displacement[..., None, :],
    ) / total_volume
    translated_third = (
        third_moment
        + displacement[..., :, None, None] * second_moment[..., None, :, :]
        + displacement[..., None, :, None] * second_moment[..., :, None, :]
        + displacement[..., None, None, :] * second_moment[..., :, :, None]
        + displacement[..., :, None, None]
        * displacement[..., None, :, None]
        * displacement[..., None, None, :]
    )
    aggregate_third = np.einsum("n,nijk->ijk", volume, translated_third)
    aggregate_third /= total_volume
    return total_volume, aggregate_centroid, aggregate_second, aggregate_third


@dataclass(frozen=True)
class GlobalControlVolumeTopology3D:
    """Canonical aggregate ownership and unique external face topology."""

    shape: tuple[int, int, int]
    aggregate_id: np.ndarray
    owner_index: np.ndarray
    is_merge_source: np.ndarray
    is_active_owner: np.ndarray
    retained_cut_cell: np.ndarray
    aggregate_volume: np.ndarray
    aggregate_centroid: np.ndarray
    aggregate_second_moment: np.ndarray
    aggregate_third_moment: np.ndarray
    face_id: np.ndarray
    face_axis: np.ndarray
    face_storage_index: np.ndarray
    face_minus_aggregate_id: np.ndarray
    face_plus_aggregate_id: np.ndarray
    face_measure: np.ndarray

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.shape)
        if len(shape) != 3 or any(value <= 0 for value in shape):
            raise ValueError("shape must contain three positive dimensions")
        cell_shape = shape
        arrays = {
            "aggregate_id": (self.aggregate_id, cell_shape, np.int64),
            "owner_index": (self.owner_index, cell_shape + (3,), np.int32),
            "is_merge_source": (self.is_merge_source, cell_shape, bool),
            "is_active_owner": (self.is_active_owner, cell_shape, bool),
            "retained_cut_cell": (self.retained_cut_cell, cell_shape, bool),
            "aggregate_volume": (self.aggregate_volume, cell_shape, np.float64),
            "aggregate_centroid": (
                self.aggregate_centroid,
                cell_shape + (3,),
                np.float64,
            ),
            "aggregate_second_moment": (
                self.aggregate_second_moment,
                cell_shape + (3, 3),
                np.float64,
            ),
            "aggregate_third_moment": (
                self.aggregate_third_moment,
                cell_shape + (3, 3, 3),
                np.float64,
            ),
        }
        for name, (value, expected_shape, dtype) in arrays.items():
            array = np.asarray(value, dtype=dtype)
            if array.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {array.shape}"
                )
            object.__setattr__(self, name, array)
        face_arrays = {
            "face_id": (self.face_id, np.int64),
            "face_axis": (self.face_axis, np.int32),
            "face_storage_index": (self.face_storage_index, np.int32),
            "face_minus_aggregate_id": (
                self.face_minus_aggregate_id,
                np.int64,
            ),
            "face_plus_aggregate_id": (
                self.face_plus_aggregate_id,
                np.int64,
            ),
            "face_measure": (self.face_measure, np.float64),
        }
        count = None
        for name, (value, dtype) in face_arrays.items():
            array = np.asarray(value, dtype=dtype)
            if name == "face_storage_index":
                if array.ndim != 2 or array.shape[1:] != (3,):
                    raise ValueError(
                        "face_storage_index must have shape (face_count, 3)"
                    )
            else:
                array = array.reshape((-1,))
            if count is None:
                count = array.shape[0]
            elif array.shape[0] != count:
                raise ValueError("global face arrays must have matching lengths")
            object.__setattr__(self, name, array)
        if not np.array_equal(
            self.aggregate_id,
            np.ravel_multi_index(
                tuple(np.moveaxis(self.owner_index, -1, 0)), shape
            ),
        ):
            raise ValueError("aggregate_id must equal the canonical owner index")
        if np.any(self.is_merge_source & self.is_active_owner):
            raise ValueError("a merge source cannot be an active owner")
        if np.any(self.aggregate_volume[self.is_active_owner] <= 0.0):
            raise ValueError("active aggregate owners need positive volume")


@dataclass(frozen=True)
class LocalControlVolumeGeometry3D:
    """Shard-local view compiled from ``GlobalControlVolumeTopology3D``.

    The class is intentionally host-side metadata.  The existing JAX payload
    can be compiled from it while the migration remains staged.
    """

    global_shape: tuple[int, int, int]
    shard_index: tuple[int, int, int]
    shard_counts: tuple[int, int, int]
    local_aggregate_id: np.ndarray
    local_owner_index: np.ndarray
    local_active_owner: np.ndarray
    local_face_id: np.ndarray
    local_face_axis: np.ndarray
    local_face_storage_index: np.ndarray
    local_face_minus_aggregate_id: np.ndarray
    local_face_plus_aggregate_id: np.ndarray
    local_face_measure: np.ndarray
    remote_aggregate_id: np.ndarray


def _neighbor_index(
    index: tuple[int, int, int],
    axis: int,
    direction: int,
    shape: tuple[int, int, int],
    periodic_axes: tuple[bool, bool, bool],
) -> tuple[int, int, int] | None:
    result = list(index)
    result[axis] += direction
    if 0 <= result[axis] < shape[axis]:
        return tuple(result)
    if periodic_axes[axis]:
        result[axis] %= shape[axis]
        return tuple(result)
    return None


def _face_measure_at(
    face_open_measure: tuple[np.ndarray, np.ndarray, np.ndarray],
    index: tuple[int, int, int],
    axis: int,
    direction: int,
) -> float:
    face_index = list(index)
    if direction > 0:
        face_index[axis] += 1
    return float(face_open_measure[axis][tuple(face_index)])


def build_global_control_volume_topology(
    *,
    raw_volume: np.ndarray,
    raw_centroid: np.ndarray,
    raw_second_moment: np.ndarray,
    raw_third_moment: np.ndarray,
    fluid_volume_fraction: np.ndarray,
    face_open_measure: tuple[np.ndarray, np.ndarray, np.ndarray],
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
    coordinate_periods: tuple[float, float, float] = (1.0, 1.0, 1.0),
    merge_fraction: float = 0.5,
) -> GlobalControlVolumeTopology3D:
    """Build direct, decomposition-invariant aggregate ownership.

    Every candidate source is selected from the *unmerged* global grid first,
    so source-to-source chains are impossible and the result does not depend
    on shard layout or iteration order.
    """

    raw_volume = np.asarray(raw_volume, dtype=np.float64)
    shape = raw_volume.shape
    if len(shape) != 3:
        raise ValueError("raw_volume must be three dimensional")
    raw_centroid = np.asarray(raw_centroid, dtype=np.float64)
    raw_second_moment = np.asarray(raw_second_moment, dtype=np.float64)
    raw_third_moment = np.asarray(raw_third_moment, dtype=np.float64)
    fraction = np.asarray(fluid_volume_fraction, dtype=np.float64)
    if raw_centroid.shape != shape + (3,):
        raise ValueError("raw_centroid must match raw_volume + (3,)")
    if raw_second_moment.shape != shape + (3, 3):
        raise ValueError("raw_second_moment must match raw_volume + (3, 3)")
    if raw_third_moment.shape != shape + (3, 3, 3):
        raise ValueError("raw_third_moment must match raw_volume + (3, 3, 3)")
    if fraction.shape != shape:
        raise ValueError("fluid_volume_fraction must match raw_volume")
    expected_face_shapes = (
        (shape[0] + 1, shape[1], shape[2]),
        (shape[0], shape[1] + 1, shape[2]),
        (shape[0], shape[1], shape[2] + 1),
    )
    face_open_measure = tuple(
        np.asarray(value, dtype=np.float64) for value in face_open_measure
    )
    if tuple(value.shape for value in face_open_measure) != expected_face_shapes:
        raise ValueError("face_open_measure has incompatible face shapes")
    positive = raw_volume > 0.0
    candidate_source = positive & (fraction < float(merge_fraction))
    owner_index = np.stack(np.indices(shape, dtype=np.int32), axis=-1)
    is_merge_source = np.zeros(shape, dtype=bool)
    retained_cut_cell = candidate_source.copy()
    for source_array in np.argwhere(candidate_source):
        source = tuple(int(value) for value in source_array)
        choices: list[tuple[float, float, tuple[int, int, int]]] = []
        for axis in range(3):
            for direction in (-1, 1):
                target = _neighbor_index(
                    source, axis, direction, shape, periodic_axes
                )
                if target is None or not positive[target] or candidate_source[target]:
                    continue
                measure = _face_measure_at(
                    face_open_measure, source, axis, direction
                )
                if measure <= 0.0:
                    continue
                delta = nearest_periodic_image_delta(
                    raw_centroid[target] - raw_centroid[source],
                    periodic_axes=periodic_axes,
                    periods=coordinate_periods,
                )
                choices.append((-measure, float(np.dot(delta, delta)), target))
        if not choices:
            continue
        _, _, target = min(choices)
        owner_index[source] = target
        is_merge_source[source] = True
        retained_cut_cell[source] = False
    aggregate_id = np.ravel_multi_index(tuple(np.moveaxis(owner_index, -1, 0)), shape)
    is_active_owner = positive & ~is_merge_source
    aggregate_volume = np.zeros(shape, dtype=np.float64)
    aggregate_centroid = np.zeros(shape + (3,), dtype=np.float64)
    aggregate_second = np.zeros(shape + (3, 3), dtype=np.float64)
    aggregate_third = np.zeros(shape + (3, 3, 3), dtype=np.float64)
    for owner_array in np.argwhere(is_active_owner):
        owner = tuple(int(value) for value in owner_array)
        members = np.argwhere(aggregate_id == aggregate_id[owner])
        member_index = tuple(members.T)
        volume, centroid, second, third = combine_volume_moments(
            raw_volume[member_index],
            raw_centroid[member_index],
            raw_second_moment[member_index],
            raw_third_moment[member_index],
            periodic_axes=periodic_axes,
            periods=coordinate_periods,
        )
        aggregate_volume[owner] = volume
        aggregate_centroid[owner] = centroid
        aggregate_second[owner] = second
        aggregate_third[owner] = third
    face_id: list[int] = []
    face_axis: list[int] = []
    face_storage_index: list[tuple[int, int, int]] = []
    face_minus: list[int] = []
    face_plus: list[int] = []
    face_measure: list[float] = []
    next_face_id = 0
    for axis, measures in enumerate(face_open_measure):
        for face_array in np.argwhere(measures > 0.0):
            face = tuple(int(value) for value in face_array)
            normal_index = face[axis]
            # A periodic seam is one interior interface.  Represent it at the
            # low logical face and skip the duplicate high-face image.
            if periodic_axes[axis] and normal_index == shape[axis]:
                continue
            if normal_index == 0:
                plus_storage = list(face)
                plus_storage[axis] = 0
                plus = int(aggregate_id[tuple(plus_storage)])
                if periodic_axes[axis]:
                    minus_storage = list(face)
                    minus_storage[axis] = shape[axis] - 1
                    minus = int(aggregate_id[tuple(minus_storage)])
                else:
                    minus = -1
            elif normal_index == shape[axis]:
                minus_storage = list(face)
                minus_storage[axis] -= 1
                minus = int(aggregate_id[tuple(minus_storage)])
                plus = -1
            else:
                minus_storage = list(face)
                minus_storage[axis] -= 1
                plus_storage = list(face)
                minus = int(aggregate_id[tuple(minus_storage)])
                plus = int(aggregate_id[tuple(plus_storage)])
            if minus >= 0 and plus >= 0 and minus == plus:
                continue
            if minus >= 0 and not positive[np.unravel_index(minus, shape)]:
                continue
            if plus >= 0 and not positive[np.unravel_index(plus, shape)]:
                continue
            face_id.append(next_face_id)
            face_axis.append(axis)
            face_storage_index.append(face)
            face_minus.append(minus)
            face_plus.append(plus)
            face_measure.append(float(measures[face]))
            next_face_id += 1
    return GlobalControlVolumeTopology3D(
        shape=shape,
        aggregate_id=aggregate_id,
        owner_index=owner_index,
        is_merge_source=is_merge_source,
        is_active_owner=is_active_owner,
        retained_cut_cell=retained_cut_cell,
        aggregate_volume=aggregate_volume,
        aggregate_centroid=aggregate_centroid,
        aggregate_second_moment=aggregate_second,
        aggregate_third_moment=aggregate_third,
        face_id=np.asarray(face_id, dtype=np.int64),
        face_axis=np.asarray(face_axis, dtype=np.int32),
        face_storage_index=np.asarray(face_storage_index, dtype=np.int32).reshape(
            (-1, 3)
        ),
        face_minus_aggregate_id=np.asarray(face_minus, dtype=np.int64),
        face_plus_aggregate_id=np.asarray(face_plus, dtype=np.int64),
        face_measure=np.asarray(face_measure, dtype=np.float64),
    )


def compile_local_control_volume_geometry(
    topology: GlobalControlVolumeTopology3D,
    *,
    shard_index: tuple[int, int, int],
    shard_counts: tuple[int, int, int],
) -> LocalControlVolumeGeometry3D:
    """Compile one host-side shard view from global aggregate topology."""

    shard_counts = tuple(int(value) for value in shard_counts)
    if any(value <= 0 for value in shard_counts):
        raise ValueError("shard_counts must be positive")
    if any(
        topology.shape[axis] % shard_counts[axis] for axis in range(3)
    ):
        raise ValueError("global topology must divide evenly across shards")
    owned_shape = tuple(
        topology.shape[axis] // shard_counts[axis] for axis in range(3)
    )
    start = tuple(
        int(shard_index[axis]) * owned_shape[axis] for axis in range(3)
    )
    slices = tuple(
        slice(start[axis], start[axis] + owned_shape[axis])
        for axis in range(3)
    )
    local_ids = topology.aggregate_id[slices]
    local_owner = topology.owner_index[slices]
    local_active = topology.is_active_owner[slices]
    # A source owned by another shard carries that remote aggregate ID in its
    # local map.  It must not cause the remote owner to be classified as
    # local: locality belongs to the physical active-owner cell, not to an ID
    # referenced by local storage.
    local_owner_ids = local_ids[local_active]
    local_id_set = set(int(value) for value in np.unique(local_owner_ids))
    local_face_mask = np.isin(topology.face_minus_aggregate_id, list(local_id_set)) | np.isin(
        topology.face_plus_aggregate_id,
        list(local_id_set),
    )
    local_faces = topology.face_id[local_face_mask]
    local_face_axis = topology.face_axis[local_face_mask]
    local_face_storage_index = topology.face_storage_index[local_face_mask]
    local_face_minus = topology.face_minus_aggregate_id[local_face_mask]
    local_face_plus = topology.face_plus_aggregate_id[local_face_mask]
    local_face_measure = topology.face_measure[local_face_mask]
    face_references = np.concatenate(
        (
            topology.face_minus_aggregate_id[local_face_mask],
            topology.face_plus_aggregate_id[local_face_mask],
        )
    )
    remote_ids = np.unique(
        np.concatenate((face_references, local_ids.reshape((-1,))))
    )
    remote_ids = remote_ids[(remote_ids >= 0) & ~np.isin(remote_ids, list(local_id_set))]
    return LocalControlVolumeGeometry3D(
        global_shape=topology.shape,
        shard_index=tuple(int(value) for value in shard_index),
        shard_counts=shard_counts,
        local_aggregate_id=local_ids,
        local_owner_index=local_owner,
        local_active_owner=local_active,
        local_face_id=local_faces,
        local_face_axis=local_face_axis,
        local_face_storage_index=local_face_storage_index,
        local_face_minus_aggregate_id=local_face_minus,
        local_face_plus_aggregate_id=local_face_plus,
        local_face_measure=local_face_measure,
        remote_aggregate_id=remote_ids,
    )


__all__ = [
    "GlobalControlVolumeTopology3D",
    "LocalControlVolumeGeometry3D",
    "build_global_control_volume_topology",
    "combine_volume_moments",
    "compile_local_control_volume_geometry",
    "nearest_periodic_image_delta",
]
