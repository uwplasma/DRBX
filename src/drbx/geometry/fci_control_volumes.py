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
    local_merge_source: np.ndarray
    owner_shard_index: np.ndarray
    owner_local_index: np.ndarray
    owner_is_remote: np.ndarray
    local_raw_volume: np.ndarray
    local_raw_centroid: np.ndarray
    local_raw_second_moment: np.ndarray
    local_raw_third_moment: np.ndarray
    local_aggregate_volume: np.ndarray
    local_aggregate_centroid: np.ndarray
    local_aggregate_second_moment: np.ndarray
    local_aggregate_third_moment: np.ndarray
    local_received_source_count: np.ndarray
    local_member_count: np.ndarray
    # ``local_face_*`` are evaluator rows, not merely faces visible to this
    # shard.  Their union over a decomposition is exactly the global face set.
    local_face_id: np.ndarray
    local_face_axis: np.ndarray
    local_face_storage_index: np.ndarray
    local_face_minus_aggregate_id: np.ndarray
    local_face_plus_aggregate_id: np.ndarray
    local_face_measure: np.ndarray
    local_face_evaluator_aggregate_id: np.ndarray
    local_face_evaluator_shard_index: np.ndarray
    local_face_evaluator_owner_index: np.ndarray
    local_face_evaluator_local_index: np.ndarray
    local_face_remote_target_aggregate_id: np.ndarray
    # Visibility is retained separately for host-side diagnostics/lowering.
    visible_face_id: np.ndarray
    remote_aggregate_id: np.ndarray

    def __post_init__(self) -> None:
        if any(self.global_shape[a] % self.shard_counts[a] for a in range(3)):
            raise ValueError("global_shape must divide evenly across shard_counts")
        shape = tuple(self.global_shape[a] // self.shard_counts[a] for a in range(3))
        checks = {
            "local_aggregate_id": (self.local_aggregate_id, shape),
            "local_owner_index": (self.local_owner_index, shape + (3,)),
            "local_active_owner": (self.local_active_owner, shape),
            "local_merge_source": (self.local_merge_source, shape),
            "owner_shard_index": (self.owner_shard_index, shape + (3,)),
            "owner_local_index": (self.owner_local_index, shape + (3,)),
            "owner_is_remote": (self.owner_is_remote, shape),
            "local_raw_volume": (self.local_raw_volume, shape),
            "local_raw_centroid": (self.local_raw_centroid, shape + (3,)),
            "local_raw_second_moment": (self.local_raw_second_moment, shape + (3, 3)),
            "local_raw_third_moment": (self.local_raw_third_moment, shape + (3, 3, 3)),
            "local_aggregate_volume": (self.local_aggregate_volume, shape),
            "local_aggregate_centroid": (self.local_aggregate_centroid, shape + (3,)),
            "local_aggregate_second_moment": (self.local_aggregate_second_moment, shape + (3, 3)),
            "local_aggregate_third_moment": (self.local_aggregate_third_moment, shape + (3, 3, 3)),
            "local_received_source_count": (self.local_received_source_count, shape),
            "local_member_count": (self.local_member_count, shape),
        }
        for name, (value, expected) in checks.items():
            value = np.asarray(value)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
            object.__setattr__(self, name, value)
        if np.any(np.asarray(self.local_active_owner) & np.asarray(self.owner_is_remote)):
            raise ValueError("active owners cannot be remote")
        face_count = np.asarray(self.local_face_id).size
        face_checks = {
            "local_face_axis": (self.local_face_axis, (face_count,)),
            "local_face_storage_index": (self.local_face_storage_index, (face_count, 3)),
            "local_face_minus_aggregate_id": (self.local_face_minus_aggregate_id, (face_count,)),
            "local_face_plus_aggregate_id": (self.local_face_plus_aggregate_id, (face_count,)),
            "local_face_measure": (self.local_face_measure, (face_count,)),
            "local_face_evaluator_aggregate_id": (self.local_face_evaluator_aggregate_id, (face_count,)),
            "local_face_evaluator_shard_index": (self.local_face_evaluator_shard_index, (face_count, 3)),
            "local_face_evaluator_owner_index": (self.local_face_evaluator_owner_index, (face_count, 3)),
            "local_face_evaluator_local_index": (self.local_face_evaluator_local_index, (face_count, 3)),
            "local_face_remote_target_aggregate_id": (self.local_face_remote_target_aggregate_id, (face_count,)),
        }
        for name, (value, expected) in face_checks.items():
            array = np.asarray(value)
            if array.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
            object.__setattr__(self, name, array)
        local_face_id = np.asarray(self.local_face_id, dtype=np.int64)
        if np.unique(local_face_id).size != face_count:
            raise ValueError("evaluator local_face_id values must be unique")
        if np.any(np.asarray(self.local_face_evaluator_shard_index) != np.asarray(self.shard_index)):
            raise ValueError("evaluator rows must be owned by their evaluator shard")
        if np.any(np.asarray(self.local_face_evaluator_aggregate_id) < 0):
            raise ValueError("each evaluator row needs a physical aggregate owner")
        visible = np.asarray(self.visible_face_id, dtype=np.int64)
        if np.unique(visible).size != visible.size:
            raise ValueError("visible_face_id values must be unique")
        object.__setattr__(self, "local_face_id", local_face_id)
        object.__setattr__(self, "visible_face_id", visible)


def remote_owner_halo_coordinate(
    *,
    owner_local: np.ndarray,
    owner_shard: np.ndarray,
    local_shard: np.ndarray,
    owned_shape: tuple[int, int, int],
    halo_width: int,
    shard_counts: tuple[int, int, int],
    periodic_axes: tuple[bool, bool, bool],
) -> np.ndarray:
    """Return the one-face halo address of a directly adjacent remote owner.

    Global agglomeration only permits one source to merge across one physical
    face.  A remote target must therefore lie in exactly one adjacent shard;
    periodic shard-index wrap is normalized before this contract is checked.
    """
    owner_local = np.asarray(owner_local, dtype=np.int32)
    owner_shard = np.asarray(owner_shard, dtype=np.int32)
    local_shard = np.asarray(local_shard, dtype=np.int32)
    shape = np.asarray(owned_shape, dtype=np.int32)
    counts = np.asarray(shard_counts, dtype=np.int32)
    if owner_local.shape != (3,) or owner_shard.shape != (3,) or local_shard.shape != (3,):
        raise ValueError("remote owner and shard indices must have shape (3,)")
    if np.any(owner_local < 0) or np.any(owner_local >= shape):
        raise ValueError("remote owner local index is out of range")
    delta = owner_shard - local_shard
    for axis in range(3):
        if periodic_axes[axis] and counts[axis] > 1:
            if delta[axis] == counts[axis] - 1:
                delta[axis] = -1
            elif delta[axis] == -(counts[axis] - 1):
                delta[axis] = 1
    nonzero = np.flatnonzero(delta)
    if nonzero.size != 1 or abs(int(delta[nonzero[0]])) != 1:
        raise ValueError(
            "a remote aggregate owner must be in exactly one directly adjacent shard"
        )
    axis = int(nonzero[0])
    coord = int(halo_width) + owner_local.copy()
    if delta[axis] < 0:
        coord[axis] = int(halo_width) - int(shape[axis]) + int(owner_local[axis])
    else:
        coord[axis] = int(halo_width) + int(shape[axis]) + int(owner_local[axis])
    halo_shape = shape + 2 * int(halo_width)
    if np.any(coord < 0) or np.any(coord >= halo_shape):
        raise ValueError("remote aggregate owner must land in a face halo slab")
    expected = int(halo_width) - 1 if delta[axis] < 0 else int(halo_width) + int(shape[axis])
    if int(coord[axis]) != expected:
        raise ValueError("remote aggregate owner is not on the adjacent shard face")
    return coord


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
    positive_volume_floor: float = 0.0,
    positive_mask: np.ndarray | None = None,
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
    if positive_volume_floor < 0.0:
        raise ValueError("positive_volume_floor must be nonnegative")
    positive = raw_volume > float(positive_volume_floor)
    if positive_mask is not None:
        positive_mask = np.asarray(positive_mask, dtype=bool)
        if positive_mask.shape != shape:
            raise ValueError("positive_mask must match raw_volume")
        positive &= positive_mask
    candidate_source = positive & (fraction < float(merge_fraction))
    owner_index = np.stack(np.indices(shape, dtype=np.int32), axis=-1)
    is_merge_source = np.zeros(shape, dtype=bool)
    retained_cut_cell = candidate_source.copy()
    for source_array in np.argwhere(candidate_source):
        source = tuple(int(value) for value in source_array)
        choices: list[tuple[float, float, int, tuple[int, int, int]]] = []
        for direction_ordinal, (axis, direction) in enumerate(
            ((0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1))
        ):
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
                choices.append((-measure, float(np.linalg.norm(delta)), direction_ordinal, target))
        if not choices:
            continue
        _, _, _, target = min(choices)
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
    raw_volume: np.ndarray | None = None,
    raw_centroid: np.ndarray | None = None,
    raw_second_moment: np.ndarray | None = None,
    raw_third_moment: np.ndarray | None = None,
) -> LocalControlVolumeGeometry3D:
    """Compile one host-side shard view from global aggregate topology."""

    shard_counts = tuple(int(value) for value in shard_counts)
    shard_index = tuple(int(value) for value in shard_index)
    if any(value <= 0 for value in shard_counts):
        raise ValueError("shard_counts must be positive")
    if len(shard_index) != 3 or any(value < 0 or value >= shard_counts[axis] for axis, value in enumerate(shard_index)):
        raise ValueError("shard_index must be in range for shard_counts")
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
    local_source = topology.is_merge_source[slices]
    owner_shard = local_owner // np.asarray(owned_shape, dtype=np.int32)
    owner_local = local_owner % np.asarray(owned_shape, dtype=np.int32)
    owner_remote = np.any(owner_shard != np.asarray(shard_index, dtype=np.int32), axis=-1)
    if np.any(local_active & owner_remote):
        raise ValueError("a physical active owner must be local to its shard")
    local_owner_active = local_active[tuple(np.moveaxis(owner_local, -1, 0))]
    if np.any(local_source & ~owner_remote & ~local_owner_active):
        raise ValueError("a local merge source must target a local active owner")

    def local_raw(value: np.ndarray | None, suffix: tuple[int, ...]) -> np.ndarray:
        if value is None:
            return np.zeros(owned_shape + suffix, dtype=np.float64)
        value = np.asarray(value, dtype=np.float64)
        if value.shape != topology.shape + suffix:
            raise ValueError("raw moment input has incompatible global shape")
        return value[slices]

    member_counts = np.bincount(topology.aggregate_id.reshape((-1,)), minlength=int(np.prod(topology.shape)))
    local_member_count = np.where(local_active, member_counts[local_ids], 0).astype(np.int32)
    local_received_count = np.where(local_active, member_counts[local_ids] - 1, 0).astype(np.int32)
    # A source owned by another shard carries that remote aggregate ID in its
    # local map.  It must not cause the remote owner to be classified as
    # local: locality belongs to the physical active-owner cell, not to an ID
    # referenced by local storage.
    local_owner_ids = local_ids[local_active]
    local_id_set = set(int(value) for value in np.unique(local_owner_ids))
    visible_face_mask = np.isin(topology.face_minus_aggregate_id, list(local_id_set)) | np.isin(
        topology.face_plus_aggregate_id,
        list(local_id_set),
    )
    # Canonical orientation is global minus -> plus.  Physical low boundaries
    # have no minus aggregate, hence the plus aggregate evaluates that row.
    evaluator_id = np.where(
        topology.face_minus_aggregate_id >= 0,
        topology.face_minus_aggregate_id,
        topology.face_plus_aggregate_id,
    )
    evaluator_owner = np.stack(np.unravel_index(evaluator_id, topology.shape), axis=-1)
    evaluator_shard = evaluator_owner // np.asarray(owned_shape, dtype=np.int32)
    evaluator_mask = np.all(evaluator_shard == np.asarray(shard_index, dtype=np.int32), axis=-1)
    local_faces = topology.face_id[evaluator_mask]
    local_face_axis = topology.face_axis[evaluator_mask]
    local_face_storage_index = topology.face_storage_index[evaluator_mask]
    local_face_minus = topology.face_minus_aggregate_id[evaluator_mask]
    local_face_plus = topology.face_plus_aggregate_id[evaluator_mask]
    local_face_measure = topology.face_measure[evaluator_mask]
    local_evaluator_id = evaluator_id[evaluator_mask]
    local_evaluator_owner = evaluator_owner[evaluator_mask]
    local_evaluator_shard = evaluator_shard[evaluator_mask]
    local_evaluator_local = local_evaluator_owner % np.asarray(owned_shape, dtype=np.int32)
    plus_owner = np.where(
        topology.face_plus_aggregate_id[:, None] >= 0,
        np.stack(np.unravel_index(np.maximum(topology.face_plus_aggregate_id, 0), topology.shape), axis=-1),
        -1,
    )
    plus_shard = plus_owner // np.asarray(owned_shape, dtype=np.int32)
    remote_target = np.where(
        (topology.face_plus_aggregate_id >= 0)
        & np.any(plus_shard != evaluator_shard, axis=-1),
        topology.face_plus_aggregate_id,
        -1,
    )[evaluator_mask]
    face_references = np.concatenate(
        (
            topology.face_minus_aggregate_id[visible_face_mask],
            topology.face_plus_aggregate_id[visible_face_mask],
        )
    )
    remote_ids = np.unique(
        np.concatenate((face_references, local_ids.reshape((-1,))))
    )
    remote_ids = remote_ids[(remote_ids >= 0) & ~np.isin(remote_ids, list(local_id_set))]
    return LocalControlVolumeGeometry3D(
        global_shape=topology.shape,
        shard_index=shard_index,
        shard_counts=shard_counts,
        local_aggregate_id=local_ids,
        local_owner_index=local_owner,
        local_active_owner=local_active,
        local_merge_source=local_source,
        owner_shard_index=owner_shard,
        owner_local_index=owner_local,
        owner_is_remote=owner_remote,
        local_raw_volume=local_raw(raw_volume, ()),
        local_raw_centroid=local_raw(raw_centroid, (3,)),
        local_raw_second_moment=local_raw(raw_second_moment, (3, 3)),
        local_raw_third_moment=local_raw(raw_third_moment, (3, 3, 3)),
        local_aggregate_volume=topology.aggregate_volume[slices],
        local_aggregate_centroid=topology.aggregate_centroid[slices],
        local_aggregate_second_moment=topology.aggregate_second_moment[slices],
        local_aggregate_third_moment=topology.aggregate_third_moment[slices],
        local_received_source_count=local_received_count,
        local_member_count=local_member_count,
        local_face_id=local_faces,
        local_face_axis=local_face_axis,
        local_face_storage_index=local_face_storage_index,
        local_face_minus_aggregate_id=local_face_minus,
        local_face_plus_aggregate_id=local_face_plus,
        local_face_measure=local_face_measure,
        local_face_evaluator_aggregate_id=local_evaluator_id,
        local_face_evaluator_shard_index=local_evaluator_shard,
        local_face_evaluator_owner_index=local_evaluator_owner,
        local_face_evaluator_local_index=local_evaluator_local,
        local_face_remote_target_aggregate_id=remote_target,
        visible_face_id=topology.face_id[visible_face_mask],
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
