"""Global, decomposition-invariant embedded control-volume topology.

This module is deliberately NumPy based.  It runs while constructing static
geometry, before JAX tracing and before the global mesh is split into local
shards.  Runtime JAX payloads are compiled from these records elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PolarAngularAgglomerationGeometry3D:
    """Host-side owner topology and physical volumes for production RLP.

    The PDE remains defined on the ordinary fine polar grid.  This payload
    therefore stores only the owner map and the volume moments needed by
    restriction/prolongation and owner-space linear algebra.  Face fitting is
    intentionally absent: coarse action is ``R A_f P``.
    """

    topology: "GlobalControlVolumeTopology3D"
    angular_group_size: np.ndarray
    radial_centers: np.ndarray
    radial_widths: np.ndarray
    raw_volume: np.ndarray
    raw_chart_centroid: np.ndarray
    raw_chart_second_moment: np.ndarray
    raw_chart_third_moment: np.ndarray
    aggregate_chart_volume: np.ndarray
    aggregate_chart_centroid: np.ndarray
    aggregate_chart_second_moment: np.ndarray
    aggregate_chart_third_moment: np.ndarray
    theta_period: float
    eta_period: float
    quadrature_order: int

    def __post_init__(self) -> None:
        topology = self.topology
        shape = topology.shape
        q = np.asarray(self.angular_group_size, dtype=np.int32)
        if q.shape != (shape[0],):
            raise ValueError("angular_group_size must have one entry per radial ring")
        if np.any(q < 1):
            raise ValueError("angular group sizes must be positive")
        object.__setattr__(self, "angular_group_size", q)
        for name, suffix in (
            ("radial_centers", ()), ("radial_widths", ()),
            ("raw_volume", ()), ("aggregate_chart_volume", ()),
            ("raw_chart_centroid", (3,)), ("aggregate_chart_centroid", (3,)),
            ("raw_chart_second_moment", (3, 3)),
            ("aggregate_chart_second_moment", (3, 3)),
            ("raw_chart_third_moment", (3, 3, 3)),
            ("aggregate_chart_third_moment", (3, 3, 3)),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            expected = shape + suffix if name not in {"radial_centers", "radial_widths"} else (shape[0],)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if np.any(self.radial_widths <= 0.0):
            raise ValueError("radial widths must be positive")
        if not np.isfinite(self.theta_period) or self.theta_period <= 0.0:
            raise ValueError("theta_period must be positive and finite")
        if not np.isfinite(self.eta_period) or self.eta_period <= 0.0:
            raise ValueError("eta_period must be positive and finite")


def polar_regular_chart(
    logical_points: np.ndarray,
) -> np.ndarray:
    """Map logical ``(u, theta, eta)`` points to ``(x, y, eta_tilde)``.

    The radial/poloidal coordinates are regularized analytically through
    ``x = u*cos(theta)`` and ``y = u*sin(theta)``.  eta is passed through
    unchanged.
    """

    points = np.asarray(logical_points, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 3:
        raise ValueError("logical_points must have shape (..., 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("logical_points must be finite")
    u, theta, eta = np.moveaxis(points, -1, 0)
    return np.stack((u * np.cos(theta), u * np.sin(theta), eta), axis=-1)


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


__all__ = [
    "GlobalControlVolumeTopology3D",
    "PolarAngularAgglomerationGeometry3D",
    "polar_regular_chart",
]
