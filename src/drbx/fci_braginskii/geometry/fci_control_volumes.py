"""Global, decomposition-invariant embedded control-volume topology.

This module is deliberately NumPy based.  It runs while constructing static
geometry, before JAX tracing and before the global mesh is split into local
shards.  Runtime JAX payloads are compiled from these records elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

def _quadratic_chart_exponents() -> tuple[tuple[int, int, int], ...]:
    """Deterministic total-degree-two monomials in ``(x, y, eta)``."""

    return tuple(
        (a, b, c)
        for total_degree in range(3)
        for a in range(total_degree + 1)
        for b in range(total_degree - a + 1)
        for c in (total_degree - a - b,)
    )


_POLAR_QUADRATIC_EXPONENTS = _quadratic_chart_exponents()

def control_volume_average_basis_numpy(
    centroid: np.ndarray,
    second_moment: np.ndarray,
    third_moment: np.ndarray,
    *,
    origin: np.ndarray,
    scale: np.ndarray | float,
    exponents: tuple[tuple[int, int, int], ...] = _POLAR_QUADRATIC_EXPONENTS,
) -> np.ndarray:
    """Evaluate exact monomial cell averages from central moments.

    This is the geometry-layer NumPy counterpart of the native finite-volume
    average basis.  Keeping the implementation here avoids a geometry-to-native
    dependency while letting host stencil selection condition the exact matrix
    that native lowering later solves.
    """

    centroid = np.asarray(centroid, dtype=np.float64)
    second = np.asarray(second_moment, dtype=np.float64)
    third = np.asarray(third_moment, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    selected = tuple(tuple(int(value) for value in exponent) for exponent in exponents)
    if centroid.shape[-1:] != (3,) or second.shape[-2:] != (3, 3) or third.shape[-3:] != (3, 3, 3):
        raise ValueError("centroid and central moments must have 3D trailing shapes")
    if centroid.shape[:-1] != second.shape[:-2] or centroid.shape[:-1] != third.shape[:-3]:
        raise ValueError("control-volume moment batch shapes must match")
    if origin.shape != (3,):
        raise ValueError("origin must have shape (3,)")
    if scale.ndim == 0:
        scale = np.full(3, float(scale), dtype=np.float64)
    if scale.shape != (3,) or np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("scale must be one positive scalar or three positive values")
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("exponents must be a nonempty unique sequence")
    if any(len(power) != 3 or any(value < 0 for value in power) or sum(power) > 3 for power in selected):
        raise ValueError("central moments support three nonnegative powers through degree three")

    displacement = centroid - origin
    raw_second = second + displacement[..., :, None] * displacement[..., None, :]
    raw_third = (
        third
        + displacement[..., :, None, None] * second[..., None, :, :]
        + displacement[..., None, :, None] * second[..., :, None, :]
        + displacement[..., None, None, :] * second[..., :, :, None]
        + displacement[..., :, None, None]
        * displacement[..., None, :, None]
        * displacement[..., None, None, :]
    )
    result = np.empty(centroid.shape[:-1] + (len(selected),), dtype=np.float64)
    for column, power in enumerate(selected):
        degree = sum(power)
        if degree == 0:
            value = np.ones(centroid.shape[:-1], dtype=np.float64)
        elif degree == 1:
            axis = int(np.flatnonzero(power)[0])
            value = displacement[..., axis]
        elif degree == 2:
            axes = np.repeat(np.arange(3), np.asarray(power, dtype=np.int32))
            value = raw_second[..., axes[0], axes[1]]
        else:
            axes = np.repeat(np.arange(3), np.asarray(power, dtype=np.int32))
            value = raw_third[..., axes[0], axes[1], axes[2]]
        result[..., column] = value / np.prod(
            scale ** np.asarray(power, dtype=np.float64)
        )
    return result


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
    *,
    eta_unwrap_origin: float | None = None,
    eta_period: float | None = None,
) -> np.ndarray:
    """Map logical ``(u, theta, eta)`` points to ``(x, y, eta_tilde)``.

    The radial/poloidal coordinates are regularized analytically through
    ``x = u*cos(theta)`` and ``y = u*sin(theta)``.  If ``eta_period`` is
    supplied, eta is represented in the branch centered at
    ``eta_unwrap_origin``.  The latter is useful for a stencil crossing a
    periodic eta seam; omitting it preserves the input eta values exactly.
    """

    points = np.asarray(logical_points, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 3:
        raise ValueError("logical_points must have shape (..., 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("logical_points must be finite")
    u, theta, eta = np.moveaxis(points, -1, 0)
    if eta_period is not None:
        period = float(eta_period)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("eta_period must be finite and positive")
        origin = 0.0 if eta_unwrap_origin is None else float(eta_unwrap_origin)
        if not np.isfinite(origin):
            raise ValueError("eta_unwrap_origin must be finite")
        eta = origin + (eta - origin + 0.5 * period) % period - 0.5 * period
    elif eta_unwrap_origin is not None:
        raise ValueError("eta_period is required when eta_unwrap_origin is set")
    return np.stack((u * np.cos(theta), u * np.sin(theta), eta), axis=-1)


def polar_regular_chart_jacobian(logical_points: np.ndarray) -> np.ndarray:
    """Return ``dchi/dxi`` for ``chi=(u*cos(theta),u*sin(theta),eta)``.

    The returned array has shape ``(..., 3, 3)``.  Its rows are chart
    components and its columns are logical ``(u, theta, eta)`` components.
    This Jacobian is analytic and remains finite at ``u=0``.
    """

    points = np.asarray(logical_points, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 3:
        raise ValueError("logical_points must have shape (..., 3)")
    u = points[..., 0]
    theta = points[..., 1]
    result = np.zeros(points.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 0] = np.cos(theta)
    result[..., 0, 1] = -u * np.sin(theta)
    result[..., 1, 0] = np.sin(theta)
    result[..., 1, 1] = u * np.cos(theta)
    result[..., 2, 2] = 1.0
    return result


def _validate_logical_faces(faces: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(faces, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be a one-dimensional face array")
    if not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return values


def integrate_polar_regular_chart_cell_moments(
    u_faces: np.ndarray,
    theta_faces: np.ndarray,
    eta_faces: np.ndarray,
    jacobian,
    *,
    quadrature_order: int = 3,
    eta_unwrap_origin: float | None = None,
    eta_period: float | None = None,
    jacobian_chunk_size: int = 32768,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate J-weighted regular-chart cell moments on a logical grid.

    Parameters
    ----------
    u_faces, theta_faces, eta_faces:
        One-dimensional logical face arrays.  The output shape is
        ``(len(u_faces)-1, len(theta_faces)-1, len(eta_faces)-1)``.
    jacobian:
        Vectorized callable accepting an array of shape ``(..., 3)`` and
        returning J with shape ``(...)`` (or ``(..., 1)``).
    quadrature_order:
        Tensor Gauss-Legendre order in each logical direction. Three points
        exactly integrates cubic polynomial moments when J is constant.

    Returns
    -------
    volume, centroid, second_moment, third_moment:
        J-weighted volume and normalized central moments with shapes matching
        ``GlobalControlVolumeTopology3D`` raw moment arrays.
    """

    u_faces = _validate_logical_faces(u_faces, "u_faces")
    theta_faces = _validate_logical_faces(theta_faces, "theta_faces")
    eta_faces = _validate_logical_faces(eta_faces, "eta_faces")
    order = int(quadrature_order)
    if order < 1:
        raise ValueError("quadrature_order must be positive")
    chunk_size = int(jacobian_chunk_size)
    if chunk_size < 1:
        raise ValueError("jacobian_chunk_size must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    shape = (u_faces.size - 1, theta_faces.size - 1, eta_faces.size - 1)
    lower = np.stack(np.meshgrid(u_faces[:-1], theta_faces[:-1], eta_faces[:-1], indexing="ij"), axis=-1)
    upper = np.stack(np.meshgrid(u_faces[1:], theta_faces[1:], eta_faces[1:], indexing="ij"), axis=-1)
    centers = 0.5 * (lower + upper)
    half_widths = 0.5 * (upper - lower)
    q_offsets = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape((-1, 3))
    logical = centers[..., None, :] + half_widths[..., None, :] * q_offsets[None, None, None, :, :]
    cell_count = int(np.prod(shape))
    nq = q_offsets.shape[0]
    logical_flat = logical.reshape((cell_count * nq, 3))
    j_flat = np.empty(cell_count * nq, dtype=np.float64)
    for start in range(0, logical_flat.shape[0], chunk_size):
        stop = min(start + chunk_size, logical_flat.shape[0])
        values = np.asarray(jacobian(logical_flat[start:stop]), dtype=np.float64)
        if values.shape == (stop - start, 1):
            values = values[:, 0]
        if values.shape != (stop - start,):
            raise ValueError("jacobian must return shape (...) matching logical points")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("jacobian must be finite and strictly positive")
        j_flat[start:stop] = values
    j_values = j_flat.reshape((cell_count, nq))
    chart = polar_regular_chart(
        logical.reshape(shape + (nq, 3)),
        eta_unwrap_origin=eta_unwrap_origin,
        eta_period=eta_period,
    ).reshape((cell_count, nq, 3))
    cell_scales = np.prod(half_widths, axis=-1).reshape(-1)
    reference_weights = np.einsum("i,j,k->ijk", weights, weights, weights).reshape(-1)
    weighted = j_values * cell_scales[:, None] * reference_weights[None, :]
    volume_flat = np.sum(weighted, axis=1)
    if np.any(~np.isfinite(volume_flat)) or np.any(volume_flat <= 0.0):
        raise ValueError("cell has non-positive J-weighted volume")
    centroid_flat = np.sum(weighted[..., None] * chart, axis=1) / volume_flat[:, None]
    displacement = chart - centroid_flat[:, None, :]
    second_flat = np.einsum("nq,nqa,nqb->nab", weighted, displacement, displacement) / volume_flat[:, None, None]
    third_flat = np.einsum("nq,nqa,nqb,nqc->nabc", weighted, displacement, displacement, displacement) / volume_flat[:, None, None, None]
    volume = volume_flat.reshape(shape)
    centroid = centroid_flat.reshape(shape + (3,))
    second = second_flat.reshape(shape + (3, 3))
    third = third_flat.reshape(shape + (3, 3, 3))
    return volume, centroid, second, third


def combine_volume_moments_by_aggregate(
    aggregate_id: np.ndarray,
    raw_volume: np.ndarray,
    raw_centroid: np.ndarray,
    raw_second_moment: np.ndarray,
    raw_third_moment: np.ndarray,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
    periods: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine all aggregate moments with one vectorized scatter pass."""
    ids = np.asarray(aggregate_id, dtype=np.int64).reshape(-1)
    volume = np.asarray(raw_volume, dtype=np.float64).reshape(-1)
    centroid = np.asarray(raw_centroid, dtype=np.float64).reshape((-1, 3))
    second = np.asarray(raw_second_moment, dtype=np.float64).reshape((-1, 3, 3))
    third = np.asarray(raw_third_moment, dtype=np.float64).reshape((-1, 3, 3, 3))
    if not (volume.size == ids.size == centroid.shape[0] == second.shape[0] == third.shape[0]):
        raise ValueError("aggregate moment arrays have incompatible sizes")
    unique, inverse = np.unique(ids, return_inverse=True)
    first = np.full(unique.size, ids.size, dtype=np.int64)
    np.minimum.at(first, inverse, np.arange(ids.size, dtype=np.int64))
    reference = centroid[first]
    displacement = centroid - reference[inverse]
    for axis, periodic in enumerate(periodic_axes):
        if periodic:
            period = float(periods[axis])
            if not np.isfinite(period) or period <= 0.0:
                raise ValueError("periods must be finite and positive")
            displacement[:, axis] = (displacement[:, axis] + 0.5 * period) % period - 0.5 * period
    total_volume = np.bincount(inverse, weights=volume, minlength=unique.size)
    weighted_delta = volume[:, None] * displacement
    delta_sum = np.zeros((unique.size, 3), dtype=np.float64)
    np.add.at(delta_sum, inverse, weighted_delta)
    result_centroid = reference + delta_sum / total_volume[:, None]
    centered_delta = displacement - delta_sum[inverse] / total_volume[inverse, None]
    second_terms = second + np.einsum("na,nb->nab", centered_delta, centered_delta)
    second_sum = np.zeros((unique.size, 3, 3), dtype=np.float64)
    np.add.at(second_sum, inverse, volume[:, None, None] * second_terms)
    result_second = second_sum / total_volume[:, None, None]
    d = centered_delta
    third_terms = third + np.einsum("nab,nc->nabc", second, d) + np.einsum("nac,nb->nabc", second, d) + np.einsum("nbc,na->nabc", second, d) + np.einsum("na,nb,nc->nabc", d, d, d)
    third_sum = np.zeros((unique.size, 3, 3, 3), dtype=np.float64)
    np.add.at(third_sum, inverse, volume[:, None, None, None] * third_terms)
    result_third = third_sum / total_volume[:, None, None, None]
    return unique, total_volume, result_centroid, result_second, result_third


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
            raw_delta = int(delta[axis])
            if counts[axis] == 2 and abs(raw_delta) == 1:
                extent = int(shape[axis])
                owner_coordinate = int(owner_local[axis])
                if extent <= 1:
                    raise ValueError(
                        "periodic two-shard owner direction is ambiguous when "
                        f"the owned extent is {extent}; extent-one axes have "
                        "the same lower and upper boundary-local coordinate"
                    )
                if owner_coordinate == 0:
                    delta[axis] = 1
                elif owner_coordinate == extent - 1:
                    delta[axis] = -1
                else:
                    raise ValueError(
                        "periodic two-shard remote owner direction is ambiguous: "
                        f"owner_local[{axis}]={owner_coordinate} is not a boundary "
                        f"coordinate (expected 0 or {extent - 1})"
                    )
            elif raw_delta == counts[axis] - 1:
                delta[axis] = -1
            elif raw_delta == -(counts[axis] - 1):
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


def build_global_control_volume_topology_from_owner_map(
    *,
    owner_index: np.ndarray,
    positive_mask: np.ndarray,
    raw_volume: np.ndarray,
    raw_centroid: np.ndarray,
    raw_second_moment: np.ndarray,
    raw_third_moment: np.ndarray,
    face_open_measure: tuple[np.ndarray, np.ndarray, np.ndarray],
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
    coordinate_periods: tuple[float, float, float] = (1.0, 1.0, 1.0),
    retained_cut_cell: np.ndarray | None = None,
) -> GlobalControlVolumeTopology3D:
    """Build a global topology from a prescribed direct owner map.

    ``owner_index`` is a map from every storage cell to the canonical storage
    cell that owns its aggregate.  It is intentionally required to be direct:
    an owner must own itself, so a source may not point to another source.
    This constructor is global-only; sharding is a later compilation step.
    """

    raw_volume = np.asarray(raw_volume, dtype=np.float64)
    shape = raw_volume.shape
    if len(shape) != 3:
        raise ValueError("raw_volume must be three dimensional")
    positive = np.asarray(positive_mask, dtype=bool)
    if positive.shape != shape:
        raise ValueError("positive_mask must match raw_volume")
    owner_index = np.asarray(owner_index)
    if owner_index.shape != shape + (3,):
        raise ValueError("owner_index must match raw_volume + (3,)")
    if not np.all(np.isfinite(owner_index)) or not np.array_equal(
        owner_index, owner_index.astype(np.int32)
    ):
        raise ValueError("owner_index must contain integer indices")
    owner_index = owner_index.astype(np.int32, copy=False)
    if np.any(owner_index < 0) or np.any(owner_index >= np.asarray(shape)):
        raise ValueError("owner_index contains an out-of-range owner")
    raw_centroid = np.asarray(raw_centroid, dtype=np.float64)
    raw_second_moment = np.asarray(raw_second_moment, dtype=np.float64)
    raw_third_moment = np.asarray(raw_third_moment, dtype=np.float64)
    if raw_centroid.shape != shape + (3,):
        raise ValueError("raw_centroid must match raw_volume + (3,)")
    if raw_second_moment.shape != shape + (3, 3):
        raise ValueError("raw_second_moment must match raw_volume + (3, 3)")
    if raw_third_moment.shape != shape + (3, 3, 3):
        raise ValueError("raw_third_moment must match raw_volume + (3, 3, 3)")
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
    if any(np.any(~np.isfinite(value)) or np.any(value < 0.0) for value in face_open_measure):
        raise ValueError("face_open_measure must be finite and nonnegative")
    if np.any(~np.isfinite(raw_volume)) or np.any(raw_volume < 0.0):
        raise ValueError("raw_volume must be finite and nonnegative")

    # Direct/idempotent ownership is the key invariant: following an owner
    # pointer once must already reach the canonical active owner.
    owner_at_owner = owner_index[tuple(np.moveaxis(owner_index, -1, 0))]
    if not np.array_equal(owner_at_owner, owner_index):
        raise ValueError("owner_index must be direct and idempotent; chains are not allowed")
    owner_positive = positive[tuple(np.moveaxis(owner_index, -1, 0))]
    if np.any(positive & ~owner_positive):
        raise ValueError("positive cells must map to positive owners")
    self_index = np.stack(np.indices(shape, dtype=np.int32), axis=-1)
    if np.any(~positive & np.any(owner_index != self_index, axis=-1)):
        raise ValueError("nonpositive storage cells must own themselves")
    is_merge_source = positive & np.any(owner_index != self_index, axis=-1)
    is_active_owner = positive & ~is_merge_source
    if not np.any(is_active_owner):
        raise ValueError("owner map must contain at least one positive owner")
    if retained_cut_cell is None:
        retained_cut_cell = positive & ~is_merge_source
    else:
        retained_cut_cell = np.asarray(retained_cut_cell, dtype=bool)
        if retained_cut_cell.shape != shape:
            raise ValueError("retained_cut_cell must match raw_volume")
        if np.any(is_merge_source & retained_cut_cell):
            raise ValueError("merged sources cannot be retained cut cells")

    aggregate_id = np.ravel_multi_index(tuple(np.moveaxis(owner_index, -1, 0)), shape)
    aggregate_volume = np.zeros(shape, dtype=np.float64)
    aggregate_centroid = np.zeros(shape + (3,), dtype=np.float64)
    aggregate_second = np.zeros(shape + (3, 3), dtype=np.float64)
    aggregate_third = np.zeros(shape + (3, 3, 3), dtype=np.float64)
    aggregate_volume[is_active_owner] = raw_volume[is_active_owner]
    aggregate_centroid[is_active_owner] = raw_centroid[is_active_owner]
    aggregate_second[is_active_owner] = raw_second_moment[is_active_owner]
    aggregate_third[is_active_owner] = raw_third_moment[is_active_owner]
    source_flat = np.flatnonzero(is_merge_source)
    source_owner_id = aggregate_id.reshape((-1,))[source_flat]
    if source_flat.size:
        source_order = np.argsort(source_owner_id, kind="stable")
        source_flat = source_flat[source_order]
        source_owner_id = source_owner_id[source_order]
        owner_ids, group_start = np.unique(source_owner_id, return_index=True)
        group_stop = np.concatenate(
            (group_start[1:], np.asarray((source_flat.size,), dtype=np.int64))
        )
    else:
        owner_ids = np.empty((0,), dtype=np.int64)
        group_start = np.empty((0,), dtype=np.int64)
        group_stop = np.empty((0,), dtype=np.int64)
    for owner_id, first, stop in zip(owner_ids, group_start, group_stop):
        member_flat = np.sort(
            np.concatenate((np.asarray((owner_id,), dtype=np.int64), source_flat[first:stop]))
        )
        member_index = np.unravel_index(member_flat, shape)
        volume, centroid, second, third = combine_volume_moments(
            raw_volume[member_index], raw_centroid[member_index],
            raw_second_moment[member_index], raw_third_moment[member_index],
            periodic_axes=periodic_axes, periods=coordinate_periods,
        )
        owner = np.unravel_index(int(owner_id), shape)
        aggregate_volume[owner] = volume
        aggregate_centroid[owner] = centroid
        aggregate_second[owner] = second
        aggregate_third[owner] = third

    positive_volume = float(np.sum(raw_volume[positive]))
    aggregate_volume_total = float(np.sum(aggregate_volume[is_active_owner]))
    if not np.isclose(
        aggregate_volume_total,
        positive_volume,
        rtol=1.0e-12,
        atol=1.0e-14 * max(1.0, positive_volume),
    ):
        raise ValueError(
            "aggregate volumes do not conserve the positive raw volume: "
            f"{aggregate_volume_total} != {positive_volume}"
        )

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
        shape=shape, aggregate_id=aggregate_id, owner_index=owner_index,
        is_merge_source=is_merge_source, is_active_owner=is_active_owner,
        retained_cut_cell=retained_cut_cell, aggregate_volume=aggregate_volume,
        aggregate_centroid=aggregate_centroid,
        aggregate_second_moment=aggregate_second,
        aggregate_third_moment=aggregate_third,
        face_id=np.asarray(face_id, dtype=np.int64),
        face_axis=np.asarray(face_axis, dtype=np.int32),
        face_storage_index=np.asarray(face_storage_index, dtype=np.int32).reshape((-1, 3)),
        face_minus_aggregate_id=np.asarray(face_minus, dtype=np.int64),
        face_plus_aggregate_id=np.asarray(face_plus, dtype=np.int64),
        face_measure=np.asarray(face_measure, dtype=np.float64),
    )


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
    return build_global_control_volume_topology_from_owner_map(
        owner_index=owner_index,
        positive_mask=positive,
        raw_volume=raw_volume,
        raw_centroid=raw_centroid,
        raw_second_moment=raw_second_moment,
        raw_third_moment=raw_third_moment,
        face_open_measure=face_open_measure,
        periodic_axes=periodic_axes,
        coordinate_periods=coordinate_periods,
        retained_cut_cell=retained_cut_cell,
    )
def build_nested_angular_group_profile(
    theta_cell_count: int,
    minimum_group_size: np.ndarray | tuple[float, ...],
) -> np.ndarray:
    """Return the least-coarsened nested divisor profile above ``minimum``.

    Every returned group size divides ``theta_cell_count`` and every outer
    ring group divides the adjacent inner-ring group.  The first ring is
    always collapsed to one angular owner.
    """

    ntheta = int(theta_cell_count)
    minimum = np.asarray(minimum_group_size, dtype=np.float64)
    if ntheta < 1:
        raise ValueError("theta_cell_count must be positive")
    if minimum.ndim != 1 or minimum.size < 1:
        raise ValueError("minimum_group_size must be a nonempty one-dimensional array")
    if np.any(~np.isfinite(minimum)) or np.any(minimum <= 0.0):
        raise ValueError("minimum_group_size must be positive and finite")
    if np.any(minimum > float(ntheta) + 1.0e-14):
        raise ValueError("minimum_group_size cannot exceed theta_cell_count")
    divisors = np.asarray(
        [q for q in range(1, ntheta + 1) if ntheta % q == 0],
        dtype=np.int32,
    )
    required = np.empty(minimum.size, dtype=np.int32)
    for ring, threshold in enumerate(minimum):
        admissible = divisors[divisors.astype(np.float64) >= threshold - 1.0e-14]
        if admissible.size == 0:
            raise ValueError(f"no angular group size satisfies ring {ring}")
        required[ring] = int(admissible[0])

    q = np.empty(required.size, dtype=np.int32)
    q[-1] = required[-1]
    for ring in range(required.size - 2, 0, -1):
        admissible = divisors[
            (divisors >= required[ring]) & (divisors % q[ring + 1] == 0)
        ]
        if admissible.size == 0:
            raise ValueError(f"no nested angular group size satisfies ring {ring}")
        q[ring] = int(admissible[0])
    q[0] = ntheta
    if (
        np.any(q.astype(np.float64) < minimum - 1.0e-14)
        or np.any(q[1:] > q[:-1])
        or np.any(q[:-1] % q[1:] != 0)
    ):
        raise ValueError("could not construct a nested angular profile")
    return q


def build_radius_dependent_angular_group_profile(
    u_faces: np.ndarray,
    theta_faces: np.ndarray,
    *,
    explicit_profile: np.ndarray | tuple[int, ...] | None = None,
    theta_period: float = 2.0 * np.pi,
) -> np.ndarray:
    """Return a nested divisor-based angular agglomeration profile.

    For ring ``i > 0``, the smallest admissible divisor ``q`` of ``Ntheta``
    satisfying ``q*r_i*dtheta >= dr_i`` is selected.  The profile is then
    made nested, so every outer group divides the adjacent inner group.  Ring
    zero is always represented by one owner, hence ``q[0] == Ntheta``.  An
    explicit profile is useful for deterministic geometry tests and is
    validated against the same nesting and divisor invariants (the geometric
    inequality is intentionally not imposed on explicit test profiles).
    """

    u_faces = _validate_logical_faces(u_faces, "u_faces")
    theta_faces = _validate_logical_faces(theta_faces, "theta_faces")
    theta_period = float(theta_period)
    if not np.isfinite(theta_period) or theta_period <= 0.0:
        raise ValueError("theta_period must be positive and finite")
    if not np.isclose(theta_faces[-1] - theta_faces[0], theta_period, rtol=1e-12, atol=1e-12):
        raise ValueError("theta_faces must cover exactly one theta period")
    ntheta = theta_faces.size - 1
    nr = u_faces.size - 1
    if ntheta < 1:
        raise ValueError("Ntheta must be positive")
    if explicit_profile is not None:
        q = np.asarray(explicit_profile, dtype=np.int64)
        if q.shape != (nr,):
            raise ValueError("explicit angular profile must have one entry per radial ring")
        if np.any(q < 1) or np.any(q > ntheta) or np.any(ntheta % q != 0):
            raise ValueError("explicit angular profile must contain divisors of Ntheta")
        if int(q[0]) != ntheta or np.any(q[1:] > q[:-1]):
            raise ValueError("explicit angular profile must start at Ntheta and be non-increasing")
        if np.any(q[:-1] % q[1:] != 0):
            raise ValueError(
                "explicit angular profile must be nested: every outer group "
                "must divide the adjacent inner group"
            )
        if not np.all(np.isfinite(q)) or not np.array_equal(q, q.astype(np.int32)):
            raise ValueError("explicit angular profile must contain integer values")
        return q.astype(np.int32)
    dr = np.diff(u_faces)
    radius = 0.5 * (u_faces[:-1] + u_faces[1:])
    dtheta = theta_period / float(ntheta)
    minimum = dr / np.maximum(radius * dtheta, 1.0e-300)
    minimum[0] = float(ntheta)
    return build_nested_angular_group_profile(ntheta, minimum)


def build_radius_dependent_angular_owner_map(
    radial_ring_count: int,
    theta_cell_count: int,
    eta_cell_count: int,
    angular_group_size: np.ndarray | tuple[int, ...],
) -> np.ndarray:
    """Build the direct, idempotent owner map for nested theta groups."""

    nr, ntheta, neta = int(radial_ring_count), int(theta_cell_count), int(eta_cell_count)
    q = np.asarray(angular_group_size, dtype=np.int64)
    if q.shape != (nr,) or np.any(q < 1) or np.any(ntheta % q != 0):
        raise ValueError("angular_group_size must contain divisors of Ntheta")
    if q[0] != ntheta or np.any(q[1:] > q[:-1]):
        raise ValueError("angular groups must start at Ntheta and be nested outward")
    owner = np.stack(np.indices((nr, ntheta, neta), dtype=np.int32), axis=-1)
    theta = np.arange(ntheta, dtype=np.int32)
    for ring in range(nr):
        owner[ring, :, :, 1] = (theta // int(q[ring]) * int(q[ring]))[:, None]
    return owner


def build_polar_angular_agglomeration_geometry(
    u_faces: np.ndarray,
    theta_faces: np.ndarray,
    eta_faces: np.ndarray,
    jacobian,
    *,
    quadrature_order: int = 3,
    theta_period: float = 2.0 * np.pi,
    eta_period: float | None = None,
    jacobian_chunk_size: int = 32768,
    angular_group_size: np.ndarray | tuple[int, ...] | None = None,
) -> PolarAngularAgglomerationGeometry3D:
    """Build Phase 1--2 radius-dependent angular agglomeration geometry.

    The topology is generated from a single global owner map shared by every
    eta plane.  Compact rows retain each affected physical fine subface; no
    row is emitted for an internal same-owner face, and ordinary bulk faces
    with ``q == 1`` remain outside this payload.
    """

    u_faces = _validate_logical_faces(u_faces, "u_faces")
    theta_faces = _validate_logical_faces(theta_faces, "theta_faces")
    eta_faces = _validate_logical_faces(eta_faces, "eta_faces")
    theta_period = float(theta_period)
    eta_period = float(eta_faces[-1] - eta_faces[0]) if eta_period is None else float(eta_period)
    if not np.isfinite(eta_period) or eta_period <= 0.0:
        raise ValueError("eta_period must be positive and finite")
    if not np.isclose(eta_faces[-1] - eta_faces[0], eta_period, rtol=1e-12, atol=1e-12):
        raise ValueError("eta_faces must cover exactly one eta period")
    order = int(quadrature_order)
    if order < 1:
        raise ValueError("quadrature_order must be positive")
    raw_volume, raw_centroid, raw_second, raw_third = integrate_polar_regular_chart_cell_moments(
        u_faces, theta_faces, eta_faces, jacobian,
        quadrature_order=order, jacobian_chunk_size=jacobian_chunk_size,
        eta_unwrap_origin=float(eta_faces[0]), eta_period=eta_period,
    )
    shape = raw_volume.shape
    q = build_radius_dependent_angular_group_profile(
        u_faces, theta_faces, explicit_profile=angular_group_size,
        theta_period=theta_period,
    )
    owner_index = build_radius_dependent_angular_owner_map(*shape, q)
    face_open = (
        np.ones((shape[0] + 1, shape[1], shape[2]), dtype=np.float64),
        np.ones((shape[0], shape[1] + 1, shape[2]), dtype=np.float64),
        np.ones((shape[0], shape[1], shape[2] + 1), dtype=np.float64),
    )
    face_open[0][0, :, :] = 0.0
    topology = build_global_control_volume_topology_from_owner_map(
        owner_index=owner_index,
        positive_mask=np.ones(shape, dtype=bool),
        raw_volume=raw_volume, raw_centroid=raw_centroid,
        raw_second_moment=raw_second, raw_third_moment=raw_third,
        face_open_measure=face_open,
        periodic_axes=(False, True, True),
        coordinate_periods=(1.0, theta_period, eta_period),
    )
    # The owner-map constructor also serves logical-coordinate callers.  Its
    # topology connectivity is periodic in theta, but these moments are in
    # the Cartesian chart, so recompute them with only eta unwrapped.
    aggregate_ids, aggregate_volume, aggregate_centroid, aggregate_second, aggregate_third = combine_volume_moments_by_aggregate(
        topology.aggregate_id, raw_volume, raw_centroid, raw_second, raw_third,
        periodic_axes=(False, False, True), periods=(1.0, 1.0, eta_period),
    )
    aggregate_lookup = np.searchsorted(aggregate_ids, np.flatnonzero(topology.is_active_owner))
    owner_ids = np.flatnonzero(topology.is_active_owner)
    if np.any(aggregate_lookup >= aggregate_ids.size) or np.any(aggregate_ids[aggregate_lookup] != owner_ids):
        raise ValueError("angular aggregate moments are missing an active owner")
    aggregate_volume_grid = np.zeros(shape, dtype=np.float64)
    aggregate_centroid_grid = np.zeros(shape + (3,), dtype=np.float64)
    aggregate_second_grid = np.zeros(shape + (3, 3), dtype=np.float64)
    aggregate_third_grid = np.zeros(shape + (3, 3, 3), dtype=np.float64)
    aggregate_volume_grid.reshape(-1)[owner_ids] = aggregate_volume[aggregate_lookup]
    aggregate_centroid_grid.reshape((-1, 3))[owner_ids] = aggregate_centroid[aggregate_lookup]
    aggregate_second_grid.reshape((-1, 3, 3))[owner_ids] = aggregate_second[aggregate_lookup]
    aggregate_third_grid.reshape((-1, 3, 3, 3))[owner_ids] = aggregate_third[aggregate_lookup]
    topology = replace(
        topology, aggregate_volume=aggregate_volume_grid,
        aggregate_centroid=aggregate_centroid_grid,
        aggregate_second_moment=aggregate_second_grid,
        aggregate_third_moment=aggregate_third_grid,
    )

    return PolarAngularAgglomerationGeometry3D(
        topology=topology,
        angular_group_size=q,
        radial_centers=0.5 * (u_faces[:-1] + u_faces[1:]),
        radial_widths=np.diff(u_faces),
        raw_volume=raw_volume,
        raw_chart_centroid=raw_centroid,
        raw_chart_second_moment=raw_second,
        raw_chart_third_moment=raw_third,
        aggregate_chart_volume=topology.aggregate_volume,
        aggregate_chart_centroid=topology.aggregate_centroid,
        aggregate_chart_second_moment=topology.aggregate_second_moment,
        aggregate_chart_third_moment=topology.aggregate_third_moment,
        theta_period=theta_period,
        eta_period=eta_period,
        quadrature_order=order,
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
    "PolarAngularAgglomerationGeometry3D",
    "build_global_control_volume_topology",
    "build_global_control_volume_topology_from_owner_map",
    "build_nested_angular_group_profile",
    "build_radius_dependent_angular_group_profile",
    "build_radius_dependent_angular_owner_map",
    "build_polar_angular_agglomeration_geometry",
    "combine_volume_moments",
    "combine_volume_moments_by_aggregate",
    "compile_local_control_volume_geometry",
    "control_volume_average_basis_numpy",
    "integrate_polar_regular_chart_cell_moments",
    "nearest_periodic_image_delta",
    "polar_regular_chart",
    "polar_regular_chart_jacobian",
]
