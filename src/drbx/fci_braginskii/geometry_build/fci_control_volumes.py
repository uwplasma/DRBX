"""Global, decomposition-invariant embedded control-volume topology.

This module is deliberately NumPy based.  It runs while constructing static
geometry, before JAX tracing and before the global mesh is split into local
shards.  Runtime JAX payloads are compiled from these records elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

# Definitions identical to (and closure-equivalent with) the shared module;
# re-exported here so existing import paths keep working.
from ..geometry.fci_control_volumes import (  # noqa: F401
    GlobalControlVolumeTopology3D,
)

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
    raw_radial_centroid: np.ndarray
    raw_radial_second_moment: np.ndarray
    raw_radial_third_moment: np.ndarray
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
            ("raw_radial_centroid", ()),
            ("raw_radial_second_moment", ()),
            ("raw_radial_third_moment", ()),
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
    return_logical_radial_moments: bool = False,
) -> tuple[np.ndarray, ...]:
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
        When ``return_logical_radial_moments=True``, the tuple additionally
        contains the J-weighted logical-radial centroid, second central
        moment, and third central moment.  These are distinct from the first
        component of the regular Cartesian chart.
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
    if not return_logical_radial_moments:
        return volume, centroid, second, third
    logical_radial = logical_flat[:, 0].reshape((cell_count, nq))
    radial_centroid_flat = (
        np.sum(weighted * logical_radial, axis=1) / volume_flat
    )
    radial_displacement = logical_radial - radial_centroid_flat[:, None]
    radial_second_flat = (
        np.sum(weighted * radial_displacement**2, axis=1) / volume_flat
    )
    radial_third_flat = (
        np.sum(weighted * radial_displacement**3, axis=1) / volume_flat
    )
    return (
        volume,
        centroid,
        second,
        third,
        radial_centroid_flat.reshape(shape),
        radial_second_flat.reshape(shape),
        radial_third_flat.reshape(shape),
    )


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
    (
        raw_volume,
        raw_centroid,
        raw_second,
        raw_third,
        raw_radial_centroid,
        raw_radial_second,
        raw_radial_third,
    ) = integrate_polar_regular_chart_cell_moments(
        u_faces,
        theta_faces,
        eta_faces,
        jacobian,
        quadrature_order=order,
        jacobian_chunk_size=jacobian_chunk_size,
        eta_unwrap_origin=float(eta_faces[0]),
        eta_period=eta_period,
        return_logical_radial_moments=True,
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
        raw_radial_centroid=raw_radial_centroid,
        raw_radial_second_moment=raw_radial_second,
        raw_radial_third_moment=raw_radial_third,
        aggregate_chart_volume=topology.aggregate_volume,
        aggregate_chart_centroid=topology.aggregate_centroid,
        aggregate_chart_second_moment=topology.aggregate_second_moment,
        aggregate_chart_third_moment=topology.aggregate_third_moment,
        theta_period=theta_period,
        eta_period=eta_period,
        quadrature_order=order,
    )

__all__ = [
    "GlobalControlVolumeTopology3D",
    "PolarAngularAgglomerationGeometry3D",
    "build_global_control_volume_topology_from_owner_map",
    "build_nested_angular_group_profile",
    "build_radius_dependent_angular_group_profile",
    "build_radius_dependent_angular_owner_map",
    "build_polar_angular_agglomeration_geometry",
    "combine_volume_moments",
    "combine_volume_moments_by_aggregate",
    "integrate_polar_regular_chart_cell_moments",
    "nearest_periodic_image_delta",
    "polar_regular_chart",
    "polar_regular_chart_jacobian",
]
