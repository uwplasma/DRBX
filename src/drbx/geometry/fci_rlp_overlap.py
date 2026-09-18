"""Host-side owner-overlap geometry for the experimental RLP FCI operator.

This module intentionally has no dependency on the native FCI operators.  It
turns raw polar cell faces into a small, canonical graph of pairwise links.  A
link is a geometric overlap between two *unique* control-volume owners on
adjacent eta planes; runtime parallel diffusivity is applied by the consumer
as ``chi_parallel * transmissibility``.

The builder accepts a light-weight ``raw_faces`` representation for analytic
tests and can also derive rectangular raw faces from the existing
``FciGeometry3D``/``PolarAngularAgglomerationGeometry3D`` objects.  A tracer
may be supplied by applications; in its absence the identity trace is used,
which is useful for the straight-field reference case.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping

import numpy as np

# Version 2 records the bounded, second-order geometry policy.  A cache made
# with the former over-resolved edge representation must not be reused as if
# it had the current construction identity.
RLP_PARALLEL_OVERLAP_CACHE_VERSION = 2

__all__ = [
    "RLP_PARALLEL_OVERLAP_CACHE_VERSION",
    "GlobalRlpParallelOverlapGeometry",
    "build_rlp_parallel_overlap_geometry",
    "merge_rlp_parallel_overlap_geometries",
    "compare_rlp_parallel_overlap_refinement",
    "load_rlp_parallel_overlap_geometry",
    "write_rlp_parallel_overlap_geometry",
]


def _array(value: Any, dtype: Any = float) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _polygon_area(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=float)
    if p.shape[0] < 3:
        return 0.0
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)))


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """Scalar cross product for 2-vectors (avoids NumPy 2-D deprecation)."""

    return float(a[0] * b[1] - a[1] * b[0])


def _as_ccw(poly: np.ndarray) -> np.ndarray:
    p = np.asarray(poly, dtype=float).reshape((-1, 2))
    return p if _polygon_area(p) >= 0.0 else p[::-1]


def _simplify_polygon(poly: np.ndarray, eps: float = 1.0e-14) -> np.ndarray:
    """Remove repeated and numerically collinear vertices from a polygon."""

    points = np.asarray(poly, dtype=float).reshape((-1, 2))
    if points.shape[0] == 0:
        return points
    keep = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - keep[-1]) > eps:
            keep.append(point)
    if len(keep) > 1 and np.linalg.norm(keep[0] - keep[-1]) <= eps:
        keep.pop()
    points = np.asarray(keep, dtype=float)
    changed = True
    while changed and points.shape[0] > 3:
        changed = False
        retained = []
        for index in range(points.shape[0]):
            previous = points[index - 1]
            current = points[index]
            following = points[(index + 1) % points.shape[0]]
            scale = max(
                np.linalg.norm(current - previous)
                * np.linalg.norm(following - current),
                1.0,
            )
            if abs(_cross2(current - previous, following - current)) <= eps * scale:
                changed = True
            else:
                retained.append(current)
        if retained:
            points = np.asarray(retained, dtype=float)
        else:
            break
    return points


def _is_convex_polygon(poly: np.ndarray, eps: float = 1.0e-13) -> bool:
    points = _as_ccw(_simplify_polygon(poly))
    if points.shape[0] < 3:
        return False
    crosses = np.asarray(
        [
            _cross2(
                points[(index + 1) % points.shape[0]] - points[index],
                points[(index + 2) % points.shape[0]]
                - points[(index + 1) % points.shape[0]],
            )
            for index in range(points.shape[0])
        ]
    )
    return bool(np.all(crosses >= -eps))


def _point_in_triangle(point: np.ndarray, triangle: np.ndarray, eps: float) -> bool:
    a, b, c = triangle
    return (
        _cross2(b - a, point - a) >= -eps
        and _cross2(c - b, point - b) >= -eps
        and _cross2(a - c, point - c) >= -eps
    )


def _triangulate_simple_polygon(poly: np.ndarray) -> list[np.ndarray]:
    """Ear-clip a finite simple polygon without a geometry dependency."""

    points = _as_ccw(_simplify_polygon(poly))
    polygon_area = abs(_polygon_area(points))
    if points.shape[0] < 3 or polygon_area <= 1.0e-18:
        return []
    # Sutherland--Hodgman already accepts any convex clip polygon.  Keeping a
    # convex mapped face intact avoids turning every ordinary quadrilateral
    # into four centroid-fan pieces (and up to sixteen pairwise clip calls).
    if _is_convex_polygon(points):
        return [points]
    centroid = np.mean(points, axis=0)
    centroid_fan = [
        np.asarray((centroid, points[index], points[(index + 1) % points.shape[0]]))
        for index in range(points.shape[0])
    ]
    centroid_fan = [
        triangle
        for triangle in centroid_fan
        if abs(_polygon_area(triangle)) > 1.0e-18
    ]
    fan_area = sum(abs(_polygon_area(triangle)) for triangle in centroid_fan)
    if abs(fan_area - polygon_area) <= 1.0e-11 * max(polygon_area, 1.0e-30):
        return centroid_fan
    if points.shape[0] > 256:
        raise ValueError(
            "mapped owner face is non-star-shaped and too refined for bounded ear clipping"
        )
    remaining = list(range(points.shape[0]))
    triangles: list[np.ndarray] = []
    eps = 1.0e-14 * max(1.0, float(np.max(np.abs(points))) ** 2)
    while len(remaining) > 3:
        found = False
        for local_index, current in enumerate(remaining):
            previous = remaining[local_index - 1]
            following = remaining[(local_index + 1) % len(remaining)]
            triangle = points[[previous, current, following]]
            if _cross2(triangle[1] - triangle[0], triangle[2] - triangle[1]) <= eps:
                continue
            if any(
                _point_in_triangle(points[candidate], triangle, eps)
                for candidate in remaining
                if candidate not in {previous, current, following}
            ):
                continue
            triangles.append(triangle)
            del remaining[local_index]
            found = True
            break
        if not found:
            raise ValueError("mapped owner face is invalid or self-intersecting")
    triangles.append(points[remaining])
    return triangles


def _proper_segment_intersection(
    first_a: np.ndarray,
    first_b: np.ndarray,
    second_a: np.ndarray,
    second_b: np.ndarray,
    eps: float = 1.0e-13,
) -> np.ndarray | None:
    """Return an interior crossing of two segments, if one exists."""

    first_direction = first_b - first_a
    second_direction = second_b - second_a
    denominator = _cross2(first_direction, second_direction)
    if abs(denominator) <= eps:
        return None
    delta = second_a - first_a
    first_parameter = _cross2(delta, second_direction) / denominator
    second_parameter = _cross2(delta, first_direction) / denominator
    if eps < first_parameter < 1.0 - eps and eps < second_parameter < 1.0 - eps:
        return first_a + first_parameter * first_direction
    return None


def _decompose_mapped_boundary(poly: np.ndarray) -> list[np.ndarray]:
    """Split a self-crossing mapped boundary into simple lobes.

    The owner boundary can fold in the regular chart even though its source
    boundary is simple.  Repeatedly split the first proper crossing into the
    two planar lobes; each lobe is then handled by the existing bounded
    triangulator.  Touching-only or otherwise malformed topology is rejected.
    """

    pending = [_simplify_polygon(np.asarray(poly, dtype=float))]
    simple: list[np.ndarray] = []
    while pending:
        current = _simplify_polygon(pending.pop())
        if current.shape[0] < 3:
            continue
        crossing = None
        count = current.shape[0]
        for first in range(count):
            first_next = (first + 1) % count
            for second in range(first + 1, count):
                second_next = (second + 1) % count
                if first_next == second or second_next == first:
                    continue
                point = _proper_segment_intersection(
                    current[first], current[first_next], current[second], current[second_next]
                )
                if point is not None:
                    crossing = (first, second, point)
                    break
            if crossing is not None:
                break
        if crossing is None:
            if abs(_polygon_area(current)) <= 1.0e-18:
                continue
            simple.append(current)
            continue
        first, second, point = crossing
        lobe_a = np.vstack((point, current[first + 1 : second + 1]))
        lobe_b = np.vstack((point, current[second + 1 :], current[: first + 1]))
        if lobe_a.shape[0] < 3 or lobe_b.shape[0] < 3:
            raise ValueError("mapped owner boundary has an invalid crossing lobe")
        pending.extend((lobe_a, lobe_b))
        if len(pending) > 1024:
            raise ValueError("mapped owner boundary has too many crossing lobes")
    return simple


def _clip_convex(subject: np.ndarray, clip: np.ndarray, eps: float = 1.0e-14) -> np.ndarray:
    """Sutherland--Hodgman clipping of one convex polygon by another."""

    output = _as_ccw(subject)
    clip = _as_ccw(clip)
    if output.shape[0] < 3 or clip.shape[0] < 3:
        return np.empty((0, 2), dtype=float)

    def inside(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
        return _cross2(b - a, point - a) >= -eps

    def intersection(p: np.ndarray, q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = q - p
        e = b - a
        denominator = _cross2(d, e)
        if abs(denominator) < 1.0e-30:
            return q.copy()
        t = _cross2(a - p, e) / denominator
        return p + t * d

    for index in range(clip.shape[0]):
        edge_a = clip[index]
        edge_b = clip[(index + 1) % clip.shape[0]]
        if output.shape[0] == 0:
            break
        input_poly = output
        output_list: list[np.ndarray] = []
        previous = input_poly[-1]
        previous_inside = inside(previous, edge_a, edge_b)
        for current in input_poly:
            current_inside = inside(current, edge_a, edge_b)
            if current_inside != previous_inside:
                output_list.append(intersection(previous, current, edge_a, edge_b))
            if current_inside:
                output_list.append(current)
            previous = current
            previous_inside = current_inside
        output = np.asarray(output_list, dtype=float).reshape((-1, 2)) if output_list else np.empty((0, 2))
    return output


def _regular_xy(points: np.ndarray) -> np.ndarray:
    """Convert logical ``(u, theta[, eta])`` points to the regular chart."""

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("trace points must have shape (n, 2) or (n, 3)")
    return np.column_stack((points[:, 0] * np.cos(points[:, 1]), points[:, 0] * np.sin(points[:, 1])))


def _logical_from_xy(points: np.ndarray, eta: float) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    u = np.hypot(p[:, 0], p[:, 1])
    theta = np.arctan2(p[:, 1], p[:, 0])
    return np.column_stack((u, theta, np.full(u.shape, float(eta))))


def _call_flexible(callback: Callable[..., Any], points: np.ndarray, direction: str, eta: int, eta_step: float | None = None, extra: Mapping[str, Any] | None = None) -> Any:
    """Call supported variants of the planned point-trace callback API."""

    extra = dict(extra or {})
    attempts = (
        *((lambda: callback(points, interface=eta, **extra), lambda: callback(points, eta, **extra)) if direction == "metric" else ()),
        lambda: callback(points, eta_step=eta_step, direction=direction, eta_index=eta, **extra),
        lambda: callback(points, eta_step=eta_step, direction=direction, **extra),
        lambda: callback(points, eta_step=eta_step, direction=direction, eta_index=eta),
        lambda: callback(points, eta_step=eta_step, direction=direction),
        lambda: callback(points, direction=direction, eta_index=eta),
        lambda: callback(points, direction, eta),
        lambda: callback(points, direction=direction, target_eta=eta),
        lambda: callback(points, eta, direction),
        lambda: callback(points, direction),
        lambda: callback(points),
    )
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except (TypeError, ValueError) as error:
            last_error = error
    raise TypeError("trace callback does not match a supported point-trace signature") from last_error


def _trace_result(result: Any, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundary = np.zeros(count, dtype=bool)
    length = np.ones(count, dtype=float)
    endpoint = result
    if isinstance(result, Mapping):
        endpoint = result.get("endpoint", result.get("points", result.get("coordinates")))
        if endpoint is None:
            raise ValueError("trace result mapping needs endpoint/points/coordinates")
        boundary = np.asarray(result.get("boundary", result.get("wall", False)), dtype=bool).reshape(-1)
        length = np.asarray(result.get("length", result.get("arc_length", 1.0)), dtype=float).reshape(-1)
    elif isinstance(result, (tuple, list)):
        if len(result) >= 1:
            endpoint = result[0]
        if len(result) >= 2:
            length = np.asarray(result[1], dtype=float).reshape(-1)
        if len(result) >= 3:
            boundary = np.asarray(result[2], dtype=bool).reshape(-1)
    endpoint = np.asarray(endpoint, dtype=float)
    if endpoint.ndim == 1:
        endpoint = endpoint[None, :]
    if endpoint.shape[0] != count or endpoint.shape[1] not in (2, 3):
        raise ValueError("trace endpoint must have shape (n, 2) or (n, 3)")
    if boundary.size == 1:
        boundary = np.full(count, bool(boundary[0]), dtype=bool)
    if length.size == 1:
        length = np.full(count, float(length[0]), dtype=float)
    if boundary.size != count or length.size != count:
        raise ValueError("trace boundary and length arrays must match point count")
    if not np.all(np.isfinite(endpoint)) or not np.all(np.isfinite(length)):
        raise ValueError("trace endpoints and lengths must be finite")
    return endpoint, length, boundary


@dataclass(frozen=True)
class GlobalRlpParallelOverlapGeometry:
    """Canonical owner graph generated by :func:`build_rlp_parallel_overlap_geometry`.

    ``owner_flat_ids`` are storage-flat ids of active owners.  Link endpoint
    fields are compact slots into that array, so repeated raw aliases can
    never create duplicate state entries.
    """

    raw_shape: tuple[int, int, int]
    owner_flat_ids: np.ndarray
    owner_volumes: np.ndarray
    link_owner_a: np.ndarray
    link_owner_b: np.ndarray
    link_interface: np.ndarray
    overlap_measure: np.ndarray
    transmissibility: np.ndarray
    metadata: Mapping[str, Any] | None = None
    diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        shape = tuple(int(v) for v in self.raw_shape)
        if len(shape) != 3 or any(v <= 0 for v in shape):
            raise ValueError("raw_shape must contain three positive dimensions")
        owner_ids = np.asarray(self.owner_flat_ids, dtype=np.int64).reshape(-1)
        volumes = np.asarray(self.owner_volumes, dtype=float).reshape(-1)
        if owner_ids.size == 0 or owner_ids.size != volumes.size:
            raise ValueError("owner ids and volumes must be nonempty and have equal lengths")
        if np.any(owner_ids < 0) or np.any(owner_ids >= int(np.prod(shape))) or np.unique(owner_ids).size != owner_ids.size:
            raise ValueError("owner_flat_ids must be unique in raw storage bounds")
        if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0.0):
            raise ValueError("owner_volumes must be finite and positive")
        arrays = [np.asarray(self.link_owner_a, dtype=np.int32).reshape(-1), np.asarray(self.link_owner_b, dtype=np.int32).reshape(-1), np.asarray(self.link_interface, dtype=np.int32).reshape(-1), np.asarray(self.overlap_measure, dtype=float).reshape(-1), np.asarray(self.transmissibility, dtype=float).reshape(-1)]
        if len({a.size for a in arrays}) != 1:
            raise ValueError("link arrays must have equal lengths")
        a, b, interface, measure, tau = arrays
        if np.any(a < 0) or np.any(a >= owner_ids.size) or np.any(b < 0) or np.any(b >= owner_ids.size):
            raise ValueError("link owner slots are out of range")
        if np.any(a == b) or np.any(interface < 0):
            raise ValueError("links require distinct owners and nonnegative interfaces")
        if np.any(~np.isfinite(measure)) or np.any(measure < -1.0e-14) or np.any(~np.isfinite(tau)) or np.any(tau < -1.0e-14):
            raise ValueError("overlap measures and transmissibilities must be nonnegative")
        object.__setattr__(self, "raw_shape", shape)
        object.__setattr__(self, "owner_flat_ids", owner_ids)
        object.__setattr__(self, "owner_volumes", volumes)
        for name, value in zip(("link_owner_a", "link_owner_b", "link_interface", "overlap_measure", "transmissibility"), arrays):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics or {}))

    @property
    def n_owner(self) -> int:
        return int(self.owner_flat_ids.size)

    @property
    def n_link(self) -> int:
        return int(self.link_owner_a.size)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.raw_shape

    @property
    def owner_slot(self) -> np.ndarray:
        """Dense raw-storage-to-compact owner map (``-1`` for aliases)."""

        result = np.full(int(np.prod(self.raw_shape)), -1, dtype=np.int32)
        result[self.owner_flat_ids] = np.arange(self.n_owner, dtype=np.int32)
        return result.reshape(self.raw_shape)

    @property
    def owner_volume(self) -> np.ndarray:
        return self.owner_volumes

    @property
    def owner_ids(self) -> np.ndarray:
        """Alias retained for callers that call compact ids ``owner_ids``."""

        return self.owner_flat_ids

    @property
    def link_owner_slots(self) -> np.ndarray:
        return np.column_stack((self.link_owner_a, self.link_owner_b))

    @property
    def interface_ids(self) -> np.ndarray:
        return self.link_interface

    @property
    def overlap_measures(self) -> np.ndarray:
        return self.overlap_measure

    @property
    def tau(self) -> np.ndarray:
        return self.transmissibility

    @property
    def owner_compact_ids(self) -> np.ndarray:
        return np.arange(self.n_owner, dtype=np.int32)

    @property
    def endpoint_a(self) -> np.ndarray:
        return self.link_owner_a

    @property
    def endpoint_b(self) -> np.ndarray:
        return self.link_owner_b

    @property
    def link_owner_a_slot(self) -> np.ndarray:
        return self.link_owner_a

    @property
    def link_owner_b_slot(self) -> np.ndarray:
        return self.link_owner_b

    def metadata_dict(self) -> dict[str, Any]:
        result = dict(self.metadata)
        result.setdefault("raw_shape", list(self.raw_shape))
        result.setdefault("owner_count", self.n_owner)
        result.setdefault("link_count", self.n_link)
        result["diagnostics"] = dict(self.diagnostics)
        return result


def write_rlp_parallel_overlap_geometry(
    path: str | Path,
    geometry: GlobalRlpParallelOverlapGeometry,
    *,
    identity: Mapping[str, Any],
) -> Path:
    """Atomically write the versioned owner-overlap cache payload."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": np.asarray(RLP_PARALLEL_OVERLAP_CACHE_VERSION),
        "identity_json": np.asarray(json.dumps(dict(identity), sort_keys=True)),
        "metadata_json": np.asarray(json.dumps(dict(geometry.metadata), sort_keys=True)),
        "diagnostics_json": np.asarray(json.dumps(dict(geometry.diagnostics), sort_keys=True)),
        "raw_shape": np.asarray(geometry.raw_shape, dtype=np.int32),
        "owner_flat_ids": geometry.owner_flat_ids,
        "owner_volumes": geometry.owner_volumes,
        "link_owner_a": geometry.link_owner_a,
        "link_owner_b": geometry.link_owner_b,
        "link_interface": geometry.link_interface,
        "overlap_measure": geometry.overlap_measure,
        "transmissibility": geometry.transmissibility,
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.stem}.",
            suffix=".npz",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **payload)
        temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return target


def load_rlp_parallel_overlap_geometry(
    path: str | Path,
    *,
    expected_identity: Mapping[str, Any],
) -> GlobalRlpParallelOverlapGeometry:
    """Load a cache only when its complete construction identity matches."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as cached:
        if int(cached["format_version"].item()) != RLP_PARALLEL_OVERLAP_CACHE_VERSION:
            raise ValueError("owner-overlap cache format version mismatch")
        identity = json.loads(str(cached["identity_json"].item()))
        if identity != dict(expected_identity):
            raise ValueError("owner-overlap cache identity mismatch")
        return GlobalRlpParallelOverlapGeometry(
            raw_shape=tuple(int(value) for value in cached["raw_shape"]),
            owner_flat_ids=np.array(cached["owner_flat_ids"], copy=True),
            owner_volumes=np.array(cached["owner_volumes"], copy=True),
            link_owner_a=np.array(cached["link_owner_a"], copy=True),
            link_owner_b=np.array(cached["link_owner_b"], copy=True),
            link_interface=np.array(cached["link_interface"], copy=True),
            overlap_measure=np.array(cached["overlap_measure"], copy=True),
            transmissibility=np.array(cached["transmissibility"], copy=True),
            metadata=json.loads(str(cached["metadata_json"].item())),
            diagnostics=json.loads(str(cached["diagnostics_json"].item())),
        )


def merge_rlp_parallel_overlap_geometries(
    geometries: Iterable[GlobalRlpParallelOverlapGeometry],
    *,
    expected_interfaces: Iterable[int] | None = None,
) -> GlobalRlpParallelOverlapGeometry:
    """Merge validated interface-sized owner graphs.

    Callers can build/cache one interface batch at a time and release its
    temporary polygons before constructing the next batch.  This merge keeps
    only compact owner/link arrays and rejects duplicate or missing interface
    batches instead of silently overwriting coverage diagnostics.
    """
    parts = list(geometries)
    if not parts:
        raise ValueError("at least one partial owner-overlap geometry is required")
    reference = parts[0]
    shape = tuple(reference.raw_shape)
    expected = (
        tuple(range(shape[2]))
        if expected_interfaces is None
        else tuple(dict.fromkeys(int(value) for value in expected_interfaces))
    )
    if not expected or any(value < 0 or value >= shape[2] for value in expected):
        raise ValueError("expected_interfaces must contain valid eta interfaces")
    seen: set[int] = set()
    for part in parts:
        if tuple(part.raw_shape) != shape:
            raise ValueError("partial owner-overlap geometries have incompatible raw shapes")
        if not np.array_equal(part.owner_flat_ids, reference.owner_flat_ids):
            raise ValueError("partial owner-overlap geometries have incompatible owner ids")
        if not np.array_equal(part.owner_volumes, reference.owner_volumes):
            raise ValueError("partial owner-overlap geometries have incompatible owner volumes")
        ignored_metadata = {
            "interface_indices", "merged_partial_interfaces", "partial_interface_count",
        }
        metadata_keys = (set(reference.metadata) | set(part.metadata)) - ignored_metadata
        for name in metadata_keys:
            if part.metadata.get(name) != reference.metadata.get(name):
                raise ValueError(
                    f"partial owner-overlap metadata mismatch for construction control {name!r}"
                )
        built = tuple(int(value) for value in part.diagnostics.get("interfaces_built", ()))
        if not built:
            built = tuple(int(value) for value in part.metadata.get("interface_indices", ()))
        if not built:
            raise ValueError("partial owner-overlap geometry does not identify its interfaces")
        if any(value not in expected for value in built) or seen.intersection(built):
            raise ValueError("duplicate or unexpected owner-overlap interface batch")
        seen.update(built)
    if seen != set(expected):
        missing = sorted(set(expected) - seen)
        raise ValueError(f"owner-overlap merge is missing interface(s) {missing}")

    link_keys = []
    link_measures = []
    link_tau = []
    for part in parts:
        keys = np.column_stack((part.link_owner_a, part.link_owner_b, part.link_interface))
        link_keys.append(keys)
        link_measures.append(part.overlap_measure)
        link_tau.append(part.transmissibility)
    keys = np.concatenate(link_keys, axis=0) if link_keys else np.empty((0, 3), dtype=np.int32)
    if keys.size and np.unique(keys, axis=0).shape[0] != keys.shape[0]:
        raise ValueError("owner-overlap merge contains duplicate canonical links")
    measures = np.concatenate(link_measures, axis=0) if link_measures else np.empty(0, dtype=float)
    tau = np.concatenate(link_tau, axis=0) if link_tau else np.empty(0, dtype=float)
    if keys.size:
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        keys, measures, tau = keys[order], measures[order], tau[order]

    indexed_diagnostics = (
        "source_closure_error", "destination_closure_error", "source_coverage", "destination_coverage",
        "source_expected_nonwall", "destination_expected_nonwall", "source_expected_wall", "destination_expected_wall",
        "source_covered_nonwall", "destination_covered_nonwall", "source_covered_wall", "destination_covered_wall",
        "wall_terminated_source", "wall_terminated_destination", "source_undercoverage_at_destination_wall",
        "destination_undercoverage_at_source_wall", "destination_wall_capacity_for_source_undercoverage",
        "source_wall_capacity_for_destination_undercoverage",
    )
    diagnostics: dict[str, Any] = {}
    for name in indexed_diagnostics:
        values = np.zeros(shape[2], dtype=float)
        for part in parts:
            built = tuple(int(value) for value in part.diagnostics.get("interfaces_built", ()))
            array = np.asarray(part.diagnostics.get(name, np.zeros(shape[2])), dtype=float).reshape(-1)
            if array.size != shape[2]:
                raise ValueError(f"partial diagnostic {name} has unexpected interface length")
            values[list(built)] = array[list(built)]
        diagnostics[name] = values.tolist()
    diagnostics["interfaces_built"] = list(expected)
    diagnostics["edge_refinement"] = [
        item for part in parts for item in part.diagnostics.get("edge_refinement", [])
    ]
    diagnostics["mixed_wall"] = [
        item for part in parts for item in part.diagnostics.get("mixed_wall", [])
    ]
    diagnostics["interface_work"] = sorted(
        (item for part in parts for item in part.diagnostics.get("interface_work", [])),
        key=lambda item: int(item.get("interface", -1)),
    )
    counters: dict[str, int] = {}
    for part in parts:
        for name, value in part.diagnostics.get("resource_counters", {}).items():
            value = int(value)
            if name.startswith("peak_"):
                counters[name] = max(counters.get(name, 0), value)
            else:
                counters[name] = counters.get(name, 0) + value
    diagnostics["resource_counters"] = counters
    diagnostics["elapsed_seconds"] = float(sum(float(part.diagnostics.get("elapsed_seconds", 0.0)) for part in parts))
    metadata = dict(reference.metadata)
    metadata["interface_indices"] = list(expected)
    metadata["merged_partial_interfaces"] = True
    metadata["partial_interface_count"] = len(parts)
    return GlobalRlpParallelOverlapGeometry(
        raw_shape=shape,
        owner_flat_ids=np.array(reference.owner_flat_ids, copy=True),
        owner_volumes=np.array(reference.owner_volumes, copy=True),
        link_owner_a=keys[:, 0] if keys.size else np.empty(0, dtype=np.int32),
        link_owner_b=keys[:, 1] if keys.size else np.empty(0, dtype=np.int32),
        link_interface=keys[:, 2] if keys.size else np.empty(0, dtype=np.int32),
        overlap_measure=measures,
        transmissibility=tau,
        metadata=metadata,
        diagnostics=diagnostics,
    )


@dataclass
class _Face:
    owner: int
    plane: int
    vertices: np.ndarray
    length: float = 1.0
    boundary: bool = False


@dataclass
class _MappedPiece:
    polygon: np.ndarray
    length: float
    wall: bool = False
    # Optional affine model of traced arc length over the mapped polygon.
    # Keeping this model rather than one face-wide mean retains the first
    # spatial variation needed by the second-order overlap quadrature.
    length_coefficients: np.ndarray | None = None

    def length_at(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if self.length_coefficients is None:
            return np.full(points.shape[0], float(self.length), dtype=float)
        coefficients = np.asarray(self.length_coefficients, dtype=float).reshape(3)
        return coefficients[0] + points @ coefficients[1:]


def _fit_length_model(points: np.ndarray, lengths: np.ndarray) -> np.ndarray | None:
    """Fit a stable affine mapped-chart model for traced endpoint lengths."""

    points = np.asarray(points, dtype=float).reshape((-1, 2))
    lengths = np.asarray(lengths, dtype=float).reshape(-1)
    if points.shape[0] != lengths.size or points.shape[0] < 3:
        return None
    design = np.column_stack((np.ones(points.shape[0]), points))
    if np.linalg.matrix_rank(design) < 3:
        return None
    coefficients, *_ = np.linalg.lstsq(design, lengths, rcond=None)
    if not np.all(np.isfinite(coefficients)):
        return None
    return np.asarray(coefficients, dtype=float)


def _get_attr(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if obj is not None and hasattr(obj, name):
            return getattr(obj, name)
    return default


def _shape_from_inputs(geometry: Any, owner_geometry: Any, raw_shape: Any) -> tuple[int, int, int]:
    if raw_shape is not None:
        result = tuple(int(v) for v in raw_shape)
    else:
        shape = _get_attr(geometry, ("shape",), None)
        topology = _get_attr(owner_geometry, ("topology",), owner_geometry)
        shape = shape if shape is not None else _get_attr(topology, ("shape",), None)
        if shape is None:
            raise ValueError("raw_shape is required when geometry has no shape")
        result = tuple(int(v) for v in shape)
    if len(result) != 3 or any(v <= 0 for v in result):
        raise ValueError("raw_shape must contain three positive dimensions")
    return result


def _owner_payload(geometry: Any, owner_geometry: Any, shape: tuple[int, int, int], raw_faces: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    topology = _get_attr(owner_geometry, ("topology",), owner_geometry)
    ids = _get_attr(topology, ("aggregate_id", "owner_flat_ids"), None)
    if ids is None:
        ids = _get_attr(owner_geometry, ("aggregate_id", "owner_flat_ids", "owner_map"), None)
    active = _get_attr(topology, ("is_active_owner", "active_owner"), None)
    volumes = _get_attr(owner_geometry, ("aggregate_chart_volume", "aggregate_volume", "owner_volumes", "volumes"), None)
    if volumes is None:
        volumes = _get_attr(topology, ("aggregate_volume",), None)
    if ids is not None:
        ids = np.asarray(ids, dtype=np.int64)
        if ids.shape == shape:
            if active is None:
                active = ids >= 0
            owner_ids = np.unique(ids[np.asarray(active, dtype=bool)])
        else:
            owner_ids = np.unique(ids.reshape(-1))
        owner_ids = owner_ids[owner_ids >= 0]
    elif raw_faces is not None:
        candidate = []
        for item in raw_faces:
            owner = item.get("owner", item.get("owner_flat_id")) if isinstance(item, Mapping) else getattr(item, "owner", None)
            if owner is not None:
                candidate.append(int(owner) if np.ndim(owner) == 0 else int(np.ravel_multi_index(tuple(owner), shape)))
        owner_ids = np.unique(candidate).astype(np.int64)
    else:
        raise ValueError("owner geometry must provide aggregate_id or raw_faces must provide owner ids")
    if owner_ids.size == 0:
        raise ValueError("owner geometry contains no active owners")
    if volumes is None:
        volumes_by_id = np.ones(owner_ids.size, dtype=float)
    else:
        volume_array = np.asarray(volumes, dtype=float)
        if volume_array.shape == shape:
            values = volume_array.reshape(-1)[owner_ids]
        else:
            flat = volume_array.reshape(-1)
            values = flat[owner_ids] if flat.size > int(np.max(owner_ids)) else flat[:owner_ids.size]
        volumes_by_id = values.astype(float)
    if ids is not None and np.asarray(ids).shape == shape:
        slot = {int(v): i for i, v in enumerate(owner_ids)}
        owner_map = np.full(shape, -1, dtype=np.int64)
        for flat, value in enumerate(ids.reshape(-1)):
            if int(value) in slot:
                # Keep this map in canonical storage-flat-id space.  Compact
                # slots are an output representation, never geometry ids.
                owner_map.reshape(-1)[flat] = int(value)
    else:
        owner_map = np.full(shape, -1, dtype=np.int64)
    return owner_ids, volumes_by_id, owner_map


def _grid_faces(geometry: Any, axis: str, count: int) -> np.ndarray:
    grid = _get_attr(geometry, ("grid",), None)
    axis_obj = _get_attr(grid, (axis,), None)
    faces = _get_attr(axis_obj, ("faces",), None)
    if faces is None:
        if axis == "x":
            return np.arange(count + 1, dtype=float)
        return np.linspace(0.0, 2.0 * np.pi, count + 1)
    return np.asarray(faces, dtype=float).reshape(-1)


def _make_faces(geometry: Any, owner_map: np.ndarray, shape: tuple[int, int, int], raw_faces: Any) -> list[_Face]:
    if raw_faces is not None:
        result: list[_Face] = []
        for item in raw_faces:
            get = (lambda key, default=None: item.get(key, default)) if isinstance(item, Mapping) else (lambda key, default=None: getattr(item, key, default))
            owner = get("owner", get("owner_flat_id", None))
            if owner is None:
                raise ValueError("each raw face needs an owner")
            if np.ndim(owner) != 0:
                owner = int(np.ravel_multi_index(tuple(owner), shape))
            plane = int(get("plane", get("eta", get("k", 0)))) % shape[2]
            vertices = np.asarray(get("vertices", get("polygon", None)), dtype=float)
            if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 3:
                raise ValueError("raw face vertices must have shape (n >= 3, 2)")
            result.append(_Face(int(owner), plane, vertices, float(get("length", get("arc_length", 1.0))), bool(get("boundary", False))))
        return result
    uf = _grid_faces(geometry, "x", shape[0])
    tf = _grid_faces(geometry, "y", shape[1])
    if uf.size != shape[0] + 1 or tf.size != shape[1] + 1:
        raise ValueError("geometry grid face arrays have unexpected lengths")
    result = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            owner_row = owner_map[i, j, :]
            for k in range(shape[2]):
                owner = int(owner_row[k])
                if owner < 0:
                    continue
                vertices = np.array([[uf[i], tf[j]], [uf[i + 1], tf[j]], [uf[i + 1], tf[j + 1]], [uf[i], tf[j + 1]]], dtype=float)
                result.append(_Face(owner, k, vertices))
    return result


def _dissolve_owner_faces(faces: list[_Face]) -> list[_Face]:
    """Dissolve coplanar raw faces that belong to the same RLP owner.

    Raw polar cells are geometric subfaces, not state entries.  Tracing their
    common edges independently is both wasteful and particularly ill
    conditioned for the owner touching the magnetic axis: many radial edges
    meet at the origin although they are entirely internal to one owner.
    This routine cancels oppositely oriented, coincident edges and assembles
    the remaining edges into one or more boundary rings per ``(plane, owner)``.

    The operation is deliberately a small polygon-boundary union rather than
    a general-purpose polygon clipping dependency.  It preserves nonconvex
    owners and disconnected components.  A malformed edge graph is rejected
    explicitly; callers must not silently return to point-sampled FCI rows.
    """

    if not faces:
        return []

    # Raw face coordinates can differ by a few ulps after chart conversion.
    # Quantizing only endpoint *keys* makes cancellation tolerant without
    # moving the actual geometry used for tracing and quadrature.
    all_points = np.concatenate([_regular_xy(face.vertices) for face in faces], axis=0)
    scale = max(1.0, float(np.max(np.abs(all_points))))
    snap = 1.0e-12 * scale

    def point_key(point: np.ndarray) -> tuple[int, int]:
        return tuple(np.rint(np.asarray(point, dtype=float) / snap).astype(np.int64))

    grouped: dict[tuple[int, int], list[_Face]] = {}
    for face in faces:
        grouped.setdefault((int(face.plane), int(face.owner)), []).append(face)

    result: list[_Face] = []
    for (plane, owner), group in grouped.items():
        # Keep all unpaired directed edges.  Coincident internal edges occur
        # with opposite orientation for consistently oriented raw faces.
        edge_records: dict[
            tuple[tuple[int, int], tuple[int, int]],
            list[tuple[tuple[int, int], tuple[int, int], np.ndarray, np.ndarray]],
        ] = {}
        area_before = 0.0
        length_weight = 0.0
        length_area = 0.0
        boundary_count = 0
        for face in group:
            polygon = _as_ccw(_simplify_polygon(_regular_xy(face.vertices)))
            area = abs(_polygon_area(polygon))
            if area <= 1.0e-30:
                continue
            area_before += area
            length_weight += area * max(float(face.length), 1.0e-30)
            length_area += area
            boundary_count += int(bool(face.boundary))
            for index in range(polygon.shape[0]):
                start = polygon[index]
                end = polygon[(index + 1) % polygon.shape[0]]
                start_key = point_key(start)
                end_key = point_key(end)
                if start_key == end_key:
                    continue
                undirected = tuple(sorted((start_key, end_key)))
                records = edge_records.setdefault(undirected, [])
                reverse = next(
                    (
                        record_index
                        for record_index, record in enumerate(records)
                        if record[0] == end_key and record[1] == start_key
                    ),
                    None,
                )
                if reverse is None:
                    records.append((start_key, end_key, start.copy(), end.copy()))
                else:
                    records.pop(reverse)

        boundary_edges = [record for records in edge_records.values() for record in records]
        if area_before <= 1.0e-30:
            continue
        if not boundary_edges:
            raise ValueError(
                f"owner-overlap owner={owner}, plane={plane} has no boundary after raw-face dissolution"
            )

        # Assemble each directed ring.  A valid manifold owner boundary has
        # one outgoing edge per boundary vertex.  At a point-touching corner,
        # use the smallest left turn to keep each ring locally connected.
        outgoing: dict[tuple[int, int], list[int]] = {}
        for edge_index, record in enumerate(boundary_edges):
            outgoing.setdefault(record[0], []).append(edge_index)
        unused = set(range(len(boundary_edges)))
        rings: list[np.ndarray] = []
        while unused:
            first = next(iter(unused))
            record = boundary_edges[first]
            ring_keys = [record[0]]
            ring_points = [record[2]]
            current_index = first
            previous_direction = record[3] - record[2]
            unused.remove(first)
            while True:
                current = boundary_edges[current_index]
                end_key = current[1]
                if end_key == ring_keys[0]:
                    break
                ring_keys.append(end_key)
                ring_points.append(current[3])
                candidates = [index for index in outgoing.get(end_key, ()) if index in unused]
                if not candidates:
                    raise ValueError(
                        f"owner-overlap owner={owner}, plane={plane} has an open dissolved boundary"
                    )
                if len(candidates) == 1:
                    next_index = candidates[0]
                else:
                    # Prefer the continuation with the smallest positive
                    # turning angle.  This only matters for point contacts;
                    # ordinary raw-cell unions have one candidate.
                    angles = []
                    for candidate in candidates:
                        direction = boundary_edges[candidate][3] - boundary_edges[candidate][2]
                        angle = np.arctan2(
                            _cross2(previous_direction, direction),
                            float(np.dot(previous_direction, direction)),
                        )
                        angles.append(angle if angle >= 0.0 else angle + 2.0 * np.pi)
                    next_index = candidates[int(np.argmin(angles))]
                unused.remove(next_index)
                previous_direction = boundary_edges[next_index][3] - boundary_edges[next_index][2]
                current_index = next_index
            polygon = _simplify_polygon(np.asarray(ring_points, dtype=float))
            if polygon.shape[0] >= 3 and abs(_polygon_area(polygon)) > 1.0e-30:
                rings.append(polygon)

        area_after = sum(abs(_polygon_area(ring)) for ring in rings)
        if not np.isclose(area_after, area_before, rtol=1.0e-10, atol=1.0e-14 * max(area_before, 1.0)):
            raise ValueError(
                f"owner-overlap owner={owner}, plane={plane} dissolved-area mismatch: "
                f"raw={area_before:.16e}, boundary={area_after:.16e}"
            )
        mean_length = length_weight / max(length_area, 1.0e-30)
        for ring in rings:
            logical = _logical_from_xy(ring, 0.0)[:, :2]
            result.append(
                _Face(
                    owner=owner,
                    plane=plane,
                    vertices=logical,
                    length=mean_length,
                    boundary=boundary_count == len(group),
                )
            )
    return result


def build_rlp_parallel_overlap_geometry(
    geometry: Any = None,
    owner_host_geometry: Any = None,
    *,
    raw_shape: tuple[int, int, int] | None = None,
    raw_faces: Iterable[Any] | None = None,
    trace: Callable[..., Any] | None = None,
    trace_callback: Callable[..., Any] | None = None,
    metric_callback: Callable[..., Any] | None = None,
    subdivision_tolerance: float | None = None,
    second_order_tolerance_factor: float = 5.0e-2,
    coverage_tolerance: float = 1.0e-10,
    max_subdivision: int = 8,
    max_edge_segments: int = 131072,
    max_mapped_pieces: int = 16384,
    max_overlap_candidates: int = 2_000_000,
    max_polygon_vertices: int = 256,
    trace_chunk_size: int = 32768,
    metric_batch_size: int = 32768,
    max_quadrature_points: int | None = None,
    progress_callback: Callable[[Mapping[str, Any]], Any] | None = None,
    interface_indices: Iterable[int] | None = None,
    tracer_kwargs: Mapping[str, Any] | None = None,
    tracer_grid: Any = None,
    field_evaluator: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> GlobalRlpParallelOverlapGeometry:
    """Construct the global owner-overlap graph.

    ``raw_faces`` entries are mappings (or objects) with ``owner``, ``plane``
    and ``vertices`` fields. Vertices are logical ``(u, theta)`` coordinates;
    owners may be compact flat ids or ``(i, j, k)`` indices.  The optional
    tracer returns endpoint coordinates, optionally with length and wall
    flags.  A wall-terminated face is omitted under homogeneous Neumann
    closure.  Interior closure loss raises instead of silently falling back to
    point-sampled rows.
    """

    shape = _shape_from_inputs(geometry, owner_host_geometry, raw_shape)
    if int(max_edge_segments) < 1:
        raise ValueError("max_edge_segments must be positive")
    if int(max_mapped_pieces) < 1 or int(max_overlap_candidates) < 1:
        raise ValueError("mapped-piece and overlap-candidate budgets must be positive")
    if int(max_polygon_vertices) < 3:
        raise ValueError("max_polygon_vertices must be at least three")
    if int(trace_chunk_size) < 1:
        raise ValueError("trace_chunk_size must be positive")
    if int(metric_batch_size) < 1:
        raise ValueError("metric_batch_size must be positive")
    if max_quadrature_points is not None and int(max_quadrature_points) < 1:
        raise ValueError("max_quadrature_points must be positive when supplied")
    if not np.isfinite(second_order_tolerance_factor) or second_order_tolerance_factor <= 0.0:
        raise ValueError("second_order_tolerance_factor must be finite and positive")
    if subdivision_tolerance is not None and (
        not np.isfinite(subdivision_tolerance) or subdivision_tolerance <= 0.0
    ):
        raise ValueError("subdivision_tolerance must be finite and positive when supplied")
    if interface_indices is None:
        selected_interfaces = tuple(range(shape[2]))
    else:
        selected_interfaces = tuple(dict.fromkeys(int(value) for value in interface_indices))
        if not selected_interfaces or any(value < 0 or value >= shape[2] for value in selected_interfaces):
            raise ValueError("interface_indices must contain valid eta-plane interfaces")
    explicit_faces = list(raw_faces) if raw_faces is not None else None
    owner_ids, owner_volumes, owner_map = _owner_payload(geometry, owner_host_geometry, shape, explicit_faces)
    slot_by_id = {int(v): i for i, v in enumerate(owner_ids)}
    faces = _make_faces(geometry, owner_map, shape, explicit_faces)
    if not faces:
        raise ValueError("no positive-volume owner faces were found")
    raw_face_count = len(faces)
    faces = _dissolve_owner_faces(faces)
    if not faces:
        raise ValueError("no owner boundaries remained after raw-face dissolution")
    callback = trace_callback if trace_callback is not None else trace
    if callback is None:
        callback = _get_attr(geometry, ("trace_fci_points_to_plane_from_callbacks", "trace_points_to_plane"), None)

    z_centers = _get_attr(_get_attr(geometry, ("grid",), None), ("z",), None)
    z_centers = _get_attr(z_centers, ("centers",), None)
    if z_centers is None:
        eta_centers = np.arange(shape[2], dtype=float)
        eta_period = float(shape[2])
    else:
        eta_centers = np.asarray(z_centers, dtype=float).reshape(-1)
        if eta_centers.size != shape[2]:
            raise ValueError("geometry eta centers have unexpected length")
        eta_period = _get_attr(_get_attr(geometry, ("grid",), None), ("eta_period",), None)
        if eta_period is None:
            eta_period = float(eta_centers[-1] - eta_centers[0] + (eta_centers[1] - eta_centers[0] if shape[2] > 1 else 1.0))
        eta_period = float(eta_period)

    def _interface_eta(interface: int) -> tuple[float, float, float]:
        source = float(eta_centers[interface])
        destination_index = (interface + 1) % shape[2]
        destination = float(eta_centers[destination_index])
        if destination_index == 0 and shape[2] > 1:
            destination += eta_period
        return source, destination, 0.5 * (source + destination)

    def map_points(points: np.ndarray, direction: str, target_eta: int, eta_step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if callback is None:
            endpoint = np.asarray(points, dtype=float)
            if endpoint.shape[1] == 2:
                endpoint = np.column_stack((endpoint, np.full(endpoint.shape[0], float(target_eta))))
            return endpoint, np.ones(endpoint.shape[0]), np.zeros(endpoint.shape[0], dtype=bool)
        # The production helper has the explicit signature
        # ``(grid, field_evaluator, seed_points, eta_step, ...)``.  Keep this
        # branch separate from the compact analytic callback forms above.
        if tracer_grid is not None or field_evaluator is not None:
            options = dict(tracer_kwargs or {})
            options.pop("grid", None)
            options.pop("field_evaluator", None)
            value = callback(tracer_grid, field_evaluator, points, eta_step, **options)
            return _trace_result(value, points.shape[0])
        return _trace_result(_call_flexible(callback, points, direction, target_eta, eta_step, tracer_kwargs), points.shape[0])

    # These counters are deliberately host-side and scalar.  They make the
    # expensive part of the prototype observable and let callers reject a
    # build before it can retain an unbounded collection of polygons.
    resource_counters: dict[str, int] = {
        "traced_points": 0,
        "mapped_faces": 0,
        "mapped_pieces": 0,
        "overlap_candidates": 0,
        "clip_calls": 0,
        "interfaces_completed": 0,
        "peak_pending_edge_segments": 0,
        "peak_candidate_set": 0,
        "peak_source_pieces": 0,
        "peak_destination_pieces": 0,
        "trace_evaluated_points": 0,
        "trace_padding_points": 0,
        "quadrature_real_points": 0,
        "quadrature_evaluated_points": 0,
        "quadrature_batches": 0,
        "peak_quadrature_batch_points": 0,
        "peak_quadrature_buffer_points": 0,
    }
    build_started = time.perf_counter()
    refinement_diagnostics: list[dict[str, Any]] = []
    mixed_wall_diagnostics: list[dict[str, Any]] = []
    interface_diagnostics: list[dict[str, Any]] = []

    def mapped_faces(
        input_faces: list[_Face],
        direction: str,
        target_eta: int,
        source_eta: float,
        eta_step: float,
        *,
        trace_chunk_size: int = 32768,
    ) -> list[tuple[_Face, list[_MappedPiece], float]]:
        """Trace shared face edges adaptively with bounded NumPy batches.

        Only the image of a face boundary determines its geometric overlap.
        Refining the full triangle interior created exponentially many Python
        polygon objects at 32^3.  Here each raw face is still tessellated in
        the regular chart, but adaptive midpoint tests are applied to its four
        mapped edges; the resulting simple polygon is triangulated only when
        it is nonconvex.
        """

        if not input_faces:
            return []

        # Segment record: face, edge, parameter interval, logical-chart ends,
        # mapped ends, endpoint lengths/wall flags, and absolute tolerance.
        Segment = tuple[
            int,
            int,
            float,
            float,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            float,
            float,
            bool,
            bool,
            float,
        ]

        def trace_samples(samples: np.ndarray):
            resource_counters["traced_points"] += int(samples.shape[0])
            endpoint_parts = []
            length_parts = []
            wall_parts = []
            for start in range(0, samples.shape[0], trace_chunk_size):
                stop = min(start + trace_chunk_size, samples.shape[0])
                real_samples = samples[start:stop]
                real_count = int(real_samples.shape[0])
                # Canonical input lengths keep JAX tracing/evaluator cache
                # keys stable across refinement levels.  Padding repeats a
                # real point and is removed before returning, so it cannot
                # alter geometry or wall classifications.
                bucket = 1
                while bucket < real_count:
                    bucket <<= 1
                bucket = min(int(trace_chunk_size), max(256, bucket))
                if bucket < real_count:
                    bucket = real_count
                padded = real_samples
                if bucket > real_count:
                    padded = np.concatenate(
                        (real_samples, np.repeat(real_samples[-1][None, :], bucket - real_count, axis=0)),
                        axis=0,
                    )
                resource_counters["trace_evaluated_points"] += int(bucket)
                resource_counters["trace_padding_points"] += int(bucket - real_count)
                logical = _logical_from_xy(padded, source_eta)
                endpoint, length, wall = map_points(
                    logical, direction, target_eta, eta_step
                )
                endpoint_parts.append(_regular_xy(endpoint)[:real_count])
                length_parts.append(np.asarray(length, dtype=float)[:real_count])
                wall_parts.append(np.asarray(wall, dtype=bool)[:real_count])
            return (
                np.concatenate(endpoint_parts, axis=0),
                np.concatenate(length_parts, axis=0),
                np.concatenate(wall_parts, axis=0),
            )

        regular_vertices = [_regular_xy(face.vertices) for face in input_faces]
        radial_upper = float(_grid_faces(geometry, "x", shape[0])[-1])
        touches_outer_wall = [
            bool(
                np.any(
                    np.isclose(
                        np.asarray(face.vertices, dtype=float)[:, 0],
                        radial_upper,
                        rtol=0.0,
                        atol=1.0e-12 * max(1.0, abs(radial_upper)),
                    )
                )
            )
            for face in input_faces
        ]
        initial_samples = np.concatenate(regular_vertices, axis=0)
        unique_samples, inverse = np.unique(initial_samples, axis=0, return_inverse=True)
        mapped_unique, length_unique, wall_unique = trace_samples(unique_samples)
        mapped_initial = mapped_unique[inverse]
        length_initial = length_unique[inverse]
        wall_initial = wall_unique[inverse]

        pending: list[Segment] = []
        offset = 0
        for face_index, vertices in enumerate(regular_vertices):
            count = vertices.shape[0]
            mapped_vertices = mapped_initial[offset : offset + count]
            vertex_lengths = length_initial[offset : offset + count]
            vertex_walls = wall_initial[offset : offset + count]
            offset += count
            diameter = max(
                float(np.linalg.norm(vertices[i] - vertices[j]))
                for i in range(count)
                for j in range(i + 1, count)
            )
            # The endpoint map need only be resolved to the order of the
            # evolved field.  The old 1e-6 relative target drove refinement
            # far below the second-order spatial truncation error.  By
            # default use an absolute mapped-edge error O(h^2), with h the
            # local raw-face diameter.  Explicit legacy tolerances are
            # retained as a lower bound on requested accuracy only when they
            # are no *finer* than this second-order floor.
            diameter = max(diameter, 1.0e-30)
            second_order_floor = float(second_order_tolerance_factor) * diameter * diameter
            requested = (
                second_order_floor
                if subdivision_tolerance is None
                else float(subdivision_tolerance) * diameter
            )
            tolerance = max(requested, second_order_floor)
            for edge in range(count):
                following = (edge + 1) % count
                pending.append(
                    (
                        face_index,
                        edge,
                        0.0,
                        1.0,
                        vertices[edge],
                        vertices[following],
                        mapped_vertices[edge],
                        mapped_vertices[following],
                        float(vertex_lengths[edge]),
                        float(vertex_lengths[following]),
                        bool(vertex_walls[edge]),
                        bool(vertex_walls[following]),
                        tolerance,
                    )
                )

        accepted: list[list[list[tuple[Any, ...]]]] = [
            [[] for _ in range(vertices.shape[0])] for vertices in regular_vertices
        ]
        accepted_count = 0
        maximum_pending = len(pending)
        for depth in range(max_subdivision + 1):
            if not pending:
                break
            next_pending: list[Segment] = []
            segment_chunk_size = max(1, trace_chunk_size)
            for block_start in range(0, len(pending), segment_chunk_size):
                records = pending[block_start : block_start + segment_chunk_size]
                midpoint_samples = np.stack(
                    [0.5 * (record[4] + record[5]) for record in records], axis=0
                )
                unique_midpoints, midpoint_inverse = np.unique(
                    midpoint_samples, axis=0, return_inverse=True
                )
                mapped_mid, length_mid, wall_mid = trace_samples(unique_midpoints)
                mapped_mid = mapped_mid[midpoint_inverse]
                length_mid = length_mid[midpoint_inverse]
                wall_mid = wall_mid[midpoint_inverse]
                for index, record in enumerate(records):
                    (
                        face_index,
                        edge,
                        t0,
                        t1,
                        p0,
                        p1,
                        mapped0,
                        mapped1,
                        length0,
                        length1,
                        wall0,
                        wall1,
                        tolerance,
                    ) = record
                    midpoint = midpoint_samples[index]
                    midpoint_mapped = mapped_mid[index]
                    error = float(
                        np.linalg.norm(midpoint_mapped - 0.5 * (mapped0 + mapped1))
                    )
                    wall_transition = bool(wall_mid[index]) not in {wall0, wall1}
                    wall_transition |= wall0 != wall1
                    unresolved_wall_graze = False
                    if (
                        depth >= max_subdivision
                        and error > tolerance
                        and not wall_transition
                        and touches_outer_wall[face_index]
                    ):
                        # A narrow grazing interval can lie between the usual
                        # endpoint/midpoint samples.  Confirm it with bounded
                        # interior probes before treating the segment as wall;
                        # an unexplained nonsmooth interior map still fails.
                        fractions = np.arange(1.0, 8.0, dtype=float) / 8.0
                        probes = (
                            (1.0 - fractions[:, None]) * p0[None, :]
                            + fractions[:, None] * p1[None, :]
                        )
                        _probe_mapped, _probe_length, probe_wall = trace_samples(
                            probes
                        )
                        unresolved_wall_graze = bool(np.any(probe_wall))
                    if (
                        depth >= max_subdivision
                        and error > tolerance
                        and not wall_transition
                        and not unresolved_wall_graze
                    ):
                        face = input_faces[face_index]
                        raise ValueError(
                            "owner-overlap traced edge did not meet the mapped-edge "
                            f"tolerance after {max_subdivision} subdivisions: "
                            f"owner={face.owner}, plane={face.plane}, "
                            f"direction={direction}, error={error:.6e}, "
                            f"mixed_wall={wall_transition}, "
                            f"tolerance={tolerance:.6e}"
                        )
                    midpoint_t = 0.5 * (t0 + t1)
                    if depth >= max_subdivision or (
                        error <= tolerance and not wall_transition
                    ):
                        # A wall transition is a geometric cut, not an
                        # invalid traced face.  At the bounded refinement
                        # level the midpoint is our O(h**2) approximation to
                        # that cut.  Split the terminal segment there so the
                        # polygon assembler can retain only the interior
                        # pieces.  Wall pieces are homogeneous-Neumann and
                        # therefore contribute no link or transmissibility.
                        records_to_accept = []
                        if unresolved_wall_graze:
                            # A grazing wall-hit interval can be narrower than
                            # the three sampled points and still make the
                            # endpoint map discontinuous.  On an owner that
                            # touches the physical outer wall, discard this
                            # terminal boundary sliver as Neumann wall.  Its
                            # source width is bounded by the refinement depth,
                            # so the geometric loss remains O(h^2) or smaller.
                            records_to_accept.append(
                                (
                                    t0,
                                    t1,
                                    np.array(mapped0, copy=True),
                                    np.array(mapped1, copy=True),
                                    length0,
                                    length1,
                                    True,
                                    True,
                                )
                            )
                            mixed_wall_diagnostics.append(
                                {
                                    "owner": int(input_faces[face_index].owner),
                                    "plane": int(input_faces[face_index].plane),
                                    "direction": direction,
                                    "depth": int(depth),
                                    "edge": int(edge),
                                    "error": float(error),
                                    "tolerance": float(tolerance),
                                    "classification": "unresolved-wall-graze",
                                }
                            )
                        elif depth >= max_subdivision and wall_transition:
                            records_to_accept.extend(
                                (
                                    (t0, midpoint_t, np.array(mapped0, copy=True), np.array(midpoint_mapped, copy=True), length0, float(length_mid[index]), wall0, bool(wall_mid[index])),
                                    (midpoint_t, t1, np.array(midpoint_mapped, copy=True), np.array(mapped1, copy=True), float(length_mid[index]), length1, bool(wall_mid[index]), wall1),
                                )
                            )
                            mixed_wall_diagnostics.append(
                                {
                                    "owner": int(input_faces[face_index].owner),
                                    "plane": int(input_faces[face_index].plane),
                                    "direction": direction,
                                    "depth": int(depth),
                                    "edge": int(edge),
                                    "error": float(error),
                                    "tolerance": float(tolerance),
                                }
                            )
                        else:
                            records_to_accept.append(
                                (
                                    t0,
                                    t1,
                                    np.array(mapped0, copy=True),
                                    np.array(mapped1, copy=True),
                                    length0,
                                    length1,
                                    wall0,
                                    wall1,
                                )
                            )
                        accepted[face_index][edge].extend(records_to_accept)
                        accepted_count += len(records_to_accept)
                        if accepted_count > int(max_edge_segments):
                            raise MemoryError(
                                "owner-overlap edge refinement exceeded its bounded "
                                f"segment budget ({max_edge_segments})"
                            )
                        continue
                    common = (
                        float(length_mid[index]),
                        bool(wall_mid[index]),
                        tolerance,
                    )
                    if len(next_pending) + 2 > int(max_edge_segments):
                        raise MemoryError(
                            "owner-overlap edge refinement exceeded its bounded "
                            f"pending-segment budget ({max_edge_segments})"
                        )
                    next_pending.append(
                        (
                            face_index,
                            edge,
                            t0,
                            midpoint_t,
                            np.array(p0, copy=True),
                            np.array(midpoint, copy=True),
                            np.array(mapped0, copy=True),
                            np.array(midpoint_mapped, copy=True),
                            length0,
                            common[0],
                            wall0,
                            common[1],
                            common[2],
                        )
                    )
                    next_pending.append(
                        (
                            face_index,
                            edge,
                            midpoint_t,
                            t1,
                            np.array(midpoint, copy=True),
                            np.array(p1, copy=True),
                            np.array(midpoint_mapped, copy=True),
                            np.array(mapped1, copy=True),
                            common[0],
                            length1,
                            common[1],
                            wall1,
                            common[2],
                        )
                    )
            pending = next_pending
            maximum_pending = max(maximum_pending, len(pending))
            resource_counters["peak_pending_edge_segments"] = max(
                resource_counters["peak_pending_edge_segments"], len(pending)
            )

        result = []
        mapped_piece_count = 0

        def map_mixed_wall_face_boundary(face_index: int, face: _Face) -> tuple[list[_MappedPiece], float, int]:
            """Cut a mixed face from its already-refined boundary records.

            Wall transitions are localized by the edge midpoint refinement
            above.  Reconstructing the one interior boundary run here avoids
            any additional field evaluation and closes it with the straight
            transition chord.  A second disjoint run is a topology error and
            fails closed instead of silently changing the operator.
            """
            regular = regular_vertices[face_index]
            scale = max(1.0, float(np.max(np.abs(regular))))
            eps = 1.0e-13 * scale
            nodes: list[tuple[np.ndarray, np.ndarray, float, bool]] = []

            def add_node(source, mapped, length, wall):
                source = np.asarray(source, dtype=float)
                mapped = np.asarray(mapped, dtype=float)
                if nodes and np.linalg.norm(source - nodes[-1][0]) <= eps:
                    if bool(wall) != nodes[-1][3]:
                        raise ValueError(
                            "owner-overlap mixed-wall owner="
                            f"{face.owner}, plane={face.plane} has inconsistent transition node"
                        )
                    return
                nodes.append((source, mapped, float(length), bool(wall)))

            for edge, edge_segments in enumerate(accepted[face_index]):
                edge_segments.sort(key=lambda record: record[0])
                v0 = regular[edge]
                v1 = regular[(edge + 1) % regular.shape[0]]
                for record in edge_segments:
                    t0, t1, mapped0, mapped1, length0, length1, wall0, wall1 = record
                    source0 = (1.0 - float(t0)) * v0 + float(t0) * v1
                    source1 = (1.0 - float(t1)) * v0 + float(t1) * v1
                    add_node(source0, mapped0, length0, wall0)
                    add_node(source1, mapped1, length1, wall1)
            if len(nodes) > 1 and np.linalg.norm(nodes[0][0] - nodes[-1][0]) <= eps:
                if nodes[0][3] != nodes[-1][3]:
                    raise ValueError(
                        "owner-overlap mixed-wall owner="
                        f"{face.owner}, plane={face.plane} has inconsistent cyclic transition node"
                    )
                nodes.pop()
            if len(nodes) < 3:
                raise ValueError(
                    f"owner-overlap mixed-wall owner={face.owner}, plane={face.plane} has too few boundary nodes"
                )
            flags = [node[3] for node in nodes]
            transitions = [
                i for i, flag in enumerate(flags)
                if flag != flags[(i + 1) % len(flags)]
            ]
            if not transitions:
                if all(flags):
                    return [], abs(_polygon_area(np.asarray([node[1] for node in nodes])))
                raise ValueError(
                    f"owner-overlap mixed-wall owner={face.owner}, plane={face.plane} has no wall transition"
                )
            if len(transitions) % 2:
                raise ValueError(
                    "owner-overlap mixed-wall owner="
                    f"{face.owner}, plane={face.plane} requires an even number of wall transitions; "
                    f"found {len(transitions)} transitions"
                )
            entries = [i for i in transitions if flags[i] and not flags[(i + 1) % len(flags)]]
            leaves = [i for i in transitions if not flags[i] and flags[(i + 1) % len(flags)]]
            if len(entries) != len(leaves) or not entries:
                raise ValueError(
                    "owner-overlap mixed-wall owner="
                    f"{face.owner}, plane={face.plane} has inconsistent transition orientation"
                )
            pieces: list[_MappedPiece] = []
            component_count = 0
            omitted_slivers = 0
            for enter in entries:
                # Pair this wall->interior transition with the first
                # interior->wall transition encountered cyclically.  This
                # preserves each disconnected interior run as its own polygon.
                i = (enter + 1) % len(nodes)
                interior = []
                leave = None
                while True:
                    if not flags[i]:
                        interior.append(nodes[i])
                    if i in leaves:
                        leave = i
                        break
                    i = (i + 1) % len(nodes)
                    if i == (enter + 1) % len(nodes):
                        break
                if leave is None:
                    raise ValueError(
                        "owner-overlap mixed-wall owner="
                        f"{face.owner}, plane={face.plane} has inconsistent interior component"
                    )
                # Boundary-hit endpoints are points before the target eta
                # plane, not points on the common overlap plane.  They are
                # therefore transition markers only and must never enter the
                # mapped polygon (their coordinates can be arbitrarily far
                # from the valid midpoint chart).  The first and last valid
                # midpoint samples implicitly close the retained run with a
                # straight O(h**2) transition chord.
                if len(interior) < 3:
                    omitted_slivers += 1
                    mixed_wall_diagnostics.append(
                        {
                            "owner": int(face.owner),
                            "plane": int(face.plane),
                            "direction": direction,
                            "mode": "ordered_boundary_chord",
                            "classification": "terminal-wall-sliver",
                            "component_index": int(component_count),
                            "valid_midpoint_nodes": int(len(interior)),
                            "bounded_by": "terminal O(h^2) wall transition",
                        }
                    )
                    continue
                component_points = np.asarray([node[1] for node in interior], dtype=float)
                polygon = _simplify_polygon(component_points)
                if polygon.shape[0] < 3 or abs(_polygon_area(polygon)) <= 1.0e-18:
                    omitted_slivers += 1
                    mixed_wall_diagnostics.append(
                        {
                            "owner": int(face.owner),
                            "plane": int(face.plane),
                            "direction": direction,
                            "mode": "ordered_boundary_chord",
                            "classification": "terminal-wall-sliver",
                            "component_index": int(component_count),
                            "valid_midpoint_nodes": int(len(interior)),
                            "bounded_by": "roundoff-area terminal wall transition",
                        }
                    )
                    continue
                try:
                    component_lengths = np.asarray(
                        [node[2] for node in interior], dtype=float
                    )
                    component_length_model = _fit_length_model(
                        component_points, component_lengths
                    )
                    component_pieces = []
                    for simple_boundary in _decompose_mapped_boundary(polygon):
                        simple_area = abs(_polygon_area(simple_boundary))
                        if simple_area <= 1.0e-18:
                            continue
                        component_pieces.extend(
                            _MappedPiece(
                                _as_ccw(piece),
                                float(np.mean(component_lengths)),
                                True,
                                component_length_model,
                            )
                            for piece in _triangulate_simple_polygon(simple_boundary)
                        )
                except ValueError as error:
                    raise ValueError(
                        f"owner-overlap mixed-wall owner={face.owner}, plane={face.plane} has invalid cut polygon"
                    ) from error
                if not component_pieces:
                    omitted_slivers += 1
                    mixed_wall_diagnostics.append(
                        {
                            "owner": int(face.owner),
                            "plane": int(face.plane),
                            "direction": direction,
                            "mode": "ordered_boundary_chord",
                            "classification": "terminal-wall-sliver",
                            "component_index": int(component_count),
                            "valid_midpoint_nodes": int(len(interior)),
                            "bounded_by": "roundoff-area terminal wall transition",
                        }
                    )
                    continue
                pieces.extend(component_pieces)
                component_count += 1
            if len(pieces) > int(max_mapped_pieces):
                raise MemoryError(
                    "owner-overlap mapped-piece refinement exceeded the bounded "
                    f"budget ({max_mapped_pieces})"
                )
            # Wall area is a diagnostic-only quantity; do not infer it from
            # a potentially discontinuous mapped boundary.  Zero cancels
            # from the homogeneous-Neumann interior closure accounting.
            if omitted_slivers:
                mixed_wall_diagnostics.append(
                    {
                        "owner": int(face.owner),
                        "plane": int(face.plane),
                        "direction": direction,
                        "mode": "ordered_boundary_chord",
                        "classification": "terminal-wall-slivers",
                        "omitted_components": int(omitted_slivers),
                    }
                )
            return pieces, 0.0, component_count

        for face_index, face in enumerate(input_faces):
            # A wall transition is detected from the already traced boundary
            # samples.  Only these faces enter the localized boundary-cut
            # path; the bulk of the owner graph keeps the cheap construction.
            wall_values = [
                bool(record[6])
                for edge_segments in accepted[face_index]
                for record in edge_segments
            ] + [
                bool(record[7])
                for edge_segments in accepted[face_index]
                for record in edge_segments
            ]
            mixed_wall_face = bool(wall_values) and any(wall_values) and not all(wall_values)
            if mixed_wall_face and not face.boundary:
                mapped_pieces, wall_area, component_count = map_mixed_wall_face_boundary(face_index, face)
                mapped_piece_count += len(mapped_pieces)
                if mapped_piece_count > int(max_mapped_pieces):
                    raise MemoryError(
                        "owner-overlap mapped-piece refinement exceeded the bounded "
                        f"budget ({max_mapped_pieces})"
                    )
                resource_counters["mapped_pieces"] += len(mapped_pieces)
                result.append((face, mapped_pieces, wall_area))
                mixed_wall_diagnostics.append(
                    {
                        "owner": int(face.owner),
                        "plane": int(face.plane),
                        "direction": direction,
                        "mode": "ordered_boundary_chord",
                        "component_count": int(component_count),
                        "retained_pieces": int(len(mapped_pieces)),
                        "wall_area": float(wall_area),
                    }
                )
                continue
            polygon_parts = []
            full_polygon_parts = []
            length_parts = []
            wall_parts = []
            sample_wall_parts = []
            for edge_segments in accepted[face_index]:
                edge_segments.sort(key=lambda record: record[0])
                for record in edge_segments:
                    _, _, mapped0, mapped1, length0, length1, wall0, wall1 = record
                    sample_wall_parts.extend((bool(wall0), bool(wall1)))
                    full_polygon_parts.append(mapped0)
                    if not np.array_equal(full_polygon_parts[-1], mapped1):
                        full_polygon_parts.append(mapped1)
                    if not wall0:
                        polygon_parts.append(mapped0)
                        length_parts.append(length0)
                        wall_parts.append(False)
                    if wall0 != wall1:
                        # The terminal split places the approximate wall
                        # crossing at this endpoint.  Keep it as a cut
                        # vertex, but never as an interior quadrature point.
                        transition = mapped1 if not wall0 else mapped0
                        polygon_parts.append(transition)
                        length_parts.append(length1 if not wall0 else length0)
                        wall_parts.append(True)
                    if not wall1:
                        polygon_parts.append(mapped1)
                        length_parts.append(length1)
                        wall_parts.append(False)
            polygon = _simplify_polygon(np.asarray(polygon_parts, dtype=float))
            full_polygon = _simplify_polygon(np.asarray(full_polygon_parts, dtype=float))
            if polygon.shape[0] > int(max_polygon_vertices) or full_polygon.shape[0] > int(max_polygon_vertices):
                raise ValueError(
                    "owner-overlap mapped polygon exceeded the bounded vertex "
                    f"budget ({max_polygon_vertices})"
                )
            area = abs(_polygon_area(polygon))
            full_area = abs(_polygon_area(full_polygon))
            any_wall = bool(np.any(sample_wall_parts))
            all_wall = bool(np.all(sample_wall_parts)) if sample_wall_parts else False
            if face.boundary or all_wall:
                result.append((face, [], full_area))
                continue
            if any_wall:
                # Rebuild a second boundary from non-wall edge pieces; the
                # complete image boundary is retained separately above.
                # second boundary from non-wall edge pieces; transitions are
                # closed by their O(h**2) midpoint vertices.  In the usual
                # wall-cut case this is a single simple polygon.  A zero-area
                # interior is equivalent to a fully wall-terminated face.
                interior_points = []
                interior_lengths = []
                for edge_segments in accepted[face_index]:
                    edge_segments.sort(key=lambda record: record[0])
                    for record in edge_segments:
                        _, _, mapped0, mapped1, length0, length1, wall0, wall1 = record
                        if not wall0:
                            interior_points.append(mapped0)
                            interior_lengths.append(length0)
                        if wall0 != wall1:
                            interior_points.append(mapped1 if not wall0 else mapped0)
                            interior_lengths.append(length1 if not wall0 else length0)
                        if not wall1:
                            interior_points.append(mapped1)
                            interior_lengths.append(length1)
                interior_polygon = _simplify_polygon(np.asarray(interior_points, dtype=float))
                interior_area = abs(_polygon_area(interior_polygon))
                wall_area = max(0.0, full_area - interior_area)
                if interior_area <= 1.0e-18:
                    result.append((face, [], full_area))
                    continue
                polygon = interior_polygon
                area = interior_area
                length_parts = interior_lengths
                mixed_wall_diagnostics.append(
                    {
                        "owner": int(face.owner),
                        "plane": int(face.plane),
                        "direction": direction,
                        "interior_area": float(interior_area),
                        "wall_area": float(wall_area),
                        "full_mapped_area": float(full_area),
                    }
                )
            else:
                wall_area = 0.0
            mean_length = float(np.mean(length_parts)) if length_parts else face.length
            length_model = _fit_length_model(
                np.asarray(polygon_parts, dtype=float),
                np.asarray(length_parts, dtype=float),
            ) if length_parts else None
            mapped_pieces = [
                _MappedPiece(_as_ccw(piece), mean_length, False, length_model)
                for piece in _triangulate_simple_polygon(polygon)
            ]
            resource_counters["mapped_pieces"] += len(mapped_pieces)
            mapped_piece_count += len(mapped_pieces)
            if mapped_piece_count > int(max_mapped_pieces):
                raise MemoryError(
                    "owner-overlap mapped-piece refinement exceeded its bounded "
                    f"budget ({max_mapped_pieces})"
                )
            result.append((face, mapped_pieces, wall_area))
        resource_counters["mapped_faces"] += len(input_faces)
        refinement_diagnostics.append(
            {
                "direction": direction,
                "source_eta": float(source_eta),
                "accepted_edge_segments": int(accepted_count),
                "maximum_pending_edge_segments": int(maximum_pending),
                "mixed_wall_events": int(sum(1 for item in mixed_wall_diagnostics if item.get("direction") == direction)),
            }
        )
        return result

    def quadrature_payload(
        overlap: np.ndarray,
        *,
        eta_midpoint: float,
        ell_a: float | Callable[[np.ndarray], np.ndarray],
        ell_b: float | Callable[[np.ndarray], np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return degree-2 quadrature points, weights, and link denominators.

        Clipping is performed in the regular chart, where ``dx dy =
        u du dtheta``.  The factor ``1/u`` is applied only after metric
        evaluation, exactly as in the former per-fragment implementation.
        Keeping both endpoint lengths with each point is important when a
        metric callback supplies pointwise ``ell_a``/``ell_b`` overrides.
        """

        polygon = _as_ccw(overlap)
        if polygon.shape[0] < 3:
            return (
                np.empty((0, 3), dtype=float),
                np.empty(0, dtype=float),
                np.empty(0, dtype=float),
                np.empty(0, dtype=float),
            )
        points: list[np.ndarray] = []
        weights: list[float] = []
        for index in range(1, polygon.shape[0] - 1):
            triangle = np.asarray(
                (polygon[0], polygon[index], polygon[index + 1]), dtype=float
            )
            triangle_area = abs(_polygon_area(triangle))
            if triangle_area <= 1.0e-18:
                continue
            # Degree-two symmetric triangle quadrature in Cartesian chart.
            for barycentric in (
                (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
                (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
                (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
            ):
                points.append(np.einsum("i,ij->j", barycentric, triangle))
                weights.append(triangle_area / 3.0)
        if not points:
            return (
                np.empty((0, 3), dtype=float),
                np.empty(0, dtype=float),
                np.empty(0, dtype=float),
                np.empty(0, dtype=float),
            )
        xy = np.asarray(points, dtype=float)
        def length_values(value: float | Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
            if callable(value):
                values = np.asarray(value(xy), dtype=float).reshape(-1)
                if values.size == 1:
                    values = np.full(xy.shape[0], float(values[0]), dtype=float)
            else:
                values = np.full(xy.shape[0], float(value), dtype=float)
            if values.size != xy.shape[0] or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError("owner-overlap traced lengths are nonfinite or nonpositive")
            return values
        return (
            _logical_from_xy(xy, eta_midpoint),
            np.asarray(weights, dtype=float),
            length_values(ell_a),
            length_values(ell_b),
        )

    def metric_factors(
        logical: np.ndarray,
        ell_a: np.ndarray,
        ell_b: np.ndarray,
        *,
        interface: int,
    ) -> np.ndarray:
        """Evaluate metric factors for one bounded batch of quadrature points."""

        count = int(logical.shape[0])
        J = np.ones(count, dtype=float)
        beta = np.ones(count, dtype=float)
        ell_a = np.asarray(ell_a, dtype=float).reshape(-1)
        ell_b = np.asarray(ell_b, dtype=float).reshape(-1)
        if ell_a.size != count or ell_b.size != count:
            raise ValueError("owner-overlap quadrature endpoint lengths have wrong length")
        if metric_callback is not None:
            value = _call_flexible(
                metric_callback, logical, "metric", interface, 0.0
            )
            if isinstance(value, Mapping):
                J = np.broadcast_to(
                    np.asarray(value.get("J", 1.0), dtype=float).reshape(-1),
                    J.shape,
                )
                beta = np.broadcast_to(
                    np.asarray(
                        value.get("b_eta", value.get("B_eta", 1.0)), dtype=float
                    ).reshape(-1),
                    beta.shape,
                )
                if "ell_a" in value or "ell_b" in value:
                    left = np.broadcast_to(
                        np.asarray(value.get("ell_a", 1.0), dtype=float).reshape(-1),
                        ell_a.shape,
                    )
                    right = np.broadcast_to(
                        np.asarray(value.get("ell_b", 1.0), dtype=float).reshape(-1),
                        ell_b.shape,
                    )
                    # The callback's optional overrides are absolute
                    # pointwise values; omitted endpoints retain the
                    # per-fragment lengths from the traced overlap.
                    if "ell_a" in value:
                        ell_a = left
                    if "ell_b" in value:
                        ell_b = right
            elif isinstance(value, (tuple, list)):
                if len(value) >= 1:
                    J = np.broadcast_to(np.asarray(value[0], dtype=float).reshape(-1), J.shape)
                if len(value) >= 2:
                    beta = np.broadcast_to(np.asarray(value[1], dtype=float).reshape(-1), beta.shape)
                if len(value) >= 4:
                    ell_a = np.broadcast_to(np.asarray(value[2], dtype=float).reshape(-1), ell_a.shape)
                    ell_b = np.broadcast_to(np.asarray(value[3], dtype=float).reshape(-1), ell_b.shape)
        denominator = np.maximum(ell_a + ell_b, 1.0e-30)
        # ``logical`` is (u, theta, eta); the Cartesian radius used for the
        # chart Jacobian conversion is therefore |u|, not hypot(u, theta).
        radial = np.abs(logical[:, 0])
        if (
            np.any(~np.isfinite(J))
            or np.any(~np.isfinite(beta))
            or np.any(~np.isfinite(denominator))
            or np.any(J <= 0.0)
            or np.any(radial <= 0.0)
        ):
            raise ValueError("owner-overlap metric quadrature is nonfinite or singular")
        return J * np.abs(beta) / (radial * denominator)

    # Group faces by plane and keep source/destination closure accounting in
    # the same loop that emits links, so periodic last-to-first is built once.
    by_plane = {k: [face for face in faces if face.plane == k] for k in range(shape[2])}
    # Only links on the active eta interface can share a key. Compact each
    # completed interface into NumPy arrays instead of retaining a global
    # Python dict whose per-entry overhead dominates large 64^3 builds.
    links: dict[tuple[int, int, int], list[float]] = {}
    link_key_parts: list[np.ndarray] = []
    link_measure_parts: list[np.ndarray] = []
    link_tau_parts: list[np.ndarray] = []
    compact_link_count = 0
    expected_source_nonwall = np.zeros(shape[2], dtype=float)
    expected_source_wall = np.zeros(shape[2], dtype=float)
    expected_dest_nonwall = np.zeros(shape[2], dtype=float)
    expected_dest_wall = np.zeros(shape[2], dtype=float)
    covered_source_nonwall = np.zeros(shape[2], dtype=float)
    covered_source_wall = np.zeros(shape[2], dtype=float)
    covered_dest_nonwall = np.zeros(shape[2], dtype=float)
    covered_dest_wall = np.zeros(shape[2], dtype=float)
    omitted_wall_source = np.zeros(shape[2], dtype=float)
    omitted_wall_dest = np.zeros(shape[2], dtype=float)
    max_faces_per_interface = max(
        (len(by_plane[plane]) for plane in range(shape[2])),
        default=1,
    )
    quadrature_point_cap = int(max_quadrature_points) if max_quadrature_points is not None else max(
        1, 64 * max_faces_per_interface
    )

    def flush_quadrature(
        records: list[tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        *,
        interface: int,
    ) -> None:
        """Evaluate and scatter one bounded interface quadrature batch."""

        if not records:
            return
        logical = np.concatenate([record[1] for record in records], axis=0)
        weights = np.concatenate([record[2] for record in records], axis=0)
        ell_a = np.concatenate([record[3] for record in records], axis=0)
        ell_b = np.concatenate([record[4] for record in records], axis=0)
        starts = np.cumsum(
            np.asarray([0] + [record[1].shape[0] for record in records], dtype=np.int64)
        )
        contributions = np.empty(logical.shape[0], dtype=float)
        for start in range(0, logical.shape[0], int(metric_batch_size)):
            stop = min(start + int(metric_batch_size), logical.shape[0])
            factors = metric_factors(
                logical[start:stop],
                ell_a[start:stop],
                ell_b[start:stop],
                interface=interface,
            )
            contributions[start:stop] = weights[start:stop] * factors
            resource_counters["quadrature_batches"] += 1
            resource_counters["quadrature_evaluated_points"] += int(stop - start)
            resource_counters["peak_quadrature_batch_points"] = max(
                resource_counters["peak_quadrature_batch_points"], int(stop - start)
            )
        for index, (key, _logical, _weights, _ell_a, _ell_b) in enumerate(records):
            start = int(starts[index])
            stop = int(starts[index + 1])
            links[key][1] += float(np.sum(contributions[start:stop]))

    for interface in selected_interfaces:
        links = {}
        source_plane = interface
        destination_plane = (interface + 1) % shape[2]
        source_faces = by_plane[source_plane]
        destination_faces = by_plane[destination_plane]
        eta_source, eta_destination, eta_midpoint = _interface_eta(interface)
        half_step = 0.5 * (eta_destination - eta_source)
        interface_counter_start = {
            name: int(resource_counters[name])
            for name in (
                "traced_points",
                "mapped_faces",
                "mapped_pieces",
                "overlap_candidates",
                "clip_calls",
                "quadrature_real_points",
                "quadrature_evaluated_points",
                "quadrature_batches",
            )
        }
        source_mapped = mapped_faces(
            source_faces, "forward", destination_plane, eta_source, half_step,
            trace_chunk_size=int(trace_chunk_size),
        )
        destination_mapped = mapped_faces(
            destination_faces,
            "backward",
            source_plane,
            eta_destination,
            -half_step,
            trace_chunk_size=int(trace_chunk_size),
        )
        # Explicit face boundary metadata has the same homogeneous-Neumann
        # meaning as a wall flag returned by the tracer.
        source_mapped = [
            (face, [] if face.boundary else tris, wall_area + (sum(abs(_polygon_area(piece.polygon)) for piece in tris) if face.boundary else 0.0))
            for face, tris, wall_area in source_mapped
        ]
        destination_mapped = [
            (face, [] if face.boundary else tris, wall_area + (sum(abs(_polygon_area(piece.polygon)) for piece in tris) if face.boundary else 0.0))
            for face, tris, wall_area in destination_mapped
        ]
        source_expected_by_owner: dict[int, float] = {}
        destination_expected_by_owner: dict[int, float] = {}
        source_covered_by_owner: dict[int, float] = {}
        destination_covered_by_owner: dict[int, float] = {}
        for face, tris, wall_area in source_mapped:
            omitted_wall_source[interface] += wall_area
            source_expected_by_owner[int(face.owner)] = (
                source_expected_by_owner.get(int(face.owner), 0.0) + wall_area
            )
            for piece in tris:
                area = abs(_polygon_area(piece.polygon))
                source_expected_by_owner[int(face.owner)] += area
                if piece.wall:
                    expected_source_wall[interface] += area
                else:
                    expected_source_nonwall[interface] += area
        for face, tris, wall_area in destination_mapped:
            omitted_wall_dest[interface] += wall_area
            destination_expected_by_owner[int(face.owner)] = (
                destination_expected_by_owner.get(int(face.owner), 0.0) + wall_area
            )
            for piece in tris:
                area = abs(_polygon_area(piece.polygon))
                destination_expected_by_owner[int(face.owner)] += area
                if piece.wall:
                    expected_dest_wall[interface] += area
                else:
                    expected_dest_nonwall[interface] += area
        source_pieces = [
            (face, piece, (
                float(np.min(piece.polygon[:, 0])),
                float(np.max(piece.polygon[:, 0])),
                float(np.min(piece.polygon[:, 1])),
                float(np.max(piece.polygon[:, 1])),
            ))
            for face, triangles, _ in source_mapped
            for piece in triangles
        ]
        destination_pieces = [
            (face, piece, (
                float(np.min(piece.polygon[:, 0])),
                float(np.max(piece.polygon[:, 0])),
                float(np.min(piece.polygon[:, 1])),
                float(np.max(piece.polygon[:, 1])),
            ))
            for face, triangles, _ in destination_mapped
            for piece in triangles
        ]
        resource_counters["peak_source_pieces"] = max(
            resource_counters["peak_source_pieces"], len(source_pieces)
        )
        resource_counters["peak_destination_pieces"] = max(
            resource_counters["peak_destination_pieces"], len(destination_pieces)
        )
        all_boxes = [item[2] for item in source_pieces + destination_pieces]
        if all_boxes:
            x_min = min(box[0] for box in all_boxes)
            x_max = max(box[1] for box in all_boxes)
            y_min = min(box[2] for box in all_boxes)
            y_max = max(box[3] for box in all_boxes)
        else:
            x_min = y_min = 0.0
            x_max = y_max = 1.0
        bin_count = max(
            1, min(128, int(np.ceil(np.sqrt(max(len(destination_pieces), 1)))))
        )
        x_scale = bin_count / max(x_max - x_min, 1.0e-30)
        y_scale = bin_count / max(y_max - y_min, 1.0e-30)

        def bin_interval(lower, upper, origin, scale):
            first = int(np.clip(np.floor((lower - origin) * scale), 0, bin_count - 1))
            last = int(np.clip(np.floor((upper - origin) * scale), 0, bin_count - 1))
            return range(first, last + 1)

        destination_bins: dict[tuple[int, int], list[int]] = {}
        for destination_index, (_face, _piece, box) in enumerate(destination_pieces):
            for bin_x in bin_interval(box[0], box[1], x_min, x_scale):
                for bin_y in bin_interval(box[2], box[3], y_min, y_scale):
                    destination_bins.setdefault((bin_x, bin_y), []).append(
                        destination_index
                    )
        quadrature_records: list[
            tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        quadrature_record_points = 0
        interface_quadrature_buffer_peak = 0
        for source, source_piece, source_box in source_pieces:
            candidate_indices: set[int] = set()
            for bin_x in bin_interval(
                source_box[0], source_box[1], x_min, x_scale
            ):
                for bin_y in bin_interval(
                    source_box[2], source_box[3], y_min, y_scale
                ):
                    candidate_indices.update(
                        destination_bins.get((bin_x, bin_y), ())
                    )
            resource_counters["peak_candidate_set"] = max(
                resource_counters["peak_candidate_set"], len(candidate_indices)
            )
            for destination_index in candidate_indices:
                resource_counters["overlap_candidates"] += 1
                if resource_counters["overlap_candidates"] > int(max_overlap_candidates):
                    raise MemoryError(
                        "owner-overlap intersection exceeded its bounded candidate "
                        f"budget ({max_overlap_candidates})"
                    )
                destination, destination_piece, destination_box = destination_pieces[
                    destination_index
                ]
                if (
                    destination_box[1] < source_box[0] - 1.0e-14
                    or source_box[1] < destination_box[0] - 1.0e-14
                    or source_box[3] < destination_box[2] - 1.0e-14
                    or destination_box[3] < source_box[2] - 1.0e-14
                ):
                    continue
                resource_counters["clip_calls"] += 1
                overlap = _clip_convex(
                    source_piece.polygon, destination_piece.polygon
                )
                area = abs(_polygon_area(overlap))
                source_area = abs(_polygon_area(source_piece.polygon))
                destination_area = abs(_polygon_area(destination_piece.polygon))
                roundoff_area = max(
                    1.0e-18,
                    1.0e-13 * min(source_area, destination_area),
                )
                if area <= roundoff_area:
                    continue
                source_owner = int(source.owner)
                destination_owner = int(destination.owner)
                source_covered_by_owner[source_owner] = (
                    source_covered_by_owner.get(source_owner, 0.0) + area
                )
                destination_covered_by_owner[destination_owner] = (
                    destination_covered_by_owner.get(destination_owner, 0.0) + area
                )
                if source_piece.wall:
                    covered_source_wall[interface] += area
                else:
                    covered_source_nonwall[interface] += area
                if destination_piece.wall:
                    covered_dest_wall[interface] += area
                else:
                    covered_dest_nonwall[interface] += area
                # Face owners are canonical storage-flat ids (the builder
                # converts tuple indices above); compact endpoint slots are
                # resolved only here.
                a = slot_by_id.get(int(source.owner))
                b = slot_by_id.get(int(destination.owner))
                if a is None or b is None or a == b:
                    continue
                logical, weights, ell_a, ell_b = quadrature_payload(
                    overlap,
                    eta_midpoint=eta_midpoint,
                    ell_a=source_piece.length_at,
                    ell_b=destination_piece.length_at,
                )
                key = (min(a, b), max(a, b), interface)
                links.setdefault(key, [0.0, 0.0])[0] += area
                if logical.shape[0]:
                    resource_counters["quadrature_real_points"] += int(logical.shape[0])
                    # Split large individual fragments as well as the
                    # accumulated records, keeping the temporary payload
                    # below the explicit/derived interface cap.
                    for start in range(0, logical.shape[0], quadrature_point_cap):
                        stop = min(start + quadrature_point_cap, logical.shape[0])
                        fragment = (
                            key,
                            logical[start:stop],
                            weights[start:stop],
                            ell_a[start:stop],
                            ell_b[start:stop],
                        )
                        fragment_points = int(stop - start)
                        if quadrature_record_points + fragment_points > quadrature_point_cap:
                            flush_quadrature(quadrature_records, interface=interface)
                            quadrature_records = []
                            quadrature_record_points = 0
                        quadrature_records.append(fragment)
                        quadrature_record_points += fragment_points
                        resource_counters["peak_quadrature_buffer_points"] = max(
                            resource_counters["peak_quadrature_buffer_points"],
                            quadrature_record_points,
                        )
                        interface_quadrature_buffer_peak = max(
                            interface_quadrature_buffer_peak,
                            quadrature_record_points,
                        )
        flush_quadrature(quadrature_records, interface=interface)
        resource_counters["interfaces_completed"] += 1
        interface_record = {
            "interface": int(interface),
            "source_plane": int(source_plane),
            "destination_plane": int(destination_plane),
            "source_face_count": int(len(source_faces)),
            "destination_face_count": int(len(destination_faces)),
            "source_piece_count": int(len(source_pieces)),
            "destination_piece_count": int(len(destination_pieces)),
            "canonical_link_count_cumulative": int(
                compact_link_count + len(links)
            ),
            "quadrature_buffer_peak_points": int(
                interface_quadrature_buffer_peak
            ),
            # Set before recording so cached diagnostics and callback events
            # have the same contract; dispatch itself occurs after the del
            # block below.
            "temporary_payload_released": True,
            "work": {
                name: int(resource_counters[name] - interface_counter_start[name])
                for name in interface_counter_start
            },
        }
        interface_diagnostics.append(interface_record)
        ordered_interface = sorted(links.items())
        if ordered_interface:
            link_key_parts.append(
                np.asarray([key for key, _ in ordered_interface], dtype=np.int32)
            )
            link_measure_parts.append(
                np.asarray(
                    [values[0] for _, values in ordered_interface], dtype=float
                )
            )
            link_tau_parts.append(
                np.asarray(
                    [values[1] for _, values in ordered_interface], dtype=float
                )
            )
            compact_link_count += len(ordered_interface)
        # These arrays can contain the largest temporary host payload in the
        # construction.  Release them before advancing to the next eta
        # interface; only scalar link accumulators and diagnostics survive.
        del (
            source_mapped,
            destination_mapped,
            source_pieces,
            destination_pieces,
            destination_bins,
            quadrature_records,
            all_boxes,
            links,
            ordered_interface,
        )
        if progress_callback is not None:
            progress_callback(dict(interface_record))

    expected_source = expected_source_nonwall + expected_source_wall
    expected_dest = expected_dest_nonwall + expected_dest_wall
    covered_source = covered_source_nonwall + covered_source_wall
    covered_dest = covered_dest_nonwall + covered_dest_wall
    source_over = np.maximum(covered_source - expected_source, 0.0)
    dest_over = np.maximum(covered_dest - expected_dest, 0.0)
    source_wall_over = np.maximum(covered_source_wall - expected_source_wall, 0.0)
    dest_wall_over = np.maximum(covered_dest_wall - expected_dest_wall, 0.0)
    source_under = np.maximum(expected_source - covered_source, 0.0)
    dest_under = np.maximum(expected_dest - covered_dest, 0.0)
    # A side may be short only on the common domain removed by the opposite
    # wall trace.  This permits a source interior fragment to face a
    # destination wall while still rejecting unexplained loss in a bulk map.
    opposite_dest_wall = expected_dest_wall + omitted_wall_dest
    opposite_source_wall = expected_source_wall + omitted_wall_source
    source_error = np.maximum(source_over, np.maximum(source_under - opposite_dest_wall, 0.0))
    dest_error = np.maximum(dest_over, np.maximum(dest_under - opposite_source_wall, 0.0))
    scale = np.maximum(np.maximum(expected_source, expected_dest), 1.0e-30)
    if np.any(source_error > coverage_tolerance * scale) or np.any(dest_error > coverage_tolerance * scale) or np.any(source_wall_over > coverage_tolerance * scale) or np.any(dest_wall_over > coverage_tolerance * scale):
        bad = np.flatnonzero(
            (source_error > coverage_tolerance * scale)
            | (dest_error > coverage_tolerance * scale)
            | (source_wall_over > coverage_tolerance * scale)
            | (dest_wall_over > coverage_tolerance * scale)
        )
        detail = []
        for index in bad:
            omitted_source_items = [
                item for item in mixed_wall_diagnostics
                if item.get("direction") == "forward"
                and item.get("plane") == int(index)
                and item.get("classification") in {
                    "terminal-wall-sliver", "terminal-wall-slivers"
                }
            ]
            omitted_dest_items = [
                item for item in mixed_wall_diagnostics
                if item.get("direction") == "backward"
                and item.get("plane") == int((index + 1) % shape[2])
                and item.get("classification") in {
                    "terminal-wall-sliver", "terminal-wall-slivers"
                }
            ]
            source_owner_deficits = sorted(
                (
                    {
                        "owner": int(owner),
                        "expected": float(expected),
                        "covered": float(source_covered_by_owner.get(owner, 0.0)),
                        "undercoverage": float(
                            max(expected - source_covered_by_owner.get(owner, 0.0), 0.0)
                        ),
                    }
                    for owner, expected in source_expected_by_owner.items()
                    if expected - source_covered_by_owner.get(owner, 0.0) > 1.0e-10
                ),
                key=lambda item: item["undercoverage"],
                reverse=True,
            )[:8]
            destination_owner_deficits = sorted(
                (
                    {
                        "owner": int(owner),
                        "expected": float(expected),
                        "covered": float(destination_covered_by_owner.get(owner, 0.0)),
                        "undercoverage": float(
                            max(expected - destination_covered_by_owner.get(owner, 0.0), 0.0)
                        ),
                    }
                    for owner, expected in destination_expected_by_owner.items()
                    if expected - destination_covered_by_owner.get(owner, 0.0) > 1.0e-10
                ),
                key=lambda item: item["undercoverage"],
                reverse=True,
            )[:8]
            detail.append(
                {
                    "interface": int(index),
                    "expected_source_nonwall": float(expected_source_nonwall[index]),
                    "covered_source_nonwall": float(covered_source_nonwall[index]),
                    "expected_source_wall": float(expected_source_wall[index]),
                    "covered_source_wall": float(covered_source_wall[index]),
                    "expected_destination_nonwall": float(expected_dest_nonwall[index]),
                    "covered_destination_nonwall": float(covered_dest_nonwall[index]),
                    "expected_destination_wall": float(expected_dest_wall[index]),
                    "covered_destination_wall": float(covered_dest_wall[index]),
                    "source_undercoverage": float(source_under[index]),
                    "destination_undercoverage": float(dest_under[index]),
                    "destination_wall_capacity_for_source_undercoverage": float(opposite_dest_wall[index]),
                    "source_wall_capacity_for_destination_undercoverage": float(opposite_source_wall[index]),
                    "omitted_wall_source": float(omitted_wall_source[index]),
                    "omitted_wall_destination": float(omitted_wall_dest[index]),
                    "omitted_source_sliver_owners": sorted({
                        int(item["owner"]) for item in omitted_source_items
                        if "owner" in item
                    }),
                    "omitted_destination_sliver_owners": sorted({
                        int(item["owner"]) for item in omitted_dest_items
                        if "owner" in item
                    }),
                    "largest_source_owner_undercoverage": source_owner_deficits,
                    "largest_destination_owner_undercoverage": destination_owner_deficits,
                }
            )
        raise ValueError(
            f"owner-overlap interior closure failed on interface(s) {bad.tolist()}: "
            f"source error={source_error[bad]}, destination error={dest_error[bad]}, "
            f"wall overcoverage source={source_wall_over[bad]}, destination={dest_wall_over[bad]}, "
            f"details={detail}"
        )
    wall_terminated_source = omitted_wall_source + source_under
    wall_terminated_dest = omitted_wall_dest + dest_under
    if link_key_parts:
        link_keys = np.concatenate(link_key_parts, axis=0)
        link_measure = np.concatenate(link_measure_parts, axis=0)
        link_tau = np.concatenate(link_tau_parts, axis=0)
        link_key_parts.clear()
        link_measure_parts.clear()
        link_tau_parts.clear()
        order = np.lexsort(
            (link_keys[:, 2], link_keys[:, 1], link_keys[:, 0])
        )
        link_keys = link_keys[order]
        link_measure = link_measure[order]
        link_tau = link_tau[order]
    else:
        link_keys = np.empty((0, 3), dtype=np.int32)
        link_measure = np.empty(0, dtype=float)
        link_tau = np.empty(0, dtype=float)
    return GlobalRlpParallelOverlapGeometry(
        raw_shape=shape,
        owner_flat_ids=owner_ids,
        owner_volumes=owner_volumes,
        link_owner_a=link_keys[:, 0],
        link_owner_b=link_keys[:, 1],
        link_interface=link_keys[:, 2],
        overlap_measure=link_measure,
        transmissibility=link_tau,
        metadata={
            **dict(metadata or {}),
            "raw_face_count": raw_face_count,
            "dissolved_owner_face_count": len(faces),
            "owner_face_dissolution": "coincident internal raw edges cancelled in regular x-y chart",
            "subdivision_tolerance": None if subdivision_tolerance is None else float(subdivision_tolerance),
            "second_order_tolerance_factor": float(second_order_tolerance_factor),
            "effective_tolerance_order": "O(h^2)",
            "max_edge_segments_per_interface_direction": int(max_edge_segments),
            "max_mapped_pieces": int(max_mapped_pieces),
            "max_overlap_candidates": int(max_overlap_candidates),
            "max_polygon_vertices": int(max_polygon_vertices),
            "trace_chunk_size": int(trace_chunk_size),
            "metric_batch_size": int(metric_batch_size),
            "max_quadrature_points": int(quadrature_point_cap),
            "interface_indices": list(selected_interfaces),
        },
        diagnostics={
            "source_closure_error": source_error.tolist(),
            "destination_closure_error": dest_error.tolist(),
            "source_coverage": covered_source.tolist(),
            "destination_coverage": covered_dest.tolist(),
            "source_expected_nonwall": expected_source_nonwall.tolist(),
            "destination_expected_nonwall": expected_dest_nonwall.tolist(),
            "source_expected_wall": expected_source_wall.tolist(),
            "destination_expected_wall": expected_dest_wall.tolist(),
            "source_covered_nonwall": covered_source_nonwall.tolist(),
            "destination_covered_nonwall": covered_dest_nonwall.tolist(),
            "source_covered_wall": covered_source_wall.tolist(),
            "destination_covered_wall": covered_dest_wall.tolist(),
            "wall_terminated_source": wall_terminated_source.tolist(),
            "wall_terminated_destination": wall_terminated_dest.tolist(),
            "source_undercoverage_at_destination_wall": source_under.tolist(),
            "destination_undercoverage_at_source_wall": dest_under.tolist(),
            "destination_wall_capacity_for_source_undercoverage": opposite_dest_wall.tolist(),
            "source_wall_capacity_for_destination_undercoverage": opposite_source_wall.tolist(),
            "edge_refinement": refinement_diagnostics,
            "mixed_wall": mixed_wall_diagnostics,
            "interface_work": interface_diagnostics,
            "resource_counters": resource_counters,
            "elapsed_seconds": float(time.perf_counter() - build_started),
            "interfaces_built": list(selected_interfaces),
        },
    )


def compare_rlp_parallel_overlap_refinement(
    *,
    tolerances: Iterable[float],
    builder_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a small family of graphs and compare tolerance sensitivity.

    This is intentionally an explicit diagnostic rather than an automatic
    second build in the production path.  It supports the planned
    refinement study without keeping multiple intermediate polygon meshes in
    memory: each graph is reduced to canonical link arrays before the next
    tolerance is built.
    """

    values = tuple(float(value) for value in tolerances)
    if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("tolerances must contain finite positive values")
    graphs: list[GlobalRlpParallelOverlapGeometry] = []
    summaries: list[dict[str, Any]] = []
    for tolerance in values:
        options = dict(builder_kwargs)
        options["subdivision_tolerance"] = tolerance
        graph = build_rlp_parallel_overlap_geometry(**options)
        graphs.append(graph)
        summaries.append(
            {
                "tolerance": tolerance,
                "owner_count": graph.n_owner,
                "link_count": graph.n_link,
                "resource_counters": dict(graph.diagnostics.get("resource_counters", {})),
            }
        )

    reference = graphs[-1]
    reference_keys = np.column_stack(
        (reference.link_owner_a, reference.link_owner_b, reference.link_interface)
    )
    comparisons: list[dict[str, Any]] = []
    for tolerance, graph in zip(values[:-1], graphs[:-1]):
        keys = np.column_stack((graph.link_owner_a, graph.link_owner_b, graph.link_interface))
        same_links = keys.shape == reference_keys.shape and np.array_equal(keys, reference_keys)
        if same_links and reference.transmissibility.size:
            denominator = np.maximum(np.abs(reference.transmissibility), 1.0e-30)
            relative_tau_error = float(
                np.max(np.abs(graph.transmissibility - reference.transmissibility) / denominator)
            )
        else:
            relative_tau_error = float("inf")
        comparisons.append(
            {
                "tolerance": tolerance,
                "same_canonical_links": bool(same_links),
                "relative_transmissibility_error": relative_tau_error,
            }
        )
    return {
        "reference_tolerance": values[-1],
        "summaries": summaries,
        "comparisons_to_reference": comparisons,
    }
