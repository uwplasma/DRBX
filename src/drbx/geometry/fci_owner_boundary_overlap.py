"""Trace-free construction of the lean owner-boundary overlap graph.

This module is deliberately a producer-side geometry routine.  All field-line
traces are supplied by a :class:`FciVertexTraceAtlas`; this builder only
performs owner-boundary dissolution, polygon clipping, and metric quadrature.
It therefore cannot silently change tracing accuracy or regenerate a missing
map while a simulation is being assembled.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping
import resource
import sys
import time

import numpy as np

from .fci_rlp_overlap import GlobalRlpParallelOverlapGeometry
from .fci_simulation_geometry import FciVertexTraceAtlas

__all__ = ["build_owner_boundary_overlap_geometry"]


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _regular(logical: np.ndarray) -> np.ndarray:
    p = np.asarray(logical, dtype=float)
    return np.column_stack((p[:, 0] * np.cos(p[:, 1]), p[:, 0] * np.sin(p[:, 1])))


def _area(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=float).reshape(-1, 2)
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1))) if len(p) >= 3 else 0.0


def _cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, v = np.asarray(b) - np.asarray(a), np.asarray(c) - np.asarray(a)
    return float(u[0] * v[1] - u[1] * v[0])


def _inside(p: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = 1e-14) -> bool:
    e = b - a
    return bool(e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0]) >= -eps)


def _clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    out = np.asarray(subject, dtype=float).reshape(-1, 2)
    boundary = np.asarray(clip, dtype=float).reshape(-1, 2)
    if _area(boundary) < 0:
        boundary = boundary[::-1]
    for n, a in enumerate(boundary):
        b = boundary[(n + 1) % len(boundary)]
        if not len(out):
            break
        result: list[np.ndarray] = []
        prev, prev_in = out[-1], _inside(out[-1], a, b)
        for cur in out:
            cur_in = _inside(cur, a, b)
            if cur_in != prev_in:
                edge, delta = b - a, cur - prev
                den = edge[0] * delta[1] - edge[1] * delta[0]
                if abs(den) > 1e-14:
                    t = (edge[0] * (a[1] - prev[1]) - edge[1] * (a[0] - prev[0])) / den
                    result.append(prev + t * delta)
            if cur_in:
                result.append(cur)
            prev, prev_in = cur, cur_in
        out = np.asarray(result, dtype=float).reshape(-1, 2) if result else np.empty((0, 2))
    return out


def _point_in_triangle(p: np.ndarray, tri: np.ndarray) -> bool:
    s = [_cross(tri[0], tri[1], p), _cross(tri[1], tri[2], p), _cross(tri[2], tri[0], p)]
    return min(s) >= -1e-13 or max(s) <= 1e-13


def _triangulate(poly: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    p = np.asarray(poly, dtype=float).reshape(-1, 2)
    if len(p) < 3 or abs(_area(p)) <= 1e-18:
        return []
    if _area(p) < 0:
        p = p[::-1]
    # The owner loops in the production topology are rectilinear and usually
    # convex.  Ear clipping also handles a nonconvex agglomerated owner.
    remaining = list(range(len(p)))
    triangles: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    guard = 0
    while len(remaining) > 3:
        found = False
        for pos, cur in enumerate(remaining):
            a, b = remaining[pos - 1], remaining[(pos + 1) % len(remaining)]
            tri = p[[a, cur, b]]
            if _cross(*tri) <= 1e-14:
                continue
            if any(_point_in_triangle(p[i], tri) for i in remaining if i not in (a, cur, b)):
                continue
            triangles.append((tri, (a, cur, b)))
            remaining.pop(pos)
            found = True
            break
        guard += 1
        if not found or guard > len(p) * len(p):
            raise ValueError("owner boundary is not a simple polygon")
    triangles.append((p[remaining], tuple(remaining)))
    return triangles


def _edge_key(a: tuple[int, int], b: tuple[int, int], ny: int) -> tuple[str, int, int]:
    i0, j0, i1, j1 = int(a[0]), int(a[1]) % ny, int(b[0]), int(b[1]) % ny
    if i0 != i1:
        if j0 != j1:
            raise ValueError("owner topology has a non-adjacent radial edge")
        return ("r", min(i0, i1), j0)
    if (j0 + 1) % ny == j1:
        return ("a", i0, j0)
    if (j1 + 1) % ny == j0:
        return ("a", i0, j1)
    raise ValueError("owner topology has a non-adjacent angular edge")


def _edge_loops(cells: list[tuple[int, int]], ny: int, collapse_axis: bool) -> list[list[tuple[int, int]]]:
    edges: dict[tuple[tuple[int, int], tuple[int, int]], None] = {}
    def canon(v: tuple[int, int]) -> tuple[int, int]:
        return (0, 0) if collapse_axis and v[0] == 0 else v
    for i, j in cells:
        vs = tuple(canon(v) for v in ((i, j % ny), (i + 1, j % ny), (i + 1, (j + 1) % ny), (i, (j + 1) % ny)))
        for a, b in zip(vs, vs[1:] + vs[:1]):
            if a == b:
                continue
            if (b, a) in edges:
                del edges[(b, a)]
            else:
                edges[(a, b)] = None
    loops = []
    while edges:
        key = next(iter(edges))
        del edges[key]
        loop = [key[0], key[1]]
        while loop[-1] != loop[0]:
            candidates = [k for k in edges if k[0] == loop[-1]]
            if len(candidates) != 1:
                raise ValueError("owner boundary is disconnected or branched")
            key = candidates[0]
            del edges[key]
            loop.append(key[1])
            if len(loop) > len(cells) * 8 + 8:
                raise ValueError("owner boundary loop is invalid")
        loops.append(loop[:-1])
    return loops


def _barycentric(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    den = _cross(triangle[0], triangle[1], triangle[2])
    if abs(den) <= 1e-30:
        raise ValueError("degenerate source triangle")
    p = np.asarray(point)
    w = np.asarray((_cross(p, triangle[1], triangle[2]), _cross(triangle[0], p, triangle[2]), _cross(triangle[0], triangle[1], p))) / den
    if np.min(w) < -1e-8 or not np.all(np.isfinite(w)):
        raise ValueError("overlap centroid is outside source triangle")
    w = np.maximum(w, 0.0)
    return w / max(float(np.sum(w)), 1e-30)


def _rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0**3 if sys.platform == "darwin" else value / 1024.0**2


def _metric_values(callback: Any, logical: np.ndarray, interface: int) -> np.ndarray:
    """Evaluate a metric callback with either vector or scalar conventions."""
    points = np.asarray(logical, dtype=float).reshape(-1, 3)
    if callback is None:
        return np.ones(len(points), dtype=float)
    try:
        values = callback(points, interface=interface)
    except TypeError:
        try:
            values = callback(points)
        except (TypeError, ValueError):
            values = np.asarray([callback(point, interface=interface) for point in points])
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 1:
        values = np.full(len(points), float(values[0]))
    if values.size != len(points) or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("metric callback must return one finite positive value per point")
    return values


def _raw_faces(geometry: Any, owner_geometry: Any, shape: tuple[int, int, int]) -> list[dict[str, Any]]:
    grid = _get(geometry, "grid")
    xf = np.asarray(_get(_get(grid, "x"), "faces"), dtype=float)
    yf = np.asarray(_get(_get(grid, "y"), "faces"), dtype=float)
    top = _get(owner_geometry, "topology", default=owner_geometry)
    owners = np.asarray(_get(top, "aggregate_id", default=_get(owner_geometry, "owner_map")), dtype=int)
    if owners.shape != shape or xf.size != shape[0] + 1 or yf.size != shape[1] + 1:
        raise ValueError("owner map/grid shape mismatch")
    out: list[dict[str, Any]] = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                owner = int(owners[i, j, k])
                if owner >= 0:
                    out.append({"owner": owner, "index": (i, j, k), "plane": k, "vertices": np.asarray(((xf[i], yf[j]), (xf[i + 1], yf[j]), (xf[i + 1], yf[j + 1]), (xf[i], yf[j + 1])), dtype=float)})
    return out


def _raw_faces_for_plane(
    geometry: Any,
    owner_geometry: Any,
    shape: tuple[int, int, int],
    plane: int,
) -> list[dict[str, Any]]:
    """Materialize only one eta plane of raw owner faces."""

    grid = _get(geometry, "grid")
    xf = np.asarray(_get(_get(grid, "x"), "faces"), dtype=float)
    yf = np.asarray(_get(_get(grid, "y"), "faces"), dtype=float)
    top = _get(owner_geometry, "topology", default=owner_geometry)
    owners = np.asarray(
        _get(top, "aggregate_id", default=_get(owner_geometry, "owner_map")),
        dtype=int,
    )
    if owners.shape != shape or xf.size != shape[0] + 1 or yf.size != shape[1] + 1:
        raise ValueError("owner map/grid shape mismatch")
    k = int(plane) % shape[2]
    result: list[dict[str, Any]] = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            owner = int(owners[i, j, k])
            if owner >= 0:
                result.append(
                    {
                        "owner": owner,
                        "index": (i, j, k),
                        "plane": k,
                        "vertices": np.asarray(
                            (
                                (xf[i], yf[j]),
                                (xf[i + 1], yf[j]),
                                (xf[i + 1], yf[j + 1]),
                                (xf[i], yf[j + 1]),
                            ),
                            dtype=float,
                        ),
                    }
                )
    return result


def _atlas_values(atlas: Any, direction: str, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    endpoint = np.asarray(_get(atlas, f"{direction}_endpoint", default=_get(atlas, f"{direction}_vertices")), dtype=float)
    length = np.asarray(_get(atlas, f"{direction}_length"), dtype=float)
    wall = np.asarray(_get(atlas, f"{direction}_boundary"), dtype=bool)
    expected = (shape[0] + 1, shape[1], shape[2])
    if endpoint.shape != expected + (3,) or length.shape != expected or wall.shape != expected:
        raise ValueError(f"{direction} vertex atlas must have shapes {expected + (3,)}, {expected}, {expected}")
    if (
        np.any(~np.isfinite(endpoint))
        or np.any(~np.isfinite(length))
        or np.any(length < 0.0)
        or np.any((length <= 0.0) & ~wall)
    ):
        raise ValueError("vertex atlas contains invalid trace data")
    return endpoint, length, wall


def build_owner_boundary_overlap_geometry(
    geometry: Any,
    owner_geometry: Any,
    vertex_traces: FciVertexTraceAtlas,
    *,
    cell_center_wall_masks: tuple[np.ndarray, np.ndarray],
    metric_callback: Any = None,
    raw_faces: Iterable[Any] | None = None,
    interface_indices: Iterable[int] | None = None,
    coverage_tolerance: float = 1e-10,
    metric_batch_size: int = 32768,
    progress_callback: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> GlobalRlpParallelOverlapGeometry:
    """Build the current lean owner-boundary graph from pretraced vertices.

    The absence of a tracer argument is intentional and is part of the API:
    generation and qualification own tracing, while this routine only consumes
    the resulting atlas.  ``coverage_tolerance`` is a diagnostic reference
    threshold, not a construction gate: closure excursions are recorded in
    the returned graph diagnostics without changing or rejecting the graph.
    """
    if not isinstance(vertex_traces, FciVertexTraceAtlas) and not all(hasattr(vertex_traces, n) for n in ("forward_endpoint", "backward_endpoint", "forward_length", "backward_length", "forward_boundary", "backward_boundary")):
        raise TypeError("vertex_traces must be an FciVertexTraceAtlas or compatible atlas")
    shape = tuple(int(v) for v in _get(geometry, "shape", default=_get(_get(owner_geometry, "topology", default=owner_geometry), "shape")))
    if len(shape) != 3 or coverage_tolerance <= 0 or metric_batch_size < 1:
        raise ValueError("invalid owner-boundary controls")
    fwd_v, fwd_l, fwd_w = _atlas_values(vertex_traces, "forward", shape)
    back_v, back_l, back_w = _atlas_values(vertex_traces, "backward", shape)
    if len(cell_center_wall_masks) != 2:
        raise ValueError("cell_center_wall_masks must contain forward and backward masks")
    center_walls = {
        direction: np.asarray(mask, dtype=bool)
        for direction, mask in zip(
            ("forward", "backward"), cell_center_wall_masks, strict=True
        )
    }
    if any(mask.shape != shape for mask in center_walls.values()):
        raise ValueError(f"cell-center wall masks must have shape {shape}")
    top = _get(owner_geometry, "topology", default=owner_geometry)
    owner_map = np.asarray(_get(top, "aggregate_id", default=_get(owner_geometry, "owner_map")), dtype=int)
    if owner_map.shape != shape:
        raise ValueError("owner map/grid shape mismatch")
    supplied_faces = None if raw_faces is None else list(raw_faces)
    if supplied_faces is None:
        raw_face_count = int(np.count_nonzero(owner_map >= 0))

        def faces_for_plane(plane: int) -> list[dict[str, Any]]:
            return _raw_faces_for_plane(geometry, owner_geometry, shape, plane)
    else:
        raw_face_count = len(supplied_faces)
        supplied_by_plane = {
            k: [
                face
                for face in supplied_faces
                if int(face.get("plane", face.get("index", (0, 0, 0))[2]))
                % shape[2]
                == k
            ]
            for k in range(shape[2])
        }

        def faces_for_plane(plane: int) -> list[dict[str, Any]]:
            return supplied_by_plane[int(plane) % shape[2]]

    owner_ids = np.unique(owner_map[owner_map >= 0]).astype(np.int64)
    volume = np.asarray(_get(owner_geometry, "aggregate_chart_volume", default=np.ones(shape)), dtype=float)
    volumes = volume.reshape(-1)[owner_ids] if volume.shape == shape else volume.reshape(-1)[: len(owner_ids)]
    slot = {int(v): i for i, v in enumerate(owner_ids)}
    grid = _get(geometry, "grid")
    xfaces = np.asarray(_get(_get(grid, "x"), "faces"), dtype=float); yfaces = np.asarray(_get(_get(grid, "y"), "faces"), dtype=float)
    eta = np.asarray(_get(_get(grid, "z"), "centers", default=np.arange(shape[2])), dtype=float).reshape(-1)
    selected = tuple(range(shape[2])) if interface_indices is None else tuple(dict.fromkeys(int(v) for v in interface_indices))
    if not selected or any(index < 0 or index >= shape[2] for index in selected):
        raise ValueError("invalid interface_indices")
    links: dict[tuple[int, int, int], list[float]] = {}
    counters = {
        "raw_faces": raw_face_count,
        "active_source_cells": 0,
        "terminated_source_measure": 0.0,
        "atlas_vertices_available": int(np.prod(fwd_v.shape[:-1])),
        "lean_boundary_vertices": 0,
        "dissolved_owner_face_count": 0,
        "mapped_triangles": 0,
        "bin_candidates": 0,
        "bbox_rejections": 0,
        "overlap_candidates": 0,
        "quadrature_points": 0,
        "metric_batches": 0,
        "rows": 0,
    }
    peak_rss = _rss_gib()
    build_started = time.monotonic()

    def sample_peak_rss() -> None:
        nonlocal peak_rss
        peak_rss = max(peak_rss, _rss_gib())

    diagnostics = {
        "interfaces": [],
        "construction_mode": "lean_owner_boundary_trace_free",
        "trace_substeps": int(
            _get(vertex_traces, "metadata", default={}).get("trace_substeps", 64)
            if isinstance(_get(vertex_traces, "metadata", default={}), Mapping)
            else 64
        ),
        "closure_reference_tolerance": float(coverage_tolerance),
        "resource_counters": counters,
    }
    for interface in selected:
        interface_started = time.monotonic()
        interface_start_counts = {
            name: int(counters[name])
            for name in (
                "bin_candidates",
                "bbox_rejections",
                "overlap_candidates",
                "quadrature_points",
            )
        }
        kp = (interface + 1) % shape[2]
        interface_faces = {
            interface: faces_for_plane(interface),
            kp: faces_for_plane(kp),
        }
        directed = []
        for direction, source_plane, target_plane, vv, ll, ww in (("forward", interface, kp, fwd_v, fwd_l, fwd_w), ("backward", kp, interface, back_v, back_l, back_w)):
            sample_peak_rss()
            interface_max_closure = 0.0
            interface_max_relative_closure = 0.0
            interface_closure_sum = 0.0
            interface_closure_weight = 0.0
            interface_closure_exceedances = 0
            terminated_measure = 0.0
            source_faces = interface_faces[source_plane]
            target_faces = interface_faces[target_plane]
            active = []
            for f in source_faces:
                index = tuple(int(v) for v in f.get("index", (0, 0, source_plane)))
                i, j, _ = index
                corners = ((i, j % shape[1]), (i + 1, j % shape[1]), (i + 1, (j + 1) % shape[1]), (i, (j + 1) % shape[1]))
                hit = bool(center_walls[direction][index]) or any(bool(ww[iv, jv % shape[1], source_plane]) for iv, jv in corners if not (abs(xfaces[0]) <= 1e-12 and iv == 0))
                if not hit:
                    active.append(f)
                else:
                    terminated_measure += abs(
                        _area(_regular(np.asarray(f["vertices"], dtype=float)))
                    )
            counters["active_source_cells"] += len(active)
            counters["terminated_source_measure"] += terminated_measure
            owner_cells: dict[int, list[tuple[int, int]]] = {}
            for f in active:
                owner = int(f["owner"]); i, j = tuple(f.get("index", (0, 0, source_plane)))[:2]
                owner_cells.setdefault(owner, []).append((int(i), int(j)))
            collapse_axis = abs(xfaces[0]) <= 1e-12
            source_triangles: list[tuple[int, np.ndarray, np.ndarray]] = []
            for owner, cells in owner_cells.items():
                loops = _edge_loops(cells, shape[1], collapse_axis)
                if len(loops) != 1:
                    raise ValueError(f"owner {owner} has multiple/disconnected boundary loops")
                loop = loops[0]
                counters["dissolved_owner_face_count"] += 1
                counters["lean_boundary_vertices"] += sum(
                    not (collapse_axis and iv == 0) for iv, _jv in loop
                )
                points = []; q = []
                for iv, jv in loop:
                    if collapse_axis and iv == 0:
                        continue
                    e = vv[iv, jv % shape[1], source_plane]
                    coordinates = e[:2] if str(_get(vertex_traces, "metadata", default={}).get("endpoint_coordinates", "logical")).lower() in ("physical", "cartesian") else _regular(np.asarray(e[None, :2]))[0]
                    points.append(coordinates); q.append(1.0 / ll[iv, jv % shape[1], source_plane])
                point_array = np.asarray(points)
                q_array = np.asarray(q, dtype=float)
                if _area(point_array) < 0.0:
                    point_array, q_array = point_array[::-1], q_array[::-1]
                for tri, indices in _triangulate(point_array):
                    counters["mapped_triangles"] += 1
                    source_triangles.append((owner, tri, np.asarray([q_array[i] for i in indices], dtype=float)))
            target_triangles: list[tuple[int, np.ndarray]] = []
            target_cells: dict[int, list[tuple[int, int]]] = {}
            for f in target_faces:
                owner = int(f["owner"]); i, j = tuple(f.get("index", (0, 0, target_plane)))[:2]
                target_cells.setdefault(owner, []).append((int(i), int(j)))
            for owner, cells in target_cells.items():
                loops = _edge_loops(cells, shape[1], collapse_axis)
                if len(loops) != 1:
                    raise ValueError("destination owner has multiple/disconnected boundary loops")
                points = np.asarray([_regular(np.asarray([[xfaces[i], yfaces[j % shape[1]]]], dtype=float))[0] for i, j in loops[0] if not (collapse_axis and i == 0)])
                target_triangles.extend((owner, tri) for tri, _indices in _triangulate(points))
            if not source_triangles or not target_triangles:
                directed.append({})
                diagnostics["interfaces"].append(
                    {
                        "interface": int(interface),
                        "direction": direction,
                        "active_cells": len(active),
                        "terminated_measure": terminated_measure,
                        "max_closure_error": 0.0,
                        "max_relative_closure_error": 0.0,
                        "volume_weighted_closure_error": 0.0,
                        "closure_reference_tolerance": float(coverage_tolerance),
                        "closure_reference_exceedance_count": 0,
                        "closure_within_reference_tolerance": True,
                        "conductance": 0.0,
                    }
                )
                continue
            counters["mapped_triangles"] += len(target_triangles)
            # Spatial bins retain the authoritative bounded candidate search;
            # only triangles whose boxes share a bin reach polygon clipping.
            boxes = [(tri[:, 0].min(), tri[:, 0].max(), tri[:, 1].min(), tri[:, 1].max()) for _owner, tri in target_triangles]
            bounds = np.concatenate([tri for _owner, tri in target_triangles])
            lo, hi = bounds.min(axis=0), bounds.max(axis=0)
            scale = max(float(np.max(hi - lo)), 1.0e-30)
            nb = max(1, int(np.ceil(np.sqrt(len(target_triangles)))))
            bbox_epsilon = 64.0 * np.finfo(np.float64).eps * scale
            bins: dict[tuple[int, int], list[int]] = {}

            def box_bins(box: tuple[float, float, float, float]):
                i0 = int(np.clip(np.floor((box[0] - bbox_epsilon - lo[0]) / scale * nb), 0, nb - 1))
                i1 = int(np.clip(np.floor((box[1] + bbox_epsilon - lo[0]) / scale * nb), 0, nb - 1))
                j0 = int(np.clip(np.floor((box[2] - bbox_epsilon - lo[1]) / scale * nb), 0, nb - 1))
                j1 = int(np.clip(np.floor((box[3] + bbox_epsilon - lo[1]) / scale * nb), 0, nb - 1))
                return ((ii, jj) for ii in range(i0, i1 + 1) for jj in range(j0, j1 + 1))

            for index, box in enumerate(boxes):
                for cell in box_bins(box):
                    bins.setdefault(cell, []).append(index)

            def candidates(box: tuple[float, float, float, float]) -> set[int]:
                return {index for cell in box_bins(box) for index in bins.get(cell, ())}

            out: dict[tuple[int, int, int], list[float]] = {}
            pending: list[tuple[tuple[int, int, int], float, np.ndarray, float]] = []

            def flush() -> None:
                if not pending:
                    return
                logical = np.asarray([[np.hypot(point[0], point[1]), np.arctan2(point[1], point[0]), eta[target_plane]] for _key, _area_value, point, _q_value in pending])
                metric = _metric_values(metric_callback, logical, interface)
                for (key, area_value, point, q_value), metric_value in zip(pending, metric, strict=True):
                    radial = float(np.hypot(point[0], point[1]))
                    rec = out.setdefault(key, [0.0, 0.0])
                    rec[0] += area_value
                    rec[1] += area_value * float(metric_value) * q_value / max(radial, 1e-30)
                pending.clear()
                counters["metric_batches"] += 1
            for owner, tri, q in source_triangles:
                source_area = abs(_area(tri)); covered = 0.0
                source_box = (tri[:, 0].min(), tri[:, 0].max(), tri[:, 1].min(), tri[:, 1].max())
                for target_index in candidates(source_box):
                    counters["bin_candidates"] += 1
                    if counters["bin_candidates"] % 4096 == 0:
                        sample_peak_rss()
                    target_owner, target = target_triangles[target_index]
                    target_box = boxes[target_index]
                    if target_box[1] < source_box[0] - bbox_epsilon or target_box[0] > source_box[1] + bbox_epsilon or target_box[3] < source_box[2] - bbox_epsilon or target_box[2] > source_box[3] + bbox_epsilon:
                        counters["bbox_rejections"] += 1
                        continue
                    counters["overlap_candidates"] += 1
                    overlap = _clip(tri, target); area = abs(_area(overlap))
                    if area <= max(1e-18, 1e-12 * source_area):
                        continue
                    covered += area
                    if target_owner == owner:
                        continue
                    for n in range(1, len(overlap) - 1):
                        piece = overlap[[0, n, n + 1]]; piece_area = abs(_area(piece))
                        if piece_area <= 1e-18:
                            continue
                        centroid = np.mean(piece, axis=0); qv = float(np.dot(_barycentric(centroid, tri), q))
                        key = (min(slot[owner], slot[target_owner]), max(slot[owner], slot[target_owner]), interface)
                        pending.append((key, piece_area, centroid, qv))
                        counters["quadrature_points"] += 1
                        if len(pending) >= max(1, int(metric_batch_size)):
                            flush()
                closure = abs(covered - source_area)
                relative_closure = closure / max(source_area, 1e-30)
                interface_max_closure = max(interface_max_closure, closure)
                interface_max_relative_closure = max(
                    interface_max_relative_closure, relative_closure
                )
                physical_weight = source_area * float(volumes[slot[owner]])
                interface_closure_sum += closure * physical_weight
                interface_closure_weight += physical_weight
                if relative_closure > coverage_tolerance:
                    interface_closure_exceedances += 1
            flush()
            directed.append(out)
            diagnostics["interfaces"].append(
                {
                    "interface": int(interface),
                    "direction": direction,
                    "active_cells": len(active),
                    "terminated_measure": terminated_measure,
                    "max_closure_error": interface_max_closure,
                    "max_relative_closure_error": interface_max_relative_closure,
                    "volume_weighted_closure_error": interface_closure_sum
                    / max(interface_closure_weight, 1e-30),
                    "closure_reference_tolerance": float(coverage_tolerance),
                    "closure_reference_exceedance_count": int(
                        interface_closure_exceedances
                    ),
                    "closure_within_reference_tolerance": bool(
                        interface_closure_exceedances == 0
                    ),
                    "conductance": float(sum(v[1] for v in out.values())),
                }
            )
        for key in set(directed[0]) | set(directed[1]):
            f, b = directed[0].get(key, [0.0, 0.0]), directed[1].get(key, [0.0, 0.0]); links[key] = [0.5 * (f[0] + b[0]), 0.5 * (f[1] + b[1])]
        records = [
            record
            for record in diagnostics["interfaces"]
            if record["interface"] == int(interface)
        ]
        forward_conductance = next(
            (float(record["conductance"]) for record in records if record["direction"] == "forward"),
            0.0,
        )
        backward_conductance = next(
            (float(record["conductance"]) for record in records if record["direction"] == "backward"),
            0.0,
        )
        mismatch = abs(forward_conductance - backward_conductance) / max(
            abs(forward_conductance), abs(backward_conductance), 1e-30
        )
        for record in records:
            record["forward_backward_conductance_mismatch"] = mismatch
        sample_peak_rss()
        if progress_callback is not None:
            progress_callback(
                {
                    "interface": int(interface),
                    "completed_interfaces": selected.index(interface) + 1,
                    "total_interfaces": len(selected),
                    "elapsed_seconds": time.monotonic() - build_started,
                    "interface_elapsed_seconds": time.monotonic() - interface_started,
                    "peak_rss_gib": peak_rss,
                    "resource_counters": {
                        name: int(counters[name]) - interface_start_counts[name]
                        for name in interface_start_counts
                    },
                }
            )
    ordered = sorted(links)
    counters["rows"] = len(ordered)
    diagnostics["elapsed_seconds"] = time.monotonic() - build_started
    diagnostics["peak_rss_gib"] = peak_rss
    diagnostics["max_closure_error"] = float(
        max(
            (record["max_closure_error"] for record in diagnostics["interfaces"]),
            default=0.0,
        )
    )
    diagnostics["max_relative_closure_error"] = float(
        max(
            (
                record["max_relative_closure_error"]
                for record in diagnostics["interfaces"]
            ),
            default=0.0,
        )
    )
    diagnostics["closure_reference_exceedance_count"] = int(
        sum(
            record["closure_reference_exceedance_count"]
            for record in diagnostics["interfaces"]
        )
    )
    diagnostics["closure_within_reference_tolerance"] = bool(
        diagnostics["closure_reference_exceedance_count"] == 0
    )
    diagnostics["streaming_controls"] = {
        "metric_batch_size": int(metric_batch_size),
    }
    return GlobalRlpParallelOverlapGeometry(raw_shape=shape, owner_flat_ids=owner_ids, owner_volumes=volumes, link_owner_a=np.asarray([k[0] for k in ordered], dtype=np.int32), link_owner_b=np.asarray([k[1] for k in ordered], dtype=np.int32), link_interface=np.asarray([k[2] for k in ordered], dtype=np.int32), overlap_measure=np.asarray([links[k][0] for k in ordered]), transmissibility=np.asarray([links[k][1] for k in ordered]), metadata={"construction": "owner-boundary-lean-second-order", "trace_free": True, **dict(metadata or {})}, diagnostics=diagnostics)
