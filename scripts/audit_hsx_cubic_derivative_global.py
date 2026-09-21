#!/usr/bin/env python3
"""Build the fixed balanced-cubic owner-to-face derivative on global HSX grids.

The numerical policy is the ``balanced_cubic`` policy qualified on bounded
N48/N64 samples by :mod:`audit_hsx_owner_face_derivative_adaptive`.  This file
only changes how its rows are constructed and cached.  Raw-volume owner
moments through degree three are accumulated once, donor queries are indexed
per eta plane, and the full-rank weighted least-norm solve is evaluated in
bounded batches through its 20 by 20 Gram system.

The global cache intentionally stores fixed-field derivative *actions* and
complete row diagnostics rather than hundreds of millions of sparse weights.
Every chunk records a hash of the selected donors and double-precision row
weights, while stratified rows retain their explicit donors/weights for replay.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import audit_hsx_owner_face_derivative as base  # noqa: E402
import audit_hsx_owner_face_derivative_adaptive as adaptive  # noqa: E402
import audit_hsx_poisson_common_face_oracle as oracle  # noqa: E402


SCHEMA = "drbx.hsx-balanced-cubic-global-derivative-v1"
CHUNK_SCHEMA = "drbx.hsx-balanced-cubic-global-derivative-chunk-v1"
PROFILE_SCHEMA = "drbx.hsx-balanced-cubic-global-profile-v1"
EQUIVALENCE_SCHEMA = "drbx.hsx-balanced-cubic-global-equivalence-v1"
FIELD_NAMES = base.FIELD_NAMES
EXPONENTS = adaptive.DEGREE3_EXPONENTS
POLICY = {
    **base._row_policy(
        (24, 32),
        support_mode="degree3_balanced_xy_sectors_nearest32_eta5_6_coverage_fallback",
    ),
    "degree": 3,
    "basis": list(adaptive.DEGREE3_NAMES),
    "candidate_pool_per_plane": 32,
    "planar_sectors": 8,
    "base_donors_per_plane": 24,
    "fallback_donors_per_plane": 32,
    "deficient_row_expansion_schedule": [
        {"candidate_pool_per_plane": 32, "donors_per_plane": 24},
        {"candidate_pool_per_plane": 32, "donors_per_plane": 32},
        {"candidate_pool_per_plane": 48, "donors_per_plane": 40},
        {"candidate_pool_per_plane": 64, "donors_per_plane": 48},
    ],
    "minimum_distinct_angular_columns_per_plane": 3,
    "radial_angular_planes": 5,
    "eta_face_planes": 6,
    "solver": "full-rank weighted least-norm via equivalent 20x20 Gram solve",
    "cache_representation": "fixed-field chunked derivative actions plus row hashes and stratified explicit rows",
}
BOUNDARY_CONTRACT = {
    "physical_wall_model": "legacy-velocity-trace",
    "parallel_velocity_wall_bc": "neumann",
    "neumann_ghost_scheme": "physical",
    "parallel_boundary_pairing": "characteristic-sat",
    "parallel_characteristic_wall_law": "energy-absorbing",
    "eta": "periodic",
}
MOMENT_EXPONENTS = tuple(
    (a, b) for total in range(4) for a in range(total + 1) for b in (total - a,)
)
MOMENT_SLOT = {exponent: index for index, exponent in enumerate(MOMENT_EXPONENTS)}


def _max_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / 1024.0**3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(oracle._json_value(payload), sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(oracle._json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _progress(stage: str, **details: Any) -> None:
    print(json.dumps({"event": "cubic_global", "stage": stage, **details}, sort_keys=True), flush=True)


def _policy_hash() -> str:
    return hashlib.sha256(_canonical(POLICY).encode()).hexdigest()


def _source_identity(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _implementation_identity() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        SCRIPTS / "audit_hsx_owner_face_derivative.py",
        SCRIPTS / "audit_hsx_owner_face_derivative_adaptive.py",
    )
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def _manifest_identity(
    geometry: Path, baseline: Path, resolution: int, requested_checks: Sequence[str]
) -> dict[str, Any]:
    manifest, base_geometry, topology = base._geometry_paths(geometry, resolution)
    reference = baseline / f"N{resolution}.reference.npz"
    return {
        "schema": SCHEMA,
        "resolution": int(resolution),
        "geometry": {
            "manifest": _source_identity(manifest),
            "base_geometry": _source_identity(base_geometry),
            "topology": _source_identity(topology),
        },
        "field_data": {"baseline_reference": _source_identity(reference)},
        "implementation_sha256": _implementation_identity(),
        "policy": POLICY,
        "policy_sha256": _policy_hash(),
        "fields": list(FIELD_NAMES),
        "boundary_contract": BOUNDARY_CONTRACT,
        "requested_checks": list(requested_checks),
        "precision": "float64",
    }


@dataclass
class BuildContext:
    resolution: int
    arrays: dict[str, np.ndarray]
    owner_moments: np.ndarray
    owner_eta: np.ndarray
    owner_planar_id: np.ndarray
    owner_j: np.ndarray
    plane_indices: tuple[np.ndarray, ...]
    plane_trees: tuple[cKDTree, ...]
    x_centers: np.ndarray
    y_centers: np.ndarray
    z_centers: np.ndarray
    x_faces: np.ndarray
    y_faces: np.ndarray
    z_faces: np.ndarray
    dr: float
    dtheta: float
    deta: float
    eta_period: float


def _owner_moments(arrays: Mapping[str, np.ndarray], resolution: int) -> tuple[np.ndarray, np.ndarray]:
    owner_flat = arrays["owner_flat_ids"]
    compact = np.full(resolution**3, -1, dtype=np.int64)
    compact[owner_flat] = np.arange(len(owner_flat))
    raw_owner = compact[arrays["aggregate_id"]]
    if np.any(raw_owner < 0):
        raise ValueError("raw topology contains a member without an active owner")
    volume = np.bincount(
        raw_owner, weights=arrays["raw_volume"], minlength=len(owner_flat)
    )
    moments = np.empty((len(owner_flat), len(MOMENT_EXPONENTS)), dtype=np.float64)
    for slot, (a, b) in enumerate(MOMENT_EXPONENTS):
        weighted = arrays["raw_volume"] * arrays["raw_x"]**a * arrays["raw_y"]**b
        moments[:, slot] = np.bincount(
            raw_owner, weights=weighted, minlength=len(owner_flat)
        ) / volume
    eta_weighted = np.bincount(
        raw_owner,
        weights=arrays["raw_volume"] * arrays["raw_eta"],
        minlength=len(owner_flat),
    )
    return moments, eta_weighted / volume


def _load_context(geometry: Path, baseline: Path, resolution: int) -> BuildContext:
    arrays, metadata = base._load_owner_inputs(geometry, baseline, resolution)
    arrays.pop("raw_by_owner_object", None)
    if metadata["volume_relative_mismatch"] > 1.0e-12:
        raise ValueError("stored owner observation volume does not match topology")
    _manifest, base_path, _topology = base._geometry_paths(geometry, resolution)
    with np.load(base_path, allow_pickle=False) as source:
        x_centers = np.asarray(source["grid.x.centers"], dtype=np.float64)
        y_centers = np.asarray(source["grid.y.centers"], dtype=np.float64)
        z_centers = np.asarray(source["grid.z.centers"], dtype=np.float64)
    x_faces = arrays["grid_x_faces"]
    y_faces = arrays["grid_y_faces"]
    z_faces = arrays["grid_z_faces"]
    moments, owner_eta = _owner_moments(arrays, resolution)
    owner_planar_id = arrays["owner_flat_ids"] // resolution
    owner_j = owner_planar_id % resolution
    plane_indices = tuple(
        np.flatnonzero(arrays["owner_plane"] == plane) for plane in range(resolution)
    )
    plane_trees = tuple(cKDTree(arrays["owner_centroid_xy"][indices]) for indices in plane_indices)
    return BuildContext(
        resolution=resolution,
        arrays=arrays,
        owner_moments=moments,
        owner_eta=owner_eta,
        owner_planar_id=owner_planar_id,
        owner_j=owner_j,
        plane_indices=plane_indices,
        plane_trees=plane_trees,
        x_centers=x_centers,
        y_centers=y_centers,
        z_centers=z_centers,
        x_faces=x_faces,
        y_faces=y_faces,
        z_faces=z_faces,
        dr=float(np.median(np.diff(x_faces))),
        dtheta=float(np.median(np.diff(y_faces))),
        deta=float(np.median(np.diff(z_faces))),
        eta_period=float(z_faces[-1] - z_faces[0]),
    )


def _planar_faces(context: BuildContext, axis: int, eta_index: int) -> tuple[np.ndarray, np.ndarray]:
    n = context.resolution
    if axis == 0:
        ii, jj = np.meshgrid(np.arange(n + 1), np.arange(n), indexing="ij")
        u = context.x_faces[ii]
        theta = context.y_centers[jj]
        eta = np.full_like(u, context.z_centers[eta_index], dtype=np.float64)
    elif axis == 1:
        ii, jj = np.meshgrid(np.arange(n), np.arange(n + 1), indexing="ij")
        u = context.x_centers[ii]
        theta = context.y_faces[jj]
        eta = np.full_like(u, context.z_centers[eta_index], dtype=np.float64)
    elif axis == 2:
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        u = context.x_centers[ii]
        theta = context.y_centers[jj]
        eta = np.full_like(u, context.z_faces[eta_index], dtype=np.float64)
    else:
        raise ValueError(f"invalid face axis {axis}")
    keys = np.column_stack((ii.reshape(-1), jj.reshape(-1)))
    points = np.column_stack((u.reshape(-1), theta.reshape(-1), eta.reshape(-1)))
    return keys.astype(np.int32), points


def _planes(axis: int, eta_index: int, n_eta: int) -> tuple[int, ...]:
    offsets = (-2, -1, 0, 1, 2) if axis in (0, 1) else (-3, -2, -1, 0, 1, 2)
    return tuple((eta_index + offset) % n_eta for offset in offsets)


def _nearest_pool(
    context: BuildContext,
    plane: int,
    xy: np.ndarray,
    *,
    exact: bool = False,
    pool_count: int = 32,
) -> tuple[np.ndarray, int]:
    compact = context.plane_indices[plane]
    owner_xy = context.arrays["owner_centroid_xy"]
    owner_ids = context.arrays["owner_flat_ids"]
    if exact:
        result = np.empty((len(xy), min(pool_count, len(compact))), dtype=np.int64)
        for row, point in enumerate(xy):
            radius2 = np.sum((owner_xy[compact] - point) ** 2, axis=1)
            order = np.lexsort((owner_ids[compact], radius2))
            result[row] = compact[order[: result.shape[1]]]
        return result, 0
    query_count = min(max(pool_count + 1, 2 * pool_count), len(compact))
    _distance, local = context.plane_trees[plane].query(xy, k=query_count)
    if query_count == 1:
        local = np.asarray(local)[:, None]
    candidate = compact[np.asarray(local, dtype=np.int64)]
    radius2 = np.sum((owner_xy[candidate] - xy[:, None, :]) ** 2, axis=2)
    order = np.lexsort((owner_ids[candidate], radius2), axis=1)
    candidate = np.take_along_axis(candidate, order, axis=1)
    sorted_radius2 = np.take_along_axis(radius2, order, axis=1)
    selected_pool_count = min(pool_count, candidate.shape[1])
    pool = candidate[:, :selected_pool_count].copy()
    fallback = 0
    if candidate.shape[1] > selected_pool_count:
        tied = (
            sorted_radius2[:, selected_pool_count - 1]
            == sorted_radius2[:, selected_pool_count]
        )
        for row in np.flatnonzero(tied):
            full_radius2 = np.sum((owner_xy[compact] - xy[row]) ** 2, axis=1)
            full_order = np.lexsort((owner_ids[compact], full_radius2))
            pool[row] = compact[full_order[:selected_pool_count]]
            fallback += 1
    return pool, fallback


def _sector_select(
    context: BuildContext,
    pool: np.ndarray,
    xy: np.ndarray,
    planar_scale: np.ndarray,
    count: int,
) -> np.ndarray:
    # Preserve the prototype's arithmetic, including its scaled-distance tie
    # handling.  Uniform planar scaling is mathematically redundant here, but
    # omitting the division can reverse an exact floating-point radius tie.
    displacement = (
        context.arrays["owner_centroid_xy"][pool] - xy[:, None, :]
    ) / planar_scale[:, None, None]
    angle = np.mod(np.arctan2(displacement[..., 1], displacement[..., 0]), 2.0 * np.pi)
    sector = np.floor(8.0 * angle / (2.0 * np.pi)).astype(np.int8)
    radius2 = np.sum(displacement**2, axis=2)
    owner_ids = context.arrays["owner_flat_ids"][pool]
    grouped = np.lexsort((owner_ids, radius2, sector), axis=1)
    rank = np.empty_like(sector, dtype=np.int8)
    counters = np.zeros((len(pool), 8), dtype=np.int8)
    rows = np.arange(len(pool))
    for position in range(pool.shape[1]):
        index = grouped[:, position]
        selected_sector = sector[rows, index]
        rank[rows, index] = counters[rows, selected_sector]
        counters[rows, selected_sector] += 1
    priority = rank.astype(np.int16) * 8 + sector
    order = np.argsort(priority, axis=1, kind="stable")[:, :count]
    return np.take_along_axis(pool, order, axis=1)


def _centered_observations(
    context: BuildContext,
    donors: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    batch, donor_count = donors.shape
    result = np.empty((batch, donor_count, len(EXPONENTS)), dtype=np.float64)
    eta = base._unwrap_periodic(
        context.owner_eta[donors], center[:, None, 2], context.eta_period
    )
    zeta = (eta - center[:, None, 2]) / scale[:, None, 2]
    cx = center[:, 0]
    cy = center[:, 1]
    sx = scale[:, 0]
    sy = scale[:, 1]
    for slot, (a, b, c) in enumerate(EXPONENTS):
        value = np.zeros((batch, donor_count), dtype=np.float64)
        for i in range(a + 1):
            for j in range(b + 1):
                coefficient = (
                    math.comb(a, i) * math.comb(b, j)
                    * (-cx) ** (a - i) * (-cy) ** (b - j)
                    / (sx**a * sy**b)
                )
                value += coefficient[:, None] * context.owner_moments[
                    donors, MOMENT_SLOT[(i, j)]
                ]
        if c:
            value *= zeta**c
        result[:, :, slot] = value
    return result


def _coverage(context: BuildContext, donors: np.ndarray, plane_count: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    reshaped = context.owner_j[donors].reshape(len(donors), plane_count, count)
    ordered = np.sort(reshaped, axis=2)
    distinct = 1 + np.sum(ordered[:, :, 1:] != ordered[:, :, :-1], axis=2)
    return np.min(distinct, axis=1), np.max(distinct, axis=1)


def _solve_rows(
    observation: np.ndarray, distance: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    kappa = 1.0 / (1.0 + distance**2)
    weighted = observation * kappa[:, :, None]
    gram = np.einsum("bni,bnj->bij", weighted, weighted, optimize=True)
    target = np.zeros((len(observation), len(EXPONENTS), 3), dtype=np.float64)
    target[:, 1, 0] = 1.0 / scale[:, 0]
    target[:, 2, 1] = 1.0 / scale[:, 1]
    target[:, 3, 2] = 1.0 / scale[:, 2]
    solution = np.linalg.solve(gram, target)
    weights = kappa[:, :, None] ** 2 * np.einsum(
        "bni,bij->bnj", observation, solution, optimize=True
    )
    eigenvalues = np.linalg.eigvalsh(gram)
    maximum = np.maximum(eigenvalues[:, -1], np.finfo(float).tiny)
    singular = np.sqrt(np.maximum(eigenvalues, 0.0))
    rank = np.sum(singular > base.SVD_CUTOFF * np.sqrt(maximum)[:, None], axis=1)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        condition = np.sqrt(
            maximum / np.maximum(eigenvalues[:, 0], np.finfo(float).tiny)
        )
    residual = np.max(
        np.abs(np.einsum("bni,bnj->bij", observation, weights) - target), axis=1
    )
    amplification = scale * np.sum(np.abs(weights), axis=1)
    return weights, {
        "rank": rank.astype(np.int16),
        "condition": condition,
        "residual": residual,
        "amplification": amplification,
    }


def _row_batch(
    context: BuildContext,
    axis: int,
    eta_index: int,
    points: np.ndarray,
    *,
    exact_query: bool = False,
    count: int = 24,
    pool_count: int = 32,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], int]:
    u = points[:, 0]
    theta = points[:, 1]
    center = np.column_stack((u * np.cos(theta), u * np.sin(theta), points[:, 2]))
    planar_scale = np.maximum(
        context.dr,
        np.maximum(np.abs(u), 0.5 * context.dr) * context.dtheta,
    )
    scale = np.column_stack((planar_scale, planar_scale, np.full(len(points), context.deta)))
    plane_ids = _planes(axis, eta_index, context.resolution)
    donor_parts = []
    tie_fallbacks = 0
    for plane in plane_ids:
        pool, fallback = _nearest_pool(
            context, plane, center[:, :2], exact=exact_query,
            pool_count=pool_count,
        )
        donor_parts.append(
            _sector_select(context, pool, center[:, :2], planar_scale, count)
        )
        tie_fallbacks += fallback
    donors = np.concatenate(donor_parts, axis=1)
    observation = _centered_observations(context, donors, center, scale)
    eta = base._unwrap_periodic(
        context.owner_eta[donors], center[:, None, 2], context.eta_period
    )
    distance = np.sqrt(
        ((context.arrays["owner_centroid_xy"][donors, 0] - center[:, None, 0]) / scale[:, None, 0])**2
        + ((context.arrays["owner_centroid_xy"][donors, 1] - center[:, None, 1]) / scale[:, None, 1])**2
        + ((eta - center[:, None, 2]) / scale[:, None, 2])**2
    )
    weights, diagnostics = _solve_rows(observation, distance, scale)
    minimum_coverage, maximum_coverage = _coverage(
        context, donors, len(plane_ids), count
    )
    diagnostics.update({
        "support_radius": np.max(distance, axis=1),
        "minimum_coverage": minimum_coverage,
        "maximum_coverage": maximum_coverage,
        "scale": scale,
    })
    return donors, weights, diagnostics, tie_fallbacks


def _apply_rows(
    context: BuildContext, donors: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    values = context.arrays["owner_values"][:, donors]
    return np.einsum("fbn,bnc->fbc", values, weights, optimize=True)


def _regular_to_logical_batch(regular: np.ndarray, points: np.ndarray) -> np.ndarray:
    result = np.empty_like(regular)
    u = points[:, 0]
    cosine = np.cos(points[:, 1])
    sine = np.sin(points[:, 1])
    result[..., 0] = cosine[None, :] * regular[..., 0] + sine[None, :] * regular[..., 1]
    result[..., 1] = -u[None, :] * sine[None, :] * regular[..., 0] + u[None, :] * cosine[None, :] * regular[..., 1]
    result[..., 2] = regular[..., 2]
    return result


def _chunk_identity(manifest_identity: Mapping[str, Any], axis: int, eta_index: int) -> dict[str, Any]:
    return {
        "schema": CHUNK_SCHEMA,
        "manifest_identity_sha256": hashlib.sha256(_canonical(manifest_identity).encode()).hexdigest(),
        "resolution": int(manifest_identity["resolution"]),
        "axis": int(axis),
        "eta_index": int(eta_index),
        "policy_sha256": _policy_hash(),
    }


def _chunk_paths(root: Path, axis: int, eta_index: int) -> tuple[Path, Path]:
    path = root / "chunks" / f"axis{axis}_eta{eta_index:03d}.npz"
    return path, path.with_suffix(".complete.json")


def _valid_chunk(root: Path, identity: Mapping[str, Any]) -> bool:
    path, marker_path = _chunk_paths(root, int(identity["axis"]), int(identity["eta_index"]))
    if not path.is_file() or not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("identity") == identity
        and bool(marker.get("complete"))
        and marker.get("sha256") == _sha256(path)
    )


def _build_chunk(
    context: BuildContext,
    root: Path,
    manifest_identity: Mapping[str, Any],
    axis: int,
    eta_index: int,
    *,
    batch_size: int,
    selected_rows: np.ndarray | None = None,
    write: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    keys, points = _planar_faces(context, axis, eta_index)
    if selected_rows is not None:
        keys = keys[selected_rows]
        points = points[selected_rows]
    count_rows = len(points)
    collapsed = (
        keys[:, 0] == 0 if axis == 0 else np.zeros(count_rows, dtype=bool)
    )
    logical = np.zeros((len(FIELD_NAMES), count_rows, 3), dtype=np.float64)
    rank = np.zeros(count_rows, dtype=np.int16)
    condition = np.ones(count_rows, dtype=np.float64)
    residual = np.zeros((count_rows, 3), dtype=np.float64)
    constant_residual = np.zeros(count_rows, dtype=np.float64)
    amplification = np.zeros((count_rows, 3), dtype=np.float64)
    radius = np.zeros(count_rows, dtype=np.float64)
    donor_count = np.zeros(count_rows, dtype=np.int16)
    coverage_min = np.zeros(count_rows, dtype=np.int16)
    coverage_max = np.zeros(count_rows, dtype=np.int16)
    fallback = np.zeros(count_rows, dtype=np.uint8)
    row_hash = hashlib.sha256()
    tie_fallbacks = 0
    sample_slots = np.unique(np.linspace(0, max(count_rows - 1, 0), min(9, count_rows), dtype=np.int64))
    sample_payload: list[dict[str, Any]] = []
    regular_indices = np.flatnonzero(~collapsed)
    for first in range(0, len(regular_indices), batch_size):
        selected = regular_indices[first : first + batch_size]
        donors, weights, diagnostics, tied = _row_batch(
            context, axis, eta_index, points[selected]
        )
        tie_fallbacks += tied
        logical[:, selected] = _regular_to_logical_batch(
            _apply_rows(context, donors, weights), points[selected]
        )
        row_donors = [donors[index] for index in range(len(selected))]
        row_weights = [weights[index] for index in range(len(selected))]
        donors_per_plane = np.full(len(selected), 24, dtype=np.int16)

        def failed_rows() -> np.ndarray:
            return (
                (diagnostics["rank"] < len(EXPONENTS))
                | (
                    np.max(diagnostics["residual"], axis=1)
                    > base.REPRODUCTION_TOLERANCE
                )
                | (diagnostics["minimum_coverage"] < 3)
            )

        for level, (pool_count, selected_count) in enumerate(
            ((32, 32), (48, 40), (64, 48)), start=1
        ):
            failed = failed_rows()
            if not np.any(failed):
                break
            local_rows = np.flatnonzero(failed)
            rebuilt_donors, rebuilt_weights, rebuilt, tied = _row_batch(
                context,
                axis,
                eta_index,
                points[selected[failed]],
                count=selected_count,
                pool_count=pool_count,
            )
            tie_fallbacks += tied
            logical[:, selected[failed]] = _regular_to_logical_batch(
                _apply_rows(context, rebuilt_donors, rebuilt_weights),
                points[selected[failed]],
            )
            for name in diagnostics:
                if (
                    name in rebuilt
                    and np.asarray(diagnostics[name]).shape[0] == len(selected)
                ):
                    diagnostics[name][failed] = rebuilt[name]
            for rebuilt_index, local_row in enumerate(local_rows):
                row_donors[local_row] = rebuilt_donors[rebuilt_index]
                row_weights[local_row] = rebuilt_weights[rebuilt_index]
            donors_per_plane[failed] = selected_count
            fallback[selected[failed]] = level
        failed = failed_rows()
        if np.any(failed):
            bad = selected[np.flatnonzero(failed)[0]]
            raise ValueError(
                f"balanced cubic row failed after all support expansions: "
                f"axis={axis}, eta={eta_index}, key_ij={keys[bad].tolist()}"
            )
        rank[selected] = diagnostics["rank"]
        condition[selected] = diagnostics["condition"]
        residual[selected] = diagnostics["residual"]
        amplification[selected] = diagnostics["amplification"]
        radius[selected] = diagnostics["support_radius"]
        coverage_min[selected] = diagnostics["minimum_coverage"]
        coverage_max[selected] = diagnostics["maximum_coverage"]
        constant_residual[selected] = np.asarray([
            np.max(np.abs(np.sum(row_weights[index], axis=0)))
            for index in range(len(selected))
        ])
        donor_count[selected] = donors_per_plane * len(
            _planes(axis, eta_index, context.resolution)
        )
        for local, global_row in enumerate(selected):
            local_donor = row_donors[local]
            local_weight = row_weights[local]
            row_hash.update(np.ascontiguousarray(local_donor.astype(np.int32)).view(np.uint8))
            row_hash.update(np.ascontiguousarray(local_weight).view(np.uint8))
            if global_row in sample_slots:
                sample_payload.append({
                    "row": int(global_row),
                    "key_ij": keys[global_row].tolist(),
                    "donors": local_donor.astype(np.int32).tolist(),
                    "weights_regular": local_weight.tolist(),
                })
    elapsed = time.perf_counter() - started
    regular = ~collapsed
    statistics = {
        "row_count": int(count_rows),
        "regular_row_count": int(np.sum(regular)),
        "collapsed_row_count": int(np.sum(collapsed)),
        "fallback_row_count": int(np.count_nonzero(fallback)),
        "fallback_level_counts": {
            str(level): int(np.count_nonzero(fallback == level))
            for level in (1, 2, 3)
        },
        "indexed_query_tie_fallback_count": int(tie_fallbacks),
        "maximum_condition": float(np.max(condition[regular])) if np.any(regular) else 1.0,
        "maximum_reproduction_residual": float(np.max(residual[regular])) if np.any(regular) else 0.0,
        "maximum_constant_derivative_residual": float(np.max(constant_residual[regular])) if np.any(regular) else 0.0,
        "maximum_amplification": np.max(amplification[regular], axis=0).tolist() if np.any(regular) else [0.0] * 3,
        "maximum_support_radius": float(np.max(radius[regular])) if np.any(regular) else 0.0,
        "minimum_angular_columns": int(np.min(coverage_min[regular])) if np.any(regular) else 0,
        "maximum_angular_columns": int(np.max(coverage_max[regular])) if np.any(regular) else 0,
        "condition_quantiles": np.quantile(condition[regular], (0.5, 0.9, 0.99, 1.0)).tolist() if np.any(regular) else [1.0] * 4,
        "amplification_quantiles": np.quantile(amplification[regular], (0.5, 0.9, 0.99, 1.0), axis=0).tolist() if np.any(regular) else [[0.0] * 3] * 4,
        "support_radius_quantiles": np.quantile(radius[regular], (0.5, 0.9, 0.99, 1.0)).tolist() if np.any(regular) else [0.0] * 4,
        "row_weights_sha256": row_hash.hexdigest(),
        "seconds": elapsed,
        "rows_per_second": float(np.sum(regular) / max(elapsed, np.finfo(float).tiny)),
        "maximum_rss_gib": _max_rss_gib(),
    }
    if not write:
        return statistics
    identity = _chunk_identity(manifest_identity, axis, eta_index)
    path, marker_path = _chunk_paths(root, axis, eta_index)
    arrays = {
        "identity_json": np.asarray(_canonical(identity)),
        "keys_ij": keys,
        "points_logical": points,
        "logical_gradient": logical,
        "row_rank": rank,
        "row_condition": condition,
        "row_residual": residual,
        "row_constant_residual": constant_residual,
        "row_amplification": amplification,
        "row_support_radius": radius,
        "row_donor_count": donor_count,
        "row_angular_columns_min": coverage_min,
        "row_angular_columns_max": coverage_max,
        "row_fallback": fallback,
        "sample_rows_json": np.asarray(json.dumps(sample_payload, sort_keys=True)),
        "statistics_json": np.asarray(json.dumps(statistics, sort_keys=True)),
    }
    _write_npz(path, arrays)
    _write_json(marker_path, {
        "schema": CHUNK_SCHEMA,
        "complete": True,
        "identity": identity,
        "sha256": _sha256(path),
        "statistics": statistics,
    })
    return statistics


def _profile(args: argparse.Namespace) -> dict[str, Any]:
    context = _load_context(args.geometry, args.baseline, args.resolution)
    started = time.perf_counter()
    records = []
    for axis, eta_index in ((0, 0), (0, args.resolution // 2), (1, 0), (1, args.resolution // 2), (2, 0), (2, args.resolution // 2)):
        keys, _points = _planar_faces(context, axis, eta_index)
        count = min(args.rows_per_stratum, len(keys))
        selected = np.unique(np.linspace(0, len(keys) - 1, count, dtype=np.int64))
        statistics = _build_chunk(
            context, args.output, {}, axis, eta_index,
            batch_size=args.batch_size, selected_rows=selected, write=False,
        )
        records.append({"axis": axis, "eta_index": eta_index, **statistics})
    measured_rows = sum(item["regular_row_count"] for item in records)
    measured_seconds = sum(item["seconds"] for item in records)
    n = args.resolution
    total_rows = n * ((n + 1) * n + n * (n + 1) + n * n)
    collapsed_rows = n * n
    estimate = (total_rows - collapsed_rows) * measured_seconds / measured_rows
    payload = {
        "schema": PROFILE_SCHEMA,
        "resolution": args.resolution,
        "policy": POLICY,
        "strata": records,
        "measured_regular_rows": measured_rows,
        "measured_seconds": measured_seconds,
        "estimated_global_row_build_seconds": estimate,
        "estimated_dense_gradient_bytes": int(len(FIELD_NAMES) * total_rows * 3 * 8),
        "estimated_diagnostic_bytes": int(total_rows * (2 + 8 + 3 * 8 + 3 * 8 + 8 + 2 + 2 + 2 + 1)),
        "maximum_rss_gib": _max_rss_gib(),
        "total_seconds": time.perf_counter() - started,
    }
    _write_json(args.output / "profile.json", payload)
    _progress("profile_complete", output=str((args.output / "profile.json").resolve()), estimate_seconds=estimate)
    return payload


def _aggregate_chunk_statistics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_regular = sum(int(item["regular_row_count"]) for item in records)
    fallback = sum(int(item["fallback_row_count"]) for item in records)
    return {
        "chunk_count": len(records),
        "row_count": sum(int(item["row_count"]) for item in records),
        "regular_row_count": total_regular,
        "collapsed_row_count": sum(int(item["collapsed_row_count"]) for item in records),
        "fallback_row_count": fallback,
        "fallback_fraction": float(fallback / max(total_regular, 1)),
        "fallback_level_counts": {
            str(level): sum(
                int(item.get("fallback_level_counts", {}).get(str(level), 0))
                for item in records
            )
            for level in (1, 2, 3)
        },
        "indexed_query_tie_fallback_count": sum(int(item["indexed_query_tie_fallback_count"]) for item in records),
        "maximum_condition": max(float(item["maximum_condition"]) for item in records),
        "maximum_reproduction_residual": max(float(item["maximum_reproduction_residual"]) for item in records),
        "maximum_constant_derivative_residual": max(float(item["maximum_constant_derivative_residual"]) for item in records),
        "maximum_amplification": np.max(np.asarray([item["maximum_amplification"] for item in records]), axis=0).tolist(),
        "maximum_support_radius": max(float(item["maximum_support_radius"]) for item in records),
        "minimum_angular_columns": min(int(item["minimum_angular_columns"]) for item in records if item["regular_row_count"]),
        "maximum_angular_columns": max(int(item["maximum_angular_columns"]) for item in records if item["regular_row_count"]),
        "row_build_seconds": sum(float(item["seconds"]) for item in records),
        "maximum_rss_gib": max(float(item["maximum_rss_gib"]) for item in records),
    }


def _build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    requested_checks = (
        "global_l2", "regional_budget", "A_B_C", "constant", "antisymmetry",
        "shared_face", "incidence_center", "old_action_replay",
    )
    identity = _manifest_identity(args.geometry, args.baseline, args.resolution, requested_checks)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("identity") != identity:
            raise ValueError("existing derivative cache identity does not match requested sources/policy")
        if prior.get("status") == "complete":
            expected = 3 * args.resolution
            if len(prior.get("chunks", [])) == expected and all(
                _valid_chunk(output, _chunk_identity(identity, axis, eta))
                for axis in range(3) for eta in range(args.resolution)
            ):
                _progress("cache_hit", resolution=args.resolution, output=str(output))
                return prior
    checkpoint = {
        "schema": SCHEMA,
        "status": "building",
        "identity": identity,
        "started_unix_time": time.time(),
        "chunks": [],
    }
    _write_json(manifest_path, checkpoint)
    context = _load_context(args.geometry, args.baseline, args.resolution)
    records = []
    for axis in range(3):
        for eta_index in range(args.resolution):
            chunk_identity = _chunk_identity(identity, axis, eta_index)
            path, marker = _chunk_paths(output, axis, eta_index)
            if _valid_chunk(output, chunk_identity):
                record = json.loads(marker.read_text(encoding="utf-8"))["statistics"]
                cache_hit = True
            else:
                record = _build_chunk(
                    context, output, identity, axis, eta_index,
                    batch_size=args.batch_size,
                )
                cache_hit = False
            records.append({"axis": axis, "eta_index": eta_index, **record})
            checkpoint["chunks"] = records
            checkpoint["statistics"] = _aggregate_chunk_statistics(records)
            checkpoint["maximum_rss_gib"] = _max_rss_gib()
            _write_json(manifest_path, checkpoint)
            _progress(
                "chunk_complete", resolution=args.resolution, axis=axis,
                eta_index=eta_index, seconds=record["seconds"], cache_hit=cache_hit,
            )
    checkpoint.update({
        "status": "complete",
        "complete": True,
        "statistics": _aggregate_chunk_statistics(records),
        "total_seconds": time.perf_counter() - started,
        "completed_unix_time": time.time(),
        "maximum_rss_gib": _max_rss_gib(),
    })
    _write_json(manifest_path, checkpoint)
    _progress("build_complete", resolution=args.resolution, output=str(output), seconds=checkpoint["total_seconds"])
    return checkpoint


def load_global_gradients(
    root: Path,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, Any]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("global cubic derivative manifest is incomplete")
    n = int(manifest["identity"]["resolution"])
    gradients: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    shapes = ((n + 1, n, n, 3), (n, n + 1, n, 3), (n, n, n + 1, 3))
    axis_arrays = [np.empty((len(FIELD_NAMES),) + shape, dtype=np.float64) for shape in shapes]
    for axis in range(3):
        for eta_index in range(n):
            identity = _chunk_identity(manifest["identity"], axis, eta_index)
            if not _valid_chunk(root, identity):
                raise ValueError(f"invalid derivative chunk axis={axis} eta={eta_index}")
            path, _marker = _chunk_paths(root, axis, eta_index)
            with np.load(path, allow_pickle=False) as source:
                values = np.asarray(source["logical_gradient"], dtype=np.float64)
            if axis == 0:
                axis_arrays[axis][:, :, :, eta_index, :] = values.reshape(len(FIELD_NAMES), n + 1, n, 3)
            elif axis == 1:
                axis_arrays[axis][:, :, :, eta_index, :] = values.reshape(len(FIELD_NAMES), n, n + 1, 3)
            else:
                axis_arrays[axis][:, :, :, eta_index, :] = values.reshape(len(FIELD_NAMES), n, n, 3)
    axis_arrays[2][:, :, :, n, :] = axis_arrays[2][:, :, :, 0, :]
    for field_index, field in enumerate(FIELD_NAMES):
        gradients[field] = tuple(
            axis_arrays[axis][field_index] for axis in range(3)
        )
    return gradients, manifest


def _optimized_single_row(
    context: BuildContext, key: np.ndarray, point: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    axis = int(key[0])
    eta_index = int(key[3]) % context.resolution
    donors, weights, diagnostics, _ = _row_batch(
        context, axis, eta_index, point[None, :], exact_query=True
    )
    failed = (
        diagnostics["rank"][0] < len(EXPONENTS)
        or np.max(diagnostics["residual"][0]) > base.REPRODUCTION_TOLERANCE
        or diagnostics["minimum_coverage"][0] < 3
    )
    if failed:
        donors, weights, diagnostics, _ = _row_batch(
            context, axis, eta_index, point[None, :], exact_query=True, count=32
        )
    return donors[0], weights[0], {name: value[0] for name, value in diagnostics.items()}


def _equivalence(args: argparse.Namespace) -> dict[str, Any]:
    context = _load_context(args.geometry, args.baseline, args.resolution)
    saved, metadata = base._load_cache(args.saved_cache.resolve())
    maximum_weight = 0.0
    maximum_action = 0.0
    maximum_donor_mismatch = 0
    donor_mismatches: list[dict[str, Any]] = []
    categories = {"axis": 0, "transition": 0, "ordinary": 0, "boundary": 0, "seam": 0}
    compared = 0
    for row, (key, point) in enumerate(zip(saved["face_keys"], saved["face_points"])):
        if saved["collapsed"][row]:
            continue
        donor, weights, _diagnostics = _optimized_single_row(context, key, point)
        lo, hi = saved["row_indptr"][row : row + 2]
        old_donor = saved["row_donor_index"][lo:hi]
        old_weights = saved["row_weights_regular"][lo:hi]
        if not np.array_equal(donor, old_donor):
            maximum_donor_mismatch += 1
            donor_mismatches.append({
                "row": int(row),
                "face_key": key.tolist(),
                "optimized_only": sorted(set(donor.tolist()) - set(old_donor.tolist())),
                "saved_only": sorted(set(old_donor.tolist()) - set(donor.tolist())),
            })
            continue
        maximum_weight = max(maximum_weight, float(np.max(np.abs(weights - old_weights))))
        old_action = context.arrays["owner_values"][:, old_donor] @ old_weights
        new_action = context.arrays["owner_values"][:, donor] @ weights
        maximum_action = max(maximum_action, float(np.max(np.abs(new_action - old_action))))
        compared += 1
        axis, i, j, k = (int(value) for value in key)
        if axis == 0 and i <= 1:
            categories["axis"] += 1
        elif i in (0, context.resolution - 1) or j in (0, context.resolution - 1):
            categories["boundary"] += 1
        elif k in (0, context.resolution - 1):
            categories["seam"] += 1
        elif saved["row_angular_columns_min"][row] <= 3:
            categories["transition"] += 1
        else:
            categories["ordinary"] += 1
    payload = {
        "schema": EQUIVALENCE_SCHEMA,
        "resolution": args.resolution,
        "saved_cache": _source_identity(args.saved_cache),
        "saved_policy": metadata["policy"],
        "optimized_policy": POLICY,
        "compared_regular_rows": compared,
        "donor_mismatch_rows": maximum_donor_mismatch,
        "donor_mismatches": donor_mismatches,
        "maximum_weight_absolute_difference": maximum_weight,
        "maximum_field_action_absolute_difference": maximum_action,
        "coverage_counts": categories,
        "accepted": bool(maximum_donor_mismatch == 0 and maximum_action <= args.action_tolerance),
        "action_tolerance": args.action_tolerance,
        "solver_difference": "original rectangular SVD versus equivalent full-rank Gram solve",
    }
    _write_json(args.output, payload)
    _progress("equivalence_complete", output=str(args.output.resolve()), accepted=payload["accepted"])
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    profile = sub.add_parser("profile")
    profile.add_argument("--resolution", type=int, choices=(32, 48, 64), default=32)
    profile.add_argument("--geometry", type=Path, required=True)
    profile.add_argument("--baseline", type=Path, required=True)
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument("--rows-per-stratum", type=int, default=256)
    profile.add_argument("--batch-size", type=int, default=256)
    profile.set_defaults(handler=_profile)
    build = sub.add_parser("build")
    build.add_argument("--resolution", type=int, choices=(32, 48, 64), required=True)
    build.add_argument("--geometry", type=Path, required=True)
    build.add_argument("--baseline", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--batch-size", type=int, default=256)
    build.set_defaults(handler=_build)
    equivalence = sub.add_parser("equivalence")
    equivalence.add_argument("--resolution", type=int, choices=(48, 64), required=True)
    equivalence.add_argument("--geometry", type=Path, required=True)
    equivalence.add_argument("--baseline", type=Path, required=True)
    equivalence.add_argument("--saved-cache", type=Path, required=True)
    equivalence.add_argument("--output", type=Path, required=True)
    equivalence.add_argument("--action-tolerance", type=float, default=2.0e-10)
    equivalence.set_defaults(handler=_equivalence)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
