#!/usr/bin/env python3
"""Build and replay the bounded D_O owner-to-face derivative candidate.

D_O reconstructs regular-chart derivatives from the actual midpoint/raw-volume
owner observation used by the frozen HSX MMS inputs.  Row construction depends
only on topology, raw sample locations/weights, and the fixed geometry-only
support policy; exact MMS gradients are validation data and are never inputs to
the solve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import audit_hsx_generator_factorization as factors  # noqa: E402
import audit_hsx_poisson_common_face_oracle as oracle  # noqa: E402


CACHE_SCHEMA = "drbx.hsx-owner-face-derivative-cache-v1"
CASE_SCHEMA = "drbx.hsx-owner-face-derivative-case-v1"
SUMMARY_SCHEMA = "drbx.hsx-owner-face-derivative-summary-v1"
BASIS_NAMES = (
    "1", "zx", "zy", "zeta", "zx2", "zxzy", "zxzeta",
    "zy2", "zyzeta", "zeta2",
)
FIELD_NAMES = factors.FIELD_NAMES
TRANSPORT_FIELDS = factors.TRANSPORT_FIELDS
RULES = factors.RULES
ACTIONS = factors.ACTIONS
AXES = factors.AXES
VARIANTS = ("HH", "D_phi", "D_field", "D_both", "HE")
DONOR_SCHEDULE = (12, 18, 24, 32)
SVD_CUTOFF = 1.0e-12
REPRODUCTION_TOLERANCE = 1.0e-9


def _row_policy(
    donor_schedule: Sequence[int] | None = None,
    *,
    support_mode: str = "adaptive_rank_reproduction",
) -> dict[str, Any]:
    schedule = tuple(DONOR_SCHEDULE if donor_schedule is None else donor_schedule)
    return {
        "name": "D_O",
        "degree": 2,
        "basis": list(BASIS_NAMES),
        "chart": "x=u*cos(theta), y=u*sin(theta), locally unwrapped full-2pi eta",
        "weight": "kappa=(1+r^2)^-1; minimize sum((w/kappa)^2)",
        "radial_angular_planes": 3,
        "eta_face_planes": 4,
        "support_mode": support_mode,
        "donors_per_plane_schedule": list(schedule),
        "sampling": "raw midpoint samples weighted by owner raw_volume / observed owner volume",
        "chain_rule": "du=cos(theta)dx+sin(theta)dy; dtheta=-u*sin(theta)dx+u*cos(theta)dy; deta=deta_regular",
    }


def _row_policy_sha256(policy: Mapping[str, Any] | None = None) -> str:
    selected = _row_policy() if policy is None else policy
    return hashlib.sha256(
        json.dumps(selected, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _source_identity(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _max_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / 1024.0**3


def _progress(stage: str, **details: Any) -> None:
    print(json.dumps({"event": "progress", "stage": stage, **details}, sort_keys=True), flush=True)


def _unwrap_periodic(values: np.ndarray, center: float, period: float) -> np.ndarray:
    return center + np.mod(values - center + 0.5 * period, period) - 0.5 * period


def _regular_to_logical(
    regular_gradient: np.ndarray, face_points: np.ndarray
) -> np.ndarray:
    """Transform (d_x,d_y,d_eta) to the logical covector (d_u,d_theta,d_eta)."""

    u = face_points[:, 0]
    theta = face_points[:, 1]
    cosine = np.cos(theta)
    sine = np.sin(theta)
    result = np.empty_like(regular_gradient)
    result[:, 0] = cosine * regular_gradient[:, 0] + sine * regular_gradient[:, 1]
    result[:, 1] = -u * sine * regular_gradient[:, 0] + u * cosine * regular_gradient[:, 1]
    result[:, 2] = regular_gradient[:, 2]
    return result


def _basis_from_raw_samples(
    raw_indices: np.ndarray,
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    raw_eta: np.ndarray,
    raw_volume: np.ndarray,
    face_xyz: np.ndarray,
    scale: np.ndarray,
    eta_period: float,
) -> np.ndarray:
    eta = _unwrap_periodic(raw_eta[raw_indices], face_xyz[2], eta_period)
    zx = (raw_x[raw_indices] - face_xyz[0]) / scale[0]
    zy = (raw_y[raw_indices] - face_xyz[1]) / scale[1]
    zeta = (eta - face_xyz[2]) / scale[2]
    basis = np.stack(
        (
            np.ones_like(zx), zx, zy, zeta, zx**2, zx * zy, zx * zeta,
            zy**2, zy * zeta, zeta**2,
        ),
        axis=-1,
    )
    weight = raw_volume[raw_indices]
    return np.sum(weight[:, None] * basis, axis=0) / np.sum(weight)


def _target_matrix(scale: np.ndarray) -> np.ndarray:
    target = np.zeros((len(BASIS_NAMES), 3), dtype=np.float64)
    target[1, 0] = 1.0 / scale[0]
    target[2, 1] = 1.0 / scale[1]
    target[3, 2] = 1.0 / scale[2]
    return target


def _solve_derivative_weights(
    observation: np.ndarray,
    scaled_distance: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve P.T w=t with one shared rank-revealing SVD for x/y/eta."""

    kappa = 1.0 / (1.0 + scaled_distance**2)
    matrix = observation.T * kappa[None, :]
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    cutoff = SVD_CUTOFF * singular[0]
    rank = int(np.sum(singular > cutoff))
    inverse = np.divide(
        1.0,
        singular,
        out=np.zeros_like(singular),
        where=singular > cutoff,
    )
    pseudoinverse = (vh.T * inverse[None, :]) @ u.T
    target = _target_matrix(scale)
    weights = kappa[:, None] * (pseudoinverse @ target)
    residual_by_component = np.max(np.abs(observation.T @ weights - target), axis=0)
    condition = float(singular[0] / singular[len(BASIS_NAMES) - 1]) if rank == len(BASIS_NAMES) else float("inf")
    amplification = scale * np.sum(np.abs(weights), axis=0)
    return weights, {
        "rank": rank,
        "condition": condition,
        "residual_by_component": residual_by_component,
        "amplification": amplification,
        "singular_values": singular,
    }


def _plane_offsets(axis: int) -> tuple[int, ...]:
    return (-1, 0, 1) if axis in (0, 1) else (-2, -1, 0, 1)


def _face_center_plane(axis: int, key: np.ndarray, n_eta: int) -> int:
    k = int(key[3])
    if axis in (0, 1):
        return k % n_eta
    # eta face k lies between center planes k-1 and k.
    return k % n_eta


def _candidate_planes(axis: int, key: np.ndarray, n_eta: int) -> tuple[int, ...]:
    center = _face_center_plane(axis, key, n_eta)
    if axis in (0, 1):
        return tuple((center + offset) % n_eta for offset in (-1, 0, 1))
    return tuple((center + offset) % n_eta for offset in (-2, -1, 0, 1))


def _build_row_for_face(
    key: np.ndarray,
    point: np.ndarray,
    owner_flat_ids: np.ndarray,
    owner_plane: np.ndarray,
    owner_centroid_xy: np.ndarray,
    raw_by_owner: Sequence[np.ndarray],
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    raw_eta: np.ndarray,
    raw_volume: np.ndarray,
    grid_faces: tuple[np.ndarray, np.ndarray, np.ndarray],
    eta_period: float,
    donor_schedule: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    axis = int(key[0])
    n_eta = len(grid_faces[2]) - 1
    face_xyz = np.asarray(
        [point[0] * np.cos(point[1]), point[0] * np.sin(point[1]), point[2]],
        dtype=np.float64,
    )
    dr = float(np.median(np.diff(grid_faces[0])))
    dtheta = float(np.median(np.diff(grid_faces[1])))
    deta = float(np.median(np.diff(grid_faces[2])))
    planar_scale = max(dr, max(abs(float(point[0])), 0.5 * dr) * dtheta)
    scale = np.asarray([planar_scale, planar_scale, deta], dtype=np.float64)
    planes = _candidate_planes(axis, key, n_eta)

    last = None
    schedule = tuple(DONOR_SCHEDULE if donor_schedule is None else donor_schedule)
    for donor_per_plane in schedule:
        donor_parts = []
        for plane in planes:
            compact = np.flatnonzero(owner_plane == plane)
            distance2 = np.sum((owner_centroid_xy[compact] - face_xyz[:2]) ** 2, axis=1)
            order = np.lexsort((owner_flat_ids[compact], distance2))
            donor_parts.append(compact[order[: min(donor_per_plane, len(order))]])
        donor = np.unique(np.concatenate(donor_parts))
        # Restore deterministic ordering by plane participation, distance, ID.
        plane_rank = {plane: index for index, plane in enumerate(planes)}
        eta_local = _unwrap_periodic(
            np.asarray([raw_eta[raw_by_owner[index][0]] for index in donor]),
            face_xyz[2],
            eta_period,
        )
        dx = (owner_centroid_xy[donor, 0] - face_xyz[0]) / scale[0]
        dy = (owner_centroid_xy[donor, 1] - face_xyz[1]) / scale[1]
        dz = (eta_local - face_xyz[2]) / scale[2]
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        order = np.lexsort(
            (
                owner_flat_ids[donor],
                distance,
                np.asarray([plane_rank[int(owner_plane[index])] for index in donor]),
            )
        )
        donor = donor[order]
        distance = distance[order]
        observation = np.stack([
            _basis_from_raw_samples(
                raw_by_owner[index], raw_x, raw_y, raw_eta, raw_volume,
                face_xyz, scale, eta_period,
            )
            for index in donor
        ])
        weights, diagnostic = _solve_derivative_weights(observation, distance, scale)
        last = (donor, weights, diagnostic, observation, scale, planes, donor_per_plane, distance)
        if (
            diagnostic["rank"] == len(BASIS_NAMES)
            and float(np.max(diagnostic["residual_by_component"])) <= REPRODUCTION_TOLERANCE
        ):
            break
    assert last is not None
    donor, weights, diagnostic, observation, scale, planes, donor_per_plane, distance = last
    if diagnostic["rank"] != len(BASIS_NAMES) or float(np.max(diagnostic["residual_by_component"])) > REPRODUCTION_TOLERANCE:
        raise ValueError(
            f"D_O row failed at face {key.tolist()}: rank={diagnostic['rank']}, "
            f"residual={float(np.max(diagnostic['residual_by_component'])):.3e}"
        )
    return donor, weights, {
        **diagnostic,
        "observation": observation,
        "scale": scale,
        "planes": np.asarray(planes, dtype=np.int16),
        "donor_per_plane": donor_per_plane,
        "support_radius": float(np.max(distance)),
    }


def _geometry_paths(root: Path, resolution: int) -> tuple[Path, Path, Path]:
    directory = root / f"{resolution}x{resolution}x{resolution}"
    return directory / "manifest.json", directory / "base_geometry.npz", directory / "rlp_topology.npz"


def _load_owner_inputs(
    geometry_root: Path,
    baseline_root: Path,
    resolution: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest_path, base_path, topology_path = _geometry_paths(geometry_root, resolution)
    with np.load(base_path, allow_pickle=False) as base:
        u = np.asarray(base["grid.x.centers"], dtype=np.float64)
        theta = np.asarray(base["grid.y.centers"], dtype=np.float64)
        eta = np.asarray(base["grid.z.centers"], dtype=np.float64)
        grid_faces = tuple(
            np.asarray(base[f"grid.{axis}.faces"], dtype=np.float64) for axis in "xyz"
        )
    with np.load(topology_path, allow_pickle=False) as topology:
        active = np.asarray(topology["is_active_owner"], dtype=bool)
        aggregate_id = np.asarray(topology["aggregate_id"], dtype=np.int64).reshape(-1)
        raw_volume = np.asarray(topology["raw_volume"], dtype=np.float64).reshape(-1)
        aggregate_chart_volume = np.asarray(
            topology["aggregate_chart_volume"], dtype=np.float64
        ).reshape(-1)
    shape = active.shape
    owner_flat_ids = np.flatnonzero(active.reshape(-1))
    owner_plane = np.unravel_index(owner_flat_ids, shape)[-1].astype(np.int32)
    mesh = np.meshgrid(u, theta, eta, indexing="ij")
    raw_x = (mesh[0] * np.cos(mesh[1])).reshape(-1)
    raw_y = (mesh[0] * np.sin(mesh[1])).reshape(-1)
    raw_eta = mesh[2].reshape(-1)
    compact_by_flat = np.full(int(np.prod(shape)), -1, dtype=np.int64)
    compact_by_flat[owner_flat_ids] = np.arange(len(owner_flat_ids))
    raw_by_owner = [np.flatnonzero(aggregate_id == flat) for flat in owner_flat_ids]
    observed_volume = np.asarray([np.sum(raw_volume[index]) for index in raw_by_owner])
    volume_reference = aggregate_chart_volume[owner_flat_ids]
    volume_mismatch = float(
        np.max(np.abs(observed_volume - volume_reference) / np.maximum(np.abs(volume_reference), np.finfo(float).tiny))
    )
    centroid_xy = np.asarray([
        [
            np.sum(raw_volume[index] * raw_x[index]) / np.sum(raw_volume[index]),
            np.sum(raw_volume[index] * raw_y[index]) / np.sum(raw_volume[index]),
        ]
        for index in raw_by_owner
    ])

    baseline_path = baseline_root / f"N{resolution}.reference.npz"
    with np.load(baseline_path, allow_pickle=False) as baseline:
        raw_fields = {
            "phi": np.asarray(baseline["actual_phi"], dtype=np.float64).reshape(-1),
            "actual_vorticity": np.asarray(baseline["actual_vorticity"], dtype=np.float64).reshape(-1),
            "smooth_regular_scalar": np.asarray(baseline["smooth_regular_scalar"], dtype=np.float64).reshape(-1),
            "smooth_eta_varying_scalar": np.asarray(baseline["smooth_eta_varying_scalar"], dtype=np.float64).reshape(-1),
        }
    owner_values = np.stack([
        np.asarray([
            np.sum(raw_volume[index] * raw_fields[field][index]) / np.sum(raw_volume[index])
            for index in raw_by_owner
        ])
        for field in FIELD_NAMES
    ])
    arrays = {
        "owner_flat_ids": owner_flat_ids.astype(np.int64),
        "owner_plane": owner_plane,
        "owner_centroid_xy": centroid_xy,
        "owner_values": owner_values,
        "raw_x": raw_x,
        "raw_y": raw_y,
        "raw_eta": raw_eta,
        "raw_volume": raw_volume,
        "grid_x_faces": grid_faces[0],
        "grid_y_faces": grid_faces[1],
        "grid_z_faces": grid_faces[2],
        "aggregate_id": aggregate_id,
    }
    metadata = {
        "paths": {
            "manifest": _source_identity(manifest_path),
            "base_geometry": _source_identity(base_path),
            "topology": _source_identity(topology_path),
            "baseline_reference": _source_identity(baseline_path),
        },
        "volume_relative_mismatch": volume_mismatch,
        "owner_count": int(len(owner_flat_ids)),
        "raw_count": int(len(raw_volume)),
    }
    # Force a one-dimensional object array even when every owner happens to
    # contain the same number of raw cells.
    raw_membership = np.empty(len(raw_by_owner), dtype=object)
    raw_membership[:] = raw_by_owner
    arrays["raw_by_owner_object"] = raw_membership
    return arrays, metadata


def _save_cache(path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    # Object arrays are never serialized. Convert raw membership to CSR first.
    raw_by_owner = arrays["raw_by_owner_object"]
    indptr = np.zeros(len(raw_by_owner) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(value) for value in raw_by_owner])
    serial = {name: value for name, value in arrays.items() if name != "raw_by_owner_object"}
    serial["owner_raw_indptr"] = indptr
    serial["owner_raw_indices"] = np.concatenate(list(raw_by_owner)).astype(np.int64)
    serial["metadata_json"] = np.asarray(json.dumps(oracle._json_value(metadata), sort_keys=True))
    _write_npz(path, serial)
    marker = {
        "schema": CACHE_SCHEMA,
        "complete": True,
        "cache": str(path.resolve()),
        "cache_sha256": _sha256(path),
        "array_manifest": {
            name: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)}
            for name, value in serial.items() if name != "metadata_json"
        },
    }
    _write_json(path.with_suffix(".complete.json"), marker)


def _load_cache(
    path: Path,
    *,
    validate_sources: bool = True,
    expected_policy: Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    marker_path = path.with_suffix(".complete.json")
    if not marker_path.is_file():
        raise ValueError("D_O cache completion marker is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema") != CACHE_SCHEMA or not marker.get("complete"):
        raise ValueError("D_O cache completion marker is invalid")
    if marker.get("cache_sha256") != _sha256(path):
        raise ValueError("D_O cache checksum mismatch")
    with np.load(path, allow_pickle=False) as cached:
        metadata = json.loads(str(cached["metadata_json"].item()))
        arrays = {name: np.asarray(cached[name]) for name in cached.files if name != "metadata_json"}
    if metadata.get("schema") != CACHE_SCHEMA:
        raise ValueError("D_O cache schema mismatch")
    stored_policy = metadata.get("policy")
    stored_policy_hash = metadata.get("identities", {}).get("row_policy_sha256")
    if not isinstance(stored_policy, dict) or stored_policy_hash != _row_policy_sha256(stored_policy):
        raise ValueError("stale D_O cache: derivative row policy identity mismatch")
    if expected_policy is not None and stored_policy_hash != _row_policy_sha256(expected_policy):
        raise ValueError("stale D_O cache: derivative row policy changed")
    for name, spec in marker["array_manifest"].items():
        if name not in arrays or list(arrays[name].shape) != spec["shape"] or str(arrays[name].dtype) != spec["dtype"]:
            raise ValueError(f"D_O cache array {name} has stale shape or dtype")
    if validate_sources:
        reasons = []
        for group in ("topology_sampling", "field_data", "factor_cache"):
            for name, identity in metadata["identities"][group].items():
                source = Path(identity["path"])
                if not source.is_file():
                    reasons.append(f"{group}:{name}:missing")
                elif _sha256(source) != identity["sha256"]:
                    reasons.append(f"{group}:{name}:sha256")
        if reasons:
            raise ValueError("stale D_O cache: " + ", ".join(reasons))
    return arrays, metadata


def _build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    resolution = int(args.resolution)
    output = args.output.resolve()
    if output.is_file() and output.with_suffix(".complete.json").is_file():
        try:
            policy = _row_policy()
            _, cached_metadata = _load_cache(output, expected_policy=policy)
            manifest_path, base_path, topology_path = _geometry_paths(args.geometry, resolution)
            requested = {
                "topology_sampling": {
                    "manifest": _source_identity(manifest_path),
                    "base_geometry": _source_identity(base_path),
                    "topology": _source_identity(topology_path),
                },
                "field_data": {
                    "baseline_reference": _source_identity(
                        args.baseline / f"N{resolution}.reference.npz"
                    )
                },
                "factor_cache": {"factor_cache": _source_identity(args.factor_cache)},
                "row_policy_sha256": _row_policy_sha256(policy),
            }
            if (
                int(cached_metadata["resolution"]) == resolution
                and cached_metadata["identities"] == requested
            ):
                _progress("cache_hit", resolution=resolution, output=str(output))
                return cached_metadata
        except (KeyError, ValueError):
            pass
    factor_arrays, factor_metadata = factors._load_cache(args.factor_cache.resolve())
    owner_arrays, owner_metadata = _load_owner_inputs(args.geometry, args.baseline, resolution)
    raw_by_owner = owner_arrays.pop("raw_by_owner_object")
    face_keys = factor_arrays["face_keys"]
    face_points = factor_arrays["face_points"]
    collapsed = factor_arrays["collapsed"].astype(bool)
    grid_faces = (
        owner_arrays["grid_x_faces"], owner_arrays["grid_y_faces"], owner_arrays["grid_z_faces"]
    )
    eta_period = float(grid_faces[2][-1] - grid_faces[2][0])

    donor_blocks = []
    weight_blocks = []
    indptr = np.zeros(len(face_keys) + 1, dtype=np.int64)
    rank = np.zeros(len(face_keys), dtype=np.int16)
    condition = np.ones(len(face_keys), dtype=np.float64)
    residual = np.zeros((len(face_keys), 3), dtype=np.float64)
    amplification = np.zeros((len(face_keys), 3), dtype=np.float64)
    support_radius = np.zeros(len(face_keys), dtype=np.float64)
    donor_count = np.zeros(len(face_keys), dtype=np.int16)
    donor_per_plane = np.zeros(len(face_keys), dtype=np.int16)
    scale = np.zeros((len(face_keys), 3), dtype=np.float64)
    plane_count = np.zeros(len(face_keys), dtype=np.int8)
    polynomial_check = np.zeros((len(face_keys), 3), dtype=np.float64)
    build_started = time.perf_counter()
    for face_index, (key, point) in enumerate(zip(face_keys, face_points)):
        if collapsed[face_index]:
            indptr[face_index + 1] = indptr[face_index]
            continue
        donor, weights, diagnostic = _build_row_for_face(
            key, point,
            owner_arrays["owner_flat_ids"], owner_arrays["owner_plane"],
            owner_arrays["owner_centroid_xy"], raw_by_owner,
            owner_arrays["raw_x"], owner_arrays["raw_y"], owner_arrays["raw_eta"],
            owner_arrays["raw_volume"], grid_faces, eta_period,
        )
        donor_blocks.append(donor.astype(np.int32))
        weight_blocks.append(weights)
        indptr[face_index + 1] = indptr[face_index] + len(donor)
        rank[face_index] = diagnostic["rank"]
        condition[face_index] = diagnostic["condition"]
        residual[face_index] = diagnostic["residual_by_component"]
        amplification[face_index] = diagnostic["amplification"]
        support_radius[face_index] = diagnostic["support_radius"]
        donor_count[face_index] = len(donor)
        donor_per_plane[face_index] = diagnostic["donor_per_plane"]
        scale[face_index] = diagnostic["scale"]
        plane_count[face_index] = len(diagnostic["planes"])
        # Independent direct polynomial action from the saved observation matrix.
        polynomial_check[face_index] = np.max(
            np.abs(diagnostic["observation"].T @ weights - _target_matrix(diagnostic["scale"])),
            axis=0,
        )
    row_seconds = time.perf_counter() - build_started
    donor_index = np.concatenate(donor_blocks) if donor_blocks else np.empty(0, dtype=np.int32)
    weights = np.concatenate(weight_blocks) if weight_blocks else np.empty((0, 3), dtype=np.float64)

    regular_gradient = np.zeros((len(FIELD_NAMES), len(face_keys), 3), dtype=np.float64)
    for face_index in np.flatnonzero(~collapsed):
        lo, hi = indptr[face_index : face_index + 2]
        regular_gradient[:, face_index] = owner_arrays["owner_values"][:, donor_index[lo:hi]] @ weights[lo:hi]
    logical_gradient = np.stack([
        _regular_to_logical(regular_gradient[field], face_points)
        for field in range(len(FIELD_NAMES))
    ])
    logical_gradient[:, collapsed] = 0.0

    policy = _row_policy()
    manifest_path, base_path, topology_path = _geometry_paths(args.geometry, resolution)
    baseline_path = args.baseline / f"N{resolution}.reference.npz"
    identities = {
        "topology_sampling": {
            "manifest": _source_identity(manifest_path),
            "base_geometry": _source_identity(base_path),
            "topology": _source_identity(topology_path),
        },
        "field_data": {"baseline_reference": _source_identity(baseline_path)},
        "factor_cache": {"factor_cache": _source_identity(args.factor_cache)},
        "row_policy_sha256": _row_policy_sha256(policy),
    }
    raw_membership = np.empty(len(raw_by_owner), dtype=object)
    raw_membership[:] = raw_by_owner
    output_arrays = {
        **owner_arrays,
        "face_keys": face_keys,
        "face_points": face_points,
        "collapsed": collapsed.astype(np.uint8),
        "row_indptr": indptr,
        "row_donor_index": donor_index,
        "row_weights_regular": weights,
        "row_rank": rank,
        "row_condition": condition,
        "row_residual": residual,
        "row_amplification": amplification,
        "row_support_radius": support_radius,
        "row_donor_count": donor_count,
        "row_donor_per_plane": donor_per_plane,
        "row_plane_count": plane_count,
        "row_scale": scale,
        "polynomial_check": polynomial_check,
        "regular_gradient_D_O": regular_gradient,
        "logical_gradient_D_O": logical_gradient,
        "raw_by_owner_object": raw_membership,
    }
    regular = ~collapsed
    metadata = {
        "schema": CACHE_SCHEMA,
        "resolution": resolution,
        "factor_cache_resolution": int(factor_metadata["resolution"]),
        "identities": identities,
        "policy": policy,
        "sampling_validation": owner_metadata,
        "row_diagnostics": {
            "face_count": int(len(face_keys)),
            "regular_face_count": int(np.sum(regular)),
            "collapsed_face_count": int(np.sum(collapsed)),
            "nnz": int(len(donor_index)),
            "donor_count_range": [int(np.min(donor_count[regular])), int(np.max(donor_count[regular]))],
            "donor_per_plane_range": [int(np.min(donor_per_plane[regular])), int(np.max(donor_per_plane[regular]))],
            "expansion_face_count": int(np.sum(donor_per_plane[regular] > DONOR_SCHEDULE[0])),
            "maximum_condition": float(np.max(condition[regular])),
            "maximum_residual": float(np.max(residual[regular])),
            "maximum_polynomial_check": float(np.max(polynomial_check[regular])),
            "maximum_amplification": np.max(amplification[regular], axis=0).tolist(),
            "maximum_support_radius": float(np.max(support_radius[regular])),
        },
        "timings_seconds": {
            "owner_input_and_moments": build_started - started,
            "row_build": row_seconds,
            "derivative_application": time.perf_counter() - build_started - row_seconds,
            "total_before_write": time.perf_counter() - started,
        },
        "maximum_rss_gib": _max_rss_gib(),
        "independence": {
            "exact_gradient_reads_during_row_build": 0,
            "continuum_evaluator_calls": 0,
            "model_build_calls": 0,
        },
    }
    _save_cache(output, output_arrays, metadata)
    _progress("cache_complete", resolution=resolution, output=str(output), seconds=time.perf_counter() - started)
    return metadata


def _rebuild_fixed_support(args: argparse.Namespace) -> dict[str, Any]:
    """Rebuild only derivative rows from a completed supplemental cache."""

    started = time.perf_counter()
    resolution = int(args.resolution)
    donors_per_plane = int(args.donors_per_plane)
    minimum_angular_columns = args.minimum_angular_columns
    fallback_donors = 24 if minimum_angular_columns is not None else None
    schedule = (donors_per_plane, fallback_donors) if fallback_donors is not None else (donors_per_plane,)
    policy = _row_policy(
        schedule,
        support_mode=(
            "coverage_minimum_then_fixed_fallback"
            if minimum_angular_columns is not None
            else "fixed_all_selected_faces"
        ),
    )
    if minimum_angular_columns is not None:
        policy["minimum_distinct_angular_columns_per_plane"] = int(minimum_angular_columns)
    output = args.output.resolve()
    source_path = args.source_row_cache.resolve()
    factor_path = args.factor_cache.resolve()
    expected_source = _source_identity(source_path)
    if output.is_file() and output.with_suffix(".complete.json").is_file():
        try:
            _, cached_metadata = _load_cache(output, expected_policy=policy)
            if (
                int(cached_metadata["resolution"]) == resolution
                and cached_metadata["identities"].get("supplemental_source") == expected_source
                and cached_metadata["identities"]["factor_cache"]["factor_cache"]
                == _source_identity(factor_path)
            ):
                _progress(
                    "cache_hit", resolution=resolution,
                    donors_per_plane=donors_per_plane, output=str(output),
                )
                return cached_metadata
        except (KeyError, ValueError):
            pass

    source_arrays, source_metadata = _load_cache(source_path)
    factor_arrays, factor_metadata = factors._load_cache(factor_path)
    if int(source_metadata["resolution"]) != resolution:
        raise ValueError("source row cache resolution mismatch")
    if int(factor_metadata["resolution"]) != resolution:
        raise ValueError("factor cache resolution mismatch")
    if not np.array_equal(source_arrays["face_keys"], factor_arrays["face_keys"]):
        raise ValueError("source row cache and factor cache face ordering mismatch")

    raw_indptr = source_arrays["owner_raw_indptr"]
    raw_indices = source_arrays["owner_raw_indices"]
    raw_by_owner = [
        raw_indices[raw_indptr[index] : raw_indptr[index + 1]]
        for index in range(len(raw_indptr) - 1)
    ]
    owner_names = (
        "owner_flat_ids", "owner_plane", "owner_centroid_xy", "owner_values",
        "raw_x", "raw_y", "raw_eta", "raw_volume", "grid_x_faces",
        "grid_y_faces", "grid_z_faces", "aggregate_id",
    )
    owner_arrays = {name: source_arrays[name] for name in owner_names}
    face_keys = factor_arrays["face_keys"]
    face_points = factor_arrays["face_points"]
    collapsed = factor_arrays["collapsed"].astype(bool)
    grid_faces = tuple(owner_arrays[name] for name in (
        "grid_x_faces", "grid_y_faces", "grid_z_faces"
    ))
    eta_period = float(grid_faces[2][-1] - grid_faces[2][0])

    donor_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    indptr = np.zeros(len(face_keys) + 1, dtype=np.int64)
    rank = np.zeros(len(face_keys), dtype=np.int16)
    condition = np.ones(len(face_keys), dtype=np.float64)
    residual = np.zeros((len(face_keys), 3), dtype=np.float64)
    amplification = np.zeros((len(face_keys), 3), dtype=np.float64)
    support_radius = np.zeros(len(face_keys), dtype=np.float64)
    donor_count = np.zeros(len(face_keys), dtype=np.int16)
    donor_per_plane_array = np.zeros(len(face_keys), dtype=np.int16)
    scale = np.zeros((len(face_keys), 3), dtype=np.float64)
    plane_count = np.zeros(len(face_keys), dtype=np.int8)
    polynomial_check = np.zeros((len(face_keys), 3), dtype=np.float64)
    angular_columns_min = np.zeros(len(face_keys), dtype=np.int16)
    angular_columns_max = np.zeros(len(face_keys), dtype=np.int16)
    grid_shape = tuple(len(faces) - 1 for faces in grid_faces)
    row_started = time.perf_counter()
    for face_index, (key, point) in enumerate(zip(face_keys, face_points)):
        if collapsed[face_index]:
            indptr[face_index + 1] = indptr[face_index]
            continue
        donor, weights, diagnostic = _build_row_for_face(
            key, point,
            owner_arrays["owner_flat_ids"], owner_arrays["owner_plane"],
            owner_arrays["owner_centroid_xy"], raw_by_owner,
            owner_arrays["raw_x"], owner_arrays["raw_y"], owner_arrays["raw_eta"],
            owner_arrays["raw_volume"], grid_faces, eta_period,
            donor_schedule=(donors_per_plane,),
        )
        donor_owner_ids = owner_arrays["owner_flat_ids"][donor]
        donor_multi = np.column_stack(np.unravel_index(donor_owner_ids, grid_shape))
        per_plane_coverage = [
            len(np.unique(donor_multi[donor_multi[:, 2] == plane, 1]))
            for plane in diagnostic["planes"]
        ]
        if (
            minimum_angular_columns is not None
            and min(per_plane_coverage) < minimum_angular_columns
        ):
            donor, weights, diagnostic = _build_row_for_face(
                key, point,
                owner_arrays["owner_flat_ids"], owner_arrays["owner_plane"],
                owner_arrays["owner_centroid_xy"], raw_by_owner,
                owner_arrays["raw_x"], owner_arrays["raw_y"], owner_arrays["raw_eta"],
                owner_arrays["raw_volume"], grid_faces, eta_period,
                donor_schedule=(fallback_donors,),
            )
            donor_owner_ids = owner_arrays["owner_flat_ids"][donor]
            donor_multi = np.column_stack(np.unravel_index(donor_owner_ids, grid_shape))
            per_plane_coverage = [
                len(np.unique(donor_multi[donor_multi[:, 2] == plane, 1]))
                for plane in diagnostic["planes"]
            ]
            if min(per_plane_coverage) < minimum_angular_columns:
                raise ValueError(
                    f"coverage fallback failed at face {key.tolist()}: "
                    f"minimum angular columns={min(per_plane_coverage)}"
                )
        donor_blocks.append(donor.astype(np.int32))
        weight_blocks.append(weights)
        indptr[face_index + 1] = indptr[face_index] + len(donor)
        rank[face_index] = diagnostic["rank"]
        condition[face_index] = diagnostic["condition"]
        residual[face_index] = diagnostic["residual_by_component"]
        amplification[face_index] = diagnostic["amplification"]
        support_radius[face_index] = diagnostic["support_radius"]
        donor_count[face_index] = len(donor)
        donor_per_plane_array[face_index] = diagnostic["donor_per_plane"]
        scale[face_index] = diagnostic["scale"]
        plane_count[face_index] = len(diagnostic["planes"])
        polynomial_check[face_index] = np.max(
            np.abs(
                diagnostic["observation"].T @ weights
                - _target_matrix(diagnostic["scale"])
            ),
            axis=0,
        )
        angular_columns_min[face_index] = min(per_plane_coverage)
        angular_columns_max[face_index] = max(per_plane_coverage)
    row_seconds = time.perf_counter() - row_started
    donor_index = np.concatenate(donor_blocks).astype(np.int32)
    weights = np.concatenate(weight_blocks)

    application_started = time.perf_counter()
    regular_gradient = np.zeros((len(FIELD_NAMES), len(face_keys), 3), dtype=np.float64)
    for face_index in np.flatnonzero(~collapsed):
        lo, hi = indptr[face_index : face_index + 2]
        regular_gradient[:, face_index] = (
            owner_arrays["owner_values"][:, donor_index[lo:hi]] @ weights[lo:hi]
        )
    logical_gradient = np.stack([
        _regular_to_logical(regular_gradient[field], face_points)
        for field in range(len(FIELD_NAMES))
    ])
    logical_gradient[:, collapsed] = 0.0
    application_seconds = time.perf_counter() - application_started

    raw_membership = np.empty(len(raw_by_owner), dtype=object)
    raw_membership[:] = raw_by_owner
    output_arrays = {
        **owner_arrays,
        "face_keys": face_keys,
        "face_points": face_points,
        "collapsed": collapsed.astype(np.uint8),
        "row_indptr": indptr,
        "row_donor_index": donor_index,
        "row_weights_regular": weights,
        "row_rank": rank,
        "row_condition": condition,
        "row_residual": residual,
        "row_amplification": amplification,
        "row_support_radius": support_radius,
        "row_donor_count": donor_count,
        "row_donor_per_plane": donor_per_plane_array,
        "row_plane_count": plane_count,
        "row_scale": scale,
        "row_angular_columns_min": angular_columns_min,
        "row_angular_columns_max": angular_columns_max,
        "polynomial_check": polynomial_check,
        "regular_gradient_D_O": regular_gradient,
        "logical_gradient_D_O": logical_gradient,
        "raw_by_owner_object": raw_membership,
    }
    regular = ~collapsed
    regular_indices = np.flatnonzero(regular)
    worst = regular_indices[np.argsort(np.max(amplification[regular], axis=1))[-5:][::-1]]
    identities = {
        "topology_sampling": source_metadata["identities"]["topology_sampling"],
        "field_data": source_metadata["identities"]["field_data"],
        "factor_cache": {"factor_cache": _source_identity(factor_path)},
        "supplemental_source": expected_source,
        "row_policy_sha256": _row_policy_sha256(policy),
    }
    metadata = {
        "schema": CACHE_SCHEMA,
        "resolution": resolution,
        "factor_cache_resolution": int(factor_metadata["resolution"]),
        "identities": identities,
        "policy": policy,
        "sampling_validation": source_metadata["sampling_validation"],
        "row_diagnostics": {
            "face_count": int(len(face_keys)),
            "regular_face_count": int(np.sum(regular)),
            "collapsed_face_count": int(np.sum(collapsed)),
            "nnz": int(len(donor_index)),
            "donor_count_range": [int(np.min(donor_count[regular])), int(np.max(donor_count[regular]))],
            "donor_per_plane_range": [
                int(np.min(donor_per_plane_array[regular])),
                int(np.max(donor_per_plane_array[regular])),
            ],
            "expansion_face_count": int(np.sum(donor_per_plane_array[regular] > donors_per_plane)),
            "maximum_condition": float(np.max(condition[regular])),
            "condition_quantiles": np.quantile(condition[regular], (0.5, 0.9, 0.99, 1.0)).tolist(),
            "maximum_residual": float(np.max(residual[regular])),
            "maximum_polynomial_check": float(np.max(polynomial_check[regular])),
            "maximum_amplification": np.max(amplification[regular], axis=0).tolist(),
            "amplification_quantiles_by_component": np.quantile(
                amplification[regular], (0.5, 0.9, 0.99, 1.0), axis=0
            ).tolist(),
            "maximum_support_radius": float(np.max(support_radius[regular])),
            "support_radius_quantiles": np.quantile(
                support_radius[regular], (0.5, 0.9, 0.99, 1.0)
            ).tolist(),
            "angular_columns_per_plane_range": [
                int(np.min(angular_columns_min[regular])),
                int(np.max(angular_columns_max[regular])),
            ],
            "worst_amplification_faces": [
                {
                    "face_index": int(index),
                    "face_key": face_keys[index].tolist(),
                    "condition": float(condition[index]),
                    "amplification_regular_xyz": amplification[index].tolist(),
                    "support_radius": float(support_radius[index]),
                    "angular_columns_per_plane": [
                        int(angular_columns_min[index]), int(angular_columns_max[index])
                    ],
                }
                for index in worst
            ],
        },
        "timings_seconds": {
            "cached_supplemental_load": row_started - started,
            "row_build": row_seconds,
            "derivative_application": application_seconds,
            "total_before_write": time.perf_counter() - started,
        },
        "maximum_rss_gib": _max_rss_gib(),
        "independence": {
            "exact_gradient_reads_during_row_build": 0,
            "continuum_evaluator_calls": 0,
            "model_build_calls": 0,
            "geometry_regeneration_calls": 0,
        },
    }
    _save_cache(output, output_arrays, metadata)
    _progress(
        "cache_complete", resolution=resolution, donors_per_plane=donors_per_plane,
        output=str(output), seconds=time.perf_counter() - started,
    )
    return metadata


def _stats(values: np.ndarray, target: np.ndarray, volume: np.ndarray) -> dict[str, float]:
    error = values - target
    return {
        "absolute_l2": float(np.sqrt(np.sum(volume * error**2) / np.sum(volume))),
        "signed_volume_weighted_mean_error": float(np.sum(volume * error) / np.sum(volume)),
        "maximum_absolute_error": float(np.max(np.abs(error))),
    }


def _replay(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    factor_arrays, factor_metadata = factors._load_cache(args.factor_cache.resolve())
    factor_case = json.loads(args.factor_case.resolve().read_text(encoding="utf-8"))
    row_arrays, row_metadata = _load_cache(args.row_cache.resolve())
    if int(factor_case["resolution"]) != int(factor_metadata["resolution"]):
        raise ValueError("factor case and factor cache resolution mismatch")
    if not np.array_equal(factor_arrays["face_keys"], row_arrays["face_keys"]):
        raise ValueError("D_O and factor cache face ordering mismatch")
    collapsed = factor_arrays["collapsed"].astype(bool)
    face_axis = factor_arrays["face_keys"][:, 0]
    gradient_do = row_arrays["logical_gradient_D_O"]
    owner_volume = factor_arrays["owner_volume"]
    targets = factor_arrays["targets"]
    rho_star = float(factor_metadata["rho_star"])
    generator_h = {}
    generator_do = {}
    generator_he = {}
    for field_index, field in enumerate(FIELD_NAMES):
        generator_h[field] = factors._generator_matrix(
            face_axis, factor_arrays["face_measure"], collapsed,
            factor_arrays["one_form_h"], factor_arrays["one_form_h"],
            factor_arrays["gradient_h"][field_index], factor_arrays["gradient_h"][field_index],
        )["HH"]
        generator_do[field] = factors._generator_matrix(
            face_axis, factor_arrays["face_measure"], collapsed,
            factor_arrays["one_form_h"], factor_arrays["one_form_h"],
            gradient_do[field_index], gradient_do[field_index],
        )["HH"]
        generator_he[field] = factors._generator_matrix(
            face_axis, factor_arrays["face_measure"], collapsed,
            factor_arrays["one_form_h"], factor_arrays["one_form_h"],
            factor_arrays["gradient_e"][field_index], factor_arrays["gradient_e"][field_index],
        )["HH"]

    field_payload = {}
    maximum_hh_replay = 0.0
    maximum_he_replay = 0.0
    for field_index, field in enumerate(TRANSPORT_FIELDS):
        rules_payload = {}
        owners = [
            {
                "owner": factor_arrays["owner_keys"][index].tolist(),
                "category": factor_metadata["owner_categories"][index],
                "aggregate_volume": float(owner_volume[index]),
                "continuum_target": float(targets[field_index, index]),
            }
            for index in range(len(owner_volume))
        ]
        for rule_index, rule in enumerate(RULES):
            action_values = {action: {} for action in ACTIONS}
            action_components = {action: {} for action in ACTIONS}
            choices = {
                "HH": (generator_h["phi"], generator_h[field]),
                "D_phi": (generator_do["phi"], generator_h[field]),
                "D_field": (generator_h["phi"], generator_do[field]),
                "D_both": (generator_do["phi"], generator_do[field]),
                "HE": (generator_he["phi"], generator_he[field]),
            }
            for variant, (a_generator, b_generator) in choices.items():
                a_total, a_face, a_center = factors._assemble_owner_components(
                    a_generator,
                    factor_arrays["transport"][rule_index, field_index + 1],
                    factor_arrays["center"][field_index + 1],
                    factor_arrays["face_incidence"], factor_arrays["raw_owner"], owner_volume,
                    sign=-1.0, rho_star=rho_star,
                )
                b_total, b_face, b_center = factors._assemble_owner_components(
                    b_generator,
                    factor_arrays["transport"][rule_index, 0],
                    factor_arrays["center"][0],
                    factor_arrays["face_incidence"], factor_arrays["raw_owner"], owner_volume,
                    sign=1.0, rho_star=rho_star,
                )
                action_values["A"][variant] = np.sum(a_total, axis=1)
                action_values["B"][variant] = np.sum(b_total, axis=1)
                action_values["C"][variant] = 0.5 * (
                    action_values["A"][variant] + action_values["B"][variant]
                )
                for action, total, face, center in (
                    ("A", a_total, a_face, a_center),
                    ("B", b_total, b_face, b_center),
                    ("C", 0.5 * (a_total + b_total), 0.5 * (a_face + b_face), 0.5 * (a_center + b_center)),
                ):
                    action_components[action][variant] = {"total": total, "face": face, "center": center}
            rules_payload[rule] = {}
            for action_index, action in enumerate(ACTIONS):
                maximum_hh_replay = max(
                    maximum_hh_replay,
                    float(np.max(np.abs(action_values[action]["HH"] - factor_arrays["prior_hh"][field_index, rule_index, action_index]))),
                )
                maximum_he_replay = max(
                    maximum_he_replay,
                    float(np.max(np.abs(
                        action_values[action]["HE"]
                        - np.asarray([
                            owner["results"][rule][action]["HE"]
                            for owner in factor_case["fields"][field]["owners"]
                        ], dtype=np.float64)
                    ))),
                )
                rules_payload[rule][action] = {
                    "statistics": {
                        variant: _stats(action_values[action][variant], targets[field_index], owner_volume)
                        for variant in VARIANTS
                    },
                    "change_from_HH": {
                        variant: {
                            "volume_weighted_rms": float(np.sqrt(np.sum(owner_volume * (action_values[action][variant] - action_values[action]["HH"])**2) / np.sum(owner_volume))),
                            "maximum_absolute": float(np.max(np.abs(action_values[action][variant] - action_values[action]["HH"]))),
                        }
                        for variant in ("D_phi", "D_field", "D_both", "HE")
                    },
                }
                for owner_index, row in enumerate(owners):
                    row.setdefault("results", {}).setdefault(rule, {})[action] = {
                        variant: float(action_values[action][variant][owner_index]) for variant in VARIANTS
                    }
                    row["results"][rule][action]["components"] = {
                        variant: {
                            part: dict(zip(AXES, action_components[action][variant][part][owner_index].tolist()))
                            for part in ("face", "center", "total")
                        }
                        for variant in VARIANTS
                    }
        regular = ~collapsed
        derivative_accuracy = {}
        generator_accuracy = {}
        for generator_field in ("phi", field):
            field_slot = FIELD_NAMES.index(generator_field)
            difference = gradient_do[field_slot, regular] - factor_arrays["gradient_e"][field_slot, regular]
            derivative_accuracy[generator_field] = {
                "component_rms": np.sqrt(np.mean(difference**2, axis=0)).tolist(),
                "component_max_abs": np.max(np.abs(difference), axis=0).tolist(),
                "relative_l2": float(
                    np.linalg.norm(difference)
                    / max(np.linalg.norm(factor_arrays["gradient_e"][field_slot, regular]), np.finfo(float).tiny)
                ),
            }
            generator_difference = generator_do[generator_field][regular] - generator_he[generator_field][regular]
            generator_accuracy[generator_field] = {
                "rms": float(np.sqrt(np.mean(generator_difference**2))),
                "maximum_absolute": float(np.max(np.abs(generator_difference))),
                "relative_l2": float(
                    np.linalg.norm(generator_difference)
                    / max(np.linalg.norm(generator_he[generator_field][regular]), np.finfo(float).tiny)
                ),
            }
        field_payload[field] = {
            "rules": rules_payload,
            "owners": owners,
            "derivative_accuracy": derivative_accuracy,
            "generator_accuracy": generator_accuracy,
        }

    payload = {
        "schema": CASE_SCHEMA,
        "resolution": int(factor_metadata["resolution"]),
        "factor_cache": _source_identity(args.factor_cache),
        "factor_case": _source_identity(args.factor_case),
        "row_cache": _source_identity(args.row_cache),
        "candidate": row_metadata["policy"],
        "row_diagnostics": row_metadata["row_diagnostics"],
        "sampling_validation": row_metadata["sampling_validation"],
        "replay": {
            "HH_max_abs": maximum_hh_replay,
            "HE_max_abs": maximum_he_replay,
            "model_build_calls": 0,
            "continuum_evaluator_calls": 0,
            "seconds": time.perf_counter() - started,
        },
        "fields": field_payload,
        "scope": {
            "production_changes": [],
            "interpretation": "bounded eight-owner diagnostic; cross-resolution sample changes are not convergence orders",
        },
    }
    _write_json(args.output.resolve(), payload)
    _progress("replay_complete", resolution=payload["resolution"], output=str(args.output.resolve()))
    return payload


def _merge(args: argparse.Namespace) -> dict[str, Any]:
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in args.cases]
    cases.sort(key=lambda item: item["resolution"])
    payload = {
        "schema": SUMMARY_SCHEMA,
        "resolutions": [case["resolution"] for case in cases],
        "cases": [str(path.resolve()) for path in args.cases],
        "production_changes": [],
        "candidate_C": {},
        "row_diagnostics": {str(case["resolution"]): case["row_diagnostics"] for case in cases},
        "replay": {str(case["resolution"]): case["replay"] for case in cases},
    }
    for field in TRANSPORT_FIELDS:
        payload["candidate_C"][field] = {}
        for rule in RULES:
            payload["candidate_C"][field][rule] = {
                variant: [
                    case["fields"][field]["rules"][rule]["C"]["statistics"][variant]["absolute_l2"]
                    for case in cases
                ]
                for variant in VARIANTS
            }
    _write_json(args.output.resolve(), payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--resolution", type=int, choices=(48, 64), required=True)
    build.add_argument("--geometry", type=Path, required=True)
    build.add_argument("--baseline", type=Path, required=True)
    build.add_argument("--factor-cache", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(function=_build)
    support = sub.add_parser("rebuild-support")
    support.add_argument("--resolution", type=int, choices=(48, 64), required=True)
    support.add_argument("--source-row-cache", type=Path, required=True)
    support.add_argument("--factor-cache", type=Path, required=True)
    support.add_argument("--donors-per-plane", type=int, choices=(18, 24), required=True)
    support.add_argument("--minimum-angular-columns", type=int, choices=(3,))
    support.add_argument("--output", type=Path, required=True)
    support.set_defaults(function=_rebuild_fixed_support)
    replay = sub.add_parser("replay")
    replay.add_argument("--factor-cache", type=Path, required=True)
    replay.add_argument("--factor-case", type=Path, required=True)
    replay.add_argument("--row-cache", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.set_defaults(function=_replay)
    merge = sub.add_parser("merge")
    merge.add_argument("cases", type=Path, nargs="+")
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(function=_merge)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
