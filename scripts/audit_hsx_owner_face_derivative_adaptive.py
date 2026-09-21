#!/usr/bin/env python3
"""Cached adaptive-support and truncation diagnostics for HSX D_O rows."""

from __future__ import annotations

import argparse
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
import audit_hsx_owner_face_derivative as base  # noqa: E402


ANALYSIS_SCHEMA = "drbx.hsx-owner-face-adaptive-analysis-v1"
FIELD_NAMES = base.FIELD_NAMES
TRANSPORT_FIELDS = base.TRANSPORT_FIELDS
RULES = base.RULES
ACTIONS = base.ACTIONS
AXES = base.AXES


CUBIC_EXPONENTS = (
    (3, 0, 0), (2, 1, 0), (2, 0, 1), (1, 2, 0), (1, 1, 1),
    (1, 0, 2), (0, 3, 0), (0, 2, 1), (0, 1, 2), (0, 0, 3),
)
CUBIC_NAMES = tuple(
    f"zx^{a}*zy^{b}*zeta^{c}" for a, b, c in CUBIC_EXPONENTS
)
DEGREE3_EXPONENTS = (
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0),
    (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2),
) + CUBIC_EXPONENTS
DEGREE3_NAMES = tuple(
    f"zx^{a}*zy^{b}*zeta^{c}" for a, b, c in DEGREE3_EXPONENTS
)


def _max_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / 1024.0**3


def _membership(arrays: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    ptr = arrays["owner_raw_indptr"]
    raw = arrays["owner_raw_indices"]
    return [raw[ptr[index] : ptr[index + 1]] for index in range(len(ptr) - 1)]


def _monomial_observation(
    raw_indices: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    center: np.ndarray,
    scale: np.ndarray,
    exponent: tuple[int, int, int],
    eta_period: float,
) -> float:
    eta = base._unwrap_periodic(arrays["raw_eta"][raw_indices], center[2], eta_period)
    z = np.column_stack((
        (arrays["raw_x"][raw_indices] - center[0]) / scale[0],
        (arrays["raw_y"][raw_indices] - center[1]) / scale[1],
        (eta - center[2]) / scale[2],
    ))
    value = np.prod(z ** np.asarray(exponent)[None, :], axis=1)
    weight = arrays["raw_volume"][raw_indices]
    return float(np.sum(weight * value) / np.sum(weight))


def _monomial_regular_gradient(
    xyz: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    exponent: tuple[int, int, int],
    eta_period: float,
) -> np.ndarray:
    local = xyz.copy()
    local[:, 2] = base._unwrap_periodic(local[:, 2], center[2], eta_period)
    z = (local - center[None, :]) / scale[None, :]
    result = np.zeros((len(xyz), 3), dtype=np.float64)
    for component in range(3):
        power = exponent[component]
        if power == 0:
            continue
        reduced = list(exponent)
        reduced[component] -= 1
        result[:, component] = (
            power / scale[component]
            * np.prod(z ** np.asarray(reduced)[None, :], axis=1)
        )
    return result


def _balanced_indices(
    compact: np.ndarray,
    owner_xy: np.ndarray,
    owner_ids: np.ndarray,
    face_xy: np.ndarray,
    planar_scale: float,
    count: int,
    pool_count: int = 32,
) -> np.ndarray:
    displacement = (owner_xy[compact] - face_xy[None, :]) / planar_scale
    radius2 = np.sum(displacement**2, axis=1)
    nearest = np.lexsort((owner_ids[compact], radius2))
    pool = compact[nearest[: min(pool_count, len(nearest))]]
    pool_displacement = (owner_xy[pool] - face_xy[None, :]) / planar_scale
    angle = np.mod(np.arctan2(pool_displacement[:, 1], pool_displacement[:, 0]), 2.0 * np.pi)
    sector = np.floor(8.0 * angle / (2.0 * np.pi)).astype(np.int8)
    pool_radius2 = np.sum(pool_displacement**2, axis=1)
    queues: list[list[int]] = []
    for sector_index in range(8):
        local = np.flatnonzero(sector == sector_index)
        order = np.lexsort((owner_ids[pool[local]], pool_radius2[local]))
        queues.append(pool[local[order]].tolist())
    chosen: list[int] = []
    layer = 0
    while len(chosen) < min(count, len(pool)):
        added = False
        for queue in queues:
            if layer < len(queue):
                chosen.append(queue[layer])
                added = True
                if len(chosen) == min(count, len(pool)):
                    break
        if not added:
            break
        layer += 1
    if len(chosen) < min(count, len(pool)):
        chosen_set = set(chosen)
        for value in pool:
            if int(value) not in chosen_set:
                chosen.append(int(value))
                if len(chosen) == min(count, len(pool)):
                    break
    return np.asarray(chosen, dtype=np.int64)


def _build_balanced_row(
    key: np.ndarray,
    point: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    members: Sequence[np.ndarray],
    count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    axis = int(key[0])
    grid_faces = tuple(arrays[name] for name in (
        "grid_x_faces", "grid_y_faces", "grid_z_faces"
    ))
    eta_period = float(grid_faces[2][-1] - grid_faces[2][0])
    face_xyz = np.asarray(
        [point[0] * np.cos(point[1]), point[0] * np.sin(point[1]), point[2]]
    )
    dr = float(np.median(np.diff(grid_faces[0])))
    dtheta = float(np.median(np.diff(grid_faces[1])))
    deta = float(np.median(np.diff(grid_faces[2])))
    planar_scale = max(dr, max(abs(float(point[0])), 0.5 * dr) * dtheta)
    scale = np.asarray([planar_scale, planar_scale, deta])
    planes = base._candidate_planes(axis, key, len(grid_faces[2]) - 1)
    donor_parts = []
    for plane in planes:
        compact = np.flatnonzero(arrays["owner_plane"] == plane)
        donor_parts.append(_balanced_indices(
            compact, arrays["owner_centroid_xy"], arrays["owner_flat_ids"],
            face_xyz[:2], planar_scale, count,
        ))
    donor = np.concatenate(donor_parts)
    eta_local = base._unwrap_periodic(
        np.asarray([arrays["raw_eta"][members[index][0]] for index in donor]),
        face_xyz[2], eta_period,
    )
    distance = np.sqrt(
        ((arrays["owner_centroid_xy"][donor, 0] - face_xyz[0]) / scale[0])**2
        + ((arrays["owner_centroid_xy"][donor, 1] - face_xyz[1]) / scale[1])**2
        + ((eta_local - face_xyz[2]) / scale[2])**2
    )
    observation = np.stack([
        base._basis_from_raw_samples(
            members[index], arrays["raw_x"], arrays["raw_y"], arrays["raw_eta"],
            arrays["raw_volume"], face_xyz, scale, eta_period,
        )
        for index in donor
    ])
    weights, diagnostic = base._solve_derivative_weights(observation, distance, scale)
    shape = tuple(len(faces) - 1 for faces in grid_faces)
    multi = np.column_stack(np.unravel_index(arrays["owner_flat_ids"][donor], shape))
    coverage = [
        len(np.unique(multi[multi[:, 2] == plane, 1])) for plane in planes
    ]
    return donor, weights, {
        **diagnostic,
        "scale": scale,
        "planes": np.asarray(planes),
        "support_radius": float(np.max(distance)),
        "angular_coverage": coverage,
        "observation": observation,
        "donors_per_plane": count,
    }


def _cubic_planes(axis: int, key: np.ndarray, n_eta: int) -> tuple[int, ...]:
    center = int(key[3]) % n_eta
    offsets = (-2, -1, 0, 1, 2) if axis in (0, 1) else (-3, -2, -1, 0, 1, 2)
    return tuple((center + offset) % n_eta for offset in offsets)


def _solve_degree3(
    observation: np.ndarray, scaled_distance: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    kappa = 1.0 / (1.0 + scaled_distance**2)
    matrix = observation.T * kappa[None, :]
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    cutoff = base.SVD_CUTOFF * singular[0]
    rank = int(np.sum(singular > cutoff))
    inverse = np.divide(1.0, singular, out=np.zeros_like(singular), where=singular > cutoff)
    pseudoinverse = (vh.T * inverse[None, :]) @ u.T
    target = np.zeros((len(DEGREE3_EXPONENTS), 3))
    target[1, 0] = 1.0 / scale[0]
    target[2, 1] = 1.0 / scale[1]
    target[3, 2] = 1.0 / scale[2]
    weights = kappa[:, None] * (pseudoinverse @ target)
    residual = np.max(np.abs(observation.T @ weights - target), axis=0)
    condition = (
        float(singular[0] / singular[len(DEGREE3_EXPONENTS) - 1])
        if rank == len(DEGREE3_EXPONENTS) else float("inf")
    )
    return weights, {
        "rank": rank,
        "condition": condition,
        "residual_by_component": residual,
        "amplification": scale * np.sum(np.abs(weights), axis=0),
        "singular_values": singular,
    }


def _build_cubic_row(
    key: np.ndarray,
    point: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    members: Sequence[np.ndarray],
    count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    axis = int(key[0])
    grid_faces = tuple(arrays[name] for name in (
        "grid_x_faces", "grid_y_faces", "grid_z_faces"
    ))
    eta_period = float(grid_faces[2][-1] - grid_faces[2][0])
    face_xyz = np.asarray(
        [point[0] * np.cos(point[1]), point[0] * np.sin(point[1]), point[2]]
    )
    dr = float(np.median(np.diff(grid_faces[0])))
    dtheta = float(np.median(np.diff(grid_faces[1])))
    deta = float(np.median(np.diff(grid_faces[2])))
    planar_scale = max(dr, max(abs(float(point[0])), 0.5 * dr) * dtheta)
    scale = np.asarray([planar_scale, planar_scale, deta])
    planes = _cubic_planes(axis, key, len(grid_faces[2]) - 1)
    donor = np.concatenate([
        _balanced_indices(
            np.flatnonzero(arrays["owner_plane"] == plane),
            arrays["owner_centroid_xy"], arrays["owner_flat_ids"],
            face_xyz[:2], planar_scale, count,
        )
        for plane in planes
    ])
    eta_local = base._unwrap_periodic(
        np.asarray([arrays["raw_eta"][members[index][0]] for index in donor]),
        face_xyz[2], eta_period,
    )
    distance = np.sqrt(
        ((arrays["owner_centroid_xy"][donor, 0] - face_xyz[0]) / scale[0])**2
        + ((arrays["owner_centroid_xy"][donor, 1] - face_xyz[1]) / scale[1])**2
        + ((eta_local - face_xyz[2]) / scale[2])**2
    )
    observation = np.asarray([
        [
            _monomial_observation(
                members[index], arrays, face_xyz, scale, exponent, eta_period
            )
            for exponent in DEGREE3_EXPONENTS
        ]
        for index in donor
    ])
    weights, diagnostic = _solve_degree3(observation, distance, scale)
    shape = tuple(len(faces) - 1 for faces in grid_faces)
    multi = np.column_stack(np.unravel_index(arrays["owner_flat_ids"][donor], shape))
    coverage = [
        len(np.unique(multi[multi[:, 2] == plane, 1])) for plane in planes
    ]
    return donor, weights, {
        **diagnostic,
        "scale": scale,
        "planes": np.asarray(planes),
        "support_radius": float(np.max(distance)),
        "angular_coverage": coverage,
        "observation": observation,
        "donors_per_plane": count,
    }


def _build_balanced(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    source_path = args.source_row_cache.resolve()
    factor_path = args.factor_cache.resolve()
    output = args.output.resolve()
    arrays, source_metadata = base._load_cache(source_path)
    factor_arrays, factor_metadata = factors._load_cache(factor_path)
    if int(source_metadata["resolution"]) != args.resolution:
        raise ValueError("source cache resolution mismatch")
    degree = int(getattr(args, "degree", 2))
    base_count, fallback_count = ((18, 24) if degree == 2 else (24, 32))
    policy = base._row_policy(
        (base_count, fallback_count),
        support_mode=(
            "balanced_xy_sectors_nearest32_coverage_fallback"
            if degree == 2 else
            "degree3_balanced_xy_sectors_nearest32_eta5_6_coverage_fallback"
        ),
    )
    policy.update({
        "degree": degree,
        "basis": list(base.BASIS_NAMES if degree == 2 else DEGREE3_NAMES),
        "candidate_pool_per_plane": 32,
        "planar_sectors": 8,
        "base_donors_per_plane": base_count,
        "fallback_donors_per_plane": fallback_count,
        "minimum_distinct_angular_columns_per_plane": 3,
    })
    if degree == 3:
        policy.update({"radial_angular_planes": 5, "eta_face_planes": 6})
    if output.is_file() and output.with_suffix(".complete.json").is_file():
        try:
            _, metadata = base._load_cache(output, expected_policy=policy)
            if metadata["identities"]["supplemental_source"] == base._source_identity(source_path):
                base._progress("cache_hit", resolution=args.resolution, output=str(output))
                return metadata
        except (KeyError, ValueError):
            pass

    members = _membership(arrays)
    collapsed = factor_arrays["collapsed"].astype(bool)
    face_keys = factor_arrays["face_keys"]
    face_points = factor_arrays["face_points"]
    row_ptr = np.zeros(len(face_keys) + 1, dtype=np.int64)
    donors: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    condition = np.ones(len(face_keys))
    amplification = np.zeros((len(face_keys), 3))
    residual = np.zeros((len(face_keys), 3))
    radius = np.zeros(len(face_keys))
    count = np.zeros(len(face_keys), dtype=np.int16)
    coverage = np.zeros((len(face_keys), 2), dtype=np.int16)
    scale = np.zeros((len(face_keys), 3))
    rank = np.zeros(len(face_keys), dtype=np.int16)
    row_started = time.perf_counter()
    for face_index, (key, point) in enumerate(zip(face_keys, face_points)):
        if collapsed[face_index]:
            row_ptr[face_index + 1] = row_ptr[face_index]
            continue
        selected_count = base_count
        row_builder = _build_balanced_row if degree == 2 else _build_cubic_row
        donor, row_weight, diagnostic = row_builder(
            key, point, arrays, members, selected_count
        )
        if (
            diagnostic["rank"] < (len(base.BASIS_NAMES) if degree == 2 else len(DEGREE3_EXPONENTS))
            or np.max(diagnostic["residual_by_component"]) > base.REPRODUCTION_TOLERANCE
            or min(diagnostic["angular_coverage"]) < 3
        ):
            selected_count = fallback_count
            donor, row_weight, diagnostic = row_builder(
                key, point, arrays, members, selected_count
            )
        if (
            diagnostic["rank"] < (len(base.BASIS_NAMES) if degree == 2 else len(DEGREE3_EXPONENTS))
            or np.max(diagnostic["residual_by_component"]) > base.REPRODUCTION_TOLERANCE
            or min(diagnostic["angular_coverage"]) < 3
        ):
            raise ValueError(f"balanced row failed at {key.tolist()}")
        donors.append(donor.astype(np.int32))
        weights.append(row_weight)
        row_ptr[face_index + 1] = row_ptr[face_index] + len(donor)
        condition[face_index] = diagnostic["condition"]
        amplification[face_index] = diagnostic["amplification"]
        residual[face_index] = diagnostic["residual_by_component"]
        radius[face_index] = diagnostic["support_radius"]
        count[face_index] = selected_count
        coverage[face_index] = [
            min(diagnostic["angular_coverage"]), max(diagnostic["angular_coverage"])
        ]
        scale[face_index] = diagnostic["scale"]
        rank[face_index] = diagnostic["rank"]
    row_seconds = time.perf_counter() - row_started
    donor_index = np.concatenate(donors)
    row_weights = np.concatenate(weights)
    regular_gradient = np.zeros((len(FIELD_NAMES), len(face_keys), 3))
    application_started = time.perf_counter()
    for face_index in np.flatnonzero(~collapsed):
        lo, hi = row_ptr[face_index : face_index + 2]
        regular_gradient[:, face_index] = (
            arrays["owner_values"][:, donor_index[lo:hi]] @ row_weights[lo:hi]
        )
    logical_gradient = np.stack([
        base._regular_to_logical(regular_gradient[index], face_points)
        for index in range(len(FIELD_NAMES))
    ])
    logical_gradient[:, collapsed] = 0.0
    application_seconds = time.perf_counter() - application_started

    owner_names = (
        "owner_flat_ids", "owner_plane", "owner_centroid_xy", "owner_values",
        "raw_x", "raw_y", "raw_eta", "raw_volume", "grid_x_faces",
        "grid_y_faces", "grid_z_faces", "aggregate_id",
    )
    raw_membership = np.empty(len(members), dtype=object)
    raw_membership[:] = members
    output_arrays = {
        **{name: arrays[name] for name in owner_names},
        "face_keys": face_keys, "face_points": face_points,
        "collapsed": collapsed.astype(np.uint8), "row_indptr": row_ptr,
        "row_donor_index": donor_index, "row_weights_regular": row_weights,
        "row_rank": rank, "row_condition": condition,
        "row_residual": residual, "row_amplification": amplification,
        "row_support_radius": radius, "row_donor_count": np.diff(row_ptr).astype(np.int16),
        "row_donor_per_plane": count,
        "row_plane_count": np.asarray([
            len(
                base._candidate_planes(int(k[0]), k, len(arrays['grid_z_faces']) - 1)
                if degree == 2 else
                _cubic_planes(int(k[0]), k, len(arrays['grid_z_faces']) - 1)
            )
            for k in face_keys
        ], dtype=np.int8),
        "row_scale": scale,
        "row_angular_columns_min": coverage[:, 0],
        "row_angular_columns_max": coverage[:, 1],
        "polynomial_check": residual,
        "regular_gradient_D_O": regular_gradient,
        "logical_gradient_D_O": logical_gradient,
        "raw_by_owner_object": raw_membership,
    }
    regular = ~collapsed
    metadata = {
        "schema": base.CACHE_SCHEMA,
        "resolution": int(args.resolution),
        "factor_cache_resolution": int(factor_metadata["resolution"]),
        "identities": {
            "topology_sampling": source_metadata["identities"]["topology_sampling"],
            "field_data": source_metadata["identities"]["field_data"],
            "factor_cache": {"factor_cache": base._source_identity(factor_path)},
            "supplemental_source": base._source_identity(source_path),
            "row_policy_sha256": base._row_policy_sha256(policy),
        },
        "policy": policy,
        "sampling_validation": source_metadata["sampling_validation"],
        "row_diagnostics": {
            "face_count": int(len(face_keys)),
            "regular_face_count": int(np.sum(regular)),
            "collapsed_face_count": int(np.sum(collapsed)),
            "nnz": int(len(donor_index)),
            "donor_per_plane_range": [int(np.min(count[regular])), int(np.max(count[regular]))],
            "expansion_face_count": int(np.sum(count[regular] > base_count)),
            "angular_columns_per_plane_range": [int(np.min(coverage[regular, 0])), int(np.max(coverage[regular, 1]))],
            "maximum_condition": float(np.max(condition[regular])),
            "condition_quantiles": np.quantile(condition[regular], (0.5, 0.9, 0.99, 1.0)).tolist(),
            "maximum_residual": float(np.max(residual[regular])),
            "maximum_polynomial_check": float(np.max(residual[regular])),
            "maximum_amplification": np.max(amplification[regular], axis=0).tolist(),
            "amplification_quantiles_by_component": np.quantile(amplification[regular], (0.5, 0.9, 0.99, 1.0), axis=0).tolist(),
            "maximum_support_radius": float(np.max(radius[regular])),
            "support_radius_quantiles": np.quantile(radius[regular], (0.5, 0.9, 0.99, 1.0)).tolist(),
        },
        "timings_seconds": {
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
    base._save_cache(output, output_arrays, metadata)
    base._progress("cache_complete", resolution=args.resolution, output=str(output))
    return metadata


def _action_arrays(
    factor_arrays: Mapping[str, np.ndarray],
    factor_metadata: Mapping[str, Any],
    logical_gradient: np.ndarray,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    collapsed = factor_arrays["collapsed"].astype(bool)
    face_axis = factor_arrays["face_keys"][:, 0]
    generator = {}
    for field_index, field in enumerate(FIELD_NAMES):
        generator[field] = factors._generator_matrix(
            face_axis, factor_arrays["face_measure"], collapsed,
            factor_arrays["one_form_h"], factor_arrays["one_form_h"],
            logical_gradient[field_index], logical_gradient[field_index],
        )["HH"]
    result: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    rho_star = float(factor_metadata["rho_star"])
    for field_index, field in enumerate(TRANSPORT_FIELDS):
        result[field] = {}
        for rule_index, rule in enumerate(RULES):
            a = factors._assemble_owner_components(
                generator["phi"], factor_arrays["transport"][rule_index, field_index + 1],
                factor_arrays["center"][field_index + 1], factor_arrays["face_incidence"],
                factor_arrays["raw_owner"], factor_arrays["owner_volume"],
                sign=-1.0, rho_star=rho_star,
            )
            b = factors._assemble_owner_components(
                generator[field], factor_arrays["transport"][rule_index, 0],
                factor_arrays["center"][0], factor_arrays["face_incidence"],
                factor_arrays["raw_owner"], factor_arrays["owner_volume"],
                sign=1.0, rho_star=rho_star,
            )
            result[field][rule] = {
                "A": {"total": a[0], "face": a[1], "center": a[2]},
                "B": {"total": b[0], "face": b[1], "center": b[2]},
                "C": {
                    "total": 0.5 * (a[0] + b[0]),
                    "face": 0.5 * (a[1] + b[1]),
                    "center": 0.5 * (a[2] + b[2]),
                },
            }
    return result


def _cubic_face_response(
    row_arrays: Mapping[str, np.ndarray], members: Sequence[np.ndarray]
) -> dict[str, Any]:
    regular = ~row_arrays["collapsed"].astype(bool)
    eta_period = float(row_arrays["grid_z_faces"][-1] - row_arrays["grid_z_faces"][0])
    response = np.zeros((len(row_arrays["face_keys"]), len(CUBIC_EXPONENTS), 3))
    for face_index in np.flatnonzero(regular):
        lo, hi = row_arrays["row_indptr"][face_index : face_index + 2]
        donor = row_arrays["row_donor_index"][lo:hi]
        weight = row_arrays["row_weights_regular"][lo:hi]
        point = row_arrays["face_points"][face_index]
        center = np.asarray([point[0] * np.cos(point[1]), point[0] * np.sin(point[1]), point[2]])
        scale = row_arrays["row_scale"][face_index]
        observation = np.asarray([
            [
                _monomial_observation(members[index], row_arrays, center, scale, exponent, eta_period)
                for exponent in CUBIC_EXPONENTS
            ]
            for index in donor
        ])
        # Exact derivatives of centered pure cubics vanish at the face.  Scaling
        # each reconstructed component by h makes this response dimensionless.
        response[face_index] = (observation.T @ weight) * scale[None, :]
    values = response[regular]
    return {
        "monomial_names": CUBIC_NAMES,
        "dimensionless_component_rms_by_monomial": np.sqrt(np.mean(values**2, axis=0)).tolist(),
        "dimensionless_component_max_by_monomial": np.max(np.abs(values), axis=0).tolist(),
        "overall_component_rms": np.sqrt(np.mean(values**2, axis=(0, 1))).tolist(),
        "overall_component_max": np.max(np.abs(values), axis=(0, 1)).tolist(),
    }


def _patch_cubic_sensitivity(
    row_arrays: Mapping[str, np.ndarray],
    factor_arrays: Mapping[str, np.ndarray],
    factor_metadata: Mapping[str, Any],
    members: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Apply common-chart unit cubics over each target owner's incident patch."""

    grid_shape = tuple(
        len(row_arrays[name]) - 1
        for name in ("grid_x_faces", "grid_y_faces", "grid_z_faces")
    )
    compact_by_flat = {
        int(flat): index for index, flat in enumerate(row_arrays["owner_flat_ids"])
    }
    eta_period = float(row_arrays["grid_z_faces"][-1] - row_arrays["grid_z_faces"][0])
    face_points = factor_arrays["face_points"]
    face_xyz = np.column_stack((
        face_points[:, 0] * np.cos(face_points[:, 1]),
        face_points[:, 0] * np.sin(face_points[:, 1]),
        face_points[:, 2],
    ))
    face_axis = factor_arrays["face_keys"][:, 0]
    collapsed = factor_arrays["collapsed"].astype(bool)
    rho_star = float(factor_metadata["rho_star"])
    records = []
    for target_owner, owner_key in enumerate(factor_arrays["owner_keys"]):
        flat = int(np.ravel_multi_index(tuple(owner_key), grid_shape))
        compact_target = compact_by_flat[flat]
        raw_rows = np.flatnonzero(factor_arrays["raw_owner"] == target_owner)
        incident = np.unique(factor_arrays["face_incidence"][raw_rows].reshape(-1))
        incident = incident[~collapsed[incident]]
        donor = np.unique(np.concatenate([
            row_arrays["row_donor_index"][
                row_arrays["row_indptr"][face] : row_arrays["row_indptr"][face + 1]
            ]
            for face in incident
        ]))
        center = np.asarray([
            row_arrays["owner_centroid_xy"][compact_target, 0],
            row_arrays["owner_centroid_xy"][compact_target, 1],
            row_arrays["raw_eta"][members[compact_target][0]],
        ])
        scale = np.median(row_arrays["row_scale"][incident], axis=0)
        for mode_index, exponent in enumerate(CUBIC_EXPONENTS):
            owner_value = np.zeros(len(row_arrays["owner_flat_ids"]))
            owner_value[donor] = [
                _monomial_observation(
                    members[index], row_arrays, center, scale, exponent, eta_period
                )
                for index in donor
            ]
            numerical_regular = np.zeros((len(face_points), 3))
            for face in incident:
                lo, hi = row_arrays["row_indptr"][face : face + 2]
                numerical_regular[face] = (
                    owner_value[row_arrays["row_donor_index"][lo:hi]]
                    @ row_arrays["row_weights_regular"][lo:hi]
                )
            exact_regular = np.zeros_like(numerical_regular)
            exact_regular[incident] = _monomial_regular_gradient(
                face_xyz[incident], center, scale, exponent, eta_period
            )
            numerical_logical = base._regular_to_logical(numerical_regular, face_points)
            exact_logical = base._regular_to_logical(exact_regular, face_points)
            generator_delta = factors._normal_cross(
                face_axis,
                factor_arrays["one_form_h"],
                numerical_logical - exact_logical,
            ) * factor_arrays["face_measure"]
            generator_delta[collapsed] = 0.0
            actions: dict[str, dict[str, float]] = {}
            for field_index, field in enumerate(TRANSPORT_FIELDS):
                actions[field] = {}
                for rule_index, rule in enumerate(RULES):
                    a = factors._assemble_owner_components(
                        generator_delta,
                        factor_arrays["transport"][rule_index, field_index + 1],
                        factor_arrays["center"][field_index + 1],
                        factor_arrays["face_incidence"], factor_arrays["raw_owner"],
                        factor_arrays["owner_volume"], sign=-1.0, rho_star=rho_star,
                    )[0]
                    b = factors._assemble_owner_components(
                        generator_delta,
                        factor_arrays["transport"][rule_index, 0],
                        factor_arrays["center"][0], factor_arrays["face_incidence"],
                        factor_arrays["raw_owner"], factor_arrays["owner_volume"],
                        sign=1.0, rho_star=rho_star,
                    )[0]
                    actions[field][rule] = {
                        "A": float(np.sum(a[target_owner])),
                        "B": float(np.sum(b[target_owner])),
                        "C": float(0.5 * np.sum(a[target_owner] + b[target_owner])),
                    }
            records.append({
                "owner_index": int(target_owner),
                "owner": owner_key.tolist(),
                "mode": CUBIC_NAMES[mode_index],
                "patch_center_xyz": center.tolist(),
                "patch_scale_xyz": scale.tolist(),
                "incident_face_count": int(len(incident)),
                "donor_count": int(len(donor)),
                "actions": actions,
            })
    summary: dict[str, Any] = {}
    for field in TRANSPORT_FIELDS:
        summary[field] = {}
        for rule in RULES:
            summary[field][rule] = {}
            for action in ACTIONS:
                values = np.asarray([
                    record["actions"][field][rule][action] for record in records
                ])
                summary[field][rule][action] = {
                    "rms_over_owner_modes": float(np.sqrt(np.mean(values**2))),
                    "maximum_absolute": float(np.max(np.abs(values))),
                }
    return {
        "convention": (
            "one centered/scaled xyz chart and eta branch per target owner patch; "
            "the same unit cubic is evaluated for every incident face and donor"
        ),
        "records": records,
        "summary": summary,
    }


def _analyze(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    factor_arrays, factor_metadata = factors._load_cache(args.factor_cache.resolve())
    row_arrays, row_metadata = base._load_cache(args.row_cache.resolve())
    case = json.loads(args.case.resolve().read_text(encoding="utf-8"))
    candidate = _action_arrays(factor_arrays, factor_metadata, row_arrays["logical_gradient_D_O"])
    exact = _action_arrays(factor_arrays, factor_metadata, factor_arrays["gradient_e"])
    volume = factor_arrays["owner_volume"]
    localization: dict[str, Any] = {}
    maximum_direct_identity = 0.0
    maximum_saved_action_identity = 0.0
    for field_index, field in enumerate(TRANSPORT_FIELDS):
        localization[field] = {}
        categories = [owner["category"] for owner in case["fields"][field]["owners"]]
        for rule in RULES:
            localization[field][rule] = {}
            for action in ACTIONS:
                difference = {
                    part: candidate[field][rule][action][part] - exact[field][rule][action][part]
                    for part in ("face", "center", "total")
                }
                maximum_direct_identity = max(
                    maximum_direct_identity,
                    float(np.max(np.abs(
                        difference["total"] - difference["face"] - difference["center"]
                    ))),
                )
                candidate_value = np.sum(candidate[field][rule][action]["total"], axis=1)
                exact_value = np.sum(exact[field][rule][action]["total"], axis=1)
                saved_difference = np.asarray([
                    owner["results"][rule][action]["D_both"]
                    - owner["results"][rule][action]["HE"]
                    for owner in case["fields"][field]["owners"]
                ])
                maximum_saved_action_identity = max(
                    maximum_saved_action_identity,
                    float(np.max(np.abs((candidate_value - exact_value) - saved_difference))),
                )
                target = factor_arrays["targets"][field_index]
                squared = volume * (candidate_value - target)**2
                total_squared = float(np.sum(squared))
                ranking = np.argsort(squared)[::-1]
                localization[field][rule][action] = {
                    "candidate_minus_HE": {
                        part: {
                            "signed_volume_weighted_mean_by_axis": (
                                np.sum(volume[:, None] * difference[part], axis=0) / np.sum(volume)
                            ).tolist(),
                            "volume_weighted_rms_by_axis": np.sqrt(
                                np.sum(volume[:, None] * difference[part]**2, axis=0) / np.sum(volume)
                            ).tolist(),
                        }
                        for part in ("face", "center", "total")
                    },
                    "squared_error_ranking": [
                        {
                            "owner_index": int(index),
                            "owner": factor_arrays["owner_keys"][index].tolist(),
                            "category": categories[index],
                            "fraction": float(squared[index] / total_squared) if total_squared else 0.0,
                            "signed_error": float(candidate_value[index] - target[index]),
                        }
                        for index in ranking
                    ],
                }
    members = _membership(row_arrays)
    payload = {
        "schema": ANALYSIS_SCHEMA,
        "resolution": int(factor_metadata["resolution"]),
        "factor_cache": base._source_identity(args.factor_cache),
        "row_cache": base._source_identity(args.row_cache),
        "case": base._source_identity(args.case),
        "policy": row_metadata["policy"],
        "localization": localization,
        "cubic_face_response": _cubic_face_response(row_arrays, members),
        "cubic_patch_sensitivity": _patch_cubic_sensitivity(
            row_arrays, factor_arrays, factor_metadata, members
        ),
        "checks": {
            "maximum_face_plus_center_identity": maximum_direct_identity,
            "maximum_direct_D_both_minus_HE_identity": maximum_saved_action_identity,
            "HH_replay": case["replay"]["HH_max_abs"],
            "HE_replay": case["replay"]["HE_max_abs"],
            "model_build_calls": 0,
            "continuum_evaluator_calls": 0,
        },
        "timings_seconds": {"total": time.perf_counter() - started},
        "maximum_rss_gib": _max_rss_gib(),
    }
    base._write_json(args.output.resolve(), payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-balanced")
    build.add_argument("--resolution", type=int, choices=(48, 64), required=True)
    build.add_argument("--source-row-cache", type=Path, required=True)
    build.add_argument("--factor-cache", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(function=_build_balanced, degree=2)
    cubic = sub.add_parser("build-cubic")
    cubic.add_argument("--resolution", type=int, choices=(48, 64), required=True)
    cubic.add_argument("--source-row-cache", type=Path, required=True)
    cubic.add_argument("--factor-cache", type=Path, required=True)
    cubic.add_argument("--output", type=Path, required=True)
    cubic.set_defaults(function=_build_balanced, degree=3)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--factor-cache", type=Path, required=True)
    analyze.add_argument("--row-cache", type=Path, required=True)
    analyze.add_argument("--case", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.set_defaults(function=_analyze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
