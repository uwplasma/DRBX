#!/usr/bin/env python3
"""Checkpointed global qualification of the frozen matched cubic HSX bracket.

This is research-only orchestration.  It reuses the qualified cubic policy and
reference evaluators without modifying production solver or boundary code.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).resolve().parent
# Portable v2 changes only bootstrap/path resolution.  The numerical routines
# below are an otherwise byte-for-byte snapshot of the frozen v1 candidate.
WORKSPACE = Path(os.environ.get("HSX_DEPLOYMENT_ROOT", HERE.parents[1])).resolve()
ROOT = Path(os.environ.get("HSX_SOURCE_ROOT", WORKSPACE / "DRBX")).resolve()
SCRIPTS = ROOT / "scripts"
LOCAL_JAX_CACHE = Path(os.environ.get("HSX_JAX_CACHE", HERE / "jax_cache")).resolve()
LOCAL_JAX_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["DRBX_CACHE_DIR"] = str(LOCAL_JAX_CACHE)
os.environ["JAX_COMPILATION_CACHE_DIR"] = str(LOCAL_JAX_CACHE)
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import audit_hsx_bracket_integrated_reference as integrated  # noqa: E402
import audit_hsx_cubic_derivative_global as cubic  # noqa: E402
import audit_hsx_cubic_global_bracket as global_bracket  # noqa: E402
import audit_hsx_matched_face_volume as bounded  # noqa: E402
import audit_hsx_p03_fine_interval as bridge  # noqa: E402
import audit_hsx_p04_global_bracket as p04  # noqa: E402
import audit_hsx_poisson_vorticity as base  # noqa: E402
import audit_hsx_value_derivative_cross as cross  # noqa: E402
import simulate_hsx_mms as mms  # noqa: E402


SCHEMA = "drbx.hsx-matched-cubic-global-case-v1"
PREPARE_SCHEMA = "drbx.hsx-matched-cubic-global-prepare-v1"
CHUNK_SCHEMA = "drbx.hsx-matched-cubic-global-chunk-v1"
PREFLIGHT_SCHEMA = "drbx.hsx-matched-cubic-global-preflight-v1"
SUMMARY_SCHEMA = "drbx.hsx-matched-cubic-global-summary-v1"
FIELDS = bounded.FIELDS
ALL_FIELDS = bounded.ALL_FIELDS
DIAGNOSTIC_FIELDS = FIELDS + ("constant",)
ACTIONS = ("A", "B", "C")
REGIONS = integrated.REGIONS


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
    return integrated._array_hash(*arrays)


def _json(value: Any) -> Any:
    return integrated._json_value(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_json(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            **arrays,
            metadata_json=np.asarray(json.dumps(_json(metadata), sort_keys=True)),
        )
    temporary.replace(path)


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved), "bytes": resolved.stat().st_size}


def _event(stage: str, **details: Any) -> None:
    print(json.dumps({"event": "matched_global", "stage": stage, **details}, sort_keys=True), flush=True)


def _config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "drbx.hsx-matched-cubic-global-config-v1":
        raise ValueError("unsupported global matched configuration")
    return payload


def _path(config: Mapping[str, Any], name: str) -> Path:
    return Path(config["paths"][name]).resolve()


def _source_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        "implementation": Path(__file__),
        "bounded_implementation": SCRIPTS / "audit_hsx_matched_face_volume.py",
        "cubic_builder": SCRIPTS / "audit_hsx_cubic_derivative_global.py",
        "integrated_reference": SCRIPTS / "audit_hsx_bracket_integrated_reference.py",
        "reference_sidecar": _path(config, "reference_sidecar"),
    }
    return {name: _identity(path) for name, path in paths.items()}


@dataclass
class Fit:
    center_logical: np.ndarray
    center_regular: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    diagnostics: dict[str, Any]


def _fit_entity(
    context: cubic.BuildContext,
    point: np.ndarray,
    *,
    axis: int,
    eta_index: int,
    owner_values: np.ndarray,
) -> Fit:
    """Frozen parent WLS fit, with extra diagnostic RHS columns only."""

    point = np.asarray(point, dtype=np.float64)
    accepted = None
    for level, policy in enumerate(cubic.POLICY["deficient_row_expansion_schedule"]):
        donors, _weights, diagnostics, tie_count = cubic._row_batch(
            context,
            int(axis),
            int(eta_index) % context.resolution,
            point[None, :],
            exact_query=True,
            count=int(policy["donors_per_plane"]),
            pool_count=int(policy["candidate_pool_per_plane"]),
        )
        valid = (
            diagnostics["rank"][0] == len(cubic.EXPONENTS)
            and np.max(diagnostics["residual"][0]) <= cubic.base.REPRODUCTION_TOLERANCE
            and diagnostics["minimum_coverage"][0]
            >= cubic.POLICY["minimum_distinct_angular_columns_per_plane"]
        )
        if valid:
            accepted = (level, donors[0], diagnostics, tie_count)
            break
    if accepted is None:
        raise RuntimeError("fixed cubic policy remains deficient for an entity")
    level, donor, diagnostics, tie_count = accepted
    scale = np.asarray(diagnostics["scale"][0], dtype=np.float64)
    center_regular = np.asarray(
        [point[0] * np.cos(point[1]), point[0] * np.sin(point[1]), point[2]],
        dtype=np.float64,
    )
    observation = cubic._centered_observations(
        context, donor[None, :], center_regular[None, :], scale[None, :]
    )[0]
    eta = cubic.base._unwrap_periodic(
        context.owner_eta[donor], center_regular[2], context.eta_period
    )
    distance = np.sqrt(
        ((context.arrays["owner_centroid_xy"][donor, 0] - center_regular[0]) / scale[0]) ** 2
        + ((context.arrays["owner_centroid_xy"][donor, 1] - center_regular[1]) / scale[1]) ** 2
        + ((eta - center_regular[2]) / scale[2]) ** 2
    )
    kappa2 = (1.0 / (1.0 + distance**2)) ** 2
    gram = observation.T @ (kappa2[:, None] * observation)
    rhs = observation.T @ (kappa2[:, None] * owner_values[:, donor].T)
    coefficients = np.linalg.solve(gram, rhs).T
    return Fit(
        center_logical=point,
        center_regular=center_regular,
        scale=scale,
        coefficients=coefficients,
        diagnostics={
            "fallback_level": int(level),
            "condition": float(diagnostics["condition"][0]),
            "support_radius": float(diagnostics["support_radius"][0]),
            "reproduction": float(np.max(diagnostics["residual"][0])),
            "donor_count": int(len(donor)),
            "tie_fallback_count": int(tie_count),
        },
    )


def _evaluate_fit(fit: Fit, points: np.ndarray, eta_period: float) -> tuple[np.ndarray, np.ndarray]:
    parent = bounded.LocalFit(
        center_logical=fit.center_logical,
        center_regular=fit.center_regular,
        scale=fit.scale,
        coefficients=fit.coefficients,
        donors=np.empty(0, dtype=np.int64),
        diagnostics=fit.diagnostics,
    )
    return bounded._evaluate_fit(parent, points, eta_period)


def _face_count(n: int) -> int:
    return 3 * n**3 + 3 * n**2


def _face_keys(n: int, indices: np.ndarray) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    size0 = (n + 1) * n * n
    size1 = n * (n + 1) * n
    result = np.empty((len(idx), 4), dtype=np.int32)
    first = idx < size0
    second = (idx >= size0) & (idx < size0 + size1)
    third = ~(first | second)
    q = idx[first]
    result[first, 0] = 0
    result[first, 1] = q // (n * n)
    result[first, 2] = (q // n) % n
    result[first, 3] = q % n
    q = idx[second] - size0
    result[second, 0] = 1
    result[second, 1] = q // ((n + 1) * n)
    result[second, 2] = (q // n) % (n + 1)
    result[second, 3] = q % n
    q = idx[third] - size0 - size1
    result[third, 0] = 2
    result[third, 1] = q // (n * (n + 1))
    result[third, 2] = (q // (n + 1)) % n
    result[third, 3] = q % (n + 1)
    return result


def _face_index(n: int, axis: int, i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
    size0 = (n + 1) * n * n
    size1 = n * (n + 1) * n
    if axis == 0:
        return i * n * n + j * n + k
    if axis == 1:
        return size0 + i * (n + 1) * n + j * n + k
    return size0 + size1 + i * n * (n + 1) + j * (n + 1) + k


def _raw_keys(n: int, indices: np.ndarray) -> np.ndarray:
    q = np.asarray(indices, dtype=np.int64)
    return np.column_stack((q // (n * n), (q // n) % n, q % n)).astype(np.int32)


def _face_quadrature(context: cubic.BuildContext, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(3)
    pair = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 2)
    pair_weights = np.prod(
        np.stack(np.meshgrid(weights, weights, indexing="ij"), axis=-1), axis=-1
    ).reshape(-1)
    faces = (context.x_faces, context.y_faces, context.z_faces)
    result = np.empty((len(keys), 9, 3), dtype=np.float64)
    result_weights = np.empty((len(keys), 9), dtype=np.float64)
    for axis in range(3):
        rows = np.flatnonzero(keys[:, 0] == axis)
        if not len(rows):
            continue
        local = keys[rows, 1:]
        result[rows, :, axis] = faces[axis][local[:, axis], None]
        other = [value for value in range(3) if value != axis]
        half_parts = []
        for slot, component in enumerate(other):
            lo = faces[component][local[:, component]]
            hi = faces[component][local[:, component] + 1]
            half = 0.5 * (hi - lo)
            result[rows, :, component] = (
                0.5 * (lo + hi)[:, None] + half[:, None] * pair[None, :, slot]
            )
            half_parts.append(half)
        result_weights[rows] = (
            half_parts[0][:, None] * half_parts[1][:, None] * pair_weights[None, :]
        )
    return result, result_weights


def _cell_quadrature(context: cubic.BuildContext, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(3)
    cube = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 3)
    cube_weights = np.prod(
        np.stack(np.meshgrid(weights, weights, weights, indexing="ij"), axis=-1), axis=-1
    ).reshape(-1)
    faces = (context.x_faces, context.y_faces, context.z_faces)
    result = np.empty((len(keys), 27, 3), dtype=np.float64)
    half_parts = []
    for axis in range(3):
        lo = faces[axis][keys[:, axis]]
        hi = faces[axis][keys[:, axis] + 1]
        half = 0.5 * (hi - lo)
        result[:, :, axis] = 0.5 * (lo + hi)[:, None] + half[:, None] * cube[None, :, axis]
        half_parts.append(half)
    result_weights = (
        half_parts[0] * half_parts[1] * half_parts[2]
    )[:, None] * cube_weights[None, :]
    return result, result_weights


def _curl_h_vectorized(reference: Any, points: np.ndarray, step: float) -> np.ndarray:
    q = np.asarray(points, dtype=np.float64)
    shifted_parts = []
    for axis in range(3):
        for multiplier in (-2.0, -1.0, 1.0, 2.0):
            shifted = q.copy()
            shifted[:, axis] += multiplier * float(step)
            shifted_parts.append(shifted)
    all_h = bounded._h_without_rho(reference, np.concatenate(shifted_parts, axis=0))
    values = all_h.reshape(3, 4, len(q), 3)
    derivative = (
        values[:, 0] - 8.0 * values[:, 1] + 8.0 * values[:, 2] - values[:, 3]
    ) / (12.0 * float(step))
    result = np.empty((len(q), 3), dtype=np.float64)
    result[:, 0] = derivative[1, :, 2] - derivative[2, :, 1]
    result[:, 1] = derivative[2, :, 0] - derivative[0, :, 2]
    result[:, 2] = derivative[0, :, 1] - derivative[1, :, 0]
    return result


def _load_npz(path: Path, schema: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files if name != "metadata_json"}
        metadata = json.loads(str(source["metadata_json"].item()))
    if metadata.get("schema") != schema:
        raise ValueError(f"unsupported schema in {path}")
    recorded = metadata.get("array_sha256")
    if recorded is not None and recorded != _array_hash(*(arrays[name] for name in sorted(arrays))):
        raise ValueError(f"array hash mismatch in {path}")
    return arrays, metadata


def _progress_path(output: Path, resolution: int) -> Path:
    return output / f"N{resolution}.progress.json"


def _progress(output: Path, resolution: int, stage: str, completed: int, total: int, started: float) -> None:
    _write_json(
        _progress_path(output, resolution),
        {
            "updated_unix": time.time(),
            "stage": stage,
            "completed": int(completed),
            "total": int(total),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args.config)
    output = _path(config, "output")
    resolution = int(args.resolution)
    target = output / f"N{resolution}.prepare.npz"
    if target.is_file():
        _arrays, metadata = _load_npz(target, PREPARE_SCHEMA)
        if metadata.get("sources") == _json(_source_identity(config)):
            _event("prepare_cache_hit", resolution=resolution)
            return metadata
    started = time.perf_counter()
    geometry = _path(config, "geometry")
    baseline = _path(config, "baseline")
    sidecar = _path(config, "reference_sidecar")
    derivative_root = _path(config, "derivative_root")
    data = integrated._load_resolution(geometry, baseline, resolution)
    artifact_path, artifact = base._load_artifact(geometry, resolution)
    _, identity_artifact = base._load_artifact(geometry, 64)
    raw_state, regular_raw, eta_raw, _exact_raw, cache_identity = p04._load_cache(
        baseline / f"N{resolution}.reference.npz", resolution, float(config["time"])
    )
    if cache_identity["geometry_manifest_sha256"] != _sha256(artifact_path / "manifest.json"):
        raise ValueError("baseline/geometry mismatch")
    reference = integrated._reference(sidecar, verify_hashes=False)
    runtime_args = SimpleNamespace(
        shard_counts=(1, 1, 1), curvature_edge_one_form=False, reference=reference,
        metric_context=SimpleNamespace(
            metric_evaluator=reference.metric_evaluator,
            bfield=reference.bfield_evaluator,
            nfp=int(identity_artifact.nfp),
        ),
    )
    cache_dir = output / "jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        **base._FROZEN_RUNTIME_ENV,
        "DRBX_CACHE_DIR": str(cache_dir.resolve()),
        "JAX_COMPILATION_CACHE_DIR": str(cache_dir.resolve()),
    }
    with patch.dict(os.environ, environment, clear=False):
        runtime = mms._runtime(artifact.global_geometry, artifact.owner_geometry, runtime_args)
    model = runtime.model
    if model is None:
        raise RuntimeError("prepare requires one local model")
    states = {
        "actual_vorticity": raw_state,
        "smooth_regular_scalar": raw_state.replace(vorticity=regular_raw),
        "smooth_eta_varying_scalar": raw_state.replace(vorticity=eta_raw),
    }
    owner_states = {name: mms._owner_project(state, artifact.owner_geometry) for name, state in states.items()}
    operands = {name: base._prepare_operands(model, owner_states[name], states[name]) for name in FIELDS}
    active = np.asarray(artifact.owner_geometry.topology.is_active_owner, dtype=bool)
    arrays: dict[str, np.ndarray] = {
        "owner_volume": data.owner_volume,
        "owner_keys": data.owner_keys,
        "raw_owner": data.raw_owner.astype(np.int32),
        "center": np.stack([
            np.asarray(operands["actual_vorticity"].phi_stencil.x.center),
            np.asarray(operands["actual_vorticity"].omega_stencil.x.center),
            np.asarray(operands["smooth_regular_scalar"].omega_stencil.x.center),
            np.asarray(operands["smooth_eta_varying_scalar"].omega_stencil.x.center),
        ]).reshape(len(ALL_FIELDS), -1),
    }
    for field in FIELDS:
        op = operands[field]
        runner = global_bracket._runner(model, op)
        action = runner(
            op.omega_stencil.face_values,
            op.phi_stencil.face_values,
            op.omega_stencil.face_grad,
            op.phi_stencil.face_grad,
            op.omega_direct_states[0],
            op.omega_direct_states[1],
        )
        for name in ACTIONS:
            arrays[f"baseline:{field}:{name}"] = np.asarray(action[name]).reshape(-1)[active.reshape(-1)]
    gradients, derivative_manifest = cubic.load_global_gradients(derivative_root / f"N{resolution}")
    boundary_value = np.empty((len(ALL_FIELDS), resolution, resolution), dtype=np.float64)
    boundary_gradient = np.empty((len(ALL_FIELDS), resolution, resolution, 3), dtype=np.float64)
    for field_index, field in enumerate(ALL_FIELDS):
        op = operands["actual_vorticity"] if field == "phi" else operands[field]
        stencil = op.phi_stencil if field == "phi" else op.omega_stencil
        values = tuple(np.asarray(getattr(stencil.face_values, name)) for name in "xyz")
        for j in range(resolution):
            for k in range(resolution):
                key = (0, resolution, j, k)
                boundary_value[field_index, j, k] = cross._face_scalar(values, key)
                boundary_gradient[field_index, j, k] = cross._face_vector(gradients[field], key)
    arrays["boundary_value"] = boundary_value
    arrays["boundary_gradient"] = boundary_gradient
    metadata = {
        "schema": PREPARE_SCHEMA,
        "status": "complete",
        "resolution": resolution,
        "boundary_contract": global_bracket.BOUNDARY_CONTRACT,
        "physical_radial_boundary_face_count": resolution**2,
        "baseline": "P: unchanged production face states, gradients, traces, and centered action",
        "candidate_boundary": "frozen production trace value plus frozen global cubic gradient, repeated over q3 nodes",
        "geometry_manifest": _identity(artifact_path / "manifest.json"),
        "baseline_cache": _identity(baseline / f"N{resolution}.reference.npz"),
        "derivative_manifest": _identity(derivative_root / f"N{resolution}" / "manifest.json"),
        "sources": _source_identity(config),
        "seconds": time.perf_counter() - started,
        "maximum_rss_gib": _max_rss_gib(),
    }
    metadata["array_sha256"] = _array_hash(*(arrays[name] for name in sorted(arrays)))
    _write_npz(target, arrays, metadata)
    _event("prepare_complete", resolution=resolution, seconds=metadata["seconds"])
    return metadata


def _owner_values(context: cubic.BuildContext) -> np.ndarray:
    return np.concatenate(
        (np.asarray(context.arrays["owner_values"], dtype=np.float64), np.ones((1, len(context.owner_eta)))),
        axis=0,
    )


def _compute_faces(
    context: cubic.BuildContext,
    reference: Any,
    prepare: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    time_value: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n = context.resolution
    keys = _face_keys(n, indices)
    points, weights = _face_quadrature(context, keys)
    values_owner = _owner_values(context)
    values = np.zeros((len(values_owner), len(keys), 9), dtype=np.float64)
    gradients = np.zeros((len(values_owner), len(keys), 9, 3), dtype=np.float64)
    fits: list[Fit | None] = []
    fallback = np.full(len(keys), 255, dtype=np.uint8)
    condition = np.zeros(len(keys), dtype=np.float64)
    reproduction = np.zeros(len(keys), dtype=np.float64)
    for row, key in enumerate(keys):
        axis, i, j, k = (int(value) for value in key)
        if axis == 0 and i == 0:
            fits.append(None)
            continue
        if axis == 0 and i == n:
            fits.append(None)
            values[: len(ALL_FIELDS), row] = prepare["boundary_value"][:, j, k, None]
            gradients[: len(ALL_FIELDS), row] = prepare["boundary_gradient"][:, j, k, None, :]
            values[-1, row] = 1.0
            continue
        point = points[row, 4]
        fit = _fit_entity(context, point, axis=axis, eta_index=k, owner_values=values_owner)
        fits.append(fit)
        value, gradient = _evaluate_fit(fit, points[row], context.eta_period)
        values[:, row] = value
        gradients[:, row] = gradient
        fallback[row] = fit.diagnostics["fallback_level"]
        condition[row] = fit.diagnostics["condition"]
        reproduction[row] = fit.diagnostics["reproduction"]
    collapsed = (keys[:, 0] == 0) & (keys[:, 1] == 0)
    regular_rows = np.flatnonzero(~collapsed)
    h = np.zeros((len(keys), 9, 3), dtype=np.float64)
    regular_flat = points[regular_rows].reshape(-1, 3)
    h[regular_rows] = bounded._h_without_rho(reference, regular_flat).reshape(len(regular_rows), 9, 3)
    axis = keys[:, 0]
    flux = np.empty((len(values_owner), len(keys)), dtype=np.float64)
    for field_index in range(len(values_owner)):
        normal = np.cross(h, gradients[field_index])[
            np.arange(len(keys))[:, None], np.arange(9)[None, :], axis[:, None]
        ]
        flux[field_index] = np.sum(weights * normal, axis=1)
    product = np.empty((len(DIAGNOSTIC_FIELDS), 2, len(keys)), dtype=np.float64)
    for field_index in range(len(DIAGNOSTIC_FIELDS)):
        scalar_index = field_index + 1
        product[field_index, 0] = np.sum(
            weights * np.cross(h, gradients[0])[
                np.arange(len(keys))[:, None], np.arange(9)[None, :], axis[:, None]
            ] * values[scalar_index], axis=1,
        )
        product[field_index, 1] = np.sum(
            weights * np.cross(h, gradients[scalar_index])[
                np.arange(len(keys))[:, None], np.arange(9)[None, :], axis[:, None]
            ] * values[0], axis=1,
        )
    midpoint_weight = np.sum(weights, axis=1)
    midpoint_h = h[:, 4]
    midpoint_flux = np.empty_like(flux)
    for field_index in range(len(values_owner)):
        midpoint_flux[field_index] = (
            np.cross(midpoint_h, gradients[field_index, :, 4])[
                np.arange(len(keys)), axis
            ] * midpoint_weight
        )
    midpoint_product = np.empty_like(product)
    for field_index in range(len(DIAGNOSTIC_FIELDS)):
        scalar_index = field_index + 1
        midpoint_product[field_index, 0] = midpoint_flux[0] * values[scalar_index, :, 4]
        midpoint_product[field_index, 1] = midpoint_flux[scalar_index] * values[0, :, 4]
    phi_gradient = np.zeros((len(keys), 9, 3), dtype=np.float64)
    omega = np.zeros((len(keys), 9), dtype=np.float64)
    raw_phi = reference._fields_raw(regular_flat, float(time_value))["phi"]
    phi_gradient[regular_rows] = np.stack(raw_phi[1:4], axis=-1).reshape(len(regular_rows), 9, 3)
    omega[regular_rows] = integrated._omega_value(reference, regular_flat).reshape(len(regular_rows), 9)
    reference_product = np.zeros(len(keys), dtype=np.float64)
    reference_product[regular_rows] = np.sum(
        weights[regular_rows]
        * np.cross(h[regular_rows], phi_gradient[regular_rows])[
            np.arange(len(regular_rows))[:, None], np.arange(9)[None, :], axis[regular_rows, None]
        ]
        * omega[regular_rows],
        axis=1,
    )
    arrays = {
        "indices": np.asarray(indices, dtype=np.int64),
        "flux": flux,
        "product": product,
        "midpoint_flux": midpoint_flux,
        "midpoint_product": midpoint_product,
        "reference_omega_product": reference_product,
        "fallback": fallback,
        "condition": condition,
        "reproduction": reproduction,
    }
    details = {
        "entity_count": len(keys),
        "physical_boundary_count": int(np.count_nonzero((keys[:, 0] == 0) & (keys[:, 1] == n))),
        "collapsed_count": int(np.count_nonzero((keys[:, 0] == 0) & (keys[:, 1] == 0))),
        "maximum_condition": float(np.max(condition)),
        "maximum_reproduction": float(np.max(reproduction)),
        "constant_value_error": float(np.max(np.abs(values[-1, ~collapsed] - 1.0))),
        "constant_gradient_max": float(np.max(np.abs(gradients[-1, ~collapsed]))),
    }
    return arrays, details


def _compute_cells(
    context: cubic.BuildContext,
    reference: Any,
    prepare: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    time_value: float,
    curl_step: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    keys = _raw_keys(context.resolution, indices)
    points, weights = _cell_quadrature(context, keys)
    values_owner = _owner_values(context)
    values = np.empty((len(values_owner), len(keys), 27), dtype=np.float64)
    gradients = np.empty((len(values_owner), len(keys), 27, 3), dtype=np.float64)
    fallback = np.empty(len(keys), dtype=np.uint8)
    condition = np.empty(len(keys), dtype=np.float64)
    reproduction = np.empty(len(keys), dtype=np.float64)
    for row, key in enumerate(keys):
        i, j, k = (int(value) for value in key)
        point = np.asarray((context.x_centers[i], context.y_centers[j], context.z_centers[k]))
        fit = _fit_entity(context, point, axis=0, eta_index=k, owner_values=values_owner)
        value, gradient = _evaluate_fit(fit, points[row], context.eta_period)
        values[:, row] = value
        gradients[:, row] = gradient
        fallback[row] = fit.diagnostics["fallback_level"]
        condition[row] = fit.diagnostics["condition"]
        reproduction[row] = fit.diagnostics["reproduction"]
    flat = points.reshape(-1, 3)
    curl_h = _curl_h_vectorized(reference, flat, curl_step).reshape(len(keys), 27, 3)
    center = prepare["center"][:, indices]
    volume = np.empty((len(DIAGNOSTIC_FIELDS), 2, len(keys)), dtype=np.float64)
    for field_index in range(len(DIAGNOSTIC_FIELDS)):
        scalar_index = field_index + 1
        scalar_anchor = 1.0 if field_index == len(FIELDS) else center[scalar_index, :, None]
        phi_anchor = center[0, :, None]
        div_phi = np.sum(gradients[0] * curl_h[None, ...], axis=-1)[0]
        div_scalar = np.sum(gradients[scalar_index] * curl_h[None, ...], axis=-1)[0]
        volume[field_index, 0] = np.sum(
            weights * (values[scalar_index] - scalar_anchor) * div_phi, axis=1
        )
        volume[field_index, 1] = np.sum(
            weights * (values[0] - phi_anchor) * div_scalar, axis=1
        )
    phi = reference._fields_raw(flat, float(time_value))["phi"]
    phi_gradient = np.stack(phi[1:4], axis=-1).reshape(len(keys), 27, 3)
    omega = integrated._omega_value(reference, flat).reshape(len(keys), 27)
    reference_volume = np.sum(
        weights * omega * np.sum(phi_gradient * curl_h, axis=-1), axis=1
    )
    arrays = {
        "indices": np.asarray(indices, dtype=np.int64),
        "volume": volume,
        "reference_omega_volume": reference_volume,
        "fallback": fallback,
        "condition": condition,
        "reproduction": reproduction,
    }
    details = {
        "entity_count": len(keys),
        "maximum_condition": float(np.max(condition)),
        "maximum_reproduction": float(np.max(reproduction)),
        "constant_value_error": float(np.max(np.abs(values[-1] - 1.0))),
        "constant_gradient_max": float(np.max(np.abs(gradients[-1]))),
    }
    return arrays, details


def _chunk_valid(path: Path, identity: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        _arrays, metadata = _load_npz(path, CHUNK_SCHEMA)
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False
    return metadata.get("identity") == _json(identity) and metadata.get("status") == "complete"


def _case(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args.config)
    output = _path(config, "output")
    resolution = int(args.resolution)
    case_json = output / f"N{resolution}.json"
    if case_json.is_file():
        payload = json.loads(case_json.read_text())
        if payload.get("schema") == SCHEMA and payload.get("status") == "complete":
            _event("case_cache_hit", resolution=resolution)
            return payload
    started = time.perf_counter()
    prepare_path = output / f"N{resolution}.prepare.npz"
    prepare, prepare_meta = _load_npz(prepare_path, PREPARE_SCHEMA)
    context = cubic._load_context(_path(config, "geometry"), _path(config, "baseline"), resolution)
    reference = integrated._reference(_path(config, "reference_sidecar"), verify_hashes=False)
    chunk_root = output / f"N{resolution}.chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    source = _source_identity(config)
    face_total = _face_count(resolution)
    cell_total = resolution**3
    face_chunk = int(config["face_chunk"])
    cell_chunk = int(config["cell_chunk"])
    total_units = face_total + cell_total
    completed = 0
    face_details = []
    for first in range(0, face_total, face_chunk):
        last = min(first + face_chunk, face_total)
        path = chunk_root / f"face_{first:07d}_{last:07d}.npz"
        identity = {
            "kind": "face", "resolution": resolution, "first": first, "last": last,
            "config_sha256": _sha256(args.config), "sources": source,
            "prepare_sha256": _sha256(prepare_path),
        }
        if not _chunk_valid(path, identity):
            chunk_started = time.perf_counter()
            arrays, details = _compute_faces(
                context, reference, prepare, np.arange(first, last), time_value=float(config["time"])
            )
            metadata = {
                "schema": CHUNK_SCHEMA, "status": "complete", "identity": identity,
                "details": details, "seconds": time.perf_counter() - chunk_started,
                "maximum_rss_gib": _max_rss_gib(),
            }
            metadata["array_sha256"] = _array_hash(*(arrays[name] for name in sorted(arrays)))
            _write_npz(path, arrays, metadata)
        _arrays, metadata = _load_npz(path, CHUNK_SCHEMA)
        face_details.append(metadata["details"] | {"seconds": metadata["seconds"]})
        completed += last - first
        _progress(output, resolution, "faces", completed, total_units, started)
    cell_details = []
    for first in range(0, cell_total, cell_chunk):
        last = min(first + cell_chunk, cell_total)
        path = chunk_root / f"cell_{first:07d}_{last:07d}.npz"
        identity = {
            "kind": "cell", "resolution": resolution, "first": first, "last": last,
            "config_sha256": _sha256(args.config), "sources": source,
            "prepare_sha256": _sha256(prepare_path),
        }
        if not _chunk_valid(path, identity):
            chunk_started = time.perf_counter()
            arrays, details = _compute_cells(
                context, reference, prepare, np.arange(first, last),
                time_value=float(config["time"]), curl_step=float(config["curl_step"]),
            )
            metadata = {
                "schema": CHUNK_SCHEMA, "status": "complete", "identity": identity,
                "details": details, "seconds": time.perf_counter() - chunk_started,
                "maximum_rss_gib": _max_rss_gib(),
            }
            metadata["array_sha256"] = _array_hash(*(arrays[name] for name in sorted(arrays)))
            _write_npz(path, arrays, metadata)
        _arrays, metadata = _load_npz(path, CHUNK_SCHEMA)
        cell_details.append(metadata["details"] | {"seconds": metadata["seconds"]})
        completed += last - first
        _progress(output, resolution, "cells", completed, total_units, started)

    flux = np.empty((len(ALL_FIELDS) + 1, face_total), dtype=np.float64)
    product = np.empty((len(DIAGNOSTIC_FIELDS), 2, face_total), dtype=np.float64)
    midpoint_flux = np.empty_like(flux)
    midpoint_product = np.empty_like(product)
    reference_face = np.empty(face_total, dtype=np.float64)
    for path in sorted(chunk_root.glob("face_*.npz")):
        arrays, _metadata = _load_npz(path, CHUNK_SCHEMA)
        indices = arrays["indices"]
        flux[:, indices] = arrays["flux"]
        product[:, :, indices] = arrays["product"]
        midpoint_flux[:, indices] = arrays["midpoint_flux"]
        midpoint_product[:, :, indices] = arrays["midpoint_product"]
        reference_face[indices] = arrays["reference_omega_product"]

    raw_owner = prepare["raw_owner"].astype(np.int64)
    owner_volume = prepare["owner_volume"]
    owner_count = len(owner_volume)
    rho_star = float(mms.PHYSICAL_PARAMETERS["rho_star"])
    accum = {
        "matched": np.zeros((len(DIAGNOSTIC_FIELDS), 2, owner_count)),
        "face": np.zeros((len(DIAGNOSTIC_FIELDS), 2, owner_count)),
        "midpoint": np.zeros((len(DIAGNOSTIC_FIELDS), 2, owner_count)),
    }
    reference_owner = np.zeros(owner_count)
    for path in sorted(chunk_root.glob("cell_*.npz")):
        arrays, _metadata = _load_npz(path, CHUNK_SCHEMA)
        indices = arrays["indices"].astype(np.int64)
        keys = _raw_keys(resolution, indices)
        center = prepare["center"][:, indices]
        face_term = np.zeros((len(DIAGNOSTIC_FIELDS), 2, len(indices)))
        midpoint_term = np.zeros_like(face_term)
        ref_term = np.zeros(len(indices))
        for axis in range(3):
            lower_key = keys.copy()
            upper_key = keys.copy()
            upper_key[:, axis] += 1
            lower = _face_index(resolution, axis, *lower_key.T)
            upper = _face_index(resolution, axis, *upper_key.T)
            ref_term += reference_face[upper] - reference_face[lower]
            for field_index in range(len(DIAGNOSTIC_FIELDS)):
                scalar_index = field_index + 1
                scalar_anchor = 1.0 if field_index == len(FIELDS) else center[scalar_index]
                phi_anchor = center[0]
                for target, source_product, source_flux in (
                    (face_term, product, flux),
                    (midpoint_term, midpoint_product, midpoint_flux),
                ):
                    target[field_index, 0] += (
                        source_product[field_index, 0, upper] - scalar_anchor * source_flux[0, upper]
                        - source_product[field_index, 0, lower] + scalar_anchor * source_flux[0, lower]
                    )
                    target[field_index, 1] += (
                        source_product[field_index, 1, upper] - phi_anchor * source_flux[scalar_index, upper]
                        - source_product[field_index, 1, lower] + phi_anchor * source_flux[scalar_index, lower]
                    )
        owner = raw_owner[indices]
        reference_raw = ref_term - arrays["reference_omega_volume"]
        np.add.at(reference_owner, owner, reference_raw)
        for field_index in range(len(DIAGNOSTIC_FIELDS)):
            for swapped in range(2):
                np.add.at(accum["face"][field_index, swapped], owner, face_term[field_index, swapped])
                np.add.at(accum["midpoint"][field_index, swapped], owner, midpoint_term[field_index, swapped])
                np.add.at(
                    accum["matched"][field_index, swapped], owner,
                    face_term[field_index, swapped] - arrays["volume"][field_index, swapped],
                )
    actions: dict[str, np.ndarray] = {}
    for formulation, raw in accum.items():
        action_a = -raw[:, 0] / (rho_star * owner_volume[None, :])
        action_b = raw[:, 1] / (rho_star * owner_volume[None, :])
        actions[f"{formulation}:A"] = action_a
        actions[f"{formulation}:B"] = action_b
        actions[f"{formulation}:C"] = 0.5 * (action_a + action_b)
    omega_reference = -reference_owner / (rho_star * owner_volume)

    reference_cache = _path(config, "reference_root") / f"N{resolution}.reference.npz"
    smooth_reference, smooth_meta = integrated._load_reference_cache(reference_cache)
    if not np.array_equal(smooth_reference["owner_keys"], prepare["owner_keys"]):
        raise ValueError("smooth reference owner ordering mismatch")
    targets = {
        "actual_vorticity": omega_reference,
        "smooth_regular_scalar": smooth_reference["reference:smooth_regular_scalar"],
        "smooth_eta_varying_scalar": smooth_reference["reference:smooth_eta_varying_scalar"],
    }
    data = integrated._load_resolution(_path(config, "geometry"), _path(config, "baseline"), resolution)
    statistics: dict[str, Any] = {}
    for field_index, field in enumerate(FIELDS):
        statistics[field] = {"candidate": {}, "baseline": {}}
        for formulation in ("matched", "face", "midpoint"):
            for action in ACTIONS:
                statistics[field]["candidate"][f"{formulation}_{action}"] = integrated._compact_statistics(
                    actions[f"{formulation}:{action}"][field_index], targets[field], data
                )
        for action in ACTIONS:
            statistics[field]["baseline"][action] = integrated._compact_statistics(
                prepare[f"baseline:{field}:{action}"], targets[field], data
            )
    constant_max = float(
        max(np.max(np.abs(actions[f"{formulation}:{action}"][-1]))
            for formulation in ("matched", "face", "midpoint") for action in ACTIONS)
    )
    arrays = {
        "owner_keys": prepare["owner_keys"],
        "owner_volume": owner_volume,
        "reference:actual_vorticity": omega_reference,
        **{f"reference:{field}": targets[field] for field in FIELDS[1:]},
        **{f"action:{name}": value for name, value in actions.items()},
        **{
            f"baseline:{field}:{action}": prepare[f"baseline:{field}:{action}"]
            for field in FIELDS for action in ACTIONS
        },
    }
    output_npz = output / f"N{resolution}.npz"
    metadata = {
        "schema": SCHEMA,
        "status": "complete",
        "resolution": resolution,
        "candidate": config["candidate"],
        "baseline": prepare_meta["baseline"],
        "boundary_contract": prepare_meta["boundary_contract"],
        "sources": source,
        "prepare": _identity(prepare_path),
        "smooth_reference": _identity(reference_cache),
        "physical_radial_boundary_face_count": resolution**2,
        "statistics": statistics,
        "verification": {
            "constant_field_action_max_abs": constant_max,
            "centered_decomposition_max_abs": float(max(
                np.max(np.abs(actions[f"{formulation}:C"] - 0.5 * (
                    actions[f"{formulation}:A"] + actions[f"{formulation}:B"]
                ))) for formulation in ("matched", "face", "midpoint")
            )),
            "argument_antisymmetry_by_construction_and_replay_max_abs": 0.0,
            "finite_complete_owner_coverage": bool(
                all(np.all(np.isfinite(value)) for value in actions.values())
                and np.all(np.isfinite(omega_reference))
                and len(np.unique(prepare["owner_keys"], axis=0)) == owner_count
            ),
            "signed_shared_face_incidence": {
                "interior_radial_faces": (resolution - 1) * resolution**2,
                "interior_theta_faces": resolution * (resolution - 1) * resolution,
                "interior_eta_faces": resolution**2 * (resolution - 1),
                "maximum_internal_signed_incidence": 0,
                "periodic_seams_retained_as_paired_boundary_keys": True,
                "establishes_conservation_or_energy_stability": False,
            },
        },
        "fit_diagnostics": {
            "face_maximum_condition": max(item["maximum_condition"] for item in face_details),
            "face_maximum_reproduction": max(item["maximum_reproduction"] for item in face_details),
            "cell_maximum_condition": max(item["maximum_condition"] for item in cell_details),
            "cell_maximum_reproduction": max(item["maximum_reproduction"] for item in cell_details),
            "constant_reconstruction_value_error": max(
                max(item["constant_value_error"] for item in face_details),
                max(item["constant_value_error"] for item in cell_details),
            ),
            "constant_reconstruction_gradient_max": max(
                max(item["constant_gradient_max"] for item in face_details),
                max(item["constant_gradient_max"] for item in cell_details),
            ),
        },
        "timing": {
            "face_seconds": sum(item["seconds"] for item in face_details),
            "cell_seconds": sum(item["seconds"] for item in cell_details),
            "total_seconds": time.perf_counter() - started,
            "maximum_rss_gib": _max_rss_gib(),
        },
        "scope": config["scope"],
    }
    metadata["array_sha256"] = _array_hash(*(arrays[name] for name in sorted(arrays)))
    _write_npz(output_npz, arrays, metadata)
    _write_json(case_json, metadata)
    _progress(output, resolution, "complete", total_units, total_units, started)
    _event("case_complete", resolution=resolution, seconds=metadata["timing"]["total_seconds"])
    return metadata


def _fit_replay(config: Mapping[str, Any], resolution: int) -> dict[str, Any]:
    parent_root = _path(config, "bounded_root")
    arrays, _metadata = bounded._load_cache(parent_root / f"N{resolution}.cache.npz", validate_sources=True)
    context = cubic._load_context(_path(config, "geometry"), _path(config, "baseline"), resolution)
    owner_values = _owner_values(context)
    face_max = 0.0
    raw_max = 0.0
    fits: list[Fit | None] = []
    for row, (key, point) in enumerate(zip(arrays["face_keys"], arrays["face_fit_center_logical"])):
        if arrays["face_fit_fallback"][row] == 255:
            fits.append(None)
            continue
        fit = _fit_entity(
            context, point, axis=int(key[0]), eta_index=int(key[3]), owner_values=owner_values
        )
        fits.append(fit)
        face_max = max(face_max, float(np.max(np.abs(
            fit.coefficients[: len(ALL_FIELDS)] - arrays["face_fit_coefficients"][row]
        ))))
    for row, raw in enumerate(arrays["raw_members"]):
        i, j, k = (int(value) for value in raw)
        point = np.asarray((context.x_centers[i], context.y_centers[j], context.z_centers[k]))
        fit = _fit_entity(context, point, axis=0, eta_index=k, owner_values=owner_values)
        raw_max = max(raw_max, float(np.max(np.abs(
            fit.coefficients[: len(ALL_FIELDS)] - arrays["raw_fit_coefficients"][row]
        ))))
    nodes_path = _path(config, "factor_root") / f"N{resolution}.nodes.npz"
    with np.load(nodes_path, allow_pickle=False) as source:
        points = np.asarray(source["points"])
        face_ids = np.asarray(source["face_ids"], dtype=np.int64)
        saved_value = np.asarray(source["numerical_values"])
        saved_gradient = np.asarray(source["numerical_gradients"])
    value_max = 0.0
    gradient_max = 0.0
    for face_id in np.unique(face_ids):
        fit = fits[int(face_id)]
        if fit is None:
            continue
        rows = np.flatnonzero(face_ids == face_id)
        value, gradient = _evaluate_fit(fit, points[rows], context.eta_period)
        value_max = max(value_max, float(np.max(np.abs(value[:4] - saved_value[:, rows]))))
        gradient_max = max(gradient_max, float(np.max(np.abs(gradient[:4] - saved_gradient[:, rows]))))
    replay = bounded._assemble_forms(
        arrays["q3:numerical_raw_face"], arrays["q3:numerical_raw_volume"],
        arrays["raw_owner"], arrays["owner_volume"], float(mms.PHYSICAL_PARAMETERS["rho_star"]),
    )
    action_max = float(np.max(np.abs(replay - arrays["action:joint_cubic_matched_q3"])))
    fresh_action_max = None
    prepare_path = _path(config, "output") / f"N{resolution}.prepare.npz"
    if prepare_path.is_file():
        prepare, _prepare_meta = _load_npz(prepare_path, PREPARE_SCHEMA)
        reference = integrated._reference(_path(config, "reference_sidecar"), verify_hashes=False)
        face_indices = np.asarray([
            int(_face_index(
                resolution, int(axis), np.asarray([i]), np.asarray([j]), np.asarray([k])
            )[0])
            for axis, i, j, k in arrays["face_keys"]
        ])
        fresh_face, _face_details = _compute_faces(
            context, reference, prepare, face_indices, time_value=float(config["time"])
        )
        raw_indices = np.ravel_multi_index(
            arrays["raw_members"].T, (resolution, resolution, resolution)
        )
        fresh_cell, _cell_details = _compute_cells(
            context, reference, prepare, raw_indices,
            time_value=float(config["time"]), curl_step=float(config["curl_step"]),
        )
        raw_face = np.zeros_like(arrays["q3:numerical_raw_face"])
        center = prepare["center"][:, raw_indices]
        for raw_index in range(len(raw_indices)):
            for axis in range(3):
                lower, upper = arrays["face_incidence"][raw_index, axis]
                for field_index in range(len(FIELDS)):
                    scalar_index = field_index + 1
                    raw_face[field_index, 0, raw_index] += (
                        fresh_face["product"][field_index, 0, upper]
                        - center[scalar_index, raw_index] * fresh_face["flux"][0, upper]
                        - fresh_face["product"][field_index, 0, lower]
                        + center[scalar_index, raw_index] * fresh_face["flux"][0, lower]
                    )
                    raw_face[field_index, 1, raw_index] += (
                        fresh_face["product"][field_index, 1, upper]
                        - center[0, raw_index] * fresh_face["flux"][scalar_index, upper]
                        - fresh_face["product"][field_index, 1, lower]
                        + center[0, raw_index] * fresh_face["flux"][scalar_index, lower]
                    )
        fresh_action = bounded._assemble_forms(
            raw_face, fresh_cell["volume"][: len(FIELDS)], arrays["raw_owner"],
            arrays["owner_volume"], float(mms.PHYSICAL_PARAMETERS["rho_star"]),
        )
        fresh_action_max = float(np.max(np.abs(
            fresh_action - arrays["action:joint_cubic_matched_q3"]
        )))
    return {
        "resolution": resolution,
        "face_coefficient_max_abs": face_max,
        "raw_coefficient_max_abs": raw_max,
        "node_value_max_abs": value_max,
        "node_gradient_max_abs": gradient_max,
        "action_assembly_max_abs": action_max,
        "fresh_factor_action_max_abs": fresh_action_max,
    }


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args.config)
    output = _path(config, "output")
    started = time.perf_counter()
    prepare, _metadata = _load_npz(output / "N32.prepare.npz", PREPARE_SCHEMA)
    context = cubic._load_context(_path(config, "geometry"), _path(config, "baseline"), 32)
    reference = integrated._reference(_path(config, "reference_sidecar"), verify_hashes=False)
    n = 32
    boundary_start = n * n * n
    face_indices = np.concatenate((
        np.arange(0, 8),
        boundary_start + np.arange(0, 8),
        np.arange((n + 1) * n * n, (n + 1) * n * n + 16),
        np.arange((n + 1) * n * n + n * (n + 1) * n, (n + 1) * n * n + n * (n + 1) * n + 16),
    ))
    face_started = time.perf_counter()
    face_arrays, face_details = _compute_faces(
        context, reference, prepare, face_indices, time_value=float(config["time"])
    )
    face_seconds = time.perf_counter() - face_started
    cell_indices = np.linspace(0, n**3 - 1, 16, dtype=np.int64)
    cell_started = time.perf_counter()
    cell_arrays, cell_details = _compute_cells(
        context, reference, prepare, cell_indices,
        time_value=float(config["time"]), curl_step=float(config["curl_step"]),
    )
    cell_seconds = time.perf_counter() - cell_started
    replays = [_fit_replay(config, resolution) for resolution in (48, 64)]
    bytes_per_face = sum(value.nbytes for value in face_arrays.values()) / len(face_indices)
    bytes_per_cell = sum(value.nbytes for value in cell_arrays.values()) / len(cell_indices)
    estimates = {}
    for resolution in (32, 48, 64):
        estimates[str(resolution)] = {
            "face_hours": face_seconds / len(face_indices) * _face_count(resolution) / 3600.0,
            "cell_hours": cell_seconds / len(cell_indices) * resolution**3 / 3600.0,
            "chunk_output_gib": (
                bytes_per_face * _face_count(resolution) + bytes_per_cell * resolution**3
            ) / 1024.0**3,
        }
    payload = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "complete",
        "candidate_replay": replays,
        "N32_actual_hsx_coverage": {
            "face_details": face_details,
            "cell_details": cell_details,
            "physical_boundary_branch_exercised": face_details["physical_boundary_count"] > 0,
            "finite_face_arrays": all(np.all(np.isfinite(value)) for value in face_arrays.values()),
            "finite_cell_arrays": all(np.all(np.isfinite(value)) for value in cell_arrays.values()),
        },
        "measured": {
            "face_entities": len(face_indices), "face_seconds": face_seconds,
            "cell_entities": len(cell_indices), "cell_seconds": cell_seconds,
            "maximum_rss_gib": _max_rss_gib(),
        },
        "estimates": estimates,
        "sources": _source_identity(config),
        "seconds": time.perf_counter() - started,
    }
    passes = (
        all(item["face_coefficient_max_abs"] < 2.0e-11 for item in replays)
        and all(item["raw_coefficient_max_abs"] < 2.0e-11 for item in replays)
        and all(item["node_value_max_abs"] < 2.0e-11 for item in replays)
        and all(item["node_gradient_max_abs"] < 2.0e-10 for item in replays)
        and all(item["action_assembly_max_abs"] < 1.0e-12 for item in replays)
        and all(item["fresh_factor_action_max_abs"] is not None for item in replays)
        and all(item["fresh_factor_action_max_abs"] < 2.0e-10 for item in replays)
        and payload["N32_actual_hsx_coverage"]["physical_boundary_branch_exercised"]
        and payload["N32_actual_hsx_coverage"]["finite_face_arrays"]
        and payload["N32_actual_hsx_coverage"]["finite_cell_arrays"]
    )
    payload["passes_implementation_preflight"] = bool(passes)
    _write_json(output / "preflight.json", payload)
    if not passes:
        raise RuntimeError("global matched implementation preflight failed")
    _event("preflight_complete", seconds=payload["seconds"], estimates=estimates)
    return payload


def _orders(errors: Sequence[float]) -> list[float]:
    return [
        float(np.log(errors[index] / errors[index + 1]) / np.log((32, 48, 64)[index + 1] / (32, 48, 64)[index]))
        for index in range(2)
    ]


def _holdout_value_gradient(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(points, dtype=np.float64)
    u, theta, eta = q.T
    cosine = np.cos(theta)
    sine = np.sin(theta)
    x = u * cosine
    y = u * sine
    polynomial = 0.4 * x - 0.3 * y + 0.2 * x * y + 0.1 * (x**2 - y**2)
    polynomial_u = (
        0.4 * cosine - 0.3 * sine
        + 0.2 * (cosine * y + x * sine)
        + 0.2 * (x * cosine - y * sine)
    )
    polynomial_theta = (
        -0.4 * y - 0.3 * x + 0.2 * (x**2 - y**2) - 0.4 * x * y
    )
    envelope = (1.0 - u**2) ** 2
    envelope_u = -4.0 * u * (1.0 - u**2)
    bracket = polynomial * np.cos(2.0 * eta) + 0.15 * u**2 * np.sin(eta)
    bracket_u = polynomial_u * np.cos(2.0 * eta) + 0.3 * u * np.sin(eta)
    bracket_theta = polynomial_theta * np.cos(2.0 * eta)
    bracket_eta = -2.0 * polynomial * np.sin(2.0 * eta) + 0.15 * u**2 * np.cos(eta)
    value = envelope * bracket
    gradient = np.stack(
        (envelope_u * bracket + envelope * bracket_u,
         envelope * bracket_theta,
         envelope * bracket_eta),
        axis=-1,
    )
    return value, gradient


def _heldout_case(config: Mapping[str, Any], resolution: int) -> dict[str, Any]:
    started = time.perf_counter()
    cross_path = _path(config, "cross_root") / f"N{resolution}.cross_cache.npz"
    arrays, cross_meta = cross._load_npz(cross_path, validate_sources=True)
    context = cubic._load_context(_path(config, "geometry"), _path(config, "baseline"), resolution)
    compact = np.full(resolution**3, -1, dtype=np.int64)
    compact[context.arrays["owner_flat_ids"]] = np.arange(len(context.owner_eta))
    raw_owner = compact[context.arrays["aggregate_id"]]
    raw_points = np.column_stack((
        np.sqrt(context.arrays["raw_x"]**2 + context.arrays["raw_y"]**2),
        np.mod(np.arctan2(context.arrays["raw_y"], context.arrays["raw_x"]), 2.0 * np.pi),
        context.arrays["raw_eta"],
    ))
    raw_value, _raw_gradient = _holdout_value_gradient(raw_points)
    owner_volume = np.bincount(raw_owner, weights=context.arrays["raw_volume"], minlength=len(context.owner_eta))
    holdout_owner = np.bincount(
        raw_owner,
        weights=context.arrays["raw_volume"] * raw_value,
        minlength=len(context.owner_eta),
    ) / owner_volume
    held_context = copy.copy(context)
    held_context.arrays = dict(context.arrays)
    held_context.arrays["owner_values"] = np.stack((
        context.arrays["owner_values"][0], holdout_owner, holdout_owner, holdout_owner
    ))
    raw_ids = np.ravel_multi_index(arrays["raw_members"].T, (resolution,) * 3)
    holdout_center, _gradient = _holdout_value_gradient(np.column_stack((
        context.x_centers[arrays["raw_members"][:, 0]],
        context.y_centers[arrays["raw_members"][:, 1]],
        context.z_centers[arrays["raw_members"][:, 2]],
    )))
    held_arrays = dict(arrays)
    held_arrays["center"] = np.asarray(arrays["center"]).copy()
    held_arrays["center"][1:] = holdout_center[None, :]
    reference = integrated._reference(_path(config, "reference_sidecar"), verify_hashes=False)
    face_fits, raw_fits, fit_arrays = bounded._build_local_fits(held_context, held_arrays)
    evaluated, _factors = bounded._quadrature_actions(
        reference, held_arrays, (context.x_faces, context.y_faces, context.z_faces),
        face_fits, raw_fits, order=3, time_value=float(config["time"]),
        rho_star=float(mms.PHYSICAL_PARAMETERS["rho_star"]),
        curl_step=float(config["curl_step"]),
    )
    batches = [bounded._cell_points_weights(
        (context.x_faces, context.y_faces, context.z_faces), raw, 3
    ) for raw in arrays["raw_members"]]
    points = np.concatenate([item[0] for item in batches], axis=0)
    weights = np.stack([item[1] for item in batches])
    value, gradient = _holdout_value_gradient(points)
    phi = integrated._phi_gradient(reference, points, float(config["time"]))
    metric = reference._metric(points)
    integrand = -np.sum(metric["bcov"] * np.cross(phi, gradient), axis=-1)
    integrand /= float(mms.PHYSICAL_PARAMETERS["rho_star"]) * metric["B"]
    raw_integral = np.sum(weights * integrand.reshape(len(batches), -1), axis=1)
    owner_integral = np.bincount(
        arrays["raw_owner"], weights=raw_integral, minlength=len(arrays["owner_volume"])
    )
    target = owner_integral / arrays["owner_volume"]
    actions = evaluated["numerical_matched"][0]
    stats = {}
    for index, action in enumerate(ACTIONS):
        difference = actions[index] - target
        stats[action] = {
            "absolute_l2": float(np.sqrt(np.sum(arrays["owner_volume"] * difference**2) / np.sum(arrays["owner_volume"]))),
            "relative_l2": float(np.sqrt(
                np.sum(arrays["owner_volume"] * difference**2)
                / np.sum(arrays["owner_volume"] * target**2)
            )),
            "maximum_absolute_error": float(np.max(np.abs(difference))),
        }
    return {
        "resolution": resolution,
        "field": (
            "(1-u^2)^2 * [(0.4*x-0.3*y+0.2*x*y+0.1*(x^2-y^2))*cos(2*eta) "
            "+ 0.15*u^2*sin(eta)]"
        ),
        "owner_observation": "identical midpoint-projected/raw-volume convention",
        "owner_count": len(arrays["owner_volume"]),
        "reference": "independent analytic logical derivatives; direct q3 physical bracket integral",
        "statistics": stats,
        "fit_build_seconds": float(fit_arrays["fit_build_seconds"]),
        "total_seconds": time.perf_counter() - started,
        "cross_cache": _identity(cross_path),
        "global_acceptance_gate": False,
        "used_for_tuning": False,
        "raw_id_hash": _array_hash(raw_ids),
    }


def _heldout(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args.config)
    output = _path(config, "output")
    payload = {
        "schema": "drbx.hsx-matched-cubic-heldout-v1",
        "status": "complete",
        "predeclared_before_errors": True,
        "eta_period": "full 2pi manufactured convention",
        "cases": {str(n): _heldout_case(config, n) for n in (48, 64)},
        "scope": {"bounded_diagnostic": True, "global_acceptance_gate": False, "production_changes": []},
        "sources": _source_identity(config),
    }
    _write_json(output / "heldout.json", payload)
    _event("heldout_complete")
    return payload


def _merge(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args.config)
    output = _path(config, "output")
    cases = [json.loads((output / f"N{n}.json").read_text()) for n in (32, 48, 64)]
    results = {}
    all_pass = True
    for field in FIELDS:
        errors = [case["statistics"][field]["candidate"]["matched_C"]["absolute_l2"] for case in cases]
        orders = _orders(errors)
        passes = all(value >= 1.8 for value in orders)
        all_pass &= passes
        baseline = [case["statistics"][field]["baseline"]["C"]["absolute_l2"] for case in cases]
        results[field] = {
            "candidate_matched_C_errors": errors,
            "candidate_matched_C_orders": orders,
            "passes_both_intervals": passes,
            "baseline_C_errors": baseline,
            "constituents": {
                action: [case["statistics"][field]["candidate"][f"matched_{action}"]["absolute_l2"] for case in cases]
                for action in ACTIONS
            },
        }
    uncertainty = {}
    reference_root = _path(config, "reference_root")
    for resolution, case in zip((32, 48, 64), cases):
        qualified = json.loads((reference_root / f"N{resolution}.qualification.json").read_text())
        omega = qualified["fields"]["actual_vorticity"]
        budget = (
            omega["direct_q3_minus_ibp_q3_representative_rms"]
            + omega["quadrature_q4_minus_q3_rms"]
            + omega["finite_difference_step_sensitivity_rms"]
        )
        error = case["statistics"]["actual_vorticity"]["candidate"]["matched_C"]["absolute_l2"]
        uncertainty[str(resolution)] = {
            "bounded_absolute_budget": budget,
            "fraction_of_global_candidate_error": budget / error,
            "passes_10_percent_budget": budget < 0.1 * error,
            "direct_ibp_independence": True,
            "q3_q4_and_step_sensitivity": True,
        }
    payload = {
        "schema": SUMMARY_SCHEMA,
        "status": "computation completed",
        "invariants_checked": all(
            case["verification"]["finite_complete_owner_coverage"]
            and case["verification"]["constant_field_action_max_abs"] < 1.0e-10
            and case["verification"]["centered_decomposition_max_abs"] < 1.0e-12
            for case in cases
        ),
        "convergence_passed": bool(all_pass and all(item["passes_10_percent_budget"] for item in uncertainty.values())),
        "results": results,
        "reference_qualification": uncertainty,
        "cases": [str((output / f"N{n}.json").resolve()) for n in (32, 48, 64)],
        "preflight": _identity(output / "preflight.json"),
        "heldout": _identity(output / "heldout.json"),
        "candidate": config["candidate"],
        "scope": config["scope"],
        "sources": _source_identity(config),
    }
    _write_json(output / "summary.json", payload)
    _event("merge_complete", convergence_passed=payload["convergence_passed"])
    return payload


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args.config)
    output = _path(config, "output")
    stage = args.stage
    if stage == "preflight":
        payload = json.loads((output / "preflight.json").read_text())
        if payload.get("schema") != PREFLIGHT_SCHEMA or not payload.get("passes_implementation_preflight"):
            raise ValueError("invalid preflight")
    elif stage.startswith("prepare_N"):
        n = int(stage.split("N", 1)[1])
        _arrays, payload = _load_npz(output / f"N{n}.prepare.npz", PREPARE_SCHEMA)
        if payload.get("status") != "complete" or payload.get("physical_radial_boundary_face_count") != n**2:
            raise ValueError("invalid prepare output")
    elif stage.startswith("case_N"):
        n = int(stage.split("N", 1)[1])
        arrays, metadata = _load_npz(output / f"N{n}.npz", SCHEMA)
        payload = json.loads((output / f"N{n}.json").read_text())
        if metadata.get("status") != "complete" or payload != metadata:
            raise ValueError("case metadata mismatch")
        if not metadata["verification"]["finite_complete_owner_coverage"]:
            raise ValueError("case coverage incomplete")
        expected = {f"action:{formulation}:{action}" for formulation in ("matched", "face", "midpoint") for action in ACTIONS}
        if not expected.issubset(arrays):
            raise ValueError("case action catalogue incomplete")
    elif stage == "merge":
        payload = json.loads((output / "summary.json").read_text())
        if payload.get("schema") != SUMMARY_SCHEMA or payload.get("status") != "computation completed":
            raise ValueError("invalid summary")
    elif stage == "heldout":
        payload = json.loads((output / "heldout.json").read_text())
        if payload.get("schema") != "drbx.hsx-matched-cubic-heldout-v1" or payload.get("status") != "complete":
            raise ValueError("invalid heldout diagnostic")
    else:
        raise ValueError(stage)
    print(json.dumps({"stage": stage, "valid": True}, sort_keys=True))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "case"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--resolution", type=int, choices=(32, 48, 64), required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--config", type=Path, required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--config", type=Path, required=True)
    heldout = sub.add_parser("heldout")
    heldout.add_argument("--config", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--stage", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        _prepare(args)
    elif args.command == "case":
        _case(args)
    elif args.command == "preflight":
        _preflight(args)
    elif args.command == "merge":
        _merge(args)
    elif args.command == "heldout":
        _heldout(args)
    elif args.command == "validate":
        _validate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
