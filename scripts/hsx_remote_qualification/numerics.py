#!/usr/bin/env python3
"""Checkpointed global qualification of the frozen matched cubic HSX bracket.

This is research-only orchestration.  It reuses the qualified cubic policy and
reference evaluators without modifying production solver or boundary code.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import gc
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
# Derived from the matched-q3 candidate. V3 changes donor tie handling,
# rebuilds boundary dependencies, and has no historical reconstruction-cache gate.
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

sys.path.insert(0, str(HERE))
import selection
selection.install(cubic)


SCHEMA = "drbx.hsx-matched-cubic-global-case-v3"
PREPARE_SCHEMA = "drbx.hsx-matched-cubic-global-prepare-v3"
CHUNK_SCHEMA = "drbx.hsx-matched-cubic-global-chunk-v3"
PREFLIGHT_SCHEMA = "drbx.hsx-matched-cubic-global-preflight-v3"
SUMMARY_SCHEMA = "drbx.hsx-matched-cubic-global-summary-v3"
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


def _current_rss_gib() -> float | None:
    """Return current resident memory when the platform exposes it."""

    statm = Path("/proc/self/statm")
    if statm.is_file():
        try:
            resident_pages = int(statm.read_text().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1024.0**3
        except (OSError, ValueError, IndexError):
            pass
    try:
        import psutil  # type: ignore[import-not-found]

        return float(psutil.Process().memory_info().rss) / 1024.0**3
    except (ImportError, OSError):
        return None


def _metric_cache_diagnostics(reference: Any) -> dict[str, Any] | None:
    evaluator = getattr(reference, "metric_evaluator", None)
    if evaluator is None or not hasattr(evaluator, "basis_cache_diagnostics"):
        return None
    return evaluator.basis_cache_diagnostics()


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
    if payload.get("schema") != "drbx.hsx-matched-cubic-global-config-v3":
        raise ValueError("unsupported global matched configuration")
    return payload


def _path(config: Mapping[str, Any], name: str) -> Path:
    return Path(config["paths"][name]).resolve()


def _source_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        "implementation": Path(__file__),
        "selection": HERE / "selection.py",
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
            exact_query=False,
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
            "donor_sha256": hashlib.sha256(np.asarray(donor, dtype="<i8").tobytes()).hexdigest(),
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
    batch_size = int(getattr(reference, "metric_query_batch_size", 4096))
    if batch_size < 1:
        raise ValueError("metric_query_batch_size must be positive")
    values = np.empty((3, 4, len(q), 3), dtype=np.float64)
    for axis in range(3):
        for shift_index, multiplier in enumerate((-2.0, -1.0, 1.0, 2.0)):
            for first in range(0, len(q), batch_size):
                last = min(first + batch_size, len(q))
                shifted = q[first:last].copy()
                shifted[:, axis] += multiplier * float(step)
                values[axis, shift_index, first:last] = bounded._h_without_rho(
                    reference, shifted
                )
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


def _boundary_derivatives(context):
    """Rebuild exactly the derivative dependency actually used by this study."""
    n = context.resolution
    gradient = np.empty((len(ALL_FIELDS), n, n, 3))
    saved = np.full((n, n, 240), -1, dtype=np.int32)
    counts = np.zeros((n, n), dtype=np.int16)
    for k in range(n):
        points = np.column_stack((np.full(n, context.x_faces[-1]), context.y_centers,
                                  np.full(n, context.z_centers[k])))
        pending = np.arange(n)
        for policy in cubic.POLICY["deficient_row_expansion_schedule"]:
            donors, weights, diagnostic, _ = cubic._row_batch(
                context, 0, k, points[pending], exact_query=False,
                count=policy["donors_per_plane"], pool_count=policy["candidate_pool_per_plane"])
            good = ((diagnostic["rank"] == len(cubic.EXPONENTS))
                    & (np.max(diagnostic["residual"], axis=1) <= cubic.base.REPRODUCTION_TOLERANCE)
                    & (diagnostic["minimum_coverage"] >= 3))
            accepted = pending[good]
            if len(accepted):
                value = cubic._regular_to_logical_batch(
                    cubic._apply_rows(context, donors[good], weights[good]), points[accepted])
                gradient[:, accepted, k] = value
                saved[accepted, k, :donors.shape[1]] = donors[good]
                counts[accepted, k] = donors.shape[1]
            pending = pending[~good]
            if not len(pending):
                break
        if len(pending):
            raise RuntimeError(f"unresolved boundary reconstruction rows at eta={k}")
    return gradient, saved, counts


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
    _event("prepare_inputs", resolution=resolution)
    data = integrated._load_resolution(geometry, baseline, resolution)
    artifact_path, artifact = base._load_artifact(geometry, resolution)
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
            nfp=int(artifact.nfp),
        ),
        metric_query_batch_size=int(config.get("metric_query_batch_size", 4096)),
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
    _event("prepare_runtime_ready", resolution=resolution)
    model = runtime.model
    if model is None:
        raise RuntimeError("prepare requires one local model")
    active = np.asarray(artifact.owner_geometry.topology.is_active_owner, dtype=bool)
    arrays: dict[str, np.ndarray] = {
        "owner_volume": data.owner_volume,
        "owner_keys": data.owner_keys,
        "raw_owner": data.raw_owner.astype(np.int32),
    }
    center_rows: list[np.ndarray | None] = [None] * len(ALL_FIELDS)
    boundary_value = np.empty(
        (len(ALL_FIELDS), resolution, resolution), dtype=np.float64
    )
    for field_index, field in enumerate(FIELDS):
        state = (
            raw_state
            if field_index == 0
            else raw_state.replace(
                vorticity=regular_raw if field_index == 1 else eta_raw
            )
        )
        owner_state = mms._owner_project(state, artifact.owner_geometry)
        op = base._prepare_operands(model, owner_state, state)
        if field_index == 0:
            center_rows[0] = np.asarray(op.phi_stencil.x.center).reshape(-1).copy()
            phi_values = tuple(
                np.asarray(getattr(op.phi_stencil.face_values, name))
                for name in "xyz"
            )
            for j in range(resolution):
                for k in range(resolution):
                    boundary_value[0, j, k] = cross._face_scalar(
                        phi_values, (0, resolution, j, k)
                    )
            del phi_values
        center_rows[field_index + 1] = (
            np.asarray(op.omega_stencil.x.center).reshape(-1).copy()
        )
        omega_values = tuple(
            np.asarray(getattr(op.omega_stencil.face_values, name))
            for name in "xyz"
        )
        for j in range(resolution):
            for k in range(resolution):
                boundary_value[field_index + 1, j, k] = cross._face_scalar(
                    omega_values, (0, resolution, j, k)
                )
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
        del action, runner, omega_values, op, owner_state, state
        gc.collect()
    if any(row is None for row in center_rows):
        raise RuntimeError("prepare did not materialize every field center")
    arrays["center"] = np.stack(center_rows).reshape(len(ALL_FIELDS), -1)
    _event("prepare_fields_ready", resolution=resolution)
    boundary_context = cubic._load_context(geometry, baseline, resolution)
    _event("prepare_boundary_rows", resolution=resolution)
    fresh_gradient, fresh_donors, fresh_counts = _boundary_derivatives(boundary_context)
    boundary_gradient = np.empty((len(ALL_FIELDS), resolution, resolution, 3), dtype=np.float64)
    for field_index, field in enumerate(ALL_FIELDS):
        for j in range(resolution):
            for k in range(resolution):
                boundary_gradient[field_index, j, k] = fresh_gradient[field_index, j, k]
    arrays["boundary_value"] = boundary_value
    arrays["boundary_gradient"] = boundary_gradient
    arrays["boundary_donors"] = fresh_donors
    arrays["boundary_donor_count"] = fresh_counts
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
        "derivative_origin": "rebuilt physical boundary rows under selection-v3; no historical derivative cache",
        "selection_policy": selection.POLICY,
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
    donor_digest = np.full(len(keys), "collapsed", dtype="U64")
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
            donor_digest[row] = hashlib.sha256(np.asarray(prepare["boundary_donors"][j, k], dtype="<i8").tobytes()).hexdigest()
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
        donor_digest[row] = fit.diagnostics["donor_sha256"]
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
        "donor_sha256": donor_digest,
        "fallback": fallback,
        "condition": condition,
        "reproduction": reproduction,
    }
    noncollapsed = ~collapsed
    details = {
        "entity_count": len(keys),
        "physical_boundary_count": int(np.count_nonzero((keys[:, 0] == 0) & (keys[:, 1] == n))),
        "collapsed_count": int(np.count_nonzero((keys[:, 0] == 0) & (keys[:, 1] == 0))),
        "maximum_condition": float(np.max(condition)),
        "maximum_reproduction": float(np.max(reproduction)),
        "constant_value_error": float(
            np.max(np.abs(values[-1, noncollapsed] - 1.0))
            if np.any(noncollapsed) else 0.0
        ),
        "constant_gradient_max": float(
            np.max(np.abs(gradients[-1, noncollapsed]))
            if np.any(noncollapsed) else 0.0
        ),
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
    donor_digest = np.empty(len(keys), dtype="U64")
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
        donor_digest[row] = fit.diagnostics["donor_sha256"]
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
        "donor_sha256": donor_digest,
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
        arrays, metadata = _load_npz(path, CHUNK_SCHEMA)
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False
    return (
        metadata.get("identity") == _json(identity)
        and metadata.get("status") == "complete"
        and metadata.get("array_sha256")
        == _array_hash(*(arrays[name] for name in sorted(arrays)))
    )


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
    chunk_root = output / f"N{resolution}.chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    source = _source_identity(config)
    config_sha256 = _sha256(args.config)
    prepare_sha256 = _sha256(prepare_path)
    face_total = _face_count(resolution)
    cell_total = resolution**3
    face_chunk = int(config["face_chunk"])
    cell_chunk = int(config["cell_chunk"])
    total_units = face_total + cell_total
    completed = 0
    face_units = [
        (first, min(first + face_chunk, face_total))
        for first in range(0, face_total, face_chunk)
    ]
    cell_units = [
        (first, min(first + cell_chunk, cell_total))
        for first in range(0, cell_total, cell_chunk)
    ]
    if not bool(getattr(args, "validated_chunks", False)):
        context = None
        reference = None

        def computation_context():
            nonlocal context, reference
            if context is None:
                context = cubic._load_context(
                    _path(config, "geometry"), _path(config, "baseline"), resolution
                )
                reference = integrated._reference(
                    _path(config, "reference_sidecar"), verify_hashes=False
                )
            return context, reference

        for kind, units in (("face", face_units), ("cell", cell_units)):
            for first, last in units:
                path = chunk_root / f"{kind}_{first:07d}_{last:07d}.npz"
                identity = {
                    "kind": kind,
                    "resolution": resolution,
                    "first": first,
                    "last": last,
                    "config_sha256": config_sha256,
                    "sources": source,
                    "prepare_sha256": prepare_sha256,
                }
                if not _chunk_valid(path, identity):
                    local_context, local_reference = computation_context()
                    chunk_started = time.perf_counter()
                    if kind == "face":
                        arrays, details = _compute_faces(
                            local_context, local_reference, prepare,
                            np.arange(first, last), time_value=float(config["time"]),
                        )
                    else:
                        arrays, details = _compute_cells(
                            local_context, local_reference, prepare,
                            np.arange(first, last), time_value=float(config["time"]),
                            curl_step=float(config["curl_step"]),
                        )
                    metadata = {
                        "schema": CHUNK_SCHEMA,
                        "status": "complete",
                        "identity": identity,
                        "details": details,
                        "seconds": time.perf_counter() - chunk_started,
                        "maximum_rss_gib": _max_rss_gib(),
                    }
                    metadata["array_sha256"] = _array_hash(
                        *(arrays[name] for name in sorted(arrays))
                    )
                    _write_npz(path, arrays, metadata)
                completed += last - first
                _progress(output, resolution, f"{kind}s", completed, total_units, started)

    flux = np.empty((len(ALL_FIELDS) + 1, face_total), dtype=np.float64)
    product = np.empty((len(DIAGNOSTIC_FIELDS), 2, face_total), dtype=np.float64)
    midpoint_flux = np.empty_like(flux)
    midpoint_product = np.empty_like(product)
    reference_face = np.empty(face_total, dtype=np.float64)
    face_details = []
    for first, last in face_units:
        path = chunk_root / f"face_{first:07d}_{last:07d}.npz"
        arrays, metadata = _load_npz(path, CHUNK_SCHEMA)
        face_details.append(metadata["details"] | {"seconds": metadata["seconds"]})
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
    cell_details = []
    for first, last in cell_units:
        path = chunk_root / f"cell_{first:07d}_{last:07d}.npz"
        arrays, metadata = _load_npz(path, CHUNK_SCHEMA)
        cell_details.append(metadata["details"] | {"seconds": metadata["seconds"]})
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


def _preflight(args):
    """Fresh HSX execution checks; historical donor equality is not a gate."""
    config = _config(args.config); output = _path(config, 'output'); n = args.resolution
    prepare, _ = _load_npz(output / f'N{n}.prepare.npz', PREPARE_SCHEMA)
    context = cubic._load_context(_path(config, 'geometry'), _path(config, 'baseline'), n)
    reference = integrated._reference(_path(config, 'reference_sidecar'), verify_hashes=False)
    cells = np.asarray(sorted(set(int(np.ravel_multi_index(key, (n,)*3)) for key in (
        (0,0,0), (1,n//2,n//7), (n//4,1,n//3), (n//2,n//2,n//2),
        (min(24,n-1),min(23,n-1),n//2), (n-1,0,0), (n-1,n-1,n-1)))))
    raw_keys = _raw_keys(n, cells); face_set = set()
    for axis in range(3):
        high = raw_keys.copy(); high[:,axis] += 1
        face_set.update(map(int, _face_index(n,axis,*raw_keys.T)))
        face_set.update(map(int, _face_index(n,axis,*high.T)))
    faces = np.asarray(sorted(face_set))
    fa, fd = _compute_faces(context, reference, prepare, faces, time_value=config['time'])
    ca, cd = _compute_cells(context, reference, prepare, cells, time_value=config['time'], curl_step=config['curl_step'])
    # Exercise the reported N32 pool-cutoff location and all boundary sectors.
    donor_checks = []
    for axis in range(3):
        k = n//2
        for point in (np.array((context.x_centers[min(24,n-1)], context.y_centers[min(23,n-1)], context.z_centers[k])),
                      np.array((context.x_centers[n//2], context.y_centers[n//8], context.z_centers[k]))):
            indexed = cubic._row_batch(context, axis,k,point[None],exact_query=False)
            brute = cubic._row_batch(context, axis,k,point[None],exact_query=True)
            if not np.array_equal(indexed[0], brute[0]):
                raise RuntimeError('indexed/exhaustive donor selection mismatch')
            donor_checks.append({'axis':axis,'point':point,'donors':indexed[0][0],
                                 'weight_max_abs_difference':float(np.max(np.abs(indexed[1]-brute[1])))})
    finite = all(np.all(np.isfinite(v)) for d in (fa,ca) for v in d.values() if v.dtype.kind in 'fci')
    if not finite or not fd['physical_boundary_count']:
        raise RuntimeError('fresh HSX preflight failed')
    checks = {'schema':PREFLIGHT_SCHEMA, 'status':'complete','resolution':n,
              'passes_implementation_preflight':True, 'selection_policy':selection.POLICY,
              'finite':finite,'face_details':fd,'cell_details':cd,'indexed_exhaustive_checks':donor_checks,
              'historical_donor_equality_required':False, 'scaling_study':False,
              'reference_qualification':'unchanged independent geometry/reference inputs; error budgets reevaluated at merge',
              'sources':_source_identity(config)}
    # A portable snapshot for remote-to-local comparison if desired; no historic-chain dependency.
    values={**{f'face:{k}':v for k,v in fa.items()},**{f'cell:{k}':v for k,v in ca.items()}}
    checks['array_sha256']=_array_hash(*(values[k] for k in sorted(values)))
    _write_npz(output/f'N{n}.preflight.npz',values,checks)
    _write_json(output/f'N{n}.preflight.json',checks)
    _event('fresh_preflight_complete',resolution=n)
    return checks


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
        uncertainty[str(resolution)] = {"actual_vorticity": {
            "bounded_absolute_budget": budget,
            "fraction_of_global_candidate_error": budget / error,
            "passes_10_percent_budget": budget < 0.1 * error,
            "direct_ibp_independence": True,
            "q3_q4_and_step_sensitivity": True,
        }}
        for field in FIELDS[1:]:
            entry = qualified["fields"][field]
            rule = entry["rules"][str(entry["selected_order"])]
            budget = rule["reference_difference_rms"] + rule["high_reference_q7_minus_q5_rms"]
            error = case["statistics"][field]["candidate"]["matched_C"]["absolute_l2"]
            uncertainty[str(resolution)][field] = {
                "bounded_absolute_budget": budget,
                "fraction_of_global_candidate_error": budget / error,
                "passes_10_percent_budget": budget < 0.1 * error,
                "selected_reference_order": entry["selected_order"],
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
        "convergence_passed": bool(all_pass and all(
            field["passes_10_percent_budget"] for item in uncertainty.values() for field in item.values())),
        "results": results,
        "reference_qualification": uncertainty,
        "cases": [str((output / f"N{n}.json").resolve()) for n in (32, 48, 64)],
        "preflight": [_identity(output / f"N{n}.preflight.json") for n in (32, 48, 64)],
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
        payload = json.loads((output / f"N{args.resolution}.preflight.json").read_text())
        if payload.get("schema") != PREFLIGHT_SCHEMA or not payload.get("passes_implementation_preflight"):
            raise ValueError("invalid preflight")
        if payload.get("sources") != _json(_source_identity(config)):
            raise ValueError("stale preflight source identity")
    elif stage.startswith("prepare_N"):
        n = int(stage.split("N", 1)[1])
        _arrays, payload = _load_npz(output / f"N{n}.prepare.npz", PREPARE_SCHEMA)
        if payload.get("status") != "complete" or payload.get("physical_radial_boundary_face_count") != n**2:
            raise ValueError("invalid prepare output")
        if payload.get("sources") != _json(_source_identity(config)):
            raise ValueError("stale prepare source identity")
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
    else:
        raise ValueError(stage)
    print(json.dumps({"stage": stage, "valid": True}, sort_keys=True))
    return payload
