#!/usr/bin/env python3
"""Portable spawn-safe chunk runner for the frozen matched HSX bracket.

The numerical kernels live in ``numerics.py``.  This file only
provides explicit root resolution, deterministic work manifests, persistent
worker initialization, atomic checkpoints, validation, and assembly handoff.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
CONFIG_SCHEMA = "drbx.hsx-matched-cubic-portable-config-v3"
PLAN_SCHEMA = "drbx.hsx-matched-cubic-work-plan-v3"
RECEIPT_SCHEMA = "drbx.hsx-matched-cubic-parallel-receipt-v3"
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
DEFAULT_MAX_TASKS_PER_WORKER = 16
_STATE: dict[str, Any] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@contextmanager
def _exclusive_output(output: Path):
    """Prevent duplicate writers for one checkpoint directory."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / ".parallel.lock"
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another runner holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _load_portable(path: Path) -> dict[str, Any]:
    config = _load_json(path)
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported portable configuration")
    return config


def _bootstrap(deployment_root: Path, source_root: Path, output_root: Path):
    deployment_root = deployment_root.resolve()
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    os.environ.update(THREAD_ENV)
    os.environ["HSX_DEPLOYMENT_ROOT"] = str(deployment_root)
    os.environ["HSX_SOURCE_ROOT"] = str(source_root)
    os.environ["HSX_JAX_CACHE"] = str(output_root / "jax_cache")
    for entry in (source_root / "src", source_root, source_root / "scripts"):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    name = "hsx_matched_numerics"
    if name in sys.modules:
        return sys.modules[name]
    target = HERE / "numerics.py"
    spec = importlib.util.spec_from_file_location(name, target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _materialize(
    config_path: Path, deployment_root: Path, input_root: Path, output_root: Path
) -> tuple[dict[str, Any], Path, Any]:
    portable = _load_portable(config_path)
    source_root = _resolve(deployment_root, portable.get("source_root", "DRBX"))
    output_root.mkdir(parents=True, exist_ok=True)
    inputs = {name: _resolve(input_root, value) for name, value in portable["inputs"].items()}
    original_sidecar = _load_json(inputs["reference_sidecar"])
    rewritten_sidecar = dict(original_sidecar)
    rewritten_sidecar["metric_cache"] = dict(original_sidecar["metric_cache"])
    rewritten_sidecar["makegrid"] = dict(original_sidecar["makegrid"])
    rewritten_sidecar["metric_cache"]["path"] = str(inputs["metric_cache"])
    rewritten_sidecar["makegrid"]["path"] = str(inputs["makegrid"])
    sidecar_path = output_root / "portable_reference_sidecar.json"
    _atomic_json(sidecar_path, rewritten_sidecar)
    runtime = {
        "schema": "drbx.hsx-matched-cubic-global-config-v3",
        "time": portable["time"],
        "curl_step": portable["curl_step"],
        "face_chunk": portable["face_chunk"],
        "cell_chunk": portable["cell_chunk"],
        "paths": {
            "output": str(output_root),
            "geometry": str(inputs["geometry"]),
            "baseline": str(inputs["baseline"]),
            "reference_sidecar": str(sidecar_path),
            "reference_root": str(inputs["reference_root"]),
        },
        "candidate": portable["candidate"],
        "scope": portable["scope"],
    }
    runtime_path = output_root / "runtime_configuration.json"
    _atomic_json(runtime_path, runtime)
    numeric = _bootstrap(deployment_root, source_root, output_root)
    return runtime, runtime_path, numeric


def _manifest_identity(
    portable_path: Path, input_manifest: Path, prepare: Path, numeric: Any
) -> dict[str, Any]:
    portable = _load_portable(portable_path)
    inputs = _load_json(input_manifest)
    source_manifest = prepare.parent / "campaign_manifest.json"
    source = _load_json(source_manifest) if source_manifest.is_file() else {
        "numerics_sha256": _sha256(HERE / "numerics.py")
    }
    payload = {
        "parameters": {key: portable[key] for key in ("time", "curl_step", "face_chunk", "cell_chunk", "candidate", "scope")},
        "inputs": inputs.get("content_identity", inputs),
        "sources": source.get("content_identity", source),
        "prepare_sha256": _sha256(prepare),
        "prepare_bytes": prepare.stat().st_size,
        "kernel_schema": numeric.CHUNK_SCHEMA,
    }
    return {"sha256": _digest(payload), "payload": payload}


def _unit_indices(unit: Mapping[str, Any]) -> np.ndarray:
    if "indices" in unit:
        result = np.asarray(unit["indices"], dtype=np.int64)
    else:
        result = np.arange(int(unit["first"]), int(unit["last"]), dtype=np.int64)
    if result.ndim != 1 or len(result) == 0 or len(np.unique(result)) != len(result):
        raise ValueError(f"invalid indices for unit {unit.get('id')}")
    return result


def _validate_plan(plan: Mapping[str, Any], numeric: Any) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported work plan")
    resolution = int(plan["resolution"])
    seen_ids: set[str] = set()
    seen: dict[str, set[int]] = {"face": set(), "cell": set()}
    limits = {"face": numeric._face_count(resolution), "cell": resolution**3}
    for unit in plan["units"]:
        uid, kind = unit["id"], unit["kind"]
        if uid in seen_ids or kind not in limits:
            raise ValueError("duplicate unit id or invalid kind")
        seen_ids.add(uid)
        indices = _unit_indices(unit)
        if int(indices.min()) < 0 or int(indices.max()) >= limits[kind]:
            raise ValueError(f"out-of-range {kind} index in {uid}")
        overlap = seen[kind].intersection(map(int, indices))
        if overlap:
            raise ValueError(f"duplicate {kind} coverage in {uid}: {min(overlap)}")
        seen[kind].update(map(int, indices))
    if plan.get("coverage") == "global":
        for kind, limit in limits.items():
            if seen[kind] != set(range(limit)):
                raise ValueError(f"incomplete global {kind} coverage")


def _chunk_path(output: Path, resolution: int, unit: Mapping[str, Any]) -> Path:
    root = output / f"N{resolution}.chunks"
    if "first" in unit:
        return root / f"{unit['kind']}_{int(unit['first']):07d}_{int(unit['last']):07d}.npz"
    return root / f"{unit['id']}.npz"


def _expected_execution_identity(state: Mapping[str, Any], unit: Mapping[str, Any]) -> dict[str, Any]:
    if "first" in unit:
        return {
            "kind": unit["kind"],
            "resolution": state["resolution"],
            "first": int(unit["first"]),
            "last": int(unit["last"]),
            "config_sha256": _sha256(Path(state["runtime_path"])),
            "sources": state["numeric"]._source_identity(state["runtime"]),
            "prepare_sha256": _sha256(Path(state["prepare_path"])),
        }
    return {
        "kind": unit["kind"], "resolution": state["resolution"], "unit_id": unit["id"],
        "indices_sha256": hashlib.sha256(_unit_indices(unit).tobytes()).hexdigest(),
        "runtime_sha256": _sha256(Path(state["runtime_path"])),
        "prepare_sha256": _sha256(Path(state["prepare_path"])),
    }


def _init_worker(settings: Mapping[str, Any]) -> None:
    started = time.perf_counter()
    numeric = _bootstrap(
        Path(settings["deployment_root"]), Path(settings["source_root"]), Path(settings["output_root"])
    )
    runtime = _load_json(Path(settings["runtime_path"]))
    resolution = int(settings["resolution"])
    prepare_path = Path(settings["prepare_path"])
    prepare, _ = numeric._load_npz(prepare_path, numeric.PREPARE_SCHEMA)
    context = numeric.cubic._load_context(
        Path(runtime["paths"]["geometry"]), Path(runtime["paths"]["baseline"]), resolution
    )
    reference = numeric.integrated._reference(Path(runtime["paths"]["reference_sidecar"]), verify_hashes=False)
    _STATE.clear()
    _STATE.update({
        "numeric": numeric, "runtime": runtime, "runtime_path": settings["runtime_path"],
        "resolution": resolution, "prepare": prepare, "prepare_path": str(prepare_path),
        "context": context, "reference": reference, "output_root": settings["output_root"],
        "numerical_identity": settings["numerical_identity"], "task_ordinal": 0,
        "initialization_seconds": time.perf_counter() - started,
        "fail_unit": settings.get("fail_unit"),
    })


def _run_unit(unit: Mapping[str, Any]) -> dict[str, Any]:
    state = _STATE
    numeric = state["numeric"]
    resolution = state["resolution"]
    output = Path(state["output_root"])
    path = _chunk_path(output, resolution, unit)
    identity = _expected_execution_identity(state, unit)
    if path.exists():
        try:
            arrays, metadata = numeric._load_npz(path, numeric.CHUNK_SCHEMA)
        except Exception as exc:
            raise RuntimeError(f"stale or corrupt checkpoint {path}: {exc}") from exc
        if metadata.get("identity") != numeric._json(identity):
            raise RuntimeError(f"stale identity rejected for {path}")
        if metadata.get("numerical_identity") != state["numerical_identity"]:
            raise RuntimeError(f"numerical identity rejected for {path}")
        if not np.array_equal(arrays["indices"], _unit_indices(unit)):
            raise RuntimeError(f"index coverage mismatch for {path}")
        return {"id": unit["id"], "status": "reused", "path": str(path)}
    if state.get("fail_unit") == unit["id"]:
        raise RuntimeError(f"controlled failure for {unit['id']}")
    indices = _unit_indices(unit)
    started = time.perf_counter()
    if unit["kind"] == "face":
        arrays, details = numeric._compute_faces(
            state["context"], state["reference"], state["prepare"], indices,
            time_value=float(state["runtime"]["time"]),
        )
    else:
        arrays, details = numeric._compute_cells(
            state["context"], state["reference"], state["prepare"], indices,
            time_value=float(state["runtime"]["time"]),
            curl_step=float(state["runtime"]["curl_step"]),
        )
    state["task_ordinal"] += 1
    metadata = {
        "schema": numeric.CHUNK_SCHEMA, "status": "complete", "identity": identity,
        "numerical_identity": state["numerical_identity"], "unit_id": unit["id"],
        "details": details, "seconds": time.perf_counter() - started,
        "maximum_rss_gib": numeric._max_rss_gib(), "worker_pid": os.getpid(),
        "worker_initialization_seconds": state["initialization_seconds"],
        "worker_task_ordinal": state["task_ordinal"],
    }
    metadata["array_sha256"] = numeric._array_hash(*(arrays[name] for name in sorted(arrays)))
    numeric._write_npz(path, arrays, metadata)
    return {"id": unit["id"], "status": "computed", "path": str(path), "seconds": metadata["seconds"]}


def _execute_units(
    settings: Mapping[str, Any],
    units: list[Mapping[str, Any]],
    *,
    workers: int,
    max_tasks_per_worker: int,
) -> list[dict[str, Any]]:
    """Execute chunks in spawn workers with a bounded allocator lifetime.

    The numerical kernels make many short-lived NumPy/SciPy allocations.  A
    long-lived process can retain the freed allocator arenas and eventually
    push them into swap even though the live chunk state is small.  Recycling
    every worker after a bounded number of chunks lets the operating system
    reclaim those arenas.  The one-worker case intentionally uses the same
    pool path so it cannot silently regress to an unbounded serial process.
    """

    if workers < 1:
        raise ValueError("workers must be positive")
    if max_tasks_per_worker < 1:
        raise ValueError("max_tasks_per_worker must be positive")
    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(settings,),
        maxtasksperchild=max_tasks_per_worker,
    ) as pool:
        return list(pool.imap_unordered(_run_unit, units, chunksize=1))


def _settings(args: argparse.Namespace, plan: Mapping[str, Any], runtime_path: Path, numeric: Any) -> dict[str, Any]:
    source_root = _resolve(args.deployment_root, _load_portable(args.config).get("source_root", "DRBX"))
    prepare = args.prepare or args.output_root / f"N{int(plan['resolution'])}.prepare.npz"
    identity = _manifest_identity(args.config, args.input_manifest, prepare, numeric)["sha256"]
    if identity != plan["numerical_identity"]:
        raise ValueError("plan numerical identity does not match sources/inputs/prepare")
    return {
        "deployment_root": str(args.deployment_root.resolve()), "source_root": str(source_root),
        "output_root": str(args.output_root.resolve()), "runtime_path": str(runtime_path),
        "prepare_path": str(prepare.resolve()), "resolution": int(plan["resolution"]),
        "numerical_identity": identity, "fail_unit": getattr(args, "fail_unit", None),
    }


def command_plan(args: argparse.Namespace) -> int:
    runtime, runtime_path, numeric = _materialize(args.config, args.deployment_root, args.input_root, args.output_root)
    resolution = int(args.resolution)
    prepare = args.prepare or args.output_root / f"N{resolution}.prepare.npz"
    identity = _manifest_identity(args.config, args.input_manifest, prepare, numeric)["sha256"]
    units: list[dict[str, Any]] = []
    if args.coverage == "global":
        for kind, total, size in (
            ("face", numeric._face_count(resolution), int(runtime["face_chunk"])),
            ("cell", resolution**3, int(runtime["cell_chunk"])),
        ):
            for first in range(0, total, size):
                last = min(first + size, total)
                units.append({"id": f"N{resolution}-{kind}-{first:07d}-{last:07d}", "kind": kind, "first": first, "last": last})
    else:
        cells = []
        for i, j, k in (
            (0, 0, 0), (0, resolution // 3, resolution // 5), (1, resolution // 2, resolution // 7),
            (resolution // 4, 1, resolution // 3), (resolution // 2, resolution // 2, resolution // 2),
            (resolution - 2, resolution // 4, resolution // 2),
            (resolution - 1, 0, 0), (resolution - 1, resolution - 1, resolution - 1),
        ):
            cells.append(int(np.ravel_multi_index((i, j, k), (resolution,) * 3)))
        cells = sorted(set(cells))
        keys = numeric._raw_keys(resolution, np.asarray(cells, dtype=np.int64))
        faces: set[int] = set()
        for axis in range(3):
            lower, upper = keys.copy(), keys.copy()
            upper[:, axis] += 1
            faces.update(map(int, numeric._face_index(resolution, axis, *lower.T)))
            faces.update(map(int, numeric._face_index(resolution, axis, *upper.T)))
        for parts, kind in ((np.array_split(sorted(faces), 4), "face"), (np.array_split(cells, 4), "cell")):
            for index, part in enumerate(parts):
                if len(part):
                    units.append({"id": f"N{resolution}-{kind}-test-{index:02d}", "kind": kind, "indices": [int(x) for x in part]})
    plan = {
        "schema": PLAN_SCHEMA, "resolution": resolution, "coverage": args.coverage,
        "numerical_identity": identity, "runtime_config_sha256": _sha256(runtime_path),
        "units": units,
    }
    _validate_plan(plan, numeric)
    _atomic_json(args.plan, plan)
    print(json.dumps({"plan": str(args.plan), "units": len(units), "numerical_identity": identity}, sort_keys=True))
    return 0


def command_execute(args: argparse.Namespace) -> int:
    with _exclusive_output(args.output_root):
        _, runtime_path, numeric = _materialize(args.config, args.deployment_root, args.input_root, args.output_root)
        plan = _load_json(args.plan)
        _validate_plan(plan, numeric)
        settings = _settings(args, plan, runtime_path, numeric)
        workers = int(args.workers)
        max_tasks_per_worker = int(args.max_tasks_per_worker)
        started = time.perf_counter()
        results = _execute_units(
            settings,
            plan["units"],
            workers=workers,
            max_tasks_per_worker=max_tasks_per_worker,
        )
        receipt = {
            "schema": RECEIPT_SCHEMA, "status": "complete", "resolution": plan["resolution"],
            "coverage": plan["coverage"], "workers": workers,
            "max_tasks_per_worker": max_tasks_per_worker,
            "seconds": time.perf_counter() - started,
            "numerical_identity": plan["numerical_identity"], "plan_sha256": _sha256(args.plan),
            "results": sorted(results, key=lambda item: item["id"]),
        }
        _atomic_json(args.output_root / f"N{plan['resolution']}.parallel-receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "workers", "seconds")}, sort_keys=True))
    return 0


def _validate_chunks(args: argparse.Namespace, require_receipt: bool = True) -> tuple[dict[str, Any], Any, Path]:
    _, runtime_path, numeric = _materialize(args.config, args.deployment_root, args.input_root, args.output_root)
    plan = _load_json(args.plan)
    _validate_plan(plan, numeric)
    settings = _settings(args, plan, runtime_path, numeric)
    state = dict(settings)
    state.update({"numeric": numeric, "runtime": _load_json(runtime_path)})
    for unit in plan["units"]:
        path = _chunk_path(args.output_root, int(plan["resolution"]), unit)
        if not path.is_file():
            raise RuntimeError(f"missing checkpoint: {path}")
        arrays, metadata = numeric._load_npz(path, numeric.CHUNK_SCHEMA)
        if metadata.get("identity") != numeric._json(_expected_execution_identity(state, unit)):
            raise RuntimeError(f"execution identity mismatch: {path}")
        if metadata.get("numerical_identity") != plan["numerical_identity"]:
            raise RuntimeError(f"numerical identity mismatch: {path}")
        if not np.array_equal(arrays["indices"], _unit_indices(unit)):
            raise RuntimeError(f"coverage mismatch: {path}")
    receipt = args.output_root / f"N{plan['resolution']}.parallel-receipt.json"
    if require_receipt:
        if not receipt.is_file():
            raise RuntimeError(f"missing execution receipt: {receipt}")
        saved = _load_json(receipt)
        if saved.get("numerical_identity") != plan["numerical_identity"] or saved.get("plan_sha256") != _sha256(args.plan):
            raise RuntimeError("stale execution receipt")
    return plan, numeric, runtime_path


def command_validate(args: argparse.Namespace) -> int:
    plan, _, _ = _validate_chunks(args)
    print(json.dumps({"status": "valid", "resolution": plan["resolution"], "units": len(plan["units"])}, sort_keys=True))
    return 0


def command_assemble(args: argparse.Namespace) -> int:
    with _exclusive_output(args.output_root):
        plan, numeric, runtime_path = _validate_chunks(args)
        if plan["coverage"] != "global":
            raise ValueError("scientific case assembly requires a global-coverage plan")
        namespace = argparse.Namespace(config=runtime_path, resolution=int(plan["resolution"]))
        numeric._case(namespace)
    return 0


def command_frozen_stage(args: argparse.Namespace) -> int:
    """Expose the unchanged prepare/diagnostic/merge checkpoints portably."""
    _, runtime_path, numeric = _materialize(
        args.config, args.deployment_root, args.input_root, args.output_root
    )
    namespace = argparse.Namespace(config=runtime_path)
    if args.resolution is not None:
        namespace.resolution = int(args.resolution)
    if args.stage == "prepare":
        if args.resolution is None:
            raise ValueError("prepare requires --resolution")
        numeric._prepare(namespace)
    elif args.stage == "preflight":
        numeric._preflight(namespace)
    elif args.stage == "merge":
        numeric._merge(namespace)
    elif args.stage.startswith("validate:"):
        namespace.stage = args.stage.split(":", 1)[1]
        numeric._validate(namespace)
    else:
        raise ValueError(f"unsupported frozen stage: {args.stage}")
    return 0


def _partial_signature(output: Path, plan: Mapping[str, Any], numeric: Any, prepare: Mapping[str, np.ndarray]) -> dict[str, Any]:
    resolution = int(plan["resolution"])
    chunks: dict[str, list[tuple[dict[str, np.ndarray], dict[str, Any]]]] = {"face": [], "cell": []}
    for unit in plan["units"]:
        arrays, meta = numeric._load_npz(_chunk_path(output, resolution, unit), numeric.CHUNK_SCHEMA)
        chunks[unit["kind"]].append((arrays, meta))
    face_arrays: dict[int, dict[str, np.ndarray]] = {}
    for arrays, _ in chunks["face"]:
        for row, index in enumerate(arrays["indices"]):
            face_arrays[int(index)] = {name: value[..., row] for name, value in arrays.items() if name != "indices"}
    rho_star = float(numeric.mms.PHYSICAL_PARAMETERS["rho_star"])
    owner_count = len(prepare["owner_volume"])
    accum = np.zeros((len(numeric.DIAGNOSTIC_FIELDS), 2, owner_count))
    reference = np.zeros(owner_count)
    used_cells = 0
    for arrays, _ in chunks["cell"]:
        for row, raw_index in enumerate(arrays["indices"].astype(np.int64)):
            key = numeric._raw_keys(resolution, np.asarray([raw_index]))[0]
            center = prepare["center"][:, raw_index]
            face_term = np.zeros((len(numeric.DIAGNOSTIC_FIELDS), 2))
            ref_term = 0.0
            complete = True
            for axis in range(3):
                lower_key, upper_key = key.copy(), key.copy()
                upper_key[axis] += 1
                lower = int(numeric._face_index(resolution, axis, *lower_key[:, None])[0])
                upper = int(numeric._face_index(resolution, axis, *upper_key[:, None])[0])
                if lower not in face_arrays or upper not in face_arrays:
                    complete = False
                    break
                lo, hi = face_arrays[lower], face_arrays[upper]
                ref_term += float(hi["reference_omega_product"] - lo["reference_omega_product"])
                for field_index in range(len(numeric.DIAGNOSTIC_FIELDS)):
                    scalar = field_index + 1
                    scalar_anchor = 1.0 if field_index == len(numeric.FIELDS) else center[scalar]
                    phi_anchor = center[0]
                    face_term[field_index, 0] += hi["product"][field_index, 0] - scalar_anchor * hi["flux"][0] - lo["product"][field_index, 0] + scalar_anchor * lo["flux"][0]
                    face_term[field_index, 1] += hi["product"][field_index, 1] - phi_anchor * hi["flux"][scalar] - lo["product"][field_index, 1] + phi_anchor * lo["flux"][scalar]
            if not complete:
                continue
            owner = int(prepare["raw_owner"][raw_index])
            accum[:, :, owner] += face_term - arrays["volume"][:, :, row]
            reference[owner] += ref_term - float(arrays["reference_omega_volume"][row])
            used_cells += 1
    volume = prepare["owner_volume"]
    action_a = -accum[:, 0] / (rho_star * volume[None, :])
    action_b = accum[:, 1] / (rho_star * volume[None, :])
    return {
        "used_cells": used_cells,
        "action_A_l2": float(np.linalg.norm(action_a)),
        "action_B_l2": float(np.linalg.norm(action_b)),
        "action_C_l2": float(np.linalg.norm(0.5 * (action_a + action_b))),
        "reference_l2": float(np.linalg.norm(-reference / (rho_star * volume))),
        "action_array_sha256": hashlib.sha256(np.ascontiguousarray(np.stack((action_a, action_b))).tobytes()).hexdigest(),
    }


def command_compare(args: argparse.Namespace) -> int:
    _, _, numeric = _materialize(args.config, args.deployment_root, args.input_root, args.left)
    plan = _load_json(args.plan)
    _validate_plan(plan, numeric)
    maximum = 0.0
    exact = True
    timing: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    for unit in plan["units"]:
        left_arrays, left_meta = numeric._load_npz(_chunk_path(args.left, int(plan["resolution"]), unit), numeric.CHUNK_SCHEMA)
        right_arrays, right_meta = numeric._load_npz(_chunk_path(args.right, int(plan["resolution"]), unit), numeric.CHUNK_SCHEMA)
        if set(left_arrays) != set(right_arrays):
            raise RuntimeError(f"array set mismatch for {unit['id']}")
        for name in left_arrays:
            exact = exact and np.array_equal(left_arrays[name], right_arrays[name])
            if np.issubdtype(left_arrays[name].dtype, np.number):
                maximum = max(maximum, float(np.max(np.abs(left_arrays[name] - right_arrays[name]))))
        for side, meta in (("left", left_meta), ("right", right_meta)):
            timing[side].append({key: meta.get(key) for key in ("unit_id", "seconds", "worker_pid", "worker_initialization_seconds", "worker_task_ordinal", "maximum_rss_gib")})
    prepare, _ = numeric._load_npz(args.prepare, numeric.PREPARE_SCHEMA)
    signatures = {
        "left": _partial_signature(args.left, plan, numeric, prepare),
        "right": _partial_signature(args.right, plan, numeric, prepare),
    }
    report = {
        "schema": "drbx.hsx-matched-cubic-parallel-comparison-v3", "status": "pass" if exact else "fail",
        "arrays_bitwise_equal": exact, "maximum_array_abs_difference": maximum,
        "partial_action_signatures": signatures,
        "partial_actions_equal": signatures["left"] == signatures["right"], "timing": timing,
    }
    _atomic_json(args.report, report)
    print(json.dumps({"status": report["status"], "maximum_array_abs_difference": maximum, "partial_actions_equal": report["partial_actions_equal"]}, sort_keys=True))
    return 0 if exact and report["partial_actions_equal"] else 1


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--prepare", type=Path)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    _common(plan)
    plan.add_argument("--resolution", type=int, required=True)
    plan.add_argument("--coverage", choices=("global", "test"), required=True)
    plan.add_argument("--plan", type=Path, required=True)
    execute = sub.add_parser("execute")
    _common(execute)
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--workers", type=_positive_int, required=True)
    execute.add_argument(
        "--max-tasks-per-worker",
        type=_positive_int,
        default=DEFAULT_MAX_TASKS_PER_WORKER,
        help=(
            "recycle each worker after this many chunks to release retained "
            f"allocator memory (default: {DEFAULT_MAX_TASKS_PER_WORKER})"
        ),
    )
    execute.add_argument("--fail-unit")
    validate = sub.add_parser("validate")
    _common(validate)
    validate.add_argument("--plan", type=Path, required=True)
    assemble = sub.add_parser("assemble")
    _common(assemble)
    assemble.add_argument("--plan", type=Path, required=True)
    frozen = sub.add_parser("frozen-stage")
    frozen.add_argument("--config", type=Path, required=True)
    frozen.add_argument("--deployment-root", type=Path, required=True)
    frozen.add_argument("--input-root", type=Path, required=True)
    frozen.add_argument("--output-root", type=Path, required=True)
    frozen.add_argument("--stage", required=True)
    frozen.add_argument("--resolution", type=int, choices=(32, 48, 64))
    compare = sub.add_parser("compare")
    compare.add_argument("--config", type=Path, required=True)
    compare.add_argument("--deployment-root", type=Path, required=True)
    compare.add_argument("--input-root", type=Path, required=True)
    compare.add_argument("--input-manifest", type=Path, required=True)
    compare.add_argument("--plan", type=Path, required=True)
    compare.add_argument("--prepare", type=Path, required=True)
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--report", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    return {
        "plan": command_plan, "execute": command_execute, "validate": command_validate,
        "assemble": command_assemble, "frozen-stage": command_frozen_stage,
        "compare": command_compare,
    }[args.command](args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
