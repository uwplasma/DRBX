#!/usr/bin/env python3
"""Portable, resumable CPU campaign for the bounded FCI intersection follow-up."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import queue
import resource
import shutil
import socket
import sys
import tarfile
import time
import traceback
from typing import Any, Mapping


THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
os.environ.update(THREAD_ENV)

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE_BUNDLE = HERE / "frozen_source_bundle.tar.gz"
SOURCE_MANIFEST = HERE / "source_manifest.json"
EXTERNAL_INPUT_MANIFEST = HERE / "external_input_manifest.json"
CAMPAIGN_SCHEMA = "drbx.parallel-fci-intersection-eta-remote-v1"
UNIT_SCHEMA = "drbx.parallel-fci-intersection-plane-v1"
CASE_SETTINGS = {
    "preflight": {"max_depth": 3, "order": 2, "eta_order": 1, "preflight": True},
    "base": {"max_depth": 4, "order": 2, "eta_order": 5, "preflight": False},
    "refined": {"max_depth": 6, "order": 3, "eta_order": 7, "preflight": False},
}
ARRAY_KEYS = ("tile", "recipient_index", "weight", "moment", "source")
EQUIVALENCE_RTOL = 2.0e-8
EQUIVALENCE_ATOL = 2.0e-13
_NUMERIC = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("expected a positive number")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("expected a nonnegative number")
    return parsed


@contextmanager
def _exclusive_output(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".campaign.lock"
    with lock_path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another campaign writer holds {lock_path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if target != resolved and resolved not in target.parents:
                raise RuntimeError(f"unsafe archive member {member.name}")
        bundle.extractall(destination, filter="data")


def _verify_external_inputs(input_root: Path) -> dict[str, Any]:
    manifest = _load_json(EXTERNAL_INPUT_MANIFEST)
    records = []
    for item in manifest["files"]:
        path = input_root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != int(item["size"]):
            raise RuntimeError(f"external input size mismatch: {path}")
        actual = _sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"external input hash mismatch: {path}")
        records.append({"path": str(path.resolve()), "size": size, "sha256": actual})
    return {
        "schema": manifest["schema"],
        "manifest_sha256": _sha256(EXTERNAL_INPUT_MANIFEST),
        "files": records,
    }


def _verify_source_tree(source_root: Path) -> dict[str, Any]:
    manifest = _load_json(SOURCE_MANIFEST)
    archive = manifest["archive"]
    if SOURCE_BUNDLE.stat().st_size != int(archive["size"]):
        raise RuntimeError("source bundle size mismatch")
    if _sha256(SOURCE_BUNDLE) != archive["sha256"]:
        raise RuntimeError("source bundle hash mismatch")
    records = []
    for item in manifest["files"]:
        path = source_root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(item["size"]):
            raise RuntimeError(f"frozen source size mismatch: {path}")
        actual = _sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"frozen source hash mismatch: {path}")
        records.append(item)
    return {
        "schema": manifest["schema"],
        "manifest_sha256": _sha256(SOURCE_MANIFEST),
        "archive_sha256": archive["sha256"],
        "file_count": len(records),
    }


def _ensure_link(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = target.resolve()
    if path.is_symlink():
        if path.resolve() != target:
            raise RuntimeError(f"input link target changed: {path}")
        return
    if path.exists():
        raise RuntimeError(f"cannot place immutable input link over {path}")
    path.symlink_to(target, target_is_directory=target.is_dir())


def verify(input_root: Path, output: Path) -> dict[str, Any]:
    input_root = input_root.resolve()
    output = output.resolve()
    source_root = output / "source"
    if not source_root.exists():
        _safe_extract(SOURCE_BUNDLE, source_root)
    source = _verify_source_tree(source_root)
    external = _verify_external_inputs(input_root)
    links = {
        "hsx_metric_d58d392545fd3917efeb83b6.npz": "hsx_metric_d58d392545fd3917efeb83b6.npz",
        ".hsx_metric_cache/hsx_metric_7f6f0883c40ac27342736ebc.npz": (
            "hsx_metric_d58d392545fd3917efeb83b6.npz"
        ),
        "mgrid_res2p5cm_180pln.nc": "mgrid_res2p5cm_180pln.nc",
        "prototype_runs/geometry/hsx_fci_32x32x32": (
            "geometry_artifacts/rlp_convergence_32_48_64_20260917/32x32x32"
        ),
        "prototype_runs/geometry/hsx_fci_64x64x64": (
            "geometry_artifacts/rlp_convergence_32_48_64_20260917/64x64x64"
        ),
    }
    resolved_links = {}
    for source_relative, input_relative in links.items():
        target = input_root / input_relative
        if not target.exists():
            raise FileNotFoundError(target)
        link = source_root / source_relative
        _ensure_link(link, target)
        resolved_links[source_relative] = str(target.resolve())
    payload = {
        "schema": CAMPAIGN_SCHEMA,
        "source": source,
        "external_inputs": external,
        "input_links": resolved_links,
        "source_root": str(source_root),
        "input_root": str(input_root),
        "output": str(output),
    }
    payload["identity"] = _digest({
        "schema": payload["schema"],
        "source_manifest": source["manifest_sha256"],
        "archive": source["archive_sha256"],
        "external_manifest": external["manifest_sha256"],
    })
    _atomic_json(output / "campaign_manifest.json", payload)
    return payload


def _campaign(output: Path) -> dict[str, Any]:
    payload = _load_json(output / "campaign_manifest.json")
    if payload.get("schema") != CAMPAIGN_SCHEMA:
        raise RuntimeError("campaign manifest schema mismatch")
    return payload


def _load_numeric(source_root: Path, output: Path):
    os.environ.update(THREAD_ENV)
    os.environ["DRBX_CACHE_DIR"] = str((output / "cache" / "jax").resolve())
    package = source_root / "DRBX" / "src"
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    path = (
        source_root / "work" / "parallel_fci_intersection_eta_followup_20260922"
        / "intersection" / "run_optimized.py"
    )
    name = f"parallel_fci_intersection_remote_numeric_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen numerical source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _case_tasks(numeric, case: str) -> list[dict[str, Any]]:
    setting = CASE_SETTINGS[case]
    return numeric._tasks(
        setting["max_depth"], setting["order"], setting["eta_order"], setting["preflight"]
    )


def _unit_identity(campaign: Mapping[str, Any], case: str, index: int, task: Mapping[str, Any]) -> str:
    return _digest({
        "schema": UNIT_SCHEMA,
        "campaign": campaign["identity"],
        "case": case,
        "index": int(index),
        "task": dict(task),
        "settings": CASE_SETTINGS[case],
    })


def _unit_paths(output: Path, case: str, index: int) -> tuple[Path, Path]:
    root = output / "units" / case
    stem = f"plane_{index:03d}"
    return root / f"{stem}.npz", root / f"{stem}.json"


def _load_unit(
    output: Path,
    campaign: Mapping[str, Any],
    case: str,
    index: int,
    task: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    path, receipt_path = _unit_paths(output, case, index)
    if not path.is_file() or not receipt_path.is_file():
        return None
    receipt = _load_json(receipt_path)
    expected = _unit_identity(campaign, case, index, task)
    if receipt.get("identity") != expected:
        raise RuntimeError(f"stale unit identity: {receipt_path}")
    if _sha256(path) != receipt.get("payload_sha256"):
        raise RuntimeError(f"unit payload hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != set(ARRAY_KEYS):
            raise RuntimeError(f"unit array coverage mismatch: {path}")
        arrays = {key: payload[key] for key in payload.files}
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise RuntimeError(f"unit contains nonfinite data: {path}")
    return arrays, receipt


def _init_worker(source_root: str, output: str) -> None:
    global _NUMERIC
    source = Path(source_root)
    destination = Path(output)
    os.chdir(source)
    _NUMERIC = _load_numeric(source, destination)
    _NUMERIC._worker_init()


def _probe_worker_initialization(source_root: str, output: str, result_queue) -> None:
    """Fail once, with a bounded diagnostic, before starting a respawning Pool."""
    try:
        _init_worker(source_root, output)
    except BaseException:
        result_queue.put({"ok": False, "traceback": traceback.format_exc()})
    else:
        result_queue.put({"ok": True, "peak_rss_gib": _peak_rss_gib()})


def _probe_initialization(context, source_root: Path, output: Path) -> dict[str, Any]:
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_probe_worker_initialization,
        args=(str(source_root), str(output), result_queue),
        name="campaign-initialization-probe",
    )
    process.start()
    process.join()
    try:
        result = result_queue.get(timeout=2.0)
    except queue.Empty as exc:
        raise RuntimeError(
            f"worker initialization probe produced no result (exit code {process.exitcode})"
        ) from exc
    finally:
        result_queue.close()
        result_queue.join_thread()
    if process.exitcode != 0 or not result.get("ok"):
        detail = result.get("traceback", "no traceback was returned")
        raise RuntimeError(f"worker initialization probe failed:\n{detail}")
    return result


def _peak_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (2**30 if sys.platform == "darwin" else 2**20)


def _compute_unit(payload: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    if _NUMERIC is None:
        raise RuntimeError("worker was not initialized")
    index, task = payload
    started = time.monotonic()
    data = _NUMERIC._worker_integrate(task)
    arrays = {key: np.asarray(data.pop(key)) for key in ARRAY_KEYS}
    return {
        "index": int(index),
        "task": task,
        "arrays": arrays,
        "stats": data,
        "seconds": time.monotonic() - started,
        "worker_pid": os.getpid(),
        "peak_rss_gib": _peak_rss_gib(),
    }


def _effective_workers(
    requested: int,
    *,
    memory_budget_gib: float | None,
    worker_memory_gib: float | None,
    memory_reserve_gib: float,
) -> int:
    if (memory_budget_gib is None) != (worker_memory_gib is None):
        raise ValueError("memory budget and worker memory must be supplied together")
    if memory_budget_gib is None:
        return requested
    available = memory_budget_gib - memory_reserve_gib
    cap = math.floor(available / float(worker_memory_gib))
    if cap < 1:
        raise ValueError("memory budget cannot accommodate one worker")
    return min(requested, cap)


def run_units(args: argparse.Namespace, case: str) -> dict[str, Any]:
    output = args.output.resolve()
    campaign = _campaign(output)
    source_root = Path(campaign["source_root"])
    numeric = _load_numeric(source_root, output)
    tasks = _case_tasks(numeric, case)
    missing = []
    reused = 0
    for index, task in enumerate(tasks):
        if _load_unit(output, campaign, case, index, task) is None:
            missing.append((index, task))
        else:
            reused += 1
    requested = int(args.workers)
    workers = _effective_workers(
        requested,
        memory_budget_gib=args.memory_budget_gib,
        worker_memory_gib=args.worker_memory_gib,
        memory_reserve_gib=args.memory_reserve_gib,
    )
    workers = min(workers, max(1, len(missing)))
    started = time.monotonic()
    completed = reused
    worker_records = []
    if missing:
        context = mp.get_context("spawn")
        initialization_probe = _probe_initialization(context, source_root, output)
        with context.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(str(source_root), str(output)),
            maxtasksperchild=int(args.max_tasks_per_worker),
        ) as pool:
            for result in pool.imap_unordered(_compute_unit, missing):
                index = int(result["index"])
                task = result["task"]
                path, receipt_path = _unit_paths(output, case, index)
                _atomic_npz(path, **result["arrays"])
                receipt = {
                    "schema": UNIT_SCHEMA,
                    "identity": _unit_identity(campaign, case, index, task),
                    "campaign": campaign["identity"],
                    "case": case,
                    "index": index,
                    "task": task,
                    "payload_sha256": _sha256(path),
                    "stats": result["stats"],
                    "seconds": result["seconds"],
                    "worker_pid": result["worker_pid"],
                    "peak_rss_gib": result["peak_rss_gib"],
                }
                _atomic_json(receipt_path, receipt)
                completed += 1
                worker_records.append({
                    key: receipt[key]
                    for key in ("index", "seconds", "worker_pid", "peak_rss_gib")
                })
                _atomic_json(output / "progress.json", {
                    "updated_unix": time.time(), "stage": case,
                    "completed": completed, "total": len(tasks),
                    "elapsed_seconds": time.monotonic() - started,
                })
    else:
        initialization_probe = None
    receipt = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign": campaign["identity"],
        "case": case,
        "requested_workers": requested,
        "effective_workers": workers,
        "initialization_probe": initialization_probe,
        "max_tasks_per_worker": int(args.max_tasks_per_worker),
        "memory_budget_gib": args.memory_budget_gib,
        "worker_memory_gib": args.worker_memory_gib,
        "memory_reserve_gib": args.memory_reserve_gib,
        "reused_units": reused,
        "computed_units": len(missing),
        "completed_units": completed,
        "total_units": len(tasks),
        "seconds": time.monotonic() - started,
        "workers": worker_records,
    }
    _atomic_json(output / "operations" / f"{case}_run.json", receipt)
    return receipt


def _assemble_arrays(
    output: Path, campaign: Mapping[str, Any], numeric, case: str
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    tasks = _case_tasks(numeric, case)
    recipients = numeric._recipients()
    volume = np.zeros((2, 9, len(recipients)))
    moment = np.zeros((2, 9, len(recipients), 3))
    source = np.zeros((2, 9, len(recipients), len(numeric.FIELDS)))
    plane_stats = []
    for index, task in enumerate(tasks):
        loaded = _load_unit(output, campaign, case, index, task)
        if loaded is None:
            raise RuntimeError(f"missing {case} unit {index}")
        data, receipt = loaded
        slab = int(receipt["stats"]["slab"])
        tile = data["tile"]
        recipient_index = data["recipient_index"]
        count = len(tile)
        np.add.at(volume, (np.full(count, slab), tile, recipient_index), data["weight"])
        np.add.at(moment, (np.full(count, slab), tile, recipient_index), data["moment"])
        np.add.at(source, (np.full(count, slab), tile, recipient_index), data["source"])
        plane_stats.append(receipt["stats"] | {
            "worker_seconds": receipt["seconds"],
            "peak_rss_gib": receipt["peak_rss_gib"],
        })
    tile_ij = np.asarray([[i, j] for i in numeric.TILE_I for j in numeric.TILE_J])
    return {
        "recipients": recipients,
        "volume": volume,
        "moment": moment,
        "source": source,
        "tile_ij": tile_ij,
    }, plane_stats


def assemble(output: Path, case: str) -> dict[str, Any]:
    output = output.resolve()
    campaign = _campaign(output)
    numeric = _load_numeric(Path(campaign["source_root"]), output)
    arrays, plane_stats = _assemble_arrays(output, campaign, numeric, case)
    summary: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign": campaign["identity"],
        "case": case,
        "settings": CASE_SETTINGS[case],
        "plane_stats": plane_stats,
        "optimization_totals": {
            key: int(sum(stat[key] for stat in plane_stats))
            for key in (
                "nfev", "trace_calls", "trace_requested_points", "trace_unique_points",
                "trace_padded_points", "trace_cache_hits", "integration_nodes",
                "contributing_nodes", "skipped_source_nodes",
            )
        },
    }
    if case != "preflight":
        source_root = Path(campaign["source_root"])
        shared_root = source_root / "work" / "parallel_fci_shared_patch_20260922"
        with np.load(shared_root / "sensitivity" / "exact" / "arrays.npz") as old:
            residuals = {"cap": old["tube_cap"], "source": old["tube_source"]}
            tube_volume = old["tube_volume"]
        recipients = arrays["recipients"]
        selected_index = np.asarray([
            int(np.flatnonzero(recipients == owner)[0]) for owner in numeric.SELECTED
        ])
        final = _load_json(shared_root / "final" / "summary.json")["sensitivity"]["exact"]
        qualified = np.asarray([row["qualified_action"] for row in final["owners"]])
        coordinate = np.asarray([row["coordinate_action"] for row in final["owners"]])
        archived_volume = np.asarray([row["archived_volume"] for row in final["owners"]])
        continuous_volume = np.asarray([row["continuous_coordinate_volume"] for row in final["owners"]])
        oracle = arrays["source"].sum(axis=(0, 1))
        projections = {}
        for name, residual in residuals.items():
            constant, _fraction = numeric.baseline.constant_projection(residual, arrays["volume"])
            linear, diagnostics = numeric.baseline.linear_projection(
                residual, arrays["volume"], arrays["moment"], arrays["tile_ij"]
            )
            projections[name] = {
                "constant": constant, "linear": linear, "diagnostics": diagnostics
            }
            arrays[f"{name}_constant_by_recipient"] = constant
            arrays[f"{name}_linear_by_recipient"] = linear
        selected_volume = np.asarray([
            np.sum(arrays["volume"][slab, :, selected_index[slab]]) for slab in range(2)
        ])

        def rms(value: np.ndarray) -> list[float]:
            return np.sqrt(
                np.sum(archived_volume[:, None] * value * value, axis=0)
                / np.sum(archived_volume)
            ).tolist()

        oracle_action = oracle[selected_index] / archived_volume[:, None]
        summary.update({
            "selected_owners": list(numeric.SELECTED),
            "selected_coverage_ratio": (selected_volume / continuous_volume).tolist(),
            "tube_volume_coverage_ratio": (arrays["volume"].sum(axis=2) / tube_volume).tolist(),
            "oracle_action": oracle_action.tolist(),
            "qualified_action": qualified.tolist(),
            "coordinate_action": coordinate.tolist(),
            "oracle_minus_coordinate_weighted_RMS": rms(oracle_action - coordinate),
            "projection": {},
            "patch_source_minus_saved_tube_source_integrated": (
                np.sum(arrays["source"], axis=(0, 1, 2))
                - np.sum(residuals["source"], axis=(0, 1))
            ).tolist(),
        })
        for name, residual in residuals.items():
            record = {}
            for degree in ("constant", "linear"):
                selected_action = projections[name][degree][selected_index] / archived_volume[:, None]
                record[degree] = {
                    "selected_action": selected_action.tolist(),
                    "minus_oracle_weighted_RMS": rms(selected_action - oracle_action),
                    "minus_qualified_weighted_RMS": rms(selected_action - qualified),
                    "integrated_conservation_residual": (
                        np.sum(projections[name][degree], axis=0)
                        - np.sum(residual, axis=(0, 1))
                    ).tolist(),
                }
            record["linear_fit"] = projections[name]["diagnostics"]
            summary["projection"][name] = record
        arrays.update({
            "selected_index": selected_index, "oracle_by_recipient": oracle,
            "qualified_action": qualified, "coordinate_action": coordinate,
            "archived_volume": archived_volume,
        })
    directory = output / "cases" / case
    _atomic_npz(directory / "arrays.npz", **arrays)
    summary["arrays_sha256"] = _sha256(directory / "arrays.npz")
    _atomic_json(directory / "summary.json", summary)
    return summary


def validate_case(output: Path, case: str) -> dict[str, Any]:
    output = output.resolve()
    campaign = _campaign(output)
    numeric = _load_numeric(Path(campaign["source_root"]), output)
    tasks = _case_tasks(numeric, case)
    for index, task in enumerate(tasks):
        if _load_unit(output, campaign, case, index, task) is None:
            raise RuntimeError(f"missing {case} unit {index}")
    directory = output / "cases" / case
    summary = _load_json(directory / "summary.json")
    if summary.get("campaign") != campaign["identity"] or summary.get("case") != case:
        raise RuntimeError("assembled case identity mismatch")
    if _sha256(directory / "arrays.npz") != summary.get("arrays_sha256"):
        raise RuntimeError("assembled case hash mismatch")
    if len(summary["plane_stats"]) != len(tasks):
        raise RuntimeError("assembled plane coverage incomplete")
    if max(stat["homogeneous_hint_disagreement"] for stat in summary["plane_stats"]) != 0:
        raise RuntimeError("homogeneous hint disagreement")
    if case != "preflight":
        for source in ("cap", "source"):
            for degree in ("constant", "linear"):
                residual = summary["projection"][source][degree]["integrated_conservation_residual"]
                if max(abs(value) for value in residual) > 2.0e-18:
                    raise RuntimeError("projection conservation failure")
        if min(
            summary["projection"]["cap"]["linear_fit"]["rank_min"],
            summary["projection"]["source"]["linear_fit"]["rank_min"],
        ) < 3:
            raise RuntimeError("linear fit rank failure")
    return {"case": case, "validated": True, "planes": len(tasks)}


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    run_units(args, "preflight")
    assemble(args.output, "preflight")
    validation = validate_case(args.output, "preflight")
    campaign = _campaign(args.output.resolve())
    source_root = Path(campaign["source_root"])
    reference_path = (
        source_root / "work" / "parallel_fci_intersection_eta_followup_20260922"
        / "intersection" / "preflight" / "arrays.npz"
    )
    candidate_path = args.output.resolve() / "cases" / "preflight" / "arrays.npz"
    comparisons = {}
    with np.load(reference_path, allow_pickle=False) as reference, np.load(candidate_path, allow_pickle=False) as candidate:
        if set(reference.files) != set(candidate.files):
            raise RuntimeError("preflight array coverage differs")
        for key in reference.files:
            left, right = reference[key], candidate[key]
            if left.dtype.kind in "iu":
                if not np.array_equal(left, right):
                    raise RuntimeError(f"preflight integer mismatch: {key}")
                comparisons[key] = {"exact": True}
            else:
                difference = np.abs(left - right)
                record = {"max_abs": float(np.max(difference, initial=0.0))}
                comparisons[key] = record
                if not np.allclose(
                    left, right, rtol=EQUIVALENCE_RTOL, atol=EQUIVALENCE_ATOL
                ):
                    raise RuntimeError(f"preflight floating mismatch: {key}: {record}")
    result = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign": campaign["identity"],
        "validation": validation,
        "rtol": EQUIVALENCE_RTOL,
        "atol": EQUIVALENCE_ATOL,
        "comparisons": comparisons,
    }
    _atomic_json(args.output.resolve() / "preflight.json", result)
    return result


def finalize(output: Path) -> dict[str, Any]:
    output = output.resolve()
    campaign = _campaign(output)
    base_validation = validate_case(output, "base")
    refined_validation = validate_case(output, "refined")
    base = _load_json(output / "cases" / "base" / "summary.json")
    refined = _load_json(output / "cases" / "refined" / "summary.json")
    source_root = Path(campaign["source_root"])
    eta_root = source_root / "work" / "parallel_fci_intersection_eta_followup_20260922" / "eta_gradient" / "results"
    eta = _load_json(eta_root / "summary.json")
    change = {
        "coverage_ratio": (
            np.asarray(refined["selected_coverage_ratio"])
            - np.asarray(base["selected_coverage_ratio"])
        ).tolist(),
        "oracle_action": (
            np.asarray(refined["oracle_action"]) - np.asarray(base["oracle_action"])
        ).tolist(),
        "projection": {},
    }
    for source in ("cap", "source"):
        change["projection"][source] = {}
        for degree in ("constant", "linear"):
            change["projection"][source][degree] = (
                np.asarray(refined["projection"][source][degree]["selected_action"])
                - np.asarray(base["projection"][source][degree]["selected_action"])
            ).tolist()
    result = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign": campaign["identity"],
        "validation": {"base": base_validation, "refined": refined_validation},
        "intersection": {"base": base, "refined": refined, "change": change},
        "eta_gradient": eta,
        "scope": (
            "independent boundary-aware return and fixed-cap shifted-plane derivative "
            "diagnostics; no global qualification"
        ),
    }
    arrays = {}
    for prefix, path in (
        ("intersection_base", output / "cases" / "base" / "arrays.npz"),
        ("intersection_refined", output / "cases" / "refined" / "arrays.npz"),
        ("eta", eta_root / "arrays.npz"),
    ):
        with np.load(path, allow_pickle=False) as payload:
            for key in payload.files:
                arrays[f"{prefix}_{key}"] = payload[key]
    _atomic_npz(output / "final" / "comparison.npz", **arrays)
    result["comparison_sha256"] = _sha256(output / "final" / "comparison.npz")
    _atomic_json(output / "final" / "summary.json", result)
    return result


def validate_final(output: Path) -> dict[str, Any]:
    output = output.resolve()
    campaign = _campaign(output)
    result = _load_json(output / "final" / "summary.json")
    if result.get("campaign") != campaign["identity"]:
        raise RuntimeError("final campaign identity mismatch")
    if _sha256(output / "final" / "comparison.npz") != result.get("comparison_sha256"):
        raise RuntimeError("final comparison hash mismatch")
    expected_eta = {f"f{x:g}_w{width}" for x in (1.0, 0.5, 0.25) for width in (2, 4)}
    if set(result["eta_gradient"]["aggregate"]) != expected_eta:
        raise RuntimeError("eta result coverage incomplete")
    return {"validated": True, "summary": str(output / "final" / "summary.json")}


def provenance(output: Path) -> dict[str, Any]:
    output = output.resolve()
    campaign = _campaign(output)
    records = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if ".campaign.lock" in path.name or "cache" in path.parts:
            continue
        if path.name == "provenance.json":
            continue
        records[str(path.relative_to(output))] = {
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
    result = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign": campaign["identity"],
        "files": records,
        "external_inputs": campaign["external_inputs"],
        "input_links": campaign["input_links"],
    }
    _atomic_json(output / "provenance.json", result)
    return result


def _record_operation(
    output: Path, command: str, started: float, status: str, error: str | None = None
) -> None:
    operations = output / "operations"
    operations.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": CAMPAIGN_SCHEMA,
        "command": command,
        "status": status,
        "error": error,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "argv": sys.argv,
        "started_unix": time.time() - (time.monotonic() - started),
        "finished_unix": time.time(),
        "seconds": time.monotonic() - started,
    }
    _atomic_json(operations / f"command_{int(time.time() * 1000)}_{command}.json", record)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "verify", "preflight", "run", "assemble", "validate",
            "finalize", "validate-final", "provenance",
        ),
    )
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("base", "refined"))
    parser.add_argument("--workers", type=_positive_int)
    parser.add_argument("--max-tasks-per-worker", type=_positive_int, default=16)
    parser.add_argument("--memory-budget-gib", type=_positive_float)
    parser.add_argument("--worker-memory-gib", type=_positive_float)
    parser.add_argument("--memory-reserve-gib", type=_nonnegative_float, default=1.0)
    args = parser.parse_args()
    if args.command in {"verify", "preflight"} and args.input_root is None:
        parser.error(f"{args.command} requires --input-root")
    if args.command in {"preflight", "run"} and args.workers is None:
        parser.error(f"{args.command} requires --workers chosen for the allocation")
    if args.command in {"run", "assemble", "validate"} and args.case is None:
        parser.error(f"{args.command} requires --case")

    output = args.output.resolve()
    started = time.monotonic()
    status = "failed"
    error = None
    try:
        with _exclusive_output(output):
            if args.command == "verify":
                result = verify(args.input_root, output)
            elif args.command == "preflight":
                verify(args.input_root, output)
                result = preflight(args)
            elif args.command == "run":
                result = run_units(args, args.case)
            elif args.command == "assemble":
                result = assemble(output, args.case)
            elif args.command == "validate":
                result = validate_case(output, args.case)
            elif args.command == "finalize":
                result = finalize(output)
            elif args.command == "validate-final":
                result = validate_final(output)
            else:
                result = provenance(output)
        status = "complete"
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception:
        error = traceback.format_exc()
        raise
    finally:
        try:
            _record_operation(output, args.command, started, status, error)
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
