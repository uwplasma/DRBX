#!/usr/bin/env python3
"""Global actual-HSX P/G/O bracket diagnostic for the P04 qualification.

P is the unchanged production action.  G and O replace every eligible planar
advected face value by one common degree-two reconstructed value, using
geometric owner moments or the stored midpoint-observation functional.  In the
scalar-upwind action that common value is installed on both characteristic
sides: this evaluates consistency through the upwind assembly but deliberately
does not preserve production left/right stabilization.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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

import jax
import numpy as np
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import simulate_hsx_mms as mms  # noqa: E402
from hsx_mms_continuum_reference import build_continuum_reference_from_sidecar  # noqa: E402
from drbx.geometry.fci_aggregate_reconstruction import (  # noqa: E402
    AggregateEvaluation,
    AggregateEvaluationDiagnostics,
)
from drbx.native.fci_boundaries import CoordinateFaceValues3D  # noqa: E402

import audit_hsx_face_candidate_orientation as orientation  # noqa: E402
import audit_hsx_p03_fine_interval as bridge  # noqa: E402
import audit_hsx_poisson_common_face_oracle as oracle  # noqa: E402
import audit_hsx_poisson_continuous_baseline as continuous  # noqa: E402
import audit_hsx_poisson_vorticity as base  # noqa: E402


SCHEMA = "drbx.hsx-p04-global-bracket-case-v1"
SUMMARY_SCHEMA = "drbx.hsx-p04-global-bracket-summary-v1"
ROW_CACHE_SCHEMA = "drbx.hsx-p04-global-bracket-rows-v1"
FIELDS = continuous.FIELDS
RULES = ("P", "G", "O")
PARTS = ("all", "radial", "angular")
BOUNDARY_CONTRACT = {
    "physical_wall_model": "legacy-velocity-trace",
    "parallel_velocity_wall_bc": "neumann",
    "neumann_ghost_scheme": "physical",
    "parallel_boundary_pairing": "characteristic-sat",
    "parallel_characteristic_wall_law": "energy-absorbing",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
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


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _progress(stage: str, **details: Any) -> None:
    print(json.dumps({"event": "progress", "stage": stage, **details}, sort_keys=True), flush=True)


def _max_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / (1024.0**3)


def _load_cache(path: Path, resolution: int, time_value: float):
    state, regular, eta, provenance = orientation._load_midpoint_reference_cache(
        path, expected_resolution=resolution, expected_time=time_value
    )
    with np.load(path, allow_pickle=False) as cached:
        exact = {
            name: np.asarray(cached[f"exact_{name}"], dtype=np.float64)
            for name in FIELDS
        }
    return state, regular, eta, exact, provenance


def _candidate_face_geometry(model: Any, operands: Any):
    """Return every eligible planar face key and its regular-chart point."""

    trace_masks = []
    for axis_name in "xy":
        phi = np.asarray(getattr(operands.phi_trace, f"mask_{axis_name}"), dtype=bool)
        field = np.asarray(getattr(operands.omega_trace, f"mask_{axis_name}"), dtype=bool)
        trace_masks.append(phi | field)

    centers = tuple(
        np.asarray(axis.centers_owned, dtype=np.float64)
        for axis in (model.geometry.grid.x, model.geometry.grid.y, model.geometry.grid.z)
    )
    faces = tuple(
        np.asarray(axis.faces_owned, dtype=np.float64)
        for axis in (model.geometry.grid.x, model.geometry.grid.y, model.geometry.grid.z)
    )
    owned_shape = tuple(int(value) for value in model.geometry.owned_shape)
    key_chunks = []
    point_chunks = []
    eta_chunks = []
    for axis in (0, 1):
        shape = np.asarray(getattr(operands.omega_stencil.face_values, "xy"[axis])).shape
        index = np.indices(shape, dtype=np.int32).reshape(3, -1).T
        valid = ~trace_masks[axis].reshape(-1)
        if axis == 0:
            valid &= index[:, 0] > 0
            valid &= index[:, 0] < owned_shape[0]
            u = faces[0][index[:, 0]]
            theta = centers[1][index[:, 1]]
        else:
            u = centers[0][index[:, 0]]
            theta = faces[1][index[:, 1]]
        index = index[valid]
        u = u[valid]
        theta = theta[valid]
        keys = np.column_stack(
            (np.full(index.shape[0], axis, dtype=np.int32), index)
        )
        key_chunks.append(keys)
        point_chunks.append(np.column_stack((u * np.cos(theta), u * np.sin(theta))))
        eta_chunks.append(index[:, 2].astype(np.int64, copy=False))
    keys = np.concatenate(key_chunks, axis=0)
    points = np.concatenate(point_chunks, axis=0).astype(np.float64, copy=False)
    eta_indices = np.concatenate(eta_chunks, axis=0)
    return keys, points, eta_indices


def _replace_faces_vectorized(
    original: CoordinateFaceValues3D,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    axis_filter: int | None = None,
) -> CoordinateFaceValues3D:
    arrays = [np.asarray(getattr(original, name), dtype=np.float64).copy() for name in "xyz"]
    for axis in (0, 1):
        if axis_filter is not None and axis != axis_filter:
            continue
        selected = keys[:, 0] == axis
        index = keys[selected, 1:4]
        arrays[axis][index[:, 0], index[:, 1], index[:, 2]] = values[selected]
    return CoordinateFaceValues3D(*arrays)


def _global_action_runner(model: Any, operands: Any):
    """Return one reusable JIT for P replay and common-face G/O actions."""

    def inner(field_faces, phi_faces, direct_left, direct_right):
        field_stencil = replace(operands.omega_stencil, face_values=field_faces)
        phi_stencil = replace(operands.phi_stencil, face_values=phi_faces)
        upwind = base._operator_call(
            model,
            operands.phi_stencil,
            operands.omega_stencil,
            f_trace=operands.phi_trace,
            g_trace=operands.omega_trace,
            characteristic_scheme="scalar-third-order-upwind",
            g_halo=operands.omega_halo,
            g_direct_states=(direct_left, direct_right),
        )
        action_a = base._operator_call(
            model,
            operands.phi_stencil,
            field_stencil,
            f_trace=operands.phi_trace,
            g_trace=operands.omega_trace,
            characteristic_scheme="scalar-centered",
        )
        action_b_raw = base._operator_call(
            model,
            operands.omega_stencil,
            phi_stencil,
            f_trace=operands.omega_trace,
            g_trace=operands.phi_trace,
            characteristic_scheme="scalar-centered",
        )
        centered = base._operator_call(
            model,
            phi_stencil,
            field_stencil,
            f_trace=operands.phi_trace,
            g_trace=operands.omega_trace,
            characteristic_scheme="centered",
        )
        swapped = base._operator_call(
            model,
            field_stencil,
            phi_stencil,
            f_trace=operands.omega_trace,
            g_trace=operands.phi_trace,
            characteristic_scheme="centered",
        )
        return base._return_actions_to_owner(
            model, (upwind, action_a, action_b_raw, centered, swapped)
        )[0]

    compiled = jax.jit(inner)

    def numeric(field_faces, phi_faces, direct_left, direct_right):
        values = jax.block_until_ready(
            compiled(field_faces, phi_faces, direct_left, direct_right)
        )
        return tuple(np.asarray(value) for value in values)

    return numeric


def _evaluation_cache_metadata(
    *,
    rule: str,
    resolution: int,
    manifest_hash: str,
    reference_cache_hash: str,
    sidecar_hash: str,
    keys: np.ndarray,
) -> dict[str, Any]:
    implementation_files = [
        Path(__file__).resolve(),
        ROOT / "src/drbx/geometry/fci_aggregate_reconstruction.py",
        ROOT / "scripts/audit_hsx_face_candidate_orientation.py",
    ]
    return {
        "schema": ROW_CACHE_SCHEMA,
        "rule": rule,
        "resolution": resolution,
        "geometry_manifest_sha256": manifest_hash,
        "reference_cache_sha256": reference_cache_hash,
        "reference_sidecar_sha256": sidecar_hash,
        "face_key_sha256": _array_sha256(keys),
        "degree": 2,
        "donor_selection_policy": "nearest same-eta owners; 12->96 expansion schedule",
        "condition_limit": 1.0e8,
        "row_l1_limit": 8.0,
        "weight_power": 4.0,
        "boundary_contract": BOUNDARY_CONTRACT,
        "requested_checks": [
            "global_l2", "regional_budget", "A_B_C", "constant_reproduction",
            "orientation_closure", "swapped_antisymmetry",
        ],
        "implementation_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in implementation_files
        },
    }


def _save_evaluation(path: Path, evaluation: AggregateEvaluation, metadata: Mapping[str, Any]) -> None:
    matrix = evaluation.matrix.tocsr(copy=True)
    matrix.sort_indices()
    diagnostics = evaluation.diagnostics
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(
        temporary,
        metadata=np.asarray(_canonical(metadata)),
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.asarray(matrix.shape, dtype=np.int64),
        rank=diagnostics.rank,
        condition=diagnostics.condition,
        reproduction_residual=diagnostics.reproduction_residual,
        l1_norm=diagnostics.l1_norm,
        donor_counts=diagnostics.donor_counts,
        expansion_steps=diagnostics.expansion_steps,
    )
    temporary.replace(path)


def _load_evaluation(path: Path, metadata: Mapping[str, Any], points: np.ndarray, eta: np.ndarray):
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as cached:
        if str(cached["metadata"].item()) != _canonical(metadata):
            return None
        shape = tuple(int(value) for value in cached["shape"])
        matrix = csr_matrix(
            (cached["data"], cached["indices"], cached["indptr"]), shape=shape
        )
        diagnostics = AggregateEvaluationDiagnostics(
            rank=np.asarray(cached["rank"]),
            condition=np.asarray(cached["condition"]),
            reproduction_residual=np.asarray(cached["reproduction_residual"]),
            l1_norm=np.asarray(cached["l1_norm"]),
            donor_counts=np.asarray(cached["donor_counts"]),
            degree=2,
            basis_size=6,
            rank_deficient_count=int(np.count_nonzero(cached["rank"] < 6)),
            ill_conditioned_count=int(np.count_nonzero(~np.isfinite(cached["condition"]))),
            reproduction_failure_count=int(np.count_nonzero(cached["reproduction_residual"] > 1.0e-10)),
            l1_exceeded_count=int(np.count_nonzero(cached["l1_norm"] > 8.0)),
            max_condition=float(np.max(cached["condition"])),
            max_reproduction_residual=float(np.max(cached["reproduction_residual"])),
            max_l1_norm=float(np.max(cached["l1_norm"])),
            initial_donor_count=12,
            expansion_steps=np.asarray(cached["expansion_steps"]),
        )
    return AggregateEvaluation(matrix, diagnostics, points, eta)


def _locality(evaluation: AggregateEvaluation, aggregate: Any) -> dict[str, float]:
    matrix = evaluation.matrix.tocsr()
    row_max = np.zeros(matrix.shape[0], dtype=np.float64)
    for first in range(0, matrix.shape[0], 8192):
        last = min(first + 8192, matrix.shape[0])
        counts = np.diff(matrix.indptr[first : last + 1])
        rows = np.repeat(np.arange(first, last), counts)
        columns = matrix.indices[matrix.indptr[first] : matrix.indptr[last]]
        distance = np.linalg.norm(
            aggregate.centroid_xy[columns] - evaluation.points_xy[rows], axis=1
        )
        offset = 0
        for local, count in enumerate(counts):
            row_max[first + local] = float(np.max(distance[offset : offset + count]))
            offset += int(count)
    return {
        "maximum_regular_chart_radius": float(np.max(row_max)),
        "p95_regular_chart_radius": float(np.quantile(row_max, 0.95)),
        "median_regular_chart_radius": float(np.median(row_max)),
    }


def _diagnostics(evaluation: AggregateEvaluation, aggregate: Any, points: np.ndarray):
    return {
        **oracle._evaluation_diagnostics(evaluation),
        "polynomial_reproduction": oracle._polynomial_reproduction(
            aggregate, evaluation, points
        ),
        "locality": _locality(evaluation, aggregate),
        "support_matrix_sha256": _array_sha256(evaluation.matrix.indices),
    }


def _periodic_shared_face_mismatch(face_values: CoordinateFaceValues3D) -> float:
    angular = np.asarray(face_values.y, dtype=np.float64)
    if angular.shape[1] < 2:
        return 0.0
    return float(np.max(np.abs(angular[:, 0, :] - angular[:, -1, :])))


def _face_values_at_keys(face_values: CoordinateFaceValues3D, keys: np.ndarray) -> np.ndarray:
    result = np.empty(len(keys), dtype=np.float64)
    for axis in (0, 1):
        selected = keys[:, 0] == axis
        index = keys[selected, 1:4]
        values = np.asarray(getattr(face_values, "xy"[axis]), dtype=np.float64)
        result[selected] = values[index[:, 0], index[:, 1], index[:, 2]]
    return result


def _baseline_expected(summary: Mapping[str, Any], field: str, resolution: int, variant: str):
    resolutions = [int(value) for value in summary["resolutions"]]
    index = resolutions.index(int(resolution))
    return float(summary["fields"][field]["variants"][variant]["absolute_l2"][index])


def _n32_equivalence(
    prior_path: Path,
    keys: np.ndarray,
    candidate_values: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    if not prior_path.is_file():
        return {"available": False, "path": str(prior_path)}
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    key_to_row = {tuple(int(v) for v in key): index for index, key in enumerate(keys)}
    result = {"available": True, "path": str(prior_path), "fields": {}}
    matched = prior.get("matched_functional_comparison", {})
    prior_fields_by_rule = {
        "G": matched.get("geometric_moment_fields", prior.get("fields", {})),
        "O": matched.get("midpoint_observation_fields", {}),
    }
    for field in FIELDS:
        compared = {"G": [], "O": []}
        for rule in ("G", "O"):
            prior_fields = prior_fields_by_rule[rule]
            if field not in prior_fields:
                continue
            for record in prior_fields[field]["face_records"]:
                key = tuple(int(v) for v in record["key"])
                if key not in key_to_row or record.get("collapsed_axis_face"):
                    continue
                row = key_to_row[key]
                # Both global candidates act on the stored midpoint-projected
                # owner data.  The older G artifact also carries a separately
                # labelled integrated-average control; that is not equivalent
                # to this diagnostic and must not be compared here.
                prior_value = record.get("candidate_stored_owner_face_state")
                if prior_value is not None:
                    compared[rule].append(
                        abs(float(candidate_values[rule][field][row]) - float(prior_value))
                    )
        result["fields"][field] = {
            rule: {
                "compared_face_count": len(values),
                "maximum_abs_difference": max(values, default=0.0),
            }
            for rule, values in compared.items()
        }
    return result


def _case(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    resolution = int(args.resolution)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _progress("case_start", resolution=resolution)

    artifact_path, artifact = base._load_artifact(args.geometry, resolution)
    _, identity_artifact = base._load_artifact(args.geometry, 64)
    manifest_hash = _sha256(artifact_path / "manifest.json")
    reference_cache = args.baseline / f"N{resolution}.reference.npz"
    raw_state, regular_raw, eta_raw, exact_raw, cache_provenance = _load_cache(
        reference_cache, resolution, float(args.time)
    )
    if cache_provenance["geometry_manifest_sha256"] != manifest_hash:
        raise ValueError("candidate artifact and qualified cache manifest mismatch")
    sidecar_hash = _sha256(args.reference_sidecar)
    cache_hash = _sha256(reference_cache)

    reference = build_continuum_reference_from_sidecar(
        args.reference_sidecar,
        verify_hashes=False,
        tau=mms.PHYSICAL_PARAMETERS["tau"],
        mi_over_me=mms.PHYSICAL_PARAMETERS["mi_over_me"],
        rho_star=mms.PHYSICAL_PARAMETERS["rho_star"],
        Ve_nu=mms.PHYSICAL_PARAMETERS["Ve_nu"],
        perp_diffusion=mms.PHYSICAL_PARAMETERS["density_D_perp"],
        enable_generalized_potential=True,
    )
    if cache_provenance["reference_source"].get("sidecar_sha256") != sidecar_hash:
        raise ValueError("qualified sidecar identity mismatch")

    runtime_args = SimpleNamespace(
        shard_counts=(1, 1, 1),
        curvature_edge_one_form=False,
        reference=reference,
        metric_context=SimpleNamespace(
            metric_evaluator=reference.metric_evaluator,
            bfield=reference.bfield_evaluator,
            nfp=int(identity_artifact.nfp),
        ),
    )
    runtime_started = time.perf_counter()
    with patch.dict(os.environ, base._FROZEN_RUNTIME_ENV, clear=False):
        runtime = mms._runtime(
            artifact.global_geometry, artifact.owner_geometry, runtime_args
        )
    model = runtime.model
    if model is None:
        raise RuntimeError("global P04 audit requires a single-device model")
    runtime_seconds = time.perf_counter() - runtime_started
    rho_star = float(model.parameters.rho_star)

    field_inputs = {
        "actual_vorticity": raw_state,
        "smooth_regular_scalar": raw_state.replace(vorticity=regular_raw),
        "smooth_eta_varying_scalar": raw_state.replace(vorticity=eta_raw),
    }
    owner_states = {
        name: mms._owner_project(state, artifact.owner_geometry)
        for name, state in field_inputs.items()
    }
    operands = {
        name: base._prepare_operands(model, owner_states[name], state)
        for name, state in field_inputs.items()
    }
    representative = operands["actual_vorticity"]
    keys, points_xy, eta_indices = _candidate_face_geometry(model, representative)
    _progress(
        "candidate_face_geometry",
        resolution=resolution,
        face_count=int(len(keys)),
        max_rss_gib=_max_rss_gib(),
    )

    geometric_aggregate = oracle._aggregate_geometry(artifact.owner_geometry)
    observation_aggregate, observation_report = orientation._midpoint_observation_aggregate(
        artifact.owner_geometry, artifact.global_geometry
    )
    aggregates = {"G": geometric_aggregate, "O": observation_aggregate}

    preflight = None
    if resolution == 32:
        count = min(2048, len(keys))
        selected = np.unique(np.linspace(0, len(keys) - 1, count, dtype=np.int64))
        preflight = {"row_count": int(len(selected)), "rules": {}}
        for rule, aggregate in aggregates.items():
            timer = time.perf_counter()
            evaluation = oracle.build_aggregate_evaluation(
                aggregate,
                points_xy[selected],
                eta_indices[selected],
                degree=2,
                donor_count=12,
                max_donor_count=96,
                condition_limit=1.0e8,
                weight_power=4.0,
                row_l1_limit=8.0,
                chunk_size=1024,
            )
            preflight["rules"][rule] = {
                "seconds": time.perf_counter() - timer,
                "diagnostics": _diagnostics(evaluation, aggregate, points_xy[selected]),
            }
        _progress("n32_row_preflight_complete", seconds=sum(v["seconds"] for v in preflight["rules"].values()))

    owner_values = {
        "phi": np.asarray(model._owner_field(owner_states["actual_vorticity"].phi)),
        **{
            name: np.asarray(model._owner_field(owner_states[name].vorticity))
            for name in FIELDS
        },
    }
    candidate_values: dict[str, dict[str, np.ndarray]] = {}
    reconstruction = {}
    row_started = time.perf_counter()
    for rule, aggregate in aggregates.items():
        metadata = _evaluation_cache_metadata(
            rule=rule,
            resolution=resolution,
            manifest_hash=manifest_hash,
            reference_cache_hash=cache_hash,
            sidecar_hash=sidecar_hash,
            keys=keys,
        )
        row_cache = output_dir / f"N{resolution}_{rule}_rows.npz"
        evaluation = _load_evaluation(row_cache, metadata, points_xy, eta_indices)
        cache_hit = evaluation is not None
        timer = time.perf_counter()
        if evaluation is None:
            evaluation = oracle.build_aggregate_evaluation(
                aggregate,
                points_xy,
                eta_indices,
                degree=2,
                donor_count=12,
                max_donor_count=96,
                condition_limit=1.0e8,
                weight_power=4.0,
                row_l1_limit=8.0,
                chunk_size=4096,
            )
            _save_evaluation(row_cache, evaluation, metadata)
        candidate_values[rule] = {
            name: bridge._candidate_values(aggregate, evaluation, values)
            for name, values in owner_values.items()
        }
        reconstruction[rule] = {
            "cache": str(row_cache),
            "cache_sha256": _sha256(row_cache),
            "cache_hit": cache_hit,
            "metadata": metadata,
            "diagnostics": _diagnostics(evaluation, aggregate, points_xy),
            "seconds": time.perf_counter() - timer,
        }
        del evaluation
        gc.collect()
        _progress(
            "reconstruction_complete",
            resolution=resolution,
            rule=rule,
            seconds=reconstruction[rule]["seconds"],
            cache_hit=cache_hit,
            max_rss_gib=_max_rss_gib(),
        )
    row_seconds = time.perf_counter() - row_started

    n32_equivalence = None
    if resolution == 32:
        n32_equivalence = _n32_equivalence(
            args.prior_n32.resolve(), keys, candidate_values
        )

    standard = base._region_masks(artifact.global_geometry, artifact.owner_geometry)
    masks, ordinary_layer, adjacency = bridge._refined_masks(
        artifact.owner_geometry, model.control_volume_geometry.cells, standard
    )
    baseline_summary = json.loads((args.baseline / "summary.json").read_text(encoding="utf-8"))
    exact_owner = {
        name: mms._owner_project_array(exact_raw[name], artifact.owner_geometry)
        for name in FIELDS
    }
    fields_payload = {}
    action_started = time.perf_counter()
    for name in FIELDS:
        field_started = time.perf_counter()
        op = operands[name]
        raw = base._evaluate_variants(model, op)["production_R"]
        baseline = {
            "upwind": -raw["pure_upwind"] / rho_star,
            "A": -raw["phi_centered"] / rho_star,
            "B": raw["omega_centered"] / rho_star,
            "C": -raw["centered"] / rho_star,
            "swapped": -raw["centered_swapped"] / rho_star,
        }
        runner = _global_action_runner(model, op)
        replay_raw = runner(
            op.omega_stencil.face_values,
            op.phi_stencil.face_values,
            op.omega_direct_states[0],
            op.omega_direct_states[1],
        )
        replay = {
            "upwind": -replay_raw[0] / rho_star,
            "A": -replay_raw[1] / rho_star,
            "B": replay_raw[2] / rho_star,
            "C": -replay_raw[3] / rho_star,
            "swapped": -replay_raw[4] / rho_star,
        }
        replay_max = {
            key: float(np.max(np.abs(replay[key] - baseline[key]))) for key in replay
        }

        values_by_rule: dict[str, dict[str, np.ndarray]] = {"P": baseline}
        orientation_closure = {}
        periodic_mismatch = {}
        structural = {}
        for rule in ("G", "O"):
            part_values = {}
            periodic_mismatch[rule] = {}
            for part in PARTS:
                axis_filter = None if part == "all" else (0 if part == "radial" else 1)
                field_faces = _replace_faces_vectorized(
                    op.omega_stencil.face_values,
                    keys,
                    candidate_values[rule][name],
                    axis_filter=axis_filter,
                )
                phi_faces = _replace_faces_vectorized(
                    op.phi_stencil.face_values,
                    keys,
                    candidate_values[rule]["phi"],
                    axis_filter=axis_filter,
                )
                left = _replace_faces_vectorized(
                    op.omega_direct_states[0],
                    keys,
                    candidate_values[rule][name],
                    axis_filter=axis_filter,
                )
                right = _replace_faces_vectorized(
                    op.omega_direct_states[1],
                    keys,
                    candidate_values[rule][name],
                    axis_filter=axis_filter,
                )
                raw_candidate = runner(field_faces, phi_faces, left, right)
                part_values[part] = {
                    "upwind": -raw_candidate[0] / rho_star,
                    "A": -raw_candidate[1] / rho_star,
                    "B": raw_candidate[2] / rho_star,
                    "C_direct": -raw_candidate[3] / rho_star,
                    "swapped": -raw_candidate[4] / rho_star,
                }
                periodic_mismatch[rule][part] = {
                    "field": _periodic_shared_face_mismatch(field_faces),
                    "phi": _periodic_shared_face_mismatch(phi_faces),
                }
                if part == "all":
                    structural[rule] = {
                        "common_face_left_right_max_abs": float(np.max(np.abs(
                            _face_values_at_keys(left, keys)
                            - _face_values_at_keys(right, keys)
                        )))
                    }
                del field_faces, phi_faces, left, right, raw_candidate
            values_by_rule[rule] = {
                **part_values["all"],
                "C": 0.5 * (part_values["all"]["A"] + part_values["all"]["B"]),
            }
            orientation_closure[rule] = {
                action: float(np.max(np.abs(
                    (part_values["all"][action] - baseline[action])
                    - (part_values["radial"][action] - baseline[action])
                    - (part_values["angular"][action] - baseline[action])
                )))
                for action in ("upwind", "A", "B")
            }
            structural[rule].update({
                "centered_decomposition_max_abs": float(np.max(np.abs(
                    values_by_rule[rule]["C_direct"] - values_by_rule[rule]["C"]
                ))),
                "swapped_antisymmetry_max_abs": float(np.max(np.abs(
                    values_by_rule[rule]["C_direct"] + values_by_rule[rule]["swapped"]
                ))),
            })
            del part_values
            gc.collect()

        combinations = {
            "P_P": baseline["C"],
            "G_P": 0.5 * (values_by_rule["G"]["A"] + baseline["B"]),
            "P_G": 0.5 * (baseline["A"] + values_by_rule["G"]["B"]),
            "G_G": values_by_rule["G"]["C"],
            "O_P": 0.5 * (values_by_rule["O"]["A"] + baseline["B"]),
            "P_O": 0.5 * (baseline["A"] + values_by_rule["O"]["B"]),
            "O_O": values_by_rule["O"]["C"],
        }
        action_payload = {
            "scalar_upwind": {
                rule: bridge._regional_statistics(
                    values_by_rule[rule]["upwind"], exact_owner[name],
                    artifact.owner_geometry, masks, ordinary_layer,
                )
                for rule in RULES
            },
            "centered_A": {
                rule: bridge._regional_statistics(
                    values_by_rule[rule]["A"], exact_owner[name],
                    artifact.owner_geometry, masks, ordinary_layer,
                )
                for rule in RULES
            },
            "centered_B": {
                rule: bridge._regional_statistics(
                    values_by_rule[rule]["B"], exact_owner[name],
                    artifact.owner_geometry, masks, ordinary_layer,
                )
                for rule in RULES
            },
            "centered_C": {
                key: bridge._regional_statistics(
                    value, exact_owner[name], artifact.owner_geometry, masks,
                    ordinary_layer,
                )
                for key, value in combinations.items()
            },
        }
        expected_upwind = _baseline_expected(
            baseline_summary, name, resolution, "pure_scalar_upwind"
        )
        expected_centered = _baseline_expected(
            baseline_summary, name, resolution, "current_centered_antisymmetric"
        )
        fields_payload[name] = {
            "actions": action_payload,
            "production_replay": {
                "maximum_action_abs_difference": replay_max,
                "global_l2_difference_from_qualified_summary": {
                    "scalar_upwind": abs(
                        action_payload["scalar_upwind"]["P"]["global"]["absolute_l2"]
                        - expected_upwind
                    ),
                    "centered_C": abs(
                        action_payload["centered_C"]["P_P"]["global"]["absolute_l2"]
                        - expected_centered
                    ),
                },
            },
            "orientation_closure": orientation_closure,
            "periodic_shared_face_mismatch": periodic_mismatch,
            "structural": structural,
            "constant_candidate_contract": {
                rule: {
                    "maximum_face_reproduction_abs": reconstruction[rule]["diagnostics"]["polynomial_reproduction"]["constant"],
                    "candidate_modification_completed_action": 0.0,
                    "reason": "candidate row reproduces one and replaces a common face value; the modification of a constant advected state is identically zero",
                }
                for rule in ("G", "O")
            },
            "timings_seconds": {"total": time.perf_counter() - field_started},
        }
        _progress(
            "field_complete",
            resolution=resolution,
            field=name,
            seconds=fields_payload[name]["timings_seconds"]["total"],
            max_rss_gib=_max_rss_gib(),
        )
        del values_by_rule, baseline, replay, raw, replay_raw, combinations
        gc.collect()

    groups = np.asarray(artifact.owner_geometry.angular_group_size, dtype=np.int64)
    changes = np.flatnonzero(groups[1:] != groups[:-1])
    payload = {
        "schema": SCHEMA,
        "resolution": resolution,
        "time": float(args.time),
        "geometry": {
            "requested_root": str(args.geometry.resolve()),
            "artifact": str(artifact_path.resolve()),
            "manifest_sha256": manifest_hash,
            "cache_manifest_sha256": cache_provenance["geometry_manifest_sha256"],
            "payload_equivalent": cache_provenance["geometry_manifest_sha256"] == manifest_hash,
            "angular_group_size_profile": groups.tolist(),
            "size_change_radial_face_indices": (changes + 1).tolist(),
            "adjacent_owner_pair_count": len(adjacency),
        },
        "reference": {
            "sidecar": str(args.reference_sidecar.resolve()),
            "sidecar_sha256": sidecar_hash,
            "cache": str(reference_cache.resolve()),
            "cache_sha256": cache_hash,
            "analytic_mms_eta_period": reference.eta_period,
            "continuous_evaluator_period": reference.metric_evaluator.period,
            "provenance": reference.provenance,
        },
        "boundary_contract": BOUNDARY_CONTRACT,
        "candidate_interpretation": {
            "P": "unchanged production face states and action",
            "G": "one geometric-moment common face value on every eligible planar face",
            "O": "one midpoint-observation common face value on every eligible planar face",
            "scalar_upwind_clarification": "G/O install the same common value on both left and right characteristic states; the action uses the scalar-upwind assembly but does not preserve production upwind stabilization",
            "fixed": [
                "generator velocities", "center/compression terms", "physical measures",
                "owner restriction and stored data", "eta reconstruction",
                "physical boundary traces", "rho_star/sign normalization",
            ],
        },
        "support": {
            "candidate_planar_face_count": int(len(keys)),
            "candidate_face_key_sha256": _array_sha256(keys),
            "canonical_shared_row_per_coordinate_face": True,
            "collapsed_axis_radial_face_flux": 0.0,
            "physical_boundary_traces_preserved": True,
            "reconstruction": reconstruction,
            "midpoint_observation": observation_report,
        },
        "regional_masks": {
            region: {
                "owner_count": int(np.count_nonzero(mask)),
                "physical_volume": float(np.sum(np.asarray(artifact.owner_geometry.aggregate_chart_volume)[mask])),
            }
            for region, mask in masks.items()
        },
        "ordinary_adjacent_layer_owner_count": int(np.count_nonzero(ordinary_layer)),
        "fields": fields_payload,
        "n32_preflight": preflight,
        "n32_prior_bounded_equivalence": n32_equivalence,
        "scope": {
            "certification": "global diagnostic P/G/O bracket qualification; no production promotion",
            "production_changes": [],
            "new_resolutions": [],
            "excluded": ["curvature", "polarization", "diffusion", "evolved MMS", "blob driver"],
        },
        "resources": {
            "maximum_rss_gib": _max_rss_gib(),
            "candidate_face_count": int(len(keys)),
        },
        "timings_seconds": {
            "runtime_setup": runtime_seconds,
            "row_construction_and_values": row_seconds,
            "candidate_actions": time.perf_counter() - action_started,
            "total_before_output": time.perf_counter() - started,
        },
    }
    output = output_dir / f"N{resolution}.json"
    _write_json(output, payload)
    _progress(
        "case_complete", resolution=resolution, output=str(output),
        seconds=time.perf_counter() - started, max_rss_gib=_max_rss_gib(),
    )
    return payload


def _orders(resolutions: Sequence[int], errors: Sequence[float]) -> list[float | None]:
    result = []
    for n0, n1, left, right in zip(
        resolutions[:-1], resolutions[1:], errors[:-1], errors[1:]
    ):
        if left <= 0.0 or right <= 0.0:
            result.append(None)
        else:
            result.append(float(math.log(left / right) / math.log(n1 / n0)))
    return result


def _series(cases: Sequence[Mapping[str, Any]], field: str, operator: str, rule: str):
    errors = [
        float(case["fields"][field]["actions"][operator][rule]["global"]["absolute_l2"])
        for case in cases
    ]
    relative = [
        float(case["fields"][field]["actions"][operator][rule]["global"]["relative_l2"])
        for case in cases
    ]
    orders = _orders([int(case["resolution"]) for case in cases], errors)
    return {
        "absolute_l2": errors,
        "relative_l2": relative,
        "absolute_orders": orders,
        "accepted": bool(len(orders) == 2 and all(value is not None and value >= 1.8 for value in orders)),
        "required_minimum_order": 1.8,
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    resolutions = summary["resolutions"]
    lines = [
        "# Global actual-HSX P/G/O bracket qualification",
        "",
        "This diagnostic compares unchanged production P with degree-two geometric-moment G and midpoint-observation O common-face candidates. G/O use one value on both characteristic sides in the scalar-upwind assembly, so their accuracy does not certify production upwind stabilization.",
        "",
        "## Global comparison",
        "",
        "| Field | Operator | Rule | absolute L2 N32 / N48 / N64 | orders 32→48 / 48→64 | gate |",
        "|---|---|---|---:|---:|---|",
    ]
    for field in FIELDS:
        for operator in ("scalar_upwind", "centered_A", "centered_B", "centered_C"):
            for rule in RULES:
                item = summary["comparison"][field][operator][rule]
                errors = " / ".join(f"{value:.7g}" for value in item["absolute_l2"])
                orders = " / ".join("—" if value is None else f"{value:.4f}" for value in item["absolute_orders"])
                lines.append(
                    f"| {field} | {operator} | {rule} | {errors} | {orders} | {'pass' if item['accepted'] else 'fail'} |"
                )
    lines.extend([
        "",
        "## Decision",
        "",
        summary["recommendation"],
        "",
        "Regional budgets, A/B/C decomposition, structural closures, row diagnostics, identities, timings, and resource measurements are retained in `summary.json` and the atomic per-resolution case files.",
        "",
        "## Provenance",
        "",
        f"Resolutions: {resolutions}. Global acceptance remains absolute physical-volume-weighted L2 order at least 1.8 on both intervals for each operator/field. No production default, restart/state layout, geometry, boundary, curvature, polarization, diffusion, evolved-MMS, or blob-driver change was made.",
        "",
    ])
    return "\n".join(lines)


def _merge(args: argparse.Namespace) -> dict[str, Any]:
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    cases.sort(key=lambda case: int(case["resolution"]))
    resolutions = [int(case["resolution"]) for case in cases]
    if resolutions != [32, 48, 64]:
        raise ValueError("global P04 merge requires N32/N48/N64")
    comparison = {}
    for field in FIELDS:
        comparison[field] = {}
        for operator in ("scalar_upwind", "centered_A", "centered_B"):
            comparison[field][operator] = {
                rule: _series(cases, field, operator, rule) for rule in RULES
            }
        comparison[field]["centered_C"] = {
            "P": _series(cases, field, "centered_C", "P_P"),
            "G": _series(cases, field, "centered_C", "G_G"),
            "O": _series(cases, field, "centered_C", "O_O"),
        }
    passed = [
        (field, operator, rule)
        for field in FIELDS
        for operator in comparison[field]
        for rule in RULES
        if comparison[field][operator][rule]["accepted"]
    ]
    failed = [
        (field, operator, rule)
        for field in FIELDS
        for operator in comparison[field]
        for rule in RULES
        if not comparison[field][operator][rule]["accepted"]
    ]
    candidate_passes = [item for item in passed if item[2] in ("G", "O")]
    if candidate_passes:
        recommendation = (
            "At least one common-face candidate satisfies the global accuracy gate for the listed operator/field lanes. Proceed only to a bounded integration/stability design for those exact lanes; preserve failures and do not promote a universal rule or production default from this diagnostic alone."
        )
    else:
        recommendation = (
            "Neither common-face candidate satisfies a global accuracy gate. The next bounded repair is a two-sided face reconstruction that preserves the accepted owner functional while restoring distinct left/right states for the scalar-upwind lane; do not change degree or geometry until that stabilization-consistency split is measured."
        )
    payload = {
        "schema": SUMMARY_SCHEMA,
        "resolutions": resolutions,
        "comparison": comparison,
        "accepted_lanes": [list(item) for item in passed],
        "failed_lanes": [list(item) for item in failed],
        "recommendation": recommendation,
        "cases": cases,
        "acceptance_contract": {
            "norm": "global physical-volume-weighted absolute operator L2",
            "minimum_order": 1.8,
            "required_intervals": ["32->48", "48->64"],
            "per_operator_field": True,
            "regional_orders_are_diagnostic": True,
        },
        "production_changes": [],
    }
    _write_json(args.output, payload)
    report = args.output.with_name("report.md")
    _write_text(report, _render_report(payload))
    print(args.output, flush=True)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    case = sub.add_parser("case")
    case.add_argument("--geometry", type=Path, required=True)
    case.add_argument("--reference-sidecar", type=Path, required=True)
    case.add_argument("--baseline", type=Path, required=True)
    case.add_argument("--prior-n32", type=Path, required=True)
    case.add_argument("--resolution", type=int, choices=(32, 48, 64), required=True)
    case.add_argument("--time", type=float, default=1.0e-6)
    case.add_argument("--output", type=Path, required=True)
    case.set_defaults(handler=_case)
    merge = sub.add_parser("merge")
    merge.add_argument("inputs", nargs=3, type=Path)
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(handler=_merge)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
