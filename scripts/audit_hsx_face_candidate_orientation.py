#!/usr/bin/env python3
"""Bounded N32 orientation audit for the HSX advected-face-state candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping
from unittest.mock import patch

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import simulate_hsx_mms as mms  # noqa: E402
from hsx_mms_continuum_reference import (  # noqa: E402
    build_continuum_reference_from_artifact,
    build_continuum_reference_from_sidecar,
)
from drbx.native.fci_operators import (  # noqa: E402
    _compatible_flux_generator,
)

import audit_hsx_poisson_common_face_oracle as oracle  # noqa: E402
import audit_hsx_poisson_vorticity as base  # noqa: E402


SCHEMA = "hsx-face-candidate-orientation-v1"
AXIS_NAMES = ("radial", "angular", "eta")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(oracle._json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _progress(stage: str, **details: Any) -> None:
    print(json.dumps({"event": "progress", "stage": stage, **details}), flush=True)


def _sanitize_collapsed_axis_face_records(payload: dict[str, Any]) -> int:
    """Remove nonphysical representative-state comparisons at collapsed u=0."""

    changed = 0
    for field in payload.get("fields", {}).values():
        for record in field.get("face_records", []):
            key = record.get("key", ())
            collapsed = bool(len(key) == 4 and key[0] == 0 and key[1] == 0)
            record["collapsed_axis_face"] = collapsed
            if not collapsed:
                record.setdefault("reference_midpoint_face_state_available", True)
                record.setdefault("collapsed_axis_flux", None)
                continue
            record["reference_midpoint_face_state"] = None
            record["reference_midpoint_face_state_available"] = False
            record["candidate_minus_reference"] = None
            record["collapsed_axis_flux"] = 0.0
            changed += 1
    payload.setdefault("scope", {})["collapsed_axis_record_policy"] = (
        "u=0 representative state/metric unavailable and excluded from face-state "
        "accuracy comparisons; compatible collapsed-face integrated flux is exactly zero; "
        "axis finite-volume owner residuals remain included"
    )
    return changed


def _eta_scalar(points: np.ndarray, eta_period: float) -> np.ndarray:
    q = np.asarray(points, dtype=np.float64)
    u, eta = q[:, 0], q[:, 2]
    envelope = np.maximum(1.0 - u**2, 0.0) ** 4
    return 0.08 * envelope * np.sin(2.0 * np.pi * eta / eta_period + 0.31)


def _project_scalar(projector: Any, function: Callable[[np.ndarray, float], np.ndarray]) -> np.ndarray:
    flat = np.empty(int(np.prod(projector.shape)), dtype=np.float64)
    for first, last, points, _prepared, weighted in projector.chunks:
        values = function(points, projector.reference.eta_period).reshape(
            (last - first, projector.nq)
        )
        flat[first:last] = np.sum(weighted * values, axis=1) / np.maximum(
            np.sum(weighted, axis=1), 1.0e-30
        )
    return flat.reshape(projector.shape)


def _load_midpoint_reference_cache(
    path: Path,
    *,
    expected_resolution: int,
    expected_time: float,
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as cached:
        provenance = json.loads(str(np.asarray(cached["provenance_json"]).item()))
        if provenance.get("schema") != "drbx.hsx-poisson-midpoint-reference-cache-v1":
            raise ValueError("unsupported midpoint reference cache schema")
        if int(provenance["resolution"]) != int(expected_resolution):
            raise ValueError("midpoint reference cache resolution mismatch")
        if not np.isclose(
            float(provenance["time"]), float(expected_time), rtol=0.0, atol=1.0e-15
        ):
            raise ValueError("midpoint reference cache time mismatch")
        state = base.blob.FciDrbEBState(
            **{
                name: np.asarray(cached[f"actual_{name}"], dtype=np.float64)
                for name in mms.FIELDS
            }
        )
        regular = np.asarray(cached["smooth_regular_scalar"], dtype=np.float64)
        eta = np.asarray(cached["smooth_eta_varying_scalar"], dtype=np.float64)
    return state, regular, eta, provenance


def _midpoint_observation_aggregate(
    owner_geometry: Any, geometry: Any
) -> tuple[Any, dict[str, Any]]:
    """Moments of the exact midpoint/raw-volume owner observation functional."""

    geometric = oracle._aggregate_geometry(owner_geometry)
    shape = tuple(int(value) for value in geometry.shape)
    u = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    theta = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    eta = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    logical = np.stack(np.meshgrid(u, theta, eta, indexing="ij"), axis=-1)
    x = logical[..., 0] * np.cos(logical[..., 1])
    y = logical[..., 0] * np.sin(logical[..., 1])
    raw_volume = np.asarray(owner_geometry.raw_volume, dtype=np.float64).ravel()
    aggregate_ids = np.asarray(
        owner_geometry.topology.aggregate_id, dtype=np.int64
    ).ravel()
    size = int(np.prod(shape))

    def aggregate_sum(value: np.ndarray) -> np.ndarray:
        result = np.zeros(size, dtype=np.float64)
        np.add.at(result, aggregate_ids, raw_volume * np.asarray(value).ravel())
        return result[geometric.owner_flat_ids]

    observed_volume = aggregate_sum(np.ones(shape, dtype=np.float64))
    centroid = np.stack(
        (aggregate_sum(x) / observed_volume, aggregate_sum(y) / observed_volume),
        axis=-1,
    )
    dx = x.ravel()[None, :]  # only used to make the convention explicit below
    del dx
    centroid_raw = centroid[np.searchsorted(geometric.owner_flat_ids, aggregate_ids)]
    delta_x = x.ravel() - centroid_raw[:, 0]
    delta_y = y.ravel() - centroid_raw[:, 1]
    second = np.empty((geometric.n_owner, 2, 2), dtype=np.float64)
    second[:, 0, 0] = aggregate_sum(delta_x**2) / observed_volume
    second[:, 0, 1] = aggregate_sum(delta_x * delta_y) / observed_volume
    second[:, 1, 0] = second[:, 0, 1]
    second[:, 1, 1] = aggregate_sum(delta_y**2) / observed_volume
    aggregate = oracle.AggregateGeometry(
        owner_flat_ids=geometric.owner_flat_ids,
        owner_plane=geometric.owner_plane,
        volumes=geometric.volumes,
        centroid_xy=centroid,
        central_second_xy=second,
    )
    report = {
        "raw_sample_location": "logical cell midpoint mapped to x=u*cos(theta), y=u*sin(theta)",
        "raw_projection_weight": "owner_geometry.raw_volume",
        "owner_normalization": "owner_geometry.aggregate_chart_volume",
        "owner_count": geometric.n_owner,
        "maximum_observed_volume_relative_mismatch": float(
            np.max(
                np.abs(observed_volume - geometric.volumes)
                / np.maximum(np.abs(geometric.volumes), np.finfo(float).tiny)
            )
        ),
        "geometric_vs_observation_centroid_max_abs": float(
            np.max(np.abs(geometric.centroid_xy - centroid))
        ),
        "geometric_vs_observation_second_moment_max_abs": float(
            np.max(np.abs(geometric.central_second_xy - second))
        ),
        "interpretation": (
            "diagnostic representation of midpoint-projected MMS inputs; it does not "
            "redefine evolved finite-volume owner unknowns as point samples"
        ),
    }
    return aggregate, report


def _candidate_from_aggregate(
    aggregate: Any,
    model: Any,
    face_keys: list[tuple[int, int, int, int]],
    velocities: Mapping[tuple[int, int, int, int], float],
) -> dict[str, Any]:
    points_xy = []
    eta_indices = []
    owners = []
    for key in face_keys:
        points, _ = oracle._face_points_weights(model.geometry, key, 1)
        point = points[0]
        owner = oracle._upwind_owner(model, key, velocities[key])
        owners.append(owner)
        points_xy.append(
            (float(point[0] * np.cos(point[1])), float(point[0] * np.sin(point[1])))
        )
        eta_indices.append(owner[2])
    evaluation = oracle.build_aggregate_evaluation(
        aggregate,
        np.asarray(points_xy, dtype=np.float64),
        np.asarray(eta_indices, dtype=np.int64),
        degree=2,
        donor_count=12,
        max_donor_count=96,
        condition_limit=1.0e8,
        weight_power=4.0,
        row_l1_limit=8.0,
    )
    return {
        "aggregate": aggregate,
        "production_evaluation": evaluation,
        "production_owners": dict(zip(face_keys, owners)),
        "production_points_xy": np.asarray(points_xy, dtype=np.float64),
    }


def _support_comparison(left: Any, right: Any) -> dict[str, Any]:
    left_matrix = left.matrix.tocsr()
    right_matrix = right.matrix.tocsr()
    jaccard = []
    symmetric_difference = []
    identical = 0
    for row in range(left_matrix.shape[0]):
        a = set(left_matrix.indices[left_matrix.indptr[row] : left_matrix.indptr[row + 1]])
        b = set(right_matrix.indices[right_matrix.indptr[row] : right_matrix.indptr[row + 1]])
        identical += int(a == b)
        jaccard.append(len(a & b) / max(len(a | b), 1))
        symmetric_difference.append(len(a ^ b))
    return {
        "row_count": int(left_matrix.shape[0]),
        "identical_support_row_count": int(identical),
        "identical_support_fraction": float(identical / max(left_matrix.shape[0], 1)),
        "minimum_support_jaccard": float(np.min(jaccard)),
        "maximum_symmetric_difference_count": int(np.max(symmetric_difference)),
        "unavoidable_difference": (
            "nearest-donor selection uses the observation centroids; rows whose "
            "centroids reorder may use different donors while retaining the same "
            "degree, selection policy, limits, and complete face support"
        ),
    }


def _chart_cell_average(
    geometry: Any,
    cell: tuple[int, int, int],
    order: int,
    function: Callable[[np.ndarray], np.ndarray],
    jacobian_function: Callable[[np.ndarray], np.ndarray],
) -> tuple[float, float, np.ndarray]:
    points, logical_weight = oracle._cell_points_weights(geometry, cell, order)
    chart_weight = logical_weight * np.abs(
        np.asarray(jacobian_function(points), dtype=np.float64)
    )
    values = np.asarray(function(points), dtype=np.float64)
    volume = float(np.sum(chart_weight))
    return (
        float(np.sum(chart_weight * values) / max(volume, 1.0e-30)),
        volume,
        values,
    )


def _integrated_owner_values(
    *,
    model: Any,
    owner_values: np.ndarray,
    owner_flat_ids: np.ndarray,
    donor_compact_ids: np.ndarray,
    value_function: Callable[[np.ndarray], np.ndarray],
    jacobian_function: Callable[[np.ndarray], np.ndarray],
    order: int = 4,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cells = model.control_volume_geometry.cells
    owner_i = np.asarray(cells.owner_i, dtype=np.int64)
    owner_j = np.asarray(cells.owner_j, dtype=np.int64)
    owner_k = np.asarray(cells.owner_k, dtype=np.int64)
    raw_volume = np.asarray(cells.raw_volume, dtype=np.float64)
    compact = np.asarray(owner_values, dtype=np.float64).reshape(-1)[owner_flat_ids].copy()
    records = []
    shape = owner_values.shape
    for compact_id in np.unique(np.asarray(donor_compact_ids, dtype=np.int64)):
        owner = tuple(int(v) for v in np.unravel_index(owner_flat_ids[compact_id], shape))
        members = np.argwhere(
            (owner_i == owner[0]) & (owner_j == owner[1]) & (owner_k == owner[2])
        )
        weighted_sum = 0.0
        measured_volume = 0.0
        sample_values = []
        stored_volume = 0.0
        for member_array in members:
            member = tuple(int(v) for v in member_array)
            average, volume, values = _chart_cell_average(
                model.geometry,
                member,
                order,
                value_function,
                jacobian_function,
            )
            weighted_sum += volume * average
            measured_volume += volume
            stored_volume += float(raw_volume[member])
            sample_values.extend(values.tolist())
        integrated = weighted_sum / max(measured_volume, 1.0e-30)
        stored = float(compact[compact_id])
        compact[compact_id] = integrated
        records.append(
            {
                "owner": owner,
                "compact_owner_id": int(compact_id),
                "member_count": int(len(members)),
                "stored_midpoint_projected_owner_value": stored,
                "integrated_q4_owner_value": integrated,
                "integrated_minus_stored": integrated - stored,
                "measured_chart_volume": measured_volume,
                "stored_chart_volume": stored_volume,
                "chart_volume_relative_mismatch": abs(measured_volume - stored_volume)
                / max(abs(stored_volume), 1.0e-30),
                "q4_sample_min": float(np.min(sample_values)),
                "q4_sample_max": float(np.max(sample_values)),
                "q4_sample_std": float(np.std(sample_values)),
            }
        )
    return compact, records


def _side_value(sides: Any, key: tuple[int, int, int, int], velocity: float) -> float:
    side = 0 if velocity >= 0.0 else 1
    return oracle._face_array_value(sides[side], key)


def _stage_statistics(rows: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    result = oracle._residual_statistics(rows, stage, "reference_midpoint")
    for region, stats in result.items():
        subset = [row for row in rows if row["label"] == region]
        weights = np.asarray([row["aggregate_volume"] for row in subset])
        delta = np.asarray([row[stage] - row["baseline"] for row in subset])
        stats["signed_volume_weighted_change"] = float(
            np.sum(weights * delta) / np.sum(weights)
        )
    return result


def _field_case(
    *,
    name: str,
    model: Any,
    owner_state: Any,
    raw_state: Any,
    owner_values: np.ndarray,
    raw_values: np.ndarray,
    exact_function: Callable[[np.ndarray], np.ndarray],
    face_keys: list[tuple[int, int, int, int]],
    selected: Mapping[str, list[tuple[int, int, int]]],
    candidate_payload: Mapping[str, Any],
    production_velocity: Mapping[tuple[int, int, int, int], float],
    exact_velocity: Mapping[tuple[int, int, int, int], float],
    generator_integrals: Mapping[str, Mapping[tuple[int, int, int, int], float]],
    integrate_planar_donors: bool,
    jacobian_function: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    operands = base._prepare_operands(model, owner_state, raw_state)
    sides = tuple(
        tuple(np.asarray(getattr(side, "xyz"[axis])) for axis in range(3))
        for side in operands.omega_direct_states
    )
    aggregate = candidate_payload["aggregate"]
    evaluation = candidate_payload["production_evaluation"]
    candidate_stored = dict(
        zip(face_keys, oracle._aggregate_values(aggregate, evaluation, owner_values))
    )
    integrated_records = []
    candidate_integrated = None
    if integrate_planar_donors:
        planar_rows = np.asarray([key[0] in (0, 1) for key in face_keys], dtype=bool)
        matrix = evaluation.matrix.tocsr()
        donor_ids = np.unique(
            np.concatenate(
                [
                    matrix.indices[matrix.indptr[row] : matrix.indptr[row + 1]]
                    for row in np.flatnonzero(planar_rows)
                ]
            )
        )
        compact, integrated_records = _integrated_owner_values(
            model=model,
            owner_values=owner_values,
            owner_flat_ids=aggregate.owner_flat_ids,
            donor_compact_ids=donor_ids,
            value_function=exact_function,
            jacobian_function=jacobian_function,
        )
        candidate_integrated = dict(
            zip(face_keys, np.asarray(evaluation.evaluate(compact), dtype=np.float64))
        )

    products = {
        "baseline": {},
        "candidate_planar_only": {},
        "candidate_eta_only": {},
        "candidate_all_faces": {},
    }
    if candidate_integrated is not None:
        products["candidate_planar_integrated_owner"] = {}
    reference_product = {}
    face_records = []
    for key in face_keys:
        points, weight = oracle._face_points_weights(model.geometry, key, 1)
        area = float(weight[0])
        prod = _side_value(sides, key, production_velocity[key])
        cand = float(candidate_stored[key])
        collapsed_axis = oracle._is_collapsed_axis_face(model, key)
        # A collapsed lower-u face has nonzero logical theta/eta quadrature
        # weight, but the topology sets its physical compatible flux exactly
        # to zero.  No ordinary-polar state/metric exists at u=0, and no such
        # representative value is needed for the completed residual.
        exact = None if collapsed_axis else float(exact_function(points)[0])
        axis = int(key[0])
        products["baseline"][key] = production_velocity[key] * prod * area
        products["candidate_planar_only"][key] = production_velocity[key] * (
            cand if axis in (0, 1) else prod
        ) * area
        products["candidate_eta_only"][key] = production_velocity[key] * (
            cand if axis == 2 else prod
        ) * area
        products["candidate_all_faces"][key] = production_velocity[key] * cand * area
        if candidate_integrated is not None:
            products["candidate_planar_integrated_owner"][key] = production_velocity[key] * (
                float(candidate_integrated[key]) if axis in (0, 1) else prod
            ) * area
        reference_product[key] = (
            0.0 if collapsed_axis else exact_velocity[key] * exact * area
        )
        face_records.append(
            {
                "key": key,
                "axis": AXIS_NAMES[axis],
                "labels": [],
                "production_face_state": prod,
                "candidate_stored_owner_face_state": cand,
                "candidate_integrated_owner_face_state": (
                    None if candidate_integrated is None else float(candidate_integrated[key])
                ),
                "reference_midpoint_face_state": exact,
                "reference_midpoint_face_state_available": not collapsed_axis,
                "collapsed_axis_face": collapsed_axis,
                "collapsed_axis_flux": 0.0 if collapsed_axis else None,
                "candidate_minus_production": cand - prod,
                "candidate_minus_reference": (
                    None if collapsed_axis else cand - exact
                ),
            }
        )

    cells = model.control_volume_geometry.cells
    raw_volume = np.asarray(cells.raw_volume, dtype=np.float64)
    aggregate_volume = np.asarray(cells.aggregate_volume, dtype=np.float64)
    owner_i = np.asarray(cells.owner_i, dtype=np.int64)
    owner_j = np.asarray(cells.owner_j, dtype=np.int64)
    owner_k = np.asarray(cells.owner_k, dtype=np.int64)
    production_center = np.asarray(operands.omega_stencil.x.center, dtype=np.float64)
    reference_center = np.asarray(raw_values, dtype=np.float64)
    rho_star = float(model.parameters.rho_star)
    rows = []
    for label, owners in selected.items():
        for owner in owners:
            members = [
                tuple(int(v) for v in row)
                for row in np.argwhere(
                    (owner_i == owner[0])
                    & (owner_j == owner[1])
                    & (owner_k == owner[2])
                )
            ]
            stage_values = {}
            for stage, weighted in products.items():
                action = oracle._assemble_owner_residual(
                    owner,
                    members,
                    raw_volume,
                    aggregate_volume,
                    production_center,
                    generator_integrals["production"],
                    weighted,
                )
                stage_values[stage] = -action / rho_star
            reference_action = oracle._assemble_owner_residual(
                owner,
                members,
                raw_volume,
                aggregate_volume,
                reference_center,
                generator_integrals["reference"],
                reference_product,
            )
            axis_changes = {
                "radial": stage_values["candidate_planar_only"] - stage_values["baseline"],
                "angular": 0.0,
                "eta": stage_values["candidate_eta_only"] - stage_values["baseline"],
            }
            # Split radial and angular explicitly with one-axis hybrid maps.
            for axis, axis_name in ((0, "radial"), (1, "angular")):
                weighted = {
                    key: (
                        production_velocity[key]
                        * (candidate_stored[key] if key[0] == axis else _side_value(sides, key, production_velocity[key]))
                        * float(oracle._face_points_weights(model.geometry, key, 1)[1][0])
                    )
                    for key in face_keys
                }
                action = oracle._assemble_owner_residual(
                    owner, members, raw_volume, aggregate_volume, production_center,
                    generator_integrals["production"], weighted,
                )
                axis_changes[axis_name] = -action / rho_star - stage_values["baseline"]
            rows.append(
                {
                    "label": label,
                    "owner": owner,
                    "aggregate_volume": float(aggregate_volume[owner]),
                    **stage_values,
                    "reference_midpoint": -reference_action / rho_star,
                    "signed_candidate_change_by_axis": axis_changes,
                    "axis_change_sum": sum(axis_changes.values()),
                    "all_face_change": stage_values["candidate_all_faces"] - stage_values["baseline"],
                    "axis_change_closure": sum(axis_changes.values())
                    - (stage_values["candidate_all_faces"] - stage_values["baseline"]),
                }
            )
    statistics = {stage: _stage_statistics(rows, stage) for stage in products}
    axis_summary = {}
    for axis_name in AXIS_NAMES:
        axis_summary[axis_name] = {}
        for label in selected:
            subset = [row for row in rows if row["label"] == label]
            weights = np.asarray([row["aggregate_volume"] for row in subset])
            values = np.asarray(
                [row["signed_candidate_change_by_axis"][axis_name] for row in subset]
            )
            axis_summary[axis_name][label] = {
                "signed_volume_weighted_mean": float(np.sum(weights * values) / np.sum(weights)),
                "physical_volume_weighted_rms": float(
                    np.sqrt(np.sum(weights * values**2) / np.sum(weights))
                ),
            }
    return {
        "name": name,
        "owner_records": rows,
        "stage_statistics": statistics,
        "signed_change_by_orientation": axis_summary,
        "maximum_axis_change_closure": float(
            max(abs(row["axis_change_closure"]) for row in rows)
        ),
        "integrated_owner_records": integrated_records,
        "face_records": face_records,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    _progress("matched_functional_start")
    artifact_path, artifact = base._load_artifact(args.geometry, 32)
    _, reference_artifact = base._load_artifact(args.geometry, 64)
    reference_builder = (
        build_continuum_reference_from_sidecar
        if args.reference_sidecar is not None
        else build_continuum_reference_from_artifact
    )
    reference = reference_builder(
        args.reference_sidecar if args.reference_sidecar is not None else reference_artifact,
        tau=mms.PHYSICAL_PARAMETERS["tau"],
        mi_over_me=mms.PHYSICAL_PARAMETERS["mi_over_me"],
        rho_star=mms.PHYSICAL_PARAMETERS["rho_star"],
        Ve_nu=mms.PHYSICAL_PARAMETERS["Ve_nu"],
        perp_diffusion=mms.PHYSICAL_PARAMETERS["density_D_perp"],
        enable_generalized_potential=True,
    )
    runtime_args = SimpleNamespace(
        shard_counts=(1, 1, 1),
        curvature_edge_one_form=False,
        reference=reference,
        metric_context=SimpleNamespace(
            metric_evaluator=reference.metric_evaluator,
            bfield=reference.bfield_evaluator,
            nfp=int(reference_artifact.nfp),
        ),
    )
    with patch.dict(os.environ, base._FROZEN_RUNTIME_ENV, clear=False):
        runtime = mms._runtime(artifact.global_geometry, artifact.owner_geometry, runtime_args)
    model = runtime.model
    if model is None:
        raise RuntimeError("orientation audit requires a single-device model")
    cache_provenance = None
    if args.reference_cache is None:
        projector_started = time.perf_counter()
        _progress("projector_start")
        projector = mms._QuadratureProjector(
            reference, artifact.global_geometry, artifact.owner_geometry
        )
        raw_state, _ = base._poisson_reference_fields(
            projector, float(args.time), rho_star=float(model.parameters.rho_star)
        )
        regular_raw = oracle._project_regular_scalar(projector)
        eta_raw = _project_scalar(projector, _eta_scalar)
        projector_seconds = time.perf_counter() - projector_started
        _progress("projector_complete", seconds=projector_seconds)
    else:
        cache_started = time.perf_counter()
        raw_state, regular_raw, eta_raw, cache_provenance = (
            _load_midpoint_reference_cache(
                args.reference_cache,
                expected_resolution=32,
                expected_time=float(args.time),
            )
        )
        cached_reference = cache_provenance.get("reference_source", {})
        current_reference = getattr(reference, "provenance", {})
        if cached_reference.get("sidecar_sha256") != current_reference.get(
            "sidecar_sha256"
        ):
            raise ValueError("midpoint reference cache sidecar identity mismatch")
        manifest = artifact_path / "manifest.json"
        if cache_provenance.get("geometry_manifest_sha256") != _sha256(manifest):
            raise ValueError("midpoint reference cache geometry identity mismatch")
        projector_seconds = 0.0
        _progress(
            "reference_cache_loaded", seconds=time.perf_counter() - cache_started
        )
    owner_state = mms._owner_project(raw_state, artifact.owner_geometry)
    phi_operands = base._prepare_operands(model, owner_state, raw_state)
    generator = _compatible_flux_generator(
        phi_operands.phi_stencil,
        model.geometry,
        domain=model.domain,
        axis_regular_axes=model.axis_regular_axes,
        b_floor=1.0e-30,
    )
    generator_arrays = tuple(np.asarray(v) for v in (generator.x, generator.y, generator.z))
    saved = json.loads(Path(args.saved_n32).read_text(encoding="utf-8"))
    selected = {
        label: [tuple(int(v) for v in owner) for owner in owners]
        for label, owners in saved["scope"]["selection"].items()
    }
    raw_labels, face_labels = oracle._support(model.control_volume_geometry.cells, selected)
    face_keys = sorted(face_labels)
    production_velocity = {key: oracle._face_array_value(generator_arrays, key) for key in face_keys}
    exact_velocity = {}
    for key in face_keys:
        if oracle._is_collapsed_axis_face(model, key):
            exact_velocity[key] = 0.0
        else:
            points, _ = oracle._face_points_weights(model.geometry, key, 1)
            gradient, _omega, one_form = oracle._exact_samples(reference, points, float(args.time))
            exact_velocity[key] = float(oracle._normal_velocity(key[0], one_form, gradient)[0])
    generator_integrals = {"production": {}, "reference": {}}
    for key in face_keys:
        _points, weight = oracle._face_points_weights(model.geometry, key, 1)
        generator_integrals["production"][key] = production_velocity[key] * float(weight[0])
        generator_integrals["reference"][key] = exact_velocity[key] * float(weight[0])
    candidate_tuple = oracle._aggregate_candidate_evaluation(
        artifact.owner_geometry, model, face_keys, production_velocity
    )
    candidate_payload = {
        "aggregate": candidate_tuple[0],
        "production_evaluation": candidate_tuple[1],
        "production_owners": dict(zip(face_keys, candidate_tuple[2])),
        "production_points_xy": candidate_tuple[3],
    }
    matched_payload = None
    observation_report = None
    support_comparison = None
    if bool(args.compare_midpoint_functional):
        observation_aggregate, observation_report = _midpoint_observation_aggregate(
            artifact.owner_geometry, artifact.global_geometry
        )
        matched_payload = _candidate_from_aggregate(
            observation_aggregate, model, face_keys, production_velocity
        )
        support_comparison = _support_comparison(
            candidate_payload["production_evaluation"],
            matched_payload["production_evaluation"],
        )

    cases = []
    field_specs = (
        (
            "actual_vorticity",
            raw_state,
            owner_state,
            np.asarray(model._owner_field(owner_state.vorticity)),
            np.asarray(raw_state.vorticity),
            lambda points: oracle._exact_samples(reference, points, float(args.time))[1],
            True,
        ),
        (
            "smooth_regular_scalar",
            raw_state.replace(vorticity=regular_raw),
            mms._owner_project(raw_state.replace(vorticity=regular_raw), artifact.owner_geometry),
            None,
            regular_raw,
            lambda points: oracle._regular_scalar(points, reference.eta_period),
            True,
        ),
        (
            "smooth_eta_varying_scalar",
            raw_state.replace(vorticity=eta_raw),
            mms._owner_project(raw_state.replace(vorticity=eta_raw), artifact.owner_geometry),
            None,
            eta_raw,
            lambda points: _eta_scalar(points, reference.eta_period),
            False,
        ),
    )
    reused_geometric = None
    if args.reuse_geometric_artifact is not None:
        reused_geometric = json.loads(
            args.reuse_geometric_artifact.read_text(encoding="utf-8")
        )
        if reused_geometric.get("schema") != SCHEMA:
            raise ValueError("reused geometric artifact schema mismatch")
        if reused_geometric.get("scope", {}).get("selection") != oracle._json_value(selected):
            raise ValueError("reused geometric artifact selection mismatch")
        cases = list(reused_geometric["fields"].values())
    else:
        for name, field_raw_state, field_owner_state, values, raw_values, exact, integrate in field_specs:
            if values is None:
                values = np.asarray(model._owner_field(field_owner_state.vorticity))
            cases.append(
                _field_case(
                    name=name,
                    model=model,
                    owner_state=field_owner_state,
                    raw_state=field_raw_state,
                    owner_values=values,
                    raw_values=raw_values,
                    exact_function=exact,
                    face_keys=face_keys,
                    selected=selected,
                    candidate_payload=candidate_payload,
                    production_velocity=production_velocity,
                    exact_velocity=exact_velocity,
                    generator_integrals=generator_integrals,
                    integrate_planar_donors=integrate,
                    jacobian_function=lambda points: reference._metric(points)["J"],
                )
            )
    matched_cases = []
    if matched_payload is not None:
        matched_started = time.perf_counter()
        for spec in field_specs:
            matched_cases.append(
                _field_case(
                    name=spec[0],
                    model=model,
                    owner_state=spec[2],
                    raw_state=spec[1],
                    owner_values=(
                        np.asarray(model._owner_field(spec[2].vorticity))
                        if spec[3] is None else spec[3]
                    ),
                    raw_values=spec[4],
                    exact_function=spec[5],
                    face_keys=face_keys,
                    selected=selected,
                    candidate_payload=matched_payload,
                    production_velocity=production_velocity,
                    exact_velocity=exact_velocity,
                    generator_integrals=generator_integrals,
                    integrate_planar_donors=False,
                    jacobian_function=lambda points: reference.prepare(points).J,
                )
            )
        _progress(
            "matched_fields_complete", seconds=time.perf_counter() - matched_started
        )
    payload = {
        "schema": SCHEMA,
        "resolution": 32,
        "reference_resolution": 64,
        "time": float(args.time),
        "geometry_artifact": str(artifact_path),
        "saved_selection_source": str(args.saved_n32),
        "reference_source": getattr(reference, "provenance", {
            "artifact": str((args.geometry / "64x64x64").resolve())
        }),
        "midpoint_reference_cache": (
            None if args.reference_cache is None else {
                "path": str(args.reference_cache.resolve()),
                "provenance": cache_provenance,
            }
        ),
        "scope": {
            "selection": selected,
            "selected_owner_count": sum(len(v) for v in selected.values()),
            "selected_coordinate_face_count": len(face_keys),
            "boundary_contract": saved["scope"]["boundary_contract"],
            "candidate": (
                "degree-two planar aggregate-moment face evaluator on radial/angular faces; "
                "production eta reconstruction retained in orientation-correct stage"
            ),
            "production_changes": [],
            "certification": "bounded orientation and owner-functional diagnostic",
        },
        "candidate_conditioning": oracle._evaluation_diagnostics(candidate_tuple[1]),
        "fields": {case["name"]: case for case in cases},
        "timings_seconds": {
            "projector_reference_preparation": projector_seconds,
            "total_before_output": time.perf_counter() - started,
        },
    }
    if matched_payload is not None:
        payload["matched_functional_comparison"] = {
            "observation_functional": observation_report,
            "support_comparison": support_comparison,
            "geometric_moment_conditioning": oracle._evaluation_diagnostics(
                candidate_payload["production_evaluation"]
            ),
            "midpoint_observation_conditioning": oracle._evaluation_diagnostics(
                matched_payload["production_evaluation"]
            ),
            "geometric_moment_polynomial_reproduction": oracle._polynomial_reproduction(
                candidate_payload["aggregate"],
                candidate_payload["production_evaluation"],
                candidate_payload["production_points_xy"],
            ),
            "midpoint_observation_polynomial_reproduction": oracle._polynomial_reproduction(
                matched_payload["aggregate"],
                matched_payload["production_evaluation"],
                matched_payload["production_points_xy"],
            ),
            "geometric_moment_fields": payload["fields"],
            "midpoint_observation_fields": {
                case["name"]: case for case in matched_cases
            },
            "fixed_contract": {
                "stored_owner_data": True,
                "degree": 2,
                "donor_selection_policy": "nearest same-eta owners; 12->96 expansion schedule",
                "production_velocities": True,
                "center_compression_term": True,
                "conservative_assembly": True,
                "production_eta_reconstruction": True,
            },
            "integrated_average_control": (
                "retained only in geometric_moment_fields as a separately labelled control"
            ),
            "geometric_moment_provenance": (
                "computed in this artifact"
                if args.reuse_geometric_artifact is None
                else str(args.reuse_geometric_artifact.resolve())
            ),
        }
    _sanitize_collapsed_axis_face_records(payload)
    _write(args.output, payload)
    _progress(
        "matched_functional_complete",
        output=str(args.output),
        total_seconds=time.perf_counter() - started,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--saved-n32", type=Path, required=True)
    parser.add_argument("--reference-sidecar", type=Path)
    parser.add_argument("--compare-midpoint-functional", action="store_true")
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--reuse-geometric-artifact", type=Path)
    parser.add_argument("--time", type=float, default=1.0e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"output": str(args.output), "fields": list(payload["fields"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
