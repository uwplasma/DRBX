#!/usr/bin/env python3
"""Bounded N48/N64 P03 transition-functional and centered-bracket audit.

This script replays the unchanged production actions from qualified midpoint
reference caches, constructs a disjoint owner-region budget, freezes an
eight-owner selection before evaluating candidates, and compares the existing
geometric (G) and midpoint-observation (O) degree-two face rules.  Only selected
radial/angular advected face states change; production velocities, centers,
eta reconstruction, measures, restriction, boundary traces, and stored owner
data remain fixed.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence
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
    build_continuum_reference_from_sidecar,
)
from drbx.native.fci_boundaries import CoordinateFaceValues3D  # noqa: E402
from drbx.native.fci_operators import _compatible_flux_generator  # noqa: E402

import audit_hsx_face_candidate_orientation as orientation  # noqa: E402
import audit_hsx_poisson_common_face_oracle as oracle  # noqa: E402
import audit_hsx_poisson_continuous_baseline as continuous  # noqa: E402
import audit_hsx_poisson_vorticity as base  # noqa: E402


SCHEMA = "drbx.hsx-p03-fine-interval-case-v1"
SUMMARY_SCHEMA = "drbx.hsx-p03-fine-interval-summary-v1"
FIELDS = continuous.FIELDS
RULES = ("P", "G", "O")
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
    print(json.dumps({"event": "progress", "stage": stage, **details}, sort_keys=True), flush=True)


def _load_cache(path: Path, resolution: int, time_value: float):
    state, regular, eta, provenance = orientation._load_midpoint_reference_cache(
        path,
        expected_resolution=resolution,
        expected_time=time_value,
    )
    with np.load(path, allow_pickle=False) as cached:
        exact = {
            name: np.asarray(cached[f"exact_{name}"], dtype=np.float64)
            for name in FIELDS
        }
    return state, regular, eta, exact, provenance


def _owner_tuple_arrays(cells: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        np.asarray(getattr(cells, name), dtype=np.int64)
        for name in ("owner_i", "owner_j", "owner_k")
    )


def _mark_owner(mask: np.ndarray, owner: tuple[int, int, int]) -> None:
    if all(0 <= owner[axis] < mask.shape[axis] for axis in range(3)):
        mask[owner] = True


def _owner_from_raw(owner_arrays: tuple[np.ndarray, ...], raw: tuple[int, int, int]):
    return tuple(int(array[raw]) for array in owner_arrays)


def _owner_adjacency_and_interfaces(
    cells: Any,
    groups: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, set[tuple[tuple[int, int, int], tuple[int, int, int]]]]:
    """Return true size-change owners and radial/angular owner adjacencies."""

    owner_arrays = _owner_tuple_arrays(cells)
    raw_shape = owner_arrays[0].shape
    interface = np.zeros_like(active, dtype=bool)
    adjacency: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()

    def add_pair(left_raw: tuple[int, int, int], right_raw: tuple[int, int, int]):
        left = _owner_from_raw(owner_arrays, left_raw)
        right = _owner_from_raw(owner_arrays, right_raw)
        if left == right or not active[left] or not active[right]:
            return
        edge = tuple(sorted((left, right)))
        adjacency.add(edge)
        if int(groups[left[0]]) != int(groups[right[0]]):
            _mark_owner(interface, left)
            _mark_owner(interface, right)

    nx, ny, nz = raw_shape
    for i in range(1, nx):
        for j in range(ny):
            for k in range(nz):
                add_pair((i - 1, j, k), (i, j, k))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                add_pair((i, (j - 1) % ny, k), (i, j, k))
    return interface, adjacency


def _refined_masks(host: Any, cells: Any, standard: Mapping[str, np.ndarray]):
    active = np.asarray(host.topology.is_active_owner, dtype=bool)
    groups = np.asarray(host.angular_group_size, dtype=np.int64)
    true_interface_raw, adjacency = _owner_adjacency_and_interfaces(
        cells, groups, active
    )
    axis = np.zeros_like(active)
    axis[0] = active[0]
    boundary_raw = (
        np.asarray(standard["double_hit"], dtype=bool)
        | np.asarray(standard["short_leg_topology_transition"], dtype=bool)
        | np.asarray(standard["physical_wall"], dtype=bool)
    )
    boundary = active & boundary_raw & ~axis
    true_interface = active & true_interface_raw & ~axis & ~boundary
    owner_i = np.indices(active.shape, sparse=False)[0]
    agglomerated_raw = groups[owner_i] > 1
    agglomerated = active & agglomerated_raw & ~axis & ~boundary & ~true_interface
    ordinary = active & ~axis & ~boundary & ~true_interface & ~agglomerated
    masks = {
        "axis_ring": axis,
        "physical_boundary_footprint": boundary,
        "true_size_change_interface": true_interface,
        "agglomerated_interior": agglomerated,
        "ordinary": ordinary,
    }
    assigned = np.zeros_like(active)
    for mask in masks.values():
        if np.any(assigned & mask):
            raise RuntimeError("refined regional masks overlap")
        assigned |= mask
    if not np.array_equal(assigned, active):
        raise RuntimeError("refined regional masks do not cover active owners")

    ordinary_layer = np.zeros_like(active)
    for left, right in adjacency:
        if true_interface[left] and ordinary[right]:
            ordinary_layer[right] = True
        if true_interface[right] and ordinary[left]:
            ordinary_layer[left] = True
    return masks, ordinary_layer, adjacency


def _regional_statistics(
    actual: np.ndarray,
    exact: np.ndarray,
    host: Any,
    masks: Mapping[str, np.ndarray],
    ordinary_layer: np.ndarray,
) -> dict[str, Any]:
    active = np.asarray(host.topology.is_active_owner, dtype=bool)
    volume = np.asarray(host.aggregate_chart_volume, dtype=np.float64)
    global_stats = base._weighted_statistics(actual, exact, volume, active)
    regions = {}
    for name, mask in masks.items():
        item = base._weighted_statistics(actual, exact, volume, active & mask)
        item["squared_error_fraction"] = float(
            item["squared_error"] / max(global_stats["squared_error"], 1.0e-300)
        )
        item["volume_fraction"] = float(
            item["volume"] / max(global_stats["volume"], 1.0e-300)
        )
        regions[name] = item
    layer = base._weighted_statistics(actual, exact, volume, active & ordinary_layer)
    layer["squared_error_fraction"] = float(
        layer["squared_error"] / max(global_stats["squared_error"], 1.0e-300)
    )
    return {
        "global": global_stats,
        "regions": regions,
        "ordinary_adjacent_layer_diagnostic": layer,
        "partition": {
            "active_owner_count": int(np.count_nonzero(active)),
            "assigned_owner_count": int(sum(np.count_nonzero(mask) for mask in masks.values())),
            "volume_recovery_relative": float(
                abs(sum(item["volume"] for item in regions.values()) - global_stats["volume"])
                / max(global_stats["volume"], 1.0e-300)
            ),
            "squared_error_recovery_relative": float(
                abs(
                    sum(item["squared_error"] for item in regions.values())
                    - global_stats["squared_error"]
                )
                / max(global_stats["squared_error"], 1.0e-300)
            ),
        },
    }


def _owner_coordinates(
    artifact: Any, observation_aggregate: Any
) -> dict[tuple[int, int, int], dict[str, Any]]:
    shape = artifact.global_geometry.shape
    eta = np.asarray(artifact.global_geometry.grid.z.centers, dtype=np.float64)
    result = {}
    for compact, flat in enumerate(observation_aggregate.owner_flat_ids):
        owner = tuple(int(value) for value in np.unravel_index(int(flat), shape))
        x, y = observation_aggregate.centroid_xy[compact]
        result[owner] = {
            "regular_chart": [float(x), float(y), float(eta[owner[2]])],
            "logical_owner_index": list(owner),
            "flat_owner_id": int(flat),
        }
    return result


def _periodic_distance(left: Sequence[float], right: Sequence[float]) -> float:
    dx = float(left[0]) - float(right[0])
    dy = float(left[1]) - float(right[1])
    deta = abs(float(left[2]) - float(right[2]))
    deta = min(deta, 2.0 * np.pi - deta)
    return math.sqrt(dx * dx + dy * dy + (0.2 * deta) ** 2)


def _ranked_owners(
    mask: np.ndarray,
    contribution: np.ndarray,
) -> list[tuple[int, int, int]]:
    owners = [tuple(int(v) for v in row) for row in np.argwhere(mask)]
    return sorted(
        owners,
        key=lambda owner: (-float(contribution[owner]), int(np.ravel_multi_index(owner, mask.shape))),
    )


def _choose_distinct(
    ranked: Iterable[tuple[int, int, int]],
    selected: list[tuple[int, int, int]],
    coordinates: Mapping[tuple[int, int, int], Mapping[str, Any]],
    minimum_distance: float,
) -> tuple[int, int, int]:
    ranked = list(ranked)
    for owner in ranked:
        if owner in selected:
            continue
        if all(
            _periodic_distance(
                coordinates[owner]["regular_chart"], coordinates[other]["regular_chart"]
            ) >= minimum_distance
            for other in selected
        ):
            return owner
    for owner in ranked:
        if owner not in selected:
            return owner
    raise RuntimeError("no distinct owner available for requested selection slot")


def _freeze_selection(
    *,
    resolution: int,
    artifact: Any,
    host: Any,
    masks: Mapping[str, np.ndarray],
    baseline: Mapping[str, Mapping[str, np.ndarray]],
    exact: Mapping[str, np.ndarray],
    coordinates: Mapping[tuple[int, int, int], Mapping[str, Any]],
) -> tuple[dict[str, list[tuple[int, int, int]]], list[dict[str, Any]]]:
    volume = np.asarray(host.aggregate_chart_volume, dtype=np.float64)
    contributions = {
        name: volume * np.square(baseline[name]["scalar_upwind"] - exact[name])
        for name in ("smooth_regular_scalar", "smooth_eta_varying_scalar")
    }
    spacing = 1.5 / float(resolution)
    selected: list[tuple[int, int, int]] = []
    records: list[dict[str, Any]] = []

    def add(owner, category, rationale):
        selected.append(owner)
        record = {
            "owner": list(owner),
            "category": category,
            "rationale": rationale,
            "coordinates": coordinates[owner],
            "angular_group_size": int(host.angular_group_size[owner[0]]),
            "baseline_scalar_upwind_squared_error_contribution": {
                field: float(contributions[field][owner]) for field in contributions
            },
        }
        records.append(record)

    for category, mask_name in (
        ("true_interface", "true_size_change_interface"),
        ("agglomerated_interior", "agglomerated_interior"),
    ):
        mask = masks[mask_name]
        for field in ("smooth_regular_scalar", "smooth_eta_varying_scalar"):
            owner = _choose_distinct(
                _ranked_owners(mask, contributions[field]), selected, coordinates, spacing
            )
            add(
                owner,
                category,
                f"largest available {field} scalar-upwind squared-error contribution",
            )

    ordinary = [
        tuple(int(v) for v in row) for row in np.argwhere(masks["ordinary"])
    ]
    interface_targets = [tuple(records[index]["owner"]) for index in (0, 1)]
    for target in interface_targets:
        ranked = sorted(
            ordinary,
            key=lambda owner: (
                _periodic_distance(
                    coordinates[owner]["regular_chart"], coordinates[target]["regular_chart"]
                ),
                coordinates[owner]["flat_owner_id"],
            ),
        )
        owner = _choose_distinct(ranked, selected, coordinates, spacing)
        add(owner, "ordinary_control", f"nearest ordinary owner to interface owner {target}")

    active = np.asarray(host.topology.is_active_owner, dtype=bool)
    axis_ranked = sorted(
        [tuple(int(v) for v in row) for row in np.argwhere(masks["axis_ring"])],
        key=lambda owner: (
            abs(coordinates[owner]["regular_chart"][2] - np.pi),
            coordinates[owner]["flat_owner_id"],
        ),
    )
    axis_owner = _choose_distinct(axis_ranked, selected, coordinates, 0.0)
    add(axis_owner, "axis", "geometric axis control nearest eta=pi")

    inner_mask = np.zeros_like(active)
    if active.shape[0] > 1:
        inner_mask[1] = active[1]
    inner_ranked = sorted(
        [tuple(int(v) for v in row) for row in np.argwhere(inner_mask)],
        key=lambda owner: (
            abs(math.atan2(
                coordinates[owner]["regular_chart"][1],
                coordinates[owner]["regular_chart"][0],
            ) - 0.5 * np.pi),
            abs(coordinates[owner]["regular_chart"][2] - np.pi),
            coordinates[owner]["flat_owner_id"],
        ),
    )
    inner_owner = _choose_distinct(inner_ranked, selected, coordinates, 0.0)
    add(inner_owner, "inner_ring", "non-axis i=1 geometric control near theta=pi/2, eta=pi")

    mapping = {f"{record['category']}_{index}": [tuple(record["owner"])] for index, record in enumerate(records)}
    if len(selected) != 8 or len(set(selected)) != 8:
        raise RuntimeError("selection must contain exactly eight distinct owners")
    return mapping, records


def _candidate_evaluation(aggregate: Any, model: Any, keys: Sequence[tuple[int, int, int, int]]):
    points_xy = []
    eta_indices = []
    for key in keys:
        points, _ = oracle._face_points_weights(model.geometry, key, 1)
        point = points[0]
        points_xy.append((float(point[0] * np.cos(point[1])), float(point[0] * np.sin(point[1]))))
        eta_indices.append(int(key[3]))
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
    return evaluation, np.asarray(points_xy, dtype=np.float64)


def _candidate_values(aggregate: Any, evaluation: Any, owner_values: np.ndarray) -> np.ndarray:
    compact = np.asarray(owner_values, dtype=np.float64).reshape(-1)[aggregate.owner_flat_ids]
    return np.asarray(evaluation.evaluate(compact), dtype=np.float64)


def _regular_reference_face_mask(
    model: Any, keys: Sequence[tuple[int, int, int, int]]
) -> np.ndarray:
    """Select face points where the ordinary continuum metric is defined."""

    return np.asarray(
        [not oracle._is_collapsed_axis_face(model, key) for key in keys],
        dtype=bool,
    )


def _replace_faces(
    original: CoordinateFaceValues3D,
    keys: Sequence[tuple[int, int, int, int]],
    values: Sequence[float],
    *,
    axis_filter: int | None = None,
) -> CoordinateFaceValues3D:
    arrays = [np.asarray(getattr(original, name), dtype=np.float64).copy() for name in "xyz"]
    for key, value in zip(keys, values):
        axis, i, j, k = key
        if axis_filter is not None and axis != axis_filter:
            continue
        arrays[axis][i, j, k] = float(value)
    return CoordinateFaceValues3D(*arrays)


def _candidate_face_sets(
    original: CoordinateFaceValues3D,
    keys: Sequence[tuple[int, int, int, int]],
    values: Sequence[float],
) -> dict[str, CoordinateFaceValues3D]:
    return {
        "all": _replace_faces(original, keys, values),
        "radial": _replace_faces(original, keys, values, axis_filter=0),
        "angular": _replace_faces(original, keys, values, axis_filter=1),
    }


def _changed_support_rows(
    keys: Sequence[tuple[int, int, int, int]], left: Any, right: Any
) -> list[dict[str, Any]]:
    left_matrix = left.matrix.tocsr()
    right_matrix = right.matrix.tocsr()
    result = []
    for row, key in enumerate(keys):
        a = left_matrix.indices[left_matrix.indptr[row] : left_matrix.indptr[row + 1]]
        b = right_matrix.indices[right_matrix.indptr[row] : right_matrix.indptr[row + 1]]
        if np.array_equal(a, b):
            continue
        result.append(
            {
                "key": list(key),
                "geometric_donors_compact": a.tolist(),
                "observation_donors_compact": b.tolist(),
                "support_jaccard": float(len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)),
                "symmetric_difference_count": int(len(set(a) ^ set(b))),
                "geometric_condition": float(left.diagnostics.condition[row]),
                "observation_condition": float(right.diagnostics.condition[row]),
                "geometric_l1": float(left.diagnostics.l1_norm[row]),
                "observation_l1": float(right.diagnostics.l1_norm[row]),
            }
        )
    return result


def _selected_stats(
    rows: Sequence[Mapping[str, Any]], stage: str, target: str
) -> dict[str, Any]:
    weight = np.asarray([row["aggregate_volume"] for row in rows], dtype=np.float64)
    actual = np.asarray([row[stage] for row in rows], dtype=np.float64)
    exact = np.asarray([row[target] for row in rows], dtype=np.float64)
    error = actual - exact
    denom = float(np.sum(weight))
    reference = math.sqrt(float(np.sum(weight * exact**2)) / max(denom, 1.0e-300))
    absolute = math.sqrt(float(np.sum(weight * error**2)) / max(denom, 1.0e-300))
    by_category = {}
    for category in sorted({str(row["category"]) for row in rows}):
        index = np.asarray([row["category"] == category for row in rows], dtype=bool)
        sub_weight = weight[index]
        sub_error = error[index]
        sub_exact = exact[index]
        sub_denom = float(np.sum(sub_weight))
        by_category[category] = {
            "owner_count": int(np.count_nonzero(index)),
            "volume": sub_denom,
            "absolute_l2": math.sqrt(float(np.sum(sub_weight * sub_error**2)) / max(sub_denom, 1.0e-300)),
            "reference_rms": math.sqrt(float(np.sum(sub_weight * sub_exact**2)) / max(sub_denom, 1.0e-300)),
            "squared_error": float(np.sum(sub_weight * sub_error**2)),
        }
    return {
        "owner_count": len(rows),
        "volume": denom,
        "absolute_l2": absolute,
        "reference_rms": reference,
        "relative_l2": absolute / max(reference, 1.0e-300),
        "squared_error": float(np.sum(weight * error**2)),
        "by_category": by_category,
    }


def _face_error_stats(records: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    valid = [record for record in records if not record["collapsed_axis_face"]]
    weight = np.asarray([record["logical_face_weight"] for record in valid], dtype=np.float64)
    result = {}
    for rule in RULES:
        state_error = np.asarray(
            [record[f"{prefix}_{rule}_state"] - record[f"{prefix}_exact_state"] for record in valid]
        )
        flux_error = np.asarray(
            [record[f"{prefix}_{rule}_weighted_flux"] - record[f"{prefix}_exact_weighted_flux"] for record in valid]
        )
        result[rule] = {
            "state_absolute_l2": math.sqrt(float(np.sum(weight * state_error**2)) / max(float(np.sum(weight)), 1.0e-300)),
            "weighted_flux_absolute_l2": math.sqrt(float(np.sum(flux_error**2)) / max(len(flux_error), 1)),
            "maximum_state_abs": float(np.max(np.abs(state_error))) if state_error.size else 0.0,
            "maximum_weighted_flux_abs": float(np.max(np.abs(flux_error))) if flux_error.size else 0.0,
        }
    return result


def _action_runner(model: Any, operands: Any):
    """Return one shape-stable JIT for all selected candidate substitutions."""
    # Names are static and stay outside the JIT.
    names = []
    for rule in ("G", "O"):
        for part in ("all", "radial", "angular"):
            names.append(f"upwind_{rule}_{part}")
    for rule in ("G", "O"):
        for part in ("all", "radial", "angular"):
            names.extend((f"A_{rule}_{part}", f"Braw_{rule}_{part}"))
        names.extend((f"centered_direct_{rule}", f"centered_swapped_{rule}"))

    def inner(*args):
        field_sets = args[:6]
        phi_sets = args[6:12]
        direct_sets = args[12:18]
        actions = []
        for index in range(6):
            actions.append(
                base._operator_call(
                    model, operands.phi_stencil, operands.omega_stencil,
                    f_trace=operands.phi_trace, g_trace=operands.omega_trace,
                    characteristic_scheme="scalar-third-order-upwind",
                    g_halo=operands.omega_halo, g_direct_states=direct_sets[index],
                )
            )
        for rule_offset in (0, 3):
            for part in range(3):
                index = rule_offset + part
                field_stencil = replace(operands.omega_stencil, face_values=field_sets[index])
                phi_stencil = replace(operands.phi_stencil, face_values=phi_sets[index])
                actions.append(base._operator_call(
                    model, operands.phi_stencil, field_stencil,
                    f_trace=operands.phi_trace, g_trace=operands.omega_trace,
                    characteristic_scheme="scalar-centered",
                ))
                actions.append(base._operator_call(
                    model, operands.omega_stencil, phi_stencil,
                    f_trace=operands.omega_trace, g_trace=operands.phi_trace,
                    characteristic_scheme="scalar-centered",
                ))
            field_stencil = replace(operands.omega_stencil, face_values=field_sets[rule_offset])
            phi_stencil = replace(operands.phi_stencil, face_values=phi_sets[rule_offset])
            actions.append(base._operator_call(
                model, phi_stencil, field_stencil,
                f_trace=operands.phi_trace, g_trace=operands.omega_trace,
                characteristic_scheme="centered",
            ))
            actions.append(base._operator_call(
                model, field_stencil, phi_stencil,
                f_trace=operands.omega_trace, g_trace=operands.phi_trace,
                characteristic_scheme="centered",
            ))
        return base._return_actions_to_owner(model, tuple(actions))[0]

    compiled = jax.jit(inner)

    def numeric(*args):
        values = compiled(*args)
        values = jax.block_until_ready(values)
        return {name: np.asarray(value) for name, value in zip(names, values)}

    return numeric


def _constant_closure(
    model: Any,
    selection: Mapping[str, list[tuple[int, int, int]]],
    face_keys: Sequence[tuple[int, int, int, int]],
    generator: Mapping[tuple[int, int, int, int], float],
) -> float:
    cells = model.control_volume_geometry.cells
    raw_volume = np.asarray(cells.raw_volume, dtype=np.float64)
    aggregate_volume = np.asarray(cells.aggregate_volume, dtype=np.float64)
    owner_arrays = _owner_tuple_arrays(cells)
    weighted = dict(generator)
    maximum = 0.0
    for owners in selection.values():
        for owner in owners:
            members = [
                tuple(int(v) for v in row)
                for row in np.argwhere(
                    (owner_arrays[0] == owner[0])
                    & (owner_arrays[1] == owner[1])
                    & (owner_arrays[2] == owner[2])
                )
            ]
            value = oracle._assemble_owner_residual(
                owner, members, raw_volume, aggregate_volume,
                np.ones_like(raw_volume), generator, weighted,
            )
            maximum = max(maximum, abs(float(value)))
    return maximum


def _case(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    resolution = int(args.resolution)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _progress("case_start", resolution=resolution)

    artifact_path, artifact = base._load_artifact(args.geometry, resolution)
    _, identity_artifact = base._load_artifact(args.geometry, 64)
    cache_path = args.baseline / f"N{resolution}.reference.npz"
    raw_state, regular_raw, eta_raw, exact_raw, cache_provenance = _load_cache(
        cache_path, resolution, float(args.time)
    )
    manifest_hash = _sha256(artifact_path / "manifest.json")
    if cache_provenance["geometry_manifest_sha256"] != manifest_hash:
        raise ValueError("candidate artifact and baseline cache manifest mismatch")

    reference_started = time.perf_counter()
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
    cached_reference = cache_provenance["reference_source"]
    if cached_reference.get("sidecar_sha256") != reference.provenance.get("sidecar_sha256"):
        raise ValueError("continuous reference sidecar identity mismatch")
    reference_seconds = time.perf_counter() - reference_started

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
        runtime = mms._runtime(artifact.global_geometry, artifact.owner_geometry, runtime_args)
    model = runtime.model
    if model is None:
        raise RuntimeError("fine-interval audit requires a single-device model")
    runtime_seconds = time.perf_counter() - runtime_started

    standard = base._region_masks(artifact.global_geometry, artifact.owner_geometry)
    masks, ordinary_layer, adjacency = _refined_masks(
        artifact.owner_geometry,
        model.control_volume_geometry.cells,
        standard,
    )
    observation_aggregate, observation_report = orientation._midpoint_observation_aggregate(
        artifact.owner_geometry, artifact.global_geometry
    )
    geometric_aggregate = oracle._aggregate_geometry(artifact.owner_geometry)
    coordinates = _owner_coordinates(artifact, observation_aggregate)
    rho_star = float(model.parameters.rho_star)

    field_inputs = {
        "actual_vorticity": raw_state,
        "smooth_regular_scalar": raw_state.replace(vorticity=regular_raw),
        "smooth_eta_varying_scalar": raw_state.replace(vorticity=eta_raw),
    }
    owner_states = {}
    operands = {}
    baseline_arrays = {}
    exact_owner = {}
    regional = {}
    baseline_started = time.perf_counter()
    for name, state in field_inputs.items():
        owner_state = mms._owner_project(state, artifact.owner_geometry)
        owner_states[name] = owner_state
        operands[name] = base._prepare_operands(model, owner_state, state)
        raw_actions = base._evaluate_variants(model, operands[name])["production_R"]
        baseline_arrays[name] = {
            "scalar_upwind": -raw_actions["pure_upwind"] / rho_star,
            "A": -raw_actions["phi_centered"] / rho_star,
            "B": raw_actions["omega_centered"] / rho_star,
            "centered": -raw_actions["centered"] / rho_star,
            "centered_swapped": -raw_actions["centered_swapped"] / rho_star,
        }
        exact_owner[name] = mms._owner_project_array(
            exact_raw[name], artifact.owner_geometry
        )
        regional[name] = {
            action: _regional_statistics(
                baseline_arrays[name][action], exact_owner[name],
                artifact.owner_geometry, masks, ordinary_layer,
            )
            for action in ("scalar_upwind", "centered")
        }
        identity_error = np.max(np.abs(
            baseline_arrays[name]["centered"]
            - 0.5 * (baseline_arrays[name]["A"] + baseline_arrays[name]["B"])
        ))
        regional[name]["centered_identity_max_abs"] = float(identity_error)
        regional[name]["centered_swapped_antisymmetry_max_abs"] = float(
            np.max(np.abs(
                baseline_arrays[name]["centered"]
                + baseline_arrays[name]["centered_swapped"]
            ))
        )
    baseline_seconds = time.perf_counter() - baseline_started

    selection, selection_records = _freeze_selection(
        resolution=resolution,
        artifact=artifact,
        host=artifact.owner_geometry,
        masks=masks,
        baseline=baseline_arrays,
        exact=exact_owner,
        coordinates=coordinates,
    )
    selection_path = output_dir / f"N{resolution}_selection.json"
    selection_payload = {
        "schema": "drbx.hsx-p03-fine-interval-selection-v1",
        "resolution": resolution,
        "frozen_before_candidate_evaluation": True,
        "geometry_manifest_sha256": manifest_hash,
        "reference_cache_sha256": _sha256(cache_path),
        "selection": selection,
        "records": selection_records,
    }
    _write(selection_path, selection_payload)
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    selection = {
        label: [tuple(int(v) for v in owner) for owner in owners]
        for label, owners in selection_payload["selection"].items()
    }
    _progress("selection_frozen", resolution=resolution, output=str(selection_path))

    raw_labels, face_labels = oracle._support(model.control_volume_geometry.cells, selection)
    face_keys = sorted(face_labels)
    trace = operands["actual_vorticity"]
    omega_trace_masks = tuple(
        np.asarray(getattr(trace.omega_trace, f"mask_{axis}"), dtype=bool)
        for axis in "xyz"
    )
    phi_trace_masks = tuple(
        np.asarray(getattr(trace.phi_trace, f"mask_{axis}"), dtype=bool)
        for axis in "xyz"
    )
    candidate_keys = [
        key for key in face_keys
        if key[0] in (0, 1)
        and not oracle._is_collapsed_axis_face(model, key)
        and not (key[0] == 0 and key[1] in (0, model.geometry.owned_shape[0]))
        and not bool(omega_trace_masks[key[0]][key[1], key[2], key[3]])
        and not bool(phi_trace_masks[key[0]][key[1], key[2], key[3]])
    ]
    geometric_eval, points_xy = _candidate_evaluation(
        geometric_aggregate, model, candidate_keys
    )
    observation_eval, observation_points_xy = _candidate_evaluation(
        observation_aggregate, model, candidate_keys
    )
    if not np.array_equal(points_xy, observation_points_xy):
        raise RuntimeError("candidate functionals do not share face points")

    support_changes = _changed_support_rows(candidate_keys, geometric_eval, observation_eval)
    support_summary = orientation._support_comparison(geometric_eval, observation_eval)
    candidate_contract = {
        "degree": 2,
        "donor_selection_policy": "nearest same-eta owners; 12->96 expansion schedule",
        "changed_quantities": "selected radial/angular advected face values only",
        "fixed": [
            "generator face velocities", "center/compression terms", "physical measures",
            "owner restriction", "stored owner data", "production eta reconstruction",
            "physical boundary traces", "rho_star/sign normalization",
        ],
        "geometric_conditioning": oracle._evaluation_diagnostics(geometric_eval),
        "observation_conditioning": oracle._evaluation_diagnostics(observation_eval),
        "geometric_reproduction": oracle._polynomial_reproduction(
            geometric_aggregate, geometric_eval, points_xy
        ),
        "observation_reproduction": oracle._polynomial_reproduction(
            observation_aggregate, observation_eval, points_xy
        ),
        "support_comparison": support_summary,
        "changed_support_rows": support_changes,
        "observation_functional": observation_report,
    }

    # Batch all exact face information.  Eta faces are retained for reporting
    # and common-face targets even though candidate substitutions are planar.
    exact_started = time.perf_counter()
    face_points = np.asarray([
        oracle._face_points_weights(model.geometry, key, 1)[0][0]
        for key in face_keys
    ], dtype=np.float64)
    face_weights = {
        key: float(oracle._face_points_weights(model.geometry, key, 1)[1][0])
        for key in face_keys
    }
    # The support includes the collapsed u=0 axis faces so that owner-flux
    # assembly remains complete.  The ordinary toroidal metric is singular at
    # those logical points, however, and their physical flux is identically
    # zero.  Evaluate the continuum oracle only on non-collapsed faces and
    # scatter those values back into full support-shaped arrays.
    regular_face_mask = _regular_reference_face_mask(model, face_keys)
    regular_face_points = face_points[regular_face_mask]
    prepared = reference.prepare(regular_face_points)
    raw_fields = reference._fields_raw(regular_face_points, float(args.time))
    phi_values = np.full(len(face_keys), np.nan, dtype=np.float64)
    phi_values[regular_face_mask] = np.asarray(raw_fields["phi"][0], dtype=np.float64)
    phi_gradient = np.zeros((len(face_keys), 3), dtype=np.float64)
    phi_gradient[regular_face_mask] = np.stack(raw_fields["phi"][1:4], axis=-1)
    omega_gradient, _omega_hessian = reference._local_omega_derivatives(
        regular_face_points
    )
    actual_vorticity_values = np.full(len(face_keys), np.nan, dtype=np.float64)
    actual_vorticity_values[regular_face_mask] = np.asarray(
        prepared.mms_omega, dtype=np.float64
    )
    actual_vorticity_gradient = np.zeros((len(face_keys), 3), dtype=np.float64)
    actual_vorticity_gradient[regular_face_mask] = omega_gradient
    exact_values = {"actual_vorticity": actual_vorticity_values}
    exact_gradients = {"actual_vorticity": actual_vorticity_gradient}
    for name in ("smooth_regular_scalar", "smooth_eta_varying_scalar"):
        value, gradient = continuous._smooth_value_gradient(
            name, regular_face_points, reference.eta_period
        )
        full_value = np.full(len(face_keys), np.nan, dtype=np.float64)
        full_value[regular_face_mask] = value
        full_gradient = np.zeros((len(face_keys), 3), dtype=np.float64)
        full_gradient[regular_face_mask] = gradient
        exact_values[name] = full_value
        exact_gradients[name] = full_gradient
    one_form = np.zeros((len(face_keys), 3), dtype=np.float64)
    one_form[regular_face_mask] = (
        np.asarray(prepared.bcov, dtype=np.float64)
        / np.maximum(np.asarray(prepared.B, dtype=np.float64)[:, None], 1.0e-30)
    )
    exact_seconds = time.perf_counter() - exact_started

    phi_owner_values = np.asarray(model._owner_field(owner_states["actual_vorticity"].phi))
    phi_g = _candidate_values(geometric_aggregate, geometric_eval, phi_owner_values)
    phi_o = _candidate_values(observation_aggregate, observation_eval, phi_owner_values)

    # Tiny completed-action preflight: one true interface and one ordinary owner.
    preflight_selection = {}
    for category in ("true_interface", "ordinary_control"):
        match = next(record for record in selection_records if record["category"] == category)
        preflight_selection[category] = [tuple(match["owner"])]
    _, preflight_face_labels = oracle._support(
        model.control_volume_geometry.cells, preflight_selection
    )
    preflight_keys = [
        key for key in sorted(preflight_face_labels)
        if key in set(candidate_keys)
    ]

    fields_payload = {}
    candidate_started = time.perf_counter()
    for name, state in field_inputs.items():
        field_started = time.perf_counter()
        field_owner_values = np.asarray(model._owner_field(owner_states[name].vorticity))
        field_g = _candidate_values(geometric_aggregate, geometric_eval, field_owner_values)
        field_o = _candidate_values(observation_aggregate, observation_eval, field_owner_values)
        stored_projection_mismatch = float(np.max(np.abs(
            mms._owner_project_array(np.asarray(state.vorticity), artifact.owner_geometry)
            - field_owner_values
        )))

        op = operands[name]
        field_face_sets = {}
        phi_face_sets = {}
        direct_sets = {}
        for rule, field_values, phi_values_candidate in (
            ("G", field_g, phi_g), ("O", field_o, phi_o)
        ):
            field_face_sets[rule] = _candidate_face_sets(
                op.omega_stencil.face_values, candidate_keys, field_values
            )
            phi_face_sets[rule] = _candidate_face_sets(
                op.phi_stencil.face_values, candidate_keys, phi_values_candidate
            )
            direct_sets[rule] = {}
            for part, faces in field_face_sets[rule].items():
                # The selected candidate is a common face value.  Set both
                # characteristic sides to it on changed faces; eta/boundary
                # faces retain their production states.
                axis_filter = None if part == "all" else (0 if part == "radial" else 1)
                left = _replace_faces(
                    op.omega_direct_states[0], candidate_keys, field_values,
                    axis_filter=axis_filter,
                )
                right = _replace_faces(
                    op.omega_direct_states[1], candidate_keys, field_values,
                    axis_filter=axis_filter,
                )
                direct_sets[rule][part] = (left, right)

        runner = _action_runner(model, op)
        def arguments_for(keys_subset):
            subset = set(keys_subset)
            field_sets = {}
            phi_sets = {}
            direct_subset = {}
            for rule, field_values, phi_values_candidate in (
                ("G", field_g, phi_g), ("O", field_o, phi_o)
            ):
                chosen_keys = [key for key in candidate_keys if key in subset]
                chosen_field = [
                    value for key, value in zip(candidate_keys, field_values) if key in subset
                ]
                chosen_phi = [
                    value for key, value in zip(candidate_keys, phi_values_candidate) if key in subset
                ]
                field_sets[rule] = _candidate_face_sets(
                    op.omega_stencil.face_values, chosen_keys, chosen_field
                )
                phi_sets[rule] = _candidate_face_sets(
                    op.phi_stencil.face_values, chosen_keys, chosen_phi
                )
                direct_subset[rule] = {}
                for part in ("all", "radial", "angular"):
                    axis_filter = None if part == "all" else (0 if part == "radial" else 1)
                    left = _replace_faces(
                        op.omega_direct_states[0], chosen_keys, chosen_field,
                        axis_filter=axis_filter,
                    )
                    right = _replace_faces(
                        op.omega_direct_states[1], chosen_keys, chosen_field,
                        axis_filter=axis_filter,
                    )
                    direct_subset[rule][part] = (left, right)
            result = []
            for rule in ("G", "O"):
                result.extend(field_sets[rule][part] for part in ("all", "radial", "angular"))
            for rule in ("G", "O"):
                result.extend(phi_sets[rule][part] for part in ("all", "radial", "angular"))
            for rule in ("G", "O"):
                result.extend(direct_subset[rule][part] for part in ("all", "radial", "angular"))
            return result

        argument_list = arguments_for(candidate_keys)

        preflight_seconds = None
        if name == "actual_vorticity":
            preflight_started = time.perf_counter()
            # Compile and execute the exact action path before the complete
            # eight-owner interpretation.  The full arrays retain production
            # values outside the two-owner preflight support.
            runner(*arguments_for(preflight_keys))
            preflight_seconds = time.perf_counter() - preflight_started
            _progress(
                "two_owner_preflight_complete",
                resolution=resolution,
                seconds=preflight_seconds,
                selected_face_count=len(preflight_keys),
            )
        action_started = time.perf_counter()
        candidate_raw = runner(*argument_list)
        action_seconds = time.perf_counter() - action_started

        candidate_effective = {}
        for rule in ("G", "O"):
            for part in ("all", "radial", "angular"):
                candidate_effective[f"upwind_{rule}_{part}"] = -candidate_raw[f"upwind_{rule}_{part}"] / rho_star
                candidate_effective[f"A_{rule}_{part}"] = -candidate_raw[f"A_{rule}_{part}"] / rho_star
                candidate_effective[f"B_{rule}_{part}"] = candidate_raw[f"Braw_{rule}_{part}"] / rho_star
            candidate_effective[f"centered_direct_{rule}"] = -candidate_raw[f"centered_direct_{rule}"] / rho_star
            candidate_effective[f"centered_swapped_{rule}"] = -candidate_raw[f"centered_swapped_{rule}"] / rho_star

        phi_generator = _compatible_flux_generator(
            op.phi_stencil, model.geometry, domain=model.domain,
            axis_regular_axes=model.axis_regular_axes, b_floor=1.0e-30,
        )
        field_generator = _compatible_flux_generator(
            op.omega_stencil, model.geometry, domain=model.domain,
            axis_regular_axes=model.axis_regular_axes, b_floor=1.0e-30,
        )
        phi_velocity_arrays = tuple(np.asarray(getattr(phi_generator, axis)) for axis in "xyz")
        field_velocity_arrays = tuple(np.asarray(getattr(field_generator, axis)) for axis in "xyz")
        production_field_faces = tuple(
            np.asarray(getattr(op.omega_stencil.face_values, axis)) for axis in "xyz"
        )
        production_phi_faces = tuple(
            np.asarray(getattr(op.phi_stencil.face_values, axis)) for axis in "xyz"
        )
        production_direct_sides = tuple(
            tuple(np.asarray(getattr(side, axis)) for axis in "xyz")
            for side in op.omega_direct_states
        )
        candidate_map = {
            "G": dict(zip(candidate_keys, field_g)),
            "O": dict(zip(candidate_keys, field_o)),
        }
        phi_candidate_map = {
            "G": dict(zip(candidate_keys, phi_g)),
            "O": dict(zip(candidate_keys, phi_o)),
        }
        index_by_key = {key: index for index, key in enumerate(face_keys)}
        face_records = []
        exact_maps = {rule: {} for rule in ("A", "B")}
        for key in face_keys:
            index = index_by_key[key]
            axis = key[0]
            area = face_weights[key]
            collapsed = oracle._is_collapsed_axis_face(model, key)
            v_phi_p = oracle._face_array_value(phi_velocity_arrays, key)
            v_field_p = oracle._face_array_value(field_velocity_arrays, key)
            v_phi_exact = 0.0 if collapsed else float(oracle._normal_velocity(axis, one_form[index:index+1], phi_gradient[index:index+1])[0])
            v_field_exact = 0.0 if collapsed else float(oracle._normal_velocity(axis, one_form[index:index+1], exact_gradients[name][index:index+1])[0])
            p_upwind = orientation._side_value(production_direct_sides, key, v_phi_p)
            p_a = oracle._face_array_value(production_field_faces, key)
            p_b = oracle._face_array_value(production_phi_faces, key)
            exact_field = None if collapsed else float(exact_values[name][index])
            exact_phi = None if collapsed else float(phi_values[index])
            record = {
                "key": list(key),
                "axis": AXIS_NAMES[axis],
                "labels": sorted(face_labels[key]),
                "logical_face_weight": area,
                "collapsed_axis_face": collapsed,
                "upwind_exact_state": exact_field,
                "upwind_P_state": p_upwind,
                "A_exact_state": exact_field,
                "A_P_state": p_a,
                "B_exact_state": exact_phi,
                "B_P_state": p_b,
            }
            for rule in ("G", "O"):
                changed = key in candidate_map[rule] and axis in (0, 1) and not collapsed
                record[f"upwind_{rule}_state"] = float(candidate_map[rule][key]) if changed else p_upwind
                record[f"A_{rule}_state"] = float(candidate_map[rule][key]) if changed else p_a
                record[f"B_{rule}_state"] = float(phi_candidate_map[rule][key]) if changed else p_b
            for prefix, velocity_p, velocity_exact in (
                ("upwind", v_phi_p, v_phi_exact),
                ("A", v_phi_p, v_phi_exact),
                ("B", v_field_p, v_field_exact),
            ):
                exact_state = record[f"{prefix}_exact_state"]
                record[f"{prefix}_exact_weighted_flux"] = 0.0 if collapsed else velocity_exact * exact_state * area
                for rule in RULES:
                    record[f"{prefix}_{rule}_weighted_flux"] = (
                        0.0 if collapsed else velocity_p * record[f"{prefix}_{rule}_state"] * area
                    )
            exact_maps["A"][key] = {
                "generator": v_phi_exact * area,
                "weighted": record["A_exact_weighted_flux"],
            }
            exact_maps["B"][key] = {
                "generator": v_field_exact * area,
                "weighted": record["B_exact_weighted_flux"],
            }
            face_records.append(record)

        cells = model.control_volume_geometry.cells
        raw_volume = np.asarray(cells.raw_volume, dtype=np.float64)
        aggregate_volume = np.asarray(cells.aggregate_volume, dtype=np.float64)
        owner_arrays = _owner_tuple_arrays(cells)
        raw_field = np.asarray(state.vorticity, dtype=np.float64)
        raw_phi = np.asarray(state.phi, dtype=np.float64)
        rows = []
        selection_category = {
            tuple(record["owner"]): record["category"] for record in selection_records
        }
        for owners in selection.values():
            for owner in owners:
                members = [
                    tuple(int(v) for v in row)
                    for row in np.argwhere(
                        (owner_arrays[0] == owner[0])
                        & (owner_arrays[1] == owner[1])
                        & (owner_arrays[2] == owner[2])
                    )
                ]
                common_a_raw = oracle._assemble_owner_residual(
                    owner, members, raw_volume, aggregate_volume, raw_field,
                    {key: exact_maps["A"][key]["generator"] for key in face_keys},
                    {key: exact_maps["A"][key]["weighted"] for key in face_keys},
                )
                common_b_raw = oracle._assemble_owner_residual(
                    owner, members, raw_volume, aggregate_volume, raw_phi,
                    {key: exact_maps["B"][key]["generator"] for key in face_keys},
                    {key: exact_maps["B"][key]["weighted"] for key in face_keys},
                )
                row = {
                    "owner": list(owner),
                    "category": selection_category[owner],
                    "aggregate_volume": float(aggregate_volume[owner]),
                    "continuum_target": float(exact_owner[name][owner]),
                    "common_A_target": float(-common_a_raw / rho_star),
                    "common_B_target": float(common_b_raw / rho_star),
                    "common_centered_target": float(0.5 * (-common_a_raw / rho_star + common_b_raw / rho_star)),
                    "upwind_P": float(baseline_arrays[name]["scalar_upwind"][owner]),
                    "A_P": float(baseline_arrays[name]["A"][owner]),
                    "B_P": float(baseline_arrays[name]["B"][owner]),
                    "centered_P_P": float(baseline_arrays[name]["centered"][owner]),
                }
                for rule in ("G", "O"):
                    for part in ("all", "radial", "angular"):
                        row[f"upwind_{rule}_{part}"] = float(candidate_effective[f"upwind_{rule}_{part}"][owner])
                        row[f"A_{rule}_{part}"] = float(candidate_effective[f"A_{rule}_{part}"][owner])
                        row[f"B_{rule}_{part}"] = float(candidate_effective[f"B_{rule}_{part}"][owner])
                    row[f"centered_{rule}_P"] = 0.5 * (row[f"A_{rule}_all"] + row["B_P"])
                    row[f"centered_P_{rule}"] = 0.5 * (row["A_P"] + row[f"B_{rule}_all"])
                    row[f"centered_{rule}_{rule}"] = 0.5 * (row[f"A_{rule}_all"] + row[f"B_{rule}_all"])
                    row[f"centered_direct_{rule}"] = float(candidate_effective[f"centered_direct_{rule}"][owner])
                    row[f"centered_swapped_{rule}"] = float(candidate_effective[f"centered_swapped_{rule}"][owner])
                    row[f"upwind_{rule}_orientation_closure"] = (
                        row[f"upwind_{rule}_all"] - row["upwind_P"]
                        - (row[f"upwind_{rule}_radial"] - row["upwind_P"])
                        - (row[f"upwind_{rule}_angular"] - row["upwind_P"])
                    )
                    for constituent in ("A", "B"):
                        row[f"{constituent}_{rule}_orientation_closure"] = (
                            row[f"{constituent}_{rule}_all"] - row[f"{constituent}_P"]
                            - (row[f"{constituent}_{rule}_radial"] - row[f"{constituent}_P"])
                            - (row[f"{constituent}_{rule}_angular"] - row[f"{constituent}_P"])
                        )
                    row[f"centered_{rule}_identity_mismatch"] = row[f"centered_direct_{rule}"] - row[f"centered_{rule}_{rule}"]
                    row[f"centered_{rule}_swapped_antisymmetry"] = row[f"centered_direct_{rule}"] + row[f"centered_swapped_{rule}"]
                rows.append(row)

        stats = {
            "continuum": {
                "upwind": {stage: _selected_stats(rows, stage, "continuum_target") for stage in ("upwind_P", "upwind_G_all", "upwind_O_all")},
                "centered_constituent_A": {stage: _selected_stats(rows, stage, "continuum_target") for stage in ("A_P", "A_G_all", "A_O_all")},
                "centered_constituent_B": {stage: _selected_stats(rows, stage, "continuum_target") for stage in ("B_P", "B_G_all", "B_O_all")},
                "centered_combinations": {
                    stage: _selected_stats(rows, stage, "continuum_target")
                    for stage in (
                        "centered_P_P", "centered_G_P", "centered_P_G", "centered_G_G",
                        "centered_O_P", "centered_P_O", "centered_O_O",
                    )
                },
            },
            "common_face": {
                "upwind": {stage: _selected_stats(rows, stage, "common_A_target") for stage in ("upwind_P", "upwind_G_all", "upwind_O_all")},
                "centered_constituent_A": {stage: _selected_stats(rows, stage, "common_A_target") for stage in ("A_P", "A_G_all", "A_O_all")},
                "centered_constituent_B": {stage: _selected_stats(rows, stage, "common_B_target") for stage in ("B_P", "B_G_all", "B_O_all")},
                "centered_combinations": {
                    stage: _selected_stats(rows, stage, "common_centered_target")
                    for stage in (
                        "centered_P_P", "centered_G_P", "centered_P_G", "centered_G_G",
                        "centered_O_P", "centered_P_O", "centered_O_O",
                    )
                },
            },
        }
        constant_generator = {
            key: oracle._face_array_value(phi_velocity_arrays, key) * face_weights[key]
            for key in face_keys
        }
        fields_payload[name] = {
            "owner_records": rows,
            "sample_statistics": stats,
            "face_records": face_records,
            "face_errors": {
                "upwind": _face_error_stats(face_records, "upwind"),
                "centered_A": _face_error_stats(face_records, "A"),
                "centered_B": _face_error_stats(face_records, "B"),
            },
            "stored_midpoint_projection_replay_max_abs": stored_projection_mismatch,
            "maximum_orientation_closure": float(max(
                abs(row[key]) for row in rows for key in row if key.endswith("orientation_closure")
            )),
            "maximum_centered_identity_mismatch": float(max(
                abs(row[f"centered_{rule}_identity_mismatch"]) for row in rows for rule in ("G", "O")
            )),
            "maximum_centered_swapped_antisymmetry": float(max(
                abs(row[f"centered_{rule}_swapped_antisymmetry"]) for row in rows for rule in ("G", "O")
            )),
            "constant_advected_completed_action_max_abs": _constant_closure(
                model, selection, face_keys, constant_generator
            ),
            "timings_seconds": {
                "two_owner_preflight": preflight_seconds,
                "candidate_actions": action_seconds,
                "total": time.perf_counter() - field_started,
            },
        }
        _progress("field_complete", resolution=resolution, field=name, seconds=time.perf_counter() - field_started)

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
            "sidecar_sha256": _sha256(args.reference_sidecar),
            "cache": str(cache_path.resolve()),
            "cache_sha256": _sha256(cache_path),
            "analytic_mms_eta_period": reference.eta_period,
            "continuous_evaluator_period": reference.metric_evaluator.period,
            "provenance": reference.provenance,
        },
        "boundary_contract": {
            "physical_wall_model": "legacy-velocity-trace",
            "parallel_velocity_wall_bc": "neumann",
            "neumann_ghost_scheme": "physical",
            "parallel_boundary_pairing": "characteristic-sat",
            "parallel_characteristic_wall_law": "energy-absorbing",
        },
        "regional_budget": regional,
        "regional_masks": {
            name: {
                "owner_count": int(np.count_nonzero(mask)),
                "physical_volume": float(np.sum(np.asarray(artifact.owner_geometry.aggregate_chart_volume)[mask])),
            }
            for name, mask in masks.items()
        },
        "ordinary_adjacent_layer_owner_count": int(np.count_nonzero(ordinary_layer)),
        "selection": selection_payload,
        "support": {
            "selected_owner_count": 8,
            "raw_member_count": int(len(raw_labels)),
            "incident_face_count": len(face_keys),
            "candidate_planar_face_count": len(candidate_keys),
            "candidate_contract": candidate_contract,
        },
        "fields": fields_payload,
        "scope": {
            "certification": "bounded bridge to P04; not a global candidate campaign",
            "production_changes": [],
            "new_resolutions": [],
            "candidate_rules": {"P": "production", "G": "geometric moments", "O": "midpoint/raw-volume observation moments"},
        },
        "timings_seconds": {
            "reference_load": reference_seconds,
            "runtime_setup": runtime_seconds,
            "baseline_replay": baseline_seconds,
            "exact_face_batch": exact_seconds,
            "candidate_fields": time.perf_counter() - candidate_started,
            "total_before_output": time.perf_counter() - started,
        },
    }
    output = output_dir / f"N{resolution}.json"
    _write(output, payload)
    _progress("case_complete", resolution=resolution, output=str(output), seconds=time.perf_counter() - started)
    return payload


def _orders(resolutions: Sequence[int], values: Sequence[float]) -> list[float]:
    return [
        float(math.log(left / right) / math.log(n1 / n0))
        for n0, n1, left, right in zip(resolutions[:-1], resolutions[1:], values[:-1], values[1:])
        if left > 0.0 and right > 0.0
    ]


def _merge(args: argparse.Namespace) -> dict[str, Any]:
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    cases.sort(key=lambda case: int(case["resolution"]))
    if [case["resolution"] for case in cases] != [48, 64]:
        raise ValueError("fine-interval merge requires N48 and N64")

    counterparts = []
    left_records = cases[0]["selection"]["records"]
    right_records = cases[1]["selection"]["records"]
    for left in left_records:
        nearest = min(
            right_records,
            key=lambda right: _periodic_distance(
                left["coordinates"]["regular_chart"], right["coordinates"]["regular_chart"]
            ),
        )
        counterparts.append(
            {
                "N48_owner": left["owner"],
                "N48_category": left["category"],
                "N48_group_size": left["angular_group_size"],
                "N64_owner": nearest["owner"],
                "N64_category": nearest["category"],
                "N64_group_size": nearest["angular_group_size"],
                "regular_chart_distance": _periodic_distance(
                    left["coordinates"]["regular_chart"], nearest["coordinates"]["regular_chart"]
                ),
                "topology_or_group_differs": bool(
                    left["category"] != nearest["category"]
                    or left["angular_group_size"] != nearest["angular_group_size"]
                ),
            }
        )

    baseline_acceptance = {}
    bounded = {}
    for field in FIELDS:
        baseline_acceptance[field] = {
            action: {
                "absolute_l2": [case["regional_budget"][field][action]["global"]["absolute_l2"] for case in cases],
                "diagnostic_order_48_to_64": _orders(
                    [48, 64],
                    [case["regional_budget"][field][action]["global"]["absolute_l2"] for case in cases],
                )[0],
            }
            for action in ("scalar_upwind", "centered")
        }
        bounded[field] = {}
        for target in ("continuum", "common_face"):
            bounded[field][target] = {}
            for operator, stages in cases[0]["fields"][field]["sample_statistics"][target].items():
                bounded[field][target][operator] = {}
                for stage in stages:
                    values = [case["fields"][field]["sample_statistics"][target][operator][stage]["absolute_l2"] for case in cases]
                    bounded[field][target][operator][stage] = {
                        "absolute_l2": values,
                        "cross_resolution_order": None,
                        "reason": "N48/N64 selections are independently frozen worst-cell samples; do not infer convergence order",
                    }

    payload = {
        "schema": SUMMARY_SCHEMA,
        "resolutions": [48, 64],
        "baseline": baseline_acceptance,
        "bounded_candidate": bounded,
        "nearest_selected_counterparts": counterparts,
        "cases": cases,
        "decision_contract": {
            "global_acceptance": "order >=1.8 on both 32->48 and 48->64 intervals per operator/field",
            "sample_role": "bounded mechanism evidence only; no convergence inference from unrelated worst-cell selections",
            "no_global_candidate_run": True,
            "no_production_changes": True,
        },
    }
    _write(args.output, payload)
    print(args.output, flush=True)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    case = sub.add_parser("case")
    case.add_argument("--geometry", type=Path, required=True)
    case.add_argument("--reference-sidecar", type=Path, required=True)
    case.add_argument("--baseline", type=Path, required=True)
    case.add_argument("--resolution", type=int, choices=(48, 64), required=True)
    case.add_argument("--time", type=float, default=1.0e-6)
    case.add_argument("--output", type=Path, required=True)
    case.set_defaults(handler=_case)
    merge = sub.add_parser("merge")
    merge.add_argument("inputs", nargs=2, type=Path)
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(handler=_merge)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
