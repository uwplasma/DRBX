#!/usr/bin/env python3
"""Bounded P06 point/mean/three-moment cell-face policy comparison."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]

import audit_p06_curvature_bounded as p  # noqa: E402
import audit_p06_boundary_functional as q  # noqa: E402
from drbx.geometry.fci_boundary_functional_reconstruction import (  # noqa: E402
    BoundaryRelation,
    apply_boundary_reconstruction,
    build_boundary_functional_geometry,
    build_moment_boundary_relation,
    half_open_periodic_patch_index,
    prepare_boundary_reconstruction,
)


PARENT = WORKSPACE / "work/p06_wall_followup_20260922"
ORIGINAL = WORKSPACE / "work/p06_curvature_bounded_20260921"
CORRECTED = WORKSPACE / "work/p06_curvature_corrected_20260922"
DEFAULT_OUTPUT = WORKSPACE / "work/p06_boundary_policy_20260922"
POLICIES = ("point", "mean", "three")
FACE_POLICIES = ("none",) + POLICIES
SCHEMA = "drbx.perpendicular.p06-boundary-policy-v1"


def _wall_patch(
    context: Any, point: np.ndarray, order: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    j = half_open_periodic_patch_index(
        context.y_faces, float(point[1]), 2.0 * np.pi
    )
    k = half_open_periodic_patch_index(
        context.z_faces, float(point[2]), context.eta_period
    )
    nodes, weights = p._face_quadrature(
        context,
        np.asarray([[0, context.resolution, j, k]], dtype=np.int64),
        order,
    )
    return nodes[0], weights[0], (j, k)


def _wall_geometry(context: Any, reference: Any, fit: Any, order: int = 3):
    nodes, weights, patch = _wall_patch(context, fit.center_logical, order)
    basis = replace(fit, coefficients=np.eye(fit.coefficients.shape[1]))
    value, gradient = p.numerics._evaluate_fit(basis, nodes, context.eta_period)
    metric = reference._metric(nodes)
    geometry = build_boundary_functional_geometry(
        value_rows=value.T,
        gradient_rows=gradient.transpose(1, 2, 0),
        g_contravariant=metric["gcontra"],
        jacobian=metric["J"],
        logical_quadrature_weight=weights,
        tangential_coordinates=nodes[:, 1:],
        outward_sign=1.0,
    )
    return geometry, patch


def _relation(
    context: Any,
    reference: Any,
    fit: Any,
    *,
    policy: str,
    functional: str,
) -> tuple[BoundaryRelation, tuple[int, int]]:
    if policy not in POLICIES:
        raise ValueError(f"unsupported relation policy {policy}")
    if policy == "point":
        point = np.asarray(fit.center_logical, dtype=np.float64).copy()
        point[0] = 1.0
        basis = replace(fit, coefficients=np.eye(fit.coefficients.shape[1]))
        value, gradient = p.numerics._evaluate_fit(
            basis, point[None, :], context.eta_period
        )
        metric = reference._metric(point[None, :])
        geometry = build_boundary_functional_geometry(
            value_rows=value.T,
            gradient_rows=gradient.transpose(1, 2, 0),
            g_contravariant=metric["gcontra"],
            jacobian=metric["J"],
            logical_quadrature_weight=np.ones(1),
            tangential_coordinates=point[None, 1:],
            outward_sign=1.0,
        )
        _nodes, _weights, patch = _wall_patch(context, fit.center_logical, 3)
        rows = (
            geometry.normal_derivative_rows
            if functional == "normal_derivative"
            else geometry.value_rows
        )
        return BoundaryRelation(
            constraint_rows=rows,
            rhs_map=np.ones((1, 1), dtype=np.float64),
            labels=(f"{functional}:wall-center",),
            functional=functional,
        ), patch
    geometry, patch = _wall_geometry(context, reference, fit)
    full = build_moment_boundary_relation(geometry, functional=functional)
    if policy == "three":
        return full, patch
    return BoundaryRelation(
        constraint_rows=full.constraint_rows[:1],
        rhs_map=np.ones((1, 1), dtype=np.float64),
        labels=(f"{functional}:surface-mean",),
        functional=functional,
    ), patch


def _constrain(
    context: Any,
    reference: Any,
    fit: Any,
    donors: np.ndarray,
    observation: np.ndarray,
    weight2: np.ndarray,
    owner_values: np.ndarray,
    policy: str,
) -> tuple[Any, dict[str, Any]]:
    normal_relation, normal_patch = _relation(
        context, reference, fit, policy=policy, functional="normal_derivative"
    )
    value_relation, value_patch = _relation(
        context, reference, fit, policy=policy, functional="value"
    )
    if normal_patch != value_patch:
        raise RuntimeError("normal/value patch association differs")
    normal_map = prepare_boundary_reconstruction(observation, weight2, normal_relation)
    value_map = prepare_boundary_reconstruction(observation, weight2, value_relation)
    coefficients = np.asarray(fit.coefficients).copy()
    zero_normal = np.zeros(normal_relation.rhs_map.shape[1])
    zero_value = np.zeros(value_relation.rhs_map.shape[1])
    for field in (0, 1, 2):
        coefficients[field] = np.asarray(
            apply_boundary_reconstruction(
                normal_map, owner_values[field, donors], zero_normal
            )
        )
    coefficients[4] = np.asarray(
        apply_boundary_reconstruction(
            value_map, owner_values[4, donors], zero_value
        )
    )
    identity = np.eye(observation.shape[1])
    normal_reproduction = np.asarray(
        apply_boundary_reconstruction(
            normal_map,
            observation @ identity,
            normal_relation.constraint_rows @ identity,
        )
    )
    value_reproduction = np.asarray(
        apply_boundary_reconstruction(
            value_map,
            observation @ identity,
            value_relation.constraint_rows @ identity,
        )
    )
    return replace(fit, coefficients=coefficients), {
        "patch": normal_patch,
        "constraint_rank": [normal_map.constraint_rank, value_map.constraint_rank],
        "reduced_rank": [normal_map.reduced_rank, value_map.reduced_rank],
        "constraint_residual_max": float(
            max(
                np.max(np.abs(normal_relation.constraint_rows @ coefficients[:3].T)),
                np.max(np.abs(value_relation.constraint_rows @ coefficients[4])),
            )
        ),
        "dynamic_cubic_reproduction_max": float(
            max(
                np.max(np.abs(normal_reproduction - identity)),
                np.max(np.abs(value_reproduction - identity)),
            )
        ),
        "reduced_condition_max": float(
            max(normal_map.reduced_condition, value_map.reduced_condition)
        ),
    }


def _cell_variants(
    context: Any,
    reference: Any,
    support: dict[str, np.ndarray],
    points: np.ndarray,
    owner_values: np.ndarray,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    values = {name: np.empty(points.shape[:2] + (5,)) for name in POLICIES}
    gradients = {
        name: np.empty(points.shape[:2] + (5, 3)) for name in POLICIES
    }
    diagnostics = {name: [] for name in POLICIES}
    for row, (key, nodes) in enumerate(zip(support["raw_keys"], points, strict=True)):
        center = np.asarray(
            (
                context.x_centers[key[0]],
                context.y_centers[key[1]],
                context.z_centers[key[2]],
            )
        )
        fit = p.numerics._fit_entity(
            context,
            center,
            axis=0,
            eta_index=int(key[2]),
            owner_values=owner_values,
        )
        fits = {name: fit for name in POLICIES}
        if key[0] == context.resolution - 1:
            donors, observation, weight2 = q._fit_system(context, fit)
            for name in POLICIES:
                fits[name], diag = _constrain(
                    context,
                    reference,
                    fit,
                    donors,
                    observation,
                    weight2,
                    owner_values,
                    name,
                )
                diagnostics[name].append(diag)
        for name, selected in fits.items():
            value, gradient = p.numerics._evaluate_fit(
                selected, nodes, context.eta_period
            )
            values[name][row] = value.T
            gradients[name][row] = gradient.transpose(1, 0, 2)
    return {
        name: (values[name], gradients[name]) for name in POLICIES
    }, diagnostics


def _face_variants(
    context: Any,
    support: dict[str, np.ndarray],
    reference: Any,
    owner_values: np.ndarray,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, Any]
]:
    points, _weights = p._face_quadrature(context, support["face_keys"], 3)
    states = {
        name: (
            np.empty(points.shape[:2] + (5,)),
            np.empty(points.shape[:2] + (4,)),
            np.empty(points.shape[:2] + (4,)),
        )
        for name in FACE_POLICIES
    }
    diagnostics: dict[str, list[dict[str, Any]]] = {name: [] for name in POLICIES}
    support_hash_match = True
    for row, (key, nodes) in enumerate(zip(support["face_keys"], points, strict=True)):
        axis, i, _j, k = map(int, key)
        if axis == 0 and i == 0:
            exact = p._evaluate_fields("regular_chart_heldout", reference, nodes)[0].T
            for central, left, right in states.values():
                central[row] = exact
                left[row] = exact[:, :4]
                right[row] = exact[:, :4]
            continue
        fit = p.numerics._fit_entity(
            context,
            nodes[len(nodes) // 2],
            axis=axis,
            eta_index=k,
            owner_values=owner_values,
        )
        base_value = p.numerics._evaluate_fit(fit, nodes, context.eta_period)[0].T
        radial_boundary = axis == 0 and i == context.resolution
        pair = None
        if not radial_boundary:
            live_hash, _level, _ties = p._live_side_donor_hash(
                context, nodes[len(nodes) // 2], axis=axis, eta_index=k
            )
            pair = p.p05_nodewise._fit_side_pair(
                context,
                nodes[len(nodes) // 2],
                nodes,
                axis=axis,
                eta_index=k,
                expected_donor_hash=live_hash,
            )
        for name, (central, left, right) in states.items():
            central[row] = base_value
            if pair is None:
                left[row] = base_value[:, :4]
                right[row] = base_value[:, :4]
            else:
                jump = pair.right_values[:4].T - pair.left_values[:4].T
                left[row] = base_value[:, :4] - 0.5 * jump
                right[row] = base_value[:, :4] + 0.5 * jump
        candidate_donors = pair.donors if pair is not None else q._fit_system(
            context, fit, axis=axis, eta_index=k
        )[0]
        if not q._support_reaches_wall(context, candidate_donors):
            continue
        donors, observation, central_weight2 = q._fit_system(
            context, fit, axis=axis, eta_index=k
        )
        if pair is not None and not np.array_equal(donors, pair.donors):
            support_hash_match = False
            raise ValueError("central and side supports differ")
        for name in POLICIES:
            central_fit, central_diag = _constrain(
                context,
                reference,
                fit,
                donors,
                observation,
                central_weight2,
                owner_values,
                name,
            )
            central_value = p.numerics._evaluate_fit(
                central_fit, nodes, context.eta_period
            )[0].T
            central, left, right = states[name]
            central[row] = central_value
            if pair is None:
                left[row] = central_value[:, :4]
                right[row] = central_value[:, :4]
                diagnostics[name].append(central_diag)
                continue
            left_fit = q._pair_fit(
                pair, nodes[len(nodes) // 2], pair.left_coefficients
            )
            right_fit = q._pair_fit(
                pair, nodes[len(nodes) // 2], pair.right_coefficients
            )
            left_fit, left_diag = _constrain(
                context,
                reference,
                left_fit,
                pair.donors,
                observation,
                pair.left_weight2,
                owner_values,
                name,
            )
            right_fit, right_diag = _constrain(
                context,
                reference,
                right_fit,
                pair.donors,
                observation,
                pair.right_weight2,
                owner_values,
                name,
            )
            left_value = p.numerics._evaluate_fit(
                left_fit, nodes, context.eta_period
            )[0].T
            right_value = p.numerics._evaluate_fit(
                right_fit, nodes, context.eta_period
            )[0].T
            jump = right_value[:, :4] - left_value[:, :4]
            left[row] = central_value[:, :4] - 0.5 * jump
            right[row] = central_value[:, :4] + 0.5 * jump
            diagnostics[name].extend((central_diag, left_diag, right_diag))
    compact = {
        name: {
            "map_count": len(items),
            "patches": sorted({tuple(item["patch"]) for item in items}),
            "constraint_residual_max": float(
                max((item["constraint_residual_max"] for item in items), default=0.0)
            ),
            "dynamic_cubic_reproduction_max": float(
                max((item["dynamic_cubic_reproduction_max"] for item in items), default=0.0)
            ),
            "reduced_condition_max": float(
                max((item["reduced_condition_max"] for item in items), default=0.0)
            ),
            "central_relationship_max": float(
                np.max(
                    np.abs(
                        0.5 * (states[name][1] + states[name][2])
                        - states[name][0][..., :4]
                    )
                )
            ),
            "interior_jump_rms": float(
                np.sqrt(
                    np.mean(
                        (
                            states[name][2][
                                ~(
                                    (support["face_keys"][:, 0] == 0)
                                    & np.isin(
                                        support["face_keys"][:, 1],
                                        (0, context.resolution),
                                    )
                                )
                            ]
                            - states[name][1][
                                ~(
                                    (support["face_keys"][:, 0] == 0)
                                    & np.isin(
                                        support["face_keys"][:, 1],
                                        (0, context.resolution),
                                    )
                                )
                            ]
                        )
                        ** 2
                    )
                )
            ),
        }
        for name, items in diagnostics.items()
    }
    compact["support_hash_match"] = support_hash_match
    return states, compact


def _heldout_boundary_diagnostics(
    context: Any,
    reference: Any,
    support: dict[str, np.ndarray],
    owner_values: np.ndarray,
    policy: str,
    field_name: str,
) -> dict[str, float]:
    maxima = {"normal": 0.0, "phi": 0.0, "value": 0.0, "gradient": 0.0}
    for key in support["raw_keys"]:
        if key[0] != context.resolution - 1:
            continue
        center = np.asarray(
            (
                context.x_centers[key[0]],
                context.y_centers[key[1]],
                context.z_centers[key[2]],
            )
        )
        fit = p.numerics._fit_entity(
            context,
            center,
            axis=0,
            eta_index=int(key[2]),
            owner_values=owner_values,
        )
        donors, observation, weight2 = q._fit_system(context, fit)
        fit, _diag = _constrain(
            context,
            reference,
            fit,
            donors,
            observation,
            weight2,
            owner_values,
            policy,
        )
        nodes, _weights, _patch = _wall_patch(context, center, 5)
        fitted_value, fitted_gradient = p.numerics._evaluate_fit(
            fit, nodes, context.eta_period
        )
        exact_value, exact_gradient = p._evaluate_fields(
            field_name, reference, nodes
        )
        metric = reference._metric(nodes)["gcontra"]
        normal = metric[:, 0, :] / np.sqrt(metric[:, 0, 0])[:, None]
        maxima["normal"] = max(
            maxima["normal"],
            float(
                np.max(
                    np.abs(np.einsum("qa,fqa->fq", normal, fitted_gradient[:3]))
                )
            ),
        )
        maxima["phi"] = max(maxima["phi"], float(np.max(np.abs(fitted_value[4]))))
        maxima["value"] = max(
            maxima["value"], float(np.max(np.abs(fitted_value - exact_value)))
        )
        maxima["gradient"] = max(
            maxima["gradient"],
            float(np.max(np.abs(fitted_gradient - exact_gradient))),
        )
    return maxima


def run_resolution(output: Path, n: int) -> dict[str, Any]:
    started = time.monotonic()
    parent = np.load(PARENT / f"N{n}.npz")
    old = np.load(ORIGINAL / f"N{n}.npz")
    corrected = np.load(CORRECTED / "analysis" / f"N{n}.npz")
    rows = np.asarray(parent["selected_rows"], dtype=np.int64)
    global_owners = np.asarray(old["owner_indices"], dtype=np.int64)[rows]
    context = p.cubic._load_context(p.GEOMETRY, p.BASELINE, n)
    reference = p.build_continuum_reference_from_sidecar(
        CORRECTED / "localized_reference_sidecar.json", verify_hashes=True
    )
    support = p._support(n, global_owners, p._raw_owner(context))
    points, weights = p._cell_quadrature(context, support["raw_keys"], 3)
    prepared_geometry = reference.prepare(points.reshape(-1, 3))
    wall_mask = np.asarray(parent["wall_mask"], dtype=bool)
    volume = np.asarray(parent["owner_volume"], dtype=np.float64)
    arrays: dict[str, np.ndarray] = {
        "selected_rows": rows,
        "wall_mask": wall_mask,
        "owner_volume": volume,
        "raw_keys": support["raw_keys"],
        "face_keys": support["face_keys"],
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "resolution": n,
        "selection": json.loads((PARENT / f"N{n}.selection.json").read_text()),
        "patch_convention": (
            "periodic coordinates map to half-open wall intervals [face_i,face_{i+1}); "
            "exact internal ties select the plus interval and the periodic upper seam maps to interval 0; "
            "one target-derived patch is shared by central/minus/plus fits"
        ),
        "fields": {},
        "inputs": {
            "parent_npz": p._identity(PARENT / f"N{n}.npz"),
            "original_npz": p._identity(ORIGINAL / f"N{n}.npz"),
            "corrected_npz": p._identity(CORRECTED / "analysis" / f"N{n}.npz"),
            "selection_json": p._identity(PARENT / f"N{n}.selection.json"),
        },
    }
    for field_name in p.FIELD_NAMES:
        raw_values, _raw_gradients = p._evaluate_fields(
            field_name, reference, p._raw_points(context)
        )
        owner_values = p._owner_observations(context, raw_values)
        if field_name == "corrected_frozen_mms":
            owner_values[3] = np.asarray(context.arrays["owner_values"][1])
        context.arrays["owner_values"] = owner_values
        cell_values, cell_diagnostics = _cell_variants(
            context, reference, support, points, owner_values
        )
        cells = {
            name: p._integrate_sources(
                value,
                gradient,
                prepared_geometry,
                weights,
                support["raw_local_owner"],
                len(rows),
            )
            for name, (value, gradient) in cell_values.items()
        }
        face_states, face_diagnostics = _face_variants(
            context, support, reference, owner_values
        )
        faces = {}
        for name, state in face_states.items():
            numerator, directional, wall, diagnostics = q._interface_from_states(
                context, support, reference, *(item.copy() for item in state)
            )
            evolution_volume = cells["point"]["evolution:volume"]
            faces[name] = {
                "correction": numerator / evolution_volume[:, None],
                "directional": directional / evolution_volume[:, None, None],
                "wall": wall / evolution_volume[:, None],
                "diagnostics": diagnostics,
            }
        old_face = np.asarray(parent[f"{field_name}:interior_jump"])
        no_constraint_replay = float(
            np.max(np.abs(faces["none"]["correction"] - old_face))
        )
        target = np.asarray(parent[f"{field_name}:target_q7"])
        table = {}
        for cell_policy in POLICIES:
            table[cell_policy] = {}
            for face_policy in POLICIES:
                action = (
                    cells[cell_policy]["evolution:total"]
                    + faces[face_policy]["correction"]
                )
                table[cell_policy][face_policy] = {
                    "wall_L2": q._metrics(action, target, volume, wall_mask),
                    "control_L2": q._metrics(action, target, volume, ~wall_mask),
                }
                arrays[
                    f"{field_name}:cell_{cell_policy}:face_{face_policy}:U"
                ] = action
        point_cell_replay = float(
            np.max(
                np.abs(
                    cells["point"]["evolution:total"]
                    - np.asarray(parent[f"{field_name}:constrained_wall_center:total"])
                )
            )
        )
        prior_path = (
            WORKSPACE / "work/p06_boundary_functional_20260922" / f"N{n}.npz"
        )
        with np.load(prior_path, allow_pickle=False) as prior:
            prior_three = np.asarray(prior[f"{field_name}:C_total"])
        three_cell_replay = float(
            np.max(np.abs(cells["three"]["evolution:total"] - prior_three))
        )
        exact_q3 = np.asarray(corrected[f"{field_name}:q3:evolution:total"])[rows]
        exact_q5 = np.asarray(corrected[f"{field_name}:q5:evolution:total"])[rows]
        exact_q7 = np.asarray(corrected[f"{field_name}:q7:evolution:total"])[rows]
        report["fields"][field_name] = {
            "factorial": table,
            "replay": {
                "new_no_constraint_face_vs_saved_old_face_max": no_constraint_replay,
                "point_cell_vs_parent_max": point_cell_replay,
                "three_cell_vs_prior_max": three_cell_replay,
            },
            "cell_diagnostics": {
                name: {
                    "constraint_residual_max": float(
                        max((x["constraint_residual_max"] for x in values), default=0.0)
                    ),
                    "dynamic_cubic_reproduction_max": float(
                        max((x["dynamic_cubic_reproduction_max"] for x in values), default=0.0)
                    ),
                    "reduced_condition_max": float(
                        max((x["reduced_condition_max"] for x in values), default=0.0)
                    ),
                    "patches": sorted({tuple(x["patch"]) for x in values}),
                }
                for name, values in cell_diagnostics.items()
            },
            "face_diagnostics": face_diagnostics,
            "heldout_q5": {
                name: _heldout_boundary_diagnostics(
                    context, reference, support, owner_values, name, field_name
                )
                for name in POLICIES
            },
            "reference": {
                "q3_vs_q7_L2": q._metrics(
                    exact_q3, exact_q7, volume, np.ones(len(rows), dtype=bool)
                ),
                "q5_vs_q7_L2": q._metrics(
                    exact_q5, exact_q7, volume, np.ones(len(rows), dtype=bool)
                ),
            },
            "M_plus_R_closure_max": float(
                max(
                    np.max(
                        np.abs(
                            value["evolution:material"]
                            + value["evolution:remainder"]
                            - value["evolution:total"]
                        )
                    )
                    for value in cells.values()
                )
            ),
            "physical_wall_correction_max": float(
                max(np.max(np.abs(value["wall"])) for value in faces.values())
            ),
        }
        for name, value in cells.items():
            for term in ("material", "remainder", "total"):
                arrays[f"{field_name}:cell_{name}:{term}"] = value[f"evolution:{term}"]
            arrays[f"{field_name}:cell_{name}:directional"] = value[
                "evolution:total:directional"
            ]
        for name, value in faces.items():
            arrays[f"{field_name}:face_{name}:correction"] = value["correction"]
            arrays[f"{field_name}:face_{name}:directional"] = value["directional"]
        arrays[f"{field_name}:target_q7"] = target
        arrays[f"{field_name}:exact_q3"] = exact_q3
        arrays[f"{field_name}:exact_q5"] = exact_q5
        arrays[f"{field_name}:exact_q7"] = exact_q7
    report["timing"] = {
        "seconds": time.monotonic() - started,
        "peak_gib": p._peak_gib(),
    }
    q._write_npz(output / f"N{n}.npz", arrays)
    q._write_json(output / f"N{n}.json", report)
    return report


def summarize(output: Path) -> dict[str, Any]:
    cases = [json.loads((output / f"N{n}.json").read_text()) for n in (32, 48, 64)]
    summary = {
        "schema": f"{SCHEMA}.summary",
        "resolutions": [32, 48, 64],
        "factorial": {
            field: {
                cell: {
                    face: [
                        case["fields"][field]["factorial"][cell][face]["wall_L2"]
                        for case in cases
                    ]
                    for face in POLICIES
                }
                for cell in POLICIES
            }
            for field in p.FIELD_NAMES
        },
        "validation": {
            "no_constraint_face_replay_max": max(
                case["fields"][field]["replay"][
                    "new_no_constraint_face_vs_saved_old_face_max"
                ]
                for case in cases
                for field in p.FIELD_NAMES
            ),
            "point_cell_replay_max": max(
                case["fields"][field]["replay"]["point_cell_vs_parent_max"]
                for case in cases
                for field in p.FIELD_NAMES
            ),
            "three_cell_replay_max": max(
                case["fields"][field]["replay"]["three_cell_vs_prior_max"]
                for case in cases
                for field in p.FIELD_NAMES
            ),
            "constraint_residual_max": max(
                case["fields"][field][kind][policy]["constraint_residual_max"]
                for case in cases
                for field in p.FIELD_NAMES
                for kind in ("cell_diagnostics", "face_diagnostics")
                for policy in POLICIES
            ),
            "dynamic_cubic_reproduction_max": max(
                case["fields"][field][kind][policy]["dynamic_cubic_reproduction_max"]
                for case in cases
                for field in p.FIELD_NAMES
                for kind in ("cell_diagnostics", "face_diagnostics")
                for policy in POLICIES
            ),
            "M_plus_R_closure_max": max(
                case["fields"][field]["M_plus_R_closure_max"]
                for case in cases
                for field in p.FIELD_NAMES
            ),
            "physical_wall_correction_max": max(
                case["fields"][field]["physical_wall_correction_max"]
                for case in cases
                for field in p.FIELD_NAMES
            ),
        },
        "patch_convention": cases[0]["patch_convention"],
    }
    q._write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("32", "48", "64", "summarize"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.mode == "summarize":
        summarize(args.output)
    else:
        run_resolution(args.output, int(args.mode))


if __name__ == "__main__":
    main()
