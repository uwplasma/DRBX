#!/usr/bin/env python3
"""Bounded P06 A/B/C wall-functional reconstruction comparison.

This research driver reuses the frozen P06 selections and references.  It does
not change a production selector or wall law.  A is the archived unconstrained
cubic, B is the archived single-wall-center cell constraint, and C applies the
low-order ``{1,s_theta,s_eta}`` physical-wall moments to wall-adjacent cell
fits and to central/biased face fits whose frozen donor support reaches the
radial wall.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
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
from drbx.geometry.fci_boundary_functional_reconstruction import (  # noqa: E402
    BOUNDARY_FUNCTIONAL_RECONSTRUCTION_VERSION,
    apply_boundary_reconstruction,
    build_boundary_functional_geometry,
    build_moment_boundary_relation,
    prepare_boundary_reconstruction,
)


PARENT = WORKSPACE / "work/p06_wall_followup_20260922"
ORIGINAL = WORKSPACE / "work/p06_curvature_bounded_20260921"
CORRECTED = WORKSPACE / "work/p06_curvature_corrected_20260922"
DEFAULT_OUTPUT = WORKSPACE / "work/p06_boundary_functional_20260922"
SCHEMA = "drbx.perpendicular.p06-boundary-functional-v1"


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_json(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit_system(
    context: Any, fit: Any, *, axis: int = 0, eta_index: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    policy = p.cubic.POLICY["deficient_row_expansion_schedule"][
        fit.diagnostics["fallback_level"]
    ]
    donors, _weights, _diagnostics, _ties = p.cubic._row_batch(
        context,
        int(axis),
        (
            int(np.argmin(np.abs(context.z_centers - fit.center_logical[2])))
            if eta_index is None
            else int(eta_index)
        ),
        fit.center_logical[None, :],
        exact_query=False,
        count=int(policy["donors_per_plane"]),
        pool_count=int(policy["candidate_pool_per_plane"]),
    )
    donors = donors[0]
    digest = hashlib.sha256(np.asarray(donors, dtype="<i8").tobytes()).hexdigest()
    if digest != fit.diagnostics["donor_sha256"]:
        raise ValueError("frozen cell donor identity changed")
    observation = p.cubic._centered_observations(
        context,
        donors[None, :],
        fit.center_regular[None, :],
        fit.scale[None, :],
    )[0]
    eta = p.cubic.base._unwrap_periodic(
        context.owner_eta[donors], fit.center_regular[2], context.eta_period
    )
    distance2 = np.sum(
        (
            (context.arrays["owner_centroid_xy"][donors] - fit.center_regular[:2])
            / fit.scale[:2]
        )
        ** 2,
        axis=1,
    ) + ((eta - fit.center_regular[2]) / fit.scale[2]) ** 2
    return donors, observation, (1.0 + distance2) ** -2


def _nearest_periodic(values: np.ndarray, point: float, period: float) -> int:
    delta = (np.asarray(values) - point + 0.5 * period) % period - 0.5 * period
    return int(np.argmin(np.abs(delta)))


def _wall_nodes(context: Any, point: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    j = _nearest_periodic(context.y_centers, float(point[1]), 2.0 * np.pi)
    k = _nearest_periodic(context.z_centers, float(point[2]), context.eta_period)
    nodes, weights = p._face_quadrature(
        context, np.asarray([[0, context.resolution, j, k]], dtype=np.int64), order
    )
    return nodes[0], weights[0]


def _relations(context: Any, reference: Any, fit: Any):
    nodes, weights = _wall_nodes(context, fit.center_logical, 3)
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
    return (
        geometry,
        build_moment_boundary_relation(geometry, functional="normal_derivative"),
        build_moment_boundary_relation(geometry, functional="value"),
    )


def _constrain_fit(
    context: Any,
    reference: Any,
    fit: Any,
    donors: np.ndarray,
    observation: np.ndarray,
    weight2: np.ndarray,
    owner_values: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    geometry, normal_relation, value_relation = _relations(context, reference, fit)
    normal_map = prepare_boundary_reconstruction(observation, weight2, normal_relation)
    value_map = prepare_boundary_reconstruction(observation, weight2, value_relation)
    coefficients = np.asarray(fit.coefficients).copy()
    zero = np.zeros(3, dtype=np.float64)
    for field in (0, 1, 2):
        coefficients[field] = np.asarray(
            apply_boundary_reconstruction(normal_map, owner_values[field, donors], zero)
        )
    coefficients[4] = np.asarray(
        apply_boundary_reconstruction(value_map, owner_values[4, donors], zero)
    )
    normal_defect = np.max(
        np.abs(normal_relation.constraint_rows @ coefficients[:3].T)
    )
    value_defect = np.max(np.abs(value_relation.constraint_rows @ coefficients[4]))
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
        "donor_sha256": hashlib.sha256(
            np.asarray(donors, dtype="<i8").tobytes()
        ).hexdigest(),
        "constraint_rank": [normal_map.constraint_rank, value_map.constraint_rank],
        "reduced_rank": [normal_map.reduced_rank, value_map.reduced_rank],
        "reduced_condition": [
            normal_map.reduced_condition,
            value_map.reduced_condition,
        ],
        "owner_map_max": float(
            max(np.max(np.abs(normal_map.owner_map)), np.max(np.abs(value_map.owner_map)))
        ),
        "boundary_map_max": float(
            max(
                np.max(np.abs(normal_map.boundary_map)),
                np.max(np.abs(value_map.boundary_map)),
            )
        ),
        "constraint_residual_max": float(max(normal_defect, value_defect)),
        "dynamic_cubic_reproduction_max": float(
            max(
                np.max(np.abs(normal_reproduction - identity)),
                np.max(np.abs(value_reproduction - identity)),
            )
        ),
        "moment_modes": ["1", "s_theta", "s_eta"],
        "phi_boundary_data_source": "current selected zero Dirichlet wall value",
        "omega_constraint": "none",
        "surface_measure": "abs(J)*sqrt(g^{uu})",
        "normal": "outward g^{u alpha}/sqrt(g^{uu})",
        "geometry_version": geometry.version,
    }


def _cell_moment_fit(
    context: Any, reference: Any, fit: Any, owner_values: np.ndarray
) -> tuple[Any, dict[str, Any]]:
    donors, observation, weight2 = _fit_system(context, fit)
    return _constrain_fit(
        context, reference, fit, donors, observation, weight2, owner_values
    )


def _pair_fit(pair: Any, point: np.ndarray, coefficients: np.ndarray) -> Any:
    return p.p05_nodewise._make_fit(
        point,
        pair.center_regular,
        pair.scale,
        coefficients,
        {"fallback_level": pair.diagnostics["fallback_level"],
         "donor_sha256": pair.diagnostics["donor_sha256"]},
    )


def _support_reaches_wall(context: Any, donors: np.ndarray) -> bool:
    raw = np.asarray(context.arrays["owner_flat_ids"], dtype=np.int64)[donors]
    return bool(np.any(p.numerics._raw_keys(context.resolution, raw)[:, 0] == context.resolution - 1))


def _face_states(
    context: Any,
    support: dict[str, np.ndarray],
    reference: Any,
    owner_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    points, _weights = p._face_quadrature(context, support["face_keys"], 3)
    central = np.empty((len(points), points.shape[1], 5), dtype=np.float64)
    left = np.empty((len(points), points.shape[1], 4), dtype=np.float64)
    right = np.empty_like(left)
    affected = 0
    diagnostics = []
    for row, (key, nodes) in enumerate(zip(support["face_keys"], points, strict=True)):
        axis, i, _j, k = map(int, key)
        if axis == 0 and i == 0:
            central[row] = p._evaluate_fields(
                "regular_chart_heldout", reference, nodes
            )[0].T
            left[row] = central[row, :, :4]
            right[row] = central[row, :, :4]
            continue
        fit = p.numerics._fit_entity(
            context,
            nodes[len(nodes) // 2],
            axis=axis,
            eta_index=k,
            owner_values=owner_values,
        )
        value, _gradient = p.numerics._evaluate_fit(fit, nodes, context.eta_period)
        central[row] = value.T
        if axis == 0 and i in (0, context.resolution):
            left[row] = central[row, :, :4]
            right[row] = central[row, :, :4]
            continue
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
        pair_left = pair.left_values.T
        pair_right = pair.right_values.T
        if _support_reaches_wall(context, pair.donors):
            affected += 1
            donors, observation, central_weight2 = _fit_system(
                context, fit, axis=axis, eta_index=k
            )
            if not np.array_equal(donors, pair.donors):
                raise ValueError("central and biased face fits changed donor support")
            constrained, central_diag = _constrain_fit(
                context,
                reference,
                fit,
                donors,
                observation,
                central_weight2,
                owner_values,
            )
            value, _gradient = p.numerics._evaluate_fit(
                constrained, nodes, context.eta_period
            )
            central[row] = value.T
            left_fit = _pair_fit(pair, nodes[len(nodes) // 2], pair.left_coefficients)
            right_fit = _pair_fit(pair, nodes[len(nodes) // 2], pair.right_coefficients)
            left_fit, left_diag = _constrain_fit(
                context,
                reference,
                left_fit,
                pair.donors,
                observation,
                pair.left_weight2,
                owner_values,
            )
            right_fit, right_diag = _constrain_fit(
                context,
                reference,
                right_fit,
                pair.donors,
                observation,
                pair.right_weight2,
                owner_values,
            )
            pair_left = p.numerics._evaluate_fit(
                left_fit, nodes, context.eta_period
            )[0].T
            pair_right = p.numerics._evaluate_fit(
                right_fit, nodes, context.eta_period
            )[0].T
            diagnostics.extend((central_diag, left_diag, right_diag))
        jump = pair_right[:, :4] - pair_left[:, :4]
        left[row] = central[row, :, :4] - 0.5 * jump
        right[row] = central[row, :, :4] + 0.5 * jump
    return central, left, right, {
        "affected_face_count": affected,
        "prepared_map_count": len(diagnostics),
        "constraint_residual_max": float(
            max((item["constraint_residual_max"] for item in diagnostics), default=0.0)
        ),
        "reduced_condition_max": float(
            max((max(item["reduced_condition"]) for item in diagnostics), default=0.0)
        ),
        "dynamic_cubic_reproduction_max": float(
            max((item["dynamic_cubic_reproduction_max"] for item in diagnostics), default=0.0)
        ),
        "donor_hash_count": len({item["donor_sha256"] for item in diagnostics}),
        "central_relationship_max": float(
            np.max(np.abs(0.5 * (left + right) - central[..., :4]))
        ),
    }


def _interface_from_states(
    context: Any,
    support: dict[str, np.ndarray],
    reference: Any,
    central: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    keys = support["face_keys"]
    points, weights = p._face_quadrature(context, keys, order=3)
    collapsed = (keys[:, 0] == 0) & (keys[:, 1] == 0)
    regular = ~collapsed
    rj, rb, rk = p._curvature_face_geometry(reference, points[regular].reshape(-1, 3))
    J = np.zeros(points.shape[:2]); B = np.ones(points.shape[:2])
    K = np.zeros(points.shape[:2] + (3,))
    J[regular] = rj.reshape((-1, points.shape[1]))
    B[regular] = rb.reshape((-1, points.shape[1]))
    K[regular] = rk.reshape((-1, points.shape[1], 3))
    normal = J * K[
        np.arange(len(keys))[:, None], np.arange(points.shape[1])[None, :], keys[:, 0, None]
    ] / np.maximum(B * B, 1.0e-30)
    wall_rows = np.flatnonzero((keys[:, 0] == 0) & (keys[:, 1] == context.resolution))
    for row in wall_rows:
        interior = central[row, :, :4]
        exterior, working, _fallback = p._curvature_bc_characteristic_wall_states(
            p.blob.jnp.asarray(interior), p.blob.jnp.asarray(interior),
            p.blob.jnp.asarray(B[row]), p.TAU, p.blob.jnp.asarray(normal[row]),
            interior_on_right=False, positivity_floor=p.FLOOR,
        )
        left[row] = interior
        right[row] = np.asarray(exterior)
        central[row, :, :4] = np.asarray(working)
    matrix = p._principal_matrix_numpy(central[..., :4], B)
    flux_matrix = -normal[..., None, None] * matrix
    jump = right - left
    absolute, _fallback = p._absolute_action(flux_matrix, jump)
    material = np.einsum("fqij,fqj->fqi", flux_matrix, jump)
    dplus = 0.5 * (material + absolute)
    dminus = 0.5 * (material - absolute)
    dplus[collapsed] = 0.0; dminus[collapsed] = 0.0
    integrated_plus = -np.sum(weights[..., None] * dplus, axis=1)
    integrated_minus = -np.sum(weights[..., None] * dminus, axis=1)
    raw = np.zeros((len(support["raw_indices"]), 4))
    directional = np.zeros((len(raw), 3, 4))
    wall = np.zeros_like(raw)
    lookup = {tuple(map(int, key)): row for row, key in enumerate(support["raw_keys"])}
    for face, key in enumerate(keys):
        axis = int(key[0]); is_wall = axis == 0 and int(key[1]) == context.resolution
        minus, plus = p._face_neighbors(context.resolution, key)
        for neighbor, contribution in ((minus, integrated_minus[face]), (plus, integrated_plus[face])):
            if neighbor in lookup:
                row = lookup[neighbor]
                raw[row] += contribution; directional[row, axis] += contribution
                if is_wall:
                    wall[row] += contribution
    owner_count = int(np.max(support["raw_local_owner"])) + 1
    owner = np.zeros((owner_count, 4)); owner_directional = np.zeros((owner_count, 3, 4))
    owner_wall = np.zeros((owner_count, 4))
    np.add.at(owner, support["raw_local_owner"], raw)
    np.add.at(owner_directional, support["raw_local_owner"], directional)
    np.add.at(owner_wall, support["raw_local_owner"], wall)
    return owner, owner_directional, owner_wall, {
        "jump_rms": float(np.sqrt(np.mean(jump * jump))),
        "physical_wall_correction_max": float(np.max(np.abs(owner_wall))),
        "wall_face_count": int(len(wall_rows)),
    }


def _metrics(action: np.ndarray, target: np.ndarray, volume: np.ndarray, mask: np.ndarray):
    error = action[mask] - target[mask]
    weight = volume[mask]
    return np.sqrt(np.sum(weight[:, None] * error * error, axis=0) / np.sum(weight))


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
        "selected_rows": rows, "wall_mask": wall_mask, "owner_volume": volume,
        "raw_keys": support["raw_keys"], "face_keys": support["face_keys"],
    }
    report: dict[str, Any] = {
        "schema": SCHEMA, "resolution": n,
        "selection": json.loads((PARENT / f"N{n}.selection.json").read_text()),
        "frozen_contract": {
            "degree": 3,
            "support": "selection-v3 donors and expansion level unchanged",
            "bias": p.BIAS,
            "boundary_moments": ["1", "s_theta", "s_eta"],
            "thermodynamic_boundary_data": "homogeneous physical-normal derivative",
            "phi_boundary_data": "current selected zero Dirichlet value",
            "omega_boundary_constraint": "none",
            "characteristic_model": "unchanged coupled four-field solve with recentered jump",
        },
        "fields": {},
    }
    for field_name in p.FIELD_NAMES:
        raw_values, _raw_gradients = p._evaluate_fields(
            field_name, reference, p._raw_points(context)
        )
        owner_values = p._owner_observations(context, raw_values)
        if field_name == "corrected_frozen_mms":
            owner_values[3] = np.asarray(context.arrays["owner_values"][1])
        context.arrays["owner_values"] = owner_values
        values = np.empty((len(points), points.shape[1], 5))
        gradients = np.empty(values.shape + (3,))
        cell_diags = []
        boundary_q5 = []
        for row, (key, nodes) in enumerate(zip(support["raw_keys"], points, strict=True)):
            center = np.asarray((context.x_centers[key[0]], context.y_centers[key[1]], context.z_centers[key[2]]))
            fit = p.numerics._fit_entity(
                context, center, axis=0, eta_index=int(key[2]), owner_values=owner_values
            )
            if key[0] == n - 1:
                fit, diag = _cell_moment_fit(context, reference, fit, owner_values)
                cell_diags.append(diag)
                q5, _w5 = _wall_nodes(context, center, 5)
                qv, qg = p.numerics._evaluate_fit(fit, q5, context.eta_period)
                metric = reference._metric(q5)["gcontra"]
                normal = metric[:, 0, :] / np.sqrt(metric[:, 0, 0])[:, None]
                boundary_q5.append(
                    {
                        "normal_thermodynamic_max": np.max(
                            np.abs(np.einsum("qa,fqa->fq", normal, qg[:3])), axis=1
                        ),
                        "phi_value_max": float(np.max(np.abs(qv[4]))),
                    }
                )
            value, gradient = p.numerics._evaluate_fit(fit, nodes, context.eta_period)
            values[row] = value.T; gradients[row] = gradient.transpose(1, 0, 2)
        cell = p._integrate_sources(
            values, gradients, prepared_geometry, weights,
            support["raw_local_owner"], len(rows)
        )
        central, left, right, face_diag = _face_states(
            context, support, reference, owner_values
        )
        numerator, direction_num, wall_num, interface_diag = _interface_from_states(
            context, support, reference, central, left, right
        )
        evolution_volume = cell["evolution:volume"]
        jump = numerator / evolution_volume[:, None]
        jump_directional = direction_num / evolution_volume[:, None, None]
        wall_correction = wall_num / evolution_volume[:, None]
        target = np.asarray(parent[f"{field_name}:target_q7"])
        A_C = np.asarray(parent[f"{field_name}:fit:total"])
        A_U = A_C + np.asarray(parent[f"{field_name}:interior_jump"])
        B_C = np.asarray(parent[f"{field_name}:constrained_wall_center:total"])
        B_U = B_C + np.asarray(parent[f"{field_name}:interior_jump"])
        C_C = cell["evolution:total"]
        C_U = C_C + jump
        variants = {"A_C": A_C, "A_U": A_U, "B_C": B_C, "B_U": B_U, "C_C": C_C, "C_U": C_U}
        q3_exact = np.asarray(corrected[f"{field_name}:q3:evolution:total"])[rows]
        q5_exact = np.asarray(corrected[f"{field_name}:q5:evolution:total"])[rows]
        q7_exact = np.asarray(corrected[f"{field_name}:q7:evolution:total"])[rows]
        field_report = {
            "L2": {
                name: {
                    "wall": _metrics(action, target, volume, wall_mask),
                    "control": _metrics(action, target, volume, ~wall_mask),
                }
                for name, action in variants.items()
            },
            "boundary_moments": {
                "constraint_residual_max": float(max((d["constraint_residual_max"] for d in cell_diags), default=0.0)),
                "heldout_q5": boundary_q5,
            },
            "cell_maps": {
                "count": len(cell_diags),
                "constraint_ranks": [d["constraint_rank"] for d in cell_diags],
                "reduced_ranks": [d["reduced_rank"] for d in cell_diags],
                "reduced_condition_max": float(max((max(d["reduced_condition"]) for d in cell_diags), default=0.0)),
                "owner_map_max": float(max((d["owner_map_max"] for d in cell_diags), default=0.0)),
                "boundary_map_max": float(max((d["boundary_map_max"] for d in cell_diags), default=0.0)),
                "dynamic_cubic_reproduction_max": float(max((d["dynamic_cubic_reproduction_max"] for d in cell_diags), default=0.0)),
                "donor_sha256": [d["donor_sha256"] for d in cell_diags],
            },
            "face_maps": face_diag,
            "interface": interface_diag,
            "q3_exact_vs_q7_L2": _metrics(q3_exact, q7_exact, volume, np.ones(len(rows), bool)),
            "q5_vs_q7_L2": _metrics(q5_exact, q7_exact, volume, np.ones(len(rows), bool)),
            "M_plus_R_closure_max": float(np.max(np.abs(
                cell["evolution:material"] + cell["evolution:remainder"] - cell["evolution:total"]
            ))),
            "control_change_C_C_max": float(np.max(np.abs(C_C[~wall_mask] - A_C[~wall_mask]))),
            "physical_wall_characteristic_correction_max": float(np.max(np.abs(wall_correction))),
        }
        report["fields"][field_name] = field_report
        for name, action in variants.items(): arrays[f"{field_name}:{name}"] = action
        arrays[f"{field_name}:target_q7"] = target
        arrays[f"{field_name}:C_material"] = cell["evolution:material"]
        arrays[f"{field_name}:C_remainder"] = cell["evolution:remainder"]
        arrays[f"{field_name}:C_total"] = cell["evolution:total"]
        arrays[f"{field_name}:C_directional"] = cell["evolution:total:directional"]
        arrays[f"{field_name}:C_fit_values"] = values
        arrays[f"{field_name}:C_fit_gradients"] = gradients
        arrays[f"{field_name}:U_jump"] = jump
        arrays[f"{field_name}:U_jump_directional"] = jump_directional
        arrays[f"{field_name}:physical_wall"] = wall_correction
        arrays[f"{field_name}:q3_exact"] = q3_exact
        arrays[f"{field_name}:q5_exact"] = q5_exact
        arrays[f"{field_name}:q7_exact"] = q7_exact
    report["timing"] = {"seconds": time.monotonic() - started, "peak_gib": p._peak_gib()}
    report["provenance"] = {
        "interface_version": BOUNDARY_FUNCTIONAL_RECONSTRUCTION_VERSION,
        "driver_sha256": _sha(Path(__file__)),
        "package_sha256": _sha(REPO / "src/drbx/geometry/fci_boundary_functional_reconstruction.py"),
        "parent_npz_sha256": _sha(PARENT / f"N{n}.npz"),
        "corrected_npz_sha256": _sha(CORRECTED / "analysis" / f"N{n}.npz"),
        "scope": "bounded four wall owners plus two ordinary controls; no global-order or production claim",
    }
    _write_npz(output / f"N{n}.npz", arrays)
    _write_json(output / f"N{n}.json", report)
    return report


def summarize(output: Path) -> dict[str, Any]:
    cases = [json.loads((output / f"N{n}.json").read_text()) for n in (32, 48, 64)]
    summary = {
        "schema": f"{SCHEMA}.summary", "resolutions": [32, 48, 64],
        "fields": {
            field: {
                variant: {
                    region: [case["fields"][field]["L2"][variant][region] for case in cases]
                    for region in ("wall", "control")
                }
                for variant in ("A_C", "A_U", "B_C", "B_U", "C_C", "C_U")
            }
            for field in p.FIELD_NAMES
        },
        "validation": {
            "physical_wall_correction_max": max(
                case["fields"][field]["physical_wall_characteristic_correction_max"]
                for case in cases for field in p.FIELD_NAMES
            ),
            "M_plus_R_closure_max": max(
                case["fields"][field]["M_plus_R_closure_max"]
                for case in cases for field in p.FIELD_NAMES
            ),
            "boundary_constraint_residual_max": max(
                case["fields"][field]["boundary_moments"]["constraint_residual_max"]
                for case in cases for field in p.FIELD_NAMES
            ),
            "ordinary_control_change_max": max(
                case["fields"][field]["control_change_C_C_max"]
                for case in cases for field in p.FIELD_NAMES
            ),
        },
        "decision": "bounded comparison only; use C as the candidate for a separately authorized global static qualification if it improves the complete wall-owner C/U errors without violating the recorded checks",
    }
    _write_json(output / "summary.json", summary)
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
