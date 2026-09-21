#!/usr/bin/env python3
"""Bounded actual-HSX common-face oracle for the P02 Poisson bracket.

This diagnostic intentionally leaves the production geometry, incidence, RLP
prolongation/restriction, and characteristic selector unchanged.  It samples a
small deterministic set of owner control volumes, replaces face operands by
the continuous MMS values on those *same* coordinate faces, and completes the
residual over every raw member of each selected owner.  The compact RLP
subfaces used to construct transition left/right states are audited separately.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Iterable, Mapping
from unittest.mock import patch

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import simulate_hsx_mms as mms  # noqa: E402
from hsx_mms_continuum_reference import (  # noqa: E402
    build_continuum_reference_from_artifact,
    build_continuum_reference_from_sidecar,
)
from drbx.geometry.fci_aggregate_reconstruction import (  # noqa: E402
    AggregateGeometry,
    build_aggregate_evaluation,
)
from drbx.native.fci_operators import (  # noqa: E402
    _compatible_characteristic_regular_flux,
    _compatible_flux_generator,
    _third_order_scalar_face_states_from_halo,
    build_local_control_volume_direct_face_states,
)

import audit_hsx_poisson_vorticity as base  # noqa: E402


SCHEMA = "hsx-poisson-common-face-oracle-v2"
QUADRATURE_ORDERS = (1, 2, 4)
REFERENCE_STEPS = (1.0e-4, 2.0e-4, 4.0e-4)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rms(error: np.ndarray, weight: np.ndarray | None = None) -> float:
    values = np.asarray(error, dtype=np.float64)
    if not values.size:
        return float("nan")
    if weight is None:
        return float(np.sqrt(np.mean(values**2)))
    weights = np.asarray(weight, dtype=np.float64)
    return float(np.sqrt(np.sum(weights * values**2) / np.sum(weights)))


def _relative_rms(actual: np.ndarray, exact: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    return float(
        np.sqrt(
            np.sum((actual - exact) ** 2)
            / max(np.sum(exact**2), np.finfo(float).tiny)
        )
    )


def _select_from_mask(mask: np.ndarray, count: int) -> list[tuple[int, int, int]]:
    """Choose reproducible interior-looking entries without random sampling."""

    indices = np.argwhere(np.asarray(mask, dtype=bool))
    if not len(indices):
        return []
    shape = np.asarray(mask.shape, dtype=np.float64)
    targets = ((0.37, 0.31, 0.43), (0.61, 0.67, 0.57), (0.47, 0.79, 0.23))
    chosen: list[tuple[int, int, int]] = []
    remaining = indices.copy()
    for fractions in targets[: max(1, count)]:
        target = np.asarray(fractions) * np.maximum(shape - 1.0, 1.0)
        distance = np.sum(((remaining - target) / np.maximum(shape, 1.0)) ** 2, axis=1)
        item = tuple(int(value) for value in remaining[int(np.argmin(distance))])
        chosen.append(item)
        remaining = remaining[np.any(remaining != np.asarray(item), axis=1)]
        if not len(remaining) or len(chosen) >= count:
            break
    return chosen


def _selected_owners(host: Any, masks: Mapping[str, np.ndarray], count: int) -> dict[str, list[tuple[int, int, int]]]:
    active = np.asarray(host.topology.is_active_owner, dtype=bool)
    selected: dict[str, list[tuple[int, int, int]]] = {}
    for radial_index in range(min(4, active.shape[0])):
        ring = np.zeros_like(active)
        ring[radial_index] = active[radial_index]
        selected[f"radial_ring_{radial_index}"] = _select_from_mask(ring, 1)
    radial = np.indices(active.shape)[0]
    non_axis_transition = (
        np.asarray(masks["rlp_transition_rings"], dtype=bool)
        & active
        & (radial >= 4)
        & ~np.asarray(masks["physical_wall"], dtype=bool)
    )
    ordinary = (
        np.asarray(masks["ordinary_bulk"], dtype=bool)
        & active
        & ~np.asarray(masks["physical_wall"], dtype=bool)
    )
    selected["non_axis_transition"] = _select_from_mask(non_axis_transition, count)
    selected["ordinary_control"] = _select_from_mask(ordinary, count)
    return selected


def _face_key(axis: int, cell: tuple[int, int, int], upper: bool) -> tuple[int, int, int, int]:
    index = list(cell)
    if upper:
        index[axis] += 1
    return (axis, *index)


def _support(
    cells: Any,
    selected: Mapping[str, Iterable[tuple[int, int, int]]],
) -> tuple[dict[tuple[int, int, int], str], dict[tuple[int, int, int, int], set[str]]]:
    owner_i = np.asarray(cells.owner_i, dtype=np.int64)
    owner_j = np.asarray(cells.owner_j, dtype=np.int64)
    owner_k = np.asarray(cells.owner_k, dtype=np.int64)
    raw_labels: dict[tuple[int, int, int], str] = {}
    face_labels: dict[tuple[int, int, int, int], set[str]] = {}
    for label, owners in selected.items():
        for owner in owners:
            members = np.argwhere(
                (owner_i == owner[0]) & (owner_j == owner[1]) & (owner_k == owner[2])
            )
            for raw_array in members:
                raw = tuple(int(value) for value in raw_array)
                raw_labels[raw] = label
                for axis in range(3):
                    for upper in (False, True):
                        face_labels.setdefault(_face_key(axis, raw, upper), set()).add(label)
    return raw_labels, face_labels


def _owned_axes(geometry: Any) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    grids = (geometry.grid.x, geometry.grid.y, geometry.grid.z)
    centers = tuple(np.asarray(grid.centers_owned, dtype=np.float64) for grid in grids)
    faces = tuple(np.asarray(grid.faces_owned, dtype=np.float64) for grid in grids)
    return centers, faces


def _face_points_weights(
    geometry: Any,
    key: tuple[int, int, int, int],
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    axis, i, j, k = key
    cell_index = (i, j, k)
    centers, faces = _owned_axes(geometry)
    nodes, weights_1d = np.polynomial.legendre.leggauss(int(order))
    coordinates: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for component in range(3):
        if component == axis:
            coordinates.append(np.asarray([faces[component][cell_index[component]]]))
            weights.append(np.asarray([1.0]))
        else:
            lo = faces[component][cell_index[component]]
            hi = faces[component][cell_index[component] + 1]
            coordinates.append(0.5 * (lo + hi) + 0.5 * (hi - lo) * nodes)
            weights.append(0.5 * (hi - lo) * weights_1d)
    mesh = np.meshgrid(*coordinates, indexing="ij")
    weight_mesh = np.meshgrid(*weights, indexing="ij")
    points = np.stack(mesh, axis=-1).reshape((-1, 3))
    quadrature_weight = np.prod(np.stack(weight_mesh, axis=-1), axis=-1).ravel()
    return points, quadrature_weight


def _cell_points_weights(
    geometry: Any,
    cell: tuple[int, int, int],
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Tensor Gauss points and logical-volume weights in one raw cell."""

    _, faces = _owned_axes(geometry)
    nodes, weights_1d = np.polynomial.legendre.leggauss(int(order))
    coordinates = []
    weights = []
    for axis, index in enumerate(cell):
        lo = faces[axis][index]
        hi = faces[axis][index + 1]
        coordinates.append(0.5 * (lo + hi) + 0.5 * (hi - lo) * nodes)
        weights.append(0.5 * (hi - lo) * weights_1d)
    points = np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1).reshape((-1, 3))
    weight = np.prod(
        np.stack(np.meshgrid(*weights, indexing="ij"), axis=-1), axis=-1
    ).ravel()
    return points, weight


def _physical_cell_average(
    reference: Any,
    geometry: Any,
    cell: tuple[int, int, int],
    order: int,
    value_function,
) -> tuple[float, float]:
    points, logical_weight = _cell_points_weights(geometry, cell, order)
    prepared = reference.prepare(points)
    physical_weight = logical_weight * np.abs(np.asarray(prepared.J, dtype=np.float64))
    value = np.asarray(value_function(points, prepared), dtype=np.float64)
    volume = float(np.sum(physical_weight))
    return float(np.sum(physical_weight * value) / volume), volume


def _regular_scalar(points: np.ndarray, eta_period: float) -> np.ndarray:
    """Smooth m=2 regular-chart scalar that vanishes at the outer radial edge."""

    q = np.asarray(points, dtype=np.float64)
    u, theta, eta = q[:, 0], q[:, 1], q[:, 2]
    envelope = u**2 * np.maximum(1.0 - u**2, 0.0) ** 4
    phase = 2.0 * theta - 2.0 * np.pi * eta / float(eta_period) + 0.23
    return 0.08 * envelope * np.cos(phase)


def _regular_chart_to_logical(points: np.ndarray) -> np.ndarray:
    """Convert stored compact-face ``(x, y, eta)`` points to ``(u, theta, eta)``."""

    regular = np.asarray(points, dtype=np.float64)
    return np.stack(
        (
            np.hypot(regular[..., 0], regular[..., 1]),
            np.mod(np.arctan2(regular[..., 1], regular[..., 0]), 2.0 * np.pi),
            regular[..., 2],
        ),
        axis=-1,
    )


def _project_regular_scalar(projector: Any) -> np.ndarray:
    flat = np.empty(int(np.prod(projector.shape)), dtype=np.float64)
    for first, last, points, _prepared, weighted in projector.chunks:
        values = _regular_scalar(points, projector.reference.eta_period).reshape(
            (last - first, projector.nq)
        )
        flat[first:last] = np.sum(weighted * values, axis=1) / np.maximum(
            np.sum(weighted, axis=1), 1.0e-30
        )
    return flat.reshape(projector.shape)


def _aggregate_geometry(owner_geometry: Any) -> AggregateGeometry:
    """Expose the production RLP owner moments to the tested P02 evaluator."""

    return AggregateGeometry.from_host_payload(
        {
            "topology_is_active_owner": owner_geometry.topology.is_active_owner,
            "host_aggregate_chart_volume": owner_geometry.aggregate_chart_volume,
            "host_aggregate_chart_centroid": owner_geometry.aggregate_chart_centroid,
            "host_aggregate_chart_second_moment": (
                owner_geometry.aggregate_chart_second_moment
            ),
            "host_aggregate_chart_third_moment": (
                owner_geometry.aggregate_chart_third_moment
            ),
        }
    )


def _upwind_owner(
    model: Any,
    key: tuple[int, int, int, int],
    velocity: float,
) -> tuple[int, int, int]:
    """Return the production owner selected by one shared-face velocity."""

    axis, i, j, k = key
    raw = [i, j, k]
    if velocity >= 0.0:
        raw[axis] -= 1
    shape = model.geometry.owned_shape
    # Selected controls exclude physical coordinate boundaries.  The collapsed
    # radial axis never consumes an advected state because its velocity is zero.
    raw[axis] = min(max(raw[axis], 0), shape[axis] - 1)
    cells = model.control_volume_geometry.cells
    owner = (
        int(np.asarray(cells.owner_i)[tuple(raw)]),
        int(np.asarray(cells.owner_j)[tuple(raw)]),
        int(np.asarray(cells.owner_k)[tuple(raw)]),
    )
    return owner


def _is_collapsed_axis_face(
    model: Any, key: tuple[int, int, int, int]
) -> bool:
    """Return whether ``key`` is the topologically collapsed lower-u face."""

    return bool(key[0] == 0 and key[1] == 0 and model.axis_regular_axes[0])


def _aggregate_candidate_evaluation(
    owner_geometry: Any,
    model: Any,
    face_keys: list[tuple[int, int, int, int]],
    velocities: Mapping[tuple[int, int, int, int], float],
) -> tuple[AggregateGeometry, Any, list[tuple[int, int, int]], np.ndarray]:
    """Build degree-two owner-average-to-face rows for selected HSX faces."""

    aggregate = _aggregate_geometry(owner_geometry)
    points_xy = []
    eta_indices = []
    owners = []
    for key in face_keys:
        points, _ = _face_points_weights(model.geometry, key, 1)
        point = points[0]
        owner = _upwind_owner(model, key, velocities[key])
        owners.append(owner)
        points_xy.append(
            (float(point[0] * np.cos(point[1])), float(point[0] * np.sin(point[1])))
        )
        eta_indices.append(owner[2])
    evaluation = build_aggregate_evaluation(
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
    return aggregate, evaluation, owners, np.asarray(points_xy, dtype=np.float64)


def _aggregate_values(
    aggregate: AggregateGeometry,
    evaluation: Any,
    owner_values: np.ndarray,
) -> np.ndarray:
    compact = np.asarray(owner_values).reshape(-1)[aggregate.owner_flat_ids]
    return np.asarray(evaluation.evaluate(compact), dtype=np.float64)


def _evaluation_diagnostics(evaluation: Any) -> dict[str, Any]:
    diagnostics = evaluation.diagnostics
    donors = np.asarray(diagnostics.donor_counts, dtype=np.int64)
    return {
        "degree": int(diagnostics.degree),
        "basis_size": int(diagnostics.basis_size),
        "row_count": int(donors.size),
        "minimum_rank": int(np.min(diagnostics.rank)),
        "maximum_condition_number": float(np.max(diagnostics.condition)),
        "maximum_reproduction_residual": float(
            np.max(diagnostics.reproduction_residual)
        ),
        "maximum_l1_norm": float(np.max(diagnostics.l1_norm)),
        "initial_donor_count": int(diagnostics.initial_donor_count),
        "maximum_donor_count_used": int(np.max(donors)),
        "fallback_row_count": int(
            np.count_nonzero(donors > int(diagnostics.initial_donor_count))
        ),
        "donor_histogram": {
            str(int(key)): int(value)
            for key, value in zip(*np.unique(donors, return_counts=True))
        },
    }


def _polynomial_reproduction(
    aggregate: AggregateGeometry,
    evaluation: Any,
    points_xy: np.ndarray,
) -> dict[str, float]:
    """Check degree-two point reproduction from actual HSX owner moments."""

    centroid = aggregate.centroid_xy
    second = aggregate.central_second_xy
    owner_basis = (
        np.ones(aggregate.n_owner),
        centroid[:, 0],
        centroid[:, 1],
        centroid[:, 0] ** 2 + second[:, 0, 0],
        centroid[:, 0] * centroid[:, 1] + second[:, 0, 1],
        centroid[:, 1] ** 2 + second[:, 1, 1],
    )
    x, y = points_xy[:, 0], points_xy[:, 1]
    exact_basis = (np.ones_like(x), x, y, x**2, x * y, y**2)
    names = ("constant", "x", "y", "x2", "xy", "y2")
    return {
        name: float(np.max(np.abs(evaluation.evaluate(owner) - exact)))
        for name, owner, exact in zip(names, owner_basis, exact_basis)
    }


def _candidate_contract_summary(
    model: Any,
    face_labels: Mapping[tuple[int, int, int, int], Any],
    selected: Mapping[str, Iterable[tuple[int, int, int]]],
    production_velocities: Mapping[tuple[int, int, int, int], float],
    candidate_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the supported RLP face functional before using its values."""

    aggregate = candidate_payload["aggregate"]
    production_evaluation = candidate_payload["production_evaluation"]
    reference_evaluation = candidate_payload["reference_evaluation"]
    conditioning = {
        "production_velocity_upwind": _evaluation_diagnostics(
            production_evaluation
        ),
        "reference_velocity_upwind": _evaluation_diagnostics(
            reference_evaluation
        ),
    }
    keys = sorted(face_labels)
    cells = model.control_volume_geometry.cells
    constant_owner = np.where(
        np.asarray(cells.is_active_owner, dtype=bool), 1.0, 0.0
    )
    constant_values = _aggregate_values(
        aggregate, production_evaluation, constant_owner
    )
    constant_by_key = dict(zip(keys, constant_values))
    generator_integrals = {}
    weighted_integrals = {}
    for key in keys:
        _points, weight = _face_points_weights(model.geometry, key, 1)
        generator_integrals[key] = production_velocities[key] * float(weight[0])
        weighted_integrals[key] = generator_integrals[key] * constant_by_key[key]
    owner_i = np.asarray(cells.owner_i, dtype=np.int64)
    owner_j = np.asarray(cells.owner_j, dtype=np.int64)
    owner_k = np.asarray(cells.owner_k, dtype=np.int64)
    raw_volume = np.asarray(cells.raw_volume, dtype=np.float64)
    aggregate_volume = np.asarray(cells.aggregate_volume, dtype=np.float64)
    constant_center = np.ones_like(raw_volume)
    constant_residual = []
    for owners in selected.values():
        for owner in owners:
            members = [
                tuple(int(value) for value in raw)
                for raw in np.argwhere(
                    (owner_i == owner[0])
                    & (owner_j == owner[1])
                    & (owner_k == owner[2])
                )
            ]
            value = _assemble_owner_residual(
                owner,
                members,
                raw_volume,
                aggregate_volume,
                constant_center,
                generator_integrals,
                weighted_integrals,
            )
            constant_residual.append(abs(value / float(model.parameters.rho_star)))
    production_reproduction = _polynomial_reproduction(
        aggregate,
        production_evaluation,
        candidate_payload["production_points_xy"],
    )
    reference_reproduction = _polynomial_reproduction(
        aggregate,
        reference_evaluation,
        candidate_payload["reference_points_xy"],
    )
    reproduction_max = max(
        max(production_reproduction.values()),
        max(reference_reproduction.values()),
    )
    minimum_rank = min(
        conditioning["production_velocity_upwind"]["minimum_rank"],
        conditioning["reference_velocity_upwind"]["minimum_rank"],
    )
    contracts = {
        "shared_face_evaluation": True,
        "state_or_restart_layout_changed": False,
        "constant_face_value_max_abs_error": float(
            np.max(np.abs(constant_values - 1.0))
        ),
        "constant_completed_residual_max_abs": float(np.max(constant_residual)),
        "polynomial_reproduction_max_abs_error": float(reproduction_max),
        "polynomial_reproduction_by_upwind_selection": {
            "production_velocity": production_reproduction,
            "reference_velocity": reference_reproduction,
        },
        "minimum_rank": int(minimum_rank),
        "embedded_control_volume_reconstruction_max_rows": int(
            model.control_volume_geometry.reconstruction.max_rows
        ),
        "embedded_wrapper_was_appropriate": False,
        "passed": bool(
            np.max(np.abs(constant_values - 1.0)) < 1.0e-10
            and np.max(constant_residual) < 1.0e-9
            and reproduction_max < 1.0e-8
            and minimum_rank >= 6
        ),
        "note": (
            "The candidate uses the tested P02 planar RLP aggregate-moment "
            "evaluator on actual HSX owner moments. The embedded-control-volume "
            "wrapper has zero reconstruction rows for production RLP geometry "
            "and is therefore not used. No production arrays or state/restart "
            "layout are changed."
        ),
    }
    return conditioning, contracts


def _exact_samples(reference: Any, points: np.ndarray, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact phi gradient, omega, and A=(b/B)_cov at points."""

    points = np.asarray(points, dtype=np.float64)
    raw = reference._fields_raw(points, float(time))
    grad_phi = np.stack(raw["phi"][1:4], axis=-1)
    prepared = reference.prepare(points)
    omega = np.asarray(prepared.mms_omega, dtype=np.float64)
    one_form = np.asarray(prepared.bcov, dtype=np.float64) / np.maximum(
        np.asarray(prepared.B, dtype=np.float64)[..., None], 1.0e-30
    )
    return grad_phi, omega, one_form


def _normal_velocity(axis: int, one_form: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    return np.cross(one_form, gradient)[..., int(axis)]


def _exact_face_integral(
    reference: Any,
    geometry: Any,
    key: tuple[int, int, int, int],
    time: float,
    order: int,
) -> tuple[float, float, float, float, float]:
    points, weight = _face_points_weights(geometry, key, order)
    gradient, omega, one_form = _exact_samples(reference, points, time)
    velocity = _normal_velocity(key[0], one_form, gradient)
    return (
        float(np.sum(weight * velocity)),
        float(np.sum(weight * velocity * omega)),
        float(np.sum(weight)),
        float(np.sum(weight * omega) / np.sum(weight)),
        float(np.sqrt(np.sum(weight * np.sum(gradient**2, axis=-1)) / np.sum(weight))),
    )


def _exact_regular_face_integral(
    reference: Any,
    geometry: Any,
    key: tuple[int, int, int, int],
    time: float,
    order: int,
) -> float:
    points, weight = _face_points_weights(geometry, key, order)
    gradient, _omega, one_form = _exact_samples(reference, points, time)
    velocity = _normal_velocity(key[0], one_form, gradient)
    scalar = _regular_scalar(points, reference.eta_period)
    return float(np.sum(weight * velocity * scalar))


def _face_array_value(arrays: tuple[np.ndarray, ...], key: tuple[int, int, int, int]) -> float:
    axis, i, j, k = key
    return float(arrays[axis][i, j, k])


def _assemble_owner_residual(
    owner: tuple[int, int, int],
    raw_cells: Iterable[tuple[int, int, int]],
    raw_volume: np.ndarray,
    aggregate_volume: np.ndarray,
    center: np.ndarray,
    generator_integral: Mapping[tuple[int, int, int, int], float],
    weighted_integral: Mapping[tuple[int, int, int, int], float],
) -> float:
    numerator = 0.0
    for raw in raw_cells:
        generator_incidence = 0.0
        weighted_incidence = 0.0
        for axis in range(3):
            lower = _face_key(axis, raw, False)
            upper = _face_key(axis, raw, True)
            generator_incidence += generator_integral[upper] - generator_integral[lower]
            weighted_incidence += weighted_integral[upper] - weighted_integral[lower]
        numerator += weighted_incidence - float(center[raw]) * generator_incidence
    return numerator / float(aggregate_volume[owner])


def _face_error_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"face_count": 0}
    production_velocity = np.asarray([row["production_velocity"] for row in records])
    exact_velocity = np.asarray([row["exact_midpoint_velocity"] for row in records])
    production_argument = np.asarray(
        [row["production_argument_for_production_velocity"] for row in records]
    )
    exact_argument = np.asarray([row["exact_midpoint_argument"] for row in records])
    production_weighted = production_velocity * production_argument
    exact_weighted = exact_velocity * exact_argument
    return {
        "face_count": len(records),
        "generator_velocity_absolute_rms": _rms(production_velocity - exact_velocity),
        "generator_velocity_relative_rms": _relative_rms(production_velocity, exact_velocity),
        "advected_value_absolute_rms": _rms(production_argument - exact_argument),
        "advected_value_relative_rms": _relative_rms(production_argument, exact_argument),
        "weighted_flux_absolute_rms": _rms(production_weighted - exact_weighted),
        "weighted_flux_relative_rms": _relative_rms(production_weighted, exact_weighted),
    }


def _residual_statistics(
    records: list[dict[str, Any]],
    value_key: str,
    reference_key: str,
) -> dict[str, Any]:
    """Physical-owner-volume statistics over the bounded selected sample."""

    result: dict[str, Any] = {}
    total_sse = 0.0
    for row in records:
        difference = float(row[value_key]) - float(row[reference_key])
        total_sse += float(row["aggregate_volume"]) * difference**2
    for label in sorted({row["label"] for row in records}):
        selected = [row for row in records if row["label"] == label]
        weight = np.asarray([row["aggregate_volume"] for row in selected], dtype=np.float64)
        actual = np.asarray([row[value_key] for row in selected], dtype=np.float64)
        exact = np.asarray([row[reference_key] for row in selected], dtype=np.float64)
        sse = float(np.sum(weight * (actual - exact) ** 2))
        reference_squared = float(np.sum(weight * exact**2))
        result[label] = {
            "owner_count": len(selected),
            "physical_volume": float(np.sum(weight)),
            "absolute_l2": _rms(actual - exact, weight),
            "reference_rms": _rms(exact, weight),
            "relative_l2": float(
                np.sqrt(sse / max(reference_squared, np.finfo(float).tiny))
            ),
            "squared_error": sse,
            "selected_sample_squared_error_fraction": (
                sse / total_sse if total_sse > 0.0 else None
            ),
        }
    return result


def _condition_summary(
    polynomial: Any,
    owners: Iterable[tuple[int, int, int]],
) -> dict[str, Any]:
    valid = np.asarray(polynomial.valid, dtype=bool)
    condition = np.asarray(polynomial.condition_number, dtype=np.float64)
    order = np.asarray(polynomial.polynomial_order, dtype=np.int64)
    points = list(owners)
    values = np.asarray([condition[index] for index in points], dtype=np.float64)
    return {
        "owner_count": len(points),
        "valid_count": int(sum(bool(valid[index]) for index in points)),
        "polynomial_orders": [int(order[index]) for index in points],
        "condition_numbers": values,
        "maximum_finite_condition_number": (
            float(np.max(values[np.isfinite(values)]))
            if np.any(np.isfinite(values)) else None
        ),
    }


def _compact_transition_audit(
    model: Any,
    operands: Any,
    reference: Any,
    time: float,
    selected_faces: set[tuple[int, int, int, int]],
) -> dict[str, Any]:
    faces = model.control_volume_geometry.irregular_faces
    direct = build_local_control_volume_direct_face_states(
        operands.omega_owner,
        model.geometry,
        model.domain,
        model.control_volume_geometry,
        halo_exchange=model.halo_exchange,
        topology_filler=model.topology_filler,
    )
    active = np.asarray(faces.quadrature_active, dtype=bool)
    logical_axis = np.asarray(faces.logical_axis, dtype=np.int64)
    fi = np.asarray(faces.logical_face_i, dtype=np.int64)
    fj = np.asarray(faces.logical_face_j, dtype=np.int64)
    fk = np.asarray(faces.logical_face_k, dtype=np.int64)
    row_mask = np.asarray(faces.active, dtype=bool) & (logical_axis == 0)
    row_mask &= np.asarray(
        [(0, int(i), int(j), int(k)) in selected_faces for i, j, k in zip(fi, fj, fk)],
        dtype=bool,
    )
    rows = np.flatnonzero(row_mask)
    if not len(rows):
        return {"row_count": 0, "quadrature_count": 0}
    # Compact irregular-face quadrature is stored in the axis-regular
    # (x,y,eta) chart used by the production polynomial evaluator.  The
    # continuum reference consumes logical (u,theta,eta), so convert exactly;
    # passing x/y through as u/theta can place x<0 outside the metric domain.
    regular_points = np.asarray(faces.quadrature_points, dtype=np.float64)[rows]
    points = _regular_chart_to_logical(regular_points)
    qactive = active[rows]
    exact = np.zeros(qactive.shape, dtype=np.float64)
    exact[qactive] = _exact_samples(reference, points[qactive], time)[1]
    measure = np.linalg.norm(
        np.asarray(faces.area_covector_weight, dtype=np.float64)[rows], axis=-1
    )
    valid = qactive & np.asarray(direct.valid, dtype=bool)[rows]
    minus = np.asarray(direct.minus, dtype=np.float64)[rows]
    plus = np.asarray(direct.plus, dtype=np.float64)[rows]
    return {
        "row_count": int(len(rows)),
        "quadrature_count": int(np.count_nonzero(qactive)),
        "all_production_states_valid": bool(np.all(valid | ~qactive)),
        "minus_value_physical_area_rms": _rms((minus - exact)[valid], measure[valid]),
        "plus_value_physical_area_rms": _rms((plus - exact)[valid], measure[valid]),
        "minus_value_relative_rms": _relative_rms(minus[valid], exact[valid]),
        "plus_value_relative_rms": _relative_rms(plus[valid], exact[valid]),
        "note": (
            "These compact quadrature values only construct transition-face "
            "coordinate left/right states; compact-face divergence is not used "
            "by the projected-fine production bracket."
        ),
    }


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    resolution = int(args.resolution)
    candidate_enabled = not bool(
        getattr(args, "disable_owner_polynomial_candidate", False)
    )
    if bool(getattr(args, "candidate_contracts_only", False)) and not candidate_enabled:
        raise ValueError(
            "--candidate-contracts-only cannot be combined with "
            "--disable-owner-polynomial-candidate"
        )
    artifact_path, artifact = base._load_artifact(args.geometry, resolution)
    _, reference_artifact = base._load_artifact(args.geometry, int(args.reference_resolution))
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
        raise RuntimeError("common-face audit requires a single-device model")
    projector = mms._QuadratureProjector(reference, artifact.global_geometry, artifact.owner_geometry)
    raw_state, _ = base._poisson_reference_fields(
        projector, float(args.time), rho_star=float(model.parameters.rho_star)
    )
    owner_state = mms._owner_project(raw_state, artifact.owner_geometry)
    operands = base._prepare_operands(model, owner_state, raw_state)
    omega_owner_values = np.asarray(model._owner_field(owner_state.vorticity))

    regular_raw = _project_regular_scalar(projector)
    regular_raw_state = raw_state.replace(vorticity=regular_raw)
    regular_owner_state = mms._owner_project(
        regular_raw_state, artifact.owner_geometry
    )
    regular_operands = base._prepare_operands(
        model, regular_owner_state, regular_raw_state
    )
    regular_owner_values = np.asarray(
        model._owner_field(regular_owner_state.vorticity)
    )
    masks = base._region_masks(artifact.global_geometry, artifact.owner_geometry)
    selected = _selected_owners(artifact.owner_geometry, masks, int(args.samples_per_control))
    raw_labels, face_labels = _support(model.control_volume_geometry.cells, selected)

    generator = _compatible_flux_generator(
        operands.phi_stencil,
        model.geometry,
        domain=model.domain,
        axis_regular_axes=model.axis_regular_axes,
        b_floor=1.0e-30,
    )
    generator_arrays = tuple(np.asarray(value) for value in (generator.x, generator.y, generator.z))
    production_sides = tuple(
        tuple(
            np.asarray(getattr(side, "xyz"[axis]))
            for axis in range(3)
        )
        for side in operands.omega_direct_states
    )
    raw_left, raw_right, _ = _third_order_scalar_face_states_from_halo(
        operands.raw_omega_halo,
        model.geometry,
        boundary_trace=operands.raw_omega_trace,
        axis_regular_axes=model.axis_regular_axes,
        positivity_floor=None,
    )
    raw_sides = tuple(
        tuple(np.asarray(getattr(side, "xyz"[axis])) for axis in range(3))
        for side in (raw_left, raw_right)
    )
    regular_sides = tuple(
        tuple(
            np.asarray(getattr(side, "xyz"[axis]))
            for axis in range(3)
        )
        for side in regular_operands.omega_direct_states
    )

    candidate_payload = None
    if candidate_enabled:
        candidate_keys = sorted(face_labels)
        production_velocity_by_key = {
            key: _face_array_value(generator_arrays, key) for key in candidate_keys
        }
        reference_velocity_by_key = {}
        for key in candidate_keys:
            collapsed = _is_collapsed_axis_face(model, key)
            if collapsed:
                reference_velocity_by_key[key] = 0.0
            else:
                points, _ = _face_points_weights(model.geometry, key, 1)
                gradient, _omega, one_form = _exact_samples(
                    reference, points, float(args.time)
                )
                reference_velocity_by_key[key] = float(
                    _normal_velocity(key[0], one_form, gradient)[0]
                )
        (
            aggregate,
            production_evaluation,
            production_owners,
            production_points_xy,
        ) = _aggregate_candidate_evaluation(
            artifact.owner_geometry,
            model,
            candidate_keys,
            production_velocity_by_key,
        )
        (
            _reference_aggregate,
            reference_evaluation,
            reference_owners,
            reference_points_xy,
        ) = _aggregate_candidate_evaluation(
            artifact.owner_geometry,
            model,
            candidate_keys,
            reference_velocity_by_key,
        )
        omega_candidate_production = dict(
            zip(
                candidate_keys,
                _aggregate_values(
                    aggregate, production_evaluation, omega_owner_values
                ),
            )
        )
        omega_candidate_reference = dict(
            zip(
                candidate_keys,
                _aggregate_values(
                    aggregate, reference_evaluation, omega_owner_values
                ),
            )
        )
        regular_candidate_production = dict(
            zip(
                candidate_keys,
                _aggregate_values(
                    aggregate, production_evaluation, regular_owner_values
                ),
            )
        )
        regular_candidate_reference = dict(
            zip(
                candidate_keys,
                _aggregate_values(
                    aggregate, reference_evaluation, regular_owner_values
                ),
            )
        )
        candidate_payload = {
            "aggregate": aggregate,
            "production_evaluation": production_evaluation,
            "reference_evaluation": reference_evaluation,
            "production_owners": dict(zip(candidate_keys, production_owners)),
            "reference_owners": dict(zip(candidate_keys, reference_owners)),
            "production_points_xy": production_points_xy,
            "reference_points_xy": reference_points_xy,
        }
        polynomial_conditioning, candidate_contracts = _candidate_contract_summary(
            model,
            face_labels,
            selected,
            production_velocity_by_key,
            candidate_payload,
        )
        if bool(getattr(args, "candidate_contracts_only", False)):
            payload = {
                "schema": SCHEMA,
                "resolution": resolution,
                "reference_resolution": int(args.reference_resolution),
                "time": float(args.time),
                "geometry_artifact": str(artifact_path),
                "scope": {
                    "selected_owner_count": sum(
                        len(value) for value in selected.values()
                    ),
                    "selected_coordinate_face_count": len(face_labels),
                    "selection": selected,
                    "certification": (
                        "candidate contract gate only; no residual comparison"
                    ),
                },
                "diagnostic_owner_polynomial_candidate": {
                    "enabled": True,
                    "path": (
                        "stored owner average -> tested P02 planar RLP "
                        "aggregate-moment evaluation -> one shared upwind "
                        "coordinate-face value"
                    ),
                    "conditioning": polynomial_conditioning,
                    "contracts": candidate_contracts,
                    "production_changes": [],
                },
            }
            _write_json(args.output, payload)
            return payload

    generator_integrals = {"production": {}, "reference": {}}
    weighted_integrals = {
        "velocity_production__advected_production": {},
        "velocity_production__advected_reference": {},
        "velocity_reference__advected_production": {},
        "velocity_reference__advected_reference": {},
        "velocity_production__advected_raw_input": {},
        "velocity_reference__advected_raw_input": {},
    }
    regular_weighted_integrals = {
        "velocity_production__advected_production": {},
        "velocity_production__advected_reference": {},
        "velocity_reference__advected_production": {},
        "velocity_reference__advected_reference": {},
    }
    if candidate_enabled:
        weighted_integrals.update(
            {
                "velocity_production__advected_owner_polynomial": {},
                "velocity_reference__advected_owner_polynomial": {},
            }
        )
        regular_weighted_integrals.update(
            {
                "velocity_production__advected_owner_polynomial": {},
                "velocity_reference__advected_owner_polynomial": {},
            }
        )
    oracle_generator = {order: {} for order in QUADRATURE_ORDERS}
    oracle_weighted = {order: {} for order in QUADRATURE_ORDERS}
    regular_oracle_weighted = {order: {} for order in QUADRATURE_ORDERS}
    face_records: list[dict[str, Any]] = []
    for key, labels in sorted(face_labels.items()):
        points, midpoint_weight = _face_points_weights(model.geometry, key, 1)
        area = float(midpoint_weight[0])
        production_velocity = _face_array_value(generator_arrays, key)
        # The lower radial axis face is collapsed by the production topology.
        collapsed_axis = _is_collapsed_axis_face(model, key)
        if collapsed_axis:
            exact_velocity = 0.0
            exact_argument_value = 0.0
        else:
            grad, exact_argument, one_form = _exact_samples(
                reference, points, float(args.time)
            )
            exact_velocity = float(_normal_velocity(key[0], one_form, grad)[0])
            exact_argument_value = float(exact_argument[0])

        def side_value(sides, velocity):
            side = 0 if velocity >= 0.0 else 1
            return _face_array_value(sides[side], key)

        production_argument_for_production_velocity = side_value(
            production_sides, production_velocity
        )
        production_argument_for_reference_velocity = side_value(
            production_sides, exact_velocity
        )
        raw_argument_for_production_velocity = side_value(
            raw_sides, production_velocity
        )
        raw_argument_for_reference_velocity = side_value(raw_sides, exact_velocity)
        regular_exact_argument = float(
            _regular_scalar(points, reference.eta_period)[0]
        )
        regular_production_argument_for_production_velocity = side_value(
            regular_sides, production_velocity
        )
        regular_production_argument_for_reference_velocity = side_value(
            regular_sides, exact_velocity
        )
        if not candidate_enabled:
            candidate_prod = candidate_ref = None
            candidate_prod_valid = candidate_ref_valid = False
            candidate_owner_prod = candidate_owner_ref = None
            regular_candidate_prod = regular_candidate_ref = None
            regular_candidate_prod_valid = regular_candidate_ref_valid = False
        elif collapsed_axis:
            candidate_prod = candidate_ref = 0.0
            candidate_prod_valid = candidate_ref_valid = True
            candidate_owner_prod = candidate_owner_ref = (0, 0, 0)
            regular_candidate_prod = regular_candidate_ref = 0.0
            regular_candidate_prod_valid = regular_candidate_ref_valid = True
        else:
            candidate_prod = float(omega_candidate_production[key])
            candidate_ref = float(omega_candidate_reference[key])
            candidate_prod_valid = candidate_ref_valid = True
            candidate_owner_prod = candidate_payload["production_owners"][key]
            candidate_owner_ref = candidate_payload["reference_owners"][key]
            regular_candidate_prod = float(regular_candidate_production[key])
            regular_candidate_ref = float(regular_candidate_reference[key])
            regular_candidate_prod_valid = regular_candidate_ref_valid = True
        generator_integrals["production"][key] = production_velocity * area
        generator_integrals["reference"][key] = exact_velocity * area
        face_products = {
            "velocity_production__advected_production": production_velocity
            * production_argument_for_production_velocity,
            "velocity_production__advected_reference": production_velocity
            * exact_argument_value,
            "velocity_reference__advected_production": exact_velocity
            * production_argument_for_reference_velocity,
            "velocity_reference__advected_reference": exact_velocity
            * exact_argument_value,
            "velocity_production__advected_raw_input": production_velocity
            * raw_argument_for_production_velocity,
            "velocity_reference__advected_raw_input": exact_velocity
            * raw_argument_for_reference_velocity,
        }
        if candidate_enabled:
            face_products.update(
                {
                    "velocity_production__advected_owner_polynomial": (
                        production_velocity * candidate_prod
                    ),
                    "velocity_reference__advected_owner_polynomial": (
                        exact_velocity * candidate_ref
                    ),
                }
            )
        for name, value in face_products.items():
            weighted_integrals[name][key] = value * area
        regular_products = {
            "velocity_production__advected_production": production_velocity
            * regular_production_argument_for_production_velocity,
            "velocity_production__advected_reference": production_velocity
            * regular_exact_argument,
            "velocity_reference__advected_production": exact_velocity
            * regular_production_argument_for_reference_velocity,
            "velocity_reference__advected_reference": exact_velocity
            * regular_exact_argument,
        }
        if candidate_enabled:
            regular_products.update(
                {
                    "velocity_production__advected_owner_polynomial": (
                        production_velocity * regular_candidate_prod
                    ),
                    "velocity_reference__advected_owner_polynomial": (
                        exact_velocity * regular_candidate_ref
                    ),
                }
            )
        for name, value in regular_products.items():
            regular_weighted_integrals[name][key] = value * area
        refinements = {}
        for order in QUADRATURE_ORDERS:
            if collapsed_axis:
                _, quadrature_weight = _face_points_weights(
                    model.geometry, key, order
                )
                gen, adv = 0.0, 0.0
                quadrature_area = float(np.sum(quadrature_weight))
                mean_argument = 0.0
                gradient_rms = None
                regular_adv = 0.0
            else:
                gen, adv, quadrature_area, mean_argument, gradient_rms = _exact_face_integral(
                    reference, model.geometry, key, float(args.time), order
                )
                regular_adv = _exact_regular_face_integral(
                    reference, model.geometry, key, float(args.time), order
                )
            oracle_generator[order][key] = gen
            oracle_weighted[order][key] = adv
            regular_oracle_weighted[order][key] = regular_adv
            refinements[str(order)] = {
                "generator_integral": gen,
                "weighted_integral": adv,
                "logical_face_measure": quadrature_area,
                "mean_argument": mean_argument,
                "phi_gradient_rms": gradient_rms,
            }
        face_records.append(
            {
                "key": key,
                "labels": sorted(labels),
                "production_velocity": production_velocity,
                "production_argument_for_production_velocity": (
                    production_argument_for_production_velocity
                ),
                "production_argument_for_reference_velocity": (
                    production_argument_for_reference_velocity
                ),
                "raw_input_argument_for_production_velocity": (
                    raw_argument_for_production_velocity
                ),
                "raw_input_argument_for_reference_velocity": (
                    raw_argument_for_reference_velocity
                ),
                "owner_polynomial_argument_for_production_velocity": candidate_prod,
                "owner_polynomial_argument_for_reference_velocity": candidate_ref,
                "owner_polynomial_valid": bool(
                    candidate_enabled
                    and candidate_prod_valid
                    and candidate_ref_valid
                ),
                "owner_polynomial_upwind_owner_production_velocity": (
                    candidate_owner_prod
                ),
                "owner_polynomial_upwind_owner_reference_velocity": (
                    candidate_owner_ref
                ),
                "production_generator_integral": generator_integrals["production"][key],
                "exact_midpoint_velocity": exact_velocity,
                "exact_midpoint_argument": exact_argument_value,
                "regular_scalar_exact_midpoint_argument": regular_exact_argument,
                "regular_scalar_owner_polynomial_valid": bool(
                    candidate_enabled
                    and regular_candidate_prod_valid
                    and regular_candidate_ref_valid
                ),
                "collapsed_axis_face": collapsed_axis,
                "quadrature": refinements,
            }
        )

    cells = model.control_volume_geometry.cells
    raw_volume = np.asarray(cells.raw_volume, dtype=np.float64)
    aggregate_volume = np.asarray(cells.aggregate_volume, dtype=np.float64)
    production_center = np.asarray(operands.omega_stencil.x.center, dtype=np.float64)
    exact_center = np.asarray(raw_state.vorticity, dtype=np.float64)
    regular_production_center = np.asarray(
        regular_operands.omega_stencil.x.center, dtype=np.float64
    )
    regular_exact_center = np.asarray(regular_raw, dtype=np.float64)
    production_owner = jax.jit(
        lambda: model._restrict_fine_field(
            base._operator_call(
                model,
                operands.phi_stencil,
                operands.omega_stencil,
                f_trace=operands.phi_trace,
                g_trace=operands.omega_trace,
                characteristic_scheme="scalar-third-order-upwind",
                g_halo=operands.omega_halo,
                g_direct_states=operands.omega_direct_states,
            )
        )
    )()
    production_owner = np.asarray(jax.block_until_ready(production_owner))
    owner_records = []
    rho_star = float(model.parameters.rho_star)
    owner_i = np.asarray(cells.owner_i, dtype=np.int64)
    owner_j = np.asarray(cells.owner_j, dtype=np.int64)
    owner_k = np.asarray(cells.owner_k, dtype=np.int64)
    for label, owners in selected.items():
        for owner in owners:
            raw_members = [
                tuple(int(value) for value in row)
                for row in np.argwhere(
                    (owner_i == owner[0])
                    & (owner_j == owner[1])
                    & (owner_k == owner[2])
                )
            ]
            production = _assemble_owner_residual(
                owner, raw_members, raw_volume, aggregate_volume, production_center,
                generator_integrals["production"],
                weighted_integrals[
                    "velocity_production__advected_production"
                ],
            )
            oracle = {
                str(order): _assemble_owner_residual(
                    owner, raw_members, raw_volume, aggregate_volume, exact_center,
                    oracle_generator[order], oracle_weighted[order],
                )
                for order in QUADRATURE_ORDERS
            }
            midpoint_crosses = {}
            for cross_name in (
                "velocity_production__advected_production",
                "velocity_production__advected_reference",
                "velocity_reference__advected_production",
                "velocity_reference__advected_reference",
                "velocity_production__advected_raw_input",
                "velocity_reference__advected_raw_input",
            ):
                velocity_name = (
                    "production" if cross_name.startswith("velocity_production")
                    else "reference"
                )
                for center_name, center_values in (
                    ("production", production_center),
                    ("reference", exact_center),
                ):
                    value = _assemble_owner_residual(
                        owner,
                        raw_members,
                        raw_volume,
                        aggregate_volume,
                        center_values,
                        generator_integrals[velocity_name],
                        weighted_integrals[cross_name],
                    )
                    midpoint_crosses[f"{cross_name}__center_{center_name}"] = (
                        -value / rho_star
                    )
            if candidate_enabled:
                for cross_name in (
                    "velocity_production__advected_owner_polynomial",
                    "velocity_reference__advected_owner_polynomial",
                ):
                    velocity_name = (
                        "production"
                        if cross_name.startswith("velocity_production")
                        else "reference"
                    )
                    for center_name, center_values in (
                        ("production", production_center),
                        ("reference", exact_center),
                    ):
                        value = _assemble_owner_residual(
                            owner,
                            raw_members,
                            raw_volume,
                            aggregate_volume,
                            center_values,
                            generator_integrals[velocity_name],
                            weighted_integrals[cross_name],
                        )
                        midpoint_crosses[
                            f"{cross_name}__center_{center_name}"
                        ] = -value / rho_star
            regular_crosses = {}
            for cross_name in regular_weighted_integrals:
                velocity_name = (
                    "production" if cross_name.startswith("velocity_production")
                    else "reference"
                )
                for center_name, center_values in (
                    ("production", regular_production_center),
                    ("reference", regular_exact_center),
                ):
                    value = _assemble_owner_residual(
                        owner,
                        raw_members,
                        raw_volume,
                        aggregate_volume,
                        center_values,
                        generator_integrals[velocity_name],
                        regular_weighted_integrals[cross_name],
                    )
                    regular_crosses[f"{cross_name}__center_{center_name}"] = (
                        -value / rho_star
                    )
            regular_oracle = {
                str(order): -_assemble_owner_residual(
                    owner,
                    raw_members,
                    raw_volume,
                    aggregate_volume,
                    regular_exact_center,
                    oracle_generator[order],
                    regular_oracle_weighted[order],
                ) / rho_star
                for order in QUADRATURE_ORDERS
            }
            owner_records.append(
                {
                    "label": label,
                    "owner": owner,
                    "raw_member_count": len(raw_members),
                    "aggregate_volume": float(aggregate_volume[owner]),
                    "production_effective_residual": -production / rho_star,
                    "production_operator_effective_residual": float(
                        -production_owner[owner] / rho_star
                    ),
                    "production_assembly_replay_absolute_mismatch": float(
                        abs(production - production_owner[owner]) / rho_star
                    ),
                    "oracle_effective_residual": {
                        order: -value / rho_star for order, value in oracle.items()
                    },
                    "midpoint_cross_effective_residual": midpoint_crosses,
                    "regular_scalar_midpoint_cross_effective_residual": (
                        regular_crosses
                    ),
                    "regular_scalar_oracle_effective_residual": regular_oracle,
                    "production_minus_q4_absolute": abs(production - oracle["4"]) / rho_star,
                    "q1_to_q2_absolute": abs(oracle["1"] - oracle["2"]) / rho_star,
                    "q2_to_q4_absolute": abs(oracle["2"] - oracle["4"]) / rho_star,
                }
            )

    summary_rows = []
    for row in owner_records:
        summary_rows.append(
            {
                **row,
                **row["midpoint_cross_effective_residual"],
                "reference_midpoint": row["midpoint_cross_effective_residual"][
                    "velocity_reference__advected_reference__center_reference"
                ],
                "reference_q4": row["oracle_effective_residual"]["4"],
                **{
                    f"regular__{name}": value
                    for name, value in row[
                        "regular_scalar_midpoint_cross_effective_residual"
                    ].items()
                },
                "regular_reference_midpoint": row[
                    "regular_scalar_midpoint_cross_effective_residual"
                ]["velocity_reference__advected_reference__center_reference"],
                "regular_reference_q4": row[
                    "regular_scalar_oracle_effective_residual"
                ]["4"],
            }
        )
    midpoint_cross_statistics = {
        name: _residual_statistics(summary_rows, name, "reference_midpoint")
        for name in owner_records[0]["midpoint_cross_effective_residual"]
    }
    regular_cross_statistics = {
        name: _residual_statistics(
            summary_rows, f"regular__{name}", "regular_reference_midpoint"
        )
        for name in owner_records[0][
            "regular_scalar_midpoint_cross_effective_residual"
        ]
    }

    # Matched vorticity-reference sensitivity: the same completed owner
    # residual, field, units, physical owner weights, and regional norm are
    # compared while only the metric differentiation step or face quadrature
    # changes.
    owner_members = {
        tuple(row["owner"]): [
            tuple(int(value) for value in raw)
            for raw in np.argwhere(
                (owner_i == row["owner"][0])
                & (owner_j == row["owner"][1])
                & (owner_k == row["owner"][2])
            )
        ]
        for row in owner_records
    }
    reuse_path = getattr(args, "reuse_reference_artifact", None)
    if reuse_path is not None:
        reused = json.loads(Path(reuse_path).read_text(encoding="utf-8"))
        if (
            int(reused["resolution"]) != resolution
            or int(reused["reference_resolution"]) != int(args.reference_resolution)
            or float(reused["time"]) != float(args.time)
            or reused["scope"]["selection"] != _json_value(selected)
        ):
            raise ValueError("reused reference artifact does not match this case")
        reused_sensitivity = reused["reference_sensitivity"]
        reference_sensitivity_records = reused_sensitivity["records"]
        baseline_step = float(reused_sensitivity["baseline_step"])
        sensitivity_summary = reused_sensitivity[
            "physical_owner_weighted_statistics"
        ]
        reference_sensitivity_provenance = str(Path(reuse_path))
    else:
        original_step = float(reference.finite_difference_step)
        reference_sensitivity_records = []
        for step in REFERENCE_STEPS:
            reference.finite_difference_step = float(step)
            step_generator = {2: {}, 4: {}}
            step_weighted = {2: {}, 4: {}}
            for key in face_labels:
                collapsed = _is_collapsed_axis_face(model, key)
                for order in (2, 4):
                    if collapsed:
                        gen = adv = 0.0
                    else:
                        gen, adv, *_ = _exact_face_integral(
                            reference, model.geometry, key, float(args.time), order
                        )
                    step_generator[order][key] = gen
                    step_weighted[order][key] = adv
            step_center = {}
            for order in (2, 4):
                center_values = np.array(exact_center, copy=True)
                for raw in raw_labels:
                    center_values[raw], _ = _physical_cell_average(
                        reference,
                        model.geometry,
                        raw,
                        order,
                        lambda _points, prepared: prepared.mms_omega,
                    )
                step_center[order] = center_values
            for row in owner_records:
                owner = tuple(row["owner"])
                values = {}
                for order in (2, 4):
                    action = _assemble_owner_residual(
                        owner,
                        owner_members[owner],
                        raw_volume,
                        aggregate_volume,
                        step_center[order],
                        step_generator[order],
                        step_weighted[order],
                    )
                    values[str(order)] = -action / rho_star
                reference_sensitivity_records.append(
                    {
                        "finite_difference_step": step,
                        "label": row["label"],
                        "owner": owner,
                        "aggregate_volume": row["aggregate_volume"],
                        "q2_effective_residual": values["2"],
                        "q4_effective_residual": values["4"],
                        "q2_to_q4_absolute": abs(values["2"] - values["4"]),
                    }
                )
        reference.finite_difference_step = original_step
        baseline_step = min(
            REFERENCE_STEPS, key=lambda value: abs(value - original_step)
        )
        baseline_by_owner = {
            tuple(row["owner"]): row["q4_effective_residual"]
            for row in reference_sensitivity_records
            if row["finite_difference_step"] == baseline_step
        }
        sensitivity_summary = {}
        for step in REFERENCE_STEPS:
            rows = [
                {**row, "baseline_q4": baseline_by_owner[tuple(row["owner"])]}
                for row in reference_sensitivity_records
                if row["finite_difference_step"] == step
            ]
            sensitivity_summary[str(step)] = {
                "q4_vs_baseline": _residual_statistics(
                    rows, "q4_effective_residual", "baseline_q4"
                ),
                "q2_vs_q4": _residual_statistics(
                    rows, "q2_effective_residual", "q4_effective_residual"
                ),
            }
        reference_sensitivity_provenance = "computed in this artifact"

    label_face_summaries = {}
    for label in selected:
        label_face_summaries[label] = _face_error_summary(
            [row for row in face_records if label in row["labels"]]
        )
    selected_face_keys = set(face_labels)
    compact = _compact_transition_audit(
        model, operands, reference, float(args.time), selected_face_keys
    )
    if not candidate_enabled:
        polynomial_conditioning = None
        candidate_contracts = None

    q24 = np.asarray([row["q2_to_q4_absolute"] for row in owner_records], dtype=np.float64)
    prod_q4 = np.asarray([row["production_minus_q4_absolute"] for row in owner_records], dtype=np.float64)
    replay = np.asarray(
        [row["production_assembly_replay_absolute_mismatch"] for row in owner_records],
        dtype=np.float64,
    )
    payload = {
        "schema": SCHEMA,
        "resolution": resolution,
        "reference_resolution": int(args.reference_resolution),
        "reference_source": getattr(reference, "provenance", {
            "artifact": str((args.geometry / f"{args.reference_resolution}x{args.reference_resolution}x{args.reference_resolution}").resolve())
        }),
        "time": float(args.time),
        "geometry_artifact": str(artifact_path),
        "scope": {
            "selected_owner_count": sum(len(value) for value in selected.values()),
            "selected_raw_cell_count": len(raw_labels),
            "selected_coordinate_face_count": len(face_labels),
            "selection": selected,
            "production_path": "projected-fine coordinate faces with production R restriction",
            "boundary_contract": {
                "physical_wall_model": "legacy-velocity-trace",
                "parallel_velocity_wall_bc": "neumann",
                "neumann_ghost_scheme": "physical",
                "parallel_boundary_pairing": "characteristic-sat",
                "parallel_characteristic_wall_law": "energy-absorbing",
                "coordinate_face_rule": (
                    "production physical traces; collapsed lower radial axis "
                    "flux forced to zero"
                ),
            },
            "certification": "bounded sampled mechanism diagnostic, not a full-domain qualification",
        },
        "face_errors": label_face_summaries,
        "owner_residuals": owner_records,
        "completed_residual_cross_statistics": midpoint_cross_statistics,
        "regular_scalar_control": {
            "definition": (
                "0.08*u^2*(1-u^2)^4*cos(2*theta-2*pi*eta/period+0.23)"
            ),
            "axis_regular": True,
            "outer_radial_value_and_normal_derivatives_vanish": True,
            "completed_residual_cross_statistics": regular_cross_statistics,
        },
        "reference_sensitivity": {
            "finite_difference_steps": REFERENCE_STEPS,
            "baseline_step": baseline_step,
            "records": reference_sensitivity_records,
            "physical_owner_weighted_statistics": sensitivity_summary,
            "provenance": reference_sensitivity_provenance,
            "certification": "bounded selected-owner reference qualification",
        },
        "diagnostic_owner_polynomial_candidate": {
            "enabled": candidate_enabled,
            "path": (
                "stored owner average -> tested P02 planar RLP aggregate-moment "
                "evaluation -> one shared upwind coordinate-face value"
            ),
            "conditioning": polynomial_conditioning,
            "contracts": candidate_contracts,
            "mms_vorticity_vs_q4": _residual_statistics(
                summary_rows,
                "velocity_production__advected_owner_polynomial__center_production",
                "reference_q4",
            ) if candidate_enabled and candidate_contracts["passed"] else None,
            "unchanged_baseline_vs_q4": _residual_statistics(
                summary_rows,
                "velocity_production__advected_production__center_production",
                "reference_q4",
            ),
            "regular_scalar_vs_q4": _residual_statistics(
                summary_rows,
                "regular__velocity_production__advected_owner_polynomial__center_production",
                "regular_reference_q4",
            ) if candidate_enabled and candidate_contracts["passed"] else None,
            "regular_scalar_unchanged_baseline_vs_q4": _residual_statistics(
                summary_rows,
                "regular__velocity_production__advected_production__center_production",
                "regular_reference_q4",
            ),
            "production_changes": [],
        },
        "sign_and_upwind_convention": {
            "face_velocity": "U_phi^alpha=(A cross grad(phi))^alpha",
            "positive_velocity_state": "minus/left side",
            "negative_velocity_state": "plus/right side",
            "cell_incidence": "upper face minus lower face",
            "compression_correction": (
                "subtract cell-center argument times matching generator incidence"
            ),
            "effective_rhs_sign": "-A_phi(omega)/rho_star",
        },
        "quadrature_summary": {
            "maximum_q2_to_q4_effective_residual_change": float(np.max(q24)) if q24.size else None,
            "production_minus_q4_absolute_rms": _rms(prod_q4) if prod_q4.size else None,
            "production_assembly_replay_max_abs": float(np.max(replay)) if replay.size else None,
        },
        "compact_transition_subfaces": compact,
        "face_records": face_records if bool(args.include_face_records) else None,
        "formula_changes": [],
    }
    _write_json(args.output, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--reference-resolution", type=int, default=64)
    parser.add_argument("--reference-sidecar", type=Path)
    parser.add_argument("--time", type=float, default=0.37)
    parser.add_argument("--samples-per-control", type=int, default=2)
    parser.add_argument("--include-face-records", action="store_true")
    parser.add_argument(
        "--disable-owner-polynomial-candidate",
        action="store_true",
        help="run independent midpoint/reference controls without the optional candidate",
    )
    parser.add_argument(
        "--candidate-contracts-only",
        action="store_true",
        help="stop after the cheap N32 constant and polynomial-reproduction gate",
    )
    parser.add_argument(
        "--reuse-reference-artifact",
        type=Path,
        help="reuse matched reference-sensitivity records from a prior artifact",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = run_case(args)
    result = {
        "output": str(args.output),
        "resolution": payload["resolution"],
    }
    if "quadrature_summary" in payload:
        result["quadrature_summary"] = payload["quadrature_summary"]
        result["compact_transition_subfaces"] = payload[
            "compact_transition_subfaces"
        ]
    else:
        result["candidate_contracts"] = payload[
            "diagnostic_owner_polynomial_candidate"
        ]["contracts"]
    print(json.dumps(_json_value(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
