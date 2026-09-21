#!/usr/bin/env python3
"""Bounded matched face-and-volume integration audit for the HSX bracket.

This offline research audit reuses the fixed 28-owner N48/N64 cross caches.  It
does not alter a production selector.  With ``h = b_cov/B`` and ``rho`` factored
outside the discrete assembly, the anchored A form evaluated here is

    A_h(g,f) = -1/(rho V) [ sum_faces int (h x grad g).n (f-c_f)
                            - int_cell (f-c_f) grad(g).curl(h) ].

The swapped form is ``B_h(g,f) = -A_h(f,g)`` and ``C=(A+B)/2``.  Each numerical
face or raw-cell entity owns one fixed cubic polynomial fit; values and
derivatives at all quadrature nodes come from that same fit.  Analytical and
numerical factors are cached separately so replay and reporting are evaluator
free.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import audit_hsx_bracket_integrated_reference as integrated  # noqa: E402
import audit_hsx_cubic_derivative_global as cubic  # noqa: E402
import audit_hsx_generator_factorization as factor  # noqa: E402
import audit_hsx_value_derivative_cross as cross  # noqa: E402
import audit_hsx_poisson_continuous_baseline as continuous  # noqa: E402


CACHE_SCHEMA = "drbx.hsx-matched-face-volume-cache-v1"
CASE_SCHEMA = "drbx.hsx-matched-face-volume-case-v1"
SUMMARY_SCHEMA = "drbx.hsx-matched-face-volume-summary-v1"
FIELDS = cross.TRANSPORT_FIELDS
ALL_FIELDS = cross.FIELD_NAMES
ACTIONS = ("A", "B", "C")
BASELINE_CANDIDATES = ("P_Dh_midpoint", "O2_Dh_midpoint", "C3_planar_Dh_midpoint")
ANALYTIC_CANDIDATES = (
    "analytic_midpoint_cached_geometry",
    "analytic_midpoint_exact_geometry",
    "analytic_face_q1",
    "analytic_face_q3",
    "analytic_face_q5",
    "analytic_matched_q1",
    "analytic_matched_q3",
    "analytic_matched_q5",
    "analytic_unanchored_q1",
    "analytic_unanchored_q3",
    "analytic_unanchored_q5",
)
NUMERICAL_CANDIDATES = (
    "joint_cubic_midpoint_cached_geometry",
    "joint_cubic_midpoint_exact_geometry",
    "joint_cubic_face_q1",
    "joint_cubic_face_q3",
    "joint_cubic_face_q5",
    "joint_cubic_matched_q1",
    "joint_cubic_matched_q3",
    "joint_cubic_matched_q5",
)
CANDIDATES = BASELINE_CANDIDATES + ANALYTIC_CANDIDATES + NUMERICAL_CANDIDATES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved), "size": resolved.stat().st_size}


def _max_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / 1024.0**3


def _json_value(value: Any) -> Any:
    return integrated._json_value(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            **arrays,
            metadata_json=np.asarray(json.dumps(_json_value(metadata), sort_keys=True)),
        )
    temporary.replace(path)


def _progress(stage: str, **details: Any) -> None:
    print(json.dumps({"event": "matched_face_volume", "stage": stage, **details}, sort_keys=True), flush=True)


def _implementation_hash() -> str:
    sources = [
        inspect.getsource(_fit_entity), inspect.getsource(_evaluate_fit),
        inspect.getsource(_face_points_weights), inspect.getsource(_cell_points_weights),
        inspect.getsource(_curl_h), inspect.getsource(_assemble_forms),
        inspect.getsource(_extract),
    ]
    return hashlib.sha256("\n".join(sources).encode()).hexdigest()


@dataclass
class LocalFit:
    center_logical: np.ndarray
    center_regular: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    donors: np.ndarray
    diagnostics: dict[str, Any]


def _fit_entity(
    context: cubic.BuildContext,
    point: np.ndarray,
    *,
    axis: int,
    eta_index: int,
) -> LocalFit:
    """Fit one shared cubic polynomial to actual owner observation moments."""

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
        raise RuntimeError("fixed cubic policy remains deficient for a local entity")
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
    rhs = observation.T @ (kappa2[:, None] * context.arrays["owner_values"][:, donor].T)
    coefficients = np.linalg.solve(gram, rhs).T
    fitted_observations = coefficients @ observation.T
    weighted_residual = np.sqrt(
        np.sum(kappa2[None, :] * (fitted_observations - context.arrays["owner_values"][:, donor]) ** 2, axis=1)
        / np.sum(kappa2)
    )
    return LocalFit(
        center_logical=point,
        center_regular=center_regular,
        scale=scale,
        coefficients=coefficients,
        donors=donor,
        diagnostics={
            "fallback_level": int(level),
            "rank": int(diagnostics["rank"][0]),
            "condition": float(diagnostics["condition"][0]),
            "support_radius": float(diagnostics["support_radius"][0]),
            "minimum_coverage": int(diagnostics["minimum_coverage"][0]),
            "maximum_coverage": int(diagnostics["maximum_coverage"][0]),
            "donor_count": int(len(donor)),
            "tie_fallback_count": int(tie_count),
            "maximum_reproduction_residual": float(np.max(diagnostics["residual"][0])),
            "field_weighted_fit_residual": weighted_residual,
        },
    )


def _evaluate_fit(fit: LocalFit, points: np.ndarray, eta_period: float) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(points, dtype=np.float64)
    x = q[:, 0] * np.cos(q[:, 1])
    y = q[:, 0] * np.sin(q[:, 1])
    eta = cubic.base._unwrap_periodic(q[:, 2], fit.center_regular[2], eta_period)
    z = np.column_stack((x, y, eta))
    normalized = (z - fit.center_regular[None, :]) / fit.scale[None, :]
    basis = np.empty((len(q), len(cubic.EXPONENTS)), dtype=np.float64)
    derivative = np.zeros((len(q), len(cubic.EXPONENTS), 3), dtype=np.float64)
    for slot, exponent in enumerate(cubic.EXPONENTS):
        value = np.ones(len(q), dtype=np.float64)
        for axis, power in enumerate(exponent):
            value *= normalized[:, axis] ** power
        basis[:, slot] = value
        for axis, power in enumerate(exponent):
            if power == 0:
                continue
            component = np.full(len(q), power / fit.scale[axis], dtype=np.float64)
            for other, other_power in enumerate(exponent):
                component *= normalized[:, other] ** (other_power - (1 if other == axis else 0))
            derivative[:, slot, axis] = component
    values = fit.coefficients @ basis.T
    regular_gradient = np.einsum("fs,psc->fpc", fit.coefficients, derivative, optimize=True)
    logical_gradient = np.empty_like(regular_gradient)
    cosine = np.cos(q[:, 1])
    sine = np.sin(q[:, 1])
    logical_gradient[..., 0] = (
        cosine[None, :] * regular_gradient[..., 0]
        + sine[None, :] * regular_gradient[..., 1]
    )
    logical_gradient[..., 1] = (
        -q[:, 0][None, :] * sine[None, :] * regular_gradient[..., 0]
        + q[:, 0][None, :] * cosine[None, :] * regular_gradient[..., 1]
    )
    logical_gradient[..., 2] = regular_gradient[..., 2]
    return values, logical_gradient


def _gauss_interval(lo: float, hi: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * nodes, 0.5 * (hi - lo) * weights


def _face_points_weights(
    faces: Sequence[np.ndarray], key: Sequence[int], order: int
) -> tuple[np.ndarray, np.ndarray]:
    axis, i, j, k = (int(value) for value in key)
    index = (i, j, k)
    coordinates = []
    weights = []
    for component in range(3):
        if component == axis:
            coordinates.append(np.asarray([faces[component][index[component]]]))
            weights.append(np.asarray([1.0]))
        else:
            nodes, node_weights = _gauss_interval(
                faces[component][index[component]],
                faces[component][index[component] + 1],
                order,
            )
            coordinates.append(nodes)
            weights.append(node_weights)
    return (
        np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1).reshape(-1, 3),
        np.prod(np.stack(np.meshgrid(*weights, indexing="ij"), axis=-1), axis=-1).reshape(-1),
    )


def _cell_points_weights(
    faces: Sequence[np.ndarray], cell: Sequence[int], order: int
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = []
    weights = []
    for axis, index in enumerate(cell):
        nodes, node_weights = _gauss_interval(faces[axis][int(index)], faces[axis][int(index) + 1], order)
        coordinates.append(nodes)
        weights.append(node_weights)
    return (
        np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1).reshape(-1, 3),
        np.prod(np.stack(np.meshgrid(*weights, indexing="ij"), axis=-1), axis=-1).reshape(-1),
    )


def _h_without_rho(reference: Any, points: np.ndarray) -> np.ndarray:
    metric = reference._metric(np.asarray(points, dtype=np.float64))
    return np.asarray(metric["bcov"], dtype=np.float64) / np.asarray(metric["B"], dtype=np.float64)[:, None]


def _curl_h(reference: Any, points: np.ndarray, step: float) -> np.ndarray:
    q = np.asarray(points, dtype=np.float64)
    derivative = np.empty((3, len(q), 3), dtype=np.float64)
    for axis in range(3):
        values = []
        for multiplier in (-2.0, -1.0, 1.0, 2.0):
            shifted = q.copy()
            shifted[:, axis] += multiplier * float(step)
            values.append(_h_without_rho(reference, shifted))
        derivative[axis] = (
            values[0] - 8.0 * values[1] + 8.0 * values[2] - values[3]
        ) / (12.0 * float(step))
    result = np.empty((len(q), 3), dtype=np.float64)
    result[:, 0] = derivative[1, :, 2] - derivative[2, :, 1]
    result[:, 1] = derivative[2, :, 0] - derivative[0, :, 2]
    result[:, 2] = derivative[0, :, 1] - derivative[1, :, 0]
    return result


def _analytic_fields(reference: Any, points: np.ndarray, time_value: float) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(points, dtype=np.float64)
    values = np.empty((len(ALL_FIELDS), len(q)), dtype=np.float64)
    gradients = np.empty((len(ALL_FIELDS), len(q), 3), dtype=np.float64)
    phi = reference._fields_raw(q, float(time_value))["phi"]
    values[0] = phi[0]
    gradients[0] = np.stack(phi[1:4], axis=-1)
    omega_q = q.copy()
    boundary = omega_q[:, 0] >= 1.0 - 1.0e-14
    omega_q[boundary, 0] = 1.0 - 3.0 * float(reference.finite_difference_step)
    values[1] = integrated._omega_value(reference, omega_q)
    gradients[1] = integrated._omega_gradient(reference, omega_q)
    for field_index, field in enumerate(FIELDS[1:], start=2):
        value, gradient = continuous._smooth_value_gradient(field, q, reference.eta_period)
        values[field_index] = value
        gradients[field_index] = gradient
    return values, gradients


def _normal_cross(axis: int, one_form: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    return np.cross(one_form, gradient)[:, int(axis)]


def _assemble_forms(
    face_terms: np.ndarray,
    volume_terms: np.ndarray,
    raw_owner: np.ndarray,
    owner_volume: np.ndarray,
    rho_star: float,
) -> np.ndarray:
    """Return A/B/C from raw-cell face and volume brackets.

    Inputs have shape ``(field, raw)`` for the A and swapped-B primitives.
    ``face_terms - volume_terms`` is multiplied by -1/rho for A and +1/rho
    for B.
    """

    owner_count = len(owner_volume)
    output = np.zeros((len(FIELDS), len(ACTIONS), owner_count), dtype=np.float64)
    for field_index in range(len(FIELDS)):
        raw_a = face_terms[field_index, 0] - volume_terms[field_index, 0]
        raw_swapped = face_terms[field_index, 1] - volume_terms[field_index, 1]
        owner_a = np.bincount(raw_owner, weights=raw_a, minlength=owner_count)
        owner_b = np.bincount(raw_owner, weights=raw_swapped, minlength=owner_count)
        output[field_index, 0] = -owner_a / (float(rho_star) * owner_volume)
        output[field_index, 1] = owner_b / (float(rho_star) * owner_volume)
        output[field_index, 2] = 0.5 * (output[field_index, 0] + output[field_index, 1])
    return output


def _cached_midpoint_actions(
    arrays: Mapping[str, np.ndarray], value_name: str, rho_star: float
) -> np.ndarray:
    axes = arrays["face_keys"][:, 0]
    collapsed = arrays["collapsed"].astype(bool)
    values = arrays[value_name]
    gradients = arrays["gradient_dh"]
    generators = np.empty((len(ALL_FIELDS), len(axes)), dtype=np.float64)
    for field_index in range(len(ALL_FIELDS)):
        generators[field_index] = factor._normal_cross(
            axes, arrays["one_form"], gradients[field_index]
        ) * arrays["face_measure"]
    generators[:, collapsed] = 0.0
    result = np.empty((len(FIELDS), len(ACTIONS), len(arrays["owner_volume"])), dtype=np.float64)
    for field_index in range(len(FIELDS)):
        a = cross._assemble(
            generators[0], values[field_index + 1], arrays["center"][field_index + 1], arrays,
            sign=-1.0, rho_star=rho_star,
        )
        b = cross._assemble(
            generators[field_index + 1], values[0], arrays["center"][0], arrays,
            sign=1.0, rho_star=rho_star,
        )
        result[field_index] = np.stack((a, b, 0.5 * (a + b)))
    return result


def _midpoint_actions_from_samples(
    arrays: Mapping[str, np.ndarray],
    values: np.ndarray,
    gradients: np.ndarray,
    one_form: np.ndarray,
    face_measure: np.ndarray,
    rho_star: float,
) -> np.ndarray:
    axes = arrays["face_keys"][:, 0]
    collapsed = arrays["collapsed"].astype(bool)
    generators = np.stack([
        factor._normal_cross(axes, one_form, gradients[field_index]) * face_measure
        for field_index in range(len(ALL_FIELDS))
    ])
    generators[:, collapsed] = 0.0
    result = np.empty((len(FIELDS), len(ACTIONS), len(arrays["owner_volume"])), dtype=np.float64)
    for field_index in range(len(FIELDS)):
        a = cross._assemble(
            generators[0], values[field_index + 1], arrays["center"][field_index + 1], arrays,
            sign=-1.0, rho_star=rho_star,
        )
        b = cross._assemble(
            generators[field_index + 1], values[0], arrays["center"][0], arrays,
            sign=1.0, rho_star=rho_star,
        )
        result[field_index] = np.stack((a, b, 0.5 * (a + b)))
    return result


def _restrict_cross(arrays: Mapping[str, np.ndarray], owner_indices: Sequence[int]) -> dict[str, np.ndarray]:
    selected = np.asarray(owner_indices, dtype=np.int64)
    raw_mask = np.isin(arrays["raw_owner"], selected)
    raw_rows = np.flatnonzero(raw_mask)
    old_raw_owner = arrays["raw_owner"][raw_rows]
    owner_map = {int(old): new for new, old in enumerate(selected)}
    raw_owner = np.asarray([owner_map[int(old)] for old in old_raw_owner], dtype=np.int16)
    old_incidence = arrays["face_incidence"][raw_rows]
    face_rows = np.unique(old_incidence.reshape(-1))
    face_map = np.full(len(arrays["face_keys"]), -1, dtype=np.int64)
    face_map[face_rows] = np.arange(len(face_rows))
    result: dict[str, np.ndarray] = {}
    face_first = {
        "face_keys", "face_points", "face_measure", "collapsed", "eligible_planar",
        "eta_upgrade", "physical_boundary", "exact_derivative_mask", "face_fallback",
        "one_form",
    }
    face_second = {
        "gradient_dh", "gradient_de", "values_P", "values_O2", "values_Ve_planar",
        "values_analytic_full", "values_C3_planar", "values_C3_full",
    }
    for name, value in arrays.items():
        if name in face_first:
            result[name] = value[face_rows]
        elif name in face_second:
            result[name] = value[:, face_rows]
        elif name == "center":
            result[name] = value[:, raw_rows]
        elif name in {"raw_members"}:
            result[name] = value[raw_rows]
        elif name == "raw_owner":
            result[name] = raw_owner
        elif name == "face_incidence":
            result[name] = face_map[old_incidence].astype(np.int32)
        elif name in {"owner_keys", "owner_volume", "owner_region", "owner_expanded"}:
            result[name] = value[selected]
        elif name in {"targets"}:
            result[name] = value[:, selected]
        elif name == "full_operator_replay":
            result[name] = value[:, :, selected]
    return result


def _build_local_fits(
    context: cubic.BuildContext,
    arrays: Mapping[str, np.ndarray],
) -> tuple[list[LocalFit | None], list[LocalFit], dict[str, np.ndarray]]:
    started = time.perf_counter()
    collapsed = arrays["collapsed"].astype(bool)
    face_fits: list[LocalFit | None] = []
    for row, (key, point) in enumerate(zip(arrays["face_keys"], arrays["face_points"])):
        if collapsed[row]:
            face_fits.append(None)
            continue
        face_fits.append(
            _fit_entity(context, point, axis=int(key[0]), eta_index=int(key[3]))
        )
        if (row + 1) % 100 == 0:
            _progress("face_fit_progress", completed=row + 1, total=len(arrays["face_keys"]))
    raw_fits = []
    for raw in arrays["raw_members"]:
        i, j, k = (int(value) for value in raw)
        point = np.asarray((context.x_centers[i], context.y_centers[j], context.z_centers[k]))
        raw_fits.append(_fit_entity(context, point, axis=0, eta_index=k))

    def pack(fits: Sequence[LocalFit | None], prefix: str) -> dict[str, np.ndarray]:
        count = len(fits)
        coefficients = np.full((count, len(ALL_FIELDS), len(cubic.EXPONENTS)), np.nan)
        center_logical = np.full((count, 3), np.nan)
        center_regular = np.full((count, 3), np.nan)
        scale = np.full((count, 3), np.nan)
        fallback = np.full(count, 255, dtype=np.uint8)
        condition = np.full(count, np.nan)
        support_radius = np.full(count, np.nan)
        reproduction = np.full(count, np.nan)
        fit_residual = np.full((count, len(ALL_FIELDS)), np.nan)
        donor_count = np.zeros(count, dtype=np.int16)
        indptr = [0]
        donor_parts = []
        for index, fit in enumerate(fits):
            if fit is not None:
                coefficients[index] = fit.coefficients
                center_logical[index] = fit.center_logical
                center_regular[index] = fit.center_regular
                scale[index] = fit.scale
                fallback[index] = fit.diagnostics["fallback_level"]
                condition[index] = fit.diagnostics["condition"]
                support_radius[index] = fit.diagnostics["support_radius"]
                reproduction[index] = fit.diagnostics["maximum_reproduction_residual"]
                fit_residual[index] = fit.diagnostics["field_weighted_fit_residual"]
                donor_count[index] = len(fit.donors)
                donor_parts.append(fit.donors)
            indptr.append(indptr[-1] + (0 if fit is None else len(fit.donors)))
        return {
            f"{prefix}_fit_coefficients": coefficients,
            f"{prefix}_fit_center_logical": center_logical,
            f"{prefix}_fit_center_regular": center_regular,
            f"{prefix}_fit_scale": scale,
            f"{prefix}_fit_fallback": fallback,
            f"{prefix}_fit_condition": condition,
            f"{prefix}_fit_support_radius": support_radius,
            f"{prefix}_fit_reproduction": reproduction,
            f"{prefix}_fit_field_residual": fit_residual,
            f"{prefix}_fit_donor_count": donor_count,
            f"{prefix}_fit_indptr": np.asarray(indptr, dtype=np.int64),
            f"{prefix}_fit_donors": np.concatenate(donor_parts).astype(np.int64),
        }

    packed = pack(face_fits, "face") | pack(raw_fits, "raw")
    packed["fit_build_seconds"] = np.asarray(time.perf_counter() - started)
    return face_fits, raw_fits, packed


def _fit_midpoint_samples(
    fits: Sequence[LocalFit | None], arrays: Mapping[str, np.ndarray], eta_period: float
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(ALL_FIELDS), len(fits)), dtype=np.float64)
    gradients = np.zeros((len(ALL_FIELDS), len(fits), 3), dtype=np.float64)
    physical = arrays["physical_boundary"].astype(bool)
    for row, fit in enumerate(fits):
        if fit is None:
            continue
        if physical[row]:
            values[:, row] = arrays["values_P"][:, row]
            gradients[:, row] = arrays["gradient_dh"][:, row]
        else:
            value, gradient = _evaluate_fit(fit, arrays["face_points"][row:row + 1], eta_period)
            values[:, row] = value[:, 0]
            gradients[:, row] = gradient[:, 0]
    return values, gradients


def _exact_midpoint_geometry(reference: Any, arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    points = arrays["face_points"]
    regular = ~arrays["collapsed"].astype(bool)
    one_form = np.zeros((len(points), 3), dtype=np.float64)
    one_form[regular] = _h_without_rho(reference, points[regular])
    face_measure = np.empty(len(points), dtype=np.float64)
    # Every cached face midpoint weight is exactly its logical coordinate-face area.
    face_measure[:] = arrays["face_measure"]
    return one_form, face_measure


def _quadrature_actions(
    reference: Any,
    arrays: Mapping[str, np.ndarray],
    faces: Sequence[np.ndarray],
    face_fits: Sequence[LocalFit | None],
    raw_fits: Sequence[LocalFit],
    *,
    order: int,
    time_value: float,
    rho_star: float,
    curl_step: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    raw_count = len(arrays["raw_members"])
    analytic_face = np.zeros((len(FIELDS), 2, raw_count), dtype=np.float64)
    numerical_face = np.zeros_like(analytic_face)
    analytic_face_unanchored = np.zeros_like(analytic_face)
    analytic_volume = np.zeros_like(analytic_face)
    numerical_volume = np.zeros_like(analytic_face)
    analytic_volume_unanchored = np.zeros_like(analytic_face)

    analytic_flux = np.zeros((len(ALL_FIELDS), len(arrays["face_keys"])), dtype=np.float64)
    numerical_flux = np.zeros_like(analytic_flux)
    analytic_product = np.zeros((len(FIELDS), 2, len(arrays["face_keys"])), dtype=np.float64)
    numerical_product = np.zeros_like(analytic_product)
    physical = arrays["physical_boundary"].astype(bool)

    face_batches: list[tuple[np.ndarray, np.ndarray] | None] = []
    face_point_parts = []
    face_offsets = [0]
    for key, fit in zip(arrays["face_keys"], face_fits):
        if fit is None:
            face_batches.append(None)
            face_offsets.append(face_offsets[-1])
            continue
        points, weights = _face_points_weights(faces, key, order)
        face_batches.append((points, weights))
        face_point_parts.append(points)
        face_offsets.append(face_offsets[-1] + len(points))
    all_face_points = np.concatenate(face_point_parts, axis=0)
    all_face_h = _h_without_rho(reference, all_face_points)
    all_face_value, all_face_gradient = _analytic_fields(
        reference, all_face_points, time_value
    )

    for face_index, (key, fit, batch) in enumerate(
        zip(arrays["face_keys"], face_fits, face_batches)
    ):
        if fit is None or batch is None:
            continue
        points, weights = batch
        lo, hi = face_offsets[face_index:face_index + 2]
        h = all_face_h[lo:hi]
        analytic_value = all_face_value[:, lo:hi]
        analytic_gradient = all_face_gradient[:, lo:hi]
        if physical[face_index]:
            numerical_value = np.repeat(arrays["values_P"][:, face_index:face_index + 1], len(points), axis=1)
            numerical_gradient = np.repeat(arrays["gradient_dh"][:, face_index:face_index + 1], len(points), axis=1)
        else:
            numerical_value, numerical_gradient = _evaluate_fit(fit, points, reference.eta_period)
        axis = int(key[0])
        for generator_index in range(len(ALL_FIELDS)):
            analytic_generator = _normal_cross(axis, h, analytic_gradient[generator_index])
            numerical_generator = _normal_cross(axis, h, numerical_gradient[generator_index])
            analytic_flux[generator_index, face_index] = np.dot(weights, analytic_generator)
            numerical_flux[generator_index, face_index] = np.dot(weights, numerical_generator)
        for field_index in range(len(FIELDS)):
            analytic_product[field_index, 0, face_index] = np.dot(
                weights,
                _normal_cross(axis, h, analytic_gradient[0]) * analytic_value[field_index + 1],
            )
            analytic_product[field_index, 1, face_index] = np.dot(
                weights,
                _normal_cross(axis, h, analytic_gradient[field_index + 1]) * analytic_value[0],
            )
            numerical_product[field_index, 0, face_index] = np.dot(
                weights,
                _normal_cross(axis, h, numerical_gradient[0]) * numerical_value[field_index + 1],
            )
            numerical_product[field_index, 1, face_index] = np.dot(
                weights,
                _normal_cross(axis, h, numerical_gradient[field_index + 1]) * numerical_value[0],
            )

    cell_batches = [
        _cell_points_weights(faces, raw, order) for raw in arrays["raw_members"]
    ]
    cell_offsets = np.cumsum([0] + [len(points) for points, _weights in cell_batches])
    all_cell_points = np.concatenate([points for points, _weights in cell_batches], axis=0)
    all_curl_h = _curl_h(reference, all_cell_points, curl_step)
    all_cell_value, all_cell_gradient = _analytic_fields(
        reference, all_cell_points, time_value
    )

    for raw_index, fit in enumerate(raw_fits):
        for axis in range(3):
            lower, upper = arrays["face_incidence"][raw_index, axis]
            for field_index in range(len(FIELDS)):
                analytic_face[field_index, 0, raw_index] += (
                    analytic_product[field_index, 0, upper]
                    - arrays["center"][field_index + 1, raw_index] * analytic_flux[0, upper]
                    - analytic_product[field_index, 0, lower]
                    + arrays["center"][field_index + 1, raw_index] * analytic_flux[0, lower]
                )
                analytic_face[field_index, 1, raw_index] += (
                    analytic_product[field_index, 1, upper]
                    - arrays["center"][0, raw_index] * analytic_flux[field_index + 1, upper]
                    - analytic_product[field_index, 1, lower]
                    + arrays["center"][0, raw_index] * analytic_flux[field_index + 1, lower]
                )
                numerical_face[field_index, 0, raw_index] += (
                    numerical_product[field_index, 0, upper]
                    - arrays["center"][field_index + 1, raw_index] * numerical_flux[0, upper]
                    - numerical_product[field_index, 0, lower]
                    + arrays["center"][field_index + 1, raw_index] * numerical_flux[0, lower]
                )
                numerical_face[field_index, 1, raw_index] += (
                    numerical_product[field_index, 1, upper]
                    - arrays["center"][0, raw_index] * numerical_flux[field_index + 1, upper]
                    - numerical_product[field_index, 1, lower]
                    + arrays["center"][0, raw_index] * numerical_flux[field_index + 1, lower]
                )
                analytic_face_unanchored[field_index, 0, raw_index] += (
                    analytic_product[field_index, 0, upper] - analytic_product[field_index, 0, lower]
                )
                analytic_face_unanchored[field_index, 1, raw_index] += (
                    analytic_product[field_index, 1, upper] - analytic_product[field_index, 1, lower]
                )

        points, weights = cell_batches[raw_index]
        lo, hi = cell_offsets[raw_index:raw_index + 2]
        curl_h = all_curl_h[lo:hi]
        analytic_value = all_cell_value[:, lo:hi]
        analytic_gradient = all_cell_gradient[:, lo:hi]
        numerical_value, numerical_gradient = _evaluate_fit(fit, points, reference.eta_period)
        for field_index in range(len(FIELDS)):
            analytic_div_a = np.sum(analytic_gradient[0] * curl_h, axis=1)
            analytic_div_swapped = np.sum(analytic_gradient[field_index + 1] * curl_h, axis=1)
            numerical_div_a = np.sum(numerical_gradient[0] * curl_h, axis=1)
            numerical_div_swapped = np.sum(numerical_gradient[field_index + 1] * curl_h, axis=1)
            analytic_volume[field_index, 0, raw_index] = np.dot(
                weights,
                (analytic_value[field_index + 1] - arrays["center"][field_index + 1, raw_index])
                * analytic_div_a,
            )
            analytic_volume[field_index, 1, raw_index] = np.dot(
                weights,
                (analytic_value[0] - arrays["center"][0, raw_index]) * analytic_div_swapped,
            )
            numerical_volume[field_index, 0, raw_index] = np.dot(
                weights,
                (numerical_value[field_index + 1] - arrays["center"][field_index + 1, raw_index])
                * numerical_div_a,
            )
            numerical_volume[field_index, 1, raw_index] = np.dot(
                weights,
                (numerical_value[0] - arrays["center"][0, raw_index]) * numerical_div_swapped,
            )
            analytic_volume_unanchored[field_index, 0, raw_index] = np.dot(
                weights, analytic_value[field_index + 1] * analytic_div_a
            )
            analytic_volume_unanchored[field_index, 1, raw_index] = np.dot(
                weights, analytic_value[0] * analytic_div_swapped
            )

    zeros = np.zeros_like(analytic_volume)
    actions = {
        "analytic_face": _assemble_forms(
            analytic_face, zeros, arrays["raw_owner"], arrays["owner_volume"], rho_star
        ),
        "analytic_matched": _assemble_forms(
            analytic_face, analytic_volume, arrays["raw_owner"], arrays["owner_volume"], rho_star
        ),
        "analytic_unanchored": _assemble_forms(
            analytic_face_unanchored, analytic_volume_unanchored,
            arrays["raw_owner"], arrays["owner_volume"], rho_star,
        ),
        "numerical_face": _assemble_forms(
            numerical_face, zeros, arrays["raw_owner"], arrays["owner_volume"], rho_star
        ),
        "numerical_matched": _assemble_forms(
            numerical_face, numerical_volume, arrays["raw_owner"], arrays["owner_volume"], rho_star
        ),
    }
    factors = {
        "analytic_face_flux": analytic_flux,
        "numerical_face_flux": numerical_flux,
        "analytic_face_product": analytic_product,
        "numerical_face_product": numerical_product,
        "analytic_raw_face": analytic_face,
        "numerical_raw_face": numerical_face,
        "analytic_raw_volume": analytic_volume,
        "numerical_raw_volume": numerical_volume,
        "analytic_raw_face_unanchored": analytic_face_unanchored,
        "analytic_raw_volume_unanchored": analytic_volume_unanchored,
    }
    return actions, factors


def _volume_terms_with_step(
    reference: Any,
    arrays: Mapping[str, np.ndarray],
    faces: Sequence[np.ndarray],
    raw_fits: Sequence[LocalFit],
    *,
    order: int,
    time_value: float,
    curl_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    analytic_volume = np.zeros((len(FIELDS), 2, len(raw_fits)), dtype=np.float64)
    numerical_volume = np.zeros_like(analytic_volume)
    batches = [
        _cell_points_weights(faces, raw, order) for raw in arrays["raw_members"]
    ]
    offsets = np.cumsum([0] + [len(points) for points, _weights in batches])
    all_points = np.concatenate([points for points, _weights in batches], axis=0)
    all_curl_h = _curl_h(reference, all_points, curl_step)
    all_analytic_value, all_analytic_gradient = _analytic_fields(
        reference, all_points, time_value
    )
    for raw_index, fit in enumerate(raw_fits):
        points, weights = batches[raw_index]
        lo, hi = offsets[raw_index:raw_index + 2]
        curl_h = all_curl_h[lo:hi]
        analytic_value = all_analytic_value[:, lo:hi]
        analytic_gradient = all_analytic_gradient[:, lo:hi]
        numerical_value, numerical_gradient = _evaluate_fit(fit, points, reference.eta_period)
        for field_index in range(len(FIELDS)):
            for swapped, (generator_index, transported_index, anchor_index) in enumerate((
                (0, field_index + 1, field_index + 1),
                (field_index + 1, 0, 0),
            )):
                analytic_div = np.sum(analytic_gradient[generator_index] * curl_h, axis=1)
                numerical_div = np.sum(numerical_gradient[generator_index] * curl_h, axis=1)
                anchor = arrays["center"][anchor_index, raw_index]
                analytic_volume[field_index, swapped, raw_index] = np.dot(
                    weights, (analytic_value[transported_index] - anchor) * analytic_div
                )
                numerical_volume[field_index, swapped, raw_index] = np.dot(
                    weights, (numerical_value[transported_index] - anchor) * numerical_div
                )
    return analytic_volume, numerical_volume


def _reference_targets(
    qualification_cache: Path, owner_keys: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(qualification_cache, allow_pickle=False) as source:
        lookup = {tuple(key): index for index, key in enumerate(source["owner_keys"].tolist())}
        positions = np.asarray([lookup[tuple(key)] for key in owner_keys.tolist()], dtype=np.int64)
        target = np.stack((
            np.asarray(source["reference:actual_vorticity:ibp_q4"])[positions],
            np.asarray(source["reference:smooth_regular_scalar:q3"])[positions],
            np.asarray(source["reference:smooth_eta_varying_scalar:q3"])[positions],
        ))
        uncertainty = np.stack((
            np.asarray(source["reference:actual_vorticity:ibp_q4"])[positions]
            - np.asarray(source["reference:actual_vorticity:ibp_q3"])[positions],
            np.asarray(source["reference:smooth_regular_scalar:q5"])[positions]
            - np.asarray(source["reference:smooth_regular_scalar:q3"])[positions],
            np.asarray(source["reference:smooth_eta_varying_scalar:q5"])[positions]
            - np.asarray(source["reference:smooth_eta_varying_scalar:q3"])[positions],
        ))
    return target, uncertainty, {
        "actual_vorticity": "complete IBP q4",
        "smooth_regular_scalar": "direct physical-volume q3",
        "smooth_eta_varying_scalar": "direct physical-volume q3",
    }


def _shared_incidence_check(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    maximum_internal = 0
    maximum_occurrence = 0
    internal_count = 0
    for owner in range(len(arrays["owner_volume"])):
        rows = np.flatnonzero(arrays["raw_owner"] == owner)
        signed: dict[int, int] = {}
        occurrences: dict[int, int] = {}
        for row in rows:
            for axis in range(3):
                lower, upper = arrays["face_incidence"][row, axis]
                signed[int(lower)] = signed.get(int(lower), 0) - 1
                signed[int(upper)] = signed.get(int(upper), 0) + 1
                occurrences[int(lower)] = occurrences.get(int(lower), 0) + 1
                occurrences[int(upper)] = occurrences.get(int(upper), 0) + 1
        maximum_occurrence = max(maximum_occurrence, max(occurrences.values(), default=0))
        for face, count in occurrences.items():
            if count == 2:
                internal_count += 1
                maximum_internal = max(maximum_internal, abs(signed[face]))
    return {
        "internal_face_occurrences": internal_count,
        "maximum_internal_signed_incidence": maximum_internal,
        "maximum_face_occurrence_within_owner": maximum_occurrence,
    }


def _extract(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    arrays, cross_metadata = cross._load_npz(args.cross_cache, validate_sources=True)
    _progress("cross_cache_loaded", seconds=time.perf_counter() - started)
    if args.owner_indices:
        arrays = _restrict_cross(arrays, args.owner_indices)
    resolution = int(cross_metadata["resolution"])
    rho_star = float(cross_metadata["rho_star"])
    context = cubic._load_context(args.geometry, args.baseline, resolution)
    _progress("context_loaded", resolution=resolution, seconds=time.perf_counter() - started)
    faces = (context.x_faces, context.y_faces, context.z_faces)
    reference = integrated._reference(
        args.reference_sidecar,
        finite_difference_step=float(args.omega_step),
        # The sidecar's multi-gigabyte makegrid hash was already qualified and
        # is frozen in the upstream evidence.  Revalidate the sidecar and
        # compact input caches below without re-reading that external payload.
        verify_hashes=False,
    )
    _progress("reference_loaded", resolution=resolution, seconds=time.perf_counter() - started)
    target, uncertainty, reference_policy = _reference_targets(
        args.qualification_cache, arrays["owner_keys"]
    )
    face_fits, raw_fits, fit_arrays = _build_local_fits(context, arrays)
    _progress(
        "fits_complete", resolution=resolution, faces=len(face_fits), raw_cells=len(raw_fits),
        seconds=float(fit_arrays["fit_build_seconds"]),
    )

    actions: dict[str, np.ndarray] = {
        "P_Dh_midpoint": _cached_midpoint_actions(arrays, "values_P", rho_star),
        "O2_Dh_midpoint": _cached_midpoint_actions(arrays, "values_O2", rho_star),
        "C3_planar_Dh_midpoint": _cached_midpoint_actions(arrays, "values_C3_planar", rho_star),
    }
    numerical_midpoint_value, numerical_midpoint_gradient = _fit_midpoint_samples(
        face_fits, arrays, reference.eta_period
    )
    regular_faces = ~arrays["collapsed"].astype(bool)
    analytic_midpoint_value = np.zeros((len(ALL_FIELDS), len(arrays["face_points"])))
    analytic_midpoint_gradient = np.zeros((len(ALL_FIELDS), len(arrays["face_points"]), 3))
    regular_value, regular_gradient = _analytic_fields(
        reference, arrays["face_points"][regular_faces], float(args.time)
    )
    analytic_midpoint_value[:, regular_faces] = regular_value
    analytic_midpoint_gradient[:, regular_faces] = regular_gradient
    exact_one_form, exact_face_measure = _exact_midpoint_geometry(reference, arrays)
    actions["joint_cubic_midpoint_cached_geometry"] = _midpoint_actions_from_samples(
        arrays, numerical_midpoint_value, numerical_midpoint_gradient,
        arrays["one_form"], arrays["face_measure"], rho_star,
    )
    actions["joint_cubic_midpoint_exact_geometry"] = _midpoint_actions_from_samples(
        arrays, numerical_midpoint_value, numerical_midpoint_gradient,
        exact_one_form, exact_face_measure, rho_star,
    )
    actions["analytic_midpoint_cached_geometry"] = _midpoint_actions_from_samples(
        arrays, analytic_midpoint_value, analytic_midpoint_gradient,
        arrays["one_form"], arrays["face_measure"], rho_star,
    )
    actions["analytic_midpoint_exact_geometry"] = _midpoint_actions_from_samples(
        arrays, analytic_midpoint_value, analytic_midpoint_gradient,
        exact_one_form, exact_face_measure, rho_star,
    )

    factor_arrays: dict[str, np.ndarray] = {}
    q3_factors = None
    for order in args.orders:
        order_started = time.perf_counter()
        evaluated, factors = _quadrature_actions(
            reference, arrays, faces, face_fits, raw_fits,
            order=order, time_value=float(args.time), rho_star=rho_star,
            curl_step=float(args.curl_step),
        )
        actions[f"analytic_face_q{order}"] = evaluated["analytic_face"]
        actions[f"analytic_matched_q{order}"] = evaluated["analytic_matched"]
        actions[f"analytic_unanchored_q{order}"] = evaluated["analytic_unanchored"]
        actions[f"joint_cubic_face_q{order}"] = evaluated["numerical_face"]
        actions[f"joint_cubic_matched_q{order}"] = evaluated["numerical_matched"]
        if order == 3:
            q3_factors = factors
            factor_arrays.update({f"q3:{name}": value for name, value in factors.items()})
        _progress(
            "quadrature_complete", resolution=resolution, order=order,
            seconds=time.perf_counter() - order_started,
        )
    if q3_factors is not None:
        for label, step in (("half", 0.5 * float(args.curl_step)), ("double", 2.0 * float(args.curl_step))):
            analytic_volume, numerical_volume = _volume_terms_with_step(
                reference, arrays, faces, raw_fits,
                order=3, time_value=float(args.time), curl_step=step,
            )
            actions[f"analytic_matched_q3_curl_{label}"] = _assemble_forms(
                q3_factors["analytic_raw_face"], analytic_volume,
                arrays["raw_owner"], arrays["owner_volume"], rho_star,
            )
            actions[f"joint_cubic_matched_q3_curl_{label}"] = _assemble_forms(
                q3_factors["numerical_raw_face"], numerical_volume,
                arrays["raw_owner"], arrays["owner_volume"], rho_star,
            )

    c3_full = _cached_midpoint_actions(arrays, "values_C3_full", rho_star)
    replay_max = float(np.max(np.abs(actions["O2_Dh_midpoint"] - arrays["full_operator_replay"])))
    joint_replay_max = float(
        np.max(np.abs(actions["joint_cubic_midpoint_cached_geometry"] - c3_full))
    )
    regular = ~arrays["collapsed"].astype(bool)
    value_replay_max = float(
        np.max(np.abs(numerical_midpoint_value[:, regular] - arrays["values_C3_full"][:, regular]))
    )
    gradient_replay_max = float(
        np.max(np.abs(numerical_midpoint_gradient[:, regular] - arrays["gradient_dh"][:, regular]))
    )
    zero = np.zeros((len(FIELDS), 2, len(arrays["raw_members"])), dtype=np.float64)
    constant_max = float(np.max(np.abs(_assemble_forms(
        zero, zero, arrays["raw_owner"], arrays["owner_volume"], rho_star
    ))))

    cache_arrays: dict[str, np.ndarray] = {
        "owner_keys": arrays["owner_keys"],
        "owner_volume": arrays["owner_volume"],
        "owner_region": arrays["owner_region"],
        "owner_expanded": arrays["owner_expanded"],
        "raw_members": arrays["raw_members"],
        "raw_owner": arrays["raw_owner"],
        "face_keys": arrays["face_keys"],
        "face_incidence": arrays["face_incidence"],
        "target": target,
        "reference_uncertainty_vector": uncertainty,
        "analytic_midpoint_value": analytic_midpoint_value,
        "analytic_midpoint_gradient": analytic_midpoint_gradient,
        "numerical_midpoint_value": numerical_midpoint_value,
        "numerical_midpoint_gradient": numerical_midpoint_gradient,
        "exact_midpoint_one_form": exact_one_form,
        **fit_arrays,
        **factor_arrays,
        **{f"action:{name}": value for name, value in actions.items()},
    }
    metadata = {
        "schema": CACHE_SCHEMA,
        "resolution": resolution,
        "status": "complete",
        "owner_count": len(arrays["owner_volume"]),
        "raw_cell_count": len(arrays["raw_members"]),
        "face_count": len(arrays["face_keys"]),
        "orders": list(args.orders),
        "rho_star": rho_star,
        "time": float(args.time),
        "curl_step": float(args.curl_step),
        "omega_step": float(args.omega_step),
        "reference_policy": reference_policy,
        "design_contract": {
            "H": "b_cov/(rho_star B); implementation factors 1/rho_star outside",
            "A": "-1/(rho V) [sum_faces (h cross grad g).n (f-c_f) - int (f-c_f) grad(g).curl(h)]",
            "B": "-A(f,g)",
            "C": "(A+B)/2",
            "normalization": "frozen physical owner volume",
            "anchor": "existing stored raw-cell center value, constant inside each raw cell",
            "face_fit": "one balanced-cubic fit per canonical face; shared values and derivatives",
            "volume_fit": "one balanced-cubic five-plane fit per raw-cell center; shared values and derivatives",
            "geometry": "analytical h at quadrature nodes and fourth-order finite-difference curl(h)",
            "boundary": "collapsed radial face zero; frozen numerical trace retained on physical boundary",
        },
        "verification": {
            "baseline_action_replay_max_abs": replay_max,
            "joint_cubic_midpoint_replay_max_abs": joint_replay_max,
            "joint_cubic_value_replay_max_abs": value_replay_max,
            "joint_cubic_gradient_replay_max_abs": gradient_replay_max,
            "constant_annihilation_max_abs": constant_max,
            "shared_incidence": _shared_incidence_check(arrays),
        },
        "sources": {
            "cross_cache": _identity(args.cross_cache),
            "qualification_cache": _identity(args.qualification_cache),
            "reference_sidecar": _identity(args.reference_sidecar),
            "implementation": _identity(Path(__file__)),
        },
        "implementation_sha256": _implementation_hash(),
        "timing_seconds": time.perf_counter() - started,
        "maximum_rss_gib": _max_rss_gib(),
        "scope": {
            "bounded_complete_owner_sample": True,
            "global_order_claim": False,
            "material_upwinding_claim": False,
            "production_changes": [],
        },
    }
    metadata["array_sha256"] = _array_hash(*(cache_arrays[name] for name in sorted(cache_arrays)))
    _write_npz(args.output, cache_arrays, metadata)
    _progress(
        "extract_complete", resolution=resolution, output=str(args.output.resolve()),
        seconds=metadata["timing_seconds"], maximum_rss_gib=metadata["maximum_rss_gib"],
    )
    return metadata


def _load_cache(path: Path, *, validate_sources: bool = True) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"].item()))
        arrays = {name: np.asarray(source[name]) for name in source.files if name != "metadata_json"}
    if metadata.get("schema") != CACHE_SCHEMA:
        raise ValueError("unsupported matched face-volume cache")
    if metadata.get("array_sha256") != _array_hash(*(arrays[name] for name in sorted(arrays))):
        raise ValueError("matched face-volume cache array hash mismatch")
    if validate_sources:
        if metadata.get("implementation_sha256") != _implementation_hash():
            raise ValueError("matched face-volume implementation changed")
        stale = []
        for name, identity in metadata["sources"].items():
            source = Path(identity["path"])
            if not source.is_file() or _sha256(source) != identity["sha256"]:
                stale.append(name)
        if stale:
            raise ValueError("matched face-volume source changed: " + ", ".join(stale))
    return arrays, metadata


def _effect(value: np.ndarray, volume: np.ndarray) -> dict[str, Any]:
    return {
        "volume_weighted_rms": float(np.sqrt(np.sum(volume * value**2) / np.sum(volume))),
        "signed_volume_weighted_mean": float(np.sum(volume * value) / np.sum(volume)),
        "maximum_absolute": float(np.max(np.abs(value))),
    }


def _replay(args: argparse.Namespace) -> dict[str, Any]:
    arrays, metadata = _load_cache(args.cache, validate_sources=True)
    volume = arrays["owner_volume"]
    region = arrays["owner_region"]
    expanded = arrays["owner_expanded"].astype(bool)
    available = [name.removeprefix("action:") for name in arrays if name.startswith("action:")]
    candidates = [name for name in CANDIDATES if name in available]
    results: dict[str, Any] = {}
    for field_index, field in enumerate(FIELDS):
        target = arrays["target"][field_index]
        uncertainty = _effect(arrays["reference_uncertainty_vector"][field_index], volume)
        field_result: dict[str, Any] = {
            "reference_uncertainty": uncertainty,
            "candidates": {},
            "signed_effects": {},
        }
        for candidate in candidates:
            field_result["candidates"][candidate] = {}
            action_values = arrays[f"action:{candidate}"][field_index]
            for action_index, action in enumerate(ACTIONS):
                statistics = cross._statistics(
                    action_values[action_index], target, volume, region, expanded
                )
                statistics["reference_uncertainty_fraction"] = (
                    uncertainty["volume_weighted_rms"]
                    / max(statistics["absolute_l2"], np.finfo(float).tiny)
                )
                field_result["candidates"][candidate][action] = statistics
        effect_pairs = {
            "analytic_geometry_at_points": (
                "analytic_midpoint_exact_geometry", "analytic_midpoint_cached_geometry"
            ),
            "analytic_face_integration": (
                "analytic_face_q3", "analytic_midpoint_exact_geometry"
            ),
            "analytic_volume_correction": ("analytic_matched_q3", "analytic_face_q3"),
            "analytic_anchor_equivalence": ("analytic_matched_q3", "analytic_unanchored_q3"),
            "joint_reconstruction_from_O2": (
                "joint_cubic_midpoint_cached_geometry", "O2_Dh_midpoint"
            ),
            "joint_geometry_at_points": (
                "joint_cubic_midpoint_exact_geometry", "joint_cubic_midpoint_cached_geometry"
            ),
            "joint_face_integration": (
                "joint_cubic_face_q3", "joint_cubic_midpoint_exact_geometry"
            ),
            "joint_volume_correction": (
                "joint_cubic_matched_q3", "joint_cubic_face_q3"
            ),
        }
        for label, (new, old) in effect_pairs.items():
            if new not in available or old not in available:
                continue
            field_result["signed_effects"][label] = {
                action: _effect(
                    arrays[f"action:{new}"][field_index, action_index]
                    - arrays[f"action:{old}"][field_index, action_index],
                    volume,
                )
                for action_index, action in enumerate(ACTIONS)
            }
        results[field] = field_result

    curl_sensitivity = {}
    for family in ("analytic", "joint_cubic"):
        base_name = f"{family}_matched_q3"
        if f"action:{base_name}" not in arrays:
            continue
        curl_sensitivity[family] = {}
        for label in ("half", "double"):
            shifted = f"{family}_matched_q3_curl_{label}"
            curl_sensitivity[family][label] = {
                field: {
                    action: _effect(
                        arrays[f"action:{shifted}"][field_index, action_index]
                        - arrays[f"action:{base_name}"][field_index, action_index],
                        volume,
                    )
                    for action_index, action in enumerate(ACTIONS)
                }
                for field_index, field in enumerate(FIELDS)
            }
    quadrature_sensitivity = {}
    for family in ("analytic_face", "analytic_matched", "analytic_unanchored", "joint_cubic_face", "joint_cubic_matched"):
        if f"action:{family}_q5" not in arrays:
            continue
        quadrature_sensitivity[family] = {
            field: {
                action: _effect(
                    arrays[f"action:{family}_q5"][field_index, action_index]
                    - arrays[f"action:{family}_q3"][field_index, action_index],
                    volume,
                )
                for action_index, action in enumerate(ACTIONS)
            }
            for field_index, field in enumerate(FIELDS)
        }
    payload = {
        "schema": CASE_SCHEMA,
        "resolution": metadata["resolution"],
        "owner_count": metadata["owner_count"],
        "results": results,
        "curl_step_sensitivity": curl_sensitivity,
        "quadrature_q5_minus_q3": quadrature_sensitivity,
        "verification": metadata["verification"],
        "design_contract": metadata["design_contract"],
        "runtime": {
            "extraction_seconds": metadata["timing_seconds"],
            "maximum_rss_gib": metadata["maximum_rss_gib"],
        },
        "cache": _identity(args.cache),
        "scope": metadata["scope"],
    }
    _write_json(args.output, payload)
    _progress("replay_complete", resolution=metadata["resolution"], output=str(args.output.resolve()))
    return payload


def _merge(args: argparse.Namespace) -> dict[str, Any]:
    cases = [json.loads(path.read_text()) for path in args.cases]
    cases.sort(key=lambda item: item["resolution"])
    if [case["resolution"] for case in cases] != [48, 64]:
        raise ValueError("merge requires N48 and N64 cases")
    payload = {
        "schema": SUMMARY_SCHEMA,
        "resolutions": [48, 64],
        "cases": {str(case["resolution"]): case for case in cases},
        "scope": {
            "bounded_selected_sample_only": True,
            "global_order_claim": False,
            "production_changes": [],
        },
    }
    _write_json(args.output, payload)
    _write_report(args.report, payload)
    _progress("merge_complete", output=str(args.output.resolve()), report=str(args.report.resolve()))
    return payload


def _ratio(new: float, old: float) -> str:
    return f"{new / old:.3f}x ({(new / old - 1.0) * 100:+.1f}%)"


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    primary = (
        "P_Dh_midpoint", "O2_Dh_midpoint", "C3_planar_Dh_midpoint",
        "analytic_midpoint_cached_geometry", "analytic_midpoint_exact_geometry",
        "analytic_face_q3", "analytic_matched_q3",
        "joint_cubic_midpoint_cached_geometry", "joint_cubic_midpoint_exact_geometry",
        "joint_cubic_face_q3", "joint_cubic_matched_q3",
    )
    lines = [
        "# HSX bounded matched face-and-volume integration audit",
        "",
        "This experiment uses the frozen 28 complete-owner N48/N64 samples and independent integrated references. It changes neither production reconstruction nor the boundary model. P/O2/C3 are transported-value or bounded polynomial choices paired with the already frozen cubic derivative; they are not labels for a certified production method.",
        "",
        "## Centered-C errors on the fixed sample",
        "",
        "| Candidate | Field | N48 | N64 | N64/N48 error ratio |",
        "|---|---|---:|---:|---:|",
    ]
    for candidate in primary:
        for field in FIELDS:
            values = []
            for resolution in (48, 64):
                case = summary["cases"][str(resolution)]
                item = case["results"][field]["candidates"].get(candidate)
                values.append(None if item is None else item["C"]["absolute_l2"])
            if any(value is None for value in values):
                continue
            lines.append(
                f"| {candidate} | {field} | {values[0]:.7g} | {values[1]:.7g} | {values[1]/values[0]:.4f} |"
            )
    lines.extend([
        "",
        "## Integration effects",
        "",
    ])
    for resolution in (48, 64):
        case = summary["cases"][str(resolution)]
        lines.append(f"### N{resolution}")
        lines.append("")
        for field in FIELDS:
            candidates = case["results"][field]["candidates"]
            analytic_face = candidates["analytic_face_q3"]["C"]["absolute_l2"]
            analytic_matched = candidates["analytic_matched_q3"]["C"]["absolute_l2"]
            numerical_face = candidates["joint_cubic_face_q3"]["C"]["absolute_l2"]
            numerical_matched = candidates["joint_cubic_matched_q3"]["C"]["absolute_l2"]
            numerical_midpoint = candidates["joint_cubic_midpoint_exact_geometry"]["C"]["absolute_l2"]
            lines.append(
                f"- {field}: analytical volume correction {analytic_face:.7g} → {analytic_matched:.7g} "
                f"({_ratio(analytic_matched, analytic_face)}); numerical face integration changes "
                f"{numerical_midpoint:.7g} → {numerical_face:.7g}, and the matched volume correction gives "
                f"{numerical_matched:.7g} ({_ratio(numerical_matched, numerical_face)} versus face-only)."
            )
        lines.append("")
    lines.extend([
        "## Verification and limits",
        "",
    ])
    for resolution in (48, 64):
        case = summary["cases"][str(resolution)]
        verify = case["verification"]
        lines.append(
            f"- N{resolution}: baseline replay `{verify['baseline_action_replay_max_abs']:.3g}`, "
            f"joint cubic midpoint replay `{verify['joint_cubic_midpoint_replay_max_abs']:.3g}`, "
            f"constant annihilation `{verify['constant_annihilation_max_abs']:.3g}`, "
            f"internal signed incidence `{verify['shared_incidence']['maximum_internal_signed_incidence']}`, "
            f"runtime {case['runtime']['extraction_seconds']:.1f} s, peak RSS {case['runtime']['maximum_rss_gib']:.2f} GiB."
        )
    lines.extend([
        "",
        "All A/B/C constituents, disjoint region budgets, max norms, reference-uncertainty fractions, signed effect vectors, curl-step checks, and q5−q3 checks are retained in the case JSON and cache. These selected samples establish neither global order nor material-upwind or energy stability.",
        "",
        "## Decision",
        "",
        "The numerical decision is generated only after both resolution cases are complete; interpret the tables above by separating analytical integration benefit from the same functional applied to owner-reconstructed factors. No global campaign is authorized by this result.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--geometry", type=Path, required=True)
    extract.add_argument("--baseline", type=Path, required=True)
    extract.add_argument("--reference-sidecar", type=Path, required=True)
    extract.add_argument("--cross-cache", type=Path, required=True)
    extract.add_argument("--qualification-cache", type=Path, required=True)
    extract.add_argument("--owner-indices", type=int, nargs="*", default=())
    extract.add_argument("--orders", type=int, nargs="+", choices=(1, 3, 5), default=(1, 3, 5))
    extract.add_argument("--time", type=float, default=1.0e-6)
    extract.add_argument("--curl-step", type=float, default=2.0e-4)
    extract.add_argument("--omega-step", type=float, default=2.0e-4)
    extract.add_argument("--output", type=Path, required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--cache", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("cases", nargs=2, type=Path)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "extract":
        _extract(args)
    elif args.command == "replay":
        _replay(args)
    elif args.command == "merge":
        _merge(args)
    else:
        raise ValueError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
