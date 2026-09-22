#!/usr/bin/env python3
"""Bounded actual-HSX audit of the complete four-field curvature action.

This is a research driver for P06-A/B/C.  It does not change a production
selector.  The driver keeps the current coupled characteristic matrix and
compares the live production action with consistently integrated smooth
controls on deterministic complete-owner patches.  The successful P04/P05
selection-v3 reconstruction is imported read-only through the frozen research
adapter until the shared package extraction is available.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
SCRIPTS = REPO / "scripts"
REMOTE = SCRIPTS / "hsx_remote_qualification"
CAMPAIGN = WORKSPACE / "work/p_centered_cubic_c54b0552_kFhdmt"
MATERIAL_CAMPAIGN = WORKSPACE / "work/p05_material_257bd55f_qQnlSX"
GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
BASELINE = REPO / "work/perpendicular_second_order_hsx_p01_p03/continuous_global_baseline"
SIDECAR = CAMPAIGN / "portable_reference_sidecar.json"
LOCAL_METRIC = WORKSPACE / "hsx_metric_d58d392545fd3917efeb83b6.npz"
LOCAL_MAKEGRID = WORKSPACE / "mgrid_res2p5cm_180pln.nc"
P05_NODEWISE = WORKSPACE / "work/p05_nodewise_cubic_jump_20260921/run_bounded.py"

# The research drivers import JAX eagerly, so place all runtime caches in the
# project-owned writable workspace before loading them.
os.environ.setdefault("HSX_DEPLOYMENT_ROOT", str(WORKSPACE))
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(WORKSPACE / ".jax_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(WORKSPACE / ".drbx_jax_cache"))

for entry in (REPO, SCRIPTS, REMOTE):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import audit_hsx_bracket_integrated_reference as integrated  # noqa: E402
import audit_hsx_cubic_derivative_global as cubic  # noqa: E402
import audit_hsx_matched_face_volume as matched  # noqa: E402
import simulate_hsx_mms as mms  # noqa: E402
import simulate_hsx_blob as blob  # noqa: E402
import numerics  # noqa: E402
from hsx_mms_continuum_reference import build_continuum_reference_from_sidecar  # noqa: E402
from drbx.native.fci_curvature_production_flux import (  # noqa: E402
    curvature_principal_matrix,
)
from drbx.native.fci_operators import (  # noqa: E402
    _curvature_bc_characteristic_wall_states,
)


SCHEMA = "drbx.perpendicular.p06-curvature-bounded-v1"
STATE_NAMES = ("density", "Te", "Ti", "vorticity")
EQUATION_NAMES = STATE_NAMES
DIRECTIONS = ("u", "theta", "eta")
FIELD_NAMES = ("corrected_frozen_mms", "regular_chart_heldout")
VARIANTS = ("L", "G", "C", "U", "oracle")
REFERENCE_ORDERS = (1, 3, 5)
TAU = float(mms.PHYSICAL_PARAMETERS["tau"])
TIME_VALUE = 0.37
BIAS = 0.75
FLOOR = 1.0e-12


def _load_external(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


p05_nodewise = _load_external(P05_NODEWISE, "p06_frozen_p05_nodewise")


def _peak_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / 1024.0**3


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


def _json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files if name != "metadata_json"}


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _progress(output: Path, stage: str, **details: Any) -> None:
    payload = {
        "schema": f"{SCHEMA}.progress",
        "stage": stage,
        "updated_unix": time.time(),
        **_json(details),
    }
    _write_json(output / "progress.json", payload)
    print(json.dumps({"event": "p06_curvature", "stage": stage, **_json(details)}, sort_keys=True), flush=True)


def _implementation_hash() -> str:
    functions = (
        _heldout_fields,
        _continuum_terms,
        _continuum_terms_directional,
        _select_owners,
        _support,
        _fit_volume_state,
        _integrate_sources,
        _geometry_step_sensitivity,
        _coefficient_divergence_diagnostic,
        _live_side_donor_hash,
        _curvature_face_geometry,
        _interface_correction,
        _production_action,
    )
    return hashlib.sha256("\n".join(inspect.getsource(function) for function in functions).encode()).hexdigest()


def _localized_sidecar(output: Path) -> Path:
    payload = json.loads(SIDECAR.read_text())
    if _sha256(LOCAL_METRIC) != payload["metric_cache"]["sha256"]:
        raise ValueError("qualified metric cache identity mismatch")
    if _sha256(LOCAL_MAKEGRID) != payload["makegrid"]["sha256"]:
        raise ValueError("qualified MAKEGRID identity mismatch")
    payload["metric_cache"]["path"] = str(LOCAL_METRIC.resolve())
    payload["makegrid"]["path"] = str(LOCAL_MAKEGRID.resolve())
    payload["localization"] = {
        "source": str(SIDECAR.resolve()),
        "source_sha256": _sha256(SIDECAR),
        "only_changes": ["metric_cache.path", "makegrid.path"],
    }
    path = output / "localized_reference_sidecar.json"
    _write_json(path, payload)
    return path


def _contract() -> dict[str, Any]:
    """Return the independently derived continuum/discrete curvature contract."""

    return {
        "continuum_definition": "C(f)=K^alpha partial_alpha f; K=B/(2J) curl_alpha(b_cov/B)",
        "complete_equations": {
            "density": "2/B [C(n Te) - n C(phi)]",
            "Te": "4 Te/(3B) [C(n Te)/n + (5/2) C(Te) - C(phi)]",
            "Ti": "4 Ti/(3B) [C(n Te)/n - (5 tau/2) C(Ti) - C(phi)]",
            "vorticity": "2 B/n C(n Te + tau n Ti)",
            "Vi": "0",
            "Ve": "0",
        },
        "psi": "phi + tau Ti",
        "potential_remainder": {
            "coefficient": ["-2 n/B", "-4 Te/(3B)", "-4 Ti/(3B)", "0"],
            "operand": "C(psi)",
        },
        "material_matrix_rhs": [
            ["2 Te", "2 n", "2 n tau", "0"],
            ["4 Te^2/(3n)", "14 Te/3", "4 tau Te/3", "0"],
            ["4 Ti Te/(3n)", "4 Ti/3", "-2 tau Ti", "0"],
            ["2 B^2(Te+tau Ti)/n", "2 B^2", "2 tau B^2", "0"],
        ],
        "material_action": "M=(1/B) A_rhs(n,Te,Ti,B,tau) [C(n),C(Te),C(Ti),C(omega)]^T",
        "algebra": {
            "density": "M+R=2/B[C(nTe)-nC(phi)]",
            "Te": "M+R=4Te/(3B)[C(nTe)/n+5C(Te)/2-C(phi)]",
            "Ti": "M+R=4Ti/(3B)[C(nTe)/n-5tau C(Ti)/2-C(phi)]",
            "vorticity": "M+R=2B/n C(nTe+tau nTi)",
        },
        "production_discrete": {
            "face_coefficient": "Q^alpha=J K^alpha/B",
            "characteristic_normal": "Q^alpha/B=J K^alpha/B^2",
            "owner_evolution_measure": "sum_raw(raw_volume/B)",
            "thermodynamic_rows": "Q/B times A_rhs divided by V/B gives the intended K/B symbol",
            "vorticity_row": "the B^2 matrix row compensates Q/B and V/B, giving the intended B K symbol",
            "scalar_remainder": "B/J d_alpha[(J K^alpha/B) psi]",
            "coefficient_divergence_defect": "D=B/J d_alpha(J K^alpha/B); conservative C(f)=K.grad(f)+f D",
        },
        "measure_map": {
            "owner_observation": "raw-midpoint values weighted by physical raw_volume",
            "evolution_average": "integral(J/B source)/integral(J/B)",
            "reporting_target": "integral(J source)/integral(J)",
            "comparison": "all actions are compared with the physical target; the evolution-vs-physical source gap is reported separately",
        },
        "nonlinear_product_audit": {
            "production_remainder": "owner coefficient times discrete owner C(psi)",
            "continuum_reference": "average of pointwise coefficient times C(psi)",
            "not_equal_in_general": True,
            "Ti_chain_rule": "-(10 tau Ti/(3B)) C(Ti) equals -(5 tau/(3B)) C(Ti^2) only when the same pointwise product/derivative rule applies and coefficient variation is retained",
        },
        "boundary": {
            "physical_wall_model": "legacy-velocity-trace",
            "parallel_velocity_wall_bc": "neumann",
            "neumann_ghost_scheme": "physical",
            "thermodynamic": "physical homogeneous Neumann for these two fields",
            "phi_vorticity": "existing Dirichlet treatment",
            "parallel_boundary_pairing": "characteristic-sat",
            "parallel_characteristic_wall_law": "energy-absorbing",
        },
        "rho_star": "absent from curvature; it belongs to the Poisson bracket only",
        "units": "all four returned lanes are normalized RHS rates in the current dimensionless EB normalization",
    }


def _contract_algebra_check(samples: int = 64) -> float:
    rng = np.random.default_rng(2606)
    n = 0.8 + 0.4 * rng.random(samples)
    te = 0.8 + 0.4 * rng.random(samples)
    ti = 0.8 + 0.4 * rng.random(samples)
    b = 0.7 + 0.8 * rng.random(samples)
    gradients = rng.normal(size=(samples, 5))
    cn, cte, cti, _comega, cphi = gradients.T
    matrix = np.asarray(curvature_principal_matrix(n, te, ti, b, TAU))
    material = np.einsum("nij,nj->ni", matrix, gradients[:, :4]) / b[:, None]
    cpsi = cphi + TAU * cti
    remainder = np.column_stack(
        (-2 * n / b, -4 * te / (3 * b), -4 * ti / (3 * b), np.zeros(samples))
    ) * cpsi[:, None]
    cpe = te * cn + n * cte
    cpress = cpe + TAU * (ti * cn + n * cti)
    complete = np.column_stack(
        (
            2 * (cpe - n * cphi) / b,
            4 * te / (3 * b) * (cpe / n + 2.5 * cte - cphi),
            4 * ti / (3 * b) * (cpe / n - 2.5 * TAU * cti - cphi),
            2 * b * cpress / n,
        )
    )
    return float(np.max(np.abs(material + remainder - complete)))


def _heldout_fields(points: np.ndarray, eta_period: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return values/analytic logical gradients for the predeclared held-out state."""

    q = np.asarray(points, dtype=np.float64)
    u, theta, eta = q.T
    x = u * np.cos(theta)
    y = u * np.sin(theta)
    origin = 0.0
    zeta = 2.0 * np.pi * (eta - origin) / float(eta_period)
    k = 2.0 * np.pi / float(eta_period)
    radius = x * x + y * y
    a = 1.0 - radius
    envelope = a * a
    ex = -4.0 * x * a
    ey = -4.0 * y * a
    specifications = (
        (0.08 * x + 0.03 * y * np.sin(zeta), 0.08 + 0 * x, 0.03 * np.sin(zeta), 0.03 * y * np.cos(zeta)),
        (0.06 * y + 0.02 * x * y * np.cos(zeta), 0.02 * y * np.cos(zeta), 0.06 + 0.02 * x * np.cos(zeta), -0.02 * x * y * np.sin(zeta)),
        (0.05 * x * np.cos(zeta) + 0.02 * (x * x - y * y), 0.05 * np.cos(zeta) + 0.04 * x, -0.04 * y, -0.05 * x * np.sin(zeta)),
        (0.05 * (x * x - y * y) + 0.02 * x * np.sin(zeta), 0.10 * x + 0.02 * np.sin(zeta), -0.10 * y, 0.02 * x * np.cos(zeta)),
        (0.04 * x * y + 0.03 * y * np.cos(zeta), 0.04 * y, 0.04 * x + 0.03 * np.cos(zeta), -0.03 * y * np.sin(zeta)),
    )
    backgrounds = (1.0, 1.0, 1.0, 0.0, 0.0)
    values = np.empty((5, len(q)), dtype=np.float64)
    gradients = np.empty((5, len(q), 3), dtype=np.float64)
    for field, ((function, fx, fy, fz), background) in enumerate(zip(specifications, backgrounds, strict=True)):
        values[field] = background + envelope * function
        gx = ex * function + envelope * fx
        gy = ey * function + envelope * fy
        gz = envelope * fz
        gradients[field, :, 0] = gx * np.cos(theta) + gy * np.sin(theta)
        gradients[field, :, 1] = -u * gx * np.sin(theta) + u * gy * np.cos(theta)
        gradients[field, :, 2] = k * gz
    return values, gradients, np.asarray({"eta_origin": origin, "eta_period": eta_period}, dtype=object)


def _mms_fields(reference: Any, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = reference._fields_raw(np.asarray(points, dtype=np.float64), TIME_VALUE)
    constants = {"density": 1.0, "Te": 1.0, "Ti": 1.0, "vorticity": 0.0, "phi": 0.0}
    values = np.empty((5, len(points)), dtype=np.float64)
    gradients = np.empty((5, len(points), 3), dtype=np.float64)
    names = ("density", "Te", "Ti", "vorticity", "phi")
    for field, name in enumerate(names):
        if name == "vorticity":
            # Curvature's strict material symbol has a zero omega column.  The
            # corrected frozen owner omega is supplied separately for the live
            # action; no interpolated vorticity enters this independent source.
            values[field] = 0.0
            gradients[field] = 0.0
            continue
        payload = raw[name]
        values[field] = np.asarray(payload[0]) + constants[name]
        gradients[field] = np.stack(payload[1:4], axis=-1)
    return values, gradients


def _evaluate_fields(name: str, reference: Any, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if name == "corrected_frozen_mms":
        return _mms_fields(reference, points)
    if name == "regular_chart_heldout":
        values, gradients, _ = _heldout_fields(points, reference.eta_period)
        return values, gradients
    raise ValueError(name)


def _continuum_terms(values: np.ndarray, gradients: np.ndarray, prepared: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate independent pointwise M, R, total sources."""

    n, te, ti, _omega, phi = values
    curvature_derivatives = np.einsum("pi,fpi->fp", np.asarray(prepared.K), gradients)
    matrix = np.asarray(curvature_principal_matrix(n, te, ti, np.asarray(prepared.B), TAU))
    material = np.einsum("pij,jp->pi", matrix, curvature_derivatives[:4])
    material /= np.maximum(np.asarray(prepared.B)[:, None], 1.0e-30)
    cpsi = curvature_derivatives[4] + TAU * curvature_derivatives[2]
    remainder_coeff = np.column_stack(
        (
            -2.0 * n / prepared.B,
            -4.0 * te / (3.0 * prepared.B),
            -4.0 * ti / (3.0 * prepared.B),
            np.zeros_like(n),
        )
    )
    remainder = remainder_coeff * cpsi[:, None]
    return material, remainder, material + remainder


def _continuum_terms_directional(
    values: np.ndarray,
    gradients: np.ndarray,
    prepared: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return linear curvature contributions split by logical direction.

    The result arrays have shape ``(point, direction, equation)``.  This is an
    independent continuum split: each column of K is activated in turn while
    all state coefficients and B factors remain unchanged.
    """

    n, te, ti, _omega, _phi = values
    bmag = np.asarray(prepared.B)
    matrix = np.asarray(curvature_principal_matrix(n, te, ti, bmag, TAU))
    directional_derivatives = np.einsum(
        "pd,fpd->pdf", np.asarray(prepared.K), gradients
    )
    material = np.einsum(
        "pij,pdj->pdi", matrix, directional_derivatives[..., :4]
    ) / np.maximum(bmag[:, None, None], 1.0e-30)
    cpsi = directional_derivatives[..., 4] + TAU * directional_derivatives[..., 2]
    remainder_coeff = np.column_stack(
        (
            -2.0 * n / bmag,
            -4.0 * te / (3.0 * bmag),
            -4.0 * ti / (3.0 * bmag),
            np.zeros_like(n),
        )
    )
    remainder = cpsi[..., None] * remainder_coeff[:, None, :]
    return material, remainder, material + remainder


def _raw_points(context: Any) -> np.ndarray:
    return np.stack(
        np.meshgrid(context.x_centers, context.y_centers, context.z_centers, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)


def _raw_owner(context: Any) -> np.ndarray:
    owner_flat = np.asarray(context.arrays["owner_flat_ids"], dtype=np.int64)
    compact = np.full(context.resolution**3, -1, dtype=np.int64)
    compact[owner_flat] = np.arange(len(owner_flat), dtype=np.int64)
    result = compact[np.asarray(context.arrays["aggregate_id"], dtype=np.int64)]
    if np.any(result < 0):
        raise ValueError("raw topology contains a member without an active owner")
    return result


def _owner_observations(context: Any, raw_values: np.ndarray) -> np.ndarray:
    raw_owner = _raw_owner(context)
    weights = np.asarray(context.arrays["raw_volume"], dtype=np.float64)
    owner_count = len(context.arrays["owner_flat_ids"])
    volume = np.bincount(raw_owner, weights=weights, minlength=owner_count)
    result = np.empty((raw_values.shape[0], owner_count), dtype=np.float64)
    for field in range(raw_values.shape[0]):
        result[field] = np.bincount(
            raw_owner, weights=weights * raw_values[field], minlength=owner_count
        ) / volume
    return result


def _quantile_pick(candidates: np.ndarray, owner_keys: np.ndarray, limit: int = 4) -> list[int]:
    if len(candidates) <= limit:
        return [int(value) for value in candidates]
    ordered = candidates[np.lexsort((owner_keys[candidates, 1], owner_keys[candidates, 0], owner_keys[candidates, 2]))]
    slots = np.unique(np.rint(np.linspace(0, len(ordered) - 1, limit)).astype(int))
    return [int(ordered[index]) for index in slots]


def _select_owners(n: int, owner_keys: np.ndarray, masks: Mapping[str, np.ndarray]) -> tuple[np.ndarray, list[list[str]]]:
    """Geometry-only deterministic, eta-spread, deduplicated selections."""

    active = np.ones(len(owner_keys), dtype=bool)
    ordinary = np.asarray(masks["ordinary"], dtype=bool)
    categories = (
        ("axis_adjacent", active & (owner_keys[:, 0] <= 1)),
        ("agglomerated_bulk", np.asarray(masks["agglomerated_interior"], dtype=bool)),
        ("size_transition", np.asarray(masks["true_size_change_interface"], dtype=bool)),
        ("ordinary_interior", ordinary & (owner_keys[:, 0] > 2) & (owner_keys[:, 0] < n - 2)),
        ("radial_wall", owner_keys[:, 0] == n - 1),
        ("theta_seam", ordinary & (owner_keys[:, 1] == 0)),
        ("eta_seam", ordinary & (owner_keys[:, 2] == 0)),
    )
    selected: list[int] = []
    labels: dict[int, list[str]] = {}
    for name, predicate in categories:
        picks = _quantile_pick(np.flatnonzero(predicate), owner_keys)
        for index in picks:
            labels.setdefault(index, []).append(name)
            if index not in selected:
                selected.append(index)
    selected.sort(key=lambda index: tuple(int(value) for value in owner_keys[index]))
    return np.asarray(selected, dtype=np.int64), [labels[index] for index in selected]


def _support(n: int, owner_indices: np.ndarray, raw_owner: np.ndarray) -> dict[str, np.ndarray]:
    raw_parts = [np.flatnonzero(raw_owner == int(owner)) for owner in owner_indices]
    raw_indices = np.concatenate(raw_parts).astype(np.int64)
    raw_local_owner = np.concatenate(
        [np.full(len(part), row, dtype=np.int32) for row, part in enumerate(raw_parts)]
    )
    raw_keys = numerics._raw_keys(n, raw_indices)
    global_incidence = np.empty((len(raw_indices), 3, 2), dtype=np.int64)
    all_faces = []
    for axis in range(3):
        lower = raw_keys.copy()
        upper = raw_keys.copy()
        upper[:, axis] += 1
        global_incidence[:, axis, 0] = numerics._face_index(n, axis, *lower.T)
        global_incidence[:, axis, 1] = numerics._face_index(n, axis, *upper.T)
        all_faces.extend((global_incidence[:, axis, 0], global_incidence[:, axis, 1]))
    face_indices = np.unique(np.concatenate(all_faces)).astype(np.int64)
    return {
        "raw_indices": raw_indices,
        "raw_keys": raw_keys,
        "raw_local_owner": raw_local_owner,
        "face_indices": face_indices,
        "face_keys": numerics._face_keys(n, face_indices),
        "face_incidence": np.searchsorted(face_indices, global_incidence).astype(np.int32),
    }


def _cell_quadrature(context: Any, raw_keys: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    points = []
    weights = []
    faces = (context.x_faces, context.y_faces, context.z_faces)
    for key in raw_keys:
        q, w = matched._cell_points_weights(faces, key, order)
        points.append(q)
        weights.append(w)
    return np.stack(points), np.stack(weights)


def _face_quadrature(context: Any, face_keys: np.ndarray, order: int = 3) -> tuple[np.ndarray, np.ndarray]:
    points = []
    weights = []
    faces = (context.x_faces, context.y_faces, context.z_faces)
    for key in face_keys:
        q, w = matched._face_points_weights(faces, key, order)
        points.append(q)
        weights.append(w)
    return np.stack(points), np.stack(weights)


def _fit_volume_state(context: Any, points: np.ndarray, raw_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Evaluate one frozen central selection-v3 cubic per selected raw cell."""

    values = np.empty((len(raw_keys), points.shape[1], 5), dtype=np.float64)
    gradients = np.empty((len(raw_keys), points.shape[1], 5, 3), dtype=np.float64)
    fallback = np.empty(len(raw_keys), dtype=np.uint8)
    condition = np.empty(len(raw_keys), dtype=np.float64)
    reproduction = np.empty(len(raw_keys), dtype=np.float64)
    for row, (key, nodes) in enumerate(zip(raw_keys, points, strict=True)):
        fit = numerics._fit_entity(
            context,
            np.asarray((context.x_centers[key[0]], context.y_centers[key[1]], context.z_centers[key[2]])),
            axis=0,
            eta_index=int(key[2]),
            owner_values=np.asarray(context.arrays["owner_values"]),
        )
        value, gradient = numerics._evaluate_fit(fit, nodes, context.eta_period)
        values[row] = value.T
        gradients[row] = np.moveaxis(gradient, 0, 1)
        fallback[row] = int(fit.diagnostics["fallback_level"])
        condition[row] = float(fit.diagnostics["condition"])
        reproduction[row] = float(fit.diagnostics["reproduction"])
    return values, gradients, {"fallback": fallback, "condition": condition, "reproduction": reproduction}


def _linear_volume_state(
    centers: np.ndarray,
    context: Any,
    points: np.ndarray,
    raw_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Current H-state linear control used only by geometry/integration G."""

    n = context.resolution
    shaped = centers.reshape(5, n, n, n)
    gradients = np.empty((5, n, n, n, 3), dtype=np.float64)
    spacings = (context.x_centers, context.y_centers, context.z_centers)
    for field in range(5):
        gradients[field, ..., 0] = np.gradient(shaped[field], spacings[0], axis=0, edge_order=2)
        gradients[field, ..., 1] = (
            np.roll(shaped[field], -1, axis=1) - np.roll(shaped[field], 1, axis=1)
        ) / (2.0 * context.dtheta)
        gradients[field, ..., 2] = (
            np.roll(shaped[field], -1, axis=2) - np.roll(shaped[field], 1, axis=2)
        ) / (2.0 * context.deta)
    values = np.empty((len(raw_keys), points.shape[1], 5), dtype=np.float64)
    grads = np.empty((len(raw_keys), points.shape[1], 5, 3), dtype=np.float64)
    for row, (i, j, k) in enumerate(raw_keys):
        center = np.asarray((context.x_centers[i], context.y_centers[j], context.z_centers[k]))
        delta = points[row] - center
        delta[:, 1] = (delta[:, 1] + np.pi) % (2 * np.pi) - np.pi
        delta[:, 2] = (delta[:, 2] + 0.5 * context.eta_period) % context.eta_period - 0.5 * context.eta_period
        local_gradient = gradients[:, i, j, k]
        values[row] = shaped[:, i, j, k, None].T + delta @ local_gradient.T
        grads[row] = np.broadcast_to(local_gradient[None, :, :], grads[row].shape)
    return values, grads


def _integrate_sources(
    values: np.ndarray,
    gradients: np.ndarray,
    prepared: Any,
    logical_weights: np.ndarray,
    raw_local_owner: np.ndarray,
    owner_count: int,
) -> dict[str, np.ndarray]:
    raw_count, nq = logical_weights.shape
    material, remainder, total = _continuum_terms(
        values.reshape(raw_count * nq, 5).T,
        gradients.reshape(raw_count * nq, 5, 3).transpose(1, 0, 2),
        prepared,
    )
    terms = {
        "material": material.reshape(raw_count, nq, 4),
        "remainder": remainder.reshape(raw_count, nq, 4),
        "total": total.reshape(raw_count, nq, 4),
    }
    directional = _continuum_terms_directional(
        values.reshape(raw_count * nq, 5).T,
        gradients.reshape(raw_count * nq, 5, 3).transpose(1, 0, 2),
        prepared,
    )
    directional_terms = {
        "material": directional[0].reshape(raw_count, nq, 3, 4),
        "remainder": directional[1].reshape(raw_count, nq, 3, 4),
        "total": directional[2].reshape(raw_count, nq, 3, 4),
    }
    J = np.abs(np.asarray(prepared.J)).reshape(raw_count, nq)
    B = np.asarray(prepared.B).reshape(raw_count, nq)
    result: dict[str, np.ndarray] = {}
    for measure_name, measure in (
        ("physical", logical_weights * J),
        ("evolution", logical_weights * J / np.maximum(B, 1.0e-30)),
    ):
        denominator = np.bincount(
            raw_local_owner,
            weights=np.sum(measure, axis=1),
            minlength=owner_count,
        )
        for term, pointwise in terms.items():
            owner = np.zeros((owner_count, 4), dtype=np.float64)
            for equation in range(4):
                owner[:, equation] = np.bincount(
                    raw_local_owner,
                    weights=np.sum(measure * pointwise[..., equation], axis=1),
                    minlength=owner_count,
                ) / denominator
            result[f"{measure_name}:{term}"] = owner
            owner_directional = np.zeros((owner_count, 3, 4), dtype=np.float64)
            for direction in range(3):
                for equation in range(4):
                    owner_directional[:, direction, equation] = np.bincount(
                        raw_local_owner,
                        weights=np.sum(
                            measure * directional_terms[term][..., direction, equation],
                            axis=1,
                        ),
                        minlength=owner_count,
                    ) / denominator
            result[f"{measure_name}:{term}:directional"] = owner_directional
        result[f"{measure_name}:volume"] = denominator
    return result


def _stratified_owner_rows(labels: Sequence[Sequence[str]]) -> tuple[np.ndarray, dict[str, int]]:
    """Pick the first frozen owner covering each declared category."""

    category_row: dict[str, int] = {}
    for row, item in enumerate(labels):
        for category in item:
            category_row.setdefault(category, row)
    rows = np.asarray(sorted(set(category_row.values())), dtype=np.int64)
    return rows, category_row


def _geometry_step_sensitivity(
    reference: Any,
    context: Any,
    support: Mapping[str, np.ndarray],
    labels: Sequence[Sequence[str]],
    field_name: str,
) -> dict[str, Any]:
    """Bound K's geometry-derivative step on one frozen owner/category."""

    owner_rows, category_row = _stratified_owner_rows(labels)
    raw_mask = np.isin(support["raw_local_owner"], owner_rows)
    raw_keys = support["raw_keys"][raw_mask]
    old_owner = support["raw_local_owner"][raw_mask]
    row_map = {int(owner): row for row, owner in enumerate(owner_rows)}
    local_owner = np.asarray([row_map[int(value)] for value in old_owner], dtype=np.int32)
    points, weights = _cell_quadrature(context, raw_keys, 3)
    values, gradients = _evaluate_fields(field_name, reference, points.reshape(-1, 3))
    values = values.T.reshape(len(points), points.shape[1], 5)
    gradients = gradients.transpose(1, 0, 2).reshape(len(points), points.shape[1], 5, 3)
    original_step = float(reference.finite_difference_step)
    integrated: dict[str, np.ndarray] = {}
    try:
        for name, factor in (("half", 0.5), ("default", 1.0), ("double", 2.0)):
            reference.finite_difference_step = original_step * factor
            prepared = reference.prepare(points.reshape(-1, 3))
            integrated[name] = _integrate_sources(
                values, gradients, prepared, weights, local_owner, len(owner_rows)
            )["physical:total"]
    finally:
        reference.finite_difference_step = original_step
    difference = np.maximum(
        np.abs(integrated["half"] - integrated["default"]),
        np.abs(integrated["double"] - integrated["default"]),
    )
    return {
        "owner_rows": owner_rows,
        "category_owner_row": category_row,
        "base_step": original_step,
        "half": integrated["half"],
        "default": integrated["default"],
        "double": integrated["double"],
        "max_difference": difference,
    }


def _coefficient_divergence_diagnostic(reference: Any, points: np.ndarray) -> dict[str, Any]:
    """Evaluate B/J div(J K/B) at a bounded set of regular points."""

    q = np.asarray(points, dtype=np.float64)

    def flux(x: np.ndarray) -> np.ndarray:
        metric = reference._metric(x)
        curvature = reference._curvature(x)
        return metric["J"][:, None] * curvature / np.maximum(metric["B"][:, None], 1.0e-30)

    divergence = np.zeros(len(q), dtype=np.float64)
    for axis in range(3):
        divergence += reference._derivative(
            lambda x, axis=axis: flux(x)[:, axis], q, axis
        )
    metric = reference._metric(q)
    defect = metric["B"] * divergence / np.maximum(np.abs(metric["J"]), 1.0e-30)
    return {
        "points": q,
        "B_over_J_div_JK_over_B": defect,
        "max_abs": float(np.max(np.abs(defect))),
        "rms": float(np.sqrt(np.mean(defect * defect))),
        "interpretation": "tracked conservative-vs-advective coefficient defect; not assumed zero",
    }


def _principal_matrix_numpy(state: np.ndarray, bmag: np.ndarray) -> np.ndarray:
    n, te, ti = np.moveaxis(state[..., :3], -1, 0)
    shape = state.shape[:-1]
    matrix = np.zeros(shape + (4, 4), dtype=np.float64)
    matrix[..., 0, 0] = 2 * te
    matrix[..., 0, 1] = 2 * n
    matrix[..., 0, 2] = 2 * n * TAU
    matrix[..., 1, 0] = 4 * te * te / (3 * n)
    matrix[..., 1, 1] = 14 * te / 3
    matrix[..., 1, 2] = 4 * TAU * te / 3
    matrix[..., 2, 0] = 4 * ti * te / (3 * n)
    matrix[..., 2, 1] = 4 * ti / 3
    matrix[..., 2, 2] = -2 * TAU * ti
    matrix[..., 3, 0] = 2 * bmag * bmag * (te + TAU * ti) / n
    matrix[..., 3, 1] = 2 * bmag * bmag
    matrix[..., 3, 2] = 2 * TAU * bmag * bmag
    return matrix


def _absolute_action(matrix: np.ndarray, jump: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat_matrix = matrix.reshape(-1, 4, 4)
    flat_jump = jump.reshape(-1, 4)
    result = np.empty_like(flat_jump)
    fallback = np.zeros(len(flat_jump), dtype=bool)
    for row, (operator, vector) in enumerate(zip(flat_matrix, flat_jump, strict=True)):
        eigenvalues, eigenvectors = np.linalg.eig(operator)
        condition = np.linalg.cond(eigenvectors)
        valid = (
            np.all(np.isfinite(operator))
            and np.max(np.abs(np.imag(eigenvalues))) <= 1.0e-10 * (1 + np.max(np.abs(np.real(eigenvalues))))
            and np.isfinite(condition)
            and condition <= 1.0e8
        )
        if valid:
            result[row] = np.real(eigenvectors @ (np.abs(np.real(eigenvalues)) * (np.linalg.inv(eigenvectors) @ vector)))
        else:
            result[row] = np.linalg.norm(operator) * vector
            fallback[row] = True
    return result.reshape(jump.shape), fallback.reshape(jump.shape[:-1])


def _face_neighbors(n: int, key: Sequence[int]) -> tuple[tuple[int, int, int] | None, tuple[int, int, int] | None]:
    axis, i, j, k = map(int, key)
    if axis == 0:
        return ((i - 1, j, k) if i > 0 else None, (i, j, k) if i < n else None)
    if axis == 1:
        return (i, (j - 1) % n, k), (i, j % n, k)
    return (i, j, (k - 1) % n), (i, j, k % n)


def _expected_donor_hashes(n: int, face_indices: np.ndarray) -> np.ndarray:
    return p05_nodewise._expected_donor_hashes(n, face_indices)


def _live_side_donor_hash(
    context: Any,
    point: np.ndarray,
    *,
    axis: int,
    eta_index: int,
) -> tuple[str, int, int]:
    """Replay the frozen selection-v3 donor choice on the current platform."""

    for level, policy in enumerate(cubic.POLICY["deficient_row_expansion_schedule"]):
        donors, _weights, diagnostics, tie_count = cubic._row_batch(
            context,
            int(axis),
            int(eta_index) % context.resolution,
            np.asarray(point, dtype=np.float64)[None, :],
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
            digest = hashlib.sha256(np.asarray(donors[0], dtype="<i8").tobytes()).hexdigest()
            return digest, int(level), int(tie_count)
    raise RuntimeError("qualified adaptive central support is deficient")


def _curvature_face_geometry(reference: Any, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate J, B and K at face nodes, including the exact radial wall.

    The general reference ``prepare`` method also constructs perpendicular
    metric derivatives with centered radial stencils, which are intentionally
    defined only for interior points.  Curvature needs only J/B/K.  At u=1 we
    retain the exact face location and use the standard fourth-order backward
    radial derivative for the covariant field entering curl(b_cov/B); angular
    derivatives remain the reference's fourth-order centered periodic stencil.
    """

    q = np.asarray(points, dtype=np.float64)
    metric = reference._metric(q)
    K = np.empty((len(q), 3), dtype=np.float64)
    wall = np.isclose(q[:, 0], 1.0, rtol=0.0, atol=8.0 * np.finfo(float).eps)
    interior = ~wall
    if np.any(interior):
        K[interior] = reference._curvature(q[interior])
    if np.any(wall):
        qw = q[wall]

        def covariant_over_B(x: np.ndarray) -> np.ndarray:
            local = reference._metric(x)
            return local["bcov"] / local["B"][:, None]

        h = min(float(reference.finite_difference_step), 0.05)
        radial_samples = []
        for offset in range(5):
            shifted = qw.copy()
            shifted[:, 0] -= offset * h
            radial_samples.append(covariant_over_B(shifted))
        du = (
            25.0 * radial_samples[0]
            - 48.0 * radial_samples[1]
            + 36.0 * radial_samples[2]
            - 16.0 * radial_samples[3]
            + 3.0 * radial_samples[4]
        ) / (12.0 * h)
        dtheta = reference._derivative(covariant_over_B, qw, 1)
        deta = reference._derivative(covariant_over_B, qw, 2)
        curl = np.stack(
            (
                dtheta[..., 2] - deta[..., 1],
                deta[..., 0] - du[..., 2],
                du[..., 1] - dtheta[..., 0],
            ),
            axis=-1,
        )
        K[wall] = (
            0.5
            * metric["B"][wall, None]
            * curl
            / np.maximum(np.abs(metric["J"])[wall, None], 1.0e-30)
        )
    return np.asarray(metric["J"]), np.asarray(metric["B"]), K


def _interface_correction(
    context: Any,
    support: Mapping[str, np.ndarray],
    reference: Any,
    central_face_values: np.ndarray,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Assemble recentered characteristic face fluctuations on selected owners."""

    keys = support["face_keys"]
    points, weights = _face_quadrature(context, keys, order=3)
    # Collapsed u=0 faces carry exactly zero measure/action in the production
    # topology.  Do not send their axis points through the ordinary toroidal
    # metric evaluator; fill harmless placeholders and zero them below.
    collapsed = (keys[:, 0] == 0) & (keys[:, 1] == 0)
    regular = ~collapsed
    regular_J, regular_B, regular_K = _curvature_face_geometry(
        reference, points[regular].reshape(-1, 3)
    )
    J = np.zeros(points.shape[:2], dtype=np.float64)
    B = np.ones(points.shape[:2], dtype=np.float64)
    K = np.zeros(points.shape[:2] + (3,), dtype=np.float64)
    J[regular] = regular_J.reshape((-1, points.shape[1]))
    B[regular] = regular_B.reshape((-1, points.shape[1]))
    K[regular] = regular_K.reshape((-1, points.shape[1], 3))
    normal = J * K[np.arange(len(keys))[:, None], np.arange(points.shape[1])[None, :], keys[:, 0, None]] / np.maximum(B * B, 1.0e-30)
    left = central_face_values[..., :4].copy()
    right = central_face_values[..., :4].copy()
    donor_fallback = []
    side_reproduction = []
    donor_hash_mismatches: list[dict[str, Any]] = []
    if mode == "U":
        expected = _expected_donor_hashes(context.resolution, support["face_indices"])
        for row, key in enumerate(keys):
            axis, i, _j, k = map(int, key)
            if (axis == 0 and i in (0, context.resolution)):
                continue
            point = points[row, points.shape[1] // 2]
            try:
                pair = p05_nodewise._fit_side_pair(
                    context, point, points[row], axis=axis, eta_index=k,
                    expected_donor_hash=str(expected[row]),
                )
            except ValueError as error:
                if "selection-v3 donor mismatch" not in str(error):
                    raise
                live_hash, live_level, tie_count = _live_side_donor_hash(
                    context, point, axis=axis, eta_index=k
                )
                pair = p05_nodewise._fit_side_pair(
                    context, point, points[row], axis=axis, eta_index=k,
                    expected_donor_hash=live_hash,
                )
                donor_hash_mismatches.append(
                    {
                        "face_row": int(row),
                        "face_index": int(support["face_indices"][row]),
                        "face_key": np.asarray(key, dtype=np.int64),
                        "frozen_campaign_hash": str(expected[row]),
                        "live_platform_hash": live_hash,
                        "fallback_level": live_level,
                        "indexed_query_tie_fallback_count": tie_count,
                        "classification": "cross-platform exact-radius tie; live frozen-code replay retained",
                    }
                )
            jump = (pair.right_values[:4] - pair.left_values[:4]).T
            left[row] -= 0.5 * jump
            right[row] += 0.5 * jump
            donor_fallback.append(pair.diagnostics["fallback_level"])
            side_reproduction.append(pair.diagnostics["polynomial_value_reproduction_max"])
    # C and U both retain the physical-wall characteristic branch.  The
    # smooth fields have thermodynamic Neumann compatibility and zero
    # phi/omega boundary values; the strict material wall trace is therefore
    # the thermodynamic point state with stationary interior omega.
    wall_rows = np.flatnonzero((keys[:, 0] == 0) & (keys[:, 1] == context.resolution))
    for row in wall_rows:
        interior = central_face_values[row, :, :4]
        # Reuse the exact production wall solve; no exterior polynomial fit.
        exterior, working, _fallback = _curvature_bc_characteristic_wall_states(
            blob.jnp.asarray(interior), blob.jnp.asarray(interior), blob.jnp.asarray(B[row]), TAU,
            blob.jnp.asarray(normal[row]), interior_on_right=False, positivity_floor=FLOOR,
        )
        left[row] = interior
        right[row] = np.asarray(exterior)
        central_face_values[row, :, :4] = np.asarray(working)
    matrix = _principal_matrix_numpy(central_face_values[..., :4], B)
    flux_matrix = -normal[..., None, None] * matrix
    jump = right - left
    absolute, spectral_fallback = _absolute_action(flux_matrix, jump)
    material = np.einsum("fqij,fqj->fqi", flux_matrix, jump)
    dplus = 0.5 * (material + absolute)
    dminus = 0.5 * (material - absolute)
    dplus[collapsed] = 0.0
    dminus[collapsed] = 0.0
    integrated_plus = -np.sum(weights[..., None] * dplus, axis=1)
    integrated_minus = -np.sum(weights[..., None] * dminus, axis=1)
    raw_integrated = np.zeros((len(support["raw_indices"]), 4), dtype=np.float64)
    raw_directional = np.zeros((len(support["raw_indices"]), 3, 4), dtype=np.float64)
    raw_boundary = np.zeros((len(support["raw_indices"]), 4), dtype=np.float64)
    raw_lookup = {tuple(map(int, key)): row for row, key in enumerate(support["raw_keys"])}
    for face, key in enumerate(keys):
        axis = int(key[0])
        is_wall = bool(axis == 0 and int(key[1]) == context.resolution)
        minus, plus = _face_neighbors(context.resolution, key)
        if minus in raw_lookup:
            raw_integrated[raw_lookup[minus]] += integrated_minus[face]
            raw_directional[raw_lookup[minus], axis] += integrated_minus[face]
            if is_wall:
                raw_boundary[raw_lookup[minus]] += integrated_minus[face]
        if plus in raw_lookup:
            raw_integrated[raw_lookup[plus]] += integrated_plus[face]
            raw_directional[raw_lookup[plus], axis] += integrated_plus[face]
            if is_wall:
                raw_boundary[raw_lookup[plus]] += integrated_plus[face]
    owner_count = int(np.max(support["raw_local_owner"])) + 1
    owner_integrated = np.zeros((owner_count, 4), dtype=np.float64)
    owner_directional = np.zeros((owner_count, 3, 4), dtype=np.float64)
    owner_boundary = np.zeros((owner_count, 4), dtype=np.float64)
    np.add.at(owner_integrated, support["raw_local_owner"], raw_integrated)
    np.add.at(owner_directional, support["raw_local_owner"], raw_directional)
    np.add.at(owner_boundary, support["raw_local_owner"], raw_boundary)
    diagnostics = {
        "spectral_fallback_count": int(np.count_nonzero(spectral_fallback)),
        "positivity_fallback_count": int(np.count_nonzero(central_face_values[..., :3] <= FLOOR)),
        "wall_face_count": int(len(wall_rows)),
        "interior_face_count": int(np.count_nonzero(~collapsed) - len(wall_rows)),
        "side_fit_fallback_max": int(max(donor_fallback, default=0)),
        "side_fit_reproduction_max": float(max(side_reproduction, default=0.0)),
        "frozen_donor_hash_mismatch_count": int(len(donor_hash_mismatches)),
        "frozen_donor_hash_mismatches": donor_hash_mismatches,
        "jump_rms": float(np.sqrt(np.mean(jump * jump))),
        "jump_max": float(np.max(np.abs(jump))),
    }
    return (
        owner_integrated,
        owner_directional,
        owner_boundary,
        np.stack((integrated_minus, integrated_plus), axis=1),
        diagnostics,
    )


def _owner_state_arrays(owner_keys: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    result = np.zeros((values.shape[0], n, n, n), dtype=np.float64)
    result[:, owner_keys[:, 0], owner_keys[:, 1], owner_keys[:, 2]] = values
    return result


def _production_action(
    n: int,
    artifact: Any,
    owner_keys: np.ndarray,
    owner_values: np.ndarray,
    reference: Any,
    runtime_bundle: tuple[Any, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray, tuple[Any, Any]]:
    """Run the genuine frozen production curvature action for one state."""

    if runtime_bundle is None:
        selector_args = mms._production_selector_args()
        blob._validate_flux_framework(selector_args)
        blob._configure_runtime_selectors(selector_args)
        args = SimpleNamespace(
            shard_counts=(1, 1, 1),
            curvature_edge_one_form=False,
            reference=reference,
            metric_context=SimpleNamespace(
                metric_evaluator=reference.metric_evaluator,
                bfield=reference.bfield_evaluator,
                nfp=int(artifact.nfp),
            ),
            advance_execution="compiled",
        )
        started_setup = time.perf_counter()
        runtime = mms._runtime(artifact.global_geometry, artifact.owner_geometry, args)
        setup_seconds = time.perf_counter() - started_setup
        model = runtime.model
        run = None
    else:
        model, run = runtime_bundle
        setup_seconds = 0.0
    arrays = _owner_state_arrays(owner_keys, owner_values, n)
    state = blob.FciDrbEBState(
        density=blob.jnp.asarray(arrays[0]),
        phi=blob.jnp.asarray(arrays[4]),
        Te=blob.jnp.asarray(arrays[1]),
        Ti=blob.jnp.asarray(arrays[2]),
        Vi=blob.jnp.zeros((n, n, n), dtype=blob.jnp.float64),
        Ve=blob.jnp.zeros((n, n, n), dtype=blob.jnp.float64),
        vorticity=blob.jnp.asarray(arrays[3]),
    )
    material_state = state.replace(phi=-TAU * state.Ti)
    zeros = state.zeros_like()
    if run is None:
        run = blob.jax.jit(
            lambda q: model.evaluate_stage(
                q,
                source_owned=zeros,
                phi_owned=q.phi,
                return_curvature_component_fields=True,
            )[1]
        )
    started_application = time.perf_counter()
    material = run(material_state)
    total = run(state)
    material, total = blob.jax.block_until_ready((material, total))
    application_seconds = time.perf_counter() - started_application
    material = np.asarray(material)
    total = np.asarray(total)
    remainder = total - material
    # Return fine H centers as the geometry/integration-only representation.
    face_bc = model._face_bcs(state)
    h_centers = []
    for value, boundary in (
        (state.density, face_bc.density),
        (state.Te, face_bc.Te),
        (state.Ti, face_bc.Ti),
        (state.vorticity, face_bc.vorticity),
        (state.phi, face_bc.phi),
    ):
        halo = model._prepare_rlp_reconstructed_halo(value, boundary)
        h_centers.append(np.asarray(halo[model.domain.layout.owned_slices_cell]).reshape(-1))
    diagnostics = {
        "setup_seconds": setup_seconds,
        "application_seconds": application_seconds,
        "peak_gib": _peak_gib(),
        "runtime_reused": runtime_bundle is not None,
        "execution": "genuine jitted LocalFciDrbEBRhs.evaluate_stage curvature component path",
        "directional_sum_max": float(np.max(np.abs(np.sum(total, axis=1) - np.sum(material + remainder, axis=1)))),
    }
    return material, remainder, total, diagnostics, np.stack(h_centers), (model, run)


def _metrics(action: np.ndarray, target: np.ndarray, volume: np.ndarray) -> dict[str, Any]:
    error = action - target
    total_volume = float(np.sum(volume))
    return {
        "L2_by_equation": np.sqrt(np.sum(volume[:, None] * error * error, axis=0) / total_volume),
        "max_abs_by_equation": np.max(np.abs(error), axis=0),
        "signed_mean_by_equation": np.sum(volume[:, None] * error, axis=0) / total_volume,
        "SSE_by_equation": np.sum(volume[:, None] * error * error, axis=0),
    }


def _region_metrics(action: np.ndarray, target: np.ndarray, volume: np.ndarray, labels: Sequence[Sequence[str]]) -> dict[str, Any]:
    names = sorted({name for item in labels for name in item})
    return {
        name: _metrics(action[mask], target[mask], volume[mask])
        for name in names
        for mask in [np.asarray([name in item for item in labels], dtype=bool)]
        if np.any(mask)
    }


def run_preflight(output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    _progress(output, "preflight_load")
    contract = _contract()
    algebra = _contract_algebra_check()
    context = cubic._load_context(GEOMETRY, BASELINE, 32)
    sidecar = _localized_sidecar(output)
    reference = build_continuum_reference_from_sidecar(sidecar, verify_hashes=True)
    result = _load_npz(CAMPAIGN / "N32.npz")
    material = _load_npz(MATERIAL_CAMPAIGN / "N32.npz")
    owner_keys = result["owner_keys"].astype(np.int64)
    masks = {name.split(":", 1)[1]: value for name, value in material.items() if name.startswith("region:")}
    selected, labels = _select_owners(32, owner_keys, masks)
    raw_points = _raw_points(context)
    raw_values, _ = _evaluate_fields("regular_chart_heldout", reference, raw_points)
    owner_values = _owner_observations(context, raw_values)
    context.arrays["owner_values"] = owner_values
    support = _support(32, selected[: min(4, len(selected))], _raw_owner(context))
    points, weights = _cell_quadrature(context, support["raw_keys"], 3)
    fit_values, fit_gradients, fit_diag = _fit_volume_state(context, points, support["raw_keys"])
    prepared = reference.prepare(points.reshape(-1, 3))
    integrated_terms = _integrate_sources(
        fit_values,
        fit_gradients,
        prepared,
        weights,
        support["raw_local_owner"],
        len(np.unique(support["raw_local_owner"])),
    )
    wall_points = np.column_stack(
        (np.ones(9), np.linspace(0, 2 * np.pi, 9, endpoint=False), np.linspace(0, reference.eta_period, 9, endpoint=False))
    )
    heldout_wall_values, heldout_wall_gradients, _ = _heldout_fields(wall_points, reference.eta_period)
    report = {
        "schema": f"{SCHEMA}.preflight",
        "status": "passed",
        "contract": contract,
        "checks": {
            "material_plus_remainder_algebra_max": algebra,
            "heldout_min_positive": float(np.min(raw_values[:3])),
            "heldout_wall_thermodynamic_normal_gradient_max": float(np.max(np.abs(heldout_wall_gradients[:3, :, 0]))),
            "heldout_wall_phi_omega_value_max": float(np.max(np.abs(heldout_wall_values[3:5]))),
            "cubic_reproduction_max": float(np.max(fit_diag["reproduction"])),
            "cubic_fallback_max": int(np.max(fit_diag["fallback"])),
            "finite_integrated_terms": all(np.all(np.isfinite(value)) for value in integrated_terms.values()),
        },
        "selection_preview": {"owner_indices": selected[:4], "owner_keys": owner_keys[selected[:4]], "labels": labels[:4]},
        "timing": {"seconds": time.perf_counter() - started, "peak_gib": _peak_gib()},
        "identities": _provenance(32),
    }
    if algebra > 2.0e-12:
        raise ValueError(f"continuum decomposition algebra failed: {algebra}")
    if report["checks"]["heldout_min_positive"] <= 0.0:
        raise ValueError("heldout state is not positive")
    if report["checks"]["heldout_wall_thermodynamic_normal_gradient_max"] > 2.0e-13:
        raise ValueError("heldout thermodynamic wall compatibility failed")
    if report["checks"]["heldout_wall_phi_omega_value_max"] > 2.0e-13:
        raise ValueError("heldout Dirichlet wall compatibility failed")
    if report["checks"]["cubic_reproduction_max"] > 2.0e-9:
        raise ValueError("selection-v3 cubic preflight failed")
    _write_json(output / "preflight.json", report)
    _write_json(output / "contract.json", contract)
    _progress(output, "preflight_complete", elapsed_seconds=report["timing"]["seconds"], peak_gib=report["timing"]["peak_gib"])
    return report


def _provenance(n: int) -> dict[str, Any]:
    geometry_manifest = GEOMETRY / f"{n}x{n}x{n}/manifest.json"
    return {
        "implementation_hash": _implementation_hash(),
        "git_head": os.popen(f"git -C '{REPO}' rev-parse HEAD").read().strip(),
        "sources": {
            "driver": _identity(Path(__file__)),
            "plan": _identity(REPO / "src/drbx/dev_docs/p06_curvature_bounded_plan.md"),
            "production_rhs": _identity(REPO / "src/drbx/native/fci_drb_EB_rhs.py"),
            "production_operator": _identity(REPO / "src/drbx/native/fci_operators.py"),
            "characteristic_flux": _identity(REPO / "src/drbx/native/fci_curvature_production_flux.py"),
            "frozen_selection": _identity(REPO / "scripts/hsx_remote_qualification/selection.py"),
            "frozen_numerics": _identity(REPO / "scripts/hsx_remote_qualification/numerics.py"),
            "p05_side_adapter": _identity(P05_NODEWISE),
            "reference": _identity(REPO / "hsx_mms_continuum_reference.py"),
        },
        "inputs": {
            "geometry_manifest": _identity(geometry_manifest),
            "centered_campaign": _identity(CAMPAIGN / f"N{n}.npz"),
            "centered_prepare": _identity(CAMPAIGN / f"N{n}.prepare.npz"),
            "material_campaign": _identity(MATERIAL_CAMPAIGN / f"N{n}.npz"),
            "reference_sidecar": _identity(SIDECAR),
            "metric": _identity(LOCAL_METRIC),
            "makegrid": _identity(LOCAL_MAKEGRID),
        },
        "configuration": {
            "resolution": n,
            "tau": TAU,
            "time": TIME_VALUE,
            "bias": BIAS,
            "selection": "geometry-only eta-spread up to four per declared category; deduplicated before errors",
            "owner_observation": "raw midpoint physical-volume weighted",
            "reference_orders": list(REFERENCE_ORDERS),
            "boundary": _contract()["boundary"],
        },
    }


def run_resolution(output: Path, n: int) -> dict[str, Any]:
    started = time.perf_counter()
    if not (output / "preflight.json").is_file():
        raise RuntimeError("validated P06 preflight is required")
    _progress(output, f"N{n}_load")
    context = cubic._load_context(GEOMETRY, BASELINE, n)
    reference = build_continuum_reference_from_sidecar(_localized_sidecar(output), verify_hashes=True)
    result = _load_npz(CAMPAIGN / f"N{n}.npz")
    material_campaign = _load_npz(MATERIAL_CAMPAIGN / f"N{n}.npz")
    owner_keys = result["owner_keys"].astype(np.int64)
    masks = {name.split(":", 1)[1]: value for name, value in material_campaign.items() if name.startswith("region:")}
    owner_indices, labels = _select_owners(n, owner_keys, masks)
    support = _support(n, owner_indices, _raw_owner(context))
    selection = {
        "schema": f"{SCHEMA}.selection",
        "resolution": n,
        "owner_indices": owner_indices,
        "owner_keys": owner_keys[owner_indices],
        "labels": labels,
        "raw_indices": support["raw_indices"],
        "raw_keys": support["raw_keys"],
        "face_indices": support["face_indices"],
        "face_keys": support["face_keys"],
        "selection_hash": _array_hash(owner_indices, support["raw_indices"], support["face_indices"]),
        "frozen_before_errors": True,
    }
    _write_json(output / f"N{n}.selection.json", selection)
    _progress(output, f"N{n}_selected", owners=len(owner_indices), raw_cells=len(support["raw_indices"]), faces=len(support["face_indices"]))

    raw_points = _raw_points(context)
    artifact = integrated._load_resolution(GEOMETRY, BASELINE, n).artifact
    owner_volume = result["owner_volume"][owner_indices]
    cases_arrays: dict[str, np.ndarray] = {
        "owner_indices": owner_indices,
        "owner_keys": owner_keys[owner_indices],
        "owner_volume": owner_volume,
        "raw_indices": support["raw_indices"],
        "raw_keys": support["raw_keys"],
        "raw_local_owner": support["raw_local_owner"],
        "face_indices": support["face_indices"],
        "face_keys": support["face_keys"],
        "face_incidence": support["face_incidence"],
    }
    representative_raw = []
    representative_rows, category_row = _stratified_owner_rows(labels)
    for row in representative_rows:
        representative_raw.append(int(np.flatnonzero(support["raw_local_owner"] == row)[0]))
    coefficient_divergence = _coefficient_divergence_diagnostic(
        reference, raw_points[support["raw_indices"][representative_raw]]
    )
    case_reports: dict[str, Any] = {}
    production_setup_shared = None
    for field_name in FIELD_NAMES:
        _progress(output, f"N{n}_{field_name}_observations")
        raw_values, raw_gradients = _evaluate_fields(field_name, reference, raw_points)
        owner_values = _owner_observations(context, raw_values)
        if field_name == "corrected_frozen_mms":
            # The current qualified generalized-potential owner vorticity is
            # reused verbatim.  No historical metric interpolant is sampled.
            owner_values[3] = np.asarray(context.arrays["owner_values"][1])
        context.arrays["owner_values"] = owner_values

        _progress(output, f"N{n}_{field_name}_production")
        Lm, Lr, Lt, production_diag, h_centers, production_setup_shared = _production_action(
            n, artifact, owner_keys, owner_values, reference, production_setup_shared
        )
        owner_tuple = tuple(owner_keys[owner_indices].T)
        L = {
            "material": np.moveaxis(Lm[:, :, owner_tuple[0], owner_tuple[1], owner_tuple[2]], (0, 1), (2, 1)),
            "remainder": np.moveaxis(Lr[:, :, owner_tuple[0], owner_tuple[1], owner_tuple[2]], (0, 1), (2, 1)),
            "total": np.moveaxis(Lt[:, :, owner_tuple[0], owner_tuple[1], owner_tuple[2]], (0, 1), (2, 1)),
        }
        # L arrays are (owner,direction,equation).

        quadrature_terms: dict[int, dict[str, np.ndarray]] = {}
        reference_uncertainty = None
        for order in REFERENCE_ORDERS:
            points, logical_weights = _cell_quadrature(context, support["raw_keys"], order)
            exact_values, exact_gradients = _evaluate_fields(field_name, reference, points.reshape(-1, 3))
            exact_values = exact_values.T.reshape(len(points), points.shape[1], 5)
            exact_gradients = exact_gradients.transpose(1, 0, 2).reshape(len(points), points.shape[1], 5, 3)
            prepared = reference.prepare(points.reshape(-1, 3))
            quadrature_terms[order] = _integrate_sources(
                exact_values,
                exact_gradients,
                prepared,
                logical_weights,
                support["raw_local_owner"],
                len(owner_indices),
            )
        oracle = quadrature_terms[3]
        reference_uncertainty_terms = {
            term: np.maximum(
                np.abs(quadrature_terms[5][f"physical:{term}"] - quadrature_terms[3][f"physical:{term}"]),
                np.abs(quadrature_terms[3][f"physical:{term}"] - quadrature_terms[1][f"physical:{term}"]),
            )
            for term in ("material", "remainder", "total")
        }
        step_sensitivity = _geometry_step_sensitivity(
            reference, context, support, labels, field_name
        )
        step_rows = step_sensitivity["owner_rows"]
        reference_uncertainty_terms["total"][step_rows] = np.maximum(
            reference_uncertainty_terms["total"][step_rows], step_sensitivity["max_difference"]
        )
        reference_uncertainty_l2 = {
            term: np.sqrt(
                np.sum(owner_volume[:, None] * value**2, axis=0) / np.sum(owner_volume)
            )
            for term, value in reference_uncertainty_terms.items()
        }

        points, logical_weights = _cell_quadrature(context, support["raw_keys"], 3)
        prepared = reference.prepare(points.reshape(-1, 3))
        cubic_values, cubic_gradients, cubic_diag = _fit_volume_state(context, points, support["raw_keys"])
        C_terms = _integrate_sources(
            cubic_values, cubic_gradients, prepared, logical_weights,
            support["raw_local_owner"], len(owner_indices),
        )
        linear_values, linear_gradients = _linear_volume_state(
            h_centers, context, points, support["raw_keys"]
        )
        G_terms = _integrate_sources(
            linear_values, linear_gradients, prepared, logical_weights,
            support["raw_local_owner"], len(owner_indices),
        )

        face_points, _face_weights = _face_quadrature(context, support["face_keys"], 3)
        central_face = np.empty((len(face_points), face_points.shape[1], 5), dtype=np.float64)
        face_fallback = np.zeros(len(face_points), dtype=np.uint8)
        face_condition = np.zeros(len(face_points))
        for row, (key, nodes) in enumerate(zip(support["face_keys"], face_points, strict=True)):
            axis, i, _j, k = map(int, key)
            if axis == 0 and i == 0:
                central_face[row] = _evaluate_fields(field_name, reference, nodes)[0].T
                continue
            fit = numerics._fit_entity(
                context, nodes[len(nodes) // 2], axis=axis, eta_index=k,
                owner_values=np.asarray(context.arrays["owner_values"]),
            )
            value, _gradient = numerics._evaluate_fit(fit, nodes, context.eta_period)
            central_face[row] = value.T
            face_fallback[row] = int(fit.diagnostics["fallback_level"])
            face_condition[row] = float(fit.diagnostics["condition"])
        C_numerator, C_directional_numerator, C_boundary_numerator, C_face_terms, C_interface_diag = _interface_correction(
            context, support, reference, central_face.copy(), mode="C"
        )
        U_numerator, U_directional_numerator, U_boundary_numerator, U_face_terms, U_interface_diag = _interface_correction(
            context, support, reference, central_face.copy(), mode="U"
        )
        evolution_volume = C_terms["evolution:volume"]
        C_characteristic = C_numerator / evolution_volume[:, None]
        U_jump = U_numerator / evolution_volume[:, None]
        C_characteristic_directional = C_directional_numerator / evolution_volume[:, None, None]
        U_characteristic_directional = U_directional_numerator / evolution_volume[:, None, None]
        C_boundary = C_boundary_numerator / evolution_volume[:, None]
        U_boundary = U_boundary_numerator / evolution_volume[:, None]
        actions = {
            "L": {term: np.sum(value, axis=1) for term, value in L.items()},
            "G": {term: G_terms[f"physical:{term}"] for term in ("material", "remainder", "total")},
            "C": {
                "material": C_terms["physical:material"] + C_characteristic,
                "remainder": C_terms["physical:remainder"],
                "total": C_terms["physical:total"] + C_characteristic,
            },
            "U": {
                "material": C_terms["physical:material"] + U_jump,
                "remainder": C_terms["physical:remainder"],
                "total": C_terms["physical:total"] + U_jump,
            },
            "oracle": {term: oracle[f"physical:{term}"] for term in ("material", "remainder", "total")},
        }
        directional_actions = {
            "L": L,
            "G": {term: G_terms[f"physical:{term}:directional"] for term in ("material", "remainder", "total")},
            "C": {
                "material": C_terms["physical:material:directional"] + C_characteristic_directional,
                "remainder": C_terms["physical:remainder:directional"],
                "total": C_terms["physical:total:directional"] + C_characteristic_directional,
            },
            "U": {
                "material": C_terms["physical:material:directional"] + U_characteristic_directional,
                "remainder": C_terms["physical:remainder:directional"],
                "total": C_terms["physical:total:directional"] + U_characteristic_directional,
            },
            "oracle": {term: oracle[f"physical:{term}:directional"] for term in ("material", "remainder", "total")},
        }
        targets = {term: oracle[f"physical:{term}"] for term in ("material", "remainder", "total")}
        directional_targets = {
            term: oracle[f"physical:{term}:directional"]
            for term in ("material", "remainder", "total")
        }
        field_report: dict[str, Any] = {
            "production": production_diag,
            "reference": {
                "orders": list(REFERENCE_ORDERS),
                "uncertainty_L2_by_term": reference_uncertainty_l2,
                "geometry_step_sensitivity": step_sensitivity,
                "physical_vs_evolution_target": _metrics(
                    oracle["evolution:total"], oracle["physical:total"], owner_volume
                ),
                "lineage": (
                    "current qualified continuous MetricEvaluator/MAKEGRID sidecar; "
                    "MMS owner vorticity reused from the corrected generalized-potential baseline; "
                    "no serialized polar-metric interpolant"
                ),
            },
            "reconstruction": {
                "cell_fallback_max": int(np.max(cubic_diag["fallback"])),
                "cell_condition_max": float(np.max(cubic_diag["condition"])),
                "cell_reproduction_max": float(np.max(cubic_diag["reproduction"])),
                "face_fallback_max": int(np.max(face_fallback)),
                "face_condition_max": float(np.max(face_condition)),
                "C_interface": C_interface_diag,
                "U_interface": U_interface_diag,
            },
            "variants": {},
            "directional": {},
            "characteristic_and_boundary": {
                "C_characteristic": C_characteristic,
                "U_characteristic": U_jump,
                "C_physical_wall": C_boundary,
                "U_physical_wall": U_boundary,
            },
            "limitations": {
                "G": "continuous-geometry q3 direct-volume control using the current production H centers and local linear gradients; it is not relabelled as the conservative production path",
                "C_U": "smooth direct-volume material/remainder plus the live four-field wall/interior characteristic fluctuation; global conservative/path compatibility remains unqualified",
                "oracle": "one bounded analytic field/gradient substitution; diagnostic only",
            },
        }
        for variant, terms in actions.items():
            field_report["variants"][variant] = {}
            field_report["directional"][variant] = {}
            for term in ("material", "remainder", "total"):
                metrics = _metrics(terms[term], targets[term], owner_volume)
                field_report["variants"][variant][term] = {
                    "metrics": metrics,
                    "resolved_above_reference_uncertainty": np.asarray(metrics["L2_by_equation"]) > reference_uncertainty_l2[term],
                    "regions": _region_metrics(terms[term], targets[term], owner_volume, labels),
                    "signed_error_vectors": terms[term] - targets[term],
                    "weighted_squared_error": owner_volume[:, None] * (terms[term] - targets[term]) ** 2,
                }
                field_report["directional"][variant][term] = {
                    direction: _metrics(
                        directional_actions[variant][term][:, direction_index, :],
                        directional_targets[term][:, direction_index, :],
                        owner_volume,
                    )
                    for direction_index, direction in enumerate(DIRECTIONS)
                }
        chain_product = -(10 * TAU * cubic_values[..., 2] / (3 * np.asarray(prepared.B).reshape(cubic_values.shape[:2]))) * np.einsum(
            "rqi,rqi->rq", np.asarray(prepared.K).reshape(cubic_values.shape[:2] + (3,)), cubic_gradients[..., 2, :]
        )
        ti2_gradient = 2 * cubic_values[..., 2, None] * cubic_gradients[..., 2, :]
        chain_flux = -(5 * TAU / (3 * np.asarray(prepared.B).reshape(cubic_values.shape[:2]))) * np.einsum(
            "rqi,rqi->rq", np.asarray(prepared.K).reshape(cubic_values.shape[:2] + (3,)), ti2_gradient
        )
        field_report["ion_temperature_chain_rule"] = {
            "pointwise_max_defect": float(np.max(np.abs(chain_product - chain_flux))),
            "production_owner_coefficient_times_C_not_assumed_equal_to_integrated_product": True,
        }
        case_reports[field_name] = field_report
        for variant, terms in actions.items():
            for term, value in terms.items():
                cases_arrays[f"{field_name}:{variant}:{term}"] = value
                cases_arrays[f"{field_name}:{variant}:{term}:directional"] = directional_actions[variant][term]
        for term, value in targets.items():
            cases_arrays[f"{field_name}:target:{term}"] = value
        for term, value in reference_uncertainty_terms.items():
            cases_arrays[f"{field_name}:reference_uncertainty:{term}"] = value
        cases_arrays[f"{field_name}:C_characteristic"] = C_characteristic
        cases_arrays[f"{field_name}:U_characteristic"] = U_jump
        cases_arrays[f"{field_name}:C_boundary"] = C_boundary
        cases_arrays[f"{field_name}:U_boundary"] = U_boundary
        cases_arrays[f"{field_name}:C_face_fluctuations"] = C_face_terms
        cases_arrays[f"{field_name}:U_face_fluctuations"] = U_face_terms
        cases_arrays[f"{field_name}:physical_q3_volume"] = C_terms["physical:volume"]
        cases_arrays[f"{field_name}:evolution_q3_volume"] = C_terms["evolution:volume"]

    payload = {
        "schema": f"{SCHEMA}.resolution",
        "status": "complete",
        "resolution": n,
        "selection": selection,
        "contract": _contract(),
        "fields": case_reports,
        "coefficient_divergence": coefficient_divergence,
        "provenance": _provenance(n),
        "timing": {"total_seconds": time.perf_counter() - started, "peak_gib": _peak_gib()},
        "scope": {
            "bounded_only": True,
            "global_order_claim": False,
            "production_change": False,
            "physics_change": False,
            "selection_or_bias_tuning": False,
        },
    }
    _write_npz(output / f"N{n}.npz", cases_arrays, payload)
    _write_json(output / f"N{n}.json", payload)
    _progress(output, f"N{n}_complete", elapsed_seconds=payload["timing"]["total_seconds"], peak_gib=payload["timing"]["peak_gib"])
    return payload


def _compact_metric(case: Mapping[str, Any], field: str, variant: str, term: str) -> list[float]:
    return [float(value) for value in case["fields"][field]["variants"][variant][term]["metrics"]["L2_by_equation"]]


def summarize(output: Path) -> dict[str, Any]:
    cases = [json.loads((output / f"N{n}.json").read_text()) for n in (32, 48, 64)]
    table = {
        field: {
            variant: {
                term: [_compact_metric(case, field, variant, term) for case in cases]
                for term in ("material", "remainder", "total")
            }
            for variant in VARIANTS
        }
        for field in FIELD_NAMES
    }
    scores = {
        variant: float(
            sum(
                np.sum(np.square(_compact_metric(case, field, variant, "total")))
                for case in cases
                for field in FIELD_NAMES
            )
        )
        for variant in ("L", "G", "C", "U")
    }
    if scores["G"] < 0.7 * scores["L"]:
        decision = "P06-D: repair the geometry/coefficient-product integration while preserving the live characteristic model"
        basis = "G removes at least 30% of the aggregate squared total error relative to L"
    elif scores["C"] < 0.7 * scores["G"]:
        decision = "P06-D: adopt the frozen central cubic smooth representation in a conservative/path-compatible assembly"
        basis = "C removes at least 30% of the aggregate squared total error relative to G"
    elif scores["U"] < 0.7 * scores["C"]:
        decision = "P06-D: carry the recentered coupled-characteristic correction into the path-compatible cubic assembly"
        basis = "U removes at least 30% of the aggregate squared total error relative to C"
    else:
        decision = "P06-E: freeze the identities and run the predeclared global complete-curvature qualification"
        basis = "no single bounded L→G→C→U transition removes 30% of aggregate squared total error"
    recommendation = f"Recommend {decision}. Basis: {basis}. This bounded choice is not a promotion claim."
    summary = {
        "schema": f"{SCHEMA}.summary",
        "status": "complete",
        "resolutions": [32, 48, 64],
        "equations": list(EQUATION_NAMES),
        "table": table,
        "reference_uncertainty": {
            field: [case["fields"][field]["reference"]["uncertainty_L2_by_term"] for case in cases]
            for field in FIELD_NAMES
        },
        "aggregate_squared_total_error_score": scores,
        "recommended_next_stage": decision,
        "recommendation_basis": basis,
        "recommendation_template": recommendation,
        "claims": {
            "bounded_formulation_and_mechanism_audit_complete": True,
            "global_convergence_or_certification": False,
            "curvature_promotion": False,
        },
        "timing": {
            f"N{case['resolution']}": case["timing"] for case in cases
        },
    }
    _write_json(output / "summary.json", summary)
    _write_report(output, summary, cases)
    _progress(output, "summary_complete")
    return summary


def _write_report(output: Path, summary: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# P06 bounded complete-curvature audit",
        "",
        "## Result",
        "",
        "P06-A/B/C completed on deterministic complete-owner samples at N32/N48/N64 for the corrected frozen MMS and the independent regular-chart held-out state. This is bounded mechanism evidence, not a convergence or production-promotion claim.",
        "",
        "## Formulation",
        "",
        "With `C(f)=K^alpha partial_alpha f`, `K=B/(2J) curl(b_cov/B)`, and `psi=phi+tau Ti`, the independently derived complete terms are:",
        "",
        "| equation | complete curvature RHS | potential remainder R | sign / units / measure |",
        "|---|---|---|---|",
        "| n | `2/B [C(n Te)-n C(phi)]` | `-2n/B C(psi)` | signed as returned by RHS; normalized rate; production V/B evolution measure |",
        "| Te | `4Te/(3B)[C(nTe)/n+5C(Te)/2-C(phi)]` | `-4Te/(3B) C(psi)` | signed as returned by RHS; normalized rate; production V/B evolution measure |",
        "| Ti | `4Ti/(3B)[C(nTe)/n-5tau C(Ti)/2-C(phi)]` | `-4Ti/(3B) C(psi)` | signed as returned by RHS; normalized rate; production V/B evolution measure |",
        "| omega | `2B/n C(nTe+tau nTi)` | `0` | signed as returned by RHS; normalized vorticity rate; production V/B evolution measure |",
        "| Vi, Ve | identically absent in this selected curvature model | `0` | no curvature channel |",
        "",
        "The material matrix and remainder sum to these equations to the preflight algebra tolerance. Production stores `Q=J K/B`, applies the characteristic normal `Q/B`, and divides by the owner `V/B` measure; the `B^2` vorticity row is the corresponding compensating normalization. `rho_star` is absent from curvature, while `tau` appears exactly in the Ti couplings shown. Physical-volume and `V/B` owner targets are reported separately.",
        "",
        "## Numerical comparison",
        "",
        "Entries are physical-volume-weighted L2 errors in equation order `(n, Te, Ti, omega)`.",
        "",
    ]
    for field in FIELD_NAMES:
        lines.extend((f"### {field}", "", "| N | variant | material | remainder | total |", "|---:|---|---|---|---|"))
        for case in cases:
            n = case["resolution"]
            for variant in VARIANTS:
                values = []
                for term in ("material", "remainder", "total"):
                    vector = case["fields"][field]["variants"][variant][term]["metrics"]["L2_by_equation"]
                    values.append("/".join(f"{float(value):.4e}" for value in vector))
                lines.append(f"| {n} | {variant} | {values[0]} | {values[1]} | {values[2]} |")
        lines.append("")
        lines.extend(("Total-error directional split:", "", "| N | variant | radial u | theta | eta |", "|---:|---|---|---|---|"))
        for case in cases:
            n = case["resolution"]
            for variant in VARIANTS:
                vectors = []
                for direction in DIRECTIONS:
                    vector = case["fields"][field]["directional"][variant]["total"][direction]["L2_by_equation"]
                    vectors.append("/".join(f"{float(value):.4e}" for value in vector))
                lines.append(f"| {n} | {variant} | {vectors[0]} | {vectors[1]} | {vectors[2]} |")
        lines.append("")
    lines.extend(("## Boundary, fallback, and coefficient diagnostics", "", "| N | field | C wall max | U wall max | C/U spectral fallbacks | U donor-hash tie mismatches | C/U side-fit fallback max | coefficient-divergence max |", "|---:|---|---:|---:|---:|---:|---:|---:|"))
    for case in cases:
        for field in FIELD_NAMES:
            diagnostics = case["fields"][field]
            cb = np.asarray(diagnostics["characteristic_and_boundary"]["C_physical_wall"])
            ub = np.asarray(diagnostics["characteristic_and_boundary"]["U_physical_wall"])
            ci = diagnostics["reconstruction"]["C_interface"]
            ui = diagnostics["reconstruction"]["U_interface"]
            lines.append(
                f"| {case['resolution']} | {field} | {np.max(np.abs(cb)):.4e} | {np.max(np.abs(ub)):.4e} | "
                f"{ci['spectral_fallback_count']}/{ui['spectral_fallback_count']} | "
                f"{ui['frozen_donor_hash_mismatch_count']} | "
                f"{ci['side_fit_fallback_max']}/{ui['side_fit_fallback_max']} | "
                f"{float(case['coefficient_divergence']['max_abs']):.4e} |"
            )
    lines.append("")
    max_peak = max(float(case["timing"]["peak_gib"]) for case in cases)
    lines.extend(
        (
            "## Reference and limitations",
            "",
            "The reference uses analytic field gradients and the same qualified continuous MetricEvaluator/MAKEGRID lineage at every resolution. Midpoint, q3 and q5 physical owner integrations and half/double geometry-derivative sensitivity on at least one frozen owner per category are represented by the saved uncertainty arrays; every comparison carries a per-equation resolved/unresolved flag. The corrected MMS owner vorticity comes from the current generalized-potential baseline, not a historical polar-metric interpolant.",
            "",
            "`L` is the genuine frozen production curvature component path. `G` is explicitly a continuous-geometry q3 direct-volume control of current H centers and local linear gradients, not a relabelled conservative replacement. `C/U` use frozen selection-v3 cubic values/gradients for the smooth direct-volume material and remainder; `U` adds the existing live four-field characteristic action of the fixed bias-0.75 recentered side-fit jump. The scalar P05 `abs(U_phi)` correction is never used. Physical walls retain the production characteristic solve with no exterior polynomial fit. Cross-platform exact-radius donor ties are never hidden: campaign-vs-live hashes, face identities and tie counts are saved, while the unchanged frozen selection-v3 code supplies the live support. Raw constituent keys, incident faces, physical and V/B measures, characteristic and wall contributions, and fallback counts are saved in each resumable NPZ/JSON pair.",
            "",
            "## Cost",
            "",
            f"Peak measured process RSS was {max_peak:.3f} GiB. Per-resolution setup/application/total timings are in `summary.json` and each resolution file.",
            "",
            "## Recommendation",
            "",
            summary["recommendation_template"],
            "",
            "## Proposed roadmap status paragraph",
            "",
            "**P06 bounded formulation audit complete; global qualification pending.** The P06-A/B/C evidence reconciles the independent complete continuum equations, the material/potential decomposition, the compatible conservative coefficient, and the production `V/B` evolution measure with the physical-volume reporting target. Deterministic complete-owner N32/N48/N64 samples cover axis-adjacent, agglomerated, size-transition, ordinary, radial-wall, theta-seam and eta-seam categories for both predeclared states. Live production `L`, continuous-geometry/integration `G`, frozen central cubic `C`, recentered coupled-characteristic `U`, and one analytic oracle are saved with material/remainder/total, directional, boundary, signed-vector and weighted-SSE diagnostics. This bounded result does not establish order or authorize curvature promotion; the report recommends one P06-D correction, or P06-E global qualification only if no correction is supported.",
            "",
            "## Reproduction",
            "",
            "Run `conda run -n drb python scripts/audit_p06_curvature_bounded.py --stage preflight --output <dir>`, then one `--stage resolution --resolution N` for N=32,48,64, followed by `--stage summarize`. Validate with `--stage validate`.",
        )
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")


def validate(output: Path, expected: Sequence[int] = (32, 48, 64)) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    preflight = json.loads((output / "preflight.json").read_text())
    checks["preflight_passed"] = preflight["status"] == "passed"
    implementation_hash = _implementation_hash()
    for n in expected:
        payload = json.loads((output / f"N{n}.json").read_text())
        arrays = _load_npz(output / f"N{n}.npz")
        checks[f"N{n}_status"] = payload["status"] == "complete"
        checks[f"N{n}_implementation"] = payload["provenance"]["implementation_hash"] == implementation_hash
        checks[f"N{n}_fields"] = set(payload["fields"]) == set(FIELD_NAMES)
        checks[f"N{n}_finite"] = all(np.all(np.isfinite(value)) for value in arrays.values())
        checks[f"N{n}_peak_below_4GiB"] = float(payload["timing"]["peak_gib"]) <= 4.0
        checks[f"N{n}_coverage"] = set(sum(payload["selection"]["labels"], [])) == {
            "axis_adjacent", "agglomerated_bulk", "size_transition", "ordinary_interior", "radial_wall", "theta_seam", "eta_seam"
        }
        for field in FIELD_NAMES:
            checks[f"N{n}_{field}_variants"] = set(payload["fields"][field]["variants"]) == set(VARIANTS)
    if set(expected) == {32, 48, 64}:
        summary = json.loads((output / "summary.json").read_text())
        checks["summary_complete"] = summary["status"] == "complete"
        checks["report_exists"] = (output / "report.md").is_file()
    passed = all(bool(value) for value in checks.values())
    result = {"schema": f"{SCHEMA}.validation", "passed": passed, "checks": checks}
    _write_json(output / "validation.json", result)
    if not passed:
        raise ValueError(f"P06 validation failed: {checks}")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "resolution", "summarize", "validate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, choices=(32, 48, 64))
    parser.add_argument("--expected", type=int, nargs="*", default=(32, 48, 64))
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock_path = args.output / ".writer.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.stage == "preflight":
            run_preflight(args.output)
        elif args.stage == "resolution":
            if args.resolution is None:
                raise SystemExit("--resolution is required")
            run_resolution(args.output, args.resolution)
        elif args.stage == "summarize":
            summarize(args.output)
        else:
            validate(args.output, args.expected)


if __name__ == "__main__":
    main()
