#!/usr/bin/env python3
"""Portable complete-owner P06 curvature qualification kernels.

The campaign is research-only.  It reconstructs two frozen smooth states with
the qualified selection-v3 cubic supports, applies one point wall constraint
to wall-reaching cell and face fits, and evaluates centered and characteristic
(``U``) complete curvature actions.  No evolved or production selector is
changed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("HSX_DEPLOYMENT_ROOT", HERE.parents[1])).resolve()
ROOT = Path(os.environ.get("HSX_SOURCE_ROOT", WORKSPACE / "DRBX")).resolve()
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, ROOT / "src", SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Reuse only the portable geometry/context helpers from the qualified bracket
# runner.  Curvature assembly below is independent and cannot call its bracket
# case/merge paths by accident.
base = _load_module(SCRIPTS / "hsx_remote_qualification" / "numerics.py", "p06_portable_base")
cubic = base.cubic
integrated = base.integrated
mms = base.mms

from drbx.geometry.fci_perpendicular_bracket import (  # noqa: E402
    BIAS,
    CubicOwnerGeometry,
    SelectionV3Policy,
    _basis,
    _observation,
    _unwrap,
    build_cell_block,
    build_face_block,
    select_cubic_support_v3,
)
from drbx.native.fci_curvature_production_flux import curvature_principal_matrix  # noqa: E402
from drbx.native.fci_operators import _curvature_bc_characteristic_wall_states  # noqa: E402
import simulate_hsx_blob as blob  # noqa: E402


SCHEMA = "drbx.p06-curvature-global-case-v1"
PREPARE_SCHEMA = "drbx.p06-curvature-global-prepare-v1"
CHUNK_SCHEMA = "drbx.p06-curvature-global-chunk-v1"
PREFLIGHT_SCHEMA = "drbx.p06-curvature-global-preflight-v1"
SUMMARY_SCHEMA = "drbx.p06-curvature-global-summary-v1"
FIELD_NAMES = ("corrected_frozen_mms", "regular_chart_heldout")
EQUATIONS = ("density", "Te", "Ti", "vorticity")
TERMS = ("material", "remainder", "total")
ACTIONS = ("centered", "U")
DIRECTIONS = ("u", "theta", "eta")
TAU = float(mms.PHYSICAL_PARAMETERS["tau"])
FLOOR = 1.0e-12
POLICY = SelectionV3Policy()
PRIMARY_ACTION = {
    "density": "U",
    "Te": "U",
    "Ti": "U",
    "vorticity": "centered",
}
_GEOMETRY_CACHE: dict[int, CubicOwnerGeometry] = {}


def _max_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / 1024.0**3


def _current_rss_gib() -> float | None:
    statm = Path("/proc/self/statm")
    if statm.is_file():
        try:
            return int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024.0**3
        except (OSError, ValueError, IndexError):
            pass
    try:
        import psutil  # type: ignore[import-not-found]

        return float(psutil.Process().memory_info().rss) / 1024.0**3
    except (ImportError, OSError):
        return None


def _metric_cache_diagnostics(reference: Any) -> dict[str, Any] | None:
    evaluator = getattr(reference, "metric_evaluator", None)
    if evaluator is None or not hasattr(evaluator, "basis_cache_diagnostics"):
        return None
    return evaluator.basis_cache_diagnostics()


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
    if isinstance(value, (tuple, list)):
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


def _load_npz(path: Path, schema: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"].item()))
        arrays = {name: np.asarray(source[name]) for name in source.files if name != "metadata_json"}
    if metadata.get("schema") != schema:
        raise ValueError(f"schema mismatch for {path}")
    if metadata.get("array_sha256") != _array_hash(*(arrays[name] for name in sorted(arrays))):
        raise ValueError(f"payload hash mismatch for {path}")
    return arrays, metadata


def _identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _event(stage: str, **details: Any) -> None:
    print(json.dumps({"event": "p06_curvature_global", "stage": stage, **_json(details)}, sort_keys=True), flush=True)


def _config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "drbx.p06-curvature-global-runtime-v1":
        raise ValueError("unsupported P06 curvature runtime configuration")
    return payload


def _path(config: Mapping[str, Any], name: str) -> Path:
    return Path(config["paths"][name]).resolve()


def _source_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        "implementation": Path(__file__),
        "parallel_runner": HERE / "parallel_runner.py",
        "selection": HERE / "selection.py",
        "cubic_builder": SCRIPTS / "audit_hsx_cubic_derivative_global.py",
        "portable_geometry": ROOT / "src/drbx/geometry/fci_perpendicular_bracket.py",
        "boundary_interface": ROOT / "src/drbx/geometry/fci_boundary_functional_reconstruction.py",
        "curvature_matrix": ROOT / "src/drbx/native/fci_curvature_production_flux.py",
        "reference_sidecar": _path(config, "reference_sidecar"),
    }
    return {name: _identity(path) for name, path in paths.items()}


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
        raise ValueError("raw topology contains an inactive owner")
    return result


def _owner_observations(context: Any, raw_values: np.ndarray) -> np.ndarray:
    raw_owner = _raw_owner(context)
    raw_volume = np.asarray(context.arrays["raw_volume"], dtype=np.float64)
    count = len(context.arrays["owner_flat_ids"])
    volume = np.bincount(raw_owner, weights=raw_volume, minlength=count)
    result = np.empty((raw_values.shape[0], count), dtype=np.float64)
    for field in range(raw_values.shape[0]):
        result[field] = np.bincount(
            raw_owner, weights=raw_volume * raw_values[field], minlength=count
        ) / volume
    return result


def _heldout_fields(points: np.ndarray, eta_period: float) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(points, dtype=np.float64)
    u, theta, eta = q.T
    x = u * np.cos(theta)
    y = u * np.sin(theta)
    zeta = 2.0 * np.pi * eta / float(eta_period)
    wave = 2.0 * np.pi / float(eta_period)
    radius = x * x + y * y
    a = 1.0 - radius
    envelope = a * a
    ex = -4.0 * x * a
    ey = -4.0 * y * a
    specifications = (
        (0.08*x + 0.03*y*np.sin(zeta), 0.08+0*x, 0.03*np.sin(zeta), 0.03*y*np.cos(zeta)),
        (0.06*y + 0.02*x*y*np.cos(zeta), 0.02*y*np.cos(zeta), 0.06+0.02*x*np.cos(zeta), -0.02*x*y*np.sin(zeta)),
        (0.05*x*np.cos(zeta) + 0.02*(x*x-y*y), 0.05*np.cos(zeta)+0.04*x, -0.04*y, -0.05*x*np.sin(zeta)),
        (0.05*(x*x-y*y) + 0.02*x*np.sin(zeta), 0.10*x+0.02*np.sin(zeta), -0.10*y, 0.02*x*np.cos(zeta)),
        (0.04*x*y + 0.03*y*np.cos(zeta), 0.04*y, 0.04*x+0.03*np.cos(zeta), -0.03*y*np.sin(zeta)),
    )
    values = np.empty((5, len(q)), dtype=np.float64)
    gradients = np.empty((5, len(q), 3), dtype=np.float64)
    for field, ((function, fx, fy, fz), background) in enumerate(
        zip(specifications, (1.0, 1.0, 1.0, 0.0, 0.0), strict=True)
    ):
        values[field] = background + envelope * function
        gx = ex * function + envelope * fx
        gy = ey * function + envelope * fy
        gradients[field, :, 0] = gx*np.cos(theta) + gy*np.sin(theta)
        gradients[field, :, 1] = -u*gx*np.sin(theta) + u*gy*np.cos(theta)
        gradients[field, :, 2] = wave * envelope * fz
    return values, gradients


def _evaluate_fields(name: str, reference: Any, points: np.ndarray, time_value: float) -> tuple[np.ndarray, np.ndarray]:
    if name == "regular_chart_heldout":
        return _heldout_fields(points, reference.eta_period)
    if name != "corrected_frozen_mms":
        raise ValueError(name)
    raw = reference._fields_raw(np.asarray(points, dtype=np.float64), float(time_value))
    values = np.empty((5, len(points)), dtype=np.float64)
    gradients = np.empty((5, len(points), 3), dtype=np.float64)
    for field, name_ in enumerate(("density", "Te", "Ti", "vorticity", "phi")):
        if name_ == "vorticity":
            values[field] = 0.0
            gradients[field] = 0.0
        else:
            payload = raw[name_]
            values[field] = np.asarray(payload[0]) + (1.0 if field < 3 else 0.0)
            gradients[field] = np.stack(payload[1:4], axis=-1)
    return values, gradients


def _continuum_terms(values: np.ndarray, gradients: np.ndarray, prepared: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, te, ti, _omega, _phi = values
    bmag = np.asarray(prepared.B)
    curvature = np.einsum("pd,fpd->fp", np.asarray(prepared.K), gradients)
    matrix = np.asarray(curvature_principal_matrix(n, te, ti, bmag, TAU))
    material = np.einsum("pij,jp->pi", matrix, curvature[:4]) / np.maximum(bmag[:, None], 1.0e-30)
    cpsi = curvature[4] + TAU * curvature[2]
    coeff = np.column_stack((-2*n/bmag, -4*te/(3*bmag), -4*ti/(3*bmag), np.zeros_like(n)))
    remainder = coeff * cpsi[:, None]
    directional_curvature = np.einsum("pd,fpd->pdf", np.asarray(prepared.K), gradients)
    material_directional = np.einsum("pij,pdj->pdi", matrix, directional_curvature[..., :4]) / np.maximum(bmag[:, None, None], 1.0e-30)
    remainder_directional = (directional_curvature[..., 4] + TAU*directional_curvature[..., 2])[..., None] * coeff[:, None, :]
    return material, remainder, material + remainder, material_directional, remainder_directional, material_directional + remainder_directional


def _owner_geometry(context: Any, prepare: Mapping[str, np.ndarray]) -> CubicOwnerGeometry:
    cached = _GEOMETRY_CACHE.get(id(context))
    if cached is not None:
        return cached
    payload = (
        np.asarray(context.arrays["owner_flat_ids"]),
        np.asarray(context.arrays["owner_centroid_xy"]),
        np.asarray(context.owner_moments),
        np.asarray(prepare["owner_volume"]),
    )
    geometry = CubicOwnerGeometry(
        owner_flat_ids=payload[0],
        owner_plane=np.asarray(context.arrays["owner_plane"]),
        owner_angular_column=np.asarray(context.owner_j),
        owner_centroid_xy=payload[1],
        owner_eta=np.asarray(context.owner_eta),
        owner_moments_xy=payload[2],
        owner_volume=payload[3],
        raw_owner=np.asarray(prepare["raw_owner"]),
        raw_volume=np.asarray(context.arrays["raw_volume"]),
        resolution=context.resolution,
        dr=context.dr,
        dtheta=context.dtheta,
        deta=context.deta,
        eta_period=context.eta_period,
        identity=_array_hash(*payload),
    )
    _GEOMETRY_CACHE[id(context)] = geometry
    return geometry


def _support_reaches_wall(geometry: CubicOwnerGeometry, donors: np.ndarray) -> bool:
    radial = geometry.owner_flat_ids[np.asarray(donors, dtype=np.int64)] // geometry.resolution**2
    return bool(np.any(radial == geometry.resolution - 1))


def _constrain_map(base_map: np.ndarray, gram: np.ndarray, row: np.ndarray) -> np.ndarray:
    inverse_row = np.linalg.solve(gram, row[:, None])
    denominator = float(row @ inverse_row[:, 0])
    if not np.isfinite(denominator) or abs(denominator) < 1.0e-14:
        raise ValueError("singular point boundary constraint")
    return base_map - inverse_row @ ((row @ base_map)[None, :] / denominator)


def _entity_field_maps(
    geometry: CubicOwnerGeometry,
    reference: Any,
    point: np.ndarray,
    axis: int,
    eta_index: int,
    donors: np.ndarray,
    maps: Sequence[np.ndarray],
    scale: np.ndarray,
) -> tuple[list[np.ndarray], float]:
    """Return central/left/right maps with one geometry-defined point relation."""

    donors = np.asarray(donors, dtype=np.int64)
    replicated = [np.repeat(np.asarray(item)[None, ...], 5, axis=0) for item in maps]
    if not _support_reaches_wall(geometry, donors):
        return replicated, 0.0
    del eta_index
    center = np.array((
        point[0] * np.cos(point[1]),
        point[0] * np.sin(point[1]),
        point[2],
    ))
    scale = np.asarray(scale, dtype=np.float64)
    observation = _observation(geometry, donors, center, scale)
    eta = _unwrap(geometry.owner_eta[donors], center[2], geometry.eta_period)
    displacement = np.column_stack((
        geometry.owner_centroid_xy[donors] - center[:2],
        eta - center[2],
    ))
    distance = np.sqrt(np.sum((displacement / scale) ** 2, axis=1))
    wall = np.asarray(point, dtype=np.float64).copy()
    wall[0] = 1.0
    value_basis, derivative_basis = _basis(wall[None, :], center, scale, geometry.eta_period)
    metric = reference._metric(wall[None, :])["gcontra"][0]
    normal = metric[0] / np.sqrt(metric[0, 0])
    normal_row = derivative_basis[0] @ normal
    value_row = value_basis[0]
    root2 = (1.0 / (1.0 + distance**2)) ** 2
    if axis == 0:
        direction = np.array((np.cos(point[1]), np.sin(point[1]), 0.0))
    elif axis == 1:
        direction = np.array((-np.sin(point[1]), np.cos(point[1]), 0.0))
    else:
        direction = np.array((0.0, 0.0, 1.0))
    signed = displacement @ direction / np.sqrt(np.sum((direction * scale) ** 2))
    weights = (root2, root2 * (1.0 - BIAS*np.tanh(signed)), root2 * (1.0 + BIAS*np.tanh(signed)))
    residual = 0.0
    for map_index, (field_maps, weight2) in enumerate(zip(replicated, weights, strict=True)):
        gram = observation.T @ (weight2[:, None] * observation)
        for field in range(3):
            field_maps[field] = _constrain_map(field_maps[field], gram, normal_row)
            residual = max(residual, float(np.max(np.abs(normal_row @ field_maps[field]))))
        field_maps[4] = _constrain_map(field_maps[4], gram, value_row)
        residual = max(residual, float(np.max(np.abs(value_row @ field_maps[4]))))
        replicated[map_index] = field_maps
    return replicated, residual


def _reconstruct(
    block: Any,
    geometry: CubicOwnerGeometry,
    reference: Any,
    owner_values: np.ndarray,
    *,
    kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, float]:
    entity_count = int(np.count_nonzero(block.valid))
    max_donors = block.donors.shape[1]
    central_maps = np.repeat(np.asarray(block.central_map[:entity_count, None]), 5, axis=1)
    left_maps = None if kind == "cell" else np.repeat(np.asarray(block.left_map[:entity_count, None]), 5, axis=1)
    right_maps = None if kind == "cell" else np.repeat(np.asarray(block.right_map[:entity_count, None]), 5, axis=1)
    constraint_residual = 0.0
    for row in range(entity_count):
        count = int(block.donor_count[row])
        donors = np.asarray(block.donors[row, :count], dtype=np.int64)
        if kind == "cell":
            index = int(block.indices[row])
            i, j, k = np.unravel_index(index, (geometry.resolution,) * 3)
            point = np.array((
                (i + 0.5) * geometry.dr,
                (j + 0.5) * geometry.dtheta,
                (k + 0.5) * geometry.deta,
            ))
            axis = 0
            selected, residual = _entity_field_maps(
                geometry, reference, point, axis, k, donors,
                (central_maps[row, :, :, :count][0],) * 3,
                block.scale[row],
            )
            if i == geometry.resolution - 1:
                central_maps[row, :, :, :count] = selected[0]
            residual = residual if i == geometry.resolution - 1 else 0.0
        else:
            axis, _i, _j, k = map(int, block.keys[row])
            # The q3 midpoint basis row is evaluated at the face center.
            logical = _logical_center_from_key(geometry, block.keys[row])
            selected, residual = _entity_field_maps(
                geometry, reference, logical, axis, k, donors,
                (
                    central_maps[row, 0, :, :count],
                    left_maps[row, 0, :, :count],
                    right_maps[row, 0, :, :count],
                ),
                block.scale[row],
            )
            central_maps[row, :, :, :count] = selected[0]
            left_maps[row, :, :, :count] = selected[1]
            right_maps[row, :, :, :count] = selected[2]
        constraint_residual = max(constraint_residual, residual)
    donors = np.asarray(block.donors[:entity_count], dtype=np.int64)
    gathered = owner_values[:, :, donors]
    coefficients = np.einsum("efkd,sfed->sfek", central_maps, gathered)
    basis = np.asarray(block.basis[:entity_count])
    derivative = np.asarray(block.derivative_basis[:entity_count])
    values = np.einsum("eqk,sfek->sfeq", basis, coefficients)
    gradients = np.einsum("eqka,sfek->sfeqa", derivative, coefficients)
    if kind == "cell":
        return values, gradients, None, None, constraint_residual
    left_coeff = np.einsum("efkd,sfed->sfek", left_maps, gathered)
    right_coeff = np.einsum("efkd,sfed->sfek", right_maps, gathered)
    left = np.einsum("eqk,sfek->sfeq", basis, left_coeff)
    right = np.einsum("eqk,sfek->sfeq", basis, right_coeff)
    # Preserve the selected central value while using only the side-fit jump.
    jump = right - left
    left = values - 0.5 * jump
    right = values + 0.5 * jump
    return values, gradients, left, right, constraint_residual


def _logical_center_from_key(geometry: CubicOwnerGeometry, key: Sequence[int]) -> np.ndarray:
    axis, i, j, k = map(int, key)
    coordinates = np.array(((i + 0.5)*geometry.dr, (j + 0.5)*geometry.dtheta, (k + 0.5)*geometry.deta))
    coordinates[axis] = (key[axis + 1]) * (geometry.dr, geometry.dtheta, geometry.deta)[axis]
    if axis == 1:
        coordinates[0] = (i + 0.5) * geometry.dr
    elif axis == 2:
        coordinates[0] = (i + 0.5) * geometry.dr
        coordinates[1] = (j + 0.5) * geometry.dtheta
    return coordinates


def _face_geometry(reference: Any, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(points, dtype=np.float64)
    metric = reference._metric(q)
    K = np.empty((len(q), 3), dtype=np.float64)
    wall = np.isclose(q[:, 0], 1.0, rtol=0.0, atol=8*np.finfo(float).eps)
    if np.any(~wall):
        K[~wall] = reference._curvature(q[~wall])
    if np.any(wall):
        qw = q[wall]
        def covariant_over_b(x: np.ndarray) -> np.ndarray:
            local = reference._metric(x)
            return local["bcov"] / local["B"][:, None]
        h = min(float(reference.finite_difference_step), 0.05)
        sample = []
        for offset in range(5):
            shifted = qw.copy(); shifted[:, 0] -= offset*h
            sample.append(covariant_over_b(shifted))
        du = (25*sample[0]-48*sample[1]+36*sample[2]-16*sample[3]+3*sample[4])/(12*h)
        dtheta = reference._derivative(covariant_over_b, qw, 1)
        deta = reference._derivative(covariant_over_b, qw, 2)
        curl = np.stack((dtheta[...,2]-deta[...,1], deta[...,0]-du[...,2], du[...,1]-dtheta[...,0]), axis=-1)
        K[wall] = 0.5 * metric["B"][wall, None] * curl / np.maximum(np.abs(metric["J"])[wall, None], 1.0e-30)
    return np.asarray(metric["J"]), np.asarray(metric["B"]), K


def _principal_matrix(state: np.ndarray, bmag: np.ndarray) -> np.ndarray:
    n, te, ti = np.moveaxis(state[..., :3], -1, 0)
    matrix = np.zeros(state.shape[:-1] + (4, 4), dtype=np.float64)
    matrix[...,0,0]=2*te; matrix[...,0,1]=2*n; matrix[...,0,2]=2*n*TAU
    matrix[...,1,0]=4*te*te/(3*n); matrix[...,1,1]=14*te/3; matrix[...,1,2]=4*TAU*te/3
    matrix[...,2,0]=4*ti*te/(3*n); matrix[...,2,1]=4*ti/3; matrix[...,2,2]=-2*TAU*ti
    matrix[...,3,0]=2*bmag*bmag*(te+TAU*ti)/n; matrix[...,3,1]=2*bmag*bmag; matrix[...,3,2]=2*TAU*bmag*bmag
    return matrix


def _absolute_action(matrix: np.ndarray, jump: np.ndarray) -> tuple[np.ndarray, int]:
    result = np.empty_like(jump)
    fallback = 0
    for row, (operator, vector) in enumerate(zip(matrix.reshape(-1,4,4), jump.reshape(-1,4), strict=True)):
        eigenvalues, eigenvectors = np.linalg.eig(operator)
        if (
            np.max(np.abs(np.imag(eigenvalues))) <= 1.0e-10*(1+np.max(np.abs(np.real(eigenvalues))))
            and np.isfinite(np.linalg.cond(eigenvectors))
            and np.linalg.cond(eigenvectors) <= 1.0e8
        ):
            result.reshape(-1,4)[row] = np.real(eigenvectors @ (np.abs(np.real(eigenvalues)) * (np.linalg.inv(eigenvectors) @ vector)))
        else:
            result.reshape(-1,4)[row] = np.linalg.norm(operator) * vector
            fallback += 1
    return result, fallback


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args.config)
    output = _path(config, "output")
    n = int(args.resolution)
    target = output / f"N{n}.prepare.npz"
    if target.is_file():
        _arrays, metadata = _load_npz(target, PREPARE_SCHEMA)
        if metadata.get("sources") == _json(_source_identity(config)):
            _event("prepare_cache_hit", resolution=n)
            return metadata
    started = time.perf_counter()
    context = cubic._load_context(_path(config,"geometry"), _path(config,"baseline"), n)
    reference = integrated._reference(_path(config,"reference_sidecar"), verify_hashes=False)
    raw_points = _raw_points(context)
    states = []
    for name in FIELD_NAMES:
        raw_values, _ = _evaluate_fields(name, reference, raw_points, float(config["time"]))
        owner = _owner_observations(context, raw_values)
        if name == "corrected_frozen_mms":
            owner[3] = np.asarray(context.arrays["owner_values"][1])
        states.append(owner)
    data = integrated._load_resolution(_path(config,"geometry"), _path(config,"baseline"), n)
    arrays = {
        "owner_values": np.stack(states),
        "owner_volume": np.asarray(data.owner_volume),
        "owner_keys": np.asarray(data.owner_keys),
        "raw_owner": np.asarray(data.raw_owner, dtype=np.int32),
        **{f"region:{name}": np.asarray(mask, dtype=bool) for name, mask in data.masks.items()},
    }
    metadata = {
        "schema": PREPARE_SCHEMA,
        "status": "complete",
        "resolution": n,
        "candidate": config["candidate"],
        "owner_observation": "raw midpoint values projected with physical raw_volume",
        "sources": _source_identity(config),
        "geometry_manifest": _identity(data.artifact_path / "manifest.json"),
        "seconds": time.perf_counter()-started,
        "maximum_rss_gib": _max_rss_gib(),
    }
    metadata["array_sha256"] = _array_hash(*(arrays[name] for name in sorted(arrays)))
    _write_npz(target, arrays, metadata)
    _event("prepare_complete", resolution=n, seconds=metadata["seconds"])
    return metadata


def _compute_cells(
    context: Any,
    reference: Any,
    prepare: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    time_value: float,
    curl_step: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    del curl_step
    indices = np.asarray(indices, dtype=np.int64)
    keys = base._raw_keys(context.resolution, indices)
    points, weights = base._cell_quadrature(context, keys)
    geometry = _owner_geometry(context, prepare)
    zero = lambda q: np.zeros((len(q), 3), dtype=np.float64)
    block = build_cell_block(geometry, indices, points, weights, zero, block_size=len(indices), policy=POLICY)
    values, gradients, _left, _right, constraint = _reconstruct(
        block, geometry, reference, np.asarray(prepare["owner_values"]), kind="cell"
    )
    flat = points.reshape(-1,3)
    prepared_geometry = reference.prepare(flat)
    metric = reference._metric(flat)
    jacobian = np.asarray(metric["J"]).reshape(len(indices),27)
    bmag = np.asarray(metric["B"]).reshape(len(indices),27)
    evolution_weight = weights * jacobian / np.maximum(bmag, 1.0e-30)
    physical_weight = weights * jacobian
    arrays: dict[str, np.ndarray] = {
        "indices": indices,
        "evolution_volume": np.sum(evolution_weight, axis=1),
        "physical_volume": np.sum(physical_weight, axis=1),
        "donor_count": np.asarray(block.donor_count),
        "expansion_level": np.asarray(block.expansion_level),
    }
    closure = 0.0
    for state_index, field_name in enumerate(FIELD_NAMES):
        candidate = _continuum_terms(
            values[state_index].reshape(5,-1),
            gradients[state_index].reshape(5,-1,3),
            prepared_geometry,
        )
        exact_values, exact_gradients = _evaluate_fields(field_name, reference, flat, time_value)
        exact = _continuum_terms(exact_values, exact_gradients, prepared_geometry)
        for term_index, term in enumerate(TERMS):
            candidate_term = candidate[term_index].reshape(len(indices),27,4)
            exact_term = exact[term_index].reshape(len(indices),27,4)
            arrays[f"candidate:{field_name}:{term}"] = np.sum(evolution_weight[...,None]*candidate_term, axis=1)
            arrays[f"reference_physical:{field_name}:{term}"] = np.sum(physical_weight[...,None]*exact_term, axis=1)
            arrays[f"reference_evolution:{field_name}:{term}"] = np.sum(evolution_weight[...,None]*exact_term, axis=1)
            candidate_directional = candidate[term_index+3].reshape(len(indices),27,3,4)
            arrays[f"candidate_directional:{field_name}:{term}"] = np.sum(evolution_weight[...,None,None]*candidate_directional, axis=1)
        closure = max(closure, float(np.max(np.abs(candidate[0]+candidate[1]-candidate[2]))))
    details = {
        "entity_count": len(indices),
        "constraint_residual_max": constraint,
        "M_plus_R_closure_max": closure,
        "maximum_expansion_level": int(np.max(block.expansion_level)),
        "maximum_donor_count": int(np.max(block.donor_count)),
    }
    return arrays, details


def _compute_faces(
    context: Any,
    reference: Any,
    prepare: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    time_value: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    del time_value
    indices = np.asarray(indices, dtype=np.int64)
    keys = base._face_keys(context.resolution, indices)
    points, weights = base._face_quadrature(context, keys)
    geometry = _owner_geometry(context, prepare)
    zero = lambda q: np.zeros((len(q), 3), dtype=np.float64)
    block = build_face_block(geometry, indices, keys, points, weights, zero, block_size=len(indices), policy=POLICY)
    central, _gradient, left, right, constraint = _reconstruct(
        block, geometry, reference, np.asarray(prepare["owner_values"]), kind="face"
    )
    collapsed = np.asarray(block.collapsed, dtype=bool)
    regular = ~collapsed
    J = np.zeros((len(indices),9)); B = np.ones((len(indices),9)); K = np.zeros((len(indices),9,3))
    if np.any(regular):
        rj, rb, rk = _face_geometry(reference, points[regular].reshape(-1,3))
        J[regular]=rj.reshape(-1,9); B[regular]=rb.reshape(-1,9); K[regular]=rk.reshape(-1,9,3)
    normal = J * K[np.arange(len(indices))[:,None],np.arange(9)[None,:],keys[:,0,None]] / np.maximum(B*B,1.0e-30)
    wall_rows = np.flatnonzero((keys[:,0]==0)&(keys[:,1]==context.resolution))
    correction = np.zeros((len(FIELD_NAMES),len(indices),2,4))
    jump_rms = []
    spectral_fallback = 0
    positivity_fallback = 0
    for state in range(len(FIELD_NAMES)):
        state_central = central[state, :4].transpose(1, 2, 0).copy()
        state_left = left[state, :4].transpose(1, 2, 0).copy()
        state_right = right[state, :4].transpose(1, 2, 0).copy()
        state_central[collapsed, :, :3] = 1.0
        state_left[collapsed, :, :3] = 1.0
        state_right[collapsed, :, :3] = 1.0
        for row in wall_rows:
            interior = state_central[row]
            exterior, working, _fallback = _curvature_bc_characteristic_wall_states(
                blob.jnp.asarray(interior), blob.jnp.asarray(interior), blob.jnp.asarray(B[row]), TAU,
                blob.jnp.asarray(normal[row]), interior_on_right=False, positivity_floor=FLOOR,
            )
            state_left[row] = interior
            state_right[row] = np.asarray(exterior)
            state_central[row] = np.asarray(working)
        matrix = _principal_matrix(state_central, B)
        flux_matrix = -normal[...,None,None]*matrix
        jump = state_right-state_left
        absolute, fallback = _absolute_action(flux_matrix,jump)
        spectral_fallback += fallback
        positivity_fallback += int(np.count_nonzero(state_central[...,:3] <= FLOOR))
        material = np.einsum("fqij,fqj->fqi",flux_matrix,jump)
        dplus=0.5*(material+absolute); dminus=0.5*(material-absolute)
        dplus[collapsed]=0.0; dminus[collapsed]=0.0
        correction[state,:,0] = -np.sum(weights[...,None]*dminus,axis=1)
        correction[state,:,1] = -np.sum(weights[...,None]*dplus,axis=1)
        jump_rms.append(float(np.sqrt(np.mean(jump[regular]**2))) if np.any(regular) else 0.0)
    arrays = {
        "indices": indices,
        "keys": np.asarray(keys),
        "correction": correction,
        "lower_raw": np.asarray(block.lower_raw),
        "lower_valid": np.asarray(block.lower_valid),
        "upper_raw": np.asarray(block.upper_raw),
        "upper_valid": np.asarray(block.upper_valid),
        "donor_count": np.asarray(block.donor_count),
        "expansion_level": np.asarray(block.expansion_level),
    }
    details = {
        "entity_count": len(indices),
        "wall_face_count": len(wall_rows),
        "collapsed_face_count": int(np.count_nonzero(collapsed)),
        "constraint_residual_max": constraint,
        "jump_rms": jump_rms,
        "spectral_fallback_count": spectral_fallback,
        "positivity_fallback_count": positivity_fallback,
        "maximum_expansion_level": int(np.max(block.expansion_level)),
        "maximum_donor_count": int(np.max(block.donor_count)),
    }
    return arrays, details


def _face_count(n: int) -> int:
    return base._face_count(n)


def _raw_keys(n: int, indices: np.ndarray) -> np.ndarray:
    return base._raw_keys(n, indices)


def _face_index(n: int, axis: int, i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
    return base._face_index(n, axis, i, j, k)


def _chunk_valid(path: Path, identity: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        arrays, metadata = _load_npz(path, CHUNK_SCHEMA)
    except Exception:
        return False
    return metadata.get("identity") == _json(identity) and metadata.get("status") == "complete"


def _statistics(actual: np.ndarray, exact: np.ndarray, data: Any) -> dict[str, Any]:
    return integrated._compact_statistics(np.asarray(actual), np.asarray(exact), data)


def _case(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args.config); output = _path(config,"output"); n = int(args.resolution)
    started = time.perf_counter()
    prepare, prepare_meta = _load_npz(output/f"N{n}.prepare.npz", PREPARE_SCHEMA)
    raw_owner = prepare["raw_owner"].astype(np.int64)
    owner_count = len(prepare["owner_volume"])
    accum = {
        name: np.zeros((len(FIELD_NAMES),owner_count,4),dtype=np.float64)
        for name in ("evolution_volume",)
    }
    del accum
    evolution_volume=np.zeros(owner_count); physical_volume=np.zeros(owner_count)
    candidate={field:{term:np.zeros((owner_count,4)) for term in TERMS} for field in FIELD_NAMES}
    candidate_directional={field:{term:np.zeros((owner_count,3,4)) for term in TERMS} for field in FIELD_NAMES}
    reference_physical={field:{term:np.zeros((owner_count,4)) for term in TERMS} for field in FIELD_NAMES}
    reference_evolution={field:{term:np.zeros((owner_count,4)) for term in TERMS} for field in FIELD_NAMES}
    face_correction={field:np.zeros((owner_count,4)) for field in FIELD_NAMES}
    face_directional={field:np.zeros((owner_count,3,4)) for field in FIELD_NAMES}
    chunk_root=output/f"N{n}.chunks"
    face_details=[]; cell_details=[]
    cell_paths=sorted(chunk_root.glob("cell_*.npz"))+sorted(chunk_root.glob(f"N{n}-cell-*.npz"))
    face_paths=sorted(chunk_root.glob("face_*.npz"))+sorted(chunk_root.glob(f"N{n}-face-*.npz"))
    if not cell_paths or not face_paths:
        raise RuntimeError("global assembly requires validated face and cell chunks")
    for path in cell_paths:
        arrays, metadata=_load_npz(path,CHUNK_SCHEMA); cell_details.append(metadata["details"])
        indices=arrays["indices"].astype(np.int64); owners=raw_owner[indices]
        np.add.at(evolution_volume,owners,arrays["evolution_volume"])
        np.add.at(physical_volume,owners,arrays["physical_volume"])
        for field in FIELD_NAMES:
            for term in TERMS:
                np.add.at(candidate[field][term],owners,arrays[f"candidate:{field}:{term}"])
                np.add.at(candidate_directional[field][term],owners,arrays[f"candidate_directional:{field}:{term}"])
                np.add.at(reference_physical[field][term],owners,arrays[f"reference_physical:{field}:{term}"])
                np.add.at(reference_evolution[field][term],owners,arrays[f"reference_evolution:{field}:{term}"])
    for path in face_paths:
        arrays,metadata=_load_npz(path,CHUNK_SCHEMA); face_details.append(metadata["details"])
        for row in range(len(arrays["indices"])):
            axis=int(arrays["keys"][row,0])
            for side,valid_name,raw_name in ((0,"lower_valid","lower_raw"),(1,"upper_valid","upper_raw")):
                if not bool(arrays[valid_name][row]): continue
                owner=raw_owner[int(arrays[raw_name][row])]
                for state,field in enumerate(FIELD_NAMES):
                    value=arrays["correction"][state,row,side]
                    face_correction[field][owner]+=value
                    face_directional[field][owner,axis]+=value
    if np.any(evolution_volume<=0) or np.any(physical_volume<=0):
        raise RuntimeError("incomplete complete-owner volume coverage")
    data=integrated._load_resolution(_path(config,"geometry"),_path(config,"baseline"),n)
    if not np.array_equal(data.owner_keys,prepare["owner_keys"]):
        raise ValueError("owner ordering mismatch")
    arrays_out={"owner_keys":prepare["owner_keys"],"owner_volume":physical_volume,"evolution_volume":evolution_volume}
    statistics:dict[str,Any]={}
    closure=0.0
    for field in FIELD_NAMES:
        statistics[field]={}
        target={term:reference_physical[field][term]/physical_volume[:,None] for term in TERMS}
        centered={term:candidate[field][term]/evolution_volume[:,None] for term in TERMS}
        upwind={term:centered[term].copy() for term in TERMS}
        upwind["material"] += face_correction[field]/evolution_volume[:,None]
        upwind["total"] = upwind["material"]+upwind["remainder"]
        closure=max(closure,float(np.max(np.abs(upwind["material"]+upwind["remainder"]-upwind["total"]))))
        for term in TERMS:
            arrays_out[f"target:{field}:{term}"]=target[term]
            arrays_out[f"candidate:centered:{field}:{term}"]=centered[term]
            arrays_out[f"candidate:U:{field}:{term}"]=upwind[term]
            arrays_out[f"candidate_directional:centered:{field}:{term}"]=candidate_directional[field][term]/evolution_volume[:,None,None]
            u_directional=arrays_out[f"candidate_directional:centered:{field}:{term}"].copy()
            if term in ("material","total"):
                u_directional += face_directional[field]/evolution_volume[:,None,None]
            arrays_out[f"candidate_directional:U:{field}:{term}"]=u_directional
            arrays_out[f"reference_evolution:{field}:{term}"]=reference_evolution[field][term]/evolution_volume[:,None]
        for action, values in (("centered",centered),("U",upwind)):
            statistics[field][action]={}
            for term in TERMS:
                statistics[field][action][term]={
                    equation:_statistics(values[term][:,eq],target[term][:,eq],data)
                    for eq,equation in enumerate(EQUATIONS)
                }
    output_npz=output/f"N{n}.npz"
    metadata={
        "schema":SCHEMA,"status":"complete","resolution":n,
        "candidate":config["candidate"],"scope":config["scope"],"primary_action":PRIMARY_ACTION,
        "sources":_source_identity(config),"prepare":_identity(output/f"N{n}.prepare.npz"),
        "statistics":statistics,
        "verification":{
            "finite_complete_owner_coverage":all(np.all(np.isfinite(v)) for v in arrays_out.values()),
            "M_plus_R_closure_max":closure,
            "constraint_residual_max":max([d["constraint_residual_max"] for d in cell_details+face_details],default=0.0),
            "physical_wall_model":"legacy-velocity-trace baseline unchanged; homogeneous point relation only in research reconstruction",
        },
        "timing":{"seconds":time.perf_counter()-started,"maximum_rss_gib":_max_rss_gib()},
    }
    metadata["array_sha256"]=_array_hash(*(arrays_out[name] for name in sorted(arrays_out)))
    _write_npz(output_npz,arrays_out,metadata); _write_json(output/f"N{n}.json",metadata)
    _event("case_complete",resolution=n,seconds=metadata["timing"]["seconds"])
    return metadata


def _selected_complete_owners(data: Any) -> tuple[np.ndarray, dict[str,list[int]]]:
    selected:set[int]=set(); strata:dict[str,list[int]]={}
    for name,mask in data.masks.items():
        candidates=np.flatnonzero(mask)
        if len(candidates):
            pick=[int(candidates[len(candidates)//2])]
            strata[name]=pick; selected.update(pick)
    keys=np.asarray(data.owner_keys)
    seam=np.flatnonzero((keys[:,1]==0)|(keys[:,1]==data.resolution-1)|(keys[:,2]==0)|(keys[:,2]==data.resolution-1))
    seam_pick=sorted(set(map(int,seam[[0,-1]]))) if len(seam) else []
    strata["periodic_seams"]=seam_pick; selected.update(seam_pick)
    return np.asarray(sorted(selected),dtype=np.int64),strata


def _reference_on_raw_cells(context:Any,reference:Any,raw_indices:np.ndarray,order:int,time_value:float) -> tuple[np.ndarray,np.ndarray]:
    keys=base._raw_keys(context.resolution,raw_indices)
    if order==3:
        points,weights=base._cell_quadrature(context,keys)
    else:
        # Same tensor-product logical cell rule, generalized to q5/q7.
        nodes,one=np.polynomial.legendre.leggauss(order)
        points=np.empty((len(keys),order**3,3)); weights=np.empty((len(keys),order**3))
        for row,(i,j,k) in enumerate(keys):
            axes=[]; axis_weights=[]
            for faces,index in ((context.x_faces,i),(context.y_faces,j),(context.z_faces,k)):
                lo=float(faces[index]); hi=float(faces[index+1])
                axes.append(0.5*(lo+hi)+0.5*(hi-lo)*nodes)
                axis_weights.append(0.5*(hi-lo)*one)
            mesh=np.meshgrid(*axes,indexing="ij"); wmesh=np.meshgrid(*axis_weights,indexing="ij")
            points[row]=np.stack(mesh,axis=-1).reshape(-1,3); weights[row]=np.prod(np.stack(wmesh,axis=-1),axis=-1).reshape(-1)
    flat=points.reshape(-1,3); prepared=reference.prepare(flat); metric=reference._metric(flat)
    physical=weights*np.asarray(metric["J"]).reshape(len(keys),-1)
    numerator=np.empty((len(FIELD_NAMES),len(TERMS),len(keys),4))
    for state,field in enumerate(FIELD_NAMES):
        values,gradients=_evaluate_fields(field,reference,flat,time_value)
        terms=_continuum_terms(values,gradients,prepared)
        for term in range(len(TERMS)):
            numerator[state,term]=np.sum(physical[...,None]*terms[term].reshape(len(keys),-1,4),axis=1)
    return numerator,np.sum(physical,axis=1)


def _preflight(args: argparse.Namespace) -> dict[str,Any]:
    config=_config(args.config); output=_path(config,"output"); n=int(args.resolution)
    prepare,_=_load_npz(output/f"N{n}.prepare.npz",PREPARE_SCHEMA)
    context=cubic._load_context(_path(config,"geometry"),_path(config,"baseline"),n)
    reference=integrated._reference(_path(config,"reference_sidecar"),verify_hashes=False)
    data=integrated._load_resolution(_path(config,"geometry"),_path(config,"baseline"),n)
    owners,strata=_selected_complete_owners(data)
    raw=np.flatnonzero(np.isin(prepare["raw_owner"],owners))
    keys=base._raw_keys(n,raw); faces:set[int]=set()
    for axis in range(3):
        upper=keys.copy(); upper[:,axis]+=1
        faces.update(map(int,base._face_index(n,axis,*keys.T)))
        faces.update(map(int,base._face_index(n,axis,*upper.T)))
    face_indices=np.asarray(sorted(faces),dtype=np.int64)
    fa,fd=_compute_faces(context,reference,prepare,face_indices,time_value=float(config["time"]))
    ca,cd=_compute_cells(context,reference,prepare,raw,time_value=float(config["time"]),curl_step=float(config["curl_step"]))
    raw_owner=prepare["raw_owner"].astype(np.int64)
    reference_by_order={}
    for order in (3,5,7):
        numerator,volume=_reference_on_raw_cells(context,reference,raw,order,float(config["time"]))
        owner_numerator=np.zeros((len(FIELD_NAMES),len(TERMS),len(owners),4)); owner_volume=np.zeros(len(owners))
        lookup={owner:row for row,owner in enumerate(owners.tolist())}
        local=np.asarray([lookup[int(owner)] for owner in raw_owner[raw]],dtype=np.int64)
        np.add.at(owner_volume,local,volume)
        for state in range(len(FIELD_NAMES)):
            for term in range(len(TERMS)):
                np.add.at(owner_numerator[state,term],local,numerator[state,term])
        reference_by_order[order]=owner_numerator/owner_volume[None,None,:,None]
    budget={}
    for state,field in enumerate(FIELD_NAMES):
        budget[field]={}
        for term_index,term in enumerate(TERMS):
            budget[field][term]={}
            for equation_index,equation in enumerate(EQUATIONS):
                q3=reference_by_order[3][state,term_index,:,equation_index]
                q5=reference_by_order[5][state,term_index,:,equation_index]
                q7=reference_by_order[7][state,term_index,:,equation_index]
                budget[field][term][equation]={
                    "q3_minus_q7_rms":float(np.sqrt(np.mean((q3-q7)**2))),
                    "q5_minus_q7_rms":float(np.sqrt(np.mean((q5-q7)**2))),
                    "bounded_absolute_budget":float(max(np.max(np.abs(q3-q7)),np.max(np.abs(q5-q7)))),
                }
    finite=all(np.all(np.isfinite(v)) for values in (fa,ca) for v in values.values() if v.dtype.kind in "fci")
    payload={
        "schema":PREFLIGHT_SCHEMA,"status":"complete","resolution":n,
        "passes_implementation_preflight":bool(finite and fd["wall_face_count"]>0),
        "complete_owner_indices":owners,"complete_raw_cell_count":len(raw),"face_count":len(face_indices),
        "strata":strata,"covers_axis_RLP_wall_interior_and_periodic_seams":True,
        "face_details":fd,"cell_details":cd,"reference_qualification":budget,
        "reference_rule":"global q3 target; bounded complete-owner q5/q7 differences qualify reference, not candidate acceptance",
        "sources":_source_identity(config),
    }
    snapshots={**{f"face:{k}":v for k,v in fa.items()},**{f"cell:{k}":v for k,v in ca.items()}}
    payload["snapshot_sha256"]=_array_hash(*(snapshots[name] for name in sorted(snapshots)))
    payload["array_sha256"]=_array_hash(*(snapshots[name] for name in sorted(snapshots)))
    _write_npz(output/f"N{n}.preflight.npz",snapshots,payload); _write_json(output/f"N{n}.preflight.json",payload)
    _event("preflight_complete",resolution=n,owners=len(owners),raw_cells=len(raw))
    return payload


def _orders(errors:Sequence[float]) -> list[float]:
    resolutions=(32,48,64)
    return [float(np.log(errors[i]/errors[i+1])/np.log(resolutions[i+1]/resolutions[i])) for i in range(2)]


def _merge(args:argparse.Namespace) -> dict[str,Any]:
    config=_config(args.config); output=_path(config,"output")
    cases=[json.loads((output/f"N{n}.json").read_text()) for n in (32,48,64)]
    preflight=[json.loads((output/f"N{n}.preflight.json").read_text()) for n in (32,48,64)]
    results={}; gates=[]
    for field in FIELD_NAMES:
        results[field]={}
        for equation in EQUATIONS:
            primary=PRIMARY_ACTION[equation]
            results[field][equation]={"primary_action":primary,"actions":{}}
            for action in ACTIONS:
                results[field][equation]["actions"][action]={}
                for term in TERMS:
                    errors=[case["statistics"][field][action][term][equation]["absolute_l2"] for case in cases]
                    orders=_orders(errors) if min(errors)>0 else [None,None]
                    entry={"errors":errors,"orders":orders,"acceptance_role":"primary" if action==primary else "diagnostic"}
                    if action==primary and not (equation=="vorticity" and term=="remainder"):
                        entry["passes_both_intervals"]=bool(all(order is not None and order>=1.8 for order in orders))
                        gates.append(entry["passes_both_intervals"])
                    results[field][equation]["actions"][action][term]=entry
    reference={}; reference_gate=[]
    finest=cases[-1]
    for field in FIELD_NAMES:
        reference[field]={}
        for equation in EQUATIONS:
            reference[field][equation]={}
            primary=PRIMARY_ACTION[equation]
            for term in TERMS:
                budget=float(preflight[-1]["reference_qualification"][field][term][equation]["bounded_absolute_budget"])
                error=float(finest["statistics"][field][primary][term][equation]["absolute_l2"])
                structural_zero=equation=="vorticity" and term=="remainder"
                ratio=0.0 if structural_zero else budget/max(error,1.0e-300)
                passes=structural_zero or ratio<0.1
                reference[field][equation][term]={"N64_bounded_absolute_budget":budget,"fraction_of_N64_primary_error":ratio,"passes_10_percent":passes,"structural_zero":structural_zero}
                reference_gate.append(passes)
    payload={
        "schema":SUMMARY_SCHEMA,"status":"computation completed",
        "invariants_checked":all(case["verification"]["finite_complete_owner_coverage"] and case["verification"]["M_plus_R_closure_max"]<1.0e-11 for case in cases),
        "global_accuracy_passed":bool(all(gates) and all(reference_gate)),
        "acceptance":"orders >=1.8 on both intervals for primary M/R/total components; exact-zero vorticity remainder exempt; diagnostic action never hidden in pooled score",
        "results":results,"reference_qualification":reference,"primary_action":PRIMARY_ACTION,
        "candidate":config["candidate"],"scope":config["scope"],"sources":_source_identity(config),
    }
    _write_json(output/"summary.json",payload); _event("merge_complete",global_accuracy_passed=payload["global_accuracy_passed"])
    return payload


def _validate(args:argparse.Namespace) -> dict[str,Any]:
    config=_config(args.config); output=_path(config,"output"); stage=args.stage
    if stage=="preflight":
        payload=json.loads((output/f"N{args.resolution}.preflight.json").read_text())
        if payload.get("schema")!=PREFLIGHT_SCHEMA or not payload.get("passes_implementation_preflight"):
            raise ValueError("invalid preflight")
    elif stage.startswith("prepare_N"):
        n=int(stage.split("N",1)[1]); _arrays,payload=_load_npz(output/f"N{n}.prepare.npz",PREPARE_SCHEMA)
        if payload.get("status")!="complete": raise ValueError("invalid preparation")
    elif stage.startswith("case_N"):
        n=int(stage.split("N",1)[1]); _arrays,payload=_load_npz(output/f"N{n}.npz",SCHEMA)
        if payload.get("status")!="complete" or not payload["verification"]["finite_complete_owner_coverage"]: raise ValueError("invalid case")
    elif stage=="merge":
        payload=json.loads((output/"summary.json").read_text())
        if payload.get("schema")!=SUMMARY_SCHEMA or payload.get("status")!="computation completed": raise ValueError("invalid merge")
    else:
        raise ValueError(f"unknown validation stage {stage}")
    return payload
