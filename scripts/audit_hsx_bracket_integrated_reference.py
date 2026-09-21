#!/usr/bin/env python3
"""Qualify and replay an integrated continuum reference for the HSX bracket.

This is an offline research audit.  It does not change the inexpensive
midpoint MMS, reconstruction policies, boundary model, or production source.
Smooth fields use direct physical-volume integration.  Vorticity may use the
exact logical integration-by-parts identity

    integral a^i d_i omega dq
      = boundary integral omega a^i n_i dq_face
        - integral omega d_i a^i dq,

where ``a = -(b_cov x grad(phi))/(rho_star B)``.  The complete face and volume
terms are always retained and are checked against direct volume integration on
bounded actual-HSX cells before the identity is used globally.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
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
from unittest.mock import patch

import jax
import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import audit_hsx_cubic_derivative_global as cubic  # noqa: E402
import audit_hsx_cubic_global_bracket as global_case  # noqa: E402
import audit_hsx_owner_face_derivative as derivative  # noqa: E402
import audit_hsx_p03_fine_interval as bridge  # noqa: E402
import audit_hsx_poisson_vorticity as base  # noqa: E402
import audit_hsx_value_derivative_cross as cross  # noqa: E402
from hsx_mms_continuum_reference import build_continuum_reference_from_sidecar  # noqa: E402
import simulate_hsx_mms as mms  # noqa: E402


REFERENCE_SCHEMA = "drbx.hsx-bracket-integrated-reference-v1"
ACTION_SCHEMA = "drbx.hsx-bracket-global-actions-v1"
QUALIFICATION_SCHEMA = "drbx.hsx-bracket-reference-qualification-v1"
CASE_SCHEMA = "drbx.hsx-bracket-integrated-replay-case-v1"
SUMMARY_SCHEMA = "drbx.hsx-bracket-integrated-replay-summary-v1"
FIELDS = (
    "actual_vorticity",
    "smooth_regular_scalar",
    "smooth_eta_varying_scalar",
)
RULES = ("P", "G", "O")
ACTIONS = ("upwind", "A", "B", "C")
REGIONS = (
    "ordinary",
    "agglomerated_interior",
    "true_size_change_interface",
    "axis_ring",
    "physical_boundary_footprint",
)
SAMPLE_COUNTS = {
    "ordinary": 16,
    "agglomerated_interior": 16,
    "true_size_change_interface": 12,
    "axis_ring": 8,
    "physical_boundary_footprint": 12,
}
BOUNDARY_CONTRACT = global_case.BOUNDARY_CONTRACT


def _max_rss_gib() -> float:
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


def _json_value(value: Any) -> Any:
    return base._json_value(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _progress(stage: str, **details: Any) -> None:
    print(
        json.dumps({"event": "hsx_bracket_reference", "stage": stage, **details}, sort_keys=True),
        flush=True,
    )


def _source_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size": resolved.stat().st_size,
    }


def _reference(
    sidecar: Path,
    *,
    finite_difference_step: float = 2.0e-4,
    verify_hashes: bool = True,
) -> Any:
    result = build_continuum_reference_from_sidecar(
        sidecar,
        verify_hashes=verify_hashes,
        tau=mms.PHYSICAL_PARAMETERS["tau"],
        mi_over_me=mms.PHYSICAL_PARAMETERS["mi_over_me"],
        rho_star=mms.PHYSICAL_PARAMETERS["rho_star"],
        Ve_nu=mms.PHYSICAL_PARAMETERS["Ve_nu"],
        perp_diffusion=mms.PHYSICAL_PARAMETERS["density_D_perp"],
        enable_generalized_potential=True,
    )
    result.finite_difference_step = float(finite_difference_step)
    return result


@dataclass
class ResolutionData:
    resolution: int
    artifact_path: Path
    artifact: Any
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]
    owner_volume: np.ndarray
    raw_owner: np.ndarray
    owner_keys: np.ndarray
    masks: dict[str, np.ndarray]
    ordinary_layer: np.ndarray
    active_full: np.ndarray


class StructuredOmegaEvaluator:
    """Focused spline evaluator from exact continuum raw-cell midpoint samples."""

    def __init__(
        self, values: np.ndarray, faces: Sequence[np.ndarray], *, pad: int = 6, order: int = 5
    ):
        raw = np.asarray(values, dtype=np.float64)
        if raw.ndim != 3 or len(set(raw.shape)) != 1:
            raise ValueError("structured omega samples must be one cubic tensor")
        self.shape = raw.shape
        self.pad = int(pad)
        self.order = int(order)
        self.faces = tuple(np.asarray(axis, dtype=np.float64) for axis in faces)
        self.centers = tuple(0.5 * (axis[:-1] + axis[1:]) for axis in self.faces)
        self.spacing = tuple(float(np.mean(np.diff(axis))) for axis in self.faces)
        for axis, spacing in zip(self.faces, self.spacing):
            if not np.allclose(np.diff(axis), spacing, rtol=0.0, atol=1.0e-13):
                raise ValueError("structured omega interpolation requires uniform logical faces")
        periodic = np.pad(raw, ((0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode="wrap")
        self.padded = self._radial_extrapolate(periodic)
        self.coefficients = ndimage.spline_filter(
            self.padded, order=self.order, mode="nearest"
        )

    def _radial_extrapolate(self, values: np.ndarray) -> np.ndarray:
        n = values.shape[0]
        result = np.empty((n + 2 * self.pad,) + values.shape[1:], dtype=np.float64)
        result[self.pad:self.pad + n] = values
        count = self.order + 1
        nodes = np.arange(count, dtype=np.float64)

        def lagrange(target: float) -> np.ndarray:
            weights = np.ones(count, dtype=np.float64)
            for i in range(count):
                for j in range(count):
                    if i != j:
                        weights[i] *= (target - nodes[j]) / (nodes[i] - nodes[j])
            return weights

        left = values[:count].reshape(count, -1)
        right = values[-count:].reshape(count, -1)
        for offset in range(self.pad):
            result[offset] = (lagrange(offset - self.pad) @ left).reshape(values.shape[1:])
            target = float(count + offset)
            result[self.pad + n + offset] = (lagrange(target) @ right).reshape(values.shape[1:])
        return result

    def value(self, points: np.ndarray) -> np.ndarray:
        q = np.asarray(points, dtype=np.float64).copy()
        q[:, 1] = np.mod(q[:, 1] - self.faces[1][0], self.faces[1][-1] - self.faces[1][0]) + self.faces[1][0]
        q[:, 2] = np.mod(q[:, 2] - self.faces[2][0], self.faces[2][-1] - self.faces[2][0]) + self.faces[2][0]
        q[:, 0] = np.clip(q[:, 0], self.faces[0][0], self.faces[0][-1])
        coordinates = []
        for axis in range(3):
            coordinate = (q[:, axis] - self.centers[axis][0]) / self.spacing[axis] + self.pad
            coordinates.append(coordinate)
        return ndimage.map_coordinates(
            self.coefficients,
            np.stack(coordinates, axis=0),
            order=self.order,
            mode="nearest",
            prefilter=False,
        )

    def gradient(self, points: np.ndarray, *, step: float = 2.0e-4) -> np.ndarray:
        q = np.asarray(points, dtype=np.float64)
        gradients = []
        for axis in range(3):
            h = np.full(len(q), float(step), dtype=np.float64)
            if axis == 0:
                h = np.minimum(h, 0.2 * np.maximum(q[:, 0], 1.0e-10))
                h = np.minimum(h, 0.2 * np.maximum(1.0 - q[:, 0], 1.0e-10))
                h = np.maximum(h, 1.0e-8)
            values = []
            for multiplier in (-2.0, -1.0, 1.0, 2.0):
                shifted = q.copy()
                shifted[:, axis] += multiplier * h
                values.append(self.value(shifted))
            gradients.append((values[0] - 8.0 * values[1] + 8.0 * values[2] - values[3]) / (12.0 * h))
        return np.stack(gradients, axis=-1)

    def midpoint_reproduction_max_abs(self, values: np.ndarray) -> float:
        mesh = np.stack(np.meshgrid(*self.centers, indexing="ij"), axis=-1).reshape(-1, 3)
        return float(np.max(np.abs(self.value(mesh) - np.asarray(values).reshape(-1))))


def _load_resolution(geometry: Path, baseline: Path, resolution: int) -> ResolutionData:
    arrays, metadata = derivative._load_owner_inputs(geometry, baseline, resolution)
    arrays.pop("raw_by_owner_object", None)
    artifact_path, artifact = base._load_artifact(geometry, resolution)
    active_full = np.asarray(artifact.owner_geometry.topology.is_active_owner, dtype=bool)
    owner_flat_ids = arrays["owner_flat_ids"]
    if not np.array_equal(np.flatnonzero(active_full.reshape(-1)), owner_flat_ids):
        raise ValueError("owner membership differs between topology loaders")
    compact = np.full(resolution**3, -1, dtype=np.int64)
    compact[owner_flat_ids] = np.arange(len(owner_flat_ids))
    raw_owner = compact[arrays["aggregate_id"]]
    if np.any(raw_owner < 0):
        raise ValueError("raw member lacks an active owner")
    owner_volume = np.bincount(
        raw_owner, weights=arrays["raw_volume"], minlength=len(owner_flat_ids)
    )
    stored = np.asarray(artifact.owner_geometry.aggregate_chart_volume).reshape(-1)[owner_flat_ids]
    mismatch = np.max(np.abs(owner_volume - stored) / np.maximum(np.abs(stored), 1.0e-300))
    if mismatch > 1.0e-12:
        raise ValueError("raw-member and frozen owner volumes differ")
    standard = base._region_masks(artifact.global_geometry, artifact.owner_geometry)
    owner_coordinates = np.unravel_index(
        np.asarray(arrays["aggregate_id"], dtype=np.int64).reshape(active_full.shape),
        active_full.shape,
    )
    cells = SimpleNamespace(
        owner_i=np.asarray(owner_coordinates[0], dtype=np.int64),
        owner_j=np.asarray(owner_coordinates[1], dtype=np.int64),
        owner_k=np.asarray(owner_coordinates[2], dtype=np.int64),
    )
    refined, ordinary_layer_full, _adjacency = bridge._refined_masks(
        artifact.owner_geometry, cells, standard
    )
    # The bridge helper expects runtime control-volume cells in some historical
    # artifacts.  Recompute through the already qualified global helper when
    # the lightweight object does not expose them.
    if any(np.asarray(mask).shape != active_full.shape for mask in refined.values()):
        raise ValueError("regional mask shape mismatch")
    owner_keys = np.stack(np.unravel_index(owner_flat_ids, active_full.shape), axis=-1)
    return ResolutionData(
        resolution=resolution,
        artifact_path=artifact_path,
        artifact=artifact,
        arrays=arrays,
        metadata=metadata,
        owner_volume=owner_volume,
        raw_owner=raw_owner,
        owner_keys=owner_keys,
        masks={name: np.asarray(mask).reshape(-1)[owner_flat_ids] for name, mask in refined.items()},
        ordinary_layer=np.asarray(ordinary_layer_full).reshape(-1)[owner_flat_ids],
        active_full=active_full,
    )


def _geometry_identity(geometry: Path, resolution: int) -> dict[str, Any]:
    alias = geometry / f"{resolution}x{resolution}x{resolution}"
    prototype = geometry.parents[1] / "prototype_runs" / "geometry" / f"hsx_fci_{resolution}x{resolution}x{resolution}"
    if alias.resolve() != prototype.resolve():
        raise ValueError("campaign geometry alias does not resolve to supervisor prototype geometry")
    files = {}
    for name in ("manifest.json", "base_geometry.npz", "rlp_topology.npz"):
        left = _source_identity(alias / name)
        right = _source_identity(prototype / name)
        if left["sha256"] != right["sha256"]:
            raise ValueError(f"geometry payload differs for {name}")
        files[name] = left
    return {
        "campaign_alias": str(alias.resolve()),
        "supervisor_prototype": str(prototype.resolve()),
        "same_resolved_directory": True,
        "payloads": files,
    }


def _choose_sample(data: ResolutionData, seed: int) -> tuple[np.ndarray, dict[str, Any], dict[int, list[str]]]:
    rng = np.random.default_rng(int(seed) + data.resolution)
    groups: dict[str, Any] = {}
    supplemental: dict[int, list[str]] = {}
    total_volume = float(np.sum(data.owner_volume))
    for region, count in SAMPLE_COUNTS.items():
        indices = np.flatnonzero(data.masks[region])
        if len(indices) == 0:
            raise ValueError(f"empty sampling stratum {region}")
        probability = data.owner_volume[indices] / np.sum(data.owner_volume[indices])
        draws = rng.choice(indices, int(count), replace=True, p=probability)
        groups[region] = {
            "draws": draws,
            "owner_population": len(indices),
            "physical_volume": float(np.sum(data.owner_volume[indices])),
            "global_volume_fraction": float(np.sum(data.owner_volume[indices]) / total_volume),
        }
    # Eta seam controls are diagnostics and never alter stratum weights.
    eta_index = data.owner_keys[:, 2]
    for k in (0, data.resolution - 1):
        pool = np.flatnonzero(eta_index == k)
        probability = data.owner_volume[pool] / np.sum(data.owner_volume[pool])
        for owner in rng.choice(pool, min(5, len(pool)), replace=False, p=probability):
            supplemental.setdefault(int(owner), []).append("eta_seam")
    owners = np.unique(
        np.concatenate(
            [*(np.asarray(item["draws"], dtype=np.int64) for item in groups.values()),
             np.asarray(list(supplemental), dtype=np.int64)]
        )
    )
    return owners, groups, supplemental


def _gauss(order: int, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    meshes = np.meshgrid(*([nodes] * dimension), indexing="ij")
    points = np.stack(meshes, axis=-1).reshape(-1, dimension)
    weight_mesh = np.meshgrid(*([weights] * dimension), indexing="ij")
    combined = np.prod(np.stack(weight_mesh, axis=-1), axis=-1).reshape(-1)
    return points, combined


def _raw_cells(data: ResolutionData, owners: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if owners is None:
        raw = np.arange(data.resolution**3, dtype=np.int64)
    else:
        selected = np.zeros(len(data.owner_volume), dtype=bool)
        selected[np.asarray(owners, dtype=np.int64)] = True
        raw = np.flatnonzero(selected[data.raw_owner])
    return raw, data.raw_owner[raw]


def _cell_boxes(data: ResolutionData, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = np.stack(np.unravel_index(raw, (data.resolution,) * 3), axis=-1)
    faces = (
        data.arrays["grid_x_faces"],
        data.arrays["grid_y_faces"],
        data.arrays["grid_z_faces"],
    )
    lo = np.stack([faces[axis][cells[:, axis]] for axis in range(3)], axis=-1)
    hi = np.stack([faces[axis][cells[:, axis] + 1] for axis in range(3)], axis=-1)
    return cells, lo, hi


def _points_in_boxes(lo: np.ndarray, hi: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = _gauss(order, 3)
    half = 0.5 * (hi - lo)
    center = 0.5 * (hi + lo)
    points = center[:, None, :] + half[:, None, :] * nodes[None, :, :]
    physical_weights = np.prod(half, axis=-1)[:, None] * weights[None, :]
    return points, physical_weights


def _face_points(
    lo: np.ndarray, hi: np.ndarray, axis: int, side: int, order: int
) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = _gauss(order, 2)
    other = [value for value in range(3) if value != axis]
    result = np.empty((len(lo), len(nodes), 3), dtype=np.float64)
    result[:, :, axis] = (lo if side == 0 else hi)[:, axis, None]
    half = 0.5 * (hi[:, other] - lo[:, other])
    center = 0.5 * (hi[:, other] + lo[:, other])
    result[:, :, other] = center[:, None, :] + half[:, None, :] * nodes[None, :, :]
    physical_weights = np.prod(half, axis=-1)[:, None] * weights[None, :]
    return result, physical_weights


def _omega_value(reference: Any, points: np.ndarray) -> np.ndarray:
    q = np.asarray(points, dtype=np.float64).copy()
    # Omega contains a differentiated metric tensor.  On the physical outer
    # radial face, evaluate its one-sided interior limit so the centered
    # diagnostic derivative never leaves the evaluator domain.  The offset is
    # tied to the audited finite-difference step and is therefore included in
    # the half/base/double-step sensitivity.
    boundary = q[:, 0] >= 1.0 - 1.0e-14
    q[boundary, 0] = 1.0 - 3.0 * float(reference.finite_difference_step)
    _value, du, dtheta, deta, _dt, hessian = reference._psi_raw(q)
    gradient = np.stack((du, dtheta, deta), axis=-1)
    return np.asarray(reference._perpendicular_operator(q, gradient, hessian), dtype=np.float64)


def _phi_gradient(reference: Any, points: np.ndarray, time_value: float) -> np.ndarray:
    raw = reference._fields_raw(np.asarray(points, dtype=np.float64), float(time_value))["phi"]
    return np.stack(raw[1:4], axis=-1)


def _advection_coefficient(reference: Any, points: np.ndarray, time_value: float) -> np.ndarray:
    q = np.asarray(points, dtype=np.float64)
    metric = reference._metric(q)
    return -np.cross(metric["bcov"], _phi_gradient(reference, q, time_value)) / (
        float(reference.rho_star) * metric["B"][:, None]
    )


def _coefficient_divergence(reference: Any, points: np.ndarray, time_value: float) -> np.ndarray:
    q = np.asarray(points, dtype=np.float64)
    result = np.zeros(len(q), dtype=np.float64)
    for axis in range(3):
        result += reference._derivative(
            lambda shifted, component=axis: _advection_coefficient(
                reference, shifted, time_value
            )[:, component],
            q,
            axis,
        )
    return result


def _omega_gradient(reference: Any, points: np.ndarray) -> np.ndarray:
    q = np.asarray(points, dtype=np.float64)
    return np.stack(
        tuple(reference._derivative(lambda shifted: _omega_value(reference, shifted), q, axis) for axis in range(3)),
        axis=-1,
    )


def _smooth(reference: Any, field: str, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return global_case.continuous._smooth_value_gradient(
        field, np.asarray(points, dtype=np.float64), reference.eta_period
    )


def _structured_omega(data: ResolutionData, baseline: Path) -> tuple[StructuredOmegaEvaluator, dict[str, Any]]:
    path = baseline / f"N{data.resolution}.reference.npz"
    with np.load(path, allow_pickle=False) as source:
        values = np.asarray(source["actual_vorticity"], dtype=np.float64)
    faces = (
        data.arrays["grid_x_faces"],
        data.arrays["grid_y_faces"],
        data.arrays["grid_z_faces"],
    )
    evaluator = StructuredOmegaEvaluator(values, faces)
    return evaluator, {
        "source": _source_identity(path),
        "source_array_sha256": _array_hash(values),
        "interpolation": "tensor quintic B-spline; periodic theta/eta padding; quintic radial extrapolation",
        "gradient": "fourth-order centered difference of the focused interpolant",
        "midpoint_reproduction_max_abs": evaluator.midpoint_reproduction_max_abs(values),
        "uses_owner_averages": False,
        "uses_fitted_numerical_generator": False,
    }


def _accumulate(
    result: np.ndarray,
    owner: np.ndarray,
    values: np.ndarray,
) -> None:
    np.add.at(result, owner, values)


def _evaluate_reference(
    data: ResolutionData,
    reference: Any,
    *,
    order: int,
    owners: np.ndarray | None,
    fields: Sequence[str],
    vorticity_method: str,
    time_value: float,
    chunk_cells: int,
    omega_evaluator: StructuredOmegaEvaluator | None = None,
    omega_gradient_step: float = 2.0e-4,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    raw, raw_owner = _raw_cells(data, owners)
    numerator = {field: np.zeros(len(data.owner_volume), dtype=np.float64) for field in fields}
    field_integral = {field: np.zeros(len(data.owner_volume), dtype=np.float64) for field in fields}
    integrated_volume = np.zeros(len(data.owner_volume), dtype=np.float64)
    batches = 0
    for first in range(0, len(raw), int(chunk_cells)):
        last = min(first + int(chunk_cells), len(raw))
        batch_raw = raw[first:last]
        batch_owner = raw_owner[first:last]
        _cells, lo, hi = _cell_boxes(data, batch_raw)
        points3, weights3 = _points_in_boxes(lo, hi, order)
        flat = points3.reshape(-1, 3)
        qcount = points3.shape[1]
        metric = reference._metric(flat)
        jacobian = np.abs(metric["J"]).reshape(len(batch_raw), qcount)
        volume_part = np.sum(weights3 * jacobian, axis=1)
        _accumulate(integrated_volume, batch_owner, volume_part)
        grad_phi = _phi_gradient(reference, flat, time_value)
        for field in fields:
            if field == "actual_vorticity":
                omega = (
                    _omega_value(reference, flat)
                    if omega_evaluator is None
                    else omega_evaluator.value(flat)
                )
                field_part = np.sum(weights3 * jacobian * omega.reshape(len(batch_raw), qcount), axis=1)
                _accumulate(field_integral[field], batch_owner, field_part)
                if vorticity_method in ("direct", "structured"):
                    gradient = (
                        _omega_gradient(reference, flat)
                        if omega_evaluator is None
                        else omega_evaluator.gradient(flat, step=omega_gradient_step)
                    )
                    integrand = -np.sum(metric["bcov"] * np.cross(grad_phi, gradient), axis=-1)
                    integrand /= float(reference.rho_star) * metric["B"]
                    part = np.sum(weights3 * integrand.reshape(len(batch_raw), qcount), axis=1)
                    _accumulate(numerator[field], batch_owner, part)
                elif vorticity_method == "ibp":
                    divergence = _coefficient_divergence(reference, flat, time_value)
                    part = -np.sum(weights3 * (omega * divergence).reshape(len(batch_raw), qcount), axis=1)
                    for axis in range(3):
                        for side in (0, 1):
                            face_cells = np.arange(len(batch_raw), dtype=np.int64)
                            if axis == 0 and side == 0:
                                # The physical surface represented by u=0 has
                                # zero area.  Ordinary polar metric components
                                # are singular exactly on that coordinate face,
                                # while the axis-regular compatible flux limit
                                # is identically zero.
                                face_cells = face_cells[lo[:, 0] > 1.0e-14]
                            if len(face_cells) == 0:
                                continue
                            face_points, face_weights = _face_points(
                                lo[face_cells], hi[face_cells], axis, side, order
                            )
                            face_flat = face_points.reshape(-1, 3)
                            face_count = face_points.shape[1]
                            face_omega = _omega_value(reference, face_flat)
                            coefficient = _advection_coefficient(reference, face_flat, time_value)[:, axis]
                            sign = -1.0 if side == 0 else 1.0
                            part[face_cells] += sign * np.sum(
                                face_weights
                                * (face_omega * coefficient).reshape(len(face_cells), face_count),
                                axis=1,
                            )
                    _accumulate(numerator[field], batch_owner, part)
                else:
                    raise ValueError(f"unsupported vorticity method {vorticity_method!r}")
            else:
                value, gradient = _smooth(reference, field, flat)
                field_part = np.sum(weights3 * jacobian * value.reshape(len(batch_raw), qcount), axis=1)
                _accumulate(field_integral[field], batch_owner, field_part)
                integrand = -np.sum(metric["bcov"] * np.cross(grad_phi, gradient), axis=-1)
                integrand /= float(reference.rho_star) * metric["B"]
                part = np.sum(weights3 * integrand.reshape(len(batch_raw), qcount), axis=1)
                _accumulate(numerator[field], batch_owner, part)
        batches += 1
        _progress(
            "reference_batch",
            resolution=data.resolution,
            order=order,
            method=vorticity_method,
            completed_cells=last,
            total_cells=len(raw),
            seconds=time.perf_counter() - started,
        )
    selected = np.arange(len(data.owner_volume)) if owners is None else np.asarray(owners, dtype=np.int64)
    frozen = {
        field: numerator[field][selected] / data.owner_volume[selected] for field in fields
    }
    true_volume = {
        field: numerator[field][selected] / integrated_volume[selected] for field in fields
    }
    inputs = {
        field: field_integral[field][selected] / data.owner_volume[selected] for field in fields
    }
    details = {
        "resolution": data.resolution,
        "order": int(order),
        "vorticity_method": vorticity_method,
        "owner_count": len(selected),
        "raw_cell_count": len(raw),
        "point_count": int(len(raw) * order**3),
        "batch_count": batches,
        "seconds": time.perf_counter() - started,
        "maximum_rss_gib": _max_rss_gib(),
        "true_volume_relative_to_frozen_max": float(
            np.max(np.abs(integrated_volume[selected] / data.owner_volume[selected] - 1.0))
        ),
    }
    return frozen, true_volume, inputs, details


def _weighted_rms(values: np.ndarray, volume: np.ndarray) -> float:
    return float(np.sqrt(np.sum(volume * np.asarray(values) ** 2) / np.sum(volume)))


def _sample_global_rms(
    values: np.ndarray,
    owners: np.ndarray,
    strata: Mapping[str, Mapping[str, Any]],
) -> float:
    result = 0.0
    for item in strata.values():
        positions = np.searchsorted(owners, np.asarray(item["draws"], dtype=np.int64))
        result += float(item["global_volume_fraction"]) * float(np.mean(np.asarray(values)[positions] ** 2))
    return math.sqrt(result)


def _action_identity(args: argparse.Namespace, data: ResolutionData) -> dict[str, Any]:
    resolution = data.resolution
    reference_cache = args.baseline / f"N{resolution}.reference.npz"
    derivative_manifest = args.derivative_root / f"N{resolution}" / "manifest.json"
    rows = {
        rule: _source_identity(args.p04_root / f"N{resolution}_{rule}_rows.npz")
        for rule in ("G", "O")
    }
    dependencies = (
        SCRIPTS / "audit_hsx_cubic_global_bracket.py",
        SCRIPTS / "audit_hsx_cubic_derivative_global.py",
        SCRIPTS / "audit_hsx_p04_global_bracket.py",
        SCRIPTS / "audit_hsx_poisson_vorticity.py",
    )
    return {
        "schema": ACTION_SCHEMA,
        "resolution": resolution,
        "geometry": _geometry_identity(args.geometry, resolution),
        "baseline_reference": _source_identity(reference_cache),
        "reference_sidecar": _source_identity(args.reference_sidecar),
        "derivative_manifest": _source_identity(derivative_manifest),
        "transported_value_rows": rows,
        "dependencies": {str(path.relative_to(ROOT)): _source_identity(path) for path in dependencies},
        "rules": list(RULES),
        "actions": list(ACTIONS),
        "fields": list(FIELDS),
        "time": float(args.time),
        "precision": "float64",
        "boundary_contract": BOUNDARY_CONTRACT,
    }


def _load_action_cache(path: Path, expected_identity: Mapping[str, Any] | None = None):
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files if name != "metadata_json"}
        metadata = json.loads(str(np.asarray(source["metadata_json"]).item()))
    if metadata.get("schema") != ACTION_SCHEMA:
        raise ValueError("unsupported global action cache")
    if expected_identity is not None and metadata.get("identity") != _json_value(expected_identity):
        raise ValueError("global action cache identity mismatch")
    recorded = metadata.get("array_sha256")
    if recorded != _array_hash(*(arrays[name] for name in sorted(arrays))):
        raise ValueError("global action cache array hash mismatch")
    return arrays, metadata


def _build_actions(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    data = _load_resolution(args.geometry, args.baseline, args.resolution)
    identity = _action_identity(args, data)
    if args.output.is_file():
        _arrays, metadata = _load_action_cache(args.output, identity)
        _progress("action_cache_hit", resolution=args.resolution, output=str(args.output.resolve()))
        return metadata
    artifact = data.artifact
    _, identity_artifact = base._load_artifact(args.geometry, 64)
    reference_cache = args.baseline / f"N{args.resolution}.reference.npz"
    raw_state, regular_raw, eta_raw, exact_raw, cache_provenance = global_case.p04._load_cache(
        reference_cache, args.resolution, float(args.time)
    )
    manifest_hash = _sha256(data.artifact_path / "manifest.json")
    if cache_provenance["geometry_manifest_sha256"] != manifest_hash:
        raise ValueError("midpoint field cache and geometry differ")
    reference = _reference(args.reference_sidecar, verify_hashes=True)
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
    environment = dict(base._FROZEN_RUNTIME_ENV)
    if args.jax_cache is not None:
        args.jax_cache.mkdir(parents=True, exist_ok=True)
        environment["JAX_COMPILATION_CACHE_DIR"] = str(args.jax_cache.resolve())
    with patch.dict(os.environ, environment, clear=False):
        runtime = mms._runtime(artifact.global_geometry, artifact.owner_geometry, runtime_args)
    if runtime.model is None:
        raise RuntimeError("global action extraction requires one device")
    model = runtime.model
    runtime_seconds = time.perf_counter() - runtime_started
    _progress(
        "action_runtime_complete", resolution=args.resolution,
        seconds=runtime_seconds,
    )
    raw_states = {
        "actual_vorticity": raw_state,
        "smooth_regular_scalar": raw_state.replace(vorticity=regular_raw),
        "smooth_eta_varying_scalar": raw_state.replace(vorticity=eta_raw),
    }
    owner_states = {
        field: mms._owner_project(state, artifact.owner_geometry)
        for field, state in raw_states.items()
    }
    operands = {
        field: base._prepare_operands(model, owner_states[field], raw_states[field])
        for field in FIELDS
    }
    keys, _points, _eta = global_case.p04._candidate_face_geometry(
        model, operands["actual_vorticity"]
    )
    candidate_values, value_provenance = global_case._transported_values(
        args.p04_root, args.resolution, artifact, model, owner_states, keys
    )
    gradients, derivative_manifest = cubic.load_global_gradients(
        args.derivative_root / f"N{args.resolution}"
    )
    candidate_gradient = {
        field: global_case._gradient_stencil(gradients[field]) for field in cubic.FIELD_NAMES
    }
    arrays: dict[str, np.ndarray] = {
        "owner_volume": np.asarray(artifact.owner_geometry.aggregate_chart_volume, dtype=np.float64),
        "active": np.asarray(artifact.owner_geometry.topology.is_active_owner, dtype=np.uint8),
    }
    structural: dict[str, Any] = {}
    for field in FIELDS:
        field_started = time.perf_counter()
        runner = global_case._runner(model, operands[field])
        structural[field] = {}
        for rule in RULES:
            field_faces, phi_faces, left, right = global_case._rule_arguments(
                operands[field], keys, candidate_values, rule, field
            )
            action_values = runner(
                field_faces,
                phi_faces,
                candidate_gradient[field],
                candidate_gradient["phi"],
                left,
                right,
            )
            for action in ACTIONS:
                arrays[f"action:{field}:{rule}:{action}"] = np.asarray(
                    action_values[action], dtype=np.float64
                )
            structural[field][rule] = {
                "centered_decomposition_max_abs": float(
                    np.max(np.abs(action_values["C_direct"] - action_values["C"]))
                ),
                "swapped_antisymmetry_max_abs": float(
                    np.max(np.abs(action_values["C_direct"] + action_values["swapped"]))
                ),
            }
        arrays[f"midpoint_reference:{field}"] = mms._owner_project_array(
            exact_raw[field], artifact.owner_geometry
        )
        arrays[f"midpoint_input:{field}"] = np.asarray(
            owner_states[field].vorticity, dtype=np.float64
        )
        _progress(
            "action_field_complete", resolution=args.resolution, field=field,
            seconds=time.perf_counter() - field_started,
        )
    metadata = {
        "schema": ACTION_SCHEMA,
        "identity": identity,
        "structural": structural,
        "transported_value_provenance": value_provenance,
        "derivative_statistics": derivative_manifest["statistics"],
        "timings_seconds": {
            "runtime_setup": runtime_seconds,
            "total_before_write": time.perf_counter() - started,
        },
        "maximum_rss_gib": _max_rss_gib(),
        "scope": {"production_changes": [], "reconstruction_changes": []},
    }
    metadata["array_sha256"] = _array_hash(*(arrays[name] for name in sorted(arrays)))
    arrays["metadata_json"] = np.asarray(json.dumps(_json_value(metadata), sort_keys=True))
    _write_npz(args.output, arrays)
    _progress(
        "action_cache_complete", resolution=args.resolution,
        output=str(args.output.resolve()), seconds=time.perf_counter() - started,
    )
    return metadata


def _cross_supplement(
    data: ResolutionData, cross_root: Path | None
) -> tuple[np.ndarray, dict[int, list[str]], dict[str, Any] | None]:
    if cross_root is None or data.resolution not in (48, 64):
        return np.empty(0, dtype=np.int64), {}, None
    path = cross_root / f"N{data.resolution}.cross_cache.npz"
    arrays, metadata = cross._load_npz(path, validate_sources=True)
    lookup = {tuple(key): index for index, key in enumerate(data.owner_keys.tolist())}
    owners = np.asarray([lookup[tuple(key)] for key in arrays["owner_keys"]], dtype=np.int64)
    labels = {int(owner): ["prior_value_derivative_cross"] for owner in owners}
    return owners, labels, {
        "cache": _source_identity(path),
        "schema": metadata["schema"],
        "owner_count": len(owners),
        "owner_key_hash": _array_hash(arrays["owner_keys"]),
    }


def _representative_vorticity_owners(
    strata: Mapping[str, Mapping[str, Any]], count: int = 8
) -> np.ndarray:
    selected: list[int] = []
    order = (
        "axis_ring",
        "agglomerated_interior",
        "ordinary",
        "true_size_change_interface",
        "physical_boundary_footprint",
    )
    slot = 0
    while len(selected) < count:
        region = order[slot % len(order)]
        draws = np.asarray(strata[region]["draws"], dtype=np.int64)
        candidate = int(draws[(slot // len(order)) % len(draws)])
        if candidate not in selected:
            selected.append(candidate)
        slot += 1
    return np.asarray(sorted(selected), dtype=np.int64)


def _compact_action(
    values: np.ndarray, data: ResolutionData, owners: np.ndarray
) -> np.ndarray:
    compact = np.asarray(values, dtype=np.float64).reshape(-1)[data.arrays["owner_flat_ids"]]
    return compact[np.asarray(owners, dtype=np.int64)]


def _qualify(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    data = _load_resolution(args.geometry, args.baseline, args.resolution)
    actions, action_metadata = _load_action_cache(args.action_cache)
    action_geometry = action_metadata["identity"]["geometry"]["payloads"]
    current_geometry = _geometry_identity(args.geometry, args.resolution)
    if action_geometry != current_geometry["payloads"]:
        raise ValueError("action cache and qualification geometry differ")
    owners, strata, supplemental = _choose_sample(data, args.seed)
    cross_owners, cross_labels, cross_identity = _cross_supplement(data, args.cross_root)
    for owner, labels in cross_labels.items():
        supplemental.setdefault(owner, []).extend(labels)
    owners = np.unique(np.concatenate((owners, cross_owners, np.asarray(list(supplemental), dtype=np.int64))))
    representative = _representative_vorticity_owners(strata)
    reference = _reference(args.reference_sidecar, verify_hashes=True)
    prior_metadata: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    if args.output_cache.is_file():
        with np.load(args.output_cache, allow_pickle=False) as source:
            prior_metadata = json.loads(str(np.asarray(source["metadata_json"]).item()))
            arrays = {
                name: np.asarray(source[name])
                for name in source.files if name != "metadata_json"
            }
        if not np.array_equal(arrays.get("owners"), owners):
            raise ValueError("existing qualification cache selection changed")
        if not np.array_equal(arrays.get("owner_keys"), data.owner_keys[owners]):
            raise ValueError("existing qualification cache owner keys changed")
        _progress("qualification_cache_reuse", resolution=args.resolution, arrays=len(arrays))
    arrays.update({
        "owners": owners,
        "owner_keys": data.owner_keys[owners],
        "owner_volume": data.owner_volume[owners],
        "vorticity_representative_owners": representative,
    })
    evaluations: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]] = {}
    for order in (1, 3, 5, 7):
        key = f"smooth_q{order}"
        if all(f"reference:{field}:q{order}" in arrays for field in FIELDS[1:]):
            continue
        evaluations[key] = _evaluate_reference(
            data, reference, order=order, owners=owners, fields=FIELDS[1:],
            vorticity_method="ibp", time_value=args.time, chunk_cells=args.chunk_cells,
        )
        for field in FIELDS[1:]:
            arrays[f"reference:{field}:q{order}"] = evaluations[key][0][field]
            arrays[f"reference_true_volume:{field}:q{order}"] = evaluations[key][1][field]
            arrays[f"integrated_input:{field}:q{order}"] = evaluations[key][2][field]
    for order in (2, 3, 4):
        key = f"vorticity_ibp_q{order}"
        if f"reference:{FIELDS[0]}:ibp_q{order}" in arrays:
            continue
        evaluations[key] = _evaluate_reference(
            data, reference, order=order, owners=owners, fields=(FIELDS[0],),
            vorticity_method="ibp", time_value=args.time, chunk_cells=args.chunk_cells,
        )
        arrays[f"reference:{FIELDS[0]}:ibp_q{order}"] = evaluations[key][0][FIELDS[0]]
        arrays[f"reference_true_volume:{FIELDS[0]}:ibp_q{order}"] = evaluations[key][1][FIELDS[0]]
        arrays[f"integrated_input:{FIELDS[0]}:q{order}"] = evaluations[key][2][FIELDS[0]]
    omega_evaluator, omega_evaluator_identity = _structured_omega(data, args.baseline)
    prior_evaluator = (
        prior_metadata.get("fields", {})
        .get(FIELDS[0], {})
        .get("structured_omega_evaluator")
    )
    if prior_evaluator != omega_evaluator_identity:
        for name in tuple(arrays):
            if "structured_q3" in name:
                arrays.pop(name)
    structured_runs = {}
    for step_label, step in (("half", 1.0e-4), ("base", 2.0e-4), ("double", 4.0e-4)):
        key = f"vorticity_structured_q3_{step_label}"
        if f"reference:{FIELDS[0]}:structured_q3:{step_label}" in arrays:
            continue
        structured_runs[step_label] = _evaluate_reference(
            data, reference, order=3, owners=owners, fields=(FIELDS[0],),
            vorticity_method="structured", time_value=args.time,
            chunk_cells=args.chunk_cells, omega_evaluator=omega_evaluator,
            omega_gradient_step=step,
        )
        arrays[f"reference:{FIELDS[0]}:structured_q3:{step_label}"] = structured_runs[step_label][0][FIELDS[0]]
        arrays[f"integrated_input:{FIELDS[0]}:structured_q3"] = structured_runs[step_label][2][FIELDS[0]]
    direct_runs = {}
    for step_label, step in (("half", 1.0e-4), ("base", 2.0e-4), ("double", 4.0e-4)):
        if f"vorticity_direct_q3:{step_label}" in arrays:
            continue
        step_reference = _reference(
            args.reference_sidecar, finite_difference_step=step, verify_hashes=True
        )
        direct_runs[step_label] = _evaluate_reference(
            data, step_reference, order=3, owners=representative,
            fields=(FIELDS[0],), vorticity_method="direct", time_value=args.time,
            chunk_cells=args.chunk_cells,
        )
        arrays[f"vorticity_direct_q3:{step_label}"] = direct_runs[step_label][0][FIELDS[0]]
    representative_positions = np.searchsorted(owners, representative)
    arrays["vorticity_ibp_q3_representative"] = arrays[
        f"reference:{FIELDS[0]}:ibp_q3"
    ][representative_positions]
    field_payload: dict[str, Any] = {}
    smooth_pass = True
    for field in FIELDS[1:]:
        fine = arrays[f"reference:{field}:q7"]
        high = arrays[f"reference:{field}:q5"]
        numerical = _compact_action(actions[f"action:{field}:O:C"], data, owners)
        spatial = numerical - fine
        spatial_rms = _sample_global_rms(spatial, owners, strata)
        high_uncertainty = _sample_global_rms(fine - high, owners, strata)
        rules = {}
        for order in (1, 3, 5):
            candidate = arrays[f"reference:{field}:q{order}"]
            difference = _sample_global_rms(candidate - fine, owners, strata)
            ratio = (difference + high_uncertainty) / max(spatial_rms, 1.0e-300)
            rules[str(order)] = {
                "reference_difference_rms": difference,
                "high_reference_q7_minus_q5_rms": high_uncertainty,
                "uncertainty_budget_ratio": ratio,
                "passes_10_percent_budget": bool(ratio < 0.1),
            }
        selected_order = next((order for order in (3, 5) if rules[str(order)]["passes_10_percent_budget"]), None)
        if selected_order is None:
            smooth_pass = False
        midpoint_input = _compact_action(actions[f"midpoint_input:{field}"], data, owners)
        integrated_input = arrays[f"integrated_input:{field}:q7"]
        field_payload[field] = {
            "estimated_global_spatial_rms": spatial_rms,
            "estimated_global_midpoint_reference_shift_rms": _sample_global_rms(
                fine - _compact_action(actions[f"midpoint_reference:{field}"], data, owners), owners, strata
            ),
            "estimated_global_input_midpoint_minus_integrated_rms": _sample_global_rms(
                midpoint_input - integrated_input, owners, strata
            ),
            "frozen_minus_true_volume_reference_rms": _sample_global_rms(
                fine - arrays[f"reference_true_volume:{field}:q7"], owners, strata
            ),
            "rules": rules,
            "selected_order": selected_order,
        }
    vorticity_fine = arrays[f"reference:{FIELDS[0]}:ibp_q4"]
    vorticity_q3 = arrays[f"reference:{FIELDS[0]}:ibp_q3"]
    numerical = _compact_action(actions[f"action:{FIELDS[0]}:O:C"], data, owners)
    spatial_rms = _sample_global_rms(numerical - vorticity_fine, owners, strata)
    quadrature_uncertainty = _sample_global_rms(vorticity_fine - vorticity_q3, owners, strata)
    direct_base = arrays["vorticity_direct_q3:base"]
    ibp_subset = arrays["vorticity_ibp_q3_representative"]
    volume_subset = data.owner_volume[representative]
    equivalence_uncertainty = _weighted_rms(direct_base - ibp_subset, volume_subset)
    derivative_uncertainty = max(
        _weighted_rms(arrays["vorticity_direct_q3:half"] - direct_base, volume_subset),
        _weighted_rms(arrays["vorticity_direct_q3:double"] - direct_base, volume_subset),
    )
    structured_base = arrays[f"reference:{FIELDS[0]}:structured_q3:base"]
    structured_difference = _sample_global_rms(
        structured_base - vorticity_fine, owners, strata
    )
    structured_step_sensitivity = max(
        _sample_global_rms(
            arrays[f"reference:{FIELDS[0]}:structured_q3:half"] - structured_base,
            owners,
            strata,
        ),
        _sample_global_rms(
            arrays[f"reference:{FIELDS[0]}:structured_q3:double"] - structured_base,
            owners,
            strata,
        ),
    )
    structured_subset = structured_base[representative_positions]
    structured_direct_uncertainty = _weighted_rms(
        structured_subset - direct_base, volume_subset
    )
    q4_ratio = (
        quadrature_uncertainty + equivalence_uncertainty + derivative_uncertainty
    ) / max(spatial_rms, 1.0e-300)
    q3_ratio = (
        2.0 * quadrature_uncertainty + equivalence_uncertainty + derivative_uncertainty
    ) / max(spatial_rms, 1.0e-300)
    structured_ratio = (
        structured_difference
        + quadrature_uncertainty
        + structured_step_sensitivity
        + derivative_uncertainty
    ) / max(spatial_rms, 1.0e-300)
    vorticity_order = "structured_q3" if structured_ratio < 0.1 else None
    midpoint_input = _compact_action(actions[f"midpoint_input:{FIELDS[0]}"], data, owners)
    integrated_input = arrays[f"integrated_input:{FIELDS[0]}:q4"]
    field_payload[FIELDS[0]] = {
        "estimated_global_spatial_rms": spatial_rms,
        "quadrature_q4_minus_q3_rms": quadrature_uncertainty,
        "direct_q3_minus_ibp_q3_representative_rms": equivalence_uncertainty,
        "finite_difference_step_sensitivity_rms": derivative_uncertainty,
        "structured_q3_minus_ibp_q4_rms": structured_difference,
        "structured_q3_step_sensitivity_rms": structured_step_sensitivity,
        "structured_q3_minus_direct_q3_representative_rms": structured_direct_uncertainty,
        "structured_omega_evaluator": omega_evaluator_identity,
        "rules": {
            "3": {"uncertainty_budget_ratio": q3_ratio, "passes_10_percent_budget": bool(q3_ratio < 0.1)},
            "4": {"uncertainty_budget_ratio": q4_ratio, "passes_10_percent_budget": bool(q4_ratio < 0.1)},
            "structured_q3": {
                "uncertainty_budget_ratio": structured_ratio,
                "passes_10_percent_budget": bool(structured_ratio < 0.1),
            },
        },
        "selected_order": vorticity_order,
        "estimated_global_midpoint_reference_shift_rms": _sample_global_rms(
            vorticity_fine - _compact_action(actions[f"midpoint_reference:{FIELDS[0]}"], data, owners), owners, strata
        ),
        "estimated_global_input_midpoint_minus_integrated_rms": _sample_global_rms(
            midpoint_input - integrated_input, owners, strata
        ),
        "frozen_minus_true_volume_reference_rms": _sample_global_rms(
            vorticity_fine - arrays[f"reference_true_volume:{FIELDS[0]}:ibp_q4"], owners, strata
        ),
    }
    supervisor_replay = None
    if args.supervisor_root is not None and args.resolution in (48, 64):
        supervisor_npz = args.supervisor_root / f"N{args.resolution}.npz"
        supervisor_json = args.supervisor_root / f"N{args.resolution}.json"
        supervisor_case = json.loads(supervisor_json.read_text(encoding="utf-8"))
        if supervisor_case["cache_sha256"] != cross_identity["cache"]["sha256"]:
            raise ValueError("supervisor and current cross cache identities differ")
        positions = np.searchsorted(owners, cross_owners)
        field_slots = {FIELDS[1]: 0, FIELDS[2]: 1}
        maximum = 0.0
        with np.load(supervisor_npz, allow_pickle=False) as saved:
            for order in (1, 3, 5, 7):
                for field, slot in field_slots.items():
                    maximum = max(
                        maximum,
                        float(np.max(np.abs(
                            arrays[f"reference:{field}:q{order}"][positions]
                            - np.asarray(saved[f"reference_q{order}"])[slot]
                        ))),
                    )
        supervisor_replay = {
            "npz": _source_identity(supervisor_npz),
            "json": _source_identity(supervisor_json),
            "maximum_reference_replay_abs": maximum,
            "source_validation_reenabled": True,
        }
    metadata = {
        "schema": QUALIFICATION_SCHEMA,
        "status": "complete",
        "resolution": args.resolution,
        "geometry": current_geometry,
        "reference_sidecar": _source_identity(args.reference_sidecar),
        "action_cache": _source_identity(args.action_cache),
        "cross_supplement": cross_identity,
        "supervisor_replay": supervisor_replay,
        "sampling": {
            "seed": args.seed + args.resolution,
            "strata": strata,
            "supplemental": supplemental,
            "unique_owner_count": len(owners),
            "vorticity_representative_owners": representative,
        },
        "fields": field_payload,
        "selected_policy": {
            field: field_payload[field]["selected_order"] for field in FIELDS
        },
        "passes": bool(smooth_pass and vorticity_order is not None),
        "timings": {
            name: value for name, value in prior_metadata.get("timings", {}).items()
            if name != "total"
        } | {name: value[3] for name, value in evaluations.items()} | {
            f"vorticity_direct_q3_{name}": value[3] for name, value in direct_runs.items()
        } | {
            f"vorticity_structured_q3_{name}": value[3] for name, value in structured_runs.items()
        } | {"total": time.perf_counter() - started},
        "functional": {
            "continuum": "-bcov dot (grad(phi) cross grad(f))/(rho_star*abs(J)*B)",
            "normalization_primary": "frozen physical owner volume",
            "vorticity_ibp": "complete owner face integral omega*a.n minus owner volume integral omega*div(a)",
            "a": "-(bcov cross grad(phi))/(rho_star*B)",
        },
        "scope": {
            "bounded_reference_qualification": True,
            "global_operator_orders": False,
            "production_changes": [],
        },
    }
    metadata["array_sha256"] = _array_hash(*(arrays[name] for name in sorted(arrays)))
    arrays["metadata_json"] = np.asarray(json.dumps(_json_value(metadata), sort_keys=True))
    _write_npz(args.output_cache, arrays)
    _write_json(args.output_json, metadata)
    _progress(
        "qualification_complete", resolution=args.resolution,
        passes=metadata["passes"], selected_policy=metadata["selected_policy"],
        seconds=time.perf_counter() - started,
    )
    return metadata


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    data = _load_resolution(args.geometry, args.baseline, args.resolution)
    identity = _geometry_identity(args.geometry, args.resolution)
    selected, strata, supplemental = _choose_sample(data, args.seed)
    # Two actual HSX owners exercise ordinary and agglomerated paths cheaply.
    tiny = []
    for region in ("ordinary", "agglomerated_interior"):
        tiny.append(int(np.asarray(strata[region]["draws"])[0]))
    owners = np.unique(np.asarray(tiny, dtype=np.int64))
    reference = _reference(args.reference_sidecar, verify_hashes=True)
    smooth_q1 = _evaluate_reference(
        data, reference, order=1, owners=owners,
        fields=FIELDS[1:], vorticity_method="ibp", time_value=args.time,
        chunk_cells=args.chunk_cells,
    )
    smooth_q3 = _evaluate_reference(
        data, reference, order=3, owners=owners,
        fields=FIELDS[1:], vorticity_method="ibp", time_value=args.time,
        chunk_cells=args.chunk_cells,
    )
    vorticity_direct_q1 = _evaluate_reference(
        data, reference, order=1, owners=owners,
        fields=(FIELDS[0],), vorticity_method="direct", time_value=args.time,
        chunk_cells=args.chunk_cells,
    )
    vorticity_ibp_q1 = _evaluate_reference(
        data, reference, order=1, owners=owners,
        fields=(FIELDS[0],), vorticity_method="ibp", time_value=args.time,
        chunk_cells=args.chunk_cells,
    )
    vorticity_direct_q2 = _evaluate_reference(
        data, reference, order=2, owners=owners,
        fields=(FIELDS[0],), vorticity_method="direct", time_value=args.time,
        chunk_cells=args.chunk_cells,
    )
    vorticity_ibp_q2 = _evaluate_reference(
        data, reference, order=2, owners=owners,
        fields=(FIELDS[0],), vorticity_method="ibp", time_value=args.time,
        chunk_cells=args.chunk_cells,
    )
    vorticity_ibp_q3 = _evaluate_reference(
        data, reference, order=3, owners=owners,
        fields=(FIELDS[0],), vorticity_method="ibp", time_value=args.time,
        chunk_cells=args.chunk_cells,
    )
    direct_q1 = vorticity_direct_q1[0][FIELDS[0]]
    ibp_q1 = vorticity_ibp_q1[0][FIELDS[0]]
    direct_q2 = vorticity_direct_q2[0][FIELDS[0]]
    ibp_q2 = vorticity_ibp_q2[0][FIELDS[0]]
    ibp_q3 = vorticity_ibp_q3[0][FIELDS[0]]
    payload = {
        "schema": QUALIFICATION_SCHEMA,
        "stage": "preflight",
        "resolution": args.resolution,
        "owners": owners,
        "owner_keys": data.owner_keys[owners],
        "geometry_equivalence": identity,
        "sidecar": _source_identity(args.reference_sidecar),
        "smooth_q3_minus_q1": {
            field: smooth_q3[0][field] - smooth_q1[0][field] for field in FIELDS[1:]
        },
        "vorticity": {
            "direct_q1": direct_q1,
            "ibp_q1": ibp_q1,
            "ibp_q1_minus_direct_q1": ibp_q1 - direct_q1,
            "direct_q2": direct_q2,
            "ibp_q2": ibp_q2,
            "ibp_q2_minus_direct_q2": ibp_q2 - direct_q2,
            "ibp_q3": ibp_q3,
            "ibp_q3_minus_q2": ibp_q3 - ibp_q2,
            "ibp_direct_q1_max_abs": float(np.max(np.abs(ibp_q1 - direct_q1))),
            "ibp_direct_q2_max_abs": float(np.max(np.abs(ibp_q2 - direct_q2))),
        },
        "timings": {
            "smooth_q1": smooth_q1[3],
            "smooth_q3": smooth_q3[3],
            "vorticity_direct_q1": vorticity_direct_q1[3],
            "vorticity_ibp_q1": vorticity_ibp_q1[3],
            "vorticity_direct_q2": vorticity_direct_q2[3],
            "vorticity_ibp_q2": vorticity_ibp_q2[3],
            "vorticity_ibp_q3": vorticity_ibp_q3[3],
            "total": time.perf_counter() - started,
        },
        "selection_preview": {
            "strata": strata,
            "supplemental": supplemental,
        },
        "scope": {
            "tiny_actual_hsx_preflight": True,
            "production_changes": [],
            "global_orders": False,
        },
    }
    _write_json(args.output, payload)
    _progress(
        "preflight_complete",
        output=str(args.output.resolve()),
        seconds=payload["timings"]["total"],
        ibp_direct_q2_max_abs=payload["vorticity"]["ibp_direct_q2_max_abs"],
    )
    return payload


def _profile_reference(args: argparse.Namespace) -> dict[str, Any]:
    data = _load_resolution(args.geometry, args.baseline, args.resolution)
    with np.load(args.qualification_cache, allow_pickle=False) as source:
        owners = np.asarray(source["owners"], dtype=np.int64)
    reference = _reference(args.reference_sidecar, verify_hashes=True)
    started = time.perf_counter()
    evaluated = _evaluate_reference(
        data, reference, order=args.order, owners=owners,
        fields=(FIELDS[0],), vorticity_method="ibp", time_value=args.time,
        chunk_cells=args.chunk_cells,
    )
    measured = evaluated[3]
    per_cell = measured["seconds"] / measured["raw_cell_count"]
    payload = {
        "schema": QUALIFICATION_SCHEMA,
        "stage": "vorticity_ibp_profile",
        "resolution": args.resolution,
        "order": args.order,
        "chunk_cells": args.chunk_cells,
        "measured": measured,
        "full_domain_extrapolated_seconds": per_cell * args.resolution**3,
        "full_domain_extrapolated_hours": per_cell * args.resolution**3 / 3600.0,
        "total_seconds": time.perf_counter() - started,
    }
    _write_json(args.output, payload)
    _progress(
        "profile_complete",
        resolution=args.resolution,
        order=args.order,
        extrapolated_hours=payload["full_domain_extrapolated_hours"],
    )
    return payload


def _global_reference(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    data = _load_resolution(args.geometry, args.baseline, args.resolution)
    qualification = json.loads(args.qualification_json.read_text(encoding="utf-8"))
    for field in FIELDS[1:]:
        if qualification["fields"][field]["selected_order"] != 3:
            raise ValueError(f"q3 is not qualified for {field}")
    reference = _reference(args.reference_sidecar, verify_hashes=True)
    frozen, true_volume, inputs, timing = _evaluate_reference(
        data, reference, order=3, owners=None, fields=FIELDS[1:],
        vorticity_method="ibp", time_value=args.time, chunk_cells=args.chunk_cells,
    )
    # The evaluator's timing schema includes the vorticity dispatch argument
    # even when no vorticity field is requested.
    timing["vorticity_method"] = "not evaluated"
    timing["reference_method"] = "direct tensor Gauss volume q3"
    arrays: dict[str, np.ndarray] = {
        "owner_flat_ids": data.arrays["owner_flat_ids"],
        "owner_keys": data.owner_keys,
        "owner_volume": data.owner_volume,
    }
    for field in FIELDS[1:]:
        arrays[f"reference:{field}"] = frozen[field]
        arrays[f"reference_true_volume:{field}"] = true_volume[field]
        arrays[f"integrated_input:{field}"] = inputs[field]
    metadata = {
        "schema": REFERENCE_SCHEMA,
        "status": "complete",
        "resolution": args.resolution,
        "geometry": _geometry_identity(args.geometry, args.resolution),
        "reference_sidecar": _source_identity(args.reference_sidecar),
        "qualification": _source_identity(args.qualification_json),
        "policy": {
            "smooth_fields": "direct tensor Gauss volume q3",
            "normalization": "frozen physical owner volume",
            "true_volume_sensitivity_retained": True,
            "vorticity": "not built globally; measured direct/IBP q3 extrapolation is prohibitive",
        },
        "qualification_metrics": {
            field: qualification["fields"][field]["rules"]["3"] for field in FIELDS[1:]
        },
        "timing": timing,
        "total_seconds": time.perf_counter() - started,
        "scope": {"production_changes": [], "global_vorticity_reference": False},
    }
    metadata["array_sha256"] = _array_hash(*(arrays[name] for name in sorted(arrays)))
    arrays["metadata_json"] = np.asarray(json.dumps(_json_value(metadata), sort_keys=True))
    _write_npz(args.output, arrays)
    _progress(
        "global_reference_complete", resolution=args.resolution,
        seconds=time.perf_counter() - started, output=str(args.output.resolve()),
    )
    return metadata


def _load_reference_cache(path: Path):
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files if name != "metadata_json"}
        metadata = json.loads(str(np.asarray(source["metadata_json"]).item()))
    if metadata.get("schema") != REFERENCE_SCHEMA:
        raise ValueError("unsupported integrated reference cache")
    if metadata.get("array_sha256") != _array_hash(*(arrays[name] for name in sorted(arrays))):
        raise ValueError("integrated reference cache array hash mismatch")
    return arrays, metadata


def _compact_statistics(
    actual: np.ndarray,
    exact: np.ndarray,
    data: ResolutionData,
) -> dict[str, Any]:
    volume = data.owner_volume
    difference = np.asarray(actual) - np.asarray(exact)
    squared = volume * difference**2
    exact_squared = volume * np.asarray(exact)**2
    total_squared = float(np.sum(squared))
    result = {
        "absolute_l2": math.sqrt(total_squared / float(np.sum(volume))),
        "relative_l2": math.sqrt(total_squared / max(float(np.sum(exact_squared)), 1.0e-300)),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "signed_volume_weighted_mean_error": float(np.sum(volume * difference) / np.sum(volume)),
        "owner_count": len(volume),
        "regions": {},
    }
    for region, mask in data.masks.items():
        region_squared = float(np.sum(squared[mask]))
        result["regions"][region] = {
            "absolute_l2": math.sqrt(region_squared / float(np.sum(volume[mask]))),
            "maximum_absolute_error": float(np.max(np.abs(difference[mask]))),
            "squared_error_fraction": region_squared / max(total_squared, 1.0e-300),
            "volume_fraction": float(np.sum(volume[mask]) / np.sum(volume)),
            "owner_count": int(np.count_nonzero(mask)),
        }
    return result


def _bounded_integrated_replay(
    resolution: int,
    cross_root: Path,
    qualification_cache: Path,
) -> dict[str, Any] | None:
    if resolution not in (48, 64):
        return None
    cross_path = cross_root / f"N{resolution}.cross_cache.npz"
    saved, metadata = cross._load_npz(cross_path, validate_sources=True)
    with np.load(qualification_cache, allow_pickle=False) as source:
        qualified_owners = np.asarray(source["owners"], dtype=np.int64)
        qualified_keys = np.asarray(source["owner_keys"], dtype=np.int64)
        qualified = {
            field: np.asarray(source[f"reference:{field}:q3"], dtype=np.float64)
            for field in FIELDS[1:]
        }
    key_to_position = {tuple(key): index for index, key in enumerate(qualified_keys.tolist())}
    positions = np.asarray([key_to_position[tuple(key)] for key in saved["owner_keys"]], dtype=np.int64)
    face_axis = saved["face_keys"][:, 0]
    collapsed = saved["collapsed"].astype(bool)
    generators = {}
    for field_index, field in enumerate(cross.FIELD_NAMES):
        generators[field] = {}
        for derivative_label, array_label in (("Dh", "gradient_dh"), ("De", "gradient_de")):
            value = cross.factor._normal_cross(
                face_axis, saved["one_form"], saved[array_label][field_index]
            ) * saved["face_measure"]
            value[collapsed] = 0.0
            generators[field][derivative_label] = value
    values = {
        "P": saved["values_P"],
        "O2": saved["values_O2"],
        "C3_planar": saved["values_C3_planar"],
        "C3_full": saved["values_C3_full"],
        "Ve_planar": saved["values_Ve_planar"],
    }
    cross_values = {"Vh": saved["values_O2"], "Ve": saved["values_Ve_planar"]}
    result: dict[str, Any] = {
        "cross_cache": _source_identity(cross_path),
        "qualification_cache": _source_identity(qualification_cache),
        "fields": {},
        "maximum_signed_vector_closure": 0.0,
    }
    rho_star = float(metadata["rho_star"])
    for transport_index, field in enumerate(cross.TRANSPORT_FIELDS):
        if field == FIELDS[0]:
            continue
        scalar_index = transport_index + 1
        integrated = qualified[field][positions]
        midpoint = saved["targets"][transport_index]
        field_result = {"cross": {}, "candidates_with_Dh": {}}
        for label in cross.CROSS_LABELS:
            value_choice = "Ve" if label.startswith("Ve") else "Vh"
            derivative_choice = "De" if label.endswith("De") else "Dh"
            chosen = cross_values[value_choice]
            action_a = cross._assemble(
                generators["phi"][derivative_choice], chosen[scalar_index],
                saved["center"][scalar_index], saved, sign=-1.0, rho_star=rho_star,
            )
            action_b = cross._assemble(
                generators[field][derivative_choice], chosen[0], saved["center"][0],
                saved, sign=1.0, rho_star=rho_star,
            )
            field_result["cross"][label] = {}
            for action, vector in (("A", action_a), ("B", action_b), ("C", 0.5 * (action_a + action_b))):
                field_result["cross"][label][action] = cross._statistics(
                    vector, integrated, saved["owner_volume"], saved["owner_region"],
                    saved["owner_expanded"].astype(bool),
                )
                closure = (vector - integrated) - ((vector - midpoint) - (integrated - midpoint))
                result["maximum_signed_vector_closure"] = max(
                    result["maximum_signed_vector_closure"], float(np.max(np.abs(closure)))
                )
        for label, chosen in values.items():
            action_a = cross._assemble(
                generators["phi"]["Dh"], chosen[scalar_index], saved["center"][scalar_index],
                saved, sign=-1.0, rho_star=rho_star,
            )
            action_b = cross._assemble(
                generators[field]["Dh"], chosen[0], saved["center"][0],
                saved, sign=1.0, rho_star=rho_star,
            )
            field_result["candidates_with_Dh"][label] = {
                action: cross._statistics(
                    vector, integrated, saved["owner_volume"], saved["owner_region"],
                    saved["owner_expanded"].astype(bool),
                )
                for action, vector in (
                    ("A", action_a), ("B", action_b), ("C", 0.5 * (action_a + action_b))
                )
            }
        field_result["integrated_minus_midpoint"] = cross._effect_statistics(
            integrated - midpoint, saved["owner_volume"]
        )
        result["fields"][field] = field_result
    return result


def _replay(args: argparse.Namespace) -> dict[str, Any]:
    data = _load_resolution(args.geometry, args.baseline, args.resolution)
    actions, action_metadata = _load_action_cache(args.action_cache)
    reference, reference_metadata = _load_reference_cache(args.reference_cache)
    if not np.array_equal(reference["owner_flat_ids"], data.arrays["owner_flat_ids"]):
        raise ValueError("reference and action owner membership differ")
    fields = {}
    maximum_closure = 0.0
    for field in FIELDS[1:]:
        exact = reference[f"reference:{field}"]
        midpoint = _compact_action(actions[f"midpoint_reference:{field}"], data, np.arange(len(data.owner_volume)))
        field_payload = {"actions": {}, "reference": {}}
        field_payload["reference"] = {
            "integrated_minus_midpoint": _compact_statistics(exact, midpoint, data),
            "frozen_minus_true_volume": _compact_statistics(
                exact, reference[f"reference_true_volume:{field}"], data
            ),
            "input_midpoint_minus_integrated": _compact_statistics(
                _compact_action(actions[f"midpoint_input:{field}"], data, np.arange(len(data.owner_volume))),
                reference[f"integrated_input:{field}"], data,
            ),
        }
        for rule in RULES:
            field_payload["actions"][rule] = {}
            for action in ACTIONS:
                numerical = _compact_action(
                    actions[f"action:{field}:{rule}:{action}"], data,
                    np.arange(len(data.owner_volume)),
                )
                field_payload["actions"][rule][action] = {
                    "integrated": _compact_statistics(numerical, exact, data),
                    "midpoint": _compact_statistics(numerical, midpoint, data),
                }
                closure = (numerical - exact) - ((numerical - midpoint) - (exact - midpoint))
                maximum_closure = max(maximum_closure, float(np.max(np.abs(closure))))
        fields[field] = field_payload
    payload = {
        "schema": CASE_SCHEMA,
        "resolution": args.resolution,
        "fields": fields,
        "action_cache": _source_identity(args.action_cache),
        "reference_cache": _source_identity(args.reference_cache),
        "action_identity": action_metadata["identity"],
        "action_run": {
            "timings_seconds": action_metadata["timings_seconds"],
            "maximum_rss_gib": action_metadata["maximum_rss_gib"],
            "structural": action_metadata["structural"],
        },
        "reference_identity": reference_metadata,
        "bounded_value_derivative_replay": _bounded_integrated_replay(
            args.resolution, args.cross_root, args.qualification_cache
        ) if args.cross_root is not None and args.qualification_cache is not None else None,
        "verification": {"maximum_signed_vector_closure": maximum_closure},
        "scope": {
            "global_smooth_fields": True,
            "global_vorticity": False,
            "production_changes": [],
        },
    }
    _write_json(args.output, payload)
    _progress("replay_complete", resolution=args.resolution, output=str(args.output.resolve()))
    return payload


def _orders(errors: Sequence[float]) -> list[float]:
    resolutions = (32, 48, 64)
    return [
        math.log(errors[index] / errors[index + 1])
        / math.log(resolutions[index + 1] / resolutions[index])
        for index in range(2)
    ]


def _merge(args: argparse.Namespace) -> dict[str, Any]:
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in args.cases]
    cases.sort(key=lambda item: item["resolution"])
    if [case["resolution"] for case in cases] != [32, 48, 64]:
        raise ValueError("merge requires N32/N48/N64 cases")
    results: dict[str, Any] = {}
    all_pass = True
    for field in FIELDS[1:]:
        results[field] = {}
        for rule in RULES:
            results[field][rule] = {}
            for action in ACTIONS:
                errors = [
                    case["fields"][field]["actions"][rule][action]["integrated"]["absolute_l2"]
                    for case in cases
                ]
                orders = _orders(errors)
                passed = all(order >= 1.8 for order in orders)
                all_pass &= passed
                results[field][rule][action] = {
                    "absolute_l2": errors,
                    "orders": orders,
                    "passes_both_intervals": passed,
                }
    decision = (
        "Smooth-field global replay is complete, but full bracket certification remains blocked: "
        "the qualified direct/IBP vorticity reference extrapolates to an unreasonable full-domain cost "
        "and the affordable focused interpolant did not pass the coarse-grid reference budget."
    )
    qualification = {}
    for case in cases:
        path = Path(case["reference_identity"]["qualification"]["path"])
        item = json.loads(path.read_text(encoding="utf-8"))
        qualification[str(case["resolution"])] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "smooth_q3_budget_ratio": {
                field: item["fields"][field]["rules"]["3"]["uncertainty_budget_ratio"]
                for field in FIELDS[1:]
            },
            "vorticity_direct_ibp_q3_budget_ratio":
                item["fields"][FIELDS[0]]["rules"]["3"]["uncertainty_budget_ratio"],
            "vorticity_direct_ibp_q3_passes":
                item["fields"][FIELDS[0]]["rules"]["3"]["passes_10_percent_budget"],
            "bounded_vorticity_selected_order":
                3 if item["fields"][FIELDS[0]]["rules"]["3"]["passes_10_percent_budget"] else None,
        }
        structured = item["fields"][FIELDS[0]]["rules"].get("structured_q3")
        if structured is not None:
            qualification[str(case["resolution"])]["structured_q3"] = structured
    profile_path = args.report.parent / "N64.ibp_profile_1024.json"
    profile = None
    if profile_path.exists():
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile["path"] = str(profile_path.resolve())
        profile["sha256"] = _sha256(profile_path)
    payload = {
        "schema": SUMMARY_SCHEMA,
        "resolutions": [32, 48, 64],
        "smooth_results": results,
        "all_smooth_lanes_pass": all_pass,
        "global_vorticity_status": "bounded-qualified; global reference not built due measured cost",
        "reference_qualification": qualification,
        "vorticity_cost_profile": profile,
        "timings": {
            str(case["resolution"]): {
                "actions": case.get("action_run", {}).get("timings_seconds"),
                "smooth_reference_seconds": case["reference_identity"]["timing"]["seconds"],
                "smooth_reference_total_seconds": case["reference_identity"]["total_seconds"],
                "smooth_reference_method": "direct tensor Gauss volume q3",
            }
            for case in cases
        },
        "reference_sensitivity": {
            field: {
                str(case["resolution"]): case["fields"][field]["reference"]
                for case in cases
            }
            for field in FIELDS[1:]
        },
        "decision": decision,
        "case_paths": [str(path.resolve()) for path in args.cases],
        "verification_maximum": max(
            case["verification"]["maximum_signed_vector_closure"] for case in cases
        ),
        "scope": {
            "production_changes": [],
            "full_bracket_certified": False,
            "material_upwinding_certified": False,
        },
    }
    _write_json(args.output, payload)
    _write_report(args.report, payload, cases)
    _progress("merge_complete", output=str(args.output.resolve()), report=str(args.report.resolve()))
    return payload


def _write_report(path: Path, summary: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# HSX perpendicular-bracket integrated-reference requalification",
        "",
        "The geometry payloads used by the supervisor and P campaign are byte-identical resolved artifacts. Smooth controls use direct q3 physical-volume integration with frozen owner-volume normalization. Vorticity is bounded-qualified with the complete face-plus-volume integration-by-parts functional, but was not run globally after measured N64 throughput extrapolated to about 9.9 hours.",
        "",
        "## Reference qualification and scope",
        "",
        "| N | regular q3 budget | eta q3 budget | vorticity direct/IBP q3 budget | bounded vorticity q3 |",
        "|---:|---:|---:|---:|---|",
    ]
    for resolution in (32, 48, 64):
        item = summary["reference_qualification"][str(resolution)]
        bounded = "qualified" if item["bounded_vorticity_selected_order"] == 3 else "not qualified"
        lines.append(
            f"| {resolution} | {item['smooth_q3_budget_ratio'][FIELDS[1]]:.3%} | "
            f"{item['smooth_q3_budget_ratio'][FIELDS[2]]:.3%} | "
            f"{item['vorticity_direct_ibp_q3_budget_ratio']:.3%} | {bounded} |"
        )
    structured = summary["reference_qualification"]["32"].get("structured_q3")
    if structured is not None:
        lines.extend([
            "",
            f"The direct/IBP functional passes the 10% bounded budget at every resolution. "
            f"The focused structured candidate is rejected at N32 ({structured['uncertainty_budget_ratio']:.2%} of the numerical-error budget), so no affordable global vorticity reference is selected. This does not revoke the bounded direct/IBP result.",
        ])
    profile = summary.get("vorticity_cost_profile")
    if profile is not None:
        measured = profile["measured"]
        lines.extend([
            "",
            f"The N64 timing sample evaluated {measured['raw_cell_count']} raw cells in "
            f"{measured['seconds']:.2f} s with {measured['maximum_rss_gib']:.2f} GiB peak RSS; "
            f"the measured-rate full-domain extrapolation is {profile['full_domain_extrapolated_hours']:.2f} h.",
        ])
    lines.extend([
        "",
        "| N | cached numerical actions | smooth q3 reference |",
        "|---:|---:|---:|",
    ])
    for resolution in (32, 48, 64):
        timing = summary["timings"][str(resolution)]
        action_seconds = timing["actions"]["total_before_write"]
        lines.append(
            f"| {resolution} | {action_seconds:.2f} s | "
            f"{timing['smooth_reference_total_seconds']:.2f} s |"
        )
    lines.extend([
        "",
        "Reference shifts and normalization sensitivity are retained independently:",
        "",
        "| Field | N | integrated − midpoint L2 | frozen − true-volume L2 | input midpoint − integrated L2 |",
        "|---|---:|---:|---:|---:|",
    ])
    for field in FIELDS[1:]:
        for resolution in (32, 48, 64):
            item = summary["reference_sensitivity"][field][str(resolution)]
            lines.append(
                f"| {field} | {resolution} | "
                f"{item['integrated_minus_midpoint']['absolute_l2']:.7g} | "
                f"{item['frozen_minus_true_volume']['absolute_l2']:.7g} | "
                f"{item['input_midpoint_minus_integrated']['absolute_l2']:.7g} |"
            )
    lines.extend([
        "",
        "## Production O/centered-C: midpoint versus integrated reference",
        "",
        "| Field | Reference | absolute L2 N32 / N48 / N64 | orders | N64 dominant squared-error region |",
        "|---|---|---:|---:|---|",
    ])
    for field in FIELDS[1:]:
        integrated = [
            case["fields"][field]["actions"]["O"]["C"]["integrated"]["absolute_l2"]
            for case in cases
        ]
        midpoint = [
            case["fields"][field]["actions"]["O"]["C"]["midpoint"]["absolute_l2"]
            for case in cases
        ]
        regions = cases[-1]["fields"][field]["actions"]["O"]["C"]["integrated"]["regions"]
        region, region_item = max(regions.items(), key=lambda pair: pair[1]["squared_error_fraction"])
        for label, errors in (("integrated q3", integrated), ("historical midpoint", midpoint)):
            values = " / ".join(f"{value:.7g}" for value in errors)
            orders = " / ".join(f"{value:.4f}" for value in _orders(errors))
            dominant = (
                f"{region} ({region_item['squared_error_fraction']:.2%})"
                if label == "integrated q3" else "—"
            )
            lines.append(f"| {field} | {label} | {values} | {orders} | {dominant} |")
    lines.extend([
        "",
        "## Global smooth-field replay",
        "",
        "| Field | Rule | Action | absolute L2 N32 / N48 / N64 | orders | gate |",
        "|---|---|---|---:|---:|---|",
    ])
    for field in FIELDS[1:]:
        for rule in RULES:
            for action in ACTIONS:
                item = summary["smooth_results"][field][rule][action]
                errors = " / ".join(f"{value:.7g}" for value in item["absolute_l2"])
                orders = " / ".join(f"{value:.4f}" for value in item["orders"])
                gate = "pass" if item["passes_both_intervals"] else "fail"
                lines.append(f"| {field} | {rule} | {action} | {errors} | {orders} | {gate} |")
    lines.extend([
        "",
        "## Bounded corrected interpretation",
        "",
    ])
    for case in cases:
        bounded = case.get("bounded_value_derivative_replay")
        if bounded is None:
            continue
        for field in FIELDS[1:]:
            o2 = bounded["fields"][field]["candidates_with_Dh"]["O2"]["C"]["absolute_l2"]
            c3 = bounded["fields"][field]["candidates_with_Dh"]["C3_planar"]["C"]["absolute_l2"]
            lines.append(
                f"- N{case['resolution']} {field}: C3-planar/Dh changes completed C from `{o2:.7g}` to `{c3:.7g}` ({(c3/o2-1)*100:+.2f}%)."
            )
    lines.extend([
        "",
        "The historical midpoint comparisons remain preserved. Numerical-minus-integrated, integrated-minus-midpoint, and signed-vector closures are retained in the case files; scalar norms are not treated as additive.",
        "",
        "## Recommended bounded experiment",
        "",
        "Test one jointly compatible face-value/derivative functional with matched face and owner-center quadrature on the existing deterministic agglomerated-interior and true-interface samples. The integrated replay shows that changing the reference removes the earlier apparent smooth cubic-value regression, but completed O/C still fails the fine-order gate; another isolated polynomial-degree increase is therefore not the discriminating experiment.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "This campaign changes no reconstruction, production source, boundary model, evolved MMS, blob driver, curvature, diffusion, or polarization path. Material upwinding remains uncertified.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--resolution", type=int, choices=(32, 48, 64), default=32)
    preflight.add_argument("--geometry", type=Path, required=True)
    preflight.add_argument("--baseline", type=Path, required=True)
    preflight.add_argument("--reference-sidecar", type=Path, required=True)
    preflight.add_argument("--time", type=float, default=1.0e-6)
    preflight.add_argument("--seed", type=int, default=26092200)
    preflight.add_argument("--chunk-cells", type=int, default=64)
    preflight.add_argument("--output", type=Path, required=True)
    actions = sub.add_parser("actions")
    actions.add_argument("--resolution", type=int, choices=(32, 48, 64), required=True)
    actions.add_argument("--geometry", type=Path, required=True)
    actions.add_argument("--baseline", type=Path, required=True)
    actions.add_argument("--reference-sidecar", type=Path, required=True)
    actions.add_argument("--p04-root", type=Path, required=True)
    actions.add_argument("--derivative-root", type=Path, required=True)
    actions.add_argument("--time", type=float, default=1.0e-6)
    actions.add_argument("--jax-cache", type=Path)
    actions.add_argument("--output", type=Path, required=True)
    qualify = sub.add_parser("qualify")
    qualify.add_argument("--resolution", type=int, choices=(32, 48, 64), required=True)
    qualify.add_argument("--geometry", type=Path, required=True)
    qualify.add_argument("--baseline", type=Path, required=True)
    qualify.add_argument("--reference-sidecar", type=Path, required=True)
    qualify.add_argument("--action-cache", type=Path, required=True)
    qualify.add_argument("--cross-root", type=Path)
    qualify.add_argument("--supervisor-root", type=Path)
    qualify.add_argument("--time", type=float, default=1.0e-6)
    qualify.add_argument("--seed", type=int, default=26092200)
    qualify.add_argument("--chunk-cells", type=int, default=128)
    qualify.add_argument("--output-cache", type=Path, required=True)
    qualify.add_argument("--output-json", type=Path, required=True)
    profile = sub.add_parser("profile-reference")
    profile.add_argument("--resolution", type=int, choices=(32, 48, 64), required=True)
    profile.add_argument("--geometry", type=Path, required=True)
    profile.add_argument("--baseline", type=Path, required=True)
    profile.add_argument("--reference-sidecar", type=Path, required=True)
    profile.add_argument("--qualification-cache", type=Path, required=True)
    profile.add_argument("--order", type=int, default=3)
    profile.add_argument("--chunk-cells", type=int, default=1024)
    profile.add_argument("--time", type=float, default=1.0e-6)
    profile.add_argument("--output", type=Path, required=True)
    global_reference = sub.add_parser("global-reference")
    global_reference.add_argument("--resolution", type=int, choices=(32, 48, 64), required=True)
    global_reference.add_argument("--geometry", type=Path, required=True)
    global_reference.add_argument("--baseline", type=Path, required=True)
    global_reference.add_argument("--reference-sidecar", type=Path, required=True)
    global_reference.add_argument("--qualification-json", type=Path, required=True)
    global_reference.add_argument("--time", type=float, default=1.0e-6)
    global_reference.add_argument("--chunk-cells", type=int, default=2048)
    global_reference.add_argument("--output", type=Path, required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--resolution", type=int, choices=(32, 48, 64), required=True)
    replay.add_argument("--geometry", type=Path, required=True)
    replay.add_argument("--baseline", type=Path, required=True)
    replay.add_argument("--action-cache", type=Path, required=True)
    replay.add_argument("--reference-cache", type=Path, required=True)
    replay.add_argument("--cross-root", type=Path)
    replay.add_argument("--qualification-cache", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("cases", nargs=3, type=Path)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        _preflight(args)
    elif args.command == "actions":
        _build_actions(args)
    elif args.command == "qualify":
        _qualify(args)
    elif args.command == "profile-reference":
        _profile_reference(args)
    elif args.command == "global-reference":
        _global_reference(args)
    elif args.command == "replay":
        _replay(args)
    elif args.command == "merge":
        _merge(args)
    else:
        raise ValueError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
