#!/usr/bin/env python3
"""Bounded HSX value/derivative cross and cubic transported-value audit.

One extraction per resolution freezes complete selected-owner neighborhoods and
all numerical/analytic factors needed by the compatible centered bracket.
Replay and reporting operate only on the compact cache.  The four primary
labels vary transported face values (V) and generator derivatives (D):

``VhDh`` current O degree-two planar values with the global cubic derivative,
``VeDh`` analytic values on the same eligible planar faces,
``VhDe`` analytic generator derivatives with current O values, and
``VeDe`` both analytic substitutions.

The numerical cubic-value candidates reuse the exact global cubic derivative
donor policy and stored midpoint/raw-volume owner observation functional.  The
``C3_planar`` candidate changes the same planar families as O; ``C3_full`` also
changes periodic eta faces.  Physical boundary/collapsed-axis exceptions retain
the frozen numerical contract.
"""

from __future__ import annotations

import argparse
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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import simulate_hsx_mms as mms  # noqa: E402
from hsx_mms_continuum_reference import build_continuum_reference_from_sidecar  # noqa: E402
from drbx.native.fci_operators import _compatible_flux_face_one_form  # noqa: E402

import audit_hsx_cubic_derivative_global as cubic  # noqa: E402
import audit_hsx_cubic_global_bracket as global_bracket  # noqa: E402
import audit_hsx_generator_factorization as factor  # noqa: E402
import audit_hsx_p03_fine_interval as bridge  # noqa: E402
import audit_hsx_p04_global_bracket as p04  # noqa: E402
import audit_hsx_poisson_common_face_oracle as oracle  # noqa: E402
import audit_hsx_poisson_continuous_baseline as continuous  # noqa: E402
import audit_hsx_poisson_vorticity as base  # noqa: E402


CACHE_SCHEMA = "drbx.hsx-value-derivative-cross-cache-v1"
CASE_SCHEMA = "drbx.hsx-value-derivative-cross-case-v1"
SUMMARY_SCHEMA = "drbx.hsx-value-derivative-cross-summary-v1"
PROFILE_SCHEMA = "drbx.hsx-value-derivative-cross-profile-v1"
FIELD_NAMES = cubic.FIELD_NAMES
TRANSPORT_FIELDS = continuous.FIELDS
ACTIONS = ("A", "B", "C")
CROSS_LABELS = ("VhDh", "VeDh", "VhDe", "VeDe")
VALUE_RULES = ("P", "O2", "C3_planar", "C3_full", "Ve_planar")
REGION_NAMES = (
    "axis_ring",
    "physical_boundary_footprint",
    "true_size_change_interface",
    "agglomerated_interior",
    "ordinary",
)
REGION_CODE = {name: index for index, name in enumerate(REGION_NAMES)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(oracle._json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            **arrays,
            metadata_json=np.asarray(json.dumps(oracle._json_value(metadata), sort_keys=True)),
        )
    temporary.replace(path)
    marker = {
        "schema": CACHE_SCHEMA,
        "cache": str(path.resolve()),
        "cache_sha256": _sha256(path),
        "numerical_implementation_sha256": metadata["numerical_implementation_sha256"],
        "status": "complete",
    }
    _write_json(path.with_suffix(".complete.json"), marker)


def _load_npz(path: Path, *, validate_sources: bool = True) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    marker_path = path.with_suffix(".complete.json")
    if not (path.is_file() and marker_path.is_file()):
        raise ValueError("cross cache or completion marker is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("cache_sha256") != _sha256(path):
        raise ValueError("cross cache checksum mismatch")
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"].item()))
        arrays = {name: np.asarray(source[name]) for name in source.files if name != "metadata_json"}
    if metadata.get("schema") != CACHE_SCHEMA:
        raise ValueError("cross cache schema mismatch")
    if validate_sources:
        if metadata["numerical_implementation_sha256"] != _numerical_implementation_hash():
            raise ValueError("cross cache numerical implementation changed")
        stale = []
        for group in metadata["identities"].values():
            for identity in group.values():
                source = Path(identity["path"])
                if not source.is_file() or _sha256(source) != identity["sha256"]:
                    stale.append(str(source))
        if stale:
            raise ValueError("cross cache source identity changed: " + ", ".join(stale))
    return arrays, metadata


def _identity(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}


def _max_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / 1024.0**3


def _progress(stage: str, **details: Any) -> None:
    print(json.dumps({"event": "value_derivative_cross", "stage": stage, **details}, sort_keys=True), flush=True)


def _face_vector(families: Sequence[np.ndarray], key: Sequence[int]) -> np.ndarray:
    axis, i, j, k = (int(value) for value in key)
    return np.asarray(families[axis][i, j, k], dtype=np.float64)


def _face_scalar(families: Sequence[np.ndarray], key: Sequence[int]) -> float:
    axis, i, j, k = (int(value) for value in key)
    return float(families[axis][i, j, k])


def _fallback_families(root: Path, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = (
        np.zeros((resolution + 1, resolution, resolution), dtype=np.uint8),
        np.zeros((resolution, resolution + 1, resolution), dtype=np.uint8),
        np.zeros((resolution, resolution, resolution + 1), dtype=np.uint8),
    )
    for axis in range(3):
        for eta_index in range(resolution):
            path = root / f"N{resolution}" / "chunks" / f"axis{axis}_eta{eta_index:03d}.npz"
            with np.load(path, allow_pickle=False) as source:
                keys = np.asarray(source["keys_ij"], dtype=np.int32)
                values = np.asarray(source["row_fallback"], dtype=np.uint8)
            if axis == 0:
                result[axis][keys[:, 0], keys[:, 1], eta_index] = values
            elif axis == 1:
                result[axis][keys[:, 0], keys[:, 1], eta_index] = values
            else:
                result[axis][keys[:, 0], keys[:, 1], eta_index] = values
        if axis == 2:
            result[axis][:, :, resolution] = result[axis][:, :, 0]
    return result


def _owner_fallback(cells: Any, families: Sequence[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    raw = np.maximum.reduce((
        families[0][:-1], families[0][1:],
        families[1][:, :-1], families[1][:, 1:],
        families[2][:, :, :-1], families[2][:, :, 1:],
    ))
    owner_arrays = bridge._owner_tuple_arrays(cells)
    owner_flat = np.ravel_multi_index(owner_arrays, shape)
    result = np.zeros(int(np.prod(shape)), dtype=np.uint8)
    np.maximum.at(result, owner_flat.reshape(-1), raw.reshape(-1))
    return result.reshape(shape)


def _select_owners(
    masks: Mapping[str, np.ndarray],
    owner_fallback: np.ndarray,
    angular_group_size: np.ndarray,
) -> tuple[list[tuple[int, int, int]], list[dict[str, Any]]]:
    """Geometry-stratified deterministic 28-owner selection."""

    target_counts = {
        "agglomerated_interior": 12,
        "true_size_change_interface": 6,
        "ordinary": 6,
        "axis_ring": 2,
        "physical_boundary_footprint": 2,
    }
    selected: list[tuple[int, int, int]] = []
    records: list[dict[str, Any]] = []
    shape = owner_fallback.shape

    for region in (
        "agglomerated_interior", "true_size_change_interface", "ordinary",
        "axis_ring", "physical_boundary_footprint",
    ):
        candidates = [tuple(int(v) for v in row) for row in np.argwhere(masks[region])]
        if not candidates:
            raise RuntimeError(f"no candidates for region {region}")
        count = target_counts[region]
        radial_targets = np.linspace(0.15, 0.85, count)
        eta_targets = np.mod(np.arange(count) * 0.3819660112501051 + 0.125, 1.0)
        for slot in range(count):
            want_expanded = slot % 2 == 0 and region not in {"axis_ring", "physical_boundary_footprint"}
            remaining = [owner for owner in candidates if owner not in selected]
            ranked = sorted(
                remaining,
                key=lambda owner: (
                    int((owner_fallback[owner] > 0) != want_expanded),
                    abs(owner[0] / max(shape[0] - 1, 1) - radial_targets[slot]),
                    min(
                        abs(owner[2] / shape[2] - eta_targets[slot]),
                        1.0 - abs(owner[2] / shape[2] - eta_targets[slot]),
                    ),
                    owner[1],
                    np.ravel_multi_index(owner, shape),
                ),
            )
            if not ranked:
                break
            owner = ranked[0]
            selected.append(owner)
            records.append({
                "owner": list(owner),
                "region": region,
                "slot": slot,
                "requested_support": "expanded" if want_expanded else "base",
                "observed_support": "expanded" if owner_fallback[owner] > 0 else "base",
                "maximum_incident_fallback_level": int(owner_fallback[owner]),
                "angular_group_size": int(angular_group_size[owner[0]]),
                "selection_basis": "geometry-stratified radial/eta target; no MMS value or residual used",
            })
    if len(selected) != sum(target_counts.values()) or len(set(selected)) != len(selected):
        raise RuntimeError("geometry-stratified selection did not produce 28 distinct owners")
    return selected, records


def _value_weights(
    context: cubic.BuildContext,
    donors: np.ndarray,
    points: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = points[:, 0]
    theta = points[:, 1]
    center = np.column_stack((u * np.cos(theta), u * np.sin(theta), points[:, 2]))
    observation = cubic._centered_observations(context, donors, center, scale)
    eta = cubic.base._unwrap_periodic(
        context.owner_eta[donors], center[:, None, 2], context.eta_period
    )
    distance = np.sqrt(
        ((context.arrays["owner_centroid_xy"][donors, 0] - center[:, None, 0]) / scale[:, None, 0])**2
        + ((context.arrays["owner_centroid_xy"][donors, 1] - center[:, None, 1]) / scale[:, None, 1])**2
        + ((eta - center[:, None, 2]) / scale[:, None, 2])**2
    )
    kappa = 1.0 / (1.0 + distance**2)
    weighted = observation * kappa[:, :, None]
    gram = np.einsum("bni,bnj->bij", weighted, weighted, optimize=True)
    target = np.zeros((len(points), len(cubic.EXPONENTS)), dtype=np.float64)
    target[:, 0] = 1.0
    solution = np.linalg.solve(gram, target[:, :, None])[:, :, 0]
    weights = kappa**2 * np.einsum("bni,bi->bn", observation, solution, optimize=True)
    residual = np.max(
        np.abs(np.einsum("bni,bn->bi", observation, weights) - target), axis=1
    )
    amplification = np.sum(np.abs(weights), axis=1)
    return weights, residual, amplification


def _cubic_value_rows(
    context: cubic.BuildContext,
    keys: np.ndarray,
    points: np.ndarray,
    cached_gradients: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    count = len(keys)
    values = np.full((len(FIELD_NAMES), count), np.nan, dtype=np.float64)
    diagnostics = {
        "fallback_level": np.full(count, 255, dtype=np.uint8),
        "condition": np.full(count, np.nan),
        "support_radius": np.full(count, np.nan),
        "value_amplification": np.full(count, np.nan),
        "value_reproduction_residual": np.full(count, np.nan),
        "derivative_reproduction_residual": np.full(count, np.nan),
        "derivative_action_replay": np.full(count, np.nan),
        "donor_count": np.zeros(count, dtype=np.int16),
    }
    schedule = cubic.POLICY["deficient_row_expansion_schedule"]
    groups: dict[tuple[int, int], list[int]] = {}
    for row, key in enumerate(keys):
        groups.setdefault((int(key[0]), int(key[3]) % context.resolution), []).append(row)
    for (axis, eta_index), rows in groups.items():
        pending = np.asarray(rows, dtype=np.int64)
        for level, policy in enumerate(schedule):
            if not len(pending):
                break
            donors, weights_d, row_diag, _ = cubic._row_batch(
                context, axis, eta_index, points[pending], exact_query=True,
                count=int(policy["donors_per_plane"]),
                pool_count=int(policy["candidate_pool_per_plane"]),
            )
            valid = (
                (row_diag["rank"] == len(cubic.EXPONENTS))
                & (np.max(row_diag["residual"], axis=1) <= cubic.base.REPRODUCTION_TOLERANCE)
                & (row_diag["minimum_coverage"] >= cubic.POLICY["minimum_distinct_angular_columns_per_plane"])
            )
            if level == len(schedule) - 1 and not np.all(valid):
                raise RuntimeError("fixed cubic policy remains deficient after its final expansion")
            accepted_local = np.flatnonzero(valid)
            if len(accepted_local):
                accepted = pending[accepted_local]
                accepted_donors = donors[accepted_local]
                accepted_dw = weights_d[accepted_local]
                accepted_points = points[accepted]
                accepted_scale = row_diag["scale"][accepted_local]
                weights_v, value_residual, value_amp = _value_weights(
                    context, accepted_donors, accepted_points, accepted_scale
                )
                values[:, accepted] = np.einsum(
                    "fan,an->fa", context.arrays["owner_values"][:, accepted_donors], weights_v,
                    optimize=True,
                )
                regular_gradient = np.einsum(
                    "fan,anc->fac", context.arrays["owner_values"][:, accepted_donors], accepted_dw,
                    optimize=True,
                )
                logical_gradient = cubic._regular_to_logical_batch(regular_gradient, accepted_points)
                replay = np.max(np.abs(logical_gradient - cached_gradients[:, accepted]), axis=(0, 2))
                diagnostics["fallback_level"][accepted] = level
                diagnostics["condition"][accepted] = row_diag["condition"][accepted_local]
                diagnostics["support_radius"][accepted] = row_diag["support_radius"][accepted_local]
                diagnostics["value_amplification"][accepted] = value_amp
                diagnostics["value_reproduction_residual"][accepted] = value_residual
                diagnostics["derivative_reproduction_residual"][accepted] = np.max(
                    row_diag["residual"][accepted_local], axis=1
                )
                diagnostics["derivative_action_replay"][accepted] = replay
                diagnostics["donor_count"][accepted] = accepted_donors.shape[1]
            pending = pending[~valid]
    if not np.all(np.isfinite(values)):
        raise RuntimeError("cubic value rows are incomplete")
    return values, diagnostics


def _assemble(
    generator: np.ndarray,
    transported: np.ndarray,
    center: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    *,
    sign: float,
    rho_star: float,
) -> np.ndarray:
    total, _face, _center = factor._assemble_owner_components(
        generator, transported, center,
        arrays["face_incidence"], arrays["raw_owner"], arrays["owner_volume"],
        sign=sign, rho_star=rho_star,
    )
    return np.sum(total, axis=1)


def _statistics(
    values: np.ndarray,
    targets: np.ndarray,
    volume: np.ndarray,
    region_code: np.ndarray,
    expanded: np.ndarray,
) -> dict[str, Any]:
    error = values - targets
    squared = volume * error**2
    total = float(np.sum(squared))
    reference_energy = float(np.sum(volume * targets**2))
    payload: dict[str, Any] = {
        "absolute_l2": float(np.sqrt(total / np.sum(volume))),
        "relative_l2": float(np.sqrt(total / reference_energy)) if reference_energy > 0.0 else math.inf,
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "owner_count": int(len(values)),
        "regions": {},
        "support_classes": {},
    }
    for name, code in REGION_CODE.items():
        mask = region_code == code
        contribution = float(np.sum(squared[mask]))
        payload["regions"][name] = {
            "owner_count": int(np.count_nonzero(mask)),
            "absolute_l2": float(np.sqrt(contribution / np.sum(volume[mask]))) if np.any(mask) else None,
            "squared_error_fraction": contribution / max(total, np.finfo(float).tiny),
            "maximum_absolute_error": float(np.max(np.abs(error[mask]))) if np.any(mask) else None,
        }
    for name, mask in (("base", ~expanded), ("expanded", expanded)):
        contribution = float(np.sum(squared[mask]))
        payload["support_classes"][name] = {
            "owner_count": int(np.count_nonzero(mask)),
            "absolute_l2": float(np.sqrt(contribution / np.sum(volume[mask]))) if np.any(mask) else None,
            "squared_error_fraction": contribution / max(total, np.finfo(float).tiny),
            "maximum_absolute_error": float(np.max(np.abs(error[mask]))) if np.any(mask) else None,
        }
    return payload


def _effect_statistics(value: np.ndarray, volume: np.ndarray) -> dict[str, float]:
    return {
        "volume_weighted_rms": float(np.sqrt(np.sum(volume * value**2) / np.sum(volume))),
        "signed_volume_weighted_mean": float(np.sum(volume * value) / np.sum(volume)),
        "maximum_absolute": float(np.max(np.abs(value))),
    }


def _numerical_implementation_hash() -> str:
    sources = [
        inspect.getsource(_fallback_families), inspect.getsource(_owner_fallback),
        inspect.getsource(_select_owners), inspect.getsource(_value_weights),
        inspect.getsource(_cubic_value_rows), inspect.getsource(_assemble),
        inspect.getsource(_extract),
    ]
    return hashlib.sha256("\n".join(sources).encode()).hexdigest()


def _profile(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    context = cubic._load_context(args.geometry, args.baseline, args.resolution)
    rows = []
    for axis in range(3):
        keys_ij, points = cubic._planar_faces(context, axis, 0)
        collapsed = keys_ij[:, 0] == 0 if axis == 0 else np.zeros(len(keys_ij), dtype=bool)
        regular = np.flatnonzero(~collapsed)
        selected = regular[np.unique(np.linspace(0, len(regular) - 1, min(args.rows, len(regular)), dtype=int))]
        rows.extend(
            (
                np.asarray((axis, int(keys_ij[index, 0]), int(keys_ij[index, 1]), 0), dtype=np.int32),
                points[index],
            )
            for index in selected
        )
    keys = np.asarray([row[0] for row in rows], dtype=np.int32)
    points = np.asarray([row[1] for row in rows], dtype=np.float64)
    gradients, _ = cubic.load_global_gradients(args.derivative_root / f"N{args.resolution}")
    cached = np.stack([
        np.stack([_face_vector(gradients[field], key) for key in keys])
        for field in FIELD_NAMES
    ])
    values, diagnostics = _cubic_value_rows(context, keys, points, cached)
    payload = {
        "schema": PROFILE_SCHEMA,
        "resolution": args.resolution,
        "row_count": len(keys),
        "seconds": time.perf_counter() - started,
        "finite": bool(np.all(np.isfinite(values))),
        "maximum_value_reproduction_residual": float(np.max(diagnostics["value_reproduction_residual"])),
        "maximum_derivative_action_replay": float(np.max(diagnostics["derivative_action_replay"])),
        "fallback_counts": {
            str(level): int(np.count_nonzero(diagnostics["fallback_level"] == level))
            for level in range(4)
        },
        "maximum_rss_gib": _max_rss_gib(),
    }
    _write_json(args.output, payload)
    _progress("profile_complete", output=str(args.output.resolve()), seconds=payload["seconds"])
    return payload


def _extract(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    resolution = int(args.resolution)
    artifact_path, artifact = base._load_artifact(args.geometry, resolution)
    _, identity_artifact = base._load_artifact(args.geometry, 64)
    manifest_path = artifact_path / "manifest.json"
    manifest_hash = _sha256(manifest_path)
    baseline_cache = args.baseline / f"N{resolution}.reference.npz"
    raw_state, regular_raw, eta_raw, exact_raw, baseline_provenance = p04._load_cache(
        baseline_cache, resolution, float(args.time)
    )
    if baseline_provenance["geometry_manifest_sha256"] != manifest_hash:
        raise ValueError("baseline and geometry identity mismatch")
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
    runtime_args = SimpleNamespace(
        shard_counts=(1, 1, 1), curvature_edge_one_form=False, reference=reference,
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
        raise RuntimeError("cross extraction requires one local model")
    runtime_seconds = time.perf_counter() - runtime_started
    _progress("runtime_complete", resolution=resolution, seconds=runtime_seconds)

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
    exact_owner = {
        name: np.asarray(mms._owner_project_array(exact_raw[name], artifact.owner_geometry))
        for name in TRANSPORT_FIELDS
    }

    derivative_case = args.derivative_root / f"N{resolution}"
    global_gradients, derivative_manifest = cubic.load_global_gradients(derivative_case)
    if derivative_manifest["identity"]["geometry"]["manifest"]["sha256"] != manifest_hash:
        raise ValueError("global cubic derivative and geometry identity mismatch")
    candidate_values, value_provenance = global_bracket._transported_values(
        args.p04_root, resolution, artifact, model, owner_states,
        p04._candidate_face_geometry(model, operands["actual_vorticity"])[0],
    )
    eligible_keys_array, _eligible_xy, _eligible_eta = p04._candidate_face_geometry(
        model, operands["actual_vorticity"]
    )
    eligible_keys = [tuple(int(v) for v in row) for row in eligible_keys_array]
    eligible_row = {key: index for index, key in enumerate(eligible_keys)}

    standard = base._region_masks(artifact.global_geometry, artifact.owner_geometry)
    masks, _ordinary_layer, _adjacency = bridge._refined_masks(
        artifact.owner_geometry, model.control_volume_geometry.cells, standard
    )
    fallback_families = _fallback_families(args.derivative_root, resolution)
    active_shape = tuple(np.asarray(artifact.owner_geometry.topology.is_active_owner).shape)
    incident_fallback = _owner_fallback(
        model.control_volume_geometry.cells, fallback_families, active_shape
    )
    selected_owners, selection_records = _select_owners(
        masks, incident_fallback,
        np.asarray(artifact.owner_geometry.angular_group_size, dtype=np.int64),
    )
    selection = {
        f"{record['region']}_{record['slot']}": [tuple(record["owner"])]
        for record in selection_records
    }
    _progress(
        "selection_complete", resolution=resolution, owners=len(selected_owners),
        expanded=sum(record["observed_support"] == "expanded" for record in selection_records),
    )

    _raw_labels, face_labels = oracle._support(model.control_volume_geometry.cells, selection)
    face_keys = sorted(face_labels)
    face_index = {key: index for index, key in enumerate(face_keys)}
    face_keys_array = np.asarray(face_keys, dtype=np.int32)
    face_axis = face_keys_array[:, 0].astype(np.int8)
    face_points = np.asarray([
        oracle._face_points_weights(model.geometry, key, 1)[0][0] for key in face_keys
    ], dtype=np.float64)
    face_measure = np.asarray([
        oracle._face_points_weights(model.geometry, key, 1)[1][0] for key in face_keys
    ], dtype=np.float64)
    regular_mask = bridge._regular_reference_face_mask(model, face_keys)
    collapsed = ~regular_mask
    eligible_planar = np.asarray([key in eligible_row for key in face_keys], dtype=bool)
    eta_upgrade = regular_mask & (face_axis == 2)
    physical_boundary = regular_mask & (face_axis == 0) & (face_keys_array[:, 1] == resolution)
    exact_derivative_mask = regular_mask & ~physical_boundary
    full_upgrade = eligible_planar | eta_upgrade

    cells = model.control_volume_geometry.cells
    owner_arrays = bridge._owner_tuple_arrays(cells)
    aggregate_volume = np.asarray(cells.aggregate_volume, dtype=np.float64)
    raw_members: list[tuple[int, int, int]] = []
    raw_owner = []
    for owner_index, owner in enumerate(selected_owners):
        members = np.argwhere(
            (owner_arrays[0] == owner[0])
            & (owner_arrays[1] == owner[1])
            & (owner_arrays[2] == owner[2])
        )
        for member in members:
            raw_members.append(tuple(int(v) for v in member))
            raw_owner.append(owner_index)
    raw_members_array = np.asarray(raw_members, dtype=np.int32)
    raw_owner_array = np.asarray(raw_owner, dtype=np.int16)
    face_incidence = np.empty((len(raw_members), 3, 2), dtype=np.int32)
    for raw_index, raw in enumerate(raw_members):
        for axis in range(3):
            face_incidence[raw_index, axis, 0] = face_index[oracle._face_key(axis, raw, False)]
            face_incidence[raw_index, axis, 1] = face_index[oracle._face_key(axis, raw, True)]

    one_form_families = tuple(
        np.asarray(_compatible_flux_face_one_form(model.geometry, axis, b_floor=1.0e-30))
        for axis in range(3)
    )
    one_form = np.stack([_face_vector(one_form_families, key) for key in face_keys])
    one_form[collapsed] = 0.0
    gradient_dh = np.stack([
        np.stack([_face_vector(global_gradients[field], key) for key in face_keys])
        for field in FIELD_NAMES
    ])

    values_p = np.zeros((len(FIELD_NAMES), len(face_keys)), dtype=np.float64)
    values_o = np.zeros_like(values_p)
    center = np.zeros((len(FIELD_NAMES), len(raw_members)), dtype=np.float64)
    for field_index, field in enumerate(FIELD_NAMES):
        op = operands["actual_vorticity"] if field == "phi" else operands[field]
        stencil = op.phi_stencil if field == "phi" else op.omega_stencil
        production_families = tuple(np.asarray(getattr(stencil.face_values, axis)) for axis in "xyz")
        values_p[field_index] = np.asarray([_face_scalar(production_families, key) for key in face_keys])
        values_o[field_index] = values_p[field_index]
        candidate = candidate_values["O"][field]
        for support_index, key in enumerate(face_keys):
            row = eligible_row.get(key)
            if row is not None:
                values_o[field_index, support_index] = candidate[row]
        center_array = np.asarray(stencil.x.center, dtype=np.float64)
        center[field_index] = np.asarray([center_array[tuple(raw)] for raw in raw_members])

    upgrade_rows = np.flatnonzero(full_upgrade)
    factor_checkpoint = args.cache.with_name(args.cache.stem + ".factors.npz")
    factor_identity = {
        "schema": "drbx.hsx-value-derivative-factor-checkpoint-v1",
        "resolution": resolution,
        "geometry_manifest_sha256": manifest_hash,
        "baseline_reference_sha256": _sha256(baseline_cache),
        "derivative_manifest_sha256": _sha256(derivative_case / "manifest.json"),
        "face_keys_sha256": _array_sha256(face_keys_array),
        "numerical_implementation_sha256": _numerical_implementation_hash(),
    }
    factor_identity_json = json.dumps(factor_identity, sort_keys=True)
    factor_hit = False
    if factor_checkpoint.is_file():
        try:
            with np.load(factor_checkpoint, allow_pickle=False) as saved:
                if str(saved["identity_json"].item()) == factor_identity_json:
                    exact_value = np.asarray(saved["exact_value"])
                    analytic_value = np.asarray(saved["analytic_value"])
                    gradient_de = np.asarray(saved["gradient_de"])
                    values_c3_planar = np.asarray(saved["values_c3_planar"])
                    values_c3_full = np.asarray(saved["values_c3_full"])
                    cubic_diagnostics = {
                        name: np.asarray(saved[f"cubic_{name}"])
                        for name in (
                            "fallback_level", "condition", "support_radius",
                            "value_amplification", "value_reproduction_residual",
                            "derivative_reproduction_residual", "derivative_action_replay",
                            "donor_count",
                        )
                    }
                    factor_hit = True
        except (OSError, KeyError, ValueError):
            factor_hit = False
    if not factor_hit:
        exact_value = values_o.copy()
        gradient_de = gradient_dh.copy()
        regular_points = face_points[regular_mask]
        raw_fields = reference._fields_raw(regular_points, float(args.time))
        analytic_value = np.zeros((len(FIELD_NAMES), len(face_keys)), dtype=np.float64)
        analytic_gradient = np.zeros_like(gradient_dh)
        analytic_value[0, regular_mask] = raw_fields["phi"][0]
        analytic_gradient[0, regular_mask] = np.stack(raw_fields["phi"][1:4], axis=-1)
        prepared = reference.prepare(regular_points)
        if prepared.mms_omega is None:
            raise RuntimeError("continuous vorticity value is unavailable")
        analytic_value[1, regular_mask] = np.asarray(prepared.mms_omega, dtype=np.float64)
        omega_gradient, _omega_hessian = reference._local_omega_derivatives(regular_points)
        analytic_gradient[1, regular_mask] = omega_gradient
        for field_index, field in enumerate(TRANSPORT_FIELDS[1:], start=2):
            value, gradient = continuous._smooth_value_gradient(field, regular_points, reference.eta_period)
            analytic_value[field_index, regular_mask] = value
            analytic_gradient[field_index, regular_mask] = gradient
        exact_value[:, eligible_planar] = analytic_value[:, eligible_planar]
        gradient_de[:, exact_derivative_mask] = analytic_gradient[:, exact_derivative_mask]

        cubic_context = cubic._load_context(args.geometry, args.baseline, resolution)
        cubic_values, cubic_diagnostics = _cubic_value_rows(
            cubic_context, face_keys_array[upgrade_rows], face_points[upgrade_rows],
            gradient_dh[:, upgrade_rows],
        )
        values_c3_planar = values_p.copy()
        values_c3_full = values_p.copy()
        local_planar = eligible_planar[upgrade_rows]
        values_c3_planar[:, upgrade_rows[local_planar]] = cubic_values[:, local_planar]
        values_c3_full[:, upgrade_rows] = cubic_values
        temporary = factor_checkpoint.with_name(factor_checkpoint.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                identity_json=np.asarray(factor_identity_json),
                exact_value=exact_value,
                analytic_value=analytic_value,
                gradient_de=gradient_de,
                values_c3_planar=values_c3_planar,
                values_c3_full=values_c3_full,
                **{f"cubic_{name}": value for name, value in cubic_diagnostics.items()},
            )
        temporary.replace(factor_checkpoint)
    _progress(
        "factor_checkpoint_ready", resolution=resolution, cache_hit=factor_hit,
        path=str(factor_checkpoint.resolve()),
    )

    owner_volume = np.asarray([aggregate_volume[owner] for owner in selected_owners])
    targets = np.stack([
        np.asarray([exact_owner[field][owner] for owner in selected_owners])
        for field in TRANSPORT_FIELDS
    ])
    owner_region = np.asarray([
        REGION_CODE[record["region"]] for record in selection_records
    ], dtype=np.int8)
    owner_expanded = np.asarray([
        record["observed_support"] == "expanded" for record in selection_records
    ], dtype=np.uint8)

    # Full-operator replay uses the same global O/Dh policy before restriction.
    candidate_gradient = {
        field: global_bracket._gradient_stencil(global_gradients[field]) for field in FIELD_NAMES
    }
    full_replay = np.empty((len(TRANSPORT_FIELDS), len(ACTIONS), len(selected_owners)))
    for field_index, field in enumerate(TRANSPORT_FIELDS):
        runner = global_bracket._runner(model, operands[field])
        field_faces, phi_faces, left, right = global_bracket._rule_arguments(
            operands[field], eligible_keys_array, candidate_values, "O", field
        )
        result = runner(
            field_faces, phi_faces, candidate_gradient[field], candidate_gradient["phi"], left, right
        )
        for action_index, action in enumerate(ACTIONS):
            full_replay[field_index, action_index] = np.asarray([
                result[action][owner] for owner in selected_owners
            ])
    _progress("full_operator_replay_complete", resolution=resolution)

    face_fallback = np.asarray([
        _face_scalar(fallback_families, key) for key in face_keys
    ], dtype=np.uint8)
    arrays = {
        "face_keys": face_keys_array,
        "face_points": face_points,
        "face_measure": face_measure,
        "collapsed": collapsed.astype(np.uint8),
        "eligible_planar": eligible_planar.astype(np.uint8),
        "eta_upgrade": eta_upgrade.astype(np.uint8),
        "physical_boundary": physical_boundary.astype(np.uint8),
        "exact_derivative_mask": exact_derivative_mask.astype(np.uint8),
        "face_fallback": face_fallback,
        "one_form": one_form,
        "gradient_dh": gradient_dh,
        "gradient_de": gradient_de,
        "values_P": values_p,
        "values_O2": values_o,
        "values_Ve_planar": exact_value,
        "values_analytic_full": analytic_value,
        "values_C3_planar": values_c3_planar,
        "values_C3_full": values_c3_full,
        "center": center,
        "owner_keys": np.asarray(selected_owners, dtype=np.int32),
        "owner_volume": owner_volume,
        "owner_region": owner_region,
        "owner_expanded": owner_expanded,
        "raw_members": raw_members_array,
        "raw_owner": raw_owner_array,
        "face_incidence": face_incidence,
        "targets": targets,
        "full_operator_replay": full_replay,
        "cubic_upgrade_rows": upgrade_rows,
        **{f"cubic_{name}": value for name, value in cubic_diagnostics.items()},
    }
    identities = {
        "physical": {
            "geometry_manifest": _identity(manifest_path),
            "reference_sidecar": _identity(args.reference_sidecar),
            "baseline_reference": _identity(baseline_cache),
        },
        "numerical": {
            "global_derivative_manifest": _identity(derivative_case / "manifest.json"),
            "O_rows": value_provenance["O"] | {},
            "fci_operators": _identity(ROOT / "src/drbx/native/fci_operators.py"),
            "global_cubic_builder": _identity(SCRIPTS / "audit_hsx_cubic_derivative_global.py"),
            "global_bracket_audit": _identity(SCRIPTS / "audit_hsx_cubic_global_bracket.py"),
            "continuum_reference": _identity(ROOT / "hsx_mms_continuum_reference.py"),
            "mms_driver": _identity(ROOT / "simulate_hsx_mms.py"),
        },
    }
    # Keep only the stable source identity fields from the row provenance.
    identities["numerical"]["O_rows"] = {
        "path": value_provenance["O"]["path"],
        "sha256": value_provenance["O"]["sha256"],
    }
    metadata = {
        "schema": CACHE_SCHEMA,
        "resolution": resolution,
        "time": float(args.time),
        "rho_star": float(model.parameters.rho_star),
        "field_names": list(FIELD_NAMES),
        "transport_fields": list(TRANSPORT_FIELDS),
        "actions": list(ACTIONS),
        "regions": list(REGION_NAMES),
        "selection_records": selection_records,
        "selection_contract": {
            "kind": "geometry-stratified control sample",
            "error_selected": False,
            "complete_owner_neighborhoods": True,
            "target_counts": {name: int(np.count_nonzero(owner_region == code)) for name, code in REGION_CODE.items()},
        },
        "face_family_contract": {
            "O2_and_Ve_planar": "eligible radial/angular non-trace faces from P04; other values remain production",
            "C3_planar": "same eligible planar faces as O2",
            "C3_full": "eligible planar plus periodic eta faces; physical boundary/collapsed exceptions remain production",
            "Dh": "frozen global balanced-cubic derivative on every face family",
            "De": "analytic derivative on regular non-physical-boundary faces; Dh retained on frozen physical boundary and collapsed faces",
        },
        "cubic_value_contract": {
            "policy": cubic.POLICY,
            "target": "point value at face: centered constant coefficient 1, all other monomial targets 0",
            "observation": "stored midpoint-projected/raw-volume owner moments, identical to global cubic derivative context",
            "upgrade_face_count": int(len(upgrade_rows)),
            "eligible_planar_face_count": int(np.count_nonzero(eligible_planar)),
            "eta_face_count": int(np.count_nonzero(eta_upgrade)),
            "maximum_value_reproduction_residual": float(np.max(cubic_diagnostics["value_reproduction_residual"])),
            "maximum_derivative_action_replay": float(np.max(cubic_diagnostics["derivative_action_replay"])),
            "maximum_value_amplification": float(np.max(cubic_diagnostics["value_amplification"])),
            "maximum_support_radius": float(np.max(cubic_diagnostics["support_radius"])),
            "fallback_counts": {
                str(level): int(np.count_nonzero(cubic_diagnostics["fallback_level"] == level))
                for level in range(4)
            },
        },
        "identities": identities,
        "numerical_implementation_sha256": _numerical_implementation_hash(),
        "timings_seconds": {
            "runtime_setup": runtime_seconds,
            "total_before_write": time.perf_counter() - started,
        },
        "maximum_rss_gib": _max_rss_gib(),
    }
    _write_npz(args.cache.resolve(), arrays, metadata)
    selection_payload = {
        "schema": "drbx.hsx-value-derivative-cross-selection-v1",
        "resolution": resolution,
        "geometry_manifest_sha256": manifest_hash,
        "records": selection_records,
        "selection_contract": metadata["selection_contract"],
        "owner_keys_sha256": _array_sha256(arrays["owner_keys"]),
    }
    _write_json(args.selection.resolve(), selection_payload)
    _progress("extract_complete", resolution=resolution, seconds=time.perf_counter() - started)
    return {"cache": str(args.cache.resolve()), "selection": str(args.selection.resolve())}


def _replay_cache(cache: Path) -> dict[str, Any]:
    started = time.perf_counter()
    arrays, metadata = _load_npz(cache)
    face_axis = arrays["face_keys"][:, 0]
    collapsed = arrays["collapsed"].astype(bool)
    volume = arrays["owner_volume"]
    region_code = arrays["owner_region"]
    expanded = arrays["owner_expanded"].astype(bool)
    rho_star = float(metadata["rho_star"])
    generators = {}
    for field_index, field in enumerate(FIELD_NAMES):
        generators[field] = {
            "Dh": factor._normal_cross(
                face_axis, arrays["one_form"], arrays["gradient_dh"][field_index]
            ) * arrays["face_measure"],
            "De": factor._normal_cross(
                face_axis, arrays["one_form"], arrays["gradient_de"][field_index]
            ) * arrays["face_measure"],
        }
        generators[field]["Dh"][collapsed] = 0.0
        generators[field]["De"][collapsed] = 0.0

    values_by_label = {
        label: arrays[f"values_{label}"] for label in VALUE_RULES
    }
    cross_values = {"Vh": arrays["values_O2"], "Ve": arrays["values_Ve_planar"]}
    fields = {}
    maximum_replay = 0.0
    maximum_decomposition = 0.0
    maximum_antisymmetry = 0.0
    maximum_cross_identity = 0.0
    for field_index, field in enumerate(TRANSPORT_FIELDS):
        scalar_index = field_index + 1
        target = arrays["targets"][field_index]
        cross_actions: dict[str, dict[str, np.ndarray]] = {action: {} for action in ACTIONS}
        for label in CROSS_LABELS:
            value_choice = "Ve" if label.startswith("Ve") else "Vh"
            derivative_choice = "De" if label.endswith("De") else "Dh"
            values = cross_values[value_choice]
            action_a = _assemble(
                generators["phi"][derivative_choice], values[scalar_index], arrays["center"][scalar_index], arrays,
                sign=-1.0, rho_star=rho_star,
            )
            action_b = _assemble(
                generators[field][derivative_choice], values[0], arrays["center"][0], arrays,
                sign=1.0, rho_star=rho_star,
            )
            cross_actions["A"][label] = action_a
            cross_actions["B"][label] = action_b
            cross_actions["C"][label] = 0.5 * (action_a + action_b)
            maximum_decomposition = max(
                maximum_decomposition,
                float(np.max(np.abs(cross_actions["C"][label] - 0.5 * (action_a + action_b)))),
            )
            swapped = 0.5 * (-action_b - action_a)
            maximum_antisymmetry = max(
                maximum_antisymmetry,
                float(np.max(np.abs(cross_actions["C"][label] + swapped))),
            )

        action_payload = {}
        for action_index, action in enumerate(ACTIONS):
            hh = cross_actions[action]["VhDh"]
            eh = cross_actions[action]["VeDh"]
            he = cross_actions[action]["VhDe"]
            ee = cross_actions[action]["VeDe"]
            value_only = eh - hh
            derivative_only = he - hh
            mixed = ee - eh - he + hh
            closure = ee - hh - value_only - derivative_only - mixed
            maximum_cross_identity = max(maximum_cross_identity, float(np.max(np.abs(closure))))
            maximum_replay = max(
                maximum_replay,
                float(np.max(np.abs(hh - arrays["full_operator_replay"][field_index, action_index]))),
            )
            action_payload[action] = {
                "cross": {
                    label: _statistics(cross_actions[action][label], target, volume, region_code, expanded)
                    for label in CROSS_LABELS
                },
                "effects": {
                    "value_only_change": _effect_statistics(value_only, volume),
                    "derivative_only_change": _effect_statistics(derivative_only, volume),
                    "mixed": _effect_statistics(mixed, volume),
                    "identity_closure": _effect_statistics(closure, volume),
                },
            }

        candidate_payload = {}
        candidate_vectors: dict[str, dict[str, np.ndarray]] = {}
        for label, values in values_by_label.items():
            action_a = _assemble(
                generators["phi"]["Dh"], values[scalar_index], arrays["center"][scalar_index], arrays,
                sign=-1.0, rho_star=rho_star,
            )
            action_b = _assemble(
                generators[field]["Dh"], values[0], arrays["center"][0], arrays,
                sign=1.0, rho_star=rho_star,
            )
            candidate_vectors[label] = {"A": action_a, "B": action_b, "C": 0.5 * (action_a + action_b)}
            candidate_payload[label] = {
                action: _statistics(candidate_vectors[label][action], target, volume, region_code, expanded)
                for action in ACTIONS
            }

        # Minimal operand split for C only.
        value_phi_only_a = _assemble(
            generators["phi"]["Dh"], cross_values["Vh"][scalar_index], arrays["center"][scalar_index], arrays,
            sign=-1.0, rho_star=rho_star,
        )
        value_phi_only_b = _assemble(
            generators[field]["Dh"], cross_values["Ve"][0], arrays["center"][0], arrays,
            sign=1.0, rho_star=rho_star,
        )
        value_scalar_only_a = _assemble(
            generators["phi"]["Dh"], cross_values["Ve"][scalar_index], arrays["center"][scalar_index], arrays,
            sign=-1.0, rho_star=rho_star,
        )
        value_scalar_only_b = _assemble(
            generators[field]["Dh"], cross_values["Vh"][0], arrays["center"][0], arrays,
            sign=1.0, rho_star=rho_star,
        )
        derivative_phi_only = 0.5 * (
            _assemble(
                generators["phi"]["De"], cross_values["Vh"][scalar_index], arrays["center"][scalar_index], arrays,
                sign=-1.0, rho_star=rho_star,
            )
            + cross_actions["B"]["VhDh"]
        )
        derivative_scalar_only = 0.5 * (
            cross_actions["A"]["VhDh"]
            + _assemble(
                generators[field]["De"], cross_values["Vh"][0], arrays["center"][0], arrays,
                sign=1.0, rho_star=rho_star,
            )
        )
        fields[field] = {
            "actions": action_payload,
            "cubic_value_candidates_with_Dh": candidate_payload,
            "C_operand_splits": {
                "analytic_phi_value_only": _statistics(
                    0.5 * (value_phi_only_a + value_phi_only_b), target, volume, region_code, expanded
                ),
                "analytic_scalar_value_only": _statistics(
                    0.5 * (value_scalar_only_a + value_scalar_only_b), target, volume, region_code, expanded
                ),
                "analytic_phi_derivative_only": _statistics(
                    derivative_phi_only, target, volume, region_code, expanded
                ),
                "analytic_scalar_derivative_only": _statistics(
                    derivative_scalar_only, target, volume, region_code, expanded
                ),
            },
        }

    eligible = arrays["eligible_planar"].astype(bool)
    eta = arrays["eta_upgrade"].astype(bool)
    value_face_accuracy = {}
    for field_index, field in enumerate(FIELD_NAMES):
        exact = arrays["values_Ve_planar"][field_index]
        analytic_full = arrays["values_analytic_full"][field_index]
        value_face_accuracy[field] = {}
        for label in ("O2", "C3_planar", "C3_full"):
            values = arrays[f"values_{label}"][field_index]
            value_face_accuracy[field][label] = {
                "eligible_planar_rms_difference_from_analytic": float(np.sqrt(np.mean((values[eligible] - exact[eligible])**2))),
                "eligible_planar_maximum_difference_from_analytic": float(np.max(np.abs(values[eligible] - exact[eligible]))),
            }
            if label == "C3_full" and np.any(eta):
                value_face_accuracy[field][label]["eta_face_count"] = int(np.count_nonzero(eta))
                value_face_accuracy[field][label]["eta_rms_difference_from_analytic"] = float(
                    np.sqrt(np.mean((values[eta] - analytic_full[eta])**2))
                )
                value_face_accuracy[field][label]["eta_maximum_difference_from_analytic"] = float(
                    np.max(np.abs(values[eta] - analytic_full[eta]))
                )

    return {
        "schema": CASE_SCHEMA,
        "resolution": int(metadata["resolution"]),
        "cache": {"path": str(cache.resolve()), "sha256": _sha256(cache)},
        "selection": metadata["selection_contract"] | {"records": metadata["selection_records"]},
        "face_family_contract": metadata["face_family_contract"],
        "cubic_value_contract": metadata["cubic_value_contract"],
        "fields": fields,
        "value_face_accuracy": value_face_accuracy,
        "verification": {
            "full_operator_VhDh_replay_max_abs": maximum_replay,
            "C_decomposition_max_abs": maximum_decomposition,
            "swapped_antisymmetry_max_abs": maximum_antisymmetry,
            "cross_vector_identity_max_abs": maximum_cross_identity,
            "constant_value_reproduction_max_abs": metadata["cubic_value_contract"]["maximum_value_reproduction_residual"],
            "global_derivative_action_replay_max_abs": metadata["cubic_value_contract"]["maximum_derivative_action_replay"],
        },
        "resources": {
            "extraction_timings_seconds": metadata["timings_seconds"],
            "extraction_maximum_rss_gib": metadata["maximum_rss_gib"],
            "replay_seconds": time.perf_counter() - started,
        },
        "identities": metadata["identities"],
        "scope": {
            "kind": "bounded geometry-stratified completed-owner diagnostic",
            "global_orders": False,
            "production_changes": [],
            "material_upwinding_certified": False,
        },
    }


def _replay(args: argparse.Namespace) -> dict[str, Any]:
    payload = _replay_cache(args.cache.resolve())
    _write_json(args.output.resolve(), payload)
    _progress("replay_complete", resolution=payload["resolution"], output=str(args.output.resolve()))
    return payload


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Bounded HSX value/derivative cross and cubic face-value comparison",
        "",
        "This geometry-stratified N48/N64 study uses complete selected-owner neighborhoods. Vh is current O degree-two transported value data on eligible planar faces; Dh is the frozen global cubic derivative; Ve and De are analytic substitutions under the same numerical geometry and frozen boundary contract.",
        "",
        "## Four crossed completed actions",
        "",
        "| N | field | action | VhDh | VeDh | VhDe | VeDe |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for case in summary["cases"]:
        for field in TRANSPORT_FIELDS:
            for action in ACTIONS:
                cross = case["fields"][field]["actions"][action]["cross"]
                lines.append(
                    f"| {case['resolution']} | {field} | {action} | "
                    + " | ".join(f"{cross[label]['absolute_l2']:.6g}" for label in CROSS_LABELS)
                    + " |"
                )
    lines.extend([
        "",
        "The mixed vector is `VeDe - VeDh - VhDe + VhDh`; value-only and derivative-only scalar L2 changes are not treated as additive.",
        "",
        "## Cubic transported-value candidates with Dh fixed",
        "",
        "| N | field | action | P | O2 | C3 planar | C3 full | analytic planar |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ])
    for case in summary["cases"]:
        for field in TRANSPORT_FIELDS:
            candidates = case["fields"][field]["cubic_value_candidates_with_Dh"]
            for action in ACTIONS:
                lines.append(
                    f"| {case['resolution']} | {field} | {action} | "
                    + " | ".join(
                        f"{candidates[label][action]['absolute_l2']:.6g}"
                        for label in VALUE_RULES
                    )
                    + " |"
                )
    lines.extend([
        "",
        "## Selection, reconstruction, and verification",
        "",
    ])
    for case in summary["cases"]:
        contract = case["cubic_value_contract"]
        verification = case["verification"]
        expanded = sum(
            record["observed_support"] == "expanded"
            for record in case["selection"]["records"]
        )
        lines.append(
            f"N{case['resolution']} uses {len(case['selection']['records'])} owners ({expanded} with expanded incident support), "
            f"{contract['eligible_planar_face_count']} eligible planar faces and {contract['eta_face_count']} eta faces. "
            f"Maximum cubic value reproduction residual is `{contract['maximum_value_reproduction_residual']:.3e}`; "
            f"derivative-action replay is `{contract['maximum_derivative_action_replay']:.3e}`. "
            f"Full-operator VhDh replay is `{verification['full_operator_VhDh_replay_max_abs']:.3e}`; "
            f"cross/decomposition/antisymmetry closures are at most "
            f"`{max(verification['cross_vector_identity_max_abs'], verification['C_decomposition_max_abs'], verification['swapped_antisymmetry_max_abs']):.3e}`."
        )
    lines.extend([
        "",
        "The O2/Ve/C3-planar comparison is like-for-like on the radial/angular non-trace families previously replaced by O. C3-full additionally replaces periodic eta values. Physical boundary traces and collapsed-axis exceptions remain under the frozen numerical contract.",
        "",
        "## Outcome",
        "",
        summary["decision"],
        "",
        "This is bounded diagnostic evidence, not a global order, material-upwind, production, evolved-MMS, curvature, diffusion, polarization, or blob-driver claim. Regional and base/expanded-support details, maximum errors, operand splits, identities, and timings are retained in the machine-readable cases.",
        "",
    ])
    return "\n".join(lines)


def _merge(args: argparse.Namespace) -> dict[str, Any]:
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in args.cases]
    cases.sort(key=lambda item: int(item["resolution"]))
    comparison = {}
    for field in TRANSPORT_FIELDS:
        comparison[field] = {}
        for action in ACTIONS:
            comparison[field][action] = {
                label: [
                    case["fields"][field]["cubic_value_candidates_with_Dh"][label][action]["absolute_l2"]
                    for case in cases
                ]
                for label in VALUE_RULES
            }
    ratios = {
        field: {
            label: [
                comparison[field]["C"][label][index]
                / comparison[field]["C"]["O2"][index]
                for index in range(len(cases))
            ]
            for label in ("P", "C3_planar", "C3_full", "Ve_planar")
        }
        for field in TRANSPORT_FIELDS
    }
    smooth = TRANSPORT_FIELDS[1:]
    planar_improves = all(
        ratio < 1.0 for field in smooth for ratio in ratios[field]["C3_planar"]
    )
    full_improves = all(
        ratio < 1.0 for field in smooth for ratio in ratios[field]["C3_full"]
    )
    vorticity_planar_ok = all(
        comparison["actual_vorticity"]["C"]["C3_planar"][index]
        <= max(
            comparison["actual_vorticity"]["C"]["O2"][index],
            comparison["actual_vorticity"]["C"]["P"][index],
        )
        for index in range(len(cases))
    )
    vorticity_full_ok = all(
        comparison["actual_vorticity"]["C"]["C3_full"][index]
        <= max(
            comparison["actual_vorticity"]["C"]["O2"][index],
            comparison["actual_vorticity"]["C"]["P"][index],
        )
        for index in range(len(cases))
    )
    verification_max = max(
        value
        for case in cases
        for value in case["verification"].values()
    )
    if planar_improves and vorticity_planar_ok:
        decision = (
            "The like-for-like C3-planar value candidate improves both failing smooth controls at N48 and N64 without a material sampled vorticity regression. Its fixed geometry-only policy warrants the authorized frozen global 32/48/64 static continuation; bounded rates remain non-certifying."
        )
        recommendation = "continue_global_C3_planar"
    elif full_improves and vorticity_full_ok:
        decision = (
            "C3-full, but not the like-for-like planar candidate, improves both failing smooth controls at N48 and N64 without a material sampled vorticity regression. The eta-family change is therefore material and must be carried explicitly into the authorized frozen global continuation."
        )
        recommendation = "continue_global_C3_full"
    else:
        decision = (
            "Neither fixed cubic value candidate gives a consistent bounded improvement across both failing smooth controls and resolutions without a material vorticity regression. Stop before a global repeat and use the crossed/operand-split evidence to choose the next compatible functional or integration experiment."
        )
        recommendation = "stop_bounded"
    payload = {
        "schema": SUMMARY_SCHEMA,
        "resolutions": [case["resolution"] for case in cases],
        "comparison": comparison,
        "C_error_ratios_to_O2": ratios,
        "recommendation": recommendation,
        "decision": decision,
        "verification_maximum": verification_max,
        "cases": cases,
        "case_paths": [str(path.resolve()) for path in args.cases],
        "scope": {
            "bounded": True,
            "global_orders": False,
            "production_changes": [],
            "material_upwinding_certified": False,
        },
    }
    _write_json(args.output.resolve(), payload)
    report = args.output.with_name("report.md")
    report.write_text(_render_report(payload), encoding="utf-8")
    _progress("merge_complete", output=str(args.output.resolve()), recommendation=recommendation)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    profile = sub.add_parser("profile")
    profile.add_argument("--resolution", type=int, choices=(48, 64), default=48)
    profile.add_argument("--geometry", type=Path, required=True)
    profile.add_argument("--baseline", type=Path, required=True)
    profile.add_argument("--derivative-root", type=Path, required=True)
    profile.add_argument("--rows", type=int, default=64)
    profile.add_argument("--output", type=Path, required=True)
    profile.set_defaults(function=_profile)
    extract = sub.add_parser("extract")
    extract.add_argument("--resolution", type=int, choices=(48, 64), required=True)
    extract.add_argument("--geometry", type=Path, required=True)
    extract.add_argument("--reference-sidecar", type=Path, required=True)
    extract.add_argument("--baseline", type=Path, required=True)
    extract.add_argument("--p04-root", type=Path, required=True)
    extract.add_argument("--derivative-root", type=Path, required=True)
    extract.add_argument("--time", type=float, default=1.0e-6)
    extract.add_argument("--cache", type=Path, required=True)
    extract.add_argument("--selection", type=Path, required=True)
    extract.set_defaults(function=_extract)
    replay = sub.add_parser("replay")
    replay.add_argument("--cache", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.set_defaults(function=_replay)
    merge = sub.add_parser("merge")
    merge.add_argument("cases", nargs=2, type=Path)
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(function=_merge)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
