#!/usr/bin/env python3
"""Factor the bounded HSX compatible generator into geometry and gradient errors.

The expensive ``extract`` command builds one runtime per resolution and writes a
compact, validated cache.  ``replay`` uses only that cache: it never builds the
model or calls the continuous evaluator.  The four generator labels are:

``HH``  production one-form crossed with the production face gradient
``EH``  continuous one-form crossed with the production face gradient
``HE``  production one-form crossed with the exact manufactured gradient
``EE``  continuous one-form crossed with the exact manufactured gradient

Here the first letter names the geometry factor and the second the derivative
factor.  Transported P/G/O values, centers, incidence, measures, and targets are
held fixed while the generator is changed consistently in both compatible terms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from drbx.native.fci_operators import (  # noqa: E402
    _compatible_flux_face_one_form,
    _compatible_flux_generator,
)

import audit_hsx_face_candidate_orientation as orientation  # noqa: E402
import audit_hsx_p03_failure_localization as localization  # noqa: E402
import audit_hsx_p03_fine_interval as bridge  # noqa: E402
import audit_hsx_p04_global_bracket as p04  # noqa: E402
import audit_hsx_poisson_common_face_oracle as oracle  # noqa: E402
import audit_hsx_poisson_continuous_baseline as continuous  # noqa: E402
import audit_hsx_poisson_vorticity as base  # noqa: E402


CACHE_SCHEMA = "drbx.hsx-generator-factor-cache-v1"
CASE_SCHEMA = "drbx.hsx-generator-factor-case-v1"
SUMMARY_SCHEMA = "drbx.hsx-generator-factor-summary-v1"
FIELD_NAMES = ("phi", *continuous.FIELDS)
TRANSPORT_FIELDS = continuous.FIELDS
RULES = ("P", "G", "O")
FACTORS = ("HH", "EH", "HE", "EE")
ACTIONS = ("A", "B", "C")
AXES = ("radial", "angular", "eta")


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


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _max_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / 1024.0**3


def _progress(stage: str, **details: Any) -> None:
    print(json.dumps({"event": "progress", "stage": stage, **details}, sort_keys=True), flush=True)


def _face_vector(families: Sequence[np.ndarray], key: Sequence[int]) -> np.ndarray:
    axis, i, j, k = (int(value) for value in key)
    return np.asarray(families[axis][i, j, k], dtype=np.float64)


def _face_scalar(families: Sequence[np.ndarray], key: Sequence[int]) -> float:
    axis, i, j, k = (int(value) for value in key)
    return float(families[axis][i, j, k])


def _normal_cross(axis: np.ndarray, one_form: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    """Return the selected component of ``a cross d`` for every face."""

    result = np.empty(len(axis), dtype=np.float64)
    masks = tuple(axis == value for value in range(3))
    result[masks[0]] = (
        one_form[masks[0], 1] * gradient[masks[0], 2]
        - one_form[masks[0], 2] * gradient[masks[0], 1]
    )
    result[masks[1]] = (
        one_form[masks[1], 2] * gradient[masks[1], 0]
        - one_form[masks[1], 0] * gradient[masks[1], 2]
    )
    result[masks[2]] = (
        one_form[masks[2], 0] * gradient[masks[2], 1]
        - one_form[masks[2], 1] * gradient[masks[2], 0]
    )
    return result


def _generator_matrix(
    face_axis: np.ndarray,
    face_measure: np.ndarray,
    collapsed: np.ndarray,
    one_form_h: np.ndarray,
    one_form_e: np.ndarray,
    gradient_h: np.ndarray,
    gradient_e: np.ndarray,
) -> dict[str, np.ndarray]:
    pairs = {
        "HH": (one_form_h, gradient_h),
        "EH": (one_form_e, gradient_h),
        "HE": (one_form_h, gradient_e),
        "EE": (one_form_e, gradient_e),
    }
    result = {}
    for label, (one_form, gradient) in pairs.items():
        value = _normal_cross(face_axis, one_form, gradient) * face_measure
        value = np.where(collapsed, 0.0, value)
        result[label] = value
    return result


def _identity_terms(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    geometry = values["EH"] - values["HH"]
    derivative = values["HE"] - values["HH"]
    interaction = values["EE"] - values["EH"] - values["HE"] + values["HH"]
    closure = values["EE"] - values["HH"] - geometry - derivative - interaction
    return {
        "geometry_only_change": geometry,
        "derivative_only_change": derivative,
        "bilinear_interaction": interaction,
        "identity_closure": closure,
    }


def _assemble_owner_components(
    generator: np.ndarray,
    argument: np.ndarray,
    center: np.ndarray,
    face_incidence: np.ndarray,
    raw_owner: np.ndarray,
    owner_volume: np.ndarray,
    *,
    sign: float,
    rho_star: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble face and matching center terms before owner restriction."""

    owner_count = len(owner_volume)
    face = np.zeros((owner_count, 3), dtype=np.float64)
    center_term = np.zeros((owner_count, 3), dtype=np.float64)
    for raw_index, owner_index in enumerate(raw_owner):
        for axis in range(3):
            lower, upper = face_incidence[raw_index, axis]
            face[owner_index, axis] += (
                generator[upper] * argument[upper]
                - generator[lower] * argument[lower]
            )
            center_term[owner_index, axis] -= center[raw_index] * (
                generator[upper] - generator[lower]
            )
    scale = float(sign) / (float(rho_star) * owner_volume[:, None])
    face *= scale
    center_term *= scale
    return face + center_term, face, center_term


def _statistics(values: np.ndarray, targets: np.ndarray, volumes: np.ndarray) -> dict[str, float]:
    error = values - targets
    return {
        "absolute_l2": float(np.sqrt(np.sum(volumes * error**2) / np.sum(volumes))),
        "signed_volume_weighted_mean_error": float(np.sum(volumes * error) / np.sum(volumes)),
        "maximum_absolute_error": float(np.max(np.abs(error))),
    }


def _source_identity(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _validate_source_identities(groups: Mapping[str, Any]) -> list[str]:
    reasons = []
    for group, entries in groups.items():
        if group == "derived":
            continue
        for name, identity in entries.items():
            path = Path(identity["path"])
            if not path.is_file():
                reasons.append(f"{group}:{name}:missing")
            elif _sha256(path) != identity["sha256"]:
                reasons.append(f"{group}:{name}:sha256")
    return reasons


def _save_cache(path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    payload = dict(arrays)
    payload["metadata_json"] = np.asarray(json.dumps(oracle._json_value(metadata), sort_keys=True))
    _write_npz(path, payload)
    marker = {
        "schema": CACHE_SCHEMA,
        "cache": str(path.resolve()),
        "cache_sha256": _sha256(path),
        "array_manifest": {
            name: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)}
            for name, value in arrays.items()
        },
        "complete": True,
    }
    _write_json(path.with_suffix(".complete.json"), marker)


def _load_cache(path: Path, *, validate_sources: bool = True) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    marker_path = path.with_suffix(".complete.json")
    if not marker_path.is_file():
        raise ValueError("factor cache completion marker is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema") != CACHE_SCHEMA or not marker.get("complete"):
        raise ValueError("factor cache completion marker is invalid")
    if marker.get("cache_sha256") != _sha256(path):
        raise ValueError("factor cache checksum does not match completion marker")
    with np.load(path, allow_pickle=False) as cached:
        if "metadata_json" not in cached.files:
            raise ValueError("factor cache metadata is missing")
        metadata = json.loads(str(cached["metadata_json"].item()))
        arrays = {name: np.asarray(cached[name]) for name in cached.files if name != "metadata_json"}
    if metadata.get("schema") != CACHE_SCHEMA:
        raise ValueError("factor cache schema mismatch")
    for name, spec in marker["array_manifest"].items():
        if name not in arrays:
            raise ValueError(f"factor cache array {name} is missing")
        if list(arrays[name].shape) != spec["shape"] or str(arrays[name].dtype) != spec["dtype"]:
            raise ValueError(f"factor cache array {name} has stale shape or dtype")
    if validate_sources:
        reasons = _validate_source_identities(metadata["identities"])
        if reasons:
            raise ValueError("stale factor cache: " + ", ".join(reasons))
    return arrays, metadata


def _extract(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    resolution = int(args.resolution)
    output = args.cache.resolve()
    _progress("extract_start", resolution=resolution, cache=str(output))

    artifact_path, artifact = base._load_artifact(args.geometry, resolution)
    _, identity_artifact = base._load_artifact(args.geometry, 64)
    manifest_path = artifact_path / "manifest.json"
    manifest_hash = _sha256(manifest_path)
    baseline_cache = args.baseline / f"N{resolution}.reference.npz"
    raw_state, regular_raw, eta_raw, exact_raw, baseline_provenance = p04._load_cache(
        baseline_cache, resolution, float(args.time)
    )
    if baseline_provenance["geometry_manifest_sha256"] != manifest_hash:
        raise ValueError("qualified baseline cache and geometry mismatch")

    selection_path = args.selection_root / f"N{resolution}_selection.json"
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_owners = [tuple(int(v) for v in record["owner"]) for record in selection_payload["records"]]
    selection = {
        label: [tuple(int(v) for v in owner) for owner in owners]
        for label, owners in selection_payload["selection"].items()
    }
    if selection_payload["geometry_manifest_sha256"] != manifest_hash:
        raise ValueError("frozen selection and geometry mismatch")

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
        raise RuntimeError("generator factorization requires a single-device model")
    runtime_seconds = time.perf_counter() - runtime_started
    rho_star = float(model.parameters.rho_star)
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

    raw_labels, face_labels = oracle._support(model.control_volume_geometry.cells, selection)
    del raw_labels
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
            raw_members.append(tuple(int(value) for value in member))
            raw_owner.append(owner_index)
    raw_members_array = np.asarray(raw_members, dtype=np.int32)
    raw_owner_array = np.asarray(raw_owner, dtype=np.int16)
    face_incidence = np.empty((len(raw_members), 3, 2), dtype=np.int32)
    for raw_index, raw in enumerate(raw_members):
        for axis in range(3):
            face_incidence[raw_index, axis, 0] = face_index[oracle._face_key(axis, raw, False)]
            face_incidence[raw_index, axis, 1] = face_index[oracle._face_key(axis, raw, True)]

    numerical_started = time.perf_counter()
    one_form_families = tuple(
        np.asarray(_compatible_flux_face_one_form(model.geometry, axis, b_floor=1.0e-30))
        for axis in range(3)
    )
    one_form_h = np.stack([_face_vector(one_form_families, key) for key in face_keys])
    one_form_h[collapsed] = 0.0

    gradient_h = np.zeros((len(FIELD_NAMES), len(face_keys), 3), dtype=np.float64)
    transport = np.zeros((len(RULES), len(FIELD_NAMES), len(face_keys)), dtype=np.float64)
    center = np.zeros((len(FIELD_NAMES), len(raw_members)), dtype=np.float64)
    upwind_left = np.zeros((len(TRANSPORT_FIELDS), len(face_keys)), dtype=np.float64)
    upwind_right = np.zeros_like(upwind_left)

    global_keys_array, _points_xy, _eta = p04._candidate_face_geometry(
        model, operands["actual_vorticity"]
    )
    global_keys = [tuple(int(value) for value in row) for row in global_keys_array]
    global_row = {key: index for index, key in enumerate(global_keys)}
    support_candidate_keys = [key for key in face_keys if key in global_row]
    support_candidate_rows = np.asarray([global_row[key] for key in support_candidate_keys])
    support_candidate_face = np.asarray([face_index[key] for key in support_candidate_keys])
    aggregates = {
        "G": oracle._aggregate_geometry(artifact.owner_geometry),
        "O": orientation._midpoint_observation_aggregate(
            artifact.owner_geometry, artifact.global_geometry
        )[0],
    }
    owner_values = {
        "phi": np.asarray(model._owner_field(owner_states["actual_vorticity"].phi)),
        **{
            name: np.asarray(model._owner_field(owner_states[name].vorticity))
            for name in TRANSPORT_FIELDS
        },
    }
    row_paths: dict[str, Path] = {}
    candidate_values: dict[str, dict[str, np.ndarray]] = {}
    for rule in ("G", "O"):
        row_path = args.p04_root / f"N{resolution}_{rule}_rows.npz"
        row_paths[rule] = row_path
        matrix, _diagnostics, metadata_json = localization._load_rows(row_path)
        row_metadata = json.loads(metadata_json)
        if row_metadata["geometry_manifest_sha256"] != manifest_hash:
            raise ValueError(f"{rule} saved rows and geometry mismatch")
        selected_matrix = matrix[support_candidate_rows]
        compact_ids = aggregates[rule].owner_flat_ids
        candidate_values[rule] = {
            name: np.asarray(
                selected_matrix.dot(np.asarray(values).reshape(-1)[compact_ids])
            ).reshape(-1)
            for name, values in owner_values.items()
        }

    for field_index, field in enumerate(FIELD_NAMES):
        op = operands["actual_vorticity"] if field == "phi" else operands[field]
        stencil = op.phi_stencil if field == "phi" else op.omega_stencil
        gradient_families = tuple(np.asarray(getattr(stencil.face_grad, axis)) for axis in "xyz")
        value_families = tuple(np.asarray(getattr(stencil.face_values, axis)) for axis in "xyz")
        gradient_h[field_index] = np.stack([_face_vector(gradient_families, key) for key in face_keys])
        gradient_h[field_index, collapsed] = 0.0
        transport[0, field_index] = np.asarray([_face_scalar(value_families, key) for key in face_keys])
        for rule_index, rule in enumerate(("G", "O"), start=1):
            transport[rule_index, field_index] = transport[0, field_index]
            transport[rule_index, field_index, support_candidate_face] = candidate_values[rule][field]
        center_array = np.asarray(stencil.x.center, dtype=np.float64)
        center[field_index] = np.asarray([center_array[tuple(raw)] for raw in raw_members])
        if field != "phi":
            transported_index = TRANSPORT_FIELDS.index(field)
            left_families = tuple(np.asarray(getattr(op.omega_direct_states[0], axis)) for axis in "xyz")
            right_families = tuple(np.asarray(getattr(op.omega_direct_states[1], axis)) for axis in "xyz")
            upwind_left[transported_index] = np.asarray([
                _face_scalar(left_families, key) for key in face_keys
            ])
            upwind_right[transported_index] = np.asarray([
                _face_scalar(right_families, key) for key in face_keys
            ])
    numerical_seconds = time.perf_counter() - numerical_started
    _progress("numerical_extraction_complete", resolution=resolution, seconds=numerical_seconds)

    exact_started = time.perf_counter()
    regular_points = face_points[regular_mask]
    metric = reference._metric(regular_points)
    one_form_e = np.zeros_like(one_form_h)
    one_form_e[regular_mask] = (
        np.asarray(metric["bcov"], dtype=np.float64)
        / np.maximum(np.asarray(metric["B"], dtype=np.float64)[:, None], 1.0e-30)
    )
    raw_fields = reference._fields_raw(regular_points, float(args.time))
    gradient_e = np.zeros_like(gradient_h)
    gradient_e[0, regular_mask] = np.stack(raw_fields["phi"][1:4], axis=-1)
    quick_preflight = {}
    for label in ("axis", "ordinary_control", "true_interface"):
        index = next(
            idx for idx, key in enumerate(face_keys)
            if regular_mask[idx]
            and any(str(tag).startswith(label + "_") for tag in face_labels[key])
        )
        quick_preflight[label] = list(face_keys[index])
        if not (
            np.all(np.isfinite(one_form_h[index]))
            and np.all(np.isfinite(one_form_e[index]))
            and np.all(np.isfinite(gradient_h[0, index]))
            and np.all(np.isfinite(gradient_e[0, index]))
        ):
            raise ValueError(f"nonfinite factor in {label} quick preflight")
    _progress(
        "real_hsx_preflight_complete",
        resolution=resolution,
        faces=quick_preflight,
    )
    omega_gradient, _ = reference._local_omega_derivatives(regular_points)
    gradient_e[1, regular_mask] = omega_gradient
    for field_index, field in enumerate(TRANSPORT_FIELDS[1:], start=2):
        _value, gradient = continuous._smooth_value_gradient(
            field, regular_points, reference.eta_period
        )
        gradient_e[field_index, regular_mask] = gradient
    exact_seconds = time.perf_counter() - exact_started
    _progress("exact_factor_complete", resolution=resolution, seconds=exact_seconds)

    # Real-HSX preflight categories: axis-adjacent but noncollapsed, ordinary,
    # and a true size-change/RLP-transition face.  HH must match production.
    phi_generator = _compatible_flux_generator(
        operands["actual_vorticity"].phi_stencil,
        model.geometry,
        domain=model.domain,
        axis_regular_axes=model.axis_regular_axes,
        b_floor=1.0e-30,
    )
    production_phi = tuple(np.asarray(getattr(phi_generator, axis)) for axis in "xyz")
    hh_phi = _normal_cross(face_axis, one_form_h, gradient_h[0])
    production_phi_support = np.asarray([_face_scalar(production_phi, key) for key in face_keys])
    production_phi_support[collapsed] = 0.0
    hh_replay = float(np.max(np.abs(hh_phi - production_phi_support)))
    category_owner = {
        record["category"]: tuple(int(v) for v in record["owner"])
        for record in selection_payload["records"]
    }
    axis_owner = category_owner["axis"]
    ordinary_owner = category_owner["ordinary_control"]
    transition_owner = category_owner["true_interface"]
    owner_raw = {
        owner: [raw_members[index] for index in np.flatnonzero(raw_owner_array == owner_index)]
        for owner_index, owner in enumerate(selected_owners)
    }
    def first_regular_face(owner: tuple[int, int, int]) -> tuple[int, int, int, int]:
        for raw in owner_raw[owner]:
            for axis in range(3):
                for upper in (False, True):
                    key = oracle._face_key(axis, raw, upper)
                    if regular_mask[face_index[key]]:
                        return key
        raise RuntimeError(f"owner {owner} has no regular face")
    preflight_keys = {
        "axis_adjacent_noncollapsed": first_regular_face(axis_owner),
        "ordinary": first_regular_face(ordinary_owner),
        "rlp_transition": first_regular_face(transition_owner),
    }
    for label, key in preflight_keys.items():
        index = face_index[key]
        if not (
            np.all(np.isfinite(one_form_h[index]))
            and np.all(np.isfinite(one_form_e[index]))
            and np.all(np.isfinite(gradient_h[:, index]))
            and np.all(np.isfinite(gradient_e[:, index]))
        ):
            raise ValueError(f"nonfinite factor in {label} preflight")
    collapsed_flux_max = float(np.max(np.abs(production_phi_support[collapsed]))) if np.any(collapsed) else 0.0
    if collapsed_flux_max != 0.0:
        raise ValueError("collapsed-axis integrated generator is not zero")

    prior_path = args.prior_root / f"N{resolution}.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior_hh = np.empty((len(TRANSPORT_FIELDS), len(RULES), len(ACTIONS), len(selected_owners)))
    prior_ee = np.empty_like(prior_hh)
    for field_index, field in enumerate(TRANSPORT_FIELDS):
        records = prior["fields"][field]["owner_records"]
        indexed = {tuple(row["owner"]): row for row in records}
        for rule_index, rule in enumerate(RULES):
            for action_index, action in enumerate(ACTIONS):
                prior_hh[field_index, rule_index, action_index] = [
                    indexed[owner][f"cross_{action}_{rule}_numerical"] for owner in selected_owners
                ]
                prior_ee[field_index, rule_index, action_index] = [
                    indexed[owner][f"cross_{action}_{rule}_exact_generator"] for owner in selected_owners
                ]

    owner_volume = np.asarray([aggregate_volume[owner] for owner in selected_owners])
    targets = np.stack([
        np.asarray([exact_owner[field][owner] for owner in selected_owners])
        for field in TRANSPORT_FIELDS
    ])
    category_names = [record["category"] for record in selection_payload["records"]]
    identities = {
        "physical": {
            "geometry_manifest": _source_identity(manifest_path),
            "selection": _source_identity(selection_path),
            "reference_sidecar": _source_identity(args.reference_sidecar),
        },
        "state": {
            "baseline_reference": _source_identity(baseline_cache),
            "rows_G": _source_identity(row_paths["G"]),
            "rows_O": _source_identity(row_paths["O"]),
            "fci_operators": _source_identity(ROOT / "src/drbx/native/fci_operators.py"),
            "continuum_reference": _source_identity(ROOT / "hsx_mms_continuum_reference.py"),
            "mms_driver": _source_identity(ROOT / "simulate_hsx_mms.py"),
        },
        "derived": {"prior_case": _source_identity(prior_path)},
    }
    arrays = {
        "face_keys": face_keys_array,
        "face_points": face_points,
        "face_measure": face_measure,
        "collapsed": collapsed.astype(np.uint8),
        "one_form_h": one_form_h,
        "one_form_e": one_form_e,
        "gradient_h": gradient_h,
        "gradient_e": gradient_e,
        "transport": transport,
        "center": center,
        "upwind_left": upwind_left,
        "upwind_right": upwind_right,
        "owner_keys": np.asarray(selected_owners, dtype=np.int32),
        "owner_volume": owner_volume,
        "raw_members": raw_members_array,
        "raw_owner": raw_owner_array,
        "face_incidence": face_incidence,
        "targets": targets,
        "prior_hh": prior_hh,
        "prior_ee": prior_ee,
    }
    metadata = {
        "schema": CACHE_SCHEMA,
        "resolution": resolution,
        "time": float(args.time),
        "rho_star": rho_star,
        "field_names": list(FIELD_NAMES),
        "transport_fields": list(TRANSPORT_FIELDS),
        "rules": list(RULES),
        "actions": list(ACTIONS),
        "owner_categories": category_names,
        "identities": identities,
        "layers": {
            "physical_face_geometry": ["geometry_manifest", "selection", "reference_sidecar"],
            "state_and_derivatives": [
                "baseline_reference", "rows_G", "rows_O", "fci_operators",
                "continuum_reference", "mms_driver",
            ],
            "derived_replay": ["prior_case"],
        },
        "normalization": {
            "a_h": "_compatible_flux_face_one_form = (unit b / B)_cov",
            "a_star": "continuous metric bcov/B, where bcov is covariant unit b",
            "face_measure": "logical midpoint face measure",
            "eta_geometry_period": float(reference.eta_period),
            "analytic_harmonics": "full 2pi",
        },
        "preflight": {
            "keys": {name: list(key) for name, key in preflight_keys.items()},
            "HH_phi_generator_max_abs_replay": hh_replay,
            "collapsed_phi_generator_max_abs": collapsed_flux_max,
        },
        "extraction_counters": {"model_build_calls": 1, "continuous_metric_batches": 1, "omega_gradient_batches": 1},
        "timings_seconds": {
            "runtime_setup": runtime_seconds,
            "numerical_extraction": numerical_seconds,
            "continuous_geometry_and_exact_derivatives": exact_seconds,
            "total_before_write": time.perf_counter() - started,
        },
        "maximum_rss_gib": _max_rss_gib(),
    }
    _save_cache(output, arrays, metadata)
    _progress("extract_complete", resolution=resolution, seconds=time.perf_counter() - started)
    return {"cache": str(output), "metadata": metadata}


def _replay_cache(cache_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    arrays, metadata = _load_cache(cache_path)
    resolution = int(metadata["resolution"])
    face_axis = arrays["face_keys"][:, 0]
    collapsed = arrays["collapsed"].astype(bool)
    owner_volume = arrays["owner_volume"]
    rho_star = float(metadata["rho_star"])
    field_results: dict[str, Any] = {}
    maximum_replay = {"HH": 0.0, "EE": 0.0}
    maximum_factor_identity = 0.0
    maximum_action_identity = 0.0

    regular = ~collapsed
    one_form_difference = arrays["one_form_h"][regular] - arrays["one_form_e"][regular]
    one_form_reference_norm = np.linalg.norm(arrays["one_form_e"][regular])
    factor_accuracy = {
        "one_form_h_vs_continuous": {
            "maximum_absolute_difference": float(np.max(np.abs(one_form_difference))),
            "relative_l2_difference": float(
                np.linalg.norm(one_form_difference)
                / max(one_form_reference_norm, np.finfo(float).tiny)
            ),
            "by_face_orientation": {},
        },
        "gradient_h_vs_exact": {},
    }
    for axis_index, axis_name in enumerate(AXES):
        mask = regular & (face_axis == axis_index)
        difference = arrays["one_form_h"][mask] - arrays["one_form_e"][mask]
        factor_accuracy["one_form_h_vs_continuous"]["by_face_orientation"][axis_name] = {
            "maximum_absolute_difference": float(np.max(np.abs(difference))),
            "relative_l2_difference": float(
                np.linalg.norm(difference)
                / max(np.linalg.norm(arrays["one_form_e"][mask]), np.finfo(float).tiny)
            ),
        }
    for field_index, field in enumerate(FIELD_NAMES):
        difference = arrays["gradient_h"][field_index, regular] - arrays["gradient_e"][field_index, regular]
        factor_accuracy["gradient_h_vs_exact"][field] = {
            "maximum_absolute_difference": float(np.max(np.abs(difference))),
            "relative_l2_difference": float(
                np.linalg.norm(difference)
                / max(np.linalg.norm(arrays["gradient_e"][field_index, regular]), np.finfo(float).tiny)
            ),
        }

    prior_identity = metadata["identities"]["derived"]["prior_case"]
    prior_path = Path(prior_identity["path"])
    prior_hash_matches = prior_path.is_file() and _sha256(prior_path) == prior_identity["sha256"]
    prior_sensitivity = None
    if prior_hash_matches:
        prior_payload = json.loads(prior_path.read_text(encoding="utf-8"))
        prior_sensitivity = prior_payload["reference"]["omega_gradient_sensitivity"]

    for field_index, field in enumerate(TRANSPORT_FIELDS):
        action_generators = {
            "A": _generator_matrix(
                face_axis, arrays["face_measure"], collapsed,
                arrays["one_form_h"], arrays["one_form_e"],
                arrays["gradient_h"][0], arrays["gradient_e"][0],
            ),
            "B": _generator_matrix(
                face_axis, arrays["face_measure"], collapsed,
                arrays["one_form_h"], arrays["one_form_e"],
                arrays["gradient_h"][field_index + 1], arrays["gradient_e"][field_index + 1],
            ),
        }
        factor_identity = {
            action: _identity_terms(generators) for action, generators in action_generators.items()
        }
        maximum_factor_identity = max(
            maximum_factor_identity,
            *(float(np.max(np.abs(terms["identity_closure"]))) for terms in factor_identity.values()),
        )
        rules_payload = {}
        owner_payload = [
            {
                "owner": arrays["owner_keys"][index].tolist(),
                "category": metadata["owner_categories"][index],
                "aggregate_volume": float(owner_volume[index]),
                "continuum_target": float(arrays["targets"][field_index, index]),
            }
            for index in range(len(owner_volume))
        ]
        for rule_index, rule in enumerate(RULES):
            values: dict[str, dict[str, np.ndarray]] = {action: {} for action in ACTIONS}
            components: dict[str, dict[str, dict[str, np.ndarray]]] = {
                action: {} for action in ACTIONS
            }
            for factor in FACTORS:
                a_total, a_face, a_center = _assemble_owner_components(
                    action_generators["A"][factor],
                    arrays["transport"][rule_index, field_index + 1],
                    arrays["center"][field_index + 1],
                    arrays["face_incidence"], arrays["raw_owner"], owner_volume,
                    sign=-1.0, rho_star=rho_star,
                )
                b_total, b_face, b_center = _assemble_owner_components(
                    action_generators["B"][factor],
                    arrays["transport"][rule_index, 0],
                    arrays["center"][0],
                    arrays["face_incidence"], arrays["raw_owner"], owner_volume,
                    sign=1.0, rho_star=rho_star,
                )
                values["A"][factor] = np.sum(a_total, axis=1)
                values["B"][factor] = np.sum(b_total, axis=1)
                values["C"][factor] = 0.5 * (values["A"][factor] + values["B"][factor])
                components["A"][factor] = {"face": a_face, "center": a_center, "total": a_total}
                components["B"][factor] = {"face": b_face, "center": b_center, "total": b_total}
                components["C"][factor] = {
                    "face": 0.5 * (a_face + b_face),
                    "center": 0.5 * (a_center + b_center),
                    "total": 0.5 * (a_total + b_total),
                }
            rules_payload[rule] = {}
            for action_index, action in enumerate(ACTIONS):
                terms = _identity_terms(values[action])
                maximum_action_identity = max(
                    maximum_action_identity, float(np.max(np.abs(terms["identity_closure"])))
                )
                maximum_replay["HH"] = max(
                    maximum_replay["HH"],
                    float(np.max(np.abs(values[action]["HH"] - arrays["prior_hh"][field_index, rule_index, action_index]))),
                )
                maximum_replay["EE"] = max(
                    maximum_replay["EE"],
                    float(np.max(np.abs(values[action]["EE"] - arrays["prior_ee"][field_index, rule_index, action_index]))),
                )
                rules_payload[rule][action] = {
                    "statistics": {
                        factor: _statistics(values[action][factor], arrays["targets"][field_index], owner_volume)
                        for factor in FACTORS
                    },
                    "signed_effect_statistics": {
                        name: {
                            "volume_weighted_rms": float(np.sqrt(np.sum(owner_volume * value**2) / np.sum(owner_volume))),
                            "signed_volume_weighted_mean": float(np.sum(owner_volume * value) / np.sum(owner_volume)),
                            "maximum_absolute": float(np.max(np.abs(value))),
                        }
                        for name, value in terms.items()
                    },
                }
                for owner_index, row in enumerate(owner_payload):
                    row.setdefault("results", {}).setdefault(rule, {})[action] = {
                        factor: float(values[action][factor][owner_index]) for factor in FACTORS
                    }
                    row["results"][rule][action]["effects"] = {
                        name: float(value[owner_index]) for name, value in terms.items()
                    }
                    row["results"][rule][action]["components"] = {
                        factor: {
                            part: dict(zip(AXES, components[action][factor][part][owner_index].tolist()))
                            for part in ("face", "center", "total")
                        }
                        for factor in FACTORS
                    }
        generator_qualification = {}
        for action in ("A", "B"):
            generators = action_generators[action]
            generator_qualification[action] = {
                factor: {
                    "rms_difference_from_EE": float(np.sqrt(np.mean((generators[factor] - generators["EE"])**2))),
                    "maximum_abs_difference_from_EE": float(np.max(np.abs(generators[factor] - generators["EE"]))),
                    "rms_magnitude": float(np.sqrt(np.mean(generators[factor]**2))),
                }
                for factor in FACTORS
            }
        field_results[field] = {
            "rules": rules_payload,
            "owners": owner_payload,
            "generator_qualification": generator_qualification,
        }

    return {
        "schema": CASE_SCHEMA,
        "resolution": resolution,
        "cache": {"path": str(cache_path.resolve()), "sha256": _sha256(cache_path)},
        "cache_hit": {
            "used": True,
            "model_build_calls": 0,
            "continuous_evaluator_calls": 0,
            "seconds": time.perf_counter() - started,
            "reason": "completion marker, cache checksum, schema/shapes, and physical/state source hashes match",
        },
        "factor_labels": {
            "HH": "production one-form x production gradient",
            "EH": "continuous one-form x production gradient (geometry-only substitution)",
            "HE": "production one-form x exact gradient (derivative-only substitution)",
            "EE": "continuous one-form x exact gradient",
        },
        "decomposition": "EE-HH = (EH-HH) + (HE-HH) + (EE-EH-HE+HH)",
        "replay": {
            "HH_against_prior_numerical_max_abs": maximum_replay["HH"],
            "EE_against_prior_exact_generator_max_abs": maximum_replay["EE"],
            "factor_identity_max_abs": maximum_factor_identity,
            "completed_action_identity_max_abs": maximum_action_identity,
        },
        "preflight": metadata["preflight"],
        "factor_accuracy": factor_accuracy,
        "reference_sensitivity": {
            "source": prior_identity,
            "source_hash_matches": prior_hash_matches,
            "omega_gradient": prior_sensitivity,
        },
        "extraction": {
            "timings_seconds": metadata["timings_seconds"],
            "maximum_rss_gib": metadata["maximum_rss_gib"],
            "counters": metadata["extraction_counters"],
        },
        "support": {
            "selected_owner_count": int(len(arrays["owner_keys"])),
            "raw_member_count": int(len(arrays["raw_members"])),
            "incident_face_count": int(len(arrays["face_keys"])),
            "collapsed_face_count": int(np.sum(collapsed)),
        },
        "identities": metadata["identities"],
        "fields": field_results,
        "scope": {
            "production_changes": [],
            "certification": "bounded diagnostic only; not a convergence order or operator qualification",
        },
    }


def _replay(args: argparse.Namespace) -> dict[str, Any]:
    payload = _replay_cache(args.cache.resolve())
    _write_json(args.output.resolve(), payload)
    _progress("replay_complete", resolution=payload["resolution"], output=str(args.output.resolve()))
    return payload


def _merge(args: argparse.Namespace) -> dict[str, Any]:
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in args.cases]
    cases.sort(key=lambda value: int(value["resolution"]))
    payload = {
        "schema": SUMMARY_SCHEMA,
        "resolutions": [case["resolution"] for case in cases],
        "cases": [str(path.resolve()) for path in args.cases],
        "production_changes": [],
        "four_way_C": {},
        "replay": {str(case["resolution"]): case["replay"] for case in cases},
        "cache_hit": {str(case["resolution"]): case["cache_hit"] for case in cases},
        "factor_accuracy": {
            str(case["resolution"]): case["factor_accuracy"] for case in cases
        },
        "reference_sensitivity": {
            str(case["resolution"]): case["reference_sensitivity"] for case in cases
        },
        "extraction": {str(case["resolution"]): case["extraction"] for case in cases},
        "preflight": {str(case["resolution"]): case["preflight"] for case in cases},
        "support": {str(case["resolution"]): case["support"] for case in cases},
    }
    for field in TRANSPORT_FIELDS:
        payload["four_way_C"][field] = {}
        for rule in ("O", "G", "P"):
            payload["four_way_C"][field][rule] = {
                factor: [
                    case["fields"][field]["rules"][rule]["C"]["statistics"][factor]["absolute_l2"]
                    for case in cases
                ]
                for factor in FACTORS
            }
            payload["four_way_C"][field][rule]["effects"] = {
                name: [
                    case["fields"][field]["rules"][rule]["C"]["signed_effect_statistics"][name]
                    for case in cases
                ]
                for name in (
                    "geometry_only_change", "derivative_only_change",
                    "bilinear_interaction", "identity_closure",
                )
            }
    _write_json(args.output.resolve(), payload)
    _progress("merge_complete", output=str(args.output.resolve()))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--resolution", type=int, choices=(48, 64), required=True)
    extract.add_argument("--geometry", type=Path, required=True)
    extract.add_argument("--reference-sidecar", type=Path, required=True)
    extract.add_argument("--baseline", type=Path, required=True)
    extract.add_argument("--selection-root", type=Path, required=True)
    extract.add_argument("--p04-root", type=Path, required=True)
    extract.add_argument("--prior-root", type=Path, required=True)
    extract.add_argument("--cache", type=Path, required=True)
    extract.add_argument("--time", type=float, default=1.0e-6)
    extract.set_defaults(function=_extract)
    replay = sub.add_parser("replay")
    replay.add_argument("--cache", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.set_defaults(function=_replay)
    merge = sub.add_parser("merge")
    merge.add_argument("cases", type=Path, nargs="+")
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(function=_merge)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
