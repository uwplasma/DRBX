#!/usr/bin/env python3
"""Bounded N48/N64 localization of the failed global P/G/O brackets.

The diagnostic reuses the frozen P03 owner selections, qualified midpoint
references, and serialized P04 G/O rows.  It separates advected face values,
generator/center terms, and the scalar-upwind left/right jump without changing
production code or defining a new numerical scheme.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
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
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import simulate_hsx_mms as mms  # noqa: E402
from hsx_mms_continuum_reference import build_continuum_reference_from_sidecar  # noqa: E402
from drbx.native.fci_operators import _compatible_flux_generator  # noqa: E402

import audit_hsx_face_candidate_orientation as orientation  # noqa: E402
import audit_hsx_p03_fine_interval as bridge  # noqa: E402
import audit_hsx_p04_global_bracket as p04  # noqa: E402
import audit_hsx_poisson_common_face_oracle as oracle  # noqa: E402
import audit_hsx_poisson_continuous_baseline as continuous  # noqa: E402
import audit_hsx_poisson_vorticity as base  # noqa: E402


SCHEMA = "drbx.hsx-p03-failure-localization-case-v1"
SUMMARY_SCHEMA = "drbx.hsx-p03-failure-localization-summary-v1"
FIELDS = continuous.FIELDS
RULES = ("P", "G", "O")
AXES = ("radial", "angular", "eta")
CONFIGURATIONS = (
    "numerical",
    "exact_face",
    "exact_generator",
    "both_exact",
    "both_exact_reference_center",
    "center_only",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(oracle._json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _progress(stage: str, **details: Any) -> None:
    print(json.dumps({"event": "progress", "stage": stage, **details}, sort_keys=True), flush=True)


def _max_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / 1024.0**3


def _load_rows(path: Path) -> tuple[csr_matrix, dict[str, np.ndarray], str]:
    with np.load(path, allow_pickle=False) as cached:
        shape = tuple(int(value) for value in cached["shape"])
        matrix = csr_matrix(
            (cached["data"], cached["indices"], cached["indptr"]), shape=shape
        )
        diagnostics = {
            name: np.asarray(cached[name])
            for name in (
                "rank", "condition", "reproduction_residual", "l1_norm",
                "donor_counts", "expansion_steps",
            )
        }
        metadata = str(cached["metadata"].item())
    return matrix, diagnostics, metadata


def _selected_statistics(
    records: Sequence[Mapping[str, Any]], value_key: str, target_key: str = "continuum_target"
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category in ("all", *sorted({str(row["category"]) for row in records})):
        chosen = list(records) if category == "all" else [
            row for row in records if row["category"] == category
        ]
        weight = np.asarray([row["aggregate_volume"] for row in chosen], dtype=np.float64)
        actual = np.asarray([row[value_key] for row in chosen], dtype=np.float64)
        target = np.asarray([row[target_key] for row in chosen], dtype=np.float64)
        error = actual - target
        result[category] = {
            "owner_count": len(chosen),
            "physical_volume": float(np.sum(weight)),
            "absolute_l2": float(np.sqrt(np.sum(weight * error**2) / np.sum(weight))),
            "signed_volume_weighted_mean_error": float(np.sum(weight * error) / np.sum(weight)),
            "maximum_absolute_error": float(np.max(np.abs(error))),
        }
    return result


def _face_maps(
    keys: Sequence[tuple[int, int, int, int]], values: Sequence[float]
) -> dict[tuple[int, int, int, int], float]:
    return {key: float(value) for key, value in zip(keys, values)}


def _assemble_components(
    owner: tuple[int, int, int],
    members: Sequence[tuple[int, int, int]],
    aggregate_volume: np.ndarray,
    center: np.ndarray,
    generator: Mapping[tuple[int, int, int, int], float],
    argument: Mapping[tuple[int, int, int, int], float],
    *,
    sign: float,
    rho_star: float,
) -> dict[str, Any]:
    face_by_axis = np.zeros(3, dtype=np.float64)
    center_by_axis = np.zeros(3, dtype=np.float64)
    for raw in members:
        for axis in range(3):
            lower = oracle._face_key(axis, raw, False)
            upper = oracle._face_key(axis, raw, True)
            face_by_axis[axis] += (
                generator[upper] * argument[upper]
                - generator[lower] * argument[lower]
            )
            center_by_axis[axis] -= float(center[raw]) * (
                generator[upper] - generator[lower]
            )
    scale = float(sign) / (float(rho_star) * float(aggregate_volume[owner]))
    face_by_axis *= scale
    center_by_axis *= scale
    total_by_axis = face_by_axis + center_by_axis
    return {
        "value": float(np.sum(total_by_axis)),
        "face_by_orientation": dict(zip(AXES, face_by_axis.tolist())),
        "center_by_orientation": dict(zip(AXES, center_by_axis.tolist())),
        "total_by_orientation": dict(zip(AXES, total_by_axis.tolist())),
        "face_total": float(np.sum(face_by_axis)),
        "center_total": float(np.sum(center_by_axis)),
    }


def _mean_jump(
    left: p04.CoordinateFaceValues3D,
    right: p04.CoordinateFaceValues3D,
    keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_values = p04._face_values_at_keys(left, keys)
    right_values = p04._face_values_at_keys(right, keys)
    return 0.5 * (left_values + right_values), right_values - left_values


def _states_from_mean_jump(
    original_left: p04.CoordinateFaceValues3D,
    original_right: p04.CoordinateFaceValues3D,
    keys: np.ndarray,
    mean: np.ndarray,
    jump: np.ndarray,
) -> tuple[p04.CoordinateFaceValues3D, p04.CoordinateFaceValues3D]:
    return (
        p04._replace_faces_vectorized(original_left, keys, mean - 0.5 * jump),
        p04._replace_faces_vectorized(original_right, keys, mean + 0.5 * jump),
    )


def _case(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    resolution = int(args.resolution)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _progress("case_start", resolution=resolution)

    artifact_path, artifact = base._load_artifact(args.geometry, resolution)
    _, identity_artifact = base._load_artifact(args.geometry, 64)
    manifest_hash = _sha256(artifact_path / "manifest.json")
    cache_path = args.baseline / f"N{resolution}.reference.npz"
    raw_state, regular_raw, eta_raw, exact_raw, cache_provenance = p04._load_cache(
        cache_path, resolution, float(args.time)
    )
    if cache_provenance["geometry_manifest_sha256"] != manifest_hash:
        raise ValueError("qualified reference cache and geometry mismatch")

    selection_path = args.selection_root / f"N{resolution}_selection.json"
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection_payload["geometry_manifest_sha256"] != manifest_hash:
        raise ValueError("frozen selection and geometry mismatch")
    if selection_payload["reference_cache_sha256"] != _sha256(cache_path):
        raise ValueError("frozen selection and reference cache mismatch")
    selection = {
        label: [tuple(int(v) for v in owner) for owner in owners]
        for label, owners in selection_payload["selection"].items()
    }
    selected_owners = [tuple(record["owner"]) for record in selection_payload["records"]]

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
        runtime = mms._runtime(
            artifact.global_geometry, artifact.owner_geometry, runtime_args
        )
    model = runtime.model
    if model is None:
        raise RuntimeError("localization requires a single-device model")
    runtime_seconds = time.perf_counter() - runtime_started
    rho_star = float(model.parameters.rho_star)

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
        name: mms._owner_project_array(exact_raw[name], artifact.owner_geometry)
        for name in FIELDS
    }

    # The selection was frozen before the P04 global candidates existed.  Its
    # complete raw-member support is reconstructed here without truncation.
    raw_labels, face_labels = oracle._support(
        model.control_volume_geometry.cells, selection
    )
    face_keys = sorted(face_labels)
    face_key_set = set(face_keys)
    global_keys_array, _points_xy, _eta = p04._candidate_face_geometry(
        model, operands["actual_vorticity"]
    )
    global_keys = [tuple(int(v) for v in row) for row in global_keys_array]
    global_row = {key: index for index, key in enumerate(global_keys)}
    support_candidate_keys = [key for key in face_keys if key in global_row]
    support_candidate_rows = np.asarray(
        [global_row[key] for key in support_candidate_keys], dtype=np.int64
    )
    support_keys_array = np.asarray(support_candidate_keys, dtype=np.int32)

    p04_case_path = args.p04_root / f"N{resolution}.json"
    p04_case = json.loads(p04_case_path.read_text(encoding="utf-8"))
    if p04_case["geometry"]["manifest_sha256"] != manifest_hash:
        raise ValueError("P04 case and geometry mismatch")
    if p04_case["support"]["candidate_face_key_sha256"] != p04._array_sha256(global_keys_array):
        raise ValueError("saved P04 rows do not match current face ordering")

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
            for name in FIELDS
        },
    }
    candidate_values: dict[str, dict[str, np.ndarray]] = {}
    row_evidence = {}
    for rule in ("G", "O"):
        row_path = args.p04_root / f"N{resolution}_{rule}_rows.npz"
        matrix, diagnostics, metadata = _load_rows(row_path)
        if matrix.shape[0] != len(global_keys):
            raise ValueError(f"{rule} row count does not match candidate faces")
        selected_matrix = matrix[support_candidate_rows]
        compact_ids = aggregates[rule].owner_flat_ids
        candidate_values[rule] = {
            name: np.asarray(
                selected_matrix.dot(np.asarray(values).reshape(-1)[compact_ids])
            ).reshape(-1)
            for name, values in owner_values.items()
        }
        row_evidence[rule] = {
            "path": str(row_path.resolve()),
            "sha256": _sha256(row_path),
            "metadata": json.loads(metadata),
            "selected_row_count": int(len(support_candidate_rows)),
            "selected_support_nnz": int(selected_matrix.nnz),
            "maximum_condition_number": float(np.max(diagnostics["condition"][support_candidate_rows])),
            "maximum_l1_norm": float(np.max(diagnostics["l1_norm"][support_candidate_rows])),
            "maximum_reproduction_residual": float(np.max(diagnostics["reproduction_residual"][support_candidate_rows])),
            "donor_count_range": [
                int(np.min(diagnostics["donor_counts"][support_candidate_rows])),
                int(np.max(diagnostics["donor_counts"][support_candidate_rows])),
            ],
        }

    # Batch independent exact values and gradients at every noncollapsed face
    # in the complete selected support, including eta faces.
    exact_started = time.perf_counter()
    face_points = np.asarray([
        oracle._face_points_weights(model.geometry, key, 1)[0][0]
        for key in face_keys
    ], dtype=np.float64)
    face_weights = np.asarray([
        oracle._face_points_weights(model.geometry, key, 1)[1][0]
        for key in face_keys
    ], dtype=np.float64)
    regular_mask = bridge._regular_reference_face_mask(model, face_keys)
    regular_points = face_points[regular_mask]
    prepared = reference.prepare(regular_points)
    raw_fields = reference._fields_raw(regular_points, float(args.time))
    phi_value = np.zeros(len(face_keys), dtype=np.float64)
    phi_value[regular_mask] = np.asarray(raw_fields["phi"][0], dtype=np.float64)
    phi_gradient = np.zeros((len(face_keys), 3), dtype=np.float64)
    phi_gradient[regular_mask] = np.stack(raw_fields["phi"][1:4], axis=-1)
    default_step = float(reference.finite_difference_step)
    omega_gradient, _ = reference._local_omega_derivatives(regular_points)
    sensitivity = {}
    for step in (0.5 * default_step, 2.0 * default_step):
        reference.finite_difference_step = step
        alternate, _ = reference._local_omega_derivatives(regular_points)
        sensitivity[f"step_{step:.1e}"] = {
            "relative_l2_difference": float(
                np.linalg.norm(alternate - omega_gradient)
                / max(np.linalg.norm(omega_gradient), np.finfo(float).tiny)
            ),
            "maximum_abs_difference": float(np.max(np.abs(alternate - omega_gradient))),
        }
    reference.finite_difference_step = default_step
    field_exact_values = {}
    field_exact_gradients = {}
    actual_value = np.zeros(len(face_keys), dtype=np.float64)
    actual_value[regular_mask] = np.asarray(prepared.mms_omega, dtype=np.float64)
    actual_gradient = np.zeros((len(face_keys), 3), dtype=np.float64)
    actual_gradient[regular_mask] = omega_gradient
    field_exact_values["actual_vorticity"] = actual_value
    field_exact_gradients["actual_vorticity"] = actual_gradient
    for name in ("smooth_regular_scalar", "smooth_eta_varying_scalar"):
        value, gradient = continuous._smooth_value_gradient(
            name, regular_points, reference.eta_period
        )
        full_value = np.zeros(len(face_keys), dtype=np.float64)
        full_value[regular_mask] = value
        full_gradient = np.zeros((len(face_keys), 3), dtype=np.float64)
        full_gradient[regular_mask] = gradient
        field_exact_values[name] = full_value
        field_exact_gradients[name] = full_gradient
    one_form = np.zeros((len(face_keys), 3), dtype=np.float64)
    one_form[regular_mask] = (
        np.asarray(prepared.bcov, dtype=np.float64)
        / np.maximum(np.asarray(prepared.B, dtype=np.float64)[:, None], 1.0e-30)
    )
    exact_seconds = time.perf_counter() - exact_started

    cells = model.control_volume_geometry.cells
    owner_arrays = bridge._owner_tuple_arrays(cells)
    aggregate_volume = np.asarray(cells.aggregate_volume, dtype=np.float64)
    members_by_owner = {
        owner: [
            tuple(int(v) for v in row)
            for row in np.argwhere(
                (owner_arrays[0] == owner[0])
                & (owner_arrays[1] == owner[1])
                & (owner_arrays[2] == owner[2])
            )
        ]
        for owner in selected_owners
    }
    category_by_owner = {
        tuple(record["owner"]): record["category"]
        for record in selection_payload["records"]
    }

    bridge_case = json.loads(
        (args.selection_root / f"N{resolution}.json").read_text(encoding="utf-8")
    )
    bridge_rows = {
        name: {
            tuple(record["owner"]): record
            for record in bridge_case["fields"][name]["owner_records"]
        }
        for name in FIELDS
    }

    fields_payload = {}
    action_started = time.perf_counter()
    for name in FIELDS:
        op = operands[name]
        state = field_inputs[name]
        baseline_raw = base._evaluate_variants(model, op)["production_R"]
        baseline = {
            "upwind": -baseline_raw["pure_upwind"] / rho_star,
            "A": -baseline_raw["phi_centered"] / rho_star,
            "B": baseline_raw["omega_centered"] / rho_star,
            "C": -baseline_raw["centered"] / rho_star,
        }
        runner = p04._global_action_runner(model, op)
        rule_actions = {"P": baseline}
        for rule in ("G", "O"):
            field_faces = p04._replace_faces_vectorized(
                op.omega_stencil.face_values, support_keys_array,
                candidate_values[rule][name],
            )
            phi_faces = p04._replace_faces_vectorized(
                op.phi_stencil.face_values, support_keys_array,
                candidate_values[rule]["phi"],
            )
            common_left = p04._replace_faces_vectorized(
                op.omega_direct_states[0], support_keys_array,
                candidate_values[rule][name],
            )
            common_right = p04._replace_faces_vectorized(
                op.omega_direct_states[1], support_keys_array,
                candidate_values[rule][name],
            )
            raw = runner(field_faces, phi_faces, common_left, common_right)
            rule_actions[rule] = {
                "upwind": -raw[0] / rho_star,
                "A": -raw[1] / rho_star,
                "B": raw[2] / rho_star,
                "C": -raw[3] / rho_star,
            }

        phi_generator = _compatible_flux_generator(
            op.phi_stencil, model.geometry, domain=model.domain,
            axis_regular_axes=model.axis_regular_axes, b_floor=1.0e-30,
        )
        field_generator = _compatible_flux_generator(
            op.omega_stencil, model.geometry, domain=model.domain,
            axis_regular_axes=model.axis_regular_axes, b_floor=1.0e-30,
        )
        phi_velocity = tuple(np.asarray(getattr(phi_generator, axis)) for axis in "xyz")
        field_velocity = tuple(np.asarray(getattr(field_generator, axis)) for axis in "xyz")
        production_field_faces = tuple(
            np.asarray(getattr(op.omega_stencil.face_values, axis)) for axis in "xyz"
        )
        production_phi_faces = tuple(
            np.asarray(getattr(op.phi_stencil.face_values, axis)) for axis in "xyz"
        )

        generators: dict[str, dict[str, dict[tuple[int, int, int, int], float]]] = {
            "A": {"P": {}, "exact": {}},
            "B": {"P": {}, "exact": {}},
        }
        arguments: dict[str, dict[str, dict[tuple[int, int, int, int], float]]] = {
            "A": {rule: {} for rule in (*RULES, "exact")},
            "B": {rule: {} for rule in (*RULES, "exact")},
        }
        candidate_maps = {
            rule: {
                field: _face_maps(support_candidate_keys, values)
                for field, values in candidate_values[rule].items()
            }
            for rule in ("G", "O")
        }
        for index, key in enumerate(face_keys):
            axis = key[0]
            area = float(face_weights[index])
            collapsed = not bool(regular_mask[index])
            v_phi_p = oracle._face_array_value(phi_velocity, key)
            v_field_p = oracle._face_array_value(field_velocity, key)
            v_phi_exact = 0.0 if collapsed else float(
                oracle._normal_velocity(
                    axis, one_form[index:index + 1], phi_gradient[index:index + 1]
                )[0]
            )
            v_field_exact = 0.0 if collapsed else float(
                oracle._normal_velocity(
                    axis, one_form[index:index + 1],
                    field_exact_gradients[name][index:index + 1],
                )[0]
            )
            generators["A"]["P"][key] = v_phi_p * area
            generators["A"]["exact"][key] = v_phi_exact * area
            generators["B"]["P"][key] = v_field_p * area
            generators["B"]["exact"][key] = v_field_exact * area
            p_field = oracle._face_array_value(production_field_faces, key)
            p_phi = oracle._face_array_value(production_phi_faces, key)
            arguments["A"]["P"][key] = p_field
            arguments["B"]["P"][key] = p_phi
            for rule in ("G", "O"):
                arguments["A"][rule][key] = candidate_maps[rule][name].get(key, p_field)
                arguments["B"][rule][key] = candidate_maps[rule]["phi"].get(key, p_phi)
            arguments["A"]["exact"][key] = float(field_exact_values[name][index])
            arguments["B"]["exact"][key] = float(phi_value[index])

        centers = {
            "A": {
                "P": np.asarray(op.omega_stencil.x.center, dtype=np.float64),
                "exact": np.asarray(state.vorticity, dtype=np.float64),
            },
            "B": {
                "P": np.asarray(op.phi_stencil.x.center, dtype=np.float64),
                "exact": np.asarray(state.phi, dtype=np.float64),
            },
        }
        config_contract = {
            "numerical": ("P", "rule", "P"),
            "exact_face": ("P", "exact", "P"),
            "exact_generator": ("exact", "rule", "P"),
            "both_exact": ("exact", "exact", "P"),
            "both_exact_reference_center": ("exact", "exact", "exact"),
            "center_only": ("P", "rule", "exact"),
        }

        owner_records = []
        for owner in selected_owners:
            row: dict[str, Any] = {
                "owner": list(owner),
                "category": category_by_owner[owner],
                "aggregate_volume": float(aggregate_volume[owner]),
                "raw_member_count": len(members_by_owner[owner]),
                "continuum_target": float(exact_owner[name][owner]),
            }
            components: dict[str, Any] = {}
            for rule in RULES:
                row[f"operator_upwind_{rule}"] = float(rule_actions[rule]["upwind"][owner])
                row[f"operator_A_{rule}"] = float(rule_actions[rule]["A"][owner])
                row[f"operator_B_{rule}"] = float(rule_actions[rule]["B"][owner])
                row[f"operator_C_{rule}"] = float(rule_actions[rule]["C"][owner])
                for config, (generator_name, argument_name, center_name) in config_contract.items():
                    values = {}
                    for action, sign in (("A", -1.0), ("B", 1.0)):
                        actual_argument = rule if argument_name == "rule" else argument_name
                        values[action] = _assemble_components(
                            owner,
                            members_by_owner[owner],
                            aggregate_volume,
                            centers[action][center_name],
                            generators[action][generator_name],
                            arguments[action][actual_argument],
                            sign=sign,
                            rho_star=rho_star,
                        )
                        row[f"cross_{action}_{rule}_{config}"] = values[action]["value"]
                    row[f"cross_C_{rule}_{config}"] = 0.5 * (
                        values["A"]["value"] + values["B"]["value"]
                    )
                    components[f"{rule}_{config}"] = {
                        "A": values["A"],
                        "B": values["B"],
                        "C": {
                            "value": row[f"cross_C_{rule}_{config}"],
                            "face_by_orientation": {
                                axis: 0.5 * (
                                    values["A"]["face_by_orientation"][axis]
                                    + values["B"]["face_by_orientation"][axis]
                                ) for axis in AXES
                            },
                            "center_by_orientation": {
                                axis: 0.5 * (
                                    values["A"]["center_by_orientation"][axis]
                                    + values["B"]["center_by_orientation"][axis]
                                ) for axis in AXES
                            },
                        },
                    }
            row["components"] = components
            owner_records.append(row)

        # Isolate the characteristic mean and jump on the exact same selected
        # planar faces.  The production jump orientation is right-left.
        production_mean, production_jump = _mean_jump(
            op.omega_direct_states[0], op.omega_direct_states[1], support_keys_array
        )
        exact_mean = np.asarray([
            field_exact_values[name][face_keys.index(key)]
            for key in support_candidate_keys
        ], dtype=np.float64)
        state_pairs = {"production": op.omega_direct_states}
        algebra = {}
        for rule in ("G", "O"):
            candidate_mean = candidate_values[rule][name]
            state_pairs[f"common_{rule}"] = _states_from_mean_jump(
                *op.omega_direct_states, support_keys_array,
                candidate_mean, np.zeros_like(production_jump),
            )
            state_pairs[f"retained_jump_{rule}"] = _states_from_mean_jump(
                *op.omega_direct_states, support_keys_array,
                candidate_mean, production_jump,
            )
            state_pairs[f"exact_mean_retained_jump_{rule}"] = _states_from_mean_jump(
                *op.omega_direct_states, support_keys_array,
                exact_mean, production_jump,
            )
        state_pairs["production_mean_zero_jump"] = _states_from_mean_jump(
            *op.omega_direct_states, support_keys_array,
            production_mean, np.zeros_like(production_jump),
        )
        upwind_actions = {}
        production_field_face_values = op.omega_stencil.face_values
        production_phi_face_values = op.phi_stencil.face_values
        for config, (left, right) in state_pairs.items():
            raw = runner(
                production_field_face_values, production_phi_face_values, left, right
            )
            upwind_actions[config] = -raw[0] / rho_star
            actual_mean, actual_jump = _mean_jump(left, right, support_keys_array)
            if config == "production":
                target_mean, target_jump = production_mean, production_jump
            elif config == "production_mean_zero_jump":
                target_mean, target_jump = production_mean, np.zeros_like(production_jump)
            elif config.startswith("common_"):
                target_mean = candidate_values[config[-1]][name]
                target_jump = np.zeros_like(production_jump)
            elif config.startswith("retained_jump_"):
                target_mean = candidate_values[config[-1]][name]
                target_jump = production_jump
            else:
                target_mean, target_jump = exact_mean, production_jump
            algebra[config] = {
                "maximum_mean_abs_error": float(np.max(np.abs(actual_mean - target_mean))),
                "maximum_jump_abs_error": float(np.max(np.abs(actual_jump - target_jump))),
            }
        for row in owner_records:
            owner = tuple(row["owner"])
            for config, action in upwind_actions.items():
                row[f"upwind_jump_{config}"] = float(action[owner])

        statistics = {
            "crosses": {
                action: {
                    rule: {
                        config: _selected_statistics(
                            owner_records, f"cross_{action}_{rule}_{config}"
                        )
                        for config in CONFIGURATIONS
                    }
                    for rule in RULES
                }
                for action in ("A", "B", "C")
            },
            "upwind_jump": {
                config: _selected_statistics(owner_records, f"upwind_jump_{config}")
                for config in state_pairs
            },
        }
        replay = {
            "operator_vs_explicit_numerical_max_abs": {
                action: {
                    rule: float(max(
                        abs(
                            row[f"operator_{action}_{rule}"]
                            - row[f"cross_{action}_{rule}_numerical"]
                        ) for row in owner_records
                    ))
                    for rule in RULES
                }
                for action in ("A", "B", "C")
            },
            "bridge_candidate_max_abs": {
                action: {
                    rule: float(max(
                        abs(
                            row[f"operator_{action}_{rule}"]
                            - float(bridge_rows[name][tuple(row["owner"])][
                                ({"upwind": "upwind", "A": "A", "B": "B"}[action]
                                 + f"_{rule}_all")
                            ])
                        ) for row in owner_records
                    ))
                    for rule in ("G", "O")
                }
                for action in ("upwind", "A", "B")
            },
        }
        fields_payload[name] = {
            "owner_records": owner_records,
            "statistics": statistics,
            "replay": replay,
            "upwind_mean_jump_algebra": algebra,
            "upwind_contract": {
                "mean": "0.5*(left+right)",
                "jump": "right-left",
                "retained_jump_states": "left=candidate_mean-jump/2; right=candidate_mean+jump/2",
                "fixed": [
                    "production generator velocity", "production center correction",
                    "eta states", "physical boundary traces", "compact-face states",
                ],
                "interpretation": "diagnostic flux decomposition only; not an endorsed reconstruction or stability result",
            },
        }
        del runner, rule_actions, baseline_raw
        gc.collect()
        _progress("field_complete", resolution=resolution, field=name, max_rss_gib=_max_rss_gib())

    selection_records = []
    for record in selection_payload["records"]:
        owner = tuple(record["owner"])
        selection_records.append({
            **record,
            "aggregate_volume": float(aggregate_volume[owner]),
            "raw_member_count": len(members_by_owner[owner]),
        })
    regional_context = {}
    for field in FIELDS:
        regional_context[field] = {}
        for operator, case_key in (("scalar_upwind", "scalar_upwind"), ("centered_C", "centered_C")):
            regional_context[field][operator] = {}
            for rule in RULES:
                action_key = rule if operator == "scalar_upwind" else f"{rule}_{rule}"
                regional_context[field][operator][rule] = {
                    region: p04_case["fields"][field]["actions"][case_key][action_key]["regions"][region]["squared_error_fraction"]
                    for region in ("agglomerated_interior", "ordinary", "true_size_change_interface", "axis_ring")
                }

    payload = {
        "schema": SCHEMA,
        "resolution": resolution,
        "time": float(args.time),
        "geometry": {
            "artifact": str(artifact_path.resolve()),
            "manifest_sha256": manifest_hash,
        },
        "reference": {
            "sidecar": str(args.reference_sidecar.resolve()),
            "sidecar_sha256": _sha256(args.reference_sidecar),
            "cache": str(cache_path.resolve()),
            "cache_sha256": _sha256(cache_path),
            "analytic_mms_eta_period": reference.eta_period,
            "continuous_evaluator_period": reference.metric_evaluator.period,
            "omega_gradient_sensitivity": sensitivity,
        },
        "boundary_contract": p04.BOUNDARY_CONTRACT,
        "selection": {
            "source": str(selection_path.resolve()),
            "source_sha256": _sha256(selection_path),
            "frozen_before_new_counterfactuals": True,
            "records": selection_records,
            "category_counts": {
                category: sum(row["category"] == category for row in selection_records)
                for category in sorted({row["category"] for row in selection_records})
            },
            "regional_context": regional_context,
        },
        "support": {
            "selected_owner_count": len(selected_owners),
            "raw_member_count": len(raw_labels),
            "incident_face_count": len(face_keys),
            "candidate_planar_face_count": len(support_candidate_keys),
            "complete_incident_face_support": True,
            "saved_rows": row_evidence,
        },
        "cross_definitions": {
            "numerical": "production generator, rule-specific P/G/O face value, production stored center",
            "exact_face": "production generator and matching production center correction, exact continuous midpoint advected face value",
            "exact_generator": "exact continuous generator normal velocity in weighted and center-divergence terms, rule-specific face value, production stored center",
            "both_exact": "exact generator and exact face value, production stored center",
            "both_exact_reference_center": "exact generator, exact face value, exact midpoint raw center",
            "center_only": "production generator and rule-specific face value, exact midpoint raw center",
            "quadrature": "same coordinate-face midpoint/logical-face measure used by the P04 diagnostic",
            "completed_action": "all incident radial/angular/eta faces of every raw member plus matched center/compression term",
        },
        "fields": fields_payload,
        "scope": {
            "certification": "bounded failure localization only; sample gains are not global convergence",
            "production_changes": [],
            "excluded": [
                "new global campaign", "production repair", "degree increase",
                "curvature", "diffusion/polarization", "evolved MMS",
                "sharding qualification", "blob synchronization",
            ],
        },
        "resources": {
            "maximum_rss_gib": _max_rss_gib(),
        },
        "timings_seconds": {
            "runtime_setup": runtime_seconds,
            "exact_face_batch": exact_seconds,
            "actions_and_crosses": time.perf_counter() - action_started,
            "total_before_output": time.perf_counter() - started,
        },
    }
    output = output_dir / f"N{resolution}.json"
    _write_json(output, payload)
    _progress("case_complete", resolution=resolution, output=str(output), seconds=time.perf_counter() - started)
    return payload


def _merge(args: argparse.Namespace) -> dict[str, Any]:
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    cases.sort(key=lambda value: int(value["resolution"]))
    if [case["resolution"] for case in cases] != [48, 64]:
        raise ValueError("localization merge requires N48 and N64")
    prior_summary = json.loads((args.selection_root / "summary.json").read_text(encoding="utf-8"))
    counterparts = prior_summary.get("nearest_selected_counterparts", [])

    comparison = {}
    for field in FIELDS:
        comparison[field] = {"crosses": {}, "upwind_jump": {}}
        for action in ("A", "B", "C"):
            comparison[field]["crosses"][action] = {}
            for rule in RULES:
                comparison[field]["crosses"][action][rule] = {}
                for config in CONFIGURATIONS:
                    values = [
                        case["fields"][field]["statistics"]["crosses"][action][rule][config]["all"]["absolute_l2"]
                        for case in cases
                    ]
                    comparison[field]["crosses"][action][rule][config] = values
        configs = cases[0]["fields"][field]["statistics"]["upwind_jump"].keys()
        for config in configs:
            comparison[field]["upwind_jump"][config] = [
                case["fields"][field]["statistics"]["upwind_jump"][config]["all"]["absolute_l2"]
                for case in cases
            ]

    payload = {
        "schema": SUMMARY_SCHEMA,
        "resolutions": [48, 64],
        "cases": [str(path.resolve()) for path in args.inputs],
        "comparison": comparison,
        "nearest_selected_counterparts": counterparts,
        "production_changes": [],
        "interpretation": (
            "bounded selected-owner stage decomposition; values are sample diagnostics "
            "and do not establish global convergence or stability"
        ),
    }
    _write_json(args.output, payload)

    lines = [
        "# Bounded P03 bracket-failure localization",
        "",
        "This N48/N64 diagnostic reuses the frozen eight-owner selections, qualified continuous reference, and saved P04 degree-two rows. It is not a new global campaign or production repair.",
        "",
        "## Selected-sample L2 errors",
        "",
        "| Field | Action | Rule | Numerical N48/N64 | Exact face N48/N64 | Exact generator N48/N64 | Both exact + production center N48/N64 | Both exact + reference center N48/N64 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for field in FIELDS:
        for action in ("A", "B", "C"):
            for rule in RULES:
                values = comparison[field]["crosses"][action][rule]
                fmt = lambda xs: "/".join(f"{value:.6g}" for value in xs)
                lines.append(
                    f"| {field} | {action} | {rule} | {fmt(values['numerical'])} | "
                    f"{fmt(values['exact_face'])} | {fmt(values['exact_generator'])} | "
                    f"{fmt(values['both_exact'])} | {fmt(values['both_exact_reference_center'])} |"
                )
    lines.extend([
        "",
        "## Upwind jump isolation",
        "",
        "| Field | Configuration | N48/N64 selected-sample L2 |",
        "|---|---|---:|",
    ])
    for field in FIELDS:
        for config, values in comparison[field]["upwind_jump"].items():
            lines.append(
                f"| {field} | {config} | "
                + "/".join(f"{value:.6g}" for value in values) + " |"
            )
    lines.extend([
        "",
        "## Scope",
        "",
        "All comparisons include every incident face of every raw member of each selected owner and the matching center/compression term. Exact substitutions are stage controls, not redefinitions of stored unknowns. Local sample changes and apparent local orders are diagnostic only.",
        "",
    ])
    _write_text(args.output.with_name("report.md"), "\n".join(lines))
    print(args.output)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    case = sub.add_parser("case")
    case.add_argument("--geometry", type=Path, required=True)
    case.add_argument("--reference-sidecar", type=Path, required=True)
    case.add_argument("--baseline", type=Path, required=True)
    case.add_argument("--selection-root", type=Path, required=True)
    case.add_argument("--p04-root", type=Path, required=True)
    case.add_argument("--resolution", type=int, choices=(48, 64), required=True)
    case.add_argument("--time", type=float, default=1.0e-6)
    case.add_argument("--output", type=Path, required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("inputs", nargs=2, type=Path)
    merge.add_argument("--selection-root", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "case":
        _case(args)
    else:
        _merge(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
