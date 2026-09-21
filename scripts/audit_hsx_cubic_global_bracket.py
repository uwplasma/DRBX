#!/usr/bin/env python3
"""Global static HSX qualification of the fixed balanced-cubic derivative.

The derivative candidate is applied consistently to the generator stencil in
both face-flux and matching compatible center terms.  P/G/O retain their
established transported-value meanings; in particular G/O use common planar
face values and therefore do not certify material upwind stabilization.
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
from drbx.native.fci_boundaries import FaceGradientStencil3D  # noqa: E402

import audit_hsx_cubic_derivative_global as cubic  # noqa: E402
import audit_hsx_face_candidate_orientation as orientation  # noqa: E402
import audit_hsx_p03_fine_interval as bridge  # noqa: E402
import audit_hsx_p04_global_bracket as p04  # noqa: E402
import audit_hsx_poisson_common_face_oracle as oracle  # noqa: E402
import audit_hsx_poisson_continuous_baseline as continuous  # noqa: E402
import audit_hsx_poisson_vorticity as base  # noqa: E402


SCHEMA = "drbx.hsx-balanced-cubic-global-bracket-case-v2"
SUMMARY_SCHEMA = "drbx.hsx-balanced-cubic-global-bracket-summary-v2"
FIELDS = continuous.FIELDS
RULES = ("P", "G", "O")
BOUNDARY_CONTRACT = p04.BOUNDARY_CONTRACT


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
    print(json.dumps({"event": "cubic_bracket", "stage": stage, **details}, sort_keys=True), flush=True)


def _max_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / 1024.0**3


def _load_p04_matrix(path: Path) -> tuple[csr_matrix, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata"].item()))
        shape = tuple(int(value) for value in source["shape"])
        matrix = csr_matrix(
            (source["data"], source["indices"], source["indptr"]), shape=shape
        )
    return matrix, metadata


def _transported_values(
    p04_root: Path,
    resolution: int,
    artifact: Any,
    model: Any,
    owner_states: Mapping[str, Any],
    keys: np.ndarray,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
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
    candidate: dict[str, dict[str, np.ndarray]] = {}
    provenance = {}
    for rule in ("G", "O"):
        path = p04_root / f"N{resolution}_{rule}_rows.npz"
        matrix, metadata = _load_p04_matrix(path)
        if matrix.shape[0] != len(keys):
            raise ValueError(f"{rule} transported-value row count does not match face keys")
        if metadata["face_key_sha256"] != p04._array_sha256(keys):
            raise ValueError(f"{rule} transported-value face ordering changed")
        aggregate = aggregates[rule]
        candidate[rule] = {
            name: np.asarray(
                matrix.dot(np.asarray(values).reshape(-1)[aggregate.owner_flat_ids])
            ).reshape(-1)
            for name, values in owner_values.items()
        }
        provenance[rule] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "metadata": metadata,
        }
    return candidate, provenance


def _gradient_stencil(
    values: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> FaceGradientStencil3D:
    return FaceGradientStencil3D(x=values[0], y=values[1], z=values[2])


def _hybrid_gradient(
    original: FaceGradientStencil3D,
    candidate: FaceGradientStencil3D,
    axis: int | None,
) -> FaceGradientStencil3D:
    if axis is None:
        return candidate
    values = []
    for component, name in enumerate("xyz"):
        values.append(getattr(candidate if component == axis else original, name))
    return FaceGradientStencil3D(*values)


def _runner(model: Any, operands: Any):
    """Return one JIT applying supplied values and one canonical gradient."""

    def inner(field_faces, phi_faces, field_gradient, phi_gradient, left, right):
        field_stencil = replace(
            operands.omega_stencil,
            face_values=field_faces,
            face_grad=field_gradient,
        )
        phi_stencil = replace(
            operands.phi_stencil,
            face_values=phi_faces,
            face_grad=phi_gradient,
        )
        upwind = base._operator_call(
            model,
            phi_stencil,
            field_stencil,
            f_trace=operands.phi_trace,
            g_trace=operands.omega_trace,
            characteristic_scheme="scalar-third-order-upwind",
            g_halo=operands.omega_halo,
            g_direct_states=(left, right),
        )
        action_a = base._operator_call(
            model,
            phi_stencil,
            field_stencil,
            f_trace=operands.phi_trace,
            g_trace=operands.omega_trace,
            characteristic_scheme="scalar-centered",
        )
        action_b = base._operator_call(
            model,
            field_stencil,
            phi_stencil,
            f_trace=operands.omega_trace,
            g_trace=operands.phi_trace,
            characteristic_scheme="scalar-centered",
        )
        centered = base._operator_call(
            model,
            phi_stencil,
            field_stencil,
            f_trace=operands.phi_trace,
            g_trace=operands.omega_trace,
            characteristic_scheme="centered",
        )
        swapped = base._operator_call(
            model,
            field_stencil,
            phi_stencil,
            f_trace=operands.omega_trace,
            g_trace=operands.phi_trace,
            characteristic_scheme="centered",
        )
        return base._return_actions_to_owner(
            model, (upwind, action_a, action_b, centered, swapped)
        )[0]

    compiled = jax.jit(inner)

    def numeric(*arguments: Any) -> dict[str, np.ndarray]:
        raw = jax.block_until_ready(compiled(*arguments))
        raw = tuple(np.asarray(value) for value in raw)
        rho_star = float(model.parameters.rho_star)
        result = {
            "upwind": -raw[0] / rho_star,
            "A": -raw[1] / rho_star,
            "B": raw[2] / rho_star,
            "C_direct": -raw[3] / rho_star,
            "swapped": -raw[4] / rho_star,
        }
        result["C"] = 0.5 * (result["A"] + result["B"])
        return result

    return numeric


def _rule_arguments(
    op: Any,
    keys: np.ndarray,
    candidate_values: Mapping[str, Mapping[str, np.ndarray]],
    rule: str,
    field: str,
) -> tuple[Any, Any, Any, Any]:
    if rule == "P":
        return (
            op.omega_stencil.face_values,
            op.phi_stencil.face_values,
            op.omega_direct_states[0],
            op.omega_direct_states[1],
        )
    field_faces = p04._replace_faces_vectorized(
        op.omega_stencil.face_values, keys, candidate_values[rule][field]
    )
    phi_faces = p04._replace_faces_vectorized(
        op.phi_stencil.face_values, keys, candidate_values[rule]["phi"]
    )
    left = p04._replace_faces_vectorized(
        op.omega_direct_states[0], keys, candidate_values[rule][field]
    )
    right = p04._replace_faces_vectorized(
        op.omega_direct_states[1], keys, candidate_values[rule][field]
    )
    return field_faces, phi_faces, left, right


def _action_statistics(
    actions: Mapping[str, np.ndarray],
    exact: np.ndarray,
    host: Any,
    masks: Mapping[str, np.ndarray],
    ordinary_layer: np.ndarray,
) -> dict[str, Any]:
    active = np.asarray(host.topology.is_active_owner, dtype=bool)
    volume = np.asarray(host.aggregate_chart_volume, dtype=np.float64)
    inner_ring = np.zeros_like(active)
    if active.shape[0] > 1:
        inner_ring[1] = active[1]
    payload = {}
    for action, values in actions.items():
        if action not in ("upwind", "A", "B", "C"):
            continue
        item = bridge._regional_statistics(
            values, exact, host, masks, ordinary_layer
        )
        absolute_error = np.abs(np.asarray(values) - np.asarray(exact))
        item["global"]["maximum_absolute_error"] = float(
            np.max(absolute_error[active])
        )
        for name, mask in masks.items():
            item["regions"][name]["maximum_absolute_error"] = float(
                np.max(absolute_error[active & mask])
            )
        inner = base._weighted_statistics(values, exact, volume, inner_ring)
        inner["squared_error_fraction"] = float(
            inner["squared_error"] / max(item["global"]["squared_error"], 1.0e-300)
        )
        inner["volume_fraction"] = float(
            inner["volume"] / max(item["global"]["volume"], 1.0e-300)
        )
        inner["maximum_absolute_error"] = float(
            np.max(absolute_error[inner_ring])
        )
        item["inner_ring_diagnostic"] = inner
        payload[action] = item
    return payload


def _case(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    resolution = int(args.resolution)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path, artifact = base._load_artifact(args.geometry, resolution)
    _, identity_artifact = base._load_artifact(args.geometry, 64)
    manifest_hash = _sha256(artifact_path / "manifest.json")
    reference_cache = args.baseline / f"N{resolution}.reference.npz"
    raw_state, regular_raw, eta_raw, exact_raw, cache_provenance = p04._load_cache(
        reference_cache, resolution, float(args.time)
    )
    if cache_provenance["geometry_manifest_sha256"] != manifest_hash:
        raise ValueError("qualified baseline and selected geometry do not match")
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
        raise RuntimeError("global cubic bracket audit requires one device")
    runtime_seconds = time.perf_counter() - runtime_started
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
    keys, _points, _eta = p04._candidate_face_geometry(
        model, operands["actual_vorticity"]
    )
    candidate_values, value_provenance = _transported_values(
        args.p04_root, resolution, artifact, model, owner_states, keys
    )
    gradient_root = args.derivative_root / f"N{resolution}"
    gradient_arrays, derivative_manifest = cubic.load_global_gradients(gradient_root)
    derivative_identity = derivative_manifest["identity"]
    if derivative_identity["geometry"]["manifest"]["sha256"] != manifest_hash:
        raise ValueError("derivative cache and geometry manifest mismatch")
    if derivative_identity["field_data"]["baseline_reference"]["sha256"] != _sha256(reference_cache):
        raise ValueError("derivative cache and manufactured fields mismatch")
    candidate_gradient = {
        field: _gradient_stencil(gradient_arrays[field]) for field in cubic.FIELD_NAMES
    }
    standard = base._region_masks(
        artifact.global_geometry, artifact.owner_geometry
    )
    masks, ordinary_layer, adjacency = bridge._refined_masks(
        artifact.owner_geometry, model.control_volume_geometry.cells, standard
    )
    exact_owner = {
        name: mms._owner_project_array(exact_raw[name], artifact.owner_geometry)
        for name in FIELDS
    }
    prior_case = json.loads((args.p04_root / f"N{resolution}.json").read_text(encoding="utf-8"))
    fields_payload = {}
    action_started = time.perf_counter()
    for field in FIELDS:
        field_started = time.perf_counter()
        op = operands[field]
        runner = _runner(model, op)
        field_gradient = candidate_gradient[field]
        phi_gradient = candidate_gradient["phi"]
        candidate_actions = {}
        structural = {}
        for rule in RULES:
            field_faces, phi_faces, left, right = _rule_arguments(
                op, keys, candidate_values, rule, field
            )
            actions = runner(
                field_faces, phi_faces, field_gradient, phi_gradient, left, right
            )
            candidate_actions[rule] = actions
            structural[rule] = {
                "centered_decomposition_max_abs": float(
                    np.max(np.abs(actions["C_direct"] - actions["C"]))
                ),
                "swapped_antisymmetry_max_abs": float(
                    np.max(np.abs(actions["C_direct"] + actions["swapped"]))
                ),
            }
        # Re-evaluate O with the old derivative and with one candidate
        # coordinate-family at a time.  This is both the old-action replay and
        # an incidence/center-term linearity check on the completed operator.
        field_faces, phi_faces, left, right = _rule_arguments(
            op, keys, candidate_values, "O", field
        )
        old_o = runner(
            field_faces,
            phi_faces,
            op.omega_stencil.face_grad,
            op.phi_stencil.face_grad,
            left,
            right,
        )
        axis_o = {}
        for axis in range(3):
            axis_o[axis] = runner(
                field_faces,
                phi_faces,
                _hybrid_gradient(op.omega_stencil.face_grad, field_gradient, axis),
                _hybrid_gradient(op.phi_stencil.face_grad, phi_gradient, axis),
                left,
                right,
            )
        directional_closure = {
            action: float(np.max(np.abs(
                (candidate_actions["O"][action] - old_o[action])
                - sum(axis_o[axis][action] - old_o[action] for axis in range(3))
            )))
            for action in ("upwind", "A", "B", "C")
        }
        old_expected = {
            action: float(
                prior_case["fields"][field]["actions"][
                    "scalar_upwind" if action == "upwind" else
                    f"centered_{action}"
                ]["O" if action != "C" else "O_O"]["global"]["absolute_l2"]
            )
            for action in ("upwind", "A", "B", "C")
        }
        old_statistics = _action_statistics(
            old_o, exact_owner[field], artifact.owner_geometry, masks, ordinary_layer
        )
        fields_payload[field] = {
            "actions": {
                rule: _action_statistics(
                    candidate_actions[rule], exact_owner[field],
                    artifact.owner_geometry, masks, ordinary_layer,
                )
                for rule in RULES
            },
            "old_derivative_O_control": old_statistics,
            "old_action_replay": {
                action: abs(old_statistics[action]["global"]["absolute_l2"] - old_expected[action])
                for action in old_expected
            },
            "structural": structural,
            "directional_completed_action_closure": directional_closure,
            "timings_seconds": {"total": time.perf_counter() - field_started},
        }
        _progress(
            "field_complete", resolution=resolution, field=field,
            seconds=fields_payload[field]["timings_seconds"]["total"],
            max_rss_gib=_max_rss_gib(),
        )
        del runner, candidate_actions, axis_o, old_o
        gc.collect()
    z_periodic = {
        field: float(np.max(np.abs(
            gradient_arrays[field][2][:, :, 0] - gradient_arrays[field][2][:, :, -1]
        )))
        for field in cubic.FIELD_NAMES
    }
    payload = {
        "schema": SCHEMA,
        "resolution": resolution,
        "time": float(args.time),
        "geometry": {
            "artifact": str(artifact_path.resolve()),
            "manifest_sha256": manifest_hash,
            "adjacent_owner_pair_count": len(adjacency),
        },
        "reference": {
            "sidecar": str(args.reference_sidecar.resolve()),
            "sidecar_sha256": _sha256(args.reference_sidecar),
            "cache": str(reference_cache.resolve()),
            "cache_sha256": _sha256(reference_cache),
            "provenance": reference.provenance,
        },
        "boundary_contract": BOUNDARY_CONTRACT,
        "candidate": {
            "derivative": cubic.POLICY,
            "one_canonical_derivative_for_face_and_center_terms": True,
            "transported_values": {
                "P": "existing production values and left/right traces",
                "G": "degree-two geometric-moment common planar value",
                "O": "degree-two matched midpoint-observation common planar value",
            },
            "material_upwind_clarification": "G/O share one left/right value and bypass production jump stabilization; centered results do not certify material upwinding",
        },
        "derivative_cache": {
            "root": str(gradient_root.resolve()),
            "manifest_sha256": _sha256(gradient_root / "manifest.json"),
            "manifest": derivative_manifest,
        },
        "transported_value_rows": value_provenance,
        "fields": fields_payload,
        "regional_masks": {
            name: {
                "owner_count": int(np.count_nonzero(mask)),
                "physical_volume": float(np.sum(
                    np.asarray(artifact.owner_geometry.aggregate_chart_volume)[mask]
                )),
            }
            for name, mask in masks.items()
        },
        "structural": {
            "periodic_eta_gradient_max_abs": z_periodic,
            "collapsed_axis_flux_contract": "lower radial flux remains forced to exact zero by compatible generator assembly",
            "constant_derivative_max_abs": derivative_manifest["statistics"]["maximum_constant_derivative_residual"],
            "maximum_polynomial_reproduction_residual": derivative_manifest["statistics"]["maximum_reproduction_residual"],
            "shared_face_contract": "one canonical coordinate-face derivative array is consumed by every incident raw member",
        },
        "timings_seconds": {
            "runtime_setup": runtime_seconds,
            "actions": time.perf_counter() - action_started,
            "total_before_output": time.perf_counter() - started,
        },
        "resources": {
            "maximum_rss_gib": _max_rss_gib(),
            "derivative_cache_statistics": derivative_manifest["statistics"],
        },
        "scope": {
            "production_changes": [],
            "certifies_material_upwinding": False,
            "excluded": [
                "production promotion", "evolved MMS", "curvature",
                "polarization", "diffusion", "blob driver",
            ],
        },
    }
    output = output_dir / f"N{resolution}.json"
    _write_json(output, payload)
    _progress("case_complete", resolution=resolution, output=str(output), seconds=time.perf_counter() - started)
    return payload


def _orders(resolutions: Sequence[int], errors: Sequence[float]) -> list[float | None]:
    result = []
    for n0, n1, left, right in zip(resolutions[:-1], resolutions[1:], errors[:-1], errors[1:]):
        result.append(
            None if left <= 0.0 or right <= 0.0
            else float(math.log(left / right) / math.log(n1 / n0))
        )
    return result


def _series(cases: Sequence[Mapping[str, Any]], field: str, rule: str, action: str) -> dict[str, Any]:
    errors = [
        float(case["fields"][field]["actions"][rule][action]["global"]["absolute_l2"])
        for case in cases
    ]
    relative = [
        float(case["fields"][field]["actions"][rule][action]["global"]["relative_l2"])
        for case in cases
    ]
    orders = _orders([int(case["resolution"]) for case in cases], errors)
    return {
        "absolute_l2": errors,
        "relative_l2": relative,
        "absolute_orders": orders,
        "accepted": bool(len(orders) == 2 and all(value is not None and value >= 1.8 for value in orders)),
        "minimum_order": 1.8,
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    cases = summary["cases"]
    lines = [
        "# Global HSX qualification of the balanced-cubic generator derivative",
        "",
        "The fixed cubic derivative is used consistently in face and matching-center generator terms. O/centered-C is the primary lane; A and sign-corrected B are retained separately. P/G/O name transported-value rules, not derivative variants.",
        "",
        "## Global operator results",
        "",
        "| Field | Action | Rule | absolute L2 N32 / N48 / N64 | orders 32→48 / 48→64 | gate |",
        "|---|---|---|---:|---:|---|",
    ]
    for field in FIELDS:
        for action in ("A", "B", "C", "upwind"):
            for rule in RULES:
                item = summary["comparison"][field][action][rule]
                errors = " / ".join(f"{value:.7g}" for value in item["absolute_l2"])
                orders = " / ".join("—" if value is None else f"{value:.4f}" for value in item["absolute_orders"])
                lines.append(
                    f"| {field} | {action} | {rule} | {errors} | {orders} | {'pass' if item['accepted'] else 'fail'} |"
                )
    lines.extend([
        "",
        "## Support, conditioning, and cost",
        "",
        "The base policy is 24 donors from the nearest-32 sector-balanced pool on each of five radial/angular or six eta planes. Rows that fail rank, reproduction, or three-column coverage use the fixed 32/32, 40/48, then 48/64 geometry-only fallback schedule at every resolution; no MMS value or error enters selection.",
        "",
        "| N | active faces | fallback rows (%) | levels 1 / 2 / 3 | max condition | max amplification x / y / eta | max radius | build / case (s) | peak RSS build / case (GiB) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for case in cases:
        stats = case["resources"]["derivative_cache_statistics"]
        levels = stats["fallback_level_counts"]
        lines.append(
            f"| {case['resolution']} | {stats['regular_row_count']:,} | "
            f"{stats['fallback_row_count']:,} ({100.0 * stats['fallback_fraction']:.2f}%) | "
            f"{levels['1']:,} / {levels['2']:,} / {levels['3']:,} | "
            f"{stats['maximum_condition']:.3g} | "
            f"{' / '.join(f'{value:.3g}' for value in stats['maximum_amplification'])} | "
            f"{stats['maximum_support_radius']:.3g} | "
            f"{stats['row_build_seconds']:.1f} / {case['timings_seconds']['total_before_output']:.1f} | "
            f"{stats['maximum_rss_gib']:.2f} / {case['resources']['maximum_rss_gib']:.2f} |"
        )
    lines.extend([
        "",
        "All rows retain at least three angular columns per participating eta plane. Maximum cubic reproduction and constant-derivative residuals across the ladder are "
        f"`{max(case['resources']['derivative_cache_statistics']['maximum_reproduction_residual'] for case in cases):.3e}` and "
        f"`{max(case['resources']['derivative_cache_statistics']['maximum_constant_derivative_residual'] for case in cases):.3e}`.",
        "",
        "## N64 regional O/centered-C squared-error budget",
        "",
        "| Field | axis | inner ring* | true RLP transition | agglomerated interior | ordinary | physical boundary | max abs error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    finest = cases[-1]
    region_order = (
        "axis_ring", "true_size_change_interface", "agglomerated_interior",
        "ordinary", "physical_boundary_footprint",
    )
    for field in FIELDS:
        item = finest["fields"][field]["actions"]["O"]["C"]
        fractions = [item["regions"][region]["squared_error_fraction"] for region in region_order]
        lines.append(
            f"| {field} | {100.0 * fractions[0]:.2f}% | "
            f"{100.0 * item['inner_ring_diagnostic']['squared_error_fraction']:.2f}% | "
            f"{' | '.join(f'{100.0 * value:.2f}%' for value in fractions[1:])} | "
            f"{item['global']['maximum_absolute_error']:.6g} |"
        )
    lines.extend([
        "",
        "*The inner-ring diagnostic overlaps the disjoint transition/agglomerated/ordinary/boundary partition; the other five percentages form the disjoint global squared-error budget.",
    ])
    replay_max = max(
        value
        for case in cases
        for field in case["fields"].values()
        for value in field["old_action_replay"].values()
    )
    decomposition_max = max(
        value
        for case in cases
        for field in case["fields"].values()
        for rule in field["structural"].values()
        for value in rule.values()
    )
    directional_max = max(
        value
        for case in cases
        for field in case["fields"].values()
        for value in field["directional_completed_action_closure"].values()
    )
    periodic_max = max(
        value
        for case in cases
        for value in case["structural"]["periodic_eta_gradient_max_abs"].values()
    )
    lines.extend([
        "",
        "## Replay and structural checks",
        "",
        f"Old O actions replay the qualified prior global cases to `{replay_max:.3e}` in L2. The largest centered decomposition or swapped-antisymmetry defect is `{decomposition_max:.3e}`; the completed radial/angular/eta face-plus-center closure is `{directional_max:.3e}`. Periodic eta gradient endpoints match to `{periodic_max:.3e}`. The collapsed-axis face flux remains exactly zero through the compatible generator contract, and every incident raw member consumes one canonical shared-face derivative.",
        "",
    ])
    supplements = summary.get("supplements", {})
    if supplements:
        lines.extend(["## Optimized-construction qualification", ""])
        profile = supplements.get("profile")
        if profile:
            lines.append(
                f"The stratified N{profile['resolution']} preflight measured {profile['measured_regular_rows']:,} real-HSX rows and estimated `{profile['estimated_global_row_build_seconds']:.1f}` seconds for that mesh before scaling."
            )
        for key in ("equivalence_N48", "equivalence_N64"):
            item = supplements.get(key)
            if item:
                lines.append(
                    f"N{item['resolution']} optimized equivalence compared {item['compared_regular_rows']} saved bounded rows: donor mismatches `{item['donor_mismatch_rows']}`, maximum weight difference `{item['maximum_weight_absolute_difference']:.3e}`, maximum field-action difference `{item['maximum_field_action_absolute_difference']:.3e}`."
                )
        lines.append("")
    lines.extend([
        "## Qualification decision",
        "",
        summary["decision"],
        "",
        "The common G/O transported traces bypass production upwind stabilization, so no centered result certifies material upwinding. This campaign makes no production, evolved-MMS, curvature, polarization, diffusion, or blob-driver claim.",
        "",
        "Regional squared-error budgets, maximum norms, support/conditioning/amplification diagnostics, fallback activity, structural checks, timings, memory, and complete provenance are retained in `summary.json` and the per-resolution case files.",
        "",
        "## Provenance and scope",
        "",
        f"Geometry manifest hashes: `{', '.join(case['geometry']['manifest_sha256'] for case in cases)}`. Reference-cache hashes: `{', '.join(case['reference']['cache_sha256'] for case in cases)}`. Boundary contract: legacy-velocity-trace, physical Neumann ghosts, characteristic-SAT pairing, energy-absorbing characteristic wall law, parallel-velocity Neumann, and periodic eta.",
        "",
    ])
    return "\n".join(lines)


def _merge(args: argparse.Namespace) -> dict[str, Any]:
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    cases.sort(key=lambda item: int(item["resolution"]))
    resolutions = [int(item["resolution"]) for item in cases]
    if resolutions != [32, 48, 64]:
        raise ValueError("cubic global merge requires N32/N48/N64")
    comparison = {
        field: {
            action: {
                rule: _series(cases, field, rule, action) for rule in RULES
            }
            for action in ("A", "B", "C", "upwind")
        }
        for field in FIELDS
    }
    primary_c = {
        field: comparison[field]["C"]["O"]["accepted"] for field in FIELDS
    }
    constituent = {
        field: {
            action: comparison[field][action]["O"]["accepted"]
            for action in ("A", "B")
        }
        for field in FIELDS
    }
    primary_pass = all(primary_c.values())
    constituent_pass = all(
        value for field in constituent.values() for value in field.values()
    )
    if primary_pass and constituent_pass:
        decision = (
            "O/centered-C and both constituent operands satisfy the global order gate for all three fields. This qualifies the tested static centered-bracket infrastructure only; P05 material upwinding and production integration remain separate work."
        )
    elif primary_pass:
        decision = (
            "O/centered-C satisfies the primary global order gate for all fields, but at least one A/B constituent fails. Preserve the cancellation-sensitive failure and do not mark the full bracket milestone passed."
        )
    else:
        decision = (
            "The fixed cubic candidate fails the primary O/centered-C global order gate for at least one field. Preserve the negative result; global bracket certification and production integration remain blocked on the reported mechanism."
        )
    supplements = {}
    for name in ("profile", "equivalence_N48", "equivalence_N64"):
        path = args.output.parent / f"{name}.json"
        if path.is_file():
            supplements[name] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "schema": SUMMARY_SCHEMA,
        "resolutions": resolutions,
        "comparison": comparison,
        "primary_O_centered_C": primary_c,
        "O_constituent_A_B": constituent,
        "primary_centered_C_pass": primary_pass,
        "constituent_A_B_pass": constituent_pass,
        "complete_tested_centered_infrastructure_pass": primary_pass and constituent_pass,
        "decision": decision,
        "cases": cases,
        "acceptance_contract": {
            "norm": "global physical-volume-weighted absolute operator L2",
            "minimum_order": 1.8,
            "required_intervals": ["32->48", "48->64"],
            "per_field_action": True,
            "regional_orders_are_diagnostic": True,
        },
        "production_changes": [],
        "certifies_material_upwinding": False,
        "supplements": supplements,
    }
    _write_json(args.output, payload)
    _write_text(args.output.with_name("report.md"), _render_report(payload))
    print(args.output, flush=True)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    case = sub.add_parser("case")
    case.add_argument("--geometry", type=Path, required=True)
    case.add_argument("--reference-sidecar", type=Path, required=True)
    case.add_argument("--baseline", type=Path, required=True)
    case.add_argument("--p04-root", type=Path, required=True)
    case.add_argument("--derivative-root", type=Path, required=True)
    case.add_argument("--resolution", type=int, choices=(32, 48, 64), required=True)
    case.add_argument("--time", type=float, default=1.0e-6)
    case.add_argument("--output", type=Path, required=True)
    case.set_defaults(handler=_case)
    merge = sub.add_parser("merge")
    merge.add_argument("inputs", nargs=3, type=Path)
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(handler=_merge)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
