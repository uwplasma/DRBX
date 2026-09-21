#!/usr/bin/env python3
"""Corrected-reference global HSX bracket baseline for P03.

Each resolution is evaluated in an isolated process.  Manufactured midpoint
inputs and independent continuum bracket sources use one qualified continuous
producer metric/B reference.  The production bracket, topology, restriction,
and frozen boundary selectors are not changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
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
from hsx_mms_continuum_reference import (  # noqa: E402
    build_continuum_reference_from_sidecar,
)
import audit_hsx_poisson_vorticity as base  # noqa: E402


SCHEMA = "drbx.hsx-poisson-continuous-baseline-v1"
FIELDS = (
    "actual_vorticity",
    "smooth_regular_scalar",
    "smooth_eta_varying_scalar",
)
VARIANTS = base.VARIANTS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(base._json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _progress(stage: str, **details: Any) -> None:
    print(
        json.dumps({"event": "progress", "stage": stage, **details}, sort_keys=True),
        flush=True,
    )


def _write_reference_cache(
    path: Path,
    *,
    states: Mapping[str, Any],
    exact_sources: Mapping[str, np.ndarray],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    arrays: dict[str, Any] = {
        "provenance_json": np.asarray(json.dumps(base._json_value(provenance), sort_keys=True))
    }
    actual = states["actual_vorticity"]
    for name in mms.FIELDS:
        arrays[f"actual_{name}"] = np.asarray(getattr(actual, name), dtype=np.float64)
    arrays["smooth_regular_scalar"] = np.asarray(
        states["smooth_regular_scalar"].vorticity, dtype=np.float64
    )
    arrays["smooth_eta_varying_scalar"] = np.asarray(
        states["smooth_eta_varying_scalar"].vorticity, dtype=np.float64
    )
    for name, value in exact_sources.items():
        arrays[f"exact_{name}"] = np.asarray(value, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size": path.stat().st_size,
        "schema": "drbx.hsx-poisson-midpoint-reference-cache-v1",
    }


def _smooth_value_gradient(
    name: str, points: np.ndarray, eta_period: float
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(points, dtype=np.float64)
    u, theta, eta = q.T
    a = np.maximum(1.0 - u**2, 0.0)
    k = 2.0 * np.pi / float(eta_period)
    if name == "smooth_regular_scalar":
        envelope = u**2 * a**4
        phase = 2.0 * theta - k * eta + 0.23
        value = 0.08 * envelope * np.cos(phase)
        gradient = np.stack(
            (
                0.16 * u * a**3 * (1.0 - 5.0 * u**2) * np.cos(phase),
                -0.16 * envelope * np.sin(phase),
                0.08 * k * envelope * np.sin(phase),
            ),
            axis=-1,
        )
        return value, gradient
    if name == "smooth_eta_varying_scalar":
        phase = k * eta + 0.31
        value = 0.08 * a**4 * np.sin(phase)
        gradient = np.stack(
            (
                -0.64 * u * a**3 * np.sin(phase),
                np.zeros_like(u),
                0.08 * k * a**4 * np.cos(phase),
            ),
            axis=-1,
        )
        return value, gradient
    raise ValueError(f"unsupported smooth field {name!r}")


def _project_field(
    projector: Any,
    time: float,
    *,
    rho_star: float,
    field: str,
) -> tuple[Any, np.ndarray]:
    cells = int(np.prod(projector.shape))
    values = {name: np.empty(cells, dtype=np.float64) for name in mms.FIELDS}
    exact = np.empty(cells, dtype=np.float64)
    for first, last, points, prepared, weighted in projector.chunks:
        data = projector.reference.evaluate(points, time, prepared=prepared)
        denominator = np.sum(weighted, axis=1)

        def project(array: Any) -> np.ndarray:
            shaped = np.asarray(array, dtype=np.float64).reshape(
                (last - first, projector.nq)
            )
            return np.sum(weighted * shaped, axis=1) / np.maximum(
                denominator, 1.0e-30
            )

        for name in mms.FIELDS:
            values[name][first:last] = project(data.values[name])
        if field == "actual_vorticity":
            scalar = np.asarray(data.values["vorticity"], dtype=np.float64)
            gradient = np.asarray(data.gradients["vorticity"], dtype=np.float64)
        else:
            scalar, gradient = _smooth_value_gradient(
                field, points, projector.reference.eta_period
            )
            values["vorticity"][first:last] = project(scalar)
        continuum = -np.sum(
            np.asarray(prepared.bcov, dtype=np.float64)
            * np.cross(
                np.asarray(data.gradients["phi"], dtype=np.float64), gradient
            ),
            axis=-1,
        )
        continuum /= (
            float(rho_star)
            * np.maximum(np.abs(np.asarray(prepared.J, dtype=np.float64)), 1.0e-30)
            * np.maximum(np.asarray(prepared.B, dtype=np.float64), 1.0e-30)
        )
        exact[first:last] = project(continuum)
    state = base.blob.FciDrbEBState(
        **{
            name: values[name].reshape(projector.shape)
            for name in mms.FIELDS
        }
    )
    return state, exact.reshape(projector.shape)


def _disjoint_accounting_masks(host: Any, standard: Mapping[str, np.ndarray]):
    active = np.asarray(host.topology.is_active_owner, dtype=bool)
    axis = np.zeros_like(active)
    axis[0] = active[0]
    boundary_raw = (
        np.asarray(standard["physical_wall"], dtype=bool)
        | np.asarray(standard["short_leg_topology_transition"], dtype=bool)
        | np.asarray(standard["double_hit"], dtype=bool)
    )
    transition_raw = (
        np.asarray(standard["rlp_transition_rings"], dtype=bool)
        | np.asarray(standard["rlp_rings"], dtype=bool)
    )
    boundary = active & boundary_raw & ~axis
    transition = active & transition_raw & ~axis & ~boundary
    interior = active & ~axis & ~boundary & ~transition
    return {
        "axis_ring": axis,
        "rlp_transition_or_thinned": transition,
        "physical_boundary_footprint": boundary,
        "ordinary_interior": interior,
    }


def _field_statistics(
    actual: Mapping[str, np.ndarray],
    exact: np.ndarray,
    host: Any,
    standard_masks: Mapping[str, np.ndarray],
    accounting_masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    volume = np.asarray(host.aggregate_chart_volume, dtype=np.float64)
    active = np.asarray(host.topology.is_active_owner, dtype=bool)
    result = {}
    for variant, value in actual.items():
        stats = base._variant_statistics(
            value, exact, host, standard_masks, include_rings=True
        )
        global_sse = stats["global"]["squared_error"]
        accounting = {}
        for name, mask in accounting_masks.items():
            item = base._weighted_statistics(value, exact, volume, active & mask)
            item["squared_error_fraction"] = (
                None
                if global_sse in (None, 0.0) or item["squared_error"] is None
                else float(item["squared_error"]) / float(global_sse)
            )
            accounting[name] = item
        stats["disjoint_error_accounting"] = accounting
        stats["disjoint_squared_error_fraction_sum"] = float(
            sum(
                item["squared_error_fraction"] or 0.0
                for item in accounting.values()
            )
        )
        result[variant] = stats
    return result


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    resolution = int(args.resolution)
    _progress("case_start", resolution=resolution)
    setup_started = time.perf_counter()
    artifact_path, artifact = base._load_artifact(args.geometry, resolution)
    _, identity_artifact = base._load_artifact(args.geometry, 64)
    reference_started = time.perf_counter()
    reference = build_continuum_reference_from_sidecar(
        args.reference_sidecar,
        verify_hashes=bool(args.verify_reference_hashes),
        tau=mms.PHYSICAL_PARAMETERS["tau"],
        mi_over_me=mms.PHYSICAL_PARAMETERS["mi_over_me"],
        rho_star=mms.PHYSICAL_PARAMETERS["rho_star"],
        Ve_nu=mms.PHYSICAL_PARAMETERS["Ve_nu"],
        perp_diffusion=mms.PHYSICAL_PARAMETERS["density_D_perp"],
        enable_generalized_potential=True,
    )
    reference_load_seconds = time.perf_counter() - reference_started
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
    with patch.dict(base.os.environ, base._FROZEN_RUNTIME_ENV, clear=False):
        runtime = mms._runtime(
            artifact.global_geometry, artifact.owner_geometry, runtime_args
        )
    if runtime.model is None:
        raise RuntimeError("continuous baseline requires a single-device model")
    setup_seconds = time.perf_counter() - setup_started
    projector_started = time.perf_counter()
    _progress("projector_start", resolution=resolution)
    projector = mms._QuadratureProjector(
        reference, artifact.global_geometry, artifact.owner_geometry
    )
    projector_seconds = time.perf_counter() - projector_started
    _progress(
        "projector_complete", resolution=resolution, seconds=projector_seconds
    )
    masks = base._region_masks(artifact.global_geometry, artifact.owner_geometry)
    accounting_masks = _disjoint_accounting_masks(artifact.owner_geometry, masks)
    rho_star = float(runtime.model.parameters.rho_star)
    fields = {}
    states = {}
    exact_sources = {}
    field_timings = {}
    for field in FIELDS:
        field_started = time.perf_counter()
        reference_field_started = time.perf_counter()
        raw_state, raw_exact = _project_field(
            projector, float(args.time), rho_star=rho_star, field=field
        )
        reference_field_seconds = time.perf_counter() - reference_field_started
        states[field] = raw_state
        exact_sources[field] = raw_exact
        owner_state = mms._owner_project(raw_state, artifact.owner_geometry)
        exact = mms._owner_project_array(raw_exact, artifact.owner_geometry)
        operands = base._prepare_operands(runtime.model, owner_state, raw_state)
        action_started = time.perf_counter()
        raw = base._evaluate_variants(runtime.model, operands)["production_R"]
        action_seconds = time.perf_counter() - action_started
        effective = {
            "phi_centered_action": -raw["phi_centered"] / rho_star,
            "omega_centered_reverse_action": raw["omega_centered"] / rho_star,
            "current_centered_antisymmetric": -raw["centered"] / rho_star,
            "pure_scalar_upwind": -raw["pure_upwind"] / rho_star,
            "compatible_upwind": -raw["compatible_upwind"] / rho_star,
        }
        fields[field] = {
            "manufactured_input": "physical-J weighted midpoint projection",
            "continuum_source": "independent analytic bracket at the same midpoint",
            "production_scheme": "pure_scalar_upwind",
            "variants": _field_statistics(
                effective,
                exact,
                artifact.owner_geometry,
                masks,
                accounting_masks,
            ),
        }
        field_timings[field] = {
            "reference_projection_seconds": reference_field_seconds,
            "operator_action_seconds": action_seconds,
            "total_seconds": time.perf_counter() - field_started,
        }
        _progress(
            "field_complete",
            resolution=resolution,
            field=field,
            **field_timings[field],
        )
    manifest = artifact_path / "manifest.json"
    cache_path = (
        args.reference_cache
        if args.reference_cache is not None
        else args.output.with_suffix(".reference.npz")
    )
    cache_started = time.perf_counter()
    cache = _write_reference_cache(
        cache_path,
        states=states,
        exact_sources=exact_sources,
        provenance={
            "schema": "drbx.hsx-poisson-midpoint-reference-cache-v1",
            "resolution": resolution,
            "time": float(args.time),
            "geometry_manifest_sha256": _sha256(manifest),
            "reference_source": reference.provenance,
            "sampling": "physical-J weighted cell midpoint",
            "analytic_mms_eta_period": reference.eta_period,
        },
    )
    payload = {
        "schema": SCHEMA,
        "version": 1,
        "resolution": resolution,
        "time": float(args.time),
        "geometry_artifact": str(artifact_path.resolve()),
        "geometry_manifest_sha256": _sha256(manifest),
        "reference_source": reference.provenance,
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "continuum_reference_sha256": _sha256(
                ROOT / "hsx_mms_continuum_reference.py"
            ),
            "mms_driver_sha256": _sha256(ROOT / "simulate_hsx_mms.py"),
        },
        "configuration": {
            "precision": "float64",
            "sharding": [1, 1, 1],
            "boundary_contract": {
                "physical_wall_model": "legacy-velocity-trace",
                "parallel_velocity_wall_bc": "neumann",
                "neumann_ghost_scheme": "physical",
                "parallel_boundary_pairing": "characteristic-sat",
                "parallel_characteristic_wall_law": "energy-absorbing",
            },
            "poisson_bracket_scheme": "material-scalar-third-order-upwind",
            "production_changes": [],
        },
        "fields": fields,
        "reference_cache": cache,
        "timings_seconds": {
            "setup_including_runtime": setup_seconds,
            "reference_load": reference_load_seconds,
            "projector_reference_preparation": projector_seconds,
            "fields": field_timings,
            "cache_output": time.perf_counter() - cache_started,
            "total_before_json_output": time.perf_counter() - started,
        },
    }
    output_started = time.perf_counter()
    _write(args.output, payload)
    _progress(
        "case_complete",
        resolution=resolution,
        output=str(args.output),
        output_seconds=time.perf_counter() - output_started,
        total_seconds=time.perf_counter() - started,
    )
    return payload


def _orders(resolutions: Sequence[int], values: Sequence[float]) -> list[float]:
    return [
        math.log(coarse / fine) / math.log(n_fine / n_coarse)
        for n_coarse, n_fine, coarse, fine in zip(
            resolutions[:-1], resolutions[1:], values[:-1], values[1:]
        )
    ]


def summarize(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(cases, key=lambda item: int(item["resolution"]))
    resolutions = [int(item["resolution"]) for item in ordered]
    if resolutions != [32, 48, 64]:
        raise ValueError("the qualified baseline requires resolutions 32, 48, 64")
    fields = {}
    for field in FIELDS:
        variants = {}
        for variant in VARIANTS:
            absolute = [
                case["fields"][field]["variants"][variant]["global"]["absolute_l2"]
                for case in ordered
            ]
            relative = [
                case["fields"][field]["variants"][variant]["global"]["relative_l2"]
                for case in ordered
            ]
            variants[variant] = {
                "absolute_l2": absolute,
                "absolute_order": _orders(resolutions, absolute),
                "relative_l2": relative,
                "relative_order": _orders(resolutions, relative),
            }
        production_orders = variants["pure_scalar_upwind"]["absolute_order"]
        fields[field] = {
            "variants": variants,
            "production_acceptance": {
                "required_minimum_order": 1.8,
                "orders": production_orders,
                "passed": all(value >= 1.8 for value in production_orders),
            },
        }
    payload = {
        "schema": SCHEMA,
        "version": 1,
        "resolutions": resolutions,
        "fields": fields,
        "all_production_fields_passed": all(
            value["production_acceptance"]["passed"] for value in fields.values()
        ),
        "cases": ordered,
    }
    return payload


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Exercise the actual-HSX reference, coordinates, masks, and report path."""

    started = time.perf_counter()
    artifact_path, artifact = base._load_artifact(args.geometry, 32)
    reference = build_continuum_reference_from_sidecar(
        args.reference_sidecar,
        verify_hashes=bool(args.verify_reference_hashes),
        tau=mms.PHYSICAL_PARAMETERS["tau"],
        mi_over_me=mms.PHYSICAL_PARAMETERS["mi_over_me"],
        rho_star=mms.PHYSICAL_PARAMETERS["rho_star"],
        Ve_nu=mms.PHYSICAL_PARAMETERS["Ve_nu"],
        perp_diffusion=mms.PHYSICAL_PARAMETERS["density_D_perp"],
        enable_generalized_potential=True,
    )
    geometry = artifact.global_geometry
    u = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    theta = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    eta = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    groups = np.asarray(artifact.owner_geometry.angular_group_size, dtype=np.int64)
    changes = np.flatnonzero(groups[1:] != groups[:-1])
    radial = sorted(
        set(
            [0, 1, len(u) // 2, len(u) - 2, len(u) - 1]
            + [int(value) for index in changes for value in (index, index + 1)]
        )
    )
    points = np.asarray(
        [
            [u[i], theta[j], eta[k]]
            for i in radial
            for j, k in ((0, 0), (len(theta) // 3, len(eta) // 2))
        ],
        dtype=np.float64,
    )
    reference_started = time.perf_counter()
    prepared = reference.prepare(points)
    data = reference.evaluate(points, float(args.time), prepared=prepared)
    reference_seconds = time.perf_counter() - reference_started
    source_stats = {}
    for field in FIELDS:
        if field == "actual_vorticity":
            gradient = np.asarray(data.gradients["vorticity"], dtype=np.float64)
        else:
            _value, gradient = _smooth_value_gradient(
                field, points, reference.eta_period
            )
        source = -np.sum(
            np.asarray(prepared.bcov)
            * np.cross(np.asarray(data.gradients["phi"]), gradient),
            axis=-1,
        ) / (
            float(mms.PHYSICAL_PARAMETERS["rho_star"])
            * np.maximum(np.abs(np.asarray(prepared.J)), 1.0e-30)
            * np.maximum(np.asarray(prepared.B), 1.0e-30)
        )
        source_stats[field] = {
            "finite": bool(np.all(np.isfinite(source))),
            "minimum": float(np.min(source)),
            "maximum": float(np.max(source)),
        }
    payload = {
        "schema": "drbx.hsx-poisson-continuous-preflight-v1",
        "geometry_artifact": str(artifact_path.resolve()),
        "geometry_manifest_sha256": _sha256(artifact_path / "manifest.json"),
        "reference_source": reference.provenance,
        "sample_count": int(points.shape[0]),
        "sampled_radial_indices": radial,
        "coordinate_contract": {
            "reference_input": "logical (u,theta,eta)",
            "compact_face_storage": "regular (x,y,eta), converted before reference use",
            "analytic_mms_eta_period": reference.eta_period,
            "continuous_evaluator_period": reference.metric_evaluator.period,
        },
        "boundary_contract": {
            "physical_wall_model": "legacy-velocity-trace",
            "parallel_velocity_wall_bc": "neumann",
            "neumann_ghost_scheme": "physical",
            "parallel_boundary_pairing": "characteristic-sat",
            "parallel_characteristic_wall_law": "energy-absorbing",
        },
        "sources": source_stats,
        "timings_seconds": {
            "reference_batch": reference_seconds,
            "total": time.perf_counter() - started,
        },
        "passed": bool(
            np.all(np.asarray(prepared.J) > 0.0)
            and all(item["finite"] for item in source_stats.values())
            and np.isclose(reference.eta_period, 2.0 * np.pi)
            and np.isclose(reference.metric_evaluator.period, 0.5 * np.pi)
        ),
    }
    _write(args.output, payload)
    _progress("preflight_complete", output=str(args.output), passed=payload["passed"])
    if not payload["passed"]:
        raise RuntimeError("actual-HSX continuous-reference preflight failed")
    return payload


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for resolution in (32, 48, 64):
        path = output / f"N{resolution}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "case",
            "--geometry",
            str(args.geometry.resolve()),
            "--reference-sidecar",
            str(args.reference_sidecar.resolve()),
            "--resolution",
            str(resolution),
            "--time",
            str(args.time),
            "--output",
            str(path),
            "--reference-cache",
            str(output / f"N{resolution}.reference.npz"),
        ]
        if args.verify_reference_hashes:
            command.append("--verify-reference-hashes")
        subprocess.run(command, check=True)
        paths.append(path)
    payload = summarize([json.loads(path.read_text()) for path in paths])
    payload["campaign"] = {
        "one_resolution_per_process": True,
        "midpoint_default": True,
        "full_domain_high_order_quadrature": False,
        "production_changes": [],
    }
    _write(output / "summary.json", payload)
    print(output / "summary.json")
    return payload


def run_merge(args: argparse.Namespace) -> dict[str, Any]:
    payload = summarize(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    )
    payload["campaign"] = {
        "one_resolution_per_process": True,
        "midpoint_default": True,
        "full_domain_high_order_quadrature": False,
        "production_changes": [],
        "assembled_from_valid_atomic_case_outputs": True,
    }
    _write(args.output, payload)
    _progress("summary_complete", output=str(args.output))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    case = sub.add_parser("case")
    case.add_argument("--geometry", type=Path, required=True)
    case.add_argument("--reference-sidecar", type=Path, required=True)
    case.add_argument("--resolution", type=int, required=True)
    case.add_argument("--time", type=float, default=1.0e-6)
    case.add_argument("--verify-reference-hashes", action="store_true")
    case.add_argument("--reference-cache", type=Path)
    case.add_argument("--output", type=Path, required=True)
    case.set_defaults(handler=run_case)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--geometry", type=Path, required=True)
    preflight.add_argument("--reference-sidecar", type=Path, required=True)
    preflight.add_argument("--time", type=float, default=1.0e-6)
    preflight.add_argument("--verify-reference-hashes", action="store_true")
    preflight.add_argument("--output", type=Path, required=True)
    preflight.set_defaults(handler=run_preflight)
    campaign = sub.add_parser("campaign")
    campaign.add_argument("--geometry", type=Path, required=True)
    campaign.add_argument("--reference-sidecar", type=Path, required=True)
    campaign.add_argument("--time", type=float, default=1.0e-6)
    campaign.add_argument("--verify-reference-hashes", action="store_true")
    campaign.add_argument("--output", type=Path, required=True)
    campaign.set_defaults(handler=run_campaign)
    merge = sub.add_parser("merge")
    merge.add_argument("inputs", type=Path, nargs=3)
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(handler=run_merge)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
