#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import signal
import time

import numpy as np

from jax_drb.config.boutinp import apply_bout_overrides
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.recycling_1d import advance_recycling_1d_implicit_history
from jax_drb.native.runner import _load_curated_case_config
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.parity.arrays import load_portable_array_payload
from jax_drb.parity.reference import resolve_reference_case
from jax_drb.runtime.run_config import RunConfiguration


DEFAULT_FIELDS = ("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe")
SOLVER_MODES = (
    "continuation",
    "bdf",
    "bdf_fixed_full_field_jvp",
    "bdf_active_array_jvp",
    "fixed_bdf2_jax_linearized",
    "fixed_bdf2_jax_linearized_lineax",
    "fixed_bdf2_active_array_jax_linearized",
    "fixed_bdf2_active_array_jax_linearized_lineax",
    "fixed_bdf2_promoted_active_sources_jax_linearized",
    "adaptive_be",
    "adaptive_bdf",
    "adaptive_bdf_sparse_jvp",
    "adaptive_bdf_jax_linearized",
    "adaptive_bdf_jax_linearized_lineax",
    "adaptive_bdf_active_array_jax_linearized",
    "adaptive_bdf_active_array_jax_linearized_lineax",
)
BDF_BASE_MODE = "bdf"
BDF_JVP_BACKENDS = {
    "bdf_fixed_full_field_jvp": "fixed_full_field_array",
    "bdf_active_array_jvp": "active_array",
}
BDF_PAIRWISE_CANDIDATE_MODES = tuple(BDF_JVP_BACKENDS)
FIXED_BDF2_MODES = (
    "fixed_bdf2_jax_linearized",
    "fixed_bdf2_jax_linearized_lineax",
    "fixed_bdf2_active_array_jax_linearized",
    "fixed_bdf2_active_array_jax_linearized_lineax",
    "fixed_bdf2_promoted_active_sources_jax_linearized",
)
FIXED_BDF2_STEP_SOLVER_MODES = {
    "fixed_bdf2_jax_linearized": "jax_linearized",
    "fixed_bdf2_jax_linearized_lineax": "jax_linearized_lineax",
    "fixed_bdf2_active_array_jax_linearized": "active_array_jax_linearized",
    "fixed_bdf2_active_array_jax_linearized_lineax": (
        "active_array_jax_linearized_lineax"
    ),
    "fixed_bdf2_promoted_active_sources_jax_linearized": (
        "promoted_active_sources_jax_linearized"
    ),
}
FIXED_BDF2_RHS_BACKENDS = {
    "fixed_bdf2_jax_linearized": "fixed_full_field_array",
    "fixed_bdf2_jax_linearized_lineax": "fixed_full_field_array",
    "fixed_bdf2_active_array_jax_linearized": "active_array",
    "fixed_bdf2_active_array_jax_linearized_lineax": "active_array",
    "fixed_bdf2_promoted_active_sources_jax_linearized": "promoted_active_sources",
}
ADAPTIVE_BDF_MODES = (
    "adaptive_bdf",
    "adaptive_bdf_sparse_jvp",
    "adaptive_bdf_jax_linearized",
    "adaptive_bdf_jax_linearized_lineax",
    "adaptive_bdf_active_array_jax_linearized",
    "adaptive_bdf_active_array_jax_linearized_lineax",
)
ADAPTIVE_BDF_STEP_SOLVER_MODES = {
    "adaptive_bdf": "sparse",
    "adaptive_bdf_sparse_jvp": "sparse_jvp",
    "adaptive_bdf_jax_linearized": "jax_linearized",
    "adaptive_bdf_jax_linearized_lineax": "jax_linearized_lineax",
    "adaptive_bdf_active_array_jax_linearized": "active_array_jax_linearized",
    "adaptive_bdf_active_array_jax_linearized_lineax": (
        "active_array_jax_linearized_lineax"
    ),
}
ADAPTIVE_BDF_RHS_BACKENDS = {
    "adaptive_bdf": "host_bridge",
    "adaptive_bdf_sparse_jvp": "fixed_full_field_array",
    "adaptive_bdf_jax_linearized": "fixed_full_field_array",
    "adaptive_bdf_jax_linearized_lineax": "fixed_full_field_array",
    "adaptive_bdf_active_array_jax_linearized": "active_array",
    "adaptive_bdf_active_array_jax_linearized_lineax": "active_array",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare native recycling one-step transient solver modes against committed array baselines.",
    )
    parser.add_argument(
        "--case",
        default="recycling_1d_one_step",
        choices=("recycling_1d_one_step", "recycling_dthe_one_step"),
    )
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument(
        "--mode",
        action="append",
        dest="modes",
        choices=SOLVER_MODES,
        help=(
            "Solver modes to compare. May be repeated. Use bdf_fixed_full_field_jvp "
            "or bdf_active_array_jvp to exercise the fixed-layout JVP BDF paths; "
            "use fixed_bdf2_active_array_jax_linearized for the non-SciPy "
            "active-array fixed-BDF2 path, and use "
            "fixed_bdf2_promoted_active_sources_jax_linearized for the opt-in "
            "promoted source-kernel fixed-BDF2 path. Defaults include the main "
            "supported set for the case."
        ),
    )
    parser.add_argument(
        "--field",
        action="append",
        dest="fields",
        help="Fields to report. May be repeated.",
    )
    parser.add_argument(
        "--override",
        action="append",
        dest="overrides",
        default=None,
        help=(
            "Apply a BOUT.inp-style override before building the native run configuration, "
            "for example --override solver:rtol=1e-9. May be repeated."
        ),
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=None,
        help=(
            "Override the single output-window timestep used by this comparison. Use with "
            "--diagnostics-only unless the committed baseline was generated at the same time."
        ),
    )
    parser.add_argument(
        "--max-nonlinear-iterations",
        type=int,
        default=30,
        help="Maximum nonlinear iterations passed to each implicit recycling history solve.",
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help=(
            "Run solver modes and diagnostics without comparing against the committed array "
            "baseline. Pairwise mode deltas and requested diagnostics gates still run."
        ),
    )
    parser.add_argument(
        "--mode-timeout-seconds",
        type=float,
        default=None,
        help="Fail if an individual solver mode exceeds this many wall-clock seconds.",
    )
    parser.add_argument(
        "--require-bdf-pairwise-max",
        type=float,
        default=None,
        help=(
            "Fail unless the largest active-mesh bdf-vs-BDF-JVP-candidate "
            "delta over reported fields is below this value."
        ),
    )
    parser.add_argument(
        "--require-fixed-bdf2-pairwise-max",
        type=float,
        default=None,
        help=(
            "Fail unless the largest active-mesh bdf-vs-fixed-BDF2-candidate "
            "delta over reported fields is below this value. This is a "
            "physical-output parity gate for the matrix-free fixed-BDF2 route."
        ),
    )
    parser.add_argument(
        "--require-fixed-bdf2-pairwise-l2-rel-max",
        type=float,
        default=None,
        help=(
            "Fail unless the largest active-mesh relative L2 difference between "
            "bdf and fixed-BDF2 candidates over reported fields is below this "
            "value. This complements the pointwise max-norm gate for production "
            "window screens where localized cells dominate max_abs_delta."
        ),
    )
    parser.add_argument(
        "--require-fixed-bdf2-pairwise-inventory-rel-max",
        type=float,
        default=None,
        help=(
            "Fail unless the largest unweighted active-mesh inventory relative "
            "difference between bdf and fixed-BDF2 candidates over reported "
            "fields is below this value. The inventory is an active-cell sum, "
            "not a metric-weighted volume integral."
        ),
    )
    parser.add_argument(
        "--require-fixed-jvp-diagnostics",
        action="store_true",
        help=(
            "Fail unless every requested BDF JVP mode reports the expected RHS "
            "backend, JVP Jacobian mode, zero finite-difference base-RHS Jacobian "
            "calls, and prebuilt direction-batch reuse."
        ),
    )
    parser.add_argument(
        "--require-fixed-bdf2-diagnostics",
        action="store_true",
        help=(
            "Fail unless every requested fixed_bdf2_*jax_linearized mode reports "
            "the expected fixed-layout RHS backend, JAX-linearized Jacobian "
            "actions, packed feedback-integral evolution, and at least one "
            "fixed-BDF2 linear solve attempt."
        ),
    )
    parser.add_argument(
        "--require-fixed-bdf2-max-residual",
        type=float,
        default=1.0e-5,
        help=(
            "Maximum allowed fixed_bdf2_max_residual_inf_norm when "
            "--require-fixed-bdf2-diagnostics is used."
        ),
    )
    parser.add_argument(
        "--require-fixed-bdf2-linear-preconditioner",
        default=None,
        help=(
            "Fail unless every requested fixed_bdf2_*jax_linearized mode reports "
            "this JAX-GMRES preconditioner name and at least one preconditioner build."
        ),
    )
    parser.add_argument(
        "--require-fixed-bdf2-linear-solver-backend",
        default=None,
        help=(
            "Fail unless every requested fixed_bdf2_*jax_linearized mode reports "
            "this JAX-linearized linear solver backend, for example jax_gmres or "
            "jax_bicgstab."
        ),
    )
    parser.add_argument(
        "--require-fixed-bdf2-linear-operator-jitted",
        action="store_true",
        help=(
            "Fail unless every requested fixed_bdf2_*jax_linearized mode reports "
            "JIT-wrapped linear-operator use on every JAX-linearized internal step."
        ),
    )
    parser.add_argument(
        "--require-fixed-bdf2-line-search-mode",
        default=None,
        help=(
            "Fail unless every requested fixed_bdf2_*jax_linearized mode reports "
            "this line-search mode, for example backtracking or full_step."
        ),
    )
    parser.add_argument(
        "--require-fixed-bdf2-max-linear-iterations",
        type=int,
        default=None,
        help=(
            "Fail unless every requested fixed_bdf2_*jax_linearized mode reports "
            "fixed_bdf2_total_linear_iterations at or below this value. Use this "
            "as a performance-promotion gate; it is intentionally separate from "
            "the correctness diagnostics gate."
        ),
    )
    parser.add_argument(
        "--require-fixed-bdf2-max-residual-evaluations",
        type=int,
        default=None,
        help=(
            "Fail unless every requested fixed_bdf2_*jax_linearized mode reports "
            "fixed_bdf2_total_residual_evaluation_count at or below this value. "
            "Use this to keep line-search and residual-rebuild work bounded when "
            "promoting fixed-layout recycling solvers."
        ),
    )
    parser.add_argument(
        "--require-fixed-bdf2-max-linear-operator-calls",
        type=int,
        default=None,
        help=(
            "Fail unless every requested fixed_bdf2_*jax_linearized mode reports "
            "fixed_bdf2_total_linear_operator_call_count at or below this value. "
            "This gates the actual JVP/linear-map work used by JAX-native "
            "preconditioner performance campaigns."
        ),
    )
    parser.add_argument(
        "--require-fixed-bdf2-min-linear-solve-count",
        type=int,
        default=None,
        help=(
            "Fail unless every requested fixed_bdf2_*jax_linearized mode reports "
            "fixed_bdf2_total_linear_solve_count at or above this value. Use "
            "this with direct linear-operator counting, where Python-visible "
            "operator calls are intentionally unavailable."
        ),
    )
    parser.add_argument(
        "--require-fixed-bdf2-max-linear-update-residual",
        type=float,
        default=None,
        help=(
            "Fail unless every requested fixed_bdf2_*jax_linearized mode reports "
            "fixed_bdf2_max_linear_update_residual_inf_norm at or below this "
            "finite nonnegative ceiling. Requires "
            "runtime:recycling_jax_linear_diagnose_update_residual=true."
        ),
    )
    parser.add_argument(
        "--require-fixed-bdf2-max-linear-update-relative-residual",
        type=float,
        default=None,
        help=(
            "Fail unless every requested fixed_bdf2_*jax_linearized mode reports "
            "fixed_bdf2_max_linear_update_relative_residual at or below this "
            "finite nonnegative ceiling. This is the preferred preconditioner "
            "quality gate for explicit Krylov-budget screens."
        ),
    )
    parser.add_argument(
        "--require-fixed-bdf2-max-preconditioner-builds",
        type=int,
        default=None,
        help=(
            "Fail unless every requested fixed_bdf2_*jax_linearized mode reports "
            "fixed_bdf2_total_linear_preconditioner_build_count at or below this "
            "value. This is useful when testing dynamic-preconditioner reuse."
        ),
    )
    parser.add_argument(
        "--require-fixed-bdf2-max-preconditioner-applies",
        type=int,
        default=None,
        help=(
            "Fail unless every requested fixed_bdf2_*jax_linearized mode reports "
            "fixed_bdf2_total_linear_preconditioner_apply_count at or below this "
            "value. Pair this with operator-call budgets to reject preconditioners "
            "that are applied frequently without reducing Krylov work."
        ),
    )
    parser.add_argument(
        "--require-adaptive-bdf-linear-preconditioner",
        default=None,
        help=(
            "Fail unless every requested adaptive_bdf_*jax_linearized mode reports "
            "this JAX-GMRES preconditioner name and at least one preconditioner build."
        ),
    )
    parser.add_argument(
        "--require-adaptive-bdf-no-fallback",
        action="store_true",
        help="Fail unless every requested adaptive-BDF mode reports zero minimum-dt fallback accepts.",
    )
    parser.add_argument(
        "--require-adaptive-bdf-no-unconverged-substeps",
        action="store_true",
        help="Fail unless every requested adaptive-BDF mode reports zero unconverged implicit substeps.",
    )
    parser.add_argument(
        "--require-adaptive-bdf-max-error-ratio",
        type=float,
        default=None,
        help=(
            "Fail unless every requested adaptive-BDF mode reports an embedded max error ratio "
            "at or below this threshold."
        ),
    )
    parser.add_argument(
        "--require-adaptive-bdf-max-accepted-error-ratio",
        type=float,
        default=None,
        help=(
            "Fail unless every requested adaptive-BDF mode reports a maximum accepted-step "
            "embedded error ratio at or below this threshold."
        ),
    )
    parser.add_argument(
        "--require-adaptive-bdf-max-linear-update-residual",
        type=float,
        default=None,
        help=(
            "Fail unless every requested adaptive_bdf_*jax_linearized mode reports "
            "adaptive_bdf_max_linear_update_residual_inf_norm at or below this "
            "finite nonnegative ceiling. Requires "
            "runtime:recycling_jax_linear_diagnose_update_residual=true."
        ),
    )
    parser.add_argument(
        "--require-adaptive-bdf-max-linear-update-relative-residual",
        type=float,
        default=None,
        help=(
            "Fail unless every requested adaptive_bdf_*jax_linearized mode reports "
            "adaptive_bdf_max_linear_update_relative_residual at or below this "
            "finite nonnegative ceiling."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help=(
            "Number of output windows to run per mode. Use at least 2 when "
            "requiring fixed-BDF2 diagnostics so the BDF2 corrector is actually exercised."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional lightweight JSON report path for mode timings, diagnostics, gates, and pairwise deltas.",
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    return parser.parse_args(argv)


def _resolve_output_timestep(
    args: argparse.Namespace, run_config: RunConfiguration
) -> float:
    timestep = (
        run_config.time.timestep if args.timestep is None else float(args.timestep)
    )
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("--timestep must be a positive finite value.")
    return float(timestep)


def _resolve_max_nonlinear_iterations(args: argparse.Namespace) -> int:
    iterations = int(args.max_nonlinear_iterations)
    if iterations <= 0:
        raise ValueError("--max-nonlinear-iterations must be positive.")
    return iterations


def _resolve_steps(args: argparse.Namespace) -> int:
    steps = int(args.steps)
    if steps <= 0:
        raise ValueError("--steps must be positive.")
    return steps


def _default_modes(case_name: str) -> tuple[str, ...]:
    if case_name == "recycling_1d_one_step":
        return (
            "continuation",
            "bdf",
            "bdf_fixed_full_field_jvp",
            "bdf_active_array_jvp",
            "fixed_bdf2_jax_linearized",
            "fixed_bdf2_active_array_jax_linearized",
            "adaptive_be",
            "adaptive_bdf",
        )
    return (
        "bdf",
        "bdf_fixed_full_field_jvp",
        "bdf_active_array_jvp",
        "fixed_bdf2_jax_linearized",
        "fixed_bdf2_active_array_jax_linearized",
        "adaptive_be",
        "adaptive_bdf",
    )


def _summarize_mode_errors(
    actual_variables: dict[str, np.ndarray],
    expected_variables: dict[str, np.ndarray],
    *,
    fields: tuple[str, ...],
    mesh=None,
    crop_expected: bool = False,
) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for field in fields:
        if field not in actual_variables or field not in expected_variables:
            continue
        actual = np.asarray(actual_variables[field], dtype=np.float64)
        expected = np.asarray(expected_variables[field], dtype=np.float64)
        actual = _active_mesh_view(actual, mesh)
        if crop_expected:
            expected = _active_mesh_view(expected, mesh)
        if actual.shape != expected.shape:
            rows.append((field, float("inf")))
            continue
        rows.append((field, float(np.nanmax(np.abs(actual - expected)))))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def _relative_denominator(value: float, *, floor: float = 1.0e-300) -> float:
    return max(abs(float(value)), floor)


def _summarize_pairwise_observable_errors(
    reference_variables: dict[str, np.ndarray],
    candidate_variables: dict[str, np.ndarray],
    *,
    fields: tuple[str, ...],
    mesh=None,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for field in fields:
        if field not in reference_variables or field not in candidate_variables:
            continue
        reference = _active_mesh_view(
            np.asarray(reference_variables[field], dtype=np.float64), mesh
        )
        candidate = _active_mesh_view(
            np.asarray(candidate_variables[field], dtype=np.float64), mesh
        )
        if reference.shape != candidate.shape:
            rows.append(
                {
                    "field": field,
                    "l2_relative_delta": float("inf"),
                    "inventory_relative_delta": float("inf"),
                    "inventory_delta": float("inf"),
                    "reference_inventory": float("nan"),
                }
            )
            continue
        delta = candidate - reference
        reference_norm = float(np.linalg.norm(reference.ravel(), ord=2))
        l2_relative = float(
            np.linalg.norm(delta.ravel(), ord=2)
            / _relative_denominator(reference_norm)
        )
        reference_inventory = float(np.sum(reference))
        inventory_delta = float(abs(np.sum(candidate) - reference_inventory))
        inventory_relative = float(
            inventory_delta / _relative_denominator(reference_inventory)
        )
        rows.append(
            {
                "field": field,
                "l2_relative_delta": l2_relative,
                "inventory_relative_delta": inventory_relative,
                "inventory_delta": inventory_delta,
                "reference_inventory": reference_inventory,
            }
        )
    rows.sort(
        key=lambda item: max(
            float(item["l2_relative_delta"]),
            float(item["inventory_relative_delta"]),
        ),
        reverse=True,
    )
    return rows


def _active_mesh_view(values: np.ndarray, mesh) -> np.ndarray:
    if mesh is None or values.ndim != 4:
        return values
    if values.shape[1] >= mesh.xend + 1 and values.shape[2] >= mesh.yend + 1:
        return values[:, mesh.xstart : mesh.xend + 1, mesh.ystart : mesh.yend + 1, :]
    return values


def _format_mode_error_report(
    mode: str, *, elapsed: float, rows: list[tuple[str, float]]
) -> list[str]:
    lines = [f"mode={mode} elapsed={elapsed:.3f}s"]
    for field, max_abs in rows:
        lines.append(f"  {field}: max_abs_diff={max_abs:.8e}")
    if rows:
        lines.append(f"  worst={rows[0][0]} diff={rows[0][1]:.8e}")
    return lines


def _format_mode_diagnostics_report(
    mode: str, diagnostics: dict[str, object]
) -> list[str]:
    if not diagnostics:
        return []
    lines = [f"diagnostics mode={mode}"]
    for name in sorted(diagnostics):
        value = diagnostics[name]
        if isinstance(value, float):
            formatted = f"{value:.8e}"
        else:
            formatted = str(value)
        lines.append(f"  {name}={formatted}")
    return lines


def _format_bdf_pairwise_delta_report(
    mode_variables: dict[str, dict[str, np.ndarray]],
    *,
    fields: tuple[str, ...],
    mesh=None,
) -> list[str]:
    base_mode = BDF_BASE_MODE
    if base_mode not in mode_variables:
        return []

    lines: list[str] = []
    for candidate_mode in BDF_PAIRWISE_CANDIDATE_MODES:
        if candidate_mode not in mode_variables:
            continue
        rows = _summarize_mode_errors(
            mode_variables[base_mode],
            mode_variables[candidate_mode],
            fields=fields,
            mesh=mesh,
            crop_expected=True,
        )
        lines.append(f"pairwise_delta={base_mode}_vs_{candidate_mode}")
        for field, max_abs in rows:
            lines.append(f"  {field}: max_abs_delta={max_abs:.8e}")
        if rows:
            lines.append(f"  worst={rows[0][0]} delta={rows[0][1]:.8e}")
    return lines


def _format_fixed_bdf2_pairwise_delta_report(
    mode_variables: dict[str, dict[str, np.ndarray]],
    *,
    fields: tuple[str, ...],
    mesh=None,
) -> list[str]:
    base_mode = BDF_BASE_MODE
    if base_mode not in mode_variables:
        return []

    lines: list[str] = []
    for candidate_mode in FIXED_BDF2_MODES:
        if candidate_mode not in mode_variables:
            continue
        rows = _summarize_mode_errors(
            mode_variables[base_mode],
            mode_variables[candidate_mode],
            fields=fields,
            mesh=mesh,
            crop_expected=True,
        )
        lines.append(f"pairwise_delta={base_mode}_vs_{candidate_mode}")
        for field, max_abs in rows:
            lines.append(f"  {field}: max_abs_delta={max_abs:.8e}")
        if rows:
            lines.append(f"  worst={rows[0][0]} delta={rows[0][1]:.8e}")
    return lines


def _format_fixed_bdf2_pairwise_observable_report(
    mode_variables: dict[str, dict[str, np.ndarray]],
    *,
    fields: tuple[str, ...],
    mesh=None,
) -> list[str]:
    base_mode = BDF_BASE_MODE
    if base_mode not in mode_variables:
        return []

    lines: list[str] = []
    for candidate_mode in FIXED_BDF2_MODES:
        if candidate_mode not in mode_variables:
            continue
        rows = _summarize_pairwise_observable_errors(
            mode_variables[base_mode],
            mode_variables[candidate_mode],
            fields=fields,
            mesh=mesh,
        )
        lines.append(f"pairwise_observable={base_mode}_vs_{candidate_mode}")
        for row in rows:
            field = str(row["field"])
            lines.append(
                f"  {field}: l2_relative_delta={float(row['l2_relative_delta']):.8e} "
                f"inventory_relative_delta={float(row['inventory_relative_delta']):.8e} "
                f"inventory_delta={float(row['inventory_delta']):.8e} "
                f"reference_inventory={float(row['reference_inventory']):.8e}"
            )
        if rows:
            worst_l2 = max(rows, key=lambda item: float(item["l2_relative_delta"]))
            worst_inventory = max(
                rows, key=lambda item: float(item["inventory_relative_delta"])
            )
            lines.append(
                f"  worst_l2={worst_l2['field']} "
                f"delta={float(worst_l2['l2_relative_delta']):.8e}"
            )
            lines.append(
                f"  worst_inventory={worst_inventory['field']} "
                f"delta={float(worst_inventory['inventory_relative_delta']):.8e}"
            )
    return lines


def _bdf_pairwise_worst_delta(
    mode_variables: dict[str, dict[str, np.ndarray]],
    *,
    fields: tuple[str, ...],
    mesh=None,
) -> tuple[str | None, float | None]:
    base_mode = BDF_BASE_MODE
    if base_mode not in mode_variables:
        return None, None
    worst: tuple[str, float] | None = None
    for candidate_mode in BDF_PAIRWISE_CANDIDATE_MODES:
        if candidate_mode not in mode_variables:
            continue
        rows = _summarize_mode_errors(
            mode_variables[base_mode],
            mode_variables[candidate_mode],
            fields=fields,
            mesh=mesh,
            crop_expected=True,
        )
        if not rows:
            continue
        candidate_worst = rows[0]
        if worst is None or candidate_worst[1] > worst[1]:
            worst = candidate_worst
    if worst is None:
        return None, None
    return worst


def _fixed_bdf2_pairwise_worst_delta(
    mode_variables: dict[str, dict[str, np.ndarray]],
    *,
    fields: tuple[str, ...],
    mesh=None,
) -> tuple[str | None, float | None]:
    base_mode = BDF_BASE_MODE
    if base_mode not in mode_variables:
        return None, None
    worst: tuple[str, float] | None = None
    for candidate_mode in FIXED_BDF2_MODES:
        if candidate_mode not in mode_variables:
            continue
        rows = _summarize_mode_errors(
            mode_variables[base_mode],
            mode_variables[candidate_mode],
            fields=fields,
            mesh=mesh,
            crop_expected=True,
        )
        if not rows:
            continue
        candidate_worst = rows[0]
        if worst is None or candidate_worst[1] > worst[1]:
            worst = candidate_worst
    if worst is None:
        return None, None
    return worst


def _fixed_bdf2_pairwise_observable_worst(
    mode_variables: dict[str, dict[str, np.ndarray]],
    *,
    fields: tuple[str, ...],
    mesh=None,
) -> dict[str, dict[str, str | float | None]]:
    base_mode = BDF_BASE_MODE
    empty = {"mode": None, "field": None, "delta": None}
    worst_l2: dict[str, str | float | None] | None = None
    worst_inventory: dict[str, str | float | None] | None = None
    if base_mode not in mode_variables:
        return {"l2_relative": empty.copy(), "inventory_relative": empty.copy()}
    for candidate_mode in FIXED_BDF2_MODES:
        if candidate_mode not in mode_variables:
            continue
        rows = _summarize_pairwise_observable_errors(
            mode_variables[base_mode],
            mode_variables[candidate_mode],
            fields=fields,
            mesh=mesh,
        )
        if not rows:
            continue
        candidate_l2 = max(rows, key=lambda item: float(item["l2_relative_delta"]))
        candidate_inventory = max(
            rows, key=lambda item: float(item["inventory_relative_delta"])
        )
        if (
            worst_l2 is None
            or float(candidate_l2["l2_relative_delta"]) > float(worst_l2["delta"])
        ):
            worst_l2 = {
                "mode": candidate_mode,
                "field": str(candidate_l2["field"]),
                "delta": float(candidate_l2["l2_relative_delta"]),
            }
        if (
            worst_inventory is None
            or float(candidate_inventory["inventory_relative_delta"])
            > float(worst_inventory["delta"])
        ):
            worst_inventory = {
                "mode": candidate_mode,
                "field": str(candidate_inventory["field"]),
                "delta": float(candidate_inventory["inventory_relative_delta"]),
            }
    return {
        "l2_relative": worst_l2 if worst_l2 is not None else empty.copy(),
        "inventory_relative": (
            worst_inventory if worst_inventory is not None else empty.copy()
        ),
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, Path):
        return value.name
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _build_json_report(
    *,
    case_name: str,
    configured_timestep: float,
    timestep: float,
    max_nonlinear_iterations: int,
    steps: int,
    fields: tuple[str, ...],
    modes: tuple[str, ...],
    diagnostics_only: bool,
    mode_elapsed_seconds: dict[str, float],
    mode_diagnostics: dict[str, dict[str, object]],
    bdf_pairwise_worst: tuple[str | None, float | None],
    fixed_bdf2_pairwise_worst: tuple[str | None, float | None],
    fixed_bdf2_pairwise_observable_worst: dict[str, dict[str, str | float | None]],
    adaptive_bdf_gate_errors: dict[str, list[str]],
) -> dict[str, object]:
    worst_field, worst_delta = bdf_pairwise_worst
    fixed_bdf2_worst_field, fixed_bdf2_worst_delta = fixed_bdf2_pairwise_worst
    return {
        "case": str(case_name),
        "configured_timestep": float(configured_timestep),
        "timestep": float(timestep),
        "max_nonlinear_iterations": int(max_nonlinear_iterations),
        "steps": int(steps),
        "fields": list(fields),
        "modes": list(modes),
        "diagnostics_only": bool(diagnostics_only),
        "mode_elapsed_seconds": _json_ready(mode_elapsed_seconds),
        "mode_diagnostics": _json_ready(mode_diagnostics),
        "bdf_pairwise_worst": {
            "field": worst_field,
            "delta": worst_delta,
        },
        "fixed_bdf2_pairwise_worst": {
            "field": fixed_bdf2_worst_field,
            "delta": fixed_bdf2_worst_delta,
        },
        "fixed_bdf2_pairwise_observable_worst": _json_ready(
            fixed_bdf2_pairwise_observable_worst
        ),
        "adaptive_bdf_gate_errors": _json_ready(adaptive_bdf_gate_errors),
    }


def _write_json_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _validate_bdf_jvp_diagnostics(
    mode: str,
    diagnostics: dict[str, object],
) -> list[str]:
    errors: list[str] = []
    expected_backend = BDF_JVP_BACKENDS[mode]
    if diagnostics.get("bdf_rhs_backend") != expected_backend:
        errors.append(
            f"{mode} did not report bdf_rhs_backend={expected_backend}"
        )
    if diagnostics.get("bdf_jacobian_mode") != "jvp":
        errors.append(f"{mode} did not report bdf_jacobian_mode=jvp")
    if int(diagnostics.get("bdf_jacobian_base_rhs_evaluation_count", -1)) != 0:
        errors.append(
            f"{mode} reported finite-difference base RHS Jacobian evaluations"
        )
    if int(diagnostics.get("bdf_jvp_rhs_evaluation_count", 0)) <= 0:
        errors.append(f"{mode} did not report any JVP RHS evaluations")
    if int(diagnostics.get("bdf_jvp_jacobian_prebuilt_direction_batch_uses", 0)) <= 0:
        errors.append(f"{mode} did not report prebuilt JVP direction-batch reuse")
    return errors


def _validate_fixed_full_field_jvp_diagnostics(
    diagnostics: dict[str, object],
) -> list[str]:
    return _validate_bdf_jvp_diagnostics("bdf_fixed_full_field_jvp", diagnostics)


def _fixed_bdf2_modes_to_validate(modes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(mode for mode in modes if mode in FIXED_BDF2_MODES)


def _validate_fixed_bdf2_diagnostics(
    mode: str,
    diagnostics: dict[str, object],
    *,
    max_residual_inf_norm: float | None = 1.0e-5,
    required_linear_preconditioner: str | None = None,
    required_linear_solver_backend: str | None = None,
    require_linear_operator_jitted: bool = False,
    required_line_search_mode: str | None = None,
    max_linear_iterations: int | None = None,
    max_residual_evaluations: int | None = None,
    max_linear_operator_calls: int | None = None,
    min_linear_solve_count: int | None = None,
    max_linear_update_residual_inf_norm: float | None = None,
    max_linear_update_relative_residual: float | None = None,
    max_preconditioner_builds: int | None = None,
    max_preconditioner_applies: int | None = None,
) -> list[str]:
    errors: list[str] = []
    expected_step_solver = FIXED_BDF2_STEP_SOLVER_MODES[mode]
    expected_rhs_backend = FIXED_BDF2_RHS_BACKENDS[mode]
    if diagnostics.get("fixed_bdf2_solver_mode") != mode:
        errors.append(f"{mode} did not report fixed_bdf2_solver_mode={mode}")
    if diagnostics.get("fixed_bdf2_step_solver_mode") != expected_step_solver:
        errors.append(
            f"{mode} did not report fixed_bdf2_step_solver_mode={expected_step_solver}"
        )
    if expected_rhs_backend == "fixed_full_field_array":
        rhs_steps = int(diagnostics.get("fixed_bdf2_fixed_full_field_rhs_steps", 0))
    elif expected_rhs_backend == "active_array":
        rhs_steps = int(diagnostics.get("fixed_bdf2_active_array_rhs_steps", 0))
    else:
        rhs_steps = int(
            diagnostics.get("fixed_bdf2_promoted_active_source_rhs_steps", 0)
        )
    if rhs_steps <= 0:
        errors.append(
            f"{mode} did not report any {expected_rhs_backend} fixed-layout RHS steps"
        )
    if int(diagnostics.get("fixed_bdf2_jax_linearized_action_steps", 0)) <= 0:
        errors.append(f"{mode} did not report any JAX-linearized solver steps")
    if "lineax" in expected_step_solver:
        lineax_steps = int(diagnostics.get("fixed_bdf2_lineax_action_steps", 0))
        if lineax_steps <= 0:
            errors.append(f"{mode} did not report any Lineax solver steps")
    unconverged_count = int(diagnostics.get("fixed_bdf2_unconverged_solver_steps", 0))
    if unconverged_count != 0:
        errors.append(
            f"{mode} reported {unconverged_count} unconverged fixed BDF2 implicit steps"
        )
    unknown_count = int(
        diagnostics.get("fixed_bdf2_unknown_convergence_solver_steps", 0)
    )
    if unknown_count != 0:
        errors.append(
            f"{mode} reported {unknown_count} unknown-convergence fixed BDF2 implicit steps"
        )
    linear_failed_count = int(
        diagnostics.get("fixed_bdf2_linear_solver_failed_steps", 0)
    )
    if linear_failed_count != 0:
        errors.append(
            f"{mode} reported {linear_failed_count} failed fixed BDF2 linear solves"
        )
    if diagnostics.get("fixed_bdf2_evolve_feedback_integrals") is not True:
        errors.append(f"{mode} did not evolve packed feedback integrals")
    accepted_steps = int(diagnostics.get("fixed_bdf2_startup_steps", 0)) + int(
        diagnostics.get("fixed_bdf2_bdf2_steps", 0)
    )
    if accepted_steps <= 0:
        errors.append(f"{mode} did not report any accepted fixed BDF2 intervals")
    if int(diagnostics.get("fixed_bdf2_bdf2_steps", 0)) <= 0:
        errors.append(f"{mode} did not report any actual fixed BDF2 corrector steps")
    max_residual = diagnostics.get("fixed_bdf2_max_residual_inf_norm")
    try:
        finite_residual = max_residual is not None and np.isfinite(float(max_residual))
    except (TypeError, ValueError):
        finite_residual = False
    if not finite_residual:
        errors.append(f"{mode} did not report a finite fixed BDF2 residual norm")
    elif max_residual_inf_norm is not None:
        residual_float = float(max_residual)
        if residual_float > float(max_residual_inf_norm):
            errors.append(
                f"{mode} fixed_bdf2_max_residual_inf_norm={residual_float:.8e} exceeds {float(max_residual_inf_norm):.8e}"
            )
    if required_linear_preconditioner is not None:
        errors.extend(
            _validate_required_linear_preconditioner(
                mode,
                diagnostics,
                required_linear_preconditioner=required_linear_preconditioner,
                name_key="fixed_bdf2_linear_preconditioner",
                count_key="fixed_bdf2_total_linear_preconditioner_build_count",
                seconds_key="fixed_bdf2_total_linear_preconditioner_build_seconds",
            )
        )
    if required_linear_solver_backend is not None:
        errors.extend(
            _validate_required_linear_solver_backend(
                mode,
                diagnostics,
                required_linear_solver_backend=required_linear_solver_backend,
                key="fixed_bdf2_linear_solver_backend",
            )
        )
    if require_linear_operator_jitted:
        linearized_steps = int(diagnostics.get("fixed_bdf2_jax_linearized_action_steps", 0))
        jitted_steps = int(diagnostics.get("fixed_bdf2_linear_operator_jitted_steps", 0))
        if linearized_steps <= 0 or jitted_steps != linearized_steps:
            errors.append(
                f"{mode} did not report JIT-wrapped linear operators on every "
                f"JAX-linearized step: {jitted_steps}/{linearized_steps}"
            )
    if required_line_search_mode is not None:
        expected_mode = _canonical_line_search_mode(required_line_search_mode)
        reported_mode = diagnostics.get("fixed_bdf2_line_search_mode")
        reported_name = (
            None
            if reported_mode is None
            else _canonical_line_search_mode(str(reported_mode))
        )
        if reported_name != expected_mode:
            errors.append(
                f"{mode} did not report fixed_bdf2_line_search_mode={expected_mode}; "
                f"reported {reported_name}"
            )
    if max_linear_iterations is not None:
        errors.extend(
            _validate_maximum_integer_diagnostic(
                mode,
                diagnostics,
                key="fixed_bdf2_total_linear_iterations",
                maximum=int(max_linear_iterations),
                label="fixed BDF2 linear iterations",
            )
        )
    if max_residual_evaluations is not None:
        errors.extend(
            _validate_maximum_integer_diagnostic(
                mode,
                diagnostics,
                key="fixed_bdf2_total_residual_evaluation_count",
                maximum=int(max_residual_evaluations),
                label="fixed BDF2 residual evaluations",
            )
        )
    if max_linear_operator_calls is not None:
        errors.extend(
            _validate_maximum_integer_diagnostic(
                mode,
                diagnostics,
                key="fixed_bdf2_total_linear_operator_call_count",
                maximum=int(max_linear_operator_calls),
                label="fixed BDF2 linear operator calls",
            )
        )
    if min_linear_solve_count is not None:
        errors.extend(
            _validate_minimum_integer_diagnostic(
                mode,
                diagnostics,
                key="fixed_bdf2_total_linear_solve_count",
                minimum=int(min_linear_solve_count),
                label="fixed BDF2 linear solve attempts",
            )
        )
    else:
        errors.extend(
            _validate_minimum_integer_diagnostic(
                mode,
                diagnostics,
                key="fixed_bdf2_total_linear_solve_count",
                minimum=1,
                label="fixed BDF2 linear solve attempts",
            )
        )
    if max_linear_update_residual_inf_norm is not None:
        errors.extend(
            _validate_maximum_float_diagnostic(
                mode,
                diagnostics,
                key="fixed_bdf2_max_linear_update_residual_inf_norm",
                maximum=float(max_linear_update_residual_inf_norm),
                label="fixed BDF2 linear-update residual inf-norm",
            )
        )
    if max_linear_update_relative_residual is not None:
        errors.extend(
            _validate_maximum_float_diagnostic(
                mode,
                diagnostics,
                key="fixed_bdf2_max_linear_update_relative_residual",
                maximum=float(max_linear_update_relative_residual),
                label="fixed BDF2 linear-update relative residual",
            )
        )
    if max_preconditioner_builds is not None:
        errors.extend(
            _validate_maximum_integer_diagnostic(
                mode,
                diagnostics,
                key="fixed_bdf2_total_linear_preconditioner_build_count",
                maximum=int(max_preconditioner_builds),
                label="fixed BDF2 preconditioner builds",
            )
        )
    if max_preconditioner_applies is not None:
        errors.extend(
            _validate_maximum_integer_diagnostic(
                mode,
                diagnostics,
                key="fixed_bdf2_total_linear_preconditioner_apply_count",
                maximum=int(max_preconditioner_applies),
                label="fixed BDF2 preconditioner applies",
            )
        )
    return errors


def _adaptive_bdf_modes_to_validate(modes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(mode for mode in modes if mode in ADAPTIVE_BDF_MODES)


def _validate_adaptive_bdf_diagnostics(
    mode: str,
    diagnostics: dict[str, object],
    *,
    require_no_fallback: bool,
    require_no_unconverged_substeps: bool,
    max_error_ratio: float | None,
    max_accepted_error_ratio: float | None,
    max_linear_update_residual_inf_norm: float | None = None,
    max_linear_update_relative_residual: float | None = None,
    required_linear_preconditioner: str | None = None,
) -> list[str]:
    errors: list[str] = []
    expected_step_solver = ADAPTIVE_BDF_STEP_SOLVER_MODES[mode]
    if diagnostics.get("adaptive_bdf_step_solver_mode") != expected_step_solver:
        errors.append(
            f"{mode} did not report adaptive_bdf_step_solver_mode={expected_step_solver}"
        )
    if int(diagnostics.get("adaptive_bdf_interval_count", 0)) <= 0:
        errors.append(f"{mode} did not report any adaptive BDF output intervals")
    if int(diagnostics.get("adaptive_bdf_accepted_steps", 0)) <= 0:
        errors.append(f"{mode} did not report any accepted adaptive BDF substeps")
    expected_rhs_backend = ADAPTIVE_BDF_RHS_BACKENDS[mode]
    if expected_rhs_backend == "fixed_full_field_array":
        rhs_steps = int(
            diagnostics.get("adaptive_bdf_fixed_full_field_rhs_solver_steps", 0)
        )
        if rhs_steps <= 0:
            errors.append(
                f"{mode} did not report any fixed_full_field_array adaptive BDF solver steps"
            )
    elif expected_rhs_backend == "active_array":
        rhs_steps = int(
            diagnostics.get("adaptive_bdf_active_array_rhs_solver_steps", 0)
        )
        if rhs_steps <= 0:
            errors.append(
                f"{mode} did not report any active_array adaptive BDF solver steps"
            )
    if mode == "adaptive_bdf_sparse_jvp":
        sparse_jvp_steps = int(
            diagnostics.get("adaptive_bdf_sparse_jvp_jacobian_solver_steps", 0)
        )
        if sparse_jvp_steps <= 0:
            errors.append(
                f"{mode} did not report any sparse-JVP Jacobian adaptive BDF solver steps"
            )
    elif "jax_linearized" in expected_step_solver:
        action_steps = int(
            diagnostics.get("adaptive_bdf_jax_linearized_action_solver_steps", 0)
        )
        if action_steps <= 0:
            errors.append(
                f"{mode} did not report any JAX-linearized adaptive BDF solver steps"
            )
        if expected_step_solver == "jax_linearized_lineax":
            lineax_steps = int(
                diagnostics.get("adaptive_bdf_lineax_action_solver_steps", 0)
            )
            if lineax_steps <= 0:
                errors.append(
                    f"{mode} did not report any Lineax adaptive BDF solver steps"
                )
    fallback_count = int(diagnostics.get("adaptive_bdf_minimum_dt_fallbacks", 0))
    if require_no_fallback and fallback_count != 0:
        errors.append(f"{mode} reported {fallback_count} minimum-dt fallback accepts")
    unconverged_count = int(diagnostics.get("adaptive_bdf_unconverged_solver_steps", 0))
    if require_no_unconverged_substeps and unconverged_count != 0:
        errors.append(
            f"{mode} reported {unconverged_count} unconverged adaptive BDF implicit substeps"
        )
    linear_solver_failed_count = int(
        diagnostics.get("adaptive_bdf_linear_solver_failed_steps", 0)
    )
    if require_no_unconverged_substeps and linear_solver_failed_count != 0:
        errors.append(
            f"{mode} reported {linear_solver_failed_count} failed adaptive BDF linear solves"
        )
    if max_error_ratio is not None:
        reported = diagnostics.get("adaptive_bdf_max_error_ratio")
        if reported is None:
            errors.append(f"{mode} did not report adaptive_bdf_max_error_ratio")
        else:
            reported_float = float(reported)
            if not np.isfinite(reported_float) or reported_float > float(
                max_error_ratio
            ):
                errors.append(
                    f"{mode} adaptive_bdf_max_error_ratio={reported_float:.8e} exceeds {float(max_error_ratio):.8e}"
                )
    if max_accepted_error_ratio is not None:
        reported = diagnostics.get("adaptive_bdf_max_accepted_error_ratio")
        if reported is None:
            errors.append(
                f"{mode} did not report adaptive_bdf_max_accepted_error_ratio"
            )
        else:
            reported_float = float(reported)
            if not np.isfinite(reported_float) or reported_float > float(
                max_accepted_error_ratio
            ):
                errors.append(
                    f"{mode} adaptive_bdf_max_accepted_error_ratio={reported_float:.8e} exceeds {float(max_accepted_error_ratio):.8e}"
                )
    if required_linear_preconditioner is not None:
        errors.extend(
            _validate_required_linear_preconditioner(
                mode,
                diagnostics,
                required_linear_preconditioner=required_linear_preconditioner,
                name_key="adaptive_bdf_linear_preconditioner",
                count_key="adaptive_bdf_linear_preconditioner_build_count",
                seconds_key="adaptive_bdf_linear_preconditioner_build_seconds",
            )
        )
    if max_linear_update_residual_inf_norm is not None:
        errors.extend(
            _validate_maximum_float_diagnostic(
                mode,
                diagnostics,
                key="adaptive_bdf_max_linear_update_residual_inf_norm",
                maximum=float(max_linear_update_residual_inf_norm),
                label="adaptive BDF linear-update residual inf-norm",
            )
        )
    if max_linear_update_relative_residual is not None:
        errors.extend(
            _validate_maximum_float_diagnostic(
                mode,
                diagnostics,
                key="adaptive_bdf_max_linear_update_relative_residual",
                maximum=float(max_linear_update_relative_residual),
                label="adaptive BDF linear-update relative residual",
            )
        )
    return errors


def _canonical_preconditioner_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "parallel_transport": "parallel_line",
        "transport_line": "parallel_line",
        "neutral_diffusion": "neutral_line",
        "neutral_transport": "neutral_line",
        "parallel_momentum": "momentum_line",
        "momentum_transport": "momentum_line",
        "target_line": "sheath_line",
        "target_sheath": "sheath_line",
        "target_sheath_line": "sheath_line",
        "sheath_transport": "sheath_line",
        "line_field_schur": "field_line_schur",
        "transport_field_schur": "field_line_schur",
        "field_transport_schur": "field_line_schur",
        "sheath_schur": "target_schur",
        "plasma_neutral_schur": "neutral_plasma_schur",
    }
    return aliases.get(normalized, normalized)


def _preconditioner_requires_dynamic_build(name: str | None) -> bool:
    """Return whether a requested preconditioner should report JVP-build work."""

    normalized = _canonical_preconditioner_name(str(name or ""))
    return normalized not in {"state_scale", "field_scale"}


def _canonical_linear_solver_backend(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "jax": "jax_gmres",
        "jax_scipy": "jax_gmres",
        "gmres": "jax_gmres",
        "jax_gmres": "jax_gmres",
        "bicgstab": "jax_bicgstab",
        "jax_bicgstab": "jax_bicgstab",
        "lineax": "lineax_gmres",
        "lineax_gmres": "lineax_gmres",
    }
    return aliases.get(normalized, normalized)


def _canonical_line_search_mode(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "": "backtracking",
        "default": "backtracking",
        "backtrack": "backtracking",
        "backtracking": "backtracking",
        "line_search": "backtracking",
        "full": "full_step",
        "fullstep": "full_step",
        "full_step": "full_step",
        "none": "full_step",
        "off": "full_step",
    }
    return aliases.get(normalized, normalized)


def _validate_required_linear_solver_backend(
    mode: str,
    diagnostics: dict[str, object],
    *,
    required_linear_solver_backend: str,
    key: str,
) -> list[str]:
    expected = _canonical_linear_solver_backend(required_linear_solver_backend)
    reported = diagnostics.get(key)
    reported_name = (
        None if reported is None else _canonical_linear_solver_backend(str(reported))
    )
    if reported_name != expected:
        return [
            f"{mode} did not report {key}={expected}; reported {reported_name}"
        ]
    return []


def _validate_required_linear_preconditioner(
    mode: str,
    diagnostics: dict[str, object],
    *,
    required_linear_preconditioner: str,
    name_key: str,
    count_key: str,
    seconds_key: str,
) -> list[str]:
    expected = _canonical_preconditioner_name(required_linear_preconditioner)
    reported = diagnostics.get(name_key)
    reported_name = (
        None if reported is None else _canonical_preconditioner_name(str(reported))
    )
    errors: list[str] = []
    if reported_name != expected:
        errors.append(f"{mode} did not report {name_key}={expected}")
    try:
        build_count = int(diagnostics.get(count_key, 0))
    except (TypeError, ValueError):
        build_count = 0
    if build_count < 0:
        errors.append(
            f"{mode} reported negative {expected} preconditioner build count"
        )
    elif _preconditioner_requires_dynamic_build(expected) and build_count <= 0:
        errors.append(f"{mode} did not report any {expected} preconditioner builds")
    try:
        build_seconds = float(diagnostics.get(seconds_key, float("nan")))
    except (TypeError, ValueError):
        build_seconds = float("nan")
    if not np.isfinite(build_seconds) or build_seconds < 0.0:
        errors.append(f"{mode} did not report finite nonnegative {seconds_key}")
    return errors


def _validate_maximum_integer_diagnostic(
    mode: str,
    diagnostics: dict[str, object],
    *,
    key: str,
    maximum: int,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if int(maximum) < 0:
        errors.append(f"{mode} received a negative {label} gate")
        return errors
    try:
        reported = int(diagnostics[key])
    except KeyError:
        errors.append(f"{mode} did not report {key}")
        return errors
    except (TypeError, ValueError):
        errors.append(f"{mode} did not report an integer {key}")
        return errors
    if reported > int(maximum):
        errors.append(
            f"{mode} reported {reported} {label}, exceeding {int(maximum)}"
        )
    return errors


def _validate_minimum_integer_diagnostic(
    mode: str,
    diagnostics: dict[str, object],
    *,
    key: str,
    minimum: int,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if int(minimum) < 0:
        errors.append(f"{mode} received a negative {label} gate")
        return errors
    try:
        reported = int(diagnostics[key])
    except KeyError:
        errors.append(f"{mode} did not report {key}")
        return errors
    except (TypeError, ValueError):
        errors.append(f"{mode} did not report an integer {key}")
        return errors
    if reported < int(minimum):
        errors.append(f"{mode} reported {reported} {label}, below {int(minimum)}")
    return errors


def _validate_maximum_float_diagnostic(
    mode: str,
    diagnostics: dict[str, object],
    *,
    key: str,
    maximum: float,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if not np.isfinite(float(maximum)) or float(maximum) < 0.0:
        errors.append(f"{mode} received an invalid {label} gate")
        return errors
    try:
        reported = float(diagnostics[key])
    except KeyError:
        errors.append(f"{mode} did not report {key}")
        return errors
    except (TypeError, ValueError):
        errors.append(f"{mode} did not report a finite {key}")
        return errors
    if not np.isfinite(reported):
        errors.append(f"{mode} did not report a finite {key}")
        return errors
    if reported > float(maximum):
        errors.append(
            f"{mode} reported {reported:.8e} {label}, "
            f"exceeding {float(maximum):.8e}"
        )
    return errors


class _ModeTimeoutError(TimeoutError):
    pass


def _run_with_mode_timeout(timeout_seconds: float | None, callback):
    if timeout_seconds is None:
        return callback()
    if timeout_seconds <= 0.0:
        raise ValueError("--mode-timeout-seconds must be positive.")
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        return callback()

    def _handle_timeout(_signum, _frame):
        raise _ModeTimeoutError(
            f"solver mode exceeded {float(timeout_seconds):g} seconds"
        )

    previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        return callback()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def main() -> int:
    args = _parse_args()
    if (
        args.require_fixed_bdf2_max_residual is not None
        and float(args.require_fixed_bdf2_max_residual) <= 0.0
    ):
        raise ValueError("--require-fixed-bdf2-max-residual must be positive.")
    if (
        args.require_fixed_bdf2_max_linear_iterations is not None
        and int(args.require_fixed_bdf2_max_linear_iterations) < 0
    ):
        raise ValueError(
            "--require-fixed-bdf2-max-linear-iterations must be nonnegative."
        )
    if (
        args.require_fixed_bdf2_max_linear_operator_calls is not None
        and int(args.require_fixed_bdf2_max_linear_operator_calls) < 0
    ):
        raise ValueError(
            "--require-fixed-bdf2-max-linear-operator-calls must be nonnegative."
        )
    if (
        args.require_fixed_bdf2_min_linear_solve_count is not None
        and int(args.require_fixed_bdf2_min_linear_solve_count) < 0
    ):
        raise ValueError(
            "--require-fixed-bdf2-min-linear-solve-count must be nonnegative."
        )
    if args.require_fixed_bdf2_max_linear_update_residual is not None:
        value = float(args.require_fixed_bdf2_max_linear_update_residual)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "--require-fixed-bdf2-max-linear-update-residual must be finite "
                "and nonnegative."
            )
    if args.require_fixed_bdf2_max_linear_update_relative_residual is not None:
        value = float(args.require_fixed_bdf2_max_linear_update_relative_residual)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "--require-fixed-bdf2-max-linear-update-relative-residual must be "
                "finite and nonnegative."
            )
    if (
        args.require_fixed_bdf2_max_preconditioner_builds is not None
        and int(args.require_fixed_bdf2_max_preconditioner_builds) < 0
    ):
        raise ValueError(
            "--require-fixed-bdf2-max-preconditioner-builds must be nonnegative."
        )
    if (
        args.require_fixed_bdf2_max_preconditioner_applies is not None
        and int(args.require_fixed_bdf2_max_preconditioner_applies) < 0
    ):
        raise ValueError(
            "--require-fixed-bdf2-max-preconditioner-applies must be nonnegative."
        )
    if args.require_adaptive_bdf_max_linear_update_residual is not None:
        value = float(args.require_adaptive_bdf_max_linear_update_residual)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "--require-adaptive-bdf-max-linear-update-residual must be finite "
                "and nonnegative."
            )
    if args.require_adaptive_bdf_max_linear_update_relative_residual is not None:
        value = float(args.require_adaptive_bdf_max_linear_update_relative_residual)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "--require-adaptive-bdf-max-linear-update-relative-residual must be "
                "finite and nonnegative."
            )
    if args.require_bdf_pairwise_max is not None:
        value = float(args.require_bdf_pairwise_max)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "--require-bdf-pairwise-max must be finite and nonnegative."
            )
    if args.require_fixed_bdf2_pairwise_max is not None:
        value = float(args.require_fixed_bdf2_pairwise_max)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "--require-fixed-bdf2-pairwise-max must be finite and nonnegative."
            )
    if args.require_fixed_bdf2_pairwise_l2_rel_max is not None:
        value = float(args.require_fixed_bdf2_pairwise_l2_rel_max)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "--require-fixed-bdf2-pairwise-l2-rel-max must be finite and nonnegative."
            )
    if args.require_fixed_bdf2_pairwise_inventory_rel_max is not None:
        value = float(args.require_fixed_bdf2_pairwise_inventory_rel_max)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "--require-fixed-bdf2-pairwise-inventory-rel-max must be finite and nonnegative."
            )
    case, input_path = resolve_reference_case(
        args.case, reference_root=args.reference_root
    )
    config = _load_curated_case_config(case, input_path)
    if args.overrides:
        config = apply_bout_overrides(config, args.overrides)
    run_config = RunConfiguration.from_config(config)
    output_timestep = _resolve_output_timestep(args, run_config)
    max_nonlinear_iterations = _resolve_max_nonlinear_iterations(args)
    steps = _resolve_steps(args)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    dataset_scalars = resolved_dataset_scalars(run_config)
    expected = None
    if not args.diagnostics_only:
        expected = load_portable_array_payload(
            Path(__file__).resolve().parents[1]
            / "references"
            / "baselines"
            / "reference_arrays"
            / f"{args.case}.npz"
        )["variables"]
    fields = tuple(args.fields) if args.fields else DEFAULT_FIELDS
    modes = tuple(args.modes) if args.modes else _default_modes(args.case)
    rtol = (
        float(config.parsed("solver", "rtol"))
        if config.has_option("solver", "rtol")
        else 1.0e-8
    )

    print(f"case={args.case}", flush=True)
    print(f"configured_timestep={run_config.time.timestep:g}", flush=True)
    print(f"timestep={output_timestep:g}", flush=True)
    print(f"steps={steps}", flush=True)
    print(f"max_nonlinear_iterations={max_nonlinear_iterations}", flush=True)
    if args.diagnostics_only:
        print("baseline_comparison=disabled", flush=True)
    elif args.timestep is not None and not np.isclose(
        output_timestep, run_config.time.timestep
    ):
        print(
            "warning=timestep override is active while committed-baseline comparison remains enabled",
            flush=True,
        )
    print(f"fields={fields}", flush=True)
    print(f"modes={modes}", flush=True)
    mode_variables: dict[str, dict[str, np.ndarray]] = {}
    mode_diagnostics: dict[str, dict[str, object]] = {}
    mode_elapsed_seconds: dict[str, float] = {}
    for mode in modes:
        print(f"running_mode={mode}", flush=True)
        started = time.perf_counter()
        try:
            history = _run_with_mode_timeout(
                args.mode_timeout_seconds,
                lambda: advance_recycling_1d_implicit_history(
                    config,
                    mesh=mesh,
                    metrics=metrics,
                    dataset_scalars=dataset_scalars,
                    timestep=output_timestep,
                    steps=steps,
                    solver_mode=mode,
                    residual_tolerance=rtol,
                    max_nonlinear_iterations=max_nonlinear_iterations,
                ),
            )
        except _ModeTimeoutError as exc:
            print(f"gate_failure={mode} timed out: {exc}", flush=True)
            return 2
        elapsed = time.perf_counter() - started
        mode_elapsed_seconds[mode] = float(elapsed)
        actual = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in history.variable_history.items()
        }
        mode_variables[mode] = actual
        mode_diagnostics[mode] = dict(history.diagnostics)
        rows = []
        if expected is not None:
            rows = _summarize_mode_errors(actual, expected, fields=fields, mesh=mesh)
        for line in _format_mode_error_report(mode, elapsed=elapsed, rows=rows):
            print(line)
        for line in _format_mode_diagnostics_report(mode, mode_diagnostics[mode]):
            print(line)
    for line in _format_bdf_pairwise_delta_report(
        mode_variables, fields=fields, mesh=mesh
    ):
        print(line)
    for line in _format_fixed_bdf2_pairwise_delta_report(
        mode_variables, fields=fields, mesh=mesh
    ):
        print(line)
    for line in _format_fixed_bdf2_pairwise_observable_report(
        mode_variables, fields=fields, mesh=mesh
    ):
        print(line)
    adaptive_gate_errors: dict[str, list[str]] = {}
    if args.require_fixed_jvp_diagnostics:
        errors = []
        bdf_jvp_modes = tuple(mode for mode in BDF_PAIRWISE_CANDIDATE_MODES if mode in modes)
        if not bdf_jvp_modes:
            errors.append(
                "fixed BDF JVP diagnostics were requested but no BDF JVP mode was run"
            )
        for mode in bdf_jvp_modes:
            errors.extend(
                _validate_bdf_jvp_diagnostics(
                    mode,
                    mode_diagnostics.get(mode, {}),
                )
            )
        for error in errors:
            print(f"gate_failure={error}")
        if errors:
            return 2
    if args.require_fixed_bdf2_diagnostics:
        fixed_bdf2_modes = _fixed_bdf2_modes_to_validate(modes)
        if not fixed_bdf2_modes:
            print(
                "gate_failure=fixed BDF2 diagnostics were requested but no fixed-BDF2 mode was run"
            )
            return 2
        errors = []
        for mode in fixed_bdf2_modes:
            errors.extend(
                _validate_fixed_bdf2_diagnostics(
                    mode,
                    mode_diagnostics.get(mode, {}),
                    max_residual_inf_norm=float(
                        args.require_fixed_bdf2_max_residual
                    ),
                    required_linear_preconditioner=(
                        args.require_fixed_bdf2_linear_preconditioner
                    ),
                    required_linear_solver_backend=(
                        args.require_fixed_bdf2_linear_solver_backend
                    ),
                    require_linear_operator_jitted=bool(
                        args.require_fixed_bdf2_linear_operator_jitted
                    ),
                    required_line_search_mode=(
                        args.require_fixed_bdf2_line_search_mode
                    ),
                    max_linear_iterations=(
                        args.require_fixed_bdf2_max_linear_iterations
                    ),
                    max_residual_evaluations=(
                        args.require_fixed_bdf2_max_residual_evaluations
                    ),
                    max_linear_operator_calls=(
                        args.require_fixed_bdf2_max_linear_operator_calls
                    ),
                    min_linear_solve_count=(
                        args.require_fixed_bdf2_min_linear_solve_count
                    ),
                    max_linear_update_residual_inf_norm=(
                        args.require_fixed_bdf2_max_linear_update_residual
                    ),
                    max_linear_update_relative_residual=(
                        args.require_fixed_bdf2_max_linear_update_relative_residual
                    ),
                    max_preconditioner_builds=(
                        args.require_fixed_bdf2_max_preconditioner_builds
                    ),
                    max_preconditioner_applies=(
                        args.require_fixed_bdf2_max_preconditioner_applies
                    ),
                )
            )
        for error in errors:
            print(f"gate_failure={error}")
        if errors:
            return 2
    if (
        args.require_adaptive_bdf_no_fallback
        or args.require_adaptive_bdf_no_unconverged_substeps
        or args.require_adaptive_bdf_max_error_ratio is not None
        or args.require_adaptive_bdf_max_accepted_error_ratio is not None
        or args.require_adaptive_bdf_max_linear_update_residual is not None
        or args.require_adaptive_bdf_max_linear_update_relative_residual is not None
        or args.require_adaptive_bdf_linear_preconditioner is not None
    ):
        adaptive_modes = _adaptive_bdf_modes_to_validate(modes)
        if not adaptive_modes:
            print(
                "gate_failure=adaptive BDF diagnostics were requested but no adaptive-BDF mode was run"
            )
            return 2
        errors = []
        for mode in adaptive_modes:
            mode_errors = _validate_adaptive_bdf_diagnostics(
                mode,
                mode_diagnostics.get(mode, {}),
                require_no_fallback=bool(args.require_adaptive_bdf_no_fallback),
                require_no_unconverged_substeps=bool(
                    args.require_adaptive_bdf_no_unconverged_substeps
                ),
                max_error_ratio=args.require_adaptive_bdf_max_error_ratio,
                max_accepted_error_ratio=args.require_adaptive_bdf_max_accepted_error_ratio,
                max_linear_update_residual_inf_norm=(
                    args.require_adaptive_bdf_max_linear_update_residual
                ),
                max_linear_update_relative_residual=(
                    args.require_adaptive_bdf_max_linear_update_relative_residual
                ),
                required_linear_preconditioner=(
                    args.require_adaptive_bdf_linear_preconditioner
                ),
            )
            adaptive_gate_errors[mode] = mode_errors
            errors.extend(mode_errors)
        for error in errors:
            print(f"gate_failure={error}")
        if errors:
            return 2
    bdf_pairwise_worst = _bdf_pairwise_worst_delta(
        mode_variables, fields=fields, mesh=mesh
    )
    fixed_bdf2_pairwise_worst = _fixed_bdf2_pairwise_worst_delta(
        mode_variables, fields=fields, mesh=mesh
    )
    fixed_bdf2_pairwise_observable_worst = _fixed_bdf2_pairwise_observable_worst(
        mode_variables, fields=fields, mesh=mesh
    )
    if args.output_json is not None:
        report = _build_json_report(
            case_name=args.case,
            configured_timestep=run_config.time.timestep,
            timestep=output_timestep,
            max_nonlinear_iterations=max_nonlinear_iterations,
            steps=steps,
            fields=fields,
            modes=modes,
            diagnostics_only=bool(args.diagnostics_only),
            mode_elapsed_seconds=mode_elapsed_seconds,
            mode_diagnostics=mode_diagnostics,
            bdf_pairwise_worst=bdf_pairwise_worst,
            fixed_bdf2_pairwise_worst=fixed_bdf2_pairwise_worst,
            fixed_bdf2_pairwise_observable_worst=(
                fixed_bdf2_pairwise_observable_worst
            ),
            adaptive_bdf_gate_errors=adaptive_gate_errors,
        )
        _write_json_report(args.output_json, report)
        print(f"output_json={args.output_json}")
    if args.require_bdf_pairwise_max is not None:
        worst_field, worst_delta = bdf_pairwise_worst
        if worst_delta is None:
            print(
                "gate_failure=bdf pairwise delta is unavailable; run bdf and at least one BDF JVP candidate"
            )
            return 2
        threshold = float(args.require_bdf_pairwise_max)
        print(
            f"gate=bdf_pairwise_max field={worst_field} delta={worst_delta:.8e} threshold={threshold:.8e}"
        )
        if not np.isfinite(worst_delta) or worst_delta > threshold:
            print("gate_failure=bdf pairwise delta exceeds threshold")
            return 2
    if args.require_fixed_bdf2_pairwise_max is not None:
        worst_field, worst_delta = fixed_bdf2_pairwise_worst
        if worst_delta is None:
            print(
                "gate_failure=fixed BDF2 pairwise delta is unavailable; run "
                "bdf and at least one fixed-BDF2 candidate"
            )
            return 2
        threshold = float(args.require_fixed_bdf2_pairwise_max)
        print(
            f"gate=fixed_bdf2_pairwise_max field={worst_field} "
            f"delta={worst_delta:.8e} threshold={threshold:.8e}"
        )
        if not np.isfinite(worst_delta) or worst_delta > threshold:
            print("gate_failure=fixed BDF2 pairwise delta exceeds threshold")
            return 2
    if args.require_fixed_bdf2_pairwise_l2_rel_max is not None:
        l2_worst = fixed_bdf2_pairwise_observable_worst["l2_relative"]
        worst_delta = l2_worst["delta"]
        if worst_delta is None:
            print(
                "gate_failure=fixed BDF2 pairwise L2 relative delta is unavailable; "
                "run bdf and at least one fixed-BDF2 candidate"
            )
            return 2
        threshold = float(args.require_fixed_bdf2_pairwise_l2_rel_max)
        print(
            f"gate=fixed_bdf2_pairwise_l2_relative_max "
            f"mode={l2_worst['mode']} field={l2_worst['field']} "
            f"delta={float(worst_delta):.8e} threshold={threshold:.8e}"
        )
        if not np.isfinite(float(worst_delta)) or float(worst_delta) > threshold:
            print("gate_failure=fixed BDF2 pairwise L2 relative delta exceeds threshold")
            return 2
    if args.require_fixed_bdf2_pairwise_inventory_rel_max is not None:
        inventory_worst = fixed_bdf2_pairwise_observable_worst[
            "inventory_relative"
        ]
        worst_delta = inventory_worst["delta"]
        if worst_delta is None:
            print(
                "gate_failure=fixed BDF2 pairwise inventory relative delta is unavailable; "
                "run bdf and at least one fixed-BDF2 candidate"
            )
            return 2
        threshold = float(args.require_fixed_bdf2_pairwise_inventory_rel_max)
        print(
            f"gate=fixed_bdf2_pairwise_inventory_relative_max "
            f"mode={inventory_worst['mode']} field={inventory_worst['field']} "
            f"delta={float(worst_delta):.8e} threshold={threshold:.8e}"
        )
        if not np.isfinite(float(worst_delta)) or float(worst_delta) > threshold:
            print(
                "gate_failure=fixed BDF2 pairwise inventory relative delta exceeds threshold"
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
