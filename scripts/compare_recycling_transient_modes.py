#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import time

import numpy as np

from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.recycling_1d import advance_recycling_1d_implicit_history
from jax_drb.native.runner import _load_curated_case_config
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.parity.arrays import load_portable_array_payload
from jax_drb.parity.reference import resolve_reference_case
from jax_drb.runtime.run_config import RunConfiguration


DEFAULT_FIELDS = ("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe")
SOLVER_MODES = ("continuation", "bdf", "bdf_fixed_full_field_jvp", "adaptive_be", "adaptive_bdf")
BDF_PAIRWISE_MODES = ("bdf", "bdf_fixed_full_field_jvp")


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
            "to exercise the fixed full-field JVP BDF path; defaults include the main "
            "supported set for the case."
        ),
    )
    parser.add_argument("--field", action="append", dest="fields", help="Fields to report. May be repeated.")
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    return parser.parse_args(argv)


def _default_modes(case_name: str) -> tuple[str, ...]:
    if case_name == "recycling_1d_one_step":
        return ("continuation", "bdf", "bdf_fixed_full_field_jvp", "adaptive_be", "adaptive_bdf")
    return ("bdf", "bdf_fixed_full_field_jvp", "adaptive_be", "adaptive_bdf")


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


def _active_mesh_view(values: np.ndarray, mesh) -> np.ndarray:
    if mesh is None or values.ndim != 4:
        return values
    if values.shape[1] >= mesh.xend + 1 and values.shape[2] >= mesh.yend + 1:
        return values[:, mesh.xstart : mesh.xend + 1, mesh.ystart : mesh.yend + 1, :]
    return values


def _format_mode_error_report(mode: str, *, elapsed: float, rows: list[tuple[str, float]]) -> list[str]:
    lines = [f"mode={mode} elapsed={elapsed:.3f}s"]
    for field, max_abs in rows:
        lines.append(f"  {field}: max_abs_diff={max_abs:.8e}")
    if rows:
        lines.append(f"  worst={rows[0][0]} diff={rows[0][1]:.8e}")
    return lines


def _format_bdf_pairwise_delta_report(
    mode_variables: dict[str, dict[str, np.ndarray]],
    *,
    fields: tuple[str, ...],
    mesh=None,
) -> list[str]:
    base_mode, candidate_mode = BDF_PAIRWISE_MODES
    if base_mode not in mode_variables or candidate_mode not in mode_variables:
        return []

    rows = _summarize_mode_errors(
        mode_variables[base_mode],
        mode_variables[candidate_mode],
        fields=fields,
        mesh=mesh,
        crop_expected=True,
    )
    lines = [f"pairwise_delta={base_mode}_vs_{candidate_mode}"]
    for field, max_abs in rows:
        lines.append(f"  {field}: max_abs_delta={max_abs:.8e}")
    if rows:
        lines.append(f"  worst={rows[0][0]} delta={rows[0][1]:.8e}")
    return lines


def main() -> int:
    args = _parse_args()
    case, input_path = resolve_reference_case(args.case, reference_root=args.reference_root)
    config = _load_curated_case_config(case, input_path)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    dataset_scalars = resolved_dataset_scalars(run_config)
    expected = load_portable_array_payload(
        Path(__file__).resolve().parents[1] / "references" / "baselines" / "reference_arrays" / f"{args.case}.npz"
    )["variables"]
    fields = tuple(args.fields) if args.fields else DEFAULT_FIELDS
    modes = tuple(args.modes) if args.modes else _default_modes(args.case)
    rtol = float(config.parsed("solver", "rtol")) if config.has_option("solver", "rtol") else 1.0e-8

    print(f"case={args.case}")
    print(f"timestep={run_config.time.timestep:g}")
    print(f"fields={fields}")
    print(f"modes={modes}")
    mode_variables: dict[str, dict[str, np.ndarray]] = {}
    for mode in modes:
        started = time.perf_counter()
        history = advance_recycling_1d_implicit_history(
            config,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            timestep=run_config.time.timestep,
            steps=1,
            solver_mode=mode,
            residual_tolerance=rtol,
            max_nonlinear_iterations=30,
        )
        elapsed = time.perf_counter() - started
        actual = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in history.variable_history.items()
        }
        mode_variables[mode] = actual
        rows = _summarize_mode_errors(actual, expected, fields=fields, mesh=mesh)
        for line in _format_mode_error_report(mode, elapsed=elapsed, rows=rows):
            print(line)
    for line in _format_bdf_pairwise_delta_report(mode_variables, fields=fields, mesh=mesh):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
