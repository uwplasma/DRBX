#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def _parse_args() -> argparse.Namespace:
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
        choices=("continuation", "bdf", "adaptive_be", "adaptive_bdf"),
        help="Solver modes to compare. May be repeated. Defaults to the main supported set for the case.",
    )
    parser.add_argument("--field", action="append", dest="fields", help="Fields to report. May be repeated.")
    return parser.parse_args()


def _default_modes(case_name: str) -> tuple[str, ...]:
    if case_name == "recycling_1d_one_step":
        return ("continuation", "bdf", "adaptive_be", "adaptive_bdf")
    return ("bdf", "adaptive_be", "adaptive_bdf")


def _summarize_mode_errors(
    actual_variables: dict[str, np.ndarray],
    expected_variables: dict[str, np.ndarray],
    *,
    fields: tuple[str, ...],
    mesh=None,
) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for field in fields:
        if field not in actual_variables or field not in expected_variables:
            continue
        actual = np.asarray(actual_variables[field], dtype=np.float64)
        expected = np.asarray(expected_variables[field], dtype=np.float64)
        if mesh is not None and actual.ndim == 4 and expected.ndim == 4:
            if actual.shape[1] >= mesh.xend + 1 and actual.shape[2] >= mesh.yend + 1:
                actual = actual[:, mesh.xstart : mesh.xend + 1, mesh.ystart : mesh.yend + 1, :]
        if actual.shape != expected.shape:
            rows.append((field, float("inf")))
            continue
        rows.append((field, float(np.nanmax(np.abs(actual - expected)))))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


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
        rows = _summarize_mode_errors(actual, expected, fields=fields, mesh=mesh)
        print(f"mode={mode} elapsed={elapsed:.3f}s")
        for field, max_abs in rows:
            print(f"  {field}: max_abs_diff={max_abs:.8e}")
        if rows:
            print(f"  worst={rows[0][0]} diff={rows[0][1]:.8e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
