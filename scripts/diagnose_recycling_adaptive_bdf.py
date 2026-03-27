#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.recycling_1d import (
    _build_recycling_runtime_model,
    _current_feedback_errors,
    _recycling_field_templates,
    _recycling_state_error_ratio,
    advance_recycling_1d_bdf2_step,
    advance_recycling_1d_backward_euler_step,
)
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.runtime.run_config import RunConfiguration


DEFAULT_WORKDIRS = {
    "recycling_1d_one_step": Path("/private/tmp/jax_drb_recycling_1d_one_step_inspect"),
    "recycling_dthe_one_step": Path("/private/tmp/jax_drb_recycling_dthe_one_step_inspect"),
}

TRACK_FIELDS = ("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe")
TARGET_FIELDS = ("Nd+", "NVd+", "Pe", "Nd")


def _load_reference_state(workdir: Path, field_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with Dataset(workdir / "BOUT.dmp.0.nc") as dataset:
        return {
            name: np.asarray(dataset.variables[name][-1], dtype=np.float64)
            for name in field_names
            if name in dataset.variables
        }


def _field_error_contributions(
    first_fields: dict[str, np.ndarray],
    first_integrals: dict[str, float],
    second_fields: dict[str, np.ndarray],
    second_integrals: dict[str, float],
    *,
    field_names: tuple[str, ...],
    feedback_names: tuple[str, ...],
    mesh,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> list[tuple[str, float]]:
    active = (slice(mesh.xstart, mesh.xend + 1), slice(mesh.ystart, mesh.yend + 1), slice(None))
    rows: list[tuple[str, float]] = []
    for name in field_names:
        first = np.asarray(first_fields[name], dtype=np.float64)[active]
        second = np.asarray(second_fields[name], dtype=np.float64)[active]
        scale = float(absolute_tolerance) + float(relative_tolerance) * np.maximum(np.abs(first), np.abs(second))
        contribution = float(np.sqrt(np.mean(np.square((second - first) / scale))))
        rows.append((name, contribution))
    for name in feedback_names:
        first = float(first_integrals.get(name, 0.0))
        second = float(second_integrals.get(name, 0.0))
        scale = float(absolute_tolerance) + float(relative_tolerance) * max(abs(first), abs(second), 1.0)
        contribution = float(abs(second - first) / scale)
        rows.append((f"integral:{name}", contribution))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def _target_values(fields: dict[str, np.ndarray], *, mesh) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    x = mesh.xstart
    for field in TARGET_FIELDS:
        if field not in fields:
            continue
        array = np.asarray(fields[field], dtype=np.float64)
        rows: dict[str, float] = {}
        for y in (mesh.yend - 1, mesh.yend):
            if y < mesh.ystart or y > mesh.yend:
                continue
            rows[f"y={y}"] = float(array[x, y, 0])
        values[field] = rows
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose adaptive recycling BDF substeps against a staged reference workdir."
    )
    parser.add_argument("--case", required=True, choices=tuple(DEFAULT_WORKDIRS))
    parser.add_argument("--reference-workdir", type=Path, help="Existing reference workdir with BOUT.inp and BOUT.dmp.0.nc.")
    parser.add_argument("--timestep", type=float, default=25.0)
    parser.add_argument("--initial-dt", type=float, default=5.0)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--residual-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--max-nonlinear-iterations", type=int, default=10)
    parser.add_argument("--top-fields", type=int, default=8)
    args = parser.parse_args()

    workdir = args.reference_workdir or DEFAULT_WORKDIRS[args.case]
    config = load_bout_input(workdir / "BOUT.inp")
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    dataset_scalars = resolved_dataset_scalars(run_config)
    runtime_model = _build_recycling_runtime_model(config, mesh=mesh, dataset_scalars=dataset_scalars)
    field_names = tuple(name for name in runtime_model.field_names if name in TRACK_FIELDS or name not in TRACK_FIELDS)
    feedback_names = runtime_model.feedback_names
    reference_fields = _load_reference_state(workdir, TRACK_FIELDS)
    relative_tolerance = float(config.parsed("solver", "rtol")) if config.has_option("solver", "rtol") else 1.0e-6
    absolute_tolerance = float(config.parsed("solver", "atol")) if config.has_option("solver", "atol") else 1.0e-9

    current_fields = _recycling_field_templates(runtime_model.species_templates, field_names=runtime_model.field_names)
    current_integrals = {name: 0.0 for name in feedback_names}
    previous_fields = None
    previous_integrals = None
    previous_dt = None
    remaining = float(args.timestep)
    minimum_dt = max(float(args.timestep) / 8192.0, 0.25)
    dt = min(float(args.initial_dt), remaining)

    print(f"case={args.case}")
    print(f"reference_workdir={workdir}")
    print(f"timestep={args.timestep} initial_dt={dt} minimum_dt={minimum_dt}")

    for attempt in range(1, args.attempts + 1):
        if remaining <= 1.0e-12:
            break
        dt = min(dt, remaining)
        reset_history = (
            previous_fields is None
            or previous_integrals is None
            or previous_dt is None
            or abs(float(previous_dt) - float(dt)) > 1.0e-12
        )
        branch = "startup_be" if reset_history else "bdf2"

        if reset_history:
            full_fields, full_integrals, _ = advance_recycling_1d_backward_euler_step(
                config,
                current_fields,
                runtime_model=runtime_model,
                feedback_integrals=current_integrals,
                mesh=mesh,
                metrics=metrics,
                dataset_scalars=dataset_scalars,
                timestep=dt,
                solver_mode="sparse",
                residual_tolerance=args.residual_tolerance,
                max_nonlinear_iterations=args.max_nonlinear_iterations,
            )
            half_fields, half_integrals, _ = advance_recycling_1d_backward_euler_step(
                config,
                current_fields,
                runtime_model=runtime_model,
                feedback_integrals=current_integrals,
                mesh=mesh,
                metrics=metrics,
                dataset_scalars=dataset_scalars,
                timestep=0.5 * dt,
                solver_mode="sparse",
                residual_tolerance=args.residual_tolerance,
                max_nonlinear_iterations=args.max_nonlinear_iterations,
            )
            candidate_fields, candidate_integrals, _ = advance_recycling_1d_backward_euler_step(
                config,
                half_fields,
                runtime_model=runtime_model,
                feedback_integrals=half_integrals,
                mesh=mesh,
                metrics=metrics,
                dataset_scalars=dataset_scalars,
                timestep=0.5 * dt,
                solver_mode="sparse",
                residual_tolerance=args.residual_tolerance,
                max_nonlinear_iterations=args.max_nonlinear_iterations,
            )
            error_ratio = _recycling_state_error_ratio(
                full_fields,
                full_integrals,
                candidate_fields,
                candidate_integrals,
                field_names=runtime_model.field_names,
                feedback_names=feedback_names,
                mesh=mesh,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
            comparison_rows = _field_error_contributions(
                full_fields,
                full_integrals,
                candidate_fields,
                candidate_integrals,
                field_names=runtime_model.field_names,
                feedback_names=feedback_names,
                mesh=mesh,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
            first_label = "be(dt)"
            second_label = "be(dt/2)+be(dt/2)"
        else:
            full_fields, full_integrals, _ = advance_recycling_1d_backward_euler_step(
                config,
                current_fields,
                runtime_model=runtime_model,
                feedback_integrals=current_integrals,
                mesh=mesh,
                metrics=metrics,
                dataset_scalars=dataset_scalars,
                timestep=dt,
                solver_mode="sparse",
                residual_tolerance=args.residual_tolerance,
                max_nonlinear_iterations=args.max_nonlinear_iterations,
            )
            candidate_fields, candidate_integrals, _ = advance_recycling_1d_bdf2_step(
                config,
                current_fields,
                previous_fields,
                runtime_model=runtime_model,
                feedback_integrals=current_integrals,
                previous_feedback_integrals=previous_integrals,
                mesh=mesh,
                metrics=metrics,
                dataset_scalars=dataset_scalars,
                timestep=dt,
                solver_mode="sparse",
                residual_tolerance=args.residual_tolerance,
                max_nonlinear_iterations=args.max_nonlinear_iterations,
            )
            error_ratio = _recycling_state_error_ratio(
                full_fields,
                full_integrals,
                candidate_fields,
                candidate_integrals,
                field_names=runtime_model.field_names,
                feedback_names=feedback_names,
                mesh=mesh,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
            comparison_rows = _field_error_contributions(
                full_fields,
                full_integrals,
                candidate_fields,
                candidate_integrals,
                field_names=runtime_model.field_names,
                feedback_names=feedback_names,
                mesh=mesh,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
            first_label = "be(dt)"
            second_label = "bdf2(dt)"

        candidate_feedback_errors = _current_feedback_errors(candidate_fields, controllers=runtime_model.controllers, mesh=mesh)
        current_feedback_errors = _current_feedback_errors(current_fields, controllers=runtime_model.controllers, mesh=mesh)
        accept = bool(np.isfinite(error_ratio) and error_ratio <= 1.0)
        force_accept = bool((not accept) and dt <= minimum_dt)
        print(
            f"attempt={attempt} branch={branch} remaining={remaining:.12e} dt={dt:.12e} previous_dt={previous_dt} "
            f"error_ratio={error_ratio:.12e} accept={accept} force_accept={force_accept} reset_history={reset_history}"
        )
        print(f"  compare={first_label} vs {second_label}")
        print(f"  current_feedback_errors={current_feedback_errors}")
        print(f"  candidate_feedback_errors={candidate_feedback_errors}")
        print(f"  current_integrals={current_integrals}")
        print(f"  candidate_integrals={candidate_integrals}")
        print("  top_error_contributions=")
        for name, value in comparison_rows[: max(int(args.top_fields), 1)]:
            print(f"    {name}: {value:.12e}")
        print("  target_values_current=", _target_values(current_fields, mesh=mesh))
        print("  target_values_candidate=", _target_values(candidate_fields, mesh=mesh))
        print("  target_values_reference=", _target_values(reference_fields, mesh=mesh))

        if accept or force_accept:
            previous_fields = current_fields
            previous_integrals = current_integrals
            previous_dt = dt
            current_fields = candidate_fields
            current_integrals = candidate_integrals
            remaining -= dt
            if accept and error_ratio < 0.1:
                next_dt = min(2.0 * dt, remaining if remaining > 1.0e-12 else dt)
                if abs(next_dt - dt) > 1.0e-12:
                    previous_fields = None
                    previous_integrals = None
                    previous_dt = None
                dt = next_dt
            continue

        dt = max(0.5 * dt, minimum_dt)
        previous_fields = None
        previous_integrals = None
        previous_dt = None


if __name__ == "__main__":
    main()
