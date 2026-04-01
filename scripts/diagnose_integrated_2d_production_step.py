#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native.recycling_1d import advance_recycling_1d_implicit_history, compute_recycling_1d_rhs
from jax_drb.native.reference_dump import (
    load_local_reference_snapshot,
    load_local_reference_snapshot_cache,
)
from jax_drb.native.runner import (
    _apply_species_velocity_overrides,
    _integrated_2d_initial_rhs_case_name,
    _integrated_2d_snapshot_cache_path,
    _select_integrated_2d_transient_solver_mode,
)
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.parity.reference import run_reference_case
from jax_drb.runtime.run_config import RunConfiguration


STATE_FIELDS = ("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe")
RHS_FIELDS = ("ddt(Nd+)", "ddt(Pd+)", "ddt(NVd+)", "ddt(Nd)", "ddt(Pd)", "ddt(NVd)", "ddt(Pe)")
SOURCE_FIELDS = ("SNd+", "SNVd+", "SPd+", "SNd", "SNVd", "SPd")
DIAGNOSTIC_FIELDS = ("Sd_target_recycle", "Ed_target_recycle", "Vd+", "Vd")
SCALAR_FIELDS = ("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0")
BOUNDARY_MODES = {
    "prod_mixed": {
        "preserve_dump_target_state": True,
        "preserve_dump_ion_target_state_only": True,
    },
    "all_sheath": {
        "preserve_dump_target_state": False,
        "preserve_dump_ion_target_state_only": False,
    },
}


def _max_abs_diff(a: np.ndarray, b: np.ndarray, active: tuple[slice, slice, slice]) -> tuple[float, tuple[int, ...]]:
    delta = np.abs(np.asarray(a[active], dtype=np.float64) - np.asarray(b[active], dtype=np.float64))
    flat_index = int(np.argmax(delta))
    location = np.unravel_index(flat_index, delta.shape)
    return float(delta.reshape(-1)[flat_index]), tuple(int(value) for value in location)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose integrated 2D production one-step parity by comparing the native one-step state "
            "and native RHS-on-reference-state against a fresh reference run."
        )
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--case", default="integrated_2d_production_one_step", choices=("integrated_2d_production_one_step",))
    args = parser.parse_args()

    initial_case_name = _integrated_2d_initial_rhs_case_name(args.case)
    snapshot_cache_path = _integrated_2d_snapshot_cache_path(initial_case_name)
    if not snapshot_cache_path.exists():
        raise FileNotFoundError(f"Missing committed snapshot cache: {snapshot_cache_path}")

    initial_snapshot = load_local_reference_snapshot_cache(
        snapshot_cache_path,
        field_names=STATE_FIELDS,
        optional_field_names=SOURCE_FIELDS + DIAGNOSTIC_FIELDS,
        scalar_names=SCALAR_FIELDS,
    )

    with tempfile.TemporaryDirectory(prefix="jaxdrb-integrated-2d-production-step-") as workdir:
        execution = run_reference_case(
            args.case,
            reference_root=args.reference_root,
            workdir=workdir,
            keep_workdir=True,
        )
        dump_path = Path(execution.summary.artifacts["BOUT.dmp.0.nc"])
        reference_initial = load_local_reference_snapshot(
            dump_path,
            field_names=STATE_FIELDS,
            optional_field_names=DIAGNOSTIC_FIELDS,
            scalar_names=(),
            time_index=0,
        )
        reference_final = load_local_reference_snapshot(
            dump_path,
            field_names=STATE_FIELDS + RHS_FIELDS,
            optional_field_names=SOURCE_FIELDS + DIAGNOSTIC_FIELDS,
            scalar_names=(),
            time_index=1,
        )

    config = load_bout_input(args.reference_root / "tests" / "integrated" / "2D-production" / "data" / "BOUT.inp")
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)

    density_source_overrides = {
        name: np.asarray(initial_snapshot.optional_fields[field_name], dtype=np.float64)
        for name, field_name in (("d+", "SNd+"), ("d", "SNd"))
        if field_name in initial_snapshot.optional_fields
    } or None
    pressure_source_overrides = {
        name: np.asarray(initial_snapshot.optional_fields[field_name], dtype=np.float64)
        for name, field_name in (("d+", "SPd+"), ("d", "SPd"))
        if field_name in initial_snapshot.optional_fields
    } or None
    momentum_source_overrides = {
        name: np.asarray(initial_snapshot.optional_fields[field_name], dtype=np.float64)
        for name, field_name in (("d+", "SNVd+"), ("d", "SNVd"))
        if field_name in initial_snapshot.optional_fields
    } or None
    initial_velocity_overrides = {
        name: np.asarray(initial_snapshot.optional_fields[field_name], dtype=np.float64)
        for name, field_name in (("d+", "Vd+"), ("d", "Vd"))
        if field_name in initial_snapshot.optional_fields
    } or None
    final_velocity_overrides = {
        name: np.asarray(reference_final.optional_fields[field_name], dtype=np.float64)
        for name, field_name in (("d+", "Vd+"), ("d", "Vd"))
        if field_name in reference_final.optional_fields
    } or None

    initial_fields = {name: np.asarray(value, dtype=np.float64) for name, value in initial_snapshot.fields.items()}
    if initial_velocity_overrides:
        initial_fields = _apply_species_velocity_overrides(
            config,
            field_overrides=initial_fields,
            velocity_field_overrides=initial_velocity_overrides,
        )

    solver_mode = _select_integrated_2d_transient_solver_mode(args.case, config=config, parity_mode="one_step")
    rhs_field_overrides = {name: np.asarray(value, dtype=np.float64) for name, value in reference_final.fields.items() if name in STATE_FIELDS}
    if final_velocity_overrides:
        rhs_field_overrides = _apply_species_velocity_overrides(
            config,
            field_overrides=rhs_field_overrides,
            velocity_field_overrides=final_velocity_overrides,
        )
    rhs_density_source_overrides = {
        name: np.asarray(reference_final.optional_fields[field_name], dtype=np.float64)
        for name, field_name in (("d+", "SNd+"), ("d", "SNd"))
        if field_name in reference_final.optional_fields
    } or density_source_overrides
    rhs_pressure_source_overrides = {
        name: np.asarray(reference_final.optional_fields[field_name], dtype=np.float64)
        for name, field_name in (("d+", "SPd+"), ("d", "SPd"))
        if field_name in reference_final.optional_fields
    } or pressure_source_overrides
    rhs_momentum_source_overrides = {
        name: np.asarray(reference_final.optional_fields[field_name], dtype=np.float64)
        for name, field_name in (("d+", "SNVd+"), ("d", "SNVd"))
        if field_name in reference_final.optional_fields
    } or momentum_source_overrides

    mode_histories: dict[str, dict[str, np.ndarray]] = {}
    mode_rhs: dict[str, dict[str, np.ndarray]] = {}
    for mode_name, boundary_kwargs in BOUNDARY_MODES.items():
        history = advance_recycling_1d_implicit_history(
            config,
            mesh=initial_snapshot.mesh,
            metrics=initial_snapshot.metrics,
            dataset_scalars=dataset_scalars,
            timestep=run_config.time.timestep,
            steps=1,
            initial_fields=initial_fields,
            density_source_overrides=density_source_overrides,
            pressure_source_overrides=pressure_source_overrides,
            momentum_source_overrides=momentum_source_overrides,
            solver_mode=solver_mode,
            residual_tolerance=float(config.parsed("solver", "rtol")) if config.has_option("solver", "rtol") else 1.0e-8,
            max_nonlinear_iterations=30,
            **boundary_kwargs,
        )
        mode_histories[mode_name] = {
            name: np.asarray(value[1], dtype=np.float64) for name, value in history.variable_history.items()
        }
        mode_rhs[mode_name] = compute_recycling_1d_rhs(
            config,
            mesh=reference_final.mesh,
            metrics=reference_final.metrics,
            dataset_scalars=dataset_scalars,
            field_overrides=rhs_field_overrides,
            apply_sheath_boundaries=True,
            density_source_overrides=rhs_density_source_overrides,
            pressure_source_overrides=rhs_pressure_source_overrides,
            momentum_source_overrides=rhs_momentum_source_overrides,
            **boundary_kwargs,
        ).variables

    active = (
        slice(reference_final.mesh.xstart, reference_final.mesh.xend + 1),
        slice(reference_final.mesh.ystart, reference_final.mesh.yend + 1),
        slice(None),
    )
    print(f"case={args.case}")
    print(f"solver_mode={solver_mode}")
    for mode_name in BOUNDARY_MODES:
        print(f"state max_abs_diff on active domain [{mode_name}]")
        for name in STATE_FIELDS:
            diff, location = _max_abs_diff(mode_histories[mode_name][name], reference_final.fields[name], active)
            print(f"{name} diff={diff:.12e} location={location}")
        print(f"rhs_on_reference_state max_abs_diff on active domain [{mode_name}]")
        for name in RHS_FIELDS:
            if name not in reference_final.fields:
                continue
            diff, location = _max_abs_diff(mode_rhs[mode_name][name][0], reference_final.fields[name], active)
            print(f"{name} diff={diff:.12e} location={location}")
        print(f"diagnostics_on_reference_state max_abs_diff on active domain [{mode_name}]")
        for name in ("Sd_target_recycle", "Ed_target_recycle"):
            if name not in reference_final.optional_fields or name not in mode_rhs[mode_name]:
                continue
            diff, location = _max_abs_diff(mode_rhs[mode_name][name][0], reference_final.optional_fields[name], active)
            print(f"{name} diff={diff:.12e} location={location}")
    print("target_band rhs error sweep")
    for cell in ((14, reference_final.mesh.ystart, 0), (15, reference_final.mesh.ystart, 0)):
        i, j, k = cell
        print(f"cell={cell}")
        for field in ("ddt(Pe)", "ddt(Pd+)", "ddt(NVd+)", "ddt(Nd+)"):
            if field not in reference_final.fields:
                continue
            prod_value = mode_rhs["prod_mixed"][field][0][i, j, k]
            sheath_value = mode_rhs["all_sheath"][field][0][i, j, k]
            ref_value = reference_final.fields[field][i, j, k]
            print(
                f"  {field}: "
                f"prod_err={prod_value - ref_value:.12e} "
                f"sheath_err={sheath_value - ref_value:.12e} "
                f"bc_delta={prod_value - sheath_value:.12e}"
            )
    print("reference_initial_vs_cache max_abs_diff on active domain")
    for name in STATE_FIELDS:
        diff, location = _max_abs_diff(initial_snapshot.fields[name], reference_initial.fields[name], active)
        print(f"{name} diff={diff:.12e} location={location}")


if __name__ == "__main__":
    main()
