#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.recycling_1d import (
    _build_recycling_runtime_model,
    _recycling_field_templates,
    advance_recycling_1d_backward_euler_step,
    compute_recycling_1d_rhs,
)
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.parity.recycling import extract_recycling_controller_snapshot
from jax_drb.parity.reference import run_reference_case
from jax_drb.runtime.run_config import RunConfiguration


STATE_FIELDS = ("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Nt+", "Pt+", "NVt+", "Nt", "Pt", "NVt", "Nhe+", "Phe+", "NVhe+", "Nhe", "Phe", "NVhe", "Pe")
RHS_FIELDS = ("ddt(Nd+)", "ddt(Pd+)", "ddt(NVd+)", "ddt(Nd)", "ddt(Pd)", "ddt(NVd)", "ddt(Nt+)", "ddt(Pt+)", "ddt(NVt+)", "ddt(Nt)", "ddt(Pt)", "ddt(NVt)", "ddt(Nhe+)", "ddt(Phe+)", "ddt(NVhe+)", "ddt(Nhe)", "ddt(Phe)", "ddt(NVhe)", "ddt(Pe)")


def _load_dataset_fields(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with Dataset(path) as dataset:
        return {
            name: np.asarray(dataset.variables[name][-1], dtype=np.float64)
            for name in names
            if name in dataset.variables
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a short reference recycling run against a single native backward-Euler step and against the native RHS evaluated on the reference-evolved state."
    )
    parser.add_argument("--case", required=True, choices=("recycling_1d_one_step", "recycling_dthe_one_step"))
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--timestep", type=float, default=25.0)
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix=f"jaxdrb-{args.case}-short-"))
    execution = run_reference_case(
        args.case,
        reference_root=args.reference_root,
        workdir=workdir,
        extra_overrides=(f"timestep={args.timestep}", "solver:mxstep=50000"),
    )
    dump_path = workdir / "BOUT.dmp.0.nc"
    restart_path = workdir / "BOUT.restart.0.nc"
    reference_state = _load_dataset_fields(dump_path, STATE_FIELDS)
    reference_rhs = _load_dataset_fields(dump_path, RHS_FIELDS)

    config = load_bout_input(workdir / "BOUT.inp")
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    dataset_scalars = resolved_dataset_scalars(run_config)
    runtime_model = _build_recycling_runtime_model(config, mesh=mesh, dataset_scalars=dataset_scalars)
    initial_fields = _recycling_field_templates(runtime_model.species_templates, field_names=runtime_model.field_names)

    controller_species = tuple(runtime_model.feedback_names)
    snapshot = extract_recycling_controller_snapshot(dump_path, restart_path, controller_species=controller_species)

    native_fields = {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in initial_fields.items()}
    native_fields, native_integrals, step_info = advance_recycling_1d_backward_euler_step(
        config,
        native_fields,
        runtime_model=runtime_model,
        feedback_integrals={name: 0.0 for name in runtime_model.feedback_names},
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        timestep=args.timestep,
        solver_mode="sparse",
        residual_tolerance=float(config.parsed("solver", "rtol")) if config.has_option("solver", "rtol") else 1.0e-7,
        max_nonlinear_iterations=30,
    )

    rhs_on_reference_state = compute_recycling_1d_rhs(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        field_overrides=reference_state,
        feedback_integrals=snapshot.restart_integrals,
    ).variables

    active = (slice(mesh.xstart, mesh.xend + 1), slice(mesh.ystart, mesh.yend + 1), slice(None))
    print(f"case={args.case}")
    print(f"workdir={workdir}")
    print(f"overrides={execution.summary.overrides}")
    print(f"native_step_info={step_info}")
    print("state max_abs_diff")
    for name in tuple(runtime_model.field_names):
        if name not in reference_state:
            continue
        diff = float(np.nanmax(np.abs(native_fields[name][active] - reference_state[name][active])))
        print(f"{name} {diff:.12e}")
    print("reference_state_rhs max_abs_diff")
    for name in RHS_FIELDS:
        if name not in reference_rhs or name not in rhs_on_reference_state:
            continue
        diff = float(np.nanmax(np.abs(rhs_on_reference_state[name][0][active] - reference_rhs[name][active])))
        print(f"{name} {diff:.12e}")
    print("controller_integrals")
    for name in controller_species:
        native_value = float(native_integrals.get(name, 0.0))
        reference_value = float(snapshot.restart_integrals.get(name, 0.0))
        print(f"{name} native={native_value:.12e} reference={reference_value:.12e} diff={native_value-reference_value:.12e}")


if __name__ == "__main__":
    main()
