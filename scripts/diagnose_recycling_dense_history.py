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
from jax_drb.native.recycling_1d import advance_recycling_1d_implicit_history
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.parity.reference import run_reference_case
from jax_drb.runtime.run_config import RunConfiguration


DEFAULT_FIELDS = ("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe")


def _parse_steps(raw: str) -> tuple[int, ...]:
    return tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a dense-output external-reference recycling trajectory against the native transient history."
    )
    parser.add_argument("--case", default="recycling_1d_one_step", choices=("recycling_1d_one_step", "recycling_dthe_one_step"))
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--reference-binary", type=Path)
    parser.add_argument("--output-step", type=float, default=125.0, help="Reference/native output step to compare.")
    parser.add_argument("--steps", type=int, default=40, help="Number of output steps to compare.")
    parser.add_argument("--field", action="append", dest="fields", help="Restrict comparison to these fields.")
    parser.add_argument(
        "--report-steps",
        default="1,5,10,20,40",
        help="Comma-separated list of output indices to report.",
    )
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix=f"jaxdrb-{args.case}-dense-"))
    reference_result = run_reference_case(
        args.case,
        reference_root=args.reference_root,
        reference_binary=args.reference_binary,
        workdir=workdir,
        extra_overrides=(f"nout={args.steps}", f"timestep={args.output_step:g}"),
    )
    reference_dump = Path(reference_result.summary.workdir) / "BOUT.dmp.0.nc"

    input_path = args.reference_root / (
        "tests/integrated/1D-recycling/data/BOUT.inp"
        if args.case == "recycling_1d_one_step"
        else "tests/integrated/1D-recycling-dthe/data/BOUT.inp"
    )
    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    history = advance_recycling_1d_implicit_history(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
        timestep=float(args.output_step),
        steps=int(args.steps),
        solver_mode="continuation" if args.case == "recycling_1d_one_step" else "bdf",
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=30,
    )

    fields = tuple(args.fields) if args.fields else DEFAULT_FIELDS
    report_steps = _parse_steps(args.report_steps)

    print(f"case={args.case}")
    print(f"reference_workdir={reference_result.summary.workdir}")
    print(f"output_step={args.output_step:g}")
    print(f"steps={args.steps}")

    with Dataset(reference_dump) as dataset:
        for step in report_steps:
            if step < 0 or step > args.steps:
                continue
            print(f"step={step}")
            for field in fields:
                if field not in dataset.variables or field not in history.variable_history:
                    continue
                reference = np.asarray(dataset.variables[field][step], dtype=np.float64)
                native = np.asarray(history.variable_history[field][step], dtype=np.float64)
                reference = reference[mesh.xstart : mesh.xend + 1, mesh.ystart : mesh.yend + 1, :]
                native = native[mesh.xstart : mesh.xend + 1, mesh.ystart : mesh.yend + 1, :]
                delta = np.abs(native - reference)
                location = np.unravel_index(int(np.nanargmax(delta)), delta.shape)
                print(
                    f"  {field}: max_abs={float(delta[location]):.8e} "
                    f"loc={location} native={float(native[location]):.8e} reference={float(reference[location]):.8e}"
                )


if __name__ == "__main__":
    main()
