#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jax_drb.config.boutinp import parse_bout_input
from jax_drb.native.fluid_1d import (
    advance_mms_history,
    evaluate_field_option,
)
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.runtime.run_config import RunConfiguration


_MMS_TEMPLATE = """
nout = 50
timestep = {timestep}

MXG = 0

[mesh]
nx = 1
ny = {ny}
nz = 1
Ly = 10
dy = Ly / ny
J = 1

[solver]
mxstep = 10000
rtol = 1e-7
mms = true

[model]
components = i
normalise_metric = false
Nnorm = 1e18
Bnorm = 1
Tnorm = 5

[i]
type = evolve_density, evolve_pressure, evolve_momentum
charge = 1.0
AA = 2.0
thermal_conduction = false

[Ni]
solution = 1 - 0.1*sin(t - 2.0*y)
source = -0.1*cos(t - 2.0*y) + 0.0628318530717959*cos(2*t + y)

[Pi]
solution = 0.1*cos(t + 3.0*y) + 1
source = (0.0628318530717959*cos(2*t + y)/(1 - 0.1*sin(t - 2.0*y)) - 0.0125663706143592*sin(2*t + y)*cos(t - 2.0*y)/(1 - 0.1*sin(t - 2.0*y))^2)*(0.0666666666666667*cos(t + 3.0*y) + 0.666666666666667) - 0.1*sin(t + 3.0*y) + 0.0628318530717959*(0.1*cos(t + 3.0*y) + 1)*cos(2*t + y)/(1 - 0.1*sin(t - 2.0*y)) - 0.0188495559215388*sin(t + 3.0*y)*sin(2*t + y)/(1 - 0.1*sin(t - 2.0*y)) - 0.0125663706143592*(0.1*cos(t + 3.0*y) + 1)*sin(2*t + y)*cos(t - 2.0*y)/(1 - 0.1*sin(t - 2.0*y))^2

[NVi]
solution = 0.2*sin(2*t + y)
source = -0.188495559215388*sin(t + 3.0*y) + 0.4*cos(2*t + y) + 0.0251327412287183*sin(2*t + y)*cos(2*t + y)/(1 - 0.1*sin(t - 2.0*y)) - 0.00251327412287184*sin(2*t + y)^2*cos(t - 2.0*y)/(1 - 0.1*sin(t - 2.0*y))^2
"""


def build_mms_config(*, ny: int, timestep: float) -> str:
    return _MMS_TEMPLATE.format(ny=int(ny), timestep=float(timestep))


def l2_error(numerical: np.ndarray, exact: np.ndarray, *, start: int, end: int) -> float:
    interior_numerical = np.asarray(numerical[:, start : end + 1, :], dtype=np.float64)
    interior_exact = np.asarray(exact[:, start : end + 1, :], dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(interior_numerical - interior_exact))))


def observed_order(coarse_error: float, fine_error: float) -> float:
    if coarse_error <= 0.0 or fine_error <= 0.0:
        return float("inf")
    return float(np.log(coarse_error / fine_error) / np.log(2.0))


def run_resolution(*, ny: int, timestep: float, steps: int, substeps: int) -> dict[str, object]:
    config = parse_bout_input(build_mms_config(ny=ny, timestep=timestep))
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    history = advance_mms_history(
        config,
        section="i",
        mesh=mesh,
        metrics=metrics,
        atomic_mass=2.0,
        timestep=timestep,
        steps=steps,
        substeps=substeps,
    )
    final_time = float(steps * timestep)
    exact_density = evaluate_field_option(config, "Ni", "solution", mesh=mesh, time=final_time)
    exact_pressure = evaluate_field_option(config, "Pi", "solution", mesh=mesh, time=final_time)
    exact_momentum = evaluate_field_option(config, "NVi", "solution", mesh=mesh, time=final_time)

    density_error = l2_error(np.asarray(history.density_history[-1]), np.asarray(exact_density), start=mesh.ystart, end=mesh.yend)
    pressure_error = l2_error(np.asarray(history.pressure_history[-1]), np.asarray(exact_pressure), start=mesh.ystart, end=mesh.yend)
    momentum_error = l2_error(np.asarray(history.momentum_history[-1]), np.asarray(exact_momentum), start=mesh.ystart, end=mesh.yend)

    return {
        "ny": int(ny),
        "timestep": float(timestep),
        "steps": int(steps),
        "substeps": int(substeps),
        "final_time": final_time,
        "errors": {
            "density_l2": density_error,
            "pressure_l2": pressure_error,
            "momentum_l2": momentum_error,
        },
    }


def build_convergence_report(*, resolutions: tuple[int, ...], timestep: float, steps: int, substeps: int) -> dict[str, object]:
    runs = [run_resolution(ny=ny, timestep=timestep, steps=steps, substeps=substeps) for ny in resolutions]
    orders: list[dict[str, float | int]] = []
    for coarse, fine in zip(runs[:-1], runs[1:]):
        coarse_errors = coarse["errors"]
        fine_errors = fine["errors"]
        orders.append(
            {
                "from_ny": int(coarse["ny"]),
                "to_ny": int(fine["ny"]),
                "density_order": observed_order(float(coarse_errors["density_l2"]), float(fine_errors["density_l2"])),
                "pressure_order": observed_order(float(coarse_errors["pressure_l2"]), float(fine_errors["pressure_l2"])),
                "momentum_order": observed_order(float(coarse_errors["momentum_l2"]), float(fine_errors["momentum_l2"])),
            }
        )
    return {
        "case": "fluid_1d_mms_convergence",
        "resolutions": [int(ny) for ny in resolutions],
        "timestep": float(timestep),
        "steps": int(steps),
        "substeps": int(substeps),
        "runs": runs,
        "observed_orders": orders,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a fast fluid-1D manufactured-solution convergence campaign and emit a JSON report."
    )
    parser.add_argument("--resolution", dest="resolutions", action="append", type=int, default=[], help="Interior ny resolution to include. Repeat for multiple resolutions.")
    parser.add_argument("--timestep", type=float, default=0.05, help="Timestep for each stored step.")
    parser.add_argument("--steps", type=int, default=2, help="Number of stored steps to advance before comparing to the exact solution.")
    parser.add_argument("--substeps", type=int, default=20, help="RK4 substeps per stored step.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    resolutions = tuple(args.resolutions) if args.resolutions else (32, 64, 128)
    report = build_convergence_report(
        resolutions=resolutions,
        timestep=float(args.timestep),
        steps=int(args.steps),
        substeps=int(args.substeps),
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
