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
from jax_drb.native.recycling_1d import advance_recycling_1d_implicit_history, compute_recycling_1d_rhs
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.parity.recycling import extract_recycling_controller_snapshot
from jax_drb.parity.reference import run_reference_case
from jax_drb.runtime.run_config import RunConfiguration


DEFAULT_STATE_FIELDS = ("Nd", "Pd", "NVd", "NVd+", "Nd+", "Pd+", "Pe")
DEFAULT_RHS_FIELDS = (
    "ddt(Nd)",
    "ddt(Pd)",
    "ddt(NVd)",
    "ddt(NVd+)",
    "Sd_Dpar",
    "Ed_Dpar",
    "Fd_Dpar",
    "SNVd",
    "SNVd+",
    "DivPiPar_d+",
    "Fd+d_coll",
    "Fd+e_coll",
    "Fd+d_cx",
    "Fdd+_cx",
    "Fd+_iz",
    "Fd+_rec",
)


def relative_error_metrics(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    magnitude_floor_ratio: float = 1.0e-3,
    absolute_floor: float = 1.0e-8,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    delta = np.abs(actual - reference)
    denom = np.maximum(np.abs(reference), float(absolute_floor))
    rel = delta / denom
    reference_scale = float(np.nanmax(np.abs(reference))) if reference.size else 0.0
    significant_floor = max(float(absolute_floor), float(magnitude_floor_ratio) * reference_scale)
    significant_mask = np.abs(reference) >= significant_floor
    if np.any(significant_mask):
        significant_rel = float(np.nanmax(rel[significant_mask]))
        significant_abs = float(np.nanmax(delta[significant_mask]))
        significant_count = int(np.count_nonzero(significant_mask))
    else:
        significant_rel = 0.0
        significant_abs = 0.0
        significant_count = 0
    return {
        "max_abs": float(np.nanmax(delta)) if delta.size else 0.0,
        "max_rel": float(np.nanmax(rel)) if rel.size else 0.0,
        "max_rel_significant": significant_rel,
        "max_abs_significant": significant_abs,
        "reference_scale": reference_scale,
        "significant_floor": significant_floor,
        "significant_count": significant_count,
    }


def _load_last_fields(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with Dataset(path) as dataset:
        return {
            name: np.asarray(dataset.variables[name][-1], dtype=np.float64)
            for name in names
            if name in dataset.variables
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose neutral-side recycling transient errors with magnitude-aware relative metrics."
    )
    parser.add_argument("--case", default="recycling_1d_one_step", choices=("recycling_1d_one_step", "recycling_dthe_one_step"))
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--timestep", type=float, default=25.0)
    parser.add_argument("--solver-mode", default="bdf")
    parser.add_argument("--magnitude-floor-ratio", type=float, default=1.0e-3)
    parser.add_argument("--absolute-floor", type=float, default=1.0e-8)
    parser.add_argument("--state-field", action="append", dest="state_fields")
    parser.add_argument("--rhs-field", action="append", dest="rhs_fields")
    args = parser.parse_args()

    state_fields = tuple(args.state_fields) if args.state_fields else DEFAULT_STATE_FIELDS
    rhs_fields = tuple(args.rhs_fields) if args.rhs_fields else DEFAULT_RHS_FIELDS

    workdir = Path(tempfile.mkdtemp(prefix=f"jaxdrb-{args.case}-neutral-transient-"))
    execution = run_reference_case(
        args.case,
        reference_root=args.reference_root,
        workdir=workdir,
        extra_overrides=(f"timestep={args.timestep}", "solver:mxstep=50000"),
    )
    dump_path = workdir / "BOUT.dmp.0.nc"
    restart_path = workdir / "BOUT.restart.0.nc"
    reference_state = _load_last_fields(dump_path, state_fields)
    reference_rhs = _load_last_fields(dump_path, rhs_fields)

    config = load_bout_input(workdir / "BOUT.inp")
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    dataset_scalars = resolved_dataset_scalars(run_config)
    controller_species = tuple(
        section
        for section in config.sections
        if config.has_section(section) and config.has_option(section, "type") and "upstream_density_feedback" in str(config.parsed(section, "type"))
    )
    snapshot = extract_recycling_controller_snapshot(
        dump_path,
        restart_path,
        controller_species=controller_species,
    )

    native_history = advance_recycling_1d_implicit_history(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        timestep=args.timestep,
        steps=1,
        solver_mode=args.solver_mode,
        residual_tolerance=float(config.parsed("solver", "rtol")) if config.has_option("solver", "rtol") else 1.0e-8,
        max_nonlinear_iterations=30,
    )
    native_rhs = compute_recycling_1d_rhs(
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
    print(f"solver_mode={args.solver_mode}")
    print(f"magnitude_floor_ratio={args.magnitude_floor_ratio:g}")
    print(f"absolute_floor={args.absolute_floor:g}")

    print("state field max_abs max_rel max_rel_significant significant_floor significant_count")
    for name in state_fields:
        if name not in reference_state or name not in native_history.variable_history:
            continue
        metrics_row = relative_error_metrics(
            np.asarray(native_history.variable_history[name][1], dtype=np.float64)[active],
            np.asarray(reference_state[name], dtype=np.float64)[active],
            magnitude_floor_ratio=args.magnitude_floor_ratio,
            absolute_floor=args.absolute_floor,
        )
        print(
            f"{name} {metrics_row['max_abs']:.12e} {metrics_row['max_rel']:.12e} "
            f"{metrics_row['max_rel_significant']:.12e} {metrics_row['significant_floor']:.12e} "
            f"{metrics_row['significant_count']}"
        )

    print("rhs field max_abs max_rel max_rel_significant significant_floor significant_count")
    for name in rhs_fields:
        if name not in reference_rhs or name not in native_rhs:
            continue
        actual = np.asarray(native_rhs[name], dtype=np.float64)
        if actual.ndim == 4:
            actual = actual[0]
        metrics_row = relative_error_metrics(
            actual[active],
            np.asarray(reference_rhs[name], dtype=np.float64)[active],
            magnitude_floor_ratio=args.magnitude_floor_ratio,
            absolute_floor=args.absolute_floor,
        )
        print(
            f"{name} {metrics_row['max_abs']:.12e} {metrics_row['max_rel']:.12e} "
            f"{metrics_row['max_rel_significant']:.12e} {metrics_row['significant_floor']:.12e} "
            f"{metrics_row['significant_count']}"
        )


if __name__ == "__main__":
    main()
