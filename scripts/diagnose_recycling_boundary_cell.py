#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from unittest import mock
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native.mesh import StructuredMesh, build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
import jax_drb.native.recycling_1d as recycling_1d_mod
from jax_drb.native.recycling_1d import (
    advance_recycling_1d_implicit_history,
    compute_recycling_1d_rhs,
)
from jax_drb.native.recycling_rhs_terms import IonRhsTerms
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.parity.recycling import extract_recycling_controller_snapshot
from jax_drb.parity.reference import run_reference_case
from jax_drb.runtime.run_config import RunConfiguration


DEFAULT_STATE_FIELDS = ("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe")
DEFAULT_RHS_FIELDS = (
    "ddt(Nd+)",
    "ddt(Pd+)",
    "ddt(NVd+)",
    "ddt(Nd)",
    "ddt(Pe)",
    "SNVd+",
    "SNVd",
    "Sd_target_recycle",
    "Ed_target_recycle",
    "Ve",
    "Epar",
)
DEFAULT_ION_TERM_NAMES = (
    "density_source",
    "density_transport",
    "density_total",
    "momentum_advection",
    "pressure_gradient",
    "momentum_source",
    "momentum_total",
)


def default_recycling_boundary_cells(mesh: StructuredMesh) -> tuple[tuple[int, int, int], ...]:
    return (
        (mesh.xstart, mesh.yend, 0),
        (mesh.xstart, mesh.yend - 1, 0),
    )


def _parse_cell(raw: str) -> tuple[int, int, int]:
    x_str, y_str, z_str = (piece.strip() for piece in raw.split(",", 2))
    return int(x_str), int(y_str), int(z_str)


def _load_last_fields(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with Dataset(path) as dataset:
        return {
            name: np.asarray(dataset.variables[name][-1], dtype=np.float64)
            for name in names
            if name in dataset.variables
        }


def _cell_value(values: np.ndarray, cell: tuple[int, int, int]) -> float:
    array = np.asarray(values, dtype=np.float64)
    x, y, z = cell
    if array.ndim == 4:
        return float(array[0, x, y, z])
    return float(array[x, y, z])


def _extract_ion_rhs_terms(
    config,
    *,
    mesh: StructuredMesh,
    metrics,
    dataset_scalars: dict[str, float],
    state_fields: dict[str, np.ndarray],
    feedback_integrals: dict[str, float],
    ion_name: str,
) -> IonRhsTerms:
    captured_terms: dict[str, IonRhsTerms] = {}
    original = recycling_1d_mod._assemble_ion_rhs_terms

    runtime_model = recycling_1d_mod._build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=dataset_scalars,
        field_overrides=state_fields,
    )
    species = recycling_1d_mod._override_species_fields(
        runtime_model.species_templates,
        fields=state_fields,
        mesh=mesh,
    )
    if ion_name not in species:
        raise KeyError(f"Unknown ion_name={ion_name!r}.")
    expected_mass = species[ion_name].atomic_mass

    def wrapped(**kwargs):
        terms = original(**kwargs)
        if kwargs["atomic_mass"] == expected_mass:
            captured_terms[ion_name] = terms
        return terms

    with mock.patch.object(recycling_1d_mod, "_assemble_ion_rhs_terms", side_effect=wrapped):
        recycling_1d_mod._compute_recycling_1d_rhs_from_species(
            config,
            species=species,
            controllers=runtime_model.controllers,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            feedback_integrals=feedback_integrals,
            explicit_pressure_sources=runtime_model.explicit_pressure_sources,
            density_source_overrides=runtime_model.density_source_overrides,
            pressure_source_overrides=runtime_model.pressure_source_overrides,
            momentum_source_overrides=runtime_model.momentum_source_overrides,
            preserve_dump_target_state=runtime_model.preserve_dump_target_state,
            preserve_dump_ion_target_state_only=runtime_model.preserve_dump_ion_target_state_only,
        )

    return captured_terms[ion_name]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare state and RHS terms at selected recycling boundary-adjacent cells on a short reference run.",
    )
    parser.add_argument("--case", default="recycling_1d_one_step", choices=("recycling_1d_one_step", "recycling_dthe_one_step"))
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--reference-binary", type=Path)
    parser.add_argument("--timestep", type=float, default=25.0)
    parser.add_argument("--native-solver-mode", default="continuation")
    parser.add_argument("--cell", action="append", dest="cells", help="Boundary-adjacent cell as x,y,z. May be repeated.")
    parser.add_argument("--state-field", action="append", dest="state_fields")
    parser.add_argument("--rhs-field", action="append", dest="rhs_fields")
    parser.add_argument("--ion-name", default="d+")
    args = parser.parse_args()

    state_fields = tuple(args.state_fields) if args.state_fields else DEFAULT_STATE_FIELDS
    rhs_fields = tuple(args.rhs_fields) if args.rhs_fields else DEFAULT_RHS_FIELDS

    workdir = Path(tempfile.mkdtemp(prefix=f"jaxdrb-{args.case}-boundary-cell-"))
    run_reference_case(
        args.case,
        reference_root=args.reference_root,
        reference_binary=args.reference_binary,
        workdir=workdir,
        extra_overrides=(f"timestep={args.timestep:g}", "solver:mxstep=50000"),
    )
    dump_path = workdir / "BOUT.dmp.0.nc"
    restart_path = workdir / "BOUT.restart.0.nc"

    config = load_bout_input(workdir / "BOUT.inp")
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    dataset_scalars = resolved_dataset_scalars(run_config)

    reference_state = _load_last_fields(dump_path, state_fields)
    reference_rhs_dump = _load_last_fields(dump_path, rhs_fields)
    snapshot = extract_recycling_controller_snapshot(
        dump_path,
        restart_path,
        controller_species=tuple(name[1:] for name in state_fields if name.startswith("N") and name.endswith("+") and not name.startswith("NV")),
    )
    native_history = advance_recycling_1d_implicit_history(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        timestep=float(args.timestep),
        steps=1,
        solver_mode=str(args.native_solver_mode),
        residual_tolerance=float(config.parsed("solver", "rtol")) if config.has_option("solver", "rtol") else 1.0e-8,
        max_nonlinear_iterations=30,
    )
    native_state = {
        name: np.asarray(native_history.variable_history[name][1], dtype=np.float64)
        for name in state_fields
        if name in native_history.variable_history
    }
    native_integrals = {
        name: float(history[1])
        for name, history in native_history.feedback_integral_history.items()
    }

    reference_rhs_eval = compute_recycling_1d_rhs(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        field_overrides=reference_state,
        feedback_integrals=snapshot.restart_integrals,
    ).variables
    native_rhs_eval = compute_recycling_1d_rhs(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        field_overrides=native_state,
        feedback_integrals=native_integrals,
    ).variables
    reference_ion_terms = _extract_ion_rhs_terms(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        state_fields=reference_state,
        feedback_integrals=snapshot.restart_integrals,
        ion_name=str(args.ion_name),
    )
    native_ion_terms = _extract_ion_rhs_terms(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        state_fields=native_state,
        feedback_integrals=native_integrals,
        ion_name=str(args.ion_name),
    )

    cells = tuple(_parse_cell(raw) for raw in args.cells) if args.cells else default_recycling_boundary_cells(mesh)
    print(f"case={args.case}")
    print(f"workdir={workdir}")
    print(f"timestep={args.timestep:g}")
    print(f"native_solver_mode={args.native_solver_mode}")
    for cell in cells:
        print(f"cell={cell}")
        for field in state_fields:
            if field not in reference_state or field not in native_state:
                continue
            native_value = _cell_value(native_state[field], cell)
            reference_value = _cell_value(reference_state[field], cell)
            print(
                f"  state {field}: native={native_value:.12e} "
                f"reference={reference_value:.12e} diff={native_value - reference_value:.12e}"
            )
        for field in rhs_fields:
            if field not in reference_rhs_eval or field not in native_rhs_eval:
                continue
            native_value = _cell_value(native_rhs_eval[field], cell)
            reference_value = _cell_value(reference_rhs_eval[field], cell)
            dump_value = (
                _cell_value(reference_rhs_dump[field], cell)
                if field in reference_rhs_dump
                else float("nan")
            )
            print(
                f"  rhs {field}: native={native_value:.12e} "
                f"reference_eval={reference_value:.12e} reference_dump={dump_value:.12e} "
                f"diff={native_value - reference_value:.12e}"
            )
        for term_name in DEFAULT_ION_TERM_NAMES:
            native_value = _cell_value(getattr(native_ion_terms, term_name), cell)
            reference_value = _cell_value(getattr(reference_ion_terms, term_name), cell)
            print(
                f"  ion {args.ion_name} {term_name}: native={native_value:.12e} "
                f"reference_eval={reference_value:.12e} diff={native_value - reference_value:.12e}"
            )


if __name__ == "__main__":
    main()
