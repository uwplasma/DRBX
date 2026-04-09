#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from jax_drb.config.boutinp import apply_bout_overrides, load_bout_input
from jax_drb.native.recycling_1d import (
    _apply_collision_closure,
    _build_recycling_runtime_model,
    _override_species_fields,
    _prepare_open_field_states,
)
from jax_drb.native.reference_dump import (
    load_local_reference_snapshot_cache,
    load_optional_field_history_cache,
    synthesize_local_reference_snapshot_from_active_history,
)
from jax_drb.native.runner import _apply_species_velocity_overrides, _species_optional_velocity_field_map
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.parity.reference import (
    make_default_overrides,
    merge_overrides,
    resolve_reference_case,
    _resolve_override_placeholders,
)
from jax_drb.runtime.run_config import RunConfiguration


STATE_FIELDS = (
    "Nd+",
    "Pd+",
    "NVd+",
    "Nd",
    "Pd",
    "NVd",
    "Nt+",
    "Pt+",
    "NVt+",
    "Nt",
    "Pt",
    "NVt",
    "Nhe+",
    "Phe+",
    "NVhe+",
    "Nhe",
    "Phe",
    "NVhe",
    "Pe",
)
SCALAR_FIELDS = ("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0")
OPTIONAL_VELOCITY_FIELDS = ("Vd", "Vt", "Vhe")


def _load_case_config(case_name: str, *, reference_root: Path):
    case, input_path = resolve_reference_case(case_name, reference_root=reference_root)
    overrides = merge_overrides(
        make_default_overrides(case.parity_mode),
        _resolve_override_placeholders(case.extra_overrides, reference_root=reference_root),
    )
    config = apply_bout_overrides(load_bout_input(input_path), overrides)
    return case, config


def _load_reference_final_snapshot(case_name: str, *, reference_root: Path):
    repo_root = Path(__file__).resolve().parents[1]
    snapshot_root = repo_root / "references" / "baselines" / "reference_snapshots"
    initial_snapshot = load_local_reference_snapshot_cache(
        snapshot_root / f"{case_name.removesuffix('_one_step')}_rhs_snapshot.npz",
        field_names=STATE_FIELDS,
        scalar_names=SCALAR_FIELDS,
    )
    return synthesize_local_reference_snapshot_from_active_history(
        initial_snapshot=initial_snapshot,
        array_history_path=repo_root / "references" / "baselines" / "reference_arrays" / f"{case_name}.npz",
        optional_history_path=snapshot_root / f"{case_name}_optional_history.npz",
        timestep=0.1,
        state_field_names=STATE_FIELDS,
        optional_field_names=(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the tokamak recycling ion-viscosity blocker on the committed one-step baseline. "
            "Print the local collision and DivPiPar contributions at selected full-grid cells."
        )
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("/Users/rogerio/local/hermes-3"),
        help="Path to the local Hermes-3 checkout.",
    )
    parser.add_argument(
        "--case",
        default="tokamak_recycling_dthe_one_step",
        choices=("tokamak_recycling_dthe_one_step",),
        help="Direct tokamak recycling one-step case to inspect.",
    )
    parser.add_argument(
        "--cell",
        action="append",
        default=[],
        help="Full-grid cell as x,y,z. Repeat to inspect multiple cells. Defaults to the first active lower target cell.",
    )
    args = parser.parse_args()

    reference_root = args.reference_root.expanduser().resolve()
    _, config = _load_case_config(args.case, reference_root=reference_root)
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)
    reference_final = _load_reference_final_snapshot(args.case, reference_root=reference_root)

    repo_root = Path(__file__).resolve().parents[1]
    optional_history = load_optional_field_history_cache(
        repo_root / "references" / "baselines" / "reference_snapshots" / f"{args.case}_optional_history.npz",
        field_names=OPTIONAL_VELOCITY_FIELDS,
    )
    velocity_field_overrides = {
        name: np.asarray(optional_history[field_name][1], dtype=np.float64)
        for name, field_name in _species_optional_velocity_field_map(config)
        if field_name in optional_history
    }

    field_overrides = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in reference_final.fields.items()
        if name in STATE_FIELDS
    }
    if velocity_field_overrides:
        field_overrides = _apply_species_velocity_overrides(
            config,
            field_overrides=field_overrides,
            velocity_field_overrides=velocity_field_overrides,
        )

    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=reference_final.mesh,
        dataset_scalars=dataset_scalars,
        field_overrides=field_overrides,
        preserve_dump_target_state=True,
        preserve_dump_ion_target_state_only=True,
    )
    species = _override_species_fields(runtime_model.species_templates, fields=field_overrides, mesh=reference_final.mesh)
    prepared, _, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=reference_final.mesh,
        metrics=reference_final.metrics,
        dataset_scalars=dataset_scalars,
        apply_sheath_boundaries=True,
        preserve_dump_target_state=True,
        preserve_dump_ion_target_state_only=True,
    )
    collision_terms = _apply_collision_closure(
        config,
        species,
        prepared,
        mesh=reference_final.mesh,
        metrics=reference_final.metrics,
        dataset_scalars=dataset_scalars,
    )

    if args.cell:
        cells = tuple(tuple(int(part) for part in spec.split(",")) for spec in args.cell)
    else:
        cells = ((reference_final.mesh.xstart, reference_final.mesh.ystart, 0),)

    for cell in cells:
        x_index, y_index, z_index = cell
        print(f"CELL x={x_index} y={y_index} z={z_index}")
        for species_name in ("d+", "t+", "he+"):
            if species_name not in species:
                continue
            div_pi = float(collision_terms.diagnostics.get(f"DivPiPar_{species_name}", np.zeros((1, 1, 1)))[x_index, y_index, z_index])
            momentum = float(collision_terms.momentum_source[species_name][x_index, y_index, z_index])
            energy = float(collision_terms.energy_source[species_name][x_index, y_index, z_index])
            velocity = float(prepared[species_name].velocity[x_index, y_index, z_index])
            pressure = float(prepared[species_name].pressure[x_index, y_index, z_index])
            print(
                f"  {species_name}: "
                f"DivPiPar={div_pi:.8e} "
                f"momentum_source={momentum:.8e} "
                f"energy_source={energy:.8e} "
                f"velocity={velocity:.8e} "
                f"pressure={pressure:.8e}"
            )
        print()


if __name__ == "__main__":
    main()
