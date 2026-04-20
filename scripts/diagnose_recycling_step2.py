#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.recycling_1d import compute_recycling_1d_rhs
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.parity.recycling import extract_recycling_controller_snapshot
from jax_drb.runtime.run_config import RunConfiguration


DEFAULT_COMPARE_FIELDS = (
    "Ve",
    "Epar",
    "SNVd+",
    "SNVt+",
    "SNVhe+",
    "SNVd",
    "SNVt",
    "SNVhe",
    "Fd+d_coll",
    "Fd+e_coll",
    "Ft+t_coll",
    "Ft+e_coll",
    "DivPiPar_d+",
    "DivPiPar_t+",
    "DivPiPar_he+",
    "Fd+_iz",
    "Ft+_iz",
    "Fd+_rec",
    "Ft+_rec",
    "Fdt+_cx",
    "Ft+d_cx",
    "Ftd+_cx",
    "Fd+t_cx",
    "Fdd+_cx",
    "Ftt+_cx",
)

STATE_FIELDS = (
    "Nd+",
    "Pd+",
    "NVd+",
    "Nt+",
    "Pt+",
    "NVt+",
    "Nhe+",
    "Phe+",
    "NVhe+",
    "Nd",
    "Pd",
    "NVd",
    "Nt",
    "Pt",
    "NVt",
    "Nhe",
    "Phe",
    "NVhe",
    "Pe",
)


def _load_reference_fields(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    loaded: dict[str, np.ndarray] = {}
    with Dataset(path) as dataset:
        for name in names:
            if name in dataset.variables:
                loaded[name] = np.asarray(dataset.variables[name][-1], dtype=np.float64)
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare localized Step 2 recycling diagnostics against a staged reference dump.")
    parser.add_argument("--input", required=True, type=Path, help="Path to the reference input file.")
    parser.add_argument("--reference-dir", required=True, type=Path, help="Directory containing BOUT.dmp.0.nc and BOUT.restart.0.nc.")
    parser.add_argument(
        "--controller-species",
        nargs="+",
        default=("d+", "t+", "he+"),
        help="Controller species names used to restore feedback integrals.",
    )
    parser.add_argument(
        "--field",
        action="append",
        dest="fields",
        help="Diagnostic field name to compare. May be repeated. Defaults to the standard Step 2 diagnostic set.",
    )
    args = parser.parse_args()

    compare_fields = tuple(args.fields) if args.fields else DEFAULT_COMPARE_FIELDS
    dump_path = args.reference_dir / "BOUT.dmp.0.nc"
    restart_path = args.reference_dir / "BOUT.restart.0.nc"

    config = load_bout_input(args.input)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    snapshot = extract_recycling_controller_snapshot(
        dump_path,
        restart_path,
        controller_species=tuple(args.controller_species),
    )
    field_overrides = _load_reference_fields(dump_path, STATE_FIELDS)
    reference = _load_reference_fields(dump_path, compare_fields)
    actual = compute_recycling_1d_rhs(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        field_overrides=field_overrides,
        feedback_integrals=snapshot.restart_integrals,
    ).variables

    active = (slice(None), slice(mesh.xstart, mesh.xend + 1), slice(mesh.ystart, mesh.yend + 1), slice(None))
    print(f"input={args.input}")
    print(f"reference_dir={args.reference_dir}")
    print(f"active_y=[{mesh.ystart}, {mesh.yend}]")
    print("field max_abs_diff")
    for name in compare_fields:
        if name not in reference or name not in actual:
            print(f"{name} missing")
            continue
        diff = float(np.nanmax(np.abs(actual[name][active] - reference[name][None, ...][active])))
        print(f"{name} {diff:.12e}")


if __name__ == "__main__":
    main()
