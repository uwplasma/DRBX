from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native import run_curated_case
from jax_drb.native.recycling_1d import compute_recycling_1d_rhs
from jax_drb.native.reference_dump import load_local_reference_snapshot
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.native.units import resolved_dataset_scalars


def benchmark_integrated_2d_recycling_rhs(
    *,
    reference_root: Path,
    dump_path: Path,
    input_path: Path,
    repeats: int,
) -> dict[str, float | int | str]:
    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)

    start = perf_counter()
    run_curated_case("integrated_2d_recycling_rhs", reference_root=reference_root)
    curated_case_seconds = perf_counter() - start

    start = perf_counter()
    snapshot = load_local_reference_snapshot(
        dump_path,
        field_names=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe"),
        optional_field_names=("Ne", "SPd+", "Sd_target_recycle", "Ed_target_recycle", "is_pump"),
        scalar_names=("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0"),
    )
    snapshot_load_seconds = perf_counter() - start

    pressure_source_overrides = (
        {"d+": np.asarray(snapshot.optional_fields["SPd+"], dtype=np.float64)}
        if "SPd+" in snapshot.optional_fields
        else None
    )

    start = perf_counter()
    compute_recycling_1d_rhs(
        config,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
        dataset_scalars=dataset_scalars,
        field_overrides=snapshot.fields,
        apply_sheath_boundaries=True,
        preserve_dump_target_state=True,
        pressure_source_overrides=pressure_source_overrides,
    )
    first_compute_seconds = perf_counter() - start

    start = perf_counter()
    for _ in range(repeats):
        compute_recycling_1d_rhs(
            config,
            mesh=snapshot.mesh,
            metrics=snapshot.metrics,
            dataset_scalars=dataset_scalars,
            field_overrides=snapshot.fields,
            apply_sheath_boundaries=True,
            preserve_dump_target_state=True,
            pressure_source_overrides=pressure_source_overrides,
        )
    repeat_compute_seconds = (perf_counter() - start) / float(max(repeats, 1))

    return {
        "case": "integrated_2d_recycling_rhs",
        "input_path": str(input_path),
        "dump_path": str(dump_path),
        "reference_root": str(reference_root),
        "repeats": int(repeats),
        "curated_case_seconds": float(curated_case_seconds),
        "snapshot_load_seconds": float(snapshot_load_seconds),
        "first_compute_seconds": float(first_compute_seconds),
        "repeat_compute_seconds": float(repeat_compute_seconds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the staged integrated 2D recycling RHS path.")
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("/Users/rogerio/local/hermes-3"),
        help="Reference checkout used by the staged integrated 2D case.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("/Users/rogerio/local/hermes-3/tests/integrated/2D-recycling/data/BOUT.inp"),
        help="Integrated 2D recycling input file.",
    )
    parser.add_argument(
        "--dump-path",
        type=Path,
        default=Path("/Users/rogerio/local/hermes-3/tests/integrated/2D-recycling/data/BOUT.dmp.0.nc"),
        help="Local reference dump used for the direct staged benchmark.",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Number of repeated direct RHS evaluations.")
    args = parser.parse_args()

    payload = benchmark_integrated_2d_recycling_rhs(
        reference_root=args.reference_root,
        dump_path=args.dump_path,
        input_path=args.input_path,
        repeats=args.repeats,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
