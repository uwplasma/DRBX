from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.native.reference_dump import LocalReferenceSnapshot, load_local_reference_snapshot, save_local_reference_snapshot_cache
from jax_drb.native.runner import _direct_recycling_state_field_names, _load_curated_case_config, _open_field_initial_rhs_case_name, _open_field_snapshot_cache_path
from jax_drb.parity.reference import resolve_reference_case, run_reference_case
from jax_drb.reference.paths import default_reference_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate committed snapshot caches for curated open-field recycling dump-backed cases.",
    )
    parser.add_argument("case_name")
    parser.add_argument("--reference-root", type=Path, default=default_reference_root())
    parser.add_argument("--workdir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.reference_root is None:
        raise SystemExit("Set --reference-root or JAX_DRB_REFERENCE_ROOT before generating reference caches.")
    rhs_case_name = _open_field_initial_rhs_case_name(args.case_name)
    case, input_path = resolve_reference_case(rhs_case_name, reference_root=args.reference_root)
    config = _load_curated_case_config(case, input_path)
    field_names = _direct_recycling_state_field_names(config)
    execution = run_reference_case(
        rhs_case_name,
        reference_root=args.reference_root,
        workdir=args.workdir,
        keep_workdir=True,
    )
    dump_path = Path(execution.summary.artifacts["BOUT.dmp.0.nc"])
    snapshot = load_local_reference_snapshot(
        dump_path,
        field_names=field_names,
        scalar_names=("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0"),
        time_index=0,
    )
    path = _open_field_snapshot_cache_path(rhs_case_name)
    save_local_reference_snapshot_cache(
        LocalReferenceSnapshot(
            mesh=snapshot.mesh,
            metrics=snapshot.metrics,
            fields=snapshot.fields,
            optional_fields={},
            scalar_values=snapshot.scalar_values,
        ),
        path,
    )
    print(f"cached: {rhs_case_name}")
    print(f"  snapshot: {path}")
    print(f"  dump:     {dump_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
