from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from jax_drb.native.reference_dump import (
    LocalReferenceSnapshot,
    load_local_reference_snapshot,
    save_local_reference_snapshot_cache,
    save_optional_field_history_cache,
)
from jax_drb.parity.reference import run_reference_case
from jax_drb.reference.cases import load_reference_cases


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SNAPSHOT_DIR = _REPO_ROOT / "references" / "baselines" / "reference_snapshots"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate committed snapshot/history caches for a direct tokamak dump-backed case.",
    )
    parser.add_argument("case_name")
    parser.add_argument("--reference-root", type=Path, default=Path("/Users/rogerio/local/hermes-3"))
    parser.add_argument("--workdir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    case = next((case for case in load_reference_cases() if case.name == args.case_name), None)
    if case is None:
        raise SystemExit(f"Unknown case {args.case_name!r}")
    if not case.name.startswith("tokamak_"):
        raise SystemExit(f"Case {args.case_name!r} is not a direct tokamak dump-backed rung")

    execution = run_reference_case(
        case.name,
        reference_root=args.reference_root,
        workdir=args.workdir,
        keep_workdir=True,
    )
    dump_path = Path(execution.summary.artifacts["BOUT.dmp.0.nc"])
    time_count = len(execution.summary.time_points)

    initial_snapshot = load_local_reference_snapshot(
        dump_path,
        field_names=(),
        optional_field_names=(),
        scalar_names=("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0"),
        time_index=0,
    )
    save_local_reference_snapshot_cache(
        LocalReferenceSnapshot(
            mesh=initial_snapshot.mesh,
            metrics=initial_snapshot.metrics,
            fields={},
            optional_fields={},
            scalar_values=initial_snapshot.scalar_values,
        ),
        _SNAPSHOT_DIR / f"{case.name}_snapshot.npz",
    )

    history = {name: [] for name in case.compare_variables}
    for time_index in range(time_count):
        snapshot = load_local_reference_snapshot(
            dump_path,
            field_names=case.compare_variables,
            optional_field_names=(),
            scalar_names=(),
            time_index=time_index,
        )
        for name in case.compare_variables:
            history[name].append(snapshot.fields[name])

    save_optional_field_history_cache(
        {name: np.stack(values, axis=0) for name, values in history.items()},
        _SNAPSHOT_DIR / f"{case.name}_field_history.npz",
    )

    print(f"cached: {case.name}")
    print(f"  snapshot: {_SNAPSHOT_DIR / f'{case.name}_snapshot.npz'}")
    print(f"  history:  {_SNAPSHOT_DIR / f'{case.name}_field_history.npz'}")
    print(f"  dump:     {dump_path}")
    print(f"  points:   {time_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
