#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _resolve_reference_root(args: argparse.Namespace) -> Path | None:
    if args.reference_root is not None:
        return args.reference_root.expanduser().resolve()
    value = os.environ.get("JAX_DRB_REFERENCE_ROOT")
    return None if not value else Path(value).expanduser().resolve()


def _resolve_input(args: argparse.Namespace) -> Path:
    if args.input_path is not None:
        return args.input_path.expanduser().resolve()
    root = _resolve_reference_root(args)
    if root is None:
        raise SystemExit("--reference-root, --input-path, or JAX_DRB_REFERENCE_ROOT is required.")
    case_dir = "1D-recycling-dthe" if args.case == "dthe" else "1D-recycling"
    return (root / "tests" / "integrated" / case_dir / "data" / "BOUT.inp").resolve()


def _parse_batch_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return sizes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile batched residual/JVP throughput for the fixed-layout recycling residual. "
            "This is the differentiability and parallel-throughput gate used before promoting "
            "heavier recycling solves."
        )
    )
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--case", choices=("hydrogen", "dthe"), default="dthe")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs") / "data" / "runtime_profile_artifacts" / "recycling_dthe_batched_jvp_gate",
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default=(1, 4, 16, 64))
    parser.add_argument("--timestep", type=float, default=1.0e-4)
    parser.add_argument("--perturbation-scale", type=float, default=1.0e-6)
    parser.add_argument("--fd-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--timed-runs", type=int, default=5)
    parser.add_argument("--disable-pmap", action="store_true")
    parser.add_argument(
        "--skip-objective-grad-check",
        action="store_true",
        help="Skip the reverse-mode objective-gradient check for bounded GPU throughput runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from jax_drb.validation.recycling_batched_jvp_profile import create_recycling_batched_jvp_profile_package

    report = create_recycling_batched_jvp_profile_package(
        _resolve_input(args),
        args.output_dir,
        overrides=tuple(args.override),
        batch_sizes=tuple(args.batch_sizes),
        timestep=float(args.timestep),
        perturbation_scale=float(args.perturbation_scale),
        fd_epsilon=float(args.fd_epsilon),
        timed_runs=int(args.timed_runs),
        enable_pmap=not bool(args.disable_pmap),
        check_objective_grad=not bool(args.skip_objective_grad_check),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
