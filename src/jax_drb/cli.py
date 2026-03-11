from __future__ import annotations

import argparse
from pathlib import Path

from .config.boutinp import load_bout_input
from .config.normalization import HermesNormalization
from .runtime.scheduler import expand_component_requests


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.command(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jax-drb",
        description="Inspect or run JAX-DRB inputs using Hermes-compatible configuration structure.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=False)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a BOUT.inp file and print the resolved plan.")
    inspect_parser.add_argument("input_file", type=Path)
    inspect_parser.set_defaults(command=_inspect_command)

    run_parser = subparsers.add_parser("run", help="Prepare a run plan. Full time integration is not implemented yet.")
    run_parser.add_argument("input_file", type=Path)
    run_parser.add_argument("--dry-run", action="store_true", help="Only inspect configuration and exit successfully.")
    run_parser.set_defaults(command=_run_command)

    parser.set_defaults(command=_default_command)
    return parser


def _default_command(args: argparse.Namespace) -> int:
    if getattr(args, "subcommand", None) is None:
        raise SystemExit("Use `jax-drb inspect <BOUT.inp>` or `jax-drb run <BOUT.inp> --dry-run`.")
    return args.command(args)


def _inspect_command(args: argparse.Namespace) -> int:
    config = load_bout_input(args.input_file)
    requests = expand_component_requests(config)

    print(f"input: {args.input_file}")
    print(f"sections: {', '.join(config.section_names())}")
    print(f"scheduled components: {', '.join(request.label for request in requests)}")

    try:
        normalization = HermesNormalization.from_config(config)
    except Exception as exc:  # pragma: no cover - exercised by CLI smoke coverage only
        print(f"normalization: unresolved ({exc})")
    else:
        print(
            "normalization: "
            f"Nnorm={normalization.Nnorm:g}, "
            f"Tnorm={normalization.Tnorm:g}, "
            f"Bnorm={normalization.Bnorm:g}, "
            f"Cs0={normalization.Cs0:.8e}, "
            f"Omega_ci={normalization.Omega_ci:.8e}, "
            f"rho_s0={normalization.rho_s0:.8e}"
        )

    return 0


def _run_command(args: argparse.Namespace) -> int:
    if args.dry_run:
        return _inspect_command(args)
    print("Transient execution is not implemented yet. Use --dry-run for configuration parity checks.")
    return 1
