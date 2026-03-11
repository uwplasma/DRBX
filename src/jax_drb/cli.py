from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config.boutinp import load_bout_input
from .reference.cases import resolve_reference_cases
from .runtime.run_config import RunConfiguration


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

    cases_parser = subparsers.add_parser(
        "reference-cases",
        help="Inspect the curated Hermes reference cases and report their resolved run configuration.",
    )
    cases_parser.add_argument(
        "--hermes-root",
        type=Path,
        default=_default_hermes_root(),
        help="Path to a Hermes-3 checkout used for case inspection.",
    )
    cases_parser.set_defaults(command=_reference_cases_command)

    run_case_parser = subparsers.add_parser(
        "run-reference-case",
        help="Stage, run, and summarize a curated Hermes reference case in an isolated workdir.",
    )
    run_case_parser.add_argument("case_name")
    run_case_parser.add_argument(
        "--hermes-root",
        type=Path,
        default=_default_hermes_root(),
        help="Path to a Hermes-3 checkout used for case lookup and default binary discovery.",
    )
    run_case_parser.add_argument("--hermes-binary", type=Path, default=_default_hermes_binary())
    run_case_parser.add_argument("--workdir", type=Path, default=None)
    run_case_parser.add_argument("--override", action="append", default=[], help="Additional Hermes overrides such as nout=0.")
    run_case_parser.add_argument("--json-out", type=Path, default=None, help="Write the run summary to a JSON file.")
    run_case_parser.set_defaults(command=_run_reference_case_command)

    compare_parser = subparsers.add_parser(
        "compare-summary",
        help="Compare an actual portable run summary JSON against an expected baseline JSON.",
    )
    compare_parser.add_argument("expected_json", type=Path)
    compare_parser.add_argument("actual_json", type=Path)
    compare_parser.add_argument("--scalar-rtol", type=float, default=1e-10)
    compare_parser.add_argument("--scalar-atol", type=float, default=1e-12)
    compare_parser.set_defaults(command=_compare_summary_command)

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
    run_config = RunConfiguration.from_config(config)

    print(f"input: {args.input_file}")
    print(f"sections: {', '.join(config.section_names())}")
    print(f"time: nout={run_config.time.nout}, timestep={run_config.time.timestep:g}")
    print(
        "mesh: "
        f"nx={run_config.mesh.nx}, ny={run_config.mesh.ny}, nz={run_config.mesh.nz}, "
        f"MXG={run_config.mesh.mxg}, MYG={run_config.mesh.myg}, "
        f"parallel_transform={run_config.mesh.parallel_transform.type}"
    )
    print(f"scheduled components: {', '.join(request.label for request in run_config.components)}")

    if run_config.normalization is not None:
        normalization = run_config.normalization
        print(
            "normalization: "
            f"Nnorm={normalization.Nnorm:g}, "
            f"Tnorm={normalization.Tnorm:g}, "
            f"Bnorm={normalization.Bnorm:g}, "
            f"Cs0={normalization.Cs0:.8e}, "
            f"Omega_ci={normalization.Omega_ci:.8e}, "
            f"rho_s0={normalization.rho_s0:.8e}"
        )
    else:
        print("normalization: unresolved (missing one or more of Nnorm, Tnorm, Bnorm)")

    return 0


def _reference_cases_command(args: argparse.Namespace) -> int:
    if args.hermes_root is None:
        print("reference-cases: set --hermes-root or JAX_DRB_HERMES_ROOT to a Hermes-3 checkout.")
        return 1

    resolved_cases = resolve_reference_cases(args.hermes_root)
    for resolved in resolved_cases:
        status = "missing" if not resolved.exists else resolved.case.parity_mode
        print(f"{resolved.case.name}: {status} -> {resolved.input_path}")
        if resolved.run_config is None:
            continue
        print(
            "  "
            f"nout={resolved.run_config.time.nout}, "
            f"timestep={resolved.run_config.time.timestep:g}, "
            f"components={','.join(request.label for request in resolved.run_config.components)}"
        )
    return 0


def _run_command(args: argparse.Namespace) -> int:
    if args.dry_run:
        return _inspect_command(args)
    print("Transient execution is not implemented yet. Use --dry-run for configuration parity checks.")
    return 1


def _default_hermes_root() -> Path | None:
    value = os.environ.get("JAX_DRB_HERMES_ROOT")
    return Path(value) if value else None


def _default_hermes_binary() -> Path | None:
    value = os.environ.get("JAX_DRB_HERMES_BINARY")
    return Path(value) if value else None


def _run_reference_case_command(args: argparse.Namespace) -> int:
    from .parity.hermes import run_reference_case, write_case_baseline_json

    if args.hermes_root is None:
        print("run-reference-case: set --hermes-root or JAX_DRB_HERMES_ROOT.")
        return 1

    result = run_reference_case(
        args.case_name,
        hermes_root=args.hermes_root,
        hermes_binary=args.hermes_binary,
        workdir=args.workdir,
        extra_overrides=args.override,
    )
    summary = result.summary
    print(f"case: {summary.case_name}")
    print(f"parity_mode: {summary.parity_mode}")
    print(f"workdir: {summary.workdir}")
    print(f"overrides: {', '.join(summary.overrides) if summary.overrides else '(none)'}")
    print(f"time_points: {summary.time_points}")
    print(f"compare_variables: {', '.join(summary.compare_variables) if summary.compare_variables else '(none)'}")
    for name, variable in summary.variable_summaries.items():
        delta = "n/a" if variable.max_abs_delta_last_first is None else f"{variable.max_abs_delta_last_first:.8e}"
        print(
            f"  {name}: shape={variable.shape}, min={variable.minimum:.8e}, "
            f"max={variable.maximum:.8e}, mean={variable.mean:.8e}, delta={delta}"
        )
    if args.json_out is not None:
        path = write_case_baseline_json(summary, args.json_out)
        print(f"json_out: {path}")
    return 0


def _compare_summary_command(args: argparse.Namespace) -> int:
    from .parity.compare import compare_summary_payloads, load_summary_json

    expected = load_summary_json(args.expected_json)
    actual = load_summary_json(args.actual_json)
    result = compare_summary_payloads(
        expected,
        actual,
        scalar_rtol=args.scalar_rtol,
        scalar_atol=args.scalar_atol,
    )
    if result.ok:
        print("comparison: ok")
        return 0
    print("comparison: mismatch")
    for issue in result.issues:
        print(f"  {issue.field}: {issue.message}")
    return 1
