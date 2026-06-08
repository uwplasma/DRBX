from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

from .config.boutinp import load_bout_input
from .reference.cases import resolve_reference_cases
from .reference.paths import default_reference_root as resolve_default_reference_root
from .runtime import configure_jax_runtime, resolve_runtime_precision
from .runtime.run_config import RunConfiguration


def main(argv: list[str] | None = None) -> int:
    normalized_argv = _normalize_cli_argv(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(normalized_argv)
    return args.command(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jax_drb",
        description="Inspect or run JAX-DRB inputs using the native model configuration structure.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=False)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect an input deck and print the resolved plan."
    )
    inspect_parser.add_argument("input_file", type=Path)
    inspect_parser.set_defaults(command=_inspect_command)

    cases_parser = subparsers.add_parser(
        "reference-cases",
        help="Inspect the curated reference cases and report their resolved run configuration.",
    )
    cases_parser.add_argument(
        "--reference-root",
        type=Path,
        default=_default_reference_root(),
        help="Path to the external benchmark checkout used for curated-case inspection.",
    )
    cases_parser.set_defaults(command=_reference_cases_command)

    run_case_parser = subparsers.add_parser(
        "run-reference-case",
        help="Stage, run, and summarize a curated reference case in an isolated workdir.",
    )
    run_case_parser.add_argument("case_name")
    run_case_parser.add_argument(
        "--reference-root",
        type=Path,
        default=_default_reference_root(),
        help="Path to the external benchmark checkout used for case lookup and default binary discovery.",
    )
    run_case_parser.add_argument(
        "--reference-binary", type=Path, default=_default_reference_binary()
    )
    run_case_parser.add_argument("--workdir", type=Path, default=None)
    run_case_parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional source-style overrides such as nout=0.",
    )
    run_case_parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write the run summary to a JSON file.",
    )
    run_case_parser.add_argument(
        "--arrays-out",
        type=Path,
        default=None,
        help="Write the full comparison arrays to a compressed NPZ.",
    )
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

    compare_arrays_parser = subparsers.add_parser(
        "compare-arrays",
        help="Compare an actual portable array NPZ against an expected baseline NPZ.",
    )
    compare_arrays_parser.add_argument("expected_npz", type=Path)
    compare_arrays_parser.add_argument("actual_npz", type=Path)
    compare_arrays_parser.add_argument("--scalar-rtol", type=float, default=1e-10)
    compare_arrays_parser.add_argument("--scalar-atol", type=float, default=1e-12)
    compare_arrays_parser.add_argument("--array-rtol", type=float, default=1e-10)
    compare_arrays_parser.add_argument("--array-atol", type=float, default=1e-12)
    compare_arrays_parser.set_defaults(command=_compare_arrays_command)

    compare_recycling_parser = subparsers.add_parser(
        "compare-recycling",
        help="Compare compact recycling reference and native artifacts with worst-variable/cell localization.",
    )
    compare_recycling_parser.add_argument("expected_artifact", type=Path)
    compare_recycling_parser.add_argument("actual_artifact", type=Path)
    compare_recycling_parser.add_argument(
        "--artifact-kind",
        choices=("auto", "summary", "arrays"),
        default="auto",
        help="Force JSON summary or NPZ array comparison instead of inferring from file suffixes.",
    )
    compare_recycling_parser.add_argument("--scalar-rtol", type=float, default=1e-10)
    compare_recycling_parser.add_argument("--scalar-atol", type=float, default=1e-12)
    compare_recycling_parser.add_argument("--array-rtol", type=float, default=1e-10)
    compare_recycling_parser.add_argument("--array-atol", type=float, default=1e-12)
    compare_recycling_parser.set_defaults(command=_compare_recycling_command)

    run_case_parser = subparsers.add_parser(
        "run-case",
        help="Run a curated case through the native JAX implementation and emit a portable summary.",
    )
    run_case_parser.add_argument("case_name")
    run_case_parser.add_argument(
        "--reference-root",
        type=Path,
        default=_default_reference_root(),
        help="Path to the external benchmark checkout used to locate the curated input file.",
    )
    run_case_parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write the portable summary to JSON.",
    )
    run_case_parser.add_argument(
        "--arrays-out",
        type=Path,
        default=None,
        help="Write the full comparison arrays to a compressed NPZ.",
    )
    run_case_parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional native-run overrides such as runtime:neutral_mixed_internal_substeps=8.",
    )
    run_case_parser.set_defaults(command=_run_case_command)

    validate_reference_parser = subparsers.add_parser(
        "validate-reference-baselines",
        help="Re-run committed reference cases and compare the live outputs to the stored summary baselines.",
    )
    validate_reference_parser.add_argument(
        "--reference-root",
        type=Path,
        default=_default_reference_root(),
        help="Path to the external benchmark checkout used for case lookup and binary discovery.",
    )
    validate_reference_parser.add_argument(
        "--reference-binary", type=Path, default=_default_reference_binary()
    )
    validate_reference_parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Specific case name to validate. Repeat to validate multiple cases.",
    )
    validate_reference_parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "references"
        / "baselines"
        / "reference",
        help="Directory containing committed reference summary baselines.",
    )
    validate_reference_parser.set_defaults(
        command=_validate_reference_baselines_command
    )

    analyze_drift_wave_parser = subparsers.add_parser(
        "analyze-drift-wave",
        help="Analyze a stored drift-wave array payload and report measured vs. analytic benchmark scalars.",
    )
    analyze_drift_wave_parser.add_argument("input_file", type=Path)
    analyze_drift_wave_parser.add_argument("arrays_npz", type=Path)
    analyze_drift_wave_parser.add_argument("--density-variable", default="Ni")
    analyze_drift_wave_parser.add_argument("--x-index", type=int, default=0)
    analyze_drift_wave_parser.add_argument("--y-index", type=int, default=0)
    analyze_drift_wave_parser.add_argument("--fit-points", type=int, default=10)
    analyze_drift_wave_parser.add_argument("--json-out", type=Path, default=None)
    analyze_drift_wave_parser.add_argument("--plot-out", type=Path, default=None)
    analyze_drift_wave_parser.set_defaults(command=_analyze_drift_wave_command)

    analyze_alfven_wave_parser = subparsers.add_parser(
        "analyze-alfven-wave",
        help="Analyze a stored Alfven-wave array payload and report measured vs. analytic benchmark scalars.",
    )
    analyze_alfven_wave_parser.add_argument("input_file", type=Path)
    analyze_alfven_wave_parser.add_argument("arrays_npz", type=Path)
    analyze_alfven_wave_parser.add_argument("--field-variable", default="phi")
    analyze_alfven_wave_parser.add_argument("--x-index", type=int, default=2)
    analyze_alfven_wave_parser.add_argument("--json-out", type=Path, default=None)
    analyze_alfven_wave_parser.add_argument("--plot-out", type=Path, default=None)
    analyze_alfven_wave_parser.set_defaults(command=_analyze_alfven_wave_command)

    compare_alfven_wave_parser = subparsers.add_parser(
        "compare-alfven-wave",
        help="Compare two Alfven-wave array payloads and report benchmark plus transient parity metrics.",
    )
    compare_alfven_wave_parser.add_argument("input_file", type=Path)
    compare_alfven_wave_parser.add_argument("expected_npz", type=Path)
    compare_alfven_wave_parser.add_argument("actual_npz", type=Path)
    compare_alfven_wave_parser.add_argument("--field-variable", default="phi")
    compare_alfven_wave_parser.add_argument("--x-index", type=int, default=2)
    compare_alfven_wave_parser.add_argument("--json-out", type=Path, default=None)
    compare_alfven_wave_parser.add_argument("--plot-out", type=Path, default=None)
    compare_alfven_wave_parser.set_defaults(command=_compare_alfven_wave_command)

    compare_drift_wave_parser = subparsers.add_parser(
        "compare-drift-wave",
        help="Compare two drift-wave array payloads and report benchmark plus field-error parity metrics.",
    )
    compare_drift_wave_parser.add_argument("input_file", type=Path)
    compare_drift_wave_parser.add_argument("expected_npz", type=Path)
    compare_drift_wave_parser.add_argument("actual_npz", type=Path)
    compare_drift_wave_parser.add_argument("--density-variable", default="Ni")
    compare_drift_wave_parser.add_argument("--x-index", type=int, default=0)
    compare_drift_wave_parser.add_argument("--y-index", type=int, default=0)
    compare_drift_wave_parser.add_argument("--fit-points", type=int, default=10)
    compare_drift_wave_parser.add_argument("--json-out", type=Path, default=None)
    compare_drift_wave_parser.add_argument("--plot-out", type=Path, default=None)
    compare_drift_wave_parser.set_defaults(command=_compare_drift_wave_command)

    compare_blob2d_parser = subparsers.add_parser(
        "compare-blob2d",
        help="Compare blob2d benchmark artifacts and report peak-amplitude and center-of-mass parity metrics.",
    )
    compare_blob2d_parser.add_argument("expected_artifact", type=Path)
    compare_blob2d_parser.add_argument("actual_artifact", type=Path)
    compare_blob2d_parser.add_argument("--density-variable", default="Ne")
    compare_blob2d_parser.add_argument("--background-density", type=float, default=1.0)
    compare_blob2d_parser.add_argument("--json-out", type=Path, default=None)
    compare_blob2d_parser.add_argument("--plot-out", type=Path, default=None)
    compare_blob2d_parser.set_defaults(command=_compare_blob2d_command)

    analyze_neutral_mixed_parser = subparsers.add_parser(
        "analyze-neutral-mixed",
        help="Analyze a stored neutral-mixed array payload and report compact transient metrics.",
    )
    analyze_neutral_mixed_parser.add_argument("arrays_npz", type=Path)
    analyze_neutral_mixed_parser.add_argument("--density-variable", default="Nh")
    analyze_neutral_mixed_parser.add_argument("--pressure-variable", default="Ph")
    analyze_neutral_mixed_parser.add_argument("--momentum-variable", default="NVh")
    analyze_neutral_mixed_parser.add_argument("--x-index", type=int, default=None)
    analyze_neutral_mixed_parser.add_argument("--y-index", type=int, default=None)
    analyze_neutral_mixed_parser.add_argument("--z-index", type=int, default=None)
    analyze_neutral_mixed_parser.add_argument("--json-out", type=Path, default=None)
    analyze_neutral_mixed_parser.add_argument("--plot-out", type=Path, default=None)
    analyze_neutral_mixed_parser.set_defaults(command=_analyze_neutral_mixed_command)

    compare_neutral_mixed_parser = subparsers.add_parser(
        "compare-neutral-mixed",
        help="Compare neutral-mixed analysis artifacts or array payloads and report compact parity metrics.",
    )
    compare_neutral_mixed_parser.add_argument("expected_artifact", type=Path)
    compare_neutral_mixed_parser.add_argument("actual_artifact", type=Path)
    compare_neutral_mixed_parser.add_argument("--density-variable", default="Nh")
    compare_neutral_mixed_parser.add_argument("--pressure-variable", default="Ph")
    compare_neutral_mixed_parser.add_argument("--momentum-variable", default="NVh")
    compare_neutral_mixed_parser.add_argument("--x-index", type=int, default=None)
    compare_neutral_mixed_parser.add_argument("--y-index", type=int, default=None)
    compare_neutral_mixed_parser.add_argument("--z-index", type=int, default=None)
    compare_neutral_mixed_parser.add_argument("--json-out", type=Path, default=None)
    compare_neutral_mixed_parser.add_argument("--plot-out", type=Path, default=None)
    compare_neutral_mixed_parser.set_defaults(command=_compare_neutral_mixed_command)

    neutral_substeps_parser = subparsers.add_parser(
        "diagnose-neutral-mixed-substeps",
        help="Sweep neutral-mixed internal substeps and rank hybrid-state NVh parity drivers.",
    )
    neutral_substeps_parser.add_argument(
        "--reference-root",
        type=Path,
        default=_default_reference_root(),
        help="Path to the external benchmark checkout used only when native histories must be generated.",
    )
    neutral_substeps_parser.add_argument(
        "--case-name", default="neutral_mixed_one_step"
    )
    neutral_substeps_parser.add_argument("--input-path", type=Path, default=None)
    neutral_substeps_parser.add_argument(
        "--reference-arrays-npz", type=Path, default=None
    )
    neutral_substeps_parser.add_argument("--substeps", default="1,2,3,4,6,8")
    neutral_substeps_parser.add_argument("--json-out", type=Path, default=None)
    neutral_substeps_parser.set_defaults(
        command=_diagnose_neutral_mixed_substeps_command
    )

    neutral_trace_parser = subparsers.add_parser(
        "trace-neutral-mixed-accepted-steps",
        help="Write a JAXDRB native accepted-internal-step trace for neutral-mixed NVh parity diagnostics.",
    )
    neutral_trace_parser.add_argument(
        "--reference-root",
        type=Path,
        default=_default_reference_root(),
        help="Path to the external benchmark checkout used only to resolve the input when --input-path is omitted.",
    )
    neutral_trace_parser.add_argument("--case-name", default="neutral_mixed_one_step")
    neutral_trace_parser.add_argument("--input-path", type=Path, default=None)
    neutral_trace_parser.add_argument("--internal-substeps", type=int, default=8)
    neutral_trace_parser.add_argument("--steps", type=int, default=1)
    neutral_trace_parser.add_argument(
        "--reference-trace-jsonl",
        "--reference-trace-json",
        dest="reference_trace_json",
        type=Path,
        default=None,
        help="Optional reference accepted-step JSONL/JSON trace whose times are replayed by the native solver.",
    )
    neutral_trace_parser.add_argument("--reference-stage", default="post_accepted")
    neutral_trace_parser.add_argument("--time-tolerance", type=float, default=1.0e-8)
    neutral_trace_parser.add_argument("--json-out", type=Path, required=True)
    neutral_trace_parser.set_defaults(
        command=_trace_neutral_mixed_accepted_steps_command
    )

    neutral_reference_trace_parser = subparsers.add_parser(
        "trace-neutral-mixed-reference-accepted-steps",
        help="Run a patched reference neutral-mixed case and write accepted-internal-step JSONL.",
    )
    neutral_reference_trace_parser.add_argument(
        "--reference-root", type=Path, required=True
    )
    neutral_reference_trace_parser.add_argument("--workdir", type=Path, required=True)
    neutral_reference_trace_parser.add_argument(
        "--hermes-binary", type=Path, default=None
    )
    neutral_reference_trace_parser.add_argument("--trace-out", type=Path, default=None)
    neutral_reference_trace_parser.add_argument("--species", default="h")
    neutral_reference_trace_parser.add_argument(
        "--cvode-max-order",
        type=int,
        default=None,
        help="Optional solver:cvode_max_order for constrained reference campaigns, for example 2 to match native BDF2 replay.",
    )
    neutral_reference_trace_parser.add_argument(
        "--timeout-seconds", type=float, default=120.0
    )
    neutral_reference_trace_parser.set_defaults(
        command=_trace_neutral_mixed_reference_accepted_steps_command
    )

    neutral_trace_compare_parser = subparsers.add_parser(
        "compare-neutral-mixed-accepted-traces",
        help="Compare native and reference accepted-internal-step traces for neutral-mixed NVh parity diagnostics.",
    )
    neutral_trace_compare_parser.add_argument("native_trace_json", type=Path)
    neutral_trace_compare_parser.add_argument("reference_trace_json", type=Path)
    neutral_trace_compare_parser.add_argument(
        "--reference-stage", default="post_accepted"
    )
    neutral_trace_compare_parser.add_argument(
        "--time-tolerance", type=float, default=1.0e-8
    )
    neutral_trace_compare_parser.add_argument(
        "--reference-cvode-max-order",
        type=int,
        default=None,
        help="Configured solver:cvode_max_order used for the reference JSONL, recorded in the parity JSON.",
    )
    neutral_trace_compare_parser.add_argument("--json-out", type=Path, required=True)
    neutral_trace_compare_parser.set_defaults(
        command=_compare_neutral_mixed_accepted_traces_command
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run a supported native input, write result artifacts, and optionally continue from a restart bundle.",
    )
    run_parser.add_argument("input_file", type=Path)
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only inspect configuration and exit successfully.",
    )
    run_parser.add_argument(
        "--precision",
        choices=("float32", "float64"),
        default=None,
        help="Override runtime floating-point precision for this run.",
    )
    run_parser.add_argument(
        "--case-name",
        type=str,
        default=None,
        help="Optional case label for output metadata.",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write standard run artifacts into this directory.",
    )
    run_parser.add_argument(
        "--json-out", type=Path, default=None, help="Write the portable summary JSON."
    )
    run_parser.add_argument(
        "--arrays-out", type=Path, default=None, help="Write the portable array NPZ."
    )
    run_parser.add_argument(
        "--restart-out", type=Path, default=None, help="Write the restart NPZ bundle."
    )
    run_parser.add_argument(
        "--log-out", type=Path, default=None, help="Write a verbose run log JSON."
    )
    run_parser.add_argument(
        "--restart-in",
        type=Path,
        default=None,
        help="Resume from a previously written restart NPZ bundle.",
    )
    run_parser.add_argument(
        "--resume-steps",
        type=int,
        default=None,
        help="Additional output intervals to run after loading --restart-in.",
    )
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit detailed staged terminal output for this run.",
    )
    run_parser.add_argument(
        "--quiet", action="store_true", help="Suppress the pretty terminal run summary."
    )
    run_parser.set_defaults(command=_run_command)

    parser.set_defaults(command=_default_command)
    return parser


def _default_command(args: argparse.Namespace) -> int:
    if getattr(args, "subcommand", None) is None:
        raise SystemExit(
            "Use `jax_drb inspect <input>` or `jax_drb <input> --dry-run`."
        )
    return args.command(args)


def _normalize_cli_argv(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    known_subcommands = {
        "inspect",
        "reference-cases",
        "run-reference-case",
        "compare-summary",
        "compare-arrays",
        "compare-recycling",
        "run-case",
        "validate-reference-baselines",
        "analyze-drift-wave",
        "analyze-alfven-wave",
        "compare-alfven-wave",
        "compare-drift-wave",
        "compare-blob2d",
        "analyze-neutral-mixed",
        "compare-neutral-mixed",
        "diagnose-neutral-mixed-substeps",
        "trace-neutral-mixed-accepted-steps",
        "trace-neutral-mixed-reference-accepted-steps",
        "compare-neutral-mixed-accepted-traces",
        "run",
    }
    head = argv[0]
    if head in known_subcommands or head.startswith("-"):
        return argv
    return ["run", *argv]


def _inspect_command(args: argparse.Namespace) -> int:
    config = load_bout_input(args.input_file)
    configure_jax_runtime(precision=resolve_runtime_precision(config=config))
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
    print(
        f"scheduled components: {', '.join(request.label for request in run_config.components)}"
    )

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
    if args.reference_root is None:
        print(
            "reference-cases: set --reference-root or JAX_DRB_REFERENCE_ROOT to a local reference checkout."
        )
        return 1

    resolved_cases = resolve_reference_cases(args.reference_root)
    for resolved in resolved_cases:
        status = "missing" if not resolved.exists else resolved.case.parity_mode
        print(
            f"{resolved.case.name}: {status} [{resolved.case.capability_tier}] -> {resolved.input_path}"
        )
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
    config = load_bout_input(args.input_file)
    run_config = RunConfiguration.from_config(config)
    resolved_precision = resolve_runtime_precision(
        requested=args.precision, config=config
    )
    cache_dir = configure_jax_runtime(precision=resolved_precision)
    import jax
    from .native import run_input_case
    from .native.runner import NativeRestartState, build_restart_state
    from .parity.arrays import (
        build_portable_array_payload,
        write_portable_array_payload,
    )
    from .parity.portable import write_portable_summary_payload
    from .runtime import (
        build_run_log_payload,
        load_restart_bundle,
        print_run_log,
        write_restart_bundle,
        write_run_log_payload,
    )
    from .runtime.output import build_run_event, print_run_event

    command_started_at = time.perf_counter()
    output_dir = args.output_dir or _config_path(config, "output", "directory")
    case_name = (
        args.case_name
        or _config_string(config, "output", "case_name")
        or args.input_file.stem
    )
    restart_in = args.restart_in or _config_path(config, "restart", "input")
    resume_steps = (
        args.resume_steps
        if args.resume_steps is not None
        else _config_int(config, "restart", "resume_steps")
    )
    logging_quiet = _config_bool(config, "runtime:logging", "quiet", default=False)
    logging_verbose = _config_optional_bool(config, "runtime:logging", "verbose")
    logging_verbosity = _config_string(config, "runtime:logging", "verbosity")
    if logging_verbosity is None:
        logging_verbosity = "detailed" if logging_verbose else "summary"
    if args.verbose:
        logging_verbosity = "detailed"
    emit_terminal_log = not args.quiet and not logging_quiet
    write_summary = _config_bool(config, "output", "write_summary", default=True)
    write_arrays = _config_bool(config, "output", "write_arrays", default=True)
    write_restart = _config_bool(config, "output", "write_restart", default=True)
    write_log = _config_bool(config, "output", "write_log", default=True)
    if args.json_out is None:
        args.json_out = _config_path(config, "output", "summary_json")
    if args.arrays_out is None:
        args.arrays_out = _config_path(config, "output", "arrays_npz")
    if args.restart_out is None:
        args.restart_out = _config_path(config, "output", "restart_npz")
    if args.log_out is None:
        args.log_out = _config_path(config, "output", "run_log_json")
    events: list[dict[str, Any]] = []

    def record_event(stage: str, message: str, **details: Any) -> None:
        event = build_run_event(
            stage=stage,
            message=message,
            elapsed_seconds=time.perf_counter() - command_started_at,
            details=details or None,
        )
        events.append(event)
        if emit_terminal_log:
            print_run_event(event, verbosity=logging_verbosity)

    record_event(
        "configuration",
        "Loaded input configuration",
        input_file=args.input_file,
        case_name=case_name,
        capability_tier="native_exact",
        precision=resolved_precision,
        nout=run_config.time.nout,
        timestep=run_config.time.timestep,
        output_dir=output_dir if output_dir is not None else "(none)",
        verbosity=logging_verbosity,
        verbose=logging_verbosity == "detailed",
    )
    restart_state = None
    bundle = None
    if restart_in is not None:
        bundle = load_restart_bundle(restart_in)
        restart_state = NativeRestartState(
            time_offset=bundle.current_time,
            completed_steps=bundle.completed_steps,
            configured_timestep=bundle.configured_timestep,
            variables=bundle.state_variables,
        )
        record_event(
            "restart",
            "Loaded restart bundle",
            restart_in=restart_in,
            current_time=bundle.current_time,
            completed_steps=bundle.completed_steps,
            variables=",".join(sorted(bundle.state_variables)),
            requested_resume_steps=resume_steps
            if resume_steps is not None
            else "(default)",
        )

    def relay_native_event(event: Mapping[str, Any]) -> None:
        if str(event.get("stage", "")) != "progress":
            return
        details = event.get("details")
        if isinstance(details, Mapping):
            record_event(
                str(event.get("stage", "progress")),
                str(event.get("message", "Native progress update")),
                **dict(details),
            )
        else:
            record_event(
                str(event.get("stage", "progress")),
                str(event.get("message", "Native progress update")),
            )

    started_at = time.perf_counter()
    record_event(
        "run", "Launching native run", mode="run", restart=restart_state is not None
    )
    result = run_input_case(
        args.input_file,
        case_name=case_name,
        parity_mode="run",
        restart_state=restart_state,
        output_steps=resume_steps,
        verbose=False,
        event_logger=relay_native_event,
    )
    elapsed_seconds = time.perf_counter() - started_at
    record_event(
        "run",
        "Native run completed",
        elapsed_seconds=f"{elapsed_seconds:.3f}",
        stored_states=len(result.time_points),
        compare_variables=",".join(result.variables),
    )

    output_paths: dict[str, str] = {}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.json_out is None and write_summary:
            args.json_out = output_dir / f"{case_name}_summary.json"
        if args.arrays_out is None and write_arrays:
            args.arrays_out = output_dir / f"{case_name}_arrays.npz"
        if args.restart_out is None and write_restart:
            args.restart_out = output_dir / f"{case_name}_restart.npz"
        if args.log_out is None and write_log:
            args.log_out = output_dir / f"{case_name}_run_log.json"
        record_event(
            "artifacts",
            "Resolved artifact destinations",
            summary_json=args.json_out if args.json_out is not None else "(disabled)",
            arrays_npz=args.arrays_out if args.arrays_out is not None else "(disabled)",
            restart_npz=args.restart_out
            if args.restart_out is not None
            else "(disabled)",
            run_log_json=args.log_out if args.log_out is not None else "(disabled)",
        )

    if args.json_out is not None:
        path = write_portable_summary_payload(result.payload, args.json_out)
        output_paths["summary_json"] = _sanitize_logged_path(path) or str(path)
        record_event("artifacts", "Wrote summary JSON", path=path)
    if args.arrays_out is not None:
        array_payload = build_portable_array_payload(
            case_name=str(result.payload["case_name"]),
            parity_mode=str(result.payload["parity_mode"]),
            capability_tier=str(result.payload.get("capability_tier", "native_exact")),
            compare_variables=tuple(str(name) for name in result.variables),
            component_labels=tuple(result.payload.get("component_labels", [])),
            dimensions=result.payload.get("dimensions", {}),
            time_points=tuple(float(value) for value in result.time_points),
            dataset_scalars=result.payload.get("dataset_scalars", {}),
            variables=result.variables,
            overrides=tuple(result.payload.get("overrides", [])),
            configured_nout=result.payload.get("configured_nout"),
            configured_timestep=result.payload.get("configured_timestep"),
            producer=str(result.payload.get("producer", "jax-drb")),
        )
        path = write_portable_array_payload(array_payload, args.arrays_out)
        output_paths["arrays_npz"] = _sanitize_logged_path(path) or str(path)
        record_event(
            "artifacts", "Wrote arrays NPZ", path=path, variables=len(result.variables)
        )

    restart_bundle = build_restart_state(result, parity_mode="run")
    if args.restart_out is not None and restart_bundle is not None:
        path = write_restart_bundle(restart_bundle, args.restart_out)
        output_paths["restart_npz"] = _sanitize_logged_path(path) or str(path)
        record_event(
            "artifacts",
            "Wrote restart bundle",
            path=path,
            completed_steps=restart_bundle.completed_steps,
            current_time=restart_bundle.current_time,
        )
    elif args.restart_out is not None and restart_bundle is None:
        output_paths["restart_npz"] = "(unsupported for this component set)"
        record_event(
            "artifacts",
            "Restart bundle unsupported for this run",
            path=args.restart_out,
        )

    if args.log_out is not None:
        output_paths["run_log_json"] = _sanitize_logged_path(args.log_out) or str(
            args.log_out
        )
    if output_paths:
        record_event("artifacts", "Planned run artifacts", **output_paths)

    log_payload = build_run_log_payload(
        input_file=_sanitize_logged_path(args.input_file) or args.input_file,
        case_name=case_name,
        parity_mode="run",
        capability_tier=str(result.payload.get("capability_tier", "native_exact")),
        component_labels=tuple(result.payload.get("component_labels", [])),
        time_points=tuple(float(value) for value in result.time_points),
        dimensions=result.payload.get("dimensions", {}),
        compare_variables=tuple(result.payload.get("compare_variables", [])),
        restart_supported=restart_bundle is not None,
        outputs=output_paths,
        variable_summaries=result.payload.get("variable_summaries", {}),
        run_configuration=_serialize_run_configuration(
            run_config,
            precision=resolved_precision,
            backend=jax.default_backend(),
            device=str(jax.devices()[0]) if jax.devices() else None,
            jax_version=getattr(jax, "__version__", None),
            cache_dir=cache_dir,
            elapsed_seconds=elapsed_seconds,
            output_directory=output_dir,
            logging_verbosity=logging_verbosity,
            logging_quiet=emit_terminal_log is False,
            restart_in=restart_in,
            resume_steps=resume_steps,
            working_directory=Path.cwd(),
        ),
        restart_info=_serialize_restart_info(
            restart_in=restart_in,
            loaded_bundle=bundle if restart_in is not None else None,
            requested_additional_steps=resume_steps,
            saved_bundle=restart_bundle,
        ),
        events=tuple(events),
    )
    if args.log_out is not None:
        record_event(
            "artifacts",
            "Writing verbose run log JSON",
            path=args.log_out,
            event_count=len(events),
        )
        log_payload["events"] = list(events)
        log_payload["event_count"] = len(events)
        log_payload["event_stages"] = [str(event.get("stage", "")) for event in events]
        path = write_run_log_payload(log_payload, args.log_out)
        output_paths["run_log_json"] = _sanitize_logged_path(path) or str(path)
        log_payload["outputs"] = output_paths
        log_payload["events"] = list(events)
        log_payload["event_count"] = len(events)
        log_payload["event_stages"] = [str(event.get("stage", "")) for event in events]

    if emit_terminal_log:
        print_run_log(log_payload, verbosity=logging_verbosity)
    return 0


def _serialize_run_configuration(
    run_config: RunConfiguration,
    *,
    precision: str,
    backend: str | None = None,
    device: str | None = None,
    cache_dir: Path | None = None,
    elapsed_seconds: float | None = None,
    output_directory: Path | None = None,
    logging_verbosity: str | None = None,
    logging_quiet: bool | None = None,
    restart_in: Path | None = None,
    resume_steps: int | None = None,
    jax_version: str | None = None,
    working_directory: Path | None = None,
) -> dict[str, object]:
    return {
        "time": {
            "nout": run_config.time.nout,
            "timestep": run_config.time.timestep,
        },
        "mesh": {
            "nx": run_config.mesh.nx,
            "ny": run_config.mesh.ny,
            "nz": run_config.mesh.nz,
            "mxg": run_config.mesh.mxg,
            "myg": run_config.mesh.myg,
            "file": run_config.mesh.file,
            "parallel_transform": run_config.mesh.parallel_transform.type,
        },
        "solver": {
            "type": run_config.solver.type,
            "mxstep": run_config.solver.mxstep,
            "rtol": run_config.solver.rtol,
            "atol": run_config.solver.atol,
            "use_precon": run_config.solver.use_precon,
            "cvode_max_order": run_config.solver.cvode_max_order,
        },
        "runtime": {
            "precision": precision,
            "backend": backend,
            "device": device,
            "jax_version": jax_version,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "process_id": os.getpid(),
            "compilation_cache_dir": _sanitize_logged_path(cache_dir),
            "elapsed_seconds": elapsed_seconds,
            "logging": {
                "verbosity": logging_verbosity,
                "verbose": logging_verbosity == "detailed",
                "quiet": logging_quiet,
            },
        },
        "output": {
            "directory": _sanitize_logged_path(output_directory),
            "working_directory": _sanitize_logged_path(working_directory),
        },
        "restart_request": {
            "restart_in": _sanitize_logged_path(restart_in),
            "resume_steps": resume_steps,
        },
        "components": [request.label for request in run_config.components],
        "root_scalars": dict(run_config.root_scalars),
        "model_scalars": dict(run_config.model_scalars),
    }


def _serialize_restart_info(
    *,
    restart_in: Path | None,
    loaded_bundle,
    requested_additional_steps: int | None,
    saved_bundle,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    if restart_in is not None and loaded_bundle is not None:
        payload["loaded_from"] = _sanitize_logged_path(restart_in)
        payload["start_time"] = loaded_bundle.current_time
        payload["input_completed_steps"] = loaded_bundle.completed_steps
        payload["loaded_state_variables"] = sorted(loaded_bundle.state_variables)
    if requested_additional_steps is not None:
        payload["requested_additional_steps"] = requested_additional_steps
    if saved_bundle is not None:
        payload["saved_completed_steps"] = saved_bundle.completed_steps
        payload["saved_current_time"] = saved_bundle.current_time
        payload["saved_state_variables"] = sorted(saved_bundle.state_variables)
    return payload


def _default_reference_root() -> Path | None:
    return resolve_default_reference_root()


def _sanitize_logged_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    try:
        resolved = resolved.resolve()
    except FileNotFoundError:
        resolved = resolved.absolute()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        pass
    try:
        return f"~/{resolved.relative_to(Path.home()).as_posix()}"
    except ValueError:
        return resolved.as_posix()


def _config_value(config, section: str, key: str, default: Any = None) -> Any:
    if config.has_option(section, key):
        return config.parsed(section, key)
    return default


def _config_string(
    config, section: str, key: str, default: str | None = None
) -> str | None:
    value = _config_value(config, section, key, default)
    if value is None:
        return None
    return str(value)


def _config_int(
    config, section: str, key: str, default: int | None = None
) -> int | None:
    value = _config_value(config, section, key, default)
    if value is None:
        return None
    return int(value)


def _config_bool(config, section: str, key: str, default: bool = False) -> bool:
    value = _config_value(config, section, key, default)
    return bool(value)


def _config_optional_bool(config, section: str, key: str) -> bool | None:
    if not config.has_option(section, key):
        return None
    return bool(config.parsed(section, key))


def _config_path(config, section: str, key: str) -> Path | None:
    value = _config_value(config, section, key)
    if value in (None, ""):
        return None
    return Path(str(value))


def _default_reference_binary() -> Path | None:
    value = os.environ.get("JAX_DRB_REFERENCE_BINARY")
    return Path(value) if value else None


def _run_reference_case_command(args: argparse.Namespace) -> int:
    from .parity.arrays import build_dataset_array_payload, write_portable_array_payload
    from .parity.reference import (
        find_reference_case,
        run_reference_case,
        write_case_baseline_json,
    )

    if args.reference_root is None:
        print("run-reference-case: set --reference-root or JAX_DRB_REFERENCE_ROOT.")
        return 1

    case = find_reference_case(args.case_name)
    case_input = case.input_path(args.reference_root)
    case_run_config = RunConfiguration.from_config(load_bout_input(case_input))
    result = run_reference_case(
        args.case_name,
        reference_root=args.reference_root,
        reference_binary=args.reference_binary,
        workdir=args.workdir,
        extra_overrides=args.override,
    )
    summary = result.summary
    print(f"case: {summary.case_name}")
    print(f"parity_mode: {summary.parity_mode}")
    print(f"workdir: {summary.workdir}")
    print(
        f"overrides: {', '.join(summary.overrides) if summary.overrides else '(none)'}"
    )
    print(f"time_points: {summary.time_points}")
    print(
        f"compare_variables: {', '.join(summary.compare_variables) if summary.compare_variables else '(none)'}"
    )
    for name, variable in summary.variable_summaries.items():
        delta = (
            "n/a"
            if variable.max_abs_delta_last_first is None
            else f"{variable.max_abs_delta_last_first:.8e}"
        )
        print(
            f"  {name}: shape={variable.shape}, min={variable.minimum:.8e}, "
            f"max={variable.maximum:.8e}, mean={variable.mean:.8e}, delta={delta}"
        )
    if args.json_out is not None:
        path = write_case_baseline_json(summary, args.json_out)
        print(f"json_out: {path}")
    if args.arrays_out is not None:
        array_payload = build_dataset_array_payload(
            summary.artifacts["BOUT.dmp.0.nc"],
            case_name=summary.case_name,
            parity_mode=summary.parity_mode,
            capability_tier=summary.capability_tier,
            compare_variables=summary.compare_variables,
            component_labels=summary.component_labels,
            overrides=summary.overrides,
            trim_x_guards=case.trim_x_guards,
            x_guards=case_run_config.mesh.mxg,
            trim_y_guards=case.trim_y_guards,
            y_guards=case_run_config.mesh.myg,
            configured_nout=summary.nout,
            configured_timestep=summary.timestep,
        )
        path = write_portable_array_payload(array_payload, args.arrays_out)
        print(f"arrays_out: {path}")
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


def _compare_arrays_command(args: argparse.Namespace) -> int:
    from .parity.arrays import compare_array_payloads, load_portable_array_payload

    expected = load_portable_array_payload(args.expected_npz)
    actual = load_portable_array_payload(args.actual_npz)
    result = compare_array_payloads(
        expected,
        actual,
        scalar_rtol=args.scalar_rtol,
        scalar_atol=args.scalar_atol,
        array_rtol=args.array_rtol,
        array_atol=args.array_atol,
    )
    if result.ok:
        print("comparison: ok")
        return 0
    print("comparison: mismatch")
    for issue in result.issues:
        print(f"  {issue.field}: {issue.message}")
    return 1


def _compare_recycling_command(args: argparse.Namespace) -> int:
    from .parity.diff import compare_recycling_artifacts, format_recycling_diff_report

    result = compare_recycling_artifacts(
        args.expected_artifact,
        args.actual_artifact,
        artifact_kind=args.artifact_kind,
        scalar_rtol=args.scalar_rtol,
        scalar_atol=args.scalar_atol,
        array_rtol=args.array_rtol,
        array_atol=args.array_atol,
    )
    print(format_recycling_diff_report(result))
    return 0 if result.ok else 1


def _run_case_command(args: argparse.Namespace) -> int:
    configure_jax_runtime()
    from .native import run_curated_case
    from .parity.arrays import (
        build_array_payload_from_summary_payload,
        write_portable_array_payload,
    )
    from .parity.portable import write_portable_summary_payload

    if args.reference_root is None:
        print("run-case: set --reference-root or JAX_DRB_REFERENCE_ROOT.")
        return 1

    result = run_curated_case(
        args.case_name,
        reference_root=args.reference_root,
        extra_overrides=tuple(getattr(args, "override", ()) or ()),
    )
    payload = result.payload
    print(f"case: {payload['case_name']}")
    print(f"parity_mode: {payload['parity_mode']}")
    print(f"producer: {payload['producer']}")
    print(
        f"compare_variables: {', '.join(payload['compare_variables']) if payload['compare_variables'] else '(none)'}"
    )
    for name, variable in payload["variable_summaries"].items():
        delta = variable["max_abs_delta_last_first"]
        delta_text = "n/a" if delta is None else f"{delta:.8e}"
        print(
            f"  {name}: shape={tuple(variable['shape'])}, min={variable['minimum']:.8e}, "
            f"max={variable['maximum']:.8e}, mean={variable['mean']:.8e}, delta={delta_text}"
        )
    if args.json_out is not None:
        path = write_portable_summary_payload(payload, args.json_out)
        print(f"json_out: {path}")
    if args.arrays_out is not None:
        array_payload = build_array_payload_from_summary_payload(
            payload, result.variables
        )
        path = write_portable_array_payload(array_payload, args.arrays_out)
        print(f"arrays_out: {path}")
    return 0


def _parse_substep_csv(value: str) -> tuple[int, ...]:
    substeps = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not substeps:
        raise ValueError("At least one neutral-mixed substep count is required.")
    if any(item <= 0 for item in substeps):
        raise ValueError("Neutral-mixed substep counts must be positive integers.")
    return substeps


def _diagnose_neutral_mixed_substeps_command(args: argparse.Namespace) -> int:
    from .validation import (
        build_neutral_mixed_substep_hybrid_report,
        write_neutral_mixed_substep_hybrid_json,
    )

    try:
        substeps = _parse_substep_csv(args.substeps)
    except ValueError as exc:
        print(f"diagnose-neutral-mixed-substeps: {exc}")
        return 1
    report = build_neutral_mixed_substep_hybrid_report(
        reference_root=args.reference_root,
        case_name=args.case_name,
        input_path=args.input_path,
        reference_arrays_npz=args.reference_arrays_npz,
        substeps=substeps,
    )
    print(f"diagnostic: {report['diagnostic']}")
    print(f"case: {report['case_name']}")
    print(f"field: {report['field']}")
    for point in report["sweep_points"]:
        status = str(point["status"])
        text = f"  substeps={point['internal_substeps']}: {status}"
        if status == "ok":
            fields = point["final_field_error_register"]["fields"]
            text += f", NVh_final_max_abs={fields['NVh']['max_abs']:.8e}"
        else:
            text += f", {point.get('error_type')}: {point.get('error_message')}"
        print(text)
    best = report.get("best")
    if isinstance(best, dict):
        print(
            f"best: substeps={best['internal_substeps']}, {best['metric']}={best['value']:.8e}"
        )
    if args.json_out is not None:
        path = write_neutral_mixed_substep_hybrid_json(report, args.json_out)
        print(f"json_out: {path}")
    return (
        0 if any(point.get("status") == "ok" for point in report["sweep_points"]) else 1
    )


def _trace_neutral_mixed_accepted_steps_command(args: argparse.Namespace) -> int:
    from .validation import (
        build_neutral_mixed_native_accepted_step_trace_report,
        write_neutral_mixed_native_accepted_step_trace_json,
    )

    if int(args.internal_substeps) <= 0:
        print(
            "trace-neutral-mixed-accepted-steps: --internal-substeps must be positive."
        )
        return 1
    if int(args.steps) <= 0:
        print("trace-neutral-mixed-accepted-steps: --steps must be positive.")
        return 1
    if float(args.time_tolerance) <= 0.0:
        print("trace-neutral-mixed-accepted-steps: --time-tolerance must be positive.")
        return 1
    report = build_neutral_mixed_native_accepted_step_trace_report(
        reference_root=args.reference_root,
        case_name=args.case_name,
        input_path=args.input_path,
        internal_substeps=int(args.internal_substeps),
        steps=int(args.steps),
        reference_trace_json=args.reference_trace_json,
        reference_stage=args.reference_stage,
        time_tolerance=float(args.time_tolerance),
    )
    path = write_neutral_mixed_native_accepted_step_trace_json(report, args.json_out)
    print(f"diagnostic: {report['diagnostic']}")
    print(f"case: {report['case_name']}")
    print(f"trace_point_count: {report['trace_point_count']}")
    print(f"sample_y_indices: {report['sample_y_indices']}")
    print(f"json_out: {path}")
    return 0


def _trace_neutral_mixed_reference_accepted_steps_command(
    args: argparse.Namespace,
) -> int:
    from .validation import run_neutral_mixed_hermes_accepted_step_trace

    if float(args.timeout_seconds) <= 0.0:
        print(
            "trace-neutral-mixed-reference-accepted-steps: --timeout-seconds must be positive."
        )
        return 1
    cvode_max_order = getattr(args, "cvode_max_order", None)
    if cvode_max_order is not None and int(cvode_max_order) <= 0:
        print(
            "trace-neutral-mixed-reference-accepted-steps: --cvode-max-order must be positive."
        )
        return 1
    path = run_neutral_mixed_hermes_accepted_step_trace(
        reference_root=args.reference_root,
        workdir=args.workdir,
        hermes_binary=args.hermes_binary,
        trace_jsonl_path=args.trace_out,
        timeout_seconds=float(args.timeout_seconds),
        species=args.species,
        cvode_max_order=cvode_max_order,
    )
    print("diagnostic: neutral_mixed_reference_accepted_step_trace")
    if cvode_max_order is not None:
        print(f"cvode_max_order: {int(cvode_max_order)}")
    print(f"trace_jsonl: {path}")
    return 0


def _compare_neutral_mixed_accepted_traces_command(args: argparse.Namespace) -> int:
    from .validation import (
        build_neutral_mixed_accepted_step_trace_parity_report,
        write_neutral_mixed_accepted_step_trace_parity_json,
    )

    if float(args.time_tolerance) <= 0.0:
        print(
            "compare-neutral-mixed-accepted-traces: --time-tolerance must be positive."
        )
        return 1
    reference_cvode_max_order = getattr(args, "reference_cvode_max_order", None)
    if reference_cvode_max_order is not None and int(reference_cvode_max_order) <= 0:
        print(
            "compare-neutral-mixed-accepted-traces: --reference-cvode-max-order must be positive."
        )
        return 1
    report = build_neutral_mixed_accepted_step_trace_parity_report(
        native_trace_json=args.native_trace_json,
        reference_trace_json=args.reference_trace_json,
        reference_stage=args.reference_stage,
        time_tolerance=float(args.time_tolerance),
        reference_cvode_max_order=reference_cvode_max_order,
    )
    path = write_neutral_mixed_accepted_step_trace_parity_json(report, args.json_out)
    print(f"diagnostic: {report['diagnostic']}")
    print(f"matched_trace_point_count: {report['matched_trace_point_count']}")
    ranked = report.get("ranked_fields", [])
    if ranked:
        worst = ranked[0]
        print(
            "worst_field: "
            f"{worst['field']}, target_delta={worst['max_target_adjacent_delta']:.8e}, "
            f"guard_delta={worst['max_guard_delta']:.8e}"
        )
    print(f"json_out: {path}")
    return 0


def _validate_reference_baselines_command(args: argparse.Namespace) -> int:
    from .parity.reference import validate_reference_baselines

    if args.reference_root is None:
        print(
            "validate-reference-baselines: set --reference-root or JAX_DRB_REFERENCE_ROOT."
        )
        return 1

    results = validate_reference_baselines(
        reference_root=args.reference_root,
        reference_binary=args.reference_binary,
        case_names=args.case or None,
        baseline_dir=args.baseline_dir,
    )
    ok = True
    for result in results:
        status = "ok" if result.ok else "mismatch"
        print(f"{result.case_name}: {status}")
        for issue in result.issues:
            print(f"  {issue}")
        ok = ok and result.ok
    return 0 if ok else 1


def _analyze_drift_wave_command(args: argparse.Namespace) -> int:
    from .validation import (
        analyze_drift_wave_npz,
        save_drift_wave_diagnostic_plot,
        write_drift_wave_analysis_json,
    )

    result = analyze_drift_wave_npz(
        args.arrays_npz,
        input_file=args.input_file,
        density_variable=args.density_variable,
        x_index=args.x_index,
        y_index=args.y_index,
        fit_points=args.fit_points,
    )
    benchmark = result.benchmark
    print(f"density_variable: {result.density_variable}")
    print(f"trace_index: x={result.trace_x_index}, y={result.trace_y_index}")
    print(f"fit_points: {result.fit_points}")
    print(f"wstar: {benchmark.wstar:.8e}")
    print(f"sigmapar: {benchmark.sigmapar:.8e}")
    print(f"sigmapar_over_wstar: {benchmark.sigmapar_over_wstar:.8e}")
    print(f"analytic_gamma_over_wstar: {benchmark.analytic_gamma_over_wstar:.8e}")
    print(f"analytic_omega_over_wstar: {benchmark.analytic_omega_over_wstar:.8e}")
    print(f"measured_gamma_over_wstar: {result.measured_gamma_over_wstar:.8e}")
    print(f"measured_omega_over_wstar: {result.measured_omega_over_wstar:.8e}")
    if args.json_out is not None:
        path = write_drift_wave_analysis_json(result, args.json_out)
        print(f"json_out: {path}")
    if args.plot_out is not None:
        path = save_drift_wave_diagnostic_plot(result, args.plot_out)
        print(f"plot_out: {path}")
    return 0


def _analyze_alfven_wave_command(args: argparse.Namespace) -> int:
    from .validation import (
        analyze_alfven_wave_npz,
        save_alfven_wave_diagnostic_plot,
        write_alfven_wave_analysis_json,
    )

    result = analyze_alfven_wave_npz(
        args.arrays_npz,
        input_file=args.input_file,
        field_variable=args.field_variable,
        x_index=args.x_index,
    )
    benchmark = result.benchmark
    print(f"field_variable: {result.field_variable}")
    print(f"x_index: {result.x_index}")
    print(f"kpar: {benchmark.kpar:.8e}")
    print(f"kperp: {benchmark.kperp:.8e}")
    print(f"analytic_phase_speed: {benchmark.analytic_phase_speed:.8e}")
    print(f"analytic_omega: {benchmark.analytic_omega:.8e}")
    print(f"measured_phase_speed: {result.measured_phase_speed:.8e}")
    print(f"measured_omega: {result.measured_omega:.8e}")
    print(f"relative_phase_speed_error: {result.relative_phase_speed_error:.8e}")
    if args.json_out is not None:
        path = write_alfven_wave_analysis_json(result, args.json_out)
        print(f"json_out: {path}")
    if args.plot_out is not None:
        path = save_alfven_wave_diagnostic_plot(result, args.plot_out)
        print(f"plot_out: {path}")
    return 0


def _compare_alfven_wave_command(args: argparse.Namespace) -> int:
    from .validation import (
        compare_alfven_wave_npz,
        save_alfven_wave_parity_plot,
        write_alfven_wave_parity_json,
    )

    result = compare_alfven_wave_npz(
        args.expected_npz,
        args.actual_npz,
        input_file=args.input_file,
        field_variable=args.field_variable,
        x_index=args.x_index,
    )
    print(f"field_variable: {result.expected.field_variable}")
    print(f"x_index: {result.expected.x_index}")
    print(f"expected_phase_speed: {result.expected.measured_phase_speed:.8e}")
    print(f"actual_phase_speed: {result.actual.measured_phase_speed:.8e}")
    print(f"expected_omega: {result.expected.measured_omega:.8e}")
    print(f"actual_omega: {result.actual.measured_omega:.8e}")
    print(f"phase_speed_error: {result.phase_speed_error:.8e}")
    print(f"omega_error: {result.omega_error:.8e}")
    print(f"mean_square_max_abs_error: {result.mean_square_max_abs_error:.8e}")
    print(f"mean_square_rms_error: {result.mean_square_rms_error:.8e}")
    if args.json_out is not None:
        path = write_alfven_wave_parity_json(result, args.json_out)
        print(f"json_out: {path}")
    if args.plot_out is not None:
        path = save_alfven_wave_parity_plot(result, args.plot_out)
        print(f"plot_out: {path}")
    return 0


def _compare_drift_wave_command(args: argparse.Namespace) -> int:
    from .validation import (
        compare_drift_wave_npz,
        save_drift_wave_parity_plot,
        write_drift_wave_parity_json,
    )

    result = compare_drift_wave_npz(
        args.expected_npz,
        args.actual_npz,
        input_file=args.input_file,
        density_variable=args.density_variable,
        x_index=args.x_index,
        y_index=args.y_index,
        fit_points=args.fit_points,
    )
    print(f"density_variable: {result.expected.density_variable}")
    print(
        f"trace_index: x={result.expected.trace_x_index}, y={result.expected.trace_y_index}"
    )
    print(f"fit_points: {result.expected.fit_points}")
    print(f"expected_gamma_over_wstar: {result.expected.measured_gamma_over_wstar:.8e}")
    print(f"actual_gamma_over_wstar: {result.actual.measured_gamma_over_wstar:.8e}")
    print(f"expected_omega_over_wstar: {result.expected.measured_omega_over_wstar:.8e}")
    print(f"actual_omega_over_wstar: {result.actual.measured_omega_over_wstar:.8e}")
    for name, variable_error in sorted(result.variable_errors.items()):
        print(
            f"{name}: max_abs_error={variable_error.max_abs_error:.8e}, "
            f"rms_error={variable_error.rms_error:.8e}"
        )
    if args.json_out is not None:
        path = write_drift_wave_parity_json(result, args.json_out)
        print(f"json_out: {path}")
    if args.plot_out is not None:
        path = save_drift_wave_parity_plot(result, args.plot_out)
        print(f"plot_out: {path}")
    return 0


def _compare_blob2d_command(args: argparse.Namespace) -> int:
    from .validation import (
        compare_blob2d_artifacts,
        save_blob2d_parity_plot,
        write_blob2d_parity_json,
    )

    result = compare_blob2d_artifacts(
        args.expected_artifact,
        args.actual_artifact,
        density_variable=args.density_variable,
        background_density=args.background_density,
    )
    print(f"density_variable: {result.expected.density_variable}")
    print(f"background_density: {result.expected.background_density:.8e}")
    print(f"peak_max_abs_error: {result.peak_max_abs_error:.8e}")
    print(f"peak_rms_error: {result.peak_rms_error:.8e}")
    print(
        f"center_of_mass_x_max_abs_error: {result.center_of_mass_x_max_abs_error:.8e}"
    )
    print(
        f"center_of_mass_z_max_abs_error: {result.center_of_mass_z_max_abs_error:.8e}"
    )
    if args.json_out is not None:
        path = write_blob2d_parity_json(result, args.json_out)
        print(f"json_out: {path}")
    if args.plot_out is not None:
        path = save_blob2d_parity_plot(result, args.plot_out)
        print(f"plot_out: {path}")
    return 0


def _analyze_neutral_mixed_command(args: argparse.Namespace) -> int:
    from .validation import (
        analyze_neutral_mixed_npz,
        save_neutral_mixed_diagnostic_plot,
        write_neutral_mixed_analysis_json,
    )

    result = analyze_neutral_mixed_npz(
        args.arrays_npz,
        density_variable=args.density_variable,
        pressure_variable=args.pressure_variable,
        momentum_variable=args.momentum_variable,
        x_index=args.x_index,
        y_index=args.y_index,
        z_index=args.z_index,
    )
    print(f"density_variable: {result.density_variable}")
    print(f"pressure_variable: {result.pressure_variable}")
    print(f"momentum_variable: {result.momentum_variable}")
    print(
        f"center_index: x={result.center_index_x}, y={result.center_index_y}, z={result.center_index_z}"
    )
    print(f"center_density_final: {result.center_density_history[-1]:.8e}")
    print(f"center_pressure_final: {result.center_pressure_history[-1]:.8e}")
    print(f"center_momentum_final: {result.center_momentum_history[-1]:.8e}")
    print(f"center_temperature_final: {result.center_temperature_history[-1]:.8e}")
    print(f"total_density_final: {result.total_density_history[-1]:.8e}")
    print(f"total_pressure_final: {result.total_pressure_history[-1]:.8e}")
    print(f"momentum_rms_final: {result.momentum_rms_history[-1]:.8e}")
    if args.json_out is not None:
        path = write_neutral_mixed_analysis_json(result, args.json_out)
        print(f"json_out: {path}")
    if args.plot_out is not None:
        path = save_neutral_mixed_diagnostic_plot(result, args.plot_out)
        print(f"plot_out: {path}")
    return 0


def _compare_neutral_mixed_command(args: argparse.Namespace) -> int:
    from .validation import (
        compare_neutral_mixed_artifacts,
        save_neutral_mixed_parity_plot,
        write_neutral_mixed_parity_json,
    )

    result = compare_neutral_mixed_artifacts(
        args.expected_artifact,
        args.actual_artifact,
        density_variable=args.density_variable,
        pressure_variable=args.pressure_variable,
        momentum_variable=args.momentum_variable,
        x_index=args.x_index,
        y_index=args.y_index,
        z_index=args.z_index,
    )
    print(
        f"center_index: x={result.expected.center_index_x}, y={result.expected.center_index_y}, z={result.expected.center_index_z}"
    )
    for name, series_error in sorted(result.series_errors.items()):
        print(
            f"{name}: max_abs_error={series_error.max_abs_error:.8e}, "
            f"rms_error={series_error.rms_error:.8e}"
        )
    if args.json_out is not None:
        path = write_neutral_mixed_parity_json(result, args.json_out)
        print(f"json_out: {path}")
    if args.plot_out is not None:
        path = save_neutral_mixed_parity_plot(result, args.plot_out)
        print(f"plot_out: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
