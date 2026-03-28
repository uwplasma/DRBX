from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config.boutinp import load_bout_input
from .reference.cases import resolve_reference_cases
from .runtime import configure_jax_runtime
from .runtime.run_config import RunConfiguration


def main(argv: list[str] | None = None) -> int:
    configure_jax_runtime()
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.command(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jax-drb",
        description="Inspect or run JAX-DRB inputs using the native model configuration structure.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=False)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a BOUT.inp file and print the resolved plan.")
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
        help="Path to the local reference checkout used for case inspection.",
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
        help="Path to the local reference checkout used for case lookup and default binary discovery.",
    )
    run_case_parser.add_argument("--reference-binary", type=Path, default=_default_reference_binary())
    run_case_parser.add_argument("--workdir", type=Path, default=None)
    run_case_parser.add_argument("--override", action="append", default=[], help="Additional source-style overrides such as nout=0.")
    run_case_parser.add_argument("--json-out", type=Path, default=None, help="Write the run summary to a JSON file.")
    run_case_parser.add_argument("--arrays-out", type=Path, default=None, help="Write the full comparison arrays to a compressed NPZ.")
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
        help="Path to the local private reference checkout used to locate the curated input file.",
    )
    run_case_parser.add_argument("--json-out", type=Path, default=None, help="Write the portable summary to JSON.")
    run_case_parser.add_argument("--arrays-out", type=Path, default=None, help="Write the full comparison arrays to a compressed NPZ.")
    run_case_parser.set_defaults(command=_run_case_command)

    validate_reference_parser = subparsers.add_parser(
        "validate-reference-baselines",
        help="Re-run committed reference cases and compare the live outputs to the stored summary baselines.",
    )
    validate_reference_parser.add_argument(
        "--reference-root",
        type=Path,
        default=_default_reference_root(),
        help="Path to the local private reference checkout used for case lookup and binary discovery.",
    )
    validate_reference_parser.add_argument("--reference-binary", type=Path, default=_default_reference_binary())
    validate_reference_parser.add_argument("--case", action="append", default=[], help="Specific case name to validate. Repeat to validate multiple cases.")
    validate_reference_parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "references" / "baselines" / "reference",
        help="Directory containing committed reference summary baselines.",
    )
    validate_reference_parser.set_defaults(command=_validate_reference_baselines_command)

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
    if args.reference_root is None:
        print("reference-cases: set --reference-root or JAX_DRB_REFERENCE_ROOT to a local reference checkout.")
        return 1

    resolved_cases = resolve_reference_cases(args.reference_root)
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


def _default_reference_root() -> Path | None:
    value = os.environ.get("JAX_DRB_REFERENCE_ROOT")
    return Path(value) if value else None


def _default_reference_binary() -> Path | None:
    value = os.environ.get("JAX_DRB_REFERENCE_BINARY")
    return Path(value) if value else None


def _run_reference_case_command(args: argparse.Namespace) -> int:
    from .parity.arrays import build_dataset_array_payload, write_portable_array_payload
    from .parity.reference import find_reference_case, run_reference_case, write_case_baseline_json

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
    if args.arrays_out is not None:
        array_payload = build_dataset_array_payload(
            summary.artifacts["BOUT.dmp.0.nc"],
            case_name=summary.case_name,
            parity_mode=summary.parity_mode,
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
    from .native import run_curated_case
    from .parity.arrays import build_array_payload_from_summary_payload, write_portable_array_payload
    from .parity.portable import write_portable_summary_payload

    if args.reference_root is None:
        print("run-case: set --reference-root or JAX_DRB_REFERENCE_ROOT.")
        return 1

    result = run_curated_case(args.case_name, reference_root=args.reference_root)
    payload = result.payload
    print(f"case: {payload['case_name']}")
    print(f"parity_mode: {payload['parity_mode']}")
    print(f"producer: {payload['producer']}")
    print(f"compare_variables: {', '.join(payload['compare_variables']) if payload['compare_variables'] else '(none)'}")
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
        array_payload = build_array_payload_from_summary_payload(payload, result.variables)
        path = write_portable_array_payload(array_payload, args.arrays_out)
        print(f"arrays_out: {path}")
    return 0


def _validate_reference_baselines_command(args: argparse.Namespace) -> int:
    from .parity.reference import validate_reference_baselines

    if args.reference_root is None:
        print("validate-reference-baselines: set --reference-root or JAX_DRB_REFERENCE_ROOT.")
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
    print(f"trace_index: x={result.expected.trace_x_index}, y={result.expected.trace_y_index}")
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
    print(f"center_of_mass_x_max_abs_error: {result.center_of_mass_x_max_abs_error:.8e}")
    print(f"center_of_mass_z_max_abs_error: {result.center_of_mass_z_max_abs_error:.8e}")
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
    print(f"center_index: x={result.center_index_x}, y={result.center_index_y}, z={result.center_index_z}")
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
    print(f"center_index: x={result.expected.center_index_x}, y={result.expected.center_index_y}, z={result.expected.center_index_z}")
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
