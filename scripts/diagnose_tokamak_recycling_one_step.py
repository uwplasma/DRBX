#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from jax_drb.native.runner import run_curated_case
from jax_drb.parity.arrays import (
    build_array_payload_from_summary_payload,
    build_dataset_array_payload,
    load_portable_array_payload,
)
from jax_drb.parity.compare import compare_summary_payloads, load_summary_json
from jax_drb.parity.diff import build_array_time_trace, build_scaled_array_diff_entries
from jax_drb.parity.reference import build_case_baseline_payload, resolve_reference_case, run_reference_case
from jax_drb.reference.paths import default_reference_root


def default_tokamak_recycling_cases() -> tuple[str, ...]:
    return (
        "tokamak_recycling_dthe_one_step",
        "tokamak_recycling_dthe_drifts_one_step",
        "tokamak_recycling_dthene_one_step",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose direct tokamak recycling one-step parity against committed baselines or a fresh Hermes run. "
            "Print missing compare variables, top residual fields, and time traces at the worst spatial cells."
        )
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=default_reference_root(),
        help="Path to the local Hermes-3 checkout.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help=(
            "Curated case to diagnose. Repeat to inspect multiple cases. "
            "Defaults to the direct tokamak recycling one-step set."
        ),
    )
    parser.add_argument(
        "--use-committed-baselines",
        action="store_true",
        help="Compare against committed summary/array baselines instead of rerunning Hermes.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of ranked fields to print per case.")
    parser.add_argument(
        "--trace-top",
        type=int,
        default=3,
        help="For the top N ranked fields, print the full time trace at the worst spatial location.",
    )
    parser.add_argument(
        "--near-zero-atol",
        type=float,
        default=1.0e-12,
        help="Expected max-abs threshold below which a field is treated as near-zero when reporting relative diffs.",
    )
    args = parser.parse_args()
    if args.reference_root is None:
        raise SystemExit("Set --reference-root or JAX_DRB_REFERENCE_ROOT before running live parity diagnostics.")

    case_names = tuple(args.case) if args.case else default_tokamak_recycling_cases()
    repo_root = Path(__file__).resolve().parents[1]
    summary_baseline_dir = repo_root / "references" / "baselines" / "reference"
    array_baseline_dir = repo_root / "references" / "baselines" / "reference_arrays"

    for case_name in case_names:
        case, _ = resolve_reference_case(case_name, reference_root=args.reference_root)
        native = run_curated_case(case_name, reference_root=args.reference_root)

        if args.use_committed_baselines:
            reference_summary = load_summary_json(summary_baseline_dir / f"{case_name}.json")
            reference_arrays = load_portable_array_payload(array_baseline_dir / f"{case_name}.npz")
        else:
            with tempfile.TemporaryDirectory(prefix=f"jaxdrb-{case_name}-") as workdir:
                reference = run_reference_case(
                    case_name,
                    reference_root=args.reference_root,
                    workdir=workdir,
                    keep_workdir=True,
                )
                reference_summary = build_case_baseline_payload(reference.summary)
                reference_arrays = build_dataset_array_payload(
                    reference.summary.artifacts["BOUT.dmp.0.nc"],
                    case_name=case_name,
                    parity_mode=reference.summary.parity_mode,
                    compare_variables=reference.summary.compare_variables,
                    component_labels=reference.summary.component_labels,
                    overrides=reference.summary.overrides,
                    trim_x_guards=case.trim_x_guards,
                    x_guards=2,
                    trim_y_guards=case.trim_y_guards,
                    y_guards=2,
                    configured_nout=reference.summary.nout,
                    configured_timestep=reference.summary.timestep,
                )

        print(f"CASE {case_name}")
        native_fields = tuple(native.variables)
        missing_compare_variables = tuple(
            field for field in reference_summary["compare_variables"] if field not in native_fields
        )
        if missing_compare_variables:
            print("  missing_compare_variables:", ", ".join(missing_compare_variables))

        summary = compare_summary_payloads(reference_summary, native.payload, scalar_rtol=1.0e-12, scalar_atol=1.0e-12)
        print(f"  summary_ok={summary.ok} summary_issues={len(summary.issues)}")
        for issue in summary.issues[: args.limit]:
            print(f"    summary {issue.field}: {issue.message}")

        native_arrays = build_array_payload_from_summary_payload(native.payload, native.variables)
        entries = build_scaled_array_diff_entries(
            reference_arrays["variables"],
            native_arrays["variables"],
            compare_variables=tuple(reference_summary["compare_variables"]),
            near_zero_atol=args.near_zero_atol,
        )
        ranked = sorted(entries, key=lambda entry: entry.max_abs_diff, reverse=True)
        for entry in ranked[: args.limit]:
            relative = "n/a" if entry.relative_to_expected_max is None else f"{entry.relative_to_expected_max:.8e}"
            print(
                "  field "
                f"{entry.field}: max_abs_diff={entry.max_abs_diff:.8e} "
                f"expected_abs_max={entry.expected_abs_max:.8e} "
                f"relative_to_expected_max={relative} "
                f"location={entry.max_abs_location} "
                f"expected={entry.expected_value:.8e} actual={entry.actual_value:.8e}"
            )
        for entry in ranked[: args.trace_top]:
            if len(entry.max_abs_location) <= 1:
                continue
            spatial_location = entry.max_abs_location[1:]
            trace = build_array_time_trace(
                reference_arrays["variables"],
                native_arrays["variables"],
                field=entry.field,
                spatial_location=spatial_location,
            )
            print(f"  trace {entry.field} spatial_location={spatial_location}")
            for time_index, (expected_value, actual_value, abs_diff) in enumerate(
                zip(trace.expected_series, trace.actual_series, trace.abs_diff_series, strict=True)
            ):
                print(
                    "    "
                    f"t={time_index}: expected={expected_value:.8e} actual={actual_value:.8e} abs_diff={abs_diff:.8e}"
                )
        print()


if __name__ == "__main__":
    main()
