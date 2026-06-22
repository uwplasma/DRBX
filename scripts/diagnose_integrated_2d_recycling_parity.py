#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from jax_drb.native.runner import run_curated_case
from jax_drb.parity.arrays import build_array_payload_from_summary_payload, build_dataset_array_payload
from jax_drb.parity.compare import compare_summary_payloads
from jax_drb.parity.diff import build_scaled_array_diff_entries
from jax_drb.parity.reference import build_case_baseline_payload, resolve_reference_case, run_reference_case
from jax_drb.reference.paths import default_reference_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run staged integrated 2D recycling parity and classify remaining residuals by absolute and reference-relative scale."
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=default_reference_root(),
        help="Path to the local reference checkout.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Curated case to diagnose. Repeat to inspect multiple cases. Defaults to both integrated 2D recycling staged cases.",
    )
    parser.add_argument(
        "--near-zero-atol",
        type=float,
        default=1.0e-12,
        help="Expected max-abs threshold below which a field is treated as near-zero when reporting relative diffs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Maximum number of field entries to print per case.",
    )
    args = parser.parse_args()
    if args.reference_root is None:
        raise SystemExit("Set --reference-root or JAX_DRB_REFERENCE_ROOT before running live parity diagnostics.")

    case_names = tuple(args.case) if args.case else ("integrated_2d_recycling_rhs", "integrated_2d_recycling_one_step")
    for case_name in case_names:
        case, _ = resolve_reference_case(case_name, reference_root=args.reference_root)
        print(f"CASE {case_name}")
        with tempfile.TemporaryDirectory(prefix=f"jaxdrb-{case_name}-") as workdir:
            reference = run_reference_case(case_name, reference_root=args.reference_root, workdir=workdir, keep_workdir=True)
            reference_summary = build_case_baseline_payload(reference.summary)
            native = run_curated_case(case_name, reference_root=args.reference_root)

            summary = compare_summary_payloads(reference_summary, native.payload, scalar_rtol=1.0e-12, scalar_atol=1.0e-12)
            print(f"summary_ok={summary.ok} summary_issues={len(summary.issues)}")
            for issue in summary.issues[: args.limit]:
                print(f"  summary {issue.field}: {issue.message}")

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
            native_arrays = build_array_payload_from_summary_payload(native.payload, native.variables)
            scaled_entries = build_scaled_array_diff_entries(
                reference_arrays["variables"],
                native_arrays["variables"],
                compare_variables=tuple(reference.summary.compare_variables),
                near_zero_atol=args.near_zero_atol,
            )
            ranked = sorted(scaled_entries, key=lambda entry: entry.max_abs_diff, reverse=True)
            for entry in ranked[: args.limit]:
                relative = "n/a" if entry.relative_to_expected_max is None else f"{entry.relative_to_expected_max:.8e}"
                print(
                    "  field "
                    f"{entry.field}: max_abs_diff={entry.max_abs_diff:.8e} "
                    f"expected_abs_max={entry.expected_abs_max:.8e} "
                    f"relative_to_expected_max={relative} "
                    f"near_zero_expected={entry.near_zero_expected} "
                    f"location={entry.max_abs_location} "
                    f"expected={entry.expected_value:.8e} actual={entry.actual_value:.8e}"
                )
        print()


if __name__ == "__main__":
    main()
