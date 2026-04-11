from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.native import run_curated_case
from jax_drb.native.runner import _run_open_field_recycling_one_step_case
from jax_drb.parity.arrays import build_array_payload_from_summary_payload, load_portable_array_payload
from jax_drb.parity.diff import build_scaled_array_diff_entries
from jax_drb.parity.reference import resolve_reference_case


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the default open-field one-step transient path against a full-field-template replay probe.",
    )
    parser.add_argument(
        "--case",
        default="recycling_1d_one_step",
        choices=("recycling_1d_one_step", "recycling_dthe_one_step"),
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    return parser.parse_args()


def _diff_entries(case_name: str, result):
    expected = load_portable_array_payload(
        Path(__file__).resolve().parents[1] / "references" / "baselines" / "reference_arrays" / f"{case_name}.npz"
    )
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)
    return build_scaled_array_diff_entries(
        expected["variables"],
        actual["variables"],
        compare_variables=result.payload["compare_variables"],
    )


def _print_entries(label: str, entries, *, limit: int) -> None:
    print(label)
    for entry in entries[:limit]:
        print(
            f"  {entry.field}: max_abs_diff={entry.max_abs_diff:.8e} "
            f"expected_abs_max={entry.expected_abs_max:.8e} "
            f"relative_to_expected_max={entry.relative_to_expected_max!s} "
            f"location={entry.max_abs_location}"
        )


def main() -> int:
    args = _parse_args()
    case, input_path = resolve_reference_case(args.case, reference_root=args.reference_root)

    default_result = run_curated_case(args.case, reference_root=args.reference_root)
    template_result = _run_open_field_recycling_one_step_case(
        case,
        input_path=input_path,
        reference_root=args.reference_root,
    )

    default_entries = _diff_entries(args.case, default_result)
    template_entries = _diff_entries(args.case, template_result)
    _print_entries("default native path", default_entries, limit=args.limit)
    _print_entries("field-template replay probe", template_entries, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
