from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.validation import create_native_tokamak_selected_field_package


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the first reduced native 3D selected-field parity package from a promoted "
            "tokamak short-window rung."
        )
    )
    parser.add_argument("--case-name", default="tokamak_turbulence_one_step")
    parser.add_argument("--case-label", default="tokamak_native_selected_field")
    parser.add_argument("--field-name", action="append", default=None)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_repo_root() / "docs" / "data" / "tokamak_native_selected_field_artifacts",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = create_native_tokamak_selected_field_package(
        case_name=args.case_name,
        case_label=args.case_label,
        reference_root=args.reference_root,
        output_root=args.output_root,
        field_names=tuple(args.field_name) if args.field_name else ("Ne", "Pe", "phi"),
    )
    if args.quiet:
        return
    print("\n== Native Tokamak Selected-Field Parity ==")
    print(f"  - case_name: {args.case_name}")
    print(f"  - case_label: {args.case_label}")
    print(f"  - reference_root: {args.reference_root}")
    print(f"  - field_names: {tuple(args.field_name) if args.field_name else ('Ne', 'Pe', 'phi')}")
    print(f"  - parity_json: {artifacts.parity_json_path}")
    print(f"  - parity_arrays: {artifacts.parity_arrays_npz_path}")
    print(f"  - parity_plot: {artifacts.parity_plot_png_path}")
    print(f"  - observable_report: {artifacts.observable_report_json_path}")
    print(f"  - runtime_report: {artifacts.runtime_report_json_path}")


if __name__ == "__main__":
    main()
