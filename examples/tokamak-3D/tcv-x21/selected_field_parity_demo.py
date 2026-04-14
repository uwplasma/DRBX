from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.validation import create_tcv_x21_selected_field_parity_package


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the reduced selected-field parity package for the TCV-X21 3D lane. "
            "If no workdirs are given, the demo uses the synthetic scaffold preview pair."
        )
    )
    parser.add_argument("--reference-workdir", type=Path, default=None)
    parser.add_argument("--candidate-workdir", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_repo_root() / "docs" / "data" / "tokamak_tcv_x21_selected_field_artifacts",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = create_tcv_x21_selected_field_parity_package(
        reference_workdir=args.reference_workdir,
        candidate_workdir=args.candidate_workdir,
        output_root=args.output_root,
    )
    if args.quiet:
        return
    print("\n== TCV-X21 Selected-Field Parity ==")
    print(f"  - reference_workdir: {args.reference_workdir if args.reference_workdir is not None else '<synthetic preview>'}")
    print(f"  - candidate_workdir: {args.candidate_workdir if args.candidate_workdir is not None else '<synthetic preview>'}")
    print(f"  - parity_json: {artifacts.parity_json_path}")
    print(f"  - parity_arrays: {artifacts.parity_arrays_npz_path}")
    print(f"  - parity_plot: {artifacts.parity_plot_png_path}")


if __name__ == "__main__":
    main()
