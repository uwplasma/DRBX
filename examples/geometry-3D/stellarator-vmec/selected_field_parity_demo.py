from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.validation import create_stellarator_vmec_selected_field_parity_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-equilibrium-path", type=Path, default=None)
    parser.add_argument("--candidate-equilibrium-path", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("docs/data/stellarator_vmec_selected_field_artifacts"),
    )
    args = parser.parse_args()

    artifacts = create_stellarator_vmec_selected_field_parity_package(
        reference_equilibrium_path=args.reference_equilibrium_path,
        candidate_equilibrium_path=args.candidate_equilibrium_path,
        output_root=args.output_root,
    )
    print("== Stellarator VMEC Selected-Field Parity ==")
    print(f"  - reference_equilibrium_path: {args.reference_equilibrium_path if args.reference_equilibrium_path is not None else '<synthetic preview>'}")
    print(f"  - candidate_equilibrium_path: {args.candidate_equilibrium_path if args.candidate_equilibrium_path is not None else '<materialized candidate>'}")
    print("")
    print("== Artifacts ==")
    print(f"  - parity_json: {artifacts.parity_json_path}")
    print(f"  - parity_arrays_npz: {artifacts.parity_arrays_npz_path}")
    print(f"  - parity_plot_png: {artifacts.parity_plot_png_path}")
    print(f"  - observable_report_json: {artifacts.observable_report_json_path}")
    print(f"  - source_report_json: {artifacts.source_report_json_path}")


if __name__ == "__main__":
    main()
