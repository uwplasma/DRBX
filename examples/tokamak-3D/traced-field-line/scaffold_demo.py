from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.validation import create_traced_field_line_scaffold_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a traced-field-line geometry scaffold bundle.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("docs/data/traced_field_line_scaffold_artifacts"),
        help="Directory to write the scaffold artifact bundle.",
    )
    parser.add_argument(
        "--mesh-spec",
        type=Path,
        default=None,
        help="Optional JSON mesh/metric specification. If omitted, uses a synthetic preview bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = create_traced_field_line_scaffold_package(
        output_root=args.output_root,
        mesh_spec_path=args.mesh_spec,
    )
    print("== Traced-Field-Line Scaffold ==")
    print(f"  - output_root: {args.output_root}")
    print(f"  - mesh_spec: {args.mesh_spec if args.mesh_spec is not None else '<synthetic preview>'}")
    print("")
    print("== Artifacts ==")
    print(f"  - manifest_json: {artifacts.manifest_json_path}")
    print(f"  - input_report_json: {artifacts.input_report_json_path}")
    print(f"  - validation_contract_json: {artifacts.validation_contract_json_path}")
    print(f"  - metric_report_json: {artifacts.metric_report_json_path}")
    print(f"  - metric_arrays_npz: {artifacts.metric_arrays_npz_path}")
    print(f"  - metric_plot_png: {artifacts.metric_plot_png_path}")


if __name__ == "__main__":
    main()
