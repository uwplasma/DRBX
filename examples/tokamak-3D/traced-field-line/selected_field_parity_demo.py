from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.validation import create_traced_field_line_selected_field_parity_package


DEFAULT_EXTERNAL_FCI_GRID = Path("/tmp/zoidberg_better_metric/test/mms/poloidal_const_4_2_4_1.fci.nc")
DEFAULT_EXTERNAL_FCI_CANDIDATE_GRID = Path("/tmp/zoidberg_better_metric/test/mms/radial_const_4_2_4_1.fci.nc")
DEFAULT_FIELD_NAMES = ("J", "g11", "g33")
DEFAULT_EXTERNAL_PAIR_FIELD_NAMES = ("g11", "g33")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reduced selected-field parity package for traced-field-line geometry.")
    parser.add_argument("--reference-mesh-spec", type=Path, default=None)
    parser.add_argument("--candidate-mesh-spec", type=Path, default=None)
    parser.add_argument(
        "--field-name",
        dest="field_names",
        action="append",
        default=None,
        help="Selected metric field to compare. Repeat for multiple fields.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("docs/data/traced_field_line_selected_field_artifacts"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_mesh_spec = args.reference_mesh_spec
    candidate_mesh_spec = args.candidate_mesh_spec
    field_names = tuple(args.field_names) if args.field_names else None
    if reference_mesh_spec is None and DEFAULT_EXTERNAL_FCI_GRID.exists():
        reference_mesh_spec = DEFAULT_EXTERNAL_FCI_GRID
    if (
        candidate_mesh_spec is None
        and reference_mesh_spec == DEFAULT_EXTERNAL_FCI_GRID
        and DEFAULT_EXTERNAL_FCI_CANDIDATE_GRID.exists()
    ):
        candidate_mesh_spec = DEFAULT_EXTERNAL_FCI_CANDIDATE_GRID
    if field_names is None:
        if reference_mesh_spec is not None and candidate_mesh_spec is not None:
            field_names = DEFAULT_EXTERNAL_PAIR_FIELD_NAMES
        else:
            field_names = DEFAULT_FIELD_NAMES
    artifacts = create_traced_field_line_selected_field_parity_package(
        reference_mesh_spec=reference_mesh_spec,
        candidate_mesh_spec=candidate_mesh_spec,
        output_root=args.output_root,
        field_names=field_names,
    )
    print("== Traced-Field-Line Selected-Field Parity ==")
    print(f"  - reference_mesh_spec: {reference_mesh_spec if reference_mesh_spec is not None else '<synthetic preview>'}")
    print(
        f"  - candidate_mesh_spec: "
        f"{candidate_mesh_spec if candidate_mesh_spec is not None else ('<materialized explicit candidate>' if reference_mesh_spec is not None else '<synthetic preview>')}"
    )
    print(f"  - field_names: {', '.join(field_names)}")
    print("")
    print("== Artifacts ==")
    print(f"  - parity_json: {artifacts.parity_json_path}")
    print(f"  - parity_arrays_npz: {artifacts.parity_arrays_npz_path}")
    print(f"  - parity_plot_png: {artifacts.parity_plot_png_path}")
    print(f"  - observable_report_json: {artifacts.observable_report_json_path}")
    print(f"  - source_report_json: {artifacts.source_report_json_path}")


if __name__ == "__main__":
    main()
