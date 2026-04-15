from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.validation import create_traced_field_line_selected_field_parity_package


DEFAULT_EXTERNAL_FCI_GRID = Path("/tmp/zoidberg_better_metric/test/mms/poloidal_const_4_2_4_1.fci.nc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reduced selected-field parity package for traced-field-line geometry.")
    parser.add_argument("--reference-mesh-spec", type=Path, default=None)
    parser.add_argument("--candidate-mesh-spec", type=Path, default=None)
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
    if reference_mesh_spec is None and DEFAULT_EXTERNAL_FCI_GRID.exists():
        reference_mesh_spec = DEFAULT_EXTERNAL_FCI_GRID
    artifacts = create_traced_field_line_selected_field_parity_package(
        reference_mesh_spec=reference_mesh_spec,
        candidate_mesh_spec=candidate_mesh_spec,
        output_root=args.output_root,
    )
    print("== Traced-Field-Line Selected-Field Parity ==")
    print(f"  - reference_mesh_spec: {reference_mesh_spec if reference_mesh_spec is not None else '<synthetic preview>'}")
    print(
        f"  - candidate_mesh_spec: "
        f"{candidate_mesh_spec if candidate_mesh_spec is not None else ('<derived candidate>' if reference_mesh_spec is not None else '<synthetic preview>')}"
    )
    print("")
    print("== Artifacts ==")
    print(f"  - parity_json: {artifacts.parity_json_path}")
    print(f"  - parity_arrays_npz: {artifacts.parity_arrays_npz_path}")
    print(f"  - parity_plot_png: {artifacts.parity_plot_png_path}")
    print(f"  - observable_report_json: {artifacts.observable_report_json_path}")


if __name__ == "__main__":
    main()
