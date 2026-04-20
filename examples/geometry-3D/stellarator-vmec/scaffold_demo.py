from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.validation import create_stellarator_vmec_scaffold_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("docs/data/stellarator_vmec_scaffold_artifacts"),
        help="Directory where the stellarator VMEC scaffold artifacts should be written.",
    )
    parser.add_argument(
        "--equilibrium-path",
        type=Path,
        default=None,
        help="Optional VMEC wout NetCDF file. If omitted, a deterministic synthetic equilibrium is generated.",
    )
    args = parser.parse_args()

    artifacts = create_stellarator_vmec_scaffold_package(
        output_root=args.output_root,
        equilibrium_path=args.equilibrium_path,
    )
    print(f"manifest: {artifacts.manifest_json_path}")
    print(f"input report: {artifacts.input_report_json_path}")
    print(f"validation contract: {artifacts.validation_contract_json_path}")
    print(f"profile report: {artifacts.profile_report_json_path}")
    print(f"profile arrays: {artifacts.profile_arrays_npz_path}")
    print(f"profile plot: {artifacts.profile_plot_png_path}")
    print(f"surface report: {artifacts.surface_report_json_path}")
    print(f"surface arrays: {artifacts.surface_arrays_npz_path}")
    print(f"surface plot: {artifacts.surface_plot_png_path}")
    print(f"surface movie: {artifacts.surface_gif_path}")
    print(f"observable report: {artifacts.observable_report_json_path}")


if __name__ == "__main__":
    main()
